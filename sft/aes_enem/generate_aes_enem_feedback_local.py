#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — Feedback de redação ENEM via LLM local (vLLM)
=======================================================================
Mesmo pipeline de `sft/aes_enem/generate_aes_enem_feedback.py` (redação
real do `kamel-usp/aes_enem_dataset` + notas reais por competência ->
feedback qualitativo GROUNDED nas notas, nunca inventando uma nota
diferente da real), mas falando com endpoint(s) OpenAI-compatible do
vLLM local em vez da API da Anthropic — para escalar sem custo de API,
usando a(s) GPU(s) do Netuno (ver `scripts/docker/run_vllm_serve.sh`).

Reusa de `sft/aes_enem/generate_aes_enem_feedback.py` (mesmo diretório):
HF_REPO_ID, LICENSE, REFERENCE, COMPETENCIAS, INSTRUCTION, SYSTEM_PROMPT,
detect_grade_columns, build_grades_text, count_emojis, s(). Reusa de
`synth/generate_instructions_local.py` (arquitetura de concorrência):
EndpointPool, call_with_retry, load_manifest/save_manifest_atomic.

DIFERENÇA DELIBERADA vs a versão via Claude: o `--max_examples` aqui
limita quantas LINHAS do dataset são TENTADAS (por índice), não quantas
são de fato GRAVADAS -- a versão sequencial via Claude para assim que
`n` (gravados) atinge o teto, o que não dá pra replicar de forma simples
com processamento concorrente (não dá pra saber de antemão quantas vão
ser puladas por falta de nota/redação). Na prática a diferença é pequena
(a maioria das linhas tem nota e redação válidas).

Auditoria (`--audit-sample N`, via Claude): critérios próprios
(fidelidade às notas reais/fluência/utilidade), não "grounded" genérico
-- aqui o "grounding" é literal (as 5 notas fornecidas), então a auditoria
checa se o feedback gerado é consistente com ELAS especificamente.

Uso:
    make llm-serve-0 && make llm-serve-1

    python sft/aes_enem/generate_aes_enem_feedback_local.py --dry-run
    python sft/aes_enem/generate_aes_enem_feedback_local.py \
        --out data/sft/aes_enem_local --max_examples 2000 --audit-sample 40

Saída: `<out>/aes_enem_feedback.jsonl` (Alpaca) + `manifest.json`
(+ `audit_report.json` se `--audit-sample > 0`).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from generate_aes_enem_feedback import (  # noqa: E402
    COMPETENCIAS,  # noqa: F401  (mantido p/ paridade/documentacao)
    HF_REPO_ID,
    INSTRUCTION,
    LICENSE,
    REFERENCE,
    SYSTEM_PROMPT,
    build_grades_text,
    count_emojis,
    detect_grade_columns,
    extract_grade_values,
    feedback_cites_real_grades,
    load_essays_split,
    s,
)

sys.path.insert(0, str(_THIS_DIR.parents[1] / "synth"))
from generate_instructions_local import (  # noqa: E402
    EndpointDown,
    EndpointPool,
    call_with_retry,
    load_manifest,
    save_manifest_atomic,
)

DEFAULT_MODEL = os.environ.get("LOCAL_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ")
DEFAULT_ENDPOINTS = "http://localhost:8000/v1,http://localhost:8001/v1"
DEFAULT_MAX_TOKENS = 2500
DEFAULT_MAX_EXAMPLES = 500
DEFAULT_CONCURRENCY_PER_ENDPOINT = 4
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_S = 120
SAVE_EVERY = 20

AUDIT_SYSTEM_PROMPT = """Você audita um feedback de redação ENEM gerado por outro modelo de \
linguagem (não o Claude). Receberá a REDAÇÃO, as NOTAS REAIS (dadas por um corretor humano, por \
competência C1-C5) e o FEEDBACK gerado a partir delas.

ESCALA REAL de cada competência (0 a 200, em passos de 40) -- USE ISSO pra julgar se o tom do \
feedback é coerente com a nota, não sua intuição:
- 200 = excelente domínio/desempenho
- 160 = bom domínio, com pequenas imprecisões
- 120 = domínio mediano -- NEM baixo nem alto, desempenho médio/regular com falhas reais mas \
também acertos reais (NÃO é "quase perfeito" nem "muito abaixo da média")
- 80 = domínio insuficiente
- 40 = domínio precário
- 0 = não desenvolveu o tema / fugiu do tema
Um feedback que aponta falhas reais (erros gramaticais, argumentação rasa, etc.) numa \
competência com nota 120 ou 160 está SENDO FIEL à nota, não contradizendo ela -- isso é \
esperado nesses níveis intermediários, não é erro.

Avalie três critérios, cada um true/false:
- "fiel_as_notas": o TOM do feedback é coerente com o NÍVEL da nota na escala acima (não descreve \
uma competência de nota 120/160 como se fosse perfeita, nem uma de nota 200 como se tivesse \
falha grave)? E o feedback não inventa um erro concreto (ortográfico/gramatical citando uma \
palavra ou trecho específico) que, ao conferir contra a REDAÇÃO fornecida, não existe nela?
- "fluente": o feedback está em português do Brasil correto e fluente, em texto corrido (não JSON)?
- "especifico": o feedback cita características/trechos concretos da redação (não é genérico a \
ponto de servir pra qualquer redação)?
Responda APENAS um objeto JSON, sem markdown: \
{"fiel_as_notas": bool, "fluente": bool, "especifico": bool, "nota": "..." (1 frase, so se algum \
for false)}"""


def parse_audit_json(raw_text: str) -> dict | None:
    import re
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def audit_sample_aes_enem(triples: list[tuple[str, str, str]], n: int, audit_model: str, seed: int) -> dict:
    """triples: lista de (essay, grades_text, feedback)."""
    import anthropic
    client = anthropic.Anthropic()
    rng = random.Random(seed)
    picked = rng.sample(triples, min(n, len(triples)))
    results = []
    for essay, grades_text, feedback in picked:
        user_msg = f"REDAÇÃO:\n{essay[:3000]}\n\nNOTAS REAIS:\n{grades_text}\n\nFEEDBACK:\n{feedback}"
        try:
            msg = client.messages.create(
                model=audit_model, max_tokens=300,
                system=AUDIT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = next((b.text for b in msg.content if b.type == "text"), "")
            verdict = parse_audit_json(text) or {}
        except Exception as e:
            verdict = {"error": str(e)}
        results.append({"grades": grades_text.replace("\n", " | ")[:200], "verdict": verdict})
    n_ok = sum(1 for r in results
               if r["verdict"].get("fiel_as_notas") and r["verdict"].get("fluente") and r["verdict"].get("especifico"))
    return {
        "audit_model": audit_model, "n_sampled": len(results),
        "n_pass_all_criteria": n_ok,
        "pass_rate": round(n_ok / len(results), 3) if results else None,
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Feedback de redacao ENEM via LLM local (vLLM)")
    ap.add_argument("--out", default="data/sft/aes_enem_local", help="diretorio de saida")
    ap.add_argument("--config", default=None, help="config do HF (default: descobre e usa a 1a disponivel)")
    ap.add_argument("--max_examples", type=int, default=DEFAULT_MAX_EXAMPLES,
                     help="quantas LINHAS do dataset tentar, por indice (ver docstring)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="nome do modelo servido pelo vLLM")
    ap.add_argument("--endpoints", default=DEFAULT_ENDPOINTS,
                     help="URLs base OpenAI-compatible separadas por virgula")
    ap.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY_PER_ENDPOINT,
                     help="requisicoes simultaneas POR endpoint")
    ap.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--audit-sample", type=int, default=0,
                     help="apos gerar, audita N exemplos aleatorios via Claude (custo pequeno)")
    ap.add_argument("--audit-model", default="claude-haiku-4-5")
    ap.add_argument("--overwrite", action="store_true",
                     help="permite sobrescrever um lote existente em --out cujo status/parametros "
                          "nao permitem retomada (sem isso, o comando para com erro em vez de "
                          "truncar silenciosamente um lote ja gerado)")
    args = ap.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "aes_enem_feedback.jsonl")

    try:
        from datasets import get_dataset_config_names
    except ImportError as e:
        print(f"[ERRO] 'datasets' ausente ({e}). Rode na imagem manaca-posttrain.")
        return 1

    if args.config:
        cfg = args.config
    else:
        try:
            configs = get_dataset_config_names(HF_REPO_ID)
        except Exception as e:
            print(f"[ERRO] falha ao descobrir configs de {HF_REPO_ID}: {e}")
            return 1
        if not configs:
            print(f"[ERRO] nenhuma config encontrada em {HF_REPO_ID}")
            return 1
        cfg = configs[0]
        print(f"[aes-enem-local] configs disponiveis: {configs} | usando: {cfg}")

    prev = load_manifest(out_dir)
    can_resume = (
        prev.get("status") in ("in_progress", "stopped_fatal_error")
        and prev.get("config") == cfg
        and prev.get("model") == args.model
        and prev.get("max_examples") == args.max_examples
    )
    if (not args.dry_run and not can_resume and not args.overwrite
            and os.path.exists(out_path) and os.path.getsize(out_path) > 0):
        print(f"[ERRO] {out_path} ja existe (lote anterior: status={prev.get('status')!r}, "
              f"total={prev.get('total')}, config={prev.get('config')}) e os parametros deste "
              "comando nao permitem retomada -- rodar assim SOBRESCREVERIA esse lote. Use "
              "--out <outro-diretorio> para um lote novo separado (preservando o anterior), ou "
              "--overwrite se a intencao e mesmo descartar o que esta em " + out_path + ".")
        return 1

    try:
        ds = load_essays_split(cfg)
    except Exception as e:
        print(f"[ERRO] falha ao carregar {HF_REPO_ID} ({cfg}): {e}")
        return 1

    cols = ds.column_names
    essay_col = next((c for c in ["essay", "text", "redacao", "essay_text", "corpus"] if c in cols), None)
    if not essay_col:
        print(f"[ERRO] coluna de redacao nao reconhecida em {cols}")
        return 1
    first_row = ds[0]
    single_col, comp_cols = detect_grade_columns(first_row)
    print(f"[aes-enem-local] config={cfg} | coluna redacao='{essay_col}' | "
          f"nota unica='{single_col}' | competencias='{comp_cols}' | colunas: {cols}")
    if not single_col and not comp_cols:
        print(f"[ERRO] nenhuma coluna de nota reconhecida em {cols}. "
              "Ajuste detect_grade_columns() em generate_aes_enem_feedback.py.")
        return 1

    if args.dry_run:
        grades_text = build_grades_text(first_row, single_col, comp_cols)
        print("\n[dry-run] nenhuma chamada sera feita.\n")
        print("=" * 60)
        print("SYSTEM PROMPT:\n" + SYSTEM_PROMPT)
        print("=" * 60)
        print(f"1o EXEMPLO — redacao (300 char): {s(first_row.get(essay_col))[:300]}")
        print(f"notas detectadas: {grades_text}")
        return 0

    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    pool = EndpointPool(endpoints)

    n_total = min(args.max_examples, len(ds)) if args.max_examples else len(ds)
    processed_ids: set[str] = set(prev.get("processed_ids", [])) if can_resume else set()
    n = prev.get("total", 0) if can_resume else 0
    n_no_grades = prev.get("skipped_no_grades", 0) if can_resume else 0
    n_emoji = prev.get("skipped_emoji", 0) if can_resume else 0
    n_grade_mismatch = prev.get("skipped_grade_mismatch", 0) if can_resume else 0
    n_failed = prev.get("failed_requests", 0) if can_resume else 0
    if can_resume:
        print(f"[resume] retomando de manifest anterior: {n} ja gravados, "
              f"{len(processed_ids)} exemplos ja processados")

    todo = [i for i in range(n_total) if str(i) not in processed_ids]
    print(f"[aes-enem-local] {len(todo)} a processar de {n_total} tentados "
          f"({n_total - len(todo)} ja feitos)")

    file_mode = "a" if can_resume and os.path.exists(out_path) else "w"
    write_lock = threading.Lock()
    manifest_lock = threading.Lock()
    fatal_error: list[Exception] = []
    audit_pool: list[tuple[str, str, str]] = []

    def process_one(i: int) -> None:
        nonlocal n, n_no_grades, n_emoji, n_grade_mismatch, n_failed
        if fatal_error:
            return
        ex = ds[i]
        essay = s(ex.get(essay_col))
        grades_text = build_grades_text(ex, single_col, comp_cols)
        grade_values = extract_grade_values(ex, single_col, comp_cols)
        if not essay or not grades_text:
            with manifest_lock:
                processed_ids.add(str(i))
                n_no_grades += 1
            return
        user_msg = f"Redação:\n{essay}\n\nNotas já atribuídas pelo corretor:\n{grades_text}"
        try:
            result = call_with_retry(pool, args.model, SYSTEM_PROMPT, user_msg,
                                      args.max_tokens, args.timeout, args.max_retries)
        except EndpointDown as e:
            fatal_error.append(e)
            return
        with manifest_lock:
            processed_ids.add(str(i))
            if result is None:
                n_failed += 1
                return
            feedback, _usage = result
            feedback = feedback.strip()
            if not feedback:
                n_failed += 1
                return
            if count_emojis(feedback):
                n_emoji += 1
                return
            if grade_values and not feedback_cites_real_grades(feedback, grade_values):
                n_grade_mismatch += 1
                return
            row_out = {"instruction": INSTRUCTION, "input": essay, "output": feedback,
                       "task": "aes_enem_feedback"}
            with write_lock:
                with open(out_path, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            audit_pool.append((essay, grades_text, feedback))
            n += 1

    def snapshot_manifest(status: str) -> dict:
        return {
            "dataset": "aes_enem_feedback_local", "hf_repo": HF_REPO_ID, "config": cfg,
            "license": LICENSE, "reference": REFERENCE, "model": args.model, "endpoints": endpoints,
            "essay_column": essay_col, "single_grade_column": single_col, "competency_columns": comp_cols,
            "max_examples": args.max_examples, "total": n,
            "skipped_no_grades": n_no_grades, "skipped_emoji": n_emoji,
            "skipped_grade_mismatch": n_grade_mismatch, "failed_requests": n_failed,
            "processed_ids": sorted(processed_ids),
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    if file_mode == "w":
        open(out_path, "w", encoding="utf-8").close()

    max_workers = args.concurrency * len(endpoints)
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {executor.submit(process_one, i): i for i in todo}
        try:
            for fut in as_completed(futures):
                fut.result()
                completed += 1
                if completed % SAVE_EVERY == 0:
                    save_manifest_atomic(out_dir, snapshot_manifest("in_progress"))
                    print(f"[aes-enem-local] {completed}/{len(todo)} processadas, {n} gravadas")
                if fatal_error:
                    break
        except KeyboardInterrupt:
            print("\n[INFO] Interrompido -- estado salvo, rode o mesmo comando de novo para retomar.")
            save_manifest_atomic(out_dir, snapshot_manifest("stopped_fatal_error"))
            executor.shutdown(wait=False, cancel_futures=True)
            return 130

    final_status = "stopped_fatal_error" if fatal_error else "completed"
    manifest = snapshot_manifest(final_status)

    if fatal_error:
        print(f"[ERRO FATAL] {fatal_error[0]} -- rode o mesmo comando de novo para retomar "
              f"depois de checar os servidores vLLM (docker logs manaca-vllm-0 / -1).")

    if args.audit_sample > 0 and audit_pool:
        print(f"[audit] auditando {min(args.audit_sample, len(audit_pool))} exemplos via {args.audit_model}...")
        report = audit_sample_aes_enem(audit_pool, args.audit_sample, args.audit_model, args.seed)
        with open(os.path.join(out_dir, "audit_report.json"), "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
        manifest["audit_pass_rate"] = report["pass_rate"]
        print(f"[audit] taxa de aprovacao: {report['pass_rate']} "
              f"({report['n_pass_all_criteria']}/{report['n_sampled']}) -> "
              f"{os.path.join(out_dir, 'audit_report.json')}")
        if report["pass_rate"] is not None and report["pass_rate"] < 0.8:
            print("[AVISO] taxa de aprovacao da auditoria abaixo de 80% -- "
                  "revise audit_report.json antes de usar este lote em treino.")

    save_manifest_atomic(out_dir, manifest)
    print(f"\n[aes-enem-local] TOTAL: {n:,} exemplos -> {out_path}")
    print(f"[aes-enem-local] descartados: {n_no_grades} (sem redacao/nota) | "
          f"{n_emoji} (emoji) | {n_grade_mismatch} (nao citou as notas reais) | "
          f"{n_failed} (falha de requisicao)")
    return 1 if fatal_error else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
