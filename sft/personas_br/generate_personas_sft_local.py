#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — SFT de personas brasileiras via LLM local (vLLM)
=========================================================================
Mesmo pipeline de `sft/personas_br/generate_personas_sft.py` (persona do
`nvidia/Nemotron-Personas-Brazil` -> `{instruction, input, output, task}`,
mesmo SYSTEM_PROMPT/regras/decontaminação/emoji-zero), mas falando com
endpoint(s) OpenAI-compatible do vLLM local em vez da API da Anthropic —
para escalar volume sem custo de API, usando a(s) GPU(s) do Netuno (ver
`scripts/docker/run_vllm_serve.sh`).

Reusa de `sft/personas_br/generate_personas_sft.py` (mesmo diretório):
HF_REPO_ID, LICENSE, REFERENCE, SYSTEM_PROMPT, DECONTAMINATION_SOURCES,
reservoir_sample, persona_to_text, parse_json_obj,
load_decontamination_set, is_contaminated, count_emojis,
MAX_EMOJIS_PER_OUTPUT, s(). Reusa de `synth/generate_instructions_local.py`
(arquitetura de concorrência): EndpointPool (round-robin + circuit-breaker),
call_local_llm/call_with_retry (urllib, stdlib), load_manifest/
save_manifest_atomic.

Igual ao pipeline via Claude: resume via manifest.json (aqui com
`processed_ids`, não um `processed_idx` único — threads terminam fora de
ordem) e trava de `--overwrite` (nunca sobrescreve silenciosamente um lote
`completed` ou de parâmetros diferentes em `--out`).

Auditoria (`--audit-sample N`): diferente de `generate_instructions_local.py`
(que audita "a resposta usa só o que está no Contexto"), aqui não há um
texto-fonte pra checar fidelidade — o `input` do exemplo final é
propositalmente vazio (a persona é só andaime pra gerar um pedido diverso e
plausível, não deve aparecer no exemplo de treino, senão o modelo aprenderia
a esperar uma descrição de persona em produção). Por isso a auditoria aqui
usa critérios próprios (plausível/fluente/útil) e leva a persona-fonte
guardada em memória durante a geração, não o `input` do JSONL.

Uso:
    make llm-serve-0 && make llm-serve-1

    python sft/personas_br/generate_personas_sft_local.py --dry-run
    python sft/personas_br/generate_personas_sft_local.py \
        --out data/sft/personas_br_local --num_personas 5000 --seed 100 \
        --audit-sample 60

Saída: `<out>/personas_br.jsonl` (schema Alpaca) + `manifest.json`
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
from generate_personas_sft import (  # noqa: E402
    DECONTAMINATION_SOURCES,  # noqa: F401  (mantido p/ paridade/documentacao)
    HF_REPO_ID,
    LICENSE,
    REFERENCE,
    SYSTEM_PROMPT,
    MAX_EMOJIS_PER_OUTPUT,
    count_emojis,
    is_contaminated,
    load_decontamination_set,
    parse_json_obj,
    persona_to_text,
    reservoir_sample,
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
DEFAULT_MAX_TOKENS = 2000
DEFAULT_NUM_PERSONAS = 2000
DEFAULT_CONCURRENCY_PER_ENDPOINT = 4
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_S = 120
SAVE_EVERY = 20

AUDIT_SYSTEM_PROMPT = """Você audita um exemplo de treino gerado por outro modelo de linguagem \
(não o Claude) a partir da descrição de uma PERSONA brasileira fictícia (perfil demográfico \
realista, mas pessoa não real). Receberá a PERSONA, o PEDIDO gerado (que essa persona \
supostamente faria a um assistente de IA) e a RESPOSTA gerada. Avalie três critérios, cada um \
true/false:
- "plausivel": o PEDIDO é coerente com o perfil da persona (idade, ocupação, região, \
escolaridade, objetivos) e soa como algo que uma pessoa real digitaria (não genérico, não \
fora de contexto)?
- "fluente": a RESPOSTA está em português do Brasil correto e fluente?
- "util": a RESPOSTA de fato atende ao PEDIDO, sem alucinar informação específica e \
verificável que não faz sentido (leis/números/nomes inventados)?
Responda APENAS um objeto JSON, sem markdown: \
{"plausivel": bool, "fluente": bool, "util": bool, "nota": "..." (1 frase, so se algum for false)}"""


def audit_sample_personas(triples: list[tuple[str, str, str]], n: int, audit_model: str, seed: int) -> dict:
    """triples: lista de (persona_text, instruction, output)."""
    import anthropic
    client = anthropic.Anthropic()
    rng = random.Random(seed)
    picked = rng.sample(triples, min(n, len(triples)))
    results = []
    for persona_text, instr, out in picked:
        user_msg = f"PERSONA:\n{persona_text[:1500]}\n\nPEDIDO:\n{instr}\n\nRESPOSTA:\n{out}"
        try:
            msg = client.messages.create(
                model=audit_model, max_tokens=300,
                system=AUDIT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = next((b.text for b in msg.content if b.type == "text"), "")
            verdict = parse_json_obj(text) or {}
        except Exception as e:
            verdict = {"error": str(e)}
        results.append({"instruction": instr[:200], "verdict": verdict})
    n_ok = sum(1 for r in results
               if r["verdict"].get("plausivel") and r["verdict"].get("fluente") and r["verdict"].get("util"))
    return {
        "audit_model": audit_model, "n_sampled": len(results),
        "n_pass_all_criteria": n_ok,
        "pass_rate": round(n_ok / len(results), 3) if results else None,
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="SFT de personas brasileiras via LLM local (vLLM)")
    ap.add_argument("--out", default="data/sft/personas_br_local", help="diretorio de saida")
    ap.add_argument("--num_personas", type=int, default=DEFAULT_NUM_PERSONAS)
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
    ap.add_argument("--repo_root", default=None)
    ap.add_argument("--no_decontaminate", action="store_true")
    ap.add_argument("--audit-sample", type=int, default=0,
                     help="apos gerar, audita N exemplos aleatorios via Claude (custo pequeno)")
    ap.add_argument("--audit-model", default="claude-haiku-4-5")
    ap.add_argument("--overwrite", action="store_true",
                     help="permite sobrescrever um lote existente em --out cujo status/parametros "
                          "nao permitem retomada (sem isso, o comando para com erro em vez de "
                          "truncar silenciosamente um lote ja gerado)")
    args = ap.parse_args()

    repo_root = args.repo_root or str(_THIS_DIR.parents[1])
    rng = random.Random(args.seed)

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "personas_br.jsonl")

    prev = load_manifest(out_dir)
    can_resume = (
        prev.get("status") in ("in_progress", "stopped_fatal_error")
        and prev.get("num_personas") == args.num_personas
        and prev.get("seed") == args.seed
        and prev.get("model") == args.model
    )
    if (not args.dry_run and not can_resume and not args.overwrite
            and os.path.exists(out_path) and os.path.getsize(out_path) > 0):
        print(f"[ERRO] {out_path} ja existe (lote anterior: status={prev.get('status')!r}, "
              f"total={prev.get('total')}, seed={prev.get('seed')}) e os parametros deste comando "
              "nao permitem retomada -- rodar assim SOBRESCREVERIA esse lote. Use "
              "--out <outro-diretorio> para um lote novo separado (preservando o anterior), ou "
              "--overwrite se a intencao e mesmo descartar o que esta em " + out_path + ".")
        return 1

    try:
        from datasets import load_dataset
    except ImportError as e:
        print(f"[ERRO] 'datasets' ausente ({e}). Rode na imagem manaca-posttrain.")
        return 1

    print(f"[personas-br-local] streaming {HF_REPO_ID} para reservoir-sample de {args.num_personas:,} personas...")
    try:
        ds = load_dataset(HF_REPO_ID, split="train", streaming=True)
        personas = reservoir_sample(ds, args.num_personas, rng)
    except Exception as e:
        print(f"[ERRO] falha ao carregar/amostrar {HF_REPO_ID}: {e}")
        return 1
    print(f"[personas-br-local] {len(personas):,} personas amostradas")

    if args.dry_run:
        print("\n[dry-run] nenhuma chamada sera feita.\n")
        print("=" * 60)
        print("SYSTEM PROMPT:\n" + SYSTEM_PROMPT)
        if personas:
            print("=" * 60)
            print("EXEMPLO DE PERSONA (1a amostrada):\n" + persona_to_text(personas[0])[:1500])
        return 0

    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    pool = EndpointPool(endpoints)

    decontam: set[str] = set()
    if not args.no_decontaminate:
        decontam = load_decontamination_set(repo_root)
        print(f"[decontam] {len(decontam)} prompts normalizados")

    processed_ids: set[str] = set(prev.get("processed_ids", [])) if can_resume else set()
    n = prev.get("total", 0) if can_resume else 0
    n_skip_model = prev.get("skipped_model_decidiu", 0) if can_resume else 0
    n_skip_unparseable = prev.get("skipped_json_invalido", 0) if can_resume else 0
    n_skip_contam = prev.get("skipped_contaminado", 0) if can_resume else 0
    n_skip_emoji = prev.get("skipped_emoji_pesado", 0) if can_resume else 0
    n_failed = prev.get("failed_requests", 0) if can_resume else 0
    if can_resume:
        print(f"[resume] retomando de manifest anterior: {n} ja gravados, "
              f"{len(processed_ids)} personas ja processadas")

    todo = [(i, row) for i, row in enumerate(personas) if str(i) not in processed_ids]
    print(f"[personas-br-local] {len(todo)} a processar ({len(personas) - len(todo)} ja feitas)")

    file_mode = "a" if can_resume and os.path.exists(out_path) else "w"
    write_lock = threading.Lock()
    manifest_lock = threading.Lock()
    fatal_error: list[Exception] = []
    audit_pool: list[tuple[str, str, str]] = []

    def process_one(item: tuple[int, dict]) -> None:
        nonlocal n, n_skip_model, n_skip_unparseable, n_skip_contam, n_skip_emoji, n_failed
        i, row = item
        if fatal_error:
            return
        persona_text = persona_to_text(row)
        if not persona_text:
            with manifest_lock:
                processed_ids.add(str(i))
            return
        try:
            result = call_with_retry(pool, args.model, SYSTEM_PROMPT, f"Persona:\n{persona_text[:3000]}",
                                      args.max_tokens, args.timeout, args.max_retries)
        except EndpointDown as e:
            fatal_error.append(e)
            return
        with manifest_lock:
            processed_ids.add(str(i))
            if result is None:
                n_failed += 1
                return
            content, _usage = result
            obj = parse_json_obj(content)
            if not obj or obj.get("skip"):
                n_skip_model += 1
                return
            instr, out = s(obj.get("instruction")), s(obj.get("output"))
            if not instr or not out:
                n_skip_unparseable += 1
                return
            if is_contaminated(instr, decontam):
                n_skip_contam += 1
                return
            if count_emojis(out) > MAX_EMOJIS_PER_OUTPUT:
                n_skip_emoji += 1
                return
            row_out = {"instruction": instr, "input": "", "output": out, "task": "personas_br"}
            with write_lock:
                with open(out_path, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            audit_pool.append((persona_text, instr, out))
            n += 1

    def snapshot_manifest(status: str) -> dict:
        return {
            "dataset": "personas_br_local", "hf_repo": HF_REPO_ID, "license": LICENSE, "reference": REFERENCE,
            "model": args.model, "endpoints": endpoints, "seed": args.seed, "num_personas": args.num_personas,
            "personas_sampled": len(personas), "total": n,
            "skipped_model_decidiu": n_skip_model, "skipped_json_invalido": n_skip_unparseable,
            "skipped_contaminado": n_skip_contam, "skipped_emoji_pesado": n_skip_emoji,
            "failed_requests": n_failed,
            "processed_ids": sorted(processed_ids),
            "decontaminated": not args.no_decontaminate,
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    if file_mode == "w":
        open(out_path, "w", encoding="utf-8").close()

    max_workers = args.concurrency * len(endpoints)
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {executor.submit(process_one, item): item for item in todo}
        try:
            for fut in as_completed(futures):
                fut.result()
                completed += 1
                if completed % SAVE_EVERY == 0:
                    save_manifest_atomic(out_dir, snapshot_manifest("in_progress"))
                    print(f"[personas-br-local] {completed}/{len(todo)} processadas, {n} gravadas")
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
        report = audit_sample_personas(audit_pool, args.audit_sample, args.audit_model, args.seed)
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
    print(f"[personas-br-local] TOTAL: {n:,} exemplos -> {out_path}")
    print(f"[personas-br-local] descartados: {n_skip_model} (modelo pulou) | "
          f"{n_skip_unparseable} (json invalido) | {n_skip_contam} (contaminados) | "
          f"{n_skip_emoji} (emoji) | {n_failed} (falha de requisicao)")
    return 1 if fatal_error else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
