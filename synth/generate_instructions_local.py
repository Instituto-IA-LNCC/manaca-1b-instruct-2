#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — Geração de instrução sintética via LLM local (vLLM)
===========================================================================
Mesmo pipeline de `synth/generate_instructions.py` (passagem de corpus bruto
-> UM exemplo `{instruction, input, output, task}` grounded, mesmo
SYSTEM_PROMPT/regras/decontaminação/esquema de saída), mas falando com UM OU
MAIS servidores vLLM locais (OpenAI-compatible) em vez da API da Anthropic —
pra escalar volume sem custo de API, usando a(s) GPU(s) do Netuno (ver
`scripts/docker/run_vllm_serve.sh` e `scripts/docker/README-vllm.md`).

Reusa de `synth/generate_instructions.py` (mesmo diretório, sem duplicar):
SOURCES, WORK_DIR, TASK_TYPES, SYSTEM_PROMPT, DECONTAMINATION_SOURCES,
sample_passages, build_user_message, parse_model_json,
load_decontamination_set, is_contaminated, s(). Também reusa a política de
emoji ZERO já estabelecida em sft/personas_br/generate_personas_sft.py e
sft/aes_enem/generate_aes_enem_feedback.py: o tokenizer do manaca-1b-base
não tem token dedicado pra emoji (SentencePiece treinado do zero só no
corpus PT-BR do projeto), então qualquer emoji vira ~4 tokens de
byte-fallback fora de distribuição — descarta o exemplo inteiro se algum
aparecer, em vez de só avisar no prompt (mesmo raciocínio, mesmo risco).

IMPORTANTE — por que isso é COMPLEMENTAR, não substituto do pipeline via
Claude: um LLM aberto menor tende a alucinar mais e ser menos disciplinado
em "não invente fatos além do texto" do que o Claude. Use `--audit-sample N`
pra rodar uma auditoria barata via Claude (Haiku por padrão) sobre uma
amostra aleatória do lote gerado antes de confiar no resultado inteiro —
NÃO substitui uma revisão humana de verdade, só dá um sinal de alerta cedo.

RESUMO/RETOMADA: mesmo padrão de manifest.json incremental já usado em
sft/personas_br/ e sft/aes_enem/ — se o processo cair (SSH, container),
rode o mesmo comando de novo que ele retoma sem reprocessar o que já foi
gravado (a amostragem de passagens é determinística dado --seed/--source/
--max-examples/--oversample, então o resume só precisa filtrar os ids já
processados, sem re-sortear nada).

Uso:
    make llm-serve-0 && make llm-serve-1   # sobe os 2 endpoints locais
                                            # (ver scripts/docker/README-vllm.md)

    # testar sem gastar tempo de GPU (so mostra os prompts)
    python synth/generate_instructions_local.py --source classiccpt \
        --max-examples 5 --dry-run

    python synth/generate_instructions_local.py --source classiccpt \
        --max-examples 2000 \
        --endpoints http://localhost:8000/v1,http://localhost:8001/v1

    # com auditoria via Claude sobre uma amostra (custo pequeno, so na amostra)
    ANTHROPIC_API_KEY=sk-ant-... python synth/generate_instructions_local.py \
        --source classiccpt --max-examples 2000 --audit-sample 30

Saída: `data/synth_local/<source>/<source>_synth.jsonl` + `manifest.json`
(+ `audit_report.json` se --audit-sample > 0) — mesmo esquema Alpaca dos
demais builders, consumido via `--extra` de `sft/prepare_data.py`.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_instructions import (  # noqa: E402
    DECONTAMINATION_SOURCES,  # noqa: F401  (mantido p/ paridade/documentacao)
    SOURCES,
    TASK_TYPES,
    SYSTEM_PROMPT,
    WORK_DIR,
    build_user_message,
    is_contaminated,
    load_decontamination_set,
    parse_model_json,
    s,
    sample_passages,
)

DEFAULT_MODEL = os.environ.get("LOCAL_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ")
DEFAULT_ENDPOINTS = "http://localhost:8000/v1,http://localhost:8001/v1"
DEFAULT_MAX_TOKENS = 1500
DEFAULT_MAX_EXAMPLES = 2000
DEFAULT_OVERSAMPLE = 5
DEFAULT_CONCURRENCY_PER_ENDPOINT = 4
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_S = 120
ENDPOINT_FAILURE_LIMIT = 8  # falhas consecutivas antes de tirar o endpoint de rotacao
SAVE_EVERY = 20

# Mesma politica de tolerancia ZERO de emoji ja usada em personas_br/aes_enem
# (ver comentario completo la): tokenizer do manaca-1b-base nao tem token
# dedicado pra emoji, entao qualquer emoji vira sequencia de bytes fora de
# distribuicao.
_EMOJI_RE = __import__("re").compile(
    "[\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U00002B00-\U00002BFF"
    "\U0001F1E6-\U0001F1FF"
    "]"
)
MAX_EMOJIS_PER_OUTPUT = 0


def count_emojis(text: str) -> int:
    return len(_EMOJI_RE.findall(text))


# ── Client HTTP minimo (sem dependencia extra: urllib da stdlib) ──────────────

class EndpointDown(Exception):
    pass


class EndpointPool:
    """Round-robin entre endpoints vLLM, com circuit-breaker simples: tira um
    endpoint de rotacao apos ENDPOINT_FAILURE_LIMIT falhas consecutivas.
    Levanta EndpointDown se todos caírem (não adianta continuar tentando)."""

    def __init__(self, endpoints: list[str]):
        self._all = list(endpoints)
        self._lock = threading.Lock()
        self._fail_count = {ep: 0 for ep in endpoints}
        self._alive = set(endpoints)
        self._cycle = itertools.cycle(endpoints)

    def next(self) -> str:
        with self._lock:
            for _ in range(len(self._all)):
                ep = next(self._cycle)
                if ep in self._alive:
                    return ep
            raise EndpointDown(f"todos os {len(self._all)} endpoints fora do ar: {self._all}")

    def report_success(self, ep: str) -> None:
        with self._lock:
            self._fail_count[ep] = 0

    def report_failure(self, ep: str) -> None:
        with self._lock:
            self._fail_count[ep] += 1
            if self._fail_count[ep] >= ENDPOINT_FAILURE_LIMIT and ep in self._alive:
                self._alive.discard(ep)
                print(f"[AVISO] endpoint {ep} removido da rotacao apos "
                      f"{ENDPOINT_FAILURE_LIMIT} falhas consecutivas")
            if not self._alive:
                raise EndpointDown(f"todos os {len(self._all)} endpoints fora do ar: {self._all}")


def call_local_llm(endpoint: str, model: str, system_prompt: str, user_message: str,
                    max_tokens: int, timeout_s: int) -> tuple[str, dict]:
    """POST /chat/completions no endpoint vLLM (OpenAI-compatible). Levanta
    em qualquer falha (rede/HTTP/JSON) -- quem chama decide retry/skip."""
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {}) or {}
    return content, usage


def call_with_retry(pool: EndpointPool, model: str, system_prompt: str, user_message: str,
                     max_tokens: int, timeout_s: int, max_retries: int) -> tuple[str, dict] | None:
    last_err = None
    for attempt in range(max_retries):
        ep = pool.next()
        try:
            content, usage = call_local_llm(ep, model, system_prompt, user_message, max_tokens, timeout_s)
            pool.report_success(ep)
            return content, usage
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                ConnectionError, json.JSONDecodeError, KeyError) as e:
            last_err = e
            pool.report_failure(ep)
            time.sleep(min(2 ** attempt, 10))
    print(f"[falha] {max_retries} tentativas esgotadas, ultimo erro: {last_err}")
    return None


# ── Manifest (mesmo padrao de sft/personas_br/generate_personas_sft.py) ──────

def load_manifest(out_dir: str) -> dict:
    path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fp:
                return json.load(fp)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_manifest_atomic(out_dir: str, manifest: dict) -> None:
    path = os.path.join(out_dir, "manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ── Auditoria via Claude sobre uma amostra (opcional, custo pequeno) ─────────

AUDIT_SYSTEM_PROMPT = """Você audita um exemplo de treino gerado por outro modelo de linguagem \
(não o Claude). Receberá um CONTEXTO (a passagem-fonte), uma INSTRUÇÃO e uma RESPOSTA. \
Avalie três critérios, cada um true/false:
- "grounded": a resposta usa SOMENTE informação presente no contexto (sem inventar fatos, \
números, nomes ou datas)?
- "fluente": a resposta está em português do Brasil correto e fluente (sem erros grosseiros \
de gramática/concordância, sem mistura de idioma)?
- "coerente": a instrução e a resposta formam um par sensato (a resposta realmente responde \
à instrução, não é vaga nem fora do tópico)?
Responda APENAS um objeto JSON, sem markdown: \
{"grounded": bool, "fluente": bool, "coerente": bool, "nota": "..." (1 frase, so se algum for false)}"""


def audit_sample(rows: list[dict], n: int, audit_model: str, seed: int) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    rng = random.Random(seed)
    picked = rng.sample(rows, min(n, len(rows)))
    results = []
    for row in picked:
        user_msg = f"CONTEXTO:\n{row['input']}\n\nINSTRUÇÃO:\n{row['instruction']}\n\nRESPOSTA:\n{row['output']}"
        try:
            msg = client.messages.create(
                model=audit_model, max_tokens=300,
                system=AUDIT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = next((b.text for b in msg.content if b.type == "text"), "")
            verdict = parse_model_json(text) or {}
        except Exception as e:
            verdict = {"error": str(e)}
        results.append({
            "instruction": row["instruction"][:200],
            "verdict": verdict,
        })
    n_ok = sum(1 for r in results
               if r["verdict"].get("grounded") and r["verdict"].get("fluente") and r["verdict"].get("coerente"))
    return {
        "audit_model": audit_model, "n_sampled": len(results),
        "n_pass_all_criteria": n_ok,
        "pass_rate": round(n_ok / len(results), 3) if results else None,
        "results": results,
    }


# ── Comando principal ──────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Geracao de instrucao sintetica via LLM local (vLLM)")
    ap.add_argument("--source", choices=sorted(SOURCES), required=True)
    ap.add_argument("--out", default=None, help="diretorio de saida (default: data/synth_local/<source>)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="nome do modelo servido pelo vLLM")
    ap.add_argument("--endpoints", default=DEFAULT_ENDPOINTS,
                     help="URLs base OpenAI-compatible separadas por virgula")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    ap.add_argument("--oversample", type=int, default=DEFAULT_OVERSAMPLE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY_PER_ENDPOINT,
                     help="requisicoes simultaneas POR endpoint")
    ap.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--no_decontaminate", action="store_true")
    ap.add_argument("--audit-sample", type=int, default=0,
                     help="apos gerar, audita N exemplos aleatorios via Claude (custo pequeno)")
    ap.add_argument("--audit-model", default="claude-haiku-4-5")
    ap.add_argument("--overwrite", action="store_true",
                     help="permite sobrescrever um lote existente em --out cujo status/parametros "
                          "nao permitem retomada (sem isso, o comando para com erro em vez de "
                          "truncar silenciosamente um lote ja completo ou de parametros diferentes)")
    args = ap.parse_args()

    out_dir = args.out or f"data/synth_local/{args.source}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.source}_synth.jsonl")

    # Checagem de seguranca ANTES de amostrar/gerar qualquer coisa: um lote
    # anterior em --out so pode ser retomado (append) se status e parametros
    # baterem EXATAMENTE (ver can_resume abaixo); qualquer outro caso (lote
    # 'completed', ou parametros diferentes tipo --seed novo) cai no modo
    # "w" (sobrescreve) mais abaixo -- o que ja aconteceu ao vivo aqui e
    # apagou silenciosamente um lote de 891 exemplos. Falha cedo em vez de
    # truncar sem avisar; --overwrite e a saida explicita quando a intencao
    # e mesmo descartar o que esta em --out.
    _prev_check = load_manifest(out_dir)
    _can_resume_check = (
        _prev_check.get("status") in ("in_progress", "stopped_fatal_error")
        and _prev_check.get("source") == args.source
        and _prev_check.get("model") == args.model
        and _prev_check.get("max_examples") == args.max_examples
        and _prev_check.get("oversample") == args.oversample
        and _prev_check.get("seed") == args.seed
    )
    if (not args.dry_run and not _can_resume_check and not args.overwrite
            and os.path.exists(out_path) and os.path.getsize(out_path) > 0):
        print(f"[ERRO] {out_path} ja existe (lote anterior: status={_prev_check.get('status')!r}, "
              f"total={_prev_check.get('total')}, seed={_prev_check.get('seed')}) e os parametros "
              "deste comando nao permitem retomada -- rodar assim SOBRESCREVERIA esse lote. Use "
              "--out <outro-diretorio> para um lote novo separado (preservando o anterior), ou "
              "--overwrite se a intencao e mesmo descartar o que esta em " + out_path + ".")
        return 1

    source_dir = WORK_DIR / "raw" / SOURCES[args.source]
    passages = sample_passages(source_dir, args.max_examples, args.oversample, args.seed)
    print(f"[local] {len(passages)} passagens amostradas de {source_dir}")

    system_prompt = SYSTEM_PROMPT.format(task_types=", ".join(TASK_TYPES))

    if args.dry_run:
        print("[dry-run] nenhuma chamada sera feita. Exemplos de prompt:")
        for p in passages[: min(3, len(passages))]:
            print("=" * 60)
            print(f"uid={p['uid']} id={p['id']} score={p['score']}")
            print(build_user_message(p["text"])[:500])
        return 0

    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    pool = EndpointPool(endpoints)

    decontam_set: set[str] = set()
    if not args.no_decontaminate:
        decontam_set = load_decontamination_set(args.repo_root)
        print(f"[decontam] {len(decontam_set)} prompts normalizados")

    prev = _prev_check
    can_resume = _can_resume_check
    processed_ids: set[str] = set(prev.get("processed_ids", [])) if can_resume else set()
    n = prev.get("total", 0) if can_resume else 0
    n_skip_model = prev.get("skipped_by_model", 0) if can_resume else 0
    n_bad_json = prev.get("skipped_bad_json", 0) if can_resume else 0
    n_contam = prev.get("skipped_contaminated", 0) if can_resume else 0
    n_emoji = prev.get("skipped_emoji", 0) if can_resume else 0
    n_failed = prev.get("failed_requests", 0) if can_resume else 0
    task_counts: dict[str, int] = prev.get("task_type_distribution", {}) if can_resume else {}
    if can_resume:
        print(f"[resume] retomando de manifest anterior: {n} ja gravados, "
              f"{len(processed_ids)} passagens ja processadas (puladas/gravadas)")

    todo = [p for p in passages if p["uid"] not in processed_ids]
    print(f"[local] {len(todo)} passagens a processar ({len(passages) - len(todo)} ja feitas)")

    file_mode = "a" if can_resume and os.path.exists(out_path) else "w"
    write_lock = threading.Lock()
    manifest_lock = threading.Lock()
    fatal_error: list[Exception] = []

    def process_one(p: dict) -> None:
        nonlocal n, n_skip_model, n_bad_json, n_contam, n_emoji, n_failed
        if fatal_error:
            return
        try:
            result = call_with_retry(pool, args.model, system_prompt, build_user_message(p["text"]),
                                      args.max_tokens, args.timeout, args.max_retries)
        except EndpointDown as e:
            fatal_error.append(e)
            return
        with manifest_lock:
            processed_ids.add(p["uid"])
            if result is None:
                n_failed += 1
                return
            content, _usage = result
            parsed = parse_model_json(content)
            if not parsed:
                n_bad_json += 1
                return
            if parsed.get("skip"):
                n_skip_model += 1
                return
            instr = s(parsed.get("instruction"))
            out = s(parsed.get("output"))
            task_type = s(parsed.get("task_type")) or "geral"
            if not instr or not out:
                n_bad_json += 1
                return
            if count_emojis(out) > MAX_EMOJIS_PER_OUTPUT:
                n_emoji += 1
                return
            if is_contaminated(instr, decontam_set):
                n_contam += 1
                return
            row = {
                "instruction": instr,
                "input": f"Contexto: {p['text']}",
                "output": out,
                "task": f"synth_{args.source}_{task_type}",
            }
            with write_lock:
                with open(out_path, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
            n += 1

    def snapshot_manifest(status: str) -> dict:
        return {
            "dataset": f"synth_local_{args.source}", "model": args.model,
            "endpoints": endpoints, "source": args.source,
            "max_examples": args.max_examples, "oversample": args.oversample, "seed": args.seed,
            "n_sampled": len(passages), "total": n,
            "skipped_by_model": n_skip_model, "skipped_bad_json": n_bad_json,
            "skipped_contaminated": n_contam, "skipped_emoji": n_emoji,
            "failed_requests": n_failed,
            "task_type_distribution": task_counts,
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
        futures = {executor.submit(process_one, p): p for p in todo}
        try:
            for fut in as_completed(futures):
                fut.result()
                completed += 1
                if completed % SAVE_EVERY == 0:
                    save_manifest_atomic(out_dir, snapshot_manifest("in_progress"))
                    print(f"[local] {completed}/{len(todo)} processadas, {n} gravadas")
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

    if args.audit_sample > 0 and n > 0:
        with open(out_path, encoding="utf-8") as fp:
            rows = [json.loads(line) for line in fp if line.strip()]
        print(f"[audit] auditando {min(args.audit_sample, len(rows))} exemplos via {args.audit_model}...")
        report = audit_sample(rows, args.audit_sample, args.audit_model, args.seed)
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
    print(f"[local] TOTAL: {n:,} exemplos -> {out_path}")
    print(f"[local] descartados: {n_skip_model} (modelo pulou) | {n_bad_json} (json invalido) | "
          f"{n_contam} (contaminados) | {n_emoji} (emoji) | {n_failed} (falha de requisicao)")
    return 1 if fatal_error else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
