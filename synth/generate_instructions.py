#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — Geração de instrução sintética a partir de corpus bruto
==================================================================================
PT
--
Este é o passo que faltava entre `corpus/scripts/` (passagens brutas em
Parquet: LegalPT, ClassiCC-PT, Carolina, MADLAD-400, FineWeb-2, HPLT 2.0,
Portuguese-PD) e `sft/prepare_data.py` (mix de SFT em Alpaca): transforma
cada passagem amostrada em UM exemplo `{instruction, input, output, task}`
via a API da Anthropic (Claude), grounded no texto da passagem (a resposta
não pode inventar fatos além do que está nela).

USA A MESSAGE BATCHES API (não chamadas síncronas uma a uma):
    50% mais barato, até 100.000 requisições por lote, resultados
    disponíveis por 29 dias. Isso significa que o processo é DESACOPLADO:
    `submit` manda o lote e sai na hora (não precisa ficar com o processo
    rodando por até 24h); `collect` busca os resultados depois, quando
    prontos (ou automaticamente, com `run`, que faz as duas coisas com
    polling). Ver docs/scripts equivalentes usam client.messages.create
    síncrono (bench/mtbench_pt/*, dpo/gen_dpo_pairs.py) — aqui o volume
    (milhares de passagens) e o custo justificam o Batches.

MODELO: default `claude-opus-5` (não baixamos para um modelo mais barato
por padrão — decisão de custo é do usuário via `--model`/`MODEL` env).
Thinking fica no padrão adaptativo do modelo (não desligamos).

COMO A INSTRUÇÃO É GERADA (grounded, não factual-livre):
    Um único prompt de sistema (compartilhado entre TODAS as requisições do
    lote — cache de prompt automático) pede para o modelo: (1) escolher o
    tipo de tarefa mais adequado à passagem dentre um menu configurável
    (resumo, pergunta_resposta, explicacao_simples, extracao_pontos_chave,
    reescrita_tom, questao_estilo_prova); (2) gerar uma instrução que uma
    pessoa real faria; (3) responder usando SÓ informação da passagem;
    (4) devolver tudo como um objeto JSON. A passagem inteira vira o campo
    `input` do exemplo final (mesmo padrão `Contexto: ...` já usado no
    manacá-jaster) — o modelo aprende a responder com base no contexto
    dado, não a "decorar" fatos soltos.

AMOSTRAGEM (2 passadas, não carrega texto de milhões de docs na RAM):
    Passada 1: lê só as colunas `id`/`score` de todos os shards Parquet da
    fonte -> monta um pool ranqueado por qualidade. Passada 2: sorteia
    `--max-examples` IDs desse pool (viés para score alto + aleatoriedade
    para diversidade) e RE-lê só o texto desses IDs específicos.

DECONTAMINAÇÃO (best-effort, mesma limitação já documentada em
sft/gigaverbo_v2_sft/ e sft/tucano_sft/): contra os arquivos estáticos
deste repo (bench/, dpo/dpo_seeds/, sft/safety_seeds/); NÃO cobre
ENEM/BLUEX/OAB/ASSIN2/HateBR (carregados via HF pelo lm-eval).

CUSTO: cada resultado do lote traz `usage.input_tokens`/`output_tokens`;
o manifest final soma tudo e aplica um preço aproximado por modelo (tabela
`PRICING_SNAPSHOT` abaixo, ~50% de desconto do Batches já aplicado) — é uma
ESTIMATIVA (a tabela pode ficar desatualizada; confira o preço vigente
antes de rodar em escala).

Uso:
    export ANTHROPIC_API_KEY=sk-ant-...

    # Testar sem gastar (imprime os prompts que seriam enviados)
    python synth/generate_instructions.py run --source classiccpt --max-examples 5 --dry-run

    # Fluxo completo (submete, espera, coleta) — mais simples
    python synth/generate_instructions.py run --source legalpt --max-examples 2000

    # Fluxo desacoplado (não precisa manter o processo rodando)
    python synth/generate_instructions.py submit --source classiccpt --max-examples 5000
    #   -> imprime o(s) batch_id e grava o estado em $OUT/batch_state.json
    python synth/generate_instructions.py collect --state data/synth/classiccpt/batch_state.json

Saída: `{instruction, input, output, task}` (esquema Alpaca), consumido via
`--extra` de `sft/prepare_data.py`, igual aos demais builders deste repo.

    data/synth/<source>/
    ├── batch_state.json      IDs do(s) lote(s) + passagens amostradas (para o collect)
    ├── <source>_synth.jsonl  saida final (apos collect)
    └── manifest.json         contagens, custo estimado, distribuicao de task_type, sha256

EN
--
The missing step between `corpus/scripts/` (raw Parquet passages) and
`sft/prepare_data.py` (Alpaca SFT mix): turns each sampled passage into one
`{instruction, input, output, task}` example via the Anthropic Batches API
(50% cheaper, decoupled submit/collect — no need to keep a process running
for up to 24h), grounded in the passage text (the response may not invent
facts beyond it). Default model `claude-opus-5`; the whole passage becomes
the `input` field (same `Contexto: ...` pattern already used by
manacá-jaster). Same best-effort decontamination limitation as the other
builders in this repo (does not cover HF-loaded exam benchmarks).

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pyarrow.parquet as pq
except ImportError as e:
    print(f"[ERRO] Dependência ausente: {e}")
    print("Execute: pip install -r requirements/posttrain.txt -r requirements/corpus.txt")
    sys.exit(1)


# ── Configuração ──────────────────────────────────────────────────────────────

WORK_DIR = Path(os.environ.get("WORK_DIR", Path.home() / "manaca-corpus"))

SOURCES = {
    "legalpt": "legalpt", "classiccpt": "classiccpt", "carolina": "carolina",
    "madlad400": "madlad400", "fineweb2": "fineweb2", "hplt2": "hplt2",
    "portuguese_pd": "portuguese_pd",
}

DEFAULT_MODEL = os.environ.get("MODEL", "claude-opus-5")
DEFAULT_MAX_TOKENS = 1500
DEFAULT_MAX_EXAMPLES = 2000
DEFAULT_OVERSAMPLE = 5  # tamanho do pool ranqueado por score = max_examples * este fator

TASK_TYPES = [
    "resumo", "pergunta_resposta", "explicacao_simples",
    "extracao_pontos_chave", "reescrita_tom", "questao_estilo_prova",
]

SYSTEM_PROMPT = """Você transforma uma passagem de texto em português do Brasil em UM exemplo de \
treino de instrução para um modelo de linguagem.

Regras obrigatórias:
- Responda inteiramente em português do Brasil.
- A instrução deve soar como algo que uma pessoa real pediria a um assistente de IA.
- A resposta deve usar SOMENTE informação contida na passagem fornecida. NÃO invente \
fatos, números, nomes ou datas que não estejam no texto.
- Escolha o tipo de tarefa mais adequado à passagem, um destes: {task_types}.
- Se a passagem for curta, incoerente ou não render uma instrução de qualidade, \
responda {{"skip": true}} em vez de forçar um exemplo ruim.
- Responda APENAS com um objeto JSON válido, sem markdown, sem texto antes ou depois, \
no formato exato: {{"task_type": "...", "instruction": "...", "output": "..."}}"""

# Preço aproximado por 1M tokens (input/output), USD, JA com o desconto de
# 50% do Batches API aplicado. Fonte: tabela de precos da skill claude-api
# (cache: 2026-06-24) / 2. CONFERIR o preco vigente antes de rodar em escala.
PRICING_SNAPSHOT = {
    "claude-opus-5":    (2.50, 12.50),
    "claude-sonnet-5":  (1.00, 5.00),
    "claude-haiku-4-5": (0.50, 2.50),
}

DECONTAMINATION_SOURCES = [
    ("bench/ifeval_pt/prompts.jsonl", "turns"),
    ("bench/mtbench_pt/questions.jsonl", "turns"),
    ("bench/safety_pt/heldout.jsonl", "prompt"),
    ("bench/overrefusal_pt/prompts.jsonl", "prompt"),
    ("dpo/dpo_seeds/safety_prompts.jsonl", "prompt"),
    ("sft/safety_seeds/benign_help.jsonl", "prompt"),
]


def s(x) -> str:
    return ("" if x is None else str(x)).strip()


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.strip().lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text)


def load_decontamination_set(repo_root: str) -> set[str]:
    seen: set[str] = set()
    for rel_path, field in DECONTAMINATION_SOURCES:
        path = os.path.join(repo_root, rel_path)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                val = row.get(field)
                text = val[0] if isinstance(val, list) and val else val
                text = s(text)
                if text:
                    seen.add(_normalize(text))
    return seen


def is_contaminated(instruction: str, decontam_set: set[str]) -> bool:
    if not decontam_set:
        return False
    norm = _normalize(instruction)
    if norm in decontam_set:
        return True
    return any(norm in bench or bench in norm for bench in decontam_set if len(bench) > 20)


# ── Amostragem (2 passadas) ───────────────────────────────────────────────────

def sample_passages(source_dir: Path, max_examples: int, oversample: int, seed: int) -> list[dict]:
    """Amostra passagens com viés de qualidade (score) sem carregar todo o texto na RAM."""
    shards = sorted(source_dir.glob("shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"Nenhum shard encontrado em {source_dir}. Rode o corpus/scripts/ correspondente antes.")

    # Passada 1: so id + score
    pool: list[tuple[str, float, int]] = []  # (id, score, shard_index)
    for shard_idx, shard in enumerate(shards):
        t = pq.read_table(shard, columns=["id", "score"])
        ids, scores = t["id"].to_pylist(), t["score"].to_pylist()
        pool.extend((str(i), float(sc or 0.0), shard_idx) for i, sc in zip(ids, scores))

    pool_size = min(len(pool), max_examples * oversample) or len(pool)
    pool.sort(key=lambda x: x[1], reverse=True)
    top_pool = pool[:pool_size]
    rng = random.Random(seed)
    rng.shuffle(top_pool)
    selected = top_pool[:max_examples]
    wanted_ids = {sid for sid, _, _ in selected}
    wanted_shards = {shard_idx for _, _, shard_idx in selected}

    # Passada 2: re-le so os shards necessarios, filtrando pelos ids sorteados
    found: dict[str, dict] = {}
    for shard_idx in sorted(wanted_shards):
        t = pq.read_table(shards[shard_idx], columns=["id", "text", "source", "score"])
        for i, text, src, sc in zip(t["id"].to_pylist(), t["text"].to_pylist(),
                                     t["source"].to_pylist(), t["score"].to_pylist()):
            i = str(i)
            if i in wanted_ids and i not in found:
                found[i] = {"id": i, "text": text, "source": src, "score": sc}

    return [found[sid] for sid, _, _ in selected if sid in found]


# ── Prompt e parsing ──────────────────────────────────────────────────────────

def build_user_message(passage_text: str) -> str:
    return f"Passagem:\n\n{passage_text}"


def parse_model_json(raw_text: str) -> dict | None:
    """Parsing tolerante: remove cercas de markdown, extrai o primeiro objeto {...}."""
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


# ── Estado (para desacoplar submit/collect) ───────────────────────────────────

def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)


def load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ── Submissão do(s) lote(s) ────────────────────────────────────────────────────

def cmd_submit(args) -> dict:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    source_dir = WORK_DIR / "raw" / SOURCES[args.source]
    passages = sample_passages(source_dir, args.max_examples, args.oversample, args.seed)
    print(f"[submit] {len(passages)} passagens amostradas de {source_dir}")

    system = [{"type": "text",
               "text": SYSTEM_PROMPT.format(task_types=", ".join(TASK_TYPES)),
               "cache_control": {"type": "ephemeral"}}]

    if args.dry_run:
        print("[dry-run] nenhuma chamada de API sera feita. Exemplos de prompt:")
        for p in passages[: min(3, len(passages))]:
            print("=" * 60)
            print(f"id={p['id']} score={p['score']}")
            print(build_user_message(p["text"])[:500])
        return {"dry_run": True, "n_sampled": len(passages)}

    import anthropic
    client = anthropic.Anthropic()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Chunking defensivo: Batches API aceita ate 100k requisicoes por lote.
    CHUNK = 100_000
    batch_ids = []
    passages_by_custom_id = {}
    for chunk_start in range(0, len(passages), CHUNK):
        chunk = passages[chunk_start: chunk_start + CHUNK]
        requests = []
        for p in chunk:
            custom_id = f"{args.source}_{p['id']}"
            passages_by_custom_id[custom_id] = p
            requests.append(Request(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(
                    model=args.model, max_tokens=args.max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": build_user_message(p["text"])}],
                ),
            ))
        batch = client.messages.batches.create(requests=requests)
        batch_ids.append(batch.id)
        print(f"[submit] lote criado: {batch.id} ({len(requests)} requisicoes)")

    state = {
        "source": args.source, "model": args.model, "max_tokens": args.max_tokens,
        "batch_ids": batch_ids, "n_sampled": len(passages),
        "passages_by_custom_id": passages_by_custom_id,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    state_path = out_dir / "batch_state.json"
    save_state(state_path, state)
    print(f"[submit] estado salvo em {state_path}")
    print(f"[submit] rode depois: python synth/generate_instructions.py collect --state {state_path}")
    return state


# ── Coleta e conversão dos resultados ──────────────────────────────────────────

def cmd_collect(args) -> dict:
    import anthropic
    client = anthropic.Anthropic()

    state_path = Path(args.state)
    state = load_state(state_path)
    out_dir = state_path.parent

    print(f"[collect] aguardando {len(state['batch_ids'])} lote(s)...")
    for batch_id in state["batch_ids"]:
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                print(f"[collect] {batch_id}: concluido "
                      f"(sucesso={batch.request_counts.succeeded}, erro={batch.request_counts.errored})")
                break
            print(f"[collect] {batch_id}: {batch.processing_status} "
                  f"(processando={batch.request_counts.processing})")
            if args.no_wait:
                print("[collect] --no_wait: saindo sem esperar. Rode de novo mais tarde.")
                return {"status": "pending"}
            time.sleep(args.poll_interval)

    decontam_set: set[str] = set()
    if not args.no_decontaminate:
        decontam_set = load_decontamination_set(args.repo_root)
        print(f"[decontam] {len(decontam_set)} prompts normalizados")

    passages_by_custom_id = state["passages_by_custom_id"]
    out_path = out_dir / f"{state['source']}_synth.jsonl"
    n, n_skip_model, n_bad_json, n_contam, in_tok, out_tok = 0, 0, 0, 0, 0, 0
    task_counts: dict[str, int] = {}

    with open(out_path, "w", encoding="utf-8") as fout:
        for batch_id in state["batch_ids"]:
            for result in client.messages.batches.results(batch_id):
                p = passages_by_custom_id.get(result.custom_id, {})
                if result.result.type != "succeeded":
                    continue
                msg = result.result.message
                in_tok += msg.usage.input_tokens
                out_tok += msg.usage.output_tokens
                text = next((b.text for b in msg.content if b.type == "text"), "")
                parsed = parse_model_json(text)
                if not parsed:
                    n_bad_json += 1
                    continue
                if parsed.get("skip"):
                    n_skip_model += 1
                    continue
                instr = s(parsed.get("instruction"))
                out = s(parsed.get("output"))
                task_type = s(parsed.get("task_type")) or "geral"
                if not instr or not out:
                    n_bad_json += 1
                    continue
                if is_contaminated(instr, decontam_set):
                    n_contam += 1
                    continue
                row = {
                    "instruction": instr,
                    "input": f"Contexto: {p.get('text', '')}",
                    "output": out,
                    "task": f"synth_{state['source']}_{task_type}",
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                task_counts[task_type] = task_counts.get(task_type, 0) + 1
                n += 1

    sha256 = None
    if n:
        h = hashlib.sha256()
        with open(out_path, "rb") as fp:
            for chunk in iter(lambda: fp.read(1 << 16), b""):
                h.update(chunk)
        sha256 = h.hexdigest()

    price_in, price_out = PRICING_SNAPSHOT.get(state["model"], (None, None))
    est_cost = (round(in_tok / 1e6 * price_in + out_tok / 1e6 * price_out, 2)
                if price_in is not None else None)

    manifest = {
        "dataset": f"synth_{state['source']}", "model": state["model"],
        "n_sampled": state["n_sampled"], "total": n,
        "skipped_by_model": n_skip_model, "skipped_bad_json": n_bad_json,
        "skipped_contaminated": n_contam,
        "task_type_distribution": task_counts,
        "tokens_input": in_tok, "tokens_output": out_tok,
        "estimated_cost_usd": est_cost,
        "estimated_cost_note": "Batches API (50% ja aplicado na tabela de precos). CONFERIR preco vigente.",
        "decontaminated": not args.no_decontaminate,
        "decontamination_note": (
            "Best-effort contra bench/dpo/sft/safety_seeds ESTATICOS deste repo. "
            "NAO cobre ENEM/BLUEX/OAB/ASSIN2/HateBR (carregados via HF pelo lm-eval)."
        ),
        "sha256": sha256, "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    print(f"[collect] TOTAL: {n:,} exemplos -> {out_path}")
    print(f"[collect] descartados: {n_skip_model} (modelo pulou) | {n_bad_json} (json invalido) | "
          f"{n_contam} (contaminados)")
    print(f"[collect] tokens: {in_tok:,} entrada / {out_tok:,} saida"
          + (f" | custo estimado: US${est_cost}" if est_cost is not None else ""))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Geracao de instrucao sintetica via Anthropic Batches API")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--source", choices=sorted(SOURCES), required=True)
        p.add_argument("--out", default=None, help="diretorio de saida (default: data/synth/<source>)")
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
        p.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
        p.add_argument("--oversample", type=int, default=DEFAULT_OVERSAMPLE,
                        help="tamanho do pool ranqueado por score = max-examples * este fator")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--dry-run", action="store_true", help="nao chama a API; so mostra os prompts")

    def add_collect_args(p):
        p.add_argument("--repo_root", default=".")
        p.add_argument("--no_decontaminate", action="store_true")
        p.add_argument("--no_wait", action="store_true", help="verifica uma vez e sai (nao espera o lote terminar)")
        p.add_argument("--poll_interval", type=int, default=60)

    p_submit = sub.add_parser("submit", help="amostra passagens e submete o(s) lote(s), sem esperar")
    add_common(p_submit)

    p_collect = sub.add_parser("collect", help="busca resultados de um lote ja submetido")
    p_collect.add_argument("--state", required=True, help="caminho para o batch_state.json (do submit)")
    add_collect_args(p_collect)

    p_run = sub.add_parser("run", help="submit + espera + collect, em um comando so")
    add_common(p_run)
    add_collect_args(p_run)

    args = ap.parse_args()

    if args.cmd in ("submit", "run") and args.out is None:
        args.out = f"data/synth/{args.source}"

    if args.cmd == "submit":
        cmd_submit(args)
        return 0

    if args.cmd == "collect":
        cmd_collect(args)
        return 0

    if args.cmd == "run":
        state = cmd_submit(args)
        if args.dry_run:
            return 0
        state_path = Path(args.out) / "batch_state.json"
        collect_args = argparse.Namespace(
            state=str(state_path), repo_root=args.repo_root,
            no_decontaminate=args.no_decontaminate, no_wait=False,
            poll_interval=args.poll_interval,
        )
        cmd_collect(collect_args)
        return 0

    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INFO] Interrompido. Para 'run', o(s) lote(s) ja submetido(s) continuam processando no "
              "servidor -- use 'collect --state <arquivo>' depois para buscar os resultados sem re-gastar.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
