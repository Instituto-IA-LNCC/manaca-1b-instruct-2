#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — Mineração de PROMPTS nocivos a partir do GigaVerbo-v2 Preferences
============================================================================================
PT
--
Item 1 do plano de escala do safety-SFT (ver `sft/README-safety.md` §4):
minera os `prompt` (não as respostas) dos subsets `harmful`/`harmful_reasoning`
do `Polygl0t/gigaverbo-v2-preferences` (Apache 2.0, ~8,3k exemplos nesses
dois subsets) para escalar o pool de seeds nocivos de 50 para potencialmente
centenas — cobrindo fraseado que os 50 escritos à mão não cobrem.

POR QUE ISSO NÃO TEM O PROBLEMA DE PRIVACIDADE DO ToLD-Br/HateBR (ver
`dpo/dpo_seeds/build_ambiguous_seeds.py`): o `prompt` aqui foi GERADO por um
processo sintético (mesma família de dados do `gigaverbo-v2-sft`), não é
fala de uma pessoa real identificável — reproduzir é seguro, assim como o
`mine_refusal_bank.py` já faz com o `chosen`. Este script é o par que
faltava: um extrai as RECUSAS (chosen), este extrai os PEDIDOS (prompt).

DIFERENÇA vs. `build_ambiguous_seeds.py`: aquele não reproduz texto de
terceiros (gera por template validado por evidência estatística); este
reproduz o `prompt` diretamente, porque a fonte é sintética, não social
media real. São duas fontes com riscos diferentes, tratadas de forma
diferente de propósito.

ADVERTÊNCIA (mesma do `mine_refusal_bank.py`): as categorias de dano do
`gigaverbo-v2-preferences` NÃO são documentadas publicamente — a
`categoria` atribuída aqui é heurística (detecta uma coluna de
categoria/tópico se existir; senão usa o nome do subset). Auditar uma
amostra (`--sample-audit N`) antes de treinar com isso.

Uso:
    python dpo/dpo_seeds/mine_harmful_prompts.py --sample-audit 15   # so audita
    python dpo/dpo_seeds/mine_harmful_prompts.py                    # grava

Saída: `dpo/dpo_seeds/safety_prompts_mined.jsonl` (schema idêntico ao
`safety_prompts.jsonl` original: `{id, categoria, prompt}`), já deduplicado
contra os 50 seeds originais + `dpo/dpo_seeds/safety_prompts_ambiguous.jsonl`
(se existir). Junte tudo com:

    cat dpo/dpo_seeds/safety_prompts.jsonl \\
        dpo/dpo_seeds/safety_prompts_ambiguous.jsonl \\
        dpo/dpo_seeds/safety_prompts_mined.jsonl \\
        > dpo/dpo_seeds/safety_prompts_v2.jsonl

EN
--
Mines the `prompt` field (not the harmful completion) from the
harmful/harmful_reasoning subsets of `Polygl0t/gigaverbo-v2-preferences`
(Apache 2.0) to scale the harmful-seed pool from 50 to potentially
hundreds. Safe to reproduce verbatim (synthetic text, not real people's
social-media posts) — the counterpart to `mine_refusal_bank.py`, which
mines the `chosen` (refusal) side of the same subsets. Harm categories
aren't publicly documented; audit a sample before training on this.

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata

HF_REPO_ID = "Polygl0t/gigaverbo-v2-preferences"
LICENSE    = "apache-2.0"
REFERENCE  = "Polygl0t / Tucano 2 (2026). GigaVerbo-v2 Preferences. arXiv:2603.03543."

MIN_TEXT_LEN = 15
MAX_TEXT_LEN = 800  # prompt de pedido costuma ser curto; textao aqui e sinal de schema errado

DEDUP_FILES_DEFAULT = [
    "dpo/dpo_seeds/safety_prompts.jsonl",
    "dpo/dpo_seeds/safety_prompts_ambiguous.jsonl",
]


def s(x) -> str:
    return ("" if x is None else str(x)).strip()


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.strip().lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text)


def _flatten(v) -> str:
    """Schema tolerante: string direta ou formato de mensagem [{'role','content'}, ...]."""
    if isinstance(v, list) and v:
        first_msg = v[0]
        if isinstance(first_msg, dict):
            return s(first_msg.get("content") or first_msg.get("value"))
        return s(first_msg)
    return s(v)


def passes_sanity(text: str) -> bool:
    if len(text) < MIN_TEXT_LEN or len(text) > MAX_TEXT_LEN:
        return False
    n_alpha = sum(1 for c in text if c.isalpha())
    return n_alpha / max(len(text), 1) >= 0.5


def load_dedup_set(paths: list[str]) -> set[str]:
    seen = set()
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                try:
                    seen.add(_normalize(json.loads(line)["prompt"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return seen


def main() -> int:
    ap = argparse.ArgumentParser(description="Minera prompts nocivos do gigaverbo-v2-preferences")
    ap.add_argument("--out", default="dpo/dpo_seeds/safety_prompts_mined.jsonl")
    ap.add_argument("--subsets_filter", default="harmful",
                    help="substring p/ selecionar configs (default 'harmful' -> harmful + harmful_reasoning)")
    ap.add_argument("--dedup_against", nargs="*", default=DEDUP_FILES_DEFAULT,
                    help="arquivos .jsonl existentes para deduplicar (campo 'prompt')")
    ap.add_argument("--max_per_subset", type=int, default=0, help="teto por subset (0 = sem teto)")
    ap.add_argument("--sample-audit", type=int, default=0,
                    help="se > 0, so IMPRIME N amostras aleatorias para auditoria manual (nao grava nada)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as e:
        print(f"[ERRO] 'datasets' ausente ({e}). Rode na imagem manaca-posttrain.")
        return 1

    try:
        all_configs = get_dataset_config_names(HF_REPO_ID)
    except Exception as e:
        print(f"[ERRO] falha ao descobrir configs de {HF_REPO_ID}: {e}")
        return 1

    selected = [c for c in all_configs if args.subsets_filter.lower() in c.lower()] or all_configs
    print(f"[harmful-prompts] subsets selecionados (filtro='{args.subsets_filter}'): {selected}")

    dedup_set = load_dedup_set(args.dedup_against) if not args.sample_audit else set()
    if dedup_set:
        print(f"[harmful-prompts] deduplicando contra {len(dedup_set)} prompts existentes "
              f"({args.dedup_against})")

    seen_norm: set[str] = set(dedup_set)
    mined: list[dict] = []
    counts: dict[str, int] = {}

    for cfg in selected:
        try:
            ds = load_dataset(HF_REPO_ID, cfg, split="train")
        except Exception as e:
            print(f"[aviso] subset {cfg}: falha ao carregar ({e}). PULADO.")
            counts[cfg] = 0
            continue

        first = ds[0] if len(ds) else {}
        prompt_col = next((c for c in ["prompt", "instruction", "question"] if c in first), None)
        category_col = next((c for c in ["category", "categoria", "topic", "harm_category", "risk_category"]
                              if c in first), None)
        if not prompt_col:
            print(f"[aviso] subset {cfg}: nenhuma coluna de prompt reconhecida em {list(first.keys())}. PULADO.")
            counts[cfg] = 0
            continue
        print(f"[harmful-prompts] {cfg}: coluna prompt='{prompt_col}' categoria='{category_col or 'n/a (usa subset)'}'"
              f" | campos: {list(first.keys())}")

        n_kept = 0
        for ex in ds:
            if args.max_per_subset and n_kept >= args.max_per_subset:
                break
            text = _flatten(ex.get(prompt_col))
            if not passes_sanity(text):
                continue
            norm = _normalize(text)
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            categoria = s(ex.get(category_col)) if category_col else f"importado_{cfg}"
            mined.append({
                "id": f"mined_{len(mined):04d}",
                "categoria": categoria or f"importado_{cfg}",
                "prompt": text,
                "origem": "gigaverbo-v2-preferences",
                "subset": cfg,
            })
            n_kept += 1
        counts[cfg] = n_kept
        print(f"[harmful-prompts] {cfg}: {n_kept:,} prompts unicos minerados")

    if args.sample_audit:
        import random
        rng = random.Random(args.seed)
        sample = rng.sample(mined, min(args.sample_audit, len(mined)))
        print(f"\n[audit] {len(sample)} amostras aleatorias (NADA foi gravado em disco):\n")
        for r in sample:
            print("=" * 60)
            print(f"[{r['subset']} / {r['categoria']}] {r['prompt'][:300]}")
        print("\n[audit] Confira se estes SAO de fato pedidos nocivos (nao falsos positivos) e se a "
              "'categoria' faz sentido antes de rodar sem --sample-audit.")
        return 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fp:
        for r in mined:
            fp.write(json.dumps({"id": r["id"], "categoria": r["categoria"], "prompt": r["prompt"]},
                                 ensure_ascii=False) + "\n")

    cat_dist: dict[str, int] = {}
    for r in mined:
        cat_dist[r["categoria"]] = cat_dist.get(r["categoria"], 0) + 1

    manifest = {
        "dataset": "safety_prompts_mined", "hf_repo": HF_REPO_ID, "license": LICENSE,
        "reference": REFERENCE, "subsets": counts, "total_unique": len(mined),
        "deduped_against": args.dedup_against, "category_distribution": cat_dist,
        "status": "CANDIDATO -- auditar com --sample-audit antes de usar (categorias nao documentadas na fonte)",
    }
    with open(os.path.join(os.path.dirname(args.out) or ".", "safety_prompts_mined_manifest.json"),
              "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    print(f"\n[harmful-prompts] TOTAL: {len(mined):,} novos prompts unicos -> {args.out}")
    print(f"[harmful-prompts] distribuicao por categoria: {cat_dist}")
    print("[harmful-prompts] PROXIMO PASSO: auditar uma amostra "
          f"(python {sys.argv[0]} --sample-audit 15) antes de juntar ao mix de seeds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
