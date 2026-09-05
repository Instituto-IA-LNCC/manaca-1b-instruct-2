#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — Mineração de banco de recusas a partir do GigaVerbo-v2 Preferences
=============================================================================================
PT
--
O `sft/gen_safety_sft.py` usa hoje um banco de recusa de só **8 frases fixas**
(hardcoded, `RECUSAS`/`RECUSA_AUTOLESAO`) — o modelo decora poucas frases em
vez de aprender o COMPORTAMENTO de recusar de forma variada. Este script
minera um banco muito maior de recusas REAIS e diversas a partir das
respostas `chosen` do subset `harmful`/`harmful_reasoning` do
`Polygl0t/gigaverbo-v2-preferences` (Apache 2.0, PT-BR nativo, já usado
neste repo para DPO em `dpo/gigaverbo_v2_preferences/`).

POR QUE ISSO NÃO TEM O PROBLEMA DE PRIVACIDADE DO ToLD-Br/HateBR: o campo
`chosen` desses subsets é **texto gerado por um LLM** (Qwen2.5-32B-Instruct,
já treinado para recusar com segurança), não fala de uma pessoa real — ao
contrário de tweets/comentários reais do ToLD-Br/HateBR, reproduzir esse
texto no repositório não expõe ninguém. Por isso este script escreve o
texto minerado diretamente no `.jsonl` de saída (versionável), diferente de
`dpo/dpo_seeds/build_ambiguous_seeds.py` (ver docstring lá) — nesse, o
mesmo cuidado do projeto com dados sensíveis do HateBR se aplica.

ADVERTÊNCIA (da pesquisa que motivou este script): as categorias de dano do
`gigaverbo-v2-preferences` NÃO são documentadas publicamente — não está
confirmado se o subset cobre bullying/fraseado indireto especificamente.
Este script produz um **POOL DE CANDIDATOS**, não um banco pronto para
uso — audite uma amostra (`--sample-audit N`) antes de ligar isso ao
`gen_safety_sft.py`. Essa integração é um passo SEPARADO, feito depois da
auditoria manual, não por este script.

Uso:
    python sft/safety_seeds/mine_refusal_bank.py
    python sft/safety_seeds/mine_refusal_bank.py --sample-audit 15   # so mostra amostras, nao grava

Saída: `sft/safety_seeds/refusal_bank_mined.jsonl` ({"id", "text", "source_subset"})
+ `manifest.json` (contagens por subset, taxa de deduplicação).

EN
--
Mines a large, diverse refusal-text bank from the `chosen` (safe) responses
of the harmful subsets in `Polygl0t/gigaverbo-v2-preferences` (LLM-generated
text, no privacy concern unlike real social-media comments). Produces a
CANDIDATE pool — the dataset's harm categories aren't publicly documented,
so audit a sample before wiring this into `gen_safety_sft.py`'s refusal
bank (a separate, later step).

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata

HF_REPO_ID = "Polygl0t/gigaverbo-v2-preferences"
LICENSE    = "apache-2.0"
REFERENCE  = "Polygl0t / Tucano 2 (2026). GigaVerbo-v2 Preferences. arXiv:2603.03543."

MIN_TEXT_LEN = 30
MAX_TEXT_LEN = 2000  # respostas gigantes viram texto ruim de "template de recusa"


def s(x) -> str:
    return ("" if x is None else str(x)).strip()


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.strip().lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text)


def _flatten_response(v) -> str:
    """Schema tolerante: string direta ou formato de mensagem [{'role','content'}, ...]."""
    if isinstance(v, list) and v:
        last = v[-1]
        if isinstance(last, dict):
            return s(last.get("content") or last.get("value"))
        return s(last)
    return s(v)


def passes_sanity(text: str) -> bool:
    if len(text) < MIN_TEXT_LEN or len(text) > MAX_TEXT_LEN:
        return False
    n_alpha = sum(1 for c in text if c.isalpha())
    return n_alpha / max(len(text), 1) >= 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description="Minera banco de recusas do gigaverbo-v2-preferences")
    ap.add_argument("--out", default="sft/safety_seeds/refusal_bank_mined.jsonl")
    ap.add_argument("--subsets_filter", default="harmful",
                    help="substring p/ selecionar configs (default 'harmful' -> harmful + harmful_reasoning)")
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
    print(f"[refusal-bank] subsets selecionados (filtro='{args.subsets_filter}'): {selected}")

    seen_norm: set[str] = set()
    mined: list[dict] = []
    counts: dict[str, int] = {}

    for cfg in selected:
        try:
            ds = load_dataset(HF_REPO_ID, cfg, split="train")
        except Exception as e:
            print(f"[aviso] subset {cfg}: falha ao carregar ({e}). PULADO.")
            counts[cfg] = 0
            continue

        n_kept = 0
        for ex in ds:
            if args.max_per_subset and n_kept >= args.max_per_subset:
                break
            chosen = ex.get("chosen") or ex.get("chosen_response")
            text = _flatten_response(chosen)
            if not passes_sanity(text):
                continue
            norm = _normalize(text)
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            mined.append({
                "id": f"mined_{len(mined):05d}",
                "text": text,
                "source_subset": cfg,
            })
            n_kept += 1
        counts[cfg] = n_kept
        print(f"[refusal-bank] {cfg}: {n_kept:,} recusas unicas mineradas")

    if args.sample_audit:
        import random
        rng = random.Random(args.seed)
        sample = rng.sample(mined, min(args.sample_audit, len(mined)))
        print(f"\n[audit] {len(sample)} amostras aleatorias (NADA foi gravado em disco):\n")
        for r in sample:
            print("=" * 60)
            print(f"[{r['source_subset']}] {r['text'][:400]}")
        print("\n[audit] Revise se estas respostas SAO recusas de qualidade em PT-BR antes de "
              "rodar sem --sample-audit e usar o arquivo gerado no gen_safety_sft.py.")
        return 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fp:
        for r in mined:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    sha256 = None
    if mined:
        h = hashlib.sha256()
        with open(args.out, "rb") as fp:
            for chunk in iter(lambda: fp.read(1 << 16), b""):
                h.update(chunk)
        sha256 = h.hexdigest()

    manifest = {
        "dataset": "refusal_bank_mined", "hf_repo": HF_REPO_ID, "license": LICENSE,
        "reference": REFERENCE, "subsets": counts, "total_unique": len(mined),
        "status": "CANDIDATO -- auditar antes de usar (ver docstring do script)",
        "sha256": sha256,
    }
    with open(os.path.join(os.path.dirname(args.out) or ".", "refusal_bank_manifest.json"),
              "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    print(f"\n[refusal-bank] TOTAL: {len(mined):,} recusas unicas -> {args.out}")
    print("[refusal-bank] PROXIMO PASSO: auditar uma amostra "
          f"(python {sys.argv[0]} --sample-audit 15) antes de ligar ao gen_safety_sft.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
