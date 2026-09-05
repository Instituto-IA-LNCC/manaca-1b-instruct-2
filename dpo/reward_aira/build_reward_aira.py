#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — Conversão do Reward-Aira Dataset para o esquema DPO
=============================================================================
PT
--
`nicholasKluge/reward-aira-dataset` (projeto Aira) é um dataset de
preferência PT-BR menor, Apache-2.0, complementar ao
`gigaverbo-v2-preferences` (ver dpo/gigaverbo_v2_preferences/) — mais uma
fonte de pares chosen/rejected para a segunda tentativa de DPO deste
projeto (a primeira foi resultado negativo documentado, ver
dpo/README-onpolicy.md).

Mesmo esquema de saída (`{instruction, input, chosen, rejected}`) e mesma
política: converter não é adotar — a decisão de usar DPO depende da
avaliação (segurança no MT-Bench-PT + over-refusal estável), critério já
em uso neste repositório.

Uso:
    python dpo/reward_aira/build_reward_aira.py

EN
--
`nicholasKluge/reward-aira-dataset` (Apache-2.0) — a smaller, complementary
PT-BR preference dataset for the same second-attempt-at-DPO effort as
gigaverbo_v2_preferences. Converts to this repo's DPO schema; adoption is
decided by evaluation, not by this script.

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HF_REPO_ID = "nicholasKluge/reward-aira-dataset"
LICENSE    = "apache-2.0"
REFERENCE  = "Corrêa (2024). Aira / reward-aira-dataset (HuggingFace)."


def s(x) -> str:
    return ("" if x is None else str(x)).strip()


def row_to_dpo(ex: dict) -> dict | None:
    def pick(*keys):
        for k in keys:
            v = ex.get(k)
            if v:
                return v
        return None

    prompt = pick("prompt", "instruction", "question")
    chosen = pick("chosen", "chosen_response")
    rejected = pick("rejected", "rejected_response")
    prompt_s, chosen_s, rejected_s = s(prompt), s(chosen), s(rejected)
    if not prompt_s or not chosen_s or not rejected_s or chosen_s == rejected_s:
        return None
    return {"instruction": prompt_s, "input": "", "chosen": chosen_s, "rejected": rejected_s}


def main() -> int:
    ap = argparse.ArgumentParser(description="Converte nicholasKluge/reward-aira-dataset para o esquema DPO deste repo")
    ap.add_argument("--out", default="dpo_onpolicy/reward_aira", help="diretório de saída")
    ap.add_argument("--split", default="portuguese", help="split a carregar (tenta 'train' se falhar)")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as e:
        print(f"[ERRO] 'datasets' ausente ({e}). Rode na imagem manaca-posttrain.")
        return 1

    try:
        ds = load_dataset(HF_REPO_ID, split=args.split)
    except Exception as e:
        print(f"[aviso] split '{args.split}' falhou ({e}); tentando 'train'...")
        ds = load_dataset(HF_REPO_ID, split="train")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "reward_aira.jsonl")
    n, n_bad = 0, 0
    with open(out_path, "w", encoding="utf-8") as fp:
        for ex in ds:
            row = row_to_dpo(ex)
            if not row:
                n_bad += 1
                continue
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    sha256 = None
    if n:
        h = hashlib.sha256()
        with open(out_path, "rb") as fp:
            for chunk in iter(lambda: fp.read(1 << 16), b""):
                h.update(chunk)
        sha256 = h.hexdigest()

    manifest = {
        "dataset": "reward_aira", "hf_repo": HF_REPO_ID, "license": LICENSE, "reference": REFERENCE,
        "total": n, "skipped_bad_schema": n_bad,
        "usage_note": (
            "Complementa gigaverbo_v2_preferences para a segunda tentativa de DPO. "
            "So adotar apos avaliacao mostrar melhora real (mesmo criterio do projeto)."
        ),
        "sha256": sha256,
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    print(f"[reward-aira] TOTAL: {n:,} pares ({n_bad} descartados por schema) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
