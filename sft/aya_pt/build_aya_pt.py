#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — Aya Dataset (subset PT): instrução humana nativa
=========================================================================
PT
--
O Aya Dataset (Singh et al., 2024, CohereForAI/CohereLabs) é 204.114 pares
prompt-completion anotados por FALANTES HUMANOS (não sintético, não
tradução automática) em 65 idiomas, incluindo português. Licença
Apache-2.0.

POR QUE ISSO PREENCHE UMA LACUNA REAL: o projeto tem SFT sintético
(gigaverbo-v2-sft) e traduzido automaticamente (ExtraGLUE) em abundância,
mas pouca instrução humana nativa e livre em tipo de tarefa (o manacá-
jaster é humano mas é sempre no molde de benchmark acadêmico). O Aya é
pequeno mas cobre estilo/variedade que os sintéticos tendem a homogeneizar.

FONTE: por padrão usa `botbot-ai/aya_dataset_pt` (subconjunto PT já
filtrado, se existir e carregar). Se falhar, tenta filtrar
`CohereLabs/aya_dataset` (dataset completo multilingue) pela coluna de
idioma, com detecção tolerante de nome de coluna/código ('pt', 'por',
'portuguese') -- ambos os caminhos sao tentados em sequencia via
`--source` para permitir override manual.

Uso:
    python sft/aya_pt/build_aya_pt.py
    python sft/aya_pt/build_aya_pt.py --source CohereLabs/aya_dataset

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

DEFAULT_SOURCE = "botbot-ai/aya_dataset_pt"
FALLBACK_SOURCE = "CohereLabs/aya_dataset"
LICENSE   = "apache-2.0"
REFERENCE = "Singh et al. (2024). Aya Dataset. CohereForAI/CohereLabs."

PT_LANG_CODES = {"pt", "por", "portuguese", "pt-br", "pt_br", "portuguese (brazil)"}


def s(x) -> str:
    return ("" if x is None else str(x)).strip()


def load_pt_examples(load_dataset, repo_id: str):
    """Retorna (linhas, filtrado_por_idioma: bool, coluna_idioma: str|None)."""
    ds = load_dataset(repo_id, split="train")
    cols = ds.column_names
    lang_col = next((c for c in ["language", "language_code", "lang", "iso_code"] if c in cols), None)
    if lang_col:
        rows = [ex for ex in ds if s(ex.get(lang_col)).lower() in PT_LANG_CODES]
        return rows, True, lang_col
    return list(ds), False, None


def main() -> int:
    ap = argparse.ArgumentParser(description="Converte o Aya Dataset (subset PT) para Alpaca")
    ap.add_argument("--out", default="data/sft/aya_pt", help="diretorio de saida")
    ap.add_argument("--source", default=None,
                    help=f"override do repo HF (default tenta {DEFAULT_SOURCE}, depois {FALLBACK_SOURCE})")
    ap.add_argument("--max_examples", type=int, default=0, help="teto de exemplos (0 = sem teto)")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as e:
        print(f"[ERRO] 'datasets' ausente ({e}). Rode na imagem manaca-posttrain.")
        return 1

    candidates = [args.source] if args.source else [DEFAULT_SOURCE, FALLBACK_SOURCE]
    rows, filtered, lang_col, used_repo = None, False, None, None
    for repo_id in candidates:
        print(f"[aya-pt] tentando {repo_id} ...")
        try:
            rows, filtered, lang_col = load_pt_examples(load_dataset, repo_id)
            used_repo = repo_id
            break
        except Exception as e:
            print(f"[aviso] falha ao carregar {repo_id}: {e}")
    if rows is None:
        print(f"[ERRO] nenhuma fonte carregou. Tente --source <hf_id> manualmente.")
        return 1

    print(f"[aya-pt] fonte usada: {used_repo} | filtrado por idioma (coluna '{lang_col}'): {filtered} | "
          f"{len(rows):,} exemplos brutos")

    if not rows:
        print("[ERRO] 0 exemplos apos carregar/filtrar. Verifique o nome da coluna de idioma com --source.")
        return 1

    p_col = next((c for c in ["inputs", "prompt", "instruction"] if c in rows[0]), None)
    c_col = next((c for c in ["targets", "completion", "output", "response"] if c in rows[0]), None)
    print(f"[aya-pt] colunas detectadas: prompt='{p_col}' completion='{c_col}' | todas: {list(rows[0].keys())}")
    if not p_col or not c_col:
        print(f"[ERRO] colunas de prompt/completion nao reconhecidas em {list(rows[0].keys())}")
        return 1

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "aya_pt.jsonl")
    n = 0
    with open(out_path, "w", encoding="utf-8") as fp:
        for ex in rows:
            if args.max_examples and n >= args.max_examples:
                break
            prompt, completion = s(ex.get(p_col)), s(ex.get(c_col))
            if not prompt or not completion:
                continue
            row = {"instruction": prompt, "input": "", "output": completion, "task": "aya_pt"}
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    if args.shuffle:
        with open(out_path, encoding="utf-8") as fp:
            lines = fp.readlines()
        random.Random(args.seed).shuffle(lines)
        with open(out_path, "w", encoding="utf-8") as fp:
            fp.writelines(lines)

    sha256 = None
    if n:
        h = hashlib.sha256()
        with open(out_path, "rb") as fp:
            for chunk in iter(lambda: fp.read(1 << 16), b""):
                h.update(chunk)
        sha256 = h.hexdigest()

    manifest = {
        "dataset": "aya_pt", "hf_repo": used_repo, "license": LICENSE, "reference": REFERENCE,
        "filtered_by_language_column": lang_col, "total": n,
        "shuffled": args.shuffle, "seed": args.seed, "sha256": sha256,
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    print(f"[aya-pt] TOTAL: {n:,} exemplos -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
