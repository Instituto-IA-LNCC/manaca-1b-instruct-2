#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — Projeção do tamanho do safety-SFT (sem GPU, sem API)
============================================================================
PT
--
Responde "quantos temos / é suficiente" para o safety-SFT sem precisar
rodar `gen_safety_sft.py` (que exige `--model`/checkpoint para o lado
on-policy) nem gastar API: só CONTA os arquivos de seeds/benign que já
existem no disco (hand-written + candidatos minerados/gerados, incluindo
os ainda não auditados) e aplica a mesma fórmula de `gen_safety_sft.py`
(recusa = seeds x k_refuse; ajuda = pronta + on-policy; geral = n_general)
para projetar o total. Meta: ~2.000 (acima do AnswerCarefully v2 do
LLM-jp, 1.800) -- ver `sft/README-safety.md` §6.

Uso:
    python sft/safety_seeds/project_safety_sft_size.py
    python sft/safety_seeds/project_safety_sft_size.py --k_refuse 4 --n_general 200
    python sft/safety_seeds/project_safety_sft_size.py --include_unaudited=false  # so o que ja foi auditado/aprovado

EN
--
Answers "how many do we have / is it enough" for the safety-SFT without a
GPU or API spend: counts the seed/benign files already on disk (hand-
written plus any not-yet-audited mined/generated candidates) and applies
`gen_safety_sft.py`'s own formula to project the total against the ~2,000
target (AnswerCarefully v2 = 1,800).

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (rótulo, caminho, auditado?) -- "auditado" = conteúdo hand-written/revisado;
# os candidatos de mineração/geração via API entram como NAO auditado até
# alguém rodar --sample-audit e aprovar.
SEED_FILES = [
    ("seeds originais (hand-written)", "dpo/dpo_seeds/safety_prompts.jsonl", True),
    ("seeds ambiguos (templates, evidencia ToLD-Br/HateBR)", "dpo/dpo_seeds/safety_prompts_ambiguous.jsonl", True),
    ("seeds minerados (gigaverbo-v2-preferences)", "dpo/dpo_seeds/safety_prompts_mined.jsonl", False),
    ("seeds gerados via API (Claude)", "dpo/dpo_seeds/safety_prompts_generated.jsonl", False),
]

BENIGN_FILES = [
    ("benign_help original (hand-written, inclui 5 'ajudar_em_crise')", "sft/safety_seeds/benign_help.jsonl", True),
    ("benign_help gerado via API (Claude)", "sft/safety_seeds/benign_help_generated.jsonl", False),
]


def count_jsonl(path: str) -> int:
    full = os.path.join(REPO, path)
    if not os.path.isfile(full):
        return 0
    n = 0
    with open(full, encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                n += 1
    return n


def count_pronto(path: str) -> int:
    """Quantos registros de um arquivo benign_help.jsonl-like ja tem `output` preenchido
    (ajuda_pronta, nao precisa de --model)."""
    full = os.path.join(REPO, path)
    if not os.path.isfile(full):
        return 0
    n = 0
    with open(full, encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("output"):
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Projeta o tamanho do safety-SFT a partir dos arquivos de seeds/benign no disco")
    ap.add_argument("--k_refuse", type=int, default=4, help="variantes de recusa por seed nocivo (default: receita 'safe4')")
    ap.add_argument("--n_general", type=int, default=200, help="amostra de dados gerais (default: receita 'safe4')")
    ap.add_argument("--target", type=int, default=2000, help="meta de exemplos (default 2000)")
    ap.add_argument("--include_unaudited", type=lambda x: x.lower() != "false", default=True,
                    help="incluir candidatos ainda nao auditados na projecao (default: true)")
    args = ap.parse_args()

    print(f"[projecao] k_refuse={args.k_refuse} n_general={args.n_general} "
          f"include_unaudited={args.include_unaudited} meta={args.target}\n")

    print("--- lado RECUSA (seeds nocivos) ---")
    total_seeds = 0
    for label, path, audited in SEED_FILES:
        if not audited and not args.include_unaudited:
            print(f"  [pulado -- nao auditado] {label}: {path}")
            continue
        n = count_jsonl(path)
        tag = "" if audited else " (NAO AUDITADO -- rode --sample-audit antes de treinar)"
        print(f"  {n:5d}  {label}{tag}")
        total_seeds += n
    recusa = total_seeds * args.k_refuse
    print(f"  total seeds: {total_seeds}  x k_refuse={args.k_refuse}  ->  recusa ~= {recusa}\n")

    print("--- lado AJUDA (benign-help) ---")
    total_ajuda = 0
    for label, path, audited in BENIGN_FILES:
        if not audited and not args.include_unaudited:
            print(f"  [pulado -- nao auditado] {label}: {path}")
            continue
        n = count_jsonl(path)
        pronto = count_pronto(path)
        tag = "" if audited else " (NAO AUDITADO)"
        print(f"  {n:5d}  {label}{tag}  ({pronto} com resposta pronta, {n - pronto} on-policy)")
        total_ajuda += n
    print(f"  total ajuda: {total_ajuda}\n")

    geral = args.n_general
    total = recusa + total_ajuda + geral

    print("--- PROJECAO TOTAL ---")
    print(f"  recusa: {recusa}")
    print(f"  ajuda:  {total_ajuda}")
    print(f"  geral:  {geral}")
    print(f"  TOTAL:  {total}  (meta: {args.target}, AnswerCarefully v2: 1800, atual medido: 411)")
    if total >= args.target:
        print(f"\n  >= meta ({args.target}). Falta auditar os candidatos antes de treinar de verdade.")
    else:
        falta = args.target - total
        seeds_extra = -(-falta // args.k_refuse)  # ceil
        print(f"\n  < meta ({args.target}) por {falta}. Ao ritmo atual, faltam ~{seeds_extra} "
              "seeds nocivos novos (ou mais benign-help/geral) para fechar a conta.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
