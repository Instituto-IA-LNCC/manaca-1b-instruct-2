#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manaca-1B-Instruct-2 -- Estatistica de poder das sondas (item 1 do checklist)
==============================================================================
Sem dependencias externas (so a stdlib) -- roda com `python3` puro, sem
precisar de container GPU nem instalar scipy.

Uso:
    # (a) IC de Wilson 95% para uma unica taxa
    python3 stats_power.py wilson --k 15 --n 16

    # (b) McNemar pareado entre dois modelos no MESMO conjunto de itens
    #     (arquivos jsonl com um campo booleano "recusou" e um "id" comum)
    python3 stats_power.py mcnemar --a respostas_modeloA.jsonl --b respostas_modeloB.jsonl

    # (c) Holm-Bonferroni sobre uma lista de p-values (de varias comparacoes)
    python3 stats_power.py holm --pvalues 0.001 0.04 0.02 0.20

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import json
import math
import sys


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    """IC de Wilson (score interval) para uma proporcao k/n. Retorna (p_hat, lo, hi), 0-1."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    lo = (centre - adj) / denom
    hi = (centre + adj) / denom
    return (p, max(0.0, lo), min(1.0, hi))


def binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    """P(X <= k) para X ~ Binomial(n, p). Exato via math.comb (sem scipy)."""
    return sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(0, k + 1))


def mcnemar_exact(b: int, c: int) -> float:
    """
    McNemar exato (teste binomial sobre os pares discordantes), 2 lados.
    b = casos onde A acerta/recusa e B nao (ou vice-versa, conforme convencao);
    c = o inverso. So os pares DISCORDANTES importam (b+c).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * binom_cdf(k, n, 0.5)
    return min(1.0, p)


def mcnemar_from_files(path_a: str, path_b: str, field: str = "recusou") -> dict:
    """
    Le dois jsonl com o mesmo conjunto de ids (mesmos prompts, dois modelos
    diferentes) e monta a tabela de contingencia pareada 2x2 sobre `field`
    (booleano). Retorna contagens + p-value de McNemar exato.
    """
    def load(path):
        rows = {}
        with open(path, encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                r = json.loads(line)
                rows[r["id"]] = bool(r.get(field))
        return rows

    a = load(path_a)
    b_ = load(path_b)
    ids = sorted(set(a) & set(b_))
    missing_a = set(b_) - set(a)
    missing_b = set(a) - set(b_)

    both_true = both_false = a_only = b_only = 0
    for i in ids:
        va, vb = a[i], b_[i]
        if va and vb:
            both_true += 1
        elif not va and not vb:
            both_false += 1
        elif va and not vb:
            a_only += 1
        else:
            b_only += 1

    p = mcnemar_exact(a_only, b_only)
    return {
        "n_pareado": len(ids),
        "ids_so_em_b_faltando_em_a": len(missing_a),
        "ids_so_em_a_faltando_em_b": len(missing_b),
        "ambos_true": both_true,
        "ambos_false": both_false,
        "so_A_true": a_only,
        "so_B_true": b_only,
        "p_value_mcnemar_exato": p,
    }


def holm_bonferroni(pvalues: list[float], alpha: float = 0.05) -> list[dict]:
    """
    Correcao de Holm-Bonferroni (step-down), menos conservadora que Bonferroni
    simples mas ainda controla a taxa de erro familia-wise (FWER).
    """
    m = len(pvalues)
    indexed = sorted(range(m), key=lambda i: pvalues[i])
    results = [None] * m
    reject_all_remaining = False
    for rank, idx in enumerate(indexed):
        threshold = alpha / (m - rank)
        p = pvalues[idx]
        if reject_all_remaining:
            significant = False
        else:
            significant = p <= threshold
            if not significant:
                reject_all_remaining = True
        results[idx] = {
            "p_value": p,
            "rank_ordenado": rank + 1,
            "limiar_holm": threshold,
            "significativo_apos_holm": significant,
        }
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_wilson = sub.add_parser("wilson", help="IC de Wilson 95% para uma proporcao k/n")
    p_wilson.add_argument("--k", type=int, required=True, help="numero de sucessos (ex.: recusas)")
    p_wilson.add_argument("--n", type=int, required=True, help="tamanho da amostra")

    p_mc = sub.add_parser("mcnemar", help="McNemar exato pareado entre 2 arquivos jsonl")
    p_mc.add_argument("--a", required=True, help="respostas do modelo A (jsonl com id + campo booleano)")
    p_mc.add_argument("--b", required=True, help="respostas do modelo B (jsonl com id + campo booleano)")
    p_mc.add_argument("--field", default="recusou", help="nome do campo booleano (default: recusou)")

    p_holm = sub.add_parser("holm", help="Holm-Bonferroni sobre uma lista de p-values")
    p_holm.add_argument("--pvalues", type=float, nargs="+", required=True)
    p_holm.add_argument("--alpha", type=float, default=0.05)
    p_holm.add_argument("--labels", nargs="*", default=None, help="rotulos opcionais, mesma ordem dos p-values")

    args = ap.parse_args()

    if args.cmd == "wilson":
        p, lo, hi = wilson_ci(args.k, args.n)
        print(f"k={args.k} n={args.n}")
        print(f"p_hat = {p*100:.1f}%")
        print(f"IC 95% (Wilson) = [{lo*100:.1f}%, {hi*100:.1f}%]")

    elif args.cmd == "mcnemar":
        res = mcnemar_from_files(args.a, args.b, field=args.field)
        for k, v in res.items():
            print(f"{k}: {v}")
        p = res["p_value_mcnemar_exato"]
        print(f"\n{'SIGNIFICATIVO' if p < 0.05 else 'NAO significativo'} a alpha=0.05 "
              f"(p={p:.4f}) -- lembrar de aplicar Holm-Bonferroni se houver mais de uma comparacao.")

    elif args.cmd == "holm":
        labels = args.labels or [f"comparacao_{i+1}" for i in range(len(args.pvalues))]
        results = holm_bonferroni(args.pvalues, args.alpha)
        for label, r in zip(labels, results):
            flag = "SIM" if r["significativo_apos_holm"] else "nao"
            print(f"{label}: p={r['p_value']:.4f} rank={r['rank_ordenado']} "
                  f"limiar={r['limiar_holm']:.4f} significativo_apos_holm={flag}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
