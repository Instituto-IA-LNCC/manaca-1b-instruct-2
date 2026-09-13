#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manaca-1B — Redige respostas nocivas ANTES de comitar (auditoria de segurança)
================================================================================
Baselines sem safety-tuning dedicado podem obedecer pedidos nocivos do
MT-Bench-PT (categoria `seguranca`, rubrica invertida: recusar bem = nota
alta). A NOTA do juiz é o dado cientifico; o texto operacional bruto
(instrucoes de verdade) NAO pode ir pro repo publico. Mesmo protocolo ja
usado pro `safety_2` do `instruct2-full`: julgar com o conteudo real
primeiro, redigir DEPOIS, nos dois arquivos (answers/ e judged/).

Uso:
    python3 scripts/eval/redact_answers.py \\
        --answers bench/mtbench_pt/answers/tucano-1b1-instruct.jsonl \\
        --judged  bench/mtbench_pt/judged/tucano-1b1-instruct.jsonl \\
        --ids safety_1,safety_2,safety_3,safety_4,safety_5,safety_6 \\
        --motivo "obedeceu pedido nocivo (invasao de email/explosivo/etc)"

Redige em memoria e sobrescreve os arquivos originais (faz backup .bak ao
lado). So mexe nos campos de texto (answer/resposta, justificativa_juiz,
juiz_bruto); NUNCA mexe em `nota` (o numero e o dado que entra na tabela).

Autor: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

CAMPOS_TEXTO = ["answer", "resposta", "justificativa_juiz", "juiz_bruto"]


def redigir_arquivo(path: str, ids: set[str], marcador: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as fp:
        linhas = [json.loads(l) for l in fp if l.strip()]
    n = 0
    for rec in linhas:
        if rec.get("id") in ids:
            for campo in CAMPOS_TEXTO:
                if rec.get(campo):
                    rec[campo] = marcador
            n += 1
    if n:
        shutil.copy2(path, path + ".bak")
        with open(path, "w", encoding="utf-8") as out:
            for rec in linhas:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Redige respostas nocivas antes de comitar")
    ap.add_argument("--answers", help="jsonl de respostas cruas (bench/*/answers/<label>.jsonl)")
    ap.add_argument("--judged", help="jsonl julgado (bench/mtbench_pt/judged/<label>.jsonl), opcional")
    ap.add_argument("--ids", required=True, help="ids separados por virgula (ex.: safety_1,safety_2)")
    ap.add_argument("--motivo", required=True, help="motivo da redacao, vai no marcador")
    args = ap.parse_args()

    ids = set(i.strip() for i in args.ids.split(",") if i.strip())
    marcador = f"[REDIGIDO — {args.motivo}. Nota do juiz preservada; texto bruto removido antes do commit.]"

    n_a = redigir_arquivo(args.answers, ids, marcador)
    n_j = redigir_arquivo(args.judged, ids, marcador)
    print(f"[redact] {args.answers or '(nao informado)'}: {n_a} registro(s) redigido(s)")
    if args.judged:
        print(f"[redact] {args.judged}: {n_j} registro(s) redigido(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
