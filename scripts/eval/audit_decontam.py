#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manaca-1B — Auditoria dos achados de scripts/eval/decontaminate.py

O detector de n-gram sinaliza overlap textual, mas não distingue
memorização real (a MESMA questão/trecho apareceu no treino) de falso
positivo (frase-padrão comum, cabeçalho de formulário, boilerplate de
enunciado de prova que se repete em qualquer exame). Isso só se resolve
lendo o texto -- por isso este script, em vez de mais um número agregado.

Para cada achado do relatório, imprime lado a lado:
  1. o texto do item de benchmark (via HF, mesmas fontes de decontaminate.py)
  2. o(s) n-grama(s) de 13 palavras que colidiram (não o trecho inteiro --
     só a prova concreta do overlap)
  3. o texto completo da(s) linha(s) de treino correspondentes

Uso (mesma imagem docker do decontaminate.py):
    python3 scripts/eval/audit_decontam.py \\
        --report docs/evaluation/decontamination-report.json \\
        --train data/sft_instruct2/manaca_sft.jsonl \\
        [--banco enem_pt] [--limit 30]

Autor: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import json
import sys

from decontaminate import carregar_benchmarks, normaliza, ngrams


def main() -> int:
    ap = argparse.ArgumentParser(description="Auditoria manual dos achados de decontaminacao")
    ap.add_argument("--report", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--banco", default=None, help="filtra por um banco so (ex.: enem_pt)")
    ap.add_argument("--limit", type=int, default=0, help="limite de achados a mostrar (0 = todos)")
    ap.add_argument("--n", type=int, default=13)
    args = ap.parse_args()

    with open(args.report, encoding="utf-8") as f:
        relatorio = json.load(f)
    achados = relatorio["achados"]

    if args.banco:
        achados = {k: v for k, v in achados.items() if k.startswith(args.banco + "::")}
    if args.limit:
        achados = dict(list(achados.items())[: args.limit])

    if not achados:
        print("[audit] nenhum achado para os filtros dados.")
        return 0

    print(f"[audit] carregando benchmarks (mesma fonte do decontaminate.py)...")
    bancos = carregar_benchmarks()
    texto_por_item = {}
    for rotulo, itens in bancos.items():
        for item_id, texto in itens:
            texto_por_item[(rotulo, item_id)] = texto

    # indexa so as linhas de treino que aparecem em algum achado (evita reler
    # o arquivo inteiro na memoria -- so guarda as linhas pedidas)
    linhas_pedidas = set()
    for linhas in achados.values():
        linhas_pedidas.update(linhas)
    print(f"[audit] lendo {len(linhas_pedidas)} linhas de treino do arquivo...")
    linha_por_num = {}
    with open(args.train, encoding="utf-8") as fp:
        for i, linha in enumerate(fp):
            if i in linhas_pedidas:
                r = json.loads(linha)
                linha_por_num[i] = r
                if len(linha_por_num) == len(linhas_pedidas):
                    break

    print(f"[audit] {len(achados)} achados a auditar\n" + "=" * 80)
    for chave, linhas in achados.items():
        banco, item_id = chave.split("::", 1)
        texto_bench = texto_por_item.get((banco, item_id), "<NAO ENCONTRADO -- reamostragem do HF mudou?>")
        tokens_bench = set(normaliza(texto_bench))
        ngrams_bench = set(ngrams(normaliza(texto_bench), args.n))

        print(f"\n--- {chave} (linhas de treino: {linhas}) ---")
        print(f"[BENCHMARK] {texto_bench[:500]}")

        for ln in linhas:
            r = linha_por_num.get(ln, {})
            texto_treino = " ".join(str(r.get(c, "")) for c in ("instruction", "input", "output"))
            ngrams_treino = set(ngrams(normaliza(texto_treino), args.n))
            colisao = ngrams_bench & ngrams_treino
            print(f"  [TREINO linha {ln}] {texto_treino[:500]}")
            print(f"  [N-GRAMA COLIDIDO] {' | '.join(list(colisao)[:3])}")
        print("-" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
