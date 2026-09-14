#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manaca-1B — Decontaminação: n-gramas dos benchmarks de avaliação vs. corpus de treino
======================================================================================
Checa se algum trecho longo (13-gram de palavras, mesmo limiar usado por
GPT-3/Llama pra decontaminação) dos benchmarks de avaliação aparece
literalmente no corpus de treino do instruct-2. Não detecta paráfrase — só
overlap textual quase-exato, que é o tipo mais grave (memorização direta).

Por que estes benchmarks: ENEM/OAB/BLUEX/ASSIN2/FaQuAD-NLI/HateBR/HateSpeech
NAO fazem parte do corpus de treino por nome de dataset (confirmado em
docs/data/README-instruct2.md) -- mas as fontes de web scrape em larga
escala do treino (GigaVerbo-v2 SFT, MADLAD-400, FineWeb-2, HPLT2,
ClassiCC-PT, LegalPT) podem ter capturado esse texto incidentalmente, por
estar circulando na internet (provas publicadas, datasets de NLI que
circulam em papers/repos). MT-Bench-PT/IFEval-PT/sondas de seguranca
tambem entram, por completude (risco menor, sao majoritariamente escritos
ou gerados especificamente pra este projeto).

Uso (dentro da imagem manaca-lmeval, que ja tem `datasets`):
    python3 scripts/eval/decontaminate.py \\
        --train data/sft_instruct2/manaca_sft.jsonl \\
        --out docs/evaluation/decontamination-report

So biblioteca padrao + `datasets` (ja e dependencia do lm-eval-harness).

Autor: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

N_GRAM = 13  # mesmo limiar de GPT-3/Llama pra decontaminacao textual


def normaliza(texto: str) -> list[str]:
    """minusculo, so letras/numeros/espaco, tokeniza por espaco -- robusto a
    pontuacao/acentuacao diferente entre fonte web e o dataset formatado."""
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", " ", texto, flags=re.UNICODE)
    return texto.split()


def ngrams(tokens: list[str], n: int = N_GRAM):
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


def carregar_benchmarks() -> dict:
    """Retorna {rotulo: [(id, texto), ...]} de cada benchmark de avaliacao."""
    from datasets import load_dataset

    bancos = {}

    # --- Exames de multipla escolha (via HF, mesmos ids do harness) ---
    for rotulo, hf_id, split in [
        ("enem_pt", "eduagarcia/enem_challenge", "train"),
        ("oab_pt", "eduagarcia/oab_exams", "train"),
        ("bluex_pt", "eduagarcia-temp/BLUEX_without_images", "train"),
    ]:
        try:
            ds = load_dataset(hf_id, split=split)
            itens = []
            for i, ex in enumerate(ds):
                q = str(ex.get("question", ""))
                choices = ex.get("choices", {}) or {}
                texts = choices.get("text") or []
                texto = q + " " + " ".join(str(t) for t in texts)
                itens.append((f"{rotulo}_{i}", texto))
            bancos[rotulo] = itens
            print(f"[decontam] {rotulo}: {len(itens)} itens carregados de {hf_id}")
        except Exception as e:
            print(f"[decontam] AVISO: falhou carregar {hf_id} ({e}), pulando {rotulo}")

    # --- Classificacao (via HF) ---
    for rotulo, hf_id, split, campos in [
        ("assin2_rte_pt", "assin2", "test", ["premise", "hypothesis"]),
        ("faquad_nli_pt", "ruanchaves/faquad-nli", "test", ["question", "answer"]),
        ("hatebr_pt", "ruanchaves/hatebr", "test", ["instagram_comments"]),
        ("hatespeech_pt", "hate_speech_portuguese", "train", ["text"]),
    ]:
        try:
            ds = load_dataset(hf_id, split=split)
            itens = []
            for i, ex in enumerate(ds):
                texto = " ".join(str(ex.get(c, "")) for c in campos)
                itens.append((f"{rotulo}_{i}", texto))
            bancos[rotulo] = itens
            print(f"[decontam] {rotulo}: {len(itens)} itens carregados de {hf_id}")
        except Exception as e:
            print(f"[decontam] AVISO: falhou carregar {hf_id} ({e}), pulando {rotulo}")

    # --- Locais (jsonl no repo): MT-Bench-PT, IFEval-PT, sondas de seguranca ---
    REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    locais = [
        ("mtbench_pt", "bench/mtbench_pt/questions.jsonl", lambda r: " ".join(r.get("turns", []))),
        ("ifeval_pt", "bench/ifeval_pt/prompts.jsonl", lambda r: " ".join(r.get("turns", []))),
        ("safety_heldout", "bench/safety_pt/heldout_v2.jsonl", lambda r: r.get("prompt", "")),
        ("overrefusal", "bench/overrefusal_pt/prompts_v2.jsonl", lambda r: r.get("prompt", "")),
    ]
    for rotulo, rel, extrai in locais:
        caminho = os.path.join(REPO, rel)
        if not os.path.exists(caminho):
            print(f"[decontam] AVISO: {caminho} nao existe, pulando {rotulo}")
            continue
        itens = []
        with open(caminho, encoding="utf-8") as fp:
            for i, linha in enumerate(fp):
                if not linha.strip():
                    continue
                r = json.loads(linha)
                itens.append((r.get("id", f"{rotulo}_{i}"), extrai(r)))
        bancos[rotulo] = itens
        print(f"[decontam] {rotulo}: {len(itens)} itens carregados de {rel}")

    return bancos


def main() -> int:
    ap = argparse.ArgumentParser(description="Decontaminacao: n-gramas dos benchmarks vs. corpus de treino")
    ap.add_argument("--train", required=True, help="jsonl do corpus de treino (instruction/input/output)")
    ap.add_argument("--out", default="docs/evaluation/decontamination-report")
    ap.add_argument("--n", type=int, default=N_GRAM, help=f"tamanho do n-grama (default {N_GRAM})")
    args = ap.parse_args()

    print("[decontam] carregando benchmarks de avaliacao...")
    bancos = carregar_benchmarks()
    if not bancos:
        print("[ERRO] nenhum benchmark carregado, abortando.")
        return 1

    # indice invertido: n-grama -> lista de (banco, id) que o contem
    indice = defaultdict(list)
    total_itens = 0
    for rotulo, itens in bancos.items():
        for item_id, texto in itens:
            total_itens += 1
            tokens = normaliza(texto)
            if len(tokens) < args.n:
                continue
            for ng in ngrams(tokens, args.n):
                indice[ng].append((rotulo, item_id))
    print(f"[decontam] {total_itens} itens de avaliacao, {len(indice)} n-gramas ({args.n}-gram) unicos indexados")

    print(f"[decontam] varrendo corpus de treino: {args.train}")
    achados = defaultdict(set)  # (banco, item_id) -> set of training line numbers
    n_treino = 0
    with open(args.train, encoding="utf-8") as fp:
        for linha_num, linha in enumerate(fp):
            if not linha.strip():
                continue
            n_treino += 1
            r = json.loads(linha)
            texto = " ".join(str(r.get(c, "")) for c in ("instruction", "input", "output"))
            tokens = normaliza(texto)
            if len(tokens) < args.n:
                continue
            vistos_nesta_linha = set()
            for ng in ngrams(tokens, args.n):
                for chave in indice.get(ng, ()):
                    if chave not in vistos_nesta_linha:
                        achados[chave].add(linha_num)
                        vistos_nesta_linha.add(chave)
            if n_treino % 50000 == 0:
                print(f"[decontam]   ... {n_treino} exemplos de treino varridos")

    print(f"[decontam] {n_treino} exemplos de treino varridos no total")
    print(f"[decontam] {len(achados)} itens de avaliacao com overlap de {args.n}-gram no treino")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    relatorio = {
        "n_gram": args.n,
        "n_itens_avaliacao": total_itens,
        "n_exemplos_treino": n_treino,
        "achados": {
            f"{banco}::{item_id}": sorted(linhas)
            for (banco, item_id), linhas in achados.items()
        },
        "resumo_por_banco": {
            rotulo: sum(1 for (b, _) in achados if b == rotulo)
            for rotulo in bancos
        },
    }
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)

    print()
    print("Resumo por benchmark (itens com overlap de %d-gram / total):" % args.n)
    for rotulo, itens in bancos.items():
        n_achados = relatorio["resumo_por_banco"].get(rotulo, 0)
        print(f"  {rotulo:<16} {n_achados}/{len(itens)}")
    print(f"\n[decontam] relatorio completo salvo em {args.out}.json")
    if achados:
        print("[decontam] AVISO: overlap encontrado -- audite manualmente antes de tirar conclusao")
        print("[decontam] (overlap de 13-gram pode ser falso positivo em texto de boilerplate curto,")
        print("[decontam]  ex.: cabecalhos de formulario ou frases muito comuns -- ler o texto real)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
