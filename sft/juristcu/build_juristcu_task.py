#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — Tarefa JurisTCU: julgamento de relevância jurídica PT-BR
=================================================================================
PT
--
`LeandroRibeiro/JurisTCU` (Fernandes et al., 2026, Language Resources and
Evaluation, doi:10.1007/s10579-025-09881-w) é um dataset de Recuperação de
Informação (IR) Jurídica em PT-BR: 16.045 documentos de jurisprudência do
Tribunal de Contas da União (TCU), 150 consultas (reais de log de busca +
sintéticas) e 2.250 julgamentos de relevância (score 0-3), com revisão
manual. Licença CC-BY-4.0 (confirmada na própria ficha do dataset).

POR QUE UMA TAREFA NOVA E NÃO O MOLDE DO `build_jaster.py`: assim como o
Quati (`sft/quati/build_quati_task.py`), o JurisTCU não é um dataset único
`load_dataset(hf_id, split)` — são 3 arquivos CSV relacionados por ID
(`doc.csv`, `query.csv`, `qrel.csv`) que precisam ser JUNTADOS. Este script
segue exatamente o mesmo padrão de conversão do Quati: julgamento de
relevância (não QA extrativo, que exigiria assumir que toda consulta é uma
pergunta respondível pelo documento — o dado não sustenta essa suposição).

POR QUE ISSO PREENCHE UMA LACUNA REAL: é conteúdo jurídico/institucional
brasileiro (jurisprudência do TCU) com CONSULTAS REAIS de usuários (não só
documento cru como o LegalPT) — ensina o modelo a julgar relevância em
domínio jurídico nativo, complementando o LegalPT (que é só corpus, sem
consulta) e o Quati (que é domínio geral, não jurídico).

COMO OS ARQUIVOS SÃO OBTIDOS: via `huggingface_hub.hf_hub_download` (não
`datasets.load_dataset`), pois o repositório expõe os 3 arquivos como CSV
soltos, não como configs de dataset padronizadas.

CONVERSÃO -> INSTRUÇÃO (idêntica em espírito ao Quati):
    instruction = "Dada a consulta de busca jurídica e o trecho de
                    jurisprudência do TCU abaixo, o trecho é relevante
                    para a consulta? Responda apenas Sim ou Não."
    input       = "Consulta: {query}\n\nTrecho: {ENUNCIADO + EXCERTO}"
    output      = "Sim" se score >= --relevance_threshold, senão "Não"
O score dos qrels aqui É documentado (0=irrelevante, 3=altamente relevante,
ver ficha do dataset) — mais confiável que o do Quati; default do
threshold em 2 (score >= 2 é "relevante" na escala 0-3 documentada).

EN
--
Joins JurisTCU's 3 CSV files (docs + queries + graded qrels, CC-BY-4.0)
into a binary relevance-judgment instruction task, same pattern as
sft/quati/build_quati_task.py. Legal/institutional Brazilian content with
REAL user queries (unlike LegalPT, which is corpus-only, no queries).

Uso:
    python sft/juristcu/build_juristcu_task.py
    python sft/juristcu/build_juristcu_task.py --relevance_threshold 1

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys

HF_REPO_ID = "LeandroRibeiro/JurisTCU"
LICENSE    = "cc-by-4.0"
REFERENCE  = ("Fernandes et al. (2026). JurisTCU. Language Resources and Evaluation. "
              "doi:10.1007/s10579-025-09881-w.")

DEFAULT_RELEVANCE_THRESHOLD = 2.0  # escala 0-3 documentada (0=irrelevante, 3=altamente relevante)

INSTRUCTION = ("Dada a consulta de busca jurídica e o trecho de jurisprudência do TCU "
               "abaixo, o trecho é relevante para a consulta? Responda apenas Sim ou Não.")


def s(x) -> str:
    return ("" if x is None else str(x)).strip()


def _download_csv_rows(hf_hub_download, repo_id: str, filename: str) -> list[dict]:
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    with open(path, encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def main() -> int:
    ap = argparse.ArgumentParser(description="Constrói a tarefa de relevância JurisTCU (Alpaca)")
    ap.add_argument("--out", default="data/sft/juristcu", help="diretorio de saida")
    ap.add_argument("--relevance_threshold", type=float, default=DEFAULT_RELEVANCE_THRESHOLD,
                    help=f"score >= este valor -> 'Sim' (default {DEFAULT_RELEVANCE_THRESHOLD}, escala 0-3)")
    ap.add_argument("--max_chars", type=int, default=3000,
                    help="descarta exemplos cujo prompt passa deste nº de caracteres (0 = off)")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        print(f"[ERRO] 'huggingface_hub' ausente ({e}). Rode na imagem manaca-posttrain.")
        return 1

    print(f"[juristcu] baixando doc.csv/query.csv/qrel.csv de {HF_REPO_ID} ...")
    try:
        docs = _download_csv_rows(hf_hub_download, HF_REPO_ID, "doc.csv")
        queries = _download_csv_rows(hf_hub_download, HF_REPO_ID, "query.csv")
        qrels = _download_csv_rows(hf_hub_download, HF_REPO_ID, "qrel.csv")
    except Exception as e:
        print(f"[ERRO] falha ao baixar os CSVs de {HF_REPO_ID}: {e}")
        return 1
    print(f"[juristcu] {len(docs):,} documentos | {len(queries):,} consultas | {len(qrels):,} julgamentos brutos")

    if not (docs and queries and qrels):
        print("[ERRO] um dos 3 arquivos veio vazio.")
        return 1

    doc_key_col = next((c for c in ["KEY", "key", "doc_id", "DOC_ID"] if c in docs[0]), None)
    enun_col = next((c for c in ["ENUNCIADO", "enunciado"] if c in docs[0]), None)
    exc_col = next((c for c in ["EXCERTO", "excerto"] if c in docs[0]), None)
    if not doc_key_col or not (enun_col or exc_col):
        print(f"[ERRO] colunas de documento nao reconhecidas: {list(docs[0].keys())}")
        return 1
    doc_text = {}
    for row in docs:
        key = s(row.get(doc_key_col))
        if not key:
            continue
        parts = [s(row.get(enun_col)), s(row.get(exc_col))]
        text = "\n\n".join(p for p in parts if p)
        if text:
            doc_text[key] = text
    print(f"[juristcu] {len(doc_text):,} documentos com texto valido")

    q_id_col = next((c for c in ["ID", "id", "query_id"] if c in queries[0]), None)
    q_txt_col = next((c for c in ["TEXT", "text", "query"] if c in queries[0]), None)
    if not q_id_col or not q_txt_col:
        print(f"[ERRO] colunas de consulta nao reconhecidas: {list(queries[0].keys())}")
        return 1
    query_text = {s(row.get(q_id_col)): s(row.get(q_txt_col)) for row in queries if row.get(q_id_col)}

    qr_qcol = next((c for c in ["QUERY_ID", "query_id"] if c in qrels[0]), None)
    qr_dcol = next((c for c in ["DOC_ID", "doc_id"] if c in qrels[0]), None)
    qr_scol = next((c for c in ["SCORE", "score", "relevance"] if c in qrels[0]), None)
    if not (qr_qcol and qr_dcol and qr_scol):
        print(f"[ERRO] colunas de qrel nao reconhecidas: {list(qrels[0].keys())}")
        return 1

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "juristcu.jsonl")
    n, n_long, n_sim, n_nao, n_missing = 0, 0, 0, 0, 0
    scores_seen: list[float] = []
    with open(out_path, "w", encoding="utf-8") as fp:
        for row in qrels:
            qid, did, score = s(row.get(qr_qcol)), s(row.get(qr_dcol)), row.get(qr_scol)
            query = query_text.get(qid)
            passage = doc_text.get(did)
            if not query or not passage or score in (None, ""):
                n_missing += 1
                continue
            try:
                score = float(score)
            except (TypeError, ValueError):
                n_missing += 1
                continue
            scores_seen.append(score)
            label = "Sim" if score >= args.relevance_threshold else "Não"
            input_text = f"Consulta: {query}\n\nTrecho: {passage}"
            if args.max_chars and (len(INSTRUCTION) + len(input_text)) > args.max_chars:
                n_long += 1
                continue
            fp.write(json.dumps({"instruction": INSTRUCTION, "input": input_text, "output": label,
                                 "task": "juristcu_relevance"}, ensure_ascii=False) + "\n")
            n += 1
            n_sim += label == "Sim"
            n_nao += label == "Não"

    if scores_seen:
        s_sorted = sorted(scores_seen)
        print(f"[juristcu] distribuicao de score: min={s_sorted[0]} p50={s_sorted[len(s_sorted)//2]} max={s_sorted[-1]}")

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
        "dataset": "juristcu_relevance", "hf_repo": HF_REPO_ID, "license": LICENSE, "reference": REFERENCE,
        "total": n, "sim": n_sim, "nao": n_nao, "skipped_too_long": n_long, "skipped_missing_join": n_missing,
        "relevance_threshold": args.relevance_threshold, "shuffled": args.shuffle, "seed": args.seed,
        "sha256": sha256,
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    print(f"[juristcu] TOTAL: {n:,} exemplos ({n_sim:,} Sim / {n_nao:,} Não, {n_long} longos, "
          f"{n_missing} sem join) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
