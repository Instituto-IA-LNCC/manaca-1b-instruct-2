#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — Tarefa Quati: julgamento de relevância PT-BR nativo
=============================================================================
PT
--
`unicamp-dl/quati` (Nascimento et al., 2024, arXiv:2404.06976) é o primeiro
dataset de Recuperação de Informação (IR) em PT-BR com passagens E consultas
100% escritas por falantes nativos (a maioria dos benchmarks de IR em PT é
traduzida do inglês/MS MARCO). Licença CC BY 4.0.

POR QUE UMA TAREFA NOVA E NÃO O MESMO MOLDE DO `build_jaster.py`: o Quati
não é um dataset único carregável com `load_dataset(hf_id, config, split)` —
é 3 configs relacionadas por ID que precisam ser JUNTADAS:
  - `quati_10M_qrels`     : (query_id, passage_id, score) — julgamento de
    relevância, anotado semi-automaticamente por um LLM juiz.
  - `quati_all_topics`    : (query_id, query) — as consultas.
  - `quati_10M_passages`  : (passage_id, passage) — as passagens (10M, em
    5 partes de 2M).
O `TASKS` registry do `build_jaster.py` assume um dataset por tarefa; juntar
3 exige um loader especial, por isso este é um script próprio (mesmo padrão
de saída, consumido via `--extra`, igual ao gigaverbo_v2_sft/jaster).

COMO A JUNÇÃO É FEITA (streaming, sem carregar 10M passagens na RAM):
  1. Carrega `quati_all_topics` inteiro em memória (milhares de consultas —
     pequeno). 2. Faz streaming de `quati_10M_qrels` até acumular
     `--max_examples` triplas válidas (query_id presente nos topics).
  3. Faz UMA passada de streaming por `quati_10M_passages` (as 5 partes),
     retendo só as passagens cujo `passage_id` está no conjunto necessário
     do passo 2. Isso evita carregar as 10M passagens, mas ainda exige
     escanear o dataset inteiro uma vez (custo de rede, não de RAM).

CONVERSÃO -> INSTRUÇÃO: julgamento binário de relevância (mesmo estilo dos
labels de classificação do manacá-jaster, ex.: `conv_hatebr`/`conv_assin2_nli`):
    instruction = "Dada a consulta de busca e a passagem abaixo, a passagem
                    é relevante para a consulta? Responda apenas Sim ou Não."
    input       = "Consulta: {query}\n\nPassagem: {passage}"
    output      = "Sim" se score >= --relevance_threshold, senão "Não"
NÃO geramos QA extrativo (responder a pergunta usando a passagem) porque
uma consulta de busca não é necessariamente uma pergunta respondível pelo
texto — julgamento de relevância é o que os dados realmente suportam sem
inventar suposição.

A escala do `score` dos qrels não está documentada publicamente com
precisão; o script REPORTA a distribuição observada no manifest para
calibrar `--relevance_threshold` (default 1) — mesma prática de auditoria
usada no filtro edu_score do ClassiCC-PT.

EN
--
Joins Quati's 3 HF configs (qrels + topics + passages, CC BY 4.0) into a
binary relevance-judgment instruction task — the only conversion the data
genuinely supports without inventing a QA-answerability assumption. One
streaming pass over the 10M passages is unavoidable (no random access via
HF streaming); topics are small enough to load fully in memory first.

Uso:
    python sft/quati/build_quati_task.py
    python sft/quati/build_quati_task.py --max_examples 20000 --relevance_threshold 2

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

HF_REPO_ID       = "unicamp-dl/quati"
CFG_TOPICS       = "quati_all_topics"
CFG_QRELS        = "quati_10M_qrels"
CFG_PASSAGES     = "quati_10M_passages"
LICENSE          = "cc-by-4.0"
REFERENCE        = "Nascimento et al. (2024). Quati. arXiv:2404.06976."

DEFAULT_MAX_EXAMPLES = 20_000
INSTRUCTION = ("Dada a consulta de busca e a passagem abaixo, a passagem é relevante "
               "para a consulta? Responda apenas Sim ou Não.")


def s(x) -> str:
    return ("" if x is None else str(x)).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Constrói a tarefa de relevância Quati (Alpaca)")
    ap.add_argument("--out", default="data/sft/quati", help="diretório de saída")
    ap.add_argument("--max_examples", type=int, default=DEFAULT_MAX_EXAMPLES,
                    help=f"teto de exemplos (qrels válidos) a processar (default {DEFAULT_MAX_EXAMPLES:,})")
    ap.add_argument("--relevance_threshold", type=float, default=1.0,
                    help="score >= este valor -> 'Sim' (default 1.0; calibrar pela distribuição reportada)")
    ap.add_argument("--max_chars", type=int, default=3000,
                    help="descarta exemplos cujo prompt passa deste nº de caracteres (0 = off)")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from datasets import get_dataset_split_names, load_dataset
    except ImportError as e:
        print(f"[ERRO] 'datasets' ausente ({e}). Rode na imagem manaca-posttrain.")
        return 1

    # ── 1. Topics (pequeno, carrega inteiro) ──────────────────────────────
    print(f"[quati] carregando topics ({CFG_TOPICS})...")
    try:
        # split == nome do config (script custom do HF nao usa "train" --
        # confirmado ao vivo: "Unknown split 'train'. Should be one of
        # ['quati_all_topics']").
        topics_ds = load_dataset(HF_REPO_ID, CFG_TOPICS, split=CFG_TOPICS, trust_remote_code=True)
    except Exception as e:
        print(f"[ERRO] falha ao carregar topics: {e}")
        return 1
    tf = next((c for c in ["query", "text"] if c in topics_ds.column_names), None)
    tid = next((c for c in ["query_id", "id"] if c in topics_ds.column_names), None)
    if not tf or not tid:
        print(f"[ERRO] colunas de topics nao reconhecidas: {topics_ds.column_names}")
        return 1
    topics = {s(row[tid]): s(row[tf]) for row in topics_ds if row.get(tf) and row.get(tid) is not None}
    print(f"[quati] {len(topics):,} consultas carregadas")

    # ── 2. Qrels (streaming, ate acumular max_examples validos) ───────────
    print(f"[quati] carregando qrels ({CFG_QRELS}, streaming)...")
    qrels_ds = load_dataset(HF_REPO_ID, CFG_QRELS, split=CFG_QRELS, streaming=True, trust_remote_code=True)
    first = next(iter(qrels_ds))
    qcol = next((c for c in ["query_id"] if c in first), None)
    pcol = next((c for c in ["passage_id"] if c in first), None)
    scol = next((c for c in ["score", "relevance", "label"] if c in first), None)
    if not (qcol and pcol and scol):
        print(f"[ERRO] colunas de qrels nao reconhecidas: {list(first.keys())}")
        return 1

    needed: dict[str, list[tuple]] = {}  # passage_id -> [(query_id, score), ...]
    scores_seen: list[float] = []
    n_qrels = 0
    for row in qrels_ds:
        qid, pid, score = s(row.get(qcol)), s(row.get(pcol)), row.get(scol)
        if qid not in topics or not pid or score is None:
            continue
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        needed.setdefault(pid, []).append((qid, score))
        scores_seen.append(score)
        n_qrels += 1
        if args.max_examples and n_qrels >= args.max_examples:
            break
    print(f"[quati] {n_qrels:,} qrels validos coletados | {len(needed):,} passagens unicas a buscar")
    if scores_seen:
        s_sorted = sorted(scores_seen)
        print(f"[quati] distribuicao de score: min={s_sorted[0]} p50={s_sorted[len(s_sorted)//2]} max={s_sorted[-1]}")

    if not needed:
        print("[quati] AVISO: nenhum qrel valido encontrado. Abortando.")
        return 1

    # ── 3. Uma passada nas passagens, retendo so as necessarias ───────────
    print(f"[quati] escaneando passagens ({CFG_PASSAGES}, streaming, 1 passada)...")
    # CFG_PASSAGES (10M linhas) vem fatiado em sub-splits "_part_NN" -- o
    # config nao tem um split unico com esse nome (confirmado ao vivo:
    # "Bad split: quati_10M_passages. Available splits:
    # ['quati_10M_passages_part_00', ..., '_part_04']"). Descobre os
    # sub-splits reais (em vez de fixar o numero de partes no codigo) e
    # encadeia os IterableDataset de cada um -- evita depender da sintaxe
    # "split1+split2" do datasets em modo streaming, que nao e garantida
    # pela versao usada neste projeto.
    all_splits = get_dataset_split_names(HF_REPO_ID, CFG_PASSAGES, trust_remote_code=True)
    parts = sorted(sp for sp in all_splits if sp.startswith(f"{CFG_PASSAGES}_part_"))
    if not parts:
        parts = [CFG_PASSAGES]
    print(f"[quati] passagens: {len(parts)} parte(s) -- {parts}")

    class ChainedSplits:
        """Encadeia varios splits streaming de um mesmo config. Suporta
        __iter__ multiplas vezes (reabre do primeiro split a cada chamada),
        igual a uma datasets.IterableDataset -- necessario porque o resto
        do script faz next(iter(dataset)) (deteccao de coluna) e depois um
        `for row in dataset` separado; um itertools.chain(...) puro e um
        generator de passagem unica e perderia a 1a linha nesse padrao.
        """
        def __init__(self, hf_id, cfg, parts):
            self.hf_id, self.cfg, self.parts = hf_id, cfg, parts

        def __iter__(self):
            for p in self.parts:
                yield from load_dataset(self.hf_id, self.cfg, split=p, streaming=True, trust_remote_code=True)

    passages_ds = ChainedSplits(HF_REPO_ID, CFG_PASSAGES, parts)
    pfirst = next(iter(passages_ds))
    ppid = next((c for c in ["passage_id"] if c in pfirst), None)
    ptxt = next((c for c in ["passage", "text"] if c in pfirst), None)
    if not (ppid and ptxt):
        print(f"[ERRO] colunas de passagens nao reconhecidas: {list(pfirst.keys())}")
        return 1

    found_passages: dict[str, str] = {}
    remaining = set(needed)
    for row in passages_ds:
        if not remaining:
            break
        pid = s(row.get(ppid))
        if pid in remaining:
            found_passages[pid] = s(row.get(ptxt))
            remaining.discard(pid)
    print(f"[quati] {len(found_passages):,}/{len(needed):,} passagens encontradas "
          f"({len(remaining):,} nao localizadas — dataset pode ter IDs orfaos)")

    # ── 4. Monta os exemplos ───────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "quati.jsonl")
    n, n_long, n_sim, n_nao = 0, 0, 0, 0
    with open(out_path, "w", encoding="utf-8") as fp:
        for pid, pairs in needed.items():
            passage = found_passages.get(pid)
            if not passage:
                continue
            for qid, score in pairs:
                query = topics.get(qid)
                if not query:
                    continue
                label = "Sim" if score >= args.relevance_threshold else "Não"
                input_text = f"Consulta: {query}\n\nPassagem: {passage}"
                if args.max_chars and (len(INSTRUCTION) + len(input_text)) > args.max_chars:
                    n_long += 1
                    continue
                row = {"instruction": INSTRUCTION, "input": input_text, "output": label, "task": "quati_relevance"}
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                n_sim += label == "Sim"
                n_nao += label == "Não"

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
        "dataset": "quati_relevance", "hf_repo": HF_REPO_ID, "license": LICENSE, "reference": REFERENCE,
        "total": n, "sim": n_sim, "nao": n_nao, "skipped_too_long": n_long,
        "qrels_collected": n_qrels, "passages_found": len(found_passages),
        "passages_missing": len(remaining), "relevance_threshold": args.relevance_threshold,
        "score_distribution_note": "min/p50/max impressos no log — recalibrar threshold se necessario",
        "shuffled": args.shuffle, "seed": args.seed, "sha256": sha256,
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    print(f"[quati] TOTAL: {n:,} exemplos ({n_sim:,} Sim / {n_nao:,} Não, {n_long} descartados por tamanho) "
          f"-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
