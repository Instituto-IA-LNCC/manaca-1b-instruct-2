#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — Script 03: Aquisição do Carolina Corpus v1.2 (registro/estilo)
========================================================================================
LNCC × NII/LLM-jp | Corpus-fonte para instrução sintética (Fase "instruct-2")

Baixa `carolina-c4ai/corpus-carolina` (revisão v1.2, split "corpus") e grava
em Apache Parquet + Zstandard, particionado em shards de 50.000 documentos.
Idempotente.

Papel neste repositório:
    Diversidade de REGISTRO e GÊNERO textual (literário, jornalístico,
    jurídico, transcrição oral, redes sociais) com proveniência e tipologia
    documentadas por texto — algo raro em corpora abertos. Mira diretamente
    as categorias mais fracas do MT-Bench-PT do instruct v1/v2 (escrita,
    interpretação/roleplay — ver docs/evaluation/instruct-eval-pt.md),
    hoje dominadas por instrução traduzida (Alpaca-PT) de registro uniforme.

Justificativa científica:
    Carolina (Finger et al., 2022; arXiv:2303.16098) é um corpus geral do
    português brasileiro contemporâneo (1970-presente) com proveniência,
    tipologia e versionamento por documento — metodologia WaC-WiPT.
    HF: https://huggingface.co/datasets/carolina-c4ai/corpus-carolina

ADVERTÊNCIA DE LICENÇA — igual ao tratamento de Alpaca-PT/XLSum no v1
(ver docs/data/README.md): Carolina é CC BY-NC-SA 4.0. Isso IMPEDE uso
comercial direto. Mantém-se a mesma política já adotada pelo projeto:
documentar a restrição em vez de excluir a fonte por precaução — decida
na mistura final se comercial é um requisito.

VERSÃO: o pedido original especificou v1.2; a mais recente no HF é v2.0.1
(mais texto, mesma metodologia). Este script fixa v1.2 por padrão
(--revision para trocar) — mudar de versão é uma linha, não um redesenho.

DIFERENÇA CRÍTICA DE DEPENDÊNCIA (ler antes de rodar):
    O HF `datasets` >= 4.0 REMOVEU suporte a datasets com loading script
    custom, e o Carolina ainda depende de um (`corpus-carolina.py`) para
    resolver a revisão/taxonomia. Por isso requirements/corpus.txt NÃO
    fixa `datasets` — ele reaproveita o `datasets==3.1.0` já pinado em
    requirements/posttrain.txt, que ainda suporta `trust_remote_code=True`.
    Se você isolar este script num venv com `datasets>=4.0`, ele falha com
    RuntimeError explicando exatamente isso — não é um bug deste script.

PRÉ-REQUISITOS:
    1. WORK_DIR gravável (container posttrain: /workspace/manaca-corpus)
    2. `pip install -r requirements/posttrain.txt -r requirements/corpus.txt`
       (nessa ordem — ver nota de versão acima)

Uso:
    python corpus/scripts/03_acquire_carolina.py
    python corpus/scripts/03_acquire_carolina.py --max-docs 200000
    python corpus/scripts/03_acquire_carolina.py --revision v2.0.1   # versao mais recente

Saída:
    $WORK_DIR/raw/carolina/
    ├── shard_000000.parquet
    ├── ...
    ├── download.log
    └── taxonomy_distribution.json   contagem de docs por tipologia/genero

Autor: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
Versão: 0.1.0 — Setembro 2026
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset
    from loguru import logger
    from tqdm import tqdm
except ImportError as e:
    print(f"[ERRO] Dependência ausente: {e}")
    print("Execute: pip install -r requirements/posttrain.txt -r requirements/corpus.txt")
    sys.exit(1)


# ── Configuração ──────────────────────────────────────────────────────────────

SOURCE_NAME    = "carolina"
HF_REPO_ID     = "carolina-c4ai/corpus-carolina"
HF_SPLIT       = "corpus"
DEFAULT_REVISION = "v1.2"
LICENSE        = "cc-by-nc-sa-4.0"   # NAO comercial — ver advertencia no docstring
REFERENCE      = "Finger et al. (2022). Carolina Corpus. arXiv:2303.16098."

WORK_DIR      = Path(os.environ.get("WORK_DIR", Path.home() / "manaca-corpus"))
OUTPUT_DIR    = WORK_DIR / "raw" / SOURCE_NAME
LOG_DIR       = WORK_DIR / "logs"
CKPT_DIR      = WORK_DIR / "checkpoints"
MANIFEST_PATH = CKPT_DIR / "acquisition_manifest.json"

SHARD_SIZE   = 50_000
MIN_TEXT_LEN = 50

DEFAULT_MAX_DOCS = 150_000  # corpus modesto (~2M textos); teto conservador
                             # ja garante boa diversidade de genero p/ instrucao

SCHEMA = pa.schema([
    pa.field("text",   pa.string()),
    pa.field("source", pa.string()),
    pa.field("id",     pa.string()),
    pa.field("lang",   pa.string()),
    pa.field("score",  pa.float32()),
])

# Campos candidatos onde o Carolina costuma expor tipologia/genero por doc
# (nomenclatura varia por versao do loading script — deteccao tolerante).
TAXONOMY_FIELD_CANDIDATES = ["taxonomy", "genre", "domain", "sourcename", "typology"]


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr, level="INFO", colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
    )
    logger.add(
        log_file, level="DEBUG", rotation="100 MB", retention=3, encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}",
    )
    logger.info(f"Log: {log_file}")


# ── Manifesto ─────────────────────────────────────────────────────────────────

def load_manifest() -> dict[str, Any]:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        "version": "0.1.0", "project": "Manaca-1B-Instruct-2 — LNCC x NII/LLM-jp",
        "created_at": datetime.now(timezone.utc).isoformat(), "sources": {},
    }


def save_manifest(manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(MANIFEST_PATH)


# ── Retomada automática ───────────────────────────────────────────────────────

def get_existing_shards(output_dir: Path) -> set[int]:
    existing = set()
    for f in sorted(output_dir.glob("shard_*.parquet")):
        sid = f.stem.replace("shard_", "")
        if not sid.isdigit():
            continue
        try:
            meta = pq.read_metadata(f)
            if meta.num_rows > 0:
                existing.add(int(sid))
            else:
                logger.warning(f"Shard vazio (sera re-escrito): {f.name}")
        except Exception:
            logger.warning(f"Shard corrompido (sera re-escrito): {f.name}")
    return existing


# ── Filtros de qualidade ──────────────────────────────────────────────────────

def passes_filter(text: str) -> bool:
    if not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < MIN_TEXT_LEN:
        return False
    n_alpha = sum(1 for c in text if c.isalpha())
    return n_alpha / max(len(text), 1) >= 0.50


def quality_score(text: str) -> float:
    """Score genérico [0.0-1.0] — fórmula idêntica à suite manaca/corpus."""
    import math
    if not text:
        return 0.0
    words  = text.split()
    n      = max(len(words), 1)
    alpha  = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    ttr    = len(set(w.lower() for w in words)) / n
    lscore = min(math.log1p(len(text)) / 14.0, 1.0)
    return round(0.4 * alpha + 0.3 * ttr + 0.3 * lscore, 4)


# ── Escrita de shard ──────────────────────────────────────────────────────────

def write_shard(docs: list[dict], shard_id: int, output_dir: Path) -> dict[str, Any]:
    path = output_dir / f"shard_{shard_id:06d}.parquet"
    table = pa.table(
        {
            "text": [d["text"] for d in docs], "source": [d["source"] for d in docs],
            "id": [d["id"] for d in docs], "lang": [d["lang"] for d in docs],
            "score": [d["score"] for d in docs],
        },
        schema=SCHEMA,
    )
    pq.write_table(table, path, compression="zstd")
    size_bytes = path.stat().st_size
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read(1 << 20)).hexdigest()
    return {
        "shard_id": shard_id, "path": str(path), "n_docs": len(docs),
        "size_bytes": size_bytes, "size_mb": round(size_bytes / 1e6, 1),
        "sha256_1mb": sha, "written_at": datetime.now(timezone.utc).isoformat(),
    }


def _load_carolina(revision: str):
    """
    Carrega o Carolina com `trust_remote_code=True` (loading script custom).
    Isola o erro conhecido (datasets>=4.0 removeu suporte a scripts) numa
    mensagem acionável em vez de um traceback cru.
    """
    try:
        return load_dataset(HF_REPO_ID, split=HF_SPLIT, revision=revision,
                             trust_remote_code=True, streaming=True)
    except Exception as e:
        msg = str(e)
        if "scripts are no longer supported" in msg.lower() or "trust_remote_code" in msg.lower():
            logger.error(
                "Falha esperada: sua versao instalada de `datasets` removeu suporte "
                "a loading scripts (Carolina ainda depende de um). Solucao: use o "
                "mesmo ambiente do posttrain (`datasets==3.1.0`, requirements/posttrain.txt) "
                "— NAO instale `datasets` mais novo neste venv. Ver docstring deste script."
            )
        raise


# ── Pipeline principal ────────────────────────────────────────────────────────

def acquire(max_docs: int, revision: str) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    setup_logging(OUTPUT_DIR / "download.log")
    manifest = load_manifest()

    existing     = get_existing_shards(OUTPUT_DIR)
    docs_to_skip = len(existing) * SHARD_SIZE if existing else 0
    next_shard   = (max(existing) + 1) if existing else 0

    logger.info("=" * 60)
    logger.info(f"Fonte:        {SOURCE_NAME}")
    logger.info(f"Repositorio:  {HF_REPO_ID}  (revision={revision}, split={HF_SPLIT})")
    logger.info(f"Teto de docs: {'sem limite' if max_docs <= 0 else f'{max_docs:,}'}")
    logger.info(f"Licenca:      {LICENSE}  — NAO comercial, ver docstring")
    logger.info(f"Referencia:   {REFERENCE}")
    logger.info(f"Output:       {OUTPUT_DIR}")
    if docs_to_skip:
        logger.info(f"Retomando: {len(existing)} shards existentes (~{docs_to_skip:,} docs ja salvos)")
    logger.info("=" * 60)

    logger.info(f"Conectando ao HuggingFace: {HF_REPO_ID} (trust_remote_code=True) ...")
    dataset = _load_carolina(revision)

    first = next(iter(dataset))
    text_col = next((c for c in ["text", "content", "body"] if c in first), list(first.keys())[0])
    id_col   = next((c for c in ["id", "doc_id"] if c in first), None)
    tax_col  = next((c for c in TAXONOMY_FIELD_CANDIDATES if c in first), None)
    logger.info(f"Coluna texto: '{text_col}' | ID: '{id_col or 'gerado'}' | Taxonomia: '{tax_col or 'nao encontrada'}'")
    logger.info(f"Campos disponiveis: {list(first.keys())}")

    buffer: list[dict] = []
    shard_id, total_seen, kept, skipped = next_shard, 0, 0, 0
    total_chars = 0
    taxonomy_counts: dict[str, int] = {}
    t_start = time.time()

    for item in tqdm(dataset, desc="Carolina", unit="doc", mininterval=15, file=sys.stderr):
        total_seen += 1
        if total_seen <= docs_to_skip:
            continue
        if max_docs > 0 and kept >= max_docs:
            logger.info(f"Teto de {max_docs:,} docs atingido — parando.")
            break

        text = str(item.get(text_col, "")).strip()
        if not passes_filter(text):
            skipped += 1
            continue

        taxonomy = str(item.get(tax_col, "geral")) if tax_col else "geral"
        taxonomy_counts[taxonomy] = taxonomy_counts.get(taxonomy, 0) + 1

        raw_id = item.get(id_col, "") if id_col else ""
        doc_id = str(raw_id) if raw_id else hashlib.md5(f"{SOURCE_NAME}_{total_seen}".encode()).hexdigest()[:16]

        buffer.append({
            "text": text, "source": f"{SOURCE_NAME}_{taxonomy}", "id": doc_id,
            "lang": "por_Latn_BR", "score": quality_score(text),
        })
        kept += 1
        total_chars += len(text)

        if len(buffer) >= SHARD_SIZE:
            meta = write_shard(buffer, shard_id, OUTPUT_DIR)
            elapsed = time.time() - t_start
            logger.info(
                f"Shard {shard_id:04d} | {kept:>7,} docs | {meta['size_mb']:5.0f} MB | "
                f"{kept / max(elapsed, 1):,.0f} docs/s"
            )
            manifest["sources"][SOURCE_NAME] = {
                "status": "in_progress", "hf_repo": HF_REPO_ID, "revision": revision,
                "license": LICENSE, "total_seen": total_seen, "total_kept": kept,
                "total_skip": skipped, "shards_written": shard_id + 1,
                "estimated_tokens_b": round(total_chars / 4 / 1e9, 3),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            save_manifest(manifest)
            buffer, shard_id = [], shard_id + 1

    if buffer:
        meta = write_shard(buffer, shard_id, OUTPUT_DIR)
        logger.info(f"Shard final {shard_id:04d} | {len(buffer):,} docs | {meta['size_mb']:.0f} MB")

    dist_path = OUTPUT_DIR / "taxonomy_distribution.json"
    dist_path.write_text(json.dumps({
        "source": SOURCE_NAME, "revision": revision,
        "distribution": dict(sorted(taxonomy_counts.items(), key=lambda x: x[1], reverse=True)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - t_start
    total_bytes = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.parquet"))
    est_tokens = total_chars // 4

    final: dict[str, Any] = {
        "status": "completed" if not (max_docs > 0 and kept >= max_docs) else "capped_by_max_docs",
        "hf_repo": HF_REPO_ID, "revision": revision, "license": LICENSE, "reference": REFERENCE,
        "total_seen": total_seen, "total_kept": kept, "total_skipped": skipped,
        "retention_rate": round(kept / max(total_seen, 1), 4),
        "shards_written": shard_id + 1, "total_bytes": total_bytes,
        "total_gb": round(total_bytes / 1e9, 3), "estimated_tokens": est_tokens,
        "estimated_tokens_b": round(est_tokens / 1e9, 3), "elapsed_hours": round(elapsed / 3600, 2),
        "taxonomy_distribution": taxonomy_counts, "output_dir": str(OUTPUT_DIR),
        "taxonomy_dist_path": str(dist_path), "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest["sources"][SOURCE_NAME] = final
    save_manifest(manifest)

    logger.info("=" * 60)
    logger.info(f"CONCLUIDO — {SOURCE_NAME}")
    logger.info(f"  Docs mantidos:  {kept:,}  |  Descartados: {skipped:,}")
    logger.info(f"  Taxa retencao:  {final['retention_rate']:.1%}")
    logger.info(f"  Tamanho total:  {total_bytes / 1e9:.3f} GB  |  Tokens est.: {est_tokens / 1e9:.3f}B")
    logger.info(f"  Tipologias:     {len(taxonomy_counts)}")
    logger.info(f"  Output:         {OUTPUT_DIR}")
    logger.info("=" * 60)
    logger.info("Fontes de instrucao-2 (LegalPT + ClassiCC-PT + Carolina) concluidas.")
    logger.info("Proximo passo: pipeline de geracao de instrucao sintetica (sft/ deste repo).")
    return final


def verify() -> None:
    shards = sorted(OUTPUT_DIR.glob("shard_*.parquet"))
    if not shards:
        logger.warning("Nenhum shard encontrado para verificacao.")
        return
    total_rows, errors = 0, 0
    for s in shards:
        try:
            rows = pq.read_metadata(s).num_rows
            total_rows += rows
            if rows == 0:
                errors += 1
                logger.warning(f"Shard vazio: {s.name}")
        except Exception as e:
            errors += 1
            logger.error(f"Shard corrompido {s.name}: {e}")
    logger.info(f"Verificacao: {len(shards)} shards | {total_rows:,} docs | {errors} erro(s)")

    dist_path = OUTPUT_DIR / "taxonomy_distribution.json"
    if dist_path.exists():
        dist = json.loads(dist_path.read_text(encoding="utf-8"))
        for tax, n in list(dist.get("distribution", {}).items())[:10]:
            logger.info(f"  {tax:<20} {n:>8,} docs")

    try:
        t = pq.read_table(shards[-1], columns=["text", "source", "score"])
        logger.info(f'Amostra ({shards[-1].name}) [{t["source"][0].as_py()}]: "{t["text"][0].as_py()[:80]}..." [score={t["score"][0].as_py():.2f}]')
    except Exception as e:
        logger.warning(f"Erro ao ler amostra: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Aquisicao do Carolina Corpus")
    ap.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS,
                     help=f"teto de documentos mantidos (default {DEFAULT_MAX_DOCS:,}; 0 = sem limite)")
    ap.add_argument("--revision", type=str, default=DEFAULT_REVISION,
                     help=f"revisao do dataset no HF (default '{DEFAULT_REVISION}')")
    args = ap.parse_args()
    try:
        acquire(max_docs=args.max_docs, revision=args.revision)
        verify()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuario. Re-executar para retomar do ultimo shard.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
