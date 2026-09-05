#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — Script 04: Aquisição do MADLAD-400 (pt)
=================================================================
LNCC × NII/LLM-jp | Corpus-fonte para instrução sintética (Fase "instruct-2")

Baixa `allenai/MADLAD-400` (config `pt`) e grava em Apache Parquet + Zstandard,
particionado em shards de 50.000 documentos. Idempotente.

POR QUE ESTA FONTE ESTAVA DESCARTADA E AGORA NÃO ESTÁ:
    O MADLAD-400 fazia parte do Tier 1 do PLANO de pré-treino do `manaca`
    (script `corpus/scripts/02_acquire_madlad400.py`, idêntico em estrutura a
    este). Uma auditoria contra o repositório real do base
    (`manaca-1b-base`) confirmou que o corpus final de pré-treino usou
    SOMENTE GigaVerbo (subamostra) + Ulysses Tesemõ + Wikipedia-pt (~20,1B
    tokens) — MADLAD-400 nunca foi de fato consumido, apesar do script
    existir. Ou seja: está livre de sobreposição real com o base, mesmo
    tendo sido "planejado". Ver corpus/README.md deste repo.

Justificativa científica:
    Kudugunta et al. (2024, arXiv:2309.04662) cobre 419 línguas com pipeline
    rigoroso de deduplicação. A partição `pt` complementa o GigaVerbo com
    domínios e períodos históricos distintos (2013-2023).

DIFERENÇA vs. o script homônimo do `manaca`: aquele aplica um filtro de
variante PT-BR vs PT-PT via fastText (`lid.176.bin`) porque monta um corpus
de PRÉ-TREINO de bilhões de tokens, onde pureza de variante importa em
escala. Aqui a saída alimenta uma amostra pequena para geração de instrução
sintética (MAX_DOCS, não o corpus inteiro) — não vale o custo/dependência
extra (fasttext + download de modelo) para este propósito. Não filtramos
variante; é uma simplificação deliberada, não um esquecimento.

PRÉ-REQUISITOS:
    1. WORK_DIR gravável (container posttrain: /workspace/manaca-corpus)
    2. `pip install -r requirements/posttrain.txt -r requirements/corpus.txt`

Uso:
    python corpus/scripts/04_acquire_madlad400.py
    python corpus/scripts/04_acquire_madlad400.py --max-docs 0   # sem limite

Saída:
    $WORK_DIR/raw/madlad400/
    ├── shard_000000.parquet
    ├── ...
    └── download.log

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

SOURCE_NAME  = "madlad400"
HF_REPO_ID   = "allenai/MADLAD-400"
HF_CONFIG    = "pt"
HF_SPLIT     = "clean"  # MADLAD-400 usa 'clean' como split principal (ver ficha do dataset)
LICENSE      = "apache-2.0"
REFERENCE    = "Kudugunta et al. (2024). MADLAD-400. arXiv:2309.04662."

WORK_DIR      = Path(os.environ.get("WORK_DIR", Path.home() / "manaca-corpus"))
OUTPUT_DIR    = WORK_DIR / "raw" / SOURCE_NAME
LOG_DIR       = WORK_DIR / "logs"
CKPT_DIR      = WORK_DIR / "checkpoints"
MANIFEST_PATH = CKPT_DIR / "acquisition_manifest.json"

SHARD_SIZE       = 50_000
MIN_TEXT_LEN     = 50
DEFAULT_MAX_DOCS = 300_000

SCHEMA = pa.schema([
    pa.field("text", pa.string()), pa.field("source", pa.string()),
    pa.field("id", pa.string()), pa.field("lang", pa.string()),
    pa.field("score", pa.float32()),
])


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
    logger.add(log_file, level="DEBUG", rotation="100 MB", retention=3, encoding="utf-8",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}")
    logger.info(f"Log: {log_file}")


def load_manifest() -> dict[str, Any]:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"version": "0.1.0", "project": "Manaca-1B-Instruct-2 — LNCC x NII/LLM-jp",
            "created_at": datetime.now(timezone.utc).isoformat(), "sources": {}}


def save_manifest(manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(MANIFEST_PATH)


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


def passes_filter(text: str) -> bool:
    if not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < MIN_TEXT_LEN:
        return False
    n_alpha = sum(1 for c in text if c.isalpha())
    return n_alpha / max(len(text), 1) >= 0.50


def quality_score(text: str) -> float:
    import math
    if not text:
        return 0.0
    words = text.split()
    n = max(len(words), 1)
    alpha = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    ttr = len(set(w.lower() for w in words)) / n
    lscore = min(math.log1p(len(text)) / 14.0, 1.0)
    return round(0.4 * alpha + 0.3 * ttr + 0.3 * lscore, 4)


def write_shard(docs: list[dict], shard_id: int, output_dir: Path) -> dict[str, Any]:
    path = output_dir / f"shard_{shard_id:06d}.parquet"
    table = pa.table({
        "text": [d["text"] for d in docs], "source": [d["source"] for d in docs],
        "id": [d["id"] for d in docs], "lang": [d["lang"] for d in docs],
        "score": [d["score"] for d in docs],
    }, schema=SCHEMA)
    pq.write_table(table, path, compression="zstd")
    size_bytes = path.stat().st_size
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read(1 << 20)).hexdigest()
    return {"shard_id": shard_id, "path": str(path), "n_docs": len(docs),
            "size_bytes": size_bytes, "size_mb": round(size_bytes / 1e6, 1),
            "sha256_1mb": sha, "written_at": datetime.now(timezone.utc).isoformat()}


def acquire(max_docs: int) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(OUTPUT_DIR / "download.log")
    manifest = load_manifest()

    existing = get_existing_shards(OUTPUT_DIR)
    next_shard = (max(existing) + 1) if existing else 0
    # BUG corrigido (mesma causa em todos os corpus/scripts/0X_acquire_*.py):
    # `len(existing) * SHARD_SIZE` so aproxima a posicao real no stream bruto
    # se a retencao for ~100%; com filtro de qualidade ela nao e. Usa o
    # `total_seen` real gravado no manifest da execucao anterior.
    prev = manifest.get("sources", {}).get(SOURCE_NAME, {})
    docs_to_skip = prev.get("total_seen", 0)

    logger.info("=" * 60)
    logger.info(f"Fonte: {SOURCE_NAME}  |  Repo: {HF_REPO_ID} (config={HF_CONFIG})")
    logger.info(f"Teto de docs: {'sem limite' if max_docs <= 0 else f'{max_docs:,}'}")
    logger.info(f"Licenca: {LICENSE}  |  Ref: {REFERENCE}")
    if docs_to_skip:
        logger.info(f"Retomando: {len(existing)} shards existentes | {prev.get('total_kept', 0):,} docs "
                     f"mantidos | pulando os primeiros {docs_to_skip:,} docs do stream bruto")
    logger.info("=" * 60)

    try:
        dataset = load_dataset(HF_REPO_ID, HF_CONFIG, split=HF_SPLIT, streaming=True)
    except Exception as e:
        logger.warning(f"split='{HF_SPLIT}' falhou ({e}); tentando split='train'...")
        dataset = load_dataset(HF_REPO_ID, HF_CONFIG, split="train", streaming=True)

    first = next(iter(dataset))
    text_col = next((c for c in ["text", "content"] if c in first), list(first.keys())[0])
    id_col = next((c for c in ["id", "url", "doc_id"] if c in first), None)
    logger.info(f"Coluna texto: '{text_col}'  |  ID: '{id_col or 'gerado'}'  |  Campos: {list(first.keys())}")

    buffer, shard_id = [], next_shard
    total_seen  = prev.get("total_seen", 0)
    kept        = prev.get("total_kept", 0)
    skipped     = prev.get("total_skip", prev.get("total_skipped", 0))
    total_chars = prev.get("total_chars", 0)
    stream_pos = 0
    t_start = time.time()
    for item in tqdm(dataset, desc="MADLAD-400/pt", unit="doc", mininterval=15, file=sys.stderr):
        stream_pos += 1
        if stream_pos <= docs_to_skip:
            continue
        # cap ANTES de contar (evita off-by-one -- ver 01_acquire_legalpt.py)
        if max_docs > 0 and kept >= max_docs:
            logger.info(f"Teto de {max_docs:,} docs atingido — parando.")
            break
        total_seen += 1
        text = str(item.get(text_col, "")).strip()
        if not passes_filter(text):
            skipped += 1
            continue
        raw_id = item.get(id_col, "") if id_col else ""
        buffer.append({"text": text, "source": SOURCE_NAME,
                        "id": str(raw_id) if raw_id else f"{SOURCE_NAME}_{total_seen}",
                        "lang": "por_Latn", "score": quality_score(text)})
        kept += 1
        total_chars += len(text)
        if len(buffer) >= SHARD_SIZE:
            meta = write_shard(buffer, shard_id, OUTPUT_DIR)
            elapsed = time.time() - t_start
            logger.info(f"Shard {shard_id:04d} | {kept:>8,} docs | {meta['size_mb']:5.0f} MB | "
                        f"{kept / max(elapsed, 1):,.0f} docs/s")
            manifest["sources"][SOURCE_NAME] = {
                "status": "in_progress", "hf_repo": HF_REPO_ID, "license": LICENSE,
                "total_seen": total_seen, "total_kept": kept, "total_skip": skipped,
                "total_chars": total_chars, "shards_written": shard_id + 1,
                "estimated_tokens_b": round(total_chars / 4 / 1e9, 3),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            save_manifest(manifest)
            buffer, shard_id = [], shard_id + 1

    if buffer:
        meta = write_shard(buffer, shard_id, OUTPUT_DIR)
        logger.info(f"Shard final {shard_id:04d} | {len(buffer):,} docs | {meta['size_mb']:.0f} MB")

    elapsed = time.time() - t_start
    total_bytes = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.parquet"))
    est_tokens = total_chars // 4
    final = {
        "status": "completed" if not (max_docs > 0 and kept >= max_docs) else "capped_by_max_docs",
        "hf_repo": HF_REPO_ID, "license": LICENSE, "reference": REFERENCE,
        "total_seen": total_seen, "total_kept": kept, "total_skipped": skipped,
        "total_chars": total_chars,
        "retention_rate": round(kept / max(total_seen, 1), 4), "shards_written": shard_id + 1,
        "total_bytes": total_bytes, "total_gb": round(total_bytes / 1e9, 3),
        "estimated_tokens": est_tokens, "estimated_tokens_b": round(est_tokens / 1e9, 3),
        "elapsed_hours": round(elapsed / 3600, 2), "output_dir": str(OUTPUT_DIR),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest["sources"][SOURCE_NAME] = final
    save_manifest(manifest)
    logger.info("=" * 60)
    logger.info(f"CONCLUIDO — {SOURCE_NAME}: {kept:,} docs mantidos, {total_bytes/1e9:.3f} GB, "
                f"~{est_tokens/1e9:.3f}B tokens")
    logger.info("Proximo passo: python corpus/scripts/05_acquire_fineweb2.py")
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
        except Exception:
            errors += 1
    logger.info(f"Verificacao: {len(shards)} shards | {total_rows:,} docs | {errors} erro(s)")
    try:
        t = pq.read_table(shards[-1], columns=["text", "score"])
        logger.info(f'Amostra: "{t["text"][0].as_py()[:80]}..." [score={t["score"][0].as_py():.2f}]')
    except Exception as e:
        logger.warning(f"Erro ao ler amostra: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Aquisicao do MADLAD-400 (config pt)")
    ap.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS,
                     help=f"teto de documentos mantidos (default {DEFAULT_MAX_DOCS:,}; 0 = sem limite)")
    args = ap.parse_args()
    try:
        acquire(max_docs=args.max_docs)
        verify()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n[INFO] Interrompido. Re-executar para retomar do ultimo shard.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
