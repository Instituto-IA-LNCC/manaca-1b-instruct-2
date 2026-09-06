#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — Script 02: Aquisição do ClassiCC-PT (web nativo, filtrado por qualidade)
=================================================================================================
LNCC × NII/LLM-jp | Corpus-fonte para instrução sintética (Fase "instruct-2")

Baixa `ClassiCC-Corpus/ClassiCC-PT`, filtra por qualidade usando os scores
JÁ EMBUTIDOS no dataset (`edu_score`, `stem_score`, `toxic_score` — de
classificadores BERTimbau anotados por GPT-4o), e grava em Apache Parquet +
Zstandard, particionado em shards de 50.000 documentos. Idempotente.

Papel neste repositório (diferente do `manaca`):
    Matéria-prima NATIVA em PT-BR (não traduzida) para geração de instrução
    sintética, no lugar de continuar dependendo de tradução automática
    (Alpaca-PT/OPUS-100, hoje ~64% do corpus de SFT do v1/v2 — ver
    docs/data/README.md do v1). Diferente de um pipeline de pré-treino, não
    baixamos o corpus inteiro (~120B tokens): usamos o teto MAX_DOCS e o
    filtro de `edu_score` para amostrar só os documentos de alta qualidade
    educacional, que rendem instrução sintética melhor que texto web médio.

Justificativa científica:
    ClassiCC-PT (arXiv:2509.08824) constrói um corpus web PT-BR a partir de
    3 snapshots do Common Crawl (2021-31, 2021-39, 2022-40), com pipeline
    auditável (Trafilatura + MinHash intra-crawl) e 3 classificadores
    BERTimbau (educacional/STEM/toxicidade) treinados com anotações do
    GPT-4o — o análogo PT-BR do FineWeb-Edu. Continuar-pretreinar um
    TinyLlama-1.1B neste corpus (modelo "Curió") superou mC4-PT e igualou
    o ClueWeb22-PT em benchmarks PT-BR.

    HF: https://huggingface.co/datasets/ClassiCC-Corpus/ClassiCC-PT

SOBREPOSIÇÃO COM O BASE:
    ClassiCC-PT deriva de Common Crawl, assim como partes do GigaVerbo
    (Tier 1 do base). Sobreposição de texto bruto entre agregações de CC é
    praticamente inevitável e não é o que este pipeline tenta zerar — o
    ponto é que aqui o texto é TRANSFORMADO em instrução sintética, não
    reusado como pré-treino. Ainda assim, o `id` gravado permite decontaminar
    depois (ex.: hash cruzado com o manifest do GigaVerbo) se necessário.

Filtro de qualidade (diferença chave vs. os demais scripts desta suite):
    Além do `passes_filter`/`quality_score` genéricos, usamos os campos
    nativos do dataset:
      - MIN_EDU_SCORE (default 2.0): documentos abaixo são descartados.
        Assume escala 0-5 (estilo FineWeb-Edu); o range real observado é
        logado nas estatísticas finais para auditar essa suposição.
      - MAX_TOXIC_SCORE (default 0.5): descarta conteúdo tóxico. Menos
        estrito que um pipeline de pré-treino (que zeraria toxicidade)
        porque aqui um documento sensível-mas-legítimo pode ainda virar
        um exemplo de recusa/ajuda no safety-SFT — mas nunca conteúdo
        francamente tóxico.
    Quando os campos não existem no doc (schema mudou), o filtro NÃO
    derruba o documento — ele cai para o quality_score() genérico, e um
    aviso é logado uma única vez.

PRÉ-REQUISITOS:
    1. WORK_DIR gravável (container posttrain: /workspace/manaca-corpus)
    2. `pip install -r requirements/posttrain.txt -r requirements/corpus.txt`

Uso:
    python corpus/scripts/02_acquire_classiccpt.py
    python corpus/scripts/02_acquire_classiccpt.py --max-docs 500000 --min-edu 2.5

Saída:
    $WORK_DIR/raw/classiccpt/
    ├── shard_000000.parquet
    ├── ...
    ├── download.log
    └── classiccpt_quality_stats.json   distribuicao de edu/stem/toxic_score

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

SOURCE_NAME  = "classiccpt"
HF_REPO_ID   = "ClassiCC-Corpus/ClassiCC-PT"
HF_SPLIT     = "train"
LICENSE      = "aberta (ver ficha do dataset — herda termos do Common Crawl)"
REFERENCE    = "ClassiCC-PT (2026). arXiv:2509.08824."
EST_TOKENS_B = 120  # bilhoes, corpus completo (nao baixado por inteiro aqui)

WORK_DIR      = Path(os.environ.get("WORK_DIR", Path.home() / "manaca-corpus"))
OUTPUT_DIR    = WORK_DIR / "raw" / SOURCE_NAME
LOG_DIR       = WORK_DIR / "logs"
CKPT_DIR      = WORK_DIR / "checkpoints"
MANIFEST_PATH = CKPT_DIR / "acquisition_manifest.json"

SHARD_SIZE   = 50_000
MIN_TEXT_LEN = 50

DEFAULT_MAX_DOCS      = 300_000  # ver nota de escopo no docstring
DEFAULT_MIN_EDU_SCORE = 2.0      # escala assumida 0-5 (estilo FineWeb-Edu)
DEFAULT_MAX_TOXIC     = 0.5

SCHEMA = pa.schema([
    pa.field("text",   pa.string()),
    pa.field("source", pa.string()),
    pa.field("id",     pa.string()),
    pa.field("lang",   pa.string()),
    pa.field("score",  pa.float32()),
])


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
    """Filtro de sanidade genérico (comum a toda a suite)."""
    if not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < MIN_TEXT_LEN:
        return False
    n_alpha = sum(1 for c in text if c.isalpha())
    return n_alpha / max(len(text), 1) >= 0.50


def passes_classifier_filter(item: dict, min_edu: float, max_toxic: float, warned: list[bool]) -> bool:
    """
    Filtro pelos classificadores nativos do ClassiCC-PT.
    Campos ausentes não derrubam o documento (fallback ao quality_score
    genérico) — apenas registra um aviso único, para não afogar o log.
    """
    has_edu   = "edu_score" in item
    has_toxic = "toxic_score" in item
    if not has_edu and not has_toxic:
        if not warned[0]:
            logger.warning(
                "Campos edu_score/toxic_score ausentes no dataset — "
                "usando apenas quality_score() generico para todos os docs."
            )
            warned[0] = True
        return True

    edu   = item.get("edu_score")
    toxic = item.get("toxic_score")
    if edu is not None and edu < min_edu:
        return False
    if toxic is not None and toxic > max_toxic:
        return False
    return True


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


# ── Pipeline principal ────────────────────────────────────────────────────────

def acquire(max_docs: int, min_edu: float, max_toxic: float) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    setup_logging(OUTPUT_DIR / "download.log")
    manifest = load_manifest()

    existing     = get_existing_shards(OUTPUT_DIR)
    next_shard   = (max(existing) + 1) if existing else 0

    # BUG corrigido: `docs_to_skip` NAO pode ser `shards_existentes * SHARD_SIZE`
    # -- isso assume 1 shard = SHARD_SIZE posicoes no stream bruto, o que so
    # vale com 100% de retencao. Aqui a retencao real e ~7% (a maioria e
    # descartada por filtro de qualidade/classificador), entao o stream bruto
    # ja consumido e MUITO maior que `shards*SHARD_SIZE` -- usar essa
    # estimativa reabriria o stream do zero e reprocessaria/duplicaria quase
    # todo o corpus ja salvo. Usa o `total_seen` real do manifest (posicao
    # exata onde a execucao anterior parou).
    prev         = manifest.get("sources", {}).get(SOURCE_NAME, {})
    docs_to_skip = prev.get("total_seen", 0)

    logger.info("=" * 60)
    logger.info(f"Fonte:        {SOURCE_NAME}")
    logger.info(f"Repositorio:  {HF_REPO_ID}")
    logger.info(f"Filtro:       edu_score >= {min_edu}  |  toxic_score <= {max_toxic}")
    logger.info(f"Teto de docs: {'sem limite' if max_docs <= 0 else f'{max_docs:,}'}")
    logger.info(f"Licenca:      {LICENSE}")
    logger.info(f"Referencia:   {REFERENCE}")
    logger.info(f"Output:       {OUTPUT_DIR}")
    if docs_to_skip:
        logger.info(f"Retomando: {len(existing)} shards existentes | "
                    f"{prev.get('total_kept', 0):,} docs mantidos | "
                    f"pulando os primeiros {docs_to_skip:,} docs do stream bruto")
    logger.info("=" * 60)

    logger.info(f"Conectando ao HuggingFace: {HF_REPO_ID} ...")
    try:
        dataset = load_dataset(HF_REPO_ID, split=HF_SPLIT, streaming=True)
    except Exception as e:
        logger.error(f"Falha ao carregar dataset: {e}")
        raise

    first = next(iter(dataset))
    text_col = next((c for c in ["text", "content", "body"] if c in first), list(first.keys())[0])
    # 'url' ANTES de 'id' de proposito: confirmado ao vivo que o campo 'id'
    # do ClassiCC-Corpus/ClassiCC-PT NAO identifica um documento -- e um id
    # de segmento/WARC do Common Crawl compartilhado por CENTENAS de
    # documentos diferentes (99,86% de "duplicacao" numa checagem real:
    # 600.000 linhas ja adquiridas, so 13.503 valores distintos de 'id'; um
    # unico valor cobria 609 documentos DIFERENTES numa amostra de 3.000).
    # 'url' e o campo que de fato varia por documento nesse dataset. Isso so
    # afeta aquisicoes NOVAS -- os shards ja gravados antes dessa correcao
    # tem 'id' errado, mas synth/generate_instructions.py:sample_passages()
    # nao depende mais dele (usa posicao fisica da linha no shard).
    id_col   = next((c for c in ["url", "id", "doc_id"] if c in first), None)
    logger.info(f"Coluna texto: '{text_col}'  |  Coluna ID: '{id_col or 'gerado'}'")
    logger.info(f"Campos disponiveis: {list(first.keys())}")

    buffer: list[dict] = []
    shard_id = next_shard
    total_seen  = prev.get("total_seen", 0)
    kept        = prev.get("total_kept", 0)
    skip_qual   = prev.get("skip_quality", 0)
    skip_clf    = prev.get("skip_classifier", 0)
    total_chars = prev.get("total_chars", 0)
    stream_pos = 0
    edu_scores: list[float] = []
    toxic_scores: list[float] = []
    warned = [False]
    t_start = time.time()

    for item in tqdm(dataset, desc="ClassiCC-PT", unit="doc", mininterval=15, file=sys.stderr):
        stream_pos += 1
        if stream_pos <= docs_to_skip:
            continue
        # cap ANTES de contar (evita off-by-one -- ver 01_acquire_legalpt.py)
        if max_docs > 0 and kept >= max_docs:
            logger.info(f"Teto de {max_docs:,} docs atingido — parando.")
            break
        total_seen += 1

        if not passes_classifier_filter(item, min_edu, max_toxic, warned):
            skip_clf += 1
            continue

        text = str(item.get(text_col, "")).strip()
        if not passes_filter(text):
            skip_qual += 1
            continue

        if "edu_score" in item and item["edu_score"] is not None:
            edu_scores.append(float(item["edu_score"]))
        if "toxic_score" in item and item["toxic_score"] is not None:
            toxic_scores.append(float(item["toxic_score"]))

        raw_id = item.get(id_col, "") if id_col else ""
        buffer.append({
            "text": text, "source": SOURCE_NAME,
            "id": str(raw_id) if raw_id else f"{SOURCE_NAME}_{total_seen}",
            "lang": "por_Latn", "score": quality_score(text),
        })
        kept += 1
        total_chars += len(text)

        if len(buffer) >= SHARD_SIZE:
            meta = write_shard(buffer, shard_id, OUTPUT_DIR)
            elapsed = time.time() - t_start
            logger.info(
                f"Shard {shard_id:04d} | {kept:>8,} docs | {meta['size_mb']:5.0f} MB | "
                f"{kept / max(elapsed, 1):,.0f} docs/s"
            )
            manifest["sources"][SOURCE_NAME] = {
                "status": "in_progress", "hf_repo": HF_REPO_ID, "license": LICENSE,
                "total_seen": total_seen, "total_kept": kept,
                "skip_quality": skip_qual, "skip_classifier": skip_clf,
                "total_chars": total_chars,
                "shards_written": shard_id + 1,
                "estimated_tokens_b": round(total_chars / 4 / 1e9, 3),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            save_manifest(manifest)
            buffer, shard_id = [], shard_id + 1

    if buffer:
        meta = write_shard(buffer, shard_id, OUTPUT_DIR)
        logger.info(f"Shard final {shard_id:04d} | {len(buffer):,} docs | {meta['size_mb']:.0f} MB")

    def pct(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        s = sorted(data)
        return s[int(len(s) * p / 100)]

    quality_stats = {
        "source": SOURCE_NAME, "total_kept": kept,
        "edu_score_observed_min": round(min(edu_scores), 3) if edu_scores else None,
        "edu_score_observed_max": round(max(edu_scores), 3) if edu_scores else None,
        "edu_score_p50": round(pct(edu_scores, 50), 3) if edu_scores else None,
        "toxic_score_p50": round(pct(toxic_scores, 50), 3) if toxic_scores else None,
        "toxic_score_p90": round(pct(toxic_scores, 90), 3) if toxic_scores else None,
        "min_edu_threshold_used": min_edu, "max_toxic_threshold_used": max_toxic,
        "note": (
            "Se edu_score_observed_max nao estiver perto de ~5, a suposicao de "
            "escala 0-5 (estilo FineWeb-Edu) esta errada — recalibrar MIN_EDU_SCORE."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    stats_path = OUTPUT_DIR / "classiccpt_quality_stats.json"
    stats_path.write_text(json.dumps(quality_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Estatisticas de qualidade salvas: {stats_path}")
    if edu_scores:
        logger.info(
            f"  edu_score observado: min={quality_stats['edu_score_observed_min']} "
            f"p50={quality_stats['edu_score_p50']} max={quality_stats['edu_score_observed_max']}"
        )

    elapsed = time.time() - t_start
    total_bytes = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.parquet"))
    est_tokens = total_chars // 4

    final: dict[str, Any] = {
        "status": "completed" if not (max_docs > 0 and kept >= max_docs) else "capped_by_max_docs",
        "hf_repo": HF_REPO_ID, "license": LICENSE, "reference": REFERENCE,
        "total_seen": total_seen, "total_kept": kept, "total_chars": total_chars,
        "total_skipped": skip_qual + skip_clf,
        "skip_quality": skip_qual, "skip_classifier": skip_clf,
        "retention_rate": round(kept / max(total_seen, 1), 4),
        "shards_written": shard_id + 1, "total_bytes": total_bytes,
        "total_gb": round(total_bytes / 1e9, 3), "estimated_tokens": est_tokens,
        "estimated_tokens_b": round(est_tokens / 1e9, 3), "elapsed_hours": round(elapsed / 3600, 2),
        "output_dir": str(OUTPUT_DIR), "stats_path": str(stats_path),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest["sources"][SOURCE_NAME] = final
    save_manifest(manifest)

    logger.info("=" * 60)
    logger.info(f"CONCLUIDO — {SOURCE_NAME}")
    logger.info(f"  Docs mantidos:     {kept:,}")
    logger.info(f"  Desc. classific.:  {skip_clf:,}  |  Desc. qualidade: {skip_qual:,}")
    logger.info(f"  Taxa retencao:     {final['retention_rate']:.1%}")
    logger.info(f"  Tamanho total:     {total_bytes / 1e9:.3f} GB  |  Tokens est.: {est_tokens / 1e9:.3f}B")
    logger.info(f"  Output:            {OUTPUT_DIR}")
    logger.info("=" * 60)
    logger.info("Proximo passo: python corpus/scripts/03_acquire_carolina.py")
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

    stats_path = OUTPUT_DIR / "classiccpt_quality_stats.json"
    if stats_path.exists():
        s = json.loads(stats_path.read_text(encoding="utf-8"))
        logger.info(f"  edu_score p50={s.get('edu_score_p50')} | toxic_score p90={s.get('toxic_score_p90')}")

    try:
        t = pq.read_table(shards[-1], columns=["text", "score"])
        logger.info(f'Amostra ({shards[-1].name}): "{t["text"][0].as_py()[:80]}..." [score={t["score"][0].as_py():.2f}]')
    except Exception as e:
        logger.warning(f"Erro ao ler amostra: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Aquisicao do ClassiCC-PT (filtrado por edu/toxic_score)")
    ap.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS,
                     help=f"teto de documentos mantidos (default {DEFAULT_MAX_DOCS:,}; 0 = sem limite)")
    ap.add_argument("--min-edu", type=float, default=DEFAULT_MIN_EDU_SCORE,
                     help=f"edu_score minimo (default {DEFAULT_MIN_EDU_SCORE})")
    ap.add_argument("--max-toxic", type=float, default=DEFAULT_MAX_TOXIC,
                     help=f"toxic_score maximo (default {DEFAULT_MAX_TOXIC})")
    args = ap.parse_args()
    try:
        acquire(max_docs=args.max_docs, min_edu=args.min_edu, max_toxic=args.max_toxic)
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
