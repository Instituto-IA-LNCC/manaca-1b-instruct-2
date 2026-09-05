#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — Script 01: Aquisição do LegalPT (domínio jurídico)
==========================================================================
LNCC × NII/LLM-jp | Corpus-fonte para instrução sintética (Fase "instruct-2")

Baixa `eduagarcia/LegalPT_dedup`, EXCLUINDO explicitamente o sub-corpus
`tesemo_v2` (Ulysses Tesemõ), e grava em Apache Parquet + Zstandard,
particionado em shards de 50.000 documentos. Idempotente: retoma
automaticamente do último shard salvo.

Papel neste repositório (diferente do `manaca`):
    Este NÃO é corpus de pré-treino. O Manacá-1B-Instruct-2 usa passagens
    daqui como MATÉRIA-PRIMA para um passo seguinte de geração de instrução
    sintética (documento -> {instruction, input, output} via LLM professor),
    visando o domínio jurídico onde o instruct atual mede desempenho no
    acaso (OAB ~23-25% de acurácia — ver docs/evaluation/instruct-eval-pt.md
    do v1). Por isso este script limita a extração a MAX_DOCS por padrão
    (ver abaixo) em vez de baixar o corpus inteiro (125 GiB) como faria um
    pipeline de pré-treino.

Justificativa científica:
    Niklaus et al. (2023, MultiLegalPile) e Garcia et al. (2024, LegalPT/
    RoBERTaLexPT) mostram que modelos gerais têm desempenho sistematicamente
    inferior em linguagem jurídica (vocabulário técnico, estrutura sintática
    complexa, referências normativas). O LegalPT agrega 6 sub-corpora
    jurídico-legislativos em PT-BR/PT-PT, deduplicados via MinHash LSH
    (Lee et al., 2022).

    Referência: Garcia, E. et al. (2024). RoBERTaLexPT: A Legal RoBERTa
    Model pretrained with deduplication for Portuguese.
    HF: https://huggingface.co/datasets/eduagarcia/LegalPT_dedup

POR QUE EXCLUIR `tesemo_v2` (diferença crítica deste script):
    O `manaca` (pré-treino do base) já extrai o Ulysses Tesemõ inteiro no
    Script 06 (`corpus/scripts/06_acquire_ulysses.py`, ~10B tokens). O
    LegalPT_dedup reempacota esse MESMO corpus como um de seus 6
    sub-corpora (config `tesemo_v2`, ~1,74M docs / 12,9 GB). Reincluí-lo
    aqui duplicaria dado já visto pelo base — o oposto do que se pede ao
    escolher fontes para o instruct-2. Este script itera as configs do
    LegalPT_dedup INDIVIDUALMENTE e pula `tesemo_v2` deliberadamente (não
    depende de inferir o valor da coluna `source`, que não é garantido
    documentalmente — pular a config inteira é a forma correta e auditável).

ADVERTÊNCIA DE LICENÇA (auditar antes de qualquer uso comercial):
    O LegalPT_dedup agrega fontes de licenças distintas — parte do
    MultiLegalPile (`mlp_pt_*`) tem termos mistos por sub-jurisdição/
    sub-fonte (ver Niklaus et al., 2023 e a ficha do dataset no HF).
    Este script grava a config de origem por documento (campo `source`)
    para que a licença possa ser verificada e filtrada por sub-corpus
    antes de qualquer uso além de pesquisa. NÃO presuma licença permissiva
    uniforme.

PRÉ-REQUISITOS:
    1. WORK_DIR gravável (no container posttrain: /workspace/manaca-corpus,
       bind-mount de DATA_DIR — ver .env.example)
    2. `pip install -r requirements/corpus.txt` (além de requirements/posttrain.txt
       — reaproveita o `datasets==3.1.0` já pinado no posttrain, que ainda
       suporta `trust_remote_code`; ver nota de versão em requirements/corpus.txt)

Uso:
    python corpus/scripts/01_acquire_legalpt.py
    python corpus/scripts/01_acquire_legalpt.py --max-docs 500000
    python corpus/scripts/01_acquire_legalpt.py --max-docs 0   # sem limite (corpus inteiro)

Saída:
    $WORK_DIR/raw/legalpt/
    ├── shard_000000.parquet
    ├── ...
    ├── download.log
    └── config_distribution.json   contagem de docs por sub-corpus (config)

Autor: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
Versão: 0.1.0 — Setembro 2026
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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

SOURCE_NAME  = "legalpt"
HF_REPO_ID   = "eduagarcia/LegalPT_dedup"
HF_SPLIT     = "train"
REFERENCE    = "Garcia et al. (2024). RoBERTaLexPT / LegalPT_dedup."

# Todas as configs conhecidas do LegalPT_dedup (ver source_probe_results / HF).
# `tesemo_v2` é DELIBERADAMENTE excluída (ver docstring acima).
EXCLUDED_CONFIGS = {"tesemo_v2"}
ALL_CONFIGS = [
    "acordaos_tcu",
    "datastf",
    "iudicium_textum",
    "mlp_pt_BRCAD-5",
    "mlp_pt_CJPG",
    "mlp_pt_eurlex-caselaw",
    "mlp_pt_eurlex-contracts",
    "mlp_pt_eurlex-legislation",
    "mlp_pt_legal-mc4",
    "parlamento-pt",
    # "tesemo_v2",  # excluída: já está no Tier 1 do base (manaca/corpus/scripts/06)
]

# Licença por config — melhor conhecimento atual; CONFIRMAR na origem antes
# de uso comercial (MultiLegalPile tem termos mistos por sub-fonte).
CONFIG_LICENSE = {
    "acordaos_tcu":              "dominio-publico (documento governamental, Art. 216 CF/1988)",
    "datastf":                   "dominio-publico (documento governamental, Art. 216 CF/1988)",
    "iudicium_textum":           "dominio-publico (STF, decisoes judiciais)",
    "parlamento-pt":             "aberta (portal do governo — verificar termos PT-PT)",
    "mlp_pt_BRCAD-5":            "mista/verificar (MultiLegalPile — Niklaus et al. 2023)",
    "mlp_pt_CJPG":               "mista/verificar (MultiLegalPile — Niklaus et al. 2023)",
    "mlp_pt_eurlex-caselaw":     "mista/verificar (EUR-Lex — reuso da UE, condicoes proprias)",
    "mlp_pt_eurlex-contracts":   "mista/verificar (EUR-Lex — reuso da UE, condicoes proprias)",
    "mlp_pt_eurlex-legislation": "mista/verificar (EUR-Lex — reuso da UE, condicoes proprias)",
    "mlp_pt_legal-mc4":          "mista/verificar (derivado do mc4, herda termos do Common Crawl)",
}

WORK_DIR      = Path(os.environ.get("WORK_DIR", Path.home() / "manaca-corpus"))
OUTPUT_DIR    = WORK_DIR / "raw" / SOURCE_NAME
LOG_DIR       = WORK_DIR / "logs"
CKPT_DIR      = WORK_DIR / "checkpoints"
MANIFEST_PATH = CKPT_DIR / "acquisition_manifest.json"

SHARD_SIZE   = 50_000
MIN_TEXT_LEN = 100    # texto juridico curto raramente tem valor (fiel ao manaca/06_acquire_ulysses)

# Teto de documentos MANTIDOS por padrão. Diferente do pipeline de pré-treino
# do `manaca` (que baixa o corpus inteiro): aqui a saída alimenta um passo de
# geração de instrução sintética que não precisa (nem deveria, por custo de
# API do LLM professor) de milhões de documentos. `--max-docs 0` remove o teto.
DEFAULT_MAX_DOCS = 300_000

# Schema Parquet fixo — idêntico à suite corpus/ do `manaca` (compatibilidade
# de ferramentas: dedup, validação, leitura colunar).
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
        sys.stderr,
        level="INFO",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
    )
    logger.add(
        log_file,
        level="DEBUG",
        rotation="100 MB",
        retention=3,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}",
    )
    logger.info(f"Log: {log_file}")


# ── Manifesto ─────────────────────────────────────────────────────────────────

def load_manifest() -> dict[str, Any]:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        "version":    "0.1.0",
        "project":    "Manaca-1B-Instruct-2 — LNCC x NII/LLM-jp",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources":    {},
    }


def save_manifest(manifest: dict[str, Any]) -> None:
    """Escrita atômica — garante integridade em caso de interrupção."""
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(MANIFEST_PATH)


# ── Retomada automática ───────────────────────────────────────────────────────

def get_existing_shards(output_dir: Path) -> set[int]:
    """Retorna IDs dos shards já escritos e com integridade verificada."""
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


def get_completed_configs(manifest: dict[str, Any]) -> set[str]:
    """Configs já totalmente processadas em execuções anteriores (retomada por config)."""
    src = manifest.get("sources", {}).get(SOURCE_NAME, {})
    return set(src.get("completed_configs", []))


# ── Limpeza de texto jurídico ─────────────────────────────────────────────────
# Reaproveita a mesma abordagem conservadora do manaca/06_acquire_ulysses.py:
# preserva estrutura semântica (artigos, ementas, cabeçalhos institucionais),
# remove apenas ruído de formatação.

_RE_MULTIPLAS_LINHAS  = re.compile(r"\n{3,}")
_RE_ESPACOS_MULTIPLOS = re.compile(r"[ \t]{2,}")
_RE_PAGINA            = re.compile(r"^\s*-?\s*\d+\s*-?\s*$", re.MULTILINE)


def clean_legal_text(text: str) -> str:
    """Limpeza leve de artefatos de formatação em texto jurídico-legislativo."""
    if not text:
        return ""
    text = _RE_PAGINA.sub("", text)
    text = _RE_ESPACOS_MULTIPLOS.sub(" ", text)
    text = _RE_MULTIPLAS_LINHAS.sub("\n\n", text)
    return text.strip()


# ── Filtros de qualidade ──────────────────────────────────────────────────────

def passes_filter(text: str) -> bool:
    """
    Filtros de qualidade para texto jurídico.
    Threshold alfabético 40% (vs 50% em texto web genérico) para acomodar
    números de processo, artigos, incisos e valores monetários — idêntico
    ao critério usado em manaca/06_acquire_ulysses.py, para comparabilidade.
    """
    if not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < MIN_TEXT_LEN:
        return False
    n_alpha = sum(1 for c in text if c.isalpha())
    return n_alpha / max(len(text), 1) >= 0.40


def quality_score(text: str) -> float:
    """Score de qualidade composto [0.0-1.0]. Fórmula idêntica à suite manaca/corpus."""
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
            "text":   [d["text"]   for d in docs],
            "source": [d["source"] for d in docs],
            "id":     [d["id"]     for d in docs],
            "lang":   [d["lang"]   for d in docs],
            "score":  [d["score"]  for d in docs],
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

def acquire(max_docs: int) -> dict[str, Any]:
    """
    Pipeline principal de aquisição do LegalPT (menos `tesemo_v2`).

    Diferença estrutural vs. os scripts de fonte única do `manaca`: aqui
    iteramos MÚLTIPLAS configs HuggingFace do mesmo repo_id (uma por
    sub-corpus jurídico), cada uma com seu próprio load_dataset(...,
    streaming=True). A retomada é por CONFIG COMPLETA (não por cursor
    dentro da config) — mais simples e robusta dado que cada sub-corpus
    é uma fonte HF independente.
    """
    if not (WORK_DIR.exists() or WORK_DIR.parent.exists()):
        print(f"[ERRO] WORK_DIR nao pode ser criado: {WORK_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    setup_logging(OUTPUT_DIR / "download.log")
    manifest = load_manifest()

    existing   = get_existing_shards(OUTPUT_DIR)
    shard_id   = (max(existing) + 1) if existing else 0
    completed  = get_completed_configs(manifest)

    logger.info("=" * 60)
    logger.info(f"Fonte:            {SOURCE_NAME}")
    logger.info(f"Repositorio:      {HF_REPO_ID}")
    logger.info(f"Configs incluidas: {len(ALL_CONFIGS)} de {len(ALL_CONFIGS) + len(EXCLUDED_CONFIGS)}")
    logger.info(f"Configs excluidas: {sorted(EXCLUDED_CONFIGS)} (ja no Tier 1 do base — manaca/06)")
    logger.info(f"Teto de docs:     {'sem limite' if max_docs <= 0 else f'{max_docs:,}'}")
    logger.info(f"Referencia:       {REFERENCE}")
    logger.info(f"Output:           {OUTPUT_DIR}")
    if existing:
        logger.info(f"Retomando: {len(existing)} shards existentes, {len(completed)} configs concluidas")
    logger.info("=" * 60)

    buffer: list[dict] = []
    total_seen  = 0
    kept        = 0
    skipped     = 0
    total_chars = 0
    config_counts: dict[str, int] = {}
    t_start = time.time()

    for config in ALL_CONFIGS:
        if config in completed:
            logger.info(f"[{config}] ja concluida em execucao anterior — pulando")
            continue
        if max_docs > 0 and kept >= max_docs:
            logger.info(f"Teto de {max_docs:,} docs atingido antes de [{config}] — parando")
            break

        logger.info(f"[{config}] conectando ao HuggingFace ({HF_REPO_ID}, config={config}) ...")
        try:
            dataset = load_dataset(HF_REPO_ID, config, split=HF_SPLIT, streaming=True)
        except Exception as e:
            logger.error(f"[{config}] falha ao carregar — pulando esta config: {e}")
            continue

        try:
            first = next(iter(dataset))
        except StopIteration:
            logger.warning(f"[{config}] config vazia — pulando")
            continue
        text_col = next((c for c in ["text", "content", "body"] if c in first), list(first.keys())[0])
        id_col   = next((c for c in ["id", "orig_id", "doc_id"] if c in first), None)
        logger.info(f"[{config}] coluna texto='{text_col}' id='{id_col or 'gerado'}'")

        config_kept = 0
        for item in tqdm(dataset, desc=f"legalpt/{config}", unit="doc", mininterval=15, file=sys.stderr):
            total_seen += 1
            if max_docs > 0 and kept >= max_docs:
                break

            raw_text = item.get(text_col, "")
            text = clean_legal_text(raw_text if isinstance(raw_text, str) else "")
            if not passes_filter(text):
                skipped += 1
                continue

            raw_id = item.get(id_col, "") if id_col else ""
            doc_id = str(raw_id) if raw_id else hashlib.md5(f"{config}_{total_seen}".encode()).hexdigest()[:16]

            buffer.append({
                "text":   text,
                "source": f"{SOURCE_NAME}_{config}",
                "id":     doc_id,
                "lang":   "por_Latn",
                "score":  quality_score(text),
            })
            kept += 1
            config_kept += 1
            total_chars += len(text)

            if len(buffer) >= SHARD_SIZE:
                meta = write_shard(buffer, shard_id, OUTPUT_DIR)
                elapsed = time.time() - t_start
                logger.info(
                    f"Shard {shard_id:04d} | {kept:>8,} docs | {meta['size_mb']:5.0f} MB | "
                    f"{kept / max(elapsed, 1):,.0f} docs/s | config atual: {config}"
                )
                manifest["sources"][SOURCE_NAME] = {
                    "status": "in_progress", "hf_repo": HF_REPO_ID,
                    "excluded_configs": sorted(EXCLUDED_CONFIGS),
                    "total_seen": total_seen, "total_kept": kept, "total_skip": skipped,
                    "shards_written": shard_id + 1,
                    "completed_configs": sorted(completed),
                    "estimated_tokens_b": round(total_chars / 4 / 1e9, 3),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                save_manifest(manifest)
                buffer, shard_id = [], shard_id + 1

        config_counts[config] = config_kept
        completed.add(config)
        logger.info(f"[{config}] concluida: {config_kept:,} docs mantidos")

    if buffer:
        meta = write_shard(buffer, shard_id, OUTPUT_DIR)
        logger.info(f"Shard final {shard_id:04d} | {len(buffer):,} docs | {meta['size_mb']:.0f} MB")

    dist_path = OUTPUT_DIR / "config_distribution.json"
    dist_path.write_text(json.dumps({
        "source": SOURCE_NAME,
        "excluded_configs": sorted(EXCLUDED_CONFIGS),
        "distribution": config_counts,
        "license_by_config": {c: CONFIG_LICENSE.get(c, "verificar") for c in config_counts},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - t_start
    total_bytes = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.parquet"))
    est_tokens = total_chars // 4  # ~4 chars/token PT-BR — estimado sobre chars reais, nao sobre n_docs

    final: dict[str, Any] = {
        "status": "completed" if not (max_docs > 0 and kept >= max_docs) else "capped_by_max_docs",
        "hf_repo": HF_REPO_ID, "excluded_configs": sorted(EXCLUDED_CONFIGS),
        "reference": REFERENCE, "total_seen": total_seen, "total_kept": kept,
        "total_skipped": skipped, "retention_rate": round(kept / max(total_seen, 1), 4),
        "shards_written": shard_id + 1, "total_bytes": total_bytes,
        "total_gb": round(total_bytes / 1e9, 3), "estimated_tokens": est_tokens,
        "estimated_tokens_b": round(est_tokens / 1e9, 3), "elapsed_hours": round(elapsed / 3600, 2),
        "completed_configs": sorted(completed), "config_distribution": config_counts,
        "output_dir": str(OUTPUT_DIR), "config_dist_path": str(dist_path),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest["sources"][SOURCE_NAME] = final
    save_manifest(manifest)

    logger.info("=" * 60)
    logger.info(f"CONCLUIDO — {SOURCE_NAME}")
    logger.info(f"  Docs mantidos:   {kept:,}  |  Descartados: {skipped:,}")
    logger.info(f"  Taxa retencao:   {final['retention_rate']:.1%}")
    logger.info(f"  Tamanho total:   {total_bytes / 1e9:.3f} GB  |  Tokens est.: {est_tokens / 1e9:.3f}B")
    logger.info(f"  Configs:         {dict(config_counts)}")
    logger.info(f"  Output:          {OUTPUT_DIR}")
    logger.info("=" * 60)
    logger.info("Proximo passo: python corpus/scripts/02_acquire_classiccpt.py")
    return final


def verify() -> None:
    """Verifica integridade dos shards e mostra distribuição por config."""
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
                logger.warning(f"Shard vazio: {s.name}")
                errors += 1
        except Exception as e:
            logger.error(f"Shard corrompido {s.name}: {e}")
            errors += 1
    logger.info(f"Verificacao: {len(shards)} shards | {total_rows:,} docs | {errors} erro(s)")

    dist_path = OUTPUT_DIR / "config_distribution.json"
    if dist_path.exists():
        dist = json.loads(dist_path.read_text(encoding="utf-8"))
        for cfg, n in dist.get("distribution", {}).items():
            lic = dist.get("license_by_config", {}).get(cfg, "?")
            logger.info(f"  {cfg:<28} {n:>8,} docs  [licenca: {lic}]")

    try:
        t = pq.read_table(shards[-1], columns=["text", "source", "score"])
        logger.info(f'Amostra ({shards[-1].name}): "{t["text"][0].as_py()[:80]}..." [score={t["score"][0].as_py():.2f}]')
    except Exception as e:
        logger.warning(f"Erro ao ler amostra: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Aquisicao do LegalPT (menos tesemo_v2)")
    ap.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS,
                     help=f"teto de documentos mantidos (default {DEFAULT_MAX_DOCS:,}; 0 = sem limite)")
    args = ap.parse_args()
    try:
        acquire(max_docs=args.max_docs)
        verify()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuario. Re-executar para retomar do ultimo shard/config.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
