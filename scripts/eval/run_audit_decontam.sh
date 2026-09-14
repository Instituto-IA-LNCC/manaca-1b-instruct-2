#!/usr/bin/env bash
# =============================================================================
# Manaca-1B - Auditoria manual dos achados de decontaminate.py
# -----------------------------------------------------------------------------
# Mesma imagem/cache do run_decontaminate.sh. Roda DEPOIS dele, sobre o
# decontamination-report.json ja gerado.
#
# Uso:
#   ./scripts/eval/run_audit_decontam.sh                    # todos os achados
#   ./scripts/eval/run_audit_decontam.sh --banco enem_pt     # so um banco
#
# Overrides: EVAL_IMAGE, HF_CACHE, HF_TOKEN, TRAIN_FILE, REPORT.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${EVAL_IMAGE:-manaca-lmeval:latest}"

TRAIN_FILE="${TRAIN_FILE:-data/sft_instruct2/manaca_sft.jsonl}"
TRAIN_FILE="$(realpath -m "$TRAIN_FILE")"
[ -f "$TRAIN_FILE" ] || { echo "[ERRO] corpus de treino nao encontrado: $TRAIN_FILE"; exit 1; }

REPORT="${REPORT:-docs/evaluation/decontamination-report.json}"
[ -f "$REPORT" ] || { echo "[ERRO] relatorio nao encontrado: $REPORT (rode run_decontaminate.sh antes)"; exit 1; }

HF_CACHE="${HF_CACHE:-$HOME/hf_cache_decontam}"; HF_CACHE="$(realpath -m "$HF_CACHE")"; mkdir -p "$HF_CACHE"

docker run --rm -i \
  --sysctl net.ipv6.conf.all.disable_ipv6=1 --sysctl net.ipv6.conf.default.disable_ipv6=1 \
  --user "$(id -u):$(id -g)" \
  -e HF_HOME=/hf -e HF_DATASETS_TRUST_REMOTE_CODE=1 -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$HF_CACHE":/hf \
  -v "$(dirname "$TRAIN_FILE")":/train_dir:ro \
  -v "$REPO":/work -w /work/scripts/eval \
  "$IMAGE" python3 audit_decontam.py \
    --report "/work/$REPORT" \
    --train "/train_dir/$(basename "$TRAIN_FILE")" "$@"
