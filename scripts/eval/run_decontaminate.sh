#!/usr/bin/env bash
# =============================================================================
# Manaca-1B - Checagem de decontaminação (n-gram) dentro do docker
# -----------------------------------------------------------------------------
# Roda scripts/eval/decontaminate.py dentro da imagem de avaliação (que já tem
# `datasets`, dependência do lm-eval-harness) — não precisa de GPU, só baixa os
# benchmarks do HF e varre o corpus de treino local uma vez.
#
# Uso:
#   ./scripts/eval/build_lmeval_image.sh          # se a imagem ainda nao existir
#   TRAIN_FILE=data/sft_instruct2/manaca_sft.jsonl \
#     ./scripts/eval/run_decontaminate.sh
#
# Overrides: EVAL_IMAGE, HF_CACHE, HF_TOKEN, TRAIN_FILE, OUT.
# Args extras vao para decontaminate.py (ex.: --n 10).
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${EVAL_IMAGE:-manaca-lmeval:latest}"

TRAIN_FILE="${TRAIN_FILE:-data/sft_instruct2/manaca_sft.jsonl}"
TRAIN_FILE="$(realpath -m "$TRAIN_FILE")"
[ -f "$TRAIN_FILE" ] || { echo "[ERRO] corpus de treino nao encontrado: $TRAIN_FILE"; exit 1; }

OUT="${OUT:-docs/evaluation/decontamination-report}"

# Cache PROPRIO (do usuario). Nao reuse $HOME/hf_cache_eval: ele foi populado
# pelas rodadas do lm-eval que rodavam docker como root, e com --user (abaixo) o
# processo nao consegue escrever la (PermissionError) -- mesma armadilha ja
# documentada em run_mtbench_pt.sh.
HF_CACHE="${HF_CACHE:-$HOME/hf_cache_decontam}"; HF_CACHE="$(realpath -m "$HF_CACHE")"; mkdir -p "$HF_CACHE"

echo "[decontam] imagem=$IMAGE  treino=$TRAIN_FILE  cache=$HF_CACHE"

docker run --rm -i \
  --sysctl net.ipv6.conf.all.disable_ipv6=1 --sysctl net.ipv6.conf.default.disable_ipv6=1 \
  --user "$(id -u):$(id -g)" \
  -e HF_HOME=/hf -e HF_DATASETS_TRUST_REMOTE_CODE=1 -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$HF_CACHE":/hf \
  -v "$(dirname "$TRAIN_FILE")":/train_dir:ro \
  -v "$REPO":/work -w /work \
  "$IMAGE" python3 scripts/eval/decontaminate.py \
    --train "/train_dir/$(basename "$TRAIN_FILE")" \
    --out "$OUT" "$@"
