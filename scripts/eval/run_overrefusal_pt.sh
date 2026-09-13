#!/usr/bin/env bash
# =============================================================================
# Manaca-1B - Sonda de OVER-REFUSAL (recusa excessiva) dentro do docker (GPU)
# -----------------------------------------------------------------------------
# Mede quanto o modelo recusa pedidos BENIGNOS. Contrapeso do DPO de seguranca:
# rode ANTES (no instruct-v2 SFT) e DEPOIS (no DPO) e compare. Se a recusa em
# pedidos benignos subir, o DPO virou recusador demais e regrediu.
#
# Uso:
#   ./scripts/eval/run_overrefusal_pt.sh                       # v2 SFT (padrao)
#   MODEL=/data/.../manaca-1b-instruct-v2-dpo2-merged LABEL=manaca-instruct-v2-dpo2 \
#       ./scripts/eval/run_overrefusal_pt.sh                   # o DPO
#   GPUS=device=1 ./scripts/eval/run_overrefusal_pt.sh         # escolhe a GPU
#
#   HUB (baixa do Hugging Face, ex.: baselines do nucleo obrigatorio):
#     HFID=TucanoBR/Tucano-1b1-Instruct LABEL=tucano-1b1-instruct \
#       ./scripts/eval/run_overrefusal_pt.sh --prompts bench/safety_pt/heldout_v2.jsonl \
#       --out_dir bench/safety_pt/answers
#   Cada baseline usa o SEU chat_template (run.py --prompt_style auto, default),
#   igual ao MT-Bench-PT -- nao force o Alpaca-PT do Manaca em modelo nenhum.
#
# Overrides: EVAL_IMAGE, MODEL, LABEL, GPUS, NO_GPU, HFID, HF_CACHE, HF_TOKEN.
# Args extras vao para run.py (ex.: --prompts, --out_dir, --prompt_style).
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${EVAL_IMAGE:-manaca-lmeval:latest}"

GPU_FLAG=(--gpus "${GPUS:-all}"); [ "${NO_GPU:-0}" = "1" ] && GPU_FLAG=()
NET_FLAG=(--sysctl net.ipv6.conf.all.disable_ipv6=1 --sysctl net.ipv6.conf.default.disable_ipv6=1)

MODEL_MOUNT=(); CACHE_MOUNT=(); ENV_FLAGS=()
if [ -n "${HFID:-}" ]; then
  MODEL_ARG="$HFID"
  LABEL="${LABEL:-$(basename "$HFID")}"
  HF_CACHE="${HF_CACHE:-$HOME/hf_cache_mtbench}"; HF_CACHE="$(realpath -m "$HF_CACHE")"; mkdir -p "$HF_CACHE"
  CACHE_MOUNT=(-v "$HF_CACHE":/hf)
  ENV_FLAGS=(-e HF_HOME=/hf -e HF_TOKEN="${HF_TOKEN:-}")   # online: baixa do HF
  echo "[over-refusal] modo=HUB  hfid=$HFID  rotulo=$LABEL  cache=$HF_CACHE"
else
  MODEL="${MODEL:-/data/manaca/checkpoints/manaca-1b-instruct-v2-full}"
  MODEL="$(realpath -m "$MODEL")"
  MODEL_ARG="/m"
  LABEL="${LABEL:-manaca-instruct-v2}"
  MODEL_MOUNT=(-v "$MODEL":/m:ro)
  ENV_FLAGS=(-e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1)  # offline
  [ -d "$MODEL" ] || { echo "[ERRO] modelo nao encontrado: $MODEL"; exit 1; }
  echo "[over-refusal] modo=LOCAL  imagem=$IMAGE modelo=$MODEL label=$LABEL"
fi

docker run --rm -i "${GPU_FLAG[@]}" "${NET_FLAG[@]}" \
  --user "$(id -u):$(id -g)" \
  "${ENV_FLAGS[@]}" \
  "${MODEL_MOUNT[@]}" "${CACHE_MOUNT[@]}" \
  -v "$REPO":/work -w /work \
  "$IMAGE" python bench/overrefusal_pt/run.py --model "$MODEL_ARG" --label "$LABEL" "$@"
