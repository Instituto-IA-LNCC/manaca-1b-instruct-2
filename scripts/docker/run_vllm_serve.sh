#!/usr/bin/env bash
# =============================================================================
# Manacá — sobe um servidor vLLM (OpenAI-compatible) numa ÚNICA GPU do Netuno
# =============================================================================
# Por que 1 GPU por instância, e não --tensor-parallel-size 2: as 2 RTX A5000
# do Netuno NÃO têm NVLink (ver .env.example, comentário de GPUS_PER_NODE).
# Tensor parallelism sem NVLink faz all-reduce a cada camada via PCIe, o que
# vira gargalo sério pra throughput de geração em lote. Em vez disso, suba
# DUAS instâncias independentes (uma por GPU) e distribua as chamadas entre
# as duas portas no client (round-robin) -- escala 2x sem overhead de
# sincronização entre GPUs. Um modelo ~32B em AWQ (4-bit) cabe sozinho em
# 24 GB, então essa abordagem não perde capacidade de modelo.
#
# Uso:
#   GPU=0 PORT=8000 ./scripts/docker/run_vllm_serve.sh
#   GPU=1 PORT=8001 ./scripts/docker/run_vllm_serve.sh
#   MODEL=Qwen/Qwen2.5-14B-Instruct-AWQ ./scripts/docker/run_vllm_serve.sh   # fallback menor
#
# Acompanhar (baixa o modelo na 1a vez, pode levar minutos):
#   docker logs -f manaca-vllm-<GPU>
# Testar (depois de "Uvicorn running on ..." aparecer no log):
#   curl http://localhost:<PORT>/v1/models
# Parar:
#   docker stop manaca-vllm-<GPU> && docker rm manaca-vllm-<GPU>
#
# NÃO TESTADO AINDA neste cluster -- primeira vez rodando vLLM aqui. Rode e
# cole o log se der erro (driver/CUDA da imagem vLLM incompatível com o
# driver do host é a causa mais provável de falha na largada).
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

load_env_defaults() {
    [ -f "$1" ] || return 0
    local line key
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|\#*) continue ;; esac
        key=${line%%=*}; key=${key// /}
        case "$key" in ''|*[!A-Za-z0-9_]*) continue ;; esac
        [ -z "${!key+x}" ] && export "${key}=${line#*=}"
    done < "$1"
}
load_env_defaults .env

GPU="${GPU:-0}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
# `latest` até confirmarmos uma versão que funciona neste cluster; depois de
# validar, PINE uma tag concreta aqui (ex.: v0.6.6) para reprodutibilidade --
# mesmo motivo pelo qual o resto do projeto evita "auto" em configs (ver
# scripts/docker/run_sft.sh, comentário do accelerate).
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
HF_CACHE_DIR="$(realpath -m "${HF_CACHE_DIR:-./hf-cache}")"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
mkdir -p "${HF_CACHE_DIR}"

CNAME="manaca-vllm-${GPU}"

# Mesmo fix de DNS usado em run_posttrain.sh/run_sft.sh (resolv.conf do host
# via resolvectl -- necessário pra baixar o modelo do HuggingFace).
RESOLV_MOUNT=()
RESOLV_CONF="$(realpath -m "${HF_CACHE_DIR}/.manaca-resolv-vllm-${GPU}.conf")"
if { if [ -n "${MANACA_DNS:-}" ]; then
        for ns in ${MANACA_DNS}; do echo "nameserver ${ns}"; done
     else
        resolvectl dns 2>/dev/null \
          | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}|([0-9A-Fa-f]{1,4}:){2,}[0-9A-Fa-f:]*' \
          | grep -vE '^(127\.|::1$)' | sort -u | sed 's/^/nameserver /'
        echo 'nameserver 127.0.0.53'
     fi
     echo 'options single-request-reopen timeout:2 attempts:5'; } > "${RESOLV_CONF}" 2>/dev/null; then
  RESOLV_MOUNT=(-v "${RESOLV_CONF}:/etc/resolv.conf:ro")
fi

echo "[run_vllm_serve] gpu=${GPU} port=${PORT} model=${MODEL} image=${VLLM_IMAGE}"
echo "[run_vllm_serve] container=${CNAME} (2o plano -- sobrevive ao logout do SSH)"

docker rm -f "${CNAME}" >/dev/null 2>&1 || true

docker run -d --name "${CNAME}" \
  --gpus "device=${GPU}" --network host --ipc host --shm-size=16g \
  "${RESOLV_MOUNT[@]}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "${HF_CACHE_DIR}":/root/.cache/huggingface \
  "${VLLM_IMAGE}" \
  --model "${MODEL}" \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}"

echo "[run_vllm_serve]   logs:    docker logs -f ${CNAME}"
echo "[run_vllm_serve]   pronto quando aparecer: 'Uvicorn running on http://0.0.0.0:${PORT}'"
echo "[run_vllm_serve]   testar:  curl http://localhost:${PORT}/v1/models"
echo "[run_vllm_serve]   parar:   docker stop ${CNAME} && docker rm ${CNAME}"
