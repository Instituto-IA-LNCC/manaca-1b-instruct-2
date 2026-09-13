#!/usr/bin/env bash
# =============================================================================
# Manaca-1B - Constroi manaca-lmeval-qwen3: variante do manaca-lmeval com
# transformers novo o bastante pra reconhecer a arquitetura Qwen3
# -----------------------------------------------------------------------------
# O manaca-lmeval:latest fixa transformers==4.46.3 (nov/2024) DE PROPOSITO --
# ver requirements/lmeval.txt: um transformers mais novo quebra contra o torch
# da imagem base (nvcr.io/nvidia/pytorch:24.12) com
#   ImportError: cannot import name 'TransformGetItemToIndex' from torch._dynamo
# Suporte a Qwen3 (arquitetura do Tucano2-qwen, nucleo obrigatorio) so chegou
# no transformers 4.51 (abr/2025) -- nao da pra rodar esse baseline na imagem
# pinada sem quebrar os outros baselines/o proprio Manaca, que ja funcionam
# nela.
#
# Em vez de mexer no manaca-lmeval:latest, este script cria uma imagem
# SEPARADA (manaca-lmeval-qwen3:latest) com o MINIMO bump de transformers que
# ja suporta Qwen3 (4.51.3 -- a primeira release com suporte, nao "o mais
# novo", pra isolar UMA variavel por vez e nao arriscar quebrar outra coisa
# junto). Mesmo mecanismo de build_lmeval_image.sh (pip dentro de um
# container ja existente + commit, porque `docker build`/`docker pull` de uma
# imagem base nova quebram no DNS da HPC).
#
# Uso:
#   ./scripts/eval/build_lmeval_qwen3_image.sh
#   EVAL_IMAGE=manaca-lmeval-qwen3:latest HFID=Polygl0t/Tucano2-qwen-1.5B-Instruct \
#     LABEL=tucano2-qwen-1b5-instruct ./scripts/eval/run_mtbench_pt.sh
#
# Overrides: BASE_IMAGE (default manaca-lmeval:latest), OUT_IMAGE
# (manaca-lmeval-qwen3:latest), TRANSFORMERS_VERSION (4.51.3), PYPI_IP.
#
# Se isso quebrar com o MESMO erro de torch._dynamo (4.51.3 ja exige um torch
# mais novo que o da imagem base), o proximo passo seria atualizar tambem o
# torch DENTRO DESTA imagem separada (nunca no manaca-lmeval original) -- pare
# e confira o CUDA do driver da GPU antes de tentar, um torch novo demais pode
# nao casar com o driver instalado na netuno.
# =============================================================================
set -euo pipefail

BASE="${BASE_IMAGE:-manaca-lmeval:latest}"
OUT="${OUT_IMAGE:-manaca-lmeval-qwen3:latest}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.51.3}"
PYPI_IP="${PYPI_IP:-151.101.0.223}"
NAME="lmeval-qwen3-build-$$"
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT

echo "[build] $BASE -> $OUT  (transformers==$TRANSFORMERS_VERSION, so pra rodar Qwen3/Tucano2-qwen)"
docker run --name "$NAME" --gpus all \
  --sysctl net.ipv6.conf.all.disable_ipv6=1 \
  --sysctl net.ipv6.conf.default.disable_ipv6=1 \
  -e PYPI_IP="$PYPI_IP" \
  -e TRANSFORMERS_VERSION="$TRANSFORMERS_VERSION" \
  -e PIP_EXTRA_INDEX_URL= \
  -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
  "$BASE" \
  bash -c '
    set -e
    IP="${PYPI_IP:-151.101.0.223}"
    printf "%s pypi.org\n%s files.pythonhosted.org\n" "$IP" "$IP" >> /etc/hosts
    find /etc /root /usr -name pip.conf -not -path "*/site-packages/*" -delete 2>/dev/null || true
    unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL
    for i in 1 2 3 4 5; do
      echo "== tentativa pip $i =="
      pip install --break-system-packages --index-url https://pypi.org/simple \
        --no-cache-dir --retries 20 --timeout 120 \
        "transformers==${TRANSFORMERS_VERSION}" && break
      echo "pip falhou; nova tentativa em 8s..."; sleep 8
    done
    python -c "
import transformers, torch
print(\"OK: transformers\", transformers.__version__, \"| torch\", torch.__version__)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
assert \"qwen3\" in CONFIG_MAPPING, \"qwen3 ainda nao reconhecido -- precisa de outra versao\"
print(\"OK: qwen3 reconhecido pelo transformers\")
"
  '

echo "[build] commit -> $OUT"
docker commit "$NAME" "$OUT"
echo "[build] pronto: $OUT"
echo "[build] use SO nos comandos do Tucano2-qwen (EVAL_IMAGE=$OUT); os outros"
echo "[build] baselines e o Manaca continuam na imagem original (manaca-lmeval:latest)."
