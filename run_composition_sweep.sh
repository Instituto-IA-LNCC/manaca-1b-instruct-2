#!/usr/bin/env bash
set -euo pipefail

MODEL="$(realpath -m /data/brunolsm/manaca-checkpoints/manaca-1b-instruct-2-full)"
CKPT_DIR=/data/brunolsm/manaca-checkpoints
LOGDIR=sweep_logs
mkdir -p "$LOGDIR"

declare -A PCTS=( [10]=202 [20]=455 [30]=780 [40]=1213 [50]=1819 [60]=2729 )

for pct in 10 20 30 40 50 60; do
  ngen=${PCTS[$pct]}
  echo "############################################"
  echo "### Sweep p${pct}% (N_GENERAL=${ngen}) -- $(date)"
  echo "############################################"

  docker run --rm -i --gpus device=0 \
    --sysctl net.ipv6.conf.all.disable_ipv6=1 --sysctl net.ipv6.conf.default.disable_ipv6=1 \
    --user "$(id -u):$(id -g)" \
    -e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v "$MODEL":/m:ro -v "$(pwd)":/work -w /work \
    manaca-lmeval:latest python sft/gen_safety_sft.py \
      --model /m --out sft_safety/safety_sft_sweep_p${pct}.jsonl \
      --seeds dpo/dpo_seeds/safety_prompts_v2.jsonl \
      --benign sft_safety/benign_help_v3.jsonl \
      --k_refuse 4 --general_data data/sft_instruct2/manaca_sft.jsonl --n_general ${ngen} \
    2>&1 | tee "$LOGDIR/gen_p${pct}.log"

  CKPT_DIR=$CKPT_DIR \
  ACCELERATE_CONFIG=configs/accelerate_config_ddp.yaml \
  ./scripts/docker/run_sft.sh sft \
    --model_name_or_path /workspace/checkpoints/manaca-1b-instruct-2-full \
    --data_files sft_safety/safety_sft_sweep_p${pct}.jsonl \
    --output_dir /workspace/checkpoints/manaca-1b-instruct-2-sweep-p${pct} \
    --lora_r 32 --lora_alpha 64 \
    --num_train_epochs 3 --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 --save_steps 200 \
    2>&1 | tee "$LOGDIR/train_p${pct}.log"

  M="${CKPT_DIR}/manaca-1b-instruct-2-sweep-p${pct}-merged"
  MODEL=$M LABEL=sweep-p${pct} ./scripts/eval/run_overrefusal_pt.sh \
    --prompts bench/safety_pt/heldout.jsonl --out_dir bench/safety_pt/answers \
    2>&1 | tee "$LOGDIR/eval_seguranca_p${pct}.log"
  MODEL=$M LABEL=sweep-p${pct} ./scripts/eval/run_overrefusal_pt.sh \
    2>&1 | tee "$LOGDIR/eval_overrefusal_p${pct}.log"

  echo "### p${pct}% concluido -- $(date)"
done

echo ""
echo "===== RESUMO DA VARREDURA ====="
for pct in 10 20 30 40 50 60; do
  echo "--- p${pct}% ---"
  grep -A2 "\[over-refusal\] modelo=sweep-p${pct}" "$LOGDIR/eval_seguranca_p${pct}.log" || true
  grep -A3 "\[over-refusal\] modelo=sweep-p${pct}" "$LOGDIR/eval_overrefusal_p${pct}.log" || true
done
