#!/usr/bin/env bash
# Quick DDP sanity check for expert-choice routing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "fatal: ${PY} not found or not executable; create .venv first." >&2
  exit 1
fi

GPU_IDS=${GPU_IDS:-0,1,2,3}
MAX_STEPS=${MAX_STEPS:-50}
EVAL_INTERVAL=${EVAL_INTERVAL:-0}  # default: eval only at end
SEQ_LEN=${SEQ_LEN:-128}
BATCH_SIZE=${BATCH_SIZE:-16}

IFS=',' read -ra GPU_LIST <<<"$GPU_IDS"
NUM_GPUS=${#GPU_LIST[@]}

echo "[info] Running expert-choice DDP debug with GPUs=${GPU_IDS}, MAX_STEPS=${MAX_STEPS}, EVAL_INTERVAL=${EVAL_INTERVAL}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
torchrun --standalone --nproc_per_node="${NUM_GPUS}" "${ROOT}/scripts/train_small.py" \
  --distributed \
  --device cuda \
  --data "${ROOT}/data/tinystories_train.txt" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --layers 4 \
  --dim 256 \
  --heads 4 \
  --num-experts 8 \
  --top-k 2 \
  --strategy expert-choice \
  --capacity-factor 1.25 \
  --dtype bf16 \
  --lr 3e-4 \
  --warmup-steps 50 \
  --max-steps "${MAX_STEPS}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --outdir "${ROOT}/runs/expC_expert_choice_debug_ddp"
