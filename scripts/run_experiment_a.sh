#!/usr/bin/env bash
# Run Experiment A sweeps (top1/softk × capacity) with progress output.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
TRAIN_SCRIPT="${ROOT}/scripts/train_small.py"

if [[ ! -x "$PY" ]]; then
  echo "fatal: $PY not found or not executable; create .venv first." >&2
  exit 1
fi

# Common hyperparameters for all runs (tweak here if needed).
COMMON_ARGS=(
  --data "${ROOT}/data/tinystories_train.txt"
  --seq-len 256
  --batch-size 32
  --layers 4
  --dim 256
  --heads 4
  --num-experts 8
  --ffn-mult 4
  --lr 3e-4
  --warmup-steps 50
  --max-steps 600
  --eval-interval 100
  --dtype bf16
  --device cuda
  --seed 17
  --renorm-after-drop
)

# Optional multi-GPU DDP: set GPU_IDS="0,1,2,3" to enable.
GPU_IDS_ENV="${GPU_IDS:-}"
if [[ -n "$GPU_IDS_ENV" ]]; then
  IFS=',' read -ra GPU_LIST <<<"$GPU_IDS_ENV"
else
  GPU_LIST=()
fi
NUM_GPUS=${#GPU_LIST[@]}
USE_DDP=false
if (( NUM_GPUS > 1 )); then
  USE_DDP=true
  echo "[info] Using DDP with GPUs: ${GPU_IDS_ENV} (world_size=${NUM_GPUS})"
elif [[ -n "$GPU_IDS_ENV" ]]; then
  echo "[info] Using single GPU: ${GPU_IDS_ENV}"
fi

# (name, strategy, top_k, capacity_factor)
RUNS=(
  "top1_cf1.0 top1 1 1.0"
  "top1_cf1.25 top1 1 1.25"
  "softk_k2_cf1.0 softk 2 1.0"
  "softk_k2_cf1.25 softk 2 1.25"
  "topkhard_k2_cf1.0 topk-hard 2 1.0"
  "topkhard_k2_cf1.25 topk-hard 2 1.25"
)

total=${#RUNS[@]}
idx=1
for run in "${RUNS[@]}"; do
  read -r name strategy topk cf <<<"$run"
  echo "[$idx/$total] Starting ${name} (strategy=${strategy}, top_k=${topk}, cf=${cf})"
  start=$(date +%s)
  if $USE_DDP; then
    CUDA_VISIBLE_DEVICES="$GPU_IDS_ENV" "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
      "$TRAIN_SCRIPT" \
      --distributed \
      "${COMMON_ARGS[@]}" \
      --top-k "$topk" \
      --strategy "$strategy" \
      --capacity-factor "$cf" \
      --outdir "${ROOT}/runs/${name}"
  else
    CUDA_VISIBLE_DEVICES="$GPU_IDS_ENV" "$PY" "$TRAIN_SCRIPT" \
      "${COMMON_ARGS[@]}" \
      --top-k "$topk" \
      --strategy "$strategy" \
      --capacity-factor "$cf" \
      --outdir "${ROOT}/runs/${name}"
  fi
  end=$(date +%s)
  echo "[$idx/$total] Finished ${name} in $((end - start))s"
  idx=$((idx + 1))
done

echo "All runs completed."
