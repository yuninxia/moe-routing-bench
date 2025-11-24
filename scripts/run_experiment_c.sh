#!/usr/bin/env bash
# Run Experiment C: compare multiple routing strategies at fixed hyperparameters.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
TRAIN_SCRIPT="${ROOT}/scripts/train_small.py"

if [[ ! -x "$PY" ]]; then
  echo "fatal: $PY not found or not executable; create .venv first." >&2
  exit 1
fi

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

# Common hyperparameters
MAX_STEPS=${MAX_STEPS:-1200}
EVAL_INTERVAL=${EVAL_INTERVAL:-200}
COMMON_ARGS=(
  --data "${ROOT}/data/tinystories_train.txt"
  --seq-len 256
  --batch-size 32
  --layers 4
  --dim 256
  --heads 4
  --num-experts 8
  --top-k 2
  --ffn-mult 4
  --lr 3e-4
  --warmup-steps 50
  --max-steps "${MAX_STEPS}"
  --eval-interval "${EVAL_INTERVAL}"
  --dtype bf16
  --device cuda
  --seed 17
  --renorm-after-drop
  --capacity-factor 1.25
)

# strategy name -> outdir suffix
RUNS=(
  "top1"
  "topk-hard"
  "softk"
  "hash"
  "expert-choice"
)

total=${#RUNS[@]}
idx=1
for strat in "${RUNS[@]}"; do
  name="expC_${strat//-/_}"
  echo "[$idx/$total] Starting ${name} (strategy=${strat})"
  start=$(date +%s)
  if $USE_DDP; then
    CMD=(
      "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS"
      "$TRAIN_SCRIPT"
      --distributed
      "${COMMON_ARGS[@]}"
      --strategy "$strat"
      --outdir "${ROOT}/runs/${name}"
    )
  else
    CMD=(
      "$PY" "$TRAIN_SCRIPT"
      "${COMMON_ARGS[@]}"
      --strategy "$strat"
      --outdir "${ROOT}/runs/${name}"
    )
  fi

  if [[ -n "$GPU_IDS_ENV" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_IDS_ENV" "${CMD[@]}"
  else
    "${CMD[@]}"
  fi

  end=$(date +%s)
  echo "[$idx/$total] Finished ${name} in $((end - start))s"
  idx=$((idx + 1))
done

echo "Experiment C runs completed."
