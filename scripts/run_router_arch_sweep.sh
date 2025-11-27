#!/usr/bin/env bash
# Sweep router_arch ∈ {linear, mlp, mlp_hadamard} and strategy ∈ {softk, expert-choice}
# Small config (E=8) to compare PPL vs tokens/s with consistent settings.

set -euo pipefail

# Default hyperparameters (override via env)
: "${GPU_IDS:=0,1}"
: "${OUTDIR:=runs/router_arch_sweep}"
: "${RESULTS:=results/router_arch_sweep}"
: "${MAX_STEPS:=1200}"
: "${EVAL_INTERVAL:=200}"
: "${SEQ_LEN:=256}"
: "${BATCH_SIZE:=32}"
: "${DIM:=256}"
: "${HEADS:=4}"
: "${LAYERS:=2}"
: "${NUM_EXPERTS:=8}"
: "${TOP_K:=2}"
: "${CAPACITY_FACTOR:=1.25}"
: "${LR:=3e-4}"
: "${WARMUP:=100}"
: "${FFN_MULT:=4}"
: "${DROPOUT:=0.1}"
: "${TOKENIZER:=}"  # set to HF name for BPE, leave empty for char-level
: "${RENORM_AFTER_DROP:=}" # set to 1 to enable

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
WORLD_SIZE=${#GPUS[@]}

echo "=============================================="
echo "Router Arch Sweep"
echo "GPUs: $GPU_IDS (world_size=$WORLD_SIZE)"
echo "Outdir: $OUTDIR"
echo "Results: $RESULTS"
echo "Set DRY_RUN=1 to list actions without running."
echo "=============================================="

mkdir -p "$OUTDIR" "$RESULTS"

run_cfg() {
  local arch="$1" strat="$2"
  local name="arch_${arch}_${strat}"
  local run_dir="$OUTDIR/$name"
  mkdir -p "$run_dir"

  # Skip if this run already finished (done flag or max step >= MAX_STEPS).
  if [[ -f "$run_dir/train_log.jsonl" ]]; then
    if grep -q '"done": true' "$run_dir/train_log.jsonl"; then
      echo "[SKIP] $name already completed (done flag)"
      return
    fi
    LOGFILE="$run_dir/train_log.jsonl"
    max_step=$(LOGFILE="$LOGFILE" python - <<'PY'
import json, os
log = os.environ["LOGFILE"]
mx = 0
with open(log, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            row = json.loads(line)
        except Exception:
            continue
        if 'step' in row:
            mx = max(mx, int(row['step']))
print(mx)
PY
)
    if [[ "${max_step:-0}" -ge "${MAX_STEPS:-0}" ]]; then
      echo "[SKIP] $name reached step $max_step (>= MAX_STEPS)"
      return
    fi
  fi

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY-RUN] would run $name"
    return
  fi

  echo "[RUN] $name"
  CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun --standalone --nproc_per_node=$WORLD_SIZE \
    scripts/train_small.py \
      --distributed \
      --data data/tinystories_train.txt \
      --outdir "$run_dir" \
      --seq-len $SEQ_LEN \
      --batch-size $BATCH_SIZE \
      --layers $LAYERS \
      --dim $DIM \
      --heads $HEADS \
      --dropout $DROPOUT \
      --num-experts $NUM_EXPERTS \
      --top-k $TOP_K \
      --strategy $strat \
      --capacity-factor $CAPACITY_FACTOR \
      --ffn-mult $FFN_MULT \
      --lr $LR \
      --warmup-steps $WARMUP \
      --max-steps $MAX_STEPS \
      --eval-interval $EVAL_INTERVAL \
      --dtype bf16 \
      --router-arch $arch \
      ${TOKENIZER:+--tokenizer $TOKENIZER} \
      ${RENORM_AFTER_DROP:+--renorm-after-drop}
}

ARCHES=(linear mlp mlp_hadamard)
STRATS=(softk expert-choice top1 topk-hard hash)

for arch in "${ARCHES[@]}"; do
  for strat in "${STRATS[@]}"; do
    run_cfg "$arch" "$strat"
  done
done

echo "=============================================="
echo "Sweep completed. Summaries live under $OUTDIR"
echo "To plot: python scripts/plot_router_arch_sweep.py --runs '$OUTDIR/arch_*' --out $RESULTS"
echo "=============================================="
