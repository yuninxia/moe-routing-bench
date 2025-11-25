#!/usr/bin/env bash
# Larger Scale Validation Experiment
# Tests if routing strategy findings transfer to bigger models
# Config: E=32, dim=512, layers=4 (closer to real MoE scale)

set -euo pipefail

GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -ra GPU_ARR <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARR[@]}"

MAX_STEPS="${MAX_STEPS:-2000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-200}"

DATA_FILE="${DATA_FILE:-data/tinystories_train.txt}"
OUTDIR_ROOT="${OUTDIR_ROOT:-runs}"

# Model configuration (larger scale)
# Increased E=32 for more realistic MoE scaling
NUM_EXPERTS="${NUM_EXPERTS:-32}"
DIM="${DIM:-512}"
LAYERS="${LAYERS:-4}"
FFN_MULT="${FFN_MULT:-4}"
TOP_K="${TOP_K:-2}"
SEQ_LEN="${SEQ_LEN:-256}"
BATCH_SIZE="${BATCH_SIZE:-32}"
# CF=1.5: balanced between drop rate and throughput (per colleague's suggestion)
CAPACITY_FACTOR="${CAPACITY_FACTOR:-1.5}"
DTYPE="${DTYPE:-bf16}"

# Training hyperparameters (adjusted for larger model stability)
LR="${LR:-1e-4}"              # Lower LR for stability
WARMUP_STEPS="${WARMUP_STEPS:-200}"  # Longer warmup

# Strategies to test (can override via STRATEGIES env)
# Default covers all routers to check scaling trends end-to-end.
STRATEGIES=(${STRATEGIES:-top1 topk-hard softk hash expert-choice})

echo "=============================================="
echo "Larger Scale Validation Experiment"
echo "=============================================="
echo "GPUs: $GPU_IDS ($NUM_GPUS GPUs)"
echo "Model: E=$NUM_EXPERTS, dim=$DIM, layers=$LAYERS, ffn_mult=$FFN_MULT"
echo "Routing: base_top_k=$TOP_K, CF=$CAPACITY_FACTOR"
echo "Training: $MAX_STEPS steps, eval every $EVAL_INTERVAL"
echo "Hyperparams: lr=$LR, warmup=$WARMUP_STEPS"
echo "Strategies: ${STRATEGIES[*]}"
echo "=============================================="

for STRATEGY in "${STRATEGIES[@]}"; do
    # Strategy-specific top_k (top1 forces k=1, others use TOP_K)
    if [[ "$STRATEGY" == "top1" ]]; then
        K_THIS=1
    else
        K_THIS="$TOP_K"
    fi

    RUN_NAME="larger_${STRATEGY//-/_}_E${NUM_EXPERTS}_dim${DIM}_L${LAYERS}_cf${CAPACITY_FACTOR}_K${K_THIS}"
    OUTDIR="${OUTDIR_ROOT}/${RUN_NAME}"

    if [[ -d "$OUTDIR" ]]; then
        echo "[SKIP] $RUN_NAME already exists"
        continue
    fi

    echo ""
    echo "[RUN] $RUN_NAME"
    echo "Strategy: $STRATEGY, CF: $CAPACITY_FACTOR, LR: $LR"

    START_TIME=$(date +%s)

    CUDA_VISIBLE_DEVICES="$GPU_IDS" torchrun --standalone --nproc_per_node="$NUM_GPUS" \
        scripts/train_small.py \
        --distributed \
        --device cuda \
        --data "$DATA_FILE" \
        --num-experts "$NUM_EXPERTS" \
        --dim "$DIM" \
        --layers "$LAYERS" \
        --ffn-mult "$FFN_MULT" \
        --top-k "$K_THIS" \
        --strategy "$STRATEGY" \
        --capacity-factor "$CAPACITY_FACTOR" \
        --seq-len "$SEQ_LEN" \
        --batch-size "$BATCH_SIZE" \
        --max-steps "$MAX_STEPS" \
        --eval-interval "$EVAL_INTERVAL" \
        --lr "$LR" \
        --warmup-steps "$WARMUP_STEPS" \
        --dtype "$DTYPE" \
        --outdir "$OUTDIR"

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo "[DONE] $RUN_NAME finished in ${ELAPSED}s"
done

echo ""
echo "=============================================="
echo "Larger scale experiment completed!"
echo "=============================================="

# Summarize results
echo "Summarizing results..."
python scripts/summarize_runs.py --runs "${OUTDIR_ROOT}/larger_*" --out results/larger_scale_summary.csv

echo ""
echo "Results saved to: results/larger_scale_summary.csv"
echo ""
cat results/larger_scale_summary.csv
