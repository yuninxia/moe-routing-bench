#!/usr/bin/env bash
# Quick PERFT Variants Comparison for Slides
# Runs all PERFT variants with limited steps to show convergence trends
#
# Usage: GPU_IDS=0,1,2,3 bash scripts/run_perft_variants_quick.sh
#
# Expected time: ~30-40 minutes total for all variants
# - perft_r_top2of4: ~8 min
# - perft_r_top1of4: ~8 min
# - perft_e: ~8 min
# - shared_adapter: ~8 min

set -uo pipefail
# Note: not using -e so that one failed experiment doesn't stop all others

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERFT_DIR="${ROOT}/3rdparty/PERFT-MoE"

# ============================================================
# Configuration
# ============================================================

GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -ra GPU_ARR <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARR[@]}"

# Model cache (use standard HF cache layout)
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/scratch/yx87/playground/moe-routing-bench/.cache/hf}"
mkdir -p "$MODEL_CACHE_DIR"
export HF_HOME="$MODEL_CACHE_DIR"
export HF_HUB_CACHE="$MODEL_CACHE_DIR"
export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
export HF_DATASETS_CACHE="$MODEL_CACHE_DIR/datasets"
# Avoid remote fetch once cached to stop shard lookup errors on restricted nets
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Allow running a subset: ONLY=\"perft_e\" or ONLY=\"perft_r_top2of4 perft_e\"
ONLY="${ONLY:-}"
# Sweep controls (for Fig.4-like trends)
RANK_LIST="${RANK_LIST:-16}"              # e.g., \"8 16 32\"
PEFT_EXPERTS_LIST="${PEFT_EXPERTS_LIST:-4}"  # total PEFT experts for PERFT-R
TOPK_LIST="${TOPK_LIST:-1 2}"             # activated PEFT experts

# Model (use local snapshot when offline)
SNAPSHOT_DIR="$(ls -1d "${MODEL_CACHE_DIR}/models--allenai--OLMoE-1B-7B-0924"/snapshots/* 2>/dev/null | head -1 || true)"
if [[ -z "$SNAPSHOT_DIR" ]]; then
    echo "[ERROR] Could not find local snapshot under ${MODEL_CACHE_DIR}/models--allenai--OLMoE-1B-7B-0924/snapshots"
    echo "[INFO] Please download with: HF_HOME=${MODEL_CACHE_DIR} bash scripts/download_olmoe_checkpoint.sh"
    exit 1
fi
BASE_MODEL="$SNAPSHOT_DIR"

# Quick training settings
MAX_STEPS="${MAX_STEPS:-500}"
EVAL_STEP="${EVAL_STEP:-100}"
SAVE_STEP="${SAVE_STEP:-500}"
BATCH_SIZE=16
MICRO_BATCH_SIZE=4
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
VAL_SET_SIZE=120

# Data
DATA_PATH="${PERFT_DIR}/commonsense/commonsense_170k.json"

# Output
RESULTS_DIR="${ROOT}/results/perft_variants"
mkdir -p "$RESULTS_DIR"

# ============================================================
# Setup
# ============================================================

cd "$PERFT_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PERFT_DIR}:${PYTHONPATH:-}"

# Pre-download the model to avoid race conditions in DDP
echo "[INFO] Ensuring model is fully cached..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'allenai/OLMoE-1B-7B-0924',
    cache_dir='${MODEL_CACHE_DIR}',
    local_files_only=False,
    allow_patterns=[
        'model-0000*.safetensors',
        'model.safetensors.index.json',
        'config.json',
        'tokenizer.*',
    ],
)
print('Model cache verified!')
"

echo "=============================================="
echo "PERFT Variants Quick Comparison"
echo "=============================================="
echo "GPUs: $GPU_IDS ($NUM_GPUS GPUs)"
echo "Max Steps: $MAX_STEPS per variant"
echo "Eval Every: $EVAL_STEP steps"
echo "Results: $RESULTS_DIR"
echo "=============================================="

# Check data exists
if [[ ! -f "$DATA_PATH" ]]; then
    echo "[ERROR] Training data not found: $DATA_PATH"
    echo "[INFO] Please run the full experiment first to prepare data"
    exit 1
fi

# ============================================================
# Run Function
# ============================================================

run_variant() {
    local name=$1
    shift
    local extra_args=("$@")

    local output_dir="${PERFT_DIR}/checkpoints/${name}"
    local log_file="${RESULTS_DIR}/${name}_train.log"

    echo ""
    echo "=============================================="
    echo "Running: $name"
    echo "Started: $(date)"
    echo "=============================================="

    mkdir -p "$output_dir"

    COMMON_ARGS=(
        --base_model "$BASE_MODEL"
        --data_path "$DATA_PATH"
        --output_dir "$output_dir"
        --batch_size "$BATCH_SIZE"
        --micro_batch_size "$MICRO_BATCH_SIZE"
        --max_steps "$MAX_STEPS"
        --learning_rate "$LEARNING_RATE"
        --cutoff_len 256
        --val_set_size "$VAL_SET_SIZE"
        --eval_step "$EVAL_STEP"
        --save_step "$SAVE_STEP"
        --adapter_type LoRA
        --lora_r 16
        --lora_alpha 32
        --dropout 0.05
    )

    local exit_code=0
    if (( NUM_GPUS > 1 )); then
        CUDA_VISIBLE_DEVICES="$GPU_IDS" torchrun \
            --standalone \
            --nproc_per_node="$NUM_GPUS" \
            commonsense/finetune.py "${COMMON_ARGS[@]}" "${extra_args[@]}" \
            2>&1 | tee "$log_file" || exit_code=$?
    else
        CUDA_VISIBLE_DEVICES="$GPU_IDS" python \
            commonsense/finetune.py "${COMMON_ARGS[@]}" "${extra_args[@]}" \
            2>&1 | tee "$log_file" || exit_code=$?
    fi

    if [[ $exit_code -ne 0 ]]; then
        echo "[WARNING] $name failed with exit code $exit_code, continuing..."
    else
        echo "[INFO] Completed: $name at $(date)"
    fi
}

# Check whether to run a variant based on ONLY filter
should_run() {
    local name=$1
    if [[ -z "$ONLY" ]]; then
        return 0
    fi
    for v in $ONLY; do
        if [[ "$v" == "$name" ]]; then
            return 0
        fi
    done
    return 1
}

# ============================================================
# Run All Variants
# ============================================================

START_TIME=$(date +%s)

# 1. PERFT-R sweep over experts/top-k and ranks
for experts in $PEFT_EXPERTS_LIST; do
    for topk in $TOPK_LIST; do
        for rank in $RANK_LIST; do
            name="perft_r_top${topk}of${experts}_r${rank}"
            if should_run "$name"; then
                run_variant "$name" \
                    --shared_routing_adapter True \
                    --shared_routing_adapter_num_experts "$experts" \
                    --shared_routing_adapter_num_experts_per_tok "$topk" \
                    --lora_r "$rank"
                echo "[INFO] Waiting 10s for clean process shutdown..."
                sleep 10
            fi
        done
    done
done

# 2. PERFT-E sweep over ranks (router shared from base MoE)
for rank in $RANK_LIST; do
    name="perft_e_r${rank}"
    if should_run "$name"; then
        run_variant "$name" \
            --embedded_routing_adapter True \
            --lora_r "$rank"
        echo "[INFO] Waiting 10s for clean process shutdown..."
        sleep 10
    fi
done

# 3. Shared Adapter sweep over ranks (no routing)
for rank in $RANK_LIST; do
    name="shared_adapter_r${rank}"
    if should_run "$name"; then
        run_variant "$name" \
            --shared_adapter True \
            --shared_adapter_num 1 \
            --lora_r "$rank"
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "All Variants Complete!"
echo "=============================================="
echo "Total time: $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds"
echo ""

# ============================================================
# Extract and Summarize Results
# ============================================================

echo "Extracting metrics to CSV..."

SUMMARY_FILE="${RESULTS_DIR}/variants_summary.csv"
echo "variant,step,train_loss,eval_loss" > "$SUMMARY_FILE"

for log_file in "${RESULTS_DIR}"/*_train.log; do
    [[ -e "$log_file" ]] || continue
    variant="$(basename "$log_file" "_train.log")"
    # Extract training losses using awk (avoids subshell issues)
    grep "'loss':" "$log_file" | grep -v "'eval_loss':" | awk -v var="$variant" '
    BEGIN { step = 0 }
    {
        step += 10
        match($0, /'\''loss'\'': ([0-9.]+)/, arr)
        if (arr[1] != "") {
            print var "," step "," arr[1] ","
        }
    }' >> "$SUMMARY_FILE"

    # Extract eval losses
    grep "'eval_loss':" "$log_file" | awk -v var="$variant" '
    {
        match($0, /'\''eval_loss'\'': ([0-9.]+)/, arr)
        if (arr[1] != "") {
            print var ",eval,," arr[1]
        }
    }' >> "$SUMMARY_FILE"
done

echo "Summary saved to: $SUMMARY_FILE"
echo "Log files: ${RESULTS_DIR}/*_train.log"
echo ""
echo "To plot results, run:"
echo "  python scripts/plot_perft_variants.py"
echo ""

cd "$ROOT"
