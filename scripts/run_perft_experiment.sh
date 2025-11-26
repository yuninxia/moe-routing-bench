#!/usr/bin/env bash
# PERFT-R Fine-tuning Experiment on OLMoE-1B-7B
# Supports both single-GPU and multi-GPU (DDP) training
#
# Usage:
#   Single GPU:  bash scripts/run_perft_experiment.sh
#   Multi-GPU:   GPU_IDS=0,1,2,3 bash scripts/run_perft_experiment.sh
#
# This script runs the PERFT-R (Parameter-Efficient Routed Fine-Tuning) method
# which adds an independent adapter MoE system alongside the frozen base MoE.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERFT_DIR="${ROOT}/3rdparty/PERFT-MoE"

# ============================================================
# Configuration (override via environment variables)
# ============================================================

# GPU setup
GPU_IDS="${GPU_IDS:-0}"
IFS=',' read -ra GPU_ARR <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARR[@]}"

# Model cache directory (use scratch for large models)
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/scratch/yx87/playground/models}"
mkdir -p "$MODEL_CACHE_DIR"
export HF_HOME="$MODEL_CACHE_DIR"
export HF_DATASETS_CACHE="$MODEL_CACHE_DIR/datasets"
# Note: TRANSFORMERS_CACHE is deprecated in transformers v5, use HF_HOME only

# Model
BASE_MODEL="${BASE_MODEL:-allenai/OLMoE-1B-7B-0924}"

# PERFT-R hyperparameters
ADAPTER_NUM_EXPERTS="${ADAPTER_NUM_EXPERTS:-4}"      # Number of adapter experts
ADAPTER_TOP_K="${ADAPTER_TOP_K:-2}"                  # Top-k routing for adapters
ADAPTER_TYPE="${ADAPTER_TYPE:-LoRA}"                 # LoRA or Parallel_Adapter
LORA_R="${LORA_R:-16}"                               # LoRA rank
LORA_ALPHA="${LORA_ALPHA:-32}"                       # LoRA alpha
DROPOUT="${DROPOUT:-0.05}"

# Training hyperparameters
BATCH_SIZE="${BATCH_SIZE:-16}"                       # Total batch size
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"            # Per-GPU micro batch
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
CUTOFF_LEN="${CUTOFF_LEN:-256}"                      # Max sequence length
VAL_SET_SIZE="${VAL_SET_SIZE:-120}"
EVAL_STEP="${EVAL_STEP:-80}"
SAVE_STEP="${SAVE_STEP:-80}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-${PERFT_DIR}/checkpoints/OLMoE-1B-7B.perft_r_top${ADAPTER_TOP_K}of${ADAPTER_NUM_EXPERTS}_r${LORA_R}}"

# Data path
DATA_PATH="${DATA_PATH:-${PERFT_DIR}/commonsense/commonsense_170k.json}"

# Evaluation datasets (space-separated)
EVAL_DATASETS="${EVAL_DATASETS:-boolq piqa}"

# ============================================================
# Setup
# ============================================================

echo "=============================================="
echo "PERFT-R Fine-tuning Experiment"
echo "=============================================="
echo "GPUs: $GPU_IDS ($NUM_GPUS GPUs)"
echo "Base Model: $BASE_MODEL"
echo "PERFT-R Config: Top-${ADAPTER_TOP_K}/${ADAPTER_NUM_EXPERTS} adapter experts"
echo "LoRA: r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "Training: ${NUM_EPOCHS} epochs, lr=${LEARNING_RATE}"
echo "Batch: total=${BATCH_SIZE}, micro=${MICRO_BATCH_SIZE}"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# Check if data exists
if [[ ! -f "$DATA_PATH" ]]; then
    echo "[WARNING] Training data not found: $DATA_PATH"
    echo "[INFO] Preparing Commonsense170K dataset..."

    cd "$PERFT_DIR"

    # Set environment for HuggingFace datasets
    export HF_DATASETS_ALLOW_CODE=1
    export TOKENIZERS_PARALLELISM=false

    # Download individual datasets
    echo "[INFO] Downloading commonsense datasets..."
    python commonsense/download_boolq.py || true
    python commonsense/download_piqa.py || true
    python commonsense/download_social_i_qa.py || true
    python commonsense/download_hellaswag.py || true
    python commonsense/download_winogrande.py || true
    python commonsense/download_arc_challenge.py || true
    python commonsense/download_arc_easy.py || true
    python commonsense/download_openbookqa.py || true

    # Build merged training set
    echo "[INFO] Building Commonsense170K..."
    python commonsense/build_commonsense_170k.py --output commonsense/commonsense_170k.json

    cd "$ROOT"
fi

# Check if evaluation datasets exist (needed for evaluation step)
EVAL_DATA_DIR="${PERFT_DIR}/commonsense/dataset"
if [[ ! -d "$EVAL_DATA_DIR" ]]; then
    echo "[INFO] Building evaluation datasets..."
    cd "$PERFT_DIR"
    export HF_DATASETS_ALLOW_CODE=1
    python commonsense/build_eval_sets.py
    cd "$ROOT"
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "${PERFT_DIR}/log"

# ============================================================
# Training
# ============================================================

cd "$PERFT_DIR"
export TOKENIZERS_PARALLELISM=false
# Add PERFT_DIR to PYTHONPATH so Python can find mixtral_modification and olmoe_modification
export PYTHONPATH="${PERFT_DIR}:${PYTHONPATH:-}"

TRAIN_ARGS=(
    --base_model "$BASE_MODEL"
    --data_path "$DATA_PATH"
    --output_dir "$OUTPUT_DIR"
    --batch_size "$BATCH_SIZE"
    --micro_batch_size "$MICRO_BATCH_SIZE"
    --num_epochs "$NUM_EPOCHS"
    --learning_rate "$LEARNING_RATE"
    --cutoff_len "$CUTOFF_LEN"
    --val_set_size "$VAL_SET_SIZE"
    --eval_step "$EVAL_STEP"
    --save_step "$SAVE_STEP"
    --shared_routing_adapter True
    --shared_routing_adapter_num_experts "$ADAPTER_NUM_EXPERTS"
    --shared_routing_adapter_num_experts_per_tok "$ADAPTER_TOP_K"
    --adapter_type "$ADAPTER_TYPE"
    --lora_r "$LORA_R"
    --lora_alpha "$LORA_ALPHA"
    --dropout "$DROPOUT"
)

echo ""
echo "[INFO] Starting PERFT-R training..."
echo ""

if (( NUM_GPUS > 1 )); then
    echo "[INFO] Using DDP with $NUM_GPUS GPUs"
    CUDA_VISIBLE_DEVICES="$GPU_IDS" torchrun \
        --standalone \
        --nproc_per_node="$NUM_GPUS" \
        commonsense/finetune.py "${TRAIN_ARGS[@]}"
else
    echo "[INFO] Using single GPU: $GPU_IDS"
    CUDA_VISIBLE_DEVICES="$GPU_IDS" python commonsense/finetune.py "${TRAIN_ARGS[@]}"
fi

echo ""
echo "[INFO] Training completed!"
echo "[INFO] Checkpoint saved to: $OUTPUT_DIR"
echo ""

# ============================================================
# Evaluation
# ============================================================

echo "[INFO] Starting evaluation on: $EVAL_DATASETS"
echo ""

for DATASET in $EVAL_DATASETS; do
    echo "[INFO] Evaluating on $DATASET..."

    CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python commonsense/commonsense_evaluate.py \
        --dataset "$DATASET" \
        --base_model "$BASE_MODEL" \
        --peft_model "$OUTPUT_DIR" \
        --name "perft_r_top${ADAPTER_TOP_K}of${ADAPTER_NUM_EXPERTS}_r${LORA_R}" \
        --batch_size 16 \
        --max_new_tokens 4 \
        || echo "[WARNING] Evaluation on $DATASET failed, continuing..."

    echo ""
done

echo "=============================================="
echo "PERFT-R Experiment Complete!"
echo "=============================================="
echo "Checkpoint: $OUTPUT_DIR"
echo "Logs: ${PERFT_DIR}/log/"
echo "Results: ${PERFT_DIR}/commonsense/experiment/"
echo "=============================================="

cd "$ROOT"
