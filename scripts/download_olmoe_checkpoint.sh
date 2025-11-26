#!/usr/bin/env bash
# Download all safetensor shards for allenai/OLMoE-1B-7B-0924 using the standard
# Hugging Face cache layout. This avoids missing-shard errors in torchrun.
#
# Usage:
#   bash scripts/download_olmoe_checkpoint.sh
#   CACHE_DIR=/scratch/.../hf_cache bash scripts/download_olmoe_checkpoint.sh
#
# After downloading, set HF_HOME/HF_HUB_CACHE/TRANSFORMERS_CACHE to CACHE_DIR
# when running PERFT scripts.

set -euo pipefail

MODEL="allenai/OLMoE-1B-7B-0924"
# Default to repo-local cache root (will create hub/models--allenai--OLMoE-1B-7B-0924/...)
CACHE_DIR="${CACHE_DIR:-/scratch/yx87/playground/moe-routing-bench/.cache/hf}"
export MODEL
export CACHE_DIR

mkdir -p "${CACHE_DIR}"

echo "[info] Downloading ${MODEL} into ${CACHE_DIR} (standard HF cache layout)"
CACHE_DIR="${CACHE_DIR}" MODEL="${MODEL}" python - <<'PY'
import os
from huggingface_hub import snapshot_download

cache_dir = os.environ["CACHE_DIR"]
repo = os.environ["MODEL"]
snapshot_download(
    repo,
    cache_dir=cache_dir,
    local_files_only=False,
    allow_patterns=[
        "model-0000*.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "tokenizer.*",
    ],
)
print(f"[info] Snapshot downloaded to {cache_dir}")
PY

echo "[info] Done. Export HF_HOME=${CACHE_DIR} (and HF_HUB_CACHE/TRANSFORMERS_CACHE) before running."
