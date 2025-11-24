#!/usr/bin/env bash
# Convenience wrapper: run Experiment B capacity sweep and plot in one go.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default “rich” sweep; override via env if needed.
TOKENS=${TOKENS:-32768}
HIDDEN=${HIDDEN:-8192}
EXPERTS_LIST=${EXPERTS_LIST:-64,128,256}
TOPK_LIST=${TOPK_LIST:-1,2,4}
CF_LIST=${CF_LIST:-0.9,1.0,1.05,1.1,1.15,1.25,1.5}
ITERS=${ITERS:-200}
DTYPE=${DTYPE:-float16}
DEVICE=${DEVICE:-cuda}
GPU_IDS=${GPU_IDS:-}

OUT=${OUT:-${ROOT}/results/capacity_cf_multi.jsonl}
OUT_DIR=${OUT_DIR:-${ROOT}/results}
OUT_DROP=${OUT_DROP:-${OUT_DIR}/capacity_drop_rate_multi.png}
OUT_TOK=${OUT_TOK:-${OUT_DIR}/capacity_tokens_per_s_multi.png}

export TOKENS HIDDEN EXPERTS_LIST TOPK_LIST CF_LIST ITERS DTYPE DEVICE GPU_IDS OUT

bash "${ROOT}/scripts/run_experiment_b.sh"

INPUT=${INPUT:-${OUT}}
RUN_OUT_DIR=${OUT_DIR} OUT_DIR=${OUT_DIR} INPUT=${INPUT} OUT_DROP=${OUT_DROP} OUT_TOK=${OUT_TOK} \
  bash "${ROOT}/scripts/summarize_plot_experiment_b.sh"

echo "[info] Experiment B run+plot completed"
echo " - JSONL: ${INPUT}"
echo " - Drop-rate plot: ${OUT_DROP}"
echo " - Tokens/s plot: ${OUT_TOK}"
