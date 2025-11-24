#!/usr/bin/env bash
# Summarize and plot Experiment A runs (top1/topk-hard/softk × CF) after training.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "fatal: ${PY} not found or not executable; create .venv first." >&2
  exit 1
fi

RUN_ROOT="${RUN_ROOT:-${ROOT}/runs}"
OUT_DIR="${OUT_DIR:-${ROOT}/results}"
SUMMARY="${SUMMARY:-${OUT_DIR}/experiment_a_summary.csv}"
FRONTIER="${FRONTIER:-${OUT_DIR}/experiment_a_frontier.png}"

mkdir -p "${OUT_DIR}"

# Expected run names produced by scripts/run_experiment_a.sh
RUN_NAMES=(
  top1_cf1.0
  top1_cf1.25
  softk_k2_cf1.0
  softk_k2_cf1.25
  topkhard_k2_cf1.0
  topkhard_k2_cf1.25
)

RUN_PATTERNS=()
MISSING=()
for name in "${RUN_NAMES[@]}"; do
  RUN_PATTERNS+=("${RUN_ROOT}/${name}")
  if [[ ! -f "${RUN_ROOT}/${name}/train_log.jsonl" ]]; then
    MISSING+=("${name}")
  fi
done

if (( ${#MISSING[@]} > 0 )); then
  echo "[warn] Missing train logs for: ${MISSING[*]}" >&2
fi

"${PY}" "${ROOT}/scripts/summarize_runs.py" --runs "${RUN_PATTERNS[@]}" --out "${SUMMARY}"

"${PY}" "${ROOT}/scripts/plot_frontier.py" --summary "${SUMMARY}" --out "${FRONTIER}"

OVERLAY_OUT="${OUT_DIR}/experiment_a_overlay.png"
LOG_FILES=()
for name in "${RUN_NAMES[@]}"; do
  LOG_PATH="${RUN_ROOT}/${name}/train_log.jsonl"
  [[ -f "${LOG_PATH}" ]] && LOG_FILES+=("${LOG_PATH}")
done

if (( ${#LOG_FILES[@]} > 0 )); then
  "${PY}" "${ROOT}/scripts/plot_overlay_curves.py" \
    --logs "${LOG_FILES[@]}" \
    --metrics "train_loss,val_loss,ppl" \
    --out "${OVERLAY_OUT}" \
    --title "Experiment A: Train/Val/PPL"
fi

echo "Summary CSV: ${SUMMARY}"
echo "Frontier plots: ${FRONTIER} and ${FRONTIER%.*}_tflops.png"
echo "Overlay curves: ${OVERLAY_OUT}"
