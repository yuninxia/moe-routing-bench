#!/usr/bin/env bash
# Summarize and plot Experiment C runs (multiple routing strategies at fixed hyperparams).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "fatal: $PY not found or not executable; create .venv first." >&2
  exit 1
fi

RUN_ROOT="${RUN_ROOT:-${ROOT}/runs}"
OUT_DIR="${OUT_DIR:-${ROOT}/results}"
SUMMARY="${SUMMARY:-${OUT_DIR}/expC_summary.csv}"
FRONTIER="${FRONTIER:-${OUT_DIR}/expC_frontier.png}"

mkdir -p "${OUT_DIR}"

RUN_PATTERNS=(
  "${RUN_ROOT}/expC_top1"
  "${RUN_ROOT}/expC_topk_hard"
  "${RUN_ROOT}/expC_softk"
  "${RUN_ROOT}/expC_hash"
  "${RUN_ROOT}/expC_expert_choice"
)

"${PY}" "${ROOT}/scripts/summarize_runs.py" --runs "${RUN_PATTERNS[@]}" --out "${SUMMARY}"

"${PY}" "${ROOT}/scripts/plot_frontier.py" --summary "${SUMMARY}" --out "${FRONTIER}"

OVERLAY_OUT="${OUT_DIR}/expC_overlay.png"
LOG_FILES=()
for pat in "${RUN_PATTERNS[@]}"; do
  LOG_PATH="${pat}/train_log.jsonl"
  [[ -f "${LOG_PATH}" ]] && LOG_FILES+=("${LOG_PATH}")
done

if (( ${#LOG_FILES[@]} > 0 )); then
  "${PY}" "${ROOT}/scripts/plot_overlay_curves.py" \
    --logs "${LOG_FILES[@]}" \
    --metrics "train_loss,val_loss,ppl" \
    --out "${OVERLAY_OUT}" \
    --title "Experiment C: Routing comparison"
fi

echo "Summary CSV: ${SUMMARY}"
echo "Frontier plots: ${FRONTIER} and ${FRONTIER%.*}_tflops.png"
echo "Overlay curves: ${OVERLAY_OUT}"
