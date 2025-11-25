#!/usr/bin/env bash
# Summarize and plot unified router sweep runs (produced by run_experiment_unified.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "fatal: ${PY} not found or not executable; create .venv first." >&2
  exit 1
fi

RUN_GLOB=${RUN_GLOB:-"${ROOT}/runs/unified_*"}
OUT_DIR=${OUT_DIR:-"${ROOT}/results"}
SUMMARY=${SUMMARY:-"${OUT_DIR}/unified_summary.csv"}
FRONTIER=${FRONTIER:-"${OUT_DIR}/unified_frontier.png"}
OVERLAY=${OVERLAY:-"${OUT_DIR}/unified_overlay.png"}

mkdir -p "${OUT_DIR}"

"${PY}" "${ROOT}/scripts/summarize_runs.py" --runs "${RUN_GLOB}" --out "${SUMMARY}"
"${PY}" "${ROOT}/scripts/plot_frontier.py" --summary "${SUMMARY}" --out "${FRONTIER}"

# Collect all train logs under the glob for overlay
LOGS=()
for f in ${RUN_GLOB}; do
  log="${f}/train_log.jsonl"
  [[ -f "$log" ]] && LOGS+=("$log")
done
if (( ${#LOGS[@]} > 0 )); then
  "${PY}" "${ROOT}/scripts/plot_overlay_curves.py" \
    --logs "${LOGS[@]}" \
    --metrics "train_loss,val_loss,ppl" \
    --out "${OVERLAY}" \
    --title "Unified sweep: train/val/ppl"
fi

echo "Summary CSV: ${SUMMARY}"
echo "Frontier plots: ${FRONTIER} and ${FRONTIER%.*}_tflops.png"
echo "Overlay curves: ${OVERLAY}"
