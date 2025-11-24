#!/usr/bin/env bash
# Summarize & plot Experiment B (capacity sweep) results produced by bench_capacity.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
PLOT="${ROOT}/scripts/plot_capacity.py"

if [[ ! -x "$PY" ]]; then
  echo "fatal: ${PY} not found or not executable; create .venv first." >&2
  exit 1
fi

INPUT=${INPUT:-${ROOT}/results/capacity_cf.jsonl}
OUT_DIR=${OUT_DIR:-${ROOT}/results}
OUT_DROP=${OUT_DROP:-${OUT_DIR}/capacity_drop_rate.png}
OUT_TOK=${OUT_TOK:-${OUT_DIR}/capacity_tokens_per_s.png}

mkdir -p "${OUT_DIR}"

echo "[info] Plotting avg_drop_rate from ${INPUT}"
"${PY}" "${PLOT}" --input "${INPUT}" --metric avg_drop_rate --out "${OUT_DROP}"

echo "[info] Plotting tokens_per_s from ${INPUT}"
"${PY}" "${PLOT}" --input "${INPUT}" --metric tokens_per_s --out "${OUT_TOK}"

echo "Outputs:"
echo " - ${OUT_DROP}"
echo " - ${OUT_TOK}"
