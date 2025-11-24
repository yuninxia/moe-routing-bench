#!/usr/bin/env bash
# Run Experiment B: capacity-factor microbench via bench_capacity.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
SCRIPT="${ROOT}/scripts/bench_capacity.py"

if [[ ! -x "$PY" ]]; then
  echo "fatal: ${PY} not found or not executable; create .venv first." >&2
  exit 1
fi

TOKENS=${TOKENS:-16384}
HIDDEN=${HIDDEN:-4096}
EXPERTS_LIST=${EXPERTS_LIST:-${EXPERTS:-128}}
TOPK_LIST=${TOPK_LIST:-${TOPK:-2}}
CF_LIST=${CF_LIST:-1.0,1.10,1.25,1.50}
ITERS=${ITERS:-50}
DTYPE=${DTYPE:-float16}
DEVICE=${DEVICE:-cuda}
OUT=${OUT:-${ROOT}/results/capacity_cf.jsonl}

mkdir -p "$(dirname "${OUT}")"
: > "${OUT}"

IFS=',' read -ra E_LIST <<<"${EXPERTS_LIST}"
IFS=',' read -ra K_LIST <<<"${TOPK_LIST}"

for E in "${E_LIST[@]}"; do
  for K in "${K_LIST[@]}"; do
    echo "[info] Running capacity sweep E=${E} K=${K} -> ${OUT}"
    CMD=(
      "${PY}" "${SCRIPT}"
      --num-tokens "${TOKENS}"
      --hidden-dim "${HIDDEN}"
      --num-experts "${E}"
      --top-k "${K}"
      --capacity-factors "${CF_LIST}"
      --iters "${ITERS}"
      --dtype "${DTYPE}"
      --device "${DEVICE}"
    )
    if [[ -n "${GPU_IDS:-}" ]]; then
      CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${CMD[@]}" >> "${OUT}"
    else
      "${CMD[@]}" >> "${OUT}"
    fi
  done
done

echo "[info] Done. Results appended to ${OUT}"
