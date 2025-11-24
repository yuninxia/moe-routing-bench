#!/usr/bin/env python3
"""Aggregate routing sweep logs into a CSV summary."""
from __future__ import annotations

import argparse
import glob
import json
import os
from statistics import mean, median
from typing import Any, Dict, List

import pandas as pd


def _safe_median(values: List[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(median(vals)) if vals else None


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def summarize_run(run_dir: str) -> Dict[str, Any] | None:
    log_path = os.path.join(run_dir, "train_log.jsonl")
    hparams_path = os.path.join(run_dir, "hparams.json")
    if not os.path.isfile(log_path) or not os.path.isfile(hparams_path):
        return None

    with open(hparams_path, "r", encoding="utf-8") as fh:
        hparams = json.load(fh)

    records = _read_jsonl(log_path)
    if not records:
        return None

    tokens_s = [r.get("tokens_per_s") for r in records if "tokens_per_s" in r]
    eff_tflops = [r.get("eff_tflops") for r in records if "eff_tflops" in r]
    bw_gibps = [r.get("bw_GiBps_strict") for r in records if "bw_GiBps_strict" in r]
    drop_rates = [r.get("drop_rate") for r in records if "drop_rate" in r]
    load_cvs = [r.get("load_cv") for r in records if "load_cv" in r]

    best_record = min(records, key=lambda x: x.get("ppl", float("inf")))
    final_record = records[-1]

    return {
        "run": os.path.basename(run_dir.rstrip("/")),
        "router": hparams.get("strategy", "").replace("_", "-"),
        "top_k": hparams.get("top_k"),
        "capacity_factor": hparams.get("capacity_factor"),
        "num_experts": hparams.get("num_experts"),
        "dim": hparams.get("dim"),
        "expand": hparams.get("expand"),
        "world_size": hparams.get("world_size", 1),
        "best_step": best_record.get("step"),
        "best_val_ppl": best_record.get("ppl"),
        "best_val_loss": best_record.get("val_loss"),
        "best_tokens_per_s": best_record.get("tokens_per_s"),
        "best_eff_tflops": best_record.get("eff_tflops"),
        "final_val_ppl": final_record.get("ppl"),
        "final_val_loss": final_record.get("val_loss"),
        "median_tokens_per_s": _safe_median(tokens_s),
        "median_eff_tflops": _safe_median(eff_tflops),
        "median_bw_GiBps": _safe_median(bw_gibps),
        "mean_drop_rate": mean(drop_rates) if drop_rates else None,
        "mean_load_cv": mean(load_cvs) if load_cvs else None,
        "log_path": log_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MoE routing sweeps")
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        default=["runs/routing_sweep/*"],
        help="One or more glob patterns for run directories",
    )
    parser.add_argument("--out", type=str, default="results/routing_sweep_summary.csv")
    args = parser.parse_args()

    patterns: List[str] = args.runs
    all_dirs: List[str] = []
    for pattern in patterns:
        all_dirs.extend(glob.glob(pattern))

    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(set(all_dirs)):
        summary = summarize_run(run_dir)
        if summary is not None:
            rows.append(summary)

    if not rows:
        print("No runs found for patterns", patterns)
        return

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(df)} entries")


if __name__ == "__main__":
    main()
