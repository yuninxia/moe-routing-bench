#!/usr/bin/env python3
"""Plot step-wise training metrics from a JSONL log."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def load_records(log_path: Path) -> List[dict]:
    records: List[dict] = []
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics vs step from train_log.jsonl")
    parser.add_argument("--log", type=Path, required=True, help="Path to train_log.jsonl")
    parser.add_argument("--out", type=Path, default=Path("results/training_curve.png"))
    parser.add_argument(
        "--metrics",
        type=str,
        default="train_loss,val_loss,ppl",
        help="Comma-separated metrics to plot against step",
    )
    args = parser.parse_args()

    records = load_records(args.log)
    if not records:
        raise SystemExit(f"No records found in {args.log}")

    steps = [r.get("step") for r in records if "step" in r]
    if not steps:
        raise SystemExit("Log does not contain step entries")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    plt.figure(figsize=(8, 5))
    for metric in metrics:
        values = [r.get(metric) for r in records if metric in r]
        if len(values) != len(steps):
            # align lengths by filtering pairs that contain both step and metric
            paired = [(r.get("step"), r.get(metric)) for r in records if metric in r and "step" in r]
            if not paired:
                continue
            step_vals, metric_vals = zip(*paired)
        else:
            step_vals, metric_vals = steps, values
        plt.plot(step_vals, metric_vals, label=metric)

    plt.xlabel("step")
    plt.ylabel("value")
    plt.title(f"Metrics vs step ({args.log.parent.name})")
    plt.grid(alpha=0.3)
    plt.legend()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
