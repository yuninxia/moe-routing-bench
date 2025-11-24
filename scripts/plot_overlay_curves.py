#!/usr/bin/env python3
"""Plot multiple logs on shared subplots for side-by-side comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_records(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def extract_series(records: List[dict], metric: str) -> tuple[list, list]:
    steps: List = []
    vals: List = []
    for rec in records:
        if "step" not in rec or metric not in rec:
            continue
        steps.append(rec["step"])
        vals.append(rec[metric])
    return steps, vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay metrics from multiple train logs")
    parser.add_argument("--logs", type=Path, nargs="+", required=True, help="train_log.jsonl files")
    parser.add_argument(
        "--metrics",
        type=str,
        default="train_loss,val_loss,ppl",
        help="Comma-separated metrics to plot",
    )
    parser.add_argument("--out", type=Path, default=Path("results/overlay.png"))
    parser.add_argument("--title", type=str, default="Experiment A comparison")
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        raise SystemExit("No metrics specified")

    series: Dict[str, Dict[str, tuple[list, list]]] = {}
    labels: List[str] = []
    for log_path in args.logs:
        if not log_path.exists():
            continue
        records = load_records(log_path)
        label = log_path.parent.name
        labels.append(label)
        series[label] = {}
        for metric in metrics:
            series[label][metric] = extract_series(records, metric)

    if not series:
        raise SystemExit("No valid logs loaded")

    cols = 1
    rows = len(metrics)
    fig, axes = plt.subplots(rows, cols, figsize=(7, 3 * rows), sharex=False)
    if rows == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for label in labels:
            steps, vals = series[label].get(metric, ([], []))
            if steps and vals:
                ax.plot(steps, vals, label=label)
        ax.set_title(metric)
        ax.set_xlabel("step")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    axes[0].legend(title="run", fontsize="small")
    fig.suptitle(args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved overlay plot to {args.out}")


if __name__ == "__main__":
    main()
