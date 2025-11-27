#!/usr/bin/env python3
"""Plot multiple logs on shared subplots for side-by-side comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib import cycler


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
    pairs = []
    for rec in records:
        if rec.get("done"):
            continue
        if "step" not in rec or metric not in rec:
            continue
        pairs.append((rec["step"], rec[metric]))
    if not pairs:
        return [], []
    # sort by step and de-duplicate keeping the latest value per step
    pairs.sort(key=lambda x: x[0])
    dedup = {}
    for s, v in pairs:
        dedup[s] = v
    steps = list(sorted(dedup.keys()))
    vals = [dedup[s] for s in steps]
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
    parser.add_argument("--max-legend", type=int, default=12, help="If more than this, place legend outside right.")
    parser.add_argument("--filter-cf", type=float, default=None, help="If set, keep only runs whose label contains cf<value>.")
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        raise SystemExit("No metrics specified")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    series: Dict[str, Dict[str, tuple[list, list]]] = {}
    labels: List[str] = []
    routers: List[str] = []
    short_labels: List[str] = []
    final_metric: Dict[str, float] = {}
    for log_path in args.logs:
        if not log_path.exists():
            continue
        records = load_records(log_path)
        label = log_path.parent.name
        # Optional CF filter: expect label like unified_softk_cf1_25
        if args.filter_cf is not None:
            if f"cf{str(args.filter_cf).replace('.', '_')}" not in label:
                continue
        labels.append(label)
        # crude router parse: expect names like larger_softk_... -> router=softk
        parts = label.split("_")
        router = parts[1] if len(parts) > 1 else label
        routers.append(router)
        # shorten label: just router name (CF goes into title/legend elsewhere)
        short_labels.append(router)
        series[label] = {}
        for metric in metrics:
            steps, vals = extract_series(records, metric)
            series[label][metric] = (steps, vals)
        # track final value of first metric for legend ordering
        first_metric = metrics[0]
        s0, v0 = series[label].get(first_metric, ([], []))
        final_metric[label] = v0[-1] if v0 else float("inf")

    if not series:
        raise SystemExit("No valid logs loaded")

    # color map by router name
    unique_routers = list(dict.fromkeys(routers))
    router_to_color = {
        r: color_cycle[i % len(color_cycle)] if color_cycle else None for i, r in enumerate(unique_routers)
    }

    cols = 1
    rows = len(metrics)
    fig, axes = plt.subplots(rows, cols, figsize=(7, 3 * rows), sharex=True)
    if rows == 1:
        axes = [axes]

    # order labels by final value of first metric (descending so worst on top)
    order = sorted(labels, key=lambda lbl: final_metric.get(lbl, float("inf")), reverse=True)
    short_lookup = dict(zip(labels, short_labels))

    for ax, metric in zip(axes, metrics):
        for label in order:
            short = short_lookup.get(label, label)
            steps, vals = series[label].get(metric, ([], []))
            if steps and vals:
                router = label.split("_")[1] if "_" in label else label
                color = router_to_color.get(router)
                ax.plot(steps, vals, label=short, color=color, linewidth=2.2)
        ax.set_title(metric.replace("_", " ").upper())
        ax.set_xlabel("step")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    # Legend placement: if many runs, push outside right to avoid clutter.
    loc = "upper right"
    bbox = (1.02, 1.0)
    if len(labels) > args.max_legend:
        loc = "center left"
        bbox = (1.02, 0.5)
    axes[0].legend(title="run", fontsize=9, loc=loc, bbox_to_anchor=bbox, framealpha=0.9)
    fig.suptitle(args.title, fontsize=13)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved overlay plot to {args.out}")


if __name__ == "__main__":
    main()
