#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import seaborn as sns


def load_records(path: str) -> Iterable[dict]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _expert_key(rec: dict) -> Tuple[int | None, int | None]:
    # Handle both bench_capacity (num_experts/top_k) and sweep outputs (experts/k).
    e = rec.get("experts", rec.get("num_experts"))
    k = rec.get("k", rec.get("top_k"))
    return e, k


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CF sweep metrics")
    parser.add_argument("--input", type=str, default="results/capacity_ke.jsonl")
    parser.add_argument("--metric", type=str, default="avg_drop_rate")
    parser.add_argument("--out", type=str, default="results/capacity_plot.png")
    args = parser.parse_args()

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    plt.figure(figsize=(7.0, 4.6))

    grouped = defaultdict(list)
    for rec in load_records(args.input):
        key = _expert_key(rec)
        grouped[key].append((rec.get("capacity_factor"), rec.get(args.metric)))

    if not grouped:
        raise SystemExit(f"No records found in {args.input}")

    # Determine scaling/labels
    metric = args.metric
    scale = 1.0
    ylabel = metric
    if metric in ("tokens_per_s", "tokens_per_s_per_rank"):
        scale = 1e6
        ylabel = "Throughput (Tokens/s, millions)"
    elif metric == "avg_drop_rate":
        ylabel = "Drop rate"

    color_cycle = sns.color_palette("muted", n_colors=max(1, len(grouped)))
    for idx, ((experts, k), points) in enumerate(sorted(grouped.items())):
        points = [(x, y) for x, y in points if x is not None and y is not None]
        if not points:
            continue
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] / scale for p in points]
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        plt.plot(
            xs,
            ys,
            marker="o",
            markersize=6,
            linewidth=2.2,
            label=f"E={experts}, K={k}",
            color=color,
        )

    # Highlight CF sweet spot on drop-rate plot
    if metric == "avg_drop_rate":
        plt.axvspan(1.0, 1.1, color="#dddddd", alpha=0.4, label="CF≈1.0–1.1")
        plt.text(1.105, max(ys) if grouped else 0.05, "Near-zero drop", fontsize=9, ha="right", va="bottom")

    plt.xlabel("Capacity Factor")
    plt.ylabel(ylabel)
    plt.title(f"CF Sweep: {ylabel} vs Capacity Factor")
    plt.legend(title="Experts / Top-k", fontsize=9)
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
