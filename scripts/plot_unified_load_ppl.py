#!/usr/bin/env python
"""Plot load balance metrics vs validation PPL from unified_summary.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROUTER_COLORS = {
    "softk": "#1f77b4",
    "expert-choice": "#2ca02c",
    "top1": "#d62728",
    "topk-hard": "#ff7f0e",
    "hash": "#9467bd",
}

SHAPES = {1.0: "o", 1.25: "s", 1.5: "^"}


def scatter(df: pd.DataFrame, x: str, y: str, out: Path, xlabel: str, ylabel: str) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
    plt.figure(figsize=(6.8, 4.6))
    for _, row in df.iterrows():
        c = ROUTER_COLORS.get(row["router"], "gray")
        m = SHAPES.get(row["capacity_factor"], "^")
        plt.scatter(row[x], row[y], color=c, marker=m, s=90, edgecolor="black", linewidths=0.5, alpha=0.9)
    # legends
    from matplotlib.lines import Line2D

    r_handles = [Line2D([], [], marker="o", color=ROUTER_COLORS[r], linestyle="None", markersize=7, label=r)
                 for r in ROUTER_COLORS]
    cf_values = sorted(df["capacity_factor"].unique())
    cf_handles = [Line2D([], [], marker=SHAPES.get(cf, "^"), color="black", linestyle="None", markersize=7, label=f"CF={cf}")
                  for cf in cf_values]
    leg1 = plt.legend(r_handles, [h.get_label() for h in r_handles], title="Router", loc="upper right", fontsize=10, framealpha=0.9)
    plt.gca().add_artist(leg1)
    plt.legend(cf_handles, [h.get_label() for h in cf_handles], title="Capacity", loc="lower right", fontsize=10, framealpha=0.9)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, default="results/unified_summary.csv")
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scatter(df, "mean_load_cv", "best_val_ppl", out_dir / "unified_load_vs_ppl.png",
            "Load CV (lower is better)", "Validation PPL (lower is better)")
    scatter(df, "mean_gate_entropy", "best_val_ppl", out_dir / "unified_entropy_vs_ppl.png",
            "Gate entropy", "Validation PPL (lower is better)")


if __name__ == "__main__":
    main()
