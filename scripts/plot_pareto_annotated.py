#!/usr/bin/env python3
"""Plot annotated Pareto frontier with operating regimes and recommendations.

Creates an annotated version of the unified frontier showing:
- Operating regime labels (High throughput, Balanced, Best quality)
- Trade-off arrows
- Recommendation zones
- Pareto frontier line

Usage:
    python scripts/plot_pareto_annotated.py \
        --summary results/unified_summary.csv \
        --out results/unified_frontier_annotated.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot annotated Pareto frontier")
    parser.add_argument("--summary", type=str, default="results/unified_summary.csv",
                        help="CSV produced by summarize_runs.py")
    parser.add_argument("--out", type=str, default="results/unified_frontier_annotated.png")
    args = parser.parse_args()

    df = pd.read_csv(args.summary)

    if df.empty:
        raise SystemExit("No data to plot")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Color palette for strategies
    strategy_colors = {
        "expert-choice": "#2ecc71",  # Green
        "softk": "#f39c12",          # Orange
        "hash": "#9b59b6",           # Purple
        "topk-hard": "#3498db",      # Blue
        "top1": "#e74c3c",           # Red
    }

    # Marker styles for capacity factors
    cf_markers = {
        1.0: "o",
        1.25: "s",
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    # Collect all points for Pareto frontier calculation
    all_points = []

    # Plot each strategy
    for strategy in sorted(df["router"].dropna().unique()):
        subset = df[df["router"] == strategy]
        color = strategy_colors.get(strategy, "#7f8c8d")

        for _, row in subset.iterrows():
            x = row["median_tokens_per_s"] / 1e6  # Convert to M tokens/s
            y = row["best_val_ppl"]
            cf = row["capacity_factor"]
            marker = cf_markers.get(cf, "^")

            ax.scatter(x, y, c=color, marker=marker, s=150, edgecolors="black",
                       linewidths=0.5, zorder=5)
            all_points.append((x, y, strategy, cf))

    # Calculate and draw Pareto frontier
    points = np.array([(p[0], p[1]) for p in all_points])
    # Pareto optimal: higher throughput AND lower PPL is better
    # Sort by throughput descending
    sorted_idx = np.argsort(-points[:, 0])
    pareto_points = []
    min_ppl = float("inf")
    for idx in sorted_idx:
        if points[idx, 1] < min_ppl:
            pareto_points.append(points[idx])
            min_ppl = points[idx, 1]

    if pareto_points:
        pareto_points = np.array(pareto_points)
        # Sort by throughput for line
        pareto_sorted = pareto_points[np.argsort(pareto_points[:, 0])]
        ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], "k--", linewidth=2,
                alpha=0.6, label="Pareto frontier", zorder=3)

    # Add operating regime shaded regions
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # High throughput region (right side, higher PPL acceptable)
    rect1 = mpatches.Rectangle((30, 6.4), 6, 0.8, alpha=0.15, facecolor="#e74c3c",
                                edgecolor="none", zorder=1)
    ax.add_patch(rect1)
    ax.text(33, 6.95, "High Throughput\n(latency-critical)", fontsize=9,
            ha="center", va="center", color="#c0392b", fontweight="bold")

    # Best quality region (left side, lower PPL)
    rect2 = mpatches.Rectangle((24, 5.3), 5, 0.5, alpha=0.15, facecolor="#2ecc71",
                                edgecolor="none", zorder=1)
    ax.add_patch(rect2)
    ax.text(26.5, 5.55, "Best Quality\n(accuracy-critical)", fontsize=9,
            ha="center", va="center", color="#27ae60", fontweight="bold")

    # Balanced region (middle)
    rect3 = mpatches.Rectangle((27, 5.9), 5, 0.6, alpha=0.15, facecolor="#3498db",
                                edgecolor="none", zorder=1)
    ax.add_patch(rect3)
    ax.text(29.5, 6.2, "Balanced\nTrade-off", fontsize=9,
            ha="center", va="center", color="#2980b9", fontweight="bold")

    # Add trade-off arrow
    ax.annotate("", xy=(34, 5.8), xytext=(26, 6.6),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=2,
                                connectionstyle="arc3,rad=-0.1"))
    ax.text(30, 6.45, "Quality ↔ Throughput\nTrade-off", fontsize=8,
            ha="center", va="bottom", color="gray", style="italic")

    # Add strategy labels near points (best per strategy)
    best_per_strategy = df.loc[df.groupby("router")["best_val_ppl"].idxmin()]
    label_offsets = {
        "expert-choice": (0.5, -0.15),
        "softk": (0.5, 0.1),
        "hash": (-1.5, 0.1),
        "topk-hard": (0.5, 0.1),
        "top1": (-1.0, -0.15),
    }
    for _, row in best_per_strategy.iterrows():
        strategy = row["router"]
        x = row["median_tokens_per_s"] / 1e6
        y = row["best_val_ppl"]
        offset = label_offsets.get(strategy, (0.3, 0.1))
        ax.annotate(strategy, xy=(x, y), xytext=(x + offset[0], y + offset[1]),
                    fontsize=9, fontweight="bold",
                    color=strategy_colors.get(strategy, "black"),
                    arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5, lw=0.5))

    # Add recommendation box
    rec_text = ("Recommendations:\n"
                "• Max throughput → top1\n"
                "• Best quality → expert-choice/softk\n"
                "• Balanced → topk-hard\n"
                "• CF=1.25 (■) generally better than CF=1.0 (●)")
    props = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9)
    ax.text(0.02, 0.02, rec_text, transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom", bbox=props, family="monospace")

    # Legend for strategies
    strategy_handles = [mpatches.Patch(color=color, label=strat)
                        for strat, color in strategy_colors.items()]
    # Legend for markers (CF)
    from matplotlib.lines import Line2D
    cf_handles = [Line2D([0], [0], marker="o", color="gray", label="CF=1.0",
                         markersize=8, linestyle="None"),
                  Line2D([0], [0], marker="s", color="gray", label="CF=1.25",
                         markersize=8, linestyle="None")]

    legend1 = ax.legend(handles=strategy_handles, title="Strategy", loc="upper right",
                        fontsize=8, title_fontsize=9)
    ax.add_artist(legend1)
    ax.legend(handles=cf_handles, title="Capacity Factor", loc="upper center",
              fontsize=8, title_fontsize=9, bbox_to_anchor=(0.85, 0.75))

    # Labels and title
    ax.set_xlabel("Throughput (M tokens/s)", fontsize=11)
    ax.set_ylabel("Best Validation PPL", fontsize=11)
    ax.set_title("Annotated Pareto Frontier: Routing Strategy Trade-offs (E=8, K=2)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, zorder=0)

    # Invert y-axis so lower PPL (better) is at top
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight", dpi=150)
    print(f"Saved: {args.out}")
    plt.close()


if __name__ == "__main__":
    main()
