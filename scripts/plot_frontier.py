#!/usr/bin/env python3
"""Plot throughput/TFLOPs vs perplexity frontiers from sweep summary."""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot routing Pareto frontiers")
    parser.add_argument("--summary", type=str, required=True, help="CSV produced by summarize_runs.py")
    parser.add_argument("--filter-E", type=int, default=None, help="Filter by num_experts")
    parser.add_argument("--filter-dim", type=int, default=None, help="Filter by model dim")
    parser.add_argument("--out", type=str, default="results/routing_frontier.png")
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    if args.filter_E is not None:
        df = df[df["num_experts"] == args.filter_E]
    if args.filter_dim is not None:
        df = df[df["dim"] == args.filter_dim]

    if df.empty:
        raise SystemExit("No data to plot after filtering")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Plot tokens/s vs best PPL
    plt.figure(figsize=(7, 5))
    for strategy in sorted(df["router"].dropna().unique()):
        subset = df[df["router"] == strategy]
        plt.scatter(
            subset["median_tokens_per_s"],
            subset["best_val_ppl"],
            label=strategy,
        )
    plt.xlabel("Throughput (tokens/s, aggregated across world)")
    plt.ylabel("Best validation PPL")
    plt.title("Routing frontier: Throughput vs Best PPL")
    plt.legend(title="strategy")
    plt.grid(alpha=0.3)
    plt.gca().invert_yaxis()  # Lower PPL (better) at top
    plt.savefig(args.out, bbox_inches="tight")
    print("Saved", args.out)

    # Plot TFLOPs vs best PPL
    out_tflops = args.out.replace(".png", "_tflops.png")
    plt.figure(figsize=(7, 5))
    for strategy in sorted(df["router"].dropna().unique()):
        subset = df[df["router"] == strategy]
        plt.scatter(
            subset["median_eff_tflops"],
            subset["best_val_ppl"],
            label=strategy,
        )
    plt.xlabel("Effective TFLOPs (FFN+router, observed)")
    plt.ylabel("Best validation PPL")
    plt.title("Routing frontier: TFLOPs vs Best PPL")
    plt.legend(title="strategy")
    plt.grid(alpha=0.3)
    plt.gca().invert_yaxis()  # Lower PPL (better) at top
    plt.savefig(out_tflops, bbox_inches="tight")
    print("Saved", out_tflops)


if __name__ == "__main__":
    main()
