#!/usr/bin/env python3
"""Scatter plot of load balance vs gate entropy for routing strategies."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot load_cv vs gate_entropy per router")
    parser.add_argument("--summary", type=Path, required=True, help="CSV produced by summarize_runs.py")
    parser.add_argument("--out", type=Path, default=Path("results/gate_balance.png"))
    parser.add_argument("--title", type=str, default="Load balance vs Gate entropy")
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    if df.empty:
        raise SystemExit(f"No data in {args.summary}")

    if "router" not in df.columns:
        raise SystemExit(f"{args.summary} missing 'router' column")

    # Support both 'mean_load_cv'/'mean_gate_entropy' if present; fallback to zero entropy if missing
    load_cv_col = "mean_load_cv" if "mean_load_cv" in df.columns else "load_cv"
    entropy_col = "mean_gate_entropy" if "mean_gate_entropy" in df.columns else None
    if load_cv_col not in df.columns:
        raise SystemExit(f"{args.summary} missing load_cv column")

    plt.figure(figsize=(6, 5))
    if entropy_col is None:
        df["_entropy_fill"] = 0.0
        entropy_col = "_entropy_fill"
    for router in sorted(df["router"].unique()):
        subset = df[df["router"] == router]
        plt.scatter(subset[load_cv_col], subset[entropy_col], label=router)
        for _, row in subset.iterrows():
            plt.annotate(
                f"{row.get('top_k', '')}",
                (row[load_cv_col], row[entropy_col]),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=8,
            )

    plt.xlabel("load_cv (lower is better)")
    plt.ylabel("gate_entropy")
    plt.title(args.title)
    plt.grid(alpha=0.3)
    plt.legend(title="router")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
