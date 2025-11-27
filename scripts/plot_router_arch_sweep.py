#!/usr/bin/env python
"""Plot router_arch x strategy sweep frontier and a small params table."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def load_logs(run_glob: str) -> pd.DataFrame:
    rows: List[dict] = []
    for log_path in glob.glob(f"{run_glob}/train_log.jsonl"):
        run_name = Path(log_path).parent.name
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("done"):
                    continue
                row["run"] = run_name
                rows.append(row)
    if not rows:
        raise SystemExit(f"No logs found under pattern: {run_glob}")
    df = pd.DataFrame(rows)
    return df


def best_per_run(df: pd.DataFrame) -> pd.DataFrame:
    # Take best (min) val_loss per run
    key_cols = [
        "router_arch",
        "strategy",
        "router_params",
        "num_experts",
        "top_k",
        "capacity_factor",
    ]
    group_cols = ["run"] + key_cols
    if "run" not in df.columns:
        # derive run name from log path if available
        df["run"] = df.get("run", df.get("outdir", "unknown"))
    best = (
        df.sort_values("val_loss")
        .groupby("run", as_index=False)
        .first()[["run", "val_loss", "ppl", "tokens_per_s", "gate_entropy", "mean_topk_prob"] + key_cols]
    )
    return best


def plot_frontier(best: pd.DataFrame, out: Path, show_labels: bool = False) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(7.2, 4.5))

    markers = {"linear": "o", "mlp": "s", "mlp_hadamard": "^"}
    colors = {
        "softk": "#1f77b4",         # blue
        "expert_choice": "#2ca02c", # green
        "top1": "#d62728",          # red
        "topk_hard": "#ff7f0e",     # orange
        "hash": "#9467bd",          # purple
    }
    strat_labels = {
        "softk": "SoftK",
        "expert_choice": "Expert-Choice",
        "top1": "Top-1",
        "topk_hard": "Top-k Hard",
        "hash": "Hash",
    }
    best = best.copy()
    best["tps_m"] = best["tokens_per_s"] / 1e6

    # Plot points; optional text labels
    for idx, row in best.iterrows():
        arch = row["router_arch"]
        strat = row["strategy"]
        plt.scatter(
            row["tps_m"],
            row["ppl"],
            marker=markers.get(arch, "o"),
            color=colors.get(strat, "#555555"),
            s=110,
            alpha=0.9,
            edgecolor="black",
            linewidths=0.6,
        )
        if show_labels:
            label_txt = f"{arch[0].upper()}·{strat_labels.get(strat, strat)}"
            # Small deterministic offset to reduce collisions
            dx = 0.1 * ((idx % 3) - 1)
            dy = 0.01 * ((idx % 4) - 1.5)
            plt.text(
                row["tps_m"] + dx,
                row["ppl"] + dy,
                label_txt,
                fontsize=8.0,
                ha="left",
                va="bottom",
            )

    # Legends: strategies (color) and architectures (marker)
    from matplotlib.lines import Line2D

    strat_handles = []
    for s in strat_labels:
        if s not in colors:
            continue
        strat_handles.append(
            Line2D([], [], marker="o", color=colors[s], linestyle="None", markersize=7.5, label=strat_labels[s])
        )
    arch_handles = [
        Line2D([], [], marker=markers[a], color="black", linestyle="None", markersize=7.5, label=a)
        for a in markers
    ]

    # Place legends to avoid overlap with upper-right data cluster:
    # Strategy in lower right, Router Arch in upper left.
    leg1 = plt.legend(
        strat_handles,
        [h.get_label() for h in strat_handles],
        title="Strategy",
        fontsize=9,
        loc="lower right",
        framealpha=0.9,
    )
    plt.gca().add_artist(leg1)
    plt.legend(
        arch_handles,
        [h.get_label() for h in arch_handles],
        title="Router Arch",
        fontsize=9,
        loc="upper left",
        framealpha=0.9,
    )

    plt.xlabel("Throughput (Tokens/s, millions)")
    plt.ylabel("Validation PPL (lower is better)")
    plt.title("Router Architecture × Strategy Frontier (E=8)")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    print(f"Saved frontier to {out}")


def save_table(best: pd.DataFrame, out_csv: Path) -> None:
    cols = [
        "run",
        "router_arch",
        "strategy",
        "router_params",
        "mean_topk_prob",
        "gate_entropy",
        "ppl",
        "tokens_per_s",
    ]
    best[cols].to_csv(out_csv, index=False)
    print(f"Saved summary table to {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, required=True, help="Glob for runs, e.g. 'runs/router_arch_sweep/arch_*'")
    parser.add_argument("--out", type=str, default="results/router_arch_frontier.png")
    parser.add_argument("--table", type=str, default="results/router_arch_summary.csv")
    parser.add_argument("--labels", action="store_true", help="Show inline text labels on the plot.")
    parser.add_argument("--exclude", nargs="*", default=[], help="Strategies to exclude from plotting (e.g., hash top1).")
    args = parser.parse_args()

    df = load_logs(args.runs)
    best = best_per_run(df)
    if args.exclude:
        best = best[~best["strategy"].isin(args.exclude)].reset_index(drop=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_frontier(best, out_path, show_labels=args.labels)
    save_table(best, Path(args.table))


if __name__ == "__main__":
    main()
