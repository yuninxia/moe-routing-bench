#!/usr/bin/env python3
"""Plot PERFT variants training comparison for slides (Fig.4-style frontier)."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import math

# Family styles
FAMILY_STYLE = {
    "perft_r": {"color": "#2ecc71", "marker": "o", "label": "PERFT-R"},
    "perft_e": {"color": "#e74c3c", "marker": "^", "label": "PERFT-E"},
    "shared_adapter": {"color": "#9b59b6", "marker": "d", "label": "Shared Adapter"},
}


def parse_log_file(log_path: Path) -> pd.DataFrame:
    """Parse training log file to extract metrics."""
    records = []
    step = 0

    with open(log_path, "r") as f:
        for line in f:
            # Training loss: {'loss': 2.8575, 'grad_norm': ..., 'learning_rate': ..., 'epoch': 0.0}
            if "'loss':" in line and "'eval_loss':" not in line:
                step += 10  # logging_steps=10
                match = re.search(r"'loss': ([\d.]+)", line)
                if match:
                    records.append({
                        "step": step,
                        "train_loss": float(match.group(1)),
                        "eval_loss": None,
                    })

            # Eval loss: {'eval_loss': 1.376, ...}
            if "'eval_loss':" in line:
                match = re.search(r"'eval_loss': ([\d.]+)", line)
                if match:
                    records.append({
                        "step": step,
                        "train_loss": None,
                        "eval_loss": float(match.group(1)),
                    })
    if not records:
        return pd.DataFrame(columns=["step", "train_loss", "eval_loss"])
    return pd.DataFrame(records)


def parse_variant_name(name: str) -> dict | None:
    """Extract variant metadata from name like perft_r_top2of4_r16."""
    if name.startswith("perft_r_top"):
        m = re.match(r"perft_r_top(\d+)of(\d+)(?:_r(\d+))?", name)
        if not m:
            return None
        topk, experts, rank = m.groups()
        rank = int(rank) if rank else 16
        return {
            "family": "perft_r",
            "topk": int(topk),
            "experts": int(experts),
            "rank": rank,
            "label": f"R (Top{topk}/{experts})",
        }
    if name.startswith("perft_e"):
        m = re.match(r"perft_e(?:_r(\d+))?", name)
        rank = int(m.group(1)) if m and m.group(1) else 16
        return {
            "family": "perft_e",
            "topk": None,
            "experts": None,
            "rank": rank,
            "label": f"E (r={rank})",
        }
    if name.startswith("shared_adapter"):
        m = re.match(r"shared_adapter(?:_r(\d+))?", name)
        rank = int(m.group(1)) if m and m.group(1) else 16
        return {
            "family": "shared_adapter",
            "topk": None,
            "experts": None,
            "rank": rank,
            "label": f"Shared (r={rank})",
        }
    return None


def collect_variants(results_dir: Path) -> list[dict]:
    variants = []
    for log_path in results_dir.glob("*_train.log"):
        name = log_path.name.replace("_train.log", "")
        meta = parse_variant_name(name)
        if not meta:
            continue
        df = parse_log_file(log_path)
        eval_df = df[df["eval_loss"].notna()]
        if eval_df.empty:
            continue
        # Compute approximate PPL as exp(loss)
        eval_df = eval_df.copy()
        eval_df["ppl"] = eval_df["eval_loss"].apply(lambda x: math.exp(x))
        meta.update(
            {
                "name": name,
                "log_path": log_path,
                "final_eval_loss": eval_df["eval_loss"].iloc[-1],
                "final_ppl": eval_df["ppl"].iloc[-1],
                "eval_df": eval_df,
                "train_df": df[df["train_loss"].notna()],
            }
        )
        variants.append(meta)
    return variants


def compute_efficiency(meta: dict) -> float:
    """Activated Parameter Efficiency = trainable_activated / total_activated (approx).
    Here we approximate total_activated as experts*FFN_dim, but since FFN_dim is constant across variants,
    we use (topk/experts) as the activation fraction, and (rank) as trainable fraction scaling.
    Result is normalized to rank=16 baseline.
    """
    rank_factor = meta.get("rank", 16) / 16.0
    if meta["family"] == "perft_r":
        return (meta["topk"] / meta["experts"]) * rank_factor if meta.get("experts") else rank_factor
    return rank_factor


def plot_frontier(variants: list[dict], output_path: Path):
    """Scatter/line plot mimicking Fig.4: efficiency (x) vs performance (y), size by rank."""
    if not variants:
        print("No data found for frontier plot")
        return

    # Build points
    points = []
    for v in variants:
        eff = compute_efficiency(v)
        style = FAMILY_STYLE.get(v["family"], {"color": "gray", "marker": "o", "label": v["family"]})
        alpha = 0.3 + 0.7 * (v["topk"] / v["experts"]) if v.get("experts") else 0.6
        size = 40 + v["rank"] * 3  # marker size scales with LoRA rank (Fig.4 style)
        points.append(
            {
                "family": v["family"],
                "group": f"{style['label']} (Top{v['topk']}/{v['experts']})" if v.get("experts") else style["label"],
                "label": v["label"],
                "eff": eff,
                # Use 1/ppl as performance proxy (higher is better)
                "perf": 1.0 / v["final_ppl"],
                "color": style["color"],
                "marker": style["marker"],
                "alpha": alpha,
                "size": size,
                "rank": v["rank"],
                "topk": v.get("topk"),
                "experts": v.get("experts"),
            }
        )

    # Group by line key and sort by efficiency
    grouped = {}
    for p in points:
        key = p["group"]
        grouped.setdefault(key, []).append(p)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda x: x["eff"])

    fig, ax = plt.subplots(figsize=(8, 6))
    for key, pts in grouped.items():
        effs = [p["eff"] for p in pts]
        perfs = [p["perf"] for p in pts]
        color = pts[0]["color"]
        marker = pts[0]["marker"]
        ax.plot(effs, perfs, color=color, alpha=0.35, linewidth=1.5)
        for p in pts:
            ax.scatter(
                p["eff"],
                p["perf"],
                color=p["color"],
                marker=p["marker"],
                s=p["size"],
                alpha=p["alpha"],
                edgecolor="black",
                linewidth=0.8,
                label=key,
            )
            ax.text(
                p["eff"] * 1.01,
                p["perf"] * 0.999,
                f"r{p['rank']}",
                fontsize=8,
                ha="left",
                va="top",
            )

    ax.set_xlabel("Activated Parameter Efficiency", fontsize=12)
    ax.set_ylabel("Performance (1 / PPL)", fontsize=12)
    ax.set_title("PERFT Variants Frontier", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)
    ax.legend(uniq_h, uniq_l, fontsize=9, loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_fig5_style(variants: list[dict], output_path: Path):
    """Fig.5-style: group by total experts (R family only), lines for each N, alpha by sparsity, size by rank."""
    rs = [v for v in variants if v["family"] == "perft_r" and v.get("experts")]
    if not rs:
        print("No PERFT-R variants found for Fig.5 plot")
        return
    grouped = {}
    for v in rs:
        key = v["experts"]
        grouped.setdefault(key, []).append(v)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda x: compute_efficiency(x))

    fig, ax = plt.subplots(figsize=(8, 6))
    for experts, pts in sorted(grouped.items()):
        effs = [compute_efficiency(p) for p in pts]
        perfs = [1.0 / p["final_ppl"] for p in pts]
        color = "#2ecc71"  # same as R family
        ax.plot(effs, perfs, color=color, alpha=0.35, linewidth=1.5, label=f"R (N={experts})")
        for p, eff, perf in zip(pts, effs, perfs):
            alpha = 0.3 + 0.7 * (p["topk"] / p["experts"])
            size = 40 + p["rank"] * 3
            ax.scatter(
                eff,
                perf,
                color=color,
                marker="o",
                s=size,
                alpha=alpha,
                edgecolor="black",
                linewidth=0.8,
            )
            ax.text(
                eff * 1.01,
                perf * 0.999,
                f"Top{p['topk']}/{p['experts']}, r{p['rank']}",
                fontsize=8,
                ha="left",
                va="top",
            )

    ax.set_xlabel("Activated Parameter Efficiency (relative)", fontsize=12)
    ax.set_ylabel("Performance (1 / PPL, higher is better)", fontsize=12)
    ax.set_title("PERFT-R: Experts & Sparsity (Fig.5-style)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/perft_variants"),
        help="Directory containing training logs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/perft_variants"),
        help="Directory to save plots",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading logs from: {args.results_dir}")

    variants = collect_variants(args.results_dir)

    # Generate plots
    plot_frontier(variants, args.output_dir / "perft_frontier_loss_vs_eff.png")

    print("\nDone! Plots saved to:", args.output_dir)


if __name__ == "__main__":
    main()
