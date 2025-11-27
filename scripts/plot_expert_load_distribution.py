#!/usr/bin/env python3
"""Plot expert load distribution bar charts comparing routing strategies.

This script loads trained checkpoints and runs inference on real data to capture
per-expert token counts, then visualizes the load distribution across experts.

Usage:
    # Using trained checkpoints (recommended)
    python scripts/plot_expert_load_distribution.py \
        --runs runs/unified_top1_cf1_25,runs/unified_topk_hard_cf1_25,runs/unified_softk_cf1_25,runs/unified_hash_cf1_25,runs/unified_expert_choice_cf1_25 \
        --data data/tinystories_val.txt \
        --out results/expert_load_distribution.png \
        --device cuda

    # Or use glob pattern
    python scripts/plot_expert_load_distribution.py \
        --runs "runs/unified_*_cf1_25" \
        --data data/tinystories_val.txt \
        --out results/expert_load_distribution.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import model classes from train_small.py
from scripts.train_small import TinyMoEModel, TransformerBlock, CharDataset, Vocab, aggregate_stats


# ============================================================================
# Load model from checkpoint
# ============================================================================

def load_model_from_run(run_dir: str, device: torch.device) -> Tuple[TinyMoEModel, Dict, str]:
    """Load model from a training run directory."""
    run_path = Path(run_dir)
    hparams_path = run_path / "hparams.json"
    ckpt_path = run_path / "best.pt"

    if not hparams_path.exists():
        raise FileNotFoundError(f"hparams.json not found in {run_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"best.pt not found in {run_dir}")

    with open(hparams_path, "r") as f:
        hparams = json.load(f)

    # Get vocab size from the TRAINING data (not val data)
    train_data_file = hparams.get("data", "data/tinystories_train.txt")
    if not os.path.exists(train_data_file):
        raise FileNotFoundError(f"Training data not found: {train_data_file}")

    with open(train_data_file, "r", encoding="utf-8") as f:
        text = f.read()
    vocab_size = len(set(text))

    # Build moe_kwargs
    moe_kwargs = {
        "num_experts": hparams.get("num_experts", 8),
        "top_k": hparams.get("top_k", 2),
        "ffn_mult": hparams.get("ffn_mult", 4),
        "router_name": hparams.get("router", "torch_soft"),
        "strategy": hparams.get("strategy", "softk"),
        "capacity_factor": hparams.get("capacity_factor", 1.25),
        "renorm_after_drop": hparams.get("renorm_after_drop", True),
        "load_balance_alpha": hparams.get("load_balance_alpha", 0.01),
    }

    model = TinyMoEModel(
        vocab_size=vocab_size,
        seq_len=hparams.get("seq_len", 256),
        dim=hparams.get("dim", 256),
        layers=hparams.get("layers", 4),
        heads=hparams.get("heads", 4),
        dropout=hparams.get("dropout", 0.1),
        moe_kwargs=moe_kwargs,
    )

    # Load checkpoint (strict=False to handle old checkpoints without router_modules)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        # Filter out expected missing keys (router_modules for old checkpoints)
        real_missing = [k for k in missing if "router_modules" not in k]
        if real_missing:
            print(f"Warning: missing keys: {real_missing}")
    model.to(device)
    model.eval()

    strategy = hparams.get("strategy", "unknown")
    return model, hparams, strategy


# ============================================================================
# Collect expert counts
# ============================================================================

def get_expert_counts_from_model(
    model: TinyMoEModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int = 20,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Run inference and collect expert counts from a trained model."""
    model.eval()
    all_counts = []
    all_stats = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            x = x.to(device)
            _, _, stats = model(x)

            if "expert_counts" in stats:
                counts = stats["expert_counts"].cpu().numpy()
                # Sum across layers if shape is (num_layers, num_experts)
                if counts.ndim == 2:
                    counts = counts.sum(axis=0)
                all_counts.append(counts)

            all_stats.append({
                "load_cv": stats["load_cv"].item() if "load_cv" in stats else 0.0,
                "drop_rate": stats["drop_rate"].item() if "drop_rate" in stats else 0.0,
                "gate_entropy": stats["gate_entropy"].item() if "gate_entropy" in stats else 0.0,
            })

    if all_counts:
        mean_counts = np.mean(all_counts, axis=0)
    else:
        mean_counts = np.zeros(8)

    mean_stats = {k: np.mean([s[k] for s in all_stats]) for k in all_stats[0]}
    return mean_counts, mean_stats


# ============================================================================
# Plotting
# ============================================================================

def plot_expert_load_single(
    ax: plt.Axes,
    counts: np.ndarray,
    strategy: str,
    stats: Dict[str, float],
    color: str,
) -> None:
    """Plot expert load distribution for a single strategy."""
    num_experts = len(counts)
    x = np.arange(num_experts)

    bars = ax.bar(x, counts, color=color, alpha=0.8, edgecolor="black", linewidth=0.5)

    # Add mean line
    mean_count = np.mean(counts)
    ax.axhline(mean_count, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_count:.0f}")

    # Formatting
    ax.set_xlabel("Expert ID", fontsize=10)
    ax.set_ylabel("Token Count", fontsize=10)
    ax.set_title(f"{strategy}\nCV={stats['load_cv']:.2f}, Drop={stats['drop_rate']:.1%}", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=8)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_expert_load_comparison(
    results: Dict[str, Tuple[np.ndarray, Dict[str, float]]],
    out_path: str,
    title: str = "Expert Load Distribution by Routing Strategy",
) -> None:
    """Plot expert load distributions for all strategies in a grid."""
    n_strategies = len(results)
    ncols = min(3, n_strategies)
    nrows = (n_strategies + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_strategies == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, n_strategies))

    for idx, (strategy, (counts, stats)) in enumerate(results.items()):
        plot_expert_load_single(axes[idx], counts, strategy, stats, colors[idx])

    # Hide unused axes
    for idx in range(n_strategies, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot expert load distribution from trained models")
    parser.add_argument("--runs", type=str, required=True,
                        help="Comma-separated run directories or glob pattern (e.g., 'runs/unified_*_cf1_25')")
    parser.add_argument("--data", type=str, default="data/tinystories_val.txt",
                        help="Data file for inference")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--num-batches", type=int, default=20, help="Number of batches to process")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="results/expert_load_distribution.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # Expand run directories
    if "*" in args.runs:
        run_dirs = sorted(glob.glob(args.runs))
    else:
        run_dirs = [r.strip() for r in args.runs.split(",")]

    # Filter to only existing directories with checkpoints
    valid_runs = []
    for run_dir in run_dirs:
        if os.path.exists(os.path.join(run_dir, "best.pt")) and os.path.exists(os.path.join(run_dir, "hparams.json")):
            valid_runs.append(run_dir)
        else:
            print(f"  Skipping {run_dir} (missing best.pt or hparams.json)")

    if not valid_runs:
        print("No valid runs found!")
        return

    print(f"Found {len(valid_runs)} valid runs")
    print(f"Data: {args.data}")
    print("-" * 60)

    # Load data once (we'll use the first run's seq_len and train data for vocab)
    first_hparams_path = Path(valid_runs[0]) / "hparams.json"
    with open(first_hparams_path, "r") as f:
        first_hparams = json.load(f)
    seq_len = first_hparams.get("seq_len", 256)

    # Load training data to build vocab
    train_data_file = first_hparams.get("data", "data/tinystories_train.txt")
    with open(train_data_file, "r", encoding="utf-8") as f:
        train_text = f.read()

    # Build vocab from training data
    chars = sorted(set(train_text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = chars
    vocab = Vocab(stoi=stoi, itos=itos)
    print(f"Vocab size: {vocab.size}")

    # Load inference data (val or train)
    data_file = args.data if os.path.exists(args.data) else train_data_file
    with open(data_file, "r", encoding="utf-8") as f:
        inference_text = f.read()

    # Filter out chars not in vocab
    inference_text_filtered = "".join(c for c in inference_text if c in stoi)
    if len(inference_text_filtered) < len(inference_text):
        print(f"Filtered {len(inference_text) - len(inference_text_filtered)} unknown chars from inference data")

    dataset = CharDataset(inference_text_filtered, seq_len, vocab)
    print(f"Using {data_file} for inference ({len(dataset)} samples)")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    results = {}
    for run_dir in valid_runs:
        print(f"  Loading {run_dir}...", end=" ", flush=True)
        try:
            model, hparams, strategy = load_model_from_run(run_dir, device)
            counts, stats = get_expert_counts_from_model(model, dataloader, device, args.num_batches)

            # Use strategy name as key (clean up for display)
            display_name = strategy.replace("_", "-")
            results[display_name] = (counts, stats)

            print(f"strategy={strategy}, CV={stats['load_cv']:.3f}, Drop={stats['drop_rate']:.1%}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("-" * 60)

    if not results:
        print("No results to plot!")
        return

    # Get config info for title
    num_experts = first_hparams.get("num_experts", 8)
    top_k = first_hparams.get("top_k", 2)
    capacity_factor = first_hparams.get("capacity_factor", 1.25)

    # Generate plot
    plot_expert_load_comparison(
        results,
        args.out,
        title=f"Expert Load Distribution (E={num_experts}, K={top_k}, CF={capacity_factor})"
    )

    print("Done!")


if __name__ == "__main__":
    main()
