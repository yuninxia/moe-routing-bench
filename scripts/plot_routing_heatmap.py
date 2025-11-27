#!/usr/bin/env python3
"""Plot token-expert routing assignment heatmaps from trained models.

Visualizes which experts each token is routed to, showing routing patterns
across different strategies using real trained checkpoints and data.

Usage:
    python scripts/plot_routing_heatmap.py \
        --runs "runs/unified_*_cf1_25" \
        --data data/tinystories_val.txt \
        --out results/routing_heatmap.png \
        --device cuda
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

    # Get vocab size from the TRAINING data
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
        real_missing = [k for k in missing if "router_modules" not in k]
        if real_missing:
            print(f"Warning: missing keys: {real_missing}")
    model.to(device)
    model.eval()

    strategy = hparams.get("strategy", "unknown")
    return model, hparams, strategy


# ============================================================================
# Collect routing assignments
# ============================================================================

def get_routing_assignments_from_model(
    model: TinyMoEModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_tokens_show: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and capture routing assignments from first batch.

    Returns:
        assignments: Binary matrix [num_tokens_show, num_experts]
        weights: Weight matrix [num_tokens_show, num_experts]
    """
    model.eval()
    num_experts = model.blocks[0].moe.num_experts

    # Get first batch
    x, y = next(iter(dataloader))
    x = x.to(device)

    # We need to capture routing info from the first MoE layer
    # Run forward and extract from first block's moe layer
    with torch.no_grad():
        # Get hidden states after embedding
        b, t = x.shape
        h = model.embed(x) + model.pos_embed[:, :t, :]

        # Causal mask
        mask = torch.full((t, t), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)

        # Pass through first block to get routing info
        h_ln = model.blocks[0].ln1(h)
        attn_out, _ = model.blocks[0].attn(h_ln, h_ln, h_ln, attn_mask=mask)
        h = h + model.blocks[0].dropout(attn_out)

        h_ln2 = model.blocks[0].ln2(h)

        # Get routing from MoE layer
        moe = model.blocks[0].moe
        num_tokens = h_ln2.shape[0] * h_ln2.shape[1]
        x_2d = h_ln2.reshape(num_tokens, -1)

        # Get router logits and routing decisions
        logits = moe._router_logits(x_2d)
        topk_idx, gates, k_eff = moe._route(logits)

        # Build assignment matrix
        tokens_to_show = min(num_tokens_show, num_tokens)
        assignments = np.zeros((tokens_to_show, num_experts), dtype=np.float32)
        weights = np.zeros((tokens_to_show, num_experts), dtype=np.float32)

        topk_idx_np = topk_idx[:tokens_to_show].cpu().numpy()
        if gates is not None:
            gates_np = gates[:tokens_to_show].cpu().numpy()
        else:
            gates_np = np.ones_like(topk_idx_np, dtype=np.float32) / k_eff

        for token_id in range(tokens_to_show):
            for k in range(topk_idx_np.shape[1]):
                expert_id = topk_idx_np[token_id, k]
                assignments[token_id, expert_id] = 1.0
                weights[token_id, expert_id] = gates_np[token_id, k]

    return assignments, weights


# ============================================================================
# Plotting
# ============================================================================

def plot_single_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    strategy: str,
    cmap: str = "Blues",
) -> None:
    """Plot a single routing heatmap."""
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xlabel("Expert ID", fontsize=10)
    ax.set_ylabel("Token ID", fontsize=10)
    ax.set_title(strategy, fontsize=12, fontweight="bold")

    # Set ticks
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels([str(i) for i in range(data.shape[1])], fontsize=8)

    # Reduce y-ticks for readability
    num_tokens = data.shape[0]
    ytick_step = max(1, num_tokens // 10)
    ax.set_yticks(range(0, num_tokens, ytick_step))
    ax.set_yticklabels([str(i) for i in range(0, num_tokens, ytick_step)], fontsize=8)

    return im


def plot_routing_heatmaps(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_path: str,
    title: str = "Token-Expert Routing Assignments",
) -> None:
    """Plot routing heatmaps for all strategies in a grid."""
    n_strategies = len(results)
    ncols = min(3, n_strategies)
    nrows = (n_strategies + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if n_strategies == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Use different colormaps for variety
    cmaps = ["Blues", "Greens", "Oranges", "Purples", "Reds"]

    for idx, (strategy, (assignments, weights)) in enumerate(results.items()):
        cmap = cmaps[idx % len(cmaps)]
        im = plot_single_heatmap(axes[idx], weights, strategy, cmap)

    # Hide unused axes
    for idx in range(n_strategies, len(axes)):
        axes[idx].set_visible(False)

    # Add shared colorbar on the right
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Routing Weight", fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight="bold")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def plot_routing_pattern_summary(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_path: str,
) -> None:
    """Plot summary statistics of routing patterns."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    strategies = list(results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))

    # Plot 1: Average experts per token
    avg_experts = []
    for strategy in strategies:
        assignments, _ = results[strategy]
        avg_experts.append(np.mean(np.sum(assignments, axis=1)))

    ax1 = axes[0]
    bars1 = ax1.bar(range(len(strategies)), avg_experts, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(len(strategies)))
    ax1.set_xticklabels(strategies, rotation=15, ha="right", fontsize=10)
    ax1.set_xlabel("Routing Strategy", fontsize=11)
    ax1.set_ylabel("Avg Experts per Token", fontsize=11)
    ax1.set_title("Expert Utilization per Token", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, avg_experts):
        ax1.annotate(f"{val:.2f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    # Plot 2: Expert selection frequency
    ax2 = axes[1]
    x = np.arange(results[strategies[0]][0].shape[1])  # num_experts
    width = 0.15
    for idx, strategy in enumerate(strategies):
        assignments, _ = results[strategy]
        freq = np.mean(assignments, axis=0)
        ax2.bar(x + idx * width, freq, width, label=strategy, color=colors[idx], edgecolor="black", linewidth=0.3)

    ax2.set_xlabel("Expert ID", fontsize=11)
    ax2.set_ylabel("Selection Frequency", fontsize=11)
    ax2.set_title("Expert Selection Frequency by Strategy", fontsize=12, fontweight="bold")
    ax2.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax2.set_xticklabels([str(i) for i in x], fontsize=9)
    ax2.legend(title="Strategy", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_summary = out_path.replace(".png", "_summary.png")
    plt.savefig(out_summary, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_summary}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot token-expert routing heatmaps from trained models")
    parser.add_argument("--runs", type=str, required=True,
                        help="Comma-separated run directories or glob pattern")
    parser.add_argument("--data", type=str, default="data/tinystories_val.txt",
                        help="Data file for inference")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-tokens-show", type=int, default=64, help="Number of tokens to show in heatmap")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="results/routing_heatmap.png")
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

    # Load data
    first_hparams_path = Path(valid_runs[0]) / "hparams.json"
    with open(first_hparams_path, "r") as f:
        first_hparams = json.load(f)
    seq_len = first_hparams.get("seq_len", 256)

    # Build vocab from training data
    train_data_file = first_hparams.get("data", "data/tinystories_train.txt")
    with open(train_data_file, "r", encoding="utf-8") as f:
        train_text = f.read()

    chars = sorted(set(train_text))
    stoi = {c: i for i, c in enumerate(chars)}
    vocab = Vocab(stoi=stoi, itos=chars)
    print(f"Vocab size: {vocab.size}")

    # Load inference data
    data_file = args.data if os.path.exists(args.data) else train_data_file
    with open(data_file, "r", encoding="utf-8") as f:
        inference_text = f.read()

    inference_text_filtered = "".join(c for c in inference_text if c in stoi)
    dataset = CharDataset(inference_text_filtered, seq_len, vocab)
    print(f"Using {data_file} for inference")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    results = {}
    for run_dir in valid_runs:
        print(f"  Loading {run_dir}...", end=" ", flush=True)
        try:
            model, hparams, strategy = load_model_from_run(run_dir, device)
            assignments, weights = get_routing_assignments_from_model(
                model, dataloader, device, args.num_tokens_show
            )

            display_name = strategy.replace("_", "-")
            results[display_name] = (assignments, weights)

            avg_experts = np.mean(np.sum(assignments, axis=1))
            print(f"strategy={strategy}, avg_experts/token={avg_experts:.2f}")
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

    # Generate plots
    plot_routing_heatmaps(
        results,
        args.out,
        title=f"Token-Expert Routing Heatmap (E={num_experts}, K={top_k})"
    )
    plot_routing_pattern_summary(results, args.out)

    print("Done!")


if __name__ == "__main__":
    main()
