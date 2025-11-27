#!/usr/bin/env python3
"""Visualize token routing with colored text (Mixtral-style).

Each token is colored by its primary assigned expert, showing how different
routing strategies assign tokens to experts based on content.

Usage:
    python scripts/plot_token_routing_colored.py \
        --runs "runs/unified_*_cf1_25" \
        --data data/tinystories_val.txt \
        --out results/token_routing_colored.png \
        --device cuda
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_small import TinyMoEModel, CharDataset, Vocab


# ============================================================================
# Load model from checkpoint
# ============================================================================

def load_model_from_run(run_dir: str, device: torch.device) -> Tuple[TinyMoEModel, Dict, str, Vocab]:
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

    # Get vocab from training data
    train_data_file = hparams.get("data", "data/tinystories_train.txt")
    if not os.path.exists(train_data_file):
        raise FileNotFoundError(f"Training data not found: {train_data_file}")

    with open(train_data_file, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    vocab = Vocab(stoi=stoi, itos=chars)

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
        vocab_size=vocab.size,
        seq_len=hparams.get("seq_len", 256),
        dim=hparams.get("dim", 256),
        layers=hparams.get("layers", 4),
        heads=hparams.get("heads", 4),
        dropout=hparams.get("dropout", 0.1),
        moe_kwargs=moe_kwargs,
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, _ = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        real_missing = [k for k in missing if "router_modules" not in k]
        if real_missing:
            print(f"Warning: missing keys: {real_missing}")
    model.to(device)
    model.eval()

    strategy = hparams.get("strategy", "unknown")
    return model, hparams, strategy, vocab


# ============================================================================
# Get routing assignments for text
# ============================================================================

def get_token_routing(
    model: TinyMoEModel,
    text: str,
    vocab: Vocab,
    device: torch.device,
    max_seq_len: int = 256,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Get routing assignments for each character in text.

    Returns:
        tokens: List of characters
        primary_experts: Primary expert assignment for each token
        expert_weights: Weight matrix [num_tokens, num_experts]
    """
    model.eval()
    num_experts = model.blocks[0].moe.num_experts

    # Truncate text to model's max sequence length
    text = text[:max_seq_len]

    # Encode text
    encoded = [vocab.stoi.get(c, 0) for c in text]
    x = torch.tensor([encoded], device=device)

    with torch.no_grad():
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

        # Primary expert is the first in topk
        primary_experts = topk_idx[:, 0].cpu().numpy()

        # Build weight matrix
        expert_weights = np.zeros((num_tokens, num_experts), dtype=np.float32)
        topk_idx_np = topk_idx.cpu().numpy()
        if gates is not None:
            gates_np = gates.cpu().numpy()
        else:
            gates_np = np.ones_like(topk_idx_np, dtype=np.float32) / k_eff

        for token_id in range(num_tokens):
            for k in range(topk_idx_np.shape[1]):
                expert_id = topk_idx_np[token_id, k]
                expert_weights[token_id, expert_id] = gates_np[token_id, k]

    tokens = list(text)
    return tokens, primary_experts, expert_weights


# ============================================================================
# Plotting
# ============================================================================

# Expert color palette (8 distinct colors)
EXPERT_COLORS = [
    "#e74c3c",  # Red - Expert 0
    "#3498db",  # Blue - Expert 1
    "#2ecc71",  # Green - Expert 2
    "#f39c12",  # Orange - Expert 3
    "#9b59b6",  # Purple - Expert 4
    "#1abc9c",  # Teal - Expert 5
    "#e91e63",  # Pink - Expert 6
    "#795548",  # Brown - Expert 7
]


def plot_colored_text_single(
    ax: plt.Axes,
    tokens: List[str],
    primary_experts: np.ndarray,
    strategy: str,
    max_chars_per_line: int = 80,
    num_experts: int = 8,
) -> None:
    """Plot colored text for a single strategy."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.set_title(strategy, fontsize=12, fontweight="bold", pad=10)

    # Layout parameters
    char_width = 0.0095
    line_height = 0.065
    x_start = 0.02
    y_start = 0.92

    x, y = x_start, y_start

    for i, (token, expert) in enumerate(zip(tokens, primary_experts)):
        color = EXPERT_COLORS[expert % len(EXPERT_COLORS)]

        # Handle newlines and special chars
        if token == '\n':
            x = x_start
            y -= line_height
            continue

        # Display character (use visible representation for spaces)
        display_char = token if token != ' ' else ' '

        # Add colored text
        ax.text(x, y, display_char, fontsize=9, fontfamily="monospace",
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", facecolor=color, alpha=0.2, edgecolor="none"))

        x += char_width

        # Wrap line
        if x > 0.98:
            x = x_start
            y -= line_height

        # Stop if we run out of vertical space
        if y < 0.05:
            break


def plot_colored_tokens(
    results: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]],
    out_path: str,
    num_experts: int = 8,
) -> None:
    """Plot colored token visualization for all strategies."""
    n_strategies = len(results)
    ncols = min(2, n_strategies)
    nrows = (n_strategies + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 6 * nrows))
    if n_strategies == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (strategy, (tokens, primary_experts, _)) in enumerate(results.items()):
        plot_colored_text_single(axes[idx], tokens, primary_experts, strategy,
                                 num_experts=num_experts)

    # Hide unused axes
    for idx in range(n_strategies, len(axes)):
        axes[idx].set_visible(False)

    # Add expert color legend
    legend_handles = [mpatches.Patch(color=EXPERT_COLORS[i], label=f"Expert {i}")
                      for i in range(num_experts)]
    fig.legend(handles=legend_handles, loc="lower center", ncol=num_experts,
               fontsize=9, title="Expert Assignment", title_fontsize=10,
               bbox_to_anchor=(0.5, 0.02))

    fig.suptitle("Token-Expert Routing Visualization (colored by primary expert)",
                 fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def plot_expert_distribution_by_char_type(
    results: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]],
    out_path: str,
) -> None:
    """Plot expert distribution by character type (letter, digit, space, punct)."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    char_types = {
        "letter": lambda c: c.isalpha(),
        "digit": lambda c: c.isdigit(),
        "space": lambda c: c.isspace(),
        "punct": lambda c: not c.isalnum() and not c.isspace(),
    }

    for ax, (strategy, (tokens, primary_experts, _)) in zip(axes, results.items()):
        # Count experts by char type
        type_expert_counts = {ct: np.zeros(8) for ct in char_types}

        for token, expert in zip(tokens, primary_experts):
            for ct_name, ct_fn in char_types.items():
                if ct_fn(token):
                    type_expert_counts[ct_name][expert] += 1
                    break

        # Normalize to percentages
        x = np.arange(8)
        width = 0.2
        colors = {"letter": "#3498db", "digit": "#e74c3c", "space": "#2ecc71", "punct": "#9b59b6"}

        for i, (ct_name, counts) in enumerate(type_expert_counts.items()):
            total = counts.sum()
            if total > 0:
                pct = counts / total * 100
            else:
                pct = counts
            ax.bar(x + i * width, pct, width, label=ct_name, color=colors[ct_name], alpha=0.8)

        ax.set_xlabel("Expert ID")
        ax.set_ylabel("% of char type")
        ax.set_title(strategy, fontweight="bold")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([str(i) for i in range(8)])
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Expert Preference by Character Type", fontsize=12, fontweight="bold")
    plt.tight_layout()

    out_summary = out_path.replace(".png", "_by_char_type.png")
    plt.savefig(out_summary, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_summary}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize token routing with colored text")
    parser.add_argument("--runs", type=str, required=True,
                        help="Comma-separated run directories or glob pattern")
    parser.add_argument("--data", type=str, default="data/tinystories_val.txt",
                        help="Data file for sample text")
    parser.add_argument("--num-chars", type=int, default=300,
                        help="Number of characters to visualize")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="results/token_routing_colored.png")
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

    # Filter valid runs
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
    print("-" * 60)

    # Load sample text
    with open(args.data, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Find a good sample (start of a sentence/story)
    sample_start = full_text.find("\n\n") + 2  # Skip header if any
    if sample_start < 2:
        sample_start = 0
    sample_text = full_text[sample_start:sample_start + args.num_chars]

    print(f"Sample text ({len(sample_text)} chars):")
    print(f"  '{sample_text[:80]}...'")
    print("-" * 60)

    results = {}
    for run_dir in valid_runs:
        print(f"  Loading {run_dir}...", end=" ", flush=True)
        try:
            model, hparams, strategy, vocab = load_model_from_run(run_dir, device)

            # Filter sample text to vocab
            filtered_text = "".join(c for c in sample_text if c in vocab.stoi)

            # Use model's seq_len to avoid positional embedding overflow
            seq_len = hparams.get("seq_len", 256)
            tokens, primary_experts, expert_weights = get_token_routing(
                model, filtered_text, vocab, device, max_seq_len=seq_len
            )

            display_name = strategy.replace("_", "-")
            results[display_name] = (tokens, primary_experts, expert_weights)

            # Report expert distribution
            expert_counts = np.bincount(primary_experts, minlength=8)
            print(f"strategy={strategy}, expert_dist={expert_counts}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("-" * 60)

    if not results:
        print("No results to plot!")
        return

    # Generate plots
    num_experts = hparams.get("num_experts", 8)
    plot_colored_tokens(results, args.out, num_experts)
    plot_expert_distribution_by_char_type(results, args.out)

    print("Done!")


if __name__ == "__main__":
    main()
