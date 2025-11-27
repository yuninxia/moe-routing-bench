#!/usr/bin/env python3
"""Generate MoE Evolution Timeline figure for the report.

Uses a table format for clarity.

Covers all core papers from the course requirements (Topic 2):
1. Sparsely-Gated MoE (Shazeer et al., ICLR 2017)
2. From Sparse to Soft MoE (Puigcerver et al., 2023)
3. Chain-of-Experts (Wang et al., 2024)
4. PERFT (Liu et al., 2024)
5. Learning to Specialize / DDOME (Farhat et al., 2023)

Plus key milestones: GShard, Switch Transformer, Expert Choice, DeepSeekMoE/V2/V3

Usage:
    python scripts/plot_moe_timeline.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# Academic style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'mathtext.fontset': 'stix',
    'figure.dpi': 150,
})

# Colors
COLORS = {
    'header': '#2c3e50',
    'row_even': '#f8f9fa',
    'row_odd': '#ffffff',
    'border': '#bdc3c7',
    'text': '#2c3e50',
    'highlight': '#e8f4f8',
}


def plot_moe_timeline(out_path: str) -> None:
    """Create MoE evolution timeline as a table."""

    # Data: (Year, Model, Venue, Routing Type, Key Contribution)
    data = [
        ('2017', 'Sparsely-Gated MoE', 'ICLR 2017', 'Token Choice (Top-K)',
         'Noisy gating + aux load-balance loss; 137B params'),
        ('2020', 'GShard', 'ICLR 2021', 'Token Choice (Top-2)',
         'Capacity factor + token dropping; 600B params'),
        ('2021', 'Switch Transformer', 'JMLR 2022', 'Token Choice (Top-1)',
         'Simplified to single expert; 1.6T params, 7× speedup'),
        ('2022', 'Expert Choice', 'NeurIPS 2022', 'Expert Choice',
         'Experts select tokens; perfect load balance'),
        ('2023', 'DDOME', 'arXiv 2023', 'Adaptive Gating',
         'Joint gating-expert training; 4-24% accuracy gains'),
        ('2023', 'Soft MoE', 'ICLR 2024', 'Soft Assignment',
         'Fully differentiable; no token dropping'),
        ('2024', 'DeepSeekMoE/V3', '2024', 'Hybrid',
         'Fine-grained + shared experts; aux-loss-free balancing'),
        ('2024', 'PERFT', 'arXiv 2024', 'Routed PEFT',
         'Routed adapter modules; 17% over MoE-agnostic'),
        ('2024', 'Chain-of-Experts', 'arXiv 2024', 'Sequential',
         'Expert-to-expert communication; 17-42% memory saving'),
    ]

    # Table dimensions
    n_rows = len(data)
    n_cols = 5
    col_widths = [0.8, 2.2, 1.3, 1.8, 4.0]  # Relative widths
    total_width = sum(col_widths)
    row_height = 0.5
    header_height = 0.6

    fig_width = 12
    fig_height = 0.8 + header_height + n_rows * row_height + 1.0

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, header_height + n_rows * row_height + 1.5)
    ax.axis('off')

    # Title
    ax.text(total_width / 2, header_height + n_rows * row_height + 1.1,
            'Evolution of MoE Routing Algorithms (2017–2024)',
            ha='center', fontsize=14, fontweight='bold', color=COLORS['text'])
    ax.text(total_width / 2, header_height + n_rows * row_height + 0.7,
            'From Sparse Gating to Auxiliary-Loss-Free Balancing',
            ha='center', fontsize=10, color='gray', style='italic')

    # Header
    headers = ['Year', 'Model', 'Venue', 'Routing Type', 'Key Contribution']
    x = 0
    y_header = n_rows * row_height

    for i, (header, width) in enumerate(zip(headers, col_widths)):
        # Header cell background
        rect = Rectangle((x, y_header), width, header_height,
                         facecolor=COLORS['header'], edgecolor=COLORS['border'], lw=1)
        ax.add_patch(rect)
        # Header text
        ax.text(x + width/2, y_header + header_height/2, header,
               ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        x += width

    # Data rows
    for row_idx, row_data in enumerate(data):
        y = (n_rows - 1 - row_idx) * row_height
        x = 0

        # Alternate row colors
        bg_color = COLORS['row_even'] if row_idx % 2 == 0 else COLORS['row_odd']

        for col_idx, (cell, width) in enumerate(zip(row_data, col_widths)):
            # Cell background
            rect = Rectangle((x, y), width, row_height,
                            facecolor=bg_color, edgecolor=COLORS['border'], lw=0.5)
            ax.add_patch(rect)

            # Cell text
            fontweight = 'bold' if col_idx == 1 else 'normal'
            fontsize = 9 if col_idx < 4 else 8
            ax.text(x + width/2, y + row_height/2, cell,
                   ha='center', va='center', fontsize=fontsize,
                   fontweight=fontweight, color=COLORS['text'],
                   wrap=True)
            x += width

    # Footer with key trends
    ax.text(total_width / 2, -0.4,
            'Key Trends:  Sparse → Soft routing  |  Token → Expert choice  |  '
            'Aux loss → Loss-free balancing  |  Dense → Routed PEFT',
            ha='center', fontsize=9, color='gray', style='italic')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    plot_moe_timeline("figures/moe_evolution_timeline.png")
