#!/usr/bin/env python3
"""Generate PERFT Architecture Diagram for the report.

Visualizes the four PERFT variants:
- PERFT-R (Routed): Independent routing among PEFT expert modules
- PERFT-E (Embedded): PEFT modules embedded within existing MoE using its routing
- PERFT-D (Dense): Multiple always-activated shared PEFT experts
- PERFT-S (Single): Single always-activated PEFT expert

Based on:
- PERFT Paper: https://arxiv.org/abs/2411.08212
- Source code: 3rdparty/PERFT-MoE/olmoe_modification/modeling_olmoe.py

Usage:
    python scripts/plot_perft_architecture.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
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
    'moe_expert': '#3498db',       # Blue - MoE FFN experts
    'adapter': '#e74c3c',          # Red - PEFT adapters (LoRA)
    'router': '#2ecc71',           # Green - Router
    'adapter_router': '#9b59b6',   # Purple - Adapter router (PERFT-R)
    'input': '#95a5a6',            # Gray - Input/output
    'combine': '#f39c12',          # Orange - Combine
    'frozen': '#bdc3c7',           # Light gray - Frozen
    'text': '#2c3e50',
    'border': '#34495e',
    'background': '#f8f9fa',
}


def draw_box(ax, x, y, w, h, color, label='', fontsize=8, alpha=1.0,
             edgecolor=None, linestyle='-', textcolor='white', fontweight='normal'):
    """Draw a rounded box with label."""
    if edgecolor is None:
        edgecolor = COLORS['border']
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color, edgecolor=edgecolor,
        linewidth=1.5, alpha=alpha, linestyle=linestyle
    )
    ax.add_patch(box)
    if label:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
               fontsize=fontsize, color=textcolor, fontweight=fontweight, wrap=True)


def draw_arrow(ax, start, end, color='black', style='->', connectionstyle='arc3,rad=0'):
    """Draw an arrow."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        connectionstyle=connectionstyle,
        color=color,
        linewidth=1.5,
        mutation_scale=12
    )
    ax.add_patch(arrow)


def draw_perft_variant(ax, x_offset, y_offset, variant_name, description):
    """Draw a single PERFT variant diagram."""
    # Constants
    box_h = 0.6
    expert_w = 0.8
    router_w = 1.2
    adapter_w = 0.6
    gap = 0.3

    # Title
    ax.text(x_offset + 2.5, y_offset + 4.2, variant_name,
           ha='center', fontsize=11, fontweight='bold', color=COLORS['text'])
    ax.text(x_offset + 2.5, y_offset + 3.85, description,
           ha='center', fontsize=7, color='gray', style='italic')

    # Input
    draw_box(ax, x_offset + 1.8, y_offset + 3.0, 1.4, 0.5, COLORS['input'],
            'Input h', fontsize=8, textcolor=COLORS['text'])

    if variant_name == 'PERFT-R':
        # Two separate routers
        # MoE Router (frozen)
        draw_box(ax, x_offset + 0.3, y_offset + 2.0, router_w, box_h, COLORS['frozen'],
                'MoE Router\n(frozen)', fontsize=7, textcolor=COLORS['text'], linestyle='--')
        # Adapter Router (trainable)
        draw_box(ax, x_offset + 3.5, y_offset + 2.0, router_w, box_h, COLORS['adapter_router'],
                'Adapter Router\n(trainable)', fontsize=7)

        # Arrows from input to routers
        draw_arrow(ax, (x_offset + 2.0, y_offset + 3.0), (x_offset + 0.9, y_offset + 2.6))
        draw_arrow(ax, (x_offset + 3.0, y_offset + 3.0), (x_offset + 4.1, y_offset + 2.6))

        # MoE Experts (frozen)
        for i in range(3):
            draw_box(ax, x_offset + 0.1 + i*0.5, y_offset + 0.9, 0.45, box_h, COLORS['frozen'],
                    f'E{i+1}', fontsize=7, textcolor=COLORS['text'], linestyle='--')

        # Adapter Experts (trainable LoRA)
        for i in range(3):
            draw_box(ax, x_offset + 3.3 + i*0.5, y_offset + 0.9, 0.45, box_h, COLORS['adapter'],
                    f'Δ{i+1}', fontsize=7)

        # Arrows from routers to experts
        draw_arrow(ax, (x_offset + 0.9, y_offset + 2.0), (x_offset + 0.55, y_offset + 1.5))
        draw_arrow(ax, (x_offset + 4.1, y_offset + 2.0), (x_offset + 3.75, y_offset + 1.5))

        # Combine outputs
        draw_box(ax, x_offset + 1.8, y_offset + 0.0, 1.4, 0.5, COLORS['combine'],
                'y = MoE + Δ + h', fontsize=7)

        # Arrows to output
        draw_arrow(ax, (x_offset + 0.55, y_offset + 0.9), (x_offset + 2.0, y_offset + 0.5))
        draw_arrow(ax, (x_offset + 3.75, y_offset + 0.9), (x_offset + 3.0, y_offset + 0.5))

    elif variant_name == 'PERFT-E':
        # Single router (frozen)
        draw_box(ax, x_offset + 1.4, y_offset + 2.0, router_w, box_h, COLORS['frozen'],
                'MoE Router\n(frozen)', fontsize=7, textcolor=COLORS['text'], linestyle='--')

        # Arrow from input to router
        draw_arrow(ax, (x_offset + 2.5, y_offset + 3.0), (x_offset + 2.0, y_offset + 2.6))

        # Combined Expert + Adapter pairs
        for i in range(3):
            # Expert (frozen)
            draw_box(ax, x_offset + 0.5 + i*1.6, y_offset + 0.9, 0.6, box_h, COLORS['frozen'],
                    f'E{i+1}', fontsize=7, textcolor=COLORS['text'], linestyle='--')
            # Adapter embedded
            draw_box(ax, x_offset + 1.15 + i*1.6, y_offset + 0.9, 0.4, box_h, COLORS['adapter'],
                    f'Δ{i+1}', fontsize=6)

        # Arrow from router to experts
        draw_arrow(ax, (x_offset + 2.0, y_offset + 2.0), (x_offset + 2.0, y_offset + 1.5))

        # Combine output
        draw_box(ax, x_offset + 1.8, y_offset + 0.0, 1.4, 0.5, COLORS['combine'],
                'y = Σ G(E+Δ) + h', fontsize=7)

        # Arrow to output
        draw_arrow(ax, (x_offset + 2.0, y_offset + 0.9), (x_offset + 2.5, y_offset + 0.5))

    elif variant_name == 'PERFT-D':
        # MoE Router (frozen)
        draw_box(ax, x_offset + 0.8, y_offset + 2.0, router_w, box_h, COLORS['frozen'],
                'MoE Router\n(frozen)', fontsize=7, textcolor=COLORS['text'], linestyle='--')

        # Arrow from input
        draw_arrow(ax, (x_offset + 1.8, y_offset + 3.0), (x_offset + 1.4, y_offset + 2.6))
        draw_arrow(ax, (x_offset + 3.2, y_offset + 3.0), (x_offset + 3.8, y_offset + 2.0))

        # MoE Experts (frozen)
        for i in range(3):
            draw_box(ax, x_offset + 0.3 + i*0.6, y_offset + 0.9, 0.55, box_h, COLORS['frozen'],
                    f'E{i+1}', fontsize=7, textcolor=COLORS['text'], linestyle='--')

        # Dense Adapters (always active, no routing)
        for i in range(2):
            draw_box(ax, x_offset + 3.3 + i*0.7, y_offset + 0.9, 0.6, box_h, COLORS['adapter'],
                    f'Δ{i+1}', fontsize=7)

        # Arrow from router to experts
        draw_arrow(ax, (x_offset + 1.4, y_offset + 2.0), (x_offset + 1.1, y_offset + 1.5))

        # "Always Active" label
        ax.text(x_offset + 3.8, y_offset + 1.7, 'Always\nActive',
               ha='center', fontsize=6, color='gray', style='italic')

        # Combine output
        draw_box(ax, x_offset + 1.8, y_offset + 0.0, 1.4, 0.5, COLORS['combine'],
                'y = MoE + ΣΔ + h', fontsize=7)

        # Arrows to output
        draw_arrow(ax, (x_offset + 1.1, y_offset + 0.9), (x_offset + 2.0, y_offset + 0.5))
        draw_arrow(ax, (x_offset + 3.6, y_offset + 0.9), (x_offset + 3.0, y_offset + 0.5))

    elif variant_name == 'PERFT-S':
        # MoE Router (frozen)
        draw_box(ax, x_offset + 0.8, y_offset + 2.0, router_w, box_h, COLORS['frozen'],
                'MoE Router\n(frozen)', fontsize=7, textcolor=COLORS['text'], linestyle='--')

        # Arrow from input
        draw_arrow(ax, (x_offset + 1.8, y_offset + 3.0), (x_offset + 1.4, y_offset + 2.6))
        draw_arrow(ax, (x_offset + 3.2, y_offset + 3.0), (x_offset + 3.8, y_offset + 2.0))

        # MoE Experts (frozen)
        for i in range(3):
            draw_box(ax, x_offset + 0.3 + i*0.6, y_offset + 0.9, 0.55, box_h, COLORS['frozen'],
                    f'E{i+1}', fontsize=7, textcolor=COLORS['text'], linestyle='--')

        # Single Adapter (always active)
        draw_box(ax, x_offset + 3.3, y_offset + 0.9, 1.0, box_h, COLORS['adapter'],
                'Δ₀', fontsize=8)

        # Arrow from router to experts
        draw_arrow(ax, (x_offset + 1.4, y_offset + 2.0), (x_offset + 1.1, y_offset + 1.5))

        # "Always Active" label
        ax.text(x_offset + 3.8, y_offset + 1.7, 'Always\nActive',
               ha='center', fontsize=6, color='gray', style='italic')

        # Combine output
        draw_box(ax, x_offset + 1.8, y_offset + 0.0, 1.4, 0.5, COLORS['combine'],
                'y = MoE + Δ₀ + h', fontsize=7)

        # Arrows to output
        draw_arrow(ax, (x_offset + 1.1, y_offset + 0.9), (x_offset + 2.0, y_offset + 0.5))
        draw_arrow(ax, (x_offset + 3.8, y_offset + 0.9), (x_offset + 3.0, y_offset + 0.5))


def plot_perft_architecture(out_path: str) -> None:
    """Create PERFT architecture comparison figure."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Title
    fig.suptitle('PERFT: Parameter-Efficient Routed Fine-Tuning Variants',
                fontsize=14, fontweight='bold', y=0.98)
    fig.text(0.5, 0.94,
            'Integrating LoRA adapters with MoE routing for efficient fine-tuning (Liu et al., 2024)',
            ha='center', fontsize=10, color='gray', style='italic')

    variants = [
        ('PERFT-R', 'Independent routing for adapter experts'),
        ('PERFT-E', 'Adapters embedded within MoE experts'),
        ('PERFT-D', 'Multiple dense (always-active) adapters'),
        ('PERFT-S', 'Single shared adapter'),
    ]

    for idx, (ax, (name, desc)) in enumerate(zip(axes.flat, variants)):
        ax.set_xlim(0, 5)
        ax.set_ylim(-0.5, 5)
        ax.set_aspect('equal')
        ax.axis('off')

        draw_perft_variant(ax, 0, 0, name, desc)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['frozen'], edgecolor=COLORS['border'],
                      linestyle='--', label='Frozen (Base MoE)'),
        mpatches.Patch(facecolor=COLORS['adapter'], edgecolor=COLORS['border'],
                      label='Trainable LoRA Adapter'),
        mpatches.Patch(facecolor=COLORS['adapter_router'], edgecolor=COLORS['border'],
                      label='Trainable Adapter Router'),
        mpatches.Patch(facecolor=COLORS['combine'], edgecolor=COLORS['border'],
                      label='Output Combine'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
              fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.01))

    # Mathematical formulations
    formulas = [
        r'$\mathbf{PERFT\text{-}R}:\ y = \sum_i G_i E_i(h) + \sum_j \tilde{G}_j \Delta_j(h) + h$',
        r'$\mathbf{PERFT\text{-}E}:\ y = \sum_i G_i (E_i(h) + \Delta_i(h)) + h$',
        r'$\mathbf{PERFT\text{-}D}:\ y = \sum_i G_i E_i(h) + \sum_j \Delta_j(h) + h$',
        r'$\mathbf{PERFT\text{-}S}:\ y = \sum_i G_i E_i(h) + \Delta_0(h) + h$',
    ]

    fig.text(0.5, 0.06, '  |  '.join(formulas), ha='center', fontsize=8,
            family='serif', style='italic')

    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    plt.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Saved: {out_path}")
    plt.close()


def plot_perft_architecture_single(out_path: str) -> None:
    """Create a single comprehensive PERFT diagram showing all variants side by side."""

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.6, 'PERFT: Parameter-Efficient Routed Fine-Tuning',
           ha='center', fontsize=14, fontweight='bold', color=COLORS['text'])
    ax.text(7, 7.25, 'Four variants for integrating LoRA adapters with Mixture-of-Experts (Liu et al., 2024)',
           ha='center', fontsize=10, color='gray', style='italic')

    # Draw four variants with spacing
    x_positions = [0.5, 4.0, 7.5, 11.0]
    variants = [
        ('PERFT-R', 'Independent\nAdapter Routing', 'best'),
        ('PERFT-E', 'Embedded in\nMoE Experts', 'good'),
        ('PERFT-D', 'Dense Shared\nAdapters', 'baseline'),
        ('PERFT-S', 'Single Shared\nAdapter', 'baseline'),
    ]

    box_w = 0.5
    box_h = 0.45

    for x_off, (name, desc, perf) in zip(x_positions, variants):
        # Variant title
        ax.text(x_off + 1.3, 6.8, name, ha='center', fontsize=11,
               fontweight='bold', color=COLORS['text'])
        ax.text(x_off + 1.3, 6.5, desc, ha='center', fontsize=8,
               color='gray', style='italic')

        # Performance badge
        badge_colors = {'best': '#27ae60', 'good': '#3498db', 'baseline': '#95a5a6'}
        badge_text = {'best': 'Best', 'good': 'Good', 'baseline': 'Baseline'}
        ax.text(x_off + 2.3, 6.8, badge_text[perf], ha='center', fontsize=7,
               color='white', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor=badge_colors[perf], edgecolor='none'))

        # Input
        draw_box(ax, x_off + 0.8, 5.8, 1.0, 0.4, COLORS['input'], 'Input h',
                fontsize=8, textcolor=COLORS['text'])

        if name == 'PERFT-R':
            # Two routers
            draw_box(ax, x_off + 0.2, 4.8, 1.0, 0.5, COLORS['frozen'], 'MoE\nRouter',
                    fontsize=7, textcolor=COLORS['text'], linestyle='--')
            draw_box(ax, x_off + 1.4, 4.8, 1.0, 0.5, COLORS['adapter_router'], 'LoRA\nRouter',
                    fontsize=7)

            # Arrows from input
            draw_arrow(ax, (x_off + 1.0, 5.8), (x_off + 0.7, 5.3))
            draw_arrow(ax, (x_off + 1.6, 5.8), (x_off + 1.9, 5.3))

            # MoE Experts
            for i in range(2):
                draw_box(ax, x_off + 0.1 + i*0.55, 3.6, box_w, box_h, COLORS['frozen'],
                        f'E{i+1}', fontsize=7, textcolor=COLORS['text'], linestyle='--')

            # LoRA Experts
            for i in range(2):
                draw_box(ax, x_off + 1.3 + i*0.55, 3.6, box_w, box_h, COLORS['adapter'],
                        f'Δ{i+1}', fontsize=7)

            # Arrows
            draw_arrow(ax, (x_off + 0.7, 4.8), (x_off + 0.55, 4.05))
            draw_arrow(ax, (x_off + 1.9, 4.8), (x_off + 1.75, 4.05))

            # Combine
            draw_box(ax, x_off + 0.6, 2.7, 1.4, 0.5, COLORS['combine'],
                    'y = MoE + Δ + h', fontsize=7)
            draw_arrow(ax, (x_off + 0.55, 3.6), (x_off + 1.0, 3.2))
            draw_arrow(ax, (x_off + 1.75, 3.6), (x_off + 1.6, 3.2))

        elif name == 'PERFT-E':
            # Single router
            draw_box(ax, x_off + 0.6, 4.8, 1.4, 0.5, COLORS['frozen'], 'MoE Router\n(shared)',
                    fontsize=7, textcolor=COLORS['text'], linestyle='--')

            draw_arrow(ax, (x_off + 1.3, 5.8), (x_off + 1.3, 5.3))

            # Expert + Adapter pairs
            for i in range(2):
                # Combined box
                x = x_off + 0.3 + i*1.2
                draw_box(ax, x, 3.6, 0.65, box_h, COLORS['frozen'],
                        f'E{i+1}', fontsize=7, textcolor=COLORS['text'], linestyle='--')
                draw_box(ax, x + 0.55, 3.6, 0.35, box_h, COLORS['adapter'],
                        f'Δ', fontsize=6)

            draw_arrow(ax, (x_off + 1.3, 4.8), (x_off + 1.3, 4.05))

            # Combine
            draw_box(ax, x_off + 0.6, 2.7, 1.4, 0.5, COLORS['combine'],
                    'y = Σ G(E+Δ) + h', fontsize=7)
            draw_arrow(ax, (x_off + 1.3, 3.6), (x_off + 1.3, 3.2))

        elif name == 'PERFT-D':
            # Router
            draw_box(ax, x_off + 0.3, 4.8, 1.0, 0.5, COLORS['frozen'], 'MoE\nRouter',
                    fontsize=7, textcolor=COLORS['text'], linestyle='--')

            draw_arrow(ax, (x_off + 0.9, 5.8), (x_off + 0.8, 5.3))
            draw_arrow(ax, (x_off + 1.7, 5.8), (x_off + 1.9, 4.1))

            # MoE Experts
            for i in range(2):
                draw_box(ax, x_off + 0.1 + i*0.55, 3.6, box_w, box_h, COLORS['frozen'],
                        f'E{i+1}', fontsize=7, textcolor=COLORS['text'], linestyle='--')

            # Dense adapters
            for i in range(2):
                draw_box(ax, x_off + 1.3 + i*0.55, 3.6, box_w, box_h, COLORS['adapter'],
                        f'Δ{i+1}', fontsize=7)

            draw_arrow(ax, (x_off + 0.8, 4.8), (x_off + 0.55, 4.05))
            ax.text(x_off + 1.85, 4.3, 'dense', fontsize=6, color='gray', ha='center')

            # Combine
            draw_box(ax, x_off + 0.6, 2.7, 1.4, 0.5, COLORS['combine'],
                    'y = MoE + ΣΔ + h', fontsize=7)
            draw_arrow(ax, (x_off + 0.55, 3.6), (x_off + 1.0, 3.2))
            draw_arrow(ax, (x_off + 1.75, 3.6), (x_off + 1.6, 3.2))

        elif name == 'PERFT-S':
            # Router
            draw_box(ax, x_off + 0.3, 4.8, 1.0, 0.5, COLORS['frozen'], 'MoE\nRouter',
                    fontsize=7, textcolor=COLORS['text'], linestyle='--')

            draw_arrow(ax, (x_off + 0.9, 5.8), (x_off + 0.8, 5.3))
            draw_arrow(ax, (x_off + 1.7, 5.8), (x_off + 2.0, 4.1))

            # MoE Experts
            for i in range(2):
                draw_box(ax, x_off + 0.1 + i*0.55, 3.6, box_w, box_h, COLORS['frozen'],
                        f'E{i+1}', fontsize=7, textcolor=COLORS['text'], linestyle='--')

            # Single adapter
            draw_box(ax, x_off + 1.5, 3.6, 0.8, box_h, COLORS['adapter'],
                    'Δ₀', fontsize=8)

            draw_arrow(ax, (x_off + 0.8, 4.8), (x_off + 0.55, 4.05))
            ax.text(x_off + 1.9, 4.3, 'single', fontsize=6, color='gray', ha='center')

            # Combine
            draw_box(ax, x_off + 0.6, 2.7, 1.4, 0.5, COLORS['combine'],
                    'y = MoE + Δ₀ + h', fontsize=7)
            draw_arrow(ax, (x_off + 0.55, 3.6), (x_off + 1.0, 3.2))
            draw_arrow(ax, (x_off + 1.9, 3.6), (x_off + 1.6, 3.2))

    # Separator lines
    for x in [3.5, 7.0, 10.5]:
        ax.axvline(x=x, ymin=0.25, ymax=0.9, color='#ecf0f1', linewidth=1.5, linestyle='-')

    # Legend box
    legend_y = 1.5
    legend_items = [
        (1.5, COLORS['frozen'], '--', 'Frozen (Base MoE)'),
        (4.5, COLORS['adapter'], '-', 'Trainable LoRA'),
        (7.3, COLORS['adapter_router'], '-', 'Trainable Router'),
        (10.3, COLORS['combine'], '-', 'Output Combine'),
    ]

    for x, color, ls, label in legend_items:
        draw_box(ax, x, legend_y, 0.5, 0.35, color, '', linestyle=ls)
        ax.text(x + 0.65, legend_y + 0.17, label, fontsize=8, va='center', color=COLORS['text'])

    # Key insight box
    ax.add_patch(FancyBboxPatch(
        (0.3, 0.3), 13.4, 0.9,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor='#f8f9fa', edgecolor='#3498db', linewidth=1.5
    ))
    ax.text(7, 0.95, 'Key Findings (Liu et al., 2024)',
           ha='center', fontsize=9, fontweight='bold', color=COLORS['text'])
    ax.text(7, 0.55,
           'PERFT-R > PERFT-E > PERFT-D/S  |  Up to 17% improvement over MoE-agnostic baselines  |  '
           'Independent adapter routing enables task-specific expert specialization',
           ha='center', fontsize=8, color='gray')

    plt.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)

    # Generate the single horizontal layout
    plot_perft_architecture_single("figures/perft_architecture.png")
