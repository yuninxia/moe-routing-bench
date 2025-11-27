#!/usr/bin/env python3
"""Generate publication-quality architecture diagrams for MoE Routing Bench.

Style: Academic paper figures (ICML/NeurIPS/ICLR)
- Clean, minimal design
- Grayscale + limited accent colors
- Mathematical notation
- Vector-style rendering

Usage:
    python scripts/plot_architecture_academic.py --out-dir figures/
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
import numpy as np

# Academic style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'mathtext.fontset': 'stix',
    'figure.dpi': 150,
})

# Color palette (minimal, professional)
COLORS = {
    'primary': '#2c3e50',      # Dark blue-gray
    'secondary': '#7f8c8d',    # Gray
    'accent1': '#3498db',      # Blue (for highlighting)
    'accent2': '#e74c3c',      # Red (for drops/errors)
    'accent3': '#27ae60',      # Green (for success)
    'light': '#ecf0f1',        # Light gray background
    'white': '#ffffff',
}


# ============================================================================
# Figure 1: MoE Layer Architecture (Academic Style)
# ============================================================================

def plot_moe_architecture_academic(out_path: str) -> None:
    """Publication-quality MoE architecture diagram."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    C = COLORS

    def box(x, y, w, h, text, fill='white', edge='primary', lw=1.5, fontsize=9):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=C.get(fill, fill), edgecolor=C.get(edge, edge),
                              linewidth=lw)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=C['primary'])

    def arrow(x1, y1, x2, y2, style='->', color='secondary'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle=style, color=C.get(color, color),
                                  lw=1.2, shrinkA=2, shrinkB=2))

    def text(x, y, s, **kwargs):
        # Convert color names to actual colors
        if 'color' in kwargs and kwargs['color'] in C:
            kwargs['color'] = C[kwargs['color']]
        defaults = {'ha': 'center', 'va': 'center', 'fontsize': 9, 'color': C['primary']}
        defaults.update(kwargs)
        ax.text(x, y, s, **defaults)

    # Title
    text(7, 8.6, 'MoE Feed-Forward Layer', fontsize=12, fontweight='bold')

    # Input
    box(0.5, 6.5, 2.2, 0.8, r'$\mathbf{x} \in \mathbb{R}^{T \times d}$', fill='light')
    text(1.6, 6.1, 'Input', fontsize=8, color='secondary')

    # Router
    arrow(2.7, 6.9, 3.8, 6.9)
    box(3.8, 6.3, 2.5, 1.2, r'Router $W_r$' + '\n' + r'$\mathbf{g} = W_r \mathbf{x}$', fill='light')
    text(5.05, 5.9, r'$\mathbf{g} \in \mathbb{R}^{T \times E}$', fontsize=8, color='secondary')

    # Top-K Selection (This is where the 5 routing strategies apply)
    arrow(6.3, 6.9, 7.4, 6.9)
    box(7.4, 6.3, 2.8, 1.2, r'Routing Strategy' + '\n' + r'$\mathcal{I}, \mathbf{w} = \text{Route}(\mathbf{g})$')
    # Move annotation to right side of the box, not overlapping with arrow
    text(10.5, 7.1, r'$\mathcal{I} \in \mathbb{Z}^{T \times k}$', fontsize=8, color='secondary', ha='left')

    # List the 5 routing strategies
    text(10.5, 6.65, r'top-1 | top-$k$ hard | soft-$k$', fontsize=7, color='secondary', ha='left')
    text(10.5, 6.35, r'hash | expert-choice', fontsize=7, color='secondary', ha='left')

    # Dispatch (Pack)
    arrow(8.8, 6.3, 8.8, 5.5)
    box(7.4, 4.3, 2.8, 1.1, r'Dispatch' + '\n' + r'$\mathbf{x}_e = \text{Pack}(\mathbf{x}, \mathcal{I})$')
    # Move capacity formula and tensor shape annotation to not overlap with arrow
    text(11.0, 5.2, r'$C = \lceil \alpha \cdot \frac{Tk}{E} \rceil$', fontsize=8, color='secondary', ha='left')
    text(11.0, 4.7, r'$\mathbf{x}_e \in \mathbb{R}^{E \times C \times d}$', fontsize=8, color='secondary', ha='left')

    # Experts
    arrow(8.8, 4.3, 8.8, 3.5)

    # Draw E expert boxes
    expert_y = 2.4
    expert_w = 1.1
    num_show = 6
    total_w = num_show * expert_w + 0.8
    start_x = 8.8 - total_w / 2

    for i in range(num_show):
        x = start_x + i * (expert_w + 0.15)
        if i < 3:
            label = f'$f_{i}$'
        elif i == 3:
            label = r'$\cdots$'
            box(x, expert_y, expert_w, 0.8, label, fill='white', edge='secondary', lw=1)
            continue
        else:
            label = f'$f_{{{i-1}}}$' if i == 4 else f'$f_{{E-1}}$'
        box(x, expert_y, expert_w, 0.8, label, fill='light')

    text(8.8, 3.4, r'Expert FFNs: $f_e(\mathbf{x}) = W_2 \sigma(W_1 \mathbf{x})$', fontsize=9)

    # Combine
    arrow(8.8, 2.4, 8.8, 1.6)
    box(7.4, 0.5, 2.8, 1.0, r'Combine' + '\n' + r'$\mathbf{y} = \sum_e w_e \cdot f_e(\mathbf{x}_e)$')

    # Output
    arrow(10.2, 1.0, 11.3, 1.0)
    box(11.3, 0.6, 2.2, 0.8, r'$\mathbf{y} \in \mathbb{R}^{T \times d}$', fill='light')
    text(12.4, 0.2, 'Output', fontsize=8, color='secondary')

    # Side panel: Parameters
    text(2, 4.5, 'Parameters', fontsize=9, ha='left', fontweight='bold')
    params = [
        r'$T$: tokens',
        r'$d$: hidden dim',
        r'$E$: num experts',
        r'$k$: top-$k$',
        r'$\alpha$: capacity factor',
    ]
    for i, p in enumerate(params):
        text(2, 4.0 - i * 0.45, p, fontsize=8, ha='left', color='secondary')

    # Side panel: This work
    text(2, 1.8, 'This work', fontsize=9, ha='left', fontweight='bold')
    configs = [
        r'$E=8$, $k \in \{1,2\}$',
        r'$\alpha \in \{1.0, 1.25\}$',
        r'5 routing strategies',
    ]
    for i, c in enumerate(configs):
        text(2, 1.35 - i * 0.4, c, fontsize=8, ha='left', color='secondary')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Saved: {out_path}")
    plt.close()


# ============================================================================
# Figure 2: Token Dropping Mechanism (Academic Style)
# ============================================================================

def plot_token_dropping_academic(out_path: str) -> None:
    """Publication-quality token dropping illustration."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    C = COLORS

    # ===== Left: Capacity and slot assignment =====
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title at top (will align with right panel using fig.text later)
    ax.text(5, 7.8, '(a) Capacity-based Slot Assignment', ha='center', fontsize=10, fontweight='bold')

    # Tokens - move label above the circles to avoid overlap
    ax.text(0.5, 7.2, 'Tokens:', fontsize=9, ha='left')
    token_x = [1.5, 2.3, 3.1, 3.9, 4.7, 5.5, 6.3, 7.1]
    for i, x in enumerate(token_x):
        circle = Circle((x, 6.8), 0.25, facecolor=C['light'], edgecolor=C['primary'], lw=1)
        ax.add_patch(circle)
        ax.text(x, 6.8, f'{i}', ha='center', va='center', fontsize=8)

    # Routing arrows (example: top-2 routing)
    ax.text(5, 6.1, r'Top-$k$ routing ($k=2$)', ha='center', fontsize=8, color=C['secondary'])

    # Expert buffers
    capacity = 3
    assignments = {
        0: [0, 2, 4, 6],  # 4 tokens, capacity=3, drop 1
        1: [1, 3],        # 2 tokens, fits
        2: [0, 5, 7],     # 3 tokens, fits exactly
        3: [2, 4, 6, 7],  # 4 tokens, drop 1
    }

    for e_id in range(4):
        y = 4.5 - e_id * 1.0
        ax.text(0.5, y + 0.15, f'$e_{e_id}$:', fontsize=9, ha='left')

        # Draw capacity slots
        for s in range(capacity):
            x = 1.8 + s * 0.9
            rect = Rectangle((x - 0.35, y - 0.25), 0.7, 0.5,
                            facecolor='white', edgecolor=C['secondary'], lw=1)
            ax.add_patch(rect)

        # Fill assigned tokens
        tokens = assignments[e_id]
        for s, t in enumerate(tokens[:capacity]):
            x = 1.8 + s * 0.9
            rect = Rectangle((x - 0.35, y - 0.25), 0.7, 0.5,
                            facecolor=C['light'], edgecolor=C['primary'], lw=1)
            ax.add_patch(rect)
            ax.text(x, y, f'{t}', ha='center', va='center', fontsize=8)

        # Show dropped tokens
        if len(tokens) > capacity:
            x_drop = 1.8 + capacity * 0.9 + 0.3
            for i, t in enumerate(tokens[capacity:]):
                rect = Rectangle((x_drop + i * 0.8 - 0.35, y - 0.25), 0.7, 0.5,
                                facecolor='white', edgecolor=C['accent2'], lw=1, ls='--')
                ax.add_patch(rect)
                ax.text(x_drop + i * 0.8, y, f'{t}', ha='center', va='center',
                       fontsize=8, color=C['accent2'])
                ax.text(x_drop + i * 0.8, y + 0.4, '×', ha='center', fontsize=10, color=C['accent2'])

    # Legend
    ax.text(6.5, 4.6, 'kept', fontsize=8, color=C['primary'])
    ax.text(6.5, 4.2, 'dropped', fontsize=8, color=C['accent2'])

    # Caption
    ax.text(5, 0.3, r'capacity factor $\alpha$ controls buffer size',
           ha='center', fontsize=8, color=C['secondary'])

    # ===== Right: Drop rate vs capacity factor =====
    ax = axes[1]
    ax.set_title('(b) Drop Rate vs Capacity Factor', fontsize=10, fontweight='bold', pad=10)

    # Simulated data (based on actual experimental results)
    cf_values = [0.9, 1.0, 1.05, 1.1, 1.25, 1.5]
    drop_rates = [0.12, 0.03, 0.005, 0.001, 0.0, 0.0]

    ax.plot(cf_values, drop_rates, 'o-', color=C['primary'], lw=1.5, markersize=5)
    ax.fill_between(cf_values, drop_rates, alpha=0.2, color=C['primary'])

    # Highlight recommended region
    ax.axvspan(1.05, 1.25, alpha=0.15, color=C['accent3'], label='Recommended')
    ax.axhline(y=0.01, color=C['secondary'], ls='--', lw=0.8, alpha=0.7)

    ax.set_xlabel(r'Capacity Factor $\alpha$', fontsize=9)
    ax.set_ylabel('Drop Rate', fontsize=9)
    ax.set_xlim(0.85, 1.55)
    ax.set_ylim(-0.005, 0.14)
    ax.set_xticks([0.9, 1.0, 1.1, 1.25, 1.5])
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotation
    ax.annotate(r'$\alpha \geq 1.05$: near-zero drops', xy=(1.1, 0.001), xytext=(1.2, 0.05),
               fontsize=8, color=C['secondary'],
               arrowprops=dict(arrowstyle='->', color=C['secondary'], lw=0.8))

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Saved: {out_path}")
    plt.close()


# ============================================================================
# Figure 3: Routing Strategies Comparison (Academic Style)
# ============================================================================

def plot_routing_strategies_academic(out_path: str) -> None:
    """Publication-quality routing strategy comparison."""
    fig, axes = plt.subplots(1, 5, figsize=(10, 3.5))
    C = COLORS

    strategies = [
        ('Top-1', r'$\mathcal{I} = \arg\max(\mathbf{g})$', 'T→1 expert',
         [(0,0), (1,1), (2,0), (3,2)], False),
        ('Top-$k$ Hard', r'$\mathcal{I} = \text{TopK}(\mathbf{g})$', 'T→k experts, uniform',
         [(0,[0,1]), (1,[1,2]), (2,[0,2]), (3,[2,3])], False),
        ('Soft Top-$k$', r'$w = \text{softmax}(\mathbf{g}_{\mathcal{I}})$', 'T→k experts, learned',
         [(0,[(0,0.7),(1,0.3)]), (1,[(1,0.6),(2,0.4)]), (2,[(0,0.4),(2,0.6)]), (3,[(2,0.5),(3,0.5)])], True),
        ('Hash', r'$\mathcal{I} = h(\mathrm{pos})\ \mathrm{mod}\ E$', 'deterministic, uniform',
         [(0,[0,1]), (1,[1,2]), (2,[2,3]), (3,[3,0])], False),
        ('Expert Choice', r'$\mathcal{I}_e = \text{TopK}_e(\mathbf{g}^T)$', 'E→T selection',
         [(0,[0,2]), (1,[1,3]), (2,[0,1]), (3,[2,3])], True),
    ]

    for ax, (title, formula, desc, routes, has_weights) in zip(axes, strategies):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Title
        ax.text(5, 9.5, title, ha='center', fontsize=10, fontweight='bold')
        ax.text(5, 8.8, formula, ha='center', fontsize=8, color=C['secondary'])

        # Tokens (top row)
        for i in range(4):
            x = 2 + i * 1.8
            circle = Circle((x, 7.2), 0.35, facecolor=C['light'],
                           edgecolor=C['primary'], lw=1)
            ax.add_patch(circle)
            ax.text(x, 7.2, f'$t_{i}$', ha='center', va='center', fontsize=9)

        # Experts (bottom row)
        for i in range(4):
            x = 2 + i * 1.8
            rect = FancyBboxPatch((x-0.45, 3.5), 0.9, 0.7, boxstyle="round,pad=0.02",
                                  facecolor=C['light'], edgecolor=C['primary'], lw=1)
            ax.add_patch(rect)
            ax.text(x, 3.85, f'$e_{i}$', ha='center', va='center', fontsize=9)

        # Draw routing arrows
        if title == 'Expert Choice':
            # Expert → Token (reversed)
            for e_id, t_ids in routes:
                ex = 2 + e_id * 1.8
                for t_id in t_ids:
                    tx = 2 + t_id * 1.8
                    ax.annotate('', xy=(tx, 6.85), xytext=(ex, 4.2),
                               arrowprops=dict(arrowstyle='->', color=C['accent1'],
                                             lw=1, alpha=0.7))
        elif title == 'Top-1':
            for t_id, e_id in routes:
                tx = 2 + t_id * 1.8
                ex = 2 + e_id * 1.8
                ax.annotate('', xy=(ex, 4.2), xytext=(tx, 6.85),
                           arrowprops=dict(arrowstyle='->', color=C['primary'], lw=1.2))
        elif has_weights:
            for t_id, experts in routes:
                tx = 2 + t_id * 1.8
                for e_id, w in experts:
                    ex = 2 + e_id * 1.8
                    ax.annotate('', xy=(ex, 4.2), xytext=(tx, 6.85),
                               arrowprops=dict(arrowstyle='->', color=C['primary'],
                                             lw=0.5 + w*1.5, alpha=0.5 + w*0.5))
        else:
            for t_id, e_ids in routes:
                tx = 2 + t_id * 1.8
                for e_id in e_ids:
                    ex = 2 + e_id * 1.8
                    color = C['secondary'] if title == 'Hash' else C['primary']
                    ls = '--' if title == 'Hash' else '-'
                    ax.annotate('', xy=(ex, 4.2), xytext=(tx, 6.85),
                               arrowprops=dict(arrowstyle='->', color=color, lw=1, ls=ls))

        # Description
        ax.text(5, 2.5, desc, ha='center', fontsize=8, color=C['secondary'])

        # Properties box
        if title == 'Top-1':
            props = ['$k_{eff}=1$', 'fastest', 'high drop']
        elif title == 'Top-$k$ Hard':
            props = ['$k_{eff}=k$', 'no gates', 'moderate']
        elif title == 'Soft Top-$k$':
            props = ['$k_{eff}=k$', 'learned $w$', 'best PPL']
        elif title == 'Hash':
            props = ['$k_{eff}=k$', '$w=1/k$', 'CV=0']
        else:
            props = ['$k_{eff}=k$', 'E→T first', 'balanced']

        for i, p in enumerate(props):
            ax.text(5, 1.6 - i * 0.5, p, ha='center', fontsize=8, color=C['secondary'])

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Saved: {out_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality MoE diagrams")
    parser.add_argument("--out-dir", type=str, default="figures/",
                        help="Output directory for diagrams")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Generating publication-quality diagrams...")
    print("-" * 60)

    # Figure 1: MoE Architecture
    plot_moe_architecture_academic(os.path.join(args.out_dir, "moe_architecture.png"))

    # Figure 2: Token Dropping
    plot_token_dropping_academic(os.path.join(args.out_dir, "token_dropping_mechanism.png"))

    # Figure 3: Routing Strategies
    plot_routing_strategies_academic(os.path.join(args.out_dir, "routing_strategies_comparison.png"))

    print("-" * 60)
    print("Done! Publication-quality diagrams generated.")


if __name__ == "__main__":
    main()
