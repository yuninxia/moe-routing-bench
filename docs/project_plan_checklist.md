# MoE Routing Bench – Project Checklist

## Phase 1: Core Implementation
- [x] Scope/storyline locked (Topic 2): small-scale MoE routing trade-offs (top1/topk-hard/softk) + capacity factor; link to routed PEFT/PERFT and vLLM MoE+LoRA in report intro.
- [x] Environment sanity: `pip install -e .`, `pytest -q`, smoke `python scripts/train_small.py --max-steps 50 --eval-interval 25 --device cuda`.
- [x] Data prep: TinyStories via `scripts/export_tinystories.py` into `data/`.
- [x] Unified router sweep (top1/topk-hard/softk/hash/expert_choice × CF): `GPU_IDS=0,1,2,3 MAX_STEPS=1200 EVAL_INTERVAL=200 bash scripts/run_experiment_unified.sh` → `bash scripts/summarize_plot_unified.sh`.
- [x] Capacity microbench (CF/E/K grid): `bash scripts/run_and_plot_experiment_b.sh` (CF 0.9–1.5; E=64/128/256; K=1/2/4; iters=200).
- [x] New routers implemented (`hash`, `expert_choice`) and unit-tested.
- [x] Figures/artifacts captured: unified frontier/overlay/load_vs_ppl; capacity drop/tokens.
- [x] Report drafted with findings and literature review (core papers + DeepSeek-V3, Mixtral, etc.).

## Phase 2: Final improvements
- [x] PERFT/vLLM connection: architecture diagram, pseudocode, vLLM MoE+LoRA example.
- [x] Subword tokenization support and experiment: `run_experiment_subword.sh` + `summarize_plot_subword.sh`; results in report.
- [x] Larger scale validation: E=32, dim=512, layers=4; strategies top1/topk-hard/softk/hash/expert_choice; CF=1.5; lr=1e-4; warmup=200.
- [x] Report polish: softened claims to “in tested scales/configs”; added gate balance scatter (load_cv vs gate_entropy).
- [x] PERFT small-scale frontier (500-step sweep; R/E/Shared; ranks 8/16/32; TopK/N=1/2 over N=4/8): `run_perft_variants_quick.sh` + `plot_perft_variants.py` → `results/perft_variants/perft_frontier_loss_vs_eff.png`.
- [x] Slides outline drafted (`docs/slides.md`, ~10 pages: motivation → bench → capacity → unified → larger/subword → PERFT frontier → implications → conclusion).
- [x] Router architecture add-on (low effort): sweep `router_arch ∈ {linear, mlp, mlp_hadamard}` with `strategy ∈ {softk, expert-choice}`; log PPL/tokens_per_s/router_params/mean_topk_prob and plot frontier (PPL vs tokens/s, marker shape = router arch) plus a small table (router_params, entropy, mean_topk_prob).
- [ ] Slide deck (8–12 pages) to finalize in Google Slides/LaTeX.
- [ ] LaTeX template conversion (optional): convert `docs/report.md` to ICML/NeurIPS template.
- [ ] README final check: ensure reproduction commands/results summary reflect current scripts.

## Completed experiments
| Experiment | Status | Key result |
|------------|--------|------------|
| Capacity sweep (microbench) | Done | CF 1.05–1.10 eliminates token dropping |
| Unified router sweep (E=8) | Done | expert_choice PPL=5.50, softk PPL=5.55 |
| Larger scale (E=32, dim=512) | Done | expert_choice PPL=3.86, 7.94M tok/s |
| Subword (BPE) | Done | Ranking: expert_choice > softk > top1 |
| Router arch (E=8) | Done | SoftK/EC best PPL across arches; Top-1 fastest, worst PPL |
| PERFT frontier | Done | PERFT-R dominates efficiency–performance; PERFT-E > shared |

## Resources
- Hardware: 4× NVIDIA L40S GPUs.
- Completed training times: subword ~50 min; larger scale ~2.5 hours; unified sweep ~1.5 hours; PERFT sweep ~4 hours.

## Success criteria
- Report covers core papers with detailed discussion.
- Additional experiments beyond character-level (subword and larger scale).
- PERFT/vLLM connection includes concrete examples.
- Slides ready for ~10 minute presentation (pending).
- Claims backed by experiments or citations.

## Deliverables
| Deliverable | Status | Location |
|-------------|--------|----------|
| Report (Markdown) | Complete | `docs/report.md` |
| Unified figures | Complete | `results/unified_*.png` |
| Capacity figures | Complete | `results/capacity_*.png` |
| Larger scale figures | Complete | `results/larger_scale_*.png` |
| Subword figures | Complete | `results/subword_*.png` |
| Router arch figures | Complete | `results/router_arch_frontier.png` |
| PERFT figures | Complete | `results/perft_variants/*.png` |
| Summary CSVs | Complete | `results/*_summary.csv` |
| Presentation slides | Pending | (to be added) |

## Figures for Report/Slides

### Conceptual Diagrams (Drawing)
- [x] **1.1 MoE Architecture Diagram** — `figures/moe_architecture.png` (Figure 1 in report, Page 4 in slides)
- [x] **1.2 Token Dropping Illustration** — `figures/token_dropping_mechanism.png` (Figure 3 in report, Page 6 in slides)
- [x] **1.3 Routing Strategy Comparison** — `figures/routing_strategies_comparison.png` (Figure 2 in report, Page 5 in slides)

### Data-Driven Figures (Code)
- [x] **2.1 Expert Load Distribution Bar** — `results/expert_load_distribution.png` (Figure 11 in report, Page 16 in slides)
- [x] **2.2 Token-Expert Heatmap** — `results/routing_heatmap.png`, `results/routing_heatmap_summary.png` (Figures 12-13 in report, Pages 17-18 in slides)

### Literature Context Figures
- [x] **3.1 MoE Evolution Timeline** — `figures/moe_evolution_timeline.png` (Figure 4 in report, Page 3 in slides)

### Advanced Visualizations
- [x] **4.1 Colored Token Visualization** — `results/token_routing_colored.png`, `results/token_routing_colored_by_char_type.png` (Figures 15-16 in report, Pages 20-21 in slides)
- [x] **4.2 PERFT Architecture Diagram** — `figures/perft_architecture.png` (Figure 17 in report, Page 13 in slides)

### Supplementary Figures
- [x] **5.1 Pareto Frontier Annotated** — `results/unified_frontier_annotated.png` (Figure 14 in report, Page 19 in slides)

### Figure Summary Table

| ID  | Figure                         | Type    | Output Location                          | Status |
|-----|--------------------------------|---------|------------------------------------------|--------|
| 1.1 | MoE Architecture Diagram       | Drawing | `figures/moe_architecture.png`           | [x]    |
| 1.2 | Token Dropping Illustration    | Drawing | `figures/token_dropping_mechanism.png`   | [x]    |
| 1.3 | Routing Strategy Comparison    | Drawing | `figures/routing_strategies_comparison.png` | [x] |
| 2.1 | Expert Load Distribution       | Code    | `results/expert_load_distribution.png`   | [x]    |
| 2.2 | Token-Expert Heatmap           | Code    | `results/routing_heatmap*.png`           | [x]    |
| 3.1 | MoE Evolution Timeline         | Drawing | `figures/moe_evolution_timeline.png`     | [x]    |
| 4.1 | Colored Token Visualization    | Code    | `results/token_routing_colored*.png`     | [x]    |
| 4.2 | PERFT Architecture Diagram     | Drawing | `figures/perft_architecture.png`         | [x]    |
| 5.1 | Pareto Frontier Annotated      | Code    | `results/unified_frontier_annotated.png` | [x]    |
