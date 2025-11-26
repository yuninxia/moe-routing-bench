# Slides Outline

## Page 1 – Title
- **Title:** MoE Routing Bench: Routing Trade-offs & PERFT Insights
- **Subtitle:** Capacity factor, routing frontiers, and routed adapters
- **Authors:** Yuning Xia (yx87), Daniel Zhang (dfz1), Shaoyang Zhang (sz121), Richard Xu (rgx1)
- **Course:** COMP 414/514 – Topic 2: Mixture-of-Experts and Learning to Specialize

## Page 2 – Motivation
- Routing is the bottleneck for MoE quality/stability/throughput.
- Knobs: routing strategy (top-1/hard/soft/EC/hash), capacity factor (CF), load balance.
- Gap: many large-scale results, few small, reproducible benches + PERFT linkage.
- What we did: build a lightweight bench, map Pareto frontiers (routing/CF/scale/tokenizer), bridge to PERFT-style routed adapters.

## Page 3 – Routing Design Space
- 5 routers: top-1, top-k hard, softk, expert-choice, hash (content-agnostic baseline).
- Metrics: PPL, tokens/s, drop_rate, load_cv, gate_entropy.
- Axes: hard↔soft, content-aware↔hash, CF controls dropping/load.

## Page 4 – Bench Setup
- Model/data: TinyMoE (E=8, dim=256, L=4), TinyStories (char + BPE).
- Scripts: `train_small.py`, `bench_capacity.py`, `summarize_runs.py`, `plot_frontier.py`.
- Metrics per eval: PPL, tokens/s, drop_rate, load_cv, gate_entropy, eff_TFLOPs.
- Hardware: 4× L40S; runs 500–1200 steps; all commands reproducible in repo.

## Page 5 – Capacity Factor Sweep
- Figure: `../results/capacity_drop_rate_multi.png` (+ inset `capacity_tokens_per_s_multi.png`).
- Takeaways:
  - CF 1.05–1.10 ≈ zero drops; throughput change <2%.
  - Practical default: CF ≈ 1.1–1.25.
  - CF tunes dropping/load, not content awareness.

## Page 6 – Unified Router Frontier (E=8)
- Figure: `../results/unified_frontier.png` (PPL vs tokens/s).
- Takeaways:
  - Quality: expert_choice ≈ softk < hash < topk-hard < top1.
  - Speed: top1 fastest; softk/EC ~20–25% slower.
  - Hash: perfect balance, worst PPL → balance alone is insufficient.

## Page 7 – Larger Scale & Subword
- Figures: `../results/larger_scale_frontier.png` (E=32), `../results/subword_frontier.png` (BPE).
- Takeaways:
  - Ranking persists at larger scale and with BPE.
  - Expert-choice/softk remain best-quality; trends not a toy artifact.

## Page 8 – PERFT Frontier (Adapter Routing)
- Figure: `../results/perft_variants/perft_frontier_loss_vs_eff.png` (1/PPL vs activated param efficiency).
- Config: ranks 8/16/32; TopK/N ∈ {(1,4), (2,4), (1,8), (2,8)}; 500 steps.
- Takeaways:
  - PERFT-R dominates; PERFT-E > Shared at same efficiency.
  - Sweet spot: sparse Top1/8 or Top1/4, rank 8–16; higher rank → diminishing returns.
  - Routed adapters deliver more performance per activated parameter.

## Page 9 – Implications (Systems / PERFT / Serving)
- MoE systems: use softk/EC for quality; CF≈1.1–1.25 to remove drops; hash ≠ quality fix.
- PERFT/routed PEFT: prefer routed (R/E) over shared; balanced routing keeps adapters trained.
- Serving (vLLM/Mixtral-style): top-2/soft routing + tuned CF for quality/latency balance.

## Page 10 – Conclusion & Next Steps
- Routing ranking stable across CF, scale (E=8→32), tokenizer (char→BPE); CF sweet spot identified.
- PERFT frontier matches paper trend (R > E > Shared) even in short runs.
- Next: parallel K=2 vs sequential CoE; deeper BPE runs; plug into serving stack; explore aux-loss-free balancing.
