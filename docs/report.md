# MoE Routing Bench – Report Notes

## Scope (COMP 414/514 Topic 2)

We study small-scale MoE routing trade-offs (top1/topk-hard/softk) and capacity factor effects, and discuss implications for routed PEFT (e.g., PERFT) plus recent MoE+LoRA support in vLLM.

### Alignment to Topic 2
- Implement and evaluate MoE routing variants within a Transformer (TinyMoEModel) with metrics on perplexity, throughput, drop rate, load balance, and capacity usage.
- Analyze routing/CF trade-offs and relate findings to dynamic/parameter-efficient routing trends (Switch/Soft-MoE/Chain-of-Experts/PERFT).
- Connect bench results to emerging system support for routed adapters (vLLM MoE+LoRA), outlining future integration paths.

## Course Requirements Snapshot
- Deliverables: project proposal; in-class spotlight talk (5–10 minutes); final report ≥6 pages (excluding references) using a standard conference template (ICML/NeurIPS acceptable).
- Topic 2 framing: build/review MoE routing, implement small MoE layer on a Transformer, experiment with routing strategies, and discuss parameter-efficient/dynamic routing trends.
- Project mode: blend of literature review and implementation with critical discussion (not just aggregation of papers).
- Core papers to anchor review: Sparsely-Gated MoE (Shazeer et al.), Learning to Specialize (joint gating–expert training), Chain-of-Experts, PERFT: Parameter-Efficient Routed Fine-Tuning, From Sparse to Soft Mixtures of Experts.

## Experiments & Repro commands
- Experiment A (top1/topk-hard/softk × CF 1.0/1.25): `GPU_IDS=0,1,2,3 bash scripts/run_experiment_a.sh` then `bash scripts/summarize_plot_experiment_a.sh` → `results/experiment_a_summary.csv`, `experiment_a_frontier.png`, `experiment_a_overlay.png`.
- Experiment B (capacity sweep, E∈{64,128,256}, K∈{1,2,4}, CF=0.9..1.5): `bash scripts/run_and_plot_experiment_b.sh` → `results/capacity_cf_multi.jsonl`, `capacity_drop_rate_multi.png`, `capacity_tokens_per_s_multi.png`.
- Experiment C (top1/topk-hard/softk/hash/expert_choice, CF=1.25): `GPU_IDS=0,1,2,3 MAX_STEPS=1200 EVAL_INTERVAL=200 bash scripts/run_experiment_c.sh` then `bash scripts/summarize_plot_experiment_c.sh` → `results/expC_summary.csv`, `expC_frontier.png`, `expC_frontier_tflops.png`, `expC_overlay.png`.

## Key findings (A/B/C)
- Routing Pareto (A): softk achieves the best PPL at the cost of ~20–25% lower tokens/s vs top1; topk-hard sits between. CF=1.25 improves PPL over CF=1.0 with negligible throughput loss.
- Capacity vs drop (B): across E=64/128/256 and K=1/2/4, bumping CF from 1.0→1.05–1.10 collapses avg_drop_rate from ~1–10% to ~0 while tokens/s stays flat; larger CF shows diminishing returns.
- Hash vs learned (C): hash routing has near-zero drop/load_cv and good throughput but worse PPL than learned routers—demonstrates “balance ≠ quality”; expert_choice matches learned token-choice PPL within small deltas while keeping balanced load and low drop.
- Throughput ordering (C): K dominates speed (K=1 fastest, K=2/softk slower, expert_choice similar to softk); expert count has minor effect in microbench, aligning with pack/combine cost model.
- Load/stability: softk and expert_choice exhibit lower load_cv and non-zero gate entropy; top1/topk-hard gate_entropy=0 with higher drop/load_cv; hash has perfect balance but uniform gates.

## Figures (results/)
- experiment_a_frontier.png / experiment_a_frontier_tflops.png; experiment_a_overlay.png.
- capacity_drop_rate_multi.png; capacity_tokens_per_s_multi.png.
- expC_frontier.png / expC_frontier_tflops.png; expC_overlay.png.

### Figure previews
![Experiment A Frontier](../results/experiment_a_frontier.png)
![Experiment A Overlay](../results/experiment_a_overlay.png)
![Capacity Sweep Drop Rate](../results/capacity_drop_rate_multi.png)
![Capacity Sweep Tokens/s](../results/capacity_tokens_per_s_multi.png)
![Experiment C Frontier](../results/expC_frontier.png)
![Experiment C Overlay](../results/expC_overlay.png)
