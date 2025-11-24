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
