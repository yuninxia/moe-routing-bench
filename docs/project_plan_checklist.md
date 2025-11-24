# MoE Routing Bench – Project Checklist

- [x] Lock scope/storyline in English (COMP 414/514 Topic 2): small-scale MoE routing trade-offs (top1/topk-hard/softk) + capacity factor, with discussion linking to routed PEFT/PERFT and vLLM MoE+LoRA support; capture a one-liner in `docs/report.md` for report intro.
- [x] Sanity setup on GPU box: `pip install -e .`, run `pytest -q`, and smoke `python scripts/train_small.py --max-steps 50 --eval-interval 25 --device cuda`.
- [x] Prepare TinyStories data (train/val) via `scripts/export_tinystories.py` into `data/`.
- [x] Run Experiment A training runs on H100 (at least 4 configs): top1 CF=1.0/1.25; softk (k=2) CF=1.0/1.25; optionally add topk-hard; keep model hyperparameters consistent; log under `runs/`. Command: `GPU_IDS=0,1,2,3 bash scripts/run_experiment_a.sh`
- [x] Summarize Experiment A: use `summarize_runs.py` to CSV, merge strategies, plot throughput vs best PPL frontier (`plot_frontier.py`); render at least one training curve (`plot_training_curve.py`), optionally routing stats (drop_rate/load_cv/gate_entropy vs step). Command: `bash scripts/summarize_plot_experiment_a.sh`
- [x] Run Experiment B microbench: `bench_capacity.py` CF sweep (1.0,1.10,1.25,1.50) on chosen (E,K); plot CF vs avg_drop_rate (and optionally throughput) with `plot_capacity.py`; optionally add CF vs PPL/tokens/s from training logs. Commands: `bash scripts/run_experiment_b.sh` then `bash scripts/summarize_plot_experiment_b.sh`
- [x] Implement new routers (per `docs/new_routers.md`):
  - [x] Add `hash` strategy (content-agnostic baseline) in `MoEFeedForward._route` and expose in `train_small.py`.
  - [x] Add `expert_choice` strategy (2202.09368-inspired) in `MoEFeedForward._route` with simplified expert-first selection; expose in `train_small.py`.
  - [x] Add unit tests similar to `tests/test_softk_routing.py` to validate `hash` and `expert_choice` pack/combine correctness.
- [x] Run Experiment C (new routers): compare `top1/topk-hard/softk/hash/expert_choice` at fixed hyperparams; plot throughput vs PPL frontier and overlay curves. Commands: `bash scripts/run_experiment_c.sh` then `bash scripts/summarize_plot_experiment_c.sh`.
- [x] Capture figures/artifacts under `results/` with clear filenames (frontier, training_curve, capacity_drop, optional CF_vs_PPL). Current: `results/experiment_a_frontier.png`, `results/experiment_a_overlay.png`, `results/capacity_drop_rate_multi.png`, `results/capacity_tokens_per_s_multi.png`, `results/expC_frontier.png`, `results/expC_frontier_tflops.png`.
- [ ] Analyze findings: trade-offs across routing strategies, CF impact on drop_rate/load balance/throughput/PPL; draft 5–10 key takeaways in `docs/report.md` for later use in report/slides.
- [ ] Draft report (≥6 pages, English, ICML/NeurIPS template acceptable): Introduction; Related Work (Switch/Soft-MoE/Chain-of-Experts/PERFT); Methods (TinyMoEModel, MoEFeedForward, metrics); Experiments A/B; Discussion toward routed PEFT and vLLM MoE+LoRA; Conclusion/Future work.
- [ ] Prepare slides (8–12 pages): motivation; related-work map; MoE Routing Bench overview; Experiment A frontier + training curve; Experiment B capacity plot; routed PEFT/vLLM MoE+LoRA concept slide; takeaways.
- [ ] Update README and `docs/report.md` with reproduction commands for chosen runs/plots and brief storyline/results summary.
