# MoE Routing Bench – Project Checklist

- [x] Lock scope/storyline in English (COMP 414/514 Topic 2): small-scale MoE routing trade-offs (top1/topk-hard/softk) + capacity factor, with discussion linking to routed PEFT/PERFT and vLLM MoE+LoRA support; capture a one-liner in `docs/report.md` for report intro.
- [x] Sanity setup on GPU box: `pip install -e .`, run `pytest -q`, and smoke `python scripts/train_small.py --max-steps 50 --eval-interval 25 --device cuda`.
- [x] Prepare TinyStories data (train/val) via `scripts/export_tinystories.py` into `data/`.
- [x] Run Experiment A training runs on H100 (at least 4 configs): top1 CF=1.0/1.25; softk (k=2) CF=1.0/1.25; optionally add topk-hard; keep model hyperparameters consistent; log under `runs/`. Command: `GPU_IDS=0,1,2,3 bash scripts/run_experiment_a.sh`
- [x] Summarize Experiment A: use `summarize_runs.py` to CSV, merge strategies, plot throughput vs best PPL frontier (`plot_frontier.py`); render at least one training curve (`plot_training_curve.py`), optionally routing stats (drop_rate/load_cv/gate_entropy vs step). Command: `bash scripts/summarize_plot_experiment_a.sh`
- [ ] Run Experiment B microbench: `bench_capacity.py` CF sweep (1.0,1.10,1.25,1.50) on chosen (E,K); plot CF vs avg_drop_rate (and optionally throughput) with `plot_capacity.py`; optionally add CF vs PPL/tokens/s from training logs.
- [ ] Capture figures/artifacts under `results/` with clear filenames (frontier, training_curve, capacity_drop, optional CF_vs_PPL).
- [ ] Analyze findings: trade-offs across routing strategies, CF impact on drop_rate/load balance/throughput/PPL; draft 5–10 key takeaways in `docs/report.md` for later use in report/slides.
- [ ] Draft report (≥6 pages, English, ICML/NeurIPS template acceptable): Introduction; Related Work (Switch/Soft-MoE/Chain-of-Experts/PERFT); Methods (TinyMoEModel, MoEFeedForward, metrics); Experiments A/B; Discussion toward routed PEFT and vLLM MoE+LoRA; Conclusion/Future work.
- [ ] Prepare slides (8–12 pages): motivation; related-work map; MoE Routing Bench overview; Experiment A frontier + training curve; Experiment B capacity plot; routed PEFT/vLLM MoE+LoRA concept slide; takeaways.
- [ ] Update README and `docs/report.md` with reproduction commands for chosen runs/plots and brief storyline/results summary.
