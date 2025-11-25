# MoE Routing Bench – Project Checklist

## Phase 1: Core Implementation (Completed)

- [x] Lock scope/storyline in English (COMP 414/514 Topic 2): small-scale MoE routing trade-offs (top1/topk-hard/softk) + capacity factor, with discussion linking to routed PEFT/PERFT and vLLM MoE+LoRA support; capture a one-liner in `docs/report.md` for report intro.
- [x] Sanity setup on GPU box: `pip install -e .`, run `pytest -q`, and smoke `python scripts/train_small.py --max-steps 50 --eval-interval 25 --device cuda`.
- [x] Prepare TinyStories data (train/val) via `scripts/export_tinystories.py` into `data/`.
- [x] Unified router sweep (top1/topk-hard/softk/hash/expert_choice × CF): `GPU_IDS=0,1,2,3 MAX_STEPS=1200 EVAL_INTERVAL=200 bash scripts/run_experiment_unified.sh` → `bash scripts/summarize_plot_unified.sh`
- [x] Capacity microbench (rich CF/E/K grid): `bash scripts/run_and_plot_experiment_b.sh` (defaults: CF 0.9–1.5; E=64/128/256; K=1/2/4; iters=200)
- [x] Implement new routers (per `docs/new_routers.md`):
  - [x] Add `hash` strategy (content-agnostic baseline) in `MoEFeedForward._route` and expose in `train_small.py`.
  - [x] Add `expert_choice` strategy (2202.09368-inspired) in `MoEFeedForward._route` with simplified expert-first selection; expose in `train_small.py`.
  - [x] Add unit tests similar to `tests/test_softk_routing.py` to validate `hash` and `expert_choice` pack/combine correctness.
- [x] Capture figures/artifacts under `results/` with clear filenames (frontier/overlay, capacity_drop/tokens). Current: `results/unified_frontier.png`, `results/unified_overlay.png`, `results/capacity_drop_rate_multi.png`, `results/capacity_tokens_per_s_multi.png`.
- [x] Analyze findings: trade-offs across routing strategies, CF impact on drop_rate/load balance/throughput/PPL; draft 5–10 key takeaways in `docs/report.md` for later use in report/slides.
- [x] Draft report with comprehensive literature review covering all 5 core papers + modern developments (DeepSeek-V3, Mixtral, etc.)

---

## Phase 2: Final Improvements (3-Day Sprint)

Based on peer feedback (current score estimate: 88-90), the following improvements are prioritized to reach 95+ score.

### 🔴 High Priority (Must Do – Day 1-2)

These items provide the highest score improvement with reasonable effort:

- [x] **PERFT Connection Enhancement** ✅ DONE
  - Added architecture diagram showing how routing strategies connect to PERFT variants (PERFT-R/E/D/S)
  - Added pseudocode example showing routed adapter integration
  - Expanded Section 7.3 with concrete vLLM MoE+LoRA command examples
  - Reference: [PERFT paper](https://arxiv.org/abs/2411.08212)

- [x] **Subword Tokenization Support** ✅ DONE
  - Modified `train_small.py` to support HuggingFace BPE tokenizer via `--tokenizer` flag
  - Created `scripts/run_experiment_subword.sh` for running subword experiments
  - Usage: `--tokenizer EleutherAI/gpt-neo-125M`

- [x] **Run Subword Experiment** ✅ DONE
  - Command: `GPU_IDS=0,1,2,3 bash scripts/run_experiment_subword.sh`
  - Strategies: top1, softk, expert_choice at CF=1.25
  - Results added to report Section 5.6 "Subword Tokenization Validation"
  - **Key finding**: Strategy ranking consistent with char-level (expert_choice > softk > top1)
  - Results: expert_choice PPL=31.88, softk PPL=32.07, top1 PPL=34.56

- [ ] **Prepare Slides** (~3-4 hours) – **MUST COMPLETE BY DAY 3**
  - 8-12 slides covering:
    1. Title + Team
    2. Motivation: Why MoE routing matters
    3. Literature landscape (timeline/tree of routing evolution)
    4. MoE Routing Bench overview (architecture diagram)
    5. Capacity Sweep results (drop_rate + throughput plots)
    6. Unified Router Sweep results (Pareto frontier)
    7. Larger Scale Validation results (NEW!)
    8. Key findings (3-4 bullet points)
    9. Connection to PERFT / vLLM MoE+LoRA
    10. Limitations & Future Work
    11. Conclusion + Q&A

### 🟡 Medium Priority (If Time Permits – Day 2)

These items add credibility but are not essential:

- [x] **Larger Scale Validation** ✅ DONE
  - Ran experiment with E=32, dim=512, layers=4 (larger scale)
  - Strategies: top1, topk-hard, softk, hash, expert_choice at CF=1.5
  - Added as Section 5.5 "Larger Scale Validation" in report
  - Command: `GPU_IDS=0,1,2,3 bash scripts/run_experiment_larger_scale.sh`
  - **Key findings**:
    - Strategy ranking is **scale-invariant**: expert_choice > softk > hash > topk-hard > top1
    - Expert-choice achieves both best PPL (3.86) AND best throughput (7.94M tokens/s)
    - With proper tuning (E=32, CF=1.5, lr=1e-4), drop rates are manageable (15-31%)

- [x] **Discussion Section Enhancement** ✅ DONE (via larger scale results)
  - Conclusion updated with findings 5 & 6 on scale invariance
  - Section 5.5 demonstrates generalization to larger models

### 🔵 Final Polish (For 100% Score) ✅ ALL DONE

These items address remaining gaps for a perfect score:

- [x] **Enhance Chain-of-Experts Discussion** ✅ DONE
  - Added detailed comparison of parallel vs sequential expert execution in Section 7.1
  - Connected our K=2 parallel results to CoE's sequential approach with specific numbers (823× expert combinations, 17-42% memory reduction)
  - Added comparison table with throughput, routing, memory trade-offs
  - Sources: [arXiv:2506.18945](https://arxiv.org/abs/2506.18945), [GitHub](https://github.com/ZihanWang314/CoE)

- [x] **Add Dynamic Routing Trend Analysis** ✅ DONE (New Section 7.4)
  - Added historical trajectory table (2017-2024): Sparsely-Gated → GShard → Switch → Expert Choice → Soft MoE → DeepSeek-V3
  - Detailed auxiliary-loss-free balancing mechanism with bias update rule
  - ASCII diagram positioning our 5 strategies on the static→dynamic spectrum
  - Sources: [HuggingFace MoE Balance Blog](https://huggingface.co/blog/NormalUhr/moe-balance), [arXiv:2408.15664](https://arxiv.org/html/2408.15664v1)

- [x] **Expert Specialization Analysis** ✅ DONE (New Section 7.5)
  - Added theoretical grounding from DDOME (Theorem 1: decoupled training is provably suboptimal)
  - Created specialization-balance trade-off diagram showing negative feedback loop
  - Added table analyzing gate entropy vs load CV for all 5 strategies
  - Connected our experimental observations to Farhat et al.'s predictions
  - Sources: [arXiv:2306.08586](https://arxiv.org/abs/2306.08586)

### 🟢 Low Priority (Can Defer)

- [ ] **LaTeX Template Conversion** (~2-3 hours)
  - Convert `docs/report.md` to ICML/NeurIPS LaTeX template
  - Can be done after presentation if time is tight
  - Template: https://github.com/ICML/icml_latex or NeurIPS 2024 template

- [ ] **Update README** with final reproduction commands and results summary

---

## Completed Experiments Summary

| Experiment | Status | Key Result |
|------------|--------|------------|
| Capacity Sweep (Microbench) | ✅ Done | CF=1.05-1.10 eliminates token dropping |
| Unified Router Sweep (E=8) | ✅ Done | expert_choice PPL=5.50, softk PPL=5.55 |
| Larger Scale (E=32, dim=512) | ✅ Done | expert_choice PPL=3.86, 7.94M tok/s |
| Subword Tokenization (BPE) | ✅ Done | Ranking consistent: expert_choice > softk > top1 |

---

## Hardware Resources

- **Available**: 4× NVIDIA L40S GPUs
- **Completed training times**:
  - Subword experiment (3 strategies): ~50 min total
  - Larger scale experiment (5 strategies): ~2.5 hours total
  - Unified sweep (E=8): ~1.5 hours total

---

## Success Criteria

- [x] Report addresses all 5 core papers with detailed discussion
- [x] At least one additional experiment beyond character-level (subword AND larger scale!)
- [x] PERFT/vLLM connection has concrete examples (not just mentions)
- [ ] Slides ready for 5-10 minute presentation
- [x] All claims backed by experimental evidence or citations

---

## Final Deliverables

| Deliverable | Status | Location |
|-------------|--------|----------|
| Report (Markdown) | ✅ Complete | `docs/report.md` |
| Unified Sweep Figures | ✅ Complete | `results/unified_*.png` |
| Capacity Sweep Figures | ✅ Complete | `results/capacity_*.png` |
| Larger Scale Figures | ✅ Complete | `results/larger_scale_*.png` |
| Subword Figures | ✅ Complete | `results/subword_*.png` |
| Summary CSVs | ✅ Complete | `results/*_summary.csv` |
| Presentation Slides | ⏳ Pending | TBD |
