# moe-routing-bench

A benchmark suite for comparing Mixture-of-Experts (MoE) routing strategies in small-scale transformer models. Evaluates routing algorithms across metrics: perplexity, throughput, token drop rate, load balance (CV), and gate entropy.

## Proposal
[Project Proposal](https://docs.google.com/document/d/1s6bXTtXZuCin6RbOFYJIVvWXMNcvPrVMaxMTPWTwwnU/edit?usp=sharing)
[Report (Markdown)](docs/report.md)
[Slides](docs/slides.pdf)

## Implemented Routing Strategies

| Strategy | Description | Key Characteristics |
|----------|-------------|---------------------|
| **top1** | Single highest-logit expert per token | Fastest, highest drop rate |
| **topk-hard** | Top-k experts by logit, uniform weights | Balanced speed/quality |
| **softk** | Top-k with learned softmax weights | Best quality, differentiable |
| **hash** | Deterministic hash-based assignment | Perfect load balance, no learning |
| **expert-choice** | Experts select top tokens | Best PPL + balance combined |

## Env Setup

```bash
# For L40S
module purge
module load GCCcore/12.3.0
module load Python/3.11.3

python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Hardware / Data / Models
- Hardware: 4× NVIDIA L40S (BF16), DDP; typical runs 500–2000 steps.
- Data: TinyStories (char) and BPE tokenizer (`EleutherAI/gpt-neo-125M`); PERFT sweep uses `commonsense_170k` QA.
- Models/configs: TinyMoE E=8 (dim=256, L=4, K=2; top1 uses K=1); larger scale E=32 (dim=512, L=4, K=2, CF=1.5).

## Quick Start

```bash
# Setup (HPC users adjust module load if needed)
module purge && module load GCCcore/12.3.0 Python/3.11.3  # HPC only
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Prepare TinyStories data
python scripts/export_tinystories.py --split train --out data/tinystories_train.txt
python scripts/export_tinystories.py --split validation --out data/tinystories_val.txt

# Unified router sweep (E=8, char-level)
GPU_IDS=0,1,2,3 MAX_STEPS=1200 EVAL_INTERVAL=200 bash scripts/run_experiment_unified.sh
bash scripts/summarize_plot_unified.sh
```

**Unified outputs**:
- `results/unified_summary.csv`
- `results/unified_frontier.png`
- `results/unified_overlay.png`
- (optional) `results/unified_load_vs_ppl.png` via `python scripts/plot_unified_load_ppl.py --summary results/unified_summary.csv --out-dir results`

## Reproducing All Experiments

### Experiment 1: Unified Router Sweep (E=8, char-level)

Compares all 5 routing strategies with capacity factors used in our runs (CF=[1.0, 1.25]) on a small model.

**Config**: E=8, dim=256, layers=4, K=2, CF=[1.0, 1.25]

```bash
# Run experiment (~1.5 hours on 4×L40S)
GPU_IDS=0,1,2,3 MAX_STEPS=1200 EVAL_INTERVAL=200 bash scripts/run_experiment_unified.sh

# Generate summary CSV and plots
bash scripts/summarize_plot_unified.sh
```

**Outputs**:
- `results/unified_summary.csv` - Metrics for all runs
- `results/unified_frontier.png` - PPL vs throughput Pareto frontier
- `results/unified_overlay.png` - Training curves overlay
- `results/gate_balance_unified.png` - load_cv vs gate_entropy scatter (optional: run `python scripts/plot_gate_balance.py --summary results/unified_summary.csv --out results/gate_balance_unified.png`)

### Experiment 2: Capacity Factor Microbenchmark

Isolates the effect of capacity factor on drop rate and throughput.

**Config**: E=[64, 128, 256], K=[1, 2, 4], CF=[0.9, 1.0, 1.05, 1.1, 1.25, 1.5]

```bash
# Run experiment and plot (~30 min)
bash scripts/run_and_plot_experiment_b.sh
```

**Outputs**:
- `results/capacity_cf_multi.jsonl` - Raw benchmark data
- `results/capacity_drop_rate_multi.png` - Drop rate vs CF
- `results/capacity_tokens_per_s_multi.png` - Throughput vs CF

### Experiment 3: Larger Scale Validation (E=32)

Validates that routing strategy rankings hold at larger scale.

**Config**: E=32, dim=512, layers=4, K=2, CF=1.5, lr=1e-4, warmup=200

```bash
# Run experiment (~2.5 hours on 4×L40S)
GPU_IDS=0,1,2,3 bash scripts/run_experiment_larger_scale.sh

# Generate summary and plots
bash scripts/summarize_plot_larger_scale.sh
```

**Outputs**:
- `results/larger_scale_summary.csv`
- `results/larger_scale_frontier.png`
- `results/larger_scale_overlay.png`

### Experiment 4: Subword Tokenization (BPE)

Validates that routing trade-offs hold with BPE tokenization.

**Config**: E=8, dim=256, BPE tokenizer (EleutherAI/gpt-neo-125M), CF=1.25

```bash
# Run experiment (~50 min on 4×L40S)
GPU_IDS=0,1,2,3 bash scripts/run_experiment_subword.sh

# Generate summary and plots
bash scripts/summarize_plot_subword.sh
```

**Outputs**:
- `results/subword_summary.csv`
- `results/subword_frontier.png`
- `results/subword_overlay.png`

### Quick Verification (Single GPU)

For quick testing without full experiment runs:

```bash
# Smoke test (~2 min)
python scripts/train_small.py --max-steps 200 --eval-interval 50 --device cuda

# Test specific strategy
python scripts/train_small.py \
  --data data/tinystories_train.txt \
  --router expert-choice \
  --num-experts 8 \
  --capacity-factor 1.25 \
  --max-steps 500 \
  --device cuda
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_IDS` | `0,1,2,3` | Comma-separated GPU IDs |
| `MAX_STEPS` | varies | Training steps |
| `EVAL_INTERVAL` | varies | Evaluation frequency |
| `STRATEGIES` | all | Space-separated strategies to run |

## PERFT Variants (small-scale frontier)

Reproduce PERFT-R/E vs Shared adapter (500-step) frontier on TinyStories.

```bash
# Download OLMoE checkpoint (one-time)
HF_HOME=/scratch/yx87/playground/moe-routing-bench/.cache/hf \
CACHE_DIR=$HF_HOME \
bash scripts/download_olmoe_checkpoint.sh

# Run lightweight PERFT sweep (R/E/Shared; ranks 8/16/32; TopK/N=1/2 over N=4/8)
HF_HOME=/scratch/yx87/playground/moe-routing-bench/.cache/hf \
GPU_IDS=0,1,2,3 \
RANK_LIST="8 16 32" \
PEFT_EXPERTS_LIST="4 8" \
TOPK_LIST="1 2" \
bash scripts/run_perft_variants_quick.sh

# Plot Fig.4-style frontier (performance=1/PPL vs efficiency; size=rank; alpha=TopK/N)
python scripts/plot_perft_variants.py
```

Outputs:
- `results/perft_variants/variants_summary.csv`
- `results/perft_variants/perft_frontier_loss_vs_eff.png`

### Expected Results Summary

| Experiment | Best Strategy | PPL | Throughput |
|------------|---------------|-----|------------|
| Unified (E=8) | expert-choice | 5.50 | 2.33M tok/s |
| Larger Scale (E=32) | expert-choice | 3.86 | 7.94M tok/s |
| Subword (BPE) | expert-choice | 31.88 | 2.21M tok/s |

**Key Finding**: Strategy ranking is consistent across scales and tokenization: `expert-choice > softk > hash > topk-hard > top1`
