# moe-routing-bench

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

(optional) pip3 install torch --index-url https://download.pytorch.org/whl/cu128
```

## Usage

### Install (editable)
```bash
pip install -e .
```

### Quick benches
```bash
python -m moe_routing_bench.cli bench-topk --impl torch --num-tokens 16384 --num-experts 128 --top-k 2 --hidden-dim 4096 --iters 200
python -m moe_routing_bench.cli bench-routing --impl-topk torch --num-tokens 16384 --num-experts 128 --top-k 2 --hidden-dim 4096 --iters 100
python scripts/bench_pack.py --num-tokens 16384 --hidden-dim 4096 --num-experts 128 --top-k 2 --iters 100
```

### Tests
```bash
pytest -q
```

### Additional benchmarks
```bash
python scripts/bench_pack.py --num-tokens 16384 --hidden-dim 4096 --num-experts 128 --top-k 2 --iters 100
python scripts/bench_pack_only.py --num-tokens 16384 --hidden-dim 4096 --num-experts 128 --top-k 2 --iters 100
python scripts/bench_combine_only.py --num-tokens 16384 --hidden-dim 4096 --num-experts 128 --top-k 2 --iters 100
python scripts/bench_moe_layer.py --num-tokens 8192 --hidden-dim 1024 --num-experts 32 --top-k 2 --expand 4
python scripts/sweep_pack.py --tokens 8192,16384 --hidden 2048,4096 --experts 32,64 --topk 1,2
python scripts/sweep_capacity_ke.py --experts 32,64 --topk 1,2,4 --capacity-factors 1.0,1.25 --strategy softk --backend torch_soft --output results/capacity_ke.jsonl
python scripts/bench_capacity.py --num-tokens 16384 --hidden-dim 4096 --num-experts 128 --top-k 2 --capacity-factors 1.00,1.10,1.25,1.50 --iters 50
```

### Metric conventions
- `bw_GiBps_strict`: uses `(2 + 2·top_k)·hidden_dim·dtype_bytes` per token (pack + combine read/write).
- `bytes_per_token_strict`: strict byte count per token; reported alongside bandwidth.
- `load_cv`: coefficient of variation of per-expert token counts.
- `avg_drop_rate`: fraction of assignments dropped when enforcing capacity.
- All benchmarks fix `seed=17`; adjust as needed.

### Training
```bash
python scripts/train_small.py --max-steps 200 --eval-interval 50 --device cuda
```

```bash
# Export TinyStories splits to text (requires `pip install datasets`)
python scripts/export_tinystories.py --split train --out data/tinystories_train.txt
python scripts/export_tinystories.py --split validation --out data/tinystories_val.txt
```

### Distributed & Routing sweeps
```bash
# 4×GPU DDP run (bf16, batch 128 / GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/train_small.py --distributed --device cuda \
  --data data/tinystories_train.txt --seq-len 512 --batch-size 128 \
  --max-steps 2000 --eval-interval 200 --lr 1.2e-3 --warmup-steps 50 \
  --outdir runs/tinystories_ddp_bs_128

# Strategy × capacity sweep (pure PyTorch)
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/sweep_routing_train.py \
  --data data/tinystories_train.txt --seq-len 512 --batch-per-gpu 128 \
  --num-experts-list 64,128 --router-list top1,topk-hard,softk \
  --topk-list 1,2,4 --capacity-factors 1.0,1.25,1.5 \
  --seeds 17,18,19 --max-steps 1000 --eval-interval 200 \
  --outdir-root runs/routing_sweep

# Summarise & plot Pareto frontier
python scripts/summarize_runs.py --runs 'runs/routing_sweep/*' \
  --out results/routing_sweep_summary.csv
python scripts/plot_frontier.py \
  --summary results/routing_sweep_summary.csv \
  --filter-E 128 --filter-dim 256 \
  --out results/routing_frontier.png

# Plot step-wise metrics for a single run
python scripts/plot_training_curve.py \
  --log runs/tinystories_ddp_bs_128/train_log.jsonl \
  --metrics train_loss,val_loss,ppl \
  --out results/tinystories_ddp_bs_128_curve.png
```

Each training run writes `train_log.jsonl` (tokens/s, drop-rate, load_cv, gate entropy, FLOPs, bandwidth, etc.) and `hparams.json` for reproducibility and visualisation.
