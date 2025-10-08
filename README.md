# moe-routing-bench

## Env Setup

```bash
pip3 install nvidia-cutlass-dsl
pip3 install torch --index-url https://download.pytorch.org/whl/cu128
```

## Reference
- https://veitner.bearblog.dev/bridging-math-and-code-cute-layout-algebra-in-cutedsl/
- https://veitner.bearblog.dev/an-applied-introduction-to-cutedsl/

## Usage

### Install (editable)
```bash
pip install -e .
```

### Quick benches
```bash
python -m moe_routing_bench.cli bench-topk --impl torch --num-tokens 16384 --num-experts 128 --top-k 2 --hidden-dim 4096 --iters 200
python -m moe_routing_bench.cli bench-topk --impl quack --num-tokens 16384 --num-experts 128 --top-k 2 --hidden-dim 4096 --iters 200
python -m moe_routing_bench.cli bench-routing --impl-topk torch --num-tokens 16384 --num-experts 128 --top-k 2 --hidden-dim 4096 --iters 100
python -m moe_routing_bench.cli bench-routing --impl-topk quack --num-tokens 16384 --num-experts 128 --top-k 2 --hidden-dim 4096 --iters 100
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
