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
python -m moe_routing_bench.cli bench-topk --impl quack
python -m moe_routing_bench.cli bench-routing --impl-topk torch
python -m moe_routing_bench.cli bench-routing --impl-topk quack
```

### Tests
```bash
pytest -q
```
