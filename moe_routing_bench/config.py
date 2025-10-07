from dataclasses import dataclass


@dataclass
class BenchConfig:
    num_tokens: int = 16384
    hidden_dim: int = 4096
    num_experts: int = 128
    top_k: int = 2
    dtype: str = "float16"
    device: str = "cuda"
    seed: int = 17
    warmup: int = 10
    iters: int = 100
