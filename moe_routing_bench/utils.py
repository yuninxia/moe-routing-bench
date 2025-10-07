import contextlib
import time
from typing import Tuple

import torch


def get_dtype(name: str):
    name = name.lower()
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


@contextlib.contextmanager
def cuda_timer(sync: bool = True):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"[timer] {(t1 - t0) * 1e3:.3f} ms")


def measure_latency_ms(fn, warmup: int = 10, iters: int = 100, sync: bool = True) -> Tuple[float, float]:
    if torch.cuda.is_available() and sync:
        torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available() and sync:
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available() and sync:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    import numpy as np

    return float(np.mean(times)), float(np.std(times))
