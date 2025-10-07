from typing import Tuple

import torch


def torch_topk(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    values, indices = torch.topk(logits, k, dim=-1)
    return values, indices


def quack_topk(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attempt to use QuACK's topk, falling back to torch on failure."""

    try:
        from quack.topk import topk as quack_topk_fn  # type: ignore
    except Exception:
        return torch_topk(logits, k)

    try:
        out = quack_topk_fn(logits, k)
    except Exception:
        return torch_topk(logits, k)

    if isinstance(out, (tuple, list)) and len(out) == 2:
        values, indices = out
        return values, indices

    if torch.is_tensor(out) and out.dtype in (torch.int32, torch.int64):
        indices = out
        values = torch.gather(logits, dim=-1, index=indices)
        return values, indices

    return torch_topk(logits, k)


def run_topk(logits: torch.Tensor, k: int, impl: str):
    impl = impl.lower()
    if impl == "quack":
        return quack_topk(logits, k)
    return torch_topk(logits, k)
