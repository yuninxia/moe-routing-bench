from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

Tensor = torch.Tensor


@dataclass
class RouteInfo:
    tokens: int
    experts: int
    k: int
    capacity: int
    topk_idx: torch.LongTensor
    gates: Optional[Tensor]
    slots: torch.LongTensor
    kept_mask: torch.BoolTensor
    expert_counts: torch.LongTensor
    renorm_after_drop: bool
    flat_token_idx: torch.LongTensor
    flat_out_index: torch.LongTensor
    flat_weights: Optional[Tensor]


class PackCombine:
    name: str = "base"

    def pack(
        self,
        x: Tensor,
        topk_idx: torch.LongTensor,
        gates: Optional[Tensor],
        capacity: int,
        renorm_after_drop: bool,
    ) -> tuple[Tensor, RouteInfo]:
        raise NotImplementedError

    def combine(self, y: Tensor, route: RouteInfo, out_tokens: int) -> Tensor:
        raise NotImplementedError


_REGISTRY: Dict[str, Callable[[], PackCombine]] = {}


def register_router(name: str, ctor: Callable[[], PackCombine]) -> None:
    _REGISTRY[name] = ctor


def get_router(name: str) -> PackCombine:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown router backend: {name}. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[name]()
