import torch

from moe_routing_bench.routers.base import get_router
from moe_routing_bench.topk_impls import softk_indices_and_gates, top1_indices

def _run_softk(router_name: str = "torch_soft", tokens: int = 32, experts: int = 8, hidden: int = 16, k: int = 3, cf: float = 1.0, renorm: bool = True):
    torch.manual_seed(42)
    x = torch.randn(tokens, hidden)
    logits = torch.randn(tokens, experts)
    topk_idx, gates = softk_indices_and_gates(logits, k, temperature=1.0, normalize="softmax")
    capacity = max(1, int(cf * tokens * k / experts + 0.9999))

    router = get_router(router_name)
    packed, route = router.pack(x, topk_idx, gates, capacity, renorm_after_drop=renorm)
    out = router.combine(packed, route, out_tokens=tokens)

    kept = route.kept_mask
    weights = gates.clone()
    if renorm:
        denom = (weights * kept).sum(-1, keepdim=True).clamp_min(1e-9)
        weights = torch.where(kept, weights / denom, torch.zeros_like(weights))
    else:
        weights = torch.where(kept, weights, torch.zeros_like(weights))

    recon = torch.zeros_like(x)
    for i in range(tokens):
        for j in range(k):
            if kept[i, j]:
                recon[i] += x[i] * weights[i, j]
    torch.testing.assert_close(out, recon, atol=1e-5, rtol=1e-4)


def test_softk_identity():
    _run_softk(router_name="torch_soft", tokens=64, experts=8, hidden=32, k=2, cf=1.1, renorm=True)


def test_top1_special_case():
    torch.manual_seed(0)
    tokens, experts, hidden = 32, 8, 16
    x = torch.randn(tokens, hidden)
    logits = torch.randn(tokens, experts)
    idx = top1_indices(logits)
    capacity = tokens
    router = get_router("torch_soft")
    packed, route = router.pack(x, idx, gates=None, capacity=capacity, renorm_after_drop=False)
    out = router.combine(packed, route, out_tokens=tokens)
    torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)
