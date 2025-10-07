import torch

from moe_routing_bench.routing import (
    GatingLinear,
    make_fake_batch,
    route_identity_pipeline,
    topk_logits_then_softmax_over_k,
)
from moe_routing_bench.utils import get_dtype


def test_identity_routing_equivalence():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens, hidden, experts, top_k = 1024, 256, 32, 2
    dtype_name = "float16" if device == "cuda" else "float32"
    dtype = get_dtype(dtype_name)

    x = make_fake_batch(tokens, hidden, dtype_name, device)
    gating = GatingLinear(hidden, experts, dtype=dtype, device=device)
    logits = gating(x)

    indices, weights = topk_logits_then_softmax_over_k(logits, k=top_k, impl="torch")
    y, _ = route_identity_pipeline(x, experts, indices, weights)

    x_rep = x.unsqueeze(1).expand(tokens, top_k, hidden)
    y_ref = (x_rep * weights.unsqueeze(-1)).sum(dim=1)

    torch.testing.assert_close(y, y_ref, rtol=1e-3, atol=1e-3)
