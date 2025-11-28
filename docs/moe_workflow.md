# MoE Routing Workflow: Step-by-Step

This document provides a detailed walkthrough of the Mixture-of-Experts (MoE) Soft Top-K routing process, with input/output shapes and concrete numerical examples at each step.

## Parameter Settings

This example uses the following parameters:
- Batch size B = 2
- Sequence length S = 4
- Hidden dimension D = 4
- Number of experts E = 4
- Top-K k = 2
- Capacity factor CF = 1.25
- Temperature τ = 1.0

---

## Step 0: Input Reshape

Flatten the input from `[Batch, Seq, Hidden]` to `[Tokens, Hidden]`.

```
Input: x [Batch=2, Seq=4, Hidden=4] → reshape → x_2d [T=8, D=4]
```

**Original tokens x_2d [T=8, D=4]:**
```
┌─────┬──────────────────────┐
│ t0  │ [0.1, 0.2, 0.3, 0.4] │
│ t1  │ [0.5, 0.6, 0.7, 0.8] │
│ t2  │ [0.9, 1.0, 1.1, 1.2] │
│ t3  │ [1.3, 1.4, 1.5, 1.6] │
│ t4  │ [1.7, 1.8, 1.9, 2.0] │
│ t5  │ [2.1, 2.2, 2.3, 2.4] │
│ t6  │ [2.5, 2.6, 2.7, 2.8] │
│ t7  │ [2.9, 3.0, 3.1, 3.2] │
└─────┴──────────────────────┘
```

---

## Step 1: Router Network (Gating Network)

The Router Network is a learnable linear layer (or MLP) that maps each token to scores for E experts.

**Formula:**
```
logits = x_2d @ W_r + b
```

<details>
<summary>View source code (moe_layer.py)</summary>

```python
# moe_routing_bench/modules/moe_layer.py

def _build_router_net(self, hidden_dim: int, num_experts: int) -> None:
    """Initialize router network based on router_arch."""
    arch = self.router_arch.lower()
    if arch == "linear":
        linear = nn.Linear(hidden_dim, num_experts, bias=True)
        nn.init.normal_(linear.weight, mean=0.0, std=hidden_dim ** -0.5)
        nn.init.zeros_(linear.bias)
        self.router_linear = linear
    elif arch == "mlp":
        hidden = max(64, hidden_dim // 2)
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_experts),
        )
        self.router_mlp = mlp
    elif arch == "mlp_hadamard":
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        out = nn.Linear(hidden_dim, num_experts)
        self.router_mlp = mlp
        self.router_out = out

def _router_logits(self, x_2d: Tensor) -> Tensor:
    arch = self.router_arch.lower()
    if arch == "linear":
        return self.router_linear(x_2d)      # x @ W_r + b → [T, E]
    if arch == "mlp":
        return self.router_mlp(x_2d)         # MLP(x) → [T, E]
    if arch == "mlp_hadamard":
        h = self.router_mlp(x_2d)
        return self.router_out(h * x_2d)     # out(MLP(x) ⊙ x) → [T, E]
```

</details>

Where `W_r: [D=4, E=4]`, `b: [E=4]` are learnable parameters.

**Output logits [T=8, E=4]:** (raw score of each token for each expert)
```
┌─────┬──────────────────────────┐
│     │   E0     E1     E2    E3 │
├─────┼──────────────────────────┤
│ t0  │  2.1    0.5    1.8   0.3 │
│ t1  │  0.4    2.3    0.6   1.9 │
│ t2  │  1.9    0.7    2.2   0.4 │
│ t3  │  0.6    2.1    0.5   1.7 │
│ t4  │  2.0    0.8    1.6   0.5 │
│ t5  │  0.5    1.8    0.7   2.4 │
│ t6  │  1.7    0.6    2.3   0.4 │
│ t7  │  0.8    2.0    0.6   1.5 │
└─────┴──────────────────────────┘
```

---

## Step 2: Top-K Selection

Select the top-k experts and their corresponding logit values for each token.

**Formula:**
```
topk_values, topk_idx = torch.topk(logits, k=2, dim=-1)
```

<details>
<summary>View source code (moe_layer.py)</summary>

```python
# moe_routing_bench/modules/moe_layer.py

def _route(self, logits: Tensor) -> Tuple[torch.LongTensor, Optional[Tensor], int]:
    # Returns:
    #   indices: [T, K] - selected expert IDs for each token
    #   gates:   [T, K] - softmax weights for combining expert outputs (None for hard routing)
    #   k_eff:   int    - effective k, actual number of experts per token (for capacity calc)

    if self.strategy == "top1":
        indices = top1_indices(logits)  # [T, 1] - only 1 expert per token
        gates = None                    # hard routing: uniform weight
        k_eff = 1                       # top1 always uses k=1 regardless of --top-k flag
    elif self.strategy == "topk_hard":
        indices = torch.topk(logits, self.top_k, dim=-1).indices  # [T, K]
        gates = None                    # hard routing: uniform weight (1/k each)
        k_eff = self.top_k
    elif self.strategy == "softk":
        indices, gates = softk_indices_and_gates(logits, self.top_k, temperature=1.0)
        # indices: [T, K] - which experts to use
        # gates:   [T, K] - softmax weights for weighted combination
        k_eff = self.top_k
    # ... (expert_choice, hash omitted)
    return indices, gates, k_eff
```

</details>

**Output topk_idx [T=8, k=2]:** (selected expert indices)
```
┌─────┬───────────┐
│ t0  │  E0,  E2  │  ← t0 selects E0 and E2
│ t1  │  E1,  E3  │
│ t2  │  E2,  E0  │
│ t3  │  E1,  E3  │
│ t4  │  E0,  E2  │
│ t5  │  E3,  E1  │
│ t6  │  E2,  E0  │
│ t7  │  E1,  E3  │
└─────┴───────────┘
```

**Output topk_values [T=8, k=2]:** (corresponding logit values)
```
┌─────┬─────────────┐
│ t0  │  2.1,  1.8  │
│ t1  │  2.3,  1.9  │
│ t2  │  2.2,  1.9  │
│ t3  │  2.1,  1.7  │
│ t4  │  2.0,  1.6  │
│ t5  │  2.4,  1.8  │
│ t6  │  2.3,  1.7  │
│ t7  │  2.0,  1.5  │
└─────┴─────────────┘
```

---

## Step 3: Softmax (Gate Computation)

Convert top-k logit values to normalized weights (gates).

**Formula:**
```
gates_i = exp(v_i / τ) / Σ_j exp(v_j / τ)
```

When τ=1.0, this is the standard softmax.

<details>
<summary>View source code (topk_impls.py)</summary>

```python
# moe_routing_bench/topk_impls.py

def softk_indices_and_gates(
    logits: torch.Tensor,  # [T, E] - router scores for each token-expert pair
    k: int,                # number of experts to select per token
    temperature: float = 1.0,  # τ: controls softmax sharpness (1.0 = standard softmax)
    normalize: str = "softmax",
) -> tuple[torch.Tensor, torch.Tensor]:
    # Step 1: Select top-k experts by score
    values, indices = torch.topk(logits, k, dim=-1)
    # values:  [T, K] - the k highest logit scores per token
    # indices: [T, K] - which expert IDs had those scores

    # Step 2: Convert scores to normalized weights
    if normalize == "softmax":
        gates = F.softmax(values / temperature, dim=-1)  # [T, K] - weights sum to 1
    elif normalize == "sum":
        weights = torch.clamp(values, min=0)
        gates = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)

    # Returns:
    #   indices: [T, K] - selected expert IDs (used in Pack to dispatch tokens)
    #   gates:   [T, K] - combination weights (used in Combine for weighted sum)
    return indices, gates.to(values.dtype)
```

</details>

**Example calculation (t0):**
```
v = [2.1, 1.8], τ = 1.0
gates[0] = exp(2.1) / (exp(2.1) + exp(1.8)) = 8.17 / (8.17 + 6.05) = 0.57
gates[1] = exp(1.8) / (exp(2.1) + exp(1.8)) = 6.05 / (8.17 + 6.05) = 0.43
```

**Output gates [T=8, k=2]:** (normalized weights, each row sums to 1)
```
┌─────┬─────────────────────────────────┐
│ t0  │  0.57 (E0),  0.43 (E2)          │
│ t1  │  0.60 (E1),  0.40 (E3)          │
│ t2  │  0.57 (E2),  0.43 (E0)          │
│ t3  │  0.60 (E1),  0.40 (E3)          │
│ t4  │  0.60 (E0),  0.40 (E2)          │
│ t5  │  0.65 (E3),  0.35 (E1)          │
│ t6  │  0.65 (E2),  0.35 (E0)          │
│ t7  │  0.62 (E1),  0.38 (E3)          │
└─────┴─────────────────────────────────┘
```

**This is where Hard Top-K vs Soft Top-K differ:**
- Hard Top-K: gates = [0.5, 0.5] (uniform weights, fixed)
- Soft Top-K: gates = softmax(values) (derived from learnable router network)

---

## Step 4: Pack (Dispatch)

Reorganize tokens by expert, placing them into per-expert buffers.

**Capacity calculation:**
```
capacity = ceil(CF × T × k / E) = ceil(1.25 × 8 × 2 / 4) = ceil(5.0) = 5
```

<details>
<summary>View source code (pack_soft_torch.py)</summary>

```python
# moe_routing_bench/routers/pack_soft_torch.py

class TorchSoftPackCombine(PackCombine):
    def pack(self, x, topk_idx, gates, capacity, renorm_after_drop):
        # Args:
        #   x:        [T, D]  - token embeddings to dispatch
        #   topk_idx: [T, K]  - which experts each token selected (from _route)
        #   gates:    [T, K]  - combination weights (from softmax, or None for hard routing)
        #   capacity: int     - max tokens per expert buffer
        tokens, dim = x.shape
        k = topk_idx.shape[1]

        # Step 1: Assign slots - determine position in each expert's buffer
        # Uses stable argsort to handle ties deterministically
        slots, expert_counts = _assign_slots_vectorized(topk_idx, num_experts, capacity)
        # slots:         [T, K] - slot index within expert buffer (0 to capacity-1)
        # expert_counts: [E]    - actual number of tokens assigned to each expert

        # Step 2: Mark which assignments fit within capacity (kept vs dropped)
        kept_mask = slots < capacity  # [T, K] bool - True if assignment is kept

        # Step 3: Compute flat index for scatter operation
        # packed buffer is [E*C, D], so position = expert_id * capacity + slot
        flat_out_index = expert_id * capacity + slot

        # Step 4: Scatter token embeddings into packed buffer
        packed = zeros(E × C, D)
        packed[flat_out_index] = x[token_idx]  # Copy embeddings to expert buffers

        # Step 5: Handle gates for soft routing (store for use in Combine)
        if gates is not None:
            flat_weights = gates[kept]  # Only keep weights for non-dropped assignments
            if renorm_after_drop:
                # If some assignments dropped, renormalize so weights still sum to 1
                flat_weights = flat_weights / sum_per_token

        # RouteInfo stores mapping info needed for Combine step
        return packed, RouteInfo(...)
```

</details>

**Count assignments received by each expert:**
```
E0: t0, t2, t4, t6 → 4 assignments (< C=5, all kept)
E1: t1, t3, t5, t7 → 4 assignments (< C=5, all kept)
E2: t0, t2, t4, t6 → 4 assignments (< C=5, all kept)
E3: t1, t3, t5, t7 → 4 assignments (< C=5, all kept)
```

**Pack output packed [E=4, C=5, D=4]:**
```
┌────────┬───────┬──────────────────────┬────────┐
│ Expert │ Slot  │ Token Embedding      │ Source │
├────────┼───────┼──────────────────────┼────────┤
│   E0   │   0   │ [0.1, 0.2, 0.3, 0.4] │ t0     │
│        │   1   │ [0.9, 1.0, 1.1, 1.2] │ t2     │
│        │   2   │ [1.7, 1.8, 1.9, 2.0] │ t4     │
│        │   3   │ [2.5, 2.6, 2.7, 2.8] │ t6     │
│        │   4   │ [0.0, 0.0, 0.0, 0.0] │ (empty)│
├────────┼───────┼──────────────────────┼────────┤
│   E1   │   0   │ [0.5, 0.6, 0.7, 0.8] │ t1     │
│        │   1   │ [1.3, 1.4, 1.5, 1.6] │ t3     │
│        │   2   │ [2.1, 2.2, 2.3, 2.4] │ t5     │
│        │   3   │ [2.9, 3.0, 3.1, 3.2] │ t7     │
│        │   4   │ [0.0, 0.0, 0.0, 0.0] │ (empty)│
├────────┼───────┼──────────────────────┼────────┤
│   E2   │   0   │ [0.1, 0.2, 0.3, 0.4] │ t0     │
│        │   1   │ [0.9, 1.0, 1.1, 1.2] │ t2     │
│        │   2   │ [1.7, 1.8, 1.9, 2.0] │ t4     │
│        │   3   │ [2.5, 2.6, 2.7, 2.8] │ t6     │
│        │   4   │ [0.0, 0.0, 0.0, 0.0] │ (empty)│
├────────┼───────┼──────────────────────┼────────┤
│   E3   │   0   │ [0.5, 0.6, 0.7, 0.8] │ t1     │
│        │   1   │ [1.3, 1.4, 1.5, 1.6] │ t3     │
│        │   2   │ [2.1, 2.2, 2.3, 2.4] │ t5     │
│        │   3   │ [2.9, 3.0, 3.1, 3.2] │ t7     │
│        │   4   │ [0.0, 0.0, 0.0, 0.0] │ (empty)│
└────────┴───────┴──────────────────────┴────────┘
```

**RouteInfo records the mapping (used for Combine):**
```
flat_token_idx: records original token ID for each packed position
flat_weights:   records gate weight for each assignment
flat_out_index: records position in packed tensor
```

**Token Dropping:** If an expert receives more tokens than capacity, excess tokens are dropped. No drops occurred in this example.

---

## Step 5: Expert FFN

Each expert is an independent FFN (Feed-Forward Network) that processes its assigned tokens.

**Formula:**
```
expert_out[e] = FFN_e(packed[e])
FFN: Linear(D → 4D) → GELU → Linear(4D → D)
```

<details>
<summary>View source code (experts.py)</summary>

```python
# moe_routing_bench/modules/experts.py

class GroupFFNExperts(nn.Module):
    """Each expert is an independent FFN with its own parameters.
    This is where the "mixture" happens - different experts learn different functions."""

    def __init__(self, num_experts, d_model, ffn_mult=4, activation="gelu"):
        # Each expert has INDEPENDENT parameters (not shared!)
        # This is what makes MoE powerful - each expert can specialize
        hidden = ffn_mult * d_model  # typically 4x expansion
        self.w1 = nn.Parameter(torch.empty(num_experts, d_model, hidden))  # [E, D, 4D]
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden, d_model))  # [E, 4D, D]
        self.b1 = nn.Parameter(torch.zeros(num_experts, hidden))           # [E, 4D]
        self.b2 = nn.Parameter(torch.zeros(num_experts, d_model))          # [E, D]

    def forward(self, packed_inputs, valid_load):
        # Args:
        #   packed_inputs: [E, C, D] - tokens organized by expert (from Pack)
        #   valid_load:    [E]       - how many real tokens in each expert's buffer
        outputs = zeros(E, C, D)

        for expert_id in range(E):
            load = valid_load[expert_id]  # Number of actual tokens (rest are padding)
            if load == 0:
                continue  # Skip experts with no tokens
            x_e = packed_inputs[expert_id, :load]  # [load, D] - only process real tokens

            # Standard FFN: Linear → GELU → Linear
            # Note: Each expert uses its OWN weights (w1[expert_id], w2[expert_id])
            h = x_e @ self.w1[expert_id] + self.b1[expert_id]  # [load, 4D] - expand
            h = gelu(h)                                         # non-linearity
            y_e = h @ self.w2[expert_id] + self.b2[expert_id]  # [load, D] - project back

            outputs[expert_id, :load] = y_e
        return outputs  # [E, C, D] - ready for Combine
```

</details>

**Expert FFN output expert_out [E=4, C=5, D=4]:**
```
┌────────┬───────┬──────────────────────┐
│ Expert │ Slot  │ FFN Output           │
├────────┼───────┼──────────────────────┤
│   E0   │   0   │ [y0_e0]              │ ← FFN_0(t0 embedding)
│        │   1   │ [y2_e0]              │ ← FFN_0(t2 embedding)
│        │   2   │ [y4_e0]              │ ← FFN_0(t4 embedding)
│        │   3   │ [y6_e0]              │ ← FFN_0(t6 embedding)
│        │   4   │ [0,0,0,0]            │ ← empty slot, no computation
├────────┼───────┼──────────────────────┤
│   E1   │   0   │ [y1_e1]              │ ← FFN_1(t1 embedding)
│        │   1   │ [y3_e1]              │ ← FFN_1(t3 embedding)
│        │   2   │ [y5_e1]              │ ← FFN_1(t5 embedding)
│        │   3   │ [y7_e1]              │ ← FFN_1(t7 embedding)
│        │   4   │ [0,0,0,0]            │
├────────┼───────┼──────────────────────┤
│   E2   │  ...  │ (similar to E0)      │
├────────┼───────┼──────────────────────┤
│   E3   │  ...  │ (similar to E1)      │
└────────┴───────┴──────────────────────┘
```

**Note:** Each expert has independent parameters, so `FFN_0(x) ≠ FFN_1(x)`.

---

## Step 6: Combine (Gather)

Use gates as weights to aggregate expert outputs back to original token order.

**Formula:**
```
out[token_i] = Σ_j (gate[i,j] × expert_j_output[token_i])
```

<details>
<summary>View source code (pack_soft_torch.py)</summary>

```python
# moe_routing_bench/routers/pack_soft_torch.py

class TorchSoftPackCombine(PackCombine):
    def combine(self, y, route, out_tokens):
        # Args:
        #   y:          [E×C, D]  - flattened expert outputs (from Expert FFN)
        #   route:      RouteInfo - mapping info saved during Pack
        #   out_tokens: int       - number of original tokens (T)
        out = zeros(T, D)  # Output buffer, same shape as original input

        # Step 1: Gather expert outputs using saved indices
        # route.flat_out_index tells us where each kept assignment is in the packed buffer
        contrib = y[route.flat_out_index]  # [num_kept, D]

        # Step 2: Apply gate weights (THIS IS WHERE SOFT ROUTING HAPPENS!)
        # - Soft routing: multiply each expert's output by its gate weight
        # - Hard routing: flat_weights is None, so each expert contributes equally
        if route.flat_weights is not None:
            contrib = contrib * route.flat_weights.unsqueeze(1)  # [num_kept, D]
            # Example: if token t0 has gates [0.57, 0.43] for E0, E2
            #   contrib_from_E0 = 0.57 * FFN_0(t0)
            #   contrib_from_E2 = 0.43 * FFN_2(t0)

        # Step 3: Scatter-add contributions back to original token positions
        # route.flat_token_idx tells us which original token each contribution belongs to
        out.index_add_(0, route.flat_token_idx, contrib)
        # Result: out[t0] = 0.57 * FFN_0(t0) + 0.43 * FFN_2(t0)

        return out  # [T, D] - ready to reshape back to [B, S, D]
```

</details>

**Combine computation:**
```
┌─────┬─────────────────────────────────────────────────────┬──────────────┐
│Token│ Computation                                         │ Final Output │
├─────┼─────────────────────────────────────────────────────┼──────────────┤
│ t0  │ 0.57 × y0_e0  +  0.43 × y0_e2                       │ out[0]       │
│ t1  │ 0.60 × y1_e1  +  0.40 × y1_e3                       │ out[1]       │
│ t2  │ 0.43 × y2_e0  +  0.57 × y2_e2                       │ out[2]       │
│ t3  │ 0.60 × y3_e1  +  0.40 × y3_e3                       │ out[3]       │
│ t4  │ 0.60 × y4_e0  +  0.40 × y4_e2                       │ out[4]       │
│ t5  │ 0.35 × y5_e1  +  0.65 × y5_e3                       │ out[5]       │
│ t6  │ 0.35 × y6_e0  +  0.65 × y6_e2                       │ out[6]       │
│ t7  │ 0.62 × y7_e1  +  0.38 × y7_e3                       │ out[7]       │
└─────┴─────────────────────────────────────────────────────┴──────────────┘
```

**Output out [T=8, D=4]:**
```
┌─────┬──────────────────────────────────┐
│ t0  │ 0.57×y0_e0 + 0.43×y0_e2          │
│ t1  │ 0.60×y1_e1 + 0.40×y1_e3          │
│ t2  │ 0.43×y2_e0 + 0.57×y2_e2          │
│ t3  │ 0.60×y3_e1 + 0.40×y3_e3          │
│ t4  │ 0.60×y4_e0 + 0.40×y4_e2          │
│ t5  │ 0.35×y5_e1 + 0.65×y5_e3          │
│ t6  │ 0.35×y6_e0 + 0.65×y6_e2          │
│ t7  │ 0.62×y7_e1 + 0.38×y7_e3          │
└─────┴──────────────────────────────────┘
```

**Gates are actually used here** This is the core difference between Soft Top-K and Hard Top-K.

---

## Step 7: Output Reshape

Reshape output back to original `[Batch, Seq, Hidden]` shape.

```
Output: out [T=8, D=4] → reshape → [Batch=2, Seq=4, Hidden=4]
```

---

## Summary Table

| Step | Operation | Input Shape | Output Shape | Key Formula/Note |
|------|-----------|-------------|--------------|------------------|
| 0 | Reshape | [B,S,D] | [T,D] | T = B×S |
| 1 | Router Network | [T,D] | [T,E] | logits = x @ W_r + b |
| 2 | Top-K Selection | [T,E] | [T,k], [T,k] | topk_idx, topk_values |
| 3 | Softmax | [T,k] | [T,k] | gates = softmax(values/τ) |
| 4 | Pack | [T,D] | [E,C,D] | Reorganize by expert |
| 5 | Expert FFN | [E,C,D] | [E,C,D] | y_e = FFN_e(x_e) |
| 6 | Combine | [E,C,D] | [T,D] | out[i] = Σ(gate × y) |
| 7 | Reshape | [T,D] | [B,S,D] | Restore original shape |

<details>
<summary>View complete forward pass (moe_layer.py)</summary>

```python
# moe_routing_bench/modules/moe_layer.py

def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, torch.Tensor]]:
    """Complete MoE forward pass orchestrating all 7 steps."""

    # ==================== Step 0: Flatten ====================
    # MoE operates on 2D tensor [T, D], not 3D [B, S, D]
    orig_shape = x.shape                    # Save for final reshape
    tokens = x.shape[0] * x.shape[1]        # T = batch_size × seq_len
    x_2d = x.reshape(tokens, hidden)        # [B,S,D] → [T,D]

    # ==================== Step 1: Router Network ====================
    # Learnable layer that scores each token for each expert
    logits = self._router_logits(x_2d)      # [T, E] - raw scores

    # ==================== Step 2-3: Routing ====================
    # Select top-k experts and compute combination weights
    topk_idx, gates, k_eff = self._route(logits)
    # topk_idx: [T, K] - which experts each token uses
    # gates:    [T, K] - combination weights (None for hard routing)
    # k_eff:    int    - effective k (1 for top1, else self.top_k)

    capacity = self._capacity(tokens, k_eff)  # Buffer size per expert

    # ==================== Step 4: Pack (Dispatch) ====================
    # Reorganize tokens into per-expert buffers
    router = self._get_router()
    packed, route = router.pack(x_2d, topk_idx, gates, capacity)
    # packed: [E×C, D] - tokens organized by expert
    # route:  RouteInfo - mapping info for Combine step

    # ==================== Step 5: Expert FFN ====================
    # Each expert processes its assigned tokens independently
    packed_view = packed.view(num_experts, capacity, hidden)  # [E, C, D]
    expert_out = self.experts(packed_view, route.expert_counts)
    # expert_out: [E, C, D] - each expert's output
    expert_flat = expert_out.view(num_experts * capacity, hidden)

    # ==================== Step 6: Combine (Gather) ====================
    # Aggregate expert outputs using gates as weights
    combined = router.combine(expert_flat, route, out_tokens=tokens)
    # combined: [T, D] - weighted sum of expert outputs per token

    # ==================== Step 7: Reshape ====================
    out = combined.view(*orig_shape)  # [T,D] → [B,S,D]

    # Compute auxiliary stats (drop rate, load balance, etc.)
    stats = self._compute_stats(route, capacity, tokens, k_eff, logits, gates)
    return out, stats
```

</details>

---

## Key Concepts Summary

### Pack vs Combine

| Operation | Purpose | Alternative Names |
|-----------|---------|-------------------|
| **Pack** | Dispatch tokens to per-expert buffers based on routing | Dispatch, Scatter |
| **Combine** | Aggregate expert outputs back to original order using gates | Gather, Reduce |

### Where Gates Are Used

```
Step 3 (Softmax):   gates are computed
Step 4 (Pack):      gates are stored in RouteInfo
Step 5 (Expert):    gates not involved
Step 6 (Combine):   gates used for weighted aggregation ← Actually used here!
```

### Capacity and Token Dropping

```
capacity = ceil(CF × T × k / E)
```

- **CF < 1.0**: Token dropping is guaranteed
- **CF = 1.0**: Theoretically exact, but uneven distribution causes dropping
- **CF ≥ 1.05**: Essentially eliminates dropping
- **CF = 1.25**: Recommended value with safety margin

---

## Code Reference

| Step | Code Location |
|------|---------------|
| Router Network | `moe_layer.py: _router_logits()` |
| Top-K + Softmax | `topk_impls.py: softk_indices_and_gates()` |
| Pack | `pack_soft_torch.py: TorchSoftPackCombine.pack()` |
| Expert FFN | `experts.py: GroupFFNExperts.forward()` |
| Combine | `pack_soft_torch.py: TorchSoftPackCombine.combine()` |
| Full Pipeline | `moe_layer.py: MoEFeedForward.forward()` |

---

## Appendix: Frequently Asked Questions

### Q1: What is Capacity? What is Capacity Factor (CF)? Why is the formula designed this way?

**The Problem: Static Tensor Shapes**

On GPUs, tensor shapes must be determined at compile time and cannot change dynamically. However, routing decisions are made at runtime, and we cannot know in advance exactly how many tokens will be assigned to each expert. Therefore, we must:

1. Pre-allocate a **fixed-size buffer** for each expert
2. This buffer size is called **capacity**

**Ideal Case: Perfect Uniform Distribution**

If routing were perfectly uniform, each expert would receive an equal number of tokens:

```
Tokens per expert = (Total tokens × k) / E

Example: T=8 tokens, k=2, E=4 experts
  Total assignments = 8 × 2 = 16
  Ideal per expert = 16 / 4 = 4
```

**Reality: Non-uniform Distribution**

In practice, routing is learned dynamically and will never be perfectly uniform:

```
Actual distribution might be:
  E0: 6 tokens  ← overloaded!
  E1: 5 tokens  ← overloaded!
  E2: 3 tokens
  E3: 2 tokens
  Total: 16 assignments
```

**The Capacity Formula**

```
capacity = ceil(CF × T × k / E)
         = ceil(CF × ideal_tokens_per_expert)
```

Where:
- `T × k / E` = ideal load per expert assuming uniform distribution
- `CF` = safety margin to handle non-uniform distribution
- `ceil()` = round up to ensure integer (can't process half a token)

**What Different CF Values Mean**

| CF | Meaning | Effect |
|----|---------|--------|
| CF = 1.0 | buffer = ideal uniform size | Any imbalance causes dropping |
| CF = 1.25 | buffer = ideal × 1.25 | 25% margin, tolerates mild imbalance |
| CF = 1.5 | buffer = ideal × 1.5 | 50% margin, safer |

**Visual Example**

```
CF = 1.0 (capacity = 4):
┌─────────────────────────┐
│ E0 buffer: [_][_][_][_] │     receives 6 → 2 DROPPED!
│ E1 buffer: [_][_][_][_] │     receives 5 → 1 DROPPED!
│ E2 buffer: [_][_][_][_] │     receives 3 → OK
│ E3 buffer: [_][_][_][_] │     receives 2 → OK
└─────────────────────────┘

CF = 1.5 (capacity = 6):
┌───────────────────────────────┐
│ E0 buffer: [_][_][_][_][_][_] │  receives 6 → OK!
│ E1 buffer: [_][_][_][_][_][_] │  receives 5 → OK!
│ E2 buffer: [_][_][_][_][_][_] │  receives 3 → OK (3 slots wasted)
│ E3 buffer: [_][_][_][_][_][_] │  receives 2 → OK (4 slots wasted)
└───────────────────────────────┘
```

**The Trade-off**

| Higher CF | Pros | Cons |
|-----------|------|------|
| ↑ | Less token dropping | More memory waste |
| ↑ | Better model quality | More computation overhead |

Switch Transformer paper found **CF = 1.0 ~ 1.25** provides the best balance.

**References**:
- [Mixture of Experts Explained - Hugging Face](https://huggingface.co/blog/moe)
- [Switch Transformers (Fedus et al., 2021)](https://arxiv.org/abs/2101.03961)

---

### Q2: What is the difference between Hard Top-K and Soft Top-K routing?

**The Core Difference: How expert outputs are combined**

Both Hard and Soft Top-K use the same discrete selection (choosing which k experts to activate). The difference is in the **combination weights**:

| Aspect | Hard Top-K | Soft Top-K |
|--------|-----------|------------|
| Selection | Top-k by logits | Top-k by logits |
| Gates | Uniform: [1/k, 1/k, ...] | softmax(topk_values) |
| Combination | Simple average | Weighted sum |

**Example (k=2, token selects E0 and E2):**

```
logits = [2.1, 0.5, 1.8, 0.3]  (for experts E0, E1, E2, E3)
         ↓
Top-2 selection: E0 (2.1), E2 (1.8)
         ↓
Hard Top-K:
  gates = [0.5, 0.5]
  output = 0.5 × E0_out + 0.5 × E2_out

Soft Top-K:
  gates = softmax([2.1, 1.8]) = [0.57, 0.43]
  output = 0.57 × E0_out + 0.43 × E2_out
```

The gates are *indirectly learnable* because:

1. Gates = softmax(topk_values)
2. topk_values come from logits
3. logits = Router_Network(x) = x @ W_r + b
4. **W_r is learnable**

Through backpropagation, the model learns W_r to produce logits that result in desired gate distributions.

---

### Q3: What are Pack and Combine? Why are they needed?

**The Problem**

After routing, we know which tokens go to which experts, but the data is still in original token order. Each expert needs a contiguous batch of its assigned tokens for efficient GPU computation.

```
Original order:  [t0, t1, t2, t3, t4, t5, t6, t7]
Routing:          E0  E1  E0  E1  E0  E1  E0  E1

Expert E0 needs: [t0, t2, t4, t6] as contiguous batch
Expert E1 needs: [t1, t3, t5, t7] as contiguous batch
```

**Pack (Dispatch)**

Reorganizes tokens from original order to per-expert buffers:

```
Input:  x [T, D]           (original token order)
Output: packed [E, C, D]   (grouped by expert)
```

Also records the mapping in RouteInfo for later reconstruction.

**Combine (Gather)**

Reassembles expert outputs back to original token order, applying gate weights:

```
Input:  expert_out [E, C, D]  (expert outputs)
Output: out [T, D]            (original order, weighted sum)

Formula: out[i] = Σ_j (gate[i,j] × expert_j_output[i])
```

**Alternative Names**

| Operation | Other Names |
|-----------|-------------|
| Pack | Dispatch, Scatter |
| Combine | Gather, Reduce |

---

### Q4: What is the Router Network (Gating Network)?

**Definition**

The Router Network (also called Gating Network) is a learnable neural network that determines which experts process each token. It maps token representations to expert scores.

**Standard Formula (Shazeer et al., 2017)**

```
logits = x · W_r + b
G(x) = Softmax(logits)
```

Where:
- `x ∈ ℝ^d`: Token hidden representation (input)
- `W_r ∈ ℝ^{d×E}`: **Learnable** router weight matrix
- `b ∈ ℝ^E`: Learnable bias
- `logits ∈ ℝ^E`: Raw routing scores (one per expert)

**Router Architectures in This Project**

| Architecture | Formula | Complexity |
|--------------|---------|------------|
| `linear` (default) | `logits = x · W + b` | Simple |
| `mlp` | `logits = Linear(GELU(Linear(x)))` | Medium |
| `mlp_hadamard` | `logits = Linear(MLP(x) ⊙ x)` | Complex |

**Key Point**: The router is trained jointly with experts through backpropagation. This joint training is essential—decoupled training leads to suboptimal expert specialization (Farhat et al., 2023).

---

### Q5: What is the Temperature (τ) in Softmax?

**Formula**

```
gates_i = exp(v_i / τ) / Σ_j exp(v_j / τ)
```

**Effect of Temperature**

| τ | Effect | Example: v = [5.0, 3.0] |
|---|--------|-------------------------|
| τ → 0 | One-hot (winner-take-all) | [1.0, 0.0] |
| τ = 1.0 | Standard softmax | [0.88, 0.12] |
| τ → ∞ | Uniform distribution | [0.5, 0.5] |

**Mathematical Intuition**:
- As τ → 0: `exp(v_i/τ)` grows fastest for the largest v_i, dominating the sum
- As τ → ∞: `v_i/τ → 0` for all i, so `exp(v_i/τ) → 1`, giving uniform weights

**In This Project**: τ is fixed at 1.0 (standard softmax). We include it in formulas for completeness and potential future tuning, but it currently has no effect on routing.

---

### Q6: What is E (number of experts)? Why do different experiments use different E values?

**Definition**

`E` = number of experts in the MoE layer. Each expert is an independent FFN with its own parameters.

**E in the Capacity Formula**

```
capacity = ceil(CF × T × k / E)
```

Larger E means smaller capacity per expert (more experts share the same total tokens).

**Why Different E Values in Experiments**

| Experiment | E Value | Reason |
|------------|---------|--------|
| **Capacity Sweep** | 64, 128, 256 | Microbenchmark (pack/combine only), no training cost. Large E reveals capacity behavior at scale. |
| **Unified Routing Sweep** | 8 | Full end-to-end training. E=8 is fast to train while still showing routing differences. |
| **Larger Scale Validation** | 32 | Validates that conclusions from E=8 hold at larger scale. |

**The Design Logic**

1. **Capacity Sweep** (cheap microbenchmark):
   - Only tests pack/combine operations, no gradient computation
   - Can afford large E (64-256) to see scaling behavior
   - Finding: CF ≈ 1.05–1.25 eliminates drops with minimal overhead

2. **Unified Sweep** (expensive training):
   - Full forward + backward + optimizer steps
   - E=8 is practical for quick iteration
   - Uses CF insights from capacity sweep

3. **Larger Scale** (validation):
   - E=32 confirms ranking (expert-choice > softk > hash > topk-hard > top1) holds
   - Ensures conclusions aren't artifacts of small E

**Typical E Values in Literature**

| Model | E | Notes |
|-------|---|-------|
| Switch Transformer | 8-128 | Started the modern MoE trend |
| Mixtral 8x7B | 8 | Production model |
| DeepSeek-MoE | 64 | Fine-grained experts |
| Our TinyMoE | 8 (default) | Research scale |

---

### Q7: What is Throughput? Is it training or inference throughput?

**Definition**

Throughput measures how many tokens the system processes per second.

```
throughput = tokens_processed / time_elapsed  (tokens/s)
```

**Throughput in Different Experiments**

| Experiment | What's Measured | Includes |
|------------|-----------------|----------|
| **Training experiments** | End-to-end training throughput | Forward + Backward + Optimizer step (DDP sync) |
| **Capacity sweeps** | Pack/Combine microbenchmark | Forward pass only (no gradients) |

**How It's Computed in Code**

```python
# scripts/train_small.py
tokens_processed = batch_size * seq_len * num_steps
elapsed_time = end_time - start_time
tokens_per_s = tokens_processed / elapsed_time
```

**Training vs Inference Throughput**

We only measure **training throughput** in this project. However, the relative ranking of routing strategies is expected to hold for inference:

| Strategy | Training Speed | Inference Speed (expected) |
|----------|---------------|---------------------------|
| top1 | Fastest | Fastest |
| topk-hard | Medium | Medium |
| softk | Slower | Slower |
| expert-choice | Slower | Slower |

The overhead comes from:
- More experts per token (k=2 vs k=1)
- Softmax computation for gates
- Weighted combination in Combine step

---

### Q8: What are the hyperparameters? How do they differ across experiments?

**Model Hyperparameters**

| Parameter | TinyMoE (E=8) | Large Scale (E=32) | BPE |
|-----------|---------------|-------------------|-----|
| `dim` | 256 | 512 | 256 |
| `layers` | 4 | 4 | 4 |
| `heads` | 4 | 4 | 4 |
| `ffn_mult` | 4 | 4 | 4 |
| `num_experts` (E) | 8 | 32 | 8 |

**Training Hyperparameters**

| Parameter | TinyMoE (E=8) | Large Scale (E=32) | BPE |
|-----------|---------------|-------------------|-----|
| `batch_size` (per GPU) | 32 | 32 | 32 |
| `seq_len` | 256 | 256 | 256 |
| `lr` | 3e-4 | 1e-4 | 3e-4 |
| `warmup_steps` | 50 | 200 | 50 |
| `max_steps` | 1200 | 2000 | 1200 |

**MoE-Specific Hyperparameters**

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `top_k` (k) | Experts per token | 1, 2 |
| `capacity_factor` (CF) | Buffer safety margin | 1.0, 1.25, 1.5 |
| `load_balance_alpha` | Aux loss weight | 1e-2 |
| `router_arch` | Gating network type | linear, mlp, mlp_hadamard |

**Command Line Flags**

```bash
python scripts/train_small.py \
    --num-experts 8 \
    --top-k 2 \
    --capacity-factor 1.25 \
    --router softk \
    --dim 256 \
    --layers 4 \
    --lr 3e-4 \
    --max-steps 1200
```

---

### Q9: What are the PERFT variants? What is a LoRA adapter?

**Background: PEFT and LoRA**

- **PEFT** (Parameter-Efficient Fine-Tuning): Fine-tune large models by only updating a small subset of parameters
- **LoRA** (Low-Rank Adaptation): A PEFT method that adds trainable low-rank matrices to frozen weights

```
Original:  y = Wx
LoRA:      y = Wx + BAx    where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×d}, r << d
```

Only A and B are trained; W stays frozen. This reduces trainable parameters significantly.

**PERFT: Combining MoE with PEFT**

PERFT applies MoE-style routing to PEFT adapters. Instead of one shared adapter, multiple adapter "experts" are routed to different tokens.

**PERFT Variants**

| Variant | Description | Router | Capacity |
|---------|-------------|--------|----------|
| **PERFT-R** (Routed) | LoRA experts with independent trainable router | New router (learned) | Highest |
| **PERFT-E** (Embedded) | LoRA experts share router with base MoE FFN | Shared (from base model) | Medium |
| **PERFT-D** (Dense) | All LoRA experts applied to every token (no routing) | None | Low |
| **PERFT-S** (Single) | Single shared LoRA adapter | N/A | Lowest |

**Architecture Diagram**

```
PERFT-R:  Token → [Independent Router] → Top-k LoRA Experts → Weighted Sum
PERFT-E:  Token → [Base MoE Router] → Paired LoRA Expert → Output
PERFT-D:  Token → [All LoRA Experts] → Sum
PERFT-S:  Token → [Single LoRA] → Output
```

**Performance Ranking**

From best to worst quality (at similar parameter count):

```
PERFT-R ≥ PERFT-E > PERFT-D > PERFT-S
```

**Why PERFT-R Works Best**

1. **Independent routing**: Can learn task-specific expert specialization
2. **Sparse activation**: Only k adapters per token (efficient)
3. **Capacity**: More total adapter parameters than single shared

**Code Location**

```bash
# Run PERFT variants sweep
bash scripts/run_perft_variants_quick.sh

# Plot results
python scripts/plot_perft_variants.py
```

**Key Hyperparameters for PERFT**

| Parameter | Description | Values in Our Experiments |
|-----------|-------------|--------------------------|
| `rank` (r) | LoRA rank | 8, 16, 32 |
| `peft_experts` (N) | Number of LoRA experts | 4, 8 |
| `topk` | LoRA experts per token | 1, 2 |

---

### Q10: What are Hash Routing and Expert-Choice Routing?

These are two alternative routing strategies beyond standard top-k routing.

---

#### Hash Routing

**Idea**: Assign tokens to experts based on their position, ignoring content entirely.

**Formula**:
```
expert_id = hash(token_position) % num_experts
```

**Implementation** (from `moe_layer.py`):
```python
def _route_hash(self, logits: Tensor):
    token_ids = torch.arange(tokens, device=device)
    A, B, C = 1315423911, 2654435761, 97  # Hash constants

    base = (token_ids * A + B) % num_experts
    indices[:, 0] = base
    for j in range(1, k):
        indices[:, j] = (base + j * C) % num_experts  # Spread across k experts

    gates = torch.full((tokens, k), 1.0 / k)  # Uniform weights
    return indices, gates
```

**Characteristics**:

| Aspect | Hash Routing |
|--------|--------------|
| Load Balance | **Perfect** (CV = 0) - deterministic uniform distribution |
| Content Awareness | **None** - ignores token content entirely |
| Learnable | **No** - fixed assignment |
| Gates | Uniform (1/k each) |
| Quality (PPL) | **Worst** among all strategies |

**Why Include Hash?**

Hash serves as a **baseline** to test the hypothesis: "Is load balance alone sufficient for good quality?"

**Answer: No.** Hash achieves perfect load balance but worst PPL, proving that content-aware routing is essential.

---

#### Expert-Choice Routing

**Idea**: Invert the selection direction. Instead of tokens choosing experts, **experts choose tokens**.

**Standard Top-K**: Each token picks its top-k experts
```
Token → "I want Expert 0 and Expert 2"
```

**Expert-Choice**: Each expert picks its top tokens
```
Expert 0 → "I want Token 1, Token 5, Token 7"
Expert 1 → "I want Token 0, Token 3, Token 6"
...
```

**Implementation** (from `moe_layer.py`):
```python
def _route_expert_choice(self, logits: Tensor):
    # Step 1: Each expert selects its top tokens
    scores_T = logits.transpose(0, 1)  # [E, T]
    _, top_tok = torch.topk(scores_T, k=tokens_per_expert, dim=1)

    # Step 2: Create mask of valid (expert, token) pairs
    mask = torch.zeros_like(scores_T, dtype=torch.bool)
    mask.scatter_(1, top_tok, True)
    mask_token_expert = mask.transpose(0, 1)  # [T, E]

    # Step 3: Each token picks top-k from experts that selected it
    masked_logits = logits.masked_fill(~mask_token_expert, float("-inf"))
    vals, idx = torch.topk(masked_logits, k=k, dim=-1)

    gates = torch.softmax(vals, dim=-1)
    return idx, gates
```

**Characteristics**:

| Aspect | Expert-Choice |
|--------|---------------|
| Load Balance | **Good** - experts control their own load |
| Token Dropping | **Minimal** - built-in capacity awareness |
| Learnable | **Yes** - uses router logits |
| Gates | Softmax (learned weights) |
| Quality (PPL) | **Best** or tied with softk |

**Why Expert-Choice Works Well**:

1. **Balanced by construction**: Each expert explicitly limits how many tokens it takes
2. **No capacity overflow**: Expert capacity is enforced during selection, not after
3. **Content-aware**: Still uses learned router logits for scoring

---

#### Comparison Table

| Strategy | Selection Direction | Load Balance | Content-Aware | Gates | PPL Rank |
|----------|---------------------|--------------|---------------|-------|----------|
| **top1** | Token → 1 Expert | Poor | Yes | None (hard) | 5th (worst) |
| **topk-hard** | Token → k Experts | Medium | Yes | Uniform | 4th |
| **softk** | Token → k Experts | Medium | Yes | Softmax | 2nd |
| **hash** | Position → k Experts | Perfect | No | Uniform | 5th (worst) |
| **expert-choice** | Expert → Tokens | Good | Yes | Softmax | 1st (best) |

**Key Insight**: Expert-choice achieves the best of both worlds—good load balance AND content awareness.

<details>
<summary>View source code (moe_layer.py)</summary>

```python
# moe_routing_bench/modules/moe_layer.py

def _route_hash(self, logits: Tensor) -> Tuple[torch.LongTensor, Tensor]:
    """Content-agnostic hash routing with uniform gates."""
    tokens, num_experts = logits.shape
    device, dtype, k = logits.device, logits.dtype, self.top_k

    # Deterministic hash based on token position
    token_ids = torch.arange(tokens, device=device, dtype=torch.long)
    A, B, C = 1315423911, 2654435761, 97

    base = (token_ids * A + B) % num_experts
    indices = torch.empty(tokens, k, dtype=torch.long, device=device)
    indices[:, 0] = base
    for j in range(1, k):
        indices[:, j] = (base + j * C) % num_experts

    gates = torch.full((tokens, k), 1.0 / float(k), dtype=dtype, device=device)
    return indices, gates


def _route_expert_choice(self, logits: Tensor) -> Tuple[torch.LongTensor, Tensor]:
    """Expert-choice routing: experts select top tokens, then tokens pick from selected experts."""
    tokens, num_experts = logits.shape
    k = self.top_k

    # Calculate how many tokens each expert can select
    expected = tokens * k / max(1, num_experts)
    tokens_per_expert = max(1, int(math.ceil(self.capacity_factor * expected)))
    tokens_per_expert = min(tokens_per_expert, tokens, 64)

    # Step 1: Each expert selects its top tokens
    scores_T = logits.transpose(0, 1)  # [E, T]
    _, top_tok = torch.topk(scores_T, k=tokens_per_expert, dim=1)
    mask = torch.zeros_like(scores_T, dtype=torch.bool)
    mask.scatter_(1, top_tok, True)
    mask_token_expert = mask.transpose(0, 1)  # [T, E]

    # Step 2: Tokens select top-k from experts that chose them
    masked_logits = logits.masked_fill(~mask_token_expert, float("-inf"))
    vals, idx = torch.topk(masked_logits, k=min(k, num_experts), dim=-1)

    # Fallback for tokens not selected by any expert
    mask_any = mask_token_expert.any(dim=1, keepdim=True)
    if not mask_any.all():
        std_vals, std_idx = torch.topk(logits, k=min(k, num_experts), dim=-1)
        vals = torch.where(mask_any, vals, std_vals)
        idx = torch.where(mask_any, idx, std_idx)

    gates = torch.softmax(vals, dim=-1).to(dtype)
    return idx, gates
```

</details>
