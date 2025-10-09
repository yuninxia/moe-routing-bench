#!/usr/bin/env python3
"""Minimal end-to-end MoE training script (single GPU, character LM by default)."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from moe_routing_bench.modules import MoEFeedForward

Tensor = torch.Tensor

# -----------------------------------------------------------------------------
# Data utilities


def _load_text(path: Optional[str]) -> str:
    if path is None:
        random.seed(0)
        vocab = [chr(i) for i in range(97, 123)] + [" ", "\n"]
        return "".join(random.choice(vocab) for _ in range(200000))
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class CharDataset(Dataset):
    def __init__(self, text: str, seq_len: int, vocab: Vocab) -> None:
        self.seq_len = seq_len
        self.vocab = vocab
        ids = torch.tensor(vocab.encode(text), dtype=torch.long)
        self.data = ids

    def __len__(self) -> int:
        return max(0, self.data.size(0) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + 1 + self.seq_len]
        return x, y


def build_dataloaders(path: Optional[str], seq_len: int, batch_size: int, num_workers: int = 0):
    text = _load_text(path)
    split = int(len(text) * 0.9)
    train_text, val_text = text[:split], text[split:]

    chars = sorted(set(text))
    vocab = Vocab(stoi={c: i for i, c in enumerate(chars)}, itos=list(chars))

    train_ds = CharDataset(train_text, seq_len, vocab)
    val_ds = CharDataset(val_text, seq_len, vocab)

    def collate(batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs)
        y = torch.stack(ys)
        return x, y

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate,
    )
    return train_loader, val_loader, vocab


# -----------------------------------------------------------------------------
# Model definition


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        moe_kwargs: Dict,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.moe = MoEFeedForward(hidden_dim=dim, **moe_kwargs)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tuple[Tensor, Dict[str, torch.Tensor]]:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)

        h = self.ln2(x)
        moe_out, stats = self.moe(h)
        x = x + self.dropout(moe_out)
        return x, stats


class TinyMoEModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        dim: int,
        layers: int,
        heads: int,
        dropout: float,
        moe_kwargs: Dict,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, dropout, moe_kwargs) for _ in range(layers)]
        )
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor], Dict[str, torch.Tensor]]:
        b, t = idx.shape
        x = self.embed(idx) + self.pos_embed[:, :t, :]

        # causal mask
        mask = torch.full((t, t), float("-inf"), device=idx.device)
        mask = torch.triu(mask, diagonal=1)

        stats_list = []
        for block in self.blocks:
            x, stats = block(x, attn_mask=mask)
            stats_list.append(stats)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))

        agg_stats = aggregate_stats(stats_list)
        return logits, loss, agg_stats


def aggregate_stats(stats_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not stats_list:
        return {}
    result: Dict[str, torch.Tensor] = {}
    device = next(iter(stats_list[0].values())).device
    keys = set().union(*[s.keys() for s in stats_list])
    for key in keys:
        vals = [s[key] for s in stats_list if key in s]
        if not vals:
            continue
        if key == "aux_loss":
            result[key] = torch.stack(vals).sum()
        elif key == "expert_counts":
            result[key] = torch.stack(vals)
        else:
            result[key] = torch.stack(vals).mean()
    if "aux_loss" not in result:
        result["aux_loss"] = torch.tensor(0.0, device=device)
    return result


# -----------------------------------------------------------------------------
# Training utilities


def eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    tokens = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            _, loss, _ = model(x, y)
            total_loss += float(loss.item()) * x.numel()
            tokens += x.numel()
    avg_loss = total_loss / max(1, tokens)
    ppl = math.exp(min(20, avg_loss))
    bpc = avg_loss / math.log(2)
    return {"val_loss": avg_loss, "ppl": ppl, "bpc": bpc}


# -----------------------------------------------------------------------------
# Main training loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal MoE training script")
    parser.add_argument("--data", type=str, default=None, help="Path to training text file")
    parser.add_argument("--outdir", type=str, default="runs/tiny")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None, help="Torch device (e.g., cuda, cuda:0, cpu)")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--strategy", type=str, default="softk", choices=["top1", "topk_hard", "softk"])
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--renorm-after-drop", action="store_true")
    parser.add_argument("--router", type=str, default="torch_soft", choices=["torch_soft", "quack"])
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--load-balance-alpha", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def cosine_decay(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.device is None:
        target = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target = args.device
    device = torch.device(target)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    train_loader, val_loader, vocab = build_dataloaders(args.data, args.seq_len, args.batch_size)

    moe_kwargs = dict(
        num_experts=args.num_experts,
        top_k=args.top_k,
        router_name=args.router,
        strategy=args.strategy,
        capacity_factor=args.capacity_factor,
        renorm_after_drop=args.renorm_after_drop,
        ffn_mult=args.ffn_mult,
        load_balance_alpha=args.load_balance_alpha,
    )

    model = TinyMoEModel(
        vocab_size=vocab.size,
        seq_len=args.seq_len,
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        moe_kwargs=moe_kwargs,
    ).to(device)

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore
        except Exception as exc:
            print(f"[warn] torch.compile failed: {exc}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dtype = args.dtype.lower()
    autocast_dtype = None
    if dtype == "fp16":
        autocast_dtype = torch.float16
    elif dtype == "bf16":
        autocast_dtype = torch.bfloat16
    use_autocast = autocast_dtype is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=autocast_dtype == torch.float16)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "train_log.jsonl"
    best_ppl = float("inf")
    tokens_seen = 0
    step = 0

    t0 = time.perf_counter()
    for epoch in range(10 ** 6):  # run until step reaches max_steps
        for batch in train_loader:
            step += 1
            model.train()
            lr = cosine_decay(step, args.warmup_steps, args.max_steps, args.lr)
            for param_group in optim.param_groups:
                param_group["lr"] = lr

            x, y = batch
            x = x.to(device)
            y = y.to(device)
            tokens_seen += x.numel()

            with torch.autocast(device.type, dtype=autocast_dtype, enabled=use_autocast):
                logits, ce_loss, stats = model(x, y)
                loss = ce_loss + stats.get("aux_loss", torch.tensor(0.0, device=device))

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            if step % args.eval_interval == 0 or step == args.max_steps:
                torch.cuda.synchronize() if device.type == "cuda" else None
                eval_stats = eval_model(model, val_loader, device)
                step_ms = (time.perf_counter() - t0) * 1000.0 / args.eval_interval
                t0 = time.perf_counter()
                tps = (args.batch_size * args.seq_len * args.eval_interval) / (step_ms / 1000.0)

                row = {
                    "step": step,
                    "lr": lr,
                    "train_loss": float(ce_loss.item()),
                    "val_loss": eval_stats["val_loss"],
                    "ppl": eval_stats["ppl"],
                    "bpc": eval_stats["bpc"],
                    "avg_step_ms": step_ms,
                    "tokens_per_s": tps,
                    "drop_rate": float(stats.get("drop_rate", torch.tensor(0.0)).item()),
                    "token_drop_rate": float(stats.get("token_drop_rate", torch.tensor(0.0)).item()),
                    "load_cv": float(stats.get("load_cv", torch.tensor(0.0)).item()),
                    "used_capacity": float(stats.get("used_capacity", torch.tensor(0.0)).item()),
                    "aux_loss": float(stats.get("aux_loss", torch.tensor(0.0)).item()),
                    "gate_entropy": float(stats.get("gate_entropy", torch.tensor(0.0)).item()),
                    "num_experts": args.num_experts,
                    "top_k": args.top_k,
                    "strategy": args.strategy,
                    "capacity_factor": args.capacity_factor,
                    "tokens_seen": tokens_seen,
                }
                print(json.dumps(row, ensure_ascii=False))
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")

                if eval_stats["ppl"] < best_ppl:
                    best_ppl = eval_stats["ppl"]
                    ckpt_path = outdir / "best.pt"
                    torch.save({
                        "model": model.state_dict(),
                        "vocab_size": vocab.size,
                        "vocab_itos": vocab.itos,
                        "config": vars(args),
                        "step": step,
                        "ppl": best_ppl,
                    }, ckpt_path)

            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    final = {"done": True, "steps": step, "tokens_seen": tokens_seen}
    print(json.dumps(final, ensure_ascii=False))


if __name__ == "__main__":
    main()
