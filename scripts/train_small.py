#!/usr/bin/env python3
"""Minimal end-to-end MoE training script (single GPU or DDP)."""
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
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

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


def build_dataloaders(
    path: Optional[str],
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
):
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

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        if distributed
        else None
    )
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        if distributed
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=not distributed,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, vocab, train_sampler, val_sampler


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


LOG_STAT_KEYS = [
    "drop_rate",
    "token_drop_rate",
    "load_cv",
    "used_capacity",
    "aux_loss",
    "gate_entropy",
]


def _infer_local_rank(default: int = 0) -> int:
    for key in ("LOCAL_RANK", "SLURM_LOCALID"):
        if key in os.environ:
            return int(os.environ[key])
    return default


def reduce_float(value: float, device: torch.device, distributed: bool, world_size: int) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return float(tensor.item())


def gather_log_stats(
    stats: Dict[str, torch.Tensor | float],
    device: torch.device,
    distributed: bool,
    world_size: int,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key in LOG_STAT_KEYS:
        val = stats.get(key)
        if isinstance(val, torch.Tensor):
            val = val.detach()
            if val.numel() > 1:
                val = val.mean()
            scalar = float(val.item())
        elif val is None:
            scalar = 0.0
        else:
            scalar = float(val)
        tensor = torch.tensor(scalar, device=device, dtype=torch.float64)
        if distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= world_size
        result[key] = float(tensor.item())
    return result


# -----------------------------------------------------------------------------
# Training utilities


def eval_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    distributed: bool = False,
    world_size: int = 1,
) -> Dict[str, float]:
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
    total_loss_t = torch.tensor(total_loss, device=device, dtype=torch.float64)
    tokens_t = torch.tensor(tokens, device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(total_loss_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(tokens_t, op=dist.ReduceOp.SUM)
    total_loss_val = float(total_loss_t.item())
    tokens_val = max(1.0, float(tokens_t.item()))
    avg_loss = total_loss_val / tokens_val
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
    parser.add_argument("--distributed", action="store_true", help="Enable DistributedDataParallel")
    parser.add_argument("--dist-backend", type=str, default="nccl", help="torch.distributed backend when using --distributed")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers per rank")
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

    distributed = args.distributed
    world_size = 1
    rank = 0
    local_rank = 0

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA")
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available")
        local_rank = _infer_local_rank()
        torch.cuda.set_device(local_rank)
        env_rank = int(os.environ.get("RANK", rank))
        env_world_size = int(os.environ.get("WORLD_SIZE", world_size))
        if not dist.is_initialized():
            init_kwargs = dict(
                backend=args.dist_backend,
                rank=env_rank,
                world_size=env_world_size,
            )
            if torch.cuda.is_available():
                init_kwargs["device_id"] = local_rank
            dist.init_process_group(**init_kwargs)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = torch.device("cuda", local_rank)
    else:
        if args.device is None:
            target = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            target = args.device
        device = torch.device(target)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")

    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    is_main = (rank == 0) if distributed else True
    pin_memory = device.type == "cuda"

    train_loader, val_loader, vocab, train_sampler, val_sampler = build_dataloaders(
        args.data,
        args.seq_len,
        args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        distributed=distributed,
        world_size=world_size,
        rank=rank,
    )

    if distributed:
        dist.barrier()

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
            if is_main:
                print(f"[warn] torch.compile failed: {exc}")

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dtype = args.dtype.lower()
    autocast_dtype = None
    if dtype == "fp16":
        autocast_dtype = torch.float16
    elif dtype == "bf16":
        autocast_dtype = torch.bfloat16
    use_autocast = autocast_dtype is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=autocast_dtype == torch.float16 and device.type == "cuda")

    outdir = Path(args.outdir)
    if is_main:
        outdir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    log_path = outdir / "train_log.jsonl"

    best_ppl = float("inf")
    tokens_seen = 0
    step = 0
    t0 = time.perf_counter()

    for epoch in range(10 ** 6):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            step += 1
            model.train()
            lr = cosine_decay(step, args.warmup_steps, args.max_steps, args.lr)
            for param_group in optim.param_groups:
                param_group["lr"] = lr

            x, y = batch
            x = x.to(device)
            y = y.to(device)

            tokens_this_batch = x.numel()
            tokens_tensor = torch.tensor(tokens_this_batch, device=device, dtype=torch.float64)
            if distributed:
                dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
            tokens_this_batch = int(tokens_tensor.item())
            tokens_seen += tokens_this_batch

            with torch.autocast(device.type, dtype=autocast_dtype, enabled=use_autocast):
                logits, ce_loss, stats = model(x, y)
                aux_loss = stats.get("aux_loss", torch.tensor(0.0, device=device))
                loss = ce_loss + aux_loss

            stats_detached = {
                k: (v.detach() if isinstance(v, torch.Tensor) else v)
                for k, v in stats.items()
            }

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            should_eval = (args.eval_interval > 0 and step % args.eval_interval == 0) or step == args.max_steps
            if should_eval:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if val_sampler is not None:
                    val_sampler.set_epoch(step)
                eval_stats = eval_model(
                    model,
                    val_loader,
                    device,
                    distributed=distributed,
                    world_size=world_size,
                )
                interval = max(1, args.eval_interval)
                step_ms = (time.perf_counter() - t0) * 1000.0 / interval
                t0 = time.perf_counter()
                tokens_per_s = tokens_this_batch * interval / max(1e-6, step_ms / 1000.0)

                train_loss_avg = reduce_float(float(ce_loss.item()), device, distributed, world_size)
                log_stats = gather_log_stats(stats_detached, device, distributed, world_size)

                row = {
                    "step": step,
                    "lr": lr,
                    "train_loss": train_loss_avg,
                    "val_loss": eval_stats["val_loss"],
                    "ppl": eval_stats["ppl"],
                    "bpc": eval_stats["bpc"],
                    "avg_step_ms": step_ms,
                    "tokens_per_s": tokens_per_s,
                    "drop_rate": log_stats["drop_rate"],
                    "token_drop_rate": log_stats["token_drop_rate"],
                    "load_cv": log_stats["load_cv"],
                    "used_capacity": log_stats["used_capacity"],
                    "aux_loss": log_stats["aux_loss"],
                    "gate_entropy": log_stats["gate_entropy"],
                    "num_experts": args.num_experts,
                    "top_k": args.top_k,
                    "strategy": args.strategy,
                    "capacity_factor": args.capacity_factor,
                    "tokens_seen": tokens_seen,
                }
                if is_main:
                    print(json.dumps(row, ensure_ascii=False))
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                if eval_stats["ppl"] < best_ppl:
                    best_ppl = eval_stats["ppl"]
                    if is_main:
                        state_dict = model.module.state_dict() if distributed else model.state_dict()
                        ckpt_path = outdir / "best.pt"
                        torch.save(
                            {
                                "model": state_dict,
                                "vocab_size": vocab.size,
                                "vocab_itos": vocab.itos,
                                "config": vars(args),
                                "step": step,
                                "ppl": best_ppl,
                            },
                            ckpt_path,
                        )

            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    final = {"done": True, "steps": step, "tokens_seen": tokens_seen}
    if distributed:
        dist.barrier()
    if is_main:
        print(json.dumps(final, ensure_ascii=False))
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
