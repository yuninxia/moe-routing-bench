import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="requires >=2 CUDA devices")
def test_train_small_ddp_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    log_dir = tmp_path / "ddp_smoke"
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(repo_root / "scripts" / "train_small.py"),
        "--distributed",
        "--device",
        "cuda",
        "--max-steps",
        "4",
        "--eval-interval",
        "2",
        "--seq-len",
        "64",
        "--batch-size",
        "4",
        "--layers",
        "1",
        "--dim",
        "128",
        "--heads",
        "4",
        "--num-experts",
        "4",
        "--ffn-mult",
        "2",
        "--top-k",
        "2",
        "--capacity-factor",
        "1.0",
        "--grad-clip",
        "0.0",
        "--warmup-steps",
        "1",
        "--strategy",
        "softk",
        "--outdir",
        str(log_dir),
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    subprocess.run(cmd, check=True, cwd=repo_root, env=env)

    log_file = log_dir / "train_log.jsonl"
    assert log_file.exists(), "expected train log to be written"
    with log_file.open("r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert records, "expected at least one log record"
    steps = {rec["step"] for rec in records if "step" in rec}
    assert max(steps) == 4
