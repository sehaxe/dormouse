"""Locate the op that aborts a CUDA-graph capture of `SparseMoE.forward`.

Prints the last N dispatched aten ops with the Python frame that issued them, so
the abort names one op and one `file:line` instead of "capture failed".
"""

from __future__ import annotations

import collections
import sys
import traceback

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from dormouse.config import load_run
from dormouse.model.experts import SparseMoE

LAST: collections.deque = collections.deque(maxlen=12)


class Log(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        frames = traceback.extract_stack()[:-1]
        site = ""
        for f in reversed(frames):
            if "site-packages" not in f.filename and "torch/" not in f.filename:
                site = f"{f.filename}:{f.lineno}"
                break
        LAST.append(f"{func}  @ {site}")
        return func(*args, **kwargs)


def main() -> int:
    run = load_run(
        "tiny",
        model_overrides=dict(bf16=True),
        train_overrides=dict(batch=8, seq_len=64),
    )
    cfg = run.model
    print(
        f"cfg: E={cfg.n_experts} k={cfg.moe_topk} cap={cfg.moe_capacity} "
        f"kernel={cfg.moe_kernel} gather={cfg.moe_gather} d={cfg.d_model} f={cfg.d_ffn}"
    )
    torch.manual_seed(0)
    moe = SparseMoE(cfg, device="cuda")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    x = torch.randn(n, cfg.d_model, device="cuda", dtype=torch.bfloat16)

    for _ in range(2):
        moe(x)
    torch.cuda.synchronize()
    print("eager forward OK")

    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    LAST.clear()
    try:
        with torch.cuda.graph(g), Log():
            moe(x)
    except Exception as e:  # noqa: BLE001 - the probe's whole job
        torch.cuda.synchronize()
        print(f"\nCAPTURE FAILED: {type(e).__name__}: {e}\n")
        print("last dispatched ops (oldest first):")
        for line in LAST:
            print("  ", line)
        return 1
    torch.cuda.synchronize()
    print("CAPTURE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
