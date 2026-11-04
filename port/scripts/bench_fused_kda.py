"""`fused_kda_tail` OFF vs ON on the card: parity, launches, ms — the evidence table.

    uv run --extra gpu python scripts/bench_fused_kda.py
    uv run --extra gpu python scripts/bench_fused_kda.py --config configs/m400.toml

The workload is ONE iteration's two chains at the m400b shape
(`d=1536, h=24, h_d=64, b8 x s1024`, fp32 projections, bf16 kernel outputs):

* **tail** — `q/k/v` cast + `logsigmoid(g)` + `sigmoid(beta)`: six eager passes
  vs one kernel. The census says the traffic is ~equal and five launches leave,
  so this arm is judged on launches, not bytes;
* **combine** — `y = (attn + branch)·attn_write + ffn·mlp_write`: five eager
  passes over [8192, 1536] fp32 (604 MB) vs one (201 MB). The census says
  **403 MB / 212 us at 1.9 TB/s** per iteration; the measured delta is the
  verdict on that arithmetic.

**Protocol.** Parity FIRST (A/B of the same cotangents — timing a different
function is the classic own goal), then `min of 5` synchronised fwd+bwd runs
with 3 warm. Launch counts are `torch.profiler`'s `cudaLaunchKernel` events.

**Preflight** (AGENTS §1.5: one GPU process at a time): a train process or
> 2 GiB already in use prints a SKIP banner and exits 0 — nothing measured, so
nothing to quote. This script was written while a multi-day run owned the card
and has NOT been run by its author.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time

import torch


def preflight(dev: str = "cuda") -> bool:
    """False (with a SKIP banner) when the card is busy or absent."""
    if not torch.cuda.is_available():
        print(f"SKIP: no CUDA device; the fused arm cannot be measured. ({dev})")
        return False
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out = ""
    trains = subprocess.run(
        ["pgrep", "-af", "dormouse .*trai[n]"], capture_output=True, text=True, check=False
    ).stdout.strip()
    if trains:
        print(f"SKIP: a training run holds the card ({trains}). AGENTS §1.5.")
        return False
    free, total = torch.cuda.mem_get_info(0)
    if (total - free) > 2 << 30:
        print(
            f"SKIP: {((total - free) / 2**30):.2f} GiB already in use on device 0 "
            f"({out}). One heavy thing at a time (AGENTS §1.5)."
        )
        return False
    return True


def timeit(fn, n: int = 5, warm: int = 3) -> tuple[float, float]:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts), statistics.median(ts)


def _count(fn) -> tuple[int, float]:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
    launches = sum(1 for e in prof.events() if e.name == "cudaLaunchKernel")
    busy = 0.0
    for evt in prof.events():
        if evt.device_type == torch.autograd.DeviceType.CPU:
            for k in evt.kernels:
                busy += float(k.duration)
    return launches, busy / 1e3


def _tensors(a, dev: str):
    b, t, d, h = a.batch, a.seq, a.d, a.h
    torch.manual_seed(a.seed)
    q, k, v = (torch.randn(b, t, d, device=dev, requires_grad=True) * 0.5 for _ in range(3))
    g = torch.randn(b, t, d, device=dev, requires_grad=True) * 2.0
    beta = torch.randn(b, t, h, device=dev, requires_grad=True) * 2.0
    attn = torch.randn(b, t, d, device=dev, requires_grad=True) * 0.5
    branch = torch.randn(b, t, d, device=dev, requires_grad=True) * 0.5
    ffn = torch.randn(b, t, d, device=dev, requires_grad=True) * 0.5
    g1 = (torch.randn(d, device=dev) * 0.1 + 1.0).requires_grad_(True)
    g3 = (torch.randn(d, device=dev) * 0.1 + 1.0).requires_grad_(True)
    cot_t = [torch.randn(b, t, d, device=dev) for _ in range(3)]
    cot_t += [torch.randn(b, t, d, device=dev), torch.randn(b, t, h, device=dev)]
    cot_y = torch.randn(b, t, d, device=dev)
    return (q, k, v, g, beta), (attn, branch, ffn, g1, g3), cot_t, cot_y


def _zero(*tensors) -> None:
    for t in tensors:
        t.grad = None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--d", type=int, default=1536)
    ap.add_argument("--h", type=int, default=24)
    ap.add_argument("--bw", type=float, default=1.9, help="TB/s for the census column")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if not preflight():
        return 0

    from dormouse.model import fused_kda as fk

    dev = "cuda"
    # --- parity FIRST: the measurements must be of the same function ---------
    rep = fk.parity(b=2, t=8, d=64, h=4, seed=args.seed, use_kernel=True)
    fk.assert_parity(rep)
    print("kernel parity (vs the eager lines):")
    for name, (d, m) in rep.items():
        if name != "scale":
            print(f"  {name:12s} max|Δ| {d:9.3e}   ref {m:9.3e}   rel {d / max(m, 1e-30):9.2e}")

    tp, cp, cot_t, cot_y = _tensors(args, dev)
    q, k, v, g, beta = tp
    attn, branch, ffn, g1, g3 = cp

    def tail_run(fused: bool):
        _zero(q, k, v, g, beta)
        if fused:
            outs = fk.tail(q, k, v, g, beta, torch.bfloat16)
        else:
            outs = (
                q.to(torch.bfloat16),
                k.to(torch.bfloat16),
                v.to(torch.bfloat16),
                torch.nn.functional.logsigmoid(g),
                torch.sigmoid(beta).to(torch.bfloat16),
            )
        (sum((o.float() * c).sum() for o, c in zip(outs, cot_t, strict=True))).backward()

    def combine_run(fused: bool):
        _zero(attn, branch, ffn, g1, g3)
        if fused:
            y = fk.combine(attn, branch, ffn, g1, g3)
        else:
            y = (attn * g1 + branch * g1) + ffn * g3
        (y * cot_y).sum().backward()

    def us(byte_saving: float) -> float:
        return byte_saving * 1e6 / (args.bw * 1e12) * 1e6

    for name, run, saved_mb in (
        ("tail", tail_run, 1.6),  # the double read of beta: no byte win
        ("combine", combine_run, 402.6),
    ):
        e_min, e_med = timeit(lambda run=run: run(False))
        f_min, f_med = timeit(lambda run=run: run(True))
        le, be = _count(lambda run=run: run(False))
        lf, bf = _count(lambda run=run: run(True))
        print(
            f"\n{name:8s} {'arm':7s} {'ms min':>8s} {'ms med':>8s} {'launches':>9s} {'GPU ms':>8s}"
        )
        print(f"{name:8s} {'eager':7s} {e_min:8.3f} {e_med:8.3f} {le:9d} {be:8.3f}")
        print(f"{name:8s} {'fused':7s} {f_min:8.3f} {f_med:8.3f} {lf:9d} {bf:8.3f}")
        print(
            f"{name:8s} {'delta':7s} {e_min - f_min:+8.3f} {'':8s} "
            f"{le - lf:+9d} {be - bf:+8.3f}   census: -{saved_mb:.1f} MB -> "
            f"-{saved_mb * 1e6 / (args.bw * 1e12) * 1e6:.1f} us"
        )
    print(
        f"\n  predicted per iteration: -402.6 MB (combine), -1.6 MB (tail), "
        f"-9 launches, ~{us(404.2):.1f} us at {args.bw:.2f} TB/s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
