"""Old vs fused: the per-iteration epilogue chain, the latent projection, CSA's scan.

    uv run --extra gpu python scripts/bench_epilogue.py
    uv run --extra gpu python scripts/bench_epilogue.py --shape b8s1024

Three rows, each the shape the model really runs at `medium`/`m400`
(d=1536, K=2 lanes, latent rank 256):

1. **read** — the body's pre-norm: the eager `MHCMix.pre_mix` (which computes
   `_x_norm` once) + the paper form's two `F.rms_norm` calls, against ONE
   `fused_residual.epilogue_read` pass. fwd+bwd, same cotangents.
2. **write** — the existing `fused_residual` chain (AttnRes score + Sinkhorn +
   `H_res`/`H_post` + readout), for the full boundary comparison; the row above
   is the part this task adds.
3. **latent** — four `F.linear` q/k/v/g projections on the shared latent against
   ONE GEMM over the concatenated weights; fwd+bwd, same cotangent.
4. **CSA fine selection** — the `q @ kᵀ → mask → topk → gather → softmax` scan
   of `csa.py::_level_scores`/`_select` at the same shape. MEASUREMENT ONLY: a
   sibling lane owns `csa.py`, so nothing here fuses or edits it.

**Protocol.** One warm arm, then `min of 5` (a synchronised `perf_counter`
around fwd+bwd, the same as `bench_fused_residual.py`). Launches are
`torch.profiler`'s `cudaLaunchKernel` events over ONE fwd+bwd. The fused arm's
own counters are drained once at the end and printed.

**Preflight** (AGENTS §1.5: one GPU process at a time): refuses to run when
another train process holds the card or > 2 GiB are already in use, and prints a
SKIP banner with exit 0 — nothing measured, so nothing to quote.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CPU = torch.autograd.DeviceType.CPU


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
    used = [ln for ln in out.splitlines() if ln.strip()]
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
            f"({'; '.join(used)}). One heavy thing at a time (AGENTS §1.5)."
        )
        return False
    return True


def _count(prof) -> tuple[int, float]:
    launches = sum(1 for e in prof.events() if e.name == "cudaLaunchKernel")
    busy = 0.0
    for evt in prof.events():
        if evt.device_type == CPU:
            for k in evt.kernels:
                busy += float(k.duration)
    return launches, busy / 1e3


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


def _zero(*mods) -> None:
    for m in mods:
        for p in m.parameters():
            p.grad = None


def read_arm(dev: str, b: int, s: int, d: int):
    """The body's pre-norm: eager modules vs the fused read kernel."""
    from dormouse.instrument import Counters
    from dormouse.model import fused_residual as fr
    from dormouse.model.mhc import MHCMix

    torch.manual_seed(0)
    mhc = MHCMix(d, 2, device=dev)
    gain = (torch.randn(d, device=dev) * 0.1 + 1.0).requires_grad_(True)
    gain2 = (torch.randn(d, device=dev) * 0.1 + 1.0).requires_grad_(True)
    xl = (torch.randn(b, s, 2, d, device=dev) * 0.5).requires_grad_(True)
    gc = torch.randn(b, s, d, device=dev)
    gn = torch.randn(b, s, d, device=dev) * 1e-2
    gn2 = torch.randn(b, s, d, device=dev) * 1e-2

    def loss(ctx, n1, n2):
        return (ctx * gc).sum() + (n1 * gn).sum() + (n2 * gn2).sum()

    def eager():
        _zero(mhc)
        gain.grad = gain2.grad = None
        xl.grad = None
        ctx = mhc.pre_mix(xl)
        n1 = F.rms_norm(ctx, (d,), gain, 1e-3)
        n2 = F.rms_norm(ctx, (d,), gain2, 1e-3)
        loss(ctx, n1, n2).backward()

    counters = Counters()

    def fused():
        _zero(mhc)
        gain.grad = gain2.grad = None
        xl.grad = None
        ctx, _raw, n1, n2 = fr.epilogue_read(mhc, xl, gain, gain2, 1e-3, counters)
        loss(ctx, n1, n2).backward()

    return eager, fused, counters


def write_arm(dev: str, b: int, s: int, d: int):
    """The existing fused write chain, for the full boundary row."""
    from dormouse.instrument import Counters
    from dormouse.model import fused_residual as fr
    from dormouse.model.loopblock import AttnResSlot, depth_attend
    from dormouse.model.mhc import MHCMix

    torch.manual_seed(0)
    mhc = MHCMix(d, 2, device=dev)
    slot = AttnResSlot(d, dev)
    with torch.no_grad():
        slot.query.normal_(0, 0.5)
        slot.gain.normal_(1.0, 0.1)
    xl = (torch.randn(b, s, 2, d, device=dev) * 0.5).requires_grad_(True)
    y = (torch.randn(b, s, d, device=dev) * 0.5).requires_grad_(True)
    gh = torch.randn(b, s, d, device=dev)

    def eager():
        _zero(mhc)
        slot.query.grad = slot.gain.grad = None
        xl.grad = y.grad = None
        ctx = mhc.pre_mix(xl)
        upd = depth_attend([torch.zeros_like(y), y], slot, 1e-3, 2.0)
        h = mhc(xl, upd).sum(dim=-2) / 2
        ((ctx.sum() + (h * gh).sum()) * 1e-3).backward()

    counters = Counters()

    def fused():
        _zero(mhc)
        slot.query.grad = slot.gain.grad = None
        xl.grad = y.grad = None
        ctx, raw = fr.pre(mhc, xl, counters)
        sn, h = fr.write(mhc, slot, xl, y, raw, True, 1e-3, counters)
        ((ctx.sum() + (h * gh).sum()) * 1e-3).backward()

    return eager, fused, counters


def boundary_arm(dev: str, b: int, s: int, d: int):
    """ONE iteration's boundary: read (norm chain) + write (AttnRes+mHC+readout).

    Eager is the shipped lines (`pre_mix` + two `F.rms_norm` + `depth_attend` +
    `MHCMix.forward` + the readout); fused is `epilogue_read` + `write` — the
    two launches the body sits between.
    """
    from dormouse.instrument import Counters
    from dormouse.model import fused_residual as fr
    from dormouse.model.loopblock import AttnResSlot, depth_attend
    from dormouse.model.mhc import MHCMix

    torch.manual_seed(0)
    mhc = MHCMix(d, 2, device=dev)
    slot = AttnResSlot(d, dev)
    with torch.no_grad():
        slot.query.normal_(0, 0.5)
        slot.gain.normal_(1.0, 0.1)
    gain = (torch.randn(d, device=dev) * 0.1 + 1.0).requires_grad_(True)
    gain2 = (torch.randn(d, device=dev) * 0.1 + 1.0).requires_grad_(True)
    xl = (torch.randn(b, s, 2, d, device=dev) * 0.5).requires_grad_(True)
    y = (torch.randn(b, s, d, device=dev) * 0.5).requires_grad_(True)
    gc = torch.randn(b, s, d, device=dev) * 1e-2
    gh = torch.randn(b, s, d, device=dev)

    def eager():
        _zero(mhc)
        slot.query.grad = slot.gain.grad = None
        gain.grad = gain2.grad = None
        xl.grad = y.grad = None
        ctx = mhc.pre_mix(xl)
        n1 = F.rms_norm(ctx, (d,), gain, 1e-3)
        n2 = F.rms_norm(ctx, (d,), gain2, 1e-3)
        upd = depth_attend([torch.zeros_like(y), y], slot, 1e-3, 2.0)
        h = mhc(xl, upd).sum(dim=-2) / 2
        ((ctx * gc).sum() + (n1 * gc).sum() + (n2 * gc).sum() + (h * gh).sum()).backward()

    counters = Counters()

    def fused():
        _zero(mhc)
        slot.query.grad = slot.gain.grad = None
        gain.grad = gain2.grad = None
        xl.grad = y.grad = None
        ctx, raw, n1, n2 = fr.epilogue_read(mhc, xl, gain, gain2, 1e-3, counters)
        sn, h = fr.write(mhc, slot, xl, y, raw, True, 1e-3, counters)
        ((ctx * gc).sum() + (n1 * gc).sum() + (n2 * gc).sum() + (h * gh).sum()).backward()

    return eager, fused, counters


def latent_arm(dev: str, b: int, s: int, d: int, r: int = 256):
    """Four q/k/v/g projections on the shared latent vs ONE concatenated GEMM.

    bf16, like the model's compute dtype (`part_bench.py`'s measured arm): fp32
    would put both sides on the slow MAGMA lane and measure the wrong thing.
    The `cat` of the weights is INSIDE the fused arm, because the production
    call rebuilds it per forward (a differentiable cat — the masters are leaves).
    """
    dt = torch.bfloat16
    torch.manual_seed(0)
    x = (torch.randn(b, s, r, device=dev) * 0.5).to(dt).requires_grad_(True)
    weights = [torch.randn(d, r, device=dev).to(dt).requires_grad_(True) for _ in range(4)]
    gw = [torch.randn(b, s, d, device=dev).to(dt) for _ in range(4)]

    def eager():
        x.grad = None
        for w in weights:
            w.grad = None
        outs = [F.linear(x, w) for w in weights]
        sum((o * g).sum() for o, g in zip(outs, gw, strict=True)).backward()

    def fused():
        x.grad = None
        for w in weights:
            w.grad = None
        wcat = torch.cat(weights, dim=0)
        out = F.linear(x, wcat)
        q, k, v, g = out.split(d, dim=-1)
        (q * gw[0]).sum().add_((k * gw[1]).sum()).add_((v * gw[2]).sum()).add_(
            (g * gw[3]).sum()
        ).backward()

    return eager, fused, None


def csa_arm(dev: str, b: int, s: int, kv: int = 64, chunk: int = 8, topk: int = 32):
    """The fine scan's SHAPE (`csa.py::_level_scores`/`_select`), no csa.py edit."""
    torch.manual_seed(0)
    q = torch.randn(b, s, kv, device=dev, dtype=torch.bfloat16)
    n = (s + chunk - 1) // chunk
    k = torch.randn(b, n, kv, device=dev, dtype=torch.bfloat16)
    v = torch.randn(b, n, kv, device=dev, dtype=torch.bfloat16)
    cols = torch.arange(n, device=dev).view(1, 1, -1)
    rows = torch.arange(s, device=dev).view(1, -1, 1)
    allowed = cols < (rows // chunk)  # strictly earlier chunks
    k_eff = min(topk, n)

    def select():
        scores = (q @ k.transpose(1, 2)) * (kv**-0.5)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        idx = scores.topk(k_eff, dim=-1).indices
        sel = scores.gather(2, idx)
        vv = v.unsqueeze(1).expand(b, s, n, kv).gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, kv))
        w = torch.softmax(sel, dim=-1)
        return (w.unsqueeze(-1) * vv).sum(-2)

    numel = b * s * n
    return select, numel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=1536, help="the m400 residual width")
    ap.add_argument(
        "--shape",
        action="append",
        default=None,
        help="b8s1024 and/or b1s8192 (default: both)",
    )
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not preflight():
        return 0
    dev = "cuda"
    torch.manual_seed(a.seed)
    shapes = a.shape or ["b8s1024", "b1s8192"]
    print(
        f"\n{'shape':10s} {'arm':22s} {'ms min':>8s} {'ms med':>8s} "
        f"{'launches':>9s} {'GPU ms':>8s}"
    )
    for tag in shapes:
        b, s = (8, 1024) if tag == "b8s1024" else (1, 8192)
        for name, factory in (
            ("boundary read+write", boundary_arm),
            ("read (norm chain)", read_arm),
            ("write (mHC+AttnRes)", write_arm),
            ("latent q/k/v/g", latent_arm),
        ):
            eager, fused, counters = factory(dev, b, s, a.d)
            for arm, fn in (("eager", eager), ("fused", fused)):
                ms, med = timeit(fn)
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    fn()
                launches, busy = _count(prof)
                print(
                    f"{tag:10s} {name + ' ' + arm:22s} {ms:8.3f} {med:8.3f} "
                    f"{launches:9d} {busy:8.3f}"
                )
            if counters is not None:
                drained = counters.drain()
                print(f"{'':10s} {'  counters':22s} {drained}")

        select, numel = csa_arm(dev, b, s)
        ms, med = timeit(select)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            select()
        launches, busy = _count(prof)
        print(
            f"{tag:10s} {'CSA fine select (shape)':22s} {ms:8.3f} {med:8.3f} "
            f"{launches:9d} {busy:8.3f}   [{numel / 1e6:.1f}M score elems]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
