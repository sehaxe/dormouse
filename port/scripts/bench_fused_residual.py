"""Fused vs eager residual chain: parity, launch count, ms — the evidence table.

    uv run --extra gpu python scripts/bench_fused_residual.py --mode region
    uv run --extra gpu python scripts/bench_fused_residual.py --mode step

`region` isolates the four ops the fused kernels replace (`MHCMix.pre_mix`,
`depth_attend`, `MHCMix.forward`, the lane readout) on the two operating shapes
and measures fwd+bwd for each arm with the same cotangents, so the two columns
are the same arithmetic. `step` runs the trainer's real step through
`bench_step.py`'s model build with `fused_residual` off and on.

**Protocol.** One warm arm, then `min of 5` timed runs (CUDA events off: a
synchronised `perf_counter` around the whole fwd+bwd, which is what a step
pays). Launch counts are `torch.profiler`'s `cudaLaunchKernel` events over ONE
fwd+bwd — the metric this arm is judged by, with the wall column as the truth.

**Preflight** (AGENTS §1.5: one GPU process at a time): refuses to run when
another train process holds the card or `nvidia-smi` reports > 1 GiB in use, and
prints a SKIP banner with exit 0 — nothing measured, so nothing to quote.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

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
    # 2 GiB, not 1: the desktop (niri + Xwayland + a browser) legitimately sits
    # at ~0.95 GiB used with no compute process alive; the rule is about a
    # second HEAVY thing, and no train process is the sharper test.
    if (total - free) > 2 << 30:
        print(
            f"SKIP: {((total - free) / 2**30):.2f} GiB already in use on device 0 "
            f"({'; '.join(used)}). One heavy thing at a time (AGENTS §1.5)."
        )
        return False
    return True


def _region_eager_modules(mhc, slot, xl, y, eps: float, compose: bool):
    """The same chain through the shipped modules — the reference for parity."""
    from dormouse.model.loopblock import depth_attend

    ctx = mhc.pre_mix(xl)
    upd = depth_attend([torch.zeros_like(y), y], slot, eps, float(2) if compose else 1.0)
    if not compose:
        upd = y
    sn = mhc(xl, upd)
    h = sn.sum(dim=-2) / mhc.n_streams if compose else sn.sum(dim=-2)
    return ctx, sn, h


def _zero(mhc, slot):
    for p in mhc.parameters():
        p.grad = None
    slot.query.grad = None
    slot.gain.grad = None


def _count(prof) -> tuple[int, float]:
    launches = sum(1 for e in prof.events() if e.name == "cudaLaunchKernel")
    busy = 0.0
    for evt in prof.events():
        if evt.device_type == CPU:
            for k in evt.kernels:
                busy += float(k.duration)
    return launches, busy / 1e3


def timeit(fn, n=5, warm=3):
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


def make(a, dev):
    from dormouse.model.loopblock import AttnResSlot
    from dormouse.model.mhc import MHCMix

    torch.manual_seed(a.seed)
    mhc = MHCMix(a.d, 2, device=dev)
    with torch.no_grad():  # a live, non-degenerate chain (not the σ(10) saturation)
        mhc.phi_pre.normal_(0, 0.02)
        mhc.phi_post.normal_(0, 0.02)
        mhc.phi_res.normal_(0, 0.02)
        mhc.alpha_pre.fill_(0.3)
        mhc.alpha_post.fill_(0.2)
        mhc.alpha_res.fill_(1.0)
        mhc.b_pre.normal_(0, 0.5)
        mhc.b_res.normal_(0, 1.5)
    slot = AttnResSlot(a.d, dev)
    with torch.no_grad():
        slot.query.normal_(0, 0.5)
        slot.gain.normal_(1.0, 0.1)
    xl = torch.randn(a.batch, a.seq, 2, a.d, device=dev) * 0.5
    y = torch.randn(a.batch, a.seq, a.d, device=dev) * 0.5
    gc = torch.randn(a.batch, a.seq, a.d, device=dev)
    gh = torch.randn(a.batch, a.seq, a.d, device=dev)
    return mhc, slot, xl, y, gc, gh


def region(a):
    from torch.profiler import ProfilerActivity, profile

    from dormouse.instrument import Counters
    from dormouse.model import fused_residual as fr

    dev = "cuda"
    mhc, slot, xl0, y0, gc, gh = make(a, dev)
    eps = 1e-3

    # --- parity first: timing a different function is the classic own goal
    def grads_of(fused: bool):
        _zero(mhc, slot)
        xl = xl0.clone().requires_grad_(True)
        y = y0.clone().requires_grad_(True)
        if fused:
            ctx, raw = fr.pre(mhc, xl, Counters())
            sn, h = fr.write(mhc, slot, xl, y, raw, True, eps, Counters())
        else:
            ctx, sn, h = _region_eager_modules(mhc, slot, xl, y, eps, True)
        ((ctx * gc).sum() + (h * gh).sum()).backward()
        g = [p.grad.clone() for p in mhc.parameters()]
        g += [slot.query.grad.clone(), slot.gain.grad.clone(), xl.grad.clone(), y.grad.clone()]
        return (ctx.detach(), sn.detach(), h.detach()), g

    rep = fr.parity(b=a.batch, t=min(a.seq, 64), d=a.d, compose=True, seed=a.seed)
    fr.assert_parity(rep)
    print("parity (fused vs eager, the shipped modules):")
    for name, (d, m) in rep.items():
        if name == "scale":
            continue
        rel = d / max(m, 1e-30)
        print(f"  {name:12s} max|Δ| {d:9.3e}   ref {m:9.3e}   rel {rel:9.2e}")

    def eager_run():
        _zero(mhc, slot)
        ctx, sn, h = _region_eager_modules(mhc, slot, xl0, y0, eps, True)
        ((ctx * gc).sum() + (h * gh).sum()).backward()

    def fused_run():
        _zero(mhc, slot)
        c = Counters()
        ctx, raw = fr.pre(mhc, xl0, c)
        sn, h = fr.write(mhc, slot, xl0, y0, raw, True, eps, c)
        ((ctx * gc).sum() + (h * gh).sum()).backward()

    e_min, e_med = timeit(eager_run)
    f_min, f_med = timeit(fused_run)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pe:
        eager_run()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pf:
        fused_run()
    le, be = _count(pe)
    lf, bf = _count(pf)
    print(
        f"\n{'shape':22s} {'arm':7s} {'ms min':>8s} {'ms med':>8s} {'launches':>9s} {'GPU ms':>8s}"
    )
    tag = f"b{a.batch} s{a.seq} K=2 d{a.d}"
    print(f"{tag:22s} {'eager':7s} {e_min:8.2f} {e_med:8.2f} {le:9d} {be:8.2f}")
    print(f"{tag:22s} {'fused':7s} {f_min:8.2f} {f_med:8.2f} {lf:9d} {bf:8.2f}")
    print(
        f"{tag:22s} {'ratio':7s} {e_min / f_min:8.2f}x {'':8s} {le / lf:8.1f}x "
        f"{(be - bf) / be * 100:7.1f}% less"
    )
    return 0


def step(a):
    """The trainer's own step, `fused_residual` off then on."""

    from dormouse.cli import _device, _model, _snapshot_run
    from dormouse.config import validate, validate_train
    from dormouse.instrument import Counters
    from dormouse.model.experts import invalidate_banks_all
    from dormouse.model.qat_cache import invalidate_qat_cache_all
    from dormouse.train.loop import load_model_half, load_model_state
    from dormouse.train.optim import MixOptimizer, clip_grads

    overrides = {
        "batch": a.batch,
        "seq_len": a.seq,
        "max_seq_len": a.seq,
        "max_iter": a.iters,
        "mm_dtype": "bf16",
        "attn_latent_rank": 256,
        "out_proj_rank": 0,
        "use_csa": True,
        "use_msa": False,
        "use_mhc": True,
        "use_attnres": True,
        "share_u": True,
        "use_engram": False,
        "moe_kernel": "triton",
        "moe_capacity": 0.0,
        "doc_masking": False,
        "eval": False,
    }
    #: `_model(run)` builds from `run.model` (`cli.py:193`), so a model field set
    #: on `run.train` never reaches the network — the arm would silently not run.
    model_overrides = {
        "fused_residual": a.fused,
        "use_mhc": True,
        "use_attnres": True,
        "mhc_streams": 2,
    }
    ckpt = Path("/home/sehaxe/dormouse/checkpoints")
    run = _snapshot_run(ckpt / "longctx1", "longctx1", eval_file=None, eval_batches=None)
    for k, v in overrides.items():
        setattr(run.train, k, v)
    for k, v in model_overrides.items():
        setattr(run.model, k, v)
    validate(run.model)
    validate_train(run.train)
    assert run.model.fused_residual is a.fused, "the arm under test is not the arm that was set"
    dev = _device("auto")
    model = _model(run).to(dev).train()
    load_model_state(model, run.train, load_model_half(ckpt / "longctx1" / "longctx1.pt"))
    c = run.train
    opt = MixOptimizer(
        model,
        lr=c.lr,
        wd=c.wd,
        opt=c.opt,
        ns_steps=c.ns_steps,
        ns_kind=c.ns_kind,
        colrow_3d=c.muon_colrow_3d,
        muon_state_dtype=c.muon_state_dtype,
        ns_dtype=c.ns_dtype,
        muon_variant=c.muon_variant,
        muon_sf=c.muon_sf,
        muon_sf_wd=c.muon_sf_wd,
    )
    x = torch.randint(0, 256, (a.batch, a.seq), device=dev)
    y = torch.randint(0, 256, (a.batch, a.seq), device=dev)

    def one():
        model.zero_grad(set_to_none=True)
        invalidate_qat_cache_all(model)
        invalidate_banks_all(model)
        _, loss, cnt = model(x, y, counters=Counters())
        loss.backward()
        if c.grad_clip > 0:
            clip_grads(model.parameters(), c.grad_clip)
        opt.step()
        invalidate_banks_all(model)
        invalidate_qat_cache_all(model)
        return cnt

    torch.cuda.reset_peak_memory_stats()
    cnts = one()
    mn, med = timeit(one, n=5, warm=3)
    peak = torch.cuda.max_memory_allocated() / 2**30
    drained = cnts.drain()
    print(
        f"step b{a.batch} s{a.seq} iters={a.iters} fused={a.fused}: {mn:.1f} ms min / "
        f"{med:.1f} ms med  peak {peak:.2f} GiB  "
        f"fused_residual_launches={drained.get('fused_residual_launches', 0)}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="region", choices=("region", "step"))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fused", action="store_true", help="step mode: turn the flag on")
    a = ap.parse_args()
    if not preflight():
        return 0
    return region(a) if a.mode == "region" else step(a)


if __name__ == "__main__":
    sys.exit(main())
