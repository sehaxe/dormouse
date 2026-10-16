"""Speed gates for the MSA arm's aggregation route (`sparse/msa_kernels.py`).

The correctness anchor is `tests/test_msa.py` (7 cases, all routes). THIS file
is the other half of the same claim — that the fused route is the SAME
computation, and that the route a call took is never a secret:

1. **route dispatch, any device** — `MSA_KERNEL` picks the route, an unknown
   literal is LOUD, and a route that cannot run is COUNTED with its reason
   (`msa_triton_fallback` + `counters.why`), never silent (ADR-0019);
2. **the gather plan** — `_pos_pack`'s -1 is a column that cannot be attended,
   and it is not key 0 (the fold that would silently make it one);
3. **parity, CUDA** — the fused route equals the torch route equals the naive
   per-query loop, forward (<=1e-5) and in all three gradients (<=1e-5), at the
   `small` preset's own attention shape and at a truncated one;
4. **the memory policy** — `MSA_CKPT_BYTES` keeps the b8 s512 shape
   checkpointed and lets a small shape skip the recompute, and which one ran is
   on the counter.

The bench at the bottom is NOT a test: this module's `__main__` drives it —
`uv run --extra gpu python tests/test_msa_speed.py <shape|torch|tiles|compile|mem|all>`
(machine discipline first: no `train` process, one GPU thing at a time).
"""

from __future__ import annotations

import statistics
import time

import pytest
import torch
from conftest import tiny_cfg

from dormouse.instrument import Counters, LoudError
from dormouse.model.sparse import FlexSparseSelector
from dormouse.model.sparse import attention as msa_attention
from dormouse.model.sparse.attention import (
    MSA_CKPT_BYTES,
    MSA_KERNEL,
    MSA_SLOT_CHUNK,
    _check_route,
    _ckpt_needed,
    _pos_pack,
    attend_selected_blocks,
)

try:  # the fused gather is an ABILITY: without triton the module is not importable
    from dormouse.model.sparse import msa_kernels
except ImportError:  # pragma: no cover - the CPU-only wheel
    msa_kernels = None  # type: ignore[assignment]

#: The CUDA gates. NOT a module-wide mark: the route dispatch, the plan and the
#: memory policy are testable on any device, and a gate that is skipped
#: everywhere it could fire is the defect ADR-0019 exists to prevent.
CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or msa_kernels is None,
    reason="needs a CUDA device and triton (the fused gather is the CUDA arm)",
)

DEV = "cuda"
HAS_TRITON = msa_kernels is not None


def _arm_io(b: int, t: int, h: int, hd: int, block: int, topk: int, seed: int, device: str):
    """`(q,k,v,blocks,keep,scale)` with the SELECTOR's own block plan."""
    torch.manual_seed(seed)
    cfg = tiny_cfg(d_model=h * hd, n_heads=h, head_dim=hd, msa_block=block, msa_topk=topk)
    sel = FlexSparseSelector(cfg, device=device)
    x = torch.randn(b, t, h * hd, device=device)
    blocks, keep = sel(x)
    q, k, v = (torch.randn(b, t, h, hd, device=device) for _ in range(3))
    return q, k, v, blocks, keep, 1.0 / (hd**0.5)


def _naive_selected_attention(q, k, v, blocks, keep, block, scale):
    """The longhand per-query reference (`test_msa.py`'s own, verbatim)."""
    b, t, h, hd = q.shape
    r = block
    out = torch.zeros_like(q)
    for bi in range(b):
        for i in range(t):
            keys: list[int] = []
            for s in range(blocks.shape[-1]):
                if not bool(keep[bi, i, s]):
                    continue
                start = int(blocks[bi, i, s]) * r
                keys += [p for p in range(start, start + r) if p < t and p <= i]
            assert keys, f"query {i} kept a block but has no valid key"
            ks, vs = k[bi, keys], v[bi, keys]
            sc = torch.einsum("hd,phd->hp", q[bi, i], ks) * scale
            out[bi, i] = torch.einsum("hp,phd->hd", torch.softmax(sc, dim=-1), vs)
    return out


def _torch_route(q, k, v, blocks, keep, scale, counters=None):
    """`attend_selected_blocks` with the fused route forced off."""
    saved = msa_attention.MSA_KERNEL
    msa_attention.MSA_KERNEL = "torch"
    try:
        return attend_selected_blocks(q, k, v, blocks, keep, 4, scale, counters)
    finally:
        msa_attention.MSA_KERNEL = saved


# --------------------------------------------------------------------------
# route dispatch — any device
# --------------------------------------------------------------------------


def test_route_literals_are_checked_loudly():
    assert MSA_KERNEL == "auto", "the fused route is the default"
    assert MSA_CKPT_BYTES > 0 and MSA_SLOT_CHUNK >= 1
    for name in ("auto", "torch", "triton"):
        assert _check_route(name) == name
    with pytest.raises(LoudError, match="is not a route"):
        _check_route("triton2")


def test_a_route_that_cannot_run_is_counted_with_its_reason(monkeypatch):
    """The whole point of the fallback chain: a reader of the log can tell."""
    q, k, v, blocks, keep, scale = _arm_io(1, 64, 4, 16, 4, 8, seed=11, device="cpu")
    c = Counters()
    attend_selected_blocks(q, k, v, blocks, keep, 4, scale, c)
    # A CPU input is never the fused arm's business, with triton installed or not.
    assert c["msa_triton_fallback"] == 1 and "msa_triton" not in c
    assert "cpu" in c.why or "triton is not installed" in c.why

    # `MSA_KERNEL="torch"` is the explicit request: no route counter at all.
    monkeypatch.setattr(msa_attention, "MSA_KERNEL", "torch")
    c = Counters()
    attend_selected_blocks(q, k, v, blocks, keep, 4, scale, c)
    assert "msa_triton" not in c and "msa_triton_fallback" not in c


@CUDA
def test_a_shape_the_kernel_cannot_tile_falls_back_counted():
    """head_dim 48 is real, is not a power of two, and is not silent."""
    q, k, v, blocks, keep, scale = _arm_io(1, 64, 4, 48, 4, 8, seed=12, device=DEV)
    c = Counters()
    out = attend_selected_blocks(q, k, v, blocks, keep, 4, scale, c)
    assert c["msa_triton_fallback"] == 1 and "head_dim=48" in c.why
    ref = _naive_selected_attention(q.cpu(), k.cpu(), v.cpu(), blocks.cpu(), keep.cpu(), 4, scale)
    assert (out.cpu() - ref).abs().max().item() < 1e-5


# --------------------------------------------------------------------------
# the gather plan and the memory policy — any device
# --------------------------------------------------------------------------


def test_pos_pack_marks_invalid_columns_with_minus_one():
    safe = torch.tensor([[[0, 1, 2, 3]]])
    valid = torch.tensor([[[True, False, True, False]]])
    pos = _pos_pack(safe, valid)
    assert pos.dtype == torch.int32 and pos.shape == safe.shape
    assert pos.tolist() == [[[0, -1, 2, -1]]]
    assert (pos == -1).tolist() == (~valid).tolist()
    # The fold that would be wrong: `safe * valid` sends a masked column to key
    # 0, which is a REAL key and would be attended.
    assert (safe * valid)[0, 0, 1].item() == 0


def test_ckpt_policy_bounds_the_retained_gather():
    # The measured OOM shape (b8 s512, `small`: h 12, hd 64, r 4, K 32) keeps
    # its region; the tiny test shape does not.
    assert _ckpt_needed(8, 512, 32, 4, 12, 64, 4), "b8 s512 is the measured OOM shape"
    assert _ckpt_needed(16, 512, 32, 4, 12, 64, 4)
    assert not _ckpt_needed(1, 128, 32, 4, 4, 16, 4)
    # The boundary is the whole-m gather in bytes, not a coincidence: the b*t
    # product that crosses it at unit m/r/h/hd is one element over.
    at = MSA_CKPT_BYTES // 4
    assert _ckpt_needed(1, at + 1, 1, 1, 1, 1, 4)
    assert not _ckpt_needed(1, at, 1, 1, 1, 1, 4)
    # A retained-set bound, not taste: 2 tensors x 4 loop iterations x
    # MSA_CKPT_BYTES stays well under the measured 12.9 GiB OOM.
    assert 2 * 4 * MSA_CKPT_BYTES < 12.9 * (1 << 30)


# --------------------------------------------------------------------------
# parity — CUDA
# --------------------------------------------------------------------------


@CUDA
@pytest.mark.parametrize("shape", [(2, 512, 12, 64), (2, 34, 4, 16)])
def test_fused_gather_matches_the_torch_route_and_the_naive_loop(shape):
    b, t, h, hd = shape
    q, k, v, blocks, keep, scale = _arm_io(b, t, h, hd, 4, 32, seed=13, device=DEV)
    assert not keep.all(), "the truncated/sentinel path must be in this gate"
    c = Counters()
    got = attend_selected_blocks(q, k, v, blocks, keep, 4, scale, c)
    assert c["msa_triton"] == 1, "the fused route must be the one that ran"
    torch_route = _torch_route(q, k, v, blocks, keep, scale)
    naive = _naive_selected_attention(q, k, v, blocks, keep, 4, scale)
    for name, r in (("torch", torch_route), ("naive", naive)):
        err = (got - r).abs().max().item()
        assert err < 1e-5, f"fused vs {name}: |d| = {err:.2e}"


@CUDA
@pytest.mark.parametrize("shape", [(2, 512, 12, 64), (2, 128, 4, 16)])
def test_fused_gather_gradients_match_the_torch_route(shape):
    """The Triton adjoint (with the gather's scatter-add) against torch autograd.

    The FORWARD can be right while the backward is not — the first version of
    this kernel saved the running `p` (normalized by that chunk's max) and read
    it as the softmax, which is a per-chunk `exp(m_chunk - m_final)` error and
    cost 11x on `dq`. A forward-only gate is blind to exactly that, so every
    gradient is compared, at the `small` preset's own attention shape too.
    """
    b, t, h, hd = shape
    _, k0, v0, blocks, keep, scale = _arm_io(b, t, h, hd, 4, 32, seed=14, device=DEV)

    def grads(fn):
        torch.manual_seed(15)
        q = torch.randn(b, t, h, hd, device=DEV, requires_grad=True)
        k = k0.clone().requires_grad_(True)
        v = v0.clone().requires_grad_(True)
        torch.manual_seed(16)
        g = torch.randn(b, t, h, hd, device=DEV)
        fn(q, k, v, blocks, keep, scale).mul(g).sum().backward()
        return q.grad, k.grad, v.grad

    got = grads(
        lambda q, k, v, b_, kp, sc: attend_selected_blocks(q, k, v, b_, kp, 4, sc, Counters())
    )
    ref = grads(lambda q, k, v, b_, kp, sc: _torch_route(q, k, v, b_, kp, sc))
    for name, g, e in zip(("q", "k", "v"), got, ref, strict=True):
        err = (g - e).abs().max().item() / max(e.abs().max().item(), 1e-12)
        assert err < 1e-5, f"grad {name}: rel = {err:.2e}"


@CUDA
def test_fused_gather_bf16_matches_the_fp32_route():
    """`--bf16` reaches this seam: the kernel accumulates fp32 and rounds once.

    The bound is bf16's own (2^-8), not the fp32 gate's: what is being checked is
    that the dtype casts in the kernel are wired (a bf16 softmax, or an fp32
    result written into a bf16 buffer without a cast, would land far outside).
    """
    b, t, h, hd = 2, 128, 4, 16
    q, k, v, blocks, keep, scale = _arm_io(b, t, h, hd, 4, 32, seed=23, device=DEV)
    q16, k16, v16 = q.bfloat16(), k.bfloat16(), v.bfloat16()
    got = attend_selected_blocks(q16, k16, v16, blocks, keep, 4, scale, Counters())
    assert got.dtype == torch.bfloat16
    ref = _torch_route(q16.float(), k16.float(), v16.float(), blocks, keep, scale)
    err = (got.float() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 2e-2, f"bf16 fused vs fp32 torch: rel = {err:.2e}"
    # And the gradients flow through the fp32 accumulators to bf16 inputs.
    qg = q16.clone().requires_grad_(True)
    attend_selected_blocks(qg, k16, v16, blocks, keep, 4, scale, Counters()).sum().backward()
    assert qg.grad is not None and qg.grad.dtype == torch.bfloat16 and torch.isfinite(qg.grad).all()


# --------------------------------------------------------------------------
# bench (NOT a test) — `python tests/test_msa_speed.py <mode>`
# --------------------------------------------------------------------------


def _run(fn):
    out = fn()
    out.sum().backward()
    return out


def _time(fn, iters: int = 5, warm: int = 2) -> tuple[float, float]:
    """`(median ms, peak GiB)` of one fwd+bwd pass of `fn`."""
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warm):
        _run(fn)
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _run(fn)
        torch.cuda.synchronize()
        ts.append(1e3 * (time.perf_counter() - t0))
    return statistics.median(ts), torch.cuda.max_memory_allocated() / (1 << 30)


def _route_call(which, q, kt, v, blocks, keep, r, scale):
    """One fwd+bwd of `attend_selected_blocks` through the named route."""

    def run():
        saved = msa_attention.MSA_KERNEL
        msa_attention.MSA_KERNEL = which
        try:
            qg = q.clone().requires_grad_(True)
            return attend_selected_blocks(qg, kt, v, blocks, keep, r, scale, Counters())
        finally:
            msa_attention.MSA_KERNEL = saved

    return run


def bench_shape(b: int = 8, t: int = 512, h: int = 12, hd: int = 64, k: int = 32, r: int = 4):
    """The `small` preset's own attention shape: fwd+bwd of each route."""
    q, kt, v, blocks, keep, scale = _arm_io(b, t, h, hd, r, k, seed=17, device=DEV)
    for which in ("torch", "auto"):
        try:
            ms, gib = _time(_route_call(which, q, kt, v, blocks, keep, r, scale))
            print(
                f"b{b} s{t} h{h} hd{hd} K{k}: {which:5s} fwd+bwd {ms:8.1f} ms  peak {gib:5.2f} GiB"
            )
        except torch.OutOfMemoryError as exc:  # noqa: PERF203 - a bench reports
            print(f"b{b} s{t} h{h} hd{hd} K{k}: {which:5s} OOM ({str(exc)[:60]})")


SHAPE = dict(b=8, t=512, h=12, hd=64, k=32, r=4)


def _shape_io(seed: int, b: int | None = None, t: int | None = None):
    return _arm_io(
        SHAPE["b"] if b is None else b,
        SHAPE["t"] if t is None else t,
        SHAPE["h"],
        SHAPE["hd"],
        SHAPE["r"],
        SHAPE["k"],
        seed,
        DEV,
    )


def _preflight() -> None:
    """One heavy thing at a time: refuse to bench next to a train process.

    The pattern is `dormouse trai[n]` because the port's CLI is an ARGUMENT of
    the venv python (`pgrep -ax train` matches nothing here), and the bracket
    keeps pgrep from matching this very command line.
    """
    import subprocess

    ps = subprocess.run(["pgrep", "-af", "dormouse trai[n]"], capture_output=True, text=True)
    assert not ps.stdout.strip(), f"a train process is running:\n{ps.stdout}"
    assert torch.cuda.is_available(), "the bench needs the CUDA device"
    free = torch.cuda.mem_get_info()[0] / (1 << 30)
    assert free > 12, f"only {free:.1f} GiB free on the card; another lane is using it"


def bench_chunk():
    """MSA_SLOT_CHUNK sweep at the `small` preset's own shape (torch route)."""
    print("\n== MSA_SLOT_CHUNK sweep (torch route, region policy as shipped) ==")
    for chunk in (2, 4, 8, 16, 32):
        msa_attention.MSA_SLOT_CHUNK = chunk
        q, k, v, blocks, keep, scale = _shape_io(18)
        try:
            ms, gib = _time(_route_call("torch", q, k, v, blocks, keep, SHAPE["r"], scale))
            print(f"  chunk {chunk:2d}: {ms:8.1f} ms fwd+bwd  peak {gib:5.2f} GiB")
        except torch.OutOfMemoryError:
            print(f"  chunk {chunk:2d}: OOM")
        del q, k, v, blocks, keep
        torch.cuda.empty_cache()


def bench_policy():
    """The conditional checkpoint: is the region ever a win at these shapes?"""
    print("\n== MSA_CKPT_BYTES policy (torch route) ==")
    for b, t in ((8, 512), (2, 128)):
        q, k, v, blocks, keep, scale = _shape_io(19, b=b, t=t)
        for name, budget in (("region always", 0), ("region never", 1 << 62)):
            msa_attention.MSA_CKPT_BYTES = budget
            try:
                ms, gib = _time(_route_call("torch", q, k, v, blocks, keep, SHAPE["r"], scale))
                print(f"  b{b} s{t} {name:13s}: {ms:8.1f} ms fwd+bwd  peak {gib:5.2f} GiB")
            except torch.OutOfMemoryError:
                print(f"  b{b} s{t} {name:13s}: OOM  <-- the policy exists for this")
        msa_attention.MSA_CKPT_BYTES = 256 << 20
        del q, k, v, blocks, keep
        torch.cuda.empty_cache()


def bench_compile():
    """`torch.compile` around the whole torch route (item 3 of the plan)."""
    print("\n== torch.compile of the torch route ==")
    q, k, v, blocks, keep, scale = _shape_io(20)
    saved = msa_attention.MSA_KERNEL
    msa_attention.MSA_KERNEL = "torch"

    def eager():
        qg = q.clone().requires_grad_(True)
        return msa_attention.attend_selected_blocks(qg, k, v, blocks, keep, SHAPE["r"], scale)

    compiled = torch.compile(eager, dynamic=False)
    try:
        ms, gib = _time(eager, iters=3)
        print(f"  eager:    {ms:8.1f} ms fwd+bwd  peak {gib:5.2f} GiB")
        compiled()  # the first call pays the compile; time the warm graph only
        ms, gib = _time(compiled, iters=3)
        print(f"  compiled: {ms:8.1f} ms fwd+bwd  peak {gib:5.2f} GiB")
    except Exception as exc:  # noqa: BLE001 - a bench reports what it saw
        print(f"  compiled: FAILED ({type(exc).__name__}: {str(exc)[:120]})")
    finally:
        msa_attention.MSA_KERNEL = saved


def bench_tiles():
    """The fused kernel's own tiles at the same shape."""
    print("\n== fused gather tiles (BLOCK_T x BLOCK_C x warps) ==")
    q, k, v, blocks, keep, scale = _shape_io(21)
    best: tuple[float, dict] | None = None
    for bt, bc in ((8, 1), (8, 2), (16, 1), (16, 2), (32, 2), (32, 4)):
        for warps in (4, 8, 16):
            msa_kernels.TILES.update(BLOCK_T=bt, BLOCK_C=bc, NUM_WARPS=warps)
            try:
                ms, gib = _time(_route_call("auto", q, k, v, blocks, keep, SHAPE["r"], scale))
            except Exception as exc:  # noqa: BLE001 - a bench reports what it saw
                print(f"  T{bt:3d} C{bc:3d} w{warps}: FAILED {type(exc).__name__}: {str(exc)[:60]}")
                continue
            print(f"  T{bt:3d} C{bc:3d} w{warps}: {ms:8.1f} ms fwd+bwd  peak {gib:5.2f} GiB")
            if best is None or ms < best[0]:
                best = (ms, dict(BLOCK_T=bt, BLOCK_C=bc, NUM_WARPS=warps))
    print(f"  BEST: {best[1]} at {best[0]:.1f} ms")
    msa_kernels.TILES.update(BLOCK_T=16, BLOCK_C=8, NUM_WARPS=4)


def bench_mem():
    """The peaks the report quotes: the `small` head shape at b8/b16 s512 and
    the same heads at s1024 (a shape the `small` PRESET cannot run — its
    `max_seq_len` is 512 — so it is the kernel's number, not a preset's)."""
    print("\n== peak memory, shipped settings ==")
    for b, t in ((8, 512), (16, 512), (16, 1024)):
        q, k, v, blocks, keep, scale = _shape_io(22, b=b, t=t)
        for which in ("torch", "auto"):
            try:
                ms, gib = _time(
                    _route_call(which, q, k, v, blocks, keep, SHAPE["r"], scale), iters=3
                )
                print(
                    f"  b{b:2d} s{t:4d} h{SHAPE['h']} hd{SHAPE['hd']} {which:5s}: "
                    f"{ms:8.1f} ms fwd+bwd  peak {gib:5.2f} GiB"
                )
            except torch.OutOfMemoryError:
                print(f"  b{b:2d} s{t:4d} {which:5s}: OOM")
        del q, k, v, blocks, keep
        torch.cuda.empty_cache()


MODES = {
    "shape": (bench_shape,),
    "torch": (bench_chunk, bench_policy),
    "tiles": (bench_tiles,),
    "compile": (bench_compile,),
    "mem": (bench_mem,),
    "all": (bench_shape, bench_chunk, bench_policy, bench_compile, bench_tiles, bench_mem),
}

if __name__ == "__main__":
    import sys

    _preflight()
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    assert mode in MODES, f"unknown mode {mode!r}; known: {sorted(MODES)}"
    for _fn in MODES[mode]:
        _fn()
