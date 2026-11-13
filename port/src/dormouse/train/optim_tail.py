"""The Muon tail's fused passes — the non-Newton–Schulz half of the optimizer step.

WHAT THIS IS FOR
----------------
At the FAST rung (b8 x s1024, `ns_dtype=bf16`, `muon_state_dtype=bf16`) the
optimizer stage is 57 ms of a 145.8 ms step and Newton–Schulz is 27.3 ms of it.
The remaining ~30 ms is the tail: the momentum update, the ColRow
normalization, the `-lr*scale` multiply, the firewall's `*flag`, and the
write-back. Measured on the card (`docs/reviews/2026-11-08-optimizer-tail.md`),
none of it is arithmetic — it is PASSES OVER THE SAME 378 MiB fp32 update and
the same 189 MiB bf16 momentum, at 52 and 32 bytes per element respectively.
Both numbers are pure pass-counting a reader can re-derive from the source:

  tail, per element of the update (fp32), the shipped path:
    colrow: norm(dim=1) 4 | div_ 4+4 | norm(dim=2) 4 | div_ 4+4        = 24
    `.mul_(-lr*scale)`                                                 =  8
    `.mul_(flag)`                                                      =  8
    `_foreach_add_` (update + param -> param)                          = 12
                                                                      = 52 B
  and this module's path:
    col norm 4 | row norm 4 | one fused apply (read + write) 4+4        = 16
    `_foreach_add_` (unchanged)                                        = 12
                                                                      = 28 B
  momentum, per element of the bf16 momentum, the shipped path (the state rung
  with a MATERIALIZED fp32 temporary; the reference's gradients are fp32, so
  `_add_grads_` takes its fused `_foreach_add_` route and there is no stack):
    `.to(fp32)` 2+4 | `.mul_(mu)` 4+4 | `_foreach_add_(g, alpha)` 4+4+4 |
    `copy_` back 4+2                                                   = 32 B
  and this module's two paths:
    * EXACT (the default): one kernel for `to(fp32).mul_(mu)` (2+4) + the
      SHIPPED `_add_grads_` (12) + copy (6)                            = 24 B
    * FAST (opt-in): `stack` (4+4) + ONE kernel (2+4+2)                = 16 B

Measured on the card, per group `[12, 3072, 832]` (30.67M elements, CUDA-event
median of 20): the tail 2.274 -> 1.109 ms (**2.05x**), the momentum 3.74 ms at
the shipped 32 B/elem (810 GB/s) and the FAST arm 2.03x of that.

THE ARITHMETIC IS THE REFERENCE'S, OP FOR OP — the passes are merged, not the
formulas. `colrow_normalize` normalises columns of the ORIGINAL matrix and rows
of the COLUMN-DIVIDED matrix (that order is the definition: `x/(‖x‖_col+ε)` is
what the row norm is taken of), so the fused form keeps two reductions and one
apply, and the apply performs the SAME four roundings in the same order:

    y = ((x / (c + ε)) / (r + ε)) * scale * flag

with `div.rn` / `mul.rn` at each step, and `r` reduced from the ROUNDED
`x/(c+ε)` — the kernel recomputes the divided tile in registers instead of
writing it to global memory and reading it back. The only difference left is
the SUMMATION ORDER of the two reductions (a kernel's tree over a tile vs
`torch.linalg.vector_norm`'s), which is ~1e-7 relative on the divisor and under
the 1e-6 gate this module is tested to. It is named here rather than left for a
reader to discover.

THE MOMENTUM IS BIT-EXACT, and that is not a bonus: `tests/
test_optimizer_suite.py::test_muon_bf16_state_stores_one_rounded_fp32_accumula_
tion_per_step` pins `torch.equal` against `round(mu*m + (1-mu)*g)` rounded to
bf16 ONCE. A plain `m*mu + g*alpha` in a Triton kernel contracts to an `fma`,
which rounds once where the reference rounds twice, and flips the bf16 store on
~2^-16 of 94.6M elements — a drift no 64-element test could see. So the kernel
calls `libdevice.mul_rn` / `add_rn` explicitly: round `mu*m`, round `alpha*g`,
then add. That is exactly `mom.mul_(mu)` followed by `add_(grad, alpha)`.

WHY THIS IS A NEW MODULE AND NOT AN EDIT
----------------------------------------
`train/optim.py` is shared by every lane. The two drop-in replacements here
(`momentum_update_`, `colrow_normalize_`) have the SAME signatures as the
functions they replace, so the whole arm is `install()`, which swaps the module
attributes — and the call-site diff (one five-line branch, in
`docs/reviews/2026-11-08-optimizer-tail.md`) is the version that ships. Every
ineligible input falls back to the SHIPPED function with a counted reason
(ADR-0019), so a shape/dtype/device this module does not serve can only be
slower, never different.

    from dormouse.train import optim_tail
    optim_tail.install()      # the A/B arm; the shipped call sites are untouched

Counted names: `opt_tail_momentum`, `opt_tail_colrow`, `opt_tail_tail` when a
fused kernel ran; `opt_tail_fallback` + `Counters.why` when one did not.
"""

from __future__ import annotations

import sys
import types

import torch

try:  # the CPU-only wheel has no triton: the module must still IMPORT (so the
    # fallback is countable rather than an ImportError at the call site).
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    from triton.runtime.errors import OutOfResources

    TRITON_AVAILABLE = True
    LAUNCH_REFUSALS: tuple[type[BaseException], ...] = (OutOfResources,)
except ImportError:  # pragma: no cover - the CPU-only wheel
    triton = types.ModuleType("triton")  # type: ignore[assignment]
    triton.jit = lambda fn: fn  # type: ignore[attr-defined]
    tl = types.ModuleType("triton.language")  # type: ignore[assignment]
    tl.constexpr = int  # type: ignore[attr-defined]
    libdevice = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False
    LAUNCH_REFUSALS = ()

#: ColRow's ε, `optim.colrow_normalize`'s own default. Duplicated as a module
#: constant so the kernels can take it as a launch argument, never re-derived.
COLROW_EPS = 1e-7

#: Tile shapes. `BM x BN` per program on the two reductions and on the apply.
#: Chosen for the port's Muon shapes ([12, 3072, 832] and friends): BN=64 gives
#: 13 column programs per slice (156 total, ~2.2 waves on 70 SMs) and BM=64
#: gives 48 row programs per slice (576 total) — both far past one wave, which
#: is what a memory-bound elementwise pass needs. Not autotuned on purpose
#: (`ns_kernels.py`'s rule: an `@triton.autotune` in a training loop re-tunes
#: whenever a shape key moves).
BLOCK_M = 64
BLOCK_N = 64


#: Which rounding `add_(g, alpha)` itself uses. MEASURED, not assumed: at
#: `MOMENTUM_MODE = 0` the card gate (`tests/test_optim_tail.py::
#: test_the_momentum_kernel_is_BIT_equal_to_the_shipped_bf16_recurrence`) is
#: **RED** against `optim._momentum_update_`, so the shipped default is 1 —
#: `fma(alpha, g, a)`, which is what nvcc contracts `a + alpha * b` into. The
#: gate is the authority; a mode change without a green gate is a guess.
MOMENTUM_MODE = 1

#: One kernel (`stack + kernel`, 16 B/elem, measured **2.03x** against the
#: shipped 32 B/elem). ON, and it is ON because its gate is GREEN on the card:
#: `tests/test_optim_tail.py::
#: test_the_one_kernel_momentum_arm_is_BIT_equal_or_its_mode_is_wrong` asserts
#: `torch.equal` against `optim._momentum_update_` with `MOMENTUM_FAST = True`,
#: and it passes at `MOMENTUM_MODE = 1` (it fails at 0 — that red run is how the
#: mode was chosen). Set it to False to fall back to the other exact path below,
#: which replaces only `to(fp32) + mul_` and leaves the ADD to
#: `optim._add_grads_` (24 B/elem, 1.33x, bit-identical by construction).
MOMENTUM_FAST = True

#: Launch refusals, the same list `ns_kernels` catches. A tile set the device
#: cannot fit is a COUNTED degradation, not a traceback.
_MOMENTUM_NAME = "opt_tail_momentum"
_COLROW_NAME = "opt_tail_colrow"
_TAIL_NAME = "opt_tail_tail"
_FALLBACK_NAME = "opt_tail_fallback"


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _momentum_kernel(
    mom_ptr, grads_ptr, n_elem, mu, alpha, MODE: tl.constexpr, BLOCK: tl.constexpr
):
    """`mom = round(mu*mom) (+` `alpha*g)` in fp32, then ONE store.

    WHICH ROUNDING `add_(g, alpha)` ITSELF USES IS MEASURED, NOT ASSUMED
    (`tests/test_optim_tail.py`'s bit gate, `scripts/../mom_variants.py`):
    `MOMENTUM_MODE` 0 is `round(a + round(alpha*g))`, 1 is `fma(alpha, g, a)`
    (one rounding). Both move the same bytes, so the timing is the same and the
    choice is decided only by which one reproduces the shipped recurrence
    bit-for-bit on this backend.

    `libdevice.mul_rn`/`add_rn`/`fma_rn` instead of `a*b + c`: the shorthand
    contracts to an `fma`, so a mode that means two roundings must say so.
    """
    pid = tl.program_id(0).to(tl.int64)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n_elem
    m = tl.load(mom_ptr + off, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(grads_ptr + off, mask=mask, other=0.0).to(tl.float32)
    # The scalars are materialized as fp32 vectors: Triton's Python-float
    # arguments would otherwise reach libdevice at whatever width the binder
    # picks, and `mul_rn`/`add_rn` dispatch on that width (an fp64 pair would
    # round once in double and change the bf16 store).
    mu_b = tl.full([BLOCK], mu, tl.float32)
    alpha_b = tl.full([BLOCK], alpha, tl.float32)
    scaled = libdevice.mul_rn(m, mu_b)
    if MODE == 0:
        out = libdevice.add_rn(scaled, libdevice.mul_rn(g, alpha_b))
    else:
        out = libdevice.fma_rn(alpha_b, g, scaled)
    tl.store(mom_ptr + off, out.to(mom_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _mul_kernel(mom_ptr, out_ptr, n_elem, mu, BLOCK: tl.constexpr):
    """`out = fp32(mom) * mu` — `mom.to(torch.float32).mul_(mu)` as one kernel.

    No rounding question here: `mul_` is a single multiply and the store exists
    in both forms, so `mul_rn` on the widened value is the same number.
    """
    pid = tl.program_id(0).to(tl.int64)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n_elem
    m = tl.load(mom_ptr + off, mask=mask, other=0.0).to(tl.float32)
    mu_b = tl.full([BLOCK], mu, tl.float32)
    tl.store(out_ptr + off, libdevice.mul_rn(m, mu_b), mask=mask)


@triton.jit
def _col_sq_kernel(x_ptr, cs_ptr, M, N, nc, eps, BM: tl.constexpr, BN: tl.constexpr):
    """`cs[s, j] = ‖x[s, :, j]‖₂ + eps` — the NORM, not the squared norm.

    The ε is added to the root (`x / (‖x‖ + ε)`, `optim.colrow_normalize`), so
    the sqrt is taken here and the divisor is stored ready to use.
    """
    pid = tl.program_id(0)
    s = pid // nc
    jb = pid % nc
    j = jb * BN + tl.arange(0, BN)
    jm = j < N
    base = s.to(tl.int64) * M * N
    acc = tl.zeros((BN,), tl.float32)
    for i0 in range(0, M, BM):
        i = i0 + tl.arange(0, BM)
        m = (i < M)[:, None] & jm[None, :]
        t = tl.load(x_ptr + base + i[:, None] * N + j[None, :], mask=m, other=0.0)
        acc += tl.sum(t * t, axis=0)
    tl.store(cs_ptr + s * N + j, tl.sqrt(acc) + eps, mask=jm)


@triton.jit
def _row_sq_kernel(x_ptr, cs_ptr, rs_ptr, M, N, nr, eps, BM: tl.constexpr, BN: tl.constexpr):
    """`rs[s, i] = ‖x[s, i, :] / cs[s, :]‖₂ + eps` — reduced from the ROUNDED quotient.

    This is the one place the fusion cannot be a straight op count: the shipped
    path WRITES `x/cs` and then reduces it, so a kernel that reduced `x` and
    divided afterwards would take the norm of a different quantity. The tile is
    divided by `cs` in registers (same `div.rn`, same operands) and only then
    squared and summed — no global round trip, same values to the last bit.
    """
    pid = tl.program_id(0)
    s = pid // nr
    ib = pid % nr
    i = ib * BM + tl.arange(0, BM)
    im = i < M
    base = s.to(tl.int64) * M * N
    acc = tl.zeros((BM,), tl.float32)
    for j0 in range(0, N, BN):
        j = j0 + tl.arange(0, BN)
        jm = j < N
        m = im[:, None] & jm[None, :]
        t = tl.load(x_ptr + base + i[:, None] * N + j[None, :], mask=m, other=0.0)
        c = tl.load(cs_ptr + s * N + j, mask=jm, other=1.0)
        y = t / c[None, :]
        acc += tl.sum(y * y, axis=1)
    tl.store(rs_ptr + s * M + i, tl.sqrt(acc) + eps, mask=im)


@triton.jit
def _apply_kernel(
    x_ptr,
    cs_ptr,
    rs_ptr,
    scale_ptr,
    M,
    N,
    nc,
    HAS_SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """`x = ((x / cs) / rs) * scale` IN PLACE — the same four roundings, one pass.

    `scale` carries `-lr * row_scale * flag` already multiplied on the host or on
    a 0-dim device tensor (the firewall's flag), so the two `.mul_`s the shipped
    path needs as separate passes cost nothing here. `* 1.0` is exact, which is
    what keeps a healthy step identical to the reference.
    """
    pid = tl.program_id(0)
    s = pid // nc
    jb = pid % nc
    j = jb * BN + tl.arange(0, BN)
    jm = j < N
    base = s.to(tl.int64) * M * N
    c = tl.load(cs_ptr + s * N + j, mask=jm, other=1.0)
    scale = tl.load(scale_ptr) if HAS_SCALE else 1.0
    for i0 in range(0, M, BM):
        i = i0 + tl.arange(0, BM)
        mm = (i < M)[:, None] & jm[None, :]
        t = tl.load(x_ptr + base + i[:, None] * N + j[None, :], mask=mm, other=0.0)
        r = tl.load(rs_ptr + s * M + i, mask=i < M, other=1.0)
        y = (t / c[None, :]) / r[:, None]
        if HAS_SCALE:
            y = y * scale
        tl.store(x_ptr + base + i[:, None] * N + j[None, :], y, mask=mm)


# ---------------------------------------------------------------------------
# the shipped functions this module replaces (kept for the fallback)
# ---------------------------------------------------------------------------

_REF: dict[str, object] = {}


def _reference() -> dict[str, object]:
    """The SHIPPED functions, captured before `install()` can replace them."""
    if not _REF:
        from . import optim

        _REF["momentum"] = optim._momentum_update_
        _REF["colrow"] = optim.colrow_normalize_
        _REF["add"] = optim._foreach_add_mixed_
        _REF["stack"] = optim._stack_grads
        _REF["add_grads"] = optim._add_grads_
        _REF["chunk"] = optim.MUON_CHUNK_ELEMS
    return _REF


def _count(counters, name: str) -> None:
    if counters is not None:
        counters.count(name)


def _fallback(counters, reason: str) -> None:
    if counters is not None:
        counters.count(_FALLBACK_NAME)
        counters.why = f"opt_tail: {reason}"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 1. the momentum update, fused (bf16 state rung only: that is the rung with a
#    materialized fp32 temporary)
# ---------------------------------------------------------------------------


def momentum_path_reason(mom: torch.Tensor, params: list, chunk_elems: int) -> str | None:
    """`None` when the fused kernel serves this call, else the COUNTED reason."""
    if not TRITON_AVAILABLE:
        return "no triton (the fused pass is the CUDA arm)"
    if mom.dtype is not torch.bfloat16:
        return (
            f"state dtype {mom.dtype} has no fp32 temporary to remove: the fp32 rung's "
            f"cost is `mul_` + a same-dtype foreach add, which this kernel cannot beat "
            f"without a stacked gradient"
        )
    if not mom.is_cuda or not mom.is_contiguous():
        return f"device/contiguity ({mom.device}, contiguous={mom.is_contiguous()})"
    if not params or any(p.grad is None for p in params):
        return "a parameter in the chunk has no gradient"
    if any(not p.grad.is_contiguous() for p in params):
        return "a gradient is not contiguous"
    return None


def momentum_update_(
    mom: torch.Tensor,
    params: list,
    mu: float,
    chunk_elems: int | None = None,
    *,
    counters=None,
) -> None:
    """`_momentum_update_` with ONE kernel instead of five passes.

    Drop-in for `optim._momentum_update_`: same arguments, same in-place effect,
    same arithmetic (see the module docstring: `mul_rn` + `add_rn` reproduce
    `mom.mul_(mu)` + `add_(g, alpha)` to the bit, including the single bf16
    store the state rung owes).
    """
    ref = _reference()
    chunk = ref["chunk"] if chunk_elems is None else chunk_elems
    why = momentum_path_reason(mom, params, chunk)
    if why is not None:
        _fallback(counters, why)
        ref["momentum"](mom, params, mu, chunk)  # type: ignore[operator]
        return
    n = len(params)
    n_chunk = max(1, min(n, chunk // mom[0].numel()))
    alpha = 1.0 - mu
    BLOCK = 1024
    for start in range(0, n, n_chunk):
        chunk_params = params[start : start + n_chunk]
        block = mom[start : start + n_chunk]
        if MOMENTUM_FAST:
            grads = ref["stack"](chunk_params)  # type: ignore[operator]
            n_elem = block.numel()
            grid = (-(-n_elem // BLOCK),)
            # NO refusal catch here, deliberately: this is a flat 1024-element
            # elementwise tile with no shared memory and no rank limit, so there
            # is no input-dependent refusal to degrade from — and a catch that ran
            # the reference after some chunks had already been written would apply
            # those chunks TWICE. A launch error here is a bug, not a rung.
            _momentum_kernel[grid](block, grads, n_elem, mu, alpha, MOMENTUM_MODE, BLOCK)
            continue
        # EXACT path: the kernel IS `mom.to(fp32).mul_(mu)` and nothing else; the
        # add and the bf16 store are the SHIPPED function's own calls, so no
        # rounding can disagree.
        acc = torch.empty(block.shape, dtype=torch.float32, device=block.device)
        n_elem = block.numel()
        _mul_kernel[(-(-n_elem // BLOCK),)](block, acc, n_elem, mu, BLOCK)
        ref["add_grads"](acc, chunk_params, alpha)  # type: ignore[operator]
        block.copy_(acc)
    _count(counters, _MOMENTUM_NAME)


# ---------------------------------------------------------------------------
# 2. the ColRow normalization, fused (3 passes instead of 4)
# ---------------------------------------------------------------------------


def colrow_path_reason(x: torch.Tensor, colrow_3d: str) -> str | None:
    if not TRITON_AVAILABLE:
        return "no triton (the fused pass is the CUDA arm)"
    if colrow_3d != "colrow":
        return f"colrow_3d={colrow_3d!r} collapses dims (1,2); the kernels run the 3-D 'colrow' arm"
    if x.dtype is not torch.float32:
        return f"dtype {x.dtype} (the fusion is fp32: the update is fp32 in every rung)"
    if not x.is_cuda or not x.is_contiguous():
        return f"device/contiguity ({x.device}, contiguous={x.is_contiguous()})"
    if x.ndim == 2:
        return "a 2-D tensor takes the per-matrix arm (not the batched step's)"
    if x.ndim != 3:
        return f"ndim {x.ndim}"
    return None


def _normalize_(x: torch.Tensor, eps: float, scale, counters) -> None:
    s, m, n = x.shape
    nc = -(-n // BLOCK_N)
    nr = -(-m // BLOCK_M)
    cs = torch.empty((s, n), dtype=torch.float32, device=x.device)
    rs = torch.empty((s, m), dtype=torch.float32, device=x.device)
    _col_sq_kernel[(s * nc,)](x, cs, m, n, nc, eps, BLOCK_M, BLOCK_N)
    _row_sq_kernel[(s * nr,)](x, cs, rs, m, n, nr, eps, BLOCK_M, BLOCK_N)
    if scale is None:
        _apply_kernel[(s * nc,)](x, cs, rs, x, m, n, nc, False, BLOCK_M, BLOCK_N)
    else:
        assert scale.is_cuda and scale.numel() == 1, "scale must be one device scalar"
        _apply_kernel[(s * nc,)](x, cs, rs, scale, m, n, nc, True, BLOCK_M, BLOCK_N)


def colrow_normalize_(
    x: torch.Tensor,
    eps: float = COLROW_EPS,
    *,
    colrow_3d: str = "colrow",
    counters=None,
) -> torch.Tensor:
    """`optim.colrow_normalize_` as three passes instead of four.

    Drop-in: same signature, same in-place effect, same return (the tensor).
    """
    why = colrow_path_reason(x, colrow_3d)
    if why is not None:
        _fallback(counters, why)
        return _reference()["colrow"](x, eps, colrow_3d=colrow_3d)  # type: ignore[operator]
    try:
        _normalize_(x, eps, None, counters)
    except LAUNCH_REFUSALS as e:  # pragma: no cover - needs a device
        _fallback(counters, f"the launch was refused ({type(e).__name__}: {e})")
        return _reference()["colrow"](x, eps, colrow_3d=colrow_3d)  # type: ignore[operator]
    _count(counters, _COLROW_NAME)
    return x


# ---------------------------------------------------------------------------
# 3. the whole tail: ColRow + (-lr*scale) + the firewall flag + the write-back
# ---------------------------------------------------------------------------


def tail_path_reason(
    update: torch.Tensor,
    params: list,
    row,
    colrow_3d: str,
) -> str | None:
    why = colrow_path_reason(update, colrow_3d)
    if why is not None:
        return why
    if row is not None:
        return "TSCT-2's per-expert row scale is a separate pass (not fused here)"
    if any(p.dtype is not update.dtype for p in params):
        return "the update and the parameters differ in dtype (the write-back would promote)"
    return None


def muon_tail_(
    update: torch.Tensor,
    params: list,
    scale: float,
    *,
    row=None,
    flag: torch.Tensor | None = None,
    colrow_3d: str = "colrow",
    counters=None,
) -> None:
    """`p += ((colrow(update) * (-lr*scale)) * flag)` for every row, fused.

    THE CALL-SITE DIFF replaces the four statements in
    `Muon.batched_step_from_model`'s batched branch with this one call:

        colrow_normalize_(update, colrow_3d=self.colrow_3d).mul_(-lr * scale)
        if row is not None: update.mul_(row[:, None, None])
        if flag is not None: update.mul_(flag.to(update.dtype))
        with torch.no_grad(): _foreach_add_mixed_(params_c, update)

    `scale` is `-lr * scale` (the caller's, already a host float or a 0-dim
    device tensor); `flag` is multiplied INTO the scale — `flag * scale` as a
    0-dim device tensor, so a healthy step multiplies by exactly the scale and a
    skipped one by zero, without a separate 378 MiB pass.
    """
    why = tail_path_reason(update, params, row, colrow_3d)
    if why is not None:
        _fallback(counters, why)
        _reference_tail(update, params, scale, row, flag, colrow_3d)
        return
    if torch.is_tensor(scale):
        k = scale.to(update.dtype)
    else:
        k = torch.as_tensor(scale, dtype=update.dtype, device=update.device)
    if flag is not None:
        k = k * flag.to(update.dtype)
    try:
        _normalize_(update, COLROW_EPS, k, counters)
    except LAUNCH_REFUSALS as e:  # pragma: no cover - needs a device
        _fallback(counters, f"the launch was refused ({type(e).__name__}: {e})")
        _reference_tail(update, params, scale, row, flag, colrow_3d)
        return
    with torch.no_grad():
        _reference()["add"](params, update)  # type: ignore[operator]
    _count(counters, _TAIL_NAME)


def _reference_tail(update, params, scale, row, flag, colrow_3d) -> None:
    """The shipped four statements, verbatim — the fallback and the gate's oracle."""
    ref = _reference()
    ref["colrow"](update, COLROW_EPS, colrow_3d=colrow_3d)  # type: ignore[operator]
    update.mul_(scale)
    if row is not None:
        update.mul_(row[:, None, None])
    if flag is not None:
        update.mul_(flag.to(update.dtype))
    with torch.no_grad():
        ref["add"](params, update)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# the CPU-reference decomposition: the SAME algebra in torch ops, which is what
# `tests/test_optim_tail.py` gates the kernels against and what a CPU box runs
# ---------------------------------------------------------------------------


def colrow_reference_(x: torch.Tensor, eps: float = COLROW_EPS) -> torch.Tensor:
    """The two reductions + the apply, as torch ops (the kernel's spec)."""
    c = x.norm(dim=-2, keepdim=True) + eps
    y = x / c
    r = y.norm(dim=-1, keepdim=True) + eps
    x.copy_(y / r)
    return x


def momentum_reference_(mom: torch.Tensor, grads: torch.Tensor, mu: float) -> None:
    """`mom = round(mu*mom + (1-mu)*g)` in fp32, stored in `mom`'s dtype."""
    acc = mom.to(torch.float32) * mu
    acc = acc + grads.to(torch.float32) * (1.0 - mu)
    mom.copy_(acc)


# ---------------------------------------------------------------------------
# install: the A/B arm
# ---------------------------------------------------------------------------


def install(counters=None) -> None:
    """Swap the two drop-ins into `train.optim` (the arm `ab_opt.py` runs).

    The full tail (`muon_tail_`) has no drop-in signature — it replaces four
    statements — so the trainer's write-back is reached by the call-site diff.
    This function installs what CAN be installed without editing the file, and
    says which is which rather than letting a partial arm pass for the whole.
    """
    from . import optim

    _reference()
    optim._momentum_update_ = momentum_update_
    optim.colrow_normalize_ = colrow_normalize_
    print(
        "opt_tail: momentum (bf16 rung) + colrow are INSTALLED; the fused tail "
        "(scale/flag/write-back) needs the call-site diff and is NOT active",
        file=sys.stderr,
    )


def uninstall() -> None:
    from . import optim

    ref = _reference()
    optim._momentum_update_ = ref["momentum"]
    optim.colrow_normalize_ = ref["colrow"]


__all__ = [
    "COLROW_EPS",
    "TRITON_AVAILABLE",
    "colrow_normalize_",
    "colrow_path_reason",
    "colrow_reference_",
    "install",
    "momentum_path_reason",
    "momentum_reference_",
    "momentum_update_",
    "muon_tail_",
    "tail_path_reason",
    "uninstall",
]
