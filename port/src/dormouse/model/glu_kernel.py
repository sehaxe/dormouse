"""The fused SwiGLU epilogue — `silu(gate)·up` without the `[tokens, 2·f]` write.

WHY IT EXISTS. The dense FFN's first GEMM produces two `[tokens, f]` halves
(`gate = x·Wgᵀ + bg`, `up = x·Wuᵀ`), then `F.silu(gate)·up` reads both and writes
one. In a forward pass that is three HBM round trips over `2·f` per token where
the model only ever needs `f`. One kernel that computes both dot products and
applies the epilogue in registers turns that into one write of `[tokens, f]`.

WHERE IT IS WIRED. `model/ffn.py::DenseFFN` — the `ffn_form="dense"` slot
(`config.FFN_FORMS`). That is the only form in the port whose FFN is a plain
`[tokens, 2·f]` fc1 (`gate` + `up`); the `bank` form's GLU lives at the latent
rank inside `DualLatentGLU`, and `tsct` factors the same latents, so neither has
an `f`-wide intermediate to remove. `config.validate_ffn` refuses the flag on
those forms and on any activation other than `silu`, LOUDLY, naming the escape.

THE LIVE MODEL DOES NOT RUN THIS. `configs/flagship3_dense.toml` is
`block_form="transformer"`, and that block (`model/transformer.py`) is the
control's GELU FFN with one `d → f` fc1 — there is no `silu(g)·u` in it. The
live run's FFN is therefore untouched by this module and by the flag; the same
is true of the default `ffn_form="bank"`. The shape this file is measured at
(`scripts/bench_glu.py`: 24·1024 tokens, d=832, f=3072, bf16) is the DENSE FFN's
shape at the live recipe's batch, not a shape the live checkpoint runs.

**MEASURED 2026-11-09** (RTX 5070 Ti, median of 20 warm windows, the bench
above):

| arm | ms | peak GiB | max rel vs eager |
|---|---:|---:|---:|
| eager `F.silu(F.linear)·F.linear` | 4.04 | 0.69 | — |
| `_swiglu_kernel` | 3.09 | 0.41 | 7.6e-3 |

**1.31x** on the forward and 40% less peak: the `[24576, 6144]` bf16
intermediate is 302 MB and is never allocated. The tile sweep in the same bench
found `(64,64,32,4,3)` fastest; `(128,64,64,8,3)` measured 3.21 ms and the
128-column `(64,128,64,8,2)` tied eager (4.08 ms). fp32 is NOT a win — 13.6 →
25.8 ms, the `ieee` dot gets no tensor cores — so [`glu_reason`] REFUSES fp32;
only bf16/fp16 are served. The model-level bf16 delta at this shape is max
|Δ logits| 2.3e-2 and |Δ loss| 3.3e-5 nats. The gate's bound is
`max_rel <= 1e-2`: one bf16 ULP at |h|≈1 is 7.8e-3, both arms accumulate in fp32
and round once, so it is the dtype envelope, not a fitted number.

The training caveat is the verdict's whole frame: this is a FORWARD win. A
grad-enabled forward falls back (below), and the live block is GELU, so neither
today's training step nor today's eval reaches this file. It is a building block
for a dense-SwiGLU 300M-1B arm, where the FFN is the dominant GEMM.

FORWARD-ONLY, COUNTED. The kernel is an epilogue, not an autograd Function: a
training forward needs `gate` and `up` saved for `silu`'s backward anyway, so a
fused forward with a torch backward saves the same bytes it writes. A call with
grad enabled therefore falls back COUNTED (`glu_kernel_fallback` + the reason on
`counters.why`) to the eager `F.silu(gate)·up` path. That is the same stance
`tsct.mm_fp8` takes for its lane, and it is why setting `fused_glu=true` cannot
move a training number: the fallback is the eager expression, bit for bit.

Every fast arm has a COUNTED fallback with its reason (ADR-0011/0019): no
triton in the environment, a non-CUDA device, a non-2-D or non-contiguous
operand, an unsupported dtype, autograd, and the launch's own `OutOfResources`.

House style (`train/ns_kernels.py`): module-level `triton`/`tl` import guarded
so a CPU-only install still imports this file and can COUNT its fallback; module
constants for the tile set (an `autotune` inside a training loop re-tunes
whenever a shape key moves — the treadmill `moe_kernels.py` bans); no
module-level mutable state; no host sync (every bound is a shape the host has).
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F

#: The environment's ability to run the kernel, not a dependency of the port
#: (the dion/`ns_kernels` stub pattern): a CPU-only wheel has no triton, and the
#: arm must still be IMPORTABLE so it can count its own fallback.
try:
    import triton
    import triton.language as tl
    from triton.runtime.errors import OutOfResources

    TRITON_AVAILABLE = True
    #: What a launch can REFUSE on (a tile set whose shared memory is past the
    #: device's limit). Raised on the HOST, before any driver call, so catching
    #: it is a COUNTED fallback and not a poisoned CUDA context (`ns_kernels`
    #: measured this on the same card).
    LAUNCH_REFUSALS: tuple[type[BaseException], ...] = (OutOfResources,)
except ImportError:  # pragma: no cover - the CPU-only wheel (no `gpu` extra)
    triton = types.ModuleType("triton")  # type: ignore[assignment]
    triton.jit = lambda fn: fn  # type: ignore[attr-defined]
    triton.cdiv = lambda a, b: -(-a // b)  # type: ignore[attr-defined]
    triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()  # type: ignore[attr-defined]
    tl = types.ModuleType("triton.language")  # type: ignore[assignment]
    tl.constexpr = int  # type: ignore[attr-defined]
    TRITON_AVAILABLE = False
    LAUNCH_REFUSALS = ()

#: The shape `scripts/bench_glu.py` measures, kept here so the tile table, the
#: bench and the GPU parity gate cannot drift: 24·1024 tokens (the live recipe's
#: batch × seq), the live d=832 and d_ffn=3072.
BENCH_SHAPE = (24 * 1024, 832, 3072)

#: The shipped tile set, keyed `(M, N=f, K=d)`,
#: `(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)`.
#:
#: The kernel holds TWO `[BLOCK_M, BLOCK_N]` fp32 accumulators and THREE staged
#: operands per K step (`x`, `Wg`, `Wu`), so its shared-memory need is
#: `stages·(BM·BK + 2·BN·BK)·2` bytes at bf16 — 96 KiB at
#: `(128, 64, 64, 8, 3)`, under this card's 101376 B ceiling, where a
#: symmetric `128×128` tile with two weights would not fit at any useful stage
#: count. Picked by `scripts/bench_glu.py` at the shape below; re-measure and
#: edit this table, do not add an `autotune` decorator (house rule).
GLU_TILES: dict[tuple[int, int, int], tuple[int, int, int, int, int]] = {
    (24 * 1024, 3072, 832): (64, 64, 32, 4, 3),
}
#: Any other shape takes the conservative tile: it fits, and a shape this table
#: has no measurement for should not silently borrow the live one's tuning.
DEFAULT_TILE = (64, 64, 32, 4, 3)


def tile_for(m: int, n: int, k: int) -> tuple[int, int, int, int, int]:
    """The measured tile for this `(M, N, K)`, else [`DEFAULT_TILE`]."""
    return GLU_TILES.get((int(m), int(n), int(k)), DEFAULT_TILE)


def _bias_ptr(bias: torch.Tensor | None, fallback: torch.Tensor) -> torch.Tensor:
    """A pointer the kernel never loads unless `HAS_BIAS` (Triton needs one)."""
    return fallback if bias is None else bias


@triton.jit
def _swiglu_kernel(
    X_ptr,
    WG_ptr,
    WU_ptr,
    B_ptr,
    O_ptr,
    M,
    N,
    K,
    s_xm,
    s_xk,
    s_wn,
    s_wk,
    s_om,
    s_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program per `(M-block, f-block)`: `silu(x·Wgᵀ + b)·(x·Wuᵀ)` → fp32 acc.

    `Wg`/`Wu` are `[N, K]` (torch's `nn.Linear` layout); the tile is loaded as
    `[BLOCK_K, BLOCK_N]` by swapping the two stride arguments, so the dot is the
    ordinary `A [BM, BK] · B [BK, BN]`. Both accumulators stay in registers and
    the epilogue is applied to them before the ONLY store: the `[M, 2N]`
    intermediate an eager pair of GEMMs would have written never exists.

    `input_precision="ieee"` is unconditional: it is ignored for bf16/fp16
    operands (tensor cores are still used), and for fp32 it is the house rule —
    a parity gate must not measure TF32 (`moe_kernels._prec`).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    x_ptrs = X_ptr + rm[:, None] * s_xm + rk[None, :] * s_xk
    wg_ptrs = WG_ptr + rk[:, None] * s_wk + rn[None, :] * s_wn
    wu_ptrs = WU_ptr + rk[:, None] * s_wk + rn[None, :] * s_wn
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    m_mask = rm < M
    n_mask = rn < N
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        kk = k0 * BLOCK_K + rk
        k_mask = kk < K
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        wg = tl.load(wg_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        wu = tl.load(wu_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc_g = tl.dot(x, wg, acc_g, input_precision="ieee")
        acc_u = tl.dot(x, wu, acc_u, input_precision="ieee")
        x_ptrs += BLOCK_K * s_xk
        wg_ptrs += BLOCK_K * s_wk
        wu_ptrs += BLOCK_K * s_wk
    if HAS_BIAS:
        acc_g += tl.load(B_ptr + rn, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    # `silu(g) = g·sigmoid(g)`, in fp32 registers, then ONE store.
    h = acc_g * tl.sigmoid(acc_g) * acc_u
    o_ptrs = O_ptr + rm[:, None] * s_om + rn[None, :] * s_on
    tl.store(o_ptrs, h.to(O_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


def glu_reason(
    x: object, w_gate: object, w_up: object, dt: torch.dtype | None = None
) -> str | None:
    """`None` if the fused epilogue can serve THIS call, else why not.

    The one place the arm decides, so `DenseFFN`, `fused_swiglu_or_eager` and the
    tests agree. Every branch is COUNTED by the caller, never silent (ADR-0019).
    The operand properties come before the device: a CPU test can reach the
    dtype/shape reasons, and a CUDA call with a bad shape is told the shape
    reason rather than "not CUDA" (`tsct._fp8_reason`'s order).
    """
    if not TRITON_AVAILABLE:
        return "no triton in this environment (the kernel is an ability, not a dependency)"
    if (
        not isinstance(x, torch.Tensor)
        or not isinstance(w_gate, torch.Tensor)
        or not isinstance(w_up, torch.Tensor)
    ):
        return "glu takes three tensors (x, gate weight, up weight)"
    if x.dim() != 2:
        return f"the fused epilogue is the 2-D FFN, got x {tuple(x.shape)}"
    if w_gate.dim() != 2 or w_gate.shape != w_up.shape:
        return f"gate/up must be the same [f, d], got {tuple(w_gate.shape)} / {tuple(w_up.shape)}"
    if x.shape[-1] != w_gate.shape[-1]:
        return f"the operands do not contract: x {tuple(x.shape)} vs W {tuple(w_gate.shape)}"
    d = x.dtype if dt is None else dt
    if d not in (torch.float32, torch.bfloat16, torch.float16):
        return f"dtype {d} has no kernel (fp32/bf16/fp16 in, the same out)"
    if d == torch.float32:
        return (
            "fp32 is the ieee-dot lane and measured 1.9x SLOWER than the eager "
            "cuBLAS pair at the bench shape (scripts/bench_glu.py); only bf16/fp16 "
            "are served"
        )
    if not x.is_cuda or not w_gate.is_cuda or not w_up.is_cuda:
        return f"not CUDA (the kernel is the CUDA arm; x on {x.device})"
    for name, t in (("x", x), ("gate", w_gate), ("up", w_up)):
        if not t.is_contiguous():
            return f"non-contiguous {name} (a copy would be the layout we removed)"
    if torch.is_grad_enabled() and (x.requires_grad or w_gate.requires_grad or w_up.requires_grad):
        return (
            "autograd is on (the fused epilogue is forward-only: a training step must "
            "save gate and up for silu's backward anyway, so a torch backward would "
            "keep every byte the fusion removes)"
        )
    return None


def swiglu_eager(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    dt: torch.dtype | None = None,
) -> torch.Tensor:
    """The reference this kernel must reproduce: `F.silu(F.linear)·F.linear`.

    The casts are `DenseFFN._linear`'s (x, weight and bias to the compute dtype,
    then `F.linear`), so with `dt` equal to the operands' dtype this is the eager
    expression BIT FOR BIT — that is the fallback's contract (`tests/test_glu_kernel.py`).
    """
    d = x.dtype if dt is None else dt
    xg = x if x.dtype is d else x.to(d)
    wg = w_gate if w_gate.dtype is d else w_gate.to(d)
    wu = w_up if w_up.dtype is d else w_up.to(d)
    bg = None if bias is None else (bias if bias.dtype is d else bias.to(d))
    return F.silu(F.linear(xg, wg, bg)) * F.linear(xg, wu)


def fused_swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    dt: torch.dtype | None = None,
    tile: tuple[int, int, int, int, int] | None = None,
) -> torch.Tensor:
    """The kernel only: `[M, d]·([f, d], [f, d])` → `[M, f]` in the compute dtype.

    `tile` is the caliper: `None` takes [`tile_for`], an explicit value is used
    verbatim so a bench can sweep or a test can force the device's refusal
    through the COUNTED catch. Every precondition lives in [`glu_reason`]; this
    function asserts them rather than silently computing something else.
    """
    d = x.dtype if dt is None else dt
    xg = x if x.dtype is d else x.to(d)
    wg = w_gate if w_gate.dtype is d else w_gate.to(d)
    wu = w_up if w_up.dtype is d else w_up.to(d)
    bg = None if bias is None else (bias if bias.dtype is d else bias.to(d))
    m, k = xg.shape
    n = wg.shape[0]
    bm, bn, bk, warps, stages = tile if tile is not None else tile_for(m, n, k)
    out = torch.empty((m, n), dtype=d, device=xg.device)
    _swiglu_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        xg,
        wg,
        wu,
        _bias_ptr(bg, wg),
        out,
        m,
        n,
        k,
        xg.stride(0),
        xg.stride(1),
        wg.stride(0),
        wg.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=bg is not None,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def fused_swiglu_or_eager(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    dt: torch.dtype | None = None,
    counters=None,
) -> torch.Tensor:
    """The fused epilogue, else the eager `F.silu(g)·u` COUNTED.

    `counters.count("glu_kernel")` when the kernel ran,
    `counters.count("glu_kernel_fallback")` + the reason on `counters.why` when
    it did not. The reason is REQUIRED: a fallback a reader of the log cannot see
    is the defect ADR-0019 exists to prevent, so a call with no counters is LOUD
    rather than silently uncounted (`ns_kernels.zeropower_triton_or_classic`).
    """
    from ..instrument import loud

    if counters is None:
        loud(
            "the fused GLU was called with no Counters: the arm's claim is that it "
            "can SHOW which path ran (`glu_kernel` / `glu_kernel_fallback` + reason), "
            "and an uncounted call cannot. Escape: pass a `Counters` (the model's "
            "forward seam always has one)."
        )
    why = glu_reason(x, w_gate, w_up, dt)
    if why is None:
        try:
            out = fused_swiglu(x, w_gate, w_up, bias, dt=dt)
        except LAUNCH_REFUSALS as e:  # pragma: no cover - needs a device to fire
            why = f"the launch was refused by the device ({type(e).__name__}: {e})"
        else:
            counters.count("glu_kernel")
            return out
    counters.count("glu_kernel_fallback")
    note = f"glu_kernel: {why}"
    prev = getattr(counters, "why", None)
    counters.why = f"{prev}; {note}" if prev else note  # type: ignore[attr-defined]
    return swiglu_eager(x, w_gate, w_up, bias, dt=dt)
