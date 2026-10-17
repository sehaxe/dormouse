"""The fused Triton Newton–Schulz chain — the `ns_kind="triton"` arm.

One kernel launch per shape-group chunk runs the WHOLE Jordan chain, where the
torch path (`optim.zeropower_via_ns`) issues ~53 dispatches per chain. That is
the entire point: this port's step is **host-dispatch-bound**, not FLOP-bound —
`docs/reviews/2026-10-16-b1-profile.md` §1/§3 measured a GPU residue of ≈0 ms
against a wall clock that is the enqueue, and `.bulba/memory.md:317` records
Gram-NS losing 1.2-1.4× here *because it handed back MORE calls*. Turbo-Muon's
finding (`docs/research/2026-10-17-muon-variants.md` §1 row 1, §2 #1) is the
other half of that idea: same polynomial, fewer launches.

**What is transcribed, and from where** (fetched 2026-10-17):

* `github.com/microsoft/dion`, `dion/newton_schulz_triton.py` @ `main` — the
  batched-NS Triton structure this file follows: one program per batch slice
  (`_pid_to_block`'s `batch_idx = pid // (num_pid_m * num_pid_n)`), the int64
  batch offset (`_batch_offset`: `batch_idx.to(tl.int64) * batch_stride` — an
  int32 offset overflows on a stacked `[1024, 64, 5461]` chunk), the three
  kernels' algebra (`ns_line_1_kernel` = `C = A @ Aᵀ` with the mirrored store,
  `ns_line_2_kernel` = `C = α·A@Aᵀ + β·A`, `ns_line_3_kernel` = `C = a·X + B@X`),
  and `input_precision = "ieee" if A.dtype == torch.float32 else "tf32"` — the
  house rule too (`moe_kernels._prec`: "fp32 must be IEEE, or the parity gate
  measures TF32, not this kernel").
* `github.com/thib-s/flash-newton-schulz` README @ `main` (Turbo-Muon,
  `2512.04632`, HAL `hal-05390446`) — read, and deliberately NOT transcribed:
  (a) its AOL rescaling `fast_inv_sqrt(reduce_sum(abs(W Wᵀ), axis=-1))` plus the
  four genetic-algorithm triples (`[4.6051, -9.6552, 5.6769] …`) are a
  DIFFERENT polynomial at a different iteration count, validated only together
  (its own table: 3.29 ± 0.001 at 3it vs Muon's 3.30 ± 0.002 at 3it); adopting
  them would be a math change wearing a speed arm's name, so this file keeps the
  port's Jordan triple and 5 iterations — the SAME semantics `zeropower_via_ns`
  already has. (b) Its kernel source ships through the HuggingFace `kernels`
  hub (`tboissin/newton_schulz_triton`), not the repo, so no line of it was
  readable here. (c) Nothing in its README claims a Hopper-only feature — the
  kernels are plain Triton — and nothing of theirs is transcribed either way.
  The Hopper/TMEM wall that kills Dion3's CuteDSL symmetric GEMMs on sm_120
  (`2026-10-17-muon-variants.md` §3) does NOT apply to plain Triton.

**What is OURS, and why it is not dion's three kernels.** dion round-trips the
`[r, r]` Gram and the `[r, d]` iterate through global memory three times per
iteration (three kernels per step, `X`/`A`/`B`/`C` buffers) because its shapes
are datacenter-scale (`[4096, 4096]`) where the Gram cannot live in registers.
This port's Muon factors are rank ≤ 64 (`MAX_RANK`), so the `[r, r]` Gram fits
in registers and the chain is ONE kernel: the Gram is formed and consumed
without ever reaching global memory, and `X` is streamed once per phase from the
caller's buffer. Same polynomial, same fp32 semantics, one launch.

**The cap, stated loudly.** `r = min(m, n) > MAX_RANK` falls back COUNTED to the
classic chain, with the reason on the counter. A `[2048, 2048]` matrix's Gram is
16 MB: it does not fit in registers, and the port's own economics already keep
`[d, d]` projections off Muon (`AGENTS.md` §2.3: "fp32 NS on `[768,768]` costs
~40 s/step"). Writing a second, global-memory kernel for a shape the router does
not send here would be an untested path for a case that does not occur.

House rules (`moe_kernels.py`'s, followed here): plain tensors in and out, no
module-level mutable state, no host sync (every bound is a SHAPE the host already
has), and **every fast arm has a COUNTED fallback with its reason** — `ns_triton`
when the fused chain ran, `ns_triton_fallback` + the reason when it could not
(`docs/research/2026-10-17-muon-variants.md` §2 #1's void check: "the Triton path
must be COUNTED, never a silent fallback to classic"). That includes a tile set
the DEVICE cannot fit: the b1 `(2048, 64)` shape at `BLOCK_D=128` needed 114688 B
of shared memory against this card's 101376 B and crashed at launch (measured
2026-10-17), which is a refusal to count, not a traceback to read.
"""

from __future__ import annotations

import types

import torch

#: The environment's ability to run the kernels, not a dependency of the port:
#: the CPU-only wheel has no triton, and the arm must still be IMPORTABLE so it
#: can COUNT its own fallback (the dion module's own stub pattern — its
#: `except ImportError` builds a stub `triton` so the file parses without it).
try:
    import triton
    import triton.language as tl
    from triton.runtime.errors import OutOfResources

    TRITON_AVAILABLE = True
    #: What a launch can REFUSE on (as opposed to get wrong): a tile set whose
    #: shared-memory need is past the device's limit. Triton raises this from
    #: `_init_handles`, i.e. on the HOST, before any driver call — so catching it
    #: is a COUNTED fallback, not a poisoned CUDA context (measured: the battery
    #: ran the crash and the same process kept launching the next shape).
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

from .optim import JORDAN_NS_COEFFS

#: The register cap on the small side. The `[r, r]` Gram is held in registers for
#: the whole chain, so `r²` is the binding number: 64² = 4096 fp32 values is
#: 16 registers per thread at 8 warps (32 at the shipped 4), either of which
#: leaves room for the `[r, BLOCK_D]` iterate tile. 128 would be 4× that and
#: spill. The port's Muon factors are rank 64 by construction (`model/tsct.py`),
#: so this cap is not a compromise on anything the router sends here — see the
#: module docstring.
MAX_RANK = 64

#: The streaming tile on the long side, and the warp count. Module constants on
#: purpose: a `@triton.autotune` inside a training loop re-tunes whenever a shape
#: key moves, which is the treadmill `moe_kernels.py`'s docstring bans (and dion's
#: own `_get_autotune_configs` sweeps 36 configs — "may take several minutes" on
#: first call, per its README). Re-measure with `scripts/bench_ns_triton.py` and
#: edit these, do not add a decorator.
#:
#: **64, not 128, and the reason is shared memory** (battery 2026-10-17). The
#: `[RT=64, BLOCK_D=128]` operands of `tl.dot(x, trans(x))` need **114688 B**
#: against this card's **101376 B** limit, so at 128 the b1 Muon group's
#: `(2048, 64)` shapes died at launch with `OutOfResources` — a CRASH, not a
#: counted fallback. At 64 every b1 shape launches with parity **1.7e-6 … 2.3e-6**
#: against the classic chain (fp32/ieee), and the b1 Muon group measures
#: **677.1 → 162.6 ms/step in the bench's `_median_ms` idiom** (which syncs only
#: after the loop, so the tail reps carry queue backpressure; a per-rep-synced
#: harness reads ~31 ms of true enqueue on BOTH arms) and **675.6 → 607.9 ms
#: synced** — the arm's whole claim. A tile set that still cannot fit is refused
#: COUNTED now (`zeropower_triton_or_classic`), never crashed.
BLOCK_D = 64
#: 4, not 8, and it is the same sweep: every b1 chain shape measured faster at 4
#: warps (`(2048, 64)` 1.210 vs 1.523 ms, `(64, 5461)` 5.780 vs 7.193,
#: `(513, 1024, 64)` 7.967 vs 13.005), and on the b1 Muon group the synced step
#: is 683.5 vs 768.7 ms (interleaved, 6 reps each) with the enqueue a tie. At
#: `r ≤ 64` the `[r, r]` Gram is 4096 values, so 8 warps leave most threads idle.
NUM_WARPS = 4

#: The prescale's ε, `zeropower_via_ns`'s own `x / (x.norm() + 1e-7)`.
NS_EPS = 1e-7


def _check(x: torch.Tensor) -> None:
    """The op asserts what it cannot serve — the caller counts, this refuses."""
    assert isinstance(x, torch.Tensor), f"NS takes a tensor, got {type(x).__name__}"
    assert x.ndim in (2, 3), f"NS is matrix-only or [N,d0,d1] batched, got shape {tuple(x.shape)}"
    assert x.dtype in (torch.float32, torch.bfloat16), (
        f"the kernel is fp32/bf16 in, fp32 out; got {x.dtype}"
    )
    r = min(x.shape[-2], x.shape[-1])
    assert 1 <= r <= MAX_RANK, (
        f"the [r, r] Gram must fit in registers: r={r} > MAX_RANK={MAX_RANK}. "
        f"The caller must fall back COUNTED (`triton_reason`)."
    )
    assert x.is_cuda and x.is_contiguous(), "the Triton arm wants a contiguous CUDA tensor"


def triton_reason(x: object) -> str | None:
    """`None` if the fused chain can serve THIS call, else why not.

    The one place the arm decides, so the op, the dispatcher and the tests agree.
    Every branch is COUNTED by the caller, never silent (ADR-0019).
    """
    if not TRITON_AVAILABLE:
        return "no triton in this environment (the kernels are an ability, not a dependency)"
    if not isinstance(x, torch.Tensor):
        return f"NS takes a tensor, got {type(x).__name__}"
    if x.ndim not in (2, 3):
        return f"NS is matrix-only or [N,d0,d1] batched, got shape {tuple(x.shape)}"
    if x.dtype not in (torch.float32, torch.bfloat16):
        return f"dtype {x.dtype} has no kernel (fp32/bf16 in, fp32 out)"
    r = min(x.shape[-2], x.shape[-1])
    if r > MAX_RANK:
        return (
            f"min dim {r} > MAX_RANK {MAX_RANK}: the [{r}, {r}] Gram would not fit in "
            f"registers, so the fused chain cannot serve this shape"
        )
    if not x.is_cuda:
        return f"not CUDA ({x.device}): the kernels are the CUDA arm"
    if not x.is_contiguous():
        return "non-contiguous input (the op's contract; a copy would be the launch we removed)"
    return None


@triton.jit
def _ns_chain_kernel(
    X_ptr,
    O_ptr,
    sxs,
    sos,
    sr,
    sd,
    R_,
    D_,
    A,
    B,
    C,
    EPS,
    STEPS: tl.constexpr,
    RT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """`X0 = X/(‖X‖_F + eps)`, then `X ← a·X + (b·R + c·R²)@X` `STEPS` times.

    One program per slice. `sr`/`sd` are the strides of the LOGICAL `[r, d]`
    orientation (`r = min`, `d = max`), so a tall input is handled by swapping
    two stride arguments instead of materializing a transpose — the transpose
    copy `zeropower_via_ns` pays is the same launch this kernel exists to remove.
    The caller's `X` is never written: `O` receives the prescaled iterate and the
    chain runs in place on `O`.

    Three phases over the long axis, and the Gram never leaves registers:

    1. the Frobenius norm of the slice (a masked sum of squares);
    2. `X0 = X / (‖X‖_F + eps)` written to `O` (also the copy that keeps `X` intact);
    3. per step, `R = X Xᵀ` accumulated over the `d` blocks, then
       `X ← a·X + (b·R + c·R²)@X` streamed back over the `d` blocks.

    Padded rows/columns (`r < RT`, `d` past the last block) load as exact zeros
    and the polynomial maps zeros to zeros, so the masked region is inert and
    nothing outside the shape is ever stored.
    """
    pid = tl.program_id(0)
    # int64 slice offset: dion's `_batch_offset` — an int32 product overflows on
    # the stacked `[1024, 64, 5461]` b1 chunk (3.6e8 elements).
    xb = X_ptr + pid.to(tl.int64) * sxs
    ob = O_ptr + pid.to(tl.int64) * sos
    rr = tl.arange(0, RT)
    rm = rr < R_
    nblk = tl.cdiv(D_, BLOCK_D)

    # --- 1. ‖X‖_F for THIS slice (the prescale `zeropower_via_ns` applies) ----
    sq = tl.zeros((RT,), tl.float32)
    for d0 in range(0, nblk):
        dd = d0 * BLOCK_D + tl.arange(0, BLOCK_D)
        mk = rm[:, None] & (dd < D_)[None, :]
        x = tl.load(xb + rr[:, None] * sr + dd[None, :] * sd, mask=mk, other=0.0)
        sq += tl.sum(x.to(tl.float32) * x.to(tl.float32), axis=1)
    den = tl.sqrt(tl.sum(sq)) + EPS

    # --- 2. X0 = X / den into O ----------------------------------------------
    for d0 in range(0, nblk):
        dd = d0 * BLOCK_D + tl.arange(0, BLOCK_D)
        mk = rm[:, None] & (dd < D_)[None, :]
        x = tl.load(xb + rr[:, None] * sr + dd[None, :] * sd, mask=mk, other=0.0)
        tl.store(ob + rr[:, None] * sr + dd[None, :] * sd, x.to(tl.float32) / den, mask=mk)

    # --- 3. the Jordan chain, in place on O, one launch for all steps ---------
    for _ in range(STEPS):
        gram = tl.zeros((RT, RT), tl.float32)
        for d0 in range(0, nblk):
            dd = d0 * BLOCK_D + tl.arange(0, BLOCK_D)
            mk = rm[:, None] & (dd < D_)[None, :]
            x = tl.load(ob + rr[:, None] * sr + dd[None, :] * sd, mask=mk, other=0.0)
            gram = tl.dot(x, tl.trans(x), gram, input_precision="ieee")
        poly = B * gram + C * tl.dot(gram, gram, input_precision="ieee")
        for d0 in range(0, nblk):
            dd = d0 * BLOCK_D + tl.arange(0, BLOCK_D)
            mk = rm[:, None] & (dd < D_)[None, :]
            x = tl.load(ob + rr[:, None] * sr + dd[None, :] * sd, mask=mk, other=0.0)
            tl.store(
                ob + rr[:, None] * sr + dd[None, :] * sd,
                A * x + tl.dot(poly, x, input_precision="ieee"),
                mask=mk,
            )


def zeropower_triton(x: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """The fused chain: `[m, n]` or `[B, m, n]` in, the polar factor out, fp32.

    The same semantics as `optim.zeropower_via_ns(kind="classic")` — normalize by
    `‖x‖_F + 1e-7` (per slice when batched), iterate the Jordan triple
    (`JORDAN_NS_COEFFS`, one constant, not a copy), return the polar factor in
    the caller's ORIENTATION (the chain runs on the `min(m, n)` side, exactly as
    the torch path transposes, but here by swapping two stride arguments).

    fp32 in / fp32 out. A bf16 input is upcast in-kernel and the chain still runs
    in fp32 — the kernels are fp32-native, which is why this arm SUBSUMES the
    `ns_dtype="bf16"` rung rather than composing with it
    (`docs/research/2026-10-17-muon-variants.md` §5: "the two must not ride in
    one A/B"). Exactly ONE kernel launch per call, plus the output allocation.
    """
    _check(x)
    assert isinstance(steps, int) and steps >= 0, f"steps={steps!r} must be a non-negative int"
    m, n = x.shape[-2], x.shape[-1]
    r, d = min(m, n), max(m, n)
    # The logical [r, d] orientation's strides: the transpose is free (two
    # arguments), so no `x.T.contiguous()` copy and no second launch.
    sr, sd = (x.stride(-1), x.stride(-2)) if m > n else (x.stride(-2), x.stride(-1))
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    assert out.stride(-2) == x.stride(-2) and out.stride(-1) == x.stride(-1), (
        "the output must share the input's layout, or the strides above are wrong for the store"
    )
    a, b, c = JORDAN_NS_COEFFS
    batch = x.shape[0] if x.ndim == 3 else 1
    _ns_chain_kernel[(batch,)](
        x,
        out,
        x.stride(0) if x.ndim == 3 else 0,
        out.stride(0) if x.ndim == 3 else 0,
        sr,
        sd,
        r,
        d,
        a,
        b,
        c,
        NS_EPS,
        STEPS=steps,
        RT=max(16, triton.next_power_of_2(r)),
        BLOCK_D=BLOCK_D,
        num_warps=NUM_WARPS,
    )
    return out


def zeropower_triton_or_classic(
    g: torch.Tensor,
    steps: int = 5,
    *,
    counters=None,
    ns_dtype: str = "fp32",
) -> torch.Tensor:
    """The `ns_kind="triton"` arm: the fused kernel, else the classic chain COUNTED.

    This is the whole body of the arm, so the fallback logic is testable without
    a GPU and without the optimizer: `counters` records `ns_triton` when the
    fused chain ran and `ns_triton_fallback` + the reason when it did not.
    `counters` is REQUIRED — a fallback a reader of the log cannot see is exactly
    the defect ADR-0019 exists to prevent, and "no counters passed" would be a
    silent one.

    `ns_dtype != "fp32"` is a COUNTED fallback, not a compose: the kernels run
    fp32 (see `zeropower_triton`), so the two arms cannot ride in one A/B.

    The third COUNTED reason is the LAUNCH's own refusal (a tile set whose shared
    memory is past the device's limit): it is caught here rather than allowed to
    crash, and it carries triton's own required-vs-limit numbers.
    """
    from ..instrument import loud

    if counters is None:
        loud(
            "the triton NS arm was called with no Counters: the arm's whole claim is "
            "that it can SHOW which chain ran (`ns_triton` / `ns_triton_fallback` + "
            "reason), and an uncounted call cannot. Escape: pass a `Counters` "
            "(`optim.Muon.counters` is the one the trainer drains)."
        )
    # The CONFIG reason is asked first: it is the one a reader can act on, and a
    # device reason would otherwise mask it on every CPU box.
    why = None
    if ns_dtype != "fp32":
        why = (
            f"ns_dtype={ns_dtype!r}: the Triton arm runs the chain in fp32 and SUBSUMES "
            f"the bf16 rung, so the two must not ride in one A/B"
        )
    if why is None:
        why = triton_reason(g)
    if why is None:
        try:
            out = zeropower_triton(g, steps)
        except LAUNCH_REFUSALS as e:  # pragma: no cover - needs a device to fire
            # A tile set the device cannot fit is a REFUSAL, and a refusal is a
            # degradation like any other: COUNTED, with the number the reader
            # needs (required vs limit), never a crash at launch. Reachable on
            # this card only by editing `BLOCK_D` — which is the point: the
            # constants above are the thing a sweep moves, and the sweep must
            # not be able to turn a measurement into a traceback.
            # `tests/test_ns_triton.py::test_a_shape_that_cannot_fit_falls_back_counted`
            # drives this branch by patching `BLOCK_D` to 128.
            why = f"the launch was refused by the device ({type(e).__name__}: {e})"
        else:
            counters.count("ns_triton")
            return out
    from .optim import zeropower_via_ns  # lazy: `optim` imports this module only inside its arm

    counters.count("ns_triton_fallback")
    counters.why = f"triton: {why}"  # type: ignore[attr-defined]
    # The fallback IS the classic chain, in the CALLER'S rung: `ns_dtype` is
    # forwarded, so `ns_kind="triton", ns_dtype="bf16"` degrades to the classic
    # bf16 chain rather than silently running fp32 (which is what the reason
    # above refuses to compose with). `tests/test_muon_fast.py` gates every
    # member of `optim.NS_KINDS` on the bf16 rung, so this is measured.
    return zeropower_via_ns(g, steps, "classic", ns_dtype=ns_dtype)
