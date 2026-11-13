"""TSCT: `LinearLike` reinterpreted in torch — `W = U diag(s) Vᵀ`.

`burn-spectral`'s `SpectralLinear` keeps the factors `U [in, r]`, `s [r]`,
`V [r, out]` and computes the effective weight on the fly, so a `d → f`
projection costs `r·(d + f + 1)` parameters instead of `d·f`. That is what
makes a 5461-wide FFN affordable, and it is what the retraction and the Muon
group operate on: the polar retraction is done in the FACTORED form, so fp32
Newton-Schulz over `[64, f]` replaces "~40 s/step" over `[d, d]` (AGENTS §2.3).

Precision (`docs/architecture/bf16-plan.md`): the factors are fp32 MASTERS;
the forward casts to bf16, matmuls, and returns fp32 — softmax/loss/KDA state
stay fp32 at the arm boundary. `quant = "none"` is the day-one format and it
resolves to bf16 factors; `quant = "ternary"` is the ONE STORAGE rung
(checkpoint/bank compression, unpacked inside the forward — see below); every
other rung (LittleBit semantics) is still refused at CONFIG load (see
`config._refuse_ladder_rung`), so a rung can never silently compute bf16
while the config reads a name it did not implement. 2026-11-09: the
`fp8`/`nvfp4` storage rungs and the `binary`/`mxfp4` QAT rungs were deleted —
TSCT is ternary, full stop. The activation side is a
separate field (`config.act_quant`, `model/act_quant.py`), not a rung of this
ladder: the factors stay bf16, only the activations the factor matmuls read are
fake-quantized (with an STE), and the a4.8 attention rule is
`act_quant.attn_format`.

STORAGE rung vs bf16, the exact mechanics (validated by test):
* master fp32 factors are projected to b1.58 ternary (2-bit codes + one fp32
  absmean scale) **at state dict save/load time** — the rung is a
  checkpoint-compression device and a memory format for the bank; the fp32
  master floats stay the optimizer's object inside a live run.
* the forward path never changes numerics within a run: the rung pins a
  round trip `fp32 → storage → fp32` that is BIT-EXACT to
  `ternary.ternarize`, which is where the loss is allowed to live and no
  further.
* LittleBit is NOT the storage rung semantics above (it is a QAT ladder
  step: sign(U)s·sign(V)ᵀ + learned scales, G-A..G-D gates in
  `docs/reviews/littlebit-design-2026-10-15.md` — a training-time change,
  not a checkpoint format).

Deviations from burn, both deliberate and listed in the port report:

* **no `N % 4` output padding.** burn pads `out_features` to a multiple of 4
  and slices back because cubek's matmul vectorizes on `N % 4 == 0`
  (AGENTS §2.2). torch has no such constraint, so the port drops the pad — the
  parameter count is the honest `r·(in + out + 1)` either way.
* **the STE quantizer IS in the forward** (since 2026-10-16, `factor_qat`):
  the b1.58 ternary quantizer with burn-spectral's
  annealed STE, applied to every TSCT factor through `factor_qat_project`.
  What the compression tournament MEASURED at 9.2M (3 seeds x 2k steps):
  hard STE from step 0 loses +0.050 BPB to bf16;
  the annealed ramp (alpha 0->1 over 1000 steps) recovers only 0.010 of it
  (+0.040) with a tighter spread — so at THIS scale the arm is CLOSED for
  training (the controls' spread is 0.006). The ternary storage rung
  (checkpoint compression) is untouched by that verdict.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..instrument import Counters, loud
from .qat_cache import QuantCache, note_qat_values

# ---------------------------------------------------------------------------
# COMPUTE rungs: the fp8 2-D GEMM lane (`mm_dtype`)
# ---------------------------------------------------------------------------

#: The `mm_dtype` literals — the COMPUTE precision of `mm_mixed`'s 2-D GEMMs.
#: `bf16` is the default and every number on record: cast the fp32 masters to
#: the compute dtype, let cuBLAS run the bf16 tensor-core GEMM. `fp8` runs the
#: same product through `torch._scaled_mm` (e4m3 × e4m3 → `dt` out) with
#: per-tensor amax scales computed from the bf16 operand. It is a COMPUTE lane,
#: not a storage rung: the fp32 masters and the checkpoint are untouched.
#: Measured by the spike on this card (RTX 5060 Ti, 2026-10-15; the spike dir's
#: `bench_f_fp8.py` = `docs/research/2026-10-15-torch-spike.md` phase 2, bf16
#: in/out, per-tensor scales): [32768,2048]×[2048→64] 23.8 → 39.3 TFLOP/s,
#: [32768,64]×[64→5461] 14.2 → 22.3 — **1.6-1.8×** — and `_scaled_grouped_mm`
#: refuses sm_120 outright ("only supported on CUDA devices with compute
#: capability = [9.0, 10.0]"), so the grouped/batched expert pass has no fp8
#: path and is not attempted. `config.MM_DTYPES` is the same list at the schema
#: layer (duplicated the way `NS_KINDS` is, so the config layer does not import
#: the model).
MM_DTYPES = ("bf16", "fp8")

#: The process-wide compute mode (`set_mm_dtype`). `bf16` until a trainer says
#: otherwise: a fresh process, an eval, a decode and every test run the default.
_MM_DTYPE = "bf16"

#: E4M3's own signed max — the per-tensor scale's divisor.
_FP8_MAX = 448.0

#: COUNTED fallbacks, reason → count, process-local. The `Counters` half is what
#: the log line reads; this tally is what makes the guarantee unconditional — a
#: call site that passes no `counters` still cannot fall back silently (ADR-0019).
_FP8_FALLBACKS: dict[str, int] = {}


def set_mm_dtype(mode: str) -> None:
    """`TrainConfig.mm_dtype`'s ONE entry point (`train/loop.py`, beside the
    `tf32` knob): the compute mode is process-wide, like
    `torch.set_float32_matmul_precision`, and a fresh process is `bf16`.

    A value outside [`MM_DTYPES`] is LOUD here too — the setter is reachable
    from a script or a test that never went through `load_run`, and the config
    layer's refusal is not a substitute for the seam's.
    """
    if mode not in MM_DTYPES:
        loud(
            f"mm_dtype={mode!r} is not a compute mode. Known: {list(MM_DTYPES)} — "
            f"`bf16` the bf16 tensor-core GEMM (the default and every number on "
            f"record), `fp8` the e4m3 `torch._scaled_mm` lane (2-D only, forward "
            f"only). Escape: name one of those."
        )
    global _MM_DTYPE
    _MM_DTYPE = mode


def mm_dtype() -> str:
    """The live compute mode (`set_mm_dtype`'s state, for a gate or a bench)."""
    return _MM_DTYPE


def mm_fp8_fallbacks() -> dict[str, int]:
    """The process-local reason tally of the fp8 lane's COUNTED fallbacks."""
    return dict(_FP8_FALLBACKS)


def fp8_quantize_2d(x: torch.Tensor, dt: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor amax e4m3 quantize: `(q, scale)` with `x ≈ q.float() * scale`.

    The scale is `amax(|x|) / 448` — E4M3's own signed max — computed from the
    operand in `dt`, i.e. from the same bf16 tensor the bf16 lane would have
    multiplied, and the cast is RNE onto the e4m3 grid (the spike's
    `quant_per_tensor`). `clamp_min(1e-12)` is the all-zero operand: `0/1e-12`
    quantizes to zero and the product comes back zero, where an unclamped `0/0`
    would be NaN. The reduction is `detach()`ed — the scale is a constant of the
    forward, and the lane is forward-only anyway (see [`mm_fp8`]).
    """
    amax = x.detach().abs().amax().to(torch.float32).clamp_min(1e-12)
    scale = (amax / _FP8_MAX).reshape(1)
    q = (x.float() / scale).to(torch.float8_e4m3fn).contiguous()
    return q, scale


def mm_fp8_shape_ok(k: int, n: int) -> bool:
    """The e4m3 `_scaled_mm` shape rule: `K % 16 == 0 and N % 16 == 0`.

    The one precondition that is a property of the SHAPE rather than of a live
    tensor — so a bench, or a coverage table for the model's GEMM inventory, can
    ask it without a device. [`_fp8_reason`] reads it; a shape that fails it is
    a COUNTED fallback, never a padded call (padding a K or an N would change
    the product).
    """
    return k % 16 == 0 and n % 16 == 0


def _fp8_reason(x: torch.Tensor, weight: torch.Tensor, dt: torch.dtype) -> tuple[str, str] | None:
    """`None` if the fp8 lane can run THIS call, else `(reason, detail)`.

    The ORDER is the operands' own properties first (dtype, rank, the
    16-divisibility the e4m3 kernel needs, the contraction), the environment
    after (device, autograd): a CPU test can reach the shape reasons, and a CUDA
    call with a bad shape is told the shape reason rather than "not CUDA".

    The autograd entry is a MEASUREMENT, not a guess: torch 2.13.0+cu129
    registers no derivative for the op (`RuntimeError: derivative for
    aten::_scaled_mm is not implemented`, reproduced on CPU for both
    `_scaled_mm` and `_scaled_mm_v2`), so a training forward — where the
    operands require grad — cannot use the lane. It falls back COUNTED, and the
    lane serves the no-grad forwards (eval, decode, the bench).
    """
    if not hasattr(torch, "_scaled_mm"):
        return ("this torch has no `torch._scaled_mm`", torch.__version__)
    if dt not in (torch.bfloat16, torch.float16):
        return ("the compute dtype is fp32 (the lane's out_dtype is bf16/fp16)", str(dt))
    if x.dim() != 2 or weight.dim() != 2:
        return (
            "non-2-D operand (`_scaled_mm` is the 2-D GEMM)",
            f"x{tuple(x.shape)} W{tuple(weight.shape)}",
        )
    if weight.shape[-1] != x.shape[-1]:
        return (
            "the operands do not contract",
            f"x{tuple(x.shape)} @ W{tuple(weight.shape)}.T",
        )
    k, n = x.shape[-1], weight.shape[0]
    if not mm_fp8_shape_ok(k, n):
        return (
            "K or N is not a multiple of 16 (the e4m3 `_scaled_mm` rule)",
            f"K={k} N={n}",
        )
    if not x.is_cuda or not weight.is_cuda:
        return ("not CUDA (the lane is the e4m3 kernel path)", str(x.device))
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        return (
            "autograd is on (torch's `_scaled_mm` has no backward)",
            "the lane is forward-only",
        )
    return None


def _fp8_fallback(reason: str, detail: str, counters: Counters | None) -> None:
    """The COUNTED mark of a fallback: the tally always, the log line if given."""
    _FP8_FALLBACKS[reason] = _FP8_FALLBACKS.get(reason, 0) + 1
    if counters is not None:
        counters.count("mm_fp8_fallback")
        note = f"mm_fp8: {reason} [{detail}]"
        prev = getattr(counters, "why", None)
        counters.why = f"{prev}; {note}" if prev else note  # type: ignore[attr-defined]


def mm_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    dt: torch.dtype,
    counters: Counters | None = None,
) -> torch.Tensor | None:
    """`x @ Wᵀ` through `torch._scaled_mm`, or `None` = "run the bf16 lane".

    Both operands are quantized per-tensor (e4m3) from their bf16 form and the
    GEMM runs fp8 in / `dt` out; the fp32 masters are not read here at all, so
    the storage and the optimizer's object cannot move (this is a COMPUTE rung).

    `None` is returned only after a COUNTED fallback — `mm_fp8_fallback` on the
    `Counters` (the log line) + the reason on `counters.why`, and the
    process-local [`mm_fp8_fallbacks`] tally so a call with no counters is still
    not silent (ADR-0019). Every precondition is in [`_fp8_reason`]; the one a
    training run meets is autograd, because torch 2.13's `_scaled_mm` has no
    backward.
    """
    why = _fp8_reason(x, weight, dt)
    if why is not None:
        _fp8_fallback(why[0], why[1], counters)
        return None
    a = x if x.dtype == dt else x.to(dt)
    w = weight if weight.dtype == dt else weight.to(dt)
    aq, sa = fp8_quantize_2d(a, dt)
    wq, sw = fp8_quantize_2d(w, dt)
    # `_scaled_mm` wants mat2 column-major: `wq` is [N, K] row-major, so its
    # transpose is [K, N] with strides (1, K) — the layout the fp8 kernel asks
    # for, as a VIEW (no copy; the spike's `col_major` materializes the same).
    try:
        out = torch._scaled_mm(aq, wq.t(), sa, sw, out_dtype=dt)
    except RuntimeError as e:
        # A kernel refusal for a call the preconditions accepted (an unlisted
        # stride/dtype combination on some future shape) is COUNTED and the run
        # stays on the bf16 lane — the `moe_triton_fallback` /
        # `opt_graph_fallback` rule: a failed accelerated arm degrades, it does
        # not kill the run. The quantize above is plain torch and stays OUTSIDE
        # this handler: a failure there is a bug in the lane, and bugs are LOUD.
        _fp8_fallback("`_scaled_mm` refused the call", str(e)[:160], counters)
        return None
    if counters is not None:
        counters.count("mm_fp8_gemm")
    return out


def mm_mixed(
    x: torch.Tensor,
    weight: torch.Tensor,
    dt: torch.dtype,
    counters: Counters | None = None,
) -> torch.Tensor:
    """`x @ Wᵀ` in `dt` — the port's mixed-dtype rule (F.linear). The OUT is `dt`.

    torch's `Linear` dispatched on the INPUT's dtype: fp32 masters with fp32
    activation inputs run fp32 GEMM, which on sm_120 means the MAGMA lane —
    measured 46.6% of small's forward CUDA time in the 2026-10-15 profiler
    probe (280 × ~240 µs fp32 gemms against TSCT's 13-38 µs bf16 cutlass).
    Casting per forward (masters stay fp32, exactly TSCTLinear's contract)
    routes every projection through the bf16 cuBLAS tensor-core path.

    The result dtype is the COMPUTE dtype, not fp32: with `bf16 = true` this
    returns bf16 (an fp32 `x @ Wᵀ` under bf16 would have been the fp32 GEMM the
    rule exists to avoid), so a caller that needs fp32 widens it explicitly —
    `SparseMoE`'s `down` does, and so does the MoE combine boundary, which is
    why the router's bf16 softmax weight is cast there rather than here.

    Under `mm_dtype = "fp8"` this is the ONE seam the lane enters: the product
    is attempted through [`mm_fp8`] first and the bf16 line below is the COUNTED
    fallback, never a silent one. Every call site passes `counters`, so the log
    line's `mm_fp8_gemm` / `mm_fp8_fallback` counters say which lane ran; the
    reason rides `counters.why` (the port's seam channel, the same one
    `moe_triton_fallback` uses) and [`mm_fp8_fallbacks`].
    """
    if _MM_DTYPE == "fp8":
        y = mm_fp8(x, weight, dt, counters)
        if y is not None:
            return y
    w = weight if weight.dtype == dt else weight.to(dt)
    return (x if x.dtype == dt else x.to(dt)) @ w.t()


def qat_values(
    w: torch.Tensor, mode: str, alpha: float, *, dim: tuple | None = None
) -> torch.Tensor:
    """The raw projection `q(w)` under `factor_qat` — values only, no STE.

    Split out for `QuantCache`: the cache stores ONE detached `q` per weight per
    optimizer step, and [`factor_qat_project`] re-composes the SAME expression
    (`w + alpha·(q − w)`) from it on every forward, so cached and uncached
    forwards are bit-identical by construction. The cache stores the PROJECTION,
    never the blended tensor: `w + (w_q − w)` is not bitwise `w_q`.

    `dim` is the absmean axis: `None` = per-tensor (`ternary.ternarize`), a tuple
    = per first-index slice, which is the per-EXPERT α of an `[E, ·, ·]` bank.
    """
    if mode == "none":
        return w
    if mode != "ternary":
        raise AssertionError(f"factor_qat/dense_qat={mode!r} reached the projector with no mode")
    note_qat_values(w)  # one refresh call; `w.shape[0]` is how many tensors it covers
    if dim is None:
        from .ternary import ternarize

        assert 0.0 <= alpha <= 1.0, (
            f"annealing alpha={alpha} is a schedule position in [0, 1] "
            "(1 = pure ternary, 0 = the master); refused, not clamped"
        )
        return ternarize(w)
    # The per-slice axis (banks): the same formula as `ternary.ternarize`,
    # transposed onto `dim` — the grid is ±mean(|W|),
    # ε only guards the division (exactly `ternarize`'s convention: divide by
    # `α+ε`, scale back by `α`), so a banked expert and the same expert run
    # alone land on the SAME values.
    a = w.abs().mean(dim=dim, keepdim=True)
    q = (w.float() / (a + 1e-5)).round().clamp(-1.0, 1.0) * a
    return q.to(w.dtype)


def factor_qat_project(
    w: torch.Tensor, mode: str, alpha: float, *, dim: tuple | None = None, cache=None
) -> torch.Tensor:
    """The forward's view of a TSCT factor master under `factor_qat`.

    ONE place both consumers (the loop block's `TSCTLinear` and the experts'
    `DualLatentGLU`) route through — the burn-era defect this avoids is two
    copies of one quantizer drifting apart (the tool for that is a grep).

    - `none` → the master untouched (bit-identical to every number on record);
    - `ternary` → b1.58 STE (`ternary.ste_ternary_annealed`, burn-spectral's
      measured "converges better than hard STE" annealing).

    ONE RUNG, and it is DETERMINISTIC: the projection is a pure function of the
    master, so a forward draws no RNG and a resume continues the same function.
    The S3T sampled arm — and the process-local generator its Bernoulli mask
    came from — was deleted 2026-11-13 (owner decision: bf16 masters, 4-bit
    activations, ternary weights; `config.REMOVED_QUANT_MODES`).

    `dim` picks the absmean's aggregation axis: `None` = the whole tensor (the
    factor-per-layer reading, `ternary.ternarize`'s per-TENSOR axis); `(1, 2)`
    = per first-index slice for an `[E, ·, ·]` bank, i.e. **per-EXPERT α** —
    the reading that makes a banked expert identical to the same expert run
    through `DualLatentGLU.forward` (a 2-D factor's per-tensor mean IS that
    expert's mean). The rung's gold test pins both axes.

    STE algebra: forward sees the quantized value, backward flows through the
    master (`w_eff = w + alpha·(q(w) − w).detach()`; `alpha = 0` is exactly the
    master). The masters stay fp32 — quantization lives on the compute copy
    only, exactly the storage-rung contract (`pack_tsct`'s docstring).

    PASS `cache=` WHEN THE WEIGHTS ONLY MOVE ONCE PER OPTIMIZER STEP. Every
    forward would otherwise re-run the projection (`absmean`, divide, round,
    clamp, multiply): measured 17.1 k tok/s against 40.9 k bf16 on the 9.2 M
    polygon (2026-10-21), and the projection is the whole difference. With a
    [`qat_cache.QuantCache`], `q` is computed on the first forward after the
    trainer's per-step invalidation and every later forward pays one sub + one
    add (`QuantCache.ste`); the logits are bit-identical to the uncached path
    and dL/dw is unchanged (`tests/test_qat_cache.py`). `cache=None` (every
    direct API caller that does not opt in) keeps the old per-forward path.
    """
    if mode == "none":
        return w
    if alpha == 0.0:
        # The annealing floor is the master itself: skip the projection (and
        # its launch chain) rather than compute `w + 0·(q − w)`. The cache
        # already takes this short-circuit; the uncached arms now agree.
        return w
    if cache is not None:
        # The QAT cost fix: `refresh` is the once-per-step projection (`q` is
        # DETACHED); `ste` is the per-forward sub+add. A miss is the uncached
        # math once; a hit is a dict lookup. The trainer invalidates the cache
        # after every optimizer step (`qat_cache.invalidate_qat_cache_all`).
        cache.refresh(w, mode, alpha, dim)
        return cache.ste(w)
    q = qat_values(w, mode, alpha, dim=dim)
    return (w + alpha * (q - w).detach()).to(w.dtype)


# ---------------------------------------------------------------------------
# The DENSE path: the same projector, applied to plain `[out, in]` weights
# ---------------------------------------------------------------------------

#: The dense path's projector — the SAME function object as
#: [`factor_qat_project`], deliberately. What the owner's final-quantized
#: recipe needs on a plain linear weight IS the factor rung's semantics
#: (per-tensor absmean `ternarize` under the annealed STE
#: `w + alpha·(q(w) − w).detach()`, with
#: `alpha = 0` the master bit for bit), and one object means the two paths
#: cannot drift apart. `dim=None` is the per-TENSOR axis — the BitNet
#: convention for a weight matrix (b1.58 arXiv:2402.17764 Eq. 1).
#:
#: NOT routed through it, on purpose (the BitNet convention: these are not
#: weight matrices): `nn.Embedding` tables (`embed`, DSpark's `w1`), the Engram
#: n-gram tables, every norm, every bias, and the TSCT diagonals (`s`,
#: `gate_s`, `up_s` — they ARE the scale). The TSCT FACTORS stay on
#: `factor_qat`; this name is what every `nn.Linear`/`DenseLinear` weight in
#: the forward goes through when `dense_qat != "none"`.
dense_qat_project = factor_qat_project

#: The attention sites `dense_qat_skip_attn` leaves in bf16 — ONE list, read by
#: `dense_qat_kind` below AND by `tests/test_dense_qat.py`'s coverage gate, so a
#: site cannot join the quantized set by accident. KDA's five projections, the
#: MSA pair + its selector, and the dense transformer block's packed `qkv` +
#: `proj` (the `transformer.*` names, `model/transformer.py`): BitNet b1.58
#: (arXiv:2402.17764) quantizes the FFN weight matrices, not attention, and our
#: `act_quant` already upgrades the attention path to >= 8 bits. Projecting
#: these measured +0.198 BPB (+6.5%) against the bf16 control (2026-10-21,
#: medium b8 s512, 2000 steps, one 81 920 B eval window, 3 seeds).
DENSE_QAT_ATTN_SITES = (
    "loop.attn.q_proj",
    "loop.attn.k_proj",
    "loop.attn.v_proj",
    "loop.attn.g_proj",
    "loop.attn.beta_proj",
    "loop.attn.o_proj",
    "loop.msa.q_proj",
    "loop.msa.k_proj",
    "loop.msa.v_proj",
    "loop.msa.selector.q_idx",
    "loop.msa.selector.k_idx",
    "transformer.qkv",
    "transformer.proj",
)


def dense_qat_kind(cfg, *sites: str) -> str:
    """`cfg.dense_qat`, or `"none"` when `dense_qat_skip_attn` and any site is attention.

    Every consumer names ALL the sites it owns, so a module whose sites are in
    `DENSE_QAT_ATTN_SITES` reads bf16 wholesale and a new `nn.Linear` added to
    it inherits the rule instead of silently joining the quantized set.
    """
    if cfg.dense_qat_skip_attn and any(site in DENSE_QAT_ATTN_SITES for site in sites):
        return "none"
    return cfg.dense_qat


def dense_qat_weight(weight: torch.Tensor, kind: str, alpha: float, *, cache=None):
    """`dense_qat_project`, or the master itself when the arm is off.

    The no-op arm must not even CALL the projector: `tests/test_dense_qat.py`'s
    gate d spies the function and requires zero calls for `dense_qat="none"`.
    """
    if kind == "none":
        return weight
    return dense_qat_project(weight, kind, alpha, cache=cache)


def qat_linear(linear: nn.Linear, x: torch.Tensor, kind: str, alpha: float) -> torch.Tensor:
    """`linear(x)` with `dense_qat` applied to its weight — the master when off.

    The ONE call shape for every plain `nn.Linear` site (KDA's q/k/v/g/beta,
    the Engram's three projections, MSA's, DSpark's). `F.linear` IS
    `nn.Linear.forward`'s whole body, so `kind = "none"` returns `linear(x)`
    itself — bit-identical to the call it replaces, no copy.
    """
    if kind == "none":
        return linear(x)
    # `_qat_cache` is attached by `DormouseModel.__init__` (`attach_qat_caches`);
    # a bare `nn.Linear` outside a model has none and takes the per-forward path.
    w = dense_qat_project(linear.weight, kind, alpha, cache=getattr(linear, "_qat_cache", None))
    return F.linear(x, w, linear.bias)


def init_u(in_features: int, rank: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """The `U [in, rank]` init, `N(0, 1/sqrt(in))` — the distribution the
    comment below names ("the composed map has unit-ish output scale at any
    width", which matters because ReZero's `residual_scale` starts at 1.0).

    ONE copy of the formula: the expert bank's latents (`DualLatentGLU`) and,
    under `share_u`, the single basis `SparseMoE` owns (the `plane` mode's
    `[d, rank]`, and the `dict` mode's `[d, share_u_rank]` DICTIONARY, whose
    columns each expert reads `rank` of) draw the same factor. Two
    transcriptions is how a shared basis and a per-expert one start the A/B
    from different scales and the comparison charges that to the mechanism.
    """
    return torch.randn(in_features, rank, device=device) / math.sqrt(in_features)


class TSCTLinear(nn.Module):
    """A factorized linear: `x → ((x·U) ⊙ s)·V`, masters fp32, forward in `dtype`."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        bias: bool = False,
        factor_qat: str = "none",
        factor_qat_alpha: float = 1.0,
        act_quant: str = "none",
        act_group: int = 0,
        attn_path: bool = False,
        device: torch.device | str = "cpu",
        weight_std: float | None = None,
    ) -> None:
        super().__init__()
        if rank > min(in_features, out_features):
            loud(
                f"TSCT rank {rank} exceeds min(in={in_features}, out={out_features}): "
                f"a rank above either dimension is not a factorization (burn clamps "
                f"only the lm_head, explicitly, in `model.rs`). Escape: lower `rank` "
                f"or make the projection dense (`use_tsct = false`)."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.compute_dtype = dtype
        # U ~ N(0, 1/sqrt(in)), V ~ N(0, 1/sqrt(rank)): the composed map has
        # unit-ish output scale at any width, which matters because ReZero's
        # residual_scale is initialised to 1.0 (not 0 — at 0 the block body
        # gets no gradient at all, the 2026-09-27 burn bug).
        # `weight_std` REPLACES that scale when the call site has a denser
        # contract (the transformer block: the dense arm's `nn.init.normal_`).
        # With `u ~ N(0, 1/sqrt(in))`, `s = 1` and `v ~ N(0, weight_std·sqrt(in/rank))`
        # each entry of `U diag(s) V` is a sum of `rank` products, std
        # `sqrt(rank)·(1/sqrt(in))·weight_std·sqrt(in/rank) = weight_std`.
        self.u = nn.Parameter(init_u(in_features, rank, device))
        self.s = nn.Parameter(torch.ones(rank, device=device))
        if weight_std is None:
            v = torch.randn(rank, out_features, device=device) / math.sqrt(rank)
        else:
            v = torch.randn(rank, out_features, device=device) * (
                weight_std * math.sqrt(in_features / rank)
            )
        self.v = nn.Parameter(v)
        self.bias = nn.Parameter(torch.zeros(out_features, device=device)) if bias else None
        #: The compressed-forward sprint's switch (`factor_qat_project`): the
        #: forward reads quantized factors with STE while the masters stay
        #: fp32. `s` is never quantized — it IS the scale.
        self.factor_qat = factor_qat
        self.factor_qat_alpha = factor_qat_alpha
        #: The once-per-step projection cache (`qat_cache.QuantCache`): `u` and
        #: `v` each get one detached `q` per trained step, instead of every
        #: forward re-running the quantizer (the measured 2.4x cost).
        self._qat_cache = QuantCache() if factor_qat != "none" else None
        #: The a4.8 ACTIVATION switch (`act_quant.fake_quant_act`): the `x` that
        #: feeds the factor matmuls is fake-quantized with an STE.
        #: `attn_path` marks a projection that sits INSIDE the attention arm, and
        #: then the format is upgraded to >= 8 bits (`act_quant.attn_format`) —
        #: the a4.8 recipe's rule, unconditional. No call site in the port's graph
        #: sets it yet: the KDA arm's q/k/v/g projections are dense `nn.Linear`
        #: (`kda.py:131-133`) and `LoopBlock`'s one TSCT projection (`out_proj`) is
        #: the readout AFTER attention+FFN, not inside attention.
        self.act_quant = act_quant
        self.act_group = act_group
        self.attn_path = attn_path

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        """The activation the factor matmuls see under `act_quant` (STE).

        Lazily imported: `model/act_quant.py` imports the e2m1 grid FROM this
        module (`_NVFP4_MAGNITUDES`), so a module-level import here would be a
        cycle — the same lazy form `factor_qat_project` uses for `ternary`.
        """
        if self.act_quant == "none":
            return x
        from .act_quant import attn_format, fake_quant_act

        fmt = attn_format(self.act_quant) if self.attn_path else self.act_quant
        return fake_quant_act(x, fmt, self.act_group)

    def _factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The forward's `(u, v)`: masters under `factor_qat`, else untouched."""
        if self.factor_qat == "none":
            return self.u, self.v
        f = factor_qat_project
        cache = self._qat_cache
        return f(self.u, self.factor_qat, self.factor_qat_alpha, cache=cache), f(
            self.v, self.factor_qat, self.factor_qat_alpha, cache=cache
        )

    def effective_weight(self) -> torch.Tensor:
        """`U diag(s) Vᵀ` in fp32 — the weight the optimizer and the report
        talk about. Never used in a forward (it would materialize `d·f`).
        Reads the QUANTIZED factors when the qat arm is on: the report must
        describe the network that ran, not the masters next to it."""
        u, v = self._factors()
        return (u * self.s) @ v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = self.compute_dtype
        u, v = self._factors()
        x = self._act(x)  # a4.8: the activation feeding the factor matmuls
        if dt == torch.float32:
            # The fp32 lane needs an operand matching the factors: a bf16
            # residual (`residual_dtype="bf16"`) widens here, and a module
            # whose factors were cast in a test keeps that dtype.
            x = x if x.dtype is u.dtype else x.to(u.dtype)
            y = ((x @ u) * self.s) @ v
        else:
            # Matmuls in the factor dtype with fp32 accumulate (torch's own
            # reduction), the master params untouched.
            xu = x.to(dt)
            y = ((xu @ u.to(dt)) * self.s.to(dt)) @ v.to(dt)
            y = y.to(torch.float32)
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
            f"dtype={self.compute_dtype}"
        )


# ---------------------------------------------------------------------------
# STORAGE rungs: checkpoint/bank compression of the fp32 factor masters
# ---------------------------------------------------------------------------

#: The matrix-parameter names that ARE factor matrices (everything else —
#: scales `.s`, router, engram, norms — stays fp32/goes bf16 untouched).
_FACTOR_SUFFIXES = ("u", "v", "gate_u", "up_u")

#: The e2m1 magnitude grid (0, .5, 1, 1.5, 2, 3, 4, 6). Declared here because
#: `model/mxfp.py` and `model/act_quant.py` import it as THE grid definition;
#: no TSCT rung reads it — those are ACTIVATION/checkpoint codecs, a different
#: feature.
_NVFP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _ternary_quant(t: torch.Tensor) -> tuple[torch.Tensor, float]:
    """b1.58 ternary projection → 2-bit codes + the α scalar (16× vs fp32).

    Stores the TERNARY PROJECTION of the master (`ternary.ternarize`, the same
    formula `factor_qat` trains against): codes 0/1/2 for {0, +α, −α}, four to
    a byte, α in fp32 beside them. Round-trip is bit-exact to `ternarize(t)`
    for finite t — the rung adds no rounding of its own beyond
    `ternarize`'s grid.
    """
    from .ternary import ternarize

    t = t.detach()
    q = ternarize(t)
    alpha = float(t.abs().mean())
    code = torch.zeros_like(t, dtype=torch.uint8)
    code[q > 0] = 1
    code[q < 0] = 2
    flat = code.reshape(-1)
    pad = (-flat.numel()) % 4
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8)])
    packed = flat[0::4] | (flat[1::4] << 2) | (flat[2::4] << 4) | (flat[3::4] << 6)
    return packed.to(torch.uint8), alpha


def _ternary_dequant(packed: torch.Tensor, alpha: float, n: int) -> torch.Tensor:
    """Inverse of [`_ternary_quant`]: codes → `{0, ±α}`, tail padding cut."""
    c = torch.stack(
        [packed & 3, (packed >> 2) & 3, (packed >> 4) & 3, (packed >> 6) & 3],
        dim=-1,
    ).reshape(-1)[:n]
    v = torch.zeros(n, dtype=torch.float32, device=packed.device)
    return v.masked_fill(c == 1, alpha).masked_fill(c == 2, -alpha)


FACTORS_QUANT = {
    "ternary": (_ternary_quant, _ternary_dequant),
}


def pack_tsct(t: torch.Tensor, rung: str) -> dict[str, object]:
    """One factor matrix → its storage record. Scales travel COMPUTED."""
    if t.dtype != torch.float32:
        loud(f"storage rungs quantize the fp32 masters, got dtype {t.dtype}")
    if not torch.isfinite(t).all():
        loud("storage: master factors must be finite (inf/NaN refuse, they encode wrong)")
    if rung == "ternary":
        packed, alpha = _ternary_quant(t)
        return {
            "rung": rung,
            "codes": packed,
            "alpha": alpha,
            "shape": t.shape,
            "n": t.numel(),
        }
    loud(f"pack_tsct: rung {rung!r} is not a storage rung (ternary)")


def unpack_tsct(rec: dict[str, object]) -> torch.Tensor:
    """Inverse of [`pack_tsct`] — master-precision dtype, rung-rounded values."""
    rung = rec["rung"]
    if rung == "ternary":
        return _ternary_dequant(rec["codes"], rec["alpha"], rec["n"]).reshape(rec["shape"])
    loud(f"unpack_tsct: rung {rung!r} is not a storage rung (ternary)")


def is_factor_key(key: str) -> bool:
    """The state-dict keys that carry a quantizable factor matrix.

    Either a TSCT factor suffix (`u`, `v`, `gate_u`, `up_u`) on a module, or,
    because the experts' matrix params share those suffixes through
    `DualLatentGLU`, the same check passed through the experts path.
    """
    leaf = key.rpartition(".")[2]
    return leaf in {"u", "v", "gate_u", "up_u", "weight_u"}


class DenseLinear(nn.Module):
    """The `use_tsct = false` counterpart: a plain `[out, in]` weight.

    It exists because TSCT's A/B is its own removal, not TSCT against nothing:
    at a matched parameter budget the comparison is wide-and-low-rank against
    narrow-and-dense (burn's `LinearLikeInner::Dense`, reachable since
    2026-09-28 for exactly this reason).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        bias: bool = False,
        dense_qat: str = "none",
        dense_qat_alpha: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = dtype
        #: The DENSE path's switch (`dense_qat_project`, the same projector as
        #: the factor rung): the forward reads the projected weight with an STE
        #: while the fp32 master stays the optimizer's object. `none` = the
        #: master untouched, every number on record.
        self.dense_qat = dense_qat
        self.dense_qat_alpha = dense_qat_alpha
        #: The once-per-step projection cache — the same object `TSCTLinear` and
        #: `qat_linear` read (`qat_cache.QuantCache`).
        self._qat_cache = QuantCache() if dense_qat != "none" else None
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device=device) / math.sqrt(in_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features, device=device)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = dense_qat_weight(
            self.weight, self.dense_qat, self.dense_qat_alpha, cache=self._qat_cache
        )
        if self.compute_dtype != torch.float32:
            x = x.to(self.compute_dtype)
            w = w.to(self.compute_dtype)
        y = x @ w.t()
        if self.compute_dtype != torch.float32:
            y = y.to(torch.float32)
        if self.bias is not None:
            y = y + self.bias
        return y


class DenseLowRank(nn.Module):
    """A dense `d → rank → d` readout: `2·d·rank` MAC instead of `d²`.

    The dense rung's `out_proj_rank` (`use_tsct = false`, so the readout would
    otherwise be a full `DenseLinear(d, d)`). Two dense linears, no bias and no
    nonlinearity — the composed map IS a rank-`rank` matrix, which is exactly
    the cost this flag names. Each half keeps `DenseLinear`'s compute dtype and
    `dense_qat` policy, so the readout does not become a second numerics path.
    """

    def __init__(
        self,
        in_features: int,
        rank: int,
        out_features: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        dense_qat: str = "none",
        dense_qat_alpha: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if rank > min(in_features, out_features):
            loud(
                f"out_proj_rank {rank} exceeds min({in_features}, {out_features}): "
                f"a rank above either dimension is not a factorization. Escape: "
                f"a lower rank."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.compute_dtype = dtype
        self.down = DenseLinear(
            in_features,
            rank,
            dtype=dtype,
            dense_qat=dense_qat,
            dense_qat_alpha=dense_qat_alpha,
            device=device,
        )
        self.up = DenseLinear(
            rank,
            out_features,
            dtype=dtype,
            dense_qat=dense_qat,
            dense_qat_alpha=dense_qat_alpha,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
            f"dtype={self.compute_dtype}"
        )


def make_projection(
    in_features: int,
    out_features: int,
    cfg,
    *,
    rank: int | None = None,
    bias: bool = False,
    attn_path: bool = False,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Spectral or dense, chosen once from `use_tsct` (a construction-time
    decision, so the shape of the model cannot drift afterwards).

    `attn_path` is forwarded to `TSCTLinear` and means "this projection sits
    inside the attention arm": the a4.8 activation upgrade (`act_quant.attn_format`)
    then applies to it. It is inert on the dense rung (a dense factor-less
    projection has no activation-format seam) and on `act_quant = "none"`.
    `dense_qat` is forwarded to `DenseLinear` (a TSCT projection's weights ARE
    its factors, covered by `factor_qat`; the dense path has no factors to
    quantize).
    """
    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    rank = cfg.rank if rank is None else rank
    if cfg.use_tsct:
        return TSCTLinear(
            in_features,
            out_features,
            rank,
            dtype=dtype,
            bias=bias,
            factor_qat=cfg.factor_qat,
            factor_qat_alpha=cfg.factor_qat_alpha,
            act_quant=cfg.act_quant,
            act_group=cfg.act_group,
            attn_path=attn_path,
            device=device,
        )
    return DenseLinear(
        in_features,
        out_features,
        dtype=dtype,
        bias=bias,
        dense_qat=cfg.dense_qat,
        dense_qat_alpha=cfg.factor_qat_alpha,
        device=device,
    )
