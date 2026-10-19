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
resolves to bf16 factors; the STORAGE rungs (`fp8`, `nvfp4`) exist since
2026-10-15 and dequantize inside the forward (see below); every other rung
(LittleBit semantics) is still refused at CONFIG load (see
`config._refuse_ladder_rung`), so a rung can never silently compute bf16
while the config reads a name it did not implement. The activation side is a
separate field (`config.act_quant`, `model/act_quant.py`), not a rung of this
ladder: the factors stay bf16, only the activations the factor matmuls read are
fake-quantized (with an STE), and the a4.8 attention rule is
`act_quant.attn_format`.

STORAGE rungs vs bf16, the exact mechanics (validated by test):
* master fp32 factors are quantized elementwise to the storage format with
  a per-tensor scale computed from the format's own max (E4M3 = 448,
  e2m1/grid-max = 6) **at state dict save/load time** — the rung is a
  checkpoint-compression device and a memory format for the bank; the fp32
  master floats stay the optimizer's object inside a live run.
* the forward path never changes numerics within a run: the rung pins a
  round trip `fp32 → storage → fp32 ≥` the test tolerance (1e-2 relative),
  which is where the 8-bit/4-bit losses are allowed to live and no further.
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
  the b1.58 ternary / LittleBit-binary / MXFP4 quantizers with burn-spectral's
  annealed STE, applied to every TSCT factor through `factor_qat_project`.
  What the compression tournament MEASURED at 9.2M (3 seeds x 2k steps):
  hard STE from step 0 loses +0.050 BPB (ternary) / +0.087 (binary) to bf16;
  the annealed ramp (alpha 0->1 over 1000 steps) recovers only 0.010 of it
  (+0.040) with a tighter spread — so at THIS scale the arm is CLOSED for
  training (the controls' spread is 0.006). The storage rungs (fp8/nvfp4/
  ternary, checkpoint compression) are untouched by that verdict.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..instrument import Counters, loud

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

#: E4M3's own signed max (`RUNG_MAX["fp8"]`) — the per-tensor scale's divisor.
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


#: The S3T Bernoulli stream (`factor_qat = "ternary_s3t"`): the projector has
#: no step index to seed from, so the mask comes from ONE module-level
#: generator, seeded from `ModelConfig.factor_qat_seed` and ADVANCING across
#: calls. `factor_qat_project`'s docstring owns the full statement of what that
#: costs (a resume restarts the stream); this is the mechanism.
_S3T_SEED = 0
_S3T_GENERATORS: dict[tuple[str, int], torch.Generator] = {}


def set_factor_qat_seed(seed: int) -> None:
    """`ModelConfig.factor_qat_seed`'s one entry point (`DormouseModel.__init__`).

    Clears the stream, so a model BUILD is a stream boundary: the trainer builds
    one model per process (the field is a run property), and a test that builds
    two models with one seed gets the same masks rather than the seconds.
    """
    global _S3T_SEED
    _S3T_SEED = int(seed)
    _S3T_GENERATORS.clear()


def _s3t_generator(device: torch.device | str) -> torch.Generator:
    """The process-local S3T generator for `device` — created on first use.

    Keyed by `(device, seed)` because a `torch.Generator` belongs to one device
    and because two seeds in one process must not share a stream. Never the
    global RNG (the port's rule: an explicit generator, ADR-0021).
    """
    dev = torch.device(device)
    key = (str(dev), _S3T_SEED)
    g = _S3T_GENERATORS.get(key)
    if g is None:
        g = torch.Generator(device=dev)
        g.manual_seed(_S3T_SEED)
        _S3T_GENERATORS[key] = g
    return g


def factor_qat_project(
    w: torch.Tensor, mode: str, alpha: float, *, dim: tuple | None = None
) -> torch.Tensor:
    """The forward's view of a TSCT factor master under `factor_qat`.

    ONE place both consumers (the loop block's `TSCTLinear` and the experts'
    `DualLatentGLU`) route through — the burn-era defect this avoids is two
    copies of one quantizer drifting apart (the tool for that is a grep).

    - `none` → the master untouched (bit-identical to every number on record);
    - `ternary` → b1.58 STE (`ternary.ste_ternary_annealed`, burn-spectral's
      measured "converges better than hard STE" annealing);
    - `binary` → LittleBit sign factors under the SAME annealing:
      `w → w + alpha·(sign_binary(w) − w).detach()` (the annealed STE is the
      one mechanism here; sign vs roundclip is the entire difference);
    - `mxfp4` → the OCP MX e2m1 codec (`mxfp.fake_quant_mx(w, "mxfp4")`: per-32
      e8m0 power-of-two block scales, RNE onto the e2m1 grid, the `tsct.py`
      grid itself) under the SAME annealing. `dim` is IGNORED for this rung and
      that is a claim, not a shortcut: the MX grid is per-32 along the LAST dim
      and `fake_quant_mx` pads PER ROW, so every block sits inside one `(E, d)`
      row and the whole `[E, d, r]` stack quantizes to exactly what E separate
      `[d, r]` slices get — banked == standalone, aligned `r` or padded `r`
      (pinned by `tests/test_factor_qat.py`). A per-slice call would be E
      launches producing the same bits;
    - `ternary_s3t` → S3T stochastic rounding (`ternary.stochastic_ternary`,
      arXiv:2412.04787) under the SAME annealing. `dim` IS honoured here (the
      per-slice absmean scale, exactly as `ternary` above), but the Bernoulli
      MASK cannot be: the bank draws one `[E, ·, ·]` stream and the standalone
      path draws per factor, and no RNG layout makes those two draw identical
      bits. So banked and standalone agree on the SCALE (`α_e` per expert) and
      on the estimator, not bit-for-bit on the mask — the one place in this
      projector where the two paths diverge, said out loud rather than papered
      over.

    THE DETERMINISM OF `ternary_s3t`, stated where a reader will meet it. The
    port's rule (ADR-0021) is a mask that is a pure function of `(seed, step)`.
    This seam has no step index — the projector is called from a forward, and
    the forward does not know which step it is — so the S3T mask comes from ONE
    module-level `torch.Generator`, seeded from `ModelConfig.factor_qat_seed`
    and ADVANCING across calls. Consequences, both honest:
      * within a run it is reproducible: same seed, same call layout, same bits
        (the trainer builds exactly one model per process);
      * a RESUMED run does NOT continue the mask stream — it starts a fresh
        generator at the same seed — so a resumed run's masks differ from an
        unbroken one's. That degradation is COUNTED, not silent: the trainer
        prints it at start (`train/loop.py`) and `set_factor_qat_seed` carries
        it too. The fix, if this rung ever needs one, is a step index threaded
        into the projector;
      * the banked path refreshes the quantized stack once per optimizer step
        (`SparseMoE._banks`' cache key), so there the mask is drawn per step,
        not per forward.

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
    """
    if mode == "none":
        return w
    if mode not in ("ternary", "binary", "mxfp4", "ternary_s3t"):
        raise AssertionError(f"factor_qat/dense_qat={mode!r} reached the projector with no mode")
    if mode == "mxfp4":
        from .mxfp import fake_quant_mx

        q = fake_quant_mx(w, "mxfp4")
        return (w + alpha * (q - w).detach()).to(w.dtype)
    if mode == "ternary_s3t":
        from .ternary import _check_finite, stochastic_ternary

        if dim is None:
            q = stochastic_ternary(w, _s3t_generator(w.device))
        else:
            # The per-slice S3T of `stochastic_ternary`, transposed onto `dim`
            # (see its docstring: `p = clamp(|w|/scale, 0, 1)`, STRICT `p > u`),
            # with the one whole-tensor draw the bank's layout gives. The finite
            # guard is the module's own, so both paths refuse an inf/NaN master
            # identically instead of one of them zeroing it silently.
            _check_finite(w, "factor_qat=ternary_s3t (bank slice)")
            a = w.float().abs().mean(dim=dim, keepdim=True).clamp_min(1e-12)
            keep_p = (w.float() / a).abs().clamp(0.0, 1.0)
            u = torch.rand(
                keep_p.shape,
                generator=_s3t_generator(w.device),
                dtype=torch.float32,
                device=w.device,
            )
            q = (w.float().sign() * a * (keep_p > u)).to(w.dtype)
        return (w + alpha * (q.to(w.dtype) - w).detach()).to(w.dtype)
    if dim is None:
        from .ternary import sign_binary, ste_ternary_annealed

        if mode == "ternary":
            return ste_ternary_annealed(w, alpha)
        return w + alpha * (sign_binary(w) - w).detach()
    # The per-slice axis (banks): the same formula as `ternary.ternarize` /
    # `ternary.sign_binary`, transposed onto `dim` — the grid is ±mean(|W|),
    # ε only guards the division (exactly `ternarize`'s convention: divide by
    # `α+ε`, scale back by `α`), so a banked expert and the same expert run
    # alone land on the SAME values.
    a = w.abs().mean(dim=dim, keepdim=True)
    if mode == "ternary":
        q = (w.float() / (a + 1e-5)).round().clamp(-1.0, 1.0) * a
    else:
        q = torch.sign(w) * a
    return (w + alpha * (q.to(w.dtype) - w).detach()).to(w.dtype)


# ---------------------------------------------------------------------------
# The DENSE path: the same projector, applied to plain `[out, in]` weights
# ---------------------------------------------------------------------------

#: The dense path's projector — the SAME function object as
#: [`factor_qat_project`], deliberately. What the owner's final-quantized
#: recipe needs on a plain linear weight IS the factor rung's semantics
#: (per-tensor absmean `ternarize` / LittleBit `sign_binary` / the MX e2m1
#: codec, all under the annealed STE `w + alpha·(q(w) − w).detach()`, with
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


def qat_linear(linear: nn.Linear, x: torch.Tensor, kind: str, alpha: float) -> torch.Tensor:
    """`linear(x)` with `dense_qat` applied to its weight — the master when off.

    The ONE call shape for every plain `nn.Linear` site (KDA's q/k/v/g/beta,
    the Engram's three projections, MSA's, DSpark's). `F.linear` IS
    `nn.Linear.forward`'s whole body, so `kind = "none"` returns `linear(x)`
    itself — bit-identical to the call it replaces, no copy.
    """
    if kind == "none":
        return linear(x)
    w = dense_qat_project(linear.weight, kind, alpha)
    return F.linear(x, w, linear.bias)


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
        self.u = nn.Parameter(
            torch.randn(in_features, rank, device=device) / math.sqrt(in_features)
        )
        self.s = nn.Parameter(torch.ones(rank, device=device))
        self.v = nn.Parameter(torch.randn(rank, out_features, device=device) / math.sqrt(rank))
        self.bias = nn.Parameter(torch.zeros(out_features, device=device)) if bias else None
        #: The compressed-forward sprint's switch (`factor_qat_project`): the
        #: forward reads quantized factors with STE while the masters stay
        #: fp32. `s` is never quantized — it IS the scale.
        self.factor_qat = factor_qat
        self.factor_qat_alpha = factor_qat_alpha
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
        return f(self.u, self.factor_qat, self.factor_qat_alpha), f(
            self.v, self.factor_qat, self.factor_qat_alpha
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

#: The format's own signed max, by rung: E4M3 fn grid top = 448; NVFP4's e2m1
#: grid top = 6 (0, .5, 1, 1.5, 2, 3, 4, 6 — LittleBit's grid, which is why the
#: sign rung maps onto it exactly).
RUNG_MAX = {"fp8": 448.0, "nvfp4": 6.0}

#: The matrix-parameter names that ARE factor matrices (everything else —
#: scales `.s`, router, engram, norms — stays fp32/goes bf16 untouched).
_FACTOR_SUFFIXES = ("u", "v", "gate_u", "up_u")

#: NVFP4 packing constants: ITU e2m1, magnitude grid ascending then two's-compl
#: sign per code; two codes per byte, low nibble first.
_NVFP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

#: COUNTED tally of NVFP4 blocks whose stored e4m3 scale rounded to zero —
#: process-local so a `pack_tsct` call with no counters is still not silent.
_NVFP4_ZEROED_BLOCKS = {"blocks": 0}


def nvfp4_zeroed_blocks() -> int:
    """Blocks whose NVFP4 scale rounded to zero (the COUNTED tally, no counters)."""
    return _NVFP4_ZEROED_BLOCKS["blocks"]


def _e2m1_encode(t: torch.Tensor, global_scale: float) -> tuple[torch.Tensor, torch.Tensor, int]:
    """One NVFP4 encoding pass at `global_scale`: (codes, e4m3 scales, zeroed blocks)."""
    blocks = (t * global_scale).float().reshape(-1, 16)
    amax = blocks.absolute().amax(dim=1)
    scale = amax / 6.0
    # A block of exact zeros scales to 0 → decode has to multiply by 1, not 0.
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    # The scale itself is stored e4m3 (NVIDIA's NVFP4 layout): quantize it too.
    s_fp8 = scale.to(torch.float8_e4m3fn)
    zero_blocks = int((s_fp8.float() == 0).sum())
    q = blocks / s_fp8.float().unsqueeze(-1)
    # Nearest-magnitude rounding: searchsorted over the MIDPOINTS of adjacent
    # grid entries (bucketize alone rounds "≤ boundary", not nearest).
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=t.device)
    mag_idx = torch.searchsorted(midpoints, q.absolute().reshape(-1)).reshape(blocks.shape)
    sign = (q < 0) & (mag_idx > 0)  # zero codes carry no sign
    codes = (sign.to(torch.int32) * 8 + mag_idx).reshape(-1)  # sign<<3 | mag, flat
    packed = (codes[0::2] | (codes[1::2] << 4)).to(torch.uint8)
    return packed, s_fp8, zero_blocks


def _e2m1_quant(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    """Per-16-block NVFP4 quantize: codes u8 + e4m3 scales [n/16] + a global
    fp32 correction + the number of blocks whose stored scale rounded to zero.

    NVIDIA's NVFP4 is TWO-LEVEL for exactly the underflow this would hit:
    a block whose e4m3 scale rounds to zero is lost entirely. The e4m3 scale
    grid is OCP *Microscaling Formats (MX) Specification* v1.0 §5.3.1 Table 2
    (bias 7: max normal ±448, min normal ±2⁻⁶, min subnormal ±2⁻⁹) — RNE of a
    scale at or below `2^-10` (half the min subnormal) is the zero code, and
    `torch.finfo(...).min` is NOT an exponent: it is −448.0, so `2 ** min` =
    `2^-448` made the old prescale inert (2026-10-19 audit M1). `finfo.tiny`
    (2⁻⁶) is the min NORMAL, not the zeroing boundary — a 2⁻⁶ scale is
    representable — so the retry below is driven by the encode's actual zero
    count, and the target is the MAX (448): keeping the large blocks
    representable is the choice, and blocks left below the grid are counted.

    Prescale: encode once, and if any nonzero block's scale rounded to zero,
    encode again with a power-of-two global that puts the LARGEST block scale
    at e4m3's max (448; floored, so it can never round up to NaN). The global
    comes back on decode. Aiming the LARGEST at the top — not the smallest at
    the min normal — is what one global multiplier can do: a tensor spanning
    more than e4m3's ~18 octaves cannot carry both ends, the large blocks are
    its mass, and the blocks still below the grid are returned in the fourth
    slot for `pack_tsct` to COUNT. The global is 1.0 whenever no block would
    zero, so the common path's record is byte-identical.

    Encode = quotient by the block scale then nearest grid magnitude (the
    grid is LittleBit's, so a later sign rung re-uses the same codes).
    """
    n = t.numel()
    assert n % 16 == 0, f"nvfp4 quantizes blocks of 16, got numel={n}"
    packed, s_fp8, zero_blocks = _e2m1_encode(t, 1.0)
    if not zero_blocks:
        return packed, s_fp8, 1.0, 0
    amaxs = t.float().abs().reshape(-1, 16).amax(dim=1)
    scale_max = float(amaxs.max()) / 6.0
    exponent = torch.floor(torch.log2(torch.tensor(RUNG_MAX["fp8"] / scale_max)))
    global_scale = float(2.0**exponent)
    packed, s_fp8, zero_blocks = _e2m1_encode(t, global_scale)
    return packed, s_fp8, global_scale, zero_blocks


def _e2m1_dequant(packed: torch.Tensor, scale: torch.Tensor, global_scale: float) -> torch.Tensor:
    lows = packed.to(torch.int32) & 0x0F
    highs = (packed.to(torch.int32) >> 4) & 0x0F
    codes = torch.stack((lows, highs), dim=-1).reshape(-1)
    mags = torch.as_tensor(_NVFP4_MAGNITUDES, device=packed.device)
    vals = mags[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    vals = vals[: scale.numel() * 16].reshape(-1, 16)
    return (vals.float() * scale.float().unsqueeze(-1)).reshape(-1) * (1.0 / global_scale)


def _ternary_quant(t: torch.Tensor) -> tuple[torch.Tensor, float]:
    """b1.58 ternary projection → 2-bit codes + the α scalar (16× vs fp32).

    Stores the TERNARY PROJECTION of the master (`ternary.ternarize`, the same
    formula `factor_qat` trains against): codes 0/1/2 for {0, +α, −α}, four to
    a byte, α in fp32 beside them. Round-trip is bit-exact to `ternarize(t)`
    for finite t — a storage rung may lose precision by its FORMAT'S OWN grid
    (nvfp4 stores 4-bit values), never by an extra rounding.
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
    "fp8": None,  # native dtype; no packer needed
    "nvfp4": (_e2m1_quant, _e2m1_dequant),
    "ternary": (_ternary_quant, _ternary_dequant),
}


def pack_tsct(t: torch.Tensor, rung: str, counters: Counters | None = None) -> dict[str, object]:
    """One factor matrix → its storage record. Scales travel COMPUTED.

    `counters` is where the NVFP4 rung reports blocks whose stored e4m3 scale
    rounded to zero (`nvfp4_zero_scale_blocks`, COUNTED — such a block decodes
    to zeros and a reader must see it). The process-local
    [`nvfp4_zeroed_blocks`] tally makes the report unconditional for a caller
    that passes none.
    """
    if t.dtype != torch.float32:
        loud(f"storage rungs quantize the fp32 masters, got dtype {t.dtype}")
    if not torch.isfinite(t).all():
        loud("storage: master factors must be finite (inf/NaN refuse, they encode wrong)")
    if rung == "fp8":
        scale = t.abs().amax() / RUNG_MAX["fp8"]
        scale = float(scale) if scale != 0 else 1.0
        packed = (t / scale).to(torch.float8_e4m3fn)
        return {"rung": rung, "packed": packed.reshape(t.shape), "scale": scale, "shape": t.shape}
    if rung == "nvfp4":
        packed, scale, correction, zero_blocks = _e2m1_quant(t)
        if zero_blocks:
            _NVFP4_ZEROED_BLOCKS["blocks"] += zero_blocks
            if counters is not None:
                counters.count("nvfp4_zero_scale_blocks", zero_blocks)
        return {
            "rung": rung,
            "codes": packed,
            "scale": scale,
            "global_scale": correction,
            "shape": t.shape,
            "n": t.numel(),
        }
    if rung == "ternary":
        packed, alpha = _ternary_quant(t)
        return {
            "rung": rung,
            "codes": packed,
            "alpha": alpha,
            "shape": t.shape,
            "n": t.numel(),
        }
    loud(f"pack_tsct: rung {rung!r} is not a storage rung (fp8|nvfp4|ternary)")


def unpack_tsct(rec: dict[str, object]) -> torch.Tensor:
    """Inverse of [`pack_tsct`] — master-precision dtype, rung-rounded values."""
    rung = rec["rung"]
    if rung == "fp8":
        # Encode was `packed = (t / scale)`; decode multiplies back up.
        return rec["packed"].float() * rec["scale"]
    if rung == "nvfp4":
        packed, scale = rec["codes"], rec["scale"]
        t = _e2m1_dequant(packed, scale, rec.get("global_scale", 1.0))
        return t.reshape(rec["shape"])
    if rung == "ternary":
        return _ternary_dequant(rec["codes"], rec["alpha"], rec["n"]).reshape(rec["shape"])
    loud(f"unpack_tsct: rung {rung!r} is not a storage rung (fp8|nvfp4)")


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
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device=device) / math.sqrt(in_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features, device=device)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = dense_qat_project(self.weight, self.dense_qat, self.dense_qat_alpha)
        if self.compute_dtype != torch.float32:
            x = x.to(self.compute_dtype)
            w = w.to(self.compute_dtype)
        y = x @ w.t()
        if self.compute_dtype != torch.float32:
            y = y.to(torch.float32)
        if self.bias is not None:
            y = y + self.bias
        return y


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
