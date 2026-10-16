r"""The BitNet ternary weight family, as pure torch tensor math (CPU, any device).

RECOVERY, not new design: this machinery lived in the burn tree and did not
travel with the torch port. Every formula below is transcribed from the burn
source named beside it; the burn crate docs carry the papers, the port carries
them forward (AGENTS §1.4: a claim names the file it came from).

| function | burned source | paper |
|---|---|---|
| `ternarize` | `weights.rs:19-35` | b1.58 arXiv:2402.17764 |
| `sign_binary` | `lib.rs:49-56`, dead zone dropped | LittleBit arXiv:2506.13771 |
| `ste_ternary` | `lib.rs:58-63` | — (an estimator) |
| `ste_ternary_annealed` | `lib.rs:65-82` | — (burn's schedule) |
| `stochastic_ternary` | `lib.rs:84-106` | arXiv:2412.04787 (S3T) |
| `activation_quant_8bit` | `activations.rs:9-24` | b1.58 arXiv:2402.17764 |

Paths: `lib.rs` = `vendor/dormouse-fused/crates/burn-spectral/src/lib.rs`;
`weights.rs` = `.../burn-bitnet/src/quant/weights.rs`; `activations.rs` =
`.../burn-bitnet/src/quant/activations.rs`. BitNet v2 (arXiv:2504.18415 Eq. 1)
and the 2B4T report (arXiv:2504.12285) describe the same b1.58 scheme.

WHAT IS **NOT** RECOVERED (each had a home in burn, none is wired here — YAGNI
until a lane needs it, and the caller can say which one it needs):

* N:M sparse ternary (`weight_quant_ternary_nm`, weights.rs:75-85) — Sparse-BitNet
  arXiv:2603.05168, Dual-STE mask. The name and its paper id are the whole of
  what is carried here.
* The dead-zone ternary (`burn-spectral/src/lib.rs:51-56`, its own comment says
  "no external reference exists for this threshold") and the per-column variant
  (`:115-124`) — `ternarize` is the paper form instead, see below.
* The 2-bit 5-level quantizer (`weight_quant_2bit`, weights.rs:48-73) and the
  BitNet v2 Hadamard/4-bit activation path (`bitnet_v2_quantize`,
  activations.rs:26-99).

THE NUMERIC DECISIONS, each with its source (LOUD/COUNTED doctrine: a decision
that is ours says so):

* **`ternarize` is PER-TENSOR.** burn's b1.58 quantizer takes `mean()` over the
  whole tensor (`weights.rs:22-27`, `γ = mean(|W|)`), unlike `weight_quant_2bit`
  which is per-ROW (`weights.rs:65`, `mean_dim(1)`) and unlike burn-spectral's
  `ternarize_per_column` (`lib.rs:119-124`). Matching the source: per-tensor.
* **ε = 1e-5, as `W/(α+ε)`.** The papers' RoundClip denominator guard, and
  burn's `clamp_min(1e-5)` value (`weights.rs:27`: "γ + ε guard from the
  papers"). The two spellings agree for `α ≫ ε`; on a near-zero tensor this form
  keeps the output at ±α while burn's clamp raises the levels to ±1e-5 — the
  literal form is kept because no scale inflation is the safer end of a
  degenerate input.
* **`ternarize` is a pure projection; the STE is a separate function.** burn's
  `weight_quant_b158` returns `w + (q - w.detach())` (`weights.rs:34`), which in
  the forward is exactly `q`. The split here (`ternarize` → values,
  `ste_ternary` → values + identity backward) is the same object with one fewer
  copy for a caller that only wants the codebook.
* **The dead-zone/difference.** `sign_binary` is `sign(w)·mean(|w|)` with NO
  zero level, i.e. burn-spectral's `ternarize` with the `keep` factor dropped:
  binary ±α, not ternary. `ternarize` CAN return 0 and does (`round` before the
  clamp); at `|w| = 0.3α` they disagree by exactly one α, which is pinned in
  `tests/test_ternary.py` so a refactor cannot alias them silently.
* **S3T scale is `mean(|w|)` per-TENSOR** (`burn-spectral/src/lib.rs:96`), with
  `p = clamp(|w|/scale, 0, 1)` and a STRICT `p > u` (`:98-104`), so an element at
  the scale is ALWAYS kept (analytic, not statistical — that is the
  "deterministic above the scale" property). The divider is guarded by
  `clamp_min(1e-12)` instead of burn's unguarded `into_scalar` (`:96-97`): an
  all-zero tensor must quantize to zeros, not to NaN.
* **`activation_quant_8bit` is per-TOKEN absmax** — `max` over the last dim,
  i.e. burn's `reshape([b·t, d])` then `max_dim(1)` (`activations.rs:12-13`),
  which generalizes to "reduce the last dim, every leading index is its own
  token". Its scale guard is the same `clamp_min(1e-5)` (`:13`).
* **dtype.** The masters stay fp32 (the burn and port rule: fp32 masters, casts
  only at the arm boundary). Every function here computes in fp32 and returns
  the INPUT's dtype — the "docstring hole" the burn port left on the activation
  side, stated outright: the absmax scale is fp32, the codes are int8 in
  [-127, 127] (the `clamp(-128, 127)` at `activations.rs:18` is a defensive
  bound; the scale maps onto ±127, so -128 is unreachable), and the returned
  tensor is the DEQUANTIZED value (fp32, or the input dtype under the STE).
* **Non-finite input is LOUD**, here on every entry point: a NaN poisons the
  absmean/absmax scale and silently turns the whole tensor into NaN. The escape
  is named in the message (sanitize upstream — the NaN firewall is the
  trainer's, AGENTS §1.1).
* **`activation_quant_8bit` keeps burn's STE residual** (`activations.rs:22-23`):
  forward = the quantized value, backward = identity. burn's own `assert` on
  the STE gradient is in `weights.rs`'s test module, not in the forward.

Bundle info: `packed_bits_per_weight` is the information-theoretic bits/weight
(`log2(3)` ternary, `1.0` binary) and the comment under it is the byte argument
for a b1 expert bank — the number a VRAM plan needs, not a codec (this module
has no packing path; `model/mxfp.py` is the one that packs, for MX formats).
"""

from __future__ import annotations

import math

import torch

from ..instrument import loud

#: ε of `W̃ = α·RoundClip(W/(α+ε), −1, 1)` — same 1e-5 burn clamps the scale
#: with (`quant/weights.rs:27`, `quant/activations.rs:13`).
_EPS = 1e-5

#: S3T's divider guard: `α = 0` (an all-zero tensor) must quantize to zeros.
_S3T_EPS = 1e-12


def _check_finite(t: torch.Tensor, what: str) -> None:
    """inf/NaN refuse loudly: they poison the scale this module divides by.

    `absmean`/`absmax` of a tensor holding one NaN is NaN, so the scale would be
    NaN and every output element would be NaN — a silent corruption with an
    obvious cause. A non-float tensor is refused for the same reason (an integer
    master cannot be cast to the w/(α+ε) quotient meaningfully).
    """
    assert t.is_floating_point(), f"{what}: masters are float tensors, got {t.dtype}"
    if not bool(torch.isfinite(t).all()):
        n_bad = int((~torch.isfinite(t)).sum())
        loud(
            f"{what}: {n_bad} non-finite element(s) (inf/NaN). Cause: the tensor "
            "is already corrupt, and every quantizer here divides by an "
            "absmean/absmax it would poison — one NaN makes the whole tensor "
            "NaN with nothing said. Escape: sanitize the tensor upstream (the "
            "NaN firewall is the trainer's job, AGENTS §1.1) or mask the bad "
            "elements before quantizing"
        )


def _alpha(w: torch.Tensor) -> torch.Tensor:
    """`α = mean(|W|)` as an fp32 scalar tensor — the b1.58 absmean."""
    return w.float().abs().mean()


def ternarize(w: torch.Tensor) -> torch.Tensor:
    """b1.58 ternary projection: `α·RoundClip(W/(α+ε), −1, 1)`, `α = mean(|W|)`.

    **PER-TENSOR** absmean (`quant/weights.rs:22-27`), the paper's Eq. 1
    (b1.58 arXiv:2402.17764; identical scheme in BitNet v2 arXiv:2504.18415
    Eq. 1 and 2B4T arXiv:2504.12285). `round` before the clamp is what puts
    `|w| < (α+ε)/2` on 0 and keeps the surviving magnitudes at exactly ±α
    instead of collapsing outliers onto ±α from further out.

    Pure projection: no STE, no graph trick — `ste_ternary` is the estimator.
    Returns the input's dtype (fp32 in, fp32 out; bf16 in, bf16 out).
    """
    _check_finite(w, "ternarize")
    alpha = _alpha(w)
    q = (w.float() / (alpha + _EPS)).round().clamp(-1.0, 1.0) * alpha
    return q.to(w.dtype)


def sign_binary(w: torch.Tensor) -> torch.Tensor:
    """`sign(W)·mean(|W|)` — binary ±α, NOT a ternary (no zero level).

    The `keep` factor is dropped from burn-spectral's projection
    (`burn-spectral/src/lib.rs:51-56`, whose dead zone at `0.7·mean` has, in
    burn's own words, "no external reference"): every weight becomes ±α, which
    is the LittleBit binary rung (arXiv:2506.13771) and `packed_bits_per` 1.0.
    `w == 0` gives 0 (torch's `sign(0) = 0`), so a zero master stays zero.

    THE DIFFERENCE from [`ternarize`], and it is a test: at `|w| = 0.3·α`,
    `ternarize` returns 0 (0.3 rounds to 0) while this returns `+α`.
    """
    _check_finite(w, "sign_binary")
    scale = _alpha(w)
    return (w.float().sign() * scale).to(w.dtype)


def ste_ternary(w: torch.Tensor) -> torch.Tensor:
    """`w + (tern(w) − w).detach()` — forward ternary, backward identity.

    Transcribed verbatim from `burn-spectral/src/lib.rs:58-63`: the forward
    value IS `ternarize(w)`, while `d/dw` is exactly 1 everywhere (the detached
    residual is a constant), so the fp32 master trains through a ternary
    forward. burn's paper is b1.58; the estimator itself is standard STE.
    """
    t = ternarize(w)
    return w + (t - w).detach()


def ste_ternary_annealed(w: torch.Tensor, alpha: float) -> torch.Tensor:
    """`w + alpha·(tern(w) − w).detach()` — soft-to-hard ternary annealing.

    burn-spectral/src/lib.rs:65-82, with its two endpoints: `alpha = 1` is the
    plain STE (pure ternary forward), `alpha = 0` is the untouched master.
    Ramping 0 → 1 over training is the schedule burn measured as converging
    better than hard STE from step one (lib.rs:74-77: "which converges better
    than hard STE from step one (measured)") — burn's measurement, carried here
    as theirs, not re-measured by the port.

    `alpha` is a schedule position in [0, 1] and is refused outside it (burn's
    `set_alpha` clamps silently at `lib.rs:597-599`; a clamped schedule is the
    ADR-0011 silent degradation, so this refuses instead).
    """
    assert 0.0 <= alpha <= 1.0, (
        f"annealing alpha={alpha} is a schedule position in [0, 1] "
        "(1 = pure ternary, 0 = the master); refused, not clamped"
    )
    t = ternarize(w)
    return w + alpha * (t - w).detach()


def stochastic_ternary(w: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """S3T: `sign(w)·scale·Bernoulli(|w|/scale)`, `scale = mean(|w|)` (per-tensor).

    burn-spectral/src/lib.rs:84-106 (`ternarize_stochastic`), stochastic
    rounding QAT (arXiv:2412.04787): each element keeps its sign with
    probability `min(|w|/scale, 1)` and is zeroed otherwise, so `E[w_t] = w`
    wherever `|w| ≤ scale` — the forward is ternary with the master's expected
    value and the Bernoulli noise regularizes. Elements AT or ABOVE the scale
    are kept deterministically (`p = 1 > u`), which is the difference from a
    dead zone to tune: there is none.

    `generator` is REQUIRED for reproducibility: the repo's determinism rule
    (ADR-0021) is an explicit generator, never a global RNG — same seed, same
    bits, which is what makes the training path reproducible.
    """
    _check_finite(w, "stochastic_ternary")
    wf = w.float()
    scale = wf.abs().mean().clamp_min(_S3T_EPS)
    keep_p = (wf / scale).abs().clamp(0.0, 1.0)
    u = torch.rand(keep_p.shape, generator=generator, dtype=torch.float32, device=wf.device)
    keep = (keep_p > u).to(wf.dtype)
    return (wf.sign() * scale * keep).to(w.dtype)


def activation_quant_8bit(x: torch.Tensor) -> torch.Tensor:
    """b1.58 activation side: per-TOKEN absmax → 8-bit signed, STE backward.

    burn-bitnet/src/quant/activations.rs:9-24 (`activation_quant_8bit`): the
    tensor is flattened to `[tokens, d]` and every token is scaled by its own
    absmax, `q = clamp(round(x/scale·127), -128, 127)/127·scale`. The scale maps
    onto ±127, so the reachable code range is `[-127, 127]` and the -128 clamp is
    a defensive bound only. `b1.58`'s paper is arXiv:2402.17764 (quantize the
    activations to 8 bits, the weights to ternary).

    Two honesties the burn port's docstring left open: the absmax scale is fp32
    and the RETURNED tensor is the dequantized value in the input's dtype (fp32
    masters are never destroyed), and a non-finite input refuses loudly (a NaN
    absmax would make the whole token NaN). The residual trick
    (`activations.rs:22-23`) keeps burn's STE: forward = quantized, backward =
    identity.
    """
    _check_finite(x, "activation_quant_8bit")
    assert x.dim() >= 1, "activation_quant_8bit needs a last dim to scale per token"
    xf = x.float()
    scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(_EPS)
    codes = (xf / scale * 127.0).round().clamp(-128.0, 127.0)
    assert bool((codes.abs() <= 128.0).all()), "8-bit codes are in [-128, 127]"
    q = (codes / 127.0 * scale).to(x.dtype)
    return x + (q - x).detach()


def packed_bits_per_weight(levels: int = 3) -> float:
    """Bits/weight of a symmetric level codebook: `log2(levels)`.

    Ternary `{−1, 0, +1}` (the default) ⇒ **log2(3) = 1.585** bits/weight;
    binary `{−1, +1}` (`levels=2`, [`sign_binary`]) ⇒ **1.0**. This is the
    information bound, not a format: a real packer rounds up (4 ternary weights
    per byte is the cheap 2-bit scheme, 5 per byte at 3⁵ = 243 ≤ 256 is the
    base-3 one), so quoting it next to a byte count needs the packing rule too.
    """
    assert levels >= 2, f"a level codebook has >= 2 levels, got {levels}"
    return math.log2(levels)


# The byte argument for a b1 expert bank, the number a VRAM plan needs:
# `[1024, 64, 5461]` = 1024 experts x 64 rows x 5461 cols = 357 892 096 weights.
#   bf16 (2 B/weight) .................... 715 784 192 B = 682.6 MiB
#   fp32 (4 B/weight) .................... 1 431 568 384 B = 1.33 GiB
#   2 bits (4 weights/byte) .............. 89 473 024 B = 85.3 MiB = 7.998x under bf16
#   + per-row fp32 scales (65536 rows) ... 262 144 B (0.29 % of the packed size)
#                                          -> 89 735 168 B = 7.977x under bf16
#   base-3 (5 weights/byte, log2(3)) ..... ceil(357892096/5) = 71 578 420 B
#   + the same scales .................... 71 840 564 B = 9.96x under bf16
# The scale overhead is named because it is the one part that does NOT shrink:
# per-ROW (weight_quant_2bit, weights.rs:65) is what the count above assumes; a
# per-TENSOR scale (what `ternarize` uses) is 1024 x 4 B and vanishes.
