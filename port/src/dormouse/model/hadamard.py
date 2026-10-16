"""BitNet v2's Fast Walsh-Hadamard rotation, as pure tensor math (arXiv:2504.18415).

NEW module, not wired into any model yet (a later lane wires it into the
quantization path): this is the outlier-suppressing rotation BitNet v2
(arXiv:2504.18415) puts in front of its INT4 activation quantizer, and the
machinery the burn tree had and the torch move dropped.

FORMULA SOURCE — `vendor/dormouse-fused/crates/burn-bitnet/src/fwt.rs` (the
FWHT) and `.../src/quant/activations.rs` (the joint quantization), both read
2026-10-16. Every rule below cites the burn line it transcribes:

* **The butterfly is unnormalized, the `1/sqrt(p)` is applied ONCE at the
  end** (`fwt.rs:113-141`, the loop at 115-134 then
  `out.div_scalar((p as f32).sqrt())` at `:135`; the CUDA kernel does the same,
  `fwt_cuda.rs:134-138`). The reference path is the literal matrix form
  `y = x_pad @ H_p / sqrt(p)` (`fwt.rs:93`).
* **WHERE the normalization lives is load-bearing**, and this module puts it
  INSIDE [`fwht`], applied on every call. `H_p` is symmetric with
  `H_pᵀ H_p = p·I`, so `A = H_p / sqrt(p)` is orthogonal (`Aᵀ A = I`) and
  `fwht(fwht(x)) = x`. Both directions of `v2_quantize` (rotate, then rotate
  back — the transform is its own inverse, `activations.rs:36`/`:94`) call the
  SAME function, so the constant has to be in that one function. Normalizing
  only the forward direction would leave the round trip scaled by `p` — which
  is why the constant is not a caller's choice here. (Applying `1/sqrt(2)` per
  butterfly stage instead of once at the end is algebraically identical —
  `2^(log2 p / 2) = sqrt(p)` — and only changes fp rounding; the burn code
  normalizes once at the end and so does this.)
* **The rotation is exact-invertible; only the quant is lossy**
  (`activations.rs:30-99`, the `deq_rot` of the 4-bit and 8-bit arms is
  re-rotated and added straight-through).
* **The scale maps the token max onto the FORMAT's max, not onto 1**
  (`crates/dormouse-core/src/act_quant.rs:54-59` `ActFormat::max_value` and
  `:88` `scale = block_max / fmax`). The level count is `2^(bits-1)-1`
  (int4 → 7, int8 → 127, `act_quant.rs:97`). Scaling a block's max onto 1 is
  the bug that left the e2m1 grid with 3 of its 8 magnitudes reachable
  (AGENTS §3.7 act-quant); this module must not repeat it, and
  `v2_quantize`'s level convention test pins it.
* **Per-token absmax over the FEATURE dim** (`activations.rs:35` reshapes
  `[b, t, d]` to `[b*t, d]` and the scale is `.abs().max_dim(1)` — a max over
  the feature dim, i.e. one scale per token/row of the last dim). The burn
  4-bit arm used `.abs().mean_dim(1)` (absmean) and its comment says absmean is
  safe *because* the rotation suppressed the outliers; the port takes the
  a4.8 INT level convention instead (task spec, and the reason the level test
  exists), so `v2_quantize` uses absmax for both 4 and 8 bits.
* **Straight-through** is `x + (deq_rot − x).detach()` (`activations.rs:97-98`),
  the same STE shape `act_quant` uses (`act_quant.rs:103-104`) and the same
  object the MX codec's `fake_quant_*` uses.

MEASURED, this box, torch 2.13 CPU, 2026-10-16 (seed 0):

| what | measurement |
|---|---|
| `fwht(fwht(x))` vs `x`, fp32, `[4, p]`, p = 32…4096 | max abs err ≤ **5.4e-7** (≤ 1.4e-7 rel) |
| per-row L2 norm, one `fwht`, fp32 | ≤ 7.6e-6 abs (≤ 1.2e-7 rel) |
| bf16 `fwht(fwht(x))` | ≤ 8.9e-3 rel ≈ 2.3 bf16 eps; rms ≤ 8.7e-3; norm ≤ 2.3e-3 |
| steep probe, `randn [1,64]`, `x[0,3]=1000` | max abs 1000.0 → **126.6**; kurtosis 62.0 → **1.0** |
| same probe at p=256 | max abs 1000.0 → **65.3**; kurtosis 253.9 → **1.0** |
| `v2_quantize` int4 rms err / rms(x) | **0.107** (max rel 44.8 is a near-zero element) |
| `v2_quantize` int8 on the same | **0.0054** |

**bf16 policy: SUPPORTED and dtype-preserving.** The round trip holds to
≤ 8.9e-3 relative — about 2.3 bf16 eps (2⁻⁸) — and the norm to ≤ 2.3e-3, which
is useful (it is within ~2 ULP of the artefact's own precision, and the
rotation's job is to spread the outlier, which it does at any dtype). A LOUD
refusal was the alternative; it is not taken, because refusing the dtype the
trainer actually runs under `--bf16` would push that lane back to an fp32
activation, and the measured error is at the dtype's own floor rather than
above it. `fwht` never down-casts: in bf16 it computes in bf16 and returns
bf16. `v2_quantize` promotes to fp32 for the quant (int levels want fp32
levels) and returns the input dtype.
"""

from __future__ import annotations

import math

import torch

from ..instrument import loud

#: The paper this rotation comes from (arXiv:2504.18415, BitNet v2).
PAPER = "arXiv:2504.18415"

#: `hadamard_matrix` materializes `p²` floats, and exists only for tests: at
#: 4096 it is already 64 MiB. Above this cap it refuses rather than allocate.
_HADAMARD_MAX_P = 4096


def _levels(bits: int) -> int:
    """Symmetric integer level count `2^(bits-1) - 1`: int4 → 7, int8 → 127.

    The a4.8 convention (`crates/dormouse-core/src/act_quant.rs:54-59`
    `ActFormat::max_value`, `:97`). The scale below is `token_max / levels`, so
    the row max lands ON the format's max level; using `token_max` alone
    (normalizing to ±1) is the defect AGENTS §3.7 act-quant records.
    """
    return (1 << (bits - 1)) - 1


def _check(x: torch.Tensor, what: str) -> None:
    """Float, `[..., 2^k]`, finite — or LOUD naming cause and escape."""
    if not x.is_floating_point():
        loud(
            f"FWHT is float math, got {x.dtype} for {what}: cast to fp32 or bf16 first "
            "(the butterfly is add/sub over a row, there is no integer path)"
        )
    if x.dim() == 0:
        loud(
            f"FWHT rotates the LAST dim and {what} is 0-dim (no last dim): pass a [..., 2^k] tensor"
        )
    p = x.shape[-1]
    if p == 0 or (p & (p - 1)) != 0:
        loud(
            f"FWHT needs a last dim of 2^k, got {p} for {what} (shape {tuple(x.shape)}): "
            f"pad the last dim up to {1 << (p - 1).bit_length()} (zero pad — that is what "
            f"fwt.rs:105-112 does) or pick a power-of-two width"
        )
    if not bool(torch.isfinite(x).all()):
        n = int((~torch.isfinite(x)).sum())
        loud(
            f"FWHT: {what} has {n} non-finite element(s); an inf/NaN spreads across the "
            "whole row under the butterfly, so every entry comes back corrupt. "
            "Escape: sanitize upstream (the NaN firewall belongs in the training step, "
            "AGENTS §1.1) or mask the bad elements"
        )
    assert p >= 1 and (p & (p - 1)) == 0, "checked above"


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard transform over the LAST dim, normalized by `1/sqrt(p)`.

    `p = x.shape[-1]` must be a power of two; the transform is `A = H_p / sqrt(p)`
    with `A` symmetric and orthogonal, so `fwht(fwht(x)) == x` to fp32 round-off
    (≤ 5.4e-7 for p ≤ 4096, measured; see the module doc). It is the rotation
    BitNet v2 applies before activation quantization and again to rotate back
    (arXiv:2504.18415; `fwt.rs:101-141` for the butterfly, `:135` for the
    normalization, `fwt_cuda.rs:119-137` for the kernel's identical shape).

    dtype-preserving: bf16 in → bf16 out, computed in bf16 (round-trip error
    ≤ 2.3 bf16 eps, measured). On fp32 CPU this is bit-exact arithmetic per
    stage, so the only error is the single `1/sqrt(p)` scaling per application.
    """
    _check(x, "fwht input")
    p = x.shape[-1]
    out = x
    h = 1
    while h < p:
        g = p // (2 * h)
        # radix-2 butterfly: pairs (a, b) at stride h become (a+b, a-b); the
        # reshape is the burn tensor path's `[n, p/step, 2, h]` (fwt.rs:117).
        r = out.reshape(*out.shape[:-1], g, 2, h)
        a, b = r[..., 0, :], r[..., 1, :]
        out = torch.stack((a + b, a - b), dim=-2).reshape(x.shape)
        h *= 2
    # the ONE normalization, at the end, exactly where fwt.rs:135 puts it.
    return out / math.sqrt(p)


def hadamard_matrix(p: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Explicit ±1 Hadamard matrix `H_p` (`H_{2k} = kron(H_2, H_{2k-1})`).

    The reference the crate's own tests use (`fwt.rs:75-99`, `H_{2k} = [[H_k,
    H_k], [H_k, -H_k]]`): `fwht(x) == (H_p @ xᵀ)ᵀ / sqrt(p)`. Built by Kronecker
    products rather than the butterfly recursion on purpose — an independent
    construction catches a broken butterfly. `p` is capped at
    [`_HADAMARD_MAX_P`] (it is `p²` floats and exists for tests only).
    """
    if p < 1 or (p & (p - 1)) != 0:
        loud(
            f"hadamard_matrix needs p = 2^k, got {p}: "
            f"the reference is only defined for power-of-two widths"
        )
    if p > _HADAMARD_MAX_P:
        loud(
            f"hadamard_matrix is the test reference and materializes p² floats: p={p} "
            f"asks {p * p * 4 / 2**20:.0f} MiB. Escape: compare against fwht (the "
            f"butterfly) for large p, or raise _HADAMARD_MAX_P deliberately"
        )
    h2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=dtype)
    h = torch.ones((1, 1), dtype=dtype)
    while h.shape[0] < p:
        h = torch.kron(h2, h)
    return h


def v2_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    """BitNet v2 activation quantization: rotate → per-token int quant → rotate back.

    `bits >= 16` is the identity and returns `x` unchanged, bit-exact
    (`activations.rs:31-33`). Otherwise, on `x[..., p]` (p = 2^k):

    1. `rot = fwht(x)` — the outlier-suppressing rotation (`activations.rs:36`).
    2. per-token scale `= rot.abs().amax(-1, keepdim) / (2^(bits-1)-1)` — the
       token's max magnitude in the feature dim, mapped onto the FORMAT's max
       level (7 for int4, 127 for int8; `activations.rs:35` for the per-token
       max, `act_quant.rs:54-59`/`:97` for the level convention).
    3. `q = round(rot / scale).clamp(-L, L)`, `deq = q * scale` — the lossy part.
    4. `fwht(deq)` — rotate back; the transform is its own inverse, so a spike
       comes back at its own position at its own magnitude (the only loss is
       step 3).
    5. `x + (rotated_back − x).detach()` — straight-through (`activations.rs:97-98`).

    The math runs in fp32; the result is cast back to `x.dtype`, so under
    `--bf16` the activations stay bf16 and only the quantization is fp32.
    """
    if bits >= 16:
        return x
    _check(x, "v2_quantize input")
    if bits < 2:
        loud(
            f"v2_quantize needs bits >= 2 (the level count is 2^(bits-1)-1, which is 0 "
            f"at bits=1); got bits={bits}"
        )
    levels = _levels(bits)
    xf = x.float()
    rot = fwht(xf)
    scale = rot.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / levels
    q = (rot / scale).round().clamp(-levels, levels)
    back = fwht(q * scale)
    return (xf + (back - xf).detach()).to(x.dtype)


def outlier_metric(x: torch.Tensor) -> tuple[float, float]:
    """`(max|x|, Pearson kurtosis)` — a TEST DIAGNOSTIC, never a hot-path call.

    The paper's rotation claim is that it flattens the activation distribution,
    and this is the yardstick the tests measure it with: on the steep probe
    (`randn [1, 64]`, `x[0,3]=1000`) max|x| reads 1000.0 and the kurtosis 62.0
    before the rotation, 126.6 and 1.0 after (see the module doc for the p=256
    pair). Pearson kurtosis is `E[(x-μ)⁴] / σ⁴` (a gaussian reads 3.0).

    It calls `.item()`, i.e. a host sync, so it belongs in a test or a
    docstring and nowhere a training step can reach (the port's rule 1.3).
    """
    _check(x, "outlier_metric input")
    xf = x.float()
    mu = xf.mean()
    var = (xf - mu).pow(2).mean()
    kurt = float(((xf - mu).pow(4).mean() / var.pow(2)).item())
    return float(xf.abs().max().item()), kurt
