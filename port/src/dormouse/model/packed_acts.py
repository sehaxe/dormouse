r"""Packed 4-bit SAVED activations — the Function pair that makes `act_qat` pay.

THE GAP `4c53190` LEFT. `act_qat` fake-quantized the FFN GEMM inputs onto the
e2m1 grid and kept the result in bf16. Fake-quant buys accuracy, not bytes: the
tensor the backward needs is still stored at 2 B/element, so the step's VRAM —
and therefore the batch — did not move. The owner's reason for the arm is
"поднять батч", so the storage has to shrink.

WHY A PURE WRAPPER SAVES NOTHING, AND WHAT DOES. A `torch.autograd.Function`
frees everything it does not pass to `save_for_backward`, but it cannot free a
tensor its CONSUMER saves. `pack(x) -> dequant(pack(x))` therefore leaves the
`Linear`/`@` downstream saving that bf16 output for its own `grad_W`, and the
packed copy is pure overhead — `tests/test_packed_acts.py` measures exactly that
against the real save set. The Function has to own the consumer, so
`packed_act_mm` is the quantizer AND the GEMM:

    forward   pack x (e2m1, one e4m3 scale per 16) -> dequantize -> y = xq @ Wᵀ
              save_for_backward(packed, scales, g, W)     # no bf16 x, no bf16 xq
    backward  unpack xq;  grad_x = grad_y @ W            # the STE: identity
              grad_W = grad_yᵀ @ xq

`packed_glu` owns `expert_glu` for the same reason, one level down: `silu(g) * u`
saves `g` (SiLU backward), `silu(g)` (mul backward) and `u` — three `[T, f]` bf16
tensors per layer, the widest saves in the block. Its forward fake-quantizes
`g` and `u` ONCE, computes the GLU on those values, and saves the packed pair;
backward unpacks them and re-runs the (cheap, elementwise) GLU under autograd for
the two gradients. Quantizing in the forward is what makes the backward the STE
of the forward that ran, instead of the gradient of a different function.

THE FORMAT IS `nvfp4.py`'s, never a second codec: two e2m1 nibbles per byte and
one e4m3 scale per 16 columns, 0.5625 B/element against bf16's 2 (3.56x). On the
live arm (`configs/arch_gated_swiglu.toml`, b24×1024, 12 layers, d=832, f=2048)
the FFN's bf16 saves are 462.0 MiB per layer — the gate/up input TWICE
(`mm_mixed` casts it per call), `g`, `silu(g)`, `u` and fc2's input — against
91.97 MiB packed: **4440 MiB (4.34 GiB) off the step**, 80% of those sites.
`tests/test_packed_acts.py` recomputes it from the saved shapes, not a guess.

WHAT MOVED IN THE FORWARD, said out loud. `act_qat="e2m1"` now dequantizes
through the e4m3-ROUNDED block scale (the format the fp4 kernel reads) instead
of `act_quant`'s fp32 scale, and the GLU operands are fake-quantized too. Both
are the arm, not a rounding of it; the gates compare against a fake-quant
reference built from these same values, so the backward is pinned exactly.

Autograd: no double backward (the recompute in `packed_glu` runs with gradients
enabled but does not build a second graph). No `.item()`/D2H read anywhere —
`quantize_nvfp4` keeps its per-tensor `g` on the device (ADR-0018 rule 2).

NOT THE `mm_fp8` LANE. `packed_act_mm` runs the operand's own dequantized GEMM
(`mm_mixed`'s bf16 line), so on a `mm_dtype="fp8"` run the FFN projections are
not the fp8 lane's while attention's are. That combination is not refused in
`config` yet — follow-up, named here rather than left silent.
"""

from __future__ import annotations

import torch

from ..instrument import Counters
from .experts import expert_glu
from .nvfp4 import dequantize_nvfp4, quantize_nvfp4

__all__ = ["packed_act_mm", "packed_glu"]


class _PackedActMM(torch.autograd.Function):
    """`x @ Wᵀ` for one or more weights, with `x` SAVED as 4-bit NVFP4 bytes.

    `weights` is a tuple so the gated FFN pays ONE pack for `gate` and `up`
    (two packs would store two copies and lose the win). The saved set is
    `(packed, scales, g, *weights)`: `x`'s bf16 value exists only inside the
    forward. `dt` is the compute dtype (`mm_mixed`'s rule).
    """

    @staticmethod
    def forward(ctx, x, dt, *weights):
        packed, scales, g = quantize_nvfp4(x)
        xq = dequantize_nvfp4(packed, scales, dt, global_scale=g)
        ctx.save_for_backward(packed, scales, g, *weights)
        return tuple(xq @ w.to(dt).t() for w in weights)

    @staticmethod
    def backward(ctx, *grads):
        packed, scales, g, *weights = ctx.saved_tensors
        # The unpack IS the backward's input: the STE hands x's gradient to xq,
        # and xq is only recoverable from the packed bytes now.
        xq = dequantize_nvfp4(packed, scales, grads[0].dtype, global_scale=g)
        grad_x = None
        for grad, w in zip(grads, weights, strict=True):
            step = grad @ w.to(grad.dtype)
            grad_x = step if grad_x is None else grad_x + step
        rows = xq.reshape(-1, xq.shape[-1])
        grad_w = tuple(grad.reshape(-1, grad.shape[-1]).t() @ rows for grad in grads)
        return (grad_x, None, *grad_w)


class _PackedGLU(torch.autograd.Function):
    """`expert_glu(g, u)` with `g` and `u` SAVED as 4-bit NVFP4 bytes.

    The forward fake-quantizes both operands and evaluates the GLU on the
    quantized values — so the backward is the STE of the function that ran
    (`grad_glu(xq)` returned as `grad_x`), not an approximation of a forward
    that used the bf16 values. `act`/`beta1`/`beta2` ride `ctx` (non-tensors).
    """

    @staticmethod
    def forward(ctx, g, u, act, beta1, beta2):
        pg, sg, gg = quantize_nvfp4(g)
        pu, su, gu = quantize_nvfp4(u)
        dg = dequantize_nvfp4(pg, sg, g.dtype, global_scale=gg)
        du = dequantize_nvfp4(pu, su, u.dtype, global_scale=gu)
        ctx.save_for_backward(pg, sg, gg, pu, su, gu)
        ctx.glu = (act, beta1, beta2)
        return expert_glu(act, dg, du, beta1=beta1, beta2=beta2)

    @staticmethod
    def backward(ctx, grad):
        pg, sg, gg, pu, su, gu = ctx.saved_tensors
        act, beta1, beta2 = ctx.glu
        with torch.enable_grad():
            dg = dequantize_nvfp4(pg, sg, grad.dtype, global_scale=gg).requires_grad_(True)
            du = dequantize_nvfp4(pu, su, grad.dtype, global_scale=gu).requires_grad_(True)
            grad_g, grad_u = torch.autograd.grad(
                expert_glu(act, dg, du, beta1=beta1, beta2=beta2), (dg, du), grad
            )
        return grad_g, grad_u, None, None, None


def packed_act_mm(
    x: torch.Tensor,
    dt: torch.dtype,
    *weights: torch.Tensor,
    counters: Counters | None = None,
) -> tuple[torch.Tensor, ...]:
    """`[y_i = x @ W_iᵀ]` with `x` saved packed. ONE codec: `nvfp4.py`'s.

    `x` is `[..., K]` and every `W_i` is `[N_i, K]` (`nn.Linear`'s layout), so
    the gated FFN passes `gate_proj.weight, up_proj.weight` in one call and both
    products read the single packed copy. `K` must tile 16 — `quantize_nvfp4`
    refuses otherwise, and `config` already refuses a width that does not.
    """
    if not weights:
        raise AssertionError("packed_act_mm needs at least one weight")
    for w in weights:
        if w.dim() != 2 or w.shape[-1] != x.shape[-1]:
            raise AssertionError(
                f"packed_act_mm: x{tuple(x.shape)} does not contract with "
                f"W{tuple(w.shape)}; the weights are [N, K] (`nn.Linear`)"
            )
    out = _PackedActMM.apply(x, dt, *weights)
    if counters is not None:
        counters.count("packed_act_mm")
    return out


def packed_glu(
    act: str,
    g: torch.Tensor,
    u: torch.Tensor,
    *,
    beta1: float = 4.0,
    beta2: float = 25.0,
    counters: Counters | None = None,
) -> torch.Tensor:
    """`expert_glu(act, g, u)` with both operands saved packed (see the class)."""
    out = _PackedGLU.apply(g, u, act, beta1, beta2)
    if counters is not None:
        counters.count("packed_glu")
    return out
