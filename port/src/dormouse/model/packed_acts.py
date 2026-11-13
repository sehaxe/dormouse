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
one e4m3 scale per 16 columns, 0.5625 B/element against bf16's 2 (3.56x).
The PACKING is `nvfp4.quantize_nvfp4`'s fused CUDA kernel, called with
`deq_dtype` so the packed bytes the backward saves and the dequantized value
the GEMM reads come out of ONE pass (the eager quantizer was 13.57 ms at
`[24576, 832]` against 0.31 for the pair) — the STE is unchanged: the
forward's value and the backward's `xq` are still the same bytes. On the
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

THE FORWARD GEMM IS NOW THE fp4 KERNEL, where it can run. `packed_act_mm`
asked `nvfp4.can_use_kernel` once per call (a pure shape/dtype/device fact —
static, no host read) and takes `nvfp4.nvfp4_mm` when it is true: one pack for
the whole weight tuple, no dequantized copy, and the product at the kernel's
0.20 ms instead of bf16's 1.01 at the live shape. The measured reason is the
only one that ever mattered: bytes/s is tokens/s, and the batch a packed
forward buys cannot pay a per-token loss. Two consequences, both named:

  * the weight operand is quantized to NVFP4 by the kernel's own format (a
    ternary weight's nonzeros are ONE e2m1 level times an e4m3 block scale, so
    the tensor stays ternary — the block's magnitude moves by up to one e4m3
    ulp, `tests/test_nvfp4.py` pins both halves);
  * the backward's `grad_x = grad @ w` still uses the bf16 projection, so a
    gradient path sees the weight ~one e4m3 ulp sharper than the forward's
    operand. The arm already fake-quantizes that weight to ternary; unpacking
    the packed weight per backward would cost 0.31 ms per site to remove it.

The lane is decided per call, so a CPU box (or a shape the kernel refuses) keeps
the bf16 line and its pinned values — the CPU gates in
`tests/test_packed_acts.py` and `tests/test_qat_transformer.py` never see the
fp4 lane, and `tests/test_packed_acts.py`'s two card gates pin its values.

THE STORE-ONLY ARM (`act_store="nvfp4"`, `model/transformer.py`). `act_qat`
is a NUMERICS arm: it fake-quantizes the operand the forward GEMM reads. The
same two Functions carry the codec as a MEMORY arm when `store_only` is set:
the forward multiplies the REAL values (its output is bit-identical to the
`bf16` lane) and only `save_for_backward` holds the packed copy, so the
backward's `grad_W = gradᵀ @ xq` reads the 4-bit storage rounding of an operand
the forward used exactly, while `grad_x = grad @ W` is untouched. That is the
whole numerical claim: logits EQUAL, gradients differ only by the rounding of
the saved operand. `quantize_nvfp4` needs `K % 16 == 0` and a model width need
not (the live `d=890` does not), so the STORED copy is tail-padded to the scale
grid and sliced back on unpack — zeros in the tail blocks, no second codec.

NOT THE `mm_fp8` LANE. On the bf16 lane `packed_act_mm` runs the operand's own
dequantized GEMM (`mm_mixed`'s bf16 line); on the fp4 lane it runs the fp4
kernel. Either way the FFN projections are not the fp8 lane's while
attention's are, so on a `mm_dtype="fp8"` run the two lanes are still mixed.
That combination is not refused in `config` yet — follow-up, named here rather
than left silent.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import ACT_STORES
from ..instrument import Counters, loud
from .experts import expert_glu
from .nvfp4 import can_use_kernel, dequantize_nvfp4, nvfp4_mm, quantize_nvfp4
from .tsct import mm_mixed

__all__ = ["check_store", "packed_act_mm", "packed_glu"]


def check_store(mode: str) -> None:
    """LOUD on anything outside [`ACT_STORES`], naming the escape.

    The runtime seam (`TransformerBlock.set_act_store`): a hand-built config
    reaches the block without `config.validate_model_train`.
    """
    if mode not in ACT_STORES:
        loud(
            f"act_store={mode!r} is not a saved-activation store. Known: "
            f"{list(ACT_STORES)} — 'bf16' keeps every activation at 2 B (the "
            f"historical graph, bit for bit); 'nvfp4' saves the block's GEMM "
            f"inputs and GLU operands as e2m1 nibbles with one e4m3 scale per "
            f"16, while the FORWARD still runs bf16. Escape: name one of those."
        )


def _pack(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pack `x`'s last dim for storage, padding it UP to the codec's grid.

    Returns `(packed, scales, g, k_orig)`. The tail pad is zeros, so every real
    16-column block has the same `amax` (and therefore the same e4m3 scale) it
    would have had unpadded, and `_unpack` slices the tail off again.
    """
    k = x.shape[-1]
    pad = -k % 16
    if pad:
        x = F.pad(x, (0, pad))
    packed, scales, g = quantize_nvfp4(x)
    return packed, scales, g, k


def _unpack(packed, scales, g, k: int, dtype: torch.dtype) -> torch.Tensor:
    """The inverse of [`_pack`]: the real width, at the caller's dtype."""
    xq = dequantize_nvfp4(packed, scales, dtype, global_scale=g)
    return xq if xq.shape[-1] == k else xq[..., :k]


class _PackedActMM(torch.autograd.Function):
    """`x @ Wᵀ` for one or more weights, with `x` SAVED as 4-bit NVFP4 bytes.

    `weights` is a tuple so the gated FFN pays ONE pack for `gate` and `up`
    (two packs would store two copies and lose the win). The saved set is
    `(packed, scales, g, *weights)`: `x`'s bf16 value exists only inside the
    forward. `dt` is the compute dtype (`mm_mixed`'s rule). `counters` is a
    non-tensor argument: the lane the forward took has to be readable on the log
    line, or a silent fall back to bf16 is invisible.
    """

    @staticmethod
    def forward(ctx, x, dt, counters, store_only, *weights):
        ctx.k_orig = x.shape[-1]
        if store_only:
            # THE MEMORY ARM: the forward is the bf16 line (`mm_mixed`), bit for
            # bit; only the SAVE is packed. No fp4 GEMM here — the kernel would
            # quantize the weight and read the packed operand in the forward,
            # which is `act_qat`'s numerics arm, not this one.
            packed, scales, g, _ = _pack(x.reshape(-1, x.shape[-1]))
            # Save the CAST weight (`mm_mixed`'s operand), not the fp32 master: the
            # bf16 lane's matmul saves exactly this tensor for its `grad_x`, so the
            # weight half of the save set does not grow and `grad_x` is the same
            # product.
            ws = tuple(w if w.dtype is dt else w.to(dt) for w in weights)
            ctx.save_for_backward(packed, scales, g, *ws)
            return tuple(mm_mixed(x, w, dt) for w in ws)
        # ONE pack, and the GEMM lane is decided by `can_use_kernel` — a pure
        # shape/dtype/device fact, so the branch is static and a CPU call keeps
        # the bf16 line bit for bit (`tests/test_packed_acts.py` pins it there).
        # The fp4 lane does NOT need the dequantized copy: the backward unpacks
        # from the same bytes, so the pack's second store is not paid at all.
        #
        # The GEMM is 2-D, the activation is `[..., K]` (`ln2(x)` in the live
        # block): the fp4 lane flattens and un-flattens around it, the same
        # arithmetic `@` does for the bf16 line. `_reason` refuses a 3-D operand
        # by name and should — the flattening belongs here, not there.
        rows = x.reshape(-1, x.shape[-1])
        with torch.no_grad():
            fp4 = all(can_use_kernel(rows, w, dt) for w in weights)
        if fp4:
            packed, scales, g = quantize_nvfp4(rows)
            ctx.save_for_backward(packed, scales, g, *weights)
            with torch.no_grad():
                # `cache=None`: the weight here is the per-forward `dense_qat`
                # STE tensor (a fresh object every call), so a pack cache would
                # miss every time AND retain a `[N, K]` master per site until
                # the step boundary — a VRAM regression on the arm whose whole
                # point is VRAM. The pack is still 1/3 of the GEMM it feeds.
                return tuple(
                    nvfp4_mm(rows, w, dt, counters, packed=(packed, scales, g), cache=None).view(
                        *x.shape[:-1], w.shape[0]
                    )
                    for w in weights
                )
        packed, scales, g, xq = quantize_nvfp4(x, deq_dtype=dt)
        ctx.save_for_backward(packed, scales, g, *weights)
        return tuple(xq @ w.to(dt).t() for w in weights)

    @staticmethod
    def backward(ctx, *grads):
        packed, scales, g, *weights = ctx.saved_tensors
        # The unpack IS the backward's input: `act_qat`'s STE hands x's
        # gradient to xq, and the store-only arm's saved operand is the same
        # bytes — either way xq is only recoverable from the pack now.
        xq = _unpack(packed, scales, g, ctx.k_orig, grads[0].dtype)
        grad_x = None
        for grad, w in zip(grads, weights, strict=True):
            step = grad @ w.to(grad.dtype)
            grad_x = step if grad_x is None else grad_x + step
        rows = xq.reshape(-1, xq.shape[-1])
        grad_w = tuple(grad.reshape(-1, grad.shape[-1]).t() @ rows for grad in grads)
        return (grad_x, None, None, None, *grad_w)


class _PackedGLU(torch.autograd.Function):
    """`expert_glu(g, u)` with `g` and `u` SAVED as 4-bit NVFP4 bytes.

    The forward fake-quantizes both operands and evaluates the GLU on the
    quantized values — so the backward is the STE of the function that ran
    (`grad_glu(xq)` returned as `grad_x`), not an approximation of a forward
    that used the bf16 values. `act`/`beta1`/`beta2` ride `ctx` (non-tensors).
    """

    @staticmethod
    def forward(ctx, g, u, act, beta1, beta2, store_only):
        ctx.k = (g.shape[-1], u.shape[-1])
        if store_only:
            # The MEMORY arm: the GLU runs on the REAL `g`/`u` (bit-identical to
            # the bf16 lane) and only the save is packed. The backward unpacks
            # and re-runs the GLU for the two gradients, so it reads the 4-bit
            # storage rounding of operands the forward used exactly.
            pg, sg, gg, _ = _pack(g)
            pu, su, gu, _ = _pack(u)
            ctx.save_for_backward(pg, sg, gg, pu, su, gu)
            ctx.glu = (act, beta1, beta2)
            return expert_glu(act, g, u, beta1=beta1, beta2=beta2)
        pg, sg, gg, dg = quantize_nvfp4(g, deq_dtype=g.dtype)
        pu, su, gu, du = quantize_nvfp4(u, deq_dtype=u.dtype)
        ctx.save_for_backward(pg, sg, gg, pu, su, gu)
        ctx.glu = (act, beta1, beta2)
        return expert_glu(act, dg, du, beta1=beta1, beta2=beta2)

    @staticmethod
    def backward(ctx, grad):
        pg, sg, gg, pu, su, gu = ctx.saved_tensors
        act, beta1, beta2 = ctx.glu
        kg, ku = ctx.k
        with torch.enable_grad():
            dg = _unpack(pg, sg, gg, kg, grad.dtype).requires_grad_(True)
            du = _unpack(pu, su, gu, ku, grad.dtype).requires_grad_(True)
            grad_g, grad_u = torch.autograd.grad(
                expert_glu(act, dg, du, beta1=beta1, beta2=beta2), (dg, du), grad
            )
        return grad_g, grad_u, None, None, None, None


def packed_act_mm(
    x: torch.Tensor,
    dt: torch.dtype,
    *weights: torch.Tensor,
    counters: Counters | None = None,
    store_only: bool = False,
) -> tuple[torch.Tensor, ...]:
    """`[y_i = x @ W_iᵀ]` with `x` saved packed. ONE codec: `nvfp4.py`'s.

    `x` is `[..., K]` and every `W_i` is `[N_i, K]` (`nn.Linear`'s layout), so
    the gated FFN passes `gate_proj.weight, up_proj.weight` in one call and both
    products read the single packed copy. `K` must tile 16 on this lane —
    `quantize_nvfp4` refuses otherwise, and `config` refuses a width that does
    not for `act_qat`; the store-only lane below pads instead.

    `store_only=True` is the MEMORY arm (`act_store`): the products run on the
    real `x` exactly as `mm_mixed` would, and only the backward's saved operand
    is the packed copy (`_pack` pads a width that does not tile 16).
    """
    if not weights:
        raise AssertionError("packed_act_mm needs at least one weight")
    for w in weights:
        if w.dim() != 2 or w.shape[-1] != x.shape[-1]:
            raise AssertionError(
                f"packed_act_mm: x{tuple(x.shape)} does not contract with "
                f"W{tuple(w.shape)}; the weights are [N, K] (`nn.Linear`)"
            )
    out = _PackedActMM.apply(x, dt, counters, store_only, *weights)
    if counters is not None:
        counters.count("packed_act_mm")
        if store_only:
            counters.count("act_store_mm")
    return out


def packed_glu(
    act: str,
    g: torch.Tensor,
    u: torch.Tensor,
    *,
    beta1: float = 4.0,
    beta2: float = 25.0,
    counters: Counters | None = None,
    store_only: bool = False,
) -> torch.Tensor:
    """`expert_glu(act, g, u)` with both operands saved packed (see the class)."""
    out = _PackedGLU.apply(g, u, act, beta1, beta2, store_only)
    if counters is not None:
        counters.count("packed_glu")
        if store_only:
            counters.count("act_store_glu")
    return out
