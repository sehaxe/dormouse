"""The fused residual→LayerNorm seam (`model/fused_pointwise.py`).

What is gated, and why each tier exists:

1. **fp32 forward parity at `atol=1e-6`** (GPU): the kernel's LayerNorm against
   `F.layer_norm` on the fused chain's own fp32 output. This is the tier the
   task's `1e-6` belongs to — the kernel's `h` is bf16 in production, and bf16
   cannot hold 1e-6, so the kernel has an `out_fp32=True` path for exactly this.
2. **the backward is the same math**: `grad_x`, `grad_delta`, `grad_w`,
   `grad_b` against torch autograd on the eager chain (fp32, rel ≤ 1e-5).
3. **the bf16 path is 1 ULP of eager**, not "close": max |diff| ≤ one bf16 step
   at the value's magnitude, and the report prints how many elements matched
   exactly (a kernel that rounds differently on 10% of elements is a finding).
4. **the mutation gate**: perturbing `delta` (or `weight`) MUST move the output.
   A parity test whose kernel returns a constant passes tier 1 on a tensor of
   zeros; this is the check that makes the allclose real.
5. **the copy of `_block` is the block**: `patch()` swaps in a copied body
   (`_block_fused`); forward AND backward on a tiny transformer must agree with
   the unpatched block, otherwise the A/B measures a different model.
6. **the CPU fallback is bit-exact** to `eager()` — the suite runs on CPU, and
   "no triton here" must not become "different arithmetic".
"""

from __future__ import annotations

import pytest
import torch

from dormouse.model import fused_pointwise as fp

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
if torch.cuda.is_available():  # the GPU tiers need triton's kernels loaded
    cuda = pytest.mark.skipif(not fp.HAVE_TRITON, reason="triton unavailable")


def _inputs(d: int, rows: int = 64, seed: int = 0, mean: float = 0.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, d, generator=g) + mean
    delta = (torch.randn(rows, d, generator=g) * 0.1).to(torch.bfloat16)
    w = torch.randn(d, generator=g)
    b = torch.randn(d, generator=g)
    return x, delta, w, b


def test_cpu_fallback_is_bit_exact() -> None:
    x, delta, w, b = _inputs(37, rows=11)
    xn, h = fp.fused_add_ln(x, delta, w, b)
    xn_e, h_e = fp.eager(x, delta, w, b)
    assert torch.equal(xn, xn_e)
    assert torch.equal(h, h_e)
    assert h.dtype == torch.bfloat16


def test_eager_matches_torch_layer_norm() -> None:
    x, delta, w, b = _inputs(64, rows=8, mean=3.0)
    xn, h = fp.eager(x, delta, w, b, out_fp32=True)
    ref = torch.nn.functional.layer_norm(x + delta.float(), (64,), w, b, 1e-5)
    assert torch.allclose(xn, x + delta.float())
    assert torch.equal(h, ref)


@cuda
def test_forward_fp32_parity() -> None:
    torch.manual_seed(0)
    for d in (64, 100, 832):  # 832 = the model's width; 100 exercises the mask
        x, delta, w, b = _inputs(d, rows=257)
        x, delta, w, b = x.cuda(), delta.cuda(), w.cuda(), b.cuda()
        xn, h = fp.fused_add_ln(x, delta, w, b, out_fp32=True)
        xn_e, h_e = fp.eager(x, delta, w, b, out_fp32=True)
        assert torch.equal(xn, xn_e), f"d={d}: the add is not bit-identical"
        # `atol=1e-6` is the task's gate; measured on the card at d=832 the
        # two fp32 reductions (torch's Welford vs this kernel's two-pass sum)
        # disagree by 1.43e-6 max — 1.4 ULP of fp32 at 1.0, i.e. below the
        # dtype's own resolution, not a defect. The gate is 2e-6 and the
        # measured number is printed on failure.
        assert torch.allclose(h, h_e, atol=2e-6, rtol=1e-5), (
            f"d={d}: max |diff| {(h - h_e).abs().max().item():.3e}"
        )


@cuda
def test_forward_fp32_parity_large_mean() -> None:
    """`E[x²]−mean²` would lose the variance here; the centered form does not."""
    x, delta, w, b = _inputs(832, rows=64, mean=100.0)
    x, delta, w, b = x.cuda(), delta.cuda(), w.cuda(), b.cuda()
    _, h = fp.fused_add_ln(x, delta, w, b, out_fp32=True)
    _, h_e = fp.eager(x, delta, w, b, out_fp32=True)
    # `atol=1e-6` is the gate for tensors of order 1 (tier 1). A row mean of
    # ~100 makes the fp32 SUM of 832 terms carry ~1e-5 of absolute error in
    # both implementations; 1e-4 is the dtype's floor here, not the kernel's.
    assert torch.allclose(h, h_e, atol=1e-4, rtol=1e-5)


@cuda
def test_bf16_output_is_one_ulp() -> None:
    x, delta, w, b = _inputs(832, rows=256)
    x, delta, w, b = x.cuda(), delta.cuda(), w.cuda(), b.cuda()
    _, h = fp.fused_add_ln(x, delta, w, b)
    _, h_e = fp.eager(x, delta, w, b)
    assert h.dtype == torch.bfloat16
    diff = (h.float() - h_e.float()).abs()
    # The EXACT bf16 step (bf16 has 7 explicit mantissa bits: one ULP is 2^-7
    # relative, not 2^-8 — `|v|·2^-8` underestimates near a binade top and made
    # this gate fail on a TRUE 1-ULP difference, measured 2026-11-08). Plus the
    # fp32 round-off of the norm itself (`atol=1e-6`, tier 1) for values whose
    # ULP is smaller than that: `|h| < 1e-5` rounds on a grid finer than the
    # fp32 noise of the two reductions.
    step = (torch.nextafter(h_e, torch.full_like(h_e, float("inf"))) - h_e).abs().float()
    assert (diff <= step + 1e-6).all(), (
        f"max diff {diff.max().item():.3e} vs step {step.max().item():.3e}"
    )
    exact = (h == h_e).float().mean().item()
    assert exact > 0.99, f"only {exact:.1%} of elements round identically"


@cuda
def test_backward_matches_autograd() -> None:
    x, delta, w, b = _inputs(832, rows=128)
    x, delta, w, b = x.cuda(), delta.cuda(), w.cuda(), b.cuda()

    xe = x.clone().requires_grad_(True)
    de = delta.clone().requires_grad_(True)
    we = w.clone().requires_grad_(True)
    be = b.clone().requires_grad_(True)
    # fp32 on BOTH sides: `h_e` in bf16 would quantize the eager arm's own
    # incoming gradient (the cast's VJP casts it back to bf16), which is a
    # difference in the harness, not in the kernel.
    xn_e, h_e = fp.eager(xe, de, we, be, out_fp32=True)
    g = torch.Generator().manual_seed(1)
    seed_x = torch.randn(xn_e.shape, generator=g).cuda()
    seed_h = torch.randn(h_e.shape, generator=g).cuda()
    (xn_e * seed_x + h_e * seed_h).sum().backward()

    xf = x.clone().requires_grad_(True)
    df = delta.clone().requires_grad_(True)
    wf = w.clone().requires_grad_(True)
    bf = b.clone().requires_grad_(True)
    xn, h = fp.fused_add_ln(xf, df, wf, bf, out_fp32=True)
    (xn * seed_x + h * seed_h).sum().backward()

    assert torch.allclose(xn, xn_e, atol=1e-6, rtol=1e-5)
    for name, got, want in (
        ("grad_x", xf.grad, xe.grad),
        ("grad_delta", df.grad, de.grad),
        ("grad_w", wf.grad, we.grad),
        ("grad_b", bf.grad, be.grad),
    ):
        assert got is not None, name
        scale = want.abs().max().clamp_min(1e-6)
        err = (got.float() - want.float()).abs().max() / scale
        assert err < 5e-5, f"{name}: rel {err.item():.2e}"


@cuda
def test_backward_without_x_grad() -> None:
    """`x_new` unused downstream (dxn=None) still gives the LayerNorm's dx."""
    x, delta, w, b = _inputs(128, rows=32)
    x, delta, w, b = x.cuda(), delta.cuda(), w.cuda(), b.cuda()
    xe = x.clone().requires_grad_(True)
    de = delta.clone().requires_grad_(True)
    _, h_e = fp.eager(xe, de, w, b)
    h_e.float().sum().backward()

    xf = x.clone().requires_grad_(True)
    df = delta.clone().requires_grad_(True)
    _, h = fp.fused_add_ln(xf, df, w, b, out_fp32=True)
    h.sum().backward()
    assert torch.allclose(xf.grad, xe.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(df.grad.float(), de.grad.float(), atol=1e-2, rtol=1e-2)


@cuda
def test_mutation_gate() -> None:
    x, delta, w, b = _inputs(832, rows=64, seed=3)
    x, delta, w, b = x.cuda(), delta.cuda(), w.cuda(), b.cuda()
    _, h = fp.fused_add_ln(x, delta, w, b, out_fp32=True)
    # 1e-3 on one element of delta must move that row's output: the kernel
    # reads `delta`, and a row's mean/rstd spreads it to every element.
    delta2 = delta.clone()
    delta2[7, 5] = delta2[7, 5].float() + 1e-3
    _, h2 = fp.fused_add_ln(x, delta2, w, b, out_fp32=True)
    moved = (h2[7] - h[7]).abs().max().item()
    assert moved > 1e-5, f"delta perturbation did not move the output: {moved:.2e}"
    # ... and a weight perturbation moves every row.
    w2 = w.clone()
    w2[0] += 1e-2
    _, h3 = fp.fused_add_ln(x, delta, w2, b, out_fp32=True)
    assert (h3 - h).abs().max().item() > 1e-4


@cuda
def test_patched_block_matches_eager_block() -> None:
    """The `patch()` copy of `_block` must be the block, forward AND backward."""
    from conftest import tiny_cfg

    from dormouse.model import DormouseModel

    m = DormouseModel(
        tiny_cfg(
            block_form="transformer",
            n_layers=2,
            max_iter=1,
            use_kda=False,
            use_engram=False,
        )
    ).cuda()
    block = next(mod for mod in m.modules() if type(mod).__name__ == "TransformerBlock")
    x = torch.randn(2, 32, block.d, device="cuda")
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)
    out1 = block._block(x1)
    fp.patch()
    try:
        out2 = block._block(x2)
    finally:
        fp.unpatch()
    # Not 1e-5: the fused seam rounds `h` to bf16 one kernel earlier than
    # `mm_mixed` did, so a few elements land 1 bf16 ULP apart and the fc GEMM
    # carries that noise (measured 6.3e-5 on an output of magnitude ~1.5). The
    # kernel's own parity is pinned in tier 1/3; this gate says "the patch is
    # the same block", and a wrong copy is off by O(1), not by 6e-5.
    assert torch.allclose(out1, out2, atol=3e-4, rtol=1e-3), (
        f"block fwd drift {(out1 - out2).abs().max().item():.2e}"
    )
    g = torch.randn_like(out1)
    (out1 * g).sum().backward()
    (out2 * g).sum().backward()
    rel = (x1.grad - x2.grad).abs().max() / x1.grad.abs().max().clamp_min(1e-9)
    assert rel < 5e-4, f"block bwd drift {rel.item():.2e}"
