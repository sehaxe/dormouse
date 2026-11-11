"""Gates for the PACKED 4-bit saved activations (`model/packed_acts.py`).

THE CLAIM THIS FILES PINS. `act_qat="e2m1"` must save FEWER BYTES, not just
quantize: the bf16 tensors the FFN's backward needs are stored as NVFP4 (two
e2m1 nibbles + one e4m3 scale per 16), so the step's peak — and the max batch —
moves. What is proven here, in order:

1. the FORMAT is `nvfp4.py`'s, on a hand-computed two-block example (`a`);
2. the forward value and the backward gradients are the fake-quant reference's,
   exactly (`b`,`c`) — the STE is the identity, not an approximation;
3. the packed tensors are what `save_for_backward` actually holds, and the wide
   bf16 saves are gone from the graph (`d`) — the spy is the answer to "the flag
   is on but nothing was saved";
4. the CONTRAST: a PURE `pack(x) -> dequant(x)` wrapper saves nothing at all,
   because the consumer still saves its bf16 output (`e`) — the measurement that
   forced these Functions to own their GEMM/GLU instead of wrapping only the
   quantizer;
5. the byte arithmetic at the live arm's shapes (`f`), and that the CUDA peak
   actually falls rather than being re-arranged (`g`, skipped without a card).

The codec's own gates (the 16 e2m1 levels, nibble order, the kernel layout,
bounded error, the e4m3 overflow guard) are `tests/test_nvfp4.py`'s and are not
re-copied here: this module calls `quantize_nvfp4`/`dequantize_nvfp4` and adds no
second codec.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch.autograd.graph import saved_tensors_hooks

from dormouse.config import ModelConfig
from dormouse.instrument import Counters
from dormouse.model import DormouseModel
from dormouse.model.act_quant import fake_quant_act
from dormouse.model.experts import expert_glu
from dormouse.model.nvfp4 import (
    bytes_per_elements,
    dequantize_nvfp4,
    emulated_mm,
    quantize_nvfp4,
)
from dormouse.model.packed_acts import packed_act_mm, packed_glu

BASE = dict(
    d_model=64,
    n_heads=4,
    head_dim=16,
    d_ffn=128,
    d_ffn_gelu=192,
    vocab=256,
    max_seq_len=64,
    n_layers=2,
    max_iter=1,
    block_form="transformer",
    ffn_gated=True,
    expert_act="silu",
    bf16=True,
    use_tsct=False,
    tie_embeddings=False,
    use_kda=False,
    use_engram=False,
    use_csa=False,
    layerrope=False,
)


def _model(**over) -> DormouseModel:
    torch.manual_seed(1234)
    return DormouseModel(ModelConfig(**{**BASE, **over}))


def _input(b: int = 2, t: int = 32) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randint(0, 256, (b, t))


def _saves(fn):
    """Every tensor `save_for_backward` holds while `fn` runs (a pass-through spy)."""
    seen: list[tuple[tuple[int, ...], torch.dtype, int]] = []

    def pack(t):
        seen.append((tuple(t.shape), t.dtype, t.numel() * t.element_size()))
        return t

    with saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return seen, out


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach(), b.detach()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


# ---------------------------------------------------------------------------
# (a) the hand-computed two-block example, through the packed Function
# ---------------------------------------------------------------------------

#: Block A: amax 3.0 → scale 0.5 (exactly e4m3) → norm = 2·x. Rounding cases:
#: 2.9 → 5.8 → 6.0 (saturate); 0.625 → 1.25 → tie, code 2 is even → 1.0;
#: 0.75 → 1.5 → a level. Block B: amax 6.0 → scale 1.0, every value a level.
_BLOCK_A = torch.tensor(
    [3.0, 2.9, 1.5, 0.75, 0.625, 0.5, 0.0, -0.625, -1.5, -2.9, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)
_BLOCK_B = torch.tensor(
    [6.0, 4.0, 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -4.0, -6.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)
#: The dequantized values the codec must return, derived by hand from the table
#: in `act_quant._e2m1_round` (0.5·scale = the dequantized unit of block A).
_AQ = torch.tensor([3.0, 3.0, 1.5, 0.75, 0.5, 0.5, 0.0, -0.5, -1.5, -3.0, -3.0] + [0.0] * 5)
#: Nibbles: magnitude INDEX (0, 0.5, 1, 1.5, 2, 3, 4, 6 → 0..7) | 8 on a
#: negative, element 2i in the LOW nibble. Block A: 7,7,5,3,2,2,0,10,13,15,15,0…
#: Block B: 7,6,4,2,1,0,9,10,12,14,15,0…
_PACKED = bytes(
    [0x77, 0x35, 0x22, 0xA0, 0xFD, 0x0F, 0x00, 0x00, 0x67, 0x24, 0x01, 0xA9, 0xEC, 0x0F, 0x00, 0x00]
)
#: e4m3 for 0.5 (E=6 → 0x30) and 1.0 (E=7 → 0x38).
_SCALES = bytes([0x30, 0x38])


def test_a_hand_computed_two_blocks_land_on_the_packed_bytes_and_values():
    x = torch.stack((_BLOCK_A, _BLOCK_B))
    packed, scales, g = quantize_nvfp4(x)
    assert packed.contiguous().view(-1).numpy().tobytes() == _PACKED
    assert scales.contiguous().view(torch.uint8).view(-1).numpy().tobytes() == _SCALES
    assert float(g) == 1.0  # max block scale 1.0 < 448: no second-level lift
    assert torch.equal(
        dequantize_nvfp4(packed, scales, torch.float32, global_scale=g),
        torch.stack((_AQ, _BLOCK_B)),
    )
    # ... and the Function's forward is that value times the weight, bit for bit.
    w = torch.arange(2 * 16, dtype=torch.float32).reshape(2, 16) / 7.0
    (y,) = packed_act_mm(x, torch.float32, w)
    assert torch.equal(y, torch.stack((_AQ, _BLOCK_B)) @ w.t())


# ---------------------------------------------------------------------------
# (b,c) forward = the fake-quant reference; backward = the same gradients
# ---------------------------------------------------------------------------


def _ref_quant(x: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """The reference value: the codec's dequantization, no second implementation."""
    packed, scales, g = quantize_nvfp4(x)
    return dequantize_nvfp4(packed, scales, dt, global_scale=g)


class _Ste(torch.autograd.Function):
    """`fake_quant_act`'s contract, spelled on THIS codec: forward q, grad identity."""

    @staticmethod
    def forward(ctx, x, dt):
        return _ref_quant(x.detach(), dt)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
def test_b_forward_is_the_fake_quant_value_and_c_grads_match_the_reference(dt):
    torch.manual_seed(0)
    x = torch.randn(4, 8, 32, dtype=torch.float32) * 2.0
    w = torch.randn(16, 32, dtype=torch.float32) * 0.1

    # --- the packed Function
    xp = x.clone().requires_grad_(True)
    wp = w.clone().requires_grad_(True)
    (yp,) = packed_act_mm(xp, dt, wp)
    gp = torch.randn_like(yp)
    yp.backward(gp)

    # --- the same algebra written out: fake-quant (this codec) then a plain GEMM
    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    ste = _Ste.apply(xr, dt)
    yr = ste @ wr.to(dt).t()
    yr.backward(gp)

    assert torch.equal(yp, yr), "the forward must be the reference's exact value"
    assert _rel(xp.grad, xr.grad) <= 1e-5
    assert _rel(wp.grad, wr.grad) <= 1e-5

    # ... and the STE really is the identity: at fp32 the gradient is the exact
    # one `act_quant.fake_quant_act` (the fp32-scale codec) hands back. Compare
    # only in fp32: the bf16 lane's GEMM rounds the SAME identity gradient.
    if dt is torch.float32:
        xs = x.clone().requires_grad_(True)
        (fake_quant_act(xs, "e2m1", 16) @ w.t()).backward(gp)
        assert torch.equal(xp.grad, xs.grad)


def test_c_backward_of_the_packed_glu_is_the_fake_quant_gradient():
    torch.manual_seed(0)
    g = torch.randn(4, 8, 32) * 2.0
    u = torch.randn(4, 8, 32) * 2.0

    gp = g.clone().requires_grad_(True)
    up = u.clone().requires_grad_(True)
    out = packed_glu("silu", gp, up)
    grad = torch.randn_like(out)
    out.backward(grad)

    gr = g.clone().requires_grad_(True)
    ur = u.clone().requires_grad_(True)
    dg, du = _ref_quant(gr, torch.float32), _ref_quant(ur, torch.float32)
    ref = expert_glu("silu", gr + (dg - gr).detach(), ur + (du - ur).detach())
    ref.backward(grad)

    assert _rel(out, ref) <= 1e-6
    assert _rel(gp.grad, gr.grad) <= 1e-5
    assert _rel(up.grad, ur.grad) <= 1e-5


# ---------------------------------------------------------------------------
# (d) the spy: the PACKED tensors are what the graph saved
# ---------------------------------------------------------------------------


def _ffn_saves(arm: str) -> dict[str, int]:
    """Saved-tensor bytes of the FFN activation sites, one arm, same model/input.

    `b=3, t=32` → `T = 96`, chosen so `T` is neither `d` nor `f`: a bf16 save of
    shape `[T, d]` is then unambiguous against a weight cast of shape `[d, f]`.
    """
    model = _model()
    for blk in model.transformer_blocks:
        blk.act_qat = arm
    ids, targets = _input(3), _input(3)
    seen, _ = _saves(lambda: model(ids, (targets + 1) % 256)[1])
    d, f = BASE["d_model"], BASE["d_ffn"]
    t = ids.shape[0] * ids.shape[1]
    out = {"bf16_act": 0, "bf16_f": 0, "packed_data": 0, "packed_scales": 0}
    for shape, dtype, nbytes in seen:
        if len(shape) not in (2, 3):
            continue
        rows = shape[0] * shape[1] if len(shape) == 3 else shape[0]
        if rows != t:
            continue
        if dtype is torch.bfloat16 and shape[-1] in (d, f):
            out["bf16_act"] += nbytes
            out["bf16_f"] += nbytes if shape[-1] == f else 0
        elif dtype is torch.uint8 and shape[-1] in (d // 2, f // 2):
            out["packed_data"] += nbytes
        elif dtype is torch.float8_e4m3fn and shape[-1] in (d // 16, f // 16):
            out["packed_scales"] += nbytes
    return out


def test_d_the_ffn_activations_are_saved_packed_and_the_bf16_ones_are_gone():
    off = _ffn_saves("none")
    on = _ffn_saves("e2m1")
    d, f = BASE["d_model"], BASE["d_ffn"]
    t = 3 * 32
    layers = BASE["n_layers"]
    # The bf16 arm stores the gate/up input TWICE (`mm_mixed` casts per call),
    # the three GLU saves and fc2's input at 2 B/element; the packed arm stores
    # one `[T, d]` and three `[T, f]` as nibbles, nothing wide in bf16.
    assert off["bf16_act"] - on["bf16_act"] == layers * (2 * t * d * 2 + 4 * t * f * 2)
    assert on["bf16_f"] == 0, "a wide bf16 activation survived the packed arm"
    assert on["packed_data"] == layers * (t * (d // 2) + 3 * t * (f // 2))
    assert on["packed_scales"] == layers * (t * (d // 16) + 3 * t * (f // 16))


# ---------------------------------------------------------------------------
# (e) the honest-warning measurement: a PURE wrapper saves nothing
# ---------------------------------------------------------------------------


class _PureWrapper(torch.autograd.Function):
    """The Function the brief literally describes — and it pays for nothing."""

    @staticmethod
    def forward(ctx, x):
        packed, scales, g = quantize_nvfp4(x)
        ctx.save_for_backward(packed, scales, g)
        return dequantize_nvfp4(packed, scales, x.dtype, global_scale=g)

    @staticmethod
    def backward(ctx, grad):
        return grad


def test_e_a_pure_pack_wrapper_saves_nothing_because_the_consumer_does():
    x = torch.randn(64, 32, requires_grad=True)
    w = torch.randn(16, 32, requires_grad=True)

    base, _ = _saves(lambda: x @ w.t())
    wrapped, _ = _saves(lambda: _PureWrapper.apply(x) @ w.t())

    base_bytes = sum(n for _, _, n in base)
    wrapped_bytes = sum(n for _, _, n in wrapped)
    # The consumer still saves the wrapper's bf16 OUTPUT for its grad_W, so the
    # packed copy is pure overhead. This is why `packed_act_mm` owns the GEMM.
    assert any(s == (64, 32) and d is torch.float32 for s, d, _ in wrapped)
    assert any(d is torch.uint8 for _, d, _ in wrapped)
    assert wrapped_bytes > base_bytes, (
        f"a pure wrapper saved {wrapped_bytes} B against {base_bytes} B: the "
        f"premise of 'packing saves VRAM' is FALSE unless the consumer is owned"
    )


# ---------------------------------------------------------------------------
# (f) the byte report at the live arm's own shapes
# ---------------------------------------------------------------------------


def test_f_b24x1024_12_layers_byte_report():
    """What the packed path removes, computed from the shapes it actually packs.

    The FFN sites measured by the spy above, per layer: the gate/up input is
    saved bf16 TWICE (`mm_mixed` casts it per call) and once packed; the GLU
    saves `g`, `silu(g)` and `u`, and `fc2` saves its input — four `[T, f]`
    bf16 — against three packed `[T, f]` (`g`, `u`, and fc2's input).
    """
    b, seq, d, f, layers = 24, 1024, 832, 2048, 12
    t = b * seq

    def packed(rows: int, k: int) -> int:
        return int(rows * k * bytes_per_elements(k))

    before = layers * (2 * t * d * 2 + 4 * t * f * 2)
    after = layers * (packed(t, d) + 3 * packed(t, f))
    assert before == 5_813_305_344
    assert after == 1_157_234_688
    assert (before - after) / 2**20 == pytest.approx(4440.4, abs=0.1)
    print(
        f"\nb24x1024, {layers} layers: {before / 2**20:.1f} MiB -> "
        f"{after / 2**20:.1f} MiB, saved {(before - after) / 2**20:.1f} MiB "
        f"({(before - after) / 2**30:.2f} GiB)"
    )


# ---------------------------------------------------------------------------
# no host sync, and the model still trains through both Functions
# ---------------------------------------------------------------------------


def test_g_no_host_sync_in_the_packed_forward_or_backward(monkeypatch):
    model = _model(act_qat="e2m1")
    ids = _input()

    def boom(*_a, **_k):
        raise AssertionError("host sync in the packed activation path")

    for name in ("item", "tolist", "cpu", "numpy", "__float__", "__int__"):
        monkeypatch.setattr(torch.Tensor, name, boom)
    _, loss, counters = model(ids, ids)
    loss.backward()
    assert counters.get("packed_act_mm") == 2 * BASE["n_layers"]
    assert counters.get("packed_glu") == BASE["n_layers"]


def test_h_the_packed_arm_has_grads_on_every_ffn_weight():
    model = _model(act_qat="e2m1", dense_qat="ternary")
    ids = _input()
    _, loss, _ = model(ids, ids)
    loss.backward()
    blk = model.transformer_blocks[0]
    for lin in (blk.gate_proj, blk.up_proj, blk.fc2):
        grad = lin.weight.grad
        assert grad is not None and bool(torch.isfinite(grad).all())
        assert grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# (g) the card: the peak must actually FALL, not be re-arranged
# ---------------------------------------------------------------------------

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@CUDA
def test_g_the_peak_falls_on_cuda():
    """`max_memory_allocated` on one step, same weights, flag flipped.

    If this does not pass, the arm is a no-op and must not be shipped. Each arm
    runs one warm-up step first: cuBLAS workspaces and allocator segments are
    part of the measured peak and are paid once per process, so a cold first
    arm would look cheaper than a warm second one (measured: 497 vs 522 MB
    before the warm-up was added). `min` of two measured steps per arm.
    """
    cfg = dataclasses.replace(
        ModelConfig(
            **{
                **BASE,
                "d_model": 256,
                "n_heads": 8,
                "head_dim": 32,
                "d_ffn": 512,
                "d_ffn_gelu": 768,
                "n_layers": 2,
            }
        ),
        act_qat="none",
    )
    torch.manual_seed(0)
    model = DormouseModel(cfg, device="cuda")
    ids = torch.randint(0, 256, (32, 512), device="cuda")

    def step(arm: str) -> None:
        for blk in model.transformer_blocks:
            blk.act_qat = arm
        _, loss, _ = model(ids, (ids + 1) % 256)
        loss.backward()
        model.zero_grad(set_to_none=True)

    def peak(arm: str) -> int:
        step(arm)  # warm: cuBLAS workspaces, allocator segments
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        step(arm)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    off = min(peak("none"), peak("none"))
    on = min(peak("e2m1"), peak("e2m1"))
    assert on < off * 0.97, (
        f"the packed arm peaked at {on / 2**20:.1f} MiB against bf16's "
        f"{off / 2**20:.1f} MiB: no bytes were saved and the arm is a no-op"
    )


# ---------------------------------------------------------------------------
# the forward GEMM's lane: the fp4 kernel on the card, bf16 everywhere else
# ---------------------------------------------------------------------------

HAS_FP4 = hasattr(torch, "float4_e2m1fn_x2") and torch.cuda.is_available()


@pytest.mark.skipif(not HAS_FP4, reason="needs the CUDA fp4 `_scaled_mm` (sm_120, torch >= 2.13)")
def test_the_forward_gemm_is_the_fp4_kernel_on_the_card():
    """`packed_act_mm`'s forward takes `nvfp4.nvfp4_mm` when it can run.

    Two things are pinned. The COUNTER names the lane (a silent fall back to
    bf16 is the whole 6.7% the arm was losing), and the kernel agrees with the
    EMULATED lane — same packed bytes, fp32 dequantize, fp32 GEMM — within the
    format's own band. Compared against `emulated_mm`, never a bf16 GEMM: a bf16
    reference agrees with a wrong scale swizzle to 46% (measured).
    """
    torch.manual_seed(0)
    x = (torch.randn(2, 4, 32) * 2.0).cuda()  # `[..., K]`, the live block's shape
    w = (torch.randn(16, 32) * 0.1).cuda()
    c = Counters()
    (y,) = packed_act_mm(x, torch.bfloat16, w, counters=c)
    assert y.shape == (2, 4, 16), y.shape
    assert c.get("nvfp4_gemm", 0) == 1, f"the bf16 lane ran on the card: {getattr(c, 'why', '')}"
    emu = emulated_mm(x.reshape(-1, 32), w, torch.bfloat16)
    rel = ((y.float().reshape(-1, 16) - emu.float()).norm() / emu.float().norm()).item()
    assert rel < 1e-2, f"kernel vs emulated relative error {rel:.4f}"


@pytest.mark.skipif(not HAS_FP4, reason="needs the CUDA fp4 `_scaled_mm` (sm_120, torch >= 2.13)")
def test_the_backward_does_not_depend_on_which_forward_lane_ran(monkeypatch):
    """Both lanes unpack the SAME saved bytes, so `grad_x` and `grad_W` are the
    same arithmetic on the same values — the lane moves the forward's operand
    precision (named in `packed_acts`' docstring), not the gradient.

    This is the gate that says the arm's VRAM/throughput win cannot silently
    change what it trains on: bit-identical gradients, lane or no lane.
    """
    import dormouse.model.packed_acts as pa

    torch.manual_seed(0)
    x = (torch.randn(8, 32) * 2.0)
    w = (torch.randn(16, 32) * 0.1)
    gp = torch.randn(8, 16)

    def run(bf16_lane: bool, device: str):
        if bf16_lane:
            monkeypatch.setattr(pa, "can_use_kernel", lambda *a, **k: False)
        xp = x.to(device).clone().requires_grad_(True)
        wp = w.to(device).clone().requires_grad_(True)
        (y,) = packed_act_mm(xp, torch.bfloat16, wp)
        y.backward(gp.to(device))
        return y.detach(), xp.grad, wp.grad

    y_fp4, gx_fp4, gw_fp4 = run(False, "cuda")
    monkeypatch.undo()
    y_bf16, gx_bf16, gw_bf16 = run(True, "cuda")
    assert torch.equal(gx_fp4, gx_bf16), "the forward lane moved grad_x"
    assert torch.equal(gw_fp4, gw_bf16), "the forward lane moved grad_W"
    # The forward's value DOES move: the weight operand is now 4-bit too.
    rel = ((y_fp4.float() - y_bf16.float()).norm() / y_bf16.float().norm()).item()
    assert rel < 0.2, f"the fp4 forward is {rel:.3f} from the bf16 one"
