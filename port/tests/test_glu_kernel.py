"""The fused SwiGLU epilogue (`model/glu_kernel.py`, `ModelConfig.fused_glu`).

The gates, each able to fail on its own:

a. **the fallback IS the eager expression, bit for bit** — `F.silu(gate)·up`
   with `DenseFFN._linear`'s casts, fp32 and bf16. A fallback that computed a
   slightly different function would be a second bug wearing a degradation's
   clothes.
b. **`fused_glu=false` is the model on record** — state-dict and logits sha256
   pins for the `ffn_form="dense"` tree, captured BEFORE the field existed.
c. **the flag cannot be silently inert** — `ffn_form="bank"`/`"tsct"` or a
   non-silu activation is refused with `Escape:`, on the config path and on the
   module seam (`SparseMoE.__init__` calls the same `validate_ffn`).
d. **the training path is the eager path** — with grad enabled the forward,
   loss and every gradient are bit-identical to `fused_glu=false`, and the
   fallback is COUNTED with its reason on `Counters.why` (ADR-0019).
e. **the kernel forward matches the eager reference at the bench shape** (GPU
   only; skipped on a CPU box or without triton, the `test_experts_gpu.py`
   mark). The tolerance is measured and printed, not assumed.

CPU tiers are never skipped — a gate that is skipped everywhere it could fire is
the defect ADR-0019 exists to prevent. Tier e is skipped when the card is
unavailable; the bench (`scripts/bench_glu.py`) is the number, this file is the
equivalence gate.
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest
import torch
import torch.nn.functional as F

from dormouse.config import load_preset, validate
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel
from dormouse.model import glu_kernel as gk

#: The one shape the bench measures: the dense FFN at the live recipe's batch
#: (24 × 1024 tokens), the live model's d=832 and d_ffn=3072. It lives in the
#: kernel module (`glu_kernel.BENCH_SHAPE`) so the gate, the bench and the tile
#: table cannot drift.
BENCH = gk.BENCH_SHAPE

#: The kernel vs eager relative bound. Both accumulate in fp32 and round once to
#: bf16, so the only difference is summation order (cuBLAS vs the K-blocked dot);
#: one bf16 ULP at |h| ≈ 1 is 7.8e-3, and the observed max is reported by the
#: test. 1e-2 is the bf16 envelope with headroom, not a fitted number.
TOL_REL = 1e-2

#: `ffn_form="dense"` at tiny, captured on `main` @ 3229bfd before `fused_glu`
#: existed (seed 0, `torch.manual_seed(0)` at build). Reproduced by `_dense_off`.
DENSE_GOLDEN = (
    "dd1091bff34ee5409d29e5ef5c978e4611b3dbf119a0461585c7a00331d625db",
    "53e68d16847d8f2162c038fb0f36489416db5af4b07ae9b3fdb047be58fee6c0",
    68_817,
)

DENSE_OVER = {"ffn_form": "dense", "n_experts": 1, "moe_topk": 0}


def _cfg(**over):
    base = dict(DENSE_OVER)
    base.update(over)
    return dataclasses.replace(load_preset("tiny"), **base)


def _ids(n: int = 2, t: int = 16) -> torch.Tensor:
    return (torch.arange(n * t, dtype=torch.long) % 256).reshape(n, t)


def _state_hash(model: DormouseModel) -> str:
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _logits_hash(model: DormouseModel, seed: int = 7) -> str:
    ids = _ids()
    torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        logits, loss, _ = model(ids, ids)
    assert loss is not None and torch.isfinite(loss)
    return hashlib.sha256(logits.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _tensors(dt: torch.dtype, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(5, 8, generator=g).to(dt)
    wg = torch.randn(7, 8, generator=g).to(dt)
    wu = torch.randn(7, 8, generator=g).to(dt)
    b = torch.randn(7, generator=g).to(dt)
    return x, wg, wu, b


# --------------------------------------------------------------------------- #
# a. the fallback is the eager expression, bit for bit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
def test_eager_fallback_is_silu_gate_times_up_bit_for_bit(dt):
    x, wg, wu, b = _tensors(dt)
    ref = F.silu(F.linear(x, wg, b)) * F.linear(x, wu)
    out = gk.swiglu_eager(x, wg, wu, b, dt=dt)
    assert torch.equal(out, ref), "the fallback must not be a second function"


def test_the_or_eager_fallback_is_the_eager_expression_and_counted():
    x, wg, wu, b = _tensors(torch.float32)
    ref = F.silu(F.linear(x, wg, b)) * F.linear(x, wu)
    counters = Counters()
    out = gk.fused_swiglu_or_eager(x, wg, wu, b, dt=torch.float32, counters=counters)
    assert torch.equal(out, ref)
    # CPU: the arm's own reason is "not CUDA"; the point of the gate is that the
    # reason EXISTS and is counted, not which branch a CPU box reaches.
    assert counters["glu_kernel_fallback"] == 1
    assert "glu_kernel" not in counters
    assert "glu_kernel:" in counters.why


def test_fp32_is_a_counted_refusal_not_a_slow_fast_path():
    # Measured 1.9x slower than cuBLAS at the bench shape: the flag must not run
    # a fast-path arm that is the slow path.
    x, wg, wu, _ = _tensors(torch.float32)
    why = gk.glu_reason(x, wg, wu, torch.float32)
    assert why is not None and "fp32" in why


def test_no_triton_is_a_counted_reason_not_a_skip(monkeypatch):
    monkeypatch.setattr(gk, "TRITON_AVAILABLE", False)
    assert "no triton" in (gk.glu_reason(*_tensors(torch.float32)[:3]) or "")
    counters = Counters()
    x, wg, wu, b = _tensors(torch.float32)
    out = gk.fused_swiglu_or_eager(x, wg, wu, b, dt=torch.float32, counters=counters)
    assert torch.equal(out, F.silu(F.linear(x, wg, b)) * F.linear(x, wu))
    assert counters["glu_kernel_fallback"] == 1


# --------------------------------------------------------------------------- #
# b. OFF is the model on record
# --------------------------------------------------------------------------- #
def test_the_default_is_the_pinned_dense_model_and_the_flag_is_off():
    torch.manual_seed(0)
    model = DormouseModel(_cfg())
    assert model.loop.moe.experts[0].fused_glu is False
    state, logits, params = DENSE_GOLDEN
    assert _state_hash(model) == state
    assert _logits_hash(model) == logits
    assert sum(p.numel() for p in model.parameters()) == params


# --------------------------------------------------------------------------- #
# c. a form the kernel cannot serve refuses LOUDLY
# --------------------------------------------------------------------------- #
def _refuses(cfg, needle: str) -> None:
    with pytest.raises(LoudError, match="Escape:"):
        validate(cfg)
    with pytest.raises(LoudError, match=needle):
        validate(cfg)


def test_the_flag_refuses_on_the_bank_and_tsct_forms():
    _refuses(dataclasses.replace(load_preset("tiny"), fused_glu=True), "only the `dense` form")
    _refuses(_cfg(ffn_form="tsct", fused_glu=True), "only the `dense` form")


def test_the_flag_refuses_non_silu_activations():
    _refuses(_cfg(expert_act="situ", fused_glu=True), "silu")


def test_the_module_seam_refuses_too_not_only_the_loader():
    # `SparseMoE.__init__` calls `validate_ffn` for a hand-built config that
    # never went through `validate`; the refusal must live at the seam as well.
    from dormouse.model.experts import SparseMoE

    with pytest.raises(LoudError, match="Escape:"):
        SparseMoE(dataclasses.replace(load_preset("tiny"), fused_glu=True), device="cpu")


# --------------------------------------------------------------------------- #
# d. training is the eager path: same logits, same grads, counted fallback
# --------------------------------------------------------------------------- #
def test_grad_enabled_falls_back_counted_and_is_bit_identical():
    off = DormouseModel(_cfg())
    torch.manual_seed(0)
    on = DormouseModel(_cfg(fused_glu=True))
    on.load_state_dict(off.state_dict())
    ids = _ids()

    out = {}
    for name, model in (("off", off), ("on", on)):
        model.train()
        logits, loss, counters = model(ids, ids)
        loss.backward()
        grads = {
            n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None
        }
        out[name] = (logits.detach(), loss.detach(), grads, counters)

    l_off, loss_off, g_off, c_off = out["off"]
    l_on, loss_on, g_on, c_on = out["on"]
    assert torch.equal(l_on, l_off), "the flag moved a training logit"
    assert torch.equal(loss_on, loss_off), "the flag moved the loss"
    assert g_on.keys() == g_off.keys()
    for n in g_off:
        assert torch.equal(g_on[n], g_off[n]), f"the flag moved d/d{n}"
    assert "glu_kernel" not in c_off and "glu_kernel_fallback" not in c_off
    assert c_on["glu_kernel_fallback"] > 0
    assert "glu_kernel" not in c_on
    assert "glu_kernel:" in c_on.why


# --------------------------------------------------------------------------- #
# e. (GPU) the kernel is the eager function at the bench shape
# --------------------------------------------------------------------------- #
GPU = pytest.mark.skipif(
    not (torch.cuda.is_available() and gk.TRITON_AVAILABLE),
    reason="needs CUDA + triton (the kernel is the CUDA arm; CPU runs the eager path)",
)


@GPU
def test_the_kernel_matches_eager_at_the_bench_shape():
    m, d, f = BENCH
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(m, d, generator=gen, device="cuda")
    wg = torch.randn(f, d, generator=gen, device="cuda") / d**0.5
    wu = torch.randn(f, d, generator=gen, device="cuda") / d**0.5
    b = torch.randn(f, generator=gen, device="cuda") * 0.01
    x, wg, wu, b = (t.to(torch.bfloat16) for t in (x, wg, wu, b))
    with torch.no_grad():
        assert gk.glu_reason(x, wg, wu, torch.bfloat16) is None
        ref = gk.swiglu_eager(x, wg, wu, b, dt=torch.bfloat16)
        counters = Counters()
        out = gk.fused_swiglu_or_eager(x, wg, wu, b, dt=torch.bfloat16, counters=counters)
    assert counters["glu_kernel"] == 1 and "glu_kernel_fallback" not in counters
    diff = (out.float() - ref.float()).abs()
    scale = float(ref.float().abs().max())
    rel = float(diff.max()) / max(scale, 1e-30)
    print(
        f"[glu parity] shape={BENCH} bf16 max_abs={float(diff.max()):.3e} "
        f"scale={scale:.3e} max_rel={rel:.3e} (bound {TOL_REL})"
    )
    assert rel <= TOL_REL, f"max_rel {rel:.3e} exceeds {TOL_REL}"
