"""CUDA gates for the real KDA arm (`fla.ops.kda.chunk_kda`, FLA 0.5.2).

Skipped wherever `torch.cuda.is_available()` is false, so a CPU-only CI env runs
the SDPA fallback and nothing here.

Three claims, in the order the plan needs them:

1. **every KDA parameter trains on the fused arm**, `beta_proj` included — the
   one parameter that is gradient-FREE on the shipped SDPA fallback (the
   `8fa5d4c` defect class), so this is the gate that the real arm is not that
   defect wearing a new name;
2. **the fused kernel sits a small distance off the exact recurrence**, anchored
   on a naive per-token loop. A `torch` loop is the only honest anchor: the
   chunked kernel is a different algorithm, not a different spelling, and no
   cross-substrate bitwise claim exists. The bounds are WIRING gates — a
   transposed head axis or a swapped gate is O(1) off — not a certificate;
3. **the arm the log line names is the arm that ran** (`kda=fla,n=…`).

One more, added when the fallback stopped being a different model: **the SDPA
arm L2-normalizes q/k before the attention** (`_sdpa_normalizes`), the same
norm the fused arm gets in-kernel (`use_qk_l2norm_in_kernel=True`, burn-kda
K3). Without it the two arms ran different q/k; `beta_proj` stays dead on the
fallback either way, so the counters below are unchanged.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg

from dormouse.instrument import Counters
from dormouse.model import DormouseModel
from dormouse.model.kda import KDAAttention, fla_status
from dormouse.train import kda_field

#: `find_spec`, not an import: a warmup call at collection time would be GPU work
#: in the wrong place. A CUDA box WITHOUT FLA skips these (nothing to test); a
#: CUDA box with a FLA whose warmup fails does not — `_fused_is_live` fails it.
_FLA_INSTALLED = importlib.util.find_spec("fla") is not None

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and _FLA_INSTALLED),
    reason="needs a CUDA device AND FLA (`uv sync --extra gpu`)",
)

DEV = "cuda"
#: The anchor shape the task names: b1, h8, d_head 64, seq 256.
ANCHOR = dict(d_model=512, n_heads=8, head_dim=64, max_seq_len=256)


def _fused_is_live() -> None:
    ok, why = fla_status(DEV, recheck=True)
    assert ok, f"the fused arm must be live on THIS card, not skipped: {why}"


def _recurrent(q, k, v, g, beta, scale):
    """The exact per-token gated-delta recurrence (KDA Eq 1), fp32, ~10 lines.

    `S_t = Diag(exp(g_t))·S_{t-1} + beta_t·k_t·(v_t − k_tᵀ·Diag(exp(g_t))·S_{t-1})`,
    `o_t = scale·q_tᵀ·S_t`. FLA's own `naive_recurrent_kda` agrees with this
    transcription to 1.5e-8 in fp32, so what it anchors is the kernel, not the
    formula. `q,k,v [b,t,h,d]`, `g [b,t,h,d]` log decay, `beta [b,t,h]`.
    """
    b, t, h, hd = q.shape
    S = torch.zeros(b, h, hd, v.shape[-1], dtype=torch.float32, device=q.device)
    o = torch.zeros(b, t, h, v.shape[-1], dtype=torch.float32, device=q.device)
    for i in range(t):
        kt, vt, qt = k[:, i].float(), v[:, i].float(), q[:, i].float()
        S = S * g[:, i].float().exp()[..., None]
        kS = torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S + kt.unsqueeze(-1) * (beta[:, i].float()[..., None] * (vt - kS)).unsqueeze(-2)
        o[:, i] = torch.einsum("bhk,bhkv->bhv", qt, S) * scale
    return o


def test_fused_arm_gradients_reach_every_kda_parameter():
    _fused_is_live()
    attn = KDAAttention(tiny_cfg(**ANCHOR), device=DEV)
    x = torch.randn(2, 64, 512, device=DEV)
    counters = Counters()
    out = attn(x, counters)
    assert counters["kda_fla"] == 1 and "kda_sdpa" not in counters
    assert kda_field(counters) == "kda=fla,n=1"
    out.float().pow(2).mean().backward()
    dead = [
        n
        for n, p in attn.named_parameters()
        if p.grad is None or not bool(torch.isfinite(p.grad).all()) or float(p.grad.norm()) == 0.0
    ]
    norms = {n: round(float(p.grad.norm()), 6) for n, p in attn.named_parameters()}
    print(f"[kda grads] {len(norms)} parameters, dead={dead}, |grad|={norms}")
    assert len(norms) == 5, "q,k,v,g + beta — the count the preset is priced by"
    assert dead == [], f"the fused arm must train every KDA parameter: dead={dead}"


def test_fused_arm_leaves_no_model_parameter_without_a_gradient(byte_tensor):
    """The SDPA fallback leaves exactly one parameter dead (`beta_proj`).

    On the fused arm it must leave none — that is what "the real arm" means, and
    it is checked on the whole tiny model rather than on the layer alone.
    """
    _fused_is_live()
    model = DormouseModel(tiny_cfg(moe_topk=0), device=DEV)
    ids, targets = (t.to(DEV) for t in byte_tensor())
    _, loss, counters = model(ids, targets)
    assert counters["kda_fla"] == model.cfg.max_iter
    loss.backward()
    allp = [n for n, _ in model.named_parameters()]
    dead = [n for n, p in model.named_parameters() if p.grad is None]
    finite = all(bool(torch.isfinite(p.grad).all()) for _, p in model.named_parameters())
    print(f"[kda grads] model {len(allp)} parameters, dead={dead}, finite={finite}")
    assert finite, "a non-finite gradient on the fused arm"
    assert dead == [], f"dead on the fused arm (the SDPA arm's set is beta_proj only): {dead}"


@pytest.mark.parametrize("bf16,bound", [(False, 3e-2), (True, 8e-2)])
def test_fused_kernel_is_anchored_on_the_recurrence(bf16, bound):
    """The anchor: the kernel against a torch-loop recurrence, same numbers.

    The spike measured 0.3 % (fp32) / 1.8 % (bf16) MEAN-relative at its own
    shape; at this one the max/mean sit higher (the kernel's chunked WY solve is
    a different summation order). The bound is loose on purpose.
    """
    _fused_is_live()
    torch.manual_seed(0)
    cfg = tiny_cfg(bf16=bf16, **ANCHOR)
    attn = KDAAttention(cfg, device=DEV).eval()
    x = torch.randn(1, 256, cfg.d_model, device=DEV)
    with torch.no_grad():
        q, k, v, g, beta = attn.projections(x)
        dt = attn.compute_dtype
        if dt != torch.float32:
            q, k, v, beta = q.to(dt), k.to(dt), v.to(dt), beta.to(dt)
        # The in-kernel qk L2 norm, reproduced the way the kernel does it: on
        # the ARM's dtype (burn-kda K3 puts L2Norm on q/k), not on an fp32 copy
        # the arm never sees — otherwise the reference is anchored on inputs the
        # kernel was not given.
        qn = F.normalize(q.float(), dim=-1).to(dt).float()
        kn = F.normalize(k.float(), dim=-1).to(dt).float()
        ref = _recurrent(qn, kn, v, g, beta, attn.scale)
        counters = Counters()
        out = attn(x, counters)
    assert counters["kda_fla"] == 1
    d = (out.float().reshape(ref.shape) - ref).abs()
    # Two denominators, because on this layer the output has a small mean and a
    # 15x larger max: `/mean` is the spike's `max_err_over_ref_abs_mean`, `/max`
    # is the plain relative error. Both are printed; the bound is asserted on
    # both (the spike measured 0.3 % / 1.8 % MEAN-relative).
    m = {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "max/refmax": float(d.max()) / float(ref.abs().max()),
        "max/refmean": float(d.max()) / float(ref.abs().mean()),
        "mean/refmean": float(d.mean()) / float(ref.abs().mean()),
    }
    print(f"[kda anchor] bf16={bf16} shape={tuple(out.shape)} {m}")
    assert m["max/refmax"] <= bound, f"{m} exceeds {bound}"
    assert m["mean/refmean"] <= bound, m


def test_the_in_kernel_l2norm_is_what_keeps_the_recurrence_finite():
    """Raw q/k (no L2 norm) EXPLODE the recurrence; the arm that ships norms them.

    Not a numeric nicety: at unit-variance projections the gated-delta state is
    amplified every step (`‖k‖² ≈ head_dim` against `beta ≈ 0.5`), so an arm that
    ran `chunk_kda(..., use_qk_l2norm_in_kernel=False)` would be a different and
    divergent model. This is the measurement that pinned it.
    """
    _fused_is_live()
    from fla.ops.kda import chunk_kda

    torch.manual_seed(0)
    b, t, h, hd = 1, 64, 8, 64
    q = torch.randn(b, t, h, hd, device=DEV)
    k = torch.randn(b, t, h, hd, device=DEV)
    v = torch.randn(b, t, h, hd, device=DEV)
    g = -0.5 * torch.rand(b, t, h, hd, device=DEV)
    beta = torch.sigmoid(torch.randn(b, t, h, device=DEV))
    raw, _ = chunk_kda(q, k, v, g=g, beta=beta, scale=hd**-0.5)
    normed, _ = chunk_kda(
        q, k, v, g=g, beta=beta, scale=hd**-0.5, use_qk_l2norm_in_kernel=True
    )
    print(
        f"[kda l2] raw |o|max={float(raw.abs().max()):.3e} "
        f"normed |o|max={float(normed.abs().max()):.3e}"
    )
    assert float(raw.abs().max()) > 1e3, "the raw-q/k blow-up this gate exists for"
    assert float(normed.abs().max()) < 1e2 and bool(torch.isfinite(normed).all())


def test_the_sdpa_fallback_normalizes_qk_like_the_kernel():
    """Arm symmetry: the fallback applies the SAME q/k L2 norm the kernel does.

    `_sdpa_path` must reproduce `F.normalize(q,k)` by hand — an un-normalized
    fallback is a different model from the FLA arm (and 1e17-explosive, see the
    test above), which is the defect this pins. No FLA call here; it runs on any
    CUDA box, and the `_FLA_INSTALLED` mark keeps it with the rest of the arm's
    gates.
    """
    torch.manual_seed(0)
    cfg = tiny_cfg(bf16=False, **ANCHOR)
    attn = KDAAttention(cfg, device=DEV).eval()
    t, d = 64, cfg.d_model
    x = torch.randn(1, t, d, device=DEV)
    with torch.no_grad():
        qr = attn._heads(attn.q_proj(x), 1, t)
        kr = attn._heads(attn.k_proj(x), 1, t)
        v = attn._heads(attn.v_proj(x), 1, t)
        gate = torch.sigmoid(attn.g_proj(x))
        got = attn._sdpa_path(x, 1, t, d)
        ref = F.scaled_dot_product_attention(
            F.normalize(qr, dim=-1) * attn.scale, F.normalize(kr, dim=-1), v, is_causal=True
        )
        ref = ref.transpose(1, 2).reshape(1, t, d) * gate
        raw = F.scaled_dot_product_attention(
            qr * attn.scale, kr, v, is_causal=True
        ).transpose(1, 2).reshape(1, t, d) * gate
    gap = float((got - raw).abs().max())
    print(
        f"[kda sdpa norm] |got-ref|max={float((got - ref).abs().max()):.2e} "
        f"|norm-raw|max={gap:.3e}"
    )
    assert torch.allclose(got, ref, atol=1e-6), "the fallback must normalize q and k"
    assert gap > 1e-3, "the norm must MOVE the output, not be decorative"
