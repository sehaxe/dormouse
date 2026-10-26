"""`share_u` under `factor_qat`: ONE quantized basis per site.

The factor rung used to refuse the shared basis (`config.validate`), because the
rung's banked path projects with a per-EXPERT α and a shared `[d, rank]` plane /
`[d, share_u_rank]` dictionary has no per-expert slice. This file pins the arm
that removes that refusal:

A. `factor_qat="none"` is bit-identical to the pre-change tree (state sha and
   logits sha are literals taken before the change), builds no cache, and emits
   no `share_u_qat_forward`.
B. `alpha=0` is the unquantized share_u forward bit for bit; `alpha=1` is a
   forward whose shared basis was pre-quantized IN PLACE (the fixture that would
   fail if any shared site ignored the rung or re-projected the slice).
C. Banked-vs-standalone parity, on BOTH modes: the basis slice `_banks` feeds
   `bmm` equals the basis the standalone `DualLatentGLU` reads. `dict` is the
   mode that distinguishes quantize-then-gather from gather-then-quantize, so
   its α is the whole-dictionary absmean, never the slice's.
D. The anneal reaches the shared basis: a real 4-step `train()` run with
   `factor_qat_anneal_steps=4` shows it α = 0, .25, .5, .75 in step order.
E. The shared basis is quantized ONCE per step, not once per expert: at medium
   (E=256) the two plane sites cost 3 quantizer calls (2 basis + the `v` stack),
   against the reverting strategy's 512 (2 × E) calls.
F. The quantized shared basis adds no host sync: `item`/`tolist`/`cpu`/`numpy`/
   `__float__`/`__int__` all raise through the banked forward and a cold refresh.

No timing exists (the card is busy); these are CPU semantics gates.
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest
import torch
from conftest import tiny_cfg, tiny_run, write_corpus

from dormouse.config import load_preset
from dormouse.instrument import Counters
from dormouse.model import DormouseModel
from dormouse.model import experts as experts_mod
from dormouse.model.experts import DualLatentGLU, SparseMoE
from dormouse.model.qat_cache import take_qat_refresh_counts
from dormouse.model.tsct import TSCTLinear, factor_qat_project
from dormouse.train import train

#: The fixed, non-RNG input `tests/test_share_u.py` uses.
IDS = torch.arange(2 * 64).reshape(2, 64) * 7 % 256

#: The pre-change goldens for gate A: `share_u_mode="plane"` with
#: `factor_qat="none"` at `torch.manual_seed(0)`. The shared basis is a new
#: parameter, so this pins the arm's own none-path forward (the per-expert-U
#: control is `tests/test_share_u.py` gate A's literal).
GOLDEN_PLANE_SD = "ebbbe74ef01b86fb24ae75ea39e9398c11076ef5903c882ece32790e2ab2bb57"
GOLDEN_PLANE_LOGITS = "371c1289508fa12c0b888d95c00fa3591b763f2df8b0ef68c46a7dc9b041be8a"


def _build(**over) -> DormouseModel:
    torch.manual_seed(0)
    return DormouseModel(tiny_cfg(**over), device="cpu")


def _forward(model: DormouseModel):
    return model(IDS, targets=torch.roll(IDS, -1, dims=1))


def _sha_state(model: DormouseModel) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().to(torch.float32).numpy().tobytes())
    return h.hexdigest()


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().to(torch.float32).numpy().tobytes()).hexdigest()


def _pre_quantize_factors(model: DormouseModel, kind: str = "ternary") -> int:
    """Project every `factor_qat` site IN PLACE — the alpha=1 fixture.

    The shared basis is projected WHOLE (plane pair, or the one dictionary), the
    way the rung projects it: a fixture that gathered the dictionary columns
    first and then quantized them would bake a different α and the two forwards
    would disagree for the wrong reason.
    """
    n = 0
    moe = model.loop.moe
    with torch.no_grad():
        if moe.share_u_mode != "off":
            names = ("u_dict",) if moe.share_u_mode == "dict" else ("gate_u", "up_u")
            for name in names:
                p = getattr(moe, name)
                p.copy_(factor_qat_project(p, kind, 1.0))
                n += 1
        for m in model.modules():
            if isinstance(m, TSCTLinear):
                for p in (m.u, m.v):
                    p.copy_(factor_qat_project(p, kind, 1.0))
                    n += 1
            elif isinstance(m, DualLatentGLU):
                for name in ("gate_u", "up_u"):
                    if name in m._parameters:
                        p = m._parameters[name]
                        p.copy_(factor_qat_project(p, kind, 1.0))
                        n += 1
                m.v.copy_(factor_qat_project(m.v, kind, 1.0))
                n += 1
    return n


# ---------------------------------------------------------------------------
# A. the none path is today's forward
# ---------------------------------------------------------------------------


@pytest.mark.local_pin
def test_a_none_is_bit_identical_and_owns_no_cache() -> None:
    take_qat_refresh_counts()
    model = _build(share_u_mode="plane")
    logits, loss, counters = _forward(model)
    print(
        f"\n[gate A] sd={_sha_state(model)[:12]} logits={_sha(logits)[:12]} loss={loss.item():.6f}"
    )
    assert _sha_state(model) == GOLDEN_PLANE_SD, "factor_qat=none changed the model"
    assert _sha(logits) == GOLDEN_PLANE_LOGITS, "factor_qat=none changed the forward"
    assert model.loop.moe._qat_cache is None, "no rung, no per-step cache"
    assert all(e._shared_cache is None for e in model.loop.moe.experts)
    assert "share_u_qat_forward" not in counters, "factor_qat=none must not claim the arm"
    assert counters.get("qat_refresh_calls") is None
    assert take_qat_refresh_counts() == (0, 0), "no rung, no quantizer"

    # The rung adds no parameter and no state-dict key: it changes the forward
    # of the SAME basis, never the architecture.
    rung = _build(share_u_mode="plane", factor_qat="ternary")
    assert list(rung.state_dict()) == list(model.state_dict())


# ---------------------------------------------------------------------------
# B. the annealing floor and ceiling
# ---------------------------------------------------------------------------


def test_b_alpha_zero_is_the_unquantized_forward() -> None:
    none = _build(share_u_mode="plane")
    zero = _build(share_u_mode="plane", factor_qat="ternary")
    zero.set_factor_qat_alpha(0.0)
    ln, loss_n, cn = _forward(none)
    lz, loss_z, cz = _forward(zero)
    print(f"\n[gate B0] max|Δ| = {(lz - ln).abs().max().item():.3e}")
    assert torch.equal(ln, lz), "alpha=0 must be the master forward, bit for bit"
    assert torch.equal(loss_n, loss_z)
    assert cz.get("share_u_qat_forward") == tiny_cfg().max_iter


def test_b_alpha_one_is_the_pre_quantized_basis() -> None:
    qat = _build(share_u_mode="plane", factor_qat="ternary")
    pre = _build(share_u_mode="plane")
    pre.load_state_dict(qat.state_dict())
    n = _pre_quantize_factors(pre)
    assert n >= 3, f"the fixture found only {n} factor sites"
    lq, loss_q, _ = _forward(qat)
    lp, loss_p, _ = _forward(pre)
    print(
        f"\n[gate B1] {n} factor sites pre-quantized, max|Δ| = {(lq - lp).abs().max().item():.3e}"
    )
    assert torch.equal(lq, lp), "alpha=1 must be the pre-quantized-basis forward"
    assert torch.equal(loss_q, loss_p)


# ---------------------------------------------------------------------------
# C. banked vs standalone, on both shared modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["plane", "dict"])
def test_c_banked_and_standalone_read_the_same_quantized_basis(mode: str) -> None:
    cfg = tiny_cfg(
        n_experts=2,
        moe_topk=0,
        rank=2,
        share_u_mode=mode,
        share_u_rank=8,
        factor_qat="ternary",
    )
    torch.manual_seed(0)
    moe = SparseMoE(cfg, device="cpu")
    x = torch.randn(6, moe.d, generator=torch.Generator().manual_seed(3))
    expert = moe.experts[0]
    with torch.no_grad():
        alone = expert(x)  # the standalone path: `_bases` projects the shared basis
        gate_u_b, up_u_b, v_b, gate_s_b, up_s_b = moe._banks(torch.float32)
        ref = DualLatentGLU(cfg.d_model, cfg.d_ffn, cfg.rank, dtype=torch.float32)
        ref.gate_u.copy_(gate_u_b[0])
        ref.up_u.copy_(up_u_b[0])
        ref.v.copy_(v_b[0])
        ref.gate_s.copy_(gate_s_b[0])
        ref.up_s.copy_(up_s_b[0])
        banked = ref(x)
    diff = (banked - alone).abs().max().item()
    print(f"\n[gate C {mode}] max|Δ| = {diff:.3e}")
    assert torch.equal(banked, alone), (
        f"{mode}: the bank's quantized slice must BE the standalone expert's basis"
    )


# ---------------------------------------------------------------------------
# D. one anneal for the shared basis
# ---------------------------------------------------------------------------


def test_d_alpha_ramp_reaches_the_shared_basis(tmp_path, monkeypatch) -> None:
    seen: list[tuple[int, float]] = []
    orig = experts_mod.factor_qat_project

    def spy(w, kind, alpha, *a, **kw):
        seen.append((id(w), alpha))
        return orig(w, kind, alpha, *a, **kw)

    monkeypatch.setattr(experts_mod, "factor_qat_project", spy)
    data = write_corpus(tmp_path / "c.bin")
    run = tiny_run(tmp_path, factor_qat_anneal_steps=4, steps=4)
    run = dataclasses.replace(
        run,
        model=dataclasses.replace(run.model, factor_qat="ternary", share_u_mode="plane"),
        train=dataclasses.replace(run.train, data=str(data), log_every=100, ckpt_every=100),
    )
    model = DormouseModel(run.model)
    basis = {id(model.loop.moe.gate_u), id(model.loop.moe.up_u)}
    train(model, run, resume=False, log=lambda *_: None)

    alphas = [a for wid, a in seen if wid in basis]
    distinct: list[float] = []
    for a in alphas:
        if not distinct or distinct[-1] != a:
            distinct.append(a)
    print(f"\n[gate D] shared-basis αs in step order: {distinct}")
    assert distinct == [0.0, 0.25, 0.5, 0.75], distinct
    assert len(alphas) >= 4, "the basis must be projected every step"


# ---------------------------------------------------------------------------
# E. once per site, not once per expert
# ---------------------------------------------------------------------------


def test_e_the_basis_is_quantized_once_per_site_at_medium() -> None:
    cfg = dataclasses.replace(load_preset("medium"), share_u_mode="plane", factor_qat="ternary")
    torch.manual_seed(0)
    moe = SparseMoE(cfg)
    e = cfg.n_experts
    assert e == 256, "the measured shape's bank"
    take_qat_refresh_counts()
    with torch.no_grad():
        mats = moe._banks(torch.bfloat16)
    calls, tensors = take_qat_refresh_counts()
    print(f"\n[gate E] medium (E={e}): shared basis = {calls} calls / {tensors} tensors")
    assert [m.shape[0] for m in mats[:3]] == [e, e, e]
    assert (calls, tensors) == (3, e + 2), (
        "2 basis calls (1 tensor each) + 1 stacked `v` call (E tensors)"
    )

    # The reverting strategy, measured on the same weights: one projection per
    # expert per shared site.
    take_qat_refresh_counts()
    with torch.no_grad():
        for name in ("gate_u", "up_u"):
            for _ in range(e):
                factor_qat_project(getattr(moe, name), "ternary", 1.0)
    old_calls, old_tensors = take_qat_refresh_counts()
    print(f"[gate E] medium (E={e}): per-expert = {old_calls} calls / {old_tensors} tensors")
    assert (old_calls, old_tensors) == (2 * e, 2 * e)
    assert calls < old_calls / 100, "one projection per site, not one per expert"


# ---------------------------------------------------------------------------
# F. no host sync in the quantized shared forward
# ---------------------------------------------------------------------------


def test_f_the_quantized_basis_adds_no_host_sync(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise AssertionError("host sync in the quantized shared-basis forward")

    torch.manual_seed(0)
    moe = SparseMoE(
        tiny_cfg(n_experts=4, moe_topk=0, rank=4, share_u_mode="plane", factor_qat="ternary"),
        device="cpu",
    )
    x = torch.randn(8, moe.d, generator=torch.Generator().manual_seed(1))
    with monkeypatch.context() as m:
        for name in ("item", "tolist", "cpu", "numpy", "__float__", "__int__"):
            m.setattr(torch.Tensor, name, boom)
        counters = Counters()
        out = moe(x, counters=counters)  # dense banked: `_banks` + bmm
        assert counters.get("share_u_qat_forward") == 1
        # Force a cold refresh so the quantizer itself runs under the traps.
        moe.invalidate_banks()
        for mod in moe.modules():
            cache = getattr(mod, "_qat_cache", None)
            if cache is not None:
                cache.invalidate()
        again = moe(x)
        alone = moe.experts[0](x)
    print(f"\n[gate F] dense banked {tuple(out.shape)}, routed expert {tuple(alone.shape)}")
    assert torch.isfinite(out).all() and torch.isfinite(again).all()
    assert torch.equal(out, again), "a cold refresh must be the same forward"
