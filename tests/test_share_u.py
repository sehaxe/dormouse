"""`share_u`: the expert bank's ONE input basis. Five gates, one property each.

The arm exists because a private `U [d, rank]` at rank 2 is two directions of
noise per expert (rank 16 measured +0.136 BPB, rank 8 collapsed, and "compose
many weak experts" under topk 4 measured +0.127). Sharing the INPUT basis and
leaving each expert its `s` and its write `V [rank, f]` is the mechanism that
can give a rank-2 bank any capacity at all, and `scripts/ab_rank.py` is what
measures it. These gates pin only what the mechanism must MECHANICALLY be:

A. `share_u = false` is bit-identical to the tree before the flag: the same
   parameter count, the same state-dict sha, the same forward logits sha, and
   an explicit `share_u=False` equals the default build. Goldens were taken
   from `HEAD`'s `experts.py`/`tsct.py` (2026-10-21, commit `183ca4c`) and are
   literals here — regenerate ONLY with a deliberate init/forward change.
B. With `share_u = true` the parameter count drops by EXACTLY the arithmetic:
   per routed expert `2*d*rank` removed, `2*d*rank` added once.
C. The shared basis receives a gradient from EVERY expert that used it: with a
   fixed LINEAR probe loss (linear, so the per-expert gradients add exactly),
   g(both) == g(expert 0 path) + g(expert 1 path), each part non-zero, each a
   proper same-direction component of the total. Run on BOTH dispatch arms —
   the reference loop and the capacity-padded bank (the bank materializes the
   shared basis E-fold, which is the one place this can silently break).
D. `share_u_forward` is counted once per `SparseMoE` forward: a 3-step run
   counts `3 * max_iter`, and a `share_u=false` run does not emit the field.
E. The forward touches no host: `.item()` / `.tolist()` raise inside it.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg

from dormouse.config import ModelConfig, validate
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.model.experts import DualLatentGLU, SparseMoE

#: Tiny model (conftest's TINY), `torch.manual_seed(0)` before every build, and
#: a fixed non-RNG input: `arange(128) * 7 % 256` reshaped to [2, 64].
IDS = torch.arange(2 * 64).reshape(2, 64) * 7 % 256
GOLDEN_PARAMS = 64_881
GOLDEN_SD_SHA = "f1ea6bb2273de0345913331fe9b747878796b4fb40ba5cc143fa678a7359a4b8"
GOLDEN_LOGITS_SHA = "d51cc54995a5024320c9357cfeb564acf576bd3b68e9b6f9e90ad3c7d917dac1"


def _build(**over) -> DormouseModel:
    torch.manual_seed(0)
    return DormouseModel(tiny_cfg(**over), device="cpu")


def _state_sha(model: DormouseModel) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().to(torch.float32).numpy().tobytes())
    return h.hexdigest()


def _logits_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().to(torch.float32).numpy().tobytes()).hexdigest()


def _forward(model: DormouseModel):
    return model(IDS, targets=torch.roll(IDS, -1, dims=1))


# --- A. the off path is today's model ---------------------------------------


def test_a_share_u_off_is_bit_identical() -> None:
    print(
        f"[gate A] {GOLDEN_PARAMS} params, sd={GOLDEN_SD_SHA[:12]}, logits={GOLDEN_LOGITS_SHA[:12]}"
    )
    default = _build()
    explicit = _build(share_u=False)

    assert not hasattr(default.loop.moe, "gate_u"), "share_u off must own no shared basis"
    assert param_count(default) == GOLDEN_PARAMS
    assert _state_sha(default) == GOLDEN_SD_SHA, (
        "share_u=false changed the initialised parameters: the flag must draw no RNG"
    )
    logits, _, _ = _forward(default)
    assert _logits_sha(logits) == GOLDEN_LOGITS_SHA, "share_u=false changed the forward"
    logits_explicit, _, _ = _forward(explicit)
    assert torch.equal(logits, logits_explicit)


# --- B. the parameter arithmetic --------------------------------------------


def test_b_shared_basis_drops_exactly_the_per_expert_arithmetic() -> None:
    off, on = _build(), _build(share_u=True)
    cfg = off.cfg
    d, f, r, e = cfg.d_model, cfg.d_ffn, cfg.rank, cfg.n_experts

    def bank(model: DormouseModel) -> int:
        return sum(
            p.numel()
            for n, p in model.loop.moe.named_parameters()
            if n.startswith("experts.") or n in ("gate_u", "up_u")
        )

    expect_off = e * (2 * (d * r + r) + r * f)
    expect_on = e * (2 * r + r * f) + 2 * d * r  # 2*r of scales + the write, per expert
    removed, added = e * 2 * d * r, 2 * d * r
    print(
        f"[gate B] d={d} r={r} E={e}: removed {removed} per-expert U params, "
        f"added {added} shared; bank {expect_off} -> {expect_on} "
        f"({100 * expect_on / expect_off:.1f}%)"
    )
    assert bank(off) == expect_off and bank(on) == expect_on
    assert removed == e * 2 * d * r == 4_096 and added == 2 * d * r == 1_024
    assert expect_off - expect_on == removed - added == 3_072
    assert param_count(off) - param_count(on) == removed - added
    # The per-expert parameter set left: its scale and its write, no input basis.
    assert [n for n, _ in on.loop.moe.experts[0].named_parameters()] == ["gate_s", "up_s", "v"]
    assert torch.equal(on.loop.moe.experts[0].gate_s, off.loop.moe.experts[0].gate_s)


# --- C. every expert feeds the shared basis ---------------------------------


def _two_expert_moe() -> tuple[SparseMoE, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    moe = SparseMoE(tiny_cfg(n_experts=2, moe_topk=2, rank=4, share_u=True, d_ffn=64), device="cpu")
    x = torch.randn(8, moe.d)
    with torch.no_grad():
        probs = F.softmax(moe.router(x), dim=-1)
        weight, index = probs.topk(2, dim=-1)
    weight = weight / weight.sum(dim=-1, keepdim=True)
    probe = torch.randn(8, moe.f, generator=torch.Generator().manual_seed(7))
    return moe, x, index, weight, probe


def _grad(
    moe: SparseMoE, x, index, weight, probe, *, zero_expert, banked
) -> dict[str, torch.Tensor]:
    """Shared-basis gradient of a LINEAR probe loss, with one expert's write zeroed.

    Linear in the bank's output on purpose: then `dL/dU` decomposes EXACTLY per
    expert (`g(both) == g(e0) + g(e1)`), which is what makes "every expert
    contributed" an equality and not a vibe. A quadratic loss would move the
    upstream factor when an expert is removed and the sum would not hold.
    """
    saved = [e.v.detach().clone() for e in moe.experts]
    with torch.no_grad():
        for i, e in enumerate(moe.experts):
            if i == zero_expert:
                e.v.data.zero_()
    moe.invalidate_banks()  # `.data` does not bump `_version`; the cache must not serve it
    moe.zero_grad(set_to_none=True)
    if banked:
        order, keep, _ = moe._permute(index)
        out = moe._padded_dispatch(x, weight, index, order, keep)
    else:
        out = moe._dispatch(x, weight, index, Counters())
    (out * probe).sum().backward()

    def _g(name: str) -> torch.Tensor:
        p = getattr(moe, name)
        return (p.grad if p.grad is not None else torch.zeros_like(p)).detach().clone()

    grads = {name: _g(name) for name in ("gate_u", "up_u")}
    with torch.no_grad():
        for e, v in zip(moe.experts, saved, strict=True):
            e.v.data.copy_(v)
    moe.invalidate_banks()
    return grads


@pytest.mark.parametrize("banked", [False, True], ids=["naive", "padded"])
def test_c_shared_basis_gets_a_gradient_from_every_expert(banked: bool) -> None:
    moe, x, index, weight, probe = _two_expert_moe()
    only0 = _grad(moe, x, index, weight, probe, zero_expert=1, banked=banked)
    only1 = _grad(moe, x, index, weight, probe, zero_expert=0, banked=banked)
    both = _grad(moe, x, index, weight, probe, zero_expert=None, banked=banked)

    for name in ("gate_u", "up_u"):
        g0, g1, gb = only0[name], only1[name], both[name]
        n0, n1, nb = g0.norm(), g1.norm(), gb.norm()
        cos0 = torch.dot(g0.reshape(-1), gb.reshape(-1)) / (n0 * nb + 1e-12)
        cos1 = torch.dot(g1.reshape(-1), gb.reshape(-1)) / (n1 * nb + 1e-12)
        print(
            f"[gate C {name} {'padded' if banked else 'naive'}] "
            f"|g(e0)|={float(n0):.4e} |g(e1)|={float(n1):.4e} |g(both)|={float(nb):.4e} "
            f"cos={float(cos0):.4f}/{float(cos1):.4f}"
        )
        assert n0 > 0 and n1 > 0, f"{name}: an expert did not reach the shared basis"
        assert float(n0) < float(nb) and float(n1) < float(nb), (
            f"{name}: one expert's removal did not shrink the total gradient "
            f"({float(n0):.3e}/{float(n1):.3e} vs {float(nb):.3e})"
        )
        assert torch.allclose(gb, g0 + g1, rtol=1e-4, atol=1e-6), (
            f"{name}: g(both) != g(e0)+g(e1) — a contribution is missing or duplicated"
        )
        assert 0.0 < float(cos0) < 0.999 and 0.0 < float(cos1) < 0.999, (
            f"{name}: an expert's part is not a proper same-direction component "
            f"of the total ({float(cos0):.4f}/{float(cos1):.4f})"
        )


def test_c_banked_and_reference_arms_compute_the_same_expert() -> None:
    """The shared basis is materialized E-fold for the bank — it must be the SAME one."""
    moe, x, index, weight, _ = _two_expert_moe()
    order, keep, _ = moe._permute(index)
    banked = moe._padded_dispatch(x, weight, index, order, keep)
    naive = moe._naive_dispatch(x, weight, index, order, keep, Counters())
    print(f"[gate C parity] max|padded-naive|={float((banked - naive).abs().max().detach()):.3e}")
    assert torch.allclose(banked, naive, rtol=1e-5, atol=1e-6)
    g_u, u_u, _, _, _ = moe._banks(torch.float32)
    assert torch.equal(g_u[0], g_u[-1]) and torch.equal(g_u[0], moe.gate_u)
    assert torch.equal(u_u[0], moe.up_u)


# --- D. the counter says the arm ran ----------------------------------------


def test_d_share_u_forward_is_counted_in_a_three_step_run() -> None:
    model = _build(share_u=True)
    counters = Counters()
    for _ in range(3):
        _, loss, _ = model(IDS, targets=torch.roll(IDS, -1, dims=1), counters=counters)
        loss.backward()
    want = 3 * model.cfg.max_iter
    print(f"[gate D] share_u_forward={counters.get('share_u_forward')} (want {want})")
    assert counters.get("share_u_forward") == want
    assert counters.drain().get("share_u_forward", 0) == 0, "the counter must be a host int"

    off = _build()
    plain = Counters()
    off(IDS, targets=torch.roll(IDS, -1, dims=1), counters=plain)
    assert "share_u_forward" not in plain, "share_u=false must not claim the arm ran"


# --- E. no host sync --------------------------------------------------------


def test_e_the_shared_forward_touches_no_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """The banked arms this arm adds read nothing back.

    Deliberately NOT the tiny model's routed forward: on CPU that routes to
    `_naive_dispatch`, whose `torch.unique(...).tolist()` is a PRE-EXISTING host
    read in the reference loop (the loop is the CPU path; the CUDA arms are the
    ones this rule exists for). What is gated here is what `share_u` adds: the
    E-fold materialization of the shared basis and its counter.
    """

    def boom(*_, **__):
        raise AssertionError("host sync inside the share_u forward (ADR-0018 rule 2)")

    torch.manual_seed(0)
    moe = SparseMoE(tiny_cfg(n_experts=4, moe_topk=0, rank=4, share_u=True), device="cpu")
    x = torch.randn(8, moe.d)
    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, "item", boom)
        m.setattr(torch.Tensor, "tolist", boom)
        counters = Counters()
        out = moe(x, counters=counters)  # the DENSE banked arm: `_banks` + bmm
        assert counters.get("share_u_forward") == 1
        index = torch.zeros(8, 2, dtype=torch.long)
        weight = torch.full((8, 2), 0.5)
        order, keep, _ = moe._permute(index)
        routed = moe._padded_dispatch(x, weight, index, order, keep)  # the routed bank
    print(f"[gate E] dense banked {tuple(out.shape)}, routed bank {tuple(routed.shape)}")
    assert torch.isfinite(out).all() and torch.isfinite(routed).all()


# --- the two LOUD refusals + the unbound seam -------------------------------


def test_refusals() -> None:
    with pytest.raises(LoudError, match="use_tsct=false"):
        validate(ModelConfig(**{**tiny_cfg().__dict__, "share_u": True, "use_tsct": False}))
    with pytest.raises(LoudError, match="share_u with factor_qat"):
        validate(ModelConfig(**{**tiny_cfg().__dict__, "share_u": True, "factor_qat": "int4"}))
    with pytest.raises(LoudError, match="share_u with factor_qat"):
        SparseMoE(tiny_cfg(share_u=True, factor_qat="int4"), device="cpu")
    with pytest.raises(LoudError, match="share_u with factor_qat"):
        DualLatentGLU(64, 128, 8, dtype=torch.float32, factor_qat="int4", share_u=True)
    unbound = DualLatentGLU(64, 128, 8, dtype=torch.float32, share_u=True)
    with pytest.raises(LoudError, match="no parent bound the shared basis"):
        unbound(torch.randn(4, 64))
