"""`share_u_mode="dict"`: per-expert subspaces of ONE shared dictionary.

`share_u_mode="plane"` gives every expert the SAME `rank`-dimensional input
plane, so at rank 2 the whole bank can only see two directions of a `d`-dim
state (`tests/test_share_u.py` is that arm's control). `dict` keeps the one
shared `[d, share_u_rank]` dictionary and lets each expert read `rank` of its
columns, so two experts see DIFFERENT subspaces. These gates pin the mechanism,
not its quality — no BPB claim is made here; `scripts/ab_rank2.py` is what
measures the arm, and it has not run.

A. `share_u_mode="off"` is bit-identical to the tree before the field existed:
   the same parameter count, state-dict sha and forward-logits sha as
   `test_share_u.py`'s goldens (taken from `HEAD` at commit `183ca4c`), and
   `share_u=True` resolves to the same model as `share_u_mode="plane"`.
B. The parameter arithmetic, exactly: tiny `d=64, f=32, E=4, R=8, rank=2`
   asserted against the hand sum, AND the medium-scale numbers asserted as
   literals (`r64` 78 315 520 / `r2s` 1 402 880 / `r2d` 1 464 320 / `r4d`
   2 863 104) so the printed table in `scripts/ab_rank2.py` cannot drift.
C. Two experts whose index rows differ produce different latent projections and
   different outputs for the same input — literal numbers printed — while the
   `plane` control's two experts project identically (the same plane).
D. The dictionary receives a gradient from EVERY expert that read it: support of
   `dL/du_dict` is exactly the gathered columns, each expert's part sums to the
   whole (`g(both) == g(e0) + g(e1)`), on BOTH dispatch arms (reference loop and
   capacity-padded bank, which builds the `[E, d, rank]` stack itself).
E. The forward touches no host (`.item()` / `.tolist()` raise inside it).
F. The LOUD refusals: `R < 2`, an out-of-range/float index, `dict` with an
   unknown `factor_qat`, `dict` with `use_tsct=false`, `rank >= R`, and the
   `share_u` alias disagreeing with `share_u_mode`.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg
from test_share_u import GOLDEN_LOGITS_SHA, GOLDEN_PARAMS, GOLDEN_SD_SHA

from dormouse.config import ModelConfig, validate
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.model.experts import (
    DualLatentGLU,
    SparseMoE,
    _check_dict_index,
    share_u_dict_index,
)

#: The same fixed, non-RNG input `test_share_u.py` uses.
IDS = torch.arange(2 * 64).reshape(2, 64) * 7 % 256


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


# --- A. the off path and the alias ------------------------------------------


def test_a_off_is_bit_identical_and_share_u_is_the_plane_alias() -> None:
    print(f"[gate A] {GOLDEN_PARAMS} params, logits={GOLDEN_LOGITS_SHA[:12]}")
    default = _build()
    assert default.cfg.share_u_mode == "off", "the mode must default to off"
    assert param_count(default) == GOLDEN_PARAMS
    assert _state_sha(default) == GOLDEN_SD_SHA, "the new field changed the init"
    logits, _, _ = _forward(default)
    assert _logits_sha(logits) == GOLDEN_LOGITS_SHA, "the new field changed the forward"

    explicit = _build(share_u_mode="off")
    assert torch.equal(logits, _forward(explicit)[0])

    # `share_u = true` IS `share_u_mode = "plane"`: one switch, two spellings.
    alias = _build(share_u=True, rank=2)
    plane = _build(share_u_mode="plane", rank=2)
    assert alias.cfg.share_u_mode == "plane" == plane.cfg.share_u_mode
    assert _state_sha(alias) == _state_sha(plane)
    assert torch.equal(_forward(alias)[0], _forward(plane)[0])

    # `--set share_u=False` still means OFF (the CLI's bool coercion keeps it a
    # bool; the alias then leaves the mode at its default).
    from dormouse.cli import _split_sets

    model_over, _ = _split_sets(["share_u=False", "share_u_mode=dict"])
    assert model_over == {"share_u": False, "share_u_mode": "dict"}
    cfg = ModelConfig(**{**tiny_cfg().__dict__, **model_over, "rank": 2})
    assert cfg.share_u_mode == "dict", "share_u=False must not force the plane"

    off = ModelConfig(**{**tiny_cfg().__dict__, "share_u": False})
    assert off.share_u_mode == "off"


# --- B. the parameter arithmetic --------------------------------------------


def _bank_params(moe: SparseMoE) -> int:
    return sum(
        p.numel()
        for n, p in moe.named_parameters()
        if n.startswith("experts.") or n == "u_dict" or n in ("gate_u", "up_u")
    )


def test_b_the_bank_arithmetic_is_exact() -> None:
    d, f, e, big_r, r = 64, 32, 4, 8, 2
    off = SparseMoE(tiny_cfg(d_ffn=f, n_experts=e, rank=r), device="cpu")
    plane = SparseMoE(tiny_cfg(d_ffn=f, n_experts=e, rank=r, share_u_mode="plane"), device="cpu")
    dict_ = SparseMoE(
        tiny_cfg(d_ffn=f, n_experts=e, rank=r, share_u_mode="dict", share_u_rank=big_r),
        device="cpu",
    )

    expect_off = e * (2 * (d * r + r) + r * f)
    expect_plane = e * (2 * r + r * f) + 2 * d * r
    expect_dict = e * (2 * r + r * f) + d * big_r
    print(
        f"[gate B] d={d} f={f} E={e} R={big_r} r={r}: "
        f"off={_bank_params(off)} ({expect_off}), plane={_bank_params(plane)} "
        f"({expect_plane}), dict={_bank_params(dict_)} ({expect_dict})"
    )
    assert _bank_params(off) == expect_off == 1_296
    assert _bank_params(plane) == expect_plane == 528
    assert _bank_params(dict_) == expect_dict == 784
    # The dict's per-expert parameter set is the scales and the write, nothing else.
    assert [n for n, _ in dict_.experts[0].named_parameters()] == ["gate_s", "up_s", "v"]
    assert tuple(dict_.u_dict.shape) == (d, big_r)
    assert not hasattr(dict_, "gate_u"), "dict mode owns a dictionary, not a plane"
    assert tuple(dict_.share_u_idx.shape) == (2, e, r)

    # The removal/addition arithmetic: E private [d, r] pairs out, ONE [d, R] in.
    removed, added = e * 2 * d * r, d * big_r
    assert _bank_params(off) - _bank_params(dict_) == removed - added == 512
    assert (
        param_count(DormouseModel(tiny_cfg(d_ffn=f, n_experts=e, rank=r)))
        - (
            param_count(
                DormouseModel(
                    tiny_cfg(
                        d_ffn=f,
                        n_experts=e,
                        rank=r,
                        share_u_mode="dict",
                        share_u_rank=big_r,
                    )
                )
            )
        )
        == removed - added
    )

    # MEDIUM (preset `medium`: d=1024, f=2730, E=256), the numbers
    # `scripts/ab_rank2.py` prints. Literals on purpose: a header that drifts
    # from this arithmetic is a number that reads true and lies.
    D, F_, E = 1024, 2730, 256
    r64 = E * (2 * (D * 64 + 64) + 64 * F_)
    r2s = E * (2 * 2 + 2 * F_) + 2 * D * 2
    r2d = E * (2 * 2 + 2 * F_) + D * 64
    r4d = E * (2 * 4 + 4 * F_) + D * 64
    print(
        f"[gate B medium] r64={r64:,} r2s={r2s:,} r2d={r2d:,} r4d={r4d:,} "
        f"-> r2d is {r64 / r2d:.1f}x under the control, "
        f"{r2d - r2s:,} over the rank-2 plane"
    )
    assert (r64, r2s, r2d, r4d) == (78_315_520, 1_402_880, 1_464_320, 2_863_104)


# --- C. the subspaces are not the same plane --------------------------------


def _dict_moe(n_experts: int = 2, *, rank: int = 2, big_r: int = 8) -> SparseMoE:
    torch.manual_seed(0)
    return SparseMoE(
        tiny_cfg(
            n_experts=n_experts,
            moe_topk=n_experts,
            rank=rank,
            share_u_mode="dict",
            share_u_rank=big_r,
            d_ffn=64,
        ),
        device="cpu",
    )


def test_c_two_experts_read_different_subspaces() -> None:
    moe = _dict_moe()
    x = torch.randn(8, moe.d, generator=torch.Generator().manual_seed(3))
    idx = moe.share_u_idx
    print(f"[gate C] gate rows {idx[0].tolist()}, up rows {idx[1].tolist()}")
    assert set(idx[0, 0].tolist()) != set(idx[0, 1].tolist()), (
        "the two experts must not be assigned the same column set"
    )

    g0, u0 = moe.experts[0]._bases()
    g1, u1 = moe.experts[1]._bases()
    assert not torch.equal(g0, g1), "dict mode handed both experts the same gate basis"
    assert not torch.equal(u0, u1), "dict mode handed both experts the same up basis"
    assert tuple(g0.shape) == (moe.d, 2)

    lat0, lat1 = x @ g0, x @ g1
    with torch.no_grad():
        gap = float((lat0 - lat1).norm() / lat0.norm())
    out0, out1 = moe.experts[0](x), moe.experts[1](x)
    with torch.no_grad():
        out_gap = float((out0 - out1).norm() / out0.norm())
    print(
        f"[gate C dict] |xU0-xU1|/|xU0|={gap:.4e}  |y0-y1|/|y0|={out_gap:.4e}\n"
        f"           lat0[0]={lat0[0].tolist()}  lat1[0]={lat1[0].tolist()}"
    )
    assert gap > 1e-3, "the two 2-D projections coincide: the same subspace"
    assert out_gap > 1e-3, "the two experts' outputs coincide for the same input"

    # The CONTROL, on the same input: `plane` hands both experts the SAME basis,
    # so their projections are bit-equal. That is the defect `dict` removes.
    torch.manual_seed(0)
    plane = SparseMoE(
        tiny_cfg(n_experts=2, moe_topk=2, rank=2, share_u_mode="plane", d_ffn=64),
        device="cpu",
    )
    pg0, pu0 = plane.experts[0]._bases()
    pg1, pu1 = plane.experts[1]._bases()
    with torch.no_grad():
        plane_gap = float((x @ pg0 - x @ pg1).norm())
    print(f"[gate C plane] same basis: {torch.equal(pg0, pg1)}  |xU0-xU1|={plane_gap:.1e}")
    assert torch.equal(pg0, pg1) and torch.equal(pu0, pu1)
    assert plane_gap == 0.0


# --- D. every expert feeds the dictionary -----------------------------------


def _grad(moe: SparseMoE, x, index, weight, probe, *, zero_expert, banked) -> torch.Tensor:
    """`dL/du_dict` of a LINEAR probe loss with one expert's write zeroed.

    Linear in the bank's output so `dL/dU` decomposes EXACTLY per expert — a
    quadratic loss would move the upstream factor when an expert is removed and
    "every expert contributed" would stop being an equality.
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
    grad = moe.u_dict.grad if moe.u_dict.grad is not None else torch.zeros_like(moe.u_dict)
    grad = grad.detach().clone()
    with torch.no_grad():
        for e, v in zip(moe.experts, saved, strict=True):
            e.v.data.copy_(v)
    moe.invalidate_banks()
    return grad


@pytest.mark.parametrize("banked", [False, True], ids=["naive", "padded"])
def test_d_the_dictionary_gets_every_experts_gathered_columns(banked: bool) -> None:
    moe = _dict_moe()
    x = torch.randn(8, moe.d, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        probs = F.softmax(moe.router(x), dim=-1)
        weight, index = probs.topk(2, dim=-1)
    weight = weight / weight.sum(dim=-1, keepdim=True)
    probe = torch.randn(8, moe.f, generator=torch.Generator().manual_seed(7))

    only0 = _grad(moe, x, index, weight, probe, zero_expert=1, banked=banked)
    only1 = _grad(moe, x, index, weight, probe, zero_expert=0, banked=banked)
    both = _grad(moe, x, index, weight, probe, zero_expert=None, banked=banked)

    cols0 = set(moe.share_u_idx[:, 0].reshape(-1).tolist())
    cols1 = set(moe.share_u_idx[:, 1].reshape(-1).tolist())
    assert set(moe.share_u_idx[0, 0].tolist()) != set(moe.share_u_idx[0, 1].tolist())

    def support(g: torch.Tensor) -> set[int]:
        return {c for c in range(g.shape[1]) if float(g[:, c].abs().sum()) > 0.0}

    print(
        f"[gate D {'padded' if banked else 'naive'}] expert0 cols {sorted(cols0)} "
        f"support {sorted(support(only0))}; expert1 cols {sorted(cols1)} "
        f"support {sorted(support(only1))}\n"
        f"           |g0|={float(only0.norm()):.4e} |g1|={float(only1.norm()):.4e} "
        f"|g(both)|={float(both.norm()):.4e}"
    )
    assert support(only0) == cols0, "expert 0's gradient left its own columns"
    assert support(only1) == cols1, "expert 1's gradient left its own columns"
    assert support(both) == cols0 | cols1
    # Each gathered column received something, and the whole is the sum of parts.
    for name, e in (("gate", 0), ("up", 1)):
        for col in moe.share_u_idx[e, 0].tolist():
            assert float(only0[:, col].abs().sum()) > 0, f"expert 0's {name} column {col}"
    assert torch.allclose(both, only0 + only1, rtol=1e-4, atol=1e-6), (
        "g(both) != g(e0)+g(e1): an expert's contribution is missing or duplicated"
    )


def test_d_the_banked_stack_is_the_gathered_dictionary() -> None:
    """The `[E, d, rank]` bank the padded/triton arms consume is the SAME gather."""
    moe = _dict_moe()
    gate, up, v, gate_s, up_s = moe._banks(torch.float32)
    for e, expert in enumerate(moe.experts):
        want_g, want_u = expert._bases()
        assert torch.equal(gate[e], want_g), f"expert {e}'s bank gate row is not its columns"
        assert torch.equal(up[e], want_u)
    print(
        f"[gate D bank] {tuple(gate.shape)} from {tuple(moe.u_dict.shape)}; "
        f"rows 0 and 1 differ: {not torch.equal(gate[0], gate[1])}"
    )
    assert gate.shape == (moe.n_experts, moe.d, 2)


# --- E. no host sync --------------------------------------------------------


def test_e_the_dict_forward_touches_no_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not the naive loop (its `torch.unique(...).tolist()` is pre-existing, the
    CPU reference path); what `dict` adds is the gather and the counter."""

    def boom(*_, **__):
        raise AssertionError("host sync inside the share_u dict forward (ADR-0018 rule 2)")

    torch.manual_seed(0)
    moe = SparseMoE(
        tiny_cfg(
            n_experts=4,
            moe_topk=0,
            rank=2,
            share_u_mode="dict",
            share_u_rank=8,
        ),
        device="cpu",
    )
    x = torch.randn(8, moe.d)
    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, "item", boom)
        m.setattr(torch.Tensor, "tolist", boom)
        counters = Counters()
        out = moe(x, counters=counters)  # the DENSE banked arm: `_banks` + bmm
        assert counters.get("share_u_dict_forward") == 1
        assert "share_u_forward" not in counters
        index = torch.zeros(8, 2, dtype=torch.long)
        weight = torch.full((8, 2), 0.5)
        order, keep, _ = moe._permute(index)
        routed = moe._padded_dispatch(x, weight, index, order, keep)  # the routed bank
    print(f"[gate E] dense banked {tuple(out.shape)}, routed bank {tuple(routed.shape)}")
    assert torch.isfinite(out).all() and torch.isfinite(routed).all()


def test_e_the_counters_split_plane_from_dict() -> None:
    for mode, field in (("plane", "share_u_forward"), ("dict", "share_u_dict_forward")):
        model = _build(rank=2, share_u_rank=8, share_u_mode=mode)
        counters = Counters()
        for _ in range(3):
            _, loss, _ = model(IDS, targets=torch.roll(IDS, -1, dims=1), counters=counters)
            loss.backward()
        want = 3 * model.cfg.max_iter
        print(f"[gate E {mode}] {field}={counters.get(field)} (want {want})")
        assert counters.get(field) == want
        other = "share_u_dict_forward" if mode == "plane" else "share_u_forward"
        assert other not in counters, f"{mode} claimed the other arm's counter"


# --- F. the LOUD refusals ---------------------------------------------------


def _validate(**over) -> None:
    validate(ModelConfig(**{**tiny_cfg().__dict__, **over}))


def test_f_refusals() -> None:
    with pytest.raises(LoudError, match="use_tsct=false"):
        _validate(share_u_mode="dict", use_tsct=False)
    with pytest.raises(LoudError, match="share_u with factor_qat"):
        _validate(share_u_mode="dict", factor_qat="int4")
    with pytest.raises(LoudError, match="share_u_rank"):
        _validate(share_u_mode="dict", share_u_rank=1)
    with pytest.raises(LoudError, match="no subset"):
        _validate(share_u_mode="dict", rank=8, share_u_rank=8)
    with pytest.raises(LoudError, match="is not an arm"):
        ModelConfig(**{**tiny_cfg().__dict__, "share_u_mode": "dictionary"})
    with pytest.raises(LoudError, match="two switches"):
        ModelConfig(**{**tiny_cfg().__dict__, "share_u": True, "share_u_mode": "dict"})

    # The module refuses the same pairs on a hand-built config (no loader).
    with pytest.raises(LoudError, match="share_u with factor_qat"):
        SparseMoE(tiny_cfg(share_u_mode="dict", factor_qat="int4"), device="cpu")
    with pytest.raises(LoudError, match="use_tsct=false"):
        SparseMoE(tiny_cfg(share_u_mode="dict", use_tsct=False), device="cpu")
    with pytest.raises(LoudError, match="share_u_rank"):
        SparseMoE(tiny_cfg(share_u_mode="dict", share_u_rank=1), device="cpu")
    with pytest.raises(LoudError, match="no subset"):
        SparseMoE(tiny_cfg(share_u_mode="dict", rank=8, share_u_rank=8), device="cpu")
    with pytest.raises(LoudError, match="is not an arm"):
        SparseMoE(tiny_cfg(share_u_mode="nope"), device="cpu")

    # An index that cannot gather: out of range, and a float index (which
    # `index_select` would silently truncate into a different subspace).
    with pytest.raises(LoudError, match="out of range"):
        _check_dict_index(torch.tensor([[[0, 1], [2, 99]], [[0, 1], [2, 3]]]), 8, 2, "test")
    with pytest.raises(LoudError, match="int64"):
        _check_dict_index(torch.zeros(2, 4, 2), 8, 2, "test")
    with pytest.raises(LoudError, match="must be \\[2, E, 2\\]"):
        _check_dict_index(torch.zeros(2, 4, 3, dtype=torch.long), 8, 2, "test")
    with pytest.raises(LoudError, match="dict_rank"):
        share_u_dict_index(4, 1, 1)
    with pytest.raises(LoudError, match="dict_rank"):
        share_u_dict_index(4, 4, 4)
    with pytest.raises(LoudError, match="no parent bound the shared basis"):
        DualLatentGLU(64, 128, 8, dtype=torch.float32, share_u=True)(torch.randn(4, 64))


def test_f_the_index_is_fixed_and_seeded() -> None:
    """Two builds of one config see the same subspaces; a resume does not move them."""
    a = share_u_dict_index(4, 8, 2)
    b = share_u_dict_index(4, 8, 2)
    c = share_u_dict_index(4, 8, 2, seed=1)
    print(
        f"[gate F] idx again equal: {torch.equal(a, b)}; "
        f"another seed differs: {not torch.equal(a, c)}"
    )
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    for e in range(4):
        assert len(set(a[0, e].tolist())) == 2 and len(set(a[1, e].tolist())) == 2
        assert set(a[0, e].tolist()) != set(a[1, e].tolist())
