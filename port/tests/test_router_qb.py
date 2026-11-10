"""Quantile Balancing: the router bias set from the score quantile (K3 §2.3.3).

The mechanism (`SparseMoE._qb_set_bias`) replaces the fixed-rate bias step with
the quantile that names each expert's target load `p_i = topk/E`: raise the
expert's threshold until only `p_i` of THIS batch's scores clear it. These gates
are on synthetic scores only (a router whose scores we control), so the loads
are arithmetic, not a training result:

a. a SKEWED score bank (raw loads 0.52/0.28/0.13/0.06 against a 0.25 target)
   converges to within ±10% of every target after 50 updates;
b. an already-balanced bank (each token's score vector is a permutation across
   experts) is a fixed point: the bias moves by < 1e-3 over 50 updates;
c. the update function is device-side: its source has no scalar extraction,
   list conversion or CPU copy (a CUDA graph has to hold it);
d. `moe_qb` is refused LOUDLY with no router (`n_experts=1`) and on the dense
   blend (`moe_topk=0`), each naming the escape;
e. flag OFF keeps the legacy `sign` rule to the bit and never counts `moe_qb`.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from conftest import tiny_cfg

from dormouse.instrument import Counters, LoudError
from dormouse.model.experts import SparseMoE

N = 8192


def skewed_scores(n_experts: int, seed: int = 0) -> torch.Tensor:
    """`randn` per expert plus a mean ramp: expert 0 is the busiest, by a lot."""
    g = torch.Generator().manual_seed(seed)
    means = torch.linspace(1.5, 0.0, n_experts)
    return torch.randn(N, n_experts, generator=g) + means


def loads(moe: SparseMoE, scores: torch.Tensor, topk: int) -> torch.Tensor:
    """The router's share per expert: top-k on the RAISED scores, `[E]` summing to 1."""
    index = (scores + moe.lfb_bias).topk(topk, dim=-1).indices
    counts = torch.bincount(index.reshape(-1), minlength=moe.n_experts).float()
    return counts / (scores.shape[0] * topk)


def routed_step(moe: SparseMoE, scores: torch.Tensor, topk: int, counters: Counters) -> None:
    """One routed forward's controller call, the shape `forward` uses."""
    index = (scores + moe.lfb_bias).topk(topk, dim=-1).indices
    counts = torch.bincount(index.reshape(-1), minlength=moe.n_experts)
    moe._qb_step(counts, counters, scores)


# --- a. skewed scores converge to the target load ----------------------------


@pytest.mark.parametrize("topk", [1, 2])
def test_skewed_scores_reach_the_target_load(topk: int):
    cfg = tiny_cfg(moe_qb=True, moe_topk=topk)
    moe = SparseMoE(cfg)
    scores = skewed_scores(cfg.n_experts)
    # `loads` is a SHARE of the assignments, so a uniform router's is 1/E at
    # any width (`p_i = topk/E` is the per-TOKEN fraction the controller reads).
    target = 1.0 / cfg.n_experts

    raw = loads(moe, scores, topk)
    assert (raw - target).abs().max() > 0.10 * target, (
        f"the fixture is not skewed at topk={topk}: raw loads {raw.tolist()}"
    )

    counters = Counters()
    for _ in range(50):
        routed_step(moe, scores, topk, counters)
    counters.drain()

    got = loads(moe, scores, topk)
    assert torch.all((got - target).abs() <= 0.10 * target), (
        f"target {target:.3f} (topk={topk}), loads {[round(x, 4) for x in got.tolist()]}, "
        f"raw {[round(x, 4) for x in raw.tolist()]}"
    )
    assert counters["moe_qb"] == 50, "every update must be COUNTED"
    assert "qb:" in str(counters.why), f"no why entry: {counters.why!r}"


# --- b. an already-balanced bank is a fixed point ----------------------------


def test_balanced_scores_are_a_fixed_point():
    cfg = tiny_cfg(moe_qb=True, moe_topk=2)
    moe = SparseMoE(cfg)
    g = torch.Generator().manual_seed(1)
    base = torch.randn(N, 1, generator=g).repeat(1, cfg.n_experts)
    # Every expert's score column is a permutation of the SAME values, so all
    # four quantiles are identical and the rule has nothing to move.
    perm = torch.stack([torch.randperm(cfg.n_experts, generator=g) for _ in range(N)])
    scores = torch.gather(base, 1, perm)
    before = moe.lfb_bias.clone()

    for _ in range(50):
        moe._qb_set_bias(scores)

    moved = float((moe.lfb_bias - before).abs().max())
    assert moved < 1e-3, f"the bias moved {moved:.2e} on a balanced bank: {moe.lfb_bias}"


# --- the whole routed forward runs it ---------------------------------------


def test_forward_runs_the_qb_controller():
    """The `need` gate and the dispatch: a real routed forward counts and moves it."""
    cfg = tiny_cfg(moe_qb=True, moe_topk=2)
    moe = SparseMoE(cfg)
    torch.manual_seed(0)
    x = torch.randn(64, cfg.d_model)
    counters = Counters()
    out = moe(x, counters)
    assert out.shape == (64, cfg.d_model)
    assert counters["moe_qb"] == 1, f"the routed forward skipped QB: {dict(counters)}"
    assert "moe_lfb_updates" not in counters
    assert float(moe.lfb_counts.sum()) == 64 * cfg.moe_topk
    assert torch.isfinite(moe.lfb_bias).all()


# --- c. the update is device-side --------------------------------------------


def test_update_function_never_leaves_the_device():
    src = inspect.getsource(SparseMoE._qb_set_bias)
    for banned in (".item(", ".tolist(", ".cpu(", ".numpy(", "nonzero"):
        assert banned not in src, f"`{banned}` in the step path: a host sync a graph cannot hold"


# --- d. LOUD refusals --------------------------------------------------------


def test_refuses_no_router():
    with pytest.raises(LoudError, match="n_experts=1") as e:
        SparseMoE(tiny_cfg(moe_qb=True, n_experts=1, moe_topk=1))
    assert "Escape:" in str(e.value)


def test_refuses_dense_blend():
    with pytest.raises(LoudError, match="moe_topk=0") as e:
        SparseMoE(tiny_cfg(moe_qb=True, moe_topk=0))
    assert "Escape:" in str(e.value)


# --- e. OFF keeps the legacy sign rule --------------------------------------


def test_off_follows_the_legacy_sign_rule():
    cfg = tiny_cfg(moe_qb=False, moe_topk=2, moe_lb_mode="sign", moe_lb_coef=0.05)
    moe = SparseMoE(cfg)
    scores = skewed_scores(cfg.n_experts)
    index = (scores + moe.lfb_bias).topk(cfg.moe_topk, dim=-1).indices
    counts = torch.bincount(index.reshape(-1), minlength=cfg.n_experts).float()

    counters = Counters()
    moe._lfb_step(index, counters, counts)

    expected = -moe.lfb_coef * torch.sign(counts - counts.mean())
    assert torch.allclose(moe.lfb_bias, expected), (
        f"legacy sign step moved the bias to {moe.lfb_bias.tolist()}, "
        f"the formula says {expected.tolist()}"
    )
    assert torch.equal(moe.lfb_counts, counts)
    assert counters.get("moe_lfb_updates") == 1
    assert "moe_qb" not in counters, "the QB arm counted while OFF"
