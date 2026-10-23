"""`rank_mode="uniform" | "load"` — the rank BUDGET split by routing load.

Six gates, each of which fails on the change being reverted:

A. `uniform` is BIT-IDENTICAL to the tree before the flag (pinned sha256 of
   logits, the post-step `state_dict` and the gradients, taken on this exact
   fixture at HEAD `aa28a8b`).
B. `load` starts as `uniform` (same seed, same init, same logits — the split
   only moves after the cadence fires).
C. The parameter count is EXACTLY unchanged by a reallocation — the literal
   numbers on the tiny fixture and on medium (uniform rank 32 vs load rank 32).
D. The histogram responds to traffic: all traffic to expert 0 gives it more
   rank than the rest, and the mirror traffic gives the mirror image, with the
   total held at `E·budget`.
E. No host sync in the bank build, the padded dispatch or the reallocation:
   the four conversion traps (`item`/`tolist`/`__bool__`/`__float__`) never
   fire, and the reallocation touches the host exactly twice (two small D2H
   reads) and zero times off cadence.
F. Padding is actually zero: a cold expert's padded `U`/`V` are exactly 0, its
   padded latents and `h` are exactly 0, and the rectangular bank reproduces
   the expert's own forward EXACTLY.

Plus: the loud refusals (`share_u`, `factor_qat`, the rank band, `E < 4`, the
cadence), the live-growth rule (a zero-grown `U` column would never learn), and
the checkpoint resume (the ragged shapes are rebuilt from the shapes).
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest
import torch
from conftest import tiny_cfg

from dormouse.config import RANK_CEIL, RANK_FLOOR, load_preset
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.model.experts import (
    SparseMoE,
    allocate_ranks,
    invalidate_banks_all,
    reallocate_ranks_all,
)
from dormouse.train.optim import MixOptimizer

#: A tiny load-mode config: `rank=24` is the MEAN, E=4, and the band [16, 64]
#: can hold `4·24 = 96` in more than one way (uniform 24s, or one 48/64 + floors).
LOAD = dict(
    rank=24,
    n_experts=4,
    d_model=64,
    d_ffn=128,
    moe_topk=2,
    rank_mode="load",
    rank_cadence=2,
)

#: The HEAD `aa28a8b` digests this file pins (see `_digest`): the SAME fixture,
#: the same seeds, measured on the tree before `rank_mode` existed.
HEAD_LOGITS = "7708b73300bf84a07a8f5e7534cf0dd927f3576b797abd01a89c35691a1e189f"
HEAD_STATE = "2c190372c2d59163045b9c478ce52bfdb2d4bc6aa8a69533289e821bc36bf1ca"
HEAD_GRADS = "f72a289313bb4d4351a539861d41063d525886cbadd1c4f4a28de0ebb2555df7"
HEAD_LOSS = 6.108731746673584


def _digest(items) -> str:
    h = hashlib.sha256()
    for name, t in sorted(items):
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().to(torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def _load_moe(**over) -> SparseMoE:
    cfg = tiny_cfg(**{**LOAD, **over})
    return SparseMoE(cfg, device="cpu")


def _force(moe: SparseMoE, load: list[float], budget: int = 24) -> Counters:
    """Put a load vector on the integrator and fire the cadence at step 2."""
    moe.lfb_counts.copy_(torch.tensor(load, dtype=torch.float32))
    assert moe.expert_ranks.tolist() == [budget] * moe.n_experts
    c = Counters()
    moe.reallocate_ranks(2, c)
    return c


# ---------------------------------------------------------------------------
# A. uniform is bit-identical to HEAD
# ---------------------------------------------------------------------------


def test_uniform_is_bit_identical_to_head():
    torch.manual_seed(4321)
    model = DormouseModel(tiny_cfg())
    torch.manual_seed(777)
    x = torch.randint(0, 256, (2, 64))
    y = torch.randint(0, 256, (2, 64))
    model.train()
    logits, loss, _ = model(x, y)
    assert _digest([("logits", logits)]) == HEAD_LOGITS, (
        "rank_mode='uniform' moved the forward: the default path must be the HEAD path"
    )
    loss.backward()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.step()
    assert _digest(list(model.state_dict().items())) == HEAD_STATE, (
        "uniform's state_dict moved (a new key, a buffer, or the LFB counters grew)"
    )
    assert (
        _digest([(n, p.grad) for n, p in model.named_parameters() if p.grad is not None])
        == HEAD_GRADS
    )
    assert float(loss.detach()) == HEAD_LOSS
    # The rank vector is a NON-persistent buffer: a checkpoint never carries a
    # second copy of a number the parameter shapes already hold.
    assert "loop.moe.expert_ranks" not in model.state_dict()


def test_load_starts_as_uniform():
    torch.manual_seed(99)
    mu = DormouseModel(tiny_cfg(**{**LOAD, "rank_mode": "uniform", "rank_cadence": 1000}))
    torch.manual_seed(99)
    ml = DormouseModel(tiny_cfg(**LOAD))
    # BEFORE any forward: the two models are the same model (the split has not
    # fired, so the load arm has drawn no RNG and created no different tensor).
    assert _digest(mu.state_dict().items()) == _digest(ml.state_dict().items())
    assert ml.loop.moe.expert_ranks.tolist() == [24] * 4
    assert ml.loop.moe._ragged is False and ml.loop.moe.rank_max == 24
    torch.manual_seed(5)
    x = torch.randint(0, 256, (2, 64))
    mu.train()
    ml.train()
    with torch.no_grad():
        a = mu(x)[0]
        b = ml(x)[0]
    assert torch.equal(a, b), "a load-mode model at step 0 is not the uniform model"
    # The ONE state the arm adds: the load integrator accumulates in the LOAD
    # arm even with `moe_lb_coef = 0` (the bias is inert, the load signal is
    # not) — without it the first cadence would have nothing to read.
    assert mu.loop.moe.lfb_counts.abs().sum() == 0
    assert ml.loop.moe.lfb_counts.sum() > 0


# ---------------------------------------------------------------------------
# C. the parameter count is EXACTLY unchanged
# ---------------------------------------------------------------------------


def test_tiny_parameter_count_is_unchanged_by_a_reallocation():
    torch.manual_seed(0)
    model = DormouseModel(tiny_cfg(**LOAD))
    moe = model.loop.moe
    before = param_count(model)
    # 2·d·r + r·f + 2·r = 2·64·24 + 24·128 + 2·24 = 6 192 per expert, E = 4.
    assert before == 88_593
    assert sum(p.numel() for p in moe.experts.parameters()) == 24_768
    _force(moe, [1000.0, 1.0, 1.0, 1.0])
    assert moe.expert_ranks.tolist() == [48, 16, 16, 16]
    assert sum(p.numel() for p in moe.experts.parameters()) == 24_768, (
        "a reallocation moved the expert bank's parameter count"
    )
    assert param_count(model) == before


def test_medium_parameter_count_is_unchanged_by_a_reallocation():
    cfg = dataclasses.replace(load_preset("medium"), rank=32)
    uniform = DormouseModel(cfg, device="meta")
    load = DormouseModel(dataclasses.replace(cfg, rank_mode="load"), device="meta")
    # (2d + f + 2)·E·rank = (2048 + 2730 + 2)·256·32 = 39 157 760 for the bank.
    assert param_count(uniform) == 102_646_085
    assert param_count(load) == 102_646_085
    moe = load.loop.moe
    # A fixture vector (the meta model has no data): all traffic on expert 0.
    moe.lfb_counts = torch.tensor([1000.0] + [1.0] * 255)
    moe.expert_ranks = torch.full((256,), 32, dtype=torch.long)
    c = Counters()
    moe.reallocate_ranks(0, c)
    ranks = moe.expert_ranks.tolist()
    assert sum(ranks) == 256 * 32, f"the budget moved: Σrank={sum(ranks)}"
    assert min(ranks) >= RANK_FLOOR and max(ranks) <= RANK_CEIL
    assert ranks[0] == 64 and all(r < 64 for r in ranks[1:])
    assert param_count(load) == 102_646_085, (
        "medium's parameter count moved across the reallocation"
    )
    assert sum(p.numel() for p in moe.experts.parameters()) == 39_157_760
    # An empty bucket is ABSENT (no zero counters), so sum the ones present.
    assert sum(c.get(f"pe_rank{b}", 0) for b in (16, 32, 48, 64)) == 256
    assert c["pe_rank64"] == 1


# ---------------------------------------------------------------------------
# D. the histogram responds to load
# ---------------------------------------------------------------------------


def test_histogram_follows_the_traffic():
    moe = _load_moe()
    c = _force(moe, [1000.0, 1.0, 1.0, 1.0])
    assert moe.expert_ranks.tolist() == [48, 16, 16, 16]
    assert c["pe_rank48"] == 1 and c["pe_rank16"] == 3
    assert c["pe_rank_realloc"] == 1 and c["pe_rank_changed"] == 4
    assert moe.rank_max == 48 and moe._ragged is True

    # The mirror: the same forced traffic on expert 1 must mirror the split.
    moe.lfb_counts.copy_(torch.tensor([1.0, 1000.0, 1.0, 1.0]))
    c2 = Counters()
    moe.reallocate_ranks(4, c2)
    assert moe.expert_ranks.tolist() == [16, 48, 16, 16]
    assert c2["pe_rank48"] == 1 and c2["pe_rank16"] == 3
    assert c2["pe_rank_changed"] == 2  # experts 0 and 1 swapped only

    # Flat traffic is the uniform split (the rule's fixed point).
    moe.lfb_counts.fill_(7.0)
    c3 = Counters()
    moe.reallocate_ranks(6, c3)
    assert moe.expert_ranks.tolist() == [24, 24, 24, 24]
    assert moe._ragged is False and c3["pe_rank_changed"] == 4


def test_the_allocation_rule_is_bounded_and_exact():
    # The declared rule, on the fixtures the docstring argues from.
    assert allocate_ranks([1000, 1, 1, 1], 24) == [48, 16, 16, 16]
    assert allocate_ranks([1, 1, 1, 1000], 24) == [16, 16, 16, 48]
    assert allocate_ranks([5, 5, 5, 5], 24) == [24, 24, 24, 24]
    for load in (
        [1e9, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 2, 3, 4],
        [0.5, 0.25, 12.5, 7.0],
        [1e-9, 1e9, 1e-9, 1e9],
    ):
        ranks = allocate_ranks(load, 24)
        assert sum(ranks) == 96, f"Σrank={sum(ranks)} for load={load}"
        assert all(RANK_FLOOR <= r <= RANK_CEIL for r in ranks)
        if sum(load) > 0:
            hot = max(range(4), key=lambda i: load[i])
            assert ranks[hot] == max(ranks)


def test_off_cadence_touches_nothing():
    moe = _load_moe()
    c = Counters()
    assert moe.reallocate_ranks(3, c) == []
    assert dict(c) == {}, "an off-cadence step counted something (or read the load)"
    assert moe.expert_ranks.tolist() == [24] * 4


# ---------------------------------------------------------------------------
# E. no host sync
# ---------------------------------------------------------------------------


class _Traps:
    """Patch the four conversion traps and count `.cpu()` reads."""

    names = ("item", "tolist", "__bool__", "__float__")

    def __init__(self, monkeypatch) -> None:
        self.fired: list[str] = []
        self.cpu_calls = 0
        for name in self.names:
            original = getattr(torch.Tensor, name)

            def make(name=name, original=original):
                def patched(t, *a, **k):
                    self.fired.append(name)
                    return original(t, *a, **k)

                return patched

            monkeypatch.setattr(torch.Tensor, name, make())
        original_cpu = torch.Tensor.cpu

        def cpu(t, *a, **k):
            self.cpu_calls += 1
            return original_cpu(t, *a, **k)

        monkeypatch.setattr(torch.Tensor, "cpu", cpu)


def test_no_host_sync_in_the_bank_the_dispatch_or_the_reallocation(monkeypatch):
    moe = _load_moe()
    moe.train()
    x = torch.randn(12, moe.d)
    index, weight = _routing(moe, x)

    traps = _Traps(monkeypatch)
    order, keep, _ = moe._permute(index)
    moe._padded_dispatch(x, weight, index, order, keep)
    assert traps.fired == [], f"the bank build / padded dispatch synced: {traps.fired}"
    assert traps.cpu_calls == 0

    moe.lfb_counts.copy_(torch.tensor([3.0, 2.0, 1.0, 0.0]))
    c = Counters()
    moe.reallocate_ranks(2, c)
    assert traps.fired == [], f"the reallocation used a sync trap: {traps.fired}"
    assert traps.cpu_calls == 2, (
        f"the reallocation made {traps.cpu_calls} host reads; the declared cost is 2 "
        f"(lfb_counts, expert_ranks)"
    )

    # Off cadence: zero reads, before the load tensor is even touched.
    traps.cpu_calls = 0
    moe.reallocate_ranks(3, Counters())
    assert traps.cpu_calls == 0


def _routing(moe: SparseMoE, x: torch.Tensor, seed: int = 0):
    torch.manual_seed(seed)
    with torch.no_grad():
        probs = torch.softmax(moe.router(x), dim=-1)
        weight, index = probs.topk(2, dim=-1)
    return index, weight / weight.sum(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# F. padding is exactly zero
# ---------------------------------------------------------------------------


def test_padding_is_actually_zero_and_contributes_nothing():
    moe = _load_moe()
    _force(moe, [1000.0, 1.0, 1.0, 1.0])
    gate_u, up_u, v, gate_s, up_s = moe._banks(torch.float32)
    assert tuple(gate_u.shape) == (4, 64, 48) and tuple(v.shape) == (4, 48, 128)
    for e in range(4):
        r = int(moe.expert_ranks[e])
        assert torch.equal(gate_u[e][:, r:], torch.zeros(64, 48 - r)), f"expert {e} U pad"
        assert torch.equal(up_u[e][:, r:], torch.zeros(64, 48 - r)), f"expert {e} up pad"
        assert torch.equal(v[e][r:], torch.zeros(48 - r, 128)), f"expert {e} V pad"
        assert torch.equal(gate_s[e][r:], torch.ones(48 - r)), f"expert {e} scale pad"

    # The rectangular bank reproduces the COLD expert's own forward EXACTLY, and
    # its padded latent/h are exactly zero (not merely small).
    x = torch.randn(7, moe.d)
    e = 1
    g = (x @ gate_u[e]) * gate_s[e]
    u = (x @ up_u[e]) * up_s[e]
    h = torch.nn.functional.silu(g) * u
    assert torch.equal(h[:, int(moe.expert_ranks[e]) :], torch.zeros(7, 48 - 16))
    assert torch.allclose(h @ v[e], moe.experts[e](x), atol=1e-6, rtol=1e-6), (
        "the padded bank does not reproduce the cold expert's own forward"
    )


def test_padded_dispatch_matches_the_reference_loop_with_ragged_ranks():
    moe = _load_moe()
    moe.eval()
    x = torch.randn(16, moe.d)
    index, weight = _routing(moe, x, seed=3)
    order, keep, _dropped = moe._permute(index)
    padded = moe._padded_dispatch(x, weight, index, order, keep)
    with torch.no_grad():
        reference = moe._naive_dispatch(x, weight, index, order, keep, Counters())
    assert torch.allclose(padded, reference, atol=1e-6, rtol=1e-5), (
        "the padded arm and the per-expert reference disagree once ranks are ragged"
    )


# ---------------------------------------------------------------------------
# the grown columns must LEARN (a zero-grown U column would be dead forever)
# ---------------------------------------------------------------------------


def test_growth_is_live_and_continuous():
    moe = _load_moe()
    expert = moe.experts[0]
    torch.manual_seed(11)
    x = torch.randn(9, moe.d)
    with torch.no_grad():
        before = expert(x)
    expert.set_rank(40, seed=1234)
    assert tuple(expert.gate_u.shape) == (64, 40) and tuple(expert.v.shape) == (40, 128)
    with torch.no_grad():
        after = expert(x)
    assert torch.allclose(before, after, atol=1e-6, rtol=1e-6), (
        "a rank GROWTH changed the expert's function: the new V rows must start at 0 "
        "(the residual is the GEMM's reduction order, not the new columns)"
    )
    assert torch.equal(expert.v[24:], torch.zeros(16, 128))
    assert expert.gate_u[:, 24:].abs().sum() > 0, (
        "a zero-grown U column is a dead direction (∂L/∂u = silu(g)·∂L/∂h = 0)"
    )

    opt = torch.optim.AdamW(expert.parameters(), lr=0.1)
    grads = []
    for _ in range(2):
        opt.zero_grad()
        expert(x).pow(2).mean().backward()
        grads.append(
            (
                float(expert.v.grad[24:].abs().sum()),
                float(expert.gate_u.grad[:, 24:].abs().sum()),
            )
        )
        opt.step()
    assert grads[0][0] > 0, "the grown V rows got no gradient: the growth is inert"
    assert grads[0][1] == 0.0, (
        "the grown U columns received a gradient on the first backward — the "
        "documented ordering is V-first (v rows start at zero), and breaking it "
        "means the growth init changed"
    )
    assert grads[1][1] > 0, "the grown U columns never learned: the growth is dead"


# ---------------------------------------------------------------------------
# the trainer seam: reallocate + drop the resized parameters' optimizer state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["adamw", "mix"])
def test_a_short_run_reallocates_and_stays_live(arm):
    torch.manual_seed(1)
    model = DormouseModel(tiny_cfg(**LOAD))
    moe = model.loop.moe
    opt = MixOptimizer(model, lr=1e-2, wd=0.0, opt=arm, ns_steps=2)
    torch.manual_seed(2)
    for step in range(4):
        if step % 2 == 0:
            moe.lfb_counts.copy_(
                torch.tensor([1000.0 if step == 0 else 1.0, 1.0 if step == 0 else 1000.0, 1.0, 1.0])
            )
            counters = Counters()
            resized = reallocate_ranks_all(model, step, counters)
            assert counters["pe_rank_realloc"] == 1
            if resized:
                opt.reset_state(resized)
                for old, new in resized:
                    assert old.grad is None and new.grad is None
        x = torch.randint(0, 256, (2, 16))
        y = torch.randint(0, 256, (2, 16))
        logits, loss, _ = model(x, y)
        assert torch.isfinite(loss), f"step {step}: non-finite loss after a reallocation"
        opt.zero_grad(set_to_none=True)
        loss.backward()
        dead = [n for n, p in moe.experts.named_parameters() if p.grad is None]
        assert not dead, f"step {step}: experts with no gradient: {dead}"
        opt.step()
        invalidate_banks_all(model)
        for p in moe.experts.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
    assert moe.expert_ranks.tolist() == [16, 48, 16, 16]
    # Every state entry the arm keeps is shaped like its parameter.
    for holder in (opt.adamw, getattr(opt, "muon", None)):
        if holder is None:
            continue
        for p, st in holder.state.items():
            for key, value in st.items():
                if isinstance(value, torch.Tensor) and value.dim():
                    assert value.shape == p.shape or value.shape == (p.shape[0],), (
                        f"{arm}: state[{key}] {tuple(value.shape)} != param {tuple(p.shape)}"
                    )


# ---------------------------------------------------------------------------
# the checkpoint carries the split for free (a shape IS a rank)
# ---------------------------------------------------------------------------


def test_resume_rebuilds_the_ragged_shapes_from_the_checkpoint():
    torch.manual_seed(3)
    saved = DormouseModel(tiny_cfg(**LOAD))
    _force(saved.loop.moe, [1.0, 1.0, 1.0, 1000.0])
    assert saved.loop.moe.expert_ranks.tolist() == [16, 16, 16, 48]
    sd = saved.state_dict()
    assert "loop.moe.expert_ranks" not in sd, (
        "the rank vector must not be a stored key: the parameter shapes carry it"
    )
    torch.manual_seed(3)
    fresh = DormouseModel(tiny_cfg(**LOAD))
    assert fresh.loop.moe.expert_ranks.tolist() == [24] * 4
    fresh.load_state_dict(sd, strict=True)
    assert fresh.loop.moe.expert_ranks.tolist() == [16, 16, 16, 48]
    assert fresh.loop.moe.rank_max == 48 and fresh.loop.moe._ragged is True
    for name, tensor in saved.state_dict().items():
        assert torch.equal(fresh.state_dict()[name], tensor), f"{name} did not round-trip"


# ---------------------------------------------------------------------------
# loud refusals
# ---------------------------------------------------------------------------

REFUSALS = [
    ({"rank_mode": "bogus"}, "rank_mode='bogus'"),
    ({"rank_mode": "load", "rank": RANK_FLOOR - 1}, "rank=15"),
    ({"rank_mode": "load", "rank": RANK_CEIL + 1}, "rank=65"),
    ({"rank_mode": "load", "n_experts": 3}, "n_experts=3"),
    ({"rank_mode": "load", "rank_cadence": 0}, "rank_cadence=0"),
    ({"rank_mode": "load", "share_u_mode": "plane"}, "share_u_mode='plane'"),
    ({"rank_mode": "load", "share_u_mode": "dict"}, "share_u_mode='dict'"),
    ({"rank_mode": "load", "share_u": True}, "share_u_mode='plane'"),
    ({"rank_mode": "load", "factor_qat": "ternary"}, "factor_qat='ternary'"),
]


@pytest.mark.parametrize("over,needle", REFUSALS)
def test_load_mode_refusals_are_loud(over, needle):
    from dormouse.config import validate

    cfg = tiny_cfg(**{**LOAD, **over})
    with pytest.raises(LoudError, match=needle):
        validate(cfg)
    # The module refuses the same config a hand-built loader would hand it —
    # `validate` is bypassed, so the constructor is the second lock.
    with pytest.raises(LoudError, match=needle):
        SparseMoE(cfg, device="cpu")


# ---------------------------------------------------------------------------
# the trainer seam end to end (the cadence call is wired into `train()`)
# ---------------------------------------------------------------------------


def test_the_trainer_fires_the_cadence(tmp_path):
    import dataclasses as dc

    from conftest import tiny_run, write_corpus

    from dormouse.model import DormouseModel
    from dormouse.train import train

    data = write_corpus(tmp_path / "c.bin")
    run = tiny_run(tmp_path, steps=6)
    run = dc.replace(
        run,
        model=dc.replace(
            run.model, rank=24, n_experts=4, moe_topk=2, rank_mode="load", rank_cadence=3
        ),
        train=dc.replace(run.train, data=str(data), log_every=100, ckpt_every=100),
    )
    model = DormouseModel(run.model)
    report = train(model, run, resume=False, log=lambda *_: None)
    fired = report["counters"].get("pe_rank_realloc", 0) + report["counters"].get(
        "pe_rank_no_load", 0
    )
    assert fired == 2, "the trainer did not fire the rank cadence at steps 0 and 3"
    # Step 0 fires before any forward, so its integrator is empty and the split
    # stays uniform (COUNTED `pe_rank_no_load`); step 3 carries real load.
    assert report["counters"].get("pe_rank_realloc") == 1
    assert report["counters"].get("pe_rank_no_load") == 1
    assert report["nan_skipped"] == 0


# ---------------------------------------------------------------------------
# the A/B driver is pre-registered and its arms are the two modes
# ---------------------------------------------------------------------------


def test_ab_driver_is_pre_registered():
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "scripts" / "ab_pe_rank.py").read_text()
    assert re.search(r'ARMS[^=]*=\s*\{\s*"uniform"', text), "the control arm is missing"
    assert '"load"' in text, "the load arm is missing"
    assert "SEEDS = (1, 2, 3)" in text and "STEPS = 600" in text
    assert "data_local/mix4/web.bin" in text and "web_eval.bin" in text
