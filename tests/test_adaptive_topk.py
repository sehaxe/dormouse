"""The adaptive-top-k gates (each one fails if its property breaks).

The mechanism is `model/adaptive_topk.py` + one masked top-k in
`SparseMoE.forward`: the router already computes the full logit vector, so a
per-token width costs no parameters and no training. What is gated here:

1. `mode="off"` is TODAY's model, bit for bit — a SHA-256 of the forward's
   float32 bytes pinned on the commit BEFORE the policy existed, plus the pinned
   counter line, plus a spy proving the policy is never called on that arm.
2. Every token gets exactly its chosen `k` real slots: the slot mask sums to
   `k`, the real slots are distinct experts and are the top-k prefix, unused
   weights are exactly 0 and no weight is NaN. The padded bank and the naive
   loop agree on the same mixed-width routing (the sentinel-group arithmetic).
3. The `entropy` policy on a crafted router: uniform rows -> `k_max`, peaked
   rows -> `k_min`, as LITERALS (the formula, not "it moves").
4. The width histogram counts every routed token exactly once.
5. The adaptive forward touches no host (`item`/`tolist`/`bool`/`int`/`float`
   raise) — the same cadence gate as `tests/test_mtp.py` gate 5.
6. Unknown mode, `k_min > k_max`, `k_max > 8`, `k_max > n_experts`, a dense
   (`moe_topk = 0`) adaptive arm and an inverted confidence window are LOUD.

The `margin` policy's direction and the config defaults (off, inert knobs) are
pinned next to them.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg

from dormouse.config import ModelConfig, load_preset, parse_preset, validate
from dormouse.instrument import Counters, LoudError
from dormouse.model import adaptive_topk
from dormouse.model import experts as experts_mod
from dormouse.model.adaptive_topk import choose_k, select_topk
from dormouse.model.experts import SparseMoE

#: SHA-256 of `out.detach().contiguous().numpy().tobytes()` for the tiny MoE
#: forward below, measured on HEAD `183ca4c` BEFORE `adaptive_topk.py` existed
#: (fp32 CPU, seed 0, `x` seed 7). Our own transcription — no external
#: reference exists for a weight hash.
OFF_DIGEST = "330826e9292871ac8aadfeb1f44ffd5c5c350e378da66bd3c490721c973fdbfc"
OFF_LINE = "moe_expert_calls=4 moe_naive_dispatch=1 moe_shared_forwards=12 moe_tokens=12"
OFF_DRAINED = (
    "moe_dropped_assignments=4 moe_expert_calls=4 moe_naive_dispatch=1 "
    "moe_shared_forwards=12 moe_tokens=12"
)


def _moe(**over) -> SparseMoE:
    base = dict(n_experts=4, moe_topk=2, rank=8, d_model=64, d_ffn=128, shared_rank=16)
    base.update(over)
    torch.manual_seed(0)
    return SparseMoE(tiny_cfg(**base))


def _x(n: int = 12) -> torch.Tensor:
    return torch.randn(n, 64, generator=torch.Generator().manual_seed(7))


def _forward(moe: SparseMoE, x: torch.Tensor | None = None) -> tuple[torch.Tensor, Counters]:
    c = Counters()
    out = moe(_x() if x is None else x, c)
    return out, c


# --------------------------------------------------------------------------
# gate a — mode "off" is today's model, bit for bit
# --------------------------------------------------------------------------


@pytest.mark.local_pin
def test_gate_a_off_arm_is_bit_identical_and_never_calls_the_policy(monkeypatch):
    """The knobs are INERT when the mode is off, and the digest proves it.

    `moe_topk_min=3 / moe_topk_max=8` are deliberately hostile values: on any
    other arm they would change the forward, on `off` they must not move one
    bit. The spy is the second half: if the forward took the adaptive branch
    the policy would be called even if the numbers happened to match.
    """

    def boom(*_a, **_k):
        raise AssertionError("the adaptive policy ran on the mode='off' arm")

    monkeypatch.setattr(experts_mod, "choose_k", boom)
    monkeypatch.setattr(experts_mod, "select_topk", boom)
    moe = _moe(
        moe_topk_mode="off",
        moe_topk_min=3,
        moe_topk_max=8,
        moe_entropy_lo=0.01,
        moe_entropy_hi=0.02,
    )
    out, c = _forward(moe)
    digest = hashlib.sha256(out.detach().contiguous().numpy().tobytes()).hexdigest()
    assert digest == OFF_DIGEST
    assert c.line() == OFF_LINE
    c.drain()
    assert c.line() == OFF_DRAINED
    # The default config (no new keys spelled out) is the same model.
    default = _moe()
    out2, c2 = _forward(default)
    assert torch.equal(out, out2)
    c2.drain()
    assert c2.line() == OFF_DRAINED


# --------------------------------------------------------------------------
# gate b — exactly k real slots, distinct, zero weight, no NaN
# --------------------------------------------------------------------------


def test_gate_b_exactly_k_real_slots_and_the_mask_sums():
    # A hand fixture with a known top-k order per row.
    logits = torch.tensor(
        [
            [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 5.0, 2.0, 4.0, 3.0, 0.0],
        ]
    )
    k = torch.tensor([1, 4, 2])
    index, active = select_topk(logits, k, 4)
    assert index.shape == (3, 4) and active.shape == (3, 4)
    assert active.sum(dim=-1).tolist() == k.tolist()
    # Real slots are distinct experts and the top-k prefix of the plain top-k.
    prefix = logits.topk(4, dim=-1).indices
    for row, width in enumerate(k.tolist()):
        real = index[row][active[row]]
        assert real.tolist() == prefix[row, :width].tolist()
        assert torch.unique(real).numel() == width
    # Unused slots carry weight 0 and no weight is NaN.
    probs = F.softmax(logits, dim=-1)
    weight = torch.gather(probs, 1, index) * active.to(probs.dtype)
    weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    assert torch.isfinite(weight).all()
    assert bool((weight[~active] == 0.0).all())
    assert torch.allclose(weight.sum(dim=-1), torch.ones(3))


def test_gate_b_mixed_widths_survive_the_padded_bank():
    """`_padded_dispatch` and `_naive_dispatch` see the SAME real rows.

    The padded arm sizes its bank from the per-token `pos`, so a padding slot
    that counted toward `pos` would hand a real row a slot past `cap` (an
    out-of-bounds `index_put`). This is the sentinel-group arithmetic, on CPU:
    a `k` mix of 1..4 with `moe_capacity=0.6` so the capacity also drops rows.
    """
    moe = _moe(moe_topk_mode="entropy", moe_topk_min=1, moe_topk_max=4, moe_capacity=0.6)
    x = _x()
    logits = moe.router(x)
    probs = F.softmax(logits, dim=-1)
    k = torch.tensor([1, 2, 3, 4] * 3)
    index, active = select_topk(logits, k, 4)
    weight = torch.gather(probs, 1, index) * active.to(probs.dtype)
    weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    order, keep, dropped = moe._permute(index, weight > 0)
    assert int((weight > 0).sum()) == 30  # 3 * (1+2+3+4)
    assert int(dropped) == 1 and int(keep.sum()) == 29  # capacity=0.6 drops one
    paced = moe._padded_dispatch(x, weight, index, order, keep)
    naive = moe._naive_dispatch(x, weight, index, order, keep, Counters())
    assert torch.isfinite(paced).all()
    assert torch.allclose(paced, naive, atol=1e-5), float((paced - naive).abs().max())


# --------------------------------------------------------------------------
# gate c — the entropy formula, as literals
# --------------------------------------------------------------------------


def test_gate_c_entropy_uniform_is_kmax_and_peaked_is_kmin():
    logits = torch.cat(
        [
            torch.zeros(4, 4),  # uniform: h = 1 -> k_max
            torch.tensor([[50.0, 0.0, 0.0, 0.0]] * 4),  # peaked: h ~ 0 -> k_min
        ]
    )
    k = choose_k(logits, mode="entropy", k_min=1, k_max=4, entropy_lo=0.2, entropy_hi=0.6)
    assert k.tolist() == [4, 4, 4, 4, 1, 1, 1, 1]
    # The middle of the ramp is an integer step, not a clamp: logits [2.5,0,0,0]
    # have h = 0.515, so t = (0.515-0.2)/0.4 and k = 1 + floor(3t) = 3.
    mid = choose_k(
        torch.tensor([[2.5, 0.0, 0.0, 0.0]]),
        mode="entropy",
        k_min=1,
        k_max=4,
        entropy_lo=0.2,
        entropy_hi=0.6,
    )
    assert mid.tolist() == [3]
    # Monotone: sharpening the router never WIDENS it (a grows -> h falls).
    ramp = torch.stack([torch.tensor([a, 0.0, 0.0, 0.0]) for a in torch.linspace(0, 20, 41)])
    widths = choose_k(ramp, mode="entropy", k_min=1, k_max=4, entropy_lo=0.2, entropy_hi=0.6)
    assert bool((widths[1:] <= widths[:-1]).all())


def test_gate_c_margin_is_the_logit_gap_in_the_opposite_direction():
    logits = torch.tensor(
        [
            [0.1, 0.0, 0.0, 0.0],  # gap 0.1 < lo -> the most width
            [0.7, 0.0, 0.0, 0.0],  # gap 0.7 > hi -> the least width
            [0.4, 0.0, 0.0, 0.0],  # mid window
        ]
    )
    k = choose_k(logits, mode="margin", k_min=1, k_max=4, entropy_lo=0.2, entropy_hi=0.6)
    assert k[0].item() == 4 and k[1].item() == 1
    assert k[2].item() == 3  # t = 0.5 -> 4 - floor(1.5) = 3
    off = choose_k(logits, mode="off", k_min=1, k_max=4, entropy_lo=0.2, entropy_hi=0.6)
    assert off.tolist() == [4, 4, 4]  # the control: constant k_max


# --------------------------------------------------------------------------
# gate d — the histogram counts every routed token exactly once
# --------------------------------------------------------------------------


def _crafted_router(moe: SparseMoE) -> torch.Tensor:
    """Logits `[a, 0, 0, 0]` with a in {0, 1, 2, 3}, three rows each."""
    a = torch.tensor([0.0, 1.0, 2.0, 3.0]).repeat_interleave(3)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0, 0] = 1.0
    x = torch.zeros(12, 64)
    x[:, 0] = a
    return x


def test_gate_d_the_histogram_sums_to_the_routed_tokens():
    moe = _moe(moe_topk_mode="entropy", moe_topk_min=1, moe_topk_max=4)
    x = _crafted_router(moe)
    logits = x[:, :1] * torch.tensor([1.0, 0.0, 0.0, 0.0])  # [12, 4] as the router sees them
    expected = choose_k(logits, mode="entropy", k_min=1, k_max=4, entropy_lo=0.2, entropy_hi=0.6)
    assert expected.max() > expected.min(), "the fixture must produce more than one width"
    out, c = _forward(moe, x)
    assert torch.isfinite(out).all()
    c.drain()
    hist = sum(c.get(f"moe_k{i}", 0) for i in range(1, 5))
    assert hist == 12, c.line()
    counts = torch.bincount(expected, minlength=5)
    for i in range(1, 5):
        assert c.get(f"moe_k{i}", 0) == int(counts[i])
    assert c["moe_k_mean_x100"] == (int(expected.sum()) * 100) // 12


# --------------------------------------------------------------------------
# gate e — no host sync in the adaptive forward
# --------------------------------------------------------------------------


def test_gate_e_the_adaptive_forward_touches_no_host(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("host sync inside the adaptive MoE forward (ADR-0018 rule 2)")

    moe = _moe(moe_topk_mode="entropy", moe_topk_min=1, moe_topk_max=4)
    x = _crafted_router(moe)
    # Run the PADDED arm on CPU (the arm the CUDA forward takes): the reference
    # loop (`_naive_dispatch`) iterates `unique(...).tolist()` BY DESIGN — it is
    # the slow correctness path, not the production one — so forcing the padded
    # arm is what makes this gate about the forward that will ship.
    monkeypatch.setattr(SparseMoE, "_padded_reason", lambda _self, _x: None)
    monkeypatch.setattr(torch.Tensor, "item", boom)
    monkeypatch.setattr(torch.Tensor, "tolist", boom)
    monkeypatch.setattr(torch.Tensor, "__bool__", boom)
    monkeypatch.setattr(torch.Tensor, "__int__", boom)
    monkeypatch.setattr(torch.Tensor, "__float__", boom)
    out, c = _forward(moe, x)
    # `torch.equal` returns a Python bool, so the finiteness check does not
    # need the patched `Tensor.__bool__`.
    assert torch.equal(torch.isfinite(out), torch.ones_like(out, dtype=torch.bool))
    # The device lane holds the width counters; the host line has none of them
    # until `drain()` (the log cadence), which is the point of the gate.
    assert "moe_k_mean_x100" in c._device and "moe_k_mean_x100" not in c.line()


# --------------------------------------------------------------------------
# gate f — the static refusals are LOUD
# --------------------------------------------------------------------------


def test_gate_f_the_misconfigurations_are_loud():
    with pytest.raises(LoudError, match="moe_topk_mode"):
        validate(tiny_cfg(moe_topk_mode="auto"))
    with pytest.raises(LoudError, match="moe_topk_min"):
        validate(tiny_cfg(moe_topk_mode="entropy", moe_topk_min=4, moe_topk_max=2))
    with pytest.raises(LoudError, match="moe_topk_max"):
        validate(tiny_cfg(moe_topk_mode="entropy", moe_topk_max=9))
    with pytest.raises(LoudError, match="moe_topk=0"):
        validate(tiny_cfg(moe_topk_mode="entropy", moe_topk=0))
    with pytest.raises(LoudError, match="n_experts"):
        validate(tiny_cfg(moe_topk_mode="entropy", moe_topk_max=8))
    with pytest.raises(LoudError, match="moe_entropy_lo"):
        validate(tiny_cfg(moe_topk_mode="entropy", moe_entropy_lo=0.6, moe_entropy_hi=0.6))
    # The pure policy refuses the same inputs, and the module refuses a
    # hand-built config without the loader's validate (the runtime seam).
    logits = torch.zeros(2, 4)
    with pytest.raises(LoudError, match="moe_topk_mode"):
        choose_k(logits, mode="auto", k_min=1, k_max=4, entropy_lo=0.2, entropy_hi=0.6)
    with pytest.raises(LoudError, match="moe_topk_min"):
        choose_k(logits, mode="entropy", k_min=3, k_max=2, entropy_lo=0.2, entropy_hi=0.6)
    with pytest.raises(LoudError, match="moe_topk_max"):
        choose_k(logits, mode="entropy", k_min=1, k_max=9, entropy_lo=0.2, entropy_hi=0.6)
    with pytest.raises(LoudError, match="moe_entropy_lo"):
        choose_k(logits, mode="entropy", k_min=1, k_max=4, entropy_lo=0.6, entropy_hi=0.2)
    with pytest.raises(LoudError, match="moe_topk=0"):
        _moe(moe_topk_mode="entropy", moe_topk=0)
    with pytest.raises(LoudError, match="moe_topk_max"):
        _moe(moe_topk_mode="entropy", moe_topk_max=9)


# --------------------------------------------------------------------------
# config: defaults are OFF, the knobs are real fields, presets stay off
# --------------------------------------------------------------------------


def test_config_defaults_leave_every_run_unchanged():
    cfg = ModelConfig()
    assert cfg.moe_topk_mode == "off"
    assert (cfg.moe_topk_min, cfg.moe_topk_max) == (1, 4)
    assert (cfg.moe_entropy_lo, cfg.moe_entropy_hi) == (0.2, 0.6)
    assert load_preset("small").moe_topk_mode == "off"
    # A snapshot written BEFORE the fields existed takes the defaults: the
    # strict loader fills missing keys from the dataclass (the "missing key
    # takes the class default" rule), so no old run changes.
    from dormouse.config import RunConfig

    run = RunConfig.from_snapshot("source = 'old'\n[model]\nd_model = 64\n")
    assert run.model.moe_topk_mode == "off"
    # And a preset can turn the policy on, with the width it names.
    turned_on = parse_preset(
        "d_model=64\nn_heads=4\nhead_dim=16\nn_experts=4\nmoe_topk=2\n"
        "moe_topk_mode='entropy'\nmoe_topk_min=1\nmoe_topk_max=3\n"
    )
    validate(turned_on)
    assert (turned_on.moe_topk_mode, turned_on.moe_topk_max) == ("entropy", 3)


def test_the_policy_module_is_the_one_place_the_formula_lives():
    """`adaptive_topk` is pure: same logits and knobs, same k, no module state."""
    logits = torch.randn(64, 6, generator=torch.Generator().manual_seed(11))
    args = dict(mode="entropy", k_min=1, k_max=4, entropy_lo=0.2, entropy_hi=0.6)
    a = choose_k(logits, **args)
    b = choose_k(logits.clone(), **args)
    assert torch.equal(a, b)
    assert a.shape == (64,) and a.dtype == torch.long
    uniform = adaptive_topk._normalized_entropy(torch.zeros(2, 6))
    assert torch.allclose(uniform, torch.ones(2), atol=1e-6)
    # One-expert banks have zero entropy by definition (and no division by 0).
    assert choose_k(
        torch.zeros(2, 1), mode="entropy", k_min=1, k_max=1, entropy_lo=0.2, entropy_hi=0.6
    ).tolist() == [1, 1]


# --------------------------------------------------------------------------
# the bench driver's pure logic (`scripts/bench_topk.py`, never run here)
# --------------------------------------------------------------------------


def test_bench_windows_are_valid_and_the_flop_term_scales_with_k():
    spec = importlib.util.spec_from_file_location(
        "bench_topk", Path(__file__).resolve().parents[1] / "scripts" / "bench_topk.py"
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    candidates = bench.windows()
    assert all(lo < hi for lo, hi in candidates)
    # The saturated ends are what a target of k_min / k_max selects.
    assert (-1.0, 0.0) in candidates and (1.0, 2.0) in candidates
    cfg = tiny_cfg()
    assert bench.expert_flops_per_byte(cfg, 2.0) == 2 * bench.expert_flops_per_byte(cfg, 1.0)
