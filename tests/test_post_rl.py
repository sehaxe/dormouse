"""Gates for the RL stepper (`post/rl.py`): the MiMo reward, the advantages, the step.

Every expected number here is computed by hand from the source the module names,
never read back off the implementation:

- **reward** — MiMo-V2.6 (2610.11959) Eq. 2's `R_test · S_sol · S_beh` and Eq. 4's
  gated, group-relative length term, on a 4-item table in BOTH gate states (the
  boundary case is `pass_rate == gate`, which must leave the term INACTIVE and
  COUNTED — the notes' §2b failure mode is a dead term that reads as a mechanism).
- **advantages** — GRPO's `(r − group_mean)/group_std` on a 3-reward group
  (`std = sqrt(1/6)`), a full-consensus group (exactly 0, source `"consensus"`),
  and the CRR arm against `train_steps.crr_advantage` itself.
- **the aggregation** — prompt-mean weights `1/G` and the loss `−19/6` on a
  hand table, with `token-mean` (the notes' §4.2 control) as the second row.
- **the gather** — a 2-byte response over FIXED logits, where `logp[j]` must be
  the log-prob of byte `j` under the prediction at `j−1`: the closed forms are
  `−log 4`, `−log(e+3)`, `1 − log(e+3)`, and the `logits[j]` misreading differs
  by exactly 1.
- **the step** — end to end through a fixed-logits stub, so the loss rl_step
  returns is the same hand table; plus the skip counters, the window refusal, the
  no-grad refusal, and the non-causal-MoE counter.
- **sampling** — the same generator seed gives the same bytes, sampling moves no
  weight and builds no grad.

The CUDA gate at the end is the only one that needs a card, and it self-skips
when the card is busy (AGENTS §1.5: one GPU thing at a time).
"""

from __future__ import annotations

import math
import subprocess
from types import SimpleNamespace

import pytest
import torch
from conftest import tiny_model

from dormouse.instrument import Counters, LoudError
from dormouse.post.rl import (
    RLAdvantage,
    RLConfig,
    Sample,
    group_rewards,
    group_stats,
    pg_loss,
    response_logp,
    reward,
    rl_advantages,
    rl_step,
    sample_weights,
)
from dormouse.post.rollouts import Rollout, rollout
from dormouse.post.tasks import build_tasks
from dormouse.post.train_steps import crr_advantage
from dormouse.post.verifiers import Verdict, verify
from dormouse.train.optim import MixOptimizer

#: The hidden reference `S_sol` measures a response against: 10 B, so a 10-byte
#: response is full quality, a 20-byte one is half, a 40-byte one is a quarter.
REF = "0123456789"

#: The fixed-logits table the gather and the stub-step fixtures share.
#:   row0 [0,0,0,0]  → every byte −log 4
#:   row1 [0,0,0,0]  → every byte −log 4
#:   row2 [1,0,0,0]  → byte 0: 1 − log(e+3), byte 1: −log(e+3)
#:   row3 [0,0,0,0]  → every byte −log 4
_ROWS = torch.tensor(
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
)
_LOG4 = math.log(4.0)
_LOG_E3 = math.log(math.e + 3.0)
#: A 2-byte response `[0, 1]` at positions 2,3 of a 4-byte window with prompt_len 2:
#: logp[2] = row1[0] = −log 4, logp[3] = row2[1] = −log(e+3).
_RESPONSE_A = -_LOG4 - _LOG_E3
#: The response `[1, 0]`: logp[2] = row1[1] = −log 4, logp[3] = row2[0] = 1 − log(e+3).
_RESPONSE_B = -_LOG4 + 1.0 - _LOG_E3


def _v(passed: bool, reason: str = "all tests passed") -> Verdict:
    return Verdict(passed, reason, 0.0)


def _cfg(**over) -> RLConfig:
    """The reward table's config: X=0.4, B=0.5, A=0.5, δ=0, s=0.5, γ=1."""
    base = dict(length_penalty=0.4, quantile=0.5, gate=0.5, delta=0.0, s=0.5, gamma=1.0)
    base.update(over)
    return RLConfig(**base)


def _add_task() -> dict:
    return {
        "cls": "code",
        "tid": "add",
        "prompt": "",
        "expected": "5",
        "tests": (("add(2, 3)", "5"),),
        "match": "exact",
        "budget": 128,
    }


class _FixedLogits(torch.nn.Module):
    """A model whose forward returns FIXED logits — the hand table, end to end.

    `rows` is `[t, vocab]`; `w` is a zero parameter so the loss still carries a
    graph (the `requires_grad` contract `rl_step` checks), and `seen` records the
    byte windows the step actually forwarded.
    """

    def __init__(self, rows: torch.Tensor, max_seq_len: int) -> None:
        super().__init__()
        self.rows = rows
        self.w = torch.nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(max_seq_len=max_seq_len)
        self.seen: list[list[int]] = []

    def forward(self, input_ids, targets=None, counters=None, **_kw):
        counters = Counters() if counters is None else counters
        self.seen.append(input_ids.reshape(-1).tolist())
        return self.rows + 0.0 * self.w, None, counters


# ---------------------------------------------------------------------------
# the reward: MiMo Eq. 2 · Eq. 4, both gate states
# ---------------------------------------------------------------------------


def test_reward_table_with_the_gate_closed():
    cfg = _cfg()
    verdicts = [_v(True), _v(True), _v(False, "test 'add(2, 3)' failed"), _v(False, "failed")]
    lengths = [10, 20, 30, 40]
    counters = Counters()
    terms = group_rewards(verdicts, [REF] * 4, lengths, cfg, counters=counters)
    # pass_rate = 0.5 and the gate is `> 0.5`: the term is INACTIVE — and counted.
    assert [t.total for t in terms] == [1.0, 0.5, 0.0, 0.0]
    assert [t.length_penalty for t in terms] == [0.0, 0.0, 0.0, 0.0]
    assert [t.len_penalty_active for t in terms] == [False, False, False, False]
    assert [t.r_test for t in terms] == [1.0, 1.0, 0.0, 0.0]
    assert [t.s_sol for t in terms] == [1.0, 0.5, 10 / 30, 0.25]
    assert [t.s_beh for t in terms] == [1.0, 1.0, 1.0, 1.0]
    assert counters["rl_len_penalty_gated"] == 4
    assert "rl_len_penalty_active" not in counters


def test_reward_table_with_the_gate_open():
    cfg = _cfg()
    verdicts = [_v(True), _v(True), _v(True), _v(False, "failed")]
    lengths = [10, 20, 30, 40]
    counters = Counters()
    stats = group_stats(verdicts, lengths, cfg)
    assert stats.n == 4 and stats.pass_rate == 0.75
    assert stats.quantile_len == 20.0  # median of the SUCCESSFUL lengths [10, 20, 30]
    assert stats.gate_open
    terms = group_rewards(verdicts, [REF] * 4, lengths, cfg, counters=counters)
    # ℓ*_q = 20, δ = 0, s = 0.5, γ = 1 → clip((ℓ/20 − 1)/0.5, 0, 1):
    #   10 → −1 → 0 ; 20 → 0 → 0 ; 30 → 1 → 0.4·1¹ ; the failure gets no term.
    assert [t.length_penalty for t in terms] == [0.0, 0.0, 0.4, 0.0]
    assert [t.total for t in terms] == [1.0, 0.5, 10 / 30 - 0.4, 0.0]
    assert [t.len_penalty_active for t in terms] == [True, True, True, False]
    assert counters["rl_len_penalty_active"] == 3
    assert "rl_len_penalty_gated" not in counters


def test_reward_length_ramp_moves_with_delta_and_s():
    lengths = [60, 120, 240]
    verdicts = [_v(True)] * 3
    # δ = 0, s = 0.5: the knee is at ℓ*_q and 240 B is (2 − 1)/0.5 = 2 → clip 1.
    plain = group_rewards(verdicts, [REF] * 3, lengths, _cfg())
    assert [t.length_penalty for t in plain] == [0.0, 0.0, 0.4]
    # δ = 0.5, s = 1.5: the free band is 1.5·ℓ*_q = 180 B, and 240 B is (2 − 1.5)/1 = 0.5.
    shifted = group_rewards(verdicts, [REF] * 3, lengths, _cfg(delta=0.5, s=1.5))
    assert [t.length_penalty for t in shifted] == [0.0, 0.0, 0.2]


def test_reward_needs_the_group_stats_when_the_penalty_is_on():
    with pytest.raises(LoudError, match="group-relative"):
        reward(_v(True), REF, _cfg(), length=10)


def test_reward_thrift_and_the_behaviour_flags():
    cfg = _cfg(length_penalty=0.0)
    task = _add_task()
    # A verifier REFUSAL is the deterministic behaviour flag (`verifiers.py:211`).
    refused = verify(task, "import socket\ndef add(a, b):\n    return a + b")
    assert refused.reason.startswith("refused:")
    t = reward(refused, REF, cfg, length=40)
    assert (t.r_test, t.s_sol, t.s_beh, t.total) == (0.0, 0.25, 0.0, 0.0)
    # A passing response at or under the reference's own byte cost is full quality.
    ok = verify(task, "def add(a, b):\n    return a + b")
    t = reward(ok, REF, cfg, length=5)
    assert (t.r_test, t.s_sol, t.s_beh, t.total) == (1.0, 1.0, 1.0, 1.0)
    # `quality="none"` turns the thrift proxy off (S_sol = 1) and nothing else.
    t = reward(ok, REF, _cfg(quality="none", length_penalty=0.0), length=40)
    assert (t.s_sol, t.s_beh, t.total) == (1.0, 1.0, 1.0)
    # An empty response is not a solution, whatever the reference cost.
    t = reward(_v(False, "no final answer found in the response"), REF, cfg, length=0)
    assert (t.s_sol, t.s_beh, t.total) == (0.0, 0.0, 0.0)


def test_group_stats_quantile_uses_the_successful_lengths_only():
    stats = group_stats([_v(True), _v(True), _v(True), _v(False)], [10, 20, 30, 10_000], _cfg())
    assert (stats.n, stats.pass_rate, stats.quantile_len, stats.gate_open) == (4, 0.75, 20.0, True)
    # No passes: no quantile, and the gate is shut whatever the pass rate says.
    stats = group_stats([_v(False)] * 2, [10, 20], _cfg(gate=0.0))
    assert (stats.quantile_len, stats.pass_rate, stats.gate_open) == (0.0, 0.0, False)


def test_rl_config_refuses_a_ramp_that_divides_by_zero():
    with pytest.raises(LoudError, match="divides by zero"):
        RLConfig(length_penalty=0.4, delta=0.5, s=0.5)


# ---------------------------------------------------------------------------
# the advantages
# ---------------------------------------------------------------------------


def test_group_advantages_match_a_hand_table():
    adv = rl_advantages([1.0, 0.5, 0.0], [0, 0, 0], "group")
    # mean 0.5; population std = sqrt(((0.5)² + 0² + (0.5)²)/3) = sqrt(1/6).
    std = math.sqrt(1.0 / 6.0)
    assert [a.value for a in adv] == pytest.approx([0.5 / std, 0.0, -0.5 / std])
    assert [a.source for a in adv] == ["group"] * 3
    assert [a.index for a in adv] == [0, 1, 2]
    # A full-consensus group carries no signal: exactly 0, and the source says why.
    adv = rl_advantages([1.0, 1.0], [0, 0], "group")
    assert [a.value for a in adv] == [0.0, 0.0]
    assert [a.source for a in adv] == ["consensus"] * 2


def test_advantages_align_by_index_across_interleaved_groups():
    adv = rl_advantages([1.0, 0.0, 1.0, 0.0], [7, 7, 9, 9], "group")
    assert [a.index for a in adv] == [0, 1, 2, 3]
    assert [a.group for a in adv] == [7, 7, 9, 9]
    assert [a.value for a in adv] == [1.0, -1.0, 1.0, -1.0]  # two groups, std = 0.5 each


def test_crr_mode_delegates_to_train_steps_and_reindexes():
    def r(trace, passed):
        return Rollout(trace=list(trace), n_bytes=0, verdict=Verdict(passed, "x", 0.0))

    rollouts = [
        r(["aa", "bb", "cc"], True),  # 0 pass
        r(["aa", "bb", "dd"], False),  # 1 fail, 3 strokes
        r([], False),  # 2 empty trajectory → 0
        r(["aa", "bb", "cc", "ee"], False),  # 3 fail, 4 strokes — the best failed
    ]
    task = build_tasks("math", 1, 1)[0]
    samples = [Sample.from_rollout(task, ro, 0) for ro in rollouts]
    # The shaped reward's magnitude is NOT what the CRR arm reads: the fork-replay
    # shape is defined on the pass/fail verdict, so the list below is only alignment.
    adv = rl_advantages([1.0, 0.9, 0.8, 0.7], [0, 0, 0, 0], "crr", samples=samples)
    ref = crr_advantage(rollouts)
    assert [a.value for a in adv] == [x.value for x in ref] == [1.0, -1.0, 0.0, -1.0]
    assert [a.source for a in adv] == [x.source for x in ref]
    assert [a.index for a in adv] == [0, 1, 2, 3]
    assert [a.group for a in adv] == [0, 0, 0, 0]


def test_crr_mode_needs_the_rollouts():
    with pytest.raises(LoudError, match="samples"):
        rl_advantages([1.0], [0], "crr")


# ---------------------------------------------------------------------------
# the aggregation and the loss
# ---------------------------------------------------------------------------


def test_sample_weights_prompt_mean_and_token_mean():
    lengths, groups = [2, 4, 6, 3, 9], [0, 0, 0, 1, 1]
    assert sample_weights(lengths, groups, RLConfig()) == [1 / 3, 1 / 3, 1 / 3, 1 / 2, 1 / 2]
    assert sample_weights(lengths, groups, RLConfig(agg="token-mean")) == [
        2 / 12,
        4 / 12,
        6 / 12,
        3 / 12,
        9 / 12,
    ]


def test_prompt_mean_loss_matches_a_hand_table():
    lengths, groups = [2, 4, 6, 3, 9], [0, 0, 0, 1, 1]
    logp = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0])
    advantages = [1.0, 0.5, 0.0, -1.0, -2.0]
    weights = sample_weights(lengths, groups, RLConfig())
    # Σ = (1/3)(1)(−1) + (1/3)(0.5)(−2) + 0 + (1/2)(−1)(−4) + (1/2)(−2)(−5)
    #   = −1/3 − 1/3 + 2 + 5 = 19/3 ;  N_prompts = 2 → loss = −19/6.
    assert float(pg_loss(logp, weights, advantages, groups)) == pytest.approx(-19.0 / 6.0)
    # token-mean (the control arm): Σ = −1/6 − 1/3 + 0 + 1 + 7.5 = 8 → loss = −4.
    weights = sample_weights(lengths, groups, RLConfig(agg="token-mean"))
    assert float(pg_loss(logp, weights, advantages, groups)) == pytest.approx(-4.0)


# ---------------------------------------------------------------------------
# the gather: logp[j] is the log-prob of byte j at position j−1
# ---------------------------------------------------------------------------


def test_response_logp_gather_matches_a_hand_fixture():
    logits = _ROWS  # [t, vocab] = the model's own [b*t, vocab] at b = 1
    got = float(response_logp(logits, torch.tensor([[3, 3, 0, 1]]), 2)[0])
    assert got == pytest.approx(_RESPONSE_A, rel=1e-6)
    # The `logits[j]` misreading (byte j under the prediction at j) differs by
    # exactly 1: row2[0] = 1 − log(e+3) instead of row2[1] = −log(e+3).
    assert abs(got - (-_LOG4 + 1.0 - _LOG_E3)) > 0.5


def test_response_logp_is_per_row_and_scores_only_the_response():
    logits = _ROWS.repeat(2, 1)  # [b*t, vocab] at b = 2
    # Row A: prompt 2, response [0, 1]. Row B: prompt 3, response [1].
    both = response_logp(logits, torch.tensor([[3, 3, 0, 1], [3, 3, 3, 1]]), [2, 3])
    assert float(both[0]) == pytest.approx(_RESPONSE_A, rel=1e-6)
    assert float(both[1]) == pytest.approx(-_LOG_E3, rel=1e-6)
    # The prompt's own bytes are context: with fixed logits they cannot move the sum.
    other_prompt = response_logp(_ROWS, torch.tensor([[2, 2, 0, 1]]), 2)
    assert float(other_prompt[0]) == pytest.approx(_RESPONSE_A, rel=1e-6)
    # The response's bytes do move it: [1, 0] is a different 2-byte response.
    other_response = response_logp(_ROWS, torch.tensor([[3, 3, 1, 0]]), 2)
    assert float(other_response[0]) == pytest.approx(_RESPONSE_B, rel=1e-6)


# ---------------------------------------------------------------------------
# the step
# ---------------------------------------------------------------------------


def test_rl_step_matches_a_hand_computed_loss_on_fixed_logits():
    model = _FixedLogits(_ROWS.clone(), max_seq_len=4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    samples = [
        Sample(prompt=bytes([3, 3]), response=bytes([0, 1]), group=0),  # → _RESPONSE_A
        Sample(prompt=bytes([2, 2]), response=bytes([1, 0]), group=0),  # → _RESPONSE_B
    ]
    advantages = [RLAdvantage(0, 1.0, 0, "group"), RLAdvantage(1, 0.5, 0, "group")]
    loss, counters = rl_step(model, opt, samples, advantages, RLConfig())
    # G = 2 → w = 1/2 each; N_prompts = 1:
    #   loss = −[(1/2)(1.0)(−log4 − log(e+3)) + (1/2)(0.5)(−log4 + 1 − log(e+3))]
    expect = -((1 / 2) * 1.0 * _RESPONSE_A + (1 / 2) * 0.5 * _RESPONSE_B)
    assert float(loss.detach()) == pytest.approx(expect, rel=1e-6)
    assert counters["rl_steps"] == 1 and counters["rl_samples"] == 2
    # Teacher-forced over prompt + response, one sample per forward.
    assert model.seen == [[3, 3, 0, 1], [2, 2, 1, 0]]
    # Linear in the advantage: flipping the signs flips the loss.
    flipped = [RLAdvantage(0, -1.0, 0, "group"), RLAdvantage(1, -0.5, 0, "group")]
    other, _ = rl_step(model, opt, samples, flipped, RLConfig())
    assert float(other.detach()) == pytest.approx(-float(loss.detach()), rel=1e-6)


def test_rl_step_skips_zero_advantage_and_empty_responses_with_counters():
    model = _FixedLogits(_ROWS.clone(), max_seq_len=4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    samples = [
        Sample(prompt=bytes([3, 3]), response=bytes([0, 1]), group=0),
        Sample(prompt=bytes([3, 3]), response=bytes([1, 1]), group=0),  # zero advantage
        Sample(prompt=bytes([3, 3]), response=b"", group=0),  # nothing was sampled
    ]
    advantages = [
        RLAdvantage(0, 1.0, 0, "group"),
        RLAdvantage(1, 0.0, 0, "consensus"),
        RLAdvantage(2, 1.0, 0, "group"),
    ]
    loss, counters = rl_step(model, opt, samples, advantages, RLConfig())
    assert counters["rl_steps"] == 1 and counters["rl_samples"] == 1
    assert counters["rl_skipped_no_advantage"] == 1
    assert counters["rl_skipped_empty_response"] == 1
    assert model.seen == [[3, 3, 0, 1]]  # no forward was paid for the skipped two
    # The weight is the FULL group's 1/G = 1/3, not 1/(surviving samples).
    assert float(loss.detach()) == pytest.approx(-(1 / 3) * _RESPONSE_A, rel=1e-6)


def test_rl_step_with_nothing_to_train_returns_a_grad_carrying_zero():
    model = _FixedLogits(_ROWS.clone(), max_seq_len=4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    samples = [Sample(prompt=bytes([3, 3]), response=b"", group=0)]
    loss, counters = rl_step(model, opt, samples, [RLAdvantage(0, 1.0, 0, "group")], RLConfig())
    assert float(loss.detach()) == 0.0 and bool(loss.requires_grad)
    assert counters["rl_steps"] == 1 and counters.get("rl_samples", 0) == 0
    assert counters["rl_skipped_empty_response"] == 1
    assert model.seen == []


def test_rl_step_refuses_a_window_past_max_seq_len():
    model = _FixedLogits(_ROWS.clone(), max_seq_len=3)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    samples = [Sample(prompt=bytes([3, 3]), response=bytes([0, 1]), group=0)]
    with pytest.raises(LoudError, match="window"):
        rl_step(model, opt, samples, [RLAdvantage(0, 1.0, 0, "group")], RLConfig())


def test_rl_step_backward_reaches_a_parameter_and_no_grad_is_refused():
    model = tiny_model()
    opt = MixOptimizer(model, lr=1e-4, wd=0.0, opt="adamw")
    samples = [Sample(prompt=b"solve\n", response=b"answer: 4", group=0)]
    advantages = [RLAdvantage(0, 1.0, 0, "group")]
    loss, _ = rl_step(model, opt, samples, advantages, RLConfig())
    assert bool(loss.requires_grad)
    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters())
    with pytest.raises(LoudError, match="no_grad"):
        with torch.no_grad():
            rl_step(model, opt, samples, advantages, RLConfig())


def test_rl_step_counts_the_non_causal_moe():
    # The tiny preset ships `moe_topk=2` at `moe_capacity=1.0`: a capacity-dropping
    # MoE is not prefix-causal, so the teacher-forced log-probs are not the
    # sampler's — COUNTED, never silent (`decode.py:46-49`'s rule).
    model = tiny_model()
    opt = MixOptimizer(model, lr=1e-4, wd=0.0, opt="adamw")
    samples = [Sample(prompt=b"solve\n", response=b"answer: 4", group=0)]
    _, counters = rl_step(model, opt, samples, [RLAdvantage(0, 1.0, 0, "group")], RLConfig())
    assert counters["rl_non_causal_logp"] == 1


# ---------------------------------------------------------------------------
# the rollout bridge, determinism, and the no-grad contract
# ---------------------------------------------------------------------------


def test_sample_from_rollout_recovers_the_response_bytes():
    task = build_tasks("math", 2, 1)[0]
    model = tiny_model()
    budget = len(task.prompt.encode("utf-8")) + 60
    ro = rollout(model, task, budget, greedy=True)
    sample = Sample.from_rollout(task, ro, 3)
    assert sample.prompt == task.prompt.encode("utf-8")
    assert len(ro.trace) == 20
    assert sample.response.hex() and [f"{b:02x}" for b in sample.response] == ro.trace
    assert sample.group == 3 and sample.rollout is ro


def test_seeded_sampling_is_deterministic():
    task = build_tasks("math", 1, 1)[0]
    model = tiny_model()
    budget = len(task.prompt.encode("utf-8")) + 60
    first = rollout(model, task, budget, generator=torch.Generator().manual_seed(3))
    second = rollout(model, task, budget, generator=torch.Generator().manual_seed(3))
    assert first.trace == second.trace
    assert first.n_bytes == second.n_bytes and len(first.trace) > 1


def test_sampling_touches_no_weight_and_builds_no_grad():
    task = build_tasks("math", 1, 1)[0]
    model = tiny_model()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    budget = len(task.prompt.encode("utf-8")) + 60
    sampled = rollout(model, task, budget, generator=torch.Generator().manual_seed(5))
    assert all(p.grad is None for p in model.parameters())
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), f"{name} moved during sampling"
    # The decode path is already no_grad inside; an outer no_grad changes nothing.
    with torch.no_grad():
        again = rollout(model, task, budget, generator=torch.Generator().manual_seed(5))
    assert again.trace == sampled.trace


# ---------------------------------------------------------------------------
# the CUDA gate (self-skips when the card is busy)
# ---------------------------------------------------------------------------


def _gpu_busy() -> str | None:
    """AGENTS §1.5: one GPU thing at a time — the brief's own precondition."""
    try:
        live = subprocess.run(
            ["pgrep", "-f", "dormouse trai[n]"], capture_output=True, text=True, timeout=15
        ).stdout.strip()
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001 - a gate's precondition, not a run's
        return f"cannot check the card ({e})"
    if live:
        return f"a train run holds the card (pids {live})"
    mib = int(used.split()[0]) if used else -1
    if mib >= 1024:
        return f"the card holds {mib} MiB"
    return None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_rl_step_cuda_matches_the_cpu_fp32_twin():
    """One real rl_step on the card, against the same weights on the CPU.

    The bound is a WIRING bound, not a numerics certificate: the KDA chunked
    kernel sits ~1.3e-2 of the output RMS off the exact recurrence at this shape
    (`decode.py:32-36`), so the two substrates cannot agree bit-for-bit. A wrong
    gather, a shifted position or a missing mask is O(1) off, and 10 % catches
    every one of those.
    """
    busy = _gpu_busy()
    if busy is not None:
        pytest.skip(f"one GPU thing at a time: {busy}")
    cpu = tiny_model()
    twin = {k: v.detach().clone() for k, v in cpu.state_dict().items()}
    gpu = tiny_model()
    gpu.load_state_dict(twin)
    gpu.to("cuda")
    samples = [
        Sample(prompt=b"1 + 1 = ", response=b"2", group=0),
        Sample(prompt=b"2 + 2 = ", response=b"answer: 4", group=0),
    ]
    advantages = [RLAdvantage(0, 1.0, 0, "group"), RLAdvantage(1, -1.0, 0, "group")]
    cfg = RLConfig()
    loss_cpu, counters_cpu = rl_step(
        cpu, MixOptimizer(cpu, lr=1e-4, wd=0.0, opt="adamw"), samples, advantages, cfg
    )
    loss_gpu, counters_gpu = rl_step(
        gpu, MixOptimizer(gpu, lr=1e-4, wd=0.0, opt="adamw"), samples, advantages, cfg
    )
    assert counters_cpu["rl_samples"] == counters_gpu["rl_samples"] == 2
    rel = abs(float(loss_gpu) - float(loss_cpu)) / abs(float(loss_cpu))
    assert rel <= 1e-1, f"cpu={float(loss_cpu):.6f} cuda={float(loss_gpu):.6f} rel={rel:.3e}"
    # The step really trains on the card: the graph reaches a parameter.
    assert bool(loss_gpu.requires_grad)
    loss_gpu.backward()
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0 for p in gpu.parameters()
    ), "no parameter received a gradient from the CUDA policy-gradient step"
