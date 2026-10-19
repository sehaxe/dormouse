"""Self-play pretraining with zero data (arXiv:2609.30063) — the CPU core's gates.

Four things get a gate here, one per module:

- **`utm`** — Appendix E's machine, pinned against gold vectors computed by hand
  from the appendix's own semantics (the appendix's `+++[>+.<-]F` example and
  three of Table 1's discovered program families), plus the totality property
  §2.1 promises: every instruction string is executable, so 1000 random byte
  strings must all run and none may raise.
- **`reward`** — Eq. 2, pinned against its definition three ways: `P_e` against
  the AdamW state by hand, `u = P_e ⊙ δθ_e` elementwise, and the reward against
  a central finite difference of the loss along `u`. Forward mode and the
  reverse-mode fallback must agree.
- **`pool`** — Appendix G's construction, Eq. 4's stored probabilities and
  Eq. 5's normalized weights, plus the `g_0` normalisation as a closed form.
- **`loop`** — §2.2's round end to end on a ~100k-parameter model, mutation-only.

Everything is CPU and tiny; nothing here needs a GPU.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import random

import pytest
import torch
from conftest import tiny_cfg
from torch.func import functional_call

from dormouse.config import ModelConfig
from dormouse.instrument import Counters
from dormouse.model import DormouseModel, param_count
from dormouse.selfplay import loop as sp_loop
from dormouse.selfplay import pool as sp_pool
from dormouse.selfplay import reward as sp_reward
from dormouse.selfplay import utm

# ---------------------------------------------------------------------------
# Appendix E — the machine, against gold vectors computed by hand
# ---------------------------------------------------------------------------
#
# Every expectation below is derived from Appendix E's semantics rather than
# from this implementation: tape cells modulo m=256, a circular tape, unmatched
# brackets as no-ops, `.` emitting the current cell, `F` halting, and the
# Table 4 expansions. The appendix's own worked example is `appendix_example`.


def _out(program: bytes, tape: bytes = b"", steps: int = 200, out_len: int | None = None):
    return utm.run(program, tape, steps, out_len=out_len)


def test_appendix_e_worked_example_is_the_paper_s_own_output():
    """Appendix E: `+++[>+.<-]F` "emits y = (1,2,3,0,…,0)"."""
    got = _out(b"+++[>+.<-]F", steps=100, out_len=8)
    assert list(got) == [1, 2, 3, 0, 0, 0, 0, 0]


@pytest.mark.parametrize(
    ("program", "tape", "expect"),
    [
        (b"+.", b"", [1]),  # increment then emit
        (b"-.", b"", [255]),  # decrement wraps modulo m = 256
        (b"<+.", b"", [1]),  # `<` at cell 0 wraps to the last cell
        (b",.", b"\x2a", [42]),  # `,` reads the input tape
        (b",,.", b"\x01\x02", [2]),  # and a second read advances it
        (b"F++.", b"", []),  # F halts before the `++` and the `.`
        (b"X.", b"", [16]),  # Table 4: X = [-]++++++++++++++++
        (b"+++Z.", b"", [0]),  # Table 4: Z = [-] clears the cell
        (b"+++R>.", b"", [3]),  # Table 4: R adds x into the right neighbour
        (b"++L>.", b"", [6]),  # Table 4: L adds 3x into the right neighbour
        (b"+++>+N<.", b"", [2]),  # Table 4: N subtracts x from the left
        (b"+++C>.", b"", [3]),  # Table 4: C adds x into the two right cells
        (b"+>++>+++G+.", b"", [1]),  # Table 4: G scans right to a zero cell
        (b"+>+Z>+H.", b"", [0]),  # Table 4: H scans left to a zero cell
        (b"+++W>.", b"", [1]),  # Table 4: W increments right, clears current
        (b"+++>+++<V", b"", [3, 3]),  # Table 4: V prints until a zero cell
    ],
)
def test_every_primitive_and_macro_against_its_definition(program, tape, expect):
    assert list(_out(program, tape)) == expect


def test_table_1_discovered_families_reproduce():
    """Paper Table 1's program families, under Appendix E's semantics.

    Arithmetic `+[.++]` emits 1,3,5,…; geometric `+[.L>]` emits 1,3,9,27,81
    because `L` triples into the next cell each pass; Fibonacci `,[[.C>.C>]`
    needs the random tape's first byte to be 1 (the `,` seeds the recurrence)
    and emits 1,1,2,3,5,8. The second `[` is unmatched and therefore a no-op —
    which is exactly Appendix E's rule, and the programs still work.
    """
    assert list(_out(b"+[.++]", out_len=5)) == [1, 3, 5, 7, 9]
    assert list(_out(b"+[.L>]", steps=1000, out_len=5)) == [1, 3, 9, 27, 81]
    assert list(_out(b",[[.C>.C>]", b"\x01", steps=1000, out_len=6)) == [1, 1, 2, 3, 5, 8]


def test_unmatched_brackets_are_no_ops_and_unknown_bytes_are_no_ops():
    """Appendix E: unmatched brackets are no-ops; totality extends past `A`."""
    assert list(_out(b"]+.")) == [1]  # a lone `]` falls through
    assert list(_out(b"[+.")) == [1]  # a lone `[` falls through into its body
    assert list(_out(b"+[")) == []  # and a lone `[` with nothing to emit
    assert list(_out(b"1x2[+.")) == [1]  # bytes outside the alphabet are no-ops


def test_execution_always_terminates_at_the_step_budget():
    """`+[]` is an infinite loop; Appendix E's step budget is what ends it."""
    assert _out(b"+[]", steps=1000) == b""
    # `+[.]` emits one byte per iteration, so the budget bounds the output.
    assert _out(b"+[.]", steps=7) == b"\x01\x01\x01"


def test_out_len_is_appendix_e_s_zero_padding_and_truncation():
    """Appendix E: the output is "the first T bytes emitted, zero-padded"."""
    assert _out(b"+.", out_len=4) == b"\x01\x00\x00\x00"
    assert _out(b"+[.]", steps=1000, out_len=3) == b"\x01\x01\x01"


def test_every_string_is_executable_fuzz():
    """§2.1: "we design the execution semantics so that every generated string is
    executable: programs cannot fail through syntax or memory errors". Fuzzed
    over arbitrary bytes, not just the alphabet — a caller must not be able to
    construct a program that faults."""
    rng = random.Random(20261018)
    out_len = 16
    for _ in range(1000):
        n = rng.randrange(0, 40)
        program = bytes(rng.randrange(256) for _ in range(n))
        tape = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 12)))
        got = utm.run(program, tape, rng.choice([0, 1, 7, 64, 512]), out_len=out_len)
        assert len(got) == out_len, f"{program!r} produced {len(got)} bytes"


def test_the_random_tape_makes_one_program_a_distribution():
    """§2.1: "The random input tape allows a single program to represent a
    distribution over output sequences." `,.>,.>,.>,.>,.` emits the tape's first
    five bytes, so 32 random tapes give more than one output."""
    rng = random.Random(7)
    program = b",.>,.>,.>,.>,."
    outputs = set()
    for _ in range(32):
        tape = bytes(rng.randrange(256) for _ in range(8))
        got = utm.run(program, tape, 64)
        assert got == tape[:5]
        outputs.add(got)
    assert len(outputs) > 1


# ---------------------------------------------------------------------------
# Eq. 2 — the reward, against its definition
# ---------------------------------------------------------------------------


def _sp_cfg(**over) -> ModelConfig:
    """~20k parameters: the arms on, everything small enough to be instant."""
    base = dict(
        d_model=32,
        n_heads=4,
        head_dim=8,
        d_ffn=64,
        vocab=256,
        max_seq_len=64,
        max_iter=1,
        rank=4,
        n_experts=1,
        moe_topk=0,
        engram_rows=32,
        engram_dim=8,
        bf16=False,
        shared_rank=8,
    )
    base.update(over)
    return ModelConfig(**base)


def _seq_ce(model: DormouseModel, outputs: torch.Tensor) -> torch.Tensor:
    inputs, targets = sp_reward.output_batch(outputs)
    n, t = inputs.shape
    return sp_reward.sequence_ce(model(inputs, None)[0].reshape(n, t, -1), targets).mean()


def _stepped_model(seed: int = 0, steps: int = 40, lr: float = 1e-2):
    """A model, its AdamW, and a `(theta_prev, theta)` pair 20 steps apart."""
    torch.manual_seed(seed)
    model = DormouseModel(_sp_cfg())
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    pool = [
        torch.tensor([list(bytes((i + j) % 6 for j in range(16)))], dtype=torch.long)
        for i in range(4)
    ]
    prev = None
    for step in range(steps):
        if step == steps // 2:
            prev = {k: v.detach().clone() for k, v in model.state_dict().items()}
        opt.zero_grad(set_to_none=True)
        sum(_seq_ce(model, y) for y in pool).backward()
        opt.step()
    assert prev is not None
    return model, opt, prev, pool


def test_preconditioner_is_lr_over_sqrt_vhat_plus_eps_by_hand():
    """Eq. 2's `P_e = lr/(sqrt(v̂_e) + ε)`, hand-computed from AdamW's state.

    `v̂` is the bias-corrected second moment, `exp_avg_sq / (1 − β2^t)`, which is
    the paper's `v̂_e` — the raw buffer is `v_e` and using it would put a
    `sqrt(1−β2^t)` error into every reward.
    """
    model, opt, _prev, _pool = _stepped_model()
    precond = sp_reward.adamw_preconditioner(opt, model)
    beta2 = 0.999
    checked = 0
    for name, p in model.named_parameters():
        state = opt.state.get(p)
        if state is None:  # never stepped: P is 0 by the documented convention
            assert not precond[name].any(), f"{name}: stateless parameter got non-zero P"
            continue
        vhat = state["exp_avg_sq"] / (1.0 - beta2 ** int(state["step"]))
        want = 1e-2 / (vhat.sqrt() + sp_reward.DEFAULT_EPS)
        assert torch.allclose(precond[name], want, rtol=0, atol=0), name
        checked += 1
    assert checked > 10, "the gold check must actually cover the parameters"


def test_direction_is_preconditioner_times_delta_theta_by_hand():
    """Eq. 2: `u = P_e ⊙ δθ_e` with `δθ_e = θ_{p(e)} − θ_e`, elementwise."""
    model, opt, prev, _pool = _stepped_model()
    precond = sp_reward.adamw_preconditioner(opt, model)
    u = sp_reward._direction(model, prev, precond)
    for name, p in model.named_parameters():
        want = precond[name].to(p.dtype) * (prev[name].to(p.dtype) - p.detach())
        assert torch.equal(u[name], want), name
    # A stateless parameter never moved, so δθ is exactly zero there.
    moved = sum(int(u[n].any()) for n in u)
    assert moved > 0, "the lookback window must carry real movement"


def test_reward_equals_a_central_finite_difference_of_the_loss():
    """Eq. 2's `⟨∇L(y;θ), u⟩` against `(L(θ+h·u) − L(θ−h·u)) / 2h`.

    This is the definition, evaluated without autodiff — so it checks the inner
    product itself rather than re-running the same machinery.
    """
    model, opt, prev, _pool = _stepped_model()
    precond = sp_reward.adamw_preconditioner(opt, model)
    u = sp_reward._direction(model, prev, precond)
    y = torch.tensor([list(b"\x01\x03\x02\x05" * 4)], dtype=torch.long)

    rewards = sp_reward.gradient_alignment_rewards(
        model, y, theta_prev=prev, precond=precond, method="jvp"
    )

    inputs, targets = sp_reward.output_batch(y)
    params = dict(model.named_parameters())
    # h = 1e-4 is the sweet spot between truncation and fp32 round-off
    # on this fixture (measured: 1e-2 -> 1.8e-1, 1e-4 -> 2.5e-3,
    # 1e-6 -> 2.5e-1 relative error against the JVP).
    h = 1e-4

    def loss_at(theta):
        logits = functional_call(model, theta, (inputs,), {"targets": None})[0]
        return sp_reward.sequence_ce(logits.reshape(inputs.shape[0], -1, 256), targets)

    with torch.no_grad():
        plus = {n: p + h * u[n] for n, p in params.items()}
        minus = {n: p - h * u[n] for n, p in params.items()}
        fd = float(((loss_at(plus) - loss_at(minus)) / (2 * h)).abs().mean())
    assert fd == pytest.approx(rewards[0], rel=1e-2), (fd, rewards[0])


def test_forward_mode_and_the_reverse_mode_fallback_agree():
    """The paper's mechanism and the documented fallback must be one number."""
    model, opt, prev, pool = _stepped_model()
    precond = sp_reward.adamw_preconditioner(opt, model)
    batch = torch.cat(pool[:3])
    jvp = sp_reward.gradient_alignment_rewards(
        model, batch, theta_prev=prev, precond=precond, method="jvp"
    )
    vjp = sp_reward.gradient_alignment_rewards(
        model, batch, theta_prev=prev, precond=precond, method="vjp"
    )
    assert len(jvp) == len(vjp) == 3
    for a, b in zip(jvp, vjp, strict=True):
        assert a == pytest.approx(b, rel=1e-4), (jvp, vjp)


def test_reward_orders_mastered_below_misaligned_below_aligned():
    """Eq. 2's three regimes, in §2.2's order.

    §2.2: "programs that are already mastered induce nearly zero gradients, while
    programs containing unrelated or unlearnable structure induce gradients that
    do not align with the learner's parameter movement. Both receive little
    reward. Instead, high reward is assigned to programs that induce substantial
    learning, but only in directions congruent with the learner's recent
    progress."

    Measured on this fixture (printed): a mastered sequence has `|g| = 0.016` with
    `|cos| = 5e-8`, a random-byte output has `|g| = 22.6` with `|cos| = 5.2e-5`, and
    an unlearned arithmetic ramp has `|g| = 31.5` with `|cos| = 6.7e-4`. Eq. 2 is
    the product of those two factors and both point the same way here, so the
    ordering is the paper's. The caveat that keeps it honest: the absolute value
    is **not** scale-invariant, so a program whose gradient norm is far above the
    pool's wins on magnitude alone — the reward ranks within a pool of
    comparable programs, which is what the paper's UTM outputs are.
    """
    model, opt, prev, _pool = _stepped_model()
    mastered = torch.tensor([list(b"\x02\x02\x03\x05" * 4)], dtype=torch.long)
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        _seq_ce(model, mastered).backward()
        opt.step()
    assert float(_seq_ce(model, mastered).detach()) < 0.01, "the sequence must be mastered"
    # An arithmetic ramp modulo 251: structured (the paper's Table 1 family) and
    # never trained on. Random bytes: high entropy, no structure at all.
    unlearned = torch.tensor([list(bytes((j * 5) % 251 for j in range(16)))], dtype=torch.long)
    rand = torch.randint(0, 256, (1, 16), generator=torch.Generator().manual_seed(11))
    batch = torch.cat([mastered, rand, unlearned])

    precond = sp_reward.adamw_preconditioner(opt, model)
    rewards = sp_reward.gradient_alignment_rewards(model, batch, theta_prev=prev, precond=precond)
    u = sp_reward._direction(model, prev, precond)
    u_norm = float(sum(v.pow(2).sum() for v in u.values()).sqrt())
    for i, (name, r) in enumerate(zip(("mastered", "random", "unlearned"), rewards, strict=True)):
        y = batch[i : i + 1]
        model.zero_grad(set_to_none=True)
        _seq_ce(model, y).backward()
        grads = {k: p.grad.detach() for k, p in model.named_parameters() if p.grad is not None}
        g_norm = float(sum(v.pow(2).sum() for v in grads.values()).sqrt())
        dot = sum(float((grads[k] * u[k]).sum()) for k in grads)
        print(
            f"\n{name:10s} r={r:12.6g} CE={float(_seq_ce(model, y).detach()):8.4f} "
            f"|g|={g_norm:9.4g} |cos|={abs(dot) / (g_norm * u_norm + 1e-30):.3e}"
        )
    r_mastered, r_rand, r_unlearned = rewards
    assert r_mastered < r_rand < r_unlearned, rewards


def test_reward_is_linear_in_the_direction():
    """Eq. 2 is an inner product: scaling `u` scales every reward by the same factor."""
    model, opt, prev, pool = _stepped_model()
    precond = sp_reward.adamw_preconditioner(opt, model)
    batch = torch.cat(pool[:2])
    base = sp_reward.gradient_alignment_rewards(model, batch, theta_prev=prev, precond=precond)
    scaled = sp_reward.gradient_alignment_rewards(
        model, batch, theta_prev=prev, precond={k: 3.0 * v for k, v in precond.items()}
    )
    for a, b in zip(base, scaled, strict=True):
        assert b == pytest.approx(3.0 * a, rel=1e-5)


def test_lookback_is_the_papers_floor_of_e_over_two():
    """Eq. 2: `p(e) = ⌊e/2⌋`, the growing lookback window."""
    assert [sp_reward.lookback(e) for e in range(7)] == [0, 0, 1, 1, 2, 2, 3]


def test_the_missing_snapshot_is_loud_not_a_partial_dot_product():
    """A short `θ_prev` would silently drop parameters from Eq. 2."""
    model, opt, prev, _pool = _stepped_model(steps=20)
    precond = sp_reward.adamw_preconditioner(opt, model)
    first_param = next(iter(dict(model.named_parameters())))
    short = {k: v for k, v in prev.items() if k != first_param}
    with pytest.raises(Exception, match="theta_prev is missing"):
        sp_reward._direction(model, short, precond)


# ---------------------------------------------------------------------------
# Appendix G / Eqs. 4-5 — the pool
# ---------------------------------------------------------------------------


def test_prior_sampling_terminates_and_g0_normalizes_to_a_closed_form():
    """§2.2: `g_0(x) = |A|^{-ℓ(x)}` with ℓ the tokens up to and including `F`.

    `|A| = 19`, and because the last token is always `F`, the masses over all
    programs of length ≤ L sum to `1 − (18/19)^L` — a closed form, checked here
    by enumeration for L = 3.
    """
    rng = random.Random(3)
    for _ in range(200):
        x = sp_pool.sample_prior(rng, max_len=8)
        assert x.endswith(b"F") and b"F" not in x[:-1]
        assert len(x) <= 8
    total = 0.0
    for length in (1, 2, 3):
        for body in itertools.product(sp_pool.PROGRAM_TOKENS, repeat=length - 1):
            if b"F" in bytes(body):
                continue
            total += math.exp(sp_pool.prior_logprob(bytes(body) + b"F"))
    assert total == pytest.approx(1.0 - (18 / 19) ** 3, rel=1e-12)
    assert sp_pool.prior_logprob(b"F") == pytest.approx(-math.log(19), rel=1e-12)


def test_mutation_is_one_local_edit_that_stays_in_the_alphabet():
    """Appendix G: "single-token substitutions, insertions, or deletions"."""
    rng = random.Random(5)
    base = b"+++[>+.<-]F"
    lengths = set()
    for _ in range(300):
        got = sp_pool.mutate(base, rng, max_len=32)
        assert got.endswith(b"F")
        body = got[:-1]
        assert set(body) <= set(utm.ALPHABET), got
        lengths.add(len(body))
    assert lengths == {len(base) - 2, len(base) - 1, len(base)}


def test_pool_composition_fresh_mut_replay_and_the_bank_grows():
    """Appendix G: fresh ∪ mut ∪ replay, with mutations coming from the bank."""
    cfg = sp_pool.PoolConfig(n_fresh=3, n_mut=2, n_replay=2, program_len=8, seed=1)
    pool = sp_pool.ProgramPool(cfg)
    first = pool.build(lambda n: [sp_pool.sample_prior(pool.rng, 8) for _ in range(n)])
    assert first.counts() == {"fresh": 3}
    assert pool.bank == [], "round 0 has no bank to mutate or replay from"

    # Give three programs positive reward and a stored sampling probability.
    for i, item in enumerate(first.items):
        item.reward = float(i + 1)
        item.logp = -math.log(19) * len(item.program)
    pool.absorb(first)
    assert len(pool.bank) == 3
    assert all(e.prob > 0.0 for e in pool.bank), "the bank must store the entry prob"

    second = pool.build(lambda n: [sp_pool.sample_prior(pool.rng, 8) for _ in range(n)])
    counts = second.counts()
    assert counts["fresh"] == 3 and counts["mut"] == 2 and counts["replay"] == 2
    for item in second.items:
        if item.origin == "replay":
            assert item.old_logp is not None, "Eq. 4's rho needs the stored log-prob"
        if item.origin == "mut":
            assert item.logp is None, "mutation rows are not from a proposal"


def test_the_bank_admits_only_positive_reward_and_decays_every_round():
    """Appendix G: positive reward only, decayed 0.97 each round."""
    cfg = sp_pool.PoolConfig(
        n_fresh=2, n_mut=0, n_replay=0, program_len=6, reward_decay=0.5, seed=2
    )
    pool = sp_pool.ProgramPool(cfg)
    p = pool.build(lambda n: [b"+++F"] * n)
    p.items[0].reward = 4.0
    p.items[1].reward = -1.0
    pool.absorb(p)
    assert [e.reward for e in pool.bank] == [4.0]
    q = pool.build(lambda n: [b"+++F"] * n)
    q.items[0].reward = 0.0
    q.items[1].reward = 0.0
    pool.absorb(q)
    assert [e.reward for e in pool.bank] == [2.0], "0.5 decay on the stored reward"


def test_ei_weights_are_normalized_and_zero_without_a_positive_reward():
    """Eq. 5: `w_i = [r_i]_+ / Σ_j [r_j]_+`."""
    w = sp_pool.ei_weights([0.0, 2.0, -3.0, 2.0])
    assert sum(w) == pytest.approx(1.0, rel=1e-12)
    assert w == [0.0, 0.5, 0.0, 0.5]
    assert sp_pool.ei_weights([0.0, -1.0]) == [0.0, 0.0]


def test_importance_ratio_is_clipped_at_e_plus_minus_20():
    """Eq. 4: "We clip the sequence-level importance ratio to ρ_i ∈ [e^-20, e^20]"."""
    assert sp_pool.importance_ratio(0.0, 0.0) == 1.0
    assert sp_pool.importance_ratio(math.log(4.0), 0.0) == pytest.approx(4.0)
    assert sp_pool.importance_ratio(1000.0, 0.0) == pytest.approx(math.exp(20.0))
    assert sp_pool.importance_ratio(-1000.0, 0.0) == pytest.approx(math.exp(-20.0))


def test_advantage_is_group_normalized_minus_the_solomonoff_kl():
    """Eq. 4's `A_i = (r_i − r̄)/(σ_r + ε) − β(log g_φ(x_i) − log g_0(x_i))`."""
    rewards = [1.0, 2.0, 3.0]
    logp = [-1.0, -2.0, -3.0]
    logp0 = [-1.0, -1.0, -1.0]
    beta = 0.5
    adv = sp_pool.advantages(rewards, logp, logp0, beta=beta, eps=0.0)
    mean = 2.0
    sd = math.sqrt(2 / 3)
    want = [
        (r - mean) / sd - beta * (lp - l0) for r, lp, l0 in zip(rewards, logp, logp0, strict=True)
    ]
    for a, b in zip(adv, want, strict=True):
        assert a == pytest.approx(b, rel=1e-12)


# ---------------------------------------------------------------------------
# §2.2 — the round
# ---------------------------------------------------------------------------


def _round_cfg(**over) -> sp_loop.RoundConfig:
    pool = sp_pool.PoolConfig(
        n_fresh=8,
        n_mut=4,
        n_replay=4,
        program_len=16,
        out_len=32,
        max_steps=512,
        tape_bytes=16,
        bank_top=64,
        seed=0,
    )
    base = dict(pool=pool, lr=1e-3, generator_lr=1e-3, seed=0)
    base.update(over)
    return sp_loop.RoundConfig(**base)


def _big_model() -> DormouseModel:
    """~100k parameters: the paper's smallest scale, instantiated on CPU."""
    return DormouseModel(
        tiny_cfg(
            d_model=80,
            n_heads=4,
            head_dim=20,
            d_ffn=320,
            max_iter=1,
            n_experts=1,
            moe_topk=0,
            rank=8,
            shared_rank=20,
            engram_rows=32,
        )
    )


#: A fixed natural-ish held-out sequence — the zero-shot quantity the paper
#: actually reports is bits per byte on natural data, so the smoke's monitor is
#: the learner's next-byte CE on a fixed text, never on its own UTM outputs.
HELD_OUT = (
    b"the quick brown fox jumps over the lazy dog. a byte-level model predicts the next byte. "
)[:64]


def _held_out_ce(model: DormouseModel, text: bytes = HELD_OUT) -> float:
    ids = torch.tensor(list(text), dtype=torch.long)[None, :]
    inputs, targets = ids[:, :-1], ids[:, 1:]
    n, t = inputs.shape
    with torch.no_grad():
        logits = model(inputs, None)[0].reshape(n, t, -1)
        return float(sp_reward.sequence_ce(logits, targets).mean())


def test_round_runs_mutation_only_and_the_learner_moves():
    """§2.2's round with `generator=None`, on ~100k parameters, for six rounds.

    Mutation-only mode is a deliberate first-experiment simplification (see
    `loop.py`): fresh programs come from `g_0` instead of a policy, so the only
    adaptive part of the round is Appendix G's mutation of bank programs.

    **The held-out number is printed, not asserted, and it goes UP.** Measured
    here: the learner's next-byte CE on a fixed natural-ish text rises
    monotonically (6.10 → 6.54 over six rounds at `lr=1e-3`, and faster at a
    higher `lr`), because a uniformly sampled program is almost always a no-op —
    87% of `g_0` draws emit nothing at all and 1.0% of the emitted bytes are
    non-zero, so the learner is trained to predict a constant zero. The paper's
    fixed-prior baseline is its *slow-scaling control* (figure 2), and it needs
    a 34.36B-token budget to improve at all; six rounds at 100k parameters cannot
    show the transfer the paper reports, and this test does not pretend
    otherwise. What it does assert is the mechanism: every stat finite, the
    learner step reducing the CE on the batch it stepped on, the bank filling
    from positive rewards, and the counters that prove each arm ran.
    """
    torch.manual_seed(0)
    learner = _big_model()
    assert 90_000 < param_count(learner) < 110_000, param_count(learner)
    cfg = _round_cfg()
    pool = sp_pool.ProgramPool(cfg.pool)
    held_out = [_held_out_ce(learner)]
    counters = Counters()

    seen = []
    for _ in range(6):
        stats = sp_loop.round(learner, None, pool, cfg, counters=counters)
        assert stats.n_pool >= cfg.pool.n_fresh, stats.composition
        assert stats.composition["fresh"] == cfg.pool.n_fresh
        assert stats.composition.get("replay", 0) <= cfg.pool.n_replay
        seen.append(stats.composition)
        for value in (
            stats.learner_loss,
            stats.learner_loss_after,
            stats.mean_reward,
            stats.max_reward,
        ):
            assert math.isfinite(value), stats
        assert stats.generator_loss is None, "mutation-only mode has no generator"
        assert stats.learner_loss_after < stats.learner_loss, "the learner step must work"
        held_out.append(_held_out_ce(learner))

    print(f"\nheld-out CE per round: {[round(v, 4) for v in held_out]}")
    print(f"pool composition per round: {seen}")
    print(f"counters: {counters.line()}")
    assert counters["selfplay_round"] == 6
    assert counters["selfplay_mutation_only"] == 6
    assert counters["learner_step"] == 6
    assert counters["reward_no_optimizer_state"] == 1, "round 0 has no lookback"
    # Eq. 2 ran on every round but the first, on the whole pool each time.
    assert counters["executed"] == sum(sum(c.values()) for c in seen)
    assert counters["reward_jvp"] == sum(sum(c.values()) for c in seen[1:])
    assert len(pool.bank) > 0, "positive rewards must reach the bank"
    assert pool.round == 6 and len(pool.theta) == 6
    assert seen[0] == {"fresh": cfg.pool.n_fresh}, "round 0 has no bank to mutate or replay"
    assert seen[1] == {"fresh": cfg.pool.n_fresh}, (
        "round 1 still has an empty bank: round 0's rewards are all 0 (no lookback), "
        "so the first admissions happen after round 1"
    )
    assert seen[-1].get("mut", 0) == cfg.pool.n_mut, "the bank must be feeding mutations"
    assert seen[-1].get("replay", 0) == cfg.pool.n_replay


def test_round_zero_has_no_lookback_and_so_no_reward():
    """Eq. 2's window has nothing behind it at `e = 0`: `δθ_0 = 0` exactly."""
    torch.manual_seed(0)
    learner = _big_model()
    cfg = _round_cfg()
    pool = sp_pool.ProgramPool(cfg.pool)
    stats = sp_loop.round(learner, None, pool, cfg)
    assert stats.mean_reward == 0.0 and stats.max_reward == 0.0
    assert stats.counters["reward_no_optimizer_state"] == 1
    assert "reward_jvp" not in stats.counters


def test_generator_step_runs_equations_three_to_five_and_moves_the_generator():
    """§2.2's generator update: Eq. 3-4's PG term plus Eq. 5's EI term.

    Mutation-only mode skips this half entirely, so it gets its own gate: with a
    real (tiny) generator the round must produce a finite generator loss and
    actually move the generator's parameters, and `program_logprobs` must be the
    masked `log g_φ` Eq. 4 divides by.
    """
    torch.manual_seed(0)
    learner = _big_model()
    torch.manual_seed(1)
    generator = DormouseModel(_sp_cfg())
    cfg = _round_cfg(beta=0.05, lam_ei=1.0)
    pool = sp_pool.ProgramPool(cfg.pool)
    before = {k: v.detach().clone() for k, v in generator.named_parameters()}

    counters = Counters()
    for _ in range(3):
        stats = sp_loop.round(learner, generator, pool, cfg, counters=counters)
        assert stats.generator_loss is not None and math.isfinite(stats.generator_loss)
    stats = dataclasses.replace(stats, counters=counters)
    print(f"\ngenerator loss={stats.generator_loss:.6g} counters={stats.counters.line()}")
    assert stats.counters["generator_step"] == 3
    assert stats.counters["generator_pg"] >= 3 * 4, "mutations excluded from Eq. 4"
    assert stats.counters["generator_ei"] == 3
    moved = sum(
        int(not torch.equal(before[k], v.detach())) for k, v in generator.named_parameters()
    )
    assert moved > 0, "the generator step must reach the generator's weights"


def test_program_logprobs_is_the_masked_alphabet_distribution():
    """§2.3 restricts the generator's logits to the 19 program tokens.

    Gold: for the one-token program `F`, `log g_φ` is the log-softmax over those
    19 tokens at the position after the `S` prefix, computed here by hand from
    the generator's own forward.
    """
    torch.manual_seed(3)
    generator = DormouseModel(_sp_cfg())
    got = sp_loop.program_logprobs(generator, [b"F"]).detach()
    with torch.no_grad():
        ids = torch.tensor([[sp_reward.PROGRAM_PREFIX]])
        logits = generator(ids, None)[0].reshape(1, 1, -1)[0, 0].float()
        restricted = logits[sp_loop._TOKENS]
        log_softmax = restricted - torch.logsumexp(restricted, dim=-1)
        want = float(log_softmax[sp_loop._TOKEN_INDEX[ord("F")]])
    assert float(got) == pytest.approx(want, rel=1e-6)
    assert float(got) < 0.0


def test_sample_from_generator_stays_in_the_alphabet_and_terminates():
    """§2.3's masked sampling: every token in the 19-token vocabulary."""
    torch.manual_seed(4)
    generator = DormouseModel(_sp_cfg())
    progs = sp_loop.sample_from_generator(
        generator, 5, max_len=12, rng=torch.Generator().manual_seed(0)
    )
    assert len(progs) == 5
    for x in progs:
        assert x.endswith(b"F") and b"F" not in x[:-1]
        assert set(x[:-1]) <= set(utm.ALPHABET), x
        assert len(x) <= 13


def test_a_different_learner_is_refused_by_the_pools_cached_optimizer():
    """The reward reads the optimizer state, so a swapped learner must be LOUD."""
    torch.manual_seed(0)
    learner = _big_model()
    cfg = _round_cfg()
    pool = sp_pool.ProgramPool(cfg.pool)
    sp_loop.round(learner, None, pool, cfg)
    with pytest.raises(RuntimeError, match="different model"):
        pool.learner_optimizer(_big_model(), cfg.lr)


def test_round_stats_counters_are_a_counters_object():
    """House style: the round reports through `instrument.Counters`."""
    torch.manual_seed(0)
    learner = _big_model()
    cfg = _round_cfg()
    pool = sp_pool.ProgramPool(cfg.pool)
    counters = Counters()
    stats = sp_loop.round(learner, None, pool, cfg, counters=counters)
    assert stats.counters is counters
    assert "selfplay_round=1" in counters.line()
