"""One self-play round — paper §2.2's three steps, as a function of the pool.

A round of *Self-Play Pretraining with Zero Data* is::

    1. program generation   {x_i} ~ g_φ, plus mutations and replays   (Appendix G)
    2. execution            y_i = U(x_i, ω_i), ω_i a random tape      (Appendix E)
    3. learner and generator update
         learner:   one gradient step on Eq. 1, the pool-mean CE
         generator: a GRPO step on Eq. 3's KL-regularized reward, plus
                    Eq. 5's reward-weighted SFT (expert iteration)

The reward is Eq. 2 and lives in `reward.py`; the pool is Appendix G and lives
in `pool.py`. This module is the order of those pieces and nothing else.

**Mutation-only mode.** `generator_or_none` may be `None`, and then the fresh
half of the pool is drawn from the Solomonoff prior `g_0` instead of a policy:
no generator, no PG term, no EI term, and the *only* adaptive part of the round
is the mutation operator over high-reward bank programs. This is a deliberate
first-experiment simplification, not a reading of the paper: the paper's
generator is a learned policy and Eq. 3-5 are its objective. It is the cheapest
arm that still exercises Eq. 1, Eq. 2 and Appendix G end to end, and the upgrade
path is exactly `generator_or_none=some_model` — `generator_step` below is the
Eq. 3-5 half, already wired and tested, and it takes over the moment a generator
arrives. What mutation-only mode gives up is the adaptive curriculum: with `g_0`
fixed, the pool's *proposal* distribution never concentrates on the frontier, so
only the mutations (0.97-decayed, positive-reward, bank-backed) carry any
frontier signal at all.

**The learner's optimizer is AdamW.** Eq. 2's `P_e` *is* the AdamW diagonal step
operator read out of the learner's optimizer state, so the learner has to run
one for the reward to be defined. The port's Muon+ mixer has no `v̂` per
parameter and would make Eq. 2 undefined; the upgrade path is a preconditioner
that reads whatever the real learner maintains.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from ..instrument import Counters
from . import utm
from .pool import (
    Pool,
    PoolConfig,
    ProgramPool,
    advantages,
    ei_weights,
    importance_ratio,
    prior_logprob,
    sample_prior,
)
from .reward import (
    PROGRAM_PREFIX,
    adamw_preconditioner,
    gradient_alignment_rewards,
    output_batch,
    sequence_ce,
)
from .utm import END, PROGRAM_TOKENS

__all__ = [
    "RoundConfig",
    "RoundStats",
    "learner_step",
    "generator_step",
    "program_logprobs",
    "round",
    "sample_from_generator",
]

_TOKENS = torch.tensor(list(PROGRAM_TOKENS), dtype=torch.long)
# Padded positions carry 0 and are masked out of the sum, so the default index
# never has to be meaningful; a real out-of-alphabet byte is refused below.
_TOKEN_INDEX = torch.zeros(256, dtype=torch.long)
for _i, _b in enumerate(PROGRAM_TOKENS):
    _TOKEN_INDEX[_b] = _i
_END = END[0]


def _device_of(model: torch.nn.Module) -> torch.device:
    """Where the model's parameters live — the batch seam, as in `train/loop.py`."""
    return next(model.parameters()).device


@dataclass
class RoundConfig:
    """A round's knobs. `pool` holds Appendix G's half."""

    pool: PoolConfig = field(default_factory=PoolConfig)
    #: The learner's AdamW learning rate — also Eq. 2's `lr` in `P_e`.
    lr: float = 1e-3
    #: The generator's learning rate (§2.2 tunes the generator-to-learner ratio;
    #: this is the ratio's other half, left at 1.0 until an A/B moves it).
    generator_lr: float = 1e-3
    #: β, Eq. 3's KL regularization coefficient toward the Solomonoff prior.
    beta: float = 0.0
    #: λ_EI, Eq. 5's weight on the reward-weighted SFT term (the paper's 1.0).
    lam_ei: float = 1.0
    eps: float = 1e-8
    #: `"jvp"` (Eq. 2's forward mode), `"vjp"` (the fallback), or `"auto"`.
    reward_method: str = "jvp"
    seed: int = 0
    generator_temperature: float = 1.0


@dataclass
class RoundStats:
    """What one round did, with the counters that prove each arm was entered."""

    round: int
    n_pool: int
    composition: dict[str, int]
    #: Eq. 1's pool-mean CE at the round's entry parameters.
    learner_loss: float
    #: The same quantity after the learner step, on the SAME outputs — the one
    #: number that says the learner step did something.
    learner_loss_after: float
    mean_reward: float
    max_reward: float
    #: Eq. 5's objective, `None` in mutation-only mode.
    generator_loss: float | None
    counters: Counters


def sample_from_generator(
    generator: torch.nn.Module,
    n: int,
    *,
    max_len: int,
    temperature: float = 1.0,
    rng: torch.Generator | None = None,
) -> list[bytes]:
    """`{x_i} ~ g_φ` — §2.3's masked, `S`-prefixed autoregressive sampling.

    All `n` sequences are stepped in lock-step to `max_len` tokens and each is
    then cut at its first `F`. Sampling past a sequence's `F` cannot change the
    tokens before it (the model is causal), so the truncated prefix is exactly a
    draw from `g_φ` stopped at the terminator — and the batch keeps one forward
    per position instead of one per position per sequence.

    A draw with no `F` inside the budget gets one appended, so the result is
    always a program that halts. Logits are restricted to §2.3's vocabulary
    (eight instructions, ten macros, `F`) before the softmax.
    """
    assert n > 0, "no sequences to sample"
    assert max_len >= 1, f"max_len must be >= 1, got {max_len}"
    device = _device_of(generator)
    tokens = _TOKENS.to(device)
    ids = torch.full((n, 1), PROGRAM_PREFIX, dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_len):
            logits = generator(ids, None)[0]
            logits = logits.reshape(n, ids.shape[1], -1)[:, -1, :].float()
            probs = torch.softmax(logits[:, tokens] / temperature, dim=-1)
            ids = torch.cat([ids, tokens[torch.multinomial(probs, 1, generator=rng)]], dim=1)
    out: list[bytes] = []
    for row in ids[:, 1:].tolist():
        program = bytearray()
        for b in row:
            program.append(b)
            if b == _END:
                break
        else:
            program.append(_END)
        out.append(bytes(program))
    return out


def program_logprobs(generator: torch.nn.Module, programs: Sequence[bytes]) -> torch.Tensor:
    """`log g_φ(x_i)` per program → `[N]`, differentiable (Eq. 3-4 read it).

    §2.3 restricts the generator's logits to the 19 program tokens during
    generation, so `g_φ` is the renormalized distribution over those 19 and not
    the raw 256-way byte distribution — the same restriction has to be applied
    here or the KL in Eq. 4 is against a different `g_φ` than the one that
    sampled. The sum is over the program's own tokens, padded positions masked
    out: a sequence log-probability, which is what Eqs. 3-5 are written in.
    """
    assert programs, "no programs to score"
    length = max(len(p) for p in programs)
    n = len(programs)
    inputs = torch.zeros((n, length), dtype=torch.long)
    targets = torch.zeros((n, length), dtype=torch.long)
    mask = torch.zeros((n, length), dtype=torch.bool)
    for i, program in enumerate(programs):
        if not program or program[-1] != _END:
            raise ValueError(f"program {i} does not end with the F token")
        stray = sorted(set(program) - set(PROGRAM_TOKENS))
        if stray:
            raise ValueError(f"program {i} has byte(s) outside the alphabet: {stray}")
        row = torch.tensor([PROGRAM_PREFIX, *program[:-1]], dtype=torch.long)
        inputs[i, : len(program)] = row
        targets[i, : len(program)] = torch.tensor(list(program), dtype=torch.long)
        mask[i, : len(program)] = True
    logits = generator(inputs, None)[0].reshape(n, length, -1).float()
    restricted = logits[..., _TOKENS.to(logits.device)]
    logp = restricted - torch.logsumexp(restricted, dim=-1, keepdim=True)
    index = _TOKEN_INDEX.to(targets.device)[targets]
    per_pos = logp.gather(-1, index[..., None]).squeeze(-1)
    return (per_pos * mask).sum(dim=1)


def learner_step(
    learner: torch.nn.Module,
    outputs: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    counters: Counters,
) -> tuple[float, float]:
    """Eq. 1: one gradient step on the pool-mean next-byte CE. `(before, after)`.

    `before` and `after` are both measured on the same outputs, at the entry
    parameters and after the step, so the pair is a same-batch reading of what
    the step did — a loss on a *new* pool cannot say that.
    """
    inputs, targets = output_batch(outputs)
    n, t = inputs.shape

    def pool_ce() -> float:
        with torch.no_grad():
            logits = learner(inputs, None)[0].reshape(n, t, -1)
            return float(sequence_ce(logits, targets).mean())

    before = pool_ce()
    optimizer.zero_grad(set_to_none=True)
    loss = sequence_ce(learner(inputs, None)[0].reshape(n, t, -1), targets).mean()
    assert torch.isfinite(loss), "the learner's CE must be finite"
    loss.backward()
    optimizer.step()
    counters.count("learner_step")
    return before, pool_ce()


def generator_step(
    generator: torch.nn.Module,
    pool: Pool,
    cfg: RoundConfig,
    optimizer: torch.optim.Optimizer,
    counters: Counters,
) -> float:
    """Eqs. 3-5: the GRPO policy-gradient step plus reward-weighted SFT.

    `L_gen = L_PG + λ_EI·L_EI` with `L_PG` Eq. 4 (mutation rows excluded — they
    "were not sampled from a proposal distribution with a well-defined log
    probability") and `L_EI` Eq. 5. The importance ratio and the advantage are
    both stop-gradient'd, as Eq. 4 writes them.
    """
    items = pool.items
    programs = [it.program for it in items]
    logp = program_logprobs(generator, programs)
    rewards = [it.reward for it in items]
    logp0 = [prior_logprob(p) for p in programs]
    adv = advantages(rewards, logp.detach().tolist(), logp0, beta=cfg.beta, eps=cfg.eps)
    keep = [i for i, it in enumerate(items) if it.origin != "mut"]
    pg = logp.new_zeros(())
    if keep:
        for i in keep:
            item = items[i]
            assert item.logp is not None and item.old_logp is not None, (
                "an on-policy sample must carry the log-prob Eq. 4's rho divides by"
            )
            rho = importance_ratio(item.logp, item.old_logp)
            pg = pg - rho * logp[i] * adv[i]
        pg = pg / len(keep)
        counters.count("generator_pg", len(keep))
    weights = ei_weights(rewards)
    ei = -(logp * logp.new_tensor(weights)).sum()
    counters.count("generator_ei")
    loss = pg + cfg.lam_ei * ei
    assert torch.isfinite(loss), "the generator's loss must be finite"
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    counters.count("generator_step")
    return float(loss.detach())


def _fresh_sampler(
    generator: torch.nn.Module | None,
    pool: ProgramPool,
    cfg: RoundConfig,
) -> Callable[[int], list[bytes]]:
    if generator is None:
        return lambda n: [sample_prior(pool.rng, cfg.pool.program_len) for _ in range(n)]
    return lambda n: sample_from_generator(
        generator,
        n,
        max_len=cfg.pool.program_len,
        temperature=cfg.generator_temperature,
        rng=torch.Generator(device=_device_of(generator)).manual_seed(cfg.seed + pool.round),
    )


def round(
    learner: torch.nn.Module,
    generator_or_none: torch.nn.Module | None,
    pool: ProgramPool,
    cfg: RoundConfig,
    *,
    counters: Counters | None = None,
) -> RoundStats:
    """Run one self-play round and return its [`RoundStats`].

    The pool is the state: it holds the bank, the round index, the `θ_e`
    snapshots Eq. 2's lookback reads and the two optimizers. Calling this N
    times with the same pool is N rounds; nothing else needs to persist.
    """
    counters = Counters() if counters is None else counters
    e = pool.round
    # θ_e, recorded before anything moves: Eq. 2 is defined at the round's entry
    # parameters, and θ_{p(e)} = theta[e // 2] must already be on file.
    pool.snapshot(learner)
    theta_prev = pool.theta_prev()

    logp_fn: Callable[[list[bytes]], list[float]] | None = None
    if generator_or_none is None:
        counters.count("selfplay_mutation_only")
    else:

        def _pool_logp(programs: list[bytes]) -> list[float]:
            with torch.no_grad():
                return [float(v) for v in program_logprobs(generator_or_none, programs)]

        logp_fn = _pool_logp

    built = pool.build(_fresh_sampler(generator_or_none, pool, cfg), logp_fn)

    # Step 2 — execute. ω_i is a fresh random tape per program per round (§2.1).
    outputs = [
        utm.run(
            it.program,
            pool.rng.randbytes(cfg.pool.tape_bytes),
            cfg.pool.max_steps,
            out_len=cfg.pool.out_len,
        )
        for it in built.items
    ]
    built.outputs = torch.tensor(
        [list(y) for y in outputs], dtype=torch.long, device=_device_of(learner)
    )
    counters.count("executed", len(outputs))

    # Step 3a — Eq. 2 at θ_e. Round 0 has no optimizer state and δθ_0 = 0, so the
    # reward is 0 by construction rather than by a fabricated number.
    optimizer = pool.learner_optimizer(learner, cfg.lr)
    if not optimizer.state:
        rewards = [0.0] * len(built.items)
        counters.count("reward_no_optimizer_state")
    else:
        precond = adamw_preconditioner(optimizer, learner, eps=cfg.eps)
        rewards = gradient_alignment_rewards(
            learner,
            built.outputs,
            theta_prev=theta_prev,
            precond=precond,
            method=cfg.reward_method,
            counters=counters,
        )
    for it, r in zip(built.items, rewards, strict=True):
        it.reward = r

    # Step 3b — the learner (Eq. 1) and, when there is one, the generator (§2.2).
    before, after = learner_step(learner, built.outputs, optimizer, counters)
    gen_loss = None
    if generator_or_none is not None:
        gen_loss = generator_step(
            generator_or_none,
            built,
            cfg,
            pool.generator_optimizer(generator_or_none, cfg.generator_lr),
            counters,
        )

    # Appendix G's bank update, then the round index moves.
    pool.absorb(built)
    pool.round = e + 1
    counters.count("selfplay_round")
    return RoundStats(
        round=e,
        n_pool=len(built),
        composition=built.counts(),
        learner_loss=before,
        learner_loss_after=after,
        mean_reward=sum(rewards) / len(rewards),
        max_reward=max(rewards),
        generator_loss=gen_loss,
        counters=counters,
    )
