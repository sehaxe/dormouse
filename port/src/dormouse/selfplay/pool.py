"""The program pool `B_e = B_fresh ∪ B_mut ∪ B_replay` — paper §2.2, Appendix G.

One self-play round's data is a *pool*, not a sample: fresh programs from the
current generator give global exploration, local mutations of previously
high-reward programs refine promising regions, and replay of earlier rounds
preserves structures the generator has started to forget. Appendix G fixes the
three pieces and the bank they share:

- **mutation** is "single-token substitutions, insertions, or deletions of
  positively rewarded programs drawn from the quality-diversity bank";
- **replay** is "drawing programs uniformly without replacement from the bank of
  non-replay programs produced in earlier rounds", re-executed "with fresh
  random tapes";
- the **bank** admits only programs with positive generator reward, and its
  stored rewards are "decayed by a factor of 0.97 each round, allowing newly
  useful programs to replace stale elites".

Two things this module stores that are easy to lose and expensive to lose:

- **the sampling probability each replay program entered the bank with.** Eq. 4
  corrects for off-policy samples with `ρ_i = g_φ(x_i) / g_{φ_old,i}(x_i)`, and
  Appendix G's own wording is that "its denominator is the sampling probability
  stored when the program originally entered the replay bank". Without the
  stored value there is no ρ and the bank becomes silently on-policy.
- **the fresh/replay log-probabilities** (`log g_φ(x_i)`), which Eq. 4's KL term
  and ρ both read.

`mutation-only mode` (§2.2's generator, absent). With no generator the pool is
built from the Solomonoff prior `g_0(x) = |A|^{-ℓ(x)}` instead, which is the
paper's own fixed-prior baseline — so the fresh half of a mutation-only pool is
the control condition of figure 2, not an invention. Every program is still
re-executed and still trains the learner; only the PG/EI half of the round
needs a generator.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from .reward import lookback
from .utm import ALPHABET, END, PROGRAM_TOKENS

__all__ = [
    "RHO_CLIP",
    "BankEntry",
    "Pool",
    "PoolConfig",
    "PoolItem",
    "ProgramPool",
    "advantages",
    "ei_weights",
    "importance_ratio",
    "mutate",
    "prior_logprob",
    "sample_prior",
]

#: Eq. 4: "We clip the sequence-level importance ratio to ρ_i ∈ [e^-20, e^20]".
RHO_CLIP = 20.0

_ALPHABET = ALPHABET
_END = END[0]
_LOG_ALPHABET = math.log(len(PROGRAM_TOKENS))


@dataclass
class PoolConfig:
    """The pool's half of a round's knobs (Appendix G's quantities)."""

    #: |B_fresh|, |B_mut|, |B_replay| — the three parts of the pool.
    n_fresh: int = 8
    n_mut: int = 4
    n_replay: int = 4
    #: L, the program-length bound of `x ∈ A^{≤L}` (§2.1). Also the mutation cap.
    program_len: int = 32
    #: T, the output length: `utm.run(..., out_len=T)` (Appendix E's zero-pad).
    out_len: int = 64
    #: The machine's step budget (Appendix E names one without a value).
    max_steps: int = 4096
    #: Bytes of random input tape ω drawn per program per round (§2.1).
    tape_bytes: int = 64
    #: Bank capacity. Appendix G keeps top-8 per niche; see `ProgramPool.absorb`.
    bank_top: int = 512
    #: Appendix G: "stored rewards are decayed by a factor of 0.97 each round".
    reward_decay: float = 0.97
    seed: int = 0


@dataclass
class BankEntry:
    """One archived program: its program, its stored reward, its entry prob."""

    program: bytes
    reward: float
    #: `g_{φ_old}(x)` at the moment the program entered the bank — Eq. 4's
    #: denominator when the program comes back as a replay sample.
    prob: float


@dataclass
class PoolItem:
    """One program in a round's pool, with what Eqs. 4-5 need to be computed."""

    program: bytes
    origin: str  # "fresh" | "mut" | "replay"
    #: `log g_φ(x)` under the generator that built this pool; `None` for a
    #: mutation (Appendix G: mutations "were not sampled from a proposal
    #: distribution with a well-defined log probability", so Eq. 4 drops them).
    logp: float | None = None
    #: The stored sampling log-probability — replay samples only (Eq. 4's ρ).
    old_logp: float | None = None
    #: Eq. 2's reward, filled in by the round.
    reward: float = 0.0


@dataclass
class Pool:
    """One round's pool: the items, their executed outputs, and the round index."""

    items: list[PoolItem]
    round: int
    #: `[M, T]` byte ids, `utm.run(item.program, ω_i, ..., out_len=T)` per item.
    outputs: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.items)

    @property
    def rewards(self) -> list[float]:
        return [it.reward for it in self.items]

    def counts(self) -> dict[str, int]:
        """`{origin: n}` — the pool's composition, for the round's counters."""
        out: dict[str, int] = {}
        for it in self.items:
            out[it.origin] = out.get(it.origin, 0) + 1
        return out


def sample_prior(rng: random.Random, max_len: int = 32) -> bytes:
    """Draw `x ~ g_0`, the Solomonoff prior `g_0(x) = |A|^{-ℓ(x)}` (§2.2).

    Instruction tokens are drawn i.i.d. uniform over the 19 program tokens
    (Appendix E's alphabet plus `F`) until `F` terminates the program — the
    paper's fixed-prior baseline. A draw that reaches the length bound without an
    `F` gets one appended, so every returned value is a program that halts and no
    value exceeds `max_len` (which counts the `F`, as the paper's `l(x)` does).
    """
    assert max_len >= 1, f"max_len must be >= 1, got {max_len}"
    out = bytearray()
    # L counts the whole program including the terminating F (the paper's l(x)
    # is "the number of tokens up to and including F"), so the body gets L - 1.
    for _ in range(max_len - 1):
        tok = PROGRAM_TOKENS[rng.randrange(len(PROGRAM_TOKENS))]
        out.append(tok)
        if tok == _END:
            return bytes(out)
    out.append(_END)
    return bytes(out)


def prior_logprob(program: bytes) -> float:
    """`log g_0(x) = −ℓ(x)·log|A|`, with ℓ the tokens up to and including `F`.

    `|A| = 19` (8 instructions + 10 macros + `F`) is what makes `g_0` a
    distribution rather than an unnormalized weight: the last token is always
    `F`, so `Σ_x 19^{-ℓ(x)} = Σ_ℓ (18/19)^{ℓ-1}/19 = 1`.
    """
    if not program or program[-1] != _END:
        raise ValueError(f"program must end with the F token, got {program[-1:]!r}")
    if _END in program[:-1]:
        raise ValueError("program must carry exactly one F, at its end")
    return -len(program) * _LOG_ALPHABET


def mutate(program: bytes, rng: random.Random, max_len: int = 32) -> bytes:
    """One local edit of `program`: a single-token substitution, insertion or deletion.

    Appendix G's mutation operator, over Appendix E's instruction alphabet only,
    so the result is always a program the machine can execute. The terminating
    `F` is not part of the body and is never touched.
    """
    assert max_len >= 2, f"max_len must be >= 2, got {max_len}"
    body = bytearray(program[:-1] if program.endswith(END) else program)
    ops = ["ins"] if len(body) < max_len - 1 else []
    if body:
        ops += ["sub", "del"]
    assert ops, "a program at max_len with an empty body cannot exist"
    op = rng.choice(ops)
    if op == "sub":
        body[rng.randrange(len(body))] = _ALPHABET[rng.randrange(len(_ALPHABET))]
    elif op == "ins":
        body.insert(rng.randrange(len(body) + 1), _ALPHABET[rng.randrange(len(_ALPHABET))])
    else:
        del body[rng.randrange(len(body))]
    return bytes(body) + END


def ei_weights(rewards: Sequence[float]) -> list[float]:
    """Eq. 5's `w_i = [r_i]_+ / Σ_j [r_j]_+`, normalized to sum to 1.

    `[r]_+ = max(r, 0)`, so a pool with no positive reward has no EI term at all
    (the paper's ratio is undefined there) and every weight is 0.0 — the caller
    must read the zeros as "EI is off this round", not as "uniform".
    """
    pos = [max(r, 0.0) for r in rewards]
    total = sum(pos)
    if total <= 0.0:
        return [0.0] * len(rewards)
    return [p / total for p in pos]


def importance_ratio(logp: float, old_logp: float) -> float:
    """Eq. 4's `ρ_i = g_φ(x_i) / g_{φ_old,i}(x_i)`, clipped to `e^{±20}`."""
    return math.exp(min(max(logp - old_logp, -RHO_CLIP), RHO_CLIP))


def advantages(
    rewards: Sequence[float],
    logp: Sequence[float],
    logp0: Sequence[float],
    *,
    beta: float,
    eps: float = 1e-8,
) -> list[float]:
    """Eq. 4's `A_i`, the GRPO batch-level advantage minus the Solomonoff KL.

    `r̄_e` and `σ_{r,e}` are the mean and standard deviation over the **full
    round pool**, as the paper defines them, and `σ` is the population standard
    deviation (divide by `M`) — the paper does not say, and the sample form
    changes the scale of every advantage by `sqrt(M/(M−1))`.
    """
    assert len(rewards) == len(logp) == len(logp0), "rewards and log-probs must align"
    m = len(rewards)
    assert m > 0, "an empty pool has no advantage"
    mean = sum(rewards) / m
    sd = math.sqrt(sum((r - mean) ** 2 for r in rewards) / m)
    return [
        (r - mean) / (sd + eps) - beta * (lp - l0)
        for r, lp, l0 in zip(rewards, logp, logp0, strict=True)
    ]


class ProgramPool:
    """The round's carrier: the bank, the θ history, and the round counter.

    `round()` in `loop.py` is a function of `(learner, generator, pool, cfg)`, so
    everything that must survive between two calls lives here: the archive, the
    parameter snapshots Eq. 2's lookback reads, and the optimizers (the reward's
    `P_e` is read out of the learner's optimizer state, so the state cannot be
    rebuilt per call).
    """

    def __init__(self, cfg: PoolConfig) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        #: The quality-diversity bank, Appendix G: positive-reward programs only.
        self.bank: list[BankEntry] = []
        #: `θ_e` per round — `theta[e]` is the learner at round `e`'s entry.
        self.theta: list[dict[str, torch.Tensor]] = []
        self.round = 0
        self._optimizers: dict[str, torch.optim.Optimizer] = {}
        self._owners: dict[str, int] = {}

    # --- the round's state -------------------------------------------------
    def _optimizer_for(
        self,
        key: str,
        model: torch.nn.Module,
        lr: float,
    ) -> torch.optim.Optimizer:
        """A cached AdamW for `model`, refused if the model underneath changed.

        The learner's optimizer is AdamW because Eq. 2's `P_e` **is** AdamW's
        diagonal step operator; a Muon+ group has no `v̂` and the reward would be
        undefined. (Upgrade path: a preconditioner that reads whatever the
        learner's real optimizer maintains.)
        """
        owner = id(model)
        if key in self._optimizers:
            if self._owners[key] != owner:
                raise RuntimeError(
                    f"the pool's {key} optimizer belongs to a different model: a "
                    f"new learner needs a new pool, or the reward reads another "
                    f"network's state"
                )
            return self._optimizers[key]
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        self._optimizers[key] = opt
        self._owners[key] = owner
        return opt

    def learner_optimizer(self, learner: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
        return self._optimizer_for("learner", learner, lr)

    def generator_optimizer(self, generator: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
        return self._optimizer_for("generator", generator, lr)

    # --- construction (Appendix G) ----------------------------------------
    def parents(self, n: int) -> list[BankEntry]:
        """Mutation parents, drawn from the bank (Appendix G).

        The paper draws them "uniformly across occupied niches, rather than
        uniformly across all archived programs, preserving representation for
        rarer structural behaviors such as deeply nested programs". The niches
        are **skipped** here: their descriptors (max dynamic loop depth binned
        0..8, body length bucketed at 8/16/32) are an unmeasured refinement at
        this scale, and `ponytail:` the upgrade path is the two-descriptor
        MAP-Elites of Appendix G once a run shows the bank concentrating on one
        structural family.
        """
        if not self.bank or n <= 0:
            return []
        if n >= len(self.bank):
            return list(self.bank)
        return self.rng.sample(self.bank, n)

    def replay(self, n: int) -> list[BankEntry]:
        """Replay samples: "uniformly without replacement" from the bank."""
        if not self.bank or n <= 0:
            return []
        return self.rng.sample(self.bank, min(n, len(self.bank)))

    def build(
        self,
        fresh_sampler: Callable[[int], list[bytes]],
        logp_fn: Callable[[list[bytes]], list[float]] | None = None,
    ) -> Pool:
        """Assemble round `self.round`'s pool (fresh ∪ mut ∪ replay).

        `fresh_sampler(n)` returns `n` fresh programs — the generator's samples
        when there is a generator, `g_0`'s when there is not. `logp_fn` maps a
        batch of programs to `log g_φ(·)` in one pass; without it (mutation-only
        mode) the pool carries no proposal log-probabilities and Eq. 4 has
        nothing to correct for.
        """
        cfg = self.cfg
        fresh = list(fresh_sampler(cfg.n_fresh))
        parents = self.parents(cfg.n_mut)
        replays = self.replay(cfg.n_replay)
        scored = fresh + [e.program for e in replays]
        lps: list[float | None] = (
            list(logp_fn(scored)) if logp_fn is not None else [None] * len(scored)
        )
        items: list[PoolItem] = []
        for i, program in enumerate(fresh):
            # ρ = 1 for fresh on-policy samples: the same distribution is the
            # numerator and the denominator.
            items.append(PoolItem(program, "fresh", logp=lps[i], old_logp=lps[i]))
        for entry in parents:
            items.append(PoolItem(mutate(entry.program, self.rng, cfg.program_len), "mut"))
        for j, entry in enumerate(replays):
            lp = lps[len(fresh) + j]
            old = math.log(entry.prob) if entry.prob > 0.0 else None
            items.append(PoolItem(entry.program, "replay", logp=lp, old_logp=old))
        return Pool(items=items, round=self.round)

    def absorb(self, pool: Pool) -> None:
        """Bank update after the round's rewards exist (Appendix G).

        Only non-replay programs with **positive** reward are admitted — the bank
        is "the bank of non-replay programs produced in earlier rounds", so a
        replay sample never re-enters it. Every stored reward is decayed first,
        which is what lets a newly useful program displace a stale elite.
        """
        cfg = self.cfg
        for entry in self.bank:
            entry.reward *= cfg.reward_decay
        for item in pool.items:
            if item.origin == "replay" or item.reward <= 0.0:
                continue
            prob = math.exp(item.logp) if item.logp is not None else 0.0
            self.bank.append(BankEntry(item.program, item.reward, prob))
        if len(self.bank) > cfg.bank_top:
            self.bank.sort(key=lambda e: e.reward, reverse=True)
            del self.bank[cfg.bank_top :]

    def snapshot(self, learner: torch.nn.Module) -> None:
        """Record `θ_e` for this round; Eq. 2's lookback reads `theta[e // 2]`.

        `ponytail:` the history keeps EVERY round's snapshot, so memory grows
        with the run. Eq. 2 only ever reads `⌊e/2⌋`, so keeping the last
        `ceil(e/2)+1` entries is enough — do that when a long run's RSS says so.
        """
        self.theta.append({k: v.detach().clone() for k, v in learner.state_dict().items()})

    def theta_prev(self, round_index: int | None = None) -> dict[str, torch.Tensor]:
        """`θ_{p(e)}` for the round about to run (Eq. 2's lookback).

        At round 0 the lookback is the round's own entry state, so `δθ_0 = 0`
        and every reward is 0 — the paper's window has nothing behind it yet,
        and a fabricated non-zero reward there would be an invention.
        """
        e = self.round if round_index is None else round_index
        return self.theta[lookback(e)]
