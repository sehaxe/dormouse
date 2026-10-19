"""The staged data mixture: N `.bin` sources, per-stage weights, an epoch cap.

Today the pretrain loop streams ONE filtered-web corpus. The research recipe
(`docs/research/2026-10-16-data-mixing.md` §6.1) is a 3-stage static mix over
that corpus plus acquired code/math/RU — the shape SmolLM2 shipped, where the
decay stage is the lever that pays (math 3.21 → 22.07, code 8.87 → 23.21,
2502.02737):

| stage | share of the step budget | web | code | math | RU |
|---|---|---|---|---|---|
| S1 | 0 → 70% | 0.90 | 0.10 | — | — |
| S2 | 70 → 90% | 0.80 | 0.15 | 0.05 | — |
| S3 (decay) | 90 → 100% | 0.62 | 0.20 | 0.13 | 0.05 |

`MixtureStream` mirrors [`ByteStream`][dormouse.train.data.ByteStream]'s whole
interface — `next_batch()`, `rewind()`, `skip_bytes()` — so the trainer swap is
one line (`train/loop.py:506-507`): build a `MixtureStream` instead of calling
`open_streams` for the train side. Two properties are load-bearing and are what
the burn history paid for:

* **Determinism is a pure function of `(seed, step)`.** The source of step `k`
  is drawn from a hash of `(seed, k)` and the stage of step `k` from
  `(k, steps)`, never from a global RNG stream. A resume must land on the same
  mixture an unbroken run would have had — the defect class that forced DSpark's
  stride and the adaptive-depth sample to be pure functions of the step
  (ADR-0021), and the reason a streamed RNG would be wrong here: the trainer
  jumps straight to step `start`, it does not replay the draws.
* **One source per batch**, not a per-row split: `ByteStream.next_batch` splits
  one contiguous `batch × (seq_len+1)` read into rows, so a batch is a window
  into one source's byte range. The mixture therefore decides per STEP, and the
  realized proportions converge to the weights over a run (the bound is the
  binomial one, `sqrt(q/(n·p))` at `n` steps — ±2% needs ~10k steps for a 50%
  source and ~100k for a 5% one).
* **A source is a FILE or a DIRECTORY.** A `.bin` file opens `ByteStream`; a
  directory of `*.txt` shards opens
  [`ShardStream`][dormouse.train.data.ShardStream] (`open_source`), which reads
  the mount's 6.3 MB shards directly — 227 GB is not worth duplicating into a
  concatenated `.bin` when the drive sustains 146 MB/s against a ~40 KB/s
  training read. The shard order is a SEEDED permutation
  (`_stream_seed(seed, name)`, so two sources do not draw the same order), which
  is the shuffle the burn side was missing: the shards are packed in download
  order, so reading them in order puts one crawl/period in a block and the loss
  spikes at the boundary (`plan_txt_streamer.md`: 1.46 → 8.69 at ~step 25k of a
  joint run). Resume is unaffected — a shard stream is periodic in
  `sum(shard sizes)`, so `skip_bytes` lands on the exact byte an unbroken run
  would have read.

**The plan file** (`mix_plan.toml`, every key shown; `[sources]` paths are
resolved against the plan file's own directory):

```toml
max_epochs = 4.0          # optional, default 4.0; 0 disables the guard
seed = 1                  # optional, default 1 (overridden by the run's --seed)

[[stage]]
until_frac = 0.70         # this stage covers steps [0, round(0.70 * steps))
[stage.weights]
web  = 0.90
code = 0.10

[[stage]]
until_frac = 0.90
[stage.weights]
web = 0.80
code = 0.15
math = 0.05

[[stage]]
until_frac = 1.00         # the LAST stage must reach 1.0
[stage.weights]
web  = 0.62
code = 0.20
math = 0.13
ru   = 0.05

[sources]
web  = "/home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin"
code = "mix/code.bin"
math = "mix/math.bin"
ru   = "mix/ru.bin"
```

**The epoch cap.** Repetition is not free: ≤4 epochs is the burn rule
(2305.16264) and the 2026 domain-level finding is that the optimum tracks the
domain's own loss and rises mildly with model size (2608.14071). A source whose
budget asks for more than `max_epochs` repeats is a LOUD refusal naming the
source and the escape (`--set max_epochs=N`, `0` = off) — a mixture that
silently repeats a 120 MB code slice 40× is the ADR-0019 failure class.

Every refusal here names cause AND escape (ADR-0011/ADR-0019); there are no
fallbacks, so nothing needs a counter.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import torch

from ..instrument import loud
from .data import DEFAULT_RING, ByteStream, ShardStream, open_source, shard_files, shard_sizes

#: The 4-epoch rule (2305.16264) — a source repeated more than this is refused
#: unless the plan (or `--set max_epochs=N`) says otherwise. `0` = off.
DEFAULT_MAX_EPOCHS = 4.0

#: Weights are validated as a probability vector to this tolerance.
WEIGHT_TOL = 1e-6

_MASK64 = (1 << 64) - 1
_PLAN_KEYS = {"stage", "sources", "max_epochs", "seed"}
_STAGE_KEYS = {"until_frac", "weights"}


@dataclass(frozen=True)
class MixStage:
    """One stage: its weights until `until_frac × steps` steps are spent."""

    until_frac: float
    weights: dict[str, float]


@dataclass(frozen=True)
class MixPlan:
    """The staged mixture: sources, stages, the epoch cap and the draw seed."""

    stages: tuple[MixStage, ...]
    sources: dict[str, Path]
    max_epochs: float = DEFAULT_MAX_EPOCHS
    seed: int = 1

    @property
    def names(self) -> tuple[str, ...]:
        """Source names in the canonical draw order (sorted, not file order)."""
        return tuple(sorted(self.sources))

    def weights_at(self, step: int, steps: int) -> dict[str, float]:
        """The weights in force at `step` — a pure function of the step index."""
        return self.stages[stage_index(self, step, steps)].weights


# --- the plan file ----------------------------------------------------------


def loads_mix_plan(text: str, *, base: Path | None = None, where: str = "<string>") -> MixPlan:
    """Parse a `mix_plan.toml`. Unknown keys, bad weights and bad stages are LOUD."""
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        loud(f"mix plan {where} is not TOML: {e}. Escape: fix the file.")
    unknown = sorted(set(raw) - _PLAN_KEYS)
    if unknown:
        loud(
            f"mix plan {where}: unknown key(s) {unknown}. Escape: remove them — known "
            f"keys are {sorted(_PLAN_KEYS)} ([[stage]] with until_frac + [stage.weights], "
            f"[sources], max_epochs, seed)."
        )
    stages = _parse_stages(raw.get("stage"), where)
    sources = _parse_sources(raw.get("sources"), where, base)
    for i, st in enumerate(stages):
        for name in st.weights:
            if name not in sources:
                loud(
                    f"mix plan {where}: stage {i} weights source {name!r} but [sources] "
                    f'has no `{name}` entry. Escape: add `{name} = "<path>.bin"` to '
                    f"[sources], or drop the weight."
                )
    weighted = {name for st in stages for name in st.weights}
    unused = sorted(set(sources) - weighted)
    if unused:
        loud(
            f"mix plan {where}: source(s) {unused} are declared but no stage weights "
            f"them, so they would never be read. Escape: weight them or delete the entry."
        )
    return MixPlan(
        stages=stages,
        sources=sources,
        max_epochs=_as_float(raw.get("max_epochs", DEFAULT_MAX_EPOCHS), where, "max_epochs"),
        seed=_as_int(raw.get("seed", 1), where, "seed"),
    )


def load_mix_plan(path: str | Path) -> MixPlan:
    """Load a `mix_plan.toml` from disk; relative `[sources]` paths resolve beside it."""
    p = Path(path)
    if not p.is_file():
        loud(
            f"mix plan {p} does not exist. Escape: pass a real path (`mix_plan` in the "
            f"train config), or leave `mix_plan` empty to stream the single `data` file."
        )
    return loads_mix_plan(p.read_text(), base=p.parent, where=str(p))


def dumps_mix_plan(plan: MixPlan) -> str:
    """The plan back as TOML — `loads_mix_plan(dumps_mix_plan(p)) == p`."""
    lines = [f"max_epochs = {plan.max_epochs!r}", f"seed = {plan.seed}"]
    for st in plan.stages:
        lines += ["", "[[stage]]", f"until_frac = {st.until_frac!r}", "[stage.weights]"]
        lines += [f"{name} = {json.dumps(w)}" for name, w in sorted(st.weights.items())]
    lines += ["", "[sources]"]
    lines += [f"{name} = {json.dumps(str(path))}" for name, path in sorted(plan.sources.items())]
    return "\n".join(lines) + "\n"


def _as_float(value: object, where: str, what: str) -> float:
    """A number or a LOUD refusal — a config boundary never degrades silently."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        loud(f"mix plan {where}: {what} is not a number: {value!r}. Escape: write a number.")


def _as_int(value: object, where: str, what: str) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        loud(f"mix plan {where}: {what} is not an integer: {value!r}. Escape: write an integer.")


def _parse_stages(raw: object, where: str) -> tuple[MixStage, ...]:
    if not isinstance(raw, list) or not raw:
        loud(
            f"mix plan {where}: no [[stage]] table. Escape: declare at least one stage "
            f"with `until_frac` and `[stage.weights]` (see the module docstring)."
        )
    stages: list[MixStage] = []
    prev = 0.0
    for i, st in enumerate(raw):
        if not isinstance(st, dict):
            loud(f"mix plan {where}: [[stage]] #{i} is not a table")
        unknown = sorted(set(st) - _STAGE_KEYS)
        if unknown:
            loud(f"mix plan {where}: stage {i} has unknown key(s) {unknown} (until_frac, weights)")
        if "until_frac" not in st:
            loud(f"mix plan {where}: stage {i} has no `until_frac`")
        until = _as_float(st["until_frac"], where, f"stage {i} until_frac")
        weights = {
            str(k): _as_float(v, where, f"stage {i} weight {k!r}")
            for k, v in (st.get("weights") or {}).items()
        }
        if not weights:
            loud(f"mix plan {where}: stage {i} has an empty [stage.weights]")
        if any(w < 0 for w in weights.values()):
            loud(f"mix plan {where}: stage {i} has a negative weight: {weights}")
        total = sum(weights.values())
        if abs(total - 1.0) > WEIGHT_TOL:
            loud(
                f"mix plan {where}: stage {i} weights sum to {total!r}, not 1 "
                f"({weights}). Escape: fix the weights — a mixture is a probability "
                f"vector, not a ratio."
            )
        if not 0.0 < until <= 1.0:
            loud(f"mix plan {where}: stage {i} until_frac={until!r} is not in (0, 1]")
        if until <= prev:
            loud(
                f"mix plan {where}: stage {i} until_frac={until!r} does not increase "
                f"(previous {prev!r}). Escape: stages must advance — a stage that "
                f"cannot start is a plan that lies."
            )
        stages.append(MixStage(until, weights))
        prev = until
    if stages[-1].until_frac != 1.0:
        loud(
            f"mix plan {where}: the last stage ends at {stages[-1].until_frac!r}, not "
            f"1.0 — the tail of the run would have no weights. Escape: set the last "
            f"until_frac to 1.0."
        )
    return tuple(stages)


def _parse_sources(raw: object, where: str, base: Path | None) -> dict[str, Path]:
    if not isinstance(raw, dict) or not raw:
        loud(f"mix plan {where}: no [sources] table. Escape: name each source and its .bin")
    out: dict[str, Path] = {}
    for name, value in raw.items():
        p = Path(str(value))
        if base is not None and not p.is_absolute():
            p = base / p
        out[str(name)] = p
    return out


# --- the deterministic schedule ---------------------------------------------


def _mix64(x: int) -> int:
    """splitmix64's finalizer — the repo's hash-shaped determinism, 64 bits wide."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def _fnv1a(data: bytes) -> int:
    """FNV-1a 64 — the repo's stable string hash (the Engram keys' own).

    A source NAME has to be a seed input without Python's `hash()`, which is
    randomized per process (PYTHONHASHSEED): the shard order must be the same
    bytes in every process, or a resume reads a different order than the run.
    """
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & _MASK64
    return h


def _stream_seed(seed: int, name: str) -> int:
    """The shard-order seed of ONE source: the run seed mixed with the source name.

    Two sources in one plan must not draw the same permutation by construction
    (their shards are different crawls of different corpora, and a shared order
    would re-correlate them per epoch).
    """
    return _mix64(seed ^ _fnv1a(name.encode("utf-8")))


def _unit(seed: int, step: int) -> float:
    """A `[0, 1)` draw that is a PURE function of `(seed, step)` — no RNG stream.

    The same doctrine as `train.loop._sample_depth`: a resume jumps to step
    `start` and must see the mixture an unbroken run would have seen, so nothing
    may depend on the draws that came before.
    """
    assert step >= 0, f"step must be >= 0, got {step}"
    h = (seed * 0x9E3779B97F4A7C15 + step * 0xBF58476D1CE4E5B9) & _MASK64
    return (_mix64(h) >> 11) / float(1 << 53)


def stage_bounds(plan: MixPlan, steps: int) -> tuple[int, ...]:
    """Step boundaries of the stages: `len(stages) + 1` values, first 0, last `steps`.

    The boundary is `round(until_frac × steps)` — the number a reader computes
    by hand. Truncation would put S2 at step 6999 for `0.7 × 10000` (IEEE gives
    6999.999999999999), i.e. one step away from the documented fraction.
    """
    if steps < 1:
        loud(f"steps={steps}: a mixture needs a step budget (>= 1)")
    bounds = [0]
    for st in plan.stages:
        bounds.append(round(st.until_frac * steps))
    if bounds[-1] != steps:
        loud(f"mix plan: stage boundaries {bounds} do not end at steps={steps}")
    return tuple(bounds)


def stage_index(plan: MixPlan, step: int, steps: int) -> int:
    """The stage of `step` — a pure function of `(step, steps)`.

    Past the budget the last (decay) stage continues: a run that overshoots its
    own step count does not crash on a stage lookup.
    """
    assert step >= 0, f"step must be >= 0, got {step}"
    bounds = stage_bounds(plan, steps)
    if step >= steps:
        return len(plan.stages) - 1
    for i in range(len(plan.stages)):
        if bounds[i] <= step < bounds[i + 1]:
            return i
    raise AssertionError(f"step {step} is in no stage of {bounds}")


def source_for_step(plan: MixPlan, step: int, steps: int, seed: int | None = None) -> str:
    """The source of `step`: one seeded draw against the stage's cumulative weights."""
    weights = plan.stages[stage_index(plan, step, steps)].weights
    u = _unit(plan.seed if seed is None else seed, step)
    cum = 0.0
    names = sorted(weights)
    for name in names:
        cum += weights[name]
        if u < cum:
            return name
    return names[-1]  # only reachable through fp residue at u ~ 1.0


# --- the epoch cap ----------------------------------------------------------


def source_size(name: str, path: Path) -> int:
    """Bytes on disk, LOUD when the source is missing or empty.

    A DIRECTORY source is a shard directory (`ShardStream`): its size is the sum
    of its `*.txt` shards, read through the same `shard_files`/`shard_sizes`
    calls the stream makes — so the epoch cap and the stream cannot disagree
    about what a source IS.
    """
    if path.is_dir():
        return sum(shard_sizes(shard_files(path)))
    if not path.is_file():
        loud(
            f"mix source {name!r} -> {path} does not exist. Escape: fetch the slices "
            f"(`uv run scripts/fetch_mix_sources.py`), fix the [sources] path, or drop "
            f"the source from the plan."
        )
    size = path.stat().st_size
    if size == 0:
        loud(f"mix source {name!r} -> {path} is empty: a stream with no bytes is not a run")
    return size


def per_source_bytes(plan: MixPlan, steps: int, batch: int, seq_len: int) -> dict[str, int]:
    """Expected bytes per source: `steps_in_stage × window × weight`, summed over stages.

    The draw is i.i.d. per step, so a realized count differs from this by
    `O(sqrt(n))` windows — it is the EXPECTATION the epoch cap is argued from,
    exactly as the research doc's table computes it.
    """
    window = batch * (seq_len + 1)
    bounds = stage_bounds(plan, steps)
    out = {name: 0 for name in plan.sources}
    for i, st in enumerate(plan.stages):
        n = bounds[i + 1] - bounds[i]
        for name, w in st.weights.items():
            out[name] += round(n * window * w)
    return out


def replay_bytes(
    plan: MixPlan,
    *,
    steps: int,
    batch: int,
    seq_len: int,
    upto: int | None = None,
    seed: int | None = None,
) -> dict[str, int]:
    """EXACT bytes each source serves in steps `[0, upto)` — the draw, replayed.

    `steps` is the step budget in force for those steps (on an EXTENDED resume
    it is the checkpoint's stored `steps`, not the new one: the stage schedule
    is a function of the budget, so the new one would assign a different source
    to already-taken steps — review round 3 R3-4), `upto` is how far to replay
    (default: the whole budget). `per_source_bytes` is the expectation the
    epoch cap is argued from; this is what a run with THIS seed actually read,
    and it is what a resume must reproduce (`skip_to_step`).
    """
    window = batch * (seq_len + 1)
    served = dict.fromkeys(plan.sources, 0)
    for step in range(steps if upto is None else upto):
        served[source_for_step(plan, step, steps, seed)] += window
    return served


def check_epoch_cap(
    plan: MixPlan,
    *,
    steps: int,
    batch: int,
    seq_len: int,
    max_epochs: float | None = None,
) -> dict[str, float]:
    """Per-source epochs = bytes_to_consume / source_size; over the cap is LOUD.

    `max_epochs=None` takes the plan's own value (default 4.0); `0` disables the
    guard. Returns the epochs per source so a caller can log them.
    """
    cap = plan.max_epochs if max_epochs is None else max_epochs
    wanted = per_source_bytes(plan, steps, batch, seq_len)
    epochs = {name: wanted[name] / source_size(name, path) for name, path in plan.sources.items()}
    if cap > 0:
        over = sorted((n, e) for n, e in epochs.items() if e > cap)
        if over:
            detail = ", ".join(f"{n!r} {e:.2f}x" for n, e in over)
            loud(
                f"epoch cap: {detail} over max_epochs={cap:g} at steps={steps}, "
                f"batch={batch}, seq_len={seq_len}. Repetition past ~4x degrades a MoE "
                f"(2305.16264) and the 2026 rule is per-domain (2608.14071). Escape: "
                f"--set max_epochs=N (0 = off), or cut steps/batch, or add data for "
                f"the named source(s)."
            )
    return epochs


# --- the stream -------------------------------------------------------------


class MixtureStream:
    """`ByteStream`'s interface over N sources: one whole batch per step, staged.

    A source is a `.bin` FILE or a shard DIRECTORY (`open_source`): the
    directory mode reads the mount's `*.txt` shards in a seeded shuffled order
    and is the reason this class exists at 227 GB, where a concatenated corpus
    would be a 227 GB copy of data already on disk.

    The stage and the source of a step are pure functions of `(seed, step, steps)`
    (see the module docstring), so two instances with one seed read the same bytes
    and `skip_bytes()` reproduces exactly what an unbroken run would have read —
    for a shard source too, because `ShardStream`'s byte sequence is periodic in
    `sum(shard sizes)` and the seed pins the file order. An EXTENDED run
    (`steps` grew, a legal resume) must pass the checkpoint's stored budget to
    `skip_to_step`: the schedule is a function of the budget, so the prefix has
    to be replayed under the budget it was actually read with (review round 3
    R3-4).
    """

    def __init__(
        self,
        plan: MixPlan,
        batch: int,
        seq_len: int,
        steps: int,
        *,
        seed: int | None = None,
        ring_bytes: int = DEFAULT_RING,
    ) -> None:
        if not plan.stages:
            loud("MixtureStream: the plan has no stages")
        self.plan = plan
        self.batch, self.seq_len = batch, seq_len
        self.steps = steps
        self.seed = plan.seed if seed is None else seed
        self.window = batch * (seq_len + 1)
        stage_bounds(plan, steps)  # refuse a plan whose stages do not cover the budget
        self._streams: dict[str, ByteStream | ShardStream] = {
            name: open_source(
                path,
                batch,
                seq_len,
                seed=_stream_seed(self.seed, name),
                ring_bytes=ring_bytes,
            )
            for name, path in plan.sources.items()
        }
        self._step = 0

    @property
    def step(self) -> int:
        """The step index of the NEXT batch."""
        return self._step

    @property
    def stage(self) -> int:
        """The stage index of the NEXT batch."""
        return stage_index(self.plan, self._step, self.steps)

    @property
    def weights(self) -> dict[str, float]:
        """The weights in force for the NEXT batch."""
        return self.plan.weights_at(self._step, self.steps)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """One `[batch, seq_len]` window of inputs and the byte after each."""
        name = source_for_step(self.plan, self._step, self.steps, self.seed)
        self._step += 1
        return self._streams[name].next_batch()

    def rewind(self) -> None:
        """Back to step 0 and every source's start — what an eval needs."""
        for stream in self._streams.values():
            stream.rewind()
        self._step = 0

    def skip_bytes(self, n: int, *, budget: int | None = None) -> None:
        """Fast-forward the whole mixture `n` bytes (resume), per source exactly.

        The trainer passes `start × (seq_len + 1) × batch` bytes. `budget` is the
        step budget the fast-forwarded steps were taken under (see
        `skip_to_step`): the checkpoint's stored `steps` on a resume that
        EXTENDS the run, `None` when the budget did not change.
        """
        if n < 0:
            loud(f"skip_bytes({n}): a negative fast-forward is a rewind")
        if n % self.window:
            loud(
                f"skip_bytes({n}): not a whole number of {self.window}-byte windows "
                f"(batch={self.batch}, seq_len={self.seq_len}). Escape: pass "
                f"steps × (seq_len + 1) × batch."
            )
        self.skip_to_step(n // self.window, budget=budget)

    def skip_to_step(self, step: int, *, budget: int | None = None) -> None:
        """Fast-forward to the STEP INDEX `step`, replaying the schedule the run used.

        The resume position is a function of the steps already taken, not of the
        new run's budget. `steps` is an exempt progress key, so extending a
        mixture run changes the stage boundaries (`stage_bounds(plan, steps)`);
        replaying the prefix under the NEW budget assigns already-taken steps to
        different sources and lands every source's cursor on the wrong byte
        (review round 3 R3-4). `budget` is the budget in force for steps
        `[0, step)` — the checkpoint snapshot's `steps`, which `load_ckpt`
        returns. After the fast-forward `next_batch` uses THIS stream's budget,
        i.e. the extended schedule starts at `step`.

        `budget=None` means "the budget did not change": the current stream's
        own `steps`, which is exact for a plain resume and for any caller that
        has no snapshot.
        """
        if step < 0:
            loud(f"skip_to_step({step}): a negative fast-forward is a rewind")
        schedule = self.steps if budget is None else budget
        served = replay_bytes(
            self.plan,
            steps=schedule,
            upto=step,
            batch=self.batch,
            seq_len=self.seq_len,
            seed=self.seed,
        )
        for name, stream in self._streams.items():
            stream.skip_bytes(served[name])
        self._step = step

    def close(self) -> None:
        for stream in self._streams.values():
            stream.close()


def open_mixture(
    plan_path: str | Path,
    *,
    steps: int,
    batch: int,
    seq_len: int,
    seed: int | None = None,
    max_epochs: float | None = None,
    ring_bytes: int = DEFAULT_RING,
) -> MixtureStream:
    """Load a plan, run the epoch-cap guard, and build the stream — one call.

    This is what the trainer's data seam calls when `mix_plan` is set:
    `MixtureStream(load_mix_plan(p), batch, seq_len, steps)` plus the refusal,
    which is the part a caller must not be able to forget.
    """
    plan = load_mix_plan(plan_path)
    check_epoch_cap(plan, steps=steps, batch=batch, seq_len=seq_len, max_epochs=max_epochs)
    return MixtureStream(plan, batch, seq_len, steps, seed=seed, ring_bytes=ring_bytes)
