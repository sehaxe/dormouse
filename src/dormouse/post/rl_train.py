"""The RL loop: one prompt per step, G rollouts, one policy-gradient step.

`post/rl.py` is the pure math — the MiMo reward, the two advantage arms, the
aggregation, the step — and this module is the trainer around it: the piece that
turns a checkpoint into a policy that solves tasks in fewer bytes. The loop's
shape is `train/sft.py`'s, copied rather than reinvented: the same NaN firewall
(the loss AND the device gradient norm decide the step; a non-finite one touches
no weight, is COUNTED `nan_skipped`, and two consecutive all-non-finite windows
stop the run with a `LoudError`), the same log/JSON cadence, the same
`save_ckpt`/`load_ckpt` pair, and the same `--init-from` path (`loop.init_cfg` +
`load_model_state`, which is what makes a host-table pretrain checkpoint a legal
init). The firewall's two halves are `train.sft.firewall_loss`/`gate_grads`, so
this loop cannot drift from SFT's; `train/loop.py` keeps its own device-flag
form because a CUDA-graph capture reads it.

**What is different, and why.** SFT reads a corpus; this lane GENERATES its
data. The prompts come from either the JSONL corpus `scripts/build_rl_tasks.py`
emits (`rl_corpus`) or, when that is empty, from `build_tasks(...)` at
`seed + SFT_TASK_OFFSET` — the offset is what keeps the generated prompts off
the SFT corpus, which builds from the SAME seed. The corpus is split once, by a
seeded shuffle, into a training pool and a HELD-OUT half; the split is proved
disjoint by rendered-prompt sha256 and printed, and the held-out half is never
drawn by the loop. So there is no `ByteStream` and no `--data`, but there IS an
eval line: `pass@1` (one greedy attempt) and `pass@budget` (any of
`rl_eval_samples` sampled attempts) on the held-out tasks, every
`rl_eval_every` steps plus before/after. A resume has no byte position to
fast-forward, but it re-draws the same tasks because every draw is a pure
function of `(seed, step, attempt)`.

**The difficulty band.** A step does not take the first prompt it draws: it
draws candidate prompts until the group's pass rate is in
`[rl_band_lo, rl_band_hi]` (default `[0.2, 0.8]`). A group below the band is
`rl_dropped_low` (too hard — no signal), above it `rl_dropped_high` (too easy —
consensus, no signal); after `rl_band_attempts` the last candidate is trained
anyway and `rl_band_exhausted` says the curriculum ran out. That fallback is the
difference between a curriculum and a filter that silently starves the step.

**The length penalty.** `rl_len_penalty` is ON by default at the value the
measurement in `scripts/measure_rl_length.py` supports: on 200 held-out math
tasks the longest CORRECT answer is 1.364x the median correct length (15 B vs
11 B), so at the ramp's `s = 0.5` its value is 0.727 and X = 0.1 costs it 7.3 %
of a passing reward — the largest X that stays inside the stated 10 % budget
(X = 0.2 would cost 14.5 %). The model itself passed 0/200 at d = 128 on CPU,
so the correct set is the reference answers, each verified; the incorrect set is
the model's own 200 greedy rollouts (median 39 B). The term is still gated and
COUNTED when the gate is shut.

**The purity contract.** Everything the loop draws is a pure function of
`(seed, step, sample)` — no module-level RNG, because burn's lesson (ADR-0021,
the DSpark stride defect) is that an unseeded stream makes two runs of one
config different runs and a resume a different objective:

* the task index is `random.Random(seed·1_000_003 + step)` — one draw per
  step, plus a deterministic `attempt` key for the difficulty band's
  candidates;
* each rollout gets a FRESH `torch.Generator` seeded by `(seed, step, sample)`,
  so rollout `g`'s bytes do not depend on how many bytes the earlier rollouts
  happened to emit before the budget stopped them (a shared stream would make a
  one-byte change anywhere reshuffle every later rollout).

The model's own init is not seeded by THIS function — a `RunConfig` passed
straight to `rl_train()` runs under the caller's RNG, exactly as `train()` and
`sft_train()` do. The CLI entry points seed `cfg.seed` before the build
(`cli._model` / `cli._seed_model`), so `dormouse rl` and `dormouse train` with
one seed produce one init; a resume overrides the file's value by loading the
checkpoint's weights.

**The counters.** `rl_steps` (one per `rl_step` call) and `rl_samples` (one per
sample that contributed a log-prob) are the loop's own evidence; the skips
(`rl_skipped_no_advantage`, `rl_skipped_empty_response`), the gated length
penalty (`rl_len_penalty_gated`/`_active`) and the non-causal-MoE mark
(`rl_non_causal_logp`) come from `rl.py` itself. The band adds
`rl_dropped_low`/`rl_dropped_high`/`rl_band_draws`/`rl_band_exhausted` and
`rl_zero_spread_groups` (a verdict-consensus group, whose group-normalised
advantages are exactly 0 — the counter is the evidence the loop saw one rather
than the group vanishing); the eval adds `rl_eval_calls`/`rl_eval_tasks`. `loss.backward()` and
`opt.step()` run ONLY when `rl_samples > 0`: `rl_step` returns a grad-carrying
zero when nothing contributed, and an AdamW step on zero gradients still applies
weight decay.

**The host-table path (`engram_ram`), as in SFT.** A `--init-from` checkpoint
from an `engram_ram` run stores its memory OUTSIDE the model half, so
`load_model_state` leaves the model on host tables; this loop then owns that
substrate the way `sft_train` does (enable before the optimizer exists, one
`host_step_all` per training step, `drop_pending` on a skipped step). Without it
the memory would be read every forward and never trained — the SILENT
degradation `sft.py`'s docstring names.

**Named deviation.** The LR is CONSTANT at `cfg.lr` (the WSD schedule is a
function of the tokens a run has read, and this lane reads none). A scheduled
arm belongs with an A/B, not with the wiring.
"""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

import torch

from ..config import RunConfig, rl_config, validate_train
from ..instrument import Counters, LoudError, loud
from ..model import DormouseModel
from ..model.engram import enable_host_tables, host_engrams, host_step_all
from ..model.qat_cache import invalidate_qat_cache_all
from ..train.loop import kda_field, load_ckpt, load_model_state, save_ckpt, unpack_state_dict
from ..train.optim import MixOptimizer, clip_grads
from .rl import RLConfig, Sample, group_rewards, group_stats, reward, rl_advantages, rl_step
from .rollouts import Rollout, rollout
from .tasks import SFT_TASK_OFFSET, Task, build_tasks, load_rl_tasks

__all__ = [
    "difficulty_side",
    "evaluate_holdout",
    "rl_draw_group",
    "rl_generator",
    "rl_task_index",
    "rl_train",
    "split_tasks",
    "thrift_metrics",
]


def rl_task_index(seed: int, step: int, n: int, attempt: int = 0) -> int:
    """Which of the `n` tasks a step draws — a pure function of `(seed, step, attempt)`.

    `random.Random` keyed by the tuple, never a module-level stream: the task
    schedule is then reproducible from the run's seed and the step index alone
    (see the module docstring's purity contract). `attempt` is the difficulty
    band's candidate number; `attempt=0` is the historical single draw, so a
    run with the band disabled keeps the schedule it had.
    """
    assert n > 0, "rl_task_index needs at least one task to draw from"
    assert attempt >= 0, "rl_task_index attempt counts candidates from 0"
    key = seed * 1_000_003 + step
    if attempt:
        key = key * 1_009 + attempt
    return random.Random(key).randrange(n)


def _prompt_hash(task: Task) -> str:
    """SHA-256 of the RENDERED prompt — the identity the model and the verifier see."""
    return hashlib.sha256(task.prompt.encode("utf-8")).hexdigest()


def _hash_set(hashes: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(hashes)).encode("utf-8")).hexdigest()[:16]


def split_tasks(
    tasks: list[Task], seed: int, holdout: int
) -> tuple[list[Task], list[Task], dict]:
    """A deterministic train/holdout split, proved disjoint by PROMPT hash.

    A seeded shuffle decides which tasks are held out, so the split is a pure
    function of `(tasks, seed, holdout)` and a resume lands on the same one. The
    proof is the point: the returned report carries both hash sets and
    `disjoint`, and a collision is a `LoudError` rather than a number nobody
    audits. Ids are not used — the project has paid for that already.
    """
    if holdout < 0:
        loud(f"split_tasks holdout={holdout}: a held-out count cannot be negative.")
    if holdout >= len(tasks):
        loud(
            f"split_tasks holdout={holdout} of {len(tasks)} tasks: the training pool "
            f"would be empty. Escape: a holdout smaller than the corpus."
        )
    order = list(range(len(tasks)))
    random.Random(seed * 3 + 17).shuffle(order)
    held = [tasks[i] for i in order[:holdout]]
    train = [tasks[i] for i in order[holdout:]]
    train_hashes = {_prompt_hash(t) for t in train}
    held_hashes = {_prompt_hash(t) for t in held}
    overlap = train_hashes & held_hashes
    if overlap:
        loud(
            f"split_tasks: {len(overlap)} rendered prompt(s) are in BOTH the train and "
            f"the held-out split. A split by index cannot leak; this means the corpus "
            f"itself repeats a prompt. Escape: rebuild the corpus."
        )
    report = {
        "train": len(train),
        "holdout": len(held),
        "train_sha256": _hash_set(train_hashes),
        "holdout_sha256": _hash_set(held_hashes),
        "disjoint": True,
    }
    return train, held, report


def difficulty_side(pass_rate: float, lo: float, hi: float) -> str:
    """`"keep"` inside the band, else the side the group is on.

    The band is INCLUSIVE at both ends (`[lo, hi]`), and the sides are named
    `"low"`/`"high"` so the counters say which way a prompt was dropped: an
    all-fail group is too hard (low pass rate), an all-pass group is too easy.
    A curriculum that only counted "dropped" could not tell a starving batch
    from a saturated one.
    """
    if not 0.0 <= lo <= hi <= 1.0:
        loud(f"difficulty_side band=[{lo}, {hi}]: a pass rate lies in [0, 1] and lo <= hi.")
    if not 0.0 <= pass_rate <= 1.0:
        loud(f"difficulty_side pass_rate={pass_rate}: a pass rate lies in [0, 1].")
    if pass_rate < lo:
        return "low"
    if pass_rate > hi:
        return "high"
    return "keep"


def rl_draw_group(
    model,
    tasks: list[Task],
    cfg,
    step: int,
    device: torch.device,
    counters: Counters,
) -> tuple[Task, list[Rollout]]:
    """Draw candidate prompts until one group's pass rate is in the band.

    The candidate draw is `rl_task_index(seed, step, n, attempt)`, so a resume
    re-draws the same prompts. A group outside `[rl_band_lo, rl_band_hi]` is
    COUNTED (`rl_dropped_low` / `rl_dropped_high`) and thrown away. After
    `rl_band_attempts` the LAST candidate is trained anyway and
    `rl_band_exhausted` says the curriculum ran out of draws: a fallback that
    cannot silently starve the step, which is the failure mode of a filter that
    just skips out-of-band prompts.

    A verdict-consensus group (`pass_rate` 0 or 1) is counted
    `rl_zero_spread_groups` here, before the band decides: it is the group the
    group-normalised advantage cannot learn from (its advantages are exactly 0),
    and the counter is the evidence that the loop saw one.
    """
    if not tasks:
        loud("rl_draw_group: no tasks to draw from.")
    last: tuple[Task, list[Rollout]] | None = None
    for attempt in range(cfg.rl_band_attempts):
        task = tasks[rl_task_index(cfg.seed, step, len(tasks), attempt)]
        group: list[Rollout] = [
            rollout(
                model,
                task,
                cfg.rl_budget,
                device=device,
                generator=rl_generator(
                    cfg.seed, step, attempt * cfg.rl_group + g, device
                ),
            )
            for g in range(cfg.rl_group)
        ]
        counters.count("rl_band_draws")
        pass_rate = sum(1 for r in group if r.passed) / len(group)
        if pass_rate == 0.0 or pass_rate == 1.0:
            counters.count("rl_zero_spread_groups")
        side = difficulty_side(pass_rate, cfg.rl_band_lo, cfg.rl_band_hi)
        if side == "keep":
            return task, group
        counters.count(f"rl_dropped_{side}")
        last = (task, group)
    assert last is not None, "rl_band_attempts must be >= 1"
    counters.count("rl_band_exhausted")
    return last


def evaluate_holdout(
    model,
    tasks: list[Task],
    cfg,
    device: torch.device,
    counters: Counters | None = None,
) -> dict | None:
    """The held-out measurement: `pass@1` (greedy) and `pass@budget` (G samples).

    `pass@1` is the deployment reading: ONE greedy attempt per task, the byte a
    user would get. `pass@budget` is the training-distribution reading: the
    task is solved if ANY of `rl_eval_samples` temperature-sampled attempts
    verifies inside the same byte budget — the "is the policy capable at all"
    number, which moves before greedy does. Both count the verifier's own
    verdicts; nothing here is a proxy.

    The eval tasks are the split's held-out half and are NEVER drawn by
    `rl_draw_group` (the split's hash proof is in the log).
    """
    if not tasks:
        return None
    counters = Counters() if counters is None else counters
    passed_1 = 0
    passed_budget = 0
    for j, task in enumerate(tasks):
        first = rollout(model, task, cfg.rl_budget, greedy=True, device=device)
        passed_1 += int(first.passed)
        any_pass = first.passed
        for sample in range(cfg.rl_eval_samples):
            attempt = rollout(
                model,
                task,
                cfg.rl_budget,
                device=device,
                generator=rl_generator(cfg.seed, 1_000_000 + j, sample, device),
            )
            any_pass = any_pass or attempt.passed
        passed_budget += int(any_pass)
    counters.count("rl_eval_calls")
    counters.count("rl_eval_tasks", len(tasks))
    return {
        "n": len(tasks),
        "pass@1": passed_1 / len(tasks),
        "pass@budget": passed_budget / len(tasks),
    }


def _source_tasks(run: RunConfig) -> tuple[list[Task], str]:
    """The corpus: the JSONL when one is named, else `build_tasks` + the SFT offset."""
    cfg = run.train
    if cfg.rl_corpus:
        return load_rl_tasks(cfg.rl_corpus), cfg.rl_corpus
    seed = cfg.seed + SFT_TASK_OFFSET
    return (
        build_tasks(cfg.rl_task_cls, seed=seed, n=cfg.rl_tasks),
        f"build_tasks({cfg.rl_task_cls!r}, seed={seed})",
    )


def rl_generator(seed: int, step: int, sample: int, device: torch.device) -> torch.Generator:
    """A FRESH rollout generator seeded by `(seed, step, sample)`.

    Fresh per `(step, sample)` and not one stream advanced `G` times: a rollout
    stops at a different byte count depending on what it generated, so a shared
    stream would couple every later rollout to the earlier ones' lengths. The
    device follows the model's (`multinomial` needs a generator on its own
    device).
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed * 2_000_003 + step * 1_007 + sample)
    return gen


def thrift_metrics(n: int, passed: int, solved_bytes: int) -> tuple[float, float | None]:
    """`(pass rate, mean bytes per SOLVED rollout)` — the owner's two metrics.

    `n` rollouts, `passed` of them verified, `solved_bytes` their summed byte
    cost (`Rollout.n_bytes`: prompt + response + 2 B per generation stroke). The
    mean is over the SOLVED ones — a failed attempt's bytes are not the price of
    a solution — and it is `None`, never 0.0, when nothing passed: a mean over an
    empty set is not a number, and 0.0 would read as "solutions are free".
    """
    if n <= 0:
        return 0.0, None
    return passed / n, (solved_bytes / passed if passed else None)


def _metrics_field(bytes_per_solved: float | None) -> str:
    """`bytes/solved=` for the log line: a number, or `-` when nothing passed."""
    return "-" if bytes_per_solved is None else f"{bytes_per_solved:.1f}"


def rl_train(
    run: RunConfig,
    *,
    device: torch.device | str | None = None,
    log=print,
    resume: bool = False,
    init_from: str | None = None,
) -> dict:
    """The RL loop. Builds the model from `run.model` and returns a report dict.

    `device` names where the model lives (a rollout is a decode forward, so it
    must be the same device). `init_from` loads WEIGHTS from a foreign
    checkpoint through the ONE loader (`load_model_state`), with no config gate
    and a fresh optimizer — the deliberate difference from `resume`, which
    continues THIS run and must prove the config did not move (ADR-0021).
    """
    cfg = run.train
    torch.set_float32_matmul_precision("high" if cfg.tf32 else "highest")
    # `train.sft` imports `post.tasks`, so a top-level import from this module
    # is a cycle (the same reason `init_cfg` is imported locally below).
    from ..train.sft import firewall_loss, gate_grads, refuse_unwired

    validate_train(cfg)
    refuse_unwired(cfg, "rl")
    if cfg.grad_accum != 1:
        loud(
            f"rl: train.grad_accum={cfg.grad_accum}: the RL step is ONE prompt "
            f"group — G rollouts, one policy-gradient backward, one advantage "
            f"normalisation inside that group. Accumulating k groups into one "
            f"optimizer step would average different groups' normalised "
            f"advantages under one LR, an arm this loop has never measured. "
            f"Escape: grad_accum=1 (the default), or run `dormouse train` for "
            f"accumulation."
        )
    rl_cfg: RLConfig = rl_config(cfg)
    dev = torch.device("cpu") if device is None else torch.device(device)
    model = DormouseModel(run.model).to(dev)
    model.train()
    if cfg.engram_ram:
        # The host path's substrate, switched BEFORE the optimizer exists (the
        # rule `sft_train` carries): `Engram.to_host` DELETES the table
        # parameter, so a group built first would hold a dead tensor.
        enable_host_tables(model, bias_correction=cfg.engram_bias_correction)
        if not host_engrams(model):
            loud(
                "rl engram_ram=true but the model has no Engram to host "
                "(use_engram=false?): the flag would host nothing while the run "
                "reported it on. Escape: use_engram=true, or engram_ram=false."
            )
        log(
            f"engram_ram: {len(host_engrams(model))} table set(s) in host RAM, VRAM "
            f"holds the batch's unique rows, "
            f"bias_correction={'on' if cfg.engram_bias_correction else 'off'}"
        )
    opt = MixOptimizer(
        model,
        lr=cfg.lr,
        wd=cfg.wd,
        opt=cfg.opt,
        ns_steps=cfg.ns_steps,
        ns_kind=cfg.ns_kind,
        colrow_3d=cfg.muon_colrow_3d,
        # The fields round-1 ledger m5 found silently unwired. The two precision
        # rungs and the SF weight-decay value are forwarded (pure optimizer
        # settings); the two direction arms are REFUSED above, so a reported arm
        # always ran.
        muon_state_dtype=cfg.muon_state_dtype,
        ns_dtype=cfg.ns_dtype,
        muon_variant=cfg.muon_variant,
        muon_sf=cfg.muon_sf,
        muon_sf_wd=cfg.muon_sf_wd,
    )
    if cfg.optim_8bit:
        opt.enable_bnb8bit()
    start = 0
    if init_from is not None:
        if resume:
            loud(
                "rl: --init-from and --resume together name two different sources "
                "of weights. Escape: --init-from to start a new RL run from a "
                "pretrain/SFT checkpoint, or --resume to continue an RL run."
            )
        path = Path(init_from)
        if not path.is_file():
            loud(f"rl --init-from: {path} does not exist. Escape: the pretrain run's .pt path.")
        foreign = torch.load(path, map_location="cpu", weights_only=False)
        sd = unpack_state_dict(foreign.get("model", foreign), foreign.get("model_quant_rung"))
        # `_init_cfg` describes the INIT FILE (its substrate and sidecar path),
        # not this run — the loader's own contract. Imported here, not at module
        # scope: `train.sft` imports `post.tasks`, so a top-level import of the
        # SFT module from inside the post package is a cycle.
        from ..train.loop import init_cfg as _init_cfg

        load_model_state(model, _init_cfg(cfg, path, sd, model), sd, strict=True)
        log(f"rl init: weights from {path} (step {foreign.get('step', '?')}), fresh optimizer")
    if resume:
        ck, _stored = load_ckpt(run)
        load_model_state(model, cfg, ck["model"])
        opt.load_state_dict(ck["opt"])
        start = int(ck["step"]) + 1
        log(f"rl resume: step={start} opt_steps={opt.steps}")

    tasks, source = _source_tasks(run)
    train_tasks, eval_tasks, split = split_tasks(tasks, cfg.seed, cfg.rl_holdout)
    # The leakage proof, PRINTED: the two halves are disjoint by rendered-prompt
    # sha256, not by index and not by id. A reader can re-derive both digests
    # from the corpus and check they differ.
    log(
        f"rl tasks: source={source} total={len(tasks)} train={split['train']} "
        f"holdout={split['holdout']} train_sha256={split['train_sha256']} "
        f"holdout_sha256={split['holdout_sha256']} disjoint={split['disjoint']} "
        f"(prompt sha256, sorted at printed length 16)"
    )
    total = Counters()
    window = Counters()
    nan_skipped = 0
    drowning = 0
    window_finite: list[bool] = []
    last_loss = float("nan")
    run_n = run_passed = run_bytes = 0
    win_n = win_passed = win_bytes = 0
    adv_sum = adv_n = 0
    eval_history: list[dict] = []
    last_eval: dict | None = None
    sink = None
    if cfg.json_log:
        Path(cfg.json_log).parent.mkdir(parents=True, exist_ok=True)
        sink = open(cfg.json_log, "a", encoding="utf-8")

    def _row(**kw) -> None:
        if sink is None:
            return
        import json

        sink.write(json.dumps({k: v for k, v in kw.items() if v is not None}, default=str) + "\n")
        sink.flush()

    _row(
        event="rl_run",
        source=run.source,
        steps=cfg.steps,
        lr=cfg.lr,
        opt=cfg.opt,
        task_cls=cfg.rl_task_cls,
        corpus=source,
        tasks=len(tasks),
        train_tasks=split["train"],
        holdout=split["holdout"],
        group=cfg.rl_group,
        budget=cfg.rl_budget,
        adv=cfg.rl_adv,
        band=[cfg.rl_band_lo, cfg.rl_band_hi],
        len_penalty=cfg.rl_len_penalty,
        resumed=start,
    )

    def _eval(step: int) -> None:
        """Measure the held-out half and put the numbers on the log + JSON sink."""
        nonlocal last_eval
        if not eval_tasks:
            return
        result = evaluate_holdout(model, eval_tasks, cfg, dev, total)
        assert result is not None  # `eval_tasks` is non-empty
        result["step"] = step
        last_eval = result
        eval_history.append(result)
        log(
            f"rl eval step={step} holdout={result['n']} "
            f"pass@1={result['pass@1']:.3f} pass@budget={result['pass@budget']:.3f}"
        )
        _row(event="rl_eval", **result)

    if eval_tasks:
        # "Before": the policy the run starts from. A resume measures the
        # checkpoint it loaded, which is the same statement for this segment.
        _eval(start)

    for step in range(start, cfg.steps):
        step_counters = Counters()
        # The difficulty band owns the draw: out-of-band candidate groups are
        # counted and thrown away (bounded, and the last one trains if every
        # candidate is out of band — no silent starvation).
        task, rollouts = rl_draw_group(model, train_tasks, cfg, step, dev, step_counters)
        samples = [Sample.from_rollout(task, r, 0) for r in rollouts]
        verdicts = [r.verdict for r in rollouts]
        lengths = [len(s.response) for s in samples]
        if rl_cfg.length_penalty > 0.0:
            # The term is group-relative, so it needs ONE `group_stats` (ℓ*_q +
            # the gate) fed to every item's reward — the same numbers
            # `group_rewards` computes internally, hoisted so the quantile is not
            # computed twice for a window of G. The item counters
            # (`rl_len_penalty_gated`/`_active`) ride the step's Counters, so they
            # reach the log line AND the report, like `rl_step`'s own.
            stats = group_stats(verdicts, lengths, rl_cfg)
            rewards = [
                reward(
                    v,
                    task.solution,
                    rl_cfg,
                    length=n,
                    stats=stats,
                    counters=step_counters,
                )
                for v, n in zip(verdicts, lengths, strict=True)
            ]
        else:
            rewards = group_rewards(verdicts, [task.solution] * len(samples), lengths, rl_cfg)
        advantages = rl_advantages(
            [r.total for r in rewards],
            [s.group for s in samples],
            cfg.rl_adv,
            samples=samples,
        )
        loss, step_counters = rl_step(
            model, opt, samples, advantages, rl_cfg, counters=step_counters
        )
        total.merge(step_counters)
        window.merge(step_counters)
        kept = step_counters.get("rl_samples", 0)
        for r in rollouts:
            win_n += 1
            if r.passed:
                win_passed += 1
                win_bytes += r.n_bytes
        adv_sum += sum(abs(a.value) for a in advantages)
        adv_n += len(advantages)

        # The firewall `train()`/`sft_train()` carry: ONE host read per step.
        # The read is AFTER the backward because the step's decision is the AND
        # of the loss flag and the device gradient norm — a finite loss with a
        # non-finite gradient must skip too (round-1 ledger m3), or `exp_avg_sq`
        # is poisoned permanently.
        lf = float(loss.detach())
        if kept > 0:
            step_loss, loss_finite = firewall_loss(loss)
            step_loss.backward()
            finite = bool(gate_grads(model.parameters(), loss_finite))
        else:
            finite = math.isfinite(lf)
        window_finite.append(finite)
        if finite and kept > 0:
            if cfg.grad_clip > 0:
                clip_grads(model.parameters(), cfg.grad_clip)
            opt.step()
            # The same per-step QAT-cache invalidation `train()`/`sft_train()`
            # run: a graph replay moves a parameter's value without bumping
            # `_version`, and the projection cache must not outlive the step.
            invalidate_qat_cache_all(model)
            if cfg.engram_ram:
                # Once per TRAINING step, after the optimizer — the same rule and
                # counters as `train()`/`sft_train()`.
                updates, rows = host_step_all(model, cfg.lr)
                total.count("engram_host_updates", updates)
                window.count("engram_host_updates", updates)
                total.count("engram_host_update_rows", rows)
                window.count("engram_host_update_rows", rows)
            last_loss = lf
        elif not finite:
            nan_skipped += 1
            total.count("nan_skipped")
            window.count("nan_skipped")
            opt.zero_grad(set_to_none=True)
            for m in host_engrams(model):
                m.drop_pending()
        else:
            # Every sample was skipped (no advantage, or an empty response), so
            # the loss is the grad-carrying zero `rl_step` returns: a backward
            # would build zero gradients and an AdamW step on those still applies
            # weight decay — neither runs (`rl.py`'s own contract). The loss is
            # still FINITE, so it is reported; the counters are what say the step
            # trained nothing (`rl_samples=0`).
            last_loss = lf
            opt.zero_grad(set_to_none=True)
            for m in host_engrams(model):
                m.drop_pending()

        if (step + 1) % cfg.log_every == 0:
            window.drain()
            total.drain()
            if window_finite and not any(window_finite):
                drowning += 1
            elif window_finite:
                drowning = 0
            window_finite = []
            pass_rate, bytes_per_solved = thrift_metrics(win_n, win_passed, win_bytes)
            adv = adv_sum / adv_n if adv_n else float("nan")
            log(
                " ".join(
                    [
                        f"rl step={step + 1}",
                        f"loss={last_loss:.4f}",
                        f"pass={pass_rate:.2f}",
                        f"bytes/solved={_metrics_field(bytes_per_solved)}",
                        f"adv={adv:.4f}",
                        f"rl_samples={window.get('rl_samples', 0)}",
                        f"nan_skipped={nan_skipped}",
                        kda_field(window),
                        window.line(),
                    ]
                )
            )
            if sink is not None:
                _row(
                    event="rl_log",
                    step=step + 1,
                    loss=last_loss,
                    pass_rate=pass_rate,
                    bytes_per_solved=bytes_per_solved,
                    adv=adv,
                    counters=dict(window),
                    nan_skipped=nan_skipped,
                )
            run_n += win_n
            run_passed += win_passed
            run_bytes += win_bytes
            win_n = win_passed = win_bytes = 0
            adv_sum = adv_n = 0
            window = Counters()
            if drowning >= 2:
                raise LoudError(
                    f"two consecutive windows of {cfg.log_every} RL steps were "
                    f"entirely non-finite (steps {step + 1 - 2 * cfg.log_every + 1}"
                    f"..{step + 1}); nan_skipped={nan_skipped}. The same rule "
                    f"`train()` carries: a drowning run must stop for a human, not "
                    f"exit 0. Escape: lower lr, or check the reward's inputs."
                )

        if cfg.rl_eval_every > 0 and eval_tasks and (step + 1) % cfg.rl_eval_every == 0:
            _eval(step + 1)

        if (step + 1) % cfg.ckpt_every == 0:
            weights, snapshot = save_ckpt(run, model, opt, step)
            log(f"ckpt step={step + 1} -> {weights} ({snapshot})")
            if sink is not None:
                _row(event="rl_ckpt", step=step + 1, path=str(weights))

    # "After": the last step always gets a held-out reading, even when the step
    # count is not a multiple of `rl_eval_every`.
    if eval_tasks and (not eval_history or eval_history[-1]["step"] != cfg.steps):
        _eval(cfg.steps)
    if sink is not None:
        sink.close()
    # The steps after the last log cadence: their rollouts count in the report
    # (the window's own numbers are reset at each cadence, so this adds them once).
    run_n += win_n
    run_passed += win_passed
    run_bytes += win_bytes
    run_pass_rate, run_bytes_per_solved = thrift_metrics(run_n, run_passed, run_bytes)
    return {
        "steps": cfg.steps,
        "loss": last_loss,
        "nan_skipped": nan_skipped,
        "pass_rate": run_pass_rate,
        "bytes_per_solved": run_bytes_per_solved,
        "rl_samples": total.get("rl_samples", 0),
        "counters": total,
        "opt_steps": opt.steps,
        "tasks": source,
        "train_tasks": len(train_tasks),
        "holdout_tasks": len(eval_tasks),
        "split": split,
        "eval": last_eval,
        "eval_history": eval_history,
    }
