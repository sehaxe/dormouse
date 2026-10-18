"""The RL loop: one prompt per step, G rollouts, one policy-gradient step.

`post/rl.py` is the pure math — the MiMo reward, the two advantage arms, the
aggregation, the step — and this module is the trainer around it: the piece that
turns a checkpoint into a policy that solves tasks in fewer bytes. The loop's
shape is `train/sft.py`'s, copied rather than reinvented: the same NaN firewall
(a non-finite step touches no weight, is COUNTED `nan_skipped`, and two
consecutive all-non-finite windows stop the run with a `LoudError`), the same
log/JSON cadence, the same `save_ckpt`/`load_ckpt` pair, and the same
`--init-from` path (`sft._init_cfg` + `load_model_state`, which is what makes a
host-table pretrain checkpoint a legal init).

**What is different, and why.** SFT reads a corpus; this lane GENERATES its
data. Every step draws ONE task from `build_tasks(cfg.rl_task_cls, seed,
n=cfg.rl_tasks)`, rolls out `G = cfg.rl_group` independent attempts under the
byte budget, verifies them, and takes one policy-gradient step on the group. So
there is no `ByteStream`, no `--data`, no eval line (the tasks are generated —
there is no held-out set, and the rollout's own pass rate is the honest
measurement), and a resume has no byte position to fast-forward.

**The purity contract.** Everything the loop draws is a pure function of
`(seed, step, sample)` — no module-level RNG, because burn's lesson (ADR-0021,
the DSpark stride defect) is that an unseeded stream makes two runs of one
config different runs and a resume a different objective:

* the task index is `random.Random(seed·1_000_003 + step)` — one draw per step;
* each rollout gets a FRESH `torch.Generator` seeded by `(seed, step, sample)`,
  so rollout `g`'s bytes do not depend on how many bytes the earlier rollouts
  happened to emit before the budget stopped them (a shared stream would make a
  one-byte change anywhere reshuffle every later rollout).

The model's own init is not seeded here — `train()` and `sft_train()` do not
seed it either; a caller that wants two identical runs seeds torch first, which
is what the determinism gate does.

**The counters.** `rl_steps` (one per `rl_step` call) and `rl_samples` (one per
sample that contributed a log-prob) are the loop's own evidence; the skips
(`rl_skipped_no_advantage`, `rl_skipped_empty_response`), the gated length
penalty (`rl_len_penalty_gated`/`_active`) and the non-causal-MoE mark
(`rl_non_causal_logp`) come from `rl.py` itself. `loss.backward()` and
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

import math
import random
from pathlib import Path

import torch

from ..config import RunConfig, rl_config
from ..instrument import Counters, LoudError, loud
from ..model import DormouseModel
from ..model.engram import enable_host_tables, host_engrams, host_step_all
from ..train.loop import kda_field, load_ckpt, load_model_state, save_ckpt, unpack_state_dict
from ..train.optim import MixOptimizer, clip_grads
from .rl import RLConfig, Sample, group_rewards, group_stats, reward, rl_advantages, rl_step
from .rollouts import Rollout, rollout
from .tasks import build_tasks

__all__ = ["rl_generator", "rl_task_index", "rl_train", "thrift_metrics"]


def rl_task_index(seed: int, step: int, n: int) -> int:
    """Which of the `n` tasks a step draws — a pure function of `(seed, step)`.

    `random.Random` keyed by the pair, never a module-level stream: the task
    schedule is then reproducible from the run's seed and the step index alone
    (see the module docstring's purity contract).
    """
    assert n > 0, "rl_task_index needs at least one task to draw from"
    return random.Random(seed * 1_000_003 + step).randrange(n)


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
    rl_cfg: RLConfig = rl_config(cfg)
    dev = torch.device("cpu") if device is None else torch.device(device)
    model = DormouseModel(run.model).to(dev)
    model.train()
    if cfg.engram_ram:
        # The host path's substrate, switched BEFORE the optimizer exists (the
        # rule `sft_train` carries): `Engram.to_host` DELETES the table
        # parameter, so a group built first would hold a dead tensor.
        enable_host_tables(model)
        if not host_engrams(model):
            loud(
                "rl engram_ram=true but the model has no Engram to host "
                "(use_engram=false?): the flag would host nothing while the run "
                "reported it on. Escape: use_engram=true, or engram_ram=false."
            )
        log(
            f"engram_ram: {len(host_engrams(model))} table set(s) in host RAM, VRAM "
            f"holds the batch's unique rows"
        )
    opt = MixOptimizer(
        model,
        lr=cfg.lr,
        wd=cfg.wd,
        opt=cfg.opt,
        ns_steps=cfg.ns_steps,
        ns_kind=cfg.ns_kind,
        colrow_3d=cfg.muon_colrow_3d,
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
        from ..train.sft import _init_cfg

        load_model_state(model, _init_cfg(cfg, path, sd, model), sd, strict=True)
        log(f"rl init: weights from {path} (step {foreign.get('step', '?')}), fresh optimizer")
    if resume:
        ck, _stored = load_ckpt(run)
        load_model_state(model, cfg, ck["model"])
        opt.load_state_dict(ck["opt"])
        start = int(ck["step"]) + 1
        log(f"rl resume: step={start} opt_steps={opt.steps}")

    tasks = build_tasks(cfg.rl_task_cls, seed=cfg.seed, n=cfg.rl_tasks)
    total = Counters()
    window = Counters()
    nan_skipped = 0
    drowning = 0
    window_finite: list[bool] = []
    last_loss = float("nan")
    run_n = run_passed = run_bytes = 0
    win_n = win_passed = win_bytes = 0
    adv_sum = adv_n = 0
    sink = None
    if cfg.json_log:
        sink = open(cfg.json_log, "a", encoding="utf-8")

        def _row(**kw) -> None:
            import json

            sink.write(
                json.dumps({k: v for k, v in kw.items() if v is not None}, default=str) + "\n"
            )
            sink.flush()

        _row(
            event="rl_run",
            source=run.source,
            steps=cfg.steps,
            lr=cfg.lr,
            opt=cfg.opt,
            task_cls=cfg.rl_task_cls,
            tasks=cfg.rl_tasks,
            group=cfg.rl_group,
            budget=cfg.rl_budget,
            adv=cfg.rl_adv,
            resumed=start,
        )
    for step in range(start, cfg.steps):
        task = tasks[rl_task_index(cfg.seed, step, len(tasks))]
        rollouts: list[Rollout] = [
            rollout(
                model,
                task,
                cfg.rl_budget,
                device=dev,
                generator=rl_generator(cfg.seed, step, g, dev),
            )
            for g in range(cfg.rl_group)
        ]
        samples = [Sample.from_rollout(task, r, 0) for r in rollouts]
        verdicts = [r.verdict for r in rollouts]
        lengths = [len(s.response) for s in samples]
        step_counters = Counters()
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
        lf = float(loss.detach())
        finite = math.isfinite(lf)
        window_finite.append(finite)
        if finite and kept > 0:
            loss.backward()
            if cfg.grad_clip > 0:
                clip_grads(model.parameters(), cfg.grad_clip)
            opt.step()
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

        if (step + 1) % cfg.ckpt_every == 0:
            weights, snapshot = save_ckpt(run, model, opt, step)
            log(f"ckpt step={step + 1} -> {weights} ({snapshot})")
            if sink is not None:
                _row(event="rl_ckpt", step=step + 1, path=str(weights))

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
    }
