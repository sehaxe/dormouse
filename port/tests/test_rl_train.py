"""Gates for the RL loop (`post/rl_train.py`): the trainer around the stepper.

CPU and tiny, like the rest of the port's gates: what has to be true of the RL
loop is the SEMANTICS — a step trains the group it sampled, a group that
produced nothing takes no step and says so in the counters, a checkpoint
round-trips and resumes, one seed is one run — and none of that needs a card.
The number the owner reads off a real run, `bytes/solved`, is pinned twice: as
arithmetic on a hand table, and end to end through the log line with hand-built
Rollouts.

**Two sizes are not the brief's sketch, both forced by refusals the lane
already has.** `max_seq_len=128`, because a `math` prompt is 81-86 B and
`rl_step` refuses a window past the model's `max_seq_len` (the tiny preset's 64
is shorter than the prompt alone). `rl_budget=96`, because `rollout` refuses a
budget below the prompt's own length — the `math` class's declared
`Task.budget = 64` is itself below its prompt, which is a `tasks.py` question,
not this loop's.
"""

from __future__ import annotations

import dataclasses
import importlib
import math
from pathlib import Path

import pytest
import torch
from conftest import tiny_cfg
from typer.testing import CliRunner

from dormouse.cli import app
from dormouse.config import RunConfig, TrainConfig, load_run
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel
from dormouse.post import rl_train as exported_rl_train
from dormouse.post.rl_train import rl_generator, rl_task_index, rl_train, thrift_metrics
from dormouse.post.rollouts import Rollout
from dormouse.post.tasks import Task
from dormouse.post.verifiers import Verdict
from dormouse.train.loop import load_model_half, load_model_state

#: The MODULE, for `monkeypatch`: `dormouse.post`'s `rl_train` attribute is the
#: re-exported function (it shadows the submodule), and pytest's dotted-path
#: resolver walks attributes — so the module has to be named explicitly.
_RL_TRAIN = importlib.import_module("dormouse.post.rl_train")

runner = CliRunner()


def rl_run(tmp_path: Path, **over) -> RunConfig:
    """A tiny RL run: `math` tasks, no corpus at all (the lane generates its data)."""
    cfg = TrainConfig(
        steps=3,
        batch=2,
        seq_len=64,
        lr=0.01,
        log_every=1,
        ckpt_every=1000,
        ckpt_name="r",
        out_dir=str(tmp_path / "ck"),
        rl_task_cls="math",
        rl_tasks=2,
        rl_group=2,
        rl_budget=96,
    )
    return RunConfig(
        source="rl-tiny",
        model=tiny_cfg(max_seq_len=128),
        train=dataclasses.replace(cfg, **over),
    )


def _losses(lines: list[str]) -> list[float]:
    return [float(ln.split("loss=")[1].split()[0]) for ln in lines if ln.startswith("rl step=")]


def _preset(tmp_path: Path) -> Path:
    """A tiny preset that fits a `math` prompt: the same shape `conftest.TINY` is."""
    p = tmp_path / "tiny.toml"
    p.write_text(
        "\n".join(
            [
                "d_model = 64",
                "n_heads = 4",
                "head_dim = 16",
                "d_ffn = 128",
                "max_seq_len = 128",
                "max_iter = 2",
                "rank = 8",
                "shared_rank = 16",
                "moe_topk = 2",
                "moe_capacity = 1.0",
                "engram_orders = [2, 3]",
                "engram_rows = 64",
                "engram_dim = 8",
                "bf16 = false",
            ]
        )
        + "\n"
    )
    return p


# --------------------------------------------------------------------------- the loop


def test_rl_train_runs_and_the_loss_is_finite(tmp_path: Path):
    """The e2e shape: 3 steps of `math`/2 tasks/group 2, every step trains the
    group it sampled (`rl_samples > 0`), and the counters are the run's own."""
    run = rl_run(tmp_path)
    lines: list[str] = []
    report = rl_train(run, log=lines.append)
    losses = _losses(lines)
    assert len(losses) == 3 and all(math.isfinite(v) for v in losses), losses
    assert report["loss"] == pytest.approx(losses[-1], abs=1e-4)
    assert report["nan_skipped"] == 0 and report["counters"].get("nan_skipped", 0) == 0
    # `rl_step` counts its own call and every sample that contributed a log-prob.
    assert report["counters"]["rl_steps"] == 3
    assert report["counters"]["rl_samples"] == 3 * run.train.rl_group
    assert report["opt_steps"] == 3, "every step had something to train"
    # The fields the owner reads: a pass rate in [0, 1] and a byte price per
    # solution (or `-` while nothing is solved yet).
    assert 0.0 <= report["pass_rate"] <= 1.0
    last = lines[-1]
    for field in ("pass=", "bytes/solved=", "adv=", "rl_samples=", "nan_skipped="):
        assert field in last, last
    assert (report["bytes_per_solved"] is None) == ("bytes/solved=-" in last)


def test_rl_train_writes_a_reloadable_checkpoint_and_resumes(tmp_path: Path):
    """The checkpoint is the SFT one (`save_ckpt`'s pair), so it loads through the
    ONE loader and a longer run continues it — `steps` is a progress key."""
    run = rl_run(tmp_path, steps=2, ckpt_every=2)
    report = rl_train(run, log=lambda *_: None)
    assert report["opt_steps"] == 2
    weights = Path(run.train.out_dir) / "r.pt"
    assert weights.is_file() and weights.with_suffix(".config.toml").is_file()
    assert int(torch.load(weights, map_location="cpu", weights_only=False)["step"]) == 1

    # reloadable: the same loader `eval`/`serve` use
    model = DormouseModel(run.model)
    load_model_state(model, run.train, load_model_half(weights))

    longer = dataclasses.replace(run, train=dataclasses.replace(run.train, steps=4))
    lines: list[str] = []
    again = rl_train(longer, resume=True, log=lines.append)
    assert any(ln.startswith("rl resume: step=2") for ln in lines), lines
    assert again["counters"]["rl_steps"] == 2, "a resume runs the remaining steps only"
    assert again["opt_steps"] == 4, "the optimizer state came back from the file"


def test_rl_train_is_deterministic_for_one_seed(tmp_path: Path):
    """One seed, one init → the same losses: the task draw and every rollout's
    generator are pure functions of `(seed, step, sample)`, and the loop draws
    from nothing else."""

    def once() -> list[float]:
        torch.manual_seed(0)  # the model's init is the global RNG's, as in train/sft
        lines: list[str] = []
        rl_train(rl_run(tmp_path), log=lines.append)
        return _losses(lines)

    a, b = once(), once()
    assert a == b and len(a) == 3, (a, b)


def test_rl_train_takes_no_step_when_every_rollout_is_empty(monkeypatch, tmp_path: Path):
    """A stub task whose budget cannot fit one byte: every rollout is empty, every
    reward is 0, `rl_step` skips all of them — and the loop must NOT take an
    optimizer step (an AdamW step on zero gradients still applies weight decay)."""
    prompt = "x" * 40
    stub = Task(
        cls="math",  # a real verifier class: an empty response is a FAIL, not a refusal
        tid="stub-0",
        prompt=prompt,
        expected="0",
        solution="0",
        tests=(),
        match="exact",
        budget=0,
    )
    monkeypatch.setattr(_RL_TRAIN, "build_tasks", lambda cls, seed, n: [stub] * n)
    run = rl_run(tmp_path, steps=2, rl_budget=len(prompt) + 2)  # < prompt + 3 → zero bytes
    lines: list[str] = []
    report = rl_train(run, log=lines.append)
    assert report["opt_steps"] == 0, "nothing contributed, so nothing may step"
    assert report["counters"].get("rl_samples", 0) == 0
    assert report["counters"]["rl_steps"] == 2
    assert report["counters"]["rl_skipped_no_advantage"] == 2 * run.train.rl_group
    assert report["nan_skipped"] == 0 and math.isfinite(report["loss"])
    assert report["loss"] == 0.0, "the empty step's loss is a finite zero"
    assert "rl_samples=0" in lines[-1] and "nan_skipped=0" in lines[-1]


def test_rl_init_from_loads_the_weights_and_refuses_a_resume_pair(tmp_path: Path):
    """`--init-from` is the SFT loader path (`_init_cfg` + `load_model_state`):
    the foreign weights arrive, and naming both sources is LOUD.

    `lr=0.0` is the witness: the checkpoint this run writes carries the foreign
    table (0.25) exactly, so the run really started from the init file and not
    from a fresh init.
    """
    run = rl_run(tmp_path, steps=1, lr=0.0, ckpt_every=1)
    src = DormouseModel(run.model)
    with torch.no_grad():
        for key, p in src.state_dict().items():
            if key.endswith("engram.tables"):
                p.fill_(0.25)
    foreign = tmp_path / "foreign.pt"
    torch.save({"model": src.state_dict(), "step": 7}, foreign)

    lines: list[str] = []
    report = rl_train(run, init_from=str(foreign), log=lines.append)
    assert any(ln.startswith("rl init: weights from") for ln in lines), lines
    assert math.isfinite(report["loss"])
    sd = load_model_half(Path(run.train.out_dir) / "r.pt")
    tables = [v for k, v in sd.items() if k.endswith("engram.tables")]
    assert len(tables) == 1 and float((tables[0] - 0.25).abs().max()) < 1e-6

    with pytest.raises(LoudError, match="does not exist"):
        rl_train(run, init_from=str(tmp_path / "no.pt"), log=lambda *_: None)
    with pytest.raises(LoudError, match="two different sources"):
        rl_train(run, init_from=str(foreign), resume=True, log=lambda *_: None)


def test_rl_train_skips_a_non_finite_step_and_two_windows_stop_the_run(monkeypatch, tmp_path: Path):
    """The firewall `train()`/`sft_train()` carry: a non-finite loss takes no
    optimizer step and is COUNTED, and two consecutive all-non-finite windows
    stop the run for a human rather than exiting 0."""

    def nan_step(model, opt, samples, advantages, cfg, *, counters=None):
        counters = Counters() if counters is None else counters
        counters.count("rl_steps")
        counters.count("rl_samples")
        return torch.full((), float("nan"), requires_grad=True), counters

    monkeypatch.setattr(_RL_TRAIN, "rl_step", nan_step)
    report = rl_train(rl_run(tmp_path, steps=1), log=lambda *_: None)
    assert report["nan_skipped"] == 1 and report["counters"]["nan_skipped"] == 1
    assert report["opt_steps"] == 0, "a non-finite step must touch no weight"
    with pytest.raises(LoudError, match="entirely non-finite"):
        rl_train(rl_run(tmp_path, steps=2), log=lambda *_: None)


# --------------------------------------------------------------------------- the metrics


def test_thrift_metrics_on_a_hand_table():
    """`(pass rate, bytes per SOLVED rollout)` from the three counters: 2 of 3
    solved at 10 and 20 B → 2/3 and 15 B; nothing solved is `None`, never 0.0."""
    assert thrift_metrics(3, 2, 30) == (2 / 3, 15.0)
    assert thrift_metrics(4, 1, 7) == (0.25, 7.0)
    assert thrift_metrics(2, 0, 0) == (0.0, None)
    assert thrift_metrics(0, 0, 0) == (0.0, None)


def test_the_log_line_carries_the_verdicts_own_thrift_numbers(monkeypatch, tmp_path: Path):
    """Hand-built Rollouts: 2 of 4 pass, at 10 B and 30 B → `pass=0.50
    bytes/solved=20.0`. The byte cost is `Rollout.n_bytes` (prompt + response +
    strokes), NOT the response's length — a failing attempt's bytes never enter
    the price of a solution."""
    hand = iter(
        [
            Rollout(trace=["61", "62"], n_bytes=10, verdict=Verdict(True, "sympy-equivalent", 0.0)),
            Rollout(trace=["61", "62"], n_bytes=30, verdict=Verdict(True, "sympy-equivalent", 0.0)),
            Rollout(trace=["61", "62"], n_bytes=99, verdict=Verdict(False, "not equivalent", 0.0)),
            Rollout(trace=["61", "62"], n_bytes=99, verdict=Verdict(False, "not equivalent", 0.0)),
        ]
    )
    monkeypatch.setattr(_RL_TRAIN, "rollout", lambda *a, **kw: next(hand))
    run = rl_run(tmp_path, steps=1, rl_group=4, rl_adv="group")
    lines: list[str] = []
    report = rl_train(run, log=lines.append)
    assert "pass=0.50" in lines[-1] and "bytes/solved=20.0" in lines[-1], lines[-1]
    assert (report["pass_rate"], report["bytes_per_solved"]) == (0.5, 20.0)
    assert report["counters"]["rl_samples"] == 4, "the two passes and the two failures all train"
    assert report["opt_steps"] == 1


def test_the_group_relative_length_penalty_is_evaluated_when_it_is_on(monkeypatch, tmp_path: Path):
    """`rl_len_penalty > 0` takes the `group_stats` + per-item `reward` path (the
    one `group_rewards` runs internally, hoisted): two passes of 2 B and 4 B →
    ℓ*_q = 3 B, the gate opens (0.5 > 0.4) and the term is COUNTED active on both
    passes — the branch a default (`0.0`, off) run never executes."""
    hand = iter(
        [
            Rollout(trace=["61", "62"], n_bytes=10, verdict=Verdict(True, "sympy-equivalent", 0.0)),
            Rollout(
                trace=["61", "62", "63", "64"],
                n_bytes=30,
                verdict=Verdict(True, "sympy-equivalent", 0.0),
            ),
            Rollout(trace=["61"], n_bytes=9, verdict=Verdict(False, "no final answer", 0.0)),
            Rollout(trace=["61"], n_bytes=9, verdict=Verdict(False, "no final answer", 0.0)),
        ]
    )
    monkeypatch.setattr(_RL_TRAIN, "rollout", lambda *a, **kw: next(hand))
    run = rl_run(tmp_path, steps=1, rl_group=4, rl_adv="group", rl_len_penalty=0.4)
    lines: list[str] = []
    report = rl_train(run, log=lines.append)
    assert report["counters"]["rl_len_penalty_active"] == 2, "the term ran on both passes"
    assert report["counters"].get("rl_len_penalty_gated", 0) == 0, "the gate was open"
    assert report["counters"]["rl_samples"] == 4
    assert (report["pass_rate"], report["bytes_per_solved"]) == (0.5, 20.0)
    assert "pass=0.50" in lines[-1]


def test_the_task_draw_and_the_rollout_generators_are_pure():
    """Same `(seed, step)` → the same task; the schedule moves with the step; and
    each `(step, sample)` gets its OWN generator, so no rollout's bytes depend on
    how many bytes the earlier ones emitted before the budget stopped them."""
    assert rl_task_index(7, 3, 8) == rl_task_index(7, 3, 8)
    assert len({rl_task_index(7, s, 8) for s in range(8)}) > 1
    cpu = torch.device("cpu")
    a = torch.rand(4, generator=rl_generator(7, 3, 0, cpu))
    b = torch.rand(4, generator=rl_generator(7, 3, 0, cpu))
    c = torch.rand(4, generator=rl_generator(7, 3, 1, cpu))
    assert torch.equal(a, b) and not torch.equal(a, c)


# --------------------------------------------------------------------------- config + CLI


def test_load_run_refuses_a_bogus_advantage_arm_and_a_broken_penalty(tmp_path: Path):
    """`rl_adv` is checked at load (the literal list), and the reward's own rules
    are reached by CONSTRUCTING the `RLConfig` the keys describe — one place, so
    the ramp's division by zero is refused before a run starts."""
    preset = str(_preset(tmp_path))
    with pytest.raises(LoudError, match="rl_adv='bogus'"):
        load_run(preset, train_overrides={"seq_len": 64, "rl_adv": "bogus"})
    with pytest.raises(LoudError, match="divides by zero"):
        load_run(preset, train_overrides={"seq_len": 64, "rl_len_s": 0.0})
    with pytest.raises(LoudError, match="rl_group=0"):
        load_run(preset, train_overrides={"seq_len": 64, "rl_group": 0})


def test_cli_rl_trains_and_writes_a_checkpoint(tmp_path: Path):
    r = runner.invoke(
        app,
        [
            "rl",
            str(_preset(tmp_path)),
            "--ckpt-name",
            "r",
            "--set",
            f"out_dir={tmp_path / 'ck'}",
            "--set",
            "steps=2",
            "--set",
            "seq_len=64",
            "--set",
            "log_every=1",
            "--set",
            "ckpt_every=2",
            "--set",
            "rl_task_cls=math",
            "--set",
            "rl_tasks=2",
            "--set",
            "rl_group=2",
            "--set",
            "rl_budget=96",
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code == 0, r.output
    assert (tmp_path / "ck" / "r.pt").is_file()
    assert (tmp_path / "ck" / "r.config.toml").is_file()
    assert "rl done" in r.output and "pass=" in r.output
    assert "bytes/solved=" in r.output


def test_rl_train_is_exported_from_the_post_package():
    """The lane's entry point is on the package (`from dormouse.post import rl_train`)."""
    assert exported_rl_train is rl_train
