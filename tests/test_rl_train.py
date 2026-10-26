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
import hashlib
import importlib
import json
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
from dormouse.post.rl import RLConfig, Sample, group_rewards, rl_advantages
from dormouse.post.rl_train import (
    difficulty_side,
    rl_draw_group,
    rl_generator,
    rl_task_index,
    rl_train,
    split_tasks,
    thrift_metrics,
)
from dormouse.post.rollouts import Rollout, rollout
from dormouse.post.tasks import Task, build_tasks, load_rl_tasks
from dormouse.post.verifiers import Verdict, verify
from dormouse.train.loop import load_model_half, load_model_state

#: The MODULE, for `monkeypatch`: `dormouse.post`'s `rl_train` attribute is the
#: re-exported function (it shadows the submodule), and pytest's dotted-path
#: resolver walks attributes — so the module has to be named explicitly.
_RL_TRAIN = importlib.import_module("dormouse.post.rl_train")

runner = CliRunner()


def rl_run(tmp_path: Path, *, model_over: dict | None = None, **over) -> RunConfig:
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
        rl_holdout=1,
    )
    return RunConfig(
        source="rl-tiny",
        model=tiny_cfg(max_seq_len=128, **(model_over or {})),
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
    last = [ln for ln in lines if ln.startswith("rl step=")][-1]
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
    run = rl_run(tmp_path, steps=2, rl_budget=len(prompt) + 2, rl_holdout=0)  # < prompt + 3 → zero
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
    run = rl_run(tmp_path, steps=1, rl_group=4, rl_adv="group", rl_holdout=0)
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
    run = rl_run(
        tmp_path, steps=1, rl_group=4, rl_adv="group", rl_len_penalty=0.4, rl_holdout=0
    )
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
            "--set",
            "rl_holdout=1",
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


# ---------------------------------------------------------------------------
# the corpus split, the difficulty band, and the causal advantage
# ---------------------------------------------------------------------------

#: The pin for the 20-rollout determinism gate: `build_tasks("math", 7, 20)`,
#: `tiny_cfg(max_seq_len=128)`, `torch.manual_seed(0)`, greedy rollouts at a
#: 160 B budget, each response's bytes + a NUL separator. A task or decode
#: change moves it.
_TWENTY_ROLLOUT_SHA256 = "ab71bc3d7598985f31f2b94d8e64b11797b8381ac7c7afe857e3bebd1465535b"


def _scripted_corpus(path: Path) -> Path:
    """12 short math tasks: 3 all-easy, 3 all-hard, 6 generator-parity mixed."""
    records = []
    for i in range(3):
        records.append(
            {
                "id": f"easy-{i}",
                "kind": "arith_chain",
                "prompt": (
                    f"Compute {i} + {i + 1}. Put the final value on its own last line "
                    f"as 'answer: <value>'."
                ),
                "answer": str(2 * i + 1),
                "verifier": "math",
                "difficulty_hint": 0.1,
            }
        )
        records.append(
            {
                "id": f"hard-{i}",
                "kind": "arith_chain",
                "prompt": (
                    f"Compute {i + 7} * {i + 3}. Put the final value on its own last line "
                    f"as 'answer: <value>'."
                ),
                "answer": str((i + 7) * (i + 3)),
                "verifier": "math",
                "difficulty_hint": 0.9,
            }
        )
    for i in range(6):
        records.append(
            {
                "id": f"mixed-{i}",
                "kind": "arith_chain",
                "prompt": (
                    f"Compute {i + 2} - {i}. Put the final value on its own last line "
                    f"as 'answer: <value>'."
                ),
                "answer": "2",
                "verifier": "math",
                "difficulty_hint": 0.5,
            }
        )
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _scripted_rollout(
    model, task, budget, *, device=None, generator=None, greedy=False, temp=None, **_kw
):
    """A deterministic policy over the scripted corpus, graded by the REAL verifier.

    `easy-*` always passes, `hard-*` always fails, `mixed-*` passes on even
    rollout-generator seeds. Nothing here reads the model, so the counters are
    a function of `(task, generator)` alone — which is what lets the resume
    gate compare two runs that loaded different model bytes.
    """
    if task.tid.startswith("easy"):
        passed = True
    elif task.tid.startswith("hard"):
        passed = False
    else:
        passed = generator is not None and generator.initial_seed() % 2 == 0
    response = f"answer: {task.expected}".encode() if passed else b"answer: -999999"
    verdict = verify(task.as_dict(), response.decode())
    assert verdict.passed == passed, (task.tid, verdict.reason)
    trace = [f"{b:02x}" for b in response]
    n_bytes = len(task.prompt.encode("utf-8")) + len(response) + 2 * len(trace)
    return Rollout(trace=trace, n_bytes=n_bytes, verdict=verdict)


def test_difficulty_side_is_inclusive_and_names_the_side():
    assert difficulty_side(0.2, 0.2, 0.8) == "keep"
    assert difficulty_side(0.8, 0.2, 0.8) == "keep"
    assert difficulty_side(0.199, 0.2, 0.8) == "low"
    assert difficulty_side(0.801, 0.2, 0.8) == "high"
    with pytest.raises(LoudError, match="lo <= hi"):
        difficulty_side(0.5, 0.8, 0.2)


def test_the_band_drops_all_easy_and_all_hard_and_counts_the_side(monkeypatch, tmp_path: Path):
    """A synthetic all-easy/all-hard set: one of each side, the last trains anyway."""
    easy = Task(
        cls="math", tid="easy-0", prompt="Compute 1 + 1. answer: <value>", expected="2",
        solution="answer: 2", tests=(), match="exact", budget=64,
    )
    hard = Task(
        cls="math", tid="hard-0", prompt="Compute 7 * 8. answer: <value>", expected="56",
        solution="answer: 56", tests=(), match="exact", budget=64,
    )
    cfg = dataclasses.replace(
        rl_run(tmp_path).train, rl_band_attempts=2, rl_group=2, rl_budget=128
    )
    monkeypatch.setattr(_RL_TRAIN, "rl_task_index", lambda seed, step, n, attempt=0: attempt % n)
    monkeypatch.setattr(_RL_TRAIN, "rollout", _scripted_rollout)
    counters = Counters()
    task, group = rl_draw_group(None, [easy, hard], cfg, 0, torch.device("cpu"), counters)
    assert counters["rl_dropped_high"] == 1, counters
    assert counters["rl_dropped_low"] == 1, counters
    assert counters["rl_zero_spread_groups"] == 2, counters
    assert counters["rl_band_draws"] == 2
    assert counters["rl_band_exhausted"] == 1, "the fallback must be counted, not silent"
    assert task.tid == "hard-0", "the last candidate is the one that trains"
    assert [r.passed for r in group] == [False, False]


def test_the_group_advantage_comes_from_the_verifier_and_flips_on_a_swapped_verdict():
    """Gate 3: the number is the verdict's, not the model's own log-probs.

    Hand table: two `answer: 4` passes and two `answer: 999` failures on a task
    whose expected is 4. The group-normalised advantage is positive for the
    passes and negative for the failures; swapping ONE failure's verdict to a
    pass (re-run through the real `verify`) flips that sample's sign from
    negative to positive. A log-prob-derived number would not notice the swap.
    """
    task = Task(
        cls="math",
        tid="causal-0",
        prompt="Solve 2x + 3 = 11. answer: <value>",
        expected="4",
        solution="answer: 4",
        tests=(),
        match="exact",
        budget=64,
    )
    responses = [b"answer: 4", b"answer: 4", b"answer: 999", b"answer: 999"]
    rollouts = []
    for response in responses:
        trace = [f"{b:02x}" for b in response]
        verdict = verify(task.as_dict(), response.decode())
        rollouts.append(Rollout(trace=trace, n_bytes=len(response), verdict=verdict))
    assert [r.passed for r in rollouts] == [True, True, False, False]

    cfg = RLConfig(quality="none", length_penalty=0.0)
    samples = [Sample.from_rollout(task, r, 0) for r in rollouts]
    rewards = group_rewards(
        [r.verdict for r in rollouts],
        [task.solution] * len(rollouts),
        [len(s.response) for s in samples],
        cfg,
    )
    before = rl_advantages([r.total for r in rewards], [0] * len(rollouts), "group")
    print("\nidx verdict reward advantage")
    for i, (r, rw, a) in enumerate(zip(rollouts, rewards, before, strict=True)):
        print(f"{i:>2} {'PASS' if r.passed else 'FAIL':>4} {rw.total:>6.3f} {a.value:>+9.3f}")
    assert all(before[i].value > 0 for i in (0, 1))
    assert all(before[i].value < 0 for i in (2, 3))

    swapped = list(rollouts)
    response = b"answer: 4"
    swapped[2] = Rollout(
        trace=[f"{b:02x}" for b in response],
        n_bytes=len(response),
        verdict=verify(task.as_dict(), response.decode()),
    )
    samples2 = [Sample.from_rollout(task, r, 0) for r in swapped]
    rewards2 = group_rewards(
        [r.verdict for r in swapped],
        [task.solution] * len(swapped),
        [len(s.response) for s in samples2],
        cfg,
    )
    after = rl_advantages([r.total for r in rewards2], [0] * len(swapped), "group")
    print("idx verdict advantage (one verdict swapped)")
    for i, (r, a) in enumerate(zip(swapped, after, strict=True)):
        print(f"{i:>2} {'PASS' if r.passed else 'FAIL':>4} {a.value:>+9.3f}")
    assert swapped[2].passed
    assert before[2].value < 0 < after[2].value, (before[2].value, after[2].value)


# ---------------------------------------------------------------------------
# leakage, determinism, the 20-step run and its resume
# ---------------------------------------------------------------------------


def test_the_holdout_split_is_hash_disjoint_and_never_drawn(tmp_path: Path):
    tasks = load_rl_tasks(_scripted_corpus(tmp_path / "c.jsonl"))
    train, held, proof = split_tasks(tasks, seed=1, holdout=3)
    assert (len(train), len(held)) == (9, 3)
    assert proof["disjoint"] is True
    assert proof["train_sha256"] != proof["holdout_sha256"]
    assert len(proof["train_sha256"]) == 16
    # Every draw the loop can make, over 40 steps x 4 band attempts, stays in train.
    drawn = {
        train[rl_task_index(1, step, len(train), attempt)].tid
        for step in range(40)
        for attempt in range(4)
    }
    assert drawn.isdisjoint({t.tid for t in held})
    train_hashes = {hashlib.sha256(t.prompt.encode()).hexdigest() for t in train}
    held_hashes = {hashlib.sha256(t.prompt.encode()).hexdigest() for t in held}
    assert train_hashes.isdisjoint(held_hashes)
    with pytest.raises(LoudError, match="empty"):
        split_tasks(tasks, seed=1, holdout=len(tasks))
    with pytest.raises(LoudError, match="BOTH"):
        split_tasks([tasks[0], tasks[0]], seed=1, holdout=1)


def test_twenty_rollouts_hash_to_the_pinned_sha256():
    """Same seed → the same 20 rollout byte strings, pinned by digest."""
    torch.manual_seed(0)
    model = DormouseModel(tiny_cfg(max_seq_len=128))
    digest = hashlib.sha256()
    for task in build_tasks("math", 7, 20):
        record = rollout(model, task, 160, greedy=True)
        assert record.trace, "a greedy rollout must emit at least one byte"
        digest.update(bytes.fromhex("".join(record.trace)))
        digest.update(b"\x00")
    assert digest.hexdigest() == _TWENTY_ROLLOUT_SHA256


def _twenty_step_run(
    out: Path,
    corpus: Path,
    *,
    steps: int,
    resume: bool = False,
    ckpt_every: int = 1000,
    log_every: int = 10,
    log=print,
) -> dict:
    cfg = TrainConfig(
        steps=steps,
        seed=7,  # holds out `easy-2` + `mixed-1`: a pass@1 and a pass@budget > 0
        batch=2,
        seq_len=64,
        lr=0.01,
        log_every=log_every,
        ckpt_every=ckpt_every,
        ckpt_name="r",
        out_dir=str(out),
        rl_corpus=str(corpus),
        rl_task_cls="math",
        rl_tasks=12,
        rl_group=4,
        rl_budget=128,
        rl_adv="group",
        rl_band_attempts=12,
        rl_holdout=2,
        rl_eval_every=10,
        rl_eval_samples=1,
        json_log=str(out / "log.jsonl"),
    )
    run = RunConfig(
        source="rl-twenty",
        model=tiny_cfg(max_seq_len=128, moe_capacity=0.5),
        train=cfg,
    )
    torch.manual_seed(0)  # the model's init, as in the existing determinism gate
    return rl_train(run, log=log, resume=resume)


def test_a_twenty_step_cpu_run_is_finite_writes_a_checkpoint_and_counts_everything(
    monkeypatch, tmp_path: Path
):
    corpus = _scripted_corpus(tmp_path / "c.jsonl")
    monkeypatch.setattr(_RL_TRAIN, "rollout", _scripted_rollout)
    lines: list[str] = []
    report = _twenty_step_run(
        tmp_path / "run", corpus, steps=20, ckpt_every=10, log=lines.append
    )
    counters = report["counters"]
    assert math.isfinite(report["loss"]) and report["nan_skipped"] == 0
    assert report["opt_steps"] == 20
    assert (tmp_path / "run" / "r.pt").is_file()
    assert (tmp_path / "run" / "r.config.toml").is_file()
    # The six counters the brief names, every one non-zero on this run.
    for name in (
        "rl_steps",
        "rl_samples",
        "rl_non_causal_logp",
        "rl_dropped_high",
        "rl_dropped_low",
        "rl_zero_spread_groups",
    ):
        assert counters[name] > 0, (name, dict(counters))
    assert counters["rl_steps"] == 20
    assert counters["rl_samples"] == 20 * 4
    assert counters["rl_non_causal_logp"] == 20
    # The eval ran before and after; the held-out numbers are real fractions.
    evals = [ln for ln in lines if ln.startswith("rl eval")]
    assert evals[0].startswith("rl eval step=0 ") and evals[-1].startswith("rl eval step=20 ")
    assert report["eval_history"][0]["step"] == 0
    assert report["eval_history"][-1]["step"] == 20
    assert 0.0 <= report["eval_history"][0]["pass@1"] <= 1.0
    assert 0.0 <= report["eval_history"][-1]["pass@budget"] <= 1.0
    # The split proof reached the log.
    assert any("disjoint=True" in ln for ln in lines), lines


def test_a_resume_reproduces_the_second_windows_counters(monkeypatch, tmp_path: Path):
    """A checkpoint at step 10 + a resume to 20 must count what a fresh 20-run counts."""
    corpus = _scripted_corpus(tmp_path / "c.jsonl")
    monkeypatch.setattr(_RL_TRAIN, "rollout", _scripted_rollout)
    one = _twenty_step_run(tmp_path / "a1", corpus, steps=10, ckpt_every=10)
    assert one["counters"]["rl_steps"] == 10
    assert (tmp_path / "a1" / "r.pt").is_file()
    resumed = _twenty_step_run(tmp_path / "a1", corpus, steps=20, resume=True)
    assert resumed["counters"]["rl_steps"] == 10, "a resume runs the remaining steps only"
    assert resumed["opt_steps"] == 20, "the optimizer state came back"

    _twenty_step_run(tmp_path / "full", corpus, steps=20)
    rows = [
        json.loads(line)
        for line in (tmp_path / "full" / "log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    reference = [r for r in rows if r.get("event") == "rl_log"][-1]["counters"]
    keys = (
        "rl_steps",
        "rl_samples",
        "rl_non_causal_logp",
        "rl_dropped_high",
        "rl_dropped_low",
        "rl_zero_spread_groups",
    )
    assert {k: resumed["counters"].get(k, 0) for k in keys} == {
        k: reference.get(k, 0) for k in keys
    }
