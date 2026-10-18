"""The MC bench: the byte-level score, both accuracies, the drop counter, the loud load.

Two instruments, tested separately, because they can be wrong independently:

* the SCORE (`bench/score.py`) — pinned against hand-computed log-probs from a
  fake model whose softmax is exact: a byte with a 0 logit has `p = 1/258`, a
  byte boosted to `ln 3` has `p = 3/258` (`258 = 255·1 + 3`, the fixture's own
  denominator), so every number in this file is arithmetic, not a snapshot of
  whatever torch printed;
* the LOOP (`bench/runner.py`) — accuracy math, the drop rule, the JSONL row,
  and one end-to-end run of the synthetic task on the tiny CPU preset.

The two-accuracy test is the one that matters most: raw sum and per-byte mean
are different protocols, and a fixture where they pick DIFFERENT choices is what
proves the harness reports two numbers instead of the same one twice.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from conftest import tiny_cfg, tiny_model
from typer.testing import CliRunner

from dormouse.bench import MCItem, load_task, run_bench, run_task, score_choices
from dormouse.bench.score import model_logits
from dormouse.bench.tasks import TASKS, synth_math
from dormouse.cli import app
from dormouse.config import RunConfig, TrainConfig
from dormouse.instrument import LoudError

VOCAB = 256
# The fixture's exact arithmetic. A position either carries a boost or it does
# not, and the two have different denominators:
#   boosted position   — 255 bytes at logit 0, one at ln 3 → Σe = 258:
#                        the boosted byte reads ln(3/258), any other -ln(258)
#   unboosted position — all 256 logits 0 → p = 1/256, log p = -ln(256)
LN_BOOST = math.log(3.0 / 258.0)  # -4.4543473
LN_PLAIN = -math.log(258.0)  # -5.5529596
LN_UNIFORM = -math.log(256.0)  # -5.5451774

runner = CliRunner()


# ---------------------------------------------------------------------------
# the fake model: an exact softmax, spelled out per byte string
# ---------------------------------------------------------------------------


def fake_logits(table: dict[bytes, dict[int, int]]):
    """A fake model: for the ids `S`, boost `table[S][pos]` at each named position.

    Every logit is 0 unless the table boosts it, so `log_softmax` is exact:
    the boosted byte reads `ln 3 - ln 258`, any other byte `-ln 258`. An ids
    string that is not in the table is a fixture bug and asserts.
    """

    def fn(ids: torch.Tensor) -> torch.Tensor:
        s = bytes(ids[0].tolist())
        assert s in table, f"the fixture has no entry for {s!r}"
        logits = torch.zeros(ids.shape[1], VOCAB)
        for pos, byte in table[s].items():
            logits[pos, byte] = math.log(3.0)
        return logits

    return fn


#: Three items over one context byte `c`, each with its own exact distribution:
#:   item A  choices p/q/r — the strict ordering case (p boosted, q and r not)
#:   item B  choices x/yy  — the DISAGREEMENT case (see the test below)
#:   item C  choices mm/mn/nm — two boosted positions, three distinct sums
TABLE: dict[bytes, dict[int, int]] = {
    b"cp": {0: ord("p")},
    b"cq": {},
    b"cr": {},
    b"cx": {},
    b"cyy": {1: ord("y")},
    b"cmm": {0: ord("m")},
    b"cmn": {0: ord("m"), 1: ord("n")},
    b"cnm": {},
}


def _items() -> list[MCItem]:
    return [
        MCItem("A", "c", ("p", "q", "r"), label=0),
        MCItem("B", "c", ("x", "yy"), label=1),
        MCItem("C", "c", ("mm", "mn", "nm"), label=0),
    ]


# ---------------------------------------------------------------------------
# the score
# ---------------------------------------------------------------------------


def test_score_pins_the_hand_computed_log_probs():
    scores = score_choices(fake_logits(TABLE), b"c", [b"mm", b"mn", b"nm"])
    assert [s.n_bytes for s in scores] == [2, 2, 2]
    # mm: m boosted at 0, row 1 unboosted   mn: both rows boosted   nm: neither
    assert scores[0].logp == pytest.approx(LN_BOOST + LN_UNIFORM)  # -9.9995
    assert scores[1].logp == pytest.approx(2 * LN_BOOST)  # -8.9087
    assert scores[2].logp == pytest.approx(2 * LN_UNIFORM)  # -11.0904
    assert scores[0].logp == pytest.approx(-9.9995, abs=1e-4)
    assert scores[1].mean == pytest.approx(LN_BOOST)
    assert scores[2].mean == pytest.approx(LN_UNIFORM)


def test_score_ordering_is_strict_where_the_fixture_is():
    scores = score_choices(fake_logits(TABLE), b"c", [b"p", b"q", b"r"])
    assert scores[0].logp == pytest.approx(LN_BOOST)
    assert scores[1].logp == scores[2].logp == pytest.approx(LN_UNIFORM)
    assert scores[0].logp > scores[1].logp
    assert scores[0].mean > scores[1].mean


def test_one_forward_per_choice_and_the_context_is_not_scored():
    """The count is the protocol: 1 call per choice, and `n_bytes` is the choice's."""
    calls = []

    def counting(ids: torch.Tensor) -> torch.Tensor:
        calls.append(bytes(ids[0].tolist()))
        return fake_logits(TABLE)(ids)

    scores = score_choices(counting, b"c", [b"p", b"q", b"r"])
    assert calls == [b"cp", b"cq", b"cr"]
    assert sum(s.n_bytes for s in scores) == 3  # the 1 context byte is never scored


def test_a_choice_longer_than_the_context_asserts_nothing_and_scores_every_byte():
    scores = score_choices(fake_logits(TABLE), b"c", [b"yy"])
    assert scores[0].n_bytes == 2
    assert scores[0].logp == pytest.approx(LN_UNIFORM + LN_BOOST)


# ---------------------------------------------------------------------------
# the loop: accuracy math, both columns, drops
# ---------------------------------------------------------------------------


def test_accuracy_is_hand_computed_and_the_two_columns_disagree():
    """acc_raw = 1/3, acc_mean = 2/3 on the same three items, same forwards.

    Item B is the disagreement: raw picks `x` (-5.5452 over `yy`'s -9.9995),
    the per-byte mean picks `yy` (-4.9998 over `x`'s -5.5452). One harness, two
    protocols, two answers — which is why both columns are printed.
    """
    report = run_task(_items(), logits_fn=fake_logits(TABLE), max_seq_len=64, task="fixture")
    assert (report.n, report.dropped) == (3, 0)
    assert report.acc_raw == pytest.approx(1 / 3)
    assert report.acc_mean == pytest.approx(2 / 3)
    assert report.bytes_scored == 3 + 3 + 6  # the choices of all 3 items
    assert report.counters == {}  # nothing was dropped, so nothing is counted


def test_per_byte_and_raw_pick_different_choices_on_the_crafted_item():
    scores = score_choices(fake_logits(TABLE), b"c", [b"x", b"yy"])
    raw = max(range(2), key=lambda i: scores[i].logp)
    mean = max(range(2), key=lambda i: scores[i].mean)
    assert (raw, mean) == (0, 1)


def test_overlong_items_are_dropped_with_the_counter_never_truncated():
    items = [
        MCItem("fits", "c", ("p", "q"), label=0),  # 1 + 1 = 2 bytes
        MCItem("long", "c", ("yyyyyy", "q"), label=1),  # 1 + 6 = 7 bytes
    ]
    report = run_task(items, logits_fn=fake_logits(TABLE), max_seq_len=2, task="mixed")
    assert (report.n, report.dropped) == (1, 1)
    assert report.counters == {"bench_dropped": 1}
    # the dropped item's 7 bytes are scored not-at-all: 2 is the surviving item's
    # two 1-byte choices, not its 7 truncated down to the window.
    assert report.bytes_scored == 2


def test_a_context_longer_than_the_window_is_dropped_too():
    items = [
        MCItem("ctx", "cccccc", ("p",), label=0),  # 6 + 1 = 7 > 2
        MCItem("fits", "c", ("p",), label=0),  # 1 + 1 = 2 <= 2
    ]
    report = run_task(items, logits_fn=fake_logits(TABLE), max_seq_len=2, task="ctx")
    assert (report.n, report.dropped) == (1, 1)


def test_an_all_dropped_task_refuses_loudly():
    items = [MCItem("long", "c", ("yyyyyy",), label=0)]
    with pytest.raises(LoudError) as e:
        run_task(items, logits_fn=fake_logits(TABLE), max_seq_len=2, task="empty")
    assert "0 of 1 items" in str(e.value)
    assert "Escape" in str(e.value)


def test_limit_caps_the_items_scored():
    report = run_task(_items(), logits_fn=fake_logits(TABLE), max_seq_len=64, task="t", limit=2)
    assert report.n == 2


# ---------------------------------------------------------------------------
# the tasks
# ---------------------------------------------------------------------------


def test_synth_math_is_deterministic_and_well_formed():
    a, b = synth_math(12, seed=3), synth_math(12, seed=3)
    assert a == b
    assert synth_math(12, seed=4) != a
    for item in a:
        assert len(set(item.choices)) == 4  # four DISTINCT options, one of them right
        assert 0 <= item.label < 4
        assert item.context.endswith("A:")
    # the labelled choice is the true value of the printed expression
    item = a[0]
    left, op, right = item.context.split()[1:4]
    truth = {
        "+": int(left) + int(right),
        "-": int(left) - int(right),
        "*": int(left) * int(right),
    }[op]
    assert int(item.choices[item.label]) == truth


def test_load_task_of_a_missing_file_is_loud_with_the_url(tmp_path: Path):
    with pytest.raises(LoudError) as e:
        load_task("arc_easy", tmp_path)
    msg = str(e.value)
    assert "arc_easy_test.jsonl" in msg
    assert TASKS["arc_easy"].url in msg
    assert "bench --download" in msg


def test_load_task_of_an_unknown_name_is_loud():
    with pytest.raises(LoudError) as e:
        load_task("mmlu", Path("."))
    assert "unknown task" in str(e.value)
    assert "arc_easy" in str(e.value)


def test_synth_math_needs_no_file(tmp_path: Path):
    assert len(load_task("synth_math", tmp_path, limit=3)) == 3


# ---------------------------------------------------------------------------
# the end-to-end run: a real tiny model on CPU + the JSONL row
# ---------------------------------------------------------------------------


def test_synth_task_end_to_end_on_the_tiny_cpu_model(tmp_path: Path):
    model = tiny_model()
    model.eval()
    items = load_task("synth_math", tmp_path, limit=4)
    report = run_task(
        items,
        logits_fn=lambda ids: model_logits(model, ids),
        max_seq_len=64,
        task="synth_math",
    )
    assert report.n == 4 and report.dropped == 0
    assert report.bytes_scored == sum(len(c.encode()) for i in items for c in i.choices)
    assert 0.0 <= report.acc_raw <= 1.0 and 0.0 <= report.acc_mean <= 1.0
    assert report.counters == {}


def test_run_bench_writes_one_row_per_task_and_prints_them(tmp_path: Path, capsys):
    model = tiny_model()
    model.eval()
    log = tmp_path / "bench.jsonl"
    reports = run_bench(
        lambda ids: model_logits(model, ids),
        ["synth_math", "synth_math"],
        data_dir=tmp_path,
        max_seq_len=64,
        limit=2,
        json_log=log,
        arm=None,
    )
    assert [r.task for r in reports] == ["synth_math", "synth_math"]
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 2
    assert set(rows[0]) == {
        "event",
        "task",
        "n",
        "dropped",
        "acc_raw",
        "acc_mean",
        "bytes_scored",
        "max_seq_len",
        "counters",
        "arm",
    }
    assert rows[0]["event"] == "bench" and rows[0]["n"] == 2
    assert rows[0]["max_seq_len"] == 64
    printed = capsys.readouterr().out
    assert printed.count("bench synth_math") == 2
    assert "n=2" in printed and "acc_mean=" in printed and "bytes=" in printed


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------


@pytest.fixture()
def ckpt_dir(tmp_path: Path) -> Path:
    """A checkpoint pair written directly — no train, no GPU, no card dependency.

    The `serve` fixture's rule: the CLI's contract is the snapshot/weights pair,
    and a 4-step CPU train would only re-test the trainer (`test_cli.py`'s job) —
    while touching `torch.optim`, whose own accelerator health check fails on
    this box whenever another run holds the card.
    """
    d = tmp_path / "ck"
    d.mkdir()
    run = RunConfig(
        source="tiny", model=tiny_cfg(), train=TrainConfig(ckpt_name="smoke", out_dir=str(d))
    )
    (d / "smoke.config.toml").write_text(run.snapshot_toml())
    torch.save({"model": tiny_model().state_dict(), "opt": {}, "step": 0}, d / "smoke.pt")
    return d


def test_bench_cli_runs_synth_math_on_a_checkpoint(ckpt_dir: Path, tmp_path: Path):
    log = tmp_path / "bench.jsonl"
    r = runner.invoke(
        app,
        [
            "bench",
            "--out-dir",
            str(ckpt_dir),
            "--ckpt-name",
            "smoke",
            "--tasks",
            "synth_math",
            "--limit",
            "3",
            "--json-log",
            str(log),
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "acc_mean" in r.output and "acc_raw" in r.output
    assert "counted" in r.output  # the CPU arm note, printed not hidden
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row["task"] for row in rows] == ["synth_math"]
    assert rows[0]["n"] == 3


def test_bench_cli_refuses_a_missing_task_file_by_name(ckpt_dir: Path, tmp_path: Path):
    r = runner.invoke(
        app,
        [
            "bench",
            "--out-dir",
            str(ckpt_dir),
            "--ckpt-name",
            "smoke",
            "--tasks",
            "arc_easy",
            "--data-dir",
            str(tmp_path / "empty"),
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code != 0
    assert "arc_easy" in str(r.exception) and "download" in str(r.exception)


def test_bench_cli_needs_a_checkpoint_name(ckpt_dir: Path):
    r = runner.invoke(app, ["bench", "--out-dir", str(ckpt_dir), "--device", "cpu"])
    assert r.exit_code != 0 and "needs --ckpt-name" in str(r.exception)


def test_bench_cli_preset_is_a_check_that_refuses_on_drift(ckpt_dir: Path):
    r = runner.invoke(
        app,
        [
            "bench",
            "small",
            "--out-dir",
            str(ckpt_dir),
            "--ckpt-name",
            "smoke",
            "--tasks",
            "synth_math",
            "--limit",
            "1",
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code != 0
    assert "model half differs" in str(r.exception)
