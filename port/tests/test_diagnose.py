"""`dormouse diagnose` on SYNTHETIC logs: one fixture per known pathology.

The strings are built here, not recorded from a run, because each pathology is
a shape of line or counter — the point of the command is that the shape is
recognized without a human reading the log. Every fixture asserts the row that
must fire and the evidence that makes the verdict checkable; the clean fixture
asserts the opposite direction (no FAIL on a healthy run).
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dormouse.cli import app
from dormouse.diagnose import FAIL, PASS, SKIP, inspect

runner = CliRunner()


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _toml(**tables) -> str:
    out = []
    for name, table in tables.items():
        out.append(f"[{name}]")
        out.extend(f"{key} = {json.dumps(value)}" for key, value in table.items())
        out.append("")
    return "\n".join(out)


def _snapshot(tmp_path: Path, **over) -> Path:
    model = {"n_experts": over.pop("n_experts", 512)}
    train = {
        "steps": 1000,
        "batch": 8,
        "seq_len": 1024,
        "grad_accum": 1,
        "lr": 1e-3,
        "warmup_tokens": 0,
        "decay_frac": 0.5,
        "min_lr_frac": 0.1,
        "ckpt_every": 500,
        "ckpt_name": "run",
        "out_dir": str(tmp_path / "ckpt"),
        "data": str(tmp_path / "train.bin"),
        "eval_data": str(tmp_path / "eval.bin"),
        "mix_plan": "",
    }
    train.update(over)
    path = tmp_path / "run.config.toml"
    path.write_text(_toml(model=model, train=train), encoding="utf-8")
    return path


def _row(checks, name: str):
    return next(check for check in checks if check.name == name)


def _jsonl(tmp_path: Path, rows: list[dict], name: str = "run.jsonl") -> Path:
    return _write(
        tmp_path / name,
        [json.dumps(row) for row in rows],
    )


# ---------------------------------------------------------------------------
# 1. loss NaN/inf
# ---------------------------------------------------------------------------


def test_a_nan_loss_names_the_step_it_started(tmp_path):
    log = _write(
        tmp_path / "run.log",
        [
            "╰─────────────── steps=1000 batch=8 seq=1024 lr=0.001 opt=mix ───────────────╯",
            "step=100 loss=2.5000 lr=1.000e-03 nan_skipped=0",
            "step=200 loss=nan lr=9.000e-04 nan_skipped=400",
        ],
    )
    check = _row(inspect(log, _snapshot(tmp_path)), "loss finite")
    assert check.status == FAIL
    assert "step 200" in check.evidence


def test_two_consecutive_skipped_windows_fail_even_with_a_finite_print(tmp_path):
    """The trainer prints the last FINITE loss; the firewall's counter is the truth."""
    log = _write(
        tmp_path / "run.log",
        [
            "steps=1000 batch=8 seq=1024 lr=0.001",
            "step=200 loss=2.4000 lr=1.000e-03 nan_skipped=800",
            "step=300 loss=2.4000 lr=9.000e-04 nan_skipped=800",
        ],
    )
    check = _row(inspect(log), "loss finite")
    assert check.status == FAIL
    assert "2 consecutive" in check.evidence


# ---------------------------------------------------------------------------
# 2. LFB divergence — the loader's line and the in-training counters
# ---------------------------------------------------------------------------


def test_the_loader_line_past_the_bound_is_a_failure(tmp_path):
    log = _write(
        tmp_path / "run.log",
        [
            "steps=1000 batch=8 seq=1024 lr=0.001",
            "checkpoint: lfb_bias range [-218.269, +72.667] over 1 module(s) is past "
            "±10: this controller DIVERGED",
        ],
    )
    check = _row(inspect(log), "lfb divergence")
    assert check.status == FAIL
    assert "-218.269" in check.evidence


def test_a_healthy_loader_range_passes(tmp_path):
    log = _write(
        tmp_path / "run.log",
        [
            "steps=1000 batch=8 seq=1024 lr=0.001",
            "checkpoint: lfb_bias range [-1.900, +2.000] over 1 module(s)",
        ],
    )
    assert _row(inspect(log), "lfb divergence").status == PASS


def test_in_training_counters_catch_the_collapse_without_a_checkpoint_line(tmp_path):
    rows = [
        {
            "event": "run",
            "steps": 1000,
            "batch": 8,
            "seq_len": 1024,
            "grad_accum": 1,
            "bytes_per_step": 8192,
            "lr": 1e-3,
        },
        {
            "event": "log",
            "step": 100,
            "loss": 2.4,
            "lr": 1e-3,
            "bytes_seen": 819200,
            "counters": {
                "moe_route_n": 1,
                "moe_route_live": 9,
                "moe_route_max_x1000": 500,
                "moe_route_top10_x1000": 1000,
            },
        },
    ]
    check = _row(inspect(_jsonl(tmp_path, rows), _snapshot(tmp_path)), "lfb divergence")
    assert check.status == FAIL
    assert "top-10 100%" in check.evidence


# ---------------------------------------------------------------------------
# 3. dead experts
# ---------------------------------------------------------------------------


def test_dead_experts_are_counted_against_the_bank(tmp_path):
    rows = [
        {
            "event": "run",
            "steps": 1000,
            "batch": 8,
            "seq_len": 1024,
            "grad_accum": 1,
            "bytes_per_step": 8192,
            "lr": 1e-3,
        },
        {
            "event": "log",
            "step": 100,
            "loss": 2.4,
            "lr": 1e-3,
            "bytes_seen": 819200,
            "counters": {"moe_route_n": 2, "moe_route_live": 18, "moe_route_max_x1000": 500},
        },
    ]
    check = _row(inspect(_jsonl(tmp_path, rows), _snapshot(tmp_path)), "dead experts")
    assert check.status == FAIL
    assert "9.0/512" in check.evidence


def test_a_healthy_bank_passes(tmp_path):
    rows = [
        {"event": "run", "steps": 1000, "batch": 8, "seq_len": 1024, "grad_accum": 1},
        {
            "event": "log",
            "step": 100,
            "loss": 2.4,
            "lr": 5e-4,
            "counters": {"moe_route_n": 1, "moe_route_live": 500, "moe_route_max_x1000": 20},
        },
    ]
    check = _row(inspect(_jsonl(tmp_path, rows), _snapshot(tmp_path)), "dead experts")
    assert check.status == PASS
    assert "500.0/512" in check.evidence


# ---------------------------------------------------------------------------
# 4. LR frozen / zero
# ---------------------------------------------------------------------------


def test_an_lr_that_stops_moving_inside_the_decay_is_a_failure(tmp_path):
    snap = _snapshot(
        tmp_path,
        steps=1000,
        batch=1,
        seq_len=100,
        warmup_tokens=0,
        decay_frac=0.5,
    )
    log = _write(
        tmp_path / "run.log",
        [
            "steps=1000 batch=1 seq=100 lr=0.001",
            "step=500 loss=2.0000 lr=9.000e-04",
            "step=600 loss=1.9900 lr=8.000e-04",
            "step=700 loss=1.9800 lr=8.000e-04",
            "step=800 loss=1.9700 lr=8.000e-04",
            "step=900 loss=1.9600 lr=3.000e-04",
        ],
    )
    check = _row(inspect(log, snap), "lr schedule")
    assert check.status == FAIL
    assert "steps 600..800" in check.evidence


def test_zero_lr_before_the_end_is_a_failure(tmp_path):
    log = _write(
        tmp_path / "run.log",
        [
            "steps=1000 batch=1 seq=100 lr=0.001",
            "step=100 loss=2.0000 lr=1.000e-03",
            "step=200 loss=2.0000 lr=0.000e+00",
            "step=300 loss=2.0000 lr=0.000e+00",
        ],
    )
    check = _row(inspect(log), "lr schedule")
    assert check.status == FAIL
    assert "step 200" in check.evidence


def test_a_full_cooldown_to_zero_at_the_last_step_is_expected(tmp_path):
    snap = _snapshot(tmp_path, steps=1000, batch=1, seq_len=100, min_lr_frac=0.0)
    log = _write(
        tmp_path / "run.log",
        [
            "steps=1000 batch=1 seq=100 lr=0.001",
            "step=500 loss=2.0 lr=5.000e-04",
            "step=1000 loss=1.8 lr=0.000e+00",
        ],
    )
    check = _row(inspect(log, snap), "lr schedule")
    assert check.status == PASS
    assert "cooldown" in check.evidence


# ---------------------------------------------------------------------------
# 5. dataloader stall
# ---------------------------------------------------------------------------


def test_a_step_that_consumed_less_than_its_window_is_a_stall(tmp_path):
    rows = [
        {
            "event": "run",
            "steps": 1000,
            "batch": 1,
            "seq_len": 1024,
            "grad_accum": 1,
            "bytes_per_step": 1024,
            "lr": 5e-4,
        },
        {"event": "log", "step": 10, "loss": 2.4, "lr": 5e-4, "bytes_seen": 10240},
        {"event": "log", "step": 20, "loss": 2.3, "lr": 4e-4, "bytes_seen": 19456},
    ]
    check = _row(inspect(_jsonl(tmp_path, rows)), "dataloader stall")
    assert check.status == FAIL
    assert "step 20" in check.evidence and "20,480" in check.evidence


def test_a_matching_byte_curve_passes(tmp_path):
    rows = [
        {"event": "run", "steps": 100, "batch": 8, "seq_len": 1024, "grad_accum": 1}
        | {"bytes_per_step": 8192, "lr": 3e-4},
        {"event": "log", "step": 10, "loss": 2.4, "lr": 3e-4, "bytes_seen": 81920},
        {"event": "log", "step": 20, "loss": 2.3, "lr": 2e-4, "bytes_seen": 163840},
    ]
    check = _row(inspect(_jsonl(tmp_path, rows)), "dataloader stall")
    assert check.status == PASS


# ---------------------------------------------------------------------------
# 6. eval window overlap
# ---------------------------------------------------------------------------


def test_an_eval_file_that_is_a_training_source_is_not_held_out(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"\x00" * 16)
    plan = tmp_path / "mix.toml"
    plan.write_text(f'[sources]\nweb = "{source}"\n', encoding="utf-8")
    snap = _snapshot(tmp_path, mix_plan=str(plan), eval_data=str(source))
    log = _write(
        tmp_path / "run.log",
        [
            "steps=1000 batch=8 seq_len=1024 lr=0.001",
            "eval step=500 off=0 bpb_fin=1.900 bpb_mean=1.900 fin=1.3170 loss=1.3457 "
            "over 8192 B (1 x 8 x 1024) engram=0/0",
        ],
    )
    check = _row(inspect(log, snap), "eval window overlap")
    assert check.status == FAIL
    assert "not held out" in check.evidence


# ---------------------------------------------------------------------------
# 7. checkpoint at the end
# ---------------------------------------------------------------------------


def test_a_run_that_finished_without_its_final_checkpoint_fails(tmp_path):
    snap = _snapshot(tmp_path, ckpt_every=500)
    log = _write(
        tmp_path / "run.log",
        [
            "steps=1000 batch=8 seq=1024 lr=0.001",
            "step=500 loss=2.0 lr=5.000e-04",
            "ckpt step=500 -> /tmp/run/run.pt (/tmp/run/run.config.toml)",
            "step=1000 loss=1.8 lr=1.000e-04",
            "done steps=1000 loss=1.8",
        ],
    )
    check = _row(inspect(log, snap), "checkpoint")
    assert check.status == FAIL
    assert "last `ckpt step=` is 500" in check.evidence


def test_the_jsonl_sink_has_no_checkpoint_event_so_it_is_a_skip(tmp_path):
    rows = [
        {"event": "run", "steps": 1000, "batch": 8, "seq_len": 1024, "grad_accum": 1},
        {"event": "log", "step": 100, "loss": 2.4, "lr": 3e-4, "bytes_seen": 819200},
    ]
    snap = _snapshot(tmp_path, ckpt_every=50)
    check = _row(inspect(_jsonl(tmp_path, rows), snap), "checkpoint")
    assert check.status == SKIP
    assert "no checkpoint event" in check.evidence


# ---------------------------------------------------------------------------
# the clean run: no FAIL, and exit 0 from the CLI
# ---------------------------------------------------------------------------


def _clean_log(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "clean.log",
        [
            "╭───────────────────────────────────────────────────────────╮",
            "│ small · 9,476,993 params · device cuda                     │",
            "╰─────────────── steps=1000 batch=1 seq=100 lr=0.001 opt=mix ─╯",
            "step=500 loss=2.5000 lr=1.000e-03 kda=off,n=0 "
            "moe_route_n=1 moe_route_live=8 moe_route_max_x1000=125 "
            "moe_route_top10_x1000=700",
            "step=1000 loss=1.9000 lr=1.000e-04 kda=off,n=0 "
            "moe_route_n=1 moe_route_live=8 moe_route_max_x1000=125 "
            "moe_route_top10_x1000=700",
            "eval step=1000 off=0 bpb_fin=2.000 bpb_mean=2.000 fin=1.3863 "
            "loss=1.3863 over 1000 B (10 x 1 x 100) engram=0/0",
            "ckpt step=1000 -> /tmp/run/run.pt (/tmp/run/run.config.toml)",
            "done steps=1000 loss=1.9000",
        ],
    )


def test_a_clean_run_has_no_fail_row(tmp_path):
    snap = _snapshot(
        tmp_path,
        n_experts=8,
        steps=1000,
        batch=1,
        seq_len=100,
        decay_frac=0.1,
        ckpt_every=500,
    )
    checks = inspect(_clean_log(tmp_path), snap)
    failed = [check for check in checks if check.status == FAIL]
    assert not failed, [(c.name, c.evidence) for c in failed]
    # The rows that CAN run on this log must actively pass, not merely not fail.
    assert _row(checks, "loss finite").status == PASS
    assert _row(checks, "dead experts").status == PASS
    assert _row(checks, "lr schedule").status == PASS
    assert _row(checks, "eval window overlap").status == PASS
    assert _row(checks, "checkpoint").status == PASS


def test_the_cli_prints_json_and_exits_one_on_a_fail(tmp_path):
    snap = _snapshot(tmp_path)
    log = _write(
        tmp_path / "run.log",
        [
            "steps=1000 batch=8 seq=1024 lr=0.001",
            "step=100 loss=2.5 lr=1.000e-03",
            "step=200 loss=nan lr=9.000e-04",
        ],
    )
    result = runner.invoke(app, ["diagnose", str(log), "--snapshot", str(snap), "--json"])
    assert result.exit_code == 1, result.output
    report = json.loads(result.output)
    assert report["fail"] >= 1
    assert any(row["check"] == "loss finite" and row["status"] == FAIL for row in report["checks"])


def test_the_cli_exits_zero_on_a_clean_run(tmp_path):
    snap = _snapshot(
        tmp_path,
        n_experts=8,
        steps=1000,
        batch=1,
        seq_len=100,
        decay_frac=0.1,
        ckpt_every=500,
    )
    result = runner.invoke(
        app, ["diagnose", str(_clean_log(tmp_path)), "--snapshot", str(snap), "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["fail"] == 0


def test_a_cadence_that_does_not_divide_the_run_is_not_behind(tmp_path):
    """`steps=8000`, `ckpt_every=3000` owes its last file at 6000, not 8000."""
    snap = _snapshot(tmp_path, steps=8000, batch=1, seq_len=100, ckpt_every=3000)
    log = _write(
        tmp_path / "run.log",
        [
            "steps=8000 batch=1 seq=100 lr=0.001",
            "step=6000 loss=2.0 lr=1.000e-04",
            "ckpt step=6000 -> /tmp/run/run.pt (/tmp/run/run.config.toml)",
            "step=8000 loss=1.9 lr=1.000e-04",
            "done steps=8000 loss=1.9",
        ],
    )
    assert _row(inspect(log, snap), "checkpoint").status == PASS


def test_a_due_checkpoint_with_no_line_at_all_is_a_failure(tmp_path):
    snap = _snapshot(tmp_path, steps=1000, batch=1, seq_len=100, ckpt_every=1000)
    log = _write(
        tmp_path / "run.log",
        [
            "steps=1000 batch=1 seq=100 lr=0.001",
            "step=1000 loss=2.0 lr=1.000e-04",
            "done steps=1000 loss=2.0",
        ],
    )
    check = _row(inspect(log, snap), "checkpoint")
    assert check.status == FAIL
    assert "never written" in check.evidence
