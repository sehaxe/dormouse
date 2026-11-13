"""Gates for `scripts/bench_engine.py` / `.sh` — CPU only, no card touched.

What a CPU suite can prove about a GPU benchmark is exactly what must be right
BEFORE it touches the card:

1. the bill. `flops_per_byte` must reproduce the repo's own numbers: the tiny
   config's hand-computed bill at two windows (the T-term is not a constant),
   and `flagship3_dense`'s measured 691.37 MFLOP/byte at s1024.
2. the MFU arithmetic. 137.42 MFLOP/B x 24 576 B/step at 397.1 ms is the repo's
   recorded 17.83 % of 47.7 TFLOP/s; a changed denominator fails here.
3. the steady state. `summarize` must use ONLY the last three telemetry
   windows, print their spread, and derive ms/step from the trainer's own
   `bytes_per_step`.
4. the recipe command. The argv carries `recipe_ladder.sh`'s flags (lr, opt,
   muon_lr_mult, mix_plan, doc_masking, no checkpoints) and grows `max_seq_len`
   only when the shape exceeds the config's window.
5. the refusals. `--quick` with no GPU is exit 2 and writes NOTHING; the same
   script with `--dry-run` resolves the plan without a card.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dormouse.config import load_preset

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench_engine.sh"
TINY = ROOT / "configs" / "tiny.toml"
FLAGSHIP = ROOT / "configs" / "flagship3_dense.toml"


def _load() -> object:
    """Load `scripts/bench_engine.py` by path — `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("bench_engine", SCRIPT.with_suffix(".py"))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_engine"] = mod
    spec.loader.exec_module(mod)
    return mod


bench = _load()


# --- 1. the bill -----------------------------------------------------------


def test_flop_ledger_pins_the_repo_numbers() -> None:
    """The tiny bill is hand-computable; the flagship's is MEASURED.

    tiny (KDA, d=64, 2 iterations, 4 experts top-2, rank 8, s64):
        core = 4*L*T*d = 4*2*64*64                 =  32 768 FLOP/B  (fwd)
        head = 2*(d*r + r*V) = 2*(64*8 + 8*256)    =   5 120
        body = 2*active_macs_per_token             = 119 296
        training = 3*(32 768 + 5 120 + 119 296)    = 471 552 FLOP/B
    At s128 the core doubles and nothing else moves: 569 856 FLOP/B.
    `flagship3_dense` at s1024 must equal the pinned 691.37 MFLOP/byte
    (`configs/flagship3_dense.toml`, `docs/reviews/2026-11-07-metrics.md`).
    """
    tiny = load_preset(str(TINY))
    assert bench.flops_per_byte(tiny, 64) == pytest.approx(471_552.0)
    wider = dataclasses.replace(tiny, max_seq_len=128)
    assert bench.flops_per_byte(wider, 128) == pytest.approx(569_856.0)

    flagship = load_preset(str(FLAGSHIP))
    assert bench.flops_per_byte(flagship, 1024) == pytest.approx(691.37e6, rel=1e-4)


def test_mfu_arithmetic_is_pinned() -> None:
    """The recorded point: 137.42 MFLOP/B, 24 576 B/step, 397.1 ms, 17.83 %."""
    nbytes = 8 * 1024 * 3
    tflops = bench.step_tflops(137.42e6, nbytes, 397.1)
    assert tflops == pytest.approx(8.505, rel=1e-3)
    assert bench.mfu(tflops) == pytest.approx(0.1783, abs=2e-4)
    assert bench.mfu(tflops, peak=88.7) == pytest.approx(0.0959, abs=2e-4)


# --- 2. the steady state ---------------------------------------------------


def test_summarize_uses_only_the_last_three_windows() -> None:
    events = [
        {"event": "run", "bytes_per_step": 8192},
        {"event": "log", "step": 5, "tokens_per_s": 10_000.0},  # warmup, must be ignored
        {"event": "log", "step": 10, "tokens_per_s": 50_000.0},
        {"event": "log", "step": 20, "tokens_per_s": 51_000.0},
        {"event": "log", "step": 30, "tokens_per_s": 49_000.0},
    ]
    row = bench.summarize(events, 691.37e6, 47.7)
    assert row["windows"] == 3
    assert row["bytes_per_s"] == pytest.approx(50_000.0)
    assert row["ms_step"] == pytest.approx(1e3 * 8192 / 50_000)
    assert row["spread_pct"] == pytest.approx(4.0)
    assert row["tflops"] == pytest.approx(bench.step_tflops(691.37e6, 8192, row["ms_step"]))
    assert row["mfu"] == pytest.approx(bench.mfu(row["tflops"]))

    empty = bench.summarize([{"event": "run", "bytes_per_step": 8192}], 691.37e6, 47.7)
    assert empty["windows"] == 0 and not (empty["mfu"] == empty["mfu"])


def test_peak_is_read_from_the_wrapper_line() -> None:
    log = "step=50 loss=1.0\n[bench] peak_gib=7.234\n"
    assert bench.parse_peak(log) == pytest.approx(7.234)
    assert bench.parse_peak("CUDA never started") is None


# --- 3. the command and the modes ------------------------------------------


def test_modes_are_the_contract() -> None:
    assert bench.QUICK_STEPS == 50 and len(bench.QUICK_SHAPES) == 3
    assert bench.FULL_STEPS == 200 and len(bench.FULL_SHAPES) == 7
    assert bench.parse_shapes("8x1024, 4X2048") == (
        bench.Shape(8, 1024),
        bench.Shape(4, 2048),
    )
    with pytest.raises(ValueError):
        bench.parse_shapes("8,1024")
    with pytest.raises(SystemExit):
        bench.parser().parse_args([])  # a mode is required
    with pytest.raises(SystemExit):
        bench.parser().parse_args(["--quick", "--full"])
    with pytest.raises(SystemExit):
        bench.parser().parse_args(["--quick", "--peak-tflops", "0"])  # no zero MFU denominator


def test_command_is_the_recipe_and_grows_the_window() -> None:
    base = [
        "--quick",
        "--config",
        str(TINY),
        "--shapes",
        "8x64",
        "--dry-run",
    ]
    args = bench.parser().parse_args(base)
    steps, rows = bench.plan(
        args, python="python", wrapper=Path("w.py"), out_dir=Path("/tmp/x"), stamp="t"
    )
    assert steps == 50
    cmd = rows[0]["cmd"]
    for flag in (
        "steps=50",
        "batch=8",
        "seq_len=64",
        "lr=3e-4",
        "opt=mix",
        "muon_lr_mult=12",
        "doc_masking=false",
        "eval=false",
        "ckpt_every=1000000",
    ):
        assert flag in cmd, f"{flag} missing from the recipe argv"
    assert "max_seq_len=64" not in cmd  # the config already serves this window

    args = bench.parser().parse_args(
        ["--quick", "--config", str(TINY), "--shapes", "8x128", "--fast", "--dry-run"]
    )
    _, rows = bench.plan(
        args, python="python", wrapper=Path("w.py"), out_dir=Path("/tmp/x"), stamp="t"
    )
    cmd = rows[0]["cmd"]
    assert "max_seq_len=128" in cmd
    assert "ns_dtype=bf16" in cmd and "muon_state_dtype=bf16" in cmd
    assert rows[0]["bytes_per_step"] == 8 * 128
    assert rows[0]["flops_per_byte"] == pytest.approx(569_856.0)


def test_markdown_carries_shape_and_ledger() -> None:
    rows = [
        {
            "tag": "b8s1024",
            "batch": 8,
            "seq_len": 1024,
            "steps": 50,
            "bytes_per_step": 8192,
            "ms_step": 163.84,
            "bytes_per_s": 50_000.0,
            "peak_gib": 7.2,
            "flops_per_byte": 691.37e6,
            "tflops": 5.664,
            "mfu": 5.664 / 47.7,
            "spread_pct": 4.0,
            "status": "ok",
        }
    ]
    header = {
        "mode": "quick",
        "config": "configs/flagship3_dense.toml",
        "git": "abc1234",
        "gpu": "RTX 5070 Ti",
        "params": 95_035_200,
        "steps": 50,
        "seed": 1,
        "peak_tflops": 47.7,
        "ledger": bench.LEDGER,
    }
    text = bench.markdown(rows, header)
    assert "| 8×1024 | 50 | 8,192 | 163.8 | 50,000 | 7.20 | 691.4 |" in text
    assert "11.9%" in text  # 5.664 / 47.7, the named denominator
    assert "ledger: mod_ledger.ledger" in text


def test_bad_set_is_a_loud_error_not_a_traceback() -> None:
    rc = bench.main(["--quick", "--dry-run", "--config", str(TINY), "--set", "not_a_key=1"])
    assert rc == 1


# --- 4. the refusals and the CPU plan --------------------------------------


def _run_shell(tmp_path: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    """Run the real `.sh` with no `nvidia-smi`: the CPU/no-GPU box, simulated."""
    empty = tmp_path / "emptybin"
    empty.mkdir()
    env = {
        **os.environ,
        "PATH": str(empty),
        "PY": sys.executable,
        "PYTHONPATH": str(ROOT / "src"),
    }
    return subprocess.run(
        ["/bin/bash", str(SCRIPT), *argv],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_dry_run_needs_no_gpu(tmp_path: Path) -> None:
    proc = _run_shell(
        tmp_path,
        "--quick",
        "--config",
        str(TINY),
        "--shapes",
        "8x64,4x128",
        "--out-dir",
        str(tmp_path / "out"),
        "--dry-run",
    )
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout.strip().splitlines()[-1])
    assert plan["event"] == "dry_run"
    assert plan["steps"] == 50
    assert plan["peak_tflops"] == 47.7
    assert plan["ledger"] == bench.LEDGER
    assert [s["tag"] for s in plan["shapes"]] == ["b8s64", "b4s128"]
    assert all(s["cmd"][3] == "train" for s in plan["shapes"])
    assert not (tmp_path / "out").exists(), "--dry-run must not write a run"


def test_quick_refuses_without_a_gpu(tmp_path: Path) -> None:
    out = tmp_path / "out"
    proc = _run_shell(tmp_path, "--quick", "--config", str(TINY), "--out-dir", str(out))
    assert proc.returncode == bench.EXIT_BUSY
    assert "REFUSED" in proc.stdout
    assert "no CUDA device" in proc.stdout
    assert "nothing was measured" in proc.stdout
    assert not out.exists(), "a refused bench must not leave a result file"


def test_wrapper_runs_the_real_cli_without_a_peak_on_cpu(tmp_path: Path) -> None:
    wrapper = bench.write_wrapper(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(wrapper), "version"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "dormouse" in proc.stdout
    assert "[bench] peak_gib" not in proc.stdout  # CUDA never initialised
