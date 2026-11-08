"""The compile bench's gates: the arithmetic, the counted break, the refusals.

`scripts/bench_compile.py` is a tool for a free card, so what a CPU suite can
gate is (a) the arithmetic it prints — the 47.7 TFLOP/s denominator and the
flagship's 137.42 MFLOP/byte bill, so a wrong denominator cannot pass — (b) the
program it runs, end to end on the tiny preset with dynamo's real (backend-
independent) break machinery, and (c) the two refusals: a busy card is exit 2
with the pid named, and a VRAM projection past the ceiling is exit 3.

Every gate here was shown RED on a scratch revert (the numbers in the asserts
are the measured ones, not preferences):

* `PEAK_TFLOPS` 47.7 -> 45.7 changes MFU from 17.83 % to 18.6 % (test 1).
* removing `@torch.compiler.disable` from `SparseMoE._naive_dispatch` goes from
  2 breaks / 3 graphs to 5 breaks / 6 graphs on one tiny dispatch (test 2).
* `preflight`'s busy branch -> exit 0 changes the exit code (test 3).
* dropping the projection's ceiling check lets the batch loop continue (test 4).
* treating parity as a warning instead of `SystemExit` lets a divergent arm
  through (test 5).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest
import torch
from conftest import tiny_cfg

from dormouse.config import ModelConfig
from dormouse.model import DormouseModel
from dormouse.model.experts import SparseMoE

# dynamo/inductor work; see the `slow` marker in pyproject.
pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    """Load `scripts/<name>.py` by path — `scripts/` is not a package.

    Registered in `sys.modules` first: `@dataclasses.dataclass` looks its module
    up by `__module__` while decorating, and a by-path module is not there yet.
    """
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bench = _load("bench_compile")
lane = _load("_lane")


# --- 1. the arithmetic -----------------------------------------------------


def test_mfu_arithmetic_is_pinned() -> None:
    """47.7 TFLOP/s, 137.42 MFLOP/byte, 24 576 B/step — the brief's three numbers.

    The measured point: the flagship's step is 3.377 TFLOP of work, and the
    recorded run does it at 8.5 TFLOP/s = 17.9 % MFU. A different peak or a
    different bill is a different claim, and this test is what makes it fail.
    """
    assert bench.PEAK_TFLOPS == 47.7
    assert bench.FLAGSHIP_FLOPS_PER_BYTE == pytest.approx(137.42e6)
    assert bench.FLAGSHIP_BYTES_PER_STEP == 24_576
    assert bench.bytes_per_step(8, 1024, 3) == 24_576

    nbytes = bench.bytes_per_step(8, 1024, 3)
    tflop = bench.step_flops(bench.FLAGSHIP_FLOPS_PER_BYTE, nbytes) / 1e12
    assert tflop == pytest.approx(3.377, rel=1e-3)

    tflops = bench.step_tflops(bench.FLAGSHIP_FLOPS_PER_BYTE, nbytes, 397.1)
    assert tflops == pytest.approx(8.50, rel=1e-2)
    assert bench.mfu(tflops) == pytest.approx(0.1782, rel=1e-2)
    # The default IS the pinned denominator: a caller cannot silently pass 45.7.
    assert bench.mfu(tflops) == bench.mfu(tflops, bench.PEAK_TFLOPS)
    # And a wrong denominator cannot land inside the tolerance above: the
    # 5060 Ti's 45.7 is 4.4 % off, the 5070 Ti's spec 88.7 is 46 % off.
    assert abs(bench.mfu(tflops, 45.7) - 0.1782) > 3e-3
    assert abs(bench.mfu(tflops, 88.7) - 0.1782) > 0.05


def test_vram_projection_is_linear_in_the_micro_batch() -> None:
    """Weights+state are the floor; activations grow with b (the slope is the
    activation part). 4.0 GiB steady + 2.0 GiB of activations at b=8 projects
    to 12.0 GiB at b=32 — and to 30.0 GiB from a 6/12 GiB point, which is the
    ceiling refusal the batch loop must take."""
    assert bench.project_vram(4.0, 6.0, 8, 32) == pytest.approx(12.0)
    assert bench.project_vram(6.0, 12.0, 8, 32) == pytest.approx(30.0)
    assert bench.project_vram(6.0, 12.0, 8, 32) > bench.VRAM_CEILING_GIB
    assert bench.project_vram(4.0, 6.0, 8, 32) < bench.VRAM_CEILING_GIB
    assert bench.grad_accum_for(24_576, 8, 1024) == 3
    assert bench.grad_accum_for(24_576, 16, 1024) == 2  # 1.5 does not exist
    assert bench.grad_accum_for(24_576, 32, 1024) == 1


def test_parity_is_a_hard_error_not_a_warning() -> None:
    ok = {"arm": "default", "logits": 1e-6, "loss": 1e-7, "params": 1e-7, "param": "w"}
    assert bench.parity_failures(ok) == []
    for field, bad in (("logits", 2e-4), ("loss", 1e-4 * 2), ("params", 2e-6)):
        case = dict(ok, **{field: bad})
        assert bench.parity_failures(case), f"{field}={bad} must fail the gate"
    assert bench.relative_error(torch.zeros(2), torch.zeros(2)) == 0.0
    assert bench.relative_error(torch.ones(2), torch.full((2,), 2.0)) == pytest.approx(0.5)


# --- 2. the counted break --------------------------------------------------


def test_the_reference_dispatch_is_a_counted_break() -> None:
    """`_naive_dispatch` is disabled ON PURPOSE, and the break count says so.

    Measured on one `SparseMoE(tiny_cfg())` dispatch: the raw loop is 5 breaks /
    6 graphs (`Dynamic shape operator` at `torch.unique(...)`), the disabled
    region is 2 breaks / 3 graphs whose reason NAMES the decorator. The gate is
    two-sided: the decorator is there (a silent break would print a dynamic-shape
    reason instead), and the eager result is bit-identical.
    """
    assert getattr(SparseMoE._naive_dispatch, "__wrapped__", None) is not None, (
        "a silent break is a defect: the reference dispatch must be a COUNTED "
        "`torch.compiler.disable` break"
    )
    moe = SparseMoE(tiny_cfg())
    x = torch.randn(32, 64)
    want = moe(x).detach().clone()

    torch._dynamo.reset()
    disabled = torch._dynamo.explain(moe)(x)
    reasons = [g.reason.splitlines()[0] for g in disabled.break_reasons]
    assert reasons and all("torch.compiler.disable" in r for r in reasons), reasons

    SparseMoE._naive_dispatch = SparseMoE._naive_dispatch.__wrapped__
    try:
        torch._dynamo.reset()
        raw = torch._dynamo.explain(moe)(x)
    finally:
        SparseMoE._naive_dispatch = torch.compiler.disable(SparseMoE._naive_dispatch)
    assert raw.graph_break_count > disabled.graph_break_count
    assert any("Dynamic shape operator" in g.reason for g in raw.break_reasons)
    assert torch.equal(moe(x).detach(), want), "the decorator must not change eager"


# --- 3. the busy-card refusal ----------------------------------------------


def _stub_nvidia_smi(tmp_path: Path, pid: int, mib: int) -> Path:
    """A fake `nvidia-smi` that answers the two queries `preflight` makes."""
    exe = tmp_path / "nvidia-smi"
    exe.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        f'  *query-compute-apps*) echo "{pid}, python, {mib}";;\n'
        f'  *) echo "{mib + 700}, 16303";;\n'
        "esac\n",
        encoding="utf-8",
    )
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


def test_preflight_names_the_pid_of_the_process_holding_the_card(tmp_path: Path) -> None:
    exe = _stub_nvidia_smi(tmp_path, pid=4242, mib=9000)
    old = os.environ["PATH"]
    os.environ["PATH"] = f"{tmp_path}:{old}"
    try:
        why = lane.preflight("cuda")
    finally:
        os.environ["PATH"] = old
        exe.unlink()
    assert why is not None
    assert "4242" in why, f"the pid must be named, got: {why}"


def test_a_busy_card_is_exit_2_not_a_silent_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bench, "preflight", lambda _d: "a compute process holds the card:\n777, python, 8000"
    )
    rc = bench.main(["--device", "cuda", "--preset", "tiny", "--arms", "eager", "--no-parity"])
    assert rc == bench.EXIT_BUSY == 2


# --- 4. the end-to-end run, and the VRAM refusal ---------------------------


def _tiny_args(*extra: str) -> list[str]:
    return [
        "--preset",
        "tiny",
        "--batch",
        "2",
        "--seq",
        "64",
        "--grad-accum",
        "1",
        "--device",
        "cpu",
        "--backend",
        "eager",
        "--flops-per-byte",
        "137.42e6",
        "--no-flops-check",
        "--warm",
        "0",
        "--reps",
        "1",
        *extra,
    ]


def test_the_table_counts_every_break_and_the_eager_arm_is_plain(tmp_path: Path) -> None:
    """End to end on the tiny preset: eager has 0 breaks, the compiled arm has
    more than 0 (COUNTED, never silent), and the eager row's program is the
    trainer's arithmetic."""
    out = tmp_path / "bench.json"
    rc = bench.main(_tiny_args("--arms", "eager,default", "--no-parity", "--json-out", str(out)))
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    rows = {r["arm"]: r for r in data["rows"]}
    assert rows["eager"]["breaks"] == 0 and rows["eager"]["graphs"] == 0
    assert rows["default"]["breaks"] > 0, "a fallback must be COUNTED, never silent"
    assert data["breaks"], "the explain table must name every break"
    assert "experts.py" in data["breaks"][0]["site"], data["breaks"][0]
    assert rows["eager"]["bytes_per_step"] == 128
    assert rows["eager"]["ms"] > 0


def test_the_eager_arm_is_the_trainers_own_step() -> None:
    """No wrapper, no extra arithmetic: `_step` == zero_grad + backward + step."""
    cfg = tiny_cfg()
    model = DormouseModel(cfg).train()
    start = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ids = torch.randint(0, 256, (2, 64))
    run = bench.load_run("tiny", train_overrides={"batch": 2, "seq_len": 64, "grad_accum": 1})

    opt = bench._optimizer(model, run)
    opt.zero_grad(set_to_none=True)
    _logits, loss, _c = model(ids, targets=ids)
    assert loss is not None
    loss.backward()
    opt.step()
    want = {n: p.detach().clone() for n, p in model.named_parameters()}

    model.load_state_dict(start)
    opt = bench._optimizer(model, run)
    bench._step(model, opt, ids, 1)
    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), want[name]), f"{name} moved under the bench's wrapper"


def test_a_vram_projection_past_the_ceiling_refuses() -> None:
    """`--vram-ceiling-gib -1` cannot be met by any projection, so the batch
    loop must stop at the second batch with exit 3 instead of running it."""
    rc = bench.main(
        _tiny_args(
            "--arms",
            "eager",
            "--no-parity",
            "--batch-scale",
            "--batches",
            "2,4",
            "--vram-ceiling-gib",
            "-1",
        )
    )
    assert rc == bench.EXIT_VRAM == 3


# --- 5. the form gate ------------------------------------------------------


def test_the_vanilla_form_is_gated_on_block_form() -> None:
    """Without the field the bench SKIPs and says so; it never guesses a second
    spelling of the flag. With the field it builds the form or names the config's
    own refusal."""
    if "block_form" in {f.name for f in dataclasses.fields(ModelConfig)}:
        pytest.skip("`block_form` has landed: the transformer arm is measurable here")
    with pytest.raises(SystemExit) as e:
        bench.main(_tiny_args("--form", "transformer"))
    assert "block_form" in str(e.value)
    assert "NOTHING WAS MEASURED" in str(e.value)
