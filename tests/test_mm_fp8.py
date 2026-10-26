"""The fp8 2-D compute lane: `TrainConfig.mm_dtype`, `tsct.mm_mixed`/`mm_fp8`.

Four claims, each with the mark it earns (ADR-0011/0019):

* **the default path cannot move** — sha256 of a fixed-seed 10-step `train()` on
  the tiny preset, captured BEFORE the lane existed
  (`/tmp/opencode/capture_mm_fp8.py`, the `test_opt_graph.py` house pattern;
  it reproduced byte-identically in two processes and agrees with that file's
  own pin, so the harness is cross-checked). A re-transcription can drift; a
  hash cannot. If this goes red after an edit, that edit moved the bf16 path.
* **the key reaches the compute path** — a spy on `tsct.mm_fp8` proves the
  trainer's `mm_dtype` reached the seam, and the same spy proves the lane is NOT
  entered under the default (`bf16`).
* **a bad value is LOUD**, at config load and at the setter (the seam is
  reachable from a script that never went through `load_run`).
* **an unsupported call falls back COUNTED and bit-exactly** — the counter, the
  reason, the process-local tally, and `torch.equal` against the bf16 line.

The GPU gates at the bottom are `skipif`'d on the repo's preflight
(`pgrep -f "dormouse trai[n]"` + `nvidia-smi`, `test_ns_triton.py`'s
`_card_reason`): the fp8 lane is the CUDA arm, and the card is busy with a
training run, so they are WRITTEN and SKIPPED with the reason printed — never
run against a live run. On CPU the lane cannot run at all (`_scaled_mm`'s e4m3
kernels are the CUDA path), which is itself one of the fallback reasons the CPU
tests assert.
"""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch
from conftest import tiny_cfg, tiny_run

from dormouse.config import RunConfig, load_run
from dormouse.instrument import Counters, LoudError
from dormouse.model import tsct
from dormouse.model.model import DormouseModel
from dormouse.model.tsct import mm_fp8, mm_fp8_shape_ok, mm_mixed, set_mm_dtype
from dormouse.train import train

#: `/tmp/opencode/capture_mm_fp8.py`, run against the tree BEFORE the lane
#: landed, twice (two processes, identical output): seed 0, tiny preset, 10
#: steps, lr 0.01, sha256 of every parameter's bytes in NAME order.
_PRE_CHANGE_TRAIN_SHA = "b4d2aa542e6efd76b4fb1c6560544e1aad7bd72a920206503d0a347d64c11966"


@pytest.fixture(autouse=True)
def _bf16_after_each():
    """The compute mode is process-wide (`tf32`'s shape): leave it at the default."""
    set_mm_dtype("bf16")
    yield
    set_mm_dtype("bf16")


# --- the default path is pinned ---------------------------------------------


@pytest.mark.local_pin
def test_the_default_path_is_byte_identical():
    """`mm_dtype="bf16"` must not have moved — the pre-lane hash, re-run."""
    with tempfile.TemporaryDirectory() as td:
        torch.manual_seed(0)
        run = tiny_run(Path(td), steps=10, log_every=5, lr=0.01)
        assert run.train.mm_dtype == "bf16"
        model = DormouseModel(run.model)
        train(model, run, log=lambda *_: None)
        h = hashlib.sha256()
        for _, p in sorted(model.named_parameters()):
            h.update(p.detach().numpy().tobytes())
    assert h.hexdigest() == _PRE_CHANGE_TRAIN_SHA


# --- the key reaches the compute path (the wiring claim) ---------------------


def test_the_key_reaches_the_compute_path(monkeypatch, tmp_path):
    """A spy on the lane proves `train.mm_dtype` reached the seam.

    The spy DELEGATES to the real `mm_fp8`, so the run is the real run: on CPU
    every call falls back (the lane's own device rule), which is exactly what
    the counter below is for.
    """
    seen: list[tuple[tuple[int, ...], tuple[int, ...], torch.dtype]] = []
    real = tsct.mm_fp8

    def spy(x, weight, dt, counters=None):
        seen.append((tuple(x.shape), tuple(weight.shape), dt))
        return real(x, weight, dt, counters)

    monkeypatch.setattr(tsct, "mm_fp8", spy)
    run = tiny_run(tmp_path, steps=2, log_every=1, mm_dtype="fp8")
    # bf16 compute: the lane's own dtype precondition, so the spy sees the real
    # `dt` a bf16 run hands the seam (the tiny preset is fp32 by default).
    from dataclasses import replace

    run = replace(run, model=tiny_cfg(bf16=True))
    rep = train(DormouseModel(run.model), run, log=lambda *_: None)

    assert tsct.mm_dtype() == "fp8", "the trainer did not apply the key"
    assert seen, "mm_dtype=fp8 never reached the seam"
    assert all(len(s[0]) == 2 and len(s[1]) == 2 for s in seen), "the lane is 2-D only"
    assert {s[2] for s in seen} == {torch.bfloat16}, "the lane is the bf16-compute lane"
    # CPU: every call is a COUNTED fallback, and the log line carries it.
    assert rep["counters"]["mm_fp8_fallback"] == len(seen)
    assert "mm_fp8_gemm" not in rep["counters"]


def test_the_lane_is_not_entered_under_the_default(monkeypatch, tmp_path):
    """The negative control: with `bf16` the seam is never asked at all."""
    called: list[int] = []
    monkeypatch.setattr(tsct, "mm_fp8", lambda *a, **k: called.append(1))
    run = tiny_run(tmp_path, steps=1, log_every=1)
    train(DormouseModel(run.model), run, log=lambda *_: None)
    assert called == []


def test_the_trainer_says_the_lane_is_forward_only(tmp_path):
    """`fp8` in a training run is a disclosed hybrid: the log line says so."""
    lines: list[str] = []
    run = tiny_run(tmp_path, steps=1, log_every=1, mm_dtype="fp8")
    train(DormouseModel(run.model), run, log=lines.append)
    assert any("mm_dtype=fp8" in ln and "has NO backward" in ln for ln in lines)


# --- config: LOUD, pinned, resume-safe ---------------------------------------


def test_a_bad_value_is_loud_at_load_and_at_the_setter():
    with pytest.raises(LoudError, match="mm_dtype"):
        load_run("small", train_overrides={"mm_dtype": "fp16"})
    with pytest.raises(LoudError, match="mm_dtype"):
        set_mm_dtype("fp16")
    assert tsct.mm_dtype() == "bf16", "a refused setter must not have moved the mode"


def test_the_key_round_trips_and_a_flip_is_a_config_diff():
    run = load_run("small", train_overrides={"steps": 5, "ckpt_name": "x", "mm_dtype": "fp8"})
    assert run.train.mm_dtype == "fp8"
    back = RunConfig.from_snapshot(run.snapshot_toml())
    assert back.diff_keys(run) == [] and back.train.mm_dtype == "fp8"

    from dataclasses import replace

    bf16 = replace(run, train=replace(run.train, mm_dtype="bf16"))
    assert bf16.diff_keys(run) == ["train.mm_dtype"]

    # An OLD snapshot (written before the key existed) fills the default and is
    # still a legal resume: the key must not become a spurious diff.
    text = "\n".join(ln for ln in run.snapshot_toml().splitlines() if "mm_dtype" not in ln)
    old = RunConfig.from_snapshot(text)
    assert old.train.mm_dtype == "bf16"
    assert old.diff_keys(bf16) == []


# --- the COUNTED fallback ----------------------------------------------------


@pytest.mark.parametrize(
    "xs,ws,reason",
    [
        ((4, 8), (3, 8), "multiple of 16"),  # N=3, the medium router's width
        ((4, 100), (16, 100), "multiple of 16"),  # K=100
        ((4, 3, 16), (32, 16), "non-2-D"),
        ((4, 16), (32, 16), "not CUDA"),
    ],
)
def test_an_unsupported_call_falls_back_counted(xs, ws, reason):
    """Every unsupported call is COUNTED (counter + reason + tally), never silent."""
    set_mm_dtype("fp8")
    x, w = torch.randn(*xs), torch.randn(*ws)
    counters = Counters()
    out = mm_mixed(x, w, torch.bfloat16, counters)
    assert reason in counters.why, getattr(counters, "why", None)
    assert counters["mm_fp8_fallback"] == 1
    assert "mm_fp8_gemm" not in counters
    assert any(reason in k for k in tsct.mm_fp8_fallbacks())
    # The fallback IS the bf16 line: same op, same inputs, bit for bit.
    set_mm_dtype("bf16")
    assert torch.equal(out, mm_mixed(x, w, torch.bfloat16))


def test_a_refused_scaled_mm_falls_back_counted(monkeypatch):
    """An unexpected kernel refusal must not kill a run — the accelerated-arm rule.

    `moe_triton_fallback` and `opt_graph_fallback` are the precedent: a failed
    accelerated arm degrades COUNTED and the run stays on the reference path.
    The preconditions are forced here (this is a CPU test and the lane's device
    rule would stop before the GEMM), which is the repo's own pattern for
    reaching a device-only branch (`test_experts_gpu._padded_reason`).
    """
    monkeypatch.setattr(tsct, "_fp8_reason", lambda *a, **k: None)
    monkeypatch.setattr(
        torch,
        "_scaled_mm",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no kernel image is available")),
    )
    counters = Counters()
    out = mm_fp8(torch.randn(4, 16), torch.randn(32, 16), torch.bfloat16, counters)
    assert out is None
    assert counters["mm_fp8_fallback"] == 1
    assert "refused the call" in counters.why and "no kernel image" in counters.why


def test_a_non_contracting_pair_is_counted_before_anything_runs():
    """The reason list is the lane's contract; `mm_fp8` alone can be asked.

    `mm_mixed` would then run the bf16 line, which raises on this pair — the
    same raise the caller would have had without the lane, so the fallback is
    not hiding anything; the point here is that the lane's OWN answer is a
    COUNTED `None`, not a crash inside `_scaled_mm`.
    """
    set_mm_dtype("fp8")
    c = Counters()
    assert mm_fp8(torch.randn(4, 16), torch.randn(16, 8), torch.bfloat16, c) is None
    assert "do not contract" in c.why
    assert c["mm_fp8_fallback"] == 1


def test_the_fp32_compute_dtype_falls_back_counted():
    """A `bf16 = false` run has no fp8 lane: the out_dtype is bf16/fp16 only."""
    set_mm_dtype("fp8")
    counters = Counters()
    out = mm_mixed(torch.randn(4, 16), torch.randn(32, 16), torch.float32, counters)
    assert "compute dtype is fp32" in counters.why
    assert out.dtype == torch.float32


def test_a_call_without_counters_still_cannot_be_silent():
    """The module tally is the unconditional half of the mark (ADR-0019)."""
    set_mm_dtype("fp8")
    before = tsct.mm_fp8_fallbacks().get("not CUDA (the lane is the e4m3 kernel path)", 0)
    mm_mixed(torch.randn(4, 16), torch.randn(32, 16), torch.bfloat16)  # no counters
    after = tsct.mm_fp8_fallbacks()["not CUDA (the lane is the e4m3 kernel path)"]
    assert after == before + 1


def test_the_lane_is_never_entered_outside_fp8_mode():
    """`mm_fp8` is the lane; `mm_mixed` is the ONLY caller that decides by mode."""
    counters = Counters()
    out = mm_mixed(torch.randn(4, 16), torch.randn(32, 16), torch.bfloat16, counters)
    assert dict(counters) == {} and getattr(counters, "why", None) is None
    assert out.dtype == torch.bfloat16


# --- the conversion itself (CPU-checkable) -----------------------------------


def test_the_conversion_is_the_documented_per_tensor_e4m3_grid():
    """`x ≈ q · scale`, `scale = amax/448`, and the cast is onto the real grid.

    The bound is the format's own: e4m3 has 3 mantissa bits, so the grid step at
    magnitude `m` is `m/8` and RNE lands within half of it — asserted here as
    `err <= amax/16 + 0` on the element that saturates, and `q`'s max is 448
    (the scale's own divisor) for a non-degenerate operand.
    """
    torch.manual_seed(0)
    x = torch.randn(64, 128) * 3.0
    q, s = tsct.fp8_quantize_2d(x, torch.bfloat16)
    assert q.dtype == torch.float8_e4m3fn and q.shape == x.shape
    assert s.dtype == torch.float32 and s.numel() == 1
    err = (x - q.float() * s).abs()
    assert float(q.float().abs().max()) <= 448.0
    assert float(err.max()) <= float(x.abs().max()) / 16 + 1e-6
    # The all-zero operand: clamp_min keeps the scale finite and the product 0
    # (an unclamped 0/0 would be NaN).
    zq, zs = tsct.fp8_quantize_2d(torch.zeros(8, 16), torch.bfloat16)
    assert torch.isfinite(zs).all() and float(zq.float().abs().max()) == 0.0


def test_the_shape_rule_is_the_16_divisibility():
    assert mm_fp8_shape_ok(1024, 64) and mm_fp8_shape_ok(64, 2730) is False
    assert not mm_fp8_shape_ok(100, 1024) and not mm_fp8_shape_ok(1024, 3)
    # The lane's own reason agrees with the shape helper (one rule, two readers).
    set_mm_dtype("fp8")
    c = Counters()
    mm_mixed(torch.randn(4, 1024), torch.randn(64, 1024), torch.bfloat16, c)
    assert "not CUDA" in c.why, "1024x64 is shape-legal: the CPU reason must fire"


# --- GPU gates: written, skipped, reason printed -----------------------------


def _card_reason() -> str | None:
    """`None` when the card may be used, else why not — the repo's preflight.

    `test_ns_triton.py`'s pattern: `pgrep` + `nvidia-smi`, so the check itself
    never builds a CUDA context on a card that is in use (AGENTS §1.5).
    """
    if not torch.cuda.is_available():
        return "no CUDA device"
    try:
        busy = subprocess.run(
            ["pgrep", "-f", "dormouse trai[n]"], capture_output=True, text=True, timeout=15
        ).stdout.strip()
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001 - a gate's precondition, not a run's
        return f"cannot check the card ({e})"
    if busy:
        return f"a live training run holds the card (pids {busy.replace(chr(10), ' ')})"
    mib = int(used.split()[0]) if used else -1
    if mib >= 1024:
        return f"the card already holds {mib} MiB"
    return None


_GPU_REASON = _card_reason()
CUDA = pytest.mark.skipif(_GPU_REASON is not None, reason=_GPU_REASON or "GPU gate")


@CUDA
def test_the_fp8_lane_matches_bf16_within_the_spikes_band():
    """The lane's numerics against the bf16 line, at a real 2-D shape.

    The band is the spike's own measurement (`bench_f_fp8.py`: max abs error
    3.3-4.5 % of the reference's absmax, random-but-realistic weights) with
    headroom to 6 % — e4m3's relative precision is 2^-4 worst case, so this is a
    bound the format can meet and a broken scale cannot.
    """
    set_mm_dtype("fp8")
    dt = torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(5)
    a = torch.randn(512, 1024, device="cuda", dtype=dt, generator=g)
    w = torch.randn(1024, 1024, device="cuda", dtype=dt, generator=g) / 32.0
    with torch.no_grad():
        ref = a @ w.t()
        counters = Counters()
        out = mm_mixed(a, w, dt, counters)
    assert counters["mm_fp8_gemm"] == 1 and "mm_fp8_fallback" not in counters
    err = (out.float() - ref.float()).abs().max() / ref.float().abs().max()
    assert float(err) <= 0.06, f"fp8 error {float(err):.4f} of absmax exceeds the spike's band"


@CUDA
def test_a_no_grad_model_forward_runs_the_lane_and_counts_it():
    """The real forward: `fp8` engages where autograd is off, falls back where not."""
    from conftest import tiny_model

    set_mm_dtype("fp8")
    model = tiny_model(bf16=True, n_experts=2, moe_topk=0, rank=16, shared_rank=16)
    model = model.to("cuda")
    ids = (torch.arange(2 * 64, dtype=torch.long) % 251).view(2, 64).to("cuda")
    counters = Counters()
    with torch.no_grad():
        model(ids, counters=counters)
    assert counters["mm_fp8_gemm"] > 0, "the lane never engaged in a no-grad forward"
    assert "mm_fp8_fallback" in counters, (
        "the tiny shapes include non-16 ones (the controller's N=3): the fallback must be counted"
    )


@CUDA
def test_a_grad_forward_falls_back_counted_and_the_backward_still_runs():
    """The training path: the lane stands down, the bf16 backward works.

    This is the measurement that makes `mm_dtype=fp8` a disclosed hybrid rather
    than a crash: torch 2.13's `_scaled_mm` has no derivative, so a forward
    whose operands require grad must take the COUNTED bf16 line — and the
    backward must then run to completion (grads finite, non-zero).
    """
    from conftest import tiny_model

    set_mm_dtype("fp8")
    model = tiny_model(bf16=True, n_experts=2, moe_topk=0, rank=16, shared_rank=16).to("cuda")
    ids = (torch.arange(2 * 64, dtype=torch.long) % 251).view(2, 64).to("cuda")
    counters = Counters()
    logits, loss, _ = model(ids, targets=(ids + 1) % 251, counters=counters)
    assert "mm_fp8_gemm" not in counters, "a grad forward must not enter the lane"
    assert counters["mm_fp8_fallback"] > 0
    assert "autograd is on" in counters.why
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)
