"""The device-capability layer: the policy table, the refusals, the off switch.

Table-driven over SYNTHETIC capability records (sm_75 / 86 / 120 and a
pre-Maxwell 52), because the point of the layer is what it decides for cards
this box does not have — plus one real probe of this box, which asserts only
what is true of any CUDA device and prints what it found.
"""

from __future__ import annotations

import pytest
import torch
from conftest import TINY

from dormouse.config import ModelConfig, RunConfig, TrainConfig
from dormouse.instrument import LoudError
from dormouse.train.device import (
    BF16_MIN_SM,
    COMPUTE_DTYPES,
    DEFAULT_RESERVE,
    DeviceCaps,
    apply_policy,
    cpu_caps,
    probe_device,
    resolve_policy,
)


def caps(sm: int, *, free: int = 8 << 30, total: int = 16 << 30, index: int = 0) -> DeviceCaps:
    """A synthetic capability record: the probe's shape, an sm's semantics."""
    major, minor = divmod(sm, 10)
    return DeviceCaps(
        index=index,
        name=f"synthetic sm_{sm}",
        cc_major=major,
        cc_minor=minor,
        total_mem=total,
        free_mem=free,
        pcie_gen=4,
        pcie_width=8,
    )


def a_run(**model_over) -> RunConfig:
    """A config-only run (nothing instantiated): the apply_policy seam's input."""
    return RunConfig(
        source="device-policy",
        model=ModelConfig(**{**{k: v for k, v in TINY.items()}, **model_over}),
        train=TrainConfig(),
    )


# --- the policy table -------------------------------------------------------


@pytest.mark.parametrize(
    "sm,auto,bf16_ok,fp16_ok,moe,fla",
    [
        (120, "bf16", True, True, True, True),  # this box (5060 Ti)
        (86, "bf16", True, True, True, True),  # Ampere: bf16 arrives
        (80, "bf16", True, True, True, True),  # the bf16 boundary itself
        (75, "fp16", False, True, False, True),  # GTX 1650: fp16, NO bf16
        (70, "fp16", False, True, False, True),  # Volta
        (61, "fp16", False, True, False, True),  # Pascal
        (52, "fp32", False, False, False, False),  # pre-Maxwell: fp32 only
    ],
)
def test_policy_table(sm, auto, bf16_ok, fp16_ok, moe, fla):
    """sm x bf16/fp16 -> the chosen rung, the kernel legality, the counted reason."""
    c = caps(sm)
    assert (c.bf16, c.fp16) == (bf16_ok, fp16_ok), f"sm_{sm} support flags"
    p = resolve_policy(c, "auto")
    assert p.compute_dtype == auto
    assert (p.moe_triton, p.fla_kda) == (moe, fla)
    # The COUNTED sentence names the reading, not just the verdict.
    assert f"bf16={int(bf16_ok)}" in p.reason and f"fp16={int(fp16_ok)}" in p.reason
    line = p.line()
    assert f"sm_{sm}" in line and f"compute={auto}" in line and "budget=" in line
    for dtype, ok in (("bf16", bf16_ok), ("fp16", fp16_ok), ("fp32", True)):
        if ok:
            assert resolve_policy(c, dtype).compute_dtype == dtype
        else:
            with pytest.raises(LoudError) as e:
                resolve_policy(c, dtype)
            msg = str(e.value)
            assert f"sm_{sm}" in msg and f"dtype={dtype}" in msg and "Escape" in msg


def test_a_1650_asked_for_bf16_is_loud_and_names_the_ladder():
    """The refusal that makes the fleet story honest: no silent bf16->fp16."""
    with pytest.raises(LoudError) as e:
        resolve_policy(caps(75), "bf16")
    msg = str(e.value)
    assert f"sm_{BF16_MIN_SM}" in msg, "the refusal names the sm bf16 starts at"
    assert "dtype=auto" in msg and "dtype=fp16" in msg, "the escape is named"
    assert "different experiment" in msg


def test_unknown_and_off_requests_are_loud():
    with pytest.raises(LoudError) as e:
        resolve_policy(caps(120), "fp8")
    assert "Known:" in str(e.value)
    with pytest.raises(LoudError) as e:
        resolve_policy(caps(120), "none")
    assert "OFF switch" in str(e.value)


def test_memory_budget_is_free_minus_the_reserve():
    p = resolve_policy(caps(120, free=10 << 30), "auto")
    assert p.memory_budget == (10 << 30) - DEFAULT_RESERVE
    with pytest.raises(LoudError) as e:
        resolve_policy(caps(120, free=1 << 29), "auto")
    msg = str(e.value)
    assert "nothing fits" in msg and "one heavy thing at a time" in msg


def test_unreadable_free_vram_is_not_a_number():
    """`free_mem = -1` (the card refuses a second context) -> no budget, said so."""
    p = resolve_policy(caps(120, free=-1), "auto")
    assert p.memory_budget is None
    assert "budget=unknown" in p.reason and "unreadable" in p.reason
    assert "free=unknown" in p.line()


def test_cpu_caps_is_bf16_and_unbudgeted():
    """The CPU arm: no VRAM to budget, the ladder is advisory (both rungs served)."""
    c = cpu_caps()
    assert not c.is_cuda and c.sm == 0
    assert c.bf16 and c.fp16
    p = resolve_policy(c, "auto")
    assert p.compute_dtype == "bf16" and p.memory_budget is None
    assert "host" in p.line()


# --- applying a policy to a run ---------------------------------------------


def test_apply_policy_maps_the_rung_onto_the_bf16_flag():
    run = a_run(dtype="auto", bf16=False)
    assert apply_policy(run, [resolve_policy(caps(120), "auto")]).model.bf16 is True
    assert apply_policy(run, [resolve_policy(caps(75), "fp32")]).model.bf16 is False


def test_apply_policy_refuses_fp16_with_the_f16_plan_named():
    """A 1650 under `dtype=auto` resolves fp16 — and the PORT has no fp16 forward.

    The refusal is the honest state of the fleet story: until the fp16 forward
    lands (docs/architecture/f16-plan.md is storage-only so far), the weak card
    runs fp32 or does not run.
    """
    run = a_run(dtype="auto")
    with pytest.raises(LoudError) as e:
        apply_policy(run, [resolve_policy(caps(75), "auto")])
    msg = str(e.value)
    assert "no fp16 path" in msg and "f16-plan.md" in msg and "dtype=fp32" in msg


def test_apply_policy_refuses_triton_moe_without_bf16():
    run = a_run(dtype="fp32", moe_kernel="triton")
    with pytest.raises(LoudError) as e:
        apply_policy(run, [resolve_policy(caps(120), "fp32")])
    msg = str(e.value)
    assert "moe_kernel='triton'" in msg and "bf16 operands" in msg and "padded" in msg


def test_apply_policy_refuses_a_heterogeneous_fleet_without_a_rank():
    policies = [resolve_policy(caps(120), "auto"), resolve_policy(caps(75), "auto")]
    with pytest.raises(LoudError) as e:
        apply_policy(a_run(dtype="auto"), policies)
    assert "heterogeneous" in str(e.value)
    # ...and serves it per rank, which is what the dist entry does.
    assert apply_policy(a_run(dtype="auto"), policies, rank=0).model.bf16 is True


def test_apply_policy_needs_a_real_request():
    with pytest.raises(AssertionError):
        apply_policy(a_run(dtype="none"), [resolve_policy(caps(120), "auto")])


# --- the off switch ---------------------------------------------------------


def test_dtype_none_never_probes_and_old_snapshots_round_trip(monkeypatch):
    """`dtype="none"` (the default) is a no-op: no probe, no config movement.

    Pinned three ways: the CLI helper returns the SAME object without touching
    the device layer; a snapshot written before the field existed fills the
    default and is a legal resume (no drift); and the resolved value of the
    knob is what `bf16` carries, so the apply path is visible in the file.
    """
    import dormouse.cli as cli

    run = a_run(dtype="none")
    monkeypatch.setattr(cli, "probe_device", lambda *a, **k: pytest.fail("probed"))
    assert cli._apply_dtype(run, torch.device("cuda"), log=pytest.fail) is run

    old = run.snapshot_toml().replace('dtype = "none"\n', "")
    assert 'dtype = "none"' not in old, "the fixture must model a pre-field snapshot"
    stored = RunConfig.from_snapshot(old)
    assert stored.model.dtype == "none"
    assert run.diff_keys(stored) == [], "a pre-field snapshot must resume clean"

    applied = apply_policy(a_run(dtype="bf16"), [resolve_policy(caps(120), "bf16")])
    assert 'dtype = "bf16"' in applied.snapshot_toml()


def test_the_trainer_does_not_touch_the_device_layer(monkeypatch):
    """The layer is a CLI-entry seam: a `train()` call must not probe anything."""
    from conftest import tiny_run

    import dormouse.train.device as device_mod

    def boom(*a, **k):
        raise AssertionError("train() touched the device layer")

    monkeypatch.setattr(device_mod, "probe_device", boom)
    monkeypatch.setattr(device_mod, "resolve_policy", boom)
    import tempfile
    from pathlib import Path

    from dormouse.model import DormouseModel
    from dormouse.train import train

    run = tiny_run(Path(tempfile.mkdtemp()), steps=1, log_every=1)
    assert run.model.dtype == "none"
    report = train(DormouseModel(run.model), run, log=lambda *_: None)
    assert report["steps"] == 1


# --- the real probe ---------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device to probe")
def test_real_probe_this_box():
    """One real probe: metadata only (no allocation, no kernel, no timing).

    Asserts only device-agnostic facts and PRINTS the reading, so the number a
    reader of the test log sees is this box's, not a claim about every box.
    """
    c = probe_device(0)
    print("\n" + c.line())
    assert c.is_cuda and c.name and c.sm >= 50
    assert c.total_mem > 0
    p = resolve_policy(c, "auto")
    assert p.compute_dtype in COMPUTE_DTYPES
    expected = "bf16" if c.bf16 else "fp16" if c.fp16 else "fp32"
    assert p.compute_dtype == expected
    if c.free_mem >= 0:
        assert c.free_mem <= c.total_mem
        assert p.memory_budget == c.free_mem - DEFAULT_RESERVE
    else:
        assert p.memory_budget is None and "unreadable" in p.reason
    print(p.line())
    assert "pcie=" in p.line()


def test_pcie_query_failure_is_unknown(monkeypatch):
    """A container without nvidia-smi gets `None`, never a fabricated number."""
    import subprocess

    import dormouse.train.device as device_mod

    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", boom)
    assert device_mod._pcie(0) == (None, None)
    c = DeviceCaps(0, "x", 12, 0, 1 << 30, 1 << 30)
    assert "pcie=unknown" in c.line()
