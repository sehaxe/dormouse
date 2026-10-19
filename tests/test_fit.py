"""The `--fit` resolver: which memory mechanisms a card of N GB needs.

The numbers are a MODEL (see `train/fit.py`); these gates pin the DECISIONS —
which set is engaged for a shape and a budget, that a fleet never engages the
host tables, that every override is a real config field, and that the knob off
is a no-op. The estimate's absolute accuracy is the b1 spike's cross-check,
quoted in `docs/architecture/multi-gpu.md`, not asserted here.
"""

from __future__ import annotations

import dataclasses

import pytest
from conftest import TINY

from dormouse.config import ModelConfig, RunConfig, TrainConfig
from dormouse.instrument import LoudError
from dormouse.train.fit import apply_fit, estimate, plan_fit


def a_run(model: ModelConfig | None = None, **train_over) -> RunConfig:
    return RunConfig(
        source="fit-test",
        model=model or ModelConfig(**TINY),
        train=TrainConfig(**train_over),
    )


def test_small_and_medium_fit_a_four_gb_card_with_nothing():
    """The scrap-metal headline: 142M fits 4 GB; the preset does not need help."""
    from dormouse.config import load_run

    for preset in ("small", "medium"):
        plan = plan_fit(load_run(preset), 4)
        assert plan.engaged == (), preset
        assert plan.model_over == {} and plan.train_over == {}
        assert plan.peak_gb < 4 * 0.9, plan.line()


def test_b1_on_16gb_engages_exactly_the_host_table_set():
    """1.07B: the table parameters (402M) are the VRAM the host path buys back."""
    from dormouse.config import load_run

    plan = plan_fit(load_run("b1"), 16)
    assert plan.engaged == ("moe_offload", "engram_ram")
    assert plan.train_over == {"engram_ram": True}
    assert plan.model_over == {"moe_offload": True}
    # ...and the line names every mechanism it engaged.
    line = plan.line()
    for name in plan.engaged:
        assert name.split("=")[0] in line


def test_b1_on_4gb_is_loud_with_the_arithmetic():
    """A 4 GB card cannot hold 1.07B in any configuration this repo owns."""
    from dormouse.config import load_run

    with pytest.raises(LoudError) as e:
        plan_fit(load_run("b1"), 4)
    msg = str(e.value)
    assert "cannot hold b1" in msg and "Escape" in msg and "GiB" in msg
    assert "host offload engine" in msg


def test_a_fleet_never_engages_the_host_tables():
    """Host tables are per-rank state; on a fleet the plan must not pick them."""
    from dormouse.config import load_run

    run = load_run("b1")
    run.train = dataclasses.replace(run.train, dist=True)
    plan = plan_fit(run, 16)
    assert "engram_ram" not in plan.train_over
    assert "engram_ram" not in plan.engaged
    assert plan.peak_gb < 13.5  # still fits, through the other mechanisms
    # When the only mechanism that would fit is unavailable, the refusal says so.
    with pytest.raises(LoudError) as e:
        plan_fit(run, 6)
    assert "host-table" in str(e.value)


def test_a_dense_moe_never_engages_the_router_offload():
    """`moe_offload` needs a routed bank; a dense blend would be COUNTED inert."""
    cfg = ModelConfig(
        **{
            **TINY,
            "moe_topk": 0,
            "d_model": 1024,
            "d_ffn": 2730,
            "n_heads": 64,
            "engram_rows": 65536,
            "engram_dim": 64,
        }
    )
    run = a_run(cfg, batch=8, seq_len=512)
    base = sum(estimate(cfg, batch=8, seq_len=512).values())
    off = sum(estimate(cfg, batch=8, seq_len=512, engram_ram=True).values())
    assert off < base, "the fixture's tables must be big enough to matter"
    # A budget between the two peaks: exactly one mechanism can close the gap.
    plan = plan_fit(run, ((base + off) / 2) / 2**30 / 0.9)
    assert "moe_offload" not in plan.engaged
    assert plan.engaged == ("engram_ram",), plan.line()


def test_apply_fit_records_the_mechanisms_not_the_knob():
    from dormouse.config import load_run

    run = load_run("b1")
    plan = plan_fit(run, 16)
    fitted = apply_fit(run, plan)
    snap = fitted.snapshot_toml()
    assert "engram_ram = true" in snap and "moe_offload = true" in snap
    assert fitted.train.engram_ram and fitted.model.moe_offload
    # Every override the resolver can emit is a real config field (a typo would
    # be an unknown key at load time, or worse, silently ignored).
    keys = set(plan.train_over) | set(plan.model_over)
    fields = {f.name for f in dataclasses.fields(TrainConfig)} | {
        f.name for f in dataclasses.fields(ModelConfig)
    }
    assert keys <= fields
    # The estimate is a breakdown, not a single opaque number.
    assert set(plan.components) == {"weights", "grads", "muon_state", "adamw_state", "activations"}


def test_the_knob_off_is_a_noop_and_never_plans(monkeypatch):
    import dormouse.cli as cli
    import dormouse.train.fit as fit_mod

    run = a_run()

    def boom(*a, **k):
        raise AssertionError("--fit 0 planned something")

    monkeypatch.setattr(fit_mod, "plan_fit", boom)
    assert cli._apply_fit(run, 0.0, log=pytest.fail) is run


def test_the_estimate_components_are_positive_and_shrink_with_the_mechanisms():
    cfg = ModelConfig(**TINY)
    base = estimate(cfg, batch=2, seq_len=64)
    off = estimate(
        cfg,
        batch=2,
        seq_len=64,
        engram_ram=True,
        optim_8bit=True,
        muon_state_dtype="bf16",
        moe_offload=True,
    )
    assert all(v > 0 for v in base.values())
    for key in ("weights", "grads", "muon_state", "adamw_state"):
        assert off[key] <= base[key], key
    # The routed offload only touches the activation term, and only when there
    # IS a routed bank (topk > 0).
    assert off["activations"] < base["activations"] if cfg.moe_topk > 0 else True
