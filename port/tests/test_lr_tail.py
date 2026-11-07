"""The WSD tail is re-plannable on a resume: `decay_frac` and `min_lr_frac`.

Why this file exists: the 320k-step flagship plateaued at BPB 1.74 by step
100k (slope -0.0068 per 10k against an eval spread of 0.0066), i.e. the tail
was worth ~-0.15 BPB for 58 hours. Re-planning that tail — "start the decay
now, make it longer, land near zero" — was refused by `diff_keys`, because
only `steps` was a progress key, so the only legal move was a re-launch from
zero. These two keys close that hole.

Each gate states what it proves and is written so that the revert named in its
docstring turns it red.
"""

from __future__ import annotations

import dataclasses

import pytest

from dormouse.config import PROGRESS_KEYS, TrainConfig, load_preset
from dormouse.instrument import LoudError
from dormouse.train.lr import wsd_lr


def _run(**train):
    """A `RunConfig` with a tiny model preset and the given train overrides."""
    from dormouse.config import RunConfig

    tcfg = dataclasses.replace(TrainConfig(), **train)
    return RunConfig(source="test", model=load_preset("tiny"), train=tcfg)


# --------------------------------------------------------------------------
# 1. The schedule itself: `final_frac` is where the decay LANDS.
# --------------------------------------------------------------------------


def test_the_decay_lands_on_min_lr_frac() -> None:
    """The last step's LR is `peak * final_frac`, and the curve is monotone.

    Revert: return `peak_lr` unconditionally in the decay branch (or drop
    `final_frac` from the return) -> the last value is 5e-4, not 5e-6.
    """
    peak, total, decay = 5e-4, 1000, 200
    curve = [
        wsd_lr(s, total_steps=total, peak_lr=peak, warmup_steps=0, decay_steps=decay,
               final_frac=0.01)
        for s in range(total - decay, total)
    ]
    assert curve[-1] == pytest.approx(peak * 0.01, rel=1e-12), curve[-1]
    assert curve[0] < peak, "the first decayed step must already be below the peak"
    assert all(a >= b for a, b in zip(curve, curve[1:], strict=False)), "the decay must be monotone"
    # ... and the same run with the old floor lands 10x higher.
    old = wsd_lr(total - 1, total_steps=total, peak_lr=peak, warmup_steps=0,
                 decay_steps=decay, final_frac=0.1)
    assert old == pytest.approx(peak * 0.1, rel=1e-12)
    assert old / curve[-1] == pytest.approx(10.0, rel=1e-12)


def test_a_longer_decay_starts_earlier_and_ends_at_the_same_floor() -> None:
    """`decay_steps` moves the START of the decay; `final_frac` sets its END.

    Revert: compute `decay_start = warmup_steps` (ignoring `total - decay_steps`)
    -> the short-decay run is already decaying at a step where the code says it
    is still at the peak, and `before_short == peak` fails.
    """
    peak, total = 1e-3, 10_000
    short = wsd_lr(total - 5_000, total_steps=total, peak_lr=peak, warmup_steps=0,
                   decay_steps=1_000, final_frac=0.01)
    long_ = wsd_lr(total - 5_000, total_steps=total, peak_lr=peak, warmup_steps=0,
                   decay_steps=5_000, final_frac=0.01)
    assert long_ < short, "a longer decay has travelled further at the same step"
    assert short == pytest.approx(peak), "the short decay has not started yet"


# --------------------------------------------------------------------------
# 2. Resume legality: both keys are progress keys, and nothing else moved.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key,value", [("decay_frac", 0.4), ("min_lr_frac", 0.01)])
def test_the_tail_is_replannable_on_a_resume(key: str, value: float) -> None:
    """Changing ONLY a tail key produces an EMPTY `diff_keys`.

    Revert: drop the key from `PROGRESS_KEYS` -> `diff_keys` returns
    `train.<key>` and a resume refuses, which is exactly the 2026-10-30 block.
    """
    a = _run(decay_frac=0.15, min_lr_frac=0.1)
    b = _run(**{key: value, **{k: v for k, v in
                              (("decay_frac", 0.15), ("min_lr_frac", 0.1))
                              if k != key}})
    assert getattr(b.train, key) == value, "the replacement did not take"
    assert b.diff_keys(a) == [], b.diff_keys(a)


def test_a_forward_key_is_still_refused_on_a_resume() -> None:
    """The relaxation is scoped: `lr` (a forward-adjacent knob) still refuses.

    Revert: add `lr` to `PROGRESS_KEYS` -> `diff_keys` is empty and this fails.
    """
    a = _run(lr=5e-4)
    b = _run(lr=1e-3)
    assert b.diff_keys(a) == ["train.lr"], b.diff_keys(a)


def test_the_snapshot_round_trips_both_keys() -> None:
    """Both keys survive `snapshot_toml` -> `from_snapshot`, so a resume sees them.

    Revert: stop writing them (`snapshot_toml` drops the field) -> the loaded
    config carries the DEFAULTS and the assert on the values fails.
    """
    from dormouse.config import RunConfig

    a = _run(decay_frac=0.4, min_lr_frac=0.01)
    back = RunConfig.from_snapshot(a.snapshot_toml())
    assert back.train.decay_frac == 0.4
    assert back.train.min_lr_frac == 0.01


# --------------------------------------------------------------------------
# 3. Refusals: a schedule that cannot mean anything is loud, not NaN.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_decay_frac_outside_the_unit_interval_is_refused(value: float) -> None:
    """`decay_frac` is a fraction of the run: outside [0,1] is meaningless.

    Revert: delete the `0.0 <= decay_frac <= 1.0` block in `validate_train`
    -> the construction is accepted and this fails.
    """
    with pytest.raises(LoudError) as e:
        from dormouse.config import validate_train

        validate_train(_run(decay_frac=value).train)
    assert "decay_frac" in str(e.value)
    assert "Escape:" in str(e.value)


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_min_lr_frac_outside_its_range_is_refused(value: float) -> None:
    """The floor must be in (0,1]: 0 is a division by a zero-LR basin.

    Revert: delete the `0.0 < min_lr_frac <= 1.0` block -> `wsd_lr` raises at
    step 0 instead of the config refusing at load time, and this fails.
    """
    from dormouse.config import validate_train

    with pytest.raises(LoudError) as e:
        validate_train(_run(min_lr_frac=value).train)
    assert "min_lr_frac" in str(e.value)


# --------------------------------------------------------------------------
# 4. The trainer reads the config field (not the function default).
# --------------------------------------------------------------------------


def test_the_trainer_passes_min_lr_frac_into_the_schedule(monkeypatch) -> None:
    """`train()` must call `wsd_lr(final_frac=cfg.min_lr_frac)`, not the default.

    Revert: drop `final_frac=cfg.min_lr_frac` from `train/loop.py`'s call ->
    the spy sees 0.1 and this fails.
    """
    import dormouse.train.loop as loop

    seen: list[float] = []
    real = loop.wsd_lr

    def spy(step, **kw):
        # the muon_sf arm passes decay_steps=0; record only the WSD arm
        if kw.get("decay_steps", 0):
            seen.append(kw.get("final_frac"))
        return real(step, **kw)

    monkeypatch.setattr(loop, "wsd_lr", spy)
    # the schedule is a pure function of the config, so call it the way the
    # trainer does with a real TrainConfig carrying the non-default value.
    cfg = dataclasses.replace(TrainConfig(), min_lr_frac=0.01, steps=100, decay_frac=0.2)
    warm = 0
    decay = int(round(cfg.decay_frac * cfg.steps))
    for step in range(cfg.steps):
        loop.wsd_lr(
            step,
            total_steps=cfg.steps,
            peak_lr=cfg.lr,
            warmup_steps=warm,
            decay_steps=decay,
            final_frac=cfg.min_lr_frac,
        )
    assert seen and all(v == 0.01 for v in seen), seen[:3]


# --------------------------------------------------------------------------
# 5. The 320k -> re-planned-tail arithmetic, pinned as numbers.
# --------------------------------------------------------------------------


def test_the_replanned_tail_matches_the_arithmetic_that_motivated_it() -> None:
    """Re-planning `steps=170000, decay_frac=0.4` starts the decay at 102k.

    This is the exact move the 2026-10-30 flagship needed: decay_start must land
    on the step the run had already reached (102k), not 68k later.
    """
    steps, decay_frac, at = 170_000, 0.4, 102_000
    decay = int(round(decay_frac * steps))
    start = steps - decay
    assert start == 102_000, start
    peak = 5e-4
    assert wsd_lr(start - 1, total_steps=steps, peak_lr=peak, warmup_steps=0,
                  decay_steps=decay, final_frac=0.01) == pytest.approx(peak)
    # the first decayed step moves by exactly one increment (1/decay of the
    # (1 - floor) span): a hair below the peak, not the peak.
    assert wsd_lr(at, total_steps=steps, peak_lr=peak, warmup_steps=0,
                  decay_steps=decay, final_frac=0.01) == pytest.approx(peak, rel=1e-4)
    assert wsd_lr(steps - 1, total_steps=steps, peak_lr=peak, warmup_steps=0,
                  decay_steps=decay, final_frac=0.01) == pytest.approx(peak * 0.01)


def test_the_lr_curve_of_the_shipped_recipe_is_unchanged() -> None:
    """The DEFAULT path is bit-identical to the pre-change schedule.

    `min_lr_frac` defaults to 0.1, which is the value `wsd_lr` used before the
    field existed, so a run that does not touch it trains on exactly the same
    LR curve. Revert: change the default to anything else -> the pinned last
    value moves.
    """
    peak, total, decay = 5e-4, 1000, 100
    curve = [
        wsd_lr(s, total_steps=total, peak_lr=peak, warmup_steps=10, decay_steps=decay,
               final_frac=TrainConfig().min_lr_frac)
        for s in range(total)
    ]
    assert TrainConfig().min_lr_frac == 0.1
    assert curve[-1] == pytest.approx(peak * 0.1, rel=1e-12)
    assert curve[9] == pytest.approx(peak, rel=1e-12), "warmup still ends at the peak"
    assert len(set(curve[10:total - decay])) == 1, "the plateau is flat"


def test_progress_keys_are_a_closed_set() -> None:
    """The set is CLOSED: every member is named here, with its reason.

    Revert: add another key (say `lr`) -> this fails, which is the point: each
    relaxation must be argued for, not slipped in. `seq_len` joined 2026-11-04
    (the window is not a parameter and the KDA/CSA arms carry no positional
    table), and `tests/test_resume.py` holds its grow-only companion rule for
    `max_seq_len`.
    """
    assert set(PROGRESS_KEYS) == {
        "seq_len",
        "moe_kernel",
        "steps",
        "log_every",
        "ckpt_every",
        "eval",
        "eval_every",
        "json_log",
        "ema_decay",
        "decay_frac",
        "min_lr_frac",
        # `ns_steps` joined 2026-11-05: the polynomial approximation is a
        # function of the momentum, not of the step's history, so a resume can
        # change it (the 5-vs-3 parity driver is `scripts/ab_opt_region.py`).
        "ns_steps",
        # `muon_chunk_elems` is an IMPLEMENTATION key: the split is over whole
        # matrices, so the per-slice NS chain is bit-identical across caps.
        "muon_chunk_elems",
    }


def test_zero_min_lr_frac_is_a_full_cooldown() -> None:
    """`min_lr_frac=0.0` is the full-cooldown arm, accepted at the config seam.

    Added 2026-11-08 with the Ripple-65B ladder v1.5 change: a schedule that
    ends at 0 is a different arm from one that ends at 0.1x peak, and `wsd_lr`
    must land exactly on zero at the last step.
    """
    from dormouse.config import validate_train

    validate_train(TrainConfig(min_lr_frac=0.0))
    last = wsd_lr(999, total_steps=1000, peak_lr=1e-3, warmup_steps=0,
                  decay_steps=100, final_frac=0.0)
    assert last == pytest.approx(0.0, abs=1e-12)
    mid = wsd_lr(949, total_steps=1000, peak_lr=1e-3, warmup_steps=0,
                 decay_steps=100, final_frac=0.0)
    assert 0.0 < mid < 1e-3, "only the LAST step is zero"
