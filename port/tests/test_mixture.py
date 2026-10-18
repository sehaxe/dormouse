"""The staged mixture: determinism, proportions, the stage switch, the epoch cap.

Every gate here is CPU and tiny. The three claims that carry the module are
(1) the byte sequence is a pure function of `(seed, step)`, (2) the stage switch
lands on the documented fraction and (3) a source over the repetition cap stops
the run naming its escape — all three are semantics, none needs a GPU.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from dormouse.instrument import LoudError
from dormouse.train.data import ByteStream
from dormouse.train.mixture import (
    DEFAULT_MAX_EPOCHS,
    MixPlan,
    MixStage,
    MixtureStream,
    check_epoch_cap,
    dumps_mix_plan,
    load_mix_plan,
    loads_mix_plan,
    open_mixture,
    per_source_bytes,
    replay_bytes,
    source_for_step,
    stage_bounds,
    stage_index,
)

BATCH, SEQ = 2, 8
WINDOW = BATCH * (SEQ + 1)

#: The research doc's §6.1 decay stage — the shape the plan is argued from.
DECAY = {"web": 0.62, "code": 0.20, "math": 0.13, "ru": 0.05}


def write_bin(path: Path, byte: int, n: int = 4096) -> Path:
    """A source whose every byte identifies it — `b'A' * n`, `b'B' * n`, …"""
    path.write_bytes(bytes([byte]) * n)
    return path


def sources(
    tmp_path: Path, names: tuple[str, ...] = ("web", "code", "math", "ru"), n: int = 4096
) -> dict[str, Path]:
    """One file per name; the nth name's bytes are `ord('A') + n`, so a window says who it is."""
    return {
        name: write_bin(tmp_path / f"{name}.bin", ord("A") + i, n) for i, name in enumerate(names)
    }


def plan_of(stages: list[tuple[float, dict[str, float]]], srcs: dict[str, Path], **kw) -> MixPlan:
    return MixPlan(tuple(MixStage(f, w) for f, w in stages), srcs, **kw)


def read(stream, n: int) -> bytes:
    """The raw bytes of `n` windows, so a source is identifiable from the output."""
    out = bytearray()
    for _ in range(n):
        x, _y = stream.next_batch()
        out += bytes(x.reshape(-1).tolist())
    return bytes(out)


def window_byte(stream) -> int:
    """The identifying byte of the next window — asserting it comes from ONE source."""
    x, _y = stream.next_batch()
    flat = x.reshape(-1)
    assert int(flat.min()) == int(flat.max()), "a window must come from ONE source"
    return int(flat[0])


# --- the interface the trainer swaps in --------------------------------------


def test_next_batch_mirrors_byte_stream(tmp_path):
    """A one-source mixture IS the plain stream: same shape, same shift, same bytes."""
    src = sources(tmp_path, ("web",))
    plain = ByteStream(src["web"], BATCH, SEQ)
    mixed = MixtureStream(plan_of([(1.0, {"web": 1.0})], src), BATCH, SEQ, steps=10)
    for _ in range(4):
        px, py = plain.next_batch()
        mx, my = mixed.next_batch()
        assert (px == mx).all() and (py == my).all()
        assert mx.shape == (BATCH, SEQ) == my.shape
        assert (my[:, :-1] == mx[:, 1:]).all()  # next-byte LM


def test_same_seed_is_the_same_byte_sequence(tmp_path):
    src = sources(tmp_path)
    plan = plan_of([(1.0, DECAY)], src)
    one = MixtureStream(plan, BATCH, SEQ, steps=200, seed=7)
    two = MixtureStream(plan, BATCH, SEQ, steps=200, seed=7)
    assert read(one, 200) == read(two, 200)


def test_the_seed_decides_the_mixture(tmp_path):
    """`seed` is not decoration: two seeds interleave the sources differently."""
    src = sources(tmp_path, ("web", "code"))
    plan = plan_of([(1.0, {"web": 0.5, "code": 0.5})], src)
    assert read(MixtureStream(plan, BATCH, SEQ, steps=64, seed=1), 64) != read(
        MixtureStream(plan, BATCH, SEQ, steps=64, seed=2), 64
    )


def test_rewind_returns_exactly_the_same_bytes(tmp_path):
    src = sources(tmp_path)
    stream = MixtureStream(plan_of([(1.0, DECAY)], src), BATCH, SEQ, steps=50)
    first = read(stream, 10)
    assert stream.step == 10
    stream.rewind()
    assert stream.step == 0
    assert read(stream, 10) == first


def test_skip_bytes_lands_where_an_unbroken_run_would_be(tmp_path):
    """The resume property: a fast-forward must not change the bytes that follow."""
    src = sources(tmp_path)
    plan = plan_of([(1.0, DECAY)], src)
    whole = MixtureStream(plan, BATCH, SEQ, steps=40)
    read(whole, 10)
    tail = read(whole, 30)
    resumed = MixtureStream(plan, BATCH, SEQ, steps=40)
    resumed.skip_bytes(10 * WINDOW)
    assert resumed.step == 10
    assert read(resumed, 30) == tail


def test_replay_bytes_matches_what_the_stream_actually_reads(tmp_path):
    """The schedule function and the stream agree — the resume math rests on it."""
    names = ("web", "code", "math", "ru")
    src = sources(tmp_path, names)
    by_byte = {chr(ord("A") + i): name for i, name in enumerate(names)}
    plan = plan_of([(1.0, DECAY)], src)
    stream = MixtureStream(plan, BATCH, SEQ, steps=40)
    seen = dict.fromkeys(names, 0)
    for _ in range(12):
        seen[by_byte[chr(window_byte(stream))]] += WINDOW
    assert replay_bytes(plan, steps=40, upto=12, batch=BATCH, seq_len=SEQ) == seen
    assert sum(seen.values()) == 12 * WINDOW


def test_skip_bytes_refuses_a_partial_window(tmp_path):
    src = sources(tmp_path)
    stream = MixtureStream(plan_of([(1.0, DECAY)], src), BATCH, SEQ, steps=40)
    with pytest.raises(LoudError, match="whole number"):
        stream.skip_bytes(7)


# --- the schedule ------------------------------------------------------------


def test_stage_bounds_are_the_hand_computed_fractions(tmp_path):
    src = sources(tmp_path, ("web", "code", "math"))
    plan = plan_of([(0.70, {"web": 1.0}), (0.90, {"code": 1.0}), (1.0, {"math": 1.0})], src)
    assert stage_bounds(plan, 1000) == (0, 700, 900, 1000)
    # The IEEE trap: 0.7 * 10000 is 6999.999999999999 — round, do not truncate.
    assert stage_bounds(plan, 10000) == (0, 7000, 9000, 10000)


def test_the_stage_switches_at_the_exact_fraction(tmp_path):
    """Each stage weights ONE source, so the window's own byte names its stage."""
    src = sources(tmp_path, ("web", "code", "math"))
    plan = plan_of([(0.70, {"web": 1.0}), (0.90, {"code": 1.0}), (1.0, {"math": 1.0})], src)
    for step, want, byte in (
        (0, 0, b"A"),
        (699, 0, b"A"),
        (700, 1, b"B"),
        (899, 1, b"B"),
        (900, 2, b"C"),
        (999, 2, b"C"),
    ):
        assert stage_index(plan, step, 1000) == want, f"step {step} is in the wrong stage"
        # A fresh stream fast-forwarded to the step: the byte names the stage, and
        # `skip_bytes` has to land on the same stage the step index says.
        stream = MixtureStream(plan, BATCH, SEQ, steps=1000)
        stream.skip_bytes(step * WINDOW)
        assert window_byte(stream) == byte[0], f"step {step} read the wrong source"
    assert stage_index(plan, 1000, 1000) == 2  # past the budget the decay stage continues


def test_weights_hold_over_10k_windows(tmp_path):
    """The realized proportions over 10k steps are inside 3σ of the plan's weights.

    The draw is i.i.d. per step, so the bound is binomial: `σ = sqrt(n·p·(1-p))`
    windows, i.e. a RELATIVE spread of `sqrt(q/(n·p))` — 1.5% for the 62% source
    at n=10k but 4.4% for the 5% one. A flat ±2% is not a bound i.i.d. sampling
    can honour for the small sources; 3σ is the honest one.
    """
    src = sources(tmp_path)
    plan = plan_of([(1.0, DECAY)], src)
    n = 10_000
    counts = dict.fromkeys(DECAY, 0)
    for step in range(n):
        counts[source_for_step(plan, step, n)] += 1
    assert sum(counts.values()) == n
    for name, p in DECAY.items():
        sigma = math.sqrt(n * p * (1 - p))
        assert abs(counts[name] - n * p) <= 3 * sigma, f"{name}: {counts[name]} vs {n * p}"


# --- the epoch cap -----------------------------------------------------------


def test_per_source_bytes_and_epochs_are_hand_computable(tmp_path):
    """100 steps × 10 B = 1000 B: 900 B to web, 100 B to code — 0.01 and 0.1 epochs."""
    src = sources(tmp_path, ("web", "code"))
    write_bin(src["web"], ord("A"), 90_000)
    write_bin(src["code"], ord("B"), 1_000)
    plan = plan_of([(1.0, {"web": 0.9, "code": 0.1})], src)
    assert per_source_bytes(plan, steps=100, batch=1, seq_len=9) == {"web": 900, "code": 100}
    assert check_epoch_cap(plan, steps=100, batch=1, seq_len=9) == {"web": 0.01, "code": 0.1}


def test_an_epoch_over_the_cap_is_loud_and_names_the_escape(tmp_path):
    """10k steps × 10 B: the 10% code slice (1 kB) is read 10× — over the 4× cap."""
    src = sources(tmp_path, ("web", "code"))
    write_bin(src["web"], ord("A"), 90_000)
    write_bin(src["code"], ord("B"), 1_000)
    plan = plan_of([(1.0, {"web": 0.9, "code": 0.1})], src)
    assert per_source_bytes(plan, steps=10_000, batch=1, seq_len=9)["code"] == 10_000
    with pytest.raises(LoudError) as e:
        check_epoch_cap(plan, steps=10_000, batch=1, seq_len=9)
    msg = str(e.value)
    assert "'code' 10.00x" in msg and "max_epochs=4" in msg
    assert "--set max_epochs=N (0 = off)" in msg
    assert "'web'" not in msg  # web is under the cap and must not be named
    # The escapes work: a bigger cap, or 0 = off.
    assert check_epoch_cap(plan, steps=10_000, batch=1, seq_len=9, max_epochs=10)["code"] == 10.0
    assert check_epoch_cap(plan, steps=10_000, batch=1, seq_len=9, max_epochs=0)["code"] == 10.0


def test_the_cap_defaults_to_the_4x_rule():
    """No `max_epochs` in the file -> the research's 4× rule (2305.16264)."""
    assert DEFAULT_MAX_EPOCHS == 4.0
    plan = loads_mix_plan(_plan_toml(stages=[(1.0, "a = 1.0")], sources_table='a = "a.bin"'))
    assert plan.max_epochs == 4.0


def test_the_cap_is_a_plan_key_and_an_override_beats_it(tmp_path):
    src = sources(tmp_path, ("web", "code"))
    write_bin(src["web"], ord("A"), 90_000)
    write_bin(src["code"], ord("B"), 1_000)
    plan = plan_of([(1.0, {"web": 0.9, "code": 0.1})], src, max_epochs=20.0)
    assert check_epoch_cap(plan, steps=10_000, batch=1, seq_len=9)["code"] == 10.0
    with pytest.raises(LoudError, match="max_epochs=1"):
        check_epoch_cap(plan, steps=10_000, batch=1, seq_len=9, max_epochs=1)


def test_open_mixture_runs_the_guard_before_reading(tmp_path):
    src = sources(tmp_path, ("web", "code"))
    write_bin(src["web"], ord("A"), 90_000)
    write_bin(src["code"], ord("B"), 1_000)
    path = tmp_path / "mix_plan.toml"
    path.write_text(_plan_toml(stages=[(1.0, "web = 0.9\ncode = 0.1")], sources_table=_srcs(src)))
    with pytest.raises(LoudError, match="epoch cap"):
        open_mixture(path, steps=10_000, batch=1, seq_len=9)
    stream = open_mixture(path, steps=10, batch=1, seq_len=9, max_epochs=0)
    assert len(read(stream, 3)) == 3 * 9  # a clean plan builds and reads its windows


def test_a_missing_source_file_is_loud(tmp_path):
    plan = plan_of([(1.0, {"web": 1.0})], {"web": tmp_path / "gone.bin"})
    with pytest.raises(LoudError, match="'web' .* does not exist"):
        check_epoch_cap(plan, steps=10, batch=1, seq_len=9)
    with pytest.raises(LoudError, match="does not exist"):
        MixtureStream(plan, BATCH, SEQ, steps=10)


def test_a_missing_plan_file_is_loud(tmp_path):
    with pytest.raises(LoudError, match="does not exist"):
        load_mix_plan(tmp_path / "nope.toml")


# --- the plan file -----------------------------------------------------------


def test_the_documented_plan_parses_and_round_trips(tmp_path):
    """The module docstring's example, plus `loads(dumps(p)) == p`."""
    src = sources(tmp_path)
    text = _plan_toml(
        stages=[
            (0.70, "web = 0.90\ncode = 0.10"),
            (0.90, "web = 0.80\ncode = 0.15\nmath = 0.05"),
            (1.0, "web = 0.62\ncode = 0.20\nmath = 0.13\nru = 0.05"),
        ],
        sources_table=_srcs(src),
        extra="max_epochs = 6.0\nseed = 3\n",
    )
    plan = loads_mix_plan(text)
    assert [s.until_frac for s in plan.stages] == [0.70, 0.90, 1.0]
    assert plan.stages[2].weights == DECAY
    assert (plan.max_epochs, plan.seed) == (6.0, 3)
    assert plan.names == ("code", "math", "ru", "web")
    assert plan.weights_at(0, 1000) == {"web": 0.90, "code": 0.10}
    assert plan.weights_at(700, 1000) == {"web": 0.80, "code": 0.15, "math": 0.05}
    assert loads_mix_plan(dumps_mix_plan(plan)) == plan


def test_relative_source_paths_resolve_beside_the_plan(tmp_path):
    src = sources(tmp_path, ("web",))
    (tmp_path / "mix_plan.toml").write_text(
        _plan_toml(stages=[(1.0, "web = 1.0")], sources_table='web = "web.bin"')
    )
    assert load_mix_plan(tmp_path / "mix_plan.toml").sources["web"] == src["web"]


def test_an_unknown_plan_key_is_loud():
    text = _plan_toml(stages=[(1.0, "a = 1.0")], sources_table='a = "a.bin"', extra="weights = 1\n")
    with pytest.raises(LoudError, match="unknown key"):
        loads_mix_plan(text)


def test_an_unknown_source_in_weights_is_loud():
    """A weight for a source the plan never names is a plan that cannot run."""
    text = _plan_toml(stages=[(1.0, "a = 0.5\ncode = 0.5")], sources_table='a = "a.bin"')
    with pytest.raises(LoudError, match="weights source 'code'"):
        loads_mix_plan(text)


def test_weights_that_are_not_a_probability_vector_are_loud():
    both = 'a = "a.bin"\nb = "b.bin"'
    for weights in ("a = 0.5\nb = 0.4", "a = 1.5"):
        with pytest.raises(LoudError, match="sum to"):
            loads_mix_plan(_plan_toml(stages=[(1.0, weights)], sources_table=both))
    with pytest.raises(LoudError, match="negative weight"):
        loads_mix_plan(_plan_toml(stages=[(1.0, "a = -0.5\nb = 1.5")], sources_table=both))


def test_a_stage_that_cannot_start_is_loud():
    text = _plan_toml(
        stages=[(0.90, "a = 1.0"), (0.70, "b = 1.0")], sources_table='a = "a.bin"\nb = "b.bin"'
    )
    with pytest.raises(LoudError, match="does not increase"):
        loads_mix_plan(text)


def test_a_tail_stage_that_misses_1_is_loud():
    with pytest.raises(LoudError, match="not 1.0"):
        loads_mix_plan(_plan_toml(stages=[(0.90, "a = 1.0")], sources_table='a = "a.bin"'))


def test_a_source_no_stage_weights_is_loud():
    text = _plan_toml(stages=[(1.0, "a = 1.0")], sources_table='a = "a.bin"\nb = "b.bin"')
    with pytest.raises(LoudError, match="never be read"):
        loads_mix_plan(text)


def test_a_plan_with_no_stage_is_loud():
    with pytest.raises(LoudError, match="no \\[\\[stage\\]\\]"):
        loads_mix_plan(_plan_toml(stages=[], sources_table='a = "a.bin"'))


def test_a_non_numeric_value_is_loud_not_a_ValueError():
    """A config boundary refuses with cause and escape, not with `float('off')`."""
    with pytest.raises(LoudError, match="max_epochs is not a number"):
        loads_mix_plan(
            _plan_toml(
                stages=[(1.0, "a = 1.0")], sources_table='a = "a.bin"', extra='max_epochs = "off"'
            )
        )
    with pytest.raises(LoudError, match="until_frac is not a number"):
        loads_mix_plan(
            '[[stage]]\nuntil_frac = "half"\n[stage.weights]\na = 1.0\n\n[sources]\na = "a.bin"\n'
        )
    with pytest.raises(LoudError, match="weight 'a' is not a number"):
        loads_mix_plan(_plan_toml(stages=[(1.0, 'a = "x"')], sources_table='a = "a.bin"'))


# --- helpers -----------------------------------------------------------------


def _plan_toml(*, stages: list[tuple[float, str]], sources_table: str, extra: str = "") -> str:
    out = [extra] if extra else []
    for frac, weights in stages:
        out += ["[[stage]]", f"until_frac = {frac}", "[stage.weights]", weights]
    return "\n".join(out) + f"\n\n[sources]\n{sources_table}\n"


def _srcs(src: dict[str, Path]) -> str:
    return "\n".join(f'{name} = "{path}"' for name, path in sorted(src.items()))
