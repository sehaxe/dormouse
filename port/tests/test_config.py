"""Config seam: strict keys, loud refusals, and the ported presets."""

from __future__ import annotations

import pytest

from dormouse.config import (
    DEAD_SNAPSHOT_KEYS,
    PRESET_DIR,
    PROGRESS_KEYS,
    ModelConfig,
    RunConfig,
    TrainConfig,
    load_preset,
    load_run,
    parse_preset,
)
from dormouse.instrument import LoudError


def test_presets_load_and_validate():
    small = load_preset("small")
    assert (small.d_model, small.n_heads, small.head_dim) == (768, 12, 64)
    assert small.n_experts == 3 and small.moe_topk == 0
    assert small.engram_rows == 25000 and small.engram_orders == [2, 3, 4]
    b1 = load_preset("b1")
    assert (b1.d_model, b1.n_heads, b1.head_dim, b1.d_ffn) == (2048, 32, 64, 5461)
    # max_iter 4 = the fin-metric verdict (train deeper than you infer).
    assert b1.n_experts == 1024 and b1.moe_topk == 2 and b1.max_iter == 4
    assert b1.engram_rows == 4_000_000 and b1.shared_rank == 192


def test_unknown_key_refuses_and_names_the_escape():
    """(a) An unknown key is a LOUD error naming the key and the escape."""
    with pytest.raises(LoudError) as e:
        parse_preset("d_model = 64\nn_heads = 4\nhead_dim = 16\nnot_a_key = 3\n")
    msg = str(e.value)
    assert "not_a_key" in msg
    assert "remove the key" in msg and "port the mechanism" in msg
    # A typo in a real key is the same class, and it must not silently default.
    with pytest.raises(LoudError):
        parse_preset("d_models = 64\n")
    # A deleted stub field is refused the same way — the pin that `use_gr`,
    # `use_mhc`, `use_mor`, `use_situ` and `stress` are GONE and not merely
    # unwired (a re-added field would accept the key again). `jepa_weight` left
    # this list on 2026-10-21: it is a real field again (`model/jepa.py`).
    with pytest.raises(LoudError) as e:
        parse_preset("d_model = 64\nn_heads = 4\nhead_dim = 16\nuse_gr = true\n")
    assert "use_gr" in str(e.value)


def _toml_literal(value):
    """A TOML scalar for a recorded default, so the fixture cannot drift."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return repr(value)


def test_dead_snapshot_keys_load_counted_and_do_not_reach_the_config(capsys):
    """(a)+(c) A snapshot written before the removals still loads, COUNTED.

    Every `checkpoints/*.config.toml` on disk predates the removals, so refusing
    these keys would make every existing checkpoint unreadable (resume, eval,
    decode, serve, `--init-from`). The compat list drops each one WITH ITS
    RECORDED DEFAULT, with a printed line naming the key and its commit — COUNTED,
    not silent — and the resolved config must NOT carry it: it is gone, not
    defaulted back into the schema. Pinned per key, not as a count: the count was
    the last thing to rot (it was a global 6 while the message is per table).
    """
    # The list is FLAT (one name, one entry) and the drop is section-agnostic,
    # so the model table can carry every key at its recorded default; the train
    # table carries one, which is the per-section COUNTED line.
    text = (
        "[model]\n"
        "d_model = 64\nn_heads = 4\nhead_dim = 16\n"
        + "".join(f"{k} = {_toml_literal(v)}\n" for k, (_commit, v) in DEAD_SNAPSHOT_KEYS.items())
        + "[train]\nsteps = 3\nstress = false\n"
    )
    back = RunConfig.from_snapshot(text)
    assert back.model.d_model == 64 and back.train.steps == 3
    for key in DEAD_SNAPSHOT_KEYS:
        assert not hasattr(back.model, key)
        assert not hasattr(back.train, key)
    err = capsys.readouterr().err
    # One COUNTED line per table, each naming its keys and their commits; the
    # model table carried every key, the train table carried `stress`.
    assert f"dropped {len(DEAD_SNAPSHOT_KEYS)} dead snapshot key(s)" in err
    assert "dropped 1 dead snapshot key(s)" in err
    for key, (commit, _default) in DEAD_SNAPSHOT_KEYS.items():
        assert key in err and commit in err


def test_the_dead_list_is_exactly_what_the_committed_snapshots_carry():
    """The list is enumerated from the snapshots, not guessed.

    A snapshot that carries an unknown key NOT on the list is still refused, so
    the list must cover every key the on-disk snapshots actually hold — one
    missing key leaves its run unreadable (that is how `aux_fb_weight` was
    found: a list without it still refused every snapshot that carried it).
    """
    import dataclasses as _dc
    import glob
    import tomllib as _tomllib

    known = {
        "model": {f.name for f in _dc.fields(ModelConfig)},
        "train": {f.name for f in _dc.fields(TrainConfig)},
    }
    snaps = sorted(
        glob.glob(str(PRESET_DIR.parent / "checkpoints/**/*.config.toml"), recursive=True)
    )
    assert snaps, "no committed snapshots found"
    for path in snaps:
        raw = _tomllib.loads(open(path).read())
        for section in ("model", "train"):
            for key in raw.get(section, {}):
                assert key in known[section] or key in DEAD_SNAPSHOT_KEYS, (
                    f"{path}: {section}.{key} is unknown and not on DEAD_SNAPSHOT_KEYS"
                )


def test_every_snapshot_on_disk_loads():
    """The compat list exists for exactly this: every snapshot resolves.

    The enumeration above proves every KEY is either a field or on the list; this
    proves the COMPAT PATH accepts each one — `from_snapshot` is the single seam
    resume, eval, decode, serve and `--init-from` all cross, so one refusal here
    breaks every one of them for that run (`mix1` and `sft12` were both refused
    on `model.dtype` and the six `train.dist_*` before that key existed).
    """
    import glob

    snaps = sorted(
        glob.glob(str(PRESET_DIR.parent / "checkpoints/**/*.config.toml"), recursive=True)
    )
    assert snaps, "no committed snapshots found"
    for path in snaps:
        back = RunConfig.from_snapshot(open(path).read())
        assert isinstance(back, RunConfig)


def test_a_live_non_default_value_survives_the_value_check(capsys):
    """`sw1` was trained 300 000 steps at `act_quant = "int4"` — a LIVE field.

    The value check is about DEAD keys only, so a live key is never compared to a
    default and its value survives unchanged. This is the artifact the check
    exists for: if `act_quant` were ever retired, this same snapshot must become
    the refusal below — never a quiet `none`, which is a different forward on the
    same weights.
    """
    sw1 = PRESET_DIR.parent / "checkpoints/sw1/sw1.config.toml"
    assert sw1.exists(), f"{sw1} missing — the snapshot suite needs checkpoints/"
    back = RunConfig.from_snapshot(sw1.read_text())
    assert back.model.act_quant == "int4"
    assert back.train.steps == 300_000
    assert "act_quant" not in capsys.readouterr().err


def test_a_dead_key_off_its_default_refuses_loudly():
    """The value check: a dead key is dropped only AT its recorded default.

    `use_gr = true` means the checkpoint ran a mechanism this code does not have.
    Dropping the key would resume a DIFFERENT model on the same weights, which is
    the one thing the resume gate exists to refuse, so the snapshot is refused —
    naming the key, its value, the default, the commit and the escape.
    """
    text = "[model]\nd_model = 64\nn_heads = 4\nhead_dim = 16\nuse_gr = true\n"
    with pytest.raises(LoudError) as e:
        RunConfig.from_snapshot(text)
    msg = str(e.value)
    for piece in ("use_gr", "True", "False", "0e7d25a", "Escape"):
        assert piece in msg
    # `False == 0` in Python, so a bool default and an int snapshot value are NOT
    # the same thing: a `0` where the field was a bool is off the default.
    with pytest.raises(LoudError):
        RunConfig.from_snapshot("[train]\nstress = 0\n")


def test_snapshot_unknown_key_not_on_the_dead_list_still_refuses():
    """(b) The compat list is DATED, not a relaxation of the strict schema."""
    text = "[model]\nd_model = 64\nn_heads = 4\nhead_dim = 16\nnot_a_key = 3\n"
    with pytest.raises(LoudError) as e:
        RunConfig.from_snapshot(text)
    msg = str(e.value)
    assert "not_a_key" in msg and "remove the key" in msg


def _byteflow_preset(**over) -> str:
    """A minimal valid byteflow model table, with named overrides."""
    keys: dict[str, object] = {
        "d_model": 64,
        "n_heads": 4,
        "head_dim": 16,
        "use_byteflow": "true",
        "byteflow_k_tokens": 8,
        "bf16": "false",
        "use_engram": "false",
    }
    keys.update(over)
    return "\n".join(f"{k} = {v}" for k, v in keys.items()) + "\n"


def test_use_byteflow_is_a_real_field_with_the_crates_defaults():
    """The flag loads and refuses nothing; its numbers are `ByteFlowConfig`'s own.

    (`vendor/dormouse-fused/crates/burn-byteflow/src/net.rs:290-310` — copied,
    not guessed. The presets all carry `use_byteflow = false`, so a preset
    cannot change nets by omission.)
    """
    cfg = parse_preset(_byteflow_preset())
    assert cfg.use_byteflow is True
    defaults = ModelConfig()
    assert defaults.use_byteflow is False
    assert (
        defaults.byteflow_d_local,
        defaults.byteflow_d_global,
        defaults.byteflow_k_tokens,
        defaults.byteflow_e_layers,
        defaults.byteflow_g_layers,
        defaults.byteflow_heads_local,
        defaults.byteflow_heads_global,
        defaults.byteflow_w_local,
        defaults.byteflow_d_ff_local,
        defaults.byteflow_d_ff_global,
        defaults.byteflow_bins,
        defaults.byteflow_eps2,
        defaults.byteflow_logdet,
        defaults.byteflow_max_bytes,
    ) == (512, 1536, 3200, 6, 20, 8, 12, 512, 1408, 4096, 16, 0.5, False, 8192)
    # The shipped presets keep their meaning: not one of them turns it on.
    assert load_preset("small").use_byteflow is False
    assert load_preset("b1").use_byteflow is False


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"max_seq_len": 4096, "byteflow_max_bytes": 512}, "byteflow_max_bytes"),
        ({"byteflow_k_tokens": 0}, "must be >= 1"),
        ({"byteflow_k_tokens": 1024}, "byteflow_k_tokens"),
        ({"byteflow_eps2": 0.0}, "byteflow_eps2"),
        ({"byteflow_bins": 0}, "byteflow_bins"),
        ({"byteflow_d_local": 17}, "does not divide"),
        ({"byteflow_d_global": 17}, "does not divide"),
        ({"bf16": "true", "byteflow_logdet": "true"}, "bf16 \\+ byteflow_logdet"),
        ({"use_engram": "true"}, "use_engram"),
    ],
)
def test_byteflow_coherence_is_loud_at_load(over, match):
    """(d) The byteflow arm's own fields are checked where a preset can be wrong."""
    with pytest.raises(LoudError, match=match):
        parse_preset(_byteflow_preset(**over))


def test_byteflow_runs_in_bf16_and_keeps_the_exact_chunker_fp32():
    """The dtype is a COMPUTE knob on the net (`byteflow.py`'s precision rule).

    `bf16` used to be refused here; the net now runs bf16 matmuls with fp32
    masters, so only the Cholesky pair is left refused. The preset path proves
    the loaded MODEL carries the flag, not just `ModelConfig`'s default.
    """
    assert parse_preset(_byteflow_preset(bf16="true")).bf16 is True
    logdet = parse_preset(_byteflow_preset(bf16="false", byteflow_logdet="true"))
    assert logdet.byteflow_logdet is True


def test_a_seq_len_below_the_patch_count_refuses(tmp_path):
    """K boundaries live inside the T-byte window: K > T has nothing to select."""
    path = tmp_path / "bf.toml"
    path.write_text(_byteflow_preset(byteflow_k_tokens=32))
    with pytest.raises(LoudError, match="seq_len"):
        load_run(str(path), train_overrides={"seq_len": 16})


@pytest.mark.parametrize("rung", ["fp16", "fp4", "lbit", "int4"])
def test_quant_ladder_rungs_refuse_at_load(rung):
    """(e) The training-time rungs refuse per rung, by name, at LOAD.

    fp8/nvfp4/ternary graduated to STORAGE rungs (2026-10-15, then the
    ternary rung 2026-10-16, `tests/test_ladder_storage.py`): they pack the
    factor bank and the checkpoint, fp32 masters stay the optimizer's object.
    What still refuses is the TRAINING-time set — LittleBit's gates G-A..G-D
    must close first.
    """
    with pytest.raises(LoudError) as e:
        parse_preset(f'd_model = 64\nn_heads = 4\nhead_dim = 16\nquant = "{rung}"\n')
    msg = str(e.value)
    assert rung in msg and "implemented rungs are the STORE levels" in msg
    # And the shipped presets use only the ported rung.
    assert ModelConfig().quant == "none"


def test_train_key_refusals():
    """The train half has the same two rules."""
    # `engram_ram` is a REAL field since the host path landed; the arms-off test
    # (`test_model`) uses `use_engram` for that reason. An unknown key still
    # refuses, naming the key and the escape.
    assert load_run("small", train_overrides={"engram_ram": True}).train.engram_ram is True
    with pytest.raises(LoudError) as e:
        load_run("small", train_overrides={"nope": 1})
    assert "nope" in str(e.value) and "remove the" in str(e.value)
    with pytest.raises(LoudError) as e:
        load_run("small", train_overrides={"opt": "adan"})
    assert "adan" in str(e.value)


def test_missing_key_takes_the_class_default():
    cfg = parse_preset("d_model = 128\nn_heads = 2\nhead_dim = 64\n")
    assert cfg.d_ffn == 2048  # the schema default
    assert cfg.engram_lam_max == 0.5
    assert cfg.bf16 is True


def test_validation_is_loud_not_clamping():
    with pytest.raises(LoudError, match="tile the residual width"):
        parse_preset("d_model = 100\nn_heads = 3\nhead_dim = 64\n")
    with pytest.raises(LoudError, match="refused, not clamped"):
        parse_preset("d_model = 64\nn_heads = 4\nhead_dim = 16\nn_experts = 2\nmoe_topk = 3\n")
    with pytest.raises(LoudError, match="outside"):
        parse_preset("d_model = 64\nn_heads = 4\nhead_dim = 16\nengram_lam_max = 1.5\n")


def test_seq_len_cannot_exceed_max_seq_len():
    with pytest.raises(LoudError, match="max_seq_len"):
        load_run("small", train_overrides={"seq_len": 4096})


def test_snapshot_round_trip_and_progress_exemption():
    run = load_run("small", train_overrides={"steps": 5, "ckpt_name": "x"})
    back = RunConfig.from_snapshot(run.snapshot_toml())
    assert back.diff_keys(run) == []
    # Progress keys are exempt: extending a run is a legal resume.
    from dataclasses import replace

    longer = replace(run, train=replace(run.train, steps=999, log_every=1, eval=True))
    assert longer.diff_keys(run) == []
    assert set(PROGRESS_KEYS) <= set(TrainConfig().__dataclass_fields__)
    # ... and so is the telemetry sink: it decides WHERE the measurement goes,
    # never what the model computes (audit MINOR 23, 2026-10-17).
    assert "json_log" in PROGRESS_KEYS
    moved = replace(run, train=replace(run.train, json_log="/tmp/other.jsonl"))
    assert moved.diff_keys(run) == []
    # Anything else is not.
    hotter = replace(run, train=replace(run.train, lr=run.train.lr * 2))
    assert hotter.diff_keys(run) == ["train.lr"]


def test_dspark_guards_run_outside_byteflow():
    """Audit MINOR 12: the dspark coherence checks lived in `_validate_byteflow`,
    so the default path accepted a negative γ and a zero k/stride."""
    from dormouse.config import validate

    validate(ModelConfig())  # the legal default still passes
    for bad in ({"dspark_weight": -1.0}, {"dspark_k": 0}, {"dspark_stride": 0}):
        with pytest.raises(LoudError, match="dspark"):
            validate(ModelConfig(**bad))
    # The byteflow-only interaction is still refused where it belongs.
    with pytest.raises(LoudError, match="use_byteflow"):
        validate(
            ModelConfig(
                use_byteflow=True,
                bf16=False,
                use_engram=False,
                byteflow_k_tokens=8,
                dspark_weight=0.1,
            )
        )
