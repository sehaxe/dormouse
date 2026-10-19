"""Config seam: strict keys, loud refusals, and the ported presets."""

from __future__ import annotations

from pathlib import Path

import pytest

from dormouse.config import (
    DEAD_SNAPSHOT_KEYS,
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
    # `use_mhc`, `use_mor`, `use_situ`, `jepa_weight` and `stress` are GONE and
    # not merely unwired (a re-added field would accept the key again).
    with pytest.raises(LoudError) as e:
        parse_preset("d_model = 64\nn_heads = 4\nhead_dim = 16\nuse_gr = true\n")
    assert "use_gr" in str(e.value)


def test_dead_snapshot_keys_load_counted_and_do_not_reach_the_config(capsys):
    """(a)+(c) A snapshot written before the deletions still loads, COUNTED.

    Every `checkpoints/*.config.toml` on disk was written before `0e7d25a`
    (and `39d0485`) deleted these fields, so refusing them would make every
    existing checkpoint unreadable (resume, eval, decode, serve, `--init-from`).
    The compat list drops them with a printed line naming each key and its
    commit — COUNTED, not silent — and the resolved config must NOT carry them:
    they are gone, not defaulted back into the schema.
    """
    text = (
        "[model]\n"
        "d_model = 64\nn_heads = 4\nhead_dim = 16\n"
        "use_gr = false\nuse_mhc = false\nuse_mor = false\n"
        "use_situ = false\njepa_weight = 0.0\naux_fb_weight = 0.0\n"
        "[train]\n"
        "steps = 3\nstress = false\n"
    )
    back = RunConfig.from_snapshot(text)
    assert back.model.d_model == 64 and back.train.steps == 3
    for key in DEAD_SNAPSHOT_KEYS:
        assert not hasattr(back.model, key)
        assert not hasattr(back.train, key)
    err = capsys.readouterr().err
    # Two COUNTED lines: the model table carried six of the seven, the train
    # table carried `stress`.
    assert "dropped 6 dead snapshot key(s)" in err
    assert "dropped 1 dead snapshot key(s)" in err
    for key, commit in DEAD_SNAPSHOT_KEYS.items():
        assert key in err and commit in err


def test_the_dead_list_is_exactly_what_the_committed_snapshots_carry():
    """The list is enumerated from the snapshots, not guessed.

    The snapshots are fixtures (tests/fixtures/snapshot_configs/), copied from
    real runs: the run artifacts themselves live under checkpoints/ and are
    never committed, so a test that scanned them could not run in a fresh
    clone. A snapshot that carries an unknown key NOT on the list is still
    refused, so the list has to cover every key the snapshots actually hold —
    a missing one leaves them all unreadable (this is how `aux_fb_weight`, the
    seventh, was found: the six-key list still refused all 24 on it).
    """
    import dataclasses as _dc
    import glob
    import tomllib as _tomllib

    known = {
        "model": {f.name for f in _dc.fields(ModelConfig)},
        "train": {f.name for f in _dc.fields(TrainConfig)},
    }
    snaps = sorted(
        glob.glob(
            str(Path(__file__).parent / "fixtures/snapshot_configs/**/*.config.toml"),
            recursive=True,
        )
    )
    assert snaps, "no fixture snapshots found"
    for path in snaps:
        raw = _tomllib.loads(open(path).read())
        for section in ("model", "train"):
            for key in raw.get(section, {}):
                assert key in known[section] or key in DEAD_SNAPSHOT_KEYS, (
                    f"{path}: {section}.{key} is unknown and not on DEAD_SNAPSHOT_KEYS"
                )


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
        ({"bf16": "true"}, "bf16"),
        ({"use_engram": "true"}, "use_engram"),
    ],
)
def test_byteflow_coherence_is_loud_at_load(over, match):
    """(d) The byteflow arm's own fields are checked where a preset can be wrong."""
    with pytest.raises(LoudError, match=match):
        parse_preset(_byteflow_preset(**over))


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


def test_mm_dtype_and_kda_recompute_are_live_on_sft_and_rl(tmp_path, monkeypatch):
    """M4: the two process-wide setters were called only by `train()`, so an
    SFT/RL run could carry `mm_dtype=fp8`/`kda_disable_recompute=true` in its
    snapshot while the step ran the process defaults (the SILENT class).

    The gate runs each COMMAND through the CLI with a fake loop that records
    the live module state at entry: the setters must have applied before the
    forward is built. On the pre-fix tree both commands read bf16/False.
    """
    from typer.testing import CliRunner

    from dormouse.cli import app
    from dormouse.model import kda, tsct

    preset = tmp_path / "tiny.toml"
    preset.write_text(
        "d_model = 64\n"
        "n_heads = 4\n"
        "head_dim = 16\n"
        "d_ffn = 128\n"
        "max_seq_len = 512\n"
        "max_iter = 1\n"
        "rank = 8\n"
        "shared_rank = 16\n"
        "moe_topk = 1\n"
        "engram_orders = [2, 3]\n"
        "engram_rows = 64\n"
        "engram_dim = 8\n"
        "bf16 = false\n"
    )
    data = tmp_path / "sft.bin"
    data.write_bytes(b"x" * 4096)
    seen: dict[str, object] = {}

    def capture(run):
        seen["config"] = (run.train.mm_dtype, run.train.kda_disable_recompute)
        seen["live"] = (tsct.mm_dtype(), kda.disable_recompute())

    import importlib

    rl_mod = importlib.import_module("dormouse.post.rl_train")
    sft_mod = importlib.import_module("dormouse.train.sft")

    def fake_sft(model, run, **kw):
        capture(run)
        return {"steps": 0, "loss": 0.0, "supervised_bytes": 0, "nan_skipped": 0}

    def fake_rl(run, **kw):
        capture(run)
        return {
            "steps": 0,
            "loss": 0.0,
            "pass_rate": 0.0,
            "bytes_per_solved": None,
            "rl_samples": 0,
            "nan_skipped": 0,
        }

    def fake_train(model, run, **kw):
        capture(run)
        return {"steps": 0, "loss": 0.0}

    monkeypatch.setattr(sft_mod, "sft_train", fake_sft)
    monkeypatch.setattr(rl_mod, "rl_train", fake_rl)
    monkeypatch.setattr(importlib.import_module("dormouse.train.loop"), "train", fake_train)
    from dormouse.cli import _preset_check
    from dormouse.config import load_run

    runner = CliRunner()

    def live():
        return (tsct.mm_dtype(), kda.disable_recompute())

    tsct.set_mm_dtype("bf16")
    kda.set_disable_recompute(False)
    try:
        # M6: `info` prints, and `serve`/`bench` resolve through `_preset_check`
        # against a snapshot they do not train. Neither may inherit the
        # process-wide compute state a training command would apply.
        result = runner.invoke(
            app,
            [
                "info",
                str(preset),
                "--set",
                "mm_dtype=fp8",
                "--set",
                "kda_disable_recompute=true",
            ],
        )
        assert result.exit_code == 0, result.output
        assert live() == ("bf16", False), "info inherited a training global"
        snap = load_run(str(preset))
        for what in ("served", "scored"):
            _preset_check(str(preset), snap, what=what)
            assert live() == ("bf16", False), f"{what} inherited a training global"
        # The default is the non-training one; the flag is the one seam.
        non_training = load_run(
            str(preset), train_overrides={"mm_dtype": "fp8", "kda_disable_recompute": True}
        )
        assert (non_training.train.mm_dtype, non_training.train.kda_disable_recompute) == (
            "fp8",
            True,
        )
        assert live() == ("bf16", False)
        load_run(
            str(preset),
            train_overrides={"mm_dtype": "fp8", "kda_disable_recompute": True},
            apply_compute=True,
        )
        assert live() == ("fp8", True)

        for args in (
            ["train", str(preset), "--data", str(data)],
            ["run", str(preset), "--data", str(data)],
            ["sft", str(preset), "--data", str(data)],
            ["rl", str(preset)],
        ):
            seen.clear()
            result = runner.invoke(
                app,
                [
                    *args,
                    "--set",
                    "mm_dtype=fp8",
                    "--set",
                    "kda_disable_recompute=true",
                    "--device",
                    "cpu",
                ],
            )
            assert result.exit_code == 0, result.output
            assert seen["config"] == ("fp8", True)
            assert seen["live"] == ("fp8", True), f"{args[0]}: the knob stayed inert"
    finally:
        tsct.set_mm_dtype("bf16")
        kda.set_disable_recompute(False)
