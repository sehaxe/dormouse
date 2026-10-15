"""Config seam: strict keys, loud refusals, and the ported presets."""

from __future__ import annotations

import pytest

from dormouse.config import (
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
    assert b1.n_experts == 1024 and b1.moe_topk == 2 and b1.max_iter == 2
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


@pytest.mark.parametrize(
    ("key", "value", "mechanism"),
    [
        ("use_gr", "true", "Gated Residual"),
        ("use_mhc", "true", "Hyper-Connections"),
        ("use_mor", "true", "Mixture-of-Recursions"),
        ("use_situ", "true", "SiTU-GLU"),
        ("jepa_weight", "0.05", "JEPA"),
        ("dspark_weight", "0.1", "DSpark"),
        ("aux_fb_weight", "0.1", "future-byte"),
        ("act_quant", '"int8"', "activation quantization"),
    ],
)
def test_missing_mechanisms_refuse_when_non_default(key, value, mechanism):
    """(a) A key whose mechanism is missing REFUSES when set non-default.

    `use_byteflow` used to be in this list; it left it the day the port landed
    the crate's five stages (`byteflow.py`), and its own coherence rules are
    pinned in `test_byteflow_*` below.
    """
    with pytest.raises(LoudError) as e:
        parse_preset(f"d_model = 64\nn_heads = 4\nhead_dim = 16\n{key} = {value}\n")
    msg = str(e.value)
    assert key in msg and mechanism in msg
    assert "not ported" in msg and "remove the key or port the mechanism" in msg


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


@pytest.mark.parametrize("rung", ["fp16", "fp4", "lbit", "ternary", "int4"])
def test_quant_ladder_rungs_refuse_at_load(rung):
    """(e) The training-time rungs refuse per rung, by name, at LOAD.

    fp8/nvfp4 graduated to STORAGE rungs (2026-10-15, `tests/
    test_ladder_storage.py`): they pack the factor bank and the checkpoint,
    fp32 masters stay the optimizer's object. What still refuses is the
    TRAINING-time set — LittleBit's gates G-A..G-D must close first.
    """
    with pytest.raises(LoudError) as e:
        parse_preset(f'd_model = 64\nn_heads = 4\nhead_dim = 16\nquant = "{rung}"\n')
    msg = str(e.value)
    assert rung in msg and "implemented rungs are the STORE levels" in msg
    # And the shipped presets use only the ported rung.
    assert ModelConfig().quant == "none"


def test_train_key_refusals():
    """The train half has the same two rules."""
    with pytest.raises(LoudError) as e:
        load_run("small", train_overrides={"engram_ram": True})
    assert "engram_ram" in str(e.value) and "not ported" in str(e.value)
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
    # Anything else is not.
    hotter = replace(run, train=replace(run.train, lr=run.train.lr * 2))
    assert hotter.diff_keys(run) == ["train.lr"]
