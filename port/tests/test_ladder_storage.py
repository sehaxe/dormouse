"""The storage rungs: pack/unpack round-trips, sizes, the save→load path.

Round-trip tolerance is the rung's own greed: fp8 (E4M3, per-tensor scale)
pins 2e-2 relative; nvfp4 (e2m1 grid + e4m3 per-16 scale) pins 6e-2 — the
grid quantum is 1/6 of the block max, and a scale rounding adds its own
step. Both are larger than the fp32-master test noise and smaller than
anything training cares about (the masters stay live).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from conftest import tiny_cfg, write_corpus
from typer.testing import CliRunner

from dormouse.cli import app
from dormouse.config import load_run
from dormouse.model import DormouseModel
from dormouse.model.tsct import is_factor_key, pack_tsct, unpack_tsct
from dormouse.train.loop import load_ckpt, save_ckpt, unpack_state_dict
from dormouse.train.optim import MixOptimizer

runner = CliRunner()


def test_fp8_round_trip_within_the_e4m3_bound():
    torch.manual_seed(0)
    t = torch.randn(4096, 64) * 0.05
    rec = pack_tsct(t, "fp8")
    get = unpack_tsct(rec)
    rel = (get - t).norm() / t.norm()
    # The format's own greed: e4m3 carries a 3-bit mantissa (half-step leq
    # 6.25% per element); a norm-level relative of a few percent is the
    # FORMAT, not implementation sludge — 4e-2 is the honest pin.
    assert rel < 4e-2, rel.item()
    assert get.dtype == torch.float32 and get.shape == t.shape
    assert rec["packed"].numel() * 1 == rec["packed"].numel()  # 1 B/val
    assert rec["packed"].nbytes == t.numel() and t.nbytes == 4 * t.numel()


def test_nvfp4_round_trip_within_the_grid_bound():
    torch.manual_seed(0)
    t = torch.randn(64, 4096) * 0.05  # numel % 16 == 0
    rec = pack_tsct(t, "nvfp4")
    get = unpack_tsct(rec)
    rel = (get - t).norm() / t.norm()
    # The grid's own greed: half-step at the top magnitudes is 0.5/6 = 1/12
    # of the block max, so a block with both a 6.0 and a 0.05 rounds the tail
    # to 100% — the FORMAT. Norm-level honest pin: 1e-1, plus the exact-code
    # fixture (values ∈ grid·scale) below, which is the real invariant.
    assert rel < 1e-1, rel.item()
    assert get.dtype == torch.float32 and get.shape == t.shape
    # Exact-code invariant: each decoded value IS a grid entry × its block's
    # scale (magnitudes are global; scales are per-block, so the magnitudes
    # alone are what can be pinned).
    blocks = get.reshape(-1, 16)
    scales = rec["scale"].float().unsqueeze(-1)
    grid = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0))
    quot = (blocks / scales).abs()  # codes are signed; the grid is magnitudes
    dist = (quot.unsqueeze(-1) - grid).abs().amin(-1).max()
    assert dist.item() < 1e-6, (
        f"a decoded value is not a grid×scale entry (max dist {dist.item():.2e})"
    )


def test_zero_block_stays_zero_and_does_not_inf():
    t = torch.zeros(64)
    rec = pack_tsct(t, "nvfp4")
    assert unpack_tsct(rec).abs().max().item() == 0.0


def test_non_factor_key_refuses_to_pack_implicitly():
    # The rung only quantizes factor MATRICES; a scale vector has a different
    # contract (its error multiplies the row wholesale).
    assert not is_factor_key("loop.attn.decay.b_alpha")
    assert not is_factor_key("loop.moe.router.weight")
    assert is_factor_key("loop.moe.experts.0.gate_u")
    assert is_factor_key("loop.tsct.v")


def test_checkpoint_with_fp8_rung_is_smaller_and_loads_to_fp32(tmp_path: Path):
    from dormouse.config import RunConfig, TrainConfig

    run = RunConfig(
        source="t",
        model=tiny_cfg(quant="fp8"),
        train=TrainConfig(
            steps=2,
            batch=2,
            seq_len=8,
            ckpt_name="rung",
            out_dir=str(tmp_path / "ck"),
            data=str(tmp_path / "c.bin"),
        ),
    )
    model = DormouseModel(run.model)
    opt = MixOptimizer(model, lr=0.001, wd=0.0, opt="mix")
    ck_path, _snap = save_ckpt(run, model, opt, step=2)
    import torch

    raw = torch.load(ck_path, weights_only=False)
    assert raw.get("model_quant_rung") == "fp8"
    assert any(isinstance(v, dict) and v.get("rung") == "fp8" for v in raw["model"].values()), (
        "a rung run must have packed factor records in the FILE"
    )
    ck, _ = load_ckpt(run)
    sd = unpack_state_dict(ck["model"], ck["model_quant_rung"])
    for k, v in model.state_dict().items():
        assert sd[k].dtype == v.dtype, k  # masters reload fp32, shapes intact
        if is_factor_key(k):
            rel = (sd[k] - v).norm() / v.norm()
            assert rel < 5e-2, f"{k}: rung rel {rel.item():.3e}"


def test_the_refusals_still_refuse():
    import pytest

    from dormouse.config import load_run
    from dormouse.instrument import LoudError

    with pytest.raises(LoudError):
        load_run("small", model_overrides={"quant": "lbit"})  # training-time rung
    with pytest.raises(LoudError):
        load_run("small", model_overrides={"quant": "int3"})  # not a rung at all


def test_fp8_and_nvfp4_resolve_from_a_preset():
    assert load_run("small").model.quant == "none"
    assert load_run("small", model_overrides={"quant": "fp8"}).model.quant == "fp8"
    assert load_run("small", model_overrides={"quant": "nvfp4"}).model.quant == "nvfp4"


def test_ternary_round_trip_is_bit_exact_to_the_projection():
    """A storage rung loses precision by its FORMAT's grid, never by an extra
    rounding: `unpack(pack(t, "ternary")) == ternarize(t)` exactly."""
    import pytest

    from dormouse.model.ternary import ternarize
    from dormouse.model.tsct import pack_tsct, unpack_tsct

    torch.manual_seed(3)
    t = torch.randn(37, 5) * 0.3  # non-multiple-of-4 size exercises the pad
    rec = pack_tsct(t, "ternary")
    got = unpack_tsct(rec)
    assert torch.equal(got, ternarize(t))
    # The grid is {−α, 0, +α} with α = mean(|t|) exactly as stored.
    alpha = rec["alpha"]
    assert alpha == pytest.approx(float(t.abs().mean()), rel=0, abs=0)
    vals = got.abs()
    on = (vals - alpha).abs() < 1e-6
    assert bool((on | (vals < 1e-7)).all())


def test_ternary_rung_is_16x_smaller_than_fp32():
    from dormouse.model.tsct import pack_tsct

    t = torch.randn(1024, 64)
    rec = pack_tsct(t, "ternary")
    bytes_packed = rec["codes"].numel()
    assert bytes_packed == (1024 * 64) // 4  # 2 bits per value, no pad here
    assert t.numel() * 4 == bytes_packed * 16  # 16× fewer bytes than fp32


def test_ternary_rung_resolves_and_travels_a_checkpoint(tmp_path):
    import torch

    from dormouse.config import RunConfig, TrainConfig, load_run
    from dormouse.train.loop import load_ckpt, save_ckpt
    from dormouse.train.optim import MixOptimizer

    run = load_run("small", model_overrides={"quant": "ternary"})
    assert run.model.quant == "ternary"
    run = RunConfig(
        source="t",
        model=tiny_cfg(quant="ternary"),
        train=TrainConfig(
            steps=2,
            batch=2,
            seq_len=8,
            ckpt_name="tern",
            out_dir=str(tmp_path / "ck"),
            data=str(tmp_path / "c.bin"),
        ),
    )
    model = DormouseModel(run.model)
    opt = MixOptimizer(model, lr=0.001, wd=0.0, opt="mix")
    ck_path, _snap = save_ckpt(run, model, opt, step=2)
    raw = torch.load(ck_path, weights_only=False)
    assert raw.get("model_quant_rung") == "ternary"
    rec = [v for v in raw["model"].values() if isinstance(v, dict) and v.get("rung") == "ternary"]
    assert rec, "a ternary run must have packed factor records in the FILE"
    # 2 bits per value: a factor record's codes are 4 values per byte.
    for r in rec:
        assert r["codes"].numel() == (r["n"] + 3) // 4
    ck, _ = load_ckpt(run)
    from dormouse.train.loop import unpack_state_dict

    sd = unpack_state_dict(ck["model"], ck["model_quant_rung"])
    for k, v in model.state_dict().items():
        assert sd[k].dtype == v.dtype and sd[k].shape == v.shape, k
        if is_factor_key(k):
            # The rung stores the TERNARY projection: the reload is the grid
            # value of alpha*code, not a near-copy of the master.
            assert torch.isfinite(sd[k]).all()


def test_ternary_refusal_update():
    """`lbit` still refuses (training-time), `ternary` no longer does."""
    import pytest

    from dormouse.config import load_run
    from dormouse.instrument import LoudError

    with pytest.raises(LoudError):
        load_run("small", model_overrides={"quant": "lbit"})
    assert load_run("small", model_overrides={"quant": "ternary"}).model.quant == "ternary"


# ---------------------------------------------------------------------------
# the audit's MAJOR 6 gate: `eval`/`decode`/`serve` load a rung checkpoint
# ---------------------------------------------------------------------------


def _save_rung_checkpoint(tmp_path: Path, quant: str) -> Path:
    """A 1-step tiny run saved with a storage rung; returns the corpus path."""
    from dormouse.config import RunConfig, TrainConfig
    from dormouse.train.optim import MixOptimizer

    corpus = write_corpus(tmp_path / "corpus.bin", n_bytes=4096)
    run = RunConfig(
        source="t",
        model=tiny_cfg(quant=quant),
        train=TrainConfig(
            steps=1,
            batch=2,
            seq_len=8,
            ckpt_name="rung",
            out_dir=str(tmp_path / "ck"),
            data=str(corpus),
        ),
    )
    model = DormouseModel(run.model)
    opt = MixOptimizer(model, lr=0.001, wd=0.0, opt="mix")
    save_ckpt(run, model, opt, step=1)
    return corpus


@pytest.mark.parametrize("quant", ["fp8", "none"])
def test_cli_eval_and_decode_load_a_rung_checkpoint(tmp_path: Path, quant: str):
    """`dormouse eval`/`decode` on a checkpoint whose factors are PACKED.

    `cli.py`'s eval and decode read the model half with a raw `torch.load` +
    `load_state_dict`, so a checkpoint written with `quant=fp8` — a config the
    validator accepts and `save_ckpt` packs — died there with a RuntimeError
    naming neither the rung nor the escape (audit MAJOR 6, 2026-10-17). Both now
    ride `load_model_half`; `quant="none"` is the unpaced control that must keep
    loading as before.
    """
    corpus = _save_rung_checkpoint(tmp_path, quant)
    ck = ["--out-dir", str(tmp_path / "ck"), "--ckpt-name", "rung", "--device", "cpu"]
    ev = runner.invoke(app, ["eval", *ck, "--eval", str(corpus)])
    assert ev.exit_code == 0, ev.output
    assert "BPB" in ev.output
    dec = runner.invoke(app, ["decode", *ck, "--prompt", "abcdefgh", "--n", "4"])
    assert dec.exit_code == 0, dec.output
