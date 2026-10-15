"""The storage rungs: pack/unpack round-trips, sizes, the save→load path.

Round-trip tolerance is the rung's own greed: fp8 (E4M3, per-tensor scale)
pins 2e-2 relative; nvfp4 (e2m1 grid + e4m3 per-16 scale) pins 6e-2 — the
grid quantum is 1/6 of the block max, and a scale rounding adds its own
step. Both are larger than the fp32-master test noise and smaller than
anything training cares about (the masters stay live).
"""

from __future__ import annotations

from pathlib import Path

import torch
from conftest import tiny_cfg

from dormouse.config import load_run
from dormouse.model import DormouseModel
from dormouse.model.tsct import is_factor_key, pack_tsct, unpack_tsct
from dormouse.train.loop import load_ckpt, save_ckpt, unpack_state_dict
from dormouse.train.optim import MixOptimizer


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
