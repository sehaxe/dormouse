r"""The packed QAT export: codec bit-exactness, the container, the round trip.

The pins are MEASURED on the tiny fixture (`conftest.tiny_cfg`, 64 885 params,
51 tensors, torch 2.13, CPU, 2026-10-18) and on the `medium` preset on `meta`
(no storage). The codec's own contract: `unpack_ternary(pack_ternary(t))` is
BIT-EXACT to `ternary.ternarize(t)` — the file loses precision to its STATED
formats (fp16 scale, bf16 load), never to an extra rounding. The logits bound
is `verify_roundtrip`'s measured 8.8e-3 with room for a different init.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from conftest import TINY, tiny_cfg, tiny_model

from dormouse.config import load_preset
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.model.export import (
    export_plan,
    export_qat,
    load_qat,
    pack_mxfp4,
    pack_ternary,
    unpack_mxfp4,
    unpack_ternary,
    verify_roundtrip,
)
from dormouse.model.mxfp import fake_quant_mx
from dormouse.model.ternary import ternarize
from dormouse.model.tsct import factor_qat_project, is_factor_key

#: The tiny fixture's export, measured 2026-10-18.
TINY_PAYLOAD = 102_032
TINY_FACTORS_BYTES = 4_006
TINY_FACTOR_PARAMS = 15_872
TINY_FACTOR_KEYS = 19
TINY_SCALE_KEYS = 12
TINY_BF16_BYTES = 129_770  # 2 B/param over all 64 885 params
#: measured 8.82e-3 on the tiny fixture; the stated bound keeps ~2x margin.
LOGITS_BOUND = 2e-2
#: measured 1.25e-2 for the mxfp4 rung.
MXFP4_LOGITS_BOUND = 3e-2
#: The load's two roundings: bf16 values (2^-8) + fp16 scale (2^-11).
FACTOR_REL_BOUND = 2**-8 + 2**-11
SCALE_LEAVES = ("s", "gate_s", "up_s")


def _tiny_export(tmp_path: Path, **over) -> tuple[DormouseModel, Path]:
    torch.manual_seed(0)
    model = tiny_model(**over)
    path = tmp_path / "tiny.qat.pt"
    export_qat(model, path)
    return model, path


# ---------------------------------------------------------------------------
# (a) the ternary codec — bit-exact, and LOUD where it cannot pack
# ---------------------------------------------------------------------------


def test_pack_ternary_pins_literal_codes_and_scale() -> None:
    """`[1, −1, 0.25, 0.25]` ⇒ α = 0.625, codes [1, 2, 0, 0] ⇒ byte 9.

    0.4 rounds to 0 (the dead zone), ±1.6 clamps to ±1, and the pack is
    low-code-first: `1 | 2<<2 | 0<<4 | 0<<6 = 9`.
    """
    t = torch.tensor([1.0, -1.0, 0.25, 0.25])
    codes, scale = pack_ternary(t)
    assert scale == 0.625, "(1 + 1 + 0.25 + 0.25) / 4"
    assert codes.tolist() == [9]
    got = unpack_ternary(codes, scale, (4,))
    assert got.tolist() == [0.625, -0.625, 0.0, 0.0]
    assert torch.equal(got, ternarize(t))


def test_pack_unpack_is_bit_exact_to_ternarize_fp32() -> None:
    torch.manual_seed(0)
    t = torch.randn(37, 64)  # 2368 elements, a multiple of 4
    codes, scale = pack_ternary(t)
    got = unpack_ternary(codes, scale, tuple(t.shape), t.dtype)
    assert torch.equal(got, ternarize(t))
    assert got.dtype == t.dtype and got.shape == t.shape


def test_pack_unpack_is_bit_exact_to_ternarize_bf16() -> None:
    """The scale is the fp32 absmean `ternarize` itself divides by, not a bf16 mean."""
    torch.manual_seed(1)
    t = (torch.randn(16, 8) * 0.7).bfloat16()
    codes, scale = pack_ternary(t)
    got = unpack_ternary(codes, scale, tuple(t.shape), t.dtype)
    assert torch.equal(got, ternarize(t))


def test_pack_refuses_odd_shapes_non_float_and_empty() -> None:
    with pytest.raises(LoudError, match="not a multiple of 4"):
        pack_ternary(torch.randn(6))
    with pytest.raises(LoudError, match="float tensors"):
        pack_ternary(torch.arange(8))
    with pytest.raises(LoudError, match="empty"):
        pack_ternary(torch.empty(0))


def test_pack_refuses_non_finite_and_unpack_checks_its_contract() -> None:
    with pytest.raises(LoudError, match="non-finite"):
        pack_ternary(torch.tensor([float("nan"), 1.0, 1.0, 1.0]))
    codes, scale = pack_ternary(torch.randn(8))
    with pytest.raises(AssertionError, match="4 codes per byte"):
        unpack_ternary(codes, scale, (16,))  # 4 codes cover 4 elements, not 16
    with pytest.raises(AssertionError, match="uint8"):
        unpack_ternary(codes.to(torch.int32), scale, (8,))


def test_codes_are_uint8_and_the_packing_is_two_bits_per_element() -> None:
    torch.manual_seed(2)
    t = torch.randn(64, 32)
    codes, _ = pack_ternary(t)
    assert codes.dtype == torch.uint8
    assert codes.numel() == t.numel() // 4
    assert codes.numel() * 4 == t.numel()
    fields = torch.cat([(codes >> s) & 3 for s in (0, 2, 4, 6)])
    assert fields.numel() == t.numel()
    assert int(fields.max()) <= 2, "the 2-bit codebook is {0, 1, 2}; 3 is unused"
    # and a corrupted code 3 refuses instead of silently decoding to 0
    corrupt = codes.clone()
    corrupt[0] |= 0b11_00_00_00
    with pytest.raises(AssertionError, match="not a ternary level"):
        unpack_ternary(corrupt, 0.5, tuple(t.shape))


# ---------------------------------------------------------------------------
# (b) the container and its report
# ---------------------------------------------------------------------------


def test_export_report_pins_the_tiny_size_arithmetic(tmp_path: Path) -> None:
    model, path = _tiny_export(tmp_path)
    rep = export_qat(model, path)
    sd = model.state_dict()
    factors = [k for k in sd if is_factor_key(k)]
    scales = [k for k in sd if k.rpartition(".")[2] in SCALE_LEAVES]
    rest = [k for k in sd if k not in set(factors) | set(scales)]
    assert (len(factors), len(scales), len(rest)) == (TINY_FACTOR_KEYS, TINY_SCALE_KEYS, 20)
    assert rep.counts["factors"] == TINY_FACTOR_PARAMS
    assert rep.groups["factors"] == TINY_FACTORS_BYTES
    # the arithmetic, from the shapes alone: n/4 code bytes + 2 B fp16 scale
    assert rep.groups["factors"] == sum(sd[k].numel() // 4 + 2 for k in factors)
    assert rep.groups["scales"] == sum(sd[k].numel() * 2 for k in scales)
    assert rep.groups["bf16"] == sum(sd[k].numel() * 2 for k in rest)
    assert rep.total_bytes == TINY_PAYLOAD == sum(rep.groups.values())
    assert rep.bf16_bytes == TINY_BF16_BYTES
    assert rep.fp32_bytes == 2 * TINY_BF16_BYTES
    assert rep.ratio_vs_bf16 == pytest.approx(TINY_BF16_BYTES / TINY_PAYLOAD, rel=1e-9)
    assert rep.ratio_vs_fp32 == pytest.approx(2 * TINY_BF16_BYTES / TINY_PAYLOAD, rel=1e-9)


def test_export_plan_matches_the_medium_preset_arithmetic() -> None:
    """The E=256 arithmetic on `meta` — no storage, the report function alone."""
    rep = export_plan(DormouseModel(load_preset("medium"), device="meta"))
    assert rep.counts["factors"] == 79_413_120
    assert rep.groups["factors"] == 19_854_830  # n/4 + 2 B x 775 factors
    assert rep.total_bytes == 144_849_912


def test_exported_file_size_is_what_the_report_says(tmp_path: Path) -> None:
    model, path = _tiny_export(tmp_path)
    rep = export_qat(model, path)
    assert rep.file_bytes == path.stat().st_size
    assert rep.file_bytes > rep.total_bytes, "torch.save's zip metadata is on disk too"
    assert rep.total_bytes == TINY_PAYLOAD


def test_exported_records_are_uint8_codes_and_fp16_scale(tmp_path: Path) -> None:
    _, path = _tiny_export(tmp_path)
    ck = torch.load(path, weights_only=True)
    assert ck["format"] == "dormouse-qat-export"
    assert ck["version"] == 1 and ck["mode"] == "ternary"
    rec = ck["tensors"]["loop.out_proj.u"]
    assert rec["rung"] == "ternary"
    assert rec["codes"].dtype == torch.uint8
    assert rec["codes"].numel() * 4 == rec["n"] == 64 * 8
    assert rec["scale"].dtype == torch.float16 and rec["scale"].ndim == 0
    assert rec["shape"] == (64, 8)
    assert ck["tensors"]["loop.out_proj.s"].dtype == torch.float16
    assert ck["tensors"]["loop.moe.experts.0.gate_s"].dtype == torch.float16
    assert ck["tensors"]["embed.weight"].dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# (c) the load and the round trip
# ---------------------------------------------------------------------------


def test_loaded_factors_are_the_projection_and_the_projector_is_off(tmp_path: Path) -> None:
    model, path = _tiny_export(tmp_path)
    loaded = load_qat(path, model.cfg)
    sd0, sd1 = model.state_dict(), loaded.state_dict()
    worst = 0.0
    for k, v in sd0.items():
        if is_factor_key(k):
            proj = factor_qat_project(v, "ternary", 1.0)
            alpha = max(float(proj.abs().max()), 1e-30)
            worst = max(worst, float((sd1[k] - proj).abs().max()) / alpha)
    assert worst <= FACTOR_REL_BOUND, worst
    # The projection is IN the values: re-projecting would re-scale by the
    # non-zero fraction (ternarize is not idempotent), so the loader turns it off.
    assert all(m.factor_qat == "none" for m in loaded.modules() if hasattr(m, "factor_qat"))
    assert loaded.cfg.factor_qat == "none"
    # Non-factors ride at bf16 — the file's stated precision for them.
    e0, e1 = sd0["embed.weight"], sd1["embed.weight"]
    assert torch.equal(e1, e0.to(torch.bfloat16).to(e1.dtype))


def test_roundtrip_logits_within_the_stated_bound_on_tiny(tmp_path: Path) -> None:
    model, path = _tiny_export(tmp_path, factor_qat="ternary", factor_qat_alpha=0.5)
    torch.manual_seed(3)
    x = torch.randint(0, TINY["vocab"], (2, 16))
    diff = verify_roundtrip(model, path, (x,))
    assert 0.0 <= diff < LOGITS_BOUND, diff
    # verify forces the projection for its reference and RESTORES the model's
    # own setting on the way out.
    assert model.loop.out_proj.factor_qat == "ternary"
    assert model.loop.out_proj.factor_qat_alpha == 0.5


def test_load_qat_refuses_a_foreign_file(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.pt"
    torch.save({"hello": 1}, bogus)
    with pytest.raises(LoudError, match="not a dormouse QAT export"):
        load_qat(bogus, tiny_cfg())
    with pytest.raises(LoudError, match="does not exist"):
        load_qat(tmp_path / "missing.pt", tiny_cfg())


def test_load_qat_refuses_a_key_set_mismatch_and_names_the_host_table_case(
    tmp_path: Path,
) -> None:
    model, path = _tiny_export(tmp_path)
    with pytest.raises(LoudError, match="different model config"):
        load_qat(path, tiny_cfg(use_engram=False))  # the file has keys the cfg lacks
    # The host-table path: `to_host` deletes the parameter, so the export has no
    # `engram.tables` and the load must name the `.ngram` sidecar, not guess.
    host = tiny_model()
    host.loop.engram.to_host()
    host_path = tmp_path / "host.qat.pt"
    export_qat(host, host_path)
    with pytest.raises(LoudError, match="ngram"):
        load_qat(host_path, tiny_cfg())


# ---------------------------------------------------------------------------
# (d) the optional mxfp4 rung — the same report, the mxfp codec
# ---------------------------------------------------------------------------


def test_mxfp4_pack_is_the_projectors_values() -> None:
    torch.manual_seed(4)
    t = torch.randn(64, 8) * 0.3  # rank 8 exercises the 32-pad path
    packed, scales, pad = pack_mxfp4(t)
    assert pad == 24 and packed.dtype == torch.uint8 and scales.dtype == torch.uint8
    got = unpack_mxfp4(packed, scales, tuple(t.shape), pad, dtype=torch.float32)
    assert torch.equal(got, fake_quant_mx(t, "mxfp4"))


def test_mxfp4_export_roundtrip_on_tiny(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = tiny_model()
    path = tmp_path / "tiny.mxfp4.pt"
    rep = export_qat(model, path, mode="mxfp4")
    assert rep.mode == "mxfp4"
    loaded = load_qat(path, model.cfg)
    sd0, sd1 = model.state_dict(), loaded.state_dict()
    # e2m1 magnitudes x power-of-two scales are bf16-exact, so the factor load
    # is exact (unlike ternary's fp16 scale): the logits differ only through the
    # bf16 non-factors.
    for k, v in sd0.items():
        if is_factor_key(k):
            proj = factor_qat_project(v, "mxfp4", 1.0)
            assert torch.equal(sd1[k], proj.to(sd1[k].dtype)), k
    torch.manual_seed(5)
    x = torch.randint(0, TINY["vocab"], (2, 16))
    assert verify_roundtrip(model, path, (x,)) < MXFP4_LOGITS_BOUND


# ---------------------------------------------------------------------------
# (e) the preset and the demo — the CPU/CI path
# ---------------------------------------------------------------------------


def test_tiny_preset_matches_the_test_fixture() -> None:
    """`configs/tiny.toml` and `conftest.TINY` are one shape, pinned together."""
    assert load_preset("tiny") == tiny_cfg()


def test_demo_runs_on_cpu_with_the_tiny_preset(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "qat_export_demo", root / "scripts" / "qat_export_demo.py"
    )
    assert spec is not None and spec.loader is not None
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    out = tmp_path / "tiny.qat.pt"
    rc = demo.main(
        ["--preset", "tiny", "--fresh", "--out", str(out), "--batch", "1", "--seq-len", "8"]
    )
    assert rc == 0
    assert out.is_file() and out.stat().st_size > 0
