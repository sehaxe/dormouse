r"""The `.dormouse` artifact: codec bit-exactness, the container, the round trip.

The pins are MEASURED on the tiny fixture (`conftest.tiny_cfg`, 64 885 params,
51 tensors, torch 2.13, CPU, 2026-10-19) and on the `medium` preset on `meta`
(no storage). The codec's own contract: `unpack_ternary(pack_ternary(t))` is
BIT-EXACT to `ternary.ternarize(t)` — the file loses precision to its STATED
formats (the ternary grid, fp16 dense), never to an extra rounding, and the
ternary scale rides in the header as an exact JSON number rather than an fp16
copy. The logits bound is `verify_export`'s measured 5.5e-4 on the tiny
fixture with room for a different init.
"""

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import pytest
import torch
from conftest import TINY, tiny_cfg, tiny_model

from dormouse.config import load_preset
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.model.export import (
    ALIGN,
    MAGIC,
    engine_refusal,
    export_plan,
    load_export,
    pack_mxfp4,
    pack_ternary,
    read_export,
    unpack_mxfp4,
    unpack_ternary,
    verify_export,
    write_export,
)
from dormouse.model.mxfp import fake_quant_mx
from dormouse.model.ternary import ternarize
from dormouse.model.tsct import factor_qat_project, is_factor_key

#: The tiny fixture's export, measured 2026-10-19 with dense=fp16,
#: engram=fp32 (the defaults), mode=ternary.
TINY_PAYLOAD = 105_066
TINY_FACTORS_BYTES = 3_968
TINY_FACTOR_PARAMS = 15_872
TINY_FACTOR_KEYS = 19
TINY_SCALE_KEYS = 12
TINY_ENGRAM_BYTES = 6_144
TINY_FP16_BYTES = 129_770  # 2 B/param over all 64 885 params
#: measured 5.55e-3 on the tiny fixture (2026-10-19); the bound keeps ~4x margin.
LOGITS_BOUND = 2e-2
#: measured 1.25e-2 for the mxfp4 rung.
MXFP4_LOGITS_BOUND = 3e-2
#: The load's one rounding: dense fp16 (2^-11). The factors' codes are exact and
#: their scale is exact (a JSON number), so nothing else rounds.
FACTOR_REL_BOUND = 2**-11
SCALE_LEAVES = ("s", "gate_s", "up_s")


def _tiny_export(tmp_path: Path, **over) -> tuple[DormouseModel, Path]:
    torch.manual_seed(0)
    model = tiny_model(**over)
    path = tmp_path / "tiny.dormouse"
    write_export(model, path, preset="tiny", seed=1)
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
    rep = write_export(model, path, preset="tiny", seed=1)
    sd = model.state_dict()
    factors = [k for k in sd if is_factor_key(k)]
    scales = [k for k in sd if k.rpartition(".")[2] in SCALE_LEAVES]
    rest = [
        k for k in sd if k not in set(factors) | set(scales) and not k.endswith("engram.tables")
    ]
    assert (len(factors), len(scales), len(rest)) == (TINY_FACTOR_KEYS, TINY_SCALE_KEYS, 19)
    assert rep.counts["factors"] == TINY_FACTOR_PARAMS
    assert rep.groups["factors"] == TINY_FACTORS_BYTES
    # the arithmetic, from the shapes alone: n/4 code bytes (the scale is a
    # header number, not a blob), fp16 dense, the engram at its own rung
    assert rep.groups["factors"] == sum(sd[k].numel() // 4 for k in factors)
    assert rep.groups["scales"] == sum(sd[k].numel() * 2 for k in scales)
    assert rep.groups["engram"] == TINY_ENGRAM_BYTES
    assert rep.groups["dense"] == sum(sd[k].numel() * 2 for k in rest)
    assert rep.total_bytes == TINY_PAYLOAD == sum(rep.groups.values())
    assert rep.fp16_bytes == TINY_FP16_BYTES
    assert rep.fp32_bytes == 2 * TINY_FP16_BYTES
    assert rep.ratio_vs_fp16 == pytest.approx(TINY_FP16_BYTES / TINY_PAYLOAD, rel=1e-9)
    assert rep.ratio_vs_fp32 == pytest.approx(2 * TINY_FP16_BYTES / TINY_PAYLOAD, rel=1e-9)


def test_engram_dtype_is_a_rung_and_it_moves_the_bytes(tmp_path: Path) -> None:
    """The n-gram table is 33% of `small`: fp16 must halve exactly that group."""
    torch.manual_seed(0)
    model = tiny_model()
    rep32 = export_plan(model, "ternary", engram_dtype="fp32")
    rep16 = export_plan(model, "ternary", engram_dtype="fp16")
    assert rep32.groups["engram"] == 2 * rep16.groups["engram"] == TINY_ENGRAM_BYTES
    assert rep32.groups["dense"] == rep16.groups["dense"], "the rung is the table's own"
    path = tmp_path / "t16.dormouse"
    write_export(model, path, engram_dtype="fp16")
    f = read_export(path)
    got = f.tensor("loop.engram.tables")
    want = model.loop.engram.tables.detach().to(torch.float16).to(torch.float32)
    assert torch.equal(got, want)
    with pytest.raises(LoudError, match="fp32 or fp16"):
        export_plan(model, "ternary", engram_dtype="bf16")


def test_export_plan_matches_the_medium_preset_arithmetic() -> None:
    """The E=256 arithmetic on `meta` — no storage, the report function alone."""
    rep = export_plan(
        DormouseModel(load_preset("medium"), device="meta"), "ternary", engram_dtype="fp16"
    )
    assert rep.counts["factors"] == 79_413_120
    assert rep.groups["factors"] == 19_853_280  # n/4 x 775 factors, no scale blob
    assert rep.total_bytes == 144_848_362
    # the engram rung, on the same model: fp32 doubles that group and nothing else
    rep32 = export_plan(DormouseModel(load_preset("medium"), device="meta"))
    assert rep32.groups["engram"] == 2 * rep.groups["engram"]
    assert rep32.total_bytes - rep.total_bytes == rep.groups["engram"]


def test_exported_file_size_is_what_the_report_says(tmp_path: Path) -> None:
    model, path = _tiny_export(tmp_path)
    rep = write_export(model, path, preset="tiny", seed=1)
    assert rep.file_bytes == path.stat().st_size
    assert rep.file_bytes > rep.total_bytes, "the header and the alignment are on disk too"
    assert rep.file_bytes == rep.header_bytes + rep.total_bytes + rep.pad_bytes, (
        "header + payload + inter-blob alignment is the file, exactly: nothing else is in there"
    )
    assert rep.total_bytes == TINY_PAYLOAD


def test_container_layout_is_the_documented_one(tmp_path: Path) -> None:
    """magic, header length, JSON header, absolute 8-aligned offsets, dtypes."""
    model, path = _tiny_export(tmp_path)
    buf = path.read_bytes()
    assert buf[:8] == MAGIC
    header_len, flags = struct.unpack_from("<II", buf, 8)
    assert flags == 0
    header = json.loads(buf[16 : 16 + header_len].decode("utf-8"))
    assert header["format"] == "dormouse" and header["version"] == 1
    assert header["mode"] == "ternary" and header["preset"] == "tiny"
    assert header["dtypes"] == {"dense": "fp16", "engram": "fp32"}
    assert header["seed"] == 1 and header["commit"] is None
    assert header["model"]["d_model"] == TINY["d_model"]
    # every blob lives inside the file, 8-byte aligned, and past the header
    for rec in header["tensors"]:
        assert rec["offset"] % ALIGN == 0, rec["name"]
        assert rec["offset"] + rec["length"] <= len(buf), rec["name"]
        assert rec["offset"] >= 16 + header_len
    f = read_export(path)
    assert f.names() == set(model.state_dict())
    rec = f.record("loop.out_proj.u")
    assert rec["dtype"] == "ternary" and rec["shape"] == [64, 8]
    assert rec["length"] * 4 == 64 * 8
    assert isinstance(rec["scale"], float) and rec["scale"] > 0
    assert f.record("embed.weight")["dtype"] == "fp16"


def test_factors_unpack_bit_exact_to_ternarize_of_the_masters(tmp_path: Path) -> None:
    """The container's factor round trip is the codec's: no extra rounding.

    The codes are `ternarize`'s, and the scale is the exact `mean(|W|)` the
    header carries as a JSON number — so the loaded factor is BIT-EQUAL to
    `ternarize(master)`, not merely close (an fp16 scale would cost 2^-11).
    """
    model, path = _tiny_export(tmp_path)
    f = read_export(path)
    sd = model.state_dict()
    factors = [k for k in sd if is_factor_key(k)]
    assert len(factors) == TINY_FACTOR_KEYS
    for k in factors:
        assert torch.equal(f.tensor(k), ternarize(sd[k])), k


def test_export_refuses_a_missing_engram_table_and_a_non_finite_weight(tmp_path: Path) -> None:
    # A host-table run whose `.ngram` sidecar was never restored leaves a model
    # with no `engram.tables` at all — the export must name the sidecar, not
    # write a memory-less model under the run's name.
    stripped = tiny_model()
    del stripped.loop.engram.tables
    with pytest.raises(LoudError, match="ngram"):
        write_export(stripped, tmp_path / "stripped.dormouse")
    bad = tiny_model()
    with torch.no_grad():
        bad.lm_head.u[0, 0] = float("nan")
    with pytest.raises(LoudError, match="non-finite"):
        write_export(bad, tmp_path / "nan.dormouse")


def test_host_table_export_folds_the_memory_in(tmp_path: Path) -> None:
    """An `engram_ram` model's tables are not parameters — they must still travel."""
    torch.manual_seed(0)
    model = tiny_model()
    before = model.loop.engram.tables.detach().clone()
    model.loop.engram.to_host()  # `load_model_state` does this for an engram_ram ckpt
    assert "loop.engram.tables" not in model.state_dict()
    path = tmp_path / "host.dormouse"
    write_export(model, path)
    f = read_export(path)
    assert torch.equal(f.tensor("loop.engram.tables"), before)
    loaded = load_export(f)
    assert "loop.engram.tables" in loaded.state_dict()


def test_engine_refusal_names_every_field_the_engine_cannot_run(tmp_path: Path) -> None:
    """`small` is the engine's ONE polygon; anything else is refused by name."""
    assert engine_refusal(load_preset("small")) is None
    why = engine_refusal(load_preset("medium"))
    assert why is not None
    for field in ("d_model=1024", "n_heads=16", "d_ffn=2730"):
        assert field in why, why
    # and the refusal is in the FILE, so a reader knows before it loads weights
    torch.manual_seed(0)
    model = tiny_model()
    path = tmp_path / "tiny.dormouse"
    write_export(model, path, preset="tiny")
    f = read_export(path)
    assert f.header["architecture"] == "byte-lm"
    assert f.header["engine_refusal"] is not None
    assert f.engine_refusal() is not None


def test_exported_records_are_uint8_codes_and_the_scale_is_a_header_number(
    tmp_path: Path,
) -> None:
    _, path = _tiny_export(tmp_path)
    f = read_export(path)
    rec = f.record("loop.out_proj.u")
    assert rec["dtype"] == "ternary"
    codes = f._slice(rec["offset"], rec["length"], "u8")
    assert codes.dtype == torch.uint8
    assert codes.numel() * 4 == 64 * 8
    assert isinstance(rec["scale"], float)
    assert f.record("loop.out_proj.s")["dtype"] == "fp16"
    assert f.record("loop.moe.experts.0.gate_s")["dtype"] == "fp16"
    assert f.record("embed.weight")["dtype"] == "fp16"


# ---------------------------------------------------------------------------
# (c) the load and the round trip
# ---------------------------------------------------------------------------


def test_loaded_factors_are_the_projection_and_the_projector_is_off(tmp_path: Path) -> None:
    model, path = _tiny_export(tmp_path)
    loaded = load_export(path)
    sd0, sd1 = model.state_dict(), loaded.state_dict()
    for k, v in sd0.items():
        if is_factor_key(k):
            # bit-exact against `ternarize` itself — the codes are its signs and
            # the header's scale is its exact absmean, so no rounding happens
            assert torch.equal(sd1[k], ternarize(v)), k
            # and within the stated bound of the PROJECTOR's spelling, which is
            # the STE `w + (q − w)` and therefore one fp32 ulp off `q`
            proj = factor_qat_project(v, "ternary", 1.0)
            alpha = max(float(proj.abs().max()), 1e-30)
            assert float((sd1[k] - proj).abs().max()) / alpha <= FACTOR_REL_BOUND, k
    # The projection is IN the values: re-projecting would re-scale by the
    # non-zero fraction (ternarize is not idempotent), so the loader turns it off.
    assert all(m.factor_qat == "none" for m in loaded.modules() if hasattr(m, "factor_qat"))
    assert loaded.cfg.factor_qat == "none"
    # Non-factors ride at fp16 — the file's stated precision for them.
    e0, e1 = sd0["embed.weight"], sd1["embed.weight"]
    assert torch.equal(e1, e0.to(torch.float16).to(e1.dtype))


def test_roundtrip_logits_within_the_stated_bound_on_tiny(tmp_path: Path) -> None:
    model, path = _tiny_export(tmp_path, factor_qat="ternary", factor_qat_alpha=0.5)
    torch.manual_seed(3)
    x = torch.randint(0, TINY["vocab"], (2, 16))
    diff = verify_export(model, path, (x,))
    assert 0.0 <= diff < LOGITS_BOUND, diff
    # verify forces the projection for its reference and RESTORES the model's
    # own setting on the way out.
    assert model.loop.out_proj.factor_qat == "ternary"
    assert model.loop.out_proj.factor_qat_alpha == 0.5


def test_read_export_refuses_a_foreign_a_truncated_and_a_missing_file(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.pt"
    torch.save({"hello": 1}, bogus)
    with pytest.raises(LoudError, match="not a dormouse artifact"):
        read_export(bogus)
    with pytest.raises(LoudError, match="does not exist"):
        read_export(tmp_path / "missing.dormouse")
    _, path = _tiny_export(tmp_path)
    cut = tmp_path / "cut.dormouse"
    cut.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    with pytest.raises(LoudError, match="past the end"):
        read_export(cut)


def test_load_export_refuses_a_cfg_that_differs_from_the_header(tmp_path: Path) -> None:
    _, path = _tiny_export(tmp_path)
    with pytest.raises(LoudError, match="differs from the one given"):
        load_export(path, cfg=tiny_cfg(use_engram=False))
    # the header IS the authority: no cfg argument needed, and the model it
    # builds has exactly the file's keys
    loaded = load_export(path)
    assert set(loaded.state_dict()) == read_export(path).names()


def test_serve_reads_the_artifact_the_browser_reads(tmp_path: Path) -> None:
    """One format, three readers: `--export` is `load_export` behind a socket."""
    from dormouse.serve import build_server

    _, path = _tiny_export(tmp_path)
    httpd = build_server(tmp_path, "", export=str(path), port=0, device="cpu", log=lambda *_: None)
    try:
        assert httpd.state.ckpt_name == path.name
        assert httpd.state.seed == 1, "the header's seed, not a default"
        assert set(httpd.state.model.state_dict()) == read_export(path).names()
    finally:
        httpd.server_close()


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
    path = tmp_path / "tiny.mxfp4.dormouse"
    rep = write_export(model, path, mode="mxfp4")
    assert rep.mode == "mxfp4"
    f = read_export(path)
    rec = f.record("loop.out_proj.u")
    assert rec["dtype"] == "mxfp4" and "scales_offset" in rec and rec["pad"] == 24
    loaded = load_export(f)
    sd0, sd1 = model.state_dict(), loaded.state_dict()
    # e2m1 magnitudes x power-of-two scales are bf16-exact, so the factor load
    # is exact (unlike ternary's scale, which is exact too — but the codec's
    # grid is what both are exact TO): the logits differ only through the fp16
    # non-factors.
    for k, v in sd0.items():
        if is_factor_key(k):
            proj = factor_qat_project(v, "mxfp4", 1.0)
            assert torch.equal(sd1[k], proj.to(sd1[k].dtype)), k
    torch.manual_seed(5)
    x = torch.randint(0, TINY["vocab"], (2, 16))
    assert verify_export(model, path, (x,)) < MXFP4_LOGITS_BOUND


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
    out = tmp_path / "tiny.dormouse"
    rc = demo.main(
        ["--preset", "tiny", "--fresh", "--out", str(out), "--batch", "1", "--seq-len", "8"]
    )
    assert rc == 0
    assert out.is_file() and out.stat().st_size > 0
    assert read_export(out).header["preset"] == "tiny"
