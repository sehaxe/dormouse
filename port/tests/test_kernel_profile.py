"""The kernel profile's CPU gates: the maths, the names, the JSON, the refusal.

The instrument needs a card for its real run, so everything that can lie
WITHOUT one is pinned here: a percentage that does not sum to 100, a backward
kernel attributed to the wrong forward region, a seam vocabulary with a
duplicate, a JSON that cannot be diffed, and a busier card that still profiles.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    """Load `scripts/kernel_profile.py` by path — `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "kernel_profile", ROOT / "scripts" / "kernel_profile.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # `dataclass` resolves the defining module through `sys.modules`; a
    # by-path load without this registration fails at the first @dataclass.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


kp = _load()


def _kernel(name: str, duration: float):
    return SimpleNamespace(name=name, duration=duration)


def _event(
    name: str,
    *,
    scope: int = 0,
    seq: int = -1,
    parent=None,
    kernels=(),
    nbytes: int = 0,
):
    return SimpleNamespace(
        name=name,
        scope=scope,
        sequence_nr=seq,
        cpu_parent=parent,
        kernels=list(kernels),
        self_device_memory_usage=nbytes,
        device_type=kp.CPU,
    )


def test_region_names_and_seam_keys_are_unique() -> None:
    assert len(kp.REGIONS) == len(set(kp.REGIONS))
    assert set(kp.REGIONS) == {
        "embed",
        "norm/epilogue",
        "kda_proj",
        "kda_recurrent",
        "csa_fine",
        "csa_coarse",
        "engram",
        "moe_dispatch",
        "moe_gemm",
        "attnres",
        "mhc_write",
        "head/ce",
        "optimizer",
        "clip",
    }
    keys = [(spec.module, spec.target, spec.attr) for spec in kp.SEAM_SPECS]
    assert len(keys) == len(set(keys)), "two seams would wrap the same callable"
    assert all(spec.region in kp.REGIONS for spec in kp.SEAM_SPECS)


def test_aggregation_percentages_sum_to_100_and_mean_is_total_over_calls() -> None:
    rows = kp.aggregate(
        [
            kp.KernelSample("a", "kda_proj", "gemm", 60.0, count=3),
            kp.KernelSample("b", "moe_gemm", "gemm", 30.0, count=2),
            kp.KernelSample("c", "csa_fine", "gemm", 10.0, count=1),
        ]
    )
    assert sum(row.pct for row in rows) == pytest.approx(100.0)
    for row in rows:
        assert row.pct == pytest.approx(100.0 * row.us / 100.0)
        assert kp.mean_us(row) == pytest.approx(row.us / row.count)
    assert rows[0].name == "a"
    assert kp.mean_us(rows[0]) == pytest.approx(20.0)


def test_the_same_kernel_in_two_regions_is_two_rows_and_merges_within_one() -> None:
    rows = kp.aggregate(
        [
            kp.KernelSample("x", "kda_proj", "gemm", 2.0, count=1),
            kp.KernelSample("x", "kda_proj", "gemm", 3.0, count=2),
            kp.KernelSample("x", "moe_gemm", "gemm", 5.0, count=4),
        ]
    )
    assert len(rows) == 2
    merged = next(row for row in rows if row.region == "kda_proj")
    assert merged.us == pytest.approx(5.0)
    assert merged.count == 3
    assert kp.mean_us(merged) == pytest.approx(5.0 / 3)


def test_gpu_busy_is_a_union_not_a_sum() -> None:
    assert kp.union_us([]) == 0.0
    assert kp.union_us([(0.0, 10.0), (5.0, 15.0)]) == 15.0
    assert kp.union_us([(0.0, 10.0), (10.0, 20.0)]) == 20.0
    assert kp.union_us([(20.0, 30.0), (0.0, 10.0)]) == 20.0


def test_a_backward_kernel_takes_the_forward_op_s_region() -> None:
    region = _event("kda_proj", scope=7)
    forward = _event(
        "aten::addmm",
        scope=0,
        seq=7,
        parent=region,
        kernels=[_kernel("cutlass_80_tensorop_s1688gemm", 5.0)],
        nbytes=4096,
    )
    engine = _event("autograd::engine::evaluate_function: AddmmBackward0", scope=1, seq=7)
    backward = _event(
        "aten::mm",
        scope=1,
        seq=-1,
        parent=engine,
        kernels=[_kernel("cutlass_80_tensorop_s1688gemm", 7.0)],
    )
    inside = _event(
        "aten::mul",
        scope=0,
        seq=-1,
        parent=region,
        kernels=[_kernel("vectorized_elementwise_kernel", 1.0)],
    )
    samples = kp.collect_samples([region, forward, engine, backward, inside], demangle=False)
    assert [sample.region for sample in samples] == ["kda_proj", "kda_proj", "kda_proj"]
    assert [sample.us for sample in samples] == [5.0, 7.0, 1.0]


def test_memory_bytes_are_split_only_when_one_op_launched_many_kernels() -> None:
    one = _event(
        "aten::mm",
        kernels=[_kernel("a_kernel", 1.0)],
        nbytes=100,
    )
    many = _event(
        "aten::add",
        kernels=[_kernel("b_kernel", 1.0), _kernel("c_kernel", 1.0)],
        nbytes=90,
    )
    samples = kp.collect_samples([one, many], demangle=False)
    assert sum(sample.bytes for sample in samples) == pytest.approx(190.0)
    exact = [sample for sample in samples if sample.name == "a_kernel"][0]
    split = [sample for sample in samples if sample.name != "a_kernel"]
    assert exact.bytes == pytest.approx(100.0) and not exact.bytes_estimated
    assert all(sample.bytes == pytest.approx(45.0) for sample in split)
    assert all(sample.bytes_estimated for sample in split)


def test_names_families_and_truncation() -> None:
    assert kp.canonical_name("  void   foo<int>(int, float)  ") == "void foo<int>"
    assert len(kp.short_name("x" * 100, 20)) == 20
    assert kp.family_of("cutlass_80_tensorop_s1688gemm_128x128") == "gemm"
    assert kp.family_of("void at::native::vectorized_elementwise_kernel<4, ...>") == "pointwise"
    assert kp.family_of("chunk_kda_fwd_kernel") == "attention"
    assert kp.family_of("rms_norm_kernel") == "norm"
    assert kp.family_of("mystery") == "other"


def test_install_and_restore_every_seam_on_the_tiny_model() -> None:
    """A seam that cannot resolve must fail HERE, not after a paid GPU run."""
    from conftest import tiny_model

    model = tiny_model()
    seams, _skipped = kp.install_regions(model)
    keys = [(id(seam.owner), seam.attr) for seam in seams]
    assert len(keys) == len(set(keys)), "the same callable was wrapped twice"
    assert all(seam.region in kp.REGIONS for seam in seams)
    assert any(seam.region == "embed" for seam in seams)
    assert any(seam.region == "optimizer" for seam in seams)
    kp.restore_seams(seams)

    # A restore that left an instance `forward` behind would make this second
    # install replace its own wrapper (the "already wrapped" skip is the tell).
    again, skipped_again = kp.install_regions(model)
    assert len(again) == len(seams)
    assert skipped_again == _skipped
    kp.restore_seams(again)


def test_step_hook_counts_profiler_steps_and_stops() -> None:
    import dormouse.stepaudit as stepaudit

    class FakeProfiler:
        def __init__(self) -> None:
            self.calls = 0

        def step(self) -> None:
            self.calls += 1

    prof = FakeProfiler()
    seam = kp.install_step_hook(prof, 2)
    try:
        stepaudit.end_step()
        with pytest.raises(kp._StopTraining):
            stepaudit.end_step()
    finally:
        kp.restore_seams([seam])
    assert prof.calls == 2
    assert stepaudit.end_step.__name__ == "end_step", "the trainer hook was not restored"


def test_json_roundtrips_and_the_diff_finds_the_moved_row(tmp_path: Path) -> None:
    rows = kp.aggregate(
        [
            kp.KernelSample("gemm_kernel", "kda_proj", "gemm", 60.0, count=3),
            kp.KernelSample("pointwise", "moe_dispatch", "pointwise", 40.0, count=5),
        ]
    )
    families = kp.region_family_rows(rows)
    doc = kp.payload(rows, families, {"checkpoint": "a", "step_ms": 100.0})
    path = tmp_path / "a.json"
    kp.write_json(path, doc)
    assert kp.load_json(path) == doc
    assert json.loads(json.dumps(doc)) == doc

    moved = json.loads(json.dumps(doc))
    moved["kernels"][0]["us"] += 10.0
    moved["kernels"][0]["pct"] += 1.0
    moved["kernels"][0]["count"] += 1
    diff = kp.diff_kernels(doc["kernels"], moved["kernels"])
    assert diff[0]["name"] == doc["kernels"][0]["name"]
    assert diff[0]["d_us"] == pytest.approx(10.0)
    assert diff[0]["d_calls"] == 1


def test_report_prints_both_tables(capsys) -> None:
    rows = kp.aggregate(
        [
            kp.KernelSample("cutlass_80_tensorop_s1688gemm", "kda_proj", "gemm", 60.0),
            kp.KernelSample("vectorized_elementwise_kernel", "moe_dispatch", "pointwise", 40.0),
        ]
    )
    meta = {
        "checkpoint": "m400b",
        "batch": 8,
        "seq_len": 1024,
        "warmup": 3,
        "measured": 5,
        "step_ms": 100.0,
        "kernel_ms_per_step": 100.0,
        "gpu_busy_ms_per_step": 90.0,
        "kernels_per_step": 2,
        "launches_per_step": 3,
        "bytes": 0.0,
        "attributed_pct": 100.0,
    }
    kp.print_report(rows, kp.region_family_rows(rows), meta)
    out = capsys.readouterr().out
    assert "kernels by GPU time" in out
    assert "module x kernel family" in out
    assert "%step" in out
    assert "cutlass_80_tensorop_s1688gemm" in out


def test_a_busy_card_refuses_with_exit_2_and_writes_nothing(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    monkeypatch.setattr(kp, "preflight", lambda device: "a compute process holds the card")
    out = tmp_path / "never.json"
    rc = kp.main(["--checkpoint", str(tmp_path / "missing.pt"), "--json", str(out)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err
    assert "a compute process holds the card" in err
    assert not out.exists()


def test_diff_mode_needs_no_card(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(kp, "preflight", lambda device: pytest.fail("diff must not preflight"))
    rows = kp.aggregate([kp.KernelSample("k", "kda_proj", "gemm", 10.0, count=1)])
    doc = kp.payload(rows, kp.region_family_rows(rows), {"checkpoint": "a"})
    before, after = tmp_path / "a.json", tmp_path / "b.json"
    kp.write_json(before, doc)
    kp.write_json(after, doc)
    assert kp.main(["--diff", str(before), str(after)]) == 0
    assert "diff:" in capsys.readouterr().out


def test_a_non_cuda_device_is_refused_before_any_card_call(monkeypatch) -> None:
    monkeypatch.setattr(kp, "preflight", lambda device: pytest.fail("preflight must not run"))
    assert kp.main(["--device", "cpu"]) == 2
