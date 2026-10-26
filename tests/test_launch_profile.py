"""The launch profile's phase split, busy union, and the SKIP contract.

`scripts/launch_profile.py` needs a card, so the CPU suite gates what can go
wrong without one: a kernel name landing in the wrong phase, an overlapping
kernel double-counted in "GPU busy", or the preflight silently measuring on a
busy card. Each is a way the table could lie.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    """Load `scripts/<name>.py` by path — `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prof = _load("launch_profile")


def test_phase_names_are_the_contract_the_doc_quotes() -> None:
    assert set(prof.PHASES) == {
        "kda/fla",
        "moe",
        "ns",
        "qat refresh",
        "clip",
        "optimizer",
        "norms/elementwise",
        "other",
    }


def test_the_name_fallback_covers_every_measured_group() -> None:
    assert prof.phase_of("chunk_kda_fwd_kernel", None) == "kda/fla"
    assert prof.phase_of("prepare_wy_repr_bwd_kernel", None) == "kda/fla"
    assert prof.phase_of("_glu_kernel", None) == "moe"
    assert prof.phase_of("_grouped_gemm_pair_kernel", None) == "moe"
    assert prof.phase_of("_ns_chain_kernel", None) == "ns"
    assert prof.phase_of("multi_tensor_apply_kernel", None) == "optimizer"
    assert prof.phase_of("vectorized_elementwise_kernel<4, ...>", None) == "norms/elementwise"
    assert prof.phase_of("cutlass_80_tensorop_s1688gemm_128x128", None) == "other"


def test_a_seam_wins_over_the_name_and_the_nearest_seam_wins() -> None:
    # The bank's QAT projection happens INSIDE the MoE forward: the nearest
    # annotation must be qat refresh, not moe, or the refresh disappears into
    # the expert pass.
    assert prof.phase_of("vectorized_elementwise_kernel", "qat refresh") == "qat refresh"
    assert prof.phase_of("_glu_kernel", "clip") == "clip"
    assert prof.phase_of("_ns_chain_kernel", "optimizer") == "optimizer"


def _kernel(name: str, duration: float = 1.0):
    return SimpleNamespace(name=name, duration=duration)


def _event(name: str, kernels=(), parent=None, user: bool = False):
    return SimpleNamespace(
        name=name,
        kernels=list(kernels),
        cpu_parent=parent,
        is_user_annotation=user,
        device_type=prof.CPU,
    )


def test_kernel_rows_attributes_by_the_cpu_parent_chain() -> None:
    moe = _event("moe", user=True)
    qat = _event("qat refresh", parent=moe, user=True)
    bmm = _event("aten::bmm", kernels=[_kernel("cutlass_gemm", 5.0)], parent=moe)
    mul = _event("aten::mul", kernels=[_kernel("vectorized_elementwise", 2.0)], parent=qat)
    device = SimpleNamespace(device_type=prof.CUDA, name="cutlass_gemm")
    fake = SimpleNamespace(events=lambda: [moe, qat, bmm, mul, device])
    rows = prof.kernel_rows(fake)
    assert rows == [
        ("cutlass_gemm", 5.0, "moe"),
        ("vectorized_elementwise", 2.0, "qat refresh"),
    ]


def test_busy_union_never_double_counts_overlap() -> None:
    assert prof.union_us([]) == 0.0
    assert prof.union_us([(0.0, 10.0)]) == 10.0
    assert prof.union_us([(0.0, 10.0), (5.0, 15.0)]) == 15.0
    assert prof.union_us([(0.0, 10.0), (2.0, 3.0)]) == 10.0
    assert prof.union_us([(20.0, 30.0), (0.0, 10.0)]) == 20.0
    # Adjacent spans are two kernels, not a gap: 10 + 10.
    assert prof.union_us([(0.0, 10.0), (10.0, 20.0)]) == 20.0


def test_skip_path_measures_nothing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(prof, "preflight", lambda: "the card is busy")
    assert prof.main() == 0
    out = capsys.readouterr().out
    assert "SKIP" in out
    assert "nothing was measured" in out
