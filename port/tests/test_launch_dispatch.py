"""The CPU dispatch counter: grouping, metadata exclusion, JSON and the refusal.

`scripts/launch_profile.py` grew a second instrument (the `--gpu-profile` card
profile is unchanged and still gated by `test_launch_profile.py`): a
`TorchDispatchMode` that counts ONE fwd+bwd+optimizer step on the CPU, groups by
`(aten op, module)` and writes/diffs a JSON.

There is no card here, so these gates cover what can make the table lie without
one: a metadata op counted as a launch, a module boundary misattributed, a JSON
that cannot be diffed, and a CUDA request that silently measures on a busy card.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prof = _load("launch_profile")


def _counter_for(module: torch.nn.Module, x: torch.Tensor):
    counter = prof.DispatchCounter()
    with counter.observe(module):
        module(x).sum().backward()
    return counter


def test_metadata_ops_are_not_launches() -> None:
    for op in ("aten::view", "aten::expand", "aten::transpose", "aten::detach", "aten::t"):
        assert prof.is_metadata(op), op
    for op in ("aten::addmm", "aten::bmm", "aten::topk", "aten::gather", "aten::copy_"):
        assert not prof.is_metadata(op), op


def test_counter_attributes_by_the_executing_module() -> None:
    net = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))
    x = torch.randn(4, 8, requires_grad=True)
    counter = _counter_for(net, x)
    launches = counter.launches
    # Forward: the op belongs to the Linear/ReLU that is executing, not to the
    # module that produced its input (the off-by-one a graph-only rule makes).
    assert launches[("aten::addmm", "0")] == 1
    assert launches[("aten::relu", "1")] == 1
    assert launches[("aten::addmm", "2")] == 1
    # Backward survives the same rule and reaches the same modules.
    assert launches[("aten::mm", "0")] >= 1
    assert launches[("aten::mm", "2")] >= 1
    assert launches[("aten::threshold_backward", "1")] == 1
    assert all(not prof.is_metadata(op) for op, _module in launches)
    assert sum(counter.dispatches.values()) > sum(launches.values())


def test_json_round_trip_and_diff(tmp_path: Path, capsys) -> None:
    net = torch.nn.Linear(4, 3)
    counter = _counter_for(net, torch.randn(2, 4, requires_grad=True))
    payload = {"meta": {"preset": "tiny", "batch": 2, "seq": 4, "device": "cpu", "dtype": "f32"}}
    a = tmp_path / "a.json"
    prof.write_profile(a, payload["meta"], counter)
    doc = json.loads(a.read_text())
    assert doc["meta"]["total_launches"] == sum(counter.launches.values())
    assert doc["groups"], "the profile wrote no groups"
    assert all({"op", "module", "launches", "dispatches", "share"} <= set(g) for g in doc["groups"])

    # A hand-built B: one group +3, one group gone.
    op, module = doc["groups"][0]["op"], doc["groups"][0]["module"]
    b = tmp_path / "b.json"
    b.write_text(
        json.dumps(
            {
                "meta": {**payload["meta"], "device": "cpu"},
                "groups": [
                    {
                        "op": op,
                        "module": module,
                        "launches": doc["groups"][0]["launches"] + 3,
                        "dispatches": doc["groups"][0]["dispatches"] + 3,
                        "share": 0.0,
                    }
                ],
            }
        )
    )
    assert prof.diff_profiles(a, b, top=5) == 0
    out = capsys.readouterr().out
    assert "total launches" in out
    assert "+3" in out
    assert op in out


def test_cuda_is_refused_when_the_card_is_busy(monkeypatch) -> None:
    monkeypatch.setattr(prof, "_lane_preflight", lambda device: "a compute process holds the card")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="not ours"):
        prof._resolve_device("cuda")


def test_cpu_never_consults_the_card(monkeypatch) -> None:
    def boom(_device):
        raise AssertionError("the CPU path consulted nvidia-smi")

    monkeypatch.setattr(prof, "_lane_preflight", boom)
    assert prof._resolve_device("cpu").type == "cpu"
    assert prof._resolve_device("auto").type == "cpu"


def test_cli_profiles_one_tiny_step(tmp_path: Path, capsys) -> None:
    out = tmp_path / "tiny.json"
    rc = prof.cli(
        [
            "--set",
            "use_csa=true",
            "--set",
            "csa_chunk=4",
            "--set",
            "csa_topk=4",
            "--json",
            str(out),
            "--top",
            "5",
        ]
    )
    assert rc == 0
    doc = json.loads(out.read_text())
    assert doc["meta"]["device"] == "cpu"
    assert doc["meta"]["preset"] == "tiny"
    assert doc["meta"]["total_launches"] > 0
    assert doc["groups"], "no groups on a full training step"
    assert doc["stages"]["forward"] > 0 and doc["stages"]["backward"] > 0
    text = capsys.readouterr().out
    assert "launches/step" in text
    assert "loop.csa" in text or "aten::" in text
