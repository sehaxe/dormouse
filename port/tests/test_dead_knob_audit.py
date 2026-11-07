"""Gates for `scripts/dead_knob_audit.py` — the instrument, not the fields it found.

The defect class has cost this project weeks (`muon_lr_mult`, `fused_residual`,
`use_msa`, `doc_masking`): a config knob that exists, validates, and changes no
run. These gates pin the CLASSIFIER on a temp module, pin the preset refusal,
and pin the runtime pass against the static one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PORT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "dead_knob_audit", PORT / "scripts" / "dead_knob_audit.py"
)
A = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = A
_spec.loader.exec_module(A)  # type: ignore[union-attr]

CONFIG = """\
from dataclasses import dataclass


@dataclass
class ToyConfig:
    live: int = 1
    dead: int = 2
    shadowed: int = 3
    logged: int = 4
    via_getattr: int = 5
    via_dict: int = 6
"""

CONSUMER = """\
from config import ToyConfig


class Runner:
    def __init__(self, cfg: ToyConfig):
        self.limit = cfg.shadowed

    def step(self, cfg: ToyConfig):
        self.limit = 99
        return cfg.live + getattr(cfg, "via_getattr", 0)

    def report(self, cfg: ToyConfig):
        print(f"logged={cfg.logged}")
        return cfg.get("via_dict")
"""


def _line(text: str, needle: str) -> int:
    for i, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return i
    raise AssertionError(f"{needle!r} not in fixture")


def _project(tmp_path: Path) -> tuple[Path, Path]:
    cfg = tmp_path / "config.py"
    cfg.write_text(CONFIG)
    (tmp_path / "consumer.py").write_text(CONSUMER)
    return cfg, tmp_path


def test_classifier_reads_a_temp_module(tmp_path: Path):
    """Gate (a)+(d): 1 read / 1 dead / 1 shadowed, with the right lines."""
    cfg, root = _project(tmp_path)
    reports, findings, _ = A.static_audit([cfg], [root], "nope/*.toml", {"ToyConfig"})

    assert reports["live"].verdict == "LIVE"
    assert [r.line for r in reports["live"].reads] == [_line(CONSUMER, "cfg.live")]

    assert reports["dead"].verdict == "DEAD"
    assert reports["dead"].reads == []
    assert reports["dead"].spec.line == _line(CONFIG, "dead: int")

    assert reports["shadowed"].verdict == "SHADOWED"
    writes = {(w.line, w.slot) for w in reports["shadowed"].shadow_writes}
    assert writes == {
        (_line(CONSUMER, "self.limit = cfg.shadowed"), "self.limit"),
        (_line(CONSUMER, "self.limit = 99"), "self.limit"),
    }
    assert "SHADOWED: shadowed" in findings


def test_getattr_and_dict_style_reads_are_reads(tmp_path: Path):
    """Gate (d): the classifier is not a grep for `cfg.<name>`."""
    cfg, root = _project(tmp_path)
    reports, _, _ = A.static_audit([cfg], [root], "nope/*.toml", {"ToyConfig"})

    assert reports["via_getattr"].verdict == "LIVE"
    assert [r.line for r in reports["via_getattr"].reads] == [
        _line(CONSUMER, 'getattr(cfg, "via_getattr"')
    ]
    assert reports["via_dict"].verdict == "LIVE"
    assert [r.line for r in reports["via_dict"].reads] == [_line(CONSUMER, 'cfg.get("via_dict")')]
    # A read that only builds a log line is not a live knob.
    assert reports["logged"].verdict == "LOG-ONLY"
    assert [r.line for r in reports["logged"].reads] == [_line(CONSUMER, "logged={cfg.logged}")]


def test_injected_dead_field_in_mirror_control_is_loud(tmp_path: Path, capsys):
    """Gate (b): the gate that would have caught `muon_lr_mult=8`."""
    cfg, root = _project(tmp_path)
    presets = tmp_path / "presets"
    presets.mkdir()
    real = (PORT / "configs" / "mirror_control.toml").read_text()
    (presets / "mirror_control.toml").write_text(real + "\ndead = 42\n")

    rc = A.main(
        [
            "--config",
            str(cfg),
            "--class",
            "ToyConfig",
            "--root",
            str(root),
            "--preset-glob",
            str(presets / "*.toml"),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 2, out
    assert "PRESET-USES-DEAD: dead" in out


def test_real_presets_do_not_set_a_dead_field_off_its_default(capsys):
    """The repo's own presets are clean of the muon_lr_mult shape right now."""
    rc = A.main([])
    out = capsys.readouterr().out
    assert rc != 2, out
    assert "PRESET-USES-DEAD" not in out


def test_runtime_and_static_agree_on_d_model(tmp_path: Path):
    """Gate (c): a field that is definitely used is read by both passes."""
    configs = [
        PORT / "src" / "dormouse" / "config.py",
        PORT / "src" / "dormouse" / "post" / "rl.py",
    ]
    reports, _, _ = A.static_audit(
        configs, [PORT / "src" / "dormouse"], "configs/**/*.toml", set(A.DEFAULT_CLASSES)
    )
    runtime = A.runtime_audit("mirror_control", configs, workdir=tmp_path)
    assert reports["d_model"].verdict in ("LIVE", "SHADOWED")
    assert "d_model" in runtime
