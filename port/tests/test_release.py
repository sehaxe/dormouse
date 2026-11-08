"""Release gate: the version, `--version`, and the preset lookup in both layouts.

Builds nothing. The one packaging assertion is the bridge between the loader
and `pyproject.toml`, so deleting the wheel's preset mapping fails a test here
before it fails an install; `docs/guides/release.md` runs the real wheel once.
"""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import dormouse
from dormouse import config
from dormouse.instrument import LoudError

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "configs"
WHEEL_CONFIGS = "dormouse/configs"


def test_version_resolves_from_the_package_metadata():
    assert re.fullmatch(r"\d+(?:\.\d+)+", dormouse.__version__), dormouse.__version__
    assert dormouse.__version__ == importlib.metadata.version("dormouse")


def test_dash_version_exits_zero_and_prints_the_same_version(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "dormouse.cli", "--version"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == f"dormouse {dormouse.__version__}"


def test_repo_layout_loads_from_the_checkout(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert config.PRESET_DIR == PRESETS
    assert config._candidates("small")[0].is_file()
    assert config.load_preset("small").d_model == 768


def test_installed_layout_falls_back_to_the_packaged_presets(tmp_path, monkeypatch):
    site = tmp_path / "site-packages" / "dormouse" / "configs"
    site.mkdir(parents=True)
    (site / "small.toml").write_text((PRESETS / "small.toml").read_text())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "PRESET_DIR", tmp_path / "checkout" / "configs")
    monkeypatch.setattr(config, "PACKAGE_PRESET_DIR", site)
    assert config.load_preset("small").d_model == 768


def test_no_layout_at_all_refuses_loudly_and_names_every_candidate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "PRESET_DIR", tmp_path / "checkout" / "configs")
    monkeypatch.setattr(config, "PACKAGE_PRESET_DIR", tmp_path / "site-packages" / "configs")
    with pytest.raises(LoudError, match="unknown preset") as exc:
        config.load_preset("small")
    for path in config._candidates("small"):
        assert str(path) in str(exc.value)


def test_the_wheel_maps_the_checkout_presets_onto_the_package_dir():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    force = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    sources = [Path(ROOT, src) for src, dest in force.items() if dest == WHEEL_CONFIGS]
    assert len(sources) == 1, force
    assert (sources[0] / "small.toml").is_file()
    assert config.PACKAGE_PRESET_DIR == Path(config.__file__).resolve().parent / "configs"
