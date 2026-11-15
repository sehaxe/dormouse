"""The 1B launch: the preset pair, the token stream, and the BPB conversion.

`dorm train configs/dormouse_1b.toml` is the launch command, so three things
must hold and nothing else in the repo says so:

* the model table is `configs/b1.toml` with `vocab` swapped (a diff elsewhere is
  a different model wearing a launch name);
* the smoke preset is the SAME model (a rehearsal of a different recipe proves
  nothing about the run);
* a 2-byte token stream is refused on a byte vocabulary, and `bpb` divides the
  CE by the stream's bytes-per-token so its BPB stays a per-BYTE number.
"""

from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path

import pytest

from dormouse.config import PRESET_TRAIN_DEFAULTS, load_run
from dormouse.instrument import LoudError
from dormouse.train.loop import bpb

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PRE = Path(__file__).resolve().parents[2] / "data_local" / "pre"
MIX = PRE / "mix.bin"
MIX_SMOKE = PRE / "mix_smoke.bin"


def _table(name: str) -> dict:
    """The preset's model table. A private reader on purpose: `load_preset`
    VALIDATES, and this test is about the file, not about whether it loads."""
    return tomllib.loads((CONFIGS / f"{name}.toml").read_text())


def test_the_launch_model_is_b1_with_a_32k_vocabulary():
    launch, b1 = _table("dormouse_1b"), _table("b1")
    assert launch.pop("vocab") == 32768
    assert b1.pop("vocab") == 256
    assert launch == b1, "dormouse_1b must be b1 apart from `vocab`"


def test_the_smoke_model_is_the_launch_model():
    smoke, launch = _table("dormouse_1b_smoke"), _table("dormouse_1b")
    assert smoke == launch


def test_both_launch_presets_load_and_carry_the_token_stream():
    for name in ("dormouse_1b", "dormouse_1b_smoke"):
        assert PRESET_TRAIN_DEFAULTS[name]["data_item_bytes"] == 2
        run = load_run(str(CONFIGS / f"{name}.toml"))
        assert run.model.vocab == 32768
        assert run.train.mix_plan.endswith(".toml")


@pytest.mark.skipif(not MIX.is_file(), reason="the mixture is not assembled in this checkout")
@pytest.mark.parametrize("name", ["dormouse_1b", "dormouse_1b_smoke"])
def test_the_bpb_conversion_is_the_manifest_s_own(name):
    """A BPB that is not divided by the stream's bytes-per-token is bits per
    TOKEN wearing the name of bits per byte, so the two numbers are pinned."""
    defaults = PRESET_TRAIN_DEFAULTS[name]
    manifest = MIX_SMOKE if name.endswith("smoke") else MIX
    assert (
        defaults["data_bytes_per_token"]
        == (json.loads(manifest.with_suffix(".bin.json").read_text())["bytes_per_token"])
    )


def test_a_token_stream_on_a_byte_vocabulary_is_refused(tmp_path):
    with pytest.raises(LoudError, match="TOKEN stream"):
        load_run(str(CONFIGS / "b1.toml"), train_overrides={"data_item_bytes": 2})


def test_doc_masking_and_tst_are_refused_on_a_token_stream():
    for over in ({"doc_masking": True}, {"tst_bag": 2}):
        with pytest.raises(LoudError, match="data_item_bytes=2"):
            load_run("small", train_overrides={"data_item_bytes": 2, **over})


def test_bpb_is_per_byte_of_text():
    """One nat of CE over one byte is 1/ln2 bits per byte; over 4 bytes/token
    the same nat is a quarter of that."""
    assert bpb(math.log(2)) == pytest.approx(1.0)
    assert bpb(math.log(2), 4.0) == pytest.approx(0.25)
    assert bpb(1.0) == bpb(1.0, 1.0)
