"""TASK B's verdict statistic: the A/B's divergence mark and the trimmed mean.

`scripts/ab_sf_muon.py` must keep the 2026-10-18 divergence (BPB 4.0573)
VISIBLE, not average it into the arm. A sample σ of the three points is 1.06
BPB, so a literal `3σ` would pass the very seed it should catch; the driver
uses `1.4826·MAD` (2.8e-2 here) with a 2e-3 BPB floor, and this gate runs the
real arm's numbers through it.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pytest

_SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.append(_SCRIPTS)
from ab_sf_muon import diverged, kept, verdict  # noqa: E402


def test_the_verdict_flags_the_4_06_seed_instead_of_diluting_it():
    """The 2026-10-18 numbers, through the driver's own statistic.

    The two-seed mean the earlier report quoted as "+0.045" is reproduced here
    only because the 4.0573 seed is marked and excluded — never because it was
    silently dropped.
    """
    bpbs = [2.2257, 4.0573, 2.2065]
    assert diverged(bpbs) == [False, True, False]
    assert kept(bpbs) == [2.2257, 2.2065]
    assert statistics.mean(kept(bpbs)) == pytest.approx(0.0448 + 2.1713, abs=1e-3)

    control = [2.1740, 2.1701, 2.1697]
    assert kept(control) == control, "a normal seed must not be marked divergent"
    v = verdict(control, kept(bpbs))
    assert v["delta"] == pytest.approx(-0.0448, abs=1e-3)
    assert v["spread"] == pytest.approx(0.0043, abs=1e-4)
    assert v["kept"] is False, "a slower arm must not be kept"
    assert verdict(control, [2.1600, 2.1600, 2.1600])["kept"] is True
    assert verdict(control, [2.1600])["kept"] is None, "one kept seed cannot decide"
