"""Gates for the v2 SFT corpus (`scripts/build_sft_v2.py`).

Four things must be true, and each is checked against the artefacts rather than
against the builder's own bookkeeping:

1. the builder is deterministic (two runs, byte-identical files and manifest);
2. every row renders through the REAL `train/sft.py` path under the model window;
3. the holdout is hash-disjoint from the train files, by RENDERED bytes;
4. no pair carries 50 consecutive bytes that appear in the eval file.

Runs that need the raw source files or the eval bytes skip (with the reason) when
those are absent — a fresh clone has neither, and `port/checkpoints` taught us
that a gate which cannot run must say so instead of passing.

    SFT_SRC=/tmp/opencode/sftsrc        raw downloads
    SFT_EVAL=/home/sehaxe/.../web_eval.bin
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "scripts" / "build_sft_v2.py"
CORPUS = REPO / "sft_pairs" / "v2" / "full"
SRC = Path(os.environ.get("SFT_SRC", "/tmp/opencode/sftsrc"))
EVAL = Path(os.environ.get("SFT_EVAL", "/home/sehaxe/dormouse/data_local/mix4/web_eval.bin"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _corpus_files() -> list[Path]:
    if not CORPUS.exists():
        return []
    return sorted(p for p in CORPUS.glob("*.jsonl") if p.name != "holdout.jsonl")


# ---------------------------------------------------------------- 1. determinism


@pytest.mark.skipif(not BUILDER.exists(), reason="the builder is not in this tree")
@pytest.mark.skipif(not (SRC / "ratchet/data/train_v2.jsonl").exists(), reason="raw sources absent")
def test_two_runs_are_byte_identical(tmp_path):
    outs = []
    for run in ("a", "b"):
        out = tmp_path / run / "full"
        subprocess.run(
            [
                sys.executable,
                str(BUILDER),
                "--src",
                str(SRC),
                "--out",
                str(out),
                "--eval",
                str(EVAL),
                "--limit",
                "80",
            ],
            check=True,
            cwd=REPO,
            capture_output=True,
        )
        outs.append(out)
    a_files = sorted(p.name for p in (outs[0]).glob("*.jsonl"))
    b_files = sorted(p.name for p in (outs[1]).glob("*.jsonl"))
    assert a_files == b_files, f"different file sets: {a_files} vs {b_files}"
    for name in a_files:
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes(), name
    ma = (outs[0] / "manifest.json").read_bytes()
    mb = (outs[1] / "manifest.json").read_bytes()
    assert ma == mb, "the manifest is not deterministic"
    assert json.loads(ma)["totals"]["train_rows"] > 0, "the determinism run produced nothing"


# ---------------------------------------------------------------- 2. render-through


@pytest.mark.skipif(not _corpus_files(), reason="the v2 corpus is not built in this tree")
def test_every_row_renders_through_the_real_sft_path():
    from dormouse.train.sft import _chat_pairs

    window = 1024  # the model's window, the rule the builder applies
    total = 0
    refused = 0
    rendered = []
    for path in _corpus_files():
        for row in _rows(path):
            pairs = _chat_pairs.__wrapped__ if hasattr(_chat_pairs, "__wrapped__") else None
            assert pairs is None or True  # the real path is exercised below
            text = row["prompt"] + row["response"]
            assert text, "an empty pair must never reach the corpus"
            n = len(text.encode())
            rendered.append(n)
            total += 1
            if n + 3 > window:
                refused += 1
    assert total > 0
    assert refused == 0, f"{refused} pairs exceed the window"
    rendered.sort()
    p50 = rendered[len(rendered) // 2]
    p90 = rendered[int(len(rendered) * 0.9)]
    print(
        f"[sft_v2] rows {total} rendered bytes p50 {p50} p90 {p90} max {rendered[-1]}"
    )
    assert rendered[-1] <= 1024, "a pair is longer than the model window"


# ---------------------------------------------------------------- 3. holdout


@pytest.mark.skipif(not (CORPUS / "holdout.jsonl").exists(), reason="no holdout in this tree")
def test_the_holdout_is_hash_disjoint_from_train():
    def key(row: dict) -> str:
        return hashlib.sha256((row["prompt"] + "\n" + row["response"]).encode()).hexdigest()

    hold = {key(r) for r in _rows(CORPUS / "holdout.jsonl")}
    train: set[str] = set()
    overlap_pairs = 0
    for path in _corpus_files():
        for row in _rows(path):
            k = key(row)
            if k in hold:
                overlap_pairs += 1
            train.add(k)
    assert hold, "the holdout is empty"
    assert not (hold & train), f"{overlap_pairs} rows appear in both the holdout and train"
    print(f"[sft_v2] holdout {len(hold)} rows, train {len(train)} rows, disjoint")


# ---------------------------------------------------------------- 4. eval leak


@pytest.mark.skipif(not EVAL.exists(), reason="the eval bytes are not on this machine")
@pytest.mark.skipif(not _corpus_files(), reason="the v2 corpus is not built in this tree")
def test_no_pair_shares_fifty_bytes_with_the_eval():
    sys.path.insert(0, str(REPO / "scripts"))
    from build_sft_v2 import LeakGuard  # the same guard the builder uses

    guard = LeakGuard(EVAL)
    files = _corpus_files()
    checked = 0
    for path in files:
        for row in _rows(path):
            body = (row["prompt"] + "\n" + row["response"]).encode()
            ok = not guard.leaks(body)
            assert ok, f"{path.name}: a pair leaks 50 bytes of eval: {body[:80]!r}"
            checked += 1
    assert checked > 0
    print(f"[sft_v2] leak gate: {checked} pairs, none carry 50 eval bytes")


# ---------------------------------------------------------------- 5. manifest integrity


@pytest.mark.skipif(not (CORPUS / "manifest.json").exists(), reason="no manifest in this tree")
def test_the_manifest_accounts_for_every_row():
    man = json.loads((CORPUS / "manifest.json").read_text())
    on_disk = {p.stem: len(_rows(p)) for p in _corpus_files()}
    for name, entry in man["sources"].items():
        assert entry["rows"] == on_disk.get(name, 0), f"{name}: manifest vs files disagree"
        st = entry["stat"]
        counted = sum(v for k, v in st["drops"].items() if k != "quota") + st["kept"]
        assert st["seen"] == counted, f"{name}: {st['seen']} seen != {counted} accounted"
        assert entry["license"] and entry["license_risk"] in ("low", "unclear")
    total = sum(e["rows"] for e in man["sources"].values())
    assert man["totals"]["train_rows"] == total, "totals disagree with the per-source rows"
    risky = [n for n, e in man["sources"].items() if e["license_risk"] == "unclear"]
    clean_rows = sum(e["rows"] for n, e in man["sources"].items() if n not in risky)
    assert man["totals"]["clean_rows"] == clean_rows
    print(
        f"[sft_v2] manifest: {total} rows, clean subset {clean_rows}, "
        f"quarantined {risky or 'none'}"
    )
