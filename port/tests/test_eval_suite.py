"""`scripts/eval_suite.py` — the markdown and the anchor arithmetic, CPU only.

WHY THESE CHECKS. The card quotes this script's output, so the output has to
fail in a way a reader can see when it is wrong:

1. bits/byte is `compressed * 8 / input` — a fixed byte string pins the divisor
   (quoting the compressed size as an input length is the easy mistake);
2. a compressor that is not on the box renders as **SKIP**, not 0.0000. A zero
   would read as "the model beats it by 2.5 bits/byte" on a card;
3. a skipped BPB row says SKIP and names why; it does not print a number;
4. the finished block carries the parameter count, four decimals, and the window.

No checkpoint, no torch model and no subprocess are needed: `measure` takes the
`which`/`run` seams.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_suite.py"
spec = importlib.util.spec_from_file_location("eval_suite", SCRIPT)
assert spec is not None and spec.loader is not None
eval_suite = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = eval_suite
spec.loader.exec_module(eval_suite)

PAYLOAD = b"0123456789" * 100  # 1 000 bytes, fixed on purpose
FAKE_OUT = 250  # bytes out -> 250 * 8 / 1000 = 2.0 bits/byte


def _fake_run(cmd, **_kw):
    return subprocess.CompletedProcess(cmd, 0, stdout=b"x" * FAKE_OUT, stderr=b"")


def _installed(_tool: str) -> str:
    return "/usr/bin/tool"


def _missing(_tool: str) -> None:
    return None


def test_anchor_arithmetic_on_a_fixed_byte_string(tmp_path):
    src = tmp_path / "window.bin"
    src.write_bytes(PAYLOAD)
    payload, anchors = eval_suite.measure(src, nbytes=len(PAYLOAD), which=_installed, run=_fake_run)
    assert payload == PAYLOAD
    assert len(anchors) == 3, "one row per codec, even when a row is a skip"
    for a in anchors:
        assert a.bits == 2.0, f"{a.label}: 250 B out of 1 000 B is 2.0 bits/byte"
        assert a.compressed == FAKE_OUT

    block = eval_suite.markdown_block(
        params=95_035_200,
        bpb=1.4905827,
        bpb_note="",
        anchors=anchors,
        window="data_local/best/fwedu_eval.bin[0:196608]",
        command="port/scripts/eval_suite.py --ckpt x --eval y --bytes 196608",
        scored=196_608,
    )
    assert "95,035,200" in block
    assert "| 2.0000 |" in block, "bits/byte is printed to four decimals"
    assert "1.4906" in block, "the card's number is rounded once, at print time"
    assert "bpb_fin" in block, "the card quotes the final readout, never bpb_mean"
    assert "196,608 B scored" in block
    assert "Model − best anchor (bzip2 -9): **-0.5094**" in block


def test_a_missing_compressor_is_a_loud_skip_not_a_zero(tmp_path, capsys):
    src = tmp_path / "window.bin"
    src.write_bytes(PAYLOAD)
    _, anchors = eval_suite.measure(src, nbytes=len(PAYLOAD), which=_missing, run=_fake_run)

    err = capsys.readouterr().err
    assert err.count("=== SKIP:") == 3, err
    assert "UNMEASURED, not zero" in err
    for a in anchors:
        assert a.bits is None, "a missing tool has no measurement — not 0.0"
        assert a.compressed is None
        assert "not installed" in a.note

    block = eval_suite.markdown_block(
        params=95_035_200,
        bpb=1.4905827,
        bpb_note="",
        anchors=anchors,
        window="window.bin[0:1000]",
        command="eval_suite.py",
    )
    assert block.count("**SKIP**") == 3
    assert "0.0000" not in block, "a skip rendered as 0 would be a fake win"
    assert "Model − best anchor" not in block, "no anchor ran, so there is no delta"


def test_a_skipped_bpb_row_says_skip_and_why():
    block = eval_suite.markdown_block(
        params=95_035_200,
        bpb=None,
        bpb_note="the card is busy (AGENTS 1.5)",
        anchors=[eval_suite.Anchor("bzip2 -9", 2.5, 312)],
        window="window.bin[0:1000]",
        command="eval_suite.py",
    )
    assert "**SKIP** — the card is busy (AGENTS 1.5)" in block
    assert "| 0.0000 |" not in block
    assert "Model − best anchor" not in block


def test_a_short_window_is_refused(tmp_path):
    src = tmp_path / "window.bin"
    src.write_bytes(b"only 9 B")
    try:
        eval_suite.measure(src, nbytes=100, which=_installed, run=_fake_run)
    except SystemExit as e:
        assert "--bytes 100" in str(e)
    else:  # pragma: no cover - the refusal is the assertion
        raise AssertionError("a window that runs off the file must refuse")
