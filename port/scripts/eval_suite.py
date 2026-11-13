#!/usr/bin/env -S uv run --script
"""The model card's headline numbers, in ONE command, as pasteable markdown.

    cd port
    .venv/bin/python scripts/eval_suite.py \
        --ckpt ../checkpoints/<run>/<name>.pt \
        --eval ../data_local/best/fwedu_eval.bin \
        --bytes 196608

Three measurements, three artefacts, one block:

1. **parameters** — the checkpoint's own snapshot, built on the meta device
   (`param_count`), so this row costs no VRAM and agrees with `dormouse info`.
2. **held-out BPB** — the SAME path `dormouse eval` walks (`_snapshot_run` ->
   `open_streams` -> `evaluate`), with `--eval`, `--bytes` and `--offset` pinned.
   It prints `bpb_fin`, never `bpb_mean` (`docs/reviews/2026-11-06-bpb-accounting.md`).
   If the card is busy, the row is a LOUD SKIP: no number is invented.
3. **compressor anchors** — `bzip2 -9`, `xz -9`, `zstd -19` on exactly the same
   bytes. CPU-only. A missing binary is a LOUD SKIP row, never a zero (a zero
   would read as "the compressor wins").

The output is markdown by design: paste it into the card and every number still
names the command that produced it. The compressors read a FILE, not stdin,
because zstd writes the content size in the frame header for a named file
(measured: 68 084 B vs 68 081 B on the 196 608-byte window — 2.7703 vs 2.7702
bits/byte; the card quotes the file form).

CPU-only for the counts and anchors. The BPB path takes `--device cpu` (slow but
legal), `--device cuda`, or `auto` and refuses a card another process holds
(`scripts/_lane.preflight`, AGENTS 1.5).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# `scripts/` is not a package: the shared lane guard is a sibling. APPENDED,
# never inserted — a script name at the front of `sys.path` would shadow a
# package of that name (same rule as `scripts/smoke_long.py`).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight  # noqa: E402

LN2 = 0.6931471805599453

#: (label, argv prefix). The payload is appended as a FILE path, not stdin.
CODECS = (
    ("bzip2 -9", ("bzip2", "-9", "-c")),
    ("xz -9", ("xz", "-9", "-c")),
    ("zstd -19", ("zstd", "-19", "-q", "-c")),
)


@dataclass(frozen=True)
class Anchor:
    """One compressor on the window: `bits` is None when it did not run."""

    label: str
    bits: float | None
    compressed: int | None
    note: str = ""


def anchor_bits(nbytes: int, compressed: int) -> float:
    """Compressed size → bits per input byte. One definition, two callers."""
    if nbytes <= 0:
        raise ValueError(f"anchor over {nbytes} bytes is not a measurement")
    return compressed * 8 / nbytes


def _loud_skip(what: str, why: str) -> None:
    print(
        f"=== SKIP: {what} — {why}. UNMEASURED, not zero: no number is quoted.",
        file=sys.stderr,
        flush=True,
    )


def measure(
    source: Path,
    *,
    offset: int = 0,
    nbytes: int = 196_608,
    which=shutil.which,
    run=subprocess.run,
) -> tuple[bytes, list[Anchor]]:
    """The window's bytes and one anchor per codec (skips are loud, never zero).

    `which`/`run` are parameters so the CPU test can prove the arithmetic and
    the missing-tool path without depending on what is installed on the box.
    """
    with source.open("rb") as fh:
        fh.seek(offset)
        payload = fh.read(nbytes)
    if len(payload) != nbytes:
        raise SystemExit(
            f"--bytes {nbytes} at --offset {offset} runs off {source} "
            f"({source.stat().st_size} B): a short window is a different measurement"
        )
    anchors: list[Anchor] = []
    with tempfile.TemporaryDirectory(prefix="eval_suite_") as tmp:
        window = Path(tmp) / "window.bin"
        window.write_bytes(payload)
        for label, argv in CODECS:
            tool = argv[0]
            if which(tool) is None:
                _loud_skip(label, f"{tool} is not installed (PATH lookup failed)")
                anchors.append(Anchor(label, None, None, f"{tool} not installed"))
                continue
            done = run([*argv, str(window)], capture_output=True)
            if done.returncode != 0 or not done.stdout:
                why = (done.stderr or b"").decode("utf-8", "replace").strip()[:120]
                _loud_skip(label, f"exited {done.returncode}: {why or 'no output'}")
                anchors.append(Anchor(label, None, None, f"exited {done.returncode}"))
                continue
            size = len(done.stdout)
            anchors.append(Anchor(label, anchor_bits(len(payload), size), size))
    return payload, anchors


def markdown_block(
    *,
    params: int,
    bpb: float | None,
    bpb_note: str,
    anchors: list[Anchor],
    window: str,
    command: str,
    scored: int | None = None,
) -> str:
    """The pasteable block. A skipped row says SKIP; it never prints a zero."""
    bpb_cell = f"{bpb:.4f}" if bpb is not None else f"**SKIP** — {bpb_note or 'not taken'}"
    if scored is not None and bpb is not None:
        bpb_cell += f" ({scored:,} B scored)"
    lines = [
        "| measurement | value | artifact |",
        "|---|---:|---|",
        f"| parameters | {params:,} | checkpoint snapshot, meta-device `param_count` |",
        f"| held-out BPB (`bpb_fin`) | {bpb_cell} | `{command}` · `{window}` |",
        "",
        f"**Compressor anchors** on the same bytes (`{window}`), bits per byte:",
        "",
        "| codec | bits/byte | bytes out |",
        "|---|---:|---:|",
    ]
    for a in anchors:
        bits = f"{a.bits:.4f}" if a.bits is not None else f"**SKIP** — {a.note}"
        out = f"{a.compressed:,}" if a.compressed is not None else "—"
        lines.append(f"| {a.label} | {bits} | {out} |")
    best = [a for a in anchors if a.bits is not None]
    if bpb is not None and best:
        low = min(best, key=lambda a: a.bits if a.bits is not None else float("inf"))
        assert low.bits is not None
        lines += ["", f"Model − best anchor ({low.label}): **{bpb - low.bits:+.4f}** bits/byte."]
    return "\n".join(lines)


def count_params(ckpt: Path) -> int:
    """The snapshot's own model on the meta device — never the preset."""
    from dormouse.cli import _snapshot_run
    from dormouse.model import DormouseModel, param_count

    run = _snapshot_run(ckpt.parent, ckpt.stem)
    return param_count(DormouseModel(run.model, device="meta"))


def score(
    ckpt: Path,
    eval_path: Path,
    *,
    nbytes: int,
    offset: int,
    device: str,
) -> tuple[float | None, int | None, str]:
    """The existing `dormouse eval` path. `(bpb_fin, scored, note)`; None on skip."""
    from dormouse.cli import _device, _model, _snapshot_run
    from dormouse.instrument import LoudError
    from dormouse.train.data import eval_batches_for, open_streams
    from dormouse.train.loop import evaluate, load_model_half, load_model_state

    run = _snapshot_run(
        ckpt.parent,
        ckpt.stem,
        eval_file=str(eval_path),
        eval_bytes=nbytes,
        eval_offset=offset,
    )
    run.train = dataclasses.replace(run.train, eval=True)
    try:
        dev = _device(device)
    except LoudError as e:
        _loud_skip("held-out BPB", str(e))
        return None, None, str(e)
    why = None
    if dev.type != "cpu":
        # preflight prints its verdict to stdout; the block must stay pure
        # markdown, so that line goes to stderr (a reader still sees it).
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            why = preflight(dev.type)
        if buf.getvalue().strip():
            print(buf.getvalue().strip(), file=sys.stderr)
    if why:
        _loud_skip("held-out BPB", why)
        return None, None, why
    model = _model(run).to(dev)
    load_model_state(model, run.train, load_model_half(ckpt))
    # `eval_only=True`: the snapshot's TRAIN file died with the corpus and this
    # command scores a held-out window, so building an unused train stream would
    # demand a file the measurement never reads (the mixture's own call site).
    _train, eval_stream = open_streams(run, eval_only=True)
    if eval_stream is None:
        raise SystemExit("no eval stream: the snapshot carries neither eval_data nor eval=true")
    loss, scored, fin, _per_iter = evaluate(
        model,
        eval_stream,
        eval_batches_for(run.train),
        device=None if dev.type == "cpu" else dev,
    )
    assert loss is not None
    return fin / LN2, scored, ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--ckpt", type=Path, required=True, help="path to NAME.pt (snapshot is a pair)")
    ap.add_argument("--eval", type=Path, required=True, help="the held-out eval file")
    ap.add_argument("--bytes", type=int, default=196_608, help="window width (default 196608)")
    ap.add_argument("--offset", type=int, default=0, help="pinned start byte (default 0)")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = ap.parse_args(argv)

    ckpt = args.ckpt.with_suffix(".pt")
    for path, what in ((ckpt, "checkpoint"), (ckpt.with_suffix(".config.toml"), "snapshot")):
        if not path.is_file():
            print(f"eval_suite: no {what} at {path}", file=sys.stderr)
            return 2
    if not args.eval.is_file():
        print(f"eval_suite: no eval file at {args.eval}", file=sys.stderr)
        return 2

    params = count_params(ckpt)
    _, anchors = measure(args.eval, offset=args.offset, nbytes=args.bytes)
    bpb, scored, note = score(
        ckpt, args.eval, nbytes=args.bytes, offset=args.offset, device=args.device
    )
    command = (
        f"port/scripts/eval_suite.py --ckpt {args.ckpt} --eval {args.eval} "
        f"--bytes {args.bytes} --offset {args.offset} --device {args.device}"
    )
    print(
        markdown_block(
            params=params,
            bpb=bpb,
            bpb_note=note,
            anchors=anchors,
            window=f"{args.eval}[{args.offset}:{args.offset + args.bytes}]",
            command=command,
            scored=scored,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
