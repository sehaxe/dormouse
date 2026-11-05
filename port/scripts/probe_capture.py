#!/usr/bin/env -S uv run --python /home/sehaxe/dormouse/port/.venv/bin/python --script
"""Locate the op that aborts a CUDA-graph capture, with the frame that issued it.

Two targets:

* `--target moe` (the original) captures `SparseMoE.forward` alone — the smallest
  region that isolates a dispatch blocker.
* `--target step` runs the REAL trainer with `full_graph=true` under a
  `TorchDispatchMode` that records only ops dispatched while the stream is
  capturing, then prints the last ones. `FullGraph` catches the exception and
  counts the fallback (a run must never die of the arm), so the probe reads the
  counter/log for the verdict and the ring buffer for the culprit.

This is the instrument the 2026-10-24 `order[keep]` diagnosis used: the abort
surfaces as `cudaErrorStreamCaptureInvalidated` at the NEXT op, so the last
twelve dispatched ops + their `file:line` are the evidence, not the traceback.

    # the recipe's whole step (card free; ONE GPU process, AGENTS 1.5)
    uv run --extra gpu python scripts/probe_capture.py --target step \
        --snapshot /home/sehaxe/dormouse/checkpoints/m400b/m400b.config.toml \
        --batch 8 --seq-len 1024 --steps 8

    # the per-op table (which expressions a capture can hold)
    uv run --extra gpu python scripts/probe_capture.py --ops

    # the original single-MoE probe, tiny preset
    uv run --extra gpu python scripts/probe_capture.py --target moe
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import sys
import traceback
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

#: The window the 2026-10-24 diagnosis used. The abort is one op AFTER the
#: culprit, so a ring of 12 holds the culprit and the abort.
LAST: collections.deque = collections.deque(maxlen=12)

#: Aten ops/expressions whose CUDA form is a host read. The probe NAMES the
#: first one it saw instead of printing a traceback that points at the op after
#: it. Kept deliberately short: it is a pointer for a human, and the full
#: per-op table is `probe_ops.py` (`--ops`).
NON_CAPTURABLE_MARKERS: tuple[str, ...] = (
    "nonzero",
    "tolist",
    "_local_scalar_dense",
    "unique",
    "repeat_interleave",
)


def name_the_culprit(lines) -> str | None:
    """The first recorded op whose aten name is a known host read, or `None`."""
    for line in lines:
        if any(marker in line for marker in NON_CAPTURABLE_MARKERS):
            return line
    return None


class Log(TorchDispatchMode):
    """Record dispatch calls, optionally only while a capture is active."""

    def __init__(self, *, captured_only: bool = False) -> None:
        super().__init__()
        self.captured_only = captured_only
        #: Once a capture has been seen and then ended, STOP recording: the
        #: trainer catches the abort, counts the fallback and runs the body
        #: EAGERLY, and those ops would otherwise overwrite the last 12 with
        #: the fallback's — i.e. hide the culprit behind the symptom.
        self._seen_capture = False
        self._frozen = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self._frozen:
            return func(*args, **kwargs)
        capturing = True
        if self.captured_only:
            try:
                capturing = bool(torch.cuda.is_current_stream_capturing())
            except Exception:  # noqa: BLE001 - no CUDA / no capture: record it
                capturing = True
            if capturing:
                self._seen_capture = True
            elif self._seen_capture:
                self._frozen = True
                return func(*args, **kwargs)
        if capturing:
            frames = traceback.extract_stack()[:-1]
            site = ""
            for f in reversed(frames):
                if "site-packages" not in f.filename and "torch/" not in f.filename:
                    site = f"{f.filename}:{f.lineno}"
                    break
            LAST.append(f"{func}  @ {site}")
        else:
            # Keep the ring warm even if this torch build's capture query lies:
            # the last 12 ops before the abort ARE the capture's ops (the last
            # eager step ended before it), only the `file:line` is missing.
            LAST.append(f"{func}")
        return func(*args, **kwargs)


def _report(name: str, ok: bool, reason: str | None = None) -> int:
    if ok:
        print(f"{name}: CAPTURE OK")
        return 0
    print(f"\n{name}: CAPTURE FAILED (COUNTED fallback)")
    if reason:
        print(f"reason: {reason}")
    print("last dispatched ops (oldest first):")
    for line in LAST:
        print("  ", line)
    culprit = name_the_culprit(LAST)
    if culprit is not None:
        print(f"culprit: {culprit}")
    else:
        print("culprit: not in the known host-read table — the 12 ops above bracket it")
    return 1


# ---------------------------------------------------------------------------
# target: one MoE forward (the original probe)
# ---------------------------------------------------------------------------


def probe_moe(args: argparse.Namespace) -> int:
    from dormouse.config import load_run
    from dormouse.model.experts import SparseMoE

    run = load_run(
        args.config or "tiny",
        model_overrides=dict(bf16=True),
        train_overrides=dict(batch=args.batch, seq_len=args.seq_len),
    )
    cfg = run.model
    print(
        f"cfg: E={cfg.n_experts} k={cfg.moe_topk} cap={cfg.moe_capacity} "
        f"kernel={cfg.moe_kernel} gather={cfg.moe_gather} d={cfg.d_model} f={cfg.d_ffn}"
    )
    torch.manual_seed(0)
    moe = SparseMoE(cfg, device="cuda")
    n = args.rows
    x = torch.randn(n, cfg.d_model, device="cuda", dtype=torch.bfloat16)

    for _ in range(2):
        moe(x)
    torch.cuda.synchronize()
    print("eager forward OK")

    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    LAST.clear()
    try:
        with torch.cuda.graph(g), Log():
            moe(x)
    except Exception as e:  # noqa: BLE001 - the probe's whole job
        torch.cuda.synchronize()
        return _report("moe", False, f"{type(e).__name__}: {e}")
    torch.cuda.synchronize()
    return _report("moe", True)


# ---------------------------------------------------------------------------
# target: the real trainer's whole step
# ---------------------------------------------------------------------------


def _step_run(args: argparse.Namespace):
    from dormouse.config import RunConfig, load_run

    snapshot = Path(args.snapshot) if args.snapshot else None
    if snapshot is not None and snapshot.is_file():
        run = RunConfig.from_snapshot(snapshot.read_text(encoding="utf-8"))
    else:
        run = load_run(
            args.config or "m400b",
            train_overrides=dict(
                batch=args.batch,
                seq_len=args.seq_len,
                lr=5e-4,
                wd=0.01,
                grad_clip=1.0,
                muon_state_dtype="fp32",
                ns_dtype="fp32",
                muon_chunk_elems=1 << 26,
            ),
        )
    return dataclasses.replace(
        run,
        train=dataclasses.replace(
            run.train,
            steps=args.steps,
            batch=args.batch,
            seq_len=args.seq_len,
            eval=False,
            eval_every=10**9,
            ckpt_every=10**9,
            log_every=1,
            json_log="",
            full_graph=True,
            opt_graph=False,
            torch_compile=False,
        ),
    )


def probe_step(args: argparse.Namespace) -> int:
    from dormouse.model import DormouseModel
    from dormouse.train import train
    from dormouse.train.data import ByteStream

    run = _step_run(args)
    if not torch.cuda.is_available():
        print("REFUSED: no CUDA device; the step probe needs a card")
        return 2
    torch.manual_seed(0)
    model = DormouseModel(run.model, device="cuda")
    data = Path(args.data)
    if not data.is_file():
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_bytes(bytes(range(256)) * (4 << 20))
    logs: list[str] = []
    LAST.clear()
    try:
        with Log(captured_only=True):
            report = train(
                model,
                run,
                stream=ByteStream(data, run.train.batch, run.train.seq_len),
                device="cuda",
                log=logs.append,
            )
    except Exception as e:  # noqa: BLE001 - a config refusal is also an answer
        print(f"REFUSED: {type(e).__name__}: {e}")
        return 2
    counters = report["counters"]
    if counters.get("full_graph_captured"):
        return _report("step", True)
    reason = next((line for line in logs if "capture failed" in line), None)
    return _report("step", False, reason or str(getattr(counters, "why", None)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", choices=("moe", "step"), default="moe")
    ap.add_argument("--ops", action="store_true", help="print the per-op capture table")
    ap.add_argument(
        "--config",
        default=None,
        help="preset for the probe (default: tiny for --target moe, m400b for step)",
    )
    ap.add_argument(
        "--snapshot",
        default="/home/sehaxe/dormouse/checkpoints/m400b/m400b.config.toml",
        help="the live run's config snapshot; used by --target step when it exists",
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--rows", type=int, default=256, help="MoE probe rows (n = b*s)")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--data", default="/tmp/opencode/probe_capture_bytes.bin")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ops:
        sys.path.append(str(Path(__file__).resolve().parent))
        import probe_ops

        probe_ops.main()
        return 0
    if not torch.cuda.is_available():
        print("REFUSED: no CUDA device (set CUDA_VISIBLE_DEVICES to the card)")
        return 2
    if args.target == "step":
        return probe_step(args)
    return probe_moe(args)


if __name__ == "__main__":
    raise SystemExit(main())
