#!/usr/bin/env python3
"""Activation VRAM at ONE shape: the peak map, the lever ladder, the batch ceiling.

    GPU_TIMEOUT=1800 bash scripts/gpu_lock.sh \
        .venv/bin/python scripts/bench_activation_vram.py --preset configs/flagship3_dense.toml

WHAT IT ANSWERS (three questions, one process, one card acquisition):

1. **The map** — where the step's peak actually sits. `memory_stats()` for
   allocated / reserved / peak, then a `_record_memory_history` trace of ONE
   step whose events are replayed to the moment allocated bytes are maximal;
   the top live blocks at that moment are printed with their allocating stack.
   `saved_tensors_hooks` gives the same step's SAVED set with tensor shapes
   (the trace carries stacks, not shapes), which is the table a reader needs to
   recognise the objects.
2. **The ladder** — `peak_gib`, `ms_step`, `kB/s` per arm, the same model and
   optimizer the trainer builds. An arm the config REFUSES is recorded as
   refused with the message; it is not silently skipped (rule 1.1: a lever that
   names a form it does not reach is a config that lies).

   The levers, by the field that owns them (`config.py`):

   * `segment_len=N` — segmented training: the window is cut into N-byte
     segments whose KDA state is carried, each segment inside
     `checkpoint(use_reentrant=False)`. **KDA loop form only**; refused with
     `block_form="transformer"`.
   * `activation_recompute="full"|"selective"` — whole-iteration checkpoint, or
     keep the read half and recompute the write half. **KDA loop form only.**
   * `iter_checkpoint=true` — the historical boolean alias for `"full"`.
   * `activation_offload=GIB` — pinned host offload of the largest saved
     activations, exact by construction. Works on **either** form.
   * `muon_state_dtype="bf16"` — half the Muon momentum, an optimizer-state
     lever (measured 0.99x on speed, so it is memory-only).

   `segment_len` is refused on the dense transformer form (`config.py`
   `_validate_segment`), and `activation_recompute`/`iter_checkpoint` refuse a
   model with no loop block (`model.set_recompute`). On `flagship3_dense` only
   `activation_offload` and `muon_state_dtype` exist, and the run must SAY so
   rather than report a flag that reached no code.
3. **The batch ceiling** — the largest batch at `seq_len` whose peak stays under
   `--ceiling-gib` (default 15.0). Ascending batches; the first OOM ends the
   scan because a larger batch cannot fit either.

MEASUREMENT RULES. `steps` timed steps after `warm` untimed ones (the cubecl
era's step-0 artifact has a PyTorch cousin: the first step pays allocator and
cudnn/cuBLAS setup). `peak_gib` is `torch.cuda.max_memory_allocated`, reset
after the warmup — the ALLOCATOR's peak, not `nvidia-smi` residency (reserved
and device-free are printed beside it so the gap is visible). `ms_step` is the
median. `kB/s = batch*seq/1e3 / seconds`, the repo's convention (the ladder's
56.2 kB/s is 8192 B at 145.8 ms).

The shape is synthetic bytes (`randint(0, 256)`) with `targets = ids`: this file
prices memory and rate, never quality, and the data loader is not part of
either.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import gc
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dormouse.cli import _model  # noqa: E402
from dormouse.config import load_run  # noqa: E402
from dormouse.instrument import Counters, LoudError  # noqa: E402
from dormouse.train.optim import MixOptimizer  # noqa: E402

GIB = 2**30

#: name -> TrainConfig overrides. The ladder is DATA, so adding a rung is a row.
ARMS: dict[str, dict[str, object]] = {
    "base": {},
    "segment512": {"segment_len": 512},
    "segment256": {"segment_len": 256},
    "ckpt_full": {"activation_recompute": "full"},
    "ckpt_selective": {"activation_recompute": "selective"},
    "iter_ckpt": {"iter_checkpoint": True},
    "offload4": {"activation_offload": 4.0},
    "offload8": {"activation_offload": 8.0},
    "muonstate_bf16": {"muon_state_dtype": "bf16"},
    "ckpt_full+bf16state": {"activation_recompute": "full", "muon_state_dtype": "bf16"},
    "ckpt_full+offload4": {"activation_recompute": "full", "activation_offload": 4.0},
}


# --------------------------------------------------------------------- plumbing


def build(preset: str, batch: int, seq: int, overrides: dict[str, object], device: str):
    """The trainer's own model + optimizer for one (shape, arm) cell."""
    import torch

    train = {
        "batch": batch,
        "seq_len": seq,
        "grad_accum": 1,
        "eval": False,
        "steps": 1,
        "ckpt_every": 10**9,
        "log_every": 10**9,
        **overrides,
    }
    run = load_run(preset, train_overrides=train)
    model = _model(run).to(device).train()
    c = run.train
    model.configure_actmem(
        iter_checkpoint=c.iter_checkpoint,
        activation_offload=c.activation_offload,
        activation_recompute=c.activation_recompute,
    )
    opt = MixOptimizer(
        model,
        lr=c.lr,
        wd=c.wd,
        opt=c.opt,
        ns_steps=c.ns_steps,
        ns_kind=c.ns_kind,
        colrow_3d=c.muon_colrow_3d,
        muon_state_dtype=c.muon_state_dtype,
        ns_dtype=c.ns_dtype,
        eps=c.adam_eps,
        muon_variant=c.muon_variant,
        muon_sf=c.muon_sf,
        muon_sf_wd=c.muon_sf_wd,
        muon_chunk_elems=c.muon_chunk_elems,
        muon_lr_mult=c.muon_lr_mult,
        muon_lr_mode=c.muon_lr_mode,
    )
    return run, model, opt


def step(model, opt, ids, *, backward: bool = True, segment_len: int = 0):
    """One trainer step. `segment_len` is a FORWARD ARGUMENT in this model
    (`DormouseModel.forward`), not a module attribute — the trainer passes
    `train.segment_len` per call, so a harness that sets the TrainConfig field
    and never passes it measures the one-shot path while reporting the arm on.
    """
    _, loss, counters = model(ids, targets=ids, counters=Counters(), segment_len=segment_len)
    if backward:
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return loss, counters


def _stats() -> dict:
    import torch

    s = torch.cuda.memory_stats()
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_gib": s["allocated_bytes.all.current"] / GIB,
        "allocated_peak_gib": s["allocated_bytes.all.peak"] / GIB,
        "reserved_gib": s["reserved_bytes.all.current"] / GIB,
        "reserved_peak_gib": s["reserved_bytes.all.peak"] / GIB,
        "active_peak_gib": s["active_bytes.all.peak"] / GIB,
        "inactive_split_gib": s["inactive_split_bytes.all.current"] / GIB,
        "device_free_gib": free / GIB,
        "device_total_gib": total / GIB,
        "num_alloc_retries": s["num_alloc_retries"],
        "num_ooms": s["num_ooms"],
    }


def measure(
    preset: str, batch: int, seq: int, arm: str, *, warm: int, steps: int, device: str
) -> dict:
    """One cell: peak + median step time. REFUSED arms carry the loud message."""
    import torch

    row: dict[str, object] = {"arm": arm, "batch": batch, "seq": seq}
    # Sampled BEFORE the model is built: what a previous refused/failed arm has
    # pinned through its traceback. Zero on a clean process.
    torch.cuda.synchronize()
    residual = torch.cuda.memory_allocated()
    try:
        run, model, opt = build(preset, batch, seq, ARMS[arm], device)
    except LoudError as e:
        return {**row, "status": "REFUSED", "reason": str(e).replace("\n", " ")}
    try:
        ids = torch.randint(0, 256, (batch, seq), device=device)
        seg = int(getattr(run.train, "segment_len", 0) or 0)
        for _ in range(warm):
            step(model, opt, ids, segment_len=seg)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times: list[float] = []
        counters = None
        for _ in range(steps):
            t0 = time.perf_counter()
            _, counters = step(model, opt, ids, segment_len=seg)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        peak = (torch.cuda.max_memory_allocated() - residual) / GIB
        return {
            **row,
            "status": "ok",
            "model": {"block_form": run.model.block_form, "params": sum(p.numel() for p in model.parameters())},
            "peak_gib": round(peak, 3),
            "ms_step": round(med * 1e3, 2),
            "ms_min": round(min(times) * 1e3, 2),
            "kb_s": round(batch * seq / 1e3 / med, 2),
            "steps": steps,
            "residual_gib": round(residual / GIB, 3),
            "counters": {
                k: counters.get(k, 0)
                for k in ("loop_segments", "iter_checkpoint_saved", "offload_tensors", "offload_gib_x100")
            },
            "offload": (
                dict(model._act_offload.stats())
                if getattr(model, "_act_offload", None) is not None
                else None
            ),
            "memory": {k: round(v, 3) if isinstance(v, float) else v for k, v in _stats().items()},
        }
    except LoudError as e:
        # Raised INSIDE a forward (the offload budget's refusal): the arm is
        # refused at this shape, which is a result, not a reason to kill the sweep.
        # `__traceback__` is cleared because the frame it pins holds the model,
        # the loss and its whole save set — measured as +4.1 GiB on every arm
        # that ran after a refused one.
        reason = str(e).replace("\n", " ")
        e.__traceback__ = None
        return {**row, "status": "REFUSED", "reason": reason}
    except torch.cuda.OutOfMemoryError as e:
        reason = str(e).split("\n")[0][:200]
        e.__traceback__ = None
        return {**row, "status": "OOM", "reason": reason}
    finally:
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()


# ------------------------------------------------------------------- the map


def _frame(f) -> str:
    """One allocator frame as `file.py:line name`, from a dict OR a sequence.

    `torch.cuda.memory_snapshot()` documents Frame as a TypedDict, but the
    2.13.0+cu129 build returns `[filename, line, name]` sequences — indexing a
    dict-style key there is the TypeError that killed the first map.
    """
    if isinstance(f, dict):
        return f"{Path(str(f.get('filename'))).name}:{f.get('line')} {f.get('name')}"
    try:
        return f"{Path(str(f[0])).name}:{f[1]} {f[2]}"
    except (IndexError, TypeError):
        return str(f)


class _Census:
    """`saved_tensors_hooks` pair that keeps shape, dtype and the caller's frame."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def pack(self, t):
        import torch

        if isinstance(t, torch.Tensor) and t.is_floating_point() and t.numel():
            frames = [
                f"{Path(fr.filename).name}:{fr.lineno} {fr.name}"
                for fr in traceback.extract_stack(limit=14)
                if "torch/autograd" not in fr.filename and Path(fr.filename).name != Path(__file__).name
            ]
            self.rows.append(
                {
                    "shape": tuple(t.shape),
                    "dtype": str(t.dtype).replace("torch.", ""),
                    "bytes": t.numel() * t.element_size(),
                    "at": frames[-1] if frames else "?",
                }
            )
        return t

    def unpack(self, t):
        return t


def _trace_peak(preset: str, batch: int, seq: int, device: str) -> dict:
    """The forward-end live set — the blocks and the stack that holds each one.

    The peak composition is taken at the END OF THE FORWARD, which is the
    moment the autograd save set is at its maximum: `memory_snapshot()` then
    lists every live block with its `requested_size` and the stack that
    allocated it. `_record_memory_history(enabled="state")` makes the allocator
    keep that context per live block (mode `"state"`, not `"all"`: there is no
    history ring to overflow, which is what made a 100k-entry replay lose the
    forward and report 1.1 GiB against a 4.6 GiB allocator peak).
    The step is then finished so `max_memory_allocated` reports the card's own
    peak — the forward-end set + the backward's gradients and transients.

    `saved_tensors_hooks` gives the same forward's SAVED set with shapes: the
    snapshot carries stacks, not shapes, so the two tables are one instrument.
    """
    import torch

    run, model, opt = build(preset, batch, seq, {}, device)
    ids = torch.randint(0, 256, (batch, seq), device=device)
    step(model, opt, ids)  # warm: optimizer state, cudnn workspaces, caches
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    census = _Census()
    torch.cuda.memory._record_memory_history(enabled="state", context="all", stacks="all")
    try:
        with torch.autograd.graph.saved_tensors_hooks(census.pack, census.unpack):
            _, loss, _ = model(ids, targets=ids, counters=Counters())
        torch.cuda.synchronize()
        fwd_alloc = torch.cuda.memory_allocated()
        snap = torch.cuda.memory_snapshot()
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)

    def blame(frames: list[str]) -> str:
        for f in frames:
            if "dormouse" in f:
                return f
        return frames[0] if frames else "?"

    segments = snap["segments"] if isinstance(snap, dict) else snap
    live: list[tuple[int, list[str]]] = []
    for seg in segments:
        for blk in seg["blocks"]:
            if blk.get("state") != "active_allocated":
                continue
            frames = [
                _frame(f)
                for f in (blk.get("frames") or [])
                if "torch/autograd" not in _frame(f)
            ][:6]
            live.append((blk.get("requested_size") or blk.get("size"), frames))
    live_total = sum(size for size, _ in live)
    by_frame: collections.Counter[str] = collections.Counter()
    for size, frames in live:
        by_frame[blame(frames)] += size

    # Finish the step: the allocator's own peak over the whole fwd+bwd.
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    groups: collections.Counter[tuple] = collections.Counter()
    for r in census.rows:
        groups[(r["shape"], r["dtype"], r["bytes"], r["at"])] += 1
    saved_top = [
        {
            "shape": list(shape),
            "dtype": dtype,
            "bytes_each": nbytes,
            "count": n,
            "gib": round(nbytes * n / GIB, 4),
            "at": at,
        }
        for (shape, dtype, nbytes, at), n in sorted(
            groups.items(), key=lambda kv: -(kv[0][2] * kv[1])
        )[:10]
    ]

    return {
        "status": "ok",
        "batch": batch,
        "seq": seq,
        "params_gib": round(sum(p.numel() * p.element_size() for p in model.parameters()) / GIB, 3),
        "saved_gib": round(sum(r["bytes"] for r in census.rows) / GIB, 4),
        "saved_tensors": len(census.rows),
        "forward_end_gib": round(fwd_alloc / GIB, 4),
        "live_total_gib": round(live_total / GIB, 4),
        "live_blocks": len(live),
        "allocator_peak_gib": round(peak / GIB, 4),
        "backward_gap_gib": round((peak - fwd_alloc) / GIB, 4),
        "top_blocks": [
            {"gib": round(size / GIB, 4), "frames": frames}
            for size, frames in sorted(live, key=lambda x: -x[0])[:10]
        ],
        "top_frames": [
            {"frame": k, "gib": round(v / GIB, 4), "share": round(v / live_total, 3)}
            for k, v in by_frame.most_common(12)
        ],
        "saved_top": saved_top,
        "memory": {k: round(v, 3) if isinstance(v, float) else v for k, v in _stats().items()},
    }


# -------------------------------------------------------------------- the ladder


def ladder(
    preset: str, seq: int, arm: str, batches: list[int], *, warm: int, steps: int,
    ceiling: float, device: str,
) -> list[dict]:
    out = []
    for batch in batches:
        row = measure(preset, batch, seq, arm, warm=warm, steps=steps, device=device)
        out.append(row)
        print(
            f"  ladder {arm:20s} b{batch:<3d} {row['status']:8s} "
            + (f"peak={row['peak_gib']:6.3f} GiB  {row['ms_step']:8.2f} ms  {row['kb_s']:7.1f} kB/s"
               if row["status"] == "ok" else str(row.get("reason", ""))[:90]),
            flush=True,
        )
        if row["status"] in ("OOM", "REFUSED"):
            break  # a larger batch cannot fit / a bigger forward cannot fit the budget
        if row["status"] == "ok" and row["peak_gib"] >= ceiling:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="configs/flagship3_dense.toml")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--steps", type=int, default=7)
    ap.add_argument("--ceiling-gib", type=float, default=15.0)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--batches", default="8,12,16,24,32")
    ap.add_argument(
        "--ladder-arms",
        default="",
        help="comma list of arms to ladder (default: the lowest-peak arm at --batch)",
    )
    ap.add_argument("--skip-map", action="store_true")
    ap.add_argument("--skip-ladder", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("loud: no CUDA device — this bench measures the card", file=sys.stderr)
        return 2
    device = "cuda"
    out: dict[str, object] = {
        "preset": args.preset,
        "seq": args.seq,
        "warm": args.warm,
        "steps": args.steps,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "arms": [],
        "map": None,
        "ladder": {},
    }
    print(f"== {args.preset} · b{args.batch} x s{args.seq} · {torch.cuda.get_device_name()} ==", flush=True)

    if not args.skip_map:
        try:
            out["map"] = _trace_peak(args.preset, args.batch, args.seq, device)
            m = out["map"]
            print(f"  map: saved {m['saved_gib']:.3f} GiB in {m['saved_tensors']} tensors; "
                  f"forward-end live {m['forward_end_gib']:.3f} GiB in {m['live_blocks']} blocks "
                  f"(allocator peak {m['allocator_peak_gib']:.3f})", flush=True)
        except Exception as e:  # the map must not cost the ladder
            out["map"] = {"status": "ERROR", "reason": f"{type(e).__name__}: {e}"}
            print(f"  map FAILED: {type(e).__name__}: {e}", flush=True)
            e.__traceback__ = None  # see `measure`: a live traceback pins the graph
            gc.collect()
            torch.cuda.empty_cache()

    for arm in [a for a in args.arms.split(",") if a]:
        row = measure(args.preset, args.batch, args.seq, arm, warm=args.warm, steps=args.steps, device=device)
        out["arms"].append(row)
        print(
            f"  arm {arm:20s} {row['status']:8s} "
            + (f"peak={row['peak_gib']:6.3f} GiB  {row['ms_step']:8.2f} ms  {row['kb_s']:7.1f} kB/s"
               if row["status"] == "ok" else str(row.get("reason", ""))[:100]),
            flush=True,
        )

    if not args.skip_ladder:
        ok = [r for r in out["arms"] if r["status"] == "ok"]
        if ok:
            best = min(ok, key=lambda r: r["peak_gib"])
            out["best_arm"] = best["arm"]
            names = [a for a in args.ladder_arms.split(",") if a] or [best["arm"]]
            batches = sorted({int(b) for b in args.batches.split(",") if b})
            for arm in names:
                print(f"== batch ladder: arm {arm!r} (lowest peak at b{args.batch}: {best['arm']!r}, {best['peak_gib']:.3f} GiB) ==", flush=True)
                out["ladder"][arm] = ladder(
                    args.preset, args.seq, arm, batches,
                    warm=args.warm, steps=args.steps, ceiling=args.ceiling_gib, device=device,
                )

    text = json.dumps(out, indent=2, default=str)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text)
        print(f"json -> {args.json}", flush=True)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
