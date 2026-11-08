#!/usr/bin/env -S uv run --script
"""What the DATA PATH costs per training step: windows/s, MB/s, warm and cold.

The live recipe (`configs/mix_refetch.toml`, b=8, s=1024) consumes
`batch × (seq_len + 1) = 8 200` source bytes per step and runs at ~147 ms/step,
i.e. ~56 kB/s. That number is the STEP RATE, not the loader's cost: it says how
fast the model consumes bytes, and the question this script answers is how much
of the 147 ms the loader's own CPU work takes. Measured here, not inferred:

* `MixtureStream.next_batch()` end to end (source draw + `ByteStream` ring +
  `_windows`), warm, over the plan's six sources at their real weights;
* the same window taken from each source alone, split into `_read` (mmap slice +
  ring maintenance) and `_windows` (the `[b, t]` input/target build);
* the mixture's own per-step draw overhead (`source_for_step`);
* a **cold-ish** pass: `posix_fadvise(DONTNEED)` + `madvise(DONTNEED)` on every
  source, then the first windows after the drop (the page cache no longer
  serves them — the first windows are 30-200x the warm one) and the average
  over the first few hundred windows;
* the `doc_masking=true` alternative (`DocPackedStream`) where a document
  boundary can be derived from the refetched corpus. It cannot be derived for
  the plan as a whole: `code.bin` and `ru.bin` have no `\\n\\n---\\n\\n`
  separator at all and `math.bin` has 14 in ~200 MB, and no source has a
  `.docs.json` sidecar. The four sources where it IS buildable (fwedu, cosmo,
  finemath, math) are measured so the cost of turning `doc_masking` on is a
  number rather than a guess. The sidecar is built IN MEMORY; nothing in
  `data_local/` is written.

**Cache honesty.** Warm and cold are measured on the SAME byte range of the
same instances: cold first (after the drop), then `rewind()` and warm. That
makes the pair a controlled comparison instead of two different positions.
`mmap` page faults do not show up in `/proc/self/io`'s `read_bytes` on this
kernel (measured 0), so the evidence that the drop worked is the first-batch
wall time, which is printed, not a claimed I/O counter.

**CPU only.** No CUDA, no `CUDA_VISIBLE_DEVICES` needed (torch is imported for
`_windows`'s tensors only). The `x.to(device)` copy is outside every number
here and is not the question: it moves 8 200 B per step.

    cd port && uv run scripts/bench_loader.py
    cd port && uv run scripts/bench_loader.py --steps 5000 --cold-steps 800
    cd port && uv run scripts/bench_loader.py --json /tmp/opencode/loader.json
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from dormouse.train.data import ByteStream, DocPackedStream, _windows
from dormouse.train.mixture import MixtureStream, load_mix_plan, source_for_step

#: The refetched corpus writes one document followed by this separator
#: (`scripts/refetch_pretrain.py`). It is the only boundary evidence in the
#: `.bin` — no `.docs.json` sidecar exists beside any source.
DOC_SEP = b"\n\n---\n\n"

SCAN_CHUNK = 8 << 20


def _stats(wall_ns: list[int], cpu_ns: list[int], window: int) -> dict:
    """One timing block: wall mean/median/p99, CPU mean, windows/s, MB/s.

    Wall and CPU are both kept per call. On a loaded box a 70 us call gets
    preempted and the wall mean doubles while the CPU mean does not move, so
    the CPU column is the loader's cost and the wall column is what a step
    would see when it lands on a busy core. Both are reported; neither is
    silently substituted for the other.
    """
    n = len(wall_ns)
    ordered = sorted(wall_ns)
    cpu_ordered = sorted(cpu_ns)
    wall_total = sum(wall_ns) / 1e9
    cpu_total = sum(cpu_ns) / 1e9
    return {
        "n": n,
        "wall_ms_mean": wall_total / n * 1e3,
        "wall_ms_median": ordered[n // 2] / 1e6,
        "wall_ms_p99": ordered[int(n * 0.99)] / 1e6,
        "wall_ms_max": ordered[-1] / 1e6,
        "cpu_ms_mean": cpu_total / n * 1e3,
        "cpu_ms_median": cpu_ordered[n // 2] / 1e6,
        "windows_s": n / wall_total,
        "cpu_windows_s": n / cpu_total,
        "mb_s": n * window / wall_total / 1e6,
        "cpu_mb_s": n * window / cpu_total / 1e6,
    }


def _time_calls(fn, n: int, window: int) -> dict:
    """Time `n` calls of `fn`; per-call wall AND CPU ns, so load cannot fake a cost."""
    wall: list[int] = []
    cpu: list[int] = []
    for _ in range(n):
        c0 = time.process_time_ns()
        t0 = time.perf_counter_ns()
        fn()
        wall.append(time.perf_counter_ns() - t0)
        cpu.append(time.process_time_ns() - c0)
    return _stats(wall, cpu, window)


def _fmt(block: dict) -> str:
    return (
        f"{block['cpu_ms_mean']:7.3f} cpu  "
        f"({block['wall_ms_mean']:7.3f} wall, med {block['wall_ms_median']:7.3f}, "
        f"p99 {block['wall_ms_p99']:8.3f})  "
        f"{block['cpu_windows_s']:8.0f} win/s  {block['cpu_mb_s']:6.1f} MB/s"
    )


def _source_stream(stream):
    """The `ByteStream` under a `MixtureStream` entry (or the entry itself)."""
    return getattr(stream, "source", stream)


def _provenance() -> dict:
    """The commit and the load average a number in this report has to carry."""
    try:
        head = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        head = "unknown"
    try:
        load = os.getloadavg()
    except OSError:
        load = (0.0, 0.0, 0.0)
    return {
        "git_head": head or "unknown",
        "load_avg_1m_5m_15m": [round(v, 2) for v in load],
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_count": os.cpu_count(),
    }


def _drop_cache(streams: list) -> list[str]:
    """Evict each source's page cache: `posix_fadvise` on the fd, `madvise` the map.

    Both are needed here: fadvise is the documented call, and `madvise` on the
    live mapping is what makes a remap after the drop fault instead of reading
    the old resident pages. A kernel without either is reported, not assumed.
    """
    used = []
    for stream in streams:
        st = _source_stream(stream)
        fd = getattr(st, "_fd", None)
        if fd is not None and hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            used.append("fadvise")
        mm = getattr(st, "_mm", None)
        if mm is not None and hasattr(mm, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
            mm.madvise(mmap.MADV_DONTNEED)
            used.append("madvise")
    return sorted(set(used))


def _separator_starts(path: Path, sep: bytes = DOC_SEP) -> list[int]:
    """Document starts of a `.bin` written as `doc + sep + doc + ...`, in memory.

    A chunked scan with an overlap of `len(sep) - 1` so a separator straddling a
    chunk boundary is found once. This is the boundary source a
    `scripts/build_doc_sidecar.py` sidecar would carry; it is used directly
    because writing a sidecar into `data_local/` would modify the corpus.
    """
    size = path.stat().st_size
    starts = [0]
    overlap = len(sep) - 1
    with path.open("rb") as fh:
        base = 0
        tail = b""
        while True:
            chunk = fh.read(SCAN_CHUNK)
            if not chunk:
                break
            buf = tail + chunk
            off = base - len(tail)
            i = buf.find(sep)
            while i != -1:
                start = off + i + len(sep)
                if 0 < start < size:
                    starts.append(start)
                i = buf.find(sep, i + len(sep))
            base += len(chunk)
            tail = buf[-overlap:] if overlap else b""
    assert starts[0] == 0, "a document must start at byte 0"
    assert all(a < b for a, b in zip(starts, starts[1:], strict=False)), "starts not ascending"
    assert starts[-1] < size, f"last document start {starts[-1]} >= {size}"
    return starts


def _selfcheck(plan, batch: int, seq_len: int, steps: int) -> dict:
    """The two invariants the numbers rest on, asserted before they are taken.

    1. `_windows` serves EXACTLY the bytes it was handed (the target is the next
       byte of the same row, the row boundaries do not bleed).
    2. Two `MixtureStream`s over one plan and seed serve the same bytes, so a
       timing measured on one is the sequence the trainer would read.
    """
    raw = bytes((i * 37 + 11) % 256 for i in range(batch * (seq_len + 1)))
    x, y = _windows(raw, batch, seq_len)
    flat = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long().view(batch, seq_len + 1)
    assert torch.equal(x, flat[:, :-1]), "_windows inputs are not the raw bytes"
    assert torch.equal(y, flat[:, 1:]), "_windows targets are not the next byte"
    assert x.shape == y.shape == (batch, seq_len)

    a = MixtureStream(plan, batch, seq_len, steps)
    b = MixtureStream(plan, batch, seq_len, steps)
    try:
        order = []
        for _ in range(8):
            xa, ya = a.next_batch()
            xb, yb = b.next_batch()
            assert torch.equal(xa, xb), "two same-seed mixtures disagree on inputs"
            assert torch.equal(ya, yb), "two same-seed mixtures disagree on targets"
            order.append(source_for_step(plan, a.step - 1, steps))
    finally:
        a.close()
        b.close()
    return {"served": order}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", type=Path, default=Path("configs/mix_refetch.toml"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=3000, help="warm windows timed per rep")
    ap.add_argument("--reps", type=int, default=3, help="warm blocks; the calmest is headlined")
    ap.add_argument("--per-source-steps", type=int, default=1000)
    ap.add_argument("--cold-steps", type=int, default=400)
    ap.add_argument("--cold-reps", type=int, default=3, help="drop + re-measure, keep the calmest")
    ap.add_argument("--packed-steps", type=int, default=300)
    ap.add_argument("--step-ms", type=float, default=147.0, help="the GPU step to price against")
    ap.add_argument("--json", type=Path, default=None, help="write every block here")
    ap.add_argument("--skip-cold", action="store_true")
    ap.add_argument("--skip-packed", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(1)  # the loader is one thread; keep the measurement one too

    plan = load_mix_plan(a.plan)
    window = a.batch * (a.seq_len + 1)
    steps = a.steps + a.cold_steps + 1000
    prov = _provenance()
    out: dict = {
        "plan": str(a.plan),
        "batch": a.batch,
        "seq_len": a.seq_len,
        "payload_bytes_per_window": window,
        "step_ms": a.step_ms,
        "provenance": prov,
    }
    print(f"plan {a.plan}  b={a.batch} s={a.seq_len}  payload {window} B/window")
    print(
        f"step to price against: {a.step_ms:.0f} ms | {prov['date']} | HEAD {prov['git_head']} "
        f"| load {prov['load_avg_1m_5m_15m']} on {prov['cpu_count']} cpu"
    )
    print()

    # --- document boundaries: what the doc_masking arm has to work with ------
    sep_counts = {}
    starts_by_name: dict[str, list[int]] = {}
    print("sources (separator scan; no `.docs.json` exists beside any source):")
    for name, path in plan.sources.items():
        t0 = time.perf_counter()
        starts = _separator_starts(path)
        starts_by_name[name] = starts
        sep_counts[name] = {
            "docs": len(starts),
            "separators": len(starts) - 1,
            "bytes": path.stat().st_size,
            "scan_s": time.perf_counter() - t0,
        }
        print(
            f"  {name:9s} {path.stat().st_size / 2**20:8.0f} MiB  "
            f"docs={len(starts):7d}  seps={len(starts) - 1:7d}"
        )
    out["separators"] = sep_counts
    print()

    # --- self-check ---------------------------------------------------------
    out["selfcheck"] = _selfcheck(plan, a.batch, a.seq_len, steps)
    print(f"self-check OK; first 8 draws: {' '.join(out['selfcheck']['served'])}")
    print()

    mix = MixtureStream(plan, a.batch, a.seq_len, steps)

    # --- cold-ish: drop the cache, then the first windows after it ----------
    cold = {}
    if not a.skip_cold:
        cold_reps = []
        for _ in range(a.cold_reps):
            dropped = _drop_cache(list(mix._streams.values()))
            mix.rewind()
            first = _time_calls(mix.next_batch, 8, window)
            cold_reps.append(
                {
                    "drop": dropped,
                    "first8": first,
                    "block": _time_calls(mix.next_batch, a.cold_steps, window),
                }
            )
        # The quietest pass is the estimate: dropping the cache is
        # deterministic, so the spread here is the box's load, not the disk.
        cold = min(cold_reps, key=lambda r: r["block"]["wall_ms_mean"])
        cold["reps"] = cold_reps
        print(
            f"cold-ish (dropped via {', '.join(cold['drop']) or 'NOTHING'}), {a.cold_reps} passes:"
        )
        for i, rep in enumerate(cold_reps):
            print(f"  rep {i}  first 8 {_fmt(rep['first8'])}")
            print(f"         next {a.cold_steps:4d} {_fmt(rep['block'])}")
        mix.rewind()  # same byte range as the cold pass, now resident
        print()
    else:
        # Without the cold pass the first accesses would fault and the "warm"
        # block would be a cold number: warm the streams first.
        mix.rewind()
        for _ in range(200):
            mix.next_batch()
        print("cold skipped; 200 windows read before the timed warm block")
        print()

    # --- warm: the steady state, same range ---------------------------------
    warm_reps = [_time_calls(mix.next_batch, a.steps, window) for _ in range(a.reps)]
    warm = min(warm_reps, key=lambda b: b["cpu_ms_mean"])
    out["warm"] = warm
    out["warm_reps"] = warm_reps
    print(f"warm MixtureStream.next_batch, {a.reps} blocks of {a.steps} windows:")
    for i, block in enumerate(warm_reps):
        print(f"  rep {i}  {_fmt(block)}")
    print(f"  headlined rep {warm_reps.index(warm)} (lowest CPU mean; wall is the load's)")
    print()

    # --- the mixture's own draw overhead ------------------------------------
    draw = _time_calls(
        lambda: source_for_step(plan, mix.step, steps, mix.seed), min(a.steps, 5000), window
    )
    out["source_for_step"] = draw
    print("mixture draw overhead (source_for_step, no bytes):")
    print(
        f"  {draw['wall_ms_mean']:8.4f} ms mean "
        f"(med {draw['wall_ms_median']:7.4f}, cpu {draw['cpu_ms_mean']:7.4f})"
    )
    print()

    # --- per source: read vs window build -----------------------------------
    per_source = {}
    print("per source (its own next_batch, then the internal split):")
    for name in plan.names:
        stream = mix._streams[name]
        block = _time_calls(stream.next_batch, a.per_source_steps, window)
        raw = stream._read(window)
        read = _time_calls(lambda s=stream: s._read(window), a.per_source_steps, window)
        win = _time_calls(lambda r=raw: _windows(r, a.batch, a.seq_len), a.per_source_steps, window)
        per_source[name] = {"next_batch": block, "_read": read, "_windows": win}
        print(f"  {name:9s} {_fmt(block)}")
        print(
            f"            _read {read['cpu_ms_mean']:7.3f} ms  "
            f"_windows {win['cpu_ms_mean']:7.3f} ms (CPU)"
        )
    out["per_source"] = per_source
    print()

    # --- doc-packed: only where boundaries exist ----------------------------
    packed = {}
    if not a.skip_packed:
        print("doc-packed (in-memory sidecar from `\\n\\n---\\n\\n`, one row = one doc/chunk):")
        for name, path in plan.sources.items():
            starts = starts_by_name[name]
            if len(starts) < 2:
                print(f"  {name:9s} skipped: no document boundaries in the bytes")
                continue
            src = ByteStream(path, a.batch, a.seq_len, check_min=False)
            dps = DocPackedStream(src, starts)
            try:
                for _ in range(20):
                    dps.next_batch_masked()
                block = _time_calls(dps.next_batch_masked, a.packed_steps, window)
                counts = dps.doc_counts()
            finally:
                dps.close()
            scored = counts["docmask_scored"] / (counts["docmask_windows"] * a.seq_len)
            packed[name] = {"block": block, "counts": counts, "scored_frac": scored}
            print(f"  {name:9s} {_fmt(block)}  scored {scored:.1%} of positions")
        out["packed"] = packed
        print()

    # --- what it means ------------------------------------------------------
    cpu_ms = warm["cpu_ms_mean"]
    wall_ms = warm["wall_ms_mean"]
    cpu_share = cpu_ms / a.step_ms * 100
    wall_share = wall_ms / a.step_ms * 100
    print("=" * 78)
    print(
        f"WARM loader: {cpu_ms:.3f} ms CPU / {wall_ms:.3f} ms wall mean, "
        f"{warm['wall_ms_median']:.3f} ms median per step"
    )
    print(f"  share of a {a.step_ms:.0f} ms step: CPU {cpu_share:.3f}%  wall {wall_share:.3f}%")
    print(
        f"  ceiling: {warm['cpu_windows_s']:.0f} windows/s CPU = "
        f"{warm['cpu_mb_s']:.1f} MB/s payload, "
        f"{a.step_ms / cpu_ms:.0f}x the live step's rate"
    )
    if not a.skip_cold:
        f8 = cold["first8"]
        cb = cold["block"]
        print(
            f"  cold first window: {f8['wall_ms_mean']:.3f} ms wall mean "
            f"(max {f8['wall_ms_max']:.3f}, cpu {f8['cpu_ms_mean']:.3f}), then "
            f"{cb['wall_ms_mean']:.3f} ms wall mean over {a.cold_steps} "
            f"(cpu {cb['cpu_ms_mean']:.3f}) = {cb['wall_ms_mean'] / a.step_ms * 100:.3f}% of a step"
        )
    if packed:
        pname, pdata = next(iter(packed.items()))
        p_cpu = pdata["block"]["cpu_ms_mean"]
        print(
            f"  doc_masking arm ({pname}): {p_cpu:.3f} ms CPU/window, "
            f"{p_cpu / a.step_ms * 100:.2f}% of a step IF it could be enabled on this corpus"
        )
    out["verdict"] = {
        "loader_ms_cpu_mean": cpu_ms,
        "loader_ms_wall_mean": wall_ms,
        "share_of_step_cpu_pct": cpu_share,
        "share_of_step_wall_pct": wall_share,
        "windows_s_cpu": warm["cpu_windows_s"],
        "mb_s_cpu": warm["cpu_mb_s"],
        "cold_block_wall_ms": None if a.skip_cold else cold["block"]["wall_ms_mean"],
        "cold_block_cpu_ms": None if a.skip_cold else cold["block"]["cpu_ms_mean"],
        "cold_first_window_wall_ms": None if a.skip_cold else cold["first8"]["wall_ms_mean"],
    }
    if a.json is not None:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"json -> {a.json}")

    mix.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
