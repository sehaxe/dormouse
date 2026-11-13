#!/usr/bin/env -S uv run --script
"""Is one `flagship3_dense` step limited by the GPU or by the host?

The kernel audit (`docs/reviews/2026-11-08-kernel-audit.md`) accounts for 58% of
the step as arithmetic and leaves 42% unnamed. This instrument names it from
measurements, in four modes, all on the REAL trainer (it imports `dormouse.cli`
and runs `train` in-process, so the model, optimizer, dataloader and step body
are the ones a run uses):

* ``--mode census``  — `torch.profiler` over 20 warm + 20 recorded steps: every
  executed kernel by name (count, CUDA time), every `cudaLaunchKernel`, per-stream
  totals (does anything overlap?), and the inter-kernel GAPS on the compute
  stream — the difference between "the GPU is busy" and "the GPU is waiting".
* ``--mode phases``  — no profiler: `perf_counter` around loader / forward /
  backward / firewall / clip / optimizer (HOST enqueue time), paired with
  `torch.cuda.Event` markers around the same seams (GPU time ON THE STREAM).
  Host ms << GPU ms is the definition of "the host is hidden"; the event spans
  add up to GPU busy, the period between step starts is the wall, and the
  residue is the gap that has to be explained.
* ``--mode sync``    — `torch.cuda.set_sync_debug_mode("warn")` for 3 steps,
  recording every synchronizing call with its Python frame. A step that reads
  the device once cannot hide host time.
* ``--mode loader``  — `MixtureStream.next_batch` is injected with 0 / 5 / 20 /
  40 ms of extra CPU work. If the period does not move, the loader is hidden
  under the GPU and a prefetch thread (or a stream) buys exactly zero.

Every number is printed with its step count; `--json` writes the lot so a later
change can be diffed against this file instead of re-arguing it.

Protocol: b8 x s1024, the lane-C recipe (`ns_dtype=bf16`,
`muon_state_dtype=bf16`, `muon_lr_mult=12`, `mix_refetch`), `eval=false`,
`ckpt_every=1e6`, `log_every=1000` so no report step lands in a measured window.
One GPU process at a time: run it under `scripts/gpu_lock.sh`.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
import traceback
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DATA = "/home/sehaxe/dormouse/data_local/best/fwedu_train.bin"
PRESET = "configs/flagship3_dense.toml"
OUT = Path("/tmp/opencode/hvg")


def train_argv(
    steps: int, tag: str, log_every: int = 1000, extra: list[str] | None = None
) -> list[str]:
    return [
        "train",
        PRESET,
        "--data",
        DATA,
        "--device",
        "cuda",
        "--set",
        f"steps={steps}",
        "--set",
        "batch=8",
        "--set",
        "seq_len=1024",
        "--set",
        "seed=1",
        "--set",
        "lr=3e-4",
        "--set",
        "opt=mix",
        "--set",
        "muon_lr_mult=12",
        "--set",
        "mix_plan=configs/mix_refetch.toml",
        "--set",
        "doc_masking=false",
        "--set",
        "eval=false",
        "--set",
        f"log_every={log_every}",
        "--set",
        "ckpt_every=1000000",
        "--set",
        f"out_dir={OUT / ('run_' + tag)}",
        "--set",
        "ns_dtype=bf16",
        "--set",
        "muon_state_dtype=bf16",
        "--runs-dir",
        str(OUT / "runs"),
        *(extra or []),
    ]


def run_cli(argv: list[str]) -> None:
    from dormouse.cli import main

    old = sys.argv
    sys.argv = ["dormouse", *argv]
    try:
        main()
    except SystemExit as e:  # click's standalone mode exits the process
        if e.code not in (0, None):
            raise
    finally:
        sys.argv = old


def wrap(obj, name: str, pre=None, post=None) -> None:
    """Time a callable by name without changing its contract."""
    orig = getattr(obj, name)

    def inner(*a, **kw):
        if pre is not None:
            pre()
        out = orig(*a, **kw)
        if post is not None:
            post()
        return out

    setattr(obj, name, inner)


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))
    return s[i]


# --------------------------------------------------------------------------
# phases: host enqueue time + GPU stream time, no profiler
# --------------------------------------------------------------------------
def mode_phases(args) -> dict:
    import dormouse.train.loop as loop
    from dormouse.train.mixture import MixtureStream
    from dormouse.train.optim import MixOptimizer

    step = [0]
    t = {}
    host = defaultdict(list)
    spans = defaultdict(list)
    events: list[tuple[int, dict]] = []
    wall_starts: list[float] = []

    def ev():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def on_zero_grad_pre():
        t["zg"] = time.perf_counter()
        wall_starts.append(t["zg"])
        t["ev_zg"] = ev()

    def on_zero_grad_post():
        host["zero_grad"].append(1e3 * (time.perf_counter() - t["zg"]))

    def on_loader_pre():
        t["loader"] = time.perf_counter()

    def on_loader_post():
        t["loader_end"] = time.perf_counter()
        host["loader"].append(1e3 * (t["loader_end"] - t["loader"]))
        t["ev_fwd"] = ev()

    def on_bwd_pre():
        t["bwd"] = time.perf_counter()
        t["ev_bwd"] = ev()

    def on_bwd_post():
        t["bwd_end"] = time.perf_counter()
        host["backward"].append(1e3 * (t["bwd_end"] - t["bwd"]))
        host["forward"].append(1e3 * (t["bwd"] - t["loader_end"]))
        t["ev_fire"] = ev()

    def on_clip_pre():
        t["clip"] = time.perf_counter()
        host["firewall"].append(1e3 * (t["clip"] - t["bwd_end"]))
        t["ev_opt"] = ev()

    def on_clip_post():
        host["clip"].append(1e3 * (time.perf_counter() - t["clip"]))

    def on_opt_pre():
        t["opt"] = time.perf_counter()

    def on_opt_post():
        t["opt_end"] = time.perf_counter()
        host["opt"].append(1e3 * (t["opt_end"] - t["opt"]))
        t["ev_end"] = ev()
        step[0] += 1
        events.append(
            (
                step[0],
                {
                    "fwd": (t["ev_fwd"], t["ev_bwd"]),
                    "bwd": (t["ev_bwd"], t["ev_fire"]),
                    "fire": (t["ev_fire"], t["ev_opt"]),
                    "opt": (t["ev_opt"], t["ev_end"]),
                    "step": (t["ev_zg"], t["ev_end"]),
                },
            )
        )

    wrap(MixOptimizer, "zero_grad", on_zero_grad_pre, on_zero_grad_post)
    wrap(MixtureStream, "next_batch", on_loader_pre, on_loader_post)
    wrap(torch.Tensor, "backward", on_bwd_pre, on_bwd_post)
    wrap(loop, "clip_grads", on_clip_pre, on_clip_post)
    wrap(MixOptimizer, "step", on_opt_pre, on_opt_post)

    run_cli(train_argv(args.steps, "phases"))

    torch.cuda.synchronize()
    for _n, pairs in events:
        for name, (a, b) in pairs.items():
            spans[name].append(a.elapsed_time(b))

    keep = args.skip
    rows = sorted(host)
    out = {"mode": "phases", "steps": step[0], "kept": list(range(keep + 1, step[0] + 1))}
    table = []
    for name in rows:
        xs = host[name][keep:]
        table.append(
            {
                "phase": name,
                "host_med_ms": st.median(xs),
                "host_p95_ms": pct(xs, 95),
                "host_sum_ms": sum(xs),
            }
        )
    out["host"] = table
    n_keep = min(len(host[k]) for k in rows) if rows else 0
    out["host_total_med_ms"] = st.median(
        [sum(host[k][i] for k in rows) for i in range(keep, n_keep)]
    )
    gpu = {name: (st.median(v[keep:]), pct(v[keep:], 95)) for name, v in spans.items()}
    out["gpu_stream_ms"] = gpu
    per = [wall_starts[i + 1] - wall_starts[i] for i in range(keep, len(wall_starts) - 1)]
    out["period_med_ms"] = 1e3 * st.median(per)
    out["period_p95_ms"] = 1e3 * pct(per, 95)
    out["torch_threads"] = torch.get_num_threads()

    span_of = {"forward": "fwd", "backward": "bwd", "firewall": "fire"}
    print(
        f"\n=== phases: {step[0]} steps, kept {len(keep_range(keep, step[0]))}, "
        f"threads={torch.get_num_threads()}"
    )
    print(f"{'phase':<12} {'host med':>9} {'host p95':>9} {'GPU-on-stream med':>18}")
    for row in table:
        g = gpu.get(span_of.get(row["phase"], row["phase"]), (0.0, 0.0))[0]
        print(f"{row['phase']:<12} {row['host_med_ms']:9.3f} {row['host_p95_ms']:9.3f} {g:18.3f}")
    print(f"{'TOTAL':<12} {out['host_total_med_ms']:9.3f} {'':>9} {gpu['step'][0]:18.3f}")
    print(f"{'period':<12} {out['period_med_ms']:9.3f} {out['period_p95_ms']:9.3f}")
    print(
        f"host/GPU = {out['host_total_med_ms'] / max(gpu['step'][0], 1e-9):.3f}; "
        f"headroom = {gpu['step'][0] - out['host_total_med_ms']:.2f} ms of host work per step "
        "before the CPU becomes the limiter"
    )
    return out


def keep_range(keep: int, n: int) -> range:
    return range(0) if n <= keep else range(0, n - keep)


# --------------------------------------------------------------------------
# census: kernel launches and gaps from a chrome trace
# --------------------------------------------------------------------------
def mode_census(args) -> dict:
    from torch.profiler import ProfilerActivity, profile

    from dormouse.train.optim import MixOptimizer

    warm, active = args.warm, args.active
    need = warm + active + 1
    state = {"n": 0, "p": None, "path": None, "done": False}
    state["path"] = args.trace
    mark = {"t": None}

    def pre():
        pass

    def post():
        state["n"] += 1
        n = state["n"]
        if n == warm:
            torch.cuda.synchronize()
            p = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=active, repeat=1),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            p.start()
            state["p"] = p
            mark["t"] = time.perf_counter()
        elif state["p"] is not None and not state["done"]:
            if n == warm + active:
                # Drain BEFORE the last `p.step()`: the schedule stops recording
                # on that call, and the GPU is still behind the host, so kernels
                # of the last step would otherwise be cut off.
                torch.cuda.synchronize()
                state["wall_s"] = time.perf_counter() - mark["t"]
                state["done"] = True
            state["p"].step()

    wrap(MixOptimizer, "step", pre, post)
    run_cli(train_argv(need, "census"))

    p = state["p"]
    if p is None:
        raise SystemExit("census: the profiler never started (run too short)")
    torch.cuda.synchronize()
    p.stop()
    p.export_chrome_trace(state["path"])

    wall_ms = 1e3 * state.get("wall_s", 0.0) / max(active, 1)
    return census_report(state["path"], active, args.top, wall_ms)


# --------------------------------------------------------------------------
# census parse: trace -> tables (separate so a saved trace can be re-read)
# --------------------------------------------------------------------------
def census_report(trace_path: str, active: int, top: int, wall_ms: float) -> dict:
    ev = json.load(open(trace_path))["traceEvents"]
    kernels = [e for e in ev if e.get("cat") == "kernel"]
    runtime = [e for e in ev if e.get("cat") == "cuda_runtime" or e.get("cat") == "cuda_driver"]
    memcpy = [e for e in ev if e.get("cat") in ("gpu_memcpy", "gpu_memset")]
    steps = max(1, active)

    by_count = Counter(e["name"] for e in kernels)
    by_time = defaultdict(float)
    by_mean = defaultdict(list)
    for e in kernels:
        by_time[e["name"]] += e["dur"]
        by_mean[e["name"]].append(e["dur"])

    per_stream = defaultdict(lambda: [0, 0.0])
    for e in kernels:
        s = e.get("args", {}).get("stream", -1)
        per_stream[s][0] += 1
        per_stream[s][1] += e["dur"]

    # cuBLAS/cutlass launches go through the DRIVER api (`cuLaunchKernel`), which
    # is why a runtime-only count under-reports by the GEMM count (207/step here).
    launches = Counter(
        e["name"] for e in runtime if "Launch" in e["name"] and e["name"].startswith(("cuda", "cu"))
    )
    host_sync = Counter(
        e["name"] for e in runtime if e["name"].startswith(("cudaDeviceSync", "cudaStreamSync"))
    )
    mem = defaultdict(lambda: [0, 0.0])
    for e in memcpy:
        mem[e["name"]][0] += 1
        mem[e["name"]][1] += e["dur"]

    # gaps on the busiest stream
    dom = max(per_stream, key=lambda s: per_stream[s][1]) if per_stream else -1
    ks = sorted(
        (e for e in kernels if e.get("args", {}).get("stream", -1) == dom),
        key=lambda e: e["ts"],
    )
    gaps = []
    for a, b in zip(ks, ks[1:], strict=False):
        g = b["ts"] - (a["ts"] + a["dur"])
        if g > 0:
            gaps.append((g, a["name"], b["name"]))
    gap_total = sum(g[0] for g in gaps)
    gap_big = [g for g in gaps if g[0] > 50]
    span = (ks[-1]["ts"] + ks[-1]["dur"]) - ks[0]["ts"] if ks else 0.0

    # Dispatcher attribution: every kernel carries the `External id` of the CPU op
    # that issued it, so the join is exact — no parent/child double count, which
    # `key_averages` has (`aten::mm` under `aten::matmul` counted twice). A CPU op
    # that launches N kernels is charged all N; that is the intent here.
    by_ext = defaultdict(list)
    for e in ev:
        if e.get("cat") != "cpu_op":
            continue
        x = e.get("args", {}).get("External id")
        if x is not None:
            by_ext[x].append(e)
    op_kernels = defaultdict(lambda: [0, 0.0])
    for ke in kernels:
        ops = by_ext.get(ke.get("args", {}).get("External id"))
        if not ops:
            continue
        op = min(ops, key=lambda e: e.get("dur", 0))  # innermost op with that id
        op_kernels[op["name"]][0] += 1
        op_kernels[op["name"]][1] += ke["dur"]
    disp = sorted(
        ((k, c, us / 1000.0 / steps) for k, (c, us) in op_kernels.items()),
        key=lambda r: -r[2],
    )

    total_k = sum(by_count.values())
    total_t = sum(by_time.values())
    out = {
        "mode": "census",
        "warm": 0,
        "active": active,
        "wall_profiled_ms_per_step": wall_ms,
        "kernels_per_step": total_k / steps,
        "launches_per_step": sum(launches.values()) / steps,
        "launch_mix": {k: v / steps for k, v in launches.most_common(12)},
        "host_sync_calls_per_step": {k: v / steps for k, v in host_sync.most_common(6)},
        "gpu_ms_per_step": total_t / 1000.0 / steps,
        "streams": {
            str(k): {"kernels_per_step": v[0] / steps, "ms_per_step": v[1] / 1000.0 / steps}
            for k, v in per_stream.items()
        },
        "dominant_stream": dom,
        "gap_ms_per_step": gap_total / 1000.0 / steps,
        "gaps_over_50us_per_step": len(gap_big) / steps,
        "gap_share_of_span": gap_total / span if span else 0.0,
        "top_gaps": [
            {"ms": g / 1000.0, "after": a, "before": b}
            for g, a, b in sorted(gaps, reverse=True)[:10]
        ],
        "memcpy": {
            k: {"per_step": v[0] / steps, "ms_per_step": v[1] / 1000.0 / steps}
            for k, v in mem.items()
        },
        "by_count": [
            {
                "kernel": k,
                "per_step": c / steps,
                "ms_per_step": by_time[k] / 1000.0 / steps,
                "mean_us": by_time[k] / c,
            }
            for k, c in by_count.most_common(top)
        ],
        "by_time": [
            {
                "kernel": k,
                "ms_per_step": v / 1000.0 / steps,
                "per_step": by_count[k] / steps,
                "share": v / total_t,
                "mean_us": v / by_count[k],
            }
            for k, v in sorted(by_time.items(), key=lambda kv: -kv[1])[:top]
        ],
        "dispatchers": [
            {"op": k, "per_step": c / steps, "cuda_ms_per_step": ms} for k, c, ms in disp[:top]
        ],
    }

    print(f"\n=== census: {active} recorded steps (profiled)")
    print(
        f"kernels/step {out['kernels_per_step']:.1f}  launches/step "
        f"{out['launches_per_step']:.1f}  GPU busy {out['gpu_ms_per_step']:.2f} ms/step  "
        f"profiled wall {out['wall_profiled_ms_per_step']:.2f} ms/step"
    )
    print(f"streams: {json.dumps(out['streams'])}")
    print(
        f"gaps on stream {dom}: {out['gap_ms_per_step']:.2f} ms/step "
        f"({100 * out['gap_share_of_span']:.2f}% of span), "
        f"{out['gaps_over_50us_per_step']:.1f} gaps>50us/step"
    )
    print(f"memcpy: {json.dumps(out['memcpy'])}")
    print(f"launch mix: {json.dumps(out['launch_mix'])}")
    print("\n-- top by count (per step) --")
    print(f"{'kernel':<58} {'n':>7} {'ms':>8} {'us/ea':>8}")
    for r in out["by_count"]:
        print(
            f"{r['kernel'][:58]:<58} {r['per_step']:7.1f} "
            f"{r['ms_per_step']:8.3f} {r['mean_us']:8.2f}"
        )
    print("\n-- top by CUDA time (ms/step) --")
    print(f"{'kernel':<58} {'ms':>8} {'n':>7} {'share':>7} {'us/ea':>8}")
    for r in out["by_time"]:
        print(
            f"{r['kernel'][:58]:<58} {r['ms_per_step']:8.3f} {r['per_step']:7.1f} "
            f"{100 * r['share']:6.2f}% {r['mean_us']:8.2f}"
        )
    print("\n-- top dispatchers (op -> CUDA ms/step, per step) --")
    for r in out["dispatchers"]:
        print(f"{r['op'][:62]:<62} {r['per_step']:7.1f} {r['cuda_ms_per_step']:8.3f}")
    return out


# --------------------------------------------------------------------------
# sync: catch every host read
# --------------------------------------------------------------------------
def mode_sync(args) -> dict:
    from dormouse.train.optim import MixOptimizer

    rec = {"on": False, "items": []}
    real = warnings.showwarning

    def hook(message, category, filename, lineno, file=None, line=None):
        if rec["on"]:
            frames = [
                f"{Path(f.filename).name}:{f.lineno}:{f.name}"
                for f in traceback.extract_stack()[:-1]
                if "site-packages/torch" not in f.filename
            ]
            rec["items"].append({"msg": str(message)[:160], "at": frames[-3:]})
        return real(message, category, filename, lineno, file, line)

    warnings.showwarning = hook
    torch.cuda.set_sync_debug_mode("warn")

    state = {"n": 0}

    def pre():
        pass

    def post():
        state["n"] += 1
        rec["on"] = args.measure_from <= state["n"] < args.measure_from + 3
        # `warnings` dedupes by (message, category, module, lineno) by default:
        # without "always" the second and later syncs at a fixed site are never
        # delivered, which reads as a false zero.
        if rec["on"]:
            warnings.simplefilter("always")

    wrap(MixOptimizer, "step", pre, post)
    run_cli(train_argv(args.measure_from + 4, "sync", log_every=1000))
    torch.cuda.set_sync_debug_mode("default")
    warnings.showwarning = real

    counted = Counter(i["msg"] for i in rec["items"])
    out = {
        "mode": "sync",
        "steps_measured": 3,
        "count": len(rec["items"]),
        "by_msg": counted.most_common(),
        "sites": rec["items"][:12],
    }
    print(f"\n=== sync: {len(rec['items'])} synchronizing calls in 3 measured steps")
    for m, c in counted.most_common():
        print(f"  {c:4d}  {m}")
    for i in rec["items"][:6]:
        print(f"    at {i['at']}")
    return out


# --------------------------------------------------------------------------
# loader: is the 0.114 ms CPU read on the critical path?
# --------------------------------------------------------------------------
def mode_loader(args) -> dict:
    from dormouse.train.mixture import MixtureStream
    from dormouse.train.optim import MixOptimizer

    plan = [(0, 15), (0.005, 15), (0.020, 15), (0.040, 15)]
    step = [0]
    loaders: list[float] = []
    starts: list[float] = []
    t = {}

    def delay_for(n: int) -> float:
        acc = 0
        for d, k in plan:
            acc += k
            if n <= acc:
                return d
        return 0.0

    def loader_pre():
        t["l0"] = time.perf_counter()

    def loader_post():
        d = delay_for(step[0] + 1)
        if d:
            time.sleep(d)
        loaders.append(1e3 * (time.perf_counter() - t["l0"]))

    def opt_post():
        step[0] += 1
        starts.append(time.perf_counter())

    wrap(MixtureStream, "next_batch", loader_pre, loader_post)
    wrap(MixOptimizer, "step", None, opt_post)
    run_cli(train_argv(sum(k for _, k in plan) + 2, "loader"))

    periods = [1e3 * (starts[i + 1] - starts[i]) for i in range(len(starts) - 1)]
    arms = []
    lo = 0
    for d, k in plan:
        seg = periods[lo : lo + k - 1]
        ld = loaders[lo : lo + k]
        arms.append(
            {
                "extra_ms": 1e3 * d,
                "period_med_ms": st.median(seg),
                "period_p95_ms": pct(seg, 95),
                "loader_med_ms": st.median(ld),
            }
        )
        lo += k
    base = arms[0]["period_med_ms"]
    for a in arms:
        a["delta_ms"] = a["period_med_ms"] - base
    out = {"mode": "loader", "arms": arms, "steps": step[0]}
    print(f"\n=== loader injection ({step[0]} steps)")
    print(
        f"{'extra sleep':>12} {'loader med':>11} {'period med':>11} {'period p95':>11} {'delta':>8}"
    )
    for a in arms:
        print(
            f"{a['extra_ms']:12.1f} {a['loader_med_ms']:11.3f} {a['period_med_ms']:11.2f} "
            f"{a['period_p95_ms']:11.2f} {a['delta_ms']:+8.2f}"
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["census", "phases", "sync", "loader"], required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--skip", type=int, default=5, help="phases: warm steps dropped")
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--active", type=int, default=20)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--trace", default="/tmp/opencode/hvg/census.json")
    ap.add_argument("--json", default="")
    ap.add_argument("--measure-from", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    Path(a.trace).parent.mkdir(parents=True, exist_ok=True)

    out = {"census": mode_census, "phases": mode_phases, "sync": mode_sync, "loader": mode_loader}[
        a.mode
    ](a)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"\njson -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
