"""Launch anatomy of ONE medium training step: bf16 vs ternary, b8 x s512.

    uv run --extra gpu python scripts/launch_profile.py

**The number this prints decides the next move.** The workload is launch-bound:
the same model and the same kernels run at 11 538 B/s at b8 x s512 and 40 960 B/s
at b24 x s1024, because the per-step KERNEL COUNT does not shrink with the batch
and the small shape pays it over 4 096 bytes. Before choosing between
`torch.compile` and a full-step CUDA graph, this profile counts what a step
actually enqueues and how much of the wall time the card is busy.

**What is measured, per arm** (`medium` exactly as `configs/medium.toml`
resolves; the ternary arm additionally sets `factor_qat=ternary`,
`dense_qat=ternary`, `act_quant=int4`, alpha 1.0 — the shape the 2026-10-21 QAT
bench ran):

* **launches/step** — `cudaLaunchKernel` events, the host-side count that is the
  launch-bound cost. Executed CUDA kernels are counted separately; the two
  disagree only for a launch that produced no kernel, and the difference is
  printed rather than hidden;
* **GPU busy vs wall** — the union of every kernel's `time_range` (not the sum:
  overlapping streams would double-count) against the wall of the same step.
  A launch-bound step shows a low busy share;
* **a phase split** — kda/fla, moe, ns, qat refresh, clip, optimizer,
  norms/elementwise — by `record_function` seams installed for the profiled
  step only (`SparseMoE.forward`, `KDAAttention.forward`, `tsct.qat_values`,
  plus the script's own clip/optimizer calls). Kernels with no seam fall back to
  a name rule, so an arm that does not route through a seam is still counted;
* **top 15 kernels by count and by time**, aggregated by kernel name.

**Protocol.** One warm step, then one profiled step (`torch.profiler`,
`ProfilerActivity.CPU + CUDA`, `record_shapes=False`). The wall column is the
median of three UNPROFILED steps, because the profiler's own CPU cost inflates
a launch-bound wall; the trace's busy time is paired with that honest wall. The
step is the trainer's: per-step cache invalidation, forward, backward, the fused
grad clip, `MixOptimizer.step()`, then the next step's invalidation.

**Preflight, and it is the whole reason this script is safe to run:** one GPU
process at a time (AGENTS §1.5) — `pgrep -f "dormouse trai[n]"` must be empty
and `nvidia-smi` must report under 1 GiB in use. On either, it prints a SKIP
banner and exits 0: nothing was measured, so there is no number to quote.

**Not run by this task's author**: the card is busy. Every number this file
will print is a tool waiting for a free card.
"""

from __future__ import annotations

import dataclasses
import functools
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from dormouse.config import load_run
from dormouse.model import DormouseModel
from dormouse.model.experts import invalidate_banks_all
from dormouse.model.qat_cache import invalidate_qat_cache_all
from dormouse.train.optim import MixOptimizer, clip_grads

# `scripts/` is not a package (a test loads this file by path): the lane guard is
# a sibling file, so its directory goes on the path, APPENDED not inserted (a
# script name at the front of `sys.path` would shadow a package of that name).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight as _lane_preflight  # noqa: E402
from _lane import skip  # noqa: E402

# a test pins a 0-arg `preflight`: this probe only ever measures the card.
preflight = functools.partial(_lane_preflight, "cuda")

PRESET = "medium"
BATCH, SEQ = 8, 512
WARM, EAGER_REPS = 1, 3
DEV = "cuda"
BYTES_PER_STEP = BATCH * SEQ
CUDA = torch.autograd.DeviceType.CUDA
CPU = torch.autograd.DeviceType.CPU

#: The phase labels the `record_function` seams carry. The phase table uses
#: exactly these, so a new seam cannot invent a row the reader cannot compare.
PHASES = (
    "kda/fla",
    "moe",
    "ns",
    "qat refresh",
    "clip",
    "optimizer",
    "norms/elementwise",
    "other",
)

#: Kernel-name fallbacks, first match wins. Seam annotations win over these;
#: they only cover kernels whose launch site has no seam (aten elementwise,
#: cuBLAS, and the fused Triton kernels).
_NAME_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ns", re.compile(r"_ns_chain|newton|zeropower|orthogonal|gram_", re.I)),
    (
        "kda/fla",
        re.compile(r"chunk_kda|wy_repr|fwd_h|bwd_h|l2norm|gated_delta|chunk_|fla_", re.I),
    ),
    (
        "moe",
        re.compile(r"_glu_kernel|grouped_gemm|_wgrad|_combine_bwd|_moe_|moe_", re.I),
    ),
    ("clip", re.compile(r"foreach_norm|foreach_mul|clip", re.I)),
    ("optimizer", re.compile(r"multi_tensor_apply|adam|muon|sgd|addcmul|addcdiv|lerp", re.I)),
    (
        "norms/elementwise",
        re.compile(
            r"norm|softmax|reduce|elementwise|vectorized|copy|fill|cat_|index|gather|"
            r"scatter|topk|sort|embedding|silu|sigmoid|exp|mul|add",
            re.I,
        ),
    ),
)


def _overrides(arm: str) -> dict:
    """The only difference between the two arms: the QAT rung."""
    if arm == "bf16":
        return {"factor_qat": "none", "dense_qat": "none", "act_quant": "none"}
    return {
        "factor_qat": "ternary",
        "dense_qat": "ternary",
        "act_quant": "int4",
        "factor_qat_alpha": 1.0,
    }


def _model(run, arm: str, state: dict | None) -> DormouseModel:
    model = DormouseModel(dataclasses.replace(run.model, **_overrides(arm)), device=DEV)
    if state:
        model.load_state_dict(state)
    return model


def _optimizer(model: DormouseModel, train) -> MixOptimizer:
    """The trainer's own construction (`train/loop.py:576`), field for field."""
    return MixOptimizer(
        model,
        lr=train.lr,
        wd=train.wd,
        opt=train.opt,
        ns_steps=train.ns_steps,
        ns_kind=train.ns_kind,
        colrow_3d=train.muon_colrow_3d,
        muon_state_dtype=train.muon_state_dtype,
        ns_dtype=train.ns_dtype,
        muon_variant=train.muon_variant,
        muon_sf=train.muon_sf,
        muon_sf_wd=train.muon_sf_wd,
    )


def _install_seams() -> list[tuple[object, str, object]]:
    """`record_function` around the three model seams + the script's two stages.

    The seams are process-global for the profiled step only; `_remove_seams`
    restores every original in reverse order. `qat_values` is the ONE quantizer
    every refresh path lazily imports, so the annotation covers the bank's
    stacked projection and the per-tensor one alike.
    """
    from dormouse.model import kda, tsct
    from dormouse.model.experts import SparseMoE

    saved: list[tuple[object, str, object]] = []

    def wrap(owner, attr: str, label: str) -> None:
        orig = getattr(owner, attr)

        @functools.wraps(orig)
        def wrapped(*args, **kwargs):
            with torch.profiler.record_function(label):
                return orig(*args, **kwargs)

        setattr(owner, attr, wrapped)
        saved.append((owner, attr, orig))

    wrap(SparseMoE, "forward", "moe")
    wrap(kda.KDAAttention, "forward", "kda/fla")
    wrap(tsct, "qat_values", "qat refresh")
    return saved


def _remove_seams(saved: list[tuple[object, str, object]]) -> None:
    for owner, attr, orig in reversed(saved):
        setattr(owner, attr, orig)


def _step(model, opt, train, x: torch.Tensor, y: torch.Tensor):
    """One trainer step: invalidate (step boundary), fwd, bwd, clip, optimize."""
    model.zero_grad(set_to_none=True)
    invalidate_qat_cache_all(model)
    invalidate_banks_all(model)
    _, loss, counters = model(x, y)
    loss.backward()
    if train.grad_clip > 0:
        with torch.profiler.record_function("clip"):
            clip_grads(model.parameters(), train.grad_clip)
    with torch.profiler.record_function("optimizer"):
        opt.step()
    invalidate_banks_all(model)
    invalidate_qat_cache_all(model)
    return counters


def _wall_step(model, opt, train, x, y) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _step(model, opt, train, x, y)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def _annotation(evt) -> str | None:
    """The nearest `record_function` label above a CPU event, or None."""
    node = evt
    while node is not None:
        if node.name in PHASES:
            return node.name
        if getattr(node, "is_user_annotation", False):
            return node.name if node.name in PHASES else None
        node = getattr(node, "cpu_parent", None)
    return None


def phase_of(kernel: str, annotation: str | None) -> str:
    """The phase a kernel belongs to: seam first, then the name fallback."""
    if annotation in PHASES:
        return annotation
    for phase, rule in _NAME_RULES:
        if rule.search(kernel):
            return phase
    return "other"


def kernel_rows(prof) -> list[tuple[str, float, str]]:
    """`(name, duration_us, phase)` for every executed kernel, once each.

    A CUDA kernel event is linked to exactly ONE CPU event by the profiler's
    correlation id, and that is where `cpu_parent` (the seam chain) is
    available. The raw device-event count is checked against this list by the
    caller; a mismatch is printed, not hidden.
    """
    rows: list[tuple[str, float, str]] = []
    for evt in prof.events():
        if evt.device_type != CPU:
            continue
        ann = _annotation(evt)
        for k in evt.kernels:
            rows.append((k.name, float(k.duration), phase_of(k.name, ann)))
    return rows


def union_us(intervals: list[tuple[float, float]]) -> float:
    """The length of the UNION of `[start, end)` spans, in microseconds."""
    spans = sorted((s, e) for s, e in intervals if e > s)
    if not spans:
        return 0.0
    total = 0.0
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s)


def _fmt_kernel(name: str, width: int = 82) -> str:
    return name if len(name) <= width else name[: width - 1] + "…"


def _profile(model, opt, train, x, y) -> tuple[float, object, object]:
    """Profile ONE step; return `(wall ms, profiler, counters)`."""
    saved = _install_seams()
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            counters = _step(model, opt, train, x, y)
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) * 1e3
    finally:
        _remove_seams(saved)
    return wall, prof, counters


def arm(name: str, run, state: dict) -> None:
    """One arm: warm, three eager walls, then the profiled step and its table."""
    torch.manual_seed(0)
    model = _model(run, name, state)
    opt = _optimizer(model, run.train)
    x = torch.randint(0, 256, (BATCH, SEQ), device=DEV)
    for _ in range(WARM):
        _step(model, opt, run.train, x, x)
    walls = [_wall_step(model, opt, run.train, x, x) for _ in range(EAGER_REPS)]
    torch.cuda.reset_peak_memory_stats()
    wall, prof, counters = _profile(model, opt, run.train, x, x)
    peak = torch.cuda.max_memory_allocated() / 2**30

    events = prof.events()
    launches = sum(1 for e in events if e.name == "cudaLaunchKernel")
    device = [e for e in events if e.device_type == CUDA]
    rows = kernel_rows(prof)
    busy_us = union_us([(e.time_range.start, e.time_range.end) for e in device])
    eager_ms = statistics.median(walls)

    phases = {p: [0, 0.0] for p in PHASES}
    for _, dur, ph in rows:
        phases[ph][0] += 1
        phases[ph][1] += dur

    total_kernels = len(rows) or len(device)
    total_us = sum(d for _, d, _ in rows)

    print(f"\n=== {name} — {PRESET}, b{BATCH} x s{SEQ} = {BYTES_PER_STEP} B/step ===")
    print(
        f"eager wall (median of {EAGER_REPS}) {eager_ms:8.1f} ms   "
        f"profiled wall {wall:8.1f} ms   peak {peak:5.2f} GiB"
    )
    print(
        f"launches/step (cudaLaunchKernel) {launches:6d}   executed kernels "
        f"{total_kernels:6d}   CUDA kernel events {len(device):6d}"
    )
    if total_kernels != len(device):
        print(
            f"NOTE: {total_kernels} kernels were linked to a CPU parent but "
            f"{len(device)} CUDA events were recorded; phases are read off the "
            f"linked rows, the totals off the rows."
        )
    print(
        f"GPU busy {busy_us / 1e3:7.1f} ms of a {eager_ms:7.1f} ms step "
        f"= {100.0 * busy_us / 1e3 / eager_ms:5.1f}% busy, "
        f"{100.0 * (1.0 - busy_us / 1e3 / eager_ms):5.1f}% idle"
    )
    print(f"kernel time summed {total_us / 1e3:7.1f} ms (union is the row above)")

    print(f"\n{'phase':16s} {'kernels':>8s} {'gpu ms':>8s} {'% kernels':>10s} {'% gpu':>7s}")
    for ph in PHASES:
        n, us = phases[ph]
        print(
            f"{ph:16s} {n:8d} {us / 1e3:8.1f} "
            f"{100.0 * n / max(total_kernels, 1):9.1f}% "
            f"{100.0 * us / max(total_us, 1.0):6.1f}%"
        )

    by_count: dict[str, list[float]] = {}
    by_time: dict[str, float] = {}
    for kname, dur, _ in rows:
        by_count.setdefault(kname, []).append(dur)
        by_time[kname] = by_time.get(kname, 0.0) + dur

    print("\ntop 15 kernels by count:")
    for kname, durs in sorted(by_count.items(), key=lambda kv: -len(kv[1]))[:15]:
        print(f"  {len(durs):6d}  {sum(durs) / 1e3:8.1f} ms  {_fmt_kernel(kname)}")
    print("\ntop 15 kernels by GPU time:")
    for kname, us in sorted(by_time.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {us / 1e3:8.1f} ms  {len(by_count[kname]):6d}  {_fmt_kernel(kname)}")

    # The forward's own counters: the QAT arm must show 3 stacked calls / 768
    # tensors at medium (the 2026-10-21 fix), and the profiled step's drop count
    # is the padded arm's routing context — printed, not used as a verdict.
    drain = counters.drain() if hasattr(counters, "drain") else dict(counters)
    print(
        f"\ncounters: qat_refresh_calls={drain.get('qat_refresh_calls', 0)} "
        f"qat_refresh_tensors={drain.get('qat_refresh_tensors', 0)} "
        f"qat_reuse={drain.get('qat_reuse', 0)} "
        f"moe_dropped_assignments={drain.get('moe_dropped_assignments', 0)}"
    )
    bps = BYTES_PER_STEP / (eager_ms / 1e3)
    us_per_b = eager_ms * 1e3 / BYTES_PER_STEP
    us_per_kb = us_per_b * 1024
    print(
        f"VERDICT {name}: launches/step={launches}  "
        f"GPU busy={100.0 * busy_us / 1e3 / eager_ms:.1f}%  "
        f"({bps:.0f} B/s, {us_per_b:.1f} us/B, {us_per_kb:.0f} us/KB)"
    )

    del model, opt
    torch.cuda.empty_cache()


def main() -> int:
    why = preflight()
    if why is None:
        # This probe's own ceiling is 1 GiB, tighter than the lane guard's 2.0:
        # a card already holding a second model is contended however idle it looks.
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if smi and int(smi.split()[0]) >= 1024:
            why = f"the card already holds {smi.split()[0]} MiB (this probe's ceiling is 1024)"
    if why is not None:
        return skip(why)
    run = load_run(PRESET)
    torch.manual_seed(0)
    base = _model(run, "bf16", None)
    state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    del base
    torch.cuda.empty_cache()
    print(
        f"\nshape: {PRESET} d={run.model.d_model} E={run.model.n_experts} "
        f"max_iter={run.model.max_iter}, b{BATCH} x s{SEQ}; "
        f"step = fwd + bwd + clip + optimizer (the trainer's full step)"
    )
    for name in ("bf16", "ternary"):
        arm(name, run, state)
    print(
        "\nThe verdict is per arm; the comparison is launches/step and GPU busy %. "
        "A launch-bound step keeps a low busy share while the launch count is flat."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
