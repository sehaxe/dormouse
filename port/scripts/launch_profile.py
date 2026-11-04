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

**The CPU twin — a shape-diffable launch counter.** With no `--device cuda`
this script runs ONE forward + backward + optimizer step of a CPU model under a
`TorchDispatchMode` and prints the launch GROUPS: `(aten op, module)` pairs,
where the module is the one whose forward (or backward) is executing — the
`nn.Module` hook stack. An op with no module scope is resolved from the output's
`grad_fn`, then from the nearest mapped ancestor in the arguments' `grad_fn`
chains. Metadata-only ops (`view`,
`expand`, `transpose`, `detach`, …) enqueue no kernel and are excluded from the
launch count, so the rows are a CPU *proxy* for what the card would launch — one
aten op is one kernel on the eager GPU path, which is the shape of the problem
this file exists to measure. `--json PATH` writes the full group table and
`--diff A.json B.json` prints the per-group delta sorted by absolute change, so
two shapes (b8×s1024 vs b1×s8192) are compared without a profiler or a card.

    CUDA_VISIBLE_DEVICES="" uv run python scripts/launch_profile.py \
        --preset tiny --batch 2 --seq 64 --set use_csa=true --json a.json
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
import json
import re
import statistics
import subprocess
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

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


# ---------------------------------------------------------------------------
# The CPU twin: a TorchDispatchMode launch counter and its JSON/diff protocol
# ---------------------------------------------------------------------------

#: Ops that only move metadata: they enqueue no kernel on the eager CUDA path
#: (the same rule `tests/test_muon_launch.py` uses for `_foreach_*`). A composite
#: that COPIES internally is not in this set — it is counted at the decomposed
#: op that does the copy.
METADATA_OPS = frozenset(
    {
        "alias",
        "as_strided",
        "chunk",
        "detach",
        "detach_",
        "expand",
        "expand_as",
        "imag",
        "lift_fresh",
        "narrow",
        "permute",
        "ravel",
        "real",
        "reshape",
        "resolve_conj",
        "resolve_neg",
        "select",
        "slice",
        "split",
        "split_with_sizes",
        "squeeze",
        "squeeze_",
        "t",
        "transpose",
        "transpose_",
        "unbind",
        "unsqueeze",
        "unsqueeze_",
        "view",
        "view_as",
        "_unsafe_view",
    }
)


#: Device→host reads: not a kernel launch, and the one class the step path must
#: not have (AGENTS §1.3). Counted separately so a sync cannot hide in a launch
#: row, and so the optimizer's `.item()` reads are visible in the JSON.
SYNC_OPS = frozenset({"aten::_local_scalar_dense"})


def aten_name(func) -> str:
    """`aten::addmm` for an `OpOverload`, else `str(func)` — the group key."""
    schema = getattr(func, "_schema", None)
    name = getattr(schema, "name", None)
    return name if name else str(func)


def is_metadata(op: str) -> bool:
    """True when `op` is a metadata-only aten op (no kernel launch)."""
    return op.rsplit("::", 1)[-1] in METADATA_OPS


def _tensors(value):
    """Every tensor in a hook output/arg, including one nesting level of tuples."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                yield item


class DispatchCounter(TorchDispatchMode):
    """Count one step's aten dispatches, grouped by `(op, module)`.

    Module attribution, in order:

    1. the `nn.Module` call stack: the module whose forward (or backward) is
       executing. The hooks fire around the module's own ops, so this is the
       exact scope — an op's OUTPUT node has no `grad_fn` until the module
       returns, and its arguments' nodes belong to the module BELOW it, so a
       graph-only rule is off by one at a module boundary;
    2. the OUTPUT's `grad_fn`, when the op runs outside any module scope but a
       forward hook recorded that node (a custom Function's backward, a head
       called directly);
    3. the nearest recorded node in the ARGUMENTS' `grad_fn` chains (an op whose
       module produced no scope — the generic fallback);
    4. `<forward>` / `<backward>` / `<optimizer>` — the stage the op ran in
       (the loss head, the optimizer; no module owns them).

    Hooks are installed for the counted step only. They add ~0.4% metadata
    dispatches (they `detach()` bookkeeping tensors); non-metadata ops — the
    number this table reports — are unaffected (measured on tiny, 1283 vs 1288).
    """

    def __init__(self) -> None:
        self.dispatches: Counter[tuple[str, str]] = Counter()
        self.launches: Counter[tuple[str, str]] = Counter()
        self.stages: Counter[str] = Counter()
        self.syncs: Counter[tuple[str, str]] = Counter()
        self.stage = "step"
        self._stack: list[str] = []
        self._node_module: dict[object, str] = {}
        self._hooks: list[object] = []
        self._warnings = None

    # -- hooks ------------------------------------------------------------
    def attach(self, model: torch.nn.Module) -> None:
        self._warnings = warnings.catch_warnings()
        self._warnings.__enter__()
        warnings.filterwarnings("ignore", message="Full backward hook is firing.*")
        for name, module in model.named_modules():
            label = name or "<root>"
            self._hooks.append(module.register_forward_pre_hook(self._pre(label)))
            self._hooks.append(module.register_forward_hook(self._post(label)))
            self._hooks.append(module.register_full_backward_pre_hook(self._bpre(label)))
            self._hooks.append(module.register_full_backward_hook(self._bpost(label)))
        self._stack.clear()

    def detach(self) -> None:
        for hook in reversed(self._hooks):
            hook.remove()
        self._hooks.clear()
        if self._warnings is not None:
            self._warnings.__exit__(None, None, None)
            self._warnings = None

    @contextlib.contextmanager
    def observe(self, model: torch.nn.Module):
        self.attach(model)
        try:
            with self:
                yield self
        finally:
            self.detach()

    def _pre(self, label: str):
        def hook(_module, _inputs):
            self._stack.append(label)

        return hook

    def _post(self, label: str):
        def hook(_module, _inputs, output):
            self._pop(label)
            for tensor in _tensors(output):
                if tensor.grad_fn is not None:
                    # `setdefault`: the innermost module to produce the tensor
                    # owns the node; an enclosing container must not overwrite it.
                    self._node_module.setdefault(tensor.grad_fn, label)

        return hook

    def _bpre(self, label: str):
        def hook(_module, _grad_inputs):
            self._stack.append(label)

        return hook

    def _bpost(self, label: str):
        def hook(_module, _grad_inputs, _grad_outputs):
            self._pop(label)

        return hook

    def _pop(self, label: str) -> None:
        if self._stack and self._stack[-1] == label:
            self._stack.pop()
        elif label in self._stack:  # a hook that fired out of order; stay balanced
            self._stack.remove(label)

    # -- resolution -------------------------------------------------------
    def _walk_grad_fn(self, node):
        """Nearest module along a `grad_fn` closure (BFS, bounded)."""
        frontier = [node] if node is not None else []
        seen: set[object] = set()
        for _ in range(12):
            nxt = []
            for current in frontier:
                if current is None or current in seen:
                    continue
                seen.add(current)
                module = self._node_module.get(current)
                if module is not None:
                    return module
                nxt.extend(parent for parent, _ in current.next_functions if parent is not None)
            frontier = nxt
            if not frontier:
                break
        return None

    def _resolve(self, args, out) -> str:
        if self._stack:
            return self._stack[-1]
        for tensor in _tensors(out):
            if tensor.grad_fn is not None:
                module = self._node_module.get(tensor.grad_fn)
                if module is not None:
                    return module
        for arg in args:
            if isinstance(arg, torch.Tensor):
                module = self._walk_grad_fn(arg.grad_fn)
                if module is not None:
                    return module
        return f"<{self.stage}>"

    # -- the counter ------------------------------------------------------
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        op = aten_name(func)
        if self._stack:
            # Record the node while the module scope is known; a later backward
            # op can then be resolved by its arguments' `grad_fn` chain even if
            # the backward hook does not fire (a custom Function).
            for tensor in _tensors(out):
                if tensor.grad_fn is not None:
                    self._node_module.setdefault(tensor.grad_fn, self._stack[-1])
        module = self._resolve(args, out)
        self.dispatches[(op, module)] += 1
        if op in SYNC_OPS:
            self.syncs[(op, module)] += 1
        elif not is_metadata(op):
            self.launches[(op, module)] += 1
            self.stages[self.stage] += 1
        return out

    # -- reporting --------------------------------------------------------
    def rows(self, top: int | None = 25) -> list[tuple[str, str, int, int]]:
        rows = [
            (op, module, self.launches[(op, module)], self.dispatches[(op, module)])
            for op, module in set(self.launches) | set(self.dispatches)
        ]
        rows.sort(key=lambda row: (-row[2], -row[3], row[0], row[1]))
        return rows[:top] if top else rows


def _sorted_pairs(counter: Counter) -> list[tuple[tuple[str, str], int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))


def _summary(counter: DispatchCounter) -> dict:
    total = sum(counter.launches.values())
    rows = counter.rows(None)
    return {
        "syncs": [
            {"op": op, "module": module, "count": count}
            for (op, module), count in _sorted_pairs(counter.syncs)
        ],
        "groups": [
            {
                "op": op,
                "module": module,
                "launches": launches,
                "dispatches": dispatches,
                "share": (launches / total) if total else 0.0,
            }
            for op, module, launches, dispatches in rows
        ],
        "stages": dict(counter.stages),
    }


def dispatch_report(counter: DispatchCounter, top: int = 25) -> dict:
    """Print the table; return the JSON payload (the `--json` body)."""
    total = sum(counter.launches.values())
    dispatches = sum(counter.dispatches.values())
    syncs = sum(counter.syncs.values())
    print(
        f"\nlaunches/step (non-metadata aten ops) {total:6d}   "
        f"dispatches {dispatches:6d}   metadata {dispatches - total - syncs:6d}"
    )
    stages = "  ".join(f"{name}={n}" for name, n in sorted(counter.stages.items()))
    print(f"stages: {stages}")
    if counter.syncs:
        print(f"host syncs/step: {sum(counter.syncs.values())}  (excluded from launches)")
    print(f"\n{'rank':>4s} {'launches':>9s} {'share':>6s}  {'op':34s} module")
    for rank, (op, module, launches, _disp) in enumerate(counter.rows(top), 1):
        share = 100.0 * launches / max(total, 1)
        print(f"{rank:4d} {launches:9d} {share:5.1f}%  {op:34s} {module}")
    return _summary(counter)


def write_profile(path: str | Path, meta: dict, counter: DispatchCounter) -> None:
    meta = {
        **meta,
        "total_launches": sum(counter.launches.values()),
        "total_dispatches": sum(counter.dispatches.values()),
    }
    payload = {"meta": meta, **_summary(counter)}
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def diff_profiles(a_path: str | Path, b_path: str | Path, top: int | None = None) -> int:
    """Per-group delta `B − A`, sorted by absolute change; 0 on success."""
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())

    def keyed(doc: dict) -> dict[tuple[str, str], int]:
        return {(g["op"], g["module"]): int(g["launches"]) for g in doc.get("groups", [])}

    ka, kb = keyed(a), keyed(b)
    total_a = sum(ka.values())
    total_b = sum(kb.values())
    rows = []
    for key in set(ka) | set(kb):
        delta = kb.get(key, 0) - ka.get(key, 0)
        rows.append((delta, ka.get(key, 0), kb.get(key, 0), key[0], key[1]))
    rows.sort(key=lambda row: (-abs(row[0]), row[3], row[4]))

    def meta(doc, name):
        m = doc.get("meta", {})
        return (
            f"{name}: {m.get('preset', '?')} b{m.get('batch', '?')} s{m.get('seq', '?')} "
            f"{m.get('device', '?')} {m.get('dtype', '?')}"
        )

    print(f"{meta(a, 'A')}  ->  {meta(b, 'B')}")
    print(f"total launches {total_a} -> {total_b}  ({total_b - total_a:+d})")
    print(f"\n{'delta':>9s} {'A':>9s} {'B':>9s} {'shareB':>7s}  {'op':34s} module")
    for delta, na, nb, op, module in rows[:top] if top else rows:
        share = 100.0 * nb / max(total_b, 1)
        print(f"{delta:+9d} {na:9d} {nb:9d} {share:6.1f}%  {op:34s} {module}")
    return 0


def parse_set(items: list[str]) -> dict:
    """`["use_csa=true", "csa_chunk=8"]` -> `{"use_csa": True, ...}`."""

    def literal(raw: str):
        low = raw.lower()
        if low in ("true", "false"):
            return low == "true"
        if low in ("none", "null"):
            return None
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            return raw

    overrides: dict = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set {item!r}: expected KEY=VALUE")
        overrides[key] = literal(raw)
    return overrides


def _resolve_device(name: str) -> torch.device:
    """`auto`/`cpu` stay on the CPU; `cuda` is REFUSED when the card is taken.

    The dispatch counter is a CPU instrument by default on purpose: the owner's
    training run owns the card, and a counter that executes on CUDA is still a
    counter of HOST dispatches — the card buys nothing and the lane rule
    (AGENTS §1.5) is one GPU process at a time.
    """
    if name in ("auto", "cpu"):
        return torch.device("cpu")
    if name != "cuda":
        raise RuntimeError(f"--device {name}: use auto|cpu|cuda")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda but torch.cuda.is_available() is False (is "
            "CUDA_VISIBLE_DEVICES empty?). Escape: --device cpu, or a visible card."
        )
    why = _lane_preflight("cuda")
    if why is not None:
        raise RuntimeError(
            f"--device cuda and the card is not ours:\n{why}\n"
            "ONE GPU PROCESS AT A TIME (AGENTS 1.5). Escape: --device cpu."
        )
    return torch.device("cuda")


def _dispatch_step(model, opt, train, ids, counter: DispatchCounter | None):
    """One trainer-shaped step: fwd, bwd, clip, optimizer — the counted unit."""

    def stage(name: str) -> None:
        if counter is not None:
            counter.stage = name

    model.zero_grad(set_to_none=True)
    stage("forward")
    _logits, loss, counters = model(ids, targets=ids)
    assert loss is not None, "a training forward with targets must return a loss"
    stage("backward")
    loss.backward()
    stage("optimizer")
    if train.grad_clip > 0:
        clip_grads(model.parameters(), train.grad_clip)
    opt.step()
    stage("step")
    return counters


def dispatch_main(args: argparse.Namespace, device: torch.device) -> int:
    overrides = parse_set(args.set)
    run = load_run(args.preset, model_overrides=overrides, train_overrides={"seq_len": args.seq})
    torch.manual_seed(args.seed)
    model = DormouseModel(run.model, device=device).train()
    opt = _optimizer(model, run.train)
    ids = torch.randint(0, 256, (args.batch, args.seq), device=device)
    dtype = next(model.parameters()).dtype

    for _ in range(args.warm):
        _dispatch_step(model, opt, run.train, ids, None)

    counter = DispatchCounter()
    with counter.observe(model):
        _dispatch_step(model, opt, run.train, ids, counter)

    print(
        f"\n=== dispatch profile: {args.preset}, b{args.batch} x s{args.seq}, "
        f"{device.type}, {dtype} ==="
    )
    if overrides:
        print(f"overrides: {overrides}")
    dispatch_report(counter, top=args.top)
    if args.json_path:
        meta = {
            "preset": args.preset,
            "batch": args.batch,
            "seq": args.seq,
            "device": device.type,
            "dtype": str(dtype),
            "overrides": overrides,
            "warm": args.warm,
            "torch": torch.__version__,
            "total_launches": sum(counter.launches.values()),
            "total_dispatches": sum(counter.dispatches.values()),
            "metadata_ops": sorted(METADATA_OPS),
        }
        write_profile(args.json_path, meta, counter)
        print(f"\nwrote {args.json_path}")
    return 0


def cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json", dest="json_path", default=None)
    ap.add_argument("--diff", nargs=2, metavar=("A.json", "B.json"))
    ap.add_argument(
        "--gpu-profile",
        action="store_true",
        help="the two-arm card profile (needs the card; SKIPs when it is busy)",
    )
    args = ap.parse_args(argv)
    if args.diff:
        return diff_profiles(args.diff[0], args.diff[1], top=args.top)
    if args.gpu_profile:
        return main()
    try:
        device = _resolve_device(args.device)
    except RuntimeError as exc:
        print(f"\nREFUSED: {exc}\n", file=sys.stderr)
        return 2
    return dispatch_main(args, device)


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
    raise SystemExit(cli())
