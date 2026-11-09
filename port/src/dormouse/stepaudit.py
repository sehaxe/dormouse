"""Per-step audit: host reads, python loops, allocations — CPU-runnable.

Why this exists next to `scripts/sync_audit.py`: that script counted the four
D2H reads and proved the drain was the only one on record. It cannot say what a
step ALLOCATES, which python loops it runs over modules, or whether a non-report
step touches the host at all. This module answers those three, and the trainer
brackets each step so the answer is per step instead of per run.

The instrument is a no-op unless [`install`] arms it:

* reads — `Tensor.item/.tolist/.cpu/__bool__/__float__/__int__` are wrapped and
  blamed on the first non-torch frame. A read on a non-CPU tensor is a DEVICE
  sync (the 0-sync rule, ADR-0018 rule 2); a read on a CPU tensor is a host
  read, which a GPU step must not take either. Both are counted, separately.
* loops — [`loop`] is the small context manager the hot loops wrap themselves
  in; it counts instances, iterations and the distinct modules a loop iterated.
  No `sys.settrace`: an uninstrumented loop is found by the static census
  ([`census`]), which the gate compares against the annotated set.
* allocations — `torch.empty/zeros/ones/full` (+`_like`, `Tensor.new_*`,
  `torch.cat`, `torch.stack`) are counted with their requested bytes. Peak
  allocated bytes are the CUDA allocator's high-water mark where CUDA exists;
  on CPU only the requested-bytes total is meaningful, and it is reported as
  such rather than faked.

`scripts/step_audit.py` is the CLI: `--tiny` runs the CPU preset, `--gpu` runs
the real launcher under the same patches.
"""

from __future__ import annotations

import ast
import contextlib
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch

#: The tensor methods that cross to the host. `float(t)`/`int(t)` are the same
#: read as `.item()` and the first audit missed them (a `float(cuda_scalar)` is
#: a sync even though it never spells `.item`).
READ_METHODS = ("item", "tolist", "cpu", "__bool__", "__float__", "__int__")

#: Creators whose OUTPUT is an allocation. `cat`/`stack` are here because they
#: are the step's largest implicit allocations (`ctrl`, the AttnRes stack).
_ALLOC_FUNCS = (
    "empty",
    "empty_like",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "cat",
    "stack",
)

#: `Tensor.new_*` creators — the trainer's loss accumulators use `new_zeros`.
_NEW_FUNCS = ("new_empty", "new_zeros", "new_ones", "new_full")

#: Files the static census walks. This is the TRAINING STEP's path, not the
#: whole port: the decode/eval/load paths are separate programs.
CENSUS_FILES = (
    "train/loop.py",
    "train/optim.py",
    "instrument.py",
    "model/model.py",
    "model/loopblock.py",
    "model/engram.py",
    "model/mhc.py",
    "model/layerrope.py",
    "model/csa.py",
    "model/kda.py",
    "model/decode.py",
    "model/qat_cache.py",
)


#: Host reads a CPU-only run may take on a non-report step. Everything here is
#: a path that does not exist on CUDA: the stock optimizer's CPU fallback
#: (`foreach` is CUDA-only, so `_single_tensor_adam` calls `.item()` once per
#: parameter) and the MoE reference dispatcher (`experts._naive_dispatch`, the
#: CPU path — it `.tolist()`s the unique expert ids; on CUDA the triton kernel
#: runs instead). The gate in `tests/test_step_audit.py` reads THIS tuple.
CPU_ONLY_FRAMES = ("torch.optim (cpu fallback):", ":_naive_dispatch")


def _is_cuda(t: torch.Tensor) -> bool:
    return t.device.type != "cpu"


def _blame(skip_audit: bool = True) -> str:
    """The first caller frame outside torch and (optionally) this module.

    A read whose stack contains `torch/optim/` is the STOCK optimizer's CPU
    path (`_single_tensor_adam` calls `.item()` per parameter because the
    foreach kernels are CUDA-only) and is named as such: on CUDA the same call
    takes the foreach route and reads nothing.
    """
    stack = traceback.extract_stack()[:-2]
    for fr in reversed(stack):
        if "site-packages/torch/optim/" in fr.filename:
            return f"torch.optim (cpu fallback):{fr.name}"
    here = Path(__file__).name
    for fr in reversed(stack):
        f = fr.filename
        if "site-packages/torch" in f or "/torch/" in f:
            continue
        if skip_audit and Path(f).name == here:
            continue
        if "contextlib.py" in f:
            continue
        return f"{Path(f).name}:{fr.lineno}:{fr.name}"
    return "?"


@dataclass
class LoopStat:
    """One instrumented loop in one step."""

    instances: int = 0
    iterations: int = 0
    modules: set[int] = field(default_factory=set)

    def add(self, other: LoopStat) -> None:
        self.instances += other.instances
        self.iterations += other.iterations
        self.modules |= other.modules


class LoopProbe:
    """The object `with stepaudit.loop("name") as p:` hands back; `p.tick()`."""

    __slots__ = ("name", "iterations", "modules", "stat")

    def __init__(self, name: str, stat: LoopStat) -> None:
        self.name = name
        self.stat = stat
        self.iterations = 0
        self.modules: set[int] = set()
        stat.instances += 1

    def tick(self, module: object | None = None) -> None:
        self.iterations += 1
        if module is not None:
            self.modules.add(id(module))

    def close(self) -> None:
        self.stat.iterations += self.iterations
        self.stat.modules |= self.modules


class _NoopProbe:
    __slots__ = ()

    def tick(self, module: object | None = None) -> None:
        return None


_NOOP = _NoopProbe()


@dataclass
class StepRecord:
    """Everything one step did, as counted by the patches."""

    index: int
    report: bool
    reads: Counter = field(default_factory=Counter)
    device_reads: Counter = field(default_factory=Counter)
    frames: Counter = field(default_factory=Counter)
    allocs: int = 0
    alloc_bytes: int = 0
    peak_bytes: int | None = None
    loops: dict[str, LoopStat] = field(default_factory=dict)
    #: `.contiguous()` calls whose input was ALREADY contiguous. ATen returns
    #: the same tensor for those, so this counts pure python-call waste, not a
    #: copy — the copy case (`is_contiguous()` false) is a real cost and stays.
    noop_contig: int = 0
    #: Wall clock between the bracket's two ends. On CUDA this includes the
    #: step's enqueues and any declared sync; the trainer's `--timers` line is
    #: the same interval, so the two can be read against each other.
    duration_s: float = 0.0
    #: Set by `begin_step`, consumed by `end_step` (the dataclass stays a
    #: report value; the clock is not a reading the audit exports).
    started_s: float = field(default=0.0, repr=False, compare=False)

    def line(self) -> str:
        dev = sum(self.device_reads.values())
        reads = sum(self.reads.values())
        loops = " ".join(
            f"{k}={v.instances}x{v.iterations}" + (f"/{len(v.modules)}m" if v.modules else "")
            for k, v in sorted(self.loops.items())
        )
        peak = "-" if self.peak_bytes is None else _fmt(self.peak_bytes)
        return (
            f"step {self.index:5} {'report' if self.report else '      '} "
            f"{self.duration_s * 1e3:8.2f}ms allocs={self.allocs:5} "
            f"req={_fmt(self.alloc_bytes):>9} peak={peak:>9} "
            f"reads={reads:4} (device {dev:3}) noop_contig={self.noop_contig:3} "
            f"loops=[{loops}]"
        )


def _fmt(n: int) -> str:
    for unit, scale in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if abs(n) >= scale:
            return f"{n / scale:.1f}{unit}"
    return str(n)


class StepAudit:
    """The armed instrument. One live instance at a time (see [`install`])."""

    def __init__(self) -> None:
        self.records: list[StepRecord] = []
        self._cur: StepRecord | None = None
        self._orig: dict[tuple, object] = {}
        self._cuda = torch.cuda.is_available()

    # -- installation ---------------------------------------------------
    def install(self) -> StepAudit:
        for name in READ_METHODS:
            self._patch(torch.Tensor, name, self._read_wrapper(name))
        self._patch(torch.Tensor, "contiguous", self._contig_wrapper())
        for name in _ALLOC_FUNCS:
            self._patch(torch, name, self._alloc_wrapper(getattr(torch, name)))
        for name in _NEW_FUNCS:
            self._patch(torch.Tensor, name, self._alloc_wrapper(getattr(torch.Tensor, name)))
        if self._cuda:
            torch.cuda.reset_peak_memory_stats()
        return self

    def uninstall(self) -> None:
        for (owner, name), fn in self._orig.items():
            setattr(owner, name, fn)
        self._orig.clear()
        self._cur = None

    def _patch(self, owner, name: str, wrapper) -> None:
        self._orig[(owner, name)] = getattr(owner, name)
        setattr(owner, name, wrapper)

    def _read_wrapper(self, name: str):
        orig = getattr(torch.Tensor, name)
        audit = self

        def inner(self, *a, **kw):
            rec = audit._cur
            if rec is not None:
                rec.reads[name] += 1
                rec.frames[_blame()] += 1
                if _is_cuda(self):
                    rec.device_reads[name] += 1
            return orig(self, *a, **kw)

        return inner

    def _contig_wrapper(self):
        orig = torch.Tensor.contiguous
        audit = self

        def inner(self, *a, **kw):
            rec = audit._cur
            if rec is not None and self.is_contiguous():
                rec.noop_contig += 1
            return orig(self, *a, **kw)

        return inner

    @staticmethod
    def _bytes(out: torch.Tensor) -> int:
        return int(out.numel()) * out.element_size()

    def _alloc_wrapper(self, fn):
        audit = self

        def inner(*a, **kw):
            out = fn(*a, **kw)
            rec = audit._cur
            if rec is not None and isinstance(out, torch.Tensor):
                rec.allocs += 1
                rec.alloc_bytes += self._bytes(out)
            return out

        return inner

    # -- step bracket ---------------------------------------------------
    def begin_step(self, index: int, report: bool = False) -> StepRecord:
        if self._cur is not None:
            self.end_step()
        rec = StepRecord(index=index, report=report)
        rec.started_s = time.perf_counter()
        self.records.append(rec)
        self._cur = rec
        if self._cuda:
            torch.cuda.reset_peak_memory_stats()
        return rec

    def end_step(self) -> StepRecord | None:
        rec = self._cur
        if rec is None:
            return None
        rec.duration_s = max(time.perf_counter() - rec.started_s, 1e-9)
        if self._cuda:
            rec.peak_bytes = int(torch.cuda.max_memory_allocated())
        self._cur = None
        return rec

    # -- loops ----------------------------------------------------------
    @contextlib.contextmanager
    def loop(self, name: str, modules=()):
        rec = self._cur
        if rec is None:
            yield _NOOP
            return
        stat = rec.loops.setdefault(name, LoopStat())
        probe = LoopProbe(name, stat)
        probe.modules.update(id(m) for m in modules)
        try:
            yield probe
        finally:
            probe.close()

    # -- reporting ------------------------------------------------------
    def report(self) -> dict:
        self.end_step()
        peaks = [r.peak_bytes for r in self.records if r.peak_bytes is not None]
        return {
            "steps": len(self.records),
            "allocs": sum(r.allocs for r in self.records),
            "alloc_bytes": sum(r.alloc_bytes for r in self.records),
            "reads": sum(sum(r.reads.values()) for r in self.records),
            "device_reads": sum(sum(r.device_reads.values()) for r in self.records),
            "noop_contig": sum(r.noop_contig for r in self.records),
            "peak_bytes": max(peaks) if peaks else None,
            "ms_step": 1e3 * sum(r.duration_s for r in self.records) / max(len(self.records), 1),
            "loops": {
                name: {
                    "instances": sum(r.loops.get(name, LoopStat()).instances for r in self.records),
                    "iterations": sum(
                        r.loops.get(name, LoopStat()).iterations for r in self.records
                    ),
                }
                for name in sorted({n for r in self.records for n in r.loops})
            },
        }

    def summary(self) -> str:
        rep = self.report()
        lines = [r.line() for r in self.records]
        peak = "-" if rep["peak_bytes"] is None else _fmt(rep["peak_bytes"])
        lines.append(
            f"total: {rep['steps']} steps · {rep['ms_step']:.1f} ms/step "
            f"allocs={rep['allocs']} req={_fmt(rep['alloc_bytes'])} peak={peak} "
            f"reads={rep['reads']} (device {rep['device_reads']}) "
            f"noop_contig={rep['noop_contig']}"
        )
        return "\n".join(lines)


#: The live instance. `begin_step`/`end_step` are module-level on purpose: the
#: trainer calls them once per step and they must cost one branch when disarmed.
_ACTIVE: StepAudit | None = None


def install() -> StepAudit:
    global _ACTIVE
    if _ACTIVE is not None:
        _ACTIVE.uninstall()
    _ACTIVE = StepAudit().install()
    return _ACTIVE


def uninstall() -> None:
    global _ACTIVE
    if _ACTIVE is not None:
        _ACTIVE.uninstall()
        _ACTIVE = None


def current() -> StepAudit | None:
    return _ACTIVE


def begin_step(index: int, report: bool = False) -> None:
    if _ACTIVE is not None:
        _ACTIVE.begin_step(index, report)


def end_step() -> None:
    if _ACTIVE is not None:
        _ACTIVE.end_step()


def loop(name: str, modules=()):
    """`with stepaudit.loop("name") as p:` — a no-op when disarmed."""
    if _ACTIVE is None or _ACTIVE._cur is None:
        return _noop_ctx()
    return _ACTIVE.loop(name, modules)


def iterations(name: str, iterable, modules=()):
    """`for x in stepaudit.iterations("name", xs):` — counts, no re-indent.

    Returns the bare iterator when disarmed, so a production run pays one
    branch per loop, not a generator hop per item.
    """
    if _ACTIVE is None or _ACTIVE._cur is None:
        return iter(iterable)
    return _iterations(name, iterable, modules)


def _iterations(name: str, iterable, modules=()):
    with _ACTIVE.loop(name, modules) as probe:  # type: ignore[union-attr]
        for item in iterable:
            probe.iterations += 1
            yield item


@contextlib.contextmanager
def _noop_ctx():
    yield _NOOP


# ---------------------------------------------------------------------------
# The static census: every loop in the step's files, annotated or excused.
# ---------------------------------------------------------------------------

#: Functions whose loops are NOT part of a step body. The value is the reason,
#: printed by the census so an excuse cannot hide in a docstring.
EXCUSED_FUNCTIONS = {
    "_train_impl.retract_stage": (
        "stiefel arm only (stiefel_qr=true): a fixed per-factor list built once at "
        "startup, one QR per TSCT factor, no host read"
    ),
    "DormouseModel.set_iter_checkpoint": "setup, once per run",
    "DormouseModel.set_factor_qat_alpha": "anneal arm only (factor_qat_anneal_steps>0)",
    "DormouseModel._forward_byte_lm": (
        "prelude/coda walk unique layers in sequence; they are not homogeneous "
        "(different weights, each layer's input is the previous output)"
    ),
    "unpack_state_dict": "load, once per run",
    "init_controller": "load, once per run",
    "_restore_lfb": "load, once per run",
    "_shadow_state": "EMA setup, once per run",
    "_shadow_params": "SF arm only (muon_sf)",
    "_eval_loss": "eval cadence, no backward",
    "MixOptimizer._fast_adamw_ok": "capability probe over <=8 param groups, no tensor work",
    "MixOptimizer.set_ns_steps": "resume, once per run (a PROGRESS key repair)",
    "DormouseModel.set_recompute": "setup, once per run",
    "_train_impl": "the step DRIVER itself — the bracket, not a loop inside a step (renamed from `train` by the 2026-11-08 run-registry wrapper)",
    "corpus_manifest": "checkpoint save, once per write: hashes the plan and stats each source",
    "zeropower_via_ns": "per-matrix NS, the retraction's inherent sequential chain",
    "_foreach_add_mixed_": "comprehension inside the batched momentum primitive",
    "Muon._build_groups": "setup, once per optimizer",
    "Muon.step": "the per-parameter reference step (the trainer calls batched_step_from_model)",
    "Muon.sf_switch": "SF arm only (muon_sf)",
    "Muon._orthogonalize_many": "per-shape batched chain; the loop is the dispatch over groups",
    "MixOptimizer.__init__": "construction, once per run",
    "MixOptimizer.group_grad_norms": "log cadence only (report steps)",
    "fnv": "host key reference, tests only",
    "raw_keys": "host key reference, tests only",
    "sinkhorn_l1": "host table update (engram_ram), sequential normalisation",
    "Engram.host_step": "host-table arm only (engram_ram), sequential row update",
    "enable_host_tables": "setup, once per run",
    "host_step_all": "host-table arm only (engram_ram)",
    "load_host_state": "load, once per run",
    "attach_qat_caches": "setup, once per run",
    "CSAState.reserve": "state reservation, once per decode/prefill shape",
    "CompressedSelectiveAttention._hier_fine": "CSA construction, once per module",
    "memory_table": "CSA construction, once per module",
    "KDAAttention.step_state": "decode only (one position at a time), never a training step",
    "decode_bytes": "decode only",
    "_state_refusal": "decode only",
    "_GraphStep._body": "decode graph capture only",
    "_GraphStep._capture": "decode graph capture only",
    "_GraphStep._fill": "decode graph capture only",
    "_GraphStep.step": "decode graph capture only",
    "_mtp_draft": "decode only",
    "recurrent_decode": "decode only",
}


@dataclass
class LoopSite:
    file: str
    line: int
    kind: str
    function: str
    annotated: bool
    excused: str | None

    @property
    def status(self) -> str:
        if self.annotated:
            return "counted"
        if self.excused:
            return "excused"
        return "UNMARKED"


def _qualname_stack(node, stack):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        stack.append(node.name)
    return stack


class _Census(ast.NodeVisitor):
    def __init__(self, path: str, source: str) -> None:
        self.path = path
        self.lines = source.splitlines()
        self.stack: list[str] = []
        self.sites: list[LoopSite] = []

    def _add(self, node, kind: str) -> None:
        fn = ".".join(self.stack) or "<module>"
        start = max(0, node.lineno - 3)
        line = "\n".join(self.lines[start : node.lineno])
        annotated = "stepaudit.loop(" in line or "stepaudit.iterations(" in line
        excuse = EXCUSED_FUNCTIONS.get(fn)
        if excuse is None and not annotated:
            # A `# stepaudit-excuse:` marker on the loop line is the local form,
            # for a loop whose reason is only true here.
            if "stepaudit-excuse:" in line:
                excuse = line.split("stepaudit-excuse:", 1)[1].strip()
        self.sites.append(LoopSite(self.path, node.lineno, kind, fn, annotated, excuse))

    def visit_FunctionDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_For(self, node):
        self._add(node, "for")
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_While(self, node):
        self._add(node, "while")
        self.generic_visit(node)


def census(root: str | Path | None = None) -> list[LoopSite]:
    """Every python loop in the step's files, with counted/excused state."""
    base = Path(root) if root is not None else Path(__file__).resolve().parent
    out: list[LoopSite] = []
    for rel in CENSUS_FILES:
        path = base / rel
        source = path.read_text()
        v = _Census(rel, source)
        v.visit(ast.parse(source))
        out.extend(v.sites)
    return out


def unmarked(root: str | Path | None = None) -> list[LoopSite]:
    """The loops the census cannot justify — the gate's list."""
    return [s for s in census(root) if s.status == "UNMARKED"]
