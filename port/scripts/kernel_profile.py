#!/usr/bin/env -S uv run --python /home/sehaxe/dormouse/port/.venv/bin/python --script
"""Where does one training step's GPU time go, per KERNEL, with module attribution?

`bench_step.py` answers "how fast" and `launch_profile.py` answers "how many
launches and which phase". This answers the question the last three optimisation
guesses got wrong: **which kernel, and whose kernel**. The ledger says the dense
path is ~99% of the MACs and the MoE ~1%; that is not a time share. A 199M m400
step at b8 x s1024 is ~125 ms, and a 25/38/37 matmul/pointwise/launch split is
also not a time share once the fused epilogue and the fused residual land.

HOW THE ATTRIBUTION WORKS
-------------------------
Every region is a `torch.profiler.record_function` span around a real seam:
`model.embed`, `nn.RMSNorm`/`LayerRoPENorm` + `fused_residual`, the KDA
projections and FLA recurrence, the CSA fine/coarse arms, `Engram`, the MoE
dispatch and its expert GEMMs (including `moe_kernels.combine`), `depth_attend`,
`MHCMix`, the per-iteration/final head+CE, `MixOptimizer.step`, and
`clip_grads`/`agc_clip`.

A kernel launched inside a span is attributed by walking the CPU event's
`cpu_parent` chain to the nearest `REGIONS` name. BACKWARD kernels have no such
parent: they run from `autograd::engine::evaluate_function: XBackward`, whose
`sequence_nr` is the FORWARD op's autograd node id. The collector builds a
`sequence_nr -> region` map from the forward half and resolves every backward
event through it, so an `AddmmBackward` whose forward `addmm` ran inside
`kda_proj` is still `kda_proj`. This is why `ProfilerActivity.CPU` is in the
activity list even though the report reads CUDA time: a CUDA-only trace has no
`record_function` events and no launch parents (`use_cpu=(CPU in activities)`),
i.e. no attribution at all. The CUDA activity is the one the columns measure.

Memory: `profile_memory=True` gives `self_device_memory_usage` PER CPU OP, not
per kernel. When an op launched one kernel the bytes go to it exactly; when it
launched N kernels the op's net bytes are split evenly and the row is printed
with `~` (and `bytes_estimated` in JSON). Negative values are net frees.

PROTOCOL
--------
3 warmup steps, then 5 measured steps (`schedule(warmup=3, active=5)`). The
profiled run RESUMES the real checkpoint (default `m400b`, the current
resumable m400 run — `m400`'s saved LFB bias is past the trainer's +-10 refusal)
and stops itself after the 8th step, so `steps`, the WSD schedule and the
optimizer state are the run's own, and no checkpoint is written. `eval`,
`eval_every`, `ckpt_every` and `log_every` are also pinned away from any write.
`full_graph`, `opt_graph` and `torch_compile` are REFUSED: a captured/compiled
step does not run the Python seams, so the report would be a different program
wearing the same numbers.

SAFETY
------
One GPU process at a time (AGENTS 1.5). `scripts/_lane.preflight` owns the
check; a busy card exits **2** with the reason (unlike the measurement scripts'
SKIP/0: this one was asked to profile, and it did not). Nothing here trains.

    # during a checkpoint gap (the card must be free; do NOT pass "" here)
    cd /home/sehaxe/dormouse/port
    uv run --extra gpu python scripts/kernel_profile.py \
        --checkpoint /home/sehaxe/dormouse/checkpoints/m400b/m400b.pt \
        --json /tmp/kprof_m400b.json

    # a fresh 8-step run of the m400 preset, no checkpoint writes
    uv run --extra gpu python scripts/kernel_profile.py --fresh \
        --json /tmp/kprof_fresh.json

    # compare two runs without the card
    uv run python scripts/kernel_profile.py \
        --diff /tmp/kprof_before.json /tmp/kprof_after.json

WHAT THE OUTPUT DECIDES
-----------------------
The ranked kernel table names the launch that owns the next millisecond and
whether it is forward or backward (through its region). The region x family
table says whether an arm is GEMM-bound, pointwise-bound or dispatch-bound. The
header's `GPU busy` vs `step wall` says whether the remaining time is kernels at
all. `--json` makes the next change comparable: a row that moved is the only
number an optimisation is allowed to claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import importlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# `scripts/` is not a package. APPEND (never insert): a script name at the front
# of `sys.path` would shadow a package of that name.
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight as _lane_preflight  # noqa: E402

CPU = torch.autograd.DeviceType.CPU
CUDA = torch.autograd.DeviceType.CUDA
WARMUP_STEPS = 3
MEASURED_STEPS = 5
DEFAULT_CHECKPOINT = Path("/home/sehaxe/dormouse/checkpoints/m400b/m400b.pt")
DEFAULT_PRESET = "configs/m400.toml"

#: The region vocabulary. The tables and the JSON use exactly these names, so a
#: new seam cannot invent a row a diff cannot match.
REGIONS: tuple[str, ...] = (
    "embed",
    "norm/epilogue",
    "kda_proj",
    "kda_recurrent",
    "csa_fine",
    "csa_coarse",
    "engram",
    "moe_dispatch",
    "moe_gemm",
    "attnres",
    "mhc_write",
    "head/ce",
    "optimizer",
    "clip",
)

#: Kernel families for the region x family rollup. First match wins.
FAMILIES: tuple[str, ...] = (
    "gemm",
    "attention",
    "norm",
    "reduce",
    "index",
    "copy",
    "pointwise",
    "other",
)

_FAMILY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "gemm",
        re.compile(
            r"gemm|sgemm|hgemm|cutlass|tensorop|wgmma|mma|matmul|addmm|bmm|"
            r"gemv|cublas|grouped_gemm",
            re.I,
        ),
    ),
    (
        "attention",
        re.compile(
            r"attention|flash|softmax|chunk_kda|chunk_gated|wy_repr|gated_delta|"
            r"kda|recurrent|scan|topk_softmax",
            re.I,
        ),
    ),
    ("norm", re.compile(r"rmsnorm|layer_norm|group_norm|norm|l2norm", re.I)),
    ("reduce", re.compile(r"reduce|sum|mean|amax|argmax|topk|sort", re.I)),
    (
        "index",
        re.compile(r"gather|scatter|index_select|index_add|bincount|permute|embedding", re.I),
    ),
    ("copy", re.compile(r"memcpy|memset|copy|fill|cat|clone", re.I)),
    (
        "pointwise",
        re.compile(
            r"elementwise|vectorized|silu|glu|sigmoid|exp|mul|add|sub|div|clamp|where|masked",
            re.I,
        ),
    ),
)


@dataclass(frozen=True)
class SeamSpec:
    """One class/function seam: `module.target.attr` gets `region`'s span."""

    module: str
    target: str
    attr: str
    region: str


#: The `record_function` seams. `target` is resolved from `module` (empty means
#: the module itself) and `attr` is replaced on that owner. Instance seams
#: (`model.embed`, `model.lm_head`, the CSA projection Linears, the shared
#: expert) are discovered from the live model and are not listed here — they
#: have no stable class name.
SEAM_SPECS: tuple[SeamSpec, ...] = (
    SeamSpec("torch.nn", "RMSNorm", "forward", "norm/epilogue"),
    SeamSpec("dormouse.model.fused_residual", "", "epilogue_read", "norm/epilogue"),
    SeamSpec("dormouse.model.fused_residual", "", "pre", "norm/epilogue"),
    SeamSpec("dormouse.model.fused_residual", "", "write", "norm/epilogue"),
    SeamSpec("dormouse.model.layerrope", "LayerRoPENorm", "forward", "norm/epilogue"),
    SeamSpec("dormouse.model.kda", "KDAAttention", "projections", "kda_proj"),
    SeamSpec("dormouse.model.kda", "KDAAttention", "_fla_path", "kda_recurrent"),
    SeamSpec("dormouse.model.kda", "KDAAttention", "_sdpa_path", "kda_recurrent"),
    SeamSpec("dormouse.model.kda", "KDAAttention", "_sdpa_path_carry", "kda_recurrent"),
    SeamSpec("dormouse.model.csa", "CompressedSelectiveAttention", "_pool", "csa_fine"),
    SeamSpec("dormouse.model.csa", "CompressedSelectiveAttention", "_attend", "csa_fine"),
    SeamSpec("dormouse.model.csa", "CompressedSelectiveAttention", "_select", "csa_fine"),
    SeamSpec("dormouse.model.csa", "CompressedSelectiveAttention", "_select_fused", "csa_fine"),
    SeamSpec("dormouse.model.csa", "CompressedSelectiveAttention", "_hier_fine", "csa_fine"),
    SeamSpec("dormouse.model.csa", "CompressedSelectiveAttention", "_pool_coarse", "csa_coarse"),
    SeamSpec("dormouse.model.csa", "CompressedSelectiveAttention", "_level_scores", "csa_coarse"),
    SeamSpec("dormouse.model.csa", "CompressedSelectiveAttention", "_attend_dense", "csa_coarse"),
    SeamSpec("dormouse.model.csa", "CompressedSelectiveAttention", "_dense_out", "csa_coarse"),
    SeamSpec("dormouse.model.engram", "Engram", "forward", "engram"),
    SeamSpec("dormouse.model.engram", "Engram", "slots_for", "engram"),
    SeamSpec("dormouse.model.experts", "SparseMoE", "forward", "moe_dispatch"),
    SeamSpec("dormouse.model.experts", "SparseMoE", "_dispatch", "moe_dispatch"),
    SeamSpec("dormouse.model.experts", "SparseMoE", "_gather_rows", "moe_dispatch"),
    SeamSpec("dormouse.model.experts", "SparseMoE", "_permute", "moe_dispatch"),
    SeamSpec("dormouse.model.experts", "moe_kernels", "expert_gemm_glu", "moe_gemm"),
    SeamSpec("dormouse.model.experts", "moe_kernels", "combine", "moe_dispatch"),
    SeamSpec("dormouse.model.experts", "SparseMoE", "_padded_dispatch", "moe_gemm"),
    SeamSpec("dormouse.model.experts", "SparseMoE", "_bank_dispatch", "moe_gemm"),
    SeamSpec("dormouse.model.experts", "SparseMoE", "_batched_dispatch", "moe_gemm"),
    SeamSpec("dormouse.model.experts", "SparseMoE", "_bucketed_dispatch", "moe_gemm"),
    SeamSpec("dormouse.model.experts", "SparseMoE", "_naive_dispatch", "moe_gemm"),
    SeamSpec("dormouse.model.loopblock", "", "depth_attend", "attnres"),
    SeamSpec("dormouse.model.loopblock", "LoopBlock", "_iter_ce", "head/ce"),
    SeamSpec("dormouse.model.loopblock", "", "masked_ce_mean", "head/ce"),
    SeamSpec("dormouse.model.loopblock", "", "fixed_cross_entropy", "head/ce"),
    SeamSpec("dormouse.model.mhc", "MHCMix", "forward", "mhc_write"),
    SeamSpec("dormouse.model.mhc", "MHCMix", "pre_mix", "mhc_write"),
    SeamSpec("dormouse.model.model", "TiedHead", "forward", "head/ce"),
    SeamSpec("dormouse.model.model", "", "masked_ce_mean", "head/ce"),
    SeamSpec("dormouse.model.model", "", "fixed_cross_entropy", "head/ce"),
    SeamSpec("dormouse.train.optim", "MixOptimizer", "step", "optimizer"),
    SeamSpec("dormouse.train.loop", "", "clip_grads", "clip"),
    SeamSpec("dormouse.train.loop", "", "agc_clip", "clip"),
)


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------


def canonical_name(raw: str) -> str:
    """Demangled-ish, whitespace-collapsed, runtime argument list removed.

    Grouping by the full canonical name (not the printed truncation) keeps two
    template instantiations apart; `short_name` truncates only for the table.
    """
    text = " ".join(str(raw).split())
    # The runtime argument list is not part of the kernel's identity; the
    # template arguments are, and they do not contain a bare `(` in practice.
    text = text.split("(", 1)[0].strip()
    return text or "?"


def demangle_all(names: Iterable[str]) -> dict[str, str]:
    """`c++filt` over the unique names, one subprocess for the whole profile.

    c++filt is binutils, not a Python dependency: when it is absent the raw
    names are used and nothing fails. Names without `_Z` are passed through
    without spawning anything (the CPU test suite takes that path).
    """
    unique = sorted({str(n) for n in names})
    out = {n: n for n in unique}
    if not unique or not any("_Z" in n for n in unique):
        return out
    try:
        proc = subprocess.run(
            ["c++filt"], input="\n".join(unique), capture_output=True, text=True, check=False
        )
    except OSError:
        return out
    lines = proc.stdout.splitlines()
    if proc.returncode != 0 or len(lines) != len(unique):
        return out
    return dict(zip(unique, lines, strict=True))


def family_of(name: str) -> str:
    """The coarse kernel family used by the region x family rollup."""
    for family, rule in _FAMILY_RULES:
        if rule.search(name):
            return family
    return "other"


def short_name(name: str, width: int = 58) -> str:
    return name if len(name) <= width else name[: width - 1] + "…"


# ---------------------------------------------------------------------------
# seams
# ---------------------------------------------------------------------------


@dataclass
class Seam:
    owner: Any
    attr: str
    original: Any
    had_attr: bool
    region: str


def _resolve(module_name: str, target: str) -> Any:
    owner = importlib.import_module(module_name)
    for part in target.split(".") if target else ():
        owner = getattr(owner, part)
    return owner


def _wrap(owner: Any, attr: str, region: str) -> Seam:
    """Replace `owner.attr` with a `record_function(region)` wrapper."""
    if region not in REGIONS:
        raise ValueError(f"{region!r} is not in REGIONS")
    orig = getattr(owner, attr)
    if getattr(orig, "__kprof_region__", None) is not None:
        raise RuntimeError(f"{owner!r}.{attr} is already wrapped")
    had = attr in getattr(owner, "__dict__", {})

    if isinstance(owner, type):

        @functools.wraps(orig)
        def wrapped(self, *args, **kwargs):
            with torch.profiler.record_function(region):
                return orig(self, *args, **kwargs)

    else:

        @functools.wraps(orig)
        def wrapped(*args, **kwargs):
            with torch.profiler.record_function(region):
                return orig(*args, **kwargs)

    wrapped.__kprof_region__ = region  # type: ignore[attr-defined]
    setattr(owner, attr, wrapped)
    return Seam(owner, attr, orig, had, region)


def restore_seams(seams: Sequence[Seam]) -> None:
    for seam in reversed(seams):
        if seam.had_attr:
            setattr(seam.owner, seam.attr, seam.original)
        else:
            delattr(seam.owner, seam.attr)


def install_regions(model: torch.nn.Module) -> tuple[list[Seam], list[str]]:
    """Install every class/function seam, then the live-model instance seams.

    Returns `(seams, skipped)`: `skipped` names the optional targets this
    build/host does not carry (e.g. `moe_kernels` without Triton), never hides
    a seam that was expected but broken.
    """
    from dormouse.model import csa

    seams: list[Seam] = []
    skipped: list[str] = []
    seen: set[tuple[int, str]] = set()

    def add(owner: Any, attr: str, region: str, where: str) -> None:
        key = (id(owner), attr)
        if key in seen:
            return
        try:
            seams.append(_wrap(owner, attr, region))
            seen.add(key)
        except AttributeError:
            skipped.append(f"{where}.{attr}")
        except RuntimeError:
            skipped.append(f"{where}.{attr} (already wrapped)")

    for spec in SEAM_SPECS:
        try:
            owner = _resolve(spec.module, spec.target)
        except (ImportError, AttributeError):
            skipped.append(f"{spec.module}.{spec.target}")
            continue
        if owner is None:
            skipped.append(f"{spec.module}.{spec.target}")
            continue
        add(owner, spec.attr, spec.region, f"{spec.module}.{spec.target}")

    base = getattr(model, "_orig_mod", model)
    embed = getattr(base, "embed", None)
    if isinstance(embed, torch.nn.Module):
        add(embed, "forward", "embed", "model.embed")
    head = getattr(base, "lm_head", None)
    if isinstance(head, torch.nn.Module):
        add(head, "forward", "head/ce", "model.lm_head")

    for module in base.modules():
        if not isinstance(module, csa.CompressedSelectiveAttention):
            continue
        for attr in ("W_q", "W_k", "W_v", "W_o"):
            sub = getattr(module, attr, None)
            if isinstance(sub, torch.nn.Module):
                add(sub, "forward", "csa_fine", f"CSA.{attr}")
        for attr in ("W_q_c", "W_k_c", "W_v_c", "W_o_c"):
            sub = getattr(module, attr, None)
            if isinstance(sub, torch.nn.Module):
                add(sub, "forward", "csa_coarse", f"CSA.{attr}")
        # The shared expert is MoE compute, not an expert-GEMM arm; label it
        # with the routed GEMMs so the two do not land in `other`.
        shared = getattr(module, "shared", None)
        if isinstance(shared, torch.nn.Module):
            add(shared, "forward", "moe_gemm", "SparseMoE.shared")

    return seams, skipped


class _StopTraining(Exception):
    """The tiny stop: the profiled run ends after `total` optimizer steps."""


@dataclass
class StepClock:
    """CUDA-event wall clock: one mark at the END of every training step.

    The `ProfilerStep*` annotation is host time; the trainer never synchronizes
    per step, so its range is the launch span, not the GPU's. An event recorded
    on the current stream at the end of each step, then `elapsed_time` between
    consecutive marks after one sync, is the real step wall (including the
    launch gap). The first `warmup` marks only establish the start line.
    """

    events: list[Any] = dataclasses.field(default_factory=list)

    def mark(self) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events.append(event)

    def measured_ms(self, warmup: int, measured: int) -> list[float]:
        if warmup < 1 or len(self.events) < warmup + measured:
            return []
        if not torch.cuda.is_available():
            return []
        torch.cuda.synchronize()
        return [
            float(self.events[warmup - 1 + i].elapsed_time(self.events[warmup + i]))
            for i in range(measured)
        ]


def install_step_hook(
    prof: torch.profiler.profile, total: int, clock: StepClock | None = None
) -> Seam:
    """`prof.step()` at the trainer's own step boundary, then stop after `total`.

    `stepaudit.end_step` is the LAST call in the loop body (after the optimizer,
    the bank/QAT invalidation and the counters), so the `ProfilerStep*` range is
    the step's whole host span. Waiting for `train()` to return would either run
    the schedule's remaining steps or need `steps` rewritten (which changes the
    WSD endpoint); this raises after the 8th `prof.step()` instead.
    """
    import dormouse.stepaudit as stepaudit

    orig = stepaudit.end_step
    count = 0

    @functools.wraps(orig)
    def wrapped():
        nonlocal count
        result = orig()
        if clock is not None:
            clock.mark()
        count += 1
        prof.step()
        if count >= total:
            raise _StopTraining
        return result

    had = "end_step" in getattr(stepaudit, "__dict__", {})
    stepaudit.end_step = wrapped
    return Seam(stepaudit, "end_step", orig, had, "step")


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelSample:
    """ONE kernel launch, before grouping."""

    name: str
    region: str
    family: str
    us: float
    bytes: float = 0.0
    bytes_estimated: bool = False
    count: int = 1


@dataclass
class KernelRow:
    """One grouped `(kernel, region)` row."""

    name: str
    region: str
    family: str
    us: float
    count: int
    bytes: float
    bytes_estimated: bool
    pct: float = 0.0


@dataclass
class FamilyRow:
    region: str
    family: str
    us: float
    count: int
    bytes: float
    pct: float = 0.0


def _nearest_region(event: Any) -> str | None:
    node = event
    while node is not None:
        if getattr(node, "name", None) in REGIONS:
            return node.name
        node = getattr(node, "cpu_parent", None)
    return None


def _seq_region(event: Any, seq_regions: dict[int, str]) -> str | None:
    """The forward region of a backward event, through its autograd node id."""
    node = event
    while node is not None:
        seq = getattr(node, "sequence_nr", -1)
        if seq is not None and int(seq) >= 0:
            region = seq_regions.get(int(seq))
            if region is not None:
                return region
        node = getattr(node, "cpu_parent", None)
    return None


def region_of(event: Any, seq_regions: dict[int, str]) -> str:
    return _nearest_region(event) or _seq_region(event, seq_regions) or "other"


def collect_samples(events: Sequence[Any], *, demangle: bool = True) -> list[KernelSample]:
    """Every executed CUDA kernel, once, attributed to its region and family.

    The forward half is read first: `(sequence_nr -> region)` is the bridge the
    backward half has no other way to cross. `evt.kernels` is the launch list
    the profiler already correlated; a kernel appears exactly once.
    """
    cpu_events = [e for e in events if getattr(e, "device_type", None) == CPU]
    seq_regions: dict[int, str] = {}
    for event in cpu_events:
        if int(getattr(event, "scope", 0) or 0) != 0:
            continue
        seq = getattr(event, "sequence_nr", -1)
        if seq is None or int(seq) < 0:
            continue
        region = _nearest_region(event)
        if region is not None:
            seq_regions.setdefault(int(seq), region)

    raw: list[tuple[Any, list[Any], int]] = []
    for event in cpu_events:
        kernels = list(getattr(event, "kernels", ()) or ())
        if not kernels:
            continue
        region = region_of(event, seq_regions)
        nbytes = int(getattr(event, "self_device_memory_usage", 0) or 0)
        raw.append((event, kernels, nbytes))

    names = [str(k.name) for _, kernels, _ in raw for k in kernels]
    canon = demangle_all(names) if demangle else {n: n for n in names}

    samples: list[KernelSample] = []
    for event, kernels, nbytes in raw:
        region = region_of(event, seq_regions)
        split = nbytes / len(kernels) if kernels else 0.0
        estimated = len(kernels) > 1 and nbytes != 0
        for kernel in kernels:
            name = canonical_name(canon.get(str(kernel.name), str(kernel.name)))
            samples.append(
                KernelSample(
                    name=name,
                    region=region,
                    family=family_of(name),
                    us=float(kernel.duration),
                    bytes=split,
                    bytes_estimated=estimated,
                )
            )
    return samples


def aggregate(samples: Iterable[KernelSample]) -> list[KernelRow]:
    """Group by `(kernel, region)`, then sort by time and fill `pct`/`mean`."""
    acc: dict[tuple[str, str], KernelRow] = {}
    for sample in samples:
        key = (sample.name, sample.region)
        row = acc.get(key)
        if row is None:
            acc[key] = KernelRow(
                name=sample.name,
                region=sample.region,
                family=sample.family,
                us=0.0,
                count=0,
                bytes=0.0,
                bytes_estimated=False,
            )
            row = acc[key]
        row.us += float(sample.us)
        row.count += int(sample.count)
        row.bytes += float(sample.bytes)
        row.bytes_estimated = row.bytes_estimated or sample.bytes_estimated
    rows = list(acc.values())
    total = sum(row.us for row in rows)
    for row in rows:
        row.pct = 100.0 * row.us / total if total > 0 else 0.0
    rows.sort(key=lambda row: (-row.us, row.name, row.region))
    return rows


def region_family_rows(rows: Sequence[KernelRow]) -> list[FamilyRow]:
    """`(region, family)` sums, ordered by the region's total time."""
    acc: dict[tuple[str, str], FamilyRow] = {}
    region_us: dict[str, float] = defaultdict(float)
    for row in rows:
        key = (row.region, row.family)
        item = acc.get(key)
        if item is None:
            item = acc[key] = FamilyRow(row.region, row.family, 0.0, 0, 0.0)
        item.us += row.us
        item.count += row.count
        item.bytes += row.bytes
        region_us[row.region] += row.us
    total = sum(row.us for row in rows)
    out = list(acc.values())
    for item in out:
        item.pct = 100.0 * item.us / total if total > 0 else 0.0
    out.sort(key=lambda item: (-region_us[item.region], -item.us, item.region, item.family))
    return out


def mean_us(row: KernelRow) -> float:
    return row.us / row.count if row.count else 0.0


def union_us(spans: Iterable[tuple[float, float]]) -> float:
    """Length of the UNION of `[start, end)` spans, in microseconds.

    The header's "GPU busy" is this, not the kernel-time sum: overlapping
    streams would otherwise report more busy time than the wall has. The table
    rows keep the sum because that is what per-kernel attribution adds to.
    """
    ordered = sorted((s, e) for s, e in spans if e > s)
    if not ordered:
        return 0.0
    total = 0.0
    cur_s, cur_e = ordered[0]
    for start, end in ordered[1:]:
        if start > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = start, end
        else:
            cur_e = max(cur_e, end)
    return total + cur_e - cur_s


def payload(
    rows: Sequence[KernelRow],
    families: Sequence[FamilyRow],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """The JSON document. Every row, with the full canonical name."""
    return {
        "kind": "dormouse-kernel-profile",
        "version": 1,
        "meta": dict(meta),
        "kernels": [dataclasses.asdict(row) for row in rows],
        "region_family": [dataclasses.asdict(row) for row in families],
    }


def diff_kernels(
    before: Sequence[dict[str, Any]], after: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Per `(kernel, region)` delta, sorted by absolute time change.

    Rows present on one side only get their own time as the delta, which is how
    a kernel that appeared or disappeared is made visible rather than dropped.
    """
    old = {(row["name"], row["region"]): row for row in before}
    new = {(row["name"], row["region"]): row for row in after}
    out: list[dict[str, Any]] = []
    for key in set(old) | set(new):
        a = old.get(key, {})
        b = new.get(key, {})
        out.append(
            {
                "name": key[0],
                "region": key[1],
                "us_before": float(a.get("us", 0.0)),
                "us_after": float(b.get("us", 0.0)),
                "d_us": float(b.get("us", 0.0)) - float(a.get("us", 0.0)),
                "d_pct": float(b.get("pct", 0.0)) - float(a.get("pct", 0.0)),
                "d_calls": int(b.get("count", 0)) - int(a.get("count", 0)),
                "d_bytes": float(b.get("bytes", 0.0)) - float(a.get("bytes", 0.0)),
            }
        )
    out.sort(key=lambda row: (-abs(row["d_us"]), row["name"], row["region"]))
    return out


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def write_json(path: Path, doc: dict[str, Any]) -> None:
    path = Path(path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------


def _mib(value: float) -> str:
    if value == 0:
        return "-"
    return f"{value / 2**20:,.2f}"


def print_report(
    rows: Sequence[KernelRow],
    families: Sequence[FamilyRow],
    meta: dict[str, Any],
    *,
    top: int = 30,
) -> None:
    step_ms = float(meta.get("step_ms", 0.0))
    kernel_ms = float(meta.get("kernel_ms_per_step", 0.0))
    busy_ms = float(meta.get("gpu_busy_ms_per_step", kernel_ms))
    busy = 100.0 * busy_ms / step_ms if step_ms > 0 else 0.0
    gap_ms = max(step_ms - busy_ms, 0.0)
    gap = 100.0 * gap_ms / step_ms if step_ms > 0 else 0.0
    print(
        f"\n=== {meta.get('checkpoint') or meta.get('config')} · "
        f"b{meta.get('batch')} x s{meta.get('seq_len')} · "
        f"warmup {meta.get('warmup')} + measured {meta.get('measured')} steps ==="
    )
    print(
        f"step wall {step_ms:8.2f} ms   GPU busy {busy_ms:8.2f} ms ({busy:5.1f}%)   "
        f"launch/host gap {gap_ms:7.2f} ms ({gap:5.1f}%)   "
        f"kernel sum {kernel_ms:8.2f} ms"
    )
    each = meta.get("step_ms_each") or []
    if each:
        print(
            f"  step wall source {meta.get('step_wall_source', '?')}; "
            "each: " + " ".join(f"{value:.2f}" for value in each) + " ms"
        )
    print(
        f"kernels/step {meta.get('kernels_per_step', 0):6d}   "
        f"launches/step {meta.get('launches_per_step', 0):6d}   "
        f"net device bytes {_mib(float(meta.get('bytes', 0)))} MiB   "
        f"attributed {meta.get('attributed_pct', 0.0):5.1f}%"
    )
    print(
        "  (%step is the share of measured GPU KERNEL time and sums to 100; "
        "the gap above is wall time no kernel covered)"
    )

    print("\n-- kernels by GPU time " + "-" * 54)
    print(
        f"{'#':>3} {'kernel':<58} {'region':<14} {'family':<10} "
        f"{'ms':>8} {'%step':>6} {'calls':>7} {'µs/call':>9} {'MiB':>9}"
    )
    shown = rows if top <= 0 else rows[:top]
    for i, row in enumerate(shown, 1):
        mib = _mib(row.bytes)
        if row.bytes_estimated and mib != "-":
            mib = "~" + mib
        print(
            f"{i:3d} {short_name(row.name, 58):<58} {row.region:<14} "
            f"{row.family:<10} {row.us / 1000.0:8.2f} {row.pct:6.2f} "
            f"{row.count:7d} {mean_us(row):9.1f} {mib:>9}"
        )
    if top > 0 and len(rows) > top:
        rest = rows[top:]
        print(
            f"    {'… ' + str(len(rest)) + ' more rows':<58} {'':<14} {'':<10} "
            f"{sum(r.us for r in rest) / 1000.0:8.2f} "
            f"{sum(r.pct for r in rest):6.2f} {sum(r.count for r in rest):7d}"
        )

    print("\n-- module x kernel family " + "-" * 49)
    print(
        f"{'region':<14} {'family':<10} {'ms':>8} {'%step':>6} {'calls':>7} "
        f"{'µs/call':>9} {'MiB':>9}"
    )
    for item in families:
        mean = item.us / item.count if item.count else 0.0
        mib = _mib(item.bytes)
        print(
            f"{item.region:<14} {item.family:<10} {item.us / 1000.0:8.2f} "
            f"{item.pct:6.2f} {item.count:7d} {mean:9.1f} {mib:>9}"
        )
    other_pct = sum(item.pct for item in families if item.region == "other")
    if other_pct > 0:
        print(
            f"\nother = {other_pct:.1f}% of GPU time: kernels no seam or backward "
            "node could name. The per-kernel table above still names every one."
        )


def print_diff(a: dict[str, Any], b: dict[str, Any], *, top: int = 30) -> None:
    ma, mb = a.get("meta", {}), b.get("meta", {})
    print(
        f"\n=== diff: {ma.get('checkpoint') or ma.get('config')} -> "
        f"{mb.get('checkpoint') or mb.get('config')} ==="
    )
    for key in (
        "step_ms",
        "gpu_busy_ms_per_step",
        "kernel_ms_per_step",
        "kernels_per_step",
        "launches_per_step",
    ):
        before, after = float(ma.get(key, 0.0)), float(mb.get(key, 0.0))
        print(f"  {key:<20} {before:10.2f} -> {after:10.2f}  ({after - before:+.2f})")
    rows = diff_kernels(a.get("kernels", []), b.get("kernels", []))
    print("\n-- kernels that moved " + "-" * 55)
    print(f"{'kernel':<58} {'region':<14} {'Δms':>9} {'Δ%step':>8} {'Δcalls':>8} {'ΔMiB':>9}")
    shown = rows if top <= 0 else rows[:top]
    for row in shown:
        print(
            f"{short_name(row['name'], 58):<58} {row['region']:<14} "
            f"{row['d_us'] / 1000.0:+9.2f} {row['d_pct']:+8.2f} "
            f"{row['d_calls']:+8d} {row['d_bytes'] / 2**20:+9.2f}"
        )
    if top > 0 and len(rows) > top:
        print(f"… {len(rows) - top} more rows (use --top 0)")


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def _read_start_step(checkpoint: Path) -> int:
    """The checkpoint's next step, without materialising 1.7 GiB of weights."""
    try:
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, TypeError):
        # An older/foreign file may not be mmap-able; the fallback is slower but
        # this is a header read, not a model load.
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return int(ck.get("step", 0)) + 1


def _run_from_checkpoint(args: argparse.Namespace):
    from dormouse.config import RunConfig
    from dormouse.model import DormouseModel
    from dormouse.train.loop import train

    checkpoint = Path(args.checkpoint)
    snapshot = checkpoint.with_suffix(".config.toml")
    if not checkpoint.is_file() or not snapshot.is_file():
        raise FileNotFoundError(
            f"need both {checkpoint} and {snapshot}; pass --fresh for a preset run"
        )
    run = RunConfig.from_snapshot(snapshot.read_text(encoding="utf-8"))
    for key, why in (
        ("full_graph", "a captured step hides every kernel behind one replay"),
        ("opt_graph", "a captured optimizer stage hides every kernel behind one replay"),
        ("torch_compile", "the compiled call bypasses the Python seams"),
    ):
        if getattr(run.train, key, False):
            raise ValueError(f"{key}=true: {why}; this instrument refuses it")
    start_step = _read_start_step(checkpoint)
    if run.train.steps - args.total < start_step:
        raise ValueError(
            f"checkpoint has fewer than {args.total} steps left: "
            f"steps={run.train.steps}, next step={start_step}"
        )
    # Progress keys only: no config drift, no eval, no log, no checkpoint write.
    run = dataclasses.replace(
        run,
        train=dataclasses.replace(
            run.train,
            eval=False,
            eval_every=10**9,
            ckpt_every=10**9,
            log_every=10**9,
            json_log="",
        ),
    )
    model = DormouseModel(run.model, device=args.device)
    return run, model, train, True, str(checkpoint)


def _run_fresh(args: argparse.Namespace):
    from dormouse.config import load_run
    from dormouse.model import DormouseModel
    from dormouse.train.loop import train

    if not Path(args.config).is_file():
        raise FileNotFoundError(f"config {args.config} does not exist")
    run = load_run(
        args.config,
        train_overrides={
            "steps": args.total,
            "batch": 8,
            "seq_len": 1024,
            "eval": False,
            "eval_every": 10**9,
            "ckpt_every": 10**9,
            "log_every": 10**9,
            "json_log": "",
        },
    )
    for key, why in (
        ("full_graph", "a captured step hides every kernel behind one replay"),
        ("opt_graph", "a captured optimizer stage hides every kernel behind one replay"),
        ("torch_compile", "the compiled call bypasses the Python seams"),
    ):
        if getattr(run.train, key, False):
            raise ValueError(f"{key}=true: {why}; this instrument refuses it")
    model = DormouseModel(run.model, device=args.device)
    return run, model, train, False, f"fresh {args.config}"


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Resume/fresh, profile 8 steps, return the JSON document."""
    run, model, train, resume, source = (
        _run_from_checkpoint(args) if not args.fresh else _run_fresh(args)
    )
    device = torch.device(args.device)
    start_step = _read_start_step(Path(args.checkpoint)) if resume else 0

    seams, skipped = install_regions(model)
    clock = StepClock()
    step_seam: Seam | None = None
    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
            # Not optional: record_function spans and `evt.kernels` parents are
            # CPU events (`use_cpu=(CPU in activities)`). CUDA-only = no regions.
            torch.profiler.ProfilerActivity.CPU,
        ],
        record_shapes=False,
        profile_memory=True,
        schedule=torch.profiler.schedule(
            wait=0, warmup=args.warmup, active=args.measured, repeat=1
        ),
    )
    try:
        with prof:
            step_seam = install_step_hook(prof, args.total, clock)
            try:
                train(model, run, resume=resume, device=device)
            except _StopTraining:
                pass
    finally:
        restore_seams([step_seam] if step_seam is not None else [])
        restore_seams(seams)

    events = list(prof.events())
    rows = aggregate(collect_samples(events))
    families = region_family_rows(rows)
    steps = [e for e in events if str(getattr(e, "name", "")).startswith("ProfilerStep")]
    measured = len(steps) or args.measured
    wall_ms = clock.measured_ms(args.warmup, args.measured)
    if wall_ms:
        step_ms = sum(wall_ms) / len(wall_ms)
        wall_source = "cuda_events"
    else:
        step_ms = sum(e.time_range.elapsed_us() for e in steps) / measured / 1000.0
        wall_source = "profiler_steps"
    total_us = sum(row.us for row in rows)
    kernel_ms = total_us / measured / 1000.0
    device_events = [
        e
        for e in events
        if getattr(e, "device_type", None) == CUDA
        and not str(getattr(e, "name", "")).startswith("[memory]")
    ]
    busy_us = union_us(
        (e.time_range.start, e.time_range.end) for e in device_events if e.time_range
    )
    busy_ms = busy_us / measured / 1000.0
    launches = sum(
        1
        for e in events
        if getattr(e, "device_type", None) == CPU and getattr(e, "name", "") == "cudaLaunchKernel"
    )
    attributed = (
        100.0 * sum(row.us for row in rows if row.region != "other") / total_us
        if total_us > 0
        else 0.0
    )
    meta = {
        "checkpoint": source if resume else None,
        "config": None if resume else args.config,
        "start_step": start_step,
        "batch": run.train.batch,
        "seq_len": run.train.seq_len,
        "warmup": args.warmup,
        "measured": measured,
        "step_ms": step_ms,
        "step_ms_each": wall_ms,
        "step_wall_source": wall_source,
        "kernel_ms_per_step": kernel_ms,
        "gpu_busy_ms_per_step": busy_ms,
        "kernels_per_step": int(round(sum(row.count for row in rows) / measured)),
        "launches_per_step": int(round(launches / measured)),
        "total_us": total_us,
        "bytes": sum(row.bytes for row in rows),
        "attributed_pct": attributed,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "skipped_seams": skipped,
    }
    doc = payload(rows, families, meta)
    print_report(rows, families, meta, top=args.top)
    return doc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Per-kernel GPU time with module attribution for one training step.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="resume this .pt and its .config.toml snapshot (progress keys only)",
    )
    parser.add_argument("--fresh", action="store_true", help="do not resume; run --config")
    parser.add_argument("--config", default=DEFAULT_PRESET, help="preset for --fresh")
    parser.add_argument("--json", type=Path, default=None, help="write every row here")
    parser.add_argument(
        "--diff",
        nargs=2,
        type=Path,
        metavar=("BEFORE", "AFTER"),
        default=None,
        help="compare two JSONs, no card needed",
    )
    parser.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    parser.add_argument("--measured", type=int, default=MEASURED_STEPS)
    parser.add_argument("--top", type=int, default=30, help="0 prints every row")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.measured < 1:
        print("REFUSED: --warmup must be >= 0 and --measured >= 1", file=sys.stderr)
        return 2
    args.total = args.warmup + args.measured

    if args.diff is not None:
        before, after = args.diff
        try:
            print_diff(load_json(before), load_json(after), top=args.top)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"REFUSED: cannot read the diff inputs: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.device != "cuda":
        print(
            "REFUSED: this instrument profiles CUDA kernels; --device cuda is the only "
            "valid device (the CPU path has no kernels to attribute).",
            file=sys.stderr,
        )
        return 2
    try:
        why = preflight(args.device)
    except OSError as exc:  # nvidia-smi missing is the same answer: not ours
        why = f"preflight could not read the card: {exc}"
    if why:
        print(
            f"REFUSED: {why}\n"
            "ONE GPU PROCESS AT A TIME (AGENTS 1.5): nothing was measured, and this "
            "instrument will not share the card. Re-run during a checkpoint gap.",
            file=sys.stderr,
        )
        return 2
    if not torch.cuda.is_available():
        print(
            "REFUSED: nvidia-smi sees a free card but torch reports no CUDA device; "
            "install the gpu extra (`uv run --extra gpu`) or fix the driver.",
            file=sys.stderr,
        )
        return 2

    try:
        doc = run_profile(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if args.json is not None:
        write_json(args.json, doc)
        print(f"\njson: {args.json}")
    return 0


#: Tests monkeypatch this name; `_lane` owns the one check.
preflight = _lane_preflight


if __name__ == "__main__":
    sys.exit(main())
