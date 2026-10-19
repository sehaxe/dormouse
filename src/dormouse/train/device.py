"""Per-device capability probe and the compute policy it derives.

The port's precision decision is one global boolean (`ModelConfig.bf16`) and the
kernels assume it: the Triton MoE arm needs bf16 operands, FLA's chunked KDA
needs fp16 or bf16, and the port's forward has exactly two states — bf16 or
fp32. On one homogeneous card that is enough. On a home lab of mismatched cards
(a GTX 1650 next to an RTX 5060 Ti) it is not: sm_75 has fp16 tensor cores and
**no bf16**, so "run bf16 everywhere" would silently fall off the tensor-core
path on half the fleet.

This module is the missing layer: probe each device ONCE, derive a POLICY
(compute dtype, which kernels are legal, a memory budget), and let the caller
apply it. Three rules, in the repo's vocabulary:

* **the probe is metadata only** — `torch.cuda.get_device_properties` plus one
  `nvidia-smi` query for the PCIe link. No allocation, no kernel, no timing:
  it is legal to run beside a live training job, and it never enters the step.
* **an explicit dtype the device cannot honor is LOUD** — `dtype="bf16"` on
  sm_75 dies naming the card, its sm, and the escape, because a bf16 run that
  quietly became fp16 is a different experiment wearing the requested label.
* **a derived dtype is COUNTED** — `dtype="auto"` picks the best legal rung and
  returns the sentence that says which and why, for the training log.

The dtype ladder is the tensor-core ladder, from the CUDA compute-capability
table: fp16 from sm_53 (Maxwell), bf16 from sm_80 (Ampere — bf16 tensor cores
arrive with it). `docs/architecture/bf16-plan.md` and `f16-plan.md` own what the
port's forward does with each rung; this module owns only *which rung a device
may run*, and refuses loudly where the two disagree.

`dtype = "none"` (the default) means **this module is never called**: the
`bf16` flag decides as it always has, the snapshot round-trips unchanged, and a
single-GPU run's behaviour is byte-identical to its history.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch

from ..config import DTYPES, RunConfig
from ..instrument import loud

#: fp16 tensor cores arrive with Maxwell (sm_53); bf16 with Ampere (sm_80).
#: The CUDA compute-capability table, not a measurement of this box — the
#: probe's own readings (name, sm, memory) are measurements and are printed.
FP16_MIN_SM = 53
BF16_MIN_SM = 80

#: The compute-dtype ladder, best first. `auto` walks it and takes the first
#: rung the device supports, so a fleet resolves bf16 on the new card and fp16
#: on the 1650 in the same launch — which is the entire point of the layer.
COMPUTE_DTYPES = ("bf16", "fp16", "fp32")

#: Headroom left for the allocator, the cuBLAS workspaces and the eval pass
#: when the memory budget is derived from free VRAM. A budget is a BUDGET, not
#: a guarantee: the fit knob (stage 4) is what turns it into mechanisms.
DEFAULT_RESERVE = 512 << 20


@dataclass(frozen=True)
class DeviceCaps:
    """What one visible device IS — the probe's record, and the tests' fixture.

    Frozen, so a policy cannot be argued from a record someone mutated later.
    `pcie_gen`/`pcie_width` are the link's MAXIMUM gen/width (what the slot and
    the card can do); the CURRENT gen/width that `nvidia-smi` also reports is
    idle-downclocked on this box (measured 2026-10-19: max 3x16, current 3x8)
    and is not what a bandwidth estimate may use.
    """

    index: int
    name: str
    cc_major: int
    cc_minor: int
    total_mem: int
    #: Free VRAM in bytes, or `-1` when it is UNREADABLE. Measured 2026-10-19
    #: on this box: while another process holds the card, `get_device_properties`
    #: (a driver query) still answers but `mem_get_info` (which needs a CUDA
    #: context) dies with `cudaErrorDevicesUnavailable` — the driver refuses a
    #: second context at all. `-1` is that state, and it is not a number.
    free_mem: int
    pcie_gen: int | None = None
    pcie_width: int | None = None
    #: "cuda" (the ladder derives from sm) or "cpu" (no tensor cores; both
    #: rungs are served by torch's CPU kernels, so the ladder is advisory and
    #: `auto` picks bf16 — what the presets run).
    kind: str = "cuda"

    @property
    def sm(self) -> int:
        return self.cc_major * 10 + self.cc_minor

    @property
    def bf16(self) -> bool:
        return self.sm >= BF16_MIN_SM if self.kind == "cuda" else True

    @property
    def fp16(self) -> bool:
        return self.sm >= FP16_MIN_SM if self.kind == "cuda" else True

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"

    def supports(self, dtype: str) -> bool:
        """Whether this device can run `dtype` on its tensor cores."""
        if dtype == "bf16":
            return self.bf16
        if dtype == "fp16":
            return self.fp16
        if dtype == "fp32":
            return True
        loud(
            f"dtype={dtype!r} is not a compute rung. Known: {list(COMPUTE_DTYPES)}. "
            f"Escape: name one of those, or land the rung in "
            f"`device.COMPUTE_DTYPES` before anything can ask for it."
        )

    def line(self) -> str:
        """One line, the shape the training log reads."""
        link = (
            f"pcie=gen{self.pcie_gen}x{self.pcie_width}"
            if self.pcie_gen is not None and self.pcie_width is not None
            else "pcie=unknown"
        )
        if self.is_cuda:
            mem = (
                f"{self.free_mem / 2**30:.1f}/{self.total_mem / 2**30:.1f}GiB free"
                if self.free_mem >= 0
                else f"free=unknown/{self.total_mem / 2**30:.1f}GiB total"
            )
        else:
            mem = "host"
        return f"device={self.index} {self.name!r} sm_{self.sm} {mem} {link}"


@dataclass(frozen=True)
class DevicePolicy:
    """The derived decision for one device: dtype, kernel legality, budget."""

    caps: DeviceCaps
    compute_dtype: str
    #: WHY this dtype — the COUNTED sentence (auto) or the request's own name.
    reason: str
    #: Bytes the run may use for weights + optimizer state + activations, or
    #: `None` on a device with no VRAM to budget (the CPU arm).
    memory_budget: int | None
    #: The REQUEST that produced this policy (`auto` or an explicit rung). The
    #: difference matters at the port's half of the contract: an explicit
    #: `fp16` that the port cannot run is LOUD, while `auto` resolving fp16 and
    #: being unable to run it is a COUNTED fp32 fallback for that rank.
    requested: str = "auto"

    @property
    def moe_triton(self) -> bool:
        """The Triton MoE arm's operand rule: bf16 only (today)."""
        return self.compute_dtype == "bf16"

    @property
    def fla_kda(self) -> bool:
        """FLA's chunked KDA takes fp16/bf16; fp32 falls to the SDPA arm."""
        return self.compute_dtype in ("bf16", "fp16")

    def line(self) -> str:
        """The COUNTED line: the device, the choice, why, and what it enables."""
        budget = (
            f"{self.memory_budget / 2**30:.1f}GiB" if self.memory_budget is not None else "none"
        )
        return (
            f"{self.caps.line()} compute={self.compute_dtype} why={self.reason} "
            f"budget={budget} moe_triton={int(self.moe_triton)} "
            f"fla_kda={int(self.fla_kda)}"
        )


def cpu_caps() -> DeviceCaps:
    """The CPU arm's record: host memory, no VRAM, no tensor-core ladder.

    torch's CPU kernels serve both bf16 and fp16, so the ladder is advisory
    here: `auto` resolves bf16 (what every preset runs). With the `dtype` knob
    off, the `bf16` flag decides as it always has — this record exists so a
    heterogeneous run can PRINT what the CPU arm is, not to change it.
    """
    return DeviceCaps(
        index=-1, name="cpu", cc_major=0, cc_minor=0, total_mem=0, free_mem=0, kind="cpu"
    )


def _pcie(index: int) -> tuple[int | None, int | None]:
    """`nvidia-smi`'s link MAX gen/width for device `index` — best effort.

    Best effort on purpose: a container without `nvidia-smi`, a driver that
    answers a different query, a machine with no card — none of those make the
    probe fail, they make the link unknown (`None`), which every consumer must
    treat as "cannot budget the link" rather than as a number.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=pcie.link.gen.max,pcie.link.width.max",
                "--format=csv,noheader,nounits",
                "-i",
                str(index),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip()
        gen, width = (int(v) for v in out.splitlines()[0].split(","))
        return gen, width
    except Exception:  # noqa: BLE001 - "unknown" is the designed answer
        return None, None


def probe_device(index: int = 0) -> DeviceCaps:
    """One device's capability record — metadata only (no alloc, no kernel).

    `torch.cuda.get_device_properties` opens the CUDA context (a few hundred MB
    of driver bookkeeping) and returns the static properties; `mem_get_info`
    reads the driver's free-memory counter. Neither launches a kernel nor
    touches the model's memory, which is why the probe is legal beside a live
    run (AGENTS §1.5 is about two TRAINING processes, not about a query).
    """
    if not torch.cuda.is_available():
        loud(
            "probe_device on a box with no CUDA: there is no device to probe. "
            "Escape: probe_device is only for the cuda arm — use `cpu_caps()` for "
            "the CPU arm, or install a CUDA torch."
        )
    props = torch.cuda.get_device_properties(index)
    try:
        free, total = torch.cuda.mem_get_info(index)
    except Exception:  # noqa: BLE001 - "held by another process" is the designed answer
        free, total = -1, props.total_memory
    gen, width = _pcie(index)
    return DeviceCaps(
        index=index,
        name=props.name,
        cc_major=props.major,
        cc_minor=props.minor,
        total_mem=int(total),
        free_mem=int(free),
        pcie_gen=gen,
        pcie_width=width,
    )


def resolve_policy(
    caps: DeviceCaps, requested: str = "auto", *, reserve: int = DEFAULT_RESERVE
) -> DevicePolicy:
    """The one decision function: `caps` + a request -> a policy, LOUD or COUNTED.

    `requested="auto"` walks [`COMPUTE_DTYPES`] and takes the first rung the
    device supports, returning the COUNTED reason (a reader of the log learns
    the card had no bf16, not just that fp16 ran).

    An explicit rung the device cannot honor is LOUD — naming the device, its
    sm and the escape — because the alternative (falling back quietly) turns a
    bf16 experiment into an fp16 one under the same flags.
    """
    if requested == "none":
        loud(
            "dtype='none' is the OFF switch (the `bf16` flag decides, no probe); "
            "resolving a policy for it is a wiring defect. Escape: only call "
            "resolve_policy when the run asked for a dtype."
        )
    if requested not in ("auto", *COMPUTE_DTYPES):
        loud(
            f"dtype={requested!r} is not a request. Known: {list(DTYPES)} — `none` "
            f"leaves the `bf16` flag deciding (the default), `auto` picks the best "
            f"rung the device supports, and an explicit rung is refused when the "
            f"device cannot run it. Escape: name one of those."
        )
    if requested == "auto":
        supported = [d for d in COMPUTE_DTYPES if caps.supports(d)]
        assert supported, "fp32 is always supported; an empty ladder is a defect"
        dtype = supported[0]
        reason = f"auto: bf16={int(caps.bf16)} fp16={int(caps.fp16)}"
    else:
        if not caps.supports(requested):
            loud(
                f"dtype={requested} requested but {caps.line()} does not support it: "
                f"bf16 tensor cores arrive with Ampere (sm_{BF16_MIN_SM}), fp16 with "
                f"Maxwell (sm_{FP16_MIN_SM}), and this device is sm_{caps.sm}. A run "
                f"that quietly became fp32 is a different experiment under the "
                f"requested label. Escape: dtype=auto (pick this device's best), "
                f"dtype=fp16, or dtype=fp32."
            )
        dtype = requested
        reason = f"requested {requested}"
    policy_request = requested  # the raw request, for apply_policy's contract
    budget = None
    if caps.is_cuda and caps.free_mem >= 0:
        budget = caps.free_mem - reserve
        if budget <= 0:
            loud(
                f"{caps.line()} has {caps.free_mem / 2**30:.1f} GiB free and the "
                f"budget reserves {reserve / 2**30:.1f} GiB: nothing fits. Another "
                f"process holds the card (AGENTS §1.5, one heavy thing at a time). "
                f"Escape: free the card, or lower the reserve if you know what the "
                f"run needs."
            )
    elif caps.is_cuda:
        # The card is held in a mode that refuses a second CUDA context (measured
        # on this box: `mem_get_info` dies while a training job runs, while the
        # driver's property query still answers). A budget cannot be derived
        # from a number that does not exist, so the reason says so.
        reason = f"{reason}; budget=unknown (free VRAM unreadable: another process holds the card)"
    return DevicePolicy(
        caps=caps,
        compute_dtype=dtype,
        reason=reason,
        memory_budget=budget,
        requested=policy_request,
    )


def apply_policy(
    run: RunConfig,
    policies: Sequence[DevicePolicy],
    *,
    rank: int | None = None,
    log=None,
) -> RunConfig:
    """Map a resolved policy onto the run's model half — LOUD on what the port cannot honor.

    This is the seam between "which rung the DEVICE may run" and "which rungs
    the PORT's forward actually has":

    * `bf16` -> `bf16 = true` (the plan the port ships);
    * `fp32` -> `bf16 = false` (the CPU arm's and the control's shape);
    * `fp16` -> the port's forward has no fp16 path (the f16 plan is storage
      only). An EXPLICIT `dtype=fp16` is **LOUD** — the request cannot be
      honored. An `auto` policy that resolved fp16 is a **COUNTED fp32
      fallback for that rank**: the user asked for the best available, fp32 IS
      the best the PORT can run on that card, and a mixed fleet therefore runs
      bf16 on the strong card and fp32 on the weak one — which is the point.
      `log` is required for the fallback; without it the function refuses,
      because an unannounced fallback is the failure class this repo bans.
    * `moe_kernel = "triton"` on a policy whose `moe_triton` is false -> LOUD:
      the Triton MoE arm's operands are bf16 today.

    `rank=None` is the single-device path: every policy must agree, because one
    `ModelConfig` carries one dtype. A heterogeneous fleet applies per rank
    (`rank=i`) inside the dist entry, where each process has its own config.
    """
    assert run.model.dtype != "none", (
        "apply_policy with dtype='none': the caller's off-switch check failed"
    )
    if rank is None:
        chosen = {p.compute_dtype for p in policies}
        if len(chosen) > 1:
            loud(
                f"apply_policy got a heterogeneous fleet ({sorted(chosen)}): one "
                f"ModelConfig carries one dtype, so a mixed fleet is applied per "
                f"rank inside the dist entry (rank=i), not here. Escape: run the "
                f"fleet path, or pick a dtype every device supports."
            )
        policy = policies[0]
    else:
        policy = policies[rank]
    dtype = policy.compute_dtype
    if dtype == "fp16":
        if policy.requested != "auto":
            loud(
                f"dtype=fp16 requested, and {policy.caps.line()} supports it — but the "
                f"port's forward has no fp16 path: the f16 plan "
                f"(docs/architecture/f16-plan.md) is storage-only so far, and a forward "
                f"cast to fp16 would run unmeasured numerics (the bf16 plan's "
                f"mixed-dtype rules were derived for bf16). Escape: dtype=auto (fp32 on "
                f"this card, COUNTED), dtype=fp32, or land the fp16 forward with its "
                f"gates and delete this refusal."
            )
        if log is None:
            loud(
                f"{policy.caps.line()} auto-resolves fp16 and the port has no fp16 "
                f"forward, so the run must COUNTED-fall back to fp32 — but apply_policy "
                f"was given no log to announce it with. Escape: pass log=<the run's log>."
            )
        log(
            f"{policy.caps.line()} auto-resolves fp16 (bf16={int(policy.caps.bf16)}), but "
            f"the port's forward has no fp16 path (docs/architecture/f16-plan.md is "
            f"storage-only): COUNTED fallback to fp32 for this rank. Escape: land the "
            f"fp16 forward, or accept fp32 on this card."
        )
        dtype = "fp32"
    assert dtype in COMPUTE_DTYPES, dtype
    if run.model.moe_kernel == "triton" and dtype != "bf16":
        loud(
            f"moe_kernel='triton' with compute={policy.compute_dtype} on "
            f"{policy.caps.line()}: the Triton grouped-GEMM arm takes bf16 operands "
            f"today, so it cannot serve this device. Escape: moe_kernel='padded' "
            f"(the reference arm), or dtype=bf16 on a card that supports it."
        )
    return replace(run, model=replace(run.model, bf16=dtype == "bf16"))
