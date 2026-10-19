"""Single-node multi-process data parallelism: DDP first, FSDP2 where it fits.

The trainer is a single-process loop (`train/loop.py`). This module adds the
multi-process shape around it, using nothing but torch's own distributed stack
(no new dependency): one process per GPU, `gloo` on the CPU arm and `nccl` on
the CUDA arm, and the loop itself unchanged except for the four seams a
data-parallel step needs — per-rank data, gradient reduction (inside `backward`,
via DDP/FSDP), one collective agreement on the NaN firewall's decision, and a
gradient scaling that keeps the global batch well defined when the ranks carry
different amounts of data (stage 3, `parse_capacity`).

**What is VERIFIED on this box, and what is not.** This machine has one GPU, so
nothing here has been run on a real multi-GPU fleet. What HAS been run:

* the 2-rank CPU/gloo gate (`tools/dist_gate.py`, `tests/test_distributed.py`):
  20 steps, loss curve against a single-process run at the same global batch,
  identical parameters across ranks, and the hand-computed gradient
  equivalence for unequal per-rank batches (stage 3);
* the FSDP2 integration probe, whose result is a REFUSAL, not a feature: on
  torch 2.13.0+cu129, `fully_shard` + the port's `MixOptimizer` dies at
  `opt.step()` with

      RuntimeError: aten._foreach_add_.List got mixed torch.Tensor and
      DTensor, need to convert all torch.Tensor to DTensor

  (measured 2026-10-19, 2 ranks, gloo, CPU — the Muon momentum buffer is one
  stacked `torch.Tensor` per shape group and the gradients are DTensors).
  The honest consequence is in `refuse_incompatible`: FSDP2 is available for
  the `opt="adamw"` control arm and LOUDLY refused for the Muon arm until a
  shard-aware Muon lands. DDP is the path the gates prove.

**The PCIe arithmetic** (`allreduce_seconds`, quoted in
`docs/architecture/multi-gpu.md`): a ring all-reduce moves `2(P-1)/P` payloads
per rank, so on PCIe gen3 x16 (15.76 GB/s theoretical, `LINK_EFFICIENCY`
assumed 0.8) a 138M-parameter bf16 gradient (276 MB) costs ~22 ms at 2 ranks
and ~33 ms at 4; the 1.07B shape (2.14 GB) costs ~170 ms / ~255 ms. Both are
COMPUTED from the link table, not measured — the two-card test is the
measurement, and `tools/dist_gate.py --report-link` prints the numbers with
their inputs so the owner can replace them with a reading.

**Data layout.** Rank `r` starts at byte `sum(b_i for i < r) * (seq+1)` and
advances `b_r * (seq+1)` bytes per step, so with equal batches the union of the
ranks' windows at step `s` is exactly the contiguous block a single process at
the summed batch reads at step `s`. That is what makes the gate's loss-curve
comparison an equality and not a hope. The cost is stated rather than hidden:
because each rank advances by its OWN window while the fleet's footprint per
step is `Σb`, consecutive ranks' windows overlap ACROSS steps (rank 1's step
`s` is rank 0's step `s+1`), so the fleet re-reads each block ~P times over a
run. Correct, and wasteful for a long production run — the strided sampler
(rank `r` reads blocks `r, r+P, …`) is the named follow-up, and it is not
built here because it needs a stream seek that the ring buffers do not have.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace

import torch

from ..config import TrainConfig
from ..instrument import Counters, loud
from .device import DevicePolicy, apply_policy, cpu_caps, probe_device, resolve_policy

#: The process-group backends. `nccl` is the CUDA arm's (gradients stay on the
#: device); `gloo` is the CPU arm's and the fallback a CUDA run may ASK for
#: (COUNTED: it stages every gradient through host memory).
BACKENDS = ("auto", "gloo", "nccl")

#: The sharding arms. `none` = DDP (replicate, reduce-scatter-free), `fsdp` =
#: FSDP2 `fully_shard` (shard the parameters, all-gather per forward).
SHARDS = ("none", "fsdp")

#: The gradient-reduce dtypes. `auto` = the fleet's compute dtype (bf16 on a
#: bf16 fleet, fp32 on the CPU arm), which is what the policy layer already
#: decided; `bf16`/`fp16` compress the communication, `fp32` is the exact
#: reference the gate compares against.
REDUCE_DTYPES = ("auto", "bf16", "fp16", "fp32")

#: PCIe per-lane, per-direction GB/s (the standard table: 8b/10b for gen1/2,
#: 128b/130b above). The link's MAX gen/width is what these multiply; the
#: `current` gen/width nvidia-smi reports is idle-downclocked.
PCIE_GBPS_PER_LANE = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.877}

#: The fraction of the theoretical link rate a large transfer achieves. An
#: ASSUMPTION (0.8), not a measurement on this box — it exists so the estimate
#: errs on the slow side, and the two-card test replaces it.
LINK_EFFICIENCY = 0.8


def link_gbps(gen: int | None, width: int | None) -> float | None:
    """Theoretical one-direction bandwidth of a PCIe link, or None if unknown."""
    if gen is None or width is None or gen not in PCIE_GBPS_PER_LANE:
        return None
    return PCIE_GBPS_PER_LANE[gen] * width


def allreduce_seconds(
    payload_bytes: int, ranks: int, gbps: float, *, efficiency: float = LINK_EFFICIENCY
) -> float:
    """Ring all-reduce time: `2(P-1)/P · payload / (efficiency · link)`.

    The `2(P-1)/P` is the ring algorithm's traffic per rank (one reduce-scatter
    plus one all-gather of a `1/P` slice each, so `2(P-1)/P` payloads leave and
    enter each rank); at P=2 it is exactly one payload. The link is
    ONE-DIRECTION, which is what a ring all-reduce uses at a time.
    """
    assert payload_bytes >= 0 and ranks >= 1 and gbps > 0
    return (2.0 * (ranks - 1) / ranks) * payload_bytes / (efficiency * gbps * 1e9)


def bucket_mb_for(param_bytes: int, *, min_mb: int = 4, max_mb: int = 128) -> int:
    """DDP's bucket cap, sized to the gradient payload.

    DDP reduces one bucket at a time as its gradients become ready, so the
    payload is cut so the first all-reduce overlaps the rest of the backward
    and the LAST piece is short enough not to stall the step. torch's 25 MB
    default is an NVLink-era number; the constant here is payload/8, clamped —
    a STARTING POINT to sweep on two cards, not a measured optimum (there is
    no PCIe sweep in this repo's record).
    """
    payload_mb = param_bytes / (1 << 20)
    return int(max(min_mb, min(max_mb, round(payload_mb / 8))))


@dataclass(frozen=True)
class RankShape:
    """One rank's micro-batch: the capacity profile's entry (stage 3)."""

    batch: int
    seq_len: int

    @property
    def tokens(self) -> int:
        return self.batch * self.seq_len

    def line(self) -> str:
        return f"{self.batch}x{self.seq_len}"


def parse_capacity(
    spec: str, *, world_size: int, cfg: TrainConfig, max_seq_len: int
) -> list[RankShape]:
    """`dist_capacity` -> one [`RankShape`] per rank, LOUD on anything else.

    The spec is a comma list of `B` or `BxT` entries, one per rank, in rank
    order: `"4,2"` is rank 0 at batch 4 and rank 1 at batch 2. `""` (the
    default) is `cfg.batch`/`cfg.seq_len` on every rank — the homogeneous
    shape, and the one the equal-batch gate runs. `"auto"` is resolved before
    this function is called (`init_parallel` measures a throughput index and
    splits the global batch with [`suggest_capacity`]).

    The trade this profile exists for: with a synchronous all-reduce the
    slowest rank sets the step time, so the weak card must be given LESS work
    per step, not more. A 1650 at ~1/4 of a 5060 Ti's rate should carry ~1/4 of
    the tokens — then both finish their local backward at roughly the same
    moment and the all-reduce's wait is short.
    """
    if spec == "":
        return [RankShape(cfg.batch, cfg.seq_len) for _ in range(world_size)]
    entries = [e.strip() for e in spec.split(",") if e.strip()]
    if len(entries) != world_size:
        loud(
            f"dist_capacity={spec!r} has {len(entries)} entries for a {world_size}-rank "
            f"fleet: the profile is per RANK, one entry each, in rank order. Escape: "
            f"give {world_size} entries (`4,2`), or leave it empty for `{cfg.batch}` "
            f"on every rank."
        )
    shapes: list[RankShape] = []
    for i, entry in enumerate(entries):
        try:
            b, _, t = entry.partition("x")
            shape = RankShape(int(b), int(t) if t else cfg.seq_len)
        except ValueError:
            loud(
                f"dist_capacity entry {i}={entry!r} is not `B` or `BxT`. Escape: "
                f"`4,2` (batch 4 and 2 at seq_len {cfg.seq_len}) or `4x512,2x256`."
            )
        if shape.batch < 1 or shape.seq_len < 1:
            loud(f"dist_capacity entry {i}={entry!r}: batch and seq_len must be >= 1")
        if shape.seq_len > max_seq_len:
            loud(
                f"dist_capacity entry {i}={entry!r}: seq_len {shape.seq_len} exceeds "
                f"the model's max_seq_len {max_seq_len}. Escape: shorten it, or raise "
                f"the model's max_seq_len (a model change, not a knob)."
            )
        shapes.append(shape)
    return shapes


def rank_byte_offset(shapes: list[RankShape], rank: int) -> int:
    """The byte offset rank `rank` STARTS at: the windows of the ranks before it.

    With equal shapes this is `rank · b · (seq+1)` and the fleet's union at step
    `s` is the contiguous block one process at `Σb` reads at step `s` — the
    property the equal-batch loss gate rests on. With unequal shapes the
    initial ranges are still disjoint, which is all the unequal-batch gate
    needs. (Across steps the windows overlap; see the module docstring's
    "Data layout" for the cost and the follow-up.)
    """
    assert 0 <= rank < len(shapes), f"rank {rank} outside {len(shapes)}"
    return sum(s.batch * (s.seq_len + 1) for s in shapes[:rank])


def suggest_capacity(indices: list[float], *, total_batch: int, seq_len: int) -> list[RankShape]:
    """Split `total_batch` over the ranks in proportion to their throughput index.

    The index is a measured relative rate (see `throughput_index`); the split
    keeps the GLOBAL batch (`total_batch × seq_len` tokens) fixed so the run's
    objective does not move with the fleet's shape, and every rank gets at
    least one sample. Largest-remainder rounding keeps the total exact.
    """
    assert len(indices) >= 1 and total_batch >= len(indices), (
        "the global batch must give every rank at least one sample"
    )
    assert all(i > 0 for i in indices), f"throughput indices must be positive, got {indices}"
    share = [total_batch * i / sum(indices) for i in indices]
    out = [max(1, math.floor(s)) for s in share]
    left = total_batch - sum(out)
    order = sorted(range(len(indices)), key=lambda i: share[i] - math.floor(share[i]), reverse=True)
    for i in order:
        if left <= 0:
            break
        out[i] += 1
        left -= 1
    while left > 0:
        out[0] += 1
        left -= 1
    return [RankShape(b, seq_len) for b in out]


def throughput_index(device: torch.device, *, n: int | None = None) -> float:
    """A fixed matmul's rate on `device` — the fleet's RELATIVE speed index.

    One shape, one dtype, warmed then timed with a device sync at both ends:
    the number is only meaningful against another rank's number from the same
    call, which is exactly how `dist_capacity="auto"` uses it. It runs once, at
    startup, before the model exists — it is not in any step.
    """
    import time

    if n is None:
        n = 4096 if device.type == "cuda" else 512
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    a @ b  # warm the allocator / autotuner
    if device.type == "cuda":
        torch.cuda.synchronize()
    iters = 3
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = max(time.perf_counter() - t0, 1e-9)
    return 2.0 * n**3 * iters / dt


def choose_backend(
    requested: str, device_type: str, *, nccl_ok: bool, gloo_ok: bool, log=None
) -> str:
    """The process-group backend, LOUD when the CUDA arm has no NCCL.

    The refusal the brief names: a GPU run whose NCCL is missing must STOP, not
    quietly fall back to gloo — gloo stages every gradient through host memory,
    which on PCIe turns a step into a memcpy benchmark and reads like a
    mysterious slowdown instead of a missing library. An EXPLICIT `gloo` on a
    CUDA fleet is allowed (debugging) but COUNTED.
    """
    if requested not in BACKENDS:
        loud(
            f"dist_backend={requested!r} is not a backend. Known: {list(BACKENDS)} — "
            f"`auto` is nccl on a CUDA fleet and gloo on the CPU arm. Escape: name "
            f"one of those."
        )
    if device_type == "cuda":
        if requested == "gloo":
            assert log is not None, "an explicit gloo on CUDA is COUNTED and needs a log"
            log(
                "dist_backend=gloo on a CUDA fleet: every gradient is staged through "
                "host memory (COUNTED). Correct, and PCIe-bound at best — this is a "
                "debugging arm, not a speed one."
            )
            return "gloo"
        if not nccl_ok:
            loud(
                "a CUDA fleet without NCCL: torch.distributed.is_nccl_available() is "
                "False, so the gradients have nowhere to go except through gloo's "
                "host staging (every gradient crosses PCIe twice per step). This is "
                "refused rather than silently taken because the fallback computes the "
                "right answer at a fraction of the speed. Escape: install a "
                "NCCL-capable torch (the gpu extra), run --device cpu (the gloo arm "
                "the gates use), or pass dist_backend=gloo to accept the host staging "
                "knowingly."
            )
        return "nccl"
    if requested == "nccl":
        loud(
            "dist_backend=nccl on the CPU arm: NCCL needs CUDA devices. Escape: "
            "dist_backend=gloo (the default here), or run on the GPU."
        )
    if not gloo_ok:
        loud("the CPU arm has no gloo: torch.distributed.is_gloo_available() is False")
    return "gloo"


class Parallel:
    """The seam `train()` talks to: rank identity, shapes, reduction, agreement.

    One object per process. It carries no model state: [`shard`] runs before
    the optimizer (FSDP2, so its state is sharded) and [`replicate`] after it
    (DDP, over the raw parameters the optimizer holds), and the loop's
    per-step hooks are [`shape`], [`loss_scale`] and [`all_finite`].
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        backend: str,
        device: torch.device,
        shapes: list[RankShape],
        reduce_dtype: str,
        shard_kind: str,
        policy: DevicePolicy,
        policies: list[DevicePolicy],
        bucket_mb: int,
        counters: Counters | None = None,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.device = device
        self.shapes = shapes
        self.reduce_dtype = reduce_dtype
        self.shard_kind = shard_kind
        self.policy = policy
        self.policies = policies
        self.bucket_mb = bucket_mb
        self.comm_hook = _comm_hook(reduce_dtype)
        self.counters = counters if counters is not None else Counters()
        self._flag = torch.zeros(1, dtype=torch.int32, device=device)
        #: The fleet-loss reduction's buffer, float64 so the SUM over ranks does
        #: not lose the low bits of a 5.7-scale CE. Read at the LOG cadence only.
        self._loss = torch.zeros(1, dtype=torch.float64, device=device)
        self._closed = False

    # -- identity --------------------------------------------------------
    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def shape(self) -> RankShape:
        return self.shapes[self.rank]

    @property
    def global_tokens(self) -> int:
        return sum(s.tokens for s in self.shapes)

    @property
    def loss_scale(self) -> float:
        """The factor that turns DDP's mean-of-rank-means into the token mean.

        DDP divides the summed gradients by `world_size`; the objective is the
        mean over the GLOBAL batch's tokens. So rank `r`'s loss is scaled by
        `world_size · t_r / Σt` before `backward`, and the reducer's division
        lands on `Σ (t_r/Σt)·g_r` exactly. With equal shapes the factor is 1.0
        (the homogeneous case, and the number every single-GPU run has).
        """
        return self.world_size * self.shape.tokens / self.global_tokens

    @property
    def torch_reduce_dtype(self) -> torch.dtype:
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
            self.reduce_dtype
        ]

    def byte_offset(self) -> int:
        """Where THIS rank's stream starts (the disjoint per-rank byte ranges)."""
        return rank_byte_offset(self.shapes, self.rank)

    def line(self) -> str:
        return (
            f"dist rank={self.rank}/{self.world_size} backend={self.backend} "
            f"shard={self.shard_kind} shape={self.shape.line()} "
            f"global_tokens={self.global_tokens} reduce={self.reduce_dtype} "
            f"bucket={self.bucket_mb}MiB loss_scale={self.loss_scale:.4f} "
            f"compute={self.policy.compute_dtype}"
        )

    def apply(self, run):
        """The rank's own dtype policy on the run's model half.

        A dist run always resolves per rank, so a `dtype="none"` run is recorded
        as `dtype="auto"` — the truth of what happened — and the same refusals
        (`apply_policy`) run on every rank.
        """
        if run.model.dtype == "none":
            run = replace(run, model=replace(run.model, dtype="auto"))
        return apply_policy(run, self.policies, rank=self.rank)

    # -- the loop's per-step hooks ---------------------------------------
    def all_finite(self, flag: bool) -> bool:
        """One collective: every rank skips a step iff ANY rank saw a non-finite loss.

        Without this the NaN firewall desynchronizes the fleet — a rank that
        skips `backward` never enters the reducer, and the other rank's next
        backward blocks on a collective that will never come (a hang, not an
        error). MAX over a 1-element flag is the agreement; the loop already
        pays one host read per step for the loss, so this adds a small
        all-reduce, not a new class of sync.
        """
        import torch.distributed as dist

        self._flag.fill_(0 if flag else 1)
        dist.all_reduce(self._flag, op=dist.ReduceOp.MAX)
        return bool(int(self._flag.item()) == 0)

    def fleet_loss(self, local: float) -> float:
        """The log line's loss on a fleet: the token-weighted mean of the ranks'.

        Called at the LOG cadence on every rank (the line already syncs there):
        `Σ t_r·l_r / Σt`. Without it rank 0's `loss=` is its own micro-batch's
        mean and a reader would compare a fleet run against a single-process run
        at the wrong window — the same class of error as an eval window without
        its batch. It decides nothing; NaN propagates as NaN, which is the
        honest log of a non-finite step.
        """
        import torch.distributed as dist

        self._loss.fill_(local * self.shape.tokens)
        dist.all_reduce(self._loss, op=dist.ReduceOp.SUM)
        return float(self._loss.item() / self.global_tokens)

    # -- wrapping --------------------------------------------------------
    def refuse_incompatible(self, run, *, resume: bool) -> None:
        """Every combination this module has NOT verified stops here, LOUD.

        Each refusal names a measured or named cause; none of them is a
        preference. The list is the honest boundary of the 2-rank CPU gate.
        """
        cfg, model = run.train, run.model
        if self.shard_kind == "fsdp":
            if cfg.opt != "adamw":
                loud(
                    f"dist_shard=fsdp with opt={cfg.opt!r}: the port's Muon momentum is "
                    f"ONE stacked torch.Tensor per shape group while FSDP2's gradients "
                    f"are DTensors, and the batched step dies with `aten._foreach_add_"
                    f".List got mixed torch.Tensor and DTensor` (measured 2026-10-19, 2 "
                    f"ranks, gloo, CPU). FSDP2 is available for the AdamW control arm "
                    f"and refused here until a shard-aware Muon lands. Escape: "
                    f"dist_shard=none (DDP, the path the gates prove), or opt=adamw."
                )
            if resume:
                loud(
                    "dist_shard=fsdp with resume: an FSDP2 state dict is DTensors, and "
                    "the port's single-file checkpoints are not — the resume path "
                    "(`load_model_state`) would refuse the keys or silently need "
                    "`torch.distributed.checkpoint`. Unverified here (it needs two "
                    "cards to be worth verifying). Escape: dist_shard=none, or start "
                    "a fresh run."
                )
            if cfg.ckpt_every <= cfg.steps:
                loud(
                    f"dist_shard=fsdp would checkpoint at step {cfg.ckpt_every} of "
                    f"{cfg.steps}: an FSDP2 model's state dict is DTensor-sharded, not "
                    f"the port's single-file format, so rank 0 would write a file no "
                    f"single-process loader can read. Escape: dist_shard=none, or "
                    f"ckpt_every > steps."
                )
        if cfg.engram_ram:
            loud(
                "engram_ram=true with dist: the host n-gram tables are NOT parameters "
                "(`Engram.to_host` deletes them), so the reducer never sees them — "
                "each rank would train its OWN memory and the fleet would diverge into "
                "N different models that look synchronized. Escape: engram_ram=false "
                "(the in-VRAM tables ARE parameters and DO reduce), or land a "
                "row-gather for the host tables first."
            )
        if cfg.opt_graph:
            loud(
                "opt_graph=true with dist: the CUDA-graph arm captures the optimizer "
                "stage and pins `.grad` buffers, and its interaction with the "
                "reducer's buckets is unverified here. Escape: opt_graph=false."
            )
        if cfg.torch_compile:
            loud(
                "torch_compile=true with dist: an OptimizedModule's state dict carries "
                "the `_orig_mod.` prefix and the interaction with the reducer is "
                "unverified here. Escape: torch_compile=false."
            )
        if cfg.mix_plan:
            loud(
                f"mix_plan={cfg.mix_plan} with dist: the mixture's per-source epoch caps "
                f"and its per-rank byte offset are a single-process construct "
                f"(`train/mixture.py`). Escape: a single `data` file or shard "
                f"directory, or run single-process."
            )
        if model.moe_topk > 0 and model.moe_capacity > 0:
            loud(
                f"dist with moe_topk={model.moe_topk} and moe_capacity={model.moe_capacity}: "
                f"the capacity-padded dispatch DROPS the assignments past its "
                f"per-expert cap, so WHICH tokens reach an expert depends on the "
                f"window's composition — splitting that window across ranks changes "
                f"the model, and the fleet's gradient is not the single-process "
                f"gradient (the same defect `config._validate_segment` refuses for "
                f"`segment_len`). Escape: moe_capacity=0 (DROPLESS: the routing is per "
                f"position by construction and this was measured batch-invariant to "
                f"1.2e-6 on the gate's fixture, 2026-10-19)."
            )
        if model.moe_lb_coef != 0.0:
            loud(
                f"dist with moe_lb_coef={model.moe_lb_coef}: the LFB controller is "
                f"FORWARD-MUTATING non-parameter state (`lfb_counts`/`lfb_bias` move "
                f"during every routed forward), so each rank would steer its own "
                f"controller and the ranks' models would diverge while the "
                f"PARAMETERS stayed synchronized — the worst shape of silent "
                f"divergence. Escape: moe_lb_coef=0, or land a fleet reduction of the "
                f"counts first (a follow-up, not a knob)."
            )
        if cfg.dist_reduce in ("bf16", "fp16") and self.backend == "gloo":
            loud(
                f"dist_reduce={cfg.dist_reduce} on gloo: the compress hooks exist on "
                f"any backend, but the port's CPU gate measures the EXACT fp32 "
                f"reduction, and a compressed reduce on the CPU arm buys nothing (the "
                f"link is host memory). Escape: dist_reduce=auto (fp32 on this arm), "
                f"or run on the CUDA fleet."
            )

    def shard(self, model: torch.nn.Module) -> torch.nn.Module:
        """FSDP2 `fully_shard` — BEFORE the optimizer, so its state is sharded."""
        if self.shard_kind != "fsdp":
            return model
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        mesh = init_device_mesh(self.device.type, (self.world_size,))
        fully_shard(
            model,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(reduce_dtype=self.torch_reduce_dtype),
        )
        return model

    def replicate(self, model: torch.nn.Module) -> torch.nn.Module:
        """DDP — AFTER the optimizer, over the raw parameters it holds.

        `find_unused_parameters=True` is on because the port's parameter set
        can carry arms a given step does not touch (an aux head at weight 0),
        and DDP's default would raise on the first such step. It costs one
        graph traversal per iteration — on the start line rather than left for
        the reader to discover.
        """
        if self.shard_kind != "none":
            return model
        from torch.nn.parallel import DistributedDataParallel as DDP

        if not self.bucket_mb:
            width = 2 if self.reduce_dtype in ("bf16", "fp16") else 4
            self.bucket_mb = bucket_mb_for(sum(p.numel() for p in model.parameters()) * width)
        model = DDP(
            model,
            bucket_cap_mb=self.bucket_mb,
            find_unused_parameters=True,
            device_ids=[self.device.index] if self.device.type == "cuda" else None,
        )
        if self.comm_hook is not None:
            model.register_comm_hook(None, self.comm_hook)
        return model

    def raw(self, model: torch.nn.Module) -> torch.nn.Module:
        """The unwrapped module (DDP prefixes `state_dict` keys with `module.`)."""
        from torch.nn.parallel import DistributedDataParallel as DDP

        return model.module if isinstance(model, DDP) else model

    def barrier(self) -> None:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()

    def close(self) -> None:
        import torch.distributed as dist

        if not self._closed and dist.is_initialized():
            dist.destroy_process_group()
        self._closed = True


def _comm_hook(reduce_dtype: str):
    """The DDP compression hook for a compressed reduce, or None for fp32."""
    if reduce_dtype == "fp32":
        return None
    from torch.distributed.algorithms import ddp_comm_hooks

    return (
        ddp_comm_hooks.default_hooks.bf16_compress_hook
        if reduce_dtype == "bf16"
        else ddp_comm_hooks.default_hooks.fp16_compress_hook
    )


def init_parallel(
    run,
    *,
    log=print,
    counters: Counters | None = None,
    device_type: str | None = None,
) -> Parallel:
    """Probe, resolve the fleet policy, join the process group — in that order.

    Order matters: every refusal that can be decided from the CONFIG happens
    before `init_process_group`, because a rank that dies after joining takes
    the other ranks down with a timeout instead of with the message.
    `device_type` is the launch's device (`--device cpu` must mean the gloo
    arm even on a GPU box); `None` auto-detects.
    """
    import torch.distributed as dist

    cfg = run.train
    if not cfg.dist:
        loud(
            "init_parallel on a run with dist=false: the entry point must not be "
            "reached. Escape: set train.dist=true, or call `train()` for the "
            "single-process arm."
        )
    if not dist.is_available():
        loud(
            "dist=true but torch.distributed is not available in this build. "
            "Escape: a torch build with distributed support, or dist=false."
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size <= 1:
        loud(
            f"dist=true but WORLD_SIZE={world_size}: the run was not launched under "
            f"torchrun, so there is no fleet to join. Escape: "
            f"`torchrun --standalone --nproc_per_node=2 -m dormouse train --dist ...`, "
            f"or drop --dist."
        )
    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    backend = choose_backend(
        cfg.dist_backend,
        device_type,
        nccl_ok=dist.is_nccl_available(),
        gloo_ok=dist.is_gloo_available(),
        log=log,
    )
    if backend == "nccl" and torch.cuda.device_count() < world_size:
        loud(
            f"a {world_size}-rank nccl fleet on {torch.cuda.device_count()} visible "
            f"GPU(s). Escape: one rank per GPU (CUDA_VISIBLE_DEVICES), or fewer ranks."
        )
    device = torch.device("cpu")
    caps = cpu_caps()
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        caps = probe_device(rank)
    requested = (
        run.model.dtype if run.model.dtype != "none" else ("bf16" if run.model.bf16 else "fp32")
    )
    policy = resolve_policy(caps, requested)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if cfg.dist_capacity == "auto":
        # One fixed matmul per rank, all ranks' indices gathered, the global
        # batch split in proportion. The measurement is relative BY DESIGN.
        index = throughput_index(device)
        gathered = [0.0] * world_size
        dist.all_gather_object(gathered, index)
        shapes = suggest_capacity(gathered, total_batch=cfg.batch * world_size, seq_len=cfg.seq_len)
        log(
            f"dist_capacity=auto: throughput index {[round(i, 2) for i in gathered]} -> "
            f"shapes {[s.line() for s in shapes]} (global batch {cfg.batch * world_size} kept)"
        )
    else:
        shapes = parse_capacity(
            cfg.dist_capacity,
            world_size=world_size,
            cfg=cfg,
            max_seq_len=run.model.max_seq_len,
        )
    reduce = cfg.dist_reduce
    if reduce == "auto":
        reduce = policy.compute_dtype if policy.compute_dtype in ("bf16", "fp16") else "fp32"
    if reduce not in REDUCE_DTYPES:
        loud(f"dist_reduce={reduce!r} is not a rung. Known: {list(REDUCE_DTYPES)}")
    # The gradient payload is not known here (the model is built after the
    # group); `dist_bucket_mb=0` means "auto", resolved in `replicate`.
    parallel = Parallel(
        rank=rank,
        world_size=world_size,
        backend=backend,
        device=device,
        shapes=shapes,
        reduce_dtype=reduce,
        shard_kind=cfg.dist_shard,
        policy=policy,
        policies=[policy] * world_size,
        bucket_mb=cfg.dist_bucket_mb,
        counters=counters,
    )
    return parallel


def train_dist(
    run,
    *,
    resume: bool = False,
    log=print,
    device_type: str | None = None,
    model: torch.nn.Module | None = None,
    close: bool = True,
) -> dict:
    """The multi-process entry: join the fleet, apply the rank's policy, train.

    Everything the loop does is unchanged; this function is the wiring that
    makes one process a rank — the policy lands on the run's config BEFORE the
    model is built, the device is the rank's, and the loop gets the `Parallel`
    seam.

    `model=None` builds from the (policy-applied) config, which is the normal
    path. A caller may pass a PRE-BUILT model (the gate does, so every mode
    shares one seeded init) — but only when the dtype knob is off, because a
    built model has already captured its compute dtype and the policy would
    arrive too late to reach it.

    `close=False` leaves the process group alive for the caller — needed when
    the caller must run a collective AFTER training (the gate gathers FSDP2's
    sharded parameters to fingerprint them, and `full_tensor()` cannot resolve
    a destroyed group).
    """
    from ..model import DormouseModel
    from .loop import train

    if model is not None and run.model.dtype != "none":
        loud(
            "train_dist(model=..., dtype!=none): the model was built before the "
            "rank's compute policy was applied, so its layers captured the old "
            "dtype. Escape: let train_dist build the model (model=None), or leave "
            "dtype=none."
        )
    parallel = init_parallel(run, log=log, device_type=device_type)
    try:
        run = parallel.apply(run)
        device = parallel.device
        if model is None:
            model = DormouseModel(run.model)
        model = model.to(device)
        return train(
            model,
            run,
            resume=resume,
            log=log,
            device=None if device.type == "cpu" else device,
            parallel=parallel,
        )
    finally:
        if close:
            parallel.close()
