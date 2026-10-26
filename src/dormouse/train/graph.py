"""CUDA-graph capture of the optimizer STAGE, with the LR a live device tensor.

The measured problem this attacks (not a prediction): the port's b1 step is
**host-dispatch-bound** — the card is idle 87% of a warm step (`AGENTS` §3.1,
mean utilisation 13.3%) and the optimizer is **44-53%** of a step
(`docs/research/2026-10-17-muon-variants.md` TL;DR), because the batched Muon
pass issues 46 341 `aten::mm` launches for ONE step at b8·s512 (the 2026-10-15
profiler probe). One graph replay replaces the whole stage's launches with one
launch. The GPU time is unchanged; the enqueue time is what moves.

The mechanism is `microsoft/dion`'s (`dion/cuda_graph.py`, fetched 2026-10-17)
reimplemented in this port's idiom — the optimizer stage only, never the step:

1. **warmup first.** `warmup_steps` eager steps run before any capture, so the
   optimizer state (`Muon._build_groups` builds it lazily on the first batched
   step), the cuBLAS workspaces and the caching allocator's blocks for the NS
   temporaries are all materialized. Capturing the first step instead allocates
   them INSIDE the graph, where a `cudaMalloc` aborts the capture
   (`dion/cuda_graph.py:96-101`). The reference's own default is 10
   (`cuda_graph.py:98`, read 2026-10-17); the research note's sketch says 5. This
   port takes **5** (`GRAPH_WARMUP_STEPS`): everything the list above needs is
   touched by the FIRST step, the extra four are margin for the per-shape group
   build and the chunked `bf16` path that only engages on a wide shape group, and
   the cost is 5 eager steps against a 50 000-step run.
2. **capture ONE stage** — the grad clip and the mixed optimizer step, as the
   caller's `stage` callable — inside `torch.cuda.graph(...)`, with
   `capture_error_mode="thread_local"`. torch defaults to `"global"`, which
   errors when ANY thread makes a capture-unsafe CUDA call while the capture is
   underway; `"thread_local"` is what inductor uses for its own captures and it
   still catches unsafe calls from the capturing thread, i.e. from the step
   itself (`dion/cuda_graph.py:88-116`).
3. **replay immediately once after capture.** Capture only RECORDS the ops; it
   does not execute them, so without the replay the trajectory is one step short
   (`dion/cuda_graph.py:257-274`).
4. **then `replay()` per step.**

**The LR must be a device tensor.** A captured op can only read what is a device
tensor at capture time: `lr` read as a Python float is BAKED and silently freezes
(`dion/cuda_graph.py:70-78`). So this wrapper replaces every param group's `lr`
with a 0-dim device tensor and the trainer fills it IN PLACE each step
(`set_lr` → `fill_`), before the replay. `optim.Muon.batched_step_from_model`
already reads `lr` from the group (`-lr * scale` is a device scalar op when `lr`
is a tensor); `Muon._step_one` — the singleton-shape / missing-grad path — folds
it into the update on device instead, because `add_`'s `alpha` cannot take a
tensor without a host read.

**`zero_grad` must keep the buffers.** `set_to_none=True` swaps `.grad` for a
fresh tensor every step and breaks the buffers capture pinned
(`dion/cuda_graph.py:170-181`); the trainer passes `set_to_none=False` on this
path (`train/loop.py`).

**Deviations from the reference, named:**

* **The AdamW half rides torch's `capturable=True`.** `torch.optim.AdamW` reads
  a device-tensor LR only on that path (its non-capturable branch hands the
  dispatcher a 0-dim tensor as `addcdiv_`'s `value`, which is a HOST read). So
  this wrapper sets `group["capturable"] = True` on the AdamW groups when their
  params are CUDA. That MOVES the AdamW arithmetic: `capturable` folds
  `lr / bias_correction1` into the denominator on device instead of applying it
  as a host scalar (`torch/optim/adam.py`, the `capturable or differentiable`
  branch). It is therefore **bit-identical to the arm's OWN eager path and not
  to the pre-arm float-LR path** — which is exactly what the eager-vs-replay
  gate measures (`tests/test_opt_graph.py`), and it is why `opt_graph` is a
  config key rather than a silent default.
* **`weight_decay` stays a host float.** The reference can carry it as a Tensor
  opt-in (`cuda_graph.py:39-50`); this port's WSD schedule does not rewrite `wd`
  per step, so the opt-in buys nothing and is not implemented — a scheduled `wd`
  would be baked, and that is named here rather than discovered.
* **No `release()`-before-teardown ceremony.** The reference needs it for the
  captured NCCL ops (`cuda_graph.py:135-160`); this port is single-GPU, so
  `release()` only drops the graph and restarts the warmup.
* **`torch.compile` is refused** by the trainer alongside this arm: a recompile
  attempted under an active capture aborts it (`cuda_graph.py`'s caveat), and the
  b1 form runs eager for its own measured reason.

**Refusals are LOUD, the fallback is COUNTED.** `capture()` raises
`instrument.LoudError` naming the cause AND the escape for every condition that
would freeze or corrupt a captured step. `step()` catches ANY exception out of
the capture, counts `opt_graph_fallback` + the reason, and stays EAGER for the
rest of the run — a run must never die of this arm, and a half-built graph is
never kept (`dion/cuda_graph.py:253-255`: publishing a failed graph turns one
loud error into every later step replaying garbage).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..instrument import Counters, loud

#: Eager steps before capture (`dion/cuda_graph.py:96-101` needs >= 1; the
#: justification for 5 is in the module docstring).
GRAPH_WARMUP_STEPS = 5

#: The headroom a capture needs before it is even attempted: `max(floor, SHARE of
#: the device's total VRAM)`. A capture's allocations come from a graph-private
#: pool that is PERMANENT for the graph's life, so it must be mapped out of
#: memory the run does not need back — and the pool is only HALF the problem: the
#: step after the capture still has to allocate from what is left.
#:
#: Measured on this box (2026-10-23, `final_byte`, bf16, 15.6 GiB card):
#: * the optimizer stage's pool is **88-592 MiB** depending on `MUON_CHUNK_ELEMS`
#!   (592 at the shipped chunk, 88 at 1<<22) — measured, not estimated;
#: * b8 s512 leaves ~2 GiB free and captures cleanly;
#: * b24 s1024 leaves **~100-250 MiB**, and there the capture OOMs inside
#:   `cudaStreamBeginCapture` (growable allocator, `expandable_segments:True`) or
#:   the run's NEXT forward OOMs on the fragmentation the new pool causes
#:   (default allocator). That is the 2026-10-23 b24 s1024 fault: the arm
#:   attempted a capture the card could not hold.
#:
#: 8% of 15.6 GiB is 1.25 GiB, which is 2x the largest measured pool, and it is
#: what turns that shape into a COUNTED refusal with the numbers instead of an
#: OOM or an `cudaErrorIllegalAddress` deep in the capture.
CAPTURE_HEADROOM_FLOOR = 384 << 20
CAPTURE_HEADROOM_SHARE = 0.08


def _iter_groups(opt):
    """`(param_group, is_adamw)` for every group the stage updates."""
    for group in opt.adamw.param_groups:
        yield group, True
    if opt.muon is not None:
        for group in opt.muon.param_groups:
            yield group, False


def _bind_device_lr(opt) -> tuple[list[tuple[dict, torch.Tensor]], list[tuple[dict, bool, object]]]:
    """Every optimizer group's LR as a persistent 0-dim DEVICE tensor.

    ONE definition for both graph arms (`OptimizerGraph`, `FullGraph`): a
    captured op can only read a tensor, so a Python float `lr` would be baked at
    capture and silently freeze the WSD schedule (`dion/cuda_graph.py:70-78`).
    The trainer fills these IN PLACE with `set_lr`. AdamW additionally gets
    torch's own `capturable=True` switch — without it its non-capturable branch
    hands the dispatcher a 0-dim tensor as `addcdiv_`'s `value`, i.e. a HOST
    read the capture window refuses.

    Returns `(lr_tensors, capturable_restore)`: the second list is
    `(group, key_was_present, old_value)` so `_fall_back` can put AdamW back on
    the eager path it claims to be (a `capturable=True` left behind would keep
    AdamW's captured arithmetic after the log said "stayed EAGER").
    """
    out: list[tuple[dict, torch.Tensor]] = []
    restore: list[tuple[dict, bool, object]] = []
    for group, is_adamw in _iter_groups(opt):
        params = group["params"]
        if not params:
            continue  # an empty group has no device to put a tensor on
        tensor = torch.tensor(float(group["lr"]), dtype=torch.float32, device=params[0].device)
        group["lr"] = tensor
        out.append((group, tensor))
        if is_adamw:
            restore.append((group, "capturable" in group, group.get("capturable")))
            if params[0].device.type == "cuda":
                group["capturable"] = True
                for p in params:
                    # A checkpoint written BEFORE this arm landed (or read through
                    # `load_ckpt`'s `map_location="cpu"`) carries a HOST `step`
                    # counter. torch's capturable branch asserts the counter shares
                    # the param's device, and that assert would fire in the eager
                    # WARMUP step — outside any try/except.
                    state = opt.adamw.state.get(p, {})
                    if "step" in state and state["step"].device.type != "cuda":
                        state["step"] = state["step"].to(
                            device=params[0].device, dtype=torch.float32
                        )
    return out, restore


#: The capture's error mode: what inductor uses, and what survives another
#: thread's capture-unsafe CUDA call (`dion/cuda_graph.py:88-116`).
CAPTURE_ERROR_MODE = "thread_local"


class OptimizerGraph:
    """Warmup → capture → replay around ONE optimizer stage, LR on device.

    `stage` is the caller's whole optimizer stage — the grad clip and the mixed
    optimizer step (`train/loop.py`'s `opt_stage`) — so this file owns the
    capture policy and nothing about the optimizer's contents.
    """

    #: The arm's name in every message (`_fall_back`, `capture`): the two
    #: captures share this base class, so a hardcoded prefix made a
    #: `full_graph` fallback report itself as `opt_graph`.
    arm = "opt_graph"

    def __init__(
        self,
        opt,
        stage: Callable[[], None],
        *,
        warmup_steps: int = GRAPH_WARMUP_STEPS,
        counters: Counters | None = None,
        log=None,
        allow_frozen: bool = False,
    ) -> None:
        if warmup_steps < 1:
            loud(
                f"opt_graph warmup_steps={warmup_steps}: capture needs at least one "
                f"EAGER step, because that step is what allocates the optimizer state, "
                f"the cuBLAS workspaces and the allocator blocks — capturing it "
                f"instead allocates inside the graph and aborts the capture "
                f"(`dion/cuda_graph.py:96-101`). Escape: warmup_steps >= 1."
            )
        self.opt = opt
        self.stage = stage
        self.warmup_steps = warmup_steps
        self.counters = counters
        self.log = log
        self._steps = 0
        self._graph: torch.cuda.CUDAGraph | None = None
        #: The last value `set_lr` was handed, so a fallback can hand the groups
        #: back the EXACT host float the eager path would have read (not the fp32
        #: rounding the tensor holds).
        self._lr_value = 0.0
        #: The step's own arm counters, as they read during capture: a replay runs
        #: only the recorded DEVICE ops, so a counter the step wrote would stop
        #: advancing and a COUNTED arm would go SILENT after the capture step.
        self._captured_counters: dict[str, int] = {}
        #: The COUNTED fallback's reason; once set, the run stays eager forever.
        self.reason: str | None = None
        #: `(group, tensor)`: the persistent LR tensors, and the groups that read
        #: them. `group["lr"] IS` the tensor, so the captured ops re-read the live
        #: value on every replay.
        self._lr, self._capturable_restore = _bind_device_lr(opt)
        #: The ORDERED `(group, tensor)` list `set_lr` writes; the same list the
        #: `FullGraph` arm uses, so the two arms cannot bind the LR differently.
        self.frozen: list[str] = []
        #: A parameter with no gradient for the whole warmup is structurally dead;
        #: the TRAINER sets this so one dead gain cannot disable the whole arm
        #: (see `_refusals`). The default keeps the historical refusal, which is
        #: what a direct caller (and the arm's own gate) sees.
        self.allow_frozen = allow_frozen
        #: The captured participating set: `name -> True` for every parameter whose
        #: `.grad` the captured ops actually write. A replay re-checks it (a host
        #: compare over `p.grad is not None`, no device read) because a parameter
        #: that STARVES later would be silently frozen (`dion/cuda_graph.py:231-255`).
        self._participants: dict[str, bool] = {}

    # -- the LR seam --------------------------------------------------------

    def set_lr(self, lr: float) -> None:
        """Fill every group's LR tensor IN PLACE — the trainer's per-step call.

        In place is the whole point: `group["lr"] = lr` would REPLACE the tensor
        the captured ops recorded with a float no replay reads, which is the
        freeze `dion/cuda_graph.py:70-78` documents. The value the WSD schedule
        produces must reach the TENSOR, so a float that got assigned anyway is
        folded back into it here.

        After a COUNTED fallback the arm no longer exists: the groups hold host
        floats again (`_fall_back`) and this writes the float the eager path reads,
        rather than resurrecting a tensor nothing would capture.
        """
        self._lr_value = lr
        if self.reason is not None:
            for group, _ in self._lr:
                group["lr"] = lr
            return
        for group, tensor in self._lr:
            tensor.fill_(lr)
            if group.get("lr") is not tensor:
                group["lr"] = tensor

    # -- the capture policy -------------------------------------------------

    def _groups(self):
        """`(param_group, is_adamw)` for every group the stage updates."""
        return _iter_groups(self.opt)

    def _refusals(self) -> str | None:
        """Why capture must not run, or `None`. Most actionable reason first."""
        muon = self.opt.muon
        if getattr(self.opt, "bnb8bit", False):
            return (
                "optim_8bit=true: bitsandbytes' AdamW8bit has no capturable / "
                "device-tensor-LR path, so a captured step would freeze its LR at "
                "the capture value. Escape: optim_8bit=false, or opt_graph=false."
            )
        if muon is not None and muon.sf:
            return (
                "muon_sf=true: the schedule-free wrapper's `_sf_ckp` recursion "
                "(`s += lr*lr`, `c = lr*lr/s`) is HOST arithmetic and reads `lr` as a "
                "Python float (`optim.Muon._sf_ckp`), so a captured step would freeze "
                "BOTH the iterate averaging and the LR at capture time. Escape: "
                "muon_sf=false, or opt_graph=false (putting `_sf_ckp` on device is the "
                "named follow-up, not this arm)."
            )
        if muon is not None and muon.variant != "muon":
            return (
                f"muon_variant={muon.variant!r}: the NorMuon tail applies the LR as "
                f"`torch._foreach_add_(..., alpha=-lr)` (`optim.Muon._nor_batched`), and "
                f"an `alpha` that is a 0-dim device tensor is read back to the HOST by "
                f"the dispatcher — a sync inside the capture window, which CUDA graphs "
                f"refuse. Escape: muon_variant='muon', or opt_graph=false."
            )
        names = {id(p): n for n, p in self.opt.model.named_parameters()}
        missing = [
            (names.get(id(p), f"group {gi}[{i}]"), p)
            for gi, (group, _) in enumerate(self._groups())
            for i, p in enumerate(group["params"])
            if p.requires_grad and p.grad is None
        ]
        if missing and not self.allow_frozen:
            # The step only touches params that have a gradient, so the
            # participating set FREEZES at capture: a param starved here would
            # silently stop being updated for the rest of the run
            # (`dion/cuda_graph.py:231-255`).
            more = f", ... (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            shown = ", ".join(f"{n} {tuple(p.shape)}" for n, p in missing[:5]) + more
            return (
                f"{len(missing)} parameter(s) require a gradient but have .grad=None at "
                f"capture, so every replay would leave them frozen: {shown}. Escape: "
                f"capture after every parameter has taken part in a backward pass (raise "
                f"the warmup past whatever starves them), or set requires_grad=False on "
                f"the ones meant to be frozen, or opt_graph=false."
            )
        if missing:
            # `allow_frozen` (the TRAINER's setting, and `FullGraph`'s): a
            # parameter with no gradient for the whole warmup is structurally
            # dead, so it is excluded from the captured participating set and
            # COUNTED (`opt_graph_frozen`) instead of refusing the whole arm.
            # Refusing here is how this arm measured +0 at b8 s512 without ever
            # capturing: `loop.residual_scale` under `use_mhc` is never read, so
            # its `.grad` is None on every step and the refusal fired forever.
            # The hazard the refusal exists for — a parameter starved on ONE
            # step — is caught on every replay by `_check_participants`.
            self.frozen = [n for n, _ in missing]
            if self.counters is not None:
                self.counters.count("opt_graph_frozen", len(self.frozen))
            if self.log is not None and len(missing) <= 5:
                self.log(
                    f"opt_graph: {len(self.frozen)} parameter(s) never received a gradient "
                    f"in the warmup and are excluded from the captured stage: "
                    f"{', '.join(self.frozen)}. The eager path does not train them either."
                )
        if not self._lr or not all(t.device.type == "cuda" for _, t in self._lr):
            return (
                "the optimizer's parameters are not on CUDA: a CUDA graph is a "
                "CUDA-only mechanism, and capturing on any other device is not a "
                "degradation this port can make honest. Escape: run with "
                "--device cuda, or opt_graph=false."
            )
        if self._lr:
            # FREE, after the `empty_cache` in `capture()` has returned the
            # allocator's cached blocks: the pool is mapped from exactly this.
            # (A `max_memory_reserved`-based rule was tried first and is WRONG
            # here: the previous step's activations are already freed by the
            # backward, so the peak says "full" for a capture that fits
            # perfectly — measured on the b24 s1024 gate, which captured with
            # 13.2 GiB free while its peak read 14.8 GiB.)
            free, total = torch.cuda.mem_get_info(self._lr[0][1].device)
            # ...and the run's own measured PEAK must leave room past it too:
            # the pool is permanent, so the next forward has to re-reach the
            # working set it already reached once. One rule alone is wrong in
            # one direction each, and both directions were measured here:
            # `free` alone lets b24 s1024 capture (13.2 GiB free at the capture
            # point) and then OOMs 192 MiB into the next forward; `peak` alone
            # refuses a capture-only harness whose step is already over.
            peak = torch.cuda.max_memory_reserved(self._lr[0][1].device)
            need = max(CAPTURE_HEADROOM_FLOOR, int(total * CAPTURE_HEADROOM_SHARE))
            if free < need or total - peak < need:
                return (
                    f"{free / 2**20:.0f} MiB free and {peak / 2**30:.1f} of "
                    f"{total / 2**30:.1f} GiB reserved at the run's peak: this arm needs "
                    f"{need / 2**20:.0f} MiB of headroom (a capture's "
                    f"private pool is PERMANENT for the graph's life (it is never returned to the "
                    f"allocator), so the capture would OOM here — and on a card this full "
                    f"an OOM lands either inside the capture or in the next forward, as "
                    f"the fragmentation the 2026-10-23 b24 s1024 run hit. Measured on this "
                    f"box: the optimizer stage's pool is 88-592 MiB, and a whole-step "
                    f"pool is larger than the step's own activation peak (measured: the "
                    f"forced b24 s1024 whole-step capture OOMs at a 96 MiB allocation with "
                    f"125 MiB free). Escape: run with "
                    f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (measured: what lets "
                    f"the b24 s1024 run survive a pool at all), shrink the batch/sequence, "
                    f"or {self.arm}=false."
                )
        return None

    def capture(self) -> None:
        """Capture one full stage and replay it once — LOUD on any refusal.

        The immediate replay is not optional: capture only RECORDS the ops
        (`dion/cuda_graph.py:257-274`), so a capture without it leaves the
        trajectory one step short.
        """
        torch.cuda.empty_cache()  # ONE call, before `_refusals`: the headroom rule
        # must read the allocator's real state (the previous step's cached blocks
        # are what a capture's private pool has to be mapped out of).
        why = self._refusals()
        if why is not None:
            loud(f"{self.arm}: capture refused — {why}")
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode=CAPTURE_ERROR_MODE):
            self.stage()
        # Publish ONLY on success: a graph from a failed capture is not replayable,
        # and holding it would turn one loud error into every later step replaying
        # garbage (`dion/cuda_graph.py:253-255`).
        self._graph = graph
        counters = getattr(self.opt, "counters", None)
        self._captured_counters = dict(counters) if counters else {}
        # The set the recorded ops write. `_participants` is checked on every
        # replay: a parameter that loses its gradient later would be frozen by
        # the recording, which is the hazard the old blanket refusal named.
        self._participants = {
            name: p.grad is not None
            for name, p in self.opt.model.named_parameters()
            if p.requires_grad
        }
        graph.replay()

    def _check_participants(self) -> str | None:
        """Why the recorded participating set no longer matches, or `None`.

        A host compare (`p.grad is not None`), so it adds no device read and no
        sync. The cost is one Python loop per replay, against the 815 tensors the
        step already touches.
        """
        for name, p in self.opt.model.named_parameters():
            if not p.requires_grad:
                continue
            had = self._participants.get(name)
            if had is None:
                continue
            if had != (p.grad is not None):
                return (
                    f"parameter {name} had a gradient at capture and has "
                    f"{'one' if not had else 'none'} now, so the recording's "
                    f"participating set no longer matches the step: replaying would "
                    f"freeze it (or step it on a stale buffer). Escape: run without "
                    f"opt_graph, or find the data-dependent branch that starves it."
                )
        return None

    # -- the step -----------------------------------------------------------

    def step(self) -> None:
        """One optimizer stage: eager during warmup and after a fallback, replay after.

        Called UNCONDITIONALLY, including on a step the NaN firewall skipped: the
        eager path also runs the stage on a masked step (the device flag has
        already zeroed the gradients), and `opt_graph`'s warmup/capture must be
        the same program. A capture records the stage once; the flag is what
        makes one recording serve both outcomes without a host branch.
        """
        self._steps += 1
        if self._graph is not None:
            self._replay()
            return
        if self.reason is not None or self._steps <= self.warmup_steps:
            self.stage()
            return
        try:
            self.capture()
        except Exception as e:  # noqa: BLE001 — no exception out of this arm may kill a run
            self._fall_back(f"{type(e).__name__}: {e}")
            self.stage()

    def _replay(self) -> None:
        """Replay, then redo the HOST bookkeeping the recorded device ops cannot carry.

        `MixOptimizer.step`'s Python counters advance once, at capture, and never
        again under replay — so a resume would read a stale step count — and the
        Muon arm counters (`ns_triton` / `ns_triton_fallback`) would stop being
        written at all, turning a COUNTED arm into a silent one.
        """
        why = self._check_participants()
        if why is not None:
            # The recording is no longer a faithful step: drop it and stay eager
            # rather than replaying a graph that freezes a parameter.
            self._fall_back(why)
            self.stage()
            return
        self._graph.replay()
        if hasattr(self.opt, "steps"):
            self.opt.steps += 1
        # `steps_batched` is the same value every step (one count per shape group),
        # so the capture-time value is the live one and is left alone.
        for name, n in self._captured_counters.items():
            self.opt.counters.count(name, n)

    def _fall_back(self, reason: str) -> None:
        """COUNTED: record why, print it, and stay eager for the rest of the run.

        The groups also get their HOST floats back: the eager path reads `lr` from
        the group dict, and leaving a device tensor there would make the "fallback"
        a subtly different program from the one it claims to be. A group that
        does NOT hold a device tensor is left alone — the trainer writes the SF
        Muon group's constant LR as a host float after `set_lr`, and clobbering
        it would put the fallback step on the WSD value for one step. AdamW's
        `capturable` switch is restored too (to its pre-bind value, or removed):
        leaving it True would keep the captured arithmetic while the log says
        the run "stayed EAGER".
        """
        self.reason = reason
        for group, tensor in self._lr:
            if group.get("lr") is tensor:
                group["lr"] = self._lr_value
        for group, was_present, old in self._capturable_restore:
            if was_present:
                group["capturable"] = old
            else:
                group.pop("capturable", None)
        self._capturable_restore.clear()
        if self.counters is not None:
            # ONE counter for both arms (`opt_graph_fallback` is the documented
            # name the log/JSONL schema carries), but the REASON names the arm
            # that fell back: a `full_graph` run whose log says `opt_graph` is a
            # reader who cannot tell which of the two captures failed (ADR-0019).
            self.counters.count("opt_graph_fallback")
            self.counters.why = f"{self.arm}: {reason}"  # type: ignore[attr-defined]
        if self.log is not None:
            self.log(
                f"{self.arm}: capture failed, staying EAGER for the rest of the run — {reason}"
            )

    def release(self) -> None:
        """Drop the graph and restart the warmup (the next step re-captures)."""
        graph, self._graph = self._graph, None
        self._steps = 0
        if graph is None:
            return
        torch.cuda.synchronize()  # never tear a graph down under an in-flight replay
        graph.reset()


# ===========================================================================
# `full_graph`: ONE graph for forward + backward + firewall + optimizer + zero
# ===========================================================================


def full_graph_refusal(train, model) -> str | None:
    """Why `full_graph` must not run on this config, or `None`.

    ONE entry per REAL conflict, each naming its `Escape:`. A conflict is real
    when the captured region would (a) hold a HOST read that a capture window
    refuses, (b) freeze a value the Python loop recomputes per step, or (c) be
    a different PROGRAM from the eager step it replaces. Everything else is
    allowed: the MoE top-k is a device op on buffer VALUES, which a graph
    replays correctly (the kernel reads whatever the buffer holds).

    `train` is a `TrainConfig`, `model` a `ModelConfig`.

    Everything decidable from CONFIG is here, so it fires before the first eager
    step: the optimizer arms (`optim_8bit`, `muon_sf`, `muon_variant`), the
    per-step HOST reads (`engram_ram`, `moe_offload`, `use_msa`, JEPA, DSpark,
    act_quant, QAT), and the known non-capturable arms. Only the checks that need
    the LIVE model stay in `capture()`: a parameter starved through the whole
    warmup (`.grad is None`) and the VRAM headroom, which is a device measurement.
    """
    if train.opt_graph:
        return (
            "full_graph=true with opt_graph=true names TWO captures of one region — "
            "the whole step, and the optimizer stage inside it. The optimizer graph "
            "would replay inside the superset's replay. Escape: full_graph=true alone "
            "(it implies the optimizer capture), or opt_graph=true alone."
        )
    if getattr(train, "optim_8bit", False):
        return (
            "optim_8bit=true: bitsandbytes' AdamW8bit has no capturable / "
            "device-tensor-LR path, so a captured step would freeze its LR at the "
            "capture value. Escape: optim_8bit=false, or full_graph=false."
        )
    if getattr(train, "muon_sf", False):
        return (
            "muon_sf=true: the schedule-free wrapper's `_sf_ckp` recursion "
            "(`s += lr*lr`, `c = lr*lr/s`) is HOST arithmetic and reads `lr` as a "
            "Python float (`optim.Muon._sf_ckp`), so a captured step would freeze "
            "BOTH the iterate averaging and the LR at capture time. Escape: "
            "muon_sf=false, or full_graph=false (putting `_sf_ckp` on device is the "
            "named follow-up, not this arm)."
        )
    if getattr(train, "muon_variant", "muon") != "muon":
        return (
            f"muon_variant={train.muon_variant!r}: the NorMuon tail applies the LR as "
            f"`torch._foreach_add_(..., alpha=-lr)` (`optim.Muon._nor_batched`), and "
            f"an `alpha` that is a 0-dim device tensor is read back to the HOST by "
            f"the dispatcher — a sync inside the capture window, which CUDA graphs "
            f"refuse. Escape: muon_variant='muon', or full_graph=false."
        )
    if train.grad_accum > 1:
        return (
            f"grad_accum={train.grad_accum}: the accumulation loop is PYTHON — "
            f"{train.grad_accum} forwards, a per-micro-batch firewall AND, and one "
            f"group-level gate multiply. A capture records exactly ONE micro-batch and "
            f"would replay it as if it were the group. Escape: grad_accum=1, or "
            f"full_graph=false."
        )
    if train.rand_depth:
        return (
            "rand_depth=true: the depth T is sampled from `(seed, step)` on the HOST "
            "(`loop._sample_depth`) and passed to the forward, so a capture would "
            "freeze the capture step's depth for the whole run. Escape: rand_depth=false, "
            "or full_graph=false."
        )
    if train.segment_len:
        return (
            f"segment_len={train.segment_len}: the segmented forward is a PYTHON loop "
            f"over segments whose KDA state is a graph EDGE per segment "
            f"(`loopblock._forward_segmented`), so the capture would freeze the segment "
            f"count. Escape: segment_len=0, or full_graph=false."
        )
    if getattr(model, "use_byteflow", False):
        return (
            "use_byteflow=true: the byteflow channel's loop runs over K patch latents "
            "whose count depends on the window, and its gram/cholesky bookkeeping is "
            "host-side (`byteflow.py`). Escape: use_byteflow=false, or full_graph=false."
        )
    if getattr(model, "halt_weight", 0.0) > 0.0 or getattr(model, "halt_mode", False):
        return (
            "the halt arm is armed (halt_weight>0 or halt_mode=true): in `halt_mode` "
            "the loop reads the halt probability back to the host "
            "(`loopblock.py:674` `float(torch.sigmoid(halt(step_out)))`) — a D2H inside "
            "the captured forward — and `halt_weight>0` leaves `model.last_halt` as a "
            "HOST attribute the replay never updates. Escape: halt_weight=0.0 and "
            "halt_mode=false, or full_graph=false."
        )
    if getattr(model, "use_msa", False):
        return (
            "use_msa=true: the MSA arm reads a device share back to the host per "
            "iteration (`loopblock.py:576` `float(share[i].detach())`), a D2H the "
            "capture window refuses. Escape: use_msa=false, or full_graph=false."
        )
    if train.ema_decay > 0.0:
        return (
            f"ema_decay={train.ema_decay:g}: the EMA update runs EAGERLY after the "
            f"optimizer step and `lerp_`s the shadow from the LIVE parameters. Inside "
            f"the capture it would be recorded once, at the capture step's weights, and "
            f"every replay would re-apply that step's delta. Escape: ema_decay=0.0, or "
            f"full_graph=false."
        )
    if getattr(train, "engram_ram", False):
        return (
            "engram_ram=true: the host-table Engram resolves each forward's rows with a "
            "device→host INDEX transfer (`engram._forward_host`'s `.to('cpu')`), which a "
            "capture window refuses. Escape: engram_ram=false (the in-VRAM table), or "
            "full_graph=false."
        )
    if getattr(model, "moe_offload", False):
        return (
            "moe_offload=true: the expert pass's host-offload hooks park every saved "
            "activation in host RAM and copy it back for the backward — a blocking "
            "device→host transfer inside the captured window, which a capture neither "
            "records nor replays. Escape: moe_offload=false, or full_graph=false."
        )
    if getattr(model, "jepa_weight", 0.0) > 0.0:
        return (
            f"jepa_weight={model.jepa_weight:g}: JEPA's span mask is a fresh device "
            f"tensor derived from `(seed, step)` per step, so a capture would replay the "
            f"capture step's mask forever — a stale objective, invisible in the loss. "
            f"Escape: jepa_weight=0.0, or full_graph=false."
        )
    if getattr(model, "dspark_weight", 0.0) > 0.0:
        return (
            f"dspark_weight={model.dspark_weight:g}: the DSpark term leaves "
            f"`model.last_mtp` as a HOST attribute (the log line's `mtp=`), which a "
            f"replay never updates — a COUNTED arm going silent. Escape: "
            f"dspark_weight=0.0, or full_graph=false."
        )
    if getattr(model, "act_quant", "none") != "none":
        return (
            f"act_quant={model.act_quant!r}: the activation quantizer's finiteness "
            f"guard is a HOST read of a device tensor (`act_quant.py:240` "
            f"`bool(torch.isfinite(x).all())`) inside the forward. Escape: "
            f"act_quant=none, or full_graph=false."
        )
    if (
        getattr(model, "dense_qat", "none") != "none"
        or getattr(model, "factor_qat", "none") != "none"
    ):
        return (
            "the QAT arms are armed (dense_qat/factor_qat): they keep a per-step "
            "projection cache keyed on the TENSOR (not `_version`) that the trainer "
            "invalidates AFTER the step, and they count refreshes on the host. A capture "
            "would record the capture step's projection and every replay would train on "
            "the same quantized weights. Escape: dense_qat=none and factor_qat=none, or "
            "full_graph=false."
        )
    # --- the routed MoE: only the capacity-PADDED arm has static rows --------
    #
    # Measured 2026-10-24: `order[keep]` (a BOOLEAN-mask index) is `nonzero` on
    # the device — a D2H read AND a kept-row count that changes with the
    # routing. Both are illegal in a graph: the capture aborts at the op
    # (`tests/test_full_graph.py` captures it) and a replay would keep the
    # first call's shape while the data changed. `_padded_dispatch` now keeps
    # `n·k` rows and masks the cap, which is why IT can be captured; the other
    # three arms are built around the ragged kept count itself (`offs`, the
    # per-expert loop's `unique`, the gather ABI's permuted rows), so they are
    # refused HERE, loudly, instead of falling back mid-capture.
    if getattr(model, "moe_topk", 0) > 0:
        kernel = getattr(model, "moe_kernel", "padded")
        cap = getattr(model, "moe_capacity", 0.0)
        gather = getattr(model, "moe_gather", "off")
        why = None
        if cap == 0:
            why = (
                "moe_capacity=0 (dropless) routes the routed pass to the RAGGED arms, "
                "whose row count is the capacity's own (there is no cap to pad to)"
            )
        elif kernel == "triton":
            why = (
                "moe_kernel=triton's grouped GEMM is RAGGED: `offs` and the gathered "
                "row count come from `order[keep]`, a data-dependent `nonzero`"
            )
        elif kernel == "naive":
            why = (
                "moe_kernel=naive is a PYTHON loop over `torch.unique(...).tolist()`, "
                "i.e. a host read and a per-expert row count that changes with routing"
            )
        elif gather == "kernel":
            why = (
                "moe_gather=kernel's ABI is the PERMUTED (ragged) rows, so its row "
                "count is `order[keep]`'s"
            )
        if why is not None:
            return (
                f"the routed MoE arm cannot be captured: {why}. Escape: moe_kernel="
                f"'padded' with moe_capacity>0 and moe_gather='off' (the only arm whose "
                f"rows are static), or full_graph=false."
            )
    return None


class FullGraph(OptimizerGraph):
    """ONE graph for the WHOLE training step, replayed per step.

    The captured region is `forward → backward → the device firewall → the grad
    gate → `zero_grad` → the optimizer stage`, which is what makes the step ONE
    launch instead of the measured 13 886 kernel launches at b24 s1024.

    What a capture needs and this class supplies, in order:

    1. **Static inputs.** `x`/`y` are persistent device buffers (`fill()` copies
       the host's batch in with `copy_` — never a rebind: the recording holds the
       ADDRESS). Same for the LR: `set_lr` fills the group's 0-dim tensors.
    2. **`.grad` in place.** The capturing warmup steps create every
       participating `.grad`; `zero_grad(set_to_none=False)` inside the region
       zeroes them, so the recording keeps its buffers.
    3. **No host read in the region.** A capture EXECUTES the body once, so any
       `.item()`/`float(device_tensor)`/`.to("cpu")` in it raises inside
       `cudaStreamBeginCapture` — the capture is the audit. `tests/test_full_graph.py`
       additionally traps `Tensor.item` to prove the region is clean.
    4. **Nothing per-step recomputed on the host.** The counters the forward
       wrote are recorded at capture and re-merged per replay (`on_replay`), which
       is what keeps a COUNTED arm from going SILENT after the capture step.

    A `full_graph` replay does NOT advance `_version`, so the trainer still calls
    `invalidate_banks_all`/`invalidate_qat_cache_all` after it (outside the
    region): the expert bank cache and the QAT cache are keyed on it.
    """

    arm = "full_graph"

    def __init__(
        self,
        opt,
        body: Callable[[], None],
        *,
        batch: int,
        seq_len: int,
        device,
        warmup_steps: int = GRAPH_WARMUP_STEPS,
        counters: Counters | None = None,
        log=None,
        allow_frozen: bool = True,
        on_replay: Callable[[], None] | None = None,
        on_counters: Callable[[], None] | None = None,
    ) -> None:
        assert batch > 0 and seq_len > 0, f"bad static input shape {batch}x{seq_len}"
        super().__init__(
            opt,
            body,
            warmup_steps=warmup_steps,
            counters=counters,
            log=log,
            allow_frozen=allow_frozen,
        )
        self.body = body
        self.batch = batch
        self.seq_len = seq_len
        self.device = torch.device(device)
        self.on_replay = on_replay
        #: Called on EVERY path (eager warmup, COUNTED fallback, replay): the
        #: forward's arm counters are PYTHON, so a capture freezes them and a
        #: replay never advances them. Without this a COUNTED arm goes silent
        #: the moment the arm is on (ADR-0019's defect).
        self.on_counters = on_counters
        #: The STATIC inputs. `dtype=torch.long` is the model's id dtype and the
        #: buffers are `[batch, seq_len]` exactly — the recording holds these
        #: addresses, so they are allocated once and never rebound.
        self.x = torch.zeros((batch, seq_len), dtype=torch.long, device=self.device)
        self.y = torch.zeros((batch, seq_len), dtype=torch.long, device=self.device)

    # -- the data seam ------------------------------------------------------

    def fill(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Copy the host's batch INTO the static buffers (no rebind, no sync).

        `copy_` from pageable host memory is what the eager path's `.to(device)`
        does too; the difference is that this writes the same storage every step,
        which is the whole reason the recording stays valid. A shape mismatch is
        LOUD: a short last window would otherwise be zero-padded silently.
        """
        if tuple(x.shape) != (self.batch, self.seq_len):
            loud(
                f"full_graph: batch shape {tuple(x.shape)} != the static "
                f"{self.batch}x{self.seq_len} captured. Escape: draw a full window "
                f"(the stream pads nothing), or full_graph=false."
            )
        self.x.copy_(x)
        self.y.copy_(y)

    # -- the step -----------------------------------------------------------

    def capture(self) -> None:
        """Capture the whole step and replay it once (the replay IS the step)."""
        torch.cuda.empty_cache()  # before the headroom rule: it must see the truth
        why = self._refusals()
        if why is not None:
            loud(f"full_graph: capture refused — {why}")
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        # The WARMUP's last backward left an `AccumulateGrad` node referring to
        # the DEFAULT stream. Capture runs on a side stream, so autograd refuses
        # to `cudaStreamWaitEvent` on stream 0 — it would pull a non-capturing
        # stream into the recording — and the capture dies before the first op
        # (`RuntimeError: ... stale reference to the default stream (stream 0)`).
        # torch's own switch for exactly this case redirects that node to the
        # producer's CAPTURING stream, i.e. the accumulation lands inside the
        # graph, which is where the step's accumulation belongs. Set around this
        # one capture and restored to its PROCESS-INITIAL value afterwards (the
        # C++ side exposes no getter; `False` is what torch's own docstring
        # names as the process initial state, and nothing else in this port sets
        # it) so a stale stream anywhere else is still the loud error it is today.
        torch.autograd.graph.set_override_stale_capture_stream(True)
        try:
            with torch.cuda.graph(graph, capture_error_mode=CAPTURE_ERROR_MODE):
                self.body()
        finally:
            torch.autograd.graph.set_override_stale_capture_stream(False)
        self._graph = graph
        # The recorded step's own DEVICE counters are recorded too; the HOST ones
        # (`forward`, the arm counters) are Python and stop at capture, so they
        # are re-merged per replay from this snapshot.
        self._captured_counters = {}
        self._participants = {
            name: p.grad is not None
            for name, p in self.opt.model.named_parameters()
            if p.requires_grad
        }
        graph.replay()
        self._host_bookkeeping()

    def step(self) -> None:
        """One step: eager during warmup and after a fallback, one replay after.

        Called UNCONDITIONALLY — a NaN-skipped step replays too: `full_body`
        re-fills the device flag with 1 and the micro-step's mask zeroes the whole
        contribution before the recorded optimizer stage, which is exactly what
        the eager path does with the same flag.
        """
        self._steps += 1
        if self._graph is not None:
            self._replay()
            return
        if self.reason is not None or self._steps <= self.warmup_steps:
            self.body()
            self._counters()
            return
        try:
            self.capture()
        except Exception as e:  # noqa: BLE001 — no exception out of this arm may kill a run
            self._fall_back(f"{type(e).__name__}: {e}")
            self.body()
            self._counters()

    def _replay(self) -> None:
        why = self._check_participants()
        if why is not None:
            self._fall_back(why)
            self.body()
            self._counters()
            return
        self._graph.replay()
        self._host_bookkeeping()

    def _host_bookkeeping(self) -> None:
        """The HOST half of the step, once per replay (never per device op)."""
        if hasattr(self.opt, "steps"):
            self.opt.steps += 1
        for name, n in self._captured_counters.items():
            self.opt.counters.count(name, n)
        if self.on_replay is not None:
            self.on_replay()
        self._counters()

    def _counters(self) -> None:
        """The body's own arm counters, on every path (`on_counters`)."""
        if self.on_counters is not None:
            self.on_counters()
