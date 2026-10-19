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

#: The capture's error mode: what inductor uses, and what survives another
#: thread's capture-unsafe CUDA call (`dion/cuda_graph.py:88-116`).
CAPTURE_ERROR_MODE = "thread_local"


class OptimizerGraph:
    """Warmup → capture → replay around ONE optimizer stage, LR on device.

    `stage` is the caller's whole optimizer stage — the grad clip and the mixed
    optimizer step (`train/loop.py`'s `opt_stage`) — so this file owns the
    capture policy and nothing about the optimizer's contents.
    """

    def __init__(
        self,
        opt,
        stage: Callable[[], None],
        *,
        warmup_steps: int = GRAPH_WARMUP_STEPS,
        counters: Counters | None = None,
        log=None,
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
        self._lr: list[tuple[dict, torch.Tensor]] = []
        for group, is_adamw in self._groups():
            params = group["params"]
            if not params:
                continue  # an empty group has no device to put a tensor on
            tensor = torch.tensor(float(group["lr"]), dtype=torch.float32, device=params[0].device)
            group["lr"] = tensor
            self._lr.append((group, tensor))
            if is_adamw and params[0].device.type == "cuda":
                # torch's own switch for a device-tensor LR: without it its
                # non-capturable branch reads `lr` back to the HOST as `addcdiv_`'s
                # `value` (a sync the capture window refuses). Set BEFORE the first
                # AdamW step, because the step's own device counter is created
                # lazily from this flag.
                group["capturable"] = True
                for p in params:
                    # A checkpoint written BEFORE this arm landed (or read through
                    # `load_ckpt`'s `map_location="cpu"`) carries a HOST `step`
                    # counter. torch's capturable branch asserts the counter shares
                    # the param's device, and that assert would fire in the eager
                    # WARMUP step — outside any try/except. Move it where the flag
                    # that needs it is set.
                    state = self.opt.adamw.state.get(p, {})
                    if "step" in state and state["step"].device.type != "cuda":
                        state["step"] = state["step"].to(
                            device=params[0].device, dtype=torch.float32
                        )

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
        for group in self.opt.adamw.param_groups:
            yield group, True
        if self.opt.muon is not None:
            for group in self.opt.muon.param_groups:
                yield group, False

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
        missing = [
            f"group {gi}[{i}] shape {tuple(p.shape)}"
            for gi, (group, _) in enumerate(self._groups())
            for i, p in enumerate(group["params"])
            if p.requires_grad and p.grad is None
        ]
        if missing:
            # The step only touches params that have a gradient, so the
            # participating set FREEZES at capture: a param starved here would
            # silently stop being updated for the rest of the run
            # (`dion/cuda_graph.py:231-255`).
            more = f", ... (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            shown = ", ".join(missing[:5]) + more
            return (
                f"{len(missing)} parameter(s) require a gradient but have .grad=None at "
                f"capture, so every replay would leave them frozen: {shown}. Escape: "
                f"capture after every parameter has taken part in a backward pass (raise "
                f"the warmup past whatever starves them), or set requires_grad=False on "
                f"the ones meant to be frozen, or opt_graph=false."
            )
        if not self._lr or not all(t.device.type == "cuda" for _, t in self._lr):
            return (
                "the optimizer's parameters are not on CUDA: a CUDA graph is a "
                "CUDA-only mechanism, and capturing on any other device is not a "
                "degradation this port can make honest. Escape: run with "
                "--device cuda, or opt_graph=false."
            )
        return None

    def capture(self) -> None:
        """Capture one full stage and replay it once — LOUD on any refusal.

        The immediate replay is not optional: capture only RECORDS the ops
        (`dion/cuda_graph.py:257-274`), so a capture without it leaves the
        trajectory one step short.
        """
        why = self._refusals()
        if why is not None:
            loud(f"opt_graph: capture refused — {why}")
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode=CAPTURE_ERROR_MODE):
            self.stage()
        # Publish ONLY on success: a graph from a failed capture is not replayable,
        # and holding it would turn one loud error into every later step replaying
        # garbage (`dion/cuda_graph.py:253-255`).
        self._graph = graph
        self._snapshot_counters()
        graph.replay()

    def _snapshot_counters(self) -> None:
        """Copy the capture step's INT counters; report the reason at this boundary.

        `Counters` also holds the COUNTED fallback's reason under the string key
        `why` (`instrument.py`). A replay runs only recorded device ops, so the
        int counters must be replayed by hand — but the reason is host text: if
        it enters the captured map, every replay calls `Counters.count("why",
        <str>)` and asserts (review round 1 N1, a run-killing crash). It is
        reported ONCE here, through the trainer's cumulative counters, and is
        left out of the replays.
        """
        counters = getattr(self.opt, "counters", None)
        self._captured_counters = (
            {k: v for k, v in counters.items() if isinstance(v, int)} if counters else {}
        )
        reason = counters.why if counters is not None else None
        if reason and self.counters is not None and self.counters is not counters:
            self.counters.why = reason

    # -- the step -----------------------------------------------------------

    def step(self) -> None:
        """One optimizer stage: eager during warmup and after a fallback, replay after.

        A NaN-skipped step must NOT call this (the trainer's rule): the stage is
        what advances the graph, exactly as it is what advances the optimizer.
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
        self._graph.replay()
        if hasattr(self.opt, "steps"):
            self.opt.steps += 1
        # `steps_batched` is the same value every step (one count per shape group),
        # so the capture-time value is the live one and is left alone. Only ints
        # are replayed: `_snapshot_counters` excludes the string `why`, and the
        # guard here means a hand-built/old captured map cannot resurrect N1.
        for name, n in self._captured_counters.items():
            if isinstance(n, int):
                self.opt.counters.count(name, n)

    def _fall_back(self, reason: str) -> None:
        """COUNTED: record why, print it, and stay eager for the rest of the run.

        The groups also get their HOST floats back: the eager path reads `lr` from
        the group dict, and leaving a device tensor there would make the "fallback"
        a subtly different program from the one it claims to be.
        """
        self.reason = reason
        for group, _ in self._lr:
            group["lr"] = self._lr_value
        if self.counters is not None:
            self.counters.count("opt_graph_fallback")
            self.counters.why = f"opt_graph: {reason}"  # type: ignore[attr-defined]
        if self.log is not None:
            self.log(f"opt_graph: capture failed, staying EAGER for the rest of the run — {reason}")

    def release(self) -> None:
        """Drop the graph and restart the warmup (the next step re-captures)."""
        graph, self._graph = self._graph, None
        self._steps = 0
        if graph is None:
            return
        torch.cuda.synchronize()  # never tear a graph down under an in-flight replay
        graph.reset()
