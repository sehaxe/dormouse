"""CUDA-graph capture of the optimizer stage (`train/graph.py`, `train.opt_graph`).

The arm's whole claim is that ONE replay replaces the optimizer stage's hundreds
of launches **without changing what the step computes**. The gates follow from
that, and the two halves are marked:

* **the default path cannot move** — sha256 of a fixed-seed 10-step toy run,
  captured BEFORE this file existed (`/tmp/opencode/capture_opt_graph.py`). A
  re-transcription can drift with the code; a hash cannot. If this goes red after
  an `optim.py` / `loop.py` edit, that edit moved the DEFAULT path.
* **the refusals are LOUD** and each names its escape: a non-CUDA device, a
  `requires_grad` param with `.grad is None`, `muon_sf`, `muon_variant="normuon"`,
  `optim_8bit`. **A refused or failed capture is COUNTED and the run stays EAGER**
  (`opt_graph_fallback` + the reason) — never a crash, never a half-built graph.
* **the LR really is live** — the group hands the optimizer a persistent 0-dim
  tensor, `set_lr` fills it in place, and the Muon update scales with its value.
  That last one is the counter-proof: it would FAIL if the LR were a baked host
  float, which is the trap `dion/cuda_graph.py:70-78` documents.
* **the replay redoes the host bookkeeping** the recorded device ops cannot carry
  (`MixOptimizer.steps`, the arm counters) — a COUNTED arm that goes silent after
  capture is the ADR-0019 defect.
* **GPU (bit-identity)** — eager-vs-replay for a constant LR and for a schedule
  that crosses the capture step, the counter-proof against a frozen LR, and the
  resume round-trip. Written 2026-10-17 and **NOT run**: the card was busy with the
  live b1 run, and the preflight (`test_ns_triton.py`'s) skips rather than lie.
"""

from __future__ import annotations

import hashlib
import subprocess

import pytest
import torch
from conftest import local_pin, tiny_cfg, tiny_run
from torch import nn

from dormouse.config import RunConfig, load_run
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel
from dormouse.train import train
from dormouse.train.graph import GRAPH_WARMUP_STEPS, OptimizerGraph
from dormouse.train.optim import MixOptimizer, clip_grads

# --- the pre-change reference, captured as BYTE HASHES ----------------------
#
# `/tmp/opencode/capture_opt_graph.py`, run against the tree BEFORE this arm
# landed, twice (two processes, identical output):
#   * `_ten_steps` below — the same fixed-seed 10-step Muon run
#     `tests/test_muon_variants.py` pins, which cross-checks the harness; and
#   * a 10-step `train()` on the tiny preset (seed 0), the sha256 of every
#     parameter's bytes in name order — that half covers `loop.py`'s LR writes,
#     `zero_grad` and the grad-clip call site.

_SHAPES = {(2, 3): 1, (64, 64): 2, (8, 128): 2}

_PRE_CHANGE_SHA = (
    "56fb301e85d3e8fed67c66f7fdaf43fb31e56c5ab324a9c6f70607a768944ccd",  # (2, 3) singleton
    "8d4ae51e90c5f777da8e09f2e6bcdf109ec59ebd68b59203b2ff112561935f7e",  # (64, 64)
    "aeff5e961608b40e769067e4dac52bf5af10114d3ce59d14f3187603cc48a968",  # (64, 64)
    "a043caef48fd8a793804502671e8a8baf412177cba44f25fa079acf98db168de",  # (8, 128)
    "2b4189cc2733ec5b1d0d33e1ee330d09d30555558d1cc5861af840511d1e8ec8",  # (8, 128)
)

_PRE_CHANGE_TRAIN_SHA = "b4d2aa542e6efd76b4fb1c6560544e1aad7bd72a920206503d0a347d64c11966"


def _sha(p: torch.Tensor) -> str:
    return hashlib.sha256(p.detach().numpy().tobytes()).hexdigest()


def _shape_model(seed: int = 0) -> nn.Module:
    gen = torch.Generator().manual_seed(seed)
    mod = nn.Module()
    i = 0
    for shape, count in _SHAPES.items():
        for _ in range(count):
            mod.register_parameter(f"p{i}", nn.Parameter(torch.randn(*shape, generator=gen) * 0.05))
            i += 1
    return mod


@local_pin
def test_the_default_path_is_byte_identical():
    """`opt_graph=false` (the default) must not have moved — pinned hashes.

    Two runs, both the pre-arm harness: the Muon-level 10-step run and the
    loop-level 10-step `train()`.
    """
    from dormouse.train.optim import Muon

    mod = _shape_model()
    muon = Muon(mod.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(7)
    for _ in range(10):
        for p in mod.parameters():
            p.grad = torch.randn(*p.shape, generator=gen) * 0.1
        muon.batched_step_from_model(mod)
    assert tuple(_sha(p) for p in mod.parameters()) == _PRE_CHANGE_SHA

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        torch.manual_seed(0)
        run = tiny_run(Path(td), steps=10, log_every=5, lr=0.01)
        model = DormouseModel(run.model)
        train(model, run, log=lambda *_: None)
        h = hashlib.sha256()
        for _, p in sorted(model.named_parameters()):
            h.update(p.detach().numpy().tobytes())
    assert h.hexdigest() == _PRE_CHANGE_TRAIN_SHA


# --- config: the arm round-trips, and a flip is a config diff ----------------


def test_opt_graph_round_trips_and_a_flip_is_a_config_diff():
    run = load_run("small", train_overrides={"steps": 5, "ckpt_name": "x", "opt_graph": True})
    assert run.train.opt_graph is True
    back = RunConfig.from_snapshot(run.snapshot_toml())
    assert back.diff_keys(run) == [] and back.train.opt_graph is True

    from dataclasses import replace

    eager = replace(run, train=replace(run.train, opt_graph=False))
    assert eager.diff_keys(run) == ["train.opt_graph"]

    # An OLD snapshot (written before the field existed) fills the default and is
    # still a legal resume of an eager run: the key must not become a spurious diff.
    text = "\n".join(line for line in run.snapshot_toml().splitlines() if "opt_graph" not in line)
    old = RunConfig.from_snapshot(text)
    assert old.train.opt_graph is False
    assert old.diff_keys(eager) == []


def test_the_trainer_refuses_compile_with_the_graph(tmp_path):
    """A recompile under an active capture aborts it — refuse at the door."""
    run = tiny_run(tmp_path, steps=1, opt_graph=True, torch_compile=True)
    with pytest.raises(LoudError, match="torch_compile"):
        train(DormouseModel(run.model), run, log=lambda *_: None)


# --- the refusals (LOUD) ----------------------------------------------------


def _mix(**over) -> tuple[nn.Module, MixOptimizer]:
    torch.manual_seed(0)
    model = DormouseModel(tiny_cfg())
    return model, MixOptimizer(model, lr=0.01, wd=0.0, **over)


def _with_grads(model: nn.Module) -> None:
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 0.1


def _wrap(opt: MixOptimizer, **over) -> OptimizerGraph:
    return OptimizerGraph(opt, opt.step, counters=Counters(), **over)


def test_capture_refuses_a_non_cuda_device():
    model, opt = _mix()
    _with_grads(model)
    with pytest.raises(LoudError, match="not on CUDA"):
        _wrap(opt).capture()


def test_capture_refuses_a_parameter_without_a_gradient():
    """The participating set freezes at capture, so a starved param must stop the
    capture instead of silently never being updated again."""
    model, opt = _mix()
    _with_grads(model)
    next(iter(model.parameters())).grad = None
    with pytest.raises(LoudError, match=r"\.grad=None"):
        _wrap(opt).capture()


def test_capture_refuses_the_schedule_free_arm():
    model, opt = _mix(muon_sf=True)
    _with_grads(model)
    with pytest.raises(LoudError, match="muon_sf"):
        _wrap(opt).capture()


def test_capture_refuses_the_normuon_variant():
    model, opt = _mix(muon_variant="normuon")
    _with_grads(model)
    with pytest.raises(LoudError, match="alpha"):
        _wrap(opt).capture()


def test_capture_refuses_the_8bit_adamw():
    model, opt = _mix()
    # The swap itself needs bitsandbytes; the FLAG is what capture reads, and the
    # claim is about the flag's mechanism, so it is set directly.
    opt.bnb8bit = True
    _with_grads(model)
    with pytest.raises(LoudError, match="optim_8bit"):
        _wrap(opt).capture()


def test_a_refused_capture_is_counted_and_the_run_stays_eager():
    """The whole "never kill a run" contract, on a CPU box where capture refuses."""
    model, opt = _mix()
    _with_grads(model)
    before = [p.detach().clone() for p in model.parameters()]
    counters = Counters()
    graph = OptimizerGraph(opt, opt.step, warmup_steps=1, counters=counters)
    graph.set_lr(0.01)
    graph.step()  # step 1 is the warmup: eager, no capture attempt, no counter
    assert graph.reason is None and counters.get("opt_graph_fallback") is None

    graph.set_lr(0.01)
    graph.step()  # step 2 attempts the capture, which refuses here -> COUNTED
    assert counters["opt_graph_fallback"] == 1
    assert "not on CUDA" in counters.why and graph.reason is not None
    moved = [
        not torch.equal(a, b.detach()) for a, b in zip(before, model.parameters(), strict=True)
    ]
    assert any(moved), "the stage must still have run, eagerly"
    assert not torch.is_tensor(opt.muon.param_groups[0]["lr"]), (
        "a fallback must hand the eager path its host float back"
    )

    # Permanent: one count, not one per step, and never a second capture attempt.
    graph.set_lr(0.01)
    graph.step()
    assert counters["opt_graph_fallback"] == 1 and graph._graph is None


def test_a_capture_failure_is_counted_with_its_reason():
    """Any exception out of the capture — not just a refusal — is COUNTED."""
    model, opt = _mix()
    _with_grads(model)
    counters = Counters()
    graph = OptimizerGraph(opt, opt.step, warmup_steps=1, counters=counters)
    graph._refusals = lambda: None  # pretend the card is free ...
    graph.capture = lambda: (_ for _ in ()).throw(RuntimeError("capture blew up"))
    graph.set_lr(0.01)
    graph.step()  # warmup
    graph.set_lr(0.01)
    graph.step()  # the simulated failure
    assert counters["opt_graph_fallback"] == 1
    assert "capture blew up" in counters.why and "RuntimeError" in counters.why
    assert graph._graph is None and graph.reason is not None


def test_warmup_must_allow_an_eager_step():
    model, opt = _mix()
    with pytest.raises(LoudError, match="warmup_steps"):
        OptimizerGraph(opt, opt.step, warmup_steps=0)


# --- the LR is a live device tensor ----------------------------------------


def _all_groups(opt: MixOptimizer) -> list[dict]:
    return list(opt.adamw.param_groups) + ([] if opt.muon is None else list(opt.muon.param_groups))


def test_set_lr_fills_the_tensor_in_place_and_folds_a_float_back():
    model, opt = _mix()
    graph = _wrap(opt)
    groups = _all_groups(opt)
    assert groups, "the tiny model must route something"
    for group in groups:
        assert torch.is_tensor(group["lr"]) and group["lr"].ndim == 0
    before = [group["lr"] for group in groups]

    graph.set_lr(0.007)
    for group, tensor in zip(groups, before, strict=True):
        assert group["lr"] is tensor, "the LR tensor was REPLACED, not filled in place"
        assert tensor.item() == pytest.approx(0.007)

    # A hand-rolled schedule that assigns a float replaces the tensor the captured
    # ops recorded, so it has to be folded back into the SAME object — otherwise
    # every later replay reads a tensor nobody writes to.
    for group in groups:
        group["lr"] = 0.5
    graph.set_lr(0.25)
    for group, tensor in zip(groups, before, strict=True):
        assert group["lr"] is tensor and tensor.item() == pytest.approx(0.25)


def test_the_muon_update_scales_with_the_live_lr_tensor():
    """The counter-proof at unit scale: the update must move with the tensor.

    Two identical runs, `lr` and `2·lr` (both powers of two, so the doubling is
    exact in fp32). The parameters start at ZERO so `p_after` IS the update — the
    Muon update is a function of the momentum and the grads, never of `p`, so
    zeroing moves nothing but the cancellation that would otherwise hide the last
    few ulps. If the LR were read as a host float at capture time this gate could
    not fail — which is why it also runs under replay on the card.
    """

    def run(lr: float) -> list[torch.Tensor]:
        torch.manual_seed(0)
        model = DormouseModel(tiny_cfg())
        opt = MixOptimizer(model, lr=lr, wd=0.0)
        graph = OptimizerGraph(opt, opt.step, counters=Counters())
        gen = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
                p.grad = torch.randn(p.shape, generator=gen) * 0.1
        graph.set_lr(lr)
        opt.muon.batched_step_from_model(model)
        return [p.detach().clone() for p in model.parameters()]

    a = run(2.0**-6)
    b = run(2.0**-5)
    assert max(x.abs().max().item() for x in a) > 0.0, "the arm must move a parameter"
    assert all(torch.equal(y, 2 * x) for x, y in zip(a, b, strict=True)), (
        "the Muon update did not scale with the LR tensor's value"
    )


def test_a_replay_redoes_the_host_bookkeeping():
    """`MixOptimizer.step`'s Python counters and the arm counters advance once, at
    capture; a replay runs only the recorded DEVICE ops, so the wrapper must redo
    them or a resume reads a stale step and a COUNTED arm goes silent."""
    model, opt = _mix()
    graph = _wrap(opt)

    class _Fake:
        def __init__(self) -> None:
            self.n = 0

        def replay(self) -> None:
            self.n += 1

    fake = _Fake()
    graph._graph = fake
    graph._captured_counters = {"ns_triton": 1}
    steps = opt.steps
    graph.step()
    assert fake.n == 1 and opt.steps == steps + 1
    assert opt.counters["ns_triton"] == 1


def test_release_restarts_the_warmup():
    """With no graph yet, only the second half of the contract applies — and
    skipping it would let the next step capture on a warmup the caller just
    invalidated."""
    model, opt = _mix()
    graph = _wrap(opt, warmup_steps=2)
    graph._steps = 5
    graph.release()
    assert graph._graph is None and graph._steps == 0


# --- the arm on a CPU box is a COUNTED fallback, end to end -----------------


def test_the_arm_on_cpu_falls_back_to_the_eager_path(tmp_path):
    """`opt_graph=true` on a CPU box must not crash the run: it refuses at the
    capture, COUNTS it with a reason that names its escape, and the rest of the
    run is the eager path.

    The reason here is whichever policy refusal fires first for this model (the
    tiny model starves `loop.attn.beta_proj.weight`, so the `.grad=None` rule
    wins over the CUDA one — both are LOUD and both name an escape). The CUDA
    refusal on a model with every gradient present is
    `test_capture_refuses_a_non_cuda_device`.
    """
    run = tiny_run(tmp_path, steps=GRAPH_WARMUP_STEPS + 3, log_every=100, opt_graph=True)
    lines: list[str] = []
    report = train(DormouseModel(run.model), run, log=lines.append)
    counters = report["counters"]
    assert counters["opt_graph_fallback"] == 1
    assert counters.why.startswith("opt_graph:") and "Escape:" in counters.why
    assert any("staying EAGER" in line for line in lines)
    assert report["nan_skipped"] == 0
    assert report["opt_steps"] == GRAPH_WARMUP_STEPS + 3
    assert report["loss"] == report["loss"]  # finite (not NaN)


# --- GPU: bit-identity (written; NOT run while the b1 run owns the card) -----


def _card_reason() -> str | None:
    """`None` when a GPU gate may run, else why not — the machine discipline.

    `test_ns_triton.py`'s check, kept identical on purpose: `nvidia-smi`/`pgrep`
    rather than `torch.cuda.mem_get_info()`, so the check itself builds no CUDA
    context on a card that is in use.
    """
    if not torch.cuda.is_available():
        return "no CUDA device"
    try:
        busy = subprocess.run(
            ["pgrep", "-f", "dormouse trai[n]"], capture_output=True, text=True, timeout=15
        ).stdout.strip()
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001 - a gate's precondition, not a run's
        return f"cannot check the card ({e})"
    if busy:
        return f"the live training run holds the card (pids {busy}) — one GPU thing at a time"
    mib = int(used.split()[0]) if used else -1
    if mib >= 1024:
        return f"the card already holds {mib} MiB"
    return None


_GPU_REASON = _card_reason()
CUDA = pytest.mark.skipif(_GPU_REASON is not None, reason=_GPU_REASON or "GPU gate")

DEV = "cuda"
STEPS = 8
WARMUP = 3


@pytest.fixture
def quiet_card():
    """Re-check at RUN time (the card can become busy after collection)."""
    why = _card_reason()
    if why is not None:
        pytest.skip(f"GPU busy — nothing measured: {why}")


def _stage(model: nn.Module, opt: MixOptimizer):
    """The loop's own stage: the grad clip, then the mixed step."""

    def stage() -> None:
        clip_grads(model.parameters(), 1.0)
        opt.step()

    return stage


def _run(lrs: list[float], *, warmup_steps: int) -> list[torch.Tensor]:
    """Both arms use the SAME code: the reference is the arm's own eager path.

    A bare float-LR run is NOT the reference: the arm sets AdamW's
    `capturable=True` (the only way torch reads a device-tensor LR), which folds
    the bias correction in on device and so moves the arithmetic. That deviation
    is named in `graph.py`; what the gate below pins is that capture+replay does
    not move it FURTHER.
    """
    torch.manual_seed(0)
    model = DormouseModel(tiny_cfg(), device=DEV)
    opt = MixOptimizer(model, lr=0.01, wd=0.0)
    graph = OptimizerGraph(opt, _stage(model, opt), warmup_steps=warmup_steps)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    gen = torch.Generator(device=DEV).manual_seed(7)
    for lr in lrs:
        graph.set_lr(lr)
        for p in model.parameters():
            p.grad.copy_(torch.randn(p.shape, generator=gen, device=DEV) * 0.1)
        graph.step()
    return [p.detach().clone() for p in model.parameters()]


def _maxdiff(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return max((x - y).abs().max().item() for x, y in zip(a, b, strict=True))


@CUDA
def test_a_constant_lr_replays_bit_identically(quiet_card):
    lrs = [0.01] * STEPS
    eager = _run(lrs, warmup_steps=10**9)  # never captures: the arm's eager path
    captured = _run(lrs, warmup_steps=WARMUP)
    assert _maxdiff(eager, captured) == 0.0, (
        f"capture+replay moved the params by {_maxdiff(eager, captured):.3e}"
    )


@CUDA
def test_a_scheduled_lr_crossing_the_capture_step_replays_bit_identically(quiet_card):
    """The schedule crosses the capture boundary, so a BAKED LR would show up."""
    base = 0.01
    lrs = [base * (0.25 + 1.5 * (t + 1) / STEPS) for t in range(STEPS)]
    eager = _run(lrs, warmup_steps=10**9)
    captured = _run(lrs, warmup_steps=WARMUP)
    assert _maxdiff(eager, captured) == 0.0, (
        f"the captured run did not track the schedule (diff {_maxdiff(eager, captured):.3e})"
    )
    # The counter-proof: a run whose LR IS the capture step's value must differ, or
    # this test would pass even if the LR had been baked at capture.
    frozen = _run([lrs[WARMUP]] * STEPS, warmup_steps=WARMUP)
    assert _maxdiff(captured, frozen) > 1e-3, (
        "the schedule looks FROZEN under replay — the gate cannot fail as written"
    )


@CUDA
def test_a_replay_after_a_resume_matches_the_unbroken_run(quiet_card, tmp_path):
    """`load_ckpt` reads with `map_location="cpu"` (`train/loop.py`), so the
    optimizer state comes back on the host — including AdamW's `step` counter,
    which `Optimizer.load_state_dict` moves back to the param's device for a
    `capturable` group. A host counter would assert in the eager warmup step (or
    freeze the bias correction at the resume step). This is the half of
    `load_ckpt` that touches the optimizer, without the 18 GB artifact."""
    base = 0.01
    lrs = [base * (0.25 + 1.5 * (t + 1) / STEPS) for t in range(STEPS)]
    presteps = 2

    def build():
        torch.manual_seed(0)
        model = DormouseModel(tiny_cfg(), device=DEV)
        return model, MixOptimizer(model, lr=0.01, wd=0.0)

    def draw(model, gen, *, apply: bool) -> None:
        for p in model.parameters():
            g = torch.randn(p.shape, generator=gen, device=DEV) * 0.1
            if apply:
                p.grad.copy_(g)

    def grads_zero(model) -> None:
        for p in model.parameters():
            p.grad = torch.zeros_like(p)

    # (1) the checkpoint: `presteps` steps of the ARM's own eager path, saved and
    # read back exactly the way `load_ckpt` reads it (`map_location="cpu"`).
    model_a, opt_a = build()
    grads_zero(model_a)
    gen_a = torch.Generator(device=DEV).manual_seed(7)
    warm_a = OptimizerGraph(opt_a, _stage(model_a, opt_a), warmup_steps=10**9)
    for t in range(presteps):
        warm_a.set_lr(lrs[t])
        draw(model_a, gen_a, apply=True)
        warm_a.step()
    path = tmp_path / "opt.pt"
    torch.save(opt_a.state_dict(), path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)

    # (2) the resume: both halves restored, then warmup -> capture -> replay.
    model_b, opt_b = build()
    model_b.load_state_dict(model_a.state_dict())
    opt_b.load_state_dict(loaded)
    grads_zero(model_b)
    graph_b = OptimizerGraph(opt_b, _stage(model_b, opt_b), warmup_steps=WARMUP)
    gen_b = torch.Generator(device=DEV).manual_seed(7)
    for _ in range(presteps):
        draw(model_b, gen_b, apply=False)  # keep the grad stream aligned with (3)
    for t in range(presteps, STEPS):
        graph_b.set_lr(lrs[t])
        draw(model_b, gen_b, apply=True)
        graph_b.step()

    # (3) the unbroken run: the same schedule and grads, one capture point.
    unbroken = _run(lrs, warmup_steps=WARMUP)
    resumed = [p.detach().clone() for p in model_b.parameters()]
    assert _maxdiff(unbroken, resumed) == 0.0, (
        f"the resumed graph run diverged by {_maxdiff(unbroken, resumed):.3e}"
    )
