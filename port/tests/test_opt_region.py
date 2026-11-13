"""The optimizer region: preset defaults, the bf16-state gate, and the probes.

The 2026-11-05 kernel profile made the optimizer the largest region of an m400
step (35.4 ms, 23.5%) while the config still ran the fp32 state and a
five-iteration Newton-Schulz chain. This file holds the four claims the change
rests on, each with a way to fail:

1. **the preset recipe is real** — `m400`/`medium` resolve `muon_state_dtype=bf16`
   and `muon_chunk_elems=1<<26`; `small` and an explicit `TrainConfig()` keep the
   historical schema values, and `--set`/`train=` still win.
2. **the bf16-state rung is gated on a MEASURED tolerance, not a sentence** —
   a 100-step CPU run vs fp32 prints the loss delta and the parameter delta in
   bf16 ulp at the parameter's scale, and the quality assertion is against the
   fp32 arm's own seed spread. The pins below are this machine's measurements;
   they are not an assertion that the rung is bit-identical (it is not).
3. **`ns_steps` is a working PROGRESS key** — it round-trips, `diff_keys`
   ignores it, a non-integer is refused at the config and the optimizer, and
   `MixOptimizer.set_ns_steps` repairs what torch's loader reverts.
4. **the chunk cap is a bit-identical launch/memory trade** — a changed cap
   moves no value, and the larger cap enqueues fewer kernels (the CUDA launch
   model in `scripts/bench_muon_launch.py`; the CPU dispatch counter is the
   independent cross-check for "no host syncs added").
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from dormouse.config import PRESET_TRAIN_DEFAULTS, TrainConfig, load_run
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train.optim import (
    MUON_CHUNK_ELEMS,
    MixOptimizer,
    Muon,
    _classic_ns_many,
    clip_grads,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load("bench_muon_launch")
launch_profile = _load("launch_profile")

#: The bf16-state gate fixture: `tiny` at b2 x s64, 100 steps, fp32 NS.
GATE_STEPS = 100
GATE_BATCH, GATE_SEQ = 2, 64
#: This machine's measurement (`python scripts/ab_opt_region.py gate`), pinned
#: so a drift in the rung is a red test, not a silent widening. `loss_rel` is
#: the max per-step relative loss delta vs fp32 for the same seed; `param_ulp`
#: is the largest per-tensor |p_bf16 - p_fp32| in bf16 ulp AT THE PARAMETER'S
#: SCALE (`eps_bf16 * max|p_fp32|`) after the 100 steps. The rung is NOT
#: bit-identical: a 1-ulp parameter gate is unreachable without the stochastic
#: rounding the rung's docstring names as the next rung.
#: Re-pinned 2026-11-11 with Muon's Newton–Schulz contracting on the small side
#: (`optim.py`): that is a re-association, but a fp32 1e-7 move still walks a
#: 100-step curve, so both numbers moved (loss 1.8%, ulp 16%) and the QUALITY
#: bar above — `loss_rel < seed_spread` — is what the rung is judged on.
GATE_LOSS_REL = 0.00035072604757507804
GATE_PARAM_ULP = 8.116083185513272


def _gate_run(rung: str, seed: int) -> tuple[list[float], dict[str, torch.Tensor]]:
    """One 100-step curve from `seed`, the trainer's clip included."""
    torch.manual_seed(seed)
    run = load_run(
        "tiny",
        train_overrides={
            "steps": GATE_STEPS,
            "batch": GATE_BATCH,
            "seq_len": GATE_SEQ,
            "muon_state_dtype": rung,
            "ns_dtype": "fp32",
            "ns_steps": 5,
            "muon_chunk_elems": MUON_CHUNK_ELEMS,
        },
    )
    model = DormouseModel(run.model, device=torch.device("cpu")).train()
    opt = MixOptimizer(
        model,
        lr=1e-3,
        wd=0.0,
        ns_steps=5,
        muon_state_dtype=rung,
        ns_dtype="fp32",
        muon_chunk_elems=MUON_CHUNK_ELEMS,
    )
    # A FRESH batch every step, from a generator independent of the model
    # seed: a fixed batch overfits to ~0 loss in 100 steps and the relative
    # comparison becomes 0/0 noise. Both arms see the same stream.
    data = torch.Generator().manual_seed(1234)
    losses: list[float] = []
    for _ in range(GATE_STEPS):
        x = torch.randint(0, 256, (GATE_BATCH, GATE_SEQ), generator=data)
        y = torch.randint(0, 256, (GATE_BATCH, GATE_SEQ), generator=data)
        opt.zero_grad()
        _logits, loss, _counters = model(x, y)
        loss.backward()
        if run.train.grad_clip > 0:
            clip_grads(model.parameters(), run.train.grad_clip)
        opt.step()
        losses.append(float(loss.detach()))
    return losses, {n: p.detach().clone() for n, p in model.named_parameters()}


def _max_rel(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) / max(abs(x), 1e-12) for x, y in zip(a, b, strict=True))


def _param_ulp(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    eps = torch.finfo(torch.bfloat16).eps
    tiny = torch.finfo(torch.bfloat16).tiny
    worst = 0.0
    for name, p in a.items():
        q = b[name]
        scale = max(float(p.abs().max()), tiny)
        worst = max(worst, float((p - q).abs().max()) / (eps * scale))
    return worst


# --- 1. the preset recipe ----------------------------------------------------


def test_the_measured_presets_carry_the_optimizer_recipe():
    assert PRESET_TRAIN_DEFAULTS == {
        "m400": {"muon_state_dtype": "bf16", "muon_chunk_elems": 1 << 26},
        "medium": {"muon_state_dtype": "bf16", "muon_chunk_elems": 1 << 26},
    }
    for preset, path in (("m400", "m400"), ("m400", "configs/m400.toml"), ("medium", "medium")):
        run = load_run(path)
        assert run.train.muon_state_dtype == "bf16", (preset, path)
        assert run.train.muon_chunk_elems == 1 << 26, (preset, path)
        snap = run.snapshot_toml()
        assert 'muon_state_dtype = "bf16"' in snap
        assert f"muon_chunk_elems = {1 << 26}" in snap
    # The schema and every smaller preset stay where they were.
    assert TrainConfig().muon_state_dtype == "fp32"
    assert TrainConfig().muon_chunk_elems == MUON_CHUNK_ELEMS
    small = load_run("small")
    assert small.train.muon_state_dtype == "fp32"
    assert small.train.muon_chunk_elems == MUON_CHUNK_ELEMS
    # An explicit --set (or a caller's TrainConfig) is still the last word.
    off = load_run(
        "m400",
        train_overrides={"muon_state_dtype": "fp32", "muon_chunk_elems": MUON_CHUNK_ELEMS},
    )
    assert off.train.muon_state_dtype == "fp32"
    assert off.train.muon_chunk_elems == MUON_CHUNK_ELEMS
    explicit = load_run("m400", train=TrainConfig())
    assert explicit.train.muon_state_dtype == "fp32"
    assert explicit.train.muon_chunk_elems == MUON_CHUNK_ELEMS
    # The state dtype is STRICT (a flip is a different program); the chunk cap
    # is a PROGRESS key (a split moves no value).
    assert off.diff_keys(load_run("m400")) == ["train.muon_state_dtype"]
    assert off.diff_keys(load_run("m400", train_overrides={"muon_chunk_elems": 1 << 25})) == [
        "train.muon_state_dtype"
    ]


# --- 2. the bf16-state gate --------------------------------------------------


@pytest.mark.pins
def test_the_bf16_state_rung_is_within_the_fp32_seed_spread_and_the_pins_hold():
    """The 100-step CPU gate. It explicitly does NOT claim bit equality."""
    fp32, fp32_params = _gate_run("fp32", seed=0)
    bf16, bf16_params = _gate_run("bf16", seed=0)
    # The quality bar is the CONTROL's own seed spread: a different init moves
    # the curve more than the state rung does, which is this port's rule for a
    # quality tie (AGENTS 1.2 / AB-PROTOCOL).
    fp32_other, _ = _gate_run("fp32", seed=1)
    seed_spread = _max_rel(fp32, fp32_other)
    loss_rel = _max_rel(fp32, bf16)
    param_ulp = _param_ulp(fp32_params, bf16_params)
    print(
        f"\n[bf16-state gate] 100 steps: max rel loss {loss_rel:.6g} vs fp32; "
        f"fp32 seed spread {seed_spread:.6g}; worst parameter delta "
        f"{param_ulp:.3g} bf16 ulp at scale"
    )
    assert loss_rel < seed_spread, (
        f"the state rung moved the 100-step curve further ({loss_rel:.3g}) than a "
        f"seed change does ({seed_spread:.3g}) — not a tie"
    )
    # The measured pins: a change to the rung moves these, and the test says so
    # instead of letting a wider tolerance hide it. (Captured on this CPU;
    # `scripts/ab_opt_region.py gate` prints the same numbers on demand.)
    assert loss_rel == pytest.approx(GATE_LOSS_REL, abs=1e-6)
    assert param_ulp == pytest.approx(GATE_PARAM_ULP, abs=1e-3)


# --- 3. ns_steps as a progress key ------------------------------------------


def test_ns_steps_round_trips_as_a_progress_key_and_refuses_a_non_integer():
    base = load_run("small")
    three = load_run("small", train_overrides={"ns_steps": 3})
    assert base.train.ns_steps == 5 and three.train.ns_steps == 3
    assert three.diff_keys(base) == [], "ns_steps is not exempt on a resume"
    from dormouse.config import RunConfig

    assert RunConfig.from_snapshot(three.snapshot_toml()).train.ns_steps == 3
    for bad in (3.5, 0, True, "3"):
        with pytest.raises(LoudError, match="ns_steps"):
            load_run("small", train_overrides={"ns_steps": bad})
    model = DormouseModel(
        load_run("tiny", train_overrides={"seq_len": GATE_SEQ}).model, device="cpu"
    )
    for bad in (3.5, 0, True):
        with pytest.raises(LoudError, match="ns_steps"):
            Muon(model.parameters(), ns_steps=bad)
        with pytest.raises(LoudError, match="ns_steps"):
            MixOptimizer(model, lr=1e-3, wd=0.0, ns_steps=bad)


def test_a_resume_with_a_new_ns_steps_reaches_the_live_muon_group(tmp_path):
    """The full seam, not just the helper: `train(resume=True)` re-applies it."""
    import dataclasses

    from conftest import tiny_run

    from dormouse.train.loop import train

    first = tiny_run(tmp_path, steps=2, ckpt_every=2, log_every=10)
    train(DormouseModel(first.model), first, log=lambda *_: None)
    second = dataclasses.replace(first, train=dataclasses.replace(first.train, steps=4, ns_steps=3))
    assert second.diff_keys(first) == []

    import dormouse.train.loop as train_loop

    real = train_loop.MixOptimizer
    built: list = []

    def spy(model, **kw):
        opt = real(model, **kw)
        built.append(opt)
        return opt

    train_loop.MixOptimizer = spy
    try:
        train(DormouseModel(second.model), second, resume=True, log=lambda *_: None)
    finally:
        train_loop.MixOptimizer = real
    assert built and built[0].muon is not None
    assert built[0].muon.param_groups[0]["ns_steps"] == 3, (
        "the progress key was reverted by torch's optimizer loader"
    )


def test_set_ns_steps_repairs_what_the_checkpoint_loader_reverts():
    """torch's loader replaces `param_groups`; the progress key must win."""
    torch.manual_seed(0)
    model = nn.Module()
    for name, shape in (("a", (64, 64)), ("b", (64, 64))):
        model.register_parameter(name, nn.Parameter(torch.randn(*shape) * 0.05))
    model.register_parameter("bias", nn.Parameter(torch.zeros(8)))
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    five = MixOptimizer(model, lr=1e-3, wd=0.0, ns_steps=5)
    sd = five.state_dict()
    three = MixOptimizer(model, lr=1e-3, wd=0.0, ns_steps=3)
    three.load_state_dict(sd)
    assert three.muon.param_groups[0]["ns_steps"] == 5, "the loader is expected to revert"
    three.set_ns_steps(3)
    assert three.muon.param_groups[0]["ns_steps"] == 3
    with pytest.raises(LoudError, match="ns_steps"):
        three.set_ns_steps(3.5)


# --- 4. the chunk cap: bit-identical values, fewer launches ------------------


def test_the_chunk_cap_moves_no_value():
    torch.manual_seed(0)
    xs = [
        torch.randn(3, 96, 64) * 1e-3,
        torch.randn(5, 64, 64) * 1e-3,
        torch.randn(96, 64) * 1e-3,
        torch.randn(2, 32, 128) * 1e-3,
    ]
    small = _classic_ns_many(xs, 3, chunk_elems=3 * 96 * 64)
    large = _classic_ns_many(xs, 3, chunk_elems=1 << 26)
    assert all(torch.equal(a, b) for a, b in zip(small, large, strict=True))


def _chunk_fixture(n: int = 600) -> nn.Module:
    """`n` x [1024, 64] Muon parameters: 39.3M elements, two schema chunks."""
    torch.manual_seed(0)
    model = nn.Module()
    for i in range(n):
        model.register_parameter(f"p{i}", nn.Parameter(torch.randn(1024, 64) * 0.02))
    model.register_parameter("bias", nn.Parameter(torch.zeros(8)))
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 0.05
    return model


def test_the_larger_chunk_cap_enqueues_fewer_launches_and_adds_no_host_sync():
    model = _chunk_fixture()
    base = MixOptimizer(model, lr=1e-3, wd=0.0, muon_chunk_elems=MUON_CHUNK_ELEMS)
    c0 = bench.LaunchCounter()
    with c0:
        base.step()
    big = MixOptimizer(model, lr=1e-3, wd=0.0, muon_chunk_elems=1 << 26)
    c1 = bench.LaunchCounter()
    with c1:
        big.step()
    print(f"\n[chunk cap] launches {c0.launches} -> {c1.launches} on 600 x [1024, 64]")
    assert c1.launches < c0.launches
    # The CPU dispatch counter's own sync ledger: no `.item()`/`bool()` added.
    c2 = launch_profile.DispatchCounter()
    c2.stage = "optimizer"
    with c2.observe(model):
        big.step()
    assert not c2.syncs, f"the optimizer step added host syncs: {dict(c2.syncs)}"
