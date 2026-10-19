"""The distributed lane: the 2-rank CPU/gloo gate, plus the refusals.

The integration tests run the REAL entry (`tools/dist_gate.py`, the same file the
owner runs by hand) as two OS processes under `torch.distributed.run` — no
monkeypatched process group, no in-process fake. That is the honest maximum this
one-GPU box allows: the data-parallel math and the loop's seams are proved on
the CPU arm, and the multi-GPU wiring is the same code with `nccl` and one rank
per card (which needs two cards and is stated as unverified in
`docs/architecture/multi-gpu.md`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from conftest import TINY

from dormouse.config import ModelConfig, RunConfig, TrainConfig
from dormouse.instrument import LoudError
from dormouse.train.device import cpu_caps, resolve_policy
from dormouse.train.distributed import (
    LINK_EFFICIENCY,
    Parallel,
    RankShape,
    allreduce_seconds,
    bucket_mb_for,
    choose_backend,
    link_gbps,
    parse_capacity,
    rank_byte_offset,
    suggest_capacity,
)

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tools" / "dist_gate.py"


def _run(args: list[str], tmp: Path, *, world: int = 1, timeout: int = 900) -> None:
    """Run the driver; two ranks go through torchrun, one runs plainly."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    if world == 1:
        cmd = [sys.executable, str(DRIVER), *args]
    else:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={world}",
            str(DRIVER),
            *args,
        ]
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    assert p.returncode == 0, (
        f"dist driver failed ({' '.join(cmd)}):\nstdout:\n{p.stdout[-4000:]}\n"
        f"stderr:\n{p.stderr[-4000:]}"
    )


def _load(tmp: Path, name: str) -> dict:
    return json.loads((tmp / f"{name}.json").read_text())


# --- the pure pieces --------------------------------------------------------


def test_capacity_profile_parsing_and_offsets():
    cfg = TrainConfig(batch=2, seq_len=64)
    assert parse_capacity("", world_size=2, cfg=cfg, max_seq_len=64) == [
        RankShape(2, 64),
        RankShape(2, 64),
    ]
    shapes = parse_capacity("4,2", world_size=2, cfg=cfg, max_seq_len=64)
    assert shapes == [RankShape(4, 64), RankShape(2, 64)]
    assert parse_capacity("4x32,2x16", world_size=2, cfg=cfg, max_seq_len=64) == [
        RankShape(4, 32),
        RankShape(2, 16),
    ]
    # The offset is the sum of the windows BEFORE this rank, in bytes.
    assert rank_byte_offset(shapes, 0) == 0
    assert rank_byte_offset(shapes, 1) == 4 * 65
    assert rank_byte_offset([RankShape(2, 64)] * 2, 1) == 2 * 65
    with pytest.raises(LoudError) as e:
        parse_capacity("4", world_size=2, cfg=cfg, max_seq_len=64)
    assert "one entry each" in str(e.value)
    with pytest.raises(LoudError) as e:
        parse_capacity("4x999,2", world_size=2, cfg=cfg, max_seq_len=64)
    assert "max_seq_len" in str(e.value)
    with pytest.raises(LoudError) as e:
        parse_capacity("4,bad", world_size=2, cfg=cfg, max_seq_len=64)
    assert "B` or `BxT" in str(e.value)


def test_suggest_capacity_splits_the_global_batch_by_measured_rate():
    """3:1 rates -> 3:1 batches, the global total preserved, every rank >= 1."""
    shapes = suggest_capacity([3.0, 1.0], total_batch=8, seq_len=64)
    assert [s.batch for s in shapes] == [6, 2]
    assert sum(s.batch for s in shapes) == 8
    assert [s.seq_len for s in shapes] == [64, 64]
    # Rounding keeps the total exactly, including the smallest shares.
    for indices in ([1.0, 1.0, 1.0], [10.0, 1.0, 1.0], [2.0, 3.0, 5.0]):
        out = suggest_capacity(indices, total_batch=10, seq_len=16)
        assert sum(s.batch for s in out) == 10
        assert all(s.batch >= 1 for s in out)
    with pytest.raises(AssertionError):
        suggest_capacity([1.0, 1.0], total_batch=1, seq_len=16)


def test_pcie_arithmetic_is_the_ring_formula():
    assert link_gbps(3, 16) == pytest.approx(15.76)
    assert link_gbps(None, 16) is None and link_gbps(9, 16) is None
    # P=2 moves exactly one payload; P=4 moves 1.5 payloads per rank.
    one = allreduce_seconds(276 << 20, 2, 15.76)
    assert one == pytest.approx((276 << 20) / (LINK_EFFICIENCY * 15.76e9), rel=1e-9)
    assert allreduce_seconds(276 << 20, 4, 15.76) == pytest.approx(1.5 * one, rel=1e-9)
    assert allreduce_seconds(0, 2, 10) == 0.0
    assert bucket_mb_for(276 << 20) == 34  # round(276/8 = 34.5) -> 34 (banker's)
    assert bucket_mb_for(1 << 20) == 4  # the floor
    assert bucket_mb_for(1 << 40) == 128  # the ceiling


def test_backend_choice_refuses_a_cuda_fleet_without_nccl():
    """The brief's refusal: NCCL missing on a GPU run stops, never gloo-falls-back."""
    assert choose_backend("auto", "cuda", nccl_ok=True, gloo_ok=True) == "nccl"
    assert choose_backend("auto", "cpu", nccl_ok=False, gloo_ok=True) == "gloo"
    with pytest.raises(LoudError) as e:
        choose_backend("auto", "cuda", nccl_ok=False, gloo_ok=True)
    msg = str(e.value)
    assert "without NCCL" in msg and "gloo" in msg and "Escape" in msg
    # An EXPLICIT gloo on CUDA is legal and COUNTED.
    lines: list[str] = []
    assert choose_backend("gloo", "cuda", nccl_ok=True, gloo_ok=True, log=lines.append) == "gloo"
    assert lines and "host memory" in lines[0]
    with pytest.raises(LoudError):
        choose_backend("nccl", "cpu", nccl_ok=True, gloo_ok=True)
    with pytest.raises(LoudError):
        choose_backend("mpi", "cpu", nccl_ok=True, gloo_ok=True)


def _parallel(shard: str = "none", backend: str = "gloo") -> Parallel:
    policy = resolve_policy(cpu_caps(), "fp32")
    return Parallel(
        rank=0,
        world_size=2,
        backend=backend,
        device=torch.device("cpu"),
        shapes=[RankShape(2, 64), RankShape(2, 64)],
        reduce_dtype="fp32",
        shard_kind=shard,
        policy=policy,
        policies=[policy, policy],
        bucket_mb=0,
    )


def _run_cfg(**train_over) -> RunConfig:
    return RunConfig(
        source="dist-test",
        model=ModelConfig(**{**TINY, "moe_capacity": 0.0}),
        train=TrainConfig(**train_over),
    )


def test_loss_scale_is_the_token_weighted_mean():
    p = _parallel()
    assert p.loss_scale == pytest.approx(1.0), "equal shapes = DDP's own mean"
    q = Parallel(
        rank=0,
        world_size=2,
        backend="gloo",
        device=p.device,
        shapes=[RankShape(4, 64), RankShape(2, 64)],
        reduce_dtype="fp32",
        shard_kind="none",
        policy=p.policy,
        policies=p.policies,
        bucket_mb=0,
    )
    # q's rank-0 loss is scaled by 2*256/384 = 4/3; the reducer's /2 lands on 2/3.
    assert q.loss_scale == pytest.approx(4 / 3)
    assert q.global_tokens == 6 * 64


def test_fsdp_with_muon_is_refused_with_the_measured_cause():
    """The integration probe's result: FSDP2 + the stacked Muon state dies."""
    with pytest.raises(LoudError) as e:
        _parallel("fsdp").refuse_incompatible(_run_cfg(opt="mix"), resume=False)
    msg = str(e.value)
    assert "_foreach_add_" in msg and "DTensor" in msg and "dist_shard=none" in msg
    # ...and the AdamW control arm passes the same check (with no checkpoint in
    # the budget: an FSDP2 state dict is not the port's single-file format).
    _parallel("fsdp").refuse_incompatible(_run_cfg(opt="adamw", ckpt_every=10_000), resume=False)


def test_dist_refuses_state_that_does_not_reduce():
    """Per-rank state that looks synchronized: engram_ram, LFB, capacity drops."""
    for over, needle in (
        ({"engram_ram": True}, "host n-gram tables"),
        ({"moe_lb_coef": 0.5}, "FORWARD-MUTATING"),
        ({"moe_topk": 2, "moe_capacity": 1.0}, "DROPS the assignments"),
        ({"opt_graph": True}, "opt_graph"),
        ({"torch_compile": True}, "torch_compile"),
    ):
        model_over = {**TINY, "moe_capacity": 0.0}
        model_over.update({k: v for k, v in over.items() if k.startswith("moe")})
        run = RunConfig(
            source="dist-test",
            model=ModelConfig(**model_over),
            train=TrainConfig(**{k: v for k, v in over.items() if not k.startswith("moe")}),
        )
        with pytest.raises(LoudError) as e:
            _parallel().refuse_incompatible(run, resume=False)
        assert needle in str(e.value), (over, str(e.value))


def test_train_dist_refuses_a_prebuilt_model_under_a_dtype_request():
    from dormouse.train.distributed import train_dist

    run = RunConfig(
        source="t",
        model=ModelConfig(**{**TINY, "dtype": "auto"}),
        train=TrainConfig(dist=True),
    )
    with pytest.raises(LoudError) as e:
        train_dist(run, log=lambda *_: None, model=object(), device_type="cpu")
    assert "captured the old dtype" in str(e.value)


# --- the 2-rank gate (the real entry, two OS processes) ---------------------


@pytest.fixture(scope="module")
def fleet20(tmp_path_factory) -> Path:
    """The 20-step reference and the 2-rank fleet at the same global batch."""
    out = tmp_path_factory.mktemp("fleet20")
    _run(["--mode", "single", "--out", str(out), "--steps", "20", "--batch", "4"], out)
    _run(
        ["--mode", "rank", "--out", str(out), "--steps", "20", "--batch", "2"],
        out,
        world=2,
    )
    return out


def test_two_rank_loss_curve_matches_the_single_process_run(fleet20):
    """(a) 20 steps, the fleet's curve against one process at the global batch.

    The tolerance is 1e-2 on the logged loss (4 decimals), and the residual is
    NOT a gradient error: after ONE step the parameters agree to 3e-8 relative
    (`test_one_step_gradient_matches_the_hand_computed_global_gradient`), and
    what grows afterwards is a DISCRETE routing flip — a 1e-7 parameter
    difference changes one token's top-k expert and moves that step's loss by a
    few 1e-3. The curves' endpoints agree to 2e-3.
    """
    single = _load(fleet20, "single")
    fleet = _load(fleet20, "rank0")
    assert len(single["losses"]) == len(fleet["losses"]) == 20
    diffs = [abs(a - b) for a, b in zip(single["losses"], fleet["losses"], strict=True)]
    assert max(diffs) <= 1e-2, (max(diffs), single["losses"], fleet["losses"])
    assert abs(single["losses"][-1] - fleet["losses"][-1]) <= 5e-3
    # The first step's loss is the same number: same data, same parameters.
    assert single["losses"][0] == pytest.approx(fleet["losses"][0], abs=1e-3)


def test_ranks_are_bit_identical_after_every_step(fleet20):
    """(b) the reducer's contract: identical parameters on every rank."""
    a, b = _load(fleet20, "rank0"), _load(fleet20, "rank1")
    assert a["fingerprint"] == b["fingerprint"]
    pa = torch.load(fleet20 / a["params"])
    pb = torch.load(fleet20 / b["params"])
    assert all(torch.equal(x, y) for x, y in zip(pa, pb, strict=True))


@pytest.fixture(scope="module")
def equal_one_step(tmp_path_factory) -> Path:
    """One step at the global batch: the EXACT equivalence (no chaos)."""
    out = tmp_path_factory.mktemp("equal1")
    _run(["--mode", "single", "--out", str(out), "--steps", "1", "--batch", "4"], out)
    _run(
        ["--mode", "rank", "--out", str(out), "--steps", "1", "--batch", "2"],
        out,
        world=2,
    )
    return out


def test_one_step_gradient_matches_the_hand_computed_global_gradient(equal_one_step):
    """The gradient IS the global batch's, to fp round-off (3e-8 measured)."""
    single = _load(equal_one_step, "single")
    fleet = _load(equal_one_step, "rank0")
    assert single["losses"][0] == pytest.approx(fleet["losses"][0], abs=1e-3)
    ps = torch.load(equal_one_step / single["params"])
    pf = torch.load(equal_one_step / fleet["params"])
    d = max(float((x - y).abs().max()) for x, y in zip(ps, pf, strict=True))
    scale = max(float(x.abs().max()) for x in ps)
    assert d / scale <= 1e-6, (d, scale)


@pytest.fixture(scope="module")
def unequal_one_step(tmp_path_factory) -> Path:
    """Rank 0 batch 4, rank 1 batch 2 — stage 3's capacity profile."""
    out = tmp_path_factory.mktemp("unequal1")
    _run(
        ["--mode", "gradref", "--out", str(out), "--batch", "4", "--capacity", "4,2"],
        out,
    )
    _run(
        [
            "--mode",
            "rank",
            "--out",
            str(out),
            "--steps",
            "1",
            "--batch",
            "4",
            "--capacity",
            "4,2",
        ],
        out,
        world=2,
    )
    return out


def test_unequal_batches_apply_the_token_weighted_mean(unequal_one_step):
    """(stage 3) the effective gradient is `(t0·g0 + t1·g1)/(t0+t1)`, hand-computed.

    `gradref` builds the two local gradients from the ranks' OWN byte windows
    (4x64 and 2x64 tokens at the documented offsets), weights them 4/6 and 2/6,
    and steps the same `MixOptimizer` once. The fleet's parameters after its one
    step must equal that, and the ranks must agree with each other.
    """
    ref = _load(unequal_one_step, "gradref")
    a, b = _load(unequal_one_step, "rank0"), _load(unequal_one_step, "rank1")
    assert ref["shapes"] == ["4x64", "2x64"] and ref["tokens"] == 6 * 64
    assert a["fingerprint"] == b["fingerprint"], "the ranks must stay identical"
    pr = torch.load(unequal_one_step / ref["params"])
    pf = torch.load(unequal_one_step / a["params"])
    d = max(float((x - y).abs().max()) for x, y in zip(pr, pf, strict=True))
    scale = max(float(x.abs().max()) for x in pr)
    assert d / scale <= 1e-4, (d, scale)


def test_fsdp2_adamw_two_rank_run_works(tmp_path):
    """FSDP2 survives the AdamW control arm — and only that arm (the refusal test).

    The sharded run trains 20 steps and its GATHERED parameters are identical
    across ranks. This is the evidence behind `refuse_incompatible`'s "FSDP2 is
    available for opt=adamw" sentence; the Muon arm is refused with the measured
    `_foreach_add_` error.
    """
    _run(
        [
            "--mode",
            "rank",
            "--out",
            str(tmp_path),
            "--steps",
            "20",
            "--batch",
            "2",
            "--shard",
            "fsdp",
            "--opt",
            "adamw",
        ],
        tmp_path,
        world=2,
    )
    a, b = _load(tmp_path, "rank0"), _load(tmp_path, "rank1")
    assert len(a["losses"]) == 20 and a["losses"][-1] < a["losses"][0]
    assert a["fingerprint"] == b["fingerprint"], "FSDP2's gathered params must agree"
