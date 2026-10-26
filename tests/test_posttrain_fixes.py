"""Round-1 post-training bug gates: M2, m3, m4, m5.

Four findings from `docs/reviews/2026-10-24-bug-ledger.md`, each pinned by the
gate the fix brief names and each proved to fail on a scratch revert:

* **M2** — the SFT in-loop eval never rewound its byte stream, so two evals at
  frozen weights read different windows (0.0106 nats of pure position noise).
* **m3** — SFT/RL called themselves copies of `train()`'s NaN firewall but
  checked only the loss scalar; a finite loss with a non-finite gradient stepped
  and poisoned `exp_avg_sq`. The two healthy-step SHAs below are the
  second gate: the fix may not move a bit on a healthy step.
* **m4** — `--seed` reached the snapshot but never `torch.manual_seed`, so two
  runs of one config started from different weights.
* **m5** — SFT/RL silently accepted `muon_sf`/`muon_variant`/`grad_accum`.
  The direction arms are REFUSED; SFT's `grad_accum` is HONOURED and must match
  a `batch*k` wide step (RL refuses it: one RL step is one prompt group).

Every gate is CPU and tiny; the numbers are semantic, not a card measurement.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from conftest import tiny_cfg, write_corpus
from torch import nn
from typer.testing import CliRunner

from dormouse.config import RunConfig, TrainConfig
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel
from dormouse.post.rl_train import rl_train
from dormouse.train.loop import load_model_half
from dormouse.train.sft import build_sft_corpus, sft_train

SEQ, BATCH = 64, 2
_RL_TRAIN = sys.modules["dormouse.post.rl_train"]

#: The healthy-step golden, captured from `main` (`b2e0a1e`) before the fix:
#: sha256 over `model.state_dict()` (keys sorted, raw bytes) after 3 SFT steps
#: of `_sft_run(steps=3, log_every=3)`, `torch.manual_seed(0)` — and the step-3
#: loss field verbatim. m3's fix must be a multiply-by-1.0 / selection identity.
MAIN_SFT_SHA = "a71d7f49ecfed77936a9580e5493b0b3e338198268c79326a6efe6a76c8cceb1"
MAIN_SFT_LOSS = "4.6880"
#: The same golden for 3 RL steps (`_rl_run(steps=3, ckpt_every=3)`): sha256
#: over the checkpoint's model half, `torch.manual_seed(0)`.
MAIN_RL_SHA = "ad3d8469f02b519bb4c6dfeb30d96f9f077d64e487d4731fe6781a5a1616e185"


def _sha_state(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _sha(model: nn.Module) -> str:
    return _sha_state(model.state_dict())


def _sft_run(
    tmp_path: Path, *, corpus: Path | None = None, model_over: dict | None = None, **over
) -> RunConfig:
    data = corpus if corpus is not None else tmp_path / "sft.bin"
    if corpus is None:
        build_sft_corpus(data, seed=1, seq_len=SEQ, n_tasks=32)
    cfg = TrainConfig(
        steps=3,
        batch=BATCH,
        seq_len=SEQ,
        lr=0.01,
        log_every=1,
        ckpt_every=1000,
        ckpt_name="s",
        out_dir=str(tmp_path / "ck"),
        data=str(data),
    )
    return RunConfig(
        source="fix-sft",
        model=tiny_cfg(**(model_over or {})),
        train=dataclasses.replace(cfg, **over),
    )


def _rl_run(tmp_path: Path, **over) -> RunConfig:
    cfg = TrainConfig(
        steps=3,
        batch=2,
        seq_len=64,
        lr=0.01,
        log_every=1,
        ckpt_every=1000,
        ckpt_name="r",
        out_dir=str(tmp_path / "ck"),
        rl_task_cls="math",
        rl_tasks=2,
        rl_group=2,
        rl_budget=96,
        rl_holdout=1,
    )
    return RunConfig(
        source="fix-rl",
        model=tiny_cfg(max_seq_len=128),
        train=dataclasses.replace(cfg, **over),
    )


def _preset(tmp_path: Path, *, max_seq_len: int = 64) -> Path:
    p = tmp_path / "tiny.toml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "\n".join(
            [
                "d_model = 64",
                "n_heads = 4",
                "head_dim = 16",
                "d_ffn = 128",
                f"max_seq_len = {max_seq_len}",
                "max_iter = 2",
                "rank = 8",
                "shared_rank = 16",
                "moe_topk = 2",
                "moe_capacity = 1.0",
                "engram_orders = [2, 3]",
                "engram_rows = 64",
                "engram_dim = 8",
                "bf16 = false",
            ]
        )
        + "\n"
    )
    return p


# --------------------------------------------------------------------------- M2


def test_sft_eval_rewinds_the_stream_at_frozen_weights(tmp_path: Path):
    """Two evals of one checkpoint must be the SAME number.

    The holdout is 8 records; one eval window is `eval_batches=2 × batch=2 = 4`,
    so without `rewind()` the second eval reads records 4..7 and moves the CE
    purely by position (the ledger measured 0.0106 nats on this shape). `lr=0`
    and `wd=0` freeze the weights exactly, so any difference is the stream.
    """
    corpus = tmp_path / "sft.bin"
    build_sft_corpus(corpus, seed=1, seq_len=SEQ, n_tasks=32, holdout=8)
    hold = tmp_path / "sft.bin.holdout"
    log_path = tmp_path / "run.jsonl"
    run = _sft_run(
        tmp_path,
        corpus=corpus,
        steps=4,
        log_every=10,
        lr=1e-12,
        wd=0.0,
        eval=True,
        eval_every=2,
        eval_batches=2,
        eval_data=str(hold),
        json_log=str(log_path),
    )
    sft_train(DormouseModel(run.model), run, log=lambda *_: None)
    import json

    evals = [
        json.loads(ln)
        for ln in log_path.read_text().splitlines()
        if json.loads(ln)["event"] == "sft_eval"
    ]
    assert len(evals) == 2, evals
    a, b = evals
    assert abs(a["ce"] - b["ce"]) < 1e-9, (a, b)
    bpb = lambda row: row["ce"] / torch.log(torch.tensor(2.0)).item()  # noqa: E731
    assert abs(bpb(a) - bpb(b)) < 1e-9
    assert abs(a["first"] - b["first"]) < 1e-9, (a, b)
    assert a["records"] == b["records"] == 4


# --------------------------------------------------------------------------- m3


class _InfGrad(nn.Module):
    """A finite loss whose gradient is non-finite: `logits.grad *= inf`."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x, targets=None, counters=None):
        logits, loss, counters = self.inner(x, targets, counters)
        logits.register_hook(lambda g: g * float("inf"))
        return logits, loss, counters


def test_sft_non_finite_gradient_with_a_finite_loss_is_skipped(tmp_path: Path):
    """The m3 hole: the old check was `isfinite(loss)` only, so this stepped."""
    run = _sft_run(tmp_path, steps=1)
    model = _InfGrad(DormouseModel(run.model))
    before = [p.detach().clone() for p in model.parameters()]
    report = sft_train(model, run, log=lambda *_: None)
    assert report["nan_skipped"] == 1 and report["counters"]["nan_skipped"] == 1
    assert report["opt_steps"] == 0, "a non-finite gradient must not reach the optimizer"
    for p, b in zip(model.parameters(), before, strict=True):
        assert torch.equal(p, b), "a non-finite gradient must not move a weight"


def test_rl_non_finite_gradient_with_a_finite_loss_is_skipped(monkeypatch, tmp_path: Path):
    """The same hole in the RL loop, with its own `rl_step` returning the loss."""
    seen: dict[str, list[torch.Tensor]] = {}

    def inf_grad_step(model, opt, samples, advantages, cfg, *, counters=None):
        counters = Counters() if counters is None else counters
        counters.count("rl_steps")
        counters.count("rl_samples")
        seen["before"] = [p.detach().clone() for p in model.parameters()]
        opt.zero_grad(set_to_none=True)
        loss = sum(p.float().sum() for p in model.parameters()) * 1.0
        loss.register_hook(lambda g: g * float("inf"))
        return loss, counters

    monkeypatch.setattr(_RL_TRAIN, "rl_step", inf_grad_step)
    built: list[nn.Module] = []
    real = _RL_TRAIN.DormouseModel

    def factory(cfg):
        model = real(cfg)
        built.append(model)
        return model

    monkeypatch.setattr(_RL_TRAIN, "DormouseModel", factory)
    report = rl_train(_rl_run(tmp_path, steps=1), log=lambda *_: None)
    assert report["nan_skipped"] == 1 and report["counters"]["nan_skipped"] == 1
    assert report["opt_steps"] == 0, "a non-finite gradient must not reach the optimizer"
    for p, b in zip(built[0].parameters(), seen["before"], strict=True):
        assert torch.equal(p, b), "a non-finite gradient must not move a weight"


def test_sft_healthy_three_steps_are_bit_identical_to_main(tmp_path: Path):
    """m3's second gate: the firewall is a `*1.0` / selection identity."""
    run = _sft_run(tmp_path, steps=3, log_every=3)
    torch.manual_seed(0)
    model = DormouseModel(run.model)
    lines: list[str] = []
    report = sft_train(model, run, log=lines.append)
    assert report["nan_skipped"] == 0
    assert _sha(model) == MAIN_SFT_SHA
    assert lines[-1].split("loss=")[1].split()[0] == MAIN_SFT_LOSS


def test_rl_healthy_three_steps_are_bit_identical_to_main(tmp_path: Path):
    """The same identity for the RL loop, through the saved model half."""
    run = _rl_run(tmp_path, steps=3, ckpt_every=3)
    torch.manual_seed(0)
    report = rl_train(run, log=lambda *_: None)
    assert report["nan_skipped"] == 0 and report["opt_steps"] == 3
    sd = load_model_half(Path(run.train.out_dir) / "r.pt")
    assert _sha_state(sd) == MAIN_RL_SHA


# --------------------------------------------------------------------------- m4


def _sha_in_a_fresh_process(preset: Path, seed: int) -> str:
    code = (
        "import hashlib, sys\n"
        "from dormouse.cli import _model\n"
        "from dormouse.config import load_run\n"
        "run = load_run(sys.argv[1], train_overrides={'seed': int(sys.argv[2]), 'seq_len': 64})\n"
        "m = _model(run)\n"
        "h = hashlib.sha256()\n"
        "for k, v in sorted(m.state_dict().items()):\n"
        "    h.update(k.encode())\n"
        "    h.update(v.detach().cpu().contiguous().numpy().tobytes())\n"
        "print(h.hexdigest())\n"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    out = subprocess.run(
        [sys.executable, "-c", code, str(preset), str(seed)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return out.stdout.strip()


def test_seed_decides_the_fresh_init_across_processes(tmp_path: Path):
    """Two processes, same cfg+seed → identical state_dict; seed+1 differs."""
    preset = _preset(tmp_path)
    a = _sha_in_a_fresh_process(preset, 7)
    b = _sha_in_a_fresh_process(preset, 7)
    c = _sha_in_a_fresh_process(preset, 8)
    assert a == b, "one seed must be one init, across processes"
    assert c != a, "seed+1 must be a different init"


def test_every_model_building_command_seeds_before_the_build(monkeypatch, tmp_path: Path):
    """The wiring, not just the helper: train/sft/rl all call it with the seed."""
    import dormouse.cli as cli

    calls: list[int] = []
    monkeypatch.setattr(cli, "_seed_model", lambda run: calls.append(run.train.seed))
    corpus = write_corpus(tmp_path / "corpus.bin", n_bytes=4096)
    preset = _preset(tmp_path)
    rl_preset = _preset(tmp_path / "rl", max_seq_len=128)
    runner = CliRunner()
    train = runner.invoke(
        cli.app,
        [
            "train",
            str(preset),
            "--data",
            str(corpus),
            "--set",
            "steps=1",
            "--set",
            "seq_len=64",
            "--set",
            "seed=7",
        ],
    )
    assert train.exit_code == 0, train.output
    sft = runner.invoke(
        cli.app,
        [
            "sft",
            str(preset),
            "--data",
            str(tmp_path / "sft.bin"),
            "--build-sft",
            "--set",
            "steps=1",
            "--set",
            "seq_len=64",
            "--set",
            "seed=7",
        ],
    )
    assert sft.exit_code == 0, sft.output
    rl = runner.invoke(
        cli.app,
        [
            "rl",
            str(rl_preset),
            "--set",
            "steps=1",
            "--set",
            "seq_len=128",
            "--set",
            "seed=7",
            "--set",
            "rl_tasks=2",
            "--set",
            "rl_group=2",
            "--set",
            "rl_budget=96",
            "--set",
            "rl_task_cls=math",
            "--set",
            "rl_holdout=0",
        ],
    )
    assert rl.exit_code == 0, rl.output
    assert calls == [7, 7, 7]


# --------------------------------------------------------------------------- m5


def test_rl_forwards_the_optimizer_knobs_it_accepts(monkeypatch, tmp_path: Path):
    """The same wiring gate for the RL loop (its refusals are tested above)."""
    real = _RL_TRAIN.MixOptimizer
    got: dict = {}

    def spy(*args, **kwargs):
        got.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(_RL_TRAIN, "MixOptimizer", spy)
    run = _rl_run(
        tmp_path,
        steps=0,
        rl_holdout=0,
        muon_state_dtype="bf16",
        ns_dtype="bf16",
        muon_sf_wd=0.02,
    )
    rl_train(run, log=lambda *_: None)
    assert got["muon_state_dtype"] == "bf16"
    assert got["ns_dtype"] == "bf16"
    assert got["muon_variant"] == "muon" and got["muon_sf"] is False
    assert got["muon_sf_wd"] == 0.02


def test_sft_refuses_muon_sf_by_name(tmp_path: Path):
    run = _sft_run(tmp_path, steps=1, muon_sf=True)
    with pytest.raises(LoudError, match="muon_sf"):
        sft_train(DormouseModel(run.model), run, log=lambda *_: None)


def test_sft_refuses_a_muon_variant_by_name(tmp_path: Path):
    run = _sft_run(tmp_path, steps=1, muon_variant="normuon")
    with pytest.raises(LoudError, match="muon_variant"):
        sft_train(DormouseModel(run.model), run, log=lambda *_: None)


def test_rl_refuses_muon_sf_and_grad_accum_by_name(tmp_path: Path):
    with pytest.raises(LoudError, match="muon_sf"):
        rl_train(_rl_run(tmp_path, steps=0, muon_sf=True), log=lambda *_: None)
    with pytest.raises(LoudError, match="grad_accum"):
        rl_train(_rl_run(tmp_path, steps=0, grad_accum=2), log=lambda *_: None)


def test_sft_forwards_the_optimizer_knobs_it_accepts(monkeypatch, tmp_path: Path):
    """The fields that ARE legal are wired, not just tolerated: the exact kwargs
    reach `MixOptimizer` (the round-1 hole was a silent drop, not a refusal)."""
    import dormouse.train.sft as sft_mod

    real = sft_mod.MixOptimizer
    got: dict = {}

    def spy(*args, **kwargs):
        got.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(sft_mod, "MixOptimizer", spy)
    run = _sft_run(
        tmp_path,
        steps=0,
        muon_state_dtype="bf16",
        ns_dtype="bf16",
        muon_sf_wd=0.02,
    )
    sft_train(DormouseModel(run.model), run, log=lambda *_: None)
    assert got["muon_state_dtype"] == "bf16"
    assert got["ns_dtype"] == "bf16"
    assert got["muon_variant"] == "muon" and got["muon_sf"] is False
    assert got["muon_sf_wd"] == 0.02


def test_sft_grad_accum_four_matches_a_four_times_batch_run(tmp_path: Path):
    """Honoured, not ignored: k micro-batches = one `batch*k` wide step.

    The pooled CE is a supervised-byte-weighted mean, so a plain mean of the k
    per-window CEs would NOT match; the loop backprops the summed CE and scales
    every gradient once by `1/Σ n_mask`.
    """

    def once(batch: int, accum: int) -> tuple[dict, nn.Module]:
        # `moe_capacity=0` removes the batch-composition coupling (a
        # capacity-dropping MoE's drop set is a function of the whole window),
        # so one `batch*k` wide step really is the k micro-batches' pooled mean.
        run = _sft_run(
            tmp_path,
            steps=2,
            batch=batch,
            grad_accum=accum,
            model_over={"moe_capacity": 0.0},
        )
        torch.manual_seed(0)
        model = DormouseModel(run.model)
        report = sft_train(model, run, log=lambda *_: None)
        return report, model

    one, m1 = once(BATCH, 1)
    four, m4 = once(BATCH, 4)
    wide, mw = once(BATCH * 4, 1)
    assert abs(four["loss"] - one["loss"]) > 1e-6, "grad_accum=4 must read 4x the data"
    assert four["loss"] == pytest.approx(wide["loss"], abs=1e-5)
    assert four["supervised_bytes"] == wide["supervised_bytes"]
    for p4, pw in zip(m4.parameters(), mw.parameters(), strict=True):
        assert torch.allclose(p4, pw, rtol=0, atol=1e-5), "the steps must match a wide batch"
