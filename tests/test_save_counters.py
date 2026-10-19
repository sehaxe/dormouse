"""M3: the SFT and RL saves carry the storage rung's COUNTED tally.

`save_ckpt` accepts a counters bundle for exactly one reason: `nvfp4` blocks
whose stored e4m3 scale rounds to zero decode to zeros, and the checkpoint line
must name the FILE's own count (review round 1 M3; the pretrain lane does it at
`train/loop.py`'s checkpoint block). SFT and RL called `save_ckpt` without one,
so a reader of those logs could not see the degradation at all.

Gates: a real SFT save with a deliberately tiny quantized block prints and
reports the count; a clean save adds no field; the RL save is handed a counters
bundle (its model is built inside the loop, so the seam is asserted at the call).
"""

from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path

import torch
from conftest import tiny_cfg, tiny_run

from dormouse.config import RunConfig, TrainConfig
from dormouse.model import DormouseModel
from dormouse.model.tsct import is_factor_key
from dormouse.post.rl_train import rl_train
from dormouse.train.sft import build_sft_corpus, sft_train

SEQ, BATCH = 64, 2


def _sft_run(tmp_path: Path) -> RunConfig:
    """A one-step SFT run with an `nvfp4` storage rung and its own corpus."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = tmp_path / "sft.bin"
    build_sft_corpus(data, seed=1, seq_len=SEQ)
    run = tiny_run(
        tmp_path,
        steps=1,
        batch=BATCH,
        seq_len=SEQ,
        log_every=1,
        ckpt_every=1,
        ckpt_name="s",
    )
    return dataclasses.replace(
        run,
        model=tiny_cfg(quant="nvfp4"),
        train=dataclasses.replace(run.train, data=str(data)),
    )


def _zero_one_block(model: DormouseModel) -> None:
    """One block at e4m3's top and one ~130 octaves below it, then freeze it.

    The same fixture `test_train.py` uses: the global prescale cannot save the
    small block, so `pack_tsct` must COUNT one zeroed block. Freezing stops the
    optimizer from overwriting the tiny value and hiding the defect.
    """
    params = dict(model.named_parameters())
    for key in sorted(params):
        if is_factor_key(key) and params[key].numel() >= 32:
            with torch.no_grad():
                params[key].zero_()
                flat = params[key].reshape(-1)
                flat[0] = 1.0
                flat[16] = 1e-40
            params[key].requires_grad_(False)
            return
    raise AssertionError("the tiny model has no factor matrix")


def test_sft_save_prints_the_nvfp4_zero_block_count(tmp_path: Path):
    run = _sft_run(tmp_path / "run")
    model = DormouseModel(run.model)
    _zero_one_block(model)
    lines: list[str] = []
    report = sft_train(model, run, log=lines.append)
    ckpts = [ln for ln in lines if ln.startswith("ckpt ")]
    assert ckpts, lines
    assert "nvfp4_zero_scale_blocks=1" in ckpts[0], ckpts
    assert report["counters"]["nvfp4_zero_scale_blocks"] == 1, report["counters"]


def test_a_clean_sft_save_prints_no_counter_field(tmp_path: Path):
    run = _sft_run(tmp_path / "clean")
    lines: list[str] = []
    sft_train(DormouseModel(run.model), run, log=lines.append)
    ckpts = [ln for ln in lines if ln.startswith("ckpt ")]
    assert ckpts, lines
    assert "nvfp4_zero_scale_blocks" not in ckpts[0], ckpts


def _rl_run(tmp_path: Path) -> RunConfig:
    cfg = TrainConfig(
        steps=1,
        batch=2,
        seq_len=64,
        lr=0.01,
        log_every=1,
        ckpt_every=1,
        ckpt_name="r",
        out_dir=str(tmp_path / "ck"),
        rl_task_cls="math",
        rl_tasks=2,
        rl_group=2,
        rl_budget=96,
    )
    return RunConfig(
        source="rl-tiny",
        model=tiny_cfg(max_seq_len=128),
        train=cfg,
    )


def test_rl_save_is_handed_a_counters_bundle(tmp_path: Path, monkeypatch):
    """RL builds its model inside the loop; the seam is asserted at the call.

    The spy calls the REAL `save_ckpt` (so the pair is still written) and then
    records a count on the bundle it was handed — a save that dropped the
    bundle fails the `counters is not None` assertion.
    """
    module = importlib.import_module("dormouse.post.rl_train")
    real = module.save_ckpt

    def spy(run, model, opt, step, *, counters=None):
        assert counters is not None, "the RL save dropped the counters bundle"
        weights, snapshot = real(run, model, opt, step, counters=counters)
        counters.count("nvfp4_zero_scale_blocks", 1)
        return weights, snapshot

    monkeypatch.setattr(module, "save_ckpt", spy)
    lines: list[str] = []
    report = rl_train(_rl_run(tmp_path), log=lines.append)
    ckpts = [ln for ln in lines if ln.startswith("ckpt ")]
    assert ckpts, lines
    assert "nvfp4_zero_scale_blocks=1" in ckpts[0], ckpts
    assert report["counters"]["nvfp4_zero_scale_blocks"] == 1, report["counters"]
