"""Checkpoint completeness (2026-11-07): every buffer, counter and opt tensor.

The four defects that preceded this audit — a non-persistent `lfb_bias` that
restarted the router on resume, an `opt_graph` replay freezing the bank cache
key, a `bytes_seen` that ignored `grad_accum`, and an eval that scored one
65 KB window — share one shape: STATE THAT LOOKED SAVED WAS NOT. These gates
enumerate state instead of trusting a hand-written list:

1. every named buffer is either PERSISTENT (in `state_dict`) or classified:
   saved in the checkpoint's `moe` section, derivable from the config (proved
   by rebuilding a fresh model and comparing bit for bit), or telemetry (the
   documented reset);
2. every optimizer state tensor round-trips through `state_dict` /
   `load_state_dict`, and no live per-parameter state key is dropped;
3. the objective counters (`opt.steps`, `lfb_counts`, `lfb_bias`, the host
   Engram step) are in the file, and the ones that are not (depth sample,
   QAT anneal, cache epochs) are pure functions of `(seed, step)` or rebuilds;
4. the flagship3 shape (95 035 200 params) has NO non-persistent buffer at all,
   so its checkpoint is `state_dict()` + `opt` + `step` with nothing to lose.

A new buffer that is neither persistent nor on the classification map FAILS
here: that is the regression this file exists for.
"""

from __future__ import annotations

import copy
import dataclasses

import pytest
import torch
from conftest import tiny_cfg

from dormouse.config import RunConfig, TrainConfig, load_run
from dormouse.model import DormouseModel, param_count
from dormouse.train import load_ckpt
from dormouse.train.loop import (
    _lfb_modules,
    _sample_depth,
    load_model_state,
    save_ckpt,
)
from dormouse.train.lr import anneal_alpha
from dormouse.train.optim import MixOptimizer

TFORM = dict(
    block_form="transformer",
    n_layers=2,
    max_iter=1,
    loop_blocks=1,
    use_kda=False,
    use_csa=False,
    use_mhc=False,
    use_attnres=False,
    use_engram=False,
    layerrope=False,
    n_experts=1,
    moe_topk=0,
    use_tsct=False,
)
TFORM_BF16 = dict(TFORM, bf16=True)  # flagship3 computes in bf16, masters fp32
KDAFORM = dict(block_form="kda", max_iter=2, moe_topk=2)
LAYERROPE = dict(KDAFORM, layerrope=True, max_iter=3)  # odd depth: sign moves do not cancel
SHAREU = dict(KDAFORM, use_tsct=True, share_u_mode="dict")

#: Non-persistent buffer -> how its state is carried. A name not in
#: `state_dict()`, not here, and not rebuildable fails the audit.
CLASSIFIED = {
    "loop.moe.lfb_bias": "moe-section",  # not derivable: the integrator's OUTPUT
    "loop.engram._order_idx": "derived",  # arange(len(orders))
    "loop.moe.share_u_idx": "derived",  # seeded from cfg, no global RNG
    "layerrope._applications": "telemetry",  # per-forward count, not objective
}


def _run(tmp_path, form: dict, *, coef: float = 0.0, name: str = "t") -> RunConfig:
    cfg = TrainConfig(
        steps=2,
        batch=2,
        seq_len=64,
        lr=1e-3,
        log_every=100,
        ckpt_every=100,
        ckpt_name=name,
        out_dir=str(tmp_path / "ck"),
        data=str(tmp_path / "corpus.bin"),
    )
    model = tiny_cfg(**form)
    if form.get("block_form") == "kda":
        model = dataclasses.replace(model, moe_lb_coef=coef, moe_lb_mode="sign")
    return RunConfig(source="cmp", model=model, train=cfg)


def _train_steps(model: DormouseModel, n: int = 5) -> MixOptimizer:
    """Real optimizer steps (the LFB controller moves in trains mode).

    Five, not one: with `max_iter=2` the controller's two per-step moves can
    cancel exactly (measured on this fixture: steps 0, 1 and 3 can read all
    zeros on some forms), and a completeness fixture whose buffer never leaves
    zero proves nothing.
    """
    ids = torch.arange(2 * 64, dtype=torch.long).view(2, 64) % 251
    opt = MixOptimizer(model, lr=1e-3, wd=0.01)
    for _ in range(n):
        opt.zero_grad(set_to_none=True)
        _, loss, _ = model(ids, (ids + 1) % 251)
        assert loss is not None
        loss.backward()
        opt.step()
    return opt


def _nonpersistent(model: DormouseModel) -> dict[str, torch.Tensor]:
    sd = model.state_dict()
    return {n: b for n, b in model.named_buffers() if n not in sd}


@pytest.mark.parametrize(
    "form",
    [TFORM, TFORM_BF16, KDAFORM, LAYERROPE, SHAREU],
    ids=["transformer", "transformer+bf16", "kda", "kda+layerrope", "kda+shareu"],
)
def test_every_nonpersistent_buffer_is_carried_or_provable(tmp_path, form):
    corpus = tmp_path / "corpus.bin"
    corpus.write_bytes(bytes(range(16)) * 256)
    run = _run(tmp_path, form, coef=0.5)
    model = DormouseModel(run.model)
    opt = _train_steps(model)
    weights, snapshot = save_ckpt(run, model, opt, step=4)
    assert snapshot.is_file()
    payload = torch.load(weights, map_location="cpu", weights_only=False)

    # The file's model half is EXACTLY `state_dict()` — no key added, none
    # dropped (a strict load would catch an extra; this catches a missing one
    # before the loader does).
    assert set(payload["model"]) == set(model.state_dict())
    persistent = set(payload["model"])
    for name, _buf in model.named_buffers():
        if name in persistent:
            # Persistent buffers must be in the model half — a persistent
            # buffer missing from `state_dict` is a torch bug, and one missing
            # from the FILE is a silent reset.
            assert name in payload["model"], name
    for name, buf in _nonpersistent(model).items():
        how = CLASSIFIED.get(name)
        assert how is not None, (
            f"{name} is a non-persistent buffer with no classification: either save it "
            f"in the checkpoint (a `moe`-section sibling) or prove it is derived, and add "
            f"it to CLASSIFIED with the reason. A resume that dropped it would train a "
            f"different network under the same step count."
        )
        if how == "moe-section":
            mod = name.removesuffix(".lfb_bias")
            assert mod in payload["moe"], f"{name} is not in the checkpoint's moe section"
            assert torch.equal(payload["moe"][mod], buf), name
            # Not derivable: a fresh model starts at ZERO, so dropping it is a
            # different router — the 2026-11-04 50/51-tensor repro, re-checked.
            fresh = DormouseModel(run.model)
            live = _lfb_modules(model)[mod].lfb_bias
            assert not torch.equal(live, _lfb_modules(fresh)[mod].lfb_bias), (
                "the fixture must actually move lfb_bias"
            )
        elif how == "derived":
            # Rebuilding the same config reproduces it bit for bit (no global
            # RNG, no training history): the only legal way to be non-persistent.
            fresh = DormouseModel(run.model)
            assert torch.equal(buf, dict(fresh.named_buffers())[name]), name
        elif how == "telemetry":
            assert buf.dtype in (torch.int64, torch.int32) and buf.numel() == 1
            fresh = dict(DormouseModel(run.model).named_buffers())[name]
            assert int(fresh) == 0, "telemetry must start at zero, not carry run state"
        else:  # persistent: the first branch already asserted it
            assert name in persistent

    # And the ONE loader restores the moved bias (not just the file holds it).
    ck, stored = load_ckpt(run)
    again = DormouseModel(stored.model)
    load_model_state(again, stored.train, ck["model"])
    for name, module in _lfb_modules(model).items():
        assert torch.equal(again.state_dict()[f"{name}.lfb_counts"], module.lfb_counts)
        assert torch.equal(_lfb_modules(again)[name].lfb_bias, module.lfb_bias)


@pytest.mark.parametrize(
    "form",
    [TFORM, TFORM_BF16, KDAFORM, LAYERROPE, SHAREU],
    ids=["transformer", "transformer+bf16", "kda", "kda+layerrope", "kda+shareu"],
)
def test_every_optimizer_state_tensor_round_trips(tmp_path, form):
    corpus = tmp_path / "corpus.bin"
    corpus.write_bytes(bytes(range(16)) * 256)
    run = _run(tmp_path, form)
    model = DormouseModel(run.model)
    opt = _train_steps(model)
    # Deep-copied: a file load hands the new optimizer its OWN tensors, and an
    # in-process `state_dict` would otherwise alias the live state (which hides
    # a mutation on the next step).
    saved = copy.deepcopy(opt.state_dict())
    assert set(saved) == {"steps", "muon", "adamw"}
    assert saved["steps"] == 5

    other = MixOptimizer(DormouseModel(run.model), lr=3e-4, wd=0.5)
    other.load_state_dict(saved)
    assert other.steps == opt.steps == 5
    # The fused AdamW's bias-correction counter is a HOST int; a load that
    # restored the per-parameter `step` tensors but not this one would run the
    # next step with t=1 and silently re-warm the moments.
    assert other._adamw_t == opt._adamw_t == 5
    _assert_nested_equal(saved, other.state_dict())

    # No live per-parameter state key is dropped: the checkpoint's key set for
    # each parameter index must be the live one (the check a `state_dict` that
    # forgot a new buffer would fail).
    for half in ("muon", "adamw"):
        live = getattr(opt, half).state
        if not live:
            continue
        loaded = getattr(other, half).state
        assert len(loaded) == len(live), f"{half}: {len(loaded)} of {len(live)} state entries"
        for i, st in enumerate(live.values()):
            q = list(loaded)[i]
            assert set(st) == set(loaded[q]), f"{half}[{i}]: keys {set(st)} -> {set(loaded[q])}"
            for key, value in st.items():
                if torch.is_tensor(value):
                    assert torch.equal(value, loaded[q][key]), f"{half}[{i}].{key}"


def _assert_nested_equal(a, b) -> None:
    if isinstance(a, dict):
        assert set(a) == set(b), f"{sorted(a)} != {sorted(b)}"
        for key in a:
            _assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _assert_nested_equal(x, y)
    elif torch.is_tensor(a):
        assert torch.equal(a, b)
    else:
        assert a == b, f"{a!r} != {b!r}"


def test_the_objective_counters_are_in_the_file_or_pure(tmp_path):
    """The counters that decide a step, and where each one lives.

    `opt.steps` -> the `opt` half; `lfb_counts` -> persistent model state;
    `lfb_bias` -> the `moe` section; the host Engram `step` -> the `.ngram`
    sibling; `anneal_alpha` and the adaptive depth -> pure `(seed, step)`; the
    bank/QAT caches -> rebuilt (they key on the live parameters). A counter
    with none of those homes is the defect this test names.
    """
    corpus = tmp_path / "corpus.bin"
    corpus.write_bytes(bytes(range(16)) * 256)
    run = _run(tmp_path, KDAFORM, coef=0.5)
    model = DormouseModel(run.model)
    opt = _train_steps(model)
    weights, _ = save_ckpt(run, model, opt, step=4)
    ck = torch.load(weights, map_location="cpu", weights_only=False)

    assert ck["step"] == 4  # 0-based: five steps ran
    assert ck["opt"]["steps"] == opt.steps == 5
    assert "loop.moe.lfb_counts" in ck["model"]
    assert "loop.moe.lfb_bias" not in ck["model"]
    assert "loop.moe" in ck["moe"]

    for step in (0, 1, 5, 1000):
        assert anneal_alpha(step, 7) == anneal_alpha(step, 7)
        for seed in (0, 3):
            assert _sample_depth(seed, step, 4) == _sample_depth(seed, step, 4)
    # A cache is a rebuild, not state: nothing cache-shaped may be in the file
    # (the `opt_graph` freeze was a cache KEY, not a value, and a key cannot be
    # restored from a checkpoint that never stored the parameters it is built
    # from — the trainer invalidates it once per trained step instead).
    assert not any("bank" in key or "qat" in key for key in ck["model"])
    moe = next(m for _, m in model.named_modules() if hasattr(m, "_bank_epoch"))
    assert callable(moe.invalidate_banks)
    epoch = moe._bank_epoch
    moe.invalidate_banks()
    assert moe._bank_epoch > epoch


def test_the_flagship_transformer_has_no_nonpersistent_buffer():
    """flagship3: 95 035 200 params and every byte of state in `state_dict()`."""
    run = load_run("configs/flagship3_dense.toml")
    assert run.model.block_form == "transformer"
    model = DormouseModel(run.model, device="meta")
    assert param_count(model) == 95_035_200
    sd = model.state_dict()
    buffers = dict(model.named_buffers())
    assert buffers == {}, "the vanilla form has no buffer at all"
    params = dict(model.named_parameters())
    assert set(sd) == set(params), "state_dict must be exactly the parameters"
    assert sum(p.numel() for p in params.values()) == 95_035_200


def test_the_kda_form_nonpersistent_list_is_exactly_the_classified_ones():
    """`small` (the real kda form): no unclassified non-persistent buffer."""
    run = load_run("configs/small.toml")
    assert run.model.block_form == "kda"
    model = DormouseModel(run.model, device="meta")
    nonpersistent = sorted(_nonpersistent(model))
    # The list is data: an arm that adds a buffer must teach this test its home.
    assert nonpersistent == ["loop.engram._order_idx", "loop.moe.lfb_bias"]
    for name in nonpersistent:
        assert name in CLASSIFIED, name
    # The tables ARE a parameter here (in-VRAM substrate): nothing hidden.
    assert "loop.engram.tables" in model.state_dict()
    assert model.state_dict()["loop.moe.lfb_counts"].dtype is torch.float32
    assert model.state_dict()["loop.moe.lfb_counts"].shape == (model.cfg.n_experts,)
