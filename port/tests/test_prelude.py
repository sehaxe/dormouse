"""The prelude/coda topology (`arXiv 2610.23036`, `model/prelude.py`).

`embed → prelude(0..N) → LoopBlock ×T → coda(0..M) → RMSNorm → lm_head`: a few
UNIQUE layers outside the weight-shared loop, instead of running the whole stack
`T` times. The gates, in the order the task states them:

1. `prelude_layers=0, coda_layers=0` is the `c8a2ff9` tree BIT FOR BIT — state
   dict sha256, logits sha256 and the loss, captured from a clean worktree of
   that commit before this branch existed;
2. the parameter delta is EXACTLY the new named parameters, on the tiny fixture
   and on `medium` through `dormouse info` (1 573 888 per layer at d=1024);
3. every prelude/coda parameter gets a finite, non-zero gradient in a real
   3-step trainer run, named per parameter;
4. removing either stack changes the logits after that training — both are live;
5. the loop receives the same `[b, t, d_model]` + residual-dtype contract from
   the prelude, and `max_iter` iterations still run;
6. every unsupported combination declared is a LOUD refusal with an `Escape:`;
7. no host sync in the forward (the tethered `.item()`/`.tolist()`/`__int__`/
   `__float__` patch, plus a source scan).
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest
import torch
from conftest import BATCH, SEQ, tiny_cfg, tiny_model, tiny_run
from torch import nn
from typer.testing import CliRunner

from dormouse.cli import app
from dormouse.config import ModelConfig, load_run, parse_preset
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.train import train

#: Captured from a clean `c8a2ff9` worktree (CPU, torch 2.13.0+cu129, one
#: thread, seed 1234, `conftest.tiny_model()`): the off-path goldens.
GOLDEN_STATE_SHA = "06eecd1d36bcf49ea3eb27247c32409487c3cf667fe48be1108e03cae6405b8a"
GOLDEN_LOGITS_SHA = "5de7160313c4db58b846061a299d4035ae84940f287276b380b012fdf846d9ba"
GOLDEN_LOSS = 6.200047969818115

#: `tiny_cfg()` prices: base tree and one `UniqueLayer` (3·64·32 + 64).
TINY_BASE = 64_881
TINY_LAYER = 6_208
#: The `medium` preset: base and one layer (3·1024·512 + 1024).
MEDIUM_BASE = 141_910_405
MEDIUM_LAYER = 1_573_888


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    return ids, (ids + 1) % 251


def _state_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _logits_sha(logits: torch.Tensor) -> str:
    return hashlib.sha256(logits.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def test_off_off_is_bit_identical_to_c8a2ff9():
    """Gate 1: 0/0 adds no parameter, draws no RNG, reads the old bits."""
    threads, precision = torch.get_num_threads(), torch.get_float32_matmul_precision()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    try:
        torch.manual_seed(1234)
        model = tiny_model(prelude_layers=0, coda_layers=0)
        assert isinstance(model.prelude, nn.ModuleList) and len(model.prelude) == 0
        assert isinstance(model.coda, nn.ModuleList) and len(model.coda) == 0
        assert not [k for k in model.state_dict() if k.startswith(("prelude.", "coda."))]
        assert param_count(model) == TINY_BASE
        x, y = _batch()
        with torch.no_grad():
            logits, loss, _ = model(x, y)
        assert loss is not None
        assert _state_sha(model) == GOLDEN_STATE_SHA, "the OFF state dict moved"
        assert _logits_sha(logits) == GOLDEN_LOGITS_SHA, "the OFF logits moved"
        assert float(loss.detach()) == GOLDEN_LOSS, float(loss.detach())
        print(
            f"[prelude] off/off state={GOLDEN_STATE_SHA[:16]} "
            f"logits={GOLDEN_LOGITS_SHA[:16]} loss={GOLDEN_LOSS!r}"
        )
    finally:
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision(precision)


@pytest.mark.parametrize(("pre", "post"), [(1, 0), (0, 1), (1, 1), (2, 2)])
def test_param_delta_is_exactly_the_new_named_parameters(pre, post):
    """Gate 2 (tiny): the delta is the named stack, with no hidden duplicate."""
    base = param_count(tiny_model())
    model = tiny_model(prelude_layers=pre, coda_layers=post)
    total = param_count(model)
    delta = total - base
    named_pre = sum(p.numel() for k, p in model.named_parameters() if k.startswith("prelude."))
    named_post = sum(p.numel() for k, p in model.named_parameters() if k.startswith("coda."))
    print(
        f"[prelude] tiny ({pre},{post}): {base:,} -> {total:,} (+{delta:,}) "
        f"prelude={named_pre:,} coda={named_post:,}"
    )
    assert base == TINY_BASE
    assert delta == (pre + post) * TINY_LAYER
    assert named_pre == pre * TINY_LAYER
    assert named_post == post * TINY_LAYER
    assert named_pre + named_post == delta, "a new parameter is not in the named stack"


def test_medium_counts_via_dormouse_info():
    """Gate 2 (medium): `dormouse info` agrees with the hand arithmetic."""
    runner = CliRunner()
    base = runner.invoke(app, ["info", "medium"])
    assert base.exit_code == 0, base.output
    base_n = int(base.output.split(" params")[0].split()[-1].replace(",", ""))
    print(f"[prelude] medium (0,0): {base_n:,} params via dormouse info")
    assert base_n == MEDIUM_BASE
    for pre, post in ((1, 0), (0, 1), (1, 1), (2, 2)):
        r = runner.invoke(
            app,
            [
                "info",
                "medium",
                "--set",
                f"prelude_layers={pre}",
                "--set",
                f"coda_layers={post}",
            ],
        )
        assert r.exit_code == 0, r.output
        n = int(r.output.split(" params")[0].split()[-1].replace(",", ""))
        print(f"[prelude] medium ({pre},{post}): {n:,} params (+{n - base_n:,})")
        assert n - base_n == (pre + post) * MEDIUM_LAYER


def test_every_unique_parameter_gets_a_gradient_in_a_three_step_run(tmp_path):
    """Gate 3: per parameter, finite and non-zero, after the trainer ran 3 steps."""
    torch.manual_seed(0)
    run = dataclasses.replace(
        tiny_run(tmp_path, steps=3),
        model=tiny_cfg(prelude_layers=1, coda_layers=1),
    )
    model = DormouseModel(run.model)
    train(model, run, log=lambda *_: None)
    grads: list[str] = []
    for name, p in model.named_parameters():
        if not name.startswith(("prelude.", "coda.")):
            continue
        assert p.grad is not None, f"{name}: no gradient slot after 3 steps"
        assert bool(torch.isfinite(p.grad).all()), f"{name}: non-finite gradient"
        peak = float(p.grad.abs().max())
        assert peak > 0.0, f"{name}: gradient is exactly zero"
        grads.append(f"{name}={peak:.2e}")
    print("[prelude] 3-step max|grad|: " + " ".join(grads))
    assert len(grads) == 8, grads  # 2 layers x (norm, gate, up, down)


def test_removing_either_stack_changes_the_logits_after_training(tmp_path):
    """Gate 4: both stacks are in the forward, not dead parameters."""
    torch.manual_seed(0)
    run = dataclasses.replace(
        tiny_run(tmp_path, steps=3),
        model=tiny_cfg(prelude_layers=1, coda_layers=1),
    )
    model = DormouseModel(run.model)
    train(model, run, log=lambda *_: None)
    x, _ = _batch()
    with torch.no_grad():
        before, _, _ = model(x)
    for stack in ("prelude", "coda"):
        keep = getattr(model, stack)
        assert len(keep) == 1
        setattr(model, stack, nn.ModuleList())
        try:
            with torch.no_grad():
                after, _, _ = model(x)
        finally:
            setattr(model, stack, keep)
        assert bool(torch.isfinite(before).all()) and bool(torch.isfinite(after).all())
        moved = float((before - after).abs().max())
        print(f"[prelude] removing {stack}: max|logit delta|={moved:.4e}")
        assert moved > 0.0, f"the {stack} is dead code: removing it changed nothing"


def test_the_loop_receives_the_boundary_contract_and_runs_max_iter():
    """Gate 5: shape + dtype at the prelude→loop seam, and `max_iter` iterations."""
    cfg = tiny_cfg(prelude_layers=2, coda_layers=1)
    model = DormouseModel(cfg)
    seen: dict[str, object] = {}

    def hook(_mod, args):
        h = args[0]
        seen["shape"] = tuple(h.shape)
        seen["dtype"] = h.dtype
        seen["h"] = h.detach().clone()

    handle = model.loop.register_forward_pre_hook(hook)
    x, y = _batch()
    counters = Counters()
    try:
        with torch.no_grad():
            model(x, y, counters=counters)
    finally:
        handle.remove()
    assert seen["shape"] == (BATCH, SEQ, cfg.d_model), seen["shape"]
    assert seen["dtype"] == model.embed.weight.dtype == torch.float32, seen["dtype"]
    # The loop reads the PRELUDE's output, not the embedding (two unique layers).
    with torch.no_grad():
        want = model.embed(x)
        for layer in model.prelude:
            want = layer(want)
    assert torch.equal(seen["h"], want), "the loop did not receive the prelude's output"
    assert counters["iter"] == cfg.max_iter, counters
    assert counters["depth"] == cfg.max_iter, counters
    assert model.loop._depth_ran == cfg.max_iter
    assert counters["prelude"] == 2 and counters["coda"] == 1, counters
    print(
        f"[prelude] boundary shape={seen['shape']} dtype={seen['dtype']} "
        f"iter={counters['iter']} depth={counters['depth']} "
        f"prelude={counters['prelude']} coda={counters['coda']}"
    )


def test_the_declared_unsupported_combinations_are_loud():
    """Gate 6: outside 0..=4, and `use_byteflow` — key, reason and `Escape:`."""
    for key in ("prelude_layers", "coda_layers"):
        for n in (-1, 5):
            with pytest.raises(LoudError) as e:
                parse_preset(f"{key} = {n}\n")
            msg = str(e.value)
            assert key in msg and "Escape:" in msg, msg
            print(f"[prelude] refused {key}={n}: {msg.splitlines()[0]}")
    for text in ("use_byteflow = true\nprelude_layers = 1\n",
                 "use_byteflow = true\ncoda_layers = 1\n"):
        with pytest.raises(LoudError) as e:
            parse_preset(text)
        msg = str(e.value)
        assert "use_byteflow" in msg and "Escape:" in msg, msg
        print(f"[prelude] refused {text.splitlines()[0]}: {msg.splitlines()[0]}")
    with pytest.raises(LoudError) as e:
        load_run("small", model_overrides={"use_byteflow": True, "prelude_layers": 1})
    assert "use_byteflow" in str(e.value) and "Escape:" in str(e.value)
    # The runtime seam: a hand-built model never calls `validate`.
    with pytest.raises(LoudError) as e:
        DormouseModel(ModelConfig(use_byteflow=True, prelude_layers=1))
    assert "prelude_layers=1" in str(e.value) and "Escape:" in str(e.value)


def test_the_forward_never_reads_a_device_scalar(monkeypatch):
    """Gate 7: `.item()`, `.tolist()`, `int(tensor)`, `float(tensor)` all raise."""
    import dormouse.model.prelude as prelude_mod

    src = open(prelude_mod.__file__, encoding="utf-8").read()
    assert ".item(" not in src and ".tolist(" not in src

    def boom(*_a, **_k):
        raise AssertionError("host sync in the prelude/coda forward")

    for name in ("item", "tolist", "__int__", "__float__", "__bool__"):
        monkeypatch.setattr(torch.Tensor, name, boom, raising=True)
    # The DENSE expert blend (`moe_topk=0`): the routed arm's CPU reference
    # dispatch reads `torch.unique(...).tolist()` for its loop, which is the
    # EXISTING arm and not this one (`test_mhc`'s same scoping). No targets:
    # the model's `assert torch.isfinite(loss)` is the existing training-path
    # host read, outside this arm.
    model = tiny_model(prelude_layers=1, coda_layers=1, moe_topk=0)
    x, _ = _batch()
    with torch.no_grad():
        model(x)
