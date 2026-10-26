"""The HALT HEAD: target rule, gradient flow, refusals, and the inference stop.

The mechanism is `model/halt.py`; the wiring is `DormouseModel` (build + BCE
term) and `LoopBlock._window` (the inference read). The gates below are the ones
the task names: inert at weight 0, all four parameters trained, the target taken
from the model's own per-iteration CE and never from the labels, a real early
exit, the loud refusals, and no host read added to a training forward.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
from conftest import BATCH, SEQ, tiny_cfg, tiny_model, tiny_run

from dormouse.config import ModelConfig, load_preset, validate
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.model.halt import HaltHead, halt_targets
from dormouse.train import open_streams, train
from dormouse.train.loop import evaluate


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    return ids, (ids + 1) % 251


class _AlwaysStop(torch.nn.Module):
    """A head that says stop at every iteration (a big positive logit)."""

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.full((), 10.0, dtype=h.dtype)


class _ReadLog(torch.overrides.TorchFunctionMode):
    """Record every device→host read the forward makes (ADR-0018 rule 2)."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func in (
            torch.Tensor.item,
            torch.Tensor.__float__,
            torch.Tensor.tolist,
            torch.Tensor.__int__,
        ):
            self.calls.append(str(func))
        return func(*args, **(kwargs or {}))


class _NoHostRead(_ReadLog):
    """Fail on any device→host read the forward makes."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func in (
            torch.Tensor.item,
            torch.Tensor.__float__,
            torch.Tensor.tolist,
            torch.Tensor.__int__,
        ):
            raise AssertionError(f"host read on the hot path: {func}")
        return func(*args, **(kwargs or {}))


# --- (a) weight 0 is inert: same tree, same logits, same loss ---------------


def test_weight_zero_is_bit_identical_and_adds_no_parameter():
    """`halt_weight=0` must not change the model at all — tree or numbers.

    The knob fields are set to DIFFERENT inert values on the second model (a
    weight of 0 gates them off), so the check is that the new fields reach no
    code at weight 0. The preset totals in `test_param_count.py` are the other
    half of this gate: they pin today's parameter count for `small`/`b1`.
    """
    torch.manual_seed(0)
    base = tiny_model()
    torch.manual_seed(0)
    other = tiny_model(halt_weight=0.0, halt_eps=0.9, halt_min_iter=1, halt_max_iter=2)
    assert base.halt is None and other.halt is None
    assert not any("halt" in key for key in base.state_dict())
    assert param_count(base) == param_count(other)
    sd_a, sd_b = base.state_dict(), other.state_dict()
    assert sd_a.keys() == sd_b.keys()
    assert all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)

    x, y = _batch()
    logits_a, loss_a, _ = base(x, y)
    logits_b, loss_b, _ = other(x, y)
    assert torch.equal(logits_a, logits_b)
    assert torch.equal(loss_a, loss_b)


# --- (b) every halt parameter is trained ------------------------------------


def test_every_halt_parameter_gets_a_finite_nonzero_gradient():
    model = tiny_model(halt_weight=0.1, max_iter=2)
    assert model.halt is not None
    x, y = _batch()
    _, loss, _ = model(x, y)
    assert loss is not None
    loss.backward()
    params = dict(model.halt.named_parameters())
    assert set(params) == {"down.weight", "down.bias", "up.weight", "up.bias"}
    for name, p in params.items():
        assert p.grad is not None, f"{name}: no gradient"
        assert torch.isfinite(p.grad).all(), f"{name}: non-finite gradient"
        assert p.grad.abs().sum() > 0, f"{name}: zero gradient"


# --- (c) the target is the model's own CE; the input never sees labels ------


def test_halt_target_rule_from_the_per_iteration_ce():
    """The target rule, as arithmetic — including the rand-depth truncation."""
    ce = [torch.tensor(v) for v in (3.0, 2.99, 2.98)]
    # 3.0 -> 2.99 is 0.01 >= eps: thinking helps, stop=0; 2.99 -> 2.98 is 0.01:
    # stop=0; the last of the cap is a certain stop.
    assert [float(t) for t in halt_targets(ce, 0.005, 3)] == [0.0, 0.0, 1.0]  # type: ignore[arg-type]
    # A tiny improvement is a stop.
    flat = [torch.tensor(v) for v in (3.0, 2.999, 2.999)]
    assert [float(t) for t in halt_targets(flat, 0.005, 3)] == [1.0, 1.0, 1.0]  # type: ignore[arg-type]
    # Deeper is WORSE: a negative improvement is also a stop.
    worse = [torch.tensor(v) for v in (2.0, 2.5, 2.6)]
    assert [float(t) for t in halt_targets(worse, 0.005, 3)] == [1.0, 1.0, 1.0]  # type: ignore[arg-type]
    # Truncated by rand_depth (T=2 < cap=3): the last run iteration has no
    # CE_{i+1}, so it carries no target rather than a false "stop".
    trunc = halt_targets([torch.tensor(3.0), torch.tensor(2.9)], 0.005, 3)
    assert float(trunc[0]) == 0.0  # type: ignore[arg-type]
    assert trunc[1] is None


def test_halt_input_does_not_see_targets():
    """The DSpark D1 lesson: the head's INPUT is a function of `input_ids` only.

    Two different label sequences on the same bytes must leave every per-iteration
    state identical — while the CEs (the TARGETS) move, since they are scored
    against the labels. The head therefore cannot be handed the next byte.
    """
    model = tiny_model(halt_weight=0.1, max_iter=2)
    assert model.halt is not None
    x, y1 = _batch()
    y2 = (y1 + 7) % 251
    model(x, y1)
    h1 = [t.detach().clone() for t in model.loop._per_iter_hidden]
    ce1 = [t.detach().clone() for t in model.loop._per_iter_ce]
    model(x, y2)
    h2 = [t.detach().clone() for t in model.loop._per_iter_hidden]
    ce2 = [t.detach().clone() for t in model.loop._per_iter_ce]
    assert len(h1) == len(h2) == 2
    assert all(torch.equal(a, b) for a, b in zip(h1, h2, strict=True))
    assert any(not torch.equal(a, b) for a, b in zip(ce1, ce2, strict=True))
    # ... and the target is a function of those CEs and nothing else.
    assert len(halt_targets(ce1, 0.005, 2)) == len(halt_targets(ce2, 0.005, 2)) == 2


# --- (d) inference really stops (and counts the skip) -----------------------


def test_a_stub_stop_makes_the_forward_run_exactly_one_iteration():
    """p > 0.5 at iteration 1 -> one iteration, and `halt_skipped` says so."""
    model = tiny_model(halt_weight=0.1, halt_mode=True, max_iter=3, halt_max_iter=3)
    model.halt = _AlwaysStop()
    model.eval()
    x, y = _batch()
    with torch.no_grad():
        logits, loss, counters = model(x, y)
    assert counters["iter"] == 1, counters
    assert counters["halt_stop"] == 1
    assert counters["halt_skipped"] == 2  # the cap (3) minus the one that ran
    assert model.loop._depth_ran == 1

    # The stopped forward is depth 1's forward — the same math, one iteration.
    ref = tiny_model(halt_weight=0.0, max_iter=3)
    ref.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("halt.")})
    with torch.no_grad():
        ref_logits, _, _ = ref(x, depth=1)
    assert torch.equal(logits, ref_logits)
    assert loss is not None and torch.isfinite(loss)


def test_no_stop_runs_the_whole_halt_cap_and_counts_no_skip():
    """A head that says continue runs `halt_max_iter`, not `max_iter`."""
    model = tiny_model(halt_weight=0.1, halt_mode=True, max_iter=4, halt_max_iter=2)
    model.halt = HaltHead(model.cfg.d_model)
    model.halt.up.bias.data.fill_(-100.0)  # sigmoid(-100) ~ 0: never stop
    model.eval()
    x, _ = _batch()
    with torch.no_grad():
        _, _, counters = model(x)
    assert counters["iter"] == 2, counters
    assert counters.get("halt_skipped", 0) == 0
    assert model.loop._depth_ran == 2


# --- (e) the loud refusals --------------------------------------------------


def test_halt_refusals_name_the_escape():
    cfg = tiny_cfg(max_iter=4)
    with pytest.raises(LoudError) as e:
        validate(dataclasses.replace(cfg, halt_mode=True, halt_weight=0.1, halt_max_iter=5))
    assert "halt_max_iter=5" in str(e.value) and "max_iter=4" in str(e.value)

    with pytest.raises(LoudError) as e:
        validate(dataclasses.replace(cfg, halt_mode=True))
    assert "no halt head" in str(e.value)

    with pytest.raises(LoudError) as e:
        validate(dataclasses.replace(cfg, halt_eps=-1.0))
    assert "halt_eps" in str(e.value)

    with pytest.raises(LoudError) as e:
        validate(
            dataclasses.replace(
                cfg, halt_mode=True, halt_weight=0.1, halt_min_iter=9, halt_max_iter=4
            )
        )
    assert "halt_min_iter" in str(e.value)

    # The runtime seam is the same refusal for a hand-built model (constructors
    # do not call `validate`).
    with pytest.raises(LoudError) as e:
        DormouseModel(dataclasses.replace(cfg, halt_mode=True))
    assert "no halt head" in str(e.value)
    with pytest.raises(LoudError) as e:
        DormouseModel(dataclasses.replace(cfg, halt_mode=True, halt_weight=0.1, halt_max_iter=5))
    assert "halt_max_iter" in str(e.value)

    # The defaults are a LEGAL pair even though 8 > 4: the cap only binds when
    # the mode that reads it is on (every snapshot loads the default pair).
    validate(ModelConfig())


# --- (f) no host read added to a training forward ---------------------------


def test_training_forward_adds_no_host_read():
    """The halt TERM must add no device→host read to a training forward.

    The forward already makes some (the tiny MoE's naive dispatch calls
    `tolist`), so the gate is a DIFF: the same forward at `halt_weight=0` and at
    `halt_weight=0.1` must issue the same reads. The second half proves the mode
    catches the decision read in the INFERENCE branch, so the diff is not
    vacuous: a stop head that never reads would make the skip a fake.
    """
    x, y = _batch()
    torch.manual_seed(0)
    inert = tiny_model(halt_weight=0.0, max_iter=3)
    torch.manual_seed(0)
    armed = tiny_model(halt_weight=0.1, halt_mode=True, max_iter=3, halt_max_iter=3)
    inert.train()
    armed.train()
    log_inert, log_armed = _ReadLog(), _ReadLog()
    with log_inert:
        inert(x, y)
    with log_armed:
        armed(x, y)
    assert log_armed.calls == log_inert.calls, (
        f"the halt term added host reads: {set(log_armed.calls) - set(log_inert.calls)}"
    )

    armed.halt = _AlwaysStop()
    armed.eval()
    with _NoHostRead(), pytest.raises(AssertionError, match="host read"):
        armed(x)


# --- the depth-8 path -------------------------------------------------------


def test_max_iter_eight_builds_trains_and_curves(tmp_path):
    """`max_iter=8` — the depth-8 arm's model — builds, trains, and evaluates.

    Nothing in the loop caps depth at 4: the slot-indexed parameters
    (`iter_embed`, the LOOM/MSA/AttnRes rows) are sized from `max_iter`, and the
    eval's depth curve covers 1..=8 one pass each.
    """
    torch.manual_seed(0)
    model = tiny_model(max_iter=8)
    assert model.loop.iter_embed.shape[0] == 8
    assert param_count(model) > param_count(tiny_model(max_iter=4))

    run = tiny_run(tmp_path, steps=4, log_every=2, eval=True)
    run = dataclasses.replace(run, model=tiny_cfg(max_iter=8))
    report = train(DormouseModel(run.model), run, log=lambda *_: None)
    assert report["nan_skipped"] == 0
    assert report["counters"]["depth"] == 4 * 8

    _train, ev = open_streams(run)
    assert ev is not None
    curve = evaluate(DormouseModel(run.model), ev, 2, depths=True)
    assert sorted(curve) == list(range(1, 9))
    assert all(math.isfinite(curve[d][2]) for d in curve)


def test_halt_weight_on_logs_the_halt_field(tmp_path):
    """`halt=<value>` on the counter line — the arm must show it ran."""
    run = tiny_run(tmp_path, steps=2, log_every=1)
    run = dataclasses.replace(run, model=tiny_cfg(halt_weight=0.1))
    lines: list[str] = []
    train(DormouseModel(run.model), run, log=lines.append)
    hit = [ln for ln in lines if "step=" in ln]
    assert hit and all(" halt=" in ln for ln in hit), lines
    assert load_preset("small").halt_weight == 0.0
