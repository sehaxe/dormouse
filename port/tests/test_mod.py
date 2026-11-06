"""Mixture-of-Depths over `block_form="transformer"` — the six gates, CPU only.

The arm is `model/transformer.py`'s per-block router (arXiv:2404.02258v1 §3.4):
one linear score per token, top-`C = round(capacity·T)` tokens through the block,
the rest around it, `r_i·f_i + x_i` on the selected tokens. This file gates the
claims the implementation makes, each able to fail on its own:

a. `mod_capacity=1.0, mod_top_k=0` is the PLAIN transformer, bit for bit —
   state, logits, loss and every gradient. Pins captured on clean `main`
   `7e921b3` (this worktree's HEAD) with
   `/tmp/opencode/pin_mod.py` BEFORE the edit: same seed (1234), one thread,
   `highest` matmul precision, `conftest`'s tiny transformer fixture.
b. the ledger reproduces the measured flagship bill (137.42 MFLOP/byte at
   capacity 1.0) and its capacity-0.5 ratio is computed from the ledger's own
   fixed/routed split, not typed in.
c. the routed forward and backward take NO host read: `item`/`tolist`/
   `__bool__`/`__float__`/`__int__`, `torch.nonzero` and `torch.unique` all
   raise if called (the CPU stand-in for the CUDA capture blocker).
d. the selection count is exactly `round(capacity·T)` per sequence for 20
   random batches, and the compaction is position-ordered with a permutation of
   the static slots (no off-by-one from `topk` ties).
e. the router receives a non-zero gradient on every step, and NO router exists
   when the aux weight is 0 and the capacity is 1.0 (routing off, exactly zero).
f. a 20-step CPU run at capacity 0.5 is finite and its ledger FLOPs are
   strictly below capacity 1.0 at the same batch.

`tests/test_mod_capture.py` holds the CUDA-graph gate (skipped without a card).
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch
from conftest import BATCH, SEQ, tiny_cfg, tiny_run

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dormouse.config import ModelConfig, load_preset  # noqa: E402
from dormouse.instrument import LoudError  # noqa: E402
from dormouse.model import DormouseModel  # noqa: E402
from dormouse.model.transformer import mod_capacity_tokens, mod_route  # noqa: E402
from dormouse.train import train  # noqa: E402
from dormouse.train.loop import mod_field  # noqa: E402

#: Captured on clean `main` `7e921b3` (this worktree's HEAD, BEFORE the edit),
#: `/tmp/opencode/pin_mod.py`, `conftest.TINY` + `block_form="transformer"`,
#: `max_iter=1`, `n_layers=2`, seed 1234, one thread, `highest` precision.
PIN_STATE_SHA = "a3768dde7ba0a7b7274402e3593a2668b88f3c920590045848e65d7cc91ddf3e"
PIN_LOGITS_SHA = "c33ac2373dfbfac10448862f7761fc50e2aa3552816c231f9c9709d884ed1713"
PIN_LOSS = 5.945834159851074
PIN_GRAD_SHA = "ba6158e88169284ad463383af7dcd5c6adb268377d5b7c88afbd4bf50c763de1"


def _load_ledger():
    """`scripts/mod_ledger.py` by path — `scripts/` is not a package."""
    name = "mod_ledger_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "mod_ledger.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().to(torch.float32).contiguous().numpy().tobytes()).hexdigest()


def _state_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().to(torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def _grad_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, p in sorted(model.named_parameters()):
        h.update(k.encode())
        if p.grad is None:
            h.update(b"NONE")
        else:
            h.update(p.grad.detach().to(torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def _transformer_cfg(**over) -> ModelConfig:
    """The tiny fixture as a legal vanilla transformer (same as test_block_form)."""
    base = dict(
        block_form="transformer",
        max_iter=1,
        loop_blocks=1,
        n_layers=2,
        use_kda=False,
        use_engram=False,
        use_tsct=False,
        moe_topk=0,
        n_experts=1,
        bf16=False,
    )
    base.update(over)
    return tiny_cfg(**base)


def _batch(seq: int = SEQ) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.arange(BATCH * seq, dtype=torch.long).view(BATCH, seq) % 251
    return ids, (ids + 1) % 251


def _routers(model: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    return [(n, p) for n, p in model.named_parameters() if n.endswith(".router")]


# --- (a) capacity 1.0 is the plain transformer, bit for bit ----------------


def test_capacity_one_is_bit_identical_to_the_plain_transformer():
    """Gate a: logits AND every gradient match clean `main` `7e921b3`."""
    threads, precision = torch.get_num_threads(), torch.get_float32_matmul_precision()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    try:
        torch.manual_seed(1234)
        model = DormouseModel(_transformer_cfg(mod_capacity=1.0, mod_top_k=0))
        assert _routers(model) == [], "capacity 1.0 built a router: it must be off"
        x, y = _batch()
        logits, loss, _ = model(x, y)
        assert loss is not None
        loss.backward()
        assert _state_sha(model) == PIN_STATE_SHA, "the state dict moved"
        assert _sha(logits) == PIN_LOGITS_SHA, "the logits moved"
        assert float(loss.detach()) == PIN_LOSS
        assert _grad_sha(model) == PIN_GRAD_SHA, "the gradients moved"
        assert model.last_mod_aux is None
    finally:
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision(precision)


# --- (b) the ledger algebra ------------------------------------------------


def test_ledger_reproduces_the_flagship_anchor_and_computes_the_half_ratio():
    """Gate b: c=1.0 is exactly 1.0x; c=0.5 is the split's own number, ~0.62."""
    led = _load_ledger()
    cfg = load_preset("flagship_base")
    one = led.ledger(cfg, 1.0, seq_len=1024, name="flagship")
    half = led.ledger(cfg, 0.5, seq_len=1024, name="flagship")
    assert one.total == pytest.approx(137.42, abs=0.02)
    assert one.ratio == pytest.approx(1.0, abs=0.0)
    # THE EXPECTED NUMBER IS THE FORMULA'S, not a typed 0.62: the fixed share is
    # attention core + head, the routed share is the rest.
    expected = (half.fixed + half.capacity * half.routed) / (half.fixed + half.routed)
    assert half.ratio == pytest.approx(expected, abs=1e-12)
    assert abs(expected - 0.62) <= 0.02, (
        f"capacity 0.5 gives {expected:.3f}x, not the report's ~0.62x — the "
        f"fixed/routed split moved"
    )


def test_the_transformer_ledger_prices_the_arm_and_its_bound():
    """The MoD arm's own (analytic) ledger: 0.5 capacity halves the block bill."""
    led = _load_ledger()
    cfg = load_preset("mirror_control")
    one = led.ledger(cfg, 1.0, seq_len=1024)
    half = led.ledger(cfg, 0.5, seq_len=1024)
    assert one.params == 13_478_448
    assert one.router_params == 0 and half.router_params == 12 * cfg.d_model
    assert one.total > 0
    assert half.total < one.total
    # The headline does not credit the shrunk attention core; the counterfactual
    # column does, so the true saving is between the two.
    assert half.ratio_all_routed <= half.ratio <= 1.0


# --- (c) no host sync in the routed path ----------------------------------


def test_the_routed_forward_and_backward_take_no_host_read(monkeypatch):
    """Gate c: every host-read shape the CUDA capture refuses raises here."""
    seen: list[str] = []

    def boom(name):
        def _f(*_a, **_k):
            seen.append(name)
            raise AssertionError(f"host read in the routed path: {name}")

        return _f

    for name in ("item", "tolist", "__bool__", "__float__", "__int__"):
        monkeypatch.setattr(torch.Tensor, name, boom(name), raising=True)
    for name in ("nonzero", "unique"):
        monkeypatch.setattr(torch, name, boom(name), raising=True)
    cfg = _transformer_cfg(mod_capacity=0.5, mod_aux_weight=0.01)
    model = DormouseModel(cfg)
    x, y = _batch()
    logits, loss, _ = model(x, y)
    assert loss is not None
    loss.backward()
    assert seen == [], seen
    assert logits.shape == (BATCH * SEQ, cfg.vocab)


# --- (d) the exact selection count ----------------------------------------


def test_selection_is_exactly_round_capacity_times_t_for_twenty_batches():
    """Gate d: 20 random batches x three capacities; the count is exact."""
    torch.manual_seed(7)
    for _ in range(20):
        scores = torch.randn(BATCH, SEQ)
        for capacity in (0.5, 0.25, 0.75):
            c = round(capacity * SEQ)
            idx, sel, slots = mod_route(scores, capacity, 0)
            assert idx.shape == (BATCH, c), (idx.shape, c)
            assert sel.sum(1).tolist() == [c] * BATCH
            # The compaction is a permutation of the static slots...
            assert sorted(slots[0].tolist()) == list(range(c))
            # ...and its positions are ascending (causal order preserved).
            pos = torch.arange(SEQ).expand(BATCH, SEQ).gather(1, idx)
            pbuf = torch.zeros((BATCH, c), dtype=torch.long).scatter_(1, slots, pos)
            assert torch.equal(pbuf, pbuf.sort(dim=1).values)
    # An explicit top_k overrides the fraction and is clamped to the sequence.
    assert mod_capacity_tokens(0.5, 0, 64) == 32
    idx, sel, _ = mod_route(torch.randn(BATCH, SEQ), 1.0, 7)
    assert idx.shape == (BATCH, 7) and sel.sum(1).tolist() == [7] * BATCH
    # Full capacity is every token (the "1.0 = every token selected" contract).
    assert mod_capacity_tokens(1.0, 0, SEQ) == SEQ
    with pytest.raises(LoudError) as e:
        mod_capacity_tokens(1.0, SEQ + 1, SEQ)
    assert "mod_top_k" in str(e.value) and "Escape:" in str(e.value)


# --- (e) router gradient, and the exact zero when the arm is off ----------


def test_router_gradient_flows_on_every_step_and_is_absent_when_off():
    """Gate e: 20 steps, non-zero router grad each; capacity 1.0 has no router."""
    torch.manual_seed(11)
    model = DormouseModel(_transformer_cfg(mod_capacity=0.5, mod_aux_weight=0.01))
    routers = _routers(model)
    assert routers, "routing on built no router"
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(20):
        ids = (torch.randint(0, 251, (BATCH, SEQ)) + step) % 256
        _logits, loss, _ = model(ids, (ids + 1) % 256)
        assert loss is not None and math.isfinite(float(loss.detach()))
        opt.zero_grad()
        loss.backward()
        for name, p in routers:
            assert p.grad is not None, f"{name} got no gradient at step {step}"
            assert float(p.grad.norm()) > 0.0, f"{name} has a zero grad at step {step}"
        opt.step()
    assert model.last_mod_aux is not None
    assert mod_field(model).startswith("mod_aux=")

    off = DormouseModel(_transformer_cfg(mod_capacity=1.0, mod_top_k=0, mod_aux_weight=0.0))
    assert _routers(off) == [], "the off arm must build no router (zero, not tiny)"
    x, y = _batch()
    _logits, _loss, _ = off(x, y)
    assert off.last_mod_aux is None and mod_field(off) == ""


# --- (f) a 20-step CPU run at capacity 0.5 ---------------------------------


def test_twenty_cpu_steps_at_half_capacity_are_finite_and_cheaper(tmp_path):
    """Gate f: finite loss and strictly fewer ledger FLOPs than capacity 1.0."""
    cfg = _transformer_cfg(mod_capacity=0.5, mod_aux_weight=0.01)
    run = dataclasses.replace(tiny_run(tmp_path, steps=20, log_every=10), model=cfg)
    report = train(DormouseModel(cfg), run, log=lambda *_: None)
    assert math.isfinite(report["loss"]), report["loss"]
    assert report["opt_steps"] == 20

    led = _load_ledger()
    cheap = led.ledger(cfg, 0.5, seq_len=SEQ)
    full = led.ledger(
        dataclasses.replace(cfg, mod_capacity=1.0, mod_aux_weight=0.0), 1.0, seq_len=SEQ
    )
    assert cheap.total < full.total, (cheap.total, full.total)


def test_the_log_line_carries_the_aux_value(tmp_path):
    """The `mod_aux=` field is printed on the log cadence (and absent when off)."""
    lines: list[str] = []
    cfg = _transformer_cfg(mod_capacity=0.5, mod_aux_weight=0.01)
    run = dataclasses.replace(tiny_run(tmp_path, steps=10, log_every=5), model=cfg)
    report = train(DormouseModel(cfg), run, log=lines.append)
    assert math.isfinite(report["loss"])
    hits = [line for line in lines if "mod_aux=" in line]
    assert hits, "no log line carried mod_aux="
    value = float(hits[-1].split("mod_aux=")[1].split()[0])
    assert math.isfinite(value) and value > 0.0

    control: list[str] = []
    off = _transformer_cfg(mod_capacity=1.0)
    (tmp_path / "off").mkdir(exist_ok=True)
    run = dataclasses.replace(tiny_run(tmp_path / "off", steps=5, log_every=5), model=off)
    train(DormouseModel(off), run, log=control.append)
    assert not any("mod_aux=" in line for line in control), "the off arm printed an aux"


# --- refusals --------------------------------------------------------------


@pytest.mark.parametrize(
    ("over", "named"),
    [
        ({"mod_capacity": 0.0}, "mod_capacity"),
        ({"mod_capacity": 1.5}, "mod_capacity"),
        ({"mod_top_k": -1}, "mod_top_k"),
        ({"mod_top_k": 1000}, "mod_top_k"),
        ({"mod_aux_weight": -0.1, "mod_capacity": 0.5}, "mod_aux_weight"),
        ({"mod_aux_weight": 0.01}, "mod_aux_weight"),
    ],
)
def test_mod_refuses_what_it_cannot_honour(over, named):
    """A knob that reaches no code is a config that lies: LOUD, with `Escape:`."""
    with pytest.raises(LoudError) as e:
        _transformer_cfg(**over)
    assert named in str(e.value)
    assert "Escape:" in str(e.value)


def test_kda_refuses_the_mod_knobs():
    """MoD is a transformer-form mechanism; the loop has no per-block router."""
    with pytest.raises(LoudError) as e:
        tiny_cfg(mod_capacity=0.5)
    assert "Mixture-of-Depths" in str(e.value)
    assert "Escape:" in str(e.value)


def test_the_refusals_hold_at_the_model_seam_too():
    """A hand-mutated config skips `validate` but not the model's own seam."""
    cfg = _transformer_cfg(mod_capacity=0.5)
    cfg.mod_capacity = 1.5
    with pytest.raises(LoudError) as e:
        DormouseModel(cfg)
    assert "mod_capacity" in str(e.value)
    cfg.mod_capacity = 1.0
    cfg.mod_aux_weight = 0.01
    with pytest.raises(LoudError) as e:
        DormouseModel(cfg)
    assert "mod_aux_weight" in str(e.value)

