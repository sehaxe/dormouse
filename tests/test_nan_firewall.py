"""The DEVICE NaN firewall: the step decides finiteness without a host read.

Measured problem (audit MAJOR 4): the trainer read the loss scalar to the host
on EVERY step — `float(loss.detach())` — and the firewall's decision and the
log line shared that one D2H transfer. That read is what blocks capturing a
whole training step in a CUDA graph, and it costs a sync per step on the
launch-bound box (AGENTS §3.1: the card is idle ~87% of a warm step).

The mechanism the gates below pin (`train/loop.py`, `train/optim.py`):

1. `torch.isfinite(loss.detach()).all()` stays on the device and becomes a
   0-dim 1/0 integer (`flag_dev`), filled in place (a CUDA graph captured later
   reads that storage on every replay, like the LR tensor `graph.py` documents).
2. The loss is `torch.where`-masked to 0 before backward, so the seed gradient
   is exactly zero; gradients that went non-finite through an ACTIVATION are
   `nan_to_num_`'d and multiplied by the flag.
3. The optimizer's WRITES are multiplied by the flag: Muon's update, the
   AdamW first moment, and the decoupled weight-decay factor (lifted out of
   `torch.optim` for this — a zero gradient alone still moves a parameter
   through `p.mul_(1 - lr*wd)`). A non-finite step is a no-op UPDATE, decided
   entirely on the device: no `if bool(tensor)` anywhere.
4. The skip is COUNTED as a DEVICE counter (`nan_skipped`) and drained with the
   window's other counters; the last finite loss is a device tensor, read only
   at `log_every`.

The golden in gate 1 was captured from a CLEAN HEAD worktree before this
change (`642099e`, 5 steps of `tiny_run`, seed 0), so a healthy run must be
numerically identical, not merely close.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
from pathlib import Path

import pytest
import torch
from conftest import tiny_run

from dormouse.instrument import Counters
from dormouse.model import DormouseModel
from dormouse.train.loop import train

#: The HEAD golden: sha256 over `model.state_dict()` (keys sorted, raw bytes)
#: after 5 steps of `tiny_run(steps=5, log_every=5, warmup_tokens=0)`, seed 0,
#: captured at 642099e in a clean worktree (the change must not move a bit).
HEAD_SHA = "526b6c9eed17225ef6f72a5784b5046d9bc076d11548ef131a415e85afcaad1a"
#: The same run's log line at step 5 (eager, HEAD): the loss field verbatim.
HEAD_LOSS_FIELD = "loss=2.7103"


def _sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _model(run) -> DormouseModel:
    torch.manual_seed(0)
    return DormouseModel(run.model)


class _NonFiniteLoss(torch.nn.Module):
    """The firewall's own fixture: a finite forward whose loss goes NaN.

    `poison_on` is 1-based step indices (empty = every step), so a test can put
    the NaN on the step whose LOG line must fall back to the last finite loss.
    """

    def __init__(self, inner: torch.nn.Module, poison=float("nan"), poison_on=()) -> None:
        super().__init__()
        self.inner = inner
        self.poison = poison
        self.poison_on = set(poison_on)
        self.seen = 0

    def forward(self, x, targets=None, counters=None):
        logits, loss, counters = self.inner(x, targets, counters)
        self.seen += 1
        if not self.poison_on or self.seen in self.poison_on:
            return logits, loss * self.poison, counters
        return logits, loss, counters


# --- gate 1: numerically identical training ---------------------------------


@pytest.mark.local_pin
def test_a_healthy_five_step_run_is_bit_identical_to_head(tmp_path: Path):
    """No new arithmetic may move a healthy number (the whole change is gated)."""
    run = tiny_run(tmp_path, steps=5, log_every=5, warmup_tokens=0)
    model = _model(run)
    lines: list[str] = []
    report = train(model, run, log=lines.append)
    assert _sha(model) == HEAD_SHA, (
        "the device firewall changed a healthy step's bits; the flag must be a "
        "multiply-by-1.0 / selection identity on a finite step"
    )
    assert report["nan_skipped"] == 0, "a healthy run must count zero skips"


# --- gate 2: a non-finite loss is skipped, bit-for-bit -----------------------


def test_a_non_finite_loss_is_a_device_skip(tmp_path: Path):
    run = tiny_run(tmp_path, steps=1, log_every=1)
    model = _NonFiniteLoss(_model(run))
    before = [p.detach().clone() for p in model.parameters()]
    lines: list[str] = []
    report = train(model, run, log=lines.append)
    assert report["nan_skipped"] == 1
    assert report["counters"]["nan_skipped"] == 1, "the skip must be COUNTED on the log line"
    for p, b in zip(model.parameters(), before, strict=True):
        assert torch.equal(p, b), "a non-finite loss must not move a single weight"
    # The logged loss is the last FINITE one (NaN here: the run never had one),
    # and it came from the device tensor, not a per-step read.
    assert "loss=nan" in lines[0], lines[0]


# --- gate 3: a healthy step still trains -------------------------------------


def test_a_healthy_step_still_moves_the_weights(tmp_path: Path):
    run = tiny_run(tmp_path, steps=1, log_every=1)
    model = _model(run)
    before = [p.detach().clone() for p in model.parameters()]
    report = train(model, run, log=lambda *_: None)
    assert report["nan_skipped"] == 0, "the firewall must not fire on a healthy step"
    moved = sum(not torch.equal(p, b) for p, b in zip(model.parameters(), before, strict=True))
    assert moved > 40, f"a healthy step trained almost nothing: {moved} tensors moved"


# --- gate 4: no host read in the step ---------------------------------------


def test_the_training_step_takes_no_host_read(monkeypatch, tmp_path: Path):
    """Patch the six host reads; three forward+backward+step rounds must pass.

    The ONLY exemptions, both named rather than silent:

    * `torch.optim`'s own `.item()` on its CPU STEP COUNTER (`adam._get_value`).
      That tensor is a CPU bookkeeping scalar, not a device value — the read it
      must not make is the loss/flag one, and this gate still fails on that.
    * `Counters.drain`, the DRAIN cadence itself (`log_every` here is huge, so
      the only drain is the report): the host may read device counters there,
      which is the rule this change preserves.

    `moe_topk = 0` is the `small` preset's dense blend; the SPARSE naive
    dispatch has its own pre-existing `unique(...).tolist()` (a host loop over
    active experts, out of this task's scope — follow-up).
    """
    names = ("item", "tolist", "__bool__", "__float__", "__int__", "cpu")
    real = {n: getattr(torch.Tensor, n) for n in names}

    def boom(name):
        def check(self, *a, **k):
            caller = inspect.currentframe().f_back
            module = caller.f_globals.get("__name__", "")
            if name == "item" and module.startswith("torch.optim"):
                return real[name](self, *a, **k)
            raise AssertionError(
                f"host read `{name}` during the step, called from {module} "
                f"(shape {tuple(self.shape)})"
            )

        return check

    def install() -> None:
        for n in names:
            setattr(torch.Tensor, n, boom(n))

    def restore() -> None:
        for n in names:
            setattr(torch.Tensor, n, real[n])

    real_drain = Counters.drain

    def drain(self):
        restore()  # the declared cadence read, and nothing else
        try:
            return real_drain(self)
        finally:
            install()

    # The loop's ONE declared scalar read (`loop._host_scalar`, used by the log
    # line and the report): allowed, and the only read that is.
    import dormouse.train.loop as loop_mod

    real_read = loop_mod._host_scalar

    def read(value):
        restore()
        try:
            return real_read(value)
        finally:
            install()

    monkeypatch.setattr(Counters, "drain", drain)
    monkeypatch.setattr(loop_mod, "_host_scalar", read)
    run = tiny_run(tmp_path, steps=3, log_every=10**9)
    run = dataclasses.replace(run, model=dataclasses.replace(run.model, moe_topk=0))
    model = _model(run)
    install()
    try:
        train(model, run, log=lambda *_: None)
    finally:
        restore()


# --- gate 5: the log line's loss is the eager number ------------------------


@pytest.mark.local_pin
def test_the_logged_loss_is_the_eager_number(tmp_path: Path):
    """The loss is stored on the device and read at the cadence: same digits."""
    run = tiny_run(tmp_path, steps=5, log_every=5, warmup_tokens=0)
    model = _model(run)
    lines: list[str] = []
    train(model, run, log=lines.append)
    assert lines, "the log cadence must fire"
    assert HEAD_LOSS_FIELD in lines[0], (
        f"the printed loss moved: {lines[0]!r} (HEAD: {HEAD_LOSS_FIELD})"
    )
    assert _sha(model) == HEAD_SHA


def test_a_skipped_step_logs_the_last_finite_loss(tmp_path: Path):
    """The number on a skipped step's line is the last FINITE eager one.

    HEAD kept `last_loss` from the last finite step; the device path must do the
    same through `last_loss_dev` (a `torch.where`, not a raw store). Step 3 is
    poisoned, the cadence fires there, and the line must carry step 2's eager
    value `4.9159` — with the raw store it would print `nan`.
    """
    run = tiny_run(tmp_path, steps=3, log_every=3, warmup_tokens=0)
    model = _NonFiniteLoss(_model(run), poison_on={3})
    lines: list[str] = []
    report = train(model, run, log=lines.append)
    assert report["nan_skipped"] == 1, report["nan_skipped"]
    assert "loss=4.9159" in lines[0], f"the line lost the last finite loss: {lines[0]!r}"
    assert "loss=nan" not in lines[0]


# --- gate 6: a non-finite GRADIENT is caught --------------------------------


def test_a_non_finite_gradient_is_caught_with_a_finite_loss(tmp_path: Path):
    """The burn 2026-09-27 lesson: a masked loss does not stop NaN gradients.

    A parameter hook returns `grad * inf` while the loss stays finite, so the
    loss-only test cannot see it. The device gradient norm must, and the
    update must be zero (parameters bit-identical, `nan_skipped` counted).
    """
    run = tiny_run(tmp_path, steps=1, log_every=1)
    model = _model(run)
    p = dict(model.named_parameters())["loop.norm.weight"]
    p.register_hook(lambda g: g * float("inf"))
    before = [q.detach().clone() for q in model.parameters()]
    report = train(model, run, log=lambda *_: None)
    assert report["nan_skipped"] == 1, "the gradient firewall did not fire"
    for q, b in zip(model.parameters(), before, strict=True):
        assert torch.equal(q, b), "a non-finite gradient must not reach a weight"
