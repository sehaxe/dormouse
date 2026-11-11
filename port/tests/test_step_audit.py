"""The step audit's own gates, plus the pins the fixes must not move.

Three kinds of assertion, kept apart on purpose:

* **Pins** — the CLEAN-MAIN sha256 of the tiny model's logits, its initial
  state, and its state after two training steps, and the step-2 loss. They were
  taken on `main` before any edit in this lane (see `PINS_FROM`); every fix
  here is pure-waste removal, so the off state must reproduce them exactly.
* **Instrument honesty** — the audit counts a known `.item()` and a known loop,
  and `Counters.drain` reads N counters with ONE host read.
* **The step's budget** — a non-report step takes no host read outside a
  CUDA-only fallback, and the tiny step's allocations stay bounded (the bound
  is what fails if a fresh zero/square-key pass comes back).
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest
import torch
from conftest import BATCH, SEQ, tiny_cfg, tiny_model, tiny_run

from dormouse import stepaudit
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel
from dormouse.model.engram import Engram
from dormouse.train import train

#: `main` @ 3e736ae, torch 2.13.0+cu129 CPU, `python /tmp/opencode/pin_step.py`.
#: Regenerate with that script on a clean checkout; a mismatch means the step
#: moved, not that the pin is stale.
PINS_FROM = "main@3e736ae"
LOGITS_SHA = "c7839283a79291cf713aab57476cf202ba1a68167c365197f785da08203055c2"
INIT_SHA = "06eecd1d36bcf49ea3eb27247c32409487c3cf667fe48be1108e03cae6405b8a"
#: Re-pinned 2026-11-11 (Muon's small-side Newton–Schulz): the 2-step state is
#: downstream of the NS chain, the forward-only pins above are not.
STATE_SHA = "8ce665773a42bf4200c321f97713e693e33eeaacb1d496ab99761e90109b178f"
LOSS2 = 4.442110538483


def _tensor_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().to(torch.float32).contiguous().numpy().tobytes()).hexdigest()


def _state_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().to(torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def test_off_state_is_bit_identical_to_clean_main():
    torch.manual_seed(1234)
    model = DormouseModel(tiny_cfg())
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    tgt = (ids + 1) % 251
    with torch.no_grad():
        logits, _loss, _c = model(ids, tgt)
    assert _tensor_sha(logits) == LOGITS_SHA
    assert _state_sha(model) == INIT_SHA

    torch.manual_seed(1234)
    model = DormouseModel(tiny_cfg())
    with tempfile.TemporaryDirectory() as td:
        run = tiny_run(Path(td), steps=2, log_every=1)
        report = train(model, run, log=lambda _line: None)
    assert _state_sha(model) == STATE_SHA
    assert f"{report['loss']:.12f}" == f"{LOSS2:.12f}"


def test_audit_counts_a_known_read_a_known_loop_and_an_allocation():
    audit = stepaudit.install()
    try:
        audit.begin_step(1)
        torch.zeros(3)
        torch.tensor(7).item()
        for _ in stepaudit.iterations("known.loop", range(4)):
            pass
        with stepaudit.loop("known.ctx") as probe:
            for _ in range(3):
                probe.tick()
        audit.end_step()
    finally:
        stepaudit.uninstall()
    rec = audit.records[0]
    assert rec.reads["item"] == 1
    assert rec.loops["known.loop"].instances == 1
    assert rec.loops["known.loop"].iterations == 4
    assert rec.loops["known.ctx"].instances == 1
    assert rec.loops["known.ctx"].iterations == 3
    assert rec.allocs >= 1 and rec.alloc_bytes >= 3 * 4


def test_drain_reads_n_counters_with_one_host_read():
    """The 12-sync drain is the defect this pins: N counters, ONE transfer."""
    audit = stepaudit.install()
    try:
        audit.begin_step(1, report=True)
        c = Counters()
        for i in range(12):
            c.count_device(f"c{i}", torch.tensor(i, dtype=torch.int64))
        info = [torch.tensor([1.0, 0.0]), torch.tensor(2.5), torch.tensor(7)]
        c.count_device("vec", info[0])
        c.count_device("norm", info[1])
        c.count_device("n", info[2])
        out = c.drain()
        audit.end_step()
    finally:
        stepaudit.uninstall()
    reads = sum(audit.records[0].reads.values())
    assert reads == 1, f"drain took {reads} host reads, expected 1"
    assert out["c11"] == 11 and out["n"] == 7 and out["norm"] == 2.5
    assert out["vec"] == [1, 0]
    assert c["vec[0]"] == 1 and "vec[1]" not in c


def test_drain_if_due_is_zero_cost_off_cadence():
    audit = stepaudit.install()
    try:
        audit.begin_step(1, report=False)
        c = Counters()
        c.count_device("x", torch.tensor(5))
        assert c.drain_if_due(1, 100) is None
        assert "x" in c._device, "a non-due drain must not consume the counter"
        audit.end_step()
        assert sum(audit.records[0].reads.values()) == 0
        assert c.drain_if_due(100, 100) == {"x": 5}
        assert "x" not in c._device
    finally:
        stepaudit.uninstall()


def test_drain_cadence_refuses_zero_loudly():
    c = Counters()
    with pytest.raises(LoudError, match="Escape:"):
        c.drain_if_due(1, 0)


def test_report_step_reads_once_and_non_report_steps_read_nothing_ours(tmp_path):
    """The 0-sync proof: no host read on a non-report step outside CUDA-only paths."""
    torch.manual_seed(0)
    audit = stepaudit.install()
    try:
        run = tiny_run(tmp_path, steps=4, log_every=3)
        train(DormouseModel(tiny_cfg()), run, log=lambda _line: None)
    finally:
        stepaudit.uninstall()

    for rec in audit.records:
        assert sum(rec.device_reads.values()) == 0, "the CPU run has no device tensors"
        if rec.report:
            continue
        unexpected = [
            f"{n}x {frame}"
            for frame, n in rec.frames.items()
            if not any(m in frame for m in stepaudit.CPU_ONLY_FRAMES)
        ]
        assert not unexpected, f"step {rec.index} read the host: {unexpected}"

    report = [r for r in audit.records if r.report]
    assert report, "the run must have a report step"
    drain_frames = {f: n for f, n in report[0].frames.items() if ":drain" in f}
    assert sum(drain_frames.values()) == 1, (
        f"a report step must read its counters ONCE, got {drain_frames}"
    )


def test_allocations_per_step_stay_under_the_measured_bound(tmp_path):
    """Re-introducing the per-iteration zeros/keys moves this bound."""
    torch.manual_seed(0)
    audit = stepaudit.install()
    try:
        run = tiny_run(tmp_path, steps=4, log_every=3)
        train(DormouseModel(tiny_cfg()), run, log=lambda _line: None)
    finally:
        stepaudit.uninstall()
    steady = audit.records[1:]  # step 1 builds momentum tables and lazy buffers
    assert max(r.allocs for r in steady) <= 27
    assert max(r.alloc_bytes for r in steady) <= 460_000


def test_the_engram_keys_are_computed_once_per_window(monkeypatch):
    """`max_iter` iterations over the same ids must NOT re-key the n-grams."""
    model = tiny_model(max_iter=3)
    calls: list[int] = []
    orig = Engram.slots_for

    def counted(self, input_ids):
        calls.append(int(input_ids.shape[0]))
        return orig(self, input_ids)

    monkeypatch.setattr(Engram, "slots_for", counted)
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    tgt = (ids + 1) % 251
    with torch.no_grad():
        model(ids, tgt)
    assert len(calls) == 1, f"slots_for ran {len(calls)}x for max_iter=3"


def test_precomputed_slots_are_the_same_read():
    """Passing the window's keys through is bit-identical to recomputing them."""
    torch.manual_seed(0)
    eng = Engram(tiny_cfg())
    hidden = torch.randn(BATCH, SEQ, 64)
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    slots = eng.slots_for(ids)
    with torch.no_grad():
        a = eng(hidden, ids)
        b = eng(hidden, ids, slots=slots)
    assert torch.equal(a, b)


def test_same_dtype_to_and_contiguous_are_free():
    """Item 4's premise, checked at the ATen contract instead of assumed: a
    same-dtype `.to` and a contiguous `.contiguous()` return the SAME tensor,
    so the remaining calls in the step cannot be copies. The audit's
    `noop_contig` covers the other half: a contiguous input to `.contiguous()`
    would be counted, and a step has none."""
    t = torch.zeros(4)
    assert t.contiguous() is t
    assert t.to(t.dtype) is t
    nc = torch.zeros(2, 3).t()
    assert not nc.is_contiguous()
    assert nc.contiguous() is not nc


def test_no_noop_contiguous_calls_in_a_step(tmp_path):
    torch.manual_seed(0)
    audit = stepaudit.install()
    try:
        run = tiny_run(tmp_path, steps=2, log_every=10)
        train(DormouseModel(tiny_cfg()), run, log=lambda _line: None)
    finally:
        stepaudit.uninstall()
    assert audit.report()["noop_contig"] == 0


def test_ema_foreach_matches_the_parameter_loop():
    """The EMA arm's one-foreach form is the same recurrence, bit for bit."""
    from dormouse.train.loop import _ema_update

    model = tiny_model()
    shadow = {n: p.detach().clone() for n, p in model.named_parameters()}
    ref = {k: v.clone() for k, v in shadow.items()}
    _ema_update(model, shadow, 0.9, torch.ones(()))
    for name, p in model.named_parameters():
        ref[name].lerp_(p.detach(), torch.tensor(0.1))
        assert torch.equal(shadow[name], ref[name])

    frozen = {k: v.clone() for k, v in shadow.items()}
    _ema_update(model, shadow, 0.9, torch.zeros(()))
    for name, v in shadow.items():
        assert torch.equal(v, frozen[name]), f"a gated EMA step moved {name}"


def test_every_step_loop_is_counted_or_excused():
    """The census is the anti-rot gate: a new hot loop must say what it is."""
    unmarked = stepaudit.unmarked()
    assert not unmarked, (
        "loops in the step's files with no counter and no reason:\n"
        + "\n".join(f"  {s.file}:{s.line} {s.kind} {s.function}" for s in unmarked)
        + "\nEscape: wrap the loop in stepaudit.iterations(...), or add a reason "
        "to stepaudit.EXCUSED_FUNCTIONS if it is not part of a step."
    )
    counted = [s for s in stepaudit.census() if s.status == "counted"]
    assert len(counted) >= 20, "the instrumented set collapsed — did a marker get lost?"
