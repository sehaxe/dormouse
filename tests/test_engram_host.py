"""The host-RAM Engram path (`TrainConfig.engram_ram`, burn's `--engram-ram`).

CPU-only, and not as a compromise: on this path the tables ARE plain CPU fp32
memory, so the value parity, the gradient routing and the trainer's
once-per-step contract are all observable without a GPU. What is NOT measured
here is the thing the path exists for — VRAM (nothing in this suite measures it)
and the bus cost of the row transfers (a CPU tensor's `.to("cpu")` is a no-op, so
CPU timings say nothing about the real transfer).

The device refusal is a CUDA-run rule; the host path's NUMERICS are
device-agnostic, which is why `train_loop.host_device_type` is patched where a
test needs `train` to run the host path on this CPU box (that seam has its own
docstring). The refusals themselves are tested unpatched.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from conftest import tiny_cfg, tiny_model, tiny_run

from dormouse.config import load_run
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.model.engram import enable_host_tables, host_step_all, load_host_state
from dormouse.train import load_ckpt, train
from dormouse.train import loop as train_loop
from dormouse.train.loop import load_ngram, ngram_path, save_ckpt
from dormouse.train.optim import MixOptimizer


def _patch_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the trainer take the host path on a CPU model (see the module doc)."""
    monkeypatch.setattr(train_loop, "host_device_type", lambda _m: "cuda")


def test_host_forward_is_bit_identical_to_the_in_vram_forward():
    """Value parity: the host gather reads the SAME fp32 rows, so 0 ULP.

    Tolerance: exact (`torch.equal`), and it is structural rather than lucky —
    `to_host` copies the table's values, `_forward_host` gathers those values and
    one `_read` runs on both substrates. A tolerance here would hide a
    re-ordering of the rows, which is a wrong forward, not a rounding difference.
    """
    model = tiny_model()
    ids = torch.arange(2 * 64, dtype=torch.long).view(2, 64) % 251
    targets = (ids + 1) % 251
    model.eval()
    with torch.no_grad():
        logits_vram, loss_vram, _ = model(ids, targets)
        hidden = torch.randn(2, 64, model.cfg.d_model)
        mem_vram = model.loop.engram(hidden, ids)
    assert model.loop.engram.host is None

    assert enable_host_tables(model) == 1
    assert "loop.engram.tables" not in model.state_dict()
    with torch.no_grad():
        logits_host, loss_host, _ = model(ids, targets)
        mem_host = model.loop.engram(hidden, ids)

    assert torch.equal(mem_vram, mem_host)
    assert torch.equal(logits_vram, logits_host)
    assert torch.equal(loss_vram, loss_host)


def test_gradient_reaches_only_the_host_rows_that_were_read():
    """The touched rows move; every untouched row is bit-unchanged.

    Short ids, so the touched set is a small subset of the 3 x 64 rows and
    "untouched" is not vacuous. The touched set is read off the module's own
    pending pair — the rows the update is handed — not re-derived from the
    addressing formula in the test.
    """
    model = tiny_model()
    engram = model.loop.engram
    n_rows = len(engram.orders) * engram.slots
    before = engram.tables.detach().clone().reshape(-1, engram.dim).clone()
    enable_host_tables(model)

    ids = torch.tensor([[7, 9, 7, 9]])
    hidden = torch.randn(1, 4, model.cfg.d_model, requires_grad=True)
    engram(hidden, ids).sum().backward()
    assert len(engram._pending) == 1, "one forward, one pending leaf"
    touched = engram._pending[0][1]

    updates, rows = engram.host_step(0.1)
    assert (updates, rows) == (1, int(touched.numel()))
    assert 0 < rows < n_rows, "the fixture must leave rows untouched"
    assert engram._pending == [], "the leaves die with the step"

    mask = torch.zeros(n_rows, dtype=torch.bool)
    mask[touched] = True
    after = engram.host.flat()
    assert torch.equal(after[~mask], before[~mask]), "an unread row must not move"
    assert not torch.equal(after[mask], before[mask]), "a read row must move"
    mom = engram.host.momentum.reshape(-1, engram.dim)
    assert torch.equal(mom[~mask], torch.zeros_like(mom[~mask]))
    assert float(mom[mask].abs().sum()) > 0.0


def test_the_step_sums_its_iterations_into_one_nesterov_update():
    """Two forwards over the same keys = ONE update on the SUMMED gradients.

    The loop block calls the memory once per iteration over the same
    `input_ids`, so a per-forward update would apply T Nesterov steps to T
    partial gradients — a different rule, and the one the tables would get
    silently if `host_step` read only the last leaf. The reference is
    `crates/dormouse-train/src/offload.rs::momentum_update` transcribed here:
    `m = mu*m + (1-mu)*g`, `block = mu*m + (1-mu)*g`, `t -= lr*block`, Sinkhorn
    only every 4th update (this is the first, so none). `bias_correction=False`
    on purpose: this gate pins burn's formula verbatim, the A/B's
    `ram_old_formula` arm; the corrected formula has its own gates in
    `test_engram_bias.py`.
    """
    model = tiny_model()
    engram = model.loop.engram
    enable_host_tables(model, bias_correction=False)
    ids = torch.tensor([[3, 5, 3, 11]])
    hidden = torch.randn(2, 4, model.cfg.d_model, requires_grad=True)

    engram(hidden[0:1], ids).sum().backward()
    engram(hidden[1:2], ids).sum().backward()
    assert len(engram._pending) == 2, "two iterations, two leaves"
    grads = sum(leaf.grad.detach() for leaf, _ in engram._pending)
    uniq = engram._pending[0][1]
    assert torch.equal(uniq, engram._pending[1][1]), "same keys, same rows"

    before, momentum = engram.host.rows.clone(), engram.host.momentum.clone()
    updates, rows = host_step_all(model, 0.05)
    assert (updates, rows) == (1, int(uniq.numel()))

    mu = 0.95
    ref_m = momentum.reshape(-1, engram.dim).clone()
    ref_m[uniq] = mu * ref_m[uniq] + (1.0 - mu) * grads
    ref = before.reshape(-1, engram.dim).clone()
    ref[uniq] = ref[uniq] - 0.05 * (mu * ref_m[uniq] + (1.0 - mu) * grads)
    assert torch.equal(engram.host.flat(), ref)
    assert torch.equal(engram.host.momentum, ref_m.reshape(engram.host.rows.shape))
    assert engram.host.step == 1

    # Three more updates: the Sinkhorn cadence runs at the 4th and stays finite.
    for _ in range(3):
        engram(hidden[0:1], ids).sum().backward()
        assert host_step_all(model, 0.05)[0] == 1
    assert engram.host.step == 4
    assert bool(torch.isfinite(engram.host.rows).all())
    assert not torch.equal(engram.host.rows, before)


def test_one_update_per_training_step_and_the_eval_field(tmp_path, monkeypatch):
    """A 10-step run: 10 host updates, rows counted, the eval field still live.

    The counter is the contract: `engram_host_updates` is the number of times
    the trainer touched the tables (one per TRAINING step — never per forward,
    never on an eval, never on a NaN-skipped step), and
    `engram_host_update_rows` says how much of the table the step moved. The
    eval line's `engram=<rows>/<arms>` is burn's field and must survive the
    substrate change: `engram=0/<n>` on this path would be the memory-disabled
    eval that retracted a whole arm (AGENTS §3.2).
    """
    _patch_device(monkeypatch)
    run = tiny_run(
        tmp_path, steps=10, log_every=10, eval=True, eval_every=5, eval_batches=2, engram_ram=True
    )
    model = DormouseModel(run.model)
    lines: list[str] = []
    report = train(model, run, log=lines.append)

    assert report["counters"]["engram_host_updates"] == 10
    assert report["counters"]["engram_host_update_rows"] > 0
    assert model.loop.engram.host.step == 10
    assert "loop.engram.tables" not in model.state_dict()
    assert any("engram_ram: 1 table set(s) in host RAM" in ln for ln in lines)
    log_line = [ln for ln in lines if ln.startswith("step=10")][0]
    assert "engram_host_updates=10" in log_line and "engram_host_update_rows=" in log_line

    evals = [ln for ln in lines if ln.startswith("eval ")]
    assert evals, "the run asked for evals"
    for ln in evals:
        rows, arms = ln.split(" engram=")[1].split()[0].split("/")
        assert int(rows) > 0 and int(arms) > 0, ln


def test_host_tables_round_trip_through_the_ngram_sibling(tmp_path, monkeypatch):
    """`save_ckpt` writes `<ckpt>.ngram`; `load_host_state` restores it exactly."""
    _patch_device(monkeypatch)
    run = tiny_run(tmp_path, steps=1, engram_ram=True)
    model = DormouseModel(run.model)
    train_loop.host_tables(model)
    ids = torch.arange(2 * 64, dtype=torch.long).view(2, 64) % 251
    model(ids, (ids + 1) % 251)[1].backward()
    assert host_step_all(model, 0.01)[0] == 1
    opt = MixOptimizer(model, lr=run.train.lr, wd=run.train.wd, opt=run.train.opt)
    save_ckpt(run, model, opt, 0)

    ng = ngram_path(run.train)
    assert ng.is_file() and ng.name == "t.ngram"

    fresh = DormouseModel(run.model)
    train_loop.host_tables(fresh)
    assert not torch.equal(fresh.loop.engram.host.rows, model.loop.engram.host.rows)
    load_host_state(fresh, load_ngram(run.train))
    assert torch.equal(fresh.loop.engram.host.rows, model.loop.engram.host.rows)
    assert torch.equal(fresh.loop.engram.host.momentum, model.loop.engram.host.momentum)
    assert torch.equal(fresh.loop.engram.host.counts, model.loop.engram.host.counts)
    assert int(model.loop.engram.host.counts.sum()) > 0, "the step touched rows"
    assert fresh.loop.engram.host.step == model.loop.engram.host.step == 1


def test_resume_continues_a_host_run(tmp_path, monkeypatch):
    """A legal resume loads the sibling and keeps updating the host tables."""
    _patch_device(monkeypatch)
    first = tiny_run(tmp_path, steps=2, ckpt_every=2, log_every=10, engram_ram=True)
    train(DormouseModel(first.model), first, log=lambda *_: None)
    assert ngram_path(first.train).is_file()

    second = tiny_run(tmp_path, steps=4, ckpt_every=2, log_every=10, engram_ram=True)
    assert second.diff_keys(first) == []
    model = DormouseModel(second.model)
    ck, _stored = load_ckpt(second)
    report = train(model, second, resume=True, log=lambda *_: None)
    assert report["opt_steps"] == 4, "the optimizer step count continues"
    assert model.loop.engram.host.step == 4, "the host updates continue from the sibling"
    assert "loop.engram.tables" not in ck["model"]


def test_refusals(tmp_path, monkeypatch):
    """Every way this path can be asked for and not delivered is a hard error."""
    # (1) byteflow: the channel runs over patch latents, no byte position to hash.
    # `use_engram=false` is what makes THIS refusal the first one reached — a
    # byteflow run with the memory arm still on is refused even earlier by
    # `_validate_byteflow` ("use_byteflow with use_engram=true"), which is the
    # same head-on-impossible pair with the same escape.
    with pytest.raises(LoudError) as e:
        load_run(
            "small",
            model_overrides={
                "use_byteflow": True,
                "byteflow_k_tokens": 128,
                "bf16": False,
                "use_engram": False,
            },
            train_overrides={"engram_ram": True},
        )
    assert "engram_ram" in str(e.value) and "use_byteflow=true" in str(e.value)

    # (2) use_engram=false: there is no table to host.
    with pytest.raises(LoudError) as e:
        load_run(
            "small",
            model_overrides={"use_engram": False},
            train_overrides={"engram_ram": True},
        )
    assert "use_engram=false" in str(e.value)

    # (3) a CPU model has no VRAM to save. The MODEL's device decides, not the
    # `device` argument (a caller can pass neither and still hold a cuda model).
    run = tiny_run(tmp_path, steps=1, engram_ram=True)
    model = DormouseModel(run.model)
    with pytest.raises(LoudError) as e:
        train(model, run, log=lambda *_: None)
    assert "no VRAM to buy" in str(e.value) and "engram_ram=false" in str(e.value)
    assert model.loop.engram.host is None, "a refused run must not half-switch"
    assert "loop.engram.tables" in model.state_dict()

    # (4) a host run's resume without its sibling: the model half carries no
    # table, so the pair would be a different network at the same step count.
    # The checkpoint pair is a run that did NOT use host tables — the sibling is
    # simply absent, which is the crash-between-two-writes shape.
    off = tiny_run(tmp_path, steps=2, ckpt_every=2, log_every=10)
    train(DormouseModel(off.model), off, log=lambda *_: None)
    flipped = dataclasses.replace(off, train=dataclasses.replace(off.train, engram_ram=True))
    assert not ngram_path(off.train).is_file()
    with pytest.raises(LoudError) as e:
        load_ckpt(flipped)
    assert ".ngram" in str(e.value) and "sibling" in str(e.value)

    # (5) the flag flipped between runs on one ckpt: a non-progress key. The
    # sibling's presence does not excuse it — the two halves would disagree
    # about which network this step count belongs to.
    ngram_path(off.train).write_bytes(b"a stand-in sibling: the drift check runs first")
    with pytest.raises(LoudError) as e:
        load_ckpt(flipped)
    assert "train.engram_ram" in str(e.value) and "resume refused" in str(e.value)

    # (6) a hand-built model with no Engram gets the other refusal, not a no-op.
    _patch_device(monkeypatch)
    bare = DormouseModel(dataclasses.replace(tiny_cfg(), use_engram=False))
    with pytest.raises(LoudError) as e:
        train_loop.host_tables(bare)
    assert "no Engram to host" in str(e.value)
