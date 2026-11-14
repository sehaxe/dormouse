"""Token-Superposition Training (arXiv:2605.06546v2) on a byte-level model.

A bag is `k` consecutive BYTES and the vocabulary is 256, so one position's
objective is

    loss = -log( (1/k) · Σ_i p[b_i] ),   p = softmax(logits over 256)

`tst_bag=1` is OFF and must be today's program bit for bit. The gates below are
the ones that make that claim checkable: the k=1 loss, the two hand-computed
bags, the ORDER invariance (a bag is a set — the property an ordinary CE
implementation silently loses), and the byte count the bag window reads.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg, tiny_run
from torch import nn

from dormouse.config import (
    RunConfig,
    TrainConfig,
    tst_bag_at,
    tst_bytes,
    tst_phase1_steps,
    tst_stream_bytes,
)
from dormouse.instrument import LoudError
from dormouse.losses import multihot_cross_entropy
from dormouse.model import DormouseModel
from dormouse.train import train
from dormouse.train.data import ByteStream

#: `block_form="transformer"` is the form whose loss IS one readout CE — the
#: formula above. (The KDA loop's `L_Rec` is a per-iteration single-byte CE and
#: is refused, see `test_the_loop_form_is_refused`.)
DENSE = dict(
    block_form="transformer",
    n_layers=2,
    max_iter=1,
    use_kda=False,
    use_engram=False,
    use_csa=False,
    layerrope=False,
    tie_embeddings=False,
)


def _corpus(path, n_bytes=4096, cycle=251):
    path.write_bytes(bytes(range(cycle)) * (n_bytes // cycle + 1))
    return path


# --- the formula -------------------------------------------------------------


def test_k1_is_the_ordinary_ce_bit_for_bit():
    """`k=1` is the ordinary CE — `torch.equal`, not a tolerance."""
    torch.manual_seed(0)
    logits = torch.randn(256, 256) * 4.0
    targets = torch.randint(0, 256, (256,))
    bag = multihot_cross_entropy(logits, targets.reshape(-1, 1))
    plain = -F.log_softmax(logits, dim=-1).gather(1, targets.reshape(-1, 1)).mean()
    assert torch.equal(bag, plain), (float(bag), float(plain))


def test_a_certain_bag_scores_zero():
    """Two certain bytes (p = 1 each, identical) → mean p = 1 → loss exactly 0."""
    logits = torch.tensor([[0.0, -1000.0, -1000.0, -1000.0]])
    loss = multihot_cross_entropy(logits, torch.tensor([[0, 0]]))
    assert float(loss) == 0.0


def test_two_distinct_half_probability_bytes_give_minus_log_half():
    """p = (0.5, 0.5) over two distinct bag members → `-log(0.5)`, the formula."""
    logits = torch.tensor([[0.0, 0.0, -1000.0, -1000.0]])
    loss = multihot_cross_entropy(logits, torch.tensor([[0, 1]]))
    assert float(loss) == pytest.approx(math.log(2.0), abs=1e-6)


def test_the_bag_is_not_the_average_of_two_ces():
    """The discriminating case: p = (0.8, 0.2) in one bag.

    The formula: `-log((0.8 + 0.2)/2) = -log(0.5) = log 2`. The average of two
    CEs would be `-(log 0.8 + log 0.2)/2 = 0.9163`. A bag that covers the mass
    is easy to predict; that is the arm's whole point.
    """
    logits = torch.tensor([[math.log(0.8), math.log(0.2)]])
    loss = float(multihot_cross_entropy(logits, torch.tensor([[0, 1]])))
    average_ce = -(math.log(0.8) + math.log(0.2)) / 2
    assert loss == pytest.approx(math.log(2.0), abs=1e-6)
    assert abs(loss - average_ce) > 0.2, "the fixture must separate the two formulas"


def test_the_loss_is_invariant_to_the_order_inside_a_bag():
    """A bag is a SET: shuffling it returns the same BITS, for every k."""
    torch.manual_seed(3)
    logits = torch.randn(48, 256) * 5.0
    for k in (2, 3, 4, 7):
        bags = torch.stack([torch.randperm(256)[:k] for _ in range(48)])
        base = multihot_cross_entropy(logits, bags, reduction="none")
        for _ in range(3):
            perm = torch.argsort(torch.rand(48, k), dim=1)
            shuffled = bags.gather(1, perm)
            again = multihot_cross_entropy(logits, shuffled, reduction="none")
            assert torch.equal(base, again), f"k={k} is order-dependent"


def test_duplicates_are_counted_k_times_as_the_paper_writes_it():
    """`Σ_i p[b_i]` over a bag of `k` copies of one byte is `Σ = k·p`, so the mean
    is `p` and the loss is that byte's CE — the identity the formula implies."""
    logits = torch.tensor([[0.0, 1.0]])
    twice = float(multihot_cross_entropy(logits, torch.tensor([[0, 0]])))
    once = float(multihot_cross_entropy(logits, torch.tensor([[0]])))
    assert twice == pytest.approx(once, abs=1e-6)


# --- the data path: a window carries seq_len·k bytes --------------------------


def test_the_bag_window_reads_seq_len_times_k_bytes(tmp_path):
    """Asserted on CONTENT and on the STRIDE, never inferred from the code."""
    path = _corpus(tmp_path / "corpus.bin")
    batch, seq_len, k = 2, 4, 3
    stream = ByteStream(path, batch, seq_len)
    raw = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8).long()
    span = seq_len * k + 1  # bytes per ROW

    first = raw[: batch * span].view(batch, span)
    x1, y1 = stream.next_batch(k)
    assert torch.equal(x1, first[:, :-1:k]), "position q must read the byte at q·k"
    assert torch.equal(y1, first[:, 1:].view(batch, seq_len, k))
    # The SECOND draw starts `batch·(seq_len·k+1)` bytes in: the byte count as a
    # stride, which a flat `seq_len+1` window would not produce.
    second = raw[batch * span : 2 * batch * span].view(batch, span)
    x2, y2 = stream.next_batch(k)
    assert torch.equal(x2, second[:, :-1:k])
    assert torch.equal(y2, second[:, 1:].view(batch, seq_len, k))


def test_bag_1_is_the_flat_window(tmp_path):
    """`bag=1` is `_windows` argument for argument — no bag axis, no stride."""
    path = _corpus(tmp_path / "corpus.bin")
    a, b = ByteStream(path, 2, 8), ByteStream(path, 2, 8)
    x_bag, y_bag = a.next_batch(1)
    x_flat, y_flat = b.next_batch()
    assert x_bag.shape == x_flat.shape == (2, 8) and y_bag.shape == (2, 8)
    assert torch.equal(x_bag, x_flat) and torch.equal(y_bag, y_flat)


def test_the_phase_boundary_is_a_pure_function_of_the_config(tmp_path):
    cfg = dataclasses.replace(
        tiny_run(tmp_path).train, steps=10, batch=2, seq_len=4, tst_bag=3, tst_frac=0.5
    )
    assert tst_phase1_steps(cfg) == 5
    assert [tst_bag_at(cfg, s) for s in (0, 4, 5, 9)] == [3, 3, 1, 1]
    # The byte accounting, asserted as numbers: a bag position scores
    # `k` bytes and its row reads `seq_len·k + 1` from the stream.
    assert tst_bytes(cfg, 1) == 2 * 4 * 3
    assert tst_stream_bytes(cfg, 1) == 2 * (4 * 3 + 1)
    assert tst_bytes(cfg, 5) == 5 * 2 * 4 * 3
    assert tst_stream_bytes(cfg, 5) == 5 * 2 * 13
    assert tst_stream_bytes(cfg, 6) == 5 * 2 * 13 + 2 * 5
    # An unarmed run is priced exactly as it was before the arm existed: the
    # budget is the positions product and the fast-forward its `+1` twin.
    off = dataclasses.replace(cfg, tst_bag=1)
    assert tst_phase1_steps(off) == 0
    assert tst_bytes(off, 7) == 7 * 2 * 4
    assert tst_stream_bytes(off, 7) == 7 * 2 * (4 + 1)


# --- the trainer -------------------------------------------------------------


class RecordingStream:
    """The flat stream under a recorder, so the BAG SIZE of each draw is visible."""

    def __init__(self, path, batch, seq_len):
        self.base = ByteStream(path, batch, seq_len)
        self.batch, self.seq_len = batch, seq_len
        self.calls: list[int] = []

    def next_batch(self, bag: int = 1):
        self.calls.append(bag)
        return self.base.next_batch(bag)

    def close(self):
        self.base.close()


def _dense_run(tmp_path, **over):
    base = dict(
        steps=4,
        batch=2,
        seq_len=8,
        lr=0.01,
        log_every=100,
        ckpt_every=100,
        ckpt_name="t",
        out_dir=str(tmp_path / "ckpts"),
        data=str(_corpus(tmp_path / "corpus.bin")),
    )
    cfg = TrainConfig(**{**base, **over})
    return RunConfig(source="tst", model=tiny_cfg(**DENSE), train=cfg)


def test_the_trainer_switches_phase_mid_run(tmp_path):
    """Phase 1 draws bags, the recovery phase draws flat windows, in ONE run."""
    run = _dense_run(tmp_path, tst_bag=2, tst_frac=0.5, log_every=2)
    stream = RecordingStream(_corpus(tmp_path / "corpus.bin"), 2, 8)
    lines: list[str] = []
    report = train(DormouseModel(run.model), run, stream=stream, log=lines.append)
    assert stream.calls == [2, 2, 1, 1]
    assert any(line.startswith("tst:") for line in lines), lines
    assert any("superposition phase ends at step 2" in line for line in lines), lines
    # The log line names the bag in force and the phase of the step it reports.
    assert any("tst_bag=2 tst_phase=1" in line for line in lines), lines
    assert any("tst_bag=1 tst_phase=2" in line for line in lines), lines
    counters = report["counters"]
    assert counters["tst_bags"] == 2
    assert counters["tst_bytes"] == 2 * 2 * (8 * 2 + 1) + 2 * 2 * (8 + 1)
    assert counters["tst_bag_bytes"] == 2 * 2 * (8 * 2 + 1)


def test_bag_1_is_todays_step_and_todays_loss(tmp_path, monkeypatch):
    """The gate: with `tst_bag=1` the step is the historical one.

    The forward is spied on to prove three things at once: it is called with a
    2-D target (no bag axis), the loss it returns is `torch.equal` to the
    historical `-log_softmax(...).gather(...).mean()` of the SAME logits, and the
    arm leaves no counter and no line behind.
    """
    seen: dict = {}
    original = DormouseModel.forward

    def spy(self, input_ids, targets=None, **kw):
        logits, loss, counters = original(self, input_ids, targets, **kw)
        seen.update(logits=logits, loss=loss, targets=targets)
        return logits, loss, counters

    monkeypatch.setattr(DormouseModel, "forward", spy)
    lines: list[str] = []
    run = _dense_run(tmp_path, steps=2, log_every=100)
    report = train(DormouseModel(run.model), run, log=lines.append)

    assert seen["targets"].ndim == 2, "k=1 must not grow a bag axis"
    historical = (
        -F.log_softmax(seen["logits"], dim=-1).gather(1, seen["targets"].reshape(-1, 1)).mean()
    )
    assert torch.equal(seen["loss"], historical)
    assert not [k for k in report["counters"] if k.startswith("tst_")]
    assert not [line for line in lines if line.startswith("tst:")]


def test_the_resume_fast_forward_lands_after_the_bag_bytes(tmp_path):
    """Resume accounting, checked against the FILE: the skip is the bag bytes."""
    run = _dense_run(tmp_path, steps=4, tst_bag=2, tst_frac=0.5)
    path = _corpus(tmp_path / "corpus.bin")
    raw = path.read_bytes()
    stream = RecordingStream(path, 2, 8)
    skip = tst_stream_bytes(run.train, 2)  # steps 0,1 drew 2·(8·2+1) B per row
    assert skip == 2 * 2 * 17
    stream.base.skip_bytes(skip)
    x, _y = stream.base.next_batch()
    assert int(x[0, 0]) == raw[skip]


# --- the refusals ------------------------------------------------------------


def _validate(**train_over):
    from dormouse.config import validate_model_train

    tmp = _corpus_dir()
    base = dict(
        steps=10,
        batch=2,
        seq_len=8,
        data=str(tmp / "corpus.bin"),
        out_dir=str(tmp),
        ckpt_name="t",
    )
    cfg = TrainConfig(**{**base, **train_over})
    return lambda model_over=None: validate_model_train(
        tiny_cfg(**{**DENSE, **(model_over or {})}), cfg
    )


def _corpus_dir():
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    _corpus(tmp / "corpus.bin")
    return tmp


def test_the_loop_form_is_refused():
    """`L_Rec` is a per-iteration CE against ONE next byte: no bag can score it."""
    with pytest.raises(LoudError, match="block_form='kda'"):
        _validate(tst_bag=2)({"block_form": "kda", "n_layers": 0})


def test_the_bag_size_must_be_a_count():
    with pytest.raises(LoudError, match="tst_bag=0"):
        _validate(tst_bag=0)()


def test_a_phase_that_never_fires_is_refused():
    with pytest.raises(LoudError, match="tst_frac=0.0"):
        _validate(tst_bag=2, tst_frac=0.0)()
    with pytest.raises(LoudError, match="= 0 steps"):
        _validate(steps=2, tst_bag=2, tst_frac=0.1)()


@pytest.mark.parametrize(
    "train_over, model_over, needle",
    [
        ({"doc_masking": True}, {}, "doc_masking"),
        ({"mix_plan": "plan.toml"}, {}, "mix_plan"),
        ({"full_graph": True}, {}, "full_graph"),
        ({}, {"use_byteflow": True}, "use_byteflow"),
        ({}, {"mod_aux_weight": 0.1}, "mod_aux_weight"),
        ({}, {"mtp_weight": 0.1}, "mtp_weight"),
    ],
)
def test_the_bag_arm_refuses_what_it_cannot_score(train_over, model_over, needle):
    with pytest.raises(LoudError, match=needle):
        _validate(tst_bag=2, **train_over)(model_over)


def test_the_arm_is_inert_when_off():
    """`tst_bag=1` accepts everything the refusals above name."""
    _validate()()  # writes nothing, raises nothing: 1 is the off value


def test_the_cli_set_path_carries_both_keys():
    """`--set` is schema-routed, so the fields are reachable without a preset edit."""
    from dormouse.cli import _split_sets

    _model_over, train_over = _split_sets(["tst_bag=4", "tst_frac=0.25"])
    assert train_over == {"tst_bag": 4, "tst_frac": 0.25}
    assert isinstance(train_over["tst_bag"], int)
    assert isinstance(train_over["tst_frac"], float)


def test_the_bag_loss_reaches_the_parameters(tmp_path):
    """A backward through the bag CE moves the head: the path is not detached."""
    torch.manual_seed(0)
    head = nn.Linear(4, 256)
    x = torch.randn(6, 4, requires_grad=False)
    bags = torch.randint(0, 256, (6, 3))
    loss = multihot_cross_entropy(head(x), bags)
    loss.backward()
    assert head.weight.grad is not None and float(head.weight.grad.abs().sum()) > 0.0
