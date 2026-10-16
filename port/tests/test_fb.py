"""The future-byte aux head (arXiv 2404.19737) — the pre-registered A/B arm.

`docs/research/2026-10-16-jepa-dspark-alternatives.md` is the spec: rows 1/3/4/5
cost this arm against JEPA+KoLeo (2.1% of `small` against a term whose measured
price is a halved legal batch) and §4.1 item 1 pre-registers the A/B it feeds.
"""

from __future__ import annotations

import dataclasses
import re

import pytest
import torch
from conftest import tiny_cfg, tiny_run

from dormouse.config import ModelConfig, load_preset, parse_preset, validate
from dormouse.instrument import Counters, LoudError
from dormouse.model import (
    FB_PARAMS_AT_SMALL,
    DormouseModel,
    FutureByteHead,
    future_byte_loss,
    param_count,
)
from dormouse.model.fb import valid_positions
from dormouse.train import is_muon_param, train


def _first_loss(model, run) -> float:
    from dormouse.train.data import open_streams

    stream, _ = open_streams(run)
    ids, targets = stream.next_batch()
    _, loss, _ = model(ids, targets)
    stream.rewind()
    return float(loss.detach())


def test_the_head_runs_and_the_loss_drops_at_a_nonzero_weight(tmp_path):
    """(a) 30 steps at `aux_fb_weight=0.05`: the arm RUNS and the loss descends.

    Seeded for the reason `test_model.test_loss_decreases_on_fixed_bytes` is: at
    a hot single lr the small 1-D groups make an unseeded 30-step run a coin
    flip, so the seed pins the run this gate measures.
    """
    torch.manual_seed(0)
    run = tiny_run(tmp_path, steps=30, lr=0.01, log_every=100)
    run = dataclasses.replace(run, model=tiny_cfg(aux_fb_weight=0.05))
    model = DormouseModel(run.model)
    assert model.fb is not None
    first = _first_loss(model, run)
    report = train(model, run, log=lambda *_: None)
    assert report["loss"] < first, f"loss did not drop: {first} -> {report['loss']}"
    assert report["nan_skipped"] == 0
    # The pre-registered instrument: asked == ran == steps. `0/<n>` would mean
    # the sequence was shorter than the shift and the arm trained nothing.
    assert report["counters"]["fb_asked"] == run.train.steps
    assert report["counters"]["fb_ran"] == run.train.steps
    # The head trains — the burn defect class is an arm that runs and trains nothing.
    assert model.fb.proj.weight.grad is not None


def test_the_eval_line_carries_fb_ran_over_asked(tmp_path):
    """(b) `fb=<ran>/<asked>` on the eval line, the pre-registered metric (2).

    PRE-REGISTRATION, quoted so no future agent can move it silently
    (`docs/research/2026-10-16-jepa-dspark-alternatives.md` §4.1 item 1): *"the
    A/B: `aux_fb_weight` (future-byte head) vs the port's current pure-CE
    recipe"*, 3 seeds x 2 000 steps, matched `batch . seq_len`; *"Metrics in
    order: (1) held-out BPB at matched steps, window printed; (2) `fb=<ran>/<asked>`
    — `0/<n>` means the horizon was past the sequence length and nothing trained
    (head did not run ⇒ void); (3) a probe ... because the loss cannot answer the
    question"*. *"Decision rule (unchanged): win = arm mean BPB below the pure-CE
    mean by more than the spread across the control's own 3 seeds; inside the
    spread = delete (ADR-0002)"*, and the pre-registered prediction is that at
    9.2M the head *"lands inside the spread or worse"*.

    The count is the run's TRAINING forwards (`train` passes `total`), not the
    eval's own: the field must say how much of the arm trained, not how many
    eval forwards happened.
    """
    torch.manual_seed(0)
    on = tiny_run(tmp_path, steps=4, log_every=100, eval=True, eval_every=2, eval_batches=3)
    on = dataclasses.replace(on, model=tiny_cfg(aux_fb_weight=0.05))
    lines: list[str] = []
    train(DormouseModel(on.model), on, log=lines.append)
    evals = [ln for ln in lines if ln.startswith("eval ")]
    assert evals, lines
    m = re.search(r"fb=(\d+)/(\d+)", evals[0])
    assert m, evals[0]
    assert (int(m.group(1)), int(m.group(2))) == (2, 2), evals[0]
    # Weight 0.0 is the control and it is VISIBLE, not omitted: an absent field
    # is indistinguishable from an arm nobody asked about.
    off = tiny_run(tmp_path, steps=2, log_every=100, eval=True, eval_every=2, eval_batches=3)
    off_lines: list[str] = []
    train(DormouseModel(off.model), off, log=off_lines.append)
    assert "fb=0/0" in " ".join(off_lines), off_lines


def test_a_window_shorter_than_the_shift_is_void_and_counted():
    """The `0/<n>` half of the pre-registration: ASKED, and it produced nothing.

    A one-byte window has no shifted label. The term is then an exact zero that
    bumps `fb_asked` and not `fb_ran`, so a 2 000-step A/B at a too-short
    `seq_len` is visible as a run of nothing instead of looking healthy.
    """
    head = FutureByteHead(8, 256)
    counters = Counters()
    loss = future_byte_loss(
        head,
        torch.randn(2, 1, 8),
        torch.zeros(2, 1, dtype=torch.long),
        counters,
    )
    assert float(loss) == 0.0 and dict(counters) == {"fb_asked": 1}
    assert valid_positions(1) == 0 and valid_positions(2) == 1 and valid_positions(9) == 8


def test_only_the_rolled_labels_are_read():
    """The head at `q` reads `targets[q+1]` — rolled by -1, one byte further on.

    This is DSpark's D1 read done the other way (`docs/papers/dspark.md` §7:
    the draft head was handed, at trial `q`, the byte the main logits at `q`
    had just predicted). Enumerated, one target entry at a time: the entries
    the window may NOT read have to leave the term bitwise equal.
    """
    torch.manual_seed(0)
    head = FutureByteHead(4, 8)
    h = torch.randn(1, 4, 4)
    y = torch.tensor([[1, 2, 3, 4]])

    def term(labels: torch.Tensor) -> float:
        return float(future_byte_loss(head, h, labels, Counters()).detach())

    base = term(y)
    moved = []
    for i in range(4):
        other = y.clone()
        other[0, i] = (int(other[0, i]) + 5) % 8
        moved.append(term(other) != base)
    # Rows q = 0..2 read targets[1], targets[2], targets[3]: targets[0] is not a
    # label of any row (the shift leaves it behind).
    assert moved == [False, True, True, True]


def test_the_term_shapes_the_backbone_and_never_the_lm_head(byte_tensor):
    """Untied is the arm's claim: the gradient goes into `h`, not into `lm_head`.

    The AUX term is backwarded on its own: the main CE is not in that backward,
    so an `lm_head` entry would be the aux path reaching it. `None` is the
    strong half — not a zero gradient but no entry at all, because the main
    head's output is not in the aux subgraph.
    """
    model = DormouseModel(tiny_cfg(aux_fb_weight=0.05))
    ids, targets = byte_tensor()
    captured: list[torch.Tensor] = []
    hook = model.norm.register_forward_hook(lambda _m, _args, out: captured.append(out))
    try:
        model(ids, targets)  # one real forward; nothing backwarded yet
    finally:
        hook.remove()
    assert captured, "the head must be fed the readout"
    future_byte_loss(model.fb, captured[0], targets, Counters()).backward()
    assert any(p.grad is not None for p in model.fb.parameters())
    assert any(p.grad is not None for p in model.loop.parameters())
    assert all(p.grad is None for p in model.lm_head.parameters())


def test_the_aux_signals_cannot_coexist_and_the_weight_cannot_be_negative():
    """(c) LOUD: two aux signals in one run, and a negative weight.

    The A/B attributes its verdict to ONE signal (research §4.1: "the A/B has
    one signal per run"), so `aux_fb_weight` with JEPA or DSpark on is refused
    at `validate` — the seam a hand-built config also passes through.
    """
    with pytest.raises(LoudError, match="jepa_weight"):
        validate(ModelConfig(aux_fb_weight=0.05, jepa_weight=0.05))
    with pytest.raises(LoudError, match="dspark_weight"):
        validate(ModelConfig(aux_fb_weight=0.05, dspark_weight=0.1))
    with pytest.raises(LoudError, match="cannot share a run"):
        validate(ModelConfig(aux_fb_weight=0.1, jepa_weight=0.05))
    with pytest.raises(LoudError, match="negative"):
        validate(ModelConfig(aux_fb_weight=-0.1))
    # The ported signal is a REAL field now: it loads (it left `_unsupported`),
    # and each unported one still refuses at its own key.
    cfg = parse_preset("d_model = 64\nn_heads = 4\nhead_dim = 16\naux_fb_weight = 0.1\n")
    assert cfg.aux_fb_weight == 0.1
    with pytest.raises(LoudError, match="not ported"):
        parse_preset("d_model = 64\nn_heads = 4\nhead_dim = 16\njepa_weight = 0.05\n")


def test_the_head_prices_at_2_1_percent_of_small_and_is_the_whole_difference():
    """(d) The exact count ON OUR SHAPES, and the zero-weight identity.

    Our `small` is d_model 768 / vocab 256, so the head is `768*256 + 256 =
    196 864` = 2.14% of the burn `small`'s measured 9 197 390. The research doc
    and AB-PROTOCOL quote **197 120**: that is this arithmetic plus one bias
    (burn's own constant comment writes the same slipped identity), so the
    row's "2.1%" holds either way and this test pins what the tree BUILDS.

    At `aux_fb_weight = 0.0` there is no head at all, which is what keeps
    `small`/`b1`'s measured counts (`test_param_count.py`) and their
    checkpoints unchanged — pure CE stays a valid control without a re-baseline.
    """
    cfg = load_preset("small")
    head = FutureByteHead(cfg.d_model, cfg.vocab)
    assert param_count(head) == FB_PARAMS_AT_SMALL == 768 * 256 + 256 == 196_864
    assert 100 * FB_PARAMS_AT_SMALL / 9_197_390 == pytest.approx(2.14, abs=0.01)
    off = DormouseModel(tiny_cfg())
    on = DormouseModel(tiny_cfg(aux_fb_weight=0.1))
    assert off.fb is None
    assert param_count(on) - param_count(off) == 64 * 256 + 256
    # A readout like `lm_head`: AdamW, never Muon (`train/optim.py`).
    assert on.fb is not None and not is_muon_param("fb.proj.weight", on.fb.proj.weight)
