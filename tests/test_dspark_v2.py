"""The DSpark port's own gate: D1's window, D9's Markov wiring, and the loss.

`tests/test_dspark_gold.py` is the pre-registered vector
(`docs/research/2026-10-16-jepa-dspark-alternatives.md` §4.1 item 2) and this
file runs THAT vector through the real builder — the gold file's `make_window`
and `check_position_order` are imported, not mirrored, so the two cannot drift.
The remaining tests are the three burn defects' other halves (D9's width and its
mechanism, the detached `c*`, the kink-free BCE), the counters, and the two ends
of the length budget.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg, tiny_model
from test_dspark_gold import (
    MARKOV_RANK,
    acceptance_width,
    check_position_order,
    make_window,
    slave_window,
)

from dormouse.instrument import Counters
from dormouse.model.dspark import (
    DSparkHead,
    accept_rate_target,
    anchor_count,
    dspark_aux_loss,
    stable_bce,
)

SEQ = 64


def _head(**over) -> DSparkHead:
    return DSparkHead(tiny_cfg(), "cpu", markov_rank=16, **over)


def _ids(batch: int = 2, seq: int = SEQ) -> torch.Tensor:
    return (torch.arange(batch * seq).reshape(batch, seq) * 7) % 251


def _batch(cfg, batch: int = 2, seq: int = SEQ):
    torch.manual_seed(11)
    return (
        torch.randn(batch, seq, cfg.d_model),
        _ids(batch, seq),
        torch.randn(batch, seq, cfg.vocab),
    )


def test_d1_the_gold_vector_holds_through_this_builder():
    """(a) `x[i] = i` through the TENSOR builder, against the gold vector.

    The gold file's own loop enumerates every anchor that fits (`range(0, 64, 8)`,
    0…56); burn's `n = floor((t-k-1)/stride)` (`aux.rs:331`) counts 7 (0…48) and
    never builds the last one, so this builder's windows are a SUBSET of the gold
    loop's — and every one of them must match `make_window` exactly and pass
    `check_position_order`. Both directions on the same sequence: the targets-fed
    build (`slave_window`) is a different window and FAILS the same assertion,
    which is D1 before the fix.
    """
    head = _head()
    x = list(range(SEQ))
    ids = torch.tensor(x, dtype=torch.long).reshape(1, SEQ)
    win = head.window(ids, k=4, stride=8)
    assert win.positions.tolist() == [0, 8, 16, 24, 32, 40, 48]
    y = x[1:] + [x[0]]  # the trainer's labelling: y[q] = x[q + 1]
    for i, p in enumerate(win.positions.tolist()):
        mine = [
            (p + s, int(win.token_ids[0, i, s]), int(win.target_ids[0, i, s])) for s in range(4)
        ]
        assert mine == make_window(x, p, k=4, stride=8), f"anchor {p} (D1)"
        check_position_order(mine, k=4)  # THE gold property, through the builder
        naive = slave_window(y, p, k=4, stride=8)
        assert mine != naive, "the label-fed window is a different window"
        with pytest.raises(AssertionError, match="conditioning token"):
            check_position_order(naive, k=4)
    # A non-identity sequence, batch 2: the builder reads the ids it was handed,
    # in both rows — nothing here is `position == value`.
    xr = [(i * 37) % 251 for i in range(SEQ)]
    idsr = torch.tensor([xr, xr[::-1]], dtype=torch.long)
    winr = head.window(idsr, k=4, stride=8)
    for i, p in enumerate(winr.positions.tolist()):
        for row, seq in enumerate((xr, xr[::-1])):
            mine = [
                (p + s, int(winr.token_ids[row, i, s]), int(winr.target_ids[row, i, s]))
                for s in range(4)
            ]
            assert mine == make_window(seq, p, k=4, stride=8)
    # The interface half of the fix: no label sequence is reachable from the term.
    assert "targets" not in inspect.signature(DSparkHead.aux_loss).parameters
    assert "targets" not in inspect.signature(dspark_aux_loss).parameters


def test_d9_the_acceptance_head_is_markov_conditioned():
    """(b) Eq 7's width `d_model + markov_rank`, and the mechanism behind it.

    The width is the gold file's `acceptance_width` (D9 was `d_model`); the
    mechanism is stronger than the width, so it is asserted too: with the hidden
    state held fixed, changing the previous token must move the acceptance logit
    (`burn-dspark`'s own `accept_rate_predictor_reads_the_previous_token`), and
    zeroing exactly the Markov block of the projection must collapse that
    movement to zero — which is what says the movement came from `W1[x]`.
    """
    cfg = tiny_cfg()
    head = DSparkHead(cfg, "cpu")
    assert head.conf.weight.shape == (1, acceptance_width(cfg.d_model, MARKOV_RANK))
    assert head.conf.in_features == cfg.d_model + MARKOV_RANK != cfg.d_model
    assert head.conf.bias is None, "Eq 7's projection is bias-free"
    hidden, _, logits = _batch(cfg, batch=1, seq=32)
    a = head.terms(hidden, _ids(1, 32), logits, k=4, stride=4).conf
    b = head.terms(hidden, (_ids(1, 32) + 7) % 251, logits, k=4, stride=4).conf
    assert a.shape == (1, 6, 4), tuple(a.shape)
    assert (a - b).abs().max() > 1e-6, "the acceptance head ignores W1[x_{k-1}] (D9)"
    with torch.no_grad():
        head.conf.weight[:, cfg.d_model :] = 0.0
    a2 = head.terms(hidden, _ids(1, 32), logits, k=4, stride=4).conf
    b2 = head.terms(hidden, (_ids(1, 32) + 7) % 251, logits, k=4, stride=4).conf
    assert (a2 - b2).abs().max() == 0.0, "the movement was not the Markov block"


def test_the_acceptance_target_is_detached_and_carries_no_gradient():
    """(c) `c*` is a LABEL: `.detach()` at the loss (`lib.rs:290`), asserted both ways.

    Forward value is bit-identical either way, so a gradient fixture is the only
    instrument: `∂BCE(conf, c*)/∂draft` must be None (the detached target), while
    the CE term and a deliberately LIVE `c*` both produce a non-zero gradient —
    which is what proves the assertion can fail.
    """
    cfg = tiny_cfg()
    head = _head()
    hidden, ids, logits = _batch(cfg)
    hidden.requires_grad_(True)
    terms = head.terms(hidden, ids, logits, k=4, stride=8)
    assert terms.c_star.grad_fn is None and not terms.c_star.requires_grad
    conf_term = stable_bce(terms.conf, terms.c_star).mean()
    assert torch.autograd.grad(conf_term, terms.draft, allow_unused=True)[0] is None
    ce = F.cross_entropy(terms.draft.reshape(-1, cfg.vocab), terms.target_ids.reshape(-1))
    assert torch.autograd.grad(ce, terms.draft)[0].abs().sum() > 0.0
    live = accept_rate_target(terms.draft, terms.target)
    g_live = torch.autograd.grad(stable_bce(terms.conf, live).mean(), terms.draft)[0]
    assert g_live.abs().sum() > 0.0, "the fixture cannot see a live acceptance target"


def _naive_bce(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """The usual spelling: `-(c·log σ(z) + (1-c)·log(1-σ(z)))`."""
    sig = torch.sigmoid(z)
    return -(c * sig.log() + (1.0 - c) * (1.0 - sig).log())


def test_stable_bce_is_finite_at_z_pm_30_and_matches_the_naive_form():
    """(d) `(1-c)·z + softplus(-z)`: finite where the naive form is not, equal where it is.

    `lib.rs:262-284`. In fp32 `sigmoid(30) == 1.0` exactly, so the naive form
    evaluates `log(0)`: the NaN/inf that the training log would report as a data
    problem. The stable form returns `softplus(∓30)` there — 30.0 on the wrong
    side, ~0 on the right one.
    """
    z = torch.tensor([30.0, -30.0, 30.0, -30.0])
    c = torch.tensor([0.0, 1.0, 1.0, 0.0])
    got = stable_bce(z, c)
    assert torch.isfinite(got).all(), f"stable BCE is not finite at |z| = 30: {got}"
    assert torch.allclose(got, torch.tensor([30.0, 30.0, 0.0, 0.0]), atol=1e-4)
    assert not torch.isfinite(_naive_bce(z, c)).all(), "the naive form did not overflow"
    # Moderate z, float64: the two forms are the same number.
    zz = torch.linspace(-10.0, 10.0, 41, dtype=torch.float64)
    cc = (torch.arange(41, dtype=torch.float64) / 40.0).clamp(0.0, 1.0)
    assert (stable_bce(zz, cc) - _naive_bce(zz, cc)).abs().max() < 1e-6
    # ... and the derivative is `sigmoid(z) - c` identically: no kink at z = 0,
    # which is the second half of the `lib.rs:262-284` fix.
    z.grad = None
    (zz.requires_grad_(True))
    grad = torch.autograd.grad(stable_bce(zz, cc).sum(), zz)[0]
    assert (grad - (torch.sigmoid(zz) - cc)).abs().max() < 1e-9


def test_forward_shapes_and_the_window_counters():
    """(e) Shapes, the one scalar, and `dspark=<windows>/<calls>`."""
    cfg = tiny_cfg()
    head = _head()
    hidden, ids, logits = _batch(cfg)
    hidden.requires_grad_(True)
    counters = Counters()
    loss = head.aux_loss(hidden, ids, logits, k=4, stride=8, counters=counters)
    assert loss.shape == () and torch.isfinite(loss) and float(loss.detach()) > 0.0
    terms = head.terms(hidden, ids, logits, k=4, stride=8)
    assert terms.draft.shape == terms.target.shape == (2, 7, 4, cfg.vocab)
    assert terms.target_ids.shape == terms.conf.shape == terms.c_star.shape == (2, 7, 4)
    assert counters["dspark"] == 2 * 7 and counters["dspark_asked"] == 1
    # `stride` is the anchor DENSITY, not a fraction: 4 doubles the anchors.
    head.aux_loss(hidden, ids, logits, k=4, stride=4, counters=counters)
    assert counters["dspark"] == 14 + 2 * 14 and counters["dspark_asked"] == 2
    # The module-level function IS the method, and the gradient reaches the head.
    same = dspark_aux_loss(head, hidden, ids, logits, k=4, stride=8)
    assert torch.allclose(same, loss)
    same.backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0.0
    assert head.w1.weight.grad is not None and head.conf.weight.grad is not None


def test_thirty_steps_of_head_only_training_decrease_the_loss():
    """(f) The arm trains: 30 Adam steps on the tiny model's own state.

    Head-only on purpose (the backbone's `hidden` and `logits` are computed under
    `no_grad`), so this is the head's own fit — the burn defect class is an arm
    that runs and trains nothing.
    """
    torch.manual_seed(7)
    model = tiny_model()
    cfg = model.cfg
    ids = _ids()
    with torch.no_grad():
        logits, loss, _ = model(ids)
        assert loss is None, "a label-free forward must not report a loss"
        frozen = logits.reshape(2, SEQ, cfg.vocab)
        hidden = model.embed(ids)
    head = _head()
    opt = torch.optim.Adam(head.parameters(), lr=0.02)
    first, last = None, None
    for _ in range(30):
        term = head.aux_loss(hidden, ids, frozen, k=4, stride=8)
        opt.zero_grad()
        term.backward()
        opt.step()
        value = float(term.detach())
        first = value if first is None else first
        last = value
    assert last < first, f"the head did not train: {first} -> {last}"
    assert head.w2.weight.grad is not None and head.w2.weight.grad.abs().sum() > 0.0


def test_a_sequence_shorter_than_the_first_anchor_is_zero_and_counted():
    """(g) The 0-window arm: an exact zero, COUNTED, and no NaN anywhere.

    The first anchor needs `t >= k + 1 + stride` (`aux.rs:331-336`), so `t = 4`
    at `k = 4` has no window at all. Burn's silent fallback here is the ADR-0019
    row this counter answers: the term is 0 AND the log line says `dspark=0/<n>`.
    """
    cfg = tiny_cfg()
    head = DSparkHead(cfg, "cpu", markov_rank=16, k=4, stride=8)
    hidden, ids, logits = _batch(cfg, batch=1, seq=4)
    counters = Counters()
    zero = head.aux_loss(hidden, ids, logits, counters=counters)
    assert float(zero.detach()) == 0.0 and torch.isfinite(zero).all()
    assert counters["dspark_asked"] == 1 and counters.get("dspark", 0) == 0
    assert head.window(ids).positions.numel() == 0
    assert anchor_count(12, 4, 8) == 0, "the first anchor needs t >= k + 1 + stride = 13"
    assert anchor_count(13, 4, 8) == 1
    # The same forward with a window that fits: no NaN there either, and the
    # difference is exactly the counter.
    hidden, ids, logits = _batch(cfg, batch=1, seq=13)
    term = head.aux_loss(hidden, ids, logits, counters=counters)
    assert torch.isfinite(term).all() and float(term.detach()) > 0.0
    assert counters["dspark_asked"] == 2 and counters["dspark"] == 1
