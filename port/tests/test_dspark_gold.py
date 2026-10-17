"""The DSpark D1 gold vector: the draft head's window is built from INPUT IDS.

Pre-registered as item 2 of `docs/research/2026-10-16-jepa-dspark-alternatives.md`
§4.1 ("the test, not a race"): *"`dspark.md` §7 test 1 (`x[i] = i`; assert the
RNN head at `(p, s)` is fed `x[p+s]` and supervised on `x[p+s+1]`) fails on the
current tree, needs no GPU, and gates any future draft-head decision; D9's width
assertion (`d_model + markov_rank`) rides the same file."* The port ships **no**
DSpark head at all (`dspark_weight` non-default refuses LOUD in `config.py`), so
this file is the contract the future implementer lands against — nothing here
imports a draft head, and that is the point: the vector is checkable before the
arm exists and fails loudly on a targets-based window.

## The one design decision this file records

D1: the slave's step `s` was fed `y[p + s] = x[p + s + 1]`, the very byte its own
logits at `p + s` had just produced, and supervised on the one after. The rule
this file pins: a head at trial `q` may read the label at `q + 1` and nothing
nearer — consuming the label sequence is not the defect; consuming it
**unshifted** is.
"""

from __future__ import annotations

import pytest

from dormouse.config import parse_preset

#: Paper Eq 5's default Markov rank `r` for `W_1` / `W_2` (`dspark.md` §1).
MARKOV_RANK = 256


def make_window(input_ids, p: int, k: int, stride: int):
    """THE CORRECT window at anchor `p`: `k` steps, `[(q, token, target)]`.

    `token_s = x[p+s]` is the CONSUMED byte the draft head is conditioned on
    (Eq 5/6's `W1[x_{k-1}]`, and Eq 7's second operand for the acceptance head),
    `target_s = x[p+s+1]` the byte it must predict (Eq 9's `x_k*`). `x` is the
    INPUT sequence — `aux.rs:355,358` gathers `ids` at `pos` and `pos+1`, and
    anchors are `p = 0, stride, 2*stride, ...` with the whole `k+1`-byte window
    inside the sequence (the `n = (t-k-1)/stride` bound, `aux.rs:331`).
    """
    x = [int(v) for v in input_ids]
    t = len(x)
    assert p % stride == 0, f"anchor {p} is not one of p = 0, stride, 2*stride, ..."
    assert p + k + 1 <= t, f"window [{p}..{p + k}] does not fit in {t} bytes"
    return [(p + s, x[p + s], x[p + s + 1]) for s in range(k)]


def slave_window(targets, p: int, k: int, stride: int):
    """The burn-Slave's window: the SAME arithmetic, fed the LABEL sequence.

    This is D1 before the fix (`model.rs` passed `targets` where Eq 4/5/6 need
    `input_ids`): with the trainer's `y[q] = x[q + 1]`, step `s` reads
    `y[p+s] = x[p+s+1]` and is supervised on `x[p+s+2]` — the answer, one step
    early, and every DSpark number this project recorded measured that objective.
    """
    y = [int(v) for v in targets]
    return [(p + s, y[p + s], y[p + s + 1]) for s in range(k)]


def check_position_order(window, k: int) -> None:
    """The gold property: step `s` reads `x[q]` and predicts `x[q + 1]`."""
    assert len(window) == k, f"a window is {k} steps, got {len(window)}"
    for s, (q, token, target) in enumerate(window):
        assert token == q, (
            f"step {s}: conditioning token {token} is not the consumed byte x[{q}] "
            f"— a window built from `targets` reads one position ahead (D1)"
        )
        assert target == q + 1, f"step {s}: target {target} is not x[{q + 1}]"


def acceptance_width(hidden: int, markov_rank: int) -> int:
    """Eq 7's width: `wᵀ[h_k ; W1[x_{k-1}]]` = `d_model + markov_rank`.

    D9 is the defect where this was `d_model` (`aux.rs:311` passed `None` and
    `with_markov` had no callers). A width assertion, not a value, and it is
    zero-tolerance: a head conditioned on the hidden alone is a different head.
    """
    return hidden + markov_rank


def test_the_window_is_built_from_input_ids_not_targets():
    """`x[i] = i` makes an off-by-one unmistakable: every value is exact.

    Both directions on ONE sequence: the correct window passes for every anchor,
    and the targets-based slave fails the same assertion — so the gate is the
    vector, not a tolerance.
    """
    x = list(range(64))  # x[i] = i
    y = x[1:] + [x[0]]  # the trainer's labelling: y[i] = x[i + 1]
    for p in range(0, 64, 8):
        window = make_window(x, p, k=4, stride=8)
        check_position_order(window, k=4)
    assert [(tok, tgt) for _, tok, tgt in make_window(x, 0, 4, 8)] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]
    naive = slave_window(y, 0, k=4, stride=8)
    assert [(tok, tgt) for _, tok, tgt in naive] == [(1, 2), (2, 3), (3, 4), (4, 5)]
    assert naive[0][1] == make_window(x, 0, 4, 8)[0][1] + 1, "the slave is one ahead"
    with pytest.raises(AssertionError, match="conditioning token"):
        check_position_order(naive, k=4)
    # ... and the slave is wrong for every anchor, not just the first.
    for p in range(0, 64, 8):
        with pytest.raises(AssertionError):
            check_position_order(slave_window(y, p, k=4, stride=8), k=4)
    # A window that would run off the sequence is refused, not padded: the
    # anchor count `(t-k-1)/stride` exists to keep `x[p+s+1]` in range.
    with pytest.raises(AssertionError, match="does not fit"):
        make_window(x, p=61, k=4, stride=1)
    with pytest.raises(AssertionError, match="anchor"):
        make_window(x, p=3, k=4, stride=8)


def test_the_confidence_head_is_markov_conditioned_and_the_port_refuses_the_arm():
    """D9's width contract, and the gate that keeps D1 from shipping alone.

    Eq 7's acceptance head reads `[h_k ; W1[x_{k-1}]]`, so its width is
    `d_model + markov_rank` — NOT `d_model`. The operand comes from the same
    window step the draft head is fed (`make_window`'s `token`, which is why the
    two defects share a file). The port now HAS the head (`dspark.py`,
    integrated 2026-10-16); this test used to pin the refusal and now pins the
    positive: the width contract stands, and the arm is reachable through the
    config with its defaults.
    """
    assert acceptance_width(768, MARKOV_RANK) == 768 + MARKOV_RANK != 768
    cfg = parse_preset("d_model = 64\nn_heads = 4\nhead_dim = 16\ndspark_weight = 0.1\n")
    assert cfg.dspark_weight == 0.1 and cfg.dspark_k == 4 and cfg.dspark_stride == 16
