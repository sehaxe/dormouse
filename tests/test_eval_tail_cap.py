"""m2: the eval tail's 1 MiB cap is COUNTED, and `over N B` counts DISTINCT bytes.

`open_streams` clamps the tail carve to 1 MiB; above it the eval stream wraps
and re-reads the same tail while `_eval_loss` kept adding `x.numel()`, so the
printed figure counted duplicate rows (measured: 100 x 16 x 1024 = 1 638 400 B
requested over a 1 048 576 B range). The fix keeps the cap, prints
`eval_tail_capped=requested/granted` on the eval line, and scores each byte
once. A window at or below the cap must keep its old number and line.
"""

from __future__ import annotations

import torch
from conftest import tiny_run

from dormouse.train import loop as loopmod
from dormouse.train.data import (
    ByteStream,
    distinct_scored_bytes,
    eval_window_bytes,
    open_streams,
)
from dormouse.train.loop import _eval_line, _eval_loss

CAP = 1 << 20


class _ZeroLoss:
    """Duck-typed eval model: any window is a finite zero, so `scored` is the subject."""

    last_breakdown = None

    def __init__(self) -> None:
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def __call__(self, x, y, **kw):
        assert tuple(x.shape) == (16, 1024), x.shape
        return None, torch.zeros(()), None


def big_run(tmp_path):
    path = tmp_path / "big.bin"
    path.write_bytes(bytes(range(256)) * ((2 * CAP) // 256))
    return tiny_run(
        tmp_path,
        data=str(path),
        eval=True,
        eval_batches=100,
        batch=16,
        seq_len=1024,
    )


def brute_distinct(span: int, batch: int, seq: int, batches: int) -> int:
    win = batch * (seq + 1)
    seen = set()
    for k in range(batches):
        base = (k * win) % span
        for row in range(batch):
            start = (base + row * (seq + 1)) % span
            seen.update((start + q) % span for q in range(seq))
    return len(seen)


def test_the_cap_logs_both_sizes_and_scores_distinct_bytes(tmp_path, capsys):
    run = big_run(tmp_path)
    assert eval_window_bytes(run.train) == 1_638_400
    _train, ev = open_streams(run)
    assert ev is not None
    assert ev.eval_tail_capped == (1_638_400, CAP)
    assert ev.hi - ev.lo == CAP, "the granted range is what the stream may read"

    printed = capsys.readouterr().out
    assert "eval_tail_capped=1638400/1048576" in printed, printed

    _loss, scored, _fin, _per = _eval_loss(_ZeroLoss(), ev, 100, None, None)
    distinct = distinct_scored_bytes(ev, 100)
    assert scored == distinct, "the printed figure must be the DISTINCT bytes scored"
    assert scored < 1_638_400, "a wrapped eval must not count duplicate rows"
    assert scored <= CAP


def test_distinct_matches_a_brute_force_union(tmp_path):
    path = tmp_path / "small.bin"
    path.write_bytes(bytes(range(256)) * 16)
    stream = ByteStream(path, 4, 30, lo=0, hi=500)
    batches, total = 10, 10 * 4 * 30
    expected = brute_distinct(500, 4, 30, batches)
    assert distinct_scored_bytes(stream, batches) == expected
    assert expected < total, "the fixture must wrap: 10 x 124 B over a 500 B range"


def test_a_small_window_keeps_its_old_number_and_line(tmp_path, capsys):
    run = tiny_run(tmp_path, eval=True, eval_batches=3)
    _train, ev = open_streams(run)
    assert ev is not None
    assert ev.eval_tail_capped is None
    assert "eval_tail_capped" not in capsys.readouterr().out
    scored = sum(ev.next_batch()[0].numel() for _ in range(run.train.eval_batches))
    assert scored == eval_window_bytes(run.train) == 3 * 2 * 64


def test_the_eval_line_prints_the_counter(tmp_path, monkeypatch):
    run = tiny_run(tmp_path, eval=True, eval_batches=1)
    stream = ByteStream(run.train.data, 2, 64, lo=0, hi=256)
    stream.eval_tail_capped = (1_638_400, CAP)
    monkeypatch.setattr(loopmod, "evaluate", lambda *a, **k: (1.0, CAP, 1.0, []))
    line, _loss, scored = _eval_line(torch.nn.Identity(), stream, run.train, 2, None, 5)
    assert scored == CAP
    assert "over 1048576 B" in line
    assert "eval_tail_capped=1638400/1048576" in line
