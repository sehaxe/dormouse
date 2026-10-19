"""The counter fast path: a device tensor must never be compared on the host.

`instrument.Counters.count`'s ORDER is load-bearing (the 2026-10-17 audit's
CRITICAL 1): `if n == 0` BEFORE the tensor branch builds a device bool, and
`if <bool tensor>` is a D2H read — every MoE forward paid a sync and
torch.compile's graphs broke, while the module's own docstring claimed the path
was sync-free. These gates pin the order.
"""

from __future__ import annotations

import json
import warnings

import pytest
import torch

from dormouse.instrument import Counters

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (sync-debug mode)")


def test_a_device_tensor_routes_to_count_device_not_a_comparison():
    """The tensor branch runs FIRST: the value stays on the device lane and
    only `drain()` reads it back (once, on the log cadence)."""
    c = Counters()
    c.count("x", torch.tensor(7))
    assert "x" not in c, "a device counter must not land on the host side"
    assert "x" in c._device
    c.drain()
    assert c["x"] == 7


def test_a_zero_tensor_drains_to_nothing_and_an_int_zero_returns_early():
    c = Counters()
    c.count("z", torch.tensor(0))
    c.drain()
    assert "z" not in c  # a zero device counter is not an entry
    c.count("y", 0)
    assert "y" not in c and not c._device


def test_a_positive_int_still_accumulates():
    c = Counters()
    c.count("n", 3)
    c.count("n")
    assert c["n"] == 4


@CUDA
def test_count_does_not_synchronize_a_cuda_tensor():
    """The real gate: PyTorch's sync debug mode warns on every D2H read, so
    the OLD ordering fails this test and the new one passes."""
    c = Counters()
    t = torch.ones(3, device="cuda")
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c.count("z", t)
            msgs = [str(x.message) for x in w]
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert not any("synchroniz" in m.lower() for m in msgs), msgs


# ---------------------------------------------------------------------------
# `why` (2026-10-19 audit M2): the reason rode an attribute the log never read
# ---------------------------------------------------------------------------


def test_why_reaches_the_line_and_the_json_row():
    """Eleven call sites set `counters.why`; the reader must see it in both."""
    c = Counters()
    c.count("moe_triton_fallback")
    c.why = "triton: top-k 64 > 32"
    assert c.line() == "moe_triton_fallback=1 fallback: triton: top-k 64 > 32"
    row = dict(c)  # exactly what train/loop.py hands the JSON sink
    assert row["why"] == "triton: top-k 64 > 32"
    assert json.loads(json.dumps(row))["why"] == "triton: top-k 64 > 32"
    # Empty is absent, not an empty field: a parser sees the same row shape.
    c.why = ""
    assert c.line() == "moe_triton_fallback=1"
    assert "why" not in dict(c)


def test_why_is_read_by_getattr_and_survives_a_merge():
    a, b = Counters(), Counters()
    a.why = "capacity=0: dropless"
    assert a.why == "capacity=0: dropless"
    b.why = "capacity=0: dropless"  # the same reason every step
    a.merge(b)
    assert a.why == "capacity=0: dropless", "a repeat must not grow the line"
    b.why = "triton: input is on cpu"
    a.merge(b)
    assert a.why == "capacity=0: dropless; triton: input is on cpu"


def test_a_long_counter_name_is_printed_whole():
    """No silent truncation to a 24-character parser column."""
    name = "a_counter_name_that_is_way_longer_than_24_chars"
    c = Counters()
    c.count(name)
    c.why = "x"
    assert f"{name}=1" in c.line()
    assert c.line().endswith("fallback: x")
