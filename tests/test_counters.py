"""The counter fast path: a device tensor must never be compared on the host.

`instrument.Counters.count`'s ORDER is load-bearing (the 2026-10-17 audit's
CRITICAL 1): `if n == 0` BEFORE the tensor branch builds a device bool, and
`if <bool tensor>` is a D2H read — every MoE forward paid a sync and
torch.compile's graphs broke, while the module's own docstring claimed the path
was sync-free. These gates pin the order.
"""

from __future__ import annotations

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
