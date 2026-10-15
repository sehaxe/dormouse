"""The device-side key computation: EXACT parity with the host python form
(`raw_keys`), the CPU/GPU wrap agreement, and the all-torch property (no
.tolist()-style sync in the path)."""

from __future__ import annotations

import torch

from dormouse.model.engram import raw_keys, slot_keys_gpu

ORDERS_CONST = [2, 3, 4]  # the module reads `self.orders`, the port's order table


def test_gpu_keys_equal_host_raw_keys_for_a_fixed_sequence():
    torch.manual_seed(0)
    ids = torch.randint(0, 256, (3, 40))
    got = slot_keys_gpu(ids, ORDERS_CONST)
    host = torch.tensor([raw_keys(row.tolist(), ORDERS_CONST) for row in ids], dtype=torch.int64)
    assert torch.equal(got, host)


def test_gpu_wrap_matches_cpu_bit_for_bit():
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (2, 20))
    cpu = slot_keys_gpu(ids, ORDERS_CONST)
    if torch.cuda.is_available():
        gpu = slot_keys_gpu(ids.cuda(), ORDERS_CONST)
        assert torch.equal(gpu.cpu(), cpu)


def test_zero_byte_window_is_the_offset_basis():
    # p=0, any order: the window is exactly [x0]; the hash must fold one byte.
    ids = torch.tensor([[7]])
    got = slot_keys_gpu(ids, [4])
    expect = raw_keys([7], [4])[0][0]
    assert int(got[0, 0, 0]) == expect
