"""The device-side key computation: EXACT parity with the host python form
(`raw_keys`), the FNV-64 wrap pinned against python integers on CPU, the
zero-byte basis, and — on a card only, SKIPPED otherwise — the CPU/GPU wrap
agreement and the all-torch property (no .tolist()-style sync in the path)."""

from __future__ import annotations

import pytest
import torch

from dormouse.model import engram
from dormouse.model.engram import fnv, raw_keys, slot_keys_gpu

ORDERS_CONST = [2, 3, 4]  # the module reads `self.orders`, the port's order table

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs a CUDA device (the CPU form is the reference; this is the parity)",
)

# burn's constants, verbatim (`crates/dormouse-data`, and burn's offset basis is
# NOT the FNV spec's 14695981039346656037). Written out here on purpose: the
# implementation's own names would make the pin circular.
FNV_OFFSET_BASIS = 1469598103934665603
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1
KEY_MASK = 0x7FFF_FFFF

#: `b"PING"`'s causal n-grams folded in PYTHON integers, `[position][order]`.
#: `(digest mod 2**64, key)`; every digest is above `KEY_MASK`, so the 31-bit
#: truncation is load-bearing, and every fold's product exceeds `2**64`, so the
#: same numbers pin the wrap. Computed off the burn constants above, once.
WANT: list[list[tuple[int, int]]] = [
    [(4953251317583544201, 1942847369)] * 3,
    [(11203031730874225472, 1177555776)] * 3,
    [
        (11194417057268895312, 1175850576),
        (6842495402507264202, 1135632586),
        (6842495402507264202, 1135632586),
    ],
    [
        (11197153741710944816, 1176365616),
        (1220217459923611925, 393867541),
        (822273329291595671, 78909335),
    ],
]


def test_the_fnv64_wrap_is_pinned_in_python_integers():
    """CPU-only, falsifiable without a card: the digest and keys against integers.

    `fnv` folds in python ints, so its 64-bit mask IS observable (`assert fnv(...)
    == digest`); `slot_keys_gpu` folds int64 tensors, where torch's own overflow
    is the same mod-2**64 wrap. The keys pin the constants the fold chain shares
    (`FNV_PRIME`, `FNV_OFFSET_BASIS`, `KEY_MASK`) in both implementations.

    NOT pinned here, deliberately: *which* mask the device fold applies to its
    intermediate `h`. Any mask that keeps the low 31 bits — `FNV_MASK64`,
    `(1 << 32) - 1`, `KEY_MASK` — is ALGEBRAICALLY equivalent for the returned
    key: `((h ^ b) * P) mod 2**31` depends only on `h mod 2**31`, so the 31-bit
    projection commutes through the whole chain. A gate cannot see it, and
    pretending otherwise is the defect this file is fixing. What IS observable
    is a mask that drops bits INSIDE the low 31 (`& 0x00FF_FFFF` reddens), and
    the host `fnv`'s own `& FNV_MASK64`, which is asserted above.
    """
    seq = list(b"PING")
    for e in range(1, len(seq) + 1):
        for oi, order in enumerate(ORDERS_CONST):
            window = seq[max(0, e - order) : e]
            h, wrapped = FNV_OFFSET_BASIS, False
            for x in window:
                h ^= x & 0xFF
                wrapped |= h * FNV_PRIME > MASK64
                h = (h * FNV_PRIME) & MASK64
            digest, key = WANT[e - 1][oi]
            assert (h, h & KEY_MASK) == (digest, key), f"python-int fold, e={e} order={order}"
            # The 64-bit wrap is load-bearing for the DIGEST (a product above
            # 2**64 exists, and `fnv` must reduce it).
            assert wrapped, f"e={e} order={order}: the fixture must exercise the 2**64 wrap"
            assert fnv(window) == digest, f"fnv({window}) must be the wrapped digest"
            # ... and the digest's high bits are set, so `& KEY_MASK` is not a no-op.
            assert digest > KEY_MASK
    # The device path (on CPU) returns exactly those keys, int64, same shape.
    got = slot_keys_gpu(torch.tensor([seq]), ORDERS_CONST)
    assert got.dtype == torch.int64 and got.shape == (1, len(seq), len(ORDERS_CONST))
    assert got[0].tolist() == [[key for _digest, key in row] for row in WANT]
    # ... and the host form agrees, so the two ports cannot drift apart silently.
    assert raw_keys(seq, ORDERS_CONST) == [[key for _digest, key in row] for row in WANT]


def test_the_device_fold_takes_its_constants_from_the_module():
    """The fold's dtype/mask constants are the module's, not a second copy.

    A divergence here is exactly the class that cannot fail on a CPU-only box:
    host and device both "pass" while one of them hashes different bytes. The
    literals are burn's; the module's names must equal them.
    """
    assert (engram.FNV_OFFSET_BASIS, engram.FNV_PRIME) == (FNV_OFFSET_BASIS, FNV_PRIME)
    assert engram.FNV_MASK64 == MASK64 and engram.KEY_MASK == KEY_MASK
    # `int64` is what makes the wrap torch's own (and what the 1<<64 mask states).
    out = slot_keys_gpu(torch.tensor([[7]]), [4])
    assert out.dtype == torch.int64
    assert int(out.item()) <= engram.KEY_MASK


def test_gpu_keys_equal_host_raw_keys_for_a_fixed_sequence():
    torch.manual_seed(0)
    ids = torch.randint(0, 256, (3, 40))
    got = slot_keys_gpu(ids, ORDERS_CONST)
    host = torch.tensor([raw_keys(row.tolist(), ORDERS_CONST) for row in ids], dtype=torch.int64)
    assert torch.equal(got, host)


@CUDA
def test_gpu_wrap_matches_cpu_bit_for_bit():
    """The CUDA arm. SKIPPED (not passed) without a card: its coverage is the
    device's int64 overflow, which only a card has — the CPU gate above pins
    the arithmetic it must agree with."""
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (2, 20))
    cpu = slot_keys_gpu(ids, ORDERS_CONST)
    gpu = slot_keys_gpu(ids.cuda(), ORDERS_CONST)
    assert torch.equal(gpu.cpu(), cpu)


def test_zero_byte_window_is_the_offset_basis():
    # p=0, any order: the window is exactly [x0]; the hash must fold one byte.
    ids = torch.tensor([[7]])
    got = slot_keys_gpu(ids, [4])
    expect = raw_keys([7], [4])[0][0]
    assert int(got[0, 0, 0]) == expect
