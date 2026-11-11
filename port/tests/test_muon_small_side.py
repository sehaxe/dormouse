"""The Newton–Schulz chain contracts on the SMALL side: re-associated, not changed.

The live recipe's Muon pass spends ~105 ms of a 377 ms step in this chain. For a
tall `[n, f]` the iteration is the same one whether it builds `x xᵀ` (`n × n`) or
`xᵀx` (`f × f`); only the second runs the square operand on the small side. This
file gates the change on three claims, each with a way to fail:

1. **the VALUES did not move** — the reoriented chain agrees with the pre-change
   chain (transcribed below as `_reference`, the definition the port shipped) to
   `1e-5` relative in fp32 and `1e-2` in bf16, on the four shapes the live model
   actually has plus a non-square tall/short pair. Measured on CPU: fp32
   `6.6e-9`, bf16 `2.6e-4` — a re-association, not a formula change.
2. **the shape of the work moved** — the `mm`/`bmm` OUTPUTS are dispatched and
   the gated one is `[f, f]`; a revert to `x xᵀ` on a tall matrix puts
   `[n, n]` back and the test fails on the mechanism, not only on a tolerance.
3. **the shapes that were already small are untouched** — `n <= f` is
   bit-identical, because that arm is the old code path.

`ns_dtype`, `steps` and the coefficients are the caller's; this file moves none
of them.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from dormouse.train.optim import JORDAN_NS_COEFFS, _classic_ns_many, zeropower_via_ns

STEPS = 5

#: The live model's four weight shapes (`configs/arch_gated_swiglu.toml`, b16):
#: two tall, one square, one short — the tall ones are the ones that paid for an
#: `n × n` Gram before this change.
LIVE_SHAPES = [(2048, 832), (2496, 832), (832, 832), (832, 2048)]

#: Non-square pair, neither side a live width: catches an orientation rule that
#: only holds when one dimension is 832.
PAIR = [(1000, 257), (257, 1000)]

#: The two precision bars: fp32 is a re-association and nothing else; bf16 walks
#: 2.9e-3 absolute on the tall shapes and stays inside 1e-2.
BARS = [(torch.float32, "fp32", 1e-5), (torch.bfloat16, "bf16", 1e-2)]


def _reference(g: torch.Tensor, steps: int = STEPS, ns_dtype: str = "fp32") -> torch.Tensor:
    """The pre-change chain, verbatim — the reference, not `main`'s file.

    Tall tensors are transposed into `[f, n]` and the Gram is `x xᵀ` there; the
    current code reaches the same `[f, f]` by `xᵀx` where the tensor is stored.
    """
    a, b, c = JORDAN_NS_COEFFS
    dt = torch.bfloat16 if ns_dtype == "bf16" else torch.float32
    if g.ndim == 2:
        x = g.to(torch.float32)
        x = x / (x.norm() + 1e-7)
        transposed = x.shape[0] > x.shape[1]
        if transposed:
            x = x.T.contiguous()
        x = x.to(dt)
        for _ in range(steps):
            gram = x @ x.T
            poly = b * gram + c * (gram @ gram)
            x = a * x + poly @ x
        out = x.T.contiguous() if transposed else x
        return out.to(torch.float32)
    x = g.to(torch.float32)
    x = x / (x.norm(dim=(1, 2), keepdim=True) + 1e-7)
    transposed = x.shape[1] > x.shape[2]
    if transposed:
        x = x.transpose(1, 2).contiguous()
    x = x.to(dt)
    for _ in range(steps):
        gram = torch.bmm(x, x.transpose(1, 2))
        poly = b * gram + c * torch.bmm(gram, gram)
        x = a * x + torch.bmm(poly, x)
    out = x.transpose(1, 2).contiguous() if transposed else x
    return out.to(torch.float32)


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got - ref).abs().max()) / float(ref.norm())


def _fixture(shape: tuple[int, ...], dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)).mul(1e-3).to(dtype)


@pytest.mark.parametrize("shape", LIVE_SHAPES + PAIR)
@pytest.mark.parametrize("dtype,ns,bar", BARS)
def test_the_per_matrix_chain_is_the_same_iteration_reassociated(shape, dtype, ns, bar):
    x = _fixture(shape, dtype)
    ref = _reference(x, STEPS, ns)
    got = zeropower_via_ns(x, STEPS, "classic", ns)
    assert got.shape == x.shape and got.dtype == torch.float32
    rel = _rel(got, ref)
    assert rel <= bar, f"{ns} {shape}: rel {rel:.3e} > {bar:.0e}"


@pytest.mark.parametrize("shape", LIVE_SHAPES)
@pytest.mark.parametrize("dtype,ns,bar", BARS)
def test_the_lockstep_batch_is_the_per_matrix_chain(shape, dtype, ns, bar):
    """The live path is `_classic_ns_many` (a shape group stacked into 3-D).

    Same bar as the per-matrix gate: `bmm` and `mm` are different kernels, so
    the two forms are not required to be bit-identical — but the ORIENTATION
    cannot differ between them (a per-slice transpose would be far outside).
    """
    group = _fixture((2,) + shape, dtype)
    ref = torch.stack([zeropower_via_ns(group[i], 2, "classic", ns) for i in range(2)])
    got = _classic_ns_many([group], 2, ns)[0]
    assert got.shape == group.shape
    rel = _rel(got, ref)
    assert rel <= bar, f"{ns} 3-D {shape}: rel {rel:.3e} > {bar:.0e}"


def test_the_small_side_is_bit_identical():
    """`n <= f` keeps `x xᵀ` and the polynomial on the left: the old bytes."""
    x = _fixture((832, 2048), torch.float32)
    assert torch.equal(zeropower_via_ns(x, STEPS, "classic"), _reference(x, STEPS))


class _MMShapes(TorchDispatchMode):
    """Record the shape every `mm`/`bmm` PRODUCES."""

    def __init__(self) -> None:
        self.out: list[tuple[int, ...]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if func in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default):
            self.out.append(tuple(out.shape))
        return out


@pytest.mark.parametrize("shape,f", [((2048, 832), 832), ((2496, 832), 832), ((1000, 257), 257)])
def test_the_gram_is_the_square_matrix_of_the_small_side(shape, f):
    """The mechanism: a `[f, f]` Gram, never the `[n, n]` one, on a tall input."""
    x = _fixture(shape, torch.float32)
    mode = _MMShapes()
    with mode:
        zeropower_via_ns(x, 1, "classic")
    assert (f, f) in mode.out, mode.out
    assert (shape[0], shape[0]) not in mode.out, f"{shape} built the long-side Gram: {mode.out}"


def test_a_mixed_tall_and_short_list_moves_no_more_than_a_single_one():
    """One lockstep batch over BOTH orientations: values still per matrix."""
    xs = [_fixture(s, torch.float32) for s in ((2496, 832), (832, 2048), (1000, 257))]
    refs = [zeropower_via_ns(x, 2, "classic") for x in xs]
    for got, ref in zip(_classic_ns_many(xs, 2), refs, strict=True):
        assert torch.equal(got, ref)
