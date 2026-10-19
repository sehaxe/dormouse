"""Gates for quantized saved-tensor checkpointing (`model/act_ckpt.py`).

What is claimed here, and what is deliberately NOT:

1. **the codec**: round-trip bounds on a Gaussian and a heavy-tailed fixture,
   EXACT on the formats' own grid, byte counts equal to `mxfp.py`'s own
   accounting, and a padded partial tail block (`a`-`d`);
2. **the gate**: a tensor below `min_numel` is the SAME object back (identity,
   no re-quantization), and `kind="none"` is bit-identical to no hooks at all —
   while still firing the hook, which is the point of installing it (`e`, `f`);
3. **the model**: a real tiny-preset forward+backward under the hooks — the
   LOSS is bit-identical (`torch.equal`: the hooks cannot touch the forward),
   every gradient is finite, and the worst relative gradient move is inside the
   stated bound, with the measured numbers printed (`g`);
4. **the traps**: nesting (innermost pair wins), sequential reuse, the
   checkpoint placement (hooks outside a checkpointed region capture only its
   inputs and the recompute stays; hooks inside capture the region's saves and
   the recompute disappears), LOUD on a bad kind (`h`-`k`).

**VRAM is NOT claimed here.** CPU cannot show a device peak, and the card
belongs to another job; the memory claim is the COUNTER arithmetic (bytes in,
bytes out), which is device-independent and exact. The device leg of the hook
contract is pinned by a CUDA test that skips without a card.

The numbers in the assertions are MEASURED (torch 2.13.0+cu129 CPU, seed 0,
`uv run python scripts/bench_act_ckpt.py` reproduces them); the bounds are set
loose on purpose — a gate that fails for a reason unrelated to what it names is
the same defect as a gate that cannot fail (AGENTS §2.2).
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import tiny_cfg
from torch.utils.checkpoint import checkpoint

from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel
from dormouse.model.act_ckpt import (
    KINDS,
    Packed,
    QuantSavedHooks,
    memory_report,
    pack_tensor,
    unpack_tensor,
)
from dormouse.model.mxfp import bytes_per_elements_mxfp4, bytes_per_elements_mxfp8

#: The tiny preset's own batch (`conftest.BATCH`): 116 of the 383 saves clear the
#: default 4096-element gate, so the arm is real and the run stays CI-cheap. The
#: save COUNT is batch-independent; a bigger batch only moves more of them over
#: the gate (batch 8 packs 154).
BATCH, SEQ = 2, 64

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs a CUDA device (the hook contract's device leg)",
)


def _gauss(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def _heavy_tail(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    """A log-normal magnitude: the fixture a power-of-two 4-bit grid is worst at."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g) * torch.exp(2.0 * torch.randn(shape, generator=g))


def _rel_err(t: torch.Tensor, dq: torch.Tensor) -> float:
    """`max|t − t'| / absmax(t)` — the bound's own shape (scale-free)."""
    return float((t - dq).abs().max()) / float(t.abs().max())


def _ids(batch: int = BATCH) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.arange(batch * SEQ, dtype=torch.long).view(batch, SEQ) % 251
    return ids, (ids + 1) % 251


def _run(kind: str | None, *, batch: int = BATCH, min_numel: int = 4096):
    """One tiny-preset forward+backward; `kind=None` means no hooks at all.

    A fresh model per call, seeded identically, so two calls share every
    parameter bit-for-bit and any difference is the hooks' (the `_switch`
    pattern `tests/test_offload_experts.py` uses for the same reason).
    """
    torch.manual_seed(0)
    model = DormouseModel(tiny_cfg())
    ids, targets = _ids(batch)
    counters = Counters()
    if kind is None:
        _, loss, _ = model(ids, targets)
    else:
        with QuantSavedHooks(kind, min_numel=min_numel, counters=counters):
            _, loss, _ = model(ids, targets)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    assert grads, "the tiny model must produce gradients"
    return loss.detach(), grads, counters


def _worst_rel(ref: dict[str, torch.Tensor], got: dict[str, torch.Tensor]) -> tuple[float, str]:
    worst, where = 0.0, ""
    for name, g in ref.items():
        rel = float((g - got[name]).norm()) / max(float(g.norm()), 1e-12)
        if rel > worst:
            worst, where = rel, name
    return worst, where


# --- the codec ---------------------------------------------------------------


def test_a_roundtrip_bound_on_randn():
    """(a) Gaussian: `max|t − t'| / absmax` inside the pinned bound, and nonzero."""
    t = _gauss((64, 128))
    for kind, bound in (("mxfp4", 0.13), ("mxfp8", 0.04)):
        dq = unpack_tensor(pack_tensor(t, kind))
        err = _rel_err(t, dq)
        print(f"[act_ckpt bound] randn {kind}: {err:.4e}")
        assert 0.0 < err <= bound, f"{kind}: {err} outside (0, {bound}]"


def test_b_roundtrip_bound_on_heavy_tail():
    """(b) The heavy-tailed fixture: same bound shape, measured not assumed.

    The zero fraction is the arm's real weak spot on this fixture and the
    module docstring's 75% is this assertion: one huge block amax shrinks the
    max-error ratio while the small elements under that scale fall below the
    grid's first step.
    """
    t = _heavy_tail((64, 128))
    zeros: dict[str, float] = {}
    for kind, bound in (("mxfp4", 0.06), ("mxfp8", 0.03)):
        dq = unpack_tensor(pack_tensor(t, kind))
        err = _rel_err(t, dq)
        zeros[kind] = float((dq == 0).float().mean())
        print(f"[act_ckpt bound] heavy {kind}: max/absmax={err:.4e} zeros={zeros[kind]:.3f}")
        assert 0.0 < err <= bound, f"{kind}: {err} outside (0, {bound}]"
    assert zeros["mxfp4"] >= 0.5, zeros  # the docstring's 75%
    assert zeros["mxfp8"] <= 0.01, zeros


def test_c_the_codec_is_exact_on_the_format_grid():
    """(c) Grid values under a per-block power-of-two scale come back exactly.

    This is the V2 claim in miniature (module docstring): a value already on the
    format's grid costs nothing to pack. Two blocks with different amaxes pin
    that the e8m0 scale does not leak across blocks.
    """
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    block = torch.cat([grid, -grid]).repeat(2)  # 32 values
    t = torch.cat([block, block * 2.0**-3, block * 2.0**5, -block]).reshape(4, 32)
    for kind in ("mxfp4", "mxfp8"):
        dq = unpack_tensor(pack_tensor(t, kind))
        assert torch.equal(dq, t), f"{kind} must be exact on its own grid"


def test_d_pack_shrinks_bytes_by_the_codecs_own_accounting():
    """(d) 3.76x / 1.94x against bf16, with the byte counts as exact literals."""
    n = 8192
    t = _gauss((n,)).to(torch.bfloat16)
    p4 = pack_tensor(t, "mxfp4")
    assert p4.nbytes() == bytes_per_elements_mxfp4(n) == 4352
    assert t.nbytes / p4.nbytes() == pytest.approx(3.7647, rel=1e-3)
    p8 = pack_tensor(t, "mxfp8")
    assert p8.nbytes() == bytes_per_elements_mxfp8(n) == 8448
    assert t.nbytes / p8.nbytes() == pytest.approx(1.9394, rel=1e-3)
    assert unpack_tensor(p8).dtype is torch.bfloat16


def test_d_a_partial_tail_block_is_padded_storage():
    """A numel that is not a multiple of 32: one block of pad, sliced back off."""
    t = _gauss((100,))
    p = pack_tensor(t, "mxfp4")
    assert (p.numel, tuple(p.shape)) == (100, (100,))
    assert p.nbytes() == bytes_per_elements_mxfp4(100) == 68  # 4 blocks x 17 B
    dq = unpack_tensor(p)
    assert dq.shape == t.shape and dq.dtype is t.dtype and dq.device == t.device
    assert 0.0 < _rel_err(t, dq) <= 0.13


# --- the gate ----------------------------------------------------------------


def test_e_passthrough_is_the_identity_at_the_min_numel_boundary():
    """(e) Below the gate: the SAME object back, and a real run stays bit-exact."""
    h = QuantSavedHooks("mxfp4", min_numel=64)
    at, below = _gauss((64,)), _gauss((63,))
    assert isinstance(h._pack(at), Packed)
    assert h._pack(below) is below  # identity, not a copy, not re-quantized
    assert h._unpack(below) is below
    assert (h.counters["act_ckpt_packed"], h.counters["act_ckpt_passthrough"]) == (1, 1)
    empty = torch.empty(0)
    assert h._pack(empty) is empty  # a 0-length save has no block to quantize

    # ... and on a real graph, everything under the gate: grads bit-identical.
    x = _gauss((8, 8)).requires_grad_(True)
    ref = x.detach().clone().requires_grad_(True)
    torch.sin(ref).sum().backward()  # sin saves its input: one passthrough
    c = Counters()
    with QuantSavedHooks("mxfp4", min_numel=10**9, counters=c):
        y = torch.sin(x).sum()
    y.backward()
    assert c["act_ckpt_passthrough"] == 1 and "act_ckpt_packed" not in c
    assert torch.equal(x.grad, ref.grad)


def test_f_kind_none_installs_the_hooks_and_packs_nothing():
    """(f) `none` is a real no-op context: the hook fires, nothing is counted."""
    h = QuantSavedHooks("none", min_numel=0)
    seen: list[object] = []
    inner = h._pack

    def spy(t: object) -> object:
        seen.append(t)
        return inner(t)

    h._pack = spy  # type: ignore[method-assign]
    x = _gauss((32,)).requires_grad_(True)
    with h:
        y = torch.sin(x).sum()  # sin saves its input: exactly one save
    y.backward()
    assert len(seen) == 1 and isinstance(seen[0], torch.Tensor), "the pack hook must fire"
    assert h._unpack(seen[0]) is seen[0]
    assert h.counters == {}, h.counters.line()
    assert torch.isfinite(x.grad).all()


def test_g_kind_none_is_bit_identical_to_no_hooks_on_the_tiny_model():
    """(g) The task's exactness gate: `torch.equal` on a named grad, and the loss."""
    loss_off, grads_off, _ = _run(None)
    loss_none, grads_none, c = _run("none")
    assert torch.equal(loss_off, loss_none), (float(loss_off), float(loss_none))
    name = "loop.mem_dense.weight"
    assert torch.equal(grads_off[name], grads_none[name])
    for n in grads_off:
        assert torch.equal(grads_off[n], grads_none[n]), n
    assert c == {}, c.line()


# --- the model under the hooks ----------------------------------------------


def test_h_the_tiny_model_loss_is_bit_identical_and_grads_are_bounded():
    """(h) The claim the trainer wiring rests on, with the measured numbers.

    The forward is untouched (loss `torch.equal`); the backward reads
    re-quantized saves, so every grad is finite and the worst relative move is
    bounded. The bound is stated, printed, and the 8-bit arm must beat the
    4-bit one — the ordering survives any model change, the bound does not.
    """
    loss_off, grads_off, _ = _run(None)
    loss_4, grads_4, c4 = _run("mxfp4")
    loss_8, grads_8, c8 = _run("mxfp8")
    assert torch.equal(loss_off, loss_4) and torch.equal(loss_off, loss_8)
    for tag, g in (("mxfp4", grads_4), ("mxfp8", grads_8)):
        finite = all(bool(torch.isfinite(v).all()) for v in g.values())
        assert finite, f"{tag} produced a non-finite grad"
    worst4, where4 = _worst_rel(grads_off, grads_4)
    worst8, where8 = _worst_rel(grads_off, grads_8)
    print(f"[act_ckpt grads] mxfp4 worst={worst4:.4e} @{where4}")
    print(f"[act_ckpt grads] mxfp8 worst={worst8:.4e} @{where8}")
    assert worst4 <= 0.6, f"{where4}: {worst4}"
    assert worst8 <= 0.12, f"{where8}: {worst8}"
    assert worst8 < worst4, "8 bits must move the grads less than 4"

    # The accounting, from the same run: the codec's ratios against bf16.
    r4, r8 = memory_report(c4), memory_report(c8)
    assert c4["act_ckpt_packed"] > 0 and c4["act_ckpt_passthrough"] > 0
    assert r4["packed_bytes"] == c4["act_ckpt_packed_bytes"] > 0
    assert 3.5 <= r4["ratio_bf16"] <= 3.8, r4
    assert 1.8 <= r8["ratio_bf16"] <= 1.95, r8
    print(f"[act_ckpt accounting] mxfp4 {r4}")


def test_i_memory_report_arithmetic_and_the_empty_case():
    """The report's ratios are the counters' ratios, and NaN when nothing packed."""
    c = Counters()
    c.count("act_ckpt_packed_bytes", 1000)
    c.count("act_ckpt_orig_bytes", 4000)
    c.count("act_ckpt_bf16_bytes", 2000)
    c.count("act_ckpt_packed", 3)
    r = memory_report(c)
    assert (r["packed_bytes"], r["orig_bytes"], r["bf16_bytes"]) == (1000, 4000, 2000)
    assert (r["ratio_orig"], r["ratio_bf16"], r["packed_tensors"]) == (4.0, 2.0, 3)
    empty = memory_report(Counters())
    assert empty["packed_bytes"] == 0
    assert math.isnan(empty["ratio_orig"]) and math.isnan(empty["ratio_bf16"])


def test_o_parameter_storage_saves_ride_the_hook():
    """The docstring's weight-save claim, as a gate on the live model.

    autograd saves a linear's `weight.t()` — a VIEW of the parameter — like any
    other operand. Counting the packed saves that share storage with a
    parameter is what makes "weight saves are packed too" a measurement instead
    of an assumption (the module docstring's 92-of-383, 12 of them packed, is
    this count).
    """
    torch.manual_seed(0)
    model = DormouseModel(tiny_cfg())
    ids, targets = _ids()
    storages = {p.untyped_storage().data_ptr() for p in model.parameters()}
    weights: list[tuple[int, ...]] = []
    h = QuantSavedHooks("mxfp4", min_numel=4096)
    op = h._pack

    def spy(t: object) -> object:
        out = op(t)
        if (
            isinstance(t, torch.Tensor)
            and isinstance(out, Packed)
            and t.untyped_storage().data_ptr() in storages
        ):
            weights.append(tuple(t.shape))
        return out

    h._pack = spy  # type: ignore[method-assign]
    with h:
        _, loss, _ = model(ids, targets)
    loss.backward()
    print(
        f"[act_ckpt weight saves] {len(weights)} of "
        f"{h.counters['act_ckpt_packed']} packed saves are parameter storage"
    )
    assert weights, "the tiny preset's weight views must ride the hook"
    assert len(weights) <= h.counters["act_ckpt_packed"]


# --- the traps ---------------------------------------------------------------


def test_j_nesting_the_inner_pair_wins():
    """Nested hooks: torch keeps the INNERMOST pair, so the outer counts nothing."""
    x = _gauss((64,)).requires_grad_(True)
    outer, inner = Counters(), Counters()
    with QuantSavedHooks("mxfp4", min_numel=0, counters=outer):
        with QuantSavedHooks("mxfp4", min_numel=0, counters=inner):
            y = torch.sin(x).sum()
        y.backward()
    assert inner["act_ckpt_packed"] == 1, inner.line()
    assert outer == {}, outer.line()
    assert torch.isfinite(x.grad).all()


def test_k_sequential_reuse_is_safe_and_counters_accumulate():
    """Two contexts in a row, then one object entered twice: no state leaks."""
    for c in (Counters(), Counters()):
        x = _gauss((64,)).requires_grad_(True)
        with QuantSavedHooks("mxfp4", min_numel=0, counters=c):
            y = torch.sin(x).sum()
        y.backward()
        assert c["act_ckpt_packed"] == 1, c.line()
        assert torch.isfinite(x.grad).all()
    h = QuantSavedHooks("mxfp4", min_numel=0)
    for _ in range(2):
        x = _gauss((64,)).requires_grad_(True)
        with h:
            y = torch.sin(x).sum()
        y.backward()
    assert h.counters["act_ckpt_packed"] == 2, h.counters.line()
    assert h._ctxs == [], "the hooks pairs must all be released on exit"
    # The SAME object entered twice: the stack restores the outer pair.
    x = _gauss((64,)).requires_grad_(True)
    with h:
        with h:
            y = torch.sin(x).sum()
        y.backward()
    assert h._ctxs == []
    assert h.counters["act_ckpt_packed"] == 3, h.counters.line()


def test_l_the_checkpoint_placement_trap_is_measured_not_guessed():
    """Hooks OUTSIDE a checkpointed region miss its saves; INSIDE, they replace them.

    Measured on 2.13.0 for a 3-op body: outside, the pack hook fires for the
    checkpoint node's INPUT only (1 tensor) and the body still runs twice (the
    recompute is alive); inside, every save is ours (4) and the body runs ONCE —
    checkpoint's holder list stays empty, so it has nothing to recompute. The
    two mechanisms are mutually exclusive within one region, and the counter is
    the only thing that can tell a reader which one ran (`experts.py:102-111`).
    """

    def body(z: torch.Tensor) -> torch.Tensor:
        a = torch.sin(z)
        b = torch.exp(a * 0.5)
        return (b * a).sum()

    def counted(z: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return body(z)

    calls: list[int] = []
    x = _gauss((64, 64)).requires_grad_(True)
    outside = Counters()
    with QuantSavedHooks("mxfp4", min_numel=0, counters=outside):
        y = checkpoint(counted, x, use_reentrant=False)
    fwd_calls = len(calls)
    y.backward()
    print(
        f"[act_ckpt ckpt] outside packed={outside['act_ckpt_packed']} "
        f"calls={fwd_calls}->{len(calls)}"
    )
    assert outside["act_ckpt_packed"] == 1, "only the node's input reaches an outer pair"
    assert (fwd_calls, len(calls)) == (1, 2), "the recompute must still happen"

    calls.clear()
    inner = Counters()

    def wrapped(z: torch.Tensor) -> torch.Tensor:
        with QuantSavedHooks("mxfp4", min_numel=0, counters=inner):
            return counted(z)

    x2 = _gauss((64, 64)).requires_grad_(True)
    y2 = checkpoint(wrapped, x2, use_reentrant=False)
    fwd_calls = len(calls)
    y2.backward()
    print(
        f"[act_ckpt ckpt] inside packed={inner['act_ckpt_packed']} calls={fwd_calls}->{len(calls)}"
    )
    assert inner["act_ckpt_packed"] >= 3, inner.line()
    assert (fwd_calls, len(calls)) == (1, 1), "our hooks disable the recompute"
    assert torch.isfinite(x2.grad).all()


@CUDA
def test_m_the_unpack_restores_the_device():
    """The hook contract's device leg (skipped without a card)."""
    t = _gauss((64, 128)).cuda()
    for kind in ("mxfp4", "mxfp8"):
        p = pack_tensor(t, kind)
        assert isinstance(p, Packed) and p.codes.device.type == "cuda"
        dq = unpack_tensor(p)
        assert dq.device == t.device and dq.dtype is t.dtype and dq.shape == t.shape


# --- LOUD --------------------------------------------------------------------


def test_n_a_bad_kind_is_loud_naming_the_escape():
    """No silent fallback: an unknown kind names the known ones and the escape."""
    for bad in ("int4", "int8", "fp4", "", "MXFP4"):
        with pytest.raises(LoudError) as e:
            QuantSavedHooks(bad)
        assert "Escape" in str(e.value) and "mxfp4" in str(e.value), str(e.value)
    with pytest.raises(LoudError):
        pack_tensor(_gauss((64,)), "int4")  # the pure codec refuses too
    with pytest.raises(LoudError):
        QuantSavedHooks("mxfp4", min_numel=-1)
    assert KINDS == ("none", "mxfp4", "mxfp8")
