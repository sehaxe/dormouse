"""The fused Muon tail (`train/optim_tail.py`): arithmetic, refusals, mutation.

The gate the fused passes owe is NOT a speed number — it is that merging passes
did not move a value. So:

* the ALGEBRA is gated on the CPU, where every box can run it
  (`colrow_reference_` / `momentum_reference_` are the kernel's spec in torch
  ops, and the shipped `optim.colrow_normalize_` / `optim._momentum_update_` are
  the oracle);
* the KERNELS are gated on the card, against that spec, to 1e-6 — and the bf16
  momentum store is gated to the BIT (`torch.equal`), because a bf16 ULP is
  0.8% and the repo's own `test_muon_bf16_state_stores_one_rounded_fp32_
  accumulation_per_step` pins exactly that rounding;
* every ineligible input must FALL BACK COUNTED, not silently change the
  arithmetic (ADR-0019).

Run: `uv run pytest tests/test_optim_tail.py -q`
"""

from __future__ import annotations

import subprocess

import pytest
import torch

from dormouse.instrument import Counters
from dormouse.train import optim
from dormouse.train import optim_tail as ot


def _card_reason() -> str | None:
    """`None` when a GPU gate may run, else why not (one heavy thing at a time)."""
    if not torch.cuda.is_available():
        return "no CUDA device"
    if not ot.TRITON_AVAILABLE:
        return "no triton (the fused passes are the CUDA arm)"
    try:
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001 - a gate's precondition, not a run's
        return f"nvidia-smi unavailable ({e})"
    if used and int(used.splitlines()[0].split()[0]) > 2000:
        return f"card is held ({used} MiB): this gate would measure a busy card"
    return None


CARD = _card_reason()
gpu = pytest.mark.skipif(CARD is not None, reason=CARD or "")


# ---------------------------------------------------------------------------
# 1. the algebra: the fused decomposition IS the shipped function (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(12, 96, 32), (3, 96, 96), (1, 64, 48), (5, 48, 96)])
def test_the_fused_colrow_is_the_shipped_colrow_within_1e_6(shape):
    """Revert: reduce the rows of the RAW matrix (or drop an eps) -> this fails.

    The order is the definition: columns of `x`, then rows of `x/(col+ε)`.
    """
    torch.manual_seed(0)
    x = torch.randn(shape)
    want = optim.colrow_normalize_(x.clone())
    got = ot.colrow_reference_(x.clone())
    assert torch.allclose(got, want, rtol=0, atol=1e-6), (got - want).abs().max().item()
    # The same ops in the same order: this is not "close", it is the same numbers.
    assert torch.equal(got, want), "the fused decomposition moved a value"


def test_the_fused_tail_is_the_shipped_four_statements_within_1e_6():
    """`colrow -> *(-lr*scale) -> *flag -> p += u`, fused, must be the same step."""
    torch.manual_seed(1)
    update = torch.randn(4, 64, 32)
    params = [torch.randn(64, 32) for _ in range(4)]
    ref_params = [p.clone() for p in params]
    scale, flag = -3e-3, torch.ones((), dtype=torch.int32)

    ot._reference_tail(update.clone(), ref_params, scale, None, flag, "colrow")

    u = update.clone()
    ot.colrow_reference_(u)
    u.mul_(scale)
    u.mul_(flag.to(u.dtype))
    got = [p + row for p, row in zip(params, u, strict=True)]

    for a, b in zip(got, ref_params, strict=True):
        assert torch.allclose(a, b, rtol=0, atol=1e-6)
        assert torch.equal(a, b), "the fused tail moved a value"


def test_the_fused_tail_gates_a_skipped_step_to_zero():
    """The firewall: `flag=0` must leave the parameters EXACTLY where they were."""
    torch.manual_seed(2)
    update = torch.randn(2, 32, 16)
    params = [torch.randn(32, 16) for _ in range(2)]
    before = [p.clone() for p in params]
    ot._reference_tail(
        update.clone(), params, -1e-3, None, torch.zeros((), dtype=torch.int32), "colrow"
    )
    for p, b in zip(params, before, strict=True):
        assert torch.equal(p, b), "a skipped step moved a parameter"


# ---------------------------------------------------------------------------
# 2. the mutation gate: the fused pass is a function of its inputs, not of a
#    broadcast tangent
# ---------------------------------------------------------------------------


def test_perturbing_one_gradient_moves_the_momentum():
    """A gate that cannot fail: change ONE element of one gradient."""
    torch.manual_seed(3)
    mom = torch.zeros(2, 8, 8, dtype=torch.bfloat16)
    grad = torch.zeros(2, 8, 8, dtype=torch.bfloat16)
    ot.momentum_reference_(mom, grad, 0.95)
    flat = grad.view(2, -1)
    flat[1, 37] = 1.0
    ot.momentum_reference_(mom, grad, 0.95)
    assert mom.view(2, -1)[1, 37].item() != 0.0
    assert int((mom.view(2, -1) != 0).sum()) == 1, "one gradient element moved more than one slot"


def test_perturbing_one_update_element_moves_exactly_one_parameter_element():
    """The fused apply is elementwise: a perturbation cannot spread."""
    torch.manual_seed(4)
    update = torch.randn(2, 16, 8)
    params = [torch.zeros(16, 8) for _ in range(2)]
    before = [p.clone() for p in params]
    u = update.clone()
    ot.colrow_reference_(u)
    u.mul_(-1.0)
    params[0] += u[0]
    assert not torch.equal(params[0], before[0])
    assert torch.equal(params[1], before[1]), "one row's update leaked into another parameter"


# ---------------------------------------------------------------------------
# 3. the kernel gate (card): the Launches equal the spec, and the bf16 store is
#    bit-exact against the shipped recurrence
# ---------------------------------------------------------------------------


@gpu
def test_the_colrow_kernel_is_the_reference_within_1e_6():
    torch.manual_seed(5)
    x = torch.randn(12, 256, 96, device="cuda")
    want = ot.colrow_reference_(x.clone())
    counters = Counters()
    got = ot.colrow_normalize_(x.clone(), counters=counters)
    assert counters.get("opt_tail_colrow") == 1, dict(counters)
    assert not counters.get("opt_tail_fallback"), counters.why
    diff = (got - want).abs().max().item()
    assert diff < 1e-6, diff


@gpu
def test_the_exact_momentum_path_is_BIT_equal_to_the_shipped_bf16_recurrence():
    """The shipped path of `momentum_update_`: kernel = `to(f32).mul_(mu)`, add = ATen.

    Nothing to gate here in principle (the add and the store ARE the shipped
    calls) — which is exactly why it is the default and the one-kernel arm is
    not.
    """
    torch.manual_seed(6)
    mu = 0.95
    g = torch.randn(4, 64, 32, device="cuda")
    want = torch.zeros(4, 64, 32, device="cuda", dtype=torch.bfloat16)
    params = []
    for i in range(4):
        p = torch.nn.Parameter(torch.zeros(64, 32, device="cuda"))
        p.grad = g[i].clone()
        params.append(p)
    optim._momentum_update_(want, params, mu, 1 << 25)

    got = torch.zeros_like(want)
    counters = Counters()
    ot.momentum_update_(got, params, mu, 1 << 25, counters=counters)
    assert counters.get("opt_tail_momentum") == 1, dict(counters)
    assert torch.equal(got, want), (
        (got.float() - want.float()).abs().max().item(),
        int((got != want).sum()),
    )


@gpu
def test_the_one_kernel_momentum_arm_is_BIT_equal_or_its_mode_is_wrong():
    """`MOMENTUM_FAST`: one kernel, 16 B/elem, measured 2.03x — and its MODE.

    This is the gate that decides `MOMENTUM_MODE`. At 0 (two roundings) it is
    RED against the shipped recurrence on the card; 1 is `fma(alpha, g, a)`,
    which is what nvcc contracts `a + alpha * b` into. Do not flip the mode
    without this test green.
    """
    """The state rung rounds ONCE per step; an `fma` in the kernel would not.

    94.6M elements per step make a 2^-16-per-element rounding difference a
    systematic drift, which is why this gate is `torch.equal` and not 1e-6.
    """
    torch.manual_seed(6)
    mu = 0.95
    g = torch.randn(4, 64, 32, device="cuda")
    want = torch.zeros(4, 64, 32, device="cuda", dtype=torch.bfloat16)
    params = []
    for i in range(4):
        p = torch.nn.Parameter(torch.zeros(64, 32, device="cuda"))
        p.grad = g[i].clone()
        params.append(p)
    optim._momentum_update_(want, params, mu, 1 << 25)

    got = torch.zeros_like(want)  # bf16: `zeros_like(g)` would be fp32 and the
    # arm would refuse (counted) instead of running — a gate that tests the
    # refusal, not the rounding.
    counters = Counters()
    prev = ot.MOMENTUM_FAST
    ot.MOMENTUM_FAST = True
    try:
        ot.momentum_update_(got, params, mu, 1 << 25, counters=counters)
    finally:
        ot.MOMENTUM_FAST = prev
    assert counters.get("opt_tail_momentum") == 1, dict(counters)
    assert torch.equal(got, want), (
        f"MOMENTUM_MODE={ot.MOMENTUM_MODE} is not the shipped rounding:",
        (got.float() - want.float()).abs().max().item(),
        int((got != want).sum()),
    )


@gpu
def test_the_tail_kernel_is_the_shipped_writeback_within_1e_6():
    torch.manual_seed(7)
    update = torch.randn(4, 128, 64, device="cuda")
    params = [torch.randn(128, 64, device="cuda") for _ in range(4)]
    ref = [p.clone() for p in params]
    flag = torch.ones((), dtype=torch.int32, device="cuda")
    scale = -3e-3

    ot._reference_tail(update.clone(), ref, scale, None, flag, "colrow")

    counters = Counters()
    ot.muon_tail_(update.clone(), params, scale, flag=flag, counters=counters)
    assert counters.get("opt_tail_tail") == 1, dict(counters)
    diff = max((a - b).abs().max().item() for a, b in zip(params, ref, strict=True))
    assert diff < 1e-6, diff


# ---------------------------------------------------------------------------
# 4. refusals: an ineligible call is COUNTED and falls back to the shipped code
# ---------------------------------------------------------------------------


def test_the_momentum_falls_back_counted_on_the_fp32_rung():
    """The fp32 rung has no fp32 temporary to remove -> the shipped path, counted."""
    g = torch.randn(2, 16, 8)
    mom = torch.zeros(2, 16, 8)
    params = []
    for i in range(2):
        p = torch.nn.Parameter(torch.zeros(16, 8))
        p.grad = g[i].clone()
        params.append(p)
    want = mom.clone()
    optim._momentum_update_(want, params, 0.9, 1 << 25)
    counters = Counters()
    ot.momentum_update_(mom, params, 0.9, 1 << 25, counters=counters)
    assert torch.equal(mom, want)
    assert counters.get("opt_tail_momentum") is None
    assert counters.get("opt_tail_fallback") == 1
    assert "fp32" in counters.why


@pytest.mark.parametrize(
    "kwargs",
    [
        {"colrow_3d": "frobenius"},
        {"shape": (16, 8)},  # 2-D: the per-matrix arm
        {"dtype": torch.bfloat16},
    ],
)
def test_the_colrow_falls_back_counted_when_it_cannot_serve(kwargs):
    shape = kwargs.pop("shape", (2, 16, 8))
    dtype = kwargs.pop("dtype", torch.float32)
    x = torch.randn(shape, dtype=dtype)
    want = optim.colrow_normalize_(x.clone(), colrow_3d=kwargs.get("colrow_3d", "colrow"))
    counters = Counters()
    got = ot.colrow_normalize_(x.clone(), counters=counters, **kwargs)
    assert torch.equal(got, want)
    assert counters.get("opt_tail_colrow") is None
    assert counters.get("opt_tail_fallback") == 1
    assert counters.why.startswith("opt_tail:")
