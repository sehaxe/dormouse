"""The fused CSA fine-selection arm (`csa_fused`) — six gates.

What the arm does, and what each gate would catch:

1. `csa_fused=false` is the REFERENCE path, bit-identical to main. The pin is a
   sha256 of the module's output AND every parameter gradient, captured from a
   clean `main` checkout at `1549e14` BEFORE `csa_fused` existed. A default that
   moved (a dropped `+`, an off-by-one in the mask) reddens this.
2. `csa_fused=true` matches the loop in fp32 to `1e-6`, and in bf16 to bf16
   round-off. The bf16 number is reported, not asserted at `1e-6`: one bf16
   ulp is `2^-8 ≈ 3.9e-3`, so a 1e-6 gate there would be a gate that cannot
   fail for the reason it names (the same defect class as the `1e-5` absolute
   merge-state gate in AGENTS §2.2).
3. A ring WRAP — the decode state carried across a chunk boundary — matches:
   same outputs, same `n`, same ring bytes. This is the `append_many` hoist.
4. `csa_fine=false` still builds no fine projection, and `use_csa=false` builds
   no module — the flag cannot resurrect a branch it is not supposed to touch.
5. No host read on the fused step path (`.item`/`.tolist`/`.__bool__`/`.float`
   patched to raise).
6. The CPU-provable launch arithmetic: the fused select lowers the module's
   non-metadata dispatch count, and the state build is no longer O(t/chunk) in
   dispatches — the count at t=64 equals the count at t=256.

Every gate was proven by a scratch revert (see the task report): restoring
either the select chain or the per-chunk append loop reddens gates 2/3/6.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch
from conftest import TINY, tiny_cfg

from dormouse.config import ModelConfig, validate
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel
from dormouse.model.csa import CompressedSelectiveAttention

ROOT = Path(__file__).resolve().parents[1]

#: Output + parameter-gradient sha of the REFERENCE module at seed 1234, input
#: generator 99, `chunk=4 topk=4`; captured from clean `main` at `1549e14`.
CSA_LOOP_SHA = "2dace6f5e373c03f64d09bf85f3c25bc05ff608ca0d4eb63a38aafa70dd4526a"
#: The same fixture on the fused arm, for the "the flag is live" half.
CSA_FUSED_SHA = "3105e7393cf0151a30831d36ac61047ecede8f1a9306d4372f78ec2079bacb30"

BATCH, SEQ = 2, 64
SEED = 1234
INPUT_SEED = 99


def _cfg(**over) -> ModelConfig:
    base = {**TINY, "use_csa": True, "csa_chunk": 4, "csa_topk": 4}
    base.update(over)
    return ModelConfig(**base)


def _module(fused: bool, **over) -> CompressedSelectiveAttention:
    torch.manual_seed(SEED)
    return CompressedSelectiveAttention(_cfg(csa_fused=fused, **over))


def _input(t: int = SEQ, seed: int = INPUT_SEED) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, t, TINY["d_model"], generator=gen)


def _sha(attn: CompressedSelectiveAttention) -> str:
    x = _input()
    y = attn(x)
    loss = (y * torch.arange(1, y.numel() + 1, dtype=y.dtype).reshape_as(y) / y.numel()).sum()
    loss.backward()
    h = hashlib.sha256()
    h.update(y.detach().contiguous().numpy().tobytes())
    for name, param in sorted(attn.named_parameters()):
        grad = param.grad if param.grad is not None else torch.zeros(1)
        h.update(name.encode())
        h.update(grad.contiguous().numpy().tobytes())
    return h.hexdigest()


def _load_launch_profile():
    spec = importlib.util.spec_from_file_location(
        "launch_profile", ROOT / "scripts" / "launch_profile.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROF = _load_launch_profile()


def _module_launches(fused: bool, t: int = SEQ, state: bool = False):
    """Non-metadata dispatches of one fwd+bwd (or one `forward_state`) call."""
    attn = _module(fused)
    x = _input(t)
    counter = PROF.DispatchCounter()
    with counter.observe(attn):
        if state:
            attn.forward_state(x, None)
        else:
            attn(x).sum().backward()
    return counter


# --- 1. the off switch is the reference path -------------------------------


@pytest.mark.pins
def test_gate_1_off_is_bit_identical_to_main():
    assert ModelConfig(**TINY).csa_fused is False, "the new field must default off"
    assert _sha(_module(False)) == CSA_LOOP_SHA, "csa_fused=false moved the reference path"
    assert _sha(_module(True)) == CSA_FUSED_SHA, "csa_fused=true is not the fused path"
    assert CSA_LOOP_SHA != CSA_FUSED_SHA, "the flag is inert"


# --- 2. parity with the loop, fp32 to 1e-6, bf16 to bf16 round-off ----------


def _run(fused: bool, dtype: torch.dtype):
    attn = _module(fused)
    if dtype is not torch.float32:
        attn = attn.to(dtype)
    x = _input().to(dtype)
    y = attn(x)
    loss = (y * torch.arange(1, y.numel() + 1, dtype=y.dtype).reshape_as(y) / y.numel()).sum()
    loss.backward()
    grads = [p.grad.detach().clone() for p in attn.parameters() if p.grad is not None]
    return y.detach(), grads


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gate_2_fused_matches_the_loop(dtype):
    y_loop, g_loop = _run(False, dtype)
    y_fused, g_fused = _run(True, dtype)
    scale = float(y_loop.float().abs().max())
    err = float((y_fused.float() - y_loop.float()).abs().max())
    gerr = max(
        float((a.float() - b.float()).abs().max()) for a, b in zip(g_fused, g_loop, strict=True)
    )
    gscale = max(float(g.float().abs().max()) for g in g_loop)
    print(
        f"{str(dtype):16s} output max|Δ| {err:.3e} (scale {scale:.3e})  "
        f"grad max|Δ| {gerr:.3e} (scale {gscale:.3e})"
    )
    if dtype is torch.float32:
        assert err <= 1e-6, err
        assert gerr <= 1e-6 * gscale, (gerr, gscale)
    else:
        # One bf16 ulp is 2^-8; the contraction's rounding changes by at most
        # one ulp of the output magnitude. This is the measured round-off, NOT
        # a 1e-6 gate (which bf16 cannot honour — see the module docstring).
        tol = 2.0 * torch.finfo(torch.bfloat16).eps * max(scale, 1e-30)
        assert err <= tol, (err, tol)
        assert gerr <= 2.0 * torch.finfo(torch.bfloat16).eps * max(gscale, 1e-30), gerr


def test_gate_2b_fused_matches_the_hierarchical_path():
    """The same select is the tail of `_hier_fine`; hashits must agree there too.

    `csa_topk=64` covers every level at this window, the shape the containment
    chain is specified on (`topk < entries` hits a separate pre-existing
    `_hier_fine` scalar-scatter bug, reported as a follow-up, not this arm's).
    """
    over = dict(
        csa_chunk=4,
        csa_topk=64,
        csa_coarse_chunk=8,
        csa_mid_chunk=16,
        csa_hierarchical=True,
        use_engram=False,
        moe_topk=0,
        max_seq_len=64,
    )
    x = _input()

    def run(fused: bool):
        torch.manual_seed(SEED)
        attn = CompressedSelectiveAttention(_cfg(csa_fused=fused, **over))
        return attn(x)

    ref, got = run(False), run(True)
    err = float((ref - got).abs().max())
    print(f"hierarchical max|Δ| {err:.3e}")
    assert err <= 1e-6, err


# --- 3. the ring wrap: state carried across a chunk boundary ----------------


def test_gate_3_ring_wrap_matches():
    def run(fused: bool):
        attn = _module(fused)
        first = _input(12)  # three whole chunks at chunk=4
        second = _input(8, seed=INPUT_SEED + 1)  # two more
        with torch.no_grad():
            out1, state = attn.forward_state(first, None)
            out2, state = attn.forward_state(second, state)
            one_shot, _ = attn.forward_state(torch.cat([first, second], dim=1), None)
        return out1, out2, one_shot, state

    o1a, o2a, full_a, sa = run(False)
    o1b, o2b, full_b, sb = run(True)
    pairs = (("window 1", o1a, o1b), ("window 2", o2a, o2b), ("one shot", full_a, full_b))
    for name, a, b in pairs:
        err = float((a - b).abs().max())
        print(f"{name:10s} max|Δ| {err:.3e}")
        assert err <= 1e-6, (name, err)
    assert (sa.n, sa.count) == (sb.n, sb.count) == (5, 0), (sa.n, sa.count, sb.n, sb.count)
    assert sa.k.shape == sb.k.shape
    assert torch.equal(sa.k[: sa.n], sb.k[: sb.n]), "the bulk append changed the ring keys"
    assert torch.equal(sa.v[: sa.n], sb.v[: sb.n]), "the bulk append changed the ring values"


# --- 4. the fine level off still builds nothing ----------------------------


def test_gate_4_fine_off_builds_nothing():
    cfg = _cfg(csa_fine=False, csa_coarse_chunk=64)
    attn = CompressedSelectiveAttention(cfg)
    for name in ("W_q", "W_k", "W_v", "W_o"):
        assert not hasattr(attn, name), f"{name} was built with csa_fine=false"
    assert not any(
        name.startswith(("W_q.", "W_k.", "W_v.", "W_o.")) for name, _ in attn.named_parameters()
    )
    # The model arm: no module at all when the whole branch is off.
    model = DormouseModel(tiny_cfg(use_csa=False))
    assert model.loop.csa is None
    assert "loop.csa" not in dict(model.named_modules())
    # And the fused flag cannot ride on an arm that does not select.
    with pytest.raises(LoudError, match="csa_fused"):
        validate(tiny_cfg(use_csa=True, csa_fine=False, csa_coarse_chunk=64, csa_fused=True))
    with pytest.raises(LoudError, match="csa_fused"):
        validate(tiny_cfg(csa_fused=True))
    print("fine off: no W_q/W_k/W_v/W_o, no module; csa_fused refused on both off arms")


# --- 5. no host read on the fused path -------------------------------------


def test_gate_5_no_host_sync_on_the_fused_path(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("host sync inside the fused CSA path (ADR-0018 rule 2)")

    for name in ("item", "tolist", "float", "__float__"):
        monkeypatch.setattr(torch.Tensor, name, boom)
    monkeypatch.setattr(torch.Tensor, "bool", boom)
    monkeypatch.setattr(torch.Tensor, "__bool__", boom)
    attn = _module(True)
    x = _input(32)
    counters = Counters()
    out = attn(x, counters=counters)
    out.sum().backward()
    _, state = attn.step_state(x[:, :1], None, counters=counters)
    assert state.n == 0
    assert torch.equal(torch.isfinite(out), torch.ones_like(out, dtype=torch.bool))
    print(f"no host read: csa_forward={counters['csa_forward']} csa_step={counters['csa_step']}")


# --- 6. the launch arithmetic provable on CPU ------------------------------


def test_gate_6_fused_select_lowers_the_dispatch_count():
    loop = _module_launches(False)
    fused = _module_launches(True)
    loop_total = sum(loop.launches.values())
    fused_total = sum(fused.launches.values())
    loop_gather = sum(n for (op, _m), n in loop.launches.items() if op == "aten::gather")
    fused_gather = sum(n for (op, _m), n in fused.launches.items() if op == "aten::gather")
    print(
        f"module fwd+bwd launches: loop {loop_total} -> fused {fused_total} "
        f"({fused_total - loop_total:+d}); gather {loop_gather} -> {fused_gather}"
    )
    assert fused_total <= loop_total - 6, (loop_total, fused_total)
    assert fused_gather < loop_gather, (loop_gather, fused_gather)
    # The off arm is the reference chain, so its count must be the larger one
    # even when the flag is set on the same module instance.
    assert loop_total > 0 and fused_total > 0


def test_gate_6b_state_build_is_no_longer_per_chunk():
    small = _module_launches(True, t=64, state=True)
    large = _module_launches(True, t=256, state=True)
    loop = _module_launches(False, t=256, state=True)
    small_total = sum(small.launches.values())
    large_total = sum(large.launches.values())
    loop_total = sum(loop.launches.values())
    copy_ops = ("aten::copy_", "aten::_to_copy")
    loop_copy = sum(n for (op, _m), n in loop.launches.items() if op in copy_ops)
    large_copy = sum(n for (op, _m), n in large.launches.items() if op in copy_ops)
    print(
        f"forward_state launches: loop@256 {loop_total} -> fused@256 {large_total}; "
        f"fused@64 {small_total}; ring copies loop {loop_copy} -> fused {large_copy}"
    )
    assert large_total == small_total, "the fused state build still scales with t/chunk"
    assert large_total < loop_total / 3, (large_total, loop_total)
    assert large_copy <= 8, large_copy
