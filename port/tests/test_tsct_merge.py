"""The merge gate: `tsct_packed` and TSCT-2's arms ON at the same time.

`tsct_packed` (`4311efd`) and TSCT-2 (`9b5e6dd`) touched the same three files,
and both branches landed their own suites — neither ran the combination. This
gate is that missing measurement:

A. `tsct_packed=True` + `tsct_expert_basis=rank` (64 here, the real m400b rank)
   loads the unpacked TSCT-2 model's state dict (`strict=True`), so both arms
   hold the SAME weights, and `_banks` is `torch.equal` on all five tensors.
   The packed bank's `gate_u`/`up_u` are the folded `[E, d, rank]` effective
   basis (TSCT-2 wins the precedence at the shared plane); `v` and the two
   scales come from the packed buffers.
B. `tsct_packed=True` + `tsct_expert_lr` marks the packed `[E, rank, f]` buffer
   as E rows and applies the same per-expert row scale the unpacked `v`
   parameters get: one packed step is `torch.equal` to the unpacked step.
"""

from __future__ import annotations

import pytest
import torch
from test_tsct_packed import BASE, IDS

from dormouse.config import ModelConfig
from dormouse.model import DormouseModel, param_count
from dormouse.train.optim import MixOptimizer

TGT = torch.roll(IDS, -1, dims=1)

#: `tsct_expert_basis` is `rank` by the arm's own contract, so the combined
#: config widens both: rank 64 (m400b's), shared plane 128, selector 64.
COMBINED = dict(BASE, rank=64, shared_rank=128, tsct_expert_basis=64)


def _build(**over) -> DormouseModel:
    torch.manual_seed(0)
    return DormouseModel(ModelConfig(**{**COMBINED, **over}), device="cpu").to("cpu").train()


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
def test_packed_and_basis_bank_is_torch_equal(dt: torch.dtype) -> None:
    unpacked = _build()
    packed = _build(tsct_packed=True)
    assert packed.cfg.tsct_packed is True
    assert packed.cfg.tsct_expert_basis == 64
    packed.load_state_dict(unpacked.state_dict(), strict=True)
    assert param_count(packed) == param_count(unpacked), "the pack must not change the count"

    mu, mp = unpacked.loop.moe, packed.loop.moe
    # Distinct per-site scales: a swapped gate_s/up_s cannot hide behind ones.
    with torch.no_grad():
        for e, ex in enumerate(mu.experts):
            ex.gate_s.fill_(1.0 + e)
            ex.up_s.fill_(0.5 - e)
    packed.load_state_dict(unpacked.state_dict(), strict=True)
    mu._bank_cache = None
    mp._bank_cache = None
    a = mu._banks(dt)
    b = mp._banks(dt)
    assert len(a) == len(b) == 5, "gate_u, up_u, v, gate_s, up_s"
    # Both arms fold the selector, not stack E experts.
    assert a[0].shape == (4, 64, 64) and b[0].shape == (4, 64, 64)
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        assert torch.equal(x, y), f"bank tensor {i} moved on the packed+basis arm ({dt})"
    delta = max(float((x - y).detach().abs().max()) for x, y in zip(a, b, strict=True))
    print(
        f"[merge gate A] {dt}: bank torch.equal on 5/5, max|Δ|={delta:.1e}, "
        f"basis {tuple(a[0].shape)}"
    )


def test_packed_expert_lr_scales_the_same_rows() -> None:
    # `d_ffn=192` keeps `v [64, 192]` in its OWN shape group: at `d_ffn=128` it
    # shares `[64, 128]` with the selector `w`, and a 13-row vs 4-row NS batch
    # is only equal to ~3e-8 (bmm batching), which would hide the row scale.
    over = dict(tsct_expert_lr=1.0, moe_lb_mode="prop", moe_lb_gamma=0.002, d_ffn=192)
    unpacked = _build(**over)
    packed = _build(tsct_packed=True, **over)
    packed.load_state_dict(unpacked.state_dict(), strict=True)
    for m in (unpacked, packed):
        m.zero_grad(set_to_none=True)
        logits, _, _ = m(IDS, targets=TGT)
        logits.sum().backward()
    scripted = torch.tensor([5.0, 12.0, 3.0, 8.0])
    for m in (unpacked, packed):
        m.loop.moe.lfb_counts.copy_(scripted)

    opt_u = MixOptimizer(unpacked, lr=0.01, wd=0.0, opt="mix", tsct_expert_lr=1.0)
    opt_p = MixOptimizer(packed, lr=0.01, wd=0.0, opt="mix", tsct_expert_lr=1.0)
    opt_u.step()
    opt_p.step()
    scales = packed.loop.moe.expert_lr_scales()
    print(f"[merge gate B] scales={scales.tolist()}")
    for e in range(4):
        assert torch.equal(unpacked.loop.moe.experts[e].v, packed.loop.moe.packed_v[e]), (
            f"packed expert {e} did not get the unpacked row scale"
        )
