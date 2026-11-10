"""Per-head Newton–Schulz on the attention Q/K/V projections (K3 §2.5).

arXiv:2607.24653's stabiliser: a Q/K/V momentum matrix is split along its head
axis and each `[head_dim, cols]` block is orthogonalized ON ITS OWN, so the head
carrying the largest momentum cannot set the direction every head shares. The
claim is SCALE EQUALIZATION, and it is measured here rather than quoted.

What this file gates, in the order an auditor reads it:

* an EXTERNAL anchor — the Jordan quintic is written out HERE from the formula
  (`G = X·Xᵀ`, `X ← a·X + (b·G + c·G²)·X`) with the coefficients pinned as this
  test's own literals, applied to ONE block; the production per-head path must
  match it to 1e-5 in fp32. Nothing here asks the kernel for its reference;
* a split of ONE block (`n_heads == 1` on an unpacked projection) is the
  full-matrix call BIT FOR BIT (`torch.equal`);
* the arm's own claim: one block 10× larger → per-head update norms spread
  < 5%, the full-matrix path's > 50% on the SAME tensor;
* the wiring: the flag reaches `MixOptimizer` with the model's head geometry,
  marks ONLY the attention projections, runs both the per-parameter and the
  batched path, is COUNTED when it runs and counts NOTHING when it is off;
* the refusals: no Muon group, no head geometry, or a head axis the split cannot
  tile is LOUD, never a silent full-matrix fallback.
"""

from __future__ import annotations

import pytest
import torch
from conftest import tiny_cfg, tiny_model
from torch import nn

from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train.optim import MixOptimizer, Muon, zeropower_via_ns

#: Jordan's NS triple (`optim.JORDAN_NS_COEFFS`), written out HERE so the
#: reference cannot follow a change in the kernel: the gate is that the arm
#: implements the FORMULA, not that it agrees with itself.
A_COEF, B_COEF, C_COEF = 3.4445, -4.7750, 2.0315
NS_STEPS = 5

#: `tiny_model()`'s head geometry: `n_heads · head_dim = 64 = d_model`, so a
#: `[64, 64]` projection splits into 4 blocks of `[16, 64]`.
TINY_HEADS, TINY_HEAD_DIM = 4, 16


def reference_ns(block: torch.Tensor, steps: int = NS_STEPS) -> torch.Tensor:
    """The chain, transcribed from the formula for ONE square block.

    `zeropower_via_ns`'s 2-D branch: the Frobenius prescale (`+1e-7`), no
    transpose (the block is square), then the Jordan quintic per iteration.
    """
    x = block.to(torch.float32)
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.T
        x = A_COEF * x + (B_COEF * gram + C_COEF * (gram @ gram)) @ x
    return x


def block_norms(y: torch.Tensor, blocks: int, head_dim: int) -> torch.Tensor:
    """The Frobenius norm of every `[head_dim, ·]` block of a `[blocks·hd, ·]`."""
    return y.reshape(blocks, head_dim, y.shape[-1]).flatten(1).norm(dim=1)


def spread(norms: torch.Tensor) -> float:
    """`(max - min) / mean` — the relative spread of a per-head norm vector."""
    return float((norms.max() - norms.min()) / norms.mean())


def _muon() -> Muon:
    """A `Muon` over one unused parameter: the arm's methods are what we call."""
    return Muon([nn.Parameter(torch.zeros(TINY_HEAD_DIM, TINY_HEAD_DIM))], lr=1e-3)


def _seed_grads(model: nn.Module, seed: int = 0) -> None:
    """Fixed random gradients, by parameter order, so two models see the same."""
    gen = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.grad = torch.randn(p.shape, generator=gen) * 0.05


# --- the external anchor ----------------------------------------------------


def test_a_per_head_block_is_the_newton_schulz_iteration_itself():
    """Head *i* of the split path == the test's own NS on block *i*, to 1e-5."""
    torch.manual_seed(0)
    blocks, hd = 6, 64
    x = torch.randn(blocks * hd, hd)
    out = _muon()._orthogonalize_marked(x, NS_STEPS, (blocks, hd))
    assert out.shape == x.shape
    for i in (0, 3, 5):
        mine = reference_ns(x[i * hd : (i + 1) * hd])
        got = out[i * hd : (i + 1) * hd]
        torch.testing.assert_close(
            got,
            mine,
            atol=1e-5,
            rtol=0,
            msg=f"head {i} is not the reference NS of its own block",
        )
    # ... and the split is NOT the full-matrix chain on the same tensor, so a
    # silent fall back to the old path fails here instead of passing quietly.
    full = zeropower_via_ns(x, NS_STEPS)
    assert not torch.allclose(out, full, atol=1e-3), (
        "the per-head output equals the full-matrix output: the split did not run"
    )


@pytest.mark.parametrize("shape", [(64, 64), (64, 96), (96, 64)])
def test_one_head_is_the_full_matrix_call_bit_for_bit(shape):
    """A split of ONE block is the old call, exactly — `torch.equal`, not 1e-6."""
    torch.manual_seed(1)
    x = torch.randn(*shape)
    muon = _muon()
    out = muon._orthogonalize_marked(x, NS_STEPS, (1, shape[0]))
    assert torch.equal(out, zeropower_via_ns(x, NS_STEPS))
    assert "muon_per_head" not in muon.counters, "no split ran, so nothing is counted"


# --- K3's claim, measured ---------------------------------------------------


def test_per_head_equalizes_the_scale_and_the_full_matrix_does_not():
    """One 10× head: per-head norms spread < 5%, full-matrix > 50%."""
    torch.manual_seed(2)
    blocks, hd = 8, 64
    x = torch.randn(blocks * hd, hd)
    x[3 * hd : 4 * hd] *= 10.0  # ONE head ten times larger than the rest
    per_head = _muon()._orthogonalize_marked(x, NS_STEPS, (blocks, hd))
    full = zeropower_via_ns(x, NS_STEPS)
    ph_norms = block_norms(per_head, blocks, hd)
    fu_norms = block_norms(full, blocks, hd)
    ph, fu = spread(ph_norms), spread(fu_norms)
    assert ph < 0.05, (
        f"per-head spread {ph:.2%} (want < 5%); norms {ph_norms.tolist()}; "
        f"full-matrix spread {fu:.2%}"
    )
    assert fu > 0.5, (
        f"full-matrix spread {fu:.2%} (want > 50%); norms {fu_norms.tolist()}; "
        f"per-head spread {ph:.2%}"
    )
    # The equalization is real, not "every head got shrunk": each block keeps
    # the polar factor's magnitude `sqrt(head_dim)` up to the chain's own
    # 5-iteration truncation (~7.0 of 8.0 on a random block).
    assert torch.allclose(ph_norms, torch.full_like(ph_norms, float(hd) ** 0.5), rtol=0.2), (
        f"per-head norms {ph_norms.tolist()} are not the polar magnitude {float(hd) ** 0.5:.2f}"
    )


# --- wiring: the flag, the markers, the counters ----------------------------


def test_the_flag_marks_only_the_attention_q_k_v_projections():
    """`muon_per_head=true` marks q/k/v — and NOT the same-shaped g_proj."""
    model = tiny_model()
    opt = MixOptimizer(
        model,
        lr=1e-2,
        wd=0.0,
        muon_per_head=True,
        n_heads=TINY_HEADS,
        head_dim=TINY_HEAD_DIM,
    )
    assert opt.per_head_sites == [
        "loop.attn.q_proj.weight",
        "loop.attn.k_proj.weight",
        "loop.attn.v_proj.weight",
    ]
    for name in opt.per_head_sites:
        p = dict(model.named_parameters())[name]
        assert p._muon_per_head == (TINY_HEADS, TINY_HEAD_DIM), name
    # `g_proj` is [64, 64] like q_proj: the marker is the NAME's, not the shape's.
    assert not hasattr(model.loop.attn.g_proj.weight, "_muon_per_head")
    _seed_grads(model)
    opt.step()
    assert opt.counters.get("muon_per_head", 0) > 0, "the arm ran but counted nothing"


def test_the_packed_qkv_is_three_head_stacks():
    """`block_form="transformer"` packs q/k/v into ONE `[3·d, d]` weight."""
    model = tiny_model(block_form="transformer", max_iter=1, use_kda=False, use_engram=False)
    opt = MixOptimizer(
        model, lr=1e-2, wd=0.0, muon_per_head=True, n_heads=TINY_HEADS, head_dim=TINY_HEAD_DIM
    )
    assert opt.per_head_sites == ["transformer_blocks.0.qkv.weight"]
    p = dict(model.named_parameters())["transformer_blocks.0.qkv.weight"]
    # The q/k/v stack is a per-HEAD factor: 3 projections x 4 heads, each block
    # `head_dim=16` rows — NOT 4 blocks of 48, which would mix q, k and v.
    assert p._muon_per_head == (3 * TINY_HEADS, TINY_HEAD_DIM)
    _seed_grads(model)
    opt.step()
    assert opt.counters.get("muon_per_head", 0) == 1


def test_flag_off_is_the_full_matrix_path_and_counts_nothing():
    model = tiny_model()
    opt = MixOptimizer(model, lr=1e-2, wd=0.0)
    assert opt.per_head_sites == []
    assert not any(hasattr(p, "_muon_per_head") for p in model.parameters())
    _seed_grads(model)
    opt.step()
    assert opt.counters.get("muon_per_head", 0) == 0


def test_the_arm_moves_the_attention_projection_and_nothing_else():
    """Same weights, same grads, one step: q_proj moves, every other 2-D does not."""
    off, on = tiny_model(), tiny_model()
    on.load_state_dict(off.state_dict())
    o_off = MixOptimizer(off, lr=1e-2, wd=0.0)
    o_on = MixOptimizer(
        on, lr=1e-2, wd=0.0, muon_per_head=True, n_heads=TINY_HEADS, head_dim=TINY_HEAD_DIM
    )
    _seed_grads(off, seed=7)
    for (_, a), (_, b) in zip(off.named_parameters(), on.named_parameters(), strict=True):
        b.grad = a.grad.clone()
    o_off.step()
    o_on.step()
    got = dict(on.named_parameters())
    want = dict(off.named_parameters())
    assert not torch.equal(got["loop.attn.q_proj.weight"], want["loop.attn.q_proj.weight"]), (
        "muon_per_head=true produced the full-matrix update"
    )
    for name in ("loop.attn.g_proj.weight", "loop.mem_dense.weight"):
        assert torch.equal(got[name], want[name]), (
            f"{name} moved: the arm is not confined to the marked projections"
        )


def test_a_shared_shape_group_runs_the_split_on_the_batched_path():
    """Two marked `[64, 64]` params take the batched chunk path, and it splits."""
    torch.manual_seed(4)
    grads = torch.randn(2, 64, 64)

    def step(marked: bool) -> tuple[list[torch.Tensor], Muon]:
        params = [nn.Parameter(torch.randn(64, 64)), nn.Parameter(torch.randn(64, 64))]
        if marked:
            for p in params:
                p._muon_per_head = (TINY_HEADS, TINY_HEAD_DIM)
        muon = Muon(params, lr=1e-3)
        for i, p in enumerate(params):
            p.grad = grads[i].clone()
        muon.batched_step_from_model(None)  # one group of 2 -> the batched path
        return [p.detach().clone() for p in params], muon

    torch.manual_seed(9)
    plain, plain_opt = step(False)
    torch.manual_seed(9)
    split, split_opt = step(True)
    assert "muon_per_head" not in plain_opt.counters
    assert split_opt.counters.get("muon_per_head") == 1, "one chunk, one COUNTED entry"
    for a, b in zip(plain, split, strict=True):
        assert not torch.equal(a, b), "the marked group ran the full-matrix chain"


# --- the refusals -----------------------------------------------------------


def test_muon_per_head_without_muon_is_refused():
    with pytest.raises(LoudError, match=r"muon_per_head=true with opt='adamw'"):
        MixOptimizer(
            tiny_model(),
            lr=1e-3,
            wd=0.0,
            opt="adamw",
            muon_per_head=True,
            n_heads=TINY_HEADS,
            head_dim=TINY_HEAD_DIM,
        )


def test_muon_per_head_refuses_a_head_axis_it_cannot_tile():
    """A GQA `k_proj` (`n_kv_heads < n_heads`) is `kv_heads·head_dim` rows."""
    model = DormouseModel(tiny_cfg(n_kv_heads=2))
    with pytest.raises(LoudError, match="k_proj"):
        MixOptimizer(
            model,
            lr=1e-3,
            wd=0.0,
            muon_per_head=True,
            n_heads=TINY_HEADS,
            head_dim=TINY_HEAD_DIM,
        )


def test_muon_per_head_refuses_without_head_geometry():
    with pytest.raises(LoudError, match="head geometry"):
        MixOptimizer(tiny_model(), lr=1e-3, wd=0.0, muon_per_head=True)
