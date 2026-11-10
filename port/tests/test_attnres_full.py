"""Full depth AttnRes (N = L) on `block_form="transformer"` — CPU only.

The six gates the variant owes an A/B:

1. identity at init: the flag ON reproduces the flag OFF logits (the weight sits
   on the most recent update, so the write IS the plain residual);
2. the `S·softmax` convention: `Σ_j w_j == S`, at init and after a random query;
3. order sensitivity: permuting the sources moves the output — a fixed average
   would not;
4. a hand-computed two-source case, arithmetic written in the test;
5. no `.item()`/host sync in the forward;
6. the flag OFF path is untouched: no AttnRes parameters/keys, the ON model is
   exactly `L·(d+1)` parameters larger.
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import tiny_cfg

from dormouse.config import ModelConfig
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.model.attnres import full_depth_attend, full_depth_weights

BATCH, SEQ = 2, 16


def _cfg(**over) -> ModelConfig:
    """A legal tiny vanilla transformer, the control's own shape."""
    base = dict(
        block_form="transformer",
        max_iter=1,
        loop_blocks=1,
        n_layers=3,
        use_kda=False,
        use_engram=False,
        use_tsct=False,
        moe_topk=0,
        n_experts=1,
        bf16=False,
    )
    base.update(over)
    return tiny_cfg(**base)


def _batch() -> torch.Tensor:
    return (torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251).to(torch.long)


def _build(seed: int, **over) -> DormouseModel:
    torch.manual_seed(seed)
    return DormouseModel(_cfg(**over))


# --- 1. identity at init ----------------------------------------------------


def test_the_arm_is_the_plain_residual_at_init():
    """ONE seeded model, its state copied, the flag flipped: logits equal ≤1e-5.

    The queries are zero and the prior is `(0, −30, …)`, so the `S·softmax`
    write is exactly the current update — the residual stream the OFF model
    builds. A `S`-fold or a `1/S`-fold perturbation at step 0 would show here.
    """
    on = _build(1234, use_attnres=True)
    off = _build(1234, use_attnres=False)
    missing, unexpected = off.load_state_dict(on.state_dict(), strict=False)
    assert not missing, missing
    assert all(k.startswith("attnres_full.") for k in unexpected), unexpected

    ids = _batch()
    with torch.no_grad():
        logits_on, _, _ = on(ids)
        logits_off, _, _ = off(ids)
    assert torch.isfinite(logits_on).all()
    diff = (logits_on - logits_off).abs().max().item()
    assert diff < 1e-5, f"AttnRes at init moved the logits by {diff:.3e}"
    print(f"[attnres] identity at init: max|Δlogits| = {diff:.3e}")


# --- 2. the S·softmax weight sum -------------------------------------------


def test_weights_sum_to_the_source_count():
    """`w = S·softmax` sums to `S` at init and after a random query."""
    sources = [torch.randn(BATCH, SEQ, 64) for _ in range(3)]  # n = 3 -> S = 4
    prior = torch.tensor([0.0, -30.0, -30.0])

    w0 = full_depth_weights(sources, torch.zeros(64), prior)
    assert torch.allclose(w0.sum(-1), torch.full((BATCH, SEQ), 4.0), atol=1e-5)
    # The identity lever: the zero source absorbs n, the current update is 1.
    assert torch.allclose(w0[..., 0], torch.full((BATCH, SEQ), 3.0), atol=1e-5)
    assert torch.allclose(w0[..., 1], torch.ones(BATCH, SEQ), atol=1e-5)

    w1 = full_depth_weights(sources, torch.randn(64), prior)
    assert torch.allclose(w1.sum(-1), torch.full((BATCH, SEQ), 4.0), atol=1e-5)


# --- 3. order sensitivity ---------------------------------------------------


def test_permuting_the_sources_moves_the_output():
    """A fixed average is permutation invariant; this mechanism is not."""
    s_new = torch.randn(BATCH, SEQ, 64)
    s_old = torch.randn(BATCH, SEQ, 64)
    assert torch.allclose(0.5 * (s_new + s_old), 0.5 * (s_old + s_new))
    q = torch.randn(64)
    prior = torch.tensor([0.5, -0.3])
    a = full_depth_attend([s_new, s_old], q, prior)
    b = full_depth_attend([s_old, s_new], q, prior)
    moved = (a - b).abs().max().item()
    assert moved > 1e-3, f"order did not matter ({moved:.3e}) — wired as an average"


# --- 4. a hand-computed two-source case -------------------------------------


def test_hand_computed_two_sources():
    """`d = 4`, one token: every number written as arithmetic, not captured."""
    eps = 1e-6
    u_new = torch.tensor([[[1.0, -2.0, 0.5, 3.0]]])
    u_old = torch.tensor([[[0.0, 1.0, -1.0, 2.0]]])
    q = torch.tensor([0.5, -1.0, 2.0, 0.25])
    prior = torch.tensor([0.3, -0.2])

    def rms(x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    s_new = (q * rms(u_new)).sum(-1) + 0.3
    s_old = (q * rms(u_old)).sum(-1) - 0.2
    logits = torch.stack([torch.full_like(s_new, math.log(2.0)), s_new, s_old], -1)
    e = logits.exp()
    w = 3.0 * e / e.sum(-1, keepdim=True)
    expected = w[..., 1:2] * u_new + w[..., 2:3] * u_old

    got = full_depth_attend([u_new, u_old], q, prior, eps=eps)
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=0)


# --- 5. no host sync --------------------------------------------------------


def test_the_full_forward_reads_no_device_scalar(monkeypatch):
    # The new mixer's source must be sync-free (the block's own forward is
    # guarded by the runtime spies below; its docstring merely NAMES `.item()`).
    import dormouse.model.attnres as attnres_mod

    src = open(attnres_mod.__file__, encoding="utf-8").read()
    assert ".item(" not in src and ".tolist(" not in src

    def boom(*_a, **_k):
        raise AssertionError("host sync in the AttnRes forward")

    for name in ("item", "tolist", "__int__", "__float__"):
        monkeypatch.setattr(torch.Tensor, name, boom, raising=True)
    model = _build(7, use_attnres=True)
    with torch.no_grad():
        model(_batch())


# --- 6. the flag OFF path is untouched -------------------------------------


def test_off_path_has_no_attnres_and_on_adds_exactly_l_d_plus_l():
    off = _build(3, use_attnres=False)
    on = _build(3, use_attnres=True)
    assert not hasattr(off, "attnres_full"), "the OFF model grew an attribute"
    assert not any(k.startswith("attnres_full.") for k in off.state_dict())
    extra = {
        k: v for k, v in on.state_dict().items() if k.startswith("attnres_full.")
    }
    assert set(extra) == {"attnres_full.query", "attnres_full.prior"}
    n_layers, d = off.cfg.n_layers, off.cfg.d_model
    assert param_count(on) - param_count(off) == n_layers * (d + 1)


# --- refusals ---------------------------------------------------------------


def test_use_attnres_still_refuses_the_kda_composition_on_transformer():
    """mHC owns the K-lane residual on the loop form; the two do not mix here."""
    with pytest.raises(LoudError) as e:
        _cfg(use_attnres=True, use_mhc=True)
    assert "use_mhc" in str(e.value) and "Escape:" in str(e.value)
