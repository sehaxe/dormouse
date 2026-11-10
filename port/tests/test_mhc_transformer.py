"""mHC on `block_form="transformer"` — the K-lane residual across LAYERS.

The loop form carries its K lanes across ITERATIONS (`loopblock._window`); this
form carries the same full-width lanes across the unique block stack, one write
per block (`DormouseModel._forward_transformer_mhc`). The operator is the SAME
`mhc.MHCMix` the loop uses — Eq. 3/7/8/9 are not re-derived here — and the two
readout expressions are the loop's verbatim (`mhc_readout`): the mHC-only arm
sums the lanes, the composition divides by K.

Gates, mirroring `test_attnres_full.py` because this arm owes the same ones:

1. identity at init: ONE seeded model, its state copied, the flag flipped,
   logits equal ≤1e-5 fp32 (both arms, K = 2 and 3);
2. `H_res` is doubly stochastic in BOTH directions at 1e-6, at init and after
   random `b_res`;
3. K = 2 and K = 3 both run, and both readout expressions are pinned;
4. the four flag pairs run, and `off/off` is bit-identical to the untouched
   model (no new attribute, no new state key);
5. no `.item()`/host sync in the forward;
6. a hand-computed 2-lane, one-write case: every number is arithmetic in this
   file, not a captured run.

Plus the refusals and the measured parameter delta.
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import tiny_cfg

from dormouse.config import ModelConfig
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.model.mhc import MHCMix
from dormouse.model.model import mhc_readout

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


def _mhc_delta(d: int, k: int) -> int:
    """`3 φ + 3 α + b_pre/b_post/b_res`: `K²d(K+2) + 3 + 2K + K²`."""
    return k * k * d * (k + 2) + 3 + 2 * k + k * k


# --- 1. identity at init ----------------------------------------------------


@pytest.mark.parametrize("streams", [2, 3])
@pytest.mark.parametrize("attnres", [False, True])
def test_the_arm_is_the_plain_residual_at_init(streams, attnres):
    """ONE seeded model, its state copied, the flag flipped: logits equal ≤1e-5.

    Both arms must hand `_readout` the plain state: the mHC-only lanes carry
    `1/K` of it (its raw sum is the state) and the composition's lanes carry
    it (`/K` reads it). A `K×` state would fail the final RMSNorm's `1e-3` eps,
    which is why the lane scale is per-arm (`_forward_transformer`).
    """
    on = _build(1234, use_mhc=True, mhc_streams=streams, use_attnres=attnres)
    off = _build(1234, use_attnres=attnres)
    missing, unexpected = off.load_state_dict(on.state_dict(), strict=False)
    assert not missing, missing
    assert all(k.startswith("mhc_full.") for k in unexpected), unexpected

    ids = _batch()
    with torch.no_grad():
        logits_on, _, _ = on(ids)
        logits_off, _, _ = off(ids)
    assert torch.isfinite(logits_on).all()
    diff = (logits_on - logits_off).abs().max().item()
    assert diff < 1e-5, f"mHC at init moved the logits by {diff:.3e}"
    print(f"[mhc-transformer] K={streams} attnres={attnres} identity: max|Δlogits|={diff:.3e}")


# --- 2. the doubly-stochastic constraint -----------------------------------


def test_h_res_is_doubly_stochastic_in_both_directions():
    """Eq. 6/9: rows AND columns sum to 1 within 1e-6, init and perturbed."""
    for streams in (2, 3):
        mix = MHCMix(8, streams)
        lanes = torch.randn(2, 3, streams, 8)
        for label in ("init", "sinkhorn"):
            if label == "sinkhorn":
                # Seeded: an unseeded draw makes the 1e-6 gate flaky.
                torch.manual_seed(20261110)
                with torch.no_grad():
                    mix.b_res.normal_(0.0, 1.0)
            _, _, h_res = mix.hyper_mappings(lanes)
            row = (h_res.detach().sum(-1) - 1.0).abs().max()
            col = (h_res.detach().sum(-2) - 1.0).abs().max()
            print(f"[mhc-transformer] K={streams} {label}: row={row:.2e} col={col:.2e}")
            assert float(row) < 1e-6, (streams, label, float(row))
            assert float(col) < 1e-6, (streams, label, float(col))
            assert bool((h_res >= 0.0).all())


# --- 3. K lanes and the readout convention ---------------------------------


def test_k_two_and_three_run_and_the_readout_is_the_loops():
    """K = 2/3 build and write one lane per block; sum vs `1/K` pinned."""
    for streams in (2, 3):
        model = _build(5, use_mhc=True, mhc_streams=streams)
        counters = Counters()
        with torch.no_grad():
            logits, _, _ = model(_batch(), counters=counters)
        assert torch.isfinite(logits).all()
        assert counters["mhc"] == model.cfg.n_layers
        assert counters["transformer_block"] == model.cfg.n_layers
        assert param_count(model) - param_count(_build(5)) == _mhc_delta(
            model.cfg.d_model, streams
        )

    lanes = torch.randn(2, 3, 3, 16)
    # The mHC-only readout is the UNNORMALISED sum; the composition is `/K`.
    assert torch.equal(mhc_readout(lanes, False), lanes.sum(-2))
    assert torch.allclose(mhc_readout(lanes, True), lanes.sum(-2) / 3.0, atol=1e-7)


# --- 4. the composition and the OFF arm ------------------------------------


def test_the_flag_pairs_run_and_off_off_is_the_untouched_model():
    """(off/off, mhc/off, off/attnres, mhc/attnres) all run; OFF adds nothing."""
    ids = _batch()
    for mhc, attnres in ((False, False), (True, False), (False, True), (True, True)):
        model = _build(7, use_mhc=mhc, use_attnres=attnres)
        counters = Counters()
        with torch.no_grad():
            logits, _, _ = model(ids, counters=counters)
        assert torch.isfinite(logits).all(), (mhc, attnres)
        assert ("mhc" in counters) is mhc
        # Removing the mixer from the trained combination must move the logits
        # (the arm is in the write, not a dormant scaffold).
        if mhc:
            with torch.no_grad():
                before, _, _ = model(ids)
                model.mhc_full, keep = None, model.mhc_full
                after, _, _ = model(ids)
                model.mhc_full = keep
            assert not torch.equal(before, after)

    # `off/off` is today's control: no module, no key, the same bits.
    untouched = _build(7)
    off = _build(7, use_mhc=False, use_attnres=False)
    assert not hasattr(off, "mhc_full")
    assert not any(k.startswith(("mhc_full.", "attnres_full.")) for k in off.state_dict())
    with torch.no_grad():
        a, _, _ = untouched(ids)
        b, _, _ = off(ids)
    assert torch.equal(a, b)


# --- 5. no host sync --------------------------------------------------------


def test_the_mhc_forward_reads_no_device_scalar(monkeypatch):
    # The mixer's source must be sync-free; the model's forward is guarded by
    # the same spies the AttnRes gate uses.
    import dormouse.model.mhc as mhc_mod

    src = open(mhc_mod.__file__, encoding="utf-8").read()
    assert ".item(" not in src and ".tolist(" not in src

    def boom(*_a, **_k):
        raise AssertionError("host sync in the mHC transformer forward")

    for name in ("item", "tolist", "__int__", "__float__"):
        monkeypatch.setattr(torch.Tensor, name, boom, raising=True)
    for attnres in (False, True):
        model = _build(7, use_mhc=True, use_attnres=attnres)
        with torch.no_grad():
            model(_batch())


# --- 6. a hand-computed one-write case -------------------------------------


def test_one_write_two_lanes_is_hand_computed():
    """d = 3, K = 2, one write: every lane number is arithmetic here.

    `b_pre = b_post = 0` and the α gates at 0 make the static mappings:
    `H_pre = [.5, .5]` (convex read = the mean), `H_post = [1, 1]` rescaled to
    sum `K·c = 1` (so `.5` each), and `b_res = [[0, −ln4], [−ln4, 0]]` →
    `H_res = Sinkhorn([[1, .25], [.25, 1]]) = [[.8, .2], [.2, .8]]` (the crate's
    fixture, `test_mhc.py`). With the mHC-only lane scale `c = 1/2`:

        lanes  = ½·[[1,2,3], [4,5,6]] = [[.5,1,1.5], [2,2.5,3]]
        ctx    = mean(lanes)/c = [1.25,1.75,2.25]·2 = [2.5,3.5,4.5]
        mixed  = H_res·lanes = [[.8,1.3,1.8], [1.7,2.2,2.7]]
        out    = mixed + ½·[10,20,30] = [[5.8,11.3,16.8], [6.7,12.2,17.7]]
        readout = out[0] + out[1] = [12.5,23.5,34.5]
    """
    mix = MHCMix(3, 2)
    with torch.no_grad():
        mix.alpha_pre.zero_()
        mix.alpha_post.zero_()
        mix.alpha_res.zero_()
        mix.b_pre.fill_(0.0)
        mix.b_post.fill_(0.0)
        ln4 = math.log(4.0)
        mix.b_res.copy_(torch.tensor([[0.0, -ln4], [-ln4, 0.0]]))

    state = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])  # [1,1,K,d]
    branch = torch.tensor([[[10.0, 20.0, 30.0]]])  # the block's write y
    c = 0.5
    lanes = state * c
    ctx = mix.pre_mix(lanes, convex=True, scale=1.0 / c)
    torch.testing.assert_close(ctx, torch.tensor([[[2.5, 3.5, 4.5]]]), atol=1e-6, rtol=0)

    out = mix(lanes, branch, post_sum=2.0 * c)
    want = torch.tensor([[[[5.8, 11.3, 16.8], [6.7, 12.2, 17.7]]]])
    torch.testing.assert_close(out, want, atol=1e-6, rtol=0)
    torch.testing.assert_close(mhc_readout(out, False), torch.tensor([[[12.5, 23.5, 34.5]]]))


# --- refusals and the OFF tree ---------------------------------------------


def test_mhc_streams_below_two_is_a_loud_refusal():
    with pytest.raises(LoudError) as e:
        _cfg(use_mhc=True, mhc_streams=1)
    assert "mhc_streams" in str(e.value) and "Escape:" in str(e.value)


def test_use_mhc_with_byte_patch_is_a_loud_refusal():
    with pytest.raises(LoudError) as e:
        _cfg(use_mhc=True, byte_patch=True, use_byteflow=True)
    msg = str(e.value)
    assert "use_mhc" in msg and "byte_patch" in msg and "Escape:" in msg


def test_use_mhc_without_a_lane_seam_is_a_loud_refusal():
    """The hybrid stack runs each block once and threads no lanes."""
    with pytest.raises(LoudError) as e:
        tiny_cfg(block_form="kda", block_stack="3:1", n_layers=4, max_iter=1, use_mhc=True)
    msg = str(e.value)
    assert "use_mhc" in msg and "Escape:" in msg


def test_every_mixer_parameter_moves_in_three_steps():
    """At init the arm IS the identity, so its gradients start small — but
    every parameter of the mixer must carry a finite, non-zero one after a real
    optimizer run (an inert scaffold is the defect this refuses)."""
    ids = _batch()
    model = _build(0, use_mhc=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = None
    for _ in range(3):
        opt.zero_grad()
        _, loss, _ = model(ids, (ids + 1) % 251)
        assert loss is not None
        loss.backward()
        opt.step()
    assert loss is not None and math.isfinite(float(loss.detach()))
    seen = 0
    for name, p in model.mhc_full.named_parameters():
        seen += 1
        g = p.grad
        assert g is not None, f"{name}: no gradient slot"
        assert bool(torch.isfinite(g).all()), name
        assert float(g.abs().max()) > 0.0, f"{name}: gradient is exactly zero"
    assert seen == 9, seen  # 3 φ + 3 α + b_pre/b_post/b_res
