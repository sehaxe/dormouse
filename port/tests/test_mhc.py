"""The mHC arm (`arXiv:2512.24880`): Eq. 3/7/8/9, six gates, and the wiring.

Source is `vendor/dormouse-fused/crates/burn-mhc` (module docstring and line
citations in `model/mhc.py`); the wiring decisions are
`docs/reviews/mhc-2026-09-30.md` §3 (K=2 lanes at the LOOP BOUNDARY, the
readout, `H_pre` by the caller, `h_ctx` as `x_l`).

The gates here are the ones the burn lane's review paid for: a hand-computed
Eq. 3 fixture (the red→green table is only worth its numbers if the test was
run), the manifold's defining property in BOTH directions, the near-identity
at init as a PRINTED number, per-parameter gradient flow (not "the loss has a
gradient"), no host sync, and a loud refusal of the configurations that were
never A/B'd.
"""

from __future__ import annotations

import hashlib
import math

import pytest
import torch
from conftest import BATCH, SEQ, tiny_cfg, tiny_model

from dormouse.config import ModelConfig, load_run, parse_preset
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.model.mhc import SINKHORN_ITERS, MHCMix, sinkhorn_knopp

#: `tiny_cfg()`'s parameter count at HEAD before this port: the pin that the
#: OFF arm builds today's tree exactly (mHC adds 1 035 here: 1 024 φ + 11
#: biases/gates; `test_off_...` asserts the delta too).
TINY_PARAMS = 64_881
TINY_MHC_PARAMS = 1_035


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    return ids, (ids + 1) % 251


def _sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def test_eq3_on_the_burn_lanes_hand_fixture():
    """Eq. 3 on four numbers, with α=0 so the operator is the static biases.

    The crate's fixture (`burn-mhc/src/lib.rs:347-451`) on OUR lane layout:
    `b_res = [[0, -ln4], [-ln4, 0]]` → `Sinkhorn(exp(·)) = [[.8, .2], [.2, .8]]`
    (cross ratio 16), `b_post = [-.5, 1.5]` → `H_post = [0.75508134, 1.63514895]`,
    lanes `[[1,2],[3,4]]`, branch `[10,20]` broadcast to each lane by Eq. 3's
    `H_postᵀ F`. Lane 0 is the crate's lane-0 row (the branch was broadcast,
    not sliced — `mhc.py`'s layout note), lane 1 is the same arithmetic with
    `H_post[1]`.

    The separations the crate's table pins, on `out[0,0]`: as printed 8.950813;
    `H_post = σ(·)` (the factor 2 dropped) 5.175407 (−42.2 %); `H_res = b_res`
    (Sinkhorn dropped) 3.391930 (−62.1 %); `H_post` off by one lane 17.751490
    (+98.3 %). A "the output moved" gate is green under all four.
    """
    mix = MHCMix(2, 2)
    with torch.no_grad():
        mix.alpha_pre.zero_()
        mix.alpha_post.zero_()
        mix.alpha_res.zero_()
        mix.b_post.copy_(torch.tensor([-0.5, 1.5]))
        ln4 = math.log(4.0)
        mix.b_res.copy_(torch.tensor([[0.0, -ln4], [-ln4, 0.0]]))
    lanes = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # [1,1,n=2,d=2]
    branch = torch.tensor([[[10.0, 20.0]]])  # F(H_pre x_l), [1,1,d]
    out = mix(lanes, branch).detach()
    want = torch.tensor(
        [[[[8.95081338, 17.50162675], [18.95149028, 36.30295810]]]]  # row-major
    )
    assert torch.allclose(out, want, atol=1e-5), (out, want)
    # The projection itself, not only its effect: the whole point is H_res.
    _, _, h_res = mix.hyper_mappings(lanes)
    assert torch.allclose(h_res, torch.tensor([[[[0.8, 0.2], [0.2, 0.8]]]]), atol=1e-5), h_res
    assert SINKHORN_ITERS == 20


def test_sinkhorn_is_doubly_stochastic_and_non_negative():
    """Eq. 6/9: row AND column sums 1 to 1e-5, every entry >= 0.

    The last normalization of Eq. 9 is a column pass, so the COLUMNS are exact
    to fp and the rows are the converged direction (the paper's Fig. 7(a)); we
    assert both at the module's real `H_res`, at init and with the static bias
    moved off it, and print the two maxima.
    """
    mix = MHCMix(8, 2)
    streams = torch.randn(2, 3, 2, 8)
    for label in ("init", "perturbed"):
        if label == "perturbed":
            with torch.no_grad():
                mix.b_res.normal_(0.0, 1.0)
        _, _, h_res = mix.hyper_mappings(streams)
        row = (h_res.detach().sum(-1) - 1.0).abs().max()
        col = (h_res.detach().sum(-2) - 1.0).abs().max()
        print(f"[mhc] H_res {label}: max|rowsum-1|={row:.3e} max|colsum-1|={col:.3e}")
        assert float(row) < 1e-5, (label, float(row))
        assert float(col) < 1e-5, (label, float(col))
        assert bool((h_res >= 0.0).all()), label
    # The log-domain form is what keeps extreme logits finite: the linear form
    # overflows exp(120) and NaNs a zero column (`sinkhorn.rs:8-21`).
    extreme = torch.full((1, 1, 2, 2), 120.0)
    extreme[..., 0, 1] = -120.0
    extreme[..., 1, 0] = -120.0
    m = sinkhorn_knopp(extreme)
    assert bool(torch.isfinite(m).all()) and bool((m >= 0).all())
    assert torch.allclose(m.sum(-2), torch.ones_like(m.sum(-2)), atol=1e-5)


def test_off_is_bit_identical_to_today():
    """(a) `use_mhc=false` is today's model: same tree, sha, params, logits."""
    assert ModelConfig().use_mhc is False
    assert ModelConfig().mhc_streams == 2
    x, _ = _batch()
    torch.manual_seed(3)
    today = tiny_model()
    torch.manual_seed(3)
    off = tiny_model(use_mhc=False)
    assert param_count(today) == param_count(off) == TINY_PARAMS
    assert _sha(today) == _sha(off)
    assert not [k for k in off.state_dict() if k.startswith("loop.mhc.")]
    with torch.no_grad():
        a, _, _ = today(x)
        b, _, _ = off(x)
    assert torch.equal(a, b)
    # And the ON arm is exactly the module's count, so the delta is the arm.
    on = tiny_model(use_mhc=True)
    assert param_count(on) - param_count(off) == TINY_MHC_PARAMS
    assert not [k for k, _ in off.named_parameters() if k.startswith("loop.mhc.")]
    assert "loop.mhc.b_res" in dict(on.named_parameters())


def test_at_init_the_mix_is_rezero_at_scale_one():
    """(c) At init the operator writes `h + y`; the distance is printed.

    `b_res = 0` makes `H_res` UNIFORM (not the identity — the 2026-10-02
    fidelity fix), and uniform mixing is the identity on replicated lanes
    (`x_l = vec(h0)`, Eq. 3); `b_pre = 10` → `H_pre ≈ 1`, `b_post = 0` →
    `H_post ≈ 1`, so the write is ReZero at `residual_scale = 1` to within the
    α=0.01 dynamic term. This is the crate's gate (`mhc-2026-09-30.md` §5.3),
    with its relative bounds: `< 1e-2` so the arms start as the same function,
    `> 1e-6` so an arm that is BITWISE ReZero (the projection not in the write
    at all) fails.
    """
    torch.manual_seed(0)
    mix = MHCMix(64, 2)
    h = torch.randn(2, 4, 64)
    lanes = h.unsqueeze(-2).expand(2, 4, 2, 64)
    branch = torch.randn(2, 4, 64)
    out = mix(lanes, branch)
    want = lanes + branch.unsqueeze(-2)  # ReZero at scale 1, per lane
    rel = float(((out - want).norm() / want.norm()).detach())
    print(f"[mhc] at-init relative write distance from ReZero@1: {rel:.3e}")
    assert 1e-6 < rel < 1e-2, rel


def test_every_mhc_parameter_gets_a_finite_non_zero_gradient_in_three_steps():
    """(d) All nine parameters, per parameter, after a real 3-step run.

    Per-parameter because an aggregate hides a detached arm (the `8fa5d4c`
    shape). `b_pre = 10` sits on σ's flat top — its gradient is ~1e-7 and must
    still be NON-ZERO; `phi_res`'s path runs through the Sinkhorn, which is
    exactly where a silent `detach` would hide.
    """
    x, y = _batch()
    model = DormouseModel(tiny_cfg(use_mhc=True))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = None
    for _ in range(3):
        opt.zero_grad()
        _, loss, _ = model(x, y)
        assert loss is not None
        loss.backward()
        opt.step()
    assert loss is not None and math.isfinite(float(loss.detach()))
    seen = 0
    grads: list[str] = []
    for name, p in model.loop.mhc.named_parameters():
        seen += 1
        g = p.grad
        assert g is not None, f"{name}: no gradient slot"
        assert bool(torch.isfinite(g).all()), name
        assert float(g.abs().max()) > 0.0, f"{name}: gradient is exactly zero"
        grads.append(f"{name}={float(g.abs().max()):.2e}")
    print("[mhc] 3-step max|grad|: " + " ".join(grads))
    assert seen == 9, seen  # 3 φ + 3 α + 3 biases (b_res counts as one)


def test_the_forward_never_reads_a_device_scalar(monkeypatch):
    """(e) No host sync: `.item()`, `.tolist()` and `int(tensor)` are refused.

    Runs the mHC module and the loop's forward with those tensor methods
    monkeypatched to raise. A grep alone would miss a helper call; a patch
    alone would miss a branch the fixture does not take, so the source is
    scanned too (AST-free: the three names cannot appear in this module).
    """
    import dormouse.model.mhc as mhc_mod

    src = open(mhc_mod.__file__, encoding="utf-8").read()
    assert ".item(" not in src and ".tolist(" not in src
    # `int(`/`float(` in this module only ever take Python ints (`int(d_model)`);
    # a tensor would go through `__int__`/`__float__`, which the patch arms.

    def boom(*_a, **_k):
        raise AssertionError("host sync in the mHC forward")

    for name in ("item", "tolist", "__int__", "__float__"):
        monkeypatch.setattr(torch.Tensor, name, boom, raising=True)
    x, _ = _batch()
    # The DENSE expert blend (`moe_topk=0`): the routed arm's CPU reference
    # dispatch legitimately reads `torch.unique(...).tolist()` for its loop, and
    # that is the EXISTING arm, not mHC. The gate is the mHC path end to end.
    model = DormouseModel(tiny_cfg(use_mhc=True, moe_topk=0))
    with torch.no_grad():
        model(x)
        model.loop(
            torch.randn(BATCH, SEQ, tiny_cfg().d_model),
            None,
            None,
            None,
            counters=Counters(),
        )


def test_wiring_counts_one_mhc_write_per_iteration():
    """(f)+wiring: `mhc=<iters>` when on, the key absent when off."""
    x, _ = _batch()
    on, off = tiny_model(use_mhc=True), tiny_model()
    c_on, c_off = Counters(), Counters()
    with torch.no_grad():
        on(x, counters=c_on)
        off(x, counters=c_off)
    assert c_on["mhc"] == on.cfg.max_iter == c_on["iter"]
    assert "mhc" not in c_off
    assert c_off["rezero_write"] == off.cfg.max_iter
    # The arm is IN the write: dropping the module (not rebuilding the model)
    # falls back to ReZero on the same weights and moves the logits.
    with torch.no_grad():
        before, _, _ = on(x)
        on.loop.mhc = None
        after, _, _ = on(x)
    assert torch.isfinite(before).all() and torch.isfinite(after).all()
    assert not torch.equal(before, after)


def test_mhc_streams_other_than_two_is_a_loud_refusal():
    """(f) K != 2 is LOUD, on and off; only the A/B'd configuration ships."""
    for text in ("use_mhc = true\nmhc_streams = 4\n", "mhc_streams = 4\n"):
        with pytest.raises(LoudError) as e:
            parse_preset(text)
        assert "mhc_streams" in str(e.value)
    with pytest.raises(LoudError) as e:
        load_run("small", model_overrides={"use_mhc": True, "mhc_streams": 4})
    assert "only K=2 ships" in str(e.value)
    with pytest.raises(LoudError) as e:
        parse_preset("use_mhc = true\nmhc_streams = 0\n")
    assert "mhc_streams" in str(e.value)


def test_refuses_the_arms_that_share_the_write_or_placement():
    """The combos that break or confound the contract are refused, not merged.

    `use_attnres` is NOT in this list: it composes with mHC on the same write
    (mHC owns the lanes, AttnRes the update) and is covered by
    `test_the_pair_composes_through_preset_override_and_load_run` below.
    """
    for key in ("loom", "use_byteflow"):
        with pytest.raises(LoudError) as e:
            parse_preset(f"use_mhc = true\n{key} = true\n")
        assert key in str(e.value), (key, str(e.value))
        assert "use_mhc" in str(e.value)
        # The legal partner is named, so a reader hitting the refusal has an
        # escape rather than a dead end.
        assert "use_attnres" in str(e.value), str(e.value)


def test_the_pair_composes_through_preset_override_and_load_run():
    """`use_mhc` + `use_attnres` is a legal configuration at every seam."""
    cfg = parse_preset("use_mhc = true\nuse_attnres = true\n")
    assert cfg.use_mhc is True and cfg.use_attnres is True and cfg.mhc_streams == 2
    run = load_run("small", model_overrides={"use_mhc": True, "use_attnres": True})
    assert run.model.use_mhc is True and run.model.use_attnres is True
    text = run.snapshot_toml()
    assert "use_mhc = true" in text and "use_attnres = true" in text


def test_the_flag_is_a_real_modelconfig_field():
    """Preset, override and CLI seam: the flag loads and keeps K=2."""
    assert parse_preset("use_mhc = true\n").use_mhc is True
    run = load_run("small", model_overrides={"use_mhc": True})
    assert run.model.use_mhc is True and run.model.mhc_streams == 2
    assert load_run("small").model.use_mhc is False
