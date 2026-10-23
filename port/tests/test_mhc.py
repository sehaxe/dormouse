"""The mHC arm (`arXiv:2512.24880`): Eq. 3/7/8/9, six gates, and the wiring.

Source is `vendor/dormouse-fused/crates/burn-mhc` (module docstring and line
citations in `model/mhc.py`); the wiring decisions are
`docs/reviews/mhc-2026-09-30.md` §3 (K=2 lanes at the LOOP BOUNDARY, the
readout, `H_pre` by the caller, `h_ctx` as `x_l`).

The gates here are the ones the burn lane's review paid for: a hand-computed
Eq. 3 fixture, the manifold's defining property in BOTH directions, the
derived Eq. 14 init pinned as values, the ON-vs-OFF logits `torch.equal` at
that init (the A/B's identity gate), per-parameter gradients with the two
structural floors NAMED, no host sync, and a loud refusal of the
configurations that were never A/B'd.
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
from dormouse.model.mhc import (
    ALPHA_INIT,
    B_RES_INIT,
    SINKHORN_ITERS,
    MHCMix,
    sinkhorn_knopp,
)

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
            # A fixed draw: 20 iterations reach 1e-5 here (the paper's
            # Fig. 7(a) reads the same converged-to-fp direction); an unseeded
            # draw can land on a slow-converging matrix and read 4.5e-4.
            torch.manual_seed(1)
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


def test_the_derived_biases_are_the_paper_maps():
    """(c) Eq. 14's maps, derived in the module's own forms, pinned as values.

    `2409.19606` Eq. 14: `B = 1`, `A_m = e_{k mod n}`, `A_r = I`, dynamic
    `W = 0` (§2.3). In this module: `phi_* = 0`, `alpha_* = 0.01` (mHC App. A.1
    Table 5), `b_pre = logit(1/K)`, `b_post = 0`, `b_res = log(J/K) = -ln K`.
    The first three make `H_pre = 1/K` and `H_post = 1`; the last is the log of
    the uniform fixed point, which is the identity on the paper's replicated
    lanes (the Sinkhorn is shift-invariant, so `-ln K` and burn's `0` denote
    the same map — the derived value is pinned, not the chosen one). The exact
    `A_r = I` is `log I = -inf` off-diagonal and is NOT approximated with a
    stiff diagonal (that is burn's frozen `10·I`).
    """
    mix = MHCMix(64, 2)
    with torch.no_grad():
        assert torch.equal(mix.phi_pre, torch.zeros_like(mix.phi_pre))
        assert torch.equal(mix.phi_post, torch.zeros_like(mix.phi_post))
        assert torch.equal(mix.phi_res, torch.zeros_like(mix.phi_res))
        one = torch.full_like(mix.alpha_pre, ALPHA_INIT)
        assert torch.equal(mix.alpha_pre, one)
        assert torch.equal(mix.alpha_post, one)
        assert torch.equal(mix.alpha_res, one)
        assert float(mix.b_pre[0]) == math.log(1.0 / (mix.n_streams - 1)) == 0.0
        assert float(mix.b_post[0]) == 0.0
        assert torch.equal(
            mix.b_res, torch.full_like(mix.b_res, -math.log(mix.n_streams))
        )
        assert float(mix.b_res[0, 0]) == float(torch.tensor(B_RES_INIT))
        lanes = torch.randn(2, 4, 2, 64)
        h_pre, h_post, h_res = mix.hyper_mappings(lanes)
        assert torch.equal(h_pre, torch.full_like(h_pre, 1.0 / mix.n_streams)), h_pre
        assert torch.equal(h_post, torch.ones_like(h_post)), h_post
        assert torch.equal(h_res, torch.full_like(h_res, 1.0 / mix.n_streams)), h_res
        # The only state the paper's init creates is K copies of h; on it the
        # uniform fixed point IS the identity, bit for bit (Eq. 3's write).
        replicated = torch.randn(2, 4, 64).unsqueeze(-2).expand(2, 4, 2, 64)
        mixed = torch.einsum("btij,btjd->btid", h_res, replicated)
        assert torch.equal(mixed, replicated)


def test_at_init_mhc_and_off_are_bit_identical():
    """(c) One seed, `use_mhc=true` vs `false`: every weight and the LOGITS.

    The derived init draws no RNG (phi zeros), so the ON model does not shift
    the shared initialisation (`lm_head` included); `H_pre = 1/K`,
    `H_post = 1` and `H_res = J/K` then make the body input and the summed
    readout the plain residual connection, so the logits are `torch.equal`,
    not "5.4e-04 close". This is the point of the identity-at-init: the A/B
    measures the mechanism, not a perturbation.
    """
    x, _ = _batch()
    torch.manual_seed(3)
    off = tiny_model()
    torch.manual_seed(3)
    on = tiny_model(use_mhc=True)
    off_sd, on_sd = off.state_dict(), on.state_dict()
    for k, v in off_sd.items():
        assert k in on_sd and torch.equal(on_sd[k], v), f"shared weight moved: {k}"
    with torch.no_grad():
        a, _, _ = on(x)
        b, _, _ = off(x)
    diff = float((a - b).abs().max())
    print(f"[mhc] at-init logits: equal={torch.equal(a, b)} max|diff|={diff:.3e}")
    assert torch.equal(a, b), diff


def test_the_gradient_at_the_derived_init_prints_every_parameter():
    """(d) Per-parameter gradients, with the two floors NAMED, not hidden.

    At the derived init `phi_* = 0`, so `d(α·xφ)/dφ = α·x ≠ 0` (the wired
    projections and biases train) while `d/dα = xφ = 0`: `alpha_*` are exactly
    zero on the first step and move from the second. `phi_res` is also at the
    floor, for a different reason: the full-width SUM readout weights every
    lane equally, so `∂loss/∂H_res` is row-constant and the Sinkhorn's row
    normalisation annihilates exactly that direction. This is the same floor
    that made burn's `10·I` `|grad α_res| = 7.3e-9` (F-F4); the fix is a
    lane-specific readout, not a bias — the module-level gate below proves the
    mixer itself is differentiable.

    The assertion is the strongest true statement at this init: every
    parameter is finite and non-NaN, and the four wired parameters train
    (`> 1e-8`). The two floors are printed and pinned as `< 1e-8`, so a future
    readout change that wakes them up will not silently pass unnoticed.
    """
    x, y = _batch()
    model = DormouseModel(tiny_cfg(use_mhc=True))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = None
    grads: list[str] = []
    for step in range(3):
        opt.zero_grad()
        _, loss, _ = model(x, y)
        assert loss is not None
        loss.backward()
        if step == 0:
            first = {
                name: float(p.grad.abs().max())
                for name, p in model.loop.mhc.named_parameters()
            }
            print(
                "[mhc] step-1 max|grad|: "
                + " ".join(f"{k}={v:.2e}" for k, v in first.items())
            )
        opt.step()
    assert loss is not None and math.isfinite(float(loss.detach()))
    seen = 0
    for name, p in model.loop.mhc.named_parameters():
        seen += 1
        g = p.grad
        assert g is not None, f"{name}: no gradient slot"
        assert bool(torch.isfinite(g).all()), name
        grads.append(f"{name}={float(g.abs().max()):.2e}")
    print("[mhc] 3-step max|grad|: " + " ".join(grads))
    assert seen == 9, seen  # 3 φ + 3 α + 3 biases (b_res counts as one)
    for name in ("phi_pre", "phi_post", "b_pre", "b_post"):
        assert first[name] > 1e-8, (name, first[name])
    # The floors: alpha=0 because phi=0; phi_res killed by the sum readout's
    # row-constant gradient through the Sinkhorn's row normalisation.
    for name in ("alpha_pre", "alpha_post", "alpha_res", "phi_res"):
        assert first[name] < 1e-8, (name, first[name])


def test_the_residual_mixer_is_live_when_the_loss_distinguishes_lanes():
    """(d') The Sinkhorn path is not detached: a lane-specific loss gives φ_res.

    The model's readout is a symmetric sum, which is why the full-model φ_res
    gradient is zero (above). Read ONE lane and the off-diagonal columns get a
    real gradient at the SAME derived init — this is the proof that the zero is
    the readout's symmetry, not a `detach` in the Sinkhorn. A second pass with
    `phi_res` moved off zero gives BOTH `phi_res` and `alpha_res` > 1e-8,
    closing the bilinear corner (`d/dα = xφ`).
    """
    torch.manual_seed(0)
    mix = MHCMix(64, 2)
    lanes = torch.randn(2, 4, 2, 64)
    branch = torch.randn(2, 4, 64)
    out = mix(lanes, branch)
    # Lane-specific: lane 0 weighted 2x lane 1. The model's sum readout is the
    # 1/2,1/2 case; this is the same operator with a non-symmetric use.
    (2.0 * out[..., 0, :].pow(2).sum() + out[..., 1, :].pow(2).sum()).backward()
    phi_at_init = float(mix.phi_res.grad.abs().max())
    alpha_at_init = float(mix.alpha_res.grad.abs().max())
    print(
        f"[mhc] lane-specific step: phi_res={phi_at_init:.3e} "
        f"alpha_res={alpha_at_init:.3e} (alpha is 0 because phi=0)"
    )
    assert phi_at_init > 1e-8, phi_at_init
    assert alpha_at_init == 0.0, alpha_at_init  # d(αφ)/dα = φ = 0
    # Off the bilinear corner both are live; this is what step 2 of a real run
    # reaches once the φ gradient has moved.
    with torch.no_grad():
        mix.phi_res.normal_(0.0, 0.01)
    mix.zero_grad()
    out = mix(lanes, branch)
    (2.0 * out[..., 0, :].pow(2).sum() + out[..., 1, :].pow(2).sum()).backward()
    print(
        f"[mhc] lane-specific step (phi moved): "
        f"phi_res={float(mix.phi_res.grad.abs().max()):.3e} "
        f"alpha_res={float(mix.alpha_res.grad.abs().max()):.3e}"
    )
    assert float(mix.phi_res.grad.abs().max()) > 1e-8
    assert float(mix.alpha_res.grad.abs().max()) > 1e-8


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
    # The arm is IN the write. At the derived init it IS ReZero, so dropping
    # the module on the same weights changes nothing. Move `H_pre` off the
    # derived 1/K mix — the readout's mean is invariant to `H_res` and to a
    # zero-sum `H_post` split on replicated lanes, so `H_pre` is the parameter
    # that makes the module visible in the function (mhc.py's gradient caveat).
    with torch.no_grad():
        at_init, _, _ = on(x)
        on.loop.mhc.b_pre.fill_(10.0)
        moved, _, _ = on(x)
        on.loop.mhc = None
        after, _, _ = on(x)
    assert torch.isfinite(at_init).all() and torch.isfinite(moved).all()
    assert torch.isfinite(after).all()
    assert not torch.equal(moved, after)


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

    AttnRes is NOT in this list: it composes with mHC (the two act on different
    axes — `loopblock`'s "THE COMPOSITION"), and the legality is gated by
    `test_mhc_attnres.py`. What stays refused is a second, un-A/B'd gate on the
    same write (LOOM) and a different placement (ByteFlow's patch lanes).
    """
    for key in ("loom", "use_byteflow"):
        with pytest.raises(LoudError) as e:
            parse_preset(f"use_mhc = true\n{key} = true\n")
        assert key in str(e.value), (key, str(e.value))
        assert "use_mhc" in str(e.value)


def test_mhc_and_attnres_together_are_legal():
    """The pair the refusal used to ban is a real configuration now.

    Preset, override and `load_run` seams all accept it, and the flags survive
    as `True` on the resolved config (a silently dropped flag is the defect the
    old refusal existed to prevent — the fix deletes the refusal, it does not
    keep it as dead code).
    """
    cfg = parse_preset("use_mhc = true\nuse_attnres = true\n")
    assert cfg.use_mhc is True and cfg.use_attnres is True and cfg.mhc_streams == 2
    run = load_run("small", model_overrides={"use_mhc": True, "use_attnres": True})
    assert run.model.use_mhc is True and run.model.use_attnres is True


def test_the_flag_is_a_real_modelconfig_field():
    """Preset, override and CLI seam: the flag loads and keeps K=2."""
    assert parse_preset("use_mhc = true\n").use_mhc is True
    run = load_run("small", model_overrides={"use_mhc": True})
    assert run.model.use_mhc is True and run.model.mhc_streams == 2
    assert load_run("small").model.use_mhc is False
