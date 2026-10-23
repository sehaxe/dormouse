"""The composition `use_mhc + use_attnres` — eight gates on one forward.

Both arms act on the loop-boundary residual statement, on different axes: mHC
owns the K-lane STATE (`H_pre` reads it, `H_post`/`H_res` write it, readout =
`Σ lanes / K`), AttnRes owns WHICH UPDATE the write carries. The order (and
the equation each step implements) is in `loopblock.py`'s module docstring,
"THE COMPOSITION":

    x_l = x + e_i                                     # the crate's h_ctx
    ctx = H_pre x_l                                   # Eq. 3's H_pre term
    y   = Attn + Engram + MoE                         # the body, unchanged
    a   = Σ_j w_j u_j,  w = S·softmax(q_i·RMSNorm(u)) # AttnRes over UPDATES
          u_0 = 0 (embedding update: H_res already carries it), u_1 = y
    x'  = H_res x_l + H_postᵀ a                       # Eq. 3, unchanged
    out = Σ x' / K                                    # 2409.19606 §2.1

The `S·softmax` lever is what makes the pair's at-init forward the plain
pre-norm residual path (`S = |U| = 2`, q_i = 0 -> every w_j = 1 -> a = y),
which is the same identity argument `mhc.py`'s Eq. 14 init makes on the mHC
side. Gates, in order:

1. `off/off` is bit-identical to pristine HEAD `aa28a8b` (pinned sha of
   `state_dict` + logits on a fixed tiny fixture).
2. `mhc-only` and `attnres-only` are bit-identical to the PRE-composition base
   (`a25f7b9`, the mHC lane's Eq. 14 commit) — the composition must not
   perturb either single-arm path.
3. both arms build, both code paths run in the SAME forward (function spies),
   the counters fire once/iteration and once/forward, and every parameter of
   BOTH mechanisms gets a finite, non-zero gradient (named in the failure).
4. at the derived mHC init + the AttnRes zero query the composition is the
   plain residual path, `torch.equal` against the ReZero arm (printed 0.0).
5. removing either mechanism from the trained combination moves the logits
   (neither is dead code) — the numbers are printed.
6. the LOUD refusals still name their escape; the old `use_mhc`+`use_attnres`
   refusal is DELETED, not left as dead code (source scan + legality).
7. no host sync in the composition's forward (`.item`/`.tolist`/`int`/`float`
   on a tensor all raise, and the source cannot contain the first two).
8. config: the pair is legal through preset, override and `load_run`, and the
   old refusal's text is gone from `config.py`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math

import pytest
import torch
from conftest import tiny_cfg, tiny_model

import dormouse.model.loopblock as loopblock_mod
from dormouse.config import ModelConfig, load_run, parse_preset
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.model.loopblock import AttnResSlot, depth_attend
from dormouse.model.mhc import MHCMix

BATCH, SEQ = 2, 64
SEED = 1234

TINY_PARAMS = 64_881
MHC_PARAMS = 1_035  # 3 φ [2d·n] + 3 α + b_pre/b_post/b_res
ATTNRES_PARAMS = 256  # max_iter 2 × (query + gain) × d 64
BOTH_PARAMS = TINY_PARAMS + MHC_PARAMS + ATTNRES_PARAMS

#: sha256(state_dict keys+bytes | logits bytes), the fixture below, seed 1234.
#: `OFF_OFF_SHA` was measured on PRISTINE HEAD (aa28a8b, before the mHC lane's
#: port landed in this tree); the two single-arm hashes on `a25f7b9`, the
#: commit this feature branched from.
OFF_OFF_SHA = "beb23f76773c5f63a8b5ed5a549fb3c451174e9f4622c2549246cc63b4d0d318"
MHC_ONLY_SHA = "a53a8d57f3301559f8608a75116ada2c59d7401aab762e30556d9e3d17e8ced9"
ATTNRES_ONLY_SHA = "c11ecd3780c1882b25c4f859e575251f42ad15f08365088b263c49a1b6682c2e"
BOTH_SHA = "803fc9daea1a757b566ee3f0754ed7d1197ae6b71e1b20953f45d308234c3330"


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    return ids, (ids + 1) % 251


def _sha(model: torch.nn.Module, logits: torch.Tensor) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    h.update(b"|logits|")
    h.update(logits.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _pinned(over: dict) -> str:
    torch.manual_seed(SEED)
    model = DormouseModel(ModelConfig(**{**dataclasses.asdict(tiny_cfg()), **over}))
    with torch.no_grad():
        logits, _, _ = model(_batch()[0])
    return _sha(model, logits)


def _train_steps(model: DormouseModel, steps: int = 3) -> float:
    """A real optimizer run — the arms must MOVE for "removing it" to mean anything."""
    x, y = _batch()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        opt.zero_grad()
        _, loss, _ = model(x, y)
        assert loss is not None
        loss.backward()
        opt.step()
    return float(loss.detach())


# --- 1. the OFF arm ---------------------------------------------------------


def test_off_off_is_bit_identical_to_pristine_head():
    """(1) Neither flag: the pinned hash of pristine HEAD `aa28a8b`."""
    assert ModelConfig().use_mhc is False and ModelConfig().use_attnres is False
    assert _pinned({}) == OFF_OFF_SHA
    torch.manual_seed(SEED)
    model = tiny_model()
    assert param_count(model) == TINY_PARAMS
    assert not [k for k in model.state_dict() if k.startswith(("loop.mhc.", "loop.attnres."))]


# --- 2. each single arm -----------------------------------------------------


def test_each_single_arm_keeps_its_pre_composition_hash():
    """(2) mHC-only and AttnRes-only are bit-identical to the branch point.

    The hash covers the parameter TREE as well as the logits, so a new module,
    a renamed buffer or an extra multiply on either path is red — "the
    composition only runs when both flags are on" is not a comment here.
    """
    assert _pinned({"use_mhc": True}) == MHC_ONLY_SHA
    assert _pinned({"use_attnres": True}) == ATTNRES_ONLY_SHA
    assert param_count(tiny_model(use_mhc=True)) == TINY_PARAMS + MHC_PARAMS
    assert param_count(tiny_model(use_attnres=True)) == TINY_PARAMS + ATTNRES_PARAMS


# --- 3. both, in one forward: spies, counters, gradients --------------------


def test_both_arms_run_train_and_count_in_one_forward(monkeypatch):
    """(3) One forward runs BOTH paths; three steps give both arms gradients.

    The spies wrap the two functions the composition must call — the AttnRes
    mixture (`depth_attend`) and mHC's Eq. 3 write (`MHCMix.forward`) — and
    count them in the SAME forward. The counters are the on-log evidence:
    `mhc_attnres_forward` once per window/forward, the write counters once per
    iteration (so `max_iter` on the tiny config).

    The gradient half is per PARAMETER, not an aggregate (an aggregate hides a
    detached arm): all four AttnRes vectors are non-zero after three steps.
    mHC's `phi_res`/`alpha_res`/`b_res` sit on the symmetric-sum-readout floor
    (`mhc.py`'s "Gradient caveat"), so that symmetry is broken the way the mHC
    lane's own gate breaks it — `phi_res` moved off its derived zero — and all
    nine are then asserted non-zero. Every name is in the failure message.
    """
    x, y = _batch()
    model = tiny_model(use_mhc=True, use_attnres=True)
    assert param_count(model) == BOTH_PARAMS
    assert model.loop.mhc is not None and model.loop.attnres is not None

    calls = {"attnres": 0, "mhc": 0}
    real_attend, real_mhc = depth_attend, MHCMix.forward

    def spy_attend(*a, **k):
        calls["attnres"] += 1
        return real_attend(*a, **k)

    def spy_mhc(self, *a, **k):
        calls["mhc"] += 1
        return real_mhc(self, *a, **k)

    monkeypatch.setattr(loopblock_mod, "depth_attend", spy_attend)
    monkeypatch.setattr(MHCMix, "forward", spy_mhc)

    counters = Counters()
    _, _, counters = model(x, counters=counters)
    assert calls == {"attnres": model.cfg.max_iter, "mhc": model.cfg.max_iter}, calls
    assert counters["mhc_attnres_forward"] == 1, dict(counters)
    assert counters["mhc_attnres_write"] == model.cfg.max_iter, dict(counters)
    assert counters["attnres_write"] == model.cfg.max_iter, dict(counters)
    assert counters["mhc"] == model.cfg.max_iter, dict(counters)
    assert "rezero_write" not in counters
    print(
        f"[both] one forward: depth_attend x{calls['attnres']} "
        f"MHCMix.forward x{calls['mhc']} mhc_attnres_forward="
        f"{counters['mhc_attnres_forward']} writes={counters['mhc_attnres_write']}"
    )

    monkeypatch.setattr(loopblock_mod, "depth_attend", real_attend)
    monkeypatch.setattr(MHCMix, "forward", real_mhc)
    loss = _train_steps(model, 3)
    assert math.isfinite(loss)

    for name, p in model.loop.attnres.named_parameters():
        g = p.grad
        assert g is not None, f"attnres.{name}: no gradient slot"
        assert bool(torch.isfinite(g).all()), f"attnres.{name}: non-finite gradient"
        assert float(g.abs().max()) > 0.0, f"attnres.{name}: gradient is exactly zero"

    # Break the lane symmetry the sum readout imposes (mhc.py's caveat), then
    # require EVERY mHC parameter non-zero — the same treatment as
    # `test_the_residual_mixer_is_live_when_the_loss_distinguishes_lanes`.
    with torch.no_grad():
        model.loop.mhc.phi_res.normal_(0.0, 0.1)
    model.zero_grad()
    _, loss, _ = model(x, y)
    assert loss is not None
    loss.backward()

    grads: list[str] = []
    for name, p in model.loop.mhc.named_parameters():
        g = p.grad
        assert g is not None, f"mhc.{name}: no gradient slot"
        assert bool(torch.isfinite(g).all()), f"mhc.{name}: non-finite gradient"
        assert float(g.abs().max()) > 0.0, f"mhc.{name}: gradient is exactly zero"
        grads.append(f"{name}={float(g.abs().max()):.2e}")
    print(
        "[both] 3-step max|grad|: "
        + " ".join(
            f"attnres.{n}={float(p.grad.abs().max()):.2e}"
            for n, p in model.loop.attnres.named_parameters()
        )
        + " | mhc."
        + " mhc.".join(grads)
    )


# --- 4. identity at init ----------------------------------------------------


def test_the_composition_is_the_plain_residual_path_at_init():
    """(4) Derived mHC init + AttnRes zero query == ReZero, `torch.equal`.

    The derived mHC init is `b_pre = ln(1/(K-1))`, `b_post = 0`,
    `b_res = -ln K` (`mhc.py`); AttnRes is `query = 0`, `gain = 1`. Then
    `H_pre = 1/K`, `H_post = 1`, `H_res = J/K`, and the composition's write is

        w = 2·softmax([0, 0]) = [1, 1],  a = 1·0 + 1·y = y,

    so `x' = x + y` and the readout is `h + y`: the plain pre-norm residual
    path, bit for bit. The state_dicts are compared key by key too, so the
    claim is "the same function on the same weights", not "two runs that
    happened to land close".
    """
    x, _ = _batch()
    torch.manual_seed(3)
    off = tiny_model()
    torch.manual_seed(3)
    both = tiny_model(use_mhc=True, use_attnres=True)
    off_sd, both_sd = off.state_dict(), both.state_dict()
    for k, v in off_sd.items():
        assert k in both_sd and torch.equal(both_sd[k], v), f"shared weight moved: {k}"
    with torch.no_grad():
        a, _, _ = off(x)
        b, _, _ = both(x)
    diff = float((a - b).abs().max())
    print(
        f"[both] at-init logits: torch.equal={torch.equal(a, b)} max|diff|={diff:.3e} "
        "(bound: exact equality)"
    )
    assert torch.equal(a, b), diff
    # The lever itself, on four numbers: zero-query weights are 1, the zero
    # update contributes exactly nothing, the body update passes exactly.
    slot = AttnResSlot(64)
    y = torch.randn(1, 1, 64)
    got = depth_attend([torch.zeros_like(y), y], slot, tiny_cfg().norm_eps, scale=2.0)
    assert torch.equal(got, y), (got - y).abs().max()


# --- 5. neither mechanism is dead code --------------------------------------


def test_removing_either_mechanism_changes_the_logits():
    """(5) Trained combination: drop mHC, drop AttnRes, the logits MOVE.

    At init AttnRes is the identity (gate 4), so "removing it changes nothing"
    would be true there by construction; after three steps the query has moved
    and the write is no longer `y`. Both removals are done on the SAME trained
    weights — the mechanism is removed, not replaced by a fresh model.
    """
    x, _ = _batch()
    torch.manual_seed(7)
    model = tiny_model(use_mhc=True, use_attnres=True)
    _train_steps(model, 3)
    with torch.no_grad():
        both, _, _ = model(x)
        model.loop.use_attnres = False  # -> mHC-only on the same weights
        no_ar, _, _ = model(x)
        model.loop.use_attnres = True
        model.loop.mhc = None  # -> AttnRes-only on the same weights
        no_mhc, _, _ = model(x)
    d_ar = float((both - no_ar).abs().max())
    d_mhc = float((both - no_mhc).abs().max())
    print(f"[both] remove AttnRes: max|dlogit|={d_ar:.3e}; remove mHC: max|dlogit|={d_mhc:.3e}")
    assert torch.isfinite(no_ar).all() and torch.isfinite(no_mhc).all()
    assert not torch.equal(both, no_ar), d_ar
    assert not torch.equal(both, no_mhc), d_mhc
    assert d_ar > 0.0 and d_mhc > 0.0


# --- 6. the refusals that remain, and the one that is gone ------------------


def test_loud_refusals_name_the_escape_and_the_legal_partner():
    """(6) LOOM/ByteFlow stay refused; the message names escape + legal partner."""
    for key in ("loom", "use_byteflow"):
        with pytest.raises(LoudError) as e:
            parse_preset(f"use_mhc = true\n{key} = true\n")
        msg = str(e.value)
        assert key in msg and "use_mhc" in msg
        assert "Escape:" in msg and "use_attnres" in msg, msg
        assert "scripts/ab_mhc_attnres.py" in msg, msg
    # The old pair refusal's own words must not survive anywhere in the source.
    src = open(__import__("dormouse.config", fromlist=["x"]).__file__, encoding="utf-8").read()
    assert "two write operators in one run" not in src
    assert "one experiment with two arms" not in src


# --- 7. no host sync --------------------------------------------------------


def test_the_composition_forward_reads_no_device_scalar(monkeypatch):
    """(7) `.item`/`.tolist`/`int(tensor)`/`float(tensor)` all raise.

    The dense CPU expert blend (`moe_topk=0`) is used so the routed arm's own
    `unique(...).tolist()` — an existing CPU reference, not this composition —
    is out of the path. The source scan catches a branch the fixture misses.
    """
    src = open(loopblock_mod.__file__, encoding="utf-8").read()
    assert ".item(" not in src and ".tolist(" not in src

    def boom(*_a, **_k):
        raise AssertionError("host sync in the composition forward")

    for name in ("item", "tolist", "__int__", "__float__"):
        monkeypatch.setattr(torch.Tensor, name, boom, raising=True)
    x, _ = _batch()
    model = tiny_model(use_mhc=True, use_attnres=True, moe_topk=0)
    with torch.no_grad():
        model(x)
        model.loop(
            torch.randn(BATCH, SEQ, tiny_cfg().d_model),
            None,
            None,
            None,
            counters=Counters(),
        )


# --- 8. config: legal everywhere, and the old refusal is deleted ------------


def test_the_pair_is_legal_through_preset_override_and_load_run():
    """(8) The composition is a real configuration at every seam."""
    cfg = parse_preset("use_mhc = true\nuse_attnres = true\n")
    assert cfg.use_mhc is True and cfg.use_attnres is True and cfg.mhc_streams == 2
    run = load_run("small", model_overrides={"use_mhc": True, "use_attnres": True})
    assert run.model.use_mhc is True and run.model.use_attnres is True
    # The snapshot a resumed run would carry keeps BOTH keys.
    text = run.snapshot_toml()
    assert "use_mhc = true" in text and "use_attnres = true" in text
