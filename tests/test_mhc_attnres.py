"""The composition `use_mhc + use_attnres` — nine gates on one forward.

Both arms act on the loop-boundary residual statement, on different axes: mHC
owns the K-lane STATE (`H_pre` reads it, `H_post`/`H_res` write it), AttnRes
owns WHICH UPDATE the write carries. The order (and the equation each step
implements) is in `loopblock.py`'s module docstring, "THE COMPOSITION":

    x_l = x + e_i                                     # the crate's h_ctx
    ctx = H_pre x_l                                   # Eq. 3's H_pre term
    y   = Attn + Engram + MoE                         # the body, unchanged
    a   = Σ_j w_j u_j,  w = S·softmax(q_i·RMSNorm(u)) # AttnRes over UPDATES
          u_0 = 0 (embedding update: H_res already carries it), u_1 = y
    x'  = H_res x_l + H_postᵀ a                       # Eq. 3, unchanged
    out = Σ x' / K                                    # 2409.19606 §2.1

The `S·softmax` lever is the identity at zero query (`S = |U| = 2`, every
`w_j = 1`, so `a = y`): the composition's at-init write is the mHC-only write
bit-for-bit. That is the identity this base supports — its mHC init is the
burn/`b_pre = 10` one (pinned by `test_mhc.py`), whose `H_pre ≈ K`, so an
exact equality to the PLAIN ReZero path is not available without changing the
mHC-only arm. The test prints that bound instead of hiding it.

Gates, in order:

1. `off/off` is bit-identical to `c8a2ff9`'s OFF path (pinned sha of
   `state_dict` + logits on a fixed tiny fixture).
2. `mhc-only` and `attnres-only` are bit-identical to `c8a2ff9`'s single-arm
   paths — the composition must not perturb either.
3. both arms build, both code paths run in the SAME forward (function spies),
   the counters fire once/forward and once/iteration, and every parameter of
   BOTH mechanisms gets a finite, non-zero gradient (named in the failure).
4. at init the AttnRes lever is exactly the identity (`torch.equal`) and the
   composition's update equals the mHC-only update; the full-model distance to
   ReZero is PRINTED because this base's mHC init is not ReZero.
5. removing either mechanism from the trained combination moves the logits
   (neither is dead code) — the numbers are printed.
6. the LOUD refusals still name their escape; the old `use_mhc`+`use_attnres`
   refusal is DELETED, not left as dead code (source scan + legality), and the
   config docstring names the composition.
7. the halt head still trains with BOTH mechanisms on: `halt_weight > 0`
   builds, `model.last_halt` is populated, and every halt parameter gets a
   finite gradient (the wiring this composition must not have displaced).
8. no host sync in the composition's forward (`.item`/`.tolist`/`int`/`float`
   on a tensor all raise, and the source cannot contain the first two).
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest
import torch
from conftest import tiny_cfg, tiny_model

import dormouse.model.loopblock as loopblock_mod
from dormouse.config import ModelConfig, parse_preset
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
#: Measured on `c8a2ff9` (the commit this branch is based on):
#: `OFF_OFF_SHA` = neither flag; the single-arm hashes = each arm alone.
OFF_OFF_SHA = "beb23f76773c5f63a8b5ed5a549fb3c451174e9f4622c2549246cc63b4d0d318"
MHC_ONLY_SHA = "09ab4fb246b64da9ce9384329e658e7655fc1d8f685aade847f6be62403704a4"
ATTNRES_ONLY_SHA = "c11ecd3780c1882b25c4f859e575251f42ad15f08365088b263c49a1b6682c2e"


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


@pytest.mark.local_pin
def test_off_off_is_bit_identical_to_the_base_commit():
    """(1) Neither flag: the pinned hash of `c8a2ff9`."""
    assert ModelConfig().use_mhc is False and ModelConfig().use_attnres is False
    assert _pinned({}) == OFF_OFF_SHA
    torch.manual_seed(SEED)
    model = tiny_model()
    assert param_count(model) == TINY_PARAMS
    assert not [k for k in model.state_dict() if k.startswith(("loop.mhc.", "loop.attnres."))]


# --- 2. each single arm -----------------------------------------------------


@pytest.mark.local_pin
def test_each_single_arm_keeps_its_pre_composition_hash():
    """(2) mHC-only and AttnRes-only are bit-identical to `c8a2ff9`.

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
    assert torch.isfinite(torch.tensor(loss))

    grads: list[str] = []
    for name, p in model.loop.attnres.named_parameters():
        g = p.grad
        assert g is not None, f"attnres.{name}: no gradient slot"
        assert bool(torch.isfinite(g).all()), f"attnres.{name}: non-finite gradient"
        assert float(g.abs().max()) > 0.0, f"attnres.{name}: gradient is exactly zero"
        grads.append(f"attnres.{name}={float(g.abs().max()):.2e}")

    for name, p in model.loop.mhc.named_parameters():
        g = p.grad
        assert g is not None, f"mhc.{name}: no gradient slot"
        assert bool(torch.isfinite(g).all()), f"mhc.{name}: non-finite gradient"
        assert float(g.abs().max()) > 0.0, f"mhc.{name}: gradient is exactly zero"
        grads.append(f"mhc.{name}={float(g.abs().max()):.2e}")
    print("[both] 3-step max|grad|: " + " ".join(grads))


# --- 4. identity at init ----------------------------------------------------


def test_the_composition_is_the_mhc_write_at_init(monkeypatch):
    """(4) Zero query => `a = y`; the composition adds nothing to the mHC write.

    The exact lever is asserted on four numbers: `S·softmax([0, 0]) = [1, 1]`,
    so `depth_attend([0, y], scale=2) == y` with `torch.equal`. On the model,
    the second argument `MHCMix.forward` receives from the composition and from
    the mHC-only arm is compared key for key: at init they are the same tensor.

    The full-model distance to the plain ReZero path is PRINTED, not asserted
    equal: this base's mHC init is `b_pre = 10` (`H_pre ≈ K`), so the mHC arm
    itself is 1.34 relative away from ReZero and no exact equality exists to
    claim without changing the pinned single-arm behaviour.
    """
    x, _ = _batch()
    torch.manual_seed(3)
    mhc_only = tiny_model(use_mhc=True)
    torch.manual_seed(3)
    both = tiny_model(use_mhc=True, use_attnres=True)

    seen: dict[str, list[torch.Tensor]] = {"mhc_only": [], "both": []}
    who = {"name": "mhc_only"}

    real_mhc = MHCMix.forward

    def spy_mhc(self, streams, x_in):
        seen[who["name"]].append(x_in.detach().clone())
        return real_mhc(self, streams, x_in)

    monkeypatch.setattr(MHCMix, "forward", spy_mhc)
    with torch.no_grad():
        mhc_logits, _, _ = mhc_only(x)
        who["name"] = "both"
        both_logits, _, _ = both(x)
    monkeypatch.setattr(MHCMix, "forward", real_mhc)
    assert len(seen["mhc_only"]) == len(seen["both"]) == 2, {k: len(v) for k, v in seen.items()}
    for i, (a, b) in enumerate(zip(seen["mhc_only"], seen["both"], strict=True)):
        assert torch.equal(a, b), f"iteration {i}: update moved at init"

    slot = AttnResSlot(64)
    y = torch.randn(1, 1, 64)
    got = depth_attend([torch.zeros_like(y), y], slot, tiny_cfg().norm_eps, scale=2.0)
    assert torch.equal(got, y), (got - y).abs().max()

    # The printed bound: the same function on shared weights is not available
    # (mHC consumes RNG before `lm_head`), so compare ReZero's own distance.
    torch.manual_seed(3)
    off = tiny_model()
    with torch.no_grad():
        off_logits, _, _ = off(x)
    rel_off = float((off_logits - both_logits).norm() / off_logits.norm())
    rel_mhc = float((off_logits - mhc_logits).norm() / off_logits.norm())
    print(
        f"[both] at-init vs ReZero: rel={rel_off:.4f} "
        f"(mHC-only rel={rel_mhc:.4f}); torch.equal=False "
        "(base mHC init b_pre=10, H_pre~K: exact ReZero equality not reachable)"
    )


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
    """(6) LOOM/ByteFlow stay refused; the pair refusal is DELETED, not dead.

    Source scan, not just behaviour: a refusal left behind under a flag that
    can never reach it is dead code that documents the opposite of what ships.
    """
    for key in ("loom", "use_byteflow"):
        with pytest.raises(LoudError) as e:
            parse_preset(f"use_mhc = true\n{key} = true\n")
        msg = str(e.value)
        assert key in msg and "use_mhc" in msg
        assert "Escape:" in msg and "use_attnres" in msg, msg
        assert "scripts/ab_mhc_attnres.py" in msg, msg
    src = open(__import__("dormouse.config", fromlist=["x"]).__file__, encoding="utf-8").read()
    assert "two write operators in one run" not in src
    assert "one experiment with two arms" not in src
    # The field's own docstring names the composition and calls the pair legal.
    assert "COMPOSES with `use_mhc`" in src
    assert "THE COMPOSITION" in src
    loop_src = open(loopblock_mod.__file__, encoding="utf-8").read()
    assert "THE COMPOSITION" in loop_src


# --- 7. the halt head, with both mechanisms on ------------------------------


def test_the_halt_head_still_trains_with_both_mechanisms_on():
    """(7) `halt_weight > 0` + the pair: builds, trains, `last_halt` populated.

    The halt head is wired THROUGH the loop's `collect_hidden`/`halt` seam
    (`model.py`'s loss). The composition changes the write dispatch on the same
    code path, so this is the regression that would otherwise only appear as a
    training crash or a silently zero halt term.
    """
    x, y = _batch()
    torch.manual_seed(11)
    model = tiny_model(use_mhc=True, use_attnres=True, halt_weight=0.1)
    assert model.halt is not None
    assert model.loop.mhc is not None and model.loop.attnres is not None
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        opt.zero_grad()
        _, loss, _ = model(x, y)
        assert loss is not None
        loss.backward()
        opt.step()
    assert model.last_halt is not None, "the halt term never reached the model"
    halted = float(model.last_halt)
    assert torch.isfinite(model.last_halt).all(), model.last_halt
    assert model.loop._depth_ran == model.cfg.max_iter, model.loop._depth_ran
    assert len(model.loop._per_iter_hidden) == len(model.loop._per_iter_ce) == model.cfg.max_iter
    held = 0
    for name, p in model.halt.named_parameters():
        g = p.grad
        assert g is not None and bool(torch.isfinite(g).all()), f"halt.{name}"
        held += 1
    print(f"[both] halt: last_halt={halted:.6f} params={held} depth={model.loop._depth_ran}")


# --- 8. no host sync --------------------------------------------------------


def test_the_composition_forward_reads_no_device_scalar(monkeypatch):
    """(8) `.item`/`.tolist`/`int(tensor)`/`float(tensor)` all raise.

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
