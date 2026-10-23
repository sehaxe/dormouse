"""Golden gates for `dense_qat` — the DENSE path's QAT switch.

The owner's decision (2026-10-18): the next training run trains in the FINAL
QUANTIZED FORM — ternary weights everywhere, 4-bit activations — exactly as
inference will run. The factor rung (`factor_qat`) already covers every TSCT
factor; `dense_qat` covers what it does not: every plain `nn.Linear`/
`DenseLinear` weight in the forward.

The property under test is the WIRING, not the quantizer (`tests/test_ternary.py`
/ `test_mxfp.py` own the formats): `tsct.dense_qat_project` IS
`factor_qat_project` (one function object), so what these gates pin is

- `none` leaves every existing number bit-identical, and `alpha = 0` is the
  same forward bit for bit (the annealing floor);
- at `alpha = 1` the forward IS a forward whose weights were pre-quantized in
  place — the fixture `_pre_quantize` constructs;
- gradients still reach the fp32 masters through the STE;
- a COVERAGE walk proves EVERY `nn.Linear`/`DenseLinear` weight in the forward
  actually went through the projector (the module list is written out below —
  a new dense weight fails this file until it is covered);
- ONE alpha drives both paths (`set_factor_qat_alpha`'s walk), pinned by a spy
  on both projectors during a real `train()` run;
- the config refusals are LOUD.

The two deliberate exclusions, both the BitNet convention (b1.58
arXiv:2402.17764 quantizes weight matrices): `nn.Embedding` tables (`embed`,
DSpark's `w1`), the Engram n-gram tables, every norm and every bias, and the
TSCT diagonals (`s`, `gate_s`, `up_s` — they ARE the scale). `use_byteflow` is
refused with a non-`none` `dense_qat` (the ByteFlow net's projections are a
separate tree; `config._validate_byteflow`).
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import pytest
import torch
from conftest import tiny_cfg, tiny_run, write_corpus
from torch import nn

from dormouse.config import load_run
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel
from dormouse.model.tsct import (
    DENSE_QAT_ATTN_SITES,
    DenseLinear,
    dense_qat_project,
    factor_qat_project,
    set_factor_qat_seed,
)
from dormouse.train import train

#: The pre-change golden for gate b: `dense_qat_skip_attn=false` must reproduce
#: the 3.2733 arm's forward bit for bit. Captured at commit 257f1fc with
#: `torch.set_num_threads(1)`, seed 1234, on the coverage fixture below.
PRECHANGE = Path(__file__).resolve().parent / "fixtures" / "dense_qat_attn_prechange.pt"


def _model(**over) -> DormouseModel:
    return DormouseModel(tiny_cfg(n_experts=4, moe_topk=2, **over))


def _coverage_model(**over) -> DormouseModel:
    """The full dense tree: KDA + MSA + dSpark, and DenseLinear out_proj/lm_head."""
    base = dict(use_tsct=False, use_msa=True, dspark_weight=0.05, dense_qat="ternary")
    return _model(**{**base, **over})


def _pre_quantize(model: nn.Module, kind: str = "ternary", *, skip_attn: bool = True) -> int:
    """The alpha=1 fixture: project every covered dense weight IN PLACE.

    `skip_attn` mirrors the model's own `dense_qat_skip_attn`: the attention
    sites stay masters when the default rule is on, or the two forwards would
    disagree for the wrong reason.

    Returns how many weights moved, so a caller can assert the walk found the
    module set it expected (a model that grew a dense weight the fixture does
    not see would fail the coverage gate, not this count).
    """
    n = 0
    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, (nn.Linear, DenseLinear)):
                if skip_attn and name in DENSE_QAT_ATTN_SITES:
                    continue
                m.weight.copy_(dense_qat_project(m.weight, kind, 1.0))
                n += 1
    return n


def _spy_projectors(monkeypatch) -> dict[str, list[float]]:
    """Patch every `dormouse.*` namespace binding a projector; record the alphas.

    `from .tsct import factor_qat_project` binds the FUNCTION in the importing
    module, so a spy on `tsct` alone would miss the call sites — the walk over
    the loaded `dormouse.*` modules patches every binding there is. The two
    names are separate bindings even though they are the same object (the
    alias), which is exactly what lets this test tell the two paths apart.
    """
    seen: dict[str, list[float]] = {"factor": [], "dense": []}
    for name, mod in list(sys.modules.items()):
        if not name.startswith("dormouse."):
            continue
        for attr, key in (("factor_qat_project", "factor"), ("dense_qat_project", "dense")):
            fn = getattr(mod, attr, None)
            if fn is None:
                continue

            def spy(w, kind, alpha, *a, _key=key, _fn=fn, **kw):
                seen[_key].append((id(w), alpha))
                return _fn(w, kind, alpha, *a, **kw)

            monkeypatch.setattr(mod, attr, spy)
    return seen


def _blocks(xs: list[float]) -> list[float]:
    """Collapse consecutive duplicates: the distinct alphas in step order."""
    out: list[float] = []
    for x in xs:
        if not out or out[-1] != x:
            out.append(x)
    return out


# ---------------------------------------------------------------------------
# the projector itself (the shared semantics)
# ---------------------------------------------------------------------------


def test_none_is_the_master_untouched():
    torch.manual_seed(0)
    w = torch.randn(8, 8)
    assert dense_qat_project(w, "none", 1.0) is w  # not a copy — the same tensor


@pytest.mark.parametrize("kind", ["ternary", "binary", "mxfp4"])
def test_dense_projector_is_the_factor_projector(kind):
    """SAME semantics, pinned per rung: the dense name IS the factor function."""
    torch.manual_seed(0)
    w = torch.randn(16, 32) * 0.2
    for alpha in (0.0, 0.25, 1.0):
        assert torch.equal(
            dense_qat_project(w, kind, alpha), factor_qat_project(w, kind, alpha)
        ), f"{kind} at alpha={alpha} diverged from the factor path"
    assert torch.equal(dense_qat_project(w, kind, 0.0), w), "alpha=0 must be the master"


def test_alpha_one_is_the_ternary_projection():
    torch.manual_seed(0)
    w = torch.randn(16, 32) * 0.2
    q = dense_qat_project(w, "ternary", 1.0)
    a = float(w.abs().mean())
    on_grid = ((q.abs() - a).abs() < 1e-5) | (q.abs() < 1e-6)
    assert bool(on_grid.all()), "alpha=1 must land on {0, ±mean(|W|)}"
    assert bool((q.abs() < 1e-6).any()), "a random master must produce real zeros"
    assert not torch.equal(q, w)


def test_s3t_dense_mask_is_a_seeded_stream():
    torch.manual_seed(0)
    w = torch.randn(16, 16)
    set_factor_qat_seed(7)
    a = dense_qat_project(w, "ternary_s3t", 1.0)
    set_factor_qat_seed(7)
    b = dense_qat_project(w, "ternary_s3t", 1.0)
    set_factor_qat_seed(8)
    c = dense_qat_project(w, "ternary_s3t", 1.0)
    assert torch.equal(a, b), "same seed must reproduce the dense mask exactly"
    assert not torch.equal(a, c), "a different seed must draw a different mask"


# ---------------------------------------------------------------------------
# the forward: alpha=0 identity, alpha=1 pre-quantized fixture
# ---------------------------------------------------------------------------


def test_alpha_zero_forward_is_bit_identical():
    q = _model(dense_qat="ternary")
    q.set_factor_qat_alpha(0.0)
    off = _model()
    off.load_state_dict(q.state_dict())
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        lq, loss_q, _ = q(x, x)
        lo, loss_o, _ = off(x, x)
    assert torch.equal(lq, lo), "alpha=0 must be the master forward, bit for bit"
    assert torch.equal(loss_q, loss_o)


def test_alpha_one_forward_equals_pre_quantized_weights():
    """The fixture: model B's weights ARE model A's projected weights.

    If any covered site ignored `dense_qat`, A would multiply by the master and
    B by the projection, and the two logits would differ — so this is a wiring
    gate over the whole covered set, not a second copy of the quantizer.
    """
    a = _model(dense_qat="ternary")
    b = _model()
    b.load_state_dict(a.state_dict())
    n = _pre_quantize(b)
    assert n >= 7, f"the fixture found only {n} dense weights"
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        la, _, _ = a(x, x)
        lb, _, _ = b(x, x)
    assert torch.equal(la, lb), "alpha=1 must be the pre-quantized-weight forward"


def test_use_tsct_false_dense_readout_is_covered():
    """`use_tsct = false` makes out_proj/lm_head DenseLinears — the dense rung."""
    a = _model(use_tsct=False, dense_qat="ternary")
    b = _model(use_tsct=False)
    b.load_state_dict(a.state_dict())
    _pre_quantize(b)
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        la, _, _ = a(x, x)
        lb, _, _ = b(x, x)
    assert torch.equal(la, lb)
    _, loss, _ = a(x, x)
    loss.backward()
    for name in ("lm_head", "loop.out_proj"):
        m = a.get_submodule(name)
        g = m.weight.grad
        assert g is not None and bool(torch.isfinite(g).all()) and g.abs().sum() > 0, name


def test_dense_qat_moves_the_logits_and_stays_finite():
    a = _model()
    b = _model(dense_qat="ternary")
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        la, _, _ = a(x, x)
        lb, _, _ = b(x, x)
    assert bool(torch.isfinite(lb).all())
    assert not torch.equal(la, lb), "a quantized dense forward must move the logits"


@pytest.mark.parametrize("mode", ["binary", "mxfp4", "ternary_s3t"])
def test_each_dense_rung_runs_and_moves_the_logits(mode):
    set_factor_qat_seed(0)
    a = _model()
    b = _model(dense_qat=mode)
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        la, _, _ = a(x, x)
        lb, _, _ = b(x, x)
    assert bool(torch.isfinite(lb).all())
    assert not torch.equal(la, lb), f"{mode} left the dense forward untouched"


# ---------------------------------------------------------------------------
# STE gradients reach the dense masters
# ---------------------------------------------------------------------------


def test_ste_gradient_flows_to_the_dense_weights():
    model = _model(dense_qat="ternary")
    x = torch.randint(0, 256, (2, 16))
    _, loss, _ = model(x, x)
    loss.backward()
    checked = 0
    for name, m in model.named_modules():
        if not isinstance(m, (nn.Linear, DenseLinear)):
            continue
        if name == "loop.attn.beta_proj":
            # The CPU path is the SDPA fallback, which reads q/k/v/g directly
            # and never beta (the fused FLA arm and the decode step do; the
            # `projections()` seam is gated below).
            continue
        g = m.weight.grad
        assert g is not None, f"{name} got no gradient"
        assert bool(torch.isfinite(g).all()) and g.abs().sum() > 0, name
        checked += 1
    assert checked >= 9, f"only {checked} dense weights had a gradient to check"


def test_kda_projections_seam_is_quantized():
    """`KDAAttention.projections` is the FLA arm's and the decode step's entry.

    On CPU the training forward takes the SDPA fallback, which reads q/k/v/g
    directly (and never beta), so this seam is exercised here on purpose: it is
    the same `qat_linear` call site the GPU training forward and every decode
    step use. `dense_qat_skip_attn=false` is the arm under this gate; the
    default-skip side of the same seam is `test_skip_attn_keeps_attention...`.
    """
    a = _model()
    b = _model(dense_qat="ternary", dense_qat_skip_attn=False)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        pa = a.loop.attn.projections(x)
        pb = b.loop.attn.projections(x)
    for name, ta, tb in zip("qkvgb", pa, pb, strict=True):
        assert not torch.equal(ta, tb), f"projections() ignored dense_qat on {name}"


# ---------------------------------------------------------------------------
# COVERAGE: every nn.Linear/DenseLinear weight in the forward is classified
# ---------------------------------------------------------------------------

#: The dense weights `dense_qat` projects in EVERY arm — the FFN/controller/
#: memory/readout set BitNet b1.58 quantizes. A new dense weight anywhere in
#: the tree fails the coverage gate until it is routed through
#: `dense_qat_project` AND classified here or in `ATTN_SITES`.
COVERED = (
    "loop.controller",
    "loop.mem_dense",
    "loop.out_proj",
    "loop.moe.router",
    "loop.moe.down",
    "loop.engram.query_proj",
    "loop.engram.key_proj",
    "loop.engram.value_proj",
    "dspark.w2",
    "dspark.joint",
    "dspark.conf",
    "lm_head",
)

#: The attention sites — read from the ONE list the implementation
#: (`tsct.dense_qat_kind`) reads. Projected ONLY with
#: `dense_qat_skip_attn=false`, the arm the 3.2733 measurement ran.
ATTN_SITES = DENSE_QAT_ATTN_SITES


def _walk(model: nn.Module) -> dict[str, nn.Module]:
    return {
        name: m
        for name, m in model.named_modules()
        if isinstance(m, (nn.Linear, DenseLinear))
    }


def _walk_forward(model: DormouseModel) -> None:
    x = torch.randint(0, 256, (2, 64))
    model(x, x)  # the training forward, aux term included
    # The FLA arm's entry is not reachable on CPU (see the seam test above);
    # call it so beta_proj is classified on the same code path the GPU training
    # forward runs. Both arms own `projections` (full attention has q/k/v).
    model.loop.attn.projections(torch.randn(2, 4, 64))


def _arm_attn_sites(model: nn.Module) -> tuple[str, ...]:
    """The attention sites THIS arm owns: `kda` has g/beta, `full` has o_proj.

    The ONE list (`DENSE_QAT_ATTN_SITES`) names both arms' sites because it is
    the list `dense_qat_kind` reads; a tree can only own one arm's modules.
    """
    walk = _walk(model)
    return tuple(name for name in ATTN_SITES if name in walk)


def _projected_names(model: nn.Module, seen: dict[str, list[float]]) -> set[str]:
    ids = {wid for wid, _ in seen["dense"]}
    return {name for name, m in _walk(model).items() if id(m.weight) in ids}


@pytest.mark.parametrize("attn_kind", ["kda", "full"])
def test_coverage_every_dense_weight_is_classified(monkeypatch, attn_kind):
    """The default arm: every site is walked, exactly the non-attention ones project."""
    seen = _spy_projectors(monkeypatch)
    model = _coverage_model(attn_kind=attn_kind)
    _walk_forward(model)

    walk = _walk(model)
    sites = _arm_attn_sites(model)
    assert sorted(walk) == sorted((*COVERED, *sites)), (
        "the covered-module list moved: a dense weight was added or removed"
    )
    assert set(ATTN_SITES) - set(sites) == (
        {"loop.attn.g_proj", "loop.attn.beta_proj"}
        if attn_kind == "full"
        else {"loop.attn.o_proj"}
    ), "an arm's attention sites moved: kda owns g/beta, full owns o_proj"
    assert not set(COVERED) & set(sites), "a site cannot be both classes"
    projected = _projected_names(model, seen)
    assert projected == set(COVERED), (
        f"projected under the default: unexpected {sorted(projected - set(COVERED))}, "
        f"missed {sorted(set(COVERED) - projected)}"
    )
    assert len(seen["dense"]) >= len(COVERED), "fewer projection calls than sites"


@pytest.mark.parametrize("attn_kind", ["kda", "full"])
def test_coverage_skip_off_projects_every_site(monkeypatch, attn_kind):
    """Gate b's wiring side: with the default rule off, today's full set projects."""
    seen = _spy_projectors(monkeypatch)
    model = _coverage_model(dense_qat_skip_attn=False, attn_kind=attn_kind)
    _walk_forward(model)

    projected = _projected_names(model, seen)
    assert projected == set(_walk(model)), (
        f"skip=false must project the whole tree: missed {sorted(set(_walk(model)) - projected)}"
    )
    assert len(seen["dense"]) >= len(_walk(model)), "fewer projection calls than sites"


# ---------------------------------------------------------------------------
# `dense_qat_skip_attn`: attention is never below 8 bits (the BitNet rule)
# ---------------------------------------------------------------------------


def test_skip_attn_defaults_on_and_loads():
    assert tiny_cfg().dense_qat_skip_attn is True
    on = load_run("small", model_overrides={"dense_qat": "ternary"})
    assert on.model.dense_qat_skip_attn is True
    off = load_run(
        "small", model_overrides={"dense_qat": "ternary", "dense_qat_skip_attn": False}
    )
    assert off.model.dense_qat_skip_attn is False


@pytest.mark.parametrize("attn_kind", ["kda", "full"])
def test_gate_a_skip_attn_projects_no_attention_weight(monkeypatch, attn_kind):
    """The projector sees ZERO attention weights and EVERY other dense weight."""
    seen = _spy_projectors(monkeypatch)
    model = _coverage_model(attn_kind=attn_kind)
    _walk_forward(model)

    walk = _walk(model)
    sites = _arm_attn_sites(model)
    hit = {wid for wid, _ in seen["dense"]}
    attn_ids = {id(walk[name].weight) for name in sites}
    assert len(attn_ids) == len(sites), "the walk does not resolve the arm's sites"
    assert not hit & attn_ids, (
        f"attention weights reached the projector: "
        f"{sorted(n for n in sites if id(walk[n].weight) in hit)}"
    )
    missed = [name for name in COVERED if id(walk[name].weight) not in hit]
    assert not missed, f"non-attention sites the projector never saw: {missed}"


def test_gate_b_skip_off_is_the_pre_change_forward():
    """`dense_qat_skip_attn=false` IS the measured 3.2733 arm's forward, bit for bit.

    The golden was captured from the code before this flag existed (commit
    257f1fc) on the coverage fixture at seed 1234. Both sides pin one thread:
    CPU float32 reduction order follows the thread count, so the golden is the
    same bits only at the recorded width (see the fixture's capture note).
    """
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(1234)
        model = _coverage_model(dense_qat_skip_attn=False)
        x = torch.randint(0, 256, (2, 16))
        with torch.no_grad():
            logits, _, _ = model(x, x)
        assert torch.equal(logits, torch.load(PRECHANGE))
    finally:
        torch.set_num_threads(old)


def test_gate_c_skip_attn_keeps_attention_on_its_master(monkeypatch):
    """Attention's effective weight is the raw master (dtype and values), and the
    FFN/controller/router sites are still quantized in the same forward."""
    seen = _spy_projectors(monkeypatch)
    on = _coverage_model()
    off = _coverage_model(dense_qat="none")
    off.load_state_dict(on.state_dict())

    x = torch.randint(0, 256, (2, 64))
    with torch.no_grad():
        on(x, x)
        xp = torch.randn(2, 4, 64)
        pa = on.loop.attn.projections(xp)
        pb = off.loop.attn.projections(xp)
        xm = torch.randn(2, 64, 64)
        ya = on.loop.msa._forward(xm, Counters(), None)
        yb = off.loop.msa._forward(xm, Counters(), None)

    for name, ta, tb in zip("qkvgb", pa, pb, strict=True):
        assert torch.equal(ta, tb), f"attention {name} moved under skip_attn"
        assert ta.dtype == tb.dtype
    assert torch.equal(pa[0], on.loop.attn.q_proj(xp).reshape(2, 4, 4, 16))
    assert torch.equal(ya, yb), "the MSA arm must read its masters under skip_attn"

    hit = {wid for wid, _ in seen["dense"]}
    for name in ("loop.controller", "loop.mem_dense", "loop.moe.router", "loop.moe.down"):
        assert id(on.get_submodule(name).weight) in hit, f"{name} must still be quantized"


@pytest.mark.parametrize("skip", [True, False])
def test_gate_d_skip_attn_is_inert_with_dense_qat_none(monkeypatch, skip):
    seen = _spy_projectors(monkeypatch)
    model = _coverage_model(dense_qat="none", dense_qat_skip_attn=skip)
    _walk_forward(model)
    assert seen["dense"] == [] and seen["factor"] == [], (
        "dense_qat=none must call neither projector, whatever skip_attn says"
    )


def test_gate_e_skip_attn_adds_no_host_sync(monkeypatch):
    """Every host-read primitive is made to raise; the skipped attention path must not use one.

    Scoped to the KDA arm + its `projections` seam (the sites this flag
    changes): the full CPU forward's MoE dispatch already calls `.tolist()`,
    a pre-existing sync that predates and is unrelated to this switch.
    """
    model = _coverage_model()
    x = torch.randn(2, 8, 64)

    def boom(*_a, **_k):
        raise AssertionError("host sync in the skip_attn path")

    for name in ("item", "tolist", "cpu", "numpy", "__float__", "__int__"):
        monkeypatch.setattr(torch.Tensor, name, boom)
    with torch.no_grad():
        model.loop.attn.projections(x)
        model.loop.attn(x)


# ---------------------------------------------------------------------------
# one alpha for both paths (the shared anneal)
# ---------------------------------------------------------------------------


def test_ramp_reaches_every_dense_consumer():
    model = _model(use_tsct=False, use_msa=True, dspark_weight=0.05, dense_qat="ternary")
    model.set_factor_qat_alpha(0.25)
    hit = 0
    for m in model.modules():
        if hasattr(m, "dense_qat_alpha"):
            assert m.dense_qat_alpha == 0.25
            hit += 1
    assert hit >= 9, "LoopBlock + KDA + MoE + Engram + MSA pair + DSpark + the dense head"
    # ... and the factor consumers still get the SAME value.
    for m in model.modules():
        if hasattr(m, "factor_qat_alpha"):
            assert m.factor_qat_alpha == 0.25


def test_alpha_seen_by_both_paths_is_identical(tmp_path, monkeypatch):
    """The spy sees EVERY projection during a real annealed `train()` run.

    The distinct alphas in step order must be `min(1, step/N)` on BOTH paths:
    one schedule, one alpha, no second ramp.
    """
    seen = _spy_projectors(monkeypatch)
    data = write_corpus(tmp_path / "c.bin")
    run = tiny_run(tmp_path, factor_qat_anneal_steps=4, steps=5)
    run = dataclasses.replace(
        run,
        model=dataclasses.replace(run.model, factor_qat="ternary", dense_qat="ternary"),
        train=dataclasses.replace(run.train, data=str(data), log_every=100, ckpt_every=100),
    )
    train(DormouseModel(run.model), run, resume=False, log=lambda *_: None)
    want = [0.0, 0.25, 0.5, 0.75, 1.0]
    factor_alphas = [a for _, a in seen["factor"]]
    dense_alphas = [a for _, a in seen["dense"]]
    assert _blocks(factor_alphas) == want, _blocks(factor_alphas)
    assert _blocks(dense_alphas) == want, _blocks(dense_alphas)
    # Both paths ran every step (an empty list would pass the blocks check).
    assert len(seen["factor"]) >= 5 and len(seen["dense"]) >= 5


# ---------------------------------------------------------------------------
# config: LOUD refusals, the shared alpha/seed
# ---------------------------------------------------------------------------


def test_config_refuses_a_bad_literal():
    with pytest.raises(LoudError, match="dense_qat"):
        load_run("small", model_overrides={"dense_qat": "int4"})
    # The legal pairs load.
    load_run("small", model_overrides={"dense_qat": "ternary"})
    load_run("small", model_overrides={"dense_qat": "ternary", "factor_qat": "ternary"})
    load_run("small", model_overrides={"dense_qat": "mxfp4"})


def test_config_refuses_dense_qat_with_byteflow():
    over = {
        "use_byteflow": True,
        "bf16": False,
        "use_engram": False,
        "byteflow_k_tokens": 128,
        "dense_qat": "ternary",
    }
    with pytest.raises(LoudError, match="dense_qat"):
        load_run("small", model_overrides=over)
    # The same byteflow config without dense_qat loads.
    load_run("small", model_overrides={**over, "dense_qat": "none"})


def test_config_alpha_and_seed_are_shared():
    # alpha with only the dense path on is NOT inert (it drives dense_qat).
    load_run("small", model_overrides={"dense_qat": "ternary", "factor_qat_alpha": 0.5})
    with pytest.raises(LoudError, match="inert"):
        load_run("small", model_overrides={"factor_qat_alpha": 0.5})
    # The S3T seed is live on the dense path too.
    load_run("small", model_overrides={"dense_qat": "ternary_s3t", "factor_qat_seed": 7})
    with pytest.raises(LoudError, match="inert"):
        load_run("small", model_overrides={"dense_qat": "ternary", "factor_qat_seed": 7})


# ---------------------------------------------------------------------------
# the 20-step CPU smoke: ternary weights + int4 activations, counters on the line
# ---------------------------------------------------------------------------


def test_20_step_cpu_smoke_trains_and_logs_counters(tmp_path):
    data = write_corpus(tmp_path / "c.bin")
    run = tiny_run(tmp_path, steps=20, log_every=10, factor_qat_anneal_steps=10)
    run = dataclasses.replace(
        run,
        model=dataclasses.replace(
            run.model, factor_qat="ternary", dense_qat="ternary", act_quant="int4"
        ),
        train=dataclasses.replace(run.train, data=str(data)),
    )
    lines: list[str] = []
    report = train(DormouseModel(run.model), run, resume=False, log=lines.append)
    assert math.isfinite(report["loss"])
    assert report["counters"]["dense_qat_arms"] == 20, "one per forward"
    assert report["counters"]["act_quant_arms"] > 0
    step_lines = [ln for ln in lines if ln.startswith("step=")]
    assert len(step_lines) == 2
    assert "dense_qat_arms=10" in step_lines[-1] and "act_quant_arms=" in step_lines[-1]
