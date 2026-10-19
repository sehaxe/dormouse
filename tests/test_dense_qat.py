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

import pytest
import torch
from conftest import tiny_cfg, tiny_run, write_corpus
from torch import nn

from dormouse.config import load_run
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.model.tsct import (
    DenseLinear,
    dense_qat_project,
    factor_qat_project,
    set_factor_qat_seed,
)
from dormouse.train import train


def _model(**over) -> DormouseModel:
    return DormouseModel(tiny_cfg(n_experts=4, moe_topk=2, **over))


def _pre_quantize(model: nn.Module, kind: str = "ternary") -> int:
    """The alpha=1 fixture: project every covered dense weight IN PLACE.

    Returns how many weights moved, so a caller can assert the walk found the
    module set it expected (a model that grew a dense weight the fixture does
    not see would fail the coverage gate, not this count).
    """
    n = 0
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Linear, DenseLinear)):
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
        assert torch.equal(dense_qat_project(w, kind, alpha), factor_qat_project(w, kind, alpha)), (
            f"{kind} at alpha={alpha} diverged from the factor path"
        )
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
    assert n >= 10, f"the fixture found only {n} dense weights"
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
    step use.
    """
    a = _model()
    b = _model(dense_qat="ternary")
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        pa = a.loop.attn.projections(x)
        pb = b.loop.attn.projections(x)
    for name, ta, tb in zip("qkvgb", pa, pb, strict=True):
        assert not torch.equal(ta, tb), f"projections() ignored dense_qat on {name}"


# ---------------------------------------------------------------------------
# COVERAGE: every nn.Linear/DenseLinear weight in the forward is projected
# ---------------------------------------------------------------------------

#: The dense weights this fixture's forward must project, by module name. A new
#: dense weight anywhere in the tree fails `test_coverage_...` below until it is
#: routed through `dense_qat_project` AND listed here.
COVERED = (
    "loop.controller",
    "loop.mem_dense",
    "loop.out_proj",
    "loop.attn.q_proj",
    "loop.attn.k_proj",
    "loop.attn.v_proj",
    "loop.attn.g_proj",
    "loop.attn.beta_proj",
    "loop.moe.router",
    "loop.moe.down",
    "loop.engram.query_proj",
    "loop.engram.key_proj",
    "loop.engram.value_proj",
    "loop.msa.q_proj",
    "loop.msa.k_proj",
    "loop.msa.v_proj",
    "loop.msa.selector.q_idx",
    "loop.msa.selector.k_idx",
    "dspark.w2",
    "dspark.joint",
    "dspark.conf",
    "lm_head",
)


def test_coverage_every_dense_weight_is_projected(monkeypatch):
    seen = _spy_projectors(monkeypatch)
    # Every dense arm ON, so the walk sees the whole tree: use_tsct=false makes
    # out_proj/lm_head DenseLinears, use_msa builds the MSA pair, dspark_weight
    # builds the draft head.
    model = _model(use_tsct=False, use_msa=True, dspark_weight=0.05, dense_qat="ternary")
    x = torch.randint(0, 256, (2, 64))
    model(x, x)  # the training forward, aux term included
    # The FLA arm's entry is not reachable on CPU (see the seam test above);
    # call it so beta_proj's coverage is proven on the same code path the GPU
    # training forward runs.
    model.loop.attn.projections(torch.randn(2, 4, 64))

    walk = {name: m for name, m in model.named_modules() if isinstance(m, (nn.Linear, DenseLinear))}
    projected = {wid for wid, _ in seen["dense"]}
    missed = sorted(name for name, m in walk.items() if id(m.weight) not in projected)
    assert not missed, f"dense weights the projector never saw: {missed}"
    assert sorted(walk) == sorted(COVERED), (
        "the covered-module list moved: a dense weight was added or removed"
    )
    assert len(seen["dense"]) >= len(COVERED), "fewer projection calls than sites"


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
