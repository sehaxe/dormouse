"""The per-optimizer-step QAT cache (`model/qat_cache.py`) and its wiring.

The measured cost (2026-10-21, 9.2 M polygon): the full-quantized recipe ran at
17.1 k tok/s against 40.9 k bf16 because every forward re-quantized every
weight. The cache computes `q(w)` once per trained step and every forward
re-composes `w + alpha·(q − w).detach()`. The gates below pin the three things
that make that a cache and not a behavior change:

* the forward VALUE is bit-identical to the uncached path (`torch.equal`);
* the GRADIENT is the same STE contract (every parameter, 1e-6 relative);
* the INVALUATION is the trainer's, never `_version` (a graph replay moves a
  value without bumping it — memory ID 77 / `tests/test_bank_invalidation.py`),
  and the stale-cache bug is proven to exist without it.
"""

from __future__ import annotations

import dataclasses

import torch
from conftest import tiny_cfg, tiny_run, write_corpus

from dormouse.instrument import Counters
from dormouse.model import DormouseModel
from dormouse.model.qat_cache import (
    QuantCache,
    attach_qat_caches,
    invalidate_qat_cache_all,
    qat_project_stacked,
    take_qat_refresh_counts,
)
from dormouse.model.tsct import DenseLinear, TSCTLinear, factor_qat_project


def _model(**over) -> DormouseModel:
    return DormouseModel(tiny_cfg(n_experts=4, moe_topk=2, **over))


def _strip_caches(model: DormouseModel) -> None:
    """The uncached control: every module falls back to the per-forward path."""
    for m in model.modules():
        if getattr(m, "_qat_cache", None) is not None:
            m._qat_cache = None


def _forward(model: DormouseModel, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits, _, _ = model(x, x)
    return logits


def _pair(**over) -> tuple[DormouseModel, DormouseModel]:
    """Identical weights: A with the cache, B on the old per-forward path."""
    torch.manual_seed(0)
    a = _model(**over)
    b = _model(**over)
    b.load_state_dict(a.state_dict())
    _strip_caches(b)
    return a, b


# ---------------------------------------------------------------------------
# the helper itself
# ---------------------------------------------------------------------------


def test_refresh_is_detached_and_a_hit_reuses_the_tensor():
    torch.manual_seed(0)
    w = torch.randn(8, 16)
    cache = QuantCache()
    q0 = cache.refresh(w, "ternary", 1.0)
    assert not q0.requires_grad, "the cached projection must be detached"
    q1 = cache.refresh(w, "ternary", 1.0)
    assert q1 is q0, "a hit must return the SAME tensor (no re-quantization)"
    assert cache.hits == 1
    # A different alpha or mode is a miss: the anneal moves every step.
    assert cache.refresh(w, "ternary", 0.5) is not q0
    assert cache.refresh(w, "binary", 1.0) is not q0
    cache.invalidate()
    assert cache.refresh(w, "ternary", 1.0) is not q0, "invalidate must force a rebuild"


def test_ste_is_the_identity_estimator():
    torch.manual_seed(0)
    w = torch.randn(8, 16, requires_grad=True)
    cache = QuantCache()
    cache.refresh(w, "ternary", 1.0)
    y = cache.ste(w)
    y.sum().backward()
    assert w.grad is not None and torch.equal(w.grad, torch.ones_like(w))


# ---------------------------------------------------------------------------
# gate a: both switches off is the untouched model
# ---------------------------------------------------------------------------


def test_none_attaches_no_cache_and_adds_no_parameter():
    off = _model()
    assert off._qat_caches == [], "qat=none must not build a cache"
    on = _model(factor_qat="ternary", dense_qat="ternary")
    assert sum(p.numel() for p in on.parameters()) == sum(
        p.numel() for p in off.parameters()
    ), "QAT must not add parameters"
    assert on._qat_caches, "the quantized model must collect its caches"
    x = torch.randint(0, 256, (2, 16))
    assert torch.equal(_forward(off, x), _forward(off, x))


def test_none_projector_ignores_a_cache():
    torch.manual_seed(0)
    w = torch.randn(8, 8)
    assert factor_qat_project(w, "none", 1.0, cache=QuantCache()) is w


def test_none_switches_do_no_refresh_and_leave_the_model_bit_identical():
    """Gate: `factor_qat=none`, `dense_qat=none` — zero refreshes, same bits."""
    torch.manual_seed(0)
    a = _model()
    b = _model()
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 256, (2, 16))
    take_qat_refresh_counts()
    c = Counters()
    la = a(x, x, c)[0]
    assert take_qat_refresh_counts() == (0, 0), "QAT off must run no quantizer"
    assert c.get("qat_refresh_calls", 0) == 0
    assert torch.equal(la, _forward(b, x))


# ---------------------------------------------------------------------------
# gate b: cached forward value == uncached, bit for bit
# ---------------------------------------------------------------------------


def test_projector_cache_is_bit_identical_per_rung_and_alpha():
    torch.manual_seed(0)
    w = torch.randn(16, 32) * 0.2
    for kind in ("ternary", "binary", "mxfp4"):
        for alpha in (0.25, 0.5, 1.0):
            cold = factor_qat_project(w, kind, alpha)
            cache = QuantCache()
            warm = factor_qat_project(w, kind, alpha, cache=cache)
            again = factor_qat_project(w, kind, alpha, cache=cache)
            assert torch.equal(warm, cold), (kind, alpha)
            assert torch.equal(again, cold), (kind, alpha)
            assert cache.hits == 1, (kind, alpha)


def test_stacked_refresh_is_bit_identical_to_the_per_tensor_refresh():
    """Gate: ONE call over `[E, ·, ·]` == E calls, elementwise, all rungs.

    The bank site's whole correctness claim: the stacked projection is the same
    expression on the same values, so slicing a stacked bank is bit-identical to
    projecting that expert alone. `alpha = 0` is the master (both paths
    short-circuit), 0.5 the annealed blend, 1.0 the pure grid.
    """
    torch.manual_seed(0)
    stack = torch.randn(4, 16, 24) * 0.2  # 4 x (16, 24): both dims exercised
    slices = [stack[e] for e in range(stack.shape[0])]
    for kind in ("ternary", "binary", "mxfp4"):
        for alpha in (0.0, 0.5, 1.0):
            stacked, per = QuantCache(), QuantCache()
            qs = stacked.refresh_stacked(stack, kind, alpha, dim=(1, 2))
            qp = torch.stack([per.refresh(s, kind, alpha) for s in slices])
            assert torch.equal(qs, qp), (kind, alpha, "projection")
            assert torch.equal(
                stacked.ste(stack), torch.stack([per.ste(s) for s in slices])
            ), (kind, alpha, "ste")
            # The uncached arm is the same expression too (alpha=0 included).
            assert torch.equal(
                qat_project_stacked(stack, kind, alpha, dim=(1, 2)),
                torch.stack([factor_qat_project(s, kind, alpha) for s in slices]),
            ), (kind, alpha, "uncached")
    # One stacked call, one entry: a second refresh is a hit, not a second pass.
    cache = QuantCache()
    first = cache.refresh_stacked(stack, "ternary", 1.0, dim=(1, 2))
    assert cache.refresh_stacked(stack, "ternary", 1.0, dim=(1, 2)) is first
    assert cache.hits == 1


def test_cached_model_logits_are_bit_identical_to_the_uncached_model():
    for over in (
        {"factor_qat": "ternary", "dense_qat": "ternary"},
        {"factor_qat": "ternary", "dense_qat": "ternary", "act_quant": "int4"},
        {"factor_qat": "binary", "dense_qat": "binary"},
        {"dense_qat": "mxfp4"},
    ):
        a, b = _pair(**over)
        x = torch.randint(0, 256, (2, 16))
        la, lb = _forward(a, x), _forward(b, x)
        assert torch.equal(la, lb), over
        assert not torch.equal(la, torch.zeros_like(la)), over


def test_s3t_mask_is_drawn_once_per_step_not_per_forward():
    """The cache changes s3t's stream on purpose: one mask per step (COUNTED)."""
    torch.manual_seed(0)
    w = torch.randn(16, 16)
    cache = QuantCache()
    a = factor_qat_project(w, "ternary_s3t", 1.0, cache=cache)
    b = factor_qat_project(w, "ternary_s3t", 1.0, cache=cache)
    assert torch.equal(a, b), "one step must reuse the drawn mask"
    c = factor_qat_project(w, "ternary_s3t", 1.0)
    d = factor_qat_project(w, "ternary_s3t", 1.0)
    assert not torch.equal(c, d), "the uncached path draws per call (the old cost)"


# ---------------------------------------------------------------------------
# gate c: the STE contract — same gradient as the uncached path
# ---------------------------------------------------------------------------


def test_gradients_match_the_uncached_path_on_every_parameter():
    a, b = _pair(factor_qat="ternary", dense_qat="ternary", act_quant="int4")
    x = torch.randint(0, 256, (2, 16))
    a(x, x)[1].sum().backward()
    b(x, x)[1].sum().backward()
    checked = 0
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert na == nb
        ga, gb = pa.grad, pb.grad
        assert (ga is None) == (gb is None), na
        if ga is None:
            continue
        rel = float((ga - gb).norm() / (gb.norm() + 1e-30))
        assert rel <= 1e-6, f"{na}: relative gradient difference {rel}"
        checked += 1
    assert checked >= 40, f"only {checked} parameters carried a gradient"


def test_stacked_ste_gradient_matches_the_unbatched_path():
    """Gate: the bank's one-call STE keeps the parameter-grad contract.

    `qat_project_stacked(stack)` vs E `factor_qat_project(parameter)` calls on
    the SAME weights (medium's widths, E=8 for speed): every parameter's
    gradient must agree to 1e-6 relative. The mechanism is identity STE in both,
    so a regression here means the stacked residual leaked into the graph or a
    slice lost its master.
    """
    from dormouse.config import load_preset
    from dormouse.model.experts import SparseMoE

    torch.manual_seed(0)
    cfg = dataclasses.replace(
        load_preset("medium"), n_experts=8, factor_qat="ternary", dense_qat="ternary"
    )
    moe = SparseMoE(cfg)
    for kind in ("ternary", "binary", "mxfp4"):
        for name in ("gate_u", "up_u", "v"):
            params = [getattr(ex, name) for ex in moe.experts]
            for p in params:
                p.grad = None
            qat_project_stacked(torch.stack(params), kind, 0.5, dim=(1, 2)).pow(2).sum().backward()
            got = [p.grad.detach().clone() for p in params]
            for p in params:
                p.grad = None
            sum((factor_qat_project(p, kind, 0.5) ** 2).sum() for p in params).backward()
            for i, p in enumerate(params):
                rel = float((got[i] - p.grad).norm() / (p.grad.norm() + 1e-30))
                assert rel <= 1e-6, f"{kind}/{name}[{i}]: relative {rel}"


# ---------------------------------------------------------------------------
# gate d: the trainer's invalidation, not `_version`
# ---------------------------------------------------------------------------


def test_stale_cache_bug_exists_and_the_invalidation_fixes_it():
    torch.manual_seed(0)
    model = _model(factor_qat="ternary")
    x = torch.randint(0, 256, (2, 16))
    _forward(model, x)  # warm every cache

    lin = model.loop.out_proj
    assert isinstance(lin, TSCTLinear)
    u_before = lin._factors()[0].detach().clone()
    p = lin.u
    version = p._version
    moved_from = p.detach().clone()

    # The graph-replay shape on CPU: the VALUE moves, `_version` does not.
    p.data.add_(3.0)
    assert not torch.equal(p, moved_from), "the value must have moved"
    assert p._version == version, "the version must stay frozen"

    cache_entry_before = lin._qat_cache._q[id(p)][4]
    _forward(model, x)
    u_stale = lin._factors()[0]
    cache_entry_stale = lin._qat_cache._q[id(p)][4]
    assert cache_entry_stale is cache_entry_before, "without invalidation the q is stale"
    # The stale forward serves the OLD quantized value (fp32 re-add noise only),
    # not the new projection — this is the bug the trainer's call prevents.
    fresh = factor_qat_project(p.detach(), "ternary", 1.0)
    assert float((u_stale.detach() - u_before).abs().max()) < 1e-6
    assert float((u_stale.detach() - fresh).abs().max()) > 1e-3

    assert invalidate_qat_cache_all(model) > 0
    assert lin._qat_cache._q == {}, "invalidation must drop the stale projection"
    u_fresh = lin._factors()[0]
    assert not torch.equal(u_fresh.detach(), u_before), "after invalidation the new value is seen"
    assert float((u_fresh.detach() - fresh).abs().max()) < 1e-6


def test_attach_qat_caches_reaches_every_plain_linear():
    model = _model(dense_qat="ternary")
    linears = [m for m in model.modules() if type(m) is torch.nn.Linear]
    assert linears, "the tiny model must have plain linears"
    assert all(getattr(m, "_qat_cache", None) is not None for m in linears)
    caches = attach_qat_caches(model, dense=True)
    assert len({id(c) for c in caches}) == len(caches), "no cache may be collected twice"


def _spy_train_invalidation(monkeypatch) -> list[int]:
    import dormouse.model.qat_cache as qc
    import dormouse.train.loop as loop_mod

    calls: list[int] = []
    real = qc.invalidate_qat_cache_all

    def spy(model):
        calls.append(1)
        return real(model)

    monkeypatch.setattr(loop_mod, "invalidate_qat_cache_all", spy)
    return calls


def test_the_trainer_invalidates_every_trained_step(monkeypatch, tmp_path):
    calls = _spy_train_invalidation(monkeypatch)
    run = tiny_run(
        tmp_path,
        steps=3,
        batch=2,
        seq_len=64,
        log_every=10**9,
        ckpt_every=10**9,
        eval=False,
        data=str(write_corpus(tmp_path / "probe.bin")),
    )
    run = dataclasses.replace(
        run,
        model=dataclasses.replace(
            run.model, factor_qat="ternary", dense_qat="ternary", bf16=False
        ),
    )
    from dormouse.train.loop import train

    train(DormouseModel(run.model), run, resume=False, log=lambda *_: None)
    assert len(calls) == 3, f"3 trained steps must invalidate 3 times, saw {len(calls)}"


# ---------------------------------------------------------------------------
# gate e: alpha = 0 is the bf16 path, exactly
# ---------------------------------------------------------------------------


def test_alpha_zero_is_bit_identical_to_the_bf16_model():
    torch.manual_seed(0)
    q = _model(factor_qat="ternary", dense_qat="ternary")
    q.set_factor_qat_alpha(0.0)
    off = _model()
    off.load_state_dict(q.state_dict())
    x = torch.randint(0, 256, (2, 16))
    assert torch.equal(_forward(q, x), _forward(off, x))


# ---------------------------------------------------------------------------
# gate f: `qat_reuse` counts the hits, so the log proves the path ran
# ---------------------------------------------------------------------------


def test_qat_reuse_counts_cache_hits_per_forward():
    torch.manual_seed(0)
    model = _model(factor_qat="ternary", dense_qat="ternary")
    x = torch.randint(0, 256, (2, 16))
    c = Counters()
    model(x, x, c)
    assert c.get("qat_reuse", 0) > 0, "the warm path must report its hits"
    assert c.get("dense_qat_arms") == 1
    first = c["qat_reuse"]
    c2 = Counters()
    model(x, x, c2)
    assert c2["qat_reuse"] >= first, "a second forward must still be served by the cache"

    _strip_caches(model)
    c3 = Counters()
    model(x, x, c3)
    assert c3.get("qat_reuse", 0) == 0, "the uncached path reports no reuse"


def test_dense_linear_and_qat_linear_take_the_cache():
    torch.manual_seed(0)
    lin = DenseLinear(8, 8, dtype=torch.float32, dense_qat="ternary")
    w = lin.weight.detach().clone()
    x = torch.randn(4, 8)
    with torch.no_grad():
        once = lin(x)
        twice = lin(x)
    assert torch.equal(once, twice)
    assert lin._qat_cache.hits == 1
    cold = torch.nn.functional.linear(x, factor_qat_project(w, "ternary", 1.0))
    assert torch.equal(once, cold)


# ---------------------------------------------------------------------------
# gate g: the medium bank site is ONE call per projection, not E
# ---------------------------------------------------------------------------


def test_medium_bank_site_refreshes_in_one_call_not_E():
    """Gate: medium's expert bank (E=256) refreshes in 3 calls, not 768.

    The measured problem: the bank was quantized once per expert per factor, so
    a step issued E quantizer chains per projection. The site must issue ONE
    call per projection NAME (`gate_u`, `up_u`, `v` — different shapes, so three
    is the floor) carrying E tensors each. The pre-fix strategy is measured on
    the SAME weights in this test, so the comparison cannot rot.

    CPU note: the CPU forward routes through the naive reference arm (the padded
    bank is the CUDA path), which keeps its per-expert `DualLatentGLU` caches by
    design — materializing an fp32 bank is exactly what that arm exists to
    avoid. This gate exercises the bank SEAM the CUDA step runs.
    """
    from dormouse.config import load_preset
    from dormouse.model.experts import SparseMoE

    torch.manual_seed(0)
    cfg = dataclasses.replace(load_preset("medium"), factor_qat="ternary")
    moe = SparseMoE(cfg)
    e = cfg.n_experts
    assert e == 256, "the measured shape's bank"

    take_qat_refresh_counts()
    with torch.no_grad():
        mats = moe._banks(torch.bfloat16)
    calls, tensors = take_qat_refresh_counts()
    assert [m.shape[0] for m in mats[:3]] == [e, e, e], "the whole bank, one entry per expert"
    print(
        f"\nmedium bank (E={e}): stacked = {calls} calls / {tensors} tensors",
        flush=True,
    )
    assert calls == 3, f"one call per projection name, not {calls}"
    assert tensors == 3 * e, f"one call must carry E tensors, got {tensors}"

    # The pre-fix seam, measured on the same site: `[1, ·, ·]` per expert.
    take_qat_refresh_counts()
    with torch.no_grad():
        for name in ("gate_u", "up_u", "v"):
            torch.stack(
                [
                    factor_qat_project(
                        getattr(ex, name).unsqueeze(0), "ternary", 1.0, dim=(1, 2)
                    )[0]
                    for ex in moe.experts
                ]
            )
    old_calls, old_tensors = take_qat_refresh_counts()
    print(
        f"medium bank (E={e}): per-tensor = {old_calls} calls / {old_tensors} tensors",
        flush=True,
    )
    assert (old_calls, old_tensors) == (3 * e, 3 * e)
    assert calls < old_calls / 100, "the stack must be E-fold fewer launches"


# ---------------------------------------------------------------------------
# gate h: no host sync anywhere in the refresh path
# ---------------------------------------------------------------------------


def test_refresh_path_never_syncs(monkeypatch):
    """Gate: the quantizer chain reads nothing back to the host.

    Every host-read primitive a torch path can use to turn a device value into
    a Python number is made to raise; the refresh must not call any of them
    (`qat_refresh_tensors` counts shapes, which are host metadata, never data).
    """

    def boom(*_a, **_k):
        raise AssertionError("host sync in the refresh path")

    for name in ("item", "tolist", "cpu", "numpy", "__float__", "__int__"):
        monkeypatch.setattr(torch.Tensor, name, boom)
    torch.manual_seed(0)
    stack = torch.randn(3, 8, 16) * 0.2
    for kind in ("ternary", "binary", "mxfp4"):
        cache = QuantCache()
        cache.refresh_stacked(stack, kind, 1.0, dim=(1, 2))
        cache.refresh(stack[0], kind, 1.0)
        qat_project_stacked(stack, kind, 0.5, dim=(1, 2))
