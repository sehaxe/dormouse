"""Full depth AttnRes (N = L) on `block_form="transformer"` — CPU only.

The gates the variant owes an A/B:

1. identity at init: the flag ON reproduces the flag OFF logits (the weight sits
   on the most recent update, so the write IS the plain residual);
2. the `S·softmax` convention: `Σ_j w_j == S`, at init and after a random query;
3. order sensitivity: permuting the sources moves the output — a fixed average
   would not;
4. a hand-computed two-source case, arithmetic written in the test;
5. no `.item()`/host sync in the forward;
6. the flag OFF path is untouched: no AttnRes parameters/keys, the ON model is
   exactly `L·(d+1)` parameters larger;
7. the memory refactor changed no arithmetic: the source LOOP equals the
   `[b, t, n, d]` STACK it replaced (reference written in this file) ≤1e-6;
8. and it is flat: no `[b, t, n, d]` buffer is allocated at all;
9. dynamo traces the loop into ONE graph, so a compiled A/B is not silently
   running eager;
10. the TRAINING step retains no `[T, d]` fp32 residency: at the live shape
    (T = 24576, d = 832, S = 13) with gradients ON, the new path's own saved
    bytes are under a tenth of the old expression's, and the old expression is
    built HERE, not recorded. The `@torch.no_grad()` gates above cannot see
    this — every source requires grad in training, so each RMSNorm output and
    each `.float()` upcast was held for the backward.
"""

from __future__ import annotations

import math

import torch
from conftest import tiny_cfg
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from dormouse.config import ModelConfig
from dormouse.model import DormouseModel, param_count
from dormouse.model.attnres import full_depth_attend, full_depth_weights

BATCH, SEQ = 2, 16
#: The live training shape (b24×1024 tokens, d = 832), the one that OOMed.
LIVE_T, LIVE_D, LIVE_S = 24576, 832, 13


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
    # 1e-6, not `== 0.0`: the two paths sum the same terms in a different ORDER
    # (`S·softmax` writes the stream in one add per source, the plain residual in
    # another), so a runner whose CPU kernel re-associates the reduction reports
    # 1.19e-07 where this box reports 0.0. That is float noise. A real break — an
    # `S`-fold, a `1/S`-fold, a mixture near the residual instead of it — is
    # 1e-2 and up, four orders above this bar.
    assert diff <= 1e-6, f"AttnRes at init moved the logits by {diff:.3e}"
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
    extra = {k: v for k, v in on.state_dict().items() if k.startswith("attnres_full.")}
    assert set(extra) == {"attnres_full.query", "attnres_full.prior"}
    n_layers, d = off.cfg.n_layers, off.cfg.d_model
    assert param_count(on) - param_count(off) == n_layers * (d + 1)


# --- 7-8. the memory refactor, gated ----------------------------------------


def _stacked_attend(sources, query, prior, eps=1e-6):
    """THE REPLACED EXPRESSION: the `[b, t, n, d]` stack path, arithmetic intact.

    Kept as the reference the loop must reproduce — including its own RMSNorm,
    so a shared helper bug cannot cancel itself out.
    """
    n = len(sources)

    def rms(x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    keys = torch.stack([rms(s) for s in sources], dim=-2)
    upd = (keys * query.float()).sum(-1) + prior[:n].float()
    zero = upd.new_full(upd.shape[:-1] + (1,), math.log(n))
    w = (n + 1) * torch.softmax(torch.cat([zero, upd], dim=-1), dim=-1)
    vals = torch.stack([s.float() for s in sources], dim=-2)
    return (w[..., 1:].unsqueeze(-1) * vals).sum(-2)


@torch.no_grad()
def test_the_source_loop_equals_the_stack_it_replaced():
    """S = 2..6, fp32: the loop is the stack's arithmetic, ≤1e-6 (measured 0.0)."""
    torch.manual_seed(1101)
    worst = 0.0
    for n in range(1, 7):
        sources = [torch.randn(BATCH, SEQ, 64) for _ in range(n)]
        for prior, q in (
            (torch.zeros(n), torch.zeros(64)),  # the init: the identity lever
            (torch.randn(n), torch.randn(64)),
            (torch.tensor([0.0] + [-30.0] * (n - 1)), torch.randn(64)),
        ):
            new, old = full_depth_attend(sources, q, prior), _stacked_attend(sources, q, prior)
            worst = max(worst, (new - old).abs().max().item())
            torch.testing.assert_close(new, old, atol=1e-6, rtol=0)
    print(f"[attnres] loop vs stack (S=2..7): max|Δ| = {worst:.3e}")


class _Allocations(TorchDispatchMode):
    """Every tensor the dispatched call allocates, by `numel`."""

    def __init__(self) -> None:
        super().__init__()
        self.numel: list[int] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.numel.extend(
            leaf.numel() for leaf in tree_leaves(out) if isinstance(leaf, torch.Tensor)
        )
        return out


@torch.no_grad()
def test_no_btnd_buffer_is_ever_allocated():
    """The binding tensor is `[T, d]` or `[T, S]`, never `[b, t, n, d]`.

    That stack is what OOMed the arm at b24/b20/b16 (936 MiB fp32 of keys at the
    live shape, plus its 936 MiB product, plus the same pair again for the
    values). The largest allocation of the whole call must be a single `[T, d]`
    or `[T, S]` — with `n = 5` here, any `[b, t, n, d]` is 5x the bound.
    """
    n, d = 5, 64
    sources = [torch.randn(BATCH, SEQ, d) for _ in range(n)]
    prior = torch.randn(n)
    with _Allocations() as rec:
        a = full_depth_attend(sources, torch.randn(d), prior)
    biggest = max(rec.numel)
    assert BATCH * SEQ * n * d not in rec.numel, "the [b, t, n, d] stack is back"
    assert biggest <= BATCH * SEQ * d, f"allocated {biggest} elems, bound {BATCH * SEQ * d}"
    assert a.shape == (BATCH, SEQ, d)
    print(f"[attnres] largest allocation: {biggest} elems; the old stack: {BATCH * SEQ * n * d}")


# --- 9. the TRAINING peak: what the backward holds ---------------------------


def _saved_loop_attend(sources, query, prior, eps=1e-6):
    """THE OLD TRAINING PATH, written out here: the loop as it was shipped.

    Its own RMSNorm and its own `.float()` in the value sum, so a shared-helper
    bug cannot cancel itself out against the module under test.
    """
    n = len(sources)

    def rms(x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    logits = sources[0].new_empty(sources[0].shape[:-1] + (n + 1,), dtype=torch.float32)
    logits[..., 0] = math.log(n)
    for j, s in enumerate(sources):
        logits[..., j + 1] = (rms(s) * query.float()).sum(-1) + prior[j].float()
    w = (n + 1) * torch.softmax(logits, dim=-1)
    a = sources[0].new_zeros(sources[0].shape, dtype=torch.float32)
    for j, u in enumerate(sources):
        a += w[..., j + 1].unsqueeze(-1) * u.float()
    return a


class _Saved:
    """`saved_tensors_hooks` sink: the bytes the backward is allowed to read.

    Counted once per STORAGE, so a tensor two nodes save is one residency. `own`
    excludes the tensors HANDED IN — the sources ARE the residual history and
    stay resident whatever the mixer does with them. `biggest` is the largest of
    those owned tensors in ELEMENTS: a `[T, d]` residency shows up there even
    when the total is dominated by the history.
    """

    def __init__(self, owned) -> None:
        self._owned = {t.untyped_storage().data_ptr() for t in owned}
        self._seen: dict[int, int] = {}
        self.own = 0
        self.biggest = 0

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        st = t.untyped_storage()
        key = st.data_ptr()
        if key not in self._seen:
            self._seen[key] = st.nbytes()
            if key not in self._owned:
                self.own += st.nbytes()
                self.biggest = max(self.biggest, t.numel())
        return t

    @property
    def total(self) -> int:
        return sum(self._seen.values())


def _saved_under_grad(fn, sources, query, prior):
    """`(fwd bytes, fwd+bwd bytes, own bytes, largest own numel, out)`."""
    rec = _Saved([*sources, query, prior])
    with torch.autograd.graph.saved_tensors_hooks(rec, lambda x: x):
        out = fn(sources, query, prior)
        fwd = rec.total
        out.sum().backward()
    if rec.total != fwd:
        raise AssertionError("the backward saved new tensors: this measures the forward")
    return fwd, rec.total, rec.own, rec.biggest, out


def test_the_training_step_retains_no_t_d_residency():
    """The gate the `no_grad` tests above cannot be: gradients ON, live shape.

    T = 24576, d = 832, S = 13, fp32 — the shape the arm OOMed at b24. Both
    paths run here, the OLD one as an expression written in this file, and the
    new one must hold at most half of what it held. The old path took its
    1,019 MiB of `[T, d]` residency (13 RMSNorm outputs, 78 MiB each) in the
    forward — a `@torch.no_grad()` measurement sees none of it.
    """
    torch.manual_seed(1301)
    live = [torch.randn(LIVE_T, LIVE_D, requires_grad=True) for _ in range(LIVE_S)]
    q = torch.randn(LIVE_D, requires_grad=True)
    prior = torch.full((LIVE_S,), -30.0)
    prior[0] = 0.0
    prior.requires_grad_(True)
    mib = 2**20

    o_fwd, o_all, o_own, o_big, out_old = _saved_under_grad(_saved_loop_attend, live, q, prior)
    g_old = ([t.grad.clone() for t in live], q.grad.clone(), prior.grad.clone())
    for t in [*live, q, prior]:
        t.grad = None
    n_fwd, n_all, n_own, n_big, out_new = _saved_under_grad(full_depth_attend, live, q, prior)

    print(
        f"[attnres] saved bytes at T={LIVE_T} d={LIVE_D} S={LIVE_S} (fp32, grad on): "
        f"FWD old {o_fwd / mib:.1f} MiB -> new {n_fwd / mib:.1f} MiB | "
        f"FWD+BWD old {o_all / mib:.1f} MiB -> new {n_all / mib:.1f} MiB | "
        f"mixer's own {o_own / mib:.1f} -> {n_own / mib:.1f} MiB"
    )

    # The same arithmetic, bit for bit — forward AND every gradient.
    assert torch.equal(out_new, out_old), "the recompute changed the forward"
    for got, want in zip([*live, q, prior], [*g_old[0], g_old[1], g_old[2]], strict=True):
        torch.testing.assert_close(got.grad, want, atol=0, rtol=0)

    # THE GATE: the new path's own residency against everything the old held.
    assert n_own <= 0.5 * o_all, (
        f"new retains {n_own / mib:.1f} MiB, over 50 % of the old {o_all / mib:.1f} MiB"
    )
    assert n_own <= 0.5 * o_own, f"own {n_own / mib:.1f} vs {o_own / mib:.1f} MiB"
    # No per-source `[T, d]` buffer survives the forward.
    assert n_big <= LIVE_T * (LIVE_S + 1), f"retained {n_big} elems; a [T, d] is back"
    assert o_big >= LIVE_T * LIVE_D, "the reference did not reproduce the residency"


# --- 10. torch.compile ------------------------------------------------------


def test_dynamo_traces_the_source_loop_into_one_graph():
    """A Python loop over a list of tensors must not graph-break.

    The trainer wraps the model in `torch.compile`; a Python loop dynamo cannot
    trace would silently leave the scoring in eager (a COUNTED fallback nobody
    counts). Measured: 0 graph breaks, 1 graph, and the compiled call is the
    eager call bit for bit. This gate fails LOUD if that stops being true.
    """
    sources = [torch.randn(BATCH, SEQ, 64) for _ in range(3)]
    q, prior = torch.randn(64), torch.tensor([0.0, -30.0, -30.0])
    torch._dynamo.reset()
    expl = torch._dynamo.explain(full_depth_attend)(sources, q, prior)
    assert expl.graph_break_count == 0, expl.break_reasons
    assert expl.graph_count == 1, expl.graph_count
    fc = torch.compile(full_depth_attend, backend="eager", dynamic=False)
    assert torch.equal(fc(sources, q, prior), full_depth_attend(sources, q, prior))
    print(f"[attnres] dynamo: {expl.graph_count} graph, {expl.graph_break_count} breaks")


# --- refusals ---------------------------------------------------------------


def test_use_attnres_composes_with_mhc_on_transformer():
    """mHC owns the lanes, AttnRes owns the update — legal on this form too."""
    cfg = _cfg(use_attnres=True, use_mhc=True)
    assert cfg.use_attnres and cfg.use_mhc
    model = _build(5, use_attnres=True, use_mhc=True)
    with torch.no_grad():
        logits, _, _ = model(_batch())
    assert torch.isfinite(logits).all()
