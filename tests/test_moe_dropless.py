"""The dropless MoE route: the static row-block bound and the empty-block exit.

`moe_capacity = 0` keeps every routing assignment, so the row-block grid cannot
take the tight capacity bound any more — the max group size lives in `offs`, on
the device, and reading it back is the sync ADR-0018 rule 2 bans in a step. The
kernels therefore take a STATIC bound the host reads from a SHAPE
(`moe_kernels._dropless_cap`, `R` rows in one group) and every program exits
on-device when its block is empty: an empty expert costs a program launch, two
`offs` loads and a branch, never GEMM work.

The gates here, in the order the mechanism needs them:

1. **the static bound serves ragged groups**, including several EMPTY ones and a
   1-row group — and the test asserts the launch really had more blocks than had
   work, so the early-exit path is the one under test (the tight-`cap` sibling is
   `tests/test_moe_kernel.py::test_ragged_offsets_hold_for_empty_and_short_groups`);
2. **the dropless dispatch is the reference loop's function**, fwd and fwd+bwd,
   per parameter and on `dx`;
3. **the segmented forward still routes identically** with the grouped GEMM
   (the long-context claim, now on the arm that can actually serve `b1`'s E=1024
   instead of ~1024 tiny matmuls per pass);
4. **the fallback is COUNTED**, with its reason, wherever the kernel cannot run.

Every CUDA gate is skipped without a card; the CPU gate (4) is not.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from conftest import SEQ, tiny_cfg

from dormouse.instrument import Counters
from dormouse.model import DormouseModel
from dormouse.model import moe_kernels as mk
from dormouse.model.experts import SparseMoE

DEV = "cuda"
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
BM = mk.TILES["BLOCK_M"]
#: The b1 shape's row count per expert pass at topk=2 (the bench's shape too).
B1_TOKENS = 16 * 1024


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """`max|a-b| / max|b|` — the form `tests/test_moe_kernel.py` reports."""
    d = (a.float() - b.float()).abs().max()
    return float(d) / float(b.float().abs().max().clamp_min(1e-12))


def _moe(e: int = 4, **over) -> SparseMoE:
    """A tiny dropless MoE on the device: capacity 0, the grouped GEMM asked for."""
    cfg = tiny_cfg(moe_topk=2, moe_capacity=0, moe_kernel="triton", n_experts=e, **over)
    return SparseMoE(cfg, device=DEV)


def _route(moe: SparseMoE, x: torch.Tensor, hi: int | None = None):
    """`(weight, index, order, keep)` from the real router, optionally over `hi` experts.

    `hi` is how an EMPTY expert is built without hand-writing `offs`: the router's
    top-k is clipped to `[0, hi)`, so every expert at or above `hi` gets no rows
    at all and the kernel's grid still covers it.
    """
    with torch.no_grad():
        probs = F.softmax(moe.router(x), dim=-1)
        if hi is not None:
            probs = probs[:, :hi]
        weight, index = probs.topk(2, dim=-1)
        weight = weight / weight.sum(-1, keepdim=True)
    order, keep, _ = moe._permute(index)
    return weight, index, order, keep


def _blocks(rows: int) -> int:
    return -(-rows // BM)


# --------------------------------------------------- 1. the static bound, ragged


@CUDA
@pytest.mark.parametrize("bf16,bound", [(False, 1e-4), (True, 2e-3)])
def test_the_static_bound_serves_empty_and_short_groups(bf16, bound):
    """`cap=None` (the dropless bound) vs a per-group reference, empties included.

    Three of the six groups own NO rows and one owns a single row, so the grid
    this launches has more blocks than have work — the early-exit path. The
    assertion that the launch really was over-sized is what keeps this gate from
    silently becoming the tight-`cap` gate next door.
    """
    torch.manual_seed(0)
    e, d, r, f = 6, 96, 24, 130
    counts = torch.tensor([0, 1, 3, 7, 0, 2], device=DEV, dtype=torch.int32)
    rows_total = int(counts.sum())
    offs = torch.zeros(e + 1, dtype=torch.int32, device=DEV)
    offs[1:] = torch.cumsum(counts, 0)
    seg = torch.repeat_interleave(torch.arange(e, device=DEV), counts.to(torch.int64))
    dt = torch.bfloat16 if bf16 else torch.float32
    xg = torch.randn(rows_total, d, device=DEV).to(dt)
    gu = torch.randn(e, d, r, device=DEV).to(dt)
    uu = torch.randn(e, d, r, device=DEV).to(dt)
    gs = torch.randn(e, r, device=DEV).to(dt)
    us = torch.randn(e, r, device=DEV).to(dt)
    v = torch.randn(e, r, f, device=DEV).to(dt)

    got = mk.expert_gemm_glu(xg, gu, uu, gs, us, v, offs, cap=None)
    ref = torch.zeros(rows_total, f, dtype=dt, device=DEV)
    for i in range(e):
        sel = seg == i
        if not bool(sel.any()):
            continue  # an EMPTY expert contributes nothing, and must not be filled
        # The PADDED arm's own chain (bf16 matmul -> scale -> silu -> @v), which
        # is what the kernel's rounding order mirrors on purpose.
        g = (xg[sel] @ gu[i]) * gs[i]
        u = (xg[sel] @ uu[i]) * us[i]
        ref[sel] = (F.silu(g) * u) @ v[i]

    launched = _blocks(rows_total) * e
    with_work = sum(_blocks(int(c)) for c in counts)
    rel = _rel(got, ref)
    print(
        f"[dropless ragged] bf16={bf16} counts={counts.tolist()} rows={rows_total} "
        f"blocks launched={launched} with_work={with_work} max_rel={rel:.3e}"
    )
    assert launched > with_work, "the empty blocks must be in the launch (early-exit path)"
    assert int(counts[0]) == 0 and int(offs[0]) == 0 and int(offs[-1]) == rows_total
    assert rel <= bound, f"{rel:.3e} exceeds {bound}"

    # The plain grouped GEMM rides the same bound and the same exit.
    h = torch.randn(rows_total, r, device=DEV).to(dt)
    y = mk.expert_gemm(h, v, offs, cap=None)
    y_ref = torch.zeros(rows_total, f, dtype=dt, device=DEV)
    for i in range(e):
        sel = seg == i
        if bool(sel.any()):
            y_ref[sel] = h[sel] @ v[i]
    rel_y = _rel(y, y_ref)
    print(f"[dropless ragged] grouped GEMM max_rel={rel_y:.3e}")
    assert rel_y <= bound


# ------------------------------------- 2. the dispatch arm is the same function


@CUDA
@pytest.mark.parametrize("empty", [False, True])
def test_dropless_dispatch_matches_the_reference_loop(empty):
    """The grouped GEMM with the static bound == the per-expert reference loop.

    `empty=True` clips the router to two of the four experts, so half the grid's
    experts own nothing: the arm that runs there is the early-exit path, and the
    reference loop simply never visits them.
    """
    torch.manual_seed(0)
    moe = _moe(e=4)
    x = torch.randn(64, moe.d, device=DEV)
    weight, index, order, keep = _route(moe, x, hi=2 if empty else None)
    with torch.no_grad():
        got = moe._triton_dispatch(x, weight, index, order, keep, dropless=True)
        ref = moe._naive_dispatch(x, weight, index, order, keep, Counters())
    used = int(torch.unique(index).numel())
    rel = _rel(got, ref)
    print(
        f"[dropless dispatch] empty={empty} experts_used={used}/{moe.n_experts} "
        f"rows={x.shape[0]} kept={int(keep.sum())} max_rel={rel:.3e}"
    )
    assert rel <= 1e-4, f"{rel:.3e}"
    if empty:
        assert used == 2, "the fixture must leave experts empty"


@CUDA
def test_dropless_gradients_match_the_reference_loop():
    """fwd+bwd parity: every expert parameter and `dx`, grouped GEMM vs the loop.

    The routing here hits EVERY expert (no `hi`): an expert with no rows gets a
    zero gradient from the grouped GEMM's `_wgrad` (`out` is filled for all `E`)
    and NO gradient from the module-based loop, so comparing parameters that one
    arm never touched would compare `zeros` against `None` — a real asymmetry of
    the two arms, and not the parity this gate is about.
    """
    torch.manual_seed(0)
    moe = _moe(e=4)
    x = torch.randn(64, moe.d, device=DEV)
    weight, index, order, keep = _route(moe, x)

    def run(arm) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        moe.zero_grad(set_to_none=True)
        xi = x.detach().clone().requires_grad_(True)
        arm(xi).float().pow(2).mean().backward()
        grads = {n: p.grad.detach().clone() for n, p in moe.experts.named_parameters()}
        assert all(p.grad is not None for p in moe.experts.parameters())
        return grads, xi.grad.detach().clone()

    ref, dx_ref = run(lambda xi: moe._naive_dispatch(xi, weight, index, order, keep, Counters()))
    got, dx = run(lambda xi: moe._triton_dispatch(xi, weight, index, order, keep, dropless=True))
    per = {n: _rel(got[n], ref[n]) for n in ref}
    worst = max(per.values())
    dx_rel = _rel(dx, dx_ref)
    print(
        f"[dropless grads] worst param rel={worst:.3e} dx rel={dx_rel:.3e} "
        f"worst={max(per, key=per.get)}"
    )
    assert worst <= 1e-4, f"parameter grads disagree: {per}"
    assert dx_rel <= 1e-4, f"dx disagrees: {dx_rel:.3e}"


# ------------------------------- 3. the segmented forward on the dropless arm


def _router_spy(monkeypatch) -> list[torch.Tensor]:
    """Record every routed `index` the MoE dispatches, in call order."""
    seen: list[torch.Tensor] = []
    real = SparseMoE._dispatch

    def spy(self, x, weight, index, counters):
        seen.append(index.detach().clone())
        return real(self, x, weight, index, counters)

    monkeypatch.setattr(SparseMoE, "_dispatch", spy)
    return seen


@CUDA
def test_segmented_dropless_triton_routes_identically(byte_tensor, monkeypatch):
    """The long-context gate, on the arm the sparse presets will actually run.

    Same claim as `tests/test_longctx.py::test_dropless_sparse_routing_...` — the
    segmented window routes EXACTLY the one-shot indices and the logits agree —
    but with the grouped GEMM doing the work (`moe_triton_gemm` on the counter)
    instead of the reference loop, and with the static row-block bound in play.
    """
    torch.manual_seed(0)
    model = DormouseModel(tiny_cfg(moe_topk=2, moe_capacity=0, moe_kernel="triton"), device=DEV)
    ids, targets = byte_tensor()
    ids, targets = ids.to(DEV), targets.to(DEV)
    b, t = ids.shape
    T, S = model.cfg.max_iter, t // 16
    seen = _router_spy(monkeypatch)
    with torch.no_grad():
        ref_logits, _, ref_c = model(ids, targets)
        ref = list(seen)
        seen.clear()
        logits, _, c = model(ids, targets, segment_len=16)
        seg = list(seen)
    monkeypatch.undo()
    assert len(ref) == T and len(seg) == T * S, f"{len(ref)} / {len(seg)} dispatches"
    same = []
    for i in range(T):
        one = ref[i].reshape(b, t, -1)
        many = torch.cat([seg[s * T + i].reshape(b, t // S, -1) for s in range(S)], dim=1)
        same.append(bool(torch.equal(one, many)))
    d = float((logits - ref_logits).abs().max())
    print(
        f"[dropless triton] max|dlogits|={d:.3e} routing identical={same} "
        f"dropless={c['moe_dropless']} triton={c['moe_triton_gemm']} "
        f"naive={c.get('moe_naive_dispatch', 0)}"
    )
    assert all(same), f"the routing must be identical: {same}"
    assert d <= 1e-4, f"segmented logits differ by {d}"
    assert c["moe_triton_gemm"] == T * S and "moe_naive_dispatch" not in c
    assert c.drain().get("moe_dropped_assignments", 0) == 0
    assert ref_c["moe_triton_gemm"] == T


# ------------------------------------------------- 4. the counted fallback


def test_the_fallback_is_counted_where_the_kernel_cannot_run(byte_tensor):
    """CPU: the grouped GEMM cannot run, so the reference loop does — and says so.

    The reason is the DEVICE, not the capacity: `capacity=0` stopped being a
    reason the moment the kernel could take the static bound. A reader of the
    log line sees `moe_dropless`, `moe_naive_dispatch` and `moe_triton_fallback`
    together, so the arm that produced the loss is never a guess.
    """
    torch.manual_seed(0)
    model = DormouseModel(tiny_cfg(moe_topk=2, moe_capacity=0, moe_kernel="triton"))
    ids, targets = byte_tensor()
    _, loss, c = model(ids, targets)
    assert loss is not None and torch.isfinite(loss)
    print(f"[dropless fallback] {c.line()} why={c.why}")
    assert c["moe_dropless"] == model.cfg.max_iter
    assert c["moe_naive_dispatch"] == model.cfg.max_iter
    assert c["moe_triton_fallback"] == model.cfg.max_iter
    assert "input is on cpu" in c.why
    assert "capacity=0" not in c.why, "the capacity is no longer a reason the kernel is out"


def test_moe_kernel_naive_is_the_configs_own_choice(byte_tensor):
    """`moe_kernel=naive` + dropless: the reference loop, and no triton counter."""
    torch.manual_seed(0)
    model = DormouseModel(tiny_cfg(moe_topk=2, moe_capacity=0, moe_kernel="naive"))
    ids, targets = byte_tensor()
    _, _, c = model(ids, targets)
    print(f"[dropless naive] {c.line()} why={c.why}")
    assert c["moe_dropless"] == model.cfg.max_iter
    assert c["moe_naive_dispatch"] == model.cfg.max_iter
    assert "moe_triton_fallback" not in c and "moe_triton_gemm" not in c
    assert "moe_kernel=naive" in c.why
    assert SEQ  # the fixture's window, named so the file reads as the longctx lane
