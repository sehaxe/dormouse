"""The routed pass's cheap half as ONE kernel: softmax + LFB bias + top-k, and a
counting permutation that replaces the stable argsort.

**Why these two and nothing else.** `scripts/kernel_profile.py` puts 31.6 ms of
the m400 step in `moe_dispatch`; `scripts/dispatch_census.py` prices that region
op by op and shows which part of it is routing arithmetic. Two rows are pure
routing and are several launches each:

* the route: `softmax(logits)` + `logits + lfb_bias` + `topk` + `gather`
  (`experts.SparseMoE.forward`) — 4 tensor ops, and `topk` alone is 5-6 CUDA
  kernels on this stack (`mbtopk::computeBlockwiseWithinKCounts`,
  `computeBlockDigitCounts`, `computeDigitCumSum`, `gatherTopK`, plus the fill).
  The fusion is legal because every one of them reads and writes the same
  `[n, E]` matrix whose row is INDEPENDENT of every other row: one program per
  token computes the max, the softmax denominator, the biased top-k and the
  gathered gate in registers, and writes only `index [n, k]` and `weight
  [n, k]`. The `[n, E]` values are never written back, so the fused pass is one
  read of 8 MB (bf16, n=8192, E=512) against ~5 passes and ~9 launches.
* the permute: `argsort(safe, stable=True)` over `n·k = 16 384` ids — on this
  stack a CUB radix sort in 12 launches (`DeviceRadixSortHistogram`,
  `OnesweepKernel` ×3, `ExclusiveSum`, ...). The permutation is a stable sort by
  a SMALL integer key, so a counting sort is O(n): `bincount` + `cumsum` give
  each expert its row range, and one program per expert writes its rows in the
  order they appear. Stable by construction, and therefore element-wise equal to
  the stable argsort — `tests/test_moe_route_fused.py` pins that on ids with
  duplicates and on the adaptive width's sentinel slot.

**Tie rule.** `torch.topk` does not define the order of equal values (its CPU
answer for four equal values at k=2 is `[2, 3]`, and `k=3` is `[2, 3, 0]` —
measured, `tests/test_moe_route_fused.py::test_the_tie_fixture_pins_the_rule`).
The fused route is therefore deliberately DEFINED: descending by the raised
score, and among equal scores the LOWEST expert id first. For distinct scores —
every real routing — that is bit-identical to `torch.topk` on the biased logits;
for a tie at the selection boundary the two may name different (equally scored)
experts, and the test says so with a literal instead of pretending otherwise.

**What it costs.** One `[n, E]` read per forward plus a `[n, E]` read+write in
the backward (`dlogits = p·(g − Σ g·p)`, the softmax adjoint, recomputed because
the forward no longer materializes `probs`). The forward cache is `logits` and
`index` — no `[n, E]` softmax output (`bias` is a buffer, not a leaf).
Everything here is device-only: no `.item()`, no boolean-mask index, no
`unique`.

**Fallbacks are COUNTED, never silent** (ADR-0019): no triton / not CUDA → the
caller's own path with `moe_route_fused_fallback` + the reason; a permute the
kernel cannot serve → `torch.argsort(stable=True)` + `moe_permute_fallback`.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception as exc:  # noqa: BLE001 — the reason is COUNTER by the caller
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_WHY: str | None = f"the triton kernels are not importable here: {exc}"
else:
    _TRITON_WHY = None


def reason(x: torch.Tensor) -> str | None:
    """`None` if these kernels can run on `x`'s device, else why not."""
    if _TRITON_WHY is not None:
        return _TRITON_WHY
    if not x.is_cuda:
        return (
            f"input is on {x.device.type}: the fused route and the counting "
            f"permute are CUDA kernels"
        )
    return None


# ---------------------------------------------------------------------------
# the route: softmax + bias + top-k in one pass over [n, E]
# ---------------------------------------------------------------------------


def route_reference(
    logits: torch.Tensor, bias: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernel's math in torch — differentiable, and the CPU gate's reference.

    The op order is the kernel's, not `SparseMoE.forward`'s: the softmax is
    computed in fp32, rounded ONCE to `logits.dtype` (that is what
    `F.softmax` stores, and the two were measured bit-equal at bf16 on this
    stack), and the selection reads `fp32(logits) + fp32(bias)` — the same
    promotion `logits + lfb_bias` performs.
    """
    assert logits.ndim == 2, f"logits must be [n, E], got {tuple(logits.shape)}"
    assert bias.shape == (logits.shape[1],), f"bias must be [E], got {tuple(bias.shape)}"
    assert 1 <= k <= logits.shape[1], f"k={k} outside [1, {logits.shape[1]}]"
    x = logits.float()
    z = x - x.amax(dim=-1, keepdim=True)
    e = z.exp()
    probs = (e / e.sum(dim=-1, keepdim=True)).to(logits.dtype)
    raised = x + bias.float()
    # `torch.topk` order for equal values is unspecified; the rule here is
    # "lowest id first", which `torch.sort(stable=True)` expresses directly.
    index = raised.sort(dim=-1, descending=True, stable=True).indices[..., :k]
    weight = torch.gather(probs, 1, index)
    return index, weight


def route_reference_backward(
    logits: torch.Tensor, index: torch.Tensor, dweight: torch.Tensor
) -> torch.Tensor:
    """The kernel's adjoint in torch: `p·(g − Σ g·p)`, `p` recomputed and rounded.

    `g` is `dweight` scattered at the top-k ids (zero elsewhere). The softmax
    Jacobian is the dense part — a missing `Σ g·p` is a 100 % error on most
    entries, so the CPU gate compares THIS formula with autograd through the same
    chain (the kernel is its transcription; only the card can check the
    transcription, which the skipped GPU gate does).
    """
    x = logits.float()
    z = x - x.amax(dim=-1, keepdim=True)
    e = z.exp()
    p = (e / e.sum(dim=-1, keepdim=True)).to(logits.dtype).float()
    g = torch.zeros_like(p).scatter(1, index, dweight.float())
    dlog = p * (g - (g * p).sum(dim=-1, keepdim=True))
    return dlog.to(logits.dtype)


if triton is not None:

    @triton.jit
    def _route_kernel(
        X, B, IDX, W,
        E: tl.constexpr,
        K: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_E)
        m = cols < E
        x = tl.load(X + row * E + cols, mask=m, other=float("-inf")).to(tl.float32)
        b = tl.load(B + cols, mask=m, other=0.0).to(tl.float32)
        mx = tl.max(x, 0)
        e = tl.where(m, tl.exp(x - mx), 0.0)
        p = e / tl.sum(e, 0)
        # The one rounding the reference path performs (`F.softmax`'s store).
        pdt = p.to(W.dtype.element_ty)
        s = x + b
        for j in tl.static_range(K):
            sj = tl.where(m, s, float("-inf"))
            top = tl.max(sj, 0)
            idx = tl.min(tl.where(sj == top, cols, E))
            tl.store(IDX + row * K + j, idx.to(IDX.dtype.element_ty))
            w = tl.sum(tl.where(cols == idx, pdt.to(tl.float32), 0.0))
            tl.store(W + row * K + j, w.to(W.dtype.element_ty))
            s = tl.where(cols == idx, float("-inf"), s)

    @triton.jit
    def _route_bwd_kernel(
        X, IDX, DW, DLOG,
        E: tl.constexpr,
        K: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_E)
        m = cols < E
        x = tl.load(X + row * E + cols, mask=m, other=float("-inf")).to(tl.float32)
        mx = tl.max(x, 0)
        e = tl.where(m, tl.exp(x - mx), 0.0)
        p = e / tl.sum(e, 0)
        # The adjoint of the forward's ONE rounding: the saved `probs` was the
        # rounded value, so the backward must read the same number.
        p = p.to(DLOG.dtype.element_ty).to(tl.float32)
        g = tl.zeros((BLOCK_E,), dtype=tl.float32)
        for j in tl.static_range(K):
            idx = tl.load(IDX + row * K + j).to(tl.int32)
            gj = tl.load(DW + row * K + j).to(tl.float32)
            g = tl.where(cols == idx, gj, g)
        s = tl.sum(g * p, 0)
        dlog = p * (g - s)
        tl.store(DLOG + row * E + cols, dlog.to(DLOG.dtype.element_ty), mask=m)


class _RouteFused(torch.autograd.Function):
    """`(index, weight)` from one kernel; backward = the softmax adjoint, one kernel.

    The forward saves `logits` (not `probs`): the backward recomputes the softmax
    denominator and writes `dlogits` in the SAME pass that reads the gathered
    `dweight`, so the `[n, E]` matrix is read once and written once — the same
    traffic as `F.softmax`'s backward alone, with the gather's zero-scatter, the
    add's backward and the softmax's own forward output removed from the graph.

    On CPU (the gates) the forward is `route_reference` and the backward is
    `route_reference_backward` — the SAME algebra as the kernels, so a gate can
    compare it with autograd on a device that has no Triton. Production only
    enters this class when `reason()` said the kernel can run; the counted
    fallback is the CALLER's.
    """

    @staticmethod
    def forward(ctx, logits, bias, k):
        assert logits.ndim == 2 and bias.shape == (logits.shape[1],)
        assert logits.is_contiguous() and bias.is_contiguous(), (
            "the route kernel reads dense rows"
        )
        ctx.set_materialize_grads(False)
        if logits.is_cuda:
            n, e = logits.shape
            index = torch.empty(n, k, dtype=torch.int64, device=logits.device)
            weight = torch.empty(n, k, dtype=logits.dtype, device=logits.device)
            block = triton.next_power_of_2(e)
            _route_kernel[(n,)](
                logits, bias, index, weight,
                E=e, K=k, BLOCK_E=block, num_warps=4,
            )
            ctx.save_for_backward(logits, index)
            ctx.mark_non_differentiable(index)
            return index, weight
        index, weight = route_reference(logits, bias, k)
        ctx.save_for_backward(logits, index)
        ctx.mark_non_differentiable(index)
        return index, weight

    @staticmethod
    def backward(ctx, dindex, dweight):
        logits, index = ctx.saved_tensors
        assert dindex is None, "the top-k index is not differentiable"
        if dweight is None:
            return None, None, None
        assert dweight.shape == index.shape and dweight.dtype == logits.dtype
        if logits.is_cuda:
            n, e = logits.shape
            dlog = torch.empty_like(logits)
            block = triton.next_power_of_2(e)
            _route_bwd_kernel[(n,)](
                logits, index, dweight, dlog,
                E=e, K=index.shape[1], BLOCK_E=block, num_warps=4,
            )
            return dlog, None, None
        # The CPU path runs the reference formula, so `_route_bwd_kernel`'s
        # algebra is gated against autograd on a device that has no Triton.
        return route_reference_backward(logits, index, dweight), None, None


def softmax_topk(
    logits: torch.Tensor, bias: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(index [n,k] int64, weight [n,k] logits.dtype)` — the routed pass's head.

    ONE entry point: on CUDA it is one Triton launch, on CPU it is the reference
    (the caller only enters when [`reason`] returned `None`, i.e. on CUDA).
    """
    return _RouteFused.apply(logits, bias, k)


# ---------------------------------------------------------------------------
# the permute: a stable counting sort by expert id
# ---------------------------------------------------------------------------


def count_permute_reference(ids: torch.Tensor, groups: int) -> torch.Tensor:
    """Stable counting sort in torch, one expert at a time — the CPU reference.

    `order[p] = i` such that `ids[order]` is `ids` grouped by expert and, inside
    a group, in ascending `i`. That is `torch.argsort(ids, stable=True)`, which
    is what the `argsort` arm runs and what the test compares against.
    """
    assert ids.ndim == 1 and ids.dtype in (torch.int32, torch.int64)
    assert groups >= 1 and ids.numel() <= (1 << 31) - 1
    from .experts import _bincount  # lazy: `experts` imports this module

    order = torch.empty(ids.numel(), dtype=torch.int64, device=ids.device)
    counts = _bincount(ids, groups)
    assert counts.numel() == groups, "an id outside the bank has no group"
    # ONE host read, and only on the CPU reference: every later write is a
    # python-int slice. The kernel path never leaves the device.
    starts = (torch.cumsum(counts, 0) - counts).tolist()
    for e in range(groups):
        pos = (ids == e).nonzero().flatten()
        if pos.numel():
            order[starts[e] : starts[e] + pos.numel()] = pos
    assert order.numel() == ids.numel()
    return order


if triton is not None:

    @triton.jit
    def _count_permute_kernel(
        IDS, STARTS, ORDER,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        e = tl.program_id(0)
        base = tl.load(STARTS + e).to(tl.int32)
        run = 0
        for off in range(0, N, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            m = idx < N
            v = tl.load(IDS + idx, mask=m, other=-1).to(tl.int32)
            hit = (v == e) & m
            rank = tl.cumsum(hit.to(tl.int32), 0) - 1
            tl.store(
                ORDER + base + run + rank,
                idx.to(ORDER.dtype.element_ty),
                mask=hit,
            )
            run += tl.sum(hit.to(tl.int32))


def count_permute(ids: torch.Tensor, groups: int) -> torch.Tensor:
    """`argsort(ids, stable=True)` as a counting sort — one program per expert.

    `ids` is the routing's flat expert id (`[n·k]`, `< groups`), `groups` is the
    bank size (one past the adaptive width's sentinel). Returns `order [n·k]`
    int64. The result is the STABLE permutation by construction: each program
    walks the ids in ascending position and appends, so equal ids keep their
    original order. Nothing here crosses to the host.
    """
    assert ids.ndim == 1 and groups >= 1, "a permute needs [n·k] ids and a bank size"
    assert ids.is_contiguous(), "the counting sort reads the ids as one dense vector"
    if not ids.is_cuda:
        return count_permute_reference(ids, groups)
    from .experts import _bincount  # lazy: `experts` imports this module

    n = ids.numel()
    ids32 = ids.to(torch.int32)
    counts = _bincount(ids32, groups).to(torch.int32)
    starts = torch.cumsum(counts, 0) - counts
    order = torch.empty(n, dtype=torch.int64, device=ids.device)
    block = 1024
    _count_permute_kernel[(groups,)](
        ids32, starts.contiguous(), order, N=n, BLOCK=block, num_warps=4
    )
    return order
