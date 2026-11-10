"""Gates for `moe_kernel="bucketed"` — the batched bank without the memory cliff.

`moe_kernel="batched"` is one `[E, cap, ·]` bank, so its memory follows
`E · max_group · d`. That is fine for a healthy router and a 12 GiB ask for a
collapsed one (measured 2026-11-04: one expert held 4 096 of 16 384 assignments
on the live bytes). The bucketed arm runs the SAME arithmetic over a plan of
`[n_b, cap_b, ·]` banks — experts within 2x of each other share a bucket, and a
bucket holds at most `MOE_BUCKET_ROWS` rows (an expert past it gets its own).

What is claimed here, in the order the task lists it:

a. one bucket (uniform routing) is the `batched` arm's arithmetic to the dtype's
   bound, dropless and capacity-bound. It was BIT-identical until 2026-11-10:
   the batched arm now applies the router weight in rank space (the OOM fix in
   `SparseMoE._bank_dispatch`), so the two tails round `h·w` differently. The
   fixture's weights are deliberately NOT powers of two — `0.5` scales exactly
   in bf16 and would hide the difference `bucketed` still carries;
b. a pathological routing (one expert owns 90% of the assignments) matches the
   per-expert loop to 1e-6 in fp32 while staying inside `MOE_BUCKET_BYTES` —
   and the single bank REFUSES the same routing (the cliff this arm removes);
c. counts `[1, 1000, 1, 1]` and `[1000, 1, 1, 1]` both pass, two buckets each;
d. a capacity-dropped row contributes exactly 0 to the output and to every
   gradient (its router weight never enters the bank);
e. the arm's only host sync is the documented `counts.tolist()` (the plan's
   `E` ints); no per-expert or per-token read is added;
f. the bucket plan is a deterministic function of the counts;
plus the LOUD `MOE_BUCKET_BYTES` refusal and the config/graph wiring (`bucketed`
is an arm; `full_graph` refuses it for the documented reason).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg
from test_moe_batched import _rel, _SyncCalls

from dormouse.instrument import Counters, LoudError
from dormouse.model import experts
from dormouse.model.experts import MOE_BUCKET_BYTES, MOE_BUCKET_ROWS, SparseMoE

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _moe(**over) -> SparseMoE:
    over.setdefault("rank", 8)
    over.setdefault("d_model", 64)
    over.setdefault("d_ffn", 128)
    over.setdefault("head_dim", 16)
    over.setdefault("n_experts", 4)
    over.setdefault("moe_topk", 2)
    over.setdefault("moe_kernel", "bucketed")
    device = over.pop("device", "cpu")
    return SparseMoE(tiny_cfg(**over), device=device)


def _index_from_counts(counts: list[int], k: int = 1) -> torch.Tensor:
    """A `[sum(counts), k]` routing whose per-expert counts are exactly `counts`."""
    flat = torch.repeat_interleave(
        torch.arange(len(counts)), torch.tensor(counts, dtype=torch.long)
    )
    return flat.reshape(-1, k).repeat_interleave(k, dim=1)


def _index_from_counts(counts: list[int], k: int = 1) -> torch.Tensor:
    """A `[sum(counts), k]` routing whose per-expert counts are exactly `counts`."""
    flat = torch.repeat_interleave(
        torch.arange(len(counts)), torch.tensor(counts, dtype=torch.long)
    )
    return flat.reshape(-1, k).repeat_interleave(k, dim=1)


# ------------------------------------------------------- a. one bucket parity
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("capacity", [0.0, 1.0])
def test_one_bucket_is_the_batched_arm(bf16, capacity):
    """Gate (a): uniform routing plans ONE bucket and reproduces the single bank."""
    torch.manual_seed(11)
    moe = _moe(bf16=bf16, moe_capacity=capacity)
    n, k = 64, 2
    index = (torch.arange(n * k) % moe.n_experts).reshape(n, k)
    # NOT `ones/k` (=0.5, an exact bf16 scaling that hides the tail difference):
    # the routing above is what makes the plan one bucket, the weight is free.
    weight = torch.rand(n, k)
    weight = weight / weight.sum(-1, keepdim=True)
    order, keep, _ = moe._permute(index)
    counts = torch.bincount(index.reshape(-1), minlength=moe.n_experts)
    assert counts.tolist() == [32, 32, 32, 32]
    plan = moe._bucket_plan(counts)
    assert len(plan) == 1 and plan[0] == ([0, 1, 2, 3], 32), plan
    x = torch.randn(n, moe.d)
    counters = Counters()
    with torch.no_grad():
        got = moe._bucketed_dispatch(x, weight, index, order, keep, counters)
        ref = moe._batched_dispatch(x, weight, index, order, keep)
    itemsize = 2 if bf16 else 4
    assert counters["moe_buckets"] == 1
    assert counters["moe_bank_bytes"] == 4 * 32 * moe.d * itemsize
    # Bit-equality held while BOTH arms scaled the `[n·k, f]` output in fp32. The
    # batched arm now applies the router weight in rank space (2026-11-10, the
    # OOM fix in `SparseMoE._bank_dispatch`), so the two tails round `h·w`
    # differently; the bound here is that gate's bound, not a new one. `bucketed`
    # still carries the old tail — giving it the same layout is a follow-up, and
    # this gate is the record that the two are no longer bit-identical.
    bound = 1e-2 if bf16 else 1e-5
    print(
        f"[one bucket] bf16={bf16} capacity={capacity} buckets=1 "
        f"bank={counters['moe_bank_bytes']} B rel={_rel(got, ref):.3e}"
    )
    assert _rel(got, ref) <= bound, f"rel={_rel(got, ref):.3e} exceeds {bound}"


# ------------------------------------------- b. pathological routing / budget
def test_pathological_routing_matches_the_loop_and_stays_in_budget():
    """Gate (b): 90% on one expert — the loop's answer, and inside the budget."""
    torch.manual_seed(3)
    moe = _moe(d_model=32, d_ffn=48, head_dim=8, rank=4, n_experts=32, moe_capacity=0.0)
    n, k = 200, 2
    flat = torch.zeros(n * k, dtype=torch.long)
    flat[int(0.9 * n * k) :] = torch.arange(n * k - int(0.9 * n * k)) % (moe.n_experts - 1) + 1
    index = flat.reshape(n, k)
    weight = torch.ones(n, k) / k
    order, keep, _ = moe._permute(index)
    counts = torch.bincount(index.reshape(-1), minlength=moe.n_experts)
    plan = moe._bucket_plan(counts)
    print(f"[pathological] counts head={counts[:4].tolist()} plan={[(m, c) for m, c in plan]}")

    # The single bank's cliff: one expert holds 360 of 400 assignments, so
    # `MOE_BANK_SAFETY` (8x the uniform 12.5) refuses before allocating.
    with pytest.raises(LoudError) as e:
        moe._batched_cap(index, weight)
    assert "busiest expert holds 360" in str(e.value)

    x = torch.randn(n, moe.d)
    counters = Counters()
    with torch.no_grad():
        got = moe._bucketed_dispatch(x, weight, index, order, keep, counters)
        ref = moe._naive_dispatch(x, weight, index, order, keep, Counters())
    delta = _rel(got, ref)
    assert counters["moe_buckets"] == len(plan) == 2, counters["moe_buckets"]
    assert counters["moe_bank_bytes"] <= MOE_BUCKET_BYTES
    assert delta <= 1e-6, f"{delta:.3e} exceeds 1e-6"
    print(
        f"[pathological] buckets={counters['moe_buckets']} "
        f"bank={counters['moe_bank_bytes']} B <= {MOE_BUCKET_BYTES} B "
        f"max|Δ|={delta:.3e}"
    )


def test_an_over_budget_plan_is_loud(monkeypatch):
    """The budget refusal names the numbers and an escape; never clamps."""
    torch.manual_seed(1)
    moe = _moe()
    index = (torch.arange(32) % moe.n_experts).reshape(16, 2)
    weight = torch.ones(16, 2) / 2
    order, keep, _ = moe._permute(index)
    x = torch.randn(16, moe.d)
    monkeypatch.setattr(experts, "MOE_BUCKET_BYTES", 1024)
    with pytest.raises(LoudError) as e:
        moe._bucketed_dispatch(x, weight, index, order, keep, Counters())
    msg = str(e.value)
    print(f"[budget] {msg}")
    assert "MiB" in msg and "Escape:" in msg and "moe_kernel='naive'" in msg
    assert "busiest expert" in msg


# ------------------------------------------------- c. both count orderings
@pytest.mark.parametrize("counts", [[1, 1000, 1, 1], [1000, 1, 1, 1]])
def test_thousand_row_expert_plans_two_buckets(counts):
    """Gate (c): a 1 000-row expert and three 1-row experts, either order."""
    torch.manual_seed(2)
    moe = _moe(moe_topk=1, moe_capacity=0.0)
    index = torch.repeat_interleave(
        torch.arange(4), torch.tensor(counts, dtype=torch.long)
    ).reshape(-1, 1)
    weight = torch.ones(index.shape)
    order, keep, _ = moe._permute(index)
    got_counts = torch.bincount(index.reshape(-1), minlength=4)
    assert got_counts.tolist() == counts
    plan = moe._bucket_plan(got_counts)
    x = torch.randn(index.shape[0], moe.d)
    counters = Counters()
    with torch.no_grad():
        got = moe._bucketed_dispatch(x, weight, index, order, keep, counters)
        ref = moe._naive_dispatch(x, weight, index, order, keep, Counters())
    delta = _rel(got, ref)
    print(f"[counts {counts}] buckets={counters['moe_buckets']} plan={plan} Δ={delta:.3e}")
    assert counters["moe_buckets"] == len(plan) == 2
    assert delta <= 1e-6


# ------------------------------------------------ d. capacity drops are zero
def test_dropped_rows_are_exactly_zero():
    """Gate (d): a dropped row's value and router weight never reach a bank."""
    torch.manual_seed(6)
    moe = _moe(moe_capacity=1.0)
    n, k = 8, 2
    # counts [6, 2, 2, 2]; cap = 4 → expert 0 drops 2 rows, experts 1-3 pad 2.
    index = torch.tensor([[0, 0], [0, 0], [0, 0], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]])
    weight = torch.ones(n, k) / k
    order, keep, dropped = moe._permute(index)
    assert not keep.all() and int(dropped) > 0
    x = torch.randn(n, moe.d, requires_grad=True)
    w = weight.clone().requires_grad_(True)
    out = moe._bucketed_dispatch(x, w, index, order, keep, Counters())
    out.square().sum().backward()
    assert w.grad is not None
    dropped_grad = float(w.grad.reshape(-1)[order[~keep]].abs().max())
    assert dropped_grad == 0.0, f"a dropped slot got router grad {dropped_grad}"

    # Changing the weight of every dropped slot cannot move the output: the
    # dump row is sliced off and `keep` zeroes the multiply, exactly as eager.
    w2 = weight.clone()
    w2.reshape(-1)[order[~keep]] = 123.0
    with torch.no_grad():
        out2 = moe._bucketed_dispatch(x.detach(), w2, index, order, keep, Counters())
    assert torch.equal(out.detach(), out2), "a dropped slot moved the output"

    moe.zero_grad(set_to_none=True)
    x2 = x.detach().clone().requires_grad_(True)
    ref = moe._naive_dispatch(x2, weight, index, order, keep, Counters())
    delta = _rel(out.detach(), ref.detach())
    buckets = len(moe._bucket_plan(torch.bincount(index.reshape(-1), minlength=4)))
    print(
        f"[drops] dropped={int(dropped)} buckets={buckets} "
        f"dropped_w_grad={dropped_grad} max|Δout|={delta:.3e}"
    )
    assert delta <= 1e-6


# ------------------------------------------------------- e. the one host read
@pytest.mark.parametrize("capacity", [0.0, 1.0])
def test_the_only_sync_is_the_plan_read(capacity):
    """Gate (e): ONE `counts.tolist()` (E ints) — no per-expert/per-token read."""
    torch.manual_seed(8)
    moe = _moe(moe_capacity=capacity)
    x = torch.randn(64, moe.d)
    torch.manual_seed(0)
    with torch.no_grad():
        probs = F.softmax(moe.router(x), dim=-1)
        weight, index = probs.topk(2, dim=-1)
    weight = weight / weight.sum(dim=-1, keepdim=True)
    with _SyncCalls() as spy:
        moe._dispatch(x, weight, index, Counters())
    print(f"[no sync] capacity={capacity} reads={spy.names}")
    assert spy.names == ["tolist"], (
        f"the bucket plan's counts read is the ONE documented sync: {spy.names}"
    )


# ------------------------------------------------------- f. deterministic plan
def test_the_bucket_plan_is_deterministic():
    """Gate (f): same counts → same plan, whatever order the rows arrived in."""
    moe = _moe(n_experts=8, moe_capacity=0.0)
    counts = torch.tensor([7, 1, 30, 1, 12, 3, 1, 30])
    first = moe._bucket_plan(counts)
    second = moe._bucket_plan(counts)
    assert first == second, (first, second)
    flat = torch.repeat_interleave(torch.arange(8), counts)
    perm = torch.randperm(flat.numel(), generator=torch.Generator().manual_seed(4))
    shuffled = torch.bincount(flat[perm], minlength=8)
    assert torch.equal(shuffled, counts)
    assert moe._bucket_plan(shuffled) == first
    print(f"[deterministic] plan={first}")


# ------------------------------------------------------------- the wiring
def test_the_literal_and_the_graph_refusal():
    """`bucketed` is a declared arm; `full_graph` refuses it with the reason."""
    from dormouse.config import MOE_KERNELS, TrainConfig, validate
    from dormouse.train.graph import full_graph_refusal

    assert "bucketed" in MOE_KERNELS
    validate(tiny_cfg(moe_kernel="bucketed", moe_capacity=1.0, moe_topk=2))
    why = full_graph_refusal(
        TrainConfig(), tiny_cfg(moe_kernel="bucketed", moe_capacity=1.0, use_msa=False)
    )
    print(f"[full_graph] {why}")
    assert why is not None and "bucketed" in why and "Escape:" in why


@CUDA
def test_the_bucketed_arm_runs_on_cuda():
    """CUDA half: the plan and the three bmms per bucket are a real program."""
    moe = _moe(moe_capacity=0.0, device="cuda")
    x = torch.randn(64, moe.d, device="cuda")
    index = (torch.arange(128, device="cuda") % moe.n_experts).reshape(64, 2)
    weight = torch.ones(64, 2, device="cuda") / 2
    order, keep, _ = moe._permute(index)
    counters = Counters()
    with torch.no_grad():
        got = moe._bucketed_dispatch(x, weight, index, order, keep, counters)
        ref = moe._naive_dispatch(x, weight, index, order, keep, Counters())
    torch.cuda.synchronize()
    print(f"[cuda] buckets={counters['moe_buckets']} bank={counters['moe_bank_bytes']}")
    assert torch.isfinite(got).all()
    assert _rel(got, ref) <= 1e-6


def test_the_row_and_byte_budgets_are_the_documented_ones():
    assert MOE_BUCKET_ROWS == 2048
    assert MOE_BUCKET_BYTES == 256 << 20
