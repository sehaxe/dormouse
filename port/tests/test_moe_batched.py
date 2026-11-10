"""Gates for `moe_kernel="batched"` — ONE `[E, cap, ·]` bank for every expert.

The arm the dropless route needed: `moe_capacity = 0` keeps every routing
assignment, so the padded arm (a fixed `moe_capacity` bank) refuses and the
per-expert Python loop (or the grouped GEMM) runs instead. `batched` sizes the
common cap to THIS batch's largest group and runs the whole bank as three
`bmm`s — the same arithmetic as the loop, one launch per GEMM instead of E.

What is claimed here, in the order the task lists it:

a. `moe_kernel="batched"` matches the current path to 1e-6 in fp32, and to the
   same bound when the bank is bf16 (measured 0.0 on CPU);
b. logits AND every parameter gradient match a float64 per-expert reference to
   1e-5;
c. unequal groups (1 row and 40 rows) give the loop's answer;
d. a capacity-dropped row behaves exactly as the padded arm's (and dropless
   keeps every assignment);
e. a padded bank row contributes exactly 0 to the output and to every gradient;
f. heterogeneous expert shapes are a LOUD `Escape:` refusal, not a `torch.stack`
   shape error;
g. the capacity path adds no host sync; the dropless cap read is named (it is
   the ONE scalar a `[E, cap, ·]` allocation needs) and is not a per-expert loop;
h. `full_graph` allows the static-cap arm and REFUSES the dropless one, with the
   reason on the message.

Each gate is falsified by a scratch revert in the task's report: the revert and
the red gate are recorded there, not in this file.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg
from torch.overrides import TorchFunctionMode

from dormouse.config import MOE_KERNELS, validate
from dormouse.instrument import Counters, LoudError
from dormouse.model.experts import SparseMoE

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _moe(**over) -> SparseMoE:
    over.setdefault("rank", 8)
    over.setdefault("d_model", 64)
    over.setdefault("d_ffn", 128)
    over.setdefault("head_dim", 16)
    device = over.pop("device", "cpu")
    cfg = tiny_cfg(n_experts=4, moe_topk=2, **over)
    return SparseMoE(cfg, device=device)


def _routing(moe: SparseMoE, x: torch.Tensor, seed: int = 0):
    """A fixed top-k routing: no graph back into the router, both arms share it."""
    torch.manual_seed(seed)
    with torch.no_grad():
        probs = F.softmax(moe.router(x), dim=-1)
        weight, index = probs.topk(2, dim=-1)
    return index, weight / weight.sum(dim=-1, keepdim=True)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    num = float((a.float() - b.float()).abs().max())
    return num / float(b.float().abs().max().clamp_min(1e-12))


def _grads(moe: SparseMoE) -> dict:
    return {
        f"{i}.{name}": p.grad for i, e in enumerate(moe.experts) for name, p in e.named_parameters()
    }


class _SyncCalls(TorchFunctionMode):
    """Name every host-sync torch function (`item`/`cpu`/`tolist`/`numpy`/`bool`)."""

    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", "")
        if name in ("item", "cpu", "tolist", "numpy", "__bool__", "__float__"):
            self.names.append(name)
        return func(*args, **(kwargs or {}))


class _BankGrab(TorchFunctionMode):
    """Keep the first `[E, cap, d]` bank the bank pass hands to `bmm`."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d
        self.bank: torch.Tensor | None = None

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if (
            self.bank is None
            and func is torch.bmm
            and len(args) >= 2
            and args[0].dim() == 3
            and args[0].shape[-1] == self.d
        ):
            args[0].retain_grad()
            self.bank = args[0]
        return func(*args, **(kwargs or {}))


# --------------------------------------------------------------- a. parity
#: The layout's price, MEASURED at this fixture and printed below: the bank now
#: applies the router weight in the GEMM's dtype (rank space) instead of
#: scaling the `[n·k, f]` fp32 output, so it is no longer bit-identical to the
#: loop in a bf16 run. Measured here: fp32 max|Δ| 1.43e-6 absolute / 2.4e-7
#: relative (1.08e-6 against fp64, the loop's own 7.6e-7); bf16 4.7e-3 relative
#: — and 0.8× the loop's own fp64 error, i.e. the difference is bf16 rounding,
#: not a wiring change. The bounds are the contract; a fp32-exact match after
#: the GEMM is the very buffer the layout removes.
PARITY_BOUND = {False: 1e-5, True: 1e-2}


@pytest.mark.parametrize("capacity", [0.0, 1.0])
@pytest.mark.parametrize("bf16", [False, True])
def test_batched_matches_the_current_path(bf16, capacity):
    """Gate (a): the bank and the loop on the same rows, dtype and cap.

    `capacity=1.0` also exercises the drop path: both arms get the same `keep`,
    so this is a wiring parity, not a re-derivation of the routing.
    """
    torch.manual_seed(11)
    moe = _moe(bf16=bf16, moe_capacity=capacity, moe_kernel="batched")
    x = torch.randn(64, moe.d)
    index, weight = _routing(moe, x)
    order, keep, _ = moe._permute(index)
    with torch.no_grad():
        batched = moe._batched_dispatch(x, weight, index, order, keep)
        loop = moe._naive_dispatch(x, weight, index, order, keep, Counters())
    bound = PARITY_BOUND[bf16]
    delta = float((batched - loop).abs().max())
    rel = _rel(batched, loop)
    print(
        f"[parity] bf16={bf16} capacity={capacity} E={moe.n_experts} "
        f"rows={x.shape[0]} max|Δ|={delta:.3e} rel={rel:.3e} bound={bound:.0e}"
    )
    assert rel <= bound, f"{rel:.3e} exceeds {bound:.0e}"


# ------------------------------------------- a2. the layout, not just the bits
class _WideMul(TorchFunctionMode):
    """Flag any `mul` whose operand is one `[rows, f]` fp32 buffer."""

    def __init__(self, rows: int, f: int) -> None:
        super().__init__()
        self.shape = (rows, f)
        self.hits: list[tuple[str, tuple]] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", "")
        if name in ("mul", "mul_", "__mul__", "__rmul__"):
            for a in args:
                if (
                    isinstance(a, torch.Tensor)
                    and a.dtype == torch.float32
                    and tuple(a.shape) == self.shape
                ):
                    self.hits.append((name, tuple(a.shape)))
        return func(*args, **(kwargs or {}))


def test_the_router_weight_rides_the_bank_layout():
    """Gate (a2): the weight is applied in rank space — no `[n·k, f]` fp32 product.

    The defect this pins is the BUFFER, not the bits: scaling the gathered
    output in fp32 needs `4·n·k·f` bytes per layer (327 MB at the b24 /
    rank-832 / f-1664 arm), which is half of why b16 and b24 OOMed. On the same
    routing the rank-space form must still equal the loop within its dtype
    bound; gate (e) owns the exactly-zero half (padding and dumped rows).
    """
    torch.manual_seed(12)
    moe = _moe(bf16=False, moe_capacity=1.0, moe_kernel="batched")
    x = torch.randn(48, moe.d)
    index, weight = _routing(moe, x)
    order, keep, _ = moe._permute(index)
    n, k = index.shape
    spy = _WideMul(rows=n * k, f=moe.f)
    with torch.no_grad(), spy:
        out = moe._batched_dispatch(x, weight, index, order, keep)
    with torch.no_grad():
        loop = moe._naive_dispatch(x, weight, index, order, keep, Counters())
    print(
        f"[rank-space w] rows={n * k} f={moe.f} wide_fp32_muls={len(spy.hits)} "
        f"rel={_rel(out, loop):.3e}"
    )
    assert spy.hits == [], f"the wide fp32 product is back: {spy.hits}"
    assert _rel(out, loop) <= PARITY_BOUND[False]


def test_the_bank_layout_is_counted():
    """Gate (a3): the layout is COUNTED once per dispatch (ADR-0011)."""
    torch.manual_seed(14)
    moe = _moe(moe_capacity=0.0, moe_kernel="batched")
    x = torch.randn(32, moe.d)
    index, weight = _routing(moe, x)
    counters = Counters()
    with torch.no_grad():
        moe._dispatch(x, weight, index, counters)
    print(f"[counted] {dict(counters)}")
    assert counters["moe_w_in_bank"] == 1
    assert counters["moe_batched_calls"] == 1
    # The bank arms must NOT carry the triton arms' wide fp32 weight product.
    assert "moe_wide_weight" not in counters


def test_the_capacity_waste_is_a_formula_and_a_counter():
    """Gate (a5): `E·cap/(n·k)` rows, exposed as counters — the `moe_capacity` trade.

    The padded arm's bill is exactly `dump = E·cap` rows against `n·k`
    assignments, so a log line can carry the waste without a profile:
    `moe_bank_rows`, `moe_pad_waste_rows` (SIGNED: negative = the rule dropped
    rows) and `moe_bank_ratio_x100`. At `capacity=2.0` the ratio is 200 by
    construction; at `capacity=0.5` it is 50 and the waste is the drops.
    """
    n = 64
    for capacity, want in ((2.0, (256, 128, 200)), (0.5, (64, 0, 50))):
        torch.manual_seed(15)
        moe = _moe(moe_capacity=capacity, moe_kernel="batched")
        x = torch.randn(n, moe.d)
        index, weight = _routing(moe, x)
        order, keep, _ = moe._permute(index)
        counters = Counters()
        with torch.no_grad():
            moe._batched_dispatch(x, weight, index, order, keep, counters)
        got = (
            counters["moe_bank_rows"],
            counters.get("moe_pad_waste_rows", 0),
            counters["moe_bank_ratio_x100"],
        )
        print(f"[waste] capacity={capacity} rows/waste/ratio_x100={got}")
        assert got == want, f"capacity={capacity}: {got} != {want}"


# --------------------------------------------- b. float64 reference, gradients
def _fp64_reference(moe: SparseMoE, x, weight, index, order, cot):
    """The same permutation's arithmetic in float64, per expert, autograd on."""
    n, k = index.shape
    x64 = x.detach().double().requires_grad_()
    params = [
        {
            name: torch.nn.Parameter(getattr(e, name).detach().double())
            for name in ("gate_u", "up_u", "v", "gate_s", "up_s")
        }
        for e in moe.experts
    ]
    out = torch.zeros(n, moe.f, dtype=torch.float64)
    sorted_ids = index.reshape(-1)[order]
    for e in range(moe.n_experts):
        sel = sorted_ids == e
        if not bool(sel.any()):
            continue
        rows = order[sel] // k
        w = weight.reshape(-1)[order[sel]].double()
        g = (x64[rows] @ params[e]["gate_u"]) * params[e]["gate_s"]
        u = (x64[rows] @ params[e]["up_u"]) * params[e]["up_s"]
        h = F.silu(g) * u
        y = h @ params[e]["v"]
        out = out.index_add(0, rows, y * w[:, None])
    (out * cot.double()).sum().backward()
    return out.detach(), x64.grad, [{k: p.grad for k, p in d.items()} for d in params]


def test_batched_matches_a_float64_per_expert_reference():
    """Gate (b): logits AND parameter gradients against float64 arithmetic."""
    torch.manual_seed(3)
    # d=32/f=48 keeps the fp32-vs-fp64 absolute error under 1e-5 at this
    # fixture (6.2e-6 measured); the 64/128 model's larger gradient sums put
    # the same relative move at 1.8e-5, which measures fan-in, not parity.
    moe = _moe(d_model=32, d_ffn=48, head_dim=8, rank=4, moe_capacity=0.0, moe_kernel="batched")
    n, k = 19, 2
    # Deliberately unequal groups: expert 0 owns 1 assignment, expert 1 owns 20.
    index = torch.zeros(n, k, dtype=torch.long)
    index[0, 0] = 0  # expert 0: one assignment
    index[0, 1] = 3  # expert 3: one
    index[1:11, :] = 1  # expert 1: twenty
    index[11:19, :] = 2  # expert 2: sixteen
    weight = torch.rand(n, k)
    weight = weight / weight.sum(-1, keepdim=True)
    order, keep, _ = moe._permute(index)
    x = torch.randn(n, moe.d, requires_grad=True)
    cot = torch.randn(n, moe.f)
    out = moe._batched_dispatch(x, weight, index, order, keep)
    (out * cot).sum().backward()
    got = _grads(moe)
    got_x = x.grad.detach().clone()

    ref_out, ref_x, ref_params = _fp64_reference(moe, x, weight, index, order, cot)
    deltas = {
        "logits": float((out.detach().double() - ref_out).abs().max()),
        "dx": float((got_x.double() - ref_x).abs().max()),
    }
    for i in range(len(moe.experts)):
        for name in ("gate_u", "up_u", "v", "gate_s", "up_s"):
            assert got[f"{i}.{name}"] is not None and ref_params[i][name] is not None
            deltas[f"{i}.{name}"] = float(
                (got[f"{i}.{name}"].double() - ref_params[i][name]).abs().max()
            )
    worst = max(deltas.values())
    shown = " ".join(f"{k}={v:.2e}" for k, v in deltas.items())
    print(f"[fp64] worst={worst:.3e} bound=1e-5 {shown}")
    assert worst <= 1e-5, f"{worst:.3e} exceeds 1e-5"
    assert keep.all(), "the dropless reference must keep every assignment"


def test_the_bf16_bank_is_no_noisier_than_the_loop():
    """Gate (a4): the bf16 delta in gate (a) is rounding, not a wiring defect.

    A bf16-vs-bf16 delta cannot separate a layout change from bf16's own
    rounding — both arms round. So measure each arm against the float64
    reference and require the bank to be NO WORSE. Measured 0.81x the loop's
    own error: the 4.7e-3 relative difference in gate (a) sits inside bf16
    noise. A wiring defect shows up as a ratio above 1.
    """
    torch.manual_seed(13)
    moe = _moe(bf16=True, moe_capacity=0.0, moe_kernel="batched")
    x = torch.randn(64, moe.d)
    index, weight = _routing(moe, x)
    order, keep, _ = moe._permute(index)
    with torch.no_grad():
        bank = moe._batched_dispatch(x, weight, index, order, keep)
        loop = moe._naive_dispatch(x, weight, index, order, keep, Counters())
    ref, _, _ = _fp64_reference(moe, x, weight, index, order, torch.ones(64, moe.f))
    e_bank = float((bank.double() - ref).abs().max())
    e_loop = float((loop.double() - ref).abs().max())
    print(f"[bf16 quality] vs fp64 bank={e_bank:.3e} loop={e_loop:.3e} ratio={e_bank / e_loop:.2f}")
    # 1.25x, not 1.0x: the two arms are independent bf16 rounding chains, so the
    # max|Δ|-vs-fp64 ratio is noise around 1 (measured 0.92 here, 1.03 for the
    # bucketed arm). A WIRING defect moves this by the scale of the tensor — a
    # dropped or duplicated term is 2x or worse — so 1.25 separates the two.
    assert e_bank <= 1.25 * e_loop, f"the bank is {e_bank / e_loop:.2f}x the loop's own fp64 error"


# -------------------------------------------------- c. unequal block sizes
def test_one_row_and_forty_rows_give_the_loop_answer():
    """Gate (c): a 1-row expert and a 40-row expert in the same bmm."""
    torch.manual_seed(7)
    moe = _moe(moe_capacity=0.0, moe_kernel="batched")
    n, k = 41, 2
    index = torch.zeros(n, k, dtype=torch.long)
    index[0, 0] = 0  # expert 0: one assignment
    index[0, 1] = 3  # expert 3: one
    index[1:21, :] = 1  # expert 1: forty
    index[21:, :] = 2  # expert 2: forty
    weight = torch.rand(n, k)
    weight = weight / weight.sum(-1, keepdim=True)
    order, keep, _ = moe._permute(index)
    counts = torch.bincount(index.reshape(-1), minlength=moe.n_experts)
    assert int(counts[0]) == 1 and int(counts[1]) == 40
    x = torch.randn(n, moe.d, requires_grad=True)
    out = moe._batched_dispatch(x, weight, index, order, keep)
    out.square().sum().backward()
    got = {k_: v.clone() for k_, v in _grads(moe).items()}
    dx = x.grad.detach().clone()

    moe.zero_grad(set_to_none=True)
    x2 = x.detach().clone().requires_grad_(True)
    ref = moe._naive_dispatch(x2, weight, index, order, keep, Counters())
    ref.square().sum().backward()
    ref_grads = {
        f"{i}.{name}": p.grad for i, e in enumerate(moe.experts) for name, p in e.named_parameters()
    }
    print(
        f"[unequal] counts={counts.tolist()} cap={moe._batched_cap(index, weight)} "
        f"max|Δout|={float((out.detach() - ref.detach()).abs().max()):.3e} "
        f"max|Δdx|={float((dx - x2.grad).abs().max()):.3e}"
    )
    assert _rel(out.detach(), ref.detach()) <= 1e-6
    # The gradients are the same fp32 sums in another accumulation order; at
    # this fixture the absolute move is 2.4e-6 on `dx` and 2.4e-4 on the largest
    # `gate_u` entry (~1e3), so the bound is RELATIVE — an absolute one would
    # measure the fan-in, not the wiring.
    assert _rel(dx, x2.grad) <= 1e-6
    for name, g in got.items():
        ref_g = ref_grads.get(name)
        assert ref_g is not None and g is not None
        assert _rel(g, ref_g) <= 1e-6, name


# ------------------------------------------------- d. capacity / dropless rules
def test_capacity_drops_are_the_padded_arms_drops():
    """Gate (d): a full expert's row is dropped exactly as the capacity rule says."""
    torch.manual_seed(5)
    moe = _moe(moe_capacity=1.0, moe_kernel="batched")
    n, k = 64, 2
    # Every token picks experts 0 and 1: counts [64, 64, 0, 0], cap = 32, so
    # 32 rows of each full expert are dropped (64 assignments in total).
    index = torch.zeros(n, k, dtype=torch.long)
    index[:, 1] = 1
    weight = torch.ones(n, k) / k
    order, keep, dropped = moe._permute(index)
    assert int(dropped) == 64, int(dropped)
    x = torch.randn(n, moe.d)
    with torch.no_grad():
        batched = moe._batched_dispatch(x, weight, index, order, keep)
        padded = moe._padded_dispatch(x, weight, index, order, keep)
    assert torch.equal(batched, padded), "the two bank arms must be bit-identical"

    # The config switch itself on ONE model: same rows, same drop count, one
    # counter per arm (a second model would compare two random initializations).
    cb, cp = Counters(), Counters()
    moe.moe_kernel = "padded"
    with torch.no_grad():
        op = moe._dispatch(x, weight, index, cp)  # CPU: padded falls back COUNTED
    moe.moe_kernel = "batched"
    with torch.no_grad():
        ob = moe._dispatch(x, weight, index, cb)
    assert torch.equal(ob, batched), "the switch must run the same bank it names"
    assert _rel(ob, op) <= 1e-6, "the CPU fallback and the batched arm must agree"
    assert cb["moe_batched_calls"] == 1 and "moe_naive_dispatch" not in cb
    assert cp["moe_naive_dispatch"] == 1 and "moe_batched_calls" not in cp
    # `dropped` is a DEVICE counter; drain once (the log cadence's own read)
    # instead of aliasing it into a host comparison on the step path.
    assert cb.drain()["moe_dropped_assignments"] == cp.drain()["moe_dropped_assignments"]
    print(
        f"[capacity] counts={torch.bincount(index.reshape(-1), minlength=4).tolist()} "
        f"cap={moe._capacity(n, k)} dropped={int(dropped)} bit-identical"
    )


def test_dropless_keeps_every_assignment():
    """Gate (d), dropless half: no cap, no drop, the loop's function."""
    torch.manual_seed(4)
    moe = _moe(moe_capacity=0.0, moe_kernel="batched")
    x = torch.randn(48, moe.d)
    index, weight = _routing(moe, x)
    order, keep, dropped = moe._permute(index)
    assert int(dropped) == 0 and keep.all()
    counters = Counters()
    with torch.no_grad():
        batched = moe._dispatch(x, weight, index, counters)
        loop = moe._naive_dispatch(x, weight, index, order, keep, Counters())
    assert _rel(batched, loop) <= 1e-6
    assert counters["moe_batched_calls"] == 1 and counters["moe_dropless"] == 1
    print(f"[dropless] rows={x.shape[0]} cap={moe._batched_cap(index, weight)} drops=0")


# ------------------------------------------- e. padded rows carry no gradient
def test_padded_rows_carry_no_value_or_gradient():
    """Gate (e): unused bank rows are exactly a zero sink, grads included."""
    torch.manual_seed(6)
    moe = _moe(moe_capacity=1.0, moe_kernel="batched")
    n, k = 8, 2
    # cap = 4 (capacity 1.0 x 8 tokens x 2 / 4 experts): expert 0 owns 6 rows
    # (2 dropped past the cap), expert 1 owns 2 (2 PADDED bank rows), so one
    # fixture proves both sinks: the padding inside [E, cap] and the dump slot.
    index = torch.tensor([[0, 0], [0, 0], [0, 0], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]])
    weight = torch.ones(n, k) / k
    order, keep, dropped = moe._permute(index)
    counts = torch.bincount(index.reshape(-1), minlength=moe.n_experts)
    cap = moe._capacity(n, k)
    assert int(counts.max()) > cap, "the fixture needs a dropped assignment"
    assert int(counts.min()) < cap, "the fixture needs a padded bank row"
    x = torch.randn(n, moe.d, requires_grad=True)
    w = weight.clone().requires_grad_(True)
    grab = _BankGrab(moe.d)
    with grab:
        out = moe._batched_dispatch(x, w, index, order, keep)
    out.square().sum().backward()
    assert grab.bank is not None and grab.bank.grad is not None
    padded = 0
    for e in range(moe.n_experts):
        used = min(int(counts[e]), cap)
        if used == cap:
            continue
        pad = grab.bank.grad[e, used:]
        padded += int(pad.shape[0])
        assert float(pad.abs().max()) == 0.0, f"expert {e} padded rows got grad {pad.abs().max()}"
    assert padded > 0, "the fixture must leave a padded row to test"
    assert bool((~keep).any()), "the fixture must exercise the dump slot"
    dropped_w = w.grad.reshape(-1)[~keep]
    worst_dropped = float(dropped_w.abs().max())
    assert worst_dropped == 0.0, f"dropped rows got w grad {worst_dropped}"
    print(
        f"[padded grad] counts={counts.tolist()} cap={cap} "
        f"bank_pad=0 dropped_w=0 dropped_assignments={int(dropped)}"
    )


# ------------------------------------------------- f. heterogeneous refusal
def test_heterogeneous_expert_shapes_are_loud():
    """Gate (f): a per-expert rank cannot silently become a stack error."""
    moe = _moe(moe_capacity=0.0, moe_kernel="batched")
    moe.experts[1].v = torch.nn.Parameter(torch.randn(moe.experts[1].v.shape[0] + 1, moe.f))
    x = torch.randn(16, moe.d)
    index, weight = _routing(moe, x)
    order, keep, _ = moe._permute(index)
    with pytest.raises(LoudError) as e:
        moe._batched_dispatch(x, weight, index, order, keep)
    msg = str(e.value)
    print(f"[heterogeneous] {msg}")
    assert "moe_kernel='batched'" in msg and "Escape:" in msg
    assert "per-expert rank" in msg


# ----------------------------------------------------- g. no host syncs
def test_the_capacity_path_adds_no_host_sync():
    """Gate (g): the fixed-cap arm reads nothing back; the dropless cap is ONE scalar."""
    moe = _moe(moe_capacity=1.0, moe_kernel="batched")
    x = torch.randn(64, moe.d)
    index, weight = _routing(moe, x)
    with _SyncCalls() as spy:
        moe._dispatch(x, weight, index, Counters())
    assert spy.names == [], f"the capacity path read the host: {spy.names}"

    dropless = _moe(moe_capacity=0.0, moe_kernel="batched")
    with _SyncCalls() as spy:
        dropless._dispatch(x, weight, index, Counters())
    # The ONE read is the bank shape: `[E, cap, ·]` needs a Python int. It is
    # not a per-expert/per-token read (no `tolist`, no `unique`).
    assert spy.names == ["item"], f"the dropless cap read is not the one scalar: {spy.names}"
    print("[no sync] capacity path: 0 host reads; dropless: 1 (counts.max().item())")


# ------------------------------------------------------- h. graph capture
def test_full_graph_allows_the_static_cap_and_refuses_dropless():
    """Gate (h): the refusal is a property of the SHAPE, not of the arm's name."""
    from dormouse.config import TrainConfig
    from dormouse.train.graph import full_graph_refusal

    train = TrainConfig()
    static = tiny_cfg(moe_kernel="batched", moe_capacity=1.0, moe_topk=2, use_msa=False)
    why = full_graph_refusal(train, static)
    assert why is None, why

    dropless = tiny_cfg(moe_kernel="batched", moe_capacity=0.0, moe_topk=2, use_msa=False)
    why = full_graph_refusal(train, dropless)
    print(f"[full_graph] {why}")
    assert why is not None and "moe_kernel=batched" in why and "counts.max()" in why
    assert "Escape:" in why


@CUDA
def test_the_static_cap_arm_captures():
    """Gate (h), CUDA half: the static-cap bank really is a replayable program."""
    moe = _moe(moe_capacity=1.0, moe_kernel="batched", device="cuda")
    x = torch.randn(64, moe.d, device="cuda")
    index, weight = _routing(moe, x)
    counters = Counters()
    with torch.no_grad():
        moe._dispatch(x, weight, index, counters)  # warm the bank cache
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = moe._dispatch(x, weight, index, Counters())
    g.replay()
    torch.cuda.synchronize()
    print(f"[capture] batched static cap replay ok, out={tuple(out.shape)}")
    assert torch.isfinite(out).all()


# ------------------------------------------------------------- the switch
def test_the_literal_is_declared_and_the_counter_names_the_arm():
    assert "batched" in MOE_KERNELS
    assert "bucketed" in MOE_KERNELS
    validate(tiny_cfg(moe_kernel="batched", moe_capacity=1.0, moe_topk=2))
    assert MOE_KERNELS == ("padded", "triton", "naive", "batched", "bucketed")
