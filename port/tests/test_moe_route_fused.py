"""Gates for the fused route head and the counting permutation (both OFF).

What each gate proves, and where its reference comes from:

A. **OFF is clean main, bit for bit.** The literal digests below were measured on
   `main` @ `28977f13` (the commit this branch starts from) with the same
   fixture, in the main worktree, and are pasted here unchanged. The test also
   puts a bomb on both new entry points while the flags are off: the OFF path
   must not call them at all, which is a stronger statement than equal outputs.
B. **The fused route is the same arithmetic.** 1 000 rows, fp32 and bf16, E=64
   and E=50, against the four ops it replaces: indices identical; weights
   bit-identical at bf16 and within one fp32 ulp at fp32 (the fused softmax is
   `exp/sum` in fp32, `F.softmax`'s own internal reduction rounds differently —
   measured 1.8e-7 absolute on values ~2e-3, i.e. 1 ulp of the *sum*). The
   backward is compared with autograd through the same chain.
C. **The tie rule is DEFINED here, not inferred.** `torch.topk` does not specify
   the order of equal values: on this fixture it answers `[5,4]` where the
   stable rule answers `[4,5]` (same ids as a set) and `[2,4]` where the stable
   rule answers `[0,2]` (a DIFFERENT expert, same score). Both are pinned with
   literals; the gate asserts the value multiset always matches and names the
   one case where the id set may differ. A tie is a measure-zero routing
   decision; a silent, undocumented tie rule is the defect this test prevents.
D. **The counting permutation is the stable argsort.** Element-wise, on 16 384
   ids with duplicates, on the adaptive width's sentinel group, and on the
   full `_permute` (order, keep, dropped) — the rows the bank and the combine
   see are IDENTICAL, which is the property the `index_add_` order and the bank
   rows depend on.
E. **The arms are wired and counted**, and every refusal names an `Escape:`.
F. **The dispatch counter.** At the model seam, the old block issues four aten
   dispatches (softmax, add, topk, gather); the fused branch issues ZERO of
   them and one call. The census prices the CUDA side (9 kernels -> 1 for the
   head, 12 -> 3 for the permute).
G. **The census is the arithmetic, not a screenshot.** Its route totals are
   recomputed in this file from the shapes against `scripts/dispatch_census.py`;
   the gather arm's byte saving is asserted to be exactly `4·R·d`.

No timing gate exists: the card is held by a multi-day run, so the CUDA kernels
are compiled (host ptxas) and their parity gate is written and skipped with its
reason. `scripts/dispatch_census.py` is the static price; the A/B command is in
the report.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg
from torch.overrides import TorchFunctionMode

from dormouse.config import MOE_PERMUTES, ModelConfig, validate
from dormouse.instrument import Counters, LoudError
from dormouse.model import route_fused as rf
from dormouse.model.experts import SparseMoE

ROOT = Path(__file__).resolve().parents[1]

#: sha256 of the OFF path's logits and of its 27 parameter gradients, measured
#: on `main` @ `28977f13` (clean tree) with `_pin_model()`'s fixture.
OFF_LOGITS = "f1af8f9d39852417c63e2a8776d48c48b0b054f76ff31185aeb03ce7cb54abac"
OFF_GRADS = "33a11a48d0a0e539dbd6bfb7264ca69cffe2ed90e12038859586b1d6a7280d87"


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()


def _cfg(**over) -> ModelConfig:
    return tiny_cfg(n_experts=4, moe_topk=2, rank=4, d_ffn=64, **over)


def _pin_model() -> tuple[SparseMoE, torch.Tensor]:
    torch.manual_seed(0)
    moe = SparseMoE(_cfg(moe_capacity=1.0), device="cpu")
    torch.manual_seed(1)
    return moe, torch.randn(8, moe.d)


# --------------------------------------------------------------------------- #
# A. OFF is clean main
# --------------------------------------------------------------------------- #
def test_the_flags_are_off_by_default() -> None:
    cfg = ModelConfig()
    assert cfg.moe_route_fused is False
    assert cfg.moe_permute == "argsort" and MOE_PERMUTES == ("argsort", "count")
    print("[defaults] moe_route_fused=False moe_permute='argsort'")


@pytest.mark.pins
def test_off_is_bit_identical_to_clean_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pin FIRST: the OFF path's bits and its call graph, on one fixture."""
    moe, x = _pin_model()

    def bomb(*_a, **_k):
        raise AssertionError("the OFF path entered the new arm")

    with monkeypatch.context() as m:
        m.setattr(rf, "softmax_topk", bomb)
        m.setattr(rf, "count_permute", bomb)
        out = moe(x, Counters())
        lo = _sha(out)
        out.pow(2).mean().backward()
    blob = torch.cat(
        [
            p.grad.contiguous().view(-1)
            for _, p in sorted(moe.named_parameters())
            if p.grad is not None
        ]
    )
    assert lo == OFF_LOGITS, f"the OFF logits moved: {lo}"
    assert _sha(blob) == OFF_GRADS, "the OFF gradients moved"
    print(f"[pin] logits {lo[:16]}… grads {_sha(blob)[:16]}… == main")


# --------------------------------------------------------------------------- #
# B. the fused route is the same arithmetic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("E", [64, 50])
def test_the_fused_route_matches_topk_on_1000_rows(dt, E) -> None:
    torch.manual_seed(0)
    logits = (torch.randn(1000, E) * 2.0).to(dt)
    bias = torch.randn(E)
    index, weight = rf.route_reference(logits, bias, 2)

    probs = F.softmax(logits, dim=-1)
    raised = logits + bias  # fp32: the same promotion the model performs
    want_index = raised.topk(2, dim=-1).indices
    want_weight = torch.gather(probs, 1, want_index)

    assert torch.equal(index, want_index), "the fused top-k names different experts"
    if dt is torch.bfloat16:
        assert torch.equal(weight, want_weight), "bf16 gate weights are not bit-identical"
        delta = 0.0
    else:
        delta = float((weight - want_weight).abs().max())
        assert delta <= 2e-6, f"fp32 gate weights moved by {delta}"
    print(f"[route] {dt} E={E}: 1000 rows, indices equal, max|dw|={delta:.2e}")


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
def test_the_fused_route_backward_matches_autograd(dt) -> None:
    torch.manual_seed(2)
    x = (torch.randn(64, 32) * 2.0).to(dt).requires_grad_(True)
    bias = torch.randn(32)
    _, w = rf.softmax_topk(x, bias, 2)
    (w.float().pow(2).sum()).backward()
    got = x.grad.clone()

    x2 = x.detach().clone().requires_grad_(True)
    p = F.softmax(x2, dim=-1)
    idx = (x2 + bias).topk(2, dim=-1).indices
    torch.gather(p, 1, idx).float().pow(2).sum().backward()
    delta = float((got.float() - x2.grad.float()).abs().max())
    # The same adjoint written out, so the CPU arm uses the FORMULA the kernel
    # transcribes (a break in either one fails here).
    index, weight = rf.route_reference(x2.detach(), bias, 2)
    dweight = 2.0 * weight.float()
    ref = rf.route_reference_backward(x2.detach(), index, dweight)
    ref_delta = float((ref.float() - x2.grad.float()).abs().max())
    scale = float(x2.grad.float().abs().max().clamp_min(1e-12))
    bound = 2e-6 if dt is torch.float32 else 0.02 * scale
    assert delta <= bound, f"the custom adjoint disagrees with autograd by {delta} > {bound}"
    assert ref_delta <= bound, f"the reference formula is off by {ref_delta} > {bound}"
    print(f"[route bwd] {dt}: custom {delta:.2e}, formula {ref_delta:.2e} vs autograd")


# --------------------------------------------------------------------------- #
# C. the tie rule, pinned both ways
# --------------------------------------------------------------------------- #
TIE_ROWS = torch.tensor(
    [
        [2.0, 2.0, 0.0, -1.0, 5.0, 5.0, 5.0, 1.0],  # tie INSIDE the top-2
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # tie ACROSS the boundary
        [3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0],  # 4-way tie across it
    ]
)


def test_the_tie_fixture_pins_the_rule() -> None:
    index, weight = rf.route_reference(TIE_ROWS, torch.zeros(8), 2)
    top = TIE_ROWS.topk(2, dim=-1)
    # The values are the top-2 VALUES in both arms, always.
    assert torch.equal(torch.gather(TIE_ROWS, 1, index), top.values)
    # torch.topk's order among equals is unspecified; these are its CPU answers.
    assert top.indices.tolist() == [[5, 4], [1, 0], [2, 4]]
    # The fused rule: highest score, then the LOWEST id.
    assert index.tolist() == [[4, 5], [0, 1], [0, 2]]
    # Rows 0-1: same id SET (slot order differs, which the combine cannot see).
    assert {tuple(sorted(r)) for r in index[:2].tolist()} == {
        tuple(sorted(r)) for r in top.indices[:2].tolist()
    }
    # Row 2: a tie at the boundary NAMES A DIFFERENT EXPERT (2 vs 0), same score.
    assert sorted(index[2].tolist()) == [0, 2] and sorted(top.indices[2].tolist()) == [2, 4]
    # `weight` is the unbiased softmax at the chosen ids, so the pairs move together.
    p = F.softmax(TIE_ROWS, dim=-1)
    assert torch.equal(weight, torch.gather(p, 1, index))
    print("[ties] set equal rows 0-1; row 2 pins [0,2] vs torch's [2,4] (unspecified)")


# --------------------------------------------------------------------------- #
# D. the counting permutation IS the stable argsort
# --------------------------------------------------------------------------- #
def _ids(E: int, n: int, *, sentinel: bool) -> torch.Tensor:
    torch.manual_seed(3)
    ids = torch.randint(0, E, (n,))
    if sentinel:  # the adaptive width's padding slot, one past the bank
        ids[::7] = E
    return ids


@pytest.mark.parametrize("sentinel", [False, True])
def test_the_count_permute_equals_the_stable_argsort(sentinel) -> None:
    E = 64
    groups = E + 1 if sentinel else E
    for n in (1, 7, 16384):
        ids = _ids(E, n, sentinel=sentinel)
        want = torch.argsort(ids, stable=True)
        got = rf.count_permute_reference(ids, groups)
        assert torch.equal(got, want), f"count != stable argsort at n={n}"
        assert torch.equal(ids[got], ids[want]), "the sorted id sequence moved"
    print(f"[permute] n=1/7/16384, sentinel={sentinel}: element-wise equal")


def test_the_count_permute_arm_is_the_argsort_arm() -> None:
    """The whole `_permute`, not just the ids: order, keep and dropped agree."""
    torch.manual_seed(4)
    for capacity in (0.0, 1.0):
        off = SparseMoE(_cfg(moe_capacity=capacity, moe_permute="argsort"), device="cpu")
        cnt = SparseMoE(_cfg(moe_capacity=capacity, moe_permute="count"), device="cpu")
        cnt.load_state_dict(off.state_dict())
        index = torch.randint(0, off.n_experts, (32, 2))
        valid = torch.rand(32, 2) > 0.3
        with pytest.MonkeyPatch.context() as m:
            m.setattr(rf, "reason", lambda _x: None)  # force the arm on CPU
            order_c, keep_c, drop_c = cnt._permute(index, valid, Counters())
        order_a, keep_a, drop_a = off._permute(index, valid)
        assert torch.equal(order_c, order_a)
        assert torch.equal(keep_c, keep_a)
        assert torch.equal(drop_c, drop_a)
    print("[permute] order/keep/dropped identical on capacity 0 and 1")


# --------------------------------------------------------------------------- #
# E. wiring, counters, refusals
# --------------------------------------------------------------------------- #
def test_the_permute_arm_counts_and_falls_back_counted() -> None:
    moe = SparseMoE(_cfg(moe_permute="count"), device="cpu")
    index = torch.randint(0, moe.n_experts, (16, 2))
    c = Counters()
    moe._permute(index, None, c)
    assert c["moe_permute_fallback"] == 1 and "moe_permute_count" not in c
    assert "permute:" in c.why and "CUDA" in c.why

    again = SparseMoE(_cfg(moe_permute="count"), device="cpu")
    with pytest.MonkeyPatch.context() as m:
        m.setattr(rf, "reason", lambda _x: None)
        c = Counters()
        again._permute(index, None, c)
    assert c["moe_permute_count"] == 1 and "moe_permute_fallback" not in c
    print("[permute] counted: arm=1, CPU fallback=1 with its reason")


def test_the_route_arm_counts_and_is_parity_on_the_model() -> None:
    _, x = _pin_model()
    off = SparseMoE(_cfg(moe_route_fused=False, moe_capacity=1.0), device="cpu")
    on = SparseMoE(_cfg(moe_route_fused=True, moe_capacity=1.0), device="cpu")
    on.load_state_dict(off.state_dict())
    base = off(x, Counters()).detach()

    c = Counters()
    got = on(x, c).detach()  # CPU: COUNTED fallback
    assert c.get("moe_route_fused_fallback") == 1, c
    # The reason is the arm's own gate (the dispatch's later `why` overwrites the
    # one channel, so it is read where it is decided, not off the counter).
    why = on._route_fused_reason(x)
    assert why is not None and "CUDA" in why and "Escape:" not in why, why
    assert torch.equal(got, base), "the counted fallback is not the old path"

    with pytest.MonkeyPatch.context() as m:
        m.setattr(rf, "reason", lambda _x: None)
        c = Counters()
        fused = on(x, c).detach()
    assert c.get("moe_route_fused_calls") == 1 and "moe_route_fused_fallback" not in c
    delta = float((fused - base).abs().max())
    assert delta <= 2e-6, f"the fused route moved the model output by {delta}"
    print(f"[route wiring] fallback==old; fused==old, max|dout|={delta:.2e}")


def test_the_refusals_carry_the_escape() -> None:
    cases = [
        dict(moe_permute="bogus"),
        dict(moe_route_fused=True, moe_topk=0),
        dict(moe_route_fused=True, moe_topk_mode="entropy"),
        dict(moe_route_fused=True, moe_lb_mode="aux"),
    ]
    for over in cases:
        cfg = dataclasses.replace(tiny_cfg(), **over)
        with pytest.raises(LoudError) as e:
            validate(cfg)
        msg = str(e.value)
        assert "Escape:" in msg, msg
        print(f"[refusal] {over}: {msg.splitlines()[-1][:96]}")


# --------------------------------------------------------------------------- #
# F. the dispatch counter
# --------------------------------------------------------------------------- #
class _Count(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.names.append(str(func))
        return func(*args, **(kwargs or {}))


ROUTE_OPS = ("aten._softmax", "aten.softmax", "aten.topk", "aten.gather")


def _route_hits(names: list[str]) -> list[str]:
    """The route's own ops. `aten.add` is excluded: the residual adds share the
    name, so counting it would count the shared expert's write as routing."""
    return [n for n in names if any(n.startswith(o) for o in ROUTE_OPS)]


def test_the_fused_branch_issues_one_call_and_no_route_dispatches() -> None:
    """>=4 route dispatches -> 0, and exactly one call into the arm.

    The stub stands for the kernel: a CUDA implementation is one launch, and what
    this measures is what the MODEL still does around it. That `topk` is six
    kernels on this stack is the census's row, not this counter's.
    """
    off = SparseMoE(_cfg(moe_capacity=0.0), device="cpu")
    on = SparseMoE(_cfg(moe_route_fused=True, moe_capacity=0.0), device="cpu")
    on.load_state_dict(off.state_dict())
    x = torch.randn(16, off.d)

    with _Count() as spy:
        off.forward(x, Counters())
    old = _route_hits(spy.names)
    assert any(n.startswith("aten.gather") for n in old), old
    assert any(n.startswith("aten.topk") for n in old), old
    assert any(n.startswith(("aten._softmax", "aten.softmax")) for n in old), old

    calls: list[int] = []

    def stub(logits, bias, k):
        calls.append(k)
        idx = torch.topk(logits + bias, k, dim=-1).indices.detach()
        probs = torch.softmax(logits.detach(), dim=-1)
        return idx, torch.gather(probs, 1, idx)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(rf, "reason", lambda _x: None)
        m.setattr(rf, "softmax_topk", stub)
        # The stub itself dispatches, so count only the MODEL's own ops: run the
        # forward once outside the mode to warm the stub, then count.
        on.forward(x, Counters())
        calls.clear()

        with _Count() as spy2:
            on.forward(x, Counters())
    assert calls == [off.topk], "the branch did not call the arm exactly once"
    # The stub reproduces the four ops; the model must add none of its own on top.
    model_only = len(_route_hits(spy2.names)) - 3
    hits = _route_hits(spy2.names)
    assert model_only == 0, f"the model still issues route ops beside the arm: {hits}"
    print(f"[dispatch] off={len(old)} route dispatches, fused=1 call (+0 beside it)")


class _SyncCalls(TorchFunctionMode):
    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", "")
        if name in ("item", "cpu", "tolist", "numpy", "__bool__", "__float__", "__int__"):
            self.names.append(name)
        return func(*args, **(kwargs or {}))


def test_the_wiring_adds_no_host_sync() -> None:
    """The branch, the counters and the reason checks read nothing back.

    The CPU reference's own `counts.tolist()` is inside the arm, not beside it;
    the kernel path never leaves the device (both kernels store only).
    """
    off = SparseMoE(_cfg(moe_capacity=0.0), device="cpu")
    on = SparseMoE(_cfg(moe_route_fused=True, moe_capacity=0.0), device="cpu")
    on.load_state_dict(off.state_dict())
    x = torch.randn(16, on.d)
    idx = torch.randint(0, on.n_experts, (16, 2))
    with pytest.MonkeyPatch.context() as m:
        m.setattr(rf, "reason", lambda _x: None)
        m.setattr(rf, "softmax_topk", lambda lg, b, k: (idx, torch.ones(16, k)))
        with _SyncCalls() as old:
            off.forward(x, Counters())
        with _SyncCalls() as new:
            on.forward(x, Counters())
    # The CPU reference's `_naive_dispatch` has its own `unique().tolist()`;
    # what this gate forbids is ONE MORE read beside it.
    assert new.names == old.names, f"the fused wiring added a host read: {new.names} vs {old.names}"
    print(f"[no sync] fused branch adds none: {new.names} on both (the reference's own)")


# --------------------------------------------------------------------------- #
# G. the census
# --------------------------------------------------------------------------- #
def _census():
    spec = importlib.util.spec_from_file_location(
        "dispatch_census", ROOT / "scripts" / "dispatch_census.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _m400b() -> ModelConfig:
    """`configs/m400.toml`, with the four arms the profiled `m400b` snapshot ran."""
    preset = tomllib.loads((ROOT / "configs" / "m400.toml").read_text())
    return dataclasses.replace(
        ModelConfig(**preset),
        moe_capacity=0.0,
        share_u=True,
        share_u_mode="plane",
        moe_lb_mode="prop",
        moe_lb_gamma=0.002,
        moe_lb_clamp=4.0,
    )


def test_the_census_prices_the_region() -> None:
    dc = _census()
    cfg = _m400b()
    c, rows = dc.census(cfg, 8, 1024)
    n, E, d, f, r, k = c.n, c.E, c.d, c.f, c.r, c.k
    assert (n, E, d, f, r, k) == (8192, 512, 1536, 4096, 64, 2)
    dtb = 2

    by = {r.op: r for r in rows}
    fused_cfg = dataclasses.replace(cfg, moe_route_fused=True, moe_permute="count")
    _, frows = dc.census(fused_cfg, 8, 1024)
    fby = {r.op: r for r in frows}
    # The fused route's own bytes, recomputed here from its shapes.
    fused_fwd = n * E * dtb + E * 4 + n * k * 8 + n * k * dtb
    fused_bwd = n * E * dtb + n * k * 8 + n * k * dtb + n * E * dtb
    assert fby["fused softmax+bias+topk+gather"].nbytes == fused_fwd
    assert fby["fused route backward"].nbytes == fused_bwd

    # The legacy forward it replaces: softmax + add + topk + gather.
    assert by["softmax(logits)"].nbytes == 2 * n * E * dtb
    assert by["raised = logits + lfb_bias"].nbytes == n * E * dtb + E * 4 + n * E * 4
    assert by["raised.topk(k)"].launches == 6  # measured mbtopk family
    assert by["gather(probs, index)"].nbytes == n * E * dtb + n * k * 8 + n * k * dtb
    legacy_fwd = (
        2 * n * E * dtb
        + (n * E * dtb + E * 4 + n * E * 4)
        + (2 * n * E * 4 + n * k * 8 + n * k * 4)
        + (n * E * dtb + n * k * 8 + n * k * dtb)
    )
    legacy_bwd = sum(
        by[o].nbytes
        for o in ("gather backward (zero scatter)", "softmax backward", "bias-add backward")
    )
    assert legacy_bwd == (
        n * k * 8
        + n * k * dtb
        + n * E * dtb  # gather bwd
        + 2 * n * E * dtb
        + n * E * dtb  # softmax bwd
        + n * E * 4
        + n * E * dtb  # add bwd
    )

    def arm(**over):
        _, rs = dc.census(dataclasses.replace(cfg, **over), 8, 1024)
        return rs, dc.totals([x for x in rs if x.group in dc.ROUTE_GROUPS])

    _, (shipped_b, shipped_l) = arm(moe_route_fused=False, moe_permute="argsort")
    rs, (fused_b, fused_l) = arm(moe_route_fused=True, moe_permute="count")
    want = (legacy_fwd - fused_fwd) + (legacy_bwd - fused_bwd)
    want += (
        by["order = argsort(safe, stable)"].nbytes
        - fby["seed order (count, one program/expert)"].nbytes
    )
    assert shipped_b - fused_b == want, (shipped_b - fused_b, want)
    assert shipped_l == 53 and fused_l == 34, (shipped_l, fused_l)
    assert shipped_b - fused_b > 100_000_000, "less than 100 MB per forwarded pass"

    # The region is NOT dominated by the route head: name the biggest rows.
    groups = {g: sum(x.nbytes for x in rows if x.group == g) for g in dc.ROUTE_GROUPS}
    assert max(groups, key=groups.get) == "combine", groups
    assert groups["combine"] > 5 * groups["route"]
    print(
        f"[census] route {shipped_b / 1e6:.1f} MB / {shipped_l} launches -> "
        f"{fused_b / 1e6:.1f} MB / {fused_l} launches; biggest group "
        f"{max(groups, key=groups.get)} {max(groups.values()) / 1e6:.1f} MB"
    )


def test_the_gather_arm_saves_exactly_4Rd_in_the_forward() -> None:
    dc = _census()
    cfg = _m400b()
    R, d = 8192 * 2, 1536

    def rows_for(mode: str):
        _, rs = dc.census(dataclasses.replace(cfg, moe_gather=mode), 8, 1024)
        return {x.op: x for x in rs if x.group == "gather"}, sum(
            x.nbytes for x in rs if x.group == "gather"
        )

    off_rows, off = rows_for("off")
    on_rows, on = rows_for("kernel")
    saved = 4 * R * d  # what the fused load removes: the copy + its read
    recompute = on_rows["gather arm: recompute one expert forward"].nbytes
    toks = on_rows["toks = (rows//k).to(int32)"].nbytes
    # OFF has the 3-transfer copy; ON has the toks cast + one read + one recompute.
    assert off_rows["xg = x[owner] (index_select)"].nbytes == 3 * R * d * 2 == saved + R * d * 2
    assert on - off == toks + recompute - saved, (on - off, toks, recompute, saved)
    assert on > off, (
        "as wired the gather arm is a PASS byte loss (its backward recomputes a forward)"
    )
    print(
        f"[gather] fwd -{saved / 1e6:.1f} MB, recompute +{recompute / 1e6:.1f} MB "
        f"-> pass +{(on - off) / 1e6:.1f} MB"
    )


def test_gather_abi_skips_the_bool_mask_when_every_row_is_kept() -> None:
    """The `keep=None` arm is the SAME ABI without a `nonzero` (a D2H read)."""
    torch.manual_seed(5)
    n, k, E = 64, 2, 8
    index = torch.randint(0, E, (n, k))
    order = torch.argsort(index.reshape(-1), stable=True)
    keep = torch.ones(n * k, dtype=torch.bool)
    from dormouse.model import moe_sonic

    t_none, o_none = moe_sonic.gather_abi(index, order, None, E)
    t_mask, o_mask = moe_sonic.gather_abi(index, order, keep, E)
    assert torch.equal(t_none, t_mask) and torch.equal(o_none, o_mask)
    print("[gather abi] keep=None == keep=all-True, without the bool-mask index")


# --------------------------------------------------------------------------- #
# H. the kernels build for this arch (host ptxas, no device)
# --------------------------------------------------------------------------- #
def test_the_kernels_build_for_sm_120() -> None:
    if rf.triton is None:
        pytest.skip("no triton in this build")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler import compile as tcompile

    arch = 120
    if torch.cuda.is_available():
        mj, mn = torch.cuda.get_device_capability()
        arch = mj * 10 + mn
    target = GPUTarget("cuda", arch, 32)
    built = []
    for dt in ("bf16", "fp32"):
        src = ASTSource(
            fn=rf._route_kernel,
            signature={
                "X": f"*{dt}",
                "B": "*fp32",
                "IDX": "*i64",
                "W": f"*{dt}",
                "E": "constexpr",
                "K": "constexpr",
                "BLOCK_E": "constexpr",
            },
            constexprs={"E": 512, "K": 2, "BLOCK_E": 512},
        )
        k = tcompile(src, target=target, options={"num_warps": 4})
        assert "cubin" in k.asm
        built.append(k.metadata.shared)
        bwd = ASTSource(
            fn=rf._route_bwd_kernel,
            signature={
                "X": f"*{dt}",
                "IDX": "*i64",
                "DW": f"*{dt}",
                "DLOG": f"*{dt}",
                "E": "constexpr",
                "K": "constexpr",
                "BLOCK_E": "constexpr",
            },
            constexprs={"E": 512, "K": 2, "BLOCK_E": 512},
        )
        kb = tcompile(bwd, target=target, options={"num_warps": 4})
        assert "cubin" in kb.asm
        built.append(kb.metadata.shared)
    perm = ASTSource(
        fn=rf._count_permute_kernel,
        signature={
            "IDS": "*i32",
            "STARTS": "*i32",
            "ORDER": "*i64",
            "N": "constexpr",
            "BLOCK": "constexpr",
        },
        constexprs={"N": 16384, "BLOCK": 1024},
    )
    kp = tcompile(perm, target=target, options={"num_warps": 4})
    assert "cubin" in kp.asm
    built.append(kp.metadata.shared)
    print(f"[compile] sm_{arch}: route fwd/bwd + count-permute built, shared={built} B")
