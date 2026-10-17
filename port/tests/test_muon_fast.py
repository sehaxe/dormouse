"""The batched Muon step's fast path: the same updates, without the copies.

The 2026-10-16 speed change makes the stacked momentum buffer the STATE
(`Muon._build_groups`) instead of a per-step `torch.stack` + `_foreach_copy_`,
and runs the tail of the step (`colrow_normalize`, the `-lr*scale` write) in
place. None of that is allowed to move a number, so the gate here is
EQUIVALENCE, and it is stated in its strongest form: **bit-exact** against the
pre-change implementation, transcribed below as `old_batched_step` (the exact
code that shipped in `d55fd1d`/`1fae2f7`), on the same model, the same grads
and the same state.

Three more things are gated, because they are what the change is:

* **the state is the stack** — after a step, `state[p]["momentum"]` is a VIEW of
  the group buffer, and the step issues ZERO `aten.stack` / `aten.copy_`
  dispatches (those copies are the whole point);
* **the checkpoint contract** — `state_dict` still carries one momentum tensor
  per parameter (the historical format), an old per-param checkpoint still
  loads, and a resumed optimizer's NEXT step is identical to the
  uninterrupted one's;
* **the singleton fallback** — a shape with one member keeps the per-param
  definition, because that is the branch the 3-D ColRow divergence would move.

The last test is a characterisation, not a gate: building this equivalence
check found that the batched step's `colrow_normalize` was ALREADY not the
per-param ColRow (a 3-D branch that normalises each slice by its Frobenius norm
twice, so the batched update is `1/sqrt(d0)` of the definition). It is pinned
here so the fix cannot rot, and it is NOT fixed by a speed commit — it moves the
trainer's update scale for every shape group with >= 2 members and so owes a
3-seed A/B (AGENTS §1.2).
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import tiny_cfg, tiny_run
from torch import nn

from dormouse.config import load_run
from dormouse.model import DormouseModel
from dormouse.train import optim, train
from dormouse.train.optim import (
    MixOptimizer,
    Muon,
    colrow_normalize,
    colrow_normalize_,
    zeropower_via_ns,
)

# --- the reference: the step this change replaces, verbatim ------------------


def old_batched_step(muon: Muon, chunk_elems: int | None = None) -> int:
    """`Muon.batched_step_from_model` as it shipped before 2026-10-16.

    Transcribed (not imported) on purpose: the point of the gate is a
    comparison of two PROGRAMS, so the reference must not be reachable from the
    code under test. Same shape grouping, same per-param fallback for a shape
    with one member, and the same chunk cap — `chunk_elems` overrides it so a
    test can put both implementations on the same boundary.
    """
    by_shape: dict[tuple[int, int], list] = {}
    for group in muon.param_groups:
        lr, mu, ns = group["lr"], group["momentum"], group["ns_steps"]
        for p in group["params"]:
            if p.grad is None or p.grad.ndim != 2:
                continue
            by_shape.setdefault((p.shape[0], p.shape[1]), []).append(p)
    n_batches = 0
    for shape, params in by_shape.items():
        for p in params:
            state = muon.state[p]
            if "momentum" not in state:
                state["momentum"] = torch.zeros_like(p)
        if len(params) < 2:
            for p in params:
                state = muon.state[p]
                buf = state["momentum"]
                buf.mul_(mu).add_(p.grad, alpha=1.0 - mu)
                update = colrow_normalize(
                    zeropower_via_ns(buf, ns, muon.ns_kind, muon.gram_restarts),
                    colrow_3d="frobenius",  # the historical arm, pinned
                )
                scale = max(1.0, shape[0] / shape[1]) ** 0.5
                with torch.no_grad():
                    p.add_(update.to(p.dtype), alpha=-lr * scale)
            continue
        n = len(params)
        # The historical cap is `(1 << 27) // (d0 * d1 * 4)` PARAMS, i.e.
        # `1 << 25` fp32 elements per stacked buffer — the same convention as
        # `optim.MUON_CHUNK_ELEMS`.
        cap = (1 << 25) if chunk_elems is None else chunk_elems
        n_chunk = max(1, min(n, cap // (shape[0] * shape[1])))
        for start in range(0, n, n_chunk):
            params_c = params[start : start + n_chunk]
            bufs = torch.stack([muon.state[p]["momentum"] for p in params_c])
            grads = torch.stack([p.grad for p in params_c])
            bufs.mul_(mu).add_(grads.detach(), alpha=1.0 - mu)
            update = colrow_normalize(
                zeropower_via_ns(bufs, ns, muon.ns_kind, muon.gram_restarts),
                colrow_3d="frobenius",  # the historical arm, pinned
            ).detach()
            scale = max(1.0, shape[0] / shape[1]) ** 0.5
            updates = update.mul(-lr * scale)
            with torch.no_grad():
                torch._foreach_add_([p.detach() for p in params_c], torch.unbind(updates))
                torch._foreach_copy_(
                    [muon.state[p]["momentum"] for p in params_c], torch.unbind(bufs)
                )
        n_batches += 1
    return n_batches


# --- fixtures ----------------------------------------------------------------


def shape_model(shapes: dict[tuple[int, int], int], seed: int = 0) -> nn.Module:
    """A container of parameters with the given shape groups — what Muon sees."""
    gen = torch.Generator().manual_seed(seed)
    mod = nn.Module()
    i = 0
    for shape, count in shapes.items():
        for _ in range(count):
            mod.register_parameter(f"p{i}", nn.Parameter(torch.randn(*shape, generator=gen) * 0.05))
            i += 1
    return mod


def _twin(shapes: dict[tuple[int, int], int], seed: int = 1) -> tuple[nn.Module, nn.Module]:
    """Two models with identical weights AND identical gradients."""
    a = shape_model(shapes)
    b = shape_model(shapes)
    b.load_state_dict(a.state_dict())
    gen = torch.Generator().manual_seed(seed)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        pa.grad = torch.randn(*pa.shape, generator=gen) * 0.1
        pb.grad = pa.grad.clone()
    return a, b


def _pair(
    shapes: dict[tuple[int, int], int], colrow_3d: str = "frobenius", **kw
) -> tuple[nn.Module, Muon, nn.Module, Muon]:
    a, b = _twin(shapes)
    kw = {"colrow_3d": colrow_3d, **kw}
    return a, Muon(a.parameters(), lr=1e-3, **kw), b, Muon(b.parameters(), lr=1e-3, **kw)


def _same_params(a: nn.Module, b: nn.Module) -> bool:
    return all(torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters(), strict=True))


def _same_momentum(oa: Muon, ob: Muon) -> bool:
    return all(
        torch.equal(oa.state[p]["momentum"], ob.state[q]["momentum"])
        for p, q in zip(oa.param_groups[0]["params"], ob.param_groups[0]["params"], strict=True)
    )


def _dispatches(fn) -> list[str]:
    seen: list[str] = []

    class _Count(torch.utils._python_dispatch.TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            seen.append(str(func))
            return func(*args, **(kwargs or {}))

    with _Count():
        fn()
    return seen


#: The b1 preset's shape groups, small enough to run on CPU in a test: the
#: rectangular pair (transposed NS), a square group (NO transpose, r = d0), the
#: wide-tall pair, and a couple of small groups. `[2048, 2048]` x 8 is the b1
#: group the bench covers on GPU; 258 G MACs of fp32 NS is not a CPU test.
B1_LIKE = {(64, 5461): 4, (2048, 64): 4, (256, 256): 3, (8, 128): 3, (64, 64): 3}


# --- the gate: the new step IS the old step ----------------------------------


@pytest.mark.parametrize(
    "shapes",
    [B1_LIKE, {(64, 64): 1}, {(64, 5461): 1, (2048, 64): 2}],
    ids=["b1-like", "singleton", "mixed"],
)
def test_the_fast_path_is_bit_exact_against_the_step_it_replaces(shapes):
    """Two steps, bit-exact on parameters AND momentum — the whole claim.

    `singleton` is the per-param fallback branch, `b1-like` mixes batched
    groups (transposed and not) with a singleton, so one comparison covers
    both.
    """
    a, oa, b, ob = _pair(shapes)
    for step in range(2):
        old_batched_step(oa)
        ob.batched_step_from_model(b)
        assert _same_params(a, b), f"step {step}: the fast path moved a weight"
        assert _same_momentum(oa, ob), f"step {step}: the fast path moved the momentum"
    assert float(next(a.parameters()).detach().abs().sum()) > 0.0  # the step did happen


def test_the_chunk_boundary_is_bit_exact(monkeypatch):
    """Many chunks (1-2 params each) must equal the unchunked step.

    Both implementations get the same tiny cap, and the cap is asserted to have
    BITTEN (more matmul dispatches) — otherwise the test would pass by running
    the ordinary path.
    """
    shapes = {(64, 64): 7, (8, 128): 5}
    a, oa, b, ob = _pair(shapes)
    cap = 1 << 12  # [64,64] = 4096 elements -> exactly one param per chunk
    monkeypatch.setattr(optim, "MUON_CHUNK_ELEMS", cap)
    chunked = _dispatches(lambda: ob.batched_step_from_model(b))
    monkeypatch.undo()
    old_batched_step(oa, chunk_elems=cap)
    assert _same_params(a, b) and _same_momentum(oa, ob)
    plain = _dispatches(lambda: old_batched_step(Muon(a.parameters(), lr=1e-3)))
    n_mm = lambda ops: sum(o.startswith(("aten.mm", "aten.bmm", "aten.matmul")) for o in ops)  # noqa: E731
    assert n_mm(chunked) > n_mm(plain), "the cap did not bite"


def test_the_momentum_update_is_the_stacked_form_bit_exactly():
    """`mom.mul_(mu)` + one fused `_foreach_add_` over the rows == stack+add.

    This is the piece that removes the per-step 1.33 GiB stack and the copy
    back, so it has to be exact rather than close.
    """
    torch.manual_seed(0)
    params = [torch.randn(64, 5461) for _ in range(4)]
    grads = [torch.randn(64, 5461) for _ in range(4)]
    mu = 0.95
    ref = torch.stack([p.clone() for p in params]).mul(mu).add(torch.stack(grads), alpha=1 - mu)

    mom = torch.stack([p.clone() for p in params])
    mom.mul_(mu)
    torch._foreach_add_(torch.unbind(mom, 0), grads, alpha=1.0 - mu)
    assert torch.equal(mom, ref)


def test_colrow_normalize_in_place_is_bit_identical():
    torch.manual_seed(0)
    x = torch.randn(3, 64, 5461)
    assert torch.equal(colrow_normalize_(x.clone()), colrow_normalize(x.clone()))


# --- the copies are gone (the reason the change exists) ----------------------


def test_the_state_is_the_stack_and_the_step_copies_nothing():
    """The gate on the WIN, not just on the math: no stack, no copy, and the
    per-parameter state is a view of the group buffer.

    The reference step is measured in the same test so the gate can fail: it
    must issue the `stack`/`copy_` the new path does not.
    """
    shapes = {(64, 5461): 4, (2048, 64): 4}
    a, oa, b, ob = _pair(shapes)
    copies = ("aten.stack", "aten.copy_", "aten._foreach_copy_", "aten._to_copy")

    old_ops = _dispatches(lambda: old_batched_step(oa))
    new_ops = _dispatches(lambda: ob.batched_step_from_model(b))
    assert [o for o in old_ops if o.startswith(copies)], "the reference lost its copies"
    assert not [o for o in new_ops if o.startswith(copies)], new_ops

    for grp in ob._groups:
        for i, p in enumerate(grp.params):
            view = ob.state[p]["momentum"]
            assert view.data_ptr() == grp.mom.data_ptr() + i * grp.mom[0].numel() * 4
            assert torch.equal(view, grp.mom[i])
    assert _same_params(a, b) and _same_momentum(oa, ob)


def test_a_singleton_shape_keeps_the_per_param_definition():
    """A group of one is stepped by `_step_one`, not through the 3-D ColRow.

    The batched branch would MOVE a singleton's update scale (the
    characterisation test below), so this is a gate and not a preference: with
    only singleton shapes, the batched step must equal the per-param definition
    bit-exactly.
    """
    shapes = {(256, 2048): 1, (64, 5461): 1, (8, 8): 1}
    a, oa, b, ob = _pair(shapes)
    ob.batched_step_from_model(b)
    lr, mu, ns = (oa.param_groups[0][k] for k in ("lr", "momentum", "ns_steps"))
    oa._build_groups()
    for grp in oa._groups:
        for p in grp.params:
            oa._step_one(p, lr, mu, ns, max(1.0, grp.shape[0] / grp.shape[1]) ** 0.5)
    assert _same_params(a, b) and _same_momentum(oa, ob)


# --- the checkpoint contract -------------------------------------------------


def test_state_dict_round_trip_continues_identically():
    """Save → load → step == step, and the file still carries one tensor per
    parameter (the historical format: `state[idx]["momentum"]`)."""
    shapes = {(64, 5461): 4, (2048, 64): 4, (256, 256): 2}
    a, oa, b, ob = _pair(shapes)
    old_batched_step(oa)
    ob.batched_step_from_model(b)

    sd = oa.state_dict()
    assert set(sd["state"][0]) == {"momentum"}
    assert sd["state"][0]["momentum"].shape == (64, 5461)

    fresh = shape_model(shapes)
    resumed = Muon(fresh.parameters(), lr=1e-3, colrow_3d="frobenius")
    resumed.load_state_dict(sd)
    assert len(resumed._groups) == 3, "the loader must restack per shape"
    for i, p in enumerate(resumed.param_groups[0]["params"]):
        assert torch.equal(resumed.state[p]["momentum"], sd["state"][i]["momentum"])

    # the NEXT step of the resumed optimizer is the uninterrupted one's:
    # `a` (stepped once, state `sd`) takes a second step; `d` starts from `sd`
    # and takes one — the same two steps, so the same weights.
    c, d = _twin(shapes)  # identical init to `a`, fresh identical grads
    d.load_state_dict(a.state_dict())  # d starts where `a` is after step 1
    for p, q in zip(a.parameters(), c.parameters(), strict=True):
        p.grad = q.grad.clone()
    resumed = Muon(d.parameters(), lr=1e-3, colrow_3d="frobenius")
    resumed.load_state_dict(sd)
    old_batched_step(oa)
    resumed.batched_step_from_model(d)
    assert _same_params(a, d), "the resumed optimizer's next step diverged"
    assert _same_momentum(oa, resumed)


def test_an_old_per_param_checkpoint_still_loads_and_steps():
    """The pre-change file: a per-param `momentum` per index, no stack.

    Built by hand (the loader must not care who wrote it), loaded into two
    twins, and stepped — new path vs reference — bit-exactly.
    """
    shapes = {(64, 5461): 3, (8, 128): 2}
    a, b = _twin(shapes)
    params = list(a.parameters())
    gen = torch.Generator().manual_seed(7)
    rows = [torch.randn(*p.shape, generator=gen) for p in params]
    sd = {
        "state": {i: {"momentum": rows[i]} for i in range(len(params))},
        "param_groups": [
            {"lr": 1e-3, "momentum": 0.95, "ns_steps": 5, "params": list(range(len(params)))}
        ],
    }
    oa = Muon(a.parameters(), lr=1e-3, colrow_3d="frobenius")
    ob = Muon(b.parameters(), lr=1e-3, colrow_3d="frobenius")
    oa.load_state_dict(sd)
    ob.load_state_dict(sd)
    assert len(ob._groups) == 2 and torch.equal(ob._groups[0].mom, torch.stack(rows[:3]))
    old_batched_step(oa)
    ob.batched_step_from_model(b)
    assert _same_params(a, b) and _same_momentum(oa, ob)


# --- the pre-existing divergence this gate FOUND (pinned, not fixed) ---------


@pytest.mark.parametrize("shape", [(64, 64), (8, 128), (16, 128), (64, 2048)])
def test_the_batched_colrow_is_not_the_per_param_colrow(shape):
    """The DEFECT, pinned on its own arm — `colrow_normalize`'s 3-D branch.

    With `colrow_3d="frobenius"` (the historical arm) the batched step
    normalises each slice by its Frobenius norm TWICE (dims `(1, 2)`) instead
    of columns (dim 1) then rows (dim 2), so its update is exactly
    `1/sqrt(d0)` of the per-param ColRow the port documents and the papers
    specify. Found 2026-10-16 by the equivalence gate above; the A/B (6 runs,
    small, 2k) measured the fix at **-0.0428 BPB vs this arm (8.7x the
    control's spread, zero overlap)**, so the DEFAULT flipped to `colrow` the
    same day. This test keeps the defect arm CHARACTERISED: the day someone
    "cleans up" `frobenius` into `colrow`, this goes red and the history of
    why the two arms exist is one docstring away.
    """
    torch.manual_seed(0)
    u = zeropower_via_ns(torch.randn(1, *shape), 5, "classic")
    batched = colrow_normalize(u, colrow_3d="frobenius")
    per_param = colrow_normalize(u[0])
    ratio = float(batched.norm() / per_param.norm())
    assert ratio == pytest.approx(1.0 / math.sqrt(shape[0]), rel=1e-5), ratio
    assert not torch.allclose(batched[0], per_param)
    # And the default IS the fix now: the same call without the kwarg matches
    # the per-param definition exactly.
    assert torch.equal(colrow_normalize(u), colrow_normalize(u, colrow_3d="colrow"))


# --- the arm that fixes it (`muon_colrow_3d`) --------------------------------


@pytest.mark.parametrize("shape", [(4, 64, 5461), (3, 64, 8), (2, 2048, 64), (3, 8, 128)])
def test_the_colrow_arm_is_the_per_param_definition_on_a_stack(shape):
    """`colrow_3d="colrow"` on `[N, d0, d1]` == `colrow_normalize` per slice.

    Bit-exact, not merely close: dims 1 and 2 are the same reductions the 2-D
    form takes over dims 0 and 1, so the batched arm is the definition itself.
    """
    torch.manual_seed(0)
    x = torch.randn(*shape)
    got = colrow_normalize(x, colrow_3d="colrow")
    ref = torch.stack([colrow_normalize(x[i]) for i in range(shape[0])])
    assert torch.equal(got, ref)
    assert torch.equal(colrow_normalize_(x.clone(), colrow_3d="colrow"), ref)
    # the defining property of the arm: every row of every slice is unit L2
    assert float((got.norm(dim=-1) - 1.0).abs().max()) < 1e-6


def test_the_colrow_arm_makes_the_batched_step_the_per_param_definition():
    """With the arm on, the batched step IS `_step_one` for EVERY parameter.

    The `frobenius` default diverges by `1/sqrt(d0)` on any group with >= 2
    members (the characterisation above); this is the same comparison with the
    fix in. Tolerance: **rel <= 1e-6** on each parameter (observed 2.5e-8), and
    the residual is the NS chain's batched-vs-2-D GEMM rounding, NOT ColRow —
    the ColRow step itself is bit-exact against the per-slice definition
    (`test_the_colrow_arm_is_the_per_param_definition_on_a_stack`).
    """
    shapes = {(64, 5461): 4, (2048, 64): 4, (256, 256): 3, (64, 64): 3}
    a, oa, b, ob = _pair(shapes, colrow_3d="colrow")
    lr, mu, ns = (oa.param_groups[0][k] for k in ("lr", "momentum", "ns_steps"))
    ob.batched_step_from_model(b)
    oa._build_groups()
    for grp in oa._groups:
        for p in grp.params:
            oa._step_one(p, lr, mu, ns, max(1.0, grp.shape[0] / grp.shape[1]) ** 0.5)
    worst = max(
        float((pa - pb).detach().norm() / (pa.detach().norm() + 1e-30))
        for pa, pb in zip(a.parameters(), b.parameters(), strict=True)
    )
    assert worst <= 1e-6, f"the colrow arm is not the per-param definition: {worst:.3e}"
    assert _same_momentum(oa, ob)
    # ... and it is NOT the frobenius default's step, on the same fixture
    c, oc, _, _ = _pair(shapes, colrow_3d="colrow")
    oc.batched_step_from_model(c)
    e, oe, _, _ = _pair(shapes)
    oe.batched_step_from_model(e)
    assert not _same_params(c, e), "the arm changed nothing — it is inert"


def test_the_frobenius_arm_reproduces_the_pre_fix_step():
    """The historical arm still exists and reproduces the pre-change step.

    The DEFAULT moved to `colrow` on 2026-10-17 (the A/B: -0.0428 BPB, 8.7x
    the control spread), so this test pins the OLD arm explicitly — the
    characterisation test above needs it, and a reader needs it to
    reproduce every number taken before the flip.
    """
    shapes = {(64, 5461): 3, (64, 64): 2}
    a, oa, b, ob = _pair(shapes, colrow_3d="frobenius")
    assert ob.colrow_3d == "frobenius"
    old_batched_step(oa)
    ob.batched_step_from_model(b)
    assert _same_params(a, b) and _same_momentum(oa, ob)


def test_the_default_colrow_arm_is_the_fix():
    """The flip: `Muon()` with no arm IS `colrow_3d="colrow"`, and the step
    it takes differs from the historical arm (the flip is not inert)."""
    shapes = {(64, 5461): 3, (64, 64): 2}
    b_old, o_old, b, ob = _pair(shapes, colrow_3d="frobenius")
    old_batched_step(o_old)
    ob.batched_step_from_model(b)

    c, oc, d, _ = _pair(shapes, colrow_3d="colrow")
    oc.batched_step_from_model(c)
    assert Muon(d.parameters(), lr=1e-3).colrow_3d == "colrow", "the default is the fix"
    assert not _same_params(b, c), "the two arms must not take the same step"
    del b_old


def test_a_bad_colrow_arm_is_loud_everywhere_it_can_be_set():
    from dormouse.instrument import LoudError

    model = DormouseModel(tiny_cfg())
    for build in (
        lambda: colrow_normalize(torch.randn(2, 4, 4), colrow_3d="ColRow"),
        lambda: Muon([nn.Parameter(torch.zeros(8, 8))], colrow_3d="frobenius_"),
        lambda: MixOptimizer(model, lr=1e-3, wd=0.0, colrow_3d="col_row"),
        lambda: load_run("small", train_overrides={"muon_colrow_3d": "colrow3d"}),
    ):
        with pytest.raises(LoudError) as e:
            build()
        assert "muon_colrow_3d=" in str(e.value) and "frobenius" in str(e.value)


def test_the_colrow_knob_round_trips_through_the_config():
    """A real config key: it loads, it snapshots, it pins the arm on a resume."""
    run = load_run("small", train_overrides={"muon_colrow_3d": "colrow"})
    assert run.train.muon_colrow_3d == "colrow"
    assert 'muon_colrow_3d = "colrow"' in run.snapshot_toml()
    other = load_run("small", train_overrides={"muon_colrow_3d": "frobenius"})
    assert run.diff_keys(other) == ["train.muon_colrow_3d"]


def test_the_trainer_forwards_the_colrow_arm(monkeypatch, tmp_path):
    """The wiring gate: `train/loop.py` must hand the config value to the
    optimizer, or the flag is inert and every A/B on it is a no-op."""
    from dormouse.train import loop as train_loop

    seen: list[str] = []
    real = train_loop.MixOptimizer

    def spy(model, **kw):
        seen.append(kw.get("colrow_3d"))
        return real(model, **kw)

    monkeypatch.setattr(train_loop, "MixOptimizer", spy)
    run = tiny_run(tmp_path, steps=2, log_every=2, muon_colrow_3d="colrow")
    train(DormouseModel(run.model), run, log=lambda *_: None)
    assert seen == ["colrow"], seen


# --- the bench (`scripts/bench_muon_ns.py step` runs this) -------------------


def b1_muon_shapes() -> dict[tuple[int, int], int]:
    """The b1 preset's Muon shape groups, read off the model on `meta`.

    Derived rather than hardcoded: a preset change moves the bench with it, and
    `meta` allocates nothing, so instantiating a 1.08B model here is free.
    """
    from dormouse.config import load_run

    with torch.device("meta"):
        model = DormouseModel(load_run("b1").model)
    shapes: dict[tuple[int, int], int] = {}
    for name, p in model.named_parameters():
        if optim.is_muon_param(name, p):
            shapes[tuple(p.shape)] = shapes.get(tuple(p.shape), 0) + 1
    return shapes


def _b1_params(shapes: dict[tuple[int, int], int], device) -> list[nn.Parameter]:
    """Real parameters and gradients for the b1 Muon groups (what Muon sees)."""
    gen = torch.Generator().manual_seed(0)
    out: list[nn.Parameter] = []
    for shape, count in shapes.items():
        for _ in range(count):
            p = nn.Parameter((torch.randn(*shape, generator=gen) * 0.05).to(device))
            p.grad = (torch.randn(*shape, generator=gen) * 0.1).to(device)
            out.append(p)
    return out


def bench_muon_b1_step(device=None, reps: int = 5, warm: int = 2) -> None:
    """Old vs new: the batched Muon step at the b1 shape groups.

    Both implementations run on the SAME parameters and gradients (restored
    between runs), so the comparison is of two programs and not of two inputs.
    Prints wall ms/step, the profiler's CUDA time and kernel mix, the aten
    dispatch counts, and peak VRAM.
    """
    import time

    from dormouse.train.optim import Muon

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        from test_muon_gram import _gpu_free

        free, why = _gpu_free()
        print(f"card: {why}")
        if not free:
            print("  REFUSING: one GPU thing at a time (AGENTS §1.5). Re-run when idle.")
            return
    shapes = b1_muon_shapes()
    n_params = sum(shapes.values())
    gib = sum(c * s[0] * s[1] * 4 for s, c in shapes.items()) / 2**30
    print(f"b1 Muon groups: {len(shapes)} shapes, {n_params} params, {gib:.2f} GiB fp32 each")
    for s, c in sorted(shapes.items(), key=lambda kv: -kv[1] * kv[0][0] * kv[0][1]):
        print(f"  {str(s):>14s} x{c:<5d} {c * s[0] * s[1] * 4 / 2**20:9.1f} MiB")

    params = _b1_params(shapes, device)
    init = [p.detach().clone() for p in params]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def step_ms(fn, reps_: int = 3, warm_: int = 1) -> float:
        for _ in range(warm_):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps_):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t) / reps_ * 1e3

    for precision in ("highest", "high"):
        torch.set_float32_matmul_precision(precision)
        print(
            f"\n=== matmul precision {precision} "
            f"({'fp32' if precision == 'highest' else 'tf32'}) ==="
        )
        rows = []
        for label in ("old", "new"):
            muon = Muon(params, lr=1e-3)
            for i, p in enumerate(params):
                p.detach().copy_(init[i])

            def run(_muon=muon, _old=label == "old") -> int:
                return old_batched_step(_muon) if _old else _muon.batched_step_from_model(None)

            for _ in range(warm):
                run()
            if device == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(reps):
                run()
            if device == "cuda":
                torch.cuda.synchronize()
            wall = (time.perf_counter() - t) / reps * 1e3
            ops = _dispatches(run)
            peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else float("nan")
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA], record_shapes=False
            ) as prof:
                run()
                if device == "cuda":
                    torch.cuda.synchronize()
            ka = prof.key_averages()
            cuda_ms = sum(k.self_device_time_total for k in ka) / 1e3
            rows.append((label, wall, cuda_ms, ops, peak, ka))
            del muon
            if device == "cuda":
                torch.cuda.empty_cache()

        def n(ops: list[str], prefixes) -> int:
            return sum(o.startswith(prefixes) for o in ops)

        print(
            f"{'impl':5s} {'wall ms':>9s} {'CUDA ms':>9s} {'mm/bmm':>7s} "
            f"{'stack':>6s} {'copy_':>6s} {'all aten':>9s} {'peak GiB':>9s}"
        )
        for label, wall, cuda_ms, ops, peak, _ka in rows:
            print(
                f"{label:5s} {wall:9.1f} {cuda_ms:9.1f} "
                f"{n(ops, ('aten.mm', 'aten.bmm', 'aten.matmul')):7d} "
                f"{n(ops, 'aten.stack'):6d} "
                f"{n(ops, ('aten.copy_', 'aten._foreach_copy_', 'aten._to_copy')):6d} "
                f"{len(ops):9d} {peak:9.2f}"
            )
        old, new = rows
        print(
            f"  step {old[1]:.1f} -> {new[1]:.1f} ms wall "
            f"({100 * (new[1] / old[1] - 1):+.1f}%), CUDA {old[2]:.1f} -> {new[2]:.1f} ms"
        )
        print("  top CUDA kernels (new):")
        for k in sorted(new[5], key=lambda k: -k.self_device_time_total)[:6]:
            print(f"    {k.self_device_time_total / 1e3:8.1f} ms  x{k.count:<5d} {k.key[:60]}")

        # The chunk cap: fewer, larger batched GEMMs against the fp32 footprint.
        print(f"  chunk cap sweep (new, precision {precision}):")
        muon = Muon(params, lr=1e-3)
        for cap in (1 << 25, 1 << 26, 1 << 27):
            optim.MUON_CHUNK_ELEMS = cap
            for i, p in enumerate(params):
                p.detach().copy_(init[i])
            try:
                ms = step_ms(lambda _m=muon: _m.batched_step_from_model(None))
            except torch.OutOfMemoryError:
                print(
                    f"    cap 2**{cap.bit_length() - 1:2d}: OOM (this bench also holds a "
                    f"2.5 GiB param snapshot the trainer does not)"
                )
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
            peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else float("nan")
            print(
                f"    cap 2**{cap.bit_length() - 1:2d} ({cap / 2**20:7.1f}M elems): "
                f"{ms:8.1f} ms   peak {peak:5.2f} GiB"
            )
        optim.MUON_CHUNK_ELEMS = 1 << 25
    torch.set_float32_matmul_precision("highest")


if __name__ == "__main__":
    bench_muon_b1_step()


# --- the trainer's entry point reaches it -----------------------------------


def test_mix_optimizer_uses_the_fast_path(byte_tensor):
    """The arm must show it ran: `MixOptimizer.step` builds the shape groups."""
    model = DormouseModel(tiny_cfg())
    ids, targets = byte_tensor()
    _, loss, _ = model(ids, targets)
    loss.backward()
    opt = MixOptimizer(model, lr=1e-3, wd=0.0)
    opt.step()
    assert opt.steps_batched > 0 and opt.muon._groups
    stacked = sum(g.mom.shape[0] for g in opt.muon._groups)
    routed = sum(
        1 for n, p in model.named_parameters() if p.ndim == 2 and optim.is_muon_param(n, p)
    )
    assert stacked == routed > 0


# --- the two dtype rungs (`muon_state_dtype`, `ns_dtype`) --------------------
#
# Both are OPT-IN and default to `fp32`, which is every number on record, so the
# gates below come in three shapes: the default must be the SAME PROGRAM it has
# always been (bit-exactly), the low-precision rung must be the rung it claims
# to be (transcribed, not merely "close"), and its deviation from `fp32` must be
# inside a bound that is DERIVED, not chosen to pass.
#
# The state rung's bound, from the arithmetic: the bf16 STORE rounds each element
# to 8 mantissa bits, i.e. rel `2^-9 = 1.95e-3`; the momentum is an EMA with
# `mu = 0.95`, so the store's error does not accumulate (it contracts by `mu`
# each step) and the relative deviation of the buffer stays at the store's own
# resolution. A Lipschitz constant of the polar factor near 1 then carries it to
# the update. Measured on the b1-like fixture below: **3.6e-3** worst per-param
# relative update difference (2.1M elements, 3 steps), 0.9998 elementwise sign
# agreement, per-param cosine >= 0.999993. The gate is 1e-2 — 2.8x the measured
# worst — and it is stated as `by design approximate`, not as an equality.
#
# Stochastic rounding (arXiv:2509.23106's 8-bit rung) is the NEXT rung and is
# NOT implemented: this is round-to-nearest storage only.


def _bf16_chain_ref(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """The bf16 NS chain, TRANSCRIBED: fp32 in -> bf16 polynomial -> fp32 out.

    Independent of the implementation under test (written from
    `zeropower_via_ns`'s fp32 body with the two casts), so the gate below is a
    comparison of two programs. A silent fallback to the fp32 chain fails it.
    """
    a, b, c = optim.JORDAN_NS_COEFFS
    x = g.to(torch.float32)
    x = x / (x.norm() + 1e-7)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T.contiguous()
    x = x.to(torch.bfloat16)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    out = x.T.contiguous() if transposed else x
    return out.to(torch.float32)


def _ortho_per_entry(o: torch.Tensor) -> float:
    """`‖O Oᵀ − I‖_F / rank` on the SMALL side — the AGENTS §2.3 metric."""
    m, n = o.shape[-2], o.shape[-1]
    prod = o @ o.transpose(-1, -2) if m <= n else o.transpose(-1, -2) @ o
    rank = min(m, n)
    return float((prod - torch.eye(rank)).norm()) / rank


def _b1_like_fixture(shape: tuple[int, int], seed: int = 0) -> torch.Tensor:
    """A momentum-shaped buffer: a decaying spectrum, the real factors' regime.

    A raw Gaussian is NOT the right fixture for an orthogonality bound: the
    Frobenius prescale leaves its `sigma_max` at ~0.125, so the fp32 chain
    itself is 5.9e-2 per-entry off the manifold there. The TSCT factors'
    momentum is a sum of gradients of a low-rank factor, i.e. heavy-spectrum.
    """
    gen = torch.Generator().manual_seed(seed)
    r = min(shape)
    s = torch.logspace(0, -2, r)
    return (
        torch.randn(*shape, generator=gen) @ torch.diag(s)
        if shape[0] >= shape[1]
        else torch.diag(s) @ torch.randn(*shape, generator=gen)
    )


def test_the_dtype_rungs_default_to_todays_behaviour_bit_exactly():
    """The flags are opt-in: the default IS the fp32 program, and so is `fp32`.

    Three statements, all bit-exact: the defaults are `fp32`; the default
    `Muon` takes the step the pre-change reference takes (the pin above is
    untouched, this is the same comparison with the dtype knobs in scope); and
    naming `fp32` EXPLICITLY is the same program as leaving it out, so the
    default is not a special path.
    """
    assert optim.OPT_DTYPES == ("fp32", "bf16")
    run = load_run("small")
    assert run.train.muon_state_dtype == "fp32" and run.train.ns_dtype == "fp32"
    shapes = {(64, 5461): 3, (2048, 64): 3, (256, 256): 2}
    a, oa, b, ob = _pair(shapes)
    assert oa.state_dtype == "fp32" and oa.ns_dtype == "fp32"
    old_batched_step(oa)
    ob.batched_step_from_model(b)
    assert _same_params(a, b) and _same_momentum(oa, ob)
    assert all(g.mom.dtype == torch.float32 for g in ob._groups)
    assert all(ob.state[p]["momentum"].dtype == torch.float32 for p in ob.param_groups[0]["params"])
    assert ob.state_dict()["state"][0]["momentum"].dtype == torch.float32

    e, f = _twin(shapes)
    oe = Muon(e.parameters(), lr=1e-3, colrow_3d="frobenius", state_dtype="fp32", ns_dtype="fp32")
    of = Muon(f.parameters(), lr=1e-3, colrow_3d="frobenius")
    oe.batched_step_from_model(e)
    of.batched_step_from_model(f)
    assert _same_params(e, f), "the explicit fp32 rung is not the default"
    assert _same_momentum(oe, of)


def test_the_bf16_state_rung_halves_the_state_and_is_finite():
    """The buffers are bf16, the arithmetic is fp32, the state is exactly half."""
    shapes = {(64, 5461): 4, (2048, 64): 4, (256, 256): 2}
    a, oa, b, ob = _pair(shapes, state_dtype="bf16")
    oa.batched_step_from_model(a)
    ob.batched_step_from_model(b)
    assert ob._groups and all(g.mom.dtype == torch.bfloat16 for g in ob._groups)
    assert all(
        ob.state[p]["momentum"].dtype == torch.bfloat16 for p in ob.param_groups[0]["params"]
    )
    assert all(bool(torch.isfinite(g.mom).all()) for g in ob._groups)
    assert all(bool(torch.isfinite(p).all()) for p in b.parameters())
    # the state is EXACTLY half — the memory claim, not an estimate
    assert sum(g.mom.numel() * g.mom.element_size() for g in ob._groups) * 2 == sum(
        g.mom.numel() * 4 for g in oa._groups
    )
    # ... and the b1 preset's own groups, read off the meta model: 2.46 -> 1.23 GiB
    b1 = b1_muon_shapes()
    fp32_bytes = sum(c * s[0] * s[1] * 4 for s, c in b1.items())
    assert fp32_bytes / 2**30 == pytest.approx(2.4616, abs=1e-3)
    assert fp32_bytes // 2 / 2**30 == pytest.approx(1.2308, abs=1e-3)


def test_the_bf16_state_rung_moves_the_params_the_same_way():
    """Sign pattern, direction and the 1e-2 bound, on the b1-like fixture.

    The tolerance is DERIVED in the section comment: bf16's store resolution is
    `2^-9 = 1.95e-3` relative and the EMA does not accumulate it, so a 1e-2 gate
    is 2.8x the measured worst (3.6e-3) and still fails a rung that lost the
    fp32 accumulation (which would be ~4e-2 out, `2^-9/(1-mu)`).
    """
    shapes = {(64, 5461): 4, (2048, 64): 4, (256, 256): 3, (64, 64): 3}
    a, b = _twin(shapes)
    before = [p.detach().clone() for p in a.parameters()]
    oa = Muon(a.parameters(), lr=1e-3, state_dtype="fp32")
    ob = Muon(b.parameters(), lr=1e-3, state_dtype="bf16")
    for _ in range(3):
        for p, q in zip(a.parameters(), b.parameters(), strict=True):
            p.grad = q.grad.clone()
        oa.batched_step_from_model(a)
        ob.batched_step_from_model(b)
    n_same = n_el = 0
    for p, q, p0 in zip(a.parameters(), b.parameters(), before, strict=True):
        d32 = p.detach() - p0
        d16 = q.detach() - p0
        assert bool(torch.isfinite(d16).all())
        n_same += int((d16.sign() == d32.sign()).sum())
        n_el += d16.numel()
        rel = float((d16 - d32).norm() / (d32.norm() + 1e-30))
        cos = float((d16.flatten() @ d32.flatten()) / (d16.norm() * d32.norm() + 1e-30))
        assert rel <= 1e-2, f"rel {rel:.3e} at {tuple(d32.shape)}"
        assert cos >= 0.9999, f"cos {cos:.6f} at {tuple(d32.shape)}"
    assert n_el > 2_000_000, "the fixture must be b1-sized"
    assert n_same / n_el >= 0.999, f"sign agreement {n_same}/{n_el}"
    assert not _same_params(a, b), "the rung moved nothing — it is inert"


def test_the_bf16_state_rung_accumulates_in_fp32_then_stores_bf16():
    """Step 1 of the bf16 rung is the fp32 rung's momentum ROUNDED to bf16.

    That is the whole claim of the rung, and it is checkable exactly at step 1
    (both rungs start from a zero buffer, so the fp32 accumulation is the same
    program): a bf16 ACCUMULATION would round `(1-mu)·g` in bf16 and the two
    would differ. From step 2 the buffers are different inputs to the same
    recurrence, which is why the deviation gate above is a tolerance.
    """
    shapes = {(64, 5461): 2, (2048, 64): 2}
    a, b = _twin(shapes)
    oa = Muon(a.parameters(), lr=1e-3, state_dtype="fp32")
    ob = Muon(b.parameters(), lr=1e-3, state_dtype="bf16")
    oa.batched_step_from_model(a)
    ob.batched_step_from_model(b)
    for g32, g16 in zip(oa._groups, ob._groups, strict=True):
        assert g16.mom.dtype == torch.bfloat16
        assert torch.equal(g16.mom, g32.mom.to(torch.bfloat16)), (
            f"the bf16 rung is not round_bf16(fp32 accumulate) at {tuple(g32.mom.shape)}"
        )


def test_the_bf16_ns_chain_is_the_transcribed_program():
    """`ns_dtype="bf16"` runs the transcribed bf16 chain — bit-exactly.

    And it is a DIFFERENT program from the fp32 chain on the same input, which
    is the "the arm must show it ran" half: a silent fallback passes nothing
    here.
    """
    g = torch.randn(64, 5461, generator=torch.Generator().manual_seed(0)) * 0.05
    got = zeropower_via_ns(g, 5, "classic", ns_dtype="bf16")
    assert got.dtype == torch.float32, "the output cast must be explicit fp32"
    assert torch.equal(got, _bf16_chain_ref(g)), "the bf16 rung is not the transcribed chain"
    assert not torch.equal(got, zeropower_via_ns(g, 5, "classic")), "the bf16 rung is inert"
    # the batched (3-D) branch runs the same chain, per slice
    x = torch.randn(3, 64, 5461, generator=torch.Generator().manual_seed(1)) * 0.05
    b3 = zeropower_via_ns(x, 5, "classic", ns_dtype="bf16")
    per = torch.stack([zeropower_via_ns(x[i], 5, "classic", ns_dtype="bf16") for i in range(3)])
    assert float((b3 - per).norm() / per.norm()) <= 1e-2  # measured 0.0 on this fixture


@pytest.mark.parametrize("kind", optim.NS_KINDS)
@pytest.mark.parametrize("shape", [(2048, 64), (64, 5461)])
def test_the_bf16_ns_chain_keeps_the_polar_property(shape, kind):
    """The bf16 chain stays on the manifold as well as the fp32 chain does.

    The bound is stated RELATIVE to the fp32 arm, because the absolute number is
    a property of the input, not of the rung: on this heavy-spectrum fixture the
    fp32 chain itself sits 4.3e-2 per-entry off `O Oᵀ = I` (5 NS iterations at
    `sigma_max ~= 1.2`), and bf16 must not make that materially worse. Measured
    worst ratio across the four arms: 1.02. Gate: 1.1x and <= 6e-2 absolute.
    """
    g = _b1_like_fixture(shape)
    ref = zeropower_via_ns(g, 5, kind)
    got = zeropower_via_ns(g, 5, kind, ns_dtype="bf16")
    o32, o16 = _ortho_per_entry(ref), _ortho_per_entry(got)
    assert bool(torch.isfinite(got).all())
    assert o16 <= 1.1 * o32, f"ortho {o16:.3e} vs fp32 {o32:.3e} at {shape}/{kind}"
    assert o16 <= 6e-2, o16


@pytest.mark.parametrize("kind", optim.NS_KINDS)
def test_the_bf16_ns_chain_keeps_the_spectrum_bounds_the_fp32_gate_states(kind):
    """The existing polar assertion's shape, with the bf16 rung in.

    `test_train.py::test_muon_update_compresses_the_spectrum_and_is_not_a_copy`
    states the property Muon's lr is calibrated against (a ~1-scale update with
    the spread squeezed out); the same fixture and the same bounds must hold
    when the polynomial runs in bf16.
    """
    heavy = torch.randn(64, 8, generator=torch.Generator().manual_seed(0)) @ torch.diag(
        torch.logspace(0, -2, 8)
    )
    o = zeropower_via_ns(heavy, 5, kind, ns_dtype="bf16")
    s_out = torch.linalg.svdvals(o)
    assert float(s_out[0] / s_out[-1]) < 2.0, "the spectrum must be compressed"
    assert 0.5 < float(s_out[-1]) and float(s_out[0]) < 1.2, "the update scale must be ~1"


@pytest.mark.parametrize("state_dtype", optim.OPT_DTYPES)
def test_the_state_dict_round_trips_on_both_state_rungs(state_dtype):
    """Save -> load -> the SAME next step, on `fp32` and on `bf16`.

    The historical format (one `momentum` tensor per parameter) is what
    `state_dict` emits either way; on the bf16 rung torch's loader upcasts the
    stored tensor to the param's fp32 dtype and `_build_groups` casts it back
    down, which is value-preserving because bf16 -> fp32 -> bf16 is exact.
    """
    shapes = {(64, 5461): 4, (2048, 64): 4, (256, 256): 2}
    a, oa, b, ob = _pair(shapes, state_dtype=state_dtype)
    oa.batched_step_from_model(a)
    ob.batched_step_from_model(b)
    assert _same_params(a, b) and _same_momentum(oa, ob)  # one program, two runs

    sd = oa.state_dict()
    want = torch.bfloat16 if state_dtype == "bf16" else torch.float32
    assert sd["state"][0]["momentum"].dtype == want
    fresh = shape_model(shapes)
    resumed = Muon(fresh.parameters(), lr=1e-3, colrow_3d="frobenius", state_dtype=state_dtype)
    resumed.load_state_dict(sd)
    assert len(resumed._groups) == 3, "the loader must restack per shape"
    for i, p in enumerate(resumed.param_groups[0]["params"]):
        assert torch.equal(resumed.state[p]["momentum"], sd["state"][i]["momentum"])
        assert resumed.state[p]["momentum"].dtype == want

    c, d = _twin(shapes)
    d.load_state_dict(a.state_dict())
    for p, q in zip(a.parameters(), c.parameters(), strict=True):
        p.grad = q.grad.clone()
    resumed = Muon(d.parameters(), lr=1e-3, colrow_3d="frobenius", state_dtype=state_dtype)
    resumed.load_state_dict(sd)
    oa.batched_step_from_model(a)
    resumed.batched_step_from_model(d)
    assert _same_params(a, d), "the resumed optimizer's next step diverged"
    assert _same_momentum(oa, resumed)


def test_the_mix_optimizer_carries_both_rungs_and_the_config_round_trips():
    """The knobs are real config keys, they reach `Muon`, and a flip is refused
    on a resume (`diff_keys`) — the two reasons a live run cannot drift."""
    model = DormouseModel(tiny_cfg())
    opt = MixOptimizer(model, lr=1e-3, wd=0.0, muon_state_dtype="bf16", ns_dtype="bf16")
    assert opt.muon.state_dtype == "bf16" and opt.muon.ns_dtype == "bf16"
    assert MixOptimizer(model, lr=1e-3, wd=0.0).muon.state_dtype == "fp32"
    run = load_run("small", train_overrides={"muon_state_dtype": "bf16", "ns_dtype": "bf16"})
    assert 'muon_state_dtype = "bf16"' in run.snapshot_toml()
    assert 'ns_dtype = "bf16"' in run.snapshot_toml()
    assert run.diff_keys(load_run("small")) == ["train.muon_state_dtype", "train.ns_dtype"]
    # a snapshot that LACKS the keys fills the default (resume-safe by construction)
    older = (
        run.snapshot_toml()
        .replace('muon_state_dtype = "bf16"\n', "")
        .replace('ns_dtype = "bf16"\n', "")
    )
    from dormouse.config import RunConfig

    assert RunConfig.from_snapshot(older).train.muon_state_dtype == "fp32"


def test_a_bad_dtype_rung_is_loud_everywhere_it_can_be_set():
    from dormouse.instrument import LoudError

    model = DormouseModel(tiny_cfg())
    for field, build in (
        ("state_dtype", lambda: Muon([nn.Parameter(torch.zeros(8, 8))], state_dtype="fp16")),
        (
            "muon_state_dtype",
            lambda: MixOptimizer(model, lr=1e-3, wd=0.0, muon_state_dtype="float32"),
        ),
        (
            "muon_state_dtype",
            lambda: load_run("small", train_overrides={"muon_state_dtype": "fp16"}),
        ),
        ("ns_dtype", lambda: zeropower_via_ns(torch.randn(4, 4), 5, ns_dtype="float16")),
        ("ns_dtype", lambda: Muon([nn.Parameter(torch.zeros(8, 8))], ns_dtype="Bf16")),
        ("ns_dtype", lambda: MixOptimizer(model, lr=1e-3, wd=0.0, ns_dtype="fp16")),
        ("ns_dtype", lambda: load_run("small", train_overrides={"ns_dtype": "bf16x"})),
    ):
        with pytest.raises(LoudError) as e:
            build()
        assert f"{field}=" in str(e.value) and "bf16" in str(e.value), str(e.value)
