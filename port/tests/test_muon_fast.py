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
from conftest import tiny_cfg
from torch import nn

from dormouse.model import DormouseModel
from dormouse.train import optim
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
                    zeropower_via_ns(buf, ns, muon.ns_kind, muon.gram_restarts)
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
                zeropower_via_ns(bufs, ns, muon.ns_kind, muon.gram_restarts)
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


def _pair(shapes: dict[tuple[int, int], int]) -> tuple[nn.Module, Muon, nn.Module, Muon]:
    a, b = _twin(shapes)
    return a, Muon(a.parameters(), lr=1e-3), b, Muon(b.parameters(), lr=1e-3)


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
    resumed = Muon(fresh.parameters(), lr=1e-3)
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
    resumed = Muon(d.parameters(), lr=1e-3)
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
    oa = Muon(a.parameters(), lr=1e-3)
    ob = Muon(b.parameters(), lr=1e-3)
    oa.load_state_dict(sd)
    ob.load_state_dict(sd)
    assert len(ob._groups) == 2 and torch.equal(ob._groups[0].mom, torch.stack(rows[:3]))
    old_batched_step(oa)
    ob.batched_step_from_model(b)
    assert _same_params(a, b) and _same_momentum(oa, ob)


# --- the pre-existing divergence this gate FOUND (pinned, not fixed) ---------


@pytest.mark.parametrize("shape", [(64, 64), (8, 128), (16, 128), (64, 2048)])
def test_the_batched_colrow_is_not_the_per_param_colrow(shape):
    """CHARACTERISATION of a pre-existing defect — `colrow_normalize`'s 3-D branch.

    The batched step normalises each slice by its Frobenius norm TWICE
    (dims `(1, 2)`) instead of columns (dim 1) then rows (dim 2), so its update
    is exactly `1/sqrt(d0)` of the per-param ColRow the port documents and the
    papers specify. Found 2026-10-16 by the equivalence gate above; the fix is
    one line in `colrow_normalize` and it MOVES the trainer's Muon update scale
    for every shape group with >= 2 members (8x at `[64, 5461]`, 45x at
    `[2048, 2048]`), so it is an A/B-gated numerics change (AGENTS §1.2) and is
    NOT landed by a speed commit.

    When the fix lands, THIS test goes red — which is the point: it is the
    marker that the A/B was owed.
    """
    torch.manual_seed(0)
    u = zeropower_via_ns(torch.randn(1, *shape), 5, "classic")
    batched = colrow_normalize(u)
    per_param = colrow_normalize(u[0])
    ratio = float(batched.norm() / per_param.norm())
    assert ratio == pytest.approx(1.0 / math.sqrt(shape[0]), rel=1e-5), ratio
    assert not torch.allclose(batched[0], per_param)


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
