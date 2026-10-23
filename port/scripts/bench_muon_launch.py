"""Kernel launches in the Muon step: the pre-change path vs the lockstep chain.

    uv run --extra gpu python scripts/bench_muon_launch.py
    uv run --extra gpu python scripts/bench_muon_launch.py --state fp32 --ns fp32

**The number this prints is a LAUNCH count, not a wall clock only.** The port's
step is host-dispatch-bound (`docs/reviews/2026-10-16-b1-profile.md`: a GPU
residue of ≈0 ms against a wall that is the enqueue), so the optimizer stage's
cost is the number of kernels it enqueues. This driver measures that number for
the two optimizer arms and times the stage; it does not score a training run.

**Arms.**

* `main` — the path this branch replaces, transcribed below from `b36ef27`
  (`batched_step_main` / `_momentum_update_main`). It is a transcription, not a
  git checkout: the gate in `tests/test_muon_launch.py` pins it against the
  byte hashes captured from a clean `main` worktree, so a drift between the two
  is a red test, not a silent comparison against the wrong program.
* `new` — `MixOptimizer.step` on this branch.

**What is counted.** `LaunchCounter` models the two CUDA rules a CPU dispatch
count does not see:

1. metadata aten ops (`detach`, `view`, `expand`, `transpose`, …) enqueue NO
   kernel and are not launches;
2. `torch._foreach_*` takes the fused `multi_tensor_apply` route only when the
   lists share a dtype/device and the tensors are dense (`ForeachUtils.h`),
   packing up to 64 (depth 2) tensors and 320 blocks per launch
   (`MultiTensorApply.cuh`); a mixed dtype falls back to one launch per tensor.

It is a MODEL of those rules, not a trace. `scripts/launch_profile.py` is the
ground truth on a free card (`cudaLaunchKernel`); this driver is what a CPU-only
gate can run and what makes the two arms comparable on the same model.

**Protocol.** `medium`, b8 × s512, deterministic gradients, one warm step, then
the MINIMUM of `--reps` steps of `MixOptimizer.step()` (the optimizer stage
alone) with the device flag `None`, i.e. the plain step. The batch/sequence are
labels for the step the optimizer belongs to: the optimizer stage never sees
them, it sees the gradient shapes. The per-group table is printed from the
built optimizer: shape, tensor count, chunks, which NS chain runs (and why a
fallback was refused), and that group's own launch count under each arm. The
lockstep chain shares its elementwise launches across groups, so the per-group
column is an upper bound; the step total is the measurement.

**Preflight.** One GPU process at a time (AGENTS §1.5): a card holding another
compute process prints a SKIP banner and exits 0, because nothing was measured.
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight, skip  # noqa: E402

from dormouse.config import load_run  # noqa: E402
from dormouse.model import DormouseModel  # noqa: E402
from dormouse.train.optim import MUON_CHUNK_ELEMS, MixOptimizer, colrow_normalize_  # noqa: E402

#: `MultiTensorApply.cuh`'s limits (depth 2 = a TensorList pair).
_FOREACH_BLOCK = 320
_FOREACH_TENSORS = 64
_FOREACH_CHUNK = 65536

#: aten ops that are VIEWS/metadata or ALLOCATIONS and enqueue no kernel.
#: `clone`, `copy_`, `_to_copy`, `stack`, `cat` are NOT here: they move bytes
#: and launch. `empty*`/`zeros*`/`ones*` reserve storage (the allocator may call
#: `cudaMalloc`, not `cudaLaunchKernel`), and the functional `_foreach_*` forms
#: dispatch them explicitly where `b * x` allocates inside its own op.
_METADATA = (
    "aten.detach",
    "aten.view",
    "aten.expand",
    "aten.permute",
    "aten.transpose",
    "aten.slice",
    "aten.select",
    "aten.as_strided",
    "aten._unsafe_view",
    "aten.alias",
    "aten.t.",
    "aten.squeeze",
    "aten.unsqueeze",
    "aten.narrow",
    "aten.split",
    "aten.unbind",
    "aten.lift_fresh",
    "aten.resolve_conj",
    "aten.resolve_neg",
    "aten.is_contiguous",
    "aten.numel",
    "aten.size",
    "aten.stride",
    "aten.dim",
    "aten.item",
    "aten._local_scalar_dense",
    "aten.format",
    "aten.empty",
    "aten.new_empty",
    "aten.zeros",
    "aten.new_zeros",
    "aten.ones",
    "aten.new_ones",
    "aten.full",
    "aten.new_full",
)


def _tensor_lists(args) -> list[list[torch.Tensor]]:
    return [
        a
        for a in args
        if isinstance(a, (list, tuple)) and a and all(isinstance(t, torch.Tensor) for t in a)
    ]


def _fast_foreach_launches(tensors: list[torch.Tensor]) -> int:
    """`multi_tensor_apply`'s launch count for one homogeneous tensor list."""
    blocks = sum(-(-t.numel() // _FOREACH_CHUNK) for t in tensors)
    return max(-(-blocks // _FOREACH_BLOCK), -(-len(tensors) // _FOREACH_TENSORS))


class LaunchCounter(TorchDispatchMode):
    """Counts the CUDA kernels a block of optimizer code would enqueue."""

    def __init__(self) -> None:
        super().__init__()
        self.launches = 0
        self.foreach: collections.Counter = collections.Counter()
        self.ops: collections.Counter = collections.Counter()
        self._depth = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if self._depth == 0:
            if name.startswith("aten._foreach_"):
                lists = _tensor_lists(args)
                homogeneous = bool(lists) and all(
                    t.dtype == lists[0][0].dtype
                    and t.device == lists[0][0].device
                    and t.is_contiguous()
                    for lst in lists
                    for t in lst
                )
                key = (name, lists[0].__len__() if lists else 0, homogeneous)
                self.foreach[key] += 1
                if homogeneous:
                    self.launches += _fast_foreach_launches(lists[0])
                else:
                    self.launches += len(lists[0]) if lists else 1
            elif name.startswith(_METADATA):
                pass
            else:
                self.launches += 1
                self.ops[name] += 1
        self._depth += 1
        try:
            return func(*args, **(kwargs or {}))
        finally:
            self._depth -= 1

    def line(self) -> str:
        return (
            f"launches={self.launches} foreach_calls={sum(self.foreach.values())} "
            f"aten_ops={sum(self.ops.values())}"
        )


# --- the pre-change program, transcribed from b36ef27 -----------------------


def _momentum_update_main(mom: torch.Tensor, params: list, mu: float) -> None:
    if mom.dtype != torch.bfloat16:
        mom.mul_(mu)
        torch._foreach_add_(torch.unbind(mom, 0), [p.grad for p in params], alpha=1.0 - mu)
        return
    n = len(params)
    n_chunk = max(1, min(n, MUON_CHUNK_ELEMS // mom[0].numel()))
    for start in range(0, n, n_chunk):
        chunk = params[start : start + n_chunk]
        acc = mom[start : start + n_chunk].to(torch.float32)
        acc.mul_(mu)
        torch._foreach_add_(torch.unbind(acc, 0), [p.grad for p in chunk], alpha=1.0 - mu)
        mom[start : start + n_chunk].copy_(acc)


def batched_step_main(muon, model, flag: torch.Tensor | None = None) -> int:
    """`Muon.batched_step_from_model` at `b36ef27`, field for field."""
    if not muon._groups:
        muon._build_groups()
    n_batches = 0
    for grp in muon._groups:
        group = muon.param_groups[grp.gi]
        lr, mu, ns = group["lr"], group["momentum"], group["ns_steps"]
        scale = max(1.0, grp.shape[0] / grp.shape[1]) ** 0.5
        params = grp.params
        if len(params) < 2 or any(p.grad is None for p in params):
            for p in params:
                if p.grad is not None:
                    muon._step_one(p, lr, mu, ns, scale, flag)
            continue
        mom = grp.mom
        _momentum_update_main(mom, params, mu)
        n = len(params)
        if muon.sf:
            s_new, c = muon._sf_ckp(lr, float(muon.state[params[0]].get("sf_s", 0.0)))
            for p in params:
                muon.state[p]["sf_s"] = s_new
        n_chunk = max(1, min(n, MUON_CHUNK_ELEMS // (grp.shape[0] * grp.shape[1])))
        for start in range(0, n, n_chunk):
            params_c = params[start : start + n_chunk]
            update = muon._orthogonalize(mom[start : start + n_chunk], ns)
            if muon.sf:
                muon._sf_batched(grp, start, len(params_c), params_c, update, lr, c)
                continue
            if muon.variant == "normuon":
                muon._nor_batched(grp, start, len(params_c), params_c, update, lr, flag)
                continue
            colrow_normalize_(update, colrow_3d=muon.colrow_3d).mul_(-lr * scale)
            if flag is not None:
                update.mul_(flag.to(update.dtype))
            with torch.no_grad():
                torch._foreach_add_([p.detach() for p in params_c], torch.unbind(update))
        n_batches += 1
    return n_batches


# --- the table ---------------------------------------------------------------


def shape_groups(muon) -> list[tuple[tuple[int, int], list]]:
    if not muon._groups:
        muon._build_groups()
    return [(grp.shape, grp.params) for grp in muon._groups]


def chunk_bounds(shape: tuple[int, int], n: int) -> list[tuple[int, int]]:
    n_chunk = max(1, min(n, MUON_CHUNK_ELEMS // (shape[0] * shape[1])))
    return [(s, min(s + n_chunk, n)) for s in range(0, n, n_chunk)]


def group_path(muon, shape: tuple[int, int], n: int) -> tuple[str, str]:
    """`(path, reason)` for a group — the same decision the step makes."""
    if muon.ns_kind != "triton":
        return "classic", f"ns_kind={muon.ns_kind!r}"
    from dormouse.train.ns_kernels import path_reason

    probe = muon._groups[[g.shape for g in muon._groups].index(shape)].mom[0]
    why = path_reason(probe, muon.ns_dtype)
    return ("triton", "") if why is None else ("classic", why)


def table(muon, arm: str) -> list[tuple]:
    """The per-group table: shape, tensors, chunks, path, reason, NS launches.

    The launch column runs THAT GROUP's Newton–Schulz chain alone under the
    counter. The new step shares the elementwise launches across groups, so the
    column is an upper bound for `new`; the step total is the measurement.
    """
    rows = []
    for shape, params in shape_groups(muon):
        n = len(params)
        path, reason = group_path(muon, shape, n)
        bounds = chunk_bounds(shape, n)
        ns = muon.param_groups[0]["ns_steps"]
        pieces = [
            muon._groups[[g.shape for g in muon._groups].index(shape)].mom[a:b] for a, b in bounds
        ]
        c = LaunchCounter()
        with c:
            if arm == "main":
                for piece in pieces:
                    muon._orthogonalize(piece, ns)
            else:
                muon._orthogonalize_many(pieces, ns)
        rows.append((shape, n, len(bounds), path, reason, c.launches))
    return rows


# --- the measurement ---------------------------------------------------------


def build(preset: str, *, state: str, ns: str, ns_kind: str, device: str):
    run = load_run(preset)
    torch.manual_seed(0)
    model = DormouseModel(run.model, device=device)
    opt = MixOptimizer(
        model,
        lr=run.train.lr,
        wd=run.train.wd,
        opt=run.train.opt,
        ns_kind=ns_kind,
        colrow_3d=run.train.muon_colrow_3d,
        muon_state_dtype=state,
        ns_dtype=ns,
    )
    gen = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.randn(p.shape, generator=gen).to(device=p.device, dtype=p.dtype)
    if device == "cuda":
        torch.cuda.synchronize()
    return model, opt


def wall_ms(opt, reps: int, warm: int) -> float:
    for _ in range(warm):
        opt.step()
    times = []
    for _ in range(reps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return min(times)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--state", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--ns", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--ns-kind", default="triton", choices=["classic", "triton"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--arm", default="both", choices=["main", "new", "both"])
    args = ap.parse_args()

    why = preflight(args.device)
    if why is not None:
        return skip(why)

    arms = ("main", "new") if args.arm == "both" else (args.arm,)
    rows: dict[str, list[tuple]] = {}
    totals: dict[str, tuple[int, float]] = {}
    for arm in arms:
        model, opt = build(
            args.preset, state=args.state, ns=args.ns, ns_kind=args.ns_kind, device=args.device
        )
        rows[arm] = table(opt.muon, arm)
        if arm == "main":
            opt.muon.batched_step_from_model = lambda m, f=None, _o=opt: batched_step_main(  # type: ignore[method-assign]
                _o.muon, m, f
            )
        ms = wall_ms(opt, args.reps, args.warm)
        c = LaunchCounter()
        with c:
            opt.step()
        totals[arm] = (c.launches, ms)
        print(
            f"\n=== {arm}: {args.preset} b{args.batch} x s{args.seq}, "
            f"state={args.state} ns={args.ns} ns_kind={args.ns_kind} ==="
        )
        print(f"optimizer stage: {ms:8.2f} ms (min of {args.reps})  {c.line()}")
        del model, opt
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(
        f"\n{'shape':>14s} {'tensors':>8s} {'chunks':>7s} {'path':>8s} "
        f"{'main NS':>8s} {'new NS':>8s}  reason"
    )
    for i, (shape, n, chunks, path, reason, launches) in enumerate(rows[arms[0]]):
        new_launches = rows["new"][i][5] if "new" in rows else launches
        print(
            f"{str(shape):>14s} {n:8d} {chunks:7d} {path:>8s} "
            f"{launches:8d} {new_launches:8d}  {reason}"
        )

    print(f"\n{'arm':>5s} {'launches':>10s} {'min ms':>9s}")
    for arm in arms:
        launches, ms = totals[arm]
        print(f"{arm:>5s} {launches:10d} {ms:9.2f}")
    if len(arms) == 2:
        old, new = totals["main"][0], totals["new"][0]
        print(
            f"\nVERDICT: {old} -> {new} launches "
            f"({100.0 * (old - new) / max(old, 1):.1f}% down) on the optimizer stage"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
