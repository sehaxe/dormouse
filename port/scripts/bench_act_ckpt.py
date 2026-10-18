"""The bench for quantized saved-tensor checkpointing (`model/act_ckpt.py`).

    uv run python scripts/bench_act_ckpt.py

What it answers, in the order the module's claims are made:

1. **the codec's error bound** on two fixtures (`randn`, heavy-tailed
   `randn·e^{2·randn}`) — max abs err / absmax, rms err / rms x, and the
   fraction the 4-bit grid flushes to zero. These are the module docstring's
   table, reproduced from the same fixture construction (imported from
   `tests/test_act_ckpt.py`, so the two files cannot drift);
2. **the bytes saved** on a real tiny-preset forward+backward, from the
   counters: packed vs original vs the bf16-equivalent, and the ratios;
3. **the loss/grad difference vs hooks off** — the loss must be BIT-IDENTICAL
   (the hooks cannot touch the forward) and the worst relative
   parameter-gradient move is printed with the tensor that owns it;
4. **wall time both ways** (1 warmup + 3 timed, min and mean), so the codec's
   cost is visible next to the memory win.

CPU only, tiny model, seconds to run: no GPU, no card preflight (AGENTS §1.5 is
about device work and there is none here). The numbers are NOT transferable to
b1 — no device, no bf16 saves, a 46-parameter model — but the counter
arithmetic is per-element and does transfer. The weight-save share is printed
too, because "the backward reads re-quantized weights" is a claim about the
packed set and not a footnote.
"""

from __future__ import annotations

import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from conftest import tiny_cfg  # noqa: E402
from test_act_ckpt import _gauss, _heavy_tail  # noqa: E402

from dormouse.instrument import Counters  # noqa: E402
from dormouse.model import DormouseModel  # noqa: E402
from dormouse.model.act_ckpt import (  # noqa: E402
    Packed,
    QuantSavedHooks,
    memory_report,
    pack_tensor,
    unpack_tensor,
)

#: The tiny preset's own batch: 116 of the 383 saves clear the 4096-element
#: gate, so the arm is real. The codec is Python-side torch ops on CPU (~2.4x
#: the hooks-off step, printed below), which is what keeps this bench at
#: REPEATS=2 — the task's CI budget is 60 s and a bigger batch buys nothing but
#: wall clock here.
BATCH, SEQ, MIN_NUMEL = 2, 64, 4096
ARMS: tuple[str | None, ...] = (None, "mxfp4", "mxfp8")
REPEATS = 2


def codec_bounds() -> None:
    """(1) The module docstring's bound table, from the shared fixtures."""
    print("== codec bounds (CPU, fp32, seed 0, [64, 128]) ==")
    print(f"{'fixture':9s} {'kind':6s} {'max/absmax':>12s} {'rms/rms':>10s} {'zeros':>7s}")
    for name, t in (("randn", _gauss((64, 128))), ("heavy", _heavy_tail((64, 128)))):
        amax, rms = float(t.abs().max()), float(t.pow(2).mean().sqrt())
        for kind in ("mxfp4", "mxfp8"):
            dq = unpack_tensor(pack_tensor(t, kind))
            e = t - dq
            zeros = float((dq == 0).float().mean())
            print(
                f"{name:9s} {kind:6s} {float(e.abs().max()) / amax:12.4e} "
                f"{float(e.pow(2).mean().sqrt()) / rms:10.4e} {zeros:7.3f}"
            )
    print()


def _model_and_batch():
    """A fresh, identically seeded tiny model: arms differ in the hooks only."""
    torch.manual_seed(0)
    model = DormouseModel(tiny_cfg())
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    return model, ids, (ids + 1) % 251


def _step(model, ids, targets, kind, counters) -> tuple[torch.Tensor, dict, float]:
    """One forward+backward under `kind` (`None` = no hooks at all)."""
    model.zero_grad(set_to_none=True)
    ctx = (
        nullcontext()
        if kind is None
        else QuantSavedHooks(kind, min_numel=MIN_NUMEL, counters=counters)
    )
    t0 = time.perf_counter()
    with ctx:
        _, loss, _ = model(ids, targets)
    loss.backward()
    dt = time.perf_counter() - t0
    named = model.named_parameters()
    grads = {n: p.grad.detach().clone() for n, p in named if p.grad is not None}
    return loss.detach(), grads, dt


def _worst_rel(ref: dict, got: dict) -> tuple[float, str]:
    worst, where = 0.0, ""
    for name, g in ref.items():
        rel = float((g - got[name]).norm()) / max(float(g.norm()), 1e-12)
        if rel > worst:
            worst, where = rel, name
    return worst, where


def weight_share() -> None:
    """The packed set's parameter-storage share (the module docstring's 92/383).

    autograd saves a linear's `weight.t()` — a view of the parameter — so a
    packed save can BE a weight. `untyped_storage().data_ptr()` identifies the
    parameter storages; they stay alive for the whole run, so an activation
    cannot be aliased to one.
    """
    model, ids, targets = _model_and_batch()
    storages = {p.untyped_storage().data_ptr() for p in model.parameters()}
    seen = packed_weights = 0
    # `mxfp8`: the packed SET is the gate's (numel), not the kind's, and the
    # 8-bit arm's codec is the cheap one — this count is kind-independent.
    h = QuantSavedHooks("mxfp8", min_numel=MIN_NUMEL)
    op = h._pack

    def spy(t: object) -> object:
        nonlocal seen, packed_weights
        out = op(t)
        if isinstance(t, torch.Tensor):
            seen += 1
            if isinstance(out, Packed) and t.untyped_storage().data_ptr() in storages:
                packed_weights += 1
        return out

    h._pack = spy  # type: ignore[method-assign]
    with h:
        _, loss, _ = model(ids, targets)
    loss.backward()
    print("== weight saves ==")
    print(
        f"{packed_weights} of {h.counters['act_ckpt_packed']} PACKED saves are parameter "
        f"storage, out of {seen} saves seen (min_numel={MIN_NUMEL} leaves the small "
        f"factor views exact)"
    )
    print()


def main() -> int:
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  (CPU only)\n")
    codec_bounds()

    print(f"== tiny preset, batch {BATCH} x seq {SEQ}, min_numel={MIN_NUMEL} ==")
    print(
        f"{'arm':7s} {'loss':>12s} {'loss==off':>9s} {'worst grad':>11s} "
        f"{'@tensor':28s} {'ms min/mean':>13s}"
    )
    losses, grads, reports, times = {}, {}, {}, {}
    for kind in ARMS:
        model, ids, targets = _model_and_batch()
        _step(model, ids, targets, kind, Counters())  # warmup, discarded
        last: Counters = Counters()
        dt: list[float] = []
        for _ in range(REPEATS):
            last = Counters()
            loss, g, seconds = _step(model, ids, targets, kind, last)
            dt.append(seconds)
        losses[kind], grads[kind], reports[kind], times[kind] = loss, g, last, dt

    ref = grads[None]
    for kind in ARMS:
        worst, where = (0.0, "—") if kind is None else _worst_rel(ref, grads[kind])
        equal = "yes" if torch.equal(losses[kind], losses[None]) else "NO"
        print(
            f"{str(kind):7s} {float(losses[kind]):12.6f} {equal:>9s} {worst:11.3e} "
            f"{where:28s} {1e3 * min(times[kind]):6.1f}/{1e3 * statistics.mean(times[kind]):6.1f}"
        )
    print()

    print("== accounting (one run per arm, from the counters) ==")
    for kind in ARMS:
        r = memory_report(reports[kind])
        if kind is None:
            print(f"{kind!s:7s} no hooks: nothing counted")
            continue
        print(
            f"{kind:7s} packed {r['packed_tensors']:4.0f} tensors  "
            f"{r['packed_bytes']:9.0f} B  vs orig {r['orig_bytes']:9.0f} B "
            f"({r['ratio_orig']:.2f}x)  vs bf16 {r['bf16_bytes']:9.0f} B "
            f"({r['ratio_bf16']:.2f}x)"
        )
    print()

    weight_share()
    print("takeaway: the loss is bit-identical and the backward's inputs are re-quantized;")
    print("the counter arithmetic is the memory win, the grad column is its price.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
