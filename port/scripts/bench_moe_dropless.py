"""The b1-shape bench for the DROPLESS expert pass: grouped GEMM vs the loop.

    uv run --extra gpu python scripts/bench_moe_dropless.py

The question this answers, and it is a cost question, not a correctness one:
`moe_capacity = 0` keeps every routing assignment, so the row-block grid takes a
STATIC bound (`moe_kernels._dropless_cap`, `R` rows in one group) and the kernels
exit on-device when a block is empty. Before that early-exit the dropless route
had to be the per-expert Python loop — and at `b1`'s E=1024 that is ~1024 tiny
matmuls per expert pass. Three arms, same rows, same weights, interleaved:

* **dropless triton** — `_triton_dispatch(dropless=True)`, the static bound, in
  its two backward flavours: `NEW` is the default fused adjoint (one combine
  kernel, one `dx` GEMM with two accumulators, the permute's own int32 row map,
  a grow-only workspace) and `CURRENT` is `fused_bwd=False` (the previous
  adjoint: two `dx` GEMMs plus `add_`, `repeat_interleave`, autograd's combine
  backward). The forward is the same code on both;
* **naive ragged** — `_naive_dispatch`, the reference loop (the arm dropless
  used to be forced onto);
* **padded cap** — `_padded_dispatch` with the capacity rule's own cap, as the
  floor the other two are measured against (`b1` runs `moe_capacity = 2.0`, so
  that cap is what a capacity'd b1 step pays today).

fwd and fwd+bwd, median of 5, peak allocated VRAM per arm, plus the parity of the
dropless arm against the loop (a bench that measures a wrong function is worse
than no bench).

Preflight (AGENTS §1.5): no `dormouse` process alive and the card holding
< 2 GiB. Otherwise it prints a SKIP banner naming what it found and exits 0 —
nothing is measured, so there is no number to quote.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from dormouse.config import ModelConfig
from dormouse.instrument import Counters
from dormouse.model import moe_kernels as mk
from dormouse.model.experts import SparseMoE

# `scripts/` is not a package (a test loads this file by path): the lane guard is
# a sibling file, so its directory goes on the path, APPENDED not inserted (a
# script name at the front of `sys.path` would shadow a package of that name).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight, skip  # noqa: E402

DEV = "cuda"
#: b1's expert geometry (`configs/b1.toml`): E 1024, d 2048, r 64, f 5461, topk 2.
E, D, RANK, F_DIM = 1024, 2048, 64, 5461
TOKENS = 8192  # x rows -> 16 384 routed rows at topk 2 (the task's shape)
REPS = 5
BM = mk.TILES["BLOCK_M"]


def _timed(fn, reps: int = REPS) -> tuple[float, float, float]:
    """`(median ms, min ms, peak GiB)` over `reps` calls, one warm-up excluded."""
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return (
        statistics.median(times) * 1e3,
        min(times) * 1e3,
        torch.cuda.max_memory_allocated() / 2**30,
    )


def main() -> int:
    busy = subprocess.run(
        ["pgrep", "-af", "dormouse sf[t]"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    why = preflight("cuda") or (f"a dormouse process is alive:\n{busy}" if busy else None)
    if why is not None:
        return skip(why)
    torch.manual_seed(11)
    cfg = ModelConfig(
        d_model=D,
        n_heads=4,
        head_dim=D // 4,
        d_ffn=F_DIM,
        rank=RANK,
        n_experts=E,
        moe_topk=2,
        moe_capacity=2.0,  # b1's own value: the padded arm's floor
        bf16=True,
    )
    moe = SparseMoE(cfg, device=DEV)
    x = torch.randn(TOKENS, D, device=DEV)
    with torch.no_grad():
        probs = F.softmax(moe.router(x), dim=-1)
        weight, index = probs.topk(2, dim=-1)
        weight = weight / weight.sum(-1, keepdim=True)
    order, keep, dropped = moe._permute(index)
    rows = int(keep.sum())
    counts = (torch.bincount(index.reshape(-1), minlength=E)).cpu()
    cap = moe._capacity(*index.shape)
    print(
        f"\n[shape] E={E} d={D} r={RANK} f={F_DIM} tokens={TOKENS} rows={rows} "
        f"dropped={int(dropped)} cap={cap} tiles={mk.TILES}",
        flush=True,
    )
    print(
        f"[grid] dropless bound = R = {rows} rows -> {(-(-rows // BM))} row-blocks/expert "
        f"({(-(-rows // BM)) * E} programs); blocks with work = "
        f"{sum(-(-int(c) // BM) for c in counts)}; max group = {int(counts.max())} rows; "
        f"padded cap {cap} -> {(-(-cap // BM)) * E} programs",
        flush=True,
    )

    with torch.no_grad():
        dropless = moe._triton_dispatch(x, weight, index, order, keep, dropless=True)
        dropless_cur = moe._triton_dispatch(
            x, weight, index, order, keep, dropless=True, fused_bwd=False
        )
        naive = moe._naive_dispatch(x, weight, index, order, keep, Counters())
        padded = moe._padded_dispatch(x, weight, index, order, keep)
    rel_n = float((dropless.float() - naive.float()).abs().max()) / float(naive.float().abs().max())
    rel_p = float((dropless.float() - padded.float()).abs().max()) / float(
        padded.float().abs().max()
    )
    rel_b = float((dropless.float() - dropless_cur.float()).abs().max()) / float(
        dropless_cur.float().abs().max()
    )
    print(
        f"[parity] dropless-triton vs naive max_rel={rel_n:.3e}; vs padded(cap {cap}) "
        f"max_rel={rel_p:.3e}; NEW-vs-CURRENT backward fwd max_rel={rel_b:.3e}",
        flush=True,
    )
    del dropless, dropless_cur, naive, padded
    torch.cuda.empty_cache()

    xp = x.clone().requires_grad_(True)
    xt = x.clone().requires_grad_(True)
    xtc = x.clone().requires_grad_(True)
    xn = x.clone().requires_grad_(True)

    def fwd_t():
        with torch.no_grad():
            moe._triton_dispatch(x, weight, index, order, keep, dropless=True)

    def fwd_n():
        with torch.no_grad():
            moe._naive_dispatch(x, weight, index, order, keep, Counters())

    def fwd_p():
        with torch.no_grad():
            moe._padded_dispatch(x, weight, index, order, keep)

    def both_t():
        moe.zero_grad(set_to_none=True)
        moe._triton_dispatch(xt, weight, index, order, keep, dropless=True).float().pow(
            2
        ).sum().backward()

    def both_tc():
        moe.zero_grad(set_to_none=True)
        out = moe._triton_dispatch(xtc, weight, index, order, keep, dropless=True, fused_bwd=False)
        out.float().pow(2).sum().backward()

    def both_n():
        moe.zero_grad(set_to_none=True)
        out = moe._naive_dispatch(xn, weight, index, order, keep, Counters())
        out.float().pow(2).sum().backward()

    def both_p():
        moe.zero_grad(set_to_none=True)
        moe._padded_dispatch(xp, weight, index, order, keep).float().pow(2).sum().backward()

    arms = {
        "dropless triton fwd": fwd_t,
        "naive ragged fwd": fwd_n,
        "padded cap fwd": fwd_p,
        "dropless triton NEW fwd+bwd": both_t,
        "dropless triton CURRENT fwd+bwd": both_tc,
        "naive ragged fwd+bwd": both_n,
        "padded cap fwd+bwd": both_p,
    }
    acc: dict[str, list] = {k: [] for k in arms}
    # Interleaved (this card's dispatch timings drift a few ms between runs of one
    # binary, so one-arm-then-the-other attributes the drift to the arm).
    for _ in range(REPS):
        for name, fn in arms.items():
            acc[name].append(_timed(fn, reps=1))
    print(f"\n{'arm':>24} {'median ms':>10} {'min ms':>9} {'peak GiB':>9}")
    for name, rows_ in acc.items():
        med = statistics.median(r[0] for r in rows_)
        lo = min(r[1] for r in rows_)
        peak = max(r[2] for r in rows_)
        print(f"{name:>24} {med:>10.3f} {lo:>9.3f} {peak:>9.2f}")
    base = statistics.median(r[0] for r in acc["dropless triton fwd"])
    loop = statistics.median(r[0] for r in acc["naive ragged fwd"])
    new_b = statistics.median(r[0] for r in acc["dropless triton NEW fwd+bwd"])
    cur_b = statistics.median(r[0] for r in acc["dropless triton CURRENT fwd+bwd"])
    print(
        f"\nthe loop is {loop / base:.1f}x the dropless grouped GEMM's forward at this "
        f"shape; the static bound launches {(-(-rows // BM)) * E} programs of which "
        f"{sum(-(-int(c) // BM) for c in counts)} have work",
        flush=True,
    )
    print(
        f"the fused backward's fwd+bwd is {new_b:.1f} ms against the previous adjoint's "
        f"{cur_b:.1f} ms ({cur_b / new_b:.2f}x) at the same forward",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
