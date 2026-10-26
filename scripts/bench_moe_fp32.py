"""The fp32 MoE expert pass at the ByteFlow shape: padded / triton / naive.

    uv run --extra gpu python scripts/bench_moe_fp32.py

The question is the one `docs/reviews/2026-10-21-battery.md` §5 left open: the
ByteFlow channel ran fp32 when these readings were taken (the net had no bf16
path then; it has one since 2026-10-21, `byteflow.py`'s precision rule — this
bench prices the fp32 lane the record was taken on), and at fp32 the shipped Triton
grouped GEMM asked for **196608 B** of shared memory against this card's
**101376 B** — so that arm ran `moe_kernel=naive`, 37x slower than the grouped
GEMM. `moe_kernels.tiles_for` now halves the wide tiles and runs one pipeline
stage at four bytes; this bench measures whether that is legal and what it buys.

The shape is the channel's loop: b=4 x T=512 bytes compressed to K=128 patch
positions (d=1024, f=2730, rank 64, E=256, topk 2; `medium`'s geometry, i.e. the
ByteFlow arm that did not run). All three arms see the SAME rows and the SAME
drops (`moe_capacity = 2.0`, the cap the padded arm needs; the byteflow recipe
itself runs dropless, so this bench prices the arm at the capacity floor).

Arms, interleaved, forward only under `no_grad` (the dispatch is what the
fallback decision is about):

* **padded** — `_padded_dispatch`, the torch `bmm` bank;
* **triton** — `_triton_dispatch(dropless=False)`, the new fp32 tiles;
* **naive** — `_naive_dispatch`, the reference loop the arm used to run.

It prints ms (median of `REPS`, one warm-up excluded), the shared-memory
estimate of each arm, and the triton-vs-padded parity. The estimate is the
module's own (`moe_kernels.smem_estimate`), calibrated on the measured refusal:
`196608 B` at the old `BLOCK_N=128`/two stages.

Preflight (AGENTS §1.5): no `dormouse` training process alive and the card
holding < 1 GiB. Otherwise it prints a SKIP banner naming what it found and
exits 0 — nothing is measured, so there is no number to quote.
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
#: The ByteFlow channel's loop shape: 4 x 512 bytes -> 128 patches each.
B, T, K = 4, 512, 128
TOKENS = B * K  # the loop's rows, 512
D, F_DIM, RANK, E = 1024, 2730, 64, 256
TOPK, CAPACITY = 2, 2.0
REPS = 10


def _timed(fn) -> float:
    """Median ms over `REPS` calls (one warm-up excluded)."""
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


def main() -> int:
    busy = subprocess.run(
        ["pgrep", "-af", "[d]ormouse (train|rl|sft|bench|serve)|[/]target/release/train"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    why = preflight("cuda") or (f"a dormouse process is alive:\n{busy}" if busy else None)
    if why is None:
        # This bench's own ceiling is 1 GiB, tighter than the lane guard's 2.0:
        # a card already holding a second model is contended however idle it looks.
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if smi and int(smi.split()[0]) >= 1024:
            why = f"the card already holds {smi.split()[0]} MiB (this bench's ceiling is 1024)"
    if why is not None:
        return skip(why)
    torch.manual_seed(11)
    cfg = ModelConfig(
        d_model=D,
        n_heads=16,
        head_dim=D // 16,
        d_ffn=F_DIM,
        rank=RANK,
        n_experts=E,
        moe_topk=TOPK,
        moe_capacity=CAPACITY,
        bf16=False,  # the fp32 lane these readings are about (the net also runs bf16 now)
        use_engram=False,
    )
    moe = SparseMoE(cfg, device=DEV)
    x = torch.randn(TOKENS, D, device=DEV)
    with torch.no_grad():
        probs = F.softmax(moe.router(x), dim=-1)
        weight, index = probs.topk(TOPK, dim=-1)
        weight = weight / weight.sum(-1, keepdim=True)
    order, keep, dropped = moe._permute(index)
    cap = moe._capacity(*index.shape)
    tiles = mk.tiles_for(torch.float32, D, F_DIM, RANK)
    smem = mk.smem_estimate(tiles, torch.float32, RANK)
    old = mk.smem_estimate(mk.TILES, torch.float32, RANK, stages=2)
    print(
        f"\n[shape] E={E} d={D} r={RANK} f={F_DIM} tokens={TOKENS} rows={int(keep.sum())} "
        f"dropped={int(dropped)} cap={cap}",
        flush=True,
    )
    print(
        f"[tiles] fp32 {tiles} stages={mk._launch_stages(torch.float32)} -> "
        f"est {smem} B <= {mk.SMEM_CEILING_BYTES} B "
        f"(old {mk.TILES} at 2 stages was {old} B, refused)",
        flush=True,
    )
    with torch.no_grad():
        padded = moe._padded_dispatch(x, weight, index, order, keep)
        triton = moe._triton_dispatch(x, weight, index, order, keep)
    rel = float((triton.float() - padded.float()).abs().max()) / float(padded.float().abs().max())
    print(f"[parity] triton vs padded max_rel={rel:.3e}", flush=True)

    arms = {
        "padded": (
            lambda: moe._padded_dispatch(x, weight, index, order, keep),
            0,
        ),
        "triton fp32": (
            lambda: moe._triton_dispatch(x, weight, index, order, keep),
            smem,
        ),
        "naive": (
            lambda: moe._naive_dispatch(x, weight, index, order, keep, Counters()),
            0,
        ),
    }
    print(f"\n{'arm':<14}{'ms':>10}{'smem est B':>14}{'vs padded':>12}", flush=True)
    with torch.no_grad():
        base = _timed(arms["padded"][0])
        for name, (fn, sm) in arms.items():
            ms = _timed(fn)
            print(f"{name:<14}{ms:>10.3f}{sm:>14}{base / ms:>11.2f}x", flush=True)
    if rel > 1e-5:
        print(f"[WARN] parity {rel:.3e} > 1e-5: the fast arm is not the same function")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
