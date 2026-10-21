"""The adaptive-width Pareto: quality per expert FLOP on ONE checkpoint.

    uv run python scripts/bench_topk.py --ckpt-name <name> --out-dir <dir>

**The question.** `moe_topk_mode` (`model/adaptive_topk.py`) spends a per-token
width instead of one global `moe_topk`. It is only worth shipping if, at the
SAME average `k`, it scores better than always running that `k` — a policy that
cannot beat `always-k` at average `k` is worthless, and this script decides that
with numbers instead of an opinion.

**Arms, on ONE eval window and ONE checkpoint:**

* `fixed k` — the control, `moe_topk=k` with the policy off (`k` in `--ks`);
* `entropy` — `moe_topk_min=1`, `moe_topk_max=max(--ks)`, and a confidence
  window TUNED PER CONTROL so the measured average `k` lands on that control's
  `k`. The window is selected by `|avg_k − target|` ONLY (ties prefer the
  narrower window); the BPB of the selected window is then read off the same
  pass. Selection never looks at BPB — that would be cherry-picking.

**What a row prints:** BPB (mean CE / ln 2, the trainer's conversion), measured
average `k`, **expert FLOPs per byte**, and **ms/step**. The FLOPs column is
the arithmetic the average width buys: `max_iter · avg_k · (4·d·r + 2·r·f)`
per token (= per byte: the model is byte-level). The shared expert and the
router are in every arm and are therefore excluded — the delta between rows is
the routed pass, which is the thing under test.

**Same window, every arm:** the eval batches are collected ONCE and replayed to
every arm, so no arm scores a different amount of text (AGENTS §2.6: a BPB is
only comparable within one window). The window in bytes is printed once.

**ms/step is a FORWARD-ONLY eval step**, median over the batches, CUDA
synchronised per batch. It is the inference policy's cost. A training step is
NOT measured here (and `moe_kernel=padded`'s bank is sized from `k_max`, so on
that arm the average width does not shrink the bmm — the ragged arms and the
grouped GEMM's `offs` are where the FLOPs column becomes wall-clock; read the
ms column with that in mind, it is the honest measured side of the table).

**Preflight:** card free (< 2 GiB, no `dormouse` process) or the run refuses
loudly — one GPU thing at a time (AGENTS §1.5). `--device cpu` bypasses the
card checks and measures the host: it validates the driver, it does not price
the card.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dormouse.config import MOE_TOPK_SLOTS_MAX, load_run  # noqa: E402
from dormouse.instrument import Counters, loud  # noqa: E402
from dormouse.model.experts import SparseMoE  # noqa: E402
from dormouse.model.model import DormouseModel  # noqa: E402
from dormouse.train.data import open_streams  # noqa: E402

# `scripts/` is not a package (a test loads this file by path): the lane guard is
# a sibling file, so its directory goes on the path, APPENDED not inserted (a
# script name at the front of `sys.path` would shadow a package of that name).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight, skip  # noqa: E402

LN2 = math.log(2.0)


def windows() -> list[tuple[float, float]]:
    """Candidate `(entropy_lo, entropy_hi)` windows, including the two ends.

    `(-1, 0)` clamps every token to `k_max` (h >= 0 always) and `(1, 2)` to
    `k_min` (h <= 1 always): those are the control-equivalence arms, and they
    are what a target of `k_max` / `k_min` selects when nothing else can.
    """
    out = [(-1.0, 0.0), (1.0, 2.0)]
    for lo in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        for delta in (0.1, 0.2, 0.3, 0.4, 0.5):
            if lo + delta <= 1.0:
                out.append((lo, lo + delta))
    return out


def expert_flops_per_byte(cfg, avg_k: float) -> float:
    """`max_iter · avg_k · (4·d·r + 2·r·f)` — the routed pass, per token (=byte)."""
    per_expert = 4 * cfg.d_model * cfg.rank + 2 * cfg.rank * cfg.d_ffn
    return float(cfg.max_iter) * avg_k * per_expert


def _moes(model: DormouseModel) -> list[SparseMoE]:
    return [m for m in model.modules() if isinstance(m, SparseMoE)]


def _avg_k(counters: Counters) -> float | None:
    """The mean width from the `moe_k{i}` histogram, or None on the fixed arms."""
    hist = {i: counters.get(f"moe_k{i}", 0) for i in range(1, MOE_TOPK_SLOTS_MAX + 1)}
    tokens = sum(hist.values())
    if tokens == 0:
        return None
    return sum(i * n for i, n in hist.items()) / tokens


def _measure(model, batches, dev: str, k_hint: float) -> tuple[float, float, float, Counters]:
    """`(BPB, mean width, median ms/step, counters)` on the SAME batch list.

    `k_hint` is the width the fixed arm was BUILT with: the policy is off there,
    so it emits no histogram and the width is the config, not a measurement.
    """
    counters = Counters()
    ce = 0.0
    times: list[float] = []
    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(dev), y.to(dev)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _logits, loss, _ = model(x, targets=y, counters=counters)
            if dev == "cuda":
                torch.cuda.synchronize()
            times.append(1e3 * (time.perf_counter() - t0))
            ce += float(loss)
    counters.drain()
    measured = _avg_k(counters)
    avg_k = float(k_hint) if measured is None else measured
    return ce / len(batches) / LN2, avg_k, statistics.median(times), counters


def _build(run, state, dev: str, **over) -> DormouseModel:
    model = DormouseModel(dataclasses.replace(run.model, **over), device=dev).eval()
    model.load_state_dict(state.get("model", state), strict=False)
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt-name", required=True)
    ap.add_argument("--out-dir", default="checkpoints/port")
    ap.add_argument("--device", default="cuda", help="cuda (the measurement) or cpu (driver smoke)")
    ap.add_argument("--batches", type=int, default=20, help="eval batches, all arms identical")
    ap.add_argument("--ks", default="1,2,4", help="the fixed-width controls")
    args = ap.parse_args()

    why = preflight(args.device)
    if why is not None:
        return skip(why)
    ks = [int(k) for k in args.ks.split(",")]
    if len(set(ks)) != len(ks) or any(k < 1 for k in ks):
        loud(f"--ks={args.ks!r}: distinct positive widths only. Escape: --ks 1,2,4.")
    run = load_run(args.ckpt_name, out_dir=args.out_dir)
    n_exp = run.model.n_experts
    widest = max(ks)
    if widest > n_exp:
        loud(
            f"--ks asks for k={widest} but the checkpoint has n_experts={n_exp}. "
            f"Escape: --ks with widths <= {n_exp}."
        )
    if widest > MOE_TOPK_SLOTS_MAX:
        loud(f"--ks k={widest} > {MOE_TOPK_SLOTS_MAX} (the instrument's histogram stops there).")
    dev = args.device
    state = torch.load(
        os.path.join(run.train.out_dir, f"{args.ckpt_name}.pt"),
        map_location=dev,
        weights_only=False,
    )
    _, ev = open_streams(run)
    if ev is None:
        loud(
            f"the run has no eval stream (train.eval={run.train.eval}, "
            f"eval_data={run.train.eval_data!r}): a Pareto needs held-out text. "
            f"Escape: a run whose train.eval is true (or an eval_data file)."
        )
    ev.rewind()
    batches = [ev.next_batch() for _ in range(args.batches)]
    window = args.batches * run.train.batch * run.train.seq_len
    print(
        f"{args.batches} eval batches x batch {run.train.batch} x seq {run.train.seq_len} "
        f"= {window} B scored on EVERY arm; device={dev}",
        flush=True,
    )

    rows = []
    for k in ks:
        model = _build(run, state, dev, moe_topk=k, moe_topk_mode="off")
        bpb, avg_k, ms, c = _measure(model, batches, dev, k)
        rows.append((f"fixed k={k}", f"moe_topk={k}", avg_k, bpb, ms, c))
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    # ONE adaptive model, the window flipped between the calibration passes so
    # the arm is identical up to the two floats under test.
    model = _build(
        run,
        state,
        dev,
        moe_topk=widest,
        moe_topk_mode="entropy",
        moe_topk_min=1,
        moe_topk_max=widest,
    )
    first = batches[:1]
    for target in ks:
        best = None
        for lo, hi in windows():
            for moe in _moes(model):
                moe.entropy_lo, moe.entropy_hi = lo, hi
            counters = Counters()
            with torch.no_grad():
                x, y = first[0]
                model(x.to(dev), targets=y.to(dev), counters=counters)
            counters.drain()
            avg_k = _avg_k(counters)
            if avg_k is None:
                loud("the calibration forward emitted no moe_k histogram: mode is not on")
            key = (abs(avg_k - target), hi - lo)
            if best is None or key < best[0]:
                best = (key, lo, hi)
        _key, lo, hi = best
        for moe in _moes(model):
            moe.entropy_lo, moe.entropy_hi = lo, hi
        bpb, avg_k, ms, c = _measure(model, batches, dev, widest)
        rows.append((f"entropy -> k~{target}", f"lo={lo:g} hi={hi:g}", avg_k, bpb, ms, c))

    print("\narm                   window          avg_k    BPB   expert GFLOPs/B   ms/step")
    for name, window_s, avg_k, bpb, ms, _c in rows:
        gflops = expert_flops_per_byte(run.model, avg_k) / 1e9
        print(f"{name:21s} {window_s:15s} {avg_k:5.2f}  {bpb:6.4f}  {gflops:12.3f}  {ms:8.2f}")
    print("\nhistograms (moe_k1..):")
    for name, _w, _avg, _bpb, _ms, c in rows:
        if c.get("moe_k1") is None:
            print(f"  {name:21s} (fixed width: the policy is off, no histogram)")
            continue
        cells = " ".join(f"k{i}={c.get(f'moe_k{i}', 0)}" for i in range(1, widest + 1))
        print(f"  {name:21s} {cells}")
    print(
        "\nfwd-only eval step; expert FLOPs exclude the shared expert and the router "
        "(identical in every arm). A policy is useful only if, at matched avg_k, its "
        "BPB is BELOW the fixed arm's."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
