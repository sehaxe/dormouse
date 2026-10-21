"""`torch.compile` retry on the medium model: dynamic=False vs dynamic=True.

    uv run --extra gpu python scripts/compile_retry.py

**Why a retry, and what was measured before.** `torch.compile(dynamic=True)` was
rejected at **0.20x** on the port: dynamo recompiled on `SparseMoE._banks`' cache
key (per-step Python values: the annealed `factor_qat_alpha`, the parameters'
`_version` tuple), spent `config.recompile_limit`, and fell back to eager at
**2939 ms/step against 573 eager** (`docs/research/2026-10-18-compile-moe.md`).
That key now sits behind `@torch.compiler.disable`, on `_banks` and on
`QuantCache.refresh_stacked`; this script re-measures the two settings the fix
makes legal again.

**What is measured, per arm** (`medium`, b8 x s512, the workload's own shape):

* **first-step ms** — the step that pays dynamo tracing + inductor compilation;
* **steady ms/step** — median of steps 2..20, with one `synchronize()` per step;
* **peak GiB** — `max_memory_allocated` over the arm, the compile's own cost;
* **dynamo's bookkeeping** — `graph_break` (breaks are legal), the `recompile`
  counter where this torch has it, and the `stats["unique_graphs"]` proxy where
  it does not (torch 2.13 has `frames`/`stats`/`graph_break`/`unimplemented`;
  `recompile` is absent, and `unique_graphs` IS the count of captured variants
  — one more per recompile). `unimplemented`'s `recompile_limit` hit is printed
  separately because it is the fallback that made the first attempt 0.20x.

**The gate.** A graph break is fine; a recompile storm is not. An arm is
**ILLEGAL** when it hits `recompile_limit` or captures more than
`RECOMPILE_FAIL = 20` graphs across the 20 steps, and the process exits 1. The
eager control is timed with the same step and the same starting weights.

**Protocol.** One parameter snapshot, one fresh model + `MixOptimizer` per arm,
the trainer's step (invalidate, fwd, bwd, clip, `step()`); 20 steps per arm with
no warmup before step 1, so the first step's number really includes the compile.
`dynamic=False` at medium may compile for minutes — that is a result, not a hang;
the first-step column is where it shows.

**Preflight and not-run.** `pgrep -f "dormouse trai[n]"` + `nvidia-smi` under
1 GiB, or a SKIP banner and exit 0. Written while the card is busy: **no arm in
this file has been run**, and no number here may be quoted until it is.
"""

from __future__ import annotations

import argparse
import functools
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from dormouse.config import load_run
from dormouse.model import DormouseModel
from dormouse.model.experts import invalidate_banks_all
from dormouse.model.qat_cache import invalidate_qat_cache_all
from dormouse.train.optim import MixOptimizer, clip_grads

# `scripts/` is not a package (a test loads this file by path): the lane guard is
# a sibling file, so its directory goes on the path, APPENDED not inserted (a
# script name at the front of `sys.path` would shadow a package of that name).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight as _lane_preflight  # noqa: E402
from _lane import skip  # noqa: E402

# a test pins a 0-arg `preflight`: this probe only ever measures the card.
preflight = functools.partial(_lane_preflight, "cuda")

PRESET = "medium"
BATCH, SEQ = 8, 512
STEPS = 20
DEV = "cuda"
#: A recompile STORM, not a graph break: > 20 captured variants over the 20
#: steps means dynamo is still re-tracing per step. The first attempt's failure
#: was a fallback at `recompile_limit`, counted separately and never legal.
RECOMPILE_FAIL = 20
LIMIT_KEY = "recompile_limit"


def dynamo_counts() -> dict[str, int]:
    """The counters this torch exposes, flattened to four comparable numbers.

    `recompile` exists in some versions and is a dict keyed by the frame's
    explanation; where it is absent (torch 2.13), `stats["unique_graphs"]` is the
    count of captured variants and moves by one per recompile.
    """
    c = torch._dynamo.utils.counters
    rec = dict(c.get("recompile", {}))
    return {
        "graph_break": sum(dict(c.get("graph_break", {})).values()),
        "recompile": sum(rec.values()),
        "graphs": int(dict(c.get("stats", {})).get("unique_graphs", 0)),
        "limit_hits": sum(v for k, v in dict(c.get("unimplemented", {})).items() if LIMIT_KEY in k),
    }


def counts_delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    """Per-arm deltas — dynamo's counters are process-global and never reset."""
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


def recompiles(counts: dict[str, int]) -> int:
    """The recompile count the gate reads: the counter if present, else graphs."""
    return counts["recompile"] if counts["recompile"] else counts["graphs"]


def legal(counts: dict[str, int]) -> bool:
    """A graph break is fine; a recompile storm or the eager fallback is not."""
    return counts["limit_hits"] == 0 and recompiles(counts) <= RECOMPILE_FAIL


def _model(run, state: dict | None) -> DormouseModel:
    model = DormouseModel(run.model, device=DEV)
    if state:
        model.load_state_dict(state)
    return model


def _optimizer(model: DormouseModel, train) -> MixOptimizer:
    return MixOptimizer(
        model,
        lr=train.lr,
        wd=train.wd,
        opt=train.opt,
        ns_steps=train.ns_steps,
        ns_kind=train.ns_kind,
        colrow_3d=train.muon_colrow_3d,
        muon_state_dtype=train.muon_state_dtype,
        ns_dtype=train.ns_dtype,
        muon_variant=train.muon_variant,
        muon_sf=train.muon_sf,
        muon_sf_wd=train.muon_sf_wd,
    )


def _step(model, opt, train, x: torch.Tensor) -> None:
    """The trainer's step, identical in every arm (the optimizer object differs)."""
    model.zero_grad(set_to_none=True)
    invalidate_qat_cache_all(model)
    invalidate_banks_all(model)
    _, loss, _ = model(x, x)
    loss.backward()
    if train.grad_clip > 0:
        clip_grads(model.parameters(), train.grad_clip)
    opt.step()
    invalidate_banks_all(model)
    invalidate_qat_cache_all(model)


def run_arm(name: str, run, state: dict, steps: int) -> dict:
    """One arm: `(first_ms, steady_ms, peak_gib, counters)` or an ERROR row."""
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    model = _model(run, state)
    opt = _optimizer(model, run.train)
    fwd = model
    if name != "eager":
        torch._dynamo.reset()
        fwd = torch.compile(model, dynamic=name == "dynamic")
    x = torch.randint(0, 256, (BATCH, SEQ), device=DEV)
    before = dynamo_counts()
    times: list[float] = []
    try:
        for _ in range(steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _step(fwd, opt, run.train, x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
    except Exception as e:  # noqa: BLE001 - a compile failure is a RESULT of this bench
        peak = torch.cuda.max_memory_allocated() / 2**30
        del model, opt
        torch.cuda.empty_cache()
        return {"arm": name, "error": f"{type(e).__name__}: {e}", "peak": peak}
    after = dynamo_counts()
    row = {
        "arm": name,
        "first": times[0],
        "steady": statistics.median(times[1:]) if len(times) > 1 else times[0],
        "peak": torch.cuda.max_memory_allocated() / 2**30,
        "counts": counts_delta(after, before),
    }
    del model, opt
    torch.cuda.empty_cache()
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=STEPS, help="steps per arm (20)")
    args = ap.parse_args(argv)
    why = preflight()
    if why is None:
        # This probe's own ceiling is 1 GiB, tighter than the lane guard's 2.0:
        # a card already holding a second model is contended however idle it looks.
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if smi and int(smi.split()[0]) >= 1024:
            why = f"the card already holds {smi.split()[0]} MiB (this probe's ceiling is 1024)"
    if why is not None:
        return skip(why)

    run = load_run(PRESET)
    torch.manual_seed(0)
    base = _model(run, None)
    state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    del base
    torch.cuda.empty_cache()

    print(
        f"\nshape: {PRESET} d={run.model.d_model} E={run.model.n_experts} "
        f"b{BATCH} x s{SEQ}; {args.steps} steps per arm, no warmup (step 1 pays the compile)"
    )
    rows = [run_arm(name, run, state, args.steps) for name in ("eager", "dynamic", "static")]
    print(
        f"\n{'arm':9s} {'first ms':>10s} {'steady ms':>10s} {'peak GiB':>9s}"
        f" {'graph_break':>12s} {'recompile':>10s} {'graphs':>7s} {'limit':>6s}  verdict"
    )
    control = next((r for r in rows if r["arm"] == "eager"), None)
    verdicts: dict[str, str] = {}
    for row in rows:
        if "error" in row:
            print(f"{row['arm']:9s} {'—':>10s} {'—':>10s} {row['peak']:9.2f}  ERROR {row['error']}")
            verdicts[row["arm"]] = "ERROR"
            continue
        counts = row["counts"]
        ok = row["arm"] == "eager" or legal(counts)
        verdict = "control" if row["arm"] == "eager" else ("LEGAL" if ok else "ILLEGAL")
        verdicts[row["arm"]] = verdict
        speed = (
            f"  x{control['steady'] / row['steady']:.2f} vs eager"
            if control and "steady" in control
            else ""
        )
        print(
            f"{row['arm']:9s} {row['first']:10.1f} {row['steady']:10.1f} {row['peak']:9.2f}"
            f" {counts['graph_break']:12d} {counts['recompile']:10d} {counts['graphs']:7d}"
            f" {counts['limit_hits']:6d}  {verdict}{speed}"
        )

    print(
        "\nA graph break is fine; a recompile storm is not. `recompile` is 0 on "
        "torch 2.13\n(the counter does not exist), so `graphs` is the proxy: one more "
        "captured\nvariant per recompile. The gate is limit_hits == 0 and "
        f"recompiles <= {RECOMPILE_FAIL}; the eager control is not gated."
    )
    print(
        "\nWhich setting is legal: "
        + ", ".join(f"{k}={v}" for k, v in verdicts.items() if k != "eager")
        + ". A first-step number is the compile's price and is not a steady state."
    )
    return 0 if all(v in ("control", "LEGAL") for v in verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
