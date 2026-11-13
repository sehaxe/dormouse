#!/usr/bin/env -S uv run --python /home/sehaxe/dormouse/port/.venv/bin/python --script
"""Eager vs `torch.compile`: ms/step, B/s, TFLOP/s, MFU, VRAM, graph breaks.

WHY THIS MEASURES RATE, NOT KERNELS. The card's measured bf16 peak is 47.7
TFLOP/s and the flagship's own bill is 137.42 MFLOP/byte (FlopCounterMode, one
fwd+bwd at b1 x s1024 — `docs/records/reviews/2026-11-06-control-transformer.md` §1).
At b8 x s1024 x grad_accum 3 that is 3.377 TFLOP per optimizer step, so 8.5
TFLOP/s measured is 17.8 % MFU and the remaining 82 % is this lane's target.
Compile attacks exactly that: one fused inductor graph per block instead of the
eager pointwise chain (the 2026-11-05 census prices a KDA iteration's 40
pointwise launches at 2.20 ms of pure bandwidth).

THE ARMS (`--arms`, default = the four the brief names):

    eager      the model as the trainer builds it — no wrapper
    default    torch.compile(model, mode="default")
    max-autotune   torch.compile(..., mode="max-autotune-no-cudagraphs")
    trainer    torch.compile(model, dynamic=True) — `train/loop.py:1383`'s own
               wrapper, so the table prices the recipe the run would use
    strict     torch.compile(model, fullgraph=True) — opt-in; a break is an
               ERROR, which is the only way to prove a shape has none

EVERY compiled arm reports graph breaks, and a break is never silent: the
`explain` table names `file:line func` + the source line, and every row's
`breaks` column is the runtime `torch._dynamo.utils.counters["graph_break"]`
delta for that arm (a `torch.compiler.disable`d call is a COUNTED break, not a
defect; `fallbk` is the recompile-limit counter, i.e. dynamo gave up and ran
eager).

PROTOCOL. Per arm: a FRESH model reloaded from the same CPU `state_dict` (no
compiled artefact from the previous arm can be reused), `--warm` untimed steps,
then `--reps` timed runs of `--steps` steps each; the table prints the MINIMUM
ms/step, because the minimum is the number that must move. `torch._dynamo.reset()`
runs before every arm: inductor's on-disk cache still serves the kernels, so the
reset costs the dynamo/aot step, not the autotune.

PARITY (`--parity`, on by default). Same seed, same batch, same initial weights:
(1) logits and loss vs eager within 1e-4 relative — the compiled graph must BE
the eager program; (2) ONE optimizer step from those grads moves every parameter
within 1e-6 relative of the eager step. A failed parity is a LOUD error (exit 4),
never a warning: a faster arm that trains a different model is not an arm.

VRAM (`--batch-scale`). `torch.cuda.max_memory_allocated` is read per arm; the
projection to a bigger batch is `steady + (peak - steady) * b / b0`, where
`steady` is the same arm's post-step allocation (weights + optimizer state +
grads) — activations are the linear part. If the projection exceeds
`--vram-ceiling-gib` (15.0) the run REFUSES (exit 3) and prints the projection,
the floor and the slope instead of letting the card OOM.

BYTES/STEP. `grad_accum = max(1, round(--bytes-per-step / (b * s)))`. At s=1024
only b=8 can hold 24 576 B/step (g=3); b=16 and b=32 land on 32 768 B/step
because 24 576 / 16 384 = 1.5 micro-batches does not exist. Every row prints its
own B/step so the B/s column is read against it — a bigger step amortises the
optimizer over more bytes and would otherwise look like free throughput.

ONE GPU PROCESS AT A TIME (AGENTS 1.5). A compute process on the card is exit 2
with its pid named — not the usual exit-0 SKIP, because a benchmark that cannot
tell "busy" from "done" is how two runs collide.

    # during a checkpoint gap, ONE GPU process
    cd /home/sehaxe/dormouse/port
    uv run --extra gpu python scripts/bench_compile.py --preset flagship2 \\
        --snapshot /home/sehaxe/dormouse/checkpoints/flagship2/flagship2.config.toml
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch

from dormouse.config import ModelConfig, RunConfig, load_run
from dormouse.instrument import LoudError
from dormouse.train.optim import MixOptimizer

sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

#: The card's MEASURED bf16 peak (`docs/records/research/2026-11-05-small-model-mfu.md`
#: §0.3: the 5060 Ti purchase report's 45.7 and this box's 47.7). The MFU
#: denominator: a different number is a different claim, so it is a constant
#: here and pinned by `tests/test_bench_compile.py`.
PEAK_TFLOPS = 47.7

#: The flagship's bill, FlopCounterMode on `checkpoints/flagship2/flagship2.pt`
#: at b1 x s1024 fwd+bwd. The compute match every candidate is sized to.
FLAGSHIP_FLOPS_PER_BYTE = 137.42e6

#: The one shape the brief pins: 8 x 1024 x 3 = 24 576 bytes per optimizer step.
FLAGSHIP_BYTES_PER_STEP = 24_576

#: Refuse above this: the card is 16 GiB and the desktop + CUDA context hold
#: ~0.7-1.0 GiB, so 15.0 is the last budget a trainer may claim.
VRAM_CEILING_GIB = 15.0

EXIT_BUSY = 2
EXIT_VRAM = 3
EXIT_PARITY = 4
EXIT_STRICT = 5


# --- the arithmetic (pure, and the CPU suite's gate) ------------------------


def bytes_per_step(batch: int, seq: int, grad_accum: int) -> int:
    """Bytes one OPTIMIZER step consumes: the micro-batch times the accumulation."""
    return batch * seq * grad_accum


def step_flops(flops_per_byte: float, nbytes: int) -> float:
    """FLOPs one optimizer step spends (fwd+bwd), from the measured bill."""
    return flops_per_byte * nbytes


def step_tflops(flops_per_byte: float, nbytes: int, ms: float) -> float:
    """TFLOP/s implied by one step of `ms` milliseconds."""
    return step_flops(flops_per_byte, nbytes) / (ms * 1e-3) / 1e12


def mfu(tflops: float, peak: float = PEAK_TFLOPS) -> float:
    """Model-FLOPs utilisation: the fraction of the card's measured bf16 peak.

    Both arguments are TFLOP/s, so the peak's provenance is the ONLY thing that
    decides the number — `tests/test_bench_compile.py` pins 47.7 and the
    flagship shape's 8.51 TFLOP/s -> 17.8 %.
    """
    return tflops / peak


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """`max |a - b| / max(|b|, tiny)` — the parity metric, one number per tensor."""
    a = a.detach().to(torch.float64)
    b = b.detach().to(torch.float64)
    scale = b.abs().max().clamp_min(1e-12)
    return float((a - b).abs().max() / scale)


#: The parity contract (`_parity`): the compiled graph must BE the eager program.
LOGITS_TOL = 1e-4
LOSS_TOL = 1e-4
PARAM_TOL = 1e-6


def parity_failures(out: dict) -> list[str]:
    """The failed checks of one parity row — empty means the arm passed."""
    bad = []
    if out["logits"] > LOGITS_TOL:
        bad.append(f"logits {out['logits']:.2e} > {LOGITS_TOL:g}")
    if out["loss"] > LOSS_TOL:
        bad.append(f"loss {out['loss']:.2e} > {LOSS_TOL:g}")
    if out["params"] > PARAM_TOL:
        bad.append(f"param {out['param']!r} {out['params']:.2e} > {PARAM_TOL:g}")
    return bad


def project_vram(steady_gib: float, peak_gib: float, b0: int, batch: int) -> float:
    """Peak at `batch`: weights+state are the floor, activations linear in batch."""
    return steady_gib + (peak_gib - steady_gib) * batch / b0


def grad_accum_for(bytes_target: int, batch: int, seq: int) -> int:
    """The integer accumulation that lands closest to `bytes_target`."""
    return max(1, round(bytes_target / (batch * seq)))


# --- arms ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Arm:
    """One program: `compile` is the `torch.compile` kwargs, empty = eager."""

    name: str
    compile: dict

    @property
    def compiled(self) -> bool:
        return bool(self.compile)


ARMS = {
    a.name: a
    for a in (
        Arm("eager", {}),
        Arm("default", {"mode": "default"}),
        Arm("max-autotune", {"mode": "max-autotune-no-cudagraphs"}),
        Arm("trainer", {"dynamic": True}),
        Arm("strict", {"fullgraph": True}),
    )
}
DEFAULT_ARMS = "eager,default,max-autotune,trainer"


def select_arms(spec: str) -> list[Arm]:
    names = [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in ARMS]
    if unknown or not names:
        raise SystemExit(f"--arms {spec!r}: unknown arm(s) {unknown}. Known: {', '.join(ARMS)}")
    return [ARMS[n] for n in names]


# --- the harness -----------------------------------------------------------


class Clock:
    """Wall ms of a block: CUDA events on the card, perf_counter on CPU."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def __enter__(self) -> Clock:
        if self.device.type == "cuda":
            self._s = torch.cuda.Event(enable_timing=True)
            self._e = torch.cuda.Event(enable_timing=True)
            self._s.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        if self.device.type == "cuda":
            self._e.record()
            torch.cuda.synchronize(self.device)
            self.ms = self._s.elapsed_time(self._e)
        else:
            self.ms = (time.perf_counter() - self._t0) * 1e3


def _free(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _run_config(args: argparse.Namespace) -> RunConfig:
    over = dict(
        batch=args.batch,
        seq_len=args.seq,
        grad_accum=args.grad_accum,
        steps=args.warm + args.reps * args.steps,
        eval=False,
        log_every=10**9,
        ckpt_every=10**9,
        json_log="",
        data="",
        eval_data="",
        mix_plan="",
    )
    snap = Path(args.snapshot) if args.snapshot else None
    if snap is not None and snap.is_file():
        run = RunConfig.from_snapshot(snap.read_text(encoding="utf-8"))
    else:
        # The overrides go through `load_run`, not `replace`: its validation
        # ("seq_len exceeds max_seq_len") must see the shape this bench asks for,
        # not the preset's default 512 against a tiny model's 64.
        run = load_run(args.preset, train_overrides=over, model_overrides={"max_seq_len": args.seq})
    return dataclasses.replace(
        run,
        model=dataclasses.replace(run.model, max_seq_len=args.seq),
        train=dataclasses.replace(run.train, **over),
    )


def _apply_form(run: RunConfig, form: str) -> RunConfig:
    """`--form transformer` rides `block_form`; absent -> a loud SKIP, not a guess.

    The field lands with the vanilla lane (`port/src/dormouse/model/transformer.py`).
    Until then this tree's `ModelConfig` cannot express the form, and inventing a
    second spelling here is how two definitions of one flag drift apart.
    """
    if form == "current":
        return run
    if "block_form" not in {f.name for f in dataclasses.fields(ModelConfig)}:
        raise SystemExit(
            f"\n=== SKIP --form {form}: this tree's ModelConfig has no `block_form` field,\n"
            f"=== so `DormouseModel` can only build its KDA loop. The vanilla form is the\n"
            f"=== parallel lane (`block_form='transformer'`, model/transformer.py).\n"
            f"=== NOTHING WAS MEASURED. Re-run on the merged tree, or use --form current."
        )
    try:
        return dataclasses.replace(run, model=dataclasses.replace(run.model, block_form=form))
    except LoudError as e:
        raise SystemExit(
            f"\n=== SKIP --form {form}: the config refuses this form's flags:\n"
            f"=== {e}\n=== NOTHING WAS MEASURED. Name a form-native preset (--preset)."
        ) from e


def _build(run: RunConfig, device: torch.device) -> torch.nn.Module:
    """The trainer's own construction seam (`cli._model`), so the tree is the run's."""
    from dormouse.cli import _model

    return _model(run).to(device).train()


def _optimizer(model: torch.nn.Module, run: RunConfig) -> MixOptimizer:
    c = run.train
    return MixOptimizer(
        model,
        lr=c.lr,
        wd=c.wd,
        opt=c.opt,
        ns_steps=c.ns_steps,
        ns_kind=c.ns_kind,
        colrow_3d=c.muon_colrow_3d,
        muon_state_dtype=c.muon_state_dtype,
        ns_dtype=c.ns_dtype,
        muon_variant=c.muon_variant,
        muon_sf=c.muon_sf,
        muon_sf_wd=c.muon_sf_wd,
        eps=c.adam_eps,
    )


def _fresh(
    run: RunConfig, device: torch.device, state: dict
) -> tuple[torch.nn.Module, MixOptimizer]:
    model = _build(run, device)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    return model, _optimizer(model, run)


def _step(model: torch.nn.Module, opt: MixOptimizer, ids: torch.Tensor, grad_accum: int) -> None:
    """One optimizer step: `grad_accum` micro-batches, mean gradient, `opt.step()`.

    The same shape `train/loop.py`'s `micro_step` runs (loss / n_micro, backward,
    one mixed step); the firewall and the NaN gate are deliberately absent — this
    measures kernels, and the eager arm must be the trainer's own arithmetic.
    """
    opt.zero_grad(set_to_none=True)
    for _ in range(grad_accum):
        _logits, loss, _counters = model(ids, targets=ids)
        assert loss is not None, "a training step must carry targets"
        (loss / grad_accum).backward()
    opt.step()


def _dynamo() -> dict:
    counters = torch._dynamo.utils.counters
    stats = counters.get("stats") or {}
    return {
        "graph_break": dict(counters.get("graph_break") or {}),
        "unimplemented": dict(counters.get("unimplemented") or {}),
        "unique_graphs": int(stats.get("unique_graphs", 0)),
    }


def _delta(after: dict, before: dict) -> dict:
    return {
        "breaks": sum(after["graph_break"].values()) - sum(before["graph_break"].values()),
        "fallbacks": sum(after["unimplemented"].values()) - sum(before["unimplemented"].values()),
        "graphs": after["unique_graphs"] - before["unique_graphs"],
    }


def explain_breaks(
    module: torch.nn.Module, ids: torch.Tensor
) -> tuple[list[tuple[int, str, str, str]], int, int]:
    """`[(count, reason, site, source_line)]`, `graph_break_count`, `graph_count`.

    `torch._dynamo.explain` traces the EAGER module, so this runs once per bench,
    not once per arm; the per-arm `breaks` column is the runtime counter delta.
    """
    torch._dynamo.reset()
    exp = torch._dynamo.explain(module)(ids, targets=ids)
    buckets: collections.Counter = collections.Counter()
    for g in exp.break_reasons:
        frame = g.user_stack[-1] if g.user_stack else None
        site = f"{Path(frame.filename).name}:{frame.lineno} {frame.name}" if frame else "?"
        line = (frame.line or "").strip() if frame else ""
        buckets[(g.reason.splitlines()[0][:110], site, line)] += 1
    rows = [(n, reason, site, line) for (reason, site, line), n in buckets.most_common()]
    return rows, int(exp.graph_break_count), int(exp.graph_count)


def _measure(
    run: RunConfig,
    arm: Arm,
    *,
    device: torch.device,
    state: dict,
    ids: torch.Tensor,
    warm: int,
    reps: int,
    steps: int,
    backend: str,
    flops_per_byte: float,
    peak: float,
) -> dict:
    """One arm: warm, then `reps` timed runs of `steps` steps; the min wins."""
    _free(device)
    torch._dynamo.reset()
    model, opt = _fresh(run, device, state)
    target: torch.nn.Module = model
    if arm.compiled:
        target = torch.compile(model, backend=backend, **arm.compile)
    accum = run.train.grad_accum
    before = _dynamo()
    try:
        for _ in range(warm):
            _step(target, opt, ids, accum)
    except Exception as e:  # noqa: BLE001 - re-raised for every arm but `strict`
        if arm.name != "strict":
            raise
        print(
            f"\n=== STRICT: fullgraph=True refused this shape — the forward is not ONE "
            f"graph:\n=== {e}\n=== exit {EXIT_STRICT}: that is the arm's answer.\n",
            flush=True,
        )
        raise SystemExit(EXIT_STRICT) from e
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    warm_graphs = _delta(_dynamo(), before)["graphs"]
    samples: list[float] = []
    for _ in range(reps):
        with Clock(device) as total:
            for _ in range(steps):
                _step(target, opt, ids, accum)
        samples.append(total.ms / steps)
    peak_gib = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    steady_gib = torch.cuda.memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    # The optimizer step on its own, AFTER the timed loop (its grads are still
    # there). Written back-to-back so the measurement is not serialised by a
    # synchronisation inside a timed training step — the share that matters is
    # `opt.step`'s own wall time against the step's.
    opt_samples: list[float] = []
    for _ in range(reps):
        with Clock(device) as one:
            opt.step()
        opt_samples.append(one.ms)
    after = _dynamo()
    nbytes = bytes_per_step(run.train.batch, run.train.seq_len, accum)
    ms = min(samples)
    tflops = step_tflops(flops_per_byte, nbytes, ms)
    row = {
        "arm": arm.name,
        "ms": ms,
        "bps": nbytes / (ms * 1e-3),
        "tflops": tflops,
        "mfu": mfu(tflops, peak),
        "peak_gib": peak_gib,
        "steady_gib": steady_gib,
        "bytes_per_step": nbytes,
        "opt_ms": min(opt_samples),
        "opt_share": min(opt_samples) / ms if ms > 0 else 0.0,
        "graphs_warm": warm_graphs,
        **_delta(after, before),
    }
    del model, opt, target
    _free(device)
    return row


def _parity(
    run: RunConfig,
    arm: Arm,
    *,
    device: torch.device,
    state: dict,
    ids: torch.Tensor,
    backend: str,
    seed: int,
) -> dict:
    """Eager vs this arm: logits/loss <= 1e-4, one optimizer step <= 1e-6 (relative)."""

    def one(a: Arm) -> tuple[torch.Tensor, torch.Tensor, dict]:
        torch._dynamo.reset()
        model, opt = _fresh(run, device, state)
        target: torch.nn.Module = model
        if a.compiled:
            target = torch.compile(model, backend=backend, **a.compile)
        torch.manual_seed(seed)
        logits, loss, _c = target(ids, targets=ids)
        assert loss is not None
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        params = {n: p.detach().clone() for n, p in model.named_parameters()}
        return logits.detach().clone(), loss.detach().clone(), params

    ref_logits, ref_loss, ref_params = one(ARMS["eager"])
    logits, loss, params = one(arm)
    worst_name, worst = "", -1.0
    for name, p in params.items():
        rel = relative_error(p, ref_params[name])
        if rel > worst:
            worst_name, worst = name, rel
    worst = max(worst, 0.0)
    out = {
        "arm": arm.name,
        "logits": relative_error(logits, ref_logits),
        "loss": relative_error(loss, ref_loss),
        "params": worst,
        "param": worst_name,
    }
    bad = parity_failures(out)
    if bad:
        print(
            f"\n=== PARITY FAILED ({arm.name}): " + "; ".join(bad) + ".\n"
            "=== A faster arm that trains a different model is not an arm.\n",
            flush=True,
        )
        raise SystemExit(EXIT_PARITY)
    _free(device)
    return out


def _apply_route(model: torch.nn.Module, opt: MixOptimizer, route: str) -> None:
    """`fused` = the port's writer; `torch` = torch's AdamW (foreach) + batched Muon."""
    if route == "fused":
        opt.step()
        return
    if opt.muon is not None:
        opt.muon.batched_step_from_model(model)
    opt.adamw.step()


def _opt_routes(
    run: RunConfig,
    *,
    device: torch.device,
    state: dict,
    ids: torch.Tensor,
    steps: int,
    reps: int,
) -> list[dict]:
    """Requirement 4: is `opt.step` worth fusing, and does it change the run?

    Two routes from the SAME weights and the SAME batches: the port's fused
    writer (`MixOptimizer.step` -> `_foreach_adamw_step`) and torch's own AdamW
    (`opt.adamw.step`, the foreach route) beside the batched Muon. Each route's
    own wall time is the MIN of `reps` calls on one step's grads; the objective
    is checked over `steps` full steps — `bpb = loss / ln2`, exactly the
    trainer's metric — and a route that moves it by more than 1e-4 is refused,
    because "faster" is not "the same optimizer".
    """
    accum = run.train.grad_accum
    rows = []
    for route in ("fused", "torch"):
        _free(device)
        torch._dynamo.reset()
        model, opt = _fresh(run, device, state)
        bpb: list[float] = []
        timings: list[float] = []
        for k in range(steps):
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(accum):
                _logits, loss, _c = model(ids, targets=ids)
                assert loss is not None
                (loss / accum).backward()
                loss_sum += float(loss.detach()) / accum
            bpb.append(loss_sum / math.log(2.0))
            for _ in range(reps if k == steps - 1 else 1):
                with Clock(device) as clock:
                    _apply_route(model, opt, route)
                if k == steps - 1:
                    timings.append(clock.ms)
        rows.append({"route": route, "ms": min(timings), "bpb": bpb})
        del model, opt
        _free(device)
    delta = max(abs(a - b) for a, b in zip(rows[0]["bpb"], rows[1]["bpb"], strict=True))
    for r in rows:
        print(
            f"  {r['route']:<6} opt.step {r['ms']:7.2f} ms  bpb {r['bpb'][-1]:.6f} "
            f"(after {steps} steps)"
        )
    print(f"  bpb drift over {steps} steps: {delta:.2e} (gate 1e-4)")
    if delta > 1e-4:
        print(
            f"\n=== OPTIMIZER ROUTES DIVERGE: {delta:.2e} > 1e-4 bpb after {steps} "
            f"steps. The fused writer is not the same optimizer as torch's AdamW on "
            f"this model — that is a bug, not a speed/accuracy trade.\n"
            f"=== exit {EXIT_PARITY}\n",
            flush=True,
        )
        raise SystemExit(EXIT_PARITY)
    return rows


def _flop_check(run: RunConfig, device: torch.device, state: dict, seq: int) -> float:
    """The model's own bill, MFLOP per byte, at b1 x seq (fwd+bwd)."""
    from torch.utils.flop_counter import FlopCounterMode

    model = _build(run, device)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    ids = torch.zeros((1, seq), dtype=torch.long, device=device)
    counter = FlopCounterMode(display=False)
    with counter:
        _logits, loss, _c = model(ids, targets=ids)
        assert loss is not None
        loss.backward()
    total = counter.get_total_flops() / seq / 1e6
    del model
    _free(device)
    return total


# --- the table -------------------------------------------------------------


def _row(line: dict) -> str:
    mfu_txt = f"{line['mfu'] * 100:5.1f}%" if line["tflops"] > 0 else "   n/a"
    return (
        f"{line['arm']:<12} {line['ms']:9.1f} {line['bps']:10,.0f} "
        f"{line['tflops']:8.2f} {mfu_txt:>7} {line['peak_gib']:8.2f} "
        f"{line['breaks']:7d} {line['graphs']:7d} {line['fallbacks']:7d} "
        f"{line['opt_ms']:8.1f} {line['opt_share'] * 100:6.1f}%"
    )


HEADER = (
    f"{'arm':<12} {'ms/step':>9} {'B/s':>10} {'TFLOP/s':>8} {'MFU':>7} "
    f"{'peakGiB':>8} {'breaks':>7} {'graphs':>7} {'fallbk':>7} {'opt ms':>8} {'opt %':>7}"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="small", help="configs/<name>.toml")
    ap.add_argument("--snapshot", default="", help="a checkpoint's config.toml")
    ap.add_argument("--form", default="current", choices=("current", "transformer"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=3)
    ap.add_argument("--bytes-per-step", type=int, default=FLAGSHIP_BYTES_PER_STEP)
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--backend", default="inductor")
    ap.add_argument("--parity", dest="parity", action="store_true", default=True)
    ap.add_argument("--no-parity", dest="parity", action="store_false")
    ap.add_argument("--opt-arms", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--opt-steps", type=int, default=20)
    ap.add_argument("--batch-scale", action="store_true")
    ap.add_argument("--batches", default="8,16,32")
    ap.add_argument("--flops-per-byte", type=float, default=None)
    ap.add_argument("--no-flops-check", dest="flops_check", action="store_false")
    ap.add_argument("--peak-tflops", type=float, default=PEAK_TFLOPS)
    ap.add_argument("--vram-ceiling-gib", type=float, default=VRAM_CEILING_GIB)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        why = preflight("cuda")
        if why:
            # Exit 2, not the usual silent SKIP: the brief's gate, and a bench
            # that cannot tell "busy" from "done" is how two runs collide.
            print(
                f"\n=== REFUSED: {why}\n"
                f"=== exit {EXIT_BUSY}: one GPU process at a time (AGENTS 1.5).\n",
                flush=True,
            )
            return EXIT_BUSY
    run = _apply_form(_run_config(args), args.form)
    arms = select_arms(args.arms)

    model = _build(run, device)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    n_params = sum(p.numel() for p in model.parameters())
    del model
    _free(device)

    if args.flops_per_byte is None:
        measured = _flop_check(run, device, state, min(args.seq, 1024))
        flops_per_byte = measured * 1e6
        drift = flops_per_byte / FLAGSHIP_FLOPS_PER_BYTE - 1.0
        print(
            f"flops: FlopCounterMode reads {measured:.2f} MFLOP/byte at b1 x "
            f"{min(args.seq, 1024)} (the flagship's pinned bill is "
            f"{FLAGSHIP_FLOPS_PER_BYTE / 1e6:.2f}; drift {drift:+.1%})"
        )
        if abs(drift) > 0.10:
            print(
                f"flops: LOUD — the measured bill is {drift:+.1%} from the flagship's "
                f"{FLAGSHIP_FLOPS_PER_BYTE / 1e6:.2f} MFLOP/byte; the MFU column uses the "
                f"MEASURED bill, so it is this model's MFU, not the flagship's."
            )
    else:
        flops_per_byte = args.flops_per_byte
        if args.flops_check:
            measured = _flop_check(run, device, state, min(args.seq, 1024))
            drift = measured * 1e6 / flops_per_byte - 1.0
            print(
                f"flops: pinned {flops_per_byte / 1e6:.2f} MFLOP/byte; FlopCounterMode "
                f"reads {measured:.2f} (drift {drift:+.1%})"
            )
            if abs(drift) > 0.10:
                print(
                    f"flops: LOUD — the pinned denominator is {drift:+.1%} off this "
                    f"model's own bill; the MFU column is then not this model's MFU."
                )

    def _shape(batch: int, accum: int) -> RunConfig:
        return dataclasses.replace(
            run,
            model=dataclasses.replace(run.model, max_seq_len=args.seq),
            train=dataclasses.replace(run.train, batch=batch, seq_len=args.seq, grad_accum=accum),
        )

    print(
        f"\nshape b={args.batch} s={args.seq} g={args.grad_accum} -> "
        f"{bytes_per_step(args.batch, args.seq, args.grad_accum):,} B/step, "
        f"{step_flops(flops_per_byte, args.batch * args.seq * args.grad_accum) / 1e12:.3f}"
        f" TFLOP/step, {flops_per_byte / 1e6:.2f} MFLOP/B, peak {args.peak_tflops} TFLOP/s"
    )
    print(f"params: {n_params:,}  device: {device.type}  backend: {args.backend}")
    if device.type != "cuda":
        print(
            "device is not the card: ms/step is host wall time and the TFLOP/s and MFU "
            "columns are arithmetic only, NOT a hardware claim."
        )

    base = _shape(args.batch, args.grad_accum)
    ids = torch.randint(0, 256, (args.batch, args.seq), device=device)

    # One explain trace is enough: the breaks are a property of the model and
    # the shape, not of the mode. It runs on the EAGER model, before any arm.
    plain, _ = _fresh(base, device, state)
    breaks, gbc, ngraphs = explain_breaks(plain, ids)
    del plain
    _free(device)
    print(f"\ngraph breaks: {gbc} event(s) (torch._dynamo.explain; {ngraphs} graph(s))")
    if breaks:
        for n, reason, site, line in breaks:
            print(f"  {n:3d}x {site:<42} {reason}")
            if line:
                print(f"       {line}")
    else:
        print("  none — the forward is ONE graph")

    rows: list[dict] = []
    print("\n" + HEADER)
    for arm in arms:
        row = _measure(
            base,
            arm,
            device=device,
            state=state,
            ids=ids,
            warm=args.warm,
            reps=args.reps,
            steps=args.steps,
            backend=args.backend,
            flops_per_byte=flops_per_byte,
            peak=args.peak_tflops,
        )
        rows.append(row)
        print(_row(row), flush=True)
        if row["opt_share"] > 0.15:
            print(
                f"  LOUD: opt.step is {row['opt_share'] * 100:.1f}% of the step (>15%); "
                f"the fused writer is the arm to check (MixOptimizer._foreach_adamw_step)"
            )

    auto = any(r["opt_share"] > 0.15 for r in rows)
    if args.opt_arms == "on" or (args.opt_arms == "auto" and auto):
        print(f"\noptimizer routes (opt.step is >15% of the step, {args.opt_steps} steps):")
        _opt_routes(
            base,
            device=device,
            state=state,
            ids=ids,
            steps=args.opt_steps,
            reps=args.reps,
        )
    elif args.opt_arms == "off":
        print("\noptimizer routes: skipped (--opt-arms off)")

    if args.parity:
        print("\nparity vs eager (same seed, same batch, one optimizer step):")
        for arm in arms:
            if not arm.compiled:
                continue
            p = _parity(
                base,
                arm,
                device=device,
                state=state,
                ids=ids,
                backend=args.backend,
                seed=args.seed,
            )
            print(
                f"  {arm.name:<12} logits {p['logits']:.2e}  loss {p['loss']:.2e}  "
                f"worst param {p['params']:.2e} ({p['param']})"
            )

    if args.batch_scale:
        batches = [int(b) for b in args.batches.split(",") if b.strip()]
        print(
            f"\nbatch scaling (bytes/step target {args.bytes_per_step:,}; "
            f"the ceiling is {args.vram_ceiling_gib:.1f} GiB of VRAM):"
        )
        scale_rows: list[dict] = []
        for b in batches:
            accum = grad_accum_for(args.bytes_per_step, b, args.seq)
            nbytes = bytes_per_step(b, args.seq, accum)
            if scale_rows:
                p0 = scale_rows[-1]
                projected = project_vram(p0["steady_gib"], p0["peak_gib"], p0["batch"], b)
                print(
                    f"  b={b:<3} g={accum}: projected peak {projected:.2f} GiB = "
                    f"steady {p0['steady_gib']:.2f} + ({p0['peak_gib']:.2f} - "
                    f"{p0['steady_gib']:.2f}) x {b}/{p0['batch']}"
                )
                if projected > args.vram_ceiling_gib:
                    print(
                        f"\n=== REFUSED at b={b}: the projection {projected:.2f} GiB "
                        f"exceeds the {args.vram_ceiling_gib:.1f} GiB ceiling.\n"
                        f"=== steady {p0['steady_gib']:.2f} GiB, measured b={p0['batch']} "
                        f"{p0['peak_gib']:.2f} GiB, slope "
                        f"{(p0['peak_gib'] - p0['steady_gib']) / p0['batch']:.3f} GiB per "
                        f"batch element.\n=== exit {EXIT_VRAM}: refusing to OOM the card.\n",
                        flush=True,
                    )
                    return EXIT_VRAM
            if nbytes != args.bytes_per_step:
                print(
                    f"  b={b:<3} g={accum}: {nbytes:,} B/step, not "
                    f"{args.bytes_per_step:,} — {args.bytes_per_step}/{b * args.seq} = "
                    f"{args.bytes_per_step / (b * args.seq):g} micro-batches does not exist; "
                    f"this row's B/s carries {nbytes / args.bytes_per_step:.2f}x the bytes"
                )
            shape = _shape(b, accum)
            shape_ids = torch.randint(0, 256, (b, args.seq), device=device)
            chosen = [a for a in arms if a.compiled] or arms[-1:]
            for arm in chosen:
                row = _measure(
                    shape,
                    arm,
                    device=device,
                    state=state,
                    ids=shape_ids,
                    warm=args.warm,
                    reps=args.reps,
                    steps=args.steps,
                    backend=args.backend,
                    flops_per_byte=flops_per_byte,
                    peak=args.peak_tflops,
                )
                row["batch"], row["grad_accum"] = b, accum
                scale_rows.append(row)
                print("  " + _row(row), flush=True)
                if row["peak_gib"] > args.vram_ceiling_gib:
                    print(
                        f"\n=== REFUSED at b={b}: the measured peak {row['peak_gib']:.2f} "
                        f"GiB already exceeds the {args.vram_ceiling_gib:.1f} GiB ceiling.\n"
                        f"=== exit {EXIT_VRAM}: no bigger batch can be tried.\n",
                        flush=True,
                    )
                    return EXIT_VRAM
        if scale_rows:
            best = max(scale_rows, key=lambda r: r["bps"])
            print(
                f"\nbest B/s: {best['arm']} at b={best['batch']} g={best['grad_accum']} "
                f"-> {best['bps']:,.0f} B/s ({best['mfu'] * 100:.1f}% MFU, "
                f"{best['peak_gib']:.2f} GiB)"
            )
        rows = rows + scale_rows

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "peak_tflops": args.peak_tflops,
                    "flops_per_byte": flops_per_byte,
                    "params": n_params,
                    "rows": rows,
                    "breaks": [
                        {"count": n, "reason": r, "site": s, "line": ln} for n, r, s, ln in breaks
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\njson: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
