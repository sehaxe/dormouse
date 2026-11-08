"""VRAM per token: the four activation-memory arms on one card, one process.

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/bench_vram.py \
        --config configs/medium.toml --arms none,iter,selective,iter+offload4 \
        --shapes b1s8192,b2s8192,b8s1024

**What the owner gets, and what this file does NOT decide.** For every arm ×
shape it prints peak GiB (`torch.cuda.max_memory_allocated`), median ms/step and
bytes/s. The arms are the enum the trainer now carries:

| arm | `iter_checkpoint` | `activation_recompute` | `activation_offload` |
|---|---|---|---|
| `none` | false | none | 0 |
| `iter` | true | none (= full) | 0 |
| `selective` | false | selective | 0 |
| `iter+offload4` | true | none (= full) | 4.0 |

Every step is a REAL training step — forward, backward, `clip_grads`, optimizer
step — because the peak includes the optimizer's first-step state. A shape that
does not fit is reported as OOM and the sweep continues; it is not retried.

**Ledger mode** (`--ledger`) runs ONE CPU forward per arm on a small probe and
prints the resident saved-tensor bytes per token from
`model/recompute.saved_table` — the arithmetic table the report quotes, not a
device guess. It is deliberately CPU-only: the card must not be touched by a
bookkeeping pass while a training run owns it.

**Preflight** is `_lane.preflight` (no CUDA call in the parent), so a busy card
prints SKIP and exits 0 with no table. Nothing here warms up `torch.compile` or
captures a graph.

The budget of 4.0 GiB is the arm the task names (`iter+offload4`); pass
`--offload-gib` to price another. The bench prints the measured peak, not a
prediction: the ledger says what each arm SAVES, the card says whether the batch
grows.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight, skip  # noqa: E402


def _shape(text: str) -> tuple[int, int]:
    """`b4s2048` -> (4, 2048). Loud on anything else."""
    if not text.startswith("b") or "s" not in text:
        raise argparse.ArgumentTypeError(f"shape {text!r}: expected b<batch>s<seq>")
    b, s = text[1:].split("s", 1)
    return int(b), int(s)


def _arm(name: str, offload_gib: float):
    """`(iter_checkpoint, activation_recompute, activation_offload)` for an arm."""
    if name == "none":
        return False, "none", 0.0
    if name == "iter":
        return True, "none", 0.0
    if name == "selective":
        return False, "selective", 0.0
    if name.startswith("iter+offload"):
        suffix = name[len("iter+offload") :]
        # `iter+offload4` names its own budget; bare `iter+offload` uses the
        # flag. A named budget is authoritative (the report quotes that arm).
        return True, "none", float(suffix) if suffix else offload_gib
    raise argparse.ArgumentTypeError(
        f"arm {name!r}: known none, iter, selective, iter+offload<N> (the budget is --offload-gib)"
    )


@torch.no_grad()
def _ids(batch: int, seq: int, vocab: int, device: torch.device):
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(0, vocab, (batch, seq), generator=gen, device=device)
    return ids, (ids + 1) % vocab


def _step(model, opt, ids, targets, device) -> None:
    from dormouse.instrument import Counters
    from dormouse.train.optim import clip_grads

    model.zero_grad(set_to_none=True)
    _, loss, _ = model(ids, targets, Counters())
    loss.backward()
    clip_grads(model.parameters(), 1.0)
    opt.step()


def bench(
    config: str,
    arms: list[str],
    shapes: list[tuple[int, int]],
    offload_gib: float,
    reps: int,
    warm: int,
    device_name: str,
) -> None:
    from dormouse.config import load_run
    from dormouse.model import DormouseModel
    from dormouse.train.optim import MixOptimizer

    device = torch.device(device_name)
    print(
        f"{'arm':>14} {'shape':>10} {'peak GiB':>9} {'ms/step':>9} {'B/s':>10}",
        flush=True,
    )
    for b, s in shapes:
        run = load_run(
            config,
            train_overrides={"batch": b, "seq_len": s, "grad_accum": 1},
            model_overrides={"max_seq_len": s},
        )
        model = DormouseModel(run.model, device=device)
        opt = MixOptimizer(
            model,
            lr=1e-3,
            wd=0.01,
            opt=run.train.opt,
            betas=(run.train.adam_beta1, run.train.adam_beta2),
            eps=run.train.adam_eps,
        )
        ids, targets = _ids(b, s, run.model.vocab, device)
        for arm in arms:
            ckpt, recompute, offload = _arm(arm, offload_gib)
            model.configure_actmem(
                iter_checkpoint=ckpt,
                activation_offload=offload,
                activation_recompute=recompute,
            )
            try:
                for _ in range(warm):
                    _step(model, opt, ids, targets, device)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                times: list[float] = []
                for _ in range(reps):
                    t0 = time.perf_counter()
                    _step(model, opt, ids, targets, device)
                    torch.cuda.synchronize(device)
                    times.append(time.perf_counter() - t0)
                peak = torch.cuda.max_memory_allocated(device) / 2**30
                ms = statistics.median(times) * 1e3
                print(
                    f"{arm:>14} {f'b{b}s{s}':>10} {peak:>9.2f} {ms:>9.1f} "
                    f"{b * s / (ms / 1e3):>10.1f}",
                    flush=True,
                )
            except torch.cuda.OutOfMemoryError:
                print(f"{arm:>14} {f'b{b}s{s}':>10} {'OOM':>9}", flush=True)
                torch.cuda.empty_cache()
        del model, opt, ids, targets
        torch.cuda.empty_cache()


def ledger(config: str, arms: list[str], seq: int, offload_gib: float) -> None:
    """One CPU forward per arm: resident saved bytes / token from the hooks."""
    from dormouse.config import load_run
    from dormouse.instrument import Counters
    from dormouse.model import DormouseModel
    from dormouse.model.recompute import saved_table

    run = load_run(
        config,
        train_overrides={"batch": 1, "seq_len": min(seq, 512)},
        model_overrides={"max_seq_len": min(seq, 512)},
    )
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = DormouseModel(run.model)
    ids, targets = _ids(1, min(seq, 512), run.model.vocab, torch.device("cpu"))
    print(
        f"{'arm':>14} {'resident tensors':>16} {'resident MiB':>12} "
        f"{'resident B/tok':>14} {'host MiB':>9}",
        flush=True,
    )
    tokens = ids.numel()
    for arm in arms:
        ckpt, recompute, offload = _arm(arm, offload_gib)
        model.configure_actmem(
            iter_checkpoint=ckpt,
            activation_offload=offload,
            activation_recompute=recompute,
        )
        if offload:
            # The offload pair is OUTSIDE the checkpoint, so it packs the
            # checkpoint node's inputs; the counting pair then sees nothing on
            # device. Two forwards: the first profiles, the second parks.
            model(ids, targets, Counters())
            model(ids, targets, Counters())
            mech = model._act_offload
            assert mech is not None
            parked = mech.stats()["last_forward_bytes"]
            print(
                f"{arm:>14} {'0 (parked)':>16} {0 / 2**20:>12.2f} "
                f"{0 / tokens:>14.1f} {parked / 2**20:>9.2f}",
                flush=True,
            )
            continue
        table, _ = saved_table(lambda: model(ids, targets, Counters()))
        print(
            f"{arm:>14} {table.tensors:>16} {table.bytes / 2**20:>12.2f} "
            f"{table.bytes / tokens:>14.1f} {'-':>9}",
            flush=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/medium.toml")
    ap.add_argument("--arms", default="none,iter,selective,iter+offload4")
    ap.add_argument("--shapes", default="b1s8192,b2s8192,b4s8192,b8s1024")
    ap.add_argument("--offload-gib", type=float, default=4.0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ledger", action="store_true", help="CPU saved-tensor table only")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        _arm(arm, args.offload_gib)
    try:
        shapes = [_shape(s.strip()) for s in args.shapes.split(",") if s.strip()]
    except (ValueError, argparse.ArgumentTypeError) as e:
        print(f"bad --shapes: {e}", file=sys.stderr)
        return 2

    why = preflight("cpu" if args.ledger else args.device)
    if why is not None:
        return skip(why)
    if args.ledger:
        ledger(args.config, arms, max(s for _, s in shapes), args.offload_gib)
        return 0
    if not torch.cuda.is_available():
        return skip("device=cuda but torch.cuda.is_available() is False")
    bench(args.config, arms, shapes, args.offload_gib, args.reps, args.warm, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
