#!/usr/bin/env -S uv run --python /home/sehaxe/dormouse/port/.venv/bin/python --script
"""What a recurrent block costs in parameters, seconds and VRAM — measured.

Two modes, ONE process each (a CUDA context per measurement, `_lane.py`'s rule):

  run     the REAL trainer (`dormouse.train.loop.train`, the same call the CLI
          makes) for `--steps` steps. ms/step comes from the JSONL's per-window
          `tokens_per_s` (median and p95 over the windows, first window kept:
          every step is timed by the trainer's own timer) plus the wall clock
          over `train()`. Peak allocated/reserved are read at the end: nothing
          resets the allocator's high-water mark, and `stepaudit` is disarmed
          unless `install()` is called, so the number is the whole run's.
  decomp  the SAME model and the SAME `MixOptimizer` the trainer builds, 3
          synthetic steps. After the run the live objects are read: parameters,
          `.grad` bytes, the optimizer's own momentum/exp_avg tensors. The
          autograd save census comes from a `saved_tensors_hooks` pair whose
          pack hook returns the tensor UNCHANGED and only adds its bytes — no
          copy, no dtype change. That is the activation set the ledger models
          with arithmetic (`scripts/vram_ledger.py:activation_components`).

Nothing here writes a checkpoint (`ckpt_every` is forced huge by the caller's
own `--set`), and `--set` is split exactly as `dormouse train` splits it.

    bash scripts/gpu_lock.sh .venv/bin/python scripts/bench_loop_econ.py \
        --mode run --preset configs/flagship_base.toml --steps 150 \
        --set ns_dtype=bf16 --set muon_state_dtype=bf16 --set muon_lr_mult=12 \
        --json-log /tmp/opencode/econ_loop.jsonl --out /tmp/opencode/econ_loop.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PORT / "src"))

GIB = 2**30


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("run", "decomp"), default="run")
    ap.add_argument("--preset", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--data", default="../data_local/best/fwedu_train.bin")
    ap.add_argument("--mix-plan", default="configs/mix_refetch.toml")
    ap.add_argument("--json-log", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--set", action="append", default=[])
    return ap.parse_args()


def _saved_census() -> dict:
    """The autograd save set of ONE backward, in bytes. Pack is identity."""
    out = {"tensors": 0, "bytes": 0}

    def pack(t):
        out["tensors"] += 1
        out["bytes"] += t.numel() * t.element_size()
        return t

    return out, pack


def _opt_state_bytes(opt) -> tuple[int, dict]:
    """Every tensor the optimizer owns, by half. `step` counters are excluded."""
    by_half: dict[str, int] = {}
    for half in ("muon", "adamw"):
        o = getattr(opt, half, None)
        if o is None:
            by_half[half] = 0
            continue
        total = 0
        for st in o.state.values():
            for key, v in st.items():
                if key == "step" or not hasattr(v, "numel"):
                    continue
                total += v.numel() * v.element_size()
        by_half[half] = total
    return sum(by_half.values()), by_half


def main() -> int:
    a = _args()
    import torch

    from dormouse.cli import _device, _freeze_gc, _model, _split_sets
    from dormouse.config import load_run
    from dormouse.instrument import Counters
    from dormouse.train.loop import train as _train
    from dormouse.train.optim import MixOptimizer

    model_over, train_over = _split_sets(a.set)
    train_over.update(
        batch=a.batch,
        seq_len=a.seq,
        steps=a.steps,
        eval=False,
        doc_masking=False,
        data=a.data,
        mix_plan=a.mix_plan,
        out_dir=f"/tmp/opencode/econ_{a.tag or 'x'}",
        ckpt_every=10**9,
        log_every=a.log_every,
    )
    if a.json_log:
        train_over["json_log"] = a.json_log
    run = load_run(a.preset, model_overrides=model_over, train_overrides=train_over)
    dev = _device("cuda")
    torch.cuda.init()
    torch.cuda.synchronize()
    ctx_alloc = torch.cuda.memory_allocated()
    model = _model(run).to(dev)
    torch.cuda.synchronize()
    params_bytes = torch.cuda.memory_allocated()
    n_params = sum(p.numel() for p in model.parameters())
    _freeze_gc()

    out: dict = {
        "tag": a.tag,
        "mode": a.mode,
        "preset": str(a.preset),
        "source": run.source,
        "params": n_params,
        "params_bytes": params_bytes,
        "ctx_alloc_bytes": ctx_alloc,
        "batch": a.batch,
        "seq": a.seq,
        "steps": a.steps,
        "max_iter": run.model.max_iter,
        "sets": a.set,
        "ns_dtype": run.train.ns_dtype,
        "muon_state_dtype": run.train.muon_state_dtype,
        "muon_lr_mult": run.train.muon_lr_mult,
        "opt": run.train.opt,
        "lr": run.train.lr,
    }

    if a.mode == "run":
        seen: dict = {}
        step_fn = MixOptimizer.step

        def step(self, *args, **kw):
            seen["opt"] = self
            return step_fn(self, *args, **kw)

        MixOptimizer.step = step
        # `train()` imports MixOptimizer's class object from optim; patching the
        # class method is what the trainer calls. Restored at exit of main.
        t0 = time.perf_counter()
        report = _train(
            model,
            run,
            device=dev,
            runs_dir="/tmp/opencode/runs_bench",
            log=print,
        )
        wall = time.perf_counter() - t0
        torch.cuda.synchronize()
        out["wall_s"] = wall
        out["ms_step_wall"] = 1e3 * wall / max(report["steps"], 1)
        out["peak_alloc_bytes"] = int(torch.cuda.max_memory_allocated())
        out["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        out["loss"] = report["loss"]
        out["nan_skipped"] = report["nan_skipped"]
        opt = seen.get("opt")
        if opt is not None:
            state, halves = _opt_state_bytes(opt)
            out["opt_state_bytes"] = state
            out["opt_state_by_half"] = halves
            out["muon_params"] = opt.muon.n_muon if hasattr(opt.muon, "n_muon") else None
            out["n_muon"] = opt.n_muon
            out["n_adamw"] = opt.n_adamw
        out["grad_bytes"] = sum(
            p.grad.numel() * p.grad.element_size()
            for p in model.parameters()
            if p.grad is not None
        )
    else:
        opt = MixOptimizer(
            model,
            lr=run.train.lr,
            wd=run.train.wd,
            opt=run.train.opt,
            ns_steps=run.train.ns_steps,
            ns_kind=run.train.ns_kind,
            colrow_3d=run.train.muon_colrow_3d,
            muon_state_dtype=run.train.muon_state_dtype,
            ns_dtype=run.train.ns_dtype,
            muon_variant=run.train.muon_variant,
            muon_sf=run.train.muon_sf,
            muon_sf_wd=run.train.muon_sf_wd,
            muon_lr_mult=run.train.muon_lr_mult,
        )
        ids = torch.randint(0, run.model.vocab, (a.batch, a.seq), device=dev)
        # Step 1: the optimizer creates its state here.
        opt.zero_grad(set_to_none=True)
        _, loss1, _ = model(ids, targets=ids, counters=Counters())
        loss1.backward()
        opt.step()
        torch.cuda.synchronize()
        out["alloc_after_step1"] = int(torch.cuda.memory_allocated())
        out["loss_step1"] = float(loss1.detach())
        # Step 2: census + one full step's peak.
        opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        census, pack = _saved_census()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            _, loss2, _ = model(ids, targets=ids, counters=Counters())
            loss2.backward()
        torch.cuda.synchronize()
        out["peak_fwd_bwd_bytes"] = int(torch.cuda.max_memory_allocated())
        out["alloc_after_backward"] = int(torch.cuda.memory_allocated())
        out["saved_tensors"] = census["tensors"]
        out["saved_bytes"] = census["bytes"]
        out["grad_bytes"] = sum(
            p.grad.numel() * p.grad.element_size()
            for p in model.parameters()
            if p.grad is not None
        )
        opt.step()
        torch.cuda.synchronize()
        out["peak_step_bytes"] = int(torch.cuda.max_memory_allocated())
        out["alloc_after_opt"] = int(torch.cuda.memory_allocated())
        state, halves = _opt_state_bytes(opt)
        out["opt_state_bytes"] = state
        out["opt_state_by_half"] = halves
        out["n_muon"] = opt.n_muon
        out["n_adamw"] = opt.n_adamw
        out["loss_step2"] = float(loss2.detach())

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
