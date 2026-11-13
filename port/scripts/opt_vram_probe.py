"""The optimizer's own memory, against the allocator: state by tensor and dtype.

WHY THIS EXISTS. `scripts/vram_ledger.py` prices the optimizer on paper
(`optimizer_bytes`, `ns_workspace`) and `docs/records/reviews/2026-11-08-optimizer-vram.md`
found three things it cannot see: the `ns_dtype` rung has no lever in the ledger,
the whole-step update list is not priced at all, and the ledger's activation model
ignores `n_layers` on a multi-layer preset. This script is the measurement the
report's §2 table comes from — one arm per process, because the allocator's
high-water mark is per process.

WHAT IT READS, in order (all `torch.cuda.memory_allocated()`, which is live
tensor bytes, not the reserved pool):

  alloc_model      params only, after `.to("cuda")`
  alloc_pre_state  after ONE backward — params + grads (+ resident workspace)
  alloc_post_state after ONE `opt.step()` — plus the optimizer state
  alloc_backward   a second step's post-backward baseline
  peak_step        that step's `max_memory_allocated`

so `alloc_post_state - alloc_pre_state` is the state the optimizer keeps, and
`peak_step - alloc_backward` is what one step allocates transiently (the NS
chain and the update list — not state).

The arm is set by the two dtype rungs and, optionally, the chunk cap:

    python scripts/opt_vram_probe.py --state fp32 --ns fp32
    python scripts/opt_vram_probe.py --state bf16 --ns bf16 --chunk 16777216

It builds the optimizer exactly as `train/loop.py` does (`run.model`'s
`tsct_expert_lr`, all `TrainConfig` keys), on the preset's own config, at b8 x
s1024. Run it under the GPU gateway like every other GPU command:

    bash scripts/gpu_lock.sh .venv/bin/python scripts/opt_vram_probe.py --state bf16 --ns bf16
"""

from __future__ import annotations

import argparse
import json

import torch

from dormouse.cli import _model, _run
from dormouse.train.optim import MixOptimizer


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--ns", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--preset", default="flagship3_dense")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=0, help="override train.muon_chunk_elems")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    torch.cuda.init()
    dev = torch.device("cuda")
    run = _run(a.preset, [f"batch={a.batch}", f"seq_len={a.seq}", "steps=1"])
    t = run.train
    chunk = a.chunk or t.muon_chunk_elems
    model = _model(run).to(dev).train()
    torch.cuda.synchronize()
    alloc_model = torch.cuda.memory_allocated()
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    opt = MixOptimizer(
        model,
        lr=t.lr,
        wd=t.wd,
        opt=t.opt,
        ns_steps=t.ns_steps,
        ns_kind=t.ns_kind,
        colrow_3d=t.muon_colrow_3d,
        muon_state_dtype=a.state,
        ns_dtype=a.ns,
        muon_variant=t.muon_variant,
        muon_sf=t.muon_sf,
        muon_sf_wd=t.muon_sf_wd,
        muon_chunk_elems=chunk,
        muon_lr_mult=t.muon_lr_mult,
        muon_lr_mode=t.muon_lr_mode,
        eps=t.adam_eps,
        tsct_expert_lr=run.model.tsct_expert_lr,
    )

    ids = torch.randint(0, 256, (a.batch, a.seq), device=dev)
    _, loss, _ = model(ids, targets=ids)
    loss.backward()
    torch.cuda.synchronize()
    alloc_pre_state = torch.cuda.memory_allocated()
    grads_sum = sum(
        p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None
    )
    opt.step()
    torch.cuda.synchronize()
    alloc_post_state = torch.cuda.memory_allocated()

    opt.zero_grad(set_to_none=True)
    _, loss, _ = model(ids, targets=ids)
    loss.backward()
    torch.cuda.synchronize()
    alloc_backward = torch.cuda.memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    opt.step()
    torch.cuda.synchronize()
    alloc_step = torch.cuda.memory_allocated()
    peak_step = torch.cuda.max_memory_allocated()

    # The optimizer's own accounting: every tensor of `state[p]`, by key and dtype.
    by_key: dict[str, dict[str, int]] = {}
    by_dtype: dict[str, int] = {}
    for name, half in (("muon", opt.muon), ("adamw", opt.adamw)):
        if half is None:
            continue
        for st in half.state.values():
            for k, v in st.items():
                if not torch.is_tensor(v):
                    continue
                n = v.numel() * v.element_size()
                by_key.setdefault(f"{name}.{k}", {"bytes": 0, "tensors": 0})
                by_key[f"{name}.{k}"]["bytes"] += n
                by_key[f"{name}.{k}"]["tensors"] += 1
                by_dtype[str(v.dtype)] = by_dtype.get(str(v.dtype), 0) + n
    state_sum = sum(v["bytes"] for v in by_key.values())

    groups = []
    for g in opt.muon._groups:
        d0, d1 = g.shape
        per = d0 * d1
        groups.append(
            {
                "shape": [d0, d1],
                "rows": g.rows,
                "elems": per * g.rows,
                "mom_dtype": str(g.mom.dtype),
                "mom_bytes": g.mom.numel() * g.mom.element_size(),
                # `_classic_ns_many` flushes a BATCH when the next tensor would
                # cross the cap; a single tensor larger than the cap goes in
                # whole, so this is the cap's intent, not its bound.
                "chunk_rows": max(1, min(g.rows, chunk // per)),
                "chunk_elems": max(1, min(g.rows, chunk // per)) * per,
                "packed": g.packed,
            }
        )

    out = {
        "tag": a.tag or f"state={a.state}-ns={a.ns}-chunk={chunk}",
        "preset": a.preset,
        "batch": a.batch,
        "seq": a.seq,
        "state_dtype": a.state,
        "ns_dtype": a.ns,
        "params": sum(p.numel() for p in model.parameters()),
        "n_tensors": sum(1 for _ in model.parameters()),
        "param_bytes": param_bytes,
        "buffer_bytes": buffer_bytes,
        "grads_sum": grads_sum,
        "chunk_elems": chunk,
        "ns_steps": t.ns_steps,
        "alloc_model": alloc_model,
        "alloc_pre_state": alloc_pre_state,
        "alloc_post_state": alloc_post_state,
        "alloc_backward": alloc_backward,
        "alloc_step": alloc_step,
        "peak_step": peak_step,
        "grads_delta": alloc_pre_state - alloc_model,
        "state_delta": alloc_post_state - alloc_pre_state,
        "state_sum": state_sum,
        "state_by_key": by_key,
        "state_by_dtype": by_dtype,
        "muon_groups": groups,
        "step_transient": peak_step - alloc_backward,
    }
    print("PROBE " + json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
