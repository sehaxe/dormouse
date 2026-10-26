#!/usr/bin/env python3
"""The distributed gate's driver: 2 ranks, gloo, CPU, the tiny model.

This is the honest maximum this box allows (one GPU): the data-parallel MATH is
proved on the CPU arm at 2 ranks, and the multi-GPU wiring is the same code path
with `nccl` and one process per card. Three modes, one comparison each:

* `single`  — one process at the GLOBAL batch: the reference loss curve.
* `rank`    — one process per rank under torchrun: the fleet.
* `gradref` — the hand-computed token-weighted gradient for ONE step (stage 3):
              `g = (t0·g0 + t1·g1)/(t0+t1)` over the ranks' OWN batches, applied
              by the same `MixOptimizer` one step — the reference the fleet's
              post-step parameters must equal.

The gate itself is `tests/test_distributed.py`; this file is runnable by hand,
which is also how the owner checks a real two-card box:

    python tools/dist_gate.py --mode single --out /tmp/g --steps 20 --batch 4
    python -m torch.distributed.run --standalone --nproc_per_node=2 \\
        tools/dist_gate.py --mode rank --out /tmp/g --steps 20 --batch 2

The model and the corpus are the test suite's tiny fixtures (d=64, E=4, r=8,
2 iterations), written into the out dir so two modes on different machines
compare the same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

from dormouse.config import ModelConfig, RunConfig, TrainConfig
from dormouse.model import DormouseModel
from dormouse.train.data import _windows
from dormouse.train.distributed import parse_capacity, rank_byte_offset
from dormouse.train.loop import train
from dormouse.train.optim import MixOptimizer

TINY = dict(
    d_model=64,
    n_heads=4,
    head_dim=16,
    d_ffn=128,
    vocab=256,
    max_seq_len=64,
    max_iter=2,
    rank=8,
    n_experts=4,
    moe_topk=2,
    engram_rows=64,
    engram_dim=8,
    bf16=False,
    shared_rank=16,
    #: DROPLESS (0), not the preset's 1.0: a capacity-padded dispatch drops
    #: assignments by a rule that depends on the window's composition, so the
    #: fleet's split window is a different model (the gate would be measuring
    #: two different networks). The refusal in `refuse_incompatible` makes this
    #: a LOUD requirement of any distributed run, not a test-only tweak.
    moe_capacity=0.0,
)
SEQ = 64
STEPS = 20
#: Low enough that the 20-step trajectory is not chaotic: the fleet/reference
#: divergence at this lr is fp round-off amplified by a routing flip or two,
#: while lr=0.01 amplifies 1e-8 to 3.5% in 20 steps (measured 2026-10-19).
LR = 1e-3
WORLD = 2


def write_corpus(path: Path, n_bytes: int = 4096, cycle: int = 16) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(range(cycle)) * (n_bytes // cycle + 1))
    return path


def make_run(
    out: Path,
    *,
    steps: int,
    batch: int,
    capacity: str = "",
    dist: bool = False,
    lr: float = LR,
    shard: str = "none",
    opt: str = "mix",
) -> RunConfig:
    cfg = TrainConfig(
        steps=steps,
        batch=batch,
        seq_len=SEQ,
        lr=lr,
        log_every=1,
        ckpt_every=10_000,
        ckpt_name="gate",
        out_dir=str(out / "ckpts"),
        data=str(write_corpus(out / "corpus.bin")),
        dist=dist,
        dist_capacity=capacity,
        dist_reduce="fp32",  # the exact reference; bf16 is a separate gate
        dist_shard=shard,
        opt=opt,
    )
    return RunConfig(source="dist-gate", model=ModelConfig(**TINY), train=cfg)


def build(seed: int) -> DormouseModel:
    """The one init every mode shares: same seed, same model, same file order."""
    torch.manual_seed(seed)
    return DormouseModel(ModelConfig(**TINY))


def _plain(t: torch.Tensor, *, gather: bool = False) -> torch.Tensor:
    """A plain CPU tensor: FSDP2's DTensors are gathered (a COLLECTIVE — every
    rank must call this in the same order) or read as the local shard."""
    t = t.detach()
    if hasattr(t, "full_tensor"):  # DTensor (FSDP2)
        t = t.full_tensor() if gather else t.to_local()
    return t.cpu().contiguous()


def fingerprint(model: torch.nn.Module, *, gather: bool = False) -> str:
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(_plain(p, gather=gather).numpy().tobytes())
    return h.hexdigest()


def losses_from(lines: list[str]) -> list[float]:
    out = []
    for line in lines:
        if line.startswith("eval "):
            continue
        m = re.search(r"\bloss=([-0-9.eE+]+)", line)
        if m:
            out.append(float(m.group(1)))
    return out


def save(
    out: Path, name: str, model, losses: list[float], *, gather: bool = False, **extra
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    params = [_plain(p, gather=gather).clone() for p in model.parameters()]
    torch.save(params, out / f"{name}.pt")
    (out / f"{name}.json").write_text(
        json.dumps(
            {
                "name": name,
                "losses": losses,
                "fingerprint": fingerprint(model, gather=gather),
                "params": f"{name}.pt",
                **extra,
            },
            indent=1,
        )
    )


def run_single(args) -> None:
    out = Path(args.out)
    run = make_run(out, steps=args.steps, batch=args.batch, lr=args.lr)
    model = build(args.seed)
    lines: list[str] = []
    train(model, run, log=lines.append)
    save(out, "single", model, losses_from(lines), batch=args.batch, steps=args.steps)


def run_rank(args) -> None:
    from dormouse.train.distributed import train_dist

    out = Path(args.out)
    run = make_run(
        out,
        steps=args.steps,
        batch=args.batch,
        capacity=args.capacity,
        dist=True,
        lr=args.lr,
        shard=args.shard,
        opt=args.opt,
    )
    # Seeded BEFORE `train_dist` joins the fleet, so every rank builds the same
    # model (and the single-process reference can build it too). dtype is off in
    # this run, which is what makes a pre-built model legal.
    model = build(args.seed)
    lines: list[str] = []
    report = train_dist(run, log=lines.append, device_type="cpu", model=model, close=False)
    save(
        out,
        f"rank{report['rank']}",
        model,
        losses_from(lines),
        batch=args.batch,
        capacity=args.capacity,
        rank=report["rank"],
        world_size=report["world_size"],
        steps=args.steps,
        # FSDP2 shards the parameters, so the cross-rank equality claim needs
        # the GATHERED tensors (a collective every rank runs in the same order).
        gather=args.shard == "fsdp",
    )
    import torch.distributed as dist

    dist.destroy_process_group()


def run_gradref(args) -> None:
    out = Path(args.out)
    run = make_run(out, steps=1, batch=args.batch, capacity=args.capacity, lr=args.lr)
    model = build(args.seed)
    cfg = run.train
    shapes = parse_capacity(
        args.capacity, world_size=WORLD, cfg=cfg, max_seq_len=model.cfg.max_seq_len
    )
    corpus = (out / "corpus.bin").read_bytes()
    total = sum(s.tokens for s in shapes)
    acc = [torch.zeros_like(p) for p in model.parameters()]
    local = []
    for r, shape in enumerate(shapes):
        off = rank_byte_offset(shapes, r)
        raw = corpus[off : off + shape.batch * (shape.seq_len + 1)]
        x, y = _windows(raw, shape.batch, shape.seq_len)
        model.zero_grad()
        _, loss, _ = model(x, y)
        local.append(float(loss.detach()))
        loss.backward()
        share = shape.tokens / total
        for a, p in zip(acc, model.parameters(), strict=True):
            if p.grad is not None:  # a disabled arm's parameters see no gradient
                a.add_(p.grad.detach(), alpha=share)
    for p, a in zip(model.parameters(), acc, strict=True):
        if p.grad is not None:  # keep None for the arms this objective does not use
            p.grad = a
    # The same optimizer construction the trainer uses, so the reference step is
    # the fleet's step given the same gradients (a fixture, not a second trainer).
    opt = MixOptimizer(
        model,
        lr=cfg.lr,
        wd=cfg.wd,
        opt=cfg.opt,
        ns_steps=cfg.ns_steps,
        ns_kind=cfg.ns_kind,
        colrow_3d=cfg.muon_colrow_3d,
        muon_state_dtype=cfg.muon_state_dtype,
        ns_dtype=cfg.ns_dtype,
        muon_variant=cfg.muon_variant,
        muon_sf=cfg.muon_sf,
    )
    opt.step()
    save(
        out,
        "gradref",
        model,
        local,
        shapes=[s.line() for s in shapes],
        tokens=total,
        steps=1,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", required=True, choices=("single", "rank", "gradref"))
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--batch", type=int, default=2, help="the config's batch")
    p.add_argument("--capacity", default="", help="per-rank profile: '4,2'")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--shard", default="none", choices=("none", "fsdp"))
    p.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="the rank's device; the gate is `cpu` (gloo), the owner's check is `cuda` (nccl)",
    )
    p.add_argument("--opt", default="mix", choices=("mix", "adamw"))
    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    runner = {"single": run_single, "rank": run_rank, "gradref": run_gradref}[args.mode]
    runner(args)


if __name__ == "__main__":
    main()
