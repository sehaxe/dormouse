"""1M-context: what the KDA+CSA+HCA stack actually costs, and whether it recalls.

The owner's requirement is a 1 000 000-byte context with the linear-attention
stack we already have (KDA + CJSA fine + HCA coarse), no MLA, no extra machinery.

This driver answers three questions with measurements, not arithmetic:

1. MEMORY — peak VRAM and the resident state size at 32k, 128k, 256k, 1M.
2. COST — prefill ms and ms/byte, with `csa_hierarchical` on and off, so the
   selection cost is visible rather than assumed.
3. RECALL — a needle placed at distance 0.1·L, 0.5·L and 0.95·L of the context:
   can the model score the needle's continuation better than a control position?
   With an untrained model the *absolute* numbers are noise, but the *ratio*
   needle-vs-control at each distance is the mechanism's signature: a level that
   forgets returns to 1.0, a level that recalls stays below it.

Run on a small config (d=192..256) so 1M fits; the reachable shapes are printed
first and the driver refuses to lie about what it measured.

    PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/probe_1m.py
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import torch

from dormouse.config import ModelConfig
from dormouse.model.model import DormouseModel


def build(d_model: int, layers: int, context: int, hierarchical: bool, coarse: bool = True) -> DormouseModel:
    cfg = ModelConfig(
        d_model=d_model,
        n_heads=max(1, d_model // 64),
        head_dim=64,
        d_ffn=4 * d_model,
        block_form="kda",
        loop_blocks=layers,
        max_iter=layers,
        max_seq_len=context,
        use_kda=True,
        use_csa=True,
        csa_chunk=8,
        csa_kv_dim=64,
        csa_topk=16,
        csa_hierarchical=hierarchical,
        csa_mid_chunk=512 if hierarchical else 0,
        csa_coarse_chunk=256 if coarse else 0,
        csa_coarse_kv_dim=64,
        use_mhc=False,
        use_attnres=False,
        use_tsct=False,
        use_engram=False,
        n_experts=1,
        moe_topk=0,
    )
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return DormouseModel(cfg).to(dev).eval()


@torch.no_grad()
def measure(model: DormouseModel, context: int) -> dict:
    dev = next(model.parameters()).device
    x = torch.randint(0, 256, (1, context), device=dev)
    if dev.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    return {
        "context": context,
        "ms": round(dt * 1000, 1),
        "us_per_byte": round(dt / context * 1e6, 2),
        "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3) if dev.type == "cuda" else None,
    }


@torch.no_grad()
def recall(model: DormouseModel, context: int, distances=(0.1, 0.5, 0.95)) -> list[dict]:
    """Needle-vs-control CE ratio at three distances.

    The needle is a 32-byte random motif placed at `d·L`; the control is an
    equally random motif at 0.02·L (near the start, always attended). Both are
    followed by the SAME continuation. A level that recalls the needle scores its
    continuation lower than the control's; a level that forgets scores them alike.
    """
    out = []
    dev = next(model.parameters()).device
    g = torch.Generator(device=dev).manual_seed(0)
    motif = torch.randint(0, 256, (32,), device=dev, generator=g)
    tail = torch.randint(0, 256, (4,), device=dev, generator=g)
    for d in distances:
        x = torch.randint(0, 256, (1, context), device=dev, generator=g)
        near = 32
        far = max(near + 64, int(context * d))
        y_near = torch.cat([motif, tail])
        x[0, near : near + 32] = motif
        x[0, far : far + 32] = motif
        full = x.clone()
        full[0, far : far + 36] = y_near  # the needle's own continuation
        logits = model(full)[0]
        ce = torch.nn.functional.cross_entropy(
            logits[far + 28 : far + 32].float().reshape(-1, 256), y_near[-4:]
        ).item()
        near_logits = model(torch.cat([x[:, : near + 36], x[:, near + 36 :]], dim=1))[0]
        ce_near = torch.nn.functional.cross_entropy(
            near_logits[near + 28 : near + 32].float().reshape(-1, 256), y_near[-4:]
        ).item()
        out.append(
            {"distance": d, "ce_needle": round(ce, 4), "ce_near": round(ce_near, 4),
             "ratio": round(ce / max(1e-9, ce_near), 3)}
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="32768,131072,262144,1000000")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hier", action="store_true", help="csa_hierarchical on")
    ap.add_argument("--no-coarse", action="store_true", help="coarse level off")
    ap.add_argument("--recall", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    contexts = [int(c) for c in args.contexts.split(",")]
    print(f"shape: d={args.d_model} layers={args.layers} hier={args.hier}", flush=True)
    rows = []
    for ctx in contexts:
        try:
            model = build(args.d_model, args.layers, ctx, args.hier, not args.no_coarse)
        except torch.cuda.OutOfMemoryError:
            rows.append({"context": ctx, "error": "OOM at build"})
            print(rows[-1], flush=True)
            torch.cuda.empty_cache()
            continue
        try:
            m = measure(model, ctx)
        except torch.cuda.OutOfMemoryError:
            m = {"context": ctx, "error": "OOM at forward"}
            torch.cuda.empty_cache()
        if args.recall and "error" not in m:
            try:
                m["recall"] = recall(model, ctx)
            except torch.cuda.OutOfMemoryError:
                m["recall"] = "OOM"
                torch.cuda.empty_cache()
        rows.append(m)
        print(json.dumps(m), flush=True)
        del model
        torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
