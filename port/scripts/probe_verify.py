"""20 steps of `final_byte` at one shape, weights dumped for an A/B.

    uv run --extra gpu python scripts/probe_verify.py --shape b8s512 --graph 0 --out /tmp/a.pt
    uv run --extra gpu python scripts/probe_verify.py --shape b8s512 --graph 1 --out /tmp/b.pt

`--graph 1` is `full_graph=true`; the run is the REAL trainer (warmup, capture,
replay), not a re-implementation of it.
"""

from __future__ import annotations

import argparse

import torch

from dormouse.config import load_run
from dormouse.model import DormouseModel
from dormouse.train.data import ByteStream
from dormouse.train.loop import train

SHAPES = {"b8s512": (8, 512), "b24s1024": (24, 1024)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="b8s512", choices=sorted(SHAPES))
    ap.add_argument("--graph", type=int, default=0)
    ap.add_argument(
        "--nocapture",
        action="store_true",
        help="full_graph's eager program with the capture DISABLED (COUNTED fallback): "
        "isolates the capture from the arm's own non-capture deltas (device LR, "
        "AdamW capturable=True)",
    )
    ap.add_argument("--force", action="store_true", help="ignore the headroom rule")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--data", default="/tmp/opencode/cap/train.bin")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    batch, seq = SHAPES[a.shape]
    run = load_run(
        "final_byte",
        train_overrides=dict(
            batch=batch,
            seq_len=seq,
            steps=a.steps,
            eval=False,
            log_every=10**9,
            ckpt_every=10**9,
            data=a.data,
            muon_state_dtype="bf16",
            ns_dtype="bf16",
            grad_clip=1.0,
            full_graph=bool(a.graph),
        ),
        model_overrides=dict(bf16=True),
    )
    if a.force:
        from dormouse.train import graph as graph_mod

        graph_mod.CAPTURE_HEADROOM_FLOOR = 0
        graph_mod.CAPTURE_HEADROOM_SHARE = 0.0
    if a.nocapture:
        from dormouse.train import graph as graph_mod

        def _no(*_a, **_k):
            raise RuntimeError("capture disabled by --nocapture")

        graph_mod.FullGraph.capture = _no
    torch.manual_seed(0)
    model = DormouseModel(run.model, device="cuda")
    stream = ByteStream(a.data, batch, seq)
    report = train(model, run, stream=stream, device="cuda", log=lambda *x, **k: None)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, a.out)
    print(f"{a.shape} graph={a.graph} steps={report['steps']} loss={report['loss']:.6f} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
