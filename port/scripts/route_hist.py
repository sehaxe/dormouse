"""Routing histogram of a checkpoint: how many experts actually work.

    python scripts/route_hist.py --ckpt m400 --batch 4 --seq 1024

Why: the dropless batched bank needed 5.56 GiB at a fresh init and 12 GiB at the
150k checkpoint, which back-solves to ONE expert holding 3 800 and 8 192 tokens.
If true, the "512-expert" model has been doing the work of one or two, and every
MoE comparison we ran is measured on a partly dead bank. This prints the counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/home/sehaxe/dormouse/checkpoints")
    ap.add_argument("--ckpt", default="m400")
    ap.add_argument("--preset", default="configs/m400.toml")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="/home/sehaxe/dormouse/data_local/best/fwedu.bin")
    a = ap.parse_args()

    from dormouse.config import RunConfig
    from dormouse.model import DormouseModel

    base = Path(a.ckpt_dir) / a.ckpt
    # The SNAPSHOT is the run: rebuild from it so `diff_keys` is empty and the
    # histogram describes the model that actually trained (a preset edit would
    # otherwise be refused, correctly).
    run = RunConfig.from_snapshot((base / f"{a.ckpt}.config.toml").read_text())
    state = torch.load(base / f"{a.ckpt}.pt", map_location="cpu", weights_only=False)
    model = DormouseModel(run.model).to("cuda").eval()
    if "model" in state:
        model.load_state_dict(state["model"], strict=True)
    print(f"loaded {a.ckpt} at step {state.get('step')}")

    # REAL bytes: random ids hide a collapse that only real hidden states show
    # (measured 2026-11-04: random gave max 126/4096, the live run's bank needed
    # 12 GiB, i.e. one expert held ~4096 of 16384 assignments).
    raw = Path(a.data).read_bytes()
    off = a.seed * a.batch * a.seq
    chunk = raw[off : off + a.batch * a.seq + 1]
    ids = torch.tensor(list(chunk[:-1]), device="cuda", dtype=torch.long).view(a.batch, a.seq)
    targets = torch.tensor(list(chunk[1:]), device="cuda", dtype=torch.long).view(a.batch, a.seq)

    moe = model.loop.moe
    seen: list[torch.Tensor] = []
    # The router is a dense Linear on the block input; walk the model's own
    # entry point so the hidden state is exactly the training one.
    grabbed: list[torch.Tensor] = []
    # `router` is an nn.Linear the forward never CALLS (it uses `mm_mixed` on the
    # weight), so the module hook never fires — the MoE's own input is what the
    # router sees.
    moe.register_forward_pre_hook(
        lambda m, i: grabbed.append(i[0].detach().float().reshape(-1, i[0].shape[-1]))
    )
    with torch.no_grad():
        model(ids, targets=targets)

    counts_holder = getattr(moe, "_last_counts", None)
    if counts_holder is None:
        print("the arm did not record counts; falling back to a direct router probe")
        w = moe.router.weight
        x = grabbed[0].to(torch.bfloat16)
        logits = x @ w.t().to(torch.bfloat16)
        idx = logits.topk(moe.topk, dim=-1).indices
        counts = torch.bincount(idx.reshape(-1), minlength=moe.n_experts)
    else:
        counts = counts_holder
    c = counts[: moe.n_experts].float()
    total = c.sum().item()
    live = int((c > 0).sum().item())
    print(
        f"assignments {total:.0f} · live experts {live}/{moe.n_experts} "
        f"({100 * live / moe.n_experts:.1f}%) · max {c.max().item():.0f} "
        f"· mean {total / moe.n_experts:.2f}"
    )
    top = torch.topk(c, min(10, c.numel()))
    print("top-10 experts by assignments:")
    for rank, (val, ix) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
        print(f"  {rank:2}. expert {ix:4}  {val:8.0f}  {100 * val / max(1, total):5.1f}%")
    share = top.values.sum().item()
    print(f"\ntop-10 hold {100 * share / max(1, total):.1f}% of all tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
