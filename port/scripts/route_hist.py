"""Routing histogram of a checkpoint: how many experts actually work.

    python scripts/route_hist.py --ckpt m400 --batch 4 --seq 1024

Why: the dropless batched bank needed 5.56 GiB at a fresh init and 12 GiB at the
150k checkpoint, which back-solves to ONE expert holding 3 800 and 8 192 tokens.
If true, the "512-expert" model has been doing the work of one or two, and every
MoE comparison we ran is measured on a partly dead bank. This prints the counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402


def summary(counts, n_experts: int, **extra) -> dict:
    """The one routing-health record: counters the dashboard reads, no torch.

    `route_hist.py --json PATH` writes this shape so the panel can show the LAST
    histogram instead of paying for another model load. `live_experts` is the
    field the restart planner compares against E.
    """
    c = [int(v) for v in counts[:n_experts]]
    total = sum(c)
    live = sum(1 for v in c if v > 0)
    top = sorted(c, reverse=True)[:10]
    top_share = sum(top) / total if total else 0.0
    max_share = max(c) / total if total and c else 0.0
    return {
        "n_experts": int(n_experts),
        "assignments": total,
        "live_experts": live,
        "live_frac": round(live / max(1, n_experts), 6),
        "max_share": round(max_share, 6),
        "max_share_x100": round(100.0 * max_share, 2),
        "top10_share": round(top_share, 6),
        "top10_share_x100": round(100.0 * top_share, 2),
        "counts": c,
        **extra,
    }


def write_json(path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/home/sehaxe/dormouse/checkpoints")
    ap.add_argument("--ckpt", default="m400")
    ap.add_argument("--preset", default="configs/m400.toml")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="/home/sehaxe/dormouse/data_local/best/fwedu.bin")
    ap.add_argument(
        "--json",
        metavar="PATH",
        help="write the histogram as JSON (the dashboard panel reads this file)",
    )
    a = ap.parse_args()

    from dormouse.config import RunConfig
    from dormouse.model import DormouseModel
    from dormouse.train.loop import load_model_half, load_model_state

    base = Path(a.ckpt_dir) / a.ckpt
    # The SNAPSHOT is the run: rebuild from it so `diff_keys` is empty and the
    # histogram describes the model that actually trained (a preset edit would
    # otherwise be refused, correctly).
    run = RunConfig.from_snapshot((base / f"{a.ckpt}.config.toml").read_text())
    state = torch.load(base / f"{a.ckpt}.pt", map_location="cpu", weights_only=False)
    model = DormouseModel(run.model).to("cuda").eval()
    # THE ONE LOADER, not a raw `load_state_dict`: the LFB bias is a
    # non-persistent buffer that travels in the checkpoint's `moe` section, and a
    # probe that read `state["model"]` alone measured the BIAS-LESS router
    # ("80 of 512 alive") while the file's own router had 9 (measured 2026-11-04).
    weights = base / f"{a.ckpt}.pt"
    load_model_state(model, run.train, load_model_half(weights))
    bias = model.loop.moe.lfb_bias
    print(
        f"loaded {a.ckpt} at step {state.get('step')} · lfb_bias "
        f"[{float(bias.min()):+.3f}, {float(bias.max()):+.3f}]"
    )

    # REAL bytes: random ids hide a collapse that only real hidden states show
    # (measured 2026-11-04: random gave max 126/4096, the live run's bank needed
    # 12 GiB, i.e. one expert held ~4096 of 16384 assignments).
    # A SEEK, not `read_bytes()`: the corpus is 12 GB and this needs one window.
    off = a.seed * a.batch * a.seq
    with open(a.data, "rb") as fh:
        fh.seek(off)
        chunk = fh.read(a.batch * a.seq + 1)
    assert len(chunk) == a.batch * a.seq + 1, f"{a.data}: {off}+window past EOF"
    ids = torch.tensor(list(chunk[:-1]), device="cuda", dtype=torch.long).view(a.batch, a.seq)
    targets = torch.tensor(list(chunk[1:]), device="cuda", dtype=torch.long).view(a.batch, a.seq)

    moe = model.loop.moe
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
        idx = (logits + bias).topk(moe.topk, dim=-1).indices
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
    live_bias = bias.detach().float()
    print("top-10 experts by assignments:")
    pairs = zip(top.values.tolist(), top.indices.tolist(), strict=True)
    for rank, (val, ix) in enumerate(pairs, 1):
        print(f"  {rank:2}. expert {ix:4}  {val:8.0f}  {100 * val / max(1, total):5.1f}%")
    share = top.values.sum().item()
    print(f"\ntop-10 hold {100 * share / max(1, total):.1f}% of all tokens")
    print(
        f"bias range {live_bias.min():.2f} … {live_bias.max():.2f} "
        f"· |bias|>2 on {(live_bias.abs() > 2).sum().item()}/{moe.n_experts} experts"
    )
    if a.json:
        payload = summary(
            counts[: moe.n_experts].tolist(),
            moe.n_experts,
            ckpt=a.ckpt,
            step=state.get("step"),
            batch=a.batch,
            seq=a.seq,
            bias_min=float(live_bias.min()),
            bias_max=float(live_bias.max()),
            bias_over_2=int((live_bias.abs() > 2).sum().item()),
        )
        write_json(a.json, payload)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
