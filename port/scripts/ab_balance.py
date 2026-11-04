"""A/B the routing controller on REAL bytes: does the bank come alive?

    uv run --extra gpu python scripts/ab_balance.py --steps 2000          # both arms
    uv run --extra gpu python scripts/ab_balance.py --arms prop --steps 120   # smoke

**Do not run this while the card is busy** (AGENTS §1.5 — one heavy thing at a
time). `--steps 2000` at `configs/m400.toml` (b4 × s1024) is a real training run
per arm, hours not minutes; the `--steps` knob exists for the smoke length.

The finding this exists for (`docs/reviews/2026-11-04-routing-collapse-CONFIRMED.md`):
`m400` at step 149 999 carries a saved `lfb_bias` spanning `[-218.27, +72.67]`
(mean `+34.30`), and its router keeps **9 of 512 experts** alive — one holding
4096 of 8192 assignments. The bias is added to the router logits
unconditionally, so those ±200 logits decide the routing, and the router's own
scores are ±5. This driver compares, on the same bytes and the same budget:

* `sign`  — the control: LFB's cumulative bang-bang step (`moe_lb_coef=0.01`,
  the preset's). The legacy arm; the divergence was measured on it.
* `prop`  — `b -= gamma·(L_e−mean L)/max(1,mean L)` on THIS window's histogram,
  clamped to ±`moe_lb_clamp` (`gamma=0.001`, clamp `2.0`).
* `aux`   — no bias controller; the Switch/GShard term `0.01·E·Σ f·P` added to
  the loss (its gradient reaches the router and nothing else). Toggle with
  `--arms sign,aux`.

Each arm is a FRESH run (a resume refuses a controller change — the snapshot is
the identity) with `moe_routing_health=true`, so every log line carries
`moe_route_live` / `moe_route_max_x1000` / `moe_route_top10_x1000` /
`moe_route_ent_x1000` summed over the window; divide by `moe_route_n` for the
mean. At the end the script probes ONE batch of real bytes and prints the exact
histogram of the model that just trained (`_last_counts`, the same tensor the
health counters read).

**Acceptance (written before the run):** live experts ≥ 80% of E **and** top-10
share ≤ 30% after 2 000 steps, with the bias inside its clamp. A run below either
number failed the arm; a loss curve is not evidence for this question.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]


def arms(**over) -> dict:
    """The three arms as MODEL overrides; the control carries none."""
    return {
        "sign": {"moe_lb_mode": "sign"},
        "prop": {"moe_lb_mode": "prop", "moe_lb_gamma": 0.001, "moe_lb_clamp": 2.0},
        "aux": {"moe_lb_mode": "aux", "moe_aux_coef": 0.01},
    }


def probe(model, data: str, *, batch: int, seq: int, seed: int = 0) -> dict:
    """One real-bytes forward → the routing histogram the model actually runs.

    The bytes are read with a SEEK, not `read_bytes()`: the corpus is far bigger
    than the one window this needs.
    """
    import torch

    off = seed * batch * seq
    with open(data, "rb") as fh:
        fh.seek(off)
        chunk = fh.read(batch * seq + 1)
    assert len(chunk) == batch * seq + 1, f"{data}: {off}+{batch * seq + 1} past EOF"
    ids = torch.tensor(list(chunk[:-1]), dtype=torch.long).view(batch, seq)
    targets = torch.tensor(list(chunk[1:]), dtype=torch.long).view(batch, seq)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        model(ids.to(device), targets=targets.to(device))
    moe = model.loop.moe
    counts = moe._last_counts.float()
    total = float(counts.sum())
    live = int((counts > 0).sum())
    top = counts.topk(min(10, counts.numel())).values.sum()
    p = counts / counts.sum().clamp_min(1)
    ent = float(
        -(p * p.clamp_min(1e-30).log()).sum() / torch.log(torch.tensor(float(counts.numel())))
    )
    bias = moe.lfb_bias
    return {
        "live": live,
        "experts": int(counts.numel()),
        "assignments": int(total),
        "max_share": float(counts.max()) / max(total, 1.0),
        "top10_share": float(top) / max(total, 1.0),
        "entropy": ent,
        "bias_min": float(bias.min()),
        "bias_max": float(bias.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=str(ROOT / "configs/m400.toml"))
    ap.add_argument("--data", default=str(ROOT / "data_local/best/fwedu.bin"))
    ap.add_argument("--arms", default="sign,prop,aux")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--out", default="/tmp/opencode/ab_balance")
    a = ap.parse_args()

    import torch

    from dormouse.config import RunConfig, TrainConfig, load_preset, validate
    from dormouse.model import DormouseModel
    from dormouse.train import train

    if not torch.cuda.is_available():
        print("no CUDA device: this driver is a GPU run (the b4 x s1024 shape)")
        return 2
    names = [n.strip() for n in a.arms.split(",") if n.strip()]
    known = arms()
    unknown = [n for n in names if n not in known]
    if unknown:
        print(f"unknown arm(s) {unknown}; known {sorted(known)}")
        return 2

    rows = []
    for name in names:
        model_over = dict(known[name], moe_routing_health=True)
        model_cfg = load_preset(a.preset)
        model_cfg = dataclasses.replace(
            model_cfg, max_seq_len=max(a.seq, model_cfg.max_seq_len), **model_over
        )
        run = RunConfig(
            source=a.preset,
            model=model_cfg,
            train=dataclasses.replace(
                TrainConfig(),
                steps=a.steps,
                batch=a.batch,
                seq_len=a.seq,
                data=a.data,
                log_every=100,
                ckpt_every=a.steps,
                eval=False,
                ckpt_name=name,
                out_dir=str(Path(a.out) / name),
            ),
        )
        validate(run.model)
        print(f"\n=== {name}: {model_over} · {a.steps} steps · b{a.batch} x s{a.seq} ===")
        t0 = time.perf_counter()
        model = DormouseModel(run.model, device="cuda")
        report = train(model, run, device="cuda", log=print)
        dt = time.perf_counter() - t0
        hist = probe(model, a.data, batch=a.batch, seq=a.seq)
        n = report["counters"].get("moe_route_n") or 0
        health = (
            {
                k: report["counters"].get(k, 0) / n
                for k in (
                    "moe_route_live",
                    "moe_route_max_x1000",
                    "moe_route_top10_x1000",
                    "moe_route_ent_x1000",
                )
            }
            if n
            else {}
        )
        rows.append((name, hist, health, dt, report))
        print(
            f"{name}: {hist['live']}/{hist['experts']} live · "
            f"max {100 * hist['max_share']:.1f}% · top10 {100 * hist['top10_share']:.1f}% · "
            f"ent {hist['entropy']:.3f} · bias [{hist['bias_min']:+.2f}, {hist['bias_max']:+.2f}] "
            f"· {dt / max(a.steps, 1):.2f} s/step"
        )
        if health:
            print(
                f"   window means: live {health['moe_route_live']:.1f} · "
                f"max {health['moe_route_max_x1000'] / 10:.1f}% · "
                f"top10 {health['moe_route_top10_x1000'] / 10:.1f}% · "
                f"ent {health['moe_route_ent_x1000'] / 1000:.3f}"
            )

    print("\narm     live/E      max%   top10%   ent   bias[min,max]        loss")
    for name, hist, _health, _dt, report in rows:
        ok = hist["live"] >= 0.8 * hist["experts"] and hist["top10_share"] <= 0.30
        print(
            f"{name:5} {hist['live']:4}/{hist['experts']:<4} "
            f"{100 * hist['max_share']:7.1f} {100 * hist['top10_share']:8.1f} "
            f"{hist['entropy']:6.3f} [{hist['bias_min']:+8.2f},{hist['bias_max']:+8.2f}] "
            f"{report['loss']:.4f} {'PASS' if ok else 'FAIL'}"
        )
    print("\nacceptance: live >= 80% of E AND top-10 share <= 30%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
