#!/usr/bin/env python3
"""The width A/B, on the REAL recipe — the one whose answer we can act on.

`ab_cheapwidth.py` screens on a handicapped configuration (dense readout via
`use_tsct=false`, `engram_ram=true`, the 230 GB `mix_big` plan). That is correct
for pricing `out_proj_rank` (which only exists on the dense readout) but it
cannot answer "does cheap width cost quality on the model we ship": the host
Engram alone measured **+0.83 BPB**, and the dense readout replaces the whole
TSCT bank's readout path.

This driver holds everything at the shipped recipe and varies ONLY the two
attention flags, which are dense `nn.Linear`s either way and therefore
independent of the rung:

    use_tsct=true · use_mhc=true · use_attnres=true · share_u=true
    engram in VRAM · max_iter=3 · use_msa=false · mix4/web.bin

Arms: `dense` (control) / `gqa` (n_kv_heads=2) / `latent` (d/4) / `latent+gqa`.

Rule, pre-registered BEFORE the first run: an arm is adopted iff its 3-seed mean
is within 2x the control's spread of the control (a tie is a win when the arm
cuts 23-42% of the dense MAC), and REJECTED if it is worse by more than that.
"""

from __future__ import annotations

import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path.home() / "logs" / "ab_width_clean"
DATA = "/home/sehaxe/dormouse/data_local/mix4/web.bin"
EVAL = "/home/sehaxe/dormouse/data_local/mix4/web_eval.bin"
D = 1024
SEEDS = (1, 2, 3)
STEPS = 600

COMMON = dict(
    steps=STEPS,
    batch=8,
    seq_len=512,
    data=DATA,
    eval_data=EVAL,
    eval=True,
    eval_every=300,
    eval_batches=10,
    log_every=300,
    ckpt_every=10**9,
    moe_kernel="triton",
    ns_kind="triton",
    ns_steps=5,
    opt="mix",
    muon_colrow_3d="colrow",
    opt_graph=False,
    max_iter=3,
    use_msa=False,
    use_mhc=True,
    share_u=True,
    use_attnres=True,
    use_engram=True,
)

ARMS = {
    "dense": dict(attn_latent_rank=0, n_kv_heads=0),
    "gqa": dict(attn_latent_rank=0, n_kv_heads=2),
    "latent": dict(attn_latent_rank=D // 4, n_kv_heads=0),
    "latent+gqa": dict(attn_latent_rank=D // 4, n_kv_heads=2),
}


def run(arm: str, seed: int) -> float | None:
    flags = dict(COMMON, **ARMS[arm], seed=seed, ckpt_name=f"wc_{arm}_s{seed}",
                 out_dir="/tmp/opencode/wc")
    cmd = [".venv/bin/dormouse", "train", "medium"]
    for k, v in flags.items():
        cmd += ["--set", f"{k}={v}"]
    log = OUT / f"{arm}_s{seed}.log"
    OUT.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if rc != 0:
        print(f"[fail] {arm} s{seed} rc={rc} — see {log}")
        return None
    lines = [ln for ln in log.read_text(errors="ignore").splitlines() if "bpb=" in ln]
    if not lines:
        print(f"[fail] {arm} s{seed}: no eval line")
        return None
    return float(lines[-1].split("bpb=")[1].split()[0])


def main() -> int:
    rows: dict[str, list[float]] = {}
    for arm in ARMS:
        for s in SEEDS:
            print(f"[run] {arm} seed {s}", flush=True)
            b = run(arm, s)
            if b is not None:
                rows.setdefault(arm, []).append(b)
                print(f"[done] {arm} s{s} bpb={b:.4f}", flush=True)
    ctl = rows.get("dense", [])
    if len(ctl) < 2:
        print("control incomplete — no verdict")
        return 1
    cm, cs = statistics.mean(ctl), max(ctl) - min(ctl)
    bar = 2 * cs
    print(f"\ncontrol dense n={len(ctl)} mean={cm:.4f} spread={cs:.4f} bar={bar:.4f}\n")
    print(f"{'arm':12s} {'n':>2s} {'mean':>7s} {'delta':>8s}  verdict")
    for arm, v in rows.items():
        if arm == "dense":
            continue
        m = statistics.mean(v)
        d = m - cm
        if abs(d) <= bar:
            verdict = "ADOPT (tie within bar)"
        elif d > 0:
            verdict = "REJECT (worse)"
        else:
            verdict = "ADOPT (better)"
        print(f"{arm:12s} {len(v):2d} {m:7.4f} {d:+8.4f}  {verdict}")
    (OUT / "verdict.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
