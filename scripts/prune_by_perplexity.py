#!/usr/bin/env -S uv run --script
"""Score every window of a `.bin` with OUR OWN checkpoint and keep the middle band.

Why the middle: 2510.00866 ("Removing Noise, not Finding Gold") shows quality
filtering works by deleting noise, not by hunting for gold, and that
classifier-selected data can beat training on the "high-quality" set itself. So
this pruner drops BOTH tails — the windows the model already predicts far too
well (boilerplate, repeated lists) and the ones it cannot model at all (spam,
glued markup) — and keeps what is actually teachable.

Prior art for the mechanism: 2405.20541 measures ×1.45 fewer pretraining steps
from perplexity pruning with a **125M reference model**, in the over-trained and
data-constrained regimes — which is exactly our size class.

    # score 2.35 GB of web with the longctx1 checkpoint (forward only, ~5 h)
    uv run port/scripts/prune_by_perplexity.py \
        --bin   data_local/mix4/web.bin \
        --ckpt-dir checkpoints/longctx1 --ckpt-name longctx1 \
        --out-dir data_local/mix6 --window 4096 --context 512 --batch 32

Outputs, all under `--out-dir`:

    <name>.bin          the kept windows, concatenated with a blank line between
    <name>.bin.docs.json {version, size, starts} — one document per kept window
    <name>.manifest.json scores histogram, decile edges, per-decile bytes,
                        the band, and the exact byte accounting (in == kept + dropped)

The band is by *rank*, not by an absolute BPB: `--band-lo 0.10 --band-hi 0.90`
drops the bottom and top deciles of the score distribution, so the run cannot be
ruined by a global offset between checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dormouse.cli import _device, _model, _snapshot_run
from dormouse.instrument import Counters, loud
from dormouse.train.loop import load_model_half, load_model_state

LN2 = 0.6931471805599453


def _windows(n: int, w: int) -> list[tuple[int, int]]:
    return [(s, min(s + w, n)) for s in range(0, n, w)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, required=True)
    ap.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--ckpt-name", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--name", default="pruned")
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro", type=int, default=4, help="forward micro-batch (VRAM)")
    ap.add_argument(
        "--depth",
        type=int,
        default=1,
        help="loop depth for SCORING (0 = the checkpoint's max_iter). "
        "The band is by rank, so a cheaper depth only changes the "
        "scale of the scores, not which windows survive.",
    )
    ap.add_argument("--band-lo", type=float, default=0.10)
    ap.add_argument("--band-hi", type=float, default=0.90)
    ap.add_argument("--max-windows", type=int, default=0, help="0 = all (smoke test)")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    if not 0.0 <= a.band_lo < a.band_hi <= 1.0:
        loud(f"band [{a.band_lo}, {a.band_hi}] must satisfy 0 <= lo < hi <= 1")
    if a.window < 64 or a.context < 1:
        loud(f"window={a.window} context={a.context}: need window >= 64, context >= 1")

    raw = a.bin.read_bytes()
    n = len(raw)
    wins = _windows(n, a.window)
    if a.max_windows:
        wins = wins[: a.max_windows]
    print(f"{a.bin} · {n} B · {len(wins)} windows of {a.window} B (+{a.context} B context)")

    run = _snapshot_run(a.ckpt_dir, a.ckpt_name, eval_file=None, eval_batches=None)
    dev = _device(a.device)
    model = _model(run).to(dev).eval()
    load_model_state(model, run.train, load_model_half(a.ckpt_dir / f"{a.ckpt_name}.pt"))

    scores = np.empty(len(wins), dtype=np.float32)
    bytes_scored = np.empty(len(wins), dtype=np.int32)
    with torch.no_grad():
        for b0 in range(0, len(wins), a.batch):
            grp_all = wins[b0 : b0 + a.batch]
            for m0 in range(0, len(grp_all), a.micro):
                grp = grp_all[m0 : m0 + a.micro]
                seqs, spans = [], []
                for s, e in grp:
                    p0 = max(0, s - a.context)
                    seqs.append(raw[p0:e])
                    spans.append((s - p0, e - p0))  # the part we actually score
                tmax = max(len(x) for x in seqs)
                ids = torch.zeros(len(seqs), tmax, dtype=torch.long)
                for i, x in enumerate(seqs):
                    ids[i, : len(x)] = torch.frombuffer(bytearray(x), dtype=torch.uint8).long()
                ids = ids.to(dev)
                dep = None if a.depth == 0 else a.depth
                logits, _loss, _c = model(ids, counters=Counters(), depth=dep)
                logits = logits.view(len(seqs), tmax, -1).float()
                lp = F.log_softmax(logits[:, :-1], dim=-1)
                base = b0 + m0
                for i, (a0, a1) in enumerate(spans):
                    # predict bytes max(a0, 1)..a1-1 from their predecessors: the logit
                    # at position p-1 is the one that predicts the byte at p, and byte 0
                    # of the file has no predecessor to score it from.
                    lo_pos = max(a0, 1)
                    if lo_pos >= a1:
                        scores[base + i] = float("nan")
                        bytes_scored[base + i] = 0
                        continue
                    tgt = ids[i, lo_pos:a1]
                    got = lp[i, lo_pos - 1 : a1 - 1].gather(-1, tgt[:, None]).squeeze(-1)
                    scores[base + i] = float(-got.mean()) / LN2
                    bytes_scored[base + i] = int(a1 - a0)
            done = base + len(grp)
            if done % (a.batch * 20) == 0 or done == len(wins):
                ok = np.isfinite(scores[:done]) & (bytes_scored[:done] > 0)
                sofar = float(np.average(scores[:done][ok], weights=bytes_scored[:done][ok]))
                print(f"  {done}/{len(wins)} windows · mean BPB so far {sofar:.3f}", flush=True)

    lo = float(np.quantile(scores, a.band_lo))
    hi = float(np.quantile(scores, a.band_hi))
    keep = np.flatnonzero((scores >= lo) & (scores <= hi))

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out_bin = a.out_dir / f"{a.name}.bin"
    starts, parts, total = [], [], 0
    for idx in keep:
        s, e = wins[int(idx)]
        chunk = raw[s:e]
        if total:
            parts.append(b"\n\n")
            total += 2
        starts.append(total)
        parts.append(chunk)
        total += len(chunk)
    out_bin.write_bytes(b"".join(parts))
    (a.out_dir / f"{a.name}.bin.docs.json").write_text(
        json.dumps({"version": 1, "size": total, "starts": starts})
    )

    hist, edges = np.histogram(scores, bins=10)
    per_dec = {f"[{edges[i]:.3f},{edges[i + 1]:.3f})": int(hist[i]) for i in range(len(hist))}
    manifest = {
        "source": str(a.bin),
        "source_bytes": n,
        "window": a.window,
        "context": a.context,
        "checkpoint": a.ckpt_name,
        "band": [a.band_lo, a.band_hi],
        "band_bpb": [lo, hi],
        "windows_scored": len(wins),
        "windows_kept": int(len(keep)),
        "kept_bytes": total,
        "dropped_bytes": n - total,
        "fraction_kept": total / n if n else 0.0,
        "mean_bpb_all": float(np.average(scores, weights=bytes_scored)),
        "mean_bpb_kept": float(scores[keep].mean()) if len(keep) else None,
        "score_deciles": per_dec,
    }
    (a.out_dir / f"{a.name}.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "windows_kept",
                    "kept_bytes",
                    "fraction_kept",
                    "band_bpb",
                    "mean_bpb_all",
                    "mean_bpb_kept",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
