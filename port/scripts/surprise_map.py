"""The model's own surprise, byte by byte — the landing's hero image.

Not a stock illustration: a 64x16 field where each cell is one BYTE of held-out
English and its colour is `-log2 p(byte)` from the flagship3 checkpoint. The
same 1024 bytes are printed below it, so a reader can see exactly which byte
cost the model what.

    PORT/.venv/bin/python scripts/surprise_map.py \
        --ckpt-dir /tmp/opencode/ck --ckpt-name flagship3 \
        --offset 100000000 --bytes 1024 --out ../site/assets/surprise.json

CPU on purpose: the card is usually busy with a training run, and one 1024-byte
forward of a 95M model is a few seconds on a core.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from dormouse.cli import _device, _model, _snapshot_run
from dormouse.train.loop import load_model_half, load_model_state

LN2 = 0.6931471805599453


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--ckpt-name", required=True)
    ap.add_argument("--dataset", type=Path, default=Path("../data_local/best/fwedu.bin"))
    ap.add_argument("--offset", type=int, default=100_000_000)
    ap.add_argument("--bytes", type=int, default=1024)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    run = _snapshot_run(a.ckpt_dir, a.ckpt_name, eval_file=str(a.dataset), eval_batches=None)
    run.train = dataclasses.replace(
        run.train, eval_offset=a.offset, eval_batches=None, eval_bytes=a.bytes
    )
    seq = run.train.seq_len
    if a.bytes % seq:
        raise SystemExit(f"--bytes must be a multiple of seq_len={seq}")

    with open(a.dataset, "rb") as fh:
        fh.seek(a.offset)
        raw = fh.read(a.bytes + 1)
    if len(raw) < a.bytes + 1:
        raise SystemExit("dataset is shorter than the window")

    dev = _device(a.device)
    model = _model(run).to(dev)
    load_model_state(model, run.train, load_model_half(a.ckpt_dir / f"{a.ckpt_name}.pt"))
    model.eval()

    rows = []
    with torch.no_grad():
        for start in range(0, a.bytes, seq):
            chunk = raw[start : start + seq]
            nxt = raw[start + 1 : start + seq + 1]
            ids = torch.tensor(list(chunk), dtype=torch.long, device=dev)[None]
            tgt = torch.tensor(list(nxt), dtype=torch.long, device=dev)[None]
            logits, _, _ = model(ids)
            ce = F.cross_entropy(logits, tgt.reshape(-1), reduction="none")
            rows.extend((ce / LN2).tolist())

    # Printable projection of the same bytes: a high byte would render as a
    # tofu box, so anything outside 32..126 becomes '·'. The COLOURS keep the
    # true byte-value surprise; only the printed glyph is substituted.
    text = "".join(chr(b) if 32 <= b <= 126 else "·" for b in raw[: a.bytes])
    out = {
        "ckpt": a.ckpt_name,
        "offset": a.offset,
        "bytes": a.bytes,
        "cols": 64,
        "text": text,
        "bits": [round(v, 3) for v in rows],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, separators=(",", ":")))
    lo, hi = min(rows), max(rows)
    print(
        f"wrote {a.out} · {len(rows)} bytes · bits {lo:.2f}..{hi:.2f} · "
        f"mean {sum(rows) / len(rows):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
