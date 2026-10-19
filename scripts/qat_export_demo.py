"""The ternary release in one file: export a QAT-fixed model, reload it, verify.

    uv run python scripts/qat_export_demo.py --preset small --ckpt-name sft7 --out-dir checkpoints
    uv run python scripts/qat_export_demo.py --preset tiny --fresh          # CPU/CI path

WHAT IT DOES, in order: load the model (a checkpoint, or `--fresh` from the
preset), force the QAT projection at `alpha=1` (no training — the projection IS
the export's numerics), export to `<out-dir>/<name>.dormouse`, reload the file
into a fresh model, forward one batch through both, and print the size table
plus the max `|Δlogits|`. CPU-capable by design: `--preset tiny --fresh` runs
this end to end in CI; the real use is a GPU checkpoint later.

WHAT THE FILE BUYS, and what it does not: disk/RAM at rest (a ternary factor is
`n/4 + 2` bytes against bf16's `2n`). This port has no int8/LUT ternary kernel,
so the loaded model still runs bf16 matmuls — the file buys storage, not GPU
speed.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dormouse.config import RunConfig, load_preset  # noqa: E402
from dormouse.instrument import loud  # noqa: E402
from dormouse.model import DormouseModel  # noqa: E402
from dormouse.model.export import factor_qat_mode, verify_export, write_export  # noqa: E402
from dormouse.model.model import param_count  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="export a QAT-fixed model as the packed ternary release"
    )
    p.add_argument("--preset", default="small", help="preset NAME or .toml PATH")
    p.add_argument(
        "--ckpt-name", default="", help="checkpoint under --out-dir (required unless --fresh)"
    )
    p.add_argument("--out-dir", default="checkpoints", help="checkpoint directory")
    p.add_argument("--out", default="", help="export path (default <out-dir>/<name>.dormouse)")
    p.add_argument("--mode", default="ternary", choices=["ternary", "mxfp4"])
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--fresh",
        action="store_true",
        help="build from --preset instead of loading a checkpoint (the CPU/CI path)",
    )
    p.add_argument("--batch", type=int, default=2)
    p.add_argument(
        "--seq-len", type=int, default=64, help="verification window (capped at max_seq_len)"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fresh:
        cfg = load_preset(args.preset)
        model = DormouseModel(cfg, device=args.device)
        source = f"fresh model from preset {args.preset}"
    else:
        if not args.ckpt_name:
            loud("--ckpt-name is required unless --fresh (the export needs weights)")
        out_dir = Path(args.out_dir)
        weights = out_dir / f"{args.ckpt_name}.pt"
        snapshot = out_dir / f"{args.ckpt_name}.config.toml"
        if not weights.is_file() or not snapshot.is_file():
            loud(
                f"{weights} and {snapshot} must both exist: the snapshot is what "
                f"says which model the checkpoint is. Escape: fix --out-dir/"
                f"--ckpt-name, or use --fresh."
            )
        run = RunConfig.from_snapshot(snapshot.read_text())
        cfg = run.model
        model = DormouseModel(cfg, device=args.device)
        from dormouse.train.loop import load_model_half, load_model_state

        load_model_state(model, run.train, load_model_half(weights))
        source = f"{weights} (its snapshot decides the config)"
    name = args.ckpt_name or args.preset
    out = Path(args.out) if args.out else Path(args.out_dir) / f"{name}.dormouse"
    print(f"source: {source} · {param_count(model):,} params")

    # The QAT projection at alpha=1, no training: the network the file claims
    # to be. `verify_roundtrip` builds its reference forward from this state.
    with factor_qat_mode(model, args.mode, 1.0):
        report = write_export(model, out, mode=args.mode, preset=args.preset)
        print(report.table())
        seq = min(args.seq_len, cfg.max_seq_len)
        gen = torch.Generator().manual_seed(0)
        x = torch.randint(0, cfg.vocab, (args.batch, seq), generator=gen)
        diff = verify_export(model, out, (x,))
    print(
        f"verify: max |Δlogits| = {diff:.3e} over input {tuple(x.shape)} "
        f"(reference = the projection on the fp32 masters, loaded = the file)"
    )
    if not math.isfinite(diff):
        loud("non-finite logits: the loaded model is not the exported one")
    print(
        "at rest only: no int8/LUT ternary kernel in this port, so the loaded "
        "model still runs bf16 matmuls — the file buys disk/RAM, not GPU speed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
