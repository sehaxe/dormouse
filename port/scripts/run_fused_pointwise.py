#!/usr/bin/env python3
"""`dormouse.cli` with the fused pointwise seam installed — the A/B's FUSED arm.

The baseline arm is the plain CLI:

    .venv/bin/python -m dormouse.cli train configs/flagship3_dense.toml ...

and this file is the same command with one monkeypatch applied before the model
is built (`fused_pointwise.patch()` swaps `TransformerBlock._block` for a
verbatim copy of its body with the residual→ln2 seam fused). Nothing else
differs, so the two runs' ms/step difference IS the seam — and the copy makes the
shared `model/transformer.py` untouched until the proposal lands.
"""

from __future__ import annotations

import atexit
import sys


def main() -> None:
    from dormouse.model import fused_pointwise as fp

    fp.patch()
    atexit.register(lambda: print(f"[fused_pointwise] {fp.describe()}", file=sys.stderr))
    from dormouse.cli import main as cli

    cli()


if __name__ == "__main__":
    main()
