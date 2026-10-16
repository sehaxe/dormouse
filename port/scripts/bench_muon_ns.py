"""The bench for the two Newton–Schulz arms (research row 1 of the cost note).

    uv run python scripts/bench_muon_ns.py

Times `classic` against `gram` at the b1 factor shapes on CPU, counts the
matmul launches of each, and (ONLY when the card is idle — check it first,
`nvidia-smi --query-gpu=memory.used`) the same two on CUDA. The body lives in
`tests/test_muon_gram.py::bench_ns_kinds` so the equivalence gate, the bench and
`python tests/test_muon_gram.py` are one implementation; this file exists because
`scripts/` is where a bench is looked for.

What the numbers mean: the MAC column is the analytic arithmetic of each loop
(`macs`), the op columns are what actually launches. The research note predicts
the win is a LAUNCH/matmul-shape effect, not a FLOP effect — read both columns
against each other, and against the equivalence gate, which is the only thing
that licenses the swap.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from test_muon_gram import bench_ns_kinds  # noqa: E402

if __name__ == "__main__":
    bench_ns_kinds()
