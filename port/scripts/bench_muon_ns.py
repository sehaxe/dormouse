"""The bench for the two Newton–Schulz arms (research row 1 of the cost note).

    uv run python scripts/bench_muon_ns.py            # the two NS arms
    uv run python scripts/bench_muon_ns.py step       # the batched Muon step, b1

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

`step` is the 2026-10-16 Muon-speed gate: the batched optimizer step at the b1
preset's OWN Muon shape groups (read off the model on `meta`, so it tracks the
preset), old implementation vs new, wall + `torch.profiler` CUDA time + aten
dispatch counts + peak VRAM. It builds parameters and gradients only — no
forward, no backward, no AdamW — so the number it reports is exactly the Muon
batched step the profile prices at 530-850 ms of a b8 s1024 step. The reference
implementation is `tests/test_muon_fast.py::old_batched_step`, so the gate and
the bench measure the same two programs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from test_muon_fast import bench_muon_b1_step  # noqa: E402
from test_muon_gram import bench_ns_kinds  # noqa: E402

if __name__ == "__main__":
    if "step" in sys.argv[1:]:
        bench_muon_b1_step()
    else:
        bench_ns_kinds()
