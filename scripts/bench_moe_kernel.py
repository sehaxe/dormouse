"""The b1 bench for the Triton grouped-GEMM expert pass (research step 3/3).

    uv run --extra gpu python scripts/bench_moe_kernel.py

The numbers this file prints are the ones the commit and
`docs/research/2026-10-15-tsct-super-kernel.md` op 3 are judged on: parity
against the cap-padded arm, the two arms' fwd and fwd+bwd medians interleaved in
one process, and the peak allocated VRAM of each (the whole point of removing
the `[E, cap, f]` round trip).

The bodies live in `tests/test_moe_kernel.py` (`bench_small`, `bench_b1`) so the
same code is what the test module runs under `python tests/test_moe_kernel.py`;
this file exists because `scripts/` is where a bench is looked for. Machine
discipline first (AGENTS §1.5): no `train` process, card < 1 GiB, one GPU thing
at a time.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from test_moe_kernel import bench_b1, bench_small  # noqa: E402

if __name__ == "__main__":
    bench_small()
    bench_b1()
