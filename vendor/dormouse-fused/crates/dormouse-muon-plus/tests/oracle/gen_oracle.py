#!/usr/bin/env python3
"""Generate `muon_oracle.bin` — the expected values for `tests/muon_oracle.rs`.

WHAT THIS IS. A tier-(a) oracle for `dormouse-muon-plus`: the expected values are
produced by RUNNING the Muon+ AUTHORS' own implementation, which this paper
names in its own abstract.

    repo    https://github.com/K1seki221/MuonPlus
    commit  8a9ace123afedaab8ba75ea0b19315594ae1da7c   (HEAD; last push 2026-02-26)
    files   utils/optim/muon_plus.py, utils/optim/polar_express.py
    paper   arXiv:2602.21545 v1 (2026-02-25) and v2 (2026-02-26) abstract:
            "We provide our code here: https://github.com/K1seki221/MuonPlus."
            v3 (2026-05-14) DROPPED that sentence from the abstract - the repo
            is still the only code the paper ever named, and this header is
            where the tier-(a) provenance lives for v3.

The two files under `upstream/` are BYTE-IDENTICAL copies of that commit. The
sha256 of each is asserted below before anything is imported, so the pin
cannot rot silently: a rebase of the pin fails this script, loudly.

WHAT IS AND IS NOT GATED, and why - measured, not assumed. See
`transcript.txt` for the runs and `tests/muon_oracle.rs` for the bars.

  GATED, zero tolerance, by parsing the pinned SOURCE in the Rust test:
    - `a, b, c = (3.4445, -4.7750, 2.0315)` at upstream muon_plus.py:56.
      A transcription slip in a 4-significant-figure constant is the actual
      risk here, and a literal comparison catches it exactly.

  GATED, numerically, against the authors' `apply_post_polar_norm`
    (upstream muon_plus.py:87-120) at 1e-5. This is the layer that catches the
    axis / composition-order class.

  NOT GATED numerically: `zeropower_via_newtonschulz5` (muon_plus.py:53-67).
    The authors run Newton-Schulz in bfloat16 (`X = G.bfloat16()`); we run f32.
    Measured max|authors_bf16 - same arithmetic in f32| is 2.3e-2 to 2.1e-1
    over shapes x steps, while a 1e-4 RELATIVE error in `a` moves the output by
    only 1.5e-4 to 5.8e-4. Signal/noise is 0.002 to 0.02: the authors' dtype
    error is 50x to 5000x LARGER than the defect a numeric gate would be looking
    for, so any bar that passes is above the signal and catches nothing. The
    constant is gated instead, at zero tolerance. This is the honest ceiling
    and it is a real limit, not a gap left for later.

RUN
    # The venv lives OUTSIDE the crate: tools/oracle_gate.py walks tests/ for *.py,
    # so a venv here is 5,509 unregistered files and 5,516 gate violations.
    uv venv --python 3.12 /tmp/muon-oracle-venv
    uv pip install --python /tmp/muon-oracle-venv/bin/python torch numpy --index-url https://download.pytorch.org/whl/cpu
    /tmp/muon-oracle-venv/bin/python tests/oracle/gen_oracle.py
"""
import hashlib
import os
import struct
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # @torch.compile -> eager; same arithmetic

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "upstream")

# sha256 of the files as downloaded from
# https://codeload.github.com/K1seki221/MuonPlus/tar.gz/8a9ace123afedaab8ba75ea0b19315594ae1da7c
# Measured 2026-09-30. A mismatch is a hard error, never a warning.
PINNED_SHA256 = {
    "utils/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "utils/optim/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "utils/optim/muon_plus.py": "665ea6cb1f6f631f7839c430f2f2c8cbd62dd7e6b8a24fb752f1211c57ab2cab",
    "utils/optim/polar_express.py": "ac792c2da2eab161be56e06f2c9804adfc2918bfa954a551ef5b5e9b43f6a871",
}

# (m, n). min(m, n) >= 8 only: see the eps-placement note in the test file.
# The 4x3 case is the one degenerate input kept, because an exactly-zero column
# and an exactly-zero row are reachable from a masked gradient and both
# implementations must map them to exactly zero (measured: max|diff| = 0.0).
# `zeroed` lists the (row, col) positions to zero first.
CASES = [
    (16, 8, ()),
    (32, 8, ()),
    (64, 16, ()),
    (8, 8, ()),
    (8, 32, ()),
    (16, 16, ()),
    (128, 64, ()),
    (4, 3, ((0, 1), (1, 1), (2, 1), (3, 1))),  # column 1 all zero
]
MODES = ["col", "row", "col_row", "row_col"]  # upstream muon_plus.py:87-120


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def grid(m, n):
    """Deterministic, so a failure names the same numbers every run. Same
    formula as `tests/self_checks.rs`, but the fixture SHIPS the input, so the
    two sides cannot disagree about it."""
    return [((i * 2654435761) % 997) / 498.0 - 1.0 for i in range(m * n)]


def main():
    # ---- 1. the pin must be intact before anything is imported ------------
    for rel, want in sorted(PINNED_SHA256.items()):
        p = os.path.join(UPSTREAM, rel)
        got = sha256(p)
        if got != want:
            sys.stderr.write(
                "gen_oracle: PIN ROTED for %s\n  expected %s\n  got      %s\n"
                "The upstream pin is supposed to be byte-identical. Re-fetch\n"
                "https://codeload.github.com/K1seki221/MuonPlus/tar.gz/"
                "8a9ace123afedaab8ba75ea0b19315594ae1da7c and re-check.\n"
                % (rel, want, got)
            )
            return 2
    print("pin: 4 files verified byte-identical to K1seki221/MuonPlus@8a9ace12")

    sys.path.insert(0, UPSTREAM)
    import torch  # noqa: E402
    from utils.optim.muon_plus import apply_post_polar_norm  # noqa: E402

    print("torch %s, eager (@torch.compile disabled)" % torch.__version__)

    # ---- 2. the cases ------------------------------------------------------
    out = [b"BMUO", struct.pack("<II", 1, len(CASES) * len(MODES))]
    for (m, n, zeroed) in CASES:
        vals = grid(m, n)
        for (i, j) in zeroed:
            vals[i * n + j] = 0.0
        x = torch.tensor(vals, dtype=torch.float32).reshape(m, n)
        for mode in MODES:
            ref = apply_post_polar_norm(x, mode, 1e-8)
            out.append(struct.pack("<IIII", m, n, MODES.index(mode), 0))
            out.append(struct.pack(f"<{m * n}f", *x.flatten().tolist()))
            out.append(struct.pack(f"<{m * n}f", *ref.flatten().tolist()))
            print(
                "  %3dx%-3d %-8s in|max|=%.4f  out|max|=%.4f"
                % (m, n, mode, x.abs().max(), ref.abs().max())
            )

    blob = b"".join(out)
    dest = os.path.join(HERE, "muon_oracle.bin")
    with open(dest, "wb") as fh:
        fh.write(blob)
    print("wrote %s (%d bytes, %d cases)" % (dest, len(blob), len(CASES) * len(MODES)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
