#!/usr/bin/env python3
"""Generate the burn-byteflow FIELD oracle — the gut-level field compare.

    $ /home/sehaxe/oracle-venv/bin/python oracle/gen_byteflow_field_oracle.py \\
          > ../fixtures/byteflow_field_oracle.txt

TIER (b), same as `gen_byteflow_oracle.py`: no author code exists (pdf p.19,
"as soon as [the legal review] process is complete" — the brief itself records
the empty GitHub search). This file runs TORCH cpu as the arithmetic oracle
for ONE field case, Canon eq. (10) (pdf p.4):

    Canon(h_t) = w0⊙h_t + w1⊙h_{t-1} + w2⊙h_{t-2} + w3⊙h_{t-3}

with a lookahead of 3 zeros at the left edge — the straightforward reading,
and the same boundary rule the Rust satsifies by its own left-pad:

    Canon.gates[c][k] is DETERMINISTIC-FORMULA, NOT random — the possibility
    of a hidden rng asymmetry between torch (cpu, f64) and the Rust side is
    killed by having no rng in the fixture at all.

SIXTEEN significant digits (%.17g) for the outputs, because the f32 fixture
values are read by the Rust and the bar there (2e-7 absolute at these small
magnitudes) leaves no room to hide a padding or tap-order transcription slip.

Usage: python3 gen_byteflow_field_oracle.py [--out ../fixtures/byteflow_field_oracle.txt]
"""

import argparse
import math
import sys
from pathlib import Path

import torch

T = 12
D = 8


def canon_gates() -> torch.Tensor:
    """Distinct per (channel, tap), deterministic — no RNG anywhere."""
    w = torch.empty(4, D, dtype=torch.float64)
    for k in range(4):
        for c in range(D):
            w[k, c] = 0.125 * (k + 1) * math.cos(0.7 * (c + k) + 0.3)
    return w


def canon_input() -> torch.Tensor:
    h = torch.empty(T, D, dtype=torch.float64)
    for t in range(T):
        for c in range(D):
            h[t, c] = math.sin(0.37 * t + 0.11 * c) + 0.01 * t
    return h


def canon_ref(h: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """eq. (10), left pad by 3 zeros: taps reach h_{t-3}. So h_{-1}..h_{-3} = 0."""
    out = torch.zeros_like(h)
    for t in range(h.shape[0]):
        for k in range(4):
            if t - k >= 0:
                out[t] += gates[k] * h[t - k]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gates = canon_gates()
    h = canon_input()
    out = canon_ref(h, gates)
    assert out.abs().max() > 0, "the expected output must not be all-zero"

    def f64(x):
        return "\n".join(" ".join(f"{v:.17g}" for v in row) for row in x)

    def f32(x):
        x32 = x.to(torch.float32)
        return "\n".join(" ".join(f"{v:.9g}" for v in row) for row in x32)

    text = (
        "# byteflow_field_oracle.txt — torch cpu gate for theCanon eq. (10).\n"
        "# Generator: tests/oracle/gen_byteflow_field_oracle.py — torch 2.14.0+cpu.\n"
        "# Tier (b): paper math from docs/papers/2603.03583-byteflow.pdf (p.4),\n"
        "# no author code exists (pdf p.19 + the lane's GitHub search — see\n"
        "# docs/reviews/byteflow-rate-2026-10-02.md §7).\n"
        "# OUT is computed in FLOAT64 and dumped both at .17g (f64 expected)\n"
        "# and .9g (f32, what the Rust's own f32 compare bar pins against).\n"
        "#\n"
        "version torch-field-v1\n"
        f"d {D}\n"
        f"t {T}\n"
        "gates_f32\n" + f32(gates) + "\n"
        "input_f32\n" + f32(h) + "\n"
        "output_f32\n" + f32(out) + "\n"
        "output_f64\n" + f64(out) + "\n"
    )

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f"byteflow_field_oracle: wrote {p}", file=sys.stderr)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
