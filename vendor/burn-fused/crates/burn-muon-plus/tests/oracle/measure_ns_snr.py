#!/usr/bin/env python3
"""Measure WHY there is no numeric tier-(a) gate for the Newton-Schulz iteration.

The numbers this prints are quoted in `tests/muon_oracle.rs`'s module docs, so
they are committed rather than left as prose: a bar that cannot be re-derived
is a bar that cannot be checked.

    noise  = max |authors' bf16 output - the SAME arithmetic in f32|
             This is the dtype difference the authors' own code carries
             (`muon_plus.py:55`, `X = G.bfloat16()`). It is not a defect; it is
             their implementation choice, and we run f32.
    signal = max |f32(correct a,b,c) - f32(a perturbed by 1e-4 RELATIVE)|
             This is what the defect a numeric gate would hunt looks like.
    ratio  = signal / noise.  A usable gate needs ratio >> 1.

The f32 control is OUR arithmetic with the dtype changed. It is a control, not
a reference: the reference is the authors' bf16 output, which is what `noise`
is measured against. Nothing here is compared to `MuonPlus::orthogonalize`.

RUN
    /tmp/muon-oracle-venv/bin/python tests/oracle/measure_ns_snr.py
"""
import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream"))
from utils.optim.muon_plus import zeropower_via_newtonschulz5  # noqa: E402

CORRECT = (3.4445, -4.7750, 2.0315)


def grid(m, n):
    return torch.tensor(
        [((i * 2654435761) % 997) / 498.0 - 1.0 for i in range(m * n)],
        dtype=torch.float32,
    ).reshape(m, n)


def f32_control(G, steps, abc):
    """`zeropower_via_newtonschulz5` with the bf16 cast removed. A control."""
    a, b, c = abc
    X = G.float()
    if G.size(0) > G.size(1):
        X = X.mT
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    if G.size(0) > G.size(1):
        X = X.mT
    return X


def main():
    print("shape      |out|max   noise(bf16)   signal(da=1e-4)   signal(dc=1e-4)   ratio_a  ratio_c")
    print("-" * 92)
    worst_ratio = 1e9
    for steps in (1, 2, 3, 5):
        print(f"ns_steps = {steps}")
        for (m, n) in [(8, 8), (16, 8), (32, 8), (8, 32), (64, 16)]:
            g = grid(m, n)
            base = f32_control(g, steps, CORRECT)
            noise = (
                zeropower_via_newtonschulz5(g, steps).float() - base
            ).abs().max().item()
            sa = (base - f32_control(g, steps, (3.4446, -4.7750, 2.0315))).abs().max().item()
            sc = (base - f32_control(g, steps, (3.4445, -4.7750, 2.0314))).abs().max().item()
            ra, rc = sa / max(noise, 1e-30), sc / max(noise, 1e-30)
            worst_ratio = min(worst_ratio, ra, rc)
            print(
                f"{m:3d}x{n:<3d}  {base.abs().max():8.4f}   {noise:9.3e}   "
                f"{sa:14.3e}   {sc:14.3e}   {ra:6.3f}  {rc:6.3f}"
            )
    print("-" * 92)
    print(f"worst signal/noise over every shape and step count = {worst_ratio:.4f}")
    print(
        "A bar that passes every row is above the signal, so it cannot distinguish a\n"
        "wrong coefficient from the authors' dtype choice. The constant is gated\n"
        "instead, at zero tolerance, against the literal in muon_plus.py:56."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
