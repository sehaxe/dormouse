"""Goldens for burn-situ's form gate, from moonshotai/Kimi-K3's own reference.

THE REFERENCE, precisely (ADR-0020: name the external file):

  https://huggingface.co/moonshotai/Kimi-K3/raw/main/modeling_kimi_linear.py
    sha256 9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a
    class SituAndMul (lines 64-82), forward at 75-82
  https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json
    sha256 9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213
    text_config.hidden_act = "situ"
    text_config.activation_situ_beta = 4.0
    text_config.activation_situ_linear_beta = 25.0
  arXiv:2607.24653v2 Eq (12) §2.3.2, Eq (18)/(19) App. B
    (k3.pdf sha256 1a21e4d15de52eb0d4cf09cdbffc6248c497ba6897572cc5574af237c13da51d)

Fetched 2026-09-30 from the `main` revision; the file is not vendored here
because it is Moonshot's under the Kimi K3 License and 1314 lines of it would
be more noise than evidence - the sha256 above is what lets a reader check
they have the same bytes.

TIER. The FORM is tier (a): a named external implementation exists and this
crate matches it. The NUMBERS are a TRANSCRIPTION, not a run of Moonshot's
code - `SituAndMul.forward` needs torch, torch is not installed here, and
nothing gets installed without the owner saying so. The transcription keeps the
reference's own precision (`.to(torch.float32)` on both halves, so f32, not
f64) and its op order, so the goldens are f32-faithful to within a last-ulp
reassociation. No external golden file exists, and no byte-exact comparison is
possible: the reference computes `beta * tanh(gate/beta) * sigmoid(gate)`,
this crate computes `(gate/beta).tanh() * beta`, and f32 multiply is not
associative. The gates assert 1e-6 RELATIVE, which is the f32 ulp band.

    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(x.dtype)

numpy only (the f64-fixture precedent in burn-gdn2 is numpy-only too, so this
runs before any build and needs no cargo lock).
"""

import math
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

B1 = np.float32(4.0)    # activation_situ_beta        (gate branch)
B2 = np.float32(25.0)   # activation_situ_linear_beta (up branch)

f32 = np.float32


def situ(x, beta=B1, linear_beta=B2):
    """SituAndMul.forward, f32 throughout."""
    d = x.shape[-1] // 2
    gate = x[..., :d].astype(f32)
    up = x[..., d:].astype(f32)
    situ_a = beta * np.tanh(gate / beta) * (1.0 / (1.0 + np.exp(-gate)))
    if linear_beta is not None:
        up = linear_beta * np.tanh(up / linear_beta)
    return situ_a * up


def swiglu(x):
    """SwiGLU: gate * sigmoid(gate) * up, both factors unbounded."""
    d = x.shape[-1] // 2
    gate = x[..., :d].astype(f32)
    up = x[..., d:].astype(f32)
    return gate * (1.0 / (1.0 + np.exp(-gate))) * up


def silu(x):
    """What dormouse runs today: elementwise SiLU on a d->f up-projection.
    The nearest GLU-shaped reading of it, for the separation number."""
    return x.astype(f32) * (1.0 / (1.0 + np.exp(-x.astype(f32))))


# The separating fixture. H = 1, so a row IS (gate, up) and the output is a
# scalar: every number below is the arm's whole contribution at that point.
# The magnitudes are chosen to sit ON the two caps (gate ~ 3*beta1, up ~ 3*beta2),
# in the linear near-origin region, and in the negative Swish tail.
PAIRS = [
    (0.0, 0.0),
    (1.0, 1.0),
    (-1.0, 1.0),
    (-0.5, -0.25),
    (4.0, 25.0),
    (20.0, 100.0),
    (100.0, 100.0),
    (-20.0, 100.0),
    (-100.0, 50.0),
    (0.125, 7.5),
]


def row(g, u):
    return np.array([g, u], dtype=f32)


def rel(a, b):
    """The SAME symmetric relative distance the Rust gate uses, so the numbers
    in the findings file and the thresholds in the test cannot drift apart."""
    a, b = float(a), float(b)
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def main():
    x = np.stack([row(g, u) for g, u in PAIRS])

    svals, wvals = situ(x), swiglu(x)
    print("=== FIXTURE (H=1): SiTU-GLU vs the two forms it must differ from ===")
    print("    rel = |a-b| / max(|a|,|b|) - the symmetric measure the Rust gate asserts on")
    print(f"{'gate':>8} {'up':>8} | {'situ(b1=4,b2=25)':>18} {'swiglu':>14} {'silu(gate)':>12}"
          f" | {'rel vs swiglu':>13} {'rel vs silu':>12}")
    for i, (g, u) in enumerate(PAIRS):
        s = float(svals[i][0])
        w = float(wvals[i][0])
        l = float(silu(np.array([g], dtype=f32))[0])
        print(f"{g:8.3f} {u:8.3f} | {s:18.9f} {w:14.6f} {l:12.6f}"
              f" | {rel(s, w):13.6f} {rel(s, l):12.6f}")

    # The paper's own bound, Eq (19): ||SiTU-GLU(x)||_inf <= beta1 * beta2 = 100.
    print("\n=== the bound, Eq (19) ===")
    big = np.stack([row(g, u) for g, u in [(1e4, 1e4), (-1e4, 1e4), (1e4, -1e4)]])
    vals = situ(big)
    print(f"beta1*beta2 = {float(B1) * float(B2):.1f}   situ(1e4,1e4) = {float(vals[0][0]):.4f}"
          f"   max|.| over the sweep = {float(np.max(np.abs(situ(x)))):.4f}")
    print("  -> the bound holds with margin; SwiGLU on the same sweep reaches "
          f"{float(np.max(np.abs(swiglu(x)))):.1f} (unbounded)")

    # ---- gradient health: tanh saturates, so the cap is exactly where the
    # gradient dies. The derivative is taken by CENTRAL DIFFERENCES on the
    # same formula, and cross-checked against the closed form. The first
    # version of this script used a hand-written closed form and got two
    # things wrong - a spurious beta factor on the tanh term and sigma^2
    # where the derivative of sigmoid is sigma*(1-sigma) - which put the
    # gate gradient at 4.0 where it is really 1.8e-4. Two hand-derived
    # formulas in one script is the exact reason the cross-check is here.
    def situ_f64(g, u, beta=4.0, linear_beta=25.0):
        """The reference formula in f64, for the derivative only."""
        s = 1.0 / (1.0 + math.exp(-g))
        return beta * math.tanh(g / beta) * s * linear_beta * math.tanh(u / linear_beta)

    def swiglu_f64(g, u):
        return g * (1.0 / (1.0 + math.exp(-g))) * u

    # FACTOR derivatives, not output derivatives: the output is
    # gate_factor(g) * up_factor(u), so its d/dg carries the up factor as a
    # constant multiplier and cannot be compared against the gate factor's own
    # derivative. Comparing factors to factors is what makes the cross-check
    # below a check rather than a tautology.
    def gate_factor(g, beta=4.0):
        return beta * math.tanh(g / beta) * (1.0 / (1.0 + math.exp(-g)))

    def swish_factor(g):
        return g * (1.0 / (1.0 + math.exp(-g)))

    def up_factor(u, beta=25.0):
        return beta * math.tanh(u / beta)

    def cd1(fn, z, h=1e-6):
        return (fn(z + h) - fn(z - h)) / (2 * h)

    def closed_gate(g, beta=4.0):
        # d/dg [ beta*tanh(g/beta)*sigmoid(g) ]
        s = 1.0 / (1.0 + math.exp(-g))
        return (1.0 / math.cosh(g / beta) ** 2) * s + beta * math.tanh(g / beta) * s * (1.0 - s)

    def closed_up(u, beta=25.0):
        # d/du [ beta*tanh(u/beta) ] = sech^2(u/beta)
        return 1.0 / math.cosh(u / beta) ** 2

    print("\n=== gradient of the GATE factor (beta1=4), vs SwiGLU's Swish gate ===")
    print("    (f64 central differences resolve ~1e-8 here; below that the")
    print("     CLOSED FORM is the number and the numeric one is noise, so the")
    print("     cross-check is asserted only where the numeric value can carry it)")
    print(f"{'gate':>8} {'d situ/dgate':>14} {'closed form':>14} {'d swish/dgate':>14} {'ratio':>10}")
    for g in [0.5, 1.0, 4.0, 8.0, 12.0, 16.0, 20.0, 40.0]:
        num = cd1(gate_factor, g)
        clo = closed_gate(g)
        if abs(clo) > 1e-6:
            rel_err = abs(num - clo) / abs(clo)
            assert rel_err < 1e-5, f"closed form disagrees with the numeric derivative at g={g}: {rel_err}"
        sw = cd1(swish_factor, g)
        print(f"{g:8.2f} {num:14.6e} {clo:14.6e} {sw:14.6e} {clo / sw:10.2e}")

    print("\n=== gradient of the UP factor (beta2=25); SwiGLU's is the identity, 1.0 ===")
    print(f"{'up':>8} {'d situ/dup':>14} {'closed form':>14} {'ratio to 1.0':>12}")
    for u in [7.5, 25.0, 50.0, 75.0, 100.0, 200.0]:
        num = cd1(up_factor, u)
        clo = closed_up(u)
        if abs(clo) > 1e-6:
            rel_err = abs(num - clo) / abs(clo)
            assert rel_err < 1e-5, f"closed form disagrees with the numeric derivative at u={u}: {rel_err}"
        print(f"{u:8.2f} {num:14.6e} {clo:14.6e} {clo:12.2e}")

    # The health question, stated as a threshold rather than a feeling: the
    # f32 noise floor. A branch whose gradient falls under ~1e-7 relative is
    # indistinguishable from zero in f32 - that is where "the cap killed the
    # gradient" becomes a measurement instead of a worry.
    print("\n=== is the cap's gradient dead in f32? ===")
    for g, u in [(4.0, 25.0), (8.0, 50.0), (12.0, 75.0), (20.0, 100.0), (40.0, 200.0)]:
        out = situ_f64(g, u)
        dg = abs(closed_gate(g) * up_factor(u))
        du = abs(closed_up(u) * gate_factor(g))
        print(f"  gate={g:6.2f} up={u:6.2f}  |out|={abs(out):9.4f}  "
              f"rel d/dgate={dg / abs(out):.3e}  rel d/dup={du / abs(out):.3e}")
    print("  f32 eps = 1.19e-07: a relative gradient under it cannot move the")
    print("  output by even one ulp, so those rows are numerically dead.")

    # ---- the literals to pin, f32, printed as Rust f32 bit patterns' decimals
    print("\n=== GOLDENS to pin (f32, reference op order) ===")
    vals = svals
    for (g, u), v in zip(PAIRS, vals):
        print(f"  ({g:>7.3f}, {u:>7.3f}) -> {float(v[0])!r}   hex {float(v[0]).hex()}")


if __name__ == "__main__":
    main()
