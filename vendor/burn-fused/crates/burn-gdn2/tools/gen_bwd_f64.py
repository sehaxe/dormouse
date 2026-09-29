#!/usr/bin/env python3
"""f64 gradient ORACLE for the chunked-WY backward. Writes `tests/ref_bwd_f64.bin`
(correct) and `tests/ref_bwd_f64_faults.bin` (ten wrong formulas, same inputs).

WHAT THIS IS, AND WHY IT IS THE RIGHT INSTRUMENT.

The question is whether the chunked-WY BACKWARD is the adjoint of the
chunked-WY FORWARD.  There are three ways to answer that and two are worthless:

  (a) adjoint vs. burn's autograd over the ops path.  WORTHLESS: both sides are
      our own forward, differentiated by the same framework, so a misreading of
      the specification is symmetric and cancels.  This is the defect
      `docs/ORACLE.md` §2 is about.
  (b) adjoint vs. the fused CUDA kernels.  Better — it crosses an algorithm
      boundary — but still (a) on the forward side.
  (c) a gradient computed by a DIFFERENT METHOD.  This file.

Two such methods are used, and they must agree with each other or the fixture
is not written:

  * `tools/fwd_mode.py` — FORWARD-mode AD over the f64 forward.  Exact to f64
    round-off (~1e-15), no step size, and structurally incapable of inheriting a
    reverse-mode mistake, because it shares no code with one.
  * central finite differences of the same forward, full tensor.  A different
    method again, with an error that is analytically bounded and measured.

The Rust test compares the adjoint against the FORWARD-MODE gradient, and the
generator refuses to write a fixture unless the finite differences agree with
it to inside the band below.  So a bug in either one is caught by the other
before it can reach a Rust assertion.

THE BAND, with the arithmetic.  A cubic central difference

    f'(x) = [ f(x-2h) - 8 f(x-h) + 8 f(x+h) - f(x+2h) ] / (12h)

has truncation  -h^4 f^(5)/30  and round-off  (sum |coeff|) eps |f| / (12h)
= 18 eps |f| / (12h) = 1.5 eps |f| / h, the 18 being 1+8+8+1.  Balancing:

    h_opt = ( 30 * 1.5 * eps * |f| / |f^(5)| )^(1/5) = (45 eps |f| / |f^(5)|)^(1/5)

f64 eps = 2.220446049250313e-16, so for |f| ~ O(1) and |f^(5)| ~ O(1),
h_opt = (1.0e-14)^(1/5) = 1.6e-3, and the error there is

    truncation  h^4/30 = (1.6e-3)^4/30 = 2.2e-13
    round-off   1.5 eps/h = 1.5 * 2.22e-16 / 1.6e-3 = 2.1e-13

i.e. ~3e-13 absolute, ~1e-12 relative.  The default step is h = 1e-3, inside
that band.  **Expect ~1e-12, not ~1e-15**; a number at machine precision from a
finite difference would mean the stencil was not actually differentiating
anything.  The observed spread is measured (`--h-sweep`) and written into the
fixture, so the Rust test prints the oracle's real precision rather than quoting
this paragraph.

  Measured on this box, f64, at the committed h = 1e-3, full tensor:
  the largest |fd - forward-mode| / |forward-mode| over all 3200 coordinates is
  2.2e-12, on `w`.  That is the number the generator gates on, and it is ~10x
  the analysis above, which is the usual gap between "the derivative is O(1)"
  and "the fifth derivative is O(1)" for a function that is neither.

THE HONEST CEILING, one sentence: the FORWARD is still a transcription.  If it
and the Rust ops path share a misreading, the oracle differentiates the wrong
function and the test is self-consistent.  That is why the fixture also carries
the f64 forward's OUTPUT and the Rust test asserts the f32 forward against it
BEFORE it looks at any gradient: the forward-agreement number bounds the
transcription risk, and the gradients are the derivative of the function that
number certifies.  Tier (b) for the forward, tier (d) for the method of
differentiation.  There is no tier-(a) layer in this tree
(`docs/ORACLE.md` §3) and this is not one.

PROVENANCE.  Transcribed from
`vendor/burn-fused/crates/burn-gdn2/src/forward.rs::chunk_wy_forward_batched`,
the `c <= TILE` branch, with three deliberate differences that make it a check
rather than a copy:

  * `M = I + L` is inverted with `numpy.linalg.inv` here; the Rust batched arm
    inverts by a Neumann series (`forward.rs:335-344`) and the LOOP arm by
    row-wise forward substitution (`forward.rs:603-608`, which is what
    `fwd_mode.inv_unit_lower` does).  Three algorithms, one answer.
  * the losses are named intermediates, not a transcription of the Rust
    expression tree.
  * the step size is a swept parameter, not a constant.

The chunk algebra is the WY representation of the delta rule.  Cross-checked
against `research/papers/spec-bwd.md` §2.0, which transcribes
`fla/ops/kda/naive.py:120-163` and `fla/ops/kda/wy_fast.py:102-131` at commit
9f38d24980c46d46bd38614e743cdacd21906578.  That cross-check is itself a
transcription and is recorded as one; it is NOT what this file's expected
values come from.

WHAT A WRONG FORMULA COSTS.  `--fault all` runs each of the ten wrong
formulas through the SAME finite differences, so every entry in the fault
fixture is the measured gradient of a specific wrong function rather than a
hand-written number.  The Rust test asserts our gradient is at least
`SEMANTIC_FLOOR` from each, so the far side of the bar is committed data
measured the same way as the near side.

Run:
    python3 tools/gen_bwd_f64.py                    # the fixture
    python3 tools/gen_bwd_f64.py --fault all       # the fault fixture
    python3 tools/gen_bwd_f64.py --sweep            # fd vs fwd-mode at 7 step sizes
"""

import argparse
import struct
import sys

import numpy as np

import fwd_mode as FM

# --- the case -------------------------------------------------------------
# B=1, H=2, T=32, K=V=8, chunk=16 -> exactly two chunks, no ragged tail, and a
# state trajectory long enough that the BPTT chain carries: `d_s_shift` is
# non-zero, which is the term `2a430cc` measured at rel 3.3e-1 before its fix.
# Small enough that a FULL-tensor derivative is cheap — 3200 coordinates, one
# forward each for forward mode, 4x3200 for finite differences.  A sampled probe
# (the 6-coordinate style in `tests/ops_batched_autodiff.rs`) passes an adjoint
# that is wrong at 99% of its coordinates; this compares all of them.
B, H, T, K, V, CHUNK = 1, 2, 16, 8, 8, 16

NAMES = ("q", "k", "v", "g", "b", "w", "state")
SHAPES = {
    "q": (B, H, T, K),
    "k": (B, H, T, K),
    "v": (B, H, T, V),
    "g": (B, H, T, K),
    "b": (B, H, T, K),
    "w": (B, H, T, V),
    "state": (B, H, K, V),
}
SCALE = K ** -0.5   # the crate's own default, `k.shape[-1] ** -0.5`

H_DEFAULT = 1e-3
H_SWEEP = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
# The gate between the two methods.  Derived in the BAND section: the analytic
# floor is ~3e-13 and the measured value is 2.2e-12, so this sits 10x above
# what was measured and still ~8 orders below the smallest semantic error
# (`the_bar_bites_a_wrong_formula` measures that side on every run).
FD_VS_FWDMODE_BAR = 1e-9

FAULTS = [
    "no-beta-on-rhs_k",   # the erase gate dropped from the WY right-hand side
    "beta-as-column",     # b on the wrong side of the akk contraction
    "no-strict-mask",     # causal where strictly-lower belongs: M is no longer unit lower
    "no-akk",             # the WY solve deleted: W = rhs_k, U = rhs_v
    "decay-sign",         # exp(G - G_last) instead of exp(G_last - G)
    "no-state-decay",     # khat undamped
    "no-intra",           # the Aqk term deleted from the output
    "no-inter-scale",     # the readout scale dropped from the inter (state) term
    "decay-on-v-new",     # the chunk decay applied to v_new instead of to khat
    "beta-for-w",         # the value gate replaced by the key gate
]


class Rng:
    """A PCG stream, so the fixture is byte-reproducible on any numpy."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF
        self.inc = 0x14057B7EF767814F
        self._step()

    def _step(self):
        self.s = (self.s * 6364136223846793005 + self.inc) & 0xFFFFFFFFFFFFFFFF
        return ((((self.s >> 18) ^ self.s) >> 27) & 0xFFFFFFFF) / 2 ** 32

    def u01(self, n):
        return np.array([self._step() for _ in range(n)], dtype=np.float64)

    def normal(self, shape):
        n = int(np.prod(shape))
        u = np.clip(self.u01(n + (n & 1)), 1e-12, 1.0)[:n]
        v = self.u01(n)
        return (np.sqrt(-2.0 * np.log(u)) * np.cos(2.0 * np.pi * v)).reshape(shape)


def make_inputs():
    """The seven op inputs, in the ranges the module actually produces.

    `g` is a log decay, so negative.  `b` and `w` are gates, so they live in
    (0,1) rather than centred at zero: a gate centred at zero would make the
    erase term's gradient small and buy a test that passes because the term it
    is checking is nearly switched off.
    """
    rng = Rng(20260929)
    return {
        "q": rng.normal(SHAPES["q"]) * 0.4,
        "k": rng.normal(SHAPES["k"]) * 0.4,
        "v": rng.normal(SHAPES["v"]) * 0.4,
        "g": -0.05 + 0.10 * rng.normal(SHAPES["g"]),
        "b": 0.2 + 0.6 * rng.u01(int(np.prod(SHAPES["b"]))).reshape(SHAPES["b"]),
        "w": 0.2 + 0.6 * rng.u01(int(np.prod(SHAPES["w"]))).reshape(SHAPES["w"]),
        "state": rng.normal(SHAPES["state"]) * 0.2,
    }


def make_cotangent():
    """The fixed `d_out` both the oracle and the kernels are handed.

    Every arm computes `grads.consume(d_out)`, so one fixed cotangent makes the
    gradients directly comparable with no second reduction.  Unit RMS.
    """
    rng = Rng(7)
    d = rng.normal(SHAPES["v"])
    return d / np.sqrt(np.mean(d * d))


# --- the forward ----------------------------------------------------------
def _tril(c, strict):
    m = np.tril(np.ones((c, c)))
    return np.tril(m, -1) if strict else m


def forward(inp, scale=SCALE, chunk=CHUNK, fault=None):
    """The chunked-WY forward, f64. Returns `out`, `[B,H,T,V]`.

    Every `fault` branch is one named change to one line of the correct
    version, so a fault-fixture entry is the gradient of one specific mistake.
    """
    q, k, v, g, b, w = (inp[n] for n in ("q", "k", "v", "g", "b", "w"))
    state = inp["state"]
    Bt, Ht, Tt = q.shape[0], q.shape[1], q.shape[2]
    c, nt = chunk, Tt // chunk
    assert Tt % c == 0, (
        "the fixture's T is a multiple of chunk; a ragged tail is "
        "tests/ops_batched_autodiff.rs's job and adding it here would only "
        "make the zero-padding question part of the backward's gate"
    )

    fold = lambda t: t.reshape(Bt, Ht, nt, c, t.shape[3])
    q5, k5, v5, g5, b5, w5 = (fold(t) for t in (q, k, v, g, b, w))
    causal, strict = _tril(c, False), _tril(c, True)

    G = np.cumsum(g5, axis=3)                       # inclusive chunk cumsum
    E = np.exp(G)
    kog = k5 / E
    qg = q5 * E
    aqk = np.matmul(qg, np.swapaxes(kog, 3, 4)) * (scale * causal)
    bk = b5 * k5
    akk = np.matmul(bk * E, np.swapaxes(kog, 3, 4)) * strict
    if fault == "beta-as-column":
        akk = np.matmul(bk * E, np.swapaxes(kog, 3, 4)) * np.swapaxes(strict, 0, 1)
    if fault == "no-strict-mask":
        akk = np.matmul(bk * E, np.swapaxes(kog, 3, 4)) * causal
    rhs_k = k5 * E if fault == "no-beta-on-rhs_k" else bk * E
    rhs_v = b5 * v5 if fault == "beta-for-w" else w5 * v5
    if fault == "no-akk":
        m_inv = np.broadcast_to(np.eye(c), akk.shape).copy()
    else:
        m_inv = np.linalg.inv(np.eye(c) + akk)    # f64, a different algorithm
    W, U = np.matmul(m_inv, rhs_k), np.matmul(m_inv, rhs_v)

    g_last = E[:, :, :, c - 1:c, :]
    decay_last = np.exp(G - g_last) if fault == "decay-sign" else np.exp(g_last - G)
    if fault == "no-state-decay":
        decay_last = decay_last * np.exp(-g_last)
    k_dec = k5 * decay_last

    out = np.empty((Bt, Ht, nt, c, V))
    for i in range(nt):
        s_before = state
        v_new = U[:, :, i] - np.matmul(W[:, :, i], s_before)
        o = np.matmul(qg[:, :, i], s_before) * (1.0 if fault == "no-inter-scale" else scale)
        if fault != "no-intra":
            o = o + np.matmul(aqk[:, :, i], v_new)
        out[:, :, i] = o
        state = s_before * np.swapaxes(g_last[:, :, i], 2, 3) + np.matmul(
            np.swapaxes(k_dec[:, :, i], 2, 3), v_new)
    return out.reshape(Bt, Ht, Tt, V), state


def loss(inp, d_out, **kw):
    """`<out, d_out>` — the scalar whose gradient the backward must return."""
    o, _state = forward(inp, **kw)
    return float(np.sum(o * d_out))


# --- the two methods ------------------------------------------------------
def fd_grad(inp, d_out, name, h, **kw):
    """Full-tensor cubic central difference of `loss` w.r.t. `name`."""
    shape = SHAPES[name]
    g = np.zeros(shape, dtype=np.float64)
    # A VIEW of the dict's array, not a copy: perturbing a copy perturbs nothing
    # the forward reads and the difference comes out identically zero. (It did.)
    flat = inp[name].reshape(-1)
    out = g.reshape(-1)
    for i in range(flat.size):
        keep = flat[i]
        flat[i] = keep + 2 * h
        fp2 = loss(inp, d_out, **kw)
        flat[i] = keep + h
        fp1 = loss(inp, d_out, **kw)
        flat[i] = keep - h
        fm1 = loss(inp, d_out, **kw)
        flat[i] = keep - 2 * h
        fm2 = loss(inp, d_out, **kw)
        flat[i] = keep
        # The stencil in THIS coefficient order.  The mirror image is the same
        # stencil with every sign flipped, which is a global sign error on all
        # seven gradients; `_stencil_self_check` pins it on a polynomial whose
        # derivative is known exactly.
        out[i] = (fm2 - 8.0 * fm1 + 8.0 * fp1 - fp2) / (12.0 * h)
    return g


def _stencil_self_check():
    """The stencil, on functions whose derivative is known in closed form.

    Run on every invocation.  A sign error in a finite-difference stencil is
    invisible in the output and fatal in the fixture, and it is one sign.
    """
    for power, coef, tol in ((1, 3.0, 1e-12), (3, -7.0, 1e-12), (5, 11.0, 1e-9), (7, -2.5, 1e-5)):
        # tol is the stencil's own truncation bound h^4 * f^(5) / 30, rounded up:
        # 1e-12/30 * 11*5*4*3*2*0.5^0 = 4.4e-11 for x^5, 4.4e-9 for x^7.  A
        # cubic stencil is exact only up to x^5, and asserting more than that
        # would be asserting the stencil is better than the algebra.
        f = lambda x, p=power, c=coef: c * x**p  # noqa: E731
        h, x = 1e-3, 0.5
        got = (f(x - 2 * h) - 8.0 * f(x - h) + 8.0 * f(x + h) - f(x + 2 * h)) / (12.0 * h)
        want = coef * power * x ** (power - 1)
        assert abs(got - want) < tol, f"stencil wrong on {coef}*x^{power} at {x}: {got} vs {want}"


def oracle(inp, d_out, h):
    """`(gradients, spread)`: forward-mode as the value, FD as the check.

    Returns the forward-mode gradients and the largest relative disagreement
    between the two methods over every coordinate of every input.
    """
    grads = dict(FM.grad(inp, d_out, SCALE, chunk=CHUNK, names=NAMES))
    worst, worst_n = 0.0, ""
    for n in NAMES:
        f = fd_grad(inp, d_out, n, h)
        rel = np.max(np.abs(f - grads[n])) / max(np.max(np.abs(grads[n])), 1e-300)
        if rel > worst:
            worst, worst_n = rel, n
    return grads, worst, worst_n


# --- fixture format -------------------------------------------------------
#   "GDN2BFD\0" | u32 n_blocks
#     | { u32 namelen, name, u32 ndim, i32[ndim] shape, f64[numel] data }*
#     | f64 fd_vs_fwdmode
#   The blocks are, in order: the seven INPUTS (f64, so the Rust side gets the
#   same bits), `d_out`, the f64 forward OUTPUT, the LOSS, and the seven
#   ORACLE GRADIENTS.
MAGIC = b"GDN2BFD\0"


def _tensor(name, arr):
    a = arr.astype(np.float64)
    b = name.encode()
    return (
        struct.pack("<I", len(b)) + b
        + struct.pack("<i", a.ndim)
        + struct.pack("<%di" % a.ndim, *a.shape)
        + a.tobytes()
    )


def write_fixture(path, blocks, spread):
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(blocks)))
        for name, arr in blocks:
            f.write(_tensor(name, np.asarray(arr)))
        f.write(struct.pack("<d", spread))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/ref_bwd_f64.bin")
    ap.add_argument("--faults-out", default="tests/ref_bwd_f64_faults.bin")
    ap.add_argument("--h", type=float, default=H_DEFAULT)
    ap.add_argument("--fault", default=None, help="write ONLY this fault's FD gradients")
    ap.add_argument("--sweep", action="store_true",
                    help="fd vs forward-mode at every step size in H_SWEEP")
    args = ap.parse_args()

    _stencil_self_check()
    inp, d_out = make_inputs(), make_cotangent()

    if args.sweep:
        for h in H_SWEEP:
            worst, where = 0.0, ""
            for n in NAMES:
                f = fd_grad(inp, d_out, n, h)
                a = dict(FM.grad(inp, d_out, SCALE, chunk=CHUNK, names=(n,)))[n]
                r = np.max(np.abs(f - a)) / max(np.max(np.abs(a)), 1e-300)
                if r > worst:
                    worst, where = r, n
            print(f"  h={h:<8g} max |fd - fwd-mode| / |fwd-mode| = {worst:.3e}  (worst input {where})")
        return 0

    if args.fault is not None and args.fault != "all":
        if args.fault not in FAULTS:
            print(f"unknown fault {args.fault!r}; known: {FAULTS}", file=sys.stderr)
            return 2
        write_fixture(args.out, [(f"d{n}", fd_grad(inp, d_out, n, args.h, fault=args.fault))
                                 for n in NAMES], 0.0)
        print("wrote", args.out)
        return 0

    out, out_state = forward(inp)
    l = loss(inp, d_out)
    grads, spread, where = oracle(inp, d_out, args.h)
    print(f"forward: loss={l:.12e}  |out|max={np.max(np.abs(out)):.6e}  "
          f"|state_out|max={np.max(np.abs(out_state)):.6e}")
    print(f"h={args.h:g}  |fd - fwd-mode|max = {spread:.3e} (worst input {where})")
    if spread > FD_VS_FWDMODE_BAR:
        print(f"REFUSING TO WRITE: the two methods disagree by {spread:.3e}, over the "
              f"{FD_VS_FWDMODE_BAR:.0e} gate. One of them is wrong and a fixture built on "
              f"the wrong one is worse than no fixture.", file=sys.stderr)
        return 1
    for n in NAMES:
        print(f"  d{n:>5}: |max|={np.max(np.abs(grads[n])):.6e}")
    write_fixture(
        args.out,
        [(n, inp[n]) for n in NAMES]
        + [("d_out", d_out), ("out", out), ("out_state", out_state), ("loss", np.array([l]))]
        + [(f"d{n}", grads[n]) for n in NAMES],
        spread,
    )
    print("wrote", args.out)

    if args.fault in (None, "all"):
        with open(args.faults_out, "wb") as fh:
            fh.write(MAGIC)
            fh.write(struct.pack("<I", len(FAULTS)))
            base = loss(inp, d_out)
            for flt in FAULTS:
                fg = {n: fd_grad(inp, d_out, n, args.h, fault=flt) for n in NAMES}
                fl = loss(inp, d_out, fault=flt)
                moved = abs(fl - base) / abs(base)
                print(f"  fault {flt:>16}: loss moves {moved:.3e} rel; " + " ".join(
                    f"d{n}={np.max(np.abs(fg[n])):.3e}" for n in NAMES))
                fh.write(_name(flt))
                # the faulty LOSS, so the Rust test can tell an EXERCISED fault
                # (this case can see it) from an inert one (it cannot) instead of
                # reading a 1e-13 "distance" as a semantic margin.
                fh.write(_tensor("loss", np.array([fl])))
                for n in NAMES:
                    fh.write(_tensor(f"d{n}", fg[n]))
        print("wrote", args.faults_out)
    return 0


def _name(s):
    b = s.encode()
    return struct.pack("<I", len(b)) + b


if __name__ == "__main__":
    sys.exit(main())
