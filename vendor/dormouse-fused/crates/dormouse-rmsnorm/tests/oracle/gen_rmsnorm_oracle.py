#!/usr/bin/env python3
"""Generate the RMSNorm golden fixture from TWO INDEPENDENT upstream
implementations, both RUN, neither of them ours.

    $ uv venv --python 3.12 /tmp/opencode/oracle-venv
    $ VIRTUAL_ENV=/tmp/opencode/oracle-venv uv pip install \
          --index-url https://download.pytorch.org/whl/cpu torch numpy
    $ VIRTUAL_ENV=/tmp/opencode/oracle-venv uv pip install triton flash-linear-attention
    $ /tmp/opencode/oracle-venv/bin/python gen_rmsnorm_oracle.py --dump \
          > ../fixtures/rmsnorm_oracle.txt

`--dump` also re-extracts the two upstream sources into `upstream/`, so the
byte-pinned copies next to this file are never hand-edited (they are a
`inspect.getsource()` transcript of what actually ran).  Run it from this
directory; both paths below are relative to it.

WHAT THE EXPECTED VALUE CAME FROM
---------------------------------
1. `torch.nn.functional.rms_norm` -> `torch.rms_norm`, PyTorch's OWN ATen
   operator.  Shipped as the `torch==2.14.0+cpu` wheel from
   download.pytorch.org; the C++ it dispatches to is
   `aten/src/ATen/native/layer_norm.cpp::rms_norm` in
   https://github.com/pytorch/pytorch at tag `v2.14.0`.
2. `fla.modules.layernorm.rms_norm_ref`, flash-linear-attention's own
   torch-level reference for the operator its Triton kernels accelerate.
   Shipped as the `flash-linear-attention==0.5.2` wheel on PyPI, from
   https://github.com/fla-org/flash-linear-attention; the sha256 of the
   installed `fla/modules/layernorm.py` goes into the fixture, so the exact
   bytes are identified even though the wheel carries no git revision.

Fetched and run 2026-09-29 on this box, CPU only.  Neither is Zhang &
Sennrich 2019 (arXiv:1910.07467), the paper this crate implements, which
ships no code at all.  See tests/rmsnorm_oracle.rs for what that does and
does not license.

THE DISCRIMINATING COLUMNS
--------------------------
`out_eps_outside`, `out_no_eps`, `out_layernorm` and `out_axis1` are the four
ways this formula most often goes wrong.  They are emitted so the test can
MEASURE how far the fixture's correct column sits from each wrong one, instead
of asserting a bound nobody checked.

WHY A SCALE IS THE ONLY LEVER, and the arithmetic
------------------------------------------------
With `r = sqrt(mean(x^2))` the row's own rms and `e = eps`, the correct answer
divides by `sqrt(r^2 + e)` and each wrong formula divides by something else:

    eps-OUTSIDE   (r + e)          gap = |(r+e)/sqrt(r^2+e) - 1| ~ e/r
    eps-DROPPED   sqrt(r^2)        gap = |sqrt(r^2+e)/r - 1|       ~ e/(2 r^2)
    eps-1e-7      torch's default  same shape as INSIDE, different constant

Both gaps VANISH as r grows -- the first as 1/r, the second as 1/r^2 -- so on
`main` (r ~ 2.1) the eps-dropped gap is `1e-5/(2*2.1^2) = 1.1e-6` and the
eps-outside gap is `1e-5/2.1 = 4.8e-6`: both at or below any honest f32
tolerance, which is exactly why "close enough" cannot decide this question.

The fixture therefore carries the eps-decisive cases at r ~ 1e-5..2e-3, where
e/r and e/(2r^2) are both O(1) and the gaps are O(1) -- four to five orders of
magnitude above the tolerance.  `big` (r ~ 8e3) is in the fixture for the
opposite reason: it carries NO eps power at all (both gaps < 1.3e-9) so the
test can MEASURE that and must not ask it for any.

An eps-decisive case also needs a row whose mean is NOT ~0, or `out_layernorm`
(the centred/variance misreading) lands on top of the right answer: on a
zero-mean row `x - mean == x`.  `tiny` / `zero_row` / `d1` carry the sin-shaped
`base`, `micro` is the one that cannot and measures 2e-2 (still 2000x TOL).

`d1` is d = 1, so its gain is ONE element and no per-feature broadcast can
break on it.  That is a property of the shape, not an oversight: `d2` is the
smallest axis where a collapsed broadcast is observable, and the test exempts
d == 1 by that arithmetic and asserts the d == 1 case is PRESENT.
"""

import hashlib
import inspect
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import fla
import fla.modules.layernorm as fla_ln
from fla.modules.layernorm import rms_norm_ref

FLA_FILE = inspect.getsourcefile(fla_ln)
FLA_SHA = hashlib.sha256(open(FLA_FILE, "rb").read()).hexdigest()
FLA_REL = "site-packages/" + FLA_FILE.split("site-packages/")[-1]
FLA_VERSION = fla.__version__
TORCH_VERSION = torch.__version__

# The two upstreams must not disagree by more than this (relative to the output
# scale) or the fixture is recording a conflict instead of an answer.
UPSTREAM_AGREEMENT = 1e-6


def g9(v) -> str:
    """NINE significant digits.  An f32 needs 9 to round-trip EXACTLY;
    `f"{v:g}"` is six and silently turns every 0.499750137 column into
    0.49975 -- which once failed a test on CORRECT code."""
    return "%.9g" % float(np.float32(v))


def fmt(vals) -> str:
    return " ".join(g9(v) for v in np.asarray(vals, dtype=np.float32).ravel())


def f64(x):
    return x.astype(np.float64)


# ── the wrong formulas, spelled out so they can be compared ─────────────────
def out_eps_outside(x, w, eps):
    """eps OUTSIDE the sqrt: x / (sqrt(mean(x^2)) + eps) * w."""
    rms = np.sqrt((f64(x) ** 2).mean(-1, keepdims=True))
    return (x / (rms + eps) * w).astype(np.float32)


def out_no_eps(x, w, eps):
    """x / sqrt(mean(x^2)) * w -- eps dropped."""
    rms = np.sqrt((f64(x) ** 2).mean(-1, keepdims=True))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x / rms * w).astype(np.float32)


def out_layernorm(x, w, eps):
    """The classic misreading: centre the row and use the VARIANCE."""
    xd = f64(x)
    mu = xd.mean(-1, keepdims=True)
    var = ((xd - mu) ** 2).mean(-1, keepdims=True)
    return ((xd - mu) / np.sqrt(var + eps) * w).astype(np.float32)


def out_axis1(x, w, eps):
    """Reduce over the TIME axis instead of the feature axis.  Only
    well-formed for a cube whose last two axes are equal."""
    assert x.shape[1] == x.shape[2]
    rstd = 1.0 / np.sqrt((f64(x) ** 2).mean(1, keepdims=True) + eps)
    return (x * rstd * w).astype(np.float32)


# ── the cases ──────────────────────────────────────────────────────────────
def ramp(d):
    return np.array([0.5 + 0.25 * j for j in range(d)], dtype=np.float32)


def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    b, t, d = 2, 3, 8
    i = np.arange(b * t * d, dtype=np.float64)
    base = (np.sin(i * 0.37) * 3.0).reshape(b, t, d)
    # ONE draw, reused by `micro` and `micro_eps0`.  Drawing twice made those
    # two cases different datasets, so the test that compares their outputs was
    # measuring the DATA difference and calling it the eps difference.
    micro_x = (1e-4 * rng.standard_normal((1, 4, 8))).astype(np.float32)

    # 1. the everyday case: ordinary activations, non-constant gain.  eps is
    #    ~5e-6 relative here -- below any tolerance, ON PURPOSE, because it is
    #    the case that shows the tolerance is not being asked to carry eps.
    cases.append(("main", base.copy(), ramp(d), 1e-5))

    # 2. THE discriminating case.  rms ~ 2.1e-3 against eps = 1e-5, so
    #    e/r = 4.7e-3 and the eps-OUTSIDE gap is |(r+e)/sqrt(r^2+e) - 1| = 0.44
    #    while on `main` the same quantity is 4.8e-6: four orders of magnitude,
    #    from the DATA ALONE.  The eps-inside formula is the one being pinned.
    cases.append(("tiny", (1e-3 * base).astype(np.float32), ramp(d), 1e-5))

    # 3. rms ~ 9.1e-5: e/r = 0.11, both gaps O(1).  A reduction that forgot to
    #    divide by d, or normalised over the wrong axis, is off by O(1) rather
    #    than O(1e-5).  This row's mean is ~0 by construction (it is noise), so
    #    `out_layernorm` sits closest to the right answer here of any case --
    #    2.0e-2, still 2000x TOL.  Stated so nobody reads 2e-2 as "decisive".
    cases.append(("micro", micro_x.copy(), ramp(d), 1e-5))

    # 4. THE SAME DATA at eps = 0.  eps-inside and no-eps COINCIDE here by
    #    construction, so this case carries no eps power -- and it is the case
    #    that catches an eps threaded through as a constant 1e-5.  The previous
    #    revision drew a SECOND random tensor, which made `micro` vs
    #    `micro_eps0` a comparison of two datasets; the fixture said "the same
    #    data at eps = 0" and was not.
    cases.append(("micro_eps0", micro_x.copy(), ramp(d), 0.0))

    # 5. an exactly-zero row: rms = sqrt(0 + eps), output 0.  This is where
    #    eps-outside degenerates (0/eps = 0, agreeing) while no-eps is 0/0, and
    #    where a `+ eps` AFTER the division would return eps*w.  The zero row
    #    is scale-free, so the rest of the tensor is scaled to the decisive
    #    band and the non-zero rows decide eps too.
    z = (1e-3 * base).astype(np.float32)
    z[1, 2, :] = 0.0
    cases.append(("zero_row", z, ramp(d), 1e-5))

    # 6. d = 1: mean(x^2) = x^2, so the output is x/sqrt(x^2 + eps) * w -- and
    #    with |x| chosen BELOW sqrt(eps) the eps-decisive and the d=1 questions
    #    are the same question.  The gain is ONE element and therefore constant:
    #    no per-feature broadcast exists on a length-1 axis, which is why the
    #    test exempts d == 1 and demands a d == 2 case instead.  Values straddle
    #    zero so the sign is exercised too.
    cases.append(("d1", np.array([[[2.5e-5], [-7.0e-5]]], dtype=np.float32),
                  np.array([3.0], np.float32), 1e-5))

    # 7. d = 2: the SMALLEST axis on which a per-feature gain broadcast is
    #    observable at all, so `the_fixture_gains_are_not_constant` has a case
    #    it can actually enforce on.  Two wildly different gains: a broadcast
    #    that collapsed them, transposed them, or averaged them is off by O(1).
    #    Small magnitude so this case is ALSO eps-decisive.
    d2 = (1e-3 * base[:1, :2, :2]).astype(np.float32)
    cases.append(("d2", d2, np.array([-2.5, 3.0], np.float32), 1e-5))

    # 8. d = 13: not a power of two and not a multiple of 4, so a reduction
    #    that assumes a padded tail shows up.  Ordinary magnitude: this is the
    #    padding case, not the eps case.
    j = np.arange(13, dtype=np.float64)
    cases.append(("odd_d", (np.cos(j * 1.1) * 2.0).reshape(1, 1, 13), ramp(13), 1e-5))

    # 9. large magnitude: mean(x^2) ~ 6e7, so eps moves the answer by ~1.3e-9
    #    relative, BELOW any tolerance.  In the fixture so the test can MEASURE
    #    that and not demand eps power it does not have.
    cases.append(("big", (1e4 * rng.standard_normal((1, 2, 8))).astype(np.float32),
                  ramp(d), 1e-5))

    # 10. the same data as `main` at torch's own DEFAULT eps (1.1920929e-7,
    #     finfo(f32).eps) instead of 1e-5.  Proves the eps is a parameter of the
    #     call and not a constant baked into either side.
    cases.append(("eps_1e7", base.copy(), ramp(d), 1.1920929e-7))

    # 11. negative gains: the output is signed, and an abs() or a relu
    #     somewhere upstream would be invisible on `main`.
    cases.append(("neg_gain", base.copy(), -ramp(d), 1e-5))

    # 12. a square cube, so "reduce over the last axis" and "reduce over the
    #     middle axis" are both well-formed.  This is the reduction-axis case;
    #     without it a mean over dim 1 would be invisible.
    cube = rng.standard_normal((1, 8, 8)).astype(np.float32)
    cases.append(("axis_confusion", cube, ramp(8), 1e-5))

    return cases


TORCH_REPO = "https://github.com/pytorch/pytorch"
FLA_REPO = "https://github.com/fla-org/flash-linear-attention"

TORCH_HEADER = """\
# VERBATIM source of the two PyTorch entry points that were EXECUTED to
# build ../fixtures/rmsnorm_oracle.txt, extracted with inspect.getsource()
# from torch %s (%s) at tag v%s.  The arithmetic they dispatch to
# is COMPILED -- aten/src/ATen/native/layer_norm.cpp::rms_norm, shipped in
# the same wheel -- and is not quotable here, which is why the row records
# the version and the wheel source rather than a C++ excerpt.
"""


def dump_upstream():
    """Re-extract the byte-pinned upstream sources next to this file.

    These two files are the provenance of every number in the fixture, so they
    are written by `inspect.getsource()` and never by hand. torch's own tag
    is printed in the header because it is what makes THAT half reproducible;
    flash-linear-attention is a wheel with no revision, so its sha256 is the
    only handle (see the fixture header). Both headers carry the upstream
    repo URL, which is what ORACLE-TIERS.tsv's tier-(a) rows are required to
    name (tools/oracle_gate.py rule R3).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    up = os.path.join(here, "upstream")
    os.makedirs(up, exist_ok=True)

    torch_src = (
        TORCH_HEADER % (TORCH_VERSION, TORCH_REPO, TORCH_VERSION.split("+")[0])
        + "\n# ---- torch.nn.functional.rms_norm --------------------------------\n"
        + inspect.getsource(F.rms_norm)
        + "\n# ---- torch.nn.RMSNorm.forward -----------------------------------\n"
        + inspect.getsource(nn.RMSNorm.forward)
    )
    with open(os.path.join(up, "torch_rms_norm.py"), "w", encoding="utf-8") as fh:
        fh.write(torch_src)

    fla_src = """\
# VERBATIM source of fla.modules.layernorm.rms_norm_ref, extracted with
# inspect.getsource() from the file that was EXECUTED to build
# ../fixtures/rmsnorm_oracle.txt.  sha256 of that file:
#   %s
# Upstream: %s
#   a PyPI wheel with NO git revision, so the sha256 above is the only handle
#   for the exact bytes these came from.
# Do not edit; re-extract with tests/oracle/gen_rmsnorm_oracle.py --dump.

%s""" % (FLA_SHA, FLA_REPO, inspect.getsource(rms_norm_ref))
    with open(os.path.join(up, "fla_rms_norm_ref.py"), "w", encoding="utf-8") as fh:
        fh.write(fla_src)
    sys.stderr.write(
        "dumped upstream/{torch_rms_norm.py,fla_rms_norm_ref.py}; "
        "sha256(%s) = %s\n" % (FLA_REL, FLA_SHA))


def rel_gap(a, b):
    """Max |a-b| / max(|b|,1) -- the same statistic rmsnorm_oracle.rs uses."""
    a = f64(a).ravel()
    b = f64(b).ravel()
    scale = np.maximum(np.abs(b), 1.0)
    return float(np.max(np.abs(a - b) / scale))


# The test's own bounds, restated here so the GENERATOR refuses to emit a
# fixture the test would only reject.  A golden file that its own gate calls
# vacuous is the defect tier (a) exists to remove; the cheapest place to catch
# it is before it is written.
TOL_REL = 1e-5
DISCRIM_FACTOR = 2.0
EPS_DECISIVE = ["tiny", "micro", "zero_row", "d1", "d2"]
WRONG_COLUMNS = ["out_eps_outside", "out_no_eps", "out_layernorm"]


def selfcheck(name, x, w, eps, outs):
    """Every claim the test makes about this case, checked before it is written.

    Returns (problems, notes): a problem REFUSES the fixture, a note is a
    shape fact the test also knows about.
    """
    problems, notes = [], []
    b, t, d = x.shape
    assert len(w) == d, "%s: gain has %d entries for d = %d" % (name, len(w), d)
    if d > 1 and float(np.max(w) - np.min(w)) <= 0.5:
        problems.append("%s: gain spans [%g, %g] -- a constant gain hides a "
                        "broken broadcast" % (name, np.min(w), np.max(w)))
    if d == 1:
        # A length-1 gain vector is constant BY SHAPE; there is no per-feature
        # axis on which a broadcast can break.  Reported, not enforced: the test
        # makes the same exemption and demands a d >= 2 case exists instead.
        notes.append("%s is d == 1, its gain is one element by shape, so it "
                     "cannot test a per-feature broadcast" % name)
    for col in WRONG_COLUMNS:
        gap = rel_gap(outs["out_torch"], outs[col])
        if name in EPS_DECISIVE and gap <= DISCRIM_FACTOR * TOL_REL:
            problems.append(
                "%s: eps-decisive but only %s is %.3gx TOL from `%s`"
                % (name, name, gap / TOL_REL, col))
    if name == "big":
        for col in WRONG_COLUMNS[:2]:
            gap = rel_gap(outs["out_torch"], outs[col])
            if gap > TOL_REL:
                problems.append("big: `%s` is %s away -- the 'big carries no eps "
                                "power' claim is stale" % (col, gap))
    if name == "micro_eps0":
        gap = rel_gap(outs["out_torch"], outs["out_no_eps"])
        if gap > TOL_REL:
            problems.append("micro_eps0: at eps = 0 the eps-inside and no-eps "
                            "columns must coincide, they are %s apart" % gap)
    return problems, notes


def main():
    names = [n for n, _, _, _ in build_cases()]
    cases = build_cases()
    want_dump = "--dump" in sys.argv

    print("# RMSNorm oracle fixture -- GENERATED, do not hand-edit.")
    print("# generator: tests/oracle/gen_rmsnorm_oracle.py")
    print("# regenerate: see the command in that file's docstring (needs torch + fla)")
    print("# upstream 1: torch==%s -- torch.rms_norm, ATen, %s v%s"
          % (TORCH_VERSION, TORCH_REPO, TORCH_VERSION.split("+")[0]))
    print("# upstream 2: flash-linear-attention==%s -- fla/modules/layernorm.py, %s"
          % (FLA_VERSION, FLA_REPO))
    print("#             sha256(%s) = %s" % (FLA_REL, FLA_SHA))
    print("# numbers: 9 significant digits (%.9g) of the f32 the reference produced")
    print("# out_eps_outside / out_no_eps / out_layernorm / out_axis1 are the WRONG")
    print("# formulas: discriminating columns, so a margin is measured, not assumed.")
    print("")

    worst_upstream = 0.0
    rows, problems, notes = [], [], []
    for name, x, w, eps in cases:
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        wt = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))

        out_torch = F.rms_norm(xt, (w.shape[0],), wt, float(eps)).numpy()
        out_fla = rms_norm_ref(xt, wt, None, eps=float(eps)).numpy()

        gap = float(np.max(np.abs(f64(out_torch) - f64(out_fla))))
        scale = float(np.max(np.abs(f64(out_torch))))
        ratio = gap / max(scale, 1.0)
        worst_upstream = max(worst_upstream, ratio)
        if ratio > UPSTREAM_AGREEMENT:
            sys.exit("REF %s: torch and fla disagree by %.3e (scale %.3e, ratio %.3e)"
                     "; refusing to emit a fixture" % (name, gap, scale, ratio))

        outs = {
            "out_torch": out_torch,
            "out_fla": out_fla,
            "out_eps_outside": out_eps_outside(x, w, eps),
            "out_no_eps": out_no_eps(x, w, eps),
            "out_layernorm": out_layernorm(x, w, eps),
        }
        if x.shape[1] == x.shape[2]:
            outs["out_axis1"] = out_axis1(x, w, eps)
        p, n = selfcheck(name, x, w, eps, outs)
        problems += p
        notes += n
        rows.append((name, x, w, eps, outs))

    if problems:
        sys.stderr.write("\n".join("FIXTURE PROBLEM: " + p for p in problems) + "\n")
        sys.exit("refusing to emit: the fixture would not pass its own guards")

    if want_dump:
        dump_upstream()

    print("meta.cases: %s" % " ".join(names))
    print("meta.upstream_fla_sha256: %s" % FLA_SHA)
    print("meta.upstream_fla_path: %s" % FLA_REL)
    print("meta.upstream_fla_version: %s" % FLA_VERSION)
    print("meta.upstream_torch_version: %s" % TORCH_VERSION)
    print("")

    for name, x, w, eps, outs in rows:
        print("case.%s.dims: %s" % (name, " ".join(str(v) for v in x.shape)))
        print("case.%s.eps: %s" % (name, g9(eps)))
        print("case.%s.x: %s" % (name, fmt(x)))
        print("case.%s.w: %s" % (name, fmt(w)))
        for col in ("out_torch", "out_fla", "out_eps_outside", "out_no_eps",
                    "out_layernorm", "out_axis1"):
            if col in outs:
                print("case.%s.%s: %s" % (name, col, fmt(outs[col])))
        print("")

    sys.stderr.write("upstream mutual agreement, worst over %d cases: %.3e relative\n"
                     % (len(cases), worst_upstream))
    for note in notes:
        sys.stderr.write("note: %s\n" % note)
    sys.stderr.write("fixture self-check: %d cases, every guard satisfiable\n"
                     % len(cases))


if __name__ == "__main__":
    main()
