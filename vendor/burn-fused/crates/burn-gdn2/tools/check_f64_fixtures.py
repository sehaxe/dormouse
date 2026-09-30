#!/usr/bin/env python3
"""Is every committed `ref_f64*.bin` the output of THIS generator?

WHY THIS FILE EXISTS, and the defect it was written for. `ca45600` changed one
line of the f64 generator's maths -

    -    g = g.T.reshape(H, t, HK)                        # head-major
    +    g = g.T.reshape(t, H, HK).transpose(1, 0, 2)      # token-major, then [H,T,HK]

- regenerated `ref_f64.bin` and `ref_f64_faults.bin`, and did NOT regenerate
`ref_f64_broad.bin`. The 1000-case breadth fixture was therefore left holding
the head-major layout bug, which is correct at T=1 and wrong at every T>1, and
`tests/oracle_breadth.rs` + `tests/oracle_chunk.rs` went red on 976/1000 cases
with `1.851e-01` at case 1. The suite spent a day being read as "the f64 oracle
found a real pre-existing defect in the kernel". It had not: it found a
generator that no longer matched its fixture, which is this file's subject and
nothing else.

The class is the one the previous generation died of. `ref_data.bin` agreed with
the replicate-pad bug because BOTH generators shared it; this fixture agreed
with a generator that was itself wrong, for the same reason and one commit
later. A fixture is evidence, and evidence goes stale silently unless something
re-derives it.

WHAT IT CHECKS, and why the two tolerances are not the same. Weights and inputs
come out of a splitmix64 + Box-Muller stream consumed in a fixed order, so they
must be BIT-identical - no tolerance, because any difference there is a changed
input, not rounding. The OUTPUTS are f64 numpy reductions whose summation order
depends on the array's memory layout, and a layout change moves the last bit.
Measured on this generator against its own committed files: <= 4.4e-15 relative
(`ref_f64_faults.bin`, `decay-sign`), typically ~7e-16. So the f64 bar is
`OUT_TOL = 1e-12`, which is 3.6x above the worst observed round-off and 4.1e+11
below the smallest wrong formula (4.08e-01, `output-gate-sigmoid`) - i.e. the
bar has margin on BOTH sides and cannot pass a semantic change or trip on
rounding. It is deliberately NOT `git diff --exit-code`: that recipe is in two
test headers and it is wrong, because a correct regeneration is not
byte-identical (see `oracle_breadth.rs`'s header).

    python3 tools/check_f64_fixtures.py          # from crates/burn-gdn2
    tools/check_f64_fixtures.py --verbose        # per-case worst, not just per-file

Exits 0 if all three fixtures are current, 1 with the offending file named if
any is stale.
"""

import argparse
import contextlib
import io
import os
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import gen_reference_f64 as G  # noqa: E402

TESTS = "tests"

# See the module docstring for where 1e-12 comes from and why it is not
# byte-equality.
OUT_TOL = 1e-12

HEADER_F64 = b"GDN2F64\0"
HEADER_FAULT = b"GDN2FLT\0"


def parse_main(raw):
    """`GDN2F64` -> (d, tensors{name: (shape, f32-as-f64)}, cases[(t, x, y)])."""
    off = 0
    assert raw[:8] == HEADER_F64, "not a GDN2F64 fixture"
    off = 8
    d = struct.unpack_from("<4I", raw, off)
    off += 16
    off += 8  # f64 expand_v
    use_sc, allow_neg, n_tensors = raw[off], raw[off + 1], raw[off + 2]
    off += 3
    tens = {}
    for _ in range(n_tensors):
        (nl,) = struct.unpack_from("<I", raw, off)
        off += 4
        name = raw[off : off + nl].decode()
        off += nl
        (nd, n) = struct.unpack_from("<2I", raw, off)
        off += 8
        shape = struct.unpack_from(f"<{nd}i", raw, off)
        off += 4 * nd
        tens[name] = (shape, np.frombuffer(raw, "<f4", n, off).astype(np.float64).copy())
        off += 4 * n
    (nc,) = struct.unpack_from("<I", raw, off)
    off += 4
    cases = []
    for _ in range(nc):
        (t,) = struct.unpack_from("<I", raw, off)
        off += 4
        x = np.frombuffer(raw, "<f4", t * d[0], off).astype(np.float64).reshape(t, d[0])
        off += 4 * t * d[0]
        y = np.frombuffer(raw, "<f8", t * d[0], off).reshape(t, d[0])
        off += 8 * t * d[0]
        cases.append((t, x.copy(), y.copy()))
    assert off == len(raw), f"trailing bytes: stopped at {off} of {len(raw)}"
    return d, tens, cases, (use_sc, allow_neg)


def parse_faults(raw):
    """`GDN2FLT` -> (case_index, T, input_fnv1a64, {name: y})."""
    assert raw[:8] == HEADER_FAULT, "not a GDN2FLT fixture"
    ci, nf, t, h = struct.unpack_from("<3IQ", raw, 8)
    off = 28
    out = {}
    for _ in range(nf):
        (nl,) = struct.unpack_from("<I", raw, off)
        off += 4
        name = raw[off : off + nl].decode()
        off += nl
        out[name] = np.frombuffer(raw, "<f8", t * 64, off).copy()
        off += 8 * t * 64
    assert off == len(raw), f"trailing bytes: stopped at {off} of {len(raw)}"
    return ci, t, h, out


def rel(a, b):
    """max|a-b| / max|b|, the same relative measure `oracle_breadth.rs` uses."""
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def check_main(name, committed, regenerated, verbose):
    dc, tc, cc, hc = parse_main(committed)
    dr, tr, cr, hr = parse_main(regenerated)
    bad = []

    if dc != dr or hc != hr:
        bad.append(f"  header differs: committed {dc} {hc} vs regenerated {dr} {hr}")
    if sorted(tc) != sorted(tr):
        bad.append(f"  tensor set differs: {sorted(tc)} vs {sorted(tr)}")
        return bad
    for k in sorted(tc):
        if tc[k][0] != tr[k][0]:
            bad.append(f"  {k}: shape {tc[k][0]} vs {tr[k][0]}")
        elif not np.array_equal(tc[k][1], tr[k][1]):
            # f32 -> f64 keeps the bits, so this is exact equality on the
            # stored f32 values. No tolerance: the RNG stream is deterministic.
            bad.append(f"  {k}: WEIGHTS DIFFER (max {rel(tr[k][1], tc[k][1]):.3e})")
    if len(cc) != len(cr):
        bad.append(f"  case count {len(cc)} vs {len(cr)}")
        return bad

    worst_y, worst_i, worst_x = 0.0, None, 0.0
    for i, ((t, x, y), (rt, rx, ry)) in enumerate(zip(cc, cr)):
        if t != rt:
            bad.append(f"  case {i}: T {t} vs {rt}")
            continue
        worst_x = max(worst_x, rel(rx, x))
        d = rel(ry, y)
        if d > worst_y:
            worst_y, worst_i = d, i
    if worst_x != 0.0:
        bad.append(f"  INPUTS DIFFER (worst {worst_x:.3e} rel)")
    if worst_y > OUT_TOL:
        t = cc[worst_i][0]
        bad.append(
            f"  OUTPUTS DIFFER: worst {worst_y:.6e} rel at case {worst_i} (T={t}), "
            f"bar {OUT_TOL:.0e}"
        )
    if verbose:
        print(f"    {name}: weights exact, inputs exact, outputs {worst_y:.3e} rel "
              f"(bar {OUT_TOL:.0e})")
    return bad


def check_faults(name, committed, regenerated, verbose):
    ci, t, h, fc = parse_faults(committed)
    ri, rt, rh, fr = parse_faults(regenerated)
    bad = []
    if (ci, t, h) != (ri, rt, rh):
        bad.append(f"  anchor differs: committed {(ci, t, hex(h))} vs "
                   f"regenerated {(ri, rt, hex(rh))}")
        return bad
    if sorted(fc) != sorted(fr):
        bad.append(f"  fault set differs: {sorted(fc)} vs {sorted(fr)}")
        return bad
    worst, worst_f = 0.0, None
    for k in sorted(fc):
        d = rel(fr[k], fc[k])
        if d > worst:
            worst, worst_f = d, k
    if worst > OUT_TOL:
        bad.append(f"  OUTPUTS DIFFER: worst {worst:.6e} rel ({worst_f}), bar {OUT_TOL:.0e}")
    if verbose:
        print(f"    {name}: anchor case {ci} (T={t}) input fnv1a64 {h:#018x} matches, "
              f"outputs {worst:.3e} rel")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    P = G.make_params(G.Rng(G.SEED))
    tmp = __file__.rsplit("/", 2)[0]  # crates/burn-gdn2

    import os
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        # The 18-case matrix and its wrong-formula companion, written by ONE
        # generator call because that is the only way the fault anchor and the
        # main fixture can be consistent. `write_fault_fixture` prints what it
        # wrote; a checker's own output is its verdict, so that line is dropped.
        G.write_fixture(os.path.join(d, "ref_f64.bin"), P)
        with contextlib.redirect_stdout(io.StringIO()):
            G.write_fault_fixture(os.path.join(d, "ref_f64_faults.bin"), P)
        G.write_fixture(
            os.path.join(d, "ref_f64_broad.bin"), P, None,
            [G.broad_seq_len(i) for i in range(G.BROAD_CASES)],
        )

        jobs = [
            ("ref_f64.bin", check_main),
            ("ref_f64_faults.bin", check_faults),
            ("ref_f64_broad.bin", check_main),
        ]
        stale = []
        for fname, fn in jobs:
            path = os.path.join(tmp, TESTS, fname)
            if not os.path.exists(path):
                stale.append((fname, ["  committed file is MISSING"]))
                continue
            problems = fn(
                fname, open(path, "rb").read(), open(os.path.join(d, fname), "rb").read(),
                args.verbose,
            )
            if problems:
                stale.append((fname, problems))
            elif args.verbose:
                pass
            else:
                print(f"  OK  {fname}")

    if not stale:
        print("OK  every committed ref_f64*.bin is the output of this generator "
              f"(f32 exact, f64 to {OUT_TOL:.0e}).")
        return 0

    print()
    print("STALE FIXTURE. The generator and the committed bytes disagree:")
    for fname, problems in stale:
        print(f"  {TESTS}/{fname}")
        for p in problems:
            print(p)
    print()
    print("READ THIS BEFORE REGENERATING. A disagreement here is EVIDENCE, not")
    print("noise: the previous generation of this fixture agreed with the")
    print("replicate-pad bug, and regenerating it over that disagreement would")
    print("have re-committed the bug as the truth. Work out WHICH side moved -")
    print("`git log -p -- tools/gen_reference_f64.py` for a generator change, and")
    print("whether the production code moved with it - and write the explanation")
    print("down before you write the bytes. Then, from crates/burn-gdn2:")
    print("    python3 tools/gen_reference_f64.py --broad        # the sweep")
    print("    python3 tools/gen_reference_f64.py               # matrix + faults")
    print("and re-run this. Do not widen OUT_TOL to make it pass: the wrong")
    print("formulas are 4.1e-01 away, so a real defect can never be within 1e-12")
    print("and a tolerance that has to be widened was never measuring the right")
    print("thing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
