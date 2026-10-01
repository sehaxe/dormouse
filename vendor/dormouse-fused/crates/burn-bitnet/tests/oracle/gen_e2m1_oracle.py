#!/usr/bin/env python3
"""The e2m1 grid and its TIE RULE, from torchao's own MX-FP4 quantizer, run.

    $ uv venv --python 3.12 /tmp/opencode/oracle-venv
    $ VIRTUAL_ENV=/tmp/opencode/oracle-venv uv pip install \
          --index-url https://download.pytorch.org/whl/cpu torch numpy tabulate
    $ git clone https://github.com/pytorch/ao.git     # need not be run again
    $ /tmp/opencode/oracle-venv/bin/python gen_e2m1_oracle.py \
          --torchao /path/to/ao > ../fixtures/e2m1_oracle.txt

WHAT RAN, AND WHY IT IS TIER (a)
--------------------------------
Two files from https://github.com/pytorch/ao, both executed, neither written
here:

1. `torchao/prototype/mx_formats/fp_format_spec.py` -- the file that states
   the OCP MX FP4 grid, `float4_e2m1_interesting_values`, as (value, formula,
   sign, exponent, mantissa) rows. Comment in that file: "values for
   fp4_e2m1, as defined in the OCP spec for MXFP4 ... other than the sign,
   there are only 8 values, so just create the table by hand". The spec it
   cites by URL is
   https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
   §5.3.2.
2. `torchao/prototype/mx_formats/mx_tensor.py::to_mx` -- torchao's REAL
   block quantizer, which emits packed 4-bit CODES for
   `torch.float4_e2m1fn_x2`. This is what decides the tie rule, and it is a
   different question from what the grid contains.

    repo   https://github.com/pytorch/ao
    commit 3972ed015091f659418dedf12edb980a8ca56b53  (2026-09-25)
    torch  2.14.0+cpu   (CPU only; the fp4 dtype has no CPU cast kernel, which
           is WHY the reference has to be torchao's quantizer and not a bare
           `tensor.to(torch.float4_e2m1fn_x2)` -- that raises
           `NotImplementedError: "copy_kernel" not implemented for
           'Float4_e2m1fn_x2'`)

WHAT THIS SETTLES, and it is the question the crate's own history got wrong
---------------------------------------------------------------------------
AGENTS.md §3.2 records that `--act-quant fp4` "was not e2m1 at all" until
`9b343d3`: the mantissa rule emitted 0.75, which is not a level of the format,
and the caller scaled each block's max onto 1 instead of onto the format's max
(6), so only {0, 0.5, 0.75, 1} of the eight magnitudes were reachable.

**The grid is now confirmed correct** against torchao's own table: eight
magnitudes, exactly 0, 0.5, 1, 1.5, 2, 3, 4, 6, with 0.75 absent, and 0.5
correctly identified as the single subnormal.

**The TIE RULE was never checked, and it is wrong.** The interior tie points
of the grid are 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0. At every one of them
torchao picks the level with the EVEN CODE -- round-half-to-even, the IEEE
default and what the format's hardware does. Our `fp4_round`
(`crates/dormouse-core/src/act_quant.rs:123-142`) uses
`mask_fill(a >= lo)`, which sends every tie UP the grid, i.e. away from zero
on the positive side. The two disagree at 4 of the 7 interior ties:
0.25, 1.25, 2.5 and 5.0. The disagreement is one full grid step, and at
5.0 it is 4.0 vs 6.0 -- a 50 % error on that input.

This is a defect in `dormouse-core`, NOT in burn-bitnet: `burn-bitnet` has no
e2m1 at all. See docs/archive/research/tier-a-references.md.

NINE SIGNIFICANT DIGITS. `%.9g`, never `%g` -- an f32 needs 9 to round-trip
exactly.
"""

import argparse
import sys

import numpy as np
import torch

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
TORCHAO_REPO = "https://github.com/pytorch/ao"
TORCHAO_SHA = "3972ed015091f659418dedf12edb980a8ca56b53"


def g9(v) -> str:
    return "%.9g" % float(np.float32(v))


def load_reference(ao_root):
    sys.path.insert(0, ao_root)
    from torchao.prototype.mx_formats.mx_tensor import to_mx
    from torchao.prototype.mx_formats.fp_format_spec import (
        float4_e2m1_interesting_values as table,
    )
    # Decode the codes from torchao's OWN (value, s, e, m) rows. The sign bit
    # is the MSB of the 4-bit code, so codes 8..15 are the negatives.
    codes = {}
    for value, _formula, s, e, m, _label in table:
        codes[int(s + e + m, 2)] = value
    return to_mx, codes, table


def quantize(to_mx, codes, values, block=32):
    """torchao's real answer for `values`, as dequantized f32 in the input's units."""
    while len(values) % block:
        values = list(values) + [0.0]
    x = torch.tensor([values], dtype=torch.float32)
    scale, data = to_mx(x, torch.float4_e2m1fn_x2, block)
    s = float(scale.flatten()[0])
    out = []
    for byte in data.flatten().tolist():
        # two fp4 per byte, low nibble first
        for nib in (byte & 0xF, (byte >> 4) & 0xF):
            out.append(codes[nib] * s)
    return out[: len(values)], s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torchao", required=True, help="path to a clone of pytorch/ao")
    args = ap.parse_args()
    to_mx, codes, table = load_reference(args.torchao)

    # ---- 1. the grid, as torchao states it -------------------------------
    mags = sorted({abs(v) for v in codes.values()})
    assert mags == E2M1, "torchao's magnitudes %s != the expected grid %s" % (mags, E2M1)
    assert 0.75 not in mags, "0.75 is not an e2m1 level; if it appears the premise is wrong"
    assert 0.5 in mags and sum(1 for v in mags if v and v < 1) == 1, "exactly one subnormal, 0.5"

    # ---- 2. the tie rule, as torchao's QUANTIZER decides it --------------
    ties = [0.5 * (E2M1[i - 1] + E2M1[i]) for i in range(1, len(E2M1))]
    # pad to the block with a value that is exactly on the grid
    probes = ties + [-t for t in ties] + [0.0] * ((-len(ties) * 2) % 32)
    ref, scale = quantize(to_mx, codes, probes)
    if scale != 1.0:
        # The tie POINTS move with the block scale, so a block whose max is a
        # tie point would test the wrong values. Force scale 1 by keeping the
        # largest magnitude in the block at exactly 6.0 and everything else
        # below it.
        raise SystemExit("block scale is %g, not 1.0 -- the tie points in this "
                         "block are not the grid's tie points" % scale)

    print("# e2m1 grid and TIE RULE -- GENERATED, do not hand-edit.")
    print("# generator: tests/oracle/gen_e2m1_oracle.py")
    print("# upstream: %s" % TORCHAO_REPO)
    print("# commit:   %s  (2026-09-25)" % TORCHAO_SHA)
    print("# file 1:   torchao/prototype/mx_formats/fp_format_spec.py")
    print("#           float4_e2m1_interesting_values  (the OCP MX FP4 grid,")
    print("#           spec section 5.3.2, cited by URL in that file)")
    print("# file 2:   torchao/prototype/mx_formats/mx_tensor.py::to_mx")
    print("#           torchao's real block quantizer -- it decides the TIE RULE")
    print("# torch:    %s  (CPU)" % torch.__version__)
    print("# spec:     https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf")
    print("#")
    print("# grid.magnitudes: the eight positive e2m1 levels, ascending.")
    print("# 0.75 is ABSENT -- it is not a level of the format (AGENTS.md 3.2).")
    print("# grid.tie_X: torchao's own answer at interior tie X.")
    print("# grid.tie_X_rne: what round-half-to-EVEN gives, for comparison.")
    print("# grid.tie_X_ties_up: what our `mask_fill(a >= lo)` gives.")
    print("")
    print("meta.torchao_commit: %s" % TORCHAO_SHA)
    print("meta.torch_version: %s" % torch.__version__)
    print("meta.scale: %s" % g9(scale))
    print("grid.magnitudes: %s" % " ".join(g9(m) for m in mags))
    print("grid.count: %d" % len(mags))
    print("grid.has_0.75: no")
    print("grid.subnormal: %s" % g9(0.5))
    print("")
    disagree = []
    for i, t in enumerate(ties):
        got = ref[i]
        # the two candidates, and round-half-to-even by CODE parity
        j = [k for k in range(1, len(E2M1)) if E2M1[k - 1] < t < E2M1[k]][0]
        a, b = E2M1[j - 1], E2M1[j]
        rne = a if (j - 1) % 2 == 0 else b
        ties_up = b
        if abs(got - rne) > 1e-9:
            raise SystemExit("torchao is NOT round-half-to-even at %g: got %g, "
                             "RNE says %g -- the premise of this file is wrong" % (t, got, rne))
        if abs(ties_up - rne) > 1e-9:
            disagree.append(t)
        print("grid.tie_%s: %s" % (g9(t).replace(".", "p").replace("-", "m"), g9(got)))
        print("grid.tie_%s_rne: %s" % (g9(t).replace(".", "p").replace("-", "m"), g9(rne)))
        print("grid.tie_%s_ties_up: %s" % (g9(t).replace(".", "p").replace("-", "m"), g9(ties_up)))
        print("grid.tie_%s_agrees: %s" % (g9(t).replace(".", "p").replace("-", "m"),
                                          "yes" if abs(ties_up - rne) < 1e-9 else "NO"))
    print("")
    print("grid.torchao_is_round_half_to_even: yes")
    print("grid.ties_where_ours_differs: %d" % len(disagree))
    print("grid.ties_where_ours_differs_list: %s"
          % " ".join(g9(t) for t in disagree))
    sys.stderr.write(
        "grid confirmed: %d magnitudes, 0.75 absent; torchao is round-half-to-even "
        "at all %d interior ties; our ties-up rule differs at %d of them (%s)\n"
        % (len(mags), len(ties), len(disagree),
           ", ".join("%g" % t for t in disagree)))


if __name__ == "__main__":
    main()
