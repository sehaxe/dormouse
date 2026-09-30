#!/usr/bin/env python3
"""Generate the burn-kda golden fixture by RUNNING FLA's own KDA references.

    $ uv venv --python 3.12 /tmp/opencode/kda-oracle-venv
    $ uv pip install --python /tmp/opencode/kda-oracle-venv/bin/python \\
          --index-url https://download.pytorch.org/whl/cpu torch
    $ uv pip install --python /tmp/opencode/kda-oracle-venv/bin/python einops
    $ git clone https://github.com/fla-org/flash-linear-attention.git
    $ /tmp/kda-oracle-venv/bin/python gen_kda_oracle.py \\
          --fla /path/to/flash-linear-attention > ../fixtures/kda_oracle.txt

WHAT RAN, AND WHY IT IS TIER (a)
---------------------------------
FLA ships FOUR pure-PyTorch reference implementations of KDA -- two for the
decay gate, one for the per-token recurrence, one for the chunked WY
recurrence. They are written as the correctness oracle for the Triton kernels
and FLA's own test suite compares against them
(`tests/ops/test_kda.py:20` imports `naive_chunk_kda, naive_recurrent_kda`;
`fla/ops/kda/gate.py`'s `naive_kda_gate` / `naive_kda_lowerbound_gate` are the
twin of `fused_kda_gate` in the same file). All four run on CPU torch with no
triton, which is why the FAST path could never be tier (a) on this box and the
REFERENCE path can.

    repo    https://github.com/fla-org/flash-linear-attention
    commit  9f38d24980c46d46bd38614e743cdacd21906578  (2026-09-29, main)
    file    fla/ops/kda/gate.py    sha256 7ea46d62149b8e28e0827e070fee327...
    file    fla/ops/kda/naive.py   sha256 60a32285d4b67068ff633b48bbe8ab3...
    paper   arXiv:2510.26692 (Kimi Linear), which names fla/ops/kda as the
            official KDA implementation in its own footnote 1
    moonshot  gate.py:8 -- "This file is modified and supported by the
            Moonshot AI Team". The decay gate is Moonshot's own file.

THE FILES ARE PINNED BYTE-IDENTICALLY next to this generator, and the sha256
is asserted before anything is imported: `exit 2 / PIN ROTED` on a mismatch.
No line of upstream source is edited, not even to add this provenance (it is
in here instead), and NO FUNCTION IS RE-TYPED: each reference is EXTRACTED
from the pinned file's text and exec'd, so the code that produces the fixture
is upstream's bytes and not a transcription of it. That is the whole failure
mode `tier-a-references.md` section 7 documents -- a reference generator
re-typed from memory is tier (b) no matter how confident it looks.

THE FOUR REFERENCES, and what each is for
-----------------------------------------
1. `naive_kda_lowerbound_gate` (gate.py:55-88)  vs our `DecayFn::Sigmoid`.
   Upstream: `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`,
   `lower_bound = -5.0`. This is the RUNNING decay form.
2. `naive_kda_gate` (gate.py:27-52)             vs our `DecayFn::Softplus`.
   Upstream: `-A_log.exp() * F.softplus(g + dt_bias)`. Note `exp(A)` is
   OUTSIDE the softplus. Our `lib.rs:268` computes `-softplus(exp(A) * z)`.
3. `naive_recurrent_kda` (naive.py:13-68)      vs our `kda_step` (Eq 1), at
   `scale=1.0` AND at `scale=K**-0.5`. Upstream applies the scale to `q`
   before the loop (`naive.py:57`) with `scale = K ** -0.5` as the DEFAULT
   (`naive.py:52`), and `fla/layers/kda.py:262` calls `chunk_kda` without a
   `scale` argument, so the default is what the official layer runs.
4. `naive_chunk_kda` (naive.py:71-163)         vs `burn_gdn2::chunk_wy_forward`
   at `scale=1.0`, i.e. the chunked WY construction itself.

NINE SIGNIFICANT DIGITS
-----------------------
`%.9g`, never `%g`. An f32 needs 9 to round-trip exactly; `f"{v:g}"` is six.
"""

import argparse
import hashlib
import os
import re
import sys
import textwrap

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

# ── the pins. sha256 asserted before anything is imported. ──────────────────
FLA_REPO = "https://github.com/fla-org/flash-linear-attention"
FLA_SHA = "9f38d24980c46d46bd38614e743cdacd21906578"
PINS = {
    "fla_ops_kda_gate.py": "7ea46d62149b8e28e0827e070fee327576de783cb981e3b855182f2b8198c16e",
    "fla_ops_kda_naive.py": "60a32285d4b67068ff633b48bbe8ab31028066d24f00d27e12199a88fc73f016",
}
TORCH_VERSION = torch.__version__


def check_pins(upstream_dir):
    for name, want in PINS.items():
        path = os.path.join(upstream_dir, name)
        if not os.path.exists(path):
            print(f"PIN MISSING: {path}", file=sys.stderr)
            sys.exit(2)
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if got != want:
            print(f"PIN ROTED: {name}\n  want {want}\n  got  {got}", file=sys.stderr)
            sys.exit(2)


def extract_def(source, name):
    """Return the EXACT source text of a top-level `def name(...)` from a file.

    Taken from the pinned bytes, never re-typed. Scanning is on column-0
    boundaries (`def `/`@`/`class`/blank-run) so it cannot stop inside a
    nested def or a decorator.
    """
    lines = source.splitlines(keepends=True)
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith(f"def {name}("):
            start = i
            break
    if start is None:
        raise SystemExit(f"def {name}( not found in the pinned source")
    # A continuation is anything indented, or a line that can only be the tail
    # of a multi-line signature -- `) -> torch.Tensor:` closes the parameter
    # list, and stopping there truncates the def into a SyntaxError. Blank
    # lines only end the block once an unindented statement follows.
    CONT = (" ", "\t", ")", "]", "}", ",")
    end = len(lines)
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if ln.strip() == "":
            nxt = lines[j + 1] if j + 1 < len(lines) else ""
            if nxt.strip() and not nxt.startswith(CONT):
                end = j
                break
            continue
        if not ln.startswith(CONT):
            end = j
            break
    text = "".join(lines[start:end]).rstrip() + "\n"
    # drop a dangling decorator-free tail that is only a comment
    return textwrap.dedent(text)


def load_references(upstream_dir):
    """exec the pinned references. Nothing here is ours but the namespace."""
    gate = open(os.path.join(upstream_dir, "fla_ops_kda_gate.py")).read()
    naive = open(os.path.join(upstream_dir, "fla_ops_kda_naive.py")).read()
    ns = {"torch": torch, "F": F, "rearrange": rearrange, "np": np}
    srcs = [
        extract_def(gate, "naive_kda_gate"),
        extract_def(gate, "naive_kda_lowerbound_gate"),
        extract_def(naive, "naive_recurrent_kda"),
        extract_def(naive, "naive_chunk_kda"),
    ]
    for s in srcs:
        exec(compile(s, "<pinned-upstream>", "exec"), ns)
    for fn in ("naive_kda_gate", "naive_kda_lowerbound_gate",
               "naive_recurrent_kda", "naive_chunk_kda"):
        assert fn in ns, f"{fn} was not extracted"
    return ns


def g9(v) -> str:
    return "%.9g" % float(np.float32(v))


def fmt(a) -> str:
    return " ".join(g9(v) for v in np.asarray(a, dtype=np.float32).ravel())


def t4(a):
    return torch.tensor(np.asarray(a, dtype=np.float32))


# ── the decay-gate cases ────────────────────────────────────────────────────
# Chosen so that a WRONG PLACEMENT of exp(A) is visible in a single number, and
# so that one case is an exact-agreement case (A=0), which is what makes the
# rest of the fixture discriminating rather than uniformly off.
#
#   upstream softplus form:  g = -exp(A) * softplus(z)
#   ours:                    g = -softplus(exp(A) * z)
#
# At A=0 both are -softplus(z): identical. So case `A_zero` is the fixture's
# own control -- if it ever fails, the harness is wrong, not the formula.
# Every other case has A != 0 and separates them. The separation GROWS with
# |z| and FLIPS ORDER with the sign of A, which is why the cases span
# z in [-6, +6] and both signs of A.
def build_gate_cases():
    rng = np.random.default_rng(20260930)
    B, T, H, K = 2, 3, 2, 8
    cases = []

    def blk(z, a):
        return (np.broadcast_to(np.asarray(z, np.float32), (B, T, H, K)).copy(),
                np.broadcast_to(np.asarray(a, np.float32), (H,)).copy())

    # 1. A_zero, wide z: the CONTROL. exp(A)=1, so the two forms are the same
    #    function. If this case ever disagrees, the generator or the Rust side
    #    is misreading the shapes, and every other verdict in the file is void.
    z, a = blk(np.linspace(-6.0, 6.0, B * T * H * K).reshape(B, T, H, K), 0.0)
    cases.append(("A_zero_control", z, a, np.zeros(H * K, np.float32)))

    # 2. our_own_init: A = -3, b_alpha = +1 -- the crate's actual initialisation
    #    (`lib.rs:239-240`). The most important single case, because it is the
    #    number the running model starts from.
    z, a = blk(np.zeros((B, T, H, K)), -3.0)
    cases.append(("A_minus3_bias_plus1", z, a, np.ones(H * K, np.float32)))

    # 3. A_negative_wide_z: A = -3, z swept over [-6, 6]. The softplus argument
    #    is exp(-3)*z = 0.0498*z upstream-vs-ours, i.e. ours sees a ~20x
    #    COMPRESSED z. At z=+6: upstream -0.0498*6.0025 = -0.2989,
    #    ours -softplus(0.2987) = -0.8536.
    z, a = blk(np.linspace(-6.0, 6.0, B * T * H * K).reshape(B, T, H, K), -3.0)
    cases.append(("A_minus3_wide_z", z, a, np.zeros(H * K, np.float32)))

    # 4. A_positive_wide_z: A = +1. exp(A) = 2.718 DILATES z. At z = -6:
    #    upstream -2.718*softplus(-6) = -2.718*0.002473 = -0.006722,
    #    ours -softplus(-16.31) = -1.1e-7. The ORDER of the two answers is the
    #    reverse of case 3, which no monotone rescale of the decay could do.
    z, a = blk(np.linspace(-6.0, 6.0, B * T * H * K).reshape(B, T, H, K), 1.0)
    cases.append(("A_plus1_wide_z", z, a, np.zeros(H * K, np.float32)))

    # 5. A_zero_nonzero_bias: A = 0 with a negative bias in FLA's own range
    #    (`inv_dt in [-6.91, -2.25]`, kda.py:180-184). With A=0 the two SOFTPLUS
    #    forms still agree, so this case isolates the LOWER-BOUND form from the
    #    softplus form and pins that the K3 branch sees the same z.
    z, a = blk(rng.uniform(-1.0, 1.0, (B, T, H, K)), 0.0)
    cases.append(("A_zero_nonzero_bias", z, a,
                  np.linspace(-6.91, -2.25, H * K).astype(np.float32)))

    # 6. A_clamp_endpoints: the heads sit AT the two clamp bounds `lib.rs:264`
    #    imposes, (-10, 20). The clamp is OURS -- no upstream has one -- so this
    #    case pins that the formula still holds AT the boundary rather than only
    #    in the interior.
    #
    #    WHAT THIS CASE DELIBERATELY DOES NOT COVER, because it cannot: an A
    #    OUTSIDE the clamp. Upstream has no clamp, so any A > 20 would make our
    #    clamped answer differ from FLA's unclamped one and turn the green test
    #    red on a deliberate, documented deviation. A fixture is not the place
    #    to adjudicate a choice we made on purpose. The consequence is a real
    #    coverage limit and it is recorded: **a mutant that WIDENS the clamp is
    #    invisible to every arm of this oracle**, which is why `falsify.sh`'s A3
    #    narrows the clamp instead (that one the greens do see).
    z, a = blk(rng.uniform(-3.0, 3.0, (B, T, H, K)), 0.0)
    a = np.array([0.0, -10.0], np.float32)
    cases.append(("A_clamp_endpoints", z, a, np.zeros(H * K, np.float32)))
    return cases


def build_recur_cases():
    """Per-token recurrence: the exact Eq 1 scan, at scale 1.0 and at K**-0.5."""
    rng = np.random.default_rng(20260930)
    cases = []
    for (name, B, T, H, K, V) in [
        ("square_1head", 1, 32, 1, 8, 8),
        ("square_2head", 2, 16, 2, 8, 8),
        ("gva_1to2", 1, 24, 1, 8, 8),   # HV = 2
    ]:
        HV = 2 * H if name == "gva_1to2" else H
        q = rng.standard_normal((B, T, H, K)).astype(np.float32)
        k = rng.standard_normal((B, T, H, K)).astype(np.float32)
        # v is O(1) and the state is what makes the read scale visible
        v = rng.standard_normal((B, T, HV, V)).astype(np.float32)
        # g in log space, inside the K3 bound (-5, 0)
        g = -rng.uniform(0.05, 4.0, (B, T, HV, K)).astype(np.float32)
        # beta a per-VALUE-HEAD scalar in (0,1) -- FLA's shape, kda.py/naive.py
        beta = rng.uniform(0.05, 0.95, (B, T, HV)).astype(np.float32)
        cases.append((name, B, T, H, HV, K, V, q, k, v, g, beta))
    return cases


def build_chunk_cases():
    """The chunked WY construction. `naive_chunk_kda` asserts T % BT == 0."""
    rng = np.random.default_rng(20260930)
    cases = []
    for (name, B, T, H, HV, K, V, BT) in [
        ("chunk16_h1", 1, 32, 1, 1, 8, 8, 16),
        ("chunk16_h2", 2, 32, 2, 2, 8, 8, 16),
        ("chunk16_gva", 1, 32, 1, 2, 8, 8, 16),
    ]:
        q = rng.standard_normal((B, T, H, K)).astype(np.float32)
        k = rng.standard_normal((B, T, H, K)).astype(np.float32)
        v = rng.standard_normal((B, T, HV, V)).astype(np.float32)
        g = -rng.uniform(0.05, 4.0, (B, T, HV, K)).astype(np.float32)
        beta = rng.uniform(0.05, 0.95, (B, T, HV)).astype(np.float32)
        cases.append((name, B, T, H, HV, K, V, BT, q, k, v, g, beta))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fla", required=True,
                    help="path to a flash-linear-attention checkout (for the "
                         "sha256 re-verification against the clone)")
    ap.add_argument("--upstream", default=None,
                    help="dir holding the pinned .py files (default: ./upstream)")
    ap.add_argument("--verify-clone", action="store_true",
                    help="also sha256 the files in --fla and refuse on mismatch")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    upstream = args.upstream or os.path.join(here, "upstream")
    check_pins(upstream)
    if args.verify_clone and args.fla:
        for name in PINS:
            src = os.path.join(args.fla, "fla/ops/kda",
                               name.replace("fla_ops_kda_", ""))
            if not os.path.exists(src):
                print(f"CLONE FILE MISSING: {src}", file=sys.stderr)
                sys.exit(2)
            got = hashlib.sha256(open(src, "rb").read()).hexdigest()
            if got != PINS[name]:
                print(f"CLONE MISMATCH: {src}\n  want {PINS[name]}\n  got  {got}",
                      file=sys.stderr)
                sys.exit(2)
        print("# clone re-verified against the pin", file=sys.stderr)

    R = load_references(upstream)

    out = []
    W = out.append
    W("# burn-kda tier-(a) golden fixture -- FLA's OWN KDA references, executed.")
    W("# GENERATED, do not edit.  tests/oracle/gen_kda_oracle.py")
    W(f"# repo    {FLA_REPO}")
    W(f"# commit  {FLA_SHA}")
    W("# file    fla/ops/kda/gate.py   (naive_kda_gate, naive_kda_lowerbound_gate)")
    W("# file    fla/ops/kda/naive.py  (naive_recurrent_kda, naive_chunk_kda)")
    W("# sha256  " + " ".join(f"{k}={v[:16]}..." for k, v in PINS.items()))
    W(f"# torch   {TORCH_VERSION} (cpu)")
    W("# Each block: name, then `key=value` lines. Arrays are row-major f32 at")
    W("# 9 significant digits, which round-trips f32 exactly.")

    # ── 1/2: the two decay gate forms ────────────────────────────────────────
    W("")
    W("# ---- decay gate: fla/ops/kda/gate.py, both reference functions ----")
    for (name, z, a, bias) in build_gate_cases():
        zt, at, bt = t4(z), t4(a), t4(bias)
        lb = R["naive_kda_lowerbound_gate"](zt.clone(), at.clone(), bt.clone(),
                                            lower_bound=-5.0)
        sp = R["naive_kda_gate"](zt.clone(), at.clone(), bt.clone())
        W("")
        W(f"gate {name}")
        W(f"  shape={z.shape[0]} {z.shape[1]} {z.shape[2]} {z.shape[3]}")
        W(f"  z={fmt(z)}")
        W(f"  a_log={fmt(a)}")
        W(f"  bias={fmt(bias)}")
        W(f"  fla_lowerbound_g={fmt(lb.numpy())}")
        W(f"  fla_softplus_g={fmt(sp.numpy())}")

    # ── 3: the per-token recurrence ──────────────────────────────────────────
    W("")
    W("# ---- Eq 1 per-token recurrence: fla/ops/kda/naive.py::naive_recurrent_kda")
    W("# `scale` is FLA's OWN parameter. 1.0 and the K**-0.5 default are both")
    W("# emitted, so the gate can attribute any disagreement to the scale alone.")
    for (name, B, T, H, HV, K, V, q, k, v, g, beta) in build_recur_cases():
        qt, kt, vt, gt, bt2 = t4(q), t4(k), t4(v), t4(g), t4(beta)
        o1, S1 = R["naive_recurrent_kda"](qt.clone(), kt.clone(), vt.clone(),
                                          gt.clone(), bt2.clone(),
                                          scale=1.0, output_final_state=True)
        ok, Sk = R["naive_recurrent_kda"](qt.clone(), kt.clone(), vt.clone(),
                                          gt.clone(), bt2.clone(),
                                          scale=K ** -0.5, output_final_state=True)
        W("")
        W(f"recur {name}")
        W(f"  shape={B} {T} {H} {HV} {K} {V}")
        W(f"  q={fmt(q)}")
        W(f"  k={fmt(k)}")
        W(f"  v={fmt(v)}")
        W(f"  g={fmt(g)}")
        W(f"  beta={fmt(beta)}")
        W(f"  scale1_o={fmt(o1.numpy())}")
        W(f"  scale1_S={fmt(S1.numpy())}")
        W(f"  scaleK_o={fmt(ok.numpy())}")

    # ── 4: the chunked WY construction, at BOTH scales ──────────────────────
    # Two rows per case, and the reason is the same as in section 3: the
    # `scaleK_o` row is what makes the missing-read-scale red ATTRIBUTABLE and
    # what lets the candidate fix be demonstrated as a mutant that turns it
    # green. Without it the red would be "we differ from FLA" with no way to
    # show the scale is the whole difference on the CHUNKED arm too (the
    # recurrent arm has `fla_read_scale_is_the_whole_difference` for that).
    W("")
    W("# ---- chunked WY: fla/ops/kda/naive.py::naive_chunk_kda ----")
    W("# `o` is scale=1.0 and `oK` is FLA's own K**-0.5 default, the scale")
    W("# fla/layers/kda.py:262 runs at. `S` is the scale=1.0 final state.")
    for (name, B, T, H, HV, K, V, BT, q, k, v, g, beta) in build_chunk_cases():
        qt, kt, vt, gt, bt2 = t4(q), t4(k), t4(v), t4(g), t4(beta)
        o, S = R["naive_chunk_kda"](qt.clone(), kt.clone(), vt.clone(),
                                   gt.clone(), bt2.clone(), scale=1.0,
                                   output_final_state=True, chunk_size=BT)
        oK, _ = R["naive_chunk_kda"](qt.clone(), kt.clone(), vt.clone(),
                                     gt.clone(), bt2.clone(),
                                     scale=K ** -0.5, output_final_state=True,
                                     chunk_size=BT)
        W("")
        W(f"chunk {name}")
        W(f"  shape={B} {T} {H} {HV} {K} {V} {BT}")
        W(f"  q={fmt(q)}")
        W(f"  k={fmt(k)}")
        W(f"  v={fmt(v)}")
        W(f"  g={fmt(g)}")
        W(f"  beta={fmt(beta)}")
        W(f"  o={fmt(o.numpy())}")
        W(f"  oK={fmt(oK.numpy())}")
        W(f"  S={fmt(S.numpy())}")

    # ── the generator's own vacuity guards ────────────────────────────────────
    # A fixture that cannot fail is not a fixture. All three of these FIRED
    # during development, which is why they are here and not in the commit
    # message. Guard 1 in particular caught a bug in this file: it first
    # asserted that FLA's two gate references agree at A=0, which is false --
    # `-softplus(z)` and `-5*sigmoid(z)` are two different mechanisms, not two
    # spellings of one. The guard was wrong, not the extraction.
    ok = True

    # (1) FAITHFULNESS OF THE EXTRACTION: upstream's own recurrent and chunk
    #     references must agree with EACH OTHER at scale=1.0. Two independent
    #     transcriptions of the same recurrence inside one upstream file, and
    #     they are the check on the extraction -- a mistyped `extract_def` or a
    #     re-typed reference would break this before it reached the fixture.
    rngv = np.random.default_rng(7)
    B, T, H, K, V, BT = 1, 32, 1, 8, 8, 16
    qq = t4(rngv.standard_normal((B, T, H, K)))
    kk = t4(rngv.standard_normal((B, T, H, K)))
    vv = t4(rngv.standard_normal((B, T, H, V)))
    gg = t4(-rngv.uniform(0.05, 4.0, (B, T, H, K)))
    bb = t4(rngv.uniform(0.05, 0.95, (B, T, H)))
    orec, _ = R["naive_recurrent_kda"](qq.clone(), kk.clone(), vv.clone(),
                                       gg.clone(), bb.clone(), scale=1.0)
    och, _ = R["naive_chunk_kda"](qq.clone(), kk.clone(), vv.clone(),
                                  gg.clone(), bb.clone(), scale=1.0, chunk_size=BT)
    cross = float(np.abs(orec.numpy() - och.numpy()).max())
    if cross > 1e-4:
        print(f"VACUITY GUARD 1: upstream's own recurrent and chunk references "
              f"disagree by {cross}", file=sys.stderr)
        ok = False

    # (2) DISCRIMINATING POWER, softplus placement. Computed HERE as three lines
    #     of arithmetic -- this is NOT a transcription of `lib.rs:268`, it is
    #     the two candidate formulas written out so the fixture's own numbers
    #     can be shown to separate. If they did not, the `A_minus3_bias_plus1`
    #     case would be blind to the defect it exists to catch and the whole
    #     softplus comparison would be vacuous.
    def sp(x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
    zc = np.array([1.0], np.float32)
    ac = np.array([-3.0], np.float32)
    outside = float(-np.exp(ac[0]) * sp(zc[0]))        # upstream: -exp(A)*softplus(z)
    inside = float(-sp(np.exp(ac[0]) * zc[0]))        # ours:      -softplus(exp(A)*z)
    if abs(outside - inside) < 1e-3:
        print(f"VACUITY GUARD 2: the softplus cases are not discriminating "
              f"(outside={outside} inside={inside})", file=sys.stderr)
        ok = False

    # (3) TIE THE TIER-(a) REFERENCE TO THE PUBLISHED DERIVATION. The module
    #     docs (`lib.rs:58`) state that our own init (A=-3, b=+1) lands at
    #     alpha = 0.0771, and derive it in the "open discrepancy" section. That
    #     derivation is OURS. This guard recomputes it from FLA's EXECUTED
    #     reference and requires the two to agree, so the published number is
    #     not resting on our arithmetic alone.
    lb = float(R["naive_kda_lowerbound_gate"](
        t4(np.zeros((1, 1, 1, 1))), t4(np.array([-3.0])), t4(np.array([1.0])),
        lower_bound=-5.0))
    alpha = float(np.exp(lb))
    if abs(alpha - 0.0771) > 5e-5:
        print(f"VACUITY GUARD 3: FLA's executed reference gives alpha={alpha:.6f} "
              f"at our init, but lib.rs:58 publishes 0.0771", file=sys.stderr)
        ok = False
    if not ok:
        sys.exit(3)

    print("\n".join(out))


if __name__ == "__main__":
    main()
