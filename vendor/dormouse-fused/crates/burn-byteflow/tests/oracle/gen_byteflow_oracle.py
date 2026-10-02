#!/usr/bin/env python3
"""Generate the burn-byteflow coding-rate golden fixture — the ARXIV F64 ORACLE.

    $ python3 oracle/gen_byteflow_oracle.py > fixtures/byteflow_oracle.txt

TIER (b). There is no author code to run: the paper says the full source
"will make the full source code ... publicly available ... as soon as [the
legal review] process is complete" (pdf p.19, Reproducibility Statement), none
is up, and a GitHub search for it came back empty. So the strongest permitted
tier is (b): this generator implements the paper's equations 11-12 READ
FROM THE PDF (`docs/papers/2603.03583-byteflow.pdf`, sha256 pinned in
docs/papers/provenance.tsv) in numpy f64, independently of the Rust, from the
text alone. The Rust test compares `marginal_gains_exact` /
`coding_rate_exact` against these values; if the two disagree, ONE of them
has read the paper wrong, and the number in the fixture is the one that was
read from the paper.

The math this oracle pins (pdf p.5, §3.2; p.16, Appendix A):

    R_eps(h_1:t) = 1/2 log det( I_(t·d) + (d_local/eps^2) * h_1:t h_1:t^T )   (11)

    dR_t = R_eps(h_1:t) - R_eps(h_1:t-1)                                      (12)

The T×T form of (11) is never built; the Sylvester identity
det(I_T + c·HH^T) = det(I_d + c·H^T H) replaces it with d×d everywhere
(ps == 2*ln(sum of the diagonal of the Cholesky)), but the DETERMINANT
IDENTITY is ours, not the paper's — the paper writes the T×T form. That is a
transcription decision carried in the comment above `prefix_logdets`
(src/chunk.rs) and the fixture checks it indirectly: an orthonormal row has
known closed-form values (case 1.1), computed WITHOUT any Sylvester step.

THE FIVE CASES in the fixture
-----------------------------
1. `orthonormal` — rows are the first k unit basis vectors of R^d:
   R must equal k/2 * ln(1 + d/eps^2) exactly BY THE PAPER's eigenvalue
   reading (each contributes ln(1 + d/eps^2)); dR_t must be that constant
   for every t. Checks the eps^2 and d_local factors of eq. (11) alone.
2. `rank_collapse` — rows 2 and 3 identical: R must be finite, LESS than
   case 1 at the same eps, and dR_3 must be SMALLER than dR_2 (novelty
   falling when the representation stops adding directions).
3. `outlier` — 11 small iid rows then one huge (100x) row: dR must peak AT
   THE OUTLIER (top-1 by dR = that position).
4. `telescope` — sum of all dR_t == R_eps(h_1:T) up to f32 rounding (the
   prefix identity between eq. 11 and eq. 12).
5. `signs_positive` — every prefix rate >= 0 on an arbitrary matrix
   (log det of a SPD matrix is a nondecreasing function of prefix length).
   Also the per-fixture eps^2 and d_local echo, so the reader SEES which
   parameters the numbers were produced at.

The exact input matrices are echoed in the fixture as f32 rows, %.9g, so the
test rebuilds the same input from the file and never trusts "same crate".

SIXTEEN SIGNIFICANT DIGITS (`.17g`) for the oracle values: the comparison is
f64-vs-f64 in the Rust fixture (rel bar 2e-9, the Cholesky-of-the-same-matical
condition is inside the bar with margin).

Usage: python3 gen_byteflow_oracle.py [--out fixtures/byteflow_oracle.txt]
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def coding_rate_eps(H: np.ndarray, eps2: float) -> float:
    """eq. (11): 1/2 logdet(I + (d/eps2) H^T H) — d x d via Sylvester.

    H: [T, d] row-major block of h_1:T. I_(t·d) + (d/eps2)HH^T has the same
    nonzero eigenvalues as I_d + (d/eps2) H^T H (plus T-d zeros), so the log
    det is identical. Addressed in the header; not the paper's own form.
    """
    T, d = H.shape
    c = d / eps2
    lam, _ = np.linalg.eigh(c * (H.T @ H))
    return 0.5 * float(np.log1p(lam).sum())


def marginal_gains_eps(H: np.ndarray, eps2: float) -> np.ndarray:
    """eq. (12): dR_t for t = 1..T; dR_1 is R(h_1:1) - R(empty) = R(h_1)."""
    T = H.shape[0]
    out = np.empty(T, dtype=np.float64)
    prev = 0.0
    for t in range(1, T + 1):
        r = coding_rate_eps(H[:t], eps2)
        out[t - 1] = r - prev
        prev = r
    return out


def cholesky_prefix_logdets(H: np.ndarray, eps2: float) -> np.ndarray:
    """The SAME prefix rates via the CHOLESKY route the Rust computes.

    Independent of the eigenvalue route above; a disagreement would be a
    numpy-internal identity failure (not expected), kept because the Rust
    code path is Cholesky and its failure mode (non-SPD -> None -> panic)
    needs a reference that produces the same values — `prefix_logdets` is
    literally transcribed, numerics and all.
    """
    T, d = H.shape
    c = d / eps2
    gram = np.zeros((d, d), dtype=np.float64)
    out = np.empty(T, dtype=np.float64)
    for p in range(T):
        row = H[p].astype(np.float64)
        gram += np.outer(row, row)
        m = np.eye(d) + c * gram
        # R carries the ½ of eq. (11): det = exp(2·Σ ln l_ii), so
        # ½·ln det = Σ ln l_ii. The eigen route already returns the PAPER's
        # R (½ included), so this route lands the ½ in too — the bare det
        # (2·Σ ln l_ii) would be exactly twice the paper's rate.
        l = np.linalg.cholesky(m)
        out[p] = float(np.log(np.diag(l)).sum())
    return out


def dump_matrix(name: str, H: np.ndarray) -> str:
    lines = [f"# matrix {name} [{H.shape[0]}x{H.shape[1]}]"]
    for row in H:
        lines.append("row " + " ".join(f"{v:.9g}" for v in row))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(2603)
    lines = [
        "# byteflow_oracle.txt — the arXiv 2603.03583 f64 R_eps / dR fixture.",
        "# Generator: tests/oracle/gen_byteflow_oracle.py (numpy f64, eigen route).",
        "# Tier (b): paper math, no author code exists (see the generator header",
        "# and docs/papers/provenance.tsv). Format: named cases, each with its",
        "# input rows (f32, %.9g) and the expected values (%.17g), because the",
        "# f32 inputs must be rebuilt identically on the Rust side.",
        "#",
        "# case 1.1 orthonormal_closed_form — eq.(11) against the paper's own",
        "# eigenvalue reading: k orthonormal rows give k/2 * ln(1 + d/eps2).",
        "# case 1.2 orthonormal_marginals — the SAME inequality read as dR:",
        "# each of the first k positions carries the same ln(1 + d/eps2)/2.",
        "# case 2.1 rank_collapse_less_than_orthonormal",
        "# case 2.2 rank_collapse_dR_falls_when_directions_stop",
        "# case 3.1 outlier_dR_peaks_at_the_outlier",
        "# case 4.1 telescope_sum_dr_equals_full_rate",
        "# case 5.1 arbitrary_prefix_rates_nonneg",
        "# case 5.2 cholesky_route_equals_eigen_route",
        "#",
    ]

    eps2 = 0.5

    # 1. orthonormal: d=4, three orthonormal rows.
    H = np.zeros((3, 4))
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0
    r = coding_rate_eps(H, eps2)
    expected = 1.5 * np.log(1.0 + 4.0 / eps2)
    lines += [
        "case orthonormal_closed_form",
        dump_matrix("orthonormal", H),
        f"eps2 {eps2:.17g}",
        f"rate {r:.17g}",
        f"closed_form {expected:.17g}",
        "",
    ]
    dr = marginal_gains_eps(H, eps2)
    lines += [
        "case orthonormal_marginals",
        dump_matrix("orthonormal", H),
        f"eps2 {eps2:.17g}",
        *[f"dr{i+1} {v:.17g}" for i, v in enumerate(dr)],
        "",
    ]

    # 2. rank collapse: identical rows 2..3, compare against the orthonormal
    # case at the same eps and check dR_2 > dR_3.
    H2 = np.zeros((3, 2))
    H2[0, 0] = H2[1, 0] = H2[2, 0] = 1.0
    H2[1, 1] = 1.0  # rows: [1,0], [1,1], [1,0] — row 3 repeats row 1
    r2 = coding_rate_eps(H2, eps2)
    orthonormal_at_d2 = 1.5 * np.log(1.0 + 2.0 / eps2)
    dr2 = marginal_gains_eps(H2, eps2)
    lines += [
        "case rank_collapse",
        dump_matrix("rank_collapse", H2),
        f"eps2 {eps2:.17g}",
        f"rate {r2:.17g}",
        f"orthonormal_same_shape {orthonormal_at_d2:.17g}",
        f"dr1 {dr2[0]:.17g}",
        f"dr2 {dr2[1]:.17g}",
        f"dr3 {dr2[2]:.17g}",
        "",
    ]

    # 3. outlier: 11 small rows then one 100x row.
    H3 = rng.standard_normal((12, 6)) * 0.5
    H3[11] *= 100.0
    dr3 = marginal_gains_eps(H3, eps2)
    lines += [
        "case outlier",
        dump_matrix("outlier", H3),
        f"eps2 {eps2:.17g}",
        f"argmax_dr {int(np.argmax(dr3))}",
        *[f"dr{i+1} {v:.17g}" for i, v in enumerate(dr3)],
        "",
    ]

    # 4. telescope on an arbitrary matrix.
    H4 = rng.standard_normal((9, 6))
    dr4 = marginal_gains_eps(H4, eps2)
    r4 = coding_rate_eps(H4, eps2)
    lines += [
        "case telescope",
        dump_matrix("telescope", H4),
        f"eps2 {eps2:.17g}",
        f"rate {r4:.17g}",
        f"sum_dr {float(dr4.sum()):.17g}",
        *[f"dr{i+1} {v:.17g}" for i, v in enumerate(dr4)],
        "",
    ]

    # 5. negativity probe + the Cholesky/eigen agreement on the same matrix.
    H5 = rng.standard_normal((17, 8))
    dr5 = marginal_gains_eps(H5, eps2)
    chol5 = cholesky_prefix_logdets(H5, eps2)
    lines += [
        "case signs",
        dump_matrix("signs", H5),
        f"eps2 {eps2:.17g}",
        f"min_dr {float(dr5.min()):.17g}",
        *[f"dr{i+1} {v:.17g}" for i, v in enumerate(dr5)],
        f"cholesky_rate_last {chol5[-1]:.17g}",
        f"eigen_rate_last {coding_rate_eps(H5, eps2):.17g}",
        "",
    ]

    text = "\n".join(lines) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"byteflow_oracle: wrote {out} ({len(lines)} lines)", file=sys.stderr)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
