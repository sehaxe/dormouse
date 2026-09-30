"""Dry-run the two class-A GATES in numpy before spending a cargo slot on them.

The Rust gates are the deliverable; this is a cheap check that their LOGIC is
right (a gate that cannot fail, or that fails for a reason other than the one
it names, is worse than no gate). Same fixtures, same arithmetic, f32 dtype
where the Rust is f32.

Run: /tmp/opencode/oracle-venv/bin/python -u gate_dryrun.py
"""

import numpy as np

f32 = np.float32
A, B, C = 15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0
POWER_ITERS = 5
SAFETY = 1.05
BASIN = np.sqrt(7.0 / 3.0)
K = 64
N = 768


def dct_basis(n, k):
    """The Rust fixture `ortho_factor_768x64`, verbatim."""
    out = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        for j in range(k):
            scale = 1.0 / np.sqrt(n) if j == 0 else np.sqrt(2.0 / n)
            out[i, j] = scale * np.cos(np.pi * (2 * i + 1) * j / (2 * n))
    return out


def det_factor(rows, cols):
    """The Rust fixture `det_factor`, verbatim."""
    d = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            d[i, j] = np.sin((i * 7 + j * 13) * 1.0) * 0.5 + np.cos(
                (i * 3 + j * 5) * 1.0
            )
    return d


def canonical(m):
    """The transpose lib.rs:211-215 does: small side first, Gram on [c,c].

    The first cut of this file skipped it and formed the [768,768] Gram, which
    is a DIFFERENT matrix with a different spectrum — that alone made the
    on-manifold estimate exact and the retracted sigma_max come out at 0.68
    instead of 1.0, i.e. it looked exactly like the Frobenius-prescale failure
    this crate spent an audit disproving.
    """
    m = m.astype(np.float32)
    return m.T if m.shape[0] > m.shape[1] else m


def sigma_est(m):
    """lib.rs:229-248, in f32 like the Rust, on the CANONICAL form."""
    m = canonical(m)
    g = (m @ m.T).astype(f32)
    v = g.sum(axis=1).astype(f32)
    for _ in range(POWER_ITERS):
        vn = np.sqrt((v * v).sum()).astype(f32)
        vn = max(vn, f32(1e-12))
        v = (v / vn).astype(f32)
        v = (g @ v).astype(f32)
    gv = (g @ v).astype(f32)
    vgv = (v * gv).sum().astype(f32)
    vv = (v * v).sum().astype(f32)
    return f32(np.sqrt(f32(vgv / max(vv, f32(1e-14)))))


def polar(m, iters):
    """lib.rs:255-260 + prescale, f32, on the CANONICAL form."""
    m = canonical(m)
    g = (m @ m.T).astype(f32)
    v = g.sum(axis=1).astype(f32)
    for _ in range(POWER_ITERS):
        vn = max(f32(np.sqrt((v * v).sum())), f32(1e-12))
        v = (v / vn).astype(f32)
        v = (g @ v).astype(f32)
    gv = (g @ v).astype(f32)
    vgv = (v * gv).sum().astype(f32)
    vv = (v * v).sum().astype(f32)
    s = f32(np.sqrt(f32(vgv / max(vv, f32(1e-14)))))
    s = max(s, f32(1e-7))
    m = (m / f32(s * f32(SAFETY))).astype(f32)
    a, b, c = f32(A), f32(B), f32(C)
    for _ in range(iters):
        xx = (m @ m.T).astype(f32)
        xx2 = (xx @ xx).astype(f32)
        poly = (xx * b + xx2 * c).astype(f32)
        m = (a * m + poly @ m).astype(f32)
    return m.T if m.shape[0] < m.shape[1] else m


def per_entry(u):
    return f32(np.linalg.norm(u.T @ u - np.eye(u.shape[1])) / u.shape[1])


def smax_of(r):
    k = r.shape[1]
    return f32(np.sqrt(f32(np.sum(r.T @ r) / f32(k))))


def gate_basin():
    print("GATE 1  the_quintic_basin_is_sqrt_7_over_3")
    edge = np.sqrt(7.0 / 3.0)
    p = lambda s: 15.0 / 8.0 * s - 5.0 / 4.0 * s**3 + 3.0 / 8.0 * s**5
    dp = lambda s: 15.0 / 8.0 - 15.0 / 4.0 * s**2 + 15.0 / 8.0 * s**4
    ok = True
    ok &= abs(p(1.0) - 1.0) < 1e-15
    ok &= abs(dp(1.0)) < 1e-15
    for s in [0.1, 0.5, 0.95, 1.4, 1.5]:
        ok &= abs(dp(s) - 15.0 / 8.0 * (s * s - 1) ** 2) < 1e-14
    ok &= abs(p(edge) - edge) < 1e-12
    ok &= dp(edge) > 1.0
    def it(s):
        # saturate at inf the way f64 does, so the "diverges" branch is
        # comparable with the Rust rather than raising
        for _ in range(60):
            try:
                s = p(s)
            except OverflowError:
                return float("inf")
            if not np.isfinite(s):
                return float("inf")
        return s
    for s in [1.0, 1.2, 1.4, 1.5, 1.52]:
        ok &= abs(it(s) - 1.0) < 1e-9
    for s in [1.53, 1.6, 2.0]:
        ok &= abs(it(s)) > 1e6
    print(f"  {'PASS' if ok else 'FAIL'}  (1.52 -> {it(1.52):.3e}, 1.53 -> {it(1.53):.3e})")
    return ok


def gate_estimate():
    print("GATE 2  the_sigma_estimate_is_a_lower_bound_and_the_1_05_factor_is_not_what_saves_it")
    basis = dct_basis(N, K)
    ok = True

    def build(s):
        # Each SINGULAR VALUE scales its own column. The first cut of this
        # did `x += basis[:, i:i+1] * si` against a [N,K] zero tensor, which
        # broadcasts the column across all K — the fixture was 64x too large
        # and every number below was wrong. The dtype is asserted because
        # that failure is silent.
        x = np.zeros((N, K), dtype=np.float32)
        for i, si in enumerate(s):
            x[:, i : i + 1] = (basis[:, i : i + 1] * si).astype(np.float32)
        assert np.allclose(
            np.diag(x.T @ x), np.array(s, dtype=np.float32) ** 2, rtol=1e-4
        ), "fixture does not have the prescribed singular values"
        return x

    flat = [1.0] * K
    est_flat = sigma_est(build(flat))
    print(f"  on-manifold estimate {est_flat:.7f} (must be 1 to 1e-5)")
    ok &= abs(est_flat - 1.0) < 1e-5
    prescaled_flat = 1.0 / (est_flat * SAFETY)
    print(f"  on-manifold prescaled {prescaled_flat:.6f} (must be < 1.0)")
    ok &= prescaled_flat < 1.0

    # A NEAR-FLAT spectrum, which is the honest worst case: a flat Gram is
    # where (lambda2/lambda1)^k decays slowest. A log-spaced spectrum (the
    # obvious choice) measures only 1.4% of error.
    for label, s in [
        ("flat 0.9      ", [0.9] * K),
        ("two-level 1/0.9", [1.0] + [0.9] * (K - 1)),
    ]:
        x = build(s)
        est = sigma_est(x)
        frob = f32(np.sqrt(np.sum(x.astype(np.float64) ** 2)))
        true_smax = 1.0
        prescaled = true_smax / (float(est) * SAFETY)
        print(
            f"  {label}: est/sigma_1 = {est / true_smax:.6f} "
            f"({(1 - est / true_smax) * 100:.1f}% low), ||X||_F = {frob:.3f}, "
            f"prescaled = {prescaled:.6f}"
        )
        ok &= est <= true_smax
        ok &= est <= frob
        ok &= prescaled > 1.0
        ok &= prescaled < BASIN
        r = polar(x, 3)
        so = smax_of(r)
        print(f"      retracted sigma_max {so:.7f} (must be 1 to 1e-3)")
        ok &= abs(so - 1.0) < 1e-3
    # the steep spectrum, recorded because it is the one that does NOT refute
    # the comment — a reader deciding whether the gate is cherry-picked needs
    # to see the case that would have supported the old text.
    for spread in [0.01]:
        s = [spread ** (i / (K - 1)) for i in range(K)]
        est = sigma_est(build(s))
        print(
            f"  (recorded) log-spaced 1->{spread}: est/sigma_1 = {est:.6f}, "
            f"prescaled = {1.0 / (float(est) * SAFETY):.6f} — only "
            f"{(1 - float(est)) * 100:.1f}% low, so this fixture would NOT have "
            f"refuted the comment"
        )
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def classb1():
    print("CLASS B-1  retraction_error_grows_with_spectral_spread  (must be RED)")
    # The DCT basis, matching the Rust fixture. The first cut used
    # `polar(det_factor(768,64), 20)`, which is NOT orthonormal at rank 64
    # (measured max |Q^T Q - I| = 0.94, spectrum already 548:1) — so every
    # spread printed the SAME number and the test measured the fixture.
    q = dct_basis(N, K)
    err = np.abs(q.T @ q - np.eye(K)).max()
    print(f"  basis max |Q^T Q - I| = {err:.3e} (must be < 1e-4)")
    red = False
    for spread in [0.3, 0.1, 0.01]:
        s = np.array([spread ** (i / (K - 1)) for i in range(K)], dtype=np.float32)
        m = (q.astype(np.float32) @ np.diag(s)).astype(np.float32)
        r = polar(m, 3)
        pe = per_entry(r)
        fires = pe > 1e-3
        red |= fires
        print(
            f"  spread {spread:<6.3}: per-entry {pe:.3e} "
            f"({pe / 1e-3:6.1f}x latch) {'LATCH FIRES' if fires else ''}"
        )
    print(f"  {'RED as intended' if red else 'NOT RED — the test is a decoration'}")
    return red


def classb2():
    print("CLASS B-2  sigma_max_estimate_diverges  (must be RED)")
    LAM_REST = 0.05
    ones = np.ones(K)
    v = det_factor(K, 1).reshape(K)
    perp = v - ones * (v @ ones) / (ones @ ones)
    u1 = perp / np.linalg.norm(perp)
    leak = abs(u1 @ ones)
    print(f"  fixture leak into the all-ones direction: {leak:.3e} (must be < 1e-5)")
    if leak >= 1e-5:
        print("  NOT RED — the fixture is not adversarial; the assert would fire early")
        return False
    # A FACTOR, not the Gram: passing G forms G.G whose top eigenvalue is 1.0
    # and the estimate comes back correct — the first cut of this test, and a
    # green gate over the finding.
    proj = np.outer(u1, u1)
    m = proj + np.sqrt(LAM_REST) * (np.eye(K) - proj)
    print(f"  fixture max |M M^T - G| = {np.abs(m @ m.T - (proj + LAM_REST*(np.eye(K)-proj))).max():.2e}")
    true_smax = np.linalg.svd(m, compute_uv=False)[0]
    est = sigma_est(m)
    prescaled = true_smax / (float(est) * SAFETY)
    print(
        f"  true sigma_max {true_smax:.6f}; code's estimate {est:.6f}; "
        f"prescaled {prescaled:.4f} vs basin edge {BASIN:.4f}"
    )
    red = est <= 0.9 * true_smax
    print(f"  {'RED as intended' if red else 'NOT RED — the test is a decoration'}")
    return red


if __name__ == "__main__":
    a = gate_basin()
    b = gate_estimate()
    c = classb1()
    d = classb2()
    print()
    print(f"class-A gates green: basin={a} estimate={b}")
    print(f"class-B tests red-on-purpose: B-1={c} B-2={d}")
