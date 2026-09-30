// Standalone instrument for the spectral audit (2026-10-01).
//
// WHAT: a transcription of burn-spectral's polar retraction (src/lib.rs:208-266)
// and of the Stiefel metric (param.rs:205 / lib.rs:1340), in f32, at dormouse
// `small`'s real factor shapes. It prices the per-step retraction and answers
// "what does it defend against, and can that happen at our shapes".
//
// WHY STANDALONE: `cargo test -p burn-spectral` from vendor/burn-fused is a
// 686-package / 41 GB / ~1400 s cold build (AGENTS.md 2.5, and the 2026-09-29
// memory entry about the freeze three parallel agents caused). This file needs
// no cargo, no burn, and no vendor target dir: 0.4 s to compile.
//
// HONESTY TIER: tier (b) transcription of our own code. The ONLY external
// anchor is burn-spectral's own doc-comment table (src/lib.rs:188-191), which
// section A reproduces. If those four numbers move, this file is wrong and
// every number after them is void -- that is why they are asserted in prose
// rather than in code: a green assert on a wrong transcription is a green lie.
//
// RUN: rustc -O -o /tmp/polar_probe tools/polar_probe.rs && /tmp/polar_probe

type M = Vec<f32>; // row-major

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> M {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            if av == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += av * b[p * n + j];
            }
        }
    }
    c
}
fn transpose(a: &[f32], m: usize, n: usize) -> M {
    let mut t = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            t[j * m + i] = a[i * n + j];
        }
    }
    t
}
fn scale_in_place(a: &mut [f32], s: f32) {
    for v in a.iter_mut() {
        *v *= s;
    }
}
fn frob(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Per-entry orthonormality error of a [rows, k] factor: ||U^T U - I||_F / k.
/// Same definition as `LinearLike::max_ortho` (param.rs:205) and the crate's
/// `ortho_err_per_entry` test helper (lib.rs:1340).
fn ortho_per_entry(u: &[f32], rows: usize, k: usize) -> f32 {
    let g = matmul(&transpose(u, rows, k), u, k, rows, k); // U^T U [k,k]
    let mut acc = 0.0f32;
    for i in 0..k {
        for j in 0..k {
            let want = if i == j { 1.0 } else { 0.0 };
            let d = g[i * k + j] - want;
            acc += d * d;
        }
    }
    acc.sqrt() / k as f32
}

/// Largest singular value of a [rows, k] factor, by power iteration on U^T U.
fn sigma_max(m: &[f32], rows: usize, k: usize) -> f32 {
    let g = matmul(&transpose(m, rows, k), m, k, rows, k);
    let mut v = vec![1.0f32; k];
    for _ in 0..200 {
        let nv = matmul(&g, &v, k, k, 1);
        let n = frob(&nv);
        for i in 0..k {
            v[i] = nv[i] / n;
        }
    }
    let gv = matmul(&g, &v, k, k, 1);
    let mut vgv = 0.0f32;
    for i in 0..k {
        vgv += v[i] * gv[i];
    }
    vgv.sqrt()
}

/// burn-spectral lib.rs:229-266, sigma_max prescale, verbatim op order.
fn polar_sigma_max(x: &[f32], rows: usize, k: usize, iters: usize, power: usize) -> M {
    let (mut m, c, r, transposed) = if rows > k {
        (transpose(x, rows, k), k, rows, true)
    } else {
        (x.to_vec(), rows, k, false)
    };
    let g = matmul(&m, &transpose(&m, c, r), c, r, c); // [c,c]
    let mut v = vec![0.0f32; c]; // g . 1
    for i in 0..c {
        for j in 0..c {
            v[i] += g[i * c + j];
        }
    }
    for _ in 0..power {
        let mut vv = 0.0f32;
        for i in 0..c {
            vv += v[i] * v[i];
        }
        let vn = vv.sqrt().max(1e-12);
        for i in 0..c {
            v[i] /= vn;
        }
        v = matmul(&g, &v, c, c, 1);
    }
    let gv = matmul(&g, &v, c, c, 1);
    let mut vgv = 0.0f32;
    let mut vv = 0.0f32;
    for i in 0..c {
        vgv += v[i] * gv[i];
        vv += v[i] * v[i];
    }
    let sigma = (vgv / vv.max(1e-14)).sqrt().max(1e-7);
    scale_in_place(&mut m, 1.0 / (sigma * 1.05));
    let (a, b, cc) = (15.0f32 / 8.0, -5.0f32 / 4.0, 3.0f32 / 8.0);
    for _ in 0..iters {
        let xx = matmul(&m, &transpose(&m, c, r), c, r, c);
        let xx2 = matmul(&xx, &xx, c, c, c);
        let mut poly = xx.clone();
        scale_in_place(&mut poly, b);
        for i in 0..c * c {
            poly[i] += cc * xx2[i];
        }
        let pm = matmul(&poly, &m, c, c, r);
        for i in 0..m.len() {
            m[i] = a * m[i] + pm[i];
        }
    }
    if transposed {
        transpose(&m, c, r)
    } else {
        m
    }
}

/// The rejected alternative the doc comment warns about (lib.rs:200-204):
/// Frobenius prescale instead of sigma_max. Kept ONLY as the anchor for the
/// doc-comment table; nothing in the crate uses it.
fn polar_frobenius(x: &[f32], rows: usize, k: usize, iters: usize) -> M {
    let (mut m, c, r, transposed) = if rows > k {
        (transpose(x, rows, k), k, rows, true)
    } else {
        (x.to_vec(), rows, k, false)
    };
    let mut fro = 0.0f32;
    for v in m.iter() {
        fro += v * v;
    }
    scale_in_place(&mut m, 1.0 / fro.sqrt());
    let (a, b, cc) = (15.0f32 / 8.0, -5.0f32 / 4.0, 3.0f32 / 8.0);
    for _ in 0..iters {
        let xx = matmul(&m, &transpose(&m, c, r), c, r, c);
        let xx2 = matmul(&xx, &xx, c, c, c);
        let mut poly = xx.clone();
        scale_in_place(&mut poly, b);
        for i in 0..c * c {
            poly[i] += cc * xx2[i];
        }
        let pm = matmul(&poly, &m, c, c, r);
        for i in 0..m.len() {
            m[i] = a * m[i] + pm[i];
        }
    }
    if transposed {
        transpose(&m, c, r)
    } else {
        m
    }
}

/// Exactly-orthonormal [rows, k] factor (rows >= k), f32, two-pass MGS.
fn ortho_factor(rows: usize, k: usize, seed: u64) -> M {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / 8388608.0) - 1.0 // [-1,1)
    };
    let mut a = vec![0.0f32; rows * k];
    for v in a.iter_mut() {
        *v = next();
    }
    for p in 0..k {
        for q in 0..p {
            let mut d = 0.0f32;
            for i in 0..rows {
                d += a[i * k + p] * a[i * k + q];
            }
            for i in 0..rows {
                a[i * k + p] -= d * a[i * k + q];
            }
        }
        let mut nrm = 0.0f32;
        for i in 0..rows {
            nrm += a[i * k + p] * a[i * k + p];
        }
        nrm = nrm.sqrt();
        for i in 0..rows {
            a[i * k + p] /= nrm;
        }
    }
    a
}

/// A drift of Frobenius size `delta`, `frac` of it along one FIXED direction
/// (a persistent gradient sign) and the rest fresh every call (a random walk).
fn drift(u: &[f32], rows: usize, k: usize, delta: f32, frac: f32, seed: u64) -> M {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / 8388608.0) - 1.0
    };
    let mut sys = vec![0.0f32; rows * k];
    for v in sys.iter_mut() {
        *v = next();
    }
    let sn = frob(&sys);
    for v in sys.iter_mut() {
        *v *= delta * frac / sn;
    }
    let mut rnd = vec![0.0f32; rows * k];
    for v in rnd.iter_mut() {
        *v = next();
    }
    let rn = frob(&rnd);
    for v in rnd.iter_mut() {
        *v *= delta * (1.0 - frac).sqrt() / rn;
    }
    (0..rows * k).map(|i| u[i] + sys[i] + rnd[i]).collect()
}

/// RMS over columns of 1 - cos(angle) between corresponding columns: how far
/// the retraction ROTATES the factor, which is the only part a scale-equivariant
/// factor quantizer cannot absorb.
fn col_angle(a: &[f32], b: &[f32], rows: usize, k: usize) -> f32 {
    let mut acc = 0.0f32;
    for j in 0..k {
        let (mut ab, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..rows {
            ab += a[i * k + j] * b[i * k + j];
            na += a[i * k + j] * a[i * k + j];
            nb += b[i * k + j] * b[i * k + j];
        }
        let c = ab / (na.sqrt() * nb.sqrt()).max(1e-20);
        acc += (1.0 - c).max(0.0);
    }
    (acc / k as f32).sqrt()
}

fn main() {
    let (rows, k) = (768usize, 64usize);
    let u = ortho_factor(rows, k, 0x1234_5678_9abc_def1);

    println!("== A. ANCHOR vs burn-spectral lib.rs:188-191 (f32, [768,64], 3 NS iters)");
    let pf = polar_frobenius(&u, rows, k, 3);
    let ps = polar_sigma_max(&u, rows, k, 3, 5);
    println!(
        "  frobenius prescale : per-entry {:.4e}  sigma {:.4}     (doc: 6.4e-2 / 0.6992)",
        ortho_per_entry(&pf, rows, k),
        sigma_max(&pf, rows, k)
    );
    println!(
        "  sigma_max prescale : per-entry {:.4e}  sigma {:.6}   (doc: 3.0e-7 / 1.0000)",
        ortho_per_entry(&ps, rows, k),
        sigma_max(&ps, rows, k)
    );

    println!("\n== B. THE RETRACTION'S OWN FLOOR (3 iters), every shape `small` retracts");
    for (r, kk, n) in [
        (768usize, 64usize, "gate_up.u, down.v, out_proj.u/v, lm_head.u"),
        (2048, 64, "gate_up.v, down.u"),
        (256, 64, "lm_head.v"),
    ] {
        let f = ortho_factor(r, kk, 0xdead_beef_cafe_0001 ^ r as u64);
        let p = polar_sigma_max(&f, r, kk, 3, 5);
        let e = ortho_per_entry(&p, r, kk);
        println!("  [{r:5},{kk}] {n:44} {e:.3e}   margin to the 1e-3 latch {:.0}x", 1e-3 / e);
    }

    println!("\n== C. DRIFT LADDER [768,64]: with and without the retraction");
    let fp8 = 2f32.powf(-3.0) / 2.0; // e4m3: 3 mantissa bits -> half-step, relative
    println!(
        "  {:>9} {:>5} | {:>12} {:>12} | {:>11} | {:>9} {:>8}",
        "delta", "sys%", "ortho noR", "ortho 3it", "col angle", "latch", "vs fp8"
    );
    for steps in [1u32, 4, 10, 100, 1000] {
        let delta = 8e-4 * steps as f32; // Muon+ ColRow: ||dU||_F = lr*sqrt(k) = 8e-4
        for frac in [1.0f32, 0.0] {
            let d = drift(&u, rows, k, delta, frac, 0x5eed_1234_0000_0001 ^ steps as u64);
            let o_no = ortho_per_entry(&d, rows, k);
            let p = polar_sigma_max(&d, rows, k, 3, 5);
            let ca = col_angle(&d, &p, rows, k);
            println!(
                "  {:>9.1e} {:>4.0}% | {:>12.3e} {:>12.3e} | {:>11.3e} | {:>9} {:>7.2e}",
                delta,
                frac * 100.0,
                o_no,
                ortho_per_entry(&p, rows, k),
                ca,
                if o_no > 1e-3 { "CROSSED" } else { "-" },
                ca / fp8
            );
        }
    }
    println!("  (e4m3 half-step relative noise {fp8:.2e}; last column = col angle / that)");

    println!("\n== D. STEPS TO THE 1e-3 ONE-WAY LATCH, per optimizer group (routing.rs:95-98)");
    // Expert + Readout factors -> Muon+ ColRow: every row of the update is
    // unit-L2, so ||dU||_F = lr*sqrt(k). lm_head's factors -> AdamW ("Rest"),
    // per-coordinate step ~lr over an [in,k] factor.
    for (name, delta) in [("Muon+ ColRow, 15 factors", 8e-4f32), ("AdamW head, 1 factor", 2.2e-2)] {
        let o1 = ortho_per_entry(&drift(&u, rows, k, delta, 0.5, 7), rows, k);
        println!(
            "  {name:26} per-step ||dU||_F {delta:.1e} -> one step {o1:.2e}/entry | \
             random walk {:.0} steps | systematic {:.0} steps",
            (1e-3 / o1).powi(2),
            1e-3 / o1
        );
    }
    println!("  retracting every step : factor sits at the 2.3e-8 floor plus one step of drift");

    println!("\n== F. WHERE AGENTS.md 2.3's 'NS-3 floor ~4e-3 raw / 6e-5 per entry' COMES FROM");
    println!("   (it is quoted as the justification for the 1e-3 per-entry latch)");
    for spread in [1.0f32, 2.0, 3.0, 10.0] {
        // X = Q . diag(1, 1/s, 1/s, ...) : an on-the-column-space factor whose
        // SINGULAR VALUES are spread by `spread` (the polar retraction is exact
        // on a Stiefel point, so the spread has to be built in)
        let mut x = u.clone();
        for j in 1..k {
            for i in 0..rows {
                x[i * k + j] /= spread;
            }
        }
        let p = polar_sigma_max(&x, rows, k, 3, 5);
        println!(
            "  input spread {spread:5.1}:1 -> retracted per-entry {:.3e}  ({:.1}x the latch)",
            ortho_per_entry(&p, rows, k),
            ortho_per_entry(&p, rows, k) / 1e-3
        );
    }

    println!("\n== E. RETRACTION ARITHMETIC PER STEP (small: 16 factors, 3 shapes)");
    let mut flops = 0.0f64;
    for (n_fac, in_f) in [(8usize, 768usize), (6, 2048), (1, 256)] {
        flops += n_fac as f64 * 4.0 * 2.0 * (64.0 * 64.0 * in_f as f64);
    }
    let us = flops / 13.2e12 * 1e6;
    println!("  matmul FLOPs/step {flops:.3e} = {us:.1} us at cuBLAS fp32 13.2 TF/s");
    println!(
        "  measured retr 25.4 ms => the arithmetic is {:.2}% of it",
        us / 25400.0 * 100.0
    );
    println!("  ~880 launches (55/factor) => {:.0} us per launch", 25400.0 / 880.0);
}
