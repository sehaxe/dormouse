// The batched chunk path against the loop path, arm by arm, on identical
// inputs. Written BEFORE the rewrite, so the bar below was fixed while the
// batched arm was still the loop — i.e. it was not fitted to a number the
// rewrite produced.
//
// ## Why this file exists
//
// The chunked-WY forward is a numerical rewrite candidate: the loop arm spends
// ~68% of its per-chunk tensor ops inverting a unit lower-triangular 16x16
// matrix ONE ROW AT A TIME, 15 times per chunk, 32 chunks per forward. The
// batched arm does all chunks at once and inverts with the finite Neumann
// series `(I+L)^-1 = I - L + L^2 - ... + (-L)^(c-1)`, which is an exact identity
// (not a truncation) because a strictly lower triangular c x c matrix has
// `L^c = 0`.
//
// A rewrite like that is either right or quietly wrong, and "quietly" is the
// dangerous direction: the wrong answer is still a plausible-looking attention
// output, so only a differential comparison against the untouched loop can tell
// them apart. The loop arm is therefore the reference and stays unmodified.
//
// ## What is asserted, and why this bar
//
// `REL_BAR = 1e-4`, on the deviation normalised by the tensor's own max |value|
// (a scale-relative measure, so a case whose output happens to be small is not
// held to an absolute bar it does not owe).
//
// The bar is derived, not fitted. Every quantity both arms compute is either
// bitwise identical (elementwise ops on the same inputs; the masks, the
// cumsum, the exp) or a matmul of identical inputs in a different kernel
// (`[B,H,c,K] @ [B,H,K,c]` per chunk vs `[B,H,n,c,K] @ [B,H,n,K,c]` for all of
// them), which differs only in the gemm's accumulation order. The one genuine
// difference is `M^-1`: a finite sum of c terms (Neumann) against a row-by-row
// triangular solve. Its error is `O(eps * sum_j |(-L)^j| / |M^-1|)`, i.e. f32
// round-off amplified by the conditioning of `M = I + L`, over a sum of at most
// 16 terms. eps_f32 = 1.2e-7, so even a condition number of ~30 lands at ~1e-5.
//
// 1e-4 sits in the gap between that (1e-6..1e-5, case dependent) and a real
// bug: a transposed operand, an off-by-one chunk, a wrong mask, a stale state
// or a transposed m_inv are all O(1) relative errors, four orders of magnitude
// above the bar. The `||L||_inf` diagnostic printed per case is what says
// whether a case sat near the top of the numerical range, so a reading close to
// the bar is explicable rather than mysterious.
//
// ## The cases
//
// chunk 16 (the trainer), 8, 4 and a ragged tail (T not a multiple of
// chunk_size, which the batched arm handles by zero-padding the last chunk —
// see the padding note in forward.rs); K/V at the trainer's preset (64) and
// small (8); batch 1 and 3; zero and non-zero incoming state; and a
// correlated-key case, which is the adversarial one for the Neumann series
// because it drives `akk` towards its worst case (`<k_i,k_j> -> 1` for unit
// keys, so `||L||_inf` approaches c-1) while staying inside the delta rule's
// stability requirement.

#![cfg(feature = "std")]
#![allow(deprecated)]

use burn::tensor::{Device, Distribution, Tensor};
use burn_gdn2::forward::{chunk_wy_forward_batched, chunk_wy_forward_loop};

/// Scale-relative deviation bar. Derived in the module comment, not fitted.
const REL_BAR: f32 = 1e-4;

struct Case {
    name: &'static str,
    batch: usize,
    heads: usize,
    time: usize,
    k_dim: usize,
    v_dim: usize,
    chunk: usize,
    /// `g` per step: `(lo, hi)` in log-decay. K3's floor is -5, 0 is no decay.
    g_range: (f32, f32),
    /// Noise added to one shared key direction per head. Large (1e3) = plain
    /// i.i.d. random keys; small = near-collinear, which is the regime that
    /// drives `akk` (and so the Neumann sum's cancellation) to its worst case.
    noise: f32,
    /// Upper end of the erase gate `b`; 1.0 is the delta rule's limit.
    b_hi: f32,
    state: bool,
}

fn make(
    c: &Case,
    dev: &Device,
) -> (
    Tensor<4>,
    Tensor<4>,
    Tensor<4>,
    Tensor<4>,
    Tensor<4>,
    Tensor<4>,
    Tensor<4>,
) {
    dev.seed(7);
    let [nb, nh, nt, nk] = [c.batch, c.heads, c.time, c.k_dim];
    let q = Tensor::<4>::random([nb, nh, nt, nk], Distribution::Normal(0.0, 1.0), dev);
    // Keys are L2-normalized per (position, head), as the module does: the
    // delta-rule operator (I - b k k^T) is only stable for unit keys.
    let kraw = if c.noise < 1.0 {
        // one shared direction per head + a little noise => <k_i,k_j> ~ 1
        let base = Tensor::<4>::random([nb, nh, 1, nk], Distribution::Normal(0.0, 1.0), dev);
        let noise = Tensor::<4>::random([nb, nh, nt, nk], Distribution::Normal(0.0, c.noise as f64), dev);
        base.repeat_dim(2, nt) + noise
    } else {
        Tensor::<4>::random([nb, nh, nt, nk], Distribution::Normal(0.0, 1.0), dev)
    };
    let k = kraw.clone() / kraw.powf_scalar(2.0).sum_dim(3).sqrt();
    let v = Tensor::<4>::random([nb, nh, nt, c.v_dim], Distribution::Normal(0.0, 1.0), dev);
    // b is the erase gate: a sigmoid output, so [0,1].
    let b = Tensor::<4>::random([nb, nh, nt, nk], Distribution::Uniform(0.0, c.b_hi as f64), dev);
    // w_gate is the write gate; the KDA path feeds 1, the GDN-2 path a sigmoid.
    let w = Tensor::<4>::random(
        [nb, nh, nt, c.v_dim],
        Distribution::Uniform(0.5, 1.0),
        dev,
    );
    let g = Tensor::<4>::random(
        [nb, nh, nt, nk],
        Distribution::Uniform(c.g_range.0 as f64, c.g_range.1 as f64),
        dev,
    );
    let s = if c.state {
        Tensor::<4>::random([nb, nh, nk, c.v_dim], Distribution::Normal(0.0, 0.5), dev)
    } else {
        Tensor::<4>::zeros([nb, nh, nk, c.v_dim], dev)
    };
    (q, k, v, g, b, w, s)
}

fn host(t: &Tensor<4>) -> Vec<f32> {
    t.clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// max |a-b| / max |a| over both tensors.
fn rel_dev(a: &Tensor<4>, b: &Tensor<4>) -> f32 {
    let (ha, hb) = (host(a), host(b));
    assert_eq!(ha.len(), hb.len(), "shape mismatch");
    let scale = ha.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let diff = ha
        .iter()
        .zip(hb.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    if scale == 0.0 {
        diff
    } else {
        diff / scale
    }
}

/// The max row sum of `akk` in the FIRST chunk — the quantity that decides how
/// much cancellation the Neumann series has to survive. Computed here from the
/// same formula the arms use, on the host, so it is a diagnostic and not a
/// second implementation of the thing under test.
fn akk_row_sum(c: &Case, g: &Tensor<4>, k: &Tensor<4>, b: &Tensor<4>) -> f32 {
    let [bs, hh, tt, kd] = [c.batch, c.heads, c.time, c.k_dim];
    let take = |t: &Tensor<4>| -> Vec<f32> {
        host(t).chunks(kd).take(bs * hh * tt).flatten().copied().collect()
    };
    let (gk, kk, bk) = (take(g), take(k), take(b));
    let mut worst = 0.0f32;
    for bi in 0..bs {
        for hi in 0..hh {
            for i in 0..c.chunk.min(tt) {
                // E over the chunk, per channel
                let mut e = vec![0.0f32; kd];
                for d in 0..kd {
                    let mut s = 0.0;
                    for t0 in 0..=i {
                        s += gk[((bi * hh + hi) * tt + t0) * kd + d];
                    }
                    e[d] = s.exp();
                }
                let mut rowsum = 0.0f32;
                for j in 0..i {
                    let mut s = 0.0f64;
                    for d in 0..kd {
                        let ek = {
                            let mut acc = 0.0;
                            for t0 in 0..=j {
                                acc += gk[((bi * hh + hi) * tt + t0) * kd + d];
                            }
                            acc.exp()
                        };
                        s += (bk[((bi * hh + hi) * tt + i) * kd + d] * e[d]
                            * kk[((bi * hh + hi) * tt + j) * kd + d]
                            / ek) as f64;
                    }
                    rowsum += s as f32;
                }
                worst = worst.max(rowsum);
            }
        }
    }
    worst
}

fn cases() -> Vec<Case> {
    vec![
        // the trainer's own shape of thing, at a length that divides evenly
        Case {
            name: "trainer preset, chunk 16, T=512",
            batch: 2,
            heads: 2,
            time: 512,
            k_dim: 64,
            v_dim: 64,
            chunk: 16,
            g_range: (-5.0, -0.01),
            noise: 1e3,
            b_hi: 1.0,
            state: true,
        },
        // the same at a ragged tail: 6 full chunks + one of 4
        Case {
            name: "chunk 16, ragged tail (T=100)",
            batch: 1,
            heads: 2,
            time: 100,
            k_dim: 64,
            v_dim: 64,
            chunk: 16,
            g_range: (-5.0, -0.01),
            noise: 1e3,
            b_hi: 1.0,
            state: true,
        },
        // one chunk shorter than the whole sequence
        Case {
            name: "chunk 16, T=9 (one ragged chunk only)",
            batch: 1,
            heads: 1,
            time: 9,
            k_dim: 8,
            v_dim: 8,
            chunk: 16,
            g_range: (-5.0, -0.01),
            noise: 1e3,
            b_hi: 1.0,
            state: false,
        },
        // a shorter chunk: 2c of it, so the Neumann series is 8 terms
        Case {
            name: "chunk 8, T=64",
            batch: 3,
            heads: 2,
            time: 64,
            k_dim: 16,
            v_dim: 8,
            chunk: 8,
            g_range: (-5.0, -0.01),
            noise: 1e3,
            b_hi: 1.0,
            state: true,
        },
        // chunk 4: the series is 4 terms
        Case {
            name: "chunk 4, T=64",
            batch: 1,
            heads: 3,
            time: 64,
            k_dim: 8,
            v_dim: 8,
            chunk: 4,
            g_range: (-5.0, -0.01),
            noise: 1e3,
            b_hi: 1.0,
            state: false,
        },
        // mild decay, the regime tests/test_chunk.rs uses for its 1e-4 bar
        Case {
            name: "chunk 16, mild decay, no incoming state",
            batch: 2,
            heads: 1,
            time: 128,
            k_dim: 32,
            v_dim: 32,
            chunk: 16,
            g_range: (-0.15, -0.01),
            noise: 1e3,
            b_hi: 1.0,
            state: false,
        },
        // NEARLY ADVERSARIAL: collinear unit keys with the full erase gate, but
        // still decaying, so the off-diagonals of akk survive a few steps.
        Case {
            name: "chunk 16, collinear keys, decaying",
            batch: 2,
            heads: 2,
            time: 128,
            k_dim: 64,
            v_dim: 64,
            chunk: 16,
            g_range: (-5.0, -0.01),
            noise: 0.01,
            b_hi: 1.0,
            state: true,
        },
        // ADVERSARIAL, and legal: no decay (alpha = 1 is the K3 sigmoid's limit
        // at z -> -inf) with collinear unit keys and b at its limit. Then
        // akk[i][j] -> b_i for every j < i, so ||L||_inf -> c-1 = 15, which is
        // the regime where a Neumann sum of 15 powers cancels hardest. This is
        // the case that decides whether the finite series is usable at all.
        Case {
            name: "chunk 16, no decay + collinear keys (worst case)",
            batch: 2,
            heads: 2,
            time: 128,
            k_dim: 64,
            v_dim: 64,
            chunk: 16,
            g_range: (-0.01, -0.0001),
            noise: 0.01,
            b_hi: 1.0,
            state: true,
        },
        // past TILE: the batched arm cannot express this and the loop must
        // still be the thing that runs. Asserted, not assumed.
        Case {
            name: "chunk 32 (past TILE — loop arm only)",
            batch: 1,
            heads: 1,
            time: 128,
            k_dim: 16,
            v_dim: 16,
            chunk: 32,
            g_range: (-5.0, -0.01),
            noise: 1e3,
            b_hi: 1.0,
            state: true,
        },
    ]
}

#[test]
fn the_batched_chunk_path_agrees_with_the_loop() {
    let dev = Device::ndarray();
    let scale = 64f64.powf(-0.5);
    let mut worst_overall = 0.0f32;
    for c in cases() {
        let (q, k, v, g, b, w, s) = make(&c, &dev);
        let args = (q.clone(), k.clone(), v.clone(), g.clone(), b.clone(), w.clone(), s.clone());
        let (lo, ls, sc_loop) = chunk_wy_forward_loop(
            args.0.clone(),
            args.1.clone(),
            args.2.clone(),
            args.3.clone(),
            args.4.clone(),
            args.5.clone(),
            args.6.clone(),
            scale,
            c.chunk,
            None,
        );
        burn_gdn2::alloc_trace::reset_batched_declined();
        if c.chunk > burn_gdn2::forward::TILE {
            // past TILE: the dispatcher must route to the loop. Check the
            // dispatcher's own answer, not just the helper.
            burn_gdn2::set_chunk_path(burn_gdn2::ChunkPath::Batched);
            let (bo, bs) = burn_gdn2::chunk_wy_forward(
                args.0.clone(),
                args.1.clone(),
                args.2.clone(),
                args.3.clone(),
                args.4.clone(),
                args.5.clone(),
                args.6.clone(),
                scale,
                c.chunk,
            );
            let (d, why) = burn_gdn2::alloc_trace::batched_declined();
            assert_eq!(d, 1, "{}: the batched arm declined without being counted", c.name);
            println!(
                "{:<46} past TILE: routed to the loop, counted ({why}); dev out {:e} state {:e}",
                c.name,
                rel_dev(&lo, &bo),
                rel_dev(&ls, &bs)
            );
            continue;
        }
        let (bo, bs, sc) =
            chunk_wy_forward_batched(args.0, args.1, args.2, args.3, args.4, args.5, args.6, scale, c.chunk);

        let d_out = rel_dev(&lo, &bo);
        let d_state = rel_dev(&ls, &bs);
        let l_norm = akk_row_sum(&c, &g, &k, &b);
        // the scratch is the custom node's backward input: it must carry the
        // same four values per chunk, or that backward differentiates a
        // different function than the one whose gradient was checked.
        assert_eq!(
            sc.chunks.len(),
            c.time.div_ceil(c.chunk),
            "{}: scratch chunk count",
            c.name
        );
        assert_eq!(sc_loop.chunks.len(), sc.chunks.len(), "{}: reference scratch", c.name);
        let mut d_scratch = 0.0f32;
        for (a, b) in sc_loop.chunks.iter().zip(sc.chunks.iter()) {
            d_scratch = d_scratch
                .max(rel_dev(&a.g_exp, &b.g_exp))
                .max(rel_dev(&a.q_gated, &b.q_gated))
                .max(rel_dev(&a.aqk, &b.aqk))
                .max(rel_dev(&a.m_inv, &b.m_inv));
        }
        worst_overall = worst_overall.max(d_out).max(d_state).max(d_scratch);
        println!(
            "{:<46} ||L||_inf {l_norm:5.2}  out {d_out:.2e}  state {d_state:.2e}  scratch {d_scratch:.2e}",
            c.name
        );
        assert!(
            d_scratch < REL_BAR,
            "{}: chunk scratch deviates {d_scratch:.3e} (bar {REL_BAR:.1e}) — the custom-node \
             backward would differentiate a different function than the forward produced",
            c.name
        );
        assert!(
            d_out < REL_BAR,
            "{}: output deviates {d_out:.3e} (bar {REL_BAR:.1e}, ||L||_inf {l_norm:.2})",
            c.name
        );
        assert!(
            d_state < REL_BAR,
            "{}: final state deviates {d_state:.3e} (bar {REL_BAR:.1e}, ||L||_inf {l_norm:.2})",
            c.name
        );
    }
    println!("worst scale-relative deviation over all cases: {worst_overall:.3e}");
}
