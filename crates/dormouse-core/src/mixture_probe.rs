//! The per-iteration expert mixture, captured on request.
//!
//! WHY IT EXISTS. The routing A/B (arXiv 2605.09165 §6.1, "Routing
//! predominantly diverges across loops") is decided by ONE number: does the
//! same token get the same experts on a different pass through the same
//! shared FFNs? Measuring it needs the per-iteration mixture weights, and
//! `forward_full_state` computes them and drops them. This is the seam that
//! reads them back.
//!
//! It is not throwaway instrumentation for one lane: the same seam is the
//! routing arm's REGRESSION GATE. The arm's claim is "routed selection
//! differs from the fixed mixture", and the only honest way to check that is
//! to measure both with the same instrument - a gate that used a different
//! one would be comparing two numbers from two programs.
//!
//! COST. `record` is one thread-local `Option` check per loop iteration and
//! nothing at all when disarmed: no device sync, no allocation, no counter on
//! the eval line. Disarmed is the default, and it is what every production
//! forward sees. [`take`] is where the host read happens, and only a test or
//! the diagnostic program calls it.
//!
//! WHAT IS CAPTURED. One `[b*t, n_experts]` tensor per EXECUTED iteration, in
//! iteration order - the weights the FFN branch actually multiplied each
//! expert's output by. Under the fixed mixture that is the controller's
//! softmax over the expert columns; under `moe_topk > 0` it is the
//! top-k-renormalised weights, whose SUPPORT is exactly the selected set, so
//! "the metric must CHANGE" is a statement about the mixture the model really
//! used, not about a re-derivation of it.

use std::cell::RefCell;

use burn::tensor::Tensor;

thread_local! {
    /// `Some` = armed. `vec![..]` = the capture in progress.
    static SINK: RefCell<Option<Vec<Tensor<2>>>> = const { RefCell::new(None) };
}

/// Arm the capture. The next `forward_full_state` on this thread records one
/// `[b*t, n_experts]` tensor per executed iteration.
///
/// Idempotent: arming twice discards the first capture rather than nesting.
pub fn arm() {
    SINK.with(|s| *s.borrow_mut() = Some(Vec::new()));
}

/// Take the capture and disarm. `None` if nothing was armed, which is also how
/// a caller sees that it forgot to arm - the same `Option` the sink holds, so
/// "nothing captured" and "never armed" are one honest value.
pub fn take() -> Option<Vec<Tensor<2>>> {
    SINK.with(|s| s.borrow_mut().take())
}

/// Is the capture armed? A test asserts this is false after an ordinary
/// forward, which is the zero-default property: nothing is recorded unless
/// somebody asked.
pub fn armed() -> bool {
    SINK.with(|s| s.borrow().is_some())
}

/// Record one iteration's mixture. Called at the branch, hot path.
#[inline]
pub(crate) fn record(mix: &Tensor<2>) {
    SINK.with(|s| {
        if let Some(v) = s.borrow_mut().as_mut() {
            v.push(mix.clone());
        }
    });
}

/// What one capture says about the expert mixture.
#[derive(Debug, Clone)]
pub struct MixtureStats {
    /// Mean mixture per iteration, one simplex point of length `n_experts`.
    pub mean: Vec<Vec<f64>>,
    /// Per iteration `i > 0`: mean over positions of the COSINE between
    /// iteration `i`'s weight vector and iteration 0's. `1.0` = the two
    /// passes weight the experts identically.
    pub cross_iter_cosine: Vec<f64>,
    /// Per iteration `i > 0`: fraction of positions whose single largest
    /// expert is the same as on iteration 0.
    pub cross_iter_top1_agree: Vec<f64>,
    /// Per iteration `i > 0`: fraction of positions whose SELECTED SET equals
    /// iteration 0's. This is the paper's "identical" count (arXiv
    /// 2605.09165v2 p.6 reports 4-14% identical, 25-53% disjoint).
    ///
    /// On the FIXED mixture the selected set of a token is the whole bank on
    /// both passes - every expert runs - so this is 1.0 and
    /// [`MixtureStats::cross_iter_disjoint`] is 0.0 BY ARITHMETIC. That is why
    /// these two are only comparable to the anchors on a ROUTED arm, and why
    /// [`MixtureStats::mean_support`] exists: it is the check that the dense
    /// endpoint really is "every expert selected" and has not quietly lost
    /// entries to softmax underflow.
    pub cross_iter_identical: Vec<f64>,
    /// Per iteration `i > 0`: fraction of positions whose selected set shares
    /// NO expert with iteration 0's. The other half of the anchor pair. At
    /// k=1 this is exactly `1 - cross_iter_identical` (a singleton set is
    /// either equal or disjoint); the two are only independent at k >= 2,
    /// which is why the anchors are quoted as a PAIR and why both are read.
    pub cross_iter_disjoint: Vec<f64>,
    /// Mean number of STRICTLY POSITIVE entries per row over the whole
    /// capture. `n_experts` on the fixed mixture (every expert runs),
    /// exactly `k` on a routed arm.
    ///
    /// This is the instrument's own honesty check. If it reads below
    /// `n_experts` on the dense path, a softmax underflowed to 0.0, the
    /// "support" stopped being the selection, and the identical/disjoint
    /// numbers above would be measuring the underflow rather than the routing.
    pub mean_support: f64,
    /// Share of total mixture mass per expert, over every position of every
    /// iteration. Uniform is `1/n_experts`.
    pub load: Vec<f64>,
    /// `exp(entropy(load))`: the effective number of experts. `1.0` = one
    /// expert takes everything, `n_experts` = a uniform mixture.
    pub effective_experts: f64,
}

/// Reduce a capture to [`MixtureStats`]. Host arithmetic over the capture's
/// own numbers; the only device read is the one `take` already made.
pub fn stats(cap: &[Tensor<2>]) -> MixtureStats {
    assert!(
        !cap.is_empty(),
        "mixture_probe::stats: empty capture (arm() before the forward?)"
    );
    let rows: Vec<Vec<Vec<f64>>> = cap
        .iter()
        .map(|t| {
            let v: Vec<f32> = t
                .clone()
                .into_data()
                .try_to_vec()
                .expect("mixture_probe::stats: reading the captured mixture back");
            let [b, e] = t.dims();
            assert_eq!(
                v.len(),
                b * e,
                "mixture_probe::stats: {} values for a [{}, {}] tensor",
                v.len(),
                b,
                e
            );
            v.chunks(e)
                .map(|r| r.iter().map(|x| *x as f64).collect())
                .collect()
        })
        .collect();
    let e = rows[0][0].len();
    let cos = |a: &[f64], b: &[f64]| {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    };
    let top1 = |r: &[f64]| {
        r.iter()
            .enumerate()
            .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            })
            .0
    };
    let mean: Vec<Vec<f64>> = rows
        .iter()
        .map(|it| {
            (0..e)
                .map(|j| it.iter().map(|r| r[j]).sum::<f64>() / it.len() as f64)
                .collect()
        })
        .collect();
    let cross_iter_cosine: Vec<f64> = (1..rows.len())
        .map(|i| {
            rows[i]
                .iter()
                .zip(&rows[0])
                .map(|(a, b)| cos(a, b))
                .sum::<f64>()
                / rows[i].len() as f64
        })
        .collect();
    let cross_iter_top1_agree: Vec<f64> = (1..rows.len())
        .map(|i| {
            rows[i]
                .iter()
                .zip(&rows[0])
                .filter(|(a, b)| top1(a) == top1(b))
                .count() as f64
                / rows[i].len() as f64
        })
        .collect();
    // The SELECTED SET is the row's support (its strictly positive entries).
    // On the fixed mixture that is the whole bank on every pass; on a routed
    // arm it is exactly the top-k. Both statistics below are over that set,
    // which is what the Sparse-Layers anchors are about - "which experts did
    // this token use", not "how did it weight all of them".
    let support = |r: &[f64]| -> Vec<usize> { (0..r.len()).filter(|&j| r[j] > 0.0).collect() };
    let cross_iter_identical: Vec<f64> = (1..rows.len())
        .map(|i| {
            rows[i]
                .iter()
                .zip(&rows[0])
                .filter(|(a, b)| support(a) == support(b))
                .count() as f64
                / rows[i].len() as f64
        })
        .collect();
    let cross_iter_disjoint: Vec<f64> = (1..rows.len())
        .map(|i| {
            rows[i]
                .iter()
                .zip(&rows[0])
                .filter(|(a, b)| !support(a).iter().any(|j| support(b).contains(j)))
                .count() as f64
                / rows[i].len() as f64
        })
        .collect();
    let mean_support: f64 = rows
        .iter()
        .flat_map(|it| it.iter())
        .map(|r| support(r).len() as f64)
        .sum::<f64>()
        / (rows.iter().map(|it| it.len()).sum::<usize>()) as f64;
    let total: f64 = rows
        .iter()
        .flat_map(|it| it.iter())
        .map(|r| r.iter().sum::<f64>())
        .sum();
    let load: Vec<f64> = (0..e)
        .map(|j| {
            rows.iter()
                .flat_map(|it| it.iter())
                .map(|r| r[j])
                .sum::<f64>()
                / total
        })
        .collect();
    let entropy: f64 = load
        .iter()
        .filter(|p| **p > 0.0)
        .map(|p| -(p * p.ln()))
        .sum();
    MixtureStats {
        mean,
        cross_iter_cosine,
        cross_iter_top1_agree,
        cross_iter_identical,
        cross_iter_disjoint,
        mean_support,
        load,
        effective_experts: entropy.exp(),
    }
}

/// Per iteration: the fraction of positions where two captures of the same
/// model on DIFFERENT inputs pick a different top-1 expert.
///
/// This is the input-dependence half of the question ("does any input change
/// the selection at all"). A number near 0 means the mixture is a function of
/// the iteration alone - a learned per-pass bias wearing a mixture's clothes.
pub fn top1_disagreement(a: &[Tensor<2>], b: &[Tensor<2>]) -> Vec<f64> {
    assert_eq!(
        a.len(),
        b.len(),
        "top1_disagreement: captures have different depths"
    );
    let top1 = |t: &Tensor<2>| -> Vec<usize> {
        let v: Vec<f32> = t
            .clone()
            .into_data()
            .try_to_vec()
            .expect("mixture_probe: reading a captured mixture back");
        let [_, e] = t.dims();
        v.chunks(e)
            .map(|r| {
                r.iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                        if x > bv {
                            (i, x)
                        } else {
                            (bi, bv)
                        }
                    })
                    .0
            })
            .collect()
    };
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let (tx, ty) = (top1(x), top1(y));
            assert_eq!(
                tx.len(),
                ty.len(),
                "top1_disagreement: captures have different batch shapes"
            );
            tx.iter().zip(&ty).filter(|(p, q)| p != q).count() as f64 / tx.len() as f64
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData};

    fn dev() -> Device {
        Device::flex()
    }

    fn t(rows: Vec<Vec<f64>>) -> Tensor<2> {
        let e = rows[0].len();
        let flat: Vec<f32> = rows.iter().flatten().map(|x| *x as f32).collect();
        Tensor::<2>::from_data(TensorData::new(flat, [rows.len(), e]), &dev())
    }

    /// The three numbers the diagnostic is read for, on a fixture where the
    /// answer is known by construction: two iterations, three experts, each
    /// row a one-hot.
    ///
    /// The case that separates the two weight statistics, and it is worth being
    /// explicit because the obvious expectation is WRONG: a PERMUTATION of the
    /// weights has per-position cosine **0**, not 1. `[1,0,0]` and `[0,1,0]` are
    /// orthogonal. An identical copy has cosine 1. Both agree nowhere / agree
    /// everywhere on the top-1. An earlier version of this test asserted
    /// cosine 1 for the permutation and was red - which is the reason the
    /// comment is here rather than the assertion alone.
    #[test]
    fn stats_separates_a_permutation_from_a_copy() {
        let a = t(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let permuted = t(vec![
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0],
        ]);
        let copy = t(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let perm = stats(&[a.clone(), permuted]);
        assert!(
            perm.cross_iter_cosine[0].abs() < 1e-9,
            "a permutation of the weights is ORTHOGONAL (cosine 0), got {}",
            perm.cross_iter_cosine[0]
        );
        assert_eq!(
            perm.cross_iter_top1_agree[0], 0.0,
            "a permutation agrees on no position"
        );
        let same = stats(&[a, copy]);
        assert!(
            (same.cross_iter_cosine[0] - 1.0).abs() < 1e-9,
            "an identical iteration has cosine 1, got {}",
            same.cross_iter_cosine[0]
        );
        assert_eq!(
            same.cross_iter_top1_agree[0], 1.0,
            "an identical iteration agrees everywhere"
        );
        for s in [&perm, &same] {
            assert!(
                (s.effective_experts - 3.0).abs() < 1e-9,
                "uniform load over 3 experts is 3, got {}",
                s.effective_experts
            );
            for (l, p) in s.load.iter().zip([1.0 / 3.0; 3]) {
                assert!((l - p).abs() < 1e-9, "load must be uniform: {l} vs {p}");
            }
        }
    }

    /// `effective_experts` is the load-concentration read: one expert taking
    /// everything is 1.0, not 0. A metric that runs the wrong way is worse
    /// than no metric, so the endpoints are pinned here.
    #[test]
    fn effective_experts_runs_from_one_to_n() {
        let collapsed = t(vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]]);
        assert!((stats(&[collapsed]).effective_experts - 1.0).abs() < 1e-9);
        let half = t(vec![vec![0.5, 0.5, 0.0], vec![0.5, 0.5, 0.0]]);
        assert!((stats(&[half]).effective_experts - 2.0).abs() < 1e-9);
    }

    /// Input dependence: two captures that differ on every position report
    /// disagreement 1.0 per iteration, and identical captures report 0.0.
    #[test]
    fn top1_disagreement_is_the_input_dependence_read() {
        let a = t(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
        let b = t(vec![vec![0.0, 0.0, 1.0], vec![0.0, 1.0, 0.0]]);
        let d = top1_disagreement(std::slice::from_ref(&a), std::slice::from_ref(&b));
        assert_eq!(d.len(), 1);
        assert!(
            (d[0] - 0.5).abs() < 1e-9,
            "one of two positions flipped: {}",
            d[0]
        );
        assert_eq!(
            top1_disagreement(std::slice::from_ref(&a), std::slice::from_ref(&a))[0],
            0.0
        );
    }

    /// ZERO DEFAULT. An ordinary forward records nothing, and the sink stays
    /// disarmed - the capture is opt-in, not a cost every step pays.
    #[test]
    fn disarmed_is_the_default() {
        assert!(!armed(), "a fresh thread must be disarmed");
        assert!(
            take().is_none(),
            "take() on a disarmed sink is None, not an empty capture"
        );
        let _ = t(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        record(&t(vec![vec![1.0, 0.0], vec![0.0, 1.0]]));
        assert!(take().is_none(), "record() while disarmed must be a no-op");
        arm();
        assert!(armed());
        record(&t(vec![vec![1.0, 0.0], vec![0.0, 1.0]]));
        record(&t(vec![vec![0.5, 0.5], vec![0.5, 0.5]]));
        let got = take().expect("armed capture");
        assert_eq!(got.len(), 2, "one tensor per recorded iteration");
        assert!(!armed(), "take() disarms");
    }

    /// The ANCHOR PAIR - `cross_iter_identical` and `cross_iter_disjoint` -
    /// read as the Sparse-Layers paper reports them (4-14% identical, 25-53%
    /// disjoint). Both endpoints are pinned on fixtures where the answer is
    /// known, because at k=1 a singleton set is EITHER equal or disjoint, and
    /// an instrument that cannot report 0% identical is not measuring the
    /// anchors.
    ///
    /// This is also the falsification for the dense endpoint: a routed arm
    /// whose support is the whole bank would read identical 1.0 / disjoint 0.0,
    /// which is indistinguishable from the dense path - so `mean_support` is
    /// what separates them, and both are asserted here.
    #[test]
    fn anchor_pair_reads_zero_and_one_at_both_ends() {
        // Every expert selected on both passes: the FIXED mixture's shape.
        // identical 1.0, disjoint 0.0, support = n_experts.
        let dense = t(vec![
            vec![0.4, 0.3, 0.2, 0.1],
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.25, 0.25, 0.25, 0.25],
        ]);
        let s = stats(&[dense.clone(), dense.clone()]);
        assert_eq!(
            s.cross_iter_identical[0], 1.0,
            "an all-experts set is identical"
        );
        assert_eq!(s.cross_iter_disjoint[0], 0.0, "and never disjoint");
        assert!(
            (s.mean_support - 4.0).abs() < 1e-9,
            "dense support is every expert, got {}",
            s.mean_support
        );

        // A ROUTED k=1 capture: one live expert per row, and the winner moves.
        let routed = t(vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ]);
        let moved = t(vec![
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
        ]);
        let s = stats(&[routed.clone(), moved]);
        assert!(
            (s.mean_support - 1.0).abs() < 1e-9,
            "a top-1 capture selects exactly one expert, got {}",
            s.mean_support
        );
        // Row 1 picks the same expert on both passes; rows 0 and 2 do not.
        assert!(
            (s.cross_iter_identical[0] - 1.0 / 3.0).abs() < 1e-9,
            "one of three rows agreed: got {}",
            s.cross_iter_identical[0]
        );
        assert!(
            (s.cross_iter_disjoint[0] - 2.0 / 3.0).abs() < 1e-9,
            "and the other two are disjoint: got {}",
            s.cross_iter_disjoint[0]
        );
        // At k=1 the two are exact complements - the relation the anchors'
        // 25-53% / 4-14% pair does NOT satisfy, which is the reason the
        // anchors are a k=2 statement and top-2 is a separate configuration.
        assert!(
            (s.cross_iter_identical[0] + s.cross_iter_disjoint[0] - 1.0).abs() < 1e-9,
            "at k=1 identical and disjoint must partition the positions"
        );
        // And an UNCHANGED routed capture is fully identical.
        let s = stats(&[routed.clone(), routed]);
        assert_eq!(s.cross_iter_identical[0], 1.0);
        assert_eq!(s.cross_iter_disjoint[0], 0.0);
    }

    /// An empty capture is a caller error, not a statistic: `stats` on it
    /// would divide by zero rows and report a confident 0.
    #[test]
    #[should_panic(expected = "empty capture")]
    fn stats_refuses_an_empty_capture() {
        stats(&[]);
    }
}
