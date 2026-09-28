//! MoR - Mixture-of-Recursions (arXiv 2507.10524) routing for the loop.
//!
//! The routing MECHANISM lives in `burn-mor` (vendored): the shared linear
//! router ([`MoRRouter`]) and the top-k-index primitive
//! ([`burn_mor::topk_indices`]). This module is the WIRING: how the loop's
//! iteration slots become a per-position top-k, and the four structural
//! ingredients that make MoR stable where PonderNet's probabilistic halting
//! collapsed (ADR-0013).
//!
//! 1. Fixed-capacity rank-based top-k ([`route`]): the top-`k` slots always
//!    fill. There is no learned threshold to drift past, so the active set
//!    cannot empty - the failure mode of `lambda -> 0`.
//! 2. BCE auxiliary against a per-batch-RECOMPUTED target ([`route`]): the
//!    label IS the top-k membership re-derived from the scores of THIS
//!    batch, so it cannot go stale or be detached from the scores. It drives
//!    the scores bimodal (membership becomes near-binary) and, unlike a
//!    learned weighting, its total mass is the constant `k`.
//! 3. A floor of one recursion: `k >= 1` is asserted, so every position
//!    passes at least one recursion.
//! 4. Unweighted LM loss: the caller masks the readout and the CE with the
//!    same 0/1 mask and divides by the fixed `k` - no `p_n`, nothing
//!    continuous that can decay the loss toward zero.
use burn::tensor::activation::softplus;
use burn::tensor::{Int, Tensor};

pub use burn_mor::MoRRouter;

/// Effective `k`: never below 1 (ingredient 3), never above the number of
/// slots that actually ran. The clamp to `n` is the depth-override path
/// (`--eval-depths` runs a truncated loop, a measurement, not a config);
/// `mor_k > max_iter` is a config error and `config::validate` refuses it
/// loudly instead. The caller divides the masked readout and CE by this, so
/// it is a host-side constant — reading the count off the mask instead would
/// sync the device every step.
pub fn eff_k(k: usize, n: usize) -> usize {
    assert!(
        k >= 1,
        "MoR top-k k must be >= 1 (the floor of one recursion), got {k}"
    );
    k.min(n.max(1))
}

/// Per-position top-k over the `n` iteration slots: `(mask, aux)`.
///
/// `scores` is `[b, t, n]` - one router score per (position, slot). The mask
/// is `[b, t, n]` of exact 0/1 with exactly `k` ones per position, and `aux`
/// is the unweighted BCE scalar whose label is that same mask.
pub fn route(scores: Tensor<3>, k: usize) -> (Tensor<3>, Tensor<1>) {
    let [b, t, n] = scores.dims();
    let k = eff_k(k, n);
    let dev = scores.device();
    // THE shared primitive: burn-mor's top-k-indices over the candidate axis.
    // This is the crate's committed one - `argsort_descending` + narrow, which
    // exists precisely because cubecl's argtopk reduce emitted garbage indices
    // (`topk.rs:9`, our sm_120 ILLEGAL_ADDRESS). The newer argtopk-based
    // `topk_indices_last` (ADR-0015, the repaired cubek-reduce) is the
    // intended replacement and is a one-line swap here; at wiring time it did
    // not compile (`.int()` on an already-Int `argtopk` result).
    let idx = burn_mor::topk_indices(scores.clone().reshape([b * t, n]), k, 1);
    // Binary membership from those indices with NO gather, NO scatter, NO
    // host round trip and NO bool->float cast (that cast is broken on cubecl,
    // AGENTS.md 2026-09-27): one-hot the picks with `mask_fill` on a FLOAT
    // tensor - the sanctioned idiom - and sum the k rows.
    //
    // Both Int tensors here come from the same device, so they share its Int
    // dtype: `topk_indices` must NOT hand back a hard-coded one. Sum the k
    // axis (dim 1) - `sum_dim` is keepdim in burn 0.22, so the result is
    // `[b*t, 1, n]` and the reshape below is exact.
    let ar = Tensor::<1, Int>::arange(0..n as i64, &dev).reshape([1, 1, n]);
    let eq = idx.reshape([b * t, k, 1]).equal(ar); // [b*t, k, n]
    let mask = Tensor::<3>::zeros([b * t, k, n], &dev)
        .mask_fill(eq, 1.0)
        .sum_dim(1)
        .reshape([b, t, n]);
    // Ingredient 2: the label is a FRESH top-k of THESE scores (computed four
    // lines above, from this batch, inside this call) - not a stored target.
    // BCEWithLogits in the stable form: burn's softplus falls back to the
    // identity above 20, so the large |score| the BCE itself drives cannot
    // overflow into inf/NaN.
    let lab = mask.clone().reshape([b * t, n]).detach();
    let s = scores.reshape([b * t, n]);
    let hit = lab.clone().mul(softplus(s.clone().neg(), 1.0));
    let miss = lab.neg().add_scalar(1.0).mul(softplus(s, 1.0));
    (mask, hit.add(miss).mean().reshape([1]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, TensorData};

    // Device::default() is the CPU backend (burn-flex) without --features cuda
    // and Cuda with it, so these run the routing math on the real backend in
    // both gates. The cubecl-only defects (garbage argtopk indices, broken
    // casts) are invisible to a CPU-only run (AGENTS.md, 2026-09-27).
    fn dev() -> burn::tensor::Device {
        burn::tensor::Device::default()
    }

    /// Ingredient 1 + 3: every position gets EXACTLY k selected slots, and
    /// the floor holds - k=1 still selects one, k=0 is refused loudly.
    #[test]
    fn selects_exactly_k_per_position_with_a_floor_of_one() {
        let dev = dev();
        let (b, t, n) = (2usize, 5usize, 4usize);
        let scores = Tensor::<3>::random([b, t, n], Distribution::Default, &dev);
        for k in 1..=n {
            let (mask, _) = route(scores.clone(), k);
            assert_eq!(mask.dims(), [b, t, n]);
            // Per position, the count is k - the set always fills.
            let counts: Vec<f32> = mask
                .clone()
                .sum_dim(2)
                .into_data()
                .try_to_vec()
                .expect("counts readable");
            assert_eq!(counts.len(), b * t);
            assert!(
                counts.iter().all(|c| (*c - k as f32).abs() < 1e-6),
                "k={k}: every position must select exactly {k}, got {counts:?}"
            );
            // And the mask is binary, not a soft weight (ingredient 2's
            // bimodality target is reached by the BCE, not by the mask).
            let m: Vec<f32> = mask.into_data().try_to_vec().expect("mask readable");
            assert!(
                m.iter().all(|v| *v == 0.0 || *v == 1.0),
                "mask must be 0/1, got {m:?}"
            );
        }
        // The floor: k=1 is one pass, not zero, and k=0 is a loud refusal.
        let (mask, _) = route(scores.clone(), 1);
        let counts: Vec<f32> = mask.sum_dim(2).into_data().try_to_vec().unwrap();
        assert!(counts.iter().all(|c| (*c - 1.0).abs() < 1e-6));
        assert!(
            std::panic::catch_unwind(|| {
                let _ = route(scores.clone(), 0);
            })
            .is_err(),
            "k=0 must panic, not silently drop the position"
        );
    }

    /// Ingredient 2: the BCE label is a top-k of the CURRENT scores. Checked
    /// against an INDEPENDENT host-side argsort of the same rows, and the
    /// loss is checked against the closed-form BCE for exactly that label -
    /// a stale or re-used label would disagree with both.
    #[test]
    fn bce_label_is_a_fresh_topk_of_the_current_scores() {
        let dev = dev();
        // Deliberately distinct values per row: no ties, so the top-k set is
        // unambiguous on every backend.
        let rows: [[f32; 4]; 2] = [[5.0, 1.0, 4.0, 2.0], [0.0, 9.0, 1.0, 8.0]];
        let flat = |r: &[[f32; 4]; 2]| r.iter().flatten().copied().collect::<Vec<f32>>();
        let scores = Tensor::<3>::from_data(TensorData::new(flat(&rows), [1, 2, 4]), &dev);
        let host_label = |r: [f32; 4], k: usize| -> Vec<f32> {
            let mut order: Vec<usize> = (0..4).collect();
            order.sort_by(|a, b| r[*b].partial_cmp(&r[*a]).unwrap());
            let mut lab = vec![0.0; 4];
            for i in order.iter().take(k) {
                lab[*i] = 1.0;
            }
            lab
        };
        for k in 1..=4 {
            let (mask, bce) = route(scores.clone(), k);
            let got: Vec<f32> = mask.into_data().try_to_vec().expect("mask readable");
            for (r, row) in rows.iter().enumerate() {
                assert_eq!(
                    &got[r * 4..r * 4 + 4],
                    &host_label(*row, k),
                    "k={k} row {r}: label is not a fresh top-k of the current scores"
                );
            }
            // The loss the BCE used must be the BCE of THAT label: compare
            // against the closed form on the host.
            let mut host = 0.0f64;
            for r in rows.iter() {
                let lab = host_label(*r, k);
                for (i, &s) in r.iter().enumerate() {
                    let y = lab[i] as f64;
                    let sp = |x: f64| (1.0 + x.exp()).ln();
                    host += y * sp(-(s as f64)) + (1.0 - y) * sp(s as f64);
                }
            }
            host /= (rows.len() * 4) as f64;
            let got_loss: f32 = bce.into_scalar();
            assert!(
                (got_loss as f64 - host).abs() < 1e-4,
                "k={k}: BCE {got_loss} != closed form for the returned label {host}"
            );
        }
        // Non-staleness, stated as a test: move one row's ranking and the
        // label MUST move with it. A cached or stale target would not.
        let bumped: [[f32; 4]; 2] = [[5.0, 1.0, 4.0, 2.0], [0.0, 9.0, 1.0, 30.0]];
        let (mask2, _) = route(
            Tensor::<3>::from_data(TensorData::new(flat(&bumped), [1, 2, 4]), &dev),
            2,
        );
        let got2: Vec<f32> = mask2.into_data().try_to_vec().unwrap();
        assert_eq!(
            &got2[4..8],
            &host_label(bumped[1], 2),
            "a new batch must re-derive the label"
        );
        assert_ne!(&got2[0..4], &got2[4..8], "distinct rows must not share a label");
    }
}
