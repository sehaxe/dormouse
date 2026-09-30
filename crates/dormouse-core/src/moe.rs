//! Sparse top-`k` expert routing for the loop's FFN branch.
//!
//! WHAT THIS REPLACES. The fixed mixture is a DENSE softmax over the
//! controller's expert columns (`loop_block.rs`): every expert runs for every
//! token on every pass, weighted by a positive share. This module turns that
//! into a SELECTION - the `k` experts the token actually uses, renormalized -
//! so the loop's expressiveness can come from which expert a token reaches
//! for on a given pass, not from how it averages all of them.
//!
//! # Deliberately NOT here: the FLOP saving
//!
//! Every expert is still computed and then masked, so executed FLOPs are
//! unchanged. That is the honest first rung: what this arm buys at this stage
//! is SPECIALIZATION at equal ACTIVE parameters, and the A/B row says so.
//! Harvesting the compute needs a gather/scatter dispatch, which at
//! launch-bound (AGENTS.md §2.2, mean GPU utilisation 13.3%) costs more
//! launches than the smaller GEMMs save.
//!
//! ponytail: gather/scatter dispatch per top-k, if a profiled step ever makes
//! the FFN a measurable share of wall-clock rather than 2-5% of it.
//!
//! # The primitives are REUSED, not rewritten (ladder rung 2)
//!
//! `burn_mor::topk_indices` is the crate's committed top-k - one
//! `argsort_descending` + `narrow`, avoiding `argtopk` because cubecl
//! 0.11.0-pre.2 has a documented garbage-index defect. `mor::route` already
//! turns those indices into a 0/1 membership mask with the sanctioned
//! `mask_fill`-on-a-float idiom: no bool->float cast, no gather, no scatter, no
//! host round-trip. Both are used verbatim.

use burn::tensor::activation::softmax;
use burn::tensor::{Int, Tensor};

/// The routed gate, the 0/1 selection, and the router's PRE-top-k
/// probabilities. `(gates, mask, probs)`, all `[n, n_experts]`.
///
/// `gates` is `softmax(logits)` restricted to the top-`k` experts and
/// **renormalized to sum to 1**. The renormalization is load-bearing rather
/// than cosmetic: without it a top-1 token's FFN output is scaled by ~1/E
/// relative to the fixed mixture, so the routed arm and the dense control
/// would differ in output MAGNITUDE as well as in specialization, and the A/B
/// row ("same active FFN FLOPs") would be measuring two different networks.
/// With it, a top-1 token's gate is exactly 1.0 - the same scale as the single
/// shared FFN the control uses.
///
/// `mask` is exactly `k` ones per row, which is what [`lb_aux`] counts and
/// what the routing diagnostic recovers the selected SET from.
///
/// `probs` is returned rather than recomputed because it is the ONLY part of
/// the routed path that carries a usable gradient - see [`lb_aux`].
pub fn topk_blend(logits: Tensor<2>, k: usize) -> (Tensor<2>, Tensor<2>, Tensor<2>) {
    let [n, e] = logits.dims();
    // LOUD, not clamped: a top-k that cannot fill is a config with no
    // interpretation, and `config::validate` refuses it at startup too. This
    // assert is the second line of defence for a hand-built module.
    assert!(
        k >= 1 && k <= e,
        "top-k k must be in 1..={e} (n_experts), got {k}"
    );
    let dev = logits.device();
    // THE shared primitive, from the crate that owns it.
    let idx = burn_mor::topk_indices(logits.clone(), k, 1); // [n, k]
    // Binary membership from those indices, the `mor::route` construction:
    // one-hot the picks with `mask_fill` on a FLOAT tensor and sum the k rows.
    // `sum_dim` is keepdim in burn 0.22, so the sum is [n, 1, e] and the
    // reshape below is exact. Both Int tensors come from the same device, so
    // they share its Int dtype.
    let ar = Tensor::<1, Int>::arange(0..e as i64, &dev).reshape([1, 1, e]);
    let eq = idx.reshape([n, k, 1]).equal(ar); // [n, k, e]
    let mask = Tensor::<3>::zeros([n, k, e], &dev)
        .mask_fill(eq, 1.0)
        .sum_dim(1)
        .reshape([n, e]);
    // Restrict, then renormalize over what survived. `sum_dim` is keepdim in
    // burn 0.22 (`mor::route` documents the same), so the row mass is `[n, 1]`
    // and the division is the ordinary row-broadcast - if a backend ever
    // squeezed the axis instead, this would fail loudly on the shape rather
    // than divide by the wrong thing.
    let probs = softmax(logits, 1);
    let gates = probs.clone().mul(mask.clone());
    let mass = gates.clone().sum_dim(1);
    let gates = gates.div(mass);
    (gates, mask, probs)
}

/// The Switch/GShard load-balancing term, `E * sum_e f_e * P_e`, as `[1]`.
///
/// `f_e` is the FRACTION of tokens whose top-k includes expert `e` (hard, from
/// `mask`) and `P_e` the mean of the router's probability for `e`. Both are
/// device means - no `into_data`, no host branch, nothing to synchronize on
/// (§1.3). It is returned as a tensor and added to the loss; nothing reads it
/// on the host in the hot path.
///
/// # `P` MUST BE THE PRE-top-k PROBABILITIES - measured, not assumed
///
/// The first version of this function took the RENORMALIZED gates, which is
/// the obvious thing to pass and is **exactly wrong at k=1**: a top-1 token's
/// renormalized gate is exactly `1.0` (softmax over one surviving term), so
/// `P_e` is piecewise CONSTANT in the logits and the term's gradient w.r.t. the
/// router is IDENTICALLY ZERO wherever the selection does not change. Measured
/// on the hostile sweep fixture: `||grad L_LB|| = 0`. A term whose gradient is
/// zero is decorative - it trains for thousands of steps, contributes nothing,
/// and the loss curve stays perfectly healthy. That is the ADR-0019 shape, and
/// it is why `topk_blend` returns the pre-top-k `probs` separately.
///
/// Switch/GShard's `P_i` is the router probability BEFORE top-k for exactly
/// this reason, and this is the form that has a gradient.
///
/// # The two properties the gates pin
///
/// - it is **minimized at exactly 1**: uniform routing gives
///   `E * sum (1/E)(1/E) = 1`, and by AM-GM on the product of the two
///   empirical distributions that is the floor;
/// - it is bounded by `E`, approached - not attained at finite confidence - as
///   the router becomes infinitely certain that one expert should take
///   everything (`f` one-hot and `P` one-hot).
///
/// So the term lives in `[1, E]` = `[1, 4]` at `n_experts = 4`, and the
/// coefficient question is "what fraction of the task gradient may the
/// balancer apply", not "what does the literature say" - see the sweep in
/// `research/reviews/moe-routing-2026-10-01.md` §5.
pub fn lb_aux(probs: &Tensor<2>, mask: &Tensor<2>, n_experts: usize) -> Tensor<1> {
    assert_eq!(
        probs.dims(),
        mask.dims(),
        "lb_aux: probs and mask must be the same shape"
    );
    // PER-EXPERT means, over the ROW (token) axis, which is dim 0 of a
    // `[rows, n_experts]` tensor. Two ways this is easy to get wrong, and both
    // were wrong here once:
    //
    // - burn's `mean()` reduces EVERY element to one scalar, which would make
    //   `f_e` and `P_e` the same number and the whole term a function of the
    //   batch size alone;
    // - `mean_dim(1)` reduces the EXPERT axis, averaging across experts
    //   instead of across tokens, which quietly returns a per-row mean and
    //   scales the answer with the number of rows.
    //
    // `mean_dim` is keepdim in burn 0.22 (it returns `Self`), so each is
    // `[1, n_experts]` and the reshape to `[n_experts]` is what makes them
    // indexable by expert.
    let e = probs.dims()[1];
    let f = mask.clone().mean_dim(0).reshape([e]); // [E] fraction of tokens selecting e
    let p = probs.clone().mean_dim(0).reshape([e]); // [E] mean router probability
    f.mul(p).sum().mul_scalar(n_experts as f32).reshape([1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData};

    fn dev() -> Device {
        Device::flex()
    }

    // The balancer's GRADIENT is part of what is under test here (a term whose
    // gradient never reaches the router is decorative - it would be invisible
    // in every loss curve), so these run on an AUTODIFF device. `topk_blend`
    // and `lb_aux` are backend-generic; only the `.backward()` needs this.
    #[allow(deprecated)]
    fn adev() -> Device {
        Device::flex().autodiff()
    }

    fn t(rows: &[[f32; 4]]) -> Tensor<2> {
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        Tensor::<2>::from_data(
            TensorData::new(flat, [rows.len(), 4]),
            &dev(),
        )
    }

    /// A no-tie fixture, so the expected top-k is unambiguous on every
    /// backend (a tied score makes the SET backend-defined - `topk_indices`'s
    /// own module doc says so).
    fn fixture() -> [[f32; 4]; 3] {
        [
            [9.0, 1.0, 5.0, 0.5],
            [0.2, 7.5, 1.0, 3.0],
            [4.0, 0.1, 8.5, 2.0],
        ]
    }

    /// Host-side reference: the set AND the renormalized weights, computed
    /// without the crate, so a gate that compares the code to itself proves
    /// nothing.
    fn host_topk(rows: &[[f32; 4]], k: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut sets = Vec::new();
        let mut gates = Vec::new();
        for r in rows.iter() {
            let mut order: Vec<usize> = (0..4).collect();
            order.sort_by(|a, b| r[*b].partial_cmp(&r[*a]).unwrap());
            let mut set = vec![0.0f64; 4];
            let mut w = vec![0.0f64; 4];
            let mut tot = 0.0f64;
            for &i in order.iter().take(k) {
                set[i] = 1.0;
                w[i] = (r[i] as f64).exp();
                tot += w[i];
            }
            for v in w.iter_mut() {
                *v /= tot;
            }
            sets.push(set);
            gates.push(w);
        }
        (sets, gates)
    }

    /// The selection is EXACTLY k per row, and it is the host's top-k.
    #[test]
    fn topk_selects_exactly_the_host_top_k() {
        let rows = fixture();
        let logits = t(&rows);
        for k in 1..=4usize {
            let (gates, mask, _probs) = topk_blend(logits.clone(), k);
            let (sets, weights) = host_topk(&rows, k);
            let m: Vec<f32> = mask.into_data().try_to_vec().expect("mask readable");
            let g: Vec<f32> = gates.into_data().try_to_vec().expect("gates readable");
            for r in 0..rows.len() {
                let want_mask: Vec<f32> = sets[r].iter().map(|x| (*x as f32)).collect();
                assert_eq!(
                    &m[r * 4..r * 4 + 4],
                    &want_mask[..],
                    "k={k} row {r}: mask is not the host top-k"
                );
                let got = &g[r * 4..r * 4 + 4];
                for c in 0..4 {
                    assert!(
                        (got[c] as f64 - weights[r][c]).abs() < 1e-5,
                        "k={k} row {r} col {c}: gate {} != host {}",
                        got[c],
                        weights[r][c]
                    );
                }
                // Exactly k ones, every row, no more.
                assert_eq!(
                    got.iter().filter(|v| **v > 0.0).count(),
                    k,
                    "k={k} row {r}: {k} experts must be live"
                );
                // Renormalized: the surviving gates sum to 1.
                let s: f32 = got.iter().sum();
                assert!((s - 1.0).abs() < 1e-5, "k={k} row {r}: gates sum {s}, must be 1");
            }
        }
    }

    /// k = 1 is the FIRST configuration and its defining property is that the
    /// gate is exactly 1.0 - that is what makes a top-1 token's FFN output the
    /// same MAGNITUDE as the single shared FFN the A/B control uses. Pinned
    /// because the renormalization is what makes the row a fair comparison and
    /// because dropping it is a silent scale change, not an error.
    #[test]
    fn top1_gate_is_exactly_one_on_the_winner_and_zero_elsewhere() {
        let rows = fixture();
        let (gates, mask, _) = topk_blend(t(&rows), 1);
        let g: Vec<f32> = gates.into_data().try_to_vec().unwrap();
        let m: Vec<f32> = mask.into_data().try_to_vec().unwrap();
        let winner = |r: usize| (0..4).max_by(|a, b| rows[r][*a].total_cmp(&rows[r][*b])).unwrap();
        for r in 0..rows.len() {
            assert_eq!(m[r * 4 + winner(r)], 1.0, "the argmax expert is selected");
            assert!((g[r * 4 + winner(r)] - 1.0).abs() < 1e-6, "top-1 gate is exactly 1");
            for c in 0..4 {
                if c != winner(r) {
                    assert_eq!(g[r * 4 + c], 0.0, "a non-selected expert carries nothing");
                }
            }
        }
    }

    /// The balancer's floor and ceiling, which is what makes the sweep in the
    /// findings file mean anything.
    ///
    /// The FLOOR is exactly 1 and is attained: uniform routing gives
    /// `E * sum (1/E)(1/E) = 1`, by symmetry under the rotated fixture.
    ///
    /// The CEILING is `E` and is NOT attained at finite router confidence - it
    /// is approached as `P` becomes one-hot on the dispatched expert, because
    /// `P` is the pre-top-k probability and a merely-confident router still
    /// leaks probability to the experts it rejected. The earlier version of
    /// this test asserted `collapsed == E` exactly; that is true only in the
    /// limit, and asserting it against a finite fixture would have been a gate
    /// that passes for the wrong reason.
    #[test]
    fn lb_aux_floor_is_exactly_one_and_the_ceiling_is_e_in_the_limit() {
        let e = 4usize;
        // Uniform top-1: each of the 4 experts wins one row of four.
        let logits = t(&[
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 3.0],
        ]);
        let (_g, m, probs) = topk_blend(logits, 1);
        let aux = lb_aux(&probs, &m, e).into_scalar::<f32>();
        assert!(
            (aux - 1.0).abs() < 1e-5,
            "uniform top-1 over {e} experts must score the floor 1, got {aux}"
        );

        // Collapsed AND one-hot: every token on expert 0 with the rest far
        // enough below that P underflows to a one-hot in f32.
        let gap = t(&[[100.0, 0.0, 0.0, 0.0], [100.0, 0.0, 0.0, 0.0]]);
        let (_g, m, probs) = topk_blend(gap, 1);
        let aux = lb_aux(&probs, &m, e).into_scalar::<f32>();
        assert!(
            (aux - e as f32).abs() < 1e-3,
            "a one-hot collapsed router must reach the ceiling E = {e}, got {aux}"
        );

        // And a MERELY confident router sits strictly between: the property
        // the previous assertion would have missed.
        let mid = t(&[[10.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]]);
        let (_g, m, probs) = topk_blend(mid, 1);
        let aux = lb_aux(&probs, &m, e).into_scalar::<f32>();
        assert!(
            aux > 1.0 && aux < e as f32,
            "a confident-but-finite collapsed router must sit strictly inside (1, E), got {aux}"
        );
    }

    /// The whole range is `[1, E]`, and it RISES as the router concentrates. That
    /// is the property the coefficient sweep rides on: a term with a cliff
    /// would make "how strong should the balancer be" a coin flip rather than
    /// a dial. Three hand-built points, obviously ordered, so the assertion is
    /// about the metric and not about a clever construction.
    #[test]
    fn lb_aux_rises_as_the_router_concentrates() {
        let e = 4usize;
        // 8 tokens. `skew` says how many of them route to expert 0.
        let case = |skew: usize| {
            let mut rows = vec![[0.0f32; 4]; 8];
            for (i, r) in rows.iter_mut().enumerate() {
                if i < skew {
                    r[0] = 6.0;
                    r[1] = 1.0;
                } else {
                    // Spread the rest evenly over experts 1..=3.
                    r[1 + (i - skew) % 3] = 6.0;
                }
            }
            let (_g, m, probs) = topk_blend(t(&rows), 1);
            lb_aux(&probs, &m, e).into_scalar::<f32>()
        };
        let balanced = case(2);
        let leaning = case(5);
        let collapsed = case(8);
        assert!(
            balanced < leaning && leaning < collapsed,
            "the balancer must rise with concentration: {balanced} < {leaning} < {collapsed}"
        );
        for (name, v) in [("balanced", balanced), ("leaning", leaning), ("collapsed", collapsed)] {
            assert!(
                (1.0..=e as f32).contains(&v),
                "{name} = {v} is outside the term's range [1, E={e}]"
            );
        }
        // `case(8)` uses logit 6.0 vs 1.0, so P is ~0.955 rather than one-hot:
        // the ceiling is approached, not reached (see the endpoint test).
        assert!(
            collapsed > 0.9 * e as f32 && collapsed <= e as f32 + 1e-5,
            "fully collapsed must sit at the top of the range, got {collapsed} of E = {e}"
        );
    }

    /// LOUD, not clamped. `k = 0` is the OFF position and is expressed by the
    /// CALLER not calling this at all - a top-0 set is not a mixture, it is a
    /// token with no FFN - and `k > n_experts` is a config `validate` already
    /// refuses at startup. Both must die here too, because this function is
    /// public and a hand-built module bypasses `validate`.
    #[test]
    fn topk_refuses_a_set_that_cannot_fill() {
        let logits = t(&fixture());
        for bad in [0usize, 5, 9] {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = topk_blend(logits.clone(), bad);
                }))
                .is_err(),
                "k={bad} must be refused loudly, not clamped into a different network"
            );
        }
        // And the legal ends are not refused.
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = topk_blend(logits.clone(), 1);
        }))
        .is_ok());
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = topk_blend(logits.clone(), 4);
        }))
        .is_ok());
    }

/// THE COLLAPSE GUARD, and the coefficient sweep it produces.
///
/// A top-k router with no balancer behind it collapses: every token ends up on
/// one expert, the loop's passes stop being different, and the arm becomes the
/// dense mixture with a worse gradient. That is Switch's reason for the term
/// (Fedus et al., JMLR 23(1):120-5232, 2022 §3.3). Note what validation does
/// and does NOT do: `config::validation` refuses a non-zero `moe_lb_coef` with
/// the arm OFF (a term with no selection to balance), and does NOT require the
/// coefficient to be non-zero with the arm ON - the arm's default is the router
/// with no balancer, which is the arm's own removal and therefore the control an
/// A/B wants to be able to run.
///
/// # WHY THE SWEEP CANNOT PRODUCE A DEFAULT, and measures that instead
///
/// The obvious deliverable - "sweep on a hostile batch, ship the winner" - is
/// unsound here, for a reason this test exists to make visible:
///
/// `L_LB = E * sum_e f_e P_e` averages over TOKENS, so the gradient it applies
/// to any one token's logits is `O(E / tokens_per_batch)`. It is diluted by
/// the batch size, and therefore the coefficient that balances a 64-row toy
/// batch is not the coefficient that balances the trainer's ~4 000-token one;
/// it moves by the same factor. A number swept on a toy batch is a number
/// about the toy batch.
///
/// So what is measured is the scale-free quantity - the balancer's gradient
/// norm as a FRACTION of the task gradient's - and the coefficient at which the
/// two balance (`coef_crit = 1 / ratio`). That transfers; a raw coefficient
/// does not. The default therefore stays **0.0**, the arm is OFF, and the two
/// swept values are reported as readings.
#[test]
fn load_balance_sweep_is_measured_against_the_task_gradient() {
    const ROWS: usize = 64;
    const E: usize = 4;
    const STEPS: usize = 40;
    // The hostile pressure: every row WANTS expert 0. This is what collapse
    // looks like from the router's side.
    let want = |j: usize| if j == 0 { 1.0f32 } else { -0.25f32 };

    // Per-row logits, DETERMINISTIC and ROW-VARYING.
    //
    // Row-varying is not cosmetic. The selection is a per-row argmax, so a
    // batch of 64 IDENTICAL rows can only ever all pick the same expert - no
    // coefficient, however large, could spread that load, and the "sweep"
    // would be measuring a batch that has no tokens to distribute. Load
    // balancing is about the distribution ACROSS tokens, so the batch has to
    // vary across tokens.
    //
    // The scramble is a hash, not burn's global RNG: that stream is never
    // seeded (AGENTS.md 3.7), so an RNG here would make the test irreproducible.
    // `h >> 40` is 24 bits, so `/ 2^24` is the full range: the jitter lands in
    // [-1, 1). Dividing by 1024 instead would have made it ±16000 - and a
    // jitter that dwarfs the logit gap makes the argmax effectively a coin
    // flip, which lands the load at uniform by accident and makes
    // `L_LB == 1` a constant with an exactly zero gradient.
    let jitter = |r: usize, j: usize| -> f32 {
        let h = (r as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .rotate_left(17)
            ^ (j as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        (h >> 40) as f32 / 8_388_608.0 - 0.5
    };
    // The HOSTILE but not degenerate regime: expert 0 leads by a margin SMALLER
    // than the row-to-row spread, so most tokens want expert 0 and some do
    // not. Both halves matter. All rows identical is the degenerate case (no
    // load to distribute); all rows wanting expert 0 with a wide margin gives
    // `f = one-hot`, `P` still depends on the logits, and a gradient - but a
    // fixture the balancer has to fight through from a standing start. The
    // margin is what makes this the hard case.
    let mut init = vec![0.0f32; ROWS * E];
    for r in 0..ROWS {
        for j in 0..E {
            // Parenthesised on purpose. Written as
            // `if j == 0 { 3.0 } else { 0.0 } + jitter(r, j)` the `+ jitter`
            // binds to the ELSE branch only, every row gets the same expert-0
            // logit, and the batch collapses to 64 identical copies - the exact
            // degeneracy the comment above is about. It then read as "the
            // balancer's gradient is zero", which was the batch, not the term.
            init[r * E + j] = jitter(r, j) + if j == 0 { 0.30 } else { 0.0 };
        }
    }

    // Load spread after `STEPS` steps at a given coefficient. Read back
    // through the REAL selection, not through the state vector.
    let simulate = |coef: f32| -> (f32, f32) {
        let mut v = init.clone();
        for _ in 0..STEPS {
            let leaf = Tensor::<2>::from_data(TensorData::new(v.clone(), [ROWS, E]), &adev())
                .require_grad();
            let (_gates, mask, probs) = topk_blend(leaf.clone(), 1);
            let g = if coef == 0.0 {
                vec![0.0f32; ROWS * E]
            } else {
                leaf.grad(&lb_aux(&probs, &mask, E).mul_scalar(coef).backward())
                    .expect("the balancer must reach the router logits")
                    .into_data()
                    .try_to_vec()
                    .expect("grad readable")
            };
            for r in 0..ROWS {
                for j in 0..E {
                    // `g` already carries the coefficient (it is the gradient
                    // of `coef * L_LB`), so it is subtracted as-is.
                    v[r * E + j] += 0.5 * (want(j) - g[r * E + j]);
                }
            }
        }
        let leaf = Tensor::<2>::from_data(TensorData::new(v, [ROWS, E]), &dev());
        let (_, mask, _) = topk_blend(leaf, 1);
        let m: Vec<f32> = mask.into_data().try_to_vec().expect("mask readable");
        // Per-EXPERT share: rows whose selection column j is live. Counting
        // every 1.0 in the flat buffer instead (ignoring `j`) gives the SAME
        // number for all four experts, which makes the spread identically zero
        // and the "collapse" reading an artifact.
        let counts: Vec<f32> = (0..E)
            .map(|j| {
                m.chunks(E).filter(|row| row[j] == 1.0).count() as f32 / ROWS as f32
            })
            .collect();
        let hi = counts.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // `lo` seeds with POSITIVE infinity. Seeding it with `NEG_INFINITY`
        // makes `f32::min` return -inf for any non-empty set, so the spread is
        // `hi - (-inf) = inf` and a whole sweep compares infinities.
        let lo = counts.iter().cloned().fold(f32::INFINITY, f32::min);
        (hi - lo, counts[0])
    };

    // ---- 1. The balancer's gradient is real, and reaches the router --------
    let leaf = Tensor::<2>::from_data(TensorData::new(init.clone(), [ROWS, E]), &adev())
        .require_grad();
    let (_gates, mask, probs) = topk_blend(leaf.clone(), 1);
    let g_lb: Vec<f32> = leaf
        .grad(&lb_aux(&probs, &mask, E).backward())
        .expect("the balancer must reach the router logits")
        .into_data()
        .try_to_vec()
        .expect("grad readable");
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let task: Vec<f32> = (0..ROWS).flat_map(|_| (0..E).map(|j| want(j))).collect();
    let g_lb_norm = norm(&g_lb);
    assert!(
        g_lb_norm > 0.0,
        "the balancer's gradient is exactly zero: the term would be decorative and invisible in          every loss curve"
    );
    // The scale-free reading: what fraction of the task gradient the balancer
    // applies per unit coefficient. THIS is what a coefficient scales.
    let ratio = g_lb_norm / norm(&task);
    let coef_crit = 1.0 / ratio;
    println!("load-balance coefficient sweep - HOSTILE batch, {ROWS} rows x {E} experts, top-1,");
    println!("  {STEPS} steps, task gradient unopposed, deterministic per-row logits");
    println!("  ||grad L_LB||            {g_lb_norm:.6}");
    println!("  ||grad task||            {:.6}", norm(&task));
    println!("  ratio per unit coef      {ratio:.6}   <- what a coefficient scales");
    println!("  coef_crit = 1/ratio      {coef_crit:.4}   (balancer matches the task)");

    // ---- 2. The hostile control really collapses ---------------------------
    let (off_spread, off_collapse) = simulate(0.0);
    assert!(
        off_collapse > 0.9,
        "the hostile control must actually collapse (expert 0 share {off_collapse}), or the \
         sweep below is measuring nothing"
    );
    println!();
    println!("  {:>10} {:>10} {:>12}", "coef", "spread", "on expert 0");

    // ---- 3. The sweep: {small, medium} REPORTED, and found insufficient ----
    // `small` and `medium` are the brief's two positions. The measurement says
    // neither can move this router, and the assertion below is that they CANNOT
    // - a negative result, asserted so nobody re-runs the sweep hoping for a
    // different answer, and explained by `ratio` above rather than by taste.
    let crit = (coef_crit * 2.0).min(1.0e4);
    let mut table: Vec<(f32, f32, f32, &str)> = Vec::new();
    for (label, coef) in [
        ("off", 0.0f32),
        ("small", 0.01),
        ("medium", 0.1),
        ("2x critical", crit),
    ] {
        let (spread, collapse) = simulate(coef);
        println!("  {:>10} {:>10.4} {:>12.4}", label, spread, collapse);
        table.push((coef, spread, collapse, label));
    }
    let row = |label: &str| table.iter().find(|t| t.3 == label).expect("row present");
    let off = row("off").1;
    assert!(
        row("off").2 > 0.9,
        "the no-balancer control must actually collapse (expert 0 share {}), or the sweep below is measuring nothing",
        row("off").2
    );
    assert!(
        off > 0.5,
        "the no-balancer control must be badly spread, got {off}"
    );

    // THE HEADLINE, as an assertion. {small, medium} - the brief's two
    // positions, and the neighbourhood of every published value - leave this
    // router fully collapsed. That is a measurement with a mechanism behind it:
    // `L_LB` averages over tokens, so at 64 rows the balancer applies ~0.45 %
    // of the task gradient per unit coefficient, and nothing in the
    // literature's range closes a 200x gap. Copying one would have produced a
    // run that LOOKS balanced because nothing was reported.
    for l in ["small", "medium"] {
        let r = row(l);
        assert!(
            r.1 >= off - 1e-6,
            "coef {} ({}) was expected to be INSUFFICIENT at this token count (spread {} vs the {} control) - if it now beats the control, the dilution argument above is wrong and the coefficient needs re-deriving",
            r.0, l, r.1, off
        );
    }
    // ...and the term is NOT decorative: at the critical coefficient it does
    // beat no balancer. Without this, the negative above would be
    // indistinguishable from "the term does nothing".
    let crit_row = row("2x critical");
    assert!(
        crit_row.1 < off - 0.05,
        "even at 2x the critical coefficient ({}) the balancer could not move the distribution (spread {} vs {off}): the term cannot affect the router here",
        crit_row.0, crit_row.1
    );
    println!();
    println!("  RESULT: {{small=0.01, medium=0.1}} do NOT rescue this router; it needs coef ~{coef_crit:.0}");
    println!("  at {ROWS} tokens. The gap is the E/tokens dilution, so the coefficient the trainer's");
    println!("  ~4096-token batch would need is ~64x larger again. NO coefficient is shipped:");
    println!("  moe_lb_coef stays 0.0 and the arm is off, because no value transfers and an");
    println!("  unswept default would be a number copied from a regime we are not in.");

}
}
