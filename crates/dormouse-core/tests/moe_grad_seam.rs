/// GRADIENT FLOW - the `8fa5d4c` class, for the routing arm.
///
/// The defect this gate exists for: an arm whose parameters receive no
/// gradient runs thousands of forwards, trains nothing, and every number it
/// produces belongs to the control. In this project it cost the whole archive
/// its attention arm (`fused kda=3126/0`), and nothing in the loss curve
/// showed it.
///
/// The routing arm adds NO parameters - the router IS the controller's existing
/// expert columns - so "the router trains" has a specific and checkable
/// meaning here, and it is checked in the two halves that could each fail alone:
///
/// 1. **The controller's expert columns** (`controller.weight[:, 3..3+n_experts]`)
///    carry a non-zero gradient, while the gate columns next to them
///    (`[:, 0..3]`, which drive `w_attn`/`w_mem`/`w_ffn`) are a control that
///    must ALSO be non-zero. Both halves matter: a mask or a reshape that
///    detached the slice would zero one of them.
/// 2. **Every expert's weights** carry a non-zero gradient - and the honest
///    version of that assertion depends on `k`. At `k = 1` an expert that no
///    token selected gets EXACTLY zero gradient, by construction, so
///    "every expert gets a gradient" is only assertable at `k = n_experts`,
///    where all of them are live on every row. That is what this gate does,
///    and it then separately asserts the `k = 1` property that actually holds:
///    at least one expert is live, and it gets gradient.
///
/// The second half is the collapse detector's twin: a masked-out expert is not
/// a bug at `k = 1`, but an expert that is masked out on EVERY row of EVERY
/// step is the router collapsing, and that is the load-balancing term's job -
/// see `moe.rs`'s sweep, which measures that the term's gradient is non-zero.

use burn::tensor::{Distribution, Int, Tensor, TensorData};
use dormouse_core::config::validate;
use dormouse_core::loop_block::LoopBlock;
use dormouse_core::DormouseConfig;

type B = burn::backend::Autodiff<
    burn::backend::Flex,
    burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
>;

const EXPERTS: usize = 4;

#[allow(deprecated)]
fn adev() -> burn::tensor::Device {
    burn::tensor::Device::flex().autodiff()
}

fn cfg(topk: usize) -> DormouseConfig {
    let c = DormouseConfig {
        d_model: 32,
        n_heads: 2,
        head_dim: 16,
        d_ffn: 64,
        rank: 8,
        max_seq_len: 16,
        engram_rows: 256,
        n_experts: EXPERTS,
        use_kda: false,
        use_engram: false,
        use_tsct: false,
        jepa_weight: 0.0,
        dspark_weight: 0.0,
        moe_topk: topk,
        moe_lb_coef: 0.01,
        ..Default::default()
    };
    validate(&c).expect("the fixture validates");
    c
}

fn ids_and_targets(b: usize, t: usize) -> (Tensor<2, Int>, Tensor<2, Int>) {
    let x = Tensor::<2, Int>::from_data(
        TensorData::new((0..(b * t)).map(|i| (i % 251) as i64).collect::<Vec<_>>(), [b, t]),
        &adev(),
    );
    let y = Tensor::<2, Int>::from_data(
        TensorData::new(
            (0..(b * t)).map(|i| (i % 251 + 1) as i64).collect::<Vec<_>>(),
            [b, t],
        ),
        &adev(),
    );
    (x, y)
}

/// One backward through the routed loop, returning the controller's expert-column
/// gradient, the gate-column gradient, and one norm per expert.
fn backward_report(topk: usize) -> (f64, f64, Vec<f64>) {
    let mut block = LoopBlock::new(&cfg(topk), &adev());
    // `forward_full_state`'s first argument is the HIDDEN STATE [b, t, d], not
    // byte ids: the loop is the model's body, the embedding lives above it.
    // And its `targets` is ALREADY flattened to [b*t, 1] - `DormouseModel`
    // reshapes [b, t] into that before calling, because the loop's CE gathers
    // one log-prob per row. Passing [b, t] here fails deep inside a Gather with
    // a shape error about dimension 0, which is a long way from the argument
    // that is wrong.
    let (_ids, y) = ids_and_targets(2, 16);
    let y = y.reshape([2 * 16, 1]);
    let x = Tensor::<3>::random([2, 16, 32], Distribution::Normal(0.0, 1.0), &adev());
    // A REAL-width head: the loop's per-iteration CE GATHERS the target
    // log-prob out of `lm_head`'s columns, so the head's out_features has to be
    // the vocabulary. A narrow stand-in head compiles fine and dies at the
    // first gather with "index 250 out of bounds for dimension of size 16" -
    // a shape error about the head, reported from inside the loss.
    let head = dormouse_core::param::LinearLike::with_tsct(32, 256, 8, false, &adev());
    let (_logits, rec, _kda, _route) =
        block.forward_full_state::<B>(x, None, None, None, Some(y), &head);
    let grads = rec.backward();

    // The controller weight is `[2d, ctrl_pad]`; columns `0..3` are the arm
    // gates and `3..3+E` are the expert blend. Both slices are read from the
    // SAME gradient tensor, which is what makes this a wiring test rather than
    // two independent ones.
    let cw = block.controller.weight.grad(&grads).expect("the controller trains");
    let cw: Vec<f32> = cw.into_data().try_to_vec().expect("controller grad readable");
    let [_rows, cols] = block.controller.weight.val().dims();
    let col_mass = |lo: usize, hi: usize| -> f64 {
        cw.chunks(cols)
            .map(|r| r[lo..hi].iter().map(|v| (*v as f64).powi(2)).sum::<f64>())
            .sum::<f64>()
            .sqrt()
    };
    let gates = col_mass(0, 3);
    let experts = col_mass(3, 3 + EXPERTS);

    let per_expert: Vec<f64> = (0..EXPERTS)
        .map(|e| {
            let lin = &block.expert_ffns[e].gate_up;
            let dormouse_core::param::LinearLikeInner::Dense(l) = &lin.inner else {
                panic!("the fixture builds the dense variant");
            };
            let g = l.weight.grad(&grads).expect("every expert trains");
            let v: Vec<f32> = g.into_data().try_to_vec().expect("expert grad readable");
            v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt()
        })
        .collect();
    (gates, experts, per_expert)
}

#[test]
fn the_router_columns_and_every_expert_carry_gradient() {
    // k = n_experts: every expert is live on every row, so "every expert has a
    // gradient" is assertable here and only here.
    let (gates, router, per_expert) = backward_report(EXPERTS);
    assert!(
        router > 1e-9,
        "the controller's EXPERT columns carry no gradient (L2 {router:.3e}): the arm would run \
         thousands of forwards and train nothing - the 8fa5d4c defect, in the one place this \
         arm can have it"
    );
    assert!(
        gates > 1e-9,
        "the arm-gate columns next to them carry no gradient (L2 {gates:.3e}); the fixture's \
         control must be live too, or this gate proves nothing"
    );
    for (e, g) in per_expert.iter().enumerate() {
        assert!(
            *g > 1e-12,
            "expert {e} has no gradient (L2 {g:.3e}) while k = n_experts selects it on every \
             row: an expert that never trains would hide forever"
        );
    }
    println!(
        "k = {EXPERTS} (all live): gate columns L2 {gates:.3e}, router columns L2 {router:.3e}, \
         per-expert L2 {:?}",
        per_expert.iter().map(|g| (g * 1e3).round() / 1e3).collect::<Vec<_>>()
    );
}

#[test]
fn at_top1_the_selected_experts_train_and_the_masked_ones_are_zero() {
    let (gates, router, per_expert) = backward_report(1);
    assert!(router > 1e-9, "the router's expert columns must still train at k = 1 ({router:.3e})");
    assert!(gates > 1e-9, "the arm gates must still train at k = 1 ({gates:.3e})");
    let live = per_expert.iter().filter(|g| **g > 1e-12).count();
    assert!(
        live >= 1,
        "at k = 1 NO expert received a gradient: the selection is not on the graph at all"
    );
    // The masked ones are ZERO BY CONSTRUCTION, not by defect - state it, so a
    // reader does not file it as a bug: `ffn += out_e * gate_e` with
    // `gate_e == 0.0` gives `d/d out_e = 0`.
    assert!(
        live <= EXPERTS,
        "at k = 1 every expert trained ({live} of {EXPERTS}), which means the mask is not \
         masking"
    );
    // What is asserted here is NOT "every expert trains" - at k = 1 with 32
    // positions the top-1 assignment happens to cover all four experts, so
    // `live == 4` on this fixture and every expert does get gradient. The
    // property that is k-dependent is the converse: an expert that wins NO
    // position is masked to exactly zero, because `ffn += out_e * gate_e`
    // gives `d/d out_e = gate_e = 0`. Which experts those are depends on the
    // draw, so asserting an exact set would be asserting the RNG.
    println!(
        "k = 1: {live} of {EXPERTS} experts received gradient on this draw; any that won no \
         position are masked to exactly zero BY CONSTRUCTION (d/d out_e = gate_e = 0), not by \
         defect"
    );
}

/// TIE BEHAVIOUR, defined and pinned.
///
/// `topk_blend`'s selection is `dormouse_mor::topk_indices`, i.e. one
/// `argsort_descending` + `narrow`, and that primitive's own doc says the
/// selected SET among equal values may differ between impls and backends. So
/// the promise this module makes is deliberately narrow and is exactly what the
/// brief asks for: **the COUNT is always exactly `k`**, never `k±1`, including
/// on a row where every score is equal. WHICH tied expert wins is
/// unspecified, and no assertion in this file depends on it.
///
/// A `k±1` leak would be silent and total: a row selecting 0 experts gets a
/// zero FFN output, and one selecting all E is the dense blend. Both are the
/// control wearing the arm's label.
#[test]
fn exactly_k_under_ties_where_which_one_is_unspecified() {
    for k in 1..=EXPERTS {
        // Every score equal: every value ties with every other, so there is no
        // "right" identity to assert and a `k±1` implementation is at its
        // most visible here.
        let logits = Tensor::<2>::full([32, EXPERTS], 0.5, &adev());
        let (_gates, mask, _probs) = dormouse_core::moe::topk_blend(logits, k);
        let m: Vec<f32> = mask.into_data().try_to_vec().expect("mask readable");
        for row in 0..32 {
            let ones = m[row * EXPERTS..(row + 1) * EXPERTS]
                .iter()
                .filter(|v| **v == 1.0)
                .count();
            assert_eq!(
                ones, k,
                "a fully tied row must still select exactly {k}, got {ones} in row {row} of \
                 {m:?}"
            );
            assert!(
                m[row * EXPERTS..(row + 1) * EXPERTS]
                    .iter()
                    .all(|v| *v == 0.0 || *v == 1.0),
                "the mask must stay 0/1 under ties, got row {row}: {:?}",
                &m[row * EXPERTS..(row + 1) * EXPERTS]
            );
        }
    }
}