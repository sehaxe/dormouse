//! The sparse-routing seam: the FFN branch's expert blend, which is where the
//! loop stops being a fixed mixture and becomes a router.
//!
//! `moe.rs`'s own tests pin the SELECTION ARITHMETIC against a host-side
//! top-k. This file pins the WIRING - that the branch is off by default, that
//! the arm changes the mixture the FFN branch actually multiplied by, and that
//! the router receives the pass number.
//!
//! Every comparison here is one model against a MUTATED COPY OF ITSELF, so
//! the only difference is the thing under test. Two separately constructed
//! models differ in their random weights and these gates would pass for the
//! wrong reason (`gr_seam.rs` says so, and it learned it).
//!
//! Run: `cargo test -p dormouse-core --test moe_routing_seam`

use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn::backend::Autodiff;
use burn::module::{Module, Param};
use burn::tensor::{Distribution, Tensor};
use dormouse_core::config::validate;
use dormouse_core::loop_block::LoopBlock;
use dormouse_core::mixture_probe;
use dormouse_core::param::LinearLike;
use dormouse_core::probe;
use dormouse_core::DormouseConfig;

// The CPU backend (burn-flex) under the SAME checkpointing strategy the
// trainer uses. Nothing here is device-specific: what is under test is which
// weight vector the FFN branch uses.
type B = Autodiff<burn::backend::Flex, BalancedCheckpointing>;

const BATCH: usize = 2;
const SEQ: usize = 16;
const EXPERTS: usize = 4;

/// THE FIRST CONFIGURATION: 4 experts, top-1. `moe_topk = 0` is the dense
/// control; `n_experts = 4` is what the corrected design specifies (the
/// 8-expert top-2 evidence sits at 168M+ active params, and `small` is 9.2M).
fn cfg(topk: usize) -> DormouseConfig {
    let c = DormouseConfig {
        d_model: 32,
        n_heads: 2,
        head_dim: 16,
        d_ffn: 64,
        rank: 8,
        max_seq_len: SEQ,
        engram_rows: 256,
        n_experts: EXPERTS,
        // The arms that are not under test are off, so a failure names the
        // routing branch rather than the Engram's or KDA's.
        use_kda: false,
        use_engram: false,
        use_tsct: false,
        jepa_weight: 0.0,
        dspark_weight: 0.0,
        moe_topk: topk,
        ..Default::default()
    };
    validate(&c).expect("the fixture validates");
    c
}

#[allow(deprecated)]
fn adev() -> burn::tensor::Device {
    burn::tensor::Device::flex().autodiff()
}

fn x() -> Tensor<3> {
    Tensor::<3>::random([BATCH, SEQ, 32], Distribution::Normal(0.0, 1.0), &adev())
}

fn head() -> LinearLike {
    LinearLike::with_tsct(32, 16, 8, false, &adev())
}

/// One capture of the mixture the FFN branch really used.
fn capture(block: &LoopBlock, x: Tensor<3>, head: &LinearLike) -> Vec<Tensor<2>> {
    mixture_probe::arm();
    let _ = block.forward_full_state::<B>(x, None, None, None, None, head);
    let cap = mixture_probe::take().expect("armed before the forward");
    assert!(
        !mixture_probe::armed(),
        "take() disarms - a sink left armed would grow on every later forward"
    );
    cap
}

fn rows(cap: &[Tensor<2>]) -> Vec<Vec<f32>> {
    let t = &cap[0];
    let v: Vec<f32> = t
        .clone()
        .into_data()
        .try_to_vec()
        .expect("mixture readable");
    let [n, e] = t.dims();
    assert_eq!(v.len(), n * e);
    v.chunks(e).map(|r| r.to_vec()).collect()
}

/// ZERO DEFAULT, in the three forms it can actually fail in.
///
/// 1. The counter: an arm that ran without being counted, or that was counted
///    without running, is the ADR-0019 class. `moe=0/<n>` must be the off state.
/// 2. The PARAMETER SET: the arm reuses the controller's existing expert
///    columns and builds nothing, so an off-arm model and an on-arm model with
///    otherwise identical configs have the same parameter count - which is the
///    claim that keeps every existing checkpoint loadable and keeps queue row 1
///    (pure CE) from needing a re-baseline. This is the assertion that would go
///    red the day someone adds a second router head.
/// 3. The blend: off, the mixture the FFN branch used is a FULL-SUPPORT
///    simplex point (every expert has a positive weight), which is the dense
///    blend's defining property.
///
/// What this does NOT prove, stated so nobody reads more into it: bit-identity
/// against a build from before the field existed. It proves the arm is off,
/// allocates nothing, and produces the dense mixture's signature.
#[test]
fn off_is_the_dense_mixture_and_allocates_nothing() {
    let head = head();
    let c0 = cfg(0);
    let c1 = cfg(1);
    let n0 = Module::num_params(&LoopBlock::new(&c0, &adev()));
    let n1 = Module::num_params(&LoopBlock::new(&c1, &adev()));
    assert_eq!(
        n0,
        n1,
        "the routing arm added {} parameters: an off-arm checkpoint would stop \
         loading, and every measured preset parameter count would move",
        n1 - n0
    );

    let block = LoopBlock::new(&c0, &adev());
    probe::reset();
    let cap = capture(&block, x(), &head);
    assert_eq!(
        cap.len(),
        block.max_iter,
        "one capture per executed iteration"
    );
    let r = rows(&cap);
    for (i, row) in r.iter().enumerate() {
        assert_eq!(row.len(), EXPERTS);
        let s: f32 = row.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-5,
            "iteration {i}: the blend must sum to 1, got {s}"
        );
        assert!(
            row.iter().all(|v| *v > 0.0),
            "iteration {i}: the DENSE blend gives every expert a positive weight, got {row:?} - \
             a zero means something masked this row, and with moe_topk = 0 nothing should"
        );
    }
    assert_eq!(
        probe::count(probe::MOE_ROUTE),
        0,
        "the routing counter moved with the arm OFF"
    );
}

/// THE ARM CHANGES THE MIXTURE THE FFN BRANCH USED - not a re-derivation of
/// it. Same weights, same input, one config field flipped: with top-1 the
/// recorded mixture must have support exactly 1 per row, which the dense
/// capture above shows it does not have.
#[test]
fn routed_support_is_exactly_one_and_the_counter_follows() {
    let head = head();
    let block = LoopBlock::new(&cfg(1), &adev());
    probe::reset();
    let cap = capture(&block, x(), &head);
    assert_eq!(
        probe::count(probe::MOE_ROUTE),
        block.max_iter as u64,
        "one routing decision per executed iteration"
    );
    let r = rows(&cap);
    for (i, row) in r.iter().enumerate() {
        let live: Vec<usize> = (0..EXPERTS).filter(|&j| row[j] > 0.0).collect();
        assert_eq!(
            live.len(),
            1,
            "iteration {i}: top-1 must leave exactly one expert live, got {live:?} in {row:?}"
        );
        assert!(
            (row[live[0]] - 1.0).abs() < 1e-5,
            "iteration {i}: the live gate must be exactly 1 (renormalized), got {}",
            row[live[0]]
        );
        for (j, v) in row.iter().enumerate() {
            if j != live[0] {
                assert_eq!(
                    *v, 0.0,
                    "iteration {i}: a non-selected expert carries nothing"
                );
            }
        }
    }
}

/// THE ROUTER SEES THE PASS NUMBER.
///
/// The routing decision is per (position, pass). This is the cheapest possible
/// proof that the pass index reaches the router, and it is a proof by mutation:
/// at depth 1 the ONLY pass signal in play is `iter_embed[0]`, the input is
/// fixed, and every weight is fixed - so if the mixture does not move when
/// `iter_embed[0]` moves, the router provably ignores which pass it is on.
///
/// This is the gate that justifies NOT adding a separate pass-embedding to the
/// router input: the loop already adds the slot's row to `h` before the
/// controller reads it (`add_iter`, then `cat([h_ctx, h0])`), so the property
/// holds at zero parameter cost. A future refactor that moves the controller
/// upstream of `add_iter` turns this red.
#[test]
fn the_router_receives_the_pass_index() {
    let head = head();
    let c = DormouseConfig {
        max_iter: 1,
        ..cfg(0)
    };
    let mut block = LoopBlock::new(&c, &adev());
    let x = x();
    let before = rows(&capture(&block, x.clone(), &head));

    // Mutate ONE row of the iteration embedding in place: same module, same
    // weights, same input.
    let e = block
        .iter_embed
        .val()
        .clone()
        .into_data()
        .try_to_vec::<f32>()
        .expect("iter_embed readable");
    let moved: Vec<f32> = e
        .iter()
        .enumerate()
        .map(|(i, v)| if i < 32 { -*v - 1.0 } else { *v })
        .collect();
    block.iter_embed = Param::from_tensor(Tensor::<2>::from_data(
        burn::tensor::TensorData::new(moved, [1, 32]),
        &adev(),
    ));

    let after = rows(&capture(&block, x, &head));
    let diff: f32 = before[0]
        .iter()
        .zip(&after[0])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff > 1e-6,
        "changing the iteration embedding moved nothing (max {diff:.3e}): the router does not \
         receive the pass index, so routing would be identical on every pass"
    );
}

/// The balancer term reaches the LOSS, weighted, and only when it has a
/// selection to balance. This is the gate for the ADR-0019 shape the repo has
/// already been bitten by: a config field that reads like an objective and
/// contributes nothing while the loss curve stays healthy.
#[test]
fn the_balancer_reaches_the_loss_only_with_a_selection() {
    use dormouse_core::DormouseModel;
    let mut c = cfg(1);
    c.d_ffn = 64;
    c.d_model = 32;
    // The model must be small enough for a CPU forward; `cfg` already is.
    let ids = Tensor::<2, burn::tensor::Int>::from_data(
        burn::tensor::TensorData::new(
            (0..(BATCH * SEQ))
                .map(|i| (i % 251) as i64)
                .collect::<Vec<_>>(),
            [BATCH, SEQ],
        ),
        &adev(),
    );
    let targets = Tensor::<2, burn::tensor::Int>::from_data(
        burn::tensor::TensorData::new(
            (0..(BATCH * SEQ))
                .map(|i| (i % 251 + 1) as i64)
                .collect::<Vec<_>>(),
            [BATCH, SEQ],
        ),
        &adev(),
    );

    // Off: `moe_topk = 0` (the real off position) with the default
    // `moe_lb_coef = 0`. JEPA/DSpark are 0 in `cfg`, so the only possible term
    // is the balancer, and there is none.
    let mut off = c.clone();
    off.moe_topk = 0;
    off.moe_lb_coef = 0.0;
    let m_off = DormouseModel::new(&off, &adev());
    let (_l, _r, _k, aux_off) =
        m_off.forward_with_hidden::<B>(ids.clone(), None, None, Some(targets.clone()), None);
    assert!(
        aux_off.is_none(),
        "with moe_topk = 0 the objective carried an auxiliary term: {:?}",
        aux_off.map(|t| t.into_scalar::<f32>())
    );

    // On: a non-zero coefficient adds a finite, POSITIVE term - the balancer
    // is >= 1 by construction (`moe::lb_aux`'s range), so after weighting it is
    // a strictly positive addition to the loss and not a cancelling one.
    let mut on = c.clone();
    on.moe_lb_coef = 0.01;
    let m_on = DormouseModel::new(&on, &adev());
    let (_l, _r, _k, aux_on) = m_on.forward_with_hidden::<B>(ids, None, None, Some(targets), None);
    let t = aux_on
        .expect("a balancer with a selection must produce a term")
        .into_scalar::<f32>();
    assert!(
        t.is_finite() && t > 0.0,
        "the balancer must be a finite positive addition, got {t}"
    );
    assert!(
        t <= 0.01 * EXPERTS as f32 * (1.0 + 1e-5),
        "the balancer is bounded by coef * E = {}, got {t}",
        0.01 * EXPERTS as f32
    );
}
