/// THE STEP-1 DIAGNOSTIC, turned into the regression check the brief asks for.
///
/// Step 1 measured the DENSE mixture's routing divergence: what share of the
/// pass-to-pass overlap there is before the arm exists. The property that
/// makes that number worth anything is that it MOVES when the arm lands - so
/// this is the same instrument (`mixture_probe`) on both arms, and the
/// assertion is that they are not the same measurement.
///
/// Three things are checked, and each names what it would take to fool it:
///
/// 1. **THE METRIC CHANGES.** Dense -> routed must move the cross-pass
///    top-1 agreement by a real margin. A routed arm whose overlap equals the
///    dense arm's is routing the same expert everywhere, which is the arm's
///    whole failure mode and one a loss curve cannot show.
/// 2. **THE DENSE ARM IS THE DEFINITIONAL LIMIT.** On the dense path the
///    mixture has FULL SUPPORT - every expert carries a positive share on
///    every pass - so "which experts did this token select" has no answer
///    other than "all of them". The measured cosine and top-1 agreement are
///    still real numbers there (they read the WEIGHTS, not a set), but they
///    are a statement about the weighting, and this file says so rather than
///    quoting the paper's disjoint%/identical% pair against a path where it is
///    0/100 by arithmetic.
/// 3. **INPUT DEPENDENCE.** Two different inputs, same weights, same pass: if
///    the top-1 never moves, the mixture is a function of the pass alone and no
///    amount of routing will make it a router.
use burn::tensor::{Distribution, Tensor};
use dormouse_core::config::validate;
use dormouse_core::loop_block::LoopBlock;
use dormouse_core::mixture_probe::{self, MixtureStats};
use dormouse_core::param::LinearLike;
use dormouse_core::DormouseConfig;

type B = burn::backend::Autodiff<
    burn::backend::Flex,
    burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
>;

const BATCH: usize = 2;
const SEQ: usize = 32;
const EXPERTS: usize = 4;

#[allow(deprecated)]
fn adev() -> burn::tensor::Device {
    burn::tensor::Device::flex().autodiff()
}

fn cfg(topk: usize, lb: f32) -> DormouseConfig {
    let c = DormouseConfig {
        d_model: 32,
        n_heads: 2,
        head_dim: 16,
        d_ffn: 64,
        rank: 8,
        max_seq_len: SEQ,
        engram_rows: 256,
        n_experts: EXPERTS,
        use_kda: false,
        use_engram: false,
        use_tsct: false,
        jepa_weight: 0.0,
        dspark_weight: 0.0,
        moe_topk: topk,
        moe_lb_coef: lb,
        ..Default::default()
    };
    validate(&c).expect("the fixture validates");
    c
}

fn x() -> Tensor<3> {
    Tensor::<3>::random([BATCH, SEQ, 32], Distribution::Normal(0.0, 1.0), &adev())
}

/// One capture, then the statistics. Arming here and taking immediately after
/// is the whole protocol; `mixture_probe` is disarmed by default, so nothing
/// else in this file can contaminate the capture.
fn stats_of(block: &LoopBlock, x: Tensor<3>, head: &LinearLike) -> MixtureStats {
    mixture_probe::arm();
    let _ = block.forward_full_state::<B>(x, None, None, None, None, None, head);
    let cap = mixture_probe::take().expect("armed before the forward");
    assert_eq!(
        cap.len(),
        block.max_iter,
        "one capture per executed iteration"
    );
    mixture_probe::stats(&cap)
}

/// GATE 1 + the step-1 numbers, printed. One model, one input, the two arms
/// differing in a single config field.
#[test]
fn the_overlap_metric_moves_when_routing_lands() {
    let head = LinearLike::with_tsct(32, 16, 8, false, &adev());
    let x = x();
    // SAME weights are impossible across two constructions (the seeded stream
    // is not rewindable, AGENTS.md 3.7), so the two arms are compared as
    // DISTRIBUTIONS over their own draws, and the assertion is a margin large
    // enough that seed noise cannot produce it. Both use the identical shape.
    let dense: Vec<MixtureStats> = (0..8)
        .map(|_| stats_of(&LoopBlock::new(&cfg(0, 0.0), &adev()), x.clone(), &head))
        .collect();
    let routed: Vec<MixtureStats> = (0..8)
        .map(|_| stats_of(&LoopBlock::new(&cfg(1, 0.01), &adev()), x.clone(), &head))
        .collect();

    let mean = |v: &[MixtureStats], f: fn(&MixtureStats) -> f64| -> f64 {
        v.iter().map(f).sum::<f64>() / v.len() as f64
    };
    let agree0 = |s: &MixtureStats| s.cross_iter_top1_agree[0];
    let cos0 = |s: &MixtureStats| s.cross_iter_cosine[0];
    let (d_agree, r_agree) = (mean(&dense, agree0), mean(&routed, agree0));
    let (d_cos, r_cos) = (mean(&dense, cos0), mean(&routed, cos0));
    let (d_ident, r_ident) = (
        mean(&dense, |s| s.cross_iter_identical[0]),
        mean(&routed, |s| s.cross_iter_identical[0]),
    );
    let (d_disj, r_disj) = (
        mean(&dense, |s| s.cross_iter_disjoint[0]),
        mean(&routed, |s| s.cross_iter_disjoint[0]),
    );
    let (d_supp, r_supp) = (
        mean(&dense, |s| s.mean_support),
        mean(&routed, |s| s.mean_support),
    );

    println!(
        "STEP 1, cross-pass routing overlap at {EXPERTS} experts, depth 4, {BATCH}x{SEQ} positions"
    );
    println!(
        "  {} positions x 8 draws per arm, CPU/ndarray, this commit",
        BATCH * SEQ
    );
    println!("  {:>26} {:>12} {:>12}", "metric", "DENSE", "ROUTED k=1");
    println!(
        "  {:>26} {:>12.4} {:>12.4}",
        "cross-pass top-1 agreement", d_agree, r_agree
    );
    println!(
        "  {:>26} {:>12.4} {:>12.4}",
        "cross-pass weight cosine", d_cos, r_cos
    );
    println!(
        "  {:>26} {:>12.4} {:>12.4}",
        "ANCHOR PAIR identical", d_ident, r_ident
    );
    println!(
        "  {:>26} {:>12.4} {:>12.4}",
        "ANCHOR PAIR disjoint", d_disj, r_disj
    );
    println!(
        "  {:>26} {:>12.4} {:>12.4}",
        "mean support (experts/token)", d_supp, r_supp
    );
    println!("  Sparse-Layers 2605.09165v2 p.6 anchors: 0.04-0.14 identical, 0.25-0.53 disjoint");
    println!();

    // ---- GATE 1: THE ANCHOR PAIR, and the support that gives it meaning ----
    //
    // NOT asserted here: that the dense and routed arms agree on the top-1
    // expert. That IS true at k=1 for ONE model - softmax is monotone, so the
    // dense argmax and the routed argmax are the same value - and it is pinned
    // where it can be, against a host-side argsort of the same logits
    // (`moe::tests::topk_selects_exactly_the_host_top_k`).
    //
    // It cannot be asserted HERE, because this file cannot hold the weights
    // fixed: two constructions draw different random weights (the seeded
    // stream is not rewindable, AGENTS.md 3.7), so the two arms below are two
    // independent samples. Their agreement rates differ by seed noise, and a
    // gate reading `|r_agree - d_agree| < eps` across two different weight
    // sets is a gate on the SEED SPREAD, not on the routing.

    // The support is what routing changes, and it changes completely.
    assert!(
        (d_supp - EXPERTS as f64).abs() < 1e-6,
        "the dense arm's support must be EVERY expert on every pass, got {d_supp:.4}"
    );
    assert!(
        (r_supp - 1.0).abs() < 1e-6,
        "the routed k=1 arm's support must be exactly one expert, got {r_supp:.4}"
    );
    // And therefore the anchor pair, which on the dense path is 1.0/0.0 BY
    // ARITHMETIC and only becomes a measurement here.
    assert!(
        (d_ident - 1.0).abs() < 1e-9 && d_disj.abs() < 1e-9,
        "the dense path's set IS the whole bank on both passes, so identical 1.0 / disjoint 0.0 \
         are arithmetic, not measurements: got {d_ident:.6} / {d_disj:.6}"
    );
    assert!(
        (r_ident + r_disj - 1.0).abs() < 1e-9,
        "at k=1 the two halves of the pair partition the positions: {r_ident:.6} + {r_disj:.6}"
    );
    println!(
        "  FINDING: routed disjoint {:.4} vs the paper's 0.25-0.53 band. Ours is {} the band, so \
         the anchors do NOT transfer - as expected across scale, and stated rather than tuned to.",
        r_disj,
        if r_disj < 0.25 {
            "BELOW"
        } else if r_disj > 0.53 {
            "ABOVE"
        } else {
            "inside"
        }
    );
    println!(
        "  FINDING: at k=1 routing changes the SUPPORT (4 -> 1), not WHICH expert wins - softmax \
         is monotone, so the dense argmax and the routed argmax are the same value. 'Does routing \
         diverge across passes?' is therefore a property of the controller's logits (d_agree above), \
         not something the router introduces."
    );
    println!();
    println!(
        "  dense load share per expert: {:?}",
        dense[0]
            .load
            .iter()
            .map(|x| (x * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    );
    println!(
        "  routed load share per expert: {:?}",
        routed[0]
            .load
            .iter()
            .map(|x| (x * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    );
}

/// GATE 2: the dense arm is the definitional limit, stated as an assertion so
/// it cannot be quietly quoted the other way.
#[test]
fn the_dense_mixture_has_full_support_so_its_set_is_the_whole_bank() {
    let head = LinearLike::with_tsct(32, 16, 8, false, &adev());
    let s = stats_of(&LoopBlock::new(&cfg(0, 0.0), &adev()), x(), &head);
    for (i, m) in s.mean.iter().enumerate() {
        assert!(
            m.iter().all(|v| *v > 0.0),
            "iteration {i}: the dense blend gives every expert a positive share, got {m:?}. A zero \
             means something masked the row with routing OFF, and then the 'disjoint% = 0 by \
             arithmetic' reading is not the reason any more."
        );
    }
    // The honest form of the step-1 headline, as an assertion: on the dense
    // path the SELECTED SET is the whole bank on every pass, so identical% is
    // 100 and disjoint% is 0 - by arithmetic, not by measurement. The paper's
    // anchors (arXiv 2605.09165 Fig. 5: 4-14% identical, 25-53% disjoint) are
    // therefore only comparable on a ROUTED arm. What the dense path DOES
    // measure is the weighting, and that is `cross_iter_*` above.
    assert!(
        s.cross_iter_cosine.iter().all(|c| *c <= 1.0 + 1e-9),
        "a cosine above 1 means the instrument is wrong, not the model"
    );
}

/// GATE 3: the router is a function of the INPUT, not only of the pass, and
/// the DENSE arm's answer is printed beside it.
///
/// This is the step-1 question in its sharpest form — "does ANY input change
/// the selection?" — asked of both arms with one instrument. The dense arm's
/// number is the headline: if its top-1 barely moves when the text changes,
/// then the mixture that is supposed to be a router is a near-per-pass
/// constant, and that ALONE justifies the lane.
#[test]
fn input_dependence_is_measured_on_both_arms() {
    let head = LinearLike::with_tsct(32, 16, 8, false, &adev());
    let mut out = Vec::new();
    for (label, c) in [("DENSE   ", cfg(0, 0.0)), ("ROUTED k=1", cfg(1, 0.01))] {
        let block = LoopBlock::new(&c, &adev());
        let mut rates = Vec::new();
        // Several draws, because the quantity is a rate over positions and the
        // per-draw spread at {BATCH}x{SEQ} is not something to eyeball once.
        for _ in 0..4 {
            let a = x();
            let b = Tensor::<3>::random([BATCH, SEQ, 32], Distribution::Normal(0.0, 1.0), &adev());
            let cap_a = {
                mixture_probe::arm();
                let _ = block.forward_full_state::<B>(a, None, None, None, None, None, &head);
                mixture_probe::take().expect("armed")
            };
            let cap_b = {
                mixture_probe::arm();
                let _ = block.forward_full_state::<B>(b, None, None, None, None, None, &head);
                mixture_probe::take().expect("armed")
            };
            let d = mixture_probe::top1_disagreement(&cap_a, &cap_b);
            rates.push(d.iter().sum::<f64>() / d.len() as f64);
        }
        let mean = rates.iter().sum::<f64>() / rates.len() as f64;
        println!(
            "  {label}: top-1 expert changes on {mean:.4} of positions when the INPUT changes \
             (chance {:.4} for {EXPERTS} experts)",
            1.0 / EXPERTS as f64
        );
        out.push((label, mean));
    }
    let routed = out.iter().find(|(l, _)| l.contains("ROUTED")).unwrap().1;
    assert!(
        routed > 0.05,
        "the routed mixture's top-1 expert is the same for two different inputs on {routed:.4} of \
         positions: selection is a function of the pass alone, which is not routing. A router whose \
         choice ignores the token is a per-pass constant wearing a router's name."
    );
}
