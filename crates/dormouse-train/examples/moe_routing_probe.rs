//! STEP 1 DIAGNOSTIC: does the expert mixture route differently per pass?
//!
//! The claim under test is arXiv 2605.09165 §6.1's, transplanted to this
//! loop: in a weight-tied model the expressiveness has to come from the
//! SELECTION changing, and their measurement is a pass-to-pass overlap
//! count on the expert sets (4-14% identical, 25-53% disjoint, k=2 of E=8).
//! Our FFN is already a mixture of `n_experts` TSCT experts; the question is
//! whether its mixture is a router in that sense or a fixed average.
//!
//! This is a PROGRAM, not a test: the numbers come from a model whose
//! weights have moved, and a trained model is a moving target (the precedent
//! is `examples/export_divergence.rs`). What it prints is exactly what
//! `crates/dormouse-core/src/mixture_probe.rs` computes, so the number in
//! `docs/reviews/moe-routing-2026-10-01.md` and the number the routing
//! arm's gate later compares against come from one instrument.
//!
//! CPU/ndarray on purpose: it must be able to run beside a GPU training run
//! (AGENTS.md 1.5), and the question is about the mixture, not the backend.
//!
//! ```sh
//! cargo run --release -p dormouse-train --example moe_routing_probe -- \
//!   --preset small --steps 60 --probe-steps 0,10,30,60
//! ```

use std::path::PathBuf;

use burn::module::Module;
use burn::tensor::{Device, Int, Tensor, TensorData};

use dormouse_core::mixture_probe::{self, MixtureStats};
use dormouse_core::{DormouseConfig, DormouseModel};

#[derive(clap::Parser, Debug)]
struct Args {
    /// Preset whose execution fields (arms, depth, n_experts, weights) the
    /// fixture keeps. Only the WIDTHS shrink - see `fixture()`.
    #[arg(long, default_value = "small")]
    preset: String,
    #[arg(long, default_value = "60")]
    steps: usize,
    /// Steps at which to print the mixture table. `0` is the untrained
    /// model, which is the number that says whether the arm is needed AT ALL.
    #[arg(long, default_value = "0,10,30,60", value_delimiter = ',')]
    probe_steps: Vec<usize>,
    #[arg(long, default_value = "2")]
    batch: usize,
    #[arg(long, default_value = "256")]
    seq_len: usize,
    #[arg(long, default_value = "1e-3")]
    lr: f64,
    /// Real bytes. Defaults to this repo's own `AGENTS.md` - real English
    /// prose, ~90 KB, in-tree and drive-independent. Point it at the corpus
    /// for a corpus-shaped distribution.
    #[arg(long)]
    corpus: Option<PathBuf>,
    /// Selected experts per token per pass. **0 = the fixed mixture (the
    /// control and the default); 1 = the FIRST routed configuration; 2 = the
    /// second.** The same program measures all three with one instrument -
    /// comparing two numbers produced by two different programs is how a gate
    /// ends up comparing two different networks.
    #[arg(long, default_value = "0")]
    moe_topk: usize,
    /// Expert bank size. The corrected design's first configuration is 4
    /// experts at top-1 (the 8-expert top-2 evidence sits at 168M+ active
    /// params and `small` is 9.2M), so 4 is the default here too.
    #[arg(long, default_value = "4")]
    n_experts: usize,
}

/// The preset with its WIDTHS shrunk, the same rule `tests/preset_exec.rs`
/// uses and for the same reason: a `small` forward is minutes on this
/// backend, and nothing here is decided by a width. Everything that decides
/// what executes is the preset's own value and is printed below.
fn fixture(c: &DormouseConfig, n_experts: usize, moe_topk: usize) -> DormouseConfig {
    DormouseConfig {
        d_model: 192,
        n_heads: 6,
        head_dim: 32,
        d_ffn: 384,
        rank: 32,
        max_seq_len: 256,
        engram_rows: 4096,
        n_experts,
        moe_topk,
        ..c.clone()
    }
}

/// One batch: bytes, the one-byte-shifted labels, and the FNV n-gram keys.
///
/// The keys are `dormouse_data::raw_keys` - the data crate's own derivation,
/// the same one the trainer and the decode seam use. Not a hand-rolled third
/// FNV: that mistake is written up in `export_divergence.rs`.
fn batch(bytes: &[u8], dev: &Device) -> (Tensor<2, Int>, Tensor<3, Int>, Tensor<2, Int>) {
    let (b, t) = (1usize, bytes.len());
    let ids: Vec<i64> = bytes.iter().map(|&x| x as i64).collect();
    let y: Vec<i64> = bytes
        .iter()
        .skip(1)
        .map(|&x| x as i64)
        .chain(std::iter::once(bytes[0] as i64))
        .collect();
    (
        Tensor::from_data(TensorData::new(ids, [b, t]), dev),
        Tensor::from_data(
            TensorData::new(dormouse_data::raw_keys(bytes), [b, t, 3]),
            dev,
        ),
        Tensor::from_data(TensorData::new(y, [b, t]), dev),
    )
}

fn read_corpus(a: &Args) -> Vec<u8> {
    let path = a
        .corpus
        .clone()
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../AGENTS.md"));
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("corpus {} unreadable: {e}", path.display()));
    assert!(
        bytes.len() >= a.batch * a.seq_len * 4,
        "corpus {} is {} B; need >= {} for batch {} x seq {} x 4",
        path.display(),
        bytes.len(),
        a.batch * a.seq_len * 4,
        a.batch,
        a.seq_len
    );
    println!(
        "corpus       {} ({} B of real text)",
        path.display(),
        bytes.len()
    );
    bytes
}

/// One capture: arm, forward, take. The capture is the mixture the FFN
/// branch actually used, per executed iteration.
fn capture(
    m: &DormouseModel,
    b: &(Tensor<2, Int>, Tensor<3, Int>, Tensor<2, Int>),
) -> Vec<Tensor<2>> {
    mixture_probe::arm();
    let (_logits, _rec, _kda, _aux) = m.forward_with_hidden::<dormouse_train::Backend>(
        b.0.clone(),
        Some(b.1.clone()),
        None,
        Some(b.2.clone()),
        None,
    );
    mixture_probe::take().expect("armed before the forward; the seam disarms on take")
}

fn print_stats(s: &MixtureStats, label: &str, e: usize) {
    println!("--- mixture @ {label}");
    for (i, m) in s.mean.iter().enumerate() {
        let w: Vec<String> = m.iter().map(|x| format!("{x:.3}")).collect();
        println!("  iter {i}  mean[{}]  (uniform {:.3})", w.join(", "), 1.0 / e as f64);
    }
    if s.cross_iter_cosine.is_empty() {
        println!("  depth 1: no cross-iteration pair to compare");
    }
    for (i, (c, a)) in s
        .cross_iter_cosine
        .iter()
        .zip(&s.cross_iter_top1_agree)
        .enumerate()
    {
        println!(
            "  iter {} vs 0:  mean-weight cosine {c:.4}   top-1 agreement {a:.4}   \
             (chance {:.4})",
            i + 1,
            1.0 / e as f64
        );
    }
    // THE ANCHOR PAIR. Sparse-Layers (arXiv 2605.09165v2 p.6) reports 25-53%
    // of tokens getting a FULLY DISJOINT expert pair between two loop passes
    // and only 4-14% an identical one. These two numbers ARE the comparison,
    // and `support` says which configuration produced them: on the fixed
    // mixture the support is every expert, so disjoint is 0 and identical is 1
    // BY ARITHMETIC (findings file §4) - the pair is only meaningful on a
    // routed arm, and this line is what tells the two apart.
    for (i, (id, dj)) in s
        .cross_iter_identical
        .iter()
        .zip(&s.cross_iter_disjoint)
        .enumerate()
    {
        println!(
            "  iter {} vs 0:  ANCHOR PAIR  identical {id:.4}  disjoint {dj:.4}   \
             (Sparse-Layers 2605.09165v2 p.6: 0.04-0.14 identical, 0.25-0.53 disjoint)",
            i + 1
        );
    }
    println!(
        "  support  {:.3} experts per token  ({e} = dense: every expert runs on every pass; \
         below {e} on a dense run is softmax underflow, not routing)",
        s.mean_support
    );
    let load: Vec<String> = s.load.iter().map(|x| format!("{:.4}", x)).collect();
    println!(
        "  load     [{}]  effective experts {:.3} of {e}  (1.000 = one expert takes everything, {e} = uniform)",
        load.join(", "),
        s.effective_experts
    );
}

fn main() {
    let a = <Args as clap::Parser>::parse();
    let preset = dormouse_core::config::load_config(&a.preset).expect("the preset loads");
    let cfg = fixture(&preset, a.n_experts, a.moe_topk);
    dormouse_core::config::validate(&cfg).expect("the fixture validates");
    println!(
        "preset       {} -> fixture: d_model {} d_ffn {} rank {} | arms kda={} engram={} tsct={} \
         gr={} attnres={} mor={} | depth {} | n_experts {} | moe_topk {} | vocab {}",
        a.preset,
        cfg.d_model,
        cfg.d_ffn,
        cfg.rank,
        cfg.use_kda,
        cfg.use_engram,
        cfg.use_tsct,
        cfg.use_gr,
        cfg.use_attnres,
        cfg.use_mor,
        cfg.max_iter,
        cfg.n_experts,
        cfg.moe_topk,
        cfg.vocab
    );
    println!(
        "CONFIGURATION  {}",
        if cfg.moe_topk == 0 {
            "FIXED MIXTURE (the control): every expert runs on every pass, so the selected set is \
             the whole bank and identical/disjoint read 1.0/0.0 by arithmetic"
        } else {
            "ROUTED: top-k selection per (position, pass) - this is where the anchor pair bites"
        }
    );
    println!(
        "PARAM COUNT  width-only shrink: the shape fields are the PRESET's, so the mixture is \
         the one a `{}` run routes.",
        a.preset
    );

    let corpus = read_corpus(&a);
    let chunk = a.batch * a.seq_len;
    let dev = dormouse_train::device();
    let mut model = DormouseModel::new(&cfg, &dev);
    let mut optim = dormouse_train::build_optim(&model, &dormouse_train::TrainCfg::default());
    let e = cfg.n_experts;

    // The window the probes use is FIXED, so every table below is about the
    // same bytes and the step index is the only thing that moves.
    let probe_bytes = &corpus[0..chunk];
    let probe_batch = batch(probe_bytes, &dev);
    // A second, DISJOINT window: the input-dependence half of the question.
    let other_off = corpus.len() / 2 - chunk / 2;
    let other_batch = batch(&corpus[other_off..other_off + chunk], &dev);

    let mut probes = a.probe_steps.clone();
    probes.sort_unstable();
    probes.dedup();

    for step in 0..=a.steps {
        if probes.contains(&step) {
            let cap = capture(&model, &probe_batch);
            assert_eq!(
                cap.len(),
                cfg.max_iter,
                "one captured mixture per EXECUTED iteration, and nothing else captured \
                 (a doubled capture here would mean a second forward ran under the arm)"
            );
            let s = mixture_probe::stats(&cap);
            print_stats(&s, &format!("step {step}"), e);

            if step == 0 {
                // INPUT DEPENDENCE. Same weights, same iteration, different
                // text: if the top-1 expert does not move, the mixture is a
                // function of the pass alone and no amount of routing will
                // make it a router.
                let other = capture(&model, &other_batch);
                let d = mixture_probe::top1_disagreement(&cap, &other);
                let mean_d: f64 = d.iter().sum::<f64>() / d.len() as f64;
                println!(
                    "  INPUT DEPENDENCE: top-1 changes on {:.4} of positions when the TEXT changes \
                     (per iteration {:?}); {:.4} is the chance rate for {e} experts",
                    mean_d,
                    d.iter().map(|x| (x * 10000.0).round() / 100.0).collect::<Vec<_>>(),
                    1.0 / e as f64
                );
                if mean_d <= 1.0 / e as f64 * 1.05 {
                    println!(
                        "  VERDICT: the mixture is at or below CHANCE input dependence - selection \
                         is effectively input-independent. That is the number that justifies the \
                         routing arm on its own."
                    );
                }
            }
            println!();
        }
        if step == a.steps {
            break;
        }
        let off = (step * chunk) % corpus.len().saturating_sub(chunk + 1);
        let b = batch(&corpus[off..off + chunk], &dev);
        let (_logits, rec, _kda, _aux) = model.forward_with_hidden::<dormouse_train::Backend>(
            b.0,
            Some(b.1),
            None,
            Some(b.2),
            None,
        );
        let loss = model.loss::<dormouse_train::Backend>(rec);
        let l: f32 = loss
            .clone()
            .into_data()
            .try_to_vec()
            .expect("loss readback")[0];
        let grads = burn::optim::GradientsParams::from_grads(loss.backward(), &model);
        model.retract_tsct(3);
        model = optim.step(a.lr, model, grads);
        assert!(l.is_finite(), "step {step}: loss {l} is not finite");
        // Leave the sink disarmed between steps: the step forward above is
        // NOT a measurement, and a capture left armed would silently grow.
        assert!(
            !mixture_probe::armed(),
            "the step forward must not record a mixture"
        );
    }
    println!("done. The mixture tables above are the instrument; the gate that reuses it is");
    println!("`crates/dormouse-core/tests/moe_routing_seam.rs`.");
    let _ = Module::num_params(&model);
}
