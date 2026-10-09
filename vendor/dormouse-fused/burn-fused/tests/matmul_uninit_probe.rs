#![cfg(all(feature = "cuda", feature = "autodiff"))]
//! ONE matmul, six shapes, under `compute-sanitizer --tool initcheck`.
//!
//! ## Why this file exists
//!
//! The 0.22 fp32 b10 s512 recipe corrupts its own forward: at step 1 the
//! loss reads 1e8..1e16 while every parameter is bit-identical to step 0
//! (proved at `--lr 0`, where the update is a no-op), and the held-out eval
//! on those same weights reads NaN. compute-sanitizer says the KDA ops path
//! reads never-written global memory inside `matmul_entry` / a cast kernel,
//! and the persistent pool is what supplies huge values there (measured:
//! pool enabled/size-match 10/10 explosions, disabled 1/3, enforced 0/1, all
//! with a zero step-time difference). `memcheck` reports no out-of-bounds
//! access, so nothing is being written out of range - a READ is landing on
//! memory no kernel ever wrote.
//!
//! That leaves exactly two candidates, and this test separates them:
//!
//! 1. **The kernel reads a tensor's row tail past its writer** (an upstream
//!    cubek-matmul 0.3.0 defect). Then a plain matmul on an unaligned inner
//!    dimension reproduces here with no model code at all.
//! 2. **One of our tensors has a hole** (a producer that writes only part of
//!    its buffer). Then this file stays clean and the hunt continues in our
//!    ops with the address this sanitizer prints.
//!
//! The shapes are chosen to cover both: aligned (64/64/64, 512/64/256) and
//! deliberately unaligned inner and outer dims (6, 62, 3/5), because a
//! vectorized load of width 2 or 4 over a row that is not a multiple of that
//! width is the only mechanism by which a matmul can read unwritten bytes
//! without writing any.
//!
//! Run it:
//! `compute-sanitizer --tool initcheck --launch-timeout 120 \
//!    cargo test -p burn-fused --release --features cuda,autodiff \
//!    --test matmul_uninit_probe -- --nocapture`
//! The assertion inside is only "finite"; the verdict is the sanitizer's
//! event count, printed by the harness, and a non-zero count is a FAILURE of
//! the upstream kernel, not of this test.

use burn::tensor::{Device, Distribution, Tensor};

fn dev() -> Device {
    Device::cuda(0)
}

#[test]
fn one_matmul_per_shape_reads_only_what_was_written() {
    let d = dev();
    let shapes: [(usize, usize, usize); 6] = [
        (64, 64, 64),   // aligned baseline
        (64, 6, 64),    // inner dim not a multiple of the vec width
        (64, 62, 64),   // inner dim 2 mod 4
        (7, 3, 5),      // everything unaligned
        (512, 64, 256), // the model's own [b*t, d] x [d, f] at b10 s512
        (10, 512, 64),  // the shape whose transpose appears in the WY view
    ];
    for (m, k, n) in shapes {
        let a = Tensor::<2>::random([m, k], Distribution::Normal(0.0, 1.0), &d);
        let b = Tensor::<2>::random([k, n], Distribution::Normal(0.0, 1.0), &d);
        let c = a.matmul(b);
        let v: Vec<f32> = c.into_data().try_to_vec().unwrap();
        assert_eq!(v.len(), m * n, "shape {m}x{k}x{n}: wrong element count");
        assert!(
            v.iter().all(|x| x.is_finite()),
            "shape {m}x{k}x{n}: non-finite product"
        );
        println!("matmul {m}x{k}x{n}: {} elements, all finite", v.len());
    }
}

// ── level 2: the module around the chunk op ──────────────────────────────
//
// The bare chunk op is clean at the production shape (the green
// gpu_production_shape gate, and the six shapes above), but
// `burn-kda --test ops_grad_cuda` and `--example kda_step_probe` both read
// never-written memory under this sanitizer. What they add over the bare op
// is exactly `KdaModule`: the projections, the decay, the RMS-gated output.
// This is that module at the same small shape those two use, so the event
// count localises the hole to the module rather than to the chunk algebra.
#[test]
fn kda_module_forward_reads_only_what_was_written() {
    use burn_fused::burn_kda::{DecayFn, KdaConfig, KdaModule};
    let d = dev();
    let c = KdaConfig {
        hidden_size: 64,
        num_heads: 2,
        head_dim: 16,
        num_v_heads: Some(2),
        use_short_conv: false,
        decay_fn: DecayFn::Sigmoid,
        chunk_size: 16,
        ..Default::default()
    };
    let km = KdaModule::new(&c, 0.9, &d);
    let x = Tensor::<3>::random([1, 64, c.hidden_size], Distribution::Normal(0.0, 1.0), &d);
    let (y, s) = km.forward_train_state::<burn_cuda::Cuda>(x.clone(), None);
    let yv: Vec<f32> = y.into_data().try_to_vec().unwrap();
    let sv: Vec<f32> = s.into_data().try_to_vec().unwrap();
    assert!(yv.iter().all(|x| x.is_finite()), "non-finite module output");
    assert!(sv.iter().all(|x| x.is_finite()), "non-finite module state");
    println!(
        "kda module: out {} elems, state {} elems, all finite",
        yv.len(),
        sv.len()
    );
}

// ── level 3: which half of the module ────────────────────────────────────
#[test]
fn kda_projections_only_read_only_what_was_written() {
    use burn_fused::burn_kda::{DecayFn, KdaConfig, KdaModule};
    let d = dev();
    let c = KdaConfig {
        hidden_size: 64,
        num_heads: 2,
        head_dim: 16,
        num_v_heads: Some(2),
        use_short_conv: false,
        decay_fn: DecayFn::Sigmoid,
        chunk_size: 16,
        ..Default::default()
    };
    let km = KdaModule::new(&c, 0.9, &d);
    let x = Tensor::<3>::random([1, 64, c.hidden_size], Distribution::Normal(0.0, 1.0), &d);
    let (q, k, v, g, bk, bv, gate) = km.project_for_test(x);
    for (name, t) in [("q", &q), ("k", &k), ("v", &v), ("g", &g), ("bk", &bk), ("bv", &bv), ("gate", &gate)] {
        let vals: Vec<f32> = t.clone().into_data().try_to_vec().unwrap();
        assert!(vals.iter().all(|x| x.is_finite()), "{name}: non-finite");
        println!("kda proj {name}: {:?} {} elems", t.dims(), vals.len());
    }
}

// ── level 4: the Linear pattern ──────────────────────────────────────────
//
// burn's `nn::Linear` computes `x @ Wᵀ`: the second operand arrives as a
// TRANSPOSED view, not a contiguous matrix. The module's `beta_proj` is the
// narrowest one it owns - out_features = n_heads = 2 here, which is the same
// value the sanitizer names (`rhs_f32_rhs_size_2`). If a transposed rhs is
// what the vectorized loader mishandles, this reproduces with no model code.
#[test]
fn linear_pattern_reads_only_what_was_written() {
    let d = dev();
    for out_features in [2usize, 4, 8, 32, 64] {
        let x = Tensor::<2>::random([64, 64], Distribution::Normal(0.0, 1.0), &d);
        let w = Tensor::<2>::random([out_features, 64], Distribution::Normal(0.0, 1.0), &d);
        let y = x.clone().matmul(w.clone().transpose());
        let vals: Vec<f32> = y.into_data().try_to_vec().unwrap();
        assert_eq!(vals.len(), 64 * out_features);
        assert!(vals.iter().all(|v| v.is_finite()), "out={out_features}");
        // and the same comparison against a contiguous rhs, so a drift shows up
        let w_t = w.transpose().mul_scalar(1.0); // the repo's own materialise idiom
        let y2 = x.matmul(w_t);
        let v2: Vec<f32> = y2.into_data().try_to_vec().unwrap();
        let worst = vals
            .iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("linear out_features={out_features}: worst transposed-vs-contiguous {worst:.3e}");
        assert!(worst < 1e-5, "out_features={out_features}: {worst:.3e}");
    }
}

// ── level 5: the pieces of `project`, one test each ───────────────────────
#[test]
fn decay_only_reads_only_what_was_written() {
    use burn_fused::burn_kda::{DecayFn, KdaDecay};
    let d = dev();
    let dec = KdaDecay::new(64, 2, 16, 16, -5.0, DecayFn::Sigmoid, &d);
    let x = Tensor::<3>::random([1, 64, 64], Distribution::Normal(0.0, 1.0), &d);
    let a = dec.forward(x);
    let vals: Vec<f32> = a.into_data().try_to_vec().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()), "decay produced non-finite alpha");
    println!("decay: {} elems, all finite", vals.len());
}

#[test]
fn norm_only_reads_only_what_was_written() {
    use burn_fused::burn_gdn2::l2_normalize;
    let d = dev();
    let _ = &d;
    let x = Tensor::<3>::random([1, 64, 32], Distribution::Normal(0.0, 1.0), &d);
    let n = l2_normalize(x, 1e-6);
    let vals: Vec<f32> = n.into_data().try_to_vec().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()), "l2 normalized to non-finite");
    println!("l2_normalize: {} elems, all finite", vals.len());
}

#[test]
fn narrow_linear_sigmoid_reads_only_what_was_written() {
    use burn::nn::{Linear, LinearConfig};
    use burn::tensor::activation;
    let d = dev();
    // the beta_proj pattern: [B, T, D] -> [B, T, H] with H = 2, then sigmoid
    let lin: Linear = LinearConfig::new(64, 2).init(&d);
    let x = Tensor::<3>::random([1, 64, 64], Distribution::Normal(0.0, 1.0), &d);
    let z = activation::sigmoid(lin.forward(x));
    let vals: Vec<f32> = z.into_data().try_to_vec().unwrap();
    assert!(vals.iter().all(|v| v.is_finite() && *v > 0.0 && *v < 1.0));
    println!("narrow linear + sigmoid: {} elems in (0,1)", vals.len());
}

// ── level 6: split the narrow-linear repro into its parts ────────────────
fn narrow_linear_variant(out_features: usize, bias: bool, act: bool, batch: usize) {
    use burn::nn::{Linear, LinearConfig};
    use burn::tensor::activation;
    let d = dev();
    let cfg = LinearConfig::new(64, out_features);
    let lin: Linear = if bias { cfg.init(&d) } else { cfg.with_bias(false).init(&d) };
    let x = Tensor::<3>::random([batch, 64, 64], Distribution::Normal(0.0, 1.0), &d);
    let z = lin.forward(x);
    let z = if act { activation::sigmoid(z) } else { z.mul_scalar(1.0) };
    let vals: Vec<f32> = z.into_data().try_to_vec().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()));
    println!("narrow out={out_features} bias={bias} act={act} batch={batch}: {} elems", vals.len());
}

#[test] fn v_bias_only()      { narrow_linear_variant(2, true, false, 1); }
#[test] fn v_act_only()       { narrow_linear_variant(2, false, true, 1); }
#[test] fn v_bias_and_act()   { narrow_linear_variant(2, true, true, 1); }
#[test] fn v_wide()           { narrow_linear_variant(64, true, true, 1); }
#[test] fn v_batch8()         { narrow_linear_variant(2, true, true, 8); }
#[test] fn v_batch10()        { narrow_linear_variant(2, true, true, 10); }

// ── level 7: does the uninitialized VALUE reach the result? ──────────────
//
// The sanitizer says the narrow matmul READS memory no kernel wrote. A read
// is only a defect if the value arrives in the output: the lanes may be
// masked in the epilogue and the read is then pure noise (still an upstream
// defect, but not the training corruption). So compute the same product
// twice - once through the narrow path, once with the rhs MATERIALISED into
// a contiguous buffer - and compare bitwise. A difference means the garbage
// is in the numbers; equality means it is masked and the corruption has
// another cause.
#[test]
fn narrow_matmul_value_is_masked_or_not() {
    use burn::tensor::activation;
    let d = dev();
    for out_features in [2usize, 3, 4] {
        let x = Tensor::<3>::random([10, 64, 64], Distribution::Normal(0.0, 1.0), &d);
        let w = Tensor::<2>::random([out_features, 64], Distribution::Normal(0.0, 1.0), &d);
        let b = Tensor::<1>::random([out_features], Distribution::Normal(0.0, 1.0), &d);

        let narrow = activation::sigmoid(
            x.clone()
                .reshape([640, 64])
                .matmul(w.clone().transpose())
                .add(b.clone().reshape([1, out_features])),
        )
        .reshape([10, 64, out_features]);

        let reference = activation::sigmoid(
            x.reshape([640, 64])
                .matmul(w.transpose().mul_scalar(1.0))
                .add(b.reshape([1, out_features])),
        )
        .reshape([10, 64, out_features]);

        let a: Vec<f32> = narrow.into_data().try_to_vec().unwrap();
        let c: Vec<f32> = reference.into_data().try_to_vec().unwrap();
        let worst = a
            .iter()
            .zip(c.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let scale = c.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
        println!(
            "narrow out_features={out_features}: worst |narrow - contiguous| = {worst:.3e} \
             (scale {scale:.3e}, rel {:.3e})",
            worst / scale
        );
        assert!(
            worst / scale < 1e-6,
            "out_features={out_features}: the uninitialized read reaches the result ({worst:.3e})"
        );
    }
}

// ── level 8: is the module's forward deterministic in ONE process? ───────
//
// Two runs of the same forward on the same weights and the same input. The
// per-run randomness of the corruption cannot show up here (one process, one
// autotune cache); a difference would be an in-process nondeterminism (a
// reduction order, a race the sanitizer missed). The second half adds the
// allocation churn of a full backward + a Muon-style update, which is what
// the trainer does between its steps.
#[test]
fn module_forward_is_deterministic_within_one_process() {
    use burn_fused::burn_kda::{DecayFn, KdaConfig, KdaModule};
    let d = dev();
    let c = KdaConfig {
        hidden_size: 64,
        num_heads: 2,
        head_dim: 16,
        num_v_heads: Some(2),
        use_short_conv: false,
        decay_fn: DecayFn::Sigmoid,
        chunk_size: 16,
        ..Default::default()
    };
    // One module, one device (the autodiff one), because KdaModule carries no
    // backend parameter in 0.22: `forward_train_state` is generic over B.
    let d2 = dev().autodiff();
    let km = KdaModule::new(&c, 0.9, &d2);
    let x = Tensor::<3>::random([1, 64, c.hidden_size], Distribution::Normal(0.0, 1.0), &d2);
    let (y1, s1) = km.forward_train_state::<burn::backend::Autodiff<burn_cuda::Cuda>>(x.clone(), None);
    let (y2, s2) = km.forward_train_state::<burn::backend::Autodiff<burn_cuda::Cuda>>(x.clone(), None);
    let a: Vec<f32> = y1.into_data().try_to_vec().unwrap();
    let b: Vec<f32> = y2.into_data().try_to_vec().unwrap();
    let c1: Vec<f32> = s1.into_data().try_to_vec().unwrap();
    let c2: Vec<f32> = s2.into_data().try_to_vec().unwrap();
    assert_eq!(a, b, "two forwards of the same graph input differ: in-process nondeterminism");
    assert_eq!(c1, c2, "two forwards' state differs: in-process nondeterminism");

    // Now the trainer's shape of thing: one forward+backward, then a forward.
    let xa = Tensor::<3>::random([1, 64, c.hidden_size], Distribution::Normal(0.0, 1.0), &d2).require_grad();
    let (ya, sa) = km.forward_train_state::<burn::backend::Autodiff<burn_cuda::Cuda>>(xa.clone(), None);
    let loss = ya.clone().powf_scalar(2.0).sum().add(sa.powf_scalar(2.0).sum());
    let _g = loss.backward();
    let (yb, sb) = km.forward_train_state::<burn::backend::Autodiff<burn_cuda::Cuda>>(xa.clone(), None);
    let pa: Vec<f32> = yb.into_data().try_to_vec().unwrap();
    let pb: Vec<f32> = sb.into_data().try_to_vec().unwrap();
    println!(
        "post-backward forward: {} + {} elems, all finite = {}",
        pa.len(),
        pb.len(),
        pa.iter().chain(pb.iter()).all(|v| v.is_finite())
    );
    let first: Vec<f32> = ya.into_data().try_to_vec().unwrap();
    assert_eq!(first, pa, "a forward after a backward differs from the forward before it");
}

// ── level 9: is one candidate WRONG, not just different? ─────────────────
//
// The shape the autotuner re-picks between runs: m=64, n=2048, k=64 (the
// TSCT expert's second factor matmul, [*, rank] x [rank, d_ffn]). This prints
// a checksum so a driver can correlate it with the candidate the autotuner
// chose (the decisions sink names it). A checksum that differs from the
// others by more than f32 reassociation (~1e-6) is a kernel computing the
// wrong product, and that is the defect.
#[test]
fn fixed_shape_checksum() {
    let d = dev();
    d.seed(42); // same inputs in every process, so a different checksum is the kernel's
    let a = Tensor::<2>::random([64, 64], Distribution::Normal(0.0, 1.0), &d);
    let b = Tensor::<2>::random([64, 2048], Distribution::Normal(0.0, 1.0), &d);
    let c = a.matmul(b);
    let v: Vec<f32> = c.into_data().try_to_vec().unwrap();
    let sum: f64 = v.iter().map(|&x| x as f64).sum();
    let abs: f64 = v.iter().map(|&x| (x as f64).abs()).sum();
    let bits: u64 = v
        .iter()
        .fold(0u64, |h, x| h.wrapping_mul(31).wrapping_add(x.to_bits() as u64));
    println!(
        "CHECKSUM m64_n2048_k64: n={} sum={sum:.9e} abs={abs:.9e} bits={bits:016x}",
        v.len()
    );
}

// ── level 10: the value-level proof, on the trainer's device ─────────────
//
// Everything above is a correlation (4 exploding runs vs 2 clean). This is
// the direct form: the same two shapes the trainer tunes, on the AUTOdiff
// device the trainer uses, seeded, so every process computes the same
// product. A different checksum between processes is a candidate computing a
// different answer, and the decisions sink names which candidate won.
#[test]
fn checksum_on_the_trainers_device() {
    use burn::backend::Autodiff;
    use burn_cuda::Cuda;
    let d = Device::cuda(0).autodiff();
    d.seed(42);
    for (m, k, n) in [(512usize, 128usize, 1024usize), (64usize, 64usize, 2048usize)] {
        let a = Tensor::<2>::random([m, k], Distribution::Normal(0.0, 1.0), &d).require_grad();
        let b = Tensor::<2>::random([k, n], Distribution::Normal(0.0, 1.0), &d).require_grad();
        // The tuner has nothing to compare on a single call: repeat the same
        // shape, the way a training loop does.
        let mut c = a.clone().matmul(b.clone());
        for _ in 0..6 {
            c = a.clone().matmul(b.clone());
        }
        let _ = c.clone().backward();
        let v: Vec<f32> = c.into_data().try_to_vec().unwrap();
        let bits: u64 = v
            .iter()
            .fold(0u64, |h, x| h.wrapping_mul(31).wrapping_add(x.to_bits() as u64));
        let sum: f64 = v.iter().map(|&x| x as f64).sum();
        println!(
            "TRAINERDEV m{m}_n{n}_k{k}: n={} sum={sum:.9e} bits={bits:016x}",
            v.len()
        );
    }
    let _ = std::marker::PhantomData::<Autodiff<Cuda>>;
}

// ── level 11: does a gather's backward ZERO the untouched rows? ──────────
//
// The named fingerprint of the corruption is `loop_block.engram.memory`.
// Its forward is `table.gather(0, idx)` where idx touches only the rows the
// batch hashed to. The Adam step then reads the WHOLE gradient tensor, so
// every row the batch did not touch must arrive as exactly 0. If the backward
// allocates that buffer uninitialised and only scatter-adds the touched rows,
// the untouched ones are allocator garbage - and Adam turns a 1e18 into a
// 1e18 * lr write. Two processes, two allocator states, one corrupted table.
//
// This is the whole claim in one assertion: the untouched rows of a gather's
// gradient are zero.
#[test]
fn gather_backward_zeroes_the_untouched_rows() {
    let d = Device::cuda(0).autodiff();
    d.seed(7);
    let rows = 4096usize;
    let width = 96usize;
    let table = Tensor::<2>::random([rows, width], Distribution::Normal(0.0, 1.0), &d).require_grad();
    // touch 512 of the 4096 rows, the way a batch's hashes touch a table
    let hits: Vec<i64> = (0..512).map(|i| (i * 7 % rows as i64) as i64).collect();
    let idx = Tensor::<2, burn::tensor::Int>::from_data(
        burn::tensor::TensorData::new(hits.clone(), [512, 1]),
        &d,
    )
    .expand([512, width]);
    let out = table.clone().gather(0, idx);
    let loss = out.powf_scalar(2.0).sum();
    let grads = loss.backward();
    let g = table.grad(&grads).expect("the table gets a gradient");
    let v: Vec<f32> = g.into_data().try_to_vec().unwrap();
    let untouched: Vec<usize> = (0..rows).filter(|r| !hits.contains(&(*r as i64))).collect();
    let mut worst = 0.0f32;
    let mut worst_row = usize::MAX;
    for &r in &untouched {
        let s = &v[r * width..r * width + width];
        let m = s.iter().fold(0.0f32, |a, x| a.max(x.abs()));
        if m > worst {
            worst = m;
            worst_row = r;
        }
    }
    println!(
        "gather backward: {} untouched rows, worst |grad| on row {worst_row} = {worst:.6e}",
        untouched.len()
    );
    assert_eq!(
        worst, 0.0,
        "row {worst_row} was never gathered, yet its gradient is {worst:.6e}: the gradient \\
         buffer is not zeroed, so every untouched entry is allocator garbage"
    );
}

// ── level 12: is the seeded device RNG sound on a large tensor? ──────────
//
// The engram memory table is `engram_rows` rounded up to a power of two
// (32 768) times 96, initialised from the seeded device RNG. Two identical
// processes get different tables, and the difference is not rounding: rows
// differ by 1e18 absolute, which a Normal(0, 1) draw cannot produce. This is
// the smallest possible form of that claim: one seeded draw, one assertion
// about what a Normal draw can contain.
#[test]
fn a_seeded_normal_draw_stays_in_the_distribution() {
    // The model builds its parameters on the autodiff device (the trainer's
    // log says `checkpointing: Disabled`), so the init ops are dispatched
    // through the autodiff bridge - a different path from the bare device.
    let d = Device::cuda(0).autodiff();
    d.seed(1);
    for (rows, width) in [(32768usize, 96usize), (98304usize, 96usize), (1000usize, 96usize)] {
        let t = Tensor::<2>::random([rows, width], Distribution::Normal(0.0, 1.0), &d);
        let v: Vec<f32> = t.into_data().try_to_vec().unwrap();
        let max = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let nonfinite = v.iter().filter(|x| !x.is_finite()).count();
        println!(
            "seeded Normal draw [{rows},{width}]: max |x| = {max:.4e}, non-finite = {nonfinite} \
             (a 3.1M-draw from N(0,1) should top out near 5.5)"
        );
        assert!(
            max < 20.0,
            "[{rows},{width}]: a Normal(0,1) draw contains {max:.4e} - the RNG kernel produced \\
             values no distribution can"
        );
        assert_eq!(nonfinite, 0, "[{rows},{width}]: {nonfinite} non-finite draws");
    }
}
