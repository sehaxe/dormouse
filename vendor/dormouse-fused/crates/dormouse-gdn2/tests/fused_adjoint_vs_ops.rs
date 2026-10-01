#![cfg(all(feature = "cuda", feature = "autodiff"))]
//! The fused ADJOINT against the tensor-ops path, on CUDA, on one cotangent.
//!
//! # What this is, and what it is not
//!
//! `tests/fused_chunk_verify.rs::fused_op_grads_match_tensor_path_cuda` claims
//! to compare "the fused-op backward" with the tensor path. It does not: the
//! "fused" side of that pair is `ChunkWy::backward`, which took the TENSOR
//! branch - the fused-forward branch was gated on `fused_forced_off()`, so it
//! could only run when the kill switch said the fused path was OFF. Both sides
//! of that comparison were the tensor adjoint, so its `rel` numbers were
//! tensor-vs-tensor and its tolerance was never exercised against a kernel.
//!
//! This test calls the kernel path directly - `fused_chunk_forward_scratch`
//! for the exports, `fused_chunk_backward` for the gradients - on BARE
//! tensors, and takes the reference from burn's own autograd over
//! `chunk_wy_forward`'s per-op implementation. Two different algorithms, one
//! cotangent, seven gradients.
//!
//! The state output is NOT in the comparison: `ChunkWy` returns it as an
//! untracked leaf, so no gradient reaches it on either path (module header,
//! `src/autodiff.rs`). `d_s` here is the gradient of the INPUT state, which
//! both paths do compute.
//!
//! Run: `cargo test -p dormouse-gdn2 --features cuda,autodiff --test fused_adjoint_vs_ops -- --nocapture`

use burn::tensor::{Distribution, Tensor, TensorData};
use dormouse_gdn2::kernel::chunk_adjoint_cube::cuda::{fused_chunk_backward, FusedBackwardInputs};
use dormouse_gdn2::kernel::chunk_cube::cuda::fused_chunk_forward_scratch;
use dormouse_gdn2::{chunk_wy_forward, CudaBare};

const NAMES: [&str; 7] = ["q", "k", "v", "g", "b", "w", "s"];

/// Relative bound for every fused-adjoint gradient, measured against the
/// largest magnitude in the tensor.
///
/// The derivation is the CPU twin of this comparison, green since the tensor
/// adjoint landed (`tests/autodiff_chunk.rs::fused_grads_match_tensor_path`,
/// tolerance `RELATOL`): the tensor adjoint against burn's per-op autograd,
/// same seven inputs, same f32. Its worst input is `k` at ~1e-3 relative,
/// because k's gradient takes the double trip through the triangular solve
/// and the `k/E · E` cancellation; the other six land at 1e-4 or below. The
/// fused adjoint is a THIRD reassociation of the same algebra, so the same
/// conditioning applies and this bar is 3x that worst case. A value above it
/// is not fp32 noise for this shape: it is a finding, not a tolerance to be
/// widened.
const GRAD_REL_TOL: f32 = 3e-3;

/// `max|a-b| / max|a|` over the tensor, the measure every other test here uses.
fn rel(a: &Tensor<4>, b: &Tensor<4>) -> (f32, f32) {
    assert_eq!(a.dims(), b.dims(), "shape mismatch in the comparison");
    let (av, bv) = (vals(a), vals(b));
    let (mut max_abs, mut scale) = (0.0f32, 0.0f32);
    for (x, y) in av.iter().zip(bv.iter()) {
        max_abs = max_abs.max((x - y).abs());
        scale = scale.max(x.abs()).max(y.abs());
    }
    (max_abs / scale.max(1e-30), scale)
}

fn vals(t: &Tensor<4>) -> Vec<f32> {
    t.clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}

/// The same values, on another device, as a fresh leaf.
fn leaf_of(t: &Tensor<4>, dev: &burn::tensor::Device) -> Tensor<4> {
    Tensor::<4>::from_data(TensorData::new(vals(t), t.dims().to_vec()), dev).require_grad()
}

#[test]
fn fused_adjoint_matches_the_ops_path_on_cuda() {
    // TWO shapes, and the smaller one is the DIAGNOSTIC. `time = chunk` is a
    // single chunk: the BPTT chain through the state has nothing to carry, so
    // `d_k_bptt` and `d_e_bptt` are identically zero and every other term is
    // intra-chunk. `time = 4*chunk` is the same thing plus three chained
    // chunks. A gradient wrong at BOTH shapes is wrong inside BK1; one right
    // at one chunk and wrong at four is wrong in the chain.
    let mut worst = (0.0f32, "");
    for (batch, heads, time, k_dim, v_dim, chunk) in [
        (1usize, 2usize, 16usize, 32usize, 32usize, 16usize),
        (1, 2, 64, 32, 32, 16),
    ] {
        println!(
            "\n--- b={batch} h={heads} t={time} k={k_dim} v={v_dim} chunk={chunk} \
             ({} chunk(s) per sequence) ---",
            time / chunk
        );
        let w = compare(batch, heads, time, k_dim, v_dim, chunk);
        if w.0 > worst.0 {
            worst = w;
        }
    }
    assert!(
        worst.0 < GRAD_REL_TOL,
        "the fused adjoint disagrees with the ops path: worst input {} at \
         rel={:.3e} (bound {GRAD_REL_TOL:.1e}) - above fp32 noise for this shape, \
         so this is a wrong adjoint, not a loose tolerance",
        worst.1,
        worst.0
    );
}

/// Seven fused-adjoint gradients against burn's autograd over the ops path, at
/// one shape. Returns `(worst relative error, the input it was on)`; prints
/// every input first, so a failure names what is wrong rather than only where
/// the scan stopped.
fn compare(
    batch: usize,
    heads: usize,
    time: usize,
    k_dim: usize,
    v_dim: usize,
    chunk: usize,
) -> (f32, &'static str) {
    let scale = (k_dim as f64).powf(-0.5);

    let bare: burn::tensor::Device = Default::default();
    bare.seed(11);
    let r = |shape: [usize; 4], m: f64, s: f64| {
        Tensor::<4>::random(shape, Distribution::Normal(m, s), &bare)
    };
    // The layout the KDA module produces: q/k/v unit-ish, g a small negative
    // log-decay, b/w small gates, a non-zero initial state.
    let inputs = [
        r([batch, heads, time, k_dim], 0.0, 0.4),  // q
        r([batch, heads, time, k_dim], 0.0, 0.4),  // k
        r([batch, heads, time, v_dim], 0.0, 0.4),  // v
        r([batch, heads, time, k_dim], -0.5, 0.2), // g
        r([batch, heads, time, k_dim], 0.0, 0.3),  // b
        r([batch, heads, time, v_dim], 0.0, 0.3),  // w
        r([batch, heads, k_dim, v_dim], 0.0, 0.2), // state
    ];
    let d_out = r([batch, heads, time, v_dim], 0.0, 1.0);

    // ---- the fused kernels, on bare CUDA --------------------------------
    let (fused_out, _fused_state, io) = fused_chunk_forward_scratch::<CudaBare>(
        inputs[0].clone(),
        inputs[1].clone(),
        inputs[2].clone(),
        inputs[3].clone(),
        inputs[4].clone(),
        inputs[5].clone(),
        inputs[6].clone(),
        scale,
        chunk,
    )
    .expect("the fused forward must engage on the bare CUDA backend");
    let fbi = FusedBackwardInputs {
        m_inv: io.m_inv,
        aqk: io.aqk,
        qgt: io.qgt,
        glast: io.glast,
        v_new: io.v_new,
        states: io.states,
        w: io.w,
        u: io.u,
        gexp: io.gexp,
    };
    let fused = fused_chunk_backward::<CudaBare>(
        &fbi, &inputs[1], &inputs[2], &inputs[4], &inputs[5], &d_out, scale, chunk,
    )
    .expect("the fused adjoint must engage on the bare CUDA backend");
    let fused = [
        fused.d_q.clone(),
        fused.d_k.clone(),
        fused.d_v.clone(),
        fused.d_g.clone(),
        fused.d_b.clone(),
        fused.d_w.clone(),
        fused.d_s.clone(),
    ];

    // ---- the reference: burn's autograd over the per-op chunk loop ------
    let ad = burn::tensor::Device::autodiff(bare);
    let leaves: Vec<Tensor<4>> = inputs.iter().map(|t| leaf_of(t, &ad)).collect();
    let (ops_out, _ops_state) = chunk_wy_forward(
        leaves[0].clone(),
        leaves[1].clone(),
        leaves[2].clone(),
        leaves[3].clone(),
        leaves[4].clone(),
        leaves[5].clone(),
        leaves[6].clone(),
        scale,
        chunk,
    );
    let cotangent =
        Tensor::<4>::from_data(TensorData::new(vals(&d_out), d_out.dims().to_vec()), &ad);
    let grads = (ops_out.clone() * cotangent).sum().backward();

    // The forward first: if these disagree, every gradient number below is
    // noise and the comparison is meaningless.
    let (fwd, _) = rel(&fused_out, &ops_out);
    println!("forward out: rel={fwd:.3e}");
    assert!(
        fwd < 1e-3,
        "the fused forward drifted off the ops path: {fwd:.3e} - the gradients \
         below would not be comparable"
    );

    // By INDEX, not by shape: q/k/g/b all share one shape and v/w another, so
    // a shape lookup would compare d_v against q's gradient.
    let mut worst = (0.0f32, "");
    for (i, (name, gf)) in NAMES.iter().zip(fused.iter()).enumerate() {
        let gr = leaves[i]
            .grad(&grads)
            .unwrap_or_else(|| panic!("no ops-path gradient for input {name}"));
        let (d, scale_ref) = rel(gf, &gr);
        println!("gradient {name}: rel={d:.3e}  |ref|max={scale_ref:.3e}");
        if d > worst.0 {
            worst = (d, name);
        }
    }
    worst
}

/// The forward numbers, at the shape the trainer runs, on the same GPU, in one
/// process: the fused kernels against the tensor-op chunk loop.
///
/// # Why this is a separate `#[ignore]`d test
///
/// A timing is not a gate - it would flake and then be deleted, and a deleted
/// timing is the measurement this whole exercise is for. It is a bench with a
/// test harness, run on demand:
///
/// ```text
/// cargo test -p dormouse-gdn2 --features cuda,autodiff --test fused_adjoint_vs_ops \
///   -- --ignored --nocapture --test-threads=1
/// ```
///
/// # Why it syncs
///
/// `Client::memory_usage()` is a blocking submit to the server, so it is both
/// the device sync a timer needs and the allocator's own counters. Timing a
/// launch-bound workload without one measures the LAUNCHES, not the work -
/// which is how the earlier "0.23 ms" bench in this crate went wrong (see
/// `tests/alloc_probe.rs`, whose header says the same thing).
///
/// Shape: the trainer's own. `configs/small.toml` is `n_heads 12`,
/// `head_dim 64`, `max_seq_len 512`, and `dormouse-core/src/attention.rs`
/// sets `chunk_size: 16` (the fused kernels refuse `c > 16`). Batch is from
/// `GDN2_B`, default 8: the batch the 25.8 s/step attention measurement was
/// taken at.
#[test]
#[ignore = "a bench, not a gate: measures wall clock and is run on demand"]
fn bench_fused_vs_ops_forward_at_the_training_shape() {
    use std::time::Instant;

    let (batch, heads, time, k_dim, v_dim) = (
        std::env::var("GDN2_B")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8),
        12usize,
        512usize,
        64usize,
        64usize,
    );
    let chunk = 16usize;
    let scale = (k_dim as f64).powf(-0.5);

    let bare: burn::tensor::Device = Default::default();
    bare.seed(3);
    let r = |shape: [usize; 4], m: f64, s: f64| {
        Tensor::<4>::random(shape, Distribution::Normal(m, s), &bare)
    };
    let inputs = [
        r([batch, heads, time, k_dim], 0.0, 0.4),
        r([batch, heads, time, k_dim], 0.0, 0.4),
        r([batch, heads, time, v_dim], 0.0, 0.4),
        r([batch, heads, time, k_dim], -0.5, 0.2),
        r([batch, heads, time, k_dim], 0.0, 0.3),
        r([batch, heads, time, v_dim], 0.0, 0.3),
        r([batch, heads, k_dim, v_dim], 0.0, 0.2),
    ];
    let d_out = r([batch, heads, time, v_dim], 0.0, 1.0);

    // the sync, borrowed from a tensor's own CubeTensor
    let client = {
        use std::any::Any;
        let t = Tensor::<1>::zeros([1], &bare);
        let prim = t
            .try_into_primitive::<CudaBare>()
            .ok()
            .expect("cuda tensor");
        let cube = (&prim as &dyn Any)
            .downcast_ref::<burn_cubecl::tensor::CubeTensor>()
            .expect("CubeTensor");
        cube.client.clone()
    };
    let ms = |f: &mut dyn FnMut()| -> f64 {
        let _sync = client.memory_usage();
        let t0 = Instant::now();
        f();
        let _sync = client.memory_usage();
        t0.elapsed().as_secs_f64() * 1e3
    };

    let fused_fwd = || {
        fused_chunk_forward_scratch::<CudaBare>(
            inputs[0].clone(),
            inputs[1].clone(),
            inputs[2].clone(),
            inputs[3].clone(),
            inputs[4].clone(),
            inputs[5].clone(),
            inputs[6].clone(),
            scale,
            chunk,
        )
    };
    let runs = 3;
    for _ in 0..2 {
        fused_fwd();
    }
    let f_fwd = (0..runs)
        .map(|_| {
            ms(&mut || {
                let _ = fused_fwd();
            })
        })
        .sum::<f64>()
        / runs as f64;

    // ---- fused kernels, the adjoint on top ------------------------------
    let (_out, _, io) = fused_fwd().expect("the fused forward must engage");
    let fbi = FusedBackwardInputs {
        m_inv: io.m_inv,
        aqk: io.aqk,
        qgt: io.qgt,
        glast: io.glast,
        v_new: io.v_new,
        states: io.states,
        w: io.w,
        u: io.u,
        gexp: io.gexp,
    };
    let f_bwd = (0..runs)
        .map(|_| {
            ms(&mut || {
                let _ = fused_chunk_backward::<CudaBare>(
                    &fbi, &inputs[1], &inputs[2], &inputs[4], &inputs[5], &d_out, scale, chunk,
                );
            })
        })
        .sum::<f64>()
        / runs as f64;

    // ---- the tensor-op chunk loop, on the trainer's backend --------------
    let ad = burn::tensor::Device::autodiff(bare);
    let leaves: Vec<Tensor<4>> = inputs.iter().map(|t| leaf_of(t, &ad)).collect();
    let ops_fwd = || {
        chunk_wy_forward(
            leaves[0].clone(),
            leaves[1].clone(),
            leaves[2].clone(),
            leaves[3].clone(),
            leaves[4].clone(),
            leaves[5].clone(),
            leaves[6].clone(),
            scale,
            chunk,
        )
    };
    let (ops_out, _) = ops_fwd();
    let cotangent =
        Tensor::<4>::from_data(TensorData::new(vals(&d_out), d_out.dims().to_vec()), &ad);
    let o_fwd = ms(&mut || {
        let _ = ops_fwd();
    });
    let o_bwd = ms(&mut || {
        let _ = (ops_out.clone() * cotangent.clone()).sum().backward();
    });

    println!(
        "\nchunked WY, b={batch} h={heads} t={time} k={k_dim} v={v_dim} chunk={chunk} \
         ({} chunks/seq, {runs} runs averaged)\n\
         fused kernels: forward {f_fwd:8.3} ms   adjoint {f_bwd:8.3} ms\n\
         tensor ops   : forward {o_fwd:8.3} ms   backward{o_bwd:8.3} ms\n\
         forward speedup {:.1}x, fwd+adjoint speedup {:.1}x",
        time / chunk,
        o_fwd / f_fwd,
        o_fwd / (f_fwd + f_bwd)
    );
}
