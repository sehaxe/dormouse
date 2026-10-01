//! Is a KDA training step the KERNEL or the ALLOCATOR?
//!
//! `docs/research/2026-09-27-kda-sota-ceiling.md` §2.5 puts the whole question on
//! one number: the fused chunk kernels' own floor is ~80-100 us, we measure
//! ~120 ms per call, and the hypothesis is that 248 MB of fresh scratch per
//! forward misses the cubecl pool. This harness is the test that document asks
//! for, at the PRODUCTION shape, with the three measurements that separate the
//! two hypotheses:
//!
//!   - **enqueue vs total.** The cubecl backend is async, so `Instant` around
//!     a forward only measures the host's enqueue. Every number here is
//!     `enqueue` + `gpu tail` with a real D2H read in between. If the enqueue
//!     alone is the cost, the work is on the host - i.e. the allocator.
//!   - **cold vs warm.** The first call into an empty pool pays every
//!     `cudaMalloc`; later calls with the same shapes should hit. A kernel
//!     cost is identical in both; an allocator cost is not.
//!   - **per-chunk scaling.** 32 chunks vs 4 chunks at fixed batch. A cost
//!     that scales with chunks is in the chunk loop; a flat one is not.
//!
//!   cargo run --release -p burn-kda --example kda_step_probe --features cuda
//!
//! NOTE on the binary: built inside `vendor/dormouse-fused` this links *unpatched*
//! registry cubecl (that workspace has no `[patch.crates-io]`), so the
//! authoritative numbers come from
//! `crates/dormouse-train/examples/kda_alloc_probe.rs`, which links the same
//! patched cubecl the trainer does. This harness is the shape-independent
//! cross-check.

#![cfg(feature = "cuda")]

use burn::backend::autodiff::checkpoint::strategy::{BalancedCheckpointing, NoCheckpointing};
use burn::backend::autodiff::Autodiff;
use burn::module::Module;
use burn::tensor::{Device, Tensor, TensorData};
use burn_kda::{KdaConfig, KdaModule};

/// The production shape: `small` preset, batch 10, seq 512, 12 heads,
/// K=V=64, chunk 16, fp32.
const B: usize = 10;
const T: usize = 512;
const D: usize = 768;
const H: usize = 12;
const HK: usize = 64;
const CHUNK: usize = 16;

fn main() {
    let dev = Device::cuda(0);
    let adev = dev.clone().autodiff().gradient_checkpointing();
    let adev_no = dev.clone().autodiff();

    let cfg = KdaConfig {
        hidden_size: D,
        num_heads: H,
        head_dim: HK,
        use_short_conv: false, // the preset disables it (it NaN'd on this box)
        chunk_size: CHUNK,
        ..Default::default()
    };
    let m = KdaModule::new(&cfg, 0.0, &adev);
    // The fused-op dispatch is a TypeId check on the whole backend type, so
    // `Autodiff<Cuda, Balanced>` MISSES it and falls to the tensor chunk path.
    // A second module on a NoCheckpointing device is the only way to compare
    // the two paths in one process: a module's tensors and the input must
    // share their checkpointing strategy or the op asserts.
    let m_no = KdaModule::new(&cfg, 0.0, &adev_no);

    let mut st = 0x243F6A88_5A30_8D3Du64;
    let mut next = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        (st % 10000) as f32 / 10000.0 - 0.5
    };
    let data: Vec<f32> = (0..B * T * D).map(|_| next()).collect();
    let data_short: Vec<f32> = (0..B * 64 * D).map(|_| next()).collect();
    let x = |t: usize, d: &[f32], dev: &Device| {
        Tensor::<3>::from_data(TensorData::new(d.to_vec(), [B, t, D]), dev).require_grad()
    };

    // One call, with the host/GPU split at every boundary. `into_data` is a
    // real D2H, so each boundary is a genuine device sync.
    let mut call = |i: usize, t: usize, d: &[f32], fused: bool| {
        let xi = if fused {
            x(t, d, &adev_no)
        } else {
            x(t, d, &adev)
        };
        let t0 = std::time::Instant::now();
        let y = if fused {
            m_no.forward_train::<Autodiff<burn_cuda::Cuda, NoCheckpointing>>(xi.clone())
        } else {
            m.forward_train::<Autodiff<burn_cuda::Cuda, BalancedCheckpointing>>(xi.clone())
        };
        let t1 = std::time::Instant::now(); // host: allocations + launches
        let _ = y.clone().into_data();
        let t2 = std::time::Instant::now(); // GPU drained
        let loss = y.clone().sum();
        let t3 = std::time::Instant::now();
        let grads = loss.backward();
        let bytes = xi
            .grad(&grads)
            .map(|t| t.into_data().bytes.len())
            .unwrap_or(0);
        let t4 = std::time::Instant::now(); // host: backward allocations + launches
        let _ = bytes;
        let t5 = std::time::Instant::now(); // GPU drained
        let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f32() * 1e3;
        println!(
            "  iter {i} t={t:<4} fwd enq {:8.1} + gpu {:7.1} = {:8.1} ms | bwd enq {:8.1} + gpu {:7.1} = {:8.1} ms",
            ms(t0, t1), ms(t1, t2), ms(t0, t2),
            ms(t3, t4), ms(t4, t5), ms(t3, t5),
        );
    };

    println!("KDA production-shape probe: b={B} t={T} d={D} heads={H} K=V={HK} chunk={CHUNK}");
    println!("  (BalancedCheckpointing = the trainer's backend, TENSOR chunk path;");
    println!("   NoCheckpointing = the same math behind ONE fused autodiff node)\n");
    println!("  A. trainer today: Autodiff<Cuda, Balanced> -> tensor chunk path");
    for i in 0..5 {
        call(i, T, &data, false);
    }
    println!("\n  B. Autodiff<Cuda, NoCheckpointing> -> fused CUDA kernels, one node");
    for i in 0..5 {
        call(i, T, &data, true);
    }
    println!("\n  C. per-chunk scaling, tensor path, 4 chunks instead of 32:");
    for i in 0..3 {
        call(i, 64, &data_short, false);
    }
    println!("\n  D. per-chunk scaling, fused path, 4 chunks instead of 32:");
    for i in 0..3 {
        call(i, 64, &data_short, true);
    }
    println!("\n  params in module: {}", m.num_params());
}
