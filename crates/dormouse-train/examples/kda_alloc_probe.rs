//! Is a KDA training step the KERNEL or the ALLOCATOR?
//!
//! `research/2026-09-27-kda-sota-ceiling.md` §2.5 puts the question on one
//! number: the fused chunk kernels' own floor is ~80-100 us, we measure
//! ~120 ms per call, and the hypothesis is that 248 MB of fresh scratch per
//! forward misses the cubecl pool. This is the test that document asks for.
//!
//! It is built HERE, in the trainer's own dependency graph, because a harness
//! built inside `vendor/burn-fused` links *unpatched* registry cubecl (no
//! `[patch.crates-io]` in that workspace) and that allocator is a different
//! one - the whole question is about this allocator.
//!
//! What it reports, per call at the production shape (b=10, t=512, 12 heads,
//! K=V=64, chunk 16, fp32 - the trainer's `Autodiff<Cuda, Balanced>`):
//!
//!   - **enqueue vs total.** cubecl is async, so `Instant` around a forward
//!     only measures the host. A real `Client::sync()` barrier separates the
//!     two, so "the host is the bottleneck" becomes a measurement, not a
//!     guess.
//!   - **cold vs warm vs after-cleanup.** The first call into an empty pool
//!     pays every `cudaMalloc`; `client.memory_cleanup()` hands the cached
//!     pages back to the driver, so the same call can be re-run cold on
//!     demand. Kernel cost is identical in all three; allocator cost is not.
//!   - **`memory_usage()` before/after**: `number_allocs`, `bytes_in_use`,
//!     `bytes_reserved`, `bytes_padding`. Reserved growing per call = pool
//!     miss = the driver is being asked for more memory.
//!
//! Run (release, quiet GPU, ONE process):
//!   cargo run --release -p dormouse-train --example kda_alloc_probe \
//!       --no-default-features --features cuda

use burn::backend::autodiff::Autodiff;
use burn::backend::autodiff::checkpoint::strategy::{BalancedCheckpointing, NoCheckpointing};
use burn::tensor::{Device, Tensor, TensorData};
use burn_dispatch::{DispatchDevice, devices::CubeDevice};
use cubecl_cuda::CudaRuntime;
use cubecl_runtime::runtime::Runtime as _;
use dormouse_core::attention::AdaptiveAttention;
use std::time::Instant;

/// The trainer's backend, verbatim from crates/dormouse-train/src/lib.rs.
type Ad = Autodiff<burn_cuda::Cuda, BalancedCheckpointing>;
/// The same math behind burn-kda's single fused autodiff node.
type NoCkpt = Autodiff<burn_cuda::Cuda, NoCheckpointing>;

/// The production shape: `small` preset, batch 10, seq 512, 12 heads,
/// K=V=64, chunk 16, fp32 (dormouse-train: d=768, 4 loop iterations).
/// The batch is an argument because the pool high-waters ~900 MB per call at
/// b=10, which does not fit beside another process on this 16 GB card - the
/// same collision that makes this harness a bad neighbour.
const T: usize = 512;
const D: usize = 768;
const H: usize = 12;
const HK: usize = 64;
const CHUNK: usize = 16;

fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1_048_576.0
}

/// The cubecl client behind the device: `sync` for a real barrier,
/// `memory_usage` for the accounting, `memory_cleanup` to force the cold path.
fn client_of(device: &Device) -> cubecl_runtime::client::Client {
    fn cuda(device: &Device) -> &burn_cuda::CudaDevice {
        match device.as_dispatch() {
            DispatchDevice::Cube(CubeDevice::Cuda(dev)) => dev,
            DispatchDevice::Autodiff(a) => match &**a {
                DispatchDevice::Cube(CubeDevice::Cuda(dev)) => dev,
                other => panic!("expected a CUDA device, got {other:?}"),
            },
            other => panic!("expected a CUDA device, got {other:?}"),
        }
    }
    CudaRuntime::client(cuda(device))
}

fn main() {
    let b: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let dev = Device::cuda(0);
    let adev = dev.clone().autodiff().gradient_checkpointing();
    let client = client_of(&dev);
    let attn = AdaptiveAttention::new(D, H, HK, &adev);

    // Deterministic input, no RNG drift between calls.
    let mut st = 0x243F6A88_5A30_8D3Du64;
    let mut next = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        (st % 10000) as f32 / 10000.0 - 0.5
    };
    let data: Vec<f32> = (0..b * T * D).map(|_| next()).collect();
    let x = || {
        Tensor::<3>::from_data(TensorData::new(data.clone(), [b, T, D]), &adev).require_grad()
    };

    println!(
        "KDA allocator probe: b={b} t={T} d={D} heads={H} K=V={HK} chunk={CHUNK} \
         ({} chunks)",
        T / CHUNK
    );
    println!("  backend: Autodiff<Cuda, BalancedCheckpointing> - the trainer's\n");

    // One forward+backward on a fresh graph, with a real device barrier at
    // every boundary. Returns the reserved bytes after the forward and after
    // the backward, and the pair's wall time.
    let mut call = |label: &str| -> (u64, u64, f32, f32) {
        let xi = x();
        let u0 = client.memory_usage();

        let t0 = Instant::now();
        let y = attn.gdn2.forward_train::<Ad>(xi.clone());
        let t1 = Instant::now(); // host: allocations + launches
        cubecl_environment::future::block_on(client.sync()).expect("device barrier");
        let t2 = Instant::now(); // GPU drained
        let u1 = client.memory_usage();

        let loss = y.clone().sum();
        let t3 = Instant::now();
        let grads = loss.backward();
        let bytes = attn
            .gdn2
            .q_proj
            .weight
            .grad(&grads)
            .map(|t| t.into_data().bytes.len())
            .unwrap_or(0);
        let t4 = Instant::now(); // host: backward allocations + launches
        cubecl_environment::future::block_on(client.sync()).expect("device barrier");
        let t5 = Instant::now(); // GPU drained
        let u2 = client.memory_usage();
        drop(grads);
        assert!(bytes > 0, "probe: no param gradient - untracked graph");

        let ms = |a: Instant, b: Instant| (b - a).as_secs_f32() * 1e3;
        println!(
            "  {label:<24} fwd host {:7.1} + gpu {:6.1} = {:7.1} ms | bwd host {:7.1} + gpu {:6.1} = {:7.1} ms",
            ms(t0, t1), ms(t1, t2), ms(t0, t2),
            ms(t3, t4), ms(t4, t5), ms(t3, t5),
        );
        println!(
            "  {:<24} allocs {:4} -> {:4} -> {:4} | in use {:6.1} -> {:6.1} -> {:6.1} MB | reserved {:6.1} -> {:6.1} -> {:6.1} MB (+{:.1}) | padding {:5.1} -> {:5.1} -> {:5.1} MB",
            "",
            u0.number_allocs, u1.number_allocs, u2.number_allocs,
            mb(u0.bytes_in_use), mb(u1.bytes_in_use), mb(u2.bytes_in_use),
            mb(u0.bytes_reserved), mb(u1.bytes_reserved), mb(u2.bytes_reserved),
            mb(u2.bytes_reserved.saturating_sub(u0.bytes_reserved)),
            mb(u0.bytes_padding), mb(u1.bytes_padding), mb(u2.bytes_padding),
        );
        (u1.bytes_reserved, u2.bytes_reserved, ms(t0, t2), ms(t3, t5))
    };

    call("call 1 (cold pool)");
    call("call 2 (warm)");
    let (reserved_fwd, reserved_bwd, t_fwd, t_bwd) = call("call 3 (warm)");

    // ── the same math behind ONE fused autodiff node ─────────────────
    // burn-kda's fused-op dispatch is a TypeId check on the WHOLE backend type
    // (`Autodiff<CudaBare>` = NoCheckpointing), so the trainer's
    // `Autodiff<Cuda, Balanced>` misses it and runs the ~150-ops-per-chunk
    // tensor path. A module on a NoCheckpointing device is the only way to
    // price the fused node in the same process: a module's tensors and the
    // input must share their checkpointing strategy or the op asserts.
    //
    // Runs here, before the rest, because the pool high-waters ~900 MB per
    // call and a second module does not fit after five of them.
    let adev_no = dev.clone().autodiff();
    let attn_no = AdaptiveAttention::new(D, H, HK, &adev_no);
    let x_no = || {
        Tensor::<3>::from_data(TensorData::new(data.clone(), [b, T, D]), &adev_no).require_grad()
    };
    let mut fused = |i: usize| -> (f32, f32) {
        let xi = x_no();
        let t0 = Instant::now();
        let y = attn_no.gdn2.forward_train::<NoCkpt>(xi.clone());
        let t1 = Instant::now();
        cubecl_environment::future::block_on(client.sync()).expect("device barrier");
        let t2 = Instant::now();
        let f_ms = (t2 - t0).as_secs_f32() * 1e3;
        let loss = y.clone().sum();
        let t3 = Instant::now();
        let grads = loss.backward();
        let bytes = attn_no
            .gdn2
            .q_proj
            .weight
            .grad(&grads)
            .map(|t| t.into_data().bytes.len())
            .unwrap_or(0);
        let t4 = Instant::now();
        cubecl_environment::future::block_on(client.sync()).expect("device barrier");
        let t5 = Instant::now();
        drop(grads);
        assert!(bytes > 0, "fused arm: no param gradient - untracked graph");
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f32() * 1e3;
        println!(
            "  iter {i} fwd host {:7.1} + gpu {:6.1} = {:7.1} ms | bwd host {:7.1} + gpu {:6.1} = {:7.1} ms",
            ms(t0, t1), ms(t1, t2), f_ms,
            ms(t3, t4), ms(t4, t5), ms(t3, t5),
        );
        (f_ms, ms(t3, t5))
    };
    println!("\n  F. Autodiff<Cuda, NoCheckpointing> -> the fused CUDA kernels, one node:");
    fused(0);
    let (f_fused, b_fused) = fused(1);
    fused(2);

    // The decisive pair: hand every cached page back to the driver and run the
    // identical call. If it returns to the cold time, the cost is the
    // allocator; if it stays warm, the pool was serving it and the cost is
    // the kernel.
    let before = client.memory_usage();
    client.memory_cleanup();
    cubecl_environment::future::block_on(client.sync()).expect("device barrier");
    let drained = client.memory_usage();
    println!(
        "\n  memory_cleanup(): reserved {:.1} -> {:.1} MB ({} allocs dropped)\n",
        mb(before.bytes_reserved),
        mb(drained.bytes_reserved),
        before.number_allocs - drained.number_allocs,
    );
    call("call 4 (after cleanup)");
    call("call 5 (warm again)");

    // Forward-only, steady state: the cheapest possible KDA call, to price
    // the pass a no-grad teacher forward would pay.
    let u_before = client.memory_usage();
    let t0 = Instant::now();
    for _ in 0..2 {
        let y = attn.gdn2.forward_train::<Ad>(x());
        let _ = y.into_data();
    }
    let warm = t0.elapsed().as_secs_f32() * 1e3 / 2.0;
    let u_after = client.memory_usage();
    println!("\n  2 warm forward-only calls: {warm:.1} ms/call");
    println!(
        "  forward-only allocs {} -> {} (+{}), reserved {:.1} -> {:.1} MB (+{:.1})",
        u_before.number_allocs,
        u_after.number_allocs,
        u_after.number_allocs as i64 - u_before.number_allocs as i64,
        mb(u_before.bytes_reserved),
        mb(u_after.bytes_reserved),
        mb(u_after.bytes_reserved.saturating_sub(u_before.bytes_reserved)),
    );

    println!(
        "\n  => tensor path (what the trainer runs): {t_fwd:.0} ms fwd + {t_bwd:.0} ms bwd = {:.0} ms | \
         fused op: {f_fused:.0} ms fwd + {b_fused:.0} ms bwd = {:.0} ms | {:.2}x on the fwd+bwd pair",
        t_fwd + t_bwd,
        f_fused + b_fused,
        (t_fwd + t_bwd) / (f_fused + b_fused),
    );
    println!(
        "     one fwd+bwd call reserves {:.1} MB (fwd {:.1}, after bwd {:.1}); a 4-iteration step \
         runs 4 of these plus 4 no-grad teacher forwards.",
        mb(reserved_fwd),
        mb(reserved_fwd),
        mb(reserved_bwd),
    );
}
