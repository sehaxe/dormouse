#![cfg(all(feature = "cuda", feature = "autodiff"))]
//! burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![allow(deprecated)]
//! Allocation accounting for the chunked WY forward and backward.
//!
//! Every number here is measured against cubecl's OWN allocator counters
//! (`Client::memory_usage()`: live allocations, bytes in use, bytes reserved),
//! not against a model of the code. The per-site table from `alloc_trace` is
//! cross-checked against that aggregate: the two must agree to within the
//! buffers allocated outside the traced sites.
//!
//! `memory_usage()` is a blocking submit to the server, so it doubles as the
//! device sync the timing needs — an `Instant` around a forward without one
//! measures launch time, which is how the earlier "0.23 ms" bench went wrong.
//!
//! Shape from the environment (defaults = the production shape):
//! `GDN2_B` (10), `GDN2_T` (512), `GDN2_H` (12), `GDN2_K` (64), `GDN2_V` (64).
//!
//! Run: cargo test --release -p dormouse-gdn2 --features "cuda,autodiff" \
//!        --test alloc_probe -- --ignored --nocapture --test-threads=1

use std::any::Any;
use std::time::Instant;

use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn::tensor::{Distribution, Tensor};
use dormouse_gdn2::{alloc_trace, chunk_autodiff_or_plain, chunk_wy_forward, CudaBare};

type AdBal = burn::backend::Autodiff<CudaBare, BalancedCheckpointing>;

/// `GDN2_POOL=exclusive` (default) installs the same pool the trainer installs
/// (`init_pools`: `ExclusivePages`); `GDN2_POOL=none` leaves the allocator
/// unpooled, which is the "every allocation is a raw cudaMalloc" case. The
/// delta between the two IS the pool-miss cost, measured rather than assumed.
fn install_pool(c: &burn_cubecl::cubecl::client::Client) {
    if std::env::var("GDN2_POOL").as_deref() == Ok("none") {
        return;
    }
    use cubecl_runtime::config::memory::{MemoryPoolsConfig, MemoryPoolsPreset};
    let cfg = MemoryPoolsConfig::Preset(MemoryPoolsPreset::ExclusivePages);
    let _ = burn_cubecl::cubecl::client::Client::install_memory_pools(c, &cfg);
}

fn env_usize(k: &str, d: usize) -> usize {    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn b() -> usize {
    env_usize("GDN2_B", 10)
}
fn t() -> usize {
    env_usize("GDN2_T", 512)
}
fn h() -> usize {
    env_usize("GDN2_H", 12)
}
fn k() -> usize {
    env_usize("GDN2_K", 64)
}
fn v() -> usize {
    env_usize("GDN2_V", 64)
}
/// 16 is final: f32 underflows `k/exp(cumsum(g))` below cumsum(g) = -88, i.e.
/// chunk > 17 at the K3 floor g = -5 (FlashKDA picked 16 for the same reason).
fn chunk() -> usize {
    env_usize("GDN2_CHUNK", 16)
}

/// The cubecl client, borrowed from a tensor's own `CubeTensor` (no extra
/// dependency on the dispatch types).
fn client() -> burn_cubecl::cubecl::client::Client {
    let dev: burn::tensor::Device = Default::default();
    let t = Tensor::<1>::zeros([1], &dev);
    let prim = t.try_into_primitive::<CudaBare>().ok().expect("cuda tensor");
    let cube: &burn_cubecl::tensor::CubeTensor = (&prim as &dyn Any)
        .downcast_ref()
        .expect("CubeTensor");
    cube.client.clone()
}

#[derive(Debug, Clone, Copy)]
struct Delta {
    allocs: i64,
    in_use: i64,
    reserved: i64,
}

type Usage = (u64, u64, u64);

fn usage(c: &burn_cubecl::cubecl::client::Client) -> Usage {
    let u = c.memory_usage();
    (u.number_allocs, u.bytes_in_use, u.bytes_reserved)
}

fn delta(before: Usage, after: Usage) -> Delta {
    Delta {
        allocs: after.0 as i64 - before.0 as i64,
        in_use: after.1 as i64 - before.1 as i64,
        reserved: after.2 as i64 - before.2 as i64,
    }
}

fn mb(v: i64) -> f64 {
    v as f64 / 1e6
}

fn loss_of(out: &Tensor<4>) -> Tensor<1, burn::tensor::Float> {
    out.clone().powf_scalar(2.0).sum()
}

fn print_sites(phase: &str, sites: &[(&'static str, u64, u64)]) {
    if sites.is_empty() {
        println!("  {phase}: NO traced sites -> the fused kernels did not run");
        return;
    }
    println!("  {phase} sites (traced, MB/call):");
    for (label, n, bytes) in sites.iter().take(9) {
        println!(
            "    {:>4} x {:>8.3}  =  {:>8.3}  {label}",
            n,
            *bytes as f64 / 1e6 / (*n).max(1) as f64,
            *bytes as f64 / 1e6
        );
    }
}

/// Time + allocator deltas + per-site table for one forward and one backward.
fn measure(
    what: &str,
    c: &burn_cubecl::cubecl::client::Client,
    fwd: impl Fn([Tensor<4>; 7]) -> (Tensor<4>, Tensor<4>),
) {
    let dev = burn::tensor::Device::autodiff(Default::default());
    let inp = inputs(&dev);

    for _ in 0..2 {
        let (o, _s) = fwd(inp.clone());
        let l: Tensor<1, burn::tensor::Float> = loss_of(&o);
        let _v: f32 = l.clone().into_scalar();
        let _ = l.backward();
    }

    alloc_trace::reset();
    alloc_trace::reset_iterations();
    let u0 = usage(c);
    let t0 = Instant::now();
    let (out, _state) = fwd(inp.clone());
    let u1 = usage(c);
    let fwd_ms = t0.elapsed().as_secs_f64() * 1e3;
    let d_fwd = delta(u0, u1);
    let fwd_sites = alloc_trace::table();
    let iters_fwd = alloc_trace::chunk_iterations();

    alloc_trace::reset();
    let l = loss_of(&out);
    let _v: f32 = l.clone().into_scalar();
    let u2 = usage(c);
    let t1 = Instant::now();
    let _grads = l.backward();
    let u3 = usage(c);
    let bwd_ms = t1.elapsed().as_secs_f64() * 1e3;
    let d_bwd = delta(u2, u3);
    let bwd_sites = alloc_trace::table();
    let iters_bwd = alloc_trace::chunk_iterations() - iters_fwd;

    println!("\n=== {what} ===");
    println!("shape b={} t={} h={} K={} V={} chunk={}", b(), t(), h(), k(), v(), chunk());
    println!(
        "{:<9} {:>9} {:>8} {:>10} {:>12}",
        "phase", "ms", "+allocs", "+MB live", "+MB reserved"
    );
    println!(
        "{:<9} {:>9.3} {:>8} {:>10.2} {:>12.2}",
        "forward", fwd_ms, d_fwd.allocs, mb(d_fwd.in_use), mb(d_fwd.reserved)
    );
    println!(
        "{:<9} {:>9.3} {:>8} {:>10.2} {:>12.2}",
        "backward", bwd_ms, d_bwd.allocs, mb(d_bwd.in_use), mb(d_bwd.reserved)
    );
    let traced: u64 = fwd_sites.iter().map(|e| e.2).sum::<u64>()
        + bwd_sites.iter().map(|e| e.2).sum::<u64>();
    let measured = d_fwd.in_use.max(0) as u64 + d_bwd.in_use.max(0) as u64;
    println!(
        "accounted: traced {:.2} MB of {:.2} MB live ({:.0}%)",
        mb(traced as i64),
        mb(measured as i64),
        100.0 * traced as f64 / measured.max(1) as f64
    );
    // The checkpointer question: `iters` counts EXECUTED chunk-loop
    // iterations, so a backward that re-executes the forward shows up here as
    // extra iterations and extra allocations.
    println!(
        "chunk-loop iterations: {iters_fwd} executed in the forward, \
         {iters_bwd} more during the backward (recompute if > 0)"
    );
    print_sites("forward", &fwd_sites);
    print_sites("backward", &bwd_sites);
}

/// Seven projected inputs `[B,H,T,K/V]` in the layout the module hands the
/// chunk kernels.
fn inputs_nograd(dev: &burn::tensor::Device) -> [Tensor<4>; 7] {
    let (b, h, t, k, v) = (b(), h(), t(), k(), v());
    let r = |dims: [usize; 4], m: f64, s: f64| {
        Tensor::<4>::random(dims, Distribution::Normal(m, s), dev)
    };
    [
        r([b, h, t, k], 0.0, 1.0),
        r([b, h, t, k], 0.0, 1.0),
        r([b, h, t, v], 0.0, 1.0),
        r([b, h, t, k], -0.5, 0.1),
        r([b, h, t, k], 0.0, 0.5),
        r([b, h, t, v], 0.0, 0.5),
        r([b, h, k, v], 0.0, 0.1),
    ]
}

fn inputs(dev: &burn::tensor::Device) -> [Tensor<4>; 7] {
    let (b, h, t, k, v) = (b(), h(), t(), k(), v());
    let r = |dims: [usize; 4], m: f64, s: f64| {
        Tensor::<4>::random(dims, Distribution::Normal(m, s), dev).require_grad()
    };
    [
        r([b, h, t, k], 0.0, 1.0),  // q
        r([b, h, t, k], 0.0, 1.0),  // k
        r([b, h, t, v], 0.0, 1.0),  // v
        r([b, h, t, k], -0.5, 0.1), // g: negative log decay
        r([b, h, t, k], 0.0, 0.5),  // b: erase gate
        r([b, h, t, v], 0.0, 0.5),  // w: write gate
        r([b, h, k, v], 0.0, 0.1),  // state
    ]
}

/// The fused path: one autodiff node over 3 forward + 2 backward kernels.
#[test]
#[ignore]
fn alloc_fused_node() {
    let c = client();
    install_pool(&c);
    measure(
        "fused node, Autodiff<Cuda> (NoCheckpointing)",
        &c,
        |inp| {
            chunk_autodiff_or_plain::<CudaBare>(
                inp[0].clone(),
                inp[1].clone(),
                inp[2].clone(),
                inp[3].clone(),
                inp[4].clone(),
                inp[5].clone(),
                inp[6].clone(),
                1.0,
                chunk(),
            )
        },
    );
}

/// The trainer's exact backend. `dormouse-kda`'s `is_autodiff_cuda` guard compares
/// the backend `TypeId` against `Autodiff<CudaBare>` = `NoCheckpointing`, so
/// this backend is the one that decides whether the fused kernels engage.
#[test]
#[ignore]
fn alloc_balanced_backend() {
    let c = client();
    install_pool(&c);
    measure(
        "chunk_wy_forward on Autodiff<Cuda, BalancedCheckpointing> (the trainer)",
        &c,
        |inp| {
            chunk_wy_forward(
                inp[0].clone(),
                inp[1].clone(),
                inp[2].clone(),
                inp[3].clone(),
                inp[4].clone(),
                inp[5].clone(),
                inp[6].clone(),
                1.0,
                chunk(),
            )
        },
    );
}

/// The same tensor path on `Autodiff<Cuda>`, for a like-for-like comparison.
#[test]
#[ignore]
fn alloc_tensor_path_nc() {
    let c = client();
    install_pool(&c);
    measure(
        "chunk_wy_forward tensor path, Autodiff<Cuda> (NoCheckpointing)",
        &c,
        |inp| {
            chunk_wy_forward(
                inp[0].clone(),
                inp[1].clone(),
                inp[2].clone(),
                inp[3].clone(),
                inp[4].clone(),
                inp[5].clone(),
                inp[6].clone(),
                1.0,
                chunk(),
            )
        },
    );
}

/// The bare forward with no graph at all: the kernel cost floor the fwd_ms
/// numbers above must be compared against, averaged over 5 runs.
#[test]
#[ignore]
fn alloc_bare_forward() {
    let c = client();
    install_pool(&c);
    let dev: burn::tensor::Device = Default::default();
    // no require_grad: this one runs on the bare backend.
    let inp = inputs_nograd(&dev);
    for _ in 0..2 {
        let _ = chunk_wy_forward(
            inp[0].clone(),
            inp[1].clone(),
            inp[2].clone(),
            inp[3].clone(),
            inp[4].clone(),
            inp[5].clone(),
            inp[6].clone(),
            1.0,
            chunk(),
        );
    }
    let runs = 5;
    alloc_trace::reset();
    let u0 = usage(&c);
    let t0 = Instant::now();
    for _ in 0..runs {
        let _ = chunk_wy_forward(
            inp[0].clone(),
            inp[1].clone(),
            inp[2].clone(),
            inp[3].clone(),
            inp[4].clone(),
            inp[5].clone(),
            inp[6].clone(),
            1.0,
            chunk(),
        );
    }
    let u1 = usage(&c);
    let ms = t0.elapsed().as_secs_f64() * 1e3 / runs as f64;
    let d = delta(u0, u1);
    println!(
        "\n=== bare fused forward on CudaBare, {runs} runs averaged (b={} t={} h={} K={} V={} chunk={}) ===",
        b(), t(), h(), k(), v(), chunk()
    );
    println!(
        "  {:.3} ms/call, +{} allocs, +{:.2} MB live, +{:.2} MB reserved",
        ms,
        d.allocs / runs,
        mb(d.in_use) / runs as f64,
        mb(d.reserved) / runs as f64
    );
    print_sites("forward", &alloc_trace::table());
}
