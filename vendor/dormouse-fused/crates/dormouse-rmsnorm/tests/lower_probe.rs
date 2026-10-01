//! # THE LOWERING LADDER — a permanent bisect instrument, not a throwaway.
//!
//! ## Why this file exists
//!
//! `src/fused.rs`'s kernel has failed to lower on this backend since it was
//! written, and the diagnosis was wrong twice:
//!
//! * `34c5631` said the shared-memory width was sized by a runtime value, and
//!   added `#[comptime]` to the *launch signature* while leaving
//!   `let threads = 256usize;` in the body.
//! * `9ac0377` did it properly — `#[comptime] threads` in the signature, one
//!   `pub const THREADS` for both `CubeDim` and `Shared::new_slice` — and
//!   `tests/fused_kernel_gate.rs` reproduced the **identical** error.
//!
//! So the shared-memory width was never the whole cause, and the account
//! written in `fused.rs`'s doc comment after `34c5631` is retracted by
//! measurement. `tests/fused_kernel_gate.rs` is the gate that the fix must turn
//! green; THIS file is what says *which rung of the kernel* is at fault, and it
//! stays in the tree so the next person starts from a measurement instead of a
//! story.
//!
//! ## The shape of the error is a hint, and it was read as one
//!
//! ```
//! the lowered module does not verify: Compilation error: verification failed.
//! Expected operand type llvm.ptr, but found builtin.integer
//! ```
//!
//! That is a **pointer where an integer is expected** — the opposite of a size
//! problem, which is what every "the shared array was sized wrong" story
//! predicts. It is the signature of a barrier or a shared-memory access
//! reaching the LLVM dialect as a raw value instead of a pointer, so the ladder
//! below spends most of its rungs on barriers rather than on sizes.
//!
//! ## How to read the ladder
//!
//! Each rung is a separate `#[test]` and must be run in its own process
//! (`--exact <name>`): a launch failure poisons the CUDA context, so one bad
//! rung takes the rest of the binary down with it and the report becomes
//! useless. Every rung checks **two** things separately, and the split matters:
//!
//! * **did it lower** — the kernel wrote a value this file can predict, so a
//!   rung that computes the wrong RMSNorm is still a *passing* rung of the
//!   ladder. An earlier version of this file compared every rung against the
//!   RMSNorm reference, which made p0/p1/p3 look broken when they had in fact
//!   lowered perfectly — a ladder that cannot tell "failed to compile" from
//!   "computed something else" measures the wrong thing.
//! * **what it computed** — a per-rung exact expected value.
//!
//! The rungs, in order, each adding one thing to the one before it:
//!
//! | rung | adds | |
//! |---|---|---|
//! | `p0_three_buffers` | 3 buffers, an index, a multiply. no shared, no barrier | baseline |
//! | `p1_runtime_scalar` | a runtime `f32` arg through `F::cast_from` | |
//! | `p2_shared_write_sync_read` | `Shared::new_slice` at a `#[comptime]` size, one write, ONE `sync_cube()`, one read by thread 0 | **passes: the comptime width and shared memory are not the fault** |
//! | `p3_cond_write_then_bcast` | a conditional write to shared, a SECOND barrier, then a read by **all** threads | |
//! | `p4_barrier_in_while` | **no shared at all**: a `while` loop with a `sync_cube()` in it, then a trivial write | the minimal suspect |
//! | `p5_tree_reduction` | the real kernel's whole reduction: shared, strided accumulate, `sync_cube()`, the halving `while` with a barrier in it, `sync_cube()`, broadcast read | |
//! | `p6_serial_reduction` | the SAME kernel with the halving `while` replaced by thread 0's serial accumulation — the subtraction that isolates the tree | |
//! | `p7_full_kernel` | `src/fused.rs`'s kernel, verbatim | the failure itself |
//!
//! Run: `cargo test -p dormouse-rmsnorm --features cuda --test lower_probe -- --exact <rung>`

#![cfg(feature = "cuda")]

use burn::tensor::{Device, Tensor, TensorData};
use std::any::Any;

use burn_cubecl::tensor::CubeTensor;
use cubecl::client::Client;
use cubecl::prelude::*;

type B = burn_cubecl::CubeBackend;

/// 8 is small enough that a single thread can reduce a row, and `2^3` so the
/// halving loop in `p5` terminates in three steps against a 256-lane block.
const D: usize = 8;
const ROWS: usize = 12;
const THREADS: u32 = 256;

fn down2(t: Tensor<2>) -> CubeTensor {
    let prim = t.clone().try_into_primitive::<B>().unwrap();
    let c = (&prim as &dyn Any).downcast_ref::<CubeTensor>().unwrap();
    c.clone()
}
fn down1(t: Tensor<1>) -> CubeTensor {
    let prim = t.clone().try_into_primitive::<B>().unwrap();
    let c = (&prim as &dyn Any).downcast_ref::<CubeTensor>().unwrap();
    c.clone()
}

struct Io {
    out: Tensor<2>,
    outc: CubeTensor,
    client: Client,
    x: CubeTensor,
    w: CubeTensor,
    xvals: Vec<f32>,
    wvals: Vec<f32>,
}

fn io() -> Io {
    let dev = Device::cuda(0);
    let xvals: Vec<f32> = (0..ROWS * D).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
    let wvals: Vec<f32> = (0..D).map(|i| 0.5 + 0.25 * i as f32).collect();
    let xt = Tensor::<2>::from_data(TensorData::new(xvals.clone(), [ROWS, D]), &dev);
    let wt = Tensor::<1>::from_data(TensorData::new(wvals.clone(), [D]), &dev);
    let outt = Tensor::<2>::empty([ROWS, D], &dev);
    let xc = down2(xt);
    let client = xc.client.clone();
    let wc = down1(wt);
    let outc = down2(outt.clone());
    Io { out: outt, outc, client, x: xc, w: wc, xvals, wvals }
}

impl Io {
    /// What the device actually holds. A lowering failure surfaces HERE as
    /// "The bytes were never written ... the lowered module does not verify" —
    /// this read is the only place that message can come from.
    fn read(&self) -> Vec<f32> {
        self.out
            .clone()
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
    fn x_at(&self, r: usize, i: usize) -> f32 {
        self.xvals[r * D + i]
    }
    fn w_at(&self, i: usize) -> f32 {
        self.wvals[i]
    }
    /// `did it lower` AND `what it computed`, as ONE number. Relative, so a
    /// rung is judged on the shape of its own answer and not on its scale.
    fn worst(&self, want: &dyn Fn(usize, usize) -> f32) -> f64 {
        let got = self.read();
        assert_eq!(got.len(), ROWS * D, "the output has {} elements, not {}", got.len(), ROWS * D);
        let mut worst = 0.0f64;
        for r in 0..ROWS {
            for i in 0..D {
                let w = f64::from(want(r, i));
                let d = (f64::from(got[r * D + i]) - w).abs() / w.abs().max(1.0);
                worst = worst.max(d);
            }
        }
        worst
    }
    /// For a rung whose arithmetic is a deliberate STUB: the only question is
    /// whether it lowered, so only that is asserted. `p3` is this case — its
    /// accumulate is `partial[tid] = x[base]`, not a sum of squares, so its
    /// output is not RMSNorm and a numeric check would be measuring the stub.
    fn check_lowered(&self, rung: &str) {
        let got = self.read();
        assert!(
            got.iter().any(|v| v.is_finite()),
            "{rung} read back nothing finite - it did not lower"
        );
        eprintln!("{rung}: LOWERED (arithmetic is a stub here, so the value is not checked)");
    }
    /// The assert every rung ends with. Loose on purpose (1e-4 relative): a
    /// rung is not a correctness test, it is a lowering test, and a rung that
    /// lowers has already told us everything this file is for.
    fn check(&self, rung: &str, want: &dyn Fn(usize, usize) -> f32) {
        let worst = self.worst(want);
        eprintln!("{rung}: LOWERED, worst rel {worst:e} against this rung's own expected value");
        assert!(worst < 1e-4, "{rung} lowered but computed the wrong value: worst {worst:e}");
    }
}

// ── p0: three buffers, an index, a multiply ────────────────────────────────

#[cube(launch_unchecked)]
fn p0<F: Float>(x: &[F], w: &[F], out: &mut [F], #[comptime] d: u32) {
    let row = CUBE_POS_X as usize;
    let d = d as usize;
    if UNIT_POS_X == 0u32 {
        let mut i = 0;
        while i < d {
            out[row * d + i] = x[row * d + i] * w[i];
            i += 1;
        }
    }
}

#[test]
fn p0_three_buffers() {
    let io = io();
    let client = io.client.clone();
    unsafe {
        p0::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(ROWS as u32, 1, 1),
            CubeDim::new_3d(THREADS, 1, 1),
            BufferArg::from_raw_parts(io.x.handle.clone(), ROWS * D),
            BufferArg::from_raw_parts(io.w.handle.clone(), D),
            BufferArg::from_raw_parts(io.outc.handle.clone(), ROWS * D),
            D as u32,
        );
    }
    io.check("p0_three_buffers", &|r, i| io.x_at(r, i) * io.w_at(i));
}

// ── p1: + a runtime scalar through F::cast_from ────────────────────────────

#[cube(launch_unchecked)]
fn p1<F: Float>(x: &[F], w: &[F], out: &mut [F], eps: f32, #[comptime] d: u32) {
    let row = CUBE_POS_X as usize;
    let d = d as usize;
    if UNIT_POS_X == 0u32 {
        let mut i = 0;
        while i < d {
            out[row * d + i] = x[row * d + i] * w[i] * F::cast_from(eps + 1.0_f32);
            i += 1;
        }
    }
}

#[test]
fn p1_runtime_scalar() {
    let io = io();
    let client = io.client.clone();
    let eps = 1e-5f32;
    unsafe {
        p1::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(ROWS as u32, 1, 1),
            CubeDim::new_3d(THREADS, 1, 1),
            BufferArg::from_raw_parts(io.x.handle.clone(), ROWS * D),
            BufferArg::from_raw_parts(io.w.handle.clone(), D),
            BufferArg::from_raw_parts(io.outc.handle.clone(), ROWS * D),
            eps,
            D as u32,
        );
    }
    io.check("p1_runtime_scalar", &|r, i| io.x_at(r, i) * io.w_at(i) * (eps + 1.0));
}

// ── p2: + Shared::new_slice at a #[comptime] size, one write, ONE barrier ──

#[cube(launch_unchecked)]
fn p2<F: Float>(x: &[F], w: &[F], out: &mut [F], #[comptime] d: u32, #[comptime] threads: u32) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let d = d as usize;
    let threads = threads as usize;
    let base = row * d;
    let mut partial = Shared::<[F]>::new_slice(threads);
    partial[tid] = x[base];
    sync_cube();
    if tid == 0 {
        let mut i = 0;
        while i < d {
            out[base + i] = x[base + i] * w[i] * partial[0];
            i += 1;
        }
    }
}

#[test]
fn p2_shared_write_sync_read() {
    let io = io();
    let client = io.client.clone();
    unsafe {
        p2::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(ROWS as u32, 1, 1),
            CubeDim::new_3d(THREADS, 1, 1),
            BufferArg::from_raw_parts(io.x.handle.clone(), ROWS * D),
            BufferArg::from_raw_parts(io.w.handle.clone(), D),
            BufferArg::from_raw_parts(io.outc.handle.clone(), ROWS * D),
            D as u32,
            THREADS,
        );
    }
    io.check("p2_shared_write_sync_read", &|r, i| io.x_at(r, i) * io.w_at(i) * io.x_at(r, 0));
}

// ── p3: + a CONDITIONAL write to shared, a SECOND barrier, a read by ALL ──

#[cube(launch_unchecked)]
fn p3<F: Float>(x: &[F], w: &[F], out: &mut [F], eps: f32, #[comptime] d: u32, #[comptime] threads: u32) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let d = d as usize;
    let threads = threads as usize;
    let base = row * d;
    let mut partial = Shared::<[F]>::new_slice(threads);
    partial[tid] = x[base];
    sync_cube();
    if tid == 0 {
        partial[0] = (partial[0] / F::cast_from(d as f32) + F::cast_from(eps)).sqrt();
    }
    sync_cube();
    let inv = F::new(1.0_f32) / partial[0];
    let mut i = tid;
    while i < d {
        out[base + i] = x[base + i] * inv * w[i];
        i += threads;
    }
}

#[test]
fn p3_cond_write_then_bcast() {
    let io = io();
    let client = io.client.clone();
    let eps = 1e-5f32;
    unsafe {
        p3::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(ROWS as u32, 1, 1),
            CubeDim::new_3d(THREADS, 1, 1),
            BufferArg::from_raw_parts(io.x.handle.clone(), ROWS * D),
            BufferArg::from_raw_parts(io.w.handle.clone(), D),
            BufferArg::from_raw_parts(io.outc.handle.clone(), ROWS * D),
            eps,
            D as u32,
            THREADS,
        );
    }
    io.check_lowered("p3_cond_write_then_bcast");
}

// ── p4: NO SHARED AT ALL. A `while` loop with a barrier in it. ────────────
//
// The minimal suspect. If this lowers, a barrier in a loop is fine and the
// fault is in shared memory; if it does not, the fault is the barrier and this
// six-line kernel is the whole bug.

#[cube(launch_unchecked)]
fn p4<F: Float>(x: &[F], w: &[F], out: &mut [F], #[comptime] d: u32) {
    let row = CUBE_POS_X as usize;
    let d = d as usize;
    let base = row * d;
    if UNIT_POS_X == 0u32 {
        let mut s = 4u32;
        while s > 0 {
            sync_cube();
            s /= 2;
        }
        let mut i = 0;
        while i < d {
            out[base + i] = x[base + i] * w[i];
            i += 1;
        }
    }
}

#[test]
fn p4_barrier_in_while() {
    let io = io();
    let client = io.client.clone();
    unsafe {
        p4::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(ROWS as u32, 1, 1),
            CubeDim::new_3d(THREADS, 1, 1),
            BufferArg::from_raw_parts(io.x.handle.clone(), ROWS * D),
            BufferArg::from_raw_parts(io.w.handle.clone(), D),
            BufferArg::from_raw_parts(io.outc.handle.clone(), ROWS * D),
            D as u32,
        );
    }
    io.check("p4_barrier_in_while", &|r, i| io.x_at(r, i) * io.w_at(i));
}

// ── p5: the real kernel's WHOLE reduction, tree and all ───────────────────

#[cube(launch_unchecked)]
fn p5<F: Float>(x: &[F], w: &[F], out: &mut [F], eps: f32, #[comptime] d: u32, #[comptime] threads: u32) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let d = d as usize;
    let threads = threads as usize;
    let base = row * d;
    let mut partial = Shared::<[F]>::new_slice(threads);

    let mut sum = F::new(0.0_f32);
    let mut i = tid;
    while i < d {
        let v = x[base + i];
        sum += v * v;
        i += threads;
    }
    partial[tid] = sum;
    sync_cube();
    let mut s = threads / 2;
    while s > 0 {
        if tid < s {
            let other = partial[tid + s];
            partial[tid] += other;
        }
        sync_cube();
        s /= 2;
    }
    let inv = F::new(1.0_f32) / (partial[0] / F::cast_from(d as f32) + F::cast_from(eps)).sqrt();
    let mut i = tid;
    while i < d {
        out[base + i] = x[base + i] * inv * w[i];
        i += threads;
    }
}

#[test]
// THIS RUNG IS A REPRODUCER, NOT A TEST. It must NOT lower, so running it
// fails on purpose and that failure is the evidence. `#[ignore]` keeps
// `cargo test -p dormouse-rmsnorm` green for the next reader while the
// reproducer stays in the tree. To see it fail:
//   cargo test -p dormouse-rmsnorm --features cuda --test lower_probe -- --ignored --exact p5_tree_reduction
#[ignore = "must NOT lower: this is the minimal reproducer, see the header"]
fn p5_tree_reduction() {
    let io = io();
    let client = io.client.clone();
    let eps = 1e-5f32;
    unsafe {
        p5::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(ROWS as u32, 1, 1),
            CubeDim::new_3d(THREADS, 1, 1),
            BufferArg::from_raw_parts(io.x.handle.clone(), ROWS * D),
            BufferArg::from_raw_parts(io.w.handle.clone(), D),
            BufferArg::from_raw_parts(io.outc.handle.clone(), ROWS * D),
            eps,
            D as u32,
            THREADS,
        );
    }
    io.check("p5_tree_reduction", &|r, i| {
        let ss: f64 = (0..D).map(|k| f64::from(io.x_at(r, k)).powi(2)).sum();
        let inv = 1.0 / ((ss / D as f64) + f64::from(eps)).sqrt();
        (f64::from(io.x_at(r, i)) * inv * f64::from(io.w_at(i))) as f32
    });
}

// ── p6: p5 with the halving `while` REPLACED by thread 0's serial sum ─────
//
// The subtraction. p5 has the tree and p6 does not; everything else is shared,
// which is what makes the pair a bisect rather than two anecdotes.

#[cube(launch_unchecked)]
fn p6<F: Float>(x: &[F], w: &[F], out: &mut [F], eps: f32, #[comptime] d: u32, #[comptime] threads: u32) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let d = d as usize;
    let threads = threads as usize;
    let base = row * d;
    let mut partial = Shared::<[F]>::new_slice(threads);

    let mut sum = F::new(0.0_f32);
    let mut i = tid;
    while i < d {
        let v = x[base + i];
        sum += v * v;
        i += threads;
    }
    partial[tid] = sum;
    sync_cube();
    if tid == 0 {
        let mut a = 1;
        while a < threads {
            // Into a temp first: `partial[a] += partial[b]` would alias a
            // mutable and an immutable borrow of the same slice.
            let other = partial[a];
            partial[0] += other;
            a += 1;
        }
        partial[0] = (partial[0] / F::cast_from(d as f32) + F::cast_from(eps)).sqrt();
    }
    sync_cube();
    let inv = F::new(1.0_f32) / partial[0];
    let mut i = tid;
    while i < d {
        out[base + i] = x[base + i] * inv * w[i];
        i += threads;
    }
}

#[test]
fn p6_serial_reduction() {
    let io = io();
    let client = io.client.clone();
    let eps = 1e-5f32;
    unsafe {
        p6::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(ROWS as u32, 1, 1),
            CubeDim::new_3d(THREADS, 1, 1),
            BufferArg::from_raw_parts(io.x.handle.clone(), ROWS * D),
            BufferArg::from_raw_parts(io.w.handle.clone(), D),
            BufferArg::from_raw_parts(io.outc.handle.clone(), ROWS * D),
            eps,
            D as u32,
            THREADS,
        );
    }
    io.check("p6_serial_reduction", &|r, i| {
        let ss: f64 = (0..D).map(|k| f64::from(io.x_at(r, k)).powi(2)).sum();
        let inv = 1.0 / ((ss / D as f64) + f64::from(eps)).sqrt();
        (f64::from(io.x_at(r, i)) * inv * f64::from(io.w_at(i))) as f32
    });
}

// ── p7: `src/fused.rs`'s kernel, verbatim ─────────────────────────────────

#[cube(launch_unchecked)]
fn p7<F: Float>(
    x: &[F],
    w: &[F],
    out: &mut [F],
    eps: f32,
    #[comptime] d: u32,
    #[comptime] threads: u32,
) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let d = d as usize;
    let threads = threads as usize;
    let base = row * d;
    let mut partial = Shared::<[F]>::new_slice(threads);

    let mut sum = F::new(0.0_f32);
    let mut i = tid;
    while i < d {
        let v = x[base + i];
        sum += v * v;
        i += threads;
    }
    partial[tid] = sum;
    sync_cube();
    let mut s = threads / 2;
    while s > 0 {
        if tid < s {
            // Read into a temp first: cubecl shared-memory indexing is plain
            // Index/IndexMut, so `partial[a] += partial[b]` would alias a
            // mutable and an immutable borrow of the same slice.
            let other = partial[tid + s];
            partial[tid] += other;
        }
        sync_cube();
        s /= 2;
    }
    if tid == 0 {
        partial[0] = (partial[0] / F::cast_from(d as f32) + F::cast_from(eps)).sqrt();
    }
    sync_cube();
    let inv = F::new(1.0_f32) / partial[0];
    let mut i = tid;
    while i < d {
        out[base + i] = x[base + i] * inv * w[i];
        i += threads;
    }
}

#[test]
// THIS RUNG IS A REPRODUCER, NOT A TEST. It must NOT lower, so running it
// fails on purpose and that failure is the evidence. `#[ignore]` keeps
// `cargo test -p dormouse-rmsnorm` green for the next reader while the
// reproducer stays in the tree. To see it fail:
//   cargo test -p dormouse-rmsnorm --features cuda --test lower_probe -- --ignored --exact p7_full_kernel
#[ignore = "must NOT lower: this is the minimal reproducer, see the header"]
fn p7_full_kernel() {
    let io = io();
    let client = io.client.clone();
    let eps = 1e-5f32;
    unsafe {
        p7::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(ROWS as u32, 1, 1),
            CubeDim::new_3d(THREADS, 1, 1),
            BufferArg::from_raw_parts(io.x.handle.clone(), ROWS * D),
            BufferArg::from_raw_parts(io.w.handle.clone(), D),
            BufferArg::from_raw_parts(io.outc.handle.clone(), ROWS * D),
            eps,
            D as u32,
            THREADS,
        );
    }
    io.check("p7_full_kernel", &|r, i| {
        let ss: f64 = (0..D).map(|k| f64::from(io.x_at(r, k)).powi(2)).sum();
        let inv = 1.0 / ((ss / D as f64) + f64::from(eps)).sqrt();
        (f64::from(io.x_at(r, i)) * inv * f64::from(io.w_at(i))) as f32
    });
}

// ── p9: the tree's shared access, loop-varying index, NO barrier ──────────
//
// p5 fails and p4 (a barrier in a `while`, nothing indexed) passes, so the
// fault needs BOTH. This rung removes the barrier and keeps the loop-varying
// shared index, which is the subtraction that says whether the barrier is
// necessary to the failure or merely a bystander.

#[cube(launch_unchecked)]
fn p9<F: Float>(x: &[F], w: &[F], out: &mut [F], eps: f32, #[comptime] d: u32, #[comptime] threads: u32) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let d = d as usize;
    let threads = threads as usize;
    let base = row * d;
    let mut partial = Shared::<[F]>::new_slice(threads);

    let mut sum = F::new(0.0_f32);
    let mut i = tid;
    while i < d {
        let v = x[base + i];
        sum += v * v;
        i += threads;
    }
    partial[tid] = sum;
    sync_cube();
    let mut s = threads / 2;
    while s > 0 {
        if tid < s {
            let other = partial[tid + s];
            partial[tid] += other;
        }
        s /= 2;
    }
    let inv = F::new(1.0_f32) / (partial[tid] / F::cast_from(d as f32) + F::cast_from(eps)).sqrt();
    let mut i = tid;
    while i < d {
        out[base + i] = x[base + i] * inv * w[i];
        i += threads;
    }
}

#[test]
// THIS RUNG IS A REPRODUCER, NOT A TEST. It must NOT lower, so running it
// fails on purpose and that failure is the evidence. `#[ignore]` keeps
// `cargo test -p dormouse-rmsnorm` green for the next reader while the
// reproducer stays in the tree. To see it fail:
//   cargo test -p dormouse-rmsnorm --features cuda --test lower_probe -- --ignored --exact p9_tree_no_barrier_in_loop
#[ignore = "must NOT lower: this is the minimal reproducer, see the header"]
fn p9_tree_no_barrier_in_loop() {
    let io = io();
    let client = io.client.clone();
    let eps = 1e-5f32;
    unsafe {
        p9::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(ROWS as u32, 1, 1),
            CubeDim::new_3d(THREADS, 1, 1),
            BufferArg::from_raw_parts(io.x.handle.clone(), ROWS * D),
            BufferArg::from_raw_parts(io.w.handle.clone(), D),
            BufferArg::from_raw_parts(io.outc.handle.clone(), ROWS * D),
            eps,
            D as u32,
            THREADS,
        );
    }
    // RACE BY CONSTRUCTION — each thread reads its OWN `partial[tid]`, which is
    // only correct after the full tree. The value is not the point; the
    // lowering is.
    io.check_lowered("p9_tree_no_barrier_in_loop");
}

// ── p10: p5's tree in the form the WORKING kernel uses ───────────────────
//
// `dormouse-attnres/src/fused_attnres.rs:351-357` is a log-step reduction with a
// barrier in it, and it runs on every training step in this project. Its shape
// is `for k in 0..lg` — a loop whose trip count is `#[comptime]`, so cubecl
// unrolls it and every barrier lands in straight-line code. This rung is p5
// with exactly that change and nothing else. If it lowers while p5 does not,
// the fix for `src/fused.rs` is one loop header.

#[cube(launch_unchecked)]
fn p10<F: Float>(
    x: &[F],
    w: &[F],
    out: &mut [F],
    eps: f32,
    #[comptime] d: u32,
    #[comptime] threads: u32,
    #[comptime] log_threads: u32,
) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let d = d as usize;
    let threads = threads as usize;
    let lg = log_threads as usize;
    let base = row * d;
    let mut partial = Shared::<[F]>::new_slice(threads);

    let mut sum = F::new(0.0_f32);
    let mut i = tid;
    while i < d {
        let v = x[base + i];
        sum += v * v;
        i += threads;
    }
    partial[tid] = sum;
    sync_cube();
    // THE ONLY DIFFERENCE FROM p5: `while s > 0 { ... s /= 2 }` becomes
    // `for k in 0..lg` over a `#[comptime]` trip count, so cubecl unrolls it
    // and every barrier lands in straight-line code. `lg` is a PARAMETER and
    // not computed in the body — computing it here would make the trip count a
    // runtime value again and the rung would test nothing.
    for k in 0..lg {
        let s = threads >> (k + 1);
        if tid < s {
            let other = partial[tid + s];
            partial[tid] += other;
        }
        sync_cube();
    }
    if tid == 0 {
        partial[0] = (partial[0] / F::cast_from(d as f32) + F::cast_from(eps)).sqrt();
    }
    sync_cube();
    let inv = F::new(1.0_f32) / partial[0];
    let mut i = tid;
    while i < d {
        out[base + i] = x[base + i] * inv * w[i];
        i += threads;
    }
}

#[test]
fn p10_comptime_for_loop_tree() {
    let io = io();
    let client = io.client.clone();
    let eps = 1e-5f32;
    let log_threads = THREADS.trailing_zeros();
    unsafe {
        p10::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(ROWS as u32, 1, 1),
            CubeDim::new_3d(THREADS, 1, 1),
            BufferArg::from_raw_parts(io.x.handle.clone(), ROWS * D),
            BufferArg::from_raw_parts(io.w.handle.clone(), D),
            BufferArg::from_raw_parts(io.outc.handle.clone(), ROWS * D),
            eps,
            D as u32,
            THREADS,
            log_threads,
        );
    }
    io.check("p10_comptime_for_loop_tree", &|r, i| {
        let ss: f64 = (0..D).map(|k| f64::from(io.x_at(r, k)).powi(2)).sum();
        let inv = 1.0 / ((ss / D as f64) + f64::from(eps)).sqrt();
        (f64::from(io.x_at(r, i)) * inv * f64::from(io.w_at(i))) as f32
    });
}

// ── p8: `src/fused.rs`'s OWN kernel, through the crate's own launch ───────
//
// Every other rung launches by hand. This one goes through `RMSNorm::forward`
// -> `fused::rmsnorm_cuda`, so it is the only rung that can see a fault in the
// LAUNCH rather than in the body — and in particular the one difference the
// hand-launched rungs do not share with it: the crate passes its
// `BufferArg`s by MOVE (`from_raw_parts(x_c.handle, ..)`) while every rung here
// and `fused_attnres.rs:504,516` pass `.handle.clone()`.

#[test]
fn p8_the_crates_own_kernel_lowers() {
    let dev = Device::cuda(0);
    let x = Tensor::<3>::from_floats([[[1.0f32, 2.0, 3.0, 4.0]]], &dev);
    let got: Vec<f32> = dormouse_rmsnorm::RMSNorm::new(4, 1e-5, &dev)
        .forward(x)
        .into_data()
        .try_to_vec()
        .expect("the crate's own arm read back nothing: the kernel did not lower");
    eprintln!("the crate's own arm returned {got:?}");
    assert_eq!(got.len(), 4);
}

// ── p11: the SMALL-d shapes, and the envelope the fused arm now claims ────
//
// `tests/rmsnorm_kernel_cuda.rs` caught a 5.2e-1 relative error on the fixture
// case `d2` (`dims 1 2 2`, so two rows of TWO elements) that
// `fused_kernel_gate.rs` cannot see: it only tries d in {4, 8, 16, 32}. The
// arithmetic is right and the shapes are tiny, which is where a launch whose
// trailing cubes silently do not execute goes wrong. `tests/d2_isolate.rs` is
// the bisect; `fused.rs` answers it with a `d < MIN_FUSED_D` refusal.
//
// So the contract this rung pins is the ENVELOPE, both halves: `d < 4` the arm
// must DECLINE (and the tensor path must answer correctly), and `d >= 4` it
// must RUN and be right. A change that lifts the guard without fixing the
// underlying launch goes red here.

fn p11_run(x: &[f32], d: usize, w: &[f32], eps: f32, cuda: &Device) -> Option<Vec<f32>> {
    let rows = x.len() / d;
    let t = Tensor::<2>::from_data(TensorData::new(x.to_vec(), [rows, d]), cuda);
    dormouse_rmsnorm::fused::rmsnorm_cuda::<B>(
        t,
        Tensor::<1>::from_data(TensorData::new(w.to_vec(), [d]), cuda),
        eps,
    )
    .map(|o| o.into_data().try_to_vec().expect("the fused arm read back nothing"))
}

fn p11_worst(x: &[f32], d: usize, w: &[f32], eps: f32, got: &[f32]) -> f64 {
    let rows = x.len() / d;
    let mut worst = 0.0f64;
    for r in 0..rows {
        let row = &x[r * d..(r + 1) * d];
        let ss: f64 = row.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        for i in 0..d {
            let want = f64::from(row[i]) / (ss / d as f64 + f64::from(eps)).sqrt()
                * f64::from(w[i]);
            worst = worst.max((f64::from(got[r * d + i]) - want).abs() / want.abs().max(1.0));
        }
    }
    worst
}

#[test]
fn p11_the_claimed_envelope() {
    let eps = 1e-5f32;
    let cuda = Device::cuda(0);
    let cases: Vec<(usize, Vec<f32>, Vec<f32>)> = vec![
        // The two shapes that were WRONG, and the d=13 strided tail, each at a
        // mean-square where eps is decisive (1e-7 against eps=1e-5, so any error
        // in the sum is multiplied up).
        (1, vec![2.5e-5, -7.0e-5], vec![3.0]),
        (2, vec![0.0, 1.08484633e-3, 5.41788817e-4, -5.61883964e-4], vec![-2.5, 3.0]),
        (3, (0..9).map(|i| (i as f32 * 0.9).sin() * 2.0).collect(), vec![0.5, 0.75, 1.0]),
        // And two widths INSIDE the envelope, which must run and be right.
        (4, (0..12).map(|i| (i as f32 * 0.9).sin() * 2.0).collect(),
            (0..4).map(|i| 0.5 + 0.25 * i as f32).collect()),
        (13, (0..13).map(|i| (i as f32 * 0.9).sin() * 2.0).collect(),
            (0..13).map(|i| 0.5 + 0.25 * i as f32).collect()),
    ];
    for (d, x, w) in &cases {
        match p11_run(x, *d, w, eps, &cuda) {
            None => {
                assert!(
                    *d < dormouse_rmsnorm::fused::MIN_FUSED_D,
                    "the fused arm declined at d={d}, which is INSIDE the envelope it claims \
                     (d >= {}). Either the launch's trailing cubes have stopped executing and \
                     the guard should widen, or this decline is a regression.",
                    dormouse_rmsnorm::fused::MIN_FUSED_D
                );
                eprintln!("  d={d:<3} DECLINED (below MIN_FUSED_D) - correct, and counted");
            }
            Some(got) => {
                assert!(
                    *d >= dormouse_rmsnorm::fused::MIN_FUSED_D,
                    "the fused arm RAN at d={d}, below MIN_FUSED_D: the launch's trailing \
                     cubes do not execute there, so `got` is a partially unwritten tensor. \
                     This is the SILENT wrong answer ADR-0019 exists to prevent."
                );
                let worst = p11_worst(x, *d, w, eps, &got);
                eprintln!("  d={d:<3} RAN, worst {worst:e} vs the f64 definition");
                assert!(
                    worst < 1e-5,
                    "d={d}: the FUSED arm disagrees with the f64 definition by {worst:e}"
                );
            }
        }
    }
}
