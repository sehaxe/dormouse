// ISOLATE: at d=2 the trailing cubes are never written. Which part of the
// kernel drops them? Four reductions, identical tails, d=2 rows=4.
#![cfg(feature = "cuda")]

use burn::tensor::{Device, Tensor, TensorData};
use std::any::Any;

use burn_cubecl::tensor::CubeTensor;
use cubecl::client::Client;
use cubecl::prelude::*;

type B = burn_cubecl::CubeBackend;

const D: usize = 2;
const ROWS: usize = 4;
const THREADS: u32 = 256;
const LOG_THREADS: u32 = 8;

struct Io {
    out: Tensor<2>,
    outc: CubeTensor,
    client: Client,
    x: CubeTensor,
    w: CubeTensor,
    xvals: Vec<f32>,
    wvals: Vec<f32>,
}

fn down2(t: Tensor<2>) -> CubeTensor {
    let prim = t.clone().try_into_primitive::<B>().unwrap();
    (&prim as &dyn Any).downcast_ref::<CubeTensor>().unwrap().clone()
}
fn down1(t: Tensor<1>) -> CubeTensor {
    let prim = t.clone().try_into_primitive::<B>().unwrap();
    (&prim as &dyn Any).downcast_ref::<CubeTensor>().unwrap().clone()
}

fn crate_io() -> Io {
    let dev = Device::cuda(0);
    let xvals: Vec<f32> = (0..ROWS * D).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
    let wvals: Vec<f32> = (0..D).map(|i| 0.5 + 0.25 * i as f32).collect();
    let xt = Tensor::<2>::from_data(TensorData::new(xvals.clone(), [ROWS, D]), &dev);
    let wt = Tensor::<1>::from_data(TensorData::new(wvals.clone(), [D]), &dev);
    // SENTINEL, not `empty`: an uninitialised buffer makes "the cube never ran"
    // and "the cube wrote 0.0" indistinguishable, and those are different bugs.
    let outt = Tensor::<2>::full([ROWS, D], -999.0f32, &dev);
    let xc = down2(xt);
    let client = xc.client.clone();
    let wc = down1(wt);
    let outc = down2(outt.clone());
    Io { out: outt, outc, client, x: xc, w: wc, xvals, wvals }
}

impl Io {
    fn read(&self) -> Vec<f32> {
        self.out.clone().into_data().bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
    }
    fn want(&self) -> Vec<f32> {
        let eps = 1e-5f32;
        (0..ROWS).map(|r| {
            let row = &self.xvals[r * D..(r + 1) * D];
            let ss: f64 = row.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
            (0..D).map(|i| (f64::from(row[i]) / (ss / D as f64 + f64::from(eps)).sqrt()
                * f64::from(self.wvals[i])) as f32).collect::<Vec<_>>()
        }).flatten().collect()
    }
    fn report(&self, rung: &str) {
        let got = self.read();
        let want = self.want();
        let mut worst = 0.0f64;
        for (a, b) in got.iter().zip(&want) {
            worst = worst.max((f64::from(*a) - f64::from(*b)).abs() / f64::from(*b).abs().max(1.0));
        }
        eprintln!("{rung:<28} worst {worst:>10.3e}  got {got:?}");
    }
}

// A: the real kernel — tree reduction, conditional sqrt, broadcast.
#[cube(launch_unchecked)]
fn ka<F: Float>(x: &[F], w: &[F], out: &mut [F], eps: f32,
    #[comptime] d: u32, #[comptime] threads: u32, #[comptime] lg: u32) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let d = d as usize; let threads = threads as usize; let lg = lg as usize;
    let base = row * d;
    let mut partial = Shared::<[F]>::new_slice(threads);
    let mut sum = F::new(0.0_f32);
    let mut i = tid;
    while i < d { let v = x[base + i]; sum += v * v; i += threads; }
    partial[tid] = sum;
    sync_cube();
    for k in 0..lg {
        let s = threads >> (k + 1);
        if tid < s { let o = partial[tid + s]; partial[tid] += o; }
        sync_cube();
    }
    if tid == 0 { partial[0] = (partial[0] / F::cast_from(d as f32) + F::cast_from(eps)).sqrt(); }
    sync_cube();
    let inv = F::new(1.0_f32) / partial[0];
    let mut i = tid;
    while i < d { out[base + i] = x[base + i] * inv * w[i]; i += threads; }
}

// B: tree reduction, but NO conditional write — every thread does the sqrt on
// its own copy of partial[0].
#[cube(launch_unchecked)]
fn kb<F: Float>(x: &[F], w: &[F], out: &mut [F], eps: f32,
    #[comptime] d: u32, #[comptime] threads: u32, #[comptime] lg: u32) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let d = d as usize; let threads = threads as usize; let lg = lg as usize;
    let base = row * d;
    let mut partial = Shared::<[F]>::new_slice(threads);
    let mut sum = F::new(0.0_f32);
    let mut i = tid;
    while i < d { let v = x[base + i]; sum += v * v; i += threads; }
    partial[tid] = sum;
    sync_cube();
    for k in 0..lg {
        let s = threads >> (k + 1);
        if tid < s { let o = partial[tid + s]; partial[tid] += o; }
        sync_cube();
    }
    let inv = F::new(1.0_f32) /
        (partial[0] / F::cast_from(d as f32) + F::cast_from(eps)).sqrt();
    let mut i = tid;
    while i < d { out[base + i] = x[base + i] * inv * w[i]; i += threads; }
}

// C: NO shared memory, NO barrier — thread 0 does the whole row.
#[cube(launch_unchecked)]
fn kc<F: Float>(x: &[F], w: &[F], out: &mut [F], eps: f32, #[comptime] d: u32) {
    let row = CUBE_POS_X as usize;
    let d = d as usize;
    let base = row * d;
    if UNIT_POS_X == 0u32 {
        let mut s = F::new(0.0_f32);
        let mut i = 0;
        while i < d { let v = x[base + i]; s += v * v; i += 1; }
        let inv = F::new(1.0_f32) / (s / F::cast_from(d as f32) + F::cast_from(eps)).sqrt();
        let mut i = 0;
        while i < d { out[base + i] = x[base + i] * inv * w[i]; i += 1; }
    }
}

// D: shared reduce + broadcast, but the OUTPUT is written serially by thread 0
// (so the strided output loop is not in the picture at all).
#[cube(launch_unchecked)]
fn kd<F: Float>(x: &[F], w: &[F], out: &mut [F], eps: f32,
    #[comptime] d: u32, #[comptime] threads: u32, #[comptime] lg: u32) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let d = d as usize; let threads = threads as usize; let lg = lg as usize;
    let base = row * d;
    let mut partial = Shared::<[F]>::new_slice(threads);
    let mut sum = F::new(0.0_f32);
    let mut i = tid;
    while i < d { let v = x[base + i]; sum += v * v; i += threads; }
    partial[tid] = sum;
    sync_cube();
    for k in 0..lg {
        let s = threads >> (k + 1);
        if tid < s { let o = partial[tid + s]; partial[tid] += o; }
        sync_cube();
    }
    if tid == 0 { partial[0] = (partial[0] / F::cast_from(d as f32) + F::cast_from(eps)).sqrt(); }
    sync_cube();
    if tid == 0 {
        let inv = F::new(1.0_f32) / partial[0];
        let mut i = 0;
        while i < d { out[base + i] = x[base + i] * inv * w[i]; i += 1; }
    }
}

macro_rules! run {
    ($name:expr, $k:ident) => {{
        let t = crate_io();
        let client = t.client.clone();
        let eps = 1e-5f32;
        unsafe {
            $k::launch_unchecked::<f32>(
                &client,
                CubeCount::Static(ROWS as u32, 1, 1),
                CubeDim::new_3d(THREADS, 1, 1),
                BufferArg::from_raw_parts(t.x.handle.clone(), ROWS * D),
                BufferArg::from_raw_parts(t.w.handle.clone(), D),
                BufferArg::from_raw_parts(t.outc.handle.clone(), ROWS * D),
                eps,
                D as u32,
                THREADS,
                LOG_THREADS,
            );
        }
        t.report($name);
    }};
}

#[test]
fn isolate_d2() {
    eprintln!("=== d={D} rows={ROWS}: -999.0 means the cube NEVER WROTE ===");
    run!("A tree+cond+bcast+strided", ka);
    let t = crate_io();
    let client = t.client.clone();
    unsafe {
        kb::launch_unchecked::<f32>(
            &client, CubeCount::Static(ROWS as u32, 1, 1), CubeDim::new_3d(THREADS, 1, 1),
            BufferArg::from_raw_parts(t.x.handle.clone(), ROWS * D),
            BufferArg::from_raw_parts(t.w.handle.clone(), D),
            BufferArg::from_raw_parts(t.outc.handle.clone(), ROWS * D),
            1e-5, D as u32, THREADS, LOG_THREADS,
        );
    }
    t.report("B tree+no-cond+bcast+strided");
    let t = crate_io();
    let client = t.client.clone();
    unsafe {
        kc::launch_unchecked::<f32>(
            &client, CubeCount::Static(ROWS as u32, 1, 1), CubeDim::new_3d(THREADS, 1, 1),
            BufferArg::from_raw_parts(t.x.handle.clone(), ROWS * D),
            BufferArg::from_raw_parts(t.w.handle.clone(), D),
            BufferArg::from_raw_parts(t.outc.handle.clone(), ROWS * D),
            1e-5, D as u32,
        );
    }
    t.report("C no-shared, thread0 serial");
    run!("D tree+cond+bcast+serial-out", kd);
}

// ── The GRID, with nothing in the kernel but one store ────────────────────
//
// The sentinel proved cubes 2..3 of a 4-cube grid never ran, with a kernel that
// has no shared memory and no barrier. So the kernel body is exonerated and the
// grid is the suspect. This is one store per cube and nothing else.

#[cube(launch_unchecked)]
fn g1<F: Float>(out: &mut [F]) {
    let c = CUBE_POS_X as usize;
    out[c] = F::new(1.0_f32);
}

#[test]
fn grid_sweep() {
    let dev = Device::cuda(0);
    for &(ncubes, nthreads) in
        &[(4u32, 256u32), (4, 64), (4, 1), (8, 256), (12, 256), (2, 256), (3, 256), (5, 256)]
    {
        let t = Tensor::<1>::full([ncubes as usize], -999.0f32, &dev);
        let tc = down1(t.clone());
        let client = tc.client.clone();
        unsafe {
            g1::launch_unchecked::<f32>(
                &client,
                CubeCount::Static(ncubes, 1, 1),
                CubeDim::new_3d(nthreads, 1, 1),
                BufferArg::from_raw_parts(tc.handle.clone(), ncubes as usize),
            );
        }
        let got: Vec<f32> = t.into_data().try_to_vec().unwrap();
        let ran = got.iter().filter(|v| **v == 1.0).count();
        let mask: String = got
            .iter()
            .map(|v| if *v == 1.0 { '1' } else { '.' })
            .collect();
        eprintln!("  cubes={ncubes:<3} threads={nthreads:<4} ran {ran}/{ncubes}  {mask}");
    }
}

// ── Same one-store kernel, but the grid width arrives as a RUNTIME argument ─
//
// `rmsnorm_cuda` launches `CubeCount::Static(rows as u32, 1, 1)` where `rows`
// comes from `x.dims()` — a runtime value. The grid sweep above passed the
// count as a literal. This rung closes that gap, and it is the last structural
// difference between "the grid works" and "the grid works in this kernel".

#[cube(launch_unchecked)]
fn g2<F: Float>(out: &mut [F], ncubes: u32) {
    let c = CUBE_POS_X as usize;
    if UNIT_POS_X == 0u32 {
        out[c] = F::new(1.0_f32) + F::cast_from(ncubes as f32);
    }
}

#[test]
fn grid_runtime_count() {
    let dev = Device::cuda(0);
    for &ncubes in &[1u32, 2, 3, 4, 5, 8, 12, 16, 17, 31, 32, 64] {
        let t = Tensor::<1>::full([ncubes as usize], -999.0f32, &dev);
        let tc = down1(t.clone());
        let client = tc.client.clone();
        unsafe {
            g2::launch_unchecked::<f32>(
                &client,
                CubeCount::Static(ncubes, 1, 1),
                CubeDim::new_3d(THREADS, 1, 1),
                BufferArg::from_raw_parts(tc.handle.clone(), ncubes as usize),
                ncubes,
            );
        }
        let got: Vec<f32> = t.into_data().try_to_vec().unwrap();
        let want = 1.0 + ncubes as f32;
        let ran = got.iter().filter(|v| **v == want).count();
        let mask: String = got
            .iter()
            .map(|v| if *v == want { '1' } else { '.' })
            .collect();
        eprintln!("  runtime cubes={ncubes:<3} ran {ran}/{ncubes}  {mask}");
        assert_eq!(ran, ncubes as usize, "a {ncubes}-cube grid ran only {ran} cubes");
    }
}

// ── Bisect from the WORKING side: g2 works, kC does not. Walk g2 -> kC. ───

// g3: three buffers + comptime d + a row-dependent store. Still one store.
#[cube(launch_unchecked)]
fn g3<F: Float>(x: &[F], w: &[F], out: &mut [F], #[comptime] d: u32) {
    let row = CUBE_POS_X as usize;
    let d = d as usize;
    if UNIT_POS_X == 0u32 {
        out[row * d] = x[row * d] * w[0];
    }
}

// g4: g3 + a `while i < d` loop over the row (so TWO stores, by two threads or
// by one).
#[cube(launch_unchecked)]
fn g4<F: Float>(x: &[F], w: &[F], out: &mut [F], #[comptime] d: u32) {
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

// g5: g4 + the strided form (`i = tid`, `i += threads`) instead of serial.
#[cube(launch_unchecked)]
fn g5<F: Float>(x: &[F], w: &[F], out: &mut [F], #[comptime] d: u32, #[comptime] threads: u32) {
    let row = CUBE_POS_X as usize;
    let d = d as usize;
    let threads = threads as usize;
    let mut i = UNIT_POS_X as usize;
    while i < d {
        out[row * d + i] = x[row * d + i] * w[i];
        i += threads;
    }
}

// g6: g4 but the loop is over a RUNTIME d passed in, not `#[comptime]`.
#[cube(launch_unchecked)]
fn g6<F: Float>(x: &[F], w: &[F], out: &mut [F], d_rt: u32) {
    let row = CUBE_POS_X as usize;
    let d = d_rt as usize;
    if UNIT_POS_X == 0u32 {
        let mut i = 0;
        while i < d {
            out[row * d + i] = x[row * d + i] * w[i];
            i += 1;
        }
    }
}

#[test]
fn walk_from_working_side() {
    // Hand-rolled per variant, because each has a different signature.
    let dev = Device::cuda(0);
    let mk = |t: &Tensor<2>| down2(t.clone());
    for which in ["g3", "g4", "g5", "g6"] {
        let xvals: Vec<f32> = (0..ROWS * D).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        let wvals: Vec<f32> = (0..D).map(|i| 0.5 + 0.25 * i as f32).collect();
        let t = Tensor::<2>::full([ROWS, D], -999.0f32, &dev);
        let xc = mk(&Tensor::<2>::from_data(TensorData::new(xvals, [ROWS, D]), &dev));
        let wc = down1(Tensor::<1>::from_data(TensorData::new(wvals, [D]), &dev));
        let tc = down2(t.clone());
        let client = xc.client.clone();
        let grid = (CubeCount::Static(ROWS as u32, 1, 1), CubeDim::new_3d(THREADS, 1, 1));
        unsafe {
            let a = BufferArg::from_raw_parts(xc.handle.clone(), ROWS * D);
            let b = BufferArg::from_raw_parts(wc.handle.clone(), D);
            let c = BufferArg::from_raw_parts(tc.handle.clone(), ROWS * D);
            match which {
                "g3" => g3::launch_unchecked::<f32>(&client, grid.0, grid.1, a, b, c, D as u32),
                "g4" => g4::launch_unchecked::<f32>(&client, grid.0, grid.1, a, b, c, D as u32),
                "g5" => {
                    g5::launch_unchecked::<f32>(&client, grid.0, grid.1, a, b, c, D as u32, THREADS)
                }
                _ => g6::launch_unchecked::<f32>(&client, grid.0, grid.1, a, b, c, D as u32),
            };
        }
        let got: Vec<f32> = t.into_data().try_to_vec().unwrap();
        let mask: String = got
            .iter()
            .map(|v| if *v == -999.0 { '.' } else { '1' })
            .collect();
        eprintln!("  {which}: wrote {mask}   raw {got:?}");
    }
}

// ── g1/g2 wrote ALL FOUR cubes with ONE buffer. g3 wrote TWO of four with
// THREE buffers and a row-dependent index. Which of the two is it? ──────────

#[cube(launch_unchecked)]
fn h2<F: Float>(_x: &[F], out: &mut [F]) {
    out[CUBE_POS_X as usize] = F::new(1.0_f32);
}

#[cube(launch_unchecked)]
fn h3<F: Float>(x: &[F], w: &[F], out: &mut [F]) {
    out[CUBE_POS_X as usize] = x[CUBE_POS_X as usize] * w[0];
}

#[cube(launch_unchecked)]
fn h4<F: Float>(x: &[F], _w: &[F], out: &mut [F], #[comptime] d: u32) {
    let row = CUBE_POS_X as usize;
    let d = d as usize;
    out[row * d] = x[row * d];
}

#[test]
fn buffer_count_vs_row_index() {
    let dev = Device::cuda(0);
    let xvals: Vec<f32> = (0..ROWS * D).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
    let wvals: Vec<f32> = (0..D).map(|i| 0.5 + 0.25 * i as f32).collect();
    let xc = down2(Tensor::<2>::from_data(TensorData::new(xvals, [ROWS, D]), &dev));
    let wc = down1(Tensor::<1>::from_data(TensorData::new(wvals, [D]), &dev));
    let client = xc.client.clone();
    let g = || (CubeCount::Static(ROWS as u32, 1, 1), CubeDim::new_3d(THREADS, 1, 1));

    let show = |t: &Tensor<2>, name: &str| {
        let got: Vec<f32> = t.clone().into_data().try_to_vec().unwrap();
        let mask: String = got
            .iter()
            .map(|v| if *v == -999.0 { '.' } else { '1' })
            .collect();
        eprintln!("  {name:<34} wrote {mask}   raw {got:?}");
    };

    // h2: TWO buffers, no row indexing.
    let t = Tensor::<2>::full([ROWS, D], -999.0f32, &dev);
    let tc = down2(t.clone());
    unsafe {
        h2::launch_unchecked::<f32>(
            &client,
            g().0,
            g().1,
            BufferArg::from_raw_parts(xc.handle.clone(), ROWS * D),
            BufferArg::from_raw_parts(tc.handle.clone(), ROWS * D),
        );
    }
    show(&t, "h2 2 buffers, out[CUBE_POS_X]");

    // h3: THREE buffers, no row indexing.
    let t = Tensor::<2>::full([ROWS, D], -999.0f32, &dev);
    let tc = down2(t.clone());
    unsafe {
        h3::launch_unchecked::<f32>(
            &client,
            g().0,
            g().1,
            BufferArg::from_raw_parts(xc.handle.clone(), ROWS * D),
            BufferArg::from_raw_parts(wc.handle.clone(), D),
            BufferArg::from_raw_parts(tc.handle.clone(), ROWS * D),
        );
    }
    show(&t, "h3 3 buffers, out[CUBE_POS_X]");

    // h4: THREE buffers, row*comptime-d indexing.
    let t = Tensor::<2>::full([ROWS, D], -999.0f32, &dev);
    let tc = down2(t.clone());
    unsafe {
        h4::launch_unchecked::<f32>(
            &client,
            g().0,
            g().1,
            BufferArg::from_raw_parts(xc.handle.clone(), ROWS * D),
            BufferArg::from_raw_parts(wc.handle.clone(), D),
            BufferArg::from_raw_parts(tc.handle.clone(), ROWS * D),
            D as u32,
        );
    }
    show(&t, "h4 3 buffers, out[row*d]");
}

// ── Two buffers, ONE store at out[CUBE_POS_X], sweeping the out LENGTH with a
// fixed 4-cube grid. If the grid is being truncated, the count of cubes that
// ran will be a function of the length, and the length is the only thing that
// changed between g1 (4, all four ran) and h2 (8, two ran). ───────────────

#[cube(launch_unchecked)]
fn h5<F: Float>(_x: &[F], out: &mut [F]) {
    out[CUBE_POS_X as usize] = F::new(1.0_f32);
}

#[test]
fn two_buffers_length_sweep() {
    let dev = Device::cuda(0);
    for &len in &[4usize, 5, 6, 7, 8, 9, 12, 16, 32] {
        let xc = down1(Tensor::<1>::full([len], 1.0f32, &dev));
        let t = Tensor::<1>::full([len], -999.0f32, &dev);
        let tc = down1(t.clone());
        let client = xc.client.clone();
        unsafe {
            h5::launch_unchecked::<f32>(
                &client,
                CubeCount::Static(4, 1, 1),
                CubeDim::new_3d(THREADS, 1, 1),
                BufferArg::from_raw_parts(xc.handle.clone(), len),
                BufferArg::from_raw_parts(tc.handle.clone(), len),
            );
        }
        let got: Vec<f32> = t.into_data().try_to_vec().unwrap();
        let ran = got.iter().filter(|v| **v == 1.0).count();
        let mask: String = got.iter().map(|v| if *v == 1.0 { '1' } else { '.' }).collect();
        eprintln!("  2 buffers, out len {len:>3}, grid 4 cubes: ran {ran}/4  {mask}");
    }
    // And the one-buffer control, same sweep.
    for &len in &[4usize, 8, 16, 32] {
        let t = Tensor::<1>::full([len], -999.0f32, &dev);
        let tc = down1(t.clone());
        let client = tc.client.clone();
        unsafe {
            h5::launch_unchecked::<f32>(
                &client,
                CubeCount::Static(4, 1, 1),
                CubeDim::new_3d(THREADS, 1, 1),
                BufferArg::from_raw_parts(tc.handle.clone(), len),
                BufferArg::from_raw_parts(tc.handle.clone(), len),
            );
        }
        let got: Vec<f32> = t.into_data().try_to_vec().unwrap();
        let ran = got.iter().filter(|v| **v == 1.0).count();
        eprintln!("  1 buffer , out len {len:>3}, grid 4 cubes: ran {ran}/4");
    }
}

// ── The only difference left between h5 (4/4 cubes ran) and h4 (2/4): the
// RANK of the output tensor. Same kernel, same grid, same lengths. ─────────

#[cube(launch_unchecked)]
fn j1<F: Float>(x: &[F], w: &[F], out: &mut [F], #[comptime] d: u32) {
    let row = CUBE_POS_X as usize;
    let d = d as usize;
    if UNIT_POS_X == 0u32 {
        out[row * d] = F::new(1.0_f32) + x[row * d] + w[0];
    }
}

#[test]
fn output_rank_matters() {
    let dev = Device::cuda(0);
    let xvals: Vec<f32> = (0..ROWS * D).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
    let wvals: Vec<f32> = (0..D).map(|i| 0.5 + 0.25 * i as f32).collect();
    let xc = down2(Tensor::<2>::from_data(TensorData::new(xvals, [ROWS, D]), &dev));
    let wc = down1(Tensor::<1>::from_data(TensorData::new(wvals, [D]), &dev));
    let client = xc.client.clone();
    let g = || (CubeCount::Static(ROWS as u32, 1, 1), CubeDim::new_3d(THREADS, 1, 1));
    let n = ROWS * D;

    // j1 with a 1-D output of the SAME element count.
    let t1 = Tensor::<1>::full([n], -999.0f32, &dev);
    let t1c = down1(t1.clone());
    unsafe {
        j1::launch_unchecked::<f32>(
            &client,
            g().0,
            g().1,
            BufferArg::from_raw_parts(xc.handle.clone(), n),
            BufferArg::from_raw_parts(wc.handle.clone(), D),
            BufferArg::from_raw_parts(t1c.handle.clone(), n),
            D as u32,
        );
    }
    let got1: Vec<f32> = t1.into_data().try_to_vec().unwrap();
    let m1: String = got1.iter().map(|v| if *v == -999.0 { '.' } else { '1' }).collect();
    eprintln!("  j1 with a 1-D out [{}]: {m1}  {got1:?}", n);

    // j1 with a 2-D output of shape [ROWS, D] — the identical launch otherwise.
    let t2 = Tensor::<2>::full([ROWS, D], -999.0f32, &dev);
    let t2c = down2(t2.clone());
    unsafe {
        j1::launch_unchecked::<f32>(
            &client,
            g().0,
            g().1,
            BufferArg::from_raw_parts(xc.handle.clone(), n),
            BufferArg::from_raw_parts(wc.handle.clone(), D),
            BufferArg::from_raw_parts(t2c.handle.clone(), n),
            D as u32,
        );
    }
    let got2: Vec<f32> = t2.into_data().try_to_vec().unwrap();
    let m2: String = got2.iter().map(|v| if *v == -999.0 { '.' } else { '1' }).collect();
    eprintln!("  j1 with a 2-D out [{},{}]: {m2}  {got2:?}", ROWS, D);

    // And a 2-D output that is NOT 2x2: [8, 1] — same element count again.
    let t3 = Tensor::<2>::full([n, 1], -999.0f32, &dev);
    let t3c = down2(t3.clone());
    unsafe {
        j1::launch_unchecked::<f32>(
            &client,
            g().0,
            g().1,
            BufferArg::from_raw_parts(xc.handle.clone(), n),
            BufferArg::from_raw_parts(wc.handle.clone(), D),
            BufferArg::from_raw_parts(t3c.handle.clone(), n),
            D as u32,
        );
    }
    let got3: Vec<f32> = t3.into_data().try_to_vec().unwrap();
    let m3: String = got3.iter().map(|v| if *v == -999.0 { '.' } else { '1' }).collect();
    eprintln!("  j1 with a 2-D out [{n},1]: {m3}  {got3:?}");
}

// ── RACE? Read the SAME buffer three times: immediately, after an explicit
// client.sync(), and after a second sync. A race shows up as the three reads
// disagreeing. ──────────────────────────────────────────────────────────────

#[cube(launch_unchecked)]
fn j2<F: Float>(x: &[F], w: &[F], out: &mut [F], #[comptime] d: u32) {
    let row = CUBE_POS_X as usize;
    let d = d as usize;
    if UNIT_POS_X == 0u32 {
        out[row * d] = F::new(1.0_f32) + x[row * d] + w[0];
    }
}

#[test]
fn is_it_a_race() {
    let dev = Device::cuda(0);
    let xvals: Vec<f32> = (0..ROWS * D).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
    let wvals: Vec<f32> = (0..D).map(|i| 0.5 + 0.25 * i as f32).collect();
    let xc = down2(Tensor::<2>::from_data(TensorData::new(xvals, [ROWS, D]), &dev));
    let wc = down1(Tensor::<1>::from_data(TensorData::new(wvals, [D]), &dev));
    let client = xc.client.clone();
    let n = ROWS * D;
    let t = Tensor::<2>::full([ROWS, D], -999.0f32, &dev);
    let tc = down2(t.clone());
    unsafe {
        j2::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(ROWS as u32, 1, 1),
            CubeDim::new_3d(THREADS, 1, 1),
            BufferArg::from_raw_parts(xc.handle.clone(), n),
            BufferArg::from_raw_parts(wc.handle.clone(), D),
            BufferArg::from_raw_parts(tc.handle.clone(), n),
            D as u32,
        );
    }
    for round in 1..=3 {
        let got: Vec<f32> = t.clone().into_data().try_to_vec().unwrap();
        eprintln!("  read {round} (no explicit sync): {got:?}");
    }
    futures_lite::future::block_on(client.sync()).expect("sync");
    let got: Vec<f32> = t.clone().into_data().try_to_vec().unwrap();
    eprintln!("  read 4 (after client.sync()):  {got:?}");
    eprintln!("  expected 1.5 + x[{{0,2,4,6}}] = {:?}", {
        let x: Vec<f32> = (0..ROWS * D).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        (0..ROWS).map(|r| 1.5 + x[r * D]).collect::<Vec<f32>>()
    });
}

// ── Characterise the SECOND bug: 2 of 4 cubes run, deterministically, with a
// 4-cube grid and a 4-line kernel. Sweep the OUTPUT SHAPE at a fixed element
// count. ─────────────────────────────────────────────────────────────────────

#[test]
fn output_shape_sweep() {
    let dev = Device::cuda(0);
    let n = 16usize;
    let xvals: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
    let xc = down1(Tensor::<1>::from_data(TensorData::new(xvals, [n]), &dev));
    let client = xc.client.clone();
    // 4 cubes, each storing ONE f32 at its own CUBE_POS_X. No shared memory, no
    // barrier, no comptime d, no row indexing. If this drops cubes, the cube
    // id itself is being clamped.
    for &shape in &[
        [1usize, 16],
        [2, 8],
        [4, 4],
        [8, 2],
        [16, 1],
        [2, 1],
        [3, 1],
        [4, 1],
        [1, 4],
        [1, 2],
        [1, 1],
        [5, 1],
    ] {
        let t = Tensor::<2>::full(shape, -999.0f32, &dev);
        let tc = down2(t.clone());
        unsafe {
            h5::launch_unchecked::<f32>(
                &client,
                CubeCount::Static(4, 1, 1),
                CubeDim::new_3d(THREADS, 1, 1),
                BufferArg::from_raw_parts(xc.handle.clone(), n),
                BufferArg::from_raw_parts(tc.handle.clone(), n),
            );
        }
        let got: Vec<f32> = t.into_data().try_to_vec().unwrap();
        let ran = got.iter().filter(|v| **v == 1.0).count();
        let mask: String = got
            .iter()
            .take(8)
            .map(|v| if *v == 1.0 { '1' } else { '.' })
            .collect();
        eprintln!(
            "  out shape [{:>2},{:>2}] (len {n}), 4 cubes: ran {ran}/4  first8 {mask}",
            shape[0], shape[1]
        );
    }
}

// ── Does the BLOCK SIZE change the truncation? If a smaller CubeDim runs all
// the cubes, the fix is a launch parameter rather than a refusal. ──────────

#[test]
fn cubedim_sweep_on_failing_shapes() {
    let dev = Device::cuda(0);
    let n = 16usize;
    let xvals: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
    let xc = down1(Tensor::<1>::from_data(TensorData::new(xvals, [n]), &dev));
    let client = xc.client.clone();
    for &shape in &[[8usize, 2usize], [4, 2], [2, 1], [1, 1]] {
        for &dim in &[32u32, 64, 128, 256, 512, 1024] {
            for &ncubes in &[4u32] {
                let t = Tensor::<2>::full(shape, -999.0f32, &dev);
                let tc = down2(t.clone());
                unsafe {
                    h5::launch_unchecked::<f32>(
                        &client,
                        CubeCount::Static(ncubes, 1, 1),
                        CubeDim::new_3d(dim, 1, 1),
                        BufferArg::from_raw_parts(xc.handle.clone(), n),
                        BufferArg::from_raw_parts(tc.handle.clone(), n),
                    );
                }
                let got: Vec<f32> = t.into_data().try_to_vec().unwrap();
                let ran = got.iter().filter(|v| **v == 1.0).count();
                eprintln!(
                    "  shape [{:>2},{:>2}] dim {dim:>4} grid {ncubes}: ran {ran}/{ncubes}{}",
                    shape[0],
                    shape[1],
                    if ran == ncubes as usize { "" } else { "   <-- TRUNCATED" }
                );
            }
        }
    }
}
