//! The bf16 capability probe: can a cubecl CUDA kernel on THIS stack read and
//! write bf16 STORAGE while accumulating in f32?
//!
//! The fused chunk kernels are `F: Float` generic, but the CUDA backend they
//! run on (cubecl 0.11.0-pre.4, LLVM/nvptx) has no bf16 element type at all:
//! `restrict_to_llvm_backend` in `cubecl-cuda/src/runtime.rs` drops it from the
//! advertised features ("the dialect this backend lowers through has no type
//! for them, so there is nothing to put in a register"), the nvptx lowering in
//! `cubecl-llvm` has no bf16 rules (bf16 exists only in the amdgpu matrix
//! path), and asking for one fails at kernel-compile time:
//!
//!   compiling 'cast_element_i_f32_o_bf16_n_4' for sm_120: Compilation error:
//!   invalid input program.
//!   Type cube.bf16 does not have a conversion to LLVM type implemented
//!
//! i.e. not just "no bf16 math" - burn cannot compile a kernel that WRITES a
//! bf16 buffer, so the `--bf16` mode dies on its first `cast(BF16)` (see
//! `bf16_cast_does_not_compile`, ignored by design: it is a report, not a
//! requirement).
//!
//! What DOES work is bf16 storage reached as `u16` bit patterns: a bf16 is the
//! top 16 bits of the f32 it came from, so `f32::reinterpret(u32 << 16)` and
//! the round-to-nearest-even truncation back are integer ops plus a bitcast -
//! no bf16 value ever enters the dialect. That is the primitive a bf16 fused
//! chunk kernel would be built from, so this file pins it down against f64 on
//! the host: a measured kernel, not a hope.
//!
//! The second half pins the CURRENT fallback surface: the fused chunk entry
//! point takes f32 and refuses everything else, and what that costs on the
//! production shape.
//!
//! Run: cargo test --release -p burn-gdn2 --features cuda --test lowp_bf16_cuda -- --nocapture --test-threads=1

#![cfg(feature = "cuda")]

use burn::backend::{Backend, DispatchKindConversion};
use burn::tensor::{Device, DispatchTensor, Distribution, FloatDType, Tensor};
use burn_gdn2::kernel::chunk_cube::cuda::fused_chunk_forward_scratch;
use burn_gdn2::CudaBare;
use cubecl::prelude::*;
use std::any::Any;

type B = CudaBare;

/// Rows x K dot product: bf16 storage in, f32 accumulation, f32 out, plus an
/// f32 -> bf16 round trip of the result into a third buffer.
///
/// Every buffer is typed `u16` (the bf16 bit pattern) or `f32` - the kernel
/// contains no bf16 value, so the LLVM backend can lower all of it.
#[cube(launch_unchecked)]
fn bf16_dot_kernel(
    lhs: &[u16],     // [rows, k] bf16 bits
    rhs: &[u16],     // [rows, k] bf16 bits
    out: &mut [f32], // [rows] the f32 accumulator result
    rt: &mut [u16],  // [2*rows] f32 -> bf16, round-to-nearest-even; one word
    // per f32 element's low half, so the host readback stays
    // a plain f32 read
    #[comptime] k: u32,
) {
    let row = CUBE_POS_X as usize;
    let ku = k as usize;
    let mut acc = f32::new(0.0_f32);
    let mut i = 0;
    while i < ku {
        // bf16 -> f32: the bf16 word IS the high half of the f32.
        let a = f32::reinterpret(u32::cast_from(lhs[row * ku + i]) << 16u32);
        let b = f32::reinterpret(u32::cast_from(rhs[row * ku + i]) << 16u32);
        acc += a * b;
        i += 1;
    }
    out[row] = acc;
    // f32 -> bf16, round-to-nearest-even: add half an ulp, then break the tie
    // to even on the truncated bit (the standard RNE idiom).
    let u = u32::reinterpret(acc);
    let r = u + 0x7fff_u32 + ((u >> 16u32) & 1u32);
    rt[row * 2] = u16::cast_from(r >> 16u32);
}

/// Owned copy of the underlying `CubeTensor` of `t`: the buffer is all the
/// kernel gets, and this test deliberately hands a kernel the bytes of a
/// dtype the backend cannot type.
fn cube_of<B: Backend, const D: usize>(t: &Tensor<D>) -> burn_cubecl::tensor::CubeTensor
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let prim = t.clone().try_into_primitive::<B>().ok().expect("inner");
    let cube = (&prim as &dyn Any)
        .downcast_ref::<burn_cubecl::tensor::CubeTensor>()
        .expect("cube tensor");
    cube.clone()
}

/// Host-side bf16 rounding: the reference the kernel must match bit for bit.
fn bf16(x: f32) -> half::bf16 {
    half::bf16::from_f32(x)
}

#[test]
fn bf16_storage_with_f32_accumulation() {
    let dev: Device = Default::default();
    dev.seed(7);
    let (rows, k) = (64usize, 128usize);
    let a = Tensor::<2>::random([rows, k], Distribution::Default, &dev);
    let b = Tensor::<2>::random([rows, k], Distribution::Default, &dev);
    let av: Vec<f32> = a.clone().into_data().try_to_vec().expect("f32");
    let bv: Vec<f32> = b.clone().into_data().try_to_vec().expect("f32");

    // bf16-rounded operands, packed two per 32-bit word and uploaded through an
    // f32 tensor as raw bits (from_data is an H2D memcpy, it computes nothing).
    // Building the buffer out of words rather than a burn bf16 tensor is the
    // whole point: burn cannot compile a kernel that writes bf16 on this
    // backend, but a buffer of bf16 bits is just bytes, and a kernel may read
    // or write it as u16 (little-endian: lo | hi << 16).
    let pack = |v: &[f32]| -> Vec<f32> {
        v.chunks(2)
            .map(|p| {
                let lo = u32::from(u16::from_le_bytes(bf16(p[0]).to_le_bytes()));
                let hi = u32::from(u16::from_le_bytes(
                    bf16(*p.get(1).unwrap_or(&0.0)).to_le_bytes(),
                ));
                f32::from_bits(lo | (hi << 16))
            })
            .collect()
    };
    let upload = |words: Vec<f32>| {
        let n2 = words.len();
        Tensor::<2>::from_data(burn::tensor::TensorData::new(words, [n2, 1]), &dev)
    };
    let a_packed = upload(pack(&av));
    let b_packed = upload(pack(&bv));
    let out = Tensor::<2>::zeros([rows, 1], &dev);
    let rt = Tensor::<2>::zeros([rows, 1], &dev);
    let a_c = cube_of::<B, 2>(&a_packed);
    let b_c = cube_of::<B, 2>(&b_packed);
    let out_c = cube_of::<B, 2>(&out);
    let rt_c = cube_of::<B, 2>(&rt);
    let client = a_c.client.clone();
    unsafe {
        bf16_dot_kernel::launch_unchecked(
            &client,
            CubeCount::Static(rows as u32, 1, 1),
            CubeDim::new_1d(32),
            BufferArg::from_raw_parts(a_c.handle, rows * k),
            BufferArg::from_raw_parts(b_c.handle, rows * k),
            BufferArg::from_raw_parts(out_c.handle, rows),
            BufferArg::from_raw_parts(rt_c.handle, 2 * rows),
            k as u32,
        );
    }

    let got: Vec<f32> = out.clone().into_data().try_to_vec().expect("f32");
    // `rt` is an f32-typed burn buffer whose first two bytes per element the
    // kernel wrote as a u16 bf16 word (the upper half is still the zeros it was
    // allocated with), so the word is the low half of the f32 bit pattern.
    let rt_got: Vec<u16> = rt
        .clone()
        .into_data()
        .try_to_vec::<f32>()
        .expect("f32")
        .iter()
        .map(|x| (x.to_bits() & 0xFFFF) as u16)
        .collect();

    // f64 reference over the SAME (bf16-rounded) operands: this isolates the
    // f32 accumulator's error from the intended bf16 storage error.
    let mut want = vec![0f64; rows];
    for r in 0..rows {
        let mut s = 0f64;
        for i in 0..k {
            s += f64::from(f32::from(bf16(av[r * k + i])))
                * f64::from(f32::from(bf16(bv[r * k + i])));
        }
        want[r] = s;
    }
    let max_rel = want.iter().zip(got.iter()).fold(0f64, |m, (w, g)| {
        m.max((w - *g as f64).abs() / w.abs().max(1e-6))
    });
    println!("bf16 storage + f32 acc: max rel err vs f64 = {max_rel:.3e} over {rows}x{k} dots");
    // bf16 eps is 2^-8 = 3.9e-3. Landing under 1e-5 proves the accumulator is
    // f32 and the operands were expanded to f32, not accumulated in bf16.
    assert!(
        max_rel < 1e-5,
        "f32 accumulation over bf16 operands should be f32-accurate, got {max_rel:e}"
    );

    // f32 -> bf16 is bit-exact against the host RNE rounding, ties included.
    // The bf16 word is the HIGH half of the f32 the host rounds to.
    for r in 0..rows {
        let host = f32::from(half::bf16::from_f32(got[r]));
        let host_word = (host.to_bits() >> 16) as u16;
        assert_eq!(
            rt_got[r], host_word,
            "row {r}: kernel bf16 word {:#06x} != host RNE {:#06x} ({host:e})",
            rt_got[r], host_word,
        );
    }
    println!("f32 -> bf16 round trip: bit-exact vs half::bf16::from_f32 on {rows} rows");
}

/// The current fallback surface, executable. The one dtype gate that decides
/// fused kernels vs the ~150-op tensor chunk loop lives in
/// `chunk_cube::cuda::fused_chunk_forward_scratch`.
#[test]
fn fused_chunk_gate_is_f32_only() {
    let dev: Device = Default::default();
    dev.seed(11);
    let (b, h, t, kd) = (2usize, 4usize, 64usize, 64usize);
    let mk = || {
        (
            Tensor::<4>::random([b, h, t, kd], Distribution::Default, &dev),
            Tensor::<4>::random([b, h, t, kd], Distribution::Default, &dev),
            Tensor::<4>::random([b, h, t, kd], Distribution::Default, &dev),
            Tensor::<4>::random([b, h, t, kd], Distribution::Default, &dev)
                .neg()
                .div_scalar(2.0),
            Tensor::<4>::random([b, h, t, kd], Distribution::Default, &dev),
            Tensor::<4>::random([b, h, t, kd], Distribution::Default, &dev),
        )
    };
    let state = Tensor::<4>::zeros([b, h, kd, kd], &dev);
    let cs = 16usize;

    let f = mk();
    let fused_f32 =
        fused_chunk_forward_scratch::<B>(f.0, f.1, f.2, f.3, f.4, f.5, state.clone(), 1.0, cs);
    assert!(fused_f32.is_some(), "f32 must take the fused kernels");
    println!("fused gate: f32            -> fused kernels (3 launches per sequence)");

    // bf16 inputs are legal burn tensors (dormouse keeps its residual stream in
    // bf16) but cannot be kernel arguments, so the gate refuses them and the
    // dispatcher runs the tensor-ops chunk path.
    let f = mk();
    let bf = |t: Tensor<4>| t.clone().cast(FloatDType::BF16);
    let fused_bf16 = fused_chunk_forward_scratch::<B>(
        bf(f.0),
        bf(f.1),
        bf(f.2),
        bf(f.3),
        bf(f.4),
        bf(f.5),
        state.clone().cast(FloatDType::BF16),
        1.0,
        cs,
    );
    assert!(
        fused_bf16.is_none(),
        "bf16 is expected to fall back to the tensor-ops chunk path today"
    );
    println!("fused gate: bf16           -> tensor-ops fallback");

    // The other fallback is numerical, not a dtype one: chunk > 16 underflows
    // the K/exp(cumsum g) factor in f32.
    let big = 32usize;
    let f = mk();
    let f_big = fused_chunk_forward_scratch::<B>(f.0, f.1, f.2, f.3, f.4, f.5, state, 1.0, big);
    assert!(f_big.is_none(), "chunk 32 must fall back (underflow limit)");
    println!("fused gate: chunk 32       -> tensor-ops fallback (f32 underflow below -88)");
}

/// What the dtype gate costs on the production shape (b=10, t=512, 12 heads,
/// K=V=64, chunk 16), and what the `--bf16` boundary buys.
///
/// Every arm syncs the device inside the timed region: the cubecl backend is
/// async, so `Instant` around a forward only measures the enqueue.
#[test]
fn production_shape_cost_of_the_dtype_gate() {
    const B: usize = 10;
    const H: usize = 12;
    const T: usize = 512;
    const K: usize = 64;
    const CS: usize = 16;
    let dev: Device = Default::default();
    dev.seed(3);
    let mk = || Tensor::<4>::random([B, H, T, K], Distribution::Default, &dev);
    let g = mk().neg().div_scalar(2.0); // log-decay, bounded like the K3 floor
    let q = mk();
    let k = mk();
    let v = mk();
    let b = mk();
    let w = mk();
    let state = Tensor::<4>::zeros([B, H, K, K], &dev);

    // 5 reps, report the min (least contended) and the median: the tensor arm
    // is launch-bound, so its cost is a function of how busy the CPU is, and a
    // single mean hides that.
    let time = |f: &mut dyn FnMut()| -> (f32, f32) {
        f();
        let reps = 5;
        let mut ms = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t0 = std::time::Instant::now();
            f();
            ms.push(t0.elapsed().as_secs_f32() * 1000.0);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (ms[0], ms[reps / 2])
    };

    // 1. the chunk pass exactly as dormouse calls it: F32 in, fused kernels.
    let (f_min, f_med) = time(&mut || {
        let o = fused_chunk_forward_scratch::<B>(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            b.clone(),
            w.clone(),
            state.clone(),
            1.0,
            CS,
        )
        .expect("f32 takes the fused path")
        .0;
        std::hint::black_box(o.mean().into_scalar::<f32>());
    });

    // 2. the same pass as a dtype-following caller would run it: the gate
    //    refuses, so the dispatcher runs the tensor-ops chunk loop. The bf16
    //    round trip such a caller would also pay is NOT in this arm and
    //    cannot be: burn cannot compile a bf16 cast on this backend (see
    //    `bf16_cast_does_not_compile`), which is the whole finding.
    let (t_min, t_med) = time(&mut || {
        let o = burn_gdn2::chunk_wy_forward(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            b.clone(),
            w.clone(),
            state.clone(),
            1.0,
            CS,
        )
        .0;
        std::hint::black_box(o.mean().into_scalar::<f32>());
    });

    println!("production shape [10,12,512,64] chunk 16, device-synced, 5 reps (min / median):");
    println!("  fused chunk kernels (f32 in)   {f_min:7.2} / {f_med:7.2} ms");
    println!("  tensor-ops chunk loop (fallback){t_min:7.2} / {t_med:7.2} ms");
    println!(
        "  fallback premium                {:7.2} / {:7.2} ms  ({:.0}x / {:.0}x)",
        t_min - f_min,
        t_med - f_med,
        t_min / f_min,
        t_med / f_med
    );
}

/// burn's own f32 -> bf16 cast does not compile on this backend (cubecl's LLVM
/// nvptx path has no bf16 type). Ignored by design: it is a report of the
/// current stack, not a requirement - when upstream lands bf16 lowering this
/// test should start passing.
///
/// Run on its own: cargo test --release -p burn-gdn2 --features cuda --test lowp_bf16_cuda -- --ignored --nocapture
#[test]
#[ignore = "documents the bf16 blocking on the LLVM backend, not a requirement"]
fn bf16_cast_does_not_compile() {
    let dev: Device = Default::default();
    let x = Tensor::<2>::random([64, 64], Distribution::Default, &dev);
    let y = x.clone().cast(FloatDType::BF16);
    let got = y.cast(FloatDType::F32).mean().into_scalar::<f32>();
    let want = x.mean().into_scalar::<f32>();
    println!("bf16 cast round trip: {got} vs f32 {want}");
    assert!((got - want).abs() < 0.1 * want.abs().max(1e-6));
}
