//! GPU kernel for TSCT: ternary-SVD linear as add-only matmuls.
//!
//! Works on any cubecl CUDA device (any NVIDIA GPU; tested on a 5060 but
//! not tied to it). The weights are packed 16-ternary-per-u32 (2 bits
//! each); the kernel decodes {0,1,2} -> {-1,0,+1} branch-free and
//! accumulates with plain adds.
//!
//! One thread computes one output element y[b, j]:
//!   z[b, i] = sum_j x[b, j] * U[j, i]          (m x k GEMM)
//!   y[b, j] = sum_i z[b, i] * s[i] * V[j, i]   (k x n GEMM with scale)
//! Both stages run in one launch; x is read from global once.
use std::any::Any;

#[cfg(feature = "cuda")]
use cubecl::prelude::*;

/// Pack a row-major ternary matrix `[m, n]` into u32 (16 values per word,
/// 2 bits each: 0=-1, 1=0, 2=+1), column-major layout (column j's pack
/// starts at `j * words_per_col`).
#[cfg(feature = "cuda")]
pub fn pack_u32(w: &[i8], m: usize, n: usize) -> Vec<u32> {
    let words = m.div_ceil(16);
    let mut out = vec![0u32; words * n];
    for j in 0..n {
        for i in 0..m {
            let v = match w[i * n + j] {
                -1 => 0u32,
                0 => 1u32,
                1 => 2u32,
                other => panic!("not ternary: {other}"),
            };
            out[j * words + i / 16] |= v << (2 * (i % 16));
        }
    }
    out
}

#[cfg(feature = "cuda")]
#[cube(launch_unchecked)]
fn tsct_linear_kernel<F: Float>(
    x: &[F],       // [B, M]
    upack: &[u32], // packed U: [K, words(M)]
    vpack: &[u32], // packed V: [K, words(N)]
    s: &[F],       // [K]
    out: &mut [F], // [B, N]
    #[comptime] m: u32,
    #[comptime] k: u32,
    #[comptime] n: u32,
    #[comptime] u_words: u32,
    #[comptime] v_words: u32,
    #[comptime] n_chunk: u32,
) {
    let b = CUBE_POS_X as usize;
    let chunk = CUBE_POS_Y as usize;
    let t = UNIT_POS_X as usize;
    let m = m as usize;
    let k = k as usize;
    let n = n as usize;
    let u_words = u_words as usize;
    let v_words = v_words as usize;
    let n_chunk = n_chunk as usize;

    // stage 0: copy x[b] into shared (block-wide, one read from global)
    let mut xs = Shared::<[F]>::new_slice(m);
    let mut idx = t;
    while idx < m {
        xs[idx] = x[b * m + idx];
        idx += n_chunk;
    }
    sync_cube();

    // stage 1: parallel z[k] via atomic fetch_add into shared. Each thread
    // walks its strided slice of x and contributes to all ranks (m*k total
    // work split across the block, not n*m*k).
    let zs = Shared::<[Atomic<F>]>::new_slice(k);
    if t < k {
        zs[t].store(F::new(0.0_f32));
    }
    sync_cube();
    let mut idx = t;
    while idx < m {
        let xv = xs[idx];
        for i in 0..k {
            let word = upack[i * u_words + idx / 16];
            let v = (word >> (2 * ((idx % 16) as u32))) & 3u32;
            let val = if v == 2u32 {
                F::new(1.0_f32)
            } else if v == 0u32 {
                F::new(-1.0_f32)
            } else {
                F::new(0.0_f32)
            };
            let _ = zs[i].fetch_add(xv * val);
        }
        idx += n_chunk;
    }
    sync_cube();

    // stage 2: each thread computes y for its own j: k ops only
    let j = chunk * n_chunk + t;
    if j < n {
        let mut acc = F::new(0.0_f32);
        for i in 0..k {
            let word = vpack[i * v_words + j / 16];
            let v = (word >> (2 * ((j % 16) as u32))) & 3u32;
            let val = if v == 2u32 {
                F::new(1.0_f32)
            } else if v == 0u32 {
                F::new(-1.0_f32)
            } else {
                F::new(0.0_f32)
            };
            acc += zs[i].load() * s[i] * val;
        }
        out[b * n + j] = acc;
    }
}

/// Run the fused TSCT linear on the CUDA cubecl backend. Returns `None`
/// when the input is not on the CUDA runtime.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn tsct_linear_cuda(
    x: &burn::tensor::Tensor<2>,
    u_pack: &[u32],
    v_pack: &[u32],
    s: &[f32],
    m: usize,
    k: usize,
    n: usize,
    b: usize,
) -> Option<burn::tensor::Tensor<2>> {
    use burn_cubecl::tensor::CubeTensor;
    use cubecl::prelude::CubeElement;
    type CB = burn_cubecl::CubeBackend;

    let prim = x.clone().try_into_primitive::<CB>().ok()?;
    let cube = (&prim as &dyn Any).downcast_ref::<CubeTensor>()?;
    let client = cube.client.clone();

    let out = burn::tensor::Tensor::<2>::zeros([b, n], &x.device());
    let out_prim = out.clone().try_into_primitive::<CB>().ok()?;
    let out_cube = (&out_prim as &dyn Any)
        .downcast_ref::<CubeTensor>()?
        .clone();
    let x_handle = cube.handle.clone();
    let out_handle = out_cube.handle.clone();

    let u_bytes: &[u8] = u32::as_bytes(u_pack);
    let v_bytes: &[u8] = u32::as_bytes(v_pack);
    let s_bytes: &[u8] = f32::as_bytes(s);
    let u_buffer = client.create_from_slice(u_bytes);
    let v_buffer = client.create_from_slice(v_bytes);
    let s_buffer = client.create_from_slice(s_bytes);

    let n_chunk = 256u32.min(n as u32);
    let cubes_y = (n as u32).div_ceil(n_chunk);
    let cube_count = CubeCount::Static(b as u32, cubes_y, 1);
    let cube_dim = CubeDim::new_3d(n_chunk, 1, 1);
    unsafe {
        tsct_linear_kernel::launch_unchecked::<f32>(
            &client,
            cube_count,
            cube_dim,
            BufferArg::from_raw_parts(x_handle, b * m),
            BufferArg::from_raw_parts(u_buffer, k * m.div_ceil(16)),
            BufferArg::from_raw_parts(v_buffer, k * n.div_ceil(16)),
            BufferArg::from_raw_parts(s_buffer, k),
            BufferArg::from_raw_parts(out_handle, b * n),
            m as u32,
            k as u32,
            n as u32,
            (m.div_ceil(16)) as u32,
            (n.div_ceil(16)) as u32,
            n_chunk,
        );
    }
    Some(out)
}
