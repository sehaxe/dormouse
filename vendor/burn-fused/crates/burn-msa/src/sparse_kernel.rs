//! Fused CUDA sparse attention kernel (burn 0.21 / cubecl 0.10).
//!
//! One cube per (batch·kv-head, query position); thread `qi` computes the
//! attended-token scores for query head `h_kv·qpkv + qi`, softmax, and the
//! weighted output. Only the top-K selected blocks' tokens are touched:
//! O(S·topk·block_size·d) instead of the tensor path's O(S²).

use burn::backend::Backend;
use burn::tensor::{Device, Int, Tensor};
use cubecl::prelude::*;
use std::any::Any;

/// Force a dense row-major `D`-dim tensor. cubecl's reshape materializes a
/// copy whenever the input is not already contiguous (burn-std
/// `reshape_action`: rank shrink -> Recompute -> copy_into), and the
/// reshape back to `dims` is a lazy view of that dense buffer. The fused
/// kernels index flat row-major, so every operand must be exactly dense:
/// permuted views (SparseAttention's `to_4d` reshape+swap_dims) and
/// row-pitched buffers (cubecl pads 2D row widths to
/// `next_pow2(width_bytes).clamp(16, 512)`, e.g. topk=6 -> 24B rows ->
/// pitch 32B) would otherwise be misread.
fn dense4<const D: usize, K>(t: Tensor<D, K>) -> Tensor<D, K>
where
    K: burn::tensor::kind::Basic,
{
    let dims = t.dims();
    let n = dims.iter().product::<usize>();
    t.reshape::<1, _>([n]).reshape::<D, _>(dims)
}

/// Allocate a dense (never row-pitched) `D`-dim tensor: a fresh 1D buffer
/// is always contiguous and the reshape back is a lazy dense view. The
/// fused kernels write flat row-major, so a pitched output (non-pow2 last
/// dim) would scatter rows into the padding.
fn empty_dense4<const D: usize>(dims: [usize; D], device: &Device) -> Tensor<D> {
    let n = dims.iter().product::<usize>();
    Tensor::<1>::empty([n], device).reshape::<D, _>(dims)
}

/// Allocate a zero-initialized dense (never row-pitched) `D`-dim tensor:
/// a fresh 1D buffer is contiguous and the reshape back is a lazy dense
/// view. The fused kernels write flat row-major, so a pitched output
/// (non-pow2 last dim) would scatter rows into the padding; the backward
/// kernels accumulate with fetch_add, so the buffer must be zeros.
fn zeros_dense4<const D: usize>(dims: [usize; D], device: &Device) -> Tensor<D> {
    let n = dims.iter().product::<usize>();
    Tensor::<1>::zeros([n], device).reshape::<D, _>(dims)
}
#[cube(launch_unchecked)]
fn msa_sparse_attn_kernel<F: Float>(
    q: &[F],       // [B, Hq, S, d]
    k: &[F],       // [B, Hkv, S, d]
    v: &[F],       // [B, Hkv, S, d]
    bi: &[i32],    // [B, Hkv, S, topk] (burn Int is i32)
    out: &mut [F], // [B, Hq, S, d]
    ba: &mut [F],  // [B, Hkv, S, topk]
    scale: f32,
    seq_kv: u32,
    #[comptime] n_heads_kv: u32,
    #[comptime] qpkv: u32,
    #[comptime] d: u32,
    #[comptime] topk: u32,
    #[comptime] block_size: u32,
    #[comptime] attended: u32,
    #[comptime] causal: bool,
) {
    let bh = CUBE_POS_X as usize; // b·Hkv + h_kv
    let t = CUBE_POS_Y as usize;
    let qi = UNIT_POS_X as usize;
    let d = d as usize;
    let topk = topk as usize;
    let bs = block_size as usize;
    let qpkv = qpkv as usize;
    let n_hkv = n_heads_kv as usize;
    let skv = seq_kv as usize;
    let n_att = attended as usize;

    if qi < qpkv {
        let h_kv = bh % n_hkv;
        let b = bh / n_hkv;
        let qflat = (b * (n_hkv * qpkv) + h_kv * qpkv + qi) * skv * d + t * d;
        let kvflat = (b * n_hkv + h_kv) * skv * d;
        let birow = (b * n_hkv + h_kv) * skv * topk + t * topk;

        let mut scores = Shared::<[F]>::new_slice(n_att * qpkv);
        let mut max_s = F::new(-3.0e38_f32);
        for m in 0..n_att {
            let blk = m / bs;
            let off = m % bs;
            let mut idxi = bi[birow + blk] * (bs as i32) + off as i32;
            if idxi < 0 {
                idxi = 0;
            }
            if (idxi as u32) >= seq_kv {
                idxi = (seq_kv - 1) as i32;
            }
            let idx = idxi as usize;
            let mut acc = F::new(0.0_f32);
            for dd in 0..d {
                acc += q[qflat + dd] * k[kvflat + idx * d + dd];
            }
            let sc = acc * F::cast_from(scale);
            if causal && idx > t {
                scores[qi * n_att + m] = F::new(0.0_f32);
            } else {
                scores[qi * n_att + m] = sc;
                if sc > max_s {
                    max_s = sc;
                }
            }
        }
        let mut sum = F::new(0.0_f32);
        for m in 0..n_att {
            let e = (scores[qi * n_att + m] - max_s).exp();
            scores[qi * n_att + m] = e;
            sum += e;
        }
        let inv = F::new(1.0_f32) / sum;
        for m in 0..n_att {
            scores[qi * n_att + m] *= inv;
        }

        for dd in 0..d {
            let mut acc = F::new(0.0_f32);
            for m in 0..n_att {
                let blk = m / bs;
                let off = m % bs;
                let mut idxi = bi[birow + blk] * (bs as i32) + off as i32;
                if idxi < 0 {
                    idxi = 0;
                }
                if (idxi as u32) >= seq_kv {
                    idxi = (seq_kv - 1) as i32;
                }
                let idx = idxi as usize;
                acc += scores[qi * n_att + m] * v[kvflat + idx * d + dd];
            }
            out[qflat + dd] = acc;
        }

        sync_cube();
        if qi == 0 {
            for blk in 0..topk {
                let mut acc = F::new(0.0_f32);
                for qi2 in 0..qpkv {
                    for off in 0..bs {
                        acc += scores[qi2 * n_att + blk * bs + off];
                    }
                }
                ba[(b * n_hkv + h_kv) * skv * topk + t * topk + blk] =
                    acc * F::cast_from(1.0_f32 / qpkv as f32);
            }
        }
    }
}
/// Fused sparse-attention backward. Mirrors the forward kernel (same grid and
/// score/softmax recompute), then applies the attention-backward formulas:
/// dq per query (no race), dk/dv accumulated into the selected slots with
/// atomic fetch_add (multiple queries/positions attend the same key).
#[cube(launch_unchecked)]
fn msa_backward_kernel<F: Float>(
    q: &[F],              // [B, Hq, S, d]
    k: &[F],              // [B, Hkv, S, d]
    v: &[F],              // [B, Hkv, S, d]
    bi: &[i32],           // [B, Hkv, S, topk]
    dout: &[F],           // [B, Hq, S, d]
    dq: &mut [F],         // [B, Hq, S, d]
    dk: &mut [Atomic<F>], // [B, Hkv, S, d]
    dv: &mut [Atomic<F>], // [B, Hkv, S, d]
    scale: f32,
    seq_kv: u32,
    #[comptime] n_heads_kv: u32,
    #[comptime] qpkv: u32,
    #[comptime] d: u32,
    #[comptime] topk: u32,
    #[comptime] block_size: u32,
    #[comptime] attended: u32,
    #[comptime] causal: bool,
) {
    let bh = CUBE_POS_X as usize;
    let t = CUBE_POS_Y as usize;
    let qi = UNIT_POS_X as usize;
    let d = d as usize;
    let topk = topk as usize;
    let bs = block_size as usize;
    let qpkv = qpkv as usize;
    let n_hkv = n_heads_kv as usize;
    let skv = seq_kv as usize;
    let n_att = attended as usize;

    if qi < qpkv {
        let h_kv = bh % n_hkv;
        let b = bh / n_hkv;
        let qflat = (b * (n_hkv * qpkv) + h_kv * qpkv + qi) * skv * d + t * d;
        let kvflat = (b * n_hkv + h_kv) * skv * d;
        let birow = (b * n_hkv + h_kv) * skv * topk + t * topk;

        let mut scores = Shared::<[F]>::new_slice(n_att * qpkv);
        let mut max_s = F::new(-3.0e38_f32);
        for m in 0..n_att {
            let blk = m / bs;
            let off = m % bs;
            let mut idxi = bi[birow + blk] * (bs as i32) + off as i32;
            if idxi < 0 {
                idxi = 0;
            }
            if (idxi as u32) >= seq_kv {
                idxi = (seq_kv - 1) as i32;
            }
            let idx = idxi as usize;
            let mut acc = F::new(0.0_f32);
            for dd in 0..d {
                acc += q[qflat + dd] * k[kvflat + idx * d + dd];
            }
            let sc = acc * F::cast_from(scale);
            if causal && idx > t {
                scores[qi * n_att + m] = F::new(0.0_f32);
            } else {
                scores[qi * n_att + m] = sc;
                if sc > max_s {
                    max_s = sc;
                }
            }
        }
        let mut sum = F::new(0.0_f32);
        for m in 0..n_att {
            let e = (scores[qi * n_att + m] - max_s).exp();
            scores[qi * n_att + m] = e;
            sum += e;
        }
        let inv = F::new(1.0_f32) / sum;
        for m in 0..n_att {
            scores[qi * n_att + m] *= inv;
        }

        // backward
        let mut d_w = Shared::<[F]>::new_slice(n_att * qpkv);
        let mut wd_sum = F::new(0.0_f32);
        for m in 0..n_att {
            let blk = m / bs;
            let off = m % bs;
            let mut idxi = bi[birow + blk] * (bs as i32) + off as i32;
            if idxi < 0 {
                idxi = 0;
            }
            if (idxi as u32) >= seq_kv {
                idxi = (seq_kv - 1) as i32;
            }
            let idx = idxi as usize;
            let mut acc = F::new(0.0_f32);
            for dd in 0..d {
                acc += dout[qflat + dd] * v[kvflat + idx * d + dd];
            }
            d_w[qi * n_att + m] = acc;
            wd_sum += scores[qi * n_att + m] * acc;
        }
        for m in 0..n_att {
            let blk = m / bs;
            let off = m % bs;
            let mut idxi = bi[birow + blk] * (bs as i32) + off as i32;
            if idxi < 0 {
                idxi = 0;
            }
            if (idxi as u32) >= seq_kv {
                idxi = (seq_kv - 1) as i32;
            }
            let idx = idxi as usize;
            let w = scores[qi * n_att + m];
            let dsc = w * (d_w[qi * n_att + m] - wd_sum);
            for dd in 0..d {
                dq[qflat + dd] += dsc * k[kvflat + idx * d + dd] * F::cast_from(scale);
                dk[kvflat + idx * d + dd].fetch_add(dsc * q[qflat + dd] * F::cast_from(scale));
                dv[kvflat + idx * d + dd].fetch_add(w * dout[qflat + dd]);
            }
        }
    }
}

#[cfg(feature = "cuda")]
/// Fused sparse-attention backward on the bare CUDA backend.
#[allow(clippy::too_many_arguments)]
pub fn msa_backward_cuda<B: Backend>(
    q: &Tensor<4>,
    k: &Tensor<4>,
    v: &Tensor<4>,
    block_indices: &Tensor<4, Int>,
    scale: f64,
    block_size: usize,
    n_heads_kv: usize,
    _n_heads_q: usize,
    causal: bool,
    d_out: &Tensor<4>,
) -> Option<(Tensor<4>, Tensor<4>, Tensor<4>)>
where
    burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
{
    use burn_cubecl::tensor::CubeTensor;
    let cube = |t: &Tensor<4>| -> Option<CubeTensor> {
        let prim = t.clone().try_into_primitive::<B>().ok()?;
        let c = (&prim as &dyn Any).downcast_ref::<CubeTensor>()?;
        Some(c.clone())
    };
    let cubei = |t: &Tensor<4, Int>| -> Option<CubeTensor> {
        let prim = t.clone().try_into_primitive::<B>().ok()?;
        let c = (&prim as &dyn Any).downcast_ref::<CubeTensor>()?;
        Some(c.clone())
    };
    let q = dense4(q.clone());
    let k = dense4(k.clone());
    let v = dense4(v.clone());
    let block_indices = dense4(block_indices.clone());
    let d_out = dense4(d_out.clone());
    let q_c = cube(&q)?;
    let k_c = cube(&k)?;
    let v_c = cube(&v)?;
    let bi_c = cubei(&block_indices)?;
    let do_c = cube(&d_out)?;
    let client = q_c.client.clone();
    futures_lite::future::block_on(client.sync()).ok();

    let [batch, h_q, seq_q, d] = q.dims();
    let seq_kv = k.dims()[2];
    let qpkv = h_q / n_heads_kv;
    let topk = block_indices.dims()[3];
    // same guard as the forward: the fused kernel is only verified for topk >= 4
    if topk < 4 {
        return None;
    }
    let attended = topk * block_size;

    let dq = zeros_dense4([batch, h_q, seq_q, d], &q.device());
    let dk = zeros_dense4([batch, n_heads_kv, seq_kv, d], &q.device());
    let dv = zeros_dense4([batch, n_heads_kv, seq_kv, d], &q.device());

    let dq_c = cube(&dq)?;
    let dk_c = cube(&dk)?;
    let dv_c = cube(&dv)?;

    let cube_dim = CubeDim::new_3d(qpkv as u32, 1, 1);
    let cube_count = CubeCount::Static((batch * n_heads_kv) as u32, seq_q as u32, 1);
    unsafe {
        msa_backward_kernel::launch_unchecked::<f32>(
            &client,
            cube_count,
            cube_dim,
            BufferArg::from_raw_parts(q_c.handle, batch * h_q * seq_q * d),
            BufferArg::from_raw_parts(k_c.handle, batch * n_heads_kv * seq_kv * d),
            BufferArg::from_raw_parts(v_c.handle, batch * n_heads_kv * seq_kv * d),
            BufferArg::from_raw_parts(bi_c.handle, batch * n_heads_kv * seq_q * topk),
            BufferArg::from_raw_parts(do_c.handle, batch * h_q * seq_q * d),
            BufferArg::from_raw_parts(dq_c.handle, batch * h_q * seq_q * d),
            BufferArg::from_raw_parts(dk_c.handle, batch * n_heads_kv * seq_kv * d),
            BufferArg::from_raw_parts(dv_c.handle, batch * n_heads_kv * seq_kv * d),
            (1.0 / scale) as f32,
            seq_kv as u32,
            n_heads_kv as u32,
            qpkv as u32,
            d as u32,
            topk as u32,
            block_size as u32,
            attended as u32,
            causal,
        );
    }
    Some((dq, dk, dv))
}

/// Launch the fused sparse attention on CUDA. Returns `None` when `B` is not
/// the bare CUDA `CubeBackend`.
#[allow(clippy::too_many_arguments)]
pub fn sparse_attn_cuda<B: Backend>(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    block_indices: Tensor<4, Int>,
    scale: f64,
    block_size: usize,
    n_heads_kv: usize,
    n_heads_q: usize,
    causal: bool,
) -> Option<(Tensor<4>, Tensor<4>)>
where
    burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
{
    use burn_cubecl::tensor::CubeTensor;
    let cube = |t: &Tensor<4>| -> Option<CubeTensor> {
        let prim = t.clone().try_into_primitive::<B>().ok()?;
        let c = (&prim as &dyn Any).downcast_ref::<CubeTensor>()?;
        Some(c.clone())
    };
    let cubei = |t: &Tensor<4, Int>| -> Option<CubeTensor> {
        let prim = t.clone().try_into_primitive::<B>().ok()?;
        let c = (&prim as &dyn Any).downcast_ref::<CubeTensor>()?;
        Some(c.clone())
    };
    let q = dense4(q);
    let k = dense4(k);
    let v = dense4(v);
    let block_indices = dense4(block_indices);
    let q_c = cube(&q)?;
    let k_c = cube(&k)?;
    let v_c = cube(&v)?;
    let bi_c = cubei(&block_indices)?;
    let client = q_c.client.clone();
    futures_lite::future::block_on(client.sync()).ok();

    let [batch, h_q, seq_q, d] = q.dims();
    // ponytail: fused kernel only verified for topk >= 4. For topk < 4,
    // the kernel reads wrong bi/k/v rows on non-block-aligned query positions
    // (cubecl 0.11-pre; the guard was re-verified with compute-sanitizer +
    // deterministic probes, the defect is in the compiled kernel, not the
    // host code, and integer division is NOT the cause). Tensor fallback keeps
    // correctness guaranteed; revisit when cubecl's kernel cache/codegen
    // matures or the kernel is rewritten without per-cube shared state.
    if block_indices.dims()[3] < 4 {
        return None;
    }
    let seq_kv = k.dims()[2];
    let _ = n_heads_q;
    let qpkv = h_q / n_heads_kv;
    let topk = block_indices.dims()[3];
    let attended = topk * block_size;

    let out = empty_dense4([batch, h_q, seq_q, d], &q.device());
    let ba = empty_dense4([batch, n_heads_kv, seq_q, topk], &q.device());
    let out_c = cube(&out)?;
    let ba_c = cube(&ba)?;

    let cube_dim = CubeDim::new_3d(qpkv as u32, 1, 1);
    let cube_count = CubeCount::Static((batch * n_heads_kv) as u32, seq_q as u32, 1);
    unsafe {
        msa_sparse_attn_kernel::launch_unchecked::<f32>(
            &client,
            cube_count,
            cube_dim,
            BufferArg::from_raw_parts(q_c.handle, batch * h_q * seq_q * d),
            BufferArg::from_raw_parts(k_c.handle, batch * n_heads_kv * seq_kv * d),
            BufferArg::from_raw_parts(v_c.handle, batch * n_heads_kv * seq_kv * d),
            BufferArg::from_raw_parts(bi_c.handle, batch * n_heads_kv * seq_q * topk),
            BufferArg::from_raw_parts(out_c.handle, batch * h_q * seq_q * d),
            BufferArg::from_raw_parts(ba_c.handle, batch * n_heads_kv * seq_q * topk),
            (1.0 / scale) as f32,
            seq_kv as u32,
            n_heads_kv as u32,
            qpkv as u32,
            d as u32,
            topk as u32,
            block_size as u32,
            attended as u32,
            causal,
        );
    }

    let _ = attended;
    Some((out, ba))
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::attention::sparse_attn_batched_gqa;
    use burn::tensor::{Device, Distribution, Tensor, TensorData};
    use burn_cubecl::CubeBackend;
    type Cuda = CubeBackend;

    #[test]
    #[ignore]
    fn pt_compare_bench() {
        type B = Cuda;
        let device = Device::default();
        #[allow(clippy::type_complexity)] // bench config table
        let configs: &[(
            &str,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
        )] = &[
            ("small", 128, 4, 2, 32, 16, 32, 4, 1, 256, 256),
            ("med", 256, 8, 2, 32, 32, 64, 4, 1, 256, 1024),
            ("large", 512, 8, 2, 64, 64, 64, 8, 1, 256, 4096),
            ("xl", 768, 12, 4, 64, 64, 128, 16, 1, 512, 8192),
            ("prefill_8k", 512, 8, 2, 64, 64, 64, 8, 1, 512, 8192),
            ("decode_64k", 512, 8, 2, 64, 64, 64, 8, 1, 1, 65536),
        ];
        for &(name, dm, nq, nk, dh, di, bs, tk, batch, sq, sk) in configs {
            let _ = dm;
            let q: Tensor<4> =
                Tensor::<4>::random([batch, nq, sq, dh], Distribution::Normal(0.0, 1.0), &device);
            let k: Tensor<4> =
                Tensor::<4>::random([batch, nk, sk, dh], Distribution::Normal(0.0, 1.0), &device);
            let v: Tensor<4> =
                Tensor::<4>::random([batch, nk, sk, dh], Distribution::Normal(0.0, 1.0), &device);
            let nb = sk.div_ceil(bs);
            let bi_data: Vec<i32> = (0..(batch * nk * sq * tk) as i32)
                .map(|i| {
                    (i * tk as i32 + i / (sq as i32 * tk as i32)) % (nb as i32 * tk as i32)
                        / tk as i32
                })
                .collect();
            let bi: Tensor<4, Int> =
                Tensor::<4, Int>::from_data(TensorData::new(bi_data, [batch, nk, sq, tk]), &device)
                    .reshape([batch, nk, sq, tk]);
            let scale = (di as f64).sqrt();
            for _ in 0..10 {
                let (o, _) = sparse_attn_cuda::<B>(
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    bi.clone(),
                    scale,
                    bs,
                    nk,
                    nq,
                    false,
                )
                .unwrap();
                o.clone().to_data();
            }
            let runs = if batch * sq <= 256 {
                50
            } else if batch * sq <= 4096 {
                30
            } else {
                15
            };
            let t0 = std::time::Instant::now();
            for _ in 0..runs {
                let (o, _) = sparse_attn_cuda::<B>(
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    bi.clone(),
                    scale,
                    bs,
                    nk,
                    nq,
                    false,
                )
                .unwrap();
                o.clone().to_data();
            }
            let ms = t0.elapsed().as_secs_f64() / runs as f64 * 1e3;
            println!("{name:<12} attn {ms:>8.3} ms");
        }
    }

    #[test]
    #[ignore]
    fn final_bench() {
        type B = Cuda;
        let device = Device::default();
        let (b2, hq2, hkv2, s2, d2, topk2, bs2) = (4usize, 32, 8, 2048, 64, 16, 32);
        let scale2 = d2 as f64;
        let q2: Tensor<4> =
            Tensor::<4>::random([b2, hq2, s2, d2], Distribution::Normal(0.0, 1.0), &device);
        let k2: Tensor<4> =
            Tensor::<4>::random([b2, hkv2, s2, d2], Distribution::Normal(0.0, 1.0), &device);
        let v2: Tensor<4> =
            Tensor::<4>::random([b2, hkv2, s2, d2], Distribution::Normal(0.0, 1.0), &device);
        let nb = s2 / bs2;
        let bi_data: Vec<i32> = (0..(b2 * hkv2 * s2 * topk2) as i32)
            .map(|i| {
                (i * topk2 as i32 + i / (s2 as i32 * topk2 as i32)) % (nb as i32 * topk2 as i32)
                    / topk2 as i32
            })
            .collect();
        let bi2: Tensor<4, Int> =
            Tensor::<4, Int>::from_data(TensorData::new(bi_data, [b2, hkv2, s2, topk2]), &device)
                .reshape([b2, hkv2, s2, topk2]);
        q2.clone().to_data();
        for _ in 0..5 {
            let _ = sparse_attn_cuda::<B>(
                q2.clone(),
                k2.clone(),
                v2.clone(),
                bi2.clone(),
                scale2,
                bs2,
                hkv2,
                hq2,
                false,
            );
        }
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            let _ = sparse_attn_cuda::<B>(
                q2.clone(),
                k2.clone(),
                v2.clone(),
                bi2.clone(),
                scale2,
                bs2,
                hkv2,
                hq2,
                false,
            );
        }
        println!("FUSED {:?}", t0.elapsed() / 20);
        let t0 = std::time::Instant::now();
        for _ in 0..3 {
            let _ = sparse_attn_batched_gqa(
                q2.clone(),
                k2.clone(),
                v2.clone(),
                bi2.clone(),
                scale2,
                bs2,
                hkv2,
                hq2,
                false,
            );
        }
        println!("ORIG  {:?}", t0.elapsed() / 3);
    }

    #[test]
    fn kernel_matches_batched_gqa() {
        type B = Cuda;
        let device = Device::default();
        let b = 2;
        let hq = 8;
        let hkv = 2;
        let s = 64;
        let d = 16;
        let topk = 4;
        let bs = 8;
        let scale = d as f64;

        let q: Tensor<4> =
            Tensor::<4>::random([b, hq, s, d], Distribution::Normal(0.0, 1.0), &device);
        let k: Tensor<4> =
            Tensor::<4>::random([b, hkv, s, d], Distribution::Normal(0.0, 1.0), &device);
        let v: Tensor<4> =
            Tensor::<4>::random([b, hkv, s, d], Distribution::Normal(0.0, 1.0), &device);
        let nblocks = s / bs;
        let bi_data: Vec<i32> = (0..(b * hkv * s * topk) as i32)
            .map(|i| {
                let n = nblocks as i32 * topk as i32;
                (i * topk as i32 + i / (s as i32 * topk as i32)) % n / topk as i32
            })
            .collect();
        let bi: Tensor<4, Int> =
            Tensor::<4, Int>::from_data(TensorData::new(bi_data, [b, hkv, s, topk]), &device)
                .reshape([b, hkv, s, topk]);

        let (o_ref, ba_ref) = sparse_attn_batched_gqa(
            q.clone(),
            k.clone(),
            v.clone(),
            bi.clone(),
            scale,
            bs,
            hkv,
            hq,
            false,
        );
        let (o_k, ba_k) = sparse_attn_cuda::<B>(
            q.clone(),
            k.clone(),
            v.clone(),
            bi.clone(),
            scale,
            bs,
            hkv,
            hq,
            false,
        )
        .expect("kernel should run");

        let o_diff = (o_ref.clone() - o_k.clone())
            .abs()
            .max()
            .into_scalar::<f32>();
        let ba_diff = (ba_ref.clone() - ba_k.clone())
            .abs()
            .max()
            .into_scalar::<f32>();
        assert!(o_diff < 1e-3, "out max diff {o_diff}");
        assert!(ba_diff < 1e-3, "ba max diff {ba_diff}");
    }

    /// topk=6 (24B rows -> pitched to 32B) and seq=96: the fused path must
    /// dense-materialize the pitched block_indices/ba buffers. nb >= 4 guard
    /// passes, so this runs the raw kernel.
    #[test]
    fn kernel_matches_batched_gqa_non_pow2() {
        type B = Cuda;
        let device = Device::default();
        let b = 2;
        let hq = 8;
        let hkv = 2;
        let s = 96;
        let d = 16;
        let topk = 6;
        let bs = 8;
        let scale = d as f64;

        let q: Tensor<4> =
            Tensor::<4>::random([b, hq, s, d], Distribution::Normal(0.0, 1.0), &device);
        let k: Tensor<4> =
            Tensor::<4>::random([b, hkv, s, d], Distribution::Normal(0.0, 1.0), &device);
        let v: Tensor<4> =
            Tensor::<4>::random([b, hkv, s, d], Distribution::Normal(0.0, 1.0), &device);
        let nblocks = s / bs;
        let bi_data: Vec<i32> = (0..(b * hkv * s * topk) as i32)
            .map(|i| {
                let n = nblocks as i32 * topk as i32;
                (i * topk as i32 + i / (s as i32 * topk as i32)) % n / topk as i32
            })
            .collect();
        let bi: Tensor<4, Int> =
            Tensor::<4, Int>::from_data(TensorData::new(bi_data, [b, hkv, s, topk]), &device)
                .reshape([b, hkv, s, topk]);

        let (o_ref, ba_ref) = sparse_attn_batched_gqa(
            q.clone(),
            k.clone(),
            v.clone(),
            bi.clone(),
            scale,
            bs,
            hkv,
            hq,
            false,
        );
        let (o_k, ba_k) = sparse_attn_cuda::<B>(
            q.clone(),
            k.clone(),
            v.clone(),
            bi.clone(),
            scale,
            bs,
            hkv,
            hq,
            false,
        )
        .expect("kernel should run");

        let o_diff = (o_ref.clone() - o_k.clone())
            .abs()
            .max()
            .into_scalar::<f32>();
        let ba_diff = (ba_ref.clone() - ba_k.clone())
            .abs()
            .max()
            .into_scalar::<f32>();
        assert!(o_diff < 1e-3, "out max diff {o_diff}");
        assert!(ba_diff < 1e-3, "ba max diff {ba_diff}");
    }

    /// SparseAttention::forward_sparse feeds swap_dims views (to_4d:
    /// reshape + swap_dims of a dense [B, S, H, d]); the fused kernels must
    /// dense-materialize them or they read the permuted buffer flat.
    #[test]
    fn kernel_matches_batched_gqa_view_inputs() {
        type B = Cuda;
        let device = Device::default();
        let b = 1;
        let hq = 4;
        let hkv = 2;
        let s = 32;
        let d = 8;
        let topk = 4;
        let bs = 8;
        let scale = d as f64;

        let qd = Tensor::<4>::random([b, s, hq, d], Distribution::Normal(0.0, 1.0), &device);
        let q = qd.swap_dims(1, 2); // [B, Hq, S, d] non-dense view
        let kd = Tensor::<4>::random([b, s, hkv, d], Distribution::Normal(0.0, 1.0), &device);
        let k = kd.swap_dims(1, 2);
        let vd = Tensor::<4>::random([b, s, hkv, d], Distribution::Normal(0.0, 1.0), &device);
        let v = vd.swap_dims(1, 2);
        let nblocks = s / bs;
        let bi_data: Vec<i32> = (0..(b * hkv * s * topk) as i32)
            .map(|i| {
                let n = nblocks as i32 * topk as i32;
                (i * topk as i32 + i / (s as i32 * topk as i32)) % n / topk as i32
            })
            .collect();
        let bi: Tensor<4, Int> =
            Tensor::<4, Int>::from_data(TensorData::new(bi_data, [b, hkv, s, topk]), &device)
                .reshape([b, hkv, s, topk]);

        let (o_ref, ba_ref) = sparse_attn_batched_gqa(
            q.clone(),
            k.clone(),
            v.clone(),
            bi.clone(),
            scale,
            bs,
            hkv,
            hq,
            false,
        );
        let (o_k, ba_k) = sparse_attn_cuda::<B>(
            q.clone(),
            k.clone(),
            v.clone(),
            bi.clone(),
            scale,
            bs,
            hkv,
            hq,
            false,
        )
        .expect("kernel should run");

        let o_diff = (o_ref.clone() - o_k.clone())
            .abs()
            .max()
            .into_scalar::<f32>();
        let ba_diff = (ba_ref.clone() - ba_k.clone())
            .abs()
            .max()
            .into_scalar::<f32>();
        assert!(o_diff < 1e-3, "out max diff {o_diff}");
        assert!(ba_diff < 1e-3, "ba max diff {ba_diff}");
    }
}
#[cfg(all(test, feature = "cuda"))]
mod dtype_probe {
    use burn::tensor::{Device, Int, Tensor};

    #[test]
    fn probe_int_dtype() {
        let device = Device::default();
        let bi: Tensor<4, Int> = Tensor::<4, Int>::from_data(
            burn::tensor::TensorData::new(vec![1i64, 2, 1, 2], [1, 1, 2, 2]),
            &device,
        );
        let prim = bi.clone().try_into_primitive::<burn_cuda::Cuda>().unwrap();
        let c = (&prim as &dyn std::any::Any)
            .downcast_ref::<burn_cubecl::tensor::CubeTensor>()
            .unwrap();
        println!(
            "int dtype: {:?}, bytes for 4 elems: {:?}",
            c.dtype,
            bi.into_data().bytes
        );
    }
}
