use burn::backend::Backend;
use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::tensor::activation;
use burn::tensor::{Device, Int, Tensor};

use crate::config::MsaConfig;

#[derive(Module, Debug)]
pub struct SparseAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    #[module(skip)]
    pub cfg: MsaConfig,
}

impl SparseAttention {
    pub fn new(cfg: &MsaConfig, device: &Device) -> Self {
        Self {
            q_proj: LinearConfig::new(cfg.d_model, cfg.n_heads_q * cfg.d_head)
                .with_bias(false)
                .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                .init(device),
            k_proj: LinearConfig::new(cfg.d_model, cfg.n_heads_kv * cfg.d_head)
                .with_bias(false)
                .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                .init(device),
            v_proj: LinearConfig::new(cfg.d_model, cfg.n_heads_kv * cfg.d_head)
                .with_bias(false)
                .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                .init(device),
            out_proj: LinearConfig::new(cfg.n_heads_q * cfg.d_head, cfg.d_model)
                .with_bias(false)
                .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                .init(device),
            cfg: cfg.clone(),
        }
    }

    fn to_4d(t: Tensor<3>, n_heads: usize, d_head: usize) -> Tensor<4> {
        let [batch, seq, _] = t.dims();
        t.reshape::<4, _>([batch, seq, n_heads, d_head])
            .swap_dims(1, 2)
    }

    pub fn forward_sparse<B: Backend>(
        &self,
        q: Tensor<3>,
        k: Tensor<3>,
        v: Tensor<3>,
        block_indices: Tensor<4, Int>,
    ) -> Tensor<3>
    where
        burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
    {
        self.forward_sparse_with_weights::<B>(q, k, v, block_indices)
            .0
    }

    pub fn forward_sparse_with_weights<B: Backend>(
        &self,
        q: Tensor<3>,
        k: Tensor<3>,
        v: Tensor<3>,
        block_indices: Tensor<4, Int>,
    ) -> (Tensor<3>, Tensor<4>)
    where
        burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
    {
        let (out_4d, block_attn) = {
            #[cfg(all(feature = "cuda", feature = "autodiff"))]
            {
                type CudaBare = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;
                if let Some(res) = crate::autodiff::sparse_attn_autodiff::<CudaBare>(
                    Self::to_4d(q.clone(), self.cfg.n_heads_q, self.cfg.d_head),
                    Self::to_4d(k.clone(), self.cfg.n_heads_kv, self.cfg.d_head),
                    Self::to_4d(v.clone(), self.cfg.n_heads_kv, self.cfg.d_head),
                    block_indices.clone(),
                    (self.cfg.d_head as f64).sqrt(),
                    self.cfg.block_size,
                    self.cfg.n_heads_kv,
                    self.cfg.n_heads_q,
                    self.cfg.causal,
                ) {
                    res
                } else {
                    if let Some(res) = crate::sparse_kernel::sparse_attn_cuda::<B>(
                        Self::to_4d(q.clone(), self.cfg.n_heads_q, self.cfg.d_head),
                        Self::to_4d(k.clone(), self.cfg.n_heads_kv, self.cfg.d_head),
                        Self::to_4d(v.clone(), self.cfg.n_heads_kv, self.cfg.d_head),
                        block_indices.clone(),
                        (self.cfg.d_head as f64).sqrt(),
                        self.cfg.block_size,
                        self.cfg.n_heads_kv,
                        self.cfg.n_heads_q,
                        self.cfg.causal,
                    ) {
                        res
                    } else {
                        sparse_attn_batched_gqa(
                            Self::to_4d(q, self.cfg.n_heads_q, self.cfg.d_head),
                            Self::to_4d(k, self.cfg.n_heads_kv, self.cfg.d_head),
                            Self::to_4d(v, self.cfg.n_heads_kv, self.cfg.d_head),
                            block_indices,
                            (self.cfg.d_head as f64).sqrt(),
                            self.cfg.block_size,
                            self.cfg.n_heads_kv,
                            self.cfg.n_heads_q,
                            self.cfg.causal,
                        )
                    }
                }
            }
            #[cfg(not(all(feature = "cuda", feature = "autodiff")))]
            {
                sparse_attn_batched_gqa(
                    Self::to_4d(q, self.cfg.n_heads_q, self.cfg.d_head),
                    Self::to_4d(k, self.cfg.n_heads_kv, self.cfg.d_head),
                    Self::to_4d(v, self.cfg.n_heads_kv, self.cfg.d_head),
                    block_indices,
                    (self.cfg.d_head as f64).sqrt(),
                    self.cfg.block_size,
                    self.cfg.n_heads_kv,
                    self.cfg.n_heads_q,
                    self.cfg.causal,
                )
            }
        };
        let [b, _, s, _] = out_4d.dims();
        let out_3d =
            out_4d
                .swap_dims(1, 2)
                .reshape::<3, _>([b, s, self.cfg.n_heads_q * self.cfg.d_head]);
        (self.out_proj.forward(out_3d), block_attn)
    }

    pub fn forward_dense<B: Backend>(&self, q: Tensor<3>, k: Tensor<3>, v: Tensor<3>) -> Tensor<3> {
        let cfg = &self.cfg;
        let d_head = cfg.d_head;
        let n_heads_q = cfg.n_heads_q;
        let n_heads_kv = cfg.n_heads_kv;
        let scale = (d_head as f64).sqrt();
        let q_per_kv = n_heads_q / n_heads_kv;

        let q_4d = Self::to_4d(q, n_heads_q, d_head);
        let k_4d = Self::to_4d(k, n_heads_kv, d_head);
        let v_4d = Self::to_4d(v, n_heads_kv, d_head);

        let [_, _, seq_q, _] = q_4d.dims();
        let seq_kv = k_4d.dims()[2];

        if q_per_kv > 1 {
            let k_exp = k_4d.repeat(&[1, q_per_kv, 1, 1]);
            let s = q_4d.matmul(k_exp.swap_dims(2, 3)).div_scalar(scale);
            let v_exp = v_4d.repeat(&[1, q_per_kv, 1, 1]);
            let scores = causal_mask_4d(s, seq_q, seq_kv, cfg.causal);
            let attn = softmax(scores, 3);
            let out_4d = attn.matmul(v_exp);
            let [b, _, s, _] = out_4d.dims();
            let out_3d = out_4d
                .swap_dims(1, 2)
                .reshape::<3, _>([b, s, n_heads_q * d_head]);
            self.out_proj.forward(out_3d)
        } else {
            let s = q_4d.matmul(k_4d.swap_dims(2, 3)).div_scalar(scale);
            let scores = causal_mask_4d(s, seq_q, seq_kv, cfg.causal);
            let attn = softmax(scores, 3);
            let out_4d = attn.matmul(v_4d);
            let [b, _, s, _] = out_4d.dims();
            let out_3d = out_4d
                .swap_dims(1, 2)
                .reshape::<3, _>([b, s, n_heads_q * d_head]);
            self.out_proj.forward(out_3d)
        }
    }
}

fn causal_mask_4d(scores: Tensor<4>, seq_q: usize, seq_kv: usize, causal: bool) -> Tensor<4> {
    if !causal || seq_q <= 1 {
        return scores;
    }
    let device = scores.device();
    let q_pos =
        Tensor::<1, Int>::arange(0..seq_q as i64, &device).reshape::<4, _>([1, 1, seq_q, 1]);
    let kv_pos =
        Tensor::<1, Int>::arange(0..seq_kv as i64, &device).reshape::<4, _>([1, 1, 1, seq_kv]);
    scores.add(
        kv_pos
            .greater(q_pos)
            .float()
            .mul_scalar(crate::NEG_INF_SAFE),
    )
}

/// 5D batched GQA sparse attention - blockwise KV-outer pass (MiniMax MSA,
/// arXiv 2606.13392).
///
/// Shapes:
/// - q: [batch, n_heads_q, seq_q, d_head]
/// - k, v: [batch, n_heads_kv, seq_kv, d_head]
/// - block_indices: [batch, n_heads_kv, seq_q, topk]
///
/// Returns (output [batch, n_heads_q, seq_q, d_head], block_attn [batch, n_heads_kv, seq_q, topk]).
///
/// The selected top-K blocks are processed one at a time in KV-outer order
/// with an online softmax (running max/sum), so peak memory is linear in
/// seq_kv: the [B, Hkv, qpkv, S, d]/[B, Hkv, qpkv, S, topk] accumulators plus
/// a single [B, Hkv, qpkv, S, block_size] score block. A full
/// [B, Hkv, qpkv, S, S] scores matrix is never materialized.
///
/// Invalid (beyond-seq_kv) token slots of the last block are clamped to
/// seq_kv-1 and keep their attention (duplicating the tail token); under
/// `causal` the mask (idx > q_pos) removes them, matching the fused kernel.
#[allow(clippy::too_many_arguments)]
pub fn sparse_attn_batched_gqa(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    block_indices: Tensor<4, Int>,
    scale: f64,
    block_size: usize,
    n_heads_kv: usize,
    n_heads_q: usize,
    causal: bool,
) -> (Tensor<4>, Tensor<4>) {
    let batch = q.dims()[0];
    let seq_q = q.dims()[2];
    let d_head = q.dims()[3];
    let seq_kv = k.dims()[2];
    let qpkv = n_heads_q / n_heads_kv;
    let topk = block_indices.dims()[3];
    let device = q.device();

    let q_s = q
        .reshape::<5, _>([batch, n_heads_kv, qpkv, seq_q, d_head])
        .swap_dims(2, 3); // [B, Hkv, S, qpkv, d] (S as batch dim)

    let offsets = Tensor::<1, Int>::arange(0..block_size as i64, &device)
        .reshape::<4, _>([1, 1, 1, block_size]);
    let q_pos =
        Tensor::<1, Int>::arange(0..seq_q as i64, &device).reshape::<4, _>([1, 1, seq_q, 1]);

    // Online softmax accumulators (all linear in S).
    let mut m = Tensor::<5>::full([batch, n_heads_kv, qpkv, seq_q, 1], -3.0e38_f32, &device);
    let mut l = Tensor::<5>::zeros([batch, n_heads_kv, qpkv, seq_q, 1], &device);
    let mut out = Tensor::<5>::zeros([batch, n_heads_kv, qpkv, seq_q, d_head], &device);
    // Per-block unnormalized attention sums, rescaled to the running max so
    // the final /l turns them into per-block attention (for `ba`).
    let mut ba_acc = Tensor::<5>::zeros([batch, n_heads_kv, qpkv, seq_q, topk], &device);

    for blk in 0..topk {
        let (scores_b, _k_b, v_b, _idx) = block_scores(
            &q_s,
            &k,
            &v,
            &block_indices,
            &offsets,
            &q_pos,
            blk,
            block_size,
            seq_kv,
            scale,
            causal,
        );

        let m_new = m.clone().max_pair(scores_b.clone().max_dim(4)); // [B,Hkv,qpkv,S,1]
        let alpha = m.sub(m_new.clone()).exp(); // [B,Hkv,qpkv,S,1]
        let e_b = scores_b.sub(m_new.clone()).exp(); // [B,Hkv,qpkv,S,bs]
        let e_sum = e_b.clone().sum_dim(4); // [B,Hkv,qpkv,S,1]

        ba_acc = ba_acc.mul(alpha.clone()).slice_assign(
            [0..batch, 0..n_heads_kv, 0..qpkv, 0..seq_q, blk..blk + 1],
            e_sum.clone(),
        );
        out = out
            .mul(alpha.clone())
            .add(e_b.swap_dims(2, 3).matmul(v_b).swap_dims(2, 3));
        m = m_new;
        l = l.mul(alpha).add(e_sum);
    }

    let out_4d = out
        .div(l.clone()) // [B,Hkv,qpkv,S,d]
        .reshape::<4, _>([batch, n_heads_q, seq_q, d_head]);
    let ba = ba_acc
        .div(l)
        .sum_dim(2)
        .squeeze_dim::<4>(2)
        .div_scalar(qpkv as f64); // [B,Hkv,S,topk]

    (out_4d, ba)
}

/// Per-block gathered keys/values and scaled (optionally causally masked)
/// scores for block `blk` of the top-K selection.
///
/// Returns (scores_b [B,Hkv,qpkv,S,bs], k_b, v_b [B,Hkv,S,bs,d],
/// idx [B,Hkv,S,bs]) with token indices clamped to [0, seq_kv-1] — the tail
/// duplicates the last token, and under `causal` the mask removes
/// idx > q_pos. `q_s` is [B,Hkv,S,qpkv,d] (S as a batch dim).
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_scores(
    q_s: &Tensor<5>,
    k: &Tensor<4>,
    v: &Tensor<4>,
    block_indices: &Tensor<4, Int>,
    offsets: &Tensor<4, Int>,
    q_pos: &Tensor<4, Int>,
    blk: usize,
    block_size: usize,
    seq_kv: usize,
    scale: f64,
    causal: bool,
) -> (Tensor<5>, Tensor<5>, Tensor<5>, Tensor<4, Int>) {
    let [batch, n_heads_kv, seq_q, _qpkv, d_head] = q_s.dims();

    let base = block_indices
        .clone()
        .slice([0..batch, 0..n_heads_kv, 0..seq_q, blk..blk + 1]); // [B,Hkv,S,1]
    let idx = base
        .mul_scalar(block_size as i64)
        .add(offsets.clone())
        .clamp_min(0)
        .clamp_max((seq_kv - 1) as i64); // [B,Hkv,S,bs]

    // Flat gather over (bs, d): each token position p maps to p*d + {0..d-1}.
    // A 4D k cannot be gathered by a 5D [B,Hkv,S,bs,d] index, so (bs,d) is
    // merged into one index dim and the result is reshaped back.
    let d_arange = Tensor::<1, Int>::arange(0..d_head as i64, &q_s.device())
        .reshape::<5, _>([1, 1, 1, 1, d_head]);
    let idx_flat = idx
        .clone()
        .unsqueeze_dim::<5>(4)
        .mul_scalar(d_head as i64)
        .add(d_arange)
        .reshape::<4, _>([batch, n_heads_kv, seq_q, block_size * d_head]) // [B,Hkv,S,bs*d]
        .reshape::<3, _>([batch, n_heads_kv, seq_q * block_size * d_head]); // [B,Hkv,S*bs*d]
    let k_b = k
        .clone()
        .reshape::<3, _>([batch, n_heads_kv, seq_kv * d_head])
        .gather(2, idx_flat.clone())
        .reshape::<5, _>([batch, n_heads_kv, seq_q, block_size, d_head]); // [B,Hkv,S,bs,d]
    let v_b = v
        .clone()
        .reshape::<3, _>([batch, n_heads_kv, seq_kv * d_head])
        .gather(2, idx_flat)
        .reshape::<5, _>([batch, n_heads_kv, seq_q, block_size, d_head]);

    let mut scores = q_s
        .clone()
        .matmul(k_b.clone().swap_dims(3, 4)) // [B,Hkv,S,qpkv,bs]
        .swap_dims(2, 3) // [B,Hkv,qpkv,S,bs]
        .div_scalar(scale);
    if causal {
        scores = scores.add(
            idx.clone()
                .unsqueeze_dim::<5>(2)
                .greater(q_pos.clone().unsqueeze_dim::<5>(2))
                .float()
                .mul_scalar(crate::NEG_INF_SAFE),
        );
    }

    (scores, k_b, v_b, idx)
}

#[inline]
pub fn softmax<const D: usize>(t: Tensor<D>, dim: usize) -> Tensor<D> {
    activation::softmax(t, dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;
    use burn::tensor::{Distribution, TensorData};

    fn device() -> Device {
        Device::ndarray()
    }

    fn extract_f32(t: Tensor<4>) -> Vec<f32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn sparse_attn_per_token(
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        block_indices: Tensor<4, Int>,
        scale: f64,
        block_size: usize,
        n_heads_kv: usize,
        n_heads_q: usize,
        causal: bool,
    ) -> (Tensor<4>, Tensor<4>) {
        let batch = q.dims()[0];
        let seq_q = q.dims()[2];
        let d_head = q.dims()[3];
        let seq_kv = k.dims()[2];
        let q_per_kv = n_heads_q / n_heads_kv;
        let topk = block_indices.dims()[3];
        let attended = topk * block_size;

        let offsets_5d = Tensor::<1, Int>::arange(0..block_size as i64, &q.device())
            .reshape::<5, _>([1, 1, 1, 1, block_size]);

        let mut out_rows = Vec::with_capacity(n_heads_kv);
        let mut aw_rows = Vec::with_capacity(n_heads_kv);

        for h_kv in 0..n_heads_kv {
            let k_h = k.clone().slice([0..batch, h_kv..h_kv + 1]);
            let v_h = v.clone().slice([0..batch, h_kv..h_kv + 1]);
            let mut attn_outputs = Vec::with_capacity(seq_q);
            let mut attn_weights = Vec::with_capacity(seq_q);

            for t in 0..seq_q {
                let blk = block_indices
                    .clone()
                    .slice([0..batch, h_kv..h_kv + 1, t..t + 1]);
                let blk_scaled = blk.mul_scalar(block_size as i64);
                let blk_5d = blk_scaled
                    .unsqueeze_dim::<5>(4)
                    .repeat(&[1, 1, 1, 1, block_size]);
                let idx = blk_5d
                    .add(offsets_5d.clone())
                    .reshape::<4, _>([batch, 1, attended, 1])
                    .repeat(&[1, 1, 1, d_head])
                    .clamp_max((seq_kv - 1) as i64);

                let k_sel = k_h.clone().gather(2, idx.clone());
                let v_sel = v_h.clone().gather(2, idx.clone());

                let q_t = q.clone().slice([
                    0..batch,
                    (h_kv * q_per_kv)..((h_kv + 1) * q_per_kv),
                    t..t + 1,
                ]);

                let scores = q_t.matmul(k_sel.swap_dims(2, 3)).div_scalar(scale);
                let scores = if causal {
                    // same clamp semantics as the batched path: idx > t masked
                    let pos = idx
                        .slice([0..batch, 0..1, 0..attended, 0..1])
                        .swap_dims(2, 3); // [B,1,1,attended]
                    scores.add(
                        pos.greater_scalar(t as i64)
                            .float()
                            .mul_scalar(crate::NEG_INF_SAFE),
                    )
                } else {
                    scores
                };
                let attn_n = softmax_4d(scores, 3);

                let aw = attn_n
                    .clone()
                    .reshape::<5, _>([batch, q_per_kv, 1, topk, block_size])
                    .sum_dim(4)
                    .squeeze_dim::<4>(4);

                attn_outputs.push(attn_n.matmul(v_sel));
                attn_weights.push(aw);
            }
            out_rows.push(Tensor::cat(attn_outputs, 2));
            let aw_seq = Tensor::cat(attn_weights, 2);
            aw_rows.push(aw_seq.mean_dim(1));
        }

        let block_attn = Tensor::cat(aw_rows, 1);
        (Tensor::cat(out_rows, 1), block_attn)
    }

    fn softmax_4d(t: Tensor<4>, dim: usize) -> Tensor<4> {
        let mx = t.clone().max_dim(dim);
        let shifted = t.sub(mx);
        let ex = shifted.exp();
        let sum = ex.clone().sum_dim(dim);
        ex.div(sum)
    }

    #[test]
    fn batched_matches_per_token() {
        let d_head = 8;

        for &seq_q in [1usize, 2, 4, 7].iter() {
            for &seq_kv in [4usize, 8, 12, 16].iter() {
                // multiples of block_size=4
                for &n_heads_kv in [1usize, 2].iter() {
                    for &topk in [1usize, 3].iter() {
                        let n_heads_q = n_heads_kv * 2;
                        let block_size = 4;
                        let scale = (d_head as f64).sqrt();
                        let batch = 1;

                        let q = Tensor::<4>::random(
                            [batch, n_heads_q, seq_q, d_head],
                            Distribution::Normal(0.0, 1.0),
                            &device(),
                        );
                        let k = Tensor::<4>::random(
                            [batch, n_heads_kv, seq_kv, d_head],
                            Distribution::Normal(0.0, 1.0),
                            &device(),
                        );
                        let v = Tensor::<4>::random(
                            [batch, n_heads_kv, seq_kv, d_head],
                            Distribution::Normal(0.0, 1.0),
                            &device(),
                        );

                        let n_blocks = seq_kv.div_ceil(block_size);
                        let topk_actual = topk.min(n_blocks);
                        let per_group: Vec<i64> = (0..seq_q as i64)
                            .flat_map(|_| (0..topk_actual as i64).collect::<Vec<_>>())
                            .collect();
                        let idx_data: Vec<i64> = per_group.repeat(n_heads_kv);
                        let bi = Tensor::<4, Int>::from_data(
                            TensorData::new(idx_data, [batch, n_heads_kv, seq_q, topk_actual]),
                            &device(),
                        );

                        let (ref_out, _) = sparse_attn_per_token(
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            bi.clone(),
                            scale,
                            block_size,
                            n_heads_kv,
                            n_heads_q,
                            false,
                        );
                        let (batched_out, _) = sparse_attn_batched_gqa(
                            q, k, v, bi, scale, block_size, n_heads_kv, n_heads_q, false,
                        );

                        let ref_vals = extract_f32(ref_out);
                        let batched_vals = extract_f32(batched_out);

                        assert_eq!(ref_vals.len(), batched_vals.len());

                        for (i, (r, b)) in ref_vals.iter().zip(batched_vals.iter()).enumerate() {
                            if r.is_nan() && b.is_nan() {
                                continue;
                            }
                            if *r == 0.0 && *b == 0.0 {
                                continue;
                            }
                            let abs_diff = (r - b).abs();
                            let rel_diff = abs_diff / r.abs().max(b.abs()).max(f32::EPSILON);
                            assert!(
                                abs_diff <= 1e-5 || rel_diff <= 1e-4,
                                "Mismatch idx={} seq_q={} seq_kv={} kh={} tk={}: ref={} got={} abs={:.2e} rel={:.2e}",
                                i, seq_q, seq_kv, n_heads_kv, topk, r, b, abs_diff, rel_diff
                            );
                        }
                    }
                }
            }
        }
    }

    /// Non-multiple seq_kv: the last block's invalid token slots are clamped
    /// to seq_kv-1 and (non-causal) keep their attention — the blockwise
    /// path must reproduce the per-token reference exactly, causal or not.
    #[test]
    fn batched_matches_per_token_non_multiple() {
        let d_head = 8;
        let block_size = 4;

        for &seq_q in [1usize, 3, 7].iter() {
            for &seq_kv in [5usize, 13, 17, 22].iter() {
                for &n_heads_kv in [1usize, 2].iter() {
                    for &topk in [1usize, 3].iter() {
                        for causal in [false, true] {
                            let n_heads_q = n_heads_kv * 2;
                            let scale = (d_head as f64).sqrt();
                            let batch = 1;

                            let q = Tensor::<4>::random(
                                [batch, n_heads_q, seq_q, d_head],
                                Distribution::Normal(0.0, 1.0),
                                &device(),
                            );
                            let k = Tensor::<4>::random(
                                [batch, n_heads_kv, seq_kv, d_head],
                                Distribution::Normal(0.0, 1.0),
                                &device(),
                            );
                            let v = Tensor::<4>::random(
                                [batch, n_heads_kv, seq_kv, d_head],
                                Distribution::Normal(0.0, 1.0),
                                &device(),
                            );

                            let n_blocks = seq_kv.div_ceil(block_size);
                            let topk_actual = topk.min(n_blocks);
                            let per_group: Vec<i64> = (0..seq_q as i64)
                                .flat_map(|_| (0..topk_actual as i64).collect::<Vec<_>>())
                                .collect();
                            let idx_data: Vec<i64> = per_group.repeat(n_heads_kv);
                            let bi = Tensor::<4, Int>::from_data(
                                TensorData::new(idx_data, [batch, n_heads_kv, seq_q, topk_actual]),
                                &device(),
                            );

                            let (ref_out, ref_ba) = sparse_attn_per_token(
                                q.clone(),
                                k.clone(),
                                v.clone(),
                                bi.clone(),
                                scale,
                                block_size,
                                n_heads_kv,
                                n_heads_q,
                                causal,
                            );
                            let (batched_out, batched_ba) = sparse_attn_batched_gqa(
                                q, k, v, bi, scale, block_size, n_heads_kv, n_heads_q, causal,
                            );

                            for (name, r, b) in
                                [("out", ref_out, batched_out), ("ba", ref_ba, batched_ba)]
                            {
                                let rv = extract_f32(r);
                                let bv = extract_f32(b);
                                assert_eq!(rv.len(), bv.len());
                                for (i, (a, c)) in rv.iter().zip(bv.iter()).enumerate() {
                                    if a.is_nan() && c.is_nan() {
                                        continue;
                                    }
                                    if *a == 0.0 && *c == 0.0 {
                                        continue;
                                    }
                                    let abs_diff = (a - c).abs();
                                    let rel_diff =
                                        abs_diff / a.abs().max(c.abs()).max(f32::EPSILON);
                                    assert!(
                                        abs_diff <= 1e-5 || rel_diff <= 1e-4,
                                        "{name} mismatch idx={} seq_q={} seq_kv={} hkv={} topk={} causal={}: ref={} got={} abs={:.2e} rel={:.2e}",
                                        i, seq_q, seq_kv, n_heads_kv, topk, causal, a, c,
                                        abs_diff, rel_diff
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
