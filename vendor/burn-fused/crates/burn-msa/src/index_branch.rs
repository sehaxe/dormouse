use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::tensor::Device;
use burn::tensor::{ops::PadMode, Int, Tensor};

use crate::config::MsaConfig;

#[derive(Module, Debug)]
pub struct IndexBranch {
    pub q_proj: Linear,
    pub k_proj: Linear,
}

impl IndexBranch {
    pub fn new(cfg: &MsaConfig, device: &Device) -> Self {
        Self {
            q_proj: LinearConfig::new(cfg.d_model, cfg.n_heads_kv * cfg.d_idx)
                .with_bias(false)
                .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                .init(device),
            k_proj: LinearConfig::new(cfg.d_model, cfg.d_idx)
                .with_bias(false)
                .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                .init(device),
        }
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<3>,
        kv_states: Tensor<3>,
        n_heads_kv: usize,
        d_idx: usize,
    ) -> (Tensor<4>, Tensor<4>) {
        let q_3d = self.q_proj.forward(hidden_states);
        let k_3d = self.k_proj.forward(kv_states);

        let [batch, seq_q, _] = q_3d.dims();
        let [_, _seq_kv, _] = k_3d.dims();

        let q_idx = q_3d
            .reshape::<4, _>([batch, seq_q, n_heads_kv, d_idx])
            .swap_dims(1, 2);

        let k_idx = k_3d.unsqueeze_dims(&[1]);

        (q_idx, k_idx)
    }

    pub fn compute_block_scores(
        &self,
        q_idx: Tensor<4>,
        k_idx: Tensor<4>,
        block_size: usize,
        scale: f64,
        causal: bool,
    ) -> Tensor<4> {
        let [batch, h_kv, seq_q, d_idx] = q_idx.dims();
        let seq_kv = k_idx.dims()[2];
        let n_blocks = seq_kv.div_ceil(block_size);

        // Chunked block scores: matmul q·k_chunk^T over `CHUNK` blocks at a
        // time instead of the full [B, Hkv, S, S] matrix, then block-max each
        // chunk. This caps the peak tensor at [B, Hkv, S, CHUNK·block_size]
        // (~8× less than the full S² scores at CHUNK=8) — the dominant memory
        // consumer in MSA training at long sequences.
        const CHUNK: usize = 8;
        let padded_kv = n_blocks * block_size;
        let k_pad = if padded_kv > seq_kv {
            k_idx.pad(
                [(0, 0), (0, 0), (0, padded_kv - seq_kv), (0, 0)],
                PadMode::Constant(0.0),
            )
        } else {
            k_idx
        };
        let device = q_idx.device();
        let q_pos =
            Tensor::<1, Int>::arange(0..seq_q as i64, &device).reshape::<4, _>([1, 1, seq_q, 1]);
        let kv_pos = Tensor::<1, Int>::arange(0..padded_kv as i64, &device)
            .reshape::<4, _>([1, 1, 1, padded_kv]);

        let mut out_chunks = Vec::with_capacity(n_blocks.div_ceil(CHUNK));
        for start in (0..n_blocks).step_by(CHUNK) {
            let end = (start + CHUNK).min(n_blocks);
            let c = end - start;
            let k_chunk = k_pad.clone().slice([
                0..batch,
                0..h_kv,
                start * block_size..end * block_size,
                0..d_idx,
            ]);
            let mut sc = q_idx
                .clone()
                .matmul(k_chunk.swap_dims(2, 3))
                .div_scalar(scale);

            // causal mask: kv_pos > q_pos -> -1e4 (only for real tokens)
            if causal && seq_q > 1 && seq_kv > 1 {
                let q_pos_c = q_pos.clone();
                let kv_pos_c =
                    kv_pos
                        .clone()
                        .slice([0..1, 0..1, 0..1, start * block_size..end * block_size]);
                sc = sc.add(
                    kv_pos_c
                        .greater(q_pos_c)
                        .float()
                        .mul_scalar(crate::NEG_INF_SAFE),
                );
            }
            // padding: tokens >= seq_kv -> masked out of the block max.
            // Finite NEG_INF_SAFE, not -inf: (1-invalid) * -inf would turn
            // the 0 side into NaN (0 * inf) and poison every real column.
            if padded_kv > seq_kv {
                let invalid = kv_pos
                    .clone()
                    .slice([0..1, 0..1, 0..1, start * block_size..end * block_size])
                    .lower_equal_elem((seq_kv - 1) as i64)
                    .float();
                sc = sc.mul(invalid.clone()).add(
                    invalid
                        .neg()
                        .add_scalar(1.0)
                        .mul_scalar(crate::NEG_INF_SAFE),
                );
            }
            out_chunks.push(
                sc.reshape::<5, _>([batch, h_kv, seq_q, c, block_size])
                    .max_dim(4)
                    .squeeze_dim::<4>(4),
            );
        }
        Tensor::cat(out_chunks, 3)
    }
}
