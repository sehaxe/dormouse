//! dormouse-core - all-bf16 mini Aria on burn-fused kernels (CUDA)
pub mod act_quant;
pub mod aux;
pub mod attention;
pub mod config;
#[cfg(test)]
mod dspark_oracle;
pub mod future_byte;
pub mod gr;
pub mod loop_block;
pub mod model;
pub mod mor;
pub mod param;
pub mod probe;
pub mod routing;

pub use attention::{fused_seam_counts, kda_seam_counts, AdaptiveAttention};
pub use aux::{AuxHeads, TEACHER_MOMENTUM};
pub use config::{ActQuant, DormouseConfig};
pub use loop_block::{ExpertFFN, LoopBlock};
pub use model::DormouseModel;
pub use param::LinearLike;
pub use routing::{Group, GroupCounts, Role, Routed, Routing};

pub fn fnv_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, Int, Tensor};
    #[test]
    fn fnv_deterministic() {
        assert_eq!(fnv_hash(b"hello"), fnv_hash(b"hello"));
        assert_ne!(fnv_hash(b"hello"), fnv_hash(b"world"));
    }

    /// The in-loop L_Rec formula (gather the target log-prob) must equal the
    /// retired one-hot product row-for-row — the CE swap of 2026-09-04 was a
    /// bandwidth optimization, not a semantics change.
    #[test]
    fn gather_ce_matches_one_hot() {
        let dev = burn::tensor::Device::flex();
        let (n, v): (usize, usize) = (128, 256);
        let logits = Tensor::<2>::random([n, v], Distribution::Normal(0.0, 4.0), &dev);
        let tgt: Vec<i64> = (0..n).map(|i| (i * 7 + 3) as i64 % v as i64).collect();
        let lp = burn::tensor::activation::log_softmax(logits.clone(), 1);
        // new: gather; old: one-hot elementwise product
        let idx: Tensor<2, Int> = Tensor::from_data(
            burn::tensor::TensorData::new(tgt.clone(), [n, 1]),
            &dev,
        );
        let gathered = lp.clone().gather(1, idx);
        let one_hot = Tensor::<1, Int>::from_data(
            burn::tensor::TensorData::new(tgt.clone(), [n]),
            &dev,
        )
        .one_hot::<2>(v)
        .cast(burn::tensor::FloatDType::F32);
        let product = (lp * one_hot).sum_dim(1);
        let diff = (gathered - product).abs().max().into_scalar::<f32>();
        assert!(diff < 1e-4, "gather CE diverges from one-hot CE: {diff:.2e}");
    }
}