//! dormouse-core - all-bf16 mini Aria on burn-fused kernels (CUDA)
pub mod act_quant;
pub mod aux;
pub mod attention;
pub mod config;
pub mod gr;
pub mod loop_block;
pub mod model;
pub mod param;

pub use attention::AdaptiveAttention;
pub use aux::{AuxHeads, TEACHER_MOMENTUM};
pub use config::{ActQuant, DormouseConfig};
pub use loop_block::{ExpertFFN, LoopBlock};
pub use model::DormouseModel;
pub use param::LinearLike;

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
    #[test]
    fn fnv_deterministic() {
        assert_eq!(fnv_hash(b"hello"), fnv_hash(b"hello"));
        assert_ne!(fnv_hash(b"hello"), fnv_hash(b"world"));
    }
}