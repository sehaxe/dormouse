//! dormouse-core - all-bf16 mini Aria on burn-fused kernels (CUDA)
pub mod act_quant;
pub mod attention;
pub mod config;
pub mod gr;
pub mod loop_block;
pub mod model;
pub mod param;

pub use attention::AdaptiveAttention;
pub use config::DormouseConfig;
pub use loop_block::{ExpertFFN, LoopBlock};
pub use model::DormouseModel;
pub use param::{bf16_on, LinearLike};

#[cfg(test)]
mod tests {
    #[test]
    fn fnv_deterministic() {
        // FNV-1a, mirroring dormouse_data::fnv (hashing lives on the data
        // side; this only pins the contract).
        let fnv = |b: &[u8]| {
            let mut h: u64 = 1469598103934665603;
            for &x in b {
                h ^= x as u64;
                h = h.wrapping_mul(1099511628211);
            }
            h
        };
        assert_eq!(fnv(b"hello"), fnv(b"hello"));
        assert_ne!(fnv(b"hello"), fnv(b"world"));
    }
}