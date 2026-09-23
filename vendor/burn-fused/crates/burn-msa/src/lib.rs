// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
pub const NEG_INF_SAFE: f32 = -1e4;

pub mod attention;
#[cfg(all(feature = "cuda", feature = "autodiff"))]
pub mod autodiff;
pub mod cache;
pub mod config;
pub mod index_branch;
pub mod kernel;
pub mod loss;
pub mod module;
#[cfg(feature = "cuda")]
pub mod sparse_kernel;
pub mod topk;

pub use attention::softmax;
pub use attention::sparse_attn_batched_gqa;
pub use attention::SparseAttention;
pub use cache::MsaCache;
pub use config::MsaConfig;
pub use index_branch::IndexBranch;
pub use loss::KlAlignmentLoss;
pub use module::MsaModule;
pub use module::MsaOutput;
pub use topk::TopKSelector;
