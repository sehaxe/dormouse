//! ByteFlow Net for Burn — tokenizer-free language modeling through adaptive
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//! byte compression ([Deng et al., ICLR 2026](https://arxiv.org/abs/2603.03583),
//! arXiv 2603.03583).
//!
//! One research, one crate: the full five-stage hierarchy of the paper.
//!
//! - [`chunk`] — coding-rate chunking (downsampling): exact lossy rate
//!   `R_ε(h) = ½ log det(I + (d/ε²) H Hᵀ)` and its Appendix B L2 streaming
//!   approximation, Top-K boundary selection with forced BOS position.
//! - [`net`] — local encoder / global transformer blocks (SWA + Canon layers +
//!   SwiGLU), multi-linear upsampling with large residual, symmetric decoder.
//!
//! ```no_run
//! use burn::tensor::{Int, Tensor};
//! use burn_byteflow::{ByteFlowConfig, ByteFlowNet};
//!
//! let device = burn::tensor::Device::ndarray();
//! let net = ByteFlowNet::init(ByteFlowConfig::default(), &device);
//! let bytes = Tensor::<2, Int>::zeros([1, 8192], &device);
//! let logits = net.forward(bytes); // [1, 8192, 256]
//! ```

pub mod chunk;
pub mod net;
pub mod stream;

pub use chunk::{
    coding_rate_exact, marginal_gains_exact, marginal_gains_l2, select_positions, RateMode,
};
pub use net::{
    ByteFlowConfig, ByteFlowNet, CanonLayer, FlowAttention, FlowBlock, RopeTable, VOCAB,
};
pub use stream::RatePatcher;
