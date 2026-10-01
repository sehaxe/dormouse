//! # dormouse-gdn2
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! Gated DeltaNet 2 (GDN-2) - a linear‑complexity recurrent token mixer
//! with channel‑wise erase/write gates.
//!
//! ## Quick start
//!
//! ```rust
//! use dormouse_gdn2::{Gdn2Config, Gdn2Mode, GatedDeltaNet2};
//! use burn::tensor::{Device, Distribution, Tensor};
//!
//! let device = Device::ndarray();
//! let config = Gdn2Config {
//!     hidden_size: 64,
//!     num_heads: 2,
//!     head_dim: 32,
//!     mode: Gdn2Mode::Chunk,
//!     ..Default::default()
//! };
//! let model = GatedDeltaNet2::new(&config, &device);
//!
//! // Training: chunked WY forward over the full sequence.
//! let x = Tensor::<3>::random(
//!     [1, 32, 64],
//!     Distribution::Normal(0.0, 1.0),
//!     &device,
//! );
//! let output = model.forward_train::<burn::backend::NdArray>(x);
//! assert_eq!(output.shape().dims(), [1, 32, 64]);
//!
//! // Inference: token-by-token with persistent state.
//! let mut state = None;
//! let token = Tensor::<3>::random(
//!     [1, 1, 64],
//!     Distribution::Normal(0.0, 1.0),
//!     &device,
//! );
//! let output = model.forward::<burn::backend::NdArray>(token, &mut state, true);
//! assert_eq!(output.shape().dims(), [1, 1, 64]);
//! ```
//!
//! ## Features
//!
//! - **`std`** (default) - standard library support
//! - **`autodiff`** - differentiation support (required for training)
//! - **`cuda`** - CUDA backend support
//! - **`binary-tests`** - the 1000-case fixture-backed oracle tests
//!   (`tests/oracle_breadth.rs`, `tests/oracle_chunk.rs`). The fixture is
//!   `tests/ref_f64_broad.bin`, the f64 output of `tools/gen_reference_f64.py`:
//!   a **transcription** of arXiv:2605.22791 §3.1 Eq. 8-12, with the three
//!   details that are not in the paper each cited to the authors' own source.
//!   Compared at 1e-3 **relative**. Not bit-for-bit, and not tier (a) - a
//!   bit-for-bit claim needs NVlabs' Triton kernel in the tree. Not in `default`
//!   (it is a fork-CI marker no crate reads); `tools/lib_gate.sh` passes it
//!   explicitly. It replaced an f32 transcription of this crate's OWN algorithm,
//!   which could only ever prove self-consistency and had replicate-padded the
//!   short conv exactly as the kernel wrongly did. See `docs/protocols/ORACLE.md` §2-§3
//!   and the header of `tests/oracle_breadth.rs`.

pub mod alloc_trace;
pub mod config;
pub mod forward;
pub mod kernel;
pub mod l2norm;
pub mod module;
pub mod short_conv;

#[cfg(feature = "autodiff")]
pub mod cuda_dispatch;

#[cfg(feature = "autodiff")]
pub mod autodiff;

pub use config::{Gdn2Config, Gdn2Mode};
pub use forward::{chunk_path, chunk_wy_forward, set_chunk_path, ChunkPath};
pub use kernel::fused_recurrent::fused_recurrent_forward;
pub use l2norm::{l2_normalize, l2_normalize_4d};
pub use module::{rms_norm_gate_per_head, GatedDeltaNet2, Gdn2State, ProjectedInputs};
pub use short_conv::{short_conv_1d, SHORT_CONV_CACHE, SHORT_CONV_KERNEL};

#[cfg(feature = "autodiff")]
pub use autodiff::{chunk_autodiff_or_plain, chunk_wy_forward_autodiff, chunk_wy_forward_autodiff_s};

#[cfg(feature = "autodiff")]
pub use cuda_dispatch::{backend_matches, rebuild, strip, AdNode, Fallback, Fused};

#[cfg(all(feature = "cuda", feature = "autodiff"))]
pub use cuda_dispatch::{
    chunk_dispatch, custom_node_backward, dispatch_asked, fused_calls, fused_declined, ops_path,
    reset_fused_calls, seam_counts, try_strip, FusedCudaAutodiff,
};

#[cfg(feature = "cuda")]
pub use kernel::fused_recurrent_cube::cuda::CudaBare;
