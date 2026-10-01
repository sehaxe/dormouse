//! # burn-gdn2
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! Gated DeltaNet 2 (GDN-2) - a linear‑complexity recurrent token mixer
//! with channel‑wise erase/write gates.
//!
//! ## Quick start
//!
//! ```rust
//! use burn_gdn2::{Gdn2Config, Gdn2Mode, GatedDeltaNet2};
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

// The seam's COUNTERS are called from two halves of the crate that are gated
// differently - the fused KERNELS by `cuda` (`kernel/chunk_cube.rs`,
// `kernel/chunk_adjoint_cube.rs`) and the fused autodiff NODE by `autodiff`
// (`autodiff.rs`). Gating this module on one of them made the counters
// unreachable from the other, which is not a slow build but a COMPILE ERROR
// (E0433) in `cuda` without `autodiff` and in `autodiff` without `cuda` - and
// it did: `cargo check -p burn-gdn2 --features cuda` has not compiled since the
// counters moved, and the CI job that reports it (`cuda-compiles` in
// `.github/workflows/fused-library.yml`) has been red in every run it has ever
// had. `any(...)` is the fix and `all(...)` would be the same bug wearing a
// hat: the seam is what makes an arm visible, so a build that cannot count it
// is a build that cannot report it.
#[cfg(any(feature = "cuda", feature = "autodiff"))]
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

// Which arm ran, and how to ask. These three need only `Backend`, so they
// exist in a `cuda`-only build too - and a `cuda`-only build is what a
// downstream crate gets: `crates/burn-kda/src/fused.rs:118-122` is gated on
// `cuda` and names `burn_gdn2::Fused` / `burn_gdn2::Fallback`, so under the
// old `autodiff`-only gate this re-export took `cargo check -p burn-kda
// --features cuda`, the facade's `std,cuda` combination and
// `cargo check -p burn-fused-benches` down with it.
#[cfg(any(feature = "cuda", feature = "autodiff"))]
pub use cuda_dispatch::{backend_matches, Fallback, Fused};

#[cfg(feature = "autodiff")]
pub use cuda_dispatch::{rebuild, strip, AdNode};

#[cfg(all(feature = "cuda", feature = "autodiff"))]
pub use cuda_dispatch::{
    chunk_dispatch, custom_node_backward, dispatch_asked, fused_calls, fused_declined, ops_path,
    reset_fused_calls, seam_counts, try_strip, FusedCudaAutodiff,
};

#[cfg(feature = "cuda")]
pub use kernel::fused_recurrent_cube::cuda::CudaBare;
