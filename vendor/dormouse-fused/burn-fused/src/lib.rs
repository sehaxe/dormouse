//! # burn-fused
//!
//! One dependency for the whole fused-kernel workspace: every member
//! crate is re-exported here, and the feature flags are derived from the
//! members' own manifests. Both lists are GENERATED - run
//! `tools/gen_facade.py` after adding a crate or a feature, and CI
//! (`tools/gen_facade.py --check`) fails if you forget.
//!
//! Not affiliated with the official burn project. Every crate cites its
//! paper source in its README.
//!
//! Re-exports are unconditional on purpose: a member behind a `cfg` could
//! lose its `pub use` and still compile, which is exactly the rot this
//! file exists to prevent. What the features control is each member's own
//! runtime features, not whether the crate is reachable.
//!
//! Integration: `INTEGRATION.md` (this crate's README, and what CI
//! compiles). Which backend types reach the fused kernels, which features
//! exist, and the precision story per mechanism are all answered there.
//!
//! ```
//! # fn main() {
//! use burn::backend::NdArray;
//! use burn::tensor::{Distribution, Tensor};
//! use burn_fused::burn_kda::KdaModule;
//!
//! let device = Default::default();
//! let x = Tensor::<3>::random([1, 16, 128], Distribution::Default, &device);
//! let layer = KdaModule::new(&Default::default(), 0.0, &device);
//! let y = layer.forward_train::<NdArray>(x);
//! assert_eq!(y.dims(), [1, 16, 128]);
//! # }
//! ```

#[cfg(feature = "cuda")]
pub use burn_cuda;

#[cfg(feature = "autodiff")]
pub use burn_autodiff;

pub use burn_attnres;
pub use burn_bitnet;
pub use burn_dspark;
pub use burn_eggroll;
pub use burn_engram;
pub use burn_gdn2;
pub use burn_jepa;
pub use burn_kda;
pub use burn_mhc;
pub use burn_mor;
pub use burn_muon_plus;
pub use burn_parcae;
pub use burn_ptrn;
pub use burn_rmsnorm;
pub use burn_rope;
pub use burn_situ;
pub use burn_spectral;
pub use burn_swiglu;
