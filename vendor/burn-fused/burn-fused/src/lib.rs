//! # burn-fused
//!
//! Community fused-kernel ecosystem for [Burn](https://burn.dev): a single
//! dependency that pulls in the whole workspace of fused technologies behind
//! feature flags.
//!
//! Not affiliated with the official burn project. Every crate cites its paper
//! source in its README.
//!
//! Every workspace crate is re-exported here unconditionally (all are
//! non-optional dependencies); the `std`/`cuda` features only toggle each
//! member's runtime features, so `use burn_fused::burn_<name>` always works
//! while the feature matrix controls what the members themselves enable.

pub use burn_antihall;
pub use burn_attnres;
#[cfg(feature = "cuda")]
pub use burn_autodiff;
pub use burn_bitnet;
pub use burn_byteflow;
pub use burn_diffusionblocks;
pub use burn_dspark;
pub use burn_eggroll;
pub use burn_engram;
pub use burn_es;
pub use burn_fastblt;
pub use burn_gdn2;
pub use burn_jepa;
pub use burn_kda;
pub use burn_mhc;
pub use burn_mod;
pub use burn_mor;
pub use burn_msa;
pub use burn_mtp;
pub use burn_muon_plus;
pub use burn_nope;
pub use burn_parcae;
pub use burn_ptrn;
pub use burn_rmsnorm;
pub use burn_rope;
pub use burn_sct;
pub use burn_situ;
pub use burn_spectral;
pub use burn_swiglu;
pub use burn_ttt;
