# TODO — burn-* crate roadmap (community-facing)

## Priority: autodiff-aware fused kernels (burn-gdn2, shared by burn-kda)
- [x] Stage 1: custom autodiff op — fused forward + tensor-path backward
      (exact grads). `src/autodiff.rs`: `chunk_wy_forward_autodiff` +
      `GatedDeltaNet2::forward_train_fused`. Backward = matrix-level WY
      adjoint (M^-T solves, full inter-chunk BPTT through the state).
      Verified: `tests/autodiff_chunk.rs` (op grads == tensor grads on CPU
      ref + finite differences + model-level grads). CUDA training:
      forward+backward ~5x faster than the per-op tensor path.
- [ ] Stage 2: hand-written backward kernels: intra (transposed WY solve +
      grads w.r.t. q/k/v/g/b/w), inter (chunked BPTT). Bonus: log-space
      decay inside the kernel -> fused works at chunk 64, kda cap removed.
- [ ] New-state grad tracking (currently `new_state` is an untracked leaf;
      input-state grads are exact). Needed for manual BPTT chaining through
      the returned state.

## Community polish (when reached)
- [ ] MSRV / rust-version in Cargo.toml, CI with GPU tests
- [ ] docs.rs rendering check (do kernels show in cargo doc?)
- [ ] examples/ per crate (quick-start runnable)

## Later: write cubecl kernels directly (user's call, per-crate as needed)
