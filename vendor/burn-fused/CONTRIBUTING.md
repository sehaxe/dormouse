# Contributing

## Adding a technology

1. New crate `crates/burn-<name>` on Burn 0.22, edition 2021.
2. Uniform feature matrix: `std`, `cuda`, `autodiff` (list every optional dep
   with `dep:` syntax). Anything you add that is not one of those three
   (or is documented in `NOT_PUBLIC` in `tools/gen_facade.py`) fails
   `gen_facade.py --check` until someone decides whether users get it.
3. Add the crate to the workspace `members` and run `tools/gen_facade.py`:
   the facade's dependency list, re-export list and feature flags are
   generated. Do not hand-edit them; CI checks.
4. Fused dispatch pattern: accept any `B: Backend`, attempt the fused path
   via `try_into_primitive` + downcast to `CubeBackend<CudaRuntime>`, fall back
   to the tensor chain. Never let a fused path silently return wrong results.
5. The kernel is `f32`-only (`launch_unchecked::<f32>` hand-computes its byte
   length). If your fused entry point cannot be sure of the input dtype,
   check it and fall back, the way `burn-gdn2`/`burn-rmsnorm`/`burn-swiglu`
   do - `INTEGRATION.md` keeps the table of which members check and which
   will hand garbage to a `bf16` buffer.
6. Gradient support: burn-autodiff `Ops`/`Backward`/`Checkpointer`, const-generic
   N parents. A fused forward without a fused backward is acceptable only when
   the op is never trained (e.g. FWT is self-adjoint; say so in the README).
7. Tests: correctness `fuse == tensor` AND an independent gradient check
   (finite-difference or raw burn-autodiff). The cross-check alone misses
   systematic math errors. Cover the states your API accepts: a call that
   panics only for `state = None` (or only when `T != heads`) is untested,
   not correct.
8. Bench: add a case to `benches/src/main.rs`, seed the baseline on the GPU
   runner (`workflow_dispatch` with `update-baselines`), update the README table.
9. Document the paper/technique source in the crate README.

## Standards

- `cargo fmt` and `cargo clippy --workspace --all-targets -- -D warnings` clean.
- Every new kernel: one launch, no per-step syncs, no intermediate allocs
  where avoidable. `client.sync()` must be `block_on`'d (a dropped future is a no-op).
- Fused CUDA kernels must check the buffer dtype (f32 only) and return `None`
  for anything else so the tensor fallback keeps non-f32 inputs correct.
- Numbers in READMEs must come from `benches/`, not one-off runs. The README
  benchmark table mirrors `bench/baselines.json` (the CI-checked source);
  speedup ratios live in `bench/PYTORCH_COMPARISON.md`.
- Versions: the meta-crate and benches inherit from `[workspace.package]`;
  leaf crates keep their own version numbers (several are vendored standalone
  projects) but MUST share dependency versions through
  `[workspace.dependencies]` (`dep = { workspace = true, features = [...] }`).
- Workspace MSRV is 1.85; a leaf crate may raise `rust-version` locally when
  it genuinely needs newer rustc (currently: burn-gdn2 → 1.95) — note it here.
- GPU-side gotchas documented in this repo's CI are load-bearing: `H%8==0`
  alignment for coalesced writes, `a/b == (a/b)*b` integer division, deferred
  `random`/`zeros` materialization.

## CI

- `ci.yml`: fmt, clippy, workspace tests, and the facade's generated-file
  check + feature matrix (`tools/test-feature-matrix.sh`, CPU runner).
- `bench.yml`: perf regression gate on the self-hosted GPU runner, tolerance 20%.
  Run it with `update-baselines: true` to re-seed after intentional changes.
