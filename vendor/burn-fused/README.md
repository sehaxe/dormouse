# burn-fused

Community fused-kernel ecosystem for [Burn](https://burn.dev): one workspace,
one meta-crate, 100+ technologies in sight. Every kernel is fused (single
launch), every forward has a fused backward where it matters, and every
performance claim is backed by a benchmark that CI re-checks on every PR.

Not affiliated with the official burn project.

## Layout

```
crates/            28 crates: fused kernels + ops-only technologies (burn-rope, burn-mhc, burn-ttt, ...)
burn-fused/        the meta-crate: `burn-fused = { features = ["cuda"] }`
benches/           perf harness + regression gate (GPU runner)
```

## Quick start

```toml
[dependencies]
burn-fused = { git = "https://github.com/sehaxe/burn-fused", features = ["cuda"] }
```

```rust
use burn_fused::burn_rope::{apply_rope_4d, precompute_freqs};
// or any other crate in the workspace
```

**Integrating this workspace? Read [`burn-fused/INTEGRATION.md`](burn-fused/INTEGRATION.md)**
(it is the facade's crates.io readme): which `burn` version you need, the
feature flags, which backend types actually reach a fused kernel (and the
`fusion` footgun that silently disables all of them), the precision support
per mechanism, and a three-line gated delta net.

To add a new technology: new crate under `crates/`, add it to the workspace
members, run `tools/gen_facade.py` (it generates the facade's re-export list
and its feature flags from the manifests) and commit the result, add a case to
`benches/src/main.rs`, seed the baseline. See `CONTRIBUTING.md`.

## Benchmarks (RTX 3090, burn 0.22.0-pre.2, bare CUDA backend)

Fused timings below are the committed CI baselines (`bench/baselines.json`,
re-checked on every PR, fails on >20% regressions). Speedup ranges vs the
naive tensor chain and the PyTorch head-to-head are documented separately in
`bench/PYTORCH_COMPARISON.md` (measured 2026-08-09; the fused column there
predates later kernel work, so trust the baselines table for current numbers).

| kernel | fused baseline (ms) | speedup vs naive tensor chain | crate |
|--------|--------------------:|------------------------------|-------|
| AttnRes depth-attend (24×1×2048×4096) | 6.70 | 2-10.6× | burn-attnres |
| FWT (bitnet, 256×512) | 0.011 | 11× | burn-bitnet |
| GDN2 chunked prefill (1×4×128×64) | 0.113 | ~104× vs per-token loop | burn-gdn2 |
| KDA chunked prefill (1×8×128×64) | 0.119 | reuses GDN2 fused chunk | burn-kda |
| Sinkhorn (MHC, 8×2048×16×16) | 1.90 | 19-41K× | burn-mhc |
| Muon+ column-row norm (2048×5120) | 0.43 | 96× | burn-muon-plus |
| RoPE (4×2048×32×128, chunked cube) | 0.71 | 1.5× | burn-rope |
| SCT QR from_dense (512×256, 15 sweeps) | 52.3 | batched-retraction path | burn-sct |
| Situ-GLU (2048×5120) | 0.32 | 2.2× | burn-situ |

`BURN_DEVICE=cuda cargo run -p burn-fused-benches --release` reproduces the
fused timings. Configs are fixed so regressions are comparable run to run.

## Conventions

- Burn 0.22.0-pre.4 (the baselines below were measured on pre.2), Rust stable
  (workspace MSRV 1.85; leaf crates
  may raise it locally — currently burn-gdn2 requires 1.95), edition 2021, MIT.
- Fused dispatch: `try_into_primitive` + downcast to the bare
  `CubeBackend<CudaRuntime>`; grads via burn-autodiff `Ops`/`Backward`/`Checkpointer`.
  Fused kernels hard-require f32 buffers and fall back to the tensor path for
  any other dtype.
- Every non-trivial kernel ships a finite-difference (or burn-autodiff)
  gradient check in `tests/` and a benchmark entry in `benches/`.
- Feature matrix is uniform across kernel crates: `std`, `cuda`, `autodiff`
  (ops-only crates ship `std`). The facade's flag list is GENERATED from
  these manifests by `tools/gen_facade.py`; CI fails if it drifts.

## Roadmap

- [x] perf-regression coverage for all 10 workspace crates (bench harness + baselines)
- [x] rope flat-grid kernel (262k -> 8k cubes, 1.6-2.3x faster)
- [x] msa topk<4: root cause investigated (cubecl kernel-level defect), safe guard kept, documented in `sparse_kernel.rs`
- [x] port all crates from 0.21 to 0.22 (workspace now has 28 crates)
- [ ] flagship `burn-flash-attention`
- [ ] training example + inference example per family
- [ ] CI benches on CPU/ROCm/Metal/WebGPU in addition to CUDA
