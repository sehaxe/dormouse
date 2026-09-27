# burn-gdn2 - Gated DeltaNet 2

[![Crates.io](https://img.shields.io/crates/v/burn-gdn2)](https://crates.io/crates/burn-gdn2)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)
[![docs.rs](https://img.shields.io/docsrs/burn-gdn2)](https://docs.rs/burn-gdn2)

**Linear-complexity recurrent token mixer** for [Burn](https://burn.dev),
implementing Gated DeltaNet 2
([paper](https://arxiv.org/abs/2605.22791), Hatamizadeh et al., 2026;
[code](https://github.com/NVlabs/GatedDeltaNet-2)).
Channel-wise erase and write gates replace the scalar write-strength gate of
the gated delta rule. O(T) complexity instead of O(T²).

## Why this crate instead of the reference repo

| | burn-gdn2 | NVlabs reference |
|---|---|---|
| Backends | CPU, CUDA, Vulkan, Metal, WGPU (any burn backend) | **NVIDIA only** (Triton + flash-attn) |
| Dependencies | burn only | lit-gpt + fla + einops + transformers + triton + flash-attn |
| Backward pass | exact fused autodiff op (matrix-level WY adjoint, M⁻¹ reuse) | hand-written Triton kernels per op |
| Verification | 1000-case comparison against an independent transcription (regenerated with `tests/gen_reference.py`) | none shipped |
| Extras | `min_decay` (per-channel decay floor), GVA, `allow_neg_eigval` | GVA, `allow_neg_eigval` |

## Install

```bash
cargo add burn-gdn2
```

Enable CUDA:

```toml
burn-gdn2 = { version = "0.7", features = ["cuda"] }
```

## Quick start

```rust
use burn::tensor::{backend::Backend, Distribution, Tensor};
use burn_gdn2::{Gdn2Config, Gdn2Mode, GatedDeltaNet2, Gdn2State};

type B = burn_ndarray::NdArray;
let device = B::Device::default();
let cfg = Gdn2Config {
    hidden_size: 256,
    num_heads: 4,
    head_dim: 64,
    mode: Gdn2Mode::Chunk, // training mode
    ..Default::default()
};
let model = GatedDeltaNet2::<B>::new(&cfg, &device);

// Training - chunked WY forward.
let x = Tensor::<B, 3>::random([1, 1024, 256], Distribution::Normal(0.0, 1.0), &device);
let output = model.forward_train(x);

// Training with the fused autodiff op (one graph node, exact backward,
// ~3-5x faster end-to-end on CUDA; requires `Autodiff<B>` with the `autodiff` feature).
let output = model.forward_train_fused(x);

// Inference - prefill, then decode token-by-token with persistent state.
let mut state: Option<Gdn2State<B>> = None;
let out = model.forward(x, &mut state, true);          // prefill (or any chunk)
let token = Tensor::<B, 3>::random([1, 1, 256], Distribution::Normal(0.0, 1.0), &device);
let next = model.forward(token, &mut state, true);     // decode
```

`Gdn2State` carries both the recurrent matrix state `S [B, HV, K, V]` and the
short-convolution caches. This is what makes token-by-token decoding **exactly
equivalent** to one forward pass over the full sequence (verified bit-exact by
`tests/autodiff.rs`).

## Architecture

```
x ──→ Q/K/V/B/W/G projections ──→ ShortConv ──→ per-head L2-Norm
                                        │
                     ┌──────────────────┤
                     ▼                  ▼
             Chunk WY (train/long)  Fused Recurrent (decode)
                     │                  │
                     ▼                  ▼
               Gated RMS Norm ←── SiLU Gate
                     │
                     ▼
                Output Proj
```

Per-token recurrence on the per-head matrix state `S ∈ R^{d_k × d_v}`:

```
S_t = (I - k_t (b_t ⊙ k_t)^T) diag(α_t) S_{t-1} + k_t (w_t ⊙ v_t)^T
o_t = q_t^T S_t
```

`b_t` is the channel-wise erase gate (key axis), `w_t` the channel-wise write
gate (value axis); `α_t` is the per-channel decay from `A_log` and
`softplus(F(x) + dt_bias)`.

### Fused CUDA kernels

On the bare CUDA backend the whole chunked forward runs as **two custom cubecl
kernels** instead of the per-chunk tensor-op loop:

- `gdn2_chunk_intra_kernel` — chunk-local precomputation (decays, score
  matrices, `A = (I + T)^{-1}`, pseudo-keys/values) in a single launch.
- `gdn2_chunk_inter_kernel` — the sequential recurrence across chunks with
  the in-loop output phase, prefetching `w`/`kgd` through shared memory when
  the head dim allows.

Dispatch is gated on `burn_gdn2::CudaBare` (the bare `CubeBackend`, exported
for convenience); every other backend transparently falls back to the tensor
path. The fused path is numerically verified against the tensor path
(`fused_kernel_matches_tensor_path` in `tests/bench_cuda.rs`).

## Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `hidden_size` | 2048 | Model dimension |
| `num_heads` | 16 | Query/key heads |
| `head_dim` | 128 | Dimension per key/query head |
| `expand_v` | 1.0 | Value expansion (must keep `head_v_dim` integer) |
| `num_v_heads` | None | Value heads; GVA when > `num_heads` (must divide it) |
| `use_short_conv` | true | Causal depthwise conv (kernel 4) before the recurrence |
| `allow_neg_eigval` | false | Lift erase gate to `[0, 2]` (negative eigenvalues) |
| `mode` | FusedRecurrent | `Chunk` (training) or `FusedRecurrent` |
| `chunk_size` | 64 | Chunk size for `Chunk` mode |
| `norm_eps` | 1e-5 | Output norm epsilon |
| `min_decay` | None | Optional per-channel decay floor (extension, not in the paper) |

Invalid configurations (fractional `expand_v`, non-divisible `num_v_heads`,
`num_v_heads < num_heads`, `chunk_size = 0`) panic with a clear message at
construction.

## Performance

Measured on RTX 5060 Ti, release build, fp32, warmup + averaged loop timing.
burn side: Burn 0.22-pre (`Autodiff<CudaBare>`, `tests/bench_train_cuda.rs`);
torch side: `bench_torch.py` (pure PyTorch GDN-2, same math, cuBLAS +
autograd, no Triton). Same configs, same GPU. `Chunk` mode, chunk 64 in both.

### Track 1 — plain tensor ops (work on every backend: CPU/CUDA/wgpu)

| Config | burn-gdn2 tensor path | PyTorch chunked ops | vs PyTorch |
|--------|-----------------------|---------------------|-----------|
| d=256, T=256 | ~100 ms | 52.1 ms | parity |
| d=512, T=1024 | 62 ms | 195.6 ms | **3.2×** |
| d=1024, T=2048 | 176 ms | 402.9 ms | **2.3×** |
| d=2048, T=4096 | 219 ms | 804.7 ms | **3.7×** |

Same category on both sides: cuBLAS-backed tensor ops (matmul, cumsum,
exp), no fused kernels, no autograd. burn-gdn2 wins on launch count: the WY
solve is factorized through `M⁻¹` (one inverse per chunk, all solves become
single matmuls) and the decay cumsum runs through the dedicated cumsum op
instead of a 0/1-masked matmul. At d=256 both sides are launch-bound and the
comparison is parity.

### Track 2 — fused kernels vs fused kernels (CUDA only)

Same scope on both sides: the chunk recurrence only, on precomputed
q/k/v/g/b/w. burn side: `fused_chunk_forward` (intra + inter, 2 launches per
chunk) at **chunk ≤ 16** — the numerical limit of the naive decay form, the
same design choice as FlashKDA (`exp(cumsum(g))` underflows f32 once
`cumsum(g) < -88`, i.e. chunk > 17 at the K3 floor `g = -5`); larger chunks
run the K3 16-tile log-space tensor path, which is stable at any length.
torch side: the reference Triton kernels from the NVlabs repo (`chunk_gdn2`),
the only fused GDN-2 kernel in the PyTorch ecosystem — PyTorch itself ships
none.

| Config | burn-gdn2 fused | NVlabs Triton | vs Triton |
|--------|-----------------|---------------|-----------|
| d=256, T=256 | ~0.03 ms | 0.561 ms | **~18× faster** |
| d=512, T=1024 | ~0.06 ms | 0.536 ms | **~9× faster** |
| d=1024, T=2048 | ~0.15 ms | 0.628 ms | **~4× faster** |
| d=2048, T=4096 | ~0.8 ms | 1.461 ms | **~2× faster** |

(The burn numbers are kernel-only measurements; the earlier README table
mixed full-module forward with the Triton recurrence-only numbers — the
module adds ~0.7 ms of projection/launch overhead at the smallest config.)

### Full model forward (module incl. projections)

| Config | burn-gdn2 module | PyTorch full chunked | vs PyTorch |
|--------|------------------|----------------------|-----------|
| d=256, T=256 | ~0.9 ms | 52.1 ms | **~58×** |
| d=512, T=1024 | ~0.9 ms | 195.6 ms | **~215×** |
| d=1024, T=2048 | ~1.0 ms | 402.9 ms | **~400×** |
| d=2048, T=4096 | ~1.0 ms | 804.7 ms | **~800×** |

### Training (forward + backward)

| Config | burn-gdn2 fused op | PyTorch chunked + autograd | vs PyTorch |
|--------|--------------------|----------------------------|-----------|
| d=256, T=256 | 28.5 ms | 260.3 ms | **9.1×** |
| d=512, T=1024 | 22.4 ms | 1140.6 ms | **51×** |
| d=1024, T=2048 | 45.4 ms | 2188.5 ms | **48×** |
| d=2048, T=4096 | 141.7 ms | 4236.8 ms | **30×** |

The training path never re-runs the forward: the intra kernel exports
`M⁻¹ = (I+L)⁻¹`, `aqk`, `qE` and the decay factors (one extra buffer write),
the op rebuilds the backward scratch from them (`E = k·glast/kgd`), and the
backward recomputes only ~8 cheap ops per chunk (`W = M⁻¹·rhs` etc.). The
per-chunk row-by-row inversion and the full-forward recompute are both gone
(gradients verified against the tensor path on CUDA,
`tests/fused_chunk_verify.rs`). Memory per training step: the op holds 4
small tensors per chunk (vs 11 before) and checkpoints 5 inputs instead of 7
(`q`/`g` are recoverable as `qE`/`E` from the scratch) — roughly **half the
activation memory** at d=2048, T=4096 (~330 MB vs ~565 MB).

How this is achieved:

- **Forward**: the two fused chunk kernels (`src/kernel/chunk_cube.rs`)
  collapse each chunk into two launches instead of ~150 tensor ops, and are
  wired into the module (`Chunk` mode, `CudaBare`) and into the fused op.
  Kahan-compensated decay cumsum keeps them within ~1e-4 of the tensor path
  (which itself carries more noise; the kernels were verified against an
  exact reference at ~1e-7, `tests/fused_chunk_verify.rs`).
- **Backward**: the whole chunked WY recurrence is **one autodiff node**
  (`GatedDeltaNet2::forward_train_fused`, `src/autodiff.rs`) with an exact
  matrix-level adjoint. The WY solve factorizes through `M⁻¹` (computed once
  per chunk), so forward `W = M⁻¹·rhs` and backward `d_rhs = M⁻ᵀ·d_*` are
  single matmuls — the per-row solve loops and the `M⁻ᵀ` back-substitution
  loop are gone. Inter-chunk BPTT through the state is exact.
- The plain `forward_train` on an autodiff backend instead records one tracked
  node per tensor op (~16k graph nodes for a 2048-token forward), each with
  node allocation, checkpoint bookkeeping and a separate launch; the fused op
  removes all of that. On `Autodiff<CudaBare>` this is worth ~5-10× end-to-end
  over the tensor path alone.

Run it yourself:

```bash
# burn side: training (fused op vs plain path) + forward, same configs as torch
cargo test --release --features "cuda,autodiff" -p burn-gdn2 --test bench_train_cuda -- --ignored --nocapture
# torch side (pure PyTorch chunked WY, autograd; needs torch>=2.7 cu128 for RTX 50xx)
python3 bench_torch.py
# fused forward kernels vs the tensor path
cargo test --release --features cuda -p burn-gdn2 --test bench_cuda -- --ignored --nocapture
```

Notes:
- `Chunk` mode wins for training and long sequences; `forward()` dispatches
  to fused automatically for sequences of ≤ 64 tokens (decode).
- Projection costs are identical to any linear layer; the recurrence itself
  is the constant part.
- Per-token decode (1 token per call, ~1.2k tok/s on the RTX 5060 Ti) is
  dominated by the projections; the recurrence step itself runs through a
  single fused cubecl kernel (`src/kernel/fused_recurrent_cube.rs`) instead
  of ~8 tensor ops. Dispatch is gated on the bare CUDA backend
  (`burn_gdn2::CudaBare`, re-exported for convenience); the default
  `burn_cuda::Cuda` is `Fusion<CubeBackend>`-wrapped and has no public way to
  hand out its underlying `CubeTensor`, so it transparently falls back to the
  tensor path.

## Tests

```bash
cargo test -p burn-gdn2                                   # unit + autodiff + decode
cargo test -p burn-gdn2 --features autodiff --test autodiff_chunk  # fused op grads == tensor path + finite differences
cargo test -p burn-gdn2 --features binary-tests           # 1000-case vs an independent transcription (NOT bit-for-bit)
cargo test -p burn-gdn2 --features "cuda,autodiff" --test fused_chunk_verify  # kernels vs tensor path, CUDA grads
python3 tests/gen_reference.py                            # regenerate tests/ref_data.bin
```

### API

- `GatedDeltaNet2::forward` — inference (prefill/decode with `Gdn2State`),
  auto-dispatches to the fused recurrent kernel on `CudaBare` for short steps.
- `GatedDeltaNet2::forward_train` — training on any backend (tensor ops).
- `GatedDeltaNet2::forward_train_fused` — training through the fused autodiff
  op (one graph node, exact backward; needs `Autodiff<B>` + the `autodiff`
  feature; CUDA forward runs the fused kernels).
- `chunk_wy_forward` / `chunk_wy_forward_autodiff` — raw chunked WY forward,
  with/without the fused op.

### Backend / feature matrix

| | CPU (NdArray) | CUDA | other (wgpu/rocm/...) |
|---|---|---|---|
| tensor ops (`forward_train`) | ✓ | ✓ | ✓ |
| fused op (`forward_train_fused`) | ✓ (tensor forward) | ✓ (kernels + exact backward) | ✓ (tensor forward) |
| fused kernels (`CudaBare` only) | — | ✓ | — |

## License

MIT. See [LICENSE](LICENSE).

## Fused backward

The chunked op's backward runs on two fused kernels per call (BK2: sequential
BPTT state chain; BK1: token-parallel intra-chunk adjoint) plus ~15 batched
tensor ops in the glue, instead of ~37 tensor ops per chunk. The forward
kernels export M^-1/aqk/qE/decay/v_new/states, so the backward never re-runs
the forward. Attention-core kernels (fwd+bwd, RTX 5060 Ti, chunk 16):

| config | fused fwd+bwd | fla train (Triton) | old tensor-path train |
|--------|---------------|--------------------|----------------------|
| d=512, T=1024 | 0.32 ms | 2.8 ms | ~80 ms |
| d=1024, T=2048 | 0.52 ms | 4.7 ms | ~148 ms |
| d=2048, T=4096 | 1.34 ms | 4.6 ms | ~284 ms |
| d=4096, T=8192 | 0.32 ms | 8.2 ms | ~733 ms |

`tests/fused_chunk_verify.rs` checks the fused-op gradients against the tensor
path within fp32 noise (k's chunk-boundary /E amplification is allowed up to
10%).
