# burn-msa - MiniMax Sparse Attention

[![Crates.io](https://img.shields.io/crates/v/burn-msa)](https://crates.io/crates/burn-msa)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)
[![MS-DSA](https://img.shields.io/badge/stack-Burn%20%2B%20cubecl-purple.svg)](#architecture)

**Block-sparse attention** for the [Burn](https://burn.dev) deep learning
framework. Reduces attention from `O(S_q × S_kv)` to `O(S_q × topk × block_size)`
by selecting the top-K KV blocks per query position via a lightweight index branch.

> Paper: [MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392) (Lai et
> al., 2026). Reference code: [MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA).

Per-GQA-group top-k block selection with GPU-sync-free indexer. 5D batched
matmul kernel launches **one kernel per KV head** instead of per-token per-head.
Works on any Burn backend: CUDA, WGPU, NdArray.

## Requirements

- **Rust**: ≥ 1.85 (workspace MSRV)
- **Burn**: 0.22
- **GPU** (optional): CUDA or Vulkan/Metal via WGPU

## Install

```bash
cargo add burn-msa
```

Or in `Cargo.toml`:

```toml
[dependencies]
burn-msa = "0.2"
```

Enable CUDA:

```toml
burn-msa = { version = "0.2", features = ["cuda"] }
```

## Quick start

```rust
use burn_ndarray::NdArray;
use burn_msa::{MsaConfig, MsaModule};

let device = Default::default();
let cfg = MsaConfig::new(768, 12, 4, 64, 64);
let module = MsaModule::<NdArray>::new(&cfg, &device);

// Self-attention: Q, K, V from the same input
let x = Tensor::random([1, 128, 768], Distribution::Normal(0.0, 1.0), &device);
let result = module.forward(x);
let output = result.output;  // [batch, seq, d_model]

// Cross-attention: separate Q and KV inputs
let q = Tensor::random([1, 64, 768], Distribution::Normal(0.0, 1.0), &device);
let kv = Tensor::random([1, 256, 768], Distribution::Normal(0.0, 1.0), &device);
let result = module.forward_cross(q, kv);

// Dense GQA fallback (no sparsity)
let output = module.forward_dense(hidden_states, kv_states);
```

## Architecture

```
Input [B, T, D] ──→ IndexBranch ──→ TopKSelector ──→ SparseAttention ──→ Output
                      │                   │                  │
                      │ Q·K^T / √d_idx   │ topk blocks      │ gather K/V
                      │ per-block max    │ per GQA group    │ 5D matmul
                      │                  │                  │ softmax
                      │                  │                  │ output proj
```

1. **Index branch** projects Q/K into low-dim space (`d_idx`), scores each
   KV block via max-pooling.
2. **TopK selector** picks `topk` highest-scoring blocks per GQA group -
   GPU-sync-free.
3. **Sparse attention** gathers K/V from selected blocks via single batched
   gather, then 5D batched matmul for scores and output.

## Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `d_model` | 1152 | Hidden size |
| `n_heads_q` | 18 | Query heads |
| `n_heads_kv` | 1 | KV heads (GQA factor) |
| `d_head` | 64 | Head dimension |
| `d_idx` | 32 | Index projection dimension |
| `block_size` | 128 | Tokens per KV block |
| `topk` | 16 | Blocks selected per query |
| `causal` | true | Causal masking |
| `use_kl_loss` | true | KL alignment loss for index branch |
| `kl_coeff` | 0.1 | KL loss weight |



## KV-cache decoding

```rust
use burn_msa::MsaCache;

let cache = MsaCache::new(k, v, k_idx);
let cache = cache.update(new_k, new_v, new_k_idx);
```

## Layout

```
src/
  lib.rs              Crate root, re-exports
  attention.rs        SparseAttention, 5D batched GQA kernel
  index_branch.rs     Low-dim Q/K projections, block scoring
  topk.rs             GPU-sync-free top-K block selector
  module.rs           MsaModule: unified forward API
  config.rs           MsaConfig with validation
  loss.rs             KL alignment loss
  cache.rs            KV-cache for incremental decoding
  kernel/             Experimental cubecl kernels
tests/
  basic.rs            22 unit + integration tests
  bench.rs            10-section performance benchmarks
  bench_vs_pt.rs      Head-to-head PyTorch comparison
examples/
  simple_msa.rs       Minimal usage example
```


## Performance (RTX 3090, CUDA, burn 0.22)

| Config | Tensor path | Fused | Speedup |
|--------|-------------|-------|---------|
| b=4, hq=32, hkv=8, S=2048, d=64, topk=16, bs=32 | 1.89 s | **24.7 ms** | **76×** |

The fused kernel touches only the top-K selected blocks'
`O(S·topk·block_size·d)` instead of the tensor path's masked `O(S²)` attention.

## Training

The fused sparse attention runs as a single tracked node under
`Autodiff<Cuda>`. Its backward is a fused kernel that mirrors the forward
(same grid and score/softmax recompute) and applies the attention-backward
formulas: dq per query, dk/dv accumulated into the selected slots with cubecl
`Atomic::fetch_add`. Verified fused backward == tensor-path backward
(dq/dk/dv < 2e-3). `block_indices` are non-differentiable; the block-attention
output `ba` is returned untracked (secondary KL signal).

## Inference

Forward-only builds use the bare CUDA fused sparse attention directly. The
index branch (`TopKSelector`) picks the top-K blocks per query; the attention
kernel then reads only those blocks.

## License

MIT. See [LICENSE](LICENSE).
