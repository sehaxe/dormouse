# burn-engram - Conditional Memory for Burn

> CI: this badge's workflow was deleted (`4963c3a`) — the gate is `../../.github/workflows/fused-library.yml`, and it has no GPU job (for CUDA: `tools/gpu-gate.sh`).
[![Crates.io](https://img.shields.io/crates/v/burn-engram)](https://crates.io/crates/burn-engram)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)

**Conditional memory** for the Burn framework. N-gram hash embedding with
multi-head gated fusion - a new sparsity axis complementary to MoE. O(1)
lookup, deterministic addressing.

> Paper: [Engram](https://arxiv.org/abs/2601.07372) (DeepSeek AI, 2026).
> 27B scale: MMLU +3.4, BBH +5.0, ARC +3.7, HumanEval +3.0, MATH +2.4.

## Install

```bash
cargo add burn-engram
```

## Quick start

```rust
use burn_engram::EngramModule;

// Single-head (HC=1): simple gated memory read
let mem = EngramModule::new(&[100_000, 100_000, 100_000], 128, 512, 1, &device);
let out = mem.forward(hashed_ids, hidden_states_4d); // [B, L, 1, D]

// Multi-head (HC=4) with short conv: full paper architecture
let mem = EngramModule::new(&[100_000, 100_000], 64, 256, 4, &device)
    .with_short_conv(4, 3, &device);
let out = mem.forward(hashed_ids, hidden_states_4d); // [B, L, 4, 256]
```

## API

| Export | What |
|--------|------|
| `MultiHashEmbedding` | N embedding tables with offset addressing |
| `EngramModule` | Full pipeline: hash lookup → N key projs → gated fusion → short conv |
| `depthwise_conv_1d` | Causal depthwise conv (paper's ShortConv) |
| `compute_gate` | sigmoid(RMSNorm(h)^T RMSNorm(k) / √d) — paper Eq 4 |

## How it works

```
hashed_ids → MultiHashEmbedding → value_proj → value[B,L,D]
                                → key_proj[N] → key[B,L,D] × N
hidden[B,L,HC,D] → RMSNorm → query[HC] ────────┘
gate[HC] = sigmoid(√|key·query/√d| × sign(key·query/√d))
out[B,L,HC,D] = gate[HC] · value + depthwise_conv(gate[HC] · value)
```

Hash computation is the user's responsibility (CPU-side n-gram hashing).
Memory read and gated fusion operate entirely on GPU with zero CPU syncs.

## License

MIT. See [LICENSE](LICENSE).
