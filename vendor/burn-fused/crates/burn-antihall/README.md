# burn-antihall - Anti-Hallucination Toolkit for Burn

[![CI](https://github.com/sehaxe/burn-antihall/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/burn-antihall/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/burn-antihall)](https://crates.io/crates/burn-antihall)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)

**Hallucination suppression and detection** for the [Burn](https://burn.dev)
framework. Based on 5 papers (2025-2026).

| Paper | Capability |
|-------|-----------|
| [2512.01797](https://arxiv.org/abs/2512.01797) (Dec 2025) | <0.1% FFN neurons cause hallucinations |
| [2604.19765](https://arxiv.org/abs/2604.19765) (Apr 2026) | H-Neurons don't generalize across domains |
| [2607.00158](https://arxiv.org/abs/2607.00158) (Jul 2026) | Readable-but-not-controllable: detection works (AUROC 0.77-0.86), neuron-level suppression unreliable |
| [LLM-CAS](https://arxiv.org/abs/2512.18623) (AAAI 2026) | Context-dependent perturbation > static |
| HALL-OPT (Nature 2026, no arXiv preprint) | 94.3% detection accuracy |

## Install

```bash
cargo add burn-antihall
```

## Quick start

```rust
use burn_antihall::{HallSuppressor, HallDetector};

// Suppression - paper 1
let sup = HallSuppressor::new(2048, &device);
let x = sup.forward(ffn_out);

// Domain-aware suppression - papers 1+2
let sup = HallSuppressor::new(2048, &device)
    .with_domain_proj(64, &device);
let x = sup.forward_domain(ffn_out, domain_emb);

// Fully adaptive - papers 1-4
let sup = HallSuppressor::new(2048, &device)
    .with_domain_proj(64, &device)
    .with_context_proj(&device);
let x = sup.forward_adaptive(ffn_out, domain_emb, hidden_states);

// Detection - paper 5
let det = HallDetector::new(512, &device);
let prob = det.prob(hidden_states);  // [B, T, 1] ∈ [0,1]
```

## License

MIT. See [LICENSE](LICENSE).
