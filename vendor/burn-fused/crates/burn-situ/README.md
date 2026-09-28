# burn-situ - SiTU-GLU Activation for Burn

> **Not in the dormouse build.** No crate under `crates/dormouse-{core,data,train,cli}/`
> depends on this one; the incoming edges are `[dev-dependencies]` of `burn-spectral`,
> `benches/cpu_probe` and `burn-fused-benches`. dormouse's FFN uses burn's own
> `activation::silu` (`crates/dormouse-core/src/loop_block.rs:467`), not this gated
> variant. Kept as a **reference port**; fate table: `docs/library-crate-fate.md`.

[![CI](https://github.com/sehaxe/burn-situ/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/burn-situ/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/burn-situ)](https://crates.io/crates/burn-situ)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)

Sigmoid-Tanh Unit (SiTU) gated linear activation - bounded GLU variant from
Kimi K3. Applies `softcap(x, beta) = beta * tanh(x / beta)` to gate and up
branches, preventing activation explosion in deep models.

> Paper: [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653)
> (Moonshot, 2026). Bounded GLU for MXFP4 quantization stability.

## Install

```bash
cargo add burn-situ
```

## Quick start

```rust
use burn_situ::{softcap, situ_glu};

let capped = softcap(activations, 1.0);         // tanh soft-cap
let gated = situ_glu(gate_up, hidden, 1.0, 1.0); // SiTU-gated FFN
```

## API

| Export | What |
|--------|------|
| `softcap(x, beta)` | `beta * tanh(x / beta)` |
| `situ_glu(gu, h, bg, bu)` | Bounded gating: softcap(gate)·sigmoid(gate) · softcap(up) |


## Performance (RTX 3090, CUDA, burn 0.22)

| Config | Tensor path | Fused | Speedup |
|--------|-------------|-------|---------|
| N=2048, H=5120 | 32.0 ms | **14.3 ms** | **2.2×** |

The whole SiTU-GLU equation (two softcaps + Swish gate factor) is one
elementwise launch instead of ~7 tensor passes. Fused dispatch requires
`H % 8 == 0` (cubecl CUDA codegen corrupts coalesced stores on row strides
that are not 32-byte multiples); other shapes fall back to the tensor path.

## Training

Under an autodiff backend (`Autodiff<Cuda>`) SiTU-GLU registers as a single
tracked node: the fused forward plus an exact elementwise backward (recomputed
from the checkpointed input). The fused backward is checked against the
tensor-path backward on CUDA (<1e-3) — a comparison of two of our own
formulations of the same derivative, so kind (d) under
`docs/adr/0020-oracle-discipline.md`: **no external reference exists**, and
"verified" is withdrawn from that claim.

The fused seam was `NoCheckpointing`-only until `87c2e02`, so under
`Autodiff<Cuda, BalancedCheckpointing>` — the trainer's backend — this node fell
back to the tensor path until then.

## Inference

Forward-only builds use the bare CUDA fused kernel directly (no autodiff
graph overhead). `softcap` shares the same elementwise path.

## License

MIT. See [LICENSE](LICENSE).

