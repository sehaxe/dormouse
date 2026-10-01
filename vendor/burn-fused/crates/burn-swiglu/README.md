# burn-swiglu - SiLU-Gated Linear Unit for Burn

> **Not in the dormouse build.** No crate under `crates/dormouse-{core,data,train,cli}/`
> depends on this one; the incoming edges are `benches/cpu_probe` and the `burn-fused`
> facade. The FFN it would replace is a line of `burn::activation::silu`, so adopting
> it would add a dependency to save nothing. Kept as a **reference port** because the
> probe measures its fused kernel; fate table: `docs/library-crate-fate.md`.

> CI: this badge's workflow was deleted (`4963c3a`) — the gate is `../../.github/workflows/fused-library.yml`, and it has no GPU job (for CUDA: `tools/gpu-gate.sh`).
[![Crates.io](https://img.shields.io/crates/v/burn-swiglu)](https://crates.io/crates/burn-swiglu)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)

Standard **SwiGLU** feed-forward network for the [Burn](https://burn.dev) deep
learning framework. SiLU-gated linear unit - the default FFN activation in
every major language model (LLaMA, Mistral, Qwen, DeepSeek, Gemma).

> Paper: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
> (Shazeer, 2020).

## Install

```bash
cargo add burn-swiglu
```

## Quick start

```rust
use burn_swiglu::{SwiGLU, swiglu_gate};

let device = Default::default();
let ffn = SwiGLU::<NdArray>::new(512, 2048, &device);

// [B, T, 512] → [B, T, 512]
let y = ffn.forward(x);
```

## API

| Export | What | Shape |
|--------|------|-------|
| `SwiGLU` | FFN module with `gate_up` + `down` projection | `d → 2h → h → d` |
| `swiglu_gate` | `SiLU(half_left) * half_right` | `[B, T, 2h] → [B, T, h]` |

## How it works

```
x -> gate_up(x) -> [gate | up] -> SiLU(gate) * up -> down(x) -> output
```

The `gate_up` projection produces a `[B, T, 2h]` tensor. `swiglu_gate` splits
it in half, applies `SiLU` to the left half (gate), and multiplies element-wise
by the right half (up). Result passes through a `Linear(h, d)` output projection.

## License

MIT. See [LICENSE](LICENSE).
