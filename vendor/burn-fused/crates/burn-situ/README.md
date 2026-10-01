# burn-situ - SiTU-GLU Activation for Burn

> **Wired into dormouse (2026-10-01), OFF by default.** This crate is now a
> normal `[dependencies]` entry of `crates/dormouse-core`, read by the expert
> FFN when `use_situ` is true (`crates/dormouse-core/src/loop_block.rs`, the
> `use_situ` branch of the expert loop). It was a reference port until then and
> this banner said so; fate table: `docs/architecture/library-crate-fate.md`.
>
> **Default features only — the fused CUDA kernel below is NOT reachable from
> the trainer.** It has never executed in any job in this repo
> (`vendor/burn-fused/tools/gpu-gate.sh:73-77` names burn-situ among the crates
> whose GPU tests have never run), and the burn-rmsnorm precedent is a kernel
> that compiled for its whole life and never ran. Enabling `burn-situ/cuda`
> needs a CUDA gate that compares the fused output against the tensor path on
> the form fixture first — the tensor path IS the reference form, and it costs
> ~66 extra kernel launches per step (~1% of a warm step).
>
> **Use `K3_GATE_BETA` / `K3_UP_BETA`, not 1.0.** Kimi K3 runs β₁ = 4 (gate) and
> β₂ = 25 (up) — `moonshotai/Kimi-K3@main` `config.json`
> (`activation_situ_beta`, `activation_situ_linear_beta`) and arXiv:2607.24653v2
> §2.3.2. The `situ_glu` defaults and `SituAndMul.__init__`'s own `beta=1.0`
> are the library's, not the model's.

> CI: this badge's workflow was deleted (`4963c3a`) — the gate is `../../.github/workflows/fused-library.yml`, and it has no GPU job (for CUDA: `tools/gpu-gate.sh`).
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
use burn_situ::{situ_glu, softcap, K3_GATE_BETA, K3_UP_BETA};

// K3's own values. `situ_glu(.., 1.0, 1.0)` is the library default and is NOT
// what Kimi K3 runs: beta1 = 4 on the gate, beta2 = 25 on the up branch.
let capped = softcap(activations, K3_GATE_BETA);                    // tanh soft-cap
let gated = situ_glu(gate_up, hidden, K3_GATE_BETA, K3_UP_BETA);    // SiTU-gated FFN
// `gate_up` is [N, 2*hidden]: Eq (12) reads Wg x and Wu x separately.
```

## API

| Export | What |
|--------|------|
| `softcap(x, beta)` | `beta * tanh(x / beta)` |
| `situ_glu(gu, h, bg, bu)` | Bounded gating: softcap(gate)·sigmoid(gate) · softcap(up) |
| `K3_GATE_BETA` | `4.0` — Kimi K3's gate cap. **Pass this, not `1.0`** |
| `K3_UP_BETA` | `25.0` — Kimi K3's up cap. Same |


## Performance (RTX 3090, CUDA, burn 0.22)

| Config | Tensor path | Fused | Speedup |
|--------|-------------|-------|---------|
| N=2048, H=5120 | 32.0 ms | **14.3 ms** | **2.2×** |

**These figures have no config, date or commit behind them, and the fused
column describes a path that has never executed in any job in this repo**
(`gpu-gate.sh:73-77`). Treat the table as the author's claim about a kernel
nobody here has watched run, not as a measurement of ours. The tensor path IS
gated (against Moonshot's own numbers) and is what dormouse runs.

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

