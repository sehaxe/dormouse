# dormouse-rope - Rotary Position Embedding for Burn

> **Not in the dormouse build.** No crate under `crates/dormouse-{core,data,train,cli}/`
> depends on this one. It is a `[dev-dependencies]` entry of `dormouse-spectral` (one
> example, `tsct_diag.rs`) and a `path =` dependency of the library's own
> `benches/cpu_probe`, so the library's CI builds it and nothing that trains does.
> **Recommendation: WIRE** — see `docs/architecture/library-crate-fate.md`. The model has no
> positional encoding at all today, and `AGENTS.md:586` says the attention arm
> keeps RoPE. Left in place rather than deleted because the bench and the example
> name it.

> CI: this badge's workflow was deleted (`4963c3a`) — the gate is `../../.github/workflows/fused-library.yml`, and it has no GPU job (for CUDA: `tools/gpu-gate.sh`).
[![Crates.io](https://img.shields.io/crates/v/burn-rope)](https://crates.io/crates/burn-rope)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)

**Rotary Position Embedding** with **YaRN** context-length extrapolation for
the [Burn](https://burn.dev) deep learning framework.

> Papers: [RoFormer](https://arxiv.org/abs/2104.09864) (Su et al., 2021),
> [YaRN](https://arxiv.org/abs/2309.00071) (Peng et al., 2023).

## Install

```bash
cargo add dormouse-rope
```

## Quick start

```rust
use dormouse_rope::RotaryEmbedding;

let rope = RotaryEmbedding::new(512, 8, 4096, 10000.0, 1.0, &device);

// Apply to query and key tensors [B, T, D]
let (q_rope, k_rope) = rope.forward_qk(q, k);

// Or apply to a single tensor
let x_rope = rope.forward(x);
```

### Standalone functions

```rust
use dormouse_rope::{precompute_freqs, apply_rope_3d, apply_rope_4d};

let (cos, sin) = precompute_freqs(64, 2048, 10000.0, 8.0, &device);
let x_rope = apply_rope_3d(x, cos, sin, n_heads);
```

## API

| Export | Shape | What |
|--------|-------|------|
| `precompute_freqs` | `(cos, sin)` | Plain RoPE tables `[T, HD/2]` |
| `precompute_freqs_yarn` | `(cos, sin)` | YaRN tables: NTK-by-parts ramp + temperature (Eqs 10-15) |
| `RotaryEmbedding::yarn` | module | YaRN-configured rotary embedding |
| `apply_rope_3d` | `[B,T,D] → [B,T,D]` | RoPE to 3D tensor |
| `apply_rope_4d` | `[B,T,NH,HD] → [B,T,NH,HD]` | RoPE to pre-reshaped 4D tensor |
| `RotaryEmbedding` | Module | Precomputed module with `forward()` / `forward_qk()` |

## How it works

RoPE encodes position by rotating query and key vectors in 2D subspaces:

```
q_rot[m] = q[m] * cos(m·θ) + rotate(q[m]) * sin(m·θ)
```

With YaRN, the base frequency is scaled for longer contexts:

```
base' = base · yarn_factor^(d / (d-2))
```

Set `yarn_factor = 1.0` for standard RoPE, `> 1.0` for extended context.


## Performance (RTX 3090, CUDA, burn 0.22)

| Config | Tensor path | Fused | Speedup |
|--------|-------------|-------|---------|
| b=4, t=2048, nh=32, hd=128 | 22.4 ms | **14.7 ms** | **1.5×** |

The rotation (x1·c − x2·s, x1·s + x2·c) is a single elementwise launch per
row instead of 4 muls + 2 adds + a `[.., hd]` cat.

## Training

RoPE runs as a single tracked node under `Autodiff<Cuda>`: fused forward and
a fused elementwise backward (d_x from the checkpointed input and recomputed
cos/sin). The freqs are treated as constants (learnable freqs fall back to the
tensor path). The fused backward is checked against the tensor-path backward
(<1e-3): two of our own formulations of the same derivative compared to each
other, which is kind (d) under `docs/adr/0020-oracle-discipline.md`, not
verification. **No external reference exists** for either path.

The fused seam was `NoCheckpointing`-only until `861b9f7`, so under
`Autodiff<Cuda, BalancedCheckpointing>` — the trainer's backend — this node fell
back to the tensor path until then. The fused backward has been reachable on the
autodiff path since; it has not been re-measured since, and the performance table
above predates that.

## Inference

Forward-only builds use the bare CUDA fused kernel directly. Precomputed
cos/sin tables (`precompute_freqs` / `precompute_freqs_yarn`) are reused across
calls.

## License

MIT. See [LICENSE](LICENSE).

