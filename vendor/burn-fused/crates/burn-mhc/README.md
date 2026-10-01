# burn-mhc - Manifold-Constrained Hyper-Connections for Burn

> **Not in the dormouse build.** No crate under `crates/dormouse-{core,data,train,cli}/`
> depends on this one; the only incoming edges are the `burn-fused` facade and the
> `burn-fused-benches` probe. Kept as a **reference port**: `docs/archive/architecture/PLAN-minimal-core.md`
> §M2 names it as a residual-stream A/B arm, and that A/B has not been run. Fate
> table and reasoning: `docs/library-crate-fate.md`.

> CI: this badge's workflow was deleted (`4963c3a`) — the gate is `../../.github/workflows/fused-library.yml`, and it has no GPU job (for CUDA: `tools/gpu-gate.sh`).
[![Crates.io](https://img.shields.io/crates/v/burn-mhc)](https://crates.io/crates/burn-mhc)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)

Manifold-Constrained Hyper-Connections (mHC) for Burn — full implementation of
[arXiv:2512.24880](https://arxiv.org/abs/2512.24880) (DeepSeek, 2025):

- **Eq 3**: n-stream residual `x_{l+1} = H_res x_l + (H_post)^T F(H_pre x_l)`
- **Eq 7**: first-order hyper-network — input-dependent mappings from
  `RMSNorm(vec(x))` linear projections + static biases, gating factors α init 0.01
- **Eq 8**: `H_pre = sigmoid`, `H_post = 2·sigmoid` (non-negativity prevents
  signal cancellation), `H_res = Sinkhorn-Knopp(...)`
- **Eq 9**: Sinkhorn-Knopp entropic projection (t_max = 20) onto the Birkhoff
  polytope — `H_res` is doubly stochastic (row/col sums = 1, spectral norm ≤ 1,
  compositional closure), restoring the identity-mapping property

At init the block reduces to the standard residual `h + Σ branches`
(`H_pre ≈ 1`, `H_post ≈ 1`, `H_res ≈ I`).

## Install

```bash
cargo add burn-mhc
```

## Quick start

```rust
use burn_mhc::MhcBlock;

let mhc = MhcBlock::new(4, d_model, &device); // n=4 streams, D = n·C
let out = mhc.forward(h, &[ffn_out]);         // h, ffn_out: [B, T, D]
```

## API

| Export | What |
|--------|------|
| `MhcBlock` | Full mHC block: hyper-net + Birkhoff-projected residual mixing |
| `sinkhorn_knopp` | Entropic projection onto the doubly stochastic manifold (Eq 9) |
| `SINKHORN_ITERS` | t_max = 20 (paper App. A.1) |
| `ALPHA_INIT` | Gating factor init 0.01 (paper App. A.1) |


## Performance (RTX 3090, CUDA, burn 0.22)

`SINKHORN_ITERS=20` Sinkhorn-Knopp is fused into one launch per (batch, time)
matrix (alternating row/column normalizations, `sync_cube()` between phases).

**The forward row of this table is retracted. It measured kernel enqueue, not
the kernel.** `sinkhorn_bench` (`src/sinkhorn_cuda.rs:253-281`) reads the clock
at line 266 with no device read inside the timed loop; the only
`into_scalar()` in that function's neighbourhood is at line 463, in the
*backward* bench. What the 1.65 µs / 2.55 µs are is the host-side cost of
queuing the launch. The tensor-path loop at lines 267-275 has the same defect,
so the 32.4 ms / 104.7 ms are enqueue too and the **19,600× / 41,000× ratio is
a ratio of CPU dispatch overhead, not a speedup**: 4.2 M elements × 40
normalization phases cannot execute in 1.65 µs on any GPU.

**mhc is not the only bench in this library with a clock read and no flush, and
that is now a library-wide item.** rope (`rope_cuda.rs:284`), bitnet
(`fwt_cuda.rs:548`), situ (`fused_situ.rs:353`) and attnres
(`fused_attnres.rs:1050`) all flush inside the timed loop. These do not:
`burn-gdn2/tests/bench_cuda.rs:34-44` and `bench_train_cuda.rs:18-24` (the
shared `time_it` helper behind every number in burn-gdn2's performance tables),
`burn-muon-plus/src/lib.rs:447` (`ns_bench`), `fused_kernels.rs:243`
(`ortho_bench`) and `:302` (`step_bench`), and this file. Every speedup those
print is a ratio of CPU dispatch overhead until a flush is added.

| Op | Config | Tensor path | Fused | Speedup |
|----|--------|-------------|-------|---------|
| forward | [8, 2048, 16] | ~~32.4 ms~~ enqueue only | ~~1.65 µs~~ enqueue only | **RETRACTED** |
| forward | [4, 1024, 64] | ~~104.7 ms~~ enqueue only | ~~2.55 µs~~ enqueue only | **RETRACTED** |
| backward | [8, 2048, 16] | 67.1 ms | **17.7 ms** | **4×** |

**A real forward number is recoverable and has not been measured.** The fix is
one line in the timed loop — `let _: f32 = k.clone().sum().into_scalar();` after
each `sinkhorn_cuda` call, mirroring `bench_fused_bwd.rs` in burn-gdn2 — and
the same flush inside the tensor loop, after which the ratio is meaningful. It
is a `src/` change and is not made here; until someone makes it, this crate has
no measured forward speedup.

The backward row is the only kernel-time measurement in the crate: it flushes
at `sinkhorn_cuda.rs:463` after the loop and amortizes over 20 calls, so
17.7 ms includes the kernels' execution. It still carries no date or commit
(ADR-0020 rule 1) and predates the retractions above.

Numerics: the fused forward matches the tensor path and is doubly stochastic
to <1e-2 (`sinkhorn_cuda.rs:224-249`); the fused backward matches the
tensor-path backward to <1e-3 (`sinkhorn_cuda.rs:480-503`) and the analytic
gradient matches central finite differences to 2e-2
(`sinkhorn_cuda.rs:520-573`). **No external reference exists** — all three
compare this crate against its own tensor path, which is a self-consistency
check, not a verification against anything published (ADR-0020).

## Training

The fused Sinkhorn runs as a single tracked node under `Autodiff<Cuda>`; the
backward is a fused kernel that recomputes the forward trajectory's per-step
row/col sums in shared memory and reverses the 2·iters normalizations
(`d_m_pre = d/s − m_pre·Σd/s²`). The fused backward matches the tensor-path
backward to <1e-3 (`sinkhorn_cuda.rs:480-503`); that is a self-comparison, and
**no external reference exists**. Training and inference both use the fused
kernels: the node strips to bare and gates on `TypeId` of the *inner* backend
(`sinkhorn_cuda.rs:349-360`), the pattern burn-gdn2's adjoint only learned at
`277b442`. That gate is on the **bare** `CubeBackend`, so under burn's default
`Cuda` (which is `Fusion<CubeBackend>`-wrapped) the fused node does not engage
and `sinkhorn_knopp` falls back to the tensor loop — **silently**, with no
counter (a COUNTED-or-LOUD item, ADR-0011, still open in this crate).

## Inference

Forward-only builds use the bare CUDA fused Sinkhorn directly. **This crate has
no measured forward speedup** — see the retraction above; the "2 µs per (b, t)
matrix against a 32-105 ms tensor path" sentence that used to stand here was
enqueue time on both sides.

## License

MIT. See [LICENSE](LICENSE).

