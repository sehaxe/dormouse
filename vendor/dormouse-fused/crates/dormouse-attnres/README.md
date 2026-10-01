# dormouse-attnres - Attention Residuals for Burn

> **IN the dormouse build, as a feature.** Owner decision 2026-09-30 18:45:
> this crate was a deletion candidate and is now the mechanism behind
> `use_attnres` in `dormouse-core` (`crates/dormouse-core/src/loop_block.rs`),
> which replaces the loop's ReZero residual accumulation with a softmax mixture
> over `{token embedding, block-body outputs so far}` — Eq. 1/3/4. **Off by
> default** (checkpoint compatibility, the `use_gr` pattern), one learned
> pseudo-query per iteration slot, and **Full mode only**: `BlockAttnRes` is
> still wrong against Eq. 6 and is deliberately not wired.
>
> **No author code exists** — `MoonshotAI/Attention-Residuals` is 6 files and
> has never shipped source — so everything here is a tier-(b) transcription
> against the PDF, not a verification. The equation-by-equation record, the
> `1/sqrt(d)` decision and its measurement, and the two derivative defects this
> wiring exposed are in
> `docs/research/2026-09-30-attnres-integration.md`. The A/B against ReZero
> is queue row 7 in `docs/protocols/AB-PROTOCOL.md` and has **not** been run.
>
> The fate table entry that said "reference port" is stale on this point:
> `docs/architecture/library-crate-fate.md`.

> CI: this badge's workflow was deleted (`4963c3a`) — the gate is `../../.github/workflows/fused-library.yml`, and it has no GPU job (for CUDA: `tools/gpu-gate.sh`).
[![Crates.io](https://img.shields.io/crates/v/burn-attnres)](https://crates.io/crates/burn-attnres)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)

Learned depth-wise attention over layer outputs. Drop-in replacement for
fixed residual accumulation. Mitigates PreNorm dilution: output magnitudes
stay bounded, gradients distribute uniformly.

> Paper: [Attention Residuals](https://arxiv.org/abs/2603.15031) (Moonshot/Kimi, 2026).
> Key results: GPQA +7.5, HumanEval +3.1, MMLU +1.1 on Kimi Linear 48B.

## Install

```bash
cargo add dormouse-attnres
```

## Quick start

```rust
use dormouse_attnres::{AttnRes, BlockAttnRes, depth_attend};

// Full: query attends over all previous hidden states
let attn = AttnRes::new(512, &device);
let out = attn.forward(&history); // [h0, h1, ..., hN] -> [B,T,D]

// Block: groups into blocks for efficiency at scale
let attn = BlockAttnRes::new(512, 8, &device);
let out = attn.forward(&history);

// Raw depth attention (no parameters)
let out = depth_attend(&history, query);
```

## API

| Export | What |
|--------|------|
| `AttnRes` | Full: attend over ALL previous states |
| `BlockAttnRes` | Block: attend over block summaries |
| `depth_attend` | Core: softmax over depth via learned query |


## Performance (RTX 3090, CUDA, burn 0.22)

The Full AttnRes hot path (`depth_attend`) is 8 tensor passes over
`[L, B, T, D]` on the naive path. The fused CUDA path processes the history in
chunks of G=8 layers (online softmax, running state aliased to the output), so
peak memory is `(G+1)·B·T·D` — ~2× less than the `(L+1)·B·T·D` stack — with
exact full-depth weights.

| Op | Config | Tensor path | Fused | Speedup |
|----|--------|-------------|-------|---------|
| forward | L=24, b=1, t=2048, d=4096 | 80.3 ms | **7.6 ms** | **10.6×** |
| forward | L=8, b=2, t=2048, d=5120 | 14.7 ms | **6.9 ms** | **2.1×** |
| backward | L=24, b=1, t=2048, d=4096 | 129.4 ms | **46.7 ms** | **2.8×** |

## Training

The fused `depth_attend` runs as a single tracked node under `Autodiff<Cuda>`
with a fused backward (per (b,t) cube: RMS scores over L via tree reductions,
softmax, then d_h_l and d_q in one launch; the history is stacked with a fast
copy kernel). The fused backward is checked against the tensor-path backward
(dh/dq < 1e-2) — that is a **self-consistency check between two of our own
formulations of the same derivative**, so under `docs/adr/0020-oracle-discipline.md`
it is kind (d), *not* verification: **no external reference exists**, and the
word "verified" is withdrawn from this line. The check is real and worth having;
what it cannot do is catch a transcription error shared by both sides.

Also note the fused seam was `NoCheckpointing`-only until `fdf9b20`, so on the
trainer's backend (`Autodiff<Cuda, BalancedCheckpointing>`) this node fell back
to the tensor path until that commit. Any number in this file measured before it
is not a fused-kernel number.
Parent count is capped at 64 history layers + query (const-generic op).

## Inference

Forward-only builds use the bare CUDA fused chunked `depth_attend` directly
(no graph overhead). The streaming `BlockAttnRes` path collapses ~15 launches
per layer output to ~4 (`source_score` norm+dot and the online-softmax merge,
each one launch).

## License

MIT. See [LICENSE](LICENSE).

