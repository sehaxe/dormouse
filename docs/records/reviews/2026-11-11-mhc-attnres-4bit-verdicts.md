# Three 2026-11-11 verdicts: mHC, AttnRes, 4-bit activations

All three were built, gated on CPU, and then measured on the card. Two lose and
one is unrunnable. This note exists so none of them is re-litigated from memory.

## 1. mHC in the transformer block — LOST

| arm | BPB @ 4 880 steps, 120 MB, b16 |
|---|---:|
| control (gated FFN) | **1.4861** |
| mHC, K=2 (`use_mhc=true`) | **1.5714** |

**+0.0853 BPB**, which is 8–10× the single-seed spread (0.008–0.011). Cost:
+936 MiB of retained lanes (78 MiB per layer at b24×1024) and +13 323 params.

Verdict: **measured refusal at 95M.** The mechanism is not wrong in principle —
it is wrong at our depth. Its init is exact identity, so at 12 layers it starts
as a 1e-9 perturbation and must earn back the depth it was given; the loop form
already showed the same pattern (mHC's own paper works at 30+ layers).

Status: stays as a flag, off. Re-open only with a multi-seed run at ≥48 layers.

## 2. AttnRes (full N=L) — UNRUNNABLE on 16 GB

| batch | result |
|---|---|
| b24 | OOM, "Tried to allocate 858 MiB" |
| b20 | OOM |
| b16 | OOM, "Tried to allocate 48 MiB" |

Cause: the score tensor `[T, S, d]` with `S = i+1` reaches **507 MiB at layer
12** (b24×1024, bf16), plus a retained `[L, T, d]` history (468 MiB). The
mechanism costs 0.0375 % of the step's MACs and 2–3 GiB of VRAM.

**No verdict exists** — not "it loses", but "it could not be measured". The fix
is arithmetic, not tuning: score one source at a time (`S` passes of `[T, d]`),
which drops the peak from 507 MiB to 39 MiB per layer with identical math. Until
that lands, any AttnRes claim from this repo is unfounded.

## 3. 4-bit activations (NVFP4) — broke even on memory, LOST on throughput

The kernel work was real: `quantize_nvfp4` went 13.57 → 0.32 ms (43×), and the
bare fp4 `_scaled_mm` is **3.16× faster** than bf16 at our live shape
(`[24576,832]x[832,2048]`: 0.32 vs 1.01 ms), 7.5× at 4096³.

But the *arm* measures a loss:

| arm | max batch | ms/step @b24 | bytes/s | peak @b24 |
|---|---:|---:|---:|---:|
| bf16 activations | 24 (OOM at 28) | 340.5 | **72.2 kB/s** | 12.82 GiB |
| e2m1 packed | **36** | 547.3 | 67.4 kB/s | 8.98 GiB |

**−6.7 % throughput** for +50 % batch headroom and −3.8 GiB peak.

The reason is the one that kills most "we saved memory so we go faster" plans:
**bytes/s is tokens/s, and the step is linear in batch.** Freeing VRAM raises the
batch but not the per-token cost, so throughput cannot improve unless the *per
token* cost falls. The only lever that does that is the GEMM itself (0.20 vs
0.99 ms), and reaching it needs the quantization folded into the GEMM epilogue —
the pack (0.32 ms) + blocked scale (0.09) + output scale (0.29) chain currently
costs more than the GEMM saves.

Verdict: **keep the kernel, drop the training arm** until a fused epilogue
lands. The max-batch gain is real and worth having for *other* arms competing
for VRAM, at a 6.7 % throughput price.

## What all three have in common

Every one of them was param-cheap and FLOP-cheap on paper and lost on the axis
nobody prices in the paper: **memory traffic and batch linearity**. That is the
same lesson as the MoE cap measurement — and it is why the density question must
be asked in bytes/s and GB/s, never in "% of FLOPs".
