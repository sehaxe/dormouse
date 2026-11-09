# TSCT, ternary only — the specification of record

**Date:** 2026-11-09. **Status:** canonical spec. It supersedes the
`fp8 → NVFP4 → ternary` ladder in `docs/reviews/littlebit-design-2026-10-15.md`
as the plan for TSCT: the owner's directive is that **TSCT is ternary, full
stop** — `sign`/binary, fp8 and NVFP4 are not rungs of this design and are not
built. Everything below is either a measurement with a source, arithmetic shown
here, or an explicitly marked absence.

The formal core of the maths is being proved in Lean 4 under `proof/` (see §7);
this file states what is being proved and what is deliberately not.

---

## 1. The object

A weight matrix is replaced by **r ternary rank-1 terms, one scale each**:

    W  ≈  Σ_{k<r}  s_k · (u_k ⊗ v_k),    u_k ∈ {−1,0,+1}^d,  v_k ∈ {−1,0,+1}^f,  s_k ∈ ℝ

Storage per term: `d + f` ternary elements (log₂3 = 1.585 bit each) + one FP16
scale. Training keeps FP32 masters of the latents and applies the ternary
projection through a straight-through estimator (`tsct.py`, `factor_qat` rungs
`ternary`, `binary`, `mxfp4`, `ternary_s3t`; the last is the Bernoulli S3T
sampler).

**The forward never materialises `d·f`.** `tsct.py:681` computes
`y = ((x @ u) * s) @ v` in the factor dtype with FP32 accumulate. That is the
whole compute story: `d·r + r·f` MAC instead of `d·f`.

## 2. Storage, exactly (b1 expert: d=2048, r=64, f=5461)

| what | elements | bits | bytes |
|---|---:|---:|---:|
| U ternary | 2048·64 = 131 072 | 1.585 (packed) / 2 (2-bit) | 25 964 / 32 768 |
| V ternary | 64·5461 = 349 504 | 1.585 / 2 | 69 247 / 87 376 |
| scales (per-row, per-col, per-rank, FP16) | 2048+64+64+5461+64 = 7 701 | 16 | 15 402 |
| **total, ternary** | | | **≈ 110.6 KB (8.7×) / ≈ 135.5 KB (7.1×)** |
| *for contrast:* sign-only factors | 480 576 | 1 | 75.5 KB (12.8×) |
| *for contrast:* bf16 factors (today's master) | 480 576 | 16 | 964.0 KB (1×) |

**The number to quote is 8.7× at exact 1.585-bit packing, 7.1× with plain 2-bit
packing.** The 12.8× figure in the LittleBit design is the *binary* (1-bit)
variant and does not apply to this design. Ternary costs 45 % more bytes than
binary; it buys a zero state, which is the compressibility and sparsity story.

## 3. What is measured (and what each number cost)

| claim | number | conditions | source |
|---|---|---|---|
| TSCT is BPB-neutral at equal parameters | **+0.0042** BPB | 95M, 120 MB, b24, lr 3e-4, muon×12, seed 1: 1.5067 vs 1.5052 | `docs/reviews/2026-11-09-architecture-matrix.md` |
| Ternary from step 0 through a hard STE costs quality | **+0.050** BPB (ternary), +0.087 (binary) | medium preset, vs bf16 factors | `tsct.py:47`, `docs/reviews/tsct2-design-2026-10-04.md` |
| The model's matrices are **not** low-rank in a useful sense | rank 64 holds **43.4 %** of `down`'s energy; 99 % energy needs **941 of 1024** singular values | trained `medium` checkpoint, CPU SVD | `docs/research/2026-11-07-tsct-ceiling.md` §1 |
| A factored bank does not beat dense on BPB at 95M | −0.067 in dense's favour at equal active MAC (600 steps, 3 seeds) | bank E=512 vs dense 69.3M | same, §0 |
| Factored FFN shrinks the block 5.0× in params, 1.72× in MAC | 753 664 vs 3 769 664 params; 4.21 vs 7.23 MMAC/byte | d=832, f=3072, r=64 (live shape) | `dormouse info configs/flagship3_dense.toml` |
| Ternary factors fit NVFP4 e2m1 exactly | — | so a 4-bit hardware path can carry them, if one existed | `docs/research/2026-10-15-tsct-super-kernel.md` |
| **End-to-end decode speed with TSCT** | **UNMEASURED** | arithmetic says 1.6× fewer bytes per byte-step (95.0M → 58.4M) | absence, not a zero |

## 4. What "ideal" means — the objective, stated so it can fail

TSCT is not judged by BPB alone. Its contract:

> At a fixed **quality budget** (≤ +0.02 BPB versus the bf16 dense control),
> a TSCT model must occupy ≤ 1/7 of the bytes and decode ≥ 1.5× faster.

Both halves must hold. A TSCT that is 7× smaller but costs +0.05 BPB has not
met the contract; neither has one that is free in quality but does not decode
faster (that is a disk-format win, not an inference win).

Corollary that follows from §3 and must not be argued away: **because the
matrices are not low-rank, ternary-TSCT is a compression mechanism, not a
free substitution.** The honest framing is "buy 8.7× fewer bytes for ≤ 0.02 BPB",
which is exactly the owner's reinvestment rule: the freed bytes get spent on
width or on more experts, and the *reinvested* model is what must be measured.

## 5. The four gates (replaces G-A..G-D of the LittleBit design)

| gate | statement | cost | kill condition |
|---|---|---|---|
| **T-A storage** | packed ternary + scales ≤ 1/7 of bf16 for the same shapes; round-trip decode of the packed form reproduces the factors exactly | CPU, seconds | any pack/unpack mismatch |
| **T-B quality at fixed size** | ternary arm within **+0.02 BPB** of bf16 factors, 3 seeds, 120 MB | 3 × 25 min | tie or loss = the arm is deleted (ADR-0002) |
| **T-C QAT-finish** | annealing the ternary projection over the last 20 % of a run recovers ≥ 60 % of the +0.050 BPB STE penalty | 2 × 2 GB runs (≈ 14 h) | if it does not, ternary ships only as an export format for a bf16-trained model, never as the training dtype |
| **T-D inference** | `bench_decode.py` on two checkpoints (bank vs ternary-TSCT): bytes/s at 512/1024/2048 context, chunked and recurrent | 1 h | < 1.3× speed ⇒ the win is disk-only and must be reported as such |

Gates run in order. T-B before T-C because annealing cannot rescue an arm that
loses at fixed size.

## 6. What the ideal implementation looks like (the target design)

1. **Masters stay FP32** latents; the ternary projection is a *view* used in the
   forward. Muon updates the latents (this is what makes STE training stable).
2. **Per-rank scales are optimal in closed form** (§7, theorem 1), so the only
   learned scale is the one that cannot be derived: per-row/per-col absorb the
   fan-in asymmetry that a single `s_k` cannot.
3. **One residual term** (rank 32, same scheme) if T-B shows the primary term
   cannot reach +0.02 — the LittleBit paper's own remedy, ~+41 KB per expert.
4. **Packed at rest, dequantised per step** for compute: no ternary MMA exists on
   sm_120 (tcgen05 is B200-only; `docs/research/2026-10-15-tsct-super-kernel.md`),
   so the kernel story is "fewer bytes through HBM", not "faster MACs".
5. **Export = the packed form + the decoder**, checked by re-loading into a fresh
   process and comparing logits to the training-time forward (the same gate the
   repo already uses for checkpoints).

## 7. The Lean core (`proof/`)

Proved in Lean 4 (core + Std, no Mathlib), over `Rat`/`LinearOrderedField`:

1. **Optimal scale.** For fixed ternary `(u,v)` the error-minimising scale is
   `s* = ⟨W, u⊗v⟩ / ‖u⊗v‖²`, and the residual is `‖W‖² − ⟨W,u⊗v⟩²/‖u⊗v‖²`.
2. **Per-rank scales dominate one shared scale** (the reason `diag(s)` exists).
3. **Term stacking cannot hurt** (triangle inequality on the Frobenius norm) —
   this is what licenses the residual branch in §6.3.
4. **Capacity.** The number of distinct ternary rank-1 matrices is `3^(d+f−1)`
   — the honest limit on "how many function shapes one bank can store", and the
   arithmetic that stops the "1.06B stored = 23B dense" claim from being read as
   an information-capacity claim.

**What Lean does not prove, and must not be reported as proved:** that a
ternary-trained model reaches a given BPB; that the STE gradient is unbiased
(it is not, and its bias is the +0.050 BPB of §3); that the kernels are correct.
Lean proves the algebra of the projection, nothing about the weights it is
applied to.

## 8. Open questions, with the experiment that closes each

| question | experiment |
|---|---|
| Does annealing recover the +0.050? | T-C |
| Is ternary better than binary at equal bytes? | T-B with both arms, iso-bits |
| Does TSCT decode faster end to end? | T-D |
| Does the freed 39 % of bytes buy back more than 0.02 BPB when reinvested in width? | 95M dense vs (TSCT at ~95M bytes but wider), 3 seeds, 120 MB |
| Can a residual term close the gap at no extra MAC? | rank-64 primary + rank-32 residual vs rank-96 primary at iso-bytes |
