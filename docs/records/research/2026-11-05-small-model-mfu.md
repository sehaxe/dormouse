# Small-model MFU at b8×s1024 on one RTX 5070 Ti

**2026-11-05** · research memo, no code changes · target shape: 199.3M byte-level
looped MoE, `d=1536`, 3 iterations, 512 experts top-2, rank 64, byte vocab 256,
`b8×s1024` (8 192 bytes/step), RTX 5070 Ti 16 GB.

Evidence marks: **[M]** measured (source named), **[D]** derived arithmetic from
measured constants, **[S]** speculation/inference, explicitly flagged. A claim
without a mark is either a definition or a direct quote from the source beside it.

---

## TL;DR (confidence in brackets)

1. **A 124M–523M dense transformer on one consumer card reaches 35–47 % MFU at
   16k tokens/step, and ~36 % at 524k tokens/step; at 8k tokens the published
   number is absent and the projection is 20–30 %.** The best clean bf16
   measurement is this repo's own 523M spike: 35.7 % eager, 46.6 %
   `torch.compile`, 99.9 % GPU utilisation at `b16×1024` **[M]**. [S1 §4]
   [medium]

2. **The linear-attention kernel is a different regime: FLA's `chunk_kda` runs
   at 1.8–3.6 % of tensor peak on a GB200, and the best hand-written KDA
   forward (FlashKDA) is 4.1 %.** FlashKDA is 2.3× faster but is forward-only,
   D=128, SM90/100 — it cannot train our D=64/sm_120 model **[M]**. [S11][S25]
   [high]

3. **The arithmetic floor is not 3 %.** With the project's own
   `width.macs_per_token` ledger (latent rank 256, f=4096) the 100 %-MFU floor
   is **300–326 kB/s**; with the attention-only ledger (KDA projections +
   `out_proj`, which is what the 0.77 TFLOP/s figure matches) it is
   **1.17 MB/s**. At 22 kB/s we are at **1.9–7.3 % MFU**, i.e. 14–53× below the
   arithmetic floor, not 1.5× **[D]**. [S6] [high on arithmetic, low on which
   ledger is real]

4. **The 1.7 % is therefore real but not a GEMM-shape story.** The measured
   parts of one iteration sum to ~25 ms of a 106–124 ms iteration: dense
   projections 0.30 ms, MoE 2.5 ms, norm/mHC chain 1.67 ms, KDA fwd+bwd
   27.9 ms at the b1 shape **[M]**. [S2][S5] The missing ~80 ms is the actual
   problem, and no kernel profile of the m400 step exists in the tree. [high on
   the gap, low on its cause]

5. **`N=256` is not the cliff; `N=64` and `K=64` are.** On the same sm_120
   class, `M=16384` bf16 cuBLAS measured 48.96 TFLOP/s at `N=2048`, 39.90 at
   `N=256` (91 % of the big shape), 14.02 at `K=64` (32 %), and 10.04 at
   `N=64` (23 %) **[M]**. [S1 §2] NVIDIA's tile/wave-quantisation guide
   explains why: 64×64 is the least efficient tile, and N=256 at M=8192 is
   still 1.8 waves on a 70-SM card, not a half-empty one **[M]**. [S9][S10]
   [high]

6. **60 kB/s is not supported by any single documented lever.** It requires
   2.7× on a step whose measured GEMM/MoE parts are already tiny. The credible
   stack-up is 1.3× (depth 3→2) × 1.3× (fuse the non-KDA path) × 1.2×
   (optimizer/fusion) × 1.1× (occupancy) ≈ 2.4×, i.e. **~50–55 kB/s**, and only
   if those savings are on disjoint paths. 60 kB/s needs the profile to find a
   single fixable ~80 ms path, or a 3× training-capable KDA kernel that the
   literature does not currently have. [S] [low for 60k, medium for 35–45k]

---

## 0. The measurement, and the three ledgers that do not agree

### 0.1 What is actually measured

| quantity | value | source |
|---|---:|---|
| parameters | 199,275,551 | `m400` run banner, [S7] |
| geometry | d 1536 · 3 iterations · 512 experts top-2 · rank 64 · byte vocab 256 | `port/configs/m400.toml`, [S7] |
| batch × seq | 8 × 1024 = 8 192 bytes/step | run command, [S7] |
| throughput | 22.0–23.9 kB/s (older), 26.8 kB/s (newer `m400b` slices) | `plan_restart.py` `MEASURED_BPS=22_000`; `~/logs/m400.jsonl` |
| step time at 22 kB/s | 372 ms/step, 124 ms/iteration | 8192 / 22 000 |
| step time at 25.6 kB/s | 320 ms/step, 106.7 ms/iteration | `part_bench.py` / `.bulba/goal.md` |
| VRAM | 14.2 GiB of 16 at b8×s1024 | brief |
| claimed MFU | 1.7 % = 0.77 TFLOP/s of 45.7 | brief |

The brief's 320 ms and 125 ms are not the same measurement: 8 192 / 320 ms =
25.6 kB/s, while 8 192 / 372 ms = 22.0 kB/s. Both are on record. Use 372 ms for
the 22 kB/s claim and 320 ms for the 25.6 kB/s claim; do not mix them.

### 0.2 Three different MAC ledgers are in circulation

| ledger | what it counts | MAC/token | train MFLOP/byte (×6) | 100 %-MFU floor at 45.7 TF |
|---|---|---:|---:|---:|
| **A — attention-only** | KDA projections + `out_proj`, latent rank 256, MHA (kvh=0) | 6.49 M | 38.9 | **1.175 MB/s** |
| **A′ — attention-only, GQA 4** | same, `n_kv_heads=4` | 4.52 M | 27.1 | 1.686 MB/s |
| **B — full `macs_per_token`** | `iters·(attn + out_proj + d·f)` with f=4096 | 25.36 M | 152.2 | **300 kB/s** |
| **B′ — full, GQA 4** | same, `n_kv_heads=4` | 23.40 M | 140.4 | 326 kB/s |
| **C — MoE dense-down ledger** | MoE only: 114.9 GMAC/step = 14.0 M MAC/token | 14.0 M | 84.1 | 543 kB/s |

Sources: A/B from `port/src/dormouse/width.py::macs_per_token` **[M]** [S6];
C from `docs/records/reviews/2026-11-04-balance-fix-verdict.md` **[M]** [S4]. The
`0.77 TFLOP/s` at 22 kB/s implies **35 MFLOP/byte**, which sits between A′ and
A and is 4.3× below B. **So the 1.7 % is not the model's MFU under the full
ledger; it is the attention-only ledger's MFU.** Under B it would be 6.8 %; under
B with the real bf16 tensor peak (see §0.3) it would be 3.8 %.

This is not nit-picking. The three ledgers disagree by up to 3.9×, which is
larger than every speedup this memo discusses. **Any MFU number must name its
ledger and its peak.** The first deliverable of the profiling task is to
reconcile ledger A with the profiler's actual kernel FLOPs.

### 0.3 The denominator is also suspect

The RTX 5070 Ti spec sheet (Wikipedia's RTX 50-series table, sourced to NVIDIA's
Blackwell whitepaper) gives **[M]** [S10]:

- 8 960 CUDA cores, **70 SMs**, 2.45 GHz boost, **43.9 TFLOP/s FP32**;
- **177.4 TFLOP/s dense tensor** FP16-with-FP16-accumulate (354.8 sparse);
- the FP16-with-**FP32**-accumulate rate is half that, **≈88.7 TFLOP/s** (the
  table's own footnote defines the 177.4 column as FP8/FP32-accum or
  FP16/FP16-accum) **[D]**.

The brief's **45.7 TFLOP/s** is almost exactly the **RTX 5060 Ti's measured
bf16 ceiling** in this repo's purchase report ("Замерено у нас только 5060 Ti:
45.7 TFLOP/s bf16 при 180 Вт"), not the 5070 Ti's tensor peak **[M]**. If the
5070 Ti run uses bf16 with FP32 accumulate, its true peak is ~88.7 TF and the
0.77 TF is **0.87 % MFU**, not 1.7 %. If it uses FP32 CUDA cores, 45.7 is the
right denominator and 1.7 % stands. **The dtype/instruction mix is unverified;
this is the single biggest uncertainty in the MFU number.** [S]

A quick check settles it: run one cuBLAS bf16 GEMM at
`M=8192, N=1536, K=1536` on the 5070 Ti and compare to 88.7 TF. If it reaches
80+ TF, the 45.7 denominator is wrong by ~2×; if it reaches ~45 TF, the card is
leaving half its tensor cores unused for another reason.

---

## 1. What MFU comparable small models actually achieve

### 1.1 The data points

| model / source | params | tokens/step | hardware | achieved | MFU | evidence |
|---|---:|---:|---|---:|---:|---|
| **dense transformer, this repo's spike** [S1 §4] | 523 M | 16 384 (b16×1024) | RTX 5060 Ti, bf16 | 45 861 tok/s, 15.7 TF eager; 59 901 tok/s, 20.5 TF compiled | **35.7 % eager / 46.6 % compiled** of 44 TF | **[M]** |
| **nanoGPT 124M, 1×4090** [S18] | 124 M | 524 288 (16×32k) | RTX 4090, FP8 blocks | ~160 000 tok/s peak | ~36 % of bf16 peak; ~18 % of FP8 peak | **[M]** throughput, **[D]** MFU |
| **modded-nanogpt record #92** [S17] | 124 M | ~131 k (stage 1) | 8×H100, FP8 | 39.9 s wall, 1 194 steps | not reported; FP8 makes a bf16-peak ratio meaningless | **[M]** time, no MFU |
| **FLA `chunk_kda`, GB200** [S11] | kernel | T=8192, H=96, D=128 | GB200 | 2.327 ms fwd | **1.8–3.6 %** (40.1 TF of 2 250) | **[M]** time, **[D]** MFU |
| **FlashKDA, GB200** [S11] | kernel | same | GB200 | 1.009 ms fwd (2.31×) | **4.1 %** (92.4 TF) | **[M]** time, **[D]** MFU |
| **MoE padded bmm, b1/E1024** [S1 §3] | 492 M experts | 16 384 | RTX 5060 Ti, bf16 | 52.6 ms fwd+bwd, 3.60 padded TF | **8.2 %** of 44 TF | **[M]** |
| **small-N GEMM, N=64** [S1 §2] | — | M=16 384 | RTX 5060 Ti, bf16 | 10.04 TF | **23 %** | **[M]** |
| **short-K GEMM, K=64** [S1 §2] | — | M=16 384 | RTX 5060 Ti, bf16 | 14.02 TF | **32 %** | **[M]** |
| **N=256 GEMM** [S1 §2] | — | M=16 384 | RTX 5060 Ti, bf16 | 39.90 TF | **91 %** | **[M]** |
| **N=2048 GEMM** [S1 §2] | — | M=16 384 | RTX 5060 Ti, bf16 | 48.96 TF | **111 %** of the 44 TF reference | **[M]** |

The 111 % is not a violation: the 44 TF reference is a measured *training-step*
ceiling, not the card's absolute GEMM peak; cuBLAS's best shape beats it.

### 1.2 What the table says

- **A dense transformer at 16k tokens can hit 35–47 % MFU on one consumer
  card.** This is the realistic ceiling for GEMM-shaped work at our token
  count. At 8k tokens the ceiling is lower because M halves and the wave/tile
  quantisation is coarser, but M=8192 is still 64 M-tiles at 128 — enough to
  fill 70 SMs. The projection is **20–30 % MFU** for a dense bf16 transformer
  at b8×s1024. **[D]**
- **A KDA kernel is at 2–4 % MFU even on a GB200.** This is not a consumer-card
  artifact; it is the algorithm's tile shape (64×64 chunks, sequential state).
  FlashKDA's 2.3× is the best hand-written result and it is still 4.1 % of peak.
  The KDA paper's own efficiency claims are **decode/long-context** (6× decoding
  throughput at 1M; 2.3× at 1M), not training MFU **[M]** [S13]. Mamba-2's
  headline 2–8× is against Mamba's fused scan, likewise not MFU **[M]** [S14].
- **A grouped MoE at small rows/expert is at ~8 % MFU.** It is not the place to
  look for a 2.7×.
- **The literature has almost nothing at 8k-token batches.** The speedruns use
  131k–524k tokens/step; papers report tokens/s or loss, rarely MFU. The closest
  clean bf16 measurement is the repo's own 523M spike at 16k tokens. **This is a
  literature gap, not a solved question.** [S]

**What I searched and what each source actually reports.** nanoGPT/modded-nanogpt
report wall-clock and tokens/s, not MFU (the 4090 fork is the only one whose
throughput I could convert); Mamba-2 and FLA report kernel ms or tokens/s, not
model MFU; llama.cpp reports inference tokens/s, not training MFU; the CUDA/
cuBLAS material reports GEMM TFLOP/s, not model MFU. The only explicit MFU
numbers I found at 100–500M scale are the repo's own 523M spike and a
third-party nanoGPT-on-A100 blog (37.8 % MFU at 200k tok/s), which is
anecdotal. **Do not treat a number from a speedrun as MFU unless it states
one.** [S]

### 1.3 The realistic ceiling for *this* model

If the step were all dense GEMM at N≥1536, 80–100 % MFU is available (325 kB/s
× 0.8 = 260 kB/s). If the step is 60 % KDA at 3 % and 40 % GEMM at 30 %, the
weighted MFU is `1/(0.6/0.03 + 0.4/0.30) = 4.7 %` → **~15 kB/s** under the full
ledger, or **~55 kB/s** under the attention-only ledger. The same arithmetic
gives two very different answers because the ledger is unresolved. The honest
ceiling statement is a range: **5–10 % MFU for a KDA/MoE mix at this shape,
20–35 % for a dense transformer.** 60 kB/s sits at **5.1 % MFU (A) or 18.5 %
(B)**; the former is plausible for this mix, the latter is at the top of the
dense-transformer range. [D]

---

## 2. Is 1.7 % normal, or is something structurally wrong?

### 2.1 The arithmetic floor

Using the project's own formula `train FLOPs = 3 × 2 × macs_per_token` **[M]**
[S6], and the two ledgers:

| ledger | train MFLOP/byte | floor at 45.7 TF | floor at 88.7 TF | MFU at 22 kB/s (45.7) | MFU at 60 kB/s (45.7) |
|---|---:|---:|---:|---:|---:|
| **A** attention-only (kvh=0) | 38.9 | **1.175 MB/s** | 2.28 MB/s | **1.87 %** | **5.11 %** |
| **A′** attention-only (kvh=4) | 27.1 | 1.686 MB/s | 3.27 MB/s | 1.30 % | 3.55 % |
| **B** full `macs_per_token` (kvh=0) | 152.2 | **300 kB/s** | 583 kB/s | **7.33 %** | **20.0 %** |
| **B′** full (kvh=4) | 140.4 | 326 kB/s | 632 kB/s | 6.76 % | 18.5 % |

**Worked example (the formula the brief asked for).** For ledger A:
`8192 bytes/step × 6.49 M MAC/byte = 53.2 GMAC/step`; `× 2 = 106.4 GFLOP`
forward; `× 3 = 319 GFLOP` train; at 45.7 TF the arithmetic floor is
**6.98 ms/step**, i.e. `8192 / 0.00698 = 1.17 MB/s`. For ledger B:
`8192 × 25.36 M = 207.8 GMAC/step`, `× 2 × 3 = 1.247 TFLOP`, floor
**27.3 ms/step**, i.e. **300 kB/s**. The measured 372 ms is **13× (B) to 53×
(A)** the floor. **[D]**

**The floor does not say 3 %.** It says 300 kB/s–1.7 MB/s, i.e. 14–77× the
current 22 kB/s. A 3 % floor at 22 kB/s would require ~62 MFLOP/byte, which is
between ledger A and ledger B; neither of the project's own ledgers gives 3 %.
**At 22 kB/s the card is doing 0.6–3.4 TFLOP/s of a 45.7 TF peak: 93–99 % of
its arithmetic is idle.** [D]

### 2.2 But the card is not idle

`nvidia-smi` during the m400b run shows **12.5 GiB used and 83 % GPU
utilisation** **[M]**. The SMs are busy with something that is not FLOPs:
memory traffic, elementwise chains, the KDA recurrence, the optimizer, or
stalls. This is the signature of a **latency/memory-bound step**, not a
throughput-bound GEMM step. Tuning GEMM tiles cannot fix it; reducing the
number of dependent small ops or fusing them can.

### 2.3 The measured parts do not add up to the step

| part | measured ms | share of 106.7 ms | source |
|---|---:|---:|---|
| dense attention projections (4×latent + out, synthetic arm) | **0.30** | 0.3 % | `part_bench.py` [S5] |
| one expert on its 32 rows | 0.23 | 0.2 % | [S5] |
| all 512 experts, Python loop (not our path) | 99.56 | — | [S5] |
| per-expert weights, one bmm (the fixed path) | 11.45 | 10.7 % | [S5] |
| norm/mHC-style elementwise chain (6 ops) | **1.67** | 1.6 % | [S5] |
| MoE (whole pass, `triton` arm) | **2.5** | 2.3 % | [S4] |
| **sum of measured parts** | **~25** | **~23 %** | — |
| **iteration budget (106.7 ms at 25.6 kB/s)** | **106.7** | 100 % | [S5] |
| **unaccounted** | **~81** | **~76 %** | — |

The 27.9 ms KDA fwd+bwd figure from the b1 study [S2] would fill most of that
81 ms if it transfers to m400 (d=1536/H=24 is 0.75× the b1 width, so ~21 ms per
call; ×3 calls = 63 ms). But the b1 figure is at `d=2048/H=32`, and the FLA
benchmark at our exact shape on a GB200 implies only ~0.1–0.3 ms per call
**[M]** [S11][S12]. The two brackets differ by ~20×. **There is no kernel
profile of m400, so the 81 ms cannot be attributed from the available data.**
This is the central open question.

### 2.4 Verdict on Q2

**1.7 % is not "normal" for a dense transformer, but it is normal for a
KDA-dominated step.** If the unaccounted 81 ms is the KDA kernel and its
backward, the whole step is running at roughly the KDA kernel's own 2–4 % MFU,
and there is no structural bug — just a low-MFU algorithm. If the 81 ms is
CSA/Engram/optimizer/elementwise, there is a structural problem that fusion can
fix. **The evidence does not distinguish these two cases; a `torch.profiler`
top-10 kernel table would.** [high on the gap, low on the cause]

---

## 3. What makes small-batch / skinny GEMMs slow on consumer GPUs

### 3.1 The premise check: M=8192 is not small

On a 70-SM RTX 5070 Ti, with 128×128 CTA tiles, one wave is 70 CTAs. At
`M=8192` there are `8192/128 = 64` M-tiles, so:

| N | N-tiles (128) | CTAs | waves | wave efficiency |
|---:|---:|---:|---:|---:|
| 64 | 1 | 64 | 1 | 64/70 = 91 % (but only 64 SMs used) |
| 128 | 1 | 64 | 1 | 91 % |
| 256 | 2 | 128 | 2 | 128/140 = 91 % |
| 512 | 4 | 256 | 4 | 256/280 = 91 % |
| 1536 | 12 | 768 | 11 | 768/770 = **99.7 %** |
| 6144 | 48 | 3 072 | 44 | 3 072/3 080 = **99.7 %** |

So **N=256 is not half-efficient because of waves**; it is 91 % wave-efficient,
and the measured cuBLAS rate at N=256 was 39.90 TF (91 % of the N=2048 rate)
**[M]** [S1 §2]. **N=64 is the cliff**: only 64 CTAs (less than one wave), and
the 64×64 tile is the least efficient tile in NVIDIA's own table **[M]** [S9].
**K=64 is the second cliff**: only four 16-wide k-steps in the mainloop, so the
prologue/epilogue and load latency dominate; measured 14.02 TF (32 %) **[M]**
[S1 §2].

### 3.2 NVIDIA's own numbers

The Matrix Multiplication Background User's Guide [S9] gives the canonical
examples **[M]**:

- **Tile quantisation:** with 256×128 tiles and `M=27648, K=4096`, going from
  `N=128` to `N=136` keeps the same number of tiles (2 per row) but the second
  tile is only 6.25 % useful; duration is constant, GFLOPS collapse. The same
  effect makes `N=256` fine (2 full tiles) and `N=64` bad (one 64-wide tile,
  the least efficient class).
- **Wave quantisation:** on a 108-SM A100, `M=2304, N=1536` is exactly 108
  tiles (one full wave); `N=1544` adds a 9-tile tail wave, so GFLOPS roughly
  halve and duration roughly doubles. On a 70-SM 5070 Ti the same logic
  applies at different N values; the wave boundaries are `N ≈ 70·128/64 = 140`
  for the M=8192 case, then every 140 columns. N=256 and N=1536 both land near
  full waves; N=64 does not.
- **Tile sizes:** 256×128 and 128×256 are "most efficient"; 64×64 is "least
  efficient". K should be a multiple of 8 for FP16 (64 for A100-class), and
  performance improves as K grows.

### 3.3 The sm_120-specific constraint

Consumer Blackwell (sm_120) has **no tcgen05/TMEM**; it runs warp-level
`mma.sync` (m16n8k16, plus an NVFP4 m16n8k64 atom), not the 256×128 tcgen05
tiles of B200 **[M]** [S2, citing the Colfax sm_120 tutorial]. That means the
effective CTA tile is 64/128, the 64×64 class is common, and the "large tile"
escape hatch that datacenter GEMMs use is unavailable. This is exactly why the
KDA chunk kernel — 64×64 chunks, sequential state — is at 2–4 % of peak even
on a GB200, and why a consumer card does not rescue it.

### 3.4 Guidance / thresholds

| dimension | threshold | why |
|---|---|---|
| **N** | ≥ 256 for 2 full 128-tiles; ≥ 512 for 4 waves; ≥ 1024–1536 for near-peak | N=64 is 23 % MFU; N=256 is 91 %; N=1536 is ~100 % [S1][S9] |
| **K** | ≥ 64 minimum; ≥ 128–256 to amortise the mainloop | K=64 measured 32 %; K=1536 is full rate [S1] |
| **M** | ≥ ~9 000 rows for one full wave at N=128 on 70 SMs; 8 192 is 0.91 wave, acceptable | M=8192 is not the problem; M=1024 would be [S9][S10] |
| **tile** | avoid 64×64; prefer 128×128 or 128×64 | NVIDIA's own efficiency table [S9] |
| **dtype** | bf16 with FP32 accumulate = half the FP16/FP16-accumulate peak on this class | spec footnote [S10] |

**The practical conclusion for dormouse:** the dense KDA projections at
`N=256/1536` and the `out_proj` at `N=64/1536` are already in the good-to-fair
range; the 0.30 ms measurement confirms it. The bad shapes are the **rank-64
factor pairs** (`N=64`, then `K=64`) and the **KDA chunk kernel** (64×64). If
you want GEMM-level MFU, widen the factor rank or fuse the two factor GEMMs
into one dense `d×d` GEMM — but that multiplies FLOPs by `d/(2r) ≈ 12×`, so it
is a loss unless the small GEMMs are below ~8 % MFU. At the measured 10–14 TF
(23–32 %), the low-rank form is still the right trade. **Do not "fix" N=256;
it is not broken.**

---

## 4. Techniques that raise MFU at small scale — documented gains and applicability

| technique | documented gain | at what scale | applies to m400? | evidence |
|---|---|---|---|---|
| **Sequence packing** | up to **2×** BERT phase-2 pre-training; 50–89 % of tokens are padding in variable-length NLP data | variable-length sequences | **No.** Our byte stream is a fixed 1024-window, zero padding. Gain = 0. | arXiv:2107.02027 [S16] |
| **N-widening / fused QKV** | **2.2×** at B8×S1024 vs three `nn.Linear` in one open-source kernel; FasterTransformer claims ~3× end-to-end at small batch vs XLA | small batch, eager baseline | **Small.** Our projections are already batched; the measured block is 0.30 ms (0.3 % of step). A 4→1 launch fusion saves microseconds. The 2.2× baseline is PyTorch eager, not our graph. | `qkv_fusion` README [S23]; NVIDIA FasterTransformer README [S26] |
| **Persistent kernels / Stream-K** | Stream-K: up to **14×** peak speedup on bad GEMM geometries, better average over 32 824 shapes; MonoMoE **1.54×** over vLLM grouped GEMM (decode); Argus **2.1×** vs PyTorch CUDA Graphs on TinyLlama decode | bad tiles / decode | **Unlikely to give 2× in training.** CUDA graph on our step gave **+3 %**; kernels <10 µs are 0.16 % of GPU time. Persistent decode kernels win on launch/memory-bound decode, not on a 106 ms training step. | arXiv:2301.03598 [S15]; arXiv:2609.04244 [S19]; arXiv:2609.12299 [S20]; arXiv:2611.00495 [S21] |
| **CUDA graphs** | measured **+3 %** on this step; the literature's 1.2–1.5× applies to launch-bound small models | small models, many launches | **Already done; exhausted.** +3 % is the proof that launch overhead is not the bottleneck. | internal measurement, [S1 §4] |
| **`torch.compile`** | **1.31×** at 523M (dynamic=True), 1.20× static; **0** or 0.20× at m400; KDA is `@torch.compiler.disable` in FLA | dense transformer, static shapes | **No.** The biggest kernel is excluded from compilation; the MoE's data-dependent shapes recompile. | [S1 §4]; FLA `chunk.py:178` [S8]; test ledger [S3] |
| **cuBLASLt heuristics** | tile selection only; no documented training-scale gain at these shapes | any | **No evidence.** It cannot fix K=64 or N=64; it may choose a better tile for N=256, already 91 %. | [S9] |
| **bf16/FP16 accumulate** | spec: FP16/FP16-accum is **177.4 TF** vs FP16/FP32-accum **~88.7 TF** on the 5070 Ti (2×); PyTorch exposes `allow_bf16_reduced_precision_reduction` and `allow_fp16_accumulation` | GEMM-bound | **Not our bottleneck; accuracy risk.** The step is not GEMM-bound. Verify loss parity before touching it. | [S10]; PyTorch source [S24] |
| **`allow_tf32`** | measured **1.4×** on fp32 GEMM in this repo; irrelevant under bf16 | fp32 GEMM | **No.** Already bf16. | [S1 §2] |
| **FlashAttention varlen / packing** | 50–73 % of peak on A100 for long-sequence attention; 2× with packing only when padding exists | long/variable sequences | **No.** Fixed 1024 windows; KDA is not FlashAttention. | arXiv:2107.02027 [S16] |
| **Optimizer fusion** | Triton NS **68.1 → 18.6 ms**; `opt_graph` **−12 ms/step** (3 239 → 2 launches) | 199M–1B | **Yes, cheap.** If the optimizer is 30–60 ms of the step, these save 10–30 ms (3–8 %). | test ledger [S3] |
| **KDA `disable_recompute`** | **0.2 %**, +960 MiB | b1 | **No.** | test ledger [S3] |
| **FP8 `_scaled_mm`** | **1.6–1.8×** on the rank-64 GEMMs; forward-only; `_scaled_grouped_mm` refuses sm_120 | rank-64 GEMM | **No.** The rank-64 GEMMs are a small share; the grouped path is unavailable. | [S1 §2] |

**The technique table's message:** every documented 2×+ lever either (a) needs
padding/variable lengths we do not have, (b) is a decode/inference kernel, or
(c) targets a part of the step that is already <3 ms. The only universally
applicable cheap wins are optimizer fusion and the depth reduction.

---

## 5. Is the 3-iteration loop hurting GEMM shapes?

**No, and the literature has nothing that says otherwise.**

- Each iteration runs the same weight matrices at **M=8192**, which is a
  healthy M (64 M-tiles at 128, 0.91–11 waves depending on N). Repeating a
  weight at M=8192 is good for tensor cores and good for L2: the active
  per-iteration weights (KDA projections + out_proj + active experts, a few MB
  in bf16) fit in the 5070 Ti's L2, so iterations 2 and 3 should be cache-warm
  **[S]**. The loop does not create a bad GEMM shape.
- The loop's real cost is **sequential dependency**: iteration *n+1* reads
  iteration *n*'s state, so the three iterations cannot be concatenated into
  one `M=24576` GEMM. That is inherent to recurrent depth, not a shape bug. The
  hidden state is only ~25 MB per iteration; writing/reading it three times is
  ~150 MB/step, ~0.17 ms at 900 GB/s — negligible **[D]**.
- **The literature gap is real.** An arXiv search for "looped transformer"
  returns 208 hits; in the 20 I read, the ones that touch efficiency are about
  **memory** (MELT, arXiv:2605.07721: KV-cache growth with depth) or
  **compute-accuracy** (SpiralFormer, arXiv:2602.11698: multi-resolution
  recursion), not MFU or kernel shapes. **I found no paper that reports MFU or
  GEMM efficiency for looped depth; absence of evidence, not proof of absence.**
  [S22] [high that the gap exists; medium that the loop is neutral]
- **The one useful implication:** if depth 3 is not buying BPB over depth 2 at
  equal bytes, cutting to 2 is a free 1.3–1.5× on the whole step. The project
  already has a depth curve and a `--rand-depth` arm; use them. [S7]

---

## 6. Ranked list: the 5 changes most likely to move 22 kB/s toward 60 kB/s

### 6.1 The sensitivity frame

Let the step be 372 ms at 22 kB/s. Let `s` be the KDA time share and `x_K`,
`x_R` the speedups on KDA and the rest. New throughput is
`8192 / (372·(s/x_K + (1-s)/x_R))` ms. The table shows what 60 kB/s (136.5 ms)
requires:

| KDA share `s` | KDA speedup | rest speedup | new step | new kB/s |
|---:|---:|---:|---:|---:|
| 0.2 | 1× | 2× | 223 ms | 36.7 |
| 0.2 | 3× | 2× | 174 ms | 47.2 |
| 0.2 | 3× | 3× | 124 ms | 66.1 |
| 0.4 | 2× | 2× | 186 ms | 44.0 |
| 0.4 | 3× | 2× | 161 ms | 50.8 |
| 0.4 | 3× | 3× | 124 ms | 66.1 |
| 0.6 | 3× | 2× | 149 ms | 55.1 |
| 0.6 | 3× | 3× | 124 ms | 66.1 |

**To reach 60 kB/s you need roughly a 3× KDA and a 3× rest, or a 2.7× on
whichever path owns the unaccounted 81 ms.** No single documented technique
does that. The ranked list is therefore a *conditional* plan, not a forecast.

### 6.2 The five

| # | change | expected factor | what proves it | confidence |
|---:|---|---:|---|---|
| **1** | **Profile the step by kernel, not by MAC ledger.** `torch.profiler` with `record_shapes=True`, top-10 self CUDA, NVTX ranges per module (KDA, CSA, Engram, MoE, out_proj, optimizer). Reconcile the kernel FLOPs with ledger A and B. | **0× (gate)** | The top-10 table sums to 106–124 ms; the largest line names the target. Without it, every change below is a guess. | **high** |
| **2** | **Cut KDA passes 3→2 (or use the adaptive-depth arm).** This removes one-third of the KDA, MoE, and norm work; it is a config change, not a kernel. | **1.3–1.5×** (372 → 250–285 ms → 29–33 kB/s) if depth 2 holds BPB | Depth 2 vs 3 at equal bytes, 3 seeds, same window; BPB within the control spread, and ms/step. | **medium** (quality risk) |
| **3** | **Fuse the non-KDA path:** CSA top-k/gather, Engram gather/scatter, the mHC/AttnRes/LayerRoPE chain, and the optimizer (Triton NS + `opt_graph` + fused AdamW). | **1.2–1.4×** if the unaccounted 81 ms is here (372 → 266–310 ms → 26–31 kB/s); optimizer alone 1.05–1.1× | Kernel count and ms of the fused stage before/after; BPB parity; `nvidia-smi` utilisation. | **medium** |
| **4** | **Replace or supplement FLA's KDA with a training-capable sm_120 kernel** (mma.sync, register-resident state, chunk 64, fused backward), or switch the mixer to GDN-2. FlashKDA itself is forward-only/D=128/SM90, so it is not usable. | **1.2–1.5× overall** if KDA is 20–40 % and the kernel is 2–3× faster; **1.5–2×** if KDA is 60 % | Isolated `chunk_kda` fwd+bwd at b8,s1024,H24,D64 before/after; BPB parity; 0 NaN. | **low** (effort/risk) |
| **5** | **Raise occupancy/batch via activation checkpointing + FP8/offload.** b8 uses 14.2 GiB; b16 OOMs. Checkpointing `iter_checkpoint`/`selective` could fit b12–16 at +20–30 % recompute. | **1.1–1.2×** (the 5060 Ti batch ladder was +11 % at b16, +6.5 % at b32) | Tokens/s at b8 vs b12/b16, VRAM peak, BPB parity. | **low–medium** |

**Stack-up:** `1.3 × 1.3 × 1.2 × 1.1 = 2.23×` → **49 kB/s**; with a 1.15× on the
last, **56 kB/s**. 60 kB/s needs one more 1.1–1.2×, or a 2.7× on the single
largest path. **The evidence supports 35–45 kB/s with medium confidence; 60 kB/s
is low confidence and conditional on the profile.** [S]

### 6.3 What is closed (do not spend time here)

- **MoE:** 2.5 ms of 125 ms (2 %); its MAC ledger is 89.7 % one dense `f→d`
  GEMM. Closed as a class. [S4]
- **Dense projections:** 0.30 ms (0.3 %). Closed.
- **CUDA graph:** +3 %, already in. Closed.
- **`torch.compile`:** 0 at m400; KDA is disabled from it. Closed.
- **Sequence packing:** 0 (fixed-length bytes). Closed.
- **TF32:** 1.4× on fp32, irrelevant under bf16. Closed.
- **`N=256` widening:** N=256 is already 91 % of the big-GEMM rate; the cliff is
  N=64. Closed.
- **bf16/FP16 accumulate:** only matters if the step is GEMM-bound; it is not.
  Revisit only after the profile shows GEMM >50 %.

---

## 7. VERIFIED (sourced) vs SPECULATION (my inference)

### 7.1 VERIFIED

- The m400 geometry and parameter count: `port/configs/m400.toml`; the run
  banner prints **199,275,551 params** [S7].
- The measured part costs: dense projections **0.30 ms**, norm chain **1.67
  ms**, MoE **2.5 ms**, Python-loop MoE **99.56 ms** vs bmm **11.45 ms**,
  iteration budget **106.67 ms** at 25.6 kB/s [S5][S4].
- The 523M dense-transformer spike: **35.7 % MFU eager, 46.6 % compiled,
  99.9 % GPU utilisation** at b16×1024 on the 5060 Ti [S1 §4].
- The 5060 Ti GEMM table: **48.96 TF at N=2048, 39.90 at N=256, 14.02 at K=64,
  10.04 at N=64** [S1 §2].
- FLA `chunk_kda` at GB200: **2.327 ms** at T=8192,H=96,D=128; FlashKDA
  **1.009 ms (2.31×)**, forward-only [S11]. FLA KDA is **1.8–3.6 %** of peak;
  FlashKDA **4.1 %** [S25].
- FLA's `chunk_kda` is `@torch.compiler.disable` and accepts chunk sizes 32/64
  only [S8].
- Optimizer: Triton NS **68.1 → 18.6 ms**; `opt_graph` **−12 ms/step** [S3].
- NVIDIA tile/wave quantisation: N=128→136 doubles duration at constant tile
  count; N=1536→1544 on a 108-SM A100 roughly halves GFLOPS; 64×64 is the
  least efficient tile [S9].
- RTX 5070 Ti: **8 960 CUDA cores, 70 SMs, 43.9 TF FP32, 177.4 TF dense
  tensor (FP16/FP16 accum)** [S10].
- The project's own bf16 measurement on the 5060 Ti is **45.7 TF**; the
  purchase report says that number is the 5060 Ti's, not the 5070 Ti's [S1/S10].
- Sequence packing's 2× applies to variable-length data with 50–89 % padding
  [S16].
- Stream-K's up-to-14× is for bad GEMM geometries [S15]; decode megakernels
  MonoMoE/Argus are 1.5–2.1× [S19][S20]; a 2026 study warns persistent-CTA
  gains are often misread [S21].
- No looped-transformer paper reports MFU or GEMM efficiency; the efficiency
  papers are about memory (MELT) or compute-accuracy (SpiralFormer) [S22].

### 7.2 SPECULATION (explicitly marked)

- **SPECULATION:** the unaccounted ~81 ms/iteration is KDA/CSA/Engram/
  optimizer/elementwise. No kernel profile exists. The b1 KDA figure (27.9 ms)
  and the FLA GB200 figure (~0.1–0.3 ms at our shape) bracket it by ~20×, so I
  cannot pick a value. **This is the first thing to measure.**
- **SPECULATION:** the 45.7 TF denominator is the 5060 Ti's measured bf16 rate,
  not the 5070 Ti's tensor peak. If the run uses bf16/FP32-accumulate, the true
  MFU is ~0.9 %, not 1.7 %. A single cuBLAS bf16 GEMM at M=8192,N=1536,K=1536
  would settle it.
- **SPECULATION:** a custom sm_120 KDA training kernel could be 2–4× faster
  than FLA's Triton kernel. This is based on FlashKDA's 2.31× forward-only
  result and the internal b1 note's "worth ~3–4× on a kernel that is then
  ~1–3 ms"; no sm_120 training kernel exists to measure.
- **SPECULATION:** the loop's weight reuse helps L2 locality. The active
  weights are a few MB and fit L2, but no per-iteration timing exists to prove
  iterations 2–3 are faster.
- **SPECULATION:** the stack-up `1.3×1.3×1.2×1.1 ≈ 2.2×` assumes the savings
  are on disjoint paths. If two changes hit the same 80 ms, the product
  collapses.

### 7.3 Confidence by question

| question | confidence | why |
|---|---|---|
| Q1 comparable MFU | **medium** | strong dense-transformer anchor at 16k tokens; no published 8k-token number; KDA anchor is kernel-level |
| Q2 floor / structural verdict | **high on arithmetic, low on cause** | the floor is reproducible; the missing 81 ms is unprofiled |
| Q3 GEMM thresholds | **high** | NVIDIA guide + same-class measurements; the N=256 claim is contradicted by both |
| Q4 technique gains | **medium** | gains are documented at other scales/shapes; applicability is projected |
| Q5 looped depth | **high that the literature is silent; medium that the loop is neutral** | 208 arXiv hits; the 20 read do not report MFU |
| Q6 60 kB/s | **low** | no single 2.7× lever; stack-up is a projection |

---

## 8. Sources

Internal (file paths are relative to `/home/sehaxe/dormouse`):

- **[S1]** `docs/records/research/2026-10-15-torch-spike.md` — §2 (M=16384 bf16 GEMM
  table: 48.96/39.90/14.02/10.04 TF), §4 (523M `LoopBlock-lite` at b16×1024:
  357.3 ms eager / 273.5 ms compiled, 15.7/20.5 TF, 99.9 % GPU util).
- **[S2]** `docs/records/research/2026-10-15-tsct-super-kernel.md` — §0 (KDA fwd+bwd
  27.9 ms at b1 shape; sm_120 has no tcgen05/TMEM; 64×64 tiles; GEMM table).
- **[S3]** `docs/records/reviews/2026-10-21-test-ledger.md` — §4 (Triton NS 68.1 →
  18.6 ms; `opt_graph` −12 ms/step; KDA `disable_recompute` 0.2 %;
  `torch.compile` 0.20× in one test).
- **[S4]** `docs/records/reviews/2026-11-04-balance-fix-verdict.md` — MoE MAC ledger
  (114.9 GMAC/step; 89.7 % dense `f→d`; MoE 2.5 ms of 125 ms).
- **[S5]** `port/scripts/part_bench.py` and the `.bulba/goal.md` output —
  0.30 ms dense, 0.23 ms one expert, 99.56 ms Python loop, 11.45 ms bmm,
  1.67 ms norm chain, 106.67 ms/iteration.
- **[S6]** `port/src/dormouse/width.py::macs_per_token` — the full ledger
  formula and `attn_macs`/`out_proj_macs`.
- **[S7]** `port/configs/m400.toml`; `port/scripts/plan_restart.py`
  (`MEASURED_BPS=22_000`); `~/logs/m400.jsonl` (22.0–23.9 kB/s);
  `~/logs/m400b.jsonl` (22.7–23.9 kB/s).
- **[S8]** FLA 0.5.2 installed source,
  `port/.venv/lib/python3.12/site-packages/fla/ops/kda/chunk.py:178`
  (`@torch.compiler.disable`), `:390` (chunk size 32/64), `:49` (default 64).
- **[S25]** `docs/records/research/2026-09-27-kda-sota-ceiling.md` §1.3 — FlashKDA
  GB200 2.31×; 92.4 TF vs FLA 40.1 TF; 4.1 % vs 1.8–3.6 % of 2 250 TF. Note:
  the document's step-time claims at the top are retracted; only the kernel
  benchmark table is used here.

External:

- **[S9]** NVIDIA, *Matrix Multiplication Background User's Guide* —
  https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
  (tile quantisation, wave quantisation, tile sizes, K alignment).
- **[S10]** Wikipedia, *GeForce RTX 50 series* —
  https://en.wikipedia.org/wiki/GeForce_RTX_50_series (spec table: 5070 Ti
  8 960 cores / 70 SMs / 43.9 TF FP32 / 177.4 TF tensor dense; footnote defines
  the tensor column).
- **[S11]** MoonshotAI, *FlashKDA* `BENCHMARK_GB200.md`, generated 2026-05-26 —
  https://github.com/MoonshotAI/FlashKDA/blob/master/BENCHMARK_GB200.md
  (FLA `chunk_kda` 2.3271 ms vs FlashKDA 1.0087 ms at T=8192,H=96,D=128;
  forward only).
- **[S12]** FLA README benchmark, commit `0a484709`, GB200, CUDA 12.9,
  PyTorch 2.9.0 — B8,T1024,H8,D64: `chunk_gdn` fwd 0.631 ms / fwdbwd 1.974 ms.
- **[S13]** Kimi Linear, arXiv:2510.26692 (KDA; 6× decoding throughput at 1M;
  no training MFU).
- **[S14]** Mamba-2, arXiv:2405.21060 (SSD; 2–8× faster than Mamba's scan; no
  MFU).
- **[S15]** Stream-K, arXiv:2301.03598 (up to 14× peak speedup on bad GEMM
  geometries; better average over 32 824 shapes).
- **[S16]** Krell et al., *Efficient Sequence Packing without
  Cross-contamination*, arXiv:2107.02027 (50–89 % padding; 2× BERT phase-2).
- **[S17]** modded-nanogpt record #92, log
  `records/track_1_short/2026-08-30_ANVIL2/this_pr/01b2f094-...txt`
  (39 808 ms, 1 194 steps, 8×H100; FP8, no MFU reported).
- **[S18]** `Deveraux-Parker/nanoGPT_1GPU_SPEEDRUN` README (124M, 1×4090,
  ~160k tok/s peak, 90 min to 3.25; batch 16×32k in the script).
- **[S19]** MonoMoE, arXiv:2609.04244 (1.54× over vLLM grouped GEMM, decode).
- **[S20]** Argus, arXiv:2609.12299 (persistent TinyLlama decode 1.65 vs 4.92
  ms/token; 2.1× vs PyTorch CUDA Graphs).
- **[S21]** *Understanding GPU Megakernels: Why Residency Can Mislead*,
  arXiv:2611.00495 (PDL 5.7 %; persistent-CTA gains are resource-dependent).
- **[S22]** MELT, arXiv:2605.07721 (looped-transformer memory, not MFU);
  SpiralFormer, arXiv:2602.11698 (multi-resolution recursion, compute-accuracy,
  no MFU).
- **[S23]** `hilaryKChen/qkv_fusion` README (2.2× at B8×S1024 vs three
  `nn.Linear`; custom CUDA, eager baseline).
- **[S24]** PyTorch source `torch/backends/cuda/__init__.py` (the
  `allow_fp16_reduced_precision_reduction`,
  `allow_bf16_reduced_precision_reduction`, `allow_fp16_accumulation` flags).
- **[S26]** NVIDIA/FasterTransformer README —
  https://github.com/NVIDIA/FasterTransformer (`~3×` at small batch vs
  TensorFlow XLA under FP16 on T4; `1.2–3×` vs PyTorch for FT-Decoder).

---

## 9. Open questions (in the order that resolves the most uncertainty)

1. **Kernel profile of the m400 step.** Top-10 self CUDA, NVTX per module,
   `record_shapes=True`. Which line owns the 81 ms? This single measurement
   decides whether 60 kB/s is plausible or impossible.
2. **Denominator check.** One cuBLAS bf16 GEMM at `M=8192, N=1536, K=1536` on
   the 5070 Ti. Is the card's bf16/FP32-accum rate ~45 or ~85 TF? This decides
   whether the current MFU is 1.7 % or 0.9 %.
3. **Ledger reconciliation.** Count the MACs of the *executed* modules (not the
   `macs_per_token` formula) and compare to the profiler's FLOPs. Does the
   shared `d×f` down exist per token, or is it the MoE's top-2 `f→d`? The
   answer changes the floor by 3.6×.
4. **KDA time at m400 shape.** Isolated `chunk_kda` fwd+bwd at
   `b8,s1024,H24,D64`, one call, CUDA events. The b1 figure (27.9 ms) and the
   GB200 figure (~0.1–0.3 ms) cannot both be right for m400.
5. **Depth curve.** BPB at depth 2 vs 3 at equal bytes, 3 seeds. If depth 2 is
   within the control spread, the 1.3–1.5× is free.
6. **Optimizer share.** Time `opt.step`, `clip_grad_norm_`, and `zero_grad` at
   m400. If the optimizer is >30 ms, the Triton NS + `opt_graph` wins are real.
