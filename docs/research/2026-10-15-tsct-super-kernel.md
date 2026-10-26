# The TSCT super-kernel study — one RTX 5060 Ti, sm_120, ThunderKittens-style, inside the PyTorch port

**2026-10-15.** Question the owner asked: what does "TSCT must get more optimized, never
less" (plan `docs/decisions/2026-10-15-1b-readiness-plan.md` §7) mean MECHANICALLY on this
card, for the PyTorch port at `port/`, in October 2026. Scope: the custom compute the port
runs itself (experts, dispatch, the TSCT quant ladder); everything else is a dependency by
the standing principle ("use the best existing kernel"; plan §4.1). Evidence marks: **[M]**
= measured (source named), **[D]** = derived arithmetic from measured constants, **[S]**
= speculation/inference, explicitly flagged. Web sources fetched 2026-10-15; repo numbers
carry file:line.

## 0. The measured world this study builds on

| constant | value | source |
|---|---|---|
| card | RTX 5060 Ti 16 GB, sm_120 (12,0), driver 615.78.08, torch 2.13.0+cu129, triton 3.7.1 | spike `docs/research/2026-10-15-torch-spike.md:60-77` |
| b1 shape | d 2048 · H 32×64 · d_ffn 5461 · rank 64 · 2 iters · E 1024 topk-2 · shared rank 192 · 1,066,354,113 params, ~57M active/token (port counts ~37.9M touched per pass) | plan `:67-87`; `configs/b1.toml:8-39,89-120`; tests/test_param_count.py |
| the sm_120 quirk | SM12x has NO tcgen05/TMEM; warp-level `mma.sync` + NVFP4 blockscaled `m16n8k64` atom; TMA, CLC, PDL available; SM10x kernels incompatible, SM8x kernels run reasonably | Colfax tutorial (2026-06-20); **already known in-repo**: `docs/reviews/tsct2-design-2026-10-04.md:28` ("sm_120: block-scaled MMA mxf4nvf4, ~99 КБ smem, нет tcgen05") |
| expert pass (PORT, THIS card) | **50.6 ms / 6.05 GiB fwd-only**, padded bmm arm, E=1024 r=64 f=5461 cap=64, 16384 rows bf16; naive arm 571 ms (11.3×); parity padded-vs-naive fp32 1.8e-6 | `.bulba/memory.md:306` |
| spike expert variant | fwd 21.4 ms / fwd+bwd 52.6 ms (cap-padded bmm), 2.94 / 3.60 padded TFLOP/s; `torch._grouped_mm` 44.2 ms = 2× SLOWER, REJECTED; ragged grouped ~19 µs/group flat | spike `:143-157`, plan `:352-356`; memory.md:306 |
| bf16 GEMM rates at b1 shapes [M] | [16384,2048]×[2048,2048] 48.96 TFLOP/s · [16384,2048]×[2048,64] **10.04** · [16384,64]×[64,5461] **14.02** · [16384,2048]×[2048,256] 39.90 | spike `:96-99` |
| capacity ladder | cap 48 = 0.01 % dropped, −6 % wall vs cap 64; cap 128 = +23 % | spike `:148-157` |
| step breakdown at b16 s1024 (spike model) | 357.3 ms eager / 273.5 compiled (dynamic=True); 99.9 % GPU util; 683 launches/step; kernels <10 µs = 0.16 % of GPU time; gemm+attn 55.0 % · elementwise 26.9 % · optim 14.9 % | spike `:228-272` |
| KDA arm | FLA 0.5.2 `chunk_kda` fwd+bwd 27.9 ms at model shape = the dependency, with the COUNTED SDPA fallback | plan `:352-356`; `port/.../kda.py:1-49` |
| named headroom | MoE permutation traffic 7.9 ms; fp8 casts ~2.9 ms | plan `:366-368` |
| Liger fused CE | excluded everywhere (~40× wrong loss, then illegal memory access) | plan `:361-363` |

Derived step frame [D]: expert pass ≈ 50.6 ms fwd/pass → ×2 iters = 101 ms; fwd+bwd at the
spike's measured ratio 52.6/21.4 = 2.455× ≈ 124 ms/pass → 248 ms/step; KDA 27.9 ×2 = 56 ms;
so the expert pass is ~0.6–0.7 of a 380–480 ms eager port step (KDA + expert = ~65 %).
Every millisecond saved on the pass is counted twice. **This is the arithmetic that ranks
everything below.**

## 1. The kernel list — ranked by (win × maintained LOC)

Inputs/outputs are the port's own tensors: pure torch in/out (§4). "Maintained LOC" counts
code dormouse would own and debug at 3am, tests included.

| # | op | in → out | baseline [ms] | expected best [ms] | win | est. owned LOC | kind |
|---|---|---|---|---|---|---|---|
| 1 | **bank rebuild** (stack 5×[E,·] banks + zero-padded x-bank per forward, `experts.py:202-248`) | x [16384,2048] bf16 + 1024×{u,u',v,s,s'} → [E,64,2048]·2 + [E,64,5461] | inside the 50.6 (≈1.2 GB written/pass [D]) | ~1 (cache per optimizer step) | ~5-8/pass | ~30 + invalidation gate | delete, not write |
| 2 | **permute + gather + scatter** (`_permute` argsort 32k, `_padded_dispatch` index_put/index_add, `experts.py:168-264`) | index [n,2] → kept rows [R,·] | 7.9 [M, plan:368] | 2-3 (compiled or fused align) | ~5/iteration | ~0 (torch.compile) or ~80 (fused align) | reuse |
| 3 | **expert bank pass fwd+bwd** (3 bmms + GLU + shared expert + shared down, `experts.py:216-264`) | [R,2048] → [R,5461] | 50.6 fwd / ~124 fwd+bwd [D] | **25-35** via unpadded grouped GEMM + fused GLU; hard floor ≈20 at bench-A rates [D] | — | ~400-600 | **the one kernel worth writing** |
| 4 | **KDA fwd+bwd** (`kda.py:_fla_path`) | [b,t,d] → [b,t,d] | 27.9 [M] | 27.9 (FLA is the kernel) | 0 | 0 | dependency |
| 5 | **fp8/fp4 factor storage rung** (gentle: factors packed, dequant to bf16 per step) | experts 1.26 GB bf16 [D] → 0.63 (fp8) / ~0.33 (NVFP4) / ~0.19 (sign+scales) GB | storage win, 0 ms | 0 ms (compute unchanged) | 0 | ~200-300/rung (littlebit-design §2 estimate) | ladder rung, not kernel |
| 6 | fp8 headroom (act casts at arm boundary) | — | 2.9 [M] | 1-2 via preprocessed banks | ~1 | ~50 | tune, defer |

Why #3 has a ≈20 ms floor: the bank's own FLOPs at cap 64 are 2·(2·65536·2048·64) + 2·(65536·64·5461)
= 34.4 + 45.9 GF [D]; at the bench-A measured rates for these exact shapes (10.04 / 14.02 [M],
spike:96-99) that is 6.7 ms; adding shared expert (~5) and the shared down [16384,5461]→[5461,2048]
367 GF at 48.96 → 7.5 gives ≈19-20 ms. The measured 50.6 runs at ~1.6 TFLOP/s of useful work —
the win is tile/padding efficiency, NOT more precision or launches. Note this pass is
compute-limited at the skinny N=64 latents, not bandwidth-limited: bank traffic alone
(~1.25 GB [D]) at ~450 GB/s is ~2.8 ms; even the perfect kernel stays in the ~20s.

**Rank by win × maintained-LOC → the drop-everything list is: (1) bank cache + (2) compiled
permute first — days, near-zero LOC; then (3), the single Triton grouped-GEMM project;
then (5), the storage rungs — which need no kernels at all.** #4 and #6 are τ≈0 work items.

## 2. Write vs reuse — per candidate, with the TK reality check

| op | torch.compile / torch-native | Triton (write) | torchao / torchtitan (reuse) | vLLM fused_moe | ThunderKittens (CUDA C++) | CuteDSL (CUTLASS python) | verdict |
|---|---|---|---|---|---|---|---|
| KDA | FLA dependency [M 27.9] | — | FLA 0.5.2 green on sm_120 [M] | no KDA | TK has DeltaNet/scan kernels but untested vs FLA here | — | **reuse FLA; 0 custom LOC** |
| expert bank pass | `bmm` padded = today [M]; `_grouped_mm` measured 2× slower at E=1024 [M]; compile 1.2-1.31× step [M] | **grouped GEMM: candidates = pytorch/torchtitan `triton_contiguous_group_gemm` (BF16, training+inference, 1.42-2.62× vs manual loop on DSV3 16B, H100; torch blog 2025-08-18) + GLU fusion; or liger-style; autograd via wgrad/dgrad same groups** | torchao moe_training: differentiable mxfp8/nvfp4 grouped GEMM stack, PR #4598 merged 2026-08-19, **Blackwell SM100+ only** (`is_sm_at_least_100()` gate; sm_120 wheel path unverified [S]) | `fused_moe` triton = inference/serving kernel, **no backward** — usable only as an alignment/dispatch reference | TK 2.0 (2026-01-11): header-only CUDA C++, MXFP8+NVFP4, no pip anymore, per-kernel Makefile builds, MIT; sm_120 is a supported TEST arch but the marquee Blackwell GEMM kernels are tcgen05 (B200) — they don't apply here [M Colfax/TK README] | CUTLASS has `blackwell_geforce` CuTeDSL blockscaled_gemm examples on sm_120 [M Colfax]; for plain bf16 grouped at r=64 it is over-kill vs Triton | **first compile+cache (0 LOC), then ONE Triton grouped-GEMM write (~400-600); TK/CuteDSL rejected for this op** |
| permute/dispatch/align | argsort+cumsum+index_put compiled [S≈5ms saved] | `moe_align_block_size`-class fusion (vLLM TritonMoE, arXiv:2605.23911 Apr 2026 = fwd-only) | torchao quant kernels have grouped-swiggle patterns [PR #4598] | vLLM's align+dispatch kernels = forward-only but the **align part is reusable** | TK fused dispatch+GEMM = multi-GPU EP work (fluffy-kittens, 2025-11-17) — not a single-card op | — | **reuse (compile / vLLM align kernel); do not write** |
| fp8 casts | torchao float8 Linear = stable (B200-centred tables [M]); on sm_120 unproven here | — | candle | — | — | — | **defer until profiled** (2.9 ms named) |
| NVFP4 / fp8 factor COMPUTE | `_scaled_grouped_mm` fp8/nvfp4: measured absent for sm_120 in the spike's stack ("sm_90/sm_100-only", plan:361-364) | Triton NVFP4 = SR via inline PTX (torchao #3264/#3384 pattern) — heavy | **torchao closed the NVFP4 training tracker Oct 7 2026** (issue #3293: #4369 SR+RHT, #4487 CuteDSL RHT, #4598 grouped autograd stack on SM100+, TE-bitwise oracles, DSV3 200M-token convergence) | vLLM/SGLang support dense NVFP4 inference on Blackwell [S third-party] | TK 2.0 ships NVFP4 kernels (B200-class) | Colfax: hw path exists on sm_12x; cuBLAS NVFP4 GEMM on SM12x [M]; **not the grouped shape we need from torch today** | **wait/third-party; write NOTHING before torchao ships sm_120 or the wheel provably runs it** |
| sign (LittleBit) factors | bf16-deconstruct = no kernel | ONE Triton "decode" kernel (sign/ternary/fp4 codes → bf16 in-kernel + standard mma, scales in epilogue) ~150-250 LOC — see §3b | torchao training docs list NO sub1bit training kernels (nvfp4 QAT = QAT-emulation only, perf 🔴 in the Mar-2026 table) | none | none shipped (BitNet/T-MAC = inference engines, LUT-int8, no backward) | none | **bf16-deconstruct first (no kernel); the fused decode kernel only after G-B wins** |
| steps/optimizer/graphs | compile dynamic=True [M best]; CUDA graphs "worth almost nothing" at b1 [M spike:264-272] | — | bnb8bit vs compile = recorded trade [M] | — | — | — | **reuse** |

**ThunderKittens heterogeneity reality-check.** TK is a CUDA C++20 header library: headers +
pybind/Makefile per kernel, CUDA 12.8+, gcc-11 (README, fetched 2026-10-15). Inside this
port that means a build system the repo does not have (uv/ruff/pytest only; spike ran with
"no compiler toolchain beyond what the wheels ship", spike:83). TK 2.0 dropped its
pip-package face, so each kernel is a separately-compiled .cu with an arch flag — worth it
ONLY where Tensor-side Triton cannot reach, and on sm_120 nothing our model needs is in
that class (no tcgen05; NVFP4 exists but the consumer path in CUTLASS/CuTeDSL is already
natively wired by NVIDIA themselves [M Colfax]). The Megakernel line (blog 2025-05-27;
repo HazyResearch/Megakernels) is batch-1 decode, memory-bound, hand-scheduled per
instruction template, L40S/B200-targeted, forward-only — its paper-number context is vLLM
2.5×/1.5× H100-follow-ups and compiler follow-ups (MPK arXiv:2512.22219 cites Spector et
al.'s manual Llama-1B megakernel; AutoMegaKernel arXiv:2606.09682) — none of that maps to
single-card bf16 TRAINING at 99.9 % util [M spike breakdown]. Cite as inspiration
(interpreter/counter pattern), not adoption.

## 3. The TSCT STE-quant ladder — its EXECUTION plan

State today: the ladder (fp8 → NVFP4 → int4 → ternary → sign) exists **as gates only** —
`tsct.py:13-15` refuses every non-`"none"` rung at config load (`_refuse_ladder_rung`), and
the forward computes bf16 factors with fp32 masters and no quantizer (`tsct.py:79-91`). A
rung has never been run. The LittleBit design is pre-registered with its A/B gates G-A..G-D
(`docs/reviews/littlebit-design-2026-10-15.md:77-90`) and ~200-300 LOC per rung (`:60-63`).

### 3a. fp8/NVFP4 factors stored as their pack — the STORAGE rung (no kernel bank)

Mechanically: masters stay fp32 (retraction/Muon live on latents); after the per-step
reconstruction (littlebit-design §2.3's "update master → Gram-NS → Q(master)"), materialize
the **bf16 bank once per optimizer step** in the rung format and dequantize to bf16 for the
bmms — or, lazily, cache the dequantized bank between steps with the version counter from
below (§5 step 1) and quantize in place. The forward path is IDENTICAL bf16 math, so the
storage win is free of kernel risk [D]. What it buys, at b1 [D]:

| storage | routed+shared bank bytes | vs bf16 |
|---|---|---|
| bf16 (today) | 628.3M params × 2 B ≈ **1.26 GB** | 1× |
| fp8 factors | ≈ **0.63 GB** | 2× |
| NVFP4 (4-bit + UE4M3 scale/16) | ≈ **0.33 GB** | ~3.8× |
| sign(U),sign(V) + fp16 row/col/s-rank scales | ≈ **0.19 GB** (dual-latent corrected; see §3c) | ~6.6× |

The quant/dequant cost is per step (not per pass) and rides the existing retraction seam;
compute stays bf16 — exactly the burn semantics the spike's fp8 negative already spelled
("factors stored low-bit, computed fp32 masters → bf16 matmul", plan:361-366). **This is
the part of "super-optimized TSCT" that needs ZERO kernels**: it changes what is stored,
not how the GEMM is written.

### 3b. LittleBit — the cheapest expert GEMM is a sign-multiply; what fuses on sm_120

LittleBit (arXiv:2506.13771 v5, NeurIPS 2025; repo SamsungLabs/LittleBit) factorizes
W = (row-scales) ⊙ sign(U) ⊙ diag(s) ⊙ sign(V)ᵀ ⊙ (col-scales) with residual compensation.
On THIS card, three execution paths exist, and they are not equal:

1. **bf16 sign-decompose (no kernel, day-one rung).** `sign(U)` ∈ {−1,+1} is EXACTLY a bf16
   value; store 1-bit codes + fp16 scales, decode to bf16 bank per step, run today's bmms.
   Storage rung per §3a; the G-A..G-D A/B runs unblocked. `[M]` the LittleBit repo itself is
   E2E QAT + SmoothSign in pure PyTorch (torch 2.8, conda env, no custom CUDA kernel in
   the README build matrix) — the deconstruct-then-standard-mmu is what shipping code uses.
2. **NVFP4 blockscaled mma on sm_120 (the hardware path).** The sign/ternary codes fit
   e2m1's grid exactly ({0,±0.5,±1,±1.5,±2,±3,±4,±6} [M Colfax]); one PTX instruction
   `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` computes the
   sign-GEMM at the FP4 tensor rate with UE4M3 block scales. Hardware exists TODAY [M:
   Colfax; cuBLAS NVFP4 GEMM on SM12x]. What does not exist [S, probe item]: a torch
   wheel path that routes OUR grouped r=64 GEMMs there (`_scaled_grouped_mm` measured
   absent on sm_120; torchao's NVFP4 grouped stack is gated SM100+, closed Oct 7 2026).
   When it lands, the sign rung becomes a DEQUANT-FREE format: the pack IS the operand.
   But note the rate reality [D]: the win from a 4× dtype rate only pays where tiles are
   full; our latents are N=64 — the same tile-starved GEMMs §1 priced. LittleBit's "11.6×"
   is a paper's potential for WIDE matrices (arXiv:2506.13771 abstract), not an sm_120
   skinny-N measurement.
3. **bit-serial / LUT-integer (XNOR-popcount, T-MAC/LUT-tensorcore int8).** The old
   tri-pixel plan's conclusion stands (`littlebit-design:33-36`: the kernel speedup needs
   popcount/XNOR cores the cubecl path lacked) — in torch, these exist only as
   inference engines (BitNet.cpp arXiv:2502.11880, T-MAC) with no backward and no training
   story. `[M]` No training-side implementation is known to exist. REJECT for training.

**The one fused kernel worth considering, ONLY after G-B wins:** a Triton kernel that
loads the 1-bit/4-bit packed factors + fp16 row/col/s scales and produces
`(x @ Û)·diag-s·(V̂)` per tile with a standard bf16 mma in-kernel (the decode is K-sized:
for r=64, gate/up/sign words are 16 B/expert-rank — decode traffic is noise). It turns out
this is the SAME kernel shape as the NVFP4 store-decode of path 1 — one Triton kernel
serves the sign rung now and gray-scales into NVFP4 when the hardware path arrives.
Estimate ~200-400 LOC + STE backward (sign ≡ straight-through slope 1; scales are plain
learnable vectors; `act_quant.rs`-class machinery) — the honest cost of the rung, priced
against littlebit-design's 200-300 LOC for burn-side format+STE only.

### 3c. A correction the ladder arithmetic owes

`littlebit-design:30-31` prices the b1 expert at "bf16 U,V = 964 KB" and "sign+scales ≈
75.6 KB" — those are the plan §2.2 SINGLE-latent prices. The port's dual-latent GLU
(TSCT2 §14; `b1.toml:75` = 611,776 param/expert) doubles the U side: **bf16 ≈ 1.22 MB,
sign+scales ≈ ~109 KB/expert** [D] — ≈11.3×, not ≈12.8×, and "1B of expert capacity in
~150 MB" becomes **~190 MB**. The gates (G-A/G-B) are unaffected; the capacity E-growth
statement softens by that correction. Flag for the next revision of that design doc.

## 4. Interface discipline for reusable-but-not-a-goal kernels

The port's kernels are instruments in a training loop, not the goal; their API must let
torch.kill-them-myself happen at any time. House rule, 8 lines:

1. Every custom op is a function of tensors: `Tensor [R, K]` / `Tensor [E, K, N]` in,
   plain `torch.Tensor` out — dtype/device/layout in == contract declared in the docstring.
2. No module-level mutable state: no cached buffers inside the op, no workspace allocated
   in `forward()`; any cache lives in the CALLING module and is invalidated by a step-key
   gate (§5 step 1 — "a cache would silently outlive an optimizer step", `experts.py:208`).
3. No custom allocators, no CUDA graphs INSIDE the op; graphs capture the caller's stage.
4. Autograd correct or explicitly absent: a fused `forward()` ships with its backward as an
   `autograd.Function` whose grads match the naive path to ≤1e-6 (the padded-vs-naive
   precedent, `.bulba/memory.md:306`); anything less is LOUD, not silent.
5. Every fast arm registers a COUNTED fallback (FLA→SDPA is the template, `kda.py:157-168`);
   counters name the arm, reason on the counter object (`counters.why`).
6. No host sync: no `.item()`/bool→numeric on device (ADR-0018 rule 2 echo); capacity
   drops, kept-row counts stay device-side until the declared log cadence.
7. Same rows, same drops, same counter semantics as the reference path — parity by
   construction, not hope (`_permute` is "one place, both arms", `experts.py:168-184`).
8. Swappability test: replacing the op with the naive path (or bmm path) inside the same
   module must not change a parameter count or a loss curve beyond the parity gate.
   If it fails #8, the op is not "reusable but not a goal" — it is a rewrite.

## 5. What I would NOT build — the traps

1. **A Triton autotune treadmill for what torch.compile already covers.** The step's 26.9%
   elementwise share is inductor's job; the permute train (argsort/cumsum/index_add) is
   pure-torch and compile-able in ~0 LOC [S, but the win is the 7.9 ms named traffic]; only
   0.16 % of GPU time sits in <10 µs kernels, so micro-fusion buys ~nothing [M spike:251-272].
   Compile with `dynamic=True` — measured faster (273.5 vs 298.0) and the only variant that
   survives the grouped MoE's data-dependent shapes (recompile_limit hit at dynamic=False) [M].
2. **A custom fp8/NVFP4 grouped kernel before torchao ships sm_120.** The ecosystem closed
   this exact gap for SM100+ on 2026-10-07 (torchao #3293, with TE-bitwise oracles and
   200M-token convergence runs) [M]; hardware-side the sm_120 blockscaled path exists [M
   Colfax]. Writing our own CUTLASS/CuTeDSL variant duplicates months of layout work
   (interleaved scales, quad-thread scale TV layouts, swizzles — Colfax tutorial §3) to
   beat the clock by weeks at most, ON AN UNVALIDATED RUNG: no port rung above bf16 has
   ever run (`tsct.py:13-15,23-25`). And consumer-NVFP4 maturity is genuinely bleeding
   edge on GeForce — NVIDIA's own NVFP4 Qwen3.6-35B model reports illegal-instruction /
   CUDA-graph-replay hangs on RTX 5090 sm_120 (HF discussion, nvidia/Qwen3.6-.../9) [M].
3. **A megakernel before launches dominate — they don't.** GPU util is 99.9 %, 683
   launches/step, sub-10µs kernels = 0.16 % of GPU time, so the launch-bubble harvest the
   megakernel exists for is already spent; the step is GEMM+memory-bound (55.0 % + 26.9 %)
   [M spike:251-272]. The megakernel line targets batch-1 decode, forward-only, hand
   schedules per model; porting to a training backward on sm_120 is the maximal-LOC
   answer to a problem we don't have. PDL/CLC (available on SM12x [M Colfax]) and
   one-stage graph capture cover any residual gap cheaply.
4. **TK-as-own-tech.** Not one extension, not one .cu — until a Triton kernel measurably
   cannot reach a needed loop class (it cannot be proven until #3 is attempted), a CUDA
   build matrix (nvcc, gcc, per-kernel arch flags, no pip) buys risks, not rates [M
   README/spike]. The same verdict as burn's AGENTS §2.2 "route big matmuls through
   cuBLAS; do not write kernels", transposed into torch.
5. **Skipping the A/B.** Every rung/rkernel above is a mechanism, and this repo's law is
   A/B-or-death (ADR-0002). expert-kernel verdicts come from isolated-pass benches then the
   full step; rung verdicts come from G-B on 9.2M, 3 seeds, not from a benchmark win alone.
6. **The b1.toml capacity confusion.** The measured 50.6/6.05 profile is at cap=64
   (cf 2.0, memory.md:306) but `configs/b1.toml:28` sets `moe_capacity = 1.0` (cap 32 at
   16k tokens) — cheaper but never measured on this card, and routing is untrained, so the
   spike's drop-rates understate skew risk [D+]. Re-profile at the RUNNING cap before any
   arm A/B quotes expert-ms.

### The recommended 3-step order, with the arithmetic that picks it

**Step 1 — free wins (≈1 day, ~0-50 LOC).** (a) bank cache per optimizer step with the
step-key invalidation gate; deletes ≈1.2 GB/pass of bank re-stacking [D] ≈ 5-8 ms/pass ×2;
(b) `torch.compile(dynamic=True)` over the whole loop block (1.2-1.31× [M]); (c) pin cap to
the measured ladder's pick (48 = 0.01 % dropped, −6 % [M]) and re-profile at b1.toml's
actual cap; (d) re-run the pass probe: if the pass now sits ≥60 % off its ≈20 ms bench-A
floor [D], go to step 2; else stop — the pass is already in its headroom's top quartile.

**Step 2 — the ONE kernel: unpadded persistent grouped GEMM (+in-kernel GLU + bf16 scatter),
Triton (≈400-600 LOC, or adapted from torchtitan's `triton_contiguous_group_gemm`).**
Removes the cap padding (half the padded work is zeros at cap 64 [M: useful_flops_ratio
0.5]), the 716 MB [E,cap,f] intermediate [M spike:400-404], and the bank stack. Expected
pass: 50.6 → 25-35 fwd / corresponding fwd+bwd [D]. Backward = 2 grouped calls (dx via
weight-transposed groups, dw per factor) in the same kernel family — measure before
believing every piece. Gate: parity ≤1e-6 vs naive, then the step-level A/B; if it loses
to the padded bmm at any shape, the padded bmm stays (COUNTED).

**Step 3 — the ladder as STORAGE, not as kernels (≈200-300 LOC/rung).** fp8 factors →
NVFP4 pack → sign+scales, each rung = dequant-to-bf16 per step (§3a), banks cached, G-A..G-D
applied verbatim from `littlebit-design §4`. LittleBit runs as soon as its A/B says so
(owner decision point plan §6.5); the fused decode kernel (§3b path 2) is written only if
(a) G-B wins AND (b) the storage traffic is profiled as a real share of the pass
(bandwidth-bound it is not, yet [D]). Order of steps 3a/3b/3c IS the littlebit-design's own
§5 order; this study adds nothing that changes it.

Why this order: the pass is 0.6-0.7 of the step [D] — it is the only place a 5-LOC/day
change (step 1) and a 5-day kernel (step 2) can move the same number; the quant ladder
(§3) buys capacity/VRAM, not wall-clock, and can run GPU-light in parallel with step 2's
bench policy; and nothing above requires the torch wheel to ship anything it doesn't
already ship (torch 2.13 + triton 3.7.1 [M spike:47-60]).

## 6. VERIFIED / SPECULATION

**VERIFIED (sourced):** the b1 shape and param counts (b1.toml + tests); expert pass 50.6 ms
/ 6.05 GiB padded profile and 1.8e-6 parity (memory.md:306); spike's 21.4/52.6 ms, 339×,
grouped-vs-padded ladder, `_grouped_mm` rejection, bench-A GEMM rates, 99.9 % util +
profiler shares, [E,cap,f] 716 MB, Liger exclusion, dynamic=True recompile hazard (spike
with file:line); KDA FLA green 27.9 ms + numeric non-comparability (plan:352-356; kda.py);
7.9/2.9 ms named headroom (plan:366-368); littlebit gates and ~200-300 LOC (design doc):
G-A..G-D; littlebit repo = pure-PyTorch QAT, no kernels (README, fetched); TK 2.0 state —
header-only, no pip, sm_120 in tests, Vera Rubin, NVFP4+MXFP8 (README, fetched); sm_120 =
no tcgen05, NVFP4 blockscaled mma.sync m16n8k64, TMA/CLC/PDL, SM10x-kernel incompatibility
(Colfax 2026-06-20; tsct2-design:28 already recorded it); torchao NVFP4 training tracker
closed 2026-10-07 with grouped autograd stack merged 2026-08-19 gated `is_sm_at_least_100()`
(PR #4598, issue #3293); torchao official nvfp4-Linear perf still 🔴 in the Mar-2026 table;
torchtitan Triton grouped GEMM 1.42-2.62× vs loop (torch blog 2025-08-18); megakernel
numbers (blog 2025-05-27: <1 ms H100, 2.5×/3.5× vs vLLM H100/B200, breakdown).

**SPECULATION (my inference, marked):** the fwd/bwd ratio 2.455× and the ≈20 ms bench-A
floor for the port's pass are derived arithmetic, not port measurements; the ≈25-35 ms step-2
target is a projection with that floor; step-time totals for the port (380-480 ms) are
extrapolation from spike pieces + expert profile — no b1 port step has been benchmarked
end-to-end [S]; torchao's `is_sm_at_least_100()` includes sm_120 numerically, but whether
its NVFP4 grouped stack RUNS on the sm_120 torch wheel is unprobed [S — 30-min probe gate];
the megakernel paper's standalone arXiv id I could not confirm from fetched sources — the
verified citations are the blog (title/date/numbers above), the GitHub repo, and the
citing papers (arXiv:2512.22219, arXiv:2606.09682); the 7.9 ms permute win from ~0-LOC
compilation is the named-headroom number re-attributed [S until profiled]. LittleBit-2
(arXiv:2603.00042, ICML 2026) is init-only (Joint-ITQ) and adds no kernel [M README].

## 7. Sources

**Repo (file:line).** `docs/decisions/2026-10-15-1b-readiness-plan.md` (§2.2 shape :67-87;
§4.1 spike verdict :352-368; §7 :477-486; TSCT2 quotes :253-258, :249-252);
`docs/research/2026-10-15-torch-spike.md` (:60-77 env; :87-127 bench A; :130-192 bench B;
:195-272 bench C + profiler; :360-404 memory/LOC); `.bulba/memory.md:306` (port b1 profile);
`src/dormouse/model/experts.py` (docstring :18-35; `_permute` :168-184; `_banks`
:202-214; `_padded_dispatch` :216-264); `src/dormouse/model/kda.py:1-49,157-210`;
`src/dormouse/model/tsct.py:1-27,140-155`; `configs/b1.toml:8-39,55-90,120`;
`docs/reviews/littlebit-design-2026-10-15.md:1-100`; `docs/reviews/tsct2-design-2026-10-04.md:28,139-170`.

**Web (fetched 2026-10-15).** ThunderKittens README (github.com/HazyResearch/ThunderKittens;
TK 2.0 2026-01-11; Vera Rubin 2026-09-10; papers arXiv:2410.20399 / arXiv:2511.13940); TK
Blackwell blog (hazyresearch 2025-03-15: TMA/tcgen05/tiles on B200); Colfax, "NVFP4
Blockscaled GEMM on RTX Pro Blackwell (SM12x)" (2026-06-20: no-tcgen05 PTX `mxf4nvf4`
m16n8k64; CUTLASS blackwell_geforce CuTeDSL examples; cuBLAS NVFP4 on SM12x; 60 % of FP4
peak / ~3× over FP16 in tutorial kernel); torchao issue #3293 "NVFP4 Training Tracker"
(closed 2026-10-07) + PR #4598 (merged 2026-08-19, differentiable NVFP4 GroupedExperts,
SM100+ gate); torchao training docs (updated 2026-03-25: nvfp4 perf 🔴, mxfp8 ~2× on B200);
pytorch blog, "Accelerating MoEs with a Triton Persistent Cache-Aware Grouped GEMM"
(2025-08-18: 1.42-2.62×, torchtitan `kernels/triton_contiguous_group_gemm`); HazyResearch,
"Look Ma, No Bubbles!" (2025-05-27; megakernel numbers) + HazyResearch/Megakernels +
MPK (arXiv:2512.22219) + AutoMegaKernel (arXiv:2606.09682) as the citing chain; "Loads and
Loads of Fluffy Kittens" (2025-11-17: fused dispatch+GEMM = multi-GPU EP, paper
arXiv:2511.13940); LittleBit arXiv:2506.13771 v5 (11.6× = potential claim) + SamsungLabs/LittleBit
README (+ LittleBit-2 arXiv:2603.00042, init-only); BitNet.cpp arXiv:2502.11880 / T-MAC,
LUT Tensor Core (ASPLOS-2025) — integer-path implementations, inference-only;
TritonMoE (arXiv:2605.23911) fwd-fused dispatch; vLLM fused_moe docs (inference kernel,
no backward); HF nvidia/Qwen3.6-35B-A3B-NVFP4 discussions/9 (sm_120 NVFP4 stability failures,
GeForce).
