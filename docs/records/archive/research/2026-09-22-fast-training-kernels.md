# Fast training kernels for dormouse's step time: what 2025-2026 practice says

Date: 2026-09-22. Scope: why the dormouse train step runs at 6.7-8.3 s/step (batch 10 x 512 bytes, 7.5M params, 8-iteration LoopBlock, JEPA teacher, Muon+ ns=8, per-step polar retract, host-RAM Engram) and which current techniques fix it, targeting 5-10x bytes/s on one RTX 5060 Ti.

## TL;DR

The step is not FLOP-bound or bandwidth-bound. Back-of-envelope: ~1.8 TFLOPs per step (8 weight-shared iterations, fwd+bwd, no-grad teacher) against ~24 bf16 TFLOPS peak means ~75 ms at 100% MFU; 6.7 s implies roughly 2-3% MFU (my arithmetic from the measured split, not a profiler number). The published answer to this regime is the one the speedrun community converged on: run the recurrence and matmuls in bf16 tensor-core kernels, fuse the small ops, and then compound many 1-3% wins. dormouse already wrote the kernels (fused/, 1.7-2.0x measured); the blocked part is backward coverage, which is exactly where the remaining 2-3x lives. Newton-Schulz amortization, host-Adam overlap, and CUDA-graph-style launch elimination are real but second-order: together they are under 10% of the step. The honest 5-10x comes from fused backward (2x) x teacher cost reduction (1.2x) x bigger bytes-per-step (1.5x) x burn fusion (1.1-1.3x) x fp8 on the big GEMMs (1.1-1.3x).

## 1. modded-nanogpt and nanochat: what actually moved wall clock

The NanoGPT speedrun cut 8xH100 GPT-2 training from 45 min to 1.126 min (40x) and from 10B to under 400M tokens (25x) in 91 records [1]. No single technique did more than ~30%; the record is a compound of small wins. Per-record wall-clock deltas from the official log table [1]:

| Record | Technique | Time change | Delta |
| --- | --- | --- | --- |
| 3 | Muon introduced | 31.4 -> 24.9 min | -21% wall, and on 2x4090 the token budget fell 5.07B -> 3.04B at unchanged throughput [2] |
| 10 | bf16 activations | 8.2 -> 7.8 min | -5% |
| 12 | 1024-ctx dense -> 64K-ctx FlexAttention | 7.2 -> 5.03 min | -30% |
| 19 | FP8 head + offset logits | 3.4 -> 3.14 min | -8% |
| 20 | merged QKV, batched Muon | 3.14 -> 2.99 min | -5% |
| 27 | Triton symmetric matmul for Muon | 2.863 -> 2.817 min | -1.6% |
| 37 | cross-entropy in bf16 | 2.495 -> 2.483 min | -0.5% |
| 38 | Polar Express replaces Newton-Schulz | 2.483 -> 2.476 min | -0.3% |
| 39 | Adam params updated every other step | 2.476 -> 2.447 min | -1.2% |
| 44 | backward hooks on Adam (sync placement) | 2.284 -> 2.269 min | -0.7% |
| 57 | bf16 attn/mlp weights + mixed-precision Muon | 1.894 -> 1.878 min | -0.8% |
| 59 | fused Triton linear-ReLU² MLP | 1.820 -> 1.781 min | -2.1% |
| 60 | fused softcapped CE kernel | 1.781 -> 1.765 min | -0.9% |
| 68 | move bigram hash table to GPU | 1.540 -> 1.535 min | -0.3% |
| 71 | sparse bigram gradient comms | 1.528 -> 1.516 min | -0.8% |
| 74 | flattened forward | 1.485 -> 1.468 min | -1.1% |
| 80 | orthogonalize Q,K per head-pair not full matrix | 1.411 -> 1.406 min | -0.4% |
| 84-90 | FP8 MLP up/down, QKV fwd+bwd | 1.328 -> 1.126 min | ~-15% cumulative |

Lessons that transfer to dormouse:

- Muon's win is data efficiency at throughput parity, not step speed [1][2]. It changes tokens-to-loss, not s/step.
- bf16 activation storage was worth 5%; bf16 weights in the compute path, bf16 CE, and FP8 GEMMs each added fractions of a percent, stacking to ~20% over 30 records [1].
- Fusing one kernel per op-chain (MLP, CE, QK-norm+RoPE) is worth 1-2% each, and they kept stacking them [1].
- Even the "big" architectural kernel win (FlexAttention over dense 1024-ctx) was a 30% step, less than people remember [1]. A framework upgrade (PyTorch 2.4 to 2.5) was worth 9% by itself [2].
- The speedrun bans `torch.compile` config tweaks for record timing (7 min compile latency for seconds of runtime), i.e. compiler-driven fusion is standard background, not a record lever [1].

nanochat: full pipeline to GPT-2 capability in 1.65 h on 8xH100 (~$24-48). Its leaderboard wins: fp8 (3.04 -> 2.91 h), total batch 1M tokens (2.91 -> 2.76 h), a dataset swap, then two "autoresearch" tuning rounds to 1.65 h [3]. Its precision scheme is exactly dormouse's: fp32 master weights, cast to a global COMPUTE_DTYPE (bf16 on SM80+) inside each `Linear` on every forward, embeddings stored in bf16 [3]. The difference: in nanochat and the speedrun the GEMMs themselves execute in bf16 on tensor cores. In dormouse's `--bf16` mode the casts land before the `LinearLike` call and the matmul runs fp32 (the mixed-dtype NaN workaround). That is the single biggest structural divergence from published practice, and the fused/ kernels are the fix since they can do bf16 math in-kernel with fp32 accumulate.

## 2. KDA / GDN kernels: the current state of the art

The reference implementations live in flash-linear-attention (FLA, 5.8k stars): GDN shipped in Qwen3-Next (2025-09), KDA added 2025-10, KDA/GDN context-parallelism 2026-03, FlashQLA backend for GDN 2026-07, Preconditioned KDA 2026-06 [4]. All are Triton chunked-WY kernels.

Kimi Linear (the KDA paper): KDA is Gated DeltaNet with fine-grained per-channel decay; the chunkwise algorithm exploits a specialized Diagonal-Plus-Low-Rank transition form to cut compute; 3B-active/48B-total hybrid beats full MLA iso-recipe, KV cache -75%, up to 6x decode throughput at 1M context [5].

FlashKDA (Moonshot, 2026-04): CUTLASS KDA forward, 1.85-2.31x faster than FLA's `chunk_kda` at T=8192 on H20 [6][7]. Constraints that matter for dormouse: requires SM90+, K=V=128, and ships the forward kernel only [6]. It cannot be dropped into a sm_120 CubeCL trainer, and training needs backward. What is worth copying is the design, not the binary: qk L2-norm, beta sigmoid, and decay gate all fused inside the kernel; bf16 tensors with fp32 gate parameters (`A_log`, `dt_bias` fp32) [6]; chunked state passing.

FLA's own benchmark table is a useful calibration for dormouse's short sequences: at B=8/T=1024/H=8/D=64 the chunked GDN forward (0.631 ms) loses to FlashAttention (0.157 ms), while at T=8192/H=96 it wins 3x (1.265 vs 3.753 ms) [4]. Chunked linear attention is not intrinsically fast at dormouse's T=512. That is fine: dormouse's comparison is not against flash attention, it is against an fp32 tensor-op recurrence evaluated with a per-step dependency chain. FLA's chunked GDN covers a full 8k-token sequence in ~1 ms forward; dormouse spends seconds on 8 iterations of the same recurrence family in unfused fp32 tensor ops. The gap is implementation, not algorithm.

What dormouse's fused/ KDA is missing versus these: (a) bf16 tensor-core GEMMs inside the chunk (fused path is f32-only today), (b) full backward coverage (the production blocker), (c) gate/norm/beta fusion into one kernel launch per chunk rather than separate bridges. The FLA kernels implement exactly this split: small chunk-level GEMMs plus inter-chunk recurrence, with hand-written backward kernels [4].

## 3. Hashed embedding tables on GPU vs host RAM

Two published patterns:

| | Memory Layers at Scale (Meta) | Engram (DeepSeek) |
| --- | --- | --- |
| Table location | on-GPU, sharded [10][11] | host RAM, deterministic addressing [8][9] |
| Lookup | product-key top-k, fast embedding-bag gather/scatter kernels [11] | O(1) hashed n-gram lookup [8] |
| Scale shown | 128B memory params, 1T tokens [10] | Engram-27B, iso-param/iso-FLOP wins over MoE [8][9] |
| Public training kernel | yes, PyTorch embedding-bag based [11] | no; repo ships a demo mock only [9] |

Engram's paper states the design dormouse already implements: deterministic addressing is what makes host-memory offloading viable with minimal overhead [8][9]. At 48M rows (1.54B params, 18.4 GB with fp32 params + Adam m/v), the tables cannot fit a 16 GB card anyway; host RAM is forced, so the Meta-style on-GPU table is not on the table. The speedrun's bigram-hash-embedding records confirm the small-table variant: put it on-GPU (record 68, -0.3%), then exploit gradient sparsity in the optimizer update (record 71, -0.8%) [1]. dormouse's measured +2.3% step cost for every-step host-Adam is consistent with all of this; the transfer itself is not the bottleneck.

The actionable residue: the GPU-side gather (reading batch rows from the H2D buffer) and the scatter-add that produces row gradients are unfused bridges today. Fusing gather+concat into the attention/FFN epilogue and scatter-add into one kernel is the same class of win as speedrun records 59/60.

## 4. Muon cost engineering: ns=8 plus per-step retract is overkill per published practice

- Standard Muon runs 5 Newton-Schulz iterations [1][12]. dormouse pins `MUON_NS_STEPS = 8` (crates/dormouse-train/src/optim.rs:55).
- Polar Express replaces the coefficient schedule at identical wall clock (same degree-5 polynomial cost) and shows 5-6 iterations reach the accuracy beyond which even exact SVD stops improving validation loss [13]. In the speedrun it was worth 0.3% wall and a small loss gain [1].
- Newton-Schulz is routinely run in bf16 (Moonshot: halves communication) [14]; Tri Dao's Gram-NS work defaults to fp16 with a restart after iteration 2 and cuts the orthogonalization step 40-50% by iterating on the nxn Gram matrix, saving 55-68% FLOPs at aspect ratio 4 [12]. NS overhead across Muon training setups spans 2-17% of end-to-end wall clock [12].
- dormouse's measured opt 111 ms + retract 105 ms = 216 ms = ~3% of the step, already at the low end of that band, because the step is so slow for other reasons.

Conclusion for dormouse: ns 8 -> 5 with Polar Express coefficients is a free accuracy-neutral config change worth ~35 ms; raising `retract_every` from 1 to 4-8 with the existing max_ortho monitoring is worth up to ~90 ms. Both are hygiene, not levers. The report's instinct to keep masters fp32 and quantize only the forward path matches every cited recipe [3][12][13].

## 5. Precision and activation traffic

- bf16 activation storage: -5% wall in the speedrun [1]. bf16 attention/MLP weights: -0.8% [1]. bf16 CE: -0.5% [1].
- FP8 is now standard on the big GEMMs at small scale too: head (record 19), MLP up-proj (record 84), MLP down-proj with delayed scaling (record 89), QKV fwd+bwd (record 90), together ~15% on the 8xH100 record [1]; nanochat's fp8 leaderboard entry cut 0.13 h [3]. sm_120 has fp8 tensor cores; dormouse's fp8-overfit instability was observed on TSCT factors, not on these GEMM sites, so fp8 for controller/lm_head-scale matmuls stays open behind fused kernels.
- Cast-copy elimination: nanochat's scheme (fp32 masters, in-Linear cast, bf16 embedding storage) is the published shape of dormouse's `--bf16` [3]. The difference is where the cast happens: in-kernel (fused) versus as a separate tensor op before each `LinearLike`. Every separate cast is a full activation-sized read+write on a 448 GB/s bus.
- Low-rank expert matmuls at tiny scale get no dedicated literature; the speedrun answer is batching and layout: merge QKV (record 20), transpose one MLP operand to make the Muon matmul symmetric (record 27), fuse quantization into the head epilogue (record 67) [1]. The TSCT u/v factors at dormouse's rank are launch-bound; the win is fusion and batching across experts, not better GEMM kernels.

## 6. Launch overhead and the Rust stack

- NVIDIA's guidance: CUDA graphs pay off most where there are many small kernel launches [17]. Correction (2026-09-23): burn DOES ship a capture backend producing GraphIr since 0.22.0-pre.3 (#5377) — unmatured, but the launch-elimination path exists upstream; on 0.22.0-pre.3 the practical levers are burn fusion, autotune, and the memory pool.
- Burn 0.21.0 (May 2026) reworked the device handle (worker threads, lazy fire-and-forget task execution) and reports launch overhead down 5.4x on average with fusion enabled, up to 8.2x on small shapes; it added `burn.toml` with `[fusion.beam_search]`, `[cubecl.autotune] level = "minimal|balanced|extensive|full"`, compilation cache, `persistent_memory`, and `max_streams`; GEMV and top-k kernels reached LibTorch parity or better [15]. dormouse pins a burn fork at 0.22.0-pre.3 built from /home/sehaxe/burn-fused; whether that fork carries the 0.21 device-handle and fusion work should be checked once against the release notes, and `burn.toml` knobs are config-only experiments.
- burn's own flash attention is explicitly not done: the team calls compute-bound kernels hard and is building CubeK tile abstractions for them [15]. So hand-written CubeCL kernels for the loop, as fused/ does, is the right call, not waiting on upstream.
- cubecl autotune benchmarks kernel variants at runtime and caches selections [16]; dormouse already exposes `--autotune`. Setting the level high once at startup on fixed shapes costs nothing per-step afterwards.

## 7. alphaxiv scan (2025-2026)

- Engram is on alphaxiv with active discussion [8]; Polar Express likewise [13].
- LLM-written kernels are a theme (Dr. Kernel, RL-trained Triton generation, competitive with Claude-4.5 on KernelBench) [18]. The caution: KernelBench-Verified finds the best frontier model delivers 0.88x geometric mean speedup over torch under verified evaluation versus 1.43x under the unverified protocol [19]. For dormouse that reads: LLM assistance for CubeCL kernel coverage is plausible, but every generated kernel needs the same correctness harness fused/tests.rs already has.
- Nothing surfaced on alphaxiv targeting "fast small-model training on one consumer GPU" as a problem statement; the closest published artifacts remain the speedrun repos and nanochat, which are where the transferable numbers live.

## Checklist for dormouse

Ranked by expected s/step reduction on the measured 6.7-8.3 s split (fwd 2.2, bwd 4.3 incl sync, opt 0.111, retract 0.105). "Kernels" means new CubeCL kernel work in fused/; "config" means flags, burn.toml, or algorithm parameters.

| # | Item | Component | Expected s/step | Cost | Evidence | Kernels or config |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Finish fused/ backward coverage (KDA chunk bwd, MSA/Engram bridge bwd, scatter-add) | fused/backward.rs, kernels.rs | 6.7-8.3 -> ~3.4-4.5 (own 1.7-2.0x measurement) | high but kernels mostly written; correctness harness exists in fused/tests.rs | own fused measurement; FLA hand-written chunk backward as the reference structure [4] | kernels (complete, don't redesign) |
| 2 | bf16 math inside the fused kernels (tensor cores, fp32 accum, keep fp32 logits rule) | fused/kernels.rs | included in #1's 2x only if done; without it the fused path re-creates fp32 speed | medium; dtype plumbing per kernel | nanochat in-Linear bf16 GEMMs [3]; speedrun bf16 records [1] | kernels |
| 3 | Cut teacher cost: run the JEPA teacher under the fused no-grad path, and A/B computing teacher targets every k steps against jepa-weight 0 | aux.rs, jepa_targets.rs, loop_block | ~0.5-1.0 s off fwd if teacher is half of the 2.2 s fwd | low for no-grad fused run; A/B is config | no published EMA-teacher speedrun analogue; derived from our own phase split | config + kernels (reuses #1) |
| 4 | Raise bytes/step: batch 10 -> 16-20 at s512, or s768/s1024 at batch 8-10, as fused memory savings land | train cfg | 1.3-2x bytes/s, near-free if not compute-bound (it isn't at ~2-3% MFU) | low; VRAM check against the double forward | nanochat 1M-token batch [3]; speedrun batch/seq schedules [1] | config |
| 5 | Enable burn fusion + high autotune level + persistent memory via burn.toml; verify the 0.22-pre.3 fork carries the 0.21 device-handle rework | burn-fused fork, run config | ~0.2-0.6 s (8x launch-overhead figure is for tiny shapes; ours are mixed) | low, config only | burn 0.21 blog numbers [15]; cubecl autotune [16] | config |
| 6 | Overlap host-Adam D2H with next-step compute (double-buffer row grads on a side stream); read loss back only on log steps | offload.rs | ~0.2-0.4 s (measured sync cost is +2.3% step time, the stall is the recoverable part) | medium; stream plumbing in offload.rs | speedrun record 44 (Adam sync placement) [1]; our own host-adam-every measurement | config + small code |
| 7 | ns 8 -> 5 + Polar Express coefficients; `retract_every` 1 -> 4 with max_ortho watch | optim.rs:55, lib.rs cfg | ~0.05-0.12 s combined; accuracy neutral to positive | low | Polar Express 5-6 iters sufficient [13]; speedrun records 38/41 [1]; Gram-NS [12] | config |
| 8 | fp8 on the big GEMM sites (controller, lm_head, TSCT projections) with per-tensor scales, after #1 | fused/kernels.rs, act_quant.rs | 1.1-1.3x further (speedrun fp8 stack ~-15% across 7 records) | high; guard against the known TSCT fp8 overfit degradation; keep `--quant fp32` rollback | speedrun records 19/84/89/90 [1]; nanochat fp8 [3] | kernels |
| 9 | Fuse Engram gather into the consumer epilogue and row-grad scatter-add into one kernel | fused/kernels.rs, offload.rs | ~0.1-0.2 s | medium | speedrun records 68/71 (hashed table on GPU, sparse grads) [1] | kernels |

Compound estimate: 2.0 (items 1-2) x 1.2 (3) x 1.5 (4) x 1.15 (5) x 1.05 (6) x 1.1 (8) gives roughly 5x, i.e. ~1.3-1.7 s/step and the 5-10x bytes/s target once item 4 lands on top. Items 7 and 9 are hygiene and should ride along, not be sequenced.

## Refuted or corrected hypotheses

- **"Amortizing Newton-Schulz/retract is a win" (hypothesis 3): corrected to low priority.** opt + retract = 216 ms = ~3% of the step [12]; ns 8 -> 5 saves ~35 ms, retract_every 4 saves ~80 ms. Published practice also says ns=5 is enough and per-step ortho with 5 iterations is normal cost [1][13]. Do it, but it cannot contribute to 5-10x.
- **"Host-Adam overlap is a major win" (hypothesis 5): deprioritized.** The measured every-step host-Adam cost is +2.3% of step time; overlap and log-step-only loss readback recover at most ~0.4 s. The speedrun treats Adam sync placement as a ~1% item [1].
- **"CUDA-graph-like launch elimination is a top lever" (hypothesis 4): partially refuted as stated, confirmed as cheap config.** Burn's fusion cuts launch overhead up to 8.2x, but the absolute prize at the current kernel count is a few hundred ms, and after items 1-2 shrink the kernel count further it matters less. There is also ~~no CUDA graph capture in the burn/cubecl stack~~ **NO CUDA graph capture in the burn/cubecl stack; the mechanism is fusion + autotune + persistent memory [15][17].** **CORRECTED 2026-09-29, and this was the wrong way round: capture DOES exist and is CONFIRMED WORKING ON THIS GPU** - `vendor/cubecl-fix/cubecl-cuda/tests/graph.rs` is 5/5 green here (capture+replay, input rewrite, intermediate recycling, pool-growth refusal, many launches with dynamic metadata), and the client API is `graph_prepare` -> `start_capture` -> `stop_capture` -> `Graph::replay` (`cubecl-runtime/src/client.rs:1274+`). **It is not wired into the trainer.** The documented blocker: a capture window refuses stream reads, syncs and handle writes (`client.rs:1288-1296`), and a training step must do all three (write parameters, read the loss scalar for the NaN firewall and the log line) - so the first version has to capture ONE stage that needs no host round-trip, not the whole step.
  **And the claim this bullet was reasoning about has since been measured directly, in the direction the bullet did not predict:** the workload is launch-bound to a degree nobody had instrumented. Over a 150-step warm run at batch 32 the GPU is **13.3 % utilised on average with 142 of 180 samples at <=5 %** (`nvidia-smi` at 2 Hz, 2026-09-29, `benches/history.tsv`). Too many small kernels to fill the SMs. "Launch overhead is the lever" was the right instinct and the wrong mechanism: fusion + autotune + persistent memory is not what will fix it, graph capture is, and it is available today.
- **"Proper KDA/attention kernels are win (1)" (hypothesis 1): confirmed as the top lever, with a correction.** FlashKDA itself is not portable (SM90+ CUTLASS, K=V=128, forward only) [6][7]. What transfers is the chunked-WY-with-fused-gates design in bf16, which fused/ already implements in CubeCL. The missing piece is backward coverage, not kernel design or a new external dependency.
- **"Fusing the teacher forward is win (2)" (hypothesis 2): plausible but unverified by external sources.** None of the cited speedrun or nanochat material runs an EMA teacher; the ~0.5-1.0 s estimate is arithmetic from our own phase split. Treat as an internal A/B, not a literature-backed win.
- **Precision plan correction:** the target of "Muon moments bf16 + stochastic rounding" is consistent with published practice (NS in bf16/fp16 [12][13][14]), but every cited recipe keeps master weights fp32 and casts per forward [3]; dormouse's plan and current code already agree on this. The actual divergence is that dormouse's GEMMs run fp32 under `--bf16`; that is what item 2 fixes.

## References

[1] KellerJordan/modded-nanogpt README, record history and rules. https://github.com/KellerJordan/modded-nanogpt
[2] T. Romero, "NanoGPT Speedrun Worklog" (2xRTX 4090 replication). https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/
[3] karpathy/nanochat README, leaderboard and precision section. https://github.com/karpathy/nanochat
[4] fla-org/flash-linear-attention README, kernel list and benchmarks. https://github.com/fla-org/flash-linear-attention
[5] Kimi Team, "Kimi Linear: An Expressive, Efficient Attention Architecture", arXiv:2510.26692. https://arxiv.org/abs/2510.26692
[6] MoonshotAI/FlashKDA README. https://github.com/MoonshotAI/FlashKDA
[7] FlashKDA BENCHMARK_H20.md. https://github.com/MoonshotAI/FlashKDA/blob/master/BENCHMARK_H20.md
[8] DeepSeek, "Conditional Memory via Scalable Lookup" (Engram), arXiv:2601.07372. https://arxiv.org/abs/2601.07372 (alphaxiv: https://www.alphaxiv.org/abs/2601.07372)
[9] deepseek-ai/Engram repo. https://github.com/deepseek-ai/Engram
[10] Berges et al., "Memory Layers at Scale", arXiv:2412.09764. https://arxiv.org/abs/2412.09764
[11] facebookresearch/memory repo (product-key implementation on Meta Lingua). https://github.com/facebookresearch/memory
[12] T. Dao, "Gram Newton-Schulz: A Fast, Hardware-Aware Newton-Schulz Algorithm for Muon" (2026). https://tridao.me/blog/2026/gram-newton-schulz/
[13] Amsel, Persson, Musco, Gower, "The Polar Express", arXiv:2505.16932. https://www.alphaxiv.org/abs/2505.16932
[14] "Muon is Scalable for LLM Training", arXiv:2502.16982 (bf16 NS). https://arxiv.org/abs/2502.16982
[15] Burn 0.21.0 release blog (launch overhead, burn.toml, kernel status). https://burn.dev/blog/release-0.21.0/
[16] tracel-ai/cubecl (autotune). https://github.com/tracel-ai/cubecl
[17] NVIDIA, Best Practices for PyTorch CUDA Graphs. https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/best-practices.html
[18] "Dr. Kernel: Reinforcement Learning Done Right for Triton Kernel Generation", arXiv:2602.05885. https://www.alphaxiv.org/abs/2602.05885
[19] "KernelBench-Verified", arXiv:2607.16241; original KernelBench arXiv:2502.10517. https://www.alphaxiv.org/abs/2607.16241
