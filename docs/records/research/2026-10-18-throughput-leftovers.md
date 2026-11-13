# Throughput leftovers for the port — what 2025-26 shipped that we do NOT use

**Date:** 2026-10-18. **Read-only:** this file only; no config or code touched, nothing committed
(the `docs-site/tools/manifest.mjs` line is owed, §9).
**Question (owner):** for ONE RTX 5060 Ti 16 GB (sm_120) training the byte-level MoE+linear-attention
port, what is LEFT in the 2025-26 literature/ecosystem for **throughput** (not quality) that the port
does not already have? Survey and rank eight angles: selective/adaptive recomputation · full-step CUDA
graphs · fp8 without native autograd · elementwise-chain fusion · MoE grouped GEMM on sm_120 ·
allocator/runtime knobs · linear-attention training throughput · surveys.
**Method:** ~40 arXiv MCP calls (`search_papers`, `get_abstract` on every cited ID, `search_paper_text`
on SonicMoE/Puro-2B; no 429 hit, no export-API fallback needed) **plus local verification against the
installed stack**: `port/.venv` torch **2.13.0+cu129** / triton 3.7.1 (dispatch dumps, config files,
`libc10_cuda.so` strings, `_native` op conditions — quoted below with file:line), and fetched sources:
`pytorch/torchtitan` main, `fla-org/flash-linear-attention` main + the installed FLA 0.5.2,
`linkedin/Liger-Kernel` main, `NVIDIA/cutlass` main (4.8.0), `pytorch/ao` main.
**Marks:** **[V]** = verified in the installed wheel/repo today (file named) · **[A]** = authors'
abstract read · **[T]** = full text passages read · **[ours]** = port file:line · **[S1/S2]** =
SPECULATION (S1 = sourced mechanism, transfer unmeasured; S2 = inference only).

## 0. Corrections and re-verified givens

1. **The brief's "PyTorch 2.14" is wrong for this box: the port runs `torch 2.13.0+cu129`, CUDA 12.9,
   triton 3.7.1** (`port/.venv`, read today). Every "is it in our torch" answer below is against 2.13.
2. **`torch._scaled_mm` is forward-only — re-verified, and the reason is precise** [V]: the dispatch
   dump shows only `Autograd[alias]` (a requires-grad flag pass-through), no formula — same for
   `_scaled_mm_v2` (new recipe/swizzle API) and `_grouped_mm`.
3. **`_scaled_grouped_mm` is gated to sm_90/sm_100, not sm_120 — re-verified** [V], exact string in
   `libtorch_cuda.so`: *"torch._scaled_grouped_mm is only supported on CUDA devices with compute
   capability = [9.0, 10.0], or ROCm MI300+"*.
4. **`_grouped_mm` on sm_120 takes a fallback** [V]: the wheel carries *"fallback path for
   `torch._grouped_mm`, performance may not be optimal"* — this is why the port measured it 2× slower
   than the padded bmm (plan:352-356); the kernel it names (`_f8_f8_bf16_rowwise_grouped_mm`) is not
   what runs here. torchtitan's own CUDA-graph predicate says the same from the other side
   (`cuda_graph.py`: `_grouped_mm` *"may perform internal CPU<->CUDA copies not visible in FX metadata"*).

## 1. Already in the port — do NOT re-recommend
`opt_graph` (optimizer stage captured, clip included, `train/graph.py`) · `torch.compile(dynamic=True)`
(under test now) · dropless MoE Triton kernels with fused backward (`model/moe_kernels.py`, `moe_kernel`
kill-switch) · FLA `chunk_kda` + segmented state carry (`model/kda.py:327-375`) · fused `clip_grads`
(`_foreach_norm`) · Engram host tables · MoE activation offload via `saved_tensors_hooks` inside
`checkpoint(use_reentrant=False)` · forward-only fp8 2-D `_scaled_mm` · bnb8 paged (**+180 ms, rejected**)
· Liger CE (**broken, excluded**).

## 2. The table — ranked by expected value at OUR profile, not by paper numbers

Our profile (given, medium b16·s1024): fwd 194.5 / bwd 282.0 / clip 18.3 / opt 78.3 ms = 573 ms,
**GPU 100 % util**. That single fact re-ranks everything: the literature's *launch-side* class
(CUDA graphs, foreach fusion) is already largely spent here; the *GPU-work* class (fusion, precision,
tile efficiency) is where the remaining percent lives.

| # | candidate | source (id / repo, date) | mechanism, one line | measured on | claimed gain | our effort (seam, LOC) | class |
|---|---|---|---|---|---|---|---|
| F1 | **torch-native fused RMSNorm (quack, fwd+bwd) is GATED OFF for us** | torch 2.13 `_native/ops/norm/rmsnorm_impl.py:18-25` [V] | override of `aten._fused_rms_norm*` via vendored `torch._vendor.quack` | — | — | **0 — cannot engage** (`major in (9,10)`) | [V] |
| F2 | **`torch._native` op overrides that DO pass on sm_120** | `_native/ops/{foreach_mm,scatter_add,bmm_outer_product,topk}` [V] | auto-registered at `import torch` (`torch/__init__.py:3086-3087`); per-op `cond()`; kill switch `registry.deregister_op_overrides` | — | — | 0 (already active) — the question is whether our shapes hit the conds | [V] |
| F3 | **`_foreach_mm` → cuBLASLt grouped GEMM (nvmath)** | `_native/ops/foreach_mm/impl.py:1-40`, `nvmath_impl.py` [V] | `aten::_foreach_mm` override: SM90+, bf16/fp32, needs `nvmath-python` for the cublasLt grouped path, else `_grouped_mm`, else loop | — | — (no published number; bf16-only on the Lt path) | 0 LOC; `uv add nvmath-python` if pursued | [V] + [S1] gain |
| F4 | **Selective recompute with a memory budget under compile** | torch 2.13 `autograd/graph.py:459` `region_activation_memory_budget` + `_functorch/config.py:195-215` [V]; paper `2608.08740` (2026-08-09) [A] | knapsack over the joint fwd-bwd graph picks save-vs-recompute for a budget ratio ∈[0,1]; solvers `dp` (default) / `greedy` / `ilp` / `dp_knapsack_sliding_hirschberg`; `PARTITIONER_MEMORY_BUDGET_PARETO=1` dumps the curve | solver: n=2000 vs 100 items, 25-28 % solver speedup | buys VRAM at recompute cost; compile-only, one budget per graph | wrap compiled region ~5-20 LOC + A/B | [V] mech |
| F5 | **Per-op `CheckpointPolicy` incl. CPU offload (eager)** | torch 2.13 `utils/checkpoint.py` [V] | `create_selective_checkpoint_contexts`; `MUST_SAVE/PREFER_SAVE/MUST_RECOMPUTE/PREFER_RECOMPUTE/MUST_CPU_OFFLOAD/PREFER_CPU_OFFLOAD` | — | same class as F4, eager-compatible | `experts.py`/`loopblock.py` wrap, ~50-100 | [V] mech |
| F6 | **torch-native activation offload pass with prefetch + side stream** | `_functorch/config.py:234-248`, `_activation_offloading/activation_offloading.py` [V] | AOTAutograd pass inserts `ao::offload/reload/wait_tensor`; `activation_offload_separate_stream`, `..._sink_wait`, `activation_reload_prefetch`, `activation_offload_cpu_gpu_bw` (**user must set it**; default 50 GB/s) | — | upgrade of our synchronous hooks; the ram-offload doc called prefetch "absent today" | flip 1 config + set bw ≈25 (PCIe 4.0 ×16), A/B vs our hooks | [V] mech |
| F7 | **torchtitan memory policy / SAC tagging** | `torchtitan/experiments/graph_trainer/memory_policy.py:1-30` [V] | per-node `CheckpointPolicy` from op-pattern save sets (`full_recompute_save_ops`), RNG ops always saved, `log_activation_memory_policy` instrument, `paged_stash_memory_policy.py` = port of Megatron-Core 0.19 paged stash | 8×H100/B200 class | design reference; paged stash reclaims capacity-padded MoE activations | read whole; paged stash moot for us (dropless kernels removed padding) [S2] | [V] |
| F8 | Adacc / MemFine / OOMB (recompute papers) | `2508.00806` (2025-08-01) [A] · `2511.21431` (2025-11-26) [A] · `2602.02108` (2026-02-02) [A] | Adacc: tensor-level MILP choosing recompute/retain/compress; MemFine: chunked recompute for MoE under a memory model; OOMB: million-token context activation system | Adacc 1.01-1.37× vs SOTA frameworks; MemFine −48.03 % activation memory, +4.42 % throughput vs full-recompute baseline; OOMB long-context | as published | read as evidence; the torch-native F4/F5 are the same idea with 0 deps | [A] |
| G1 | **torchtitan GraphTrainer: full train-step capture** | `torchtitan/experiments/graph_trainer/README.md:11-14` [V] | `make_fx` traces fwd+loss+bwd (**optionally `optimizer.step`**) into one FX graph, no AOTAutograd partitioning; passes: memory policy, selective remat, CPU offload, regional/full inductor, CUDA graphs | 8×H100 / B200 CI | "full train step graph capture" | read whole as the reference design; we already have opt-stage capture | [V] |
| G2 | **How they handle loss scalar / NaN checks** | `torchtitan/distributed/CUDA_GRAPH.md:20-60`, `distributed/cuda_graph.py`, `experiments/.../cuda_graph.py:110-140` [V] | opt graph **includes gradient clipping and finite checks**; `.item()`/`torch.cuda.synchronize()` are excluded by a per-node predicate (`aten._local_scalar_dense`), so the loss read stays outside; 2 eager warmups; shared pool; `_CUDAGraphGradientState` for grad refs; chunked loss/lm_head run outside the root forward | — | — | our `train/graph.py` already mirrors the opt half | [V] |
| G3 | torch.compile `mode="reduce-overhead"` | `_inductor/__init__.py:358` [V] | sets `triton.cudagraphs=True`; inductor cudagraphs are otherwise OFF unless `TORCHINDUCTOR_CUDAGRAPHS=1`; cudagraph trees warn about **`.grad` allocated during capture** (needs preallocated/eager warmup) + `torch.compiler.cudagraph_mark_step_begin()` | — | per-region, not full-step; no optimizer | free to try under the current compile test | [V] |
| G4 | PyGraph | `2503.19779` (2025-03-25) [A] | compiler transformations to widen CUDA-graph coverage + kill param-copy overhead + cost-benefit selection; "more than doubles the benefit vs PyTorch2" on 25 workloads | TorchBench/HF/TIMM | — | research prototype; upstreamed? **not verified** | [A] |
| P1 | **fp8 fwd + hp bwd via custom `autograd.Function`** | `torchao/float8/float8_linear.py:28-120` [V] | `matmul_with_hp_or_float8_args`: fp8-cast in forward, **hp tensors saved**, grad_output cast in backward — the canonical pattern our forward-only `_scaled_mm` needs | Llama3-class | — | ~150-250 LOC + numerics gate, seam = the big 2-D projections | [V] mech |
| P2 | **Puro-2B: blockwise FP8 from scratch on RTX 5090** | `2608.27370` (2026-08-27) [T] | consumer-GPU recipe: blockwise FP8 "reduces per-token execution time while maintaining comparable quality to BF16" | **sm_120, 1.4 T tokens, <$6.9 K** | the only scale+HW-matched fp8 datapoint found; §3.2 not read → details unknown | read §3.2 whole before copying anything | [T]/[S1] |
| P3 | torchao MoE training paths | `torchao/prototype/moe_training/README.md:123-127` [V] | MXFP8/NVFP4 MoE training = **SM100+**; **fp8 rowwise MoE = SM89+** (nominally ours) but built on `torch._grouped_mm` + Triton quant kernels; DeepGEMM backend needs `deep_gemm` installed (`can_use_deepgemm_grouped_training` gates only `major>=9`) | B200 cluster (1.3× vs bf16, Llama4 Scout) | — | unproven on sm_120; `_grouped_mm` fallback is the weak link | [V]/[S2] |
| M1 | **SonicMoE (the 2026 MoE kernel reference)** | `2512.14080` (2025-12-16) [T] | gather fused into the grouped GEMM's GMEM→SMEM load; tile-aware **token rounding** cuts padding waste; memory-efficient fwd/bwd with minimal activation caching (−45 % activation memory) | 96×H100 7B FSDP-2; Blackwell vs DeepGEMM | +25 %/+15 % fwd/bwd vs DeepGEMM; 1.16× kernel eff at high sparsity; 43 % TFLOPS fwd at n=256 | design ideas only — our dropless Triton kernels are the same family, minus gather fusion and token rounding | [T]/[S1] |
| M2 | CUTLASS 4.8 CuTe DSL on SM120 | `NVIDIA/cutlass` README (4.8.0, Sept 2026) [V] | SM120 is a supported target (CUDA 12.8+); new "GeForce" kernels are **dense pingpong GEMM + implicit-gemm fprop conv (fprop only)**; SM100a kernels explicitly incompatible with SM120 | — | — | no grouped-GEMM-with-backward for SM120 exists in the DSL | [V] |
| M3 | TMA-Adaptive FP8 Grouped GEMM / MonoMoE / RaMP | `2508.16584` [A] · `2609.04244` [A] · `2604.26039` [A] | Hopper-only padding-free FP8 grouped GEMM; quantized MoE **decode** megakernel; MoE **inference** megakernel polymorphism | Hopper / decode | — | reject: wrong arch / inference-only | [A] |
| A1 | Allocator/runtime knobs in 2.13 | `libc10_cuda.so` strings + `torch.cuda.memory` [V] | keys: `backend:native|cudaMallocAsync`, `expandable_segments`, `max_split_size_mb`, `roundup_power2_divisions`, `garbage_collection_threshold`, `per_process_memory_fraction`, `pinned_use_cuda_host_register`, `pinned_num_register_threads`, `release_lock_on_cudamalloc`; **cudaMallocAsync ignores max_split_size_mb / roundup / gc** and supports no snapshot/OOM observer; runtime setter `torch.cuda.memory._set_allocator_settings` | — | — | 1 env var per arm; no code | [V] |
| A2 | STAlloc | `2507.16274` (2025-07-22) [A] | spatio-temporal memory planning for training with recomputation/pipeline disruptions | — | — | read-only (our allocator is torch's) | [A] |
| L1 | **FLA `chunk_kda(..., disable_recompute=True)`** | installed FLA 0.5.2 `fla/ops/kda/chunk.py:50,193,258` [V] | skips the `qg` recompute in the backward (memory for speed); we never pass it | — | — | **one flag**, A/B | [V] |
| L2 | **FlashKDA CUTLASS backend is INFERENCE-ONLY** | installed FLA `fla/ops/kda/backends/flash_kda.py:37-70` [V] | backend registry (`FLA_FLASH_KDA`); `chunk_kda_verifier` returns False when `torch.is_grad_enabled()` | — | — | **reject for training** | [V] |
| L3 | `fused_recurrent_kda` | FLA `fused_recurrent.py` [V] | no backward; port already uses it under `no_grad` for decode | — | — | nothing to add | [V] |
| L4 | Linear-attention training papers | `2605.05838` MDN [A] · `2606.27229` CARVE [A] · `2608.06838` StateFlow [A] · `2507.01004` ZeCO [A] · `2604.21100` PGDN/PKDA [A] · `2510.26692` Kimi Linear [A] | momentum-rule chunkwise algorithm; value-axis-decoupled WY solve + megakernel; sequence **pipeline** parallelism; zero-communication sequence parallelism; curvature-aware preconditioning of KDA | MDN: "comparable training throughput" to KDA (i.e. **not** a speedup); StateFlow/ZeCO are multi-GPU; CARVE beats a GDN-2 baseline | — | read-only; nothing adoptable single-GPU | [A] |
| S1 | **Whole-read shortlist** | `2608.27370` Puro-2B [T] · `2604.05091` MegaTrain [A] · `2512.08242` Chopper [A] · `2512.14080` SonicMoE [T] | single-5090 fp8 pretraining recipe; single-H200 100B full-precision training (already in ram-offload); 8×MI300X FSDP characterization (not single-GPU); MoE kernel system | — | — | read whole: Puro-2B §3.2, SonicMoE §4-5 | [A]/[T] |
| S2 | "single-GPU training throughput survey" | — | **not found.** The 2026 surveys found are distributed (2506.09275) or characterization (Chopper). | — | — | — | [S2] |

## 3. The eight questions, answered in one line each
1. **Selective recompute:** the state of the art is now *inside our torch* — a knapsack budget under
   `torch.compile` (F4) and per-op `CheckpointPolicy` incl. CPU offload (F5); the 2026 papers (Adacc,
   MemFine) are the same idea at multi-GPU scale. Nothing to port.
2. **Full-step graphs:** torchtitan's GraphTrainer is the only 2026 framework capturing fwd+loss+bwd
   (+opt) as one graph; the loss scalar is handled by **partitioning at `.item()`** and the NaN/finite
   check is *captured inside* the optimizer graph. Our opt half already does this; the fwd+bwd half is
   worth little at 100 % util.
3. **fp8 without autograd:** the pattern is `torchao`'s `autograd.Function` (fp8 fwd, hp saved, fp8
   grads); Puro-2B proves the *class* works on a 5090. `_scaled_mm`/`_scaled_mm_v2` have no formula in
   2.13 [V].
4. **Elementwise fusion:** the torch-native fused RMSNorm exists but is gated `major in (9,10)` → we are
   excluded [V]; the live options are inductor (already testing), FLA's own fused modules (0 new deps),
   or Liger's Triton kernels (arch-generic; its CuTeDSL/cute extras are SM90/SM100/SM110 only).
5. **MoE grouped GEMM on sm_120:** `_scaled_grouped_mm` is gated to [9.0, 10.0] [V]; `_grouped_mm` runs
   its fallback here [V]; the only cuBLAS grouped path in torch is `_foreach_mm`→nvmath (bf16, needs a
   new dep); CUTLASS 4.8's SM120 kernels are dense GEMM/conv-fprop. Triton remains the answer; SonicMoE
   supplies the two ideas we lack (gather-into-GEMM, token rounding).
6. **Allocator:** the knob set is verified; `cudaMallocAsync` throws away the tuning knobs we would
   want, so the interesting one is `expandable_segments` (whether it is the 2.13 default is **not
   verified** — no getter exposed).
7. **Linear attention:** FlashKDA is inference-only [V], `fused_recurrent` has no backward [V], the
   2026 linear-attention *throughput* papers are multi-GPU or merely "comparable to KDA". The one live
   knob is `disable_recompute`.
8. **Surveys:** no single-GPU training-throughput survey exists; the whole-reads are Puro-2B,
   MegaTrain, SonicMoE, GraphTrainer's docs.

## 4. THE RANKED TOP-5 — concrete next actions, each falsifiable on this card
Ordered by (expected % at 100 % util) × (LOC). Every arm: 3 seeds, one batch×seq, window printed.

1. **Fusion pass on the block's glue (F3 + F2 + L4-adjacent).** *Bench:* `torch.profiler` kernel-time
   for the elementwise class in fwd+bwd at medium b16·s1024 with the current compile config; then swap
   `nn.RMSNorm` → FLA `RMSNormLinear`/`fused_norm_gate` (already a dependency) or Liger's Triton norm,
   and re-measure step ms + BPB. **Expected 2-6 %** step if the elementwise share after inductor is
   ≥15 % (the spike's pre-compile share was 26.9 %). *Falsify:* if the compiled elementwise share is
   <10 %, delete the arm — the compiler already did it.
2. **Re-profile the MoE pass at the RUNNING cap, then gather-into-GEMM (M1).** *Bench:* isolated
   expert-pass fwd/bwd ms vs the ≈20 ms bench-A floor (tsct §1); if ≥60 % off the floor, the next
   kernel is the persistent grouped GEMM with the gather fused into the SMEM load (+ token rounding for
   padding waste). **Expected up to 15-25 %** step if the pass is unlanded; **~0** if the dropless arm
   already runs ≤35 ms fwd. *Falsify:* the isolated pass bench itself; if the pass is within 40 % of the
   floor, stop.
3. **fp8 fwd + bf16 bwd on the big 2-D GEMMs (P1/P2).** *Bench:* the shared down-projection
   [16384,5461]×[5461,2048] fwd+bwd, bf16 vs fp8-fwd/bf16-bwd custom `autograd.Function`; then the
   step. **Expected 3-8 %** if GEMM+attn is ~55 % of GPU time and fp8 buys ≥20 % on those GEMMs;
   *Falsify:* if the isolated GEMM gain is <10 %, drop (the skinny-N shapes are tile-starved, not
   FLOP-starved).
4. **Selective recompute/offload to buy one batch rung (F4/F5/F6).** *Bench:* peak
   `torch.cuda.max_memory_allocated` + step ms + tok/s at b8·s1024 (today **UNSTABLE**, OOM ~step 16)
   and b16·s512, with `region_activation_memory_budget(0.4/0.7)` under compile, or
   `enable_activation_offloading` + `activation_reload_prefetch` + `activation_offload_cpu_gpu_bw≈25`.
   **Expected +5-15 % tok/s** if a rung is gained. *Falsify:* peak not ≥20 % lower, or BPB outside the
   control's spread.
5. **Two one-line knobs (L1 + Muon NS 5→3).** `chunk_kda(..., disable_recompute=True)` and the NS step
   count: step ms at b16·s1024, peak VRAM, BPB. **Expected 1-3 %.** *Falsify:* either knob's step delta
   below the run-to-run spread at fixed shape (repeat the same arm twice).

**Explicitly NOT worth building here (with the reason read):** fwd+bwd CUDA graph capture (launch-side;
GPU util is already 100 %, and the measured opt-stage capture gave −12 ms/step — the remaining upside
is the same order); `_scaled_grouped_mm` (gated off, §0.3); CuTeDSL grouped GEMM for SM120 (does not
exist, M2); FlashKDA / `fused_recurrent` in training (inference-only, L2/L3); paged stash (our dropless
kernels already removed the padding it reclaims, F7); MXFP8/NVFP4 MoE training (SM100+, P3); sequence
parallelism for linear attention (multi-GPU, L4); DeepFusionKernel (inference, `2602.11808`);
FlashMHF (an architecture change, `2512.06989`).

## 5. VERIFIED (sourced) — everything asserted above with its source
- torch 2.13.0+cu129, triton 3.7.1, CUDA 12.9 — `port/.venv`, read 2026-10-18 [V].
- Dispatch facts: `_scaled_mm`, `_scaled_mm_v2`, `_grouped_mm`, `_scaled_grouped_mm`, `_foreach_mm` all
  `Autograd[alias]`-only (no formula) — `torch._C._dispatch_dump`, 2026-10-18 [V].
- `_scaled_grouped_mm` cc gate string, `_grouped_mm` fallback string, cudaMallocAsync limits, allocator
  key list — `lib/libtorch_cuda.so`, `lib/libc10_cuda.so` strings [V].
- `region_activation_memory_budget` (compile-only, budget ∈[0,1], one budget per graph) —
  `autograd/graph.py:459-520`; knapsack solvers + estimator + Pareto env vars —
  `_functorch/config.py:195-231`; `dp_knapsack_sliding_hirschberg` present —
  `_functorch/_activation_checkpointing/knapsack.py:124`; `CheckpointPolicy` incl. `*_CPU_OFFLOAD` —
  `utils/checkpoint.py`; activation-offloading pass + its four config knobs —
  `_functorch/config.py:234-248`, `_functorch/_activation_offloading/activation_offloading.py:791-830` [V].
- `torch._native` auto-import (`torch/__init__.py:3086-3087`); fused RMSNorm gate `major in (9, 10)`
  (`_native/ops/norm/rmsnorm_impl.py:18-25`); `_foreach_mm` cond `major>=9` + nvmath cublasLt grouped
  dispatch table (`_native/ops/foreach_mm/impl.py:1-40`); `scatter_add` cond `major>=9` + declines under
  deterministic algorithms (`_native/ops/scatter_add/cutedsl_impl.py:60-85`); `topk` cond fp32-only
  (`_native/ops/topk/cutedsl_impl.py:102-134`); `torch/_vendor/quack` vendored [V].
- Inductor: cudagraphs off unless `TORCHINDUCTOR_CUDAGRAPHS=1`; `reduce-overhead` = `triton.cudagraphs`
  (`_inductor/config.py:1809-1825`, `_inductor/__init__.py:358`); cudagraph-trees grad-accumulation
  message + `cudagraph_mark_step_begin` (`_inductor/cudagraph_trees.py:2668-2680`) [V].
- torchtitan: GraphTrainer README (full train step, tensor-granularity policy, pass list), `CUDA_GRAPH.md`
  (opt graph includes grad clip + finite checks; `.item()`/`synchronize()` outside; 2 warmups; shared
  pool; `_CUDAGraphGradientState`; chunked loss/lm_head outside the root forward), `cuda_graph.py`
  per-node predicate, `memory_policy.py` policy set, `paged_stash_memory_policy.py` = Megatron-Core
  0.19 port — all fetched from `pytorch/torchtitan` main, 2026-10-18 [V].
- FLA: `disable_recompute` in `chunk_kda`; FlashKDA inference-only; `chunk_intra` already calls the
  token-parallel diagonal kernel; fused-modules list; news (CP for KDA 2026-03, PGDN/PKDA 2026-06,
  FlashQLA backend 2026-07) — installed FLA 0.5.2 + repo main [V].
- Liger: `nvidia-cutlass-dsl>=4.6.0` / `liger-cute-kernels` target Hopper/Blackwell SM90/SM100/SM110;
  Triton kernels arch-generic; op list — README main [V].
- CUTLASS 4.8.0: SM120 row in the supported table, SM100a-incompatible note, new GeForce kernels
  (dense pingpong GEMM, implicit-gemm fprop conv) — README main [V].
- torchao: MoE training requirements (MXFP8/NVFP4 SM100+, fp8 rowwise SM89+), `Float8Linear` autograd
  pattern, DeepGEMM gate — `pytorch/ao` main [V].
- Papers (abstracts read; every ID resolved via `get_abstract` on 2026-10-18, no 429): `2608.08740`,
  `2508.00806`, `2511.21431`, `2602.02108`, `2503.19779`, `2511.05811`, `2509.25149`, `2602.11808`,
  `2512.14080`, `2512.08242`, `2608.27370`, `2601.02609`, `2512.06989`, `2609.13612`, `2605.05838`,
  `2606.27229`, `2609.04244`, `2604.26039`, `2608.06838`, `2510.26692`, `2603.28823`, `2507.16274`,
  `2604.21100`, `2508.16584`, `2605.23911`, `2609.33916`, `2607.07953`. Full-text passages read for
  `2512.14080` (SonicMoE) and `2608.27370` (Puro-2B) [A]/[T].

## 6. SPECULATION (my inference, marked)
- **[S1] Any of the % gains in §4 transfers to our shapes.** Every source's number is at a different
  scale/hardware: SonicMoE at 96×H100/Blackwell, MemFine/Adacc multi-GPU, Puro-2B a *different* 2B
  dense model on 5090s, Liger on Hopper-class. **No ≤1B single-card throughput measurement exists for
  any candidate.** The benches in §4 exist precisely to replace these transfers.
- **[S2] `_foreach_mm`→nvmath would beat our shape-batched bmm on the skinny NS shapes.** The override
  only engages with `nvmath-python` installed and bf16 (fp32 returns `CUBLAS_STATUS_NOT_SUPPORTED`
  per its own docstring), and our optimizer is already captured by `opt_graph`, so the win would have
  to be GPU-time — unmeasured.
- **[S2] `expandable_segments` is worth a rung.** Not verified whether it is the 2.13 default, and no
  measurement exists on this card; it is a 1-env-var arm, so it is cheap to test rather than to argue.
- **[S2] The elementwise share is still ~25 % after compile.** The 26.9 % number is pre-compile (spike);
  inductor may have already fused most of it — the §4.1 measurement decides, not this document.
- **[S2] No single-GPU training-throughput survey exists.** Negative search result, not a proof.

## 7. Could not verify / open questions
1. **Puro-2B §3.2's exact FP8 mechanism** (blockwise = 1×128/128×128 like torchao's blockwise path? own
   kernels? which GEMMs in FP8, which in BF16?) — the section was not read; read it before copying.
2. **Whether `region_activation_memory_budget` and the knapsack behave under our data-dependent MoE
   graph** (dynamic shapes + graph breaks; the docstring warns a partial annotation is rejected, and one
   budget per graph means the MoE region and the rest need separate graphs).
3. **Whether inductor cudagraphs (G3) can capture our step at all** — the trees warn about `.grad`
   allocation during capture; our `zero_grad(set_to_none=False)` path exists for the opt graph only.
4. **Adacc/MemFine testbeds and whether their baselines are multi-GPU only** — abstract-level read.
5. **`torch._native` override coverage for our exact ops** — the conds pass on sm_120 for
   `foreach_mm`/`scatter_add`/`bmm_outer_product`/`topk`, but whether our shapes meet each cond
   (dtype, alignment, CoW, deterministic-algorithms flag) was not exercised (no GPU run allowed while
   the compile test owns the card).
6. Owed per §1.8: a `docs-site/tools/manifest.mjs` line for this document (one-file mandate, no commit).
