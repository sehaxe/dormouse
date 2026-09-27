# Graph Report - dormouse  (2026-09-27)

## Corpus Check
- Large corpus: 695 files · ~660,799 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 9930 nodes · 23734 edges · 393 communities (373 shown, 20 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 815 edges (avg confidence: 0.84)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- CubeBackend
- CubeBackend
- tensor.rs
- Post-training 2026 comparison report (2026-09-27)
- fused_attnres.rs
- registry.rs
- CubeBackend
- ReduceArgs
- numeric.rs
- filter.rs
- capture.rs
- memory_manage.rs
- reference_for_config()
- CudaServer
- chunk.rs
- moe_fused.rs
- Accumulator
- tsct_diag.rs
- .stream_id()
- Command<'a, D>
- loop_block.rs
- binary.rs
- KernelId
- rope_cuda.rs
- CubeTensor
- comparison.rs
- ServerError
- ReduceBlueprint
- CubeCount
- device()
- im2col.rs
- DummyServer<M>
- ServerUtilities
- integration_test.rs
- ManagedMemoryHandle
- ReduceRequirements
- MhcBlock
- CubeElement
- CubeBackend
- DirectPool
- schema.rs
- deform_conv_transpose2d.rs
- select()
- ErrorGraph
- optim.rs
- CubeTensor
- .to_output_perpendicular()
- compiler.rs
- kernel.rs
- CubeBackend
- schedule.rs
- shape_divmod()
- .group()
- gather_nd.rs
- ManagedMemoryBinding
- event.rs
- serve.rs
- driver.rs
- CudaContext
- markov.rs
- unary_float()
- ServerLogger
- offload.rs
- burn-fused
- SctLinear
- fused_recurrent_cube.rs
- tuner.rs
- gen_reference.py
- HallSuppressor
- TestCase
- collective.rs
- DummyKernel
- CudaCompiler
- AutotuneResult
- benchmarker.rs
- ReduceError
- profiler.rs
- sparse.rs
- MemoryPage
- NativeExpand
- route_block()
- ReduceStrategy
- SchedulerMultiStream<B>
- reference_mean()
- nan_extrema.rs
- server.rs
- routing.rs
- WgpuDevice
- chunk_wy_forward()
- PersistentPool
- conv_data_backward_fallback()
- ReduceOperationConfig
- PinnedMemoryStorage
- MemoryLayout
- fwt_cuda.rs
- sampler.rs
- noise.rs
- execute_interpolate()
- slice.rs
- fused_situ.rs
- AutotuneError
- contiguous.rs
- BufferBinding
- bounds_generator.rs
- TunableSet
- EventStreamBackend
- plane_topk_insert()
- jepa_aux_loss_masked()
- ByteStream
- TensorHandle
- compute.rs
- CopyDescriptor
- attention()
- problem.rs
- zero_copy()
- communication.rs
- detect.rs
- ReduceOperation
- tune_cache.rs
- taint.rs
- to_vec()
- AcceptRatePredictor
- metadata_cache.rs
- .launching()
- JepaTargets
- SpectralLinear
- as_scalar()
- mask_fill_kernel()
- as_scalar()
- curve.rs
- min_insert()
- argtopk_shared_memory.rs
- ReduceCost
- self_checks.rs
- DormouseConfig
- model_seam.rs
- logger.rs
- extrema.rs
- empty_device_dtype()
- binary_int.rs
- .forward()
- handle.rs
- ExclusiveMemoryPool
- max_insert()
- shared_sum_kernel()
- alloc_probe.rs
- LazyDeviceController
- bench_cuda.rs
- benchmark.rs
- sinkhorn_cuda.rs
- EngramModule
- SpectralRetention
- burn-spectral (Ternary Spectral Compact Training)
- StreamPool
- Capability per byte and per GB for small LMs: research sur
- module.rs
- ProfilingToken
- GpuStorage
- taint_property.rs
- observer.rs
- Failures
- .launch()
- local.rs
- queue.rs
- DormouseModel
- bit_exact.rs
- MuonPlus
- activations.rs
- CudaStreamBackend
- Category
- KdaConfig
- fused_kernels.rs
- SpectralMoE
- MemoryUsage
- Compiler
- SlicedPool
- BytesStorage
- with_bounds()
- Community 178
- qr_cpu()
- burn-kda Kimi Delta Attention (KDA) for Burn
- Captures<D>
- PerpendicularWriter
- resolve()
- cache.rs
- DeviceStream
- qtensor.rs
- SourceTemplate
- ternary_mutate()
- autodiff.rs
- autodiff.rs
- qr_cuda.rs
- polar_orthogonalize()
- AccumulatorFormat
- reduce_dim.rs
- Fast training kernels for dormouse's step time (2026-09-22
- bitforbit.rs
- CubeBackend
- .to_output_perpendicular()
- InputGenerator
- .to_output_perpendicular()
- cublas_combos.cu
- .kernel_output()
- events.rs
- AutotuneKey
- memory_pools_config.rs
- conv_transpose2d_col2im()
- .to_output_perpendicular()
- DriverError
- ReduceWriter
- reference_argmax()
- PinnedMemoryAllocController
- conv_forward_nhwc()
- graph.rs
- Executable
- Event<A>
- CubeClRuntimeConfig
- deform_conv2d.rs
- scatter.rs
- hasher.rs
- dry_run.rs
- MemoryAccess
- roofline.rs
- .to_output_perpendicular()
- .to_output_perpendicular()
- ParallelWriter
- train.rs
- fused.rs
- act_quant.rs
- reduce_with_indices_kernel()
- kernel_binop()
- byte_patch()
- alloc_trace.rs
- KdaModule
- ThroughputValue
- EventPool
- FailureStore
- A/B or death: ties delete too
- lowp_bf16_cuda.rs
- cuda_retract.rs
- operation_sets.rs
- Rung 0 — MEASURE the depth curve BPB@k (k=1..4), per-itera
- conv_autotune()
- BenchConfig
- EventApi
- matmul_autotune()
- .reduce_single()
- dormouse --rand-depth arm (uniform T ∈ 1..4 per step, both
- select_assign.rs
- chunk_cube.rs
- capture.rs
- VectorizationMode
- .reduce_shared()
- topk_with_indices_cube.rs
- fused/ grad-coverage verification (phase 1 item 1)
- ADR-0014: MSA is cut, not deferred
- kda_step_probe.rs
- backend_parity.rs
- client.rs
- host_svd.rs
- streaming.rs
- .resolve()
- weights.rs
- .forward()
- topk_gather.rs
- CompilationConfig
- memory.rs
- topk_unroll_limit.rs
- tune_key.rs
- bf16_ops.rs
- .new_quantized()
- What goes on top of random-depth training in dormouse — ad
- CubeAutotuneKey
- chunk_adjoint_cube.rs
- forward_shape()
- LaunchObserver
- .reduce_single()
- ADR-0018: dormouse-fused is a standalone project, with thr
- create_key()
- .on_stream()
- write_scope.rs
- ADR-0013 fixed depth (PonderNet halting deleted after two 
- launch.rs
- launch.rs
- Presets nano..one_b share one skeleton; RAM buys Engram ro
- runtime.rs
- bench_catalog.rs
- kda_alloc_probe.rs
- runtime.rs
- Huginn — Scaling up Test-Time Compute with Latent Reasonin
- Eviction
- create_key()
- conv_gemm_simple_async()
- staging.rs
- EventFence<A>
- .compile()
- policy.rs
- PendingDropQueue<E>
- idle_check()
- CompiledKernel
- MetadataInfoCache
- fused_matrix_static.py
- bitforbit_cuda.rs
- fused_matches_tensor()
- ttt_ce_loss()
- LaunchObservation
- TuneCache<K>
- PendingDropQueue
- MetadataCachePolicy
- Loop runs at fixed max_iter=4 with honest unweighted CE
- ComputeCmmaConfig
- precision.rs
- StressMonitor
- ckpt_roundtrip.rs
- bench_fused_bwd.rs
- DeviceProbe
- Margin exit criterion p1 − p2 on the fp32 logits (CALM's w
- slice_assign_kernel()
- BlockPartition
- AmdDevice
- MetalDevice
- .execute()
- with_indices_validation.rs
- StorageHandle
- cmp_reference.rs
- gather_kernel()
- select_kernel()
- burn-fastblt Byte-Level BLT + FastBLT for Burn
- burn-situ CI Workflow
- situ_glu()
- CudaRuntime
- notify_profiled()
- TunePlan
- Two-region DCLM filter (head trains, tail evaluates)
- include_path()
- bce_label_is_a_fresh_topk_of_the_current_scores()
- dormouse loop facts read from code: h reset to h_ctx each 
- bench_ops.rs
- .tr_execute()
- cast_element()
- bool_cast_kernel()
- conv_depthwise()
- scatter_nd_kernel()
- JepaPredictor
- burn-sct Spectral Compact Training
- cubek-reduce (CubeK Reduce)
- Think-at-Hard (TaH) arXiv 2511.08577 (Fu, You, Chen, Dai, 
- RequiredAddrType
- fused_recurrent_forward()
- bf16_matmul.rs
- infer.rs
- .to_inference()
- .enumerate_all_devices()
- PlaneReduceBlueprint
- engram_ab.sh
- repeat_dim_kernel()
- mask_fill_auto()
- random_uniform()
- burn-rope Rotary Position Embedding with YaRN
- burn-swiglu (SiLU-Gated Linear Unit for Burn)
- burn-ttt (Test-Time Training loss for Burn)
- CpuDevice
- .generate()
- GatedStream
- build.rs
- BufReader
- fence.rs
- fused_matrix.sh
- cross_kernel()
- ManagedResource
- GcTask<B>
- time_ms()
- mor_ab.sh
- mask_indices()
- Formatter
- build.rs
- ste_ternary()
- .can_read_tensor()
- TuneRecord<K>
- Anchoring<A>
- migrate-dormouse-fused.sh
- burn-situ SiTU-GLU activation
- OBSERVING
- cubek-reduce
- cublas-poc

## God Nodes (most connected - your core abstractions)
1. `Device` - 246 edges
2. `P` - 169 edges
3. `Client` - 159 edges
4. `CubeBackend` - 112 edges
5. `BufferBinding` - 99 edges
6. `Accumulator` - 95 edges
7. `CubeBackend` - 94 edges
8. `ErrorGraph` - 88 edges
9. `Handle` - 81 edges
10. `KernelId` - 70 edges

## Surprising Connections (you probably didn't know these)
- `compute_name()` --references--> `String`  [EXTRACTED]
  crates/cublas-poc/src/main.rs → vendor/cubecl-fix/cubecl-runtime/src/tune/operation.rs
- `jepa_aux_loss()` --calls--> `mask_indices()`  [INFERRED]
  crates/dormouse-core/src/aux.rs → vendor/burn-fused/crates/burn-jepa/src/mask.rs
- `dspark_aux_loss()` --calls--> `dspark_loss()`  [INFERRED]
  crates/dormouse-core/src/aux.rs → vendor/burn-fused/crates/burn-dspark/src/lib.rs
- `memory_floor_caps_the_controller()` --calls--> `t()`  [INFERRED]
  crates/dormouse-core/src/loop_block.rs → vendor/burn-fused/crates/burn-gdn2/tests/alloc_probe.rs
- `main()` --calls--> `ms()`  [INFERRED]
  crates/dormouse-train/examples/kda_alloc_probe.rs → vendor/burn-fused/crates/burn-sct/examples/bench_ops.rs

## Import Cycles
- 1-file cycle: `vendor/cubecl-fix/cubecl-runtime/src/launched.rs -> vendor/cubecl-fix/cubecl-runtime/src/launched.rs`
- 1-file cycle: `vendor/cubecl-fix/cubecl-runtime/src/tune/operation.rs -> vendor/cubecl-fix/cubecl-runtime/src/tune/operation.rs`
- 1-file cycle: `vendor/cubecl-fix/cubecl-server/src/allocator.rs -> vendor/cubecl-fix/cubecl-server/src/allocator.rs`
- 2-file cycle: `crates/dormouse-train/src/cfg.rs -> crates/dormouse-train/src/lib.rs -> crates/dormouse-train/src/cfg.rs`

## Hyperedges (group relationships)
- **Fixed-depth loop replacing PonderNet halting (collapse -> honest CE)** — 0013_fixeddepth_lambda_collapse, 0013_fixeddepth_kl_direction_wrong, 0013_fixeddepth_fixed_depth_maxiter4, 0013_fixeddepth_void_train_ce [EXTRACTED 1.00]
- **The fused perf gate chain: smoke win -> flagship loss -> two-rung rewrite with a kill switch** — 0003_fusedmeasuredecide_smoke_verdict, research_20260923_fusedflagship50_flagship_ab, 0009_fusedrewriterungs_defect_causes, 0009_fusedrewriterungs_rung1, 0009_fusedrewriterungs_kill_switch [INFERRED 0.85]
- **The MSA arc: out-of-bounds gathers -> mechanism cut -> the top-k gather primitive that unblocks its re-entry** — 0012_msabrokenpre4_oob_gathers, 0014_msacut_msa_deleted, 0015_topkgather_argtopk_sentinel, 0015_topkgather_cubek_reduce_patch, 0015_topkgather_unblocks_three [INFERRED 0.85]
- **Fused cubecl CUDA kernel family (per-README fused kernel + measured speedup + fused backward claims)** — vendor_burn_fused_crates_burn_gdn2_readme_gdn2, vendor_burn_fused_crates_burn_kda_readme_kda, vendor_burn_fused_crates_burn_mhc_readme_mhc, vendor_burn_fused_crates_burn_muon_plus_readme_muon_plus, vendor_burn_fused_crates_burn_rope_readme_rope, vendor_burn_fused_crates_burn_sct_readme_sct, vendor_burn_fused_crates_burn_situ_readme_situ [INFERRED 0.95]
- **Looped / recursive model compute and stability control** — vendor_burn_fused_crates_burn_parcae_readme_parcae, vendor_burn_fused_crates_burn_ptrn_readme_ptrn, vendor_burn_fused_crates_burn_mor_readme_mor, vendor_burn_fused_crates_burn_mod_readme_mod [INFERRED 0.85]
- **Kimi K3 mechanism family (arXiv 2607.24653)** — vendor_burn_fused_crates_burn_kda_readme_kda, vendor_burn_fused_crates_burn_nope_readme_nope, vendor_burn_fused_crates_burn_situ_readme_situ [EXTRACTED 1.00]
- **TSCT: ternary spectral compact training (SVD reparameterization, fused CUDA kernel, 2-bit inference form)** — vendor_burn_fused_crates_burn_spectral_readme_burn_spectral, vendor_burn_fused_crates_burn_spectral_readme_tsct_reparameterization, vendor_burn_fused_crates_burn_spectral_readme_ternary_projection, vendor_burn_fused_crates_burn_spectral_readme_fused_cuda_kernel, vendor_burn_fused_crates_burn_spectral_readme_two_bit_inference_form [EXTRACTED 1.00]
- **CubeCL runtime toolkit: server provides pools, stream scheduler, driver helpers, compilation pipeline** — vendor_cubecl_fix_cubecl_server_readme_cubecl_server, vendor_cubecl_fix_cubecl_server_readme_memory_pools, vendor_cubecl_fix_cubecl_server_readme_stream_scheduler, vendor_cubecl_fix_cubecl_server_readme_driver_helpers, vendor_cubecl_fix_cubecl_server_readme_compilation_pipeline [EXTRACTED 1.00]
- **cubek-reduce test feature tiers: heavy < extended < full on the CUDA backend** — vendor_cubek_fix_cubek_reduce_readme_cubek_reduce, vendor_cubek_fix_cubek_reduce_readme_cuda_backend, vendor_cubek_fix_cubek_reduce_readme_heavy_feature, vendor_cubek_fix_cubek_reduce_readme_extended_feature, vendor_cubek_fix_cubek_reduce_readme_full_feature [EXTRACTED 1.00]
- **The 2026 convergent recipe tail: one mixed RL run, then multi-teacher on-policy distillation** — research_2026_09_27_posttraining_compare_intellect_3, research_2026_09_27_posttraining_compare_mimo_v2_6, research_2026_09_27_posttraining_compare_nemotron_3_ultra, research_2026_09_27_posttraining_compare_rufus_air [EXTRACTED 1.00]
- **dormouse minimal post-training ladder: SFT floor -> execution-grounded RLVR -> on-policy distillation -> DPO/merge** — research_2026_09_27_posttraining_compare_recommended_minimal_ladder, research_2026_09_27_posttraining_compare_nanochat_sft_stage, research_2026_09_27_posttraining_compare_stage1_execution_grounded_rlvr, research_2026_09_27_posttraining_compare_stage2_on_policy_distillation, research_2026_09_27_posttraining_compare_stage3_dpo_merge [EXTRACTED 1.00]
- **Rufus-Air's four load-bearing principles (copy these, not the 8-stage pipeline)** — research_2026_09_27_posttraining_compare_four_principles_worth_copying, research_2026_09_27_posttraining_compare_sft_floor_capability_not_warmup, research_2026_09_27_posttraining_compare_productive_difficulty_band, research_2026_09_27_posttraining_compare_reward_reliability_ordering, research_2026_09_27_posttraining_compare_infra_is_the_recipe [EXTRACTED 1.00]
- **Adaptive-depth mechanisms surveyed for looped/recurrent models** — research_2026_09_27_adaptive_depth_safe_mixture_of_recursions_2507_10524, research_2026_09_27_adaptive_depth_safe_cumulative_softmax_router_shape, research_2026_09_27_adaptive_depth_safe_recurtrace_2609_03379, research_2026_09_27_adaptive_depth_safe_calm_schuster_2022, research_2026_09_27_adaptive_depth_safe_think_shallow_solve_deep_2608_18222, research_2026_09_27_adaptive_depth_safe_layerskip_elhoushi_2024, research_2026_09_27_adaptive_depth_safe_think_at_hard_2511_08577 [INFERRED 0.85]
- **Independent replications of the halting collapse (PonderNet-style penalty on the loss mass)** — research_2026_09_27_adaptive_depth_safe_pondernet_halting_collapse, research_2026_09_27_adaptive_depth_safe_halting_collapse_replication, research_2026_09_27_adaptive_depth_safe_pondernet_2107_05407, research_2026_09_27_adaptive_depth_safe_pondernet_weighted_loss_mixture, research_2026_09_27_adaptive_depth_safe_adr_0013_fixed_depth [INFERRED 0.85]
- **Measurements of the peak-then-fall (overthinking) effect as depth grows** — research_2026_09_27_adaptive_depth_safe_fixed_depth_peak_then_fall, research_2026_09_27_adaptive_depth_safe_ouro_peak_then_collapse, research_2026_09_27_adaptive_depth_safe_huginn_3_5b_settling_measurements, research_2026_09_27_adaptive_depth_safe_loop_does_not_settle, research_2026_09_27_adaptive_depth_safe_extrapolation_ceiling_1_5x, research_2026_09_27_adaptive_depth_safe_underthinking_failure [INFERRED 0.85]

## Communities (393 total, 20 thin omitted)

### Community 0 - "CubeBackend"
Cohesion: 0.05
Nodes (16): FloatTensorOps, CubeBackend, BoolDType, BoolTensor, Distribution, ExecutionError, FloatDType, FloatTensor (+8 more)

### Community 1 - "CubeBackend"
Cohesion: 0.05
Nodes (26): IntTensorOps, random_bernoulli(), CubeDevice, CubeTensor, DType, Shape, random_normal(), CubeDevice (+18 more)

### Community 2 - "tensor.rs"
Cohesion: 0.04
Nodes (59): basicfloatunarykind, basicintunarykind, burn_backend, burn_cubecl_fusion, burn_fusion, burn_ir, burn_std, cast (+51 more)

### Community 3 - "Post-training 2026 comparison report (2026-09-27)"
Cohesion: 0.05
Nodes (93): Invariant: aborted / infrastructure failures are not reward-0, ATOD arXiv 2606.27814 (small SLM agents: ALFWorld / WebShop / Search-QA), Auditing GRPO/SFT/DPO arXiv 2609.00925 (3B), Bittensor Agent Arenas arXiv 2606.10064 (Qwen3-4B, ShoppingBench), Byte sequences are ~3-4x longer per content token than BPE, Nobody has published a byte-level post-training result at any scale, Scale gap: 7.5M is 16x below the smallest end-to-end SFT+RL pipeline, Byte models need a wire protocol: role delimiters on control bytes (+85 more)

### Community 4 - "fused_attnres.rs"
Cohesion: 0.06
Nodes (72): depth_attend, depth_attend_autodiff, refcell, main(), ref_depth_attend(), Tensor, attnres_backward_bench(), attnres_bench() (+64 more)

### Community 5 - "registry.rs"
Cohesion: 0.06
Nodes (60): elemwise, FallbackOperation, FusionHandle, FusionRuntime, nhwc_relayout, Optimization, OrderedExecution, reduce_broadcasted (+52 more)

### Community 6 - "CubeBackend"
Cohesion: 0.05
Nodes (45): BackendGraph, BackendTypes, DTypeUsageSet, MemoryPoolLayout, MemoryPoolUsage, ProfileOptions, ProfileToken, SlicedPoolReport (+37 more)

### Community 7 - "ReduceArgs"
Cohesion: 0.05
Nodes (35): r_virtual, SizeOut, Tag, Vectorized, P, Instant, State, ((In, SizeIn), (Out, SizeOut)) (+27 more)

### Community 8 - "numeric.rs"
Cohesion: 0.06
Nodes (65): CollectiveTensor, DistributedOps, CubeBackend, DeviceId, FloatTensor, ReduceOperation, Self, Vec (+57 more)

### Community 9 - "filter.rs"
Cohesion: 0.07
Nodes (48): Cli, Cursor<R>, DedupSet, doc_hash128(), Example, ft_hash(), ft_hash_matches_cpp(), FT_MAGIC (+40 more)

### Community 10 - "capture.rs"
Cohesion: 0.06
Nodes (43): handle, layout, BASE_DEALLOC_PERIOD, DEALLOC_SCALE_MB, EXCLUSIVE_MEMORY_ONLY, generate_bucket_sizes(), MemoryConfiguration, MemoryDeviceProperties (+35 more)

### Community 11 - "memory_manage.rs"
Cohesion: 0.11
Nodes (57): memorypooloptions, MemoryAllocationMode, alloc_allocs_new_storage(), alloc_respects_alignment_size(), alloc_reuses_storage(), alloc_two_chunks_on_one_page(), allocate_deallocate_reallocate(), allocs_on_correct_page() (+49 more)

### Community 12 - "reference_for_config()"
Cohesion: 0.05
Nodes (61): CategoryWork, reference_all, reference_any, reference_argmax, reference_argmin, reference_argtopk, reference_max, reference_max_abs (+53 more)

### Community 13 - "CudaServer"
Cohesion: 0.09
Nodes (25): CUstream_st, CudaServer, info_buffer(), pair(), Arc, Box, Bytes, CUstream (+17 more)

### Community 14 - "chunk.rs"
Cohesion: 0.07
Nodes (52): coding_rate_exact(), constant_stream_prefers_early_positions_outlier_wins(), dev(), exact_marginal_gains_sum_to_full_rate(), exact_rate_finite_on_rank_deficient(), exact_rate_matches_closed_form_orthonormal(), l2_gains_telescope_to_total_rate(), logdet_spd() (+44 more)

### Community 15 - "moe_fused.rs"
Cohesion: 0.09
Nodes (67): cube_int2(), cube_of1(), cube_of2(), dense(), empty_dense(), empty_dense_int(), forward_moe_fused(), launch_bwd_raw() (+59 more)

### Community 16 - "Accumulator"
Cohesion: 0.06
Nodes (35): reduce_scan(), reduce_tree(), Accumulator, ArgAccumulator<P>, fuse_accumulator_inplace(), reduce_inplace(), reduce_shared_inplace(), ReduceFamily (+27 more)

### Community 17 - "tsct_diag.rs"
Cohesion: 0.06
Nodes (53): dispatchtensor, gradientsparams, module_lr_scheduler, MutexGuard, asym_kind_shapes(), Attention, attention_shapes(), attn_flops() (+45 more)

### Community 18 - ".stream_id()"
Cohesion: 0.07
Nodes (21): Features, Re, Client, ProfileWindow, Arc, Box, DeviceId, DeviceProperties (+13 more)

### Community 19 - "Command<'a, D>"
Cohesion: 0.06
Nodes (41): a_collection_names_what_was_launched_while_it_was_open(), close(), COLLECTING, Collections, Launched, LaunchedKernels, note(), OPEN (+33 more)

### Community 20 - "loop_block.rs"
Cohesion: 0.05
Nodes (40): AtomicI32, config, AdaptiveAttention, Self, GatedResidual, GR_BRANCHES, GrState, Self (+32 more)

### Community 21 - "binary.rs"
Cohesion: 0.08
Nodes (43): AddOp, AndOp, AssignOp, BinaryMaxOp, BinaryMinOp, BinaryOp, BinaryOpFamily, DivOp (+35 more)

### Community 22 - "KernelId"
Cohesion: 0.06
Nodes (32): derive_more, ExecutionMode, H, Hasher, hashset, TypeId, DynKey, Info (+24 more)

### Community 23 - "rope_cuda.rs"
Cohesion: 0.07
Nodes (50): precompute_freqs, rope_autodiff, precompute_freqs(), precompute_freqs_yarn(), Tensor, apply_rope_4d_matches_3d(), dev(), freqs_identity_at_position_zero() (+42 more)

### Community 24 - "CubeTensor"
Cohesion: 0.07
Nodes (20): BufferArg, ir, LinearViewLaunch, tensorhandle, CubeTensor, AddressType, Box, Clone (+12 more)

### Community 25 - "comparison.rs"
Cohesion: 0.11
Nodes (46): ComparisonOp, ComparisonOpFamily, equal(), equal_elem(), EqualOp, greater(), greater_elem(), greater_equal() (+38 more)

### Community 26 - "ServerError"
Cohesion: 0.09
Nodes (21): CommunicationId, Bytes, D, DeviceId, DeviceService, DynFut, ElemType, Error (+13 more)

### Community 27 - "ReduceBlueprint"
Cohesion: 0.08
Nodes (47): clamp_plane_count, cube_count_spread_with_total, HardwareProperties, support_plane(), calculate_plane_count_per_cube(), BlueprintStrategy, ReduceLaunchSettings, ReduceProblem (+39 more)

### Community 28 - "CubeCount"
Cohesion: 0.08
Nodes (20): an_undeclared_resource_among_declared_ones_over_names(), buffer_io_drives_the_read_and_write_sets(), CubeCount, CubeCountSelection, CubeDim, declared_io_answers_when_the_compiled_kernel_kept_none(), Dim3, KernelArguments (+12 more)

### Community 29 - "device()"
Cohesion: 0.10
Nodes (50): aux_losses_and_ema_teacher(), bpb(), build_model(), bytes_to_tensors(), ckpt_save_load_roundtrip(), device(), ema_teacher_for(), every_opt_mode_steps() (+42 more)

### Community 30 - "im2col.rs"
Cohesion: 0.10
Nodes (49): cubek, empty_device_dtype, init_matmul_output, Iter, matmul_autotune, MatmulSetupError, batches_per_run(), check_pointwise() (+41 more)

### Community 31 - "DummyServer<M>"
Cohesion: 0.08
Nodes (24): cubecl_common, CubeKernel, ServerStorage, DummyServer<M>, KernelTask, Other, REFUSE_PROFILES, Arc (+16 more)

### Community 32 - "ServerUtilities"
Cohesion: 0.07
Nodes (35): fixedstate, GpuContext, itertools, Collective, cube_count_spread(), IoError, LaunchError, MemoryLayoutPolicy (+27 more)

### Community 33 - "integration_test.rs"
Cohesion: 0.08
Nodes (52): RecordLevel, streamid, DummyClient, test_client(), a_compilation_is_recorded_with_its_outcome(), a_compilation_keeps_its_code_only_when_records_are_full(), a_compile_nothing_stored_leaves_no_session(), a_dry_run_drops_an_ordinary_launch() (+44 more)

### Community 34 - "ManagedMemoryHandle"
Cohesion: 0.07
Nodes (23): Binding, MemoryHandle, Clone, Debug, a_location_survives_packing_at_every_extreme(), ManagedMemoryDescriptor, ManagedMemoryHandle, MemoryLocation (+15 more)

### Community 35 - "ReduceRequirements"
Cohesion: 0.05
Nodes (35): ReduceRequirements, fill_coordinate_vector(), new_coordinates(), N, Vector, ReaderBoundChecks, ComptimeOption, I (+27 more)

### Community 36 - "MhcBlock"
Cohesion: 0.08
Nodes (36): initializer, mhcblock, static_init, ALPHA_INIT, MhcBlock, Param, Self, Tensor (+28 more)

### Community 37 - "CubeElement"
Cohesion: 0.07
Nodes (44): Acc, CubeElem, MatmulPrecision, MatrixPrecision, bf16, BoolElement, CubeElement, f16 (+36 more)

### Community 38 - "CubeBackend"
Cohesion: 0.11
Nodes (16): BoolTensorOps, bool_store(), CubeBackend, BoolDType, BoolTensor, CubeTensor, ExecutionError, FloatDType (+8 more)

### Community 39 - "DirectPool"
Cohesion: 0.09
Nodes (17): calculate_padding(), MemoryPool, PageMapping, IoError, Result, Self, Storage, Slice (+9 more)

### Community 40 - "schema.rs"
Cohesion: 0.05
Nodes (17): actformat, act_quant_from_str_forms(), ActFormat, ActQuant, default_equals_small_preset(), partial_toml_fills_schema_defaults(), D, Deserialize (+9 more)

### Community 41 - "deform_conv_transpose2d.rs"
Cohesion: 0.09
Nodes (44): FAdd, FP, InputGradients, ProxyType, cast(), CubeTensor, DType, backward_gradient_inputs() (+36 more)

### Community 42 - "select()"
Cohesion: 0.06
Nodes (43): grid_sample_bilinear_launch, GridSamplePaddingMode, fetch_value(), fetch_with_border(), fetch_with_reflection(), fetch_with_zeros(), grid_sample(), PaddingMode (+35 more)

### Community 43 - "ErrorGraph"
Cohesion: 0.12
Nodes (29): box, NonZeroU64, ManagedMemoryId, a_failure_that_tainted_nothing_is_pruned(), a_node_lives_while_something_carries_it_and_no_longer(), a_report_walks_the_skip_chain_back_to_the_root(), Claim, error() (+21 more)

### Community 44 - "optim.rs"
Cohesion: 0.08
Nodes (35): burn, Default, PathBuf, TrainCfg, build_optim(), build_optim_mode(), effective_muon_markers(), engram_table_group() (+27 more)

### Community 45 - "CubeTensor"
Cohesion: 0.10
Nodes (44): tensor_vector_size_parallel, launch_binop_int(), launch_scalar_binop_int(), CubeTensor, InputScalar, launch(), launch_unary_int(), Args (+36 more)

### Community 46 - ".to_output_perpendicular()"
Cohesion: 0.07
Nodes (28): accumulatorformat, comptime, cube, plane_topk_insert, plane_topk_merge, reaches, ReduceOutputMode, clamp_coordinate() (+20 more)

### Community 47 - "compiler.rs"
Cohesion: 0.09
Nodes (29): AsRef, compiler, RecordEffect, records, SK, SV, build_id_hash(), compilation_store() (+21 more)

### Community 48 - "kernel.rs"
Cohesion: 0.06
Nodes (26): bufferioattr, cubecl_ir, ElemType, Option, SourceKernel<K>, KernelArg, KernelMetadata, PrecompiledSource (+18 more)

### Community 49 - "CubeBackend"
Cohesion: 0.12
Nodes (16): DeformConv2dBackward, MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps, CubeBackend, AttentionModuleOptions, BoolTensor, ConvOptions (+8 more)

### Community 50 - "schedule.rs"
Cohesion: 0.12
Nodes (29): a_candidate_that_failed_late_is_disqualified_despite_good_samples(), a_noisy_candidate_is_judged_on_its_best_run(), a_sampled_candidate_reports_its_measurements(), an_eliminated_candidate_never_wins_on_its_shorter_sample_set(), BatchOutcome, Candidate, eliminates_a_candidate_that_is_clearly_behind(), keeps_a_candidate_within_the_speed_factor() (+21 more)

### Community 51 - "shape_divmod()"
Cohesion: 0.06
Nodes (42): address_type, SequenceArg, conv_transpose2d_direct_kernel(), ConvArgs, ComptimeOption, E, ElemType, FastDivmod (+34 more)

### Community 52 - ".group()"
Cohesion: 0.12
Nodes (35): AtomicU32, PriorityFunc, GROUP_COUNTER, Arc, Clone, Debug, F, Fn (+27 more)

### Community 53 - "gather_nd.rs"
Cohesion: 0.06
Nodes (20): backward_data, base, capture, conv_transpose2d, cubetensor, event, scheme, strategy (+12 more)

### Community 54 - "ManagedMemoryBinding"
Cohesion: 0.11
Nodes (15): Reclaim, ManagedMemoryBinding, Vec, SharedMemoryBindings, DynamicPool, MemoryManagement<Storage>, Debug, Display (+7 more)

### Community 55 - "event.rs"
Cohesion: 0.12
Nodes (31): SyncSender, EventStreamBackendWrapper, GcTask, GcThread, GcThread<B>, handle(), MAX_STREAMS, MultiStream (+23 more)

### Community 56 - "serve.rs"
Cohesion: 0.10
Nodes (35): axum, CacheConfig, AppState, Args, chat_completions(), ChatChoice, ChatMsg, ChatReq (+27 more)

### Community 57 - "driver.rs"
Cohesion: 0.08
Nodes (29): backtrace, computestorage, gpu, HostResource, copy_failed(), Cuda, Bytes, c_void (+21 more)

### Community 58 - "CudaContext"
Cohesion: 0.13
Nodes (27): c_char, CUctx_st, CUfunc_st, format_cpp, install, cache_namespace(), CudaCompiledKernel, CudaContext (+19 more)

### Community 59 - "markov.rs"
Cohesion: 0.13
Nodes (25): sample_tokens, dev(), gated_head_shapes(), GatedMarkovHead, greedy_draft(), rnn_head_forward(), rnn_sample_shape(), RNNHead (+17 more)

### Community 60 - "unary_float()"
Cohesion: 0.08
Nodes (36): clamp_float(), clamp_int(), Options, CubeTensor, InputScalar, BasicFloatUnary, FloatUnaryOp, FloatUnaryOpFamily (+28 more)

### Community 61 - "ServerLogger"
Cohesion: 0.08
Nodes (22): channel, Receiver, Sender, spawn_detached, Profiled, ProfileItem, ProfileLevel, Display (+14 more)

### Community 62 - "offload.rs"
Cohesion: 0.09
Nodes (28): collect(), gunzip_to_vec(), main(), read_corpus(), Path, PathBuf, Result, Vec (+20 more)

### Community 63 - "burn-fused"
Cohesion: 0.10
Nodes (40): burn-antihall, burn-attnres, burn-bitnet, burn-byteflow, burn-cubecl, burn-diffusionblocks, burn-dspark, burn-eggroll (+32 more)

### Community 64 - "SctLinear"
Cohesion: 0.11
Nodes (26): bytes_f32(), dev(), forward_shape(), from_dense_roundtrip(), from_dense_shape(), JACOBI_EPS, matmul_rt(), ortho_error_decreases() (+18 more)

### Community 65 - "fused_recurrent_cube.rs"
Cohesion: 0.08
Nodes (28): any, components, prelude, super, cube_of(), fused_step(), gdn2_step_kernel(), CubeTensor (+20 more)

### Community 66 - "tuner.rs"
Cohesion: 0.14
Nodes (31): autotuneloglevel, TuneCacheResult, check_autotune_outputs(), check_equivalence(), execute_checks(), is_decisions_enabled(), PendingBench, process_request() (+23 more)

### Community 67 - "gen_reference.py"
Cohesion: 0.08
Nodes (23): fla_ops_kda, math, struct, time, torch, torch_nn, torch_nn_functional, bench() (+15 more)

### Community 68 - "HallSuppressor"
Cohesion: 0.08
Nodes (20): module, nn, HallSuppressor, Linear, Option, Param, Self, Tensor (+12 more)

### Community 69 - "TestCase"
Cohesion: 0.10
Nodes (24): TestOutcome, reference_argtopk(), HostData, Option, Progress, reference_topk(), HostData, Option (+16 more)

### Community 70 - "collective.rs"
Cohesion: 0.12
Nodes (24): a_rank_does_not_depend_on_the_order_the_group_was_given_in(), CollectiveDriver, Collectives, Collectives<D>, device(), peer_of(), rank_in(), CommStream (+16 more)

### Community 71 - "DummyKernel"
Cohesion: 0.07
Nodes (17): cubecl_server, new, BytesResource, DummyElementwiseAddition, DummyKernel, CompilationError, Debug, Option (+9 more)

### Community 72 - "CudaCompiler"
Cohesion: 0.08
Nodes (26): ComputeKernel, CppCompiler, KernelSettings, NvptxModule, PlironCompiler, SmArch, CudaBackend, CudaCompilationOptions (+18 more)

### Community 73 - "AutotuneResult"
Cohesion: 0.11
Nodes (25): Cow, AutotuneDecision, AutotuneLogContext, AutotuneLogEvent, AutotuneLoggerExt, CheckResult, log_result(), Display (+17 more)

### Community 74 - "benchmarker.rs"
Cohesion: 0.13
Nodes (27): mutex, a_clock_that_lifts_inside_the_floor_does_not_release_the_warmup(), a_device_slow_for_the_whole_measurement_reports_its_slow_rate(), a_pass_far_under_the_target_grows_until_it_reaches_it(), a_steady_device_pays_the_floor_and_nothing_more(), a_timer_reading_zero_does_not_climb_to_the_duration_ceiling(), a_timer_reading_zero_still_stops_sampling(), a_timer_that_never_reaches_the_target_stops_growing_on_the_budget() (+19 more)

### Community 75 - "ReduceError"
Cohesion: 0.15
Nodes (34): permute, accumulator_len(), argsort(), empty_reduce_identity(), fold_leading_dims(), init_reduce_output(), init_reduce_output_dtype(), KernelReduceStrategy (+26 more)

### Community 76 - "profiler.rs"
Cohesion: 0.11
Nodes (26): ProfileTicks, Anchor, Anchor<A>, ANCHOR_MAX_AGE, Anchoring, EventProfiler, EventProfiler<A>, Open (+18 more)

### Community 77 - "sparse.rs"
Cohesion: 0.15
Nodes (32): activation_bits_4_uses_hadamard_and_stays_close_to_8bit(), bf16_forward_matches_fp32(), bf16_gradients_match_fp32(), compute_nm_mask(), dev(), dual_ste_gradient_flows_to_masked_weights(), four_bit_forward_cuda_finite_and_bf16(), fused_masked_quant_forward_and_dual_ste() (+24 more)

### Community 78 - "MemoryPage"
Cohesion: 0.12
Nodes (19): MB, MemoryBlock, MemoryJob, MemoryPage, MemoryPageSummary, MemoryTask, MemoryTaskStatus, new_memory_page() (+11 more)

### Community 79 - "NativeExpand"
Cohesion: 0.14
Nodes (14): ComptimeOptionExpand, NativeExpand, SizeIn, SliceExpand, TensorMap, Tiled, VectorizedExpand, In (+6 more)

### Community 80 - "route_block()"
Cohesion: 0.10
Nodes (26): modpredictor, routing, bce_loss_penalizes_mismatch(), dev(), ModConfig, predictor_learns_targets(), route_block_residual_and_scaling(), Default (+18 more)

### Community 81 - "ReduceStrategy"
Cohesion: 0.11
Nodes (30): routines, any_all_precision_keeps_accumulation_narrow(), ElemType, Option, launch_fused(), launch_reduce(), launch_reduce_with_indices(), prepare_reduce_launch() (+22 more)

### Community 82 - "SchedulerMultiStream<B>"
Cohesion: 0.13
Nodes (19): alloc, Schedule, Task, Arc, B, IntoIter, Iterator, Self (+11 more)

### Community 83 - "reference_mean()"
Cohesion: 0.08
Nodes (28): compilationarg, contiguous_strides, cubek_test_utils, plane, shape, reference_mean(), HostData, Option (+20 more)

### Community 84 - "nan_extrema.rs"
Cohesion: 0.17
Nodes (34): cubestrategy, case(), cube_f32(), cube_with_planes_f32(), integer_extrema_control_i32(), integer_extrema_data(), mixed_nan_extrema_data(), nan_extrema_data() (+26 more)

### Community 85 - "server.rs"
Cohesion: 0.10
Nodes (33): CUtensorMap, CUtensorMapDataType, CUtensorMapFloatOOBfill, CUtensorMapInterleave, CUtensorMapL2promotion, CUtensorMapSwizzle, future, check_tma_generic() (+25 more)

### Community 86 - "routing.rs"
Cohesion: 0.11
Nodes (28): gather_active, load_balancing_loss, sigmoid, topk_indices, dev(), gather_scatter_roundtrip(), load_balancing_loss_equals_scatter_mask(), load_balancing_loss_finite_nonneg() (+20 more)

### Community 87 - "WgpuDevice"
Cohesion: 0.10
Nodes (20): amddevice, cpudevice, cudadevice, deviceid, metaldevice, spectrallinear, CUDA_MAX_BINDINGS, a_device_round_trips_its_kind_and_its_backend() (+12 more)

### Community 88 - "chunk_wy_forward()"
Cohesion: 0.10
Nodes (29): autodiff, burn_gdn2, fused_chunk_forward, chunk_wy_forward(), ATOL, chunk64_strong_decay_stays_finite(), fused_forward_matches_plain(), fused_grads_match_finite_difference() (+21 more)

### Community 89 - "PersistentPool"
Cohesion: 0.10
Nodes (17): bytesstorage, calculate_padding, memory_management, Vec, persistent_pool(), persistent_pool_try_reserve_reuses_slice_with_padding(), PersistentPool, Display (+9 more)

### Community 90 - "conv_data_backward_fallback()"
Cohesion: 0.08
Nodes (31): conv_transpose2d_autotune, conv_data_backward_fallback(), conv_transpose1d_from_conv_transpose2d(), ConvOptions, ConvSetupError, ConvTransposeOptions, CubeTensor, N_DIM (+23 more)

### Community 91 - "ReduceOperationConfig"
Cohesion: 0.13
Nodes (30): InputsWithIndices, reduce, folds(), CubeTensor, Inputs, Out, with_reduce_bounds(), with_reduce_with_indices_bounds() (+22 more)

### Community 92 - "PinnedMemoryStorage"
Cohesion: 0.08
Nodes (17): storage, PinnedMemory, PinnedMemoryStorage, c_void, HashMap, IoError, Resource, Result (+9 more)

### Community 93 - "MemoryLayout"
Cohesion: 0.11
Nodes (19): Shape, MemoryLayout, MemoryLayoutDescriptor, Strides, contiguous_strides(), ContiguousMemoryLayoutPolicy, height_counts_every_row_not_only_the_last_dimension(), offset_handles() (+11 more)

### Community 94 - "fwt_cuda.rs"
Cohesion: 0.14
Nodes (30): ad, bitnet_bench(), cube_of(), cube_of_1(), fwt_autodiff(), fwt_backward_cuda(), fwt_cuda(), fwt_fused_backward_matches_tensor() (+22 more)

### Community 95 - "sampler.rs"
Cohesion: 0.09
Nodes (17): Benchmark, cubecl_runtime, a_real_improvement_resets_convergence(), aging_out_the_biased_sample_does_not_count_as_a_stall(), computation_is_built_from_the_reliable_samples_only(), CONVERGENCE_EPSILON, CONVERGENCE_ROUNDS, converges_after_consecutive_non_improving_samples() (+9 more)

### Community 96 - "noise.rs"
Cohesion: 0.11
Nodes (22): blockpartition, consts, DEFAULT_P_MEAN, DEFAULT_P_STD, DEFAULT_SIGMA_DATA, DEFAULT_SIGMA_MAX, DEFAULT_SIGMA_MIN, dev() (+14 more)

### Community 97 - "execute_interpolate()"
Cohesion: 0.11
Nodes (30): CubekInterpolateMode, CubekInterpolateOptions, CubekInterpolateStrategy, interpolate, interpolate_autotune, InterpolateAutotuneKey, InterpolateError, InterpolateForwardProblem (+22 more)

### Community 98 - "slice.rs"
Cohesion: 0.14
Nodes (30): fft, conv_weight_backward_fallback(), conv_weight_grad_depthwise(), conv_weight_grad_groups(), conv_weight_grad_no_groups(), ConvOptions, ConvSetupError, CubeTensor (+22 more)

### Community 99 - "fused_situ.rs"
Cohesion: 0.14
Nodes (31): situ_glu_autodiff, cube_of(), cuda_enabled(), fused_backward_matches_tensor_backward(), fused_forward_autodiff_matches(), maxdiff(), N_PARENTS, B (+23 more)

### Community 100 - "AutotuneError"
Cohesion: 0.12
Nodes (27): TuneDelegate, I, Out, Result, TuneFn, TuneFn<I, Out>, (), AutotuneOutput (+19 more)

### Community 101 - "contiguous.rs"
Cohesion: 0.12
Nodes (31): a_packed_weight_computes_the_same_product(), a_row_kernel_refuses_a_packed_tensor(), assert_close(), into_contiguous(), into_contiguous_quantized(), layout_rewrites_untile_first(), packed(), CubeDevice (+23 more)

### Community 102 - "BufferBinding"
Cohesion: 0.09
Nodes (14): BufferBinding, Option, Range, StreamMemory, Debug, Option, StreamWrapper<B>, TestStream (+6 more)

### Community 103 - "bounds_generator.rs"
Cohesion: 0.11
Nodes (21): an_unbounded_threshold_gives_no_time_limit(), AutotuneBound, bound(), Bounds, BoundsGenerator, Func, A, At (+13 more)

### Community 104 - "TunableSet"
Cohesion: 0.13
Nodes (16): KeyGenerator, Send, Sync, Arc, At, Box, Evictor, F (+8 more)

### Community 105 - "EventStreamBackend"
Cohesion: 0.10
Nodes (9): EventStreamBackend, GatedBackend, ResolvedStreams<'a, B>, Drop, Formatter, Result, ServerError, Stream (+1 more)

### Community 106 - "plane_topk_insert()"
Cohesion: 0.17
Nodes (23): ArgAccumulator, plane_topk_insert(), plane_topk_insert_values(), plane_topk_insert_with_coords(), plane_topk_merge(), plane_topk_merge_values(), plane_topk_merge_with_coords(), reaches() (+15 more)

### Community 107 - "jepa_aux_loss_masked()"
Cohesion: 0.09
Nodes (27): burn_dspark, burn_jepa, dspark_aux_loss(), DSPARK_GAMMA, ema_update(), EmaMapper, jepa_aux_loss(), jepa_aux_loss_masked() (+19 more)

### Community 108 - "ByteStream"
Cohesion: 0.17
Nodes (18): ByteStream, collect_files(), empty_file_list_is_loud(), fnv(), missing_root_is_loud(), ORDERS, raw_hashes_are_unreduced_and_reduction_is_the_callers(), read_bytes() (+10 more)

### Community 109 - "TensorHandle"
Cohesion: 0.11
Nodes (19): FusionBackend, CubeBackend, CubeFusionHandle, FallbackOperationWrapper, FallbackOperationWrapper<O>, into_tensor(), BackendIr, BoolTensor (+11 more)

### Community 110 - "compute.rs"
Cohesion: 0.08
Nodes (19): memorydeviceproperties, server, I, Vec, uninit_vec(), MB, DummyDevice, DummyRuntime (+11 more)

### Community 111 - "CopyDescriptor"
Cohesion: 0.19
Nodes (11): Bytes, DynFut, Future, IntoIterator, Resource, Result, Send, ServerError (+3 more)

### Community 112 - "attention()"
Cohesion: 0.10
Nodes (28): attention_autotune, AttentionAutotuneKey, AttentionCost, AttentionSetupError, AttentionTunables, BlackboxAcceleratedStrategy, dtype, attention() (+20 more)

### Community 113 - "problem.rs"
Cohesion: 0.12
Nodes (21): Coordinates, numericvector, build_reduce_output_layout(), ReduceOutputLayout, Coords1d, Layout, N, ReadWrite (+13 more)

### Community 114 - "zero_copy()"
Cohesion: 0.11
Nodes (28): COMPUTE_32F, COMPUTE_32F_FAST_16BF, COMPUTE_32F_FAST_16F, COMPUTE_32F_FAST_TF32, compute_name(), enter_primary_context(), err(), GEMM_DEFAULT (+20 more)

### Community 115 - "communication.rs"
Cohesion: 0.13
Nodes (23): cubecl_core, maybeuninit, ncclDataType_t, ncclRedOp_t, ncclUniqueId, Cuda, nccl_dtype_count(), CommStream (+15 more)

### Community 116 - "detect.rs"
Cohesion: 0.12
Nodes (24): ActivationOps, hallsuppressor, CubeBackend, Self, HallDetector, intervene(), Linear, Self (+16 more)

### Community 117 - "ReduceOperation"
Cohesion: 0.12
Nodes (15): instructions, DynamicAccumulator, DynamicSharedAccumulator, DynamicSharedAccumulator<P>, ReduceOperation, Config, EA, EI (+7 more)

### Community 118 - "tune_cache.rs"
Cohesion: 0.11
Nodes (25): persistence, storeerror, Answer, OpenRecording, Duration, K, OpenRecording, Option (+17 more)

### Community 119 - "taint.rs"
Cohesion: 0.15
Nodes (21): Claims, compilationerror, Ranges, smallvec, tostring, a_new_failure_takes_the_bytes_it_claims(), a_partial_write_releases_only_the_bytes_it_covers(), add() (+13 more)

### Community 120 - "to_vec()"
Cohesion: 0.15
Nodes (22): perturb, dev(), eggroll_mutate_roundtrip(), perturb_rank1_variance_bounded(), perturb_scales_with_sigma(), Tensor, Vec, sample_a_shape_and_finite() (+14 more)

### Community 121 - "AcceptRatePredictor"
Cohesion: 0.16
Nodes (23): sampling, accept_rate_loss(), accept_rate_predictor_hidden_only(), accept_rate_predictor_markov_conditioned(), accept_rate_predictor_rejects_missing_markov_embeddings(), accept_rate_target(), accept_rate_target_bounds(), AcceptRatePredictor (+15 more)

### Community 122 - "metadata_cache.rs"
Cohesion: 0.21
Nodes (15): cache(), capture_discard_unpins_without_removing(), capture_hit_on_normal_entry_pins_it(), capture_mode_caches_any_size(), capture_mode_is_unbounded(), clear_unpinned_keeps_only_graph_pinned_entries(), hit_returns_value_and_records_use(), key() (+7 more)

### Community 123 - ".launching()"
Cohesion: 0.16
Nodes (19): ExecuteScope, ExecuteScope<'a, S>, failed_writing(), Opened, FnOnce, Iterator, Option, R (+11 more)

### Community 124 - "JepaTargets"
Cohesion: 0.10
Nodes (18): BufWriter, Self, fnv(), main(), usage(), JepaTargets, JepaTargetWriter, BufReader (+10 more)

### Community 125 - "SpectralLinear"
Cohesion: 0.10
Nodes (10): qr_householder(), QuantFormat, Self, Tensor, SpectralLinear, ste_ternary_annealed(), ste_ternary_per_column(), ste_ternary_stochastic() (+2 more)

### Community 126 - "as_scalar()"
Cohesion: 0.15
Nodes (26): ematarget, mask_indices, as_scalar(), as_vec(), count_true(), dev(), ema_momentum_decays(), ema_update_tracks_student() (+18 more)

### Community 127 - "mask_fill_kernel()"
Cohesion: 0.08
Nodes (25): mask_fill(), mask_fill_kernel(), MaskFillStrategy, B, CubeTensor, DType, ElemType, InputScalar (+17 more)

### Community 128 - "as_scalar()"
Cohesion: 0.16
Nodes (23): as_scalar(), best_of_k_picks_max(), best_of_k_sampled_stays_in_range(), best_of_k_sampled_tau_zero_is_argmax(), correctness_target_marks_equal(), dev(), noise_adds_variance(), noise_is_finite() (+15 more)

### Community 129 - "curve.rs"
Cohesion: 0.11
Nodes (16): ThroughputConfig, a_sweep_size_lands_on_the_measured_grid(), ceiling_clamps_outside_the_sweep(), ceiling_interpolates_between_measured_points(), curve(), log2(), MB, MemoryCurve (+8 more)

### Community 130 - "min_insert()"
Cohesion: 0.14
Nodes (16): ReduceWithIndices, Min, min_advance(), min_finalize_with_coords(), min_insert(), plane_min_candidate(), Config, EA (+8 more)

### Community 131 - "argtopk_shared_memory.rs"
Cohesion: 0.10
Nodes (19): autotunelevel, cubek_reduce, Launch, testcase, vectorizationstrategy, CatalogEntry, Vec, strategies() (+11 more)

### Community 132 - "ReduceCost"
Cohesion: 0.12
Nodes (17): Blueprint, a_top_k_insertion_costs_three_ops_a_slot(), an_axis_of_one_costs_nothing_to_fold(), an_empty_axis_costs_nothing_to_fold(), comparing_costs_an_op_more_than_accumulating(), cost(), counts_a_coordinate_output_like_any_other_value(), f32_dtypes() (+9 more)

### Community 133 - "self_checks.rs"
Cohesion: 0.13
Nodes (24): burn_muon_plus, linearconfig, modulelearningrate, optimizer, add_noise(), blockwise_step(), blockwise_step_backward(), denoising_loss() (+16 more)

### Community 134 - "DormouseConfig"
Cohesion: 0.13
Nodes (24): apply(), candidates(), is_explicit(), load_by_name_uses_builtin_or_file(), load_config(), load_explicit_path(), nano_fused_file_parses(), parse_str() (+16 more)

### Community 135 - "model_seam.rs"
Cohesion: 0.26
Nodes (23): assert_all_finite(), aux_heads(), batch_bytes(), determinism(), device(), engram_addressing_path_receives_gradient(), engram_host_rows(), forward_smoke() (+15 more)

### Community 136 - "logger.rs"
Cohesion: 0.11
Nodes (15): display, L, LoggerSinks, Logger, register_enabled(), Arc, Default, LoggerConfig (+7 more)

### Community 137 - "extrema.rs"
Cohesion: 0.25
Nodes (25): lowest_coordinate_matching, advance_argmax(), advance_argmin(), max_identity(), min_identity(), numeric_is_nan(), plane_argmax_propagating_nan(), plane_argmin_propagating_nan() (+17 more)

### Community 138 - "empty_device_dtype()"
Cohesion: 0.23
Nodes (26): pool, PoolError, into_contiguous_aligned(), conv_direct(), ConvOptions, ConvSetupError, CubeTensor, N (+18 more)

### Community 139 - "binary_int.rs"
Cohesion: 0.23
Nodes (19): BinaryOpInt, BinaryOpIntFamily, BitwiseAndOp, BitwiseOrOp, BitwiseShlOp, BitwiseShrOp, BitwiseXorOp, kernel_binop_int() (+11 more)

### Community 140 - ".forward()"
Cohesion: 0.15
Nodes (21): asym_forward_uses_raw_v(), dev(), nm_asym_keeps_v_raw(), nm_dual_ste_grad_flows_to_masked_entries(), nm_forward_is_sparse_and_matches_reference(), nm_off_is_bit_identical_to_plain_ternary(), polar_retracts(), ste_backward_flows_through_master() (+13 more)

### Community 141 - "handle.rs"
Cohesion: 0.11
Nodes (14): Handle, KernelResource, Clone, Debug, Formatter, Result, Self, ServiceId (+6 more)

### Community 142 - "ExclusiveMemoryPool"
Cohesion: 0.13
Nodes (13): ALLOC_AFTER_FREE, ExclusiveMemoryPool, MemoryPage, Display, Formatter, IoError, MemoryPage, Option (+5 more)

### Community 143 - "max_insert()"
Cohesion: 0.15
Nodes (15): Max, max_advance(), max_finalize_with_coords(), max_insert(), plane_max_candidate(), Config, EA, EI (+7 more)

### Community 144 - "shared_sum_kernel()"
Cohesion: 0.11
Nodes (23): is_dense(), Atomic, ElemType, LinearView, N, Result, Shared, T (+15 more)

### Community 145 - "alloc_probe.rs"
Cohesion: 0.22
Nodes (24): balancedcheckpointing, alloc_balanced_backend(), alloc_bare_forward(), alloc_fused_node(), alloc_tensor_path_nc(), b(), chunk(), client() (+16 more)

### Community 146 - "LazyDeviceController"
Cohesion: 0.14
Nodes (15): format, has_contiguous_row_major_strides, Once, LazyDeviceController, AccessError, AccessPolicy, AllocationController, AllocationProperty (+7 more)

### Community 147 - "bench_cuda.rs"
Cohesion: 0.10
Nodes (19): fused_step, bench_cuda(), cfg(), fused_kernel_matches_tensor_path(), FnMut, Option, time_it(), load4() (+11 more)

### Community 148 - "benchmark.rs"
Cohesion: 0.12
Nodes (17): inputorder, Input, ascending_rises_along_the_reduce_axis(), bench(), descending_falls_along_the_reduce_axis(), every_value_in_a_row_is_distinct_at_the_benchmark_shape(), InputOrder, ReduceBench (+9 more)

### Community 149 - "sinkhorn_cuda.rs"
Cohesion: 0.14
Nodes (23): sinkhorn_autodiff, maxdiff(), B, Backward, Checkpointer, F, Gradients, Ops (+15 more)

### Community 150 - "EngramModule"
Cohesion: 0.17
Nodes (19): compute_gate(), depthwise_conv_1d(), dev(), engram_module_multi_head(), engram_module_shape(), EngramModule, gate_in_range(), gate_matches_reference_formula() (+11 more)

### Community 151 - "SpectralRetention"
Cohesion: 0.20
Nodes (18): default_init_is_contractive(), dev(), forward_contracts(), forward_shape(), full_b_matches_diagonal(), injection_scale_shape(), retention_decreases_with_a(), retention_in_unit_interval() (+10 more)

### Community 152 - "burn-spectral (Ternary Spectral Compact Training)"
Cohesion: 0.09
Nodes (26): Attention is not spectralized (Amdahl ceiling 1.6-3.3x at d=64), BitNet b1.58 absmean STE (arXiv 2504.12285), burn 0.22 / cubecl 0.11 build stack, burn-spectral (Ternary Spectral Compact Training), Expert-Choice Routing (arXiv 2202.09368), Fused CUDA training kernel for SpectralLinear, TSCT measured results (RTX 5060 Ti, 3 seeds d=64; 1 seed d=256), Newton-Schulz polar retraction of the masters (retract(3)) (+18 more)

### Community 153 - "StreamPool"
Cohesion: 0.20
Nodes (14): F, Iterator, Option, Self, Stream, StreamId, Vec, stream_index() (+6 more)

### Community 154 - "Capability per byte and per GB for small LMs: research sur"
Cohesion: 0.13
Nodes (25): The long gate: BPB at distance 512k-1M vs 1-4k, Knife via A/B: PonderNet halt head, KoLeo, fp8-forward-by-default, batch warmup, Never: output-side lookup tables, expert-choice routing, sub-4-bit training, Pending one A/B each: MSA at s512, TSCT vs plain experts, energy head, Keep list: weight-shared LoopBlock, input-side Engram, Muon+, JEPA 0.05, DSpark, 3:1 blend, bf16 storage, MoR-style routers: the only sanctioned adaptive-depth re-entry path (rejected for now), Byte models lose at small compute, close at scale, win on robustness, Data filtering is the largest documented lever at fixed compute (+17 more)

### Community 155 - "module.rs"
Cohesion: 0.17
Nodes (16): autodiffbackend, ChunkFn, ConvCache, fused_recurrent_gdn2, l2_normalize_4d, chunk_wy_dispatch(), FUSED_RECURRENT_MAX_SEQ, GatedDeltaNet2 (+8 more)

### Community 156 - "ProfilingToken"
Cohesion: 0.10
Nodes (18): grad_norm(), Gradients, profile, ProfilingToken, HashMap, ProfileDuration, Result, ServerError (+10 more)

### Community 157 - "GpuStorage"
Cohesion: 0.13
Nodes (15): drivererror, uninit_vec, AllocationKind, GpuStorage, PtrBindings, CUdeviceptr, CUstream, Debug (+7 more)

### Community 158 - "taint_property.rs"
Cohesion: 0.19
Nodes (21): kernelid, a_clean_scope_hands_the_window_its_write_set(), a_failed_scope_dooms_the_recording_window(), a_mid_launch_panic_leaves_the_write_set_tainted(), a_partial_host_write_releases_only_the_bytes_it_covers(), a_scope_outside_a_window_neither_dooms_nor_records(), a_scope_that_fails_names_the_real_error_and_logs_it(), a_scope_that_succeeds_releases_the_provisional_failure() (+13 more)

### Community 159 - "observer.rs"
Cohesion: 0.17
Nodes (19): profiling, a_measurement_is_read_back_for_an_observer_that_only_wants_durations(), an_observation_restores_the_one_it_replaced(), an_observation_that_ended_mid_read_is_not_told_the_duration(), an_observation_that_ended_mid_read_is_not_told_the_loggers_reading(), an_observer_can_keep_a_measurement_unread(), launches_arrive_in_issue_order(), measured() (+11 more)

### Community 160 - "Failures"
Cohesion: 0.11
Nodes (11): StreamFactory, EventStreamBackendWrapper<B>, MultiStream<B>, Failures, Arc, Self, SchedulerPoolMarker<B>, Factory (+3 more)

### Community 161 - ".launch()"
Cohesion: 0.17
Nodes (10): a_read_returns_bytes_iff_their_last_writer_succeeded_exclusive_pages(), a_read_returns_bytes_iff_their_last_writer_succeeded_subslices(), Buffer, Harness, Rng, Option, Range, StreamId (+2 more)

### Community 162 - "local.rs"
Cohesion: 0.14
Nodes (18): AK, autotuneloggerext, ID, local_tuner, Sets, sync, LocalTuner, LocalTuner<AK, ID> (+10 more)

### Community 163 - "queue.rs"
Cohesion: 0.28
Nodes (18): Cell, each_subsequent_flush_syncs_the_previous_fence(), first_flush_creates_fence_without_syncing(), flush_on_empty_queue_is_safe(), make_queue(), MockFence, pending_holds_previously_staged_bytes_after_flush(), pending_is_replaced_on_second_flush() (+10 more)

### Community 164 - "DormouseModel"
Cohesion: 0.17
Nodes (8): AuxHeads, DormouseModel, Embedding, Int, Option, Self, Tensor, Vec

### Community 165 - "bit_exact.rs"
Cohesion: 0.21
Nodes (23): Cursor, CFGS, bench_model(), BENCH_MODELS, bench_ndarray_short(), bench_ndarray_single(), BenchCfg, EPSILON (+15 more)

### Community 166 - "MuonPlus"
Cohesion: 0.19
Nodes (15): MUON_NORM_DIR, ModuleOptimizer, MuonPlus, MuonPlusConfig, MuonPlusState, NormDir, ns_bench(), NS_COEFFS (+7 more)

### Community 167 - "activations.rs"
Cohesion: 0.18
Nodes (20): fast_walsh_hadamard, dev(), fast_walsh_hadamard(), fast_walsh_hadamard_tensor(), fast_walsh_hadamard_tensor_body(), fwt_reference_matches_butterfly(), fwt_reference_naive(), hadamard_non_power_of_two() (+12 more)

### Community 168 - "CudaStreamBackend"
Cohesion: 0.13
Nodes (13): pinned_memory_alignment, std, create_cuda_stream(), CudaStreamBackend, Arc, CUstream, MemoryConfiguration, MemoryDeviceProperties (+5 more)

### Community 169 - "Category"
Cohesion: 0.11
Nodes (15): Problem, reducecorrectness, reducecost, reducedtypes, bench.sh script, strategies, Category, CatalogEntry (+7 more)

### Community 170 - "KdaConfig"
Cohesion: 0.23
Nodes (20): beta_is_data_dependent(), beta_stays_in_unit_interval(), cfg(), chunk64_matches_decode(), chunk_matches_decode(), chunked_decode_matches_recurrent(), dev(), forward_train_state_chunked_matches_full() (+12 more)

### Community 171 - "fused_kernels.rs"
Cohesion: 0.19
Nodes (23): cube_of(), cuda_dev(), finalize_cuda(), finalize_kernel(), full_fused(), fused_match_tensor(), momentum_cuda(), momentum_kernel() (+15 more)

### Community 172 - "SpectralMoE"
Cohesion: 0.13
Nodes (7): Int, Linear, Param, SpectralMoE, topk_indices_descending_matches_naive(), tsct_fused_env_enabled(), tsct_moe_ec_matches_reference()

### Community 173 - "MemoryUsage"
Cohesion: 0.11
Nodes (14): bytes_format(), BytesFormat, MemoryPoolKind, MemoryPoolReport, MemoryRecord, .KIND, MemoryReport, MemoryUsage (+6 more)

### Community 174 - "Compiler"
Cohesion: 0.11
Nodes (17): CompilationError, Compiler, BackTrace, BufferIOAttr, Clone, CompilationOptions, Debug, Error (+9 more)

### Community 175 - "SlicedPool"
Cohesion: 0.18
Nodes (10): Display, Formatter, IoError, MemoryPage, Option, Result, Self, Storage (+2 more)

### Community 176 - "BytesStorage"
Cohesion: 0.17
Nodes (17): AllocatedBytes, BytesStorage, .ALIGNMENT, Debug, Formatter, HashMap, IoError, Layout (+9 more)

### Community 177 - "with_bounds()"
Cohesion: 0.13
Nodes (18): arc, duration, configured_thresholds(), no_bounds(), Fn, I, K, Option (+10 more)

### Community 178 - "Community 178"
Cohesion: 0.09
Nodes (22): burn_antihall, burn_attnres, burn_bitnet, burn_byteflow, burn_diffusionblocks, burn_eggroll, burn_engram, burn_es (+14 more)

### Community 179 - "qr_cpu()"
Cohesion: 0.17
Nodes (21): __m256, __m256d, apply_pair(), cholesky_host(), cholesky_host_par(), dot3_avx_f64(), dot_pair(), hsum() (+13 more)

### Community 180 - "burn-kda Kimi Delta Attention (KDA) for Burn"
Cohesion: 0.10
Nodes (23): burn-gdn2 CI workflow, burn-gdn2 Gated DeltaNet 2, arXiv 2605.22791 Gated DeltaNet 2 (Hatamizadeh et al., 2026), NVlabs GatedDeltaNet-2 reference (Triton + flash-attn, NVIDIA only), burn-gdn2 roadmap: autodiff-aware fused kernels, burn-kda CI workflow, FlashKDA CUTLASS reference path (Moonshot), burn-kda Kimi Delta Attention (KDA) for Burn (+15 more)

### Community 181 - "Captures<D>"
Cohesion: 0.15
Nodes (11): Captures<D>, GraphDriver, Debug, Default, Formatter, Option, Result, Self (+3 more)

### Community 182 - "PerpendicularWriter"
Cohesion: 0.13
Nodes (16): PerpendicularWriter, PerpendicularWriter<'a, Out>, PerpendicularWriter<'_, Out>, Array, Coords2d, EI, N, Out (+8 more)

### Community 183 - "resolve()"
Cohesion: 0.18
Nodes (17): disabled_preset_keeps_the_capacity_budget(), drift_names_changed_keys(), engram_ram_drops_the_in_vram_table(), flag_beats_set(), model_and_data_agree_on_the_engram_orders(), mor_and_rand_depth_are_refused_together(), qk_heads_derived_from_model(), resolve() (+9 more)

### Community 184 - "cache.rs"
Cohesion: 0.17
Nodes (19): DeviceIdentity, a_container_given_fewer_cores_does_not_reuse_the_host_ceiling(), a_device_key_is_one_path_segment(), a_machine_that_gains_memory_does_not_reuse_its_ceiling(), device_key(), drop_earlier_generations(), GENERATION, GLOBAL_CACHE (+11 more)

### Community 185 - "DeviceStream"
Cohesion: 0.11
Nodes (8): DeviceStorage, HostStorage, Signal, Stream, DeviceStream, Signal, Fence, Sized

### Community 186 - "qtensor.rs"
Cohesion: 0.22
Nodes (19): dtype_to_elem_type, quantize(), CubeTensor, Option, QuantScheme, empty_qtensor(), empty_qtensor_optimized(), new_qtensor() (+11 more)

### Community 187 - "SourceTemplate"
Cohesion: 0.12
Nodes (14): Name, build_info(), KernelSource, CubeTensor, K, Send, Sync, Vec (+6 more)

### Community 188 - "ternary_mutate()"
Cohesion: 0.20
Nodes (21): antithetic_pair(), antithetic_shape(), dev(), eggroll_mutate(), eggroll_shape(), eggroll_sigma_scales_perturbation(), es_gradient(), es_gradient_points_up_hill() (+13 more)

### Community 189 - "autodiff.rs"
Cohesion: 0.16
Nodes (19): chunk_autodiff_or_plain(), chunk_wy_forward_autodiff(), ChunkWy, N_PARENTS, B, Backward, Checkpointer, Gradients (+11 more)

### Community 190 - "autodiff.rs"
Cohesion: 0.14
Nodes (13): Gdn2Config, Gdn2Mode, Default, Option, Self, cfg(), chunk_and_fused_agree(), decode_equals_full_forward() (+5 more)

### Community 191 - "qr_cuda.rs"
Cohesion: 0.18
Nodes (19): cube_of(), cube_of_int(), forward_cuda(), from_dense_cuda(), retract_cuda(), row_stride(), CubeTensor, F (+11 more)

### Community 192 - "polar_orthogonalize()"
Cohesion: 0.19
Nodes (18): bench_old_vs_batched_retraction(), naive_topk(), ortho_error(), polar_five_iters(), polar_on_random(), polar_orthogonalize(), polar_orthogonalize_batched(), polar_square_and_tall_no_divergence() (+10 more)

### Community 193 - "AccumulatorFormat"
Cohesion: 0.11
Nodes (18): AccumulatorFormat, EI, N, Out, ReadWrite, SI, T, VirtualTensor (+10 more)

### Community 194 - "reduce_dim.rs"
Cohesion: 0.18
Nodes (21): TestCase, test_all(), test_any(), test_argmax(), test_argmin(), test_argtopk_3(), test_argtopk_5(), test_case() (+13 more)

### Community 195 - "Fast training kernels for dormouse's step time (2026-09-22"
Cohesion: 0.16
Nodes (21): fused/ hand-written CUDA autodiff op for the ponder loop, ADR-0003: fused/ fast path - measure, then keep or delete, ADR-0003 smoke verdict: kept at 1.7-2.0x, Three measured causes of the fused deficit, ADR-0009: Fused rewrite - two rungs, then death, The bar fused must beat is fusion-on burn, not yesterday's burn, Hard kill switch: either rung failing deletes all ~7.4k LOC of fused/, Rung 1: seed KDA/MSA/Engram adjoints from saved per-iteration buffers (+13 more)

### Community 196 - "bitforbit.rs"
Cohesion: 0.14
Nodes (17): chunk_wy_forward, fused_recurrent_forward, silu, l2_normalize(), l2_normalize_4d(), Tensor, Option, Tensor (+9 more)

### Community 197 - "CubeBackend"
Cohesion: 0.19
Nodes (11): QTensorOps, QuantizationParametersPrimitive, TensorPrimitive, CubeBackend, ExecutionError, FloatDType, FloatTensor, QuantizedTensor (+3 more)

### Community 198 - ".to_output_perpendicular()"
Cohesion: 0.16
Nodes (9): sum, All, Config, EI, Idx, Out, Self, SI (+1 more)

### Community 199 - "InputGenerator"
Cohesion: 0.13
Nodes (14): tuneinputs, CloneInputGenerator, Func, InputGenerator, A, At, I, K (+6 more)

### Community 200 - ".to_output_perpendicular()"
Cohesion: 0.18
Nodes (10): Mean, null_input(), Config, EI, Idx, Out, Self, SI (+2 more)

### Community 201 - "cublas_combos.cu"
Cohesion: 0.13
Nodes (17): algorithm, cmath, cstdio, cstring, cublas_v2, cublasComputeType_t, cuda_fp16, cudaDataType_t (+9 more)

### Community 202 - ".kernel_output()"
Cohesion: 0.20
Nodes (13): cpu_reference, proof_shape, ramp_max_elems, proof_shape(), ReduceCorrectness, Correctness, HostData, Option (+5 more)

### Community 203 - "events.rs"
Cohesion: 0.22
Nodes (11): CUevent, Cuda, .BACKEND, CudaEvent, named(), Duration, Result, Send (+3 more)

### Community 204 - "AutotuneKey"
Cohesion: 0.11
Nodes (17): DeserializeOwned, fmt, stablehasher, fake_kernel(), FakeAutotuneKey, Display, Formatter, Result (+9 more)

### Community 205 - "memory_pools_config.rs"
Cohesion: 0.20
Nodes (19): serverlogger, a_measurement_maps_only_what_it_resolves(), a_warm_second_pass_measures_the_workload_alone(), capped_pool_respects_max_slice_size(), configure_rebuilds_pools_in_place(), direct_pool_cleanup_releases_everything_free(), direct_pool_pads_only_to_alignment(), direct_pool_reclaims_at_the_ceiling() (+11 more)

### Community 206 - "conv_transpose2d_col2im()"
Cohesion: 0.18
Nodes (19): col2im(), col2im_kernel(), Col2ImArgs, conv_transpose2d_col2im(), execute(), index(), ComptimeOption, ConvSetupError (+11 more)

### Community 207 - ".to_output_perpendicular()"
Cohesion: 0.17
Nodes (9): as_server(), Any, Config, EI, Idx, Out, Self, SI (+1 more)

### Community 208 - "DriverError"
Cohesion: 0.17
Nodes (12): checked(), CompilationError, DriverError, IoError, LaunchError, Display, Error, Formatter (+4 more)

### Community 209 - "ReduceWriter"
Cohesion: 0.17
Nodes (8): ReduceWriter, CubeType, I, Self, Writer<'a, Out>, IndicesWriter<'a, Out, Idx>, I, Self

### Community 210 - "reference_argmax()"
Cohesion: 0.11
Nodes (13): reference_argmax(), HostData, Option, Progress, reference_max_abs(), HostData, Option, Progress (+5 more)

### Community 211 - "PinnedMemoryAllocController"
Cohesion: 0.16
Nodes (13): bytes, empty_pinned_slice_mut(), PINNED_MEMORY_ALIGNMENT, PinnedMemoryAllocController, PinnedMemoryResource, AccessError, AccessPolicy, AllocationController (+5 more)

### Community 212 - "conv_forward_nhwc()"
Cohesion: 0.25
Nodes (17): conv, conv_autotune, conv_data_backward(), conv_forward(), conv_forward_nhwc(), conv_weight_backward(), ConvStrategy, ConvOptions (+9 more)

### Community 213 - "graph.rs"
Cohesion: 0.12
Nodes (11): cubecl_core_as_cubecl, cudaruntime, runtime, F, tiny_kernel(), add_one_tensor(), CAPTURE_LOCK, cuda_graph_capture_growing_the_pool_is_rejected() (+3 more)

### Community 214 - "Executable"
Cohesion: 0.25
Nodes (14): CUgraph, CUgraphExec, CUresult, checked(), count_memory_nodes(), Cuda, Executable, instantiate_recording() (+6 more)

### Community 215 - "Event<A>"
Cohesion: 0.16
Nodes (10): Deref, Target, Event<A>, Pooled<A>, Debug, Drop, Duration, Formatter (+2 more)

### Community 216 - "CubeClRuntimeConfig"
Cohesion: 0.19
Nodes (12): RuntimeConfig, CUBE_GLOBAL_CONFIG, CubeClRuntimeConfig, env_bool(), Arc, Mutex, Option, Self (+4 more)

### Community 217 - "deform_conv2d.rs"
Cohesion: 0.17
Nodes (18): bilinear_interpolate(), deform_conv2d(), deform_im2col(), deform_im2col_kernel(), DeformConv2dArgs, ComptimeOption, ConvSetupError, CubeTensor (+10 more)

### Community 218 - "scatter.rs"
Cohesion: 0.19
Nodes (18): adds_atomically(), Atomic, CubeTensor, ElemType, FastDivmod, I, Sequence, T (+10 more)

### Community 219 - "hasher.rs"
Cohesion: 0.18
Nodes (11): deterministic_and_in_range(), hash_tensor_shape(), is_prime(), matches_reference_algorithm(), NgramHasher, OddMultipliers, primes_are_prime_and_shared(), Int (+3 more)

### Community 220 - "dry_run.rs"
Cohesion: 0.16
Nodes (14): a_dry_run_spares_the_measurements(), depth(), DRY_RUN, dry_runs_nest(), DryRun, enter(), exit(), launch_mode() (+6 more)

### Community 221 - "MemoryAccess"
Cohesion: 0.18
Nodes (11): a_working_set_is_part_of_the_key(), compute_throughput_key(), DEFAULT_WORKING_SET_BYTES, MemoryAccess, MemorySpec, PROBE_VERSION, ElemType, Option (+3 more)

### Community 222 - "roofline.rs"
Cohesion: 0.22
Nodes (14): a_zero_duration_reports_nan_instead_of_dividing_by_zero(), AchievedThroughput, binding_achieved(), binding_resource(), binding_resource_is_the_one_needing_the_most_time_at_peak(), binding_resource_skips_non_normal_peaks_and_is_none_if_all_are(), bound(), ResourceBound (+6 more)

### Community 223 - ".to_output_perpendicular()"
Cohesion: 0.19
Nodes (8): MaxAbs, Config, EI, Idx, Out, Self, SI, Vector

### Community 224 - ".to_output_perpendicular()"
Cohesion: 0.19
Nodes (8): Prod, Config, EI, Idx, Out, Self, SI, Vector

### Community 225 - "ParallelWriter"
Cohesion: 0.13
Nodes (13): ParallelWriter, ParallelWriter<'a, Out>, Coords2d, EI, I, N, Out, ReadWrite (+5 more)

### Community 226 - "train.rs"
Cohesion: 0.16
Nodes (15): actquant, Args, Option, PathBuf, Vec, Args, build_run(), execute() (+7 more)

### Community 227 - "fused.rs"
Cohesion: 0.12
Nodes (12): backend, main(), cubeautotunekey, cubetuneid, device_as_cubedevice, Element, load_config, r_override (+4 more)

### Community 228 - "act_quant.rs"
Cohesion: 0.16
Nodes (13): ActFormat, fp4_round(), quant_act(), Self, Tensor, distribution, ndarray, causal_row_t_only_sees_prefix() (+5 more)

### Community 229 - "reduce_with_indices_kernel()"
Cohesion: 0.20
Nodes (18): IdxSize, reduce_kernel(), reduce_kernel_inner(), reduce_kernel_virtual(), reduce_with_indices_kernel(), reduce_with_indices_kernel_inner(), Config, EI (+10 more)

### Community 230 - "kernel_binop()"
Cohesion: 0.17
Nodes (16): ArcTan2Op, atan2(), BinaryOpFloat, BinaryOpFloatFamily, kernel_binop(), launch_binop_float(), C, CubeTensor (+8 more)

### Community 231 - "byte_patch()"
Cohesion: 0.26
Nodes (17): bltd_loss(), bltd_loss_finite_and_scaled(), byte_patch(), byte_patch_splits_long_patches(), dev(), ints(), Bool, Int (+9 more)

### Community 232 - "alloc_trace.rs"
Cohesion: 0.14
Nodes (13): bytes_of(), CHUNK_ITERATIONS, dump(), ENABLED, AtomicBool, AtomicU64, D, Mutex (+5 more)

### Community 233 - "KdaModule"
Cohesion: 0.28
Nodes (8): DecayFn, kda_step(), KdaDecay, KdaModule, Linear, Option, Param, Tensor

### Community 234 - "ThroughputValue"
Cohesion: 0.19
Nodes (9): Duration, ThroughputError, ThroughputKey, ThroughputValue, FnOnce, Result, Option, Store (+1 more)

### Community 235 - "EventPool"
Cohesion: 0.17
Nodes (11): EventPool, EventPool<A>, Pooled, A, Arc, Clone, Default, Mutex (+3 more)

### Community 236 - "FailureStore"
Cohesion: 0.24
Nodes (8): ReadFailure, ServerError, FailureStore, Iterator, Option, Result, ServerError, Vec

### Community 237 - "A/B or death: ties delete too"
Cohesion: 0.18
Nodes (17): BPB on the eval tail at a fixed step budget and memory envelope, ADR-0001: BPB at fixed budget is the score, ADR-0002: A/B or death, ties delete, Protocol: 200-500 step smoke filter then 2k+ confirm, A/B or death: ties delete too, Loud-failures doctrine: data that does not exist stops the run, ADR-0011: Loud failures - no silent data, no silent skips, NASA P10 Rule 5: assertion density >= 2 per non-trivial function (+9 more)

### Community 238 - "lowp_bf16_cuda.rs"
Cohesion: 0.14
Nodes (12): btreemap, fused_chunk_forward_scratch, main(), F, time(), bf16(), bf16_storage_with_f32_accumulation(), cube_of() (+4 more)

### Community 239 - "cuda_retract.rs"
Cohesion: 0.15
Nodes (8): burn_sct, instant, qr_cpu, cuda_dev(), gpu_forward_matches_tensor_path_non_pow2(), gpu_retract_matches_cpu(), gpu_retract_speed(), max_diff()

### Community 240 - "operation_sets.rs"
Cohesion: 0.32
Nodes (16): dummy, TestSet, addition_set(), addition_set_with_eviction(), addition_set_with_failing_compilation(), addition_set_with_failing_compilation_first(), addition_set_with_rejected_candidate(), addition_set_with_slow_candidate() (+8 more)

### Community 241 - "Rung 0 — MEASURE the depth curve BPB@k (k=1..4), per-itera"
Cohesion: 0.19
Nodes (17): A/B protocol vs fixed-depth-4: arms A0, A0', A1, A2, A3, A4, A5 with pre-registered kill criteria, The entire best measured adaptive-depth gain comes from 2.7% of inputs (+53 net correct, 56 W→R and 3 R→W), Rung 0 — MEASURE the depth curve BPB@k (k=1..4), per-iteration ΔCE, oracle label rate, EAS@k from one existing checkpoint, The Diminishing Returns of Early-Exit Decoding in Modern LLMs arXiv 2603.23701, Do Transformers Use their Depth Adaptively? arXiv 2604.12426, EAS (Early-exit effectiveness score): weighted geometric mean of skip ratio w_l = (L−l)/L and layer-to-final cosine similarity S_l, Held-out eval window: fixed rewound 100 KB (20 batches × 5 120 B), final-readout CE only, targets=None so no per-iteration CE, Window bias (systematic, not noise): 100 KB absolute BPB must never be compared to published BPBs (+9 more)

### Community 242 - "conv_autotune()"
Cohesion: 0.22
Nodes (16): conv_autotune(), create_conv_input(), create_key(), DEPTHWISE_2X4_SCALAR, DEPTHWISE_4X2_SCALAR, DEPTHWISE_8X2_LINED, DEPTHWISE_8X4_LINED, MAX_STRIDE_FACTOR (+8 more)

### Community 243 - "BenchConfig"
Cohesion: 0.15
Nodes (9): AutotuneConfig, AutotuneLevel, AutotuneLogLevel, BenchConfig, DecisionLevel, Default, LoggerConfig, LogLevel (+1 more)

### Community 244 - "EventApi"
Cohesion: 0.24
Nodes (8): EventApi, .BACKEND, Duration, Result, Send, Stream, Sync, Event

### Community 245 - "matmul_autotune()"
Cohesion: 0.12
Nodes (16): MatmulCost, MatmulProblemDefinition, MatmulTunables, TileMatmulKind, matmul_autotune(), CubeTensor, DType, Option (+8 more)

### Community 246 - ".reduce_single()"
Cohesion: 0.30
Nodes (11): reducestep, GlobalFullPlaneReduce, ComptimeOption, EI, I, N, ReadWrite, SI (+3 more)

### Community 247 - "dormouse --rand-depth arm (uniform T ∈ 1..4 per step, both"
Cohesion: 0.17
Nodes (16): Capability flattening: the real risk of the anchor term is iterations 2..4 adding nothing (BPB@4 ≈ BPB@1), Depth robustness: one checkpoint usable at every depth in the trained range, Dual-trajectory objective (full deep path + sampled shortcut, stop-gradient logit consistency), Extrapolation ceiling: recurrent depth holds ~1.5× supervised depth, 70% accuracy through depth 18, Fixed depth peaks then falls on a looped backbone: 54.71 @ T=2 → 53.62 @ 8 → 47.54 @ 16 (RecurTrace Table 2), iter_embed: learned per-iteration embedding = discrete time conditioning, already in the architecture, STARS' Jacobian regularizer is not implementable here (JVP = forward-mode AD, stack is reverse-mode), Random depth buys depth-robustness but does not make the loop settle (+8 more)

### Community 248 - "select_assign.rs"
Cohesion: 0.17
Nodes (15): CubeTensor, ElemType, F, FastDivmod, I, LinearView, Sequence, Tensor (+7 more)

### Community 249 - "chunk_cube.rs"
Cohesion: 0.23
Nodes (14): cube_of(), fused_chunk_forward(), fused_chunk_forward_scratch(), gdn2_chunk_inter_kernel(), gdn2_chunk_intra_kernel(), gdn2_chunk_trajectory_export_kernel(), inter_launch_raw(), intra_launch_raw() (+6 more)

### Community 250 - "capture.rs"
Cohesion: 0.25
Nodes (12): GraphId, Default, Captures, Graph, Refused, Bytes, D, HashMap (+4 more)

### Community 251 - "VectorizationMode"
Cohesion: 0.15
Nodes (13): BoundChecks, output_vectorization_axis(), Self, Strides, VectorizationMode, Reader<'a, P>, ComptimeOption, EI (+5 more)

### Community 252 - ".reduce_shared()"
Cohesion: 0.35
Nodes (11): GlobalFullCubeReduce, ComptimeOption, EI, I, N, ReadWrite, SI, T (+3 more)

### Community 253 - "topk_with_indices_cube.rs"
Cohesion: 0.24
Nodes (15): case(), cube_planes_argtopk_k3(), cube_planes_max_with_indices(), cube_planes_min_with_indices(), cube_planes_topk_values_only_k3(), cube_planes_topk_with_indices_k1(), cube_planes_topk_with_indices_k2(), cube_planes_topk_with_indices_k3() (+7 more)

### Community 254 - "fused/ grad-coverage verification (phase 1 item 1)"
Cohesion: 0.17
Nodes (15): ADR-0005: One config seam - resolve, validate, snapshot, Snapshot drift check on resume, Presets are data: flat TOML found by search path, Single resolve seam with fixed merge order, Arms-on backend-mismatch fix (dev_ad allocation), Coverage broadcast panic fix (rows_idx = p_final + 1), Direct Cube adjoint kernels as the escape hatch, Flagship wiring gate DM_FUSED=1 with aux burn-side (+7 more)

### Community 255 - "ADR-0014: MSA is cut, not deferred"
Cohesion: 0.23
Nodes (15): MSA ships disabled on pre.4 (DM_MSA_FUSED off, use_msa = false), Minimal repro vendor/burn-fused/crates/burn-msa/examples/msa_repro.rs, ADR-0012: MSA is broken on the pre.4 stack, MSA out-of-bounds gathers in both paths on pre.4, AdaptiveAttention wrapper kept for checkpoint param-prefix stability, A disabled arm still gets renamed, still breaks routing, still costs, Delete MSA: crate, config keys, --no-msa, blend router, QK_KV optimizer group, ADR-0014: MSA is cut, not deferred (+7 more)

### Community 256 - "kda_step_probe.rs"
Cohesion: 0.17
Nodes (12): burn_kda, B, CHUNK, D, H, HK, T, cfg() (+4 more)

### Community 257 - "backend_parity.rs"
Cohesion: 0.22
Nodes (14): assert_bool_to_float(), bf16_gemm_cuda(), bf16_matmul_cuda(), bool_to_float_cuda(), bool_to_float_ndarray(), bool_to_float_parity(), f16_gemm_cuda(), f16_matmul_cuda() (+6 more)

### Community 258 - "client.rs"
Cohesion: 0.14
Nodes (11): DeviceHandle, timingmethod, Graph, GraphHandle, profile_label(), Clone, Debug, Drop (+3 more)

### Community 259 - "host_svd.rs"
Cohesion: 0.23
Nodes (11): float, bidiag_host(), dbdsqr(), dlartg(), dlas2_smax(), F, Send, Sync (+3 more)

### Community 260 - "streaming.rs"
Cohesion: 0.16
Nodes (8): StreamPolicy, default_max_streams(), Default, LoggerConfig, LogLevel, Self, StreamingConfig, StreamingLogLevel

### Community 261 - ".resolve()"
Cohesion: 0.19
Nodes (12): vec, MemoryConfiguration, PERSISTENT_POOL_POS, pool_options_from_entry(), PoolConfigError, Display, Error, Formatter (+4 more)

### Community 262 - "weights.rs"
Cohesion: 0.34
Nodes (14): b158_roundclip_matches_paper_closed_form(), dev(), D, Tensor, Vec, to_host(), weight_2bit_ste_gradient_is_identity(), weight_2bit_values() (+6 more)

### Community 263 - ".forward()"
Cohesion: 0.23
Nodes (11): byte_group_hash_ids(), hash_deterministic_and_in_range(), hash_embeddings_shape(), hash_matches_reference_formula(), HASH_PRIMES, HashEmbeddings, Embedding, Int (+3 more)

### Community 264 - "topk_gather.rs"
Cohesion: 0.30
Nodes (14): all_ties(), check(), devices(), masked(), one_hot(), production_shape_stays_in_range(), Row, Int (+6 more)

### Community 265 - "CompilationConfig"
Cohesion: 0.16
Nodes (11): BoundsCheckMode, CompilationConfig, CompilationLogLevel, F16Evaluation, Display, Formatter, LoggerConfig, LogLevel (+3 more)

### Community 266 - "memory.rs"
Cohesion: 0.19
Nodes (11): MemoryConfig, MemoryLogLevel, MemoryPoolConfig, MemoryPoolsConfig, MemoryPoolsPreset, PersistentMemory, LoggerConfig, LogLevel (+3 more)

### Community 267 - "topk_unroll_limit.rs"
Cohesion: 0.30
Nodes (14): RoutineStrategy, case(), K, plane(), plane_eager_topk_past_unroll_limit(), plane_eager_topk_with_indices_past_unroll_limit(), plane_lazy_topk_past_unroll_limit(), plane_lazy_topk_with_indices_past_unroll_limit() (+6 more)

### Community 268 - "tune_key.rs"
Cohesion: 0.20
Nodes (10): autotunekey, serde, ConvAutotuneKey, ConvTranspose2dAutotuneKey, key(), DType, Vec, ReduceAutotuneKey (+2 more)

### Community 269 - "bf16_ops.rs"
Cohesion: 0.21
Nodes (12): burn_autodiff, nocheckpointing, bf16_matmul(), bf16_matmul_ffn_sizes_finite(), bf16_matmul_matches_fp32_with_grads(), Bf16Matmul, B, Backward (+4 more)

### Community 270 - ".new_quantized()"
Cohesion: 0.22
Nodes (9): QParamTensor, CubeTensor, DType, Fn, Option, QParams, Self, Shape (+1 more)

### Community 271 - "What goes on top of random-depth training in dormouse — ad"
Cohesion: 0.27
Nodes (14): Adaptive depth for looped models (variable compute per input), Arrabal-Campos arXiv 2608.22347, DeepLoop arXiv 2607.13491, Fixed-depth 2026 SOTA (DeepLoop, SMELT, Hyperloop, Training-Free, LoopUS, Huginn): none use trained halting, Hyperloop arXiv 2604.21254, Mixture-of-Depths arXiv 2404.02258, PALBERT arXiv 2204.03276, research/2026-09-26-ponder-replacement.md (prior session's ponder-replacement report) (+6 more)

### Community 272 - "CubeAutotuneKey"
Cohesion: 0.24
Nodes (11): SumAutotuneKey, conv_transpose2d_autotune(), create_key(), create_transpose2d_input(), ConvTransposeOptions, CubeTensor, Option, CubeAutotuneKey (+3 more)

### Community 273 - "chunk_adjoint_cube.rs"
Cohesion: 0.25
Nodes (12): cube_of(), fused_chunk_backward(), FusedBackwardInputs, FusedBackwardOutput, gdn2_chunk_inter_adjoint_kernel(), gdn2_chunk_intra_adjoint_kernel(), grp_id(), CubeTensor (+4 more)

### Community 274 - "forward_shape()"
Cohesion: 0.26
Nodes (9): dev(), forward_shape(), gate_shape(), Linear, Self, Tensor, SwiGLU, swiglu_gate() (+1 more)

### Community 275 - "LaunchObserver"
Cohesion: 0.24
Nodes (9): deliver_timed(), LaunchObserver, Recorder, Duration, Mutex, Sync, TimingMethod, Vec (+1 more)

### Community 276 - ".reduce_single()"
Cohesion: 0.37
Nodes (10): GlobalFullUnitReduce, ComptimeOption, EI, I, N, ReadWrite, SI, T (+2 more)

### Community 277 - "ADR-0018: dormouse-fused is a standalone project, with thr"
Cohesion: 0.23
Nodes (13): ADR-0017: dormouse-fused - the technology lives in the library, Every mechanism and kernel lives in the library, never inside the model, tools/migrate-dormouse-fused.sh: rename scripted for a quiet tree, vendor/burn-fused -> dormouse-fused, 26 crates, one namespace, 16 unwired crates are invisible (implemented-and-unused vs imagined), On-device NaN firewall (mask on device, sanitize on device), Forbidden: touching dormouse-core from a library crate, in any direction, Hard rule 3: beat the original on speed and on VRAM (+5 more)

### Community 278 - "create_key()"
Cohesion: 0.31
Nodes (12): acceleratedtilekind, create_key(), create_wgrad_input(), dgrad_autotune(), MAX_STRIDE_FACTOR, pow2_factor(), ConvOptions, CubeTensor (+4 more)

### Community 279 - ".on_stream()"
Cohesion: 0.19
Nodes (7): c_int, Blas, Buf, c_void, CUstream, Drop, GemmEx

### Community 280 - "write_scope.rs"
Cohesion: 0.22
Nodes (10): GlobalAlloc, ALLOCATOR, ALLOCS, Counting, cycle(), main(), measure(), AtomicUsize (+2 more)

### Community 281 - "ADR-0013 fixed depth (PonderNet halting deleted after two "
Cohesion: 0.31
Nodes (13): ADR-0013 fixed depth (PonderNet halting deleted after two measured collapses), Collapse-proof-by-construction test: does a learned quantity scale the loss, and is the target recomputed each step?, Cumulative-softmax depth threshold router (2-layer MLP on initial hidden state; token active iff p^(i) = Σ_{l≥i} π_l > 0.5), Expert-choice top-k with fixed β-percentile capacity: exactly k tokens always get through, Penalty-based halting collapses to the floor: ACT and PonderNet both hit 49.96 @ 1.00 loop — exactly dormouse's failure, on a different backbone, Mixture-of-Recursions arXiv 2507.10524 (Bae et al., NeurIPS 2025; N_r=3, 130M–1.7B), CORRECTION: ADR-0013's MoR 135M reason was wrong; the rejection stands on a different ground (the sm_120 per-depth gather), MoR-style expert-choice router rejected (Rung 5: do not build) (+5 more)

### Community 282 - "launch.rs"
Cohesion: 0.44
Nodes (12): dgrad_gemm_simple_async(), dgrad_gemm_simple_sync(), dgrad_gemm_simple_tma(), launch_backwards_data(), AcceleratedTileKind, ConvOptions, ConvSetupError, CubeTensor (+4 more)

### Community 283 - "launch.rs"
Cohesion: 0.44
Nodes (12): launch_backwards_weight(), AcceleratedTileKind, ConvOptions, ConvSetupError, CubeTensor, N, Result, Shape (+4 more)

### Community 284 - "Presets nano..one_b share one skeleton; RAM buys Engram ro"
Cohesion: 0.24
Nodes (12): Distillation enters when a teacher budget exists (teacher ~2.5x student), Presets nano..one_b share one skeleton; RAM buys Engram rows, VRAM buys preset and batch, ADR-0007: The architecture scales with hardware, by commitment, Every small-preset A/B is a vote on the flagship, Confirm runs report the train-vs-eval BPB gap, Engram grows so the small core never spends capacity on facts, ADR-0008: Reasoning over recall, RSI rungs: process automation, data flywheel, verifier-filtered self-improvement, model-scored architecture search (+4 more)

### Community 285 - "runtime.rs"
Cohesion: 0.17
Nodes (11): adapterluid, cubecl_cpp, cudevicegetluid, ptxversion, restrict_to_llvm_backend(), CudaArchitecture, DeviceProperties, MemoryConfiguration (+3 more)

### Community 286 - "bench_catalog.rs"
Cohesion: 0.20
Nodes (11): benchmarks, comparison_epsilon, reducestrategy, comparison_epsilon(), lookup(), CatalogEntry, T, Vec (+3 more)

### Community 287 - "kda_alloc_probe.rs"
Cohesion: 0.18
Nodes (10): burn_dispatch, B, CHUNK, client_of(), D, H, HK, main() (+2 more)

### Community 288 - "runtime.rs"
Cohesion: 0.18
Nodes (8): cubecl_zspace, Clone, Debug, Send, Sized, Sync, TargetProperties, Runtime

### Community 289 - "Huginn — Scaling up Test-Time Compute with Latent Reasonin"
Cohesion: 0.17
Nodes (12): Decaying survival probability p_l = 1 − (l/L)(1 − p_L), p_L = 0.5, + test-time re-weighting, Energy-guided Recursive Model (ERM) arXiv 2607.10128, Huang et al. 2016 — Deep Networks with Stochastic Depth arXiv 1603.09382, Huginn — Scaling up Test-Time Compute with Latent Reasoning arXiv 2502.05171 (Geiping et al., 3.5B / 800B tokens), Huginn 'Bad Run 1': representation collapse (hidden-state correlation → 1.0), Missing-state problem: copying K/V from lower layers into skipped layers collapses to 23.02 ROUGE-L, Sequence-level (per-row) exit is the right first rung; per-token exit only after the KV story is solved, Shallow-biased sampling with a heavy tail (Huginn log-normal-Poisson; Huang decaying survival) (+4 more)

### Community 290 - "Eviction"
Cohesion: 0.23
Nodes (9): string, Eviction, Func, A, At, K, Result, Send (+1 more)

### Community 291 - "create_key()"
Cohesion: 0.35
Nodes (11): create_key(), create_wgrad_input(), MAX_STRIDE_FACTOR, pow2_factor(), ConvOptions, CubeTensor, ElemType, N (+3 more)

### Community 292 - "conv_gemm_simple_async()"
Cohesion: 0.42
Nodes (12): conv_gemm_simple_async(), conv_gemm_simple_sync(), conv_gemm_simple_tma(), launch_convolution_forward(), AcceleratedTileKind, ConvOptions, ConvSetupError, CubeTensor (+4 more)

### Community 293 - "staging.rs"
Cohesion: 0.18
Nodes (6): FLUSH_MIN, MB, AllocationProperty, Self, STAGE_MAX, Staging

### Community 294 - "EventFence<A>"
Cohesion: 0.24
Nodes (7): EventFence<A>, Debug, Formatter, Result, Self, ServerError, Stream

### Community 295 - ".compile()"
Cohesion: 0.27
Nodes (8): CompiledKernel<C>, C, CompilationError, CompilationOptions, Display, Formatter, Result, Self

### Community 296 - "policy.rs"
Cohesion: 0.36
Nodes (9): flush_triggered_by_whichever_limit_comes_first(), flush_when_count_threshold_reached(), flush_when_size_threshold_reached(), FlushingPolicyState, no_flush_when_below_both_thresholds(), register_saturates_instead_of_overflowing(), reset_clears_state(), Bytes (+1 more)

### Community 297 - "PendingDropQueue<E>"
Cohesion: 0.18
Nodes (7): PendingDropQueue<E>, Debug, Default, Formatter, Result, Self, ServerError

### Community 298 - "idle_check()"
Cohesion: 0.24
Nodes (11): idle_check(), reduce_count(), reduction_output_base(), ComptimeOption, EI, N, ReadWrite, SI (+3 more)

### Community 299 - "CompiledKernel"
Cohesion: 0.20
Nodes (10): AtomicI8, core, format_str, COMPILATION_LEVEL, CompiledKernel, DebugInformation, BufferIOAttr, Option (+2 more)

### Community 300 - "MetadataInfoCache"
Cohesion: 0.24
Nodes (8): InfoCacheKey, Entry, Lookup, MetadataInfoCache, HashMap, HashSet, V, Vec

### Community 301 - "fused_matrix_static.py"
Cohesion: 0.22
Nodes (9): pathlib, subprocess, sys, block_end(), main(), Static half of the burn-fused quality matrix: LOC, test LOC, markers, doc lies.…, Index one past the closing brace of the block opened at/after `start`., test_loc() (+1 more)

### Community 302 - "bitforbit_cuda.rs"
Cohesion: 0.31
Nodes (10): dmax(), dump4(), IN, load4(), main(), OUT, read_f32(), Tensor (+2 more)

### Community 303 - "fused_matches_tensor()"
Cohesion: 0.36
Nodes (8): dev(), forward_shape(), fused_matches_tensor(), RMSNorm, Param, Self, Tensor, unit_variance()

### Community 304 - "ttt_ce_loss()"
Cohesion: 0.42
Nodes (10): ce_loss_masked_positions_ignored(), ce_loss_zero_when_perfect(), dev(), latent_loss_masked(), latent_loss_zero_when_equal(), Int, Tensor, token_log_probs() (+2 more)

### Community 305 - "LaunchObservation"
Cohesion: 0.22
Nodes (10): installed_observer(), LaunchObservation, OBSERVER, Arc, Debug, Drop, Formatter, Option (+2 more)

### Community 307 - "PendingDropQueue"
Cohesion: 0.18
Nodes (8): FlushingPolicy, policy(), Default, Self, PendingDropQueue, E, Option, Vec

### Community 308 - "MetadataCachePolicy"
Cohesion: 0.24
Nodes (4): CacheMode, MetadataCachePolicy, Default, Self

### Community 309 - "Loop runs at fixed max_iter=4 with honest unweighted CE"
Cohesion: 0.29
Nodes (10): ADR-0006: Architecture verdicts from the 2026-09-21 survey, Loop runs at fixed max_iter=4 with honest unweighted CE, ADR-0013: Fixed depth by default; PonderNet halting deleted, The KL direction was wrong: p||prior makes collapse free, lambda-collapse makes the PonderNet rec loss fake, Random-depth training as the rank-2 alternative kept on file, Every earlier train-CE reading from the PonderNet era is void, LOC cut scenarios A-D and the ~5.5-6k floor (+2 more)

### Community 310 - "ComputeCmmaConfig"
Cohesion: 0.24
Nodes (7): AccumulatorType, Cache, CmmaDims, ComputeCmmaConfig, ElemType, Option, select_cmma_tile()

### Community 311 - "precision.rs"
Cohesion: 0.27
Nodes (7): atomic, FP8_WARNED, kernel_precision(), Precision, precision_from_env(), AtomicBool, Option

### Community 312 - "StressMonitor"
Cohesion: 0.22
Nodes (5): Option, Self, Vec, StressMonitor, VecDeque

### Community 313 - "ckpt_roundtrip.rs"
Cohesion: 0.33
Nodes (9): ckpt_roundtrip_identical_logits(), device(), hashed_ids(), mini_nano(), nano_cfg(), Int, Tensor, dormouse_core (+1 more)

### Community 314 - "bench_fused_bwd.rs"
Cohesion: 0.24
Nodes (5): cudabare, sctlinear, bench_fused_bwd(), FnMut, time_it()

### Community 315 - "DeviceProbe"
Cohesion: 0.22
Nodes (7): CUdevice, PhysicalDevice, ServerUtilitiesHandle, CudaServer, DeviceProbe, DeviceService, Self

### Community 316 - "Margin exit criterion p1 − p2 on the fp32 logits (CALM's w"
Cohesion: 0.24
Nodes (10): CALM — Confident Adaptive Language Modeling, Schuster et al. 2022 arXiv 2207.07061 (NeurIPS 2022 oral), Confidence exit overspends and loses a little: CALM margin 5.63 loops for −14 items, top-prob 4.15 loops for −40 items, Displacement sufficient condition: R_t = Σ_{u≥t} δ_u < μ(z_t) ⇒ 'the decoded answer cannot change under further iterations', Fake-loss canary: unweighted final-state CE vs any p-weighted number, gap ≫ noise, Guess-freezing: the answer freezes early and freezing does not track difficulty, Huginn-3.5B: 2585 per-token latent trajectories, NOT ONE settles (σ_eff = 1.000 ± 0.003); carry probe 0.69 @ r=8 → 0.00 for r ≥ 32, LoopUS arXiv 2605.11011 (LoopUS-Conf 55.25 @ 3.21 loops in the RecurTrace comparison), Margin exit criterion p1 − p2 on the fp32 logits (CALM's winner) (+2 more)

### Community 317 - "slice_assign_kernel()"
Cohesion: 0.31
Nodes (10): E, ElemType, FastDivmod, LinearView, N, Sequence, Tensor, Vector (+2 more)

### Community 318 - "BlockPartition"
Cohesion: 0.27
Nodes (4): BlockPartition, partition_edges_match_schedule(), Self, Vec

### Community 319 - "AmdDevice"
Cohesion: 0.22
Nodes (6): AmdDevice, Debug, DeviceId, Formatter, Result, Self

### Community 320 - "MetalDevice"
Cohesion: 0.22
Nodes (6): MetalDevice, DeviceId, Display, Formatter, Result, Self

### Community 321 - ".execute()"
Cohesion: 0.20
Nodes (8): ReduceDimRoutine, Config, EI, N, ReadWrite, SI, T, VirtualTensor

### Community 322 - "with_indices_validation.rs"
Cohesion: 0.38
Nodes (9): accepts_matching_outputs(), accepts_min_and_max(), rejects_indices_with_mismatched_shape(), rejects_indices_with_mismatched_strides(), rejects_operation_without_indices(), Result, Vec, try_launch() (+1 more)

### Community 323 - "StorageHandle"
Cohesion: 0.39
Nodes (4): debug, Self, StorageHandle, StorageUtilization

### Community 324 - "cmp_reference.rs"
Cohesion: 0.33
Nodes (8): Read, cmp_vs_reference(), max_diff(), read_arr(), D, Tensor, Vec, tensor2()

### Community 325 - "gather_kernel()"
Cohesion: 0.22
Nodes (9): gather_kernel(), ElemType, FastDivmod, I, LinearView, LinearViewMut, Sequence, T (+1 more)

### Community 326 - "select_kernel()"
Cohesion: 0.22
Nodes (9): ElemType, FastDivmod, I, LinearView, LinearViewMut, Sequence, T, Tensor (+1 more)

### Community 327 - "burn-fastblt Byte-Level BLT + FastBLT for Burn"
Cohesion: 0.22
Nodes (9): arXiv 2412.09871 BLT (Byte Latent Transformer, Meta 2024), burn-fastblt Byte-Level BLT + FastBLT for Burn, arXiv 2605.08044 FastBLT (Meta 2026), arXiv 2212.07525 data2vec 2.0 (Baevski et al., ICML 2023), burn-jepa JEPA (data2vec 2.0) for Burn, arXiv 2304.07193 KoLeo (Caron et al., DINOv2, 2023), arXiv 2511.08544 LeJEPA (Balestriero & LeCun, 2025), burn-mtp Multi-Token Prediction for Burn (+1 more)

### Community 328 - "burn-situ CI Workflow"
Cohesion: 0.31
Nodes (9): actions/checkout@v4, burn-situ crate, burn-situ check job (cargo check --no-default-features / --all-features), burn-situ CI Workflow, burn-situ clippy job (cargo clippy --all-features -- -D warnings), burn-situ feature matrix (no-default-features + all-features), dtolnay/rust-toolchain@stable, burn-situ test job (cargo test) (+1 more)

### Community 329 - "situ_glu()"
Cohesion: 0.50
Nodes (8): dev(), Tensor, situ_finite(), situ_glu(), situ_negative_gate_tail_vanishes(), situ_shape(), softcap(), softcap_bounded()

### Community 330 - "CudaRuntime"
Cohesion: 0.22
Nodes (6): CudaRuntime, DeviceId, Shape, Strides, TargetProperties, Vec

### Community 331 - "notify_profiled()"
Cohesion: 0.31
Nodes (4): Kept, notify_profiled(), ProfileDuration, TimingRequest

### Community 332 - "TunePlan"
Cohesion: 0.44
Nodes (6): Cleanup, GroupPlan, Planned, HashMap, Vec, TunePlan

### Community 333 - "Two-region DCLM filter (head trains, tail evaluates)"
Cohesion: 0.32
Nodes (8): ADR-0004: Staged context ladder to 1M with a long gate, Eval tail growth 30 MB -> ~500 MB (0.7% of corpus), Staged context ladder 512 -> 4k -> 32k -> 256k -> 1M, ADR-0010: Corpus v2 - the eval tail never trains, Rule: eval regions excluded at filter time, structurally, Two-region DCLM filter (head trains, tail evaluates), All pre-v2 BPB numbers are contaminated and not comparable, DataDecide / RegMix: small-scale ranking predicts the 1B winner

### Community 334 - "include_path()"
Cohesion: 0.43
Nodes (7): bf16, f16, cccl_include_path(), cuda_path(), include_path(), Option, PathBuf

### Community 335 - "bce_label_is_a_fresh_topk_of_the_current_scores()"
Cohesion: 0.46
Nodes (7): bce_label_is_a_fresh_topk_of_the_current_scores(), dev(), eff_k(), route(), Tensor, selects_exactly_k_per_position_with_a_floor_of_one(), softplus

### Community 336 - "dormouse loop facts read from code: h reset to h_ctx each "
Cohesion: 0.32
Nodes (8): Allocating Recurrent Compute in Looped Language Models arXiv 2608.18230, Depth itself does pay at ~15M params — specifically the mixer arm, The Ignition Is Real, and It Lives at the Readout arXiv 2608.03263, 2608.03263: 'Intermediates were never recoverable through the tied readout (relay 0.00)' — a caution for intermediate-iteration readouts, Iteration-k logits alone lm_head(norm(step_out_k)) — trained at every k under both arms, dormouse loop facts read from code: h reset to h_ctx each iteration, mean readout, per-iteration CE, iter_embed, set_depth exists, MixerLoop: repeat the GDN mixer, apply the dense FFN once — beats FullLoop on aggregate CORE at 15M, Prefix mean readout (1/k)·Σ_{n≤k} step_out_n — trained at every k under --rand-depth

### Community 337 - "bench_ops.rs"
Cohesion: 0.46
Nodes (7): sctconfig, bench_forward(), bench_from_dense(), bench_retract(), main(), ms(), Duration

### Community 338 - ".tr_execute()"
Cohesion: 0.29
Nodes (7): TransactionOps, TransactionPrimitive, TransactionPrimitiveData, CubeBackend, ExecutionError, Result, Self

### Community 339 - "cast_element()"
Cohesion: 0.25
Nodes (8): cast_element(), ElemType, I, LinearView, LinearViewMut, N, O, Vector

### Community 340 - "bool_cast_kernel()"
Cohesion: 0.25
Nodes (8): bool_cast_kernel(), B, ElemType, LinearView, LinearViewMut, N, T, Vector

### Community 341 - "conv_depthwise()"
Cohesion: 0.25
Nodes (8): conv_depthwise(), ConvOptions, ConvSetupError, CubeTensor, DepthwiseStrategy, N, Option, Result

### Community 342 - "scatter_nd_kernel()"
Cohesion: 0.25
Nodes (8): ElemType, FastDivmod, I, LinearView, Sequence, T, Tensor, scatter_nd_kernel()

### Community 343 - "JepaPredictor"
Cohesion: 0.36
Nodes (5): JepaPredictor, LayerNorm, Linear, Self, Tensor

### Community 344 - "burn-sct Spectral Compact Training"
Cohesion: 0.25
Nodes (8): Muon: An optimizer for hidden layers in neural networks (Jordan et al., blog post), burn-muon-plus Muon+ optimizer for Burn, arXiv 2602.21545 Muon+: Towards More Effective Muon via One Additional Normalization Step for LLM Pre-training (UCSB), burn-sct CI workflow, burn-rs/burn tensor linalg qr (Householder QR, MIT), burn-sct Spectral Compact Training, arXiv 2604.00733 Spectral Compact Training (Kohlberger, 2026), EctoSpace/SCT PyTorch reference implementation

### Community 345 - "cubek-reduce (CubeK Reduce)"
Cohesion: 0.32
Nodes (8): CI never compiles the extended tier; clippy guards it separately, CUBE_TEST_MODE (Correct | Strict | PrintAll | PrintFail), cubek-reduce (CubeK Reduce), cubek-test-utils (shared test-mode env owner), CUDA backend test target (cargo test --features cubecl/cuda), Feature tier 'extended' (full reduce_dim matrix), Feature tier 'full' (implies extended, adds benchmark catalogue), Feature tier 'heavy' (Cube reduction-strategy tests)

### Community 346 - "Think-at-Hard (TaH) arXiv 2511.08577 (Fu, You, Chen, Dai, "
Cohesion: 0.29
Nodes (7): Random-depth and adaptive depth are unmeasured for a byte-level 256-vocab LM (NOT VERIFIED), Depth-adaptive Inference of Looped LMs via Continuous Depth Batching arXiv 2608.09444, Layer dropout with rate increasing with depth + shared-exit early-exit loss, LayerSkip arXiv 2404.16710 (Elhoushi et al., Meta), Ouro-1.4B / Ouro-1.7B (looped LM, measured per recurrent step on GSM8K), TaH headroom is a percentage of TOKENS, not of loss: 12–19% of tokens iterate, shipped decider skips 93%, Think-at-Hard (TaH) arXiv 2511.08577 (Fu, You, Chen, Dai, Yang, Wang)

### Community 347 - "RequiredAddrType"
Cohesion: 0.43
Nodes (4): CubeTensor, Option<CubeTensor>, RequiredAddrType, AddressType

### Community 348 - "fused_recurrent_forward()"
Cohesion: 0.43
Nodes (5): fused_recurrent_forward(), fused_recurrent_gdn2(), Option, Tensor, test_chunk_matches_fused_with_real_decay()

### Community 349 - "bf16_matmul.rs"
Cohesion: 0.52
Nodes (6): bf16_act_f32_param_backward_finite(), bf16_mixed_matmul_is_finite(), bf16_model_size_matmul_autodiff_finite(), bf16xbf16_matmul_is_finite(), dev(), f32_control_backward_arrives()

### Community 350 - "infer.rs"
Cohesion: 0.67
Nodes (6): bench(), pack_ternary(), packed_matmul(), packed_matmul_par(), Vec, scaled_matmul()

### Community 351 - ".to_inference()"
Cohesion: 0.33
Nodes (3): retract_keeps_masters_tracked(), to_inference_matches_per_column_layer(), to_inference_matches_trained_layer()

### Community 352 - ".enumerate_all_devices()"
Cohesion: 0.52
Nodes (3): DeviceId, Result, Vec

### Community 353 - "PlaneReduceBlueprint"
Cohesion: 0.48
Nodes (5): IdleMode, GlobalReduceBlueprint, PlaneMergeStrategy, PlaneReduceBlueprint, UnitReduceBlueprint

### Community 354 - "engram_ab.sh"
Cohesion: 0.67
Nodes (5): gpu_busy(), probe(), run(), engram_ab.sh script, wait_for_gpu()

### Community 355 - "repeat_dim_kernel()"
Cohesion: 0.33
Nodes (6): repeat_dim_kernel(), E, ElemType, FastDivmod, Sequence, Tensor

### Community 356 - "mask_fill_auto()"
Cohesion: 0.67
Nodes (5): mask_fill_auto(), mask_where_auto(), CubeTensor, DType, InputScalar

### Community 357 - "random_uniform()"
Cohesion: 0.47
Nodes (6): random_like_uniform(), random_uniform(), CubeDevice, CubeTensor, DType, Shape

### Community 358 - "burn-rope Rotary Position Embedding with YaRN"
Cohesion: 0.33
Nodes (6): burn-nope No Positional Encoding for Burn, arXiv 2607.24653 Kimi K3 (Moonshot, 2026), burn-rope CI workflow, burn-rope Rotary Position Embedding with YaRN, arXiv 2104.09864 RoFormer (Su et al., 2021), arXiv 2309.00071 YaRN (Peng et al., 2023)

### Community 359 - "burn-swiglu (SiLU-Gated Linear Unit for Burn)"
Cohesion: 0.33
Nodes (6): Burn 0.22 framework, burn-swiglu (SiLU-Gated Linear Unit for Burn), burn-swiglu GitHub Actions CI (badge), GLU Variants Improve Transformer (arXiv 2002.05202), NdArray (CPU) backend generic use, SwiGLU FFN as the default LM feed-forward activation

### Community 360 - "burn-ttt (Test-Time Training loss for Burn)"
Cohesion: 0.33
Nodes (6): burn-ttt (Test-Time Training loss for Burn), burn-ttt GitHub Actions CI (badge), Constant latency regardless of context length, TTT-E2E inference path out of crate scope, TTT-E2E (arXiv 2512.23675), Masked MSE TTT loss over a sliding context window

### Community 361 - "CpuDevice"
Cohesion: 0.40
Nodes (3): CpuDevice, DeviceId, Self

### Community 362 - ".generate()"
Cohesion: 0.47
Nodes (4): Func, At, I, K

### Community 363 - "GatedStream"
Cohesion: 0.40
Nodes (4): GatedEvent, GatedStream, Arc, AtomicBool

### Community 365 - "BufReader"
Cohesion: 0.40
Nodes (4): BufReader, File, Source, ParquetRecordBatchReader

### Community 366 - "fence.rs"
Cohesion: 0.40
Nodes (4): device_events, drop_queue, EventFence, A

### Community 368 - "cross_kernel()"
Cohesion: 0.40
Nodes (5): cross_kernel(), E, ElemType, LinearView, LinearViewMut

### Community 369 - "ManagedResource"
Cohesion: 0.50
Nodes (3): ManagedResource, ManagedResource<Resource>, Resource

### Community 370 - "GcTask<B>"
Cohesion: 0.40
Nodes (3): GcTask<B>, Self, T

### Community 371 - "time_ms()"
Cohesion: 0.67
Nodes (3): main(), F, time_ms()

### Community 372 - "mor_ab.sh"
Cohesion: 0.83
Nodes (3): run(), mor_ab.sh script, wait_for_gpu()

### Community 373 - "mask_indices()"
Cohesion: 0.67
Nodes (3): mask_indices(), Bool, Tensor

### Community 376 - "ste_ternary()"
Cohesion: 1.00
Nodes (3): ste_ternary(), ste_ternary_is_ternary_values(), ternarize()

### Community 378 - "TuneRecord<K>"
Cohesion: 0.67
Nodes (3): Record, TuneRecord<K>, .KIND

## Ambiguous Edges - Review These
- `LOC cut scenarios A-D and the ~5.5-6k floor` → `Loop runs at fixed max_iter=4 with honest unweighted CE`  [AMBIGUOUS]
  docs/design-minimal.md · relation: conceptually_related_to
- `burn-kda Kimi Delta Attention (KDA) for Burn` → `FlashKDA CUTLASS reference path (Moonshot)`  [AMBIGUOUS]
  vendor/burn-fused/crates/burn-kda/README.md · relation: references
- `Adaptive depth for looped models (variable compute per input)` → `Arrabal-Campos arXiv 2608.22347`  [AMBIGUOUS]
  research/2026-09-27-adaptive-depth-safe.md · relation: conceptually_related_to
- `Adaptive depth for looped models (variable compute per input)` → `Mixture-of-Depths arXiv 2404.02258`  [AMBIGUOUS]
  research/2026-09-27-adaptive-depth-safe.md · relation: conceptually_related_to
- `Adaptive depth for looped models (variable compute per input)` → `PALBERT arXiv 2204.03276`  [AMBIGUOUS]
  research/2026-09-27-adaptive-depth-safe.md · relation: conceptually_related_to
- `Adaptive depth for looped models (variable compute per input)` → `TIDE arXiv 2603.21365 (token-level early exit)`  [AMBIGUOUS]
  research/2026-09-27-adaptive-depth-safe.md · relation: conceptually_related_to
- `Adaptive depth for looped models (variable compute per input)` → `Less is More: Recursive Reasoning with Tiny Networks (TRM) arXiv 2510.04871`  [AMBIGUOUS]
  research/2026-09-27-adaptive-depth-safe.md · relation: conceptually_related_to
- `ADR-0013 fixed depth (PonderNet halting deleted after two measured collapses)` → `DeepLoop arXiv 2607.13491`  [AMBIGUOUS]
  research/2026-09-27-adaptive-depth-safe.md · relation: semantically_similar_to
- `ADR-0013 fixed depth (PonderNet halting deleted after two measured collapses)` → `Training-Free Looped Transformers arXiv 2605.23872`  [AMBIGUOUS]
  research/2026-09-27-adaptive-depth-safe.md · relation: semantically_similar_to

## Knowledge Gaps
- **251 isolated node(s):** `cublas-poc`, `OP_N`, `OP_T`, `R_32F`, `R_16F` (+246 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 2166 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **20 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `LOC cut scenarios A-D and the ~5.5-6k floor` and `Loop runs at fixed max_iter=4 with honest unweighted CE`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `burn-kda Kimi Delta Attention (KDA) for Burn` and `FlashKDA CUTLASS reference path (Moonshot)`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `Adaptive depth for looped models (variable compute per input)` and `Arrabal-Campos arXiv 2608.22347`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Adaptive depth for looped models (variable compute per input)` and `Mixture-of-Depths arXiv 2404.02258`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Adaptive depth for looped models (variable compute per input)` and `PALBERT arXiv 2204.03276`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Adaptive depth for looped models (variable compute per input)` and `TIDE arXiv 2603.21365 (token-level early exit)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Adaptive depth for looped models (variable compute per input)` and `Less is More: Recursive Reasoning with Tiny Networks (TRM) arXiv 2510.04871`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._