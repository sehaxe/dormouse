# What OLMo-core does that the port should REIMPLEMENT ITSELF

**Date:** 2026-10-17. **Read-only:** one file written; no code change, no GPU, nothing committed.
**Ask:** mine `allenai/OLMo-core` for IDEAS a single-GPU torch port should build its own way.
Nothing below is "copy file X" — every row is a mechanism, its citation, and our own sketch.
**Scope filter:** all multi-GPU machinery (FSDP/HSDP/NCCL/EP all-to-all, DeviceMesh, symmetric
memory) is out by the brief and appears only in the SKIP list.

## 0. What was read, and what was not

Fetched from `raw.githubusercontent.com/allenai/OLMo-core/main`, plus `microsoft/dion/main` —
**because OLMo-core ships no optimizer math**: `optim/muon.py:24-32` and `optim/dion.py:25-34`
raise an ImportError unless the external `dion` package is installed, and the two files are
config + parallel-mesh wrappers only. The algorithms are in `dion/`.

| read whole | read in parts (as named below) | **not read** |
|---|---|---|
| `optim/muon.py` (237) · `optim/dion.py` (176) · `optim/skip_step_optimizer.py` (116) · `AGENTS.md` (160) · `dion/muon.py` (378) · `dion/dion_simple.py` (226) · `dion/cuda_graph.py` (313) · `kernels/cuda/moe_unpermute_bwd_kernel.cu` (434) · `kernels/moe_permute_drop.py` (148) · `kernels/moe_unpermute_bwd.py` (166) | `README.md` (first 90) · `kernels/moe.py` (head + def list) · `kernels/moe_chunk_reorder.py` (head, dispatch, test) · `kernels/swiglu.py` (def list) · `kernels/grouped_mm.py` (def list) · `kernels/grouped_mm_row_offset.py` (head + signature) · `kernels/mxfp8_linear.py` (head + greps) · `nn/layer_norm.py`, `nn/moe/router.py` (greps) · `nn/moe/loss.py` (head) · `train/checkpoint.py` (:421-470, :497-560) · `train/callbacks/*` (interface + one monitor) | `kernels/mxfp8_utils.py` (2864) · `optim/moe_optimizer.py` (2468, grepped) · `train/trainer.py` (69 KB) · `dion/dion2.py`, `megabatch_base.py`, `nordion2.py` · `kernels/cuda/*.cu` other than unpermute_bwd · `nn/transformer/*` |

## 1. Ranked table

`P0` = do it next; `P1` = worth a lane; `P2` = arm behind a flag; `skip` = reason given in §3.
LOC is the sketch's size in **our** files, not theirs.

| # | idea | what OLMo-core does (file:line) | why it matters for US | our-implementation sketch | P |
|---|---|---|---|---|---|
| 1 | **CUDA-graph capture of the optimizer stage, with every hyperparameter a live device tensor** | `dion/cuda_graph.py:1-79` (rationale: "hundreds of small host ops per step… the step is a fixed-shape, repeated computation — an ideal CUDA-graph target"); `:257-274` capture ONE step after `warmup_steps`, replay it immediately; `:88-116` `capture_error_mode="thread_local"`; `:276-285` `_sync_host_hyperparams`; `:170-181` forces `zero_grad(set_to_none=False)`; `:231-255` refuses to capture while any `requires_grad` param has `.grad=None` | our own b1 profile: the optimizer is **44-53 % of the step and the step is host-dispatch-bound** (`docs/research/2026-10-17-muon-variants.md` TL;DR, `docs/reviews/2026-10-16-b1-profile.md` §1/§3). The port has no graph capture anywhere. AI2's answer to *exactly* our documented blocker (a capture window refuses stream reads/handle writes, `AGENTS.md` §3.1) is: make what the step reads a persistent device tensor filled in place, so the graph re-reads it | new `port/src/dormouse/train/graph.py` (~100: `CudaGraphMuon` wrapper — warmup, capture, `release`, `_check_params_have_grads`, device-tensor `lr`); `optim.py` — `batched_step_from_model` must read `lr` from a device tensor when one is set (~25); `loop.py` — `zero_grad(set_to_none=False)` on the captured path + `release()` before teardown; `config.py` flag | **P0** |
| 2 | **Fuse the MoE combine backward: one pass over `grad_output` producing BOTH the activation scatter and the router-probability gradient, off a direct int32 row map** | `kernels/cuda/moe_unpermute_bwd_kernel.cu:127-226` (one block per token; `float4` loads; `__ldlu` last-use reads of `input_fwd`; `float accum[topk_tile]` register array; `__shfl_xor_sync` reduction; `prob_grad[token,k]` written by `tid==0`); `:269-371` launcher with `topk_tile` dispatch 8/16/32/64/128; `:394-404` `act_grad` from a passed-in `out` buffer, `keep_mask` → zero rows; `:375-434` entry | our MoE backward is the measured weak spot (fwd+bwd 65.4 ms, **1.57× slower than the padded arm**, `moe_kernels.py:1-11`). Our combine applies the router weight in fp32 and scatters with `index_add_` (`experts.py:635-651`), so `dprobs` costs torch's autograd an extra `[R,f]` read + reduction; our backward reconstructs segments with `torch.repeat_interleave` (`moe_kernels.py:475-479`) and computes `dx` as **two** grouped GEMMs plus `add_` (`:590-591`) although the FORWARD already fuses both `tl.dot`s into one kernel (`:193-194`) | `moe_kernels.py` — (a) one `_grouped_glu_dx` kernel with two accumulators (the forward's own pattern, ~35), (b) accept a `row_id_map [n·k] int32` with `-1` = dropped, built in `experts.py::_permute` where `order` already is the sorted row index (~15 there, ~20 kernel), (c) a combine-backward kernel: scatter + warp-reduced `dprobs`, tile 8 for topk 2 (~60) | **P0** |
| 3 | **Directory-atomic checkpoints + a JSON metadata sidecar** | `train/checkpoint.py:542-560` `_temporary_wd` (yield a temp dir, then teardown); `:497-540` `_teardown_tmp_dir` → `tmp_dir.replace(str(dir))` — the whole checkpoint appears by one rename; `:437-470` `_prepare_dir` refuses a non-empty dir (`FileExistsError`) unless `save_overwrite`; `:433-435` `_save_metadata` writes a JSON metadata file | we already do **file-level** atomicity (`loop.py:168-178` `tmp.replace(weights)`, sidecar `:185-187`) but the *pair* is not atomic, which is our open defect **ADR-0021 item 7**: "a crash between the model write and the sidecar write leaves a mismatched pair that nothing can detect". A temp *directory* makes the whole artifact set appear at once — the structural fix, and the metadata file carries the `train_step` stamp that defect asks for | `loop.py::save_ckpt` (~40): write weights + `.ngram` + `meta.json` (train_step, config sha, engram slots) into `<ckpt_name>.tmp/`, then `tmp_dir.replace(target)`; `load_ckpt` refuses a missing/mismatched stamp | **P1** |
| 4 | **`SkipStepOptimizer`: a device-tensor step factor from a rolling mean + 6σ of loss AND grad-norm, sharing one spike definition with the monitor** | `optim/skip_step_optimizer.py:18-31` ("avoid host-device syncs… use `get_step_factor()`"); `:85-109` `@torch._dynamo.disable() get_step_factor` → `logical_and(loss_ok, grad_norm_ok).float()`; `:37-38` defaults `rolling_interval_length=128, sigma_factor=6`; `train/callbacks/stability_monitor.py:22-38` uses the **same** 128 / 6.0 constants; `optim/moe_optimizer.py:974-1004` shows the production optimizer consuming `latest_loss`/`latest_grad_norm` | our NaN firewall is unconditional and our stress protocol *counts* spikes (201-step median + 0.1, `AGENTS.md` §3.5 item 6) but the trainer never skips a non-NaN spike — and the sigma factor is a device tensor, which is our ADR-0018 rule 2 exactly. One definition feeding both the log and the action is the part we lack | `optim.py` — a `step_factor` device scalar multiplied into both update paths (`p.add_(..., alpha=-lr*scale*factor)`, ~25); `loop.py` — a device ring of loss/grad-norm feeding `opt.latest_*` (~30); `config.py` 2 knobs. Gate: a synthetic +100σ loss ⇒ factor 0 ⇒ params bit-identical | **P1** |
| 5 | **`out=` / grow-only workspace buffers owned by the caller** | `kernels/moe_permute_drop.py:11-70` (`_PermuteDropWorkspace` + `_WORKSPACE_BY_DEVICE`, grow-only: `if ws.max_expanded_rows >= expanded_rows: return ws`); `kernels/moe_unpermute_bwd.py:106-163` `out` parameter; `kernels/grouped_mm.py:64-88` `_check_out_buffer` / `_check_input_grad_out_buffer`; the gate idiom in `src/test/kernels/moe_chunk_reorder_test.py` — `assert out.data_ptr() == out_buffer.data_ptr()` | our kernels allocate per call (`moe_kernels.py:359-367, 496, 529`) — the b1 backward alone allocates ~12 tensors, and `dv` at 1024 experts × 64 × 5461 fp32 is ~1.4 GB. Our docstring bans *module-level* state ("no allocator, no graph, no workspace", `:17-18`), which this pattern respects: the cache lives in the CALLER and is passed in | `moe_kernels.py` — optional `out=`/`ws=` on the public fns, asserted for shape/dtype/contiguity, LOUD on mismatch (~50); `experts.py` — one grow-only workspace per (device, max_rows), **sized by the dropless bound R, never by `cap`** (~30) | **P1** |
| 6 | **Split fused weights into per-block orthogonalization with a per-block LR rescale** | `dion/muon.py:50-55` (docstring: orthogonalize row blocks of a fused 2D weight independently — "e.g. a fused QKV projection as separate Q, K, and V blocks, which may have unequal sizes under GQA… while the model keeps the single wide GEMM"); `:157-166` `compute_split_lr_scales(split_sizes, shape, adjust_lr)`; `:244-260` the rescale "converts the whole-matrix adjusted_lr into the per-block adjustment that separate parameters would receive" | our `AGENTS.md` §3.5 item 3 names this as an open delta ("fusing mixes singular directions"), and `2026-10-17-muon-variants.md` §3 rejected per-head NS because it **changes the update magnitude** (~3.5× at 12 heads). The missing piece is exactly the per-block rescale, which turns it from an LR retune into a magnitude-preserving change | `optim.py` — a `split_sizes` tuple per group; run NS on `mom[..., a:b]` per block and scale each by the same `max(1, m/n)^0.5` rule applied to the BLOCK's shape (~50); `config.py` a per-preset map. Gate: a `[2,6]` matrix split `[2,3]+[2,3]` must match two separate params to the bit | **P1** (only if a fused projection is in our Muon group) |
| 7 | **`torch._assert_async` on the momentum/grad path: a device-side finiteness abort with no host sync** | `optim/moe_optimizer.py:62-72` — "Device-side assert… without a host sync (so it's cheap enough to run every step). The process aborts asynchronously if a non-finite value is encountered" | our firewall *skips* a step and we learn about a surviving failure only when the loss is read on log cadence (`AGENTS.md` §3.8: the raw loss is copied before the mask, on log cadence only). An async assert on the momentum buffer after the sanitizer would abort at the exact step instead of the next log line | `optim.py` — one `torch._assert_async(isfinite(mom).all(), ...)` after `_momentum_update_` in both paths (~10); it is an ADDITIONAL guard, never a replacement for the firewall | **P1** |
| 8 | **Router z-loss + input jitter as named config arms** | `nn/moe/router.py:189,204,241-243` (`z_loss_weight`, `_z_loss` buffer); `nn/moe/loss.py:1-60` `load_balancing_loss` + `MoELoadBalancingLossGranularity` (local-batch vs instance); `router.py:28` imports `router_z_loss`; `:200` `bias_gamma` = DeepSeek-v3 aux-loss-free bias | we have the aux-loss-free bias (LFB, `experts.py:439-455`, measured correction included) and renormalized top-k, but no z-loss and no jitter. Cheap arms, one each | `experts.py` — `z_loss_weight` term on `logits.logsumexp(-1)`, `jitter_eps` on the router input; `config.py` 2 keys (~25) | **P2** |
| 9 | **A callback seam with per-callback `state_dict`, and a stability monitor as a callback** | `train/callbacks/callback.py:15-175` — `priority: ClassVar[int]` (`:24`), hooks `pre_step`/`pre_optim_step`/`post_step`/`post_checkpoint_saved`/`on_error`/`close` (`:103-173`), `state_dict`/`load_state_dict` on every callback (`:47-57`); `callbacks/__init__.py` ships two dozen of them — 24 names ending in `Callback`, incl. `speed_monitor`, `gpu_memory_monitor`, `gap_monitor`, `stability_monitor`, `profiler`, `sequence_length_scheduler` | our `train()` is one ~370-line function (`loop.py:433-799`) with no hook seam, and our StressMonitor state is **lost on resume** (ADR-0021 item 4). A per-callback state_dict is the structural fix — but a 20-hook framework is scaffolding we do not need | `loop.py` — TWO named hooks only (`post_step`, `on_error`/`close`) plus `to_bytes`/`from_bytes` for the stress ring (~40). Add hooks when a second consumer exists | **P2** |
| 10 | **Every norm variant is a named enum arm with a default** | `nn/layer_norm.py:26-61` `LayerNormType` (incl. `fused_rms` `:51`), then `RMSNorm` `:267` · `QwenRMSNorm` `:296` · `NemotronRMSNorm` `:322` · `CuTeRMSNorm` `:344` · `FusedRMSNorm` (Triton) `:398` · `L2Norm` `:461` | our fused RMSNorm kernel "never produced a number on ANY device in the life of this project" and is still unreachable (`AGENTS.md` §3.3). AI2's pattern — the fast path is a config arm beside the reference, not a hidden branch — is the shape our own §1.1 LOUD/COUNTED doctrine asks for | `model.py`/`config.py` — a `norm_impl = fused\|eager` arm + a parity gate; if the gate cannot be written, delete the kernel | **P2** |

## 2. Top-3 expanded

### 2.1 CUDA-graph capture of the optimizer stage, driven by live device hyperparameters

**Mechanism.** An optimizer step is a fixed-shape, repeated device computation; on a
dispatch-bound step the host enqueue is the cost. So: run `warmup_steps` eager steps (they
allocate the state, cuBLAS workspaces and — in a distributed run — NCCL buffers, `:96-101`),
then capture exactly ONE `optimizer.step()` into a `torch.cuda.CUDAGraph` and **replay it
immediately** (capture only records; without the replay the trajectory would be one step short,
`:257-274`). From then on every step is `graph.replay()` (`:304-306`).
**The load-bearing trick for our blocker:** the captured graph can only read what is a *device
tensor* at capture time. So the learning rate **is** `group["lr"]` as a device tensor — "a
`torch.optim` LR scheduler fills the tensor in place outside the graph and every replay re-reads
the live value" (`:70-78`) — and `weight_decay` may opt in by being passed as a `Tensor`
(`:39-50`). Anything read as a host float is **baked in at capture and stops changing**; the
wrapper raises rather than let a late opt-in look like it took effect (`:216-229`).
**The trap they document and we would hit:** the participating parameter set freezes at capture,
and a param whose `.grad` is still `None` (starved by the schedule) silently stops being updated
forever — so capture REFUSES while any `requires_grad` param lacks a grad (`:231-255`).
**Our version.** Wrap our `Muon`/`MixOptimizer` exactly this way and capture **the optimizer
stage only** — never the step, because our step reads the loss scalar for the firewall and the
log line, and a capture window refuses that read. Two things must change first: our `lr`/`mu`
are host floats in the group dict (`optim.py:704`) and our loop calls
`zero_grad(set_to_none=True)` (`loop.py:665,694`), which swaps `.grad` for a fresh tensor every
step and breaks the pinned buffers — flip to `set_to_none=False` on the captured path. The `sf`
arm's `_sf_ckp` recursion is deliberately HOST arithmetic (`optim.py:895-905`); under a graph it
must move on-device or the SF arm cannot be captured (a reason to capture the default arm
first). Gate: `replay` must be **bit-identical to an eager step** for a constant LR and a
scheduled LR, which is precisely the gate they wrote (`dion/cuda_graph.py:77-78`).

### 2.2 The fused combine backward: one pass, two gradients, an int32 row map

**Mechanism.** The unpermute forward computes `out[token] = Σ_k input_fwd[row_k]·p_k`. Its
backward is `d_act[row_k] += grad_out[token]·p_k` **and**
`d_p[token,k] = ⟨grad_out[token], input_fwd[row_k]⟩`. OLMo-core computes both in ONE kernel:
one block per token, `grad_out` read once into registers as `float4` fragments
(`:160-167`), then per k: multiply by the cached `p_k` from shared memory and store the row
(`:181-193`), and accumulate `grad_frag[e]·input_val[e]` into a per-k register
`accum[k]` (`:195-201`) — with `input_fwd` read through `__ldlu` (a last-use hint, so the
line is dropped after the read). The reduction is a warp `__shfl_xor_sync` butterfly
(`:206-215`) and one thread writes the row (`:217-224`). `accum[k]` is sized by a **constexpr
`topk_tile`** with a dispatch over {8,16,32,64,128} (`:290-354`), so no dynamic allocation
exists in the hot path. Dropped rows are skipped by `dest_row < 0` (`:177-179`) — a single
int32 map with `-1` marks them, no segment reconstruction anywhere.
**Our version, three pieces:**
(a) `dx = dg@gate_uᵀ + du@up_uᵀ` is TWO grouped GEMMs plus `add_` in our backward
(`moe_kernels.py:590-591`) while our forward already accumulates both `tl.dot`s in one kernel
and combines them in registers (`:193-209`) — add the twin kernel with two accumulators;
(b) build the `row_id_map [n·k] int32` once in `_permute`, where `order` already IS the sorted
row index and `offsets`/`pos` are computed anyway (`experts.py:475-503`), and delete
`_segments`'s `torch.repeat_interleave` (`moe_kernels.py:475-479`);
(c) one combine-backward kernel (scatter + warp-reduced `dprobs`, topk tile 8 for our top-2)
replacing the autograd path over `y.to(float32) * flat_w[rows]` + `out.index_add_`
(`experts.py:635-651`). Parity gate: the fused backward against the current autograd one on
saved b1 buffers, at the same tolerance our existing padded-vs-ragged gate uses.

### 2.3 Directory-atomic checkpoints + a metadata sidecar

**Mechanism.** `_temporary_wd` is a context manager: `_prepare_dir` refuses a non-empty target
directory unless `save_overwrite` is set (`:437-470`), the body yields a **temp directory** that
every rank writes into, and on exit `_teardown_tmp_dir` does `tmp_dir.replace(str(dir))` — one
rename publishes the whole checkpoint (`:497-540`). A crash mid-write leaves only a temp dir;
the target never exists in a half-written state, and `dir_is_checkpoint`/`latest_checkpoint`
(`:335-402`) can trust the directory's existence. Metadata is a separate JSON file written by
rank 0 (`:433-435`).
**Our version.** We already write each file atomically (`loop.py:168-178`,
`tmp.replace(weights)`; sidecar `:185-187`) — the *pair* is the gap, and it is our documented
open defect ADR-0021 item 7. Concretely: write `<ckpt>.pt`, `<ckpt>.ngram` and `meta.json`
into `<ckpt_name>.tmp/`, then rename the directory. `meta.json` carries the `train_step` stamp
that item 7 asks for, so a mismatched pair becomes detectable rather than silent. **The honest
blocker, which is ours not theirs:** the on-disk layout change invalidates an 18 GB production
artifact, which is why ADR-0021 deferred it — so the cheap half (stamp + refusal) can land
first and the directory move wants its own decision.

## 3. SKIP list

- **MXFP8 — sm_120-incompatible, do not port.** Their path is PyTorch's cuBLASLt MXFP8
  dispatch: `torch._scaled_mm` with `ScalingType.BlockWise1x32` and
  `SwizzleType.SWIZZLE_32_4_4` (`kernels/mxfp8_linear.py:14-18,44-46`), plus
  `aten::_scaled_grouped_mm_v2` (`kernels/scaled_grouped_mm.py:222`). PyTorch's own PR
  #147548 adds "blockwise MXFP8 support to `torch._scaled_mm` on **CUDA capability 10.0 and
  higher** devices", and PyTorch's test suite carries `mx_skip_msg = "MX gemm is only supported
  on CUDA capability 10.0+"` and "MXFP8 grouped GEMM… USE_MSLK=1 on SM100+". NVIDIA's
  Transformer-Engine support matrix lists **MXFP8 = No on 12.0** ("not yet supported on compute
  capability 12.0 and higher… limited to 10.0 through 10.x"), and TE issue #2668 adds the sharp
  part: "while the MXFP8 forward pass works on SM120, backward pass currently does not" (cuBLAS
  lacks non-TN layouts). The silicon can do it (the `mxf8f6f4.block_scale` PTX targets `12.0a`),
  the library cannot. Our port's fp8 is factor *storage*, a different mechanism that needs no
  MXFP8 MMA. Note also that we could not find any arch guard in their Python — the failure would
  be a runtime error from `_scaled_mm`, not a clean refusal.
- **Multi-GPU / NCCL / EP / symmetric memory** — `optim/muon.py:162-201`, `optim/dion.py:104-157`,
  `kernels/nccl_rma_p2p.py`, `kernels/olmo_symm_mem_*`, the `distributed_mesh` arguments
  throughout. Out of scope by the brief (single GPU, no NCCL).
- **`grouped_mm_row_offset`** (`kernels/grouped_mm_row_offset.py:20-35,147-154`) — a CUTLASS
  kernel needing `OLMO_CUTLASS_ROOT` or a hardcoded `/workspace/ref/pytorch/third_party/cutlass`,
  with a required `out` and per-group `row_start`+`batch_sizes`. New native dependency, sm_90+/
  SM100 targets, and our `offs[E+1]` Triton kernels already cover ragged groups.
- **`kernels/moe.py`'s `@triton.autotune`** (`:28-40`) — the padded-bank gather/scatter algebra
  is our padded arm already (and the arm we are trying to beat); the autotune re-tunes when a
  shape key moves, which our own house rule bans for a dispatch-bound loop
  (`moe_kernels.py:59-65`). Our fixed tile constants are the better choice here.
- **Fused SwiGLU (`kernels/swiglu.py:166,270`)** — their standalone Triton SwiGLU (with a
  `valid_prefix` variant for packed buffers) is one *extra* pass; we fuse the GLU into the
  second grouped GEMM's input through registers (`moe_kernels.py:207-233`). **We do better.**
- **`output_discard_checkpoint.py`** — a `load_inline` C++ `share_storage` extension with a
  threading lock and a pure-Python fallback. A native dep for a checkpointing refinement;
  `BalancedCheckpointing` plus our `moe_offload` already cover the need.
- **Full-rank Dion for the `[2048,2048]` projections — analyzed, not recommended.**
  `optim/dion.py:65-66` defaults `rank_fraction=1.0` ("Set to 1.0 for full-rank optimization"),
  so OLMo-core's *default* Dion is full-rank: `dion/dion_simple.py:17-64` shows the algorithm —
  `M += G`; `P = M@Q`; `P ← qr(P)` in fp32; `R = Mᵀ@P`; error feedback
  `M -= (1-μ)·P@Rᵀ`; `Q ← colnorm(R)`; `W -= lr·√(fan_out/fan_in)·P@Qᵀ`. At full rank that is
  1 QR + 3 matmuls against our NS's 10 matmuls, so the raw-FLOP case is not hopeless — but
  (i) `torch.linalg.qr` on a `[2048,2048]` fp32 matrix is a different kernel family from the
  cuBLAS GEMMs the rest of the step runs on, and **I have no measurement** of it on this card
  (SPECULATION: the arithmetic suggests ~2.7 matmul-equivalents, unmeasured); (ii) it is a
  different update rule, so it needs its own 3-seed A/B against ColRow, which is our largest
  measured win; (iii) `2026-10-17-muon-variants.md` §3 already rejected the Dion family for our
  dispatch-bound step, and our `[d,d]` projections are not in the Muon group at all
  (`AGENTS.md` §2.3). Revisit only if a `[d,d]` arm is wanted for another reason.
- **The 5-triple NS coefficient schedule** (`dion/muon.py:356-362`, five different `(a,b,c)`
  triples) — **already decided against, and measured**: our port runs ONE Jordan triple for all
  5 steps, and `AGENTS.md` §3.7 records that the fixed triple beat the schedule at 3 iterations
  at every condition number tested. Worth recording as a *fact*: AI2's production path uses the
  per-iteration table, so our divergence is deliberate. (Note their NS also runs in bf16 with an
  fp32 Frobenius prescale and a transpose when `rows > cols`, `:364-369` — the same choices our
  `ns_dtype` rung and `zeropower_via_ns` orientation already make.)
- **`optimizer_states` offload** — `optim/moe_optimizer.py:951-959` is `raise
  NotImplementedError()` with a comment. **Not an AI2 practice**; do not cite it as one.
  (Tempting for our 16 GB card, since the b1 Muon momentum for the `[1024,64,5461]` group alone
  is ~1.33 GiB — but SPECULATION: a D2H+H2D round trip of that per step is ~0.2 s at PCIe
  bandwidth, i.e. it would only pay if overlapped, and I have no measurement.)
- **`nn/layer_norm.py`'s CuTe variant** (`:344`) — CuTeDSL, datacenter-Blackwell oriented; our
  own fused RMSNorm already exists (the problem there is that it never runs, see table row 10).
- **`nn/moe/router.py`'s `bias_gamma`** (`:200`) — we already run LFB with a *measured* cumulative
  integrator (`experts.py:439-455`); nothing to take.

## 4. The meta-lesson: what a serious lab's codebase does better than ours

Structurally, three things stand out, and none of them is a kernel. **(1) Every optimisation is
opt-in behind a named arm, with the reference implementation kept beside it** — six norm types in
a config enum (`nn/layer_norm.py:26-61`), a Muon config with `cautious_wd`/`nesterov`/`flatten`/
`adjust_lr` as fields (`optim/muon.py:66-92`), a MoE permute with an explicit
`backend: Literal["cuda","triton"]` argument and a torch reference next to the CUDA kernel
(`kernels/moe_chunk_reorder.py:211-229`, whose test file asserts the reference and the buffer
identity). Their codebase is *readable as a menu of arms* — which is exactly our §1.1/§1.2
doctrine, and it is why "what do they do differently" has an answer at all. **(2) The expensive
invariants are enforced mechanically rather than documented**: the CUDA-graph wrapper refuses to
capture when a parameter has no gradient (`dion/cuda_graph.py:231-255`) and refuses when a
hyperparameter opts into being live *after* capture (`:216-229`), instead of explaining the
failure in a comment; `_check_inputs` in every kernel wrapper is a wall of `ValueError`s
(`kernels/moe_unpermute_bwd.py:27-103`) whose cost is paid once per call. Our port documents these
traps superbly — in prose, in docstrings, in `AGENTS.md` — and prose does not fail a run.
**(3) Observation is a first-class subsystem**: 26 callbacks with `priority` ordering, each with
its own `state_dict`, covering speed, GPU memory, gaps, stability spikes and profiling
(`train/callbacks/__init__.py`). We have the measurements (arguably better ones — our
retraction discipline is stricter than theirs: `AGENTS.md` §3.2 versus a README with no
retraction section), but they live in logs and documents rather than in resumable, ordered
objects. **What we do better and should keep:** counted/loud fallbacks as doctrine, per-claim
provenance with file:line and dates, and the `out=`-free "caller owns the buffers" rule that
keeps the kernels stateless. **One defect of theirs we should not import:** their "triton"
backend dispatches to `_chunk_permute_torch` (`kernels/moe_chunk_reorder.py:223-224`) — a name
that disagrees with the code it runs, the class of defect our §1.7 treats as a defect in its own
right.

## 5. VERIFIED (sourced)

- OLMo-core ships **no Muon/Dion math**: `optim/muon.py:24-32`, `optim/dion.py:25-34` import the
  external `dion` package; both files are config/mesh wrappers. The algorithms are
  `dion/muon.py` and `dion/dion_simple.py`.
- Their Muon routing (`optim/muon.py:99-119`): matrix→Muon, embedding→AdamW (wd=0), vector→AdamW,
  lm_head→AdamW; defaults `lr=0.01, mu=0.95, betas=(0.9,0.95), weight_decay=0.1, nesterov=False,
  adjust_lr=rms_norm` (`:60-92`); `torch._dynamo.config.recompile_limit` raised to 16 for the
  many grad shapes (`:204-211`).
- The CUDA-graph pattern and all its stated requirements/caveats: `dion/cuda_graph.py:1-79` and
  the code at `:88-313`.
- The fused unpermute-backward kernel and its launcher: `kernels/cuda/moe_unpermute_bwd_kernel.cu`
  (:127-226 kernel, :269-371 launcher, :375-434 entry).
- The grow-only per-device workspace: `kernels/moe_permute_drop.py:11-70`.
- The skip-step statistic and its shared constants: `optim/skip_step_optimizer.py:12-116`,
  `train/callbacks/stability_monitor.py:22-38`.
- The device-side async finiteness assert: `optim/moe_optimizer.py:62-72`.
- Directory-atomic checkpointing: `train/checkpoint.py:437-470, 497-560`.
- `offload_optimizer_states` is a `NotImplementedError` stub: `optim/moe_optimizer.py:951-959`.
- MXFP8 is compute-capability-10.0+ in PyTorch's own gate (PR #147548; `mx_skip_msg` in
  `test/test_scaled_matmul_cuda.py`) and **No on 12.0** in NVIDIA's TE support matrix, with TE
  issue #2668 stating the SM120 *backward* is unsupported.
- Our side: `optim.py:167,294-414,620-704,895-905`; `moe_kernels.py:17-18,59-65,85,193-209,
  475-485,554-595`; `experts.py:439-455,475-503,635-651`; `loop.py:168-187,433,665,694`;
  `AGENTS.md` §2.3, §3.1, §3.3, §3.5 items 3 and 6, §3.7, ADR-0021 items 4 and 7;
  `docs/research/2026-10-17-muon-variants.md` TL;DR + §3 + §5.

## 6. SPECULATION (my inference, no direct measurement)

- **SPECULATION — the payoff of §2.1 on our card.** No graph capture exists in the port and I ran
  no GPU. Based on the port's own profile (optimizer 44-53 % of a dispatch-bound step,
  `2026-10-16-b1-profile.md` §1/§3) and AI2's claim that capture "collapses the host dispatch to a
  single graph launch (GPU time unchanged)" (`dion/cuda_graph.py:7-9`), I assume the optimizer
  stage's *wall* time falls materially at fixed math. Unmeasured; the first deliverable of that
  lane must be a warm-step timing, not a BPB.
- **SPECULATION — the payoff of §2.2.** The 1.57× backward gap is measured against the padded arm;
  the split between the three pieces (fused `dx`, no `repeat_interleave`, fused combine backward)
  is not. I assume (a) and (c) dominate at b1 because they remove a full `[R,f]`/`[R,d]` pass each.
- **SPECULATION — full-rank Dion on `[2048,2048]`** (SKIP list): the 1 QR + 3 matmuls vs 10
  matmuls comparison is arithmetic, not a benchmark; `torch.linalg.qr`'s cost on this card is
  unmeasured, and QR has a different occupancy profile from the cuBLAS GEMMs.
- **SPECULATION — table row 6 applies to us at all.** I did not read `model/model.py`, so I cannot
  confirm that a fused projection (QKV or gate/up) sits in our Muon group at `medium`/b1. If none
  does, the row is moot.

## 7. Open questions

1. Does our `Muon` step contain any other host-side read that a capture window would refuse
   (`optim.py` reads `lr`, `mu`, `ns_steps`, `scale` from the group dict — all Python floats)?
2. What is our warm-step time noise floor, so a graph-capture win can be judged at all?
3. Does our `_permute` already produce the row map we would need, or does the capacity branch
   (`keep`) mean the map must be compacted first? (`experts.py:475-503` computes `order`/`keep`/
   `pos`, but the padded arm uses `kept_slots`, not a `[n·k]` map.)
4. Which callback seam is genuinely needed now — one hook for the stress ring, or none until the
   stress state actually needs to survive a resume?
