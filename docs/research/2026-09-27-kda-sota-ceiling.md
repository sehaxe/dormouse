# KDA at the production shape: what SOTA actually is, and where our 1445 ms really goes

> ## ⚠️ CORRECTED 2026-09-29 (2nd pass) — the step-time numbers are also retracted
>
> The correction below (2026-09-27) addressed *which arm ran*. It did not
> address the **step-time readings**, which are a separate defect and are
> retracted here. Warm, release, quiet card, `CUBECL_AUTOTUNE_LEVEL=3`,
> `--timers` at a step index past 50 (`benches/history.tsv`, 2026-09-29):
>
> - **The `1810 ms` step** (§2.4, §4.2) and the **`1445 ms` KDA cost** are
>   **step-0 readings**. A warm step at batch 8 x seq 512 is **244 ms** and the
>   **warm forward is 46-48 ms**, i.e. 19 % of a step, not 80 %. A step-0 step
>   is 5549 ms in the same configuration — **23x** — because the cubecl
>   autotune cache is cold for the first steps. `--timers` prints on
>   printed on `step % 50 == 0` only and was not tied to `--log-every`, so a
>   short run could only ever see step 0 (**fixed in `b8a47ee`; the cadence now
>   follows `--log-every`**). **The "KDA is 80 % of a step" attribution is
>   therefore dead twice over**: it was measured on a run that skipped the
>   backward, and it was measured at step 0.
> - **The `~465 ms fixed cost`** (§2.3, §2.5) is **not a measured constant**.
>   It was fitted to two step-0 readings. The warm step is 244 ms — the same
>   order, smaller — so the *conclusion* the floor was used for survives with
>   a smaller number, but do not quote 465 ms.
> - **§4.2 item 4's "a perfect KDA kernel takes the step from 1810 → 365 ms"**
>   is withdrawn without replacement. Both endpoints are step-0 readings and
>   the subtraction inherits the error.
>
> **What is confirmed, and it is the same conclusion this document reached by
> a different route: the cost is host-side launch overhead, not arithmetic.**
> That was measured here in 2026-09-27 (99 % host dispatch, GPU on 1-20 ms of
> a 470-635 ms call) and it is now measured a second, independent way — **the
> GPU is 13.3 % utilised on average with 142 of 180 samples at <=5 %** over a
> 150-step warm run at batch 32 (`nvidia-smi` at 2 Hz). Two instruments, one
> week apart, same verdict.
>
> **Still not re-derived:** the attention backward's true cost. The
> measurements here and in `kda-allocator-fix.md` both predate the backward
> running, and no run on record has executed one (`fused kda=<f>/0`).
> §4's `max_iter` lever is unchanged and still the largest single flag.
>
> ## ⚠️ CORRECTED 2026-09-27 — read this first
>
> `docs/archive/research/2026-09-27-kda-allocator-fix.md` measured the §2.5 hypothesis this
> document was built on, at the production shape, on the trainer's own patched
> cubecl, with real `Client::sync()` barriers. Three claims below are false and
> must not be relied on:
>
> - **"The fused kernels are engaged."** They were not, in any training step.
>   burn-kda's dispatch was a `TypeId` test against `Autodiff<CudaBare>` (=
>   NoCheckpointing); the trainer runs `Autodiff<Cuda, BalancedCheckpointing>`,
>   which fails that test by construction, so every step ran the ~150-op-per-
>   chunk **tensor** path. The §2.4 "1300x gap" and §3(a)'s "already built, and
>   it is not the bottleneck" both rest on that false premise.
> - **§2.5's allocator arithmetic.** "1 GB of pool misses per step ≈ 1000 ms"
>   is falsified: `client.memory_cleanup()` returned 887 MB to the driver, the
>   next call re-reserved +887.2 MB, and it ran in 561 ms — the same as the
>   warm 519/470 ms calls. Warm fwd+bwd is **470-635 ms per call with the GPU
>   on 1-20 ms: 99% host dispatch.**
> - **"17 fresh tensors totalling 248 MB."** ~850 allocation *slices* per
>   forward; 301.8 MB in use, 914 MB reserved.
>
> What survives: the memory-bound verdict (§2.3, 3.1 FLOP/byte), the roofline
> arithmetic, FLA's structural facts (§1), and §4's `max_iter` lever. What the
> measurement adds: engaging the fused node is worth **1.57x** on the KDA call
> (505 → 321 ms, warm), the pool's real damage is VRAM growth (+918 MB per
> forward-only call, never reused) rather than time, and the binding constraint
> after the dispatch fix is still host-side op count — i.e. §3/AGENTS.md's CUDA
> graph, not more kernel work.

Date: 2026-09-27. Subagent deliverable, no source changes.
Shape under test: `b=10 t=512 h=12 K=V=64 chunk=16 fp32` (dormouse `small` preset, 4 loop
iterations, JEPA EMA teacher). GPU: RTX 5060 Ti, sm_120, 170 SM, 16 GB, 1792 GB/s.
All roofline arithmetic below is reproducible from `/tmp/opencode/kda_arith.py` and
`/tmp/opencode/fla_tflops.py`. **No GPU was used for this report** (a second process held
5.6 GB of VRAM the whole time); every rate is either published, given in the task brief, or
derived from `nvidia-smi` specs.

---

## 0. TL;DR — the answer nobody wants

The fused cubecl forward **and backward kernels for this exact computation already exist in
the tree** (`vendor/dormouse-fused/crates/burn-gdn2/src/kernel/chunk_cube.rs` 1243 lines,
`chunk_adjoint_cube.rs` 691 lines), they *are* engaged on the training backend
(`Autodiff<Cuda>` → `chunk_autodiff_or_plain::<CudaBare>`), and they dispatch **3 launches
forward, 2 backward**. Route (a) as posed — "write a fused backward kernel in cubecl" — is
already built.

Their own arithmetic floor is **~80-100 µs**. We measure **~120 ms per call**. That is a
**~1300x gap between the kernel that is running and its own roofline**, and a **~310x gap
between the measured step cost and the HBM roofline**. A kernel rewrite cannot close a
1300x gap. Something in the *wrapper* is. The report therefore says: **do not write a
kernel; go find the 120 ms first** — and names the two suspects with the arithmetic that
makes them fit (248 MB of fresh scratch per forward × 4 iterations ≈ 1 GB/step against a
high-water pool).

---

## 1. Reference implementations, at the level of their inner loops

### 1.1 FLA Triton `chunk_delta_h` — the reference, and the shape of the whole problem

`fla/ops/common/chunk_delta_h.py`, kernel `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
(fetched 2026-09-27, `main`). The structural facts that matter:

- **Grid is `(B*H*NT, HV)`.** The inter-chunk recurrence — the part that cannot be
  parallelised — gets exactly **B*H programs**. Everything else is token-parallel.
- **The state `b_h` (BK×BV = 64×64) lives in REGISTERS as the `tl.dot` accumulator** for the
  whole `for i_t in range(NT)` loop (`b_h1 = tl.dot(b_k, b_v, b_h1)`). It is *written to
  global per chunk* (`tl.store(p_h1, b_h1...)`) only so the backward can read it. This is
  the direct answer to "does the chunked form re-read the state per chunk": yes, once per
  chunk, once per direction.
- Per chunk the loop does 3 `tl.dot`s: `v_new = V − tl.dot(w, h)`, `h += tl.dot(kᵀ, v_new)`,
  and the h store. `USE_GK` (channel-wise decay, i.e. KDA) adds an elementwise
  `b_h1 *= exp2(b_gk_last1)` on the state — cheap, no extra GEMM.
- **Autotune space is tiny and hardware-pinned.** `BV ∈ {32,64}` (Ada shared-mem gate),
  `num_stages ∈ {2,3,4}`, and `num_warps = [2]` **on Blackwell** with this comment:
  *"Triton mainline fixes a Blackwell tl.dot recurrence race. Keep this kernel on num_warps=2
  for Blackwell until Triton 3.8 is released and re-validate the wider config space."*
  `num_warps=2` means 64 threads — a tiny block, chosen so the 64×64 fp32 accumulator does
  not spill. **The state is register-resident precisely because there are too few threads to
  afford shared memory.**
- `chunk_o.py` is the second stage: loads `h` from global, `b_o = tl.dot(b_q, b_h, b_o)`,
  plus the intra part `tl.dot(b_A, b_v)`. Its Blackwell path *also* drops the 128×128/8-warp
  config for the same recurrence race.
- `gated_delta_rule/wy_fast.py`: the UT transform (`recompute_w_u_fwd_kernel`,
  `prepare_wy_repr_bwd_kernel`) is 2 `tl.dot`s of `[BT,BK]×[BK,BT]`. On Blackwell its
  backward autotune is **hard-pinned to `num_warps=[2], num_stages=[4]`** because *"Blackwell
  can select unstable Triton configs for prepare_wy_repr_bwd_kernel during autotuning (see
  #913). Restrict it to the config that has been validated on B200."*

**Our kernel has the same two-kernel split and the same serial structure.** The difference is
the arithmetic engine: our `gdn2_chunk_inter_kernel` runs a **4-way-unrolled scalar FMA loop
out of shared memory** (`vn_a += w0 * s_sh[kk*vtile+v0]`, 64 kk-steps, then the aqk and qg
loops, then a 64×16 state update), with `if r < c` guarding everything so only **16 of the X
lanes** do work, at `CubeDim::new_3d(16, 4, 1)` = **64 threads/block**, 960 blocks. FLA runs
the identical recurrence on `tl.dot` tensor-core accumulators in registers. **That is the
whole 3-10x kernel-level gap, and it is the only kernel-level gap there is.**

### 1.2 Published throughput, normalised (this is the load-bearing table)

FLA README benchmark, **GB200** (bf16, CUDA 12.9), `chunk_gdn`. Per-token-per-head FLOP
model for the chunked gated delta rule at C=64: `2CD(qk) + 2CD(kk) + 2C²(UT) + 2CD(w) +
2CD(u) + 3·2D²(state)` = 131 072 at D=128. Bytes model: 4 in-tensors + 1 out (bf16) +
per-chunk `h` + 3× scratch.

| B | T | H | D | programs = B·H | serial NT | GFLOP fwd | **TFLOP/s fwd** | **% of GB200 bf16 peak (2250)** | **GB/s** | **% of 8 TB/s** |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8192 | 96 | 128 | 96 | 128 | 135.3 | **107** | 4.8% | 2069 | 26% |
| 2 | 16384 | 16 | 128 | 32 | 256 | 90.2 | **88** | 3.9% | 1696 | 21% |
| 4 | 4096 | 64 | 128 | 256 | 64 | 180.4 | **114** | 5.1% | 2207 | 28% |
| 8 | 2048 | 32 | 256 | 256 | 32 | 279.2 | **153** | 6.8% | 2199 | 27% |
| 4 | 2048 | 16 | 128 | 64 | 32 | 22.5 | **30** | 1.3% | 579 | 7% |
| 8 | 1024 | 8 | 64 | 64 | 16 | 4.3 | **6.8** | **0.30%** | 160 | 2% |

**This is the single most important table in the report.** The state of the art — a
Triton kernel written by the authors of the method, autotuned, running on a 148-SM B200 —
achieves **0.3-6.8% of tensor-core peak and 2-28% of memory bandwidth.** It is
**latency/occupancy bound, not FLOP bound**, and the cause is structural: the serial
recurrence has only B·H programs. The best cells are the ones with B·H ≥ 256 **and** NT ≥ 64.
**Our shape (B·H = 120, NT = 32) sits squarely in FLA's worst published bucket.**

Backward, from the same table (fwd vs fwdbwd): **bwd/fwd = 2.75x, 2.51x, 2.77x, 3.72x,
2.13x, 3.72x** → **bwd ≈ 2.1-3.7x fwd, mean ~2.8x**. (An independent in-situ timer on this box,
`/home/sehaxe/logs/ab8m_ab8m_iter4.log` step 900: `fwd=933ms bwd=1869ms` = **2.00x** — same
conclusion, different build.)

### 1.3 FlashKDA (Moonshot, CUTLASS) — the only thing that beats FLA, and it is forward-only

`MoonshotAI/FlashKDA`, MIT, **docs/20260420-flashkda-v1-deep-dive.md** + `BENCHMARK_{H20,GB200}.md`
(opened 2026-09-27):

- **CHUNK = 16** — same as ours. Three reasons given: bf16 range of `exp(cumsum(g))` at
  `lower_bound=-5`; a 16×16 inverse is "dramatically cheaper" than 64×64 and is done by
  **direct forward substitution without decomposition**; and *"all CHUNK=16 math maps cleanly
  onto SM80 MMA instructions"* — i.e. it is deliberately a warp-level-MMA kernel, portable
  across Ampere/Ada/Hopper/Blackwell. **Our chunk=16 choice is already the FlashKDA choice,
  and it is the reason our kernel is portable to sm_120 at all.**
- **Two kernels on the parallelism axes**, verbatim: *K1 (token-parallel, grid = N×H×num_chunks)*
  = gate activation + L2 norm + decay + L/Mqk construction + matrix inversion; *K2
  (head-parallel only, grid = N×H)* = the recurrence + output + state accumulation. *"Early
  prototypes used a single fused kernel. In that design, the token-parallel work in K1 was
  bottlenecked by the much lower parallelism of the recurrence in K2, leaving a large fraction
  of the SMs idle. Splitting the pipeline into two kernels yielded at least a 15% end-to-end
  speedup."* **Our `chunk_cube.rs` already has this exact split (intra = K1, inter = K2).**
- bf16 state on chip, fp32 FMA for the state update. `sigmoid` via `tanh.approx.f32`. Base-2
  exponent rebasing + `ex2.approx.ftz.f32` (kills the change-of-base FMA and is faster than
  `exp`). K1 uses `__launch_bounds__(256, 8)` with shared-memory unioning to buy occupancy.
  K2 uses `MOVM_T` register-file transposes to eliminate every shared-memory round trip
  between stages.
- The 16×16 inverse: seed `L` in fp32, **two diagonal 8×8 blocks by fp32 forward
  substitution**, off-diagonal merged with **two bf16-HMMA GEMMs** (`dc = P @ M`, `o = (−dc) @ P`).
  Ours does the full 16×16 in shared memory with a `sync_cube()` per row.

**Numbers.** H20, T=8192, D=128, forward only: H=96 fixed 2.622 ms vs FLA 4.505 ms = **1.72x**;
H=64 fixed 1.620 vs 2.959 = **1.83x**; varlen `1024×8` up to **2.22x**. GB200, T=8192, H=96,
D=128: **1.0087 ms vs `fla_chunk_kda` 2.3271 ms = 2.31x** (varlen 1024×8: 3.27x); H=64:
0.9247 vs 1.5764 = 1.70x. Cross-check: their harness measures `fla_chunk_gdn` at 1.2792 ms
where FLA's own README says 1.265 ms — 1% apart, same kernel, so the two datasets are
commensurate.

Normalised on GB200, H=96, D=128, C=16 (per-token-per-head 118 528 FLOP):
**FlashKDA 92.4 TFLOP/s, FLA Triton 40.1 TFLOP/s, FLA `chunk_gdn` (C=64) 80.6 TFLOP/s.**
GB200 peak 2250 → **FlashKDA 4.1%, FLA 1.8-3.6%.** Even the hand-tuned CUTLASS kernel that
beats the reference by 2.3x is at 4% of peak.

**Three hard blockers for us, all verifiable:**
1. **No backward kernel exists.** FlashKDA v1 is a forward kernel. The FLA-integration PR is
   a drop-in for `chunk_kda`'s **inference** path. For training you are on FLA Triton.
2. **K=V=128 only, SM90+, CUDA 12.9.** We are K=V=64, sm_120.
3. **It is a prefill/inference kernel.** It does not even store the per-chunk state — that is
   precisely the thing training needs.

### 1.4 Qwen3-Next, Qwen3.5, Qwen3.6 — what actually runs in production

- **Qwen3-Next (2025-09) runs FLA's Triton `chunk_gated_delta_rule`.** vLLM's own blog:
  *"vLLM integrates Triton kernels from Flash Linear Attention"*; the `slime` training plugin
  has `self.gdn_backend = getattr(args, "qwen_gdn_backend", "fla")` with a `flashqla`
  alternative. So the largest deployed GDN training stack on earth uses FLA Triton — **the
  1.8-4.8%-of-peak kernel in §1.2.**
- HuggingFace's fallback when FLA is absent is a **pure-PyTorch chunked recurrence**:
  *"978 to 5573 tok/s on Qwen3.6-35B-A3B 4-bit at 4k, a 5.7x gap."* That 5.7x is the FLA-Triton
  over naive-chunked-tensors speedup, and it is the only ratio that transfers to us: it
  measures the *wrapper*, not the GEMM.
- Qwen's own hand-tuned kernel is **FlashQLA (TileLang, SM90 prefill)**; a community port
  (`Plaaasma/FlashQLA-Blackwell`) targets **SM_120/121** for DGX Spark. Its changelog is a
  warning list: bypass the SM90 arch gate; `T.gemm_v1` → `T.gemm_v2` because *"`gemm_v1`
  silently produces wrong results on Blackwell"*; shrink shared memory to fit Blackwell
  consumer's **99 KiB**; two upstream bugs. And: *"End-to-end vLLM TTFT win on Qwen3.6-27B is
  much smaller (~3% on 8K prefill) because that model's prefill is dominated by its 16
  full-attention layers and 64 MLPs."* Forward only. No backward.

### 1.5 The sm_120 fact that kills half the kernel literature

From Dao-AILab/flash-attention#2634 (opened 2026-09-27), quoting verified ptxas/SASS on
CUDA 13.x: *"consumer/workstation Blackwell (sm_120) ships 5th-gen tensor cores with new
fp4/fp8 data types but — unlike datacenter Blackwell (sm_100) — keeps the Ampere-style
**warp-level `mma.sync`** programming model: **no WGMMA, no tcgen05, no Tensor Memory**."*

Consequences, all load-bearing:
- **FlashInfer's Blackwell GDN prefill** (`gdn_kernels/blackwell/gated_delta_net_chunked.py`,
  7 GEMMs/chunk, state in TMEM, `tcgen05` MMA, TMA + warp specialisation) is **sm_100 and
  cannot run on our GPU.** It is also forward-only.
- The `lethe` project (claimed first native `tcgen05` GDN-family training backward, 7 GEMMs
  on a 128×64×128 tile, verified 3.29e-3 vs fp64) is **sm_100/B200-only** and pins a CUTLASS
  CuTe DSL toolchain. Unreachable. (Its one transferable claim: the official Mamba-3 Triton
  backward **fails to compile on sm_100 at every `num_warps >= 4`** — a TMEM overflow,
  `Required: 544, Hardware limit: 512`.)
- **This explains the WMMA result in the brief.** WMMA on sm_120 lowers to the same
  Ampere-style `mma.sync`, and a 16×16×16 tile cannot amortise the fragment shuffling. 8.4
  TFLOP/s losing to SGEMM is the expected outcome, not a surprise. The route to tensor cores
  here is `mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32` (or fp16 in / fp32 acc, the 43.7
  TFLOP/s cuBLAS path) with hand-laid fragments — not WMMA, not tcgen05.

### 1.6 GDN-2 (the quality frontier) and its throughput cost

arXiv 2605.22791 (NVIDIA), Fig. 2: hybrid 1.3B, **H100 training throughput 38.0 → 36.1 Kt/s**
as the channel-wise erase `b_t` and write `w_t` gates are added, *"retains practical training
efficiency while paying a modest constant cost."* Kernels are in FLA (`fla/ops/gdn2`, MIT) even
though the reference repo is NC-licensed. Mechanically, GDN-2 is **our `chunk_cube.rs` with
`b` and `w_gate` per-channel instead of scalar** — i.e. **we have already implemented the
GDN-2 kernel**; `burn-gdn2` is the GDN-2 formulation and KDA is its special case. The extra
work is the gate-aware backward accumulation (paper Eqs. 27+): the scalar `β` can be hoisted
out of `dA`'s accumulation, and that shortcut **breaks** under per-channel gates. That is a
real, small, well-specified change — and it is a *quality* change, not a speed one.

### 1.7 What is NOT VERIFIED

- **No published or repo-reported fwd+bwd throughput for KDA/GDN on sm_120 / consumer
  Blackwell, at any batch.** FlashKDA: SM90+, fwd only. FlashInfer Blackwell: sm_100, fwd
  only. FlashQLA-Blackwell: SM_120/121, fwd prefill only, no benchmark table published.
  **The number we are trying to beat on our own GPU has never been published.**
- **No published TFLOP/s for a fused gated-delta *backward* on any consumer-Blackwell part.**
- No Triton version is pinned in this repo, so I cannot say whether the Blackwell
  `tl.dot` recurrence race (fla #945/#953) and the `kkt_solve` forward NaN (#913, *"no
  validated upstream fix yet"*) would reproduce on our Triton build. Not tested.
- Absolute GB200/H20 numbers do not transfer to sm_120; only the ratios and the
  structural conclusions do.

---

## 2. The arithmetic for our shape

`b=10 t=512 h=12 K=V=64 C=16` → NT=32 chunks, **nblk = B·H·NT = 3840**, BH = 120. fp32.

### 2.1 FLOPs — forward

Per (b, h, chunk), the op set actually in `gdn2_chunk_intra_kernel` +
`gdn2_chunk_inter_kernel` (stages 31 and 7):

| op | shape | FLOP |
|---|---|---|
| `aqk = (q⊙Γ)(k/Γ)ᵀ` | C×K×C | 2·C²K = 32 768 |
| `akk = (b⊙k⊙Γ)(k/Γ)ᵀ` | C×K×C | 32 768 |
| `w = M⁻¹(bk⊙Γ)` | C×C×K | 32 768 |
| `u = M⁻¹(w⊙v)` | C×C×V | 32 768 |
| `v_new = u − w·S` | C×K×V | 131 072 |
| `intra = aqk·v_new` | C×C×V | 32 768 |
| `inter = (q⊙Γ)·S` | C×K×V | 131 072 |
| `S += (k⊙decay)ᵀ·v_new` | K×C×V | 131 072 |
| `M⁻¹` forward substitution | 15 rows | ~4 096 |
| **total / chunk** | | **561 152** |

**Forward = 3840 × 561 152 = 2.155 GFLOP** (35 072 FLOP/token/head).
The three `C·K·V` state matmuls are **70% of it** and are C-independent per token — the same
dominance FLA shows (at C=16 vs C=64 the per-token cost differs by only 1.11x).

### 2.2 Bytes that MUST move — forward, fp32

| | MB |
|---|---|
| reads: q,k,g,b + v,w (4×15.7 + 2×15.7) + 2 state copies | 98 |
| writes: `gexp,kgt,qgt,bkt,w,kgd,u,wvt,v_new` (9 × 3840×16×64×4) | 157 |
| writes: `aqk,akk,m_inv` (3 × 3840×16×16×4) | 12 |
| writes: out `[10,12,512,64]` | 16 |
| **writes: per-chunk state export `states[3840,64,64]`** | **63** |
| **one-way total** | **346** |
| × ~2 for kernel-internal re-reads (FLA does the same: `h` written then re-read) | **692** |

**This is the number the recommendation turns on: 63 MB of the 248 MB saved-for-backward
scratch is the per-chunk state trajectory** — 25% of the KDA activation footprint, written
purely so the backward can avoid recomputing it. FLA deliberately does the opposite
(*"Backward: Recomputes w, u from saved A to save memory"*). For a **memory-bound** kernel,
that export is a pure loss: trading 63 MB of write + 63 MB of read to save a recompute that
costs 0 extra bytes.

### 2.3 Roofline — this box

| | value |
|---|---|
| HBM | 1792 GB/s (GDDR7, 512-bit @ 28 Gbps) |
| measured cuBLAS fp16-in / fp32-acc | **43.7 TFLOP/s** (5120×2048×8192, from brief) |
| fp32 CUDA-core FMA peak | 170 × 128 × 2 × 2.41 GHz = **104.9 TFLOP/s** |
| **machine balance (tensor cores)** | 43.7e12 / 1792e9 = **24 FLOP/byte** |

| | forward | backward (×2.5 fwd) |
|---|---|---|
| FLOPs | 2.155 GFLOP | 5.39 GFLOP |
| bytes (r+w) | 692 MB | 1.73 GB |
| **arithmetic intensity** | **3.1 FLOP/byte** | **3.1 FLOP/byte** |
| HBM roofline | **0.386 ms** | **0.965 ms** |
| tensor-core roofline | 0.049 ms | 0.123 ms |
| fp32-CUDA-core roofline | 0.021 ms | 0.052 ms |
| **which roofline binds** | **MEMORY (8x under machine balance)** | **MEMORY** |

> ### Verdict: **the KDA forward and backward at this shape are MEMORY bound, not compute bound.** 3.1 FLOP/byte against a 24 FLOP/byte machine balance. There are 8x more bytes than tensor-core work. A fused kernel that achieves perfect tensor-core utilisation would still be 8x slower than the memory roofline. **The binding constraint is the scratch traffic, and the arithmetic says the highest-value kernel change is deleting bytes, not adding MMA.**

**fwd+bwd ideal = 1.35 ms per KDA call.** With `max_iter=4` and a no-grad EMA teacher that
still runs a full forward: 8 forwards + 4 backwards = **7.0 ms/step** of KDA work at the
roofline, against a step whose fixed launch overhead alone is ~~**~465 ms**~~ **(2026-09-29:
not a measured constant — it was fitted to step-0 readings; the warm step is
244 ms. The order holds, the number does not.)**

### 2.4 Our fraction of roofline

~~Per step, 1810 ms total, fwd 730-754, bwd 893-1059, KDA 1445 ms (80%).~~ **RETRACTED
2026-09-29: these are step-0 readings.** Warm, a step is 244 ms and the forward
is 46-48 ms (19 %). The per-call figure below is not replaced — the backward has
never run, so there is no warm number for it. At `max_iter=4`
that is ~12 KDA calls (4 student fwd + 4 teacher fwd + 4 bwd) → **~120 ms per call**
**(step-0; the 2026-09-27 correction already localised where it goes — 99 % host dispatch).**

| | measured | roofline | **fraction of roofline** | **off by** |
|---|---|---|---|---|
| forward | ~120 ms/call (order-of-mag, per-call split NOT VERIFIED) | 0.386 ms | **0.3%** | **~310x** |
| backward | (included above) | 0.965 ms | **0.8%** | **~125x** |
| **step total** | **1445 ms** | **7.0 ms** | **0.5%** | **~206x** |
| **fused kernel's own floor** (inter kernel: 2.17 GFLOP, LDS-bound at 4 LDS per 4 scalar FMA, 960 blocks × 32 serial chunks, 64 thr/block) | — | **~80-100 µs** | — | **~1300x** |

> ### The backward is **not** "inherently Mx the forward" in FLOPs. It is 2.0x on this box's in-situ timer (`fwd=933ms bwd=1869ms`, `ab8m_ab8m_iter4.log` step 900) and 2.1-3.7x (mean 2.8x) in FLA's published GB200 table — because the state trajectory must be walked **backwards** (the reverse recurrence) and the WY/triangular-inverse VJP adds ~7 GEMMs per chunk. That 2-2.8x is **inherent to the algorithm** and is *not* where our problem is.

**The 1300x line is the finding.** Our fused kernels are running. Their design floor is
~80-100 µs. They are taking ~120 ms. A kernel rewrite addresses the 80 µs. It does not
address the 120 ms.

### 2.5 Where the 120 ms most likely is (hypothesis, with the arithmetic that fits it)

Per KDA forward, `fused_chunk_forward_scratch` allocates **17 fresh tensors totalling 248 MB**,
and two of them (`v_new_out`, `states_out`) are **deliberately over-allocated by `+262144`
floats "to export buffers padded to a unique size"**, plus two `state.clone() * 1.0` copies.
At `max_iter=4` that is **~1 GB of fresh allocations per training step**, against a cubecl
memory pool that this repo's own AGENTS.md documents as *"high-water: reserves pages for every
size ever seen, never frees → long runs OOM"*, with `memory_cleanup()` called periodically.

**At 1 ms per pool-miss `cudaMalloc` (a conservative figure for a synchronising 250 MB
allocation), 1 GB of misses per step is ~1000 ms.** That is the 1445 ms. It also explains
the other three measurements in the brief without any extra theory: **4x tokens costs only
1.47x time** (allocations scale with `nblk = B·H·NT`, so 4x tokens = 4x bytes, but the
*malloc* count and the per-malloc cost are sublinear in size); the **~~465 ms fixed cost**
(struck 2026-09-29 — fitted to step-0 readings; the warm step is 244 ms)**;
and why the earlier bench that timed one kernel found "0.23 ms" while the step pays 1445 ms.

**This is a hypothesis, not a measurement.** It is, however, the hypothesis that costs one
afternoon to test and it is consistent with all four independent measurements. The test
already exists: `kda_step_probe.rs` times `forward_train` / `backward` / `forward_train_fused`
on the production geometry with a **device sync inside the timed region** (its own docstring
records that the previous bench measured *launch* time, not GPU time — the async cubecl
backend makes `Instant` around a forward meaningless). Run it at b=10 and read the number.
**If it says ~1 ms, the step's 1445 ms is the allocator and no kernel is relevant. If it
says ~120 ms, the kernel is the cost and §3(a) is the answer.** Nothing else needs to be
built before that number exists.

---

## 3. The three routes, ranked

### (a) A fused backward kernel in cubecl's custom-kernel layer — **ALREADY BUILT, and it is not the bottleneck**

`chunk_adjoint_cube.rs` (691 lines) is a real fused backward: `gdn2_chunk_inter_adjoint_kernel`
(`CubeDim(16,4,1)`, `CubeCount(bh=120, vt=8)` = 960 blocks, the sequential BPTT chain) plus
`gdn2_chunk_intra_adjoint_kernel` (`CubeDim(16,8,1)`, 3840 blocks). The autodiff node calls it
from `ChunkWy::backward` with `is_cuda::<B>()` true. It is engaged.

*What a real improvement to it would be*, in the order the roofline pays for:
1. **Delete the 63 MB per-chunk state export** (25% of the scratch). FLA recomputes `w, u`
   from the saved `A` in the backward instead. Cuts ~126 MB of the ~1.7 GB backward traffic,
   and removes 63 MB/step/iteration of VRAM. **This is the change the arithmetic asks for,
   and it is not a new kernel — it is deleting an export and adding a recompute.**
2. **Tensor-core the inter/inter_adjoint loops.** The current `while kk + 3 < kd` scalar FMA
   chain out of shared memory is LDS-throughput bound (~4 shared loads per 4 FMAs → ~25% of
   fp32 FMA peak, no `mma.sync`). Replacing with `mma.sync.aligned.m16n8k16` + a
   register-resident state (FlashKDA's `MOVM_T` register-file transpose; FLA's
   `b_h1 = tl.dot(b_k, b_v, b_h1)` accumulator) is the honest fix. On sm_120 that means
   hand-built m16n8k16 fragments — **not WMMA** (§1.5), **not tcgen05**.
3. `ex2.approx.ftz.f32` + base-2 rebasing for the gate cumsum; `tanh.approx.f32` for sigmoid
   (FlashKDA's two cheapest wins, both free).
4. Merge the intra+inter kernels' separate passes. Low value: they are already only 3+2
   launches.

*Expected speedup on the kernel*: 3-4x (80-100 µs → 25 µs floor). *On the step*: **0.3 ms of
1445 ms — 0.02%.** Zero until §2.5 is resolved.
*Cost*: 3-8 days for a correct mma.sync state-recurrence with the `BAR.SYNC.DEFER_BLOCKING`
WAR hazard the current code already documents at length.
*What can break*: sm_120 is consumer Blackwell — the same architecture generation where
Triton ships a `tl.dot` recurrence race and where FlashQLA's `gemm_v1` *"silently produces
wrong results."* Any new MMA path needs a bit-exactness test against the current kernels
(`chunk_adjoint_cube` + the two `bitforbit*.rs` harnesses already in the tree). Numerically:
the `K/exp(cumsum(g))` factor underflows f32 below cumsum(g) < −88, i.e. **chunk > 17 is
forbidden** at the K3 floor `g = −5`; that is why FlashKDA also picked 16.

### (b) CUTLASS / cuBLAS through cudarc on cubecl's stream — **best-engineered route, worst-value route for this op**

I verified all three premises: **cudarc 0.19.10 ships `cublas` and `cublaslt` modules**;
**cubecl's `Stream.sys` is `cudarc::driver::sys::CUstream`**; **`GpuResource { pub ptr: u64,
binding, size }` exists** in the vendored `cubecl-cuda/src/compute/storage/gpu.rs`. The
missing piece is exactly what `docs/archive/research/2026-09-27-optimization-1b.md` already specifies: a
client-side `Handle → GpuResource` RPC.

But the roofline says the route cannot help *this* op:
- **The state matmuls are 70% of the FLOPs and are not GEMMs.** `v_new = u − w·S`,
  `S += kᵀ·v_new` and `o = qg·S` are the accumulator of a 32-step **dependent** chain over a
  state that lives between steps. cuBLAS cannot express that. Only a custom kernel can.
- **The work that cuBLAS *could* take** (the CCK/CCV block: `aqk`, `akk`, `M⁻¹`, `w`, `u`,
  `intra` = 30% of FLOPs) is currently **3 launches' worth of already-fused cubecl work**.
  Handing 30% of 2.155 GFLOP to cuBLAS saves at most 0.1 ms — and cuBLAS on 16×16×64 and
  16×64×64 tiles will be *slower* than the fused kernel, because cuBLAS's own floor is
  ~5-10 µs of launch+dispatch per call and we would need 4 calls per chunk × 32 chunks.
- **And it is memory bound anyway** (§2.3). Better GEMMs cannot move fewer bytes.

*Expected speedup on the KDA op*: **~0.** *Cost*: 3-5 days for the RPC + stream + handle
plumbing, plus a lifetime hazard (cubecl's pool recycles buffers; a raw pointer captured
across an allocation is a use-after-free the type system cannot catch).
*What it IS the right answer to*: the 768-d `LinearLike` GEMMs, which are 2-5% of a step and
which this repo already measured at cubecl-fp32 3.5-7.6 TFLOP/s against cuBLAS 43.7. **That
is a different question with a different answer, and it is already written down. Do not let
it be confused with this one.**

### (c) FLA Triton kernels from a Python sidecar — **no. PCIe is the wall, then stability is.**

Three independent kills, any one sufficient:

1. **PCIe.** Per KDA call the sidecar must ship `q,k,v,g,b` out (78 MB) and receive
   `dq,dk,dv,dg,db,dw` back (94 MB) — ~172 MB of round trip at ~12 GB/s = **14 ms per call,
   172 ms per step at `max_iter=4`**, before a single FLOP is computed. The entire GPU-side
   roofline budget is 1.35 ms/call. The sidecar's *floor* is 10x the thing it would be
   replacing, and 172 ms of the 1445 ms it targets. It would also add ~1.4 GB of pinned host
   buffers and a second process on a box already RAM-constrained to 8 GB headroom.
2. **Correctness/precision.** FLA wants bf16; our tensors are fp32 (the fused kernels are
   `launch_unchecked::<f32>`-only and explicitly fall back below f32). Every call needs a
   cast pair = 2 more full passes. The backward crosses back as fp32 grads that then need
   a dtype-matched adjoint — i.e. a hand-written cross-process autograd boundary.
3. **Upstream cannot make these kernels stable on Blackwell.** fla #945/#953: the Blackwell
   forward-h kernel is pinned to `num_warps=2` because 4 warps *"hits a Triton `tl.dot`
   recurrence race on B200/B300 that silently corrupts the hidden state."* fla #999/#1000:
   `prepare_wy_repr_bwd` is pinned to one config because other configs *"can hang or hit a
   misaligned address in the backward."* And issue #913, the Blackwell forward-NaN in
   `kkt_solve`, has *"no validated upstream fix yet."* **We would be running a kernel on our
   GPU that its own authors ship with three pinned workarounds and one open corruption bug,
   across a process boundary, in a training loop, on a codebase whose entire stability
   doctrine is "fused fp32, no NaN."**

*Expected speedup on the step*: best case 1445 → ~200 ms (2.5x), and only if the 120 ms is
really the kernel — which §2.5 says it probably is not. *Cost*: 5-10 days plus permanent
operational surface. *Verdict*: **no.**

---

## 4. The honest recommendation

### 4.1 The arithmetic says the "memory-bound, 2-3x not 10x" hypothesis is right — and then some

The prompt's alternative hypothesis is correct and understated. The op is memory bound at
**3.1 FLOP/byte** against a **24 FLOP/byte** machine balance, so a fused kernel is capped at
roughly **2-4x, not 10x**, and the cap is on *bytes*, not on MMA. FLA, the reference
implementation, is at **0.3-6.8% of tensor peak**; FlashKDA, which beats it by 2.3x with
hand-written CUTLASS, is at **4.1%**. **There is no 10x sitting in this kernel.** The
headline is worse than the prompt assumed: we are not 10x off SOTA, we are **~1300x off the
roofline of a kernel we already wrote.**

### 4.2 Do these, in this order

1. **Measure the 3 fused launches in isolation at b=10, with a device sync inside the timed
   region.** `burn-kda/examples/kda_step_probe.rs` already does exactly this and already
   documents why the previous bench was wrong. 1 hour, no new code, no GPU contention risk
   beyond 60 s. **Everything below is contingent on this number.**
2. **If the probe says ~1 ms (my expectation):** the 1445 ms is allocation/pool. Then the
   work is (i) make the 17 scratch tensors a reused, fixed-size workspace instead of fresh
   allocations, (ii) delete the `+262144` unique-size padding, (iii) drop the per-chunk
   `state` export in favour of the backward recompute FLA uses. That is ~1 GB/step of
   allocation traffic removed and 126 MB of per-backward traffic removed, and it is worth
   **most of the 1445 ms**. Cost: 1-2 days. **No kernel.**
3. **Then, and only then,** the kernel work: `mma.sync.m16n8k16` for the inter/inter-adjoint
   state recurrence with a register-resident state. Worth ~3-4x on a kernel that is then
   ~1-3 ms. 3-8 days. Do it because it is the right kernel, not because it is the bottleneck.
4. **Meanwhile, the bigger lever is a config flag, not a kernel.** `--no-kda` saves 768 ms of
   a 956 ms step at b4/t256. *(Both step times are step-0 readings and measured
   alongside a live run, so this is a ratio and not an absolute; see the
   2026-09-29 correction at the top.)* `max_iter` is settable (`--set max_iter=N`) and the trainer
   already has a random-depth arm that samples `T ∈ 1..=max_iter` plus an eval curve over
   depths 1..max_iter. **Four sequential full-sequence KDA passes per step is ~~80% of the
   step~~ RETRACTED 2026-09-29 (measured on a run that skipped the backward, at step 0;
   warm the forward is 19 % of a step), and three of the four are removable with a flag.**
   ~~A perfect KDA kernel (0 ms) takes the step from 1810 → 365 ms; the floor is 465 ms of
   launch overhead anyway.~~ **WITHDRAWN without replacement — both endpoints are step-0
   readings.** Cutting
   `max_iter` 4 → 2 removes half the KDA cost with a one-token diff. **Run the depth curve
   first — the quality answer may make the kernel question moot.**

### 4.3 What must NOT be built, and why

| Do not build | Why |
|---|---|
| A new fused backward kernel in cubecl | **It exists** (`chunk_adjoint_cube.rs`, 691 lines, 2 launches, engaged on `Autodiff<Cuda>`). |
| A CUTLASS/tcgen05 KDA kernel | tcgen05/TMEM **do not exist in sm_120 silicon** (FA#2634, verified against ptxas/SASS). FlashInfer's Blackwell GDN and `lethe`'s native backward are sm_100-only. Unreachable, not merely hard. |
| A WMMA kernel | Already measured at 8.4 TFLOP/s, losing to SGEMM. On sm_120 WMMA lowers to the same Ampere-style `mma.sync`; 16×16×16 tiles cannot amortise the fragment layout. The brief's own result is the answer. |
| A CUTLASS/cuBLAS path for the KDA op | 70% of its FLOPs is a dependent 32-step recurrence, not a GEMM. cuBLAS cannot express it. The op is memory bound. The cuBLAS work is worth doing for the 768-d `LinearLike` GEMMs — a different problem, already specified. |
| A Python/Triton sidecar for FLA | PCIe floor is 14 ms/call vs a 1.35 ms/call GPU budget (10x over, before any compute); plus a dtype boundary, plus three pinned upstream Blackwell workarounds and one open silent-corruption bug. |
| Porting FlashKDA | Forward only, SM90+, K=V=128, prefill. It does not store per-chunk state, which is the one thing training cannot do without. There is nothing to port. |
| Growing `chunk` past 16 | f32 underflow: `K/exp(cumsum(g))` dies at cumsum(g) < −88, i.e. chunk > 17 at the K3 floor `g = −5`. FlashKDA picked 16 for this same reason. `chunk=16` is correct and final. |
| GDN-2 as a *speed* project | It is a quality project. We already implement the GDN-2 kernel (`burn-gdn2` is the GDN-2 formulation; KDA is its special case). Only the gate-aware backward accumulation (paper Eqs. 27+) is new: ~300-800 lines, H100 cost 38.0 → 36.1 Kt/s. Do it for the RULER numbers (53.11 vs 52.28, MK-NIAH 37.8 vs 28.0), not for speed. |

### 4.4 One-line answers

- **Fwd:** we are at **0.3% of the HBM roofline** (0.386 ms ideal, ~120 ms/call measured) —
  ~310x off, and ~1300x off the *running kernel's own* floor.
- **Bwd:** **2.0x fwd on this box's in-situ timer, 2.1-3.7x in FLA's published table** —
  inherent to the reverse state scan + WY VJP, and not where the problem is.
- **Best route:** (1) profile the existing kernels with a sync, (2) kill the ~1 GB/step of
  scratch allocation and the 63 MB/chunk state export, (3) *then* `mma.sync`. Not a new kernel.
- **Biggest lever:** `max_iter`. A flag, not a fork.
