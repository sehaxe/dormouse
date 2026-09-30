# Math-site inventory: the spectral stack (burn-spectral, burn-sct, burn-muon-plus)

**2026-10-01, audit pass.** One row per math site, in the order the code runs.
Read with [`reviews/spectral-audit-2026-10-01.md`](reviews/spectral-audit-2026-10-01.md),
which carries the verdicts, the seam analysis and the class-B instrumentation.
Reproduce the instrument with `rustc -O -o /tmp/polar_probe tools/polar_probe.rs`.

**Honesty tiers used in the last column** (AGENTS.md 1.4):

| tier | meaning |
|---|---|
| **(a)** | an external source prints this value/formula, and a gate in-tree pins the code to it |
| **(a-cite)** | an external source is cited for the *method*, but the source prints no number we can compare against |
| **(b)** | our own transcription of a published method. No comparison against the authors' code exists |
| **(c)** | our own construction. **No external reference exists** |
| **(d)** | dead: defined, never called |

"gate" = the test in-tree that can go red if this site breaks. "none" means the
site has no gate — which is itself a finding, not a neutral.

---

## 1. burn-spectral — the retraction chain (the trainer's live path)

| # | site | file:line | what it computes | source claim | tier | gate | verdict |
|---|---|---|---|---|---|---|---|
| S1 | `polar_orthogonalize` | `src/lib.rs:208-266` | NS cubic polar retraction: canonicalise to the small side, Gram, 5-step power iteration, Rayleigh σ_max, ÷1.05, 3× `p(s)=1.875s−1.25s³+0.375s⁵` | PolarExpress triple = arXiv:2602.21545**v3** App. **D.3**, which prints `{(aₜ,bₜ,cₜ)}` ending at exactly this triple | **(a)** | `retraction_holds_the_manifold_at_rank_64`, `retraction_puts_sigma_max_at_one` (lib.rs:1388, 1458), `sync_free_retraction_is_bit_identical_to_the_host_read_one` | **CORRECT, and I reproduced its own anchor table** (below). The one genuinely load-bearing design choice — σ_max prescale, not ‖X‖_F — is pinned by a test whose failure mode is a *different manifold*, not noise |
| S2 | `polar_orthogonalize_batched` | `src/lib.rs:275-317` | the same, `[B,m,k]`, norms as `[B,1,1]` broadcasts | — | **(a)** (same math) | `retract_batched_identity_with_per_factor_path` (1667) | correct; **never used by the trainer** (`retr_arm=batched:0` in every log) although it is the only arm with no host round-trip. See F6 |
| S3 | `retract_batched` | `src/lib.rs:324-341` | group 2-D masters by exact shape, one stacked call per group, write back | — | **(c)** | `retract_batched_deterministic` (1711) | correct. Costs 4 calls instead of 16 at `small` (3 distinct shapes). Off by default (`retract_batched: false`, `train/src/lib.rs:175`) with no recorded reason |
| S4 | `polar_retracked` | `src/lib.rs:154-162` | `polar(before).detach()`, re-flag `require_grad` only if `before` had it | burn-optim silently downgrades a stored non-leaf to an untracked leaf | **(c)** | `retract_mirrors_master_tracking` (1628), `retract_keeps_masters_tracked` (1649) | correct, and the doc explains why forcing the flag is wrong twice over (hard panic on a non-autodiff backend; un-freezes a caller's frozen master) |
| S5 | `SpectralLinear::retract` | `src/lib.rs:662-669` | the per-step entry point: U then V, `Param::from_mapped_value` with the **same** ParamId and mapper | — | **(c)** | `tsct_retract_restores_ortho` (core) | correct. Preserving ParamId is load-bearing: fresh ids would silently reset every factor's optimizer momentum (`model.rs:429-437`) |
| S6 | `SpectralMoE::retract` | `src/lib.rs:1275-1282` | same, for the MoE arm | — | **(c)** | — (arm unwired) | correct by copy; unexercised in any run |
| S7 | `ortho_error` | `src/lib.rs:344-360` | `‖UᵀU−I‖_F`, **raw**, plus a host read | — | **(c)** | — | correct, but a **naming/convention trap**: the raw F-norm here, and the *per-entry* convention that the latch's 1e-3 is stated in lives in the **caller** (`param.rs:210-213` divides by k). Same trap in `burn-sct`'s `ortho_error` (C4). See F7 |
| S8 | `qr_householder` | `src/lib.rs:689-718` | on-device Householder QR, **orthonormal init only** | LAPACK `dgeqrf` scheme, our transcription | **(b)** | — | correct as far as it goes, and it is the **documented source of the cross-process TSCT residue** (AGENTS.md 3.7: nondeterministic reductions at a *random iteration* 0…55, median 2). It reads 7 host scalars per column and materialises an `m×m` eye per column — O(n·m²) work and 67 MB for a `[4096,64]` factor, once per run. See F4 |
| S9 | `ternarize` | `src/lib.rs:51-56` | `sign(w)·mean(|w|)`, dead zone at `0.7·mean` | BitNet b1.58 (arXiv:2504.12285) for the absmean-STE; **the 0.7 dead zone is attributed to a "burn-es convention"** | **(a)** for the ternary form / **(c)** for the 0.7 | — for the 0.7 | the 0.7 is a number with no external reference and no gate. It decides which 30% of entries are zeroed, in every forward. See F8 |
| S10 | `ste_ternary`, `ste_ternary_annealed` | `src/lib.rs:60-82` | STE wrappers | BitNet b1.58 | **(a)** | — | correct |
| S11 | `ternarize_stochastic`, `ste_ternary_stochastic` | `src/lib.rs:95-113` | S3T: `sign(w)·scale·Bernoulli(|w|/scale)` | arXiv:2412.04787 | **(a-cite)** | — | correct; **not on the trainer's path** (`small.toml` sets no stochastic flag) |
| S12 | `ternarize_per_column` | `src/lib.rs:119-124` | per-column `mean(|col|)` instead of one global mean | our adaptation | **(c)** | — | correct; `per_column` is off for `small` |
| S13 | `quant_factor` | `src/lib.rs:612-654` | Fp8/Fp4 per-row absmax + STE; Bf16/Fp16 dtype casts; Fp32 raw | BitNet / standard | **(a-cite)** | `tsct_retract_restores_ortho` (round-trip) | **this is the path `small` actually runs** (`quant format: Fp8` on every log line). It is *scale-equivariant per row* — see the class-B section, that single fact decides what the retraction can and cannot buy |
| S14 | `gpu.rs::tsct_linear_kernel` | `src/gpu.rs:38+` | fused add-only ternary GEMM, 2 bits/weight | our kernel | **(c)** | — | unwired in the trainer (`TSCT_FUSED` default on but the fusion flip is parked) |
| S15 | `bf16_ops.rs::Bf16Matmul` | `src/bf16_ops.rs:22+` | real bf16 matmul in a hand-written autodiff node | Moonshot-style mixed precision | **(c)** | — | **blocked by AGENTS.md 2.1** (no bf16 tensor cores on this backend). The comment claiming the forward "executes in bf16 on tensor cores" is a claim about hardware we do not have |
| S16 | `moe_fused.rs` | `src/moe_fused.rs` (3243 lines) | fused rank-1 ternary MoE | our construction | **(c)** | — | not in the model |
| S17 | `infer.rs::pack_ternary` | `src/infer.rs:16+` | 2-bit pack for inference | — | **(c)** | — | generation path, not training |

## 2. burn-muon-plus — the third layer, for the record

| # | site | file:line | what it computes | source claim | tier | gate | verdict |
|---|---|---|---|---|---|---|---|
| M1 | `orthogonalize` | `src/lib.rs:127-146` | Muon's NS: transposes to the wide side, **Frobenius** normalise, Jordan triple `(3.4445, −4.7750, 2.0315)`, `ns_steps=8` | Jordan et al. 2024 / arXiv:2602.21545 App. **D.1** | **(a)** | `normalization_matches_the_paper_equations` | correct. **A different polynomial from S1 on purpose** (D.1 vs D.3). Both citations are right; the crate pair is the thing to look at (AGENTS.md 3.7) |
| M2 | `if nc * 4 < nr` | `src/lib.rs:146` | the "factored" branch | — | — | — | **unsatisfiable** (proved in `reviews/muon-tsct-review-a.md` §1): the canonicalising transpose at `:131` can only reorder `(rows, cols)`. Dead branch + dead `ns_combine_cuda` |
| M3 | `norm_col` / `norm_row` / `ColRow` | `src/lib.rs:343-380` | Eq. (3)–(7) of the paper; the live order is `ColRow` | arXiv:2602.21545 Eq. (7) | **(a)** | `normalization_matches_the_paper_equations` | correct. **Load-bearing for the drift arithmetic in the class-B section**: ColRow makes every row of the update unit-L2, so `‖ΔU‖_F = lr·√k = 8e-4` at `lr=1e-4, k=64` |
| M4 | `lr_scaled = lr * (m/n).max(1).sqrt()` | `src/lib.rs:479` | Jordan's `max(1, m/n)^0.5`; the paper's Eq. (4) has no `max` | deviation, documented in place | **(c)** (declared deviation) | — | correct and honestly labelled |

## 3. burn-sct — the foundation layer. **Not in the dormouse build, and duplicated in it**

Everything in this section is reachable only from `burn-spectral`'s
`[dev-dependencies]` + `examples/tsct_diag.rs`. **No training run has ever
executed a line of this crate** (grep: the only non-`burn-sct` references to it
in the whole vendor tree are `burn-spectral/Cargo.toml:33` and that example).

| # | site | file:line | what it computes | source claim | tier | gate | verdict |
|---|---|---|---|---|---|---|---|
| C1 | `SctLinear::new` → `random_orthonormal` → `orthogonalize_cpu` → `qr_cpu` | `src/lib.rs:46-56, 241-247, 323-334` | init: Gaussian → CPU Householder QR, through `into_data()` (a host round-trip on the init path) | — | **(c)** | `orthonormal_init` (513) — tolerance **0.1** on the diagonal of `UᵀU` | works; the gate cannot fail on a 1% error. See F9 |
| C2 | `SctLinear::forward` | `src/lib.rs:58-79` | `y = (x@U)·s @ Vᵀ`, paper order; CUDA arm via `forward_cuda` | SCT (arXiv:2604.00733) | **(a-cite)** | `forward_shape` (500) — a shape check | correct; the shape check says nothing about the values |
| C3 | `SctLinear::retract` | `src/lib.rs:81-118` | the paper's Eq. 5 QR retraction on both factors, `Param::consume`/`from_mapped_value` to keep ParamId; `mem::replace` to keep the refcount at 1 (a shared NdArray tensor sends `into_data` down ndarray's slow copy, ~15 ms/QR at k=128) | Eq. 5 | **(a-cite)** | `retract_restores_ortho` (529) — tolerance **0.1** | correct; the refcount comment is a real measured gotcha. On CUDA the two QRs are launched sequentially **on purpose** (two host threads on one client serialise badly) |
| C4 | `ortho_error` | `src/lib.rs:120-139` | max over U,V of `‖MᵀM−I‖_F`, raw, host read per factor | — | **(c)** | `ortho_error_decreases` (551) | correct; a *relative* test, so it cannot catch a uniformly bad retraction. Same raw-vs-per-entry trap as S7 |
| C5 | `from_dense` / `from_dense_with_iters` | `src/lib.rs:141-223` | truncated SVD by QR reduction + one-sided Jacobi on `R` ("LAPACK gesdd scheme", "~2.4× less flops") | LAPACK scheme | **(b)** | `from_dense_roundtrip` (652), `svd_cpu_roundtrip_*` (580-644) | correct; the 2.4× is a **derived estimate with no measurement attached** |
| C6 | `orthogonalize_cpu` | `src/lib.rs:318-334` | `Q,R = QR(U); U ← Q·sign(diag R)` | "Paper Eq 5 … matches the paper's `safe_qr` (PyTorch `linalg.qr` + sign flip)" | **(a-cite)** | `sign_correction` (672) — tolerance 0.1 | correct. The crate states plainly: "**There is NO comparison against the authors' code in this crate**" (`:321-322`) |
| C7 | `svd_cpu` (one-sided Hestenes Jacobi, f64 internal) | `src/lib.rs:378-488` | exact SVD of the `k×k` `R`, top-k by index sort; `JACOBI_EPS = 1e-12` | standard | **(b)** | 3 round-trip tests | correct. The "f32 would land at ~1e-3" note is arithmetic, and it is why the gate tolerance is 1e-3 |
| C8 | `matmul_rt` | `src/lib.rs:274-316` | `A·B` with `B` transposed once, AVX2 dots, row-split threads | — | **(c)** | — | correct |
| C9 | `bytes_f32` | `src/lib.rs:254-258` | zero-copy `&[f32]` over `TensorData` bytes, `debug_assert`ed alignment | — | **(c)** | — | correct; **safety rests on two `debug_assert`s**, i.e. on nothing in release. 16-byte alignment is allocator policy, not a guarantee |
| Q1 | `dot_pair`, `apply_pair`, `hsum` | `src/qr.rs:32-83` | AVX2/FMA f32 dot and axpy; 8-wide then scalar tail | — | **(c)** | — | correct; the reduction ORDER is 8 partial sums + tail, and it differs from the scalar path — a reason the two paths are not bit-equal by construction |
| Q2 | `dot3_avx_f64`, `rotate2_avx_f64`, `hsum_pd` | `src/qr.rs:88-151` | f64 SIMD for the Jacobi sweep | — | **(c)** | — | correct |
| Q3 | `r_pass` | `src/qr.rs:231-402` | Householder R pass, LAPACK `dgeqrf` layout (reflectors in `R`'s storage, `tau`, `R[i][i] = sign·‖·‖`); a wavefront parallel branch with raw pointers + `Barrier` | LAPACK | **(b)** | — | correct, and the parallel branch's "bit-identical to sequential" claim is argued from disjointness + the barrier, which is a real argument. **Ungated** |
| Q4 | `r_k_from_r` | `src/qr.rs:407-416` | extract `R` from the column-major reflector storage, applying the same `sign(diag R)` flip | — | **(a-cite)** | indirect | correct; the flip is what makes `A = Q·R` survive |
| Q5 | `q_pass` | `src/qr.rs:422-527` | build `Q` back-to-front (LAPACK `orgqr`), no `m×m` intermediate, then the sign flip at `:509-517` | LAPACK | **(b)** | `retract_restores_ortho` (loose) | correct; O(m·k²) is the right shape for m ≫ k |
| Q6 | `cholesky_host` | `src/qr.rs:536-556` | upper Cholesky of the Gram, f32, `d = max(acc,1e-30).sqrt()` | — | **(c)** | `gpu_retract_matches_cpu` (1e-4) | correct. The clamp is the honest bit: "G may be rank-deficient at f32 rounding" |
| Q7 | `cholesky_host_par` | `src/qr.rs:562-610` | f64 parallel Cholesky, "~0.5–1 s for m=4096" | — | **(c)** | — | **(d) DEAD**: `pub`, zero callers in the whole vendor tree |
| Q8 | `qr` (tensor-op batched QR, 80 lines) | `src/qr.rs:624-688` | burn-main's QR, reduced to O(m k²) | burn-rs/burn, MIT | **(a)** (upstream code) | — | **(d) DEAD**: zero callers. Its doc calls it "the reference/GPU path"; nothing calls it |
| Q9 | `qr_cpu` | `src/qr.rs:612-618` | `r_pass` + `r_k_from_r` + `q_pass` | — | — | indirect | correct; the retraction's CPU entry point |
| K1 | `sct_qr_gram_kernel` | `src/qr_cuda.rs:94-115` | one-sided Gram, one thread per `(i,j)`, writes both triangles; carries a dead `_epoch: u32` multiplied by `0.0` | — | **(c)** | — | **(d) DEAD** (no caller) |
| K2 | `sct_qr_qsolve_kernel` | `src/qr_cuda.rs:122-188` | `Q = A·R⁻¹` by forward substitution on `Rᵀ`, one thread per row, 4 columns unrolled | — | **(c)** | `gpu_retract_matches_cpu` | correct. I re-derived the transposition: each row solves `Rᵀqᵀ = aᵀ`, so `q = aR⁻¹` — the right answer, and the non-obvious part |
| K3 | `sct_gemm_kernel`, `sct_gemm_t_kernel` | `src/qr_cuda.rs:198-288` | tiled 16×16 GEMM, `float4` along K, optional per-column scale | — | **(c)** | `gpu_forward_matches_tensor_path_non_pow2` | correct. "50–100× faster than cubecl's matmul on skinny shapes (4.8 ms → ~50 µs)" is a **recorded measurement on shapes we do not run** (m=64) |
| K4 | `forward_cuda` | `src/qr_cuda.rs:295-371` | the fused TSCT forward, two launches, shape gate `b%16, k%4, m%4`, no host sync | — | **(c)** | `gpu_forward_matches_tensor_path_non_pow2` | correct, with a `TODO(gpu): re-verify on device` on the no-sync claim (`:368-369`) — i.e. the FIFO-ordering argument is **unverified**, and `retract_cuda`'s `block_on(client.sync())` is what the comment calls the one per-step barrier |
| K5 | `sct_jacobi_round_kernel`, `sct_jacobi_vpass_kernel` | `src/qr_cuda.rs:378-475` | round-robin one-sided Jacobi with deferred V rotations | — | — | — | **(d) DEAD** (~100 lines). Replaced by the host path, per `from_dense_cuda`'s own comment |
| K6 | `sct_cast_f64_kernel`, `sct_transpose_kernel` | `src/qr_cuda.rs:479-501` | f32→f64 cast, generic transpose | — | — | — | **(d) DEAD** |
| K7 | `from_dense_cuda` | `src/qr_cuda.rs:515-557` | **a CUDA-gated function that is entirely host-side**: `into_data()` on the device tensor, then `host_svd::svd_host` in f32, `sweeps.max(15)` | the vendored Golub-Kahan + dbdsqr | **(b)** | — | correct but **the name and the gate both lie about where the work happens**; "~650 s → ~5–10 s at LLM scale" is unmeasured here. See F10 |
| K8 | `retract_cuda` | `src/qr_cuda.rs:559-612` | Gram via the backend matmul → host Cholesky → one fused solve kernel → `block_on(sync)`; shape gate `m≥256, k≥16, k≤1024` | "the paper's safe_qr … orthonormal Q with non-negative R diagonal" | **(a-cite)** | `gpu_retract_matches_cpu` (1e-4, GPU-only target) | correct. **This is the kernel that would replace the trainer's 880-launch retraction, and it is unreachable from the trainer** — see the seam section |
| H1 | `svd_host` | `src/host_svd.rs:36-162` | f64 Golub-Kahan + dbdsqr driver | vendored from the burn `linalg::svd` PR (same author) | **(b)** | **none** (0 `#[test]` in the file) | 745 lines of f64 SVD with **no in-crate test and no external reference**. See F11 |
| H2 | `bidiag_host` | `src/host_svd.rs:168-606` | Householder bidiagonalisation, raw-pointer + barrier protocol | "mirroring the tensor-op version operation for operation" | **(b)** | **none** | as H1; the SAFETY argument is written down (`:12-15`), which is better than nothing |
| H3 | `dbdsqr`, `dlas2_smax`, `dlartg` | `src/host_svd.rs:612-745` | shifted QR on the bidiagonal, LAPACK `dlas2`/`dlartg` scaling | LAPACK | **(b)** | **none** | as H1. The overflow-safe scalings are transcribed correctly by inspection |

## 4. The one composition that does exist

| # | seam | file:line | what composes | verdict |
|---|---|---|---|---|
| X1 | `LinearLike` → `SpectralLinear` | `crates/dormouse-core/src/param.rs:6` | the trainer's TSCT linear **is** `burn_spectral::SpectralLinear`; the retraction it runs is `SpectralLinear::retract` → `polar_retracked` → `polar_orthogonalize`, i.e. **S1**, on 16 factors per step | **live, and it is the only one.** The module doc on `param.rs:1` says "TSCT linear via **burn-sct** `SpectralLinear`" — wrong crate, and the third name in a chain that also has `burn-spectral`'s own header calling itself `burn-tsct` |
| X2 | `SpectralLinear::retract` → `burn_sct::orthogonalize` | — | **does not exist.** `retract` takes no backend parameter at all, so it cannot call `burn_sct::orthogonalize::<B>`, and `burn_sct::qr_cuda::is_cuda::<B>()` compares `B`'s TypeId against the **bare** `CubeBackend` while the trainer's device is `Autodiff { device: Cube(Cuda(0)) }` — the same reachability wall as `burn-rmsnorm` (AGENTS.md 3.3) | the retraction the trainer runs is **not** the QR retraction the SCT paper specifies; it is a Newton-Schulz substitute that the crate's own header calls "replaces the CPU QR of SCT, which cost 40-50% of a step" |
