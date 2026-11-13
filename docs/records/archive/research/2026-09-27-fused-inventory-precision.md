# burn-fused inventory — slice: PRECISION + small op crates

Date 2026-09-27. Scope: `burn-bitnet`, `burn-sct`, `burn-spectral`, `burn-rmsnorm`, `burn-rope`,
`burn-swiglu` (6 of the 26 crates in `vendor/dormouse-fused/`, the crate to be renamed
`dormouse-fused/` per ADR-0017). All arXiv IDs below were fetched from arxiv.org and confirmed.

Read-only pass. Nothing modified. All GPU claims are traced statically against
burn 0.22.0-pre.4 source unless marked MEASURED; the GPU was occupied by a live
`train` run for most of the session (documented inline).

---

## 0. The one finding that dominates this slice

Every fused kernel in `burn-rmsnorm`, `burn-swiglu`, `burn-spectral::gpu` and
`burn-bitnet::fwt_cuda` (non-autodiff entry points) is reached through this idiom:

```rust
let prim = t.clone().try_into_primitive::<burn_cubecl::CubeBackend>().ok()?;
let c = (&prim as &dyn Any).downcast_ref::<CubeTensor>()?;
```

`burn-dispatch-0.22.0-pre.4/src/tensor.rs:481-487` (MEASURED — read, not inferred):

```rust
fn try_into_backend(tensor: DispatchTensor) -> Result<BackendTensor<$backend>, String> {
    if tensor.autodiff != DispatchAutodiffContext::Disabled {
        return Err(format!("Expected concrete {} backend with disabled autodiff context, got {:?}", ...));
    }
    ...
```

Under our trainer's `Autodiff<burn_cuda::Cuda, BalancedCheckpointing>`, *every* tensor on the
forward path has `autodiff == Enabled(...)`. So `try_into_primitive::<CubeBackend>()` returns
`Err`, the `?` yields `None`, and the caller falls through to the tensor path. **No fused kernel
that uses this idiom can ever be taken in production.** The `fused` field on
`SpectralLinear` is dead for the same reason plus a second one (see §2).

Only three call sites in the whole 26-crate library implement the escape hatch — a real custom
`Backward` op that peels the autodiff wrapper (`burn_autodiff::ops::Backward`):

- `burn-rope::rope_cuda::rope_autodiff`  (fwd + bwd kernels, `rope_backward_kernel`)
- `burn-bitnet::fwt_cuda::quant_autodiff` / `fwt_autodiff`
- `burn-bitnet::sparse::weight_quant_masked_autodiff`

**None of the three is wired into the model.** "Bringing the library to quality" for this slice
means porting that autodiff-op pattern to `burn-rmsnorm` and `burn-spectral`; the raw
`try_into_primitive` kernels are inference-only decoration as written.

---

## 1. burn-bitnet — 1815 LOC — VERDICT: WIRED (indirectly)

**What it is.** The BitNet-family quantizer library: ternary `Q_w(W) = α·RoundClip(W/(α+ε),−1,+1)`
with `α = mean|W|` for weights, absmax-8bit / absmean-4bit for activations, plus a Fast
Walsh–Hadamard rotation for outlier suppression and an N:M semi-structured sparsity layer.

**arXiv** (all VERIFIED against arxiv.org):
- `2504.18415` BitNet v2: Native 4-bit Activations with Hadamard Transformation (9 refs in code)
- `2402.17764` BitNet b1.58: All LLMs are in 1.58 Bits (5 refs)
- `2603.05168` Sparse-BitNet: 1.58-bit LLMs are Naturally Friendly to Semi-Structured Sparsity (3 refs)
- `2504.12285` BitNet b1.58 2B4T Technical Report (1 ref)
- `2407.09527` BitNet b1.58 Reloaded (1 ref)

**Headline results.** b1.58 matches full-precision Transformer LLMs at equal size and training
tokens. BitNet v2 trains with native 4-bit activations with minimal degradation. Sparse-BitNet
reports smaller degradation than fp32 baselines at equal sparsity and up to 1.30× speedup.

**Size and shape.** 1815 LOC across 8 files (`quant/weights.rs` 186, `quant/activations.rs` 187,
`fwt.rs` 155, `fwt_cuda.rs` 577, `sparse.rs` 602, `precision.rs` 69, `lib.rs` 34, `quant/mod.rs` 5).
Fused CUDA: yes — `fwt_cuda.rs` (FWT + fused quantize, with a hand-written autodiff op) and
`sparse.rs` (`SparseBitLinear` + `weight_quant_masked_autodiff`). 27 tests.

Public API someone would actually call: `weight_quant_ternary` / `weight_quant_b158`,
`quantize_tensor::<B>(x, bits)`, `activation_quant_8bit`, `bitnet_v2_quantize`,
`weight_quant_2bit`, `weight_quant_ternary_nm`, `fast_walsh_hadamard`, `kernel_precision`.

Backend features: `default = ["std"]`; `cuda` (burn-cuda + burn-cubecl + cubecl + `burn/cuda`);
`autodiff`.

**State.** Builds clean, no cuda (MEASURED: `cargo check -p burn-bitnet` OK).
`cargo test -p burn-bitnet --lib` → **17 passed, 0 failed** (MEASURED). Note only 17 of the 27
`#[test]` markers run without `cuda`; the CUDA ones are gated. No bit-for-bit external reference
harness (the `weights.rs` tests check the closed-form RoundClip algebra and the STE gradient is
exactly identity, which is the right kind of check but not a cross-implementation diff).

**Wiring.** 3 direct call sites, and it is easy to under-count this:
- `crates/dormouse-train/src/lib.rs:748` — `burn_bitnet::quantize_tensor::<Backend>(u, qfmt.bits())`
  inside the `--quant-check` diagnostic block (prints factor quant error). Not the training forward.
- `crates/dormouse-core/examples/quant_probe.rs:19,40` — same call, in an example.
- **The real wiring is indirect**: `burn-spectral::SpectralLinear::quant_factor`
  (`burn-spectral/src/lib.rs:563,567`) calls `burn_bitnet::quantize_tensor` for Fp8/Fp4, and
  `SpectralLinear::forward` (`:475,489,494`) calls `burn_bitnet::weight_quant_2bit` and
  `weight_quant_ternary_nm`. So every one of the 6 `LinearLike` layers in the model runs
  burn-bitnet code on the training forward path.

**Precision blockers.** None. Every quantizer is f32 tensor arithmetic (`abs`/`mean`/`round`/
`clamp`/`div`), no dtype casts, no Bool→Float, no matmul. `precision.rs::Precision::Fp8` is an
explicit, honest stub: it prints a one-time warning and falls back to fp32, with a comment
recording that burn 0.22 has no `FloatKind::F8`. The `fwt_cuda` fused entry points use the
§0 idiom and are unreachable under autodiff, but `quant_autodiff` is a proper custom op and
would work if wired.

---

## 2. burn-spectral — 6172 LOC src (8298 with examples) — VERDICT: BROKEN

The important one. It carries the TSCT polar retraction and the `QuantFormat` enum.

**What it is.** A ternary re-parameterization of SCT: the dense `W` is never materialized, only
orthonormal masters `U [in,k]`, `V [out,k]` and scales `s [k]`, and the forward uses their
*ternary* STE projections `{(−1,0,+1)·scale}` instead of the fp32 masters. Orthonormality is
maintained by a Newton–Schulz polar retraction on device. Also ships a rank-1 ternary MoE
(`SpectralMoE`) and a packed-ternary inference path.

**arXiv** (all VERIFIED): `2604.00733` SCT (Kohlberger et al., 1 Apr 2026) — *Permanent
Truncated SVD + Stiefel QR retraction; up to 199× memory reduction per MLP layer at rank 32;
rank 128 the efficiency sweet spot at 11.7× MLP compression*; `2504.12285` 2B4T; `2504.18415`
BitNet v2; `2412.04787` Direct Quantized Training with Stochastic Rounding; `2603.05168`
Sparse-BitNet; `2602.21545` Muon+ (NS coefficients); `2202.09368` ST-MoE (Expert-Choice).

**Size and shape.** src: `lib.rs` 2418, `moe_fused.rs` 3086, `infer.rs` 328, `gpu.rs` 179,
`bf16_ops.rs` 161 = 6172. Examples are 2126 lines and dominated by `tsct_diag.rs` (1768).

Fused CUDA: yes, three distinct mechanisms —
- `gpu.rs::tsct_linear_cuda` — 2-bit-packed ternary GEMM, one launch, both stages
- `moe_fused.rs` — MoE forward + exact backward + fused router
- `bf16_ops.rs::bf16_matmul` — custom autodiff op: bf16 GEMM forward, exact fp32 backward

Public API someone would actually call: `SpectralLinear::{new, forward, forward_quant,
forward_quant_bf16, retract, set_quant}`, `ortho_error`, `polar_orthogonalize`,
`retract_batched`, `QuantFormat`, `bf16_ops::bf16_matmul`.

Backend features: `default = ["std"]`, `cuda` (which also pulls `burn/autodiff`).
dormouse enables `burn-spectral/cuda` (`crates/dormouse-core/Cargo.toml:40`), so `gpu.rs`,
`moe_fused.rs` and `bf16_ops.rs` are all *compiled into the trainer binary*.

**State — BROKEN.** Three separate defects:

1. **Its own test suite is red.** MEASURED: `cargo test -p burn-spectral --lib` →
   **33 passed, 3 failed**. The three are `tests::polar_retracts`,
   `inference_tests::to_inference_matches_trained_layer`, and
   `inference_tests::to_inference_matches_per_column_layer`. All three call `layer.retract(n)` on
   a `Device::ndarray()` (non-autodiff) tensor and panic in
   `burn-tensor-0.22.0-pre.4/src/tensor/api/float.rs:1176` with
   *"Tensor::require_grad requires autodiff; call Tensor::autodiff first"*.
   Root cause, `burn-spectral/src/lib.rs:593-601` and `:1211-1219`:

   ```rust
   let u_ret = polar_orthogonalize(u_val, iters).detach().set_require_grad(true);
   ```

   `set_require_grad(true)` is called unconditionally. burn 0.22.0-pre.4 asserts
   `!require_grad || tensor.autodiff != Disabled` — a hard panic. **The retraction mechanism the
   model depends on most has never been run against a non-autodiff device.** Our trainer is
   `Autodiff<Cuda, BalancedCheckpointing>`, where `Enabled` holds and the call is legal, so
   **production is unaffected** — but the crate cannot pass its own tests, and the fix is
   already written elsewhere in this very library (see `burn-rmsnorm` below and
   `burn-sct/src/lib.rs:325-333`, which does `let require_grad = matrix.is_require_grad(); ...
   .set_require_grad(require_grad)`). Three crates, two workarounds, one missing.

2. **The test target cannot be built from clean.** dev-dependency `burn-muon-plus` is *pristine*
   in git and does not compile against burn 0.22.0-pre.4: 2× `E0308` at
   `burn-muon-plus/src/lib.rs:364` — `Tensor<1>.mul(Tensor<2>)`, because `Tensor::mul` is now
   rank-generic (`numeric.rs:225`, `pub fn mul(self, other: Self)`). MEASURED. This bit during
   the session only because a sibling agent's `Cargo.lock` edit forced a rebuild; from a clean
   target dir `cargo test -p burn-spectral` cannot build. `cargo check -p burn-spectral` (lib
   only) is fine.

3. **A dead field and an unreachable kernel.** `SpectralLinear` has a
   `pub fused: bool` (default `true`, `:406`) with a setter `set_fused` (`:456`) and a doc comment
   promising "the fused CUDA training kernels when eligible". `SpectralLinear::forward` (`:461`)
   **never reads `self.fused`** and never calls `gpu::tsct_linear_cuda`. Grep confirms
   `tsct_linear_cuda` has exactly one caller in the whole crate: `examples/gpu_check.rs:80,90`.
   So `gpu.rs` (179 lines) is reachable only from an example, and `SpectralLinear::fused` is a
   write-only field. `moe_fused.rs` (3086 lines, the largest file in the library) is reachable
   only from `SpectralMoE::forward` (`:1017,1021,1062`) — and the model never constructs a
   `SpectralMoE` (grep: zero references anywhere in `crates/dormouse-*`).

**Bit-for-bit verification.** Three ndarray host-reference harnesses, all good:
`two_bit_forward_matches_reference` (`:1473`), `nm_forward_is_sparse_and_matches_reference`
(`:1505`), `tsct_moe_ec_matches_reference` (`:1878`, a full host reimplementation of the
Expert-Choice pipeline). **None of them covers `SpectralLinear::forward` — the one function the
model actually runs — and none runs on CUDA.** The 6-way `TSCT_FUSED` escape hatch and the
packed-ternary `infer.rs` path are unverified against anything external.

**Wiring — 22 call sites, 6 model sites.** The model wires `SpectralLinear` 6 times through
`crates/dormouse-core/src/param.rs` (`LinearLikeInner::Tsct`): the per-expert `gate_up` and
`down`, `out_proj`, `lm_head`, and the controller/router linears. Functions used:
`SpectralLinear::new` (param.rs:38), `forward` (param.rs:105), `forward_quant::<B>` (param.rs:97,103),
`forward_quant_bf16::<B>` (param.rs:92), `retract` (param.rs:140, from `retract_tsct`, called
every step at `dormouse-train/src/lib.rs:948`), `set_quant`, `ortho_error` (param.rs:159-160,
the per-entry `max_ortho` monitor), `QuantFormat` (10 references), `bf16_ops::bf16_matmul`
(`examples/gemm_probe.rs:26` only). Nothing else in the crate is touched: no `SpectralMoE`, no
`to_inference`, no `retract_batched`, no `polar_orthogonalize_batched`, no `set_alpha` /
`set_stochastic` / `set_per_column` / `set_asym` / `set_nm` / `set_2bit` / `set_fused` / `set_2bit`.

**Correction to the brief:** the *per-SM factor-quant format selection is not in burn-spectral*.
burn-spectral only supplies the `QuantFormat` enum (`:291-309`) and a `bits()` helper. The
per-SM policy lives in our own code: `crates/dormouse-train/src/lib.rs:317 quant_format()`
(`sm >= 120 → Fp8`, `bf16 → Bf16`, else `Fp32`) with `sm_of()` at `:347` reading
`CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_*` through cudarc. The library half is a 19-line enum.

**Precision blockers — this crate is the epicentre.**

- **Bool→Float: partially hit, and inconsistently within one file.** `ternarize` (`:54`) uses
  `mag.greater(...).int().float()` — the two-step form, and this is the one that demonstrably
  works in production (it is the default TSCT forward on every one of our 6 layers). But
  `ternarize_per_column` (`:122`) and `ternarize_stochastic` (`:104`) use a bare `.float()` on
  the Bool. Per the bug documented at `crates/dormouse-train/src/lib.rs:731` ("on the dispatch
  path both the Bool->Float cast and `zeros_like().mask_fill(m, 1.0)` fed into `add` reported
  1.0 for every step of a perfectly finite loss, official_v5, twice"), the bare form is the
  suspect. Two sibling functions, same file, two different spellings of the same cast, is the
  signature of a bug that was hit once and fixed in one place only. Both are currently
  unreferenced by the model, which is luck, not design.
- **bf16 matmul: hit, and the workaround is still broken.** `bf16_ops::bf16_matmul` exists
  solely to route around it: it peels the autodiff wrapper, casts both operands to `BF16`, calls
  `.matmul()` on the *inner* backend, casts the result back to F32, and re-wraps with a custom
  `Backward` that does fp32 matmuls. Our own probe records that it does not work:
  `crates/dormouse-core/examples/gemm_probe.rs:50` — *"bf16 tensor-core path: BROKEN on pre.4 +
  cuda (its own tests in burn-spectral fail at burn-cubecl ops/tensor.rs:150), so it is not timed
  here - see docs/architecture/PLAN.md OPTIMIZATION BLOCKERS"*. The probe gates the path behind
  `GEMM_PROBE_BF16`. So `forward_quant_bf16` — reachable from production via
  `param.rs:92` whenever `--bf16` is on and the per-entry `bf16_compute` flag is set
  (`dormouse-train/src/lib.rs:473`) — calls a function whose own test in the same crate fails.
  The separate, *working* mitigation is the f32 cast at the top of `forward` and
  `forward_quant` (`:462-468`, `:518-524`, "bf16 activations times fp32 factors produce NaN in
  the matmul on this stack (measured 2026-08-29); compute in fp32").
- **f16 matmul: hit, unguarded.** `QuantFormat::Fp16` (`:561`) does
  `w.cast(F16).cast(F32)` then matmuls in the Fp32 path — a round trip that at least returns F32
  before the matmul, so `--quant fp16` may survive. But `forward_quant_bf16` has no f16 branch
  and nothing in the crate guards Fp16 against the `builtin.fp16 to implement dyn SizedType`
  class of failure. No `#[cfg]`, no runtime check, no test.

---

## 3. burn-sct — 2738 LOC src (3350 with examples+tests) — VERDICT: SUPERSEDED (by burn-spectral)

**What it is.** Spectral Compact Training proper: replace a dense `W [m,n]` with permanent
truncated-SVD factors `W = U diag(s) Vᵀ` where `U,V` are retracted onto the Stiefel manifold by
QR after every optimizer step, so the dense matrix is never materialized in training or inference.

**arXiv** `2604.00733` — **VERIFIED**: *Spectral Compact Training: Pre-Training Large Language
Models via Permanent Truncated SVD and Stiefel QR Retraction* (Kohlberger, EctoSpace).
Headline: up to 199× memory reduction per MLP layer at rank 32, 7.2 GB peak for a 70B-parameter
training step on a Steam Deck (vs 1,245 GB dense FP32+Adam), 11.7× MLP compression at rank 128
with the best perplexity, 46% GPU memory drop at rank 32 with doubled throughput.

**Size and shape.** src: `lib.rs` 693, `qr.rs` 687, `qr_cuda.rs` 613, `host_svd.rs` 745 = 2738.
Tests 259, examples 371. Fused CUDA: yes — and this is the crate that gets it *right*.
`qr_cuda.rs` computes `G = AᵀA` with the backend's cuBLAS-class matmul, does the `k×k` Cholesky on
the host, then a custom forward-substitution kernel with a 4-wide column block. Its own doc
records the honest failure mode: *"G = A^T·A squares the condition number"*.
`host_svd.rs` is a full f64 Golub–Kahan + dbdsqr SVD on the host, vendored specifically to keep
`from_dense` off the GPU.

Public API: `SctConfig::new`, `SctLinear::{new, forward, retract, ortho_error, from_dense,
from_dense_with_iters, param_count, compression_ratio}`, `qr::qr_cpu`.

Backend features: `default = ["std"]`, `cuda`, `autodiff`, `serde`, `binary-tests`.

**State — the healthiest crate in this slice.** Builds clean (MEASURED).
`cargo test -p burn-sct --lib` → **13 passed, 0 failed** (MEASURED). 17 `#[test]` markers total
including the integration tests.
**Best bit-for-bit verification in the whole library, two harnesses:**
- `tests/cuda_retract.rs::gpu_retract_matches_cpu` — MEASURED-gated on `cuda`; runs the full
  layer path on GPU at `[1024, 128]`, retracts, and diffs against `qr::qr_cpu` (LAPACK scheme)
  on the same data, asserting `max_diff < 1e-4`. This is a genuine GPU-vs-CPU-reference check.
- `tests/cmp_reference.rs` — reads pre-generated reference binaries
  (`u, v, s, x, y, u_pert, v_pert, u_ret, v_ret, w_dense, w_recon`, written by `gen_reference.py`)
  and diffs every one, including the retraction. Gated behind the non-default
  `binary-tests` feature, so it does not run in CI-by-default, but the harness exists.

**Wiring — ZERO.** `grep -rn 'burn_sct\|burn-sct' crates/` returns nothing. No dependency edge
in any `crates/dormouse-*/Cargo.toml`. Two stale traces of it survive:
- `crates/dormouse-core/src/param.rs:1` — *"param - TSCT linear via **burn-sct** SpectralLinear"*.
  Wrong crate name; the type is `burn_spectral::SpectralLinear`.
- `burn-spectral/src/lib.rs:20` — *"replaces the CPU QR of SCT, which cost 40-50% of a step"*.

**Verdict: SUPERSEDED (by burn-spectral's `polar_orthogonalize`).** burn-spectral keeps the
`W = U diag(s) Vᵀ` forward verbatim (same three lines, `SpectralLinear::forward:502-505` vs
`SctLinear::forward:76-78`) and replaces the retraction: NS polar iteration on device instead of
QR. burn-spectral is a *strict superset* of burn-sct's training role, plus ternary, plus a MoE.
The only thing burn-sct still has that burn-spectral does not is `host_svd.rs` (f64 host SVD for
`from_dense`, a checkpoint-conversion utility) and its two reference harnesses.

**Precision blockers.** None. f32/f64 arithmetic only: LAPACK on the host, Gram+Cholesky+
substitution in f32 on the GPU, f64 on the host for SVD. No dtype casts, no Bool→Float, no
matmul beyond the `AᵀA` that the backend provides in its native f32. The one numerical caveat
is self-documented and structural: the Gram squares the condition number, which is exactly the
error the NS path in burn-spectral also has to manage (see the sigma_max power-iteration
scaling comment at `burn-spectral/src/lib.rs:149-175`, which records a measured
`polar([512,512], 3) -> max entry ~1e14` divergence without it).

---

## 4. burn-rmsnorm — 235 LOC — VERDICT: WIRED (kernel dead under the trainer)

**What it is.** RMS normalization, `x / sqrt(mean(x²) + ε) * w`, as a burn `Module` with a single
learned gain of length `d_model`. That is the whole crate.

**arXiv** `1910.07467` — **VERIFIED**: *Root Mean Square Layer Normalization* (Zhang & Sennrich,
NeurIPS 2019). Headline: comparable performance to LayerNorm at 7%–64% less runtime; re-scaling
invariance and implicit learning-rate adaptation; the re-centering step is dispensable.

**Size and shape.** 121 (lib) + 114 (fused). Fused CUDA: yes — one cube per row, 256 threads,
`Shared<[F]>` partials with a log₂(256)=8-step binary tree reduction (the code comment records
that this replaced a serial accumulation by thread 0). API: `RMSNorm::new(d_model, eps, device)`,
`RMSNorm::forward(Tensor<3>)`, `fused::rmsnorm_cuda::<B>`. `forward` takes rank 3 and reshapes to
`[b*t, d]` internally. Features: `default = ["std"]`, `cuda`.

**State.** Builds clean (MEASURED). `cargo test -p burn-rmsnorm --lib` → **2 passed, 0 failed**
without cuda (MEASURED). 3 tests total.
**The cuda test is a tautology and has never tested the kernel.** `fused::cuda_tests::fused_matches_tensor`
builds on `Device::ndarray()` (`fused.rs:35-38`), so `rmsnorm_cuda` returns `None`, `forward`
takes the tensor branch, and the test compares the tensor path against a hand-recomputation of
the same tensor path. It is a self-consistency check wearing a CUDA label. **There is no test in
this crate that ever launches the kernel.**

**Wiring — 3 model sites.** `crates/dormouse-core/src/gr.rs:48` (one per GR branch),
`loop_block.rs:206` (the per-iteration pre-norm), `model.rs:54` (the final norm before
`lm_head`). Every norm in the model is this type.

**This crate carries the library's only real pre.3→pre.4 workaround** (`lib.rs:20-36`):

```rust
// pre.3 required require_grad here (Param::initialized inherits
// the flag; without it the weight froze — model_seam
// gradient_flow, 2026-09-21). pre.4 panics on require_grad for
// non-autodiff devices (raw-launch modules), and still needs the
// flag on autodiff devices — so branch on the device context.
weight: Param::initialized(ParamId::new(),
    if device.is_autodiff() { Tensor::ones([d_model], device).require_grad() }
    else { Tensor::ones([d_model], device) }),
```

The cost of getting this wrong is recorded in our own test:
`crates/dormouse-core/tests/model_seam.rs:336` — *"2 RMSNorm gains: burn-rmsnorm require_grad
bug"*. That is the bug the comment refers to, and it is why the branch exists.

**Precision blockers.** Clean, and it carries a correct one:
- The kernel is f32-only and **explicitly bails out** on any other dtype rather than
  reinterpreting a buffer (`fused.rs:88-92`): *"a bf16/f16 buffer reinterpreted as f32 would
  produce garbage silently, so bail out and let the caller's tensor path handle other dtypes"*
  (`if x_c.dtype != DType::F32 || w_c.dtype != DType::F32 { return None }`). This is exactly the
  right response to the broken bf16/f16 matmul paths: refuse, don't guess. `burn-swiglu` copied
  the same guard verbatim.
- Not affected by the Bool→Float cast (no comparisons; `powf_scalar(2.0).mean_dim(2).add_scalar(eps).sqrt()`).
- Per §0 the fused kernel is unreachable under `Autodiff<Cuda, BalancedCheckpointing>`, so this
  crate is a pure tensor-op dependency in production. 121 lines of lib.rs doing 5 tensor passes
  where the 114-line kernel beside it would do 1.

---

## 5. burn-rope — 975 LOC — VERDICT: IMPLEMENTED-UNUSED

**What it is.** Rotary position embedding (rotate half-pairs by a precomputed cos/sin table) plus
YaRN context extension via NTK-by-parts frequency interpolation with a temperature term.

**arXiv** — both **VERIFIED**: `2104.09864` RoFormer: Enhanced Transformer with Rotary Position
Embedding (Su et al.) — absolute position as a rotation matrix with explicit relative-position
dependency; and `2309.00071` YaRN: Efficient Context Window Extension of LLMs (Peng et al.) —
10× fewer tokens and 2.5× fewer training steps than prior methods to extend the window.

**Size and shape.** `rope_cuda.rs` 569, `lib.rs` 209, `rotate.rs` 125, `freqs.rs` 72.
Fused CUDA: yes — and it is **the only crate in this slice that gets the autodiff problem right**.
`rope_cuda::rope_autodiff<Inner>` is a real custom op: forward `rope_kernel`, backward
`rope_backward_kernel`, registered through `burn_autodiff::ops::Backward` with
`#[cfg(feature = "cuda")]` selecting the CUDA kernel and `#[cfg(not(feature = "cuda"))]`
falling back to tensor ops. API: `precompute_freqs`, `precompute_freqs_yarn`,
`apply_rope_3d`, `apply_rope_4d`, `RotaryEmbedding::{new, yarn, forward, forward_qk}`.
Features: `default = ["std"]`, `cuda`, `autodiff` (both needed for the fused path).

**State.** Builds clean (MEASURED). `cargo test -p burn-rope --lib` → **9 passed, 0 failed**
without cuda (MEASURED). 13 `#[test]` markers; 4 are `#[cfg(all(test, feature = "cuda"))]` or
`#[cfg(all(test, autodiff, cuda))]` and use `Device::default()` (MEASURED-gated, lines 473, 526, 527).

**Best numerical discipline in the slice.** `yarn_ramp_matches_documented_formula`
(`lib.rs:112-167`) recomputes the YaRN ramp in f64 on the host from the documented lines
(`r(d) = L·θ_d/(2π)`, `γ = clamp((r−βs)/(βf−βs),0,1)`, `f = ((1−γ)·θ/s + γ·θ)·temp`) and diffs
it in the **cos domain** — with the reason spelled out: *"wrap-safe, no acos∘cos round-trip:
below ~3e-4 rad an f32 cos collapses to 1.0, which once let an inverted ramp ship with green
tests"*. It pins the backend to ndarray because *"the ambient default backend may dispatch to a
kernel whose transcendentals are only ~1e-4-accurate"*. **That comment is the only place in the
26-crate library that admits a precision bug shipped with a green test suite.** It is also the
single most important sentence in this inventory for §6: an f32 `cos` collapsing to 1.0 for small
arguments is exactly the class of failure that makes "it trains, no NaN" a worthless signal.

**Wiring — ZERO.** `grep -rn 'burn_rope\|burn-rope' crates/` returns nothing. The model gets
RoPE from burn-kda (AGENTS.md §5 of the playbook: *"keep RoPE in the attention arm — NoPE →
endless generation after post-training"*).

**Precision blockers — structurally immune to all three.** RoPE is elementwise: slice, multiply,
subtract, concatenate. There is no matmul anywhere in the crate, so the broken bf16 and f16
matmuls cannot be reached, and there is no Bool→Float cast. The autodiff-aware kernel sidesteps
the mixed-dtype problem by construction rather than by cast-and-hope. This is the model the
other three crates should be ported to.

---

## 6. burn-swiglu — 187 LOC — VERDICT: IMPLEMENTED-UNUSED

**What it is.** The SwiGLU gated FFN: `SiLU(x·W_gate) * (x·W_up)` followed by `x·W_down`, with
the gate and up projections fused into one `[d_model, 2*hidden]` matmul.

**arXiv** `2002.05202` — **VERIFIED**: *GLU Variants Improve Transformer* (Shazeer, 2020).
Headline: testing GLU variants in the Transformer FFN, several yield quality improvements over
the standard ReLU/GELU.

**Size and shape.** `lib.rs` 113, `fused.rs` 74. Fused CUDA: yes —
`swiglu_cuda::<B>(gu: Tensor<2>, h)` → `[rows, h]`, one cube per row, 256 threads, computes
`silu(x) = x·σ(x)` inline so the sigmoid never round-trips through memory. The doc records the
win: *"one launch (vs ~4 tensor passes: slice, silu, slice, mul)"*. API: `swiglu_gate(Tensor<3>)`,
`SwiGLUConfig::{new, with_bias, init}`, `SwiGLU::forward`. Features: `default = ["std"]`, `cuda`.

**State.** Builds clean (MEASURED). `cargo test -p burn-swiglu --lib` → **2 passed, 0 failed**
(MEASURED). Both tests are ndarray **shape-only** assertions (`assert_eq!(dims, ...)`). Zero
CUDA tests — the kernel is never launched by any test in the crate.

**Wiring — ZERO.** No reference in `crates/dormouse-*`, no dependency edge. `loop_block.rs:317-321`
implements the gate by hand inside the expert FFN:

```rust
// Expert FFN: softmax blend of n_experts TSCT gate_up/silu/down
let mid = activation::silu(if bf16 { mid.cast(FloatDType::F32) } else { mid });
```

which is `burn_swiglu::swiglu_gate`'s tensor branch, inlined. Note also that `SwiGLU` holds
plain `burn::nn::Linear` (`lib.rs:45-46`), **not** `LinearLike`/TSCT — so even if wired it would
have had to drop the model's whole spectral parameterization. It is a leftover from before
`LinearLike` existed.

**Precision blockers — immune to all three.** The kernel is elementwise; `activation::silu` is
elementwise; there is no matmul inside `swiglu_gate` (the matmuls are in the two `Linear`s, which
are plain burn). The f32-only dtype bail-out is copied verbatim from burn-rmsnorm
(`fused.rs:54-58`, with a credit comment). Per §0 it is unreachable under autodiff regardless.

---

## 7. act_quant.rs vs burn-bitnet — the duplication question, line by line

`crates/dormouse-core/src/act_quant.rs` is 170 lines (48 of which are its two tests), and its own
doc comment claims: *"The weight side already exists (SpectralLinear's ternary/2-bit STE
quantizers in burn-bitnet). This module adds the activation side"*.

Taking that claim line by line, it is **partly right, and it points the merge in the opposite
direction from what one would assume.**

| act_quant construct | lines | burn-bitnet counterpart | verdict |
|---|---|---|---|
| `quant_act` scale+normalize+round+clamp+rescale+STE, `Int(8)`, `group=0` | 45-80 (~20 of the useful ones) | `quant/activations.rs:10-24` `activation_quant_8bit` | **DUPLICATE.** Same algorithm: per-row `abs().max_dim(1).clamp_min(ε)`, `·127`, `round()`, `clamp(-128,127)`, `÷127`, `·scale`, then the identical `x.add(y.sub(base))` STE. Only deltas: epsilon `1e-8` vs `1e-5`, and rank 2 vs rank 3. |
| `quant_act` `Int(4)` | 68-75 | `quant/activations.rs:120` `quantize_tensor(x, 4)` | **NEAR-DUPLICATE, DIVERGENT.** act_quant: absmax scale, `clamp(−l, l)` with `l = 2^(4−1)−1 = 7` → range [−7,7]. bitnet: **absmean** scale, `clamp(−8, 7)` → range [−8,7]. Two different 4-bit conventions behind the same user-facing flag. |
| `fp4_round` (e2m1 float4: log2 bucketing, mantissa threshold 1.25, 0.5 bucket, 0.25 zero cutoff) | 84-130 (**47 lines**) | **nothing** | **UNIQUE.** `grep -rniE 'e2m1|fp4|nibble' vendor/dormouse-fused/crates/` returns exactly one hit, and it is an unrelated mention of MXFP4 in `burn-situ/src/lib.rs:14`. e2m1 does not exist anywhere in the 26-crate library. |
| per-group scale branch (`reshape([b, d/g, g])` … `max_dim(2)` … `.repeat(&[1,1,g])` … `reshape([b,d])`) | 54-64 (11 lines) | **nothing** | **UNIQUE.** burn-bitnet has per-row (dim 1) and per-token only; the string `group` appears in it once, in an unrelated N:M doc comment (`sparse.rs:21`). |
| `ActFormat::attn()` (promote to ≥8 bits for the attention path) | 34-39 | **nothing** | **UNIQUE.** |
| `ActFormat` enum | 24-29 | **nothing** | **UNIQUE** (bitnet takes a bare `bits: usize` and infers the format). |

**Verdict: ~20 of ~100 useful lines are duplicated; ~60 are unique and load-bearing.**
`act_quant.rs` is not the redundant side — it holds the only e2m1 implementation and the only
group-scale implementation in the project. The correct consolidation is:

1. **Move `fp4_round` + the group branch INTO burn-bitnet** (as `quantize_act::<B>(x, bits,
   group)`), so `--quant fp4` and `--act-quant fp4` stop being two different algorithms.
2. **Delete one of the int8 absmax quantizers.** `activation_quant_8bit` (bitnet, rank 3) vs
   `quant_act(_, Int(8), 0)` (ours, rank 2) — same math. Our caller reshapes
   `[b,t,d] → [b*t,d]` and back (`loop_block.rs:244,249`), so once one of them takes the rank the
   caller wants, the reshape disappears too.
3. **Fix the naming collision first, because it is user-visible and actively misleading:**
   `burn_spectral::QuantFormat::Fp4` is **not** fp4. It routes to
   `burn_bitnet::quantize_tensor(w, 4)`, which is **int4 absmean** (`clamp(-8,7)`), while
   `ActFormat::Fp4` is **e2m1 float4**. Two unrelated quantizers answer to `--quant fp4` and
   `--act-quant fp4` respectively. Rename one of them before merging anything.

Note in passing that a *third* ternary quantizer exists: `burn-spectral::ternarize` (`:51`) is
`sign(w)·mean(|w|)·(|w| > 0.7·mean)` — dead zone at 0.7 — while `burn_bitnet::weight_quant_b158`
is the paper's `round(clamp(w/γ,−1,1))·γ` — dead zone effectively at 0.5, scale detached. They
disagree on the 0.5γ–0.7γ band. `burn-es::ternarize` is a fourth copy. Our model runs the
burn-spectral variant.

---

## 8. Precision blockers, consolidated

The three known backend bugs on this box, against the six crates:

| crate | `Bool → float` returns wrong value | bf16 matmul broken | f16 `SizedType` | workaround carried in code today |
|---|---|---|---|---|
| **burn-bitnet** | not reached | not reached | not reached | **None needed.** Pure f32 tensor arithmetic. `precision.rs::Precision::Fp8` is an honest stub: one-time warning, fall back to fp32, with a comment naming the missing `FloatKind::F8`. `fwt_cuda` fused paths are unreachable under autodiff (§0); `quant_autodiff` is a proper custom op and would work. |
| **burn-sct** | not reached | not reached | not reached | **None needed.** f32 CUDA QR / f32-f64 host LAPACK. The condition-number cost of the `AᵀA` Gram is documented, not worked around (it is inherent to Cholesky-QR). |
| **burn-spectral** | **HIT** in `ternarize_per_column` (`:122`) and `ternarize_stochastic` (`:104`) — bare `.float()`. The hot `ternarize` (`:54`) uses `.int().float()` and works in production. Inconsistent within one file. | **HIT.** `bf16_ops::bf16_matmul` is the workaround and `gemm_probe.rs:50` records it still fails at `burn-cubecl ops/tensor.rs:150`; the probe will not even time it. The *working* mitigation is the unconditional cast-to-F32 at the top of `forward`/`forward_quant` (`:462`, `:518`). | **HIT, unguarded.** `QuantFormat::Fp16` (`:561`) has no cfg, no runtime check, no test. Reachable via `--quant fp16`. | Mixed, and internally inconsistent. This crate is where "bringing to quality" costs the most: fix the two `.float()` calls, fix the Fp16 arm, and either fix or delete `bf16_matmul` + the `bf16_compute` plumbing that routes into it. |
| **burn-rmsnorm** | not reached | not reached | not reached | **Correct workaround present**: f32-only kernel with an explicit `DType` bail-out (`fused.rs:88-92`) rather than reinterpreting a bf16/f16 buffer. Plus the `device.is_autodiff()` branch for `require_grad` (`lib.rs:20-36`) — the fix burn-spectral is missing. |
| **burn-rope** | not reached | **immune** — no matmul in the crate; elementwise rotation only | immune | **None needed, by construction.** `rope_autodiff` is a custom fwd+bwd op; the structural fix, not a cast. |
| **burn-swiglu** | not reached | **immune** — `swiglu_gate` is elementwise; the matmuls live in two plain `burn::nn::Linear`s | immune | f32-only kernel + dtype bail-out copied from burn-rmsnorm (`fused.rs:54-58`). |
| **act_quant.rs (ours)** | **AT RISK, untested.** `fp4_round` uses the two-step `.int().cast(FloatDType::F32)` (lines 92, 95, 98, 101, 115, 120) — the form proven working in production by `burn-spectral::ternarize` — so it is *probably* fine. But its only two tests are `NdArray` (`act_quant.rs:140,158`), so **the fp4 CUDA path has never been unit-tested.** | not reached | not reached | **None carried.** The discriminating test is one line: run `fp4_round_matches_e2m1_values` on `Device::cuda(0)` and assert the outputs are not all zero. If `Int→F32` after a Bool→Int is also broken, `--act-quant fp4` is a **silent no-op** — and AGENTS.md's record that "fp4 + group 128, 100 steps, 0 NaN, convergence == fp32" would then be a **false pass**, since a no-op quantizer converges exactly like fp32. |

### The universal blocker (all four fused-kernel crates)

`burn-dispatch-0.22.0-pre.4/src/tensor.rs:481-487` — `try_into_backend` for a concrete backend
rejects any tensor whose `autodiff != Disabled`. Every fused kernel in burn-rmsnorm,
burn-swiglu, burn-spectral/gpu and burn-bitnet/fwt_cuda (non-autodiff entry) reaches the
hardware through `try_into_primitive::<CubeBackend>()`, so under
`Autodiff<burn_cuda::Cuda, BalancedCheckpointing>` they all return `None` and the caller falls
back to tensor ops. This is not one of the three named bugs; it is a fourth, and it is the
reason the three named ones have been so expensive — a working bf16 custom op, if it existed,
still could not be called through the current idiom.

The fix is already written in this library, twice: `burn-rope::rope_cuda::rope_autodiff` and
`burn-bitnet::{fwt_cuda::quant_autodiff, sparse::weight_quant_masked_autodiff}` — real
`burn_autodiff::ops::Backward` impls that peel the wrapper and re-register the output node.
**None of the three is wired into the model.** Porting that pattern to `burn-rmsnorm` (in the
model's hot path, 3 sites) and to `burn-spectral`'s TSCT GEMM is the single highest-value item
in this slice.

---

## 9. Summary table

| crate | LOC (src) | arXiv | fused CUDA reachable in prod? | state | verdict |
|---|---|---|---|---|---|
| `burn-bitnet` | 1815 | 2402.17764, 2504.18415, 2504.12285, 2603.05168, 2407.09527 | no (but `quant_autodiff` would be) | builds; 17/17 ndarray tests pass | **WIRED** (via `burn-spectral::quant_factor` on 6 layers + 3 direct sites) |
| `burn-sct` | 2738 | 2604.00733 | yes — and it is generic over `B`, so it *would* work under autodiff | builds; 13/13 ndarray tests pass; 2 real GPU-vs-CPU-reference harnesses | **SUPERSEDED** (by `burn-spectral::polar_orthogonalize`) |
| `burn-spectral` | 6172 | 2604.00733, 2504.12285, 2504.18415, 2412.04787, 2603.05168, 2602.21545, 2202.09368 | no — `SpectralLinear::forward` never reads `self.fused`; `gpu.rs` is example-only; `moe_fused.rs` needs `SpectralMoE`, never constructed | **33/36 tests FAIL** (`set_require_grad` panic, non-autodiff only); test target unbuildable from clean (dev-dep `burn-muon-plus` 2× E0308); no reference harness on the function we run | **BROKEN** (production autodiff path unaffected; 6× wired) |
| `burn-rmsnorm` | 235 | 1910.07467 | no (§0) | builds; 2/2 ndarray tests pass; the `cuda` test is a tautology on ndarray and has never launched the kernel | **WIRED** (3 sites) |
| `burn-rope` | 975 | 2104.09864, 2309.00071 | **yes** — `rope_autodiff` is a real fwd+bwd custom op (unwired) | builds; 9/9 ndarray tests pass; best f64 host reference in the library | **IMPLEMENTED-UNUSED** |
| `burn-swiglu` | 187 | 2002.05202 | no (§0) | builds; 2/2 ndarray tests pass (shape-only); zero CUDA tests | **IMPLEMENTED-UNUSED** |

**Call-site counts in `crates/dormouse-*`:** `burn-spectral` 22 (6 model sites via `LinearLike`),
`burn-bitnet` 3 direct + 5 indirect via burn-spectral, `burn-rmsnorm` 3, `burn-sct` 0,
`burn-rope` 0, `burn-swiglu` 0.
