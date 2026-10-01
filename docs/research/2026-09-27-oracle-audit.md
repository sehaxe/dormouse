# Oracle audit: what the library's verification claims are actually worth

Date: 2026-09-27. Slice: headline-claim audit of `vendor/burn-fused/`.
Read-only. No code changed, no GPU run. Snapshot `23f2b1c` + working tree.

**The claim under audit.** ADR-0018 rule 1: *"the port is verified bit-for-bit (or to a
stated tolerance with the tolerance justified) against the ORIGINAL implementation
wherever the authors shipped code ... Where no reference code exists, the crate says so
in its doc comment instead of implying verification it does not have."*

**Verdict.** The rule is the right rule and the library does not currently meet it. The
number that replaces "bit-for-bit against the reference" is below. Five README/doc
claims are outright false and are listed with replacement wording in ADR-0020.

---

## 1. The aggregate — the honest headline

| question | answer |
|---|---|
| Crates under `vendor/burn-fused/crates/` | **28** (+ the `burn-fused` facade) |
| Crates using the phrase "bit-for-bit" / "bit-exact" | **3** — `burn-gdn2`, `burn-sct`, `burn-kda` |
| …of those, with an oracle derived from an **authors'** source | **0** |
| Crates that **assert** they match an authors' *implementation* | **6** — `burn-gdn2`, `burn-sct`, `burn-dspark`, `burn-engram`, `burn-fastblt`, `burn-ttt` |
| …of those 6, with a test that actually compares against that implementation | **0** |
| Crates with a real, available external oracle (the `REFERENCE` list) | **10** — `burn-kda`, `burn-gdn2`, `burn-engram`, `burn-dspark`, `burn-mhc`, `burn-bitnet`, `burn-sct`, `burn-jepa`, `burn-fastblt`, `burn-eggroll` |
| …of those 10, with a test wired to it | **0** |
| **Crates verifying any numeric output against an authors' own source code** | **0 of 28** |

### What "bit-for-bit" resolves to, per crate

- **`burn-gdn2`** — "bit-for-bit" against `tests/ref_data.bin`, emitted by
  `tools/gen_reference.rs`, which is (per its own header, `tools/gen_reference.rs:5-6`)
  *"a line-for-line port of `tests/gen_reference.py`, which is itself a pure-PyTorch
  transcription of the original authors' layer"*. The transcription additionally states
  the Triton `fused_recurrent` kernel was *"replaced by an equivalent per-token scan"*.
  **Kind (c), three hands deep**, and the recurrence — the part that matters — is the
  part we re-derived. The crate is honest about this: `tests/bit_exact.rs:8-10` says
  verbatim *"despite this file's name this is NOT a bit-for-bit comparison"*. **The
  README is the part that is not honest.**
- **`burn-sct`** — "bit-exact" against a test that has never run. **Kind (d).**
- **`burn-kda`** — `examples/bitforbit.rs` and `examples/bitforbit_cuda.rs` are named
  "bit-for-bit" and are **dump harnesses**: they print one seeded input's output. They
  compare against nothing. **Kind (d).**

**The honest sentence:** the library verifies fused kernels against our own tensor
fallback, and transcriptions of papers against our own expectations of those papers.
That is real work and it catches real drift. It is not "bit-for-bit against the
reference implementation", and one crate's README says exactly that while the crate's
own test file says the opposite.

---

## 2. Per-crate table

Oracle kinds: **(a)** authors' own code · **(b)** our transcription of a paper ·
**(c)** transcription of a transcription · **(d)** no reference comparison.
Verdicts: **STRONG** = (a), falsifiable, runs in CI · **WEAK** = a real comparison
against a non-authors' reference, or a self-consistency/invariant check presented as
verification · **UNVERIFIED** = kind (d), i.e. no reference at all.

| crate | mechanism | arXiv claimed (in-crate) | reference the tests compare against | file | tol | verdict |
|---|---|---|---|---|---|---|
| `burn-antihall` | per-neuron hallucination gates | 2512.01797 (+3) | none — shape, `gates_in_range`, `init_near_one` | `src/lib.rs:56-155` | n/a | **UNVERIFIED** (d) |
| `burn-attnres` | softmax over previous layer outputs | 2603.15031 | our own fused vs tensor path; 1 finite-difference | `src/fused_attnres.rs:854-1530` | 1e-4 | **UNVERIFIED** (d) — also self-recorded BROKEN |
| `burn-bitnet` | b1.58 / N:M ternary quantizers | 2402.17764, 2504.18415, 2603.05168 | our own closed form — **a verbatim copy of the code under test** | `src/quant/weights.rs:105` | 1e-6 | **UNVERIFIED** (b, tautological) |
| `burn-byteflow` | adaptive byte compression | 2603.03583 | our transcription of the rate closed form | `src/chunk.rs:168-277` | 1e-4 | **WEAK** (b) |
| `burn-diffusionblocks` | block-wise diffusion schedule | 2506.14202 | our transcription of the EDM weighting | `src/noise.rs:268-300` | 1e-6 | **WEAK** (b) |
| `burn-dspark` | DSpark draft head + loss | 2607.05147 | **nothing.** Doc claims "matched against the official DeepSpec implementation" | — | n/a | **UNVERIFIED** (d) + **false claim** |
| `burn-eggroll` | rank-1 ES perturbations | 2511.16652 | distributional property (entry std ≈ σ) | `src/lib.rs:52-100` | 1e-6 | **WEAK** (b) |
| `burn-engram` | n-gram hash memory | 2601.07372 | our own restatement of the mix (and the doc admits **different constants** than the reference) | `src/hasher.rs:219` | exact | **UNVERIFIED** (b) — cannot match the reference by construction |
| `burn-es` | OpenAI ES | 1703.03864 | hardcoded paper values | `src/lib.rs:280-310` | 1e-4 | **WEAK** (b) |
| `burn-fastblt` | entropy patching / n-gram hash | 2412.09871, 2605.08044 | our own hand-computed arithmetic | `src/hash.rs:146` | exact | **UNVERIFIED** (b) — "Faithful to Meta's `patcher.py`" is untested |
| `burn-gdn2` | Gated DeltaNet-2 | **2605.22791** (README:10) | **our PyTorch transcription of `NVlabs` `lit_gpt/gdn2.py`**, Triton recurrence replaced by our scan | `tools/gen_reference.rs` → `tests/ref_data.bin` | 5e-4 abs on a ~1e-2 signal | **WEAK** (c) — the only real harness, wrong provenance |
| `burn-jepa` | EMA-teacher JEPA + KoLeo | 2212.07525, 2511.08544, 2304.07193 | none — `*_finite()`, hand-checked means | `src/lib.rs:86-255` | 1e-4 | **UNVERIFIED** (d) |
| `burn-kda` | Kimi Delta Attention | 2510.26692, 2607.24653 | **our own chunked-vs-recurrent self-consistency** | `src/lib.rs:862-1039` | 1e-3..1e-5 | **UNVERIFIED** (d) — "bitforbit" examples compare against nothing |
| `burn-mhc` | manifold hyper-connections + Sinkhorn | 2512.24880 | **Birkhoff invariants** (row/col sums = 1, ‖H‖ ≤ 1) — a property of the method, not a comparison | `src/lib.rs:50-130` | 1e-3 | **UNVERIFIED** (d) — best property check, zero oracle |
| `burn-mod` | Mixture-of-Depths routing | 2404.02258 | hardcoded expected indices | `src/lib.rs:62-130` | 1e-4 | **WEAK** (b) |
| `burn-mor` | MoR expert-choice routing | 2507.10524 | our own gather-vs-scatter equivalence | `src/lib.rs:179` | 1e-5 | **WEAK** (b) |
| `burn-mtp` | multi-token prediction | 2404.19737 | our own CE vs manual gather | `src/lib.rs:166` | exact | **WEAK** (b) |
| `burn-muon-plus` | Muon+ optimizer | 2602.21545 | our own NS/polar numerics + row/col-norm invariants | `tests/self_checks.rs`, `src/fused_kernels.rs:217` | 1e-3 | **WEAK** (b) — **but see ADR-0020: this is the one legitimate (b)**, the paper ships the rule as pseudocode |
| `burn-nope` | no positional encoding | 2607.24653 | none — 1 causal-row check | `src/lib.rs:99` | 1e-4 | **UNVERIFIED** (d) |
| `burn-parcae` | spectral retention | 2604.12946 | our own stability property | `src/lib.rs` | — | **UNVERIFIED** (d) |
| `burn-ptrn` | recursive-model test-time scaling | 2605.19943 | statistical property (RMSE ≈ 1) | `src/lib.rs:72` | **0.15** | **UNVERIFIED** (d) — tolerance wider than most of the effect |
| `burn-rmsnorm` | RMS normalization | 1910.07467 | **a hand-copied restatement of the same four lines of tensor math, on ndarray** | `src/lib.rs:105-119` | 1e-4 | **UNVERIFIED** (d) — true tautology, fused branch never taken |
| `burn-rope` | RoPE + YaRN | 2104.09864, 2309.00071 | our own f64 transcription of the ramp, compared **in the cos domain** | `src/lib.rs:112-165` | 1e-5 | **WEAK** (b) — **the best-falsified test in the library**; the inverted-ramp error is named as ~2.6 rad |
| `burn-sct` | spectral compact training | 2604.00733 | a **never-executed** harness against a gitignored, absent fixture and an absent generator | `tests/cmp_reference.rs:49` | 1e-3 | **UNVERIFIED** (d) + **false claim** |
| `burn-situ` | SiTU-GLU | 2607.24653 | our own fused vs tensor, 5 of 6 silently skip | `src/fused_situ.rs:389-500` | 1e-5 | **UNVERIFIED** (d) |
| `burn-spectral` | ternary-SCT | 7 papers incl. 2604.00733 | our own hand-written host re-implementations of the same math | `src/lib.rs:1473,1505,1878` | 1e-4 | **WEAK** (b), currently **RED** — 3 tests panic before asserting |
| `burn-swiglu` | SwiGLU | 2002.05202 | none — 2 shape tests; fused path never executed | `src/lib.rs:101-110` | n/a | **UNVERIFIED** (d) |
| `burn-ttt` | test-time training losses | 2407.04620, 2512.23675 | **nothing.** Doc claims "Matched to the official implementation" | — | n/a | **UNVERIFIED** (d) + **false claim** |

**Totals: 0 STRONG · 14 WEAK · 14 UNVERIFIED. 0 against an authors' source.**

---

## 3. Tests that could not fail

### 3a. The reference is the code under test (2)

| test | file:line | what it misses |
|---|---|---|
| `fused_matches_tensor` | `burn-rmsnorm/src/lib.rs:105-119` | **Everything.** `dev()` at `:100-102` returns `Device::ndarray()`, so the fused branch at `:46-54` is never taken, and the "reference" at `:110-117` is a verbatim copy of `forward` at `:57-63`. On ndarray `RMSNorm::forward` *is* the tensor path, so the two sides are the same computation. Cannot fail for any reason. The CI ran it as this crate's GPU gate. |
| `b158_roundclip_matches_paper_closed_form` | `burn-bitnet/src/quant/weights.rs:105-131` | Any port error. The expected value at `:118` — `(x / gamma).round().clamp(-1.0, 1.0) * gamma` — is the implementation at `:19-33` character for character. A 1e-6 tolerance on a copy of the code measures nothing. |

### 3b. A test whose reference is our own transcription (2 more, worse in a way)

| test | file:line | what it misses |
|---|---|---|
| `matches_reference_algorithm` | `burn-engram/src/hasher.rs:219-245` | Divergence from `engram_demo_v1.py`. The assert restates the implementation's own expression, and the module header at `:12-13` admits the multipliers are *"a deterministic splitmix64 stream instead of numpy's PCG64. Same structure, different constants"* — so this port **cannot** be bit-for-bit with the reference, and the test is structurally unable to notice. |
| `hash_matches_reference_formula` | `burn-fastblt/src/hash.rs:146-151` | Same shape, and the crate separately claims "Faithful to Meta's `bytelatent/data/patcher.py`" (`src/lib.rs:18`) with no test behind it. |

### 3c. Silently reporting PASS having asserted nothing (6)

`BURN_DEVICE` guards. Unset the variable and these return and are counted as passed.

| test | file:line | what it misses |
|---|---|---|
| `rope_matches_ref` | `burn-rope/src/rope_cuda.rs:242-251` | every RoPE CUDA kernel bug |
| `unaligned_h_falls_back`, `situ_fused_matches_tensor`, `fused_backward_matches_tensor_backward`, `fused_forward_autodiff_matches`, `situ_grad_matches_finite_difference` | `burn-situ/src/fused_situ.rs:372, 391, 430, 457, 491` (guard fn `:316`) | every SiTU CUDA kernel bug |
| `polar_retracts`, `to_inference_matches_trained_layer`, `to_inference_matches_per_column_layer` | `burn-spectral/src/lib.rs:1258, 2312, 2364` | `Tensor::set_require_grad(true)` is an `assert!` on a non-autodiff device; all three panic inside `retract` before reaching a single assertion. Known-red in CI by design — but a test that panics asserts nothing. |

### 3d. Feature-gated, ignored, or missing its fixture (2 classes)

| test | file:line | what it misses |
|---|---|---|
| `cmp_vs_reference` | `burn-sct/tests/cmp_reference.rs:49` | The entire bit-exactness claim. `#[cfg(feature = "binary-tests")]`, and `binary-tests = []` is **not** in `default = ["std"]` (`Cargo.toml:15`). Fixture `tests/ref_data/` absent; generator `gen_reference.py`, named at `README.md:74`, absent from the crate. Unrunnable by three independent routes. |
| 4 `alloc_probe` tests | `burn-gdn2/tests/alloc_probe.rs:238, 265, 290, 316` | The balanced-checkpointing allocation behaviour — this was the only place `Autodiff<CudaBare, BalancedCheckpointing>` appeared. All `#[ignore]`d. **Partially superseded**: a new untracked `tests/autodiff_cuda_gate.rs:96` now exercises that backend properly, which is real progress from another agent, but it also needs a GPU. |

### 3e. Backend coverage — the reason 3c matters so much

No test runs `Autodiff<Cuda, BalancedCheckpointing>` in a job that executes. The CI
GPU job is `runs-on: [self-hosted, gpu, linux]` and its own comment records that **no
runner is registered**, so it has never run; the `cuda-compiles` job does
`cargo test --no-run`, which cannot fail on a CUDA-only bug. Every
`Device::default()`-based CUDA test therefore has never executed in CI:
**9 in `burn-attnres`, 5 in `burn-mhc`, 4 each in `burn-rope` and `burn-kda`,
6 in `burn-situ`, 3 in `burn-sct`, and the fused paths of `burn-gdn2`,
`burn-spectral` and `burn-bitnet`.**

`burn-bitnet` is the sharpest case: `src/fwt_cuda.rs` contains exactly one test,
`bitnet_bench` at `:535`, and it is `#[ignore]`d and is a benchmark. **The fused CUDA
path in `burn-bitnet` has zero correctness coverage.**

### Headline count

Non-overlapping, and every member verified by reading the file in this pass:

> **28 tests cannot fail, pass without having asserted anything, or are unreachable by
> any CI job that has ever executed.**

| n | class | members |
|---|---|---|
| 2 | cannot fail at all — the reference is the code, on a device where the branch is not taken | `burn-rmsnorm/src/lib.rs:105`, `burn-bitnet/src/quant/weights.rs:105` |
| 2 | the expected value is a hand-copy of the code, so it moves with the bug and cannot detect a port error | `burn-engram/src/hasher.rs:219`, `burn-fastblt/src/hash.rs:146` |
| 9 | silently skip, or panic before reaching an assertion | `burn-rope/src/rope_cuda.rs:242` ×1, `burn-situ/src/fused_situ.rs` ×5, `burn-spectral/src/lib.rs:1258, 2312, 2364` ×3 |
| 1 | permanently unrunnable — non-default feature, absent fixture, absent generator | `burn-sct/tests/cmp_reference.rs:49` |
| 14 | CUDA-executing, on `Device::default()` / `Device::cuda(0)`, in a job whose runner does not exist | attnres ×3, mhc ×1, gdn2 ×2, sct ×1, muon-plus ×1, bitnet ×1, kda ×4, mor ×1 |

The last row is a floor, not a total. `TEST-AUDIT.md` finding 3 tabulates **~48**
CUDA-executing tests across 14 crates on that unreachable path; I did not re-count
them test by test, so the 48 is that report's figure, not mine. The 14 above are the
ones I opened and confirmed myself. The 4 `#[ignore]`d `alloc_probe` tests
(`burn-gdn2/tests/alloc_probe.rs:238, 265, 290, 316`) are excluded from the count —
they are correctly-ignored probes, not a false gate.

### 3f. The inherited fiction

Ten per-crate workflows exist (`crates/*/.github/workflows/ci.yml` for attnres,
bitnet, gdn2, kda, mhc, rope, sct, situ, plus `bench.yml`). `vendor/burn-fused` has
no `.git` and is `exclude`d from the root workspace, so **none has ever executed**.
They cannot gate anything and their presence invites the belief that they do. The real
gate is `../../.github/workflows/fused-library.yml`, which is honest about its own
limits — except that its GPU job has no runner.

---

## 4. Cost to raise WEAK → STRONG

Sources taken from `docs/research/2026-09-27-adopt-vs-port.md` §4 (already opened during
that pass) rather than re-derived. Two re-verified by me here via the GitHub trees API.

| rank | crate | cost | the external source that would be the oracle |
|---|---|---|---|
| **1** | `burn-gdn2` | **lowest** — harness, committed fixture, regenerable generator and CI diff all exist; only the reference body changes | `NVlabs/GatedDeltaNet-2` → `lit_gpt/gdn2_ops/fused_recurrent_gdn2.py` (replace the per-token scan in `tests/gen_reference.py`) |
| **2** | `burn-sct` | low — write `gen_reference.py` + commit 4 fixtures + add the job to the workflow that exists. **2,738 LOC currently claim a bit-exactness that never ran** | `EctoSpace/SCT` → `spectral_compact_training/spectral_layer.py` — **re-verified live, 3,947 bytes** |
| **3** | `burn-dspark` | low-medium — ~150 lines Python + 1 test. The loss has five weighted terms; a `gamma` error or L1-on-logits-vs-probs is invisible to all 7 current tests | `deepseek-ai/DeepSpec` → `deepspec/modeling/dspark/{loss.py, markov_head.py, common.py}` |
| 4 | `burn-mhc` | low — from *nothing* to official; `ref_mhc.py` runs on CPU | `deepseek-ai/DeepGEMM` → `third-party/tilelang_ops/ref_mhc.py` + `csrc/apis/hyperconnection.hpp` |
| 5 | `burn-kda` | highest fidelity, **highest cost**: ~120 lines Python, but the oracle is **forward-only** so the backward — the part it is wired for — stays un-orphaned | `MoonshotAI/FlashKDA` → `tests/torch_ref.py`, a bit-matching reference — **re-verified live, 10,003 bytes** |
| 6 | `burn-bitnet` | low — the quantizer bodies are 6 lines; the fidelity is in the Hadamard ordering and the N:M convention | `microsoft/BitNet` → `gpu/model.py`; `AAzdi/Sparse-BitNet` → `llm/arch/model.py` |
| 7 | `burn-engram` | medium — needs a tokenizer fixture | `deepseek-ai/Engram` → `engram_demo_v1.py` |
| 8 | `burn-fastblt` | low cost, low value — Engram supersedes it and dormouse uses neither hasher | `facebookresearch/blt` → `bytelatent/data/patcher.py` |
| 9 | `burn-eggroll` | near-zero, and it settles a **live disagreement between two of our crates**: `burn-eggroll` uses `(σ/√r)·A·Bᵀ`, `burn-es` documents `σ·A·Bᵀ` and defers. One is wrong. | `ESHyperscale/HyperscaleES` |
| 10 | `burn-jepa` | lowest, low value — KoLeo is a dozen lines. **The data2vec-2.0 leg has no confirmed oracle; I did not find one and do not assert one.** | `facebookresearch/dinov2` → `koleo_loss` |

### Corrections to the prior research

- **`burn-gdn2` already cites arXiv 2605.22791** — at `burn-gdn2/README.md:10`. The
  prior report says twice (§0 item 2 and §4 #4) that the crate "has no arXiv in the
  crate" and "add the missing arXiv 2605.22791 to the crate doc". The `src/lib.rs` doc
  comment may still lack it, but the crate as a whole cites it. Minor; the recommendation
  is half-stale.
- **`MoonshotAI/FlashKDA` default branch is not `main`.** `raw.githubusercontent.com/.../main/tests/torch_ref.py`
  404s. The trees API confirms the file exists (10,003 bytes). Anyone re-deriving this
  will hit the same 404 and should not conclude the repo is gone.
- Everything else in that report's §4 checked out against the source I read. The
  `EctoSpace` org-endpoint 404 and the `data2vec` 404s are recorded there and I did not
  retry them.

---

## 5. Files that are mid-edit, and what I did not do

- `burn-gdn2`, `burn-kda`, `burn-mor`, `burn-spectral` have modified sources;
  `burn-gdn2/src/cuda_dispatch.rs`, `burn-gdn2/tests/{autodiff_cuda_gate,lowp_bf16_cuda,zz_scratch_probe}.rs`,
  `burn-kda/tests/cuda_gate.rs`, `burn-mor/src/topk_gather.rs` and `vendor/burn-fused/tools/`
  are **untracked** — other agents are mid-work. I read them as they are and flag that
  their state is not final. In particular `autodiff_cuda_gate.rs` is a genuine
  improvement over what `TEST-AUDIT.md` recorded and I credited it as such.
- `burn-mor/src/topk_gather.rs` no longer contains the `.int()` calls the prior report
  said made it uncompilable; that file is being rewritten. I did not build it.
- I ran no tests and no GPU work, so the library's actual pass/fail state is NOT
  VERIFIED by me. Every "BROKEN"/"RED" attribution is quoted from
  `TEST-AUDIT.md` or `docs/archive/research/2026-09-27-fused-inventory-attention.md`, not re-measured.
- I changed nothing in `vendor/` or `crates/`. Two files created: this report and
  `docs/adr/0020-oracle-discipline.md`.
