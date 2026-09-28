# TEST-AUDIT — what the library's tests actually assert

Written 2026-09-27, snapshot `1fab19e`. Scope: the CI gate and the
reproducibility of the bit-for-bit claim, for the 28 crates under `crates/`
as they stood at that snapshot. Eight of them (`burn-antihall`, `burn-byteflow`,
`burn-diffusionblocks`, `burn-fastblt`, `burn-mod`, `burn-mtp`, `burn-nope`,
`burn-ttt`) were deleted 2026-09-28 as unreachable; see
`docs/library-crate-fate.md`.
Nothing in a `src/` was changed here — the crates listed under FINDINGS belong
to other agents, this file is the hand-off.

## The gate

`vendor/burn-fused` has no `.git` and the root `Cargo.toml` `exclude`s it, so no
`.github/` under it can ever execute for this copy. The fork-level workflow
that was there also tested `burn-msa`, deleted 2026-09-27 (ADR-0014); it and
the ten per-crate ones have since been deleted (`4963c3a`), which is the right
call — dead config invites the belief that a crate is gated. The gate that runs
is **`../../.github/workflows/fused-library.yml`** (root of dormouse), which
builds the fork from inside `vendor/burn-fused` because the exclude makes
`cargo test -p burn-gdn2` from the root a guaranteed "package not found".

Why the root job and not a separate repository for the fork: the fork's crates
are consumed by `path =` from `dormouse-core`, and 7 of them are load-bearing
for training. A CI that runs only in a fork's own repository can go green while
the integration that actually ships is broken — the fork-only gate would test
the wrong thing. The root job is the one that can fail on the code that trains.
What a fork repo would still buy: upstream-flavoured PRs and releases for the
library alone. That is a migration for the owner to schedule (move the
directory, repoint the 7 `path =` deps, keep the `[patch]` map), not something
to do inside a test-and-CI change.

## The bit-exact harness

`crates/burn-gdn2/tests/ref_data.bin` is now committed (6.8 MiB, 1000 cases)
and regenerable byte-for-byte by `crates/burn-gdn2/tools/gen_reference.rs`
(std-only, `rustc tools/gen_reference.rs`; splitmix64 + Box-Muller seeded 1337,
sequential f32, no thread-count dependence, so the bytes do not depend on the
platform, the core count or the rustc version - two runs `cmp` clean). CI
regenerates it and `git diff --exit-code`s it on every push, so the fixture
cannot drift from its generator unnoticed. The full precision argument, and
what the harness does **not** claim, is in the module comment of
`tests/bit_exact.rs`.

### FINDING 0: the harness is RED, and the reference is the *less* likely culprit

`binary-tests` is **not** in burn-gdn2's default features, on purpose: making it
default would put a red suite in every `cargo test -p burn-gdn2` whose cause is
not yet named, and a red default that nobody can explain gets deleted rather
than fixed. The gate is explicit instead, and it is loud - the CI job is named
`fused-lib :: 1000 bit-exact cases vs the paper reference (fused/tensor drift)`.

Measured 2026-09-27, ndarray, against the committed fixture:

```
1000 cases: max_diff = 1.38e-2,  failures = 976/1000      (EPSILON = 5e-4)
  FAIL [1] shape=1x3x64  max_diff=2.88e-3
  FAIL [3] shape=1x11x64 max_diff=6.79e-3
  FAIL [5] shape=1x37x64 max_diff=1.17e-2
```

Three things this is not:

- **Not the tolerance.** 5e-4 is 28x below the observed max_diff, and the
  measured transcription noise between two f32 implementations of this
  recurrence is 2e-6 to 2e-5 (below). Loosening `EPSILON` to make this green
  would be deleting the test's only assertion.
- **Not the fixture's format or its determinism.** A parser that mimics
  `bit_exact.rs`'s reader exactly consumes all 7 091 978 bytes with zero
  trailing bytes, finds all 17 tensors under their expected names and shapes,
  reads 1000 cases, and every value is finite.
- **Not a per-token difference.** On a *passing* single-token case, burn's
  `project()` output and the generator agree to ~2e-6 relative (q, k, v, g, b,
  w) and the scan output at t=0 agrees to ~2e-5. So the projections, the SiLU,
  the short conv at t=0, the L2 normalize, the decay, the erase, the per-head
  RMS-norm, the SiLU gate and `o_proj` are all in agreement to f32 noise.

What is left is exactly what T=1 cannot reach: the short conv's cross-token
taps, and the state carry-over. In burn the carry-over is
`kernel::fused_recurrent::fused_recurrent_forward`, which slices the *permuted*
`[B, HV, T, D]` views from `project()` one token at a time
(`slice_dim(2, t..t+1)`). A stride/offset error in that slice is invisible at
t=0 (offset 0) and wrong for every t >= 1, which is the observed signature; a
conv tap-indexing error in the generator has the same signature. The one
measurement that separates them, still to be run: print q/k/v at t=1 for case 1
(T=3) on both sides. It was not run here because the machine hit 100% disk
mid-build and the test binary would not link - so this finding is stated as
localized, not as diagnosed.

## FINDINGS — false confidence, for the crate owners

### 1. `burn-rmsnorm`: the "GPU" test runs no GPU code, and is a tautology

`crates/burn-rmsnorm/src/lib.rs:95` — `#[cfg(all(test, feature = "cuda"))] mod
cuda_tests`, whose `dev()` at `:100-102` returns `Device::ndarray()`. So the
module only *compiles* with `--features cuda`; every test in it runs on ndarray,
i.e. it exercises the tensor path. Worse, `fused_matches_tensor` (`:104-120`)
compares `RMSNorm::forward` (which, on ndarray, *is* the tensor path) against a
hand-copied restatement of the same four lines of tensor math (`:110-117`). It
cannot fail for any reason related to the fused CUDA kernel, and the old CI ran
it as the GPU gate for this crate. The real check needs `Device::cuda(0)` and a
reference that is not a copy of the code under test.

### 2. `burn-spectral`: three tests panic before they assert

`crates/burn-spectral/src/lib.rs:1227-1228` — the `mod tests` `dev()` is plain
`Device::ndarray()`, and both `SpectralLinear::retract` (`:586`, calling
`set_require_grad(true)` at `:595` and `:600`) and `SpectralMoE::retract`
(`:1208`, at `:1212` and `:1217`) re-track the masters that way. On this burn
version `Tensor::set_require_grad(true)` on a tensor without autodiff is an
`assert!`, not a no-op: `burn-tensor-0.22.0-pre.4/src/tensor/api/float.rs:1176`
("Tensor::require_grad requires autodiff; call Tensor::autodiff first"). Three
call sites construct on the non-autodiff device and call `retract`:

| test | line | `retract` at | device |
|---|---|---|---|
| `polar_retracts` | `:1258` | `:1264` | `dev()` = `Device::ndarray()` |
| `to_inference_matches_trained_layer` | `:2312` | `:2319` | `Device::ndarray()` (`:2313`) |
| `to_inference_matches_per_column_layer` | `:2364` | `:2371` | `Device::ndarray()` (`:2365`) |

(`retract_keeps_masters_tracked` at `:1274` is fine — it uses
`Device::ndarray().autodiff()`.) **KNOWN RED**: the new `ndarray-all-crates` job
runs `cargo test --workspace`, so it will be red on its first run for exactly
this reason. That is the point of writing it down rather than deleting the
tests; the fix belongs to whoever owns burn-spectral (either the test devices or
a `set_require_grad` that tolerates a non-autodiff backend).

### 3. No test anywhere runs `Autodiff<Cuda, BalancedCheckpointing>`

That is the configuration dormouse trains in, and it is the one with the
checkpointing-aware fused backward (`burn-gdn2/src/autodiff.rs:360-390` exists
precisely because the default `Autodiff` graph cannot accept the balanced
checkpoint tensors, `:370`). The only place the type appears is
`crates/burn-gdn2/tests/alloc_probe.rs:29` (`type AdBal = Autodiff<CudaBare,
BalancedCheckpointing>`), in a file where **all four tests are `#[ignore]`d** —
so it is a probe nobody runs. Every other CUDA test instantiates a bare or
default-checkpointing backend:

| crate | CUDA-executing tests | backend |
|---|---|---|
| burn-gdn2 | `tests/fused_chunk_verify.rs:28`, `tests/lowp_bf16_cuda.rs:43` | `type B = CudaBare` (bare) |
| burn-kda | `tests/fused_cuda.rs:21,41,68` / `:117` | `CudaBare`, `Autodiff<CudaBare>` (NoCheckpointing) |
| burn-sct | `tests/cuda_retract.rs:18` | `Device::cuda(0)`, `retract::<CudaBare>()` |
| burn-spectral | `src/lib.rs:1810` (1), `src/moe_fused.rs:1623` module (13) | `Device::cuda(0)`; `Device::cuda(0).autodiff()` at `:1670`, `:1732` |
| burn-spectral | `src/bf16_ops.rs:90` module (2) | `type Bare = burn_cubecl::CubeBackend` (`:98`), `type AD = Autodiff<Bare>` (`:95`) — default checkpointing |
| burn-attnres | `src/fused_attnres.rs:806` module (9) | `Device::default()` (`:855`, `:878`, `:902`) |
| burn-mhc | `src/sinkhorn_cuda.rs:218` module (5) | `Device::default()` (`:225`, `:254`) |
| burn-muon-plus | `src/fused_kernels.rs:207` module (4), `tests/bf16_matmul.rs:6` (5) | `Device::default()` (`:212`), `Device::default().autodiff()` (`:12`) |
| burn-bitnet | `src/fwt_cuda.rs:528` module (1) | `Device::default()` — and that one test is `bitnet_bench` (`:535`), a benchmark, not a correctness check |
| burn-rope | `src/rope_cuda.rs:218` module (4) | bare, and `rope_matches_ref` (`:242`) **silently returns unless `BURN_DEVICE=cuda`** (`:246`) |
| burn-situ | `src/fused_situ.rs:312` module (6) | **silently skips unless `BURN_DEVICE=cuda`** (`cuda_enabled`, `:316`; 5 skip sites `:372,391,430,457,491`) |
| burn-mor | `src/topk_gather.rs:235` | `Device::cuda(0)` via the `devices()` list (`:223-229`); the same test also runs on ndarray |
| burn-swiglu | **none** | `src/fused.rs:39 swiglu_cuda` is only called from the dispatch at `src/lib.rs:24` — the fused CUDA path is never executed by a test |
| burn-rmsnorm | `src/lib.rs:104` (1) | `Device::ndarray()` — see FINDING 1 |

Two things follow. `Device::default()` (attnres, mhc, muon-plus, bitnet) is the
burn 0.22 runtime's default dispatch device
(`burn-tensor-0.22.0-pre.4/src/device.rs:123-127` → `DispatchDevice::default()`),
so those tests' backend is a side effect of feature resolution, not a stated
intent — a test that cannot say which device it is on cannot be trusted to be a
GPU test. And the `BURN_DEVICE` guards are silent: unset the variable and six
tests across two crates report PASS having asserted nothing. The CI jobs that
run them set `BURN_DEVICE: cuda`; the crates should fail loudly instead.

## Follow-ups, not done here

- **`burn-sct` has the same disease, worse.** `tests/cmp_reference.rs:52` reads
  `tests/ref_data/{tiny,small,med,large}.bin`, that directory is gone, and
  `gen_reference.py` — referenced by `README.md:77` — does not exist in the
  crate at all. `binary-tests` there has never been runnable. Needs a generator
  port like burn-gdn2's, and the fixture committed.
- ~~Ten per-crate workflows are the same fiction.~~ **Done**: deleted in
  `4963c3a` along with the fork-level one, with the GPU job replaced by a
  local command (a polling runner on this box would sit next to the trainer and
  be OOM-killed by ram-guard). Keep it that way: the root workflow is the only
  CI these crates have.
- **`rust-toolchain.toml` says `channel = "stable"`, which floats.** For a
  library whose headline claim is bit-level reproducibility, the toolchain is
  part of the claim. Pin it (1 line) and say which version.
- **The tolerance is measured now, and it is not the problem** (FINDING 0): the
  noise floor between two f32 implementations of this recurrence is 2e-6 to
  2e-5, so 5e-4 is the right order of magnitude. What the next version of this
  gate wants is a *relative* or RMS-normalised tolerance (the fixture's outputs
  reach ~9e-3, so 5e-4 is ~5% of the signal) — but only after FINDING 0 is
  closed, or it would be tightening a threshold around an unexplained gap.
- **`burn-gdn2`'s `python3 tests/gen_reference.py` path in `README.md:262-265`**
  still tells a reader to run the torch script. It is the readable reference and
  it stays, but the runnable one is now `tools/gen_reference.rs`.
