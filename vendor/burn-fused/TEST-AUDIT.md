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

`vendor/burn-fused` has no `.git` and the root `Cargo.toml` `exclude`s it, so
the fork's own `.github/workflows/ci.yml` has never executed for this copy. It
was also wrong: it tested `burn-msa`, deleted 2026-09-27 (ADR-0014). The gate
that runs is now **`../../.github/workflows/fused-library.yml`** (root of
dormouse), which builds the fork from inside `vendor/burn-fused` because the
exclude makes `cargo test -p burn-gdn2` from the root a guaranteed
"package not found".

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
sequential f32, no thread-count dependence). CI regenerates and `git diff
--exit-code`s it on every push, so the fixture cannot drift from its generator
unnoticed. `binary-tests` is in burn-gdn2's default features, so the suite runs
on every `cargo test -p burn-gdn2` rather than only when someone remembers the
flag. The full precision argument, and what the harness does **not** claim, is
in the module comment of `tests/bit_exact.rs`.

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
- **Ten per-crate workflows are the same fiction**
  (`crates/*/.github/workflows/ci.yml` for attnres, bitnet, gdn2, kda, mhc,
  rope, sct, situ + `bench.yml`). They cannot execute here. Delete them, or port
  them when the fork gets its own repository; keeping them invites the belief
  that the crate is gated.
- **`rust-toolchain.toml` says `channel = "stable"`, which floats.** For a
  library whose headline claim is bit-level reproducibility, the toolchain is
  part of the claim. Pin it (1 line) and say which version.
- **The 5e-4 absolute tolerance is ~5% of the fixture's output scale** (outputs
  reach ~9e-3). A relative or RMS-normalised tolerance would be the stronger
  gate, but picking one needs a measured noise floor from a second independent
  transcription, not a guess.
- **`burn-gdn2`'s `python3 tests/gen_reference.py` path in `README.md:262-265`**
  still tells a reader to run the torch script. It is the readable reference and
  it stays, but the runnable one is now `tools/gen_reference.rs`.
