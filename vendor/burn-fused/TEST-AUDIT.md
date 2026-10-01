# TEST-AUDIT — what the library's tests actually assert

> **Status 2026-10-01.** Everything below was true as of the 2026-09-27 snapshot
> it names; the crate list changed 2026-09-28. One section is not merely stale
> but describes files that no longer exist: **"The bit-exact harness"** is about
> `crates/burn-gdn2/tests/ref_data.bin` and `tools/gen_reference.rs`, both
> DELETED. The fixture is now `tests/ref_f64_broad.bin` (1000 cases, f64
> NumPy, no torch) and the generator `tools/gen_reference_f64.py`; the f32
> self-transcription was removed because it could only prove self-consistency
> and it replicate-padded the short conv's left edge exactly as the kernel
> wrongly did. The CI job that checks it is `fused-library.yml`'s
> `ref-data-reproducible`. Read that section as history.

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

### FINDING 0: SOLVED 2026-09-29 — it was the reference generator, not burn

**This finding was wrong in its conclusion for three days.** It correctly
ruled out the tolerance, the fixture format and per-token arithmetic, then
localized the fault to "the short conv's cross-token taps, and the state
carry-over" and named burn's `fused_recurrent_forward` per-token
`slice_dim(2, t..t+1)` over permuted `[B, HV, T, D]` views as the likely
culprit. The fault was in `tools/gen_reference.rs` and had nothing to do with
either. `binary-tests` is now back in burn-gdn2's default features, and the CI
job (`fused-lib :: 1000 bit-exact cases vs the paper reference`) still gates
the merge.

Measured 2026-09-27, ndarray, against the then-committed fixture:

```
1000 cases: max_diff = 1.38e-2,  failures = 976/1000      (EPSILON = 5e-4)
  FAIL [1] shape=1x3x64  max_diff=2.88e-3
  FAIL [3] shape=1x11x64 max_diff=6.79e-3
  FAIL [5] shape=1x37x64 max_diff=1.17e-2
```

Measured 2026-09-29, ndarray, same tolerance, after one fix to the generator:

```
1000 cases: max_diff = 2.32e-7,  failures = 0/1000        (EPSILON = 5e-4)
```

**The defect.** `linear` fills a token-major `[T, KD]` buffer, and the code
said so at the one place it mattered (`let c = n % KD; // channel index within
[T, KD]`). But the scan read its inputs through

```rust
let at = |a: &[f32], h, ti, i, n| a[h * t * n + ti * n + i];   // head-major
```

against those same token-major buffers. The GVA expander `expand` made it
worse in two ways: its `rep == 1` early return handed back the token-major
input unchanged, and its non-trivial branch read `src[(from_h * t + ti) * n +
i]`, a head-major read of a token-major buffer. `v` and `w_gate` never went
through `expand` at all and were read head-major at their own width `VD`. So
**no** scan input was ever in the layout the scan indexed.

**Why 24/1000, exactly.** At `T == 1` the head-major and token-major
indexings coincide on every element (`h*1*n + 0*n + i == h*n + i`), so a
single-token case cannot see the bug. `gen_reference.rs` picks
`seq_len = (1 << (i % 6)) + (i % 7)`, and that is 1 exactly when `i % 42 == 0`
— 24 cases out of 1000. The 24 passes were the 24 single-token cases. The
`gen_reference.py` original does a real `q.reshape(B,T,H,HK).transpose(1,2)`,
which torch makes a free view; the Rust port reimplemented head-major layout as
flat offsets but never built the head-major buffer.

**The evidence that needed no GPU.** Regenerating with `to_head_major` applied
to q/k/g/b at width HK and to v/w_gate at width V_HEAD changes exactly 976 of
the 1000 fixture outputs and leaves the 24 single-token ones bit-identical —
independently of burn. The peak difference between the old and new fixture
outputs is 1.383e-2, which is the `1.38e-2` burn had been reporting: the
generator was the sole source of the divergence, and the tolerance, the fixture
format, burn's per-token slicing and the short conv were all innocent. The test
itself is the corroboration, and it needed no CUDA: `max_diff = 2.32e-7`,
0/1000 failures, `EPSILON` unchanged at 5e-4.

**What the three original "this is not" bullets got right, and why they did
not save it.** The tolerance really was not the problem, the fixture really was
well-formed and deterministic, and the projections really did agree to f32
noise at `t=0`. All three are measurements *at T=1*, where the bug is
invisible by construction. The audit reasoned "T=1 cannot reach cross-token
taps and state carry-over" and looked for a cross-token bug; it never asked
whether the generator's own cross-token layout was consistent, which was the
one thing that could be checked by reading 40 lines of it.

**Residual, and what it is.** `2.32e-7` is f32 reduction-order noise plus one
genuine transcription difference, named because it is the one left: burn's
`l2_normalize_4d` divides by `sqrt(ss + 1e-6)` (`module.rs` calls it with
`1e-6`), the generator by `sqrt(max(ss, 1e-12))` — an epsilon inside the root
against a clamp before it. At ~1e-6 relative on a head's 16 channels that is
the same order as the observed residual, so this measurement does not separate
the two. Fixing it is not required for green and was not done.

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
- **The tolerance is measured now, and it is not the problem** (FINDING 0, now
  closed): the gate runs at `max_diff = 2.32e-7` against `EPSILON = 5e-4`, so
  the threshold has ~2000x of headroom over the observed noise and the measured
  noise is f32 reduction order plus the one `l2_normalize_4d` eps difference
  named in FINDING 0. What this gate now wants is a *relative* or
  RMS-normalised tolerance (the fixture's outputs reach ~9e-3, so 5e-4 is ~5% of
  the signal) — the honest reason is no longer "an unexplained gap to tighten
  around", it is that an absolute threshold on a fixture with a known output
  scale is the weaker instrument.
- **`burn-gdn2`'s `python3 tests/gen_reference.py` path in `README.md:262-265`**
  still tells a reader to run the torch script. It is the readable reference and
  it stays, but the runnable one is now `tools/gen_reference.rs`.
