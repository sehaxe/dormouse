# burn-fused baseline quality matrix (snapshot 2026-09-27, 17:45–19:55)

Read-only measurement. **Nothing was fixed.** The only files added are this one
and the two probe helpers `tools/fused_matrix.sh`, `tools/fused_matrix_static.py`
(nothing committed).

Reproduce from inside `vendor/dormouse-fused/` — it is its own workspace root, the
outer workspace `exclude`s it on purpose (outer `Cargo.toml:8-10`):
```
tools/fused_matrix.sh <check|test|cuda-check|cuda-test|examples> [crate...]
python3 tools/fused_matrix_static.py
```

## Scope correction: 32 members, not 26

`vendor/dormouse-fused/Cargo.toml` `members` has **32** entries: **28 leaf crates**
under `crates/`, the `burn-fused` meta-crate, and 3 bench/probe packages
(`benches`→`burn-fused-benches`, `benches/cpu_probe`→`cpu-probe`,
`benches/launch_probe`→`launch-probe`). All 32 are covered below; the headline
counts say "of 28 leaf crates" where meta/bench packages are excluded, because
"builds / has tests" is a question about real code.

## Conditions — read before trusting any row

* **The tree was being rewritten by other agents the whole time.** 7+ foreign
  `cargo` processes shared this `target/` dir; `burn-gdn2`, `burn-kda`,
  `burn-engram`, `burn-mor`, `burn-spectral`, `burn-muon-plus`, `burn-rmsnorm`
  and the `burn-fused` meta-crate all changed under me. **Timings in this
  document are lock-contention wall clock, not build cost — ignore every `Ns`.**
* **The workspace went unresolvable for the last ~30 min of the session** (see
  "The five defects"). Rows measured after 19:45 could not run at all.
* A detached training run held 5.7 GB VRAM mid-session; CUDA work was done only
  in free windows. Never started a training run.
* `cargo test` builds `examples/` too, so a non-compiling example fails the
  *test* command — that is why `burn-kda`'s ndarray test cell is an error in
  `examples/`, not in a test.
* `burn-gdn2` and `burn-kda` hard-enable the CUDA backend in `[dev-dependencies]`
  (`burn = {features=["ndarray","autodiff","cuda",..]}`; `burn-fused-benches`
  likewise). Their "ndarray" test runs are **not** ndarray-only — they initialise
  the driver. The 499 s for `cargo test -p burn-gdn2` is CUDA stack compilation.

## The matrix (28 leaf crates)

1 ndarray `cargo check` | 2 `cargo check --features cuda` |
3 `cargo test` ndarray (+count) | 4 `cargo test --features cuda` (+count) |
5 `--examples --features cuda` | 6 declares `cuda` | 7 loc / testloc / % / markers

| crate | 1 | 2 | 3 | 4 | 5 | 6 | loc | testloc | % | mk |
|---|---|---|---|---|---|---|---|---|---|---|
| burn-attnres | PASS | PASS | PASS 8 | **FAIL** 11+1f/2ig | PASS | yes | 2124 | 525 | 24% | 0 |
| burn-bitnet | PASS | PASS | PASS 17 | PASS 17 | PASS | yes | 1815 | 597 | 32% | 0 |
| burn-gdn2 * | PASS | **n-m** | PASS 13 | **n-m** | **n-m** | yes | 6972 | 2807 | 40% | 1 |
| burn-kda * | PASS | **n-m** | **FAIL** compile | **n-m** | **FAIL** compile | yes | 2110 | 619 | 29% | 0 |
| burn-mhc | PASS | PASS | PASS 7 | PASS 8 | PASS | yes | 1011 | 307 | 30% | 0 |
| burn-muon-plus | PASS | PASS | **FAIL** compile | **FAIL** compile | PASS | yes | 1298 | 446 | 34% | 0 |
| burn-rope | PASS | PASS | PASS 9 | PASS 10 | PASS | yes | 975 | 318 | 32% | 0 |
| burn-sct | PASS | PASS | PASS 13 | FLAKE→PASS 16 | PASS | yes | 3350 | 461 | 13% | 2 |
| burn-situ | PASS | PASS | PASS 4 | PASS 6 | PASS | yes | 675 | 216 | 32% | 0 |
| burn-ttt | PASS | N-A | PASS 4 | N-A | N-A | **no** | 125 | 66 | 52% | 0 |
| burn-spectral * | PASS | PASS | **FAIL** dep | **n-m** | **FAIL** 3/7 ex | yes | 8299 | 2368 | 28% | 2 |
| burn-rmsnorm | PASS | PASS | PASS 2 | PASS 3 | PASS | yes | 235 | 43 | 18% | 0 |
| burn-nope | PASS | N-A | PASS 3 | N-A | N-A | **no** | 102 | 59 | 57% | 0 |
| burn-swiglu | PASS | PASS | PASS 2 | PASS 2 | PASS | yes | 187 | 22 | 11% | 0 |
| burn-fastblt | PASS | N-A | PASS 8 | N-A | N-A | **no** | 513 | 136 | 26% | 0 |
| burn-mtp | PASS | N-A | PASS 4 | N-A | N-A | **no** | 196 | 73 | 37% | 0 |
| burn-antihall | PASS | N-A | PASS 10 | N-A | N-A | **no** | 302 | 128 | 42% | 0 |
| burn-mor * | PASS | N-A* | PASS 10 | **n-m** | **n-m** | was yes* | 814 | 363 | 44% | 0 |
| burn-dspark | PASS | N-A | PASS 14 | N-A | N-A | **no** | 865 | 253 | 29% | 0 |
| burn-diffusionblocks | PASS | N-A | PASS 11 | N-A | N-A | **no** | 793 | 255 | 32% | 0 |
| burn-parcae | PASS | N-A | PASS 8 | N-A | N-A | **no** | 317 | 132 | 41% | 0 |
| burn-jepa | PASS | N-A | PASS 14 | N-A | N-A | **no** | 443 | 191 | 43% | 0 |
| burn-mod | PASS | N-A | PASS 4 | N-A | N-A | **no** | 411 | 131 | 31% | 0 |
| burn-ptrn | PASS | N-A | PASS 12 | N-A | N-A | **no** | 399 | 191 | 47% | 0 |
| burn-eggroll | PASS | N-A | PASS 7 | N-A | N-A | **no** | 327 | 149 | 45% | 0 |
| burn-es | PASS | N-A | PASS 9 | N-A | N-A | **no** | 345 | 148 | 42% | 0 |
| burn-engram * | PASS | N-A | PASS 10 | N-A | N-A | **no** | 586 | 171 | 29% | 0 |
| burn-byteflow | PASS | N-A | PASS 15 | N-A | N-A | **no** | 1121 | 403 | 35% | 5 |

`*` under concurrent edit → **result not reliable**. `n-m` not measured (GPU
occupied by another workload, or the workspace was unresolvable).
`burn-mor` declared `cuda` at 17:50; a concurrent agent removed the feature at
~19:40, so its column 2 changed mid-session.

Non-leaf: `burn-fused` meta (44 loc, 0 tests, declares `cuda`) PASS on check
before the facade regeneration broke resolution. `burn-fused-benches` /
`cpu-probe` / `launch-probe` are CUDA-only probes (hard `burn/cuda`), 0 tests,
**not measurable** — blocked by the unresolvable workspace.

## Derived numbers (the deliverable)

* **Builds on ndarray: 28 / 28.** Not one leaf crate fails `cargo check -p X`.
  The library's compile health is genuinely good.
* **Builds on cuda: 10 / 10 measured** (attnres, bitnet, mhc, muon-plus, rope,
  sct, situ, swiglu, rmsnorm, spectral). 2 cuda crates unmeasurable (kda, gdn2)
  only because the workspace stopped resolving.
* **Has ANY test: 28 / 28.** Every leaf crate has unit tests. "0 tests" is only
  true of the meta-crate and the 3 bench members.
* **Has a test that actually exercises the cuda path: 4 crates** ship
  `#![cfg(feature = "cuda")]` test files (kda, gdn2, sct, + none else).
  **Measured green on cuda: 1** (burn-sct, 16 tests). kda/gdn2 unmeasured,
  muon-plus broken. So **13 of the 14 cuda-advertising crates have their cuda
  path only `cargo check`-ed, never executed** — the real coverage hole.
* **Test-to-code ratio: 11 578 / 37 242 loc = 31.1%.** Test lines are **45.1%**
  of non-test lines. Nobody can claim "this library has no tests" — the ratio is
  the healthiest number in it. (Measured: 218 passing test cases on ndarray,
  73 on cuda, across 25 of 28 crates.)
* **Markers: 9 in 37 242 lines, and ZERO `unimplemented!` / `todo!()` macros
  anywhere.** 5 are `TODO(бумага)` paper-spec gaps in `burn-byteflow/src/net.rs`;
  2 are `unreachable!()` match arms (`burn-sct/src/qr.rs:451`,
  `burn-spectral/examples/tsct_diag.rs:670`); 1 `TODO(gpu)`
  (`burn-sct/src/qr_cuda.rs:369`); 1 in burn-gdn2. No half-finished stubs.
* **Doc lies: 0.** 15 crates have no `cuda` feature. A naive "mentions
  cubecl/fused" scan flags `burn-mod` and `burn-engram`, but both *document the
  absence honestly* (`burn-engram/src/lib.rs:79-81`: "this crate has no cubecl
  dep"); the 3 bench members mention cubecl because they are cubecl probes. No
  crate advertises a fused kernel it does not ship. **One exception found in the
  other direction:** `burn-gdn2/tests/lowp_bf16_cuda.rs` is named `*_cuda` and
  has no `#[cfg]` guard, so it compiles and runs on ndarray — it tests nothing
  about cuda. Same for `burn-muon-plus/tests/bf16_matmul.rs` and
  `self_checks.rs` (ungated), so the bf16-matmul landmine is untested on cuda.
* **Dead weight: 18 crates**, 13 273 loc = **35.6% of the library** —
  burn-antihall, burn-attnres, burn-byteflow, burn-diffusionblocks, burn-eggroll,
  burn-es, burn-fastblt, burn-mhc, burn-mod, burn-mtp, burn-nope, burn-parcae,
  burn-ptrn, burn-rope, burn-sct, burn-situ, burn-swiglu, burn-ttt.
  Nuance: 3 of them (`burn-sct`, `burn-rope`, `burn-situ`) are dev-deps of
  `burn-spectral`, which dormouse *does* use — they are needed to run its tests,
  not to build the product.
  Referenced by `crates/dormouse-*` (10 of 28, 23 439 loc): burn-bitnet,
  burn-dspark, burn-engram, burn-gdn2 (transitively via burn-kda), burn-jepa,
  burn-kda, burn-mor, burn-muon-plus, burn-rmsnorm, burn-spectral.

## The five defects, exact text

1. **burn-spectral/examples/tsct_diag.rs — never compiled, as suspected. CONFIRMED BROKEN.**
   ```
   error[E0107]: struct takes 0 generic arguments but 1 generic argument was supplied
     --> crates/burn-spectral/examples/tsct_diag.rs:21:32
      |
   21 | type SctBackend = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;
      |                                ^^^^^^^^^^^--------------------------- help: remove the unnecessary generics
   error[E0107]: struct takes 0 generic arguments but 1 generic argument was supplied
      --> crates/burn-spectral/examples/tsct_diag.rs:1506:28
   error: could not compile `burn-spectral` (example "tsct_diag") due to 2 previous errors
   ```
   `CubeBackend` in `burn-cubecl-0.22.0-pre.4/src/backend.rs:87` has 0 generic
   params; the file was written against a different burn API. 2 of its 7
   examples are also broken: `linear_timing.rs:83,86` →
   `error[E0433]: could not find 'fused' in 'burn_spectral'` (a module a cut
   agent deleted) and `moe_timing.rs:26` → `error[E0283]: type annotations
   needed` on `opt.step(1e-3.into(), ...)`. 4 of 7 examples compile
   (infer_bench, moe_flops, moe_timing2, gpu_check). **3/7 broken, and the
   whole `cargo check -p burn-spectral --examples` cell is FAIL.**
2. **burn-muon-plus — its entire test suite does not compile** (cells 3 and 4).
   3x `error[E0308]: mismatched types` in `tests/zero_grad.rs`
   (lines 11, 34, 35, 38, 39, 58, 60, 61):
   ```
   error[E0308]: mismatched types
      --> crates/burn-muon-plus/tests/zero_grad.rs:11:5
       | expected `GradientsParams`, found `Tensor<2>`
   note: method defined here
      --> burn-optim-0.22.0-pre.4/src/optim/module/module_optimizer.rs:163:12
   ```
   The test calls the tensor-flavoured `opt.step(lr, w, g)`; the resolved type
   wants `GradientsParams`. `warning: unused import: MuonPlus` alongside.
   Caveat: `src/lib.rs` was rewritten at 18:19, mid-measurement.
3. **burn-kda — 3 of 4 examples fail to compile, and that fails `cargo test`.**
   ```
   error[E0601]: `main` function not found in crate `bitforbit_cuda`
      --> crates/burn-kda/examples/bitforbit_cuda.rs:135:2
   error: could not compile `burn-kda` (example "bitforbit_cuda") due to 1 previous error
   error[E0601]: `main` function not found in crate `kda_step_probe`
      --> crates/burn-kda/examples/kda_step_probe.rs:139:2
   ```
   Also `kda_bench`. The examples are `#![cfg(feature = "cuda")]`-gated and
   `burn-kda/Cargo.toml` declares **no `required-features`** (0 occurrences, vs
   7 in burn-sct, 3 in burn-spectral). So plain `cargo test -p burn-kda` and
   `cargo check -p burn-kda --examples` both fail. 2 of the 3 files are
   unmodified in git → **committed, not in-flight**.
4. **burn-attnres — CUDA numerical divergence.**
   ```
   thread 'fused_attnres::tests::streaming_fused_matches_tensor_path' panicked at
     crates/burn-attnres/src/fused_attnres.rs:951:13:
   step 3: maxdiff 0.94786954
   test result: FAILED. 11 passed; 1 failed; 2 ignored; 0 measured
   ```
   The fused streaming kernel diverges from the tensor path by ~0.95 at the 4th
   step (threshold 1e-4). **Not** one of the four known landmines — a real
   numerical bug. 11 sibling cuda tests pass.
5. **burn-spectral's ndarray test cell fails on a *dependency***, not on itself:
   `cargo test -p burn-spectral` → `error[E0603]: module 'tensor' is private` at
   `crates/burn-gdn2/src/cuda_dispatch.rs:38` (`use burn_autodiff::tensor::AutodiffTensor;`
   — the module is not `pub` in `burn-autodiff-0.22.0-pre.4/src/lib.rs:30`).
   Consequence worth knowing: **a private-module slip in burn-gdn2 breaks
   burn-spectral's whole test build**, because gdn2 arrives via burn-kda.

## Environment hazards this run exposed (not crate defects)

* **A half-written manifest is a workspace-wide outage.** Twice, another agent's
  in-flight `crates/burn-gdn2/Cargo.toml` made `cargo` fail with
  `error: failed to load manifest for workspace member .../burn-gdn2` for
  *unrelated* crates. On the second occurrence a 3-crate cell (situ, swiglu,
  rmsnorm) failed this way and passed on retry.
* **The `burn-fused` facade regeneration unresolvable the whole workspace** for
  the last ~30 min: `tools/gen_facade.py` (new, untracked) generated a
  `burn-fused/Cargo.toml:36` requiring `"burn-mor/cuda"`, while
  `crates/burn-mor/Cargo.toml` has no `cuda` feature (it matches HEAD).
  ```
  error: failed to select a version for `burn-mor`.
      ... required by package `burn-fused v0.1.0 (.../burn-fused)
  package `burn-fused` depends on `burn-mor` with feature `cuda` but `burn-mor` does not have that feature.
  ```
  `cargo metadata` fails ⇒ **every** `cargo check/test` in the tree fails. That
  is what made burn-kda/burn-gdn2 cuda-check and the 3 bench packages
  unmeasurable. HEAD is fine; the working tree is not.
* **`burn-sct` cuda-test failed once with a linker error and passed on retry** —
  parallel-agent link contention, not a crate defect. Recorded as FLAKE.
* A foreign session ran `bench_all.py` on the GPU (1.2 GB) mid-run; CUDA work
  was deferred until it exited rather than run alongside it.

## Backend-landmine classification

* **No failure in this snapshot matched the four named landmines** (bool→float
  cast → 0.0, bf16 matmul at `burn-cubecl ops/tensor.rs:150`, f16
  `builtin.fp16 to implement dyn SizedType`, top-k→gather
  `CUDA_ERROR_ILLEGAL_ADDRESS`). Nothing here is BLOCKED-BY-BACKEND; all four
  failures above are BROKEN. The top-k→gather bug is being fixed concurrently in
  the new untracked `burn-mor/src/topk_gather.rs`, and
  `burn-mor/examples/topk_gather_repro.rs` is its reproducer.
* Two test files named `*_cuda` have no `#[cfg]` guard and therefore run on
  ndarray (`burn-gdn2/tests/lowp_bf16_cuda.rs`,
  `burn-muon-plus/tests/bf16_matmul.rs`) — passing results that prove nothing
  about cuda. A cheap, high-value fix: add the guards.

## Rows that are unreliable, and why

| row | why |
|---|---|
| burn-kda (2, 4, 5) | unmeasurable: workspace unresolvable; also `src/lib.rs`, `src/fused.rs`, `examples/kda_step_probe.rs` rewritten during the run; `tests/cuda_gate.rs` appeared mid-run |
| burn-gdn2 (2, 4, 5) | same; plus `src/{lib,forward,autodiff,cuda_dispatch}.rs`, both `kernel/chunk_*.rs`, `tests/bit_exact.rs` all rewritten 18:54–19:07 **during** the 499 s test run; `src/cuda_dispatch.rs` + 4 new test files untracked |
| burn-spectral (3, 4, 5) | its cell-3 failure is a *gdn2* file mid-edit; `examples/tsct_diag.rs` was written at 17:18 (28 min before I looked) and the agent may still be mid-cut. Defect 1 and 2 are stable in git, defect 5 is not |
| burn-mor (2, 4, 5) | its `cuda` feature was **removed by another agent mid-session** (present 17:50, gone 19:40); `src/lib.rs` modified; `examples/` untracked |
| burn-engram (all) | flagged for concurrent edit but produced a clean, plausible 10-test pass; no file under it changed during the window — **the most trustworthy of the `*` rows** |
| burn-muon-plus (3, 4) | `src/lib.rs` rewritten at 18:19, between my cuda-test and my ndarray test; the E0308 may already be fixed. Defect 2's *text* is real, its *currency* is not |
| bench members (all) | never measurable — the workspace would not resolve |
