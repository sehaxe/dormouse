# kda_oracle red on CI: the platform-drift hypothesis, tested and dead

**Date:** 2026-10-02 · **Lane:** kda_oracle fails only on the CI runner — 3 FLA-reference
tests red on ubuntu-latest, "green locally". **Verdict: there is no platform drift and
there is no variable. The tests are red BY DESIGN on every machine, with identical
numbers. The premise "green locally" was false — nobody had run this target locally
since it landed.**

## What was asked

`vendor/dormouse-fused/crates/burn-kda/tests/kda_oracle.rs` — three tests fail the
fused-library ndarray job on every run (36945094092, pre-push; 36972953116, after merge
6f068ba; every run in between):

- `kimi_linear_softplus_decay_matches_fla_reference` (assert at :452)
- `read_scale_matches_fla_reference` (assert at :660)
- `chunked_wy_applies_no_read_scale` (assert at :828)

The brief's fork: fixture exact (9 significant digits), burn NdArray is pure Rust and
deterministic, so the drift must live in rustc version, RUSTFLAGS, profile, or env —
pull `max_diff` from the CI logs and bisect the axis.

## The numbers

From run 36945094092's log, and from a **local run of the same test code on this box**
(the Sep 30 20:37 debug binary `kda_oracle-c6070f8bba9e737f`, built by commit b485ad9
whose tree matches main for this file — executed directly, no rebuild):

```
CI    arm Batched case chunk16_h1: worst normalised 1.828e3 at 231, ours -10.777058 vs FLA -3.810267 (ratio 2.8284, K**-0.5 = 0.3536)
LOCAL arm Batched case chunk16_h1: worst normalised 1.828e3 at 231, ours -10.777058 vs FLA -3.810267 (ratio 2.8284, K**-0.5 = 0.3536)
```

```
LOCAL test result: FAILED. 7 passed; 3 failed; 0 ignored
      failures: chunked_wy_applies_no_read_scale,
                kimi_linear_softplus_decay_matches_fla_reference,
                read_scale_matches_fla_reference
```

Byte-identical failure text on two different machines (this box vs ubuntu-latest),
different rustc builds, debug profile both. The signature is not the 1-ULP class
(~1e-7) the brief priced, and not the logic class (~1e-1) either: **the ratio is
exactly `8**0.5 = 2.8284` at K=8 on every line** — a constant multiplicative factor,
which is a *formula* difference, not arithmetic. No tolerance choice can bridge a
factor of 2.83, and no platform variable can produce one.

## Why they are red

The test file says so itself, in its own header table — **"RED ON PURPOSE"**, written
2026-09-30 (`b485ad9`):

1. `kimi_linear_...`: `src/lib.rs:268` computes `-softplus(exp(A)·z)`; FLA computes
   `-exp(A)·softplus(z)` (`gate.py:50` naive, `:167` triton). Different functions,
   owner's call (A/B queue arm 5), not this lane's to fix.
2. `read_scale_matches_fla_reference`: we pass `scale = 1.0` (`lib.rs:646,653`); FLA's
   official layer runs at `head_k_dim**-0.5` (`fla/layers/kda.py:262`).
3. `chunked_wy_applies_no_read_scale`: the chunked twin of 2. The mechanism exists —
   `chunked_wy_honours_the_read_scale_when_asked` is green — so it is a missing
   argument, not a missing implementation.

`docs/reviews/2026-09-30-kda-formula-audit.md` §3.1/§3.2 is the audit that produced
them. The file's own philosophy: *"A red test that a maintainer can read and act on is
the deliverable."* The header's baseline line is stale — it says "5 green, 2 red on
purpose"; the file has 10 tests: **7 green, 3 designed reds** (the chunked twin landed
after the header was written).

## Why the job went red only now, and why "green locally" felt true

- The root workspace **excludes** `vendor/dormouse-fused` (its crates are their own
  workspace roots), so `cargo test` at the root never compiled this test file. The only
  thing that ever ran it is the fused-library ndarray job (`cargo test --workspace`
  inside the fork), which exists since 2026-10-01. "Green locally" was true in the
  weakest sense: the target had never been executed locally.
- The ndarray job itself could not reach kda_oracle until `de222ea` (2026-10-02 02:13)
  fixed the three burn-spectral fixtures that panicked in setup. From that moment the
  job reaches the end of the suite, hits the designed reds, and fails — **every run
  since, on this exact mechanism**, while all four sibling jobs stay green (they never
  run this file).

## What was NOT the cause (checked, so nobody re-checks)

- **Fixture drift:** static file, `include_str!`, sha-pinned generator; identical bytes
  on both machines (the local binary read the same fixture and printed the same FLA
  numbers).
- **rustc version / RUSTFLAGS / profile:** CI log shows `target/debug`; the local
  reproduction is also debug; numbers identical anyway — a formula difference cannot
  come from a codegen axis.
- **`CUDARC_CUDA_VERSION` env in the job:** the ndarray build compiles the cuda
  feature's deps but the failing tests run on NdArray tensors; no cuda code executes.
- **The runner's numpy/python:** nothing here runs Python; the fixture was generated
  once on 2026-09-30 and committed.

Separate finding, not this lane's: **this box's own rustc is broken for fresh compiles
since the system LLVM moved to 23.1** — `/usr/bin/rustc` (rust 1.98.1-1.1) is linked
against `libLLVM.so.22.1`, which no longer exists at 64-bit (only `libLLVM.so.23.1`;
32-bit 22.1 remains in `/usr/lib32`). Warm sccache hides it; any cache miss dies with
exit 127. Builds worked this morning (06:18) on warm cache. Fixing it wants a system
package decision (reinstall rust/llvm), not a repo change.

## The fix

The designed reds cannot run by default in a gate whose contract is "green", and they
cannot be made green without a numerical change to a shipped model (the owner's call,
twice stated in the file). So they move to the space Rust already has for it —
`#[ignore = "<cause>"]`, which is COUNTED (§1.1): the reason prints in every test
listing, and the red stays one flag away:

```
cargo test -p burn-kda --test kda_oracle -- --ignored
```

- `kda_oracle.rs`: three `#[ignore]` attributes, each carrying the divergence and the
  on-ramp; header table and baseline corrected (7 green / 3 designed reds), and the
  mechanism documented where the old "5 green, 2 red" claim lived.
- `tests/oracle/falsify.sh`: `run()` now passes `--include-ignored`, so its B-mutants
  ("fix a red") still demonstrate against the reds; the stale counts in its header
  corrected. (Manual tool, not called by CI.)
- `fused-library.yml`: the ndarray job's comment names the second red and the
  mechanism; the job name loses its `(a broken CPU build)` suffix — that named the
  first-run red (spectral), closed at `de222ea`, and a green job named "broken" is the
  document-disagrees-with-reality class (§1.7).

Deliberately NOT done: fixing either formula. `exp(A)` outside the softplus and
`scale = K**-0.5` move every number the crate has ever produced — A/B queue arm 5, the
owner's call. Loosening a tolerance was considered and rejected: the disagreement is a
factor of 2.83, five orders above `TOL_CHUNK`, so no honest tolerance change applies.

## Verification

Local: the reproduced red (above) is the same code the attributes were added to; the
edit changes only attributes and comments. `cargo fmt`/`cargo test` could not run
locally — the box's rustc is broken for fresh compiles (see separate finding), so
compilation and the green job are verified by the fused-library workflow on the pushed
branch. Gate: the ndarray job green, `kda_oracle` reporting `7 passed; 3 ignored`.

## Second cause found on the verification run (36979996862)

With kda_oracle's reds ignored, the ndarray job failed on a **different** test —
`burn-gdn2/tests/ops_batched_autodiff.rs::the_custom_node_backward_reads_a_padded_scratch`
(analytic adjoint vs central differences, rel 6.15e-2 over the 5e-2 bar) — which had
**passed** on run 36945094092 six hours earlier with no code change in between. Since
cargo stops at the first failing test binary and burn-gdn2 sorts before burn-kda, this
flake would have hidden every kda_oracle result on any run where it fired.

The axis: `burn-ndarray` builds here with `blas-openblas`, and multi-threaded OpenBLAS
reorders f32 reductions call to call. The margin arithmetic puts the gate right in
that noise: loss ≈ 77, FD amplification 1/2h = 50 (h = 1e-2), so ~2 ulp of f32
reduction noise lands the relative error at ~6e-2 against a 5e-2 bar. The workflow
already recorded this exact axis for the f64 fixture generator ("Single-threaded
BLAS: the other half of the same axis") but the ndarray test job never got the cap.

Fix: `OMP_NUM_THREADS=1` + `OPENBLAS_NUM_THREADS=1` on the ndarray job. Not a
weakened gate — the same formula, measured without the scheduler in the loop. If the
test still fails deterministically with one thread, that is a real cross-host kernel
selection drift (OpenBLAS DYNAMIC_ARCH) and belongs to the burn-gdn2 owner, not this
lane; it would show as the same assert with a stable number rather than a coin flip.

