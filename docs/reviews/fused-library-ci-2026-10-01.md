# fused-library CI: the three red jobs, one cause each

**Date:** 2026-10-01. **Scope:** the three red jobs of `.github/workflows/fused-library.yml`
as of run `36871799646` (head `98e24b1`): `ref_f64_broad.bin regenerates
byte-identically`, `cuda feature compiles`, `feature matrix`. The `ndarray` job is
red by design (documented in the workflow itself, `burn-spectral`'s three
`retract` panics) and was not touched.

`1000 cases vs the f64 oracle` has been green throughout and is the reason the
other three could stay red for so long: it passes because the fixture is what
the test compares against, so it cannot see that the fixture stopped matching
its generator.

| job | first red | cause | fix |
|---|---|---|---|
| `ref_f64_broad.bin` | its first run, `36851463491` (`ae0505da3`); red on 14 of the 20 runs since | the job's contract was byte-equality on f64 numpy reductions, which no two hosts agree on | run the fixture checker the repo already ships (`tools/check_f64_fixtures.py`), pin numpy, pin BLAS threads |
| `cuda feature compiles` | its FIRST run, `36332113716` (`e904bcc8`, 2026-09-27) — never green | five gate defects in three crates, all "a `cfg` that does not cover its caller", plus a hand-written crate list that named a crate with no `cuda` feature | the gates, and a derived list (`gen_facade.py --cuda-matrix`) |
| `feature matrix` | its first run too, same day | the same defects, seen through the facade: `std,cuda` and `std,autodiff` | the same commits |

**The lesson worth more than the three fixes:** every one of these had been
red, loudly, for hours-to-days, and the reason nobody acted is that the two
jobs that report them *stopped at the first failure* and the third (`feature
matrix`) *printed no diagnostics* (§ follow-ups 2). A gate that names one failure
out of six is a gate that reports one sixth of the truth.

---

## 1. `ref_f64_broad.bin regenerates byte-identically`

### What was measured, before anything was changed

The job regenerates the fixture with `tools/gen_reference_f64.py` and then
`git diff --exit-code` on the bytes. Red on **14 of the last 20 runs of
unmodified code** — green, red, red, red, red, red, red, red, green, red, green,
red, green, green, red, red, green, red, red, green — on the same sha, which is
the signature of an environment axis rather than a code change. Candidates, and
what each one measured:

| axis | measured | verdict |
|---|---|---|
| numpy version | a green run (`36869874421`) and a red one (`36871799646`) both installed `numpy-2.5.3-cp312-cp312-manylinux` | **not the axis** |
| python version | regenerated under 3.14.7 and under 3.12.14, both numpy 2.5.3: `sha256 4c83b0f14228496f…` for both, and for the committed file | **not the axis** |
| BLAS threading | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` gives the identical `4c83b0f14228496f…` | **not the axis here** |
| the host CPU | left by elimination, and it is not speculation: the f64 outputs are numpy reductions whose order is picked by runtime CPU dispatch, and the generator's matmuls go through an OpenBLAS whose own build string reads `OpenBLAS 0.3.34.106.0 USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64` — `DYNAMIC_ARCH` means it selects kernels per host CPU. GitHub-hosted runners are mixed hardware, which is also why two runs ten minutes apart disagree. | **the axis** |

The generator itself already says it is f64 transcription code written "one
token at a time", and its `Rng` docstring claims the fixture is "provably
stable" without a version pin. That claim is about the RNG — stdlib integer
arithmetic, and it holds — and it was read as covering the OUTPUTS, which are
BLAS.

### The fix

**The gate the repo already ships was not wired into CI, and the file a reader
is told to consult had already ruled the CI recipe wrong.**
`crates/burn-gdn2/tests/oracle_breadth.rs:100-118` says, in its own header:

> REGENERATE, AND HOW TO KNOW IT IS THE SAME BYTES - BY RUNNING THE CHECKER, NOT
> BY `git diff`. … So `git diff --exit-code` FAILS on a correct regeneration, and
> the person following it learns to ignore it - which is one of the two ways this
> fixture went stale for a day without anyone noticing.

and names the check: `tools/check_f64_fixtures.py`, which `tools/lib_gate.sh:65`
runs. The tool's own docstring says the same thing about CI:

> It is deliberately NOT `git diff --exit-code`: that recipe is in two test
> headers and it is wrong, because a correct regeneration is not
> byte-identical.

So the wrong recipe was documented in three places and the right tool was run by
exactly one of them — the local gate. A CI job is not a place where a decision
gets recorded without also being executed.

That tool carries the measurement the contract needs: the f32 weights and
inputs compared **bit-exactly** (the RNG is deterministic, so a difference there
is a changed input, not rounding), the f64 OUTPUTS compared at `OUT_TOL = 1e-12`
relative against a measured round-off of ≤ 4.4e-15 (typically ~7e-16), and the
smallest wrong formula in the committed negative control sits at 4.08e-01 —
**4.1e+11 further out**, so the bar cannot pass a semantic change and cannot
trip on rounding. It regenerates into a temp dir, so the working tree is never
mutated and the comparison cannot pass by failing to run.

The job now runs that tool, with `numpy==2.5.3` pinned (it removes the
resolution axis even though it was not the axis) and `OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1` set (the other half of the same axis, free). The
fixture bytes are NOT re-emitted: they are correct, and the oracle job that
reads them stays green.

**Byte-equality was this job's contract, so this is a change of contract and it
is deliberate.** It is also the only contract a cross-machine job can hold. The
alternative — making the generator bit-canonical — would have to remove BLAS
from every matmul AND remove `log`/`exp`/`cos` from the parameter stream
(`A_log = log(u)`, `dt = exp(...)`, Box-Muller for the inputs), which is a
redesign of the fixture, and the transcendental axis still could not be closed
from a Broadwell host. The margin table is the evidence that 1e-12 is the right
place to put the line.

## 2. `cuda feature compiles` — red in every run of the workflow's life

`36332113716` (2026-09-27 16:08) is the first run this workflow ever had and
`cuda feature compiles` is already red in it; so is `feature matrix`. The jobs
were introduced by `dfce116` ("test(fused-lib): commit the bit-exact reference,
and a CI that can fail"), and the defect they report predates them.

**A job that stops at the first failure hides the queue behind it.** The step
died on burn-gdn2, the third crate of thirteen, so nobody ever saw the other
four defects — or the two in the workflow's own crate list. Running the loop's
intent over every declared combination found **five code defects in three
crates**, all the same shape (a `cfg` that does not cover the code that uses
it), plus two more in the list itself.

### 2.1 burn-gdn2 — the seam, counted from two halves its gate excluded

```
error[E0433]: cannot find `cuda_dispatch` in the crate root
  --> crates/burn-gdn2/src/kernel/chunk_adjoint_cube.rs:580:16
   |         crate::cuda_dispatch::note_fused_backward();
note: found an item that was configured out
  --> crates/burn-gdn2/src/lib.rs:70:9
   |  #[cfg(feature = "autodiff")]
   |  pub mod cuda_dispatch;
   |          the item is gated behind the `autodiff` feature
```

and the mirror image at `chunk_cube.rs:966` (`note_fused_forward`).

Three gates, two call sites, no combination that satisfies all of them:

| | gated on | who calls into it |
|---|---|---|
| `pub mod cuda_dispatch` (`lib.rs`) | `autodiff` | — |
| `mod fused`, holding `note_fused_forward` / `note_fused_backward` | `cuda` | `kernel/chunk_cube.rs`, `kernel/chunk_adjoint_cube.rs` — `#[cfg(feature = "cuda")]` |
| `note_untracked`, `note_tensor_branch`, `note_backward_node`, `note_fused_declined`, `fused_forced_off` | `cuda` (same `mod fused`) | `autodiff.rs` — `#[cfg(feature = "autodiff")]` |

`--features cuda` compiles two call sites into a module that does not exist;
`--features autodiff` compiles five more into functions the `cuda` gate removed.
The call sites were added by `8fa5d4c` ("the attention arm had NO gradient") and
`2a430cc`, both of which moved a counter into the gated block.

**The fix moves the gates, not the calls.** Gating the CALL SITES would compile,
and would be ADR-0019's SILENT failure wearing a green build: a fused kernel
that launches and does not count is precisely the shape of the bug that module
was written for. So the counters carry no gate and the module is
`any(cuda, autodiff)`; only the items naming `CudaBare` or `burn_autodiff` keep
one. No behaviour changes on any combination that compiled before.

### 2.2 burn-gdn2's crate root — a one-word gate that took three commands down

`Fallback` / `Fused` / `backend_matches` were re-exported under `autodiff` only,
and `crates/burn-kda/src/fused.rs:118-122` is gated `cuda` and names the first
two. So `cargo check -p burn-kda --features cuda`, the facade's `std,cuda`
combination, and `cargo check -p burn-fused-benches` were all red on the same
line. Those three move to `any(cuda, autodiff)`; `rebuild`, `strip` and
`AdNode` name `burn_autodiff` types and stay behind `autodiff`.

### 2.3 burn-rope — `any(cuda, autodiff)` on a module whose kernels need `cuda`

`rope_cuda` is gated `any(cuda, autodiff)`, but its two kernel definitions are
written in `#[cube]` and its public entry point reaches for `burn_cubecl` in its
first line. Under `autodiff` alone: `cannot find attribute cube`, five
`cannot find attribute comptime`, `cannot find module or crate burn_cubecl`. The
three items get `#[cfg(feature = "cuda")]`; the module stays `any(...)` because
`mod ad` inside it is the autodiff half and `rotate.rs` calls each half under
its own gate.

### 2.4 burn-mhc — a test nothing had ever compiled

`fused_node_carries_a_gradient` is `#[cfg(all(test, cuda, autodiff))]` and
`#[ignore = "needs a GPU"]`. It uses `Device` and `Distribution` and imported
neither. Being `#[ignore]`d it has never RUN; until `cargo check --all-targets`
reached it, no command in this repo's history had ever COMPILED it either. The
import goes inside the `fn` — at module scope it would be an unused-import
warning in every other combination.

### 2.5 The workflow's own crate list

The loop carried a hand-written list of 13 crates, commented

> Kept as an explicit list because `--features cuda` on a crate without it is a
> silent no-op.

**It is not a no-op.** Cargo rejects a feature a package does not have and names
the packages that do:

```
error: the package 'burn-mor' does not contain this feature: cuda
```

`burn-mor` is in the list and declares no `cuda` — and the workflow's own comment
two lines up records that "the hand-written list named 12 (burn-mor missing) for
months". That fix landed in the facade's GENERATED feature list and never reached
this loop, which is what a second hand-written copy of a generated fact buys.
`cuda,autodiff` was also asked of burn-muon-plus, burn-rmsnorm, burn-spectral
and burn-swiglu, none of which declares `autodiff`: four more hard errors.

The list is now derived — `gen_facade.py --cuda-matrix` reads every member
manifest and prints only combinations that exist (**12 crates, 20 combinations**),
and `--check` already fails CI when a manifest and the generated files disagree.
`cargo test --no-run` became `cargo check --all-targets`: the step's own comment
says what it is ("the crate COMPILES and cannot fail on a CUDA-only bug"),
`--all-targets` covers lib + tests + examples + benches, and it is strictly
wider than `test --no-run` — it is what found §2.4.

## 3. `feature matrix` — the same defects, seen from the facade

`tools/test-feature-matrix.sh` builds seven combinations of the facade. Two
were red:

```
FAIL  burn-fused --no-default-features --features std,autodiff
FAIL  burn-fused --no-default-features --features std,cuda
PASS  burn-fused --no-default-features --features std,cuda,autodiff
```

`std,cuda` misses the seam (§2.1 + §2.2 via burn-kda) and `std,autodiff` misses
burn-rope's kernels (§2.3); only the combination that has both was green.
`gen_facade.py --check` passed at HEAD, so the job's first gate was never the
problem. All seven PASS locally after the fixes above.

## Not fixed here, with the file that has to change

1. **`cargo fmt --all -- --check` is red on 48 files / 279 hunks** in
   `vendor/dormouse-fused`, measured locally at HEAD. The `ndarray` job runs it
   as its last step, so that step never executes today (the `test` step fails
   first) and will start failing the day the three `burn-spectral` panics are
   fixed. "The `ndarray` job is red by design" is true of the panics and only
   of them. Fixing it is a repo-wide reformat across crates other lanes own.
2. **`tools/test-feature-matrix.sh` prints no diagnostics on a FAIL**, and
   `cuda-compiles` stopped at the first broken crate. Both are the same disease
   and both cost this document its speed. The script does
   `echo "$out" | grep -E "^(error|...)"`, and the workflow sets
   `CARGO_TERM_COLOR: always`, so every `error` line arrives as
   `\e[1m\e[91merror[E0433]…` and `^error` matches nothing — the two FAIL rows
   quoted in §3 are the ONLY diagnostics the log carries, and both causes had to
   be read out of the raw log. A gate that names a failure and hides its cause
   is a gate nobody can act on; a loop that dies on crate 3 of 13 reports 3 of
   13. Strip the escapes before grepping, and collect every failure rather than
   the first.
3. **The `ref_f64_broad.bin` job installs mold** ("the repo config links through
   it"). After the fix the job runs no cargo at all, so the `apt-get` is dead
   weight. Left alone: a future cell may add cargo back, and the job's shape is
   not this lane's call.