How to know a change did not break the model

There is no end-to-end golden yet. This file is the specification for one, and
the list of what is currently unverified — because a rule that is not written
down is not followed, and "we checked it once by eye" is how a **step-0
reading** became the project's headline optimization target.

**The anecdote that replaced the old one, 2026-09-29.** A 10 809 ms step with
`opt` at 8303 ms of it was read once, off a `--timers` line, and promoted to
`AGENTS.md` §3.1 as "THE STEP IS 77% OPTIMIZER" before anyone replicated it.
Replicated on a quiet card in release: a **warm** step is ~245 ms and `opt` is
43 ms of it — a **23× step / 94× opt** error, entirely because the cubecl
autotune cache is cold for the first steps. Retracted in
`benches/history.tsv` (2026-09-29 block). The first version of this sentence
named a different number, 25858 ms/step, as the founding cautionary tale; that
figure was itself measured under load (three compilers on the box, dev
profile, eval enabled) and is struck in the same file, so it is withdrawn
rather than replaced.

**The rule that came out of it, and it is the one this file is for: a
step-time reading with no step index is not a measurement of a step.**
`--timers` used to print on `step % 50 == 0` only and was **not** tied to
`--log-every`, so every short run in this project's history could only ever see
step 0. **That is fixed** (`b8a47ee`): the cadence is now
`step == 0 || step % log_every == 0` (`crates/dormouse-train/src/lib.rs:1196`),
so a run's log cadence sets it and `benches/history.tsv` rows from today on
carry a real step index. The defect is historical, not live.

## The rule

**Every optimisation must preserve the model's output, bit for bit or within a
stated tolerance, against a fixture captured from the pre-change code.** A speed
result without that comparison is not a speed result; it is a different model
that happens to run faster.

## What already exists, and what is wrong with it

`vendor/burn-fused/crates/burn-gdn2/` has the right *shape* — a committed
fixture, a generator that is the only thing permitted to rewrite it, a test
that compares the kernel against the file, and a CI job that regenerates and
`git diff --exit-code`s so the fixture cannot drift from its generator:

- `tests/ref_data.bin` — 7 MB of committed reference activations
- `tools/gen_reference.rs` — regenerates it, and is the only thing that may
- `tests/bit_exact.rs` — compares the kernel against the file

**Copy the shape. Do not copy the instance, and do not cite it as a passing
gate.** Three defects in it, each sourced:

1. **It is RED.** 1000 cases, `max_diff = 1.38e-2`, **976/1000 failures** at
   `EPSILON = 5e-4`, measured 2026-09-27 on ndarray against this fixture
   (`tests/bit_exact.rs` module header; `vendor/burn-fused/TEST-AUDIT.md`
   FINDING 0). The audit's own analysis: not the tolerance (5e-4 is 28× below
   the observed diff and above the 2e-6..2e-5 measured transcription noise),
   not the fixture format, not a per-token difference — the 24 cases that pass
   are exactly the 24 single-token cases, and the divergence lives in what one
   token cannot exercise. Cause **not yet isolated**; the separating
   measurement is named in the audit and was not run.
2. **The 1000-case comparison does not run by default.** It is
   `#[cfg(feature = "binary-tests")]` and `binary-tests` is deliberately
   **not** in `burn-gdn2`'s `default` features (`Cargo.toml` `[features]`), so
   a plain `cargo test -p burn-gdn2` builds the file, compiles the gated test
   away, and prints green having asserted nothing — the same
   "empty binary says ok" class that `af34dda` closed for 21 other targets
   across this library. The comparison runs only in the explicit CI job
   `fused-lib :: 1000 bit-exact cases vs the paper reference`
   (`.github/workflows/fused-library.yml`, `--features binary-tests --test
   bit_exact`), which is therefore permanently red on that step. A suite you
   have to name on the command line is not a gate on the change that breaks
   it. (The file also holds two ungated ndarray *benchmarks*, which do run by
   default — so the target is not empty, which is exactly why the missing
   feature gate on the one real assertion went unnoticed.)
3. **The fixture is our own transcription, not the authors' output.** Despite
   the file's name, `bit_exact.rs` is an absolute-tolerance comparison of two
   independent implementations of the same math. Its own header says a real
   bit-for-bit claim needs NVlabs' kernel in the tree, which it is not.

**What actually is the pattern to follow, in this tree, today:** Layer 1 below
(`ops_batched_diff.rs`, green, compares against the loop rather than a stored
file, so it cannot pass if both arms drift together) and the retraction
wiring — an explicit `--features` gate plus a named CI job, which is the right
shape for a *deliberately red* suite and the wrong shape for a default one.
The fixture discipline in `bit_exact.rs` (committed fixture + sole generator +
CI regeneration diff) is worth copying on its own; the comparison and its
status are not.

## The end-to-end golden, in three layers

Each layer catches a different class, and the one that matters most is the
third, because it is the only one that can catch "the arm trains but the model
is wrong".

### Layer 1 — the op, bit-exact

`ops_batched_diff.rs` already does this against the loop rather than against a
stored file, which is stronger for a rewrite: it cannot pass if both arms drift
together. Worst measured 9.1e-6 including a hostile case (||L||_inf = 10.97).
Gradient equivalence on the trainer's backend is in `ops_batched_grad_cuda.rs`:
both arms 2.40e-2, which is the f32 finite-difference noise floor, not a
difference between them.

### Layer 2 — a step, bit-exact

**To be written.** A fixture holding, for a fixed seed and a fixed 8-step
window: the train loss per step, the seven parameter-group gradient norms, and
the held-out BPB. Captured from the current `HEAD`, committed, and asserted.

The trap to avoid, and it is a smaller trap than it was yesterday: the data
order comes from the corpus, not from `--seed`, so the fixture still has to be
a small committed corpus — a directory of a few KB in
`crates/dormouse-train/tests/fixtures/`, which also makes the test independent
of the mounted drive (AGENTS.md §2.6).

**Corrected 2026-09-28 (`4b42b6d`):** this file used to say "`--seed` is only
the JEPA mask", which was wrong. `device.seed(cfg.seed)` now governs the model
**init** as well as the mask, both pure functions of `(seed, step)`. Before
that fix two runs with identical flags produced checkpoints differing in
**34 730 605 of 43 725 616 bytes**, so every A/B in the archive compared two
different initialisations. **The gap that remains, measured in the same
commit: 409 043 differing values on a repeated run — ~4 % of the model is
process entropy the seed does not reach.** Until that is zero, a fixture
captured "for a fixed seed" is not reproducible, and Layer 2 cannot be written
honestly yet. Do not record a fixture and call it a gate before that number
moves.

### Layer 3 — the run, end to end

**To be written, and this is the one that would have caught tonight.** 200 steps
on the fixture corpus, pure CE, one seed, compared against a committed loss
curve at a stated tolerance.

**It depends on Layer 2, which does not exist yet, and the sentence this
section used to carry — "layers 1 and 2 pass today" — was false in the way
this file exists to prevent.** Layer 1 passes. Layer 2 is unwritten and is
blocked on the residual seed gap above. Layer 3 is the only layer where "the
attention arm trains" and "the attention arm helps" are different questions,
and we still do not know whether a trained attention arm improves BPB, because
no run with a gradient-carrying arm has ever completed a meaningful number of
steps.

**Build order: close the seed gap first, then Layer 2, then Layer 3.** A Layer
3 fixture recorded while ~4 % of the model is process entropy is a fixture that
fails on a re-run for reasons no one will be able to name, which is worse than
no fixture.

## Currently unverified, in one place

So that "is it correct?" has an answer that is not a feeling:

| claim | status |
|---|---|
attention arm receives a gradient **in a real run** | **NOT VERIFIED.** `ops_grad_cuda` proves the *op's* derivative against central finite differences on CUDA, and that is a real gate. It is **not** the same claim: no run on record prints a nonzero attention-backward count, and the 2026-09-29 release-binary preflight printed `fused kda=64/0` in **both** arms (`.bulba/memory.md:36`). `autodiff_nested_balanced::the_op_declines_a_nested_graph_and_the_ops_path_carries_the_gradient` — the graph shape the *trainer* builds, all seven parents intermediates under `BalancedCheckpointing` — is **RED** (`.bulba/memory.md:37`), so the `8fa5d4c` defect is live on ndarray, the CPU proxy. The op gate passing and the arm training are separate questions, and only the first is closed |
fused adjoint numerically correct | VERIFIED on the **bare** path, `fused_adjoint_vs_ops.rs` on CUDA: every gradient <= 5.5e-7, worst input `g`, bar tightened 1e-2 -> 1e-3 not loosened (`e143f67`). Two limits worth naming: it runs only under `--features cuda,autodiff`, so it is not in the default cell, and it calls the kernels directly on bare tensors, so it does **not** cover the path a training step takes |
batched ops == loop ops | VERIFIED, 9.1e-6 worst case, gradients identical |
library compiles in every target | VERIFIED, `check --workspace --all-targets`, 0 errors |
product test gate | VERIFIED, 98 passed / 0 failed |
eval reads the memory arm | VERIFIED on hardware, `engram=4/4` |
best checkpoint, 3 silent-data-loss fixes | VERIFIED, 7 tests each falsified red |
optimizer routing, live vs declared | VERIFIED, red-on-revert proof |
decode path == training path | VERIFIED, decode_seam to <1e-6, decode_wiring |
step time / throughput | MEASURED, warm, in `benches/history.tsv` 2026-09-29: **~245 ms** at batch 8 (249/240/250 at steps 50/100/150), 440 ms at batch 16, 826 ms at batch 32. **Three** earlier figures are struck in that file, not two: `25858` and `3076` (measured with three compilers on the box, dev profile, eval enabled) and the `10809` "77% optimizer" reading, which was **step 0** — a step-0 step is 5549 ms against a warm ~245 ms, a 23x error |
GPU utilisation, warm | MEASURED 2026-09-29: **mean 13.3 %, 142 of 180 samples at <=5 %** over a 150-step run at batch 32. Launch-bound, now **confirmed by instrument** rather than inferred from a step-time fit. The lever is CUDA graph capture/replay, 5/5 green in `vendor/cubecl-fix/cubecl-cuda/tests/graph.rs` and **not wired** |
**a trained attention arm improves quality** | **NOT VERIFIED — no run has ever completed this** |
fused RMSNorm | NEVER RUNS, `norm=0/N` on the trainer's backend |
f16 patch speedup | NEVER MEASURED on hardware |
`--eval-depths` window | KNOWN WRONG, reads a different window than the `bpb` above it |
`--dspark-k 0` | KNOWN WRONG, becomes k=1 at the default, and kills JEPA at `mor_bce_weight=0` |
DSpark one-position shift | KNOWN, documented, deliberately unfixed — needs its own A/B |
18 library crates unwired | KNOWN, fates recorded in docs/library-crate-fate.md |
zero A/B verdicts in the project | TRUE, and every one is listed in AGENTS.md 3.2 |
the reference-fidelity suite (`bit_exact.rs`) | **RED — 976/1000 cases fail** at `EPSILON = 5e-4` (`max_diff = 1.38e-2`), and `binary-tests` is not a default feature so it does not run by default. It is a deliberate red gate, not a passing one. See the section above |
two runs under the same seed are the same run | **NO.** ~4 % of the model (409 043 of 43 725 616 values) is process entropy the seed does not reach (`4b42b6d`). Layer 2 below cannot be written honestly until that is zero |

## Two rules that follow from the table

1. **A number without its protocol is not a number.** Every row above that says
   MEASURED names its instrument in `benches/history.tsv`. When a measurement
   is retracted it is struck, not restated with a new value, because the
   conditions that produced the wrong one are not reproducible.
   **The protocol includes the step index.** A warm step and a step-0 step are
   different numbers for the same program — 23x here — and the instrument that
   produces them (`--timers`) printed on `step % 50 == 0` only until `b8a47ee`,
   so it was trivially possible to read step 0 and believe you measured the
   workload. Fixed, and the rule is now: a bench row carries its step index.
2. **The model has not been shown to learn.** The best valid held-out number in
   the archive is 4.997 BPB, scored on a **20 480 B** window (`eval_batches ×
   batch × seq_len`, so the window is a function of the batch size), against a
   5-gram bar of 2.911–2.588 depending on the fit — and **none of those four
   bar readings is on a trainer eval window** (see `docs/AB-PROTOCOL.md` and
   `anchors.rs:22-35`), so this is a statement about the gap, not a measured
   margin. What the window disagreement does not touch: every 5-gram reading
   sits 2.1–2.4 BPB below 4.997, so no choice of bar rescues the model. The
   4.997 was produced by a run with no attention arm at all. Everything above
   is about the trainer being correct, and none of it is about the model being
   good. Those are different projects and only the second one is the point.
