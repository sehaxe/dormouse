# ORACLE — what is actually being verified in `burn-kda` and `burn-gdn2`

Written 2026-09-29 at `wt/oracle`, off `eeb3b73`. The machine-readable half of
this file is `docs/ORACLE-TIERS.tsv`; the gate that reads it is
`tools/oracle_gate.py`. The false-confidence audit that already existed and is
*not* duplicated here is `vendor/burn-fused/TEST-AUDIT.md`.

Read this before calling anything in these two crates "verified".

---

## 1. The one-line answer

**The tensor-ops path cannot be the oracle, and neither can the fused path, and
neither can any test that compares them to each other.** They are two
implementations of one specification written by one person, and they share the
larger part of their code verbatim — so a wrong specification produces two
wrong answers that agree.

---

## 2. The structural fact that makes a differential test insufficient

This is not an opinion about test design. It is a property of the call graph,
and it was read out of the tree, not assumed.

### `burn-gdn2`

| what | where |
|---|---|
| `GatedDeltaNet2::project` — Q/K/V/B/W + gates + L2 norm + decay + short conv | `src/module.rs:469` |
| the single call site in the training path | `src/module.rs:292` |
| **the branch point** — fused kernel vs ops | `src/module.rs:302` (`if update_state`), dispatch at `src/module.rs:37` `chunk_wy_dispatch` |
| the shared readout — `rms_norm_gate_per_head` + `o_proj` | `src/module.rs:369-373` |
| `short_conv_1d`, called from inside `project` | `src/short_conv.rs:22`, called `src/module.rs:495-499` |

`src/forward.rs` imports `burn::tensor` and nothing local — it *is* the ops arm.
The two arms therefore diverge at exactly one place, `chunk_wy_dispatch`, and
are the *same function* everywhere else.

### `burn-kda`

| entry point | line | calls `project` | calls `output` |
|---|---|---|---|
| `forward_train_state` (ops **and** fused, one branch) | 530 | 539 | 586 |
| `forward_train_fused` (fused autodiff node) | 596 | 603 | 618 |
| `forward` (decode / prefill) | 629 | 639 | 709 |
| `forward_recurrent` (the exact per-token scan) | 715 | 722 | 761 |

`project` is defined once, at `lib.rs:390`. `output` is defined once, at
`lib.rs:484`. **Four public entry points, one projection stage, one readout.**

### The consequence, stated as an invariant

> Everything above `project`, and everything inside `output`, is a **constant of
> every differential test in the tree**. A bug there moves the tensor-ops arm and
> the fused arm together, and the difference between them is exactly zero.

The test suite's own files know this in places and not in others. `bit_exact.rs`
was written precisely because the arm-vs-arm tests could not see `project` — it
is the only fixture that instantiates a whole `GatedDeltaNet2` and compares
anything to an outside reference. It is also red, and off by default. See §5.

---

## 3. The layers, what each is worth, and what each cannot do

Tier letters are ADR-0020's. "independence" means: over which part of the
computation does this layer's expected value vary independently of the code
under test?

| # | layer | file | independent over | tier | provenance |
|---|---|---|---|---|---|
| L1 | exact scan vs chunked form | `burn-kda/tests/cuda_gate.rs:132`, `fused_cuda.rs:20,40` | the **scan algorithm** (the chunked-WY rewrite). NOT the projections | (d) | two of our own formulations of Eq. 1 |
| L2 | batched ops vs loop ops | `burn-gdn2/tests/ops_batched_diff.rs` (9 cases) | the finite-Neumann rewrite, ragged tails, `TILE` routing | (d) | the untouched production arm |
| L3 | fused CUDA vs ops | `fused_chunk_verify.rs`, `fused_adjoint_vs_ops.rs`, `bench_cuda.rs:204` | the **kernels** (indexing, shared memory, the adjoint) | (d) | our own ops path |
| L4 | custom-node adjoint vs per-op autograd | `autodiff_chunk.rs:144` | the **analytic backward** | (d) | a second formulation of the same derivative |
| L5 | **central finite differences** | `burn-kda/tests/ops_grad_cuda.rs`, `ops_batched_grad_cuda.rs`, `ops_batched_autodiff.rs`, `autodiff_chunk.rs:200` | the **method of differentiation** — not a formulation, a different way to compute the same number | (d) | numerics, not code |
| L6 | host rank-5 references | `b5_seam_probe.rs` | rank-5 view semantics on ndarray | (d) | a scalar-index model, same file |
| L7 | seam counters / arm reachability | `cuda_gate.rs`, `autodiff_cuda_gate.rs` | *which arm ran* — not any number | (d) | counters |
| L8 | **paper transcription** | `bit_exact.rs`, `test_chunk.rs`, `ref_data.bin` | **`project`, `output`, the short conv, the recurrence itself** | **(c)** | our transcription of NVlabs `lit_gpt/gdn2.py` |
| L9 | IEEE-754 RNE | `lowp_bf16_cuda.rs` | bf16 storage semantics | (b) | the `half` crate, third-party |
| — | nothing | | | **(a)** | **no file in this tree has one** |

**L5 deserves its own line** because it is the only layer whose expected value
does not come from a second implementation. Finite differences are a different
*method*, not a different *program*: a wrong adjoint, a missing gradient, and a
gradient of a different function all fail it, and an arm-vs-arm comparison fails
none of those. This is the layer that caught the frozen attention arm, and
`burn-kda/tests/ops_grad_cuda.rs` says so in its own header. It is still tier
(d) — it validates the derivative of *our* function, not our function.

### What no layer can catch, by construction

- **L1–L7 are all (d).** Every one of them compares our code to our code. ADR-0020
  already said this about `burn-kda`'s suite ("`burn-kda`'s entire test suite is
  this") and it is equally true of L2 and L3.
- **L8 is the only layer that reaches `project` and `output`** — and it is red
  and off by default (§5).
- **L8 is tier (c), not (a).** A transcription error is symmetric: the same wrong
  reading of the paper on both sides of the comparison produces the same wrong
  number, and the test goes green. This is not a nitpick; it is the entire
  argument for preferring the authors' own bytes.

---

## 4. The honest cost of the whole enterprise

Until L8 is green, **everything above is a self-consistency check.** L1 says the
chunked rewrite matches the scan. L3 says the kernel matches the ops. L5 says the
adjoint matches differentiation of the same function. Each is a real and useful
test. None of them can tell you the model computes Gated DeltaNet. Chaining them
does not help: the chain's weakest link is not an arm, it is the specification,
and the specification appears exactly once, above the branch point, where no
differential test can see it.

**There is no tier-(a) layer in this tree.** The project's own audit says the
same at library scope: *"0 of 28 crates verify any numeric output against an
authors' own source code"* (`ADR-0020`, evidence `research/2026-09-27-oracle-audit.md`).
This file narrows that to the two crates that carry the headline.

---

## 5. The external anchor exists and is currently switched off

`tests/ref_data.bin` is committed, `tools/gen_reference.rs` regenerates it
byte-identically, and CI diffs the two — that is a real, well-built tier-(c)
anchor, and it is the best infrastructure in the tree.

It is **RED**, and the red reproduces exactly. Re-measured on this cell
2026-09-29 at `57f324b`, ndarray, `--release`, from `wt/oracle`:

```
cargo test --release -p burn-gdn2 --features binary-tests --test bit_exact --test test_chunk
  bit_exact::test_gdn2_1000_cases   1000 cases: max_diff = 1.38e-2, failures = 976/1000
  test_chunk::test_chunk_vs_reference
        chunk_size= 4: max_diff = 1.38e-2 FAIL     chunk_size=32: max_diff = 1.38e-2 FAIL
        chunk_size= 8: max_diff = 1.38e-2 FAIL     chunk_size=64: max_diff = 1.38e-2 FAIL
        chunk_size=16: max_diff = 1.38e-2 FAIL
        Chunk all sizes: max_diff = 1.38e-2, failures = 4880
```

Those are the same figures `vendor/burn-fused/TEST-AUDIT.md` FINDING 0 records
for 2026-09-27 at snapshot `1fab19e`, to three significant figures. The defect
is stable, not a flake, and the tolerance is not the problem: 5e-4 is 28× below
the observed max and the measured f32 transcription noise is 2e-6 to 2e-5.

**One new observation, and it narrows the search.** `max_diff` is *identical*
(1.38e-2) at chunk sizes 4, 8, 16, 32 and 64 — a 16× range. An error in the
chunked-WY rewrite, in the triangular inversion, or in the chunk-boundary state
carry would move when the chunk size moves. An error that is invariant under
re-chunking is **not in the chunking**: it is per-token or per-projection, which
is where `TEST-AUDIT.md` already localised it (the short conv's cross-token
taps, and `fused_recurrent_forward`'s `slice_dim(2, t..t+1)` over the *permuted*
`[B,HV,T,D]` views). A useful side effect: this **exonerates L2** — the ops
batched-vs-loop rewrite is not implicated. The one measurement that settles the
remaining fork is the one `TEST-AUDIT.md` names and nobody has run: print
`q`/`k`/`v` at `t=1` for case 1 (`T=3`) on both sides. Matching `q/k/v` puts the
divergence in that slice; differing `q/k/v` puts it in the generator's conv
tap indexing.

Two further consequences that belong in any decision about this crate:

1. **The only layer that can see `project` currently sees nothing**, because the
   feature is not default and the suite is red. Every run of
   `cargo test -p burn-gdn2` is blind to the whole projection stack.
2. The CI job that runs it is named *"1000 bit-exact cases vs the paper
   reference"* (`.github/workflows/fused-library.yml:129`) — a tier-(a) claim
   about a tier-(c) fixture, in the job name, which is the string people paste
   into issues. It is waived in `ORACLE-TIERS.tsv` with a note; the fix is one
   line and belongs to whoever owns the workflow.

### 5.1 The default CPU cell is blind to it, and that is measurable

`cargo test -p burn-gdn2 -p burn-kda --features autodiff` on this box, 2026-09-29,
runs `tests/bit_exact.rs` and reports:

```
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

**Two.** The 1000-case test is not among them. `bit_exact.rs` also contains
`bench_ndarray_short` and `bench_ndarray_single`, and those are the two that
run. `test_gdn2_1000_cases` is behind `#[cfg(feature = "binary-tests")]`, so
every ordinary `cargo test -p burn-gdn2` on this crate executes **zero** lines
of the only layer in the tree that can see `project`. A green test run on
`burn-gdn2` is not evidence about the projection stack, and reading it as such
is the most likely way this crate gets trusted by accident.

---

## 6. Blind spots: defect classes NO test in these two crates can catch

This is the deliverable. Derived by intersecting the structural invariant (§2)
with the per-file table in `ORACLE-TIERS.tsv`. Ordered by how much damage a
silent bug in the class would do.

| # | defect class | why nothing catches it | what it would cost to catch |
|---|---|---|---|
| **B1** | **`project` is the wrong function** — wrong L2-norm placement, `to_4d` permute transposed, GVA `repeat` on the wrong axis, `expand_v` head count, `allow_neg_eigval` factor, `a_log`/`dt_bias` broadcast shape, the `[1,h,1]→[1,1,kd]` reshape | above the branch point; **every** arm shares it. The only test that instantiates a whole module and compares to anything outside is L8, which is red and off by default | one green L8, or an f64 CPU reference (§8) |
| **B2** | **`output` is the wrong function** — the `1/nr` inside the SiLU, `o_norm` broadcast, the gate applied before vs after the norm, taking the readout from the state *before* the write (this exact class of bug was found and fixed in `gr.rs` on 2026-09-28, so the class is live) | same | same |
| **B3** | **`short_conv_1d` is wrong** — tap indexing, the cache carry, the padded left edge | shared by both arms; L8 is the only witness, and L8's own localisation says the short conv's cross-token taps are a prime suspect for its own failure | same |
| **B4** | **the recurrence is a misreading of the paper** — erase term without the `k k^T` projection, `beta` on the wrong side of the decay, `S^T q` vs `q^T S` | L1 and L2 both start from `project`'s output *and* encode the same reading of Eq. 1. A misreading is symmetric across arms exactly like a transcription error is across transcriptions | a **tier-(a)** anchor. See §8 |
| **B5** | **parameter initialisation** — `a_log=-3`, `dt_bias=1.0`, the Xavier gain `2^-2.5`, `min_decay`, the `o_norm` init | no test asserts an init value. `KdaConfig` fields are read by every test through the same `new()` | ~30 lines: assert the init constants against the paper's stated values. Cheapest item on this list |
| **B6** | **`b_v` is wrong** | `forward_recurrent` **discards** `_b_v` (`lib.rs:722`) and uses `b_k` for both the erase and the write term, so the scan reference is blind to the write gate by construction. They agree today only because `project` builds both from the same `beta_h` (`lib.rs:443-445`). Making `b_v` a genuinely different gate — which the K3 full-rank output gate wants — breaks the reference *silently* | make the scan take `b_v`, or add a case where `b_k != b_v` |
| **B7** | **the fused decode kernel** | `bench_cuda.rs:204` **is** a real assertion for `fused_step` — but it is a CUDA test, no GPU runner is registered, and this box was busy. It has never run in CI | a GPU window; the test already exists |
| **B8** | **`Autodiff<Cuda, BalancedCheckpointing>` numerics** | `alloc_probe.rs` is the only file with that type and all four of its tests are `#[ignore]`d. `autodiff_cuda_gate.rs` runs on it, but compares a fused path to an ops path — same (d) ceiling as everything else | an FD test on that backend, which is what `ops_grad_cuda.rs` already is for KDA — extend it to GDN2 |
| **B9** | **gradient coverage outside the probed coordinates** | `autodiff_chunk.rs:200` probes **one** coordinate per input, chosen as the argmax of the analytic gradient under test, at a 5% relative bar. `ops_batched_grad_cuda.rs` and `ops_grad_cuda.rs` are better (8 spread coordinates) but still 5%. A backward wrong at 99% of coordinates and right at 8 of them passes | raise the coordinate count; the bar is the harder half — see `TEST-AUDIT.md`'s note that 5% is loose |
| **B10** | **run-level quality** — that a *trained* attention arm improves BPB | nothing anywhere, and nothing in this file can produce it. Every layer above is about the trainer being correct, not the model being good | `VERIFICATION.md` layer 3. A different project |

**B1, B2, B3 and B4 are one class seen four ways: the specification.** B5, B6
are cheap. B9 is a known, named weakness of the strongest layer in the tree.
B10 is out of scope and is stated so nobody thinks this file closes it.

### 6.1 A red test that no CI job runs (measured on the CPU cell, 2026-09-29)

```
cargo test -p burn-gdn2 -p burn-kda --features autodiff --no-fail-fast
  autodiff_nested_balanced::the_op_declines_a_nested_graph_and_the_ops_path_carries_the_gradient
  FAILED at tests/autodiff_nested_balanced.rs:339
  "the op built a node over all-intermediate inputs: its output is a LEAF,
   which is the defect 8fa5d4c fixed"
  35 passed / 1 failed  (the whole CPU suite of both crates is 36 tests)
```

This is **pre-existing** — the file is untouched by this lane and was last
changed at `221232f`, "two more dead gates found on hardware". It asserts that
`chunk_wy_forward_autodiff_s::<NdArray, BalancedCheckpointing>` **declines**
all-intermediate inputs. On this cell it does not: it builds a node. So the
tracked-ness gate that `8fa5d4c` installed behaves differently on
`NdArray + BalancedCheckpointing` than on `CudaBare + BalancedCheckpointing`,
or has since changed.

It is ungated for a mechanical reason, not a decision. The file is
`#![cfg(feature = "autodiff")]`. CI's default CPU job
(`fused-library.yml:62`) runs `cargo test --workspace` with **no** autodiff
feature, so the file compiles out; and the feature-matrix job's autodiff step
for `burn-gdn2` (`:94`) is `cargo check`, not `cargo test`. **No job in
`fused-library.yml` executes this test on any backend.**

Why it belongs in this document: the assertion is that a *gate* refuses. A gate
that has quietly stopped refusing is precisely the defect class of §3.2 of
AGENTS.md (the dispatch gate that stayed dead for a year), and the one test
watching it is red and unwatched. I have not diagnosed it — that needs a read
of `OpsPrep::prepare` against the balanced-checkpointing node types, and a
decision about which behaviour is correct. **Do not "fix" it by inverting the
assertion**: the fixture is explicit that the decline is the intended behaviour
and that accepting these inputs is the known-bad path.

---

## 7. The enforcement

`tools/oracle_gate.py`, five rules, exit 1 on a real violation:

- **R1 COVERAGE** — every `tests/` and `examples/` file in the two crates has a
  row in `ORACLE-TIERS.tsv`, and so does any `src/`/`tools/`/`README` file that
  makes a fidelity claim. A new test cannot land without declaring what it is
  compared against.
- **R2 VOCABULARY** — a file whose tier is not `(a)` may not contain a positive
  "bit-for-bit"/"bit-exact" claim. Disclaimers pass; claims do not.
- **R3 PROVENANCE** — `(a)` needs a `github.com` URL in the file; `(b)`/`(c)`
  need an arXiv id or a named fixture+generator.
- **R4 STALE** — a row naming a file that no longer exists is a defect.
- **R5 FILENAME** — `bit_exact.rs` at tier `(c)` is a violation in its own right:
  the filename is what `cargo test --test bit_exact` prints and what gets quoted.

Wording fixed in this lane by the gate going red on it:
`burn-gdn2/src/lib.rs:49` (ADR-0020 listed this defect in the README and missed
the doc comment), `b5_seam_probe.rs:11`, and the two `burn-kda` `bitforbit`
docstrings. Three rows are waived with a written reason, because the files belong
to another lane: `bit_exact.rs`, `gen_reference.py`, and `lowp_bf16_cuda.rs`
(the last is the one *legitimate* use of the word outside tier (a) — its expected
value is `half::bf16::from_f32`, a third-party implementation of IEEE-754).

Run it: `python3 tools/oracle_gate.py`. It is not in CI yet; that is one line in
`fused-library.yml` and it belongs to whoever owns the workflow.

---

## 8. What a genuine external anchor would cost — recommendation

Three candidates, in the order I would do them.

### (1) An f64 CPU reference, literal from the paper. **DO THIS FIRST.**

A ~120-line NumPy script, no GPU, no dependencies beyond NumPy: transcribe
Eq. 1–6 of the paper (and `gdn2.py` for the projection shapes) at **float64**,
naive per-token loop, and commit the outputs for ~20 fixed cases. Compare every
arm against it with a **relative** bar.

Why this is the right first move:

- It is the only proposal that needs **no GPU**, and the GPU is the project's
  scarcest resource (ADR-0020's own note: every CUDA-executing test in the
  library has never run in CI).
- It closes **B1, B2, B3** and half of **B4** — the entire specification class,
  which is where all the risk is.
- The margin is enormous and therefore *falsifiable*, which is ADR-0020
  checklist item 3. A semantic error is **O(1) relative**; f32 reassociation
  over 8 chunk boundaries is **O(1e-6)**. A bar at 1e-3 separates them by three
  orders of magnitude, and the file can name the wrong answer it catches
  ("`b` broadcast on the wrong axis: 4x, not 1.001x").
- f64 at the top of the stack also removes a confound: today a discrepancy has
  two candidate causes — our maths, or f32 conditioning — and
  `TEST-AUDIT.md` FINDING 0 is stuck at exactly that fork.

Honest ceiling: it is still tier (b), and a shared *misreading of the paper*
survives it. Say so in the file. It is worth doing anyway, because the
alternative is tier (d).

### (2) The authors' own bytes. **WORTH ONE AFTERNOON, ONCE, IF THE ARTIFACT RUNS.**

`NVlabs/GatedDeltaNet-2` ships `lit_gpt/gdn2.py` and the Triton
`fused_recurrent_gdn2.py`. Running that on this box with fixed weights and
committing its activations is the only thing that makes the headline tier (a)
and converts the whole crate's vocabulary honestly.

Cost and risk, stated plainly: ~2–4 h; needs a GPU window (the box is busy);
`torch` + `triton` install and version pinning; and it may not run standalone
without the rest of `lit_gpt`. If it does not run within the afternoon, stop —
a partially reconstructed "reference" is worse than none, because it is tier
(b) wearing tier (a)'s clothes, which is the exact failure this file exists to
stop. ADR-0020 already names this as fix #1 at ~60 lines of Python; that
estimate is for replacing the *scan* inside the existing generator, which is a
weaker and cheaper thing than running their kernel, and I would do it only if
(2) fails.

### (3) A committed loss curve (VERIFICATION.md layers 2–3). **NOT NOW, AND NOT AN ORACLE.**

Two reasons. First, it is a *regression* gate, not a correctness anchor: it
compares the trainer to itself across commits, so it is tier (d) for the model
however many digits it pins. Second, it is **blocked**: `--seed` still leaves
409,043 differing parameter values between two identical runs (AGENTS.md §3.7),
so a fixture captured today would not be reproducible tomorrow. Fix the seed
first; the loss curve is a good project and a different one.

### What I would not do

Extend the gate to all 28 crates in this lane. Run against the whole repo
(excluding `vendor/cubecl-fix`, `target/`, `graphify-out/`) the gate's own
detector finds **67 positive fidelity-claim lines across 26 files**; six are in
`burn-kda`/`burn-gdn2` and three of those six are waived above. Fixing the rest
is a documentation project, not an oracle project, and it will collide with
four other lanes. `--scope` on the gate takes a crate list — widening it is one
argument once somebody owns the wording.
