How to know a change did not break the model

There is no end-to-end golden yet. This file is the specification for one, and
the list of what is currently unverified — because a rule that is not written
down is not followed, and "we checked it once by eye" is how 25858 ms/step
became a number three people believed.

> **For the fused library, read `docs/protocols/ORACLE.md` first.** It is the companion to
> this file and answers a question this one does not: for `burn-kda` and
> `burn-gdn2`, *which defect class each existing test can and cannot catch*, and
> which classes **no** test in those crates can catch. The short version, which
> changes how the tables below should be read: the tensor-ops path and the fused
> path share one `project` and one `output`, so a bug above the branch point
> moves both arms together and every arm-vs-arm differential in the tree sees
> **zero** difference. The only layer that can see that region is the
> `ref_data.bin` transcription, and it is currently red and behind a
> non-default feature. The machine-readable table is
> `docs/protocols/ORACLE-TIERS.tsv`; `tools/oracle_gate.py` exits 1 on a claim of
> bit-exactness against an arm-vs-arm comparison.

## The rule

**Every optimisation must preserve the model's output, bit for bit or within a
stated tolerance, against a fixture captured from the pre-change code.** A speed
result without that comparison is not a speed result; it is a different model
that happens to run faster.

## What already exists, and is the pattern to follow

`vendor/burn-fused/crates/burn-gdn2/` already does this properly and has done
since before this session:

- `tests/ref_data.bin` — 7 MB of committed reference activations
- `tools/gen_reference.rs` — regenerates it, and is the only thing that may
- `tests/bit_exact.rs` — compares the kernel against the file

Copy that shape. Do not invent a second convention.

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

The trap to avoid: `presets` and `--seed` do not make a run reproducible, because
`--seed` is only the JEPA mask and the data order comes from the corpus. So the
fixture has to be a small committed corpus, not the real one — a directory of a
few KB in `crates/dormouse-train/tests/fixtures/`, which also makes the test
independent of the mounted drive (AGENTS.md 2.6).

### Layer 3 — the run, end to end

**To be written, and this is the one that would have caught tonight.** 200 steps
on the fixture corpus, pure CE, one seed, compared against a committed loss
curve at a stated tolerance. It is the only layer where "the attention arm
trains" and "the attention arm helps" are different questions: layers 1 and 2
pass today, and we still do not know whether a trained attention arm improves
BPB, because no run with a gradient-carrying arm has ever completed a
meaningful number of steps.

## Currently unverified, in one place

So that "is it correct?" has an answer that is not a feeling:

| claim | status |
|---|---|
attention arm receives a gradient | VERIFIED, finite differences, ops_grad_cuda |
fused adjoint numerically correct | VERIFIED, all grads <= 5.5e-7, bar tightened not loosened |
batched ops == loop ops | VERIFIED, 9.1e-6 worst case, gradients identical |
library compiles in every target | VERIFIED, `check --workspace --all-targets`, 0 errors |
product test gate | VERIFIED, 98 passed / 0 failed |
eval reads the memory arm | VERIFIED on hardware, `engram=4/4` |
best checkpoint, 3 silent-data-loss fixes | VERIFIED, 7 tests each falsified red |
optimizer routing, live vs declared | VERIFIED, red-on-revert proof |
decode path == training path | VERIFIED, decode_seam to <1e-6, decode_wiring |
step time / throughput | MEASURED once, in `benches/history.tsv`; two earlier figures retracted |
**a trained attention arm improves quality** | **NOT VERIFIED — no run has ever completed this** |
fused RMSNorm | NEVER RUNS, `norm=0/N` on the trainer's backend |
f16 patch speedup | NEVER MEASURED on hardware |
`--eval-depths` window | KNOWN WRONG, reads a different window than the `bpb` above it |
`--dspark-k 0` | KNOWN WRONG, becomes k=1 at the default, and kills JEPA at `mor_bce_weight=0` |
DSpark one-position shift | KNOWN, documented, deliberately unfixed — needs its own A/B |
18 library crates unwired | KNOWN, fates recorded in docs/architecture/library-crate-fate.md |
zero A/B verdicts in the project | TRUE, and every one is listed in AGENTS.md 3.2 |

## Two rules that follow from the table

1. **A number without its protocol is not a number.** Every row above that says
   MEASURED names its instrument in `benches/history.tsv`. When a measurement
   is retracted it is struck, not restated with a new value, because the
   conditions that produced the wrong one are not reproducible.
2. **The model has not been shown to learn.** The best valid held-out number in
   the archive is 4.997 BPB against a 5-gram bar of 2.911–2.588 depending on the
   fit, and it was produced by a run with no attention arm at all. Everything
   above is about the trainer being correct, and none of it is about the model
   being good. Those are different projects and only the second one is the
   point.
