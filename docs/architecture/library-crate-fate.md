# The 28 crates of `vendor/dormouse-fused`: what each one is for

Written 2026-09-28, snapshot `422414c`. This is the table the next person should
read instead of re-deriving reachability from six manifests.

**Question it answers.** `vendor/dormouse-fused/` is this project's own
implementation of published mechanisms and its claimed differentiator. Ten of its
28 crates are compiled into a training run. The other eighteen were not — not by
`dormouse-core`, not by `dormouse-train`, not by any test of theirs, not by any
A/B. This file records the fate of all 28, the line count behind each, and the
evidence for the eight deletions that landed.

**How reachability was measured** (reproduce it, it takes ten seconds):

```bash
# the crates a training build actually compiles - the root Cargo.lock is the
# honest answer, because vendor/dormouse-fused is `exclude`d from the root workspace
grep '^name = "burn-' Cargo.lock

# every path-dependency edge into a vendored crate, and whether it is normal,
# dev or build
grep -rn 'path *= *"\.\./crates/' --include=Cargo.toml vendor/dormouse-fused/
```

## The rule used to assign a fate

ADR-0002 is a mechanism that beats **its own removal** on held-out BPB, or it is
deleted; a tie deletes it. A crate that is not wired cannot have produced that
evidence, so **delete is the default**, and the exception is a *documented reason
to stay* — a line in this repo that says an experiment or a phase would need it,
with a `file:line`. Three reachability classes, and only the first counts as
being in the product:

| class | what it means | crates |
|---|---|---|
| **W** | a normal `[dependencies]` entry of a crate under `crates/dormouse-*` | 11 |
| **d** | a `[dev-dependencies]` entry of a **wired** crate — a test or an example, never a training run | 2 (`burn-sct`, `burn-rope`) |
| **b** | a normal `path =` of a bench package inside the library (`burn-fused-benches`, `benches/cpu_probe`) — built by the library's CI, linked by nothing that trains | 5 (`burn-attnres`, `burn-mhc`, `burn-rope`, `burn-sct`, `burn-swiglu`) |
| **–** | no incoming edge except the `burn-fused` facade, which **nothing in the product depends on either** | 12 |

The classes overlap: both `d` crates also have a `b` edge, so 11 + 5 + 12 = 28.

Classes `d` and `b` are **not** being in the product. A crate in them is built
by the library's own CI (`.github/workflows/fused-library.yml`, which runs from
inside `vendor/dormouse-fused` because the root workspace `exclude`s the fork) and
by nothing that trains.

`burn-fused` itself re-exports every crate and is referenced by zero manifests
in this repo. It is a distribution convenience, not part of the build; its
dependency on the other crates does not make them reachable.

## The table

`src` is `wc -l` over `src/**/*.rs` — the implementation. `t/b` is `tests/` and
`benches/`+`examples/`. Line counts at `422414c`.

| crate | src | t/b | reach | fate | one line |
|---|---:|---:|:---:|---|---|
| `burn-spectral` | 6329 | 2122 | **W** | keep | TSCT linears + MoE router; every `LinearLike` in the model |
| `burn-gdn2` | 4871 | 4085 | **W** | keep | the gated-delta kernel `burn-kda` is built on |
| `burn-sct` | 2739 | 458 | d, b | **DELETED 2026-10-02** | non-ternary original of what `burn-spectral` does in the product |
| `burn-attnres` | 2268 | 64 | b | REFERENCE | residual A/B arm named at `PLAN-minimal-core.md` §M2 |
| `burn-bitnet` | 2088 | 0 | **W** | keep | ternary weight quant called from `burn-spectral` |
| `burn-kda` | 1232 | 845 | **W** | keep | the attention arm |
| `burn-mhc` | 1171 | 0 | b | REFERENCE | residual A/B arm named at `PLAN-minimal-core.md` §M2 |
| `burn-rope` | 1131 | 0 | d, b | **WIRE** | the attention arm has **no positional encoding at all** today |
| `burn-muon-plus` | 1055 | 336 | **W** | keep | the default optimizer |
| `burn-dspark` | 867 | 0 | **W** | keep | the draft head (replaced MTP) |
| `burn-situ` | 823 | 0 | **W** | keep | the expert FFN's activation arm (`use_situ`, arXiv:2607.24653v2 Eq 12), off by default. Wired 2026-10-01; the row it replaced said "REFERENCE — the FFN uses `activation::silu`", which was true until then. **Its fused CUDA kernel is still unreachable** (default features only; `gpu-gate.sh:73-77` says it has never run in any job) |
| `burn-engram` | 687 | 0 | **W** | keep | the hashed n-gram memory arm |
| `burn-mor` | 530 | 0 | **W** | keep | the per-position router + top-k primitive |
| `burn-jepa` | 443 | 0 | **W** | keep | the EMA-teacher latent objective |
| `burn-ptrn` | 399 | 0 | – | REFERENCE | test-time scaling of a loop; its Q-head was deleted (ADR-0013) |
| `burn-es` | 345 | 0 | – | **DELETED 2026-10-02** | self-declared duplicate of `burn-eggroll` |
| `burn-eggroll` | 327 | 0 | – | REFERENCE | ditto; overlaps `burn-es` |
| `burn-parcae` | 317 | 0 | – | **WIRE** | spectral control of a loop — the one unwired crate aimed at an open problem |
| `burn-rmsnorm` | 296 | 0 | **W** | keep | the pre-head norm |
| `burn-swiglu` | 187 | 0 | b | REFERENCE | a line of `burn::activation::silu` as a crate |
| **deleted 2026-09-28** | | | | | |
| `burn-byteflow` | 1121 | 0 | – | **DELETE** | a whole competing byte-LM architecture |
| `burn-diffusionblocks` | 793 | 0 | – | **DELETE** | diffusion; `AGENTS.md:650` lists it under "Do NOT adopt … dormouse is AR" |
| `burn-fastblt` | 625 | 0 | – | **DELETE** | superseded — its finding is the Engram arm |
| `burn-mod` | 411 | 0 | – | **DELETE** | third depth mechanism, duplicate of the wired `burn-mor` |
| `burn-antihall` | 302 | 0 | – | **DELETE** | no plan anywhere names it |
| `burn-mtp` | 196 | 0 | – | **DELETE** | `AGENTS.md:715`: DSpark is "used instead of MTP" |
| `burn-ttt` | 125 | 0 | – | **DELETE** | inference-time training loss; carried a false claim (see below) |
| `burn-nope` | 102 | 0 | – | **DELETE** | a subtraction (no positional encoding), and dormant model-side |
| **total** | **31780 → 28105** | | | | **18 of 28 were not in the build; 13 382 of 31 780 src lines (42%)** |

## The eight deletions, and why each was airtight

All eight shared the same three properties, which is what made the case: **no
incoming path-dependency edge from any manifest in the repo except the
`burn-fused` facade; no test of any other crate naming them; and no line in
`docs/`, `configs/`, `docs/architecture/post-training.md` or the A/B queue asking for them.** The
"no reason to stay" half is the judgement, and it is per-crate below — but in
each case it is a judgement the repository has already made and written down,
not one I invented.

**The reachability test almost gave a false negative on one of them.**
`burn-fused/tests/facade.rs` had a `#[test]` that constructed
`burn_fused::burn_nope::nope_attention(...)` — so `burn-nope` had a passing test
that ran in CI, and reading the tree by "does a test exercise it" would have
called it alive. It is not alive: the facade is referenced by **zero** manifests
in this repo, so that test gates a crate nothing links. The block is removed and
the fact recorded where the test was.

| crate | the written decision that settles it |
|---|---|
| `burn-diffusionblocks` | `AGENTS.md:647-650`: "**Do NOT adopt** … diffusion-EBM (dormouse is AR)" — its own text calls diffusion "an anti-goal". It trains a denoiser on a noised target; dormouse's objective is next-byte CE. There is no configuration of this product in which it runs. |
| `burn-mtp` | `AGENTS.md:715`: DSpark is "used instead of MTP". A second draft head, un-A/B'd, on a path that already chose the other one. |
| `burn-fastblt` | Its own finding already ran and became a wired crate. `.bulba/memory.md`: "BLT ablation: hash n-grams work as INPUT features, n=3-4 first, diminishing returns past 300-500K rows" — that sentence *is* `burn-engram`. Keeping the port keeps the result, not the code. |
| `burn-mod` | Per-token depth routing is `burn-mor`'s job in this product, and `burn-mor` is wired (`crates/dormouse-core/Cargo.toml`). The repo's depth arms are `--rand-depth` and `use_mor`, both un-run, and ADR-0013 already deleted learned halting. A third router is the one ADR-0002 deletes. |
| `burn-byteflow` | A competing *whole model* (local encoder → coding-rate downsampling → global transformer → upsampling → decoder). Adopting it is not a mechanism swap, it is replacing `dormouse-core`. Nothing in the plan says that, so by default it does not happen. |
| `burn-antihall` | 302 lines of hallucination detection and neuron-level suppression. No plan, config, flag or doc in this repo mentions it. `docs/architecture/post-training.md` does not either. |
| `burn-nope` | It removes positional encoding. dormouse's attention arm has none of the machinery to add, remove or compare — wiring it would mean first building RoPE, which is `burn-rope`'s fate, and then A/B-ing an ablation of it. Nothing asks for that. |
| `burn-ttt` | A test-time *training loss* for long-context inference. The product is a trainer and a 256-byte generator; the inference-time adaptation loop it belongs to does not exist. It also carried the one claim ADR-0020 named as false ("Matched to the official implementation", §Relabelling) — the claim died with the crate, which is the cleanest possible resolution. |

**What went with them, stated rather than hidden: no test file was deleted
anywhere in this pass.** None of the eight had a `tests/`, `benches/` or
`examples/` directory at all (`other_rs = 0` in the table; verified with
`git ls-tree 422414c`), so each one is a pure `src/` deletion, and no surviving
crate's test file was touched. Nothing in `crates/` referenced any of the eight
(verified: `grep` over `crates/`, `configs/`, `scripts/`, `tools/` returns
nothing). The only mechanical consequence is the
library's own gate: `vendor/dormouse-fused/Cargo.toml` members,
`burn-fused/Cargo.toml` + `src/lib.rs` + `tests/facade.rs` (those three
regenerated by `tools/gen_facade.py`, which CI re-checks with `--check`; the
workspace `members` list above is hand-maintained), `vendor/dormouse-fused/Cargo.lock`, and the crate count in the CI job name, the
library README, `INTEGRATION.md` and `TEST-AUDIT.md`.

## The ten that were kept, and what each would take

Nothing here is a recommendation to delete. It is the honest statement of what
is missing, so the next reader does not assume a kept crate is a live one.

**Recommendation: WIRE — two crates.**

- **`burn-rope` (1131).** The model's attention arm has **no positional
  encoding**: `crates/dormouse-core/src/attention.rs` contains no RoPE call and
  no position term, while `AGENTS.md:586` says of the attention arm "keep
  RoPE in the attention arm — NoPE → endless generation after post-training". That is
  a documented gap with a crate already written for it. Call site: the KDA
  query/key projection in `attention.rs`, inside the dtype cast-to-fp32 rule
  (§2.2). Earning it: a 3-seed / 2k-step A/B on held-out BPB against the
  no-position control, per `docs/protocols/AB-PROTOCOL.md`. Not done here — the call site
  is in `crates/`, which three other worktrees hold.
- **`burn-parcae` (317).** Constrains the loop's state-retention spectral norm,
  which is aimed straight at the open instability in `AGENTS.md:256` ("the
  deeper overflow path is not isolated — a bisect showed `--no-kda` stays clean
  through deep overfit"). It is the only unwired crate whose mechanism the
  product's own status section says it needs. Call site: the per-iteration
  residual write in `loop_block.rs` (the ReZero/GR branch). Earning it: the
  overfit-stress protocol (`--stress`, `train/src/stress.rs`) plus BPB — a
  stability mechanism that does not change BPB still has to stop the NaN
  episodes, so this one needs a *different* gate from every other A/B in the
  queue, and saying so is the point of listing it.

**Recommendation: DELETE — one crate, blocked by a live dependency.**

- **`burn-sct` (2739, the largest crate in the library).** `dormouse-core` uses
  `burn_spectral::SpectralLinear` (`crates/dormouse-core/src/param.rs:6`); this
  is the non-ternary original of the same idea, and `burn-spectral`'s own
  benchmark table has an `sct8` row, i.e. the experiment comparing them ran.
  It stays only because `burn-spectral`'s `[dev-dependencies]` names it
  (`examples/tsct_diag.rs`) and `burn-fused-benches` has a normal `path =` on
  it. Dropping those two edges is ~10 lines; the crate is then 2739 lines of
  second-choice linear layer in a repo whose mission is FLOPs/byte. **This is
  the single largest deletion available in the library, and it is left standing
  because both edges live in an `examples/` and a `benches/` manifest, neither
  of which this pass owns** — cutting them is a ten-line change for whoever
  does. Related: `docs/architecture/bf16-plan.md:31`
  names `burn-sct` for a "TSCT bf16 retract", and that plan is against a
  capability `AGENTS.md:187` says this machine does not have at all.

**Kept as REFERENCE — seven crates.** Each is a faithful port with a named,
unbuilt use. Their READMEs now say in their first line that they are not in the
dormouse build, so nobody reads a port as a product arm.

| crate | the reason it stays |
|---|---|
| `burn-attnres`, `burn-mhc` | the residual-stream A/B. `PLAN-minimal-core.md` §M2: four rivals (ReZero, GR, mHC, AttnRes) and "A/B-ing the residual stream is currently a diff inside the model". Two of the four live here. Deleting them makes that A/B impossible to run without re-porting 3439 lines. |
| `burn-eggroll`, `burn-es` | `docs/architecture/post-training.md:40` "EGGROLL — exploration for controllers". That phase does not exist. They overlap each other; one should go when it starts. |
| `burn-ptrn` | test-time scaling aimed at exactly this architecture (a parameter-shared loop). Blocked on a decision, not on a port: it scores rollouts with a learned Q-head, and **ADR-0013 deleted that Q-head with PonderNet**; `AGENTS.md:645` says the selection rule must be re-specified first. |
| `burn-situ`, `burn-swiglu` | measured by `benches/cpu_probe`; `burn-swiglu` in particular is a dependency for a line of `burn::activation::silu`, i.e. a probe of something the product does not use. |

## Claims retracted in this pass

ADR-0020 is the rule: "verified" and "bit-for-bit" require naming the external
file the reference came from; where there is none, the honest wording is "no
external reference exists". Three crates used the vocabulary of kind (a) for a
kind (d) check — our fused path compared against our tensor path, two
formulations of the same derivative — and said "Verified":

| where (pre-edit line) | was | now |
|---|---|---|
| `burn-attnres/README.md:66` | "Verified fused backward == tensor-path backward (dh/dq < 1e-2)" | the check stands, relabelled kind (d), **"no external reference exists"**, "verified" withdrawn; plus: the fused seam was `NoCheckpointing`-only until `fdf9b20`, so on the trainer's backend it was a fallback before that. |
| `burn-rope/README.md:85` | "Verified fused backward == tensor-path backward (<1e-3)" | same, plus the same `861b9f7` note and "the performance table predates that". |
| `burn-situ/README.md:53` | "Verified fused backward == tensor-path backward on CUDA (<1e-3)" | same, plus the same `87c2e02` note. |
| `burn-sct/README.md:116` | "no host round-trip, verified 2e-7" | "agrees with the CPU Householder path to ~2e-7 — again our own two formulations, so kind (d)". |

**The "19,600×" was already retracted before this pass** (`63e01c3`, 2026-09-28)
in both places it appeared, `burn-mhc/README.md` and `burn-gdn2/README.md`, with
the mechanism named: the timed loop reads the clock with no device read inside
it, so it times **kernel enqueue** and the flush happens after `elapsed()`. The
README's own argument for why that is a defect and not a fast kernel: 4.2 M
elements × 40 normalization phases cannot execute in 1.65 µs on any GPU. Both
READMEs mark the rows `~~struck~~` with `RETRACTED`. What remains open and is *not* fixed here: the same
defect is listed for `burn-muon-plus/src/lib.rs:447` and
`fused_kernels.rs:243,302` (already named in its own README) and
`burn-attnres/fused_attnres.rs:1050` (which is the *correct* one, it flushes).

## How this was verified, and what it does not cover

`cargo check --workspace --all-targets` (the library CI's own command) is green
for all 20 surviving crates. `cargo test --workspace --no-fail-fast` on the CPU
backend: **228 passed, 5 failed, 0 compile errors** — and all five failures are
pre-existing, documented and deliberate, in crates this pass did not touch:

| the 5 red | where it is documented as red |
|---|---|
| `burn-gdn2` `nested_balanced_graph_matches_no_checkpointing`, `the_op_accepts_a_nested_balanced_graph` — "the op refused a Balanced graph: it is still strategy-blind" | `AGENTS.md` §3.2: the counter moved *after* the gate, "so the test is **deliberately red** until the backward gate lands" |
| `burn-spectral` `polar_retracts`, `to_inference_matches_trained_layer`, `to_inference_matches_per_column_layer` — panic `Tensor::require_grad requires autodiff` | `fused-library.yml`'s own comment ("KNOWN RED on its first run") and `TEST-AUDIT.md` finding 2 |

Two more things came out of running the gate, both pre-existing:

- **`burn-fused/tests/gpu_production_shape.rs:105` did not compile** at `422414c`
  — `k_dim as f64.powf(-0.5)` is "cast cannot be followed by a method call". That
  is the one test asserting the fused gated-delta kernels *launch* at the shape
  dormouse trains (`tools/gpu-gate.sh`'s whole point), and because
  `--all-targets` stops the build, it was masking whether anything else in the
  tree compiled. One pair of parentheses fixes it. That file is outside this
  pass's declared area; it is noted here because the gate being red at HEAD is
  exactly the failure mode `fused-library.yml` was written to end.
- **`burn-sct`'s `tests/cmp_reference.rs`** is a dev-dependency test of
  `burn-spectral`; it keeps the crate reachable, which is the whole reason the
  largest unwired crate in the library is still on disk.

Not covered: the three bench packages. `cargo metadata` proves they still
resolve (`burn-sct`, `burn-mhc`, `burn-rope`, `burn-situ`, `burn-swiglu`,
`burn-attnres` all kept, so no edge was removed), but compiling them needs
`--features cuda` on the CUDA backend, which this pass did not spend the shared
build on. And CI has never run this library on a GPU at all
(`fused-library.yml` has no `cuda-tests` job, on purpose, with the reasoning in
the file), so no CUDA test was executed before or after this change.

## What I could not determine

- **Whether any of the ten kept unwired crates is a good port.** They are kept
  because the repo names a use for them, not because anything checked the code.
  ADR-0020's oracle audit is still open for all of them, and it is the first
  thing to do to any of them that gets wired.
- **Whether `burn-rope` would win its A/B.** Nobody can know before running it;
  the recommendation is that it is the most likely of the eighteen to earn its
  place, not that it will.
- **A stale comment in the product, not fixed here because `crates/` is not
  mine**: `crates/dormouse-core/src/param.rs:1` says "TSCT linear via **burn-sct**
  SpectralLinear" and line 6 is `use burn_spectral::SpectralLinear`. The comment
  names the crate this file recommends deleting. That is a `file:line` for
  whoever owns `crates/dormouse-core`.
- **Whether the product's own build is green.** Not re-run, deliberately: the
  root `Cargo.lock` names exactly the 10 wired crates and contains no reference
  to any of the eight deleted, no file under `crates/` changed, and no
  `crates/*/Cargo.toml` was touched — so a root build could not report anything
  this pass did. If it is red, it was red before.

