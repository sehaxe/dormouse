# CI quality gates — what landed, what it measured, what is still red

**Date:** 2026-10-01. **Toolchain for every local reading below:** rustc /
rustfmt / clippy **1.98.1**, cargo-deny **0.20.2**, cargo-machete **0.9.2**,
actionlint **1.7.12**, LLVM 22.1.8. **Commits:** `1f9f4b0`..`6725563` on
`main`. Every number here was produced by the command printed next to it, or
read out of a CI run named by its id; none is estimated.

This is the findings file for six jobs added to `.github/workflows/ci.yml`. It
exists because two of them are RED ON LANDING and a red gate with no number
attached is the failure mode AGENTS.md §1.1 calls a defect — so the numbers are
here, and the fixes are named.

**CLOSED IN PART, 2026-10-02 (wt/ci-final, rebased onto `1717ac3` — the
baseline numbers below were measured before 26 commits of landed lanes, so the
fresh counts were smaller):**

- `unused-deps`: **closed** (machete 10 → 0: eight deletions, two reasoned
  facade entries; `continue-on-error` dropped). §4.
- `lint` (clippy): **closed** (green under `-D warnings`; the three
  named exceptions carry `#[allow]` + reason). §2.
- `lint` (fmt): **closed except two files** — `cargo fmt` applied to all 52
  files of the four crates except `train/src/lib.rs` (129 hunks) and
  `core/src/routing.rs` (8), both held out because lanes hold uncommitted
  edits in them; `continue-on-error` stays until those land. §1.
- `fused-library / ndarray`: **closed** — the three burn-spectral fixtures
  moved to `Autodiff<NdArray>` and now assert instead of panicking in setup
  (TEST-AUDIT.md finding 2, CLOSED line); vendor formatted (280 hunks) and
  green under clippy `-D warnings`; the job's KNOWN RED note removed.

---

## The scoreboard

| job | first-run result | what the number is |
|---|---|---|
| `lint` (fmt) | **red, soft** | **629 hunks across 54 files**, per-crate below |
| `lint` (clippy) | **red, soft** | **114 diagnostics at 27 sites**, 18 clippy lints + 4 rustc lints |
| `supply-chain` | **green** | 0 advisories, 0 license failures, 0 source failures; 2 advisories FIXED to get there |
| `unused-deps` | **red, soft** | **10 findings in `crates/`**, 4 of them facade false positives |
| `coverage` | **green** | **7003 / 8158 lines = 85.84 %**; no threshold yet, by design |
| `docs` (rustdoc) | **green** | 57 s, exit 0, 7 `missing documentation` warnings, nothing denied |
| `workflow-lint` | **green after one red** | red on run 1 because of `pyflakes`; see §6 |

Both "soft" jobs carry `continue-on-error: true` and the count in the **job
name**, so a reader of the Actions page sees the debt without opening a log.
That is the ADR-0019 COUNTED mark: the degradation happens and something says
so out loud.

### Two things that were wrong in the first version, found by reading the runs

Neither was visible locally and both are the class of defect this repo keeps
paying for, so they are recorded here rather than only in the commits.

**1. `continue-on-error` on a STEP reports the step as `success`.** Run
36866436388: both lint steps were red in the log — the rustfmt step's log holds
629 `Diff in` hunks — and the API returned `conclusion: success` for both. A
green tick beside a red gate is worse than no gate, because it is read as a
pass. Moved to the **job**, which is where `cuda-compile` already had it: the
steps keep their own `failure` and the run is not blocked. Verified on run
36868907213.

**2. A failing step SKIPS every step after it.** Run 36868694110, after the fix
above made the rustfmt failure visible for the first time:

```
5 rustfmt -> failure
6 clippy  -> skipped
```

So half the job reported nothing. That is `fused-library.yml`'s failure mode
arriving by another route — its `ndarray check+test` job dies at step 6 for
exactly this reason — and a gate that does not run is indistinguishable from a
gate that passes. Fixed with `if: ${{ !cancelled() }}` on the clippy step, which
keeps the "still refuse to run on a cancelled job" property that `always()`
would have traded away. Verified on run 36868907213, where both steps report
`failure`.

---

## 1. `cargo fmt` — 629 hunks across 54 files

```
cargo fmt -p dormouse-core -p dormouse-data -p dormouse-train -p dormouse-cli -- --check
```

| crate | hunks |
|---|---|
| `dormouse-core` | 332 |
| `dormouse-train` | 231 |
| `dormouse-data` | 43 |
| `dormouse-cli` | 23 |
| **total** | **629** |

629 is what the job runs and what its comment and job name say. 651 is the same
measurement over all six workspace members: the extra 22 hunks are in
`backend-parity` and `cublas-poc`, which this workflow's `cpu` job does not
build either. Both numbers are here because a gate that quotes a figure it did
not measure is the wrong kind of wrong, and the first version of the job comment
quoted 651.

**This is not a formatting disagreement, it is an unformatted tree.** Measured:
there is no `rustfmt.toml` or `.rustfmt.toml` anywhere in the repo, and the
files are hand-wrapped at ~100 columns by hand. Two configs were tried and
neither closes it: `max_width = 120` gives 566 hunks, `use_small_heuristics =
"Max"` gives 488, both together 563. So this is 629 hunks of genuinely
rustfmt-nonconforming code, not one config choice.

**One thing deliberately NOT in this number.** `cargo fmt --all` walks local
path-based dependencies, so it also reaches **279 hunks inside
`vendor/dormouse-fused`** — a separate workspace, separately gated by
`fused-library.yml`, and `.editorconfig` deliberately withdraws rustfmt's
newline/whitespace rules under `vendor/**` so a diff against upstream stays
reviewable. Hence `-p <four crates>` in the job, and hence the vendor tree is
out of scope here.

**To close it:** one `cargo fmt -p …` commit and delete `continue-on-error`
from that step. That is the whole fix; it is mechanical and it belongs in its
own commit, not folded into a CI change.

---

## 2. `cargo clippy` — 114 diagnostics at 27 sites

```
cargo clippy -p dormouse-core -p dormouse-data -p dormouse-train -p dormouse-cli --all-targets -- -D warnings
```

Enumerated with `--cap-lints warn` (so the run completes and every diagnostic
is printed) and counted by `(file, line)`. 47 distinct messages, 119 emitted
occurrences, **114 unique `(file, line)` sites** — the difference is
`--all-targets` compiling each crate as both lib and lib-test, which reports
the same site twice. The job name says 114; that is the deduplicated count.

By lint, deduplicated:

| count | lint |
|---|---|
| 18 | `clippy::manual_is_multiple_of` |
| 15 | `clippy::doc_lazy_continuation` |
| 10 | `unused_mut` |
| 9 | `clippy::field_reassign_with_default` |
| 8 | `clippy::let_unit_value` |
| 5 | `clippy::chunks_exact_to_as_chunks` |
| 4 | `clippy::useless_conversion` |
| 4 | `clippy::cloned_ref_to_slice_refs` |
| 3 each | `clippy::manual_range_contains`, `clippy::too_many_arguments`, `unused_variables` |
| 2 each | `clippy::needless_borrow`, `clippy::unnecessary_cast`, `clippy::items_after_test_module`, `clippy::empty_line_after_doc_comments`, `clippy::mem_replace_with_default`, `unused_imports` |
| 1 each | `clippy::ptr_arg`, `clippy::needless_range_loop`, `clippy::implicit_saturating_sub`, `clippy::double_ended_iterator_last`, `clippy::missing_const_for_thread_local`, `clippy::mistyped_literal_suffixes`, `clippy::bool_comparison`, `clippy::manual_contains`, `clippy::map_clone`, `clippy::redundant_closure`, `clippy::extra_unused_type_parameters`, `unused_parens`, `unused_assignments`, `dead_code`, `non_snake_case` |

### The three that are worth reading before fixing

**`clippy::mistyped_literal_suffixes` — `crates/dormouse-train/src/offload.rs:36`.**
This is a clippy **error** under `-D warnings` and it is the reason the job
cannot be strict. The literal is `const LAYOUT_MAGIC: u32 = 0x4D_47_4E_32;` —
clippy reads the `32` at the end of a hex literal as a mistyped `i32` suffix
and wants `0x004D_474E_i32`. The code is correct; the lint is a false positive
on a magic constant, so this one needs an `#[allow]` **with a reason**, not a
rewrite.

**`unused_assignments` / `unused_variables` on `nan_reads`, `tsct_bef`,
`tsct_aft` — `crates/dormouse-train/src/lib.rs:1118-1328`.** These are in the
stress-monitor path, and §3.3 of AGENTS.md records that `StressMonitor` state
is lost on resume. A value assigned and never read next to counters named
`nan_reads` is worth a human glance before an autofix deletes the assignment:
it may be the same defect wearing a lint's clothes. **Do not `clippy --fix`
this file blind.**

**`clippy::non_snake_case` — `crates/dormouse-core/src/dspark_oracle.rs:296`,
`the_confidence_head_IS_markov_conditioned`.** Deliberate: the name asserts the
ADR-0020 claim the function exists to prove. Needs `#[allow(non_snake_case)]`
with that reason, not a rename.

### Full list

Sites marked `?` are a continuation of the diagnostic on the line above (clippy
reports several lines under one lint name and the note carries no second
`#[warn(...)]`).

<details>
<summary>114 sites, by file (click to expand)</summary>

**114 diagnostics** at 27 sites.


**crates/dormouse-cli/src/bin/generate.rs**
| 51 | `clippy::manual_range_contains` | manual `!RangeInclusive::contains` implementation |

**crates/dormouse-cli/src/bin/serve.rs**
| 119 | `clippy::manual_range_contains` | manual `!RangeInclusive::contains` implementation |

**crates/dormouse-core/src/aux.rs**
| 161 | `clippy::needless_range_loop` | the loop variable `i` is used to index `out` |
| 181 | `clippy::implicit_saturating_sub` | implicitly performing saturating subtraction |

**crates/dormouse-core/src/config/override.rs**
| 26 | `clippy::double_ended_iterator_last` | called `Iterator::last` on a `DoubleEndedIterator`; this will needlessly iterate the entire iterator |

**crates/dormouse-core/src/config/schema.rs**
| 159 | `clippy::doc_lazy_continuation` | doc list item without indentation |
| 160 | `?` | doc list item without indentation |
| 161 | `?` | doc list item without indentation |
| 162 | `?` | doc list item without indentation |
| 300 | `clippy::items_after_test_module` | items after a test module |

**crates/dormouse-core/src/config/validation.rs**
| 3 | `unused_imports` | unused import: `super::schema::ActQuant` |
| 108 | `clippy::manual_is_multiple_of` | manual implementation of `.is_multiple_of()` |
| 114 | `?` | manual implementation of `.is_multiple_of()` |
| 174 | `?` | manual implementation of `.is_multiple_of()` |

**crates/dormouse-core/src/dspark_oracle.rs**
| 101 | `clippy::chunks_exact_to_as_chunks` | using `chunks_exact` with a constant chunk size |
| 178 | `?` | using `chunks_exact` with a constant chunk size |
| 296 | `non_snake_case` | function `the_confidence_head_IS_markov_conditioned` should have a snake case name |
| 482 | `?` | using `chunks_exact` with a constant chunk size |

**crates/dormouse-core/src/gr.rs**
| 105 | `?` | the loop variable `i` is used to index `normed` |
| 141 | `?` | the loop variable `i` is used to index `branches` |

**crates/dormouse-core/src/loop_block.rs**
| 299 | `clippy::useless_conversion` | useless conversion to the same type: `burn::Tensor<2>` |
| 300 | `?` | useless conversion to the same type: `burn::Tensor<2>` |
| 826 | `?` | the loop variable `n` is used to index `step_outs` |
| 966 | `clippy::field_reassign_with_default` | field assignment outside of initializer for an instance created with Default::default() |
| 976 | `unused_mut` | variable does not need to be mutable |
| 977 | `?` | variable does not need to be mutable |
| 1002 | `?` | field assignment outside of initializer for an instance created with Default::default() |
| 1039 | `?` | field assignment outside of initializer for an instance created with Default::default() |
| 1062 | `?` | field assignment outside of initializer for an instance created with Default::default() |
| 1090 | `?` | field assignment outside of initializer for an instance created with Default::default() |
| 1112 | `?` | field assignment outside of initializer for an instance created with Default::default() |
| 1126 | `?` | variable does not need to be mutable |
| 1133 | `?` | variable does not need to be mutable |
| 1153 | `?` | field assignment outside of initializer for an instance created with Default::default() |
| 1181 | `?` | field assignment outside of initializer for an instance created with Default::default() |
| 2011 | `?` | variable does not need to be mutable |
| 2178 | `?` | field assignment outside of initializer for an instance created with Default::default() |

**crates/dormouse-core/src/mixture_probe.rs**
| 333 | `clippy::cloned_ref_to_slice_refs` | unnecessary use of `clone` to create a slice from a reference |
| 336 | `?` | unnecessary use of `clone` to create a slice from a reference |

**crates/dormouse-core/src/model.rs**
| 52 | `?` | doc list item without indentation |
| 53 | `?` | doc list item without indentation |
| 273 | `clippy::too_many_arguments` | this function has too many arguments (8/7) |

**crates/dormouse-core/src/moe.rs**
| 222 | `unused_parens` | unnecessary parentheses around closure body |
| 533 | `clippy::redundant_closure` | redundant closure |
| 551 | `unused_variables` | unused variable: `off_spread` |

**crates/dormouse-core/src/probe.rs**
| 112 | `clippy::missing_const_for_thread_local` | initializer for `thread_local` value can be made `const` |

**crates/dormouse-core/src/routing.rs**
| 485 | `?` | variable does not need to be mutable |

**crates/dormouse-core/tests/model_seam.rs**
| 376 | `unused_variables` | unused variable: `fwd` |
| 519 | `clippy::map_clone` | you are using an explicit closure for copying elements |
| 778 | `clippy::manual_contains` | using `contains()` instead of `iter().any()` is more efficient |

**crates/dormouse-core/tests/moe_grad_seam.rs**
| 29 | `clippy::empty_line_after_doc_comments` | empty line after doc comment |
| 89 | `unused_mut` | variable does not need to be mutable |

**crates/dormouse-core/tests/moe_step1_gate.rs**
| 25 | `clippy::empty_line_after_doc_comments` | empty line after doc comment |

**crates/dormouse-core/tests/preset_exec.rs**
| 572 | `clippy::bool_comparison` | equality checks against false can be replaced by a negation |

**crates/dormouse-data/src/bin/filter.rs**
| 378 | `clippy::unnecessary_cast` | casting to the same type is unnecessary (`i32` -> `i32`) |
| 895 | `clippy::too_many_arguments` | this function has too many arguments (9/7) |
| 1454 | `clippy::needless_borrow` | this expression creates a reference which is immediately dereferenced by the compiler |
| 1525 | `clippy::let_unit_value` | this let-binding has unit value |
| 1527 | `?` | this let-binding has unit value |
| 1551 | `?` | this let-binding has unit value |
| 1553 | `?` | this let-binding has unit value |
| 1570 | `?` | this let-binding has unit value |
| 1573 | `?` | this let-binding has unit value |
| 1603 | `?` | this let-binding has unit value |
| 1605 | `?` | this let-binding has unit value |

**crates/dormouse-data/src/bin/shard.rs**
| 51 | `unused_mut` | variable does not need to be mutable |
| 55 | `clippy::needless_borrow` | this expression creates a reference which is immediately dereferenced by the compiler |
| 85 | `clippy::manual_is_multiple_of` | manual implementation of `.is_multiple_of()` |

**crates/dormouse-data/src/lib.rs**
| 360 | `clippy::ptr_arg` | writing `&mut Vec` instead of `&mut [_]` involves a new object where a slice will do |
| 770 | `unused_mut` | variable does not need to be mutable |
| 807 | `clippy::manual_range_contains` | manual `Range::contains` implementation |

**crates/dormouse-train/examples/export_divergence.rs**
| 149 | `clippy::ptr_arg` | writing `&PathBuf` instead of `&Path` involves a new object where a slice will do |

**crates/dormouse-train/src/jepa_targets.rs**
| 104 | `dead_code` | method `len` is never used |
| 130 | `clippy::chunks_exact_to_as_chunks` | using `chunks_exact` with a constant chunk size |

**crates/dormouse-train/src/lib.rs**
| 478 | `clippy::extra_unused_type_parameters` | type parameter `B` goes unused in function definition |
| 565 | `clippy::too_many_arguments` | this function has too many arguments (8/7) |
| 884 | `clippy::unnecessary_cast` | casting to the same type is unnecessary (`u64` -> `u64`) |
| 1118 | `unused_assignments` | value assigned to `tsct_bef` is never read |
| 1119 | `?` | value assigned to `tsct_aft` is never read |
| 1136 | `unused_variables` | variable `nan_reads` is assigned to, but never used |
| 1149 | `?` | manual implementation of `.is_multiple_of()` |
| 1158 | `clippy::mem_replace_with_default` | replacing a value of type `T` with `T::default()` |
| 1237 | `?` | manual implementation of `.is_multiple_of()` |
| 1241 | `?` | manual implementation of `.is_multiple_of()` |
| 1256 | `?` | manual implementation of `.is_multiple_of()` |
| 1257 | `?` | manual implementation of `.is_multiple_of()` |
| 1309 | `?` | value assigned to `nan_reads` is never read |
| 1328 | `?` | value assigned to `nan_reads` is never read |
| 1355 | `?` | manual implementation of `.is_multiple_of()` |
| 1375 | `?` | manual implementation of `.is_multiple_of()` |
| 1408 | `?` | manual implementation of `.is_multiple_of()` |
| 1452 | `?` | manual implementation of `.is_multiple_of()` |
| 1474 | `?` | manual implementation of `.is_multiple_of()` |
| 1746 | `?` | manual implementation of `.is_multiple_of()` |
| 1759 | `?` | manual implementation of `.is_multiple_of()` |
| 1951 | `?` | doc list item without indentation |
| 1952 | `?` | doc list item without indentation |
| 1953 | `?` | doc list item without indentation |
| 1954 | `?` | doc list item without indentation |
| 1955 | `?` | doc list item without indentation |
| 1956 | `?` | doc list item without indentation |
| 1957 | `?` | doc list item without indentation |
| 2248 | `unused_mut` | variable does not need to be mutable |
| 2270 | `?` | useless conversion to the same type: `burn::Tensor<2>` |

**crates/dormouse-train/src/offload.rs**
| 36 | `clippy::mistyped_literal_suffixes` | mistyped literal suffix |
| 145 | `clippy::manual_is_multiple_of` | manual implementation of `.is_multiple_of()` |
| 158 | `?` | method `to_bytes` is never used |
| 200 | `?` | using `chunks_exact` with a constant chunk size |
| 262 | `clippy::items_after_test_module` | items after a test module |
| 393 | `clippy::useless_conversion` | useless conversion to the same type: `burn::Tensor<2>` |

**crates/dormouse-train/src/optim.rs**
| 553 | `unused_imports` | unused import: `burn::optim::Optimizer` |

**crates/dormouse-train/src/stress.rs**
| 78 | `?` | manual implementation of `.is_multiple_of()` |
| 142 | `clippy::doc_lazy_continuation` | doc list item without indentation |
| 143 | `?` | doc list item without indentation |

</details>

### To close it

1. `cargo clippy --fix` for the mechanical 90-ish (everything except the three
   above).
2. Three `#[allow]` with reasons, per the three notes.
3. Delete `continue-on-error`.

---

## 3. `cargo deny` — green, and it found two real advisories

```
cargo deny --log-level warn --manifest-path ./Cargo.toml --all-features check
```

| check | first run | after the two bumps below |
|---|---|---|
| advisories | **FAILED** (2) | ok |
| bans | ok (24 warnings) | ok |
| licenses | ok | ok |
| sources | ok | ok |

### The two advisories, and the four lines that fixed them

| id | crate | what | how it got here |
|---|---|---|---|
| **RUSTSEC-2026-0285** | `rustls 0.23.43` | accepted TLS 1.3 handshake messages sent at the wrong encryption level when they followed a key-changing message in the same record. The transcript is still authenticated, so it is not a MITM — but a peer could send in plaintext what should have been encrypted, without rustls rejecting the connection. | `ureq` → `tracel-llvm-bundler` → **build-dependency** of `cubecl-llvm`. Build-time only. |
| **yanked** | `chacha20 0.10.1` | yanked version, no patched release at the time | `rand 0.10.2` → `burn-backend`. Transitive. |

Both are `cargo update --precise` moves — `rustls` 0.23.43 → 0.23.45,
`chacha20` 0.10.1 → 0.10.2 — and the whole diff is four lines of `Cargo.lock`.
**Neither is a decision about our code**, which is why the fix was made rather
than an `ignore` entry: `deny.toml`'s `[advisories] ignore` is still `[]`, and
every entry added to it must carry a reason.

**One methodological note worth keeping.** `cargo update -p X --precise Y` on
this tree also re-resolves `windows-sys` and `getrandom` and drags five
unrelated crates along. The lock was instead edited directly (the two
`(name, version, checksum)` blocks) and then proved with
`cargo metadata --locked --offline`, which resolves without touching the
network. A lockfile is a resolved graph, not an opinion, and this is the way to
change two entries in it without rewriting nine.

### The 24 `bans` warnings, warn-only with a reason

- **`multiple-versions`: 43 crates carry more than one version** in the lock
  (`cargo metadata` count, not cargo-deny's). The notable clusters: `hashbrown`
  0.13/0.15/0.16/0.17, `rand` 0.8/0.9/0.10, `thiserror` 1/2, `toml` 0.8/1.1,
  `syn` 2/3, `windows-sys` 0.52/0.61. Most are burn/cubecl's own fan-out and
  are not ours to collapse. Warn-only until someone triages which are.
- **`wildcards`: 22 findings, and every one is a `path =` dependency with no
  `version` key.** cargo-deny reads that as a wildcard, and
  `allow-wildcard-paths = true` does **not** rescue it in a crate that could be
  published (it only applies to `publish = false` crates). Five crates here
  could be published — `dormouse-cli`, `dormouse-core`, `dormouse-train`, and
  `burn-kda` / `burn-spectral` in the vendored library. The closing recipe is
  `publish = false` in five `[package]` blocks, which is a statement about this
  repo's release policy and lands in files other lanes are editing, so it is the
  owner's call. Warn-only until then, with the recipe in `deny.toml`.

### The license set is measured, not remembered

16 allowed identifiers, chosen from `cargo metadata --all-features` over all
756 packages as a count of distinct SPDX operands: **MIT 659 · Apache-2.0 527 ·
Unicode-3.0 25 · Zlib 23 · BSD-3-Clause 15 · Unlicense 11 ·
Apache-2.0 WITH LLVM-exception 8 · ISC 6 · BSD-2-Clause 5 · MPL-2.0 3 ·
CC0-1.0 3 · LGPL-2.1-or-later 2 · BSL-1.0 2 · CDLA-Permissive-2.0 2 · 0BSD 1 ·
NCSA 1**. MPL-2.0 is weak-copyleft (file-level) and arrives only through
`colored` / `option-ext` / `buildid` — build-time or CLI formatting, never in a
shipped artefact.

One package, `cfg_block 0.1.1`, carries **no license field at all**. It is
reachable only on a non-linux target, so the `x86_64-unknown-linux-gnu` graph
does not contain it and the check passes. If a target is ever added that pulls
it in, cargo-deny fails it loudly — which is the point of naming it here.

---

## 4. `cargo machete` — 10 findings, 4 of them the tool's blind spot

```
cargo machete crates/
```

| crate | reported | real? |
|---|---|---|
| `dormouse-cli` | `burn`, `dormouse-data`, `serde_json` | **yes, all three** |
| `dormouse-core` | `burn-nn`, `burn-tensor`, `rand` | 2 false, **1 real** (`rand`) |
| `dormouse-train` | `burn-tensor`, `rand` | 1 false, **1 real** (`rand`) |
| `cublas-poc` | `burn-cuda` | false |
| `backend-parity` | `burn-tensor` | false |

**The false positives are one bug, four times.** `burn-tensor`, `burn-nn` and
`burn-cuda` are declared as direct dependencies but reached through the `burn`
facade re-export: the source says `use burn::tensor::Tensor`,
`use burn::nn::{Linear, LinearConfig}`, `use burn::tensor::{Device, …}`, and
`burn_tensor` appears **zero** times in any of these crates. cargo-machete
greps for the crate name and cannot see through a re-export. The right fix is
`[package.metadata.cargo-machete] ignored = [...]` **with that reason**, not a
deletion — deleting them would be a real breakage.

**The real ones.** `rand` is declared in both `dormouse-core` and
`dormouse-train` and neither has a single `rand::` in its sources. Worth noting
*why* that is not a bug hiding behind the lint: the `--rand-depth` arm lives in
`crates/dormouse-train/src/cfg.rs` and samples the loop depth from the step
index, not from a generator (§3.7). So the dependency is genuinely dead, not
the feature. `dormouse-cli`'s three are one-line deletions each: `serde_json`
and `dormouse-data` appear nowhere in its sources (`dormouse_data` appears only
inside a doc comment in `tests/decode_wiring.rs`), and `burn` likewise.

**To close it:** delete the three in `dormouse-cli`, delete `rand` from
`dormouse-core` and `dormouse-train`, add the four facade entries to
`ignored`, drop `continue-on-error`.

---

## 5. Coverage — a measurement, deliberately without a threshold

```
cargo llvm-cov --lib -p dormouse-core -p dormouse-data -p dormouse-train \
  --summary-only --lcov --output-path lcov.info
```

> **7003 / 8158 lines = 85.84 %**, from CI run **36866436388**, commit
> `e90d69a`, 2026-10-01, ubuntu-latest, `cargo-llvm-cov 0.9.1`. The lcov
> artifact from that run is the evidence; this section was read out of it.

Per file, from the artifact (28 files; the vendored library is excluded by
cargo-llvm-cov's default ignore rules, so this number is *our* crates):

| file | lines | covered |
|---|---|---|
| `dormouse-core/src/attention.rs` | 25 | 100.00 % |
| `dormouse-core/src/aux.rs` | 445 | 100.00 % |
| `dormouse-core/src/gr.rs` | 204 | 99.02 % |
| `dormouse-core/src/moe.rs` | 295 | 98.98 % |
| `dormouse-core/src/dspark_oracle.rs` | 259 | 98.46 % |
| `dormouse-core/src/mixture_probe.rs` | 271 | 98.52 % |
| `dormouse-core/src/future_byte.rs` | 341 | 98.83 % |
| `dormouse-core/src/routing.rs` | 241 | 98.76 % |
| `dormouse-core/src/config/schema.rs` | 91 | 96.70 % |
| `dormouse-core/src/config/loader.rs` | 106 | 95.28 % |
| `dormouse-core/src/act_quant.rs` | 158 | 91.77 % |
| `dormouse-core/src/loop_block.rs` | 1065 | 91.64 % |
| `dormouse-core/src/param.rs` | 133 | 80.45 % |
| `dormouse-core/src/model.rs` | 299 | 73.91 % |
| `dormouse-core/src/probe.rs` | 18 | 72.22 % |
| `dormouse-core/src/config/validation.rs` | 110 | 69.09 % |
| `dormouse-core/src/config/override.rs` | 89 | 65.17 % |
| `dormouse-data/src/lib.rs` | 655 | 90.84 % |
| `dormouse-train/src/lib.rs` | 1729 | 85.19 % |
| `dormouse-train/src/stress.rs` | 88 | 81.82 % |
| `dormouse-train/src/cfg.rs` | 195 | 97.44 % |
| `dormouse-train/src/offload.rs` | 275 | 97.45 % |
| `dormouse-train/src/optim.rs` | 361 | 95.01 % |
| `dormouse-train/src/jepa_targets.rs` | 90 | 88.89 % |
| **`dormouse-train/src/decode.rs`** | **53** | **0.00 %** |
| **`dormouse-train/src/export.rs`** | **426** | **0.00 %** |

### The number is 85.84 % and the honest reading is "two files are untested"

479 lines — **5.87 %** of the total — are `decode.rs` and `export.rs` at zero,
and both have integration tests that exercise them:
`crates/dormouse-train/tests/decode_seam.rs` and `tests/export_roundtrip.rs`.
They read as untested here **only because this job runs `--lib`** and those
are separate binaries. Without them the same number is **91.20 %**.

So the design choice below — `--lib` not `--workspace` — is worth **5.36
percentage points** on this reading, and it is worth naming which way: the
narrower command makes the number *lower*, so it is not the flattering choice.
Widening to the workspace would answer the other question, and would need its
own column in the log rather than a footnote.

The rest of the distribution is the real signal: eight files above 98 %, and the
gap concentrated in `model.rs` (73.91 %), `param.rs` (80.45 %), `probe.rs`
(72.22 %, 18 lines), `config/validation.rs` (69.09 %) and `config/override.rs`
(65.17 %). `config/override.rs` is the config `-C key=value` parser and
`validation.rs` is the refusal path — both are exactly the code AGENTS.md §1.1
depends on being right, and both are among the least covered files in the model
crate. **That is the argument for a coverage gate, and it is an argument about
which files rather than about a floor.**

**No threshold is set, and here is the number to set one from.** A defensible
first floor from this reading is **85 %**: it is 0.84 points below today's
figure, so ordinary work cannot break it, and it fails the moment either
zero-coverage file gets worse. Anything higher starts encoding "do not grow this
crate" rather than "do not lose this coverage".


**Why no threshold.** A floor invented before the first reading is a floor
nobody can satisfy or defend. The job's whole deliverable is that the number
exists, is in the log, and is in an artifact — so the next run makes "coverage
fell" a diff rather than an opinion. Choosing the floor is the owner's call and
it needs the number first.

Two decisions that shape what the number means:

- **`--lib`, not `--workspace`.** The seam tests under `crates/*/tests/` are
  separate binaries. Including them would answer "what do the integration tests
  cover", which is a different and also interesting question; this job answers
  "how much of the library does `cargo test --lib` reach".
- **The instrumented build is its own target directory**
  (`target/llvm-cov-target`), so it never invalidates the plain `target/` the
  other jobs share. The cost is a cold dependency build every run — which is
  why `timeout-minutes: 120`.

The log line and the artifact cannot disagree: the `awk` sums lcov's `LF:`
(lines found) and `LH:` (lines hit) across every record of the same file that is
uploaded. Verified against a hand-built two-file lcov fixture (1 hit / 3 found
→ `33.33%`) before it went in.

### Not measurable on this box

The local instrumented build was attempted and **aborted**: `/home` was at 97%
(15 G free) and `target/` alone is 127 G, so a second instrumented copy of the
dependency tree is a disk-fill, which AGENTS.md §2.5 records as a real failure
mode. The number below is from the CI runner, which is where the job actually
runs.

---

## 6. actionlint — and the finding about actionlint

```
actionlint            # no arguments: lints .github/workflows/*.y*ml itself
```

**Why this job exists:** `ae0505d` is the record. An unquoted colon inside a
`name:` scalar ("the facade: generated files in sync + every feature
combination") made `fused-library.yml` a YAML parse error, so the whole
workflow stopped running — five jobs, including the 1000-case f64 oracle gate,
contributed nothing, and nothing said so. The fix was one pair of quote
characters, found by reading a log after the fact.

**It earned its place inside one commit.** The first draft of this very job
contained two unquoted colons (one in a step `name:`, one in an `awk`
program inside a `run:`). actionlint caught both before either was pushed. The
gate caught the failure mode it was added for, in the file that adds it.

**Then the first CI run went red anyway, on `pyflakes`:**

```
E: Package 'pyflakes' has no installation candidate.
```

`pyflakes` is a PyPI package, not an apt one, and the step installed
`shellcheck pyflakes` in one command, so one wrong name took down the whole
job. Fixed in `e90d69a` (`pipx install pyflakes`, which is preinstalled on
`ubuntu-latest`). Worth recording *how* it failed: loudly, naming the cause and
the escape, within 7 seconds of job start. That is a gate working.

**The finding: actionlint skips a missing external checker SILENTLY.** Measured
on 1.7.12, on a workflow containing `run: echo $HOME`:

| `shellcheck` on PATH | actionlint output | exit code |
|---|---|---|
| yes | `SC2086:info:1:6: Double quote to prevent globbing…` | 1 |
| **no** | **nothing at all** | **0** |

Not a note, not a warning, not a different exit code. So a green actionlint run
cannot distinguish "every `run:` body is shellchecked" from "no `run:` body was
ever checked" — the ADR-0019 SILENT shape, living inside a linter. Hence the
extra step that proves `shellcheck` and `pyflakes` are on `PATH` in the same
step that runs actionlint, which is exactly how actionlint resolves them.
`pyflakes` is installed for `shell: python` bodies, of which this repo has none;
it is there so that adding one does not silently add zero checking.

**Pinning.** actionlint **1.7.12**, tarball verified against that release's
published SHA256 for `linux_amd64`
(`8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8`, the runner
image's platform). No path filter: the file this job protects is the file that
decides what runs, so "only when the workflows changed" can be wrong in the
same commit that breaks them.

---

## 7. rustdoc — green, and the gap in the job that already existed

```
cargo doc --no-deps -p dormouse-core -p dormouse-data -p dormouse-cli
```

57 s, exit 0, **7 `missing documentation` warnings**, nothing denied. The
warnings are deliberately not a gate: `missing_docs` is a style lint, and this
repo's documentation lives in `AGENTS.md` and `docs/`, not in per-item rustdoc.

The build is not the point — the point is that
`broken_intra_doc_links = "deny"` and `invalid_html_tags = "deny"` already sit
in the root `[workspace.lints.rustdoc]` table with every member opting in via
`[lints] workspace = true`, and this is the job where a doc link that does not
resolve stops being a warning.

**`dormouse-train` is not in this job.** It is the heavy crate and it has an
active lane. Naming the omission is the point: a gate covering three of four
crates reads as a gate covering four.

### The docs-site gate was NOT duplicated — and it has a hole

The brief asked for `cd docs-site && npm ci && npm run ingest && npm run build
&& npm run check` in `ci.yml`. **That job already exists**: `build` in
`docs-site.yml` runs exactly those four commands and fails on any dead internal
href. Measured locally on this box, 2026-10-01:

```
internal links checked: 24322
broken internal links: 0
anchor mismatches (warning): 0
check: ok
mermaid: 3 rendered, 0 fence(s) left as code
```

151 pages, 29 152 `href=` attributes total. A second copy in `ci.yml` would
build the same site twice on every push to main. Not added — the brief said
extend, not duplicate.

**The hole, reported rather than fixed:** that workflow's `paths:` filter names
`docs-site/**`, `docs/**`, `research/**`, `README.md`, `AGENTS.md`,
`CONTEXT.md`, `.bulba/**`. A PR that touches **only** a crate's doc comments —
which is exactly what the `docs` job above exists to check — runs **no** docs
gate at all, because the filter never fires. The fix is adding `crates/**` to
that filter. Not done here because `docs-site.yml` is a different lane's file
and a push to it deploys to GitHub Pages.

---

## What each job is NOT, in one list

Read this as the boundary of the work, because a gate that is not checked for
its own blind spots is the failure mode this file keeps finding:

- **fmt and clippy do not see `vendor/dormouse-fused`** (279 hunks there) or
  `backend-parity` / `cublas-poc` (22 hunks). Those are `fused-library.yml`'s and
  nobody's.
- **cargo-deny does not see non-linux targets.** `cfg_block 0.1.1`, the one
  package in 756 with no license field, is excluded by
  `targets = ["x86_64-unknown-linux-gnu"]`. Add a target, get a red gate.
- **cargo-machete cannot see through the `burn` facade**, which is 4 of its 10
  findings and the reason `ignored` entries are needed rather than deletions.
- **coverage excludes `vendor/`, `tests/`, `examples/` and the registry** by
  cargo-llvm-cov's defaults, and this job additionally runs `--lib` only — 5.36
  points of the number, named in §5.
- **actionlint silently skips a missing checker**, so the separate PATH step is
  part of the gate, not decoration (§6).
- **`cargo deny`'s `wildcards` check counts a `path =` dep as a wildcard**, which
  is why 22 of its warnings are about something other than wildcards (§3).
- **No job here runs a CUDA feature.** `cuda-compile` (existing) compiles them;
  nothing executes them, per `fused-library.yml`'s long note on why no GPU
  runner is registered.

---

## Follow-ups, cheapest first

1. **`cargo fmt -p dormouse-core -p dormouse-data -p dormouse-train -p dormouse-cli`** in one commit, then drop `continue-on-error` from the lint job. Closes 629 hunks.
2. **Delete `rand` from `dormouse-core` and `dormouse-train`, and `burn` / `serde_json` / `dormouse-data` from `dormouse-cli`**; add the four facade re-exports to `[package.metadata.cargo-machete] ignored` with the reason. Closes all 10 machete findings.
3. **`cargo clippy --fix`**, then three `#[allow]`-with-reason (offload.rs:36, dspark_oracle.rs:296, and the `nan_reads` cluster in train/src/lib.rs — **read that one first**).
4. **`publish = false` in five manifests**, then flip `bans.wildcards` to `deny`.
5. **Set the coverage floor at 85 %** (§5 gives the argument and the margin). One line: a `--fail-under`-style check on the awk's own number.
6. **Add `crates/**` to `docs-site.yml`'s `paths:` filter** — one line, different lane's file, deploys to Pages.
7. **Add `dormouse-train` to the `docs` job** when its lane lands.
8. **Give `model.rs` and `config/override.rs` a test** — 73.91 % and 65.17 % on the two files a reviewer is most likely to change.
