# Documentation coverage wave 2 — `dormouse-train` (minus `lib.rs`), `dormouse-cli`, and the 40 the CI could not see

**Date:** 2026-10-01 · **Commit base:** `db64923` · **Worktree:** `wt/docs-wave2`
**Lane:** «масштабное написание доков, волна 2». `dormouse-train/src/lib.rs`
(3 113 lines) is off limits throughout — the graph-trainer lane owns it; this
is wave 3's file.

## 1. The discrepancy, resolved: why CI was green and the dedup agent saw 40 errors

The dedup agent measured
`RUSTFLAGS="-D warnings" cargo doc -p dormouse-core -p dormouse-train` →
**40 errors, all in `dormouse-core`'s config seam** — while the CI `docs` job
was green. Both observations were correct, and the gap is four separate facts:

1. **The CI job does not deny warnings.** Its run line was
   `cargo doc --no-deps -p dormouse-core -p dormouse-data -p dormouse-cli`
   with no `-D warnings`, and its comment said so *on purpose*:
   «Those warnings are NOT a gate and deliberately so: `missing_docs` is a
   style lint» (`ci-quality-2026-10-01.md`). The 40 warnings were printed into
   CI logs and ignored.
2. **`dormouse-train` was not in the crate set at all** — its doc surface was
   not even built, so nothing about it could print.
3. **Wave 1's report landed on main without wave 1's code.** The wave-1 review
   (`doc-coverage-2026-10-01.md`) claims config/ fully documented, and the
   commit that did it — `e56cdb9` («the config seam, module docs on all four
   files») — **is not an ancestor of main** (`git merge-base --is-ancestor`
   says no; it lives only on `wt/doc-coverage`). Main did get `9e32bbb` (the
   crate root + the `#![warn(missing_docs)]` / `broken_intra_doc_links`
   attributes), so the LINTS are live on main while the config-seam docs they
   gate are not. A report that outruns its diff is the ADR-0020 failure in its
   documentation form.
4. **Nobody had run the gate with `-D warnings` before the dedup agent** —
   under plain `cargo doc` the same 40 are warnings, and warnings were "not a
   gate". The error-vs-warning split is the whole story: the measurement that
   finally saw them was the first one that denied them.

The 40 (rustdoc's own count: «could not compile `dormouse-core` (lib) due to
40 previous errors») sit entirely in the config seam: `config/override.rs`
carries 5 countable bare items (the `Override` struct + its two fields +
`parse_overrides` + `apply_overrides` — verified by reading, the file had no
docs at all), and the balance is `config/schema.rs` in its compact pre-wave-1
form — the one-line `pub enum ActQuant { Fp4, Int(u32) }` plus the bare
`DormouseConfig` fields (`#[serde(default = "d_bf16")] pub bf16: bool,` with
only a minority of fields carrying `///`). That the 40 are exhausted by
config/ is not an estimate: the worktree diff touches no other `dormouse-core`
file, and its doc build is green.

**Numbers, before and after** (both measured with rustdoc as the judge):

| measurement | main `db64923` | this branch |
|---|---|---|
| `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p core -p train -p cli` | **fails, 40 missing-doc errors (core)** | **exit 0** |
| non-fatal rustc warnings printed during the doc build (train rlib) | 8 | 8 (unchanged — §2) |
| bare pub items in the wave-2 scope (train minus lib.rs, all four cli bins, config/) | 21 non-exempt | **0** (1 `pub use` re-export exempt — rustdoc's `missing_docs` does not ask for it, wave-1 precedent) |
| `cargo test -p dormouse-train --lib` | — | **52 passed, 0 failed** |

## 2. The gate: `RUSTDOCFLAGS`, not `RUSTFLAGS`

The brief pinned `RUSTFLAGS="-D warnings"` for the gate. That exact command
**cannot go green until wave 3**, for a mechanical reason worth its own
paragraph: `RUSTFLAGS` applies to every rustc invocation the doc build makes,
and documenting `dormouse-cli` forces an rlib build of `dormouse-train`, whose
compilation carries **8 pre-existing warnings**:

| warning | where | whose file |
|---|---|---|
| `unused import: Module` | `train/src/lib.rs:17` | lib.rs lane (wave 3) |
| `nan_reads` assigned but never used, ×1 | `train/src/lib.rs:1136` | lib.rs lane |
| value assigned never read (`nan_reads` ×2) | `train/src/lib.rs:1309,1328` region | lib.rs lane |
| value assigned never read (`tsct_bef`, `tsct_aft`) | `train/src/lib.rs:1118-1119` | lib.rs lane |
| method `len` is never used | `train/src/jepa_targets.rs:144` | this lane, §4 |
| method `to_bytes` is never used | `train/src/offload.rs:171` | this lane, §4 |

Six of the eight are in the forbidden file. `RUSTDOCFLAGS="-D warnings"` gates
exactly the rustdoc units — the doc surface — and leaves the rlib warnings
visible-but-non-fatal in the build log, which is the COUNTED form of the same
information. The CI job now runs:

```
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps \
  -p dormouse-core -p dormouse-data -p dormouse-train -p dormouse-cli
```

What that enforces, and what it deliberately still does not:

- **Denied:** every rustdoc warning on all four crates — `missing_docs` for
  core and data (whose lib.rs carry the attribute), `broken_intra_doc_links`
  and `invalid_html_tags` for all four (workspace lints, every member opts in).
- **Not enforced:** `dormouse-train`'s own `missing_docs` — the attribute
  lives in its lib.rs, whose lane is in flight; adding a lint line to a file
  another lane owns is how merge conflicts are manufactured. The omission is
  named in the CI comment; adding the attribute is wave 3's one-line
  follow-up, after which the RUSTDOCFLAGS→RUSTFLAGS flip can be considered.
- The cli bins carry no pub items (arg structs are private), so
  `missing_docs` has nothing to ask of them; their `//!` module docs were
  already in place and are now reachable through the doc build.

## 3. What this lane actually wrote

Less than the brief assumed, because the prior lanes had been there: `optim.rs`
was fully documented by the dedup lane (its module doc already narrates the
`831e3a0` twin cut), `decode.rs`, `stress.rs`, `offload.rs` and all four cli
bins carried module docs, and the worktree held 441 lines of finished,
uncommitted config-seam + `export.rs`/`jepa_targets.rs` docs from an
interrupted earlier run of this same lane — reviewed line by line for factual
accuracy (every cross-reference resolves; the `engram=`/`--set` claims check
against the code) and landed rather than rewritten.

This lane's own additions:

- `config/loader.rs` — one typo (`CARGO_MANEST_DIR` → `CARGO_MANIFEST_DIR`).
- `train/export.rs` — 16 field docs (`TensorEntry` 2, `Header` 6, `Report` 8;
  the paired-comment form distributed onto the fields it described).
- `train/cfg.rs` — `RunCfg.model` / `RunCfg.train` (the second names the
  `diff_keys` progress-key exemption, `cfg.rs:112-114`).
- `train/offload.rs` — `HostNgram.slots` / `.dim` / `total_rows`, plus the
  stale-struct-comment fix in §4.
- `train/optim.rs` — `Installed.muon` / `.table`.

## 4. Found, not fixed (docs-only lane)

1. **`config/loader.rs:45-46` — a live ADR-0019 SILENT fallback in the preset
   search path.** `try_file` drops a candidate that exists but fails to read
   or parse with both error modes discarded (`read_to_string(p).ok()?`), so
   `--preset small` with a syntax error in `./configs/small.toml` silently
   falls through to the repo's copy and runs a config nobody asked for. The
   explicit-path branch of the same function returns the parse error — the
   same failure is LOUD or SILENT depending on how the preset was spelled.
   Cheapest fix: collect the failures into the terminal `Err`. Now documented
   in the module doc; the fix is a code change.
2. **`config/override.rs:15` (module doc) — four schema fields are
   unreachable from `--set`**: `use_situ`, `use_attnres`, `use_mhc`,
   `mhc_streams` have no arm in `apply_overrides`' match. The refusal is LOUD
   (`unknown config key`), so nothing trains differently from what was asked —
   but SiTU-GLU, AttnRes and mHC are all A/B queue rows and all three need a
   preset-file edit where every other arm takes a flag. Fix: four match arms,
   ~6 lines.
3. **`train/src/lib.rs:1007,1009` — the host n-gram tables hardcode
   `dim = 32`** (and seed `0x1234_5678`) instead of reading `cfg.engram_dim`.
   The schema default is 32 (`schema.rs:107`), so every shipped run is
   consistent — but `--set engram_dim=64` would build in-model Engram tables
   64 wide against host tables 32 wide (the host path's row payload is the
   hardcoded `3 × 32 = 96` floats). lib.rs is the wave-3 file; recorded at
   the call site's doc in `offload.rs`.
4. **`train/src/offload.rs:38` (pre-fix numbering) — the struct comment said
   «3 tables (3/5/8-gram)» and `dormouse_data::ORDERS = [2, 3, 4]` says
   2/3/4-gram.** Wrong comment, right code; fixed in this lane (a comment is
   this lane's scope) and listed because it is the kind of staleness
   `schema.rs:517`'s own test comment warns about.
5. **Dead code the gate surfaced** (rustc `dead_code`, non-fatal under the
   new gate): `jepa_targets.rs:144 JepaTargets::len` and
   `offload.rs:171 HostNgram::to_bytes` are called from nowhere outside
   tests — `to_bytes` looks superseded by `write_to` (same layout, streaming).
   Deletion candidates for their owner's next pass.
6. **`train/src/lib.rs:17` — `Module` is an unused import** (the first of the
   eight warnings). This lane removed the wrong twin first: the same import
   line exists in `export.rs:40`, where `Module` IS used
   (`tensors_of`'s `model.visit`), and deleting it there turned the doc build
   E0599 (`no method named visit`). Reverted within the lane; recorded as the
   trap it is — `unused import` diagnostics name the file, read them.
7. **Pre-existing at the lane's base, fixed on main while this lane ran:**
   `tools/check_doc_refs.py` named 3 dead references at `db64923` — the two
   archive documents' paper refs and the dedup review's bare ADR number (its
   real filename is `0017-dormouse-fused.md`). All three are repaired on
   `origin/main` (`a0a1dce`, `a2a5877`) by other lanes; recorded because the
   lane that measures a gate's blindness should say when someone else cured
   it. The 2 this lane created resolve with this file's landing.

## 5. Deliberately not done

- **`dormouse-train/src/lib.rs`** — wave 3, after the graph lane lands. Its
  8 of the rlib warnings and its `missing_docs` attribute are that wave's.
- **`#![warn(missing_docs)]` on the cli bins** — the bins expose no pub
  items; a lint with nothing to enforce is a ceremony. Revisit if a bin ever
  grows a pub item.
- **Exact per-line attribution of the schema.rs share of the 40** — rustdoc's
  two verdicts (40 on main, 0 here) are the measurements that matter, and
  override.rs's 5 are verified by reading; decomposing main's remaining 35 to
  individual fields wanted a contended-lock rustdoc run of main that the gate
  this lane ships makes unnecessary — the next bare field in core fails CI by
  name.
