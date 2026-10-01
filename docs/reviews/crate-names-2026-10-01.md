# The 21 library crates: `burn-*` → `dormouse-*` — 2026-10-01

Lane: the naming decision ("именование нормальное"), following the directory
rename `vendor/burn-fused` → `vendor/dormouse-fused` (ADR-0017, `0e9f14c`).

**A pure rename.** No line of logic changed; no number in this repository moved.
The gate that says so is `cargo tree -e features` before and after, compared as
a multiset (§ "The isomorphism check" below).

## What was renamed

20 member crates + the facade, in `vendor/dormouse-fused`:

| was | now |
|---|---|
| `burn-attnres` `burn-bitnet` `burn-dspark` `burn-eggroll` `burn-engram` | `dormouse-*` (same suffix) |
| `burn-es` `burn-gdn2` `burn-jepa` `burn-kda` `burn-mhc` `burn-mor` | ″ |
| `burn-muon-plus` `burn-parcae` `burn-ptrn` `burn-rmsnorm` `burn-rope` | ″ |
| `burn-sct` `burn-situ` `burn-spectral` `burn-swiglu` | ″ |
| `burn-fused` (the facade) | `dormouse-fused` |

Plus the two private probe packages the workspace members list names:
`burn-fused-benches` → `dormouse-fused-benches`.

Three names were **not** touched, deliberately, and each is a real
distinction rather than an oversight:

- **`burn-tensor`, `burn-nn`, `burn-cuda`, `burn-autodiff`, `burn-ndarray`,
  `burn-pack`, `burn-optim`, `burn-cubecl`, `burn-cubek` and bare `burn`** are
  the upstream framework. They are not ours to rename; the crates here are
  built *against* them, which is exactly what `description` fields like
  `"RMS Normalization for Burn"` correctly continue to say.
- **URLs** are locations, not names. `repository = "…/sehaxe/burn-kda"`,
  `documentation = "https://docs.rs/burn-muon-plus"` and the crates.io badges
  still carry the published names, because those crates were published under
  `burn-*` and renaming a URL would point it at a crate that does not exist.
  The substitution masks URL spans, so the link *text* beside a URL is renamed
  while the URL is not.
- **`/home/sehaxe/burn-fused`** in AGENTS.md §2.5 is the stale unreferenced
  working copy of the fork, not the library. Renaming it there would cite a
  path that does not exist.

**Follow-up, not done:** whether these crates are ever PUBLISHED under the new
namespace is the owner's call and is a separate decision from renaming them.

## Where the names lived

Beyond the 21 manifests, the name was written into six kinds of place, and each
one is a place a rename silently misses if you only edit `Cargo.toml`:

1. **Cargo manifests** — package `name`, the `path =` of every intra-library
   dep, the dependency *keys* (`dormouse-gdn2 = { path = … }`), feature
   strings (`"dormouse-kda/cuda"`, `"dep:burn-cuda"` stays `burn-cuda`), the
   workspace `members` list and `default-members`.
2. **Rust idents** — `use burn_spectral::…` → `dormouse_spectral::…`: 104 such
   lines in `crates/**` alone, plus the whole library's own `.rs`.
3. **The facade generator** — `tools/gen_facade.py` derives its re-export list
   and feature table from the member manifests, so the manifests are the
   SOURCE and the facade's `lib.rs`/`Cargo.toml`/`tests/facade.rs` are
   regenerated output. Two changes went into the generator rather than its
   output: the facade's own crate name is now read from its manifest
   (`facade_ident()`) instead of being spelled out in three places, and the
   doc example's member is looked up (`kda_ident()`) instead of hardcoded, so
   the next rename cannot half-work.
4. **CI** — `.github/workflows/fused-library.yml` (the `-p` flags, the
   `working-directory`, the 13-name CUDA crate list, the `--exclude` list),
   `.github/workflows/ci.yml`, `CODEOWNERS`, `PULL_REQUEST_TEMPLATE.md`,
   `ISSUE_TEMPLATE/`.
5. **Tools** — `lib_gate.sh`, `falsify_fused_adjoint.sh`, `fused_matrix.sh`,
   `wt.sh`, `mor_ab.sh`, `test_targets.py`, `gen_oracle_tiers.py`,
   `fused_matrix_static.py`, `check_doc_refs.py`, `migrate-dormouse-fused.sh`
   and the library's own `gpu-gate.sh` / `test-feature-matrix.sh`.
6. **Documents** — 72 files across `docs/`, `research/`, `AGENTS.md`,
   `CONTEXT.md`, `README.md`, `deny.toml`, `benches/history.tsv` and
   `docs-site/`. `docs/archive/` is history and keeps its old names.

## The isomorphism check

The claim "no numeric change" needs an instrument, not an assertion.
`cargo tree -e features --workspace` was captured in a pristine worktree at
`0e9f14c` and again after the rename, and compared as a **multiset of
(package, version, feature) lines** with the new name mapped back to the old.

Both workspaces: **isomorphic** — no package, version or feature added,
removed or changed. Library 2850 lines / 1401 distinct; root 3549 / 1861.
Baseline is `6980040`, the commit this branch is rebased onto, so the two
trees differ only by the rename.

Two details the comparison had to get right, both of which would have produced
a false RED:

- **`cargo tree` prints the crate's build path** in `(…)`. Normalising the
  names inside it also rewrote the *worktree path*, so the comparison has to
  strip the location.
- **`cargo tree` orders workspace members alphabetically by name.** Renaming
  `burn-attnres` → `dormouse-attnres` moves it in that order, so the depth
  prefixes shift for reasons that have nothing to do with the graph. The
  depth prefix is therefore dropped.

## Three defects the rename found

All three are in the tooling, not the model. Two were fixed here; the third was
already fixed upstream while this lane was queued.

### 1. A `git mv` that nests instead of renaming

The library's 6 committed `.bin` oracle fixtures live in a directory that
`git clean -fd` had already emptied of tracked content in an earlier attempt,
leaving the destination directory present on disk. `git mv burn-gdn2
dormouse-gdn2` then did what `git mv` does when the destination exists —
**it moved the crate INTO it**, producing
`crates/dormouse-gdn2/burn-gdn2/{Cargo.toml,src,…}`.

The failure was not subtle and not about the rename: cargo reported
`couldn't read crates/dormouse-gdn2/src/lib.rs`, and five gates went red
instantly. The one-line guard is in the rename script: refuse to `git mv` when
the destination already exists, and say why. **The general lesson is the
ADR-0011 one**: a tool that moves files has to check its precondition, because
`git mv` succeeding is not evidence the rename happened.

### 2. Renaming inside URLs fabricates locations

A first pass rewrote every occurrence, which produced
`dormouse-fused = { git = "https://github.com/sehaxe/dormouse-fused" }` in the
library README — a repository that does not exist — and crates.io badges
asserting a publication that has not happened. The substitution now masks
`https?://\S+` spans. **A name is a name; a URL is a location**, and only the
first one is ours to change.

### 3. A `.bin` fixture the library's own `.gitignore` ate

`vendor/dormouse-fused/.gitignore` carries `**/*.bin` with six `!` exceptions,
one per committed oracle fixture — a rule the file's own comment calls out as a
trap that once hid `ref_f64_broad.bin` for a day. The exception list named the
gdn2 fixtures and **not** `dormouse-muon-plus/tests/oracle/muon_oracle.bin`, so
that fixture was absent from a fresh checkout and
`cargo check --workspace --all-targets` died on
`couldn't read tests/oracle/muon_oracle.bin`.

This lane did not cause it and did not fix it: `98e24b1` ("the tier-a oracle
fixture returns to the index — the library gitignore's `**/*.bin` silently ate
it") landed upstream while this lane was queued behind the build lock, and this
branch is rebased onto it. **The finding worth keeping is the shape**: a
fixture excluded from the index is a test that does not exist for every reader
but the one whose disk has it, and `include_bytes!` resolving on the author's
machine is why every local build forgave it. A gate that only ever runs where
the fixture happens to exist is not a gate. `tools/test_targets.py` is the
existing guard for the `required-features` half of this; nothing yet guards the
"is the fixture in the index" half.

## Gates

| gate | result |
|---|---|
| `cargo tree -e features` isomorphism, library + root | **isomorphic** (above) |
| `cargo check --workspace --all-targets` in `vendor/dormouse-fused` (minus the 3 probe packages) | **GREEN** |
| `cargo test -p dormouse-core -p dormouse-train --lib` | **GREEN** |
| `cargo check -p dormouse-train --features dormouse-train/cuda` | **GREEN** |
| `cargo fmt --all -- --check` in `vendor/dormouse-fused` | RED, **pre-existing** — see below |
| `vendor/dormouse-fused/tools/test-feature-matrix.sh` | 5/7 combos PASS, 2 RED **pre-existing** — see below |
| `tools/check_doc_refs.py` | **2 findings, identical to a clean worktree at `6980040`** — both from `docs/archive/`. Zero added. |
| `tools/test_targets.py` | 0 findings |
| `tools/gen_facade.py --check` | in sync |
| `git grep burn-kda\|burn-gdn2\|burn-spectral` outside `docs/archive` | 0 (bar the URL exemptions above) |

**The two RED gates are RED at the pristine baseline too.** Measured on a
detached worktree at `6980040` with no changes of mine, same commands:

- `cargo fmt --all -- --check` — the reordering it wants is in
  `dormouse-attnres/src/fused_attnres.rs` and
  `dormouse-fused/tests/gpu_production_shape.rs`, two files whose **contents
  this rename does not touch** (verified: zero changed lines).
- `test-feature-matrix.sh` — the `--features std,autodiff` and
  `--features std,cuda` cells fail at `6980040` with the same error classes
  (`cannot find attribute comptime/cube`, `cannot find module burn_cubecl`) in
  `burn-rope` and `burn-gdn2`. The CUDA kernels do not lower under a
  feature combination that enables `cuda` without `autodiff`, or `autodiff`
  without the CUDA deps — a pre-existing gate failure this lane did not
  introduce and did not fix.

Both are **not mine to fix**: they are in files this rename does not touch, in
another lane's crate bodies. The honest statement is that G2 and G3 are red
before and after, by the same errors.

`docs-site/npm run check` could not run in this worktree: it reads `dist/`,
which requires `npm install` + `npm run build`, and the machine was under the
build lock for the whole of this lane. The one manifest line that named a
crate directory (`'vendor/dormouse-fused/crates/'` in `dirIndex`) is updated,
and `IA.md`'s two `burn-kda`/`burn-gdn2` references are renamed. **Not
verified** — a follow-up must run it.

## Not done, on purpose

- **`docs/archive/` keeps the old names.** It is history (AGENTS §1.4), and a
  document that described `burn-kda` at the time it described it is not wrong
  for being renamed later.
- **`repository`/`homepage`/`documentation` keep `burn-*`**, as above.
- **The publication question** is untouched.