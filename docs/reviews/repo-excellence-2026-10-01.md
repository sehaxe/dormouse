# Repo excellence: burn's practices, and what we took

Date: 2026-10-01. Lane: repo excellence. Question asked: *how is burn's repo
laid out, and what of it makes this repo maximally quality and easy to develop
in.* Sources read directly, not from memory:

| source | what was read |
|---|---|
| `tracel-ai/burn` @ `main` | root `Cargo.toml`, `rustfmt.toml`, `deny.toml`, `CONTRIBUTING.md`, `.github/pull_request_template.md`, `.github/ISSUE_TEMPLATE/{bug_report,feature_request,doc_request}.md`, `NOTICES.md`, `xtask/src/main.rs`, `crates/burn-tensor/Cargo.toml`, and the full `git/trees/main?recursive=1` listing (2 689 paths) |
| `huggingface/candle` | referenced from the existing `docs/research/2026-09-25-nasa-burn-rust-practices.md` rather than re-read; that pass already covers its dispatch and error-model shapes |
| this repo | `ls -la` root, `.github/`, all six crate manifests, `.gitignore`, `.cargo/config.toml`, `README.md`, `AGENTS.md`, `docs/`, `tools/` |

Boundaries respected: **`.github/workflows/` and `deny.toml` were not touched** —
another lane lands fmt/clippy/deny/coverage tooling there the same day.
`[workspace.lints.clippy]` is therefore declared and left **empty**, with a
comment saying why: a lint table nothing runs is the ADR-0019 SILENT shape.

---

## 1. Checklist

### What burn has, grouped by whether it helps a one-dev-plus-agents repo

| practice | burn's form | helps us? |
|---|---|---|
| `[workspace.lints]` + per-crate `[lints] workspace = true` | empty clippy table, `rustdoc` denies for `broken_intra_doc_links` / `invalid_html_tags` | **yes** — one place for lint policy instead of six copies; both denies are about a claim being *readable* |
| `[workspace.dependencies]` | ~90 version-bearing entries | **yes** — we had 15 copies of `0.22.0-pre.4` across 6 manifests |
| `rustfmt.toml` | `max_width = 100`, two lines | **partly** — see §3, we ship the two lines and not the sweep |
| `CONTRIBUTING.md` | 250 lines: change ownership, AI-assisted contributions, "bug fixes should include a regression test", "keep dependencies minimal" | **yes** — but only the practice layer; the rules stay in AGENTS.md |
| `PULL_REQUEST_TEMPLATE.md` | checklist: run-checks, book updated | **yes** — reshaped to our own required fields |
| `ISSUE_TEMPLATE/*` | bug / feature / doc-request | **one of three** — a *mechanism proposal* template, not a feature request |
| `deny.toml` | license allow-list, `unknown-registry = "deny"`, per-target graph | off-limits today |
| `NOTICES.md` | 23 KB of attribution for copied code | **no** — see §3 |
| `xtask/` | `cargo run-checks`: fmt + typos + deny + clippy + no-std + tests, one command | **already have it**, differently — `.cargo/config.toml` aliases + `tools/*.sh` + `tools/build_lock.sh` |
| `rust-toolchain.toml` | absent | n/a |
| `clippy.toml` | absent (404) | n/a |
| CI matrix (bench / wasm / no-std / GPU / valgrind / stale-PR / dependabot) | 7 workflows | off-limits today; most are N/A for one box |
| `CODE_OF_CONDUCT.md`, `CITATION.cff`, `_typos.toml`, `codecov.yml`, `benchmarks.toml` | — | no / off-limits |

### What we have that burn does not — keep it

burn has **no** AGENTS.md, **no** glossary, **no** ADR directory, **no** retraction
record, **no** evidence-provenance rule, and **no** A/B-or-death mechanism rule.
Its `CONTRIBUTING.md` says "bug fixes should include a regression test"; ours says
a mechanism is deleted unless it beats its own removal on 3 seeds. Those are not
the same project, and importing burn's process wholesale would be a downgrade.
Every file this lane landed **defers to `AGENTS.md` §1 rather than restating it**,
because a second copy of the rules is a second copy that drifts.

---

## 2. Audit table

`have` = already present · **landed** = landed by this lane · queued = named,
not done · **rejected** = not adopted, with the reason.

| practice | before | verdict |
|---|---|---|
| `[workspace.lints]` | absent | **landed** — `961c5df` |
| `rustdoc::broken_intra_doc_links = "deny"` | absent | **landed** — found 3 real broken links |
| `rustdoc::invalid_html_tags = "deny"` | absent | **landed** — found 3 bare `<…>` in CLI docs |
| `[lints] workspace = true` × 6 crates | absent | **landed** |
| `[workspace.dependencies]` | absent, 15 duplicated versions | **landed** — `a3da1a4` |
| `rustfmt.toml` (`max_width = 100`) | absent | **landed** (`§3.1`) |
| `CONTRIBUTING.md` | absent; rules in README, which deferred to AGENTS.md | **landed** — `b6a75c9` |
| PR template | absent | **landed** — `b6a75c9` |
| Issue templates | absent | **1 of 3 landed** — `b6a75c9` (`§3.2`) |
| knowledge-base manifest line for the new doc | n/a | **landed** — `b6a75c9` |
| `.editorconfig` | absent | **landed** — `5f0db7a` |
| `.github/CODEOWNERS` | absent | **landed** — `5f0db7a` |
| `SECURITY.md` | absent | **landed** — `5f0db7a` |
| `cargo fmt --all` sweep | 651 hunks in our crates, 279 in `vendor/` | **rejected here** (`§3.3`) — belongs to the fmt lane |
| `[workspace.lints.clippy]` populated | absent | **queued** — the CI lane that owns the gate fills it |
| `NOTICES.md` | absent | **rejected** (`§3.4`) |
| `xtask` / `cargo run-checks` | have `.cargo` aliases + `tools/*.sh` | **rejected** (`§3.5`) |
| dependabot, stale-PR, valgrind, wasm, no-std jobs | absent | **rejected** (`§3.6`) |
| `_typos.toml` | absent | **rejected** (`§3.6`) |
| `rust-toolchain.toml` | absent | **rejected** (`§3.6`) |

---

## 3. What landed, and what did not

### 3.1 `rustfmt.toml` — the config without the sweep

burn's is `max_width = 100` and two commented-out lines. Shipped the same two
lines. **The 651-hunk reformat is deliberately not in this lane**: it rewrites
58 of our files and 279 hunks under `vendor/`, which is a review-sized diff that
would drown four small metadata commits, and the fmt/clippy CI lane is landing
today and will have an opinion about the vendor half. The measurement is in
`rustfmt.toml`'s commit for whoever runs it: **ours 651 hunks / 58 files,
`vendor/` 279 hunks** at `max_width = 100`.

Two configs were measured and rejected as *worse*: `max_width = 110` (606 ours
but **591** in `vendor/`) and `use_small_heuristics = "Max"` (**1340** ours, up
from 651). Widening the width to fit the vendored code makes our own code less
formatted, not more. 100 is both burn's number and the default, so the config
is a statement rather than a lever.

### 3.2 Why one issue template and not three

burn has `bug_report`, `feature_request` and `doc_request`. A **doc request**
template is N/A (a doc that disagrees with the code is a defect report — bug
template, §1.7). A **feature request** template is the wrong shape for this
repo: a "feature" here is a mechanism, and the question that decides it is what
would have to be true for the mechanism to be **deleted** (`docs/protocols/AB-PROTOCOL.md`
§1.2). So the landed template asks that, asks for the gate that can fail, and
asks whether the thing has *ever run* — "implemented" and "executed" are
different claims here and the difference has retracted results before (§3.2).

The bug template asks for the eval **line** including `fused kda=`, `engram=`,
`moe=`, `tsct=`, `fb=`. A report without those may describe a run that measured
a different program than the one intended, which is exactly how the Engram
6.453 was retracted.

### 3.3 The fmt sweep, and the 30 files with no final newline

Not fixed here (reasons above). `.editorconfig` says `insert_final_newline =
true`, which is the *intent*; the 30 files are pre-existing and rustfmt will
clear them as it formats. Flagged so nobody reads the config as a claim that
the tree already complies — it is a target, and the file says so.

### 3.4 `NOTICES.md` — rejected, with the reason that matters

burn's is 23 KB and exists because burn copied whole files (PyTorch's MNIST
example, wgpu's CI config). We have exactly **one** place where upstream code
was adapted: `vendor/burn-fused/crates/burn-sct/src/qr.rs:10-16`, which says in
its own module doc that it is adapted from `burn-rs/burn`
`crates/burn-tensor/src/tensor/linalg/qr.rs`, authored by the burn-rs
maintainers, MIT/Apache-2.0. The attribution is **in the source, at the
function, with the licence** — which is stronger than a file 400 lines away in a
notices file that nobody reads. The two vendored forks carry their upstream
`LICENSE` files verbatim (`vendor/cubecl-fix/LICENSE`,
`vendor/cubek-fix/LICENSE`: "Nathaniel Simard & CubeCL Framework Contributors"),
and `vendor/burn-fused/LICENSE:24-31` states its own scope.

**One real disagreement found, not fixed, because it is another owner's file:**
`burn-sct`'s README says the adaptation is "MIT" while `qr.rs:12` says
"MIT/Apache-2.0", and burn's own root manifest says `license = "MIT OR Apache-2.0"`.
Dual-licensed upstream means the derived work may be distributed under **either**;
saying only "MIT" is the more restrictive choice, so it is not wrong — but two
files in one crate answering the same question differently is the defect class
`docs/glossary.md` exists for. Reported at
`vendor/burn-fused/crates/burn-sct/README.md:62` vs `src/qr.rs:12`.

Also worth recording: **the upstream path in that attribution no longer
exists.** `crates/burn-tensor/src/tensor/linalg/qr.rs` is 404 on burn `main`;
the file now lives at `crates/burn-linalg/src/functions/qr.rs`. The citation is
pointing at a path that has moved, which is the same dead-pointer shape
`tools/check_doc_refs.py` was written for — except this one crosses a repository
boundary, so no tool in this repo can see it. §1.4's rule (a claim names the
file the reference came from) is only as good as the path.

### 3.5 `xtask` / `cargo run-checks` — already have the capability, differently

burn's xtask is ~10 command modules wrapping fmt + typos + deny + clippy +
no-std + backend tests into `cargo run-checks`, and its PR template's only
checklist item is "I ran `cargo run-checks`". We have the same reachability
through four surfaces that already exist and are already documented:
`.cargo/config.toml` aliases (`cargo check-train`, `test-core`, `build-probe`),
`tools/wt.sh test` (two gates), `tools/lib_gate.sh`, and `tools/build_lock.sh`
(which burn has no equivalent of and which exists because five agents once ran
five cold builds at once). A Rust xtask would be a second front door to the same
commands, in a language that needs compiling to list a help text.

**The honest gap**: there is no ONE command that runs everything. When the CI
lane lands fmt/clippy/deny, the missing piece is a `tools/checks.sh` (or one
xtask subcommand) that runs the whole set under `build_lock.sh`. That is a
follow-up, and it is the shape burn's `cargo run-checks` should take here.

### 3.6 Rejected as ceremony, with the reason

- **dependabot** — a repo that bumps burn to `-pre` builds by hand, vendors five
  patched crates, and measures reproducibility would receive weekly PRs that
  change `Cargo.lock` under a machine with 3 documented freezes. `Cargo.lock` is
  an experimental record here, not a lock.
- **stale-PR / valgrind / wasm / no-std workflows** — N/A for one maintainer, one
  GPU, one target. burn runs them because it has 100+ contributors and 6 release
  targets.
- **`_typos.toml`** — burn needs it because 200 crates of upstream vocabulary
  trip it. Ours is 40-odd files of English, and the words that would false-positive
  (`fused`, `arm`, `Engram`, `iteration`) are **glossary terms with definitions**;
  a spell checker that flags them is worse than none. The real typo class in this
  repo is the doc-vs-code disagreement, and that has a tool
  (`tools/check_doc_refs.py`).
- **`rust-toolchain.toml`** — pinning a toolchain file pins the *rust-analyzer*
  and CI too, and this box has a CUDA toolkit-version pin that already lives in
  `.cargo/config.toml [env]` for a documented reason. One toolchain pin, in one
  place, is the correct shape; splitting it across a file and an env table is
  not.
- **`clippy.toml`** — burn has none (404). Nothing to copy.
- **per-crate `README.md`** for our six crates — vendored crates have them and
  should (they are the paper reference for each mechanism); our own crates are
  read through `AGENTS.md`, `docs/` and the crate's module docs.

---

## 4. Verification

| check | result |
|---|---|
| `cargo doc --no-deps` × 4 public crates | clean, 0 warnings from our crates |
| `cargo tree -e features`, 6 members, before/after both manifest commits | **byte-identical** |
| `Cargo.lock` | untouched (`git checkout` after confirming nothing needed to move; the only diff was `rustls`/`chacha20` patch bumps cargo wanted, which are not this lane's) |
| `cargo test -p dormouse-core -p dormouse-data -p dormouse-train --lib` | **52 + 15 + data green**, 0 failed |
| `python3 tools/check_doc_refs.py` | 832 files, 541 refs, **0 missing** |
| `docs-site`: `npm run ingest && npm run build && npm run check` | **ok** — 150 pages, 0 broken internal links, `dist/start-here/contributing/index.html` built |
| `.editorconfig` vs the editorconfig spec | 10 sections, `root` preamble first, no illegal keys, every value a legal literal |

## 5. Follow-ups, not done here

1. **`cargo fmt --all`** — 651 hunks ours / 279 `vendor/`. The fmt lane owns the
   vendor question; ours is a plain sweep once the width is agreed (100).
2. **Fill `[workspace.lints.clippy]`** — with whatever the clippy CI gate
   actually enforces, so the table and the gate cannot disagree.
3. **`tools/checks.sh`** — one command over the whole check set under
   `build_lock.sh`; the local stand-in for burn's `cargo run-checks`.
4. **`burn-sct` licence disagreement** — `README.md:62` ("MIT") vs `src/qr.rs:12`
   ("MIT/Apache-2.0"), and a citation pointing at an upstream path that has moved
   to `crates/burn-linalg/src/functions/qr.rs`. Vendor-lane owner's call (§3.4).
5. **30 tracked files without a final newline** — rustfmt clears them as it goes;
   no dedicated commit warranted.