# The config seams: three ADR-0019 SILENT fixes and one deletion

Date: 2026-10-01. Lane: `wt/config-seams`, off `a2a5877`. Findings from
`docs/reviews/doc-coverage-2026-10-01.md` (§3.1–§3.3) and
`docs/reviews/dead-pub-audit-2026-10-01.md` §2; each fix is one commit, gates
inline. CPU only (the seed wave owns the GPU).

## 1. `config/loader.rs` — a present-but-broken preset is COUNTED (`179bdf3`)

**Was:** `try_file` discarded both the read error and the parse error
(`read_to_string(p).ok()?` + `parse_str(&s).ok()`), so `--preset small` with a
syntax error in `./configs/small.toml` fell through to the repo copy and
trained on a config nobody asked for, with no word on any stream. The explicit
path failed loudly on the same input — the asymmetry was the finding.

**Is:** `try_file` returns `Option<Result<DormouseConfig, String>>`: absent
candidate stays a silent normal miss of a search path; a present-but-unusable
one is printed with its reason before the search moves on —

```
[config] preset candidate configs/small.toml unusable (toml parse: TOML parse
error at line 2, column 11); trying next
```

**Why COUNTED, not LOUD (one line, per ADR-0019):** the name search is a
best-match by design and the dangerous case is precisely the one where another
candidate loads — a terminal-`Err` variant can never fire there, so the
warning must fire at the moment of the skip; the run stays correct AND the
reader of the log can tell which file actually won.

**red→green, on the real code path:** the old test binary
(`dormouse_core-225a4a1058c91f40`, base `a2a5877`) run from a cwd holding a
broken `configs/small.toml`: `presets_match_original_values` passed in total
silence. The fixed binary, same command, same file: the warn above (twice —
the test loads `small` twice) and still passes. Permanent gate:
`config::loader::tests::broken_candidate_is_reported_absent_is_not`
(broken → `Some(Err)` naming the parse; absent → `None`).

## 2. `config/override.rs` — `--set` knows the four queue arms (`68cb247`)

**Was:** 36 of the schema's 40+ fields were reachable from `--set`;
`use_situ`, `use_attnres`, `use_mhc`, `mhc_streams` were reachable only from a
preset TOML — every A/B of those arms started with a hand edit under
`configs/`. The refusal was LOUD (`unknown config key`), so nothing trained
differently from what was asked; the defect was the inconsistency.

**Is:** four arms in the match (three bools via `parse_bool`, one `usize`),
mirroring `use_mor`/`mor_k`. Gate:
`config::r#override::tests::arm_flags_resolve_from_set` — one assertion per
key, from `DormouseConfig::default()`.

## 3. `param.rs` — the non-CUDA `bf16_compute` drop is COUNTED (`21557ba`)

**Was:** with `bf16_compute` on and no `cuda` feature, the bf16 matmul op does
not exist and the fp32 path ran — right answer, no counter, no line. ADR-0019
row 29 recorded it SILENT-by-cfg; doc-coverage §3.3 made it the fourth open
SILENT outside that enumeration's fixed set.

**Is:** `bf16_compute_dropped_once()` — one stderr line per process (the call
site runs on every forward):

```
[param] bf16_compute requested on non-cuda: fp32 path (counted, once)
```

The forward's doc comment no longer says "not fixed here". Gate:
`param::tests::bf16_compute_on_cpu_is_the_fp32_answer_and_says_so` — the arm
answers fp32-equal (max |Δ| < 1e-5) on the CPU test backend, and with
`--nocapture` the line prints exactly once across two forwards through the
branch. The once-per-process line is a deliberate ceiling: a per-call counter
in `probe.rs` would cost an `N_ARMS` bump for a state that cannot recur.

## 4. `dormouse-data` — `train_eval_split` deleted (this commit)

Dead per dead-pub-audit §2 (called by nothing; the trainer takes an explicit
`--eval` file), advertised by the crate doc at `lib.rs:16`. Function and doc
line deleted together; `grep train_eval_split crates/ tools/` is 0. The audit
doc carries a one-line disposition note; its record is untouched.
`collect_files` stays (it has live callers: `bin/anchors.rs`, tests, the
trainer).

## Gates

- `cargo test -p dormouse-core -p dormouse-data --lib`: **89 + 15 passed,
  0 failed** (baseline before any edit: 86 + 15, so +3 = the three new gates).
- `RUSTFLAGS="-D warnings" cargo doc --no-deps -p dormouse-core -p
  dormouse-data`: **Finished, 0 errors**. Note the honest scope: this gate
  enforces rustc lints; `RUSTDOCFLAGS="-D warnings"` still fails on **41
  pre-existing missing-documentation items** (`schema.rs` 30, `override.rs` 5
  — all at lines 6/8/24, the old pub items — `config/mod.rs` 3,
  `validation.rs` 1, `lib.rs` 1). None are from this lane; closing them is a
  docs-lane decision, not smuggled in here.
- `python3 tools/check_doc_refs.py`: the only new path it names is this file
  (resolved by its existence); the archive failures are pre-existing.
