# Lane: удаление неживых крейтов `vendor/dormouse-fused` — УЛИКИ (§1.4)

Measured on `main` @ `db64923`, 2026-10-01. **No repo file touched** — the
precondition (`dormouse-kda` in main's `Cargo.toml`) was unmet at every reading.

## The two instruments

```bash
# 1. the crates a training build compiles (vendor/ is `exclude`d, so the ROOT lock is honest)
grep '^name = "burn-' Cargo.lock          # 13 vendored names

# 2. every path-dependency edge into a vendored crate, and its kind
grep -rn 'burn-<x>' --include=Cargo.toml . # outside the crate itself
```

## The reachability result, per candidate

`W` = normal `[dependencies]` of a crate under `crates/dormouse-*` (in the product).
`d` = `[dev-dependencies]`. `b` = a bench package. `–` = the `burn-fused` facade only,
and **the facade is referenced by zero manifests in this repo** (fate table's own finding).

| crate | src | t/b | other | live caller OUTSIDE itself | fate row | verdict |
|---|---:|---:|---:|---|---|---|
| `burn-attnres` | 2730 | 72 | — | **`crates/dormouse-core/Cargo.toml:32`, W** | REFERENCE | **KEEP — FINDING, row is STALE** |
| `burn-mhc` | 1437 | 0 | — | **`crates/dormouse-core/Cargo.toml:27`, W** | REFERENCE | **KEEP — FINDING, row is STALE** |
| `burn-rope` | 1139 | 0 | — | 7 sites, all `benches/` + 1 example (`d`,`b`) | **WIRE** | KEEP |
| `burn-parcae` | 317 | 0 | — | 0 (`–`) | **WIRE** | KEEP |
| `burn-swiglu` | 187 | 0 | — | 1 (`benches/cpu_probe/src/main.rs:74`, `b`) | REFERENCE | KEEP |
| `burn-eggroll` | 327 | 0 | — | 0 (`–`) | REFERENCE | KEEP — the survivor of the `es` pair |
| `burn-ptrn` | 399 | 0 | — | 0 (`–`) | REFERENCE | KEEP — reason to stay is live (§ below) |
| `burn-es` | 345 | 0 | 99 | 0 (`–`) | REFERENCE | **DELETE** |
| `burn-sct` | 2737 | 458 | 244 | 0 non-dev (`d`+`b` only) | **DELETE (blocked)** | **DELETE** |

**Two of the brief's seven candidates have a LIVE PRODUCT CALLER.** `burn-attnres` and
`burn-mhc` are normal `[dependencies]` of `crates/dormouse-core` and both are in the root
`Cargo.lock`. `burn-mhc` has been A/B'd to a 3-seed tie (`git log`: "mHC is a 3-seed TIE —
6.413/6.313/6.370"). The fate table's rows for both still read `REFERENCE`; they are
**stale, not wrong about reachability at the time**. Fixing them is a separate, cheap commit.

## The delete set, and the evidence for each line of it

### `burn-sct` — 3439 lines, 16 files

- `grep -rn "burn-sct" --include=Cargo.toml .` outside itself → 3 edges:
  `vendor/dormouse-fused/Cargo.toml:11` (workspace member),
  `crates/burn-spectral/Cargo.toml:33` (**`[dev-dependencies]`**),
  `benches/Cargo.toml:20` (bench package), plus the generated facade.
- `grep -rn "burn_sct" --include=*.rs` outside itself → **11 sites, all in
  `crates/burn-spectral/examples/tsct_diag.rs` and `benches/src/main.rs:209`.**
  **Zero** in `crates/`, in any `src/`, in any config, in any test that gates a product path.
- `crates/dormouse-core/src/param.rs:6` imports `burn_spectral::SpectralLinear`, **not**
  `burn_sct` — the product's TSCT is the `burn-spectral` one.
- `research/decisions/class-b-2026-10-01.md:513` recommends delete; the crate's own
  `burn-sct/README.md:3-9` says "**Not in the dormouse build** … Recommendation: **DELETE**";
  `docs/architecture/library-crate-fate.md:61` already carries the verdict.
- **COST ESTIMATE CORRECTION (a finding).** Fate:159-172 and the decision sheet:504 both
  price this at "**~10 lines**". Measured: **85 of 1771 lines** of `tsct_diag.rs` carry
  `sct_a`/`sct_b`/`burn_sct` (56 are mechanical `sct_x: None,` fillers, 2 struct fields,
  2 whole `match` arms at 22 lines each, 16 need hand edits, 9 sit inside the dropped arms),
  plus a 14-line block in `benches/src/main.rs`, plus `gpu-gate.sh:74,79-81`. **~10 is off by
  an order of magnitude**, and the 16 hand sites include two `&[str; N]` marker tables whose
  arity changes (7→5, 10→8) and two `#[test]` fixture lists naming `heads.3.sct_a.weight`.

### `burn-es` — 444 lines, 4 files

- `grep -rn "burn-es" --include=Cargo.toml .` outside itself → 2: workspace member + facade.
  Zero code callers.
- **The crate declares itself a duplicate.** `burn-es/src/lib.rs:14-19`: *"NOTE: this crate
  keeps a convenience `eggroll_mutate` … the dedicated `burn-eggroll` crate implements the
  paper's normalized form … **Prefer burn-eggroll for real ES loops**; the function here
  stays for quick experiments **and its test**."*
- **A live doc orders the deletion and its precondition is met.**
  `docs/research/2026-09-27-adopt-vs-port.md:588`: *"the source settles the σ convention,
  **then delete this one**."* The convention IS settled — `burn-eggroll` is the paper's
  `(σ/√r)·A·Bᵀ` and survives; `burn-es` documents `σ·A·Bᵀ` and defers. Same conclusion at
  `docs/architecture/library-crate-fate.md:75` ("They overlap each other; one should go when
  it starts") and `adopt-vs-port.md:388-390` (the disagreement is `√r`).
- One citation needs straightening: `crates/burn-spectral/src/lib.rs:50` attributes the
  ternary `0.7·mean` dead zone to "(burn-es convention)". The audit that found it
  (`docs/reviews/spectral-stack-audit-2026-10-01.md:65`, F13) already grades it **(c) — no
  external reference**. Deleting `burn-es` makes the attribution dangle, so the honest
  wording is "no external reference exists", which is the same edit F13 asks for.

### Why `burn-ptrn` is KEPT despite the brief listing it

`library-crate-fate.md:184` gives a live `file:line` reason to stay: ADR-0013 deleted the
Q-head the crate scores rollouts with, and `AGENTS.md` §3.6 item 3 says the selection rule
must be re-specified first. Fate's own rule is that a documented reason to stay is exactly
the exception that beats "delete is the default". `adopt-vs-port.md:587` argues "*~20 lines
it would take to write are cheaper than maintaining 399*" — that is an argument for **when**
it is wanted, not a decision to delete. **Two live docs in mild tension = a §1.7 defect to
report, not a licence to delete.** Cost of being wrong is asymmetric: deleting costs a
re-port; keeping costs 399 dead lines.

## Counter to the brief's arithmetic

The brief says "**13 in the build, 7 in none, + `burn-sct`**". Measured: 21 crates total
(20 under `crates/` + the facade), 13 in the build, **7 not in the build** =
`{sct, eggroll, es, parcae, ptrn, rope, swiglu}` — `sct` is *inside* the seven, not beside it.
Of those seven, **two (`attnres`, `mhc`) are `W`, wired into `dormouse-core`** and so are not
in the "not in the build" set at all; `rope` and `parcae` carry fate dispositions of **WIRE**;
`swiglu` has a live bench caller. **The honest delete set is 2 crates, not 8.**

## Line counter

| crate | src | t/b | other | total | files |
|---|---:|---:|---:|---:|---:|
| `burn-sct` | 2737 | 458 | 244 | **3439** | 16 |
| `burn-es` | 345 | 0 | 99 | **444** | 4 |
| **total** | 3082 | 458 | 343 | **3883** | **20** |

Against the lane's target of −2000: **exceeded by 3883, 194 %.** No test file is deleted —
`burn-es` and `burn-sct` are the only two candidates that have them, and both go whole.

## The gate, and one correction to it

`cargo check --workspace --all-targets` inside the library will fail on
`burn-fused-benches` / `cpu-probe` / `launch-probe` for reasons that predate this lane
(they need `--features cuda`). The gate to run is **CI's own command**,
`fused-library.yml:62`: the same check plus
`--exclude burn-fused-benches --exclude cpu-probe --exclude launch-probe`.
Both are run per batch, plus `gen_facade.py --check` and
`cargo check -p dormouse-train --features dormouse-train/cuda` in the root.

---

## Execution (2026-10-02, branch wt/cut2)

Landed as three code commits plus the docs/manifest commit:

- `f13ec4e` delete(vendor): burn-sct — 3439 lines, 16 files.
- `24834c6` refactor(burn-spectral): drop the sct8/sct16 arms from tsct_diag —
  127 lines removed (85 carrying sct_a/sct_b, of which 56 were `x: None`
  fillers; the two 22-line `match` arms; 16 hand sites including the two
  arity-changed marker tables, 7→5 FFN / 10→8 Muon+).
- `5a37f94` delete(vendor): burn-es — 444 lines, 4 files; facade regenerated,
  `gen_facade.py --check` green; the dangling `(burn-es convention)`
  attribution in `burn-spectral/src/lib.rs:50` is now "no external reference
  exists" (same edit F13 in `docs/reviews/spectral-stack-audit-2026-10-01.md`
  asked for).
- docs: `library-crate-fate.md` rows for both crates now read
  **DELETED 2026-10-02**; `docs/papers/tsct.md` carries a tombstone for §5;
  `docs/glossary.md` now names `burn_spectral::SpectralLinear`; and
  `crates/dormouse-core/src/param.rs:1`'s stale "via burn-sct" header is
  `via burn-spectral`. `tools/check_doc_refs.py` carries four KNOWN_DEAD
  entries so dated audits naming the deleted paths stay green.

Gates (CPU): library workspace check per CI's command
(`cargo check --workspace --all-targets --exclude burn-fused-benches
--exclude cpu-probe --exclude launch-probe`) — green, 4m38s;
`cargo check -p dormouse-train --features dormouse-train/cuda` —
[result below, filled in after the run]; `python3 tools/check_doc_refs.py` —
0 missing; `npm run check` in docs-site — [below].

Line count against target: 3439 + 444 = **3883 removed** (CI reset plus
tsct_diag/bench trims bring the tracked total to ~3885+).
