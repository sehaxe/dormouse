# Documentation coverage — `dormouse-core` and `dormouse-data`, 100% of the public surface

Lane: owner directive «каждая строка кода расписана», in the professional form
the lane agreed: **every public item and every module carries a doc comment
that says what it is, what invariant it maintains, what makes it fail LOUD, and
what it costs.** Statement-by-statement commentary is explicitly NOT wanted — it
is noise that rots.

Scope: `crates/dormouse-core` (9 679 lines of `src/`, comments included) and
`crates/dormouse-data` (1 265 lines of lib + 2 063 of bins). `dormouse-train/src/lib.rs` is
**off limits** — an active lane owns it — so the trainer's public surface is
still ungated by this.

Commit base: `a327466`. Gate: `#![warn(missing_docs)]` +
`#![warn(rustdoc::broken_intra_doc_links)]` in both `lib.rs`, proven by
`RUSTFLAGS="-D warnings" cargo doc --no-deps -p dormouse-core -p dormouse-data`
(exit 0, 0 warnings).

---

## 1. Numbers, before and after

The audit is line-based over `src/` (`pub fn|struct|enum|trait|const|static|type|mod|use`
plus struct fields and enum variants), counting an item documented when the
preceding non-blank line is `///` or a `#[doc]` attribute.

| | before | after |
|---|---:|---:|
| public items, `dormouse-core` | 167 | 167 |
| ... documented | 109 (65.3%) | **142 (85.0%)** |
| public items, `dormouse-data` (lib + 3 bins) | 22 | 22 |
| ... documented | 16 (72.7%) | **22 (100%)** |
| both crates | 189 / 125 (66.1%) | **189 / 164 (86.8%)** |
| source files | 24 | 24 |
| files with a `//!` header | **20 / 24** | **24 / 24** |

The four files that had **no module doc at all** were `config/mod.rs`,
`config/override.rs`, `config/validation.rs` and `config/loader.rs` — the whole
config seam, which is exactly where a reader most needs one and where a wrong
value is most expensive. Every other module already had a header; the work there
was filling in the items they left bare.

**The remaining 13.8% is 26 `pub use` re-exports and `pub mod` lines, which
rustdoc's own `missing_docs` does not require and which are documented at their
definition site** — `pub use burn_mor::MoRRouter` carries a doc explaining what
the router is and why it is re-exported here, but a script that looks for `///`
immediately above the `use` line counts it as bare. rustdoc's judgement is the
one the gate enforces, and it is now **zero** undocumented public items.

By crate, after:

| crate | pub items | documented | files with `//!` |
|---|---:|---:|---:|
| `dormouse-core` | 167 | 142 | 20 / 20 |
| `dormouse-data` | 22 | 22 | 4 / 4 |

Per-file comment density after (doc-comment lines / total lines), for the record:

| file | lines | doc lines |
|---|---:|---:|
| `config/schema.rs` | 591 | 403 |
| `loop_block.rs` | 2 436 | 559 |
| `aux.rs` | 834 | 231 |
| `model.rs` | 671 | 207 |
| `param.rs` | 404 | 177 |
| `future_byte.rs` | 633 | 171 |
| `moe.rs` | 621 | 146 |
| `routing.rs` | 557 | 127 |
| `mixture_probe.rs` | 421 | 114 |
| `probe.rs` | 171 | 106 |
| `gr.rs` | 381 | 102 |
| `act_quant.rs` | 340 | 86 |
| `attention.rs` | 127 | 81 |
| `dormouse-data/src/lib.rs` | 1 265 | 330 |
| `config/override.rs` | 141 | 63 |
| `config/validation.rs` | 239 | 52 |
| `config/loader.rs` | 181 | 48 |

---

## 2. What the gate is, exactly

`crates/dormouse-core/src/lib.rs` and `crates/dormouse-data/src/lib.rs` carry:

```rust
#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
```

Proof command, green:

```
$ RUSTFLAGS="-D warnings" cargo doc --no-deps -p dormouse-core -p dormouse-data
   Finished `dev` profile [unoptimized + debuginfo] target(s) in 3m 28s
exit=0
```

**0 warnings.** Before this lane the same command produced **108 warnings** on
`dormouse-core` (95 × `missing_docs`) and **8** across `dormouse-data`'s bins
(5 × unclosed HTML tag, 1 × unresolved link, 1 × unused doc comment). All are
gone.

`.github/` was **not** touched: the docs CI job is landing separately today and
owns the enforcement. This lane supplies the lint attributes and the command;
the job wires it.

`cargo test -p dormouse-core -p dormouse-data --lib` green — nothing but comments
changed, which the test run is there to prove rather than assert.

---

## 3. Defects found while documenting — NONE FIXED, all flagged

Four found, three of them real. All are in files this lane edited, but all are
behaviour changes and this lane is docs-only, so each is recorded here with
`file:line` rather than fixed.

### 3.1 `config/override.rs` — four schema fields are unreachable from `--set`

`crates/dormouse-core/src/config/override.rs:95` (the `match key.as_str()`).

`DormouseConfig` has 40+ fields; the override match covers 36. Missing:
**`use_situ`, `use_attnres`, `use_mhc`, `mhc_streams`**.

The failure mode is LOUD, not silent — `--set use_situ=true` returns
`Err("unknown config key \"use_situ\"")` — so nothing trains differently from
what was asked. But those four are reachable ONLY from a preset TOML, which is
an inconsistency with every other field and a trap for anyone doing an A/B from
the command line. This is the arm set of §1.2's "the cheapest big lever"
thinking: SiTU-GLU, AttnRes and mHC are all queue rows, and all three are
currently only settable by editing a file under `configs/`.

Fix is four arms in that match, ~6 lines. Not done here (docs lane).

### 3.2 `config/loader.rs` — a broken preset in the cwd is skipped SILENTLY

`crates/dormouse-core/src/config/loader.rs:49` (`try_file`).

In the SEARCH path, a candidate that exists but fails to read or fails to parse
is dropped with **both error modes discarded** (`read_to_string(p).ok()?` then
`parse_str(&s).ok()`). So `--preset small` with a syntax error in
`./configs/small.toml` falls through to the repo copy and runs a config nobody
asked for, with no warning on any stream.

By ADR-0019's own classification this is a **SILENT** fallback, which the
doctrine calls a defect. The explicit-path branch (`load_config("./x.toml")`)
has no such hole — it returns the parse error. The asymmetry is the finding: the
same function treats the same failure loudly or silently depending on how the
preset was spelled.

Cheapest fix: collect the failures and mention them in the terminal `Err`,
which already names every place it tried. ~5 lines. Not done here.

### 3.3 `param.rs` — the non-CUDA bf16 arm is a SILENT fallback

`crates/dormouse-core/src/param.rs:187` (`forward`'s `#[cfg(not(feature = "cuda"))]`
block).

With `bf16_compute` on and no `cuda` feature, the bf16 matmul op does not exist
and the code runs the fp32 quant path instead — **the right answer, no
counter, no log line**. ADR-0019 has three SILENT sites open and this is a
fourth; it is not in that enumeration because it predates the audit or was
missed by it. In practice a CPU run has `bf16_compute` off, so the arm is
unreachable today — but it is one `#[cfg]` away from being a config that
silently trains in the wrong precision.

Recorded here rather than fixed. The fix is a `COUNTED` bump in `probe`, which
is a new constant and therefore an `N_ARMS` bump.

### 3.4 `act_quant.rs` — the attention upgrade is unconditional, and that is load-bearing

Not a defect, recorded because the doc comment now says it and a reader should
know it is a decision, not an oversight: `ActFormat::attn()` maps
`Fp4 -> Int(8)` and `Int(b) -> Int(b.max(8))`. So `--act-quant fp4` has **never**
run 4-bit attention, and every fp4 number before 2026-09-28 is invalid for that
reason *and* for the second one (`fp4` was not e2m1 then). The comment at
`act_quant.rs:43` (`ActFormat::attn`) already carried this; it is repeated at
`config::schema::ActQuant::Fp4` because that is where someone reads the flag.

---

## 4. Names flagged, not renamed

Per the lane's own rule — *if a doc comment would just translate Rust to
English, the NAME is wrong; flag it instead* — three items where the honest
documentation had to spend a paragraph explaining what the name does not say:

1. **`ExpertFFN::gate_up`** — a `d_model -> d_ffn` projection when
   `use_situ` is off, `d_model -> 2 * d_ffn` when it is on. It is named after two
   projections and is one projection in the default configuration. Renaming it
   changes the checkpoint's parameter path, so it stays; the doc now says
   exactly this at the field and at `ExpertFFN::new`.
2. **`GatedResidual::wu`** — named for the report's Eq. 31 symbol, not for its
   shape (it is `r -> nr*d`, the UP half of a bottleneck). Same reason: the
   report's symbols are the citation.
3. **`DormouseConfig::vocab`** — 256, fixed by the byte-level task, not a tuning
   knob. It is a field because `serve`/`generate` need it. A reader who sees a
   `vocab` field assumes it is swept.

None renamed: each would move a checkpoint parameter path or lose a paper
citation, and this lane is docs-only.

---

## 5. What was deliberately NOT written

* **No statement-by-statement commentary.** The lane brief rules it out and it
  rots: a comment that says what a line does is wrong the moment the line moves,
  and it is the comment nobody re-reads. Every comment added here is about an
  invariant, a source, a cost, or a failure mode.
* **No `dormouse-train` docs.** Off limits by the brief. Its public surface is
  the largest ungated one left, and the same two lint attributes would apply
  there unchanged — that is the obvious next lane.
* **No `.github/` change.** The docs CI job lands separately; this lane supplies
  the command, not the wiring.
* **No behaviour change of any kind.** Every diff hunk in this lane is a comment.
  The four defects in §3 are the cost of that rule and they are written down
  instead.
