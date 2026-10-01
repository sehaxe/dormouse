# IA — the information architecture of the dormouse knowledge base

This file is the design artifact for `docs-site/`. It was written **against the
actual filenames**, not against a wishlist; the section list below is what the
repo turned out to contain. The site's own landing page is a short orientation —
this file is the thing a maintainer edits when they add a document.

**184 pages, built from 100 canonical files, 0 broken links.**

## Running it

```bash
cd docs-site
npm install --no-audit --no-fund   # 233 MB, once
npm run dev                        # ingest + astro dev  → http://localhost:4321
npm run build                      # ingest + astro build → dist/ (static)
npm run preview                    # serve dist/          → http://localhost:4321
npm run check                      # internal links + 9 content spot-checks
```

Every `run` script ingests **first**, so a page can never be stale relative to
its source. `npm run dev` is the one to read the site with; `npm run check` is
the gate and it must be green before a commit.

## The one rule that shapes everything

**The site is a view, not a second source of truth.** Every page in
`src/content/docs/` is generated at build time by `tools/ingest.mjs` from a
canonical file that lives where it always lived (`README.md`, `AGENTS.md`,
`docs/`, `research/`, `.bulba/`). The generator reads the canonical file,
injects Starlight frontmatter, rewrites cross-file links, and writes into a
gitignored directory. There is:

- **no duplicated fact** — a number lives in one file and appears once,
- **no deletion**, **no move**, and **no edit** to any canonical file by this lane,
- **exactly one generation step**, so a site page is never stale.

This repo has a recorded history of twin-file disasters (two agents writing
different versions of one findings file at the same path). A knowledge base that
copies 387 files is the same failure with better typography. Hence: no copies.

Two consequences worth stating, because they are the price of the design:

1. **A generated page is as good as its canonical file.** If a source file has
   five `##` headings of the same rank, the site's right rail will show what the
   file shows, not what a hand-written page would show.
2. **Site structure is data, not files.** Section membership and sidebar order
   live in `tools/manifest.mjs`, which is the file to edit when a document is
   added. The manifest is committed; the pages it produces are not.

## Vocabulary

A section is a **module**. It has:

- an **interface** — one index page whose whole job is to let a reader reach
  everything in the section from one screen;
- **implementation** — the ingested documents, deep, unmodified, each with its
  own right-rail table of contents.

The test of the IA: *a reader who learns one index can reach every document in
that section.* Where that is false, the index is wrong, not the reader.

## Sections

Order is the sidebar order (`order` in the manifest).

| # | module | pages | what its index has to do |
|---|---|---|---|
| 1 | [Start here](#1-start-here) | 13 | what this is, what is measured, how to work here |
| 2 | [Architecture](#2-architecture) | 14 | the model, the words, the presets |
| 3 | [Protocols](#3-protocols) | 6 | how a number is allowed to exist |
| 4 | [ADR](#4-adr) | 24 | the decision sequence, and what reversed |
| 5 | [Research](#5-research) | 72 | evidence, newest first, in three kinds |
| 6 | [Reviews](#6-reviews) | 24 | hostile reads, not descriptions |
| 7 | [Tooling](#7-tooling) | 21 | one page per program, from its own header |
| 8 | [Archive & retracted](#8-archive--retracted) | 8 | what was claimed and taken back |
| — | landing + 404 | 2 | authored by this site |
| | **total** | **184** | |

### 1. Start here — `start-here` (13 pages)

| page | source |
|---|---|
| index | generated |
| Status | `README.md` — `## STATUS`, sections (0) and (a) |
| Quick start | `README.md` — `## Quick start` |
| CLI flags | `README.md` — `### Every flag on train` |
| Working rules | `README.md` — `## Working rules (CONTRIBUTING)` |
| License | `README.md` — `## License` |
| The agent rulebook | `AGENTS.md` (whole: rules, machine facts, measured / retracted / broken / next) |
| Program plan | `docs/PLAN.md` |
| Working notes (4) | `.bulba/{memory,plan,architecture,goal}.md` |

`AGENTS.md` is here and not in Protocols on purpose: it is *how to work in this
repo*, the same job as the CONTRIBUTING rules, and it is the only document that
carries both the machine facts and the current status. It is 1 242 lines; for a
reference that is a virtue, which is why it gets a page rather than a summary.

The four `.bulba/` notes are here because the repo's own documents cite them by
path — a retraction note is `.bulba/memory.md:15`, not a paraphrase. The other
four `.bulba/` files (`alphaxiv-analysis.md`, `fused_backprop_notes.md`,
`skills-index.md`, `stsb.md`) are agent scratch that nothing in the repo cites,
and are **deliberately not ingested**. Nothing was deleted; they are simply not
in the site.

### 2. Architecture — `architecture` (14 pages)

Interface: the model, the words, and the presets.

`CONTEXT.md` (the system map) · `docs/glossary.md` (the vocabulary and its live
list of where a document and the code disagree) · `README.md — Architecture`
(the ASCII model walk) · `— The vendored kernel library` · `— Presets` ·
`— The .dmexp model file` · `docs/design-minimal.md` (the floor: what 90 % of
the code would be) · `docs/design-review-model-2026-09-27.md` ·
`docs/PLAN-minimal-core.md` (what leaves `dormouse-core`) ·
`docs/mixture-arms.md` (three priced training arms) ·
`docs/library-crate-fate.md` (the 28 vendored crates and what each is for) ·
`POST_TRAINING.md` and `bf16_KERNEL_PLAN.md` (Russian, as written).

The glossary is the reason this module needs an interface at all: "fused",
"arm", "sidecar", "iteration", "group" and "Engram" each have more than one
live meaning, and a reader who does not hold the glossary will misread every
other page — including this one.

### 3. Protocols — `protocols` (6 pages)

The measurement instrument. If a number in this repo exists, one of these pages
is the reason it is allowed to be written down.

- `docs/AB-PROTOCOL.md` — the A/B queue and the rule that decides an arm
- `docs/ORACLE.md` — which defect class each existing test in `burn-kda` /
  `burn-gdn2` can and cannot catch
- `docs/ORACLE-TIERS.tsv` — **rendered as 127 blocks**, one per comparison,
  because 128 rows × 6 columns of sentences is unreadable as a table; the
  `.tsv` itself is linked as the artifact `tools/oracle_gate.py` reads
- `docs/VERIFICATION.md` — the specification for the missing end-to-end golden
- `research/papers/determinism-2026-09-30.md` — the cross-process seed harness

The stress protocol (`--stress`, `crates/dormouse-train/src/stress.rs`) has no
canonical prose document; it is reached through the CLI flags page and its
source, not through a page invented here.

### 4. ADR — `adr` (24 pages)

`docs/adr/0001…0023`, one page each, in numeric order. The sidebar label is the
record number and the page title is the decision statement, because the two are
different jobs: ADR-0001's first line is "BPB at fixed budget is the score" and
ADR-0023's is "ADR-0023: the inference export…", so neither heading is
self-identifying and a sidebar of full titles is unusable.

The index page names the sequence and says the only thing worth saying — three
records are worth reading *because* they were reversed: **0003** (the whole-loop
fused op measured slower than burn and was deleted), **0012** and **0014** (the
sparse-attention arm emitted garbage indices on burn 0.22.0-pre.4 and was cut
with its crate).

### 5. Research — `research` (72 pages)

Three kinds of document, three sidebar groups, because they are read for three
different reasons:

- **Notes** (`research/*.md`, 41 files) — auto-indexed, **newest first**. The
  filenames are `YYYY-MM-DD-topic.md`, so the date is parsed from the name, not
  from a frontmatter field. Reverse-chronological is the order a reader wants: a
  note from yesterday is evidence about today's tree, a note from 2026-09-21 is
  history. Undated files (a synthesis, a question list) sort after the dated
  ones instead of interleaving by filename.
- **Papers & specs** (`research/papers/*.md`, 26 pages) — our transcription of a
  formula, or our reading of an upstream reference. Read when you need the
  equation, not when you need the status.
- **Decisions** (`research/decisions/*.md`) — the owner's decision sheets, which
  are inputs to a decision, not records of one.

`research/papers/determinism-2026-09-30.md` is the one file placed in
**Protocols** instead: the determinism harness *is* the measurement instrument.
It gets one page, in Protocols; the Papers glob skips it, because one page per
canonical file is the rule.

### 6. Reviews — `reviews` (24 pages)

`research/reviews/*.md`, 23 files, newest first. Kept separate from Research
because a review is a *hostile* artifact: it was written to attack a lane, and
reading it as a description of the code is a mistake. The index says so, and
names the case where later measurement dissolved a review's finding
(`ab-wave-2026-10-01`).

### 7. Tooling — `tooling` (21 pages)

- `README.md — Tooling` (the canonical table), `— Data pipeline`,
  `— Performance`, `docs/BUILD-TIME.md`
- **16 pages, one per program in `tools/`**, each generated from **the tool's
  own header comment**. This is the only page in the site with no canonical
  `.md`: its source is the file it describes, so the description cannot drift
  from the code, because there is only one copy and it is the source.

Two of these are load-bearing for the rules rather than convenient: the
[build lock](https://github.com/sehaxe/dormouse/blob/main/tools/build_lock.sh)
exists because five concurrent builds pre-froze this workstation, and
[the worktree script](https://github.com/sehaxe/dormouse/blob/main/tools/wt.sh)
because a shared checkout makes `cargo test` a lottery.

### 8. Archive & retracted — `archive` (8 pages)

A retraction is only useful if it is findable, so:

- the unverified-claims register — `README.md` §(b)
- the known-wrong register — `README.md` §(c)
- **the 5-gram question** — `README.md` §(d), which keeps every quality number honest
- `docs/audit-2026-09-25.md` · `docs/FINDINGS-2026-09-29.md` ·
  `docs/fused-verification-2026-09-23.md` · `docs/rmsnorm-kernel-2026-09-30.md`

**The full retracted-claims register is `AGENTS.md` §3.2 and it is not copied
here.** It is linked from the index instead, because a copy is a copy that goes
stale — and it is 400 lines inside a 1 242-line rulebook whose §1.4 is the rule
the whole site is built to obey. This is the one place the brief's "a page per
retraction" is answered with a link instead, and the reason is that the
alternative is the twin-file failure this site exists to avoid.

The four documents are whole because their value is precisely that they are a
point-in-time reading with every number attached.

## Decisions taken, and why

- **Starlight, hand-rolled rather than `npm create astro@latest`.** The
  scaffolder is interactive, pulls a template over the network, and writes files
  this lane would immediately rewrite. A pinned `package.json` + three config
  files is deterministic and reviewable. Pinned: `astro@7.3.5`,
  `@astrojs/starlight@0.42.5`.
- **English default, nothing translated.** The brief asked for
  `defaultLocale: 'en'`; it is **not set**, because in Starlight it is the switch
  for i18n *routing*, and setting it makes Astro look for an `i18n` content
  collection this site does not have — a warning on every build for no
  behavioural difference. Without it the root locale is English and URLs are
  unprefixed, which is what was asked for. `POST_TRAINING.md`,
  `bf16_KERNEL_PLAN.md`, `docs/PLAN-2026-09-29.md` and `docs/FINDINGS-2026-09-29.md`
  are Russian and are rendered as written.
- **Pagefind search on, no React integration.** Search is the only thing that
  makes 184 pages navigable without the sidebar; a React integration would be a
  dependency this repo has no use for.
- **Raw GitHub links for anything not ingested.** Non-`.md` repo-relative links
  (`tools/wt.sh`, `crates/…/model.rs`, `LICENSE`) are rewritten to
  `github.com/sehaxe/dormouse/blob/main/<path>`. A code site whose source links
  all dangle is worse than no links, and this is six lines of code rather than a
  second copy of the source file.
- **"Edit page" points at the canonical file.** Each generated page carries
  `editUrl: <repo>/edit/main/<its source>`, so a reader who edits a page from
  the site edits the real file. A PR against the generated page would be a
  duplicate.
- **No CI.** The brief says so, and this box runs a 100k-step production
  training; a site build does not get a job.

## Link rewriting, and its honest failure count

Markdown is rendered as-is except for link targets. For each link the generator
resolves the target against the *source file's directory*, then:

1. **into the site** if the resolved repo-relative path is ingested → the site
   URL, anchor preserved;
2. **into the site, cross-slice** if the target is `#anchor` and this page is a
   slice of a file whose heading lives in a *different* slice (README.md is 845
   lines and eight pages; `](#presets)` on the (b) page has to reach the Presets
   page);
3. **to GitHub** if the resolved path exists in the repo and is not ingested;
4. **otherwise left exactly as written**, and counted.

Measured at the time of writing (`npm run ingest` prints this on every build):

| outcome | count |
|---|---|
| rewritten into the site | 34 |
| rewritten to github.com | 5 |
| **left as written (unresolved)** | **0** |
| **broken after the rewrite** (checked in `npm run check`, 35 716 links) | **0** |

The zero is not luck and not silence: the ingest step prints the unresolved
count and exits non-zero on anything it cannot place, and `tools/check.mjs`
walks every `href` in `dist/` and fails if one does not resolve. What is left
untouched is prose that merely looks like a link — `/home/sehaxe/logs/…`,
`~/logs/…`, `file.rs:15`, `arXiv:…`, `https://…` — which is correct, since those
are text naming a path, not navigation inside a site.

## What this IA deliberately does not do

- **No page per source file of code.** 184 pages of prose already.
- **No "all measurements" page.** `AGENTS.md` §3.1 is the registry and it is
  linked as a page, not copied into a table that would go stale silently.
- **No summary pages of any document.** A summary is a second copy with fewer
  words and no source link. The index pages navigate; they do not paraphrase.
- **No translation, no reformatting of the source text, no "cleanup".** The
  content is the deliverable; the site is the glass.