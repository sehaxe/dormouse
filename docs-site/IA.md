# IA — the information architecture of the dormouse knowledge base

This file is the design artifact for `docs-site/`. It was written **against the
actual filenames**, not against a wishlist; the section list below is what the
repo turned out to contain. The site's own landing page is a short orientation —
this file is the thing a maintainer edits when they add a document.

## The one rule that shapes everything

**The site is a view, not a second source of truth.** Every page in
`src/content/docs/` is generated at build time from a canonical file that lives
where it always lived (`README.md`, `docs/`, `research/`, `AGENTS.md`). The
generator reads the canonical file, injects Starlight frontmatter, rewrites
cross-file links, and writes into a gitignored directory. There is:

- **no duplicated fact** — a number lives in one file and appears once,
- **no deletion**, **no move**, and **no edit** to any canonical file by this lane,
- **exactly one generation step** (`npm run ingest`, wired into `prebuild`), so a
  site page is never stale relative to its source.

This repo has a recorded history of twin-file disasters (two agents writing
different versions of one findings file at the same path). A knowledge base that
copies 387 files is the same failure with better typography. Hence: no copies.

Two consequences worth stating, because they are the price of the design:

1. **A generated page is as good as its canonical file.** If a source file has
   five `#` headings of the same rank, the site's right rail will show what the
   file shows, not what a hand-written page would show.
2. **Site structure is data, not files.** Section membership and sidebar order
   live in `tools/manifest.mjs`, which is the file to edit when a document is
   added. The manifest is committed; the pages it produces are not.

## Vocabulary

A section is a **module**. It has:

- an **interface** — one index page, `sidebar.label`-titled, whose whole job is
  to let a reader reach everything in the section from one screen;
- **implementation** — the ingested documents, deep, unmodified, each with its
  own right-rail table of contents.

The test of the IA: *a reader who learns one index can reach every document in
that section.* Where that is false, the index is wrong, not the reader.

## Sections

Order is the sidebar order (`order` in the manifest). 173 pages total.

### 1. Start here — `start-here` (7 pages)

The module interface is "what is this, what is true now, and how do I work
here". Entry order:

| page | source |
|---|---|
| index | generated |
| status | `README.md` — frontmatter block + `## STATUS` (incl. the (a)–(e) snapshot tables) |
| quick start | `README.md` — `## Quick start` |
| CLI flags | `README.md` — `## Every flag on train` |
| working rules | `README.md` — `## Working rules (CONTRIBUTING)` |
| the agent rulebook | `AGENTS.md` (whole file: rules, machine facts, measured / retracted / broken / next) |
| program plan | `docs/PLAN.md` |

`AGENTS.md` is here and not in Protocols on purpose: it is *how to work in this
repo*, which is the same job as the CONTRIBUTING rules, and it is the only
document that carries both the machine facts and the current status. It is long
(≈700 lines); that is a virtue for a reference and is why it gets its own page
rather than a summary.

### 2. Architecture — `architecture` (11 pages)

Interface: the model, the words, and the presets.

`CONTEXT.md` (the system map) · `docs/glossary.md` (the vocabulary, and its live
list of where a document and the code disagree) · `README.md — Architecture`
(the ASCII model walk) · `README.md — The vendored kernel library` · `README.md —
Presets` · `README.md — The .dmexp model file` · `docs/design-minimal.md` (the
floor: what 90% of the code would be) · `docs/design-review-model-2026-09-27.md`
· `docs/PLAN-minimal-core.md` (what leaves `dormouse-core`) ·
`docs/mixture-arms.md` (three priced training arms) · `docs/library-crate-fate.md`
(the 28 vendored crates and what each is for) · `POST_TRAINING.md` and
`bf16_KERNEL_PLAN.md` (Russian, as written).

The glossary is the reason this module has an interface at all: "fused", "arm",
"sidecar", "iteration" and "Engram" each have more than one live meaning, and a
reader who does not hold the glossary will misread every other page.

### 3. Protocols — `protocols` (5 pages)

The measurement instrument. If a number in this repo exists, one of these pages
is the reason it is allowed to.

- `docs/AB-PROTOCOL.md` — the A/B queue and the rule that decides an arm
- `docs/ORACLE.md` — which defect class each existing test in `burn-kda` /
  `burn-gdn2` can and cannot catch
- `docs/ORACLE-TIERS.tsv` — **rendered as a table** (the raw file is data; the
  site renders the register as the table it is meant to be read as, and links
  to the `.tsv` on GitHub as the artifact `tools/oracle_gate.py` reads)
- `docs/VERIFICATION.md` — the specification for the missing end-to-end golden
- `research/papers/determinism-2026-09-30.md` — the cross-process seed harness

The stress protocol (`--stress`, `crates/dormouse-train/src/stress.rs`) has no
canonical prose document; it is reached through the CLI flags page and its
source, not through a page invented here.

### 4. ADR — `adr` (23 pages)

`docs/adr/0001…0023`, one page each, in numeric order, retitled
`ADR-NNNN · <first heading>` because the headings are not self-identifying
(ADR-0001's first line is "BPB at fixed budget is the score"; ADR-0023's is
"ADR-0023: the inference export…"). The ADR series is the one place where the
index page is nearly empty on purpose: the index names the sequence, links every
record, and says the only thing worth saying — decisions are recorded including
the retracted ones, and the retracted decisions are `0003`, `0012` and `0014`.

### 5. Research — `research` (70 pages)

Three kinds of document, three sidebar groups, because they are read for three
different reasons:

- **Dated notes** (`research/*.md`, 41 files) — auto-indexed, **newest first**.
  The filenames are `YYYY-MM-DD-topic.md`, so the date is parsed from the name,
  not from a frontmatter field. Reverse-chronological is the order a reader wants:
  a note from yesterday is evidence about today's tree, a note from 2026-09-21 is
  history.
- **Papers & specs** (`research/papers/*.md`, 27 files) — one page per paper, our
  transcription, or our reading of an upstream spec. These are reference material:
  read when you need the formula, not when you need the status.
- **Decisions** (`research/decisions/*.md`, 1 file) — the owner's decision sheets,
  which are inputs to a decision, not records of one.

### 6. Reviews — `reviews` (23 pages)

`research/reviews/*.md`, auto-indexed newest first. Kept separate from Research
because a review is a *hostile* artifact: it was written to attack a lane, and
reading it as a description of the code is a mistake. The index says so.

### 7. Tooling — `tooling` (10 pages)

- `README.md — Tooling` (the canonical table), `README.md — Data pipeline`,
  `README.md — Performance`, `docs/BUILD-TIME.md`
- **one page per tool**, generated from the tool's own leading comment block
  (`tools/*.sh`, `tools/*.py`). This is the only "page" in the site with no
  canonical `.md`: its source is the file it describes, so the description cannot
  drift from the code and there is nothing to duplicate.

### 8. Archive & retracted — `archive` (10 pages)

Every retracted claim gets a findable page, because the README's retraction
tables and AGENTS.md §3.2 are the two places a retraction is recorded and both
are long documents in which a retraction is one row among hundreds.

- the retracted-claims register: `AGENTS.md — §3.2 Retracted`, sliced verbatim
- `README.md — (b) What is implemented but UNVERIFIED`
- `README.md — (c) What is BROKEN or KNOWN-WRONG`
- `README.md — (d) has any checkpoint beaten a 5-gram byte counter`
- `AGENTS.md — §3.3 Broken, open, or undocumented`, sliced verbatim
- `docs/audit-2026-09-25.md` · `docs/FINDINGS-2026-09-29.md` ·
  `docs/fused-verification-2026-09-23.md` · `docs/rmsnorm-kernel-2026-09-30.md`

Slicing a file by its `##` headings is the one transformation here that needs
justifying, because it makes a generated page a *part* of a canonical file. It
is still one live copy: `README.md` is the only place the words exist, the
generator cuts them at a heading boundary and nothing else, and the full file is
also ingested whole (into Start here / Archive as applicable) so no content is
reachable only through the slicer.

## Decisions taken, and why

- **Starlight, hand-rolled rather than `npm create astro@latest`.** The
  scaffolder is interactive, pulls a template over the network, and writes files
  this lane would immediately rewrite. A pinned `package.json` + three config
  files is deterministic and reviewable.
- **English default, nothing translated.** `defaultLocale: 'en'`, no locale
  routing. `POST_TRAINING.md`, `bf16_KERNEL_PLAN.md`, `docs/PLAN-2026-09-29.md`
  and `docs/FINDINGS-2026-09-29.md` are Russian and are rendered as written.
  Translating 387 documents is a project, not a build step; pretending with a
  machine pass would produce a second source of truth in a second language.
- **Pagefind search on, no React integration.** Search is the only thing that
  makes 173 pages navigable without the sidebar; a React integration would be a
  dependency this repo has no use for.
- **Raw GitHub links for anything not ingested.** Non-`.md` repo-relative links
  (`tools/wt.sh`, `vendor/burn-fused/crates/…`, `crates/…/model.rs`) are
  rewritten to `https://github.com/sehaxe/dormouse/blob/main/<path>`. A code
  site whose source links all dangle is worse than no links, and this is 6 lines
  of code rather than a second copy of the source file.
- **No CI.** The brief says so, and this box runs a 100k-step production
  training; a site build does not get a job.

## Link rewriting, and its honest failure count

Markdown is rendered as-is except for link targets. For each link the generator
resolves the target against the *source file's directory*, then:

1. **resolves into the site** if the resolved repo-relative path is ingested →
   rewritten to the site URL, relative to the current page, anchor preserved;
2. **resolves to a repo file** (any extension, not ingested) → rewritten to a
   GitHub blob URL on `main`;
3. **otherwise** → left exactly as written.

Case 3 is a real number and it is printed by `npm run ingest`, not hidden:

| category | count |
|---|---|
| links rewritten into the site | *see `npm run ingest` output — reported every build* |
| links rewritten to GitHub | *idem* |
| **unresolved, left as-is** | *idem* |

The expected content of case 3 is: absolute filesystem paths
(`/home/sehaxe/logs/…`, `/tmp/opencode/…`), `~/logs/…` log references, `file:line`
prose references, and `arXiv:`/`https:` external links (which are correctly left
alone). Those are *not* broken links — they are prose that happens to look like a
path — so the honest statement is: **the ingest step reports how many link-shaped
things it could not map, and a reader who clicks one lands on GitHub, where it
was already pointing.** A relative link to a document that exists in neither the
site nor GitHub would be a genuine break; the count of those is reported
separately as `broken` by the ingest step, and it was **0** at the time of
writing.

## What this IA deliberately does not do

- **No page per source file of code.** 173 pages of prose already.
- **No "all measurements" page.** §3.1 of AGENTS.md is the registry and it is
  linked as a page, not copied into a table that would go stale silently.
- **No summary pages of any document.** A summary is a second copy with fewer
  words and no source link. The index pages navigate; they do not paraphrase.
- **No translation, no reformatting of the source text, no "cleanup".** The
  content is the deliverable; the site is the glass.