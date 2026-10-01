// The manifest: the site structure as data.
//
// THIS FILE is what to edit when a document is added, moves sections, or needs a
// title the document does not give itself. It is the IA in code — see IA.md for
// why the IA exists and what it argues.
//
// Shape:
//   sections  the modules. Each becomes a directory in src/content/docs/ with a
//             generated index page whose job is to let a reader reach every
//             document in it from one screen.
//   pages     explicit pages: { src, out, title?, nav?, slice?, render? }
//             `src`  repo-relative path of the CANONICAL file (read-only)
//             `out`  path under src/content/docs/ WITHOUT the .md extension
//             `slice` cut a file by heading markers { from, to } — both regexes,
//                     `to` exclusive. A slice never edits the source and never
//                     duplicates it: the parts are cut into disjoint pages.
//             `render` 'table' for a TSV/CSV that is data rather than prose.
//   globs     everything in a directory, indexed automatically. Research notes
//             are ordered newest-first because their filenames start with a date.
//   dirIndex  repo directory -> site URL, so `docs/adr/` in a link lands on the
//             ADR index instead of 404.
//   authored  pages this site owns outright (the landing page). One live copy.

export const repo = 'https://github.com/sehaxe/dormouse';
export const branch = 'main';

export const sections = [
	{
		slug: 'start-here',
		title: 'Start here',
		nav: 'Start here',
		order: 1,
		intro: `## What this is

**dormouse** is a byte-level language-model trainer that fits on one workstation:
a 16 GB consumer GPU plus host RAM. Every mechanism it has is judged by
**held-out bits-per-byte at a fixed step budget**, and this site is the record of
that judgement — including the parts where the judgement went against the code.

If you are here for one number, it is on the [status page](/start-here/status/).
If you are here because you are about to *build* on this repo, read
[the working rules](/start-here/working-rules/) and
[the agent rulebook](/start-here/agents/) first; both exist because breaking them
cost a run, a week, or a claim.

**The culture on display here is unusual and deliberate.** This project publishes
retractions next to the claims they retract, marks a number *unverified* when the
gate that would verify it is not runnable, and refuses to call anything
"bit-for-bit" without naming the external file the reference came from
([ADR-0020](/adr/0020-oracle-discipline/)). A knowledge base that hid that
would be a marketing site, and the only thing a reader of a training log really
needs is to know which lines to trust.`,
	},
	{
		slug: 'architecture',
		title: 'Architecture',
		nav: 'Architecture',
		order: 2,
		intro: `## The model, the words, and the presets

Start with [the system map](/architecture/context/) — \`CONTEXT.md\` is written to
be read end to end and to be enough to hold the system in your head. Then
[the model walk](/architecture/model/) for the actual dataflow, and
[the glossary](/architecture/glossary/) for the vocabulary.

The glossary is not optional reading for this section. **"fused", "arm",
"sidecar", "iteration", "group" and "Engram" each have more than one live
meaning in this repo**, and a reader who does not hold the glossary will
misread every other page — including this one. The glossary also ends with a live
list of the places where a document and the code disagree about a name or a
default, which is the fastest way to find out whether a page you are reading is
current.`,
	},
	{
		slug: 'protocols',
		title: 'Protocols',
		nav: 'Protocols',
		order: 3,
		intro: `## The instrument

Every number in this project exists because a protocol allowed it to be written
down. This section is those protocols: the A/B rule that decides whether a
mechanism lives, the oracle tiers that say what a "verified" comparison was
actually compared against, and the determinism harness that made
\`--seed\` measurable.

Read these before trusting a claim from any other page.`,
	},
	{
		slug: 'adr',
		title: 'Architecture decision records',
		nav: 'ADR',
		order: 4,
		intro: `## Every recorded decision, including the retracted ones

Twenty-three records, in numeric order. Each is short and each has a **Status:**
line saying whether it is in force, and — for the interesting ones — what later
evidence did to it.

Three are worth reading *because* they were reversed:

- [ADR-0003 — fused: measure, then decide](/adr/0003-fused-measure-then-decide/) —
  the whole-loop fused op measured **slower** than burn and was deleted
  (superseded by ADR-0009).
- [ADR-0012](/adr/0012-msa-broken-on-pre4/) and
  [ADR-0014](/adr/0014-msa-cut/) — the sparse-attention arm emitted garbage
  indices on burn 0.22.0-pre.4 and was cut with its crate.

A record that was reversed is not a record that failed. It is the reason the
replacement is not being re-litigated.`,
	},
	{
		slug: 'research',
		title: 'Research',
		nav: 'Research',
		order: 5,
		intro: `## Evidence, newest first

Three kinds of document, kept apart because they are read for three different
reasons:

- **Notes** are dated (\`YYYY-MM-DD-topic.md\`) and indexed **newest first**. A
  note from yesterday is evidence about today's tree; a note from 2026-09-21 is
  history. Nothing here is maintained — it is what was believed, with its
  measurements.
- **Papers & specs** are one page per paper: our transcription of a formula, or
  our reading of an upstream reference implementation. Reference material — read
  when you need the equation, not when you need the status.
- **Decisions** are the owner's decision sheets: inputs to a decision, not
  records of one.

> A note is a snapshot of a belief on a date. Where a note and
> [the agent rulebook](/start-here/agents/) disagree, **the rulebook is later**:
> it is the one that records retractions.`,
	},
	{
		slug: 'reviews',
		title: 'Reviews',
		nav: 'Reviews',
		order: 6,
		intro: `## Hostile reads of the lanes

Adversarial reviews, written to **attack** a lane rather than describe it. They
are the reason several arms were cut, several gates rewritten, and one fused
kernel shipped unverified for a week before the review caught it.

Read them as claims, not as descriptions: a review is a snapshot of a lane's
state at the moment someone went looking for what was wrong with it, and some
findings were later dissolved by a better measurement
([the A/B seed wave](/reviews/ab-wave-2026-10-01/) is the example — three of its
measurements corrected the lanes it reviewed).`,
	},
	{
		slug: 'tooling',
		title: 'Tooling',
		nav: 'Tooling',
		order: 7,
		intro: `## The programs, from the tool table

One page per tool, generated from **the tool's own header comment** — so a
description cannot drift from the code it describes, because there is only one
copy and it is the source.

Two of these are load-bearing for the rules rather than convenient: the
[build lock](https://github.com/sehaxe/dormouse/blob/main/tools/build_lock.sh)
exists because five concurrent builds pre-froze this workstation, and
[the worktree script](https://github.com/sehaxe/dormouse/blob/main/tools/wt.sh)
exists because a shared checkout makes \`cargo test\` a lottery.`,
	},
	{
		slug: 'archive',
		title: 'Archive & retracted',
		nav: 'Archive',
		order: 8,
		intro: `## What was claimed, and what was taken back

A retraction is only useful if it is findable, so this section exists: the
unverified-claims register, the known-wrong register, and the five-gram question
that keeps every quality number honest.

**The full retracted-claims register is a subsection of
[the agent rulebook](/start-here/agents/) §3.2** — it is not duplicated here,
because a copy is a copy that goes stale. It is linked rather than copied for the
same reason the whole site is generated rather than written.

The four documents below are audits of the tree at a date, kept whole because
their value is precisely that they are a point-in-time reading with every number
attached.`,
	},
];

export const dirIndex = {
	'docs/adr/': '/adr/',
	'research/': '/research/',
	'research/reviews/': '/reviews/',
	'research/papers/': '/research/papers/',
	'research/decisions/': '/research/decisions/',
	'vendor/burn-fused/crates/': '/architecture/library-crate-fate/',
};

// Where a LINK to a file that is only ingested as slices should go: the file's
// primary page. README.md is 845 lines and is cut into eight pages; the reader
// who clicks "README.md" wants the status, which is where it opens.
export const srcIndex = {
	'README.md': '/start-here/status/',
};

export const pages = [
	// ── Start here ───────────────────────────────────────────────────────────
	{ src: 'README.md', out: 'start-here/status', slice: { from: /^## STATUS$/, to: /^### \(b\)/ }, title: 'Status', nav: 'Status' },
	{ src: 'README.md', out: 'start-here/quick-start', slice: { from: /^## Quick start$/, to: /^## The model file/ }, title: 'Quick start' },
	{ src: 'README.md', out: 'start-here/cli-flags', slice: { from: /^### Every flag on `train`$/, to: /^## Tooling$/ }, title: 'Every flag on train', nav: 'CLI flags' },
	{ src: 'README.md', out: 'start-here/working-rules', slice: { from: /^## Working rules/, to: /^## Docs$/ }, title: 'Working rules', nav: 'Working rules' },
	{ src: 'README.md', out: 'start-here/license', slice: { from: /^## License$/ }, title: 'License' },
	{ src: 'AGENTS.md', out: 'start-here/agents', title: 'The agent rulebook', nav: 'The agent rulebook' },
	{ src: 'docs/PLAN.md', out: 'start-here/program-plan', title: 'Program plan' },
	// Working notes the repo's own documents reference by path.
	{ src: '.bulba/memory.md', out: 'start-here/notes/memory', title: 'Working memory', nav: 'Working memory' },
	{ src: '.bulba/plan.md', out: 'start-here/notes/plan', title: 'Working plan', nav: 'Working plan' },
	{ src: '.bulba/architecture.md', out: 'start-here/notes/architecture', title: 'Working notes: architecture', nav: 'Notes: architecture' },
	{ src: '.bulba/goal.md', out: 'start-here/notes/goal', title: 'Working notes: the goal', nav: 'Notes: the goal' },

	// ── Architecture ─────────────────────────────────────────────────────────
	{ src: 'CONTEXT.md', out: 'architecture/context', title: 'The system map', nav: 'The system map' },
	{ src: 'docs/glossary.md', out: 'architecture/glossary' },
	{ src: 'README.md', out: 'architecture/model', slice: { from: /^## Architecture$/, to: /^## The vendored kernel library/ }, title: 'The model walk', nav: 'The model walk' },
	{ src: 'README.md', out: 'architecture/vendored-library', slice: { from: /^## The vendored kernel library/, to: /^## Presets$/ }, title: 'The vendored kernel library', nav: 'The kernel library' },
	{ src: 'README.md', out: 'architecture/presets', slice: { from: /^## Presets$/, to: /^## Quick start$/ } },
	{ src: 'README.md', out: 'architecture/model-file-dmexp', slice: { from: /^## The model file/, to: /^### Every flag/ }, title: 'The model file: .dmexp', nav: 'The .dmexp model file' },
	{ src: 'docs/design-minimal.md', out: 'architecture/design-minimal' },
	{ src: 'docs/design-review-model-2026-09-27.md', out: 'architecture/design-review-model' },
	{ src: 'docs/PLAN-minimal-core.md', out: 'architecture/plan-minimal-core' },
	{ src: 'docs/mixture-arms.md', out: 'architecture/mixture-arms' },
	{ src: 'docs/library-crate-fate.md', out: 'architecture/library-crate-fate' },
	{ src: 'POST_TRAINING.md', out: 'architecture/post-training' },
	{ src: 'bf16_KERNEL_PLAN.md', out: 'architecture/bf16-plan' },

	// ── Protocols ────────────────────────────────────────────────────────────
	{ src: 'docs/AB-PROTOCOL.md', out: 'protocols/ab-protocol' },
	{ src: 'docs/ORACLE.md', out: 'protocols/oracle' },
	{ src: 'docs/ORACLE-TIERS.tsv', out: 'protocols/oracle-tiers', title: 'Oracle tiers', nav: 'Oracle tiers', render: 'tsv' },
	{ src: 'docs/VERIFICATION.md', out: 'protocols/verification', title: 'Verification' },
	{ src: 'research/papers/determinism-2026-09-30.md', out: 'protocols/determinism' },

	// ── Archive ──────────────────────────────────────────────────────────────
	{ src: 'README.md', out: 'archive/unverified', slice: { from: /^### \(b\)/, to: /^### \(c\)/ }, title: 'Implemented but UNVERIFIED', nav: 'Implemented but UNVERIFIED' },
	{ src: 'README.md', out: 'archive/broken', slice: { from: /^### \(c\)/, to: /^### \(d\)/ }, title: 'BROKEN or KNOWN-WRONG', nav: 'BROKEN or KNOWN-WRONG' },
	{ src: 'README.md', out: 'archive/five-gram', slice: { from: /^### \(d\)/, to: /^### \(e\)/ }, title: 'Has any checkpoint beaten a 5-gram byte counter?', nav: 'The 5-gram question' },
	{ src: 'docs/audit-2026-09-25.md', out: 'archive/audit-2026-09-25' },
	{ src: 'docs/FINDINGS-2026-09-29.md', out: 'archive/findings-2026-09-29' },
	{ src: 'docs/fused-verification-2026-09-23.md', out: 'archive/fused-verification-2026-09-23' },
	{ src: 'docs/rmsnorm-kernel-2026-09-30.md', out: 'archive/rmsnorm-kernel-2026-09-30' },

	// ── Tooling ──────────────────────────────────────────────────────────────
	{ src: 'README.md', out: 'tooling/overview', slice: { from: /^## Tooling$/, to: /^## Data pipeline$/ }, title: 'The tool table', nav: 'The tool table' },
	{ src: 'README.md', out: 'tooling/data-pipeline', slice: { from: /^## Data pipeline$/, to: /^## Performance$/ } },
	{ src: 'README.md', out: 'tooling/performance', slice: { from: /^## Performance$/, to: /^## Working rules/ } },
	{ src: 'docs/BUILD-TIME.md', out: 'tooling/build-time' },
];

// Everything the manifest does not enumerate, discovered and ordered here.
//
// `nav` is the SIDEBAR label and `title` the page heading, and they are not the
// same thing: a 68-item sidebar of full titles is unusable, and a page whose
// heading is a bare filename is unreadable. `listLabel` says how the generated
// index line should read when it needs both.
const topic = (f) =>
	f
		.replace(/\.md$/, '')
		.replace(/^\d{4}-\d{2}-\d{2}-/, '')
		.replace(/[-_]+/g, ' ');
const dated = (f) => /^\d{4}-\d{2}-\d{2}/.exec(f)?.[0] ?? '';
const compact = (f) => (dated(f) ? `${dated(f)} · ${topic(f)}` : topic(f));

export const globs = [
	{
		// One page per ADR. The heading is the page title ("BPB at fixed budget is
		// the score"); the record number is the sidebar label, because a reader
		// scanning the sidebar wants 0016, not the whole sentence.
		dir: 'docs/adr',
		out: 'adr',
		match: /^\d{4}-/,
		nav: (f) => `ADR-${f.slice(0, 4)}`,
		listLabel: 'nav+title',
	},
	{ dir: 'research', out: 'research/notes', match: /\.md$/, order: 'date-desc', nav: compact },
	{ dir: 'research/papers', out: 'research/papers', match: /\.md$/, nav: topic },
	{ dir: 'research/decisions', out: 'research/decisions', match: /\.md$/, nav: topic },
	{ dir: 'research/reviews', out: 'reviews', match: /\.md$/, order: 'name-desc', nav: compact },
	// A tool page IS the tool's own header comment. See tools/ingest.mjs.
	{
		dir: 'tools',
		out: 'tooling',
		match: /^[\w.-]+\.(sh|py)$/,
		render: 'header',
		slug: (f) => f.replace(/\.(sh|py)$/, ''),
		nav: (f) => f.replace(/\.(sh|py)$/, ''),
	},
];

// Pages this site owns. The landing page and nothing else: everything else is a
// view of a canonical file.
export const authored = ['index.md', '404.md'];

export default { repo, branch, sections, pages, globs, dirIndex, srcIndex, authored };