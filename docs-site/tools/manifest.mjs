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
[the agent rulebook](/start-here/agents/) §1 first; it exists because breaking it
cost a run, a week, or a claim. [Where every document lives](/start-here/docs-map/)
is the map of this site back to the files it is generated from.

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

A retraction is only useful if it is findable, so this section exists. Two of its
pages are **slices of [the agent rulebook](/start-here/agents/)** — §3.2, every
retracted claim, and §3.3, what is broken or open — because that register moved
out of README when README became a front page, and a register of retractions
belongs to the rulebook (§1.4) rather than to a marketing page. They are slices,
not copies: there is still exactly one §3.2.

The quality bar itself is in AGENTS.md §2.6 (the anchor readings, and the
statement that no held-out number in the archive beats a 5-gram byte counter).

The four audits below are audits of the tree at a date, kept whole because their
value is precisely that they are a point-in-time reading with every number
attached. **What is in this section's canonical directory, and the rule that
nothing in it may be cited as evidence, is the archive's own README**
([what is in here](/archive/what-is-archived/)) — \`docs/archive/\` also holds the
pruned history the site does not publish.`,
	},
];

export const dirIndex = {
	'docs/adr/': '/adr/',
	'research/': '/research/',
	'research/reviews/': '/reviews/',
	'research/papers/': '/research/papers/',
	'research/decisions/': '/research/decisions/',
	// docs/ is the one root every document lives under. A link to a directory
	// lands on that directory's section rather than on a 404.
	'docs/': '/start-here/docs-map/',
	'docs/architecture/': '/architecture/',
	'docs/protocols/': '/protocols/',
	'docs/research/': '/research/notes/',
	'docs/reviews/': '/reviews/',
	'docs/decisions/': '/research/decisions/',
	'docs/papers/': '/research/papers/',
	'docs/guides/': '/tooling/',
	'docs/archive/': '/archive/',
	'vendor/dormouse-fused/crates/': '/architecture/library-crate-fate/',
};

// Where a LINK to a file that is only ingested as slices should go: the file's
// primary page. README.md is 845 lines and is cut into eight pages; the reader
// who clicks "README.md" wants the status, which is where it opens.
export const srcIndex = {
	'README.md': '/start-here/status/',
};

export const pages = [
	// ── Start here ───────────────────────────────────────────────────────────
	// README.md was rewritten on 2026-10-01 as a front page (93ba203): eight
	// sections instead of the dev-log's twenty. The slices follow the sections
	// that exist; five pages whose ONLY source was a deleted section are gone
	// rather than re-anchored onto text that is not what it says - see the
	// site's IA.md for which content moved where.
	{ src: 'README.md', out: 'start-here/what-is-this', slice: { from: /^## What is this$/, to: /^## Results$/ }, title: 'What this is', nav: 'What this is' },
	{ src: 'README.md', out: 'start-here/status', slice: { from: /^## Results$/, to: /^## Quick start$/ }, title: 'Status: the measured numbers', nav: 'Status' },
	{ src: 'README.md', out: 'start-here/quick-start', slice: { from: /^## Quick start$/, to: /^## Architecture$/ }, title: 'Quick start' },
	{ src: 'README.md', out: 'start-here/license', slice: { from: /^## License$/ }, title: 'License' },
	{ src: 'AGENTS.md', out: 'start-here/agents', title: 'The agent rulebook', nav: 'The agent rulebook' },
	// CONTRIBUTING.md is the practice-level companion to AGENTS.md §1: which
	// command answers which question, and the worktree / build-lock rules. It
	// defers to AGENTS.md for the rules themselves rather than restating them,
	// so the two cannot drift into disagreeing.
	{ src: 'CONTRIBUTING.md', out: 'start-here/contributing', title: 'Contributing: the commands', nav: 'Contributing' },
	{ src: 'docs/architecture/PLAN.md', out: 'start-here/program-plan', title: 'Program plan' },
	// The map of docs/ itself: where a new document goes and how it is named.
	{ src: 'docs/README.md', out: 'start-here/docs-map', title: 'Where every document lives', nav: 'The docs map' },
	// Working notes the repo's own documents reference by path.
	{ src: '.bulba/memory.md', out: 'start-here/notes/memory', title: 'Working memory', nav: 'Working memory' },
	{ src: '.bulba/plan.md', out: 'start-here/notes/plan', title: 'Working plan', nav: 'Working plan' },
	{ src: '.bulba/architecture.md', out: 'start-here/notes/architecture', title: 'Working notes: architecture', nav: 'Notes: architecture' },
	{ src: '.bulba/goal.md', out: 'start-here/notes/goal', title: 'Working notes: the goal', nav: 'Notes: the goal' },

	// ── Architecture ─────────────────────────────────────────────────────────
	{ src: 'CONTEXT.md', out: 'architecture/context', title: 'The system map', nav: 'The system map' },
	{ src: 'docs/glossary.md', out: 'architecture/glossary' },
	{ src: 'README.md', out: 'architecture/model', slice: { from: /^## Architecture$/, to: /^## The vendored kernel library/ }, title: 'The model walk', nav: 'The model walk' },
	{ src: 'README.md', out: 'architecture/vendored-library', slice: { from: /^## The vendored kernel library/, to: /^## Documentation$/ }, title: 'The vendored kernel library', nav: 'The kernel library' },
	{ src: 'docs/architecture/design-minimal.md', out: 'architecture/design-minimal' },
	{ src: 'docs/architecture/mixture-arms.md', out: 'architecture/mixture-arms' },
	{ src: 'docs/architecture/library-crate-fate.md', out: 'architecture/library-crate-fate' },
	{ src: 'docs/architecture/post-training.md', out: 'architecture/post-training' },
	{ src: 'docs/architecture/bf16-plan.md', out: 'architecture/bf16-plan' },

	// ── Protocols ────────────────────────────────────────────────────────────
	{ src: 'docs/protocols/AB-PROTOCOL.md', out: 'protocols/ab-protocol' },
	{ src: 'docs/protocols/ORACLE.md', out: 'protocols/oracle' },
	{ src: 'docs/protocols/ORACLE-TIERS.tsv', out: 'protocols/oracle-tiers', title: 'Oracle tiers', nav: 'Oracle tiers', render: 'tsv' },
	{ src: 'docs/protocols/VERIFICATION.md', out: 'protocols/verification', title: 'Verification' },
	{ src: 'docs/protocols/determinism.md', out: 'protocols/determinism' },

	// ── Papers ────────────────────────────────────────────────────────────────
	// The provenance register is a .tsv, not prose: one row per original, naming
	// the arXiv version and its sha256. Rendered like the oracle tiers.
	{ src: 'docs/papers/provenance.tsv', out: 'research/papers/provenance', title: 'Paper provenance', nav: 'Provenance (sha256)', render: 'tsv' },

	// ── Archive ──────────────────────────────────────────────────────────────
	// The three registers the old README carried as §(b)/(c)/(d) moved into
	// AGENTS.md when README became a front page. Same content, one copy, and the
	// rulebook is where a retraction belongs anyway (§1.4).
	{ src: 'AGENTS.md', out: 'archive/retracted', slice: { from: /^## 3\.2 Retracted/, to: /^## 3\.3 / }, title: 'Every retracted claim', nav: 'Retracted claims' },
	{ src: 'AGENTS.md', out: 'archive/broken', slice: { from: /^## 3\.3 /, to: /^## 3\.4 / }, title: 'BROKEN or KNOWN-WRONG', nav: 'BROKEN or KNOWN-WRONG' },
	{ src: 'docs/archive/audit-2026-09-25.md', out: 'archive/audit-2026-09-25' },
	{ src: 'docs/archive/findings-2026-09-29.md', out: 'archive/findings-2026-09-29' },
	{ src: 'docs/archive/fused-verification-2026-09-23.md', out: 'archive/fused-verification-2026-09-23' },
	{ src: 'docs/archive/rmsnorm-kernel-2026-09-30.md', out: 'archive/rmsnorm-kernel-2026-09-30' },
	// The archive README: what may not be cited, and what earns a place here.
	{ src: 'docs/archive/README.md', out: 'archive/what-is-archived', title: 'History, not truth', nav: 'What is in here' },

	// ── Tooling ──────────────────────────────────────────────────────────────
	{ src: 'docs/guides/build-time.md', out: 'tooling/build-time' },

	// ── Guides ───────────────────────────────────────────────────────────────
	// Restored 2026-10-01. Four pages whose only source was a README section
	// the README-rewrite (93ba203) deleted: the preset table, the flag table,
	// the .dmexp format, and the working-rules checklist. Each is generated
	// from its canonical file in docs/guides/, which is the ONE place a guide
	// lives - these lines index that directory, they are not a second copy.
	{ src: 'docs/guides/presets.md', out: 'tooling/presets', title: 'Presets: what each one costs and is for', nav: 'Presets' },
	{ src: 'docs/guides/cli.md', out: 'tooling/cli', title: 'Every flag on train', nav: 'CLI flags' },
	{ src: 'docs/guides/dmexp.md', out: 'tooling/dmexp', title: 'The model file: .dmexp', nav: 'The .dmexp file' },
	{ src: 'docs/guides/first-pr.md', out: 'tooling/first-pr', title: 'Your first pull request', nav: 'First PR' },
	// The 2026-10-01 narrative wave: four "how to actually use this" guides
	// assembled from the protocols, the rulebook and the reviews - no new
	// numbers, every figure carried with its source.
	{ src: 'docs/guides/train-your-first.md', out: 'tooling/train-your-first', title: 'Train your first model', nav: 'First run' },
	{ src: 'docs/guides/determinism.md', out: 'tooling/determinism', title: 'Determinism: why --seed was the easy part', nav: 'Determinism' },
	{ src: 'docs/guides/ab-testing.md', out: 'tooling/ab-testing', title: 'A/B testing: how an arm is judged here', nav: 'A/B testing' },
	{ src: 'docs/guides/precision.md', out: 'tooling/precision', title: 'Precision on this backend: why bf16 is slower than fp32', nav: 'Precision' },
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
	{ dir: 'docs/research', out: 'research/notes', match: /\.md$/, order: 'date-desc', nav: compact },
	// Papers: the transcriptions, plus the provenance register - a .tsv
	// (file -> source -> sha256) that renders the way the oracle tiers do.
	{ dir: 'docs/papers', out: 'research/papers', match: /\.md$/, nav: topic },
	{ dir: 'docs/decisions', out: 'research/decisions', match: /\.md$/, nav: topic },
	{ dir: 'docs/reviews', out: 'reviews', match: /\.md$/, order: 'name-desc', nav: compact },
	// TRANSIENT (2026-10-01): three documents the ab-wave lane is writing right
	// now still live in research/. They move into docs/reviews/ and
	// docs/decisions/ when that lane lands; then these two globs are deleted,
	// which is why they are named here rather than left implicit.
	{ dir: 'research/reviews', out: 'reviews', match: /\.md$/, order: 'name-desc', nav: compact },
	{ dir: 'research/decisions', out: 'research/decisions', match: /\.md$/, nav: topic },
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