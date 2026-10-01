# docs/ — everything except the rulebook

One root, one naming scheme. The repo root holds exactly three documents:
[`README.md`](../README.md) (status and quick start), [`AGENTS.md`](../AGENTS.md)
(the agent rulebook — rules, machine facts, and every retraction), and
[`CONTEXT.md`](../CONTEXT.md) (the system map). **This directory holds
everything else**, and `docs/glossary.md` keeps its path because every document in
the repo links it.

| directory | what lives there | how to name it |
|---|---|---|
| `adr/` | one record per decision, with a Status line. Retracted decisions stay | `NNNN-slug.md` |
| `protocols/` | the instruments: what a number must satisfy before it may be written down | as the instrument is known |
| `research/` | dated findings — what was measured, what was concluded | `YYYY-MM-DD-slug.md` |
| `reviews/` | hostile reads of a lane, written to attack it | `YYYY-MM-DD-slug.md` |
| `decisions/` | the owner's decision sheets: inputs to a decision, not records of one | `YYYY-MM-DD-slug.md` |
| `architecture/` | the design documents: program plan, model floor, mixture arms, post-training, bf16 | as the design is known |
| `guides/` | how to run a thing on this box | as the task is known |
| `papers/` | transcriptions of other people's formulas, **the PDFs themselves**, and `provenance.tsv` (file → source → sha256) | papers are `<arXivID>-<slug>.pdf` |
| `glossary.md` | the vocabulary, and the live list of where a document and the code disagree | — |
| `archive/` | [history, not truth](archive/README.md) — never cite it in an argument | unchanged |

Two rules worth stating because they are the ones that get broken:

- **A note is a snapshot of a belief on a date.** Nothing in `research/` is
  maintained. Where a note and `AGENTS.md` disagree, the rulebook is later.
- **The site is generated from these paths.** `docs-site/tools/manifest.mjs` is
  the information architecture as data; adding a document means adding a line
  there, and `npm run check` fails on a link that resolves to nothing.

If you cannot tell where a new document belongs: it is a measurement →
`research/`; an attack on someone else's work → `reviews/`; something a reader
needs before trusting a number → `protocols/`; something superseded → leave it in
git, do not add it.