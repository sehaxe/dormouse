# docs/

**Read four files. Everything else is a record.**

| file | what it is |
|---|---|
| [`HANDBOOK.md`](HANDBOOK.md) | the whole project in one document — the rule, the recipe, the code, the formats |
| [`measurements.md`](measurements.md) | the dated table of runs, with the window and the budget |
| [`glossary.md`](glossary.md) | the vocabulary, and where a document and the code disagree |
| [`../AGENTS.md`](../AGENTS.md) | the rules a person or an agent must follow when changing the repo |

Rendered, with the clips and the reference lists:
**<https://sehaxe.github.io/dormouse/docs/>**

## Working documents

| directory | what |
|---|---|
| [`guides/`](guides/) | how to run a thing on this box — data provenance, compile, the GPU lock |
| [`protocols/`](protocols/) | how an experiment is run and how its result is stated (A/B, oracle, determinism) |
| [`adr/`](adr/) | one decision per file, with a status; a retracted decision stays, marked |
| [`papers/`](papers/) | the external references we argue against, as PDFs |

These are maintained. If one is wrong, fix it or delete it.

## Everything in `records/` is evidence, not documentation

A record says what was true on a date. It is never rewritten, never maintained,
and never the place to learn how the project works today. If a record and the
handbook disagree, the handbook is later.

| directory | count | what |
|---|---:|---|
| [`records/reviews/`](records/reviews/) | 145 | one lane, read from the inside: what was built and what it measured |
| [`records/research/`](records/research/) | 94 | literature notes and one-off measurements |
| [`records/architecture/`](records/architecture/) | 20 | design notes for a moment in the plan |
| [`records/reports/`](records/reports/) | 4 | build and purchase write-ups |
| [`records/decisions/`](records/decisions/) | 2 | pre-ADR decision notes |
| [`records/archive/`](records/archive/) | 5 | history, not truth |

Every record is indexed by date on one page of the site:
<https://sehaxe.github.io/dormouse/docs/archive/>.

## Rules for adding a document

- A record is named `YYYY-MM-DD-slug.md` and lives under `records/`.
- A guide is named for the task and lives in `guides/`; keep it current or delete it.
- If it changes how the project works, it goes in `HANDBOOK.md`, not in a new file.
- A number in a record names its window and its budget; without them it is an
  anecdote.
