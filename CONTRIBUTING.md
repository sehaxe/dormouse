# Contributing

dormouse is a byte-level language model that trains on one workstation, and it
is built to be checked rather than believed. The rules are in
[`AGENTS.md` §1](AGENTS.md) — read that section once, in full; this page is the
way in from outside and it does not restate the rulebook. If the two disagree,
AGENTS.md is right and this page is a bug (§1.7: fix one, never invent a third
rule).

You do not need a GPU to contribute: the setup and the gates below are CPU-only.

## What you need

- `git`, and [`uv`](https://docs.astral.sh/uv/) — it installs Python 3.12, so
  you do not manage an interpreter.
- A few GB of disk (the torch wheel is most of it). No corpus and no checkpoint:
  the tests build their own tiny models.
- Linux, which is what CI and the training lane run. Other platforms are
  untested; if the suite passes on yours, say so in a PR.

## Set up

```bash
git clone https://github.com/sehaxe/dormouse.git
cd dormouse/port
uv sync                     # CPU-only. The card extra is `uv sync --extra gpu`.
.venv/bin/python -c "import dormouse; print(dormouse.__file__)"
```

The last line must print a path under `port/src/`. If it does not, the install
did not take and nothing below will mean anything.

Two lanes live in this repository. `port/` is the live one: config, model,
trainer, post-training, every number on record. `crates/` is the Rust/burn lane,
**frozen since 2026-11-07** — it compiles and its tests run, but no new work
lands there. New work goes to `port/`.

**Which branch.** The live Python lane is `backup/main-local-2026-11-08`;
`origin/main` is a different, Rust-era lineage until the owner merges them
([`README.md`](README.md) says which is which). Open pull requests against the
live lane, not `main`.

If you run a training job on a shared box, run **one** at a time: two legal
processes have collided here and written garbage weights into a checkpoint
([`AGENTS.md` §1.5](AGENTS.md)).

## The gates

Run these from the repository root unless the command changes directory. A pull
request is not ready while one of them is red.

| command | what it proves |
|---|---|
| `bash scripts/gate_fast.sh` | `ruff check`; `ruff format --check` on the files changed since `origin/main`; every repository path a document names exists; no preset sets a config field that nothing reads; the claims ledger (report-only today); `cargo check --workspace` (prints `SKIP` when `cargo` is absent, so Rust is not required — if you do have Rust, this is the slowest step) |
| `cd port && CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q` | the CPU test suite — the behaviour gate |
| `cd port && .venv/bin/ruff check src scripts tests` | the lint half alone, when you want it in a second |
| `python3 tools/check_doc_refs.py` | a path you cite in a comment, a docstring or a document exists |

The fast gate prints `GATE PASSED` or `GATE FAILED`; a failure names the step,
and that step is the finding. It does **not** run `pytest`, so both are
required: `gate_fast` for the static claims, the suite for the behaviour.

The whole-tree format check is `bash scripts/gate_format.sh`. Until the legacy
sweep lands it is red on files written before the rule, which is why
`gate_fast` checks only your diff: do not reformat files you did not change.
`ruff` is the arbiter of style, not this page — see
[`docs/guides/style.md`](docs/guides/style.md).

The CUDA-marked tests need a card and skip without one; they are not part of the
contributor requirement. On a clean branch head the CI jobs `python-test`,
`python-lint`, `python-gates` and `site` must be green; `claims-ledger`,
`cuda-compile` and the Rust `lint` job report without blocking, and the reason
is written above each one in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## The one habit: a claim names its artifact

Everything here exists to make a claim checkable. A number is a claim, and a
claim names its artifact — the config, the commit, the date, and the log or test
that produced it; a number without those is deleted, not softened. A claim that
turns out to be wrong is retracted in public and **the retraction stays
published** next to the original, because a reader who cannot see what was
believed on a date cannot judge what is believed now. And a mechanism earns its
place only by beating its own removal on held-out BPB at a fixed step budget
with at least three seeds per arm — *A/B or death*; a tie deletes the mechanism.
The long form, and the retractions that taught us this, are in
[`AGENTS.md` §1.2 and §1.4](AGENTS.md) and
[`docs/protocols/AB-PROTOCOL.md`](docs/protocols/AB-PROTOCOL.md).

## Proposing a change: issue → plan → gate

**1. Issue, before the pull request.** A cold PR that changes a number or a
mechanism will be sent back for its evidence. Pick the template that asks for
yours:

- [`bug.md`](.github/ISSUE_TEMPLATE/bug.md) — the artifact first: the smallest
  command, the log path, the commit SHA, and the backend it ran on.
- [`perf.md`](.github/ISSUE_TEMPLATE/perf.md) — the baseline's command and log,
  the machine, and the decision rule written **before** the run.
- [`feature.md`](.github/ISSUE_TEMPLATE/feature.md) — what it unblocks, and what
  would have to be true for it to be deleted.
- [`mechanism_proposal.md`](.github/ISSUE_TEMPLATE/mechanism_proposal.md) — a
  model mechanism needs its removal arm named, or it is not a proposal yet.

**2. Plan.** A paragraph in the issue, before the code: what changes, what could
regress, and which gate would catch that regression. For a mechanism, the plan
also names the control arm, the seeds, and the threshold that would kill it —
written down before the run, because a threshold chosen after the result is a
description, not a decision. Agreeing on the gate before the code is the point.

**3. Gate.** Show it red, then green. A bug fix lands with a test that fails on
the un-fixed code; a new behaviour lands with a test that fails without it. Then
run `gate_fast` and the suite, and put the result in the pull request. A test you
never watched fail is decoration, and a test that cannot fail is a defect in its
own right.

## What we will not accept

- **A test deleted, skipped, or loosened to go green.** A failing test is a bug
  in the code it accuses. If the test's *claim* is wrong, say in the pull
  request which claim moved and why, and keep a gate on the new claim.
- **A number without its artifact.** Config, commit, date, and the log or test
  that produced it — or it is not a number that merges. This applies to a README
  sentence exactly as it applies to a results table.
- **A silent fallback.** A degradation is **LOUD** (an error naming the cause and
  the escape) or **COUNTED** (a counter a log reader sees). **SILENT** is a
  defect even when the fallback computes the right answer — that is what makes
  it expensive ([`AGENTS.md` §1.1](AGENTS.md)).
- **A compatibility shim for unused code.** If nothing calls it, delete it. A
  deprecated alias, a `try/except ImportError` around a dependency you require,
  or a flag kept "just in case" costs every future reader more than the deletion
  would have cost once.

## Commits and pull requests

- One task per branch, one branch per worktree ([`tools/wt.sh`](tools/wt.sh),
  [ADR-0022](docs/adr/0022-worktree-discipline.md)). Stage the paths you touched;
  no `git add .`.
- Commit style: `type(scope): the change, in the imperative` —
  `fix(moe): the router reads the mask`, `docs(readme): delete a stale count`.
- Do not push to `main`, force-push, or run a destructive git command here;
  that needs the owner's explicit approval. Local reversible actions are free.
  Run `git status` before any `checkout`, `reset` or `clean`.
- A pull request states four things: what changed, what was measured, what could
  regress, and which gate catches that regression.
  [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md) is the
  checklist; [`docs/guides/first-pr.md`](docs/guides/first-pr.md) explains where
  each line of it comes from.
- If a fix belongs in someone else's file, report it with `file:line` instead of
  smuggling it into this diff.

## Where things live

```
port/            the live lane: src/dormouse, tests/, configs/, scripts/, CHANGELOG.md
crates/          the frozen Rust lane (compiles and tests; no new work)
docs/            every document, one directory per kind — docs/README.md is the table
scripts/         repository-level gates and drivers
.github/         issue and pull request templates, CI
```

A new document goes in one of the directories listed in
[`docs/README.md`](docs/README.md) and gets a line in
`docs-site/tools/manifest.mjs`; [`AGENTS.md` §1.8](AGENTS.md) has the rule.
User-facing guides live in `docs/guides/`. Start at
[`docs/guides/first-hour.md`](docs/guides/first-hour.md) if this is your first
visit.

Security reports do not go in an issue — see [`SECURITY.md`](SECURITY.md).
