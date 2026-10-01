# Contributing to dormouse

**Read [`AGENTS.md`](AGENTS.md) §1 first.** It is the rulebook: loud failures,
A/B or death, zero host-device sync, one claim one evidence, and the machine
facts that decide which of those are even possible here. This file is only the
map of where things are and which command answers which question.

There is no open-source contribution process to speak of. This is one
maintainer's repository, and the workflow below is the one that actually runs.

## Before you edit

| you want to | do this |
|---|---|
| change code, or fix a bug | `tools/wt.sh new <task>` — one task per worktree, off `HEAD` |
| test your change | `tools/wt.sh test <task>` — runs our three crates plus the vendored-library gate |
| run something heavy (a build, a training run) | `tools/build_lock.sh run <label> -- <cmd>` |

`tools/wt.sh` is not ceremony: a worktree off `HEAD` is a **different build** if
the shared tree is dirty (ADR-0022), and uncommitted vendor patches do not
travel. Builds are serial because a mold link on a 64 GB box has frozen the
desktop three times (§2.4).

## Which command for which question

| you want | command | cost |
|---|---|---|
| does it typecheck | `cargo check-train` | ~20-60 s |
| run the test suite | `cargo test-core` / `test-data` / `test-train` | 40-90 s |
| run a short probe (20 steps, read one counter) | `cargo build-probe` | ~1-3 min |
| run a real training run | `cargo build-train` | 4-6 min |

`cargo build --release` to answer a question `cargo check` answers in 20 seconds
is the most expensive mistake available here, and it has been made four times in
one day. The reasoning is in [`docs/guides/build-time.md`](docs/guides/build-time.md).

## The four rules that most often get broken

Everything else is in `AGENTS.md` §1. These four are the ones with a recorded
cost.

1. **Every degradation is LOUD, COUNTED, or a defect.** A silent fallback
   usually computes the *right answer*, so the run looks fine and is a year
   slow. Missing data stops the run and is never synthesized; shape mismatches
   assert; `--guard` is the recovery action, not a substitute for the error.
   §1.1, and the three marks are the whole taxonomy.
2. **A bit-for-bit or "verified" claim names its external source.** "Verified
   against the reference" is not a claim. Naming the file, or saying plainly
   that no external reference exists, is. §1.4.
3. **A mechanism is judged against its own removal or deleted.** 3 seeds per
   arm, at a fixed step budget; a tie deletes the mechanism. The queue is
   [`docs/protocols/AB-PROTOCOL.md`](docs/protocols/AB-PROTOCOL.md) — read its
   "unrun" label before quoting any of it.
4. **One heavy thing at a time.** One GPU process, one build. Two legal
   processes colliding once wrote garbage weights into a checkpoint. §2.4-2.5.

## Rules for the change itself

- A file moves once. No stub at the old path, every reference rewritten in the
  same commit, and `python3 tools/check_doc_refs.py` green before you commit
  (§1.8). Adding a document means a line in `docs-site/tools/manifest.mjs`.
- Commit per task, in the repo's style (`fix(core): …`, `land(mor): …`, `!` for
  breaking). No `git add .` — stage the paths you touched.
- **No push, no force-push, no destructive git operation without the owner's
  OK.** Local reversible actions are free.
- Do not edit a file another agent is in. If a fix can only live in someone
  else's file, report it as `file:line` instead.
- Never fix unrelated things in the same diff. Follow-ups get reported.
- CPU-only tests cannot catch the backend-specific class. The gate for that is
  `cargo test -p backend-parity --features cuda --test backend_parity`, and it
  must be run for any change to a kernel, a dtype or a device path.

## Naming conventions

`docs/glossary.md` is authoritative and its last section is a live list of
places a document and the code disagree about a name. Use the word from there.
When a document and the code disagree, fix one of them — never invent a third
name (§1.7).

## License

MIT ([`LICENSE`](LICENSE)). The vendored forks under `vendor/` keep their own
upstream notices and terms; see each directory's `LICENSE` for its scope.