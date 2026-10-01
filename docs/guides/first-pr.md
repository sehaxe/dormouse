# Your first pull request: ten checks, and where each rule actually is

A checklist, not a restatement. Every line links to the section of
[`AGENTS.md`](../../AGENTS.md) that **is** the rule — the point is that the rule
has one home and this page is only a way into it. If the two ever disagree,
AGENTS.md is the one that is right (§1.7: fix one of them, never invent a third
name).

The commands and their costs are in [`CONTRIBUTING.md`](../../CONTRIBUTING.md)
and [`docs/guides/build-time.md`](build-time.md).

| # | before you commit | the rule |
|---|---|---|
| 1 | Every degradation you added is **LOUD**, **COUNTED**, or it is a defect. A missing file stops the run; a wrong shape asserts; a skipped kernel prints a counter. "The fallback computes the right answer" is the reason this class is expensive here — it means a silent bug reads as a year of lost throughput | [§1.1](../../AGENTS.md) — the three marks are the whole taxonomy |
| 2 | Your mechanism has a gate: an assertion, a counter, or a seam a test can read. If it only exists in the log, it does not exist | [§1.1](../../AGENTS.md) — "a fused/accelerated arm must be able to show it ran" |
| 3 | Every claim you wrote names its evidence: a config, a date, a commit, a `file:line`. "Verified" and "bit-for-bit" are reserved for a named external source; where none exists, say **no external reference exists** | [§1.4](../../AGENTS.md) |
| 4 | The new mechanism beats **its own removal** on held-out BPB at a fixed step budget, or it is deleted. A tie deletes the mechanism; 3 seeds per arm. If it is not A/B'd yet, say that on the page, in those words | [§1.2](../../AGENTS.md) · queue: [`docs/protocols/AB-PROTOCOL.md`](../protocols/AB-PROTOCOL.md) |
| 5 | No host-device sync crept in: no `try_into_scalar`, no `into_data`, no `blocking_read`, no host-side branch on a device value. Anything a host must know is produced by a device counter and read at a declared cadence. **Count on the host; never build a numeric indicator from a bool tensor on device** | [§1.3](../../AGENTS.md) |
| 6 | You checked what this box cannot do before you relied on it. Precision, 4-D slicing, the allocator high-water, mixed-dtype ops and the shard corpus each have a recorded failure and a recorded reason | [§2.1–§2.3](../../AGENTS.md), [§2.5](../../AGENTS.md), [§2.6](../../AGENTS.md) |
| 7 | Nothing heavy ran twice at once. One GPU process, one build, under the build lock and the `MemoryMax` cgroup; `free -g` avail ≥ 25 before a build or a run | [§1.5](../../AGENTS.md) · [§2.4](../../AGENTS.md) |
| 8 | Your branch is a worktree off a **compiling** commit (`tools/wt.sh`), you did not touch a file another agent is in, and you staged only the paths you changed — no `git add .` | [§1.6](../../AGENTS.md) |
| 9 | A document moved: the old path has no stub left, every reference was rewritten **in the same commit**, `python3 tools/check_doc_refs.py` is green, and a new document has a line in `docs-site/tools/manifest.mjs` | [§1.8](../../AGENTS.md) |
| 10 | You did not push, force-push, or run a destructive git operation. That needs the owner's OK; local reversible actions are free | [§1.6](../../AGENTS.md) |

## The two gates that are easy to forget

- **`cargo check` is not enough for a kernel, a dtype or a device path.** The
  class of bug that only appears on the CUDA backend cannot be caught by a CPU
  test suite: run `cargo test -p backend-parity --features cuda --test backend_parity`
  ([§2.1](../../AGENTS.md) — it turns "the cast is broken" into a command instead
  of a paragraph).
- **A CPU test can pass while asserting nothing.** The measured example is in
  [§3.3](../../AGENTS.md): a bit-exactness gate that was RED 976/1000 and whose
  1000-case test was feature-gated off the default cell. Read what your
  assertion can fail before you trust it.

## What "done" means here

Full tests green **and** typecheck/lint clean. "Smoke" is not done. A failing
test is a bug in the code, not a test to skip. Nothing is deleted to make a gate
green.

Fixing something unrelated in the same diff is a review finding, not a courtesy:
report it with `file:line` and leave it.