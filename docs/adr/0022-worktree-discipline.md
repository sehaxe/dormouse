# ADR-0022 — One task per worktree, and the worktree dies with the task

Date: 2026-09-27. Status: accepted. Read with ADR-0011 (loud failures),
ADR-0019 (no silent fallbacks) and ADR-0020 (oracle discipline); it is a
process ADR, not an architecture one. Tooling: `tools/wt.sh`.

## The rule

**One task per worktree, and a worktree is removed when its task lands.**

One sentence, on purpose: the failure it prevents is not a coding error, it is
two agents in one checkout, and a rule with carve-outs is a rule nobody
follows.

## Why: the cost is measured, not hypothetical

At `d96d155` the shared checkout had **135 dirty paths across 10 zones with
concurrent writers in at least 5 of them** (138 by the end of this session, as
writers kept landing). On the record:

* One in-flight edit to `crates/dormouse-data/src/lib.rs` broke `cargo check`
  for all three model crates, so **no test in the tree could run**.
  `docs/archive/architecture/design-review-model-2026-09-27.md` opens by recording that its whole
  review is `NOTHING IS RUNTIME-VERIFIED` for exactly that reason.
* A generated facade line in `vendor/dormouse-fused/Cargo.toml` broke **every
  cargo command in the tree** for ~30 min — `cargo metadata` fails ⇒ every
  `check`/`test` fails (`docs/archive/research/2026-09-27-fused-build-matrix.md`,
  "Environment hazards").
* Two agents overwrote each other's edits. Four extraction agents lost their
  output. One agent's outright failure cost the tree a broken-build window.
* All five of today's deepest findings shipped with **stale line numbers**
  because the tree moved underneath them; one report self-assesses as
  `integrity: suspect`.

None of that is a discipline problem. It is a filesystem problem: with N
agents in one working directory, `cargo test` is a lottery.

## Baseline: what already existed, and whether it survives a worktree

`git worktree list` at `d96d155` — four worktrees and **no convention at all**,
ad hoc, three on unrelated branches, two of them inside `/tmp`:

| path | commit | branch | size |
|---|---|---|---|
| `/home/sehaxe/dormouse` | `d96d155` | `main` | 42 GB `target/` |
| `/home/sehaxe/dormouse-pre4` | `5dc93dc` | `pre4-migration` | — |
| `/tmp/opencode/gatefix/wt_head` | `23f2b1c` | detached | 29 M |
| `/tmp/opencode/head-wt` | `3e2b3a4` | detached | **1.0 GB** |

`/tmp/opencode/head-wt` already holds a **1.0 GB `target/` on a 32 GB tmpfs**
(AGENTS.md: `/tmp` is tmpfs, it dies on reboot, it ate a 33.5 GB sidecar
once). A target dir on tmpfs is RAM that `ram-guard` is trying to protect.
Worktrees therefore go **outside the repo root and off `/tmp`**
(`/home/sehaxe/dormouse-wt/<task>`), which also means no `.gitignore` edit
and no new untracked noise in the shared tree.

**`.gitignore`: sane for this layout, unchanged.** `target/` is unanchored, so
it matches at any depth and a worktree's own `target/` is already ignored;
`*.bin`, `/checkpoints/`, `/runs/`, `graphify-out/cache/` are all
path-appropriate. Nothing to fix.

**`.gitattributes`: absent.** Checked rather than assumed: no tracked file
contains CRLF and `core.autocrlf` is unset (false), so on this box the absence
is a **latent** risk, not a live one. Two lines (`* text=auto eol=lf`,
`*.bin binary`) would close it before anyone checks out on Windows. Not added
here — it is not what this ADR is for, and it is a new file in a tree with 15
concurrent writers.

**The `exclude` list survives a secondary worktree — the concern was
unfounded.** `cargo metadata` in a fresh worktree of this repo exits **0**.
It works because `vendor/` is tracked (419 files), so the worktree really has
those directories, and `exclude` is resolved relative to *the workspace root*,
which is the worktree. The `workspace = true` inheritance trap the comment
warns about is therefore contained by the exclude list in every worktree, not
just the main one. **No manifest change is needed to work in a worktree.**

## The three traps, each measured rather than reasoned about

### 1. A worktree off `HEAD` is a different build than the main tree

A worktree is a checkout of a **commit**, not a copy of your working state.
`vendor/cubek-fix/` and `vendor/burn-cubecl/` are **untracked** (`??`, not `M`)
and `HEAD`'s root manifest differs from the working one:

| | `HEAD` (`d96d155`) | working tree |
|---|---|---|
| `exclude` | `dormouse-fused`, `cubecl-fix` | + `cubek-fix`, `burn-cubecl` |
| `[patch.crates-io]` | cubecl-runtime/server/cuda | + `cubek-reduce`, `burn-cubecl` |
| `burn-cubecl` in `Cargo.lock` | `source = "registry+…crates.io-index"` | **no `source`** ⇒ vendored path crate |
| `dormouse-core` dev-dep | `burn-ndarray … features = ["blas-openblas"]` | removed (BLAS-free `burn/flex`) |
| `dormouse-core` deps | — | + `dormouse-gdn2`, `dormouse-mor` (path) |
| members | 5 crates | + `crates/backend-parity` |

So a worktree off `HEAD` resolves **`burn-cubecl` and `cubek-reduce` from
crates.io, not from the two vendored patches** — it builds without the
ADR-0015 top-k fix and the ADR-0016 bool→float fix — and links **OpenBLAS**
for the test path, which the main tree deliberately stopped paying for.
`tools/wt.sh new` prints the dirty `Cargo.toml`/`Cargo.lock`/`vendor/` paths
and says out loud that the worktree is a different dependency graph. The rule
that falls out: **a task needing an uncommitted vendor patch cannot be worked
in a worktree until that patch is committed.**

### 2. `CARGO_TARGET_DIR` does **not** make a worktree warm

The obvious fix for the cold build is a shared target dir. It does not work
here, for a reason specific to us: **the vendored forks are `path`
dependencies, and their absolute path is part of the unit fingerprint**, so
every worktree rebuilds exactly the crates that sit at the *top* of the graph.
Measured in the probe worktree against the main tree's warm 42 GB `target/`:

```
Compiling cubecl-runtime v0.11.0-pre.4 (…/discipline-probe/vendor/cubecl-fix/cubecl-runtime)
Compiling cubecl-server  v0.11.0-pre.4 (…/discipline-probe/vendor/cubecl-fix/cubecl-server)
Compiling cubecl-cuda    v0.11.0-pre.4 (…/discipline-probe/vendor/cubecl-fix/cubecl-cuda)
```

The three patched crates — the ones `burn-nn`, `dormouse-spectral`, `dormouse-kda` and
all of `dormouse-*` sit on — rebuild from scratch, and later in the same cold
run the vendored `dormouse-fused` crates do the same
(`Compiling dormouse-bitnet v0.1.0 (…/discipline-probe/vendor/dormouse-fused/crates/dormouse-bitnet)`).
A shared dir reuses the *bottom* of the graph, which is the cheap part.

And the bottom-layer reuse is not even reliable. The same probe re-`Check`ed
`serde_json`, `parking_lot`, `async-channel`, `futures-lite`,
`text_placeholder`, `burn-std`, `burn-pack`, `cubecl-common`, `cubecl-ir` —
pure registry crates — because the four concurrent builds in the main tree use
different feature sets and keep invalidating each other. So a shared target dir
here buys a *partial, unstable* reuse that other agents can invalidate
underneath you at any moment.

The good news: it does not **corrupt** anything. Different fingerprint ⇒
different artifact directory; the main tree's vendored artifacts are never
overwritten. The cost is duplication, not damage — and the duplication is
already enormous. That one `target/` holds **50 distinct `cubecl-runtime`
fingerprint variants, 65 `dormouse-rmsnorm`, 69 `burn-tensor`**. Combined with
worktrees that differ in `Cargo.lock` and feature set, each worktree build
invalidates the others' registry artifacts and forces a rebuild back, in a
loop, with N agents. **A shared target dir does not make parallel work
cheaper; it converts it into a rebuild storm.** Worktrees get their own
`target/`, and the cold build is paid once per task.

### 3. The lock is real, and it is the honest ceiling

First line of the same probe, with four other agents mid-build:

```
Blocking waiting for file lock on build directory
```

That is the whole limit of this ADR in one line.

## The measurement, and the result nobody wants

Probe worktree `/home/sehaxe/dormouse-wt/discipline-probe` off `d96d155`,
trivial no-op change (`NOOP.md`, untracked), real command
`cargo test -p dormouse-core -p dormouse-data -p dormouse-train --lib -j 4`:

| run | time | outcome |
|---|---|---|
| **cold** (own `target/`) | **1387 s = 23 min 7 s**, 351 crates | **FAILED to compile** |
| warm re-run, same worktree | 4 s | same compile failure |
| shared tree, warm | 4 s | **FAILED to compile** (its own test target) |

**`HEAD` `d96d155` does not compile at all.** The build dies in
`vendor/dormouse-fused/crates/dormouse-muon-plus/src/lib.rs` with two `E0308`s
(`g_active.reshape([1, 1])` handed to `mul` — `expected D, found 2`, at
`:298` and `:364`). That file is `M` in the shared tree: the fix
(`.mul(g_active.unsqueeze())`) exists but is **uncommitted**, which is why a
worktree off `HEAD` cannot see it. Same defect class as
`docs/archive/research/2026-09-27-fused-build-matrix.md` defect 2, and it is the same
ADR-0019 silence in a new place.

The shared tree is not buildable either, in the way this ADR exists to
prevent: its `dormouse-core` **test** target has 3 `E0308`s in
`crates/dormouse-core/src/mor.rs:214` and `src/routing.rs:387` — two
**untracked** files being written by a live agent at that moment.

So the honest number is not "a worktree costs N minutes". It is:

> **A worktree costs ≥ 23 minutes before it can even fail, and today it cannot
> pass at all, because the branch point is broken.** A successful cold
> `build + test` was not observable on this commit: the 1387 s covers 351
> compiled crates and stops at `dormouse-muon-plus`, before the test targets of
> `dormouse-core`/`dormouse-train` and their three test binaries are built and
> linked. The 25–35 min figure is an **extrapolation from that partial run, not
> a measurement** — treat it as a floor to argue with, not a promise.

That is the real price of the discipline, and it is the number to argue with.
It is worth paying once per task; it is not worth paying per build, which is
why the worktree lives for the whole task and the warm number is the one that
matters in practice.

## The command

```sh
tools/wt.sh new <task>    # /home/sehaxe/dormouse-wt/<task>, branch wt/<task>, off HEAD
tools/wt.sh test <task>   # cargo test -p dormouse-core -p dormouse-data -p dormouse-train --lib
tools/wt.sh rm <task>     # drops worktree + branch; REFUSES if it holds uncommitted work
tools/wt.sh list
```

`--lib` is not optional: bare `cargo test` also builds `examples/`, and this
tree has unbuildable ones (three cuda-gated examples in `dormouse-kda`, fixed only
by commit `77bf807`, and `examples/quant_probe.rs` needs `--features cuda`).

`wt.sh test` **waits for other cargo processes to exit and refuses to start
under 25 GB available RAM**, because AGENTS.md doctrine 4 ("one heavy thing at
a time", *including builds*) stops being a nicety when `ram-guard` kills the
heaviest process and takes an in-flight agent's work with it.

## What this would have prevented today — and what it would not

**Prevented, concretely:**

* **The workspace-break window.** A half-written `vendor/dormouse-fused/Cargo.toml`
  cannot reach another agent's checkout. The 30-minute `cargo metadata` outage
  and the "no test in any of the three crates can run" window both end at the
  worktree boundary. Largest and best-evidenced win.
* **The two agents that overwrote each other.** Two writers, one index: the
  second `write` wins silently. One worktree each, and the second becomes a
  merge conflict that is *visible*.
* **The stale line numbers.** A finding read in a worktree quotes a commit, so
  `file:line` stays meaningful while the shared tree moves. This is what makes
  `integrity: suspect` unnecessary rather than merely forgivable.
* **The 1 GB `target/` on tmpfs**, and the `/tmp` worktrees that vanish on
  reboot leaving stale `git worktree` records.
* **Part of today's own wall clock.** The 1387 s cold build was spent
  recompiling the vendored forks that the main tree already had warm — pure
  path-fingerprint cost, and exactly what a single shared, serialised build
  lane would have avoided.

**NOT prevented — stated plainly, because a discipline that claims credit it
did not earn is worse than none:**

* **The four lost outputs.** If an agent's context died, a worktree does not
  recover the thinking. Worktrees protect the *filesystem*; lost output is a
  *session* failure. The fix is agents writing findings down as they go
  (ADR-0020), not this ADR.
* **The one agent that failed outright.** Its failure cost the tree a
  broken-build window *only because it was mid-edit in the shared checkout* —
  in a worktree that specific cost disappears, but the failure itself, and any
  cost it carries to shared *committed* state, does not.
* **A broken branch point.** Measured above: `HEAD` does not compile. A
  worktree inherits a bad commit exactly as faithfully as a good one. This ADR
  isolates you from a dirty tree, not from a bad commit.
* **The `mor.rs:214` / `routing.rs:387` class of failure as a *review* problem.**
  Those are untracked new files, so they are not in any worktree and not in
  any commit; the main tree is the only place they exist, and it is the only
  place they can break a build.

## What this does not fix

1. **Two agents editing the same file in different worktrees still conflict** —
   at merge time, not write time. Worktrees move the conflict later and make it
   visible; only a file-level ownership split removes it.
2. **Builds still contend.** A shared target dir means two cargo processes
   block on one lock (measured, §3). With separate target dirs they do not
   block, but they compete for 20 cores and for the RAM ceiling, and
   `ram-guard` kills the heaviest — the same failure by a different trigger.
   `wt.sh test` serialises; that is a script convention, not a kernel mutex.
3. **The cold build is paid once per task** and cannot be amortised away by
   sharing `target/` (§2: the vendored forks are path deps, so their
   fingerprint is per-worktree). Batch work into a worktree; do not churn them.
4. **Uncommitted work does not travel.** 138 dirty paths are invisible to every
   worktree, and the two biggest uncommitted items — the untracked `vendor/`
   patches — are precisely the ones that change the build.

## Consequence for dispatching work

An agent that needs an uncommitted change (a vendor patch, a half-landed
refactor) either waits for it to land or works in the shared tree at the
incident rate this ADR was written to end. Say which, in the task, before
starting. And note the precondition this session established: **branch from a
commit that compiles.** A worktree is a good isolation boundary pointed at a
broken commit.
