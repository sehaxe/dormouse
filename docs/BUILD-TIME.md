# Build time: which command, and why it costs what it costs

A 4-6 minute compile was blocking iteration four times on 2026-09-28. Every
one of those was the wrong tool. This file says which tool to reach for.

| you want | command | cost | why |
|---|---|---|---|
does it typecheck | `cargo check-train` | ~20-60 s | no codegen, no link |
run the test suite | `cargo test-core` / `test-data` / `test-train` | 40-90 s | our crates at opt-0 |
**run a short probe** (20 steps, read one counter) | `cargo build-probe` | ~1-3 min | dev profile: no LTO, our crates at opt-0 |
**run a training run** | `cargo build-train` | 4-6 min | release + `lto = "thin"` across 753 crates |

`build-probe` produces a slower binary (our four crates at opt-level 0) and
that is fine: 20 steps take seconds either way. **Do not build release to
answer a question a check can answer.** The trainer step is GPU-bound, so the
release profile's LTO buys nothing measurable for the loop itself
(`Cargo.toml` says as much) - it costs a relink on every launch.

## Why the release build is 4-6 minutes

- 753 packages, of which `burn-*` + `cubecl-*` + `pliron` dominate.
- `[profile.release] lto = "thin"` is a whole-graph link. That is the bulk of
  the wall clock and it is paid on **every relink**, not once.
- The vendored `burn-fused` crates (28) and `cubecl-fix` (3) are **path**
  dependencies, so their fingerprint is per-worktree. Editing one recompiles
  it and everything downstream of it (`burn-kda` → `dormouse-core` →
  `dormouse-train` → `dormouse-cli`), which is the whole chain. A `touch` on
  one vendored file is a 5-minute event.

## sccache

`sccache` is installed at `~/.cargo/bin/sccache` and was **not** in use
(`RUSTC_WRAPPER` unset, 0 requests recorded). It is now set in
`.cargo/config.toml [env]` with a 20 GiB cache.

It pays off exactly where this project hurts: profile changes, `touch`ed
vendored path deps, and fresh worktrees (`tools/wt.sh`, ADR-0022) all
recompile dependencies that did not change. The first build after enabling it
is a cold build and costs what a build costs today.

## What is deliberately NOT here

- **`cargo check` for everything.** `check-train` does not catch a bad link or
  a `no_std`-style cfg problem at the linker boundary. The vendored
  `cubecl-cuda` and `cubecl-ir` patches are exactly the kind of thing that
  only fails at codegen.
- **Removing `lto = "thin"`.** It is untested against the trainer's actual
  step time. It is the obvious next lever if the 4-6 min becomes the binding
  constraint again: measure one 200-step run with and without before
  removing it, because the AGPL release binary's speed is part of what the
  project sells.
- **A second target dir per agent.** The project's own rule (ADR-0022,
  AGENTS.md §1.5) is one heavy thing at a time, and a shared target dir is
  what makes parallel builds safe. Per-agent target dirs would multiply the
  disk cost of a tree that already hit "Disk full?" once during a mold link.
  sccache achieves the same benefit with one directory.
