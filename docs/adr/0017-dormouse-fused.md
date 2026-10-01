# ADR-0017 — dormouse-fused: the technology lives in the library, the model is composition

Date: 2026-09-27. Status: accepted, migration scripted, execution pending a quiet
tree.

## The decision

1. `vendor/dormouse-fused/` is renamed to `dormouse-fused/` at the repository root.
   It is not a vendor: every one of its crates is ours. Under `vendor/` it was a
   lie, and the `burn-*` crate names falsely implied upstream burn crates that
   someone could find on crates.io - which is exactly the confusion that made a
   researcher conclude "burn-fused is a dependency we import" instead of "this is
   our technology library".
2. Crate names follow: `burn-kda` -> `dormouse-kda`, `burn-gdn2` ->
   `dormouse-gdn2`, and so on for all 26 crates. One namespace, ours.
3. **Every mechanism and every kernel lives in the library, never inside the
   model.** `dormouse-core` keeps only the model, the loop, the config and the
   wiring. Anything another project could reuse is a library crate with its own
   tests, its own arXiv reference in the doc comment, and its own A/B.

## Why this is not cosmetic

- The owner asked for ready, researched implementations rather than inventions.
  A library crate is where a published mechanism lives with a citation and a
  bit-for-bit test; a function inside `dormouse-core` is where it becomes
  folklore.
- The "A/B or death" rule needs a deletable unit. A crate is deletable in one
  commit with its tests; a mechanism wired into `loop_block.rs` is a diff inside
  the model, and every deletion becomes a risky edit.
- It is the honest boundary for the 26 crates we already own. Today the model
  wires 10 of them; the other 16 (`attnres`, `mor`, `mhc`, `mod`, `fastblt`,
  `ptrn`, `ttt`, `situ`, `parcae`, `eggroll`, `byteflow`, `diffusionblocks`, `es`,
  `nope`, `mtp`, `antihall`) are invisible: nobody can tell implemented-and-unused
  from imagined. Renaming plus the inventory in research/ is what makes the
  library legible.

## What moves out of `dormouse-core`

| today | destination | why |
|-------|-------------|-----|
| `src/act_quant.rs` (170) | `dormouse-bitnet` | its own doc comment says it "adds the activation side" to the quantizers already in burn-bitnet; it is one mechanism split across two homes |
| `src/param.rs` (165) | new `dormouse-linear` | `LinearLike` + TSCT factors + per-SM quant selection + polar retraction is a technology, not a model detail |
| `src/gr.rs` (117) | new `dormouse-residual` (or a module in `dormouse-mhc`, see below) | Gated Residual is a published mechanism (Qwen3.8-Flash-Next) and is comparable to mHC and AttnRes |
| `src/aux.rs` (229) | `dormouse-jepa` / `dormouse-dspark` (the loss) + a thin composition helper in core | same reasoning: the losses belong with their implementations |
| `src/attention.rs`, `src/loop_block.rs`, `src/model.rs`, `src/config/` | stay | composition and configuration, which is what the model crate is for |

## The finding that makes the residual split sharp

We now have FOUR implementations of the residual stream and have compared NONE
of them:

- ReZero (in `loop_block.rs`, scale init 1.0)
- Gated Residual (in `gr.rs`, `use_gr`, never A/B'd)
- mHC - Manifold-Constrained Hyper-Connections, DeepSeek 2512.24880, already
  implemented in `burn-mhc`
- AttnRes - Attention Residuals, 2603.15031, already implemented in
  `burn-attnres` (2060 lines, fused CUDA kernel)

The residual stream is where a shared-weight loop is most sensitive, and it is
the one axis where we have a free, already-written alternative. That comparison
is queued; the library split is what makes it a config flag instead of a rewrite.

## Execution order (why it is not done yet)

Eleven agents are currently editing files inside `vendor/dormouse-fused/crates/*` and
`crates/dormouse-*`. A repository-wide rename changes the git path of every file
those agents are writing, so their in-flight `git add <path>` calls would stage
the wrong files or resurrect deleted paths, and their commits would be corrupted.
The migration is therefore scripted (`tools/migrate-dormouse-fused.sh`) and runs
in one shot on a quiet tree, with the full test suite as its gate.

Order of operations, when it runs:
1. `git mv vendor/dormouse-fused dormouse-fused`
2. rename each crate directory and its `package.name` / `[lib].name`
3. rewrite every import and every path dependency across the workspace
4. fix the root `Cargo.toml` `exclude` list (the vendored crates are their own
   workspace roots; the exclude exists so their `workspace = true` inheritance
   does not resolve against our root)
5. `cargo check` the whole workspace, then the full test suite
6. one commit: a pure rename, no behavior change
7. then the mechanism moves from the table above, one crate per commit
