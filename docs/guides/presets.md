# Presets: what each one costs and what it is for

Presets are **data, not code**: flat TOML in [`configs/`](../../configs), loaded
by name or by path, with **the schema defaults being `small`**
(`crates/dormouse-core/src/config/schema.rs` — asserted by
`default_equals_small_preset`). A missing key in a preset file is the schema
default, so a preset is a *diff* from `small`, not a full specification.

Load order is `resolve` (`crates/dormouse-train/src/cfg.rs:28`): **preset TOML →
`--set k=v` → typed flags**. Preset files are not a registry in Rust; adding one
is adding a file.

## The cost, measured on the instantiated model

Counts come from `cargo test -p dormouse-core --test preset_exec`
(`every_preset_states_its_cost_and_no_preset_is_a_lookup_table`), which visits a
**real `DormouseModel`** and splits the visitor's totals into memory rows and
computation. Not a closed form: a formula that drifts from the module tree is
exactly the kind of number that reads true and lies.

The memory-row count is derived, not configured. `engram_rows = 25_000` **rounds
up to a power of two** — `engram_tables`
(`crates/dormouse-core/src/loop_block.rs:62-72`) returns 32 768 rows per table
because the in-model read *masks* the hash rather than dividing by it — so every
preset carrying the default budget has **3 × 32 768 × 32 = 3 145 728** memory
parameters.

Measured 2026-10-01 at `2d0f81d`, CPU backend, under the build lock:
`cargo test -p dormouse-core --test preset_exec every_preset_states_its_cost_and_no_preset_is_a_lookup_table -- --nocapture`
(the whole run took 1 693 s; the five presets below are the `CHEAP_PRESETS` list
at `crates/dormouse-core/tests/preset_exec.rs:542`).

| preset | total | compute | memory rows | memory share | param tensors |
|---|---|---|---|---|---|
| `nano` | **7 193 002** | 4 047 274 | 3 145 728 | 43.7 % | 54 |
| `nano-fused` | **7 193 002** | 4 047 274 | 3 145 728 | 43.7 % | 54 |
| `small` | **9 197 454** | 6 051 726 | 3 145 728 | 34.2 % | 54 |
| `mor` | **9 197 454** | 6 051 726 | 3 145 728 | 34.2 % | 54 |
| `base` | **13 065 874** | 9 920 146 | 3 145 728 | 24.1 % | 54 |

`nano`/`nano-fused` and `small`/`mor` are **identical to the last digit**, and that
is the point: an A/B pair must differ in the arms and in nothing else. `nano`
vs `nano-fused` differ only in which arms run (both count the same rows — see
below); `small` vs `mor` differ only in `use_mor`, `mor_k`, `mor_bce_weight`,
which `mor_differs_from_small_in_the_three_mor_lines_only` asserts from the
resolved configs rather than assuming.

The three **wide** presets are behind `-- --ignored` because a full-width CPU
instantiation costs minutes each (`WIDE_PRESETS`,
`crates/dormouse-core/tests/preset_exec.rs:583`):

| preset | cost | provenance |
|---|---|---|
| `swift50` | not re-measured here | run `-- --ignored`; the pre-rewrite README carried `-- --ignored` for this row too |
| `one_b` | not re-measured here | as above |
| `p150` | ~160 M compute | `configs/p150.toml:10` — "core params measured with the header probe: 159,720,195", i.e. **compute only**, before the 3.15 M memory rows are added |

## What each preset is for

Geometry read from the TOML files; arms from the same files. "JEPA" means
`jepa_weight > 0`, "DSpark" means `dspark_weight > 0`.

| preset | d_model | n_heads | experts | max_iter | memory rows/order | KDA | Engram | JEPA | DSpark | what it is for |
|---|---|---|---|---|---|---|---|---|---|---|
| `nano` | 512 | 8 | 3 | 4 | 32 768 | on | on | 0.05 | 0 | the cheapest thing that runs every arm — smoke tests |
| `nano-fused` | 512 | 8 | 3 | 4 | 32 768 | **off** | **off** | 0 | 0 | "all arms off". Named after a deleted module, so the name is historical (glossary's open list) |
| `small` | 768 | 12 | 3 | 4 | 32 768 | on | on | 0.05 | 0 | **the flagship on 16 GB**, and the schema's default |
| `mor` | 768 | 12 | 3 | 4 | 32 768 | on | on | 0.05 | 0 | `small` + Mixture-of-Recursions; the A/B arm for the MoR router. Not combinable with `--rand-depth` |
| `base` | 1024 | 16 | 3 | 8 | 32 768 | on | on | 0.05 | 0 | 24 GB-class work; the JEPA teacher is a second full forward, so it OOMs a 16 GB card at batch 6 |
| `swift50` | 1024 | 16 | **8** | 8 | 32 768 | on | on | 0 | 0 | many-expert experiments — note JEPA is **off** here, so it is not `base` plus experts |
| `one_b` | 2048 | 32 | 4 | 12 | 32 768 | on | on | 0.05 | 0 | the 1 B dream; does not fit 16 GB |
| `p150` | 4096 | 64 | 4 | 12 | 32 768 | on | on | 0.05 | 0 | the 150 M-program flagship (`docs/architecture/PLAN.md`). Overrides only — every other key is the schema default, i.e. `small`'s |

Two rows are worth reading twice. `swift50` has **JEPA off** where its neighbour
`base` has it on, so it is not "`base` with more experts" — it is a different
experiment. And `nano-fused`'s name refers to a deleted module, so what it now
means is "every arm off".

**DSpark is 0 in all eight presets.** It is implemented, gated and A/B-ready; the
`small` schema default is `dspark_weight = 0.0`, and turning it on is
`--set dspark_weight=0.1`. Nothing recorded so far is a DSpark measurement —
the window-shift fix (2026-09-29) invalidated every DSpark number the project
had, and there was nothing on disk to retract.

## Two things that surprise everyone

**Every preset pays the same 3.15 M memory rows.** The budget is a ratio, so the
narrow presets pay it hardest: measured, **43.7 % of `nano`, 34.2 % of `small`,
24.1 % of `base`**.

Every one of the per-preset comments in `configs/*.toml` states a different
share — `small` says "24 %", `nano` "31.0 %", `base` "16.4 %", `swift50` "4.6 %",
`one_b` "0.2 %", `p150` "1.5 %" — because they are all 2.4 M divided by a
**retracted 7.5 M backbone count**, and the shipped table is 3 145 728 (the
power-of-two round-up) against the measured totals above. Only `base`'s figure
is close by coincidence. Those six sentences are the arithmetic
`docs/papers/engram.md:198` (E13) already flags as a bug, still in the config
files; the correction belongs to whoever owns the comments, and it is reported
here rather than applied, per AGENTS.md §1.7.

**`use_engram = false` does not shrink the model.** The tables are built
unconditionally and the flag is a runtime switch, so `nano-fused` ships the arm
off and still allocates its rows. `--engram-ram` is what actually moves them out
of the model: the config seam squeezes the in-model table to one row per order
(`crates/dormouse-train/src/cfg.rs:50-52`), because on that path the rows arrive
pre-gathered from the host and the in-VRAM table is never read.

## Fields that exist in the schema and in no preset

`use_gr`, `use_attnres`, `use_mhc` + `mhc_streams`, `use_situ`, `moe_topk`,
`moe_lb_coef`, `act_quant`, `aux_fb_weight` + `aux_fb_horizon`, and `use_mor` +
`mor_k` + `mor_bce_weight` (which appear only in `small` and `mor`). All are
**off by default**, and `preset_exec::fields_no_preset_ships_still_take_effect`
asserts each one *changes what runs* rather than trusting the default — a field
in the schema that no preset sets would otherwise be a promise nobody checks.

A shipped arm with `weight = 0.0` is not built at all — at `aux_fb_weight = 0`
the future-byte head does not exist, so those presets' parameters and checkpoints
are byte-identical to a build from before the arm shipped.

## Gates

- `cargo test -p dormouse-core --test preset_exec` — resolves every preset in
  `configs/`, prices the cheap five, and **asserts every arm a preset declares
  ON was entered** against the counters in `crates/dormouse-core/src/probe.rs`.
  The class test it exists for is a preset that declares an arm and never enters
  it: the fused gated-delta kernels were dead for a year behind exactly that
  shape.
- The same file asserts **no preset is a lookup table with a model attached**:
  memory rows must stay ≤ 50 % of the total. `nano` sits at ~44 %, so that gate
  is close to its threshold and a wider memory budget on a narrow preset would
  turn it red.
- `cargo test -p dormouse-core --test preset_exec -- --ignored` — the wide
  presets (`swift50`, `one_b`, `p150`), which cost minutes each to instantiate on
  the CPU backend.
- `mor` is **`#[ignore]`d by a real failure, not by cost**: the MoR arm panics on
  the CPU backend because `burn_mor::topk_indices` returns I64 where
  `Tensor<D, Int>` is i32 (burn-flex `tensor.rs:170`). Its cost is asserted
  anyway, from the resolved configs — `mor` must differ from `small` in exactly
  three scalars — so the A/B pair cannot drift into comparing two different
  models.

## Preset flags vs `--set`

There is no `--no-gr` and no `--no-mor`. Those arms are config, not A/B flags:
`--set use_gr=false`, `--set use_mor=true`. The flags that exist (`--no-kda`,
`--no-engram`, `--max-iter`, `--bf16`, `--act-quant`) are the ones that are
layer overrides over a preset value. Full table:
[`docs/guides/cli.md`](cli.md).