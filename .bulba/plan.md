# Plan: the minimal core — what leaves `dormouse-core`, and what dies

STATUS: AWAITING_APPROVAL

Full deliverable: **`docs/PLAN-minimal-core.md`** (written, 2026-09-27, snapshot
`d651eb1`). This file is the index.

## Goal

Move every mechanism and every kernel out of the model crates into the library,
so training is maximally easy and each mechanism is independently fixable,
testable and A/B-able (ADR-0017).

## Description

1. **Today's DAG** and its violations (M1-M6 model→library, X1-X6 the other way).
2. **Target DAG**: a move table (file → crate → LOC), the reverse list, and the
   3-5 entry-point API of each moved mechanism.
3. **Order of operations**: 12 commits C0-C11, each green alone, with the parallel
   lanes and the WAIT markers for in-flight files named.
4. **"Maximally easy to train"** as a checklist: what to read, what is on by
   default and what the evidence supports, what the default costs, which knobs are
   load-bearing.
5. **The deletion list**: 8 items, ~1 620 LOC, each with its evidence.

## Success criteria

- [ ] `dormouse-core/src` is `lib.rs` + `model.rs` + `loop_block.rs` +
      `attention.rs` + `config/` + `routing.rs` + a 12-line `aux.rs`, ≤1 250 LOC
      (`wc -l`; today 2 493).
- [ ] `cargo check -p dormouse-core -p dormouse-train --features dormouse-train/cuda`
      green at **every** one of C1..C11 (no commit red between two commits).
- [ ] Zero library-crate test file mentions `dormouse-core` (grep in CI).
- [ ] `moR`, `DM_FUSED`, `fnv_hash` return nothing from
      `grep -rn --include=*.rs --include=*.toml crates/ configs/`.
      How: the C2 deletion commit + the D1-D8 items.
- [ ] `configs/smoke.toml` + `--smoke` train 3 steps on CPU in <60 s.
- [ ] A newcomer reads 4 core files (≤1 150 LOC) plus a 60-line
      `docs/README-model.md` and knows every arm that is on, why, and whether it
      has been A/B'd.
      Measure: `wc -l` on the four files + the README.

## Acceptance

- [ ] C1's diff is provably a rename (`git diff --stat C1~1 C1` shows only names,
      paths, and the root `exclude`/`[patch]` lists).
- [ ] Each moved mechanism's library crate has a `#[cfg(test)]` module that does
      not import `dormouse-core`.
- [ ] The D1-D8 deletions are one commit with before/after `wc -l` in the message.
- [ ] The optimizer routing table lives in `crates/dormouse-core/src/routing.rs`
      and `validate_routing` (85 LOC) is gone.
      How: `grep -rn "expert_ffns\.\|engram\.key_projs" crates/` matches core, not train.

## User questions and answers

(none yet — five open questions below for the owner)

1. **MoR**: delete per D1, or land the in-flight agent's work and keep the arm?
   The code says it has never run a forward and the paper says it loses below
   135M (we are 7.5M). Recommend: delete.
2. **Default aux weights**: turn JEPA+DSpark OFF by default (removes a second full
   forward from every step, and it is the reason `base` OOMs at batch 6), or keep
   the current default until the A/B lands? Recommend: off, `--jepa-weight 0.1`
   as the arm.
3. **`configs/smoke.toml` + `--smoke`**: add, or leave the newcomer to write
   `--set engram_rows=4096` by hand? Recommend: add.
4. **The unwired library (12 797 LOC, 41%)**: publish a wired/implemented/unimplemented
   inventory in `dormouse-fused/README.md`, or start deleting unwired crates now?
   Recommend: inventory first (it is one file), delete on a second pass.
5. **`--max-iter 2`**: promote to the `small` default on the strength of
   AB-PROTOCOL 4b's cost argument, or wait for the depth curve? Recommend: run the
   A/B, then decide — the quality half is not measured.

## Tasks

- [ ] C0. Land or revert the in-flight MoR work (the tree does not compile).
- [ ] C1. `refactor!: vendor/burn-fused → dormouse-fused` (script, one commit).
- [ ] C2. `refactor(core)!: delete what lost its A/B` (D1-D4, D7, D8; ~1 620 LOC).
- [ ] C3. `feat: dormouse-ema` (60 LOC). — parallel lane A
- [ ] C4. `refactor!: LinearLike → dormouse-linear` (165). — lane A
- [ ] C5. `feat: dormouse-residual` (117 + 120, 4 arms). **WAIT loop_block.rs** — lane A
- [ ] C6. `refactor!: act_quant → dormouse-bitnet` (170). **WAIT** lane A
- [ ] C7. `refactor!: host tables → dormouse-ngram` (340). **WAIT train/lib.rs** — lane B
- [ ] C8. `refactor!: HeadWiseMuon → dormouse-muon-plus` (120). lane B
- [ ] C9. `refactor!: stress + firewall → dormouse-stability` (112). lane B
- [ ] C10. `refactor!: the aux losses go home` (125). lane A
- [x] C11. `refactor!: the model names its own params` (routing in, `validate_routing` out). LAST.
      Landed as `831e3a0`. The trainer ASSEMBLES the groups from the declaration
      (`ParamGroup::from_ids`) and `check_installed` verifies the INSTALLED groups
      against the live tree, so a group is decided in one place. The audit's
      disagreement is real and is fixed: the live `muon_group()` excluded
      `\.s$` and `inner\.Dense\.bias$` but not `inner.Dense.weight`, so
      `--set use_tsct=false` put a dense expert's `[d,d]` weight on fp32
      Newton-Schulz — the ~40 s/step case §2.3 records as solved. DEViation from
      the task line: `validate_routing` is NOT deleted, the 85-LOC string
      validator it named is. The name stays because `train_loop` calls it at
      startup for the banner counts, and deleting the only startup gate to save
      9 lines is the wrong trade. `grep -rn "expert_ffns\.\|engram.key_projs"
      crates/` now matches core, plus 6 path strings in ONE train test that name
      which parameter each assertion is about (a rename fails it loudly) — no
      group is decided from one.
- [ ] C12. `feat: smoke preset + --smoke`, `docs/README-model.md` (≤60 lines).
- [x] D1. `docs: retract the false step-time and verification claims` (prose only,
      no Rust). **Landed 2026-09-29.** Withdrawn without replacement, because the
      corrected value is not known: the 25.8 s/step batch-8 figure (no committed
      log, no `benches/history.tsv` row, struck there as measured-under-load);
      the 3076 ms no-attention figure beside it (same); the `1810 → 365 ms`
      subtraction in `kda-sota-ceiling.md` §4.2 (both endpoints step-0);
      the attention backward's true cost (never measured - no run on record has
      executed one); the `~465 ms` as a measured fixed cost and the
      `0.067 ms/token` coefficient it was fitted with (the fit overpredicts 3x at
      the batch-8 shape); the `1.6 s/step` A/B budget and every `cost` cell
      derived from it; the scale ladder's whole wall-clock column; the
      `9 h`/`30-90 days` pair in `README.md` (e), which also disagreed with
      itself. Corrected WITH a sourced value: the 10 809 ms anecdote in
      `VERIFICATION.md`; the `bit_exact.rs` "pattern to follow" recommendation
      (it is RED 976/1000 and its 1000-case test is feature-gated off the
      default cell); the `--seed` claim in Layer 2 (`4b42b6d` seeds the init, and
      4 % of the model is still process entropy); "layers 1 and 2 pass today"
      (Layer 2 is unwritten); the `VERIFIED` row for the attention arm
      (`fused kda=64/0` in both arms of the 2026-09-29 preflight, and
      `autodiff_nested_balanced` is RED); "no CUDA graph capture in the
      burn/cubecl stack" (it is 5/5 green on this GPU); the `--bf16` flag
      experiment in `PLAN.md` (bf16 matmul cannot work on this backend);
      the `6.7-8.3 s/step` baseline and `KDA ~80 %` in `PLAN-minimal-core.md`;
      the `2-5 %` arithmetic share (~16 % against a warm step). Six research
      documents corrected in place, none deleted. Every correction names
      `benches/history.tsv`, a commit, or a `file:line`.
      **Deliberately left alone:** the ~465 ms order-of-magnitude reading in
      ten files is *vindicated* by the 245 ms warm step, and the launch-bound
      reasoning that rests on it is now confirmed by an independent instrument
      (13.3 % mean GPU utilisation, 79 % of samples ≤5 %).

## Critical Files

- `crates/dormouse-core/src/loop_block.rs` (617, in flight)
- `crates/dormouse-core/src/param.rs` (165)
- `crates/dormouse-core/src/config/schema.rs` (216, in flight)
- `crates/dormouse-train/src/optim.rs` (478)
- `crates/dormouse-train/src/offload.rs` (402, in flight)
- `tools/migrate-dormouse-fused.sh`
- `docs/adr/0017-dormouse-fused.md` (the table needs 3 corrections)
- `docs/PLAN-minimal-core.md` (the deliverable)

## Risks

- High: C1's rename vs dirty paths → keep the script's dirty-tree guard.
- High: field renames break the optimizer routing table → C11 last, and move the
  table into core so it cannot drift again.
- High: the Engram 500k-row + floor fix is ~24 h old, has never trained a step,
  and is 86% of the model's params → keep ON, label it unmeasured, make it
  AB-PROTOCOL arm 0.5.
- Medium: 1 700 LOC of deletions in the same week as a 2 000-line rename → 11
  commits apart, deletion first, rename last and alone.
- Medium: 12 797 LOC (41%) of the library is unwired → inventory, then delete.
