# ADR-0024: the live recipe is a dense transformer; the loop stays for 1M

Status: accepted 2026-11-08. Records measurements taken 2026-11-05/06; no code
change lands with it.

## Context

The model this box trains is a 95M, 12-layer, 1K-window byte transformer
(`.bulba/plan.md`: BPB 1.49-1.52 after 0.25 GB). The schema default is still
the opposite — `block_form="kda"` builds the weight-shared loop — so the dense
recipe is pinned by preset (`port/configs/flagship3_dense.toml`), not by the
default.

Three measurements moved the live path off the loop:

1. **Equal compute, bytes and cadence.** A plain 13.5M dense decoder, on the
   flagship's own byte stream and update cadence, reached 1.4918 BPB at 500M
   bytes. The 102.7M KDA/Engram/MoE flagship's final readout is 1.6552 at
   1.23 GB and 1.6166 at 4.2 GB. At matched 500M bytes the gap is 0.258; on
   eight 65,536-byte windows at the 4.2 GB endpoint it is 0.126, negative on
   all eight (1σ 0.009); the control's own curve puts it at 0.12-0.32 (central
   ~0.27). The briefing's "0.13-0.37" is that gap stated coarsely.
2. **Equal active MAC, three seeds.** Dense 69.3M 2.4714 vs E=512 bank 199.3M
   2.5384, Δ = −0.0670 (spread 0.0201). The bank is dead capacity at this
   budget: 187-193 of 512 experts live at 600 steps, 8 of 512 on m400 at 150k.
   The routed bank, not the dense body, is what this kills.
3. **The literature at ≤200M.** Nothing beats a vanilla transformer in a
   compute-matched test at our scale; KDA/GDN's clean win is at 1.3B, Engram's
   at 27B, and a recurrence is worth 0.46 of a unique block at iso-FLOPs (one
   unique layer × 3 iterations ≈ 14% of a 12-layer model).

Caveats that move the control's numbers: one seed, ±0.05 BPB per figure, the
4.2 GB point is a floor-conditional fit, and the flagship endpoint had a
diverged router (`lfb_bias` [-117, +47]), so part of the bill is a controller
failure. The extrapolation-free facts are 0.258 at matched bytes and 0.126 on
8/8 windows.

## Decision

- The live recipe is the dense transformer: `block_form="transformer"`,
  `n_layers=12`, `max_iter=1`. KDA/CSA/mHC/Engram and the routed bank are not
  in the path.
- Every frontier arm stays selectable, not deleted: `block_form="kda"`,
  `block_stack="K:A"`, `value_residual`, `attn_gate`, `ffn_clip`,
  `nope_layers`, `mod_capacity` default off, and the KDA form stays
  bit-identical when selected.
- The loop is kept for one lane: **the 1M context claim**. `use_csa` is a
  KDA-loop-only flag (`_KDA_ONLY_FORM_FLAGS`) and the
  transformer form refuses it (`config.py::validate_block_form`); at 1M the
  dense form's attention is ~690 TB. A 1M claim therefore rides on a separate
  CSA artifact in the loop form.

## Consequences

- The live and old numbers are not like-for-like: the dense preset spends
  ~5.03× the flagship's fwd+bwd FLOPs/byte (691.37 vs 137.42), so any quality
  comparison must carry the FLOPs ratio.
- There is no 1M number today, for an artifact reason: the only trained CSA
  artifact (`longctx1`) is missing, and the 2026-11-08 `csa8k` run passed gates
  A and B but failed C (rank ~119/128 = chance with the needle at the window
  start), so gate D was not run. The 1M claim needs its own CSA checkpoint and
  its own protocol pass; it cannot be borrowed from the dense recipe.
- Do not read `block_form="kda"` in the schema as "the loop is the recipe"; the
  preset is the statement of record.
- The equal-MAC verdict is about a routed bank at 22 GB, not CSA/KDA at 1M.
  Quoting it as "the architecture is dead" overreaches the measurement.

## Evidence

- `docs/records/reviews/2026-11-06-control-transformer.md` §4, §6, §8 — the 0.258 /
  0.126 / 0.12-0.32 frames and their caveats.
- `docs/records/reviews/2026-11-05-architecture-review-dense.md` §0.2-0.3, "What
  survives" — equal-MAC, live-expert counts, the 81M/equal-compute mislabels.
- `docs/records/research/2026-11-06-architecture-at-100m.md` §4, §S4.
- `docs/records/research/2026-11-08-long-context-eval.md` §0, §4.6, §5.
- `docs/records/reviews/2026-11-08-night.md` §5b; `port/configs/flagship3_dense.toml`;
  `port/src/dormouse/config.py` (`ModelConfig.block_form`,
  `_KDA_ONLY_FORM_FLAGS`, `validate_block_form`); `.bulba/plan.md`.
