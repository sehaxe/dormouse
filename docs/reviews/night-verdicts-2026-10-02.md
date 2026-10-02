# Night A/B verdicts — wave 3 (2026-10-02)

Yardstick (`docs/guides/ab-testing.md` §0.1): control aux-ON 3-seed
6.387 / 6.314 / 6.329 — **mean 6.3433, seed range 0.0730**. §1.2 bar is the
control's own seed range: an arm wins iff its mean BEST HELD-OUT is below
**6.2703**; anything above is a tie, and a tie deletes the mechanism. All
runs: batch 8 × seq 512, window 81 920 B, `--retract-every 4`, 2 000 steps,
seeds 1/2/3, aux jepa 0.05 + KoLeo. The wave carries the named ±0.05
systematic (`retract-every 4` vs the control family's 1 — one paired seed
measured −0.058 for the wave's cadence; §0.3 of the wave report).

Evidence quality first (`docs/reviews/night-2026-10-02.md`, "no-train
resumes"): rows marked **[prior]** resumed a completed prior-wave checkpoint
and re-printed its number — they are their prior wave's measurement, not new
evidence. Rows marked **[dead]** did not complete.

## The table

| arm | s1 | s2 | s3 | mean | vs bar 6.2703 | verdict |
|---|---|---|---|---|---|---|
| control (aux-ON) | 6.387 | 6.314 | 6.329 | 6.3433 (range 0.0730) | — | yardstick |
| **moe** (`moe_topk=1`, top-1 of 3, lb 0) | 6.359 | 6.303 | 6.351 | **6.3377** (range 0.056) | +0.067 above | **TIE → deleted** |
| **attnres** (`use_attnres=true`) | **6.191** | 6.346 [s2r fresh] | 6.453 [s3r fresh] | **6.3300** (range 0.262) | +0.060 above | **TIE → deleted** |
| situ (`use_situ=true`) | [dead: OOM@300] | [dead: no run] | [dead: OOM@200] | — | — | **FAILED TO RUN — arm defect**, see below |
| mhc (`use_mhc=true`) | 6.413 [prior] | 6.313 [prior] | 6.370 [prior] | 6.3653 | +0.095 above | **TIE → deleted** (wave 1's verdict re-printed, `66a8f0d`) |
| fb (`aux_fb_weight=0.1`) | 6.406 | 6.391 | 6.463 | **6.4200** (range 0.072) | +0.150 above | **TIE → deleted** |

## Notes per arm

- **moe**: 3/3 fresh wave-3 seeds, engaged by construction (param-neutral
  routing; `off_is_the_dense_blend_bit_for_bit` gate; the log-line probe gap
  for `MOE_ROUTE` is the recorded §1.1 debt). Mean 6.3377 is nominally better
  than the control mean (−0.0056) and far inside the 0.0730 spread. §1.2: a
  tie deletes. **Not in the night build.** The review's 4-expert
  configuration was deliberately not what ran (top-1 of 3 = the
  param-neutral A/B, wave report §2.2); `moe_lb_coef = 0.0` by design.
- **mhc**: wave 3 added nothing (its three rows are wave 1's own checkpoints
  re-loaded). The wave-1 verdict stands unchanged: TIE at +0.0220 against
  0.0730, engaged (+6 155 params). **Deleted; not in the night build.**
  `mhc_streams = 4` remains the one follow-up the result says nothing about.
- **situ**: the only arm that failed on the MECHANISM level — 2/2 seeds that
  ran died mid-run with CUDA OOM **solo on the card** at step 200-300
  (`use_situ` doubles the FFN inner width: +393 216 params, engaged). Not a
  verdict on quality; a defect report for the arm's owner. **Not in the
  night build.**
- **attnres**: the one live candidate. s1 (fresh, complete): **6.191**, i.e.
  −0.196 paired against the control's same-seed 6.387 — **beyond the bar on
  one seed**, best train CE of the night (2.581 vs moe's 2.672-2.698), and
  engaged (+3 072 params = 9 200 526). s2's first attempt died of the disk
  incident at step 1000; the fresh re-run `s2r` (03:12-03:31): **6.346**,
  +0.032 paired against the control's same-seed 6.314. `s3r` (03:33-03:52):
  **6.453**, +0.124 paired against control 6.329. Fresh mean **6.3300** —
  TIE under §1.2, arm deleted; the paired-seed reading (−0.196 / +0.032 /
  +0.124, mean −0.013) says the mechanism's effect at this scale is noise
  with extra variance. Follow-ups for its owner, not this lane: the
  variance (range 0.262 vs control 0.0730) and the fact that at depth 2
  Full == Block (AGENTS §2.2), so this measured the cheap variant.
- **fb**: 3/3 fresh wave-3 seeds (step-0 ce 6.141-6.150 — the fb term itself
  adds ~+0.58 of loss from step 0; `fb=1503/1503` on every eval line,
  engaged, +196 864 params = 9 394 318). Mean 6.4200 is +0.150 above the
  bar — the widest loss of the wave — and its train CE (best 2.998-3.040)
  is the worst of any completed arm: the draft head's objective does not
  buy held-out bytes at this scale. §1.2: a tie deletes. **Not in the
  build.** (Numbers read from the final log segments 03:15; the verdict
  doc's earlier "pending" predates the 02:17-03:07 finishes.)

## What ships in the night build (final answer after fb + attnres re-runs)

**Nothing. Every arm ties.** attnres's fresh 3-seed mean is 6.3300
(6.191 / 6.346 / 6.453 — s3r landed 03:52), +0.060 above the 6.2703 bar:
the s1 6.191 that made the arm look like a winner was seed spread, not
signal (paired deltas −0.196 / +0.032 / +0.124; the 6.191-vs-6.406
s1-vs-s3 tension the wave started with was fresh-vs-prior evidence on
top of ordinary seed noise). §1.2: a tie deletes. The night build is the
**control recipe**: `--preset small`, aux at preset defaults (jepa 0.05 +
KoLeo, dspark 0), `--retract-every 4`, no arm flags, no moe/mhc/fb/situ.

Wave-3 roll-up (fresh-evidence means vs the 6.2703 bar): moe 6.3377,
mhc 6.3653 [prior], attnres 6.3300, fb 6.4200 — four mechanisms measured,
none survives its own removal by the §1.2 rule. attnres's range (0.262)
is 3.6× the control's (0.0730): whatever the mechanism does, it is not
variance-reduced, which is worth knowing before anyone re-runs it.
