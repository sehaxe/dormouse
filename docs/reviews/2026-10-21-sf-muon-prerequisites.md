# SF-Muon prerequisites — what the 2026-10-18 A/B measured, and what the re-run tests

**Date:** 2026-10-21. **Status:** the arm is now *legal to judge*, not judged. No
new BPB exists: the numbers below are re-read from the 2026-10-18 wave's own
JSONL and config snapshots under `~/logs/ab_muon_variants/`. The code changes
are `port/` commits `42d0f4e` (the preconditions) and `474ead8` (the driver);
**the driver has not been run**, so no quality claim is made here or by it.

The research note this implements is
[`docs/research/2026-10-16-sf-muon.md`](../research/2026-10-16-sf-muon.md);
line cites below are to that file.

## 1. What the earlier A/B actually measured

The 2026-10-18 wave: 4 arms × 3 seeds, `medium` (d_model 1024, T=4), b8 × s512,
2 000 steps, the driver's `lr=1e-3`, `warmup_tokens=0`, `opt_graph=false`,
pure CE, **one 81 920 B eval window on every run**, 0 NaN-skipped steps and 10
eval rows per run. Each run's config snapshot sits beside its JSONL (e.g.
`sf_muon_s1.config.toml`), which is the evidence for that paragraph.

| arm (`muon_variant`, `muon_sf`) | seed BPB | mean | vs control |
|---|---|---|---|
| `control` (`muon`, false) | 2.1740 · 2.1701 · 2.1697 | **2.1713** (spread 0.0043) | — |
| `sf_muon` (`muon`, true) | 2.2257 · **4.0573** · 2.2065 | 2.2161 over the two kept seeds | **+0.0448** |
| `sf_normuon` (`normuon`, true) | 2.2110 · 2.2045 · 2.2259 | 2.2138 | +0.0425 |
| `normuon` (`normuon`, false) | 2.2356 · 2.2454 · 2.2495 | 2.2435 | +0.0722 |

Three facts the headline hid:

1. **The "+0.045" is a two-seed mean.** It is `sf_muon`'s 2.2257 and 2.2065
   averaged (2.2161) against the control's 2.1713; the third seed is
   **4.0573**, 64 robust σ from that arm's median (2.2257; MAD 0.0192). The
   all-three mean is 2.8298, i.e. +0.66 BPB. The 2026-10-18 report printed the
   trimmed number without marking the seed it dropped, which is the defect the
   new driver's rule exists to prevent (`scripts/ab_sf_muon.py:195-215`).
2. **No `sf_gap` row was written.** The driver's docstring promised an extra
   held-out pass at `Y` (the gradient-evaluation iterate) beside `X`; none of
   the twelve JSONLs carries an `event=sf_gap` row, so the `X`↔`Y` gap the
   research note makes the verdict metric (§4, lines 120-121) was never
   measured.
3. **The arm was not the paper's full-model SF.** The AdamW half and the
   `engram_ram` host tables kept WSD (the hybrid `optim.Muon.sf_switch`
   documents, config `muon_sf`'s comment), and every parameter ran fp32 with
   `ns_kind` classic. Those are stated deviations, not defects; they bound what
   the number can mean.

## 2. Which precondition was violated

The research note names three gaps (lines 26-30, §3 lines 69-93). Their status
on the 2026-10-18 arm, checked line by line:

| gap | research-note lines | on the 2026-10-18 arm |
|---|---|---|
| **S1** exact polar vs finite NS+ColRow | 71-76 | **Still outside.** The arm ran `zeropower_via_ns(..., 5)` then ColRow, not `polar(M)`. No published theorem composes finite-NS with SF averaging (the note's own [S] inference); this is not fixable without changing the control, so the re-run inherits it and its verdict is about the **port's** arm, not the paper's algorithm. |
| **S2** weight decay at `Z` | 77-81 | **The precondition as written was violated; the arithmetic was not.** The note's snapshot (`optim.py:234-237`) predates commit `cbcce54` (2026-10-17), which made the SF arm inherit the trainer's `wd=0.01` and applies it at `Z` in `_sf_batched`/`_sf_one`. The 2026-10-18 configs show `wd=0.01`, so the number did sit at `Z`. What was violated is the *statement* of the precondition: no config key owned it, λ≤0 was legal, and an ordinary `wd=0` for the AdamW half would have silently left Lemma 3.1's bounded-diameter lift unmet. `muon_sf_wd` (default 0.01) now owns it, and both the config and `MixOptimizer` refuse λ≤0 (`src/dormouse/config.py:2000-2010`, `src/dormouse/train/optim.py:1091-1099`). |
| **S3** per-step TSCT retraction | 82-93 | **Not present in this port at all.** The note cites the Rust trainer (`AGENTS.md:406`); `grep -rn "retract" port/src` finds docstrings and no call site — the Python loop never calls `retract_tsct`. The 2026-10-18 arm therefore did not have the S3 violation, but it also had no record of the choice. `muon_sf_retract` (default false) records it and `true` is refused: the averaged iterate is a convex combination of retracted factors (column norms ≤ 1, equality only when they agree), and Thm 2.1's projection/Lyapunov step needs a **convex** set (`src/dormouse/config.py:2011-2024`). |

A fourth seam was found while wiring the re-run, and it would have bitten it:
under the battery's `opt_graph=true`, `OptimizerGraph.set_lr` wrote **every**
group, so an SF run — which the graph refuses and continues **eager**, COUNTED
— would have put the Muon group on the WSD decay for the whole run. That is not
the schedule-free arm, and no counter would have said so. Fixed in `42d0f4e`:
the SF LR is written back after `set_lr` (`src/dormouse/train/loop.py:763-780`)
and `_fall_back` no longer overwrites a group that holds a host float
(`src/dormouse/train/graph.py:326-338`).

So the honest answer to "which precondition was violated" is: **S2's
enforcement** (implicit, zero-able decay) and **S1** (finite-NS direction, still
open). S3 was absent rather than violated; the 4.0573 seed is why the re-run
marks divergences instead of trimming them quietly.

## 3. What the re-run will test

`scripts/ab_sf_muon.py` (commit `474ead8`) runs 3 arms × 3 seeds × 2 000
steps on `medium`, b8 × s512, the battery's COMMON, one 81 920 B eval window,
JSONL to `~/logs/ab_sf_muon/`, preflight first:

| arm | what differs |
|---|---|
| `muon` | the control: `muon_sf=false`, WSD on both halves. `muon_sf_wd` is inert (the Muon group is built with `wd=0.0`; gated by a state-dict sha + `torch.equal` logits in `tests/test_sf_muon_legal.py`). |
| `sf` | `muon_sf=true`, `muon_sf_wd=0.01` at `Z`, `muon_sf_retract=false`; constant Muon LR after warmup. |
| `sf_lr05` | `sf` at **half** the peak LR. The paper's claim is that SF removes the schedule; an SF arm that only matches at one LR is a tuned constant-LR arm, not the claim (research note §4, lines 123-127). |

The verdict, written before the first run: a seed >3σ from its arm's median is
a **divergence**, excluded from the mean and printed with its missing count
(`n=2/3`); σ is the MAD estimate with the paired-eval floor, because the sample
σ of three points (1.06 BPB on these numbers) would have passed the 4.0573 seed
it should catch. `sf` is kept iff its trimmed mean beats the control's trimmed
mean by more than the control's own seed spread; a tie deletes; `sf_lr05`
reversing the sign means the result is LR-specific. A 2 k-step tie is the
**designed** outcome (the note's prediction, lines 137-142) and escalates to
one 10× arm before deletion — not to a verdict.

Two limits of this re-run, stated so nobody reads past them: **S1 remains**
(the direction is still 5-step NS + ColRow, so the computation is the port's,
not the theorem's exact-polar one), and the endpoint is reported with its curve
but **without** the `X`↔`Y` gap — the CLI driver scores only the returned
iterate, and that gap is the follow-up this note does not discharge.

**No quality claim is made until the driver runs.** The earlier A/B measured a
2-seed mean of a mis-stated arm with a divergence hidden in the third seed; the
re-run measures the arm the preconditions describe, with the divergence on the
table.
