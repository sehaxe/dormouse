# Data quality: what the literature actually measures, and what applies to us

Read 2026-10-27. Every claim below carries its arXiv id; numbers in the "ours"
column are from this repo's own runs.

## The one thing that changes our plan

We measured our heuristic quality band at **×1.07** (2026-10-23, `mix5`) and
concluded "data is exhausted". That conclusion was about *heuristics*
(dedup, stop-words, punctuation, boilerplate). The literature's real lever is
different and it is **model-based pruning**, not hand-written filters:

| claim | evidence | what it means for us |
|---|---|---|
| A **125M reference model's perplexity** prunes a corpus well enough to give a 3B model **+2.04 avg downstream** and **1.45× fewer pretraining steps**; it also works in the **over-trained and data-constrained** regimes | 2405.20541 | we already *own* a 103M model. 1.45× is smaller than ×3-10 but it is measured, cheap, and stacks with everything else |
| Perplexity–benchmark **correlation** selection beats DSIR on 8/8 benchmarks and matches DataComp-LM's bigram classifier at 160M, improving with scale | 2409.05816 | the scoring rule is not "low perplexity is good" — it is "documents whose loss predicts benchmark gains" |
| Classifier quality filtering **improves downstream but not necessarily LM loss**, and CQF-selected data can **beat training on the high-quality set itself** | 2510.00866 | do not chase "gold documents". Removing noise is the mechanism, so a *band* (drop both tails) is the right shape, not "keep the best" |
| Quality enters a scaling law as a dimensionless **Q** with two estimators (corruption rate, deficiency) | 2510.03313 | we can predict the payoff instead of guessing, and report Q for a corpus |
| Quality filtering alone causes **diversity collapse**; diversified selection (leverage scores) is the counterweight | 2609.32484, 2504.16511 | always pair pruning with a domain floor — our mix4 has 11 domains, so a floor is cheap |
| **Up to 4 epochs** of repeated data ≈ negligible loss change vs unique data | 2305.16264 | 2 GB × 4 epochs is legitimate: 8 GB of effective tokens |
| The best repetition count **shifts with model scale**; 16→8 repeats improved a 520M run | 2610.05126 | do not assume "more epochs is better"; measure at our scale |

## What this says about "2 GB → BPB 1.0"

Three multipliers, and only one of them is currently unmeasured on our side:

```
ByteFlow            ×4-8   measured here: −0.43 BPB at 142M (capacity-matched)
perplexity pruning  ×1.45  literature (2405.20541); our heuristics gave only ×1.07
4 epochs            ×4     bounded by 2305.16264 (≤4 is ~free), with a scale caveat
```

`2 GB × 4 epochs × 1.45 × (ByteFlow 5)` ≈ **58 GB-equivalent** — which is the
regime where our own curve says ≈1.0-1.1 BPB. Without ByteFlow it lands around
1.2-1.3. So: **2 GB → 1.0 is possible only if all three stack, and ByteFlow is
the load-bearing one.**

## Our measured numbers vs these claims

| quantity | literature | ours |
|---|---|---|
| heuristic filter multiplier | — | **×1.07** (`mix5`, measured) |
| near-dup on Common Crawl | 30-50% | **0.06%** — our source is already deduped upstream |
| perplexity pruning | ×1.45 steps (125M ref) | **never run** |
| ≤4 epochs free | yes, to 9B/900B tokens | **never run at our scale** |
| quality as a scaling-law knob | Q with two estimators | we report BPB only, no Q |

The gap is not knowledge, it is **two unrun experiments**.

## Experiments, cheapest first

1. **PPL pruning** (`scripts/prune_by_perplexity.py`, written): score 4 KB windows
   with the `longctx1` checkpoint, keep the **middle band** (drop the lowest and
   highest deciles), emit `mix6/web_pruned.bin` + `.docs.json` + a manifest with
   the band edges and per-decile byte counts. Forward-only on 2.35 GB ≈ 5 h.
2. **A/B**: raw 2 GB vs pruned (same byte budget) — 600 steps × 3 seeds, our
   recipe. This is the number that decides whether the ×1.45 exists for us.
3. **Epoch A/B**: 1 epoch of 2 GB vs 4 epochs of 0.5 GB at equal compute —
   answers "repeat or broaden" for a data-constrained box.
4. Only then: ByteFlow at 8 KiB context, because that is the ×4-8 one.

## What I would not do

* Train a classifier on a "high quality" reference set — 2510.00866 shows the
  gain comes from removing noise, and our only reference set would be our own
  output (a circularity).
* Chase ×3-10 from heuristics: we measured ×1.07 and the literature's big
  multipliers are model-based, not rule-based.
* Exceed 4 epochs: 2305.16264 bounds it, 2610.05126 shows the optimum moves with
  scale, and our `max_epochs` guard already refuses loudly.
