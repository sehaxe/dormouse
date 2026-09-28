# A/B protocol (2026-09-27) — how every mechanism gets judged

The program's rule is "A/B or death: a mechanism either wins on held-out BPB
or it is deleted". This file makes that rule executable, because two
measurement bugs and one statistical finding have invalidated the naive version
of it.

## The measurement instrument (fixed 2026-09-27)

- **Fixed window.** `ByteStream::rewind()` before every eval, so eval N of every
  run scores the same bytes. Before this, the same checkpoint scored 6.443 and
  6.551 on consecutive evals (stream position only), which made every
  cross-run comparison noise.
- **100 KB per eval** (`--eval-batches 20`, 20 x 5 KB), and the eval line
  prints the byte count, so a curve documents its own protocol.
- **No graph in the eval** (`Module::valid()`): with grad tracking on, 20 eval
  forwards held never-backwarded autodiff nodes and OOM'd the card at
  15.9/16.3 GB.
- **Anchors on the same window** (`cargo run -p dormouse-data --bin anchors`):
  uniform 8.000, unigram 5.398, 5-gram 2.911 on the 1.5 MB-train split. A
  model that has not beaten the 5-gram line has not learned language, whatever
  its held-out number looks like.
- **NaN firewall**: a non-finite loss is masked to a no-op step and counted on
  the host; > 8 non-finite reads in one window is a hard stop. The count is a
  lower bound between reads and says so.

## The statistical requirement: >= 3 seeds

At 151M, an independent study found **seed variance larger than every recipe
difference it measured** (research/2026-09-27-posttraining-compare.md, citing
ufakzeka-1). Consequence: a single arm vs a single control does not decide
anything at our scale. Every A/B is **3 seeds per arm**, same data order
(the stream is seeded deterministically, and `sample_depth` is a function of the
step index, so a seed is a config difference only).

Budget: 2000 steps at ~1.6 s = 53 min per run, so one arm = 2.7 GPU-hours.
Four arms (control + 3) = ~11 GPU-hours. This is the price of an honest
verdict, and it is why mechanism work is prioritised rather than run in
parallel: the GPU is the bottleneck, not the ideas.

## The queue, in the order the evidence says to run it

| # | arm | flag | what it decides | cost |
|---|-----|------|-----------------|------|
| 0 | control | current recipe | the reference curve | 2.7 h |
| 1 | **pure CE** | `--jepa-weight 0 --dspark-weight 0` | do the aux losses earn their 26% of the step? (they have never been A/B'd, and the audit says they go if they lose) | 2.7 h |
| 2 | **dense FFN** | `--set use_tsct=false` | do the TSCT factors, the polar retraction and the quant machinery earn ~1000 lines? (the dense arm has a narrower FFN at the same param budget - that is the comparison that means something) | 2.7 h |
| 3 | **working set** | `--data .../real_ws16` | does 4 epochs over 4.8 GB beat one pass over 19 GB at equal steps? (the byte-LM recipe research's central claim) | 2.7 h |
| 4 | **rand depth** | `--rand-depth` | does trained depth-robustness pay, or is fixed-4 better? | 2.7 h |
| 4b | **depth 2 vs 4** | `--max-iter 2` | the cheapest and highest-leverage arm in the queue - see below | 2.7 h |
| 5 | **KDA decay form** | (needs the flag from the KDA agent) | our decay starts at alpha ~0.077 (a ~9-token memory); the FLA reference starts at alpha 0.2-0.999. If a longer effective memory helps, this is a technology REPLACE, not a tuning knob | 2.7 h |
| 6 | **hashed memory (Engram)** | default (in) vs `--no-engram` | RUN 2026-09-27 at the program's operating depth (`--max-iter 2`), pure CE, 3 seeds per arm, 2000 steps, batch 20 x s512. The arm ships at 25_000 rows/order x 3 orders x 32 dim = 2.4M memory params (24% of the model) behind a hard floor (`lam = min(w_mem, 0.5)`); verdict and numbers below | 5.4 h |
| 6b | **memory capacity ladder** | `--set engram_rows=100000` / `500000` | one seed per rung, not three: the measured slot-count curve (arXiv 2601.16531) peaks at 500K/order but on a 125M backbone - at ours 500K is 48M params = 86% of the model, the monopoly shape. Rung 6 says whether the arm earns its 24%; this says whether 24% is the right rung | 3.6 h |

Rule for reading a result: an arm wins if its mean held-out BPB at the same step
count is below the control's mean by more than the spread across the control's
own 3 seeds. Anything inside that spread is "no difference", and a mechanism
that cannot show a win over its own seed noise gets deleted, not kept "because
it might help later".

## What is NOT an A/B

- Step-time claims need the same-shape measurement (`--timers` at the same
  batch/seq, on a quiet GPU): the 2026-09-27 ablation (`--no-kda` 956 -> 188 ms)
  was measured alongside a live run and is a ratio, not an absolute.
- Never run two GPU processes at once. Doing so corrupted the cubecl pool,
  which then wrote garbage weights into a checkpoint and killed a run
  (2026-09-27: official_v5, three wasted resumes). The ram-guard stops a
  runaway; it does not stop two legal processes from colliding.

## Arm 4b: depth 2 vs 4 is the cheapest big lever (2026-09-27)

Two independent lines of evidence now point the same way, and both are measured:

- **Cost.** Each loop iteration is a FULL-sequence gated-delta pass, and that
  pass allocates 17 fresh tensors / 248 MB of saved scratch
  (research/2026-09-27-kda-sota-ceiling.md). Four iterations is ~1 GB/step of
  allocator traffic against a 7 ms arithmetic budget and a cubecl pool that is
  high-water and never frees. That is why KDA is ~80% of a step, and why a
  *perfect* KDA kernel would still leave the step at ~365 ms: the 465 ms
  launch-overhead floor dominates below ~80M parameters.
- **Quality.** The adaptive-depth survey found the peak-then-fall effect in the
  literature: Ouro-1.4B drops 20.47% past its peak, RecurTrace measures fixed
  depth 54.71@2 degrading to 47.54@16, and Huginn tokens that "settle" at depth
  8 carry 0.00 at >=32. Extra depth is not free capacity on a shared-weight loop.

So `--max-iter 2` halves the dominant cost AND tests the quality claim. Run it
with `--eval-depths`, which prints the held-out BPB at depths 1..=max_iter at
every eval for free (no extra training) - that curve is the deliverable, because
it says whether the model is even using the depth it pays for.

**Measure the noise floor first.** The paired-eval resolution on the fixed
window is estimated at 0.002-0.005 BPB but is NOT VERIFIED, and the real floor
is the training seed (seed variance at 151M exceeded every recipe difference we
measured). Concretely: evaluate the SAME checkpoint twice at the same depth and
confirm the numbers are bit-identical; then run two A0 seeds before believing
any win smaller than their spread.
