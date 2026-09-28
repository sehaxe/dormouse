# A/B protocol (2026-09-27, instrument corrected 2026-09-28) — how every mechanism gets judged

The program's rule is "A/B or death: a mechanism either wins on held-out BPB
or it is deleted". This file makes that rule executable, because three
measurement bugs (this header) and one statistical finding have invalidated the
naive version of it.

> **No arm in this queue has been judged, and the instrument itself was broken
> when the queue was written.** Three facts, all verified 2026-09-28 against
> `~/logs/*.log` and the `checkpoints/*.config.toml` snapshots each run wrote:
>
> 1. **The attention arm had no gradient** (`8fa5d4c`) — the hand-rolled
>    autodiff node was a leaf under `BalancedCheckpointing`, so every control
>    on record is a network with attention frozen at initialisation.
>    `train_kda_full.log` prints `fused kda=3126/0`: 3126 forwards, zero
>    backwards. **The fix is verified to compile and not verified to train.**
> 2. **The eval threw the n-gram keys away** (`7adda92`) — it passed
>    `hashed_ids = None` unconditionally, and `eval_rows` is `Some` only under
>    `--engram-ram`. So on the in-VRAM Engram path the held-out number came
>    from a network with no memory in it. It cannot bite a `--no-engram` run
>    and cannot bite the `--engram-ram` path, which is why the retraction is
>    scoped to the Engram arm and not to the project's history.
> 3. **The window is not a constant.** See below. The batch-2 ablations scored
>    20 480 B and the batch-10 recipe scored 102 400 B.
>
> A new control must clear (1) and (2) before any arm is compared to it. The
> eval line's `engram=<rows>/<arms>` and `fused kda=<f>/<b>` fields are the
> cheapest check: `engram=0/<n>` means the eval ran a memory-disabled forward,
> and `kda=<f>/0` means the attention arm ran no backward.

## The measurement instrument (fixed 2026-09-27)

- **Fixed window.** `ByteStream::rewind()` before every eval, so eval N of every
  run scores the same bytes. Before this, the same checkpoint scored 6.443 and
  6.551 on consecutive evals (stream position only), which made every
  cross-run comparison noise.
- **The window is `eval_batches × batch × seq_len` bytes**
  (`crates/dormouse-train/src/lib.rs:1417`) — **not a fixed 100 KB.** This file
  used to say "100 KB per eval (`--eval-batches 20`, 20 × 5 KB)"; that is the
  *batch-10* shape, and every run in the batch-2 ablation set reported
  `over 20480 B` — 20 480, not 100 KB, a 5× smaller window. The "20 × 5 KB"
  figure is wrong twice over: 5 KB per batch is `batch 10 × seq 512`, and at
  batch 2 it is 1 KB. `AGENTS.md` §2.6 carried the same constant, as did
  `README.md`.
  - **Consequence, and it is not a small one: a BPB is only comparable within
    one window.** The best number on record, 4.997, is a **20 480 B** window at
    depth 2; the 6.351 is a **102 400 B** window at depth 4. They are not two
    points on one curve, they are two different measurements, and no run in the
    archive has been scored on both. Every A/B in this queue must fix `batch`
    and `seq_len` across all arms and seeds, and the byte count on the eval
    line is the authority — quote it in any table.
- **No graph in the eval** (`Module::valid()`): with grad tracking on, 20 eval
  forwards held never-backwarded autodiff nodes and OOM'd the card at
  15.9/16.3 GB.
- **Anchors must be measured on the window, and no anchor in the tree is.**
  This file used to say "anchors on the same window: uniform 8.000, unigram
  5.398, 5-gram 2.911 on the 1.5 MB-train split" — and
  `crates/dormouse-data/src/bin/anchors.rs:22-27` says in its own header that
  the two windows "were never comparable": the internal `--holdout` split scores
  the *trailing quarter of its own read*, while the trainer's eval reads a fixed
  window from a different file. Four readings are in circulation, none of them on
  a trainer eval window: unigram 5.398 / 5-gram 2.911 (this line, source not
  named), unigram 5.170 / 5-gram 2.572 (`anchors.rs:3-7` and `README.md`, an
  in-corpus split of a 2 MB read), `--fit` unigram 5.011 / 5-gram 2.588
  (`anchors.rs:27-32`, the only same-*file* measurement, but over the whole
  500 MB tail rather than a window), and 2.826 vs 2.849 for one file at
  `--bytes 1M` vs `2M` — the bar moves 0.023 on the fit size alone. **Get the
  bar with `anchors --fit <filtered corpus> <eval file>` and print its `window:`
  line next to the model's.** Uniform is 8.000 by definition. What the bars do
  still support, and it is the one thing the window disagreement does not touch:
  a model that has not beaten the 5-gram line has not learned language — every
  5-gram reading (2.588 / 2.572 / 2.911) sits 2.1 to 2.4 BPB below the best
  held-out number in the archive, so no choice of bar rescues a model.
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

**That budget is void, and the replacement has not been measured.** The ~1.6
s/step came from a run whose attention backward did not execute at all
(`8fa5d4c`), so it is the cost of a model with no attention training in it. The
figure that would replace it — the tensor-op path carrying a real backward at
batch 8, reported as **25.8 s/step** against **3076 ms** without the arm — was
taken by the 2026-09-28 audit and has **no committed log and no
`benches/history.tsv` row**, so it is a single unreproduced observation and is
recorded as such rather than as the new budget. If it holds, 3 seeds × 2k steps
is ~14 h *per run* and the queue has to be re-scoped (fewer arms, or fewer
steps, or a smaller batch) before it is worth starting. **Measure one step's
wall time with `--timers` on a quiet GPU at the intended batch, and write it in
`benches/history.tsv`, before planning anything on this table.** The `cost`
column below is the old figure and is kept only so the rows line up; treat it as
"unknown".

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

Three of these rows are not merely unrun — they are **undefined against the old
control**, and running them as written would produce a number that means
nothing:

- **Row 0, the control, has to be re-baselined first.** Every control on record
  is a pre-`8fa5d4c` run, so its attention arm was frozen at initialisation. A
  fresh control must show `fused kda=<f>/<b>` with `b > 0` on its eval line.
- **Row 5 (KDA decay form) presupposes KDA trains.** Tuning the decay of an arm
  that receives no gradient measures initialisation noise. This row is blocked
  on the backward gate, not on the flag.
- **The hashed-memory arm that follows it (25 000 rows/order) is blocked on the
  eval**: with the in-VRAM Engram on, a broken eval scores the arm with no
  memory in it, which is exactly how the first Engram comparison came to look
  like a 1.46 BPB loss. Check `engram=<rows>/<arms>` is non-zero on every eval
  line of both arms.

## What is NOT an A/B

- Step-time claims need the same-shape measurement (`--timers` at the same
  batch/seq, on a quiet GPU): the 2026-09-27 ablation (`--no-kda` 956 -> 188 ms)
  was measured alongside a live run and is a ratio, not an absolute. It is also
  the only step-time measurement of removing the attention arm, and the arm was
  not training at the time — so it prices neither the arm as it was nor the arm
  as it now is. Do not reuse it as either.
- **A BPB from two different windows is not a comparison.** The window is
  `eval_batches × batch × seq_len`; 4.997 is 20 480 B and 6.351 is 102 400 B.
  If an A/B table has one arm at a different batch size from its control, the
  table is void regardless of what the numbers say.
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
  launch-overhead floor dominates below ~80M parameters. **Both cost
  attributions above are now suspect in the same way**: they were measured on a
  run where the attention arm executed no backward (`fused kda=<f>/0`,
  `8fa5d4c`), so "80% of a step" is 80% of a step that skipped the backward.
  Re-measure before using this bullet to argue depth.
- **Quality.** The adaptive-depth survey found the peak-then-fall effect in the
  literature: Ouro-1.4B drops 20.47% past its peak, RecurTrace measures fixed
  depth 54.71@2 degrading to 47.54@16, and Huginn tokens that "settle" at depth
  8 carry 0.00 at >=32. Extra depth is not free capacity on a shared-weight loop.

So `--max-iter 2` halves the dominant cost AND tests the quality claim. Run it
with `--eval-depths`, which prints the held-out BPB at depths 1..=max_iter at
every eval for free (no extra training) - that curve is the deliverable, because
it says whether the model is even using the depth it pays for. Note that
`--eval-depths` had the **same** missing-keys defect as the main eval until
`7adda92` fixed both in one hunk, so no pre-fix depth curve in any log measured
a model with memory in it.

**Measure the noise floor first.** The paired-eval resolution on the fixed
window is estimated at 0.002-0.005 BPB but is NOT VERIFIED, and the real floor
is the training seed (seed variance at 151M exceeded every recipe difference we
measured). Concretely: evaluate the SAME checkpoint twice at the same depth and
confirm the numbers are bit-identical; then run two A0 seeds before believing
any win smaller than their spread. And fix the window before doing any of it —
"the same checkpoint twice" is only a resolution measurement if the byte count
on both eval lines is identical, which is a separate check from re-running the
same command.
