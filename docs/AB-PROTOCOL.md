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

**⚠ That parenthetical is now known to be too strong, and the protocol is
"nearly" implementable rather than implemented.** `device.seed(cfg.seed)`
(`4b42b6d`, 2026-09-28) does govern the model init and the JEPA mask, both pure
functions of `(seed, step)` — before that fix two runs with identical flags
differed in **34 730 605 of 43 725 616 checkpoint bytes**, so every A/B in the
archive compared two different initialisations and charged the difference to
the arm. **MEASURED 2026-09-29 ON CUDA — AND SUPERSEDED 2026-09-30 by a
tensor-level measurement with a self-validating parser (`57237c3`,
`tools/determinism.py`, `benches/determinism.tsv`).** The table below is kept
for the record; every number in it was an artifact of measuring slot COUNTS
and misparsing the file (tensor offsets read from base 0). The count is not a
distance: the same nominal condition read 0.0000 % … 12.51 % across
repetitions while relative Frobenius stayed ≤ 1.24e-08. Three zero-step runs
of one binary, same flags, `--seed 7`, `small`, batch 2, seq
128, `--no-kda --no-engram`, 8 KB corpus, compared as f32 slots of the
checkpoint:

| pair | when | differing slots (of 18 398 028) | % |
|---|---|---|---|
| A vs B | consecutive, seconds apart | 30 682 | **0.167 %** |
| A vs C | same flags, minutes later | 2 299 790 | **12.50 %** |
| B vs C | | 2 301 566 | **12.51 %** |

**The residue is not a stable set.** The differing index sets of A/B and A/C
overlap at Jaccard **0.012** — essentially disjoint. A fixed uninitialised
region would be the same indices every time. The magnitudes are not rounding
either: median |delta| 3.4e-8 against **max 2.07e+38**.

**Four things: three established, one open.**
1. ~~**CPU is deterministic.**~~ **WITHDRAWN 2026-09-30.** The claim rested
   on a test that built the model twice and compared - and `Device::seed()`
   does not rewind a consumed stream, so the second build was never a same-seed
   build. Nothing is known about cross-process CPU reproducibility. See
   `model_seam::seeded_init_is_a_pure_function_of_the_seed_on_a_clean_stream`,
   which now asserts only that two different seeds do not collide.
2. ~~**CUDA is not**, and the magnitude depends on WHEN the run happened
   (0.17 % back-to-back, 12.5 % later). That points at machine state — the
   cubecl pool is documented high-water and never frees (`AGENTS.md` §2.2) —
   rather than at a fixed unseeded RNG in the init path.~~ **SUPERSEDED
   2026-09-30 by two tensor-level instruments (`57237c3`, `58281c8`..`8da23bb`,
   `tools/determinism/`).** Same seed across processes: all non-TSCT slots
   bit-identical, TSCT factors ≤ 1.04e-06 relative (effective W), one pair
   bit-identical at 860 s; different seed relFro 1.414–1.415 — separation
   ~10⁶; no wall-clock gap effect (ratio 1.026). The seeded RNG stream is
   proven correct (direct draws bit-identical across 9 processes). The
   residue is nondeterministic reductions in `qr_householder`
   (`burn-spectral/src/lib.rs:697-714`), entered at a random iteration —
   13.5 % of the model, a §1.3 violation carried openly, not machine state
   and not an unseeded draw. Compounding bound: 20 steps (the 50/100 rows
   never ran).
3. **409 043 is not reproducible and must not be cited as measured.** It sits
   between two values the same experiment does not produce consistently, and
   its source `4b42b6d` shipped no test.
4. ~~**OPEN, and it changes the method: the checkpoint FILE is not a sound
   instrument.** It carries 0–4 NaN slots and ~475–501 slots at |v| >= 1e30,
   mostly at stable indices (475 of ~490 overlap between runs) with a varying
   remainder, so a byte-comparison of two checkpoints measures some of that
   too. A trustworthy check compares the **tensors**, not the files.~~
   **CLOSED 2026-09-30.** The NaN/1e30 observations were the parser bug, not
   the file: read correctly, all 8 runs carry **0 NaN and 0 slots at
   |v| ≥ 1e30**. The method conclusion survives on its own merits — compare
   tensors, not files — and that is exactly what `tools/determinism.py check`
   does, with a self-validating parser whose guards each have a demonstrated
   failure.

**~~Until 2 and 4 are closed, two runs under one seed are not the same run and
the 3-seed spread may carry a per-run component.~~ CLOSED 2026-09-30.** Two
runs under one seed ARE the same run to 1.24e-08 relative Frobenius, and the
cross-seed separation is 1.1e8× wider than the same-seed residue. **The 3-seed
protocol of §1.2 is implementable as of today**; the per-arm cost stays
unknown until the smoke run re-prices it (below).

**~~Budget: 2000 steps at ~1.6 s = 53 min per run, so one arm = 2.7 GPU-hours.
Four arms (control + 3) = ~11 GPU-hours.~~ STRUCK 2026-09-29. The 1.6 s/step
was a model with no attention training in it, and a step time is not a number
without its step index.**

**The cost of one arm is currently UNKNOWN, and that is the honest state of
this table.** Both directions of the error are real and neither is a usable
budget:

- **The 1.6 s/step that priced this table is withdrawn.** It came from a run
  whose attention backward executed **zero** times (`fused kda=<f>/0`,
  `8fa5d4c`), so it is the cost of a model with no attention training in it.
- **The 25.8 s/step that would replace it is withdrawn too.** The tensor-op
  path at batch 8 reported as 25.8 s/step against 3076 ms without the arm came
  from the 2026-09-28 audit and has **no committed log and no
  `benches/history.tsv` row**. Both figures were also measured under
  conditions nobody can reproduce (load, dev profile, eval enabled), and
  `benches/history.tsv` strikes them for exactly that reason. A single
  unreproduced observation is not a budget.
- **What is measured, 2026-09-29, warm, in `benches/history.tsv`** — release,
  `small` 9 195 854 params, depth 2, fp32, aux off, `--no-engram`,
  `CUBECL_AUTOTUNE_LEVEL=3`, `--timers`:

  | batch | warm step | 2000 steps |
  |---|---|---|
  | 8 | **244 ms** | ~8 min |
  | 16 | **440 ms** | ~15 min |
  | 32 | **826 ms** | ~28 min |

  **Every one of these is a floor, not a budget**: the attention backward does
  not run in any of them, and the step it adds is the entire open question.
  A step-0 reading in the same configuration is 5549 ms — **23x** the warm
  step — so a queue planned from any un-indexed timer line is wrong by more
  than an order of magnitude in either direction.
- **One partial data point at the queue's own shape, and it is the most useful
  number here**: `tools/mor_ab.sh preflight` (`.bulba/memory.md:36`,
  2026-09-29, `small`+`mor`, batch 20 x seq 512, fp32, pure CE, release, quiet
  card, 204 800 B window) measures a **steady 1068 ms** step (fwd 251 / bwd
  674 / opt 73 / retr 69), i.e. a 2k-step arm ≈ 36 min and a 3-seed × 2-arm
  sweep ≈ 3.6 GPU-h. That same run printed `fused kda=64/0` in **both** arms,
  so it is also a floor — and it says the cost is *not* what blocks the queue.
  A control re-baseline is.

**Do not plan from the `cost` column below.** It is the withdrawn 1.6 s/step
figure, kept only so the rows line up; treat every entry as "unknown". The
cheapest way to replace it: run one step with `--timers` on a quiet GPU at the
intended batch, **at a step index past 50**, and write the row in
`benches/history.tsv` before planning anything on this table.

## The queue, in the order the evidence says to run it

<!-- `cost` is UNKNOWN in every row. The 2.7 h / 5.4 h / 3.6 h figures were derived
from a withdrawn 1.6 s/step and are struck above; see the budget section. Do not
repopulate this column from any step-time reading that has no step index. -->
| # | arm | flag | what it decides | cost |
|---|-----|------|-----------------|------|
| 0 | control | current recipe | the reference curve | unknown |
| 1 | **pure CE** | `--jepa-weight 0 --dspark-weight 0` | do the aux losses earn their 26% of the step? (they have never been A/B'd, and the audit says they go if they lose) | unknown |
| 1b | **future-byte head** | `--set aux_fb_weight=0.1` (horizon 2, default) | is the CHEAPEST auxiliary objective cheaper than the JEPA retune it would replace? The arm is **our adaptation of arXiv 2404.19737's per-horizon multi-token heads to a byte-level AR model** (untied `d_model->256` linear, CE on the byte at `t+2`, 197 120 params = 2.1% of `small`); the paper reports induction-head gains at 1-30M from lookahead prediction, and the claim reused is *that*, not its shape. Read `fb=<ran>/<asked>` on the eval line before believing any number: `0/<n>` means the horizon was at or past the sequence length and 2 000 steps trained nothing. 3 seeds, 2 000 steps, **the same `batch`/`seq_len` as the pure-CE row** or the two windows differ and the table is void. Against row 1, not against the default recipe: JEPA is on in the default. The horizon-4 row is this row's follow-up, not a flag | unknown |
| 2 | **dense FFN** | `--set use_tsct=false` | do the TSCT factors, the polar retraction and the quant machinery earn ~1000 lines? (the dense arm has a narrower FFN at the same param budget - that is the comparison that means something) | unknown |
| 3 | **working set** | `--data .../real_ws16` | does 4 epochs over 4.8 GB beat one pass over 19 GB at equal steps? (the byte-LM recipe research's central claim) | unknown |
| 4 | **rand depth** | `--rand-depth` | does trained depth-robustness pay, or is fixed-4 better? | unknown |
| 4b | **depth 2 vs 4** | `--max-iter 2` | the cheapest and highest-leverage arm in the queue - see below | unknown |
| 5 | **KDA decay form** | (needs the flag from the KDA agent) | **Provenance corrected 2026-09-29: `alpha ~0.077` is OURS, not a reference's.** `a_log = -3` / `b_alpha = +1.0` is a measured choice of this repo's; Kimi K3 §2.1.1 uses `A_h = 0` and FLA uses `inv_dt ∈ [-6.91, -2.25]` (negative, reaching `alpha ~0.95`). Under the K3 sigmoid a non-negative `z` caps `alpha` at `e^{g_min/2} = 0.0821`, so **the bias SIGN, not `A`, is the lever**, and neither knob alone gets there (`vendor/burn-fused/crates/burn-kda/src/lib.rs` module docs carry the per-source table). If a longer effective memory helps, this is a technology REPLACE, not a tuning knob | unknown |
| 6 | **hashed memory (Engram)** | default (in) vs `--no-engram` | RUN 2026-09-27 at the program's operating depth (`--max-iter 2`), pure CE, 3 seeds per arm, 2000 steps, batch 20 x s512. The arm ships at 25_000 rows/order x 3 orders x 32 dim = 2.4M memory params (24% of the model) behind a hard floor (`lam = min(w_mem, 0.5)`); verdict and numbers below | unknown |
| 6b | **memory capacity ladder** | `--set engram_rows=100000` / `500000` | one seed per rung, not three: the measured slot-count curve (arXiv 2601.16531) peaks at 500K/order but on a 125M backbone - at ours 500K is 48M params = 86% of the model, the monopoly shape. Rung 6 says whether the arm earns its 24%; this says whether 24% is the right rung | unknown |
| 7 | **AttnRes vs ReZero** | `--set use_attnres=true` | does the learned depth-attention residual earn its place against the fixed additive one at the same depth? **Full mode only** — the Block variant is not wired (it is still wrong against Eq. 6: see `research/papers/attnres-integration-2026-09-30.md` §5). 3 seeds per arm, 2000 steps, pure CE, **one batch size across both arms** so the eval window matches (§2.6). Read the `attnres` counter on the eval line before believing a number: a run that reports 0 aggregated ReZero's loss under AttnRes's name. **Not a pure weights swap** — at the mandated `w_l = 0` init iteration `n`'s state is the equal-weight mean of `n+1` sources, so the arm also changes the residual's init scale from `h + y·s` to a `1/(n+1)` average. That is the paper's stated init, and the comparison is honest about it | unknown |
| 8 | **RoPE on the KDA q/k** | `--set use_kda_rope=true` (needs the trainer flag, wired in the crate not the CLI) | **This arm is a cross-family transplant and the queue's only row whose justification is NOT pretrain parity.** FLA's official KDA layer has **no RoPE at all** — `fla/layers/kda.py` @ `9f38d249` contains zero `rotary`/`rope` occurrences, and GatedDeltaNet has no `use_rope` either. In Kimi Linear's hybrid the position comes from the *interleaved full-attention* layers (`fla/layers/attn.py:83,125`, spec at `fla/models/hybrid.py:17-23`); `get_hybrid_attention_spec`'s docstring says unassigned layers "retain their model's native mixer" (`:190-200`), which is the NoPE statement for the KDA half. So the question is not "does rope help pretraining" — it is **"is a NoPE model trainable past pretraining at all"**, and the standing answer in this repo's own playbook is no: AGENTS.md §3.5 item 5 records the Qwen3.8 finding as *NoPE → endless generation after post-training*, and the owner's goal is the full SFT→RLVR→distill loop (`POST_TRAINING.md`). A model that cannot be steered after pretraining is not a candidate for that pipeline whatever it scores at step 2000. **Run it as a post-training gate, not a BPB race:** a pretrain-parity result is not a reason to delete this arm, and a pretrain win is not a reason to keep it. If it is run as a BPB arm, 3 seeds, 2 000 steps, pure CE, one batch size across both arms (§2.6), and expect **no win** — a full-head rotation is norm-preserving, so the two arms differ only in which directions q and k are allowed to point, and at our init that is a small perturbation. Facts, formula and the placement proof: `research/reviews/rope-2026-09-30.md` | unknown |

Rule for reading a result: an arm wins if its mean held-out BPB at the same step
count is below the control's mean by more than the spread across the control's
own 3 seeds. Anything inside that spread is "no difference", and a mechanism
that cannot show a win over its own seed noise gets deleted, not kept "because
it might help later".

**Two things must be true before ANY row is run, and neither is a formality:**

- **The seed gap is closed ON THE BACKEND THE ARM RUNS ON**, measured at the
  tensor level rather than by comparing checkpoint files (see the measurement
  above: 0.167 % back-to-back on CUDA, 12.50 % minutes later, CPU exact). Until
  two runs under one seed are the same run, the spread this protocol compares
  against is not a seed spread, and a 3-seed verdict measures the seed plus
  machine state plus the arm. **This is the cheapest item on the page and it
  is not closed.**
- **A control with a gradient-carrying attention arm exists** (row 0 below).

Then, of the rows: three are not merely unrun — they are **undefined against the
old control**, and running them as written would produce a number that means
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

- Step-time claims need **three** things, and until 2026-09-29 this project
  routinely supplied at most one: the same shape (batch/seq/depth, `--timers`),
  a quiet GPU, **and a step index past 50**. `--timers` used to print on
  `step % 50 == 0` only and was not tied to `--log-every`, so an un-indexed
  reading was almost always step 0. **That defect is fixed** (`b8a47ee`: the
  cadence is now `step == 0 || step % log_every == 0`,
  `crates/dormouse-train/src/lib.rs:1196`, and a warm step is reached by step
  2), so a `benches/history.tsv` row written today carries its step index —
  quote it. The historical readings do not, and a step-0 step is **5549 ms
  against a warm ~245 ms** in the same configuration
  (`benches/history.tsv`, 2026-09-29). **A step-time reading
  with no step index is not a measurement of a step.** Under those conditions
  this file has produced, in order: `25858 ms`, `10809 ms`, and `1.58-1.84
  s/step`. All three are struck.
- The 2026-09-27 `--no-kda` ablation (956 -> 188 ms) was measured alongside a
  live run and is a ratio, not an absolute. It is also the only step-time
  measurement of removing the attention arm, and the arm was not training at
  the time — so it prices neither the arm as it was nor the arm as it now is.
  Do not reuse it as either. **Withdrawn from the cost column too:** the
  25.8 s/step batch-8 figure that would have replaced the 1.6 s budget has no
  committed log and no `benches/history.tsv` row, and the 3076 ms
  no-attention figure beside it was measured under the same unreproducible
  conditions. Both are struck in `benches/history.tsv`.
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
  high-water and never frees. That is why KDA was measured at ~80% of a step,
  and why a *perfect* KDA kernel would still have left the step at ~365 ms:
  the ~465 ms launch-overhead floor dominates below ~80M parameters.

  **Both cost attributions in this bullet are retracted, in the same way and
  for the same reason.** They were measured on a run where the attention arm
  executed no backward (`fused kda=<f>/0`, `8fa5d4c`), so "80% of a step" is
  80% of a step that skipped the backward, and the 465 ms floor was read at
  **step 0** (the `10809 ms` reading it was fitted from is a step-0 artifact —
  `benches/history.tsv`, 2026-09-29). Re-measure before using this bullet to
  argue depth.

  **What survives, and it is the part this arm is argued on: the workload is
  launch-bound, now measured directly rather than inferred from a step-time
  fit.** Over a 150-step warm run at batch 32 the GPU is **13.3 % utilised on
  average with 142 of 180 samples at <=5 %** (`benches/history.tsv`,
  2026-09-29, `nvidia-smi` at 2 Hz). The kernels are too small and too many to
  fill the SMs, so cutting arithmetic in the attention arm buys little wall
  clock while cutting *launches* buys a lot. That is the launch-bound claim the
  465 ms figure was standing in for, and it now has an instrument behind it.
  The warm step itself is **~245 ms**, so the floor argument holds with a
  smaller number than the one it was originally made with.
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

## Гигиена прогонов (2026-09-30, поправки внешнего ревью — принимаются)

- **Строка history.tsv обязана нести:** commit/config, seed, идентификатор
  фиксированного validation split, фактически обработанные БАЙТЫ, число
  обновлений, время на GPU, validation BPB, долю невалидного UTF-8, скорость
  генерации; для code-рук — результат фиксированного набора code-задач с
  проверкой выполнения.
- **Одинаковое число шагов ≠ matched-compute**: aux меняет цену шага (у JEPA —
  учительский форвард и VRAM). Первичный smoke можно сравнивать по одинаковым
  обработанным байтам; победителя по эффективности выбирать при
  сопоставимом вычислительном бюджете. На одной GPU время — практический
  ориентир при одинаковых условиях запусков.
- **Очередь строго однофакторная:** CE → CE + исправленный aux → CE + attnres.
  Не смешивать два изменения до отдельных выигрышей каждого.
- **Проверка утечки будущего — ДО aux-запусков:** target-байты не должны
  попадать в представление контекста через вход, attention или pooling.
  Формальная проверка: loss как функция от target-байтов будущих позиций —
  тождественный нуль градиента.
- **2000 шагов — цитируемый smoke для ЭТОГО окна,** не окончательный рейтинг:
  победителя повторить с другим seed и проверить на более длинном обучении.
  Убывание aux-loss без выигрыша BPB или code-задач — не основание
  возвращать ненулевой вес.
