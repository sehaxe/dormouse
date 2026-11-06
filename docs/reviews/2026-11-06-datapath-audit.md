# 2026-11-06 — data path and label alignment, `flagship2`

**Scope.** The six questions the brief asks about *bytes and labels*: the
(x, y) shift in the trainer's real batch, stream/seam integrity, train↔eval
overlap and distribution, document masking and padding, the mix plan's realized
shares, and the byte count actually consumed. Nothing about the model, the
optimizer, the eval arithmetic or any mechanism: four sibling reports own
those. No file under `port/`, `crates/` or `docs/` was modified except this
document; no checkpoint was written; no commit, no push.

**Instrument.** The live Python trainer, imported as a library and driven with
the live run's own arguments — `MixtureStream(load_mix_plan("port/configs/mix_v7.toml"),
batch=8, seq_len=1024, steps=513000, seed=1)`. That is byte-for-byte the
constructor `train/loop.py:1080-1088` calls. Read-only, CPU only
(`CUDA_VISIBLE_DEVICES=""`); the live 171k-step job was not touched. Static
byte work is plain Python (`bytes`, `mmap`-free chunked `find`, dict n-grams).

**Tags.** `[M]` measured on this box, instrument named. `[D]` derived arithmetic
from measured numbers. `[U]` unmeasured — an explicit absence, not a zero.

**State of the tree.** Commit `c057e05` plus the working-tree diff, which is
optimizer-only (`train/optim.py`, one kwarg in `train/loop.py:1191`); no
data-path line differs from what the live process (PID 716823, started 07:59)
runs. Live log: `~/logs/flagship2.jsonl` + `flagship2_stdout.log`, last eval
step 64 000, BPB 1.7423, `fin` 1.626.

---

## 0. Verdict

> **The data path is CLEAN.** After six groups of checks I found **no defect
> that changes a single training byte or a single label**: the shift is exact,
> the stream is byte-identical to `read()` at every offset tested including
> across refills, no source is re-read in this run, train and eval are
> disjoint, there is no padding or separator, and the realized mixture is
> within 0.23 percentage points of the plan in every stage. **"No defect found
> after 6 check groups" is the answer to the question the brief asked.**
>
> Three things around the data path *are* wrong, and none of them is a wrong
> byte:
>
> 1. **The no-leak gate is aimed at a corpus this run does not train on**
>    (`manifest.json` proves disjointness against `mix4/web`; v7 has no `web`
>    source). Today's fact is clean — I measured it — but nothing in the
>    pipeline would fail if `fwedu` overlapped the eval.
> 2. **The eval file's provenance is unrecorded, and it is from a corpus that
>    is not on this box**: not a carve of any of the 12 sources, not of
>    `mix4/web.bin`, and stylistically not the 46 GB raw crawl. It cannot be
>    re-derived. Its cross-corpus penalty is +0.24…+0.36 BPB at the order-3
>    instrument and +0.17…+0.18 BPB at the model (`fin`), per the sibling
>    eval report; that is the largest single term in the apparent 0.4.
> 3. **The brief's "0.53 GB seen" is 3× low.** Bytes/optimizer step is
>    24 576, not 8 192: at the last eval (64 000) the model has consumed
>    **1.573 GB**, and the previous run's 4.2 GB endpoint is step 171 000.
>
> A byte-level model at 1.57 GB is not "late"; the honest remaining gap is
> ~0.12 BPB of deep-supervision accounting (`fin` vs headline, sibling's lane)
> plus ~0.18 BPB of eval-corpus mismatch, both of which are *measurement*, and
> then whatever the mechanism/control lanes find.

---

## 1. (x, y) shift — CLEAN

The one definition of the shift is `_windows` (`train/data.py:70-79`):
`batch × (seq_len+1)` bytes → `view(batch, seq_len+1)`; `x = ids[:, :-1]`,
`y = ids[:, 1:]`. The training loop calls it through
`stream.next_batch()` (`train/loop.py:1753`) and hands `x, y` to the forward
unchanged (`loop.py:1483`); the model gathers `log_softmax(logits)[y]` with no
second shift (`model/model.py:985`, `model/loopblock.py:1427`).

A real batch, micro-step 0, source `fwedu` `[M]`:

```
x[0,:32] = b'- It means objects are Garbage C'
y[0,:32] = b' It means objects are Garbage Co'
y[:, :-1] == x[:, 1:]  ->  True   (all 8 rows, all 1023 positions)
```

and directly against the file (`fwedu.bin` offset 0):
`file[0:32] == x[0,:32]` and `file[1:33] == y[0,:32]` — the target is the next
byte of the source, not the same byte and not two ahead.

| check | result |
|---|---|
| `y[:, :-1] == x[:, 1:]` | **True**, 8/8 rows |
| target == source byte at `p+1` | **True** for row 0, and for a 16.8 MB read (§2) |
| `y == x` (copy positions) | **1.94 %** — consistent with English double letters, no artificial copy |
| NUL bytes in `x` / `y` | **0 / 0** — no padding |
| positions per optimizer step | 3 micro-batches × 8 192 = **24 576**, mean over micro-batches (`loop.py:1504`) |

**Geometry note (not a defect).** Each row is a 1 025-byte slice with
`x = [0..1024)` and `y = [1..1025)`, so consecutive rows are *not* adjacent by
one byte (`y[b,1023] != x[b+1,0]`): the byte at `b·1025+1024` is used as a
**target but never as context**, i.e. 1 byte in 1 025 (0.098 %) of the stream
is scored without ever being attended to. Every byte is still a target exactly
once, and the geometry makes a batch one contiguous 8 200-byte stretch of one
source rather than 8 unrelated documents. Both are the documented design
(`data.py:70-79`); neither inflates the loss.

## 2. Stream correctness — CLEAN

`ByteStream` is a ring over `[lo, hi)` of one file, refilled in 8 MB chunks,
wrapping to `lo` at `hi` (`data.py:372-384`); `_read` serves the ring and the
window is cut from it (`data.py:386-397`).

| test | result |
|---|---|
| 16 801 816 B read through the stream vs `f.read(n)` of `fwedu.bin` — spans **two** 8 MB refills | **identical** |
| `skip_bytes(1 000 000 003)` then one batch vs `f.seek(n)` | **identical** (the resume path is exact) |
| 8 consecutive batches (65 600 B) in one stream run | contiguous forward slice, identical to `f.read` — no row-level reorder |
| source re-read within the run? | **no** — the worst source reaches **0.716 epochs** (`instruct`; `fwedu` 0.154), so no `hi`→`lo` wrap occurs before step 171 000 |
| duplicated windows | none (each source is read strictly forward, once) |

The wrap path is *unreached* here, and that matters for the one counting defect
in §7 (D2).

## 3. Train↔eval overlap and distribution — disjoint, but a different corpus

**Overlap: 0.** `[M]` 1 000 windows of 64 bytes taken from the scored range
(bytes 0…65 599 of `web_eval.bin`) hashed with BLAKE2b-64: **0 matches** in
1 000 windows sampled from each of the 12 sources' consumed prefixes; the same
at width 32 bytes: **0**. Stronger: a 4 KB block from the eval's head and a
4 KB block from its middle appear **nowhere** in any of the 13 candidate files
(23.9 GB of mix sources + `mix4/web.bin` 2.35 GB), by a chunked byte search.
No leak. (`[U]` the 46 GB `real/corpus.bin` and its filtered sibling were not
byte-probed — see §3b.)

**Distribution.** The eval is not the training distribution:

| | H0 (bits) | bpb1 | bpb2 | non-ASCII | newlines/1 KB |
|---|---:|---:|---:|---:|---:|
| eval scored window (65 600 B) | 4.679 | 3.642 | 2.704 | 0.25 % | 6.24 |
| eval whole file (6.29 MB) | 4.597 | 3.656 | 3.024 | 0.49 % | 4.81 |
| train `fwedu` 512 KB | 4.528 | 3.592 | 2.884 | 0.82 % | — |

(the brief's whole-file anchors reproduce: order-1 3.656 vs 3.656, order-2
3.024 vs 3.024, order-3 2.387 vs 2.355 — in-sample counts; the order-3 gap is
the estimator's, not the corpus's.)

### 3b. Which corpus is the eval from? `[M]`

Smoothed interpolated order-3 model (identical estimator, 4 MiB fit), bits/byte:

| fit (4 MiB) | → eval head 65 600 B | → `fwedu` head 65 600 B |
|---|---:|---:|
| `fwedu` (train source) | **2.835** | **2.473** |
| `web_eval` itself | 2.788 | 2.513 |
| `real/corpus.bin` 46 GB raw crawl | 3.010 | 2.637 |
| `real_filtered/corpus.bin` | 3.094 | 2.679 |

Two readings:

* **Cross-corpus penalty: +0.36 BPB** (fwedu-fit: 2.473 in-domain vs 2.835 on
  the eval), symmetric in the other direction (+0.27), so the two corpora are
  genuinely different text, not one uniformly harder. The model's own
  measurement of the same thing is **+0.17…+0.18 `fin` BPB** (sibling report
  `2026-11-06-bpb-accounting.md` §5); a neural model transfers better than a
  trigram, so the trigram figure is an upper bound.
* **The eval is not the raw crawl**: raw and filtered-crawl fits are *worse* on
  it (3.010 / 3.094) than the `fwedu` fit (2.835), and a 4 KB probe of it is
  absent from both files. It is a third web-text corpus with no record of
  which one.

`[M]` Is the *scored window* representative of its own file? 95 disjoint
64 KiB windows, trigram fit on a 4 MiB block of the same file: mean 2.688,
**sd 0.115**, min 2.464, max 3.170; the scored head is 2.788 → **86th
percentile** (harder). The neural sweep of the same question (sibling, `fin`)
says the opposite: head 1.6559 vs 10-window mean 1.7129 and whole-file 1.7330
→ **0.06–0.08 BPB easier**. The two instruments disagree in sign, so the window
choice is worth **±0.1 BPB and no more**; it is not a defect, it is the
resolution of a 64 KiB eval. The *corpus* effect is the one both instruments
agree on.

## 4. Document masking and padding — CLEAN

`doc_masking=false` in the live snapshot. The loss path then takes
`loss_mask=None` (`loop.py:1753`) and `ce_final = ce_final.mean()` over all
8 192 positions (`model/model.py:985-994`); nothing is ignored, so nothing can
be padding. The stream is a byte-for-byte slice of the source (§2), so there is
no inserted separator of any kind; NUL count in the drawn batch is 0. 1.94 % of
targets equal their input byte — the natural English double-letter rate, not an
artifact.

**The 1 024-byte rows do glue documents**, and that is measured, not asserted:
`fwedu` row 0 begins mid-sentence ("- It means objects are Garbage Collected
more quickly. (Incorrect).") because documents are concatenated with `\n` at
build time and the ring cuts 1 025-byte slices across them. This is the
documented, deliberate `doc_masking=false` behaviour (the arm is off by
default); it adds no artificial token and no scored position, it only means the
first few target positions of a row are unpredictable from an unseen prefix.
`[U]` its BPB cost is not measured here; the arm that removes it
(`doc_masking=true`) is unrun on this checkpoint.

## 5. The mix plan's realized shares — CLEAN

`source_for_step` is a pure hash of `(seed, micro-step)` against the stage's
cumulative weights (`mixture.py:374-384`), stages split at
`round(until_frac × 513 000)` micro-batches. Replayed for all 513 000 draws and
measured over the first 2 000 optimizer steps `[M]`:

| stage (opt steps) | declared vs realized, worst source |
|---|---|
| S1 `[0, 71 820)` 42 % | fwedu 0.580 / **0.5803**; ru 0.165 / 0.1637; cosmo 0.120 / 0.1197 |
| S2 `[71 820, 136 800)` 80 % | fwedu 0.400 / **0.3994**; code 0.130 / 0.1299; ru 0.120 / 0.1203 |
| S3 `[136 800, 171 000)` 100 % | code 0.240 / **0.2423**; fwedu 0.220 / 0.2198; math 0.090 / 0.0911 |

12/12 sources used, no silent drop to one source; every stage is within **0.23
percentage points** of its plan (largest: S3 `code` 0.2423 vs 0.2400), and the
first 2 000 optimizer steps are within 0.77 pp (binomial noise: 1.2 sd for a
58 % source over 6 000 draws). Full-run bytes per source (exact replay of the
draw):

| source | MB | share | epochs |
|---|---:|---:|---:|
| fwedu | 1 848.7 | 43.95 % | 0.154 |
| ru | 547.1 | 13.00 % | 0.182 |
| cosmo | 438.9 | 10.43 % | 0.146 |
| code | 411.6 | 9.78 % | 0.597 |
| wiki_en | 177.5 | 4.22 % | 0.381 |
| math | 155.6 | 3.70 % | 0.246 |
| science | 155.3 | 3.69 % | 0.078 |
| finemath | 131.6 | 3.13 % | 0.088 |
| reasoning | 116.0 | 2.76 % | 0.657 |
| wiki_ru | 88.9 | 2.11 % | 0.387 |
| instruct | 86.2 | 2.05 % | 0.716 |
| rudialog | 49.3 | 1.17 % | 0.412 |

No source exceeds one epoch → the corpus is seen **once**, so "4 epochs over a
smaller corpus" is *not* what this run is doing and repetition cannot explain
the loss.

**One alignment fact worth naming**: the last 20 % of the run (S3, 0.84 GB) is
24 % code, 17 % math+finemath, 9 % science, 8 % RU and **22 % `fwedu`** — so
the model's most recent 0.84 GB is ~67 % not-English-prose, and it is then
scored on English web prose. That is a plan-vs-eval mismatch, not a data bug
(see D3).

## 6. Bytes actually consumed — CLEAN (the brief's figure is not)

`[M]` from the log: `mix_plan=… steps=513000 micro-batches (171000 x
grad_accum=3)`; per 100-step window `forward=300` and
`engram_rows=22 118 400` = 300 forwards × 3 loop iterations × 8 192 positions ×
3 orders (`engram_arms=900` = 300 × 3), i.e. exactly 3 forwards of 8 192
positions per optimizer step; `nan_skipped: 0` in **all 661** log
windows; one `run` event with `"resumed": 0` (no resume, so no `skip_bytes`,
and no double-consumed region).

| quantity | value |
|---|---|
| scored bytes per optimizer step | 8 × 1024 × 3 = **24 576** |
| bytes served by the streams per step | 3 × 8 200 = **24 600** (+0.098 % read, not scored — §1) |
| full run, scored | 171 000 × 24 576 = **4 202 496 000** |
| full run, served | 513 000 × 8 200 = **4 206 600 000** |
| **bytes seen at step 64 000** | **1 572 864 000** (1.573 GB) |
| **bytes seen at step 66 000** | **1 622 016 000** |
| bytes seen at the brief's "0.53 GB" | that is **step 21 600**, i.e. 0.526 GB |

⇒ The brief's "BPB 1.742 at 0.53 GB seen" is a 3× undercount of the *data
budget* (the brief used `batch × seq_len` = 8 192 B/step and forgot
`grad_accum`). The live run is at **1.57 GB** at its last eval, 37 % of the
4.2 GB budget — the same conclusion the sibling eval report reaches
independently. Any "it should be at 1.2–1.35 by now" argument has 3× more data
behind it than the brief assumed.

---

## 7. Defects, each with the fix and the gate

### D1 — the no-leak gate is aimed at the wrong corpus (gate defect, not a leak)

`data_local/mix4/manifest.json` records `web_eval` as a `doc_sha1_disjoint`
carve of **`mix4/web`** (`eval_units: 1, train_units: 373, overlap: 0`). The
port's `open_streams` builds the eval stream with **no relationship check at
all** between the plan's sources and the eval file (only the Rust
`ByteStream::train_and_eval` checks anything, and that is path containment:
`crates/dormouse-data/src/lib.rs:538-563`). `mix_v7.toml` contains **no `web`
source**: its largest source is `fwedu` (12 GB, 44 % of bytes) and the
manifest's proof says nothing about it. Nothing in the pipeline would fail if
`fwedu` and `web_eval` shared documents.

**Fix.** In `MixtureStream.__init__` (and `open_streams` for the single-file
mode), hash 4 KB shingles of the eval file's first `eval_window_bytes` and of
each source's **exactly the sequential prefix `per_source_bytes[name]` says
it will read**, and `loud()` on any intersection, naming both files and the
offset. That is complete for the run's actual consumption: 65 KB of eval +
4.21 GB read once, ~1 min on this disk, before step 0.

**Gate.** `tests/test_data.py::eval_windows_are_disjoint_from_every_mix_source`
— build a plan whose second source is a file containing 4 KB copied out of the
middle of the eval file; assert `MixtureStream(...)` raises. (Without the fix
the test is green-by-construction on any pair, which is what the manifest's
`overlap: 0` already demonstrates.)

### D2 — a re-read past `hi` is uncounted (ADR-0019 COUNTED class)

`ByteStream._refill` wraps `self._pos = self.lo` when a source is exhausted
(`data.py:376-382`) with **no counter and no log line**. `max_epochs` is checked
against an *expectation* (`check_epoch_cap`, `mixture.py:478-505`) before the
run; the realized epochs are computed by `replay_bytes` but never compared with
what the stream did. This run does not wrap (worst source 0.716 epochs), so no
byte is wrong today; a plan one decimal further out would repeat a source's
head in silence, which is exactly what `mix_v7.toml:83-88` says must be LOUD.

**Fix.** `ByteStream.wraps += 1` on the `hi`→`lo` branch, and print
`mix_epochs=<name>:<realized>` on the eval line (or the first log line of each
stage) from the counter.

**Gate.** `tests/test_data.py::a_wrap_is_counted` — a 3-batch file read 5
batches asserts `wraps == 1` and that the served bytes are the head again.
Plus: `check_epoch_cap` must compare the **realized** `replay_bytes` epochs, not
`per_source_bytes`, which differ by up to 1.25 % in this plan.

### D3 — the eval measures a corpus the decay stage trains away from

Numbers in §3 and §5. The plan is defensible; the *instrument* is not, because
the recorded number is compared with published FineWeb-Edu valid BPB and with
this run's own earlier numbers, while the last 0.84 GB of gradient is 67 %
non-prose.

**Fix (measurement, cheap).** Add a second eval stream carved from `fwedu`
(`data_local/best/fwedu.bin`, the 44 % source) at a position beyond the run's
consumed prefix, scored with the same `rewind()`/`eval_batches`, and print
`bpb_edu=` beside `bpb=`. Report both; quote `fin` on both (§8).

**Fix (recipe, a decision, not a patch).** If `web_eval` stays the headline,
hold `fwedu`+`cosmo`+`wiki_en` at ≥ 30–40 % in the last stage — S3 currently
has 32 % in total and 22 % `fwedu`.

**Gate.** The eval line must print the eval file's identity (`sha256[:16]`) and
the plan's in-domain share at the current stage; a plan whose last stage drops
the eval file's source family below 30 % is LOUD at construction.

### D4 — `bytes_seen` is not in the log, and the brief got it wrong by 3×

§6. The trainer logs `tokens_per_s` but never the cumulative byte count, so
"how much data has this run seen" is recomputed by hand, and a hand that
forgets `grad_accum` is off by 3×.

**Fix.** One field on every log line: `bytes_seen = (step+1) × batch ×
seq_len × grad_accum` (and the same in the eval row).

**Gate.** `tests/test_train.py::log_line_carries_bytes_seen` — a 2-step run on
a tiny corpus asserts `bytes_seen == 2 × batch × seq_len × grad_accum` and that
it equals the stream's own consumed-byte counter.

---

## 8. Out of lane, but measured here (for the owners)

* **`dormouse eval --device cpu` on this checkpoint is not this model.**
  `[M]` `--eval web_eval.bin` → 6.666 BPB `fin` / 9.110 headline;
  `--eval fwedu[1GB]` → **6.676** `fin` / 9.083. The in-domain and out-of-domain
  CPU numbers agree to 0.01 BPB, which is what a broken instrument looks like.
  The mechanism report names the cause: on CPU `KDAAttention.forward_state`
  takes the COUNTED `_sdpa_path` (`kda.py:435-441`), where `g` is an output gate
  and `beta` is unused — a different network. **No CPU measurement of this
  checkpoint is valid**; I discarded my own eval-variance attempt for this
  reason and used the n-gram instrument instead.
* **Headline vs `fin`.** `[M]` at step 64 000: headline 1.7423,
  `periter=1.3936, 1.1829, 1.1271`, `fin=1.1271` → **1.626 `fin`**. The 0.116
  BPB difference is the deep-supervision average, not the model's predictive
  BPB. (Sibling `2026-11-06-bpb-accounting.md` §6 owns the arithmetic; my
  numbers agree.)
* **`warmup_tokens` is correctly scaled by `grad_accum`** (`loop.py:1325` passes
  `batch × grad_accum`), so the token-vs-step confusion in D4 does *not* reach
  the LR schedule: warmup is 163 optimizer steps and the logged lr
  3.067e-4 at step 100 = 5e-4 × 100/163 confirms it.

## 9. What I did not check

* `[U]` The 46 GB `real/corpus.bin` and 19.7 GB `real_filtered/corpus.bin` were
  compared to the eval **by style fit only**, not by a byte probe (a full
  7-probe × 65 GB scan did not finish inside the time budget). The 13 files
  that *are* mix sources were byte-probed.
* `[U]` The BPB cost of gluing documents (`doc_masking=true` vs `false`) on
  this checkpoint.
* `[U]` Whether the *neural* model's BPB is flat across the eval file — only the
  trigram instrument was affordable, because CPU eval is invalid (§8) and the
  GPU belongs to the live run. The sibling eval report has that sweep at
  step 49 999 (`fin` 1.6559…1.8539 over 10 windows).
* `[U]` Nothing was re-run: no training step, no GPU, no checkpoint touched.
