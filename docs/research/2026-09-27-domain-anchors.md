# 2026-09-27 — Domain anchors, and what they say the mixture should be

**RE-MEASURED 2026-09-29.** Every number in the first version of this file was
ONE 2 MB window. This version re-measures every corpus at 5 window positions ×
2 window sizes × 2 counter orders — **344 successful anchor runs** (346
attempts; 2 refused a corrupt parquet shard, loudly, §5), raw rows in
`~/logs/anchor-remeasure-2026-09-29.tsv`, summary in
`~/logs/anchor-remeasure-summary-2026-09-29.txt`. **The table is a valid
RANKING and an invalid set of absolute bars; not one domain survives the
0.05-BPB reliability test.**

The data for a universal model was already on disk and already domain-separated
(`mix/`); we were training on a single filtered web mix. Every domain needs its
own bar ladder before any model result in it means anything.

## 0. The rule this file now obeys

> **An anchor is N windows, or it is a ranking, not a bar.**

- **N = 5** window positions, minimum. A single window is a *provisional
  reading* and is labelled as one wherever it appears.
- **Flag threshold: spread > 0.05 BPB ⇒ the domain is not usable as an
  absolute bar.** At 5 positions and a 0.05 threshold, **10 of 10 domains fail
  at the inherited 2 MB window and 8 of 8 fail at a one-shard window.** The
  tightest domain in the whole corpus is `web` at 0.059 (2 MB) / 0.073 (6.3 MB).
- A ranking claim needs the ranges to be **disjoint**, not the means to be
  ordered. Most of the ordering in the old table is not disjoint and does not
  survive.
- Every bar is quoted with its protocol: **counter order, fit corpus, fit size,
  scored window**. All four were unstated before today.

## 1. The instrument, and three things about it that change how a number reads

Tool unchanged: `dormouse-data`'s `anchors`, built from the working tree at
`HEAD 82993d8` — `crates/dormouse-data/**` is clean at that commit, so the
binary is HEAD's `dormouse-data` (`f6ab353`'s parquet fix is in it), release
profile, private `CARGO_TARGET_DIR=/home/sehaxe/.cache/anchors-target`, run
2026-09-29. The Sep-27 release binary in `target/release/anchors` was verified
to reproduce the new binary **exactly** on `.txt` windows (`web_00000.txt` 2 MB:
4.584 / 2.537 / 11.6% on both), so the only corpus whose reading changed is
parquet. The instrument was not modified; no file under `crates/`, `vendor/` or
`configs/` was touched by this re-measurement.

BPB from an n-gram counter is a deterministic function of the bytes it reads,
so machine load cannot bias it — these numbers are not a contended measurement
in the way a step time would be. What they are is *windowed*.

1. **Order 8 at these fit sizes is not uniformly a stronger baseline — it is a
   different bar, and which one is stronger depends on the domain.** It is
   0.30-0.87 BPB **worse** than order 5 on the five high-entropy domains
   (`web` 2.481→2.741, `reasoning` 2.406→2.793, `code` 2.520→2.829,
   `wiki` 2.527→2.814, `math` 2.614→3.086, 6.3 MB windows) and 0.11-0.20
   **better** on the three low-entropy ones (`agentic` 1.292→1.096,
   `rudialog` 2.063→1.869, `ruweb` 2.140→1.930). The predictor is the
   unseen-context rate: where order 5 already meets 7-23% unseen contexts a
   6.3 MB fit cannot fill an 8-gram table and the higher order loses; where
   order 5 meets 0-4% unseen (`agentic` 1.5%, `ruweb` 2.3%, `rudialog` 3.8%)
   the extra order pays for itself. **So "the 5-gram bar" and "the 8-gram bar"
   are not the same bar, and an anchor must say which one it is quoting; the
   one to beat is the lower of the two.** The 8-gram columns below are
   reported because they were asked for, not because they are better.
2. **The tool cannot address a window.** It reads the first `--bytes` of the
   path-sorted file list; there is no `--offset`. So "position in the corpus" is
   reachable two ways only: point it at a different shard (5 positions in the
   shard sequence), or carve the window out with `dd` first. For a single-file
   `.parquet` even that fails — the footer is at the end — so `books` positions
   are a `--bytes` sweep, which confounds position with fit size and is reported
   as such. The missing knob is one flag (`--offset`), in `crates/`, which this
   task did not touch.
3. The fit/scored split is contiguous within the read (`[0, 0.75N)` fits,
   `[0.75N, N)` scores), so the boundary falls mid-document on a parquet
   corpus. That is why `books` is non-monotonic in `--bytes`.

## 2. The canary bar — the number everything else is compared to

**There is no corpus on this machine called "canary-math", and the number
2.416 appears in no file in the repo, on the drive, in `mixture/arms.json`, in
`mixture/MANIFEST.md` or in `~/logs/`.** It is not reproducible as stated,
because no protocol was stated with it. What *is* reproducible is the bar for
the window the A/B program actually scores.

The trainer's held-out eval reads the **first `eval_batches × batch × seq_len`
bytes** of the eval dir (`train/src/lib.rs:1417`): **20,480 B at batch 2**,
**102,400 B at batch 10**, from
`real_eval_v2/eval_tail.bin`. The bar is therefore a counter **fitted on
`real_filtered_v2/corpus.bin` and scored on that exact window**, and it has two
error sources, measured separately.

### 2a. The bar, four defensible protocols

| fit corpus | scored window | 5-gram | 8-gram (weaker on this corpus, §1.1) | unseen ctx |
|---|---|---|---|---|
| 2 MB of `real_filtered_v2` | 20,480 B (batch 2) | **2.594** [2.571, 2.628] | 3.162 [3.126, 3.242] | 14.1% |
| 16 MB of `real_filtered_v2` | 20,480 B (batch 2) | **2.397** | 2.614 | 5.8% |
| 2 MB of `real_filtered_v2` | 102,400 B (batch 10) | **2.679** [2.663, 2.690] | 3.249 [3.220, 3.285] | 16.0% |
| 16 MB of `real_filtered_v2` | 102,400 B (batch 10) | **2.466** | 2.699 | 7.4% |

Brackets are min/max over **5 fit-corpus positions** (2 MB windows at 0, 2, 5,
10, 15 GB of the 20.36 GB corpus — the corpus is a per-document shuffle, so
these are independent samples of the mixture). The two 16 MB rows carry no
bracket: the fit-size sweep was run at **one** position (the front), so their
own fit-position error is unmeasured. That is a real gap, not an omission, and
it is the reason the headline bar is quoted at the 2 MB fit where the 5-position
spread is known.

### 2b. The error sources, priced

| source | what was varied | effect on the 5-gram bar |
|---|---|---|
| **which 2 MB of the fit corpus** | 5 positions over 20.36 GB | **±0.03** on the 20,480 B window (spread 0.057), **±0.014** on the 102,400 B one (spread 0.027). The filtered corpus is homogeneous; this is the *small* term. |
| **which 20,480 B of the eval tail** | 5 consecutive windows, one fixed fit corpus | **±0.26** (2.497 / 2.571 / 2.612 / 2.734 / 3.018, spread **0.521**). This is the *large* term, and on the same window it is **9× the fit term**. |
| **fit size** | 2 MB → 16 MB, same scored window | **−0.20 to −0.21** (2.594 → 2.397; 2.679 → 2.466) |
| **counter order** | 5 → 8, same bytes | **+0.57** (2.594 → 3.162) — on *this* corpus the 8-gram is the weaker counter (§1.1), so 5-gram is the bar |

### 2c. The verdict on 2.416

**2.416 does not survive as a bar.** It is not a reading this machine produces
for the math domain at any of the 10 windows measured (§2e), and for the
canary's own eval window it is 0.02 away from exactly one cell of the 2×2
above — the **16 MB fit on the 20,480 B window, 2.397** — and 0.26 away from
the 2 MB fit on the same window. It is a plausible reading of a protocol that
was never written down, not a bar.

**The defensible bar is a range, not a number:**

> **5-gram, 2 MB fit, 20,480 B eval window: 2.594 ± 0.03 (fit) — the honest
> one-line form is 2.6 ± 0.1.** Across all four defensible protocols the bar is
> **2.397 – 2.679**, a 0.28-wide band. A model number compared against a bar
> whose protocol was not written down is being compared against **±0.15** of
> protocol choice plus **±0.26** of which 20 KB window was scored.

Two things this does **not** say. It does not say the A/B program is broken:
every arm is scored on the *same* 20,480 B, so a paired comparison cancels the
window term and 0.521 is not the A/B resolution — it is the resolution for
transferring a bar to a *different* window or corpus. And it does not rescue the
current numbers: the best held-out BPB on record is 4.997 (`nokda_ce`, a
`--no-kda` model, best-of-an-overfitting curve), which is **2.3-2.6 BPB above**
the bar under every protocol in the table, so "has not beaten the counter"
stands under all of them.

### 2d. The corpora themselves, 5 positions, both windows

| corpus | window | 5-gram mean | min | max | spread |
|---|---|---|---|---|---|
| `real_filtered_v2/corpus.bin` (the training corpus) | 2 MB | 2.760 | 2.655 | 2.849 | 0.194 |
| | 6.3 MB | 2.703 | 2.580 | 2.916 | 0.336 |
| `real_eval_v2/eval_tail.bin` (the held-out tail) | 2 MB | 2.827 | 2.661 | 2.929 | 0.268 |
| | 6.3 MB | 2.623 | 2.573 | 2.693 | 0.120 |

The inherited "unigram 5.170 / 5-gram 2.572" in `anchors.rs:3-7` and
`AGENTS.md §2.6` is **not a reading of the training corpus at any window I
measured** (2.655-2.849 at 2 MB, 2.580-2.916 at 6.3 MB, unigrams 4.99-5.11) and
it is not a `web` reading either (2.434-2.596). It remains unattributed. Do not
cite 2.572 as a bar.

### 2e. The math domain, 5 positions in the shard sequence, both windows

`mix/math` sorts its two `.jsonl` files before its 13,708 `.txt` shards, so
**position 0 of the shard sequence is the container** — which is why the
inherited 2.446 is a container reading.

| position (of 13,710) | file | 2 MB: 5-gram / 8-gram | 6.3 MB: 5-gram / 8-gram |
|---|---|---|---|
| 0 | `glm51_Math.jsonl` (JSON container) | **2.446** / 3.009 | 2.560 / 3.054 |
| 3,427 | `math_03425.txt` | 2.822 / 3.466 | 2.563 / 3.007 |
| 6,855 | `math_06853.txt` | 2.978 / 3.669 | 2.648 / 3.091 |
| 10,282 | `math_10280.txt` | **3.066** / 3.742 | 2.657 / 3.120 |
| 13,709 | `math_13707.txt` | 2.821 / 3.447 | 2.640 / 3.157 |
| | **mean / spread** | **2.827 / 0.620** | **2.614 / 0.097** |
| | **mean / spread, `.txt` only** | **2.922 / 0.245** | 2.613 / 0.095 |
| carve | `glm51_Math.jsonl` +2 MB / +4 MB | 2.599 / 2.626 | — |

**The defensible math bar is 2.61 ± 0.05 at a one-shard window** (spread 0.097,
the second-tightest in the corpus), and 2.92 ± 0.12 for the `.txt` shards alone
at the inherited 2 MB window. **2.416 is below all 14 of the math readings**; the
lowest of them is the JSON container's own 2.446 at the 2 MB window, and that
container reading is not math.

## 3. The table, with error bars — every corpus on the drive

Five positions per domain: index 0, n/4, n/2, 3n/4, n−1 of the trainer's own
sorted file list (`collect_files`). **`bar? = no` means spread > 0.05 BPB.**
Windows are 2,000,000 B (the inherited protocol) and 6,300,000 B (one whole
source shard). `qa` is measured at 8,000,000 B per shard (4 readable shards;
see §5), so its row is not strictly comparable to the other nine in §3a.

### 3a. 2 MB window — the inherited protocol

| domain | on disk | 5-gram mean | min | max | **spread** | 8-gram mean | 8-gram spread | bar? |
|---|---|---|---|---|---|---|---|---|
| **agentic** | 21 GB | 1.308 | 0.555 | 1.929 | **1.374** | 1.184 | 1.651 | **no** |
| ruweb | 19 GB | 2.134 | 1.951 | 2.248 | **0.297** | 1.975 | 0.456 | **no** |
| rudialog | 6.6 GB | 2.158 | 1.875 | 2.530 | **0.655** | 2.030 | 1.185 | **no** |
| **web** (was missing) | 103 GB | 2.558 | 2.537 | 2.596 | **0.059** | 3.033 | 0.080 | **no** |
| reasoning | 2.4 GB | 2.564 | 2.148 | 2.717 | **0.569** | 3.049 | 0.815 | **no** |
| code | 90 GB | 2.666 | 2.542 | 2.791 | **0.249** | 3.164 | 0.286 | **no** |
| wiki | 14 GB | 2.705 | 2.531 | 3.024 | **0.493** | 3.271 | 0.628 | **no** |
| math | 91 GB | 2.827 | 2.446 | 3.066 | **0.620** | 3.467 | 0.733 | **no** |
| books | 2.1 GB | 2.482 | 2.434 | 3.095 | **0.650** | 2.924 | 0.858 | **no** |
| **qa** (post-fix, **8 MB windows** — §5) | 2.4 GB | **1.798** | 1.729 | 1.878 | **0.149** | 1.959 | 0.297 | **no** |

`books`' five "positions" are a `--bytes` sweep on the one 2.15 GB file
(2.482 / 2.434 / 3.095 / 2.557 / 2.445 at 2 / 4 / 6.3 / 10 / 16 MB) —
non-monotonic, because the window walks documents and the fit/scored boundary
falls mid-document.

### 3b. 6.3 MB window — one whole source shard

| domain | 5-gram mean | min | max | **spread** | unigram mean | unseen ctx % | 8-gram mean | bar? |
|---|---|---|---|---|---|---|---|---|
| **agentic** | 1.292 | 0.555 | 1.835 | **1.280** | 4.823 | 1.5 | 1.096 | **no** |
| rudialog | 2.063 | 1.843 | 2.337 | **0.494** | 4.457 | 3.8 | 1.869 | **no** |
| ruweb | 2.140 | 2.054 | 2.254 | **0.200** | 4.143 | 2.3 | 1.930 | **no** |
| reasoning | 2.406 | 2.113 | 2.566 | **0.453** | 4.978 | 10.7 | 2.793 | **no** |
| **web** | 2.481 | 2.434 | 2.507 | **0.073** | 4.614 | 7.2 | 2.741 | **no** |
| code | 2.520 | 2.473 | 2.626 | **0.153** | 4.717 | 9.7 | 2.829 | **no** |
| wiki | 2.527 | 2.495 | 2.600 | **0.105** | 4.781 | 8.5 | 2.814 | **no** |
| math | 2.614 | 2.560 | 2.657 | **0.097** | 5.056 | 13.9 | 3.086 | **no** |

Size effect (6.3 MB minus 2 MB, 5-gram means): `math` −0.213, `wiki` −0.178,
`reasoning` −0.158, `code` −0.146, `rudialog` −0.096, `web` −0.077,
`ruweb` +0.006, `agentic` −0.016. **1.5 MB of fit text is too small a sample for
every domain except the two whose bar does not move with fit size at all**
(`ruweb` +0.006, `agentic` −0.016).

### 3c. The ranking, stated separately, and only where the spread supports it

The inherited table's ordering was `agentic < ruweb < books < rudialog < wiki <
code < reasoning < qa < math`. Re-measured at 5 positions and one-shard windows,
with disjointness as the test:

| comparison | verdict |
|---|---|
| **agentic is the most redundant domain, by a wide margin** | **SURVIVES.** agentic mean 1.292 vs 2.063-2.614 for every other domain; its **max (1.835) is below the min of the next domain (rudialog 1.843)**. The only disjoint pair in the whole table. |
| ruweb vs rudialog (ruweb more redundant) | **DEAD.** ruweb 2.140 [2.054, 2.254] vs rudialog 2.063 [1.843, 2.337] — ranges overlap on both ends and rudialog's mean is *lower*. Not separable. |
| web < wiki < code < reasoning (the inherited fine order) | **DEAD.** All four are 2.406-2.527 with mutually overlapping ranges at 6.3 MB (web [2.434, 2.507], wiki [2.495, 2.600], code [2.473, 2.626], reasoning [2.113, 2.566]). One band. |
| math is the hardest text domain | **SURVIVES, barely.** math's min (2.560) is above web's max (2.507) and code's min (2.473); its mean is 0.09-0.55 above every other domain measured at this protocol. |
| qa is the hardest domain | **DEAD, and inverted** — see §5. |
| genomics has a single anchor | **DEAD** — see §4. |
| books has an anchor at all | **DEAD as an absolute bar.** One 2.15 GB file, five windows, spread 0.650. It is usable as a *domain* and not as a number. |

So the honest statement of the ranking is two sentences long: **agentic is a
class of its own; the seven domains measured at the one-shard protocol are one
undifferentiated band between 1.84 and 2.66, and their internal order is not
measurable with the instrument we have** (add `qa` at 1.73-1.88 and `books` at
2.43-3.10 and the band is 1.73-3.10). A mixture priced by per-domain BPB
differences inside that band is priced by noise.

## 4. Genomics — the contradiction, settled

Two agents measured this today and disagreed: 2.030 / 1.991 (shipped) versus
0.190 (a shard front). **Both are real readings of 2 MB windows. Neither is an
anchor, because the corpus is not one thing.** 30 windows over 6 shards × 5
offsets:

| window | unigram | 5-gram | 8-gram | unseen |
|---|---|---|---|---|
| `corpus_000` +0 MB (Drosophila chrX head: `attatattat…` satellite) | 0.521 | **0.190** | 0.190 | 0.0% |
| `corpus_000` +10 MB (**197,531 `N` of 200,000 bytes**) | 0.096 | **0.095** | 0.095 | 0.0% |
| `corpus_005` +45 MB (satellite) | 0.397 | 0.165 | 0.164 | 0.0% |
| `corpus_015` +0 MB / +10 MB (satellite / `N`) | 0.423 / 0.096 | 0.170 / 0.095 | 0.170 / 0.095 | 0.0% |
| `corpus_004` +45 MB (mixed) | 1.541 | 0.343 | 0.342 | 0.0% |
| **the other 24 windows, 6 shards** | 2.38-6.65 | **2.094-2.558** (mean 2.193) | 1.974-4.012 | 0.0-2.3% |
| `corpus_012` (the clean shard: 0.12% `N`, 84% human) at all 5 offsets | 2.38-2.99 | **2.095-2.196** | 2.090-2.191 | 0.0-0.2% |
| whole `ecoli` genome (4.7 MB, uniformly coding) | 2.071 | 2.018 | 2.007 | 0.0% |
| whole `arabidopsis` genome | 2.654 | 2.129 | 2.144 | 0.1% |
| `eval_species` *Plasmodium* / *Staphylococcus* | 2.745 / 1.994 | 1.959 / 1.977 | 1.917 / 1.964 | 0.2% / 0.0% |
| `eval_chrom` human / mouse held-out records | **0.096** | **0.095** | 0.095 | 0.0% |

**The verdict: the corpus has two modes and one of them is not genome.**

- **The high mode is 2.09-2.56 BPB (mean 2.193)** and it is where 24 of the 30
  windows sit. The shipped 2.030 / 1.991 are 0.06-0.10 *below* the low edge of
  that mode — same family, single window, and slightly optimistic. They price
  the hard half, as `MANIFEST.md §1b(iii)` already said.
- **The low mode is not "low-complexity region". It is assembly filler.** Every
  low-mode window has a unigram ≤ 0.55, and inspecting the bytes: the front of
  `corpus_000` is `attatattat…` satellite, and the +10 MB window is **98.8% the
  letter `N`**. A full scan of all 16 shards in 64 KB blocks: **111,869,952 B =
  7.32% of the corpus is single-symbol blocks, and every one of them is `N`** —
  FASTA `N`-masks for unresolved and unplaced sequence. Per shard the share
  runs **0.00% (`corpus_001`, `_008`, `_010`) to 26.11% (`corpus_000`)** and
  24.44% (`corpus_015`) — and those two are exactly the shards whose early
  windows read 0.17-0.19 (satellite at the very front, `N` by +10 MB). The
  three shards with **zero** `N` are the three whose every measured window
  reads 2.13-2.56.
- **So the byte-weighted genomics counter bar is ≈ 2.04, not 0.19.**
  0.9268 × 2.193 + 0.0732 × 0.15. Note that the *window-count* frequency of the
  low mode (6 of 30 = 20%) badly overstates its *byte* share (7.3%): a 2 MB
  window is a coarse unit, one long `N` run dominates it, and that is why the
  byte scan was needed rather than more windows.
  `MANIFEST.md §6`'s "the global number is just counting the genomics share,
  whose counter bar is 0.19" is **wrong by 1.85 BPB**, and the conclusion built
  on it (arm C looks best globally because of genomics) does not follow.
- **A genomics arm has three numbers, not one:** the in-slice real-sequence mode
  **2.19**, the held-out-species axis **1.96-1.98**, and the `N`-filler tail
  **0.095**. Report all three. And note that the `eval_chrom` axis — the one
  that is supposed to measure *unseen sequence in a seen species* — reads
  **0.095 on both files**, i.e. **the held-out human/mouse chromosomes are
  `N`-masked too**: that eval axis is currently measuring filler, not
  generalization. It needs rebuilding before it can price anything.
- **The actionable finding: 7.32% of the genomics corpus is `N`.** Dropping
  runs of `N` at shard-build time is a filter over bytes we already have, it
  removes the low mode entirely, and it leaves a tight 2.15-2.22 domain bar.
  Right now a 12.76% genomics share (arm C) spends 0.93% of the whole run
  learning to predict the letter `N`.

### 4b. Records vs bytes: "uniform in proportion" is true and false at once

`SPLIT.md` says the 16 shards are "a uniform random sample of the whole slice:
each of the 16 files contains all 12 training organisms in proportion". The
sharder is `FNV(doc) % 16`, which is uniform over **records**. Measured, by
parsing every `>` header in all 16 shards:

| | min | max | skew |
|---|---|---|---|
| **records per shard** | 247 (`corpus_012`) | 317 (`corpus_013`) | **1.28×** — uniform |
| **bytes per shard** | 28,969,923 (`corpus_010`) | 164,205,345 (`corpus_008`) | **5.67×** |
| **human share of a shard** | **1.3%** (`corpus_009`) | **84.0%** (`corpus_012`) | **65×** |
| mouse, per-shard byte share | 0.0% | 84.3% | — |
| drosophila, per-shard byte share | 0.3% | 54.6% | — |
| arabidopsis / C. elegans, per-shard byte share | 0.0% | 29.1% / 61.9% | — |

Corpus totals: human 32.15% of bytes / 14.83% of records; zebrafish 16.34% /
42.05%; drosophila 9.53% / 40.89%. **Confirmed exactly as claimed.** A trainer
whose weight is bytes is sampling a vertebrate-weighted slice with a 5.7×
per-shard byte spread, and the arms can only take all 16 shards or none — which
is the right call, because a subset is not a sample. But "all 16" is a
*vertebrate-heavy, N-padded* 1.529 GB, and the mixture should be described that
way rather than as "uniform".

## 5. `qa` — the fix landed; the old row was measuring something else

`f6ab353` (2026-09-28) is in the binary used here, and it works: the reader now
recurses into parquet structs, so `mix/qa` returns the full
**8,000,000 B** it is asked for instead of 261,977 B.

| | unigram | 5-gram | 8-gram | unseen | bytes read |
|---|---|---|---|---|---|
| **before the fix** (2026-09-27 row, in this file) | 3.526 | 3.245 | — | 26.2% | 261,977 |
| **after, 4 readable shards, 8 MB each** | 5.662 / 5.626 / 5.612 / 5.677 | **1.830 / 1.754 / 1.878 / 1.729** | 2.028 / 1.884 / 2.111 / 1.814 | 8.3-10.4% | 8,000,000 |
| | | **mean 1.798, spread 0.149** | mean 1.959 | | |

The 3.245 was `id` (a UUID) plus question stems. The real passages live in
`document.html` inside a struct, and a UUID column has a *high* 5-gram BPB
because it is incompressible noise — which is why the fix moved the number
**down by 1.4 BPB, not up**. `qa` is the **second-most counter-solvable text
domain we own** (1.73-1.88, against 1.84-2.66 for the other seven at the
6.3 MB protocol, and only `agentic` is lower), not the least, and the conclusion
"over-weight `qa`, it is the least redundant" **rests on a number that does not
describe `qa` and now describes nothing**. Two further facts about the corpus,
both new:

- **12 of 287 shards are present (4.2%), and one of the 12 present is
  corrupt**: `train-00014-of-00287.parquet` does not end in `PAR1`
  (`…83e51096`), so the reader refuses it loudly
  (`data: skipping unreadable shard`, then a loud panic when it is the only
  file). There is no `train-00012` at all. 2.535 GB on disk is 2.346 GB usable.
- The parquet stores the same passage **twice** (`document.html` and
  `document.tokens.token`), so the counter sees each passage twice. That
  duplication is in the file and the trainer will train on it; it is part of
  why 1.798 is as low as it is. Recorded, not corrected — picking fields by
  name to de-duplicate is a heuristic, and `f6ab353` says so.

## 6. Which claims from the first version of this file survive

| # | claim (2026-09-27) | verdict |
|---|---|---|
| 1 | "books is where the language lives" | **DEAD** (already retracted in this file; still dead) |
| 2 | "`books` 4.602 / 2.482" as a bar | **DEAD as a bar** — 2.434-3.095 over five windows, spread 0.650 |
| 3 | "**agentic** is nearly information-free per byte; 0.938 BPB, 0.0% unseen; 21 GB can be learned to ~0.94" | **SURVIVES in direction, DIES in number.** 1.292 mean at one-shard windows, 0.555-1.835 across 5 positions; unseen 0.0-3.0%. The 0.938 is **one** shard (`agentic_00000.txt`); the other two `.txt` positions read 1.835 and the three `.jsonl` positions 0.555 / 1.533 / 1.603 — the spread is *inside* the domain, not between its two file types. The domain is still the only one separated from everything else. **Judge the arm by whether the model goes well below ~1.3, not 0.94.** |
| 4 | "**`qa` is the least counter-solvable domain** (3.245, 26.2% unseen); over-weight it" | **DEAD AND INVERTED** — 1.798 [1.729, 1.878], 8.3-10.4% unseen. The 3.245 measured UUIDs (§5) |
| 5 | "the text domains are all broadly similar (2.25-2.66 at 5-gram); no single 'language lives here' domain" | **SURVIVES, now quantified**: the seven prose/code/math domains span **1.843-2.657** at one-shard windows, with per-domain spreads 0.07-0.49. This is the most useful sentence in the file. |
| 6 | "**ruweb** is the most redundant text (4.4% unseen, unigram 4.227)" | **DEAD.** ruweb 2.140 [2.054, 2.254] vs rudialog 2.063 [1.843, 2.337] — overlapping, rudialog's mean lower. Not separable. |
| 7 | "a 4-gram on the same data beats our 7.5M model by ~4.9 BPB" | **SURVIVES in direction, DIES in magnitude.** Every counter reading here is far below the best held-out number (4.997), so "the counter wins" stands; but the quoted 4.9 BPB gap was computed against a bar whose window was unstated. The gap is **2.3-2.6 BPB** for the canary corpus at the declared protocols (§2a). |
| 8 | "**Genomics** is cheap in bytes: 2.010 at 5-gram, 0% unseen, 1-3 GB buys the capability" | **DEAD as stated.** Two modes; 7.32% of the bytes are `N` filler; the real-sequence mode is 2.09-2.56 with 24 of 30 windows in 2.09-2.22 (§4) |
| 9 | "the mixture should allocate by information above the counter's ability" | **SURVIVES as a rule, and is now the only rule in the file that is not a number.** Its *inputs* (§3a) are single-domain readings that must be re-derived at 5 windows before any arm is priced. The g·q pricing in `mixture/MANIFEST.md` used one window per domain; `agentic`, `rudialog`, `qa` and `books` move the most under re-measurement. |
| 10 | "every arm is judged on its OWN domain's ladder; `anchors --fit` for the matching bar" | **SURVIVES and is now mandatory** — §2a is what "the matching bar" costs: 4 protocols, 2.397-2.679, and the fit/scored sizes both have to be written down. |
| 11 | `web` missing from the table | **FIXED.** web = 2.481 [2.434, 2.507] at one-shard windows, 103 GB, unseen 7.2% |
| 12 | the "5-gram web bar 2.572" quoted in `anchors.rs` / `MANIFEST.md §1c` | **UNATTRIBUTED.** web's own 5-gram range is 2.434-2.596 over 10 windows; 2.572 is inside it but no reading on this machine produces it for `web`. The `anchors.rs` header number is an in-corpus split of a 2 MB read of the *training* corpus, whose 5-position range is 2.655-2.849. Do not cite 2.572 as a web bar. |

## 7. The mixture principle, unchanged, and the two changes this forces

Allocate bytes by **information above the counter's ability** — that survives,
because it is the only claim here that is not a reading. Two concrete changes:

1. **`agentic` stays at a format slice** (the one disjoint comparison in the
   corpus), but the number to beat is **~1.3 BPB, not 0.94**, and the domain's
   own spread (0.555-1.835) is *inside* it rather than between its `.txt` and
   `.jsonl` halves — so do not price the two halves differently.
2. **Every per-domain price in `mixture/arms.json` and `MANIFEST.md` §2 is a
   single-window reading and must be re-derived** before an arm is run. The
   domains that move most are `agentic` (0.938 → 1.308), `qa` (3.245 → 1.798),
   `books` (2.482 → 2.434-3.095) and `genomics` (0.19 → 2.04 blended). The
   `g·q` *rule* is untouched; its table is void.
3. **`genomics` should be re-sharded without the `N` runs** before it is priced
   or evaluated (§4), and `eval_chrom` must be rebuilt — it currently measures
   filler.

## 8. Reproduction

    # 344 windows, one instrument, HEAD 82993d8, 2026-09-29
    CARGO_TARGET_DIR=/home/sehaxe/.cache/anchors-target \
      cargo build --release -p dormouse-data --bin anchors
    # jobs: 5 positions/domain x {2 MB, 6.3 MB} x {order 5, order 8}
    #   + books --bytes sweep (5 sizes x 2 orders)
    #   + qa 4 readable shards @ 8 MB x 2 orders (1 corrupt shard, refused)
    #   + genomics 6 shards x 5 offsets x 2 orders, 4 header-flag controls,
    #     2 whole genomes x 2 sizes, eval_species + eval_chrom (2 files each)
    #   + canary 2x2 (fit 2/16 MB x scored 20,480/102,400 B), 5 fit positions,
    #     5 scored windows
    #   + 3 domains x 2 intra-shard carves (+2 MB, +4 MB)
    # raw: ~/logs/anchor-remeasure-2026-09-29.tsv   (344 rows, one per window)
    #      ~/logs/anchor-remeasure-raw-2026-09-29.log (full tool stdout per run)
    #      ~/logs/anchor-remeasure-summary-2026-09-29.txt

Position inside a corpus is reached by pointing the tool at a different shard
or by `dd`-carving a window out first (`dd if=F of=W bs=1048576
iflag=skip_bytes,count_bytes skip=OFF count=N`); the tool has no `--offset`.
`books` and `qa` are parquet and cannot be carved, so their positions are a
`--bytes` sweep and separate shards respectively. One of qa's 13 shards is
corrupt (§5) and is excluded, loudly.
