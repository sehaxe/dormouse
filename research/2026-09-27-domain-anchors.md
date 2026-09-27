# 2026-09-27 — Domain anchors, and what they say the mixture should be

The data for a universal model was already on disk and already
domain-separated (`mix/`); we were training on a single filtered web mix.
Every domain needs its own bar ladder before any model result in it means
anything. Measured with our own tool,
`cargo run --release -p dormouse-data --bin anchors <dir> --bytes 2000000 --order 5`
(1.5 MB train / 0.5 MB held out inside each 2 MB read), against
`/mnt/e43497ab-.../aria_data/pretrain/mix/`:

| domain | on disk | uniform | unigram | 5-gram+backoff | 5-gram contexts unseen in train |
|--------|---------|---------|---------|----------------|-------------------------------|
| **agentic** | 21 GB | 8.000 | 4.789 | **0.938** | 0.0% |
| code | 90 GB | 8.000 | 4.741 | 2.643 | 13.6% |
| math | 91 GB | 8.000 | 4.966 | 2.446 | 17.1% |
| reasoning | 2.4 GB | 8.000 | 4.907 | 2.664 | 16.8% |
| rudialog | 6.6 GB | 8.000 | 4.928 | 2.530 | 11.5% |
| ruweb | 19 GB | 8.000 | 4.227 | 2.248 | 4.4% |
| wiki | 14 GB | 8.000 | 4.574 | 2.592 | 12.7% |
| books | 2.1 GB | 8.000 | 4.602 | 2.482 | 10.1% |
| **qa** | 2.4 GB | 8.000 | 3.526 | **3.245** | **26.2%** |
| **genomics** (2 genomes so far) | growing | 8.000 | 2.034 | 2.010 | 0.0% |

**Correction, first version of this file was wrong.** `books` and `qa` are
`.parquet`, and the anchors tool read files RAW — so the first measurement of
those two domains was of the thrift/snappy container, not the text: books
"7.006 / 6.759", qa unreadable. Two bugs came out of chasing it, both fixed:
`ByteStream`'s parquet branch returned the full batch length without copying
into the read buffer, so **reading any parquet corpus panicked or ingested
garbage** (the trainer included — books/qa were simply never trainable), and
the anchors tool now decodes through the trainer's own reader
(`dormouse_data::read_bytes`), so the bar and the model see the same bytes.
The corrected books row is 4.602 / 2.482 and the "books is the highest-entropy
domain, language lives there" claim was an artifact. It is dead.

## What the corrected numbers say

1. **The agentic corpus is nearly information-free per byte.** A 5-gram counter
   gets 0.938 BPB on it — templated dialogue and tool traces with 0.0% unseen
   5-gram contexts, i.e. literally memorizable at order 5. 21 GB of it can be
   "learned" to ~0.94 without understanding anything. Include a slice only to
   teach output FORMAT, and judge it by whether the model goes well below 0.94;
   the rest of the bytes buy nothing. This is the same failure mode as our
   Engram arm, but in the data.
2. **`qa` is the least counter-solvable domain** (3.245 at 5-gram, 26.2% of
   contexts unseen) — human-written, least templated. It is also small (2.4 GB).
   The honest reading: the text domains are all broadly similar (2.25-2.66 at
   5-gram), so there is no single "language lives here" domain; the mixture
   does not need a clever weighting so much as it needs to EXCLUDE the one
   degenerate source.
3. **ruweb is the most redundant text** (4.4% unseen, lowest unigram 4.227) —
   the 19 GB we have been training on is the most counter-solvable prose we
   own. That is consistent with the critic's finding that a 4-gram on the same
   data beats our 7.5M model by ~4.9 BPB.
4. **Genomics is cheap in bytes**: 2.010 at 5-gram with 0% unseen means the
   domain is fully learnable at low order, so a 1-3 GB slice buys the
   capability. The headroom is long-range (motifs, regulatory grammar), so the
   KDA ladder — not the byte count — is the genomics project.

## The resulting mixture principle

Allocate bytes by **information above the counter's ability**, not by how much
data exists or how much a domain is wanted:

- agentic: a small format-teaching slice, revisited after the SFT floor (where
  tool use is taught by execution instead — see the post-training verdict)
- code, math: large slices (real headroom, owner's stated priorities, and 90+91
  GB available)
- text prose (ruweb, wiki, books, qa, reasoning, rudialog): the bulk; qa is
  over-weighted relative to its size because it is the least redundant
- genomics: a small slice, high value per byte

Every arm is judged on its OWN domain's ladder, not a single global BPB: a
model can be good at code and uniform at qa and the global number hides both.
Per-domain evals are a small change (one eval dir per domain, same fixed-window
protocol, `anchors --fit` for the matching bar).

## The tool now needed for that: `--fit`

`anchors <eval-dir> --fit <train-dir>` fits the counters on one corpus and
scores another, which is the only way to get the bar for the exact bytes the
trainer's eval reads (its eval is the FIRST 100 KB of the eval dir, while an
internal `--holdout` split scores the TRAILING quarter — the two numbers were
never comparable, and two docs quoted different numbers for the same tool).
