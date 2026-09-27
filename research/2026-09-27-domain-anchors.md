# 2026-09-27 — Domain anchors, and what they say the mixture should be

The data for a universal model was already on disk and already
domain-separated; we were training on a single filtered web mix. Measured
anchors per domain with our own tool
(`cargo run -p dormouse-data --bin anchors <dir> --bytes 2000000 --order 5`;
1.5 MB train / 0.5 MB held out inside each 2 MB read), against
`/mnt/e43497ab-.../aria_data/pretrain/mix/`:

| domain | on disk | uniform | unigram | 5-gram+backoff | 5-gram contexts unseen in train |
|--------|---------|---------|---------|----------------|-------------------------------|
| **agentic** | 21 GB | 8.000 | 4.789 | **0.938** | 0.0% |
| **code** | 90 GB | 8.000 | 4.741 | 2.643 | 13.6% |
| **math** | 91 GB | 8.000 | 4.966 | 2.446 | 17.1% |
| **books** | 2.1 GB | 8.000 | 7.006 | **6.759** | 94.0% |
| web text (our eval slice) | 19 GB | 8.000 | 5.398 | 2.911 | 23.4% |
| **genomics** (2 genomes so far) | growing | 8.000 | 2.034 | 2.010 | 0.0% |

## What the numbers say (this is the mixture design, measured not guessed)

1. **The agentic corpus is nearly information-free per byte.** A 5-gram counter
   gets 0.938 BPB on it - the data is templated dialogue and tool traces with
   massively repeated structure (0% unseen 5-gram contexts). 21 GB of it can be
   "learned" to ~0.94 by memorization alone. It is worth including ONLY if the
   model is going to go well below 0.94 (i.e. learn the actual semantics of tool
   use), and even then a fraction of the bytes will do. This is the same failure
   mode as our Engram arm, in the data.
2. **The high-entropy prose is where language lives, and it is the smallest
   thing on disk.** books: 6.759 at 5-gram with 94% of contexts unseen - real
   natural text, nearly incompressible at order 5 - and only 2.1 GB of it. Web
   text sits at 2.911. Every mechanism that raises capacity (a memory table, a
   bigger model) pays off on THIS data, not on the templated parts.
3. **code and math sit in the middle** (2.64 / 2.45 at 5-gram, 14-17% unseen):
   genuinely learnable, with real headroom above the counter. They deserve large
   slices on the owner's priorities, and unlike agentic they are not
   counter-solvable.
4. **Genomics is cheap in bytes**: 2.010 at 5-gram with 0% unseen means the
   domain is fully learnable at low order, so a small slice (1-3 GB) buys the
   capability. The headroom is in long-range structure (motifs, regulatory
   grammar) - which is why the KDA ladder, not the byte count, is the genomics
   project.

## The resulting mixture principle

Allocate bytes by **information above the counter's ability**, not by how much
data we happen to have or how much we want a domain:

- big slice: high-entropy prose (web, wiki, books) - this is the language
- meaningful slices: code, math (real headroom, owner's stated priorities)
- small slice: genomics (cheap to learn, high value, long-range character)
- small slice: agentic (near-zero marginal information until the model can beat
  0.94; revisit after the SFT floor exists, where tool use is taught through
  execution instead - see the post-training verdict)

Every arm is judged on its OWN domain's anchor ladder, not on a single global
BPB: a model can be "good at code" and "uniform at books" and the global number
will hide it. The trainer's eval takes one directory, so per-domain evals are a
small change (one eval dir per domain, same fixed-window protocol).
