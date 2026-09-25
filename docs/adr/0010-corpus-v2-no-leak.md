# Corpus v2: the eval tail never trains

The first real corpus was built wrong: the 500 MB eval tail was carved from the
raw corpus, and the DCLM filter then consumed the whole raw corpus — eval
documents sat inside the training data, so every held-out BPB was optimistic by
an unmeasured amount. Found during the 2026-09-25 audit; fixed in
`filter.rs` (commit 8fda851) with a two-region mode: documents are routed by
their START offset — head docs train, tail docs evaluate, any document
straddling the boundary is dropped whole, and the dedup database is shared, so
a tail doc duplicating a head doc is dropped from eval (no contamination in
either direction). Old behavior is the default when `--eval-output` is absent.

Outputs: `real_filtered_v2/corpus.bin` (20.4 GB), `real_eval_v2/eval_tail.bin`
(247 MB), and `/home/sehaxe/eval_2m_v2/eval_tail.bin` — the 2 MiB cadence-eval
slice (fast feedback every 500 steps; the 500 MB tail is for milestone numbers
only). The v1 directories stay on disk untouched until the first v2 checkpoint
supersedes them. All BPB numbers from pretrain v1/v21 and earlier are
contaminated or worse and are not comparable to v2 baselines.

Rule going forward: eval regions are excluded from training data at filter
time, structurally — never by convention, never by trusting a split done
upstream of the filter.
