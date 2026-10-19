# Data: what the corpora are, where they live, how to get one

The trainer eats one thing: a flat `.bin` of raw UTF-8 bytes (`dormouse train
--data corpus.bin`), optionally with a held-out `.bin` (`--eval`). Everything
else — documents, parquet, shard directories — is a way of producing that file.

This page exists because the audit's harshest finding was reproducibility: the
corpora this project's numbers were taken on live **outside the repository** (an
external drive), so "clone → train → measure" is not true out of the box. It can
be made true; here is what is involved.

## The two corpora the numbers were taken on

| name | size | what | how it was built |
|---|---|---|---|
| **filtered web** (`real_filtered/corpus.bin`) | 19.7 GB | DCLM-classifier ≥ 0.5 + exact dedup, English-dominant web text; 272M documents → 104.2M kept | the Rust lane's `dormouse-data` pipeline (`rust-legacy` branch, `crates/dormouse-data`) |
| **the mixture** (`mix/{web,wiki,ruweb,math}`) | ~230 GB of `.txt` shards | four shard directories read in a seeded shuffle (`configs/mix_big.toml`), for the staged mixture | `aria_data/plan_txt_streamer.md` describes the download+shard layout |
| **the eval carve** (`real_eval/eval_tail.bin`) | 500 MB | a tail carve of the same crawl, **removed** from the training corpus (no leak) | a `dd`-style carve; the trainer never sees it in the stream |

The eval window a BPB is quoted over is `eval_batches × batch × seq_len` bytes and
it is printed on every eval line — a BPB without its window is not a number.

## If you have no corpus: the smallest honest recipe

1. **Pick a source you can download.** `HuggingFaceFW/fineweb-2` (ODC-By) and
   `codeparrot/codeparrot-clean` (MIT) are both non-gated; `scripts/fetch_mix_sources.py`
   in this repo already pulls slices of three sources with `duckdb` range reads
   (it exists because one `fineweb-2` shard is 4.8 GB and a plain download is
   impossible).
2. **Concatenate to one `.bin`.** The trainer wants a byte stream, not documents:
   any concatenation order works, but shuffle files first — consecutive documents
   from one crawl are one topic, and the burn lane measured a loss spike
   (1.46 → 8.69) at a category boundary in a run that read shards in download
   order.
3. **Carve an eval file.** Take a tail slice (~500 MB) **out of** the training
   file, not a copy of it: `dd if=corpus.bin of=eval_tail.bin bs=1M skip=<n>`
   then truncate the training file, or keep them as separate downloads and never
   concatenate the eval slice into the training stream.
4. **Train.** `uv run dormouse train medium --data corpus.bin --eval eval_tail.bin
   --device cuda --steps 60000 --batch 24 --set seq_len=1024 --set engram_ram=true
   --set moe_capacity=0 --set moe_kernel=triton --set ckpt_name=first --set out_dir=checkpoints/first`
5. **Measure.** `uv run dormouse bench --ckpt-name first --out-dir checkpoints/first
   --tasks arc_easy,hellaswag --limit 1000`.

Expect different numbers from the ones in the README: those were taken on the
filtered corpus at a stated window and config. The *shape* of the curve
(monotone down, no NaN, the depth curve) is what transfers.

## Shard directories instead of one `.bin`

`configs/mix_big.toml` is a *mix plan*: it names four sources, gives each a weight
per training stage (S1 → S2 → S3 decay), and caps epochs per source (`max_epochs`,
because MoE degrades past ~4 re-reads). A source is a directory of `.txt` shards;
`ShardStream` reads them in a seeded shuffled order, one file open at a time, and
the mix decides the source per step. Absolute paths in that file are one machine's
drive UUID — point them at your own copy.

## What the repository does NOT ship

- No corpus (the smallest one is 19.7 GB).
- No `data_local/` (gitignored: `*.bin`, the mix slices, the SFT corpora).
- No checkpoints (gitignored; `checkpoints/<name>/<name>.pt` plus its
  `.config.toml` snapshot and, for `engram_ram` runs, the `.ngram` sidecar —
  all three travel together or the load refuses).

The one exception is the demo's artifact: `docs-site/public/demo/model.dormouse`
(16.3 MB) is committed so the live page works, with the reasoning in
`docs-site/public/demo/README.md`.
