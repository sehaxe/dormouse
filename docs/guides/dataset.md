# Dataset: what the trainer actually reads

Three corpora live under `data_local/` (gitignored). Every number below is from a
build run on 2026-10-22/23 and is re-derivable with `port/scripts/build_mix4.py`.

## `mix4/` — the pretraining corpus (5.77 GB train, 52.9 MB held out)

Eleven domains, one `.bin` each plus `<domain>_eval.bin`:

| domain | train bytes | share | source |
|---|---:|---:|---|
| web | 2 350 670 235 | 40.8% | mount `mix/web`, light shard filters (no DCLM classifier) |
| ruweb | 866 439 984 | 15.0% | mount `mix/ruweb` |
| code | 690 009 645 | 12.0% | `codeparrot/github-code-clean` (`all-mit` + `all-apache-2.0`), 7 languages |
| math | 632 508 389 | 11.0% | local `*.txt` shards, split on `\n---\n`, equation + ≥120 prose words |
| wiki_en | 466 160 653 | 8.1% | mount `mix/wiki` |
| wiki_ru | 230 028 784 | 4.0% | `wikimedia/wikipedia` `20231101.ru` |
| reasoning | 176 568 813 | 3.1% | CoT traces with `\boxed{}` / `<|begin_of_thought|>` |
| instruct | 120 500 114 | 2.1% | local QA pairs, deduplicated |
| rudialog | 119 606 932 | 2.1% | `rudialog` |
| books | 57 503 324 | 1.0% | mount `books/` — **Common Crawl pages, not books** (named deviation) |
| qa | 57 505 003 | 1.0% | NQ-style article context + answer |

**Eval is carved before training and proved disjoint.** Whole-shard domains
exclude the eval shard and assert its SHA-1 absent from the train unit set;
parquet/doc domains select eval rows first and skip them by SHA-1.
`Domain.finish()` refuses to publish if the two SHA-1 sets intersect, and
`manifest.json` records `method: doc_sha1_disjoint`, `eval_docs`, `train_docs`,
`overlap: 0`.

Three deviations worth knowing before you quote a number:

1. `data_local/corpus_bin_filtered.bin` (19.7 GB) is **math/code, not web** —
   180/200 sampled windows carry LaTeX, 110/200 math keywords, 60/200 code,
   1/200 Cyrillic. It must not be used as the "web" share.
2. The local mount `code/` (90 GB) is **not code** (spam, web novels, release
   notes) and is unused.
3. `books`/`qa` are 2% of the mix, not the 5% the original plan asked for,
   because the local corpus for them does not exist.

## `mix5/` — the near-dup + quality-band rerun (3.1 GB)

Same builder with `--near-dup --quality-band`. **The multiplier is measured, not
assumed:** the mount shards are already deduplicated (max off-diagonal MinHash
Jaccard inside a 6.3 MB shard = 0.07), so near-dup removes 0.063% of web bytes
and 0.016% of ruweb. The quality band removes 6.38% (web) / 3.27% (ruweb).
Honest combined gain: **×1.00–1.07**, not the ×3–10 the literature promises for
raw Common Crawl.

## Document boundaries (`doc_masking`)

`doc_masking=true` makes one document per training row (whole document, or a
contiguous chunk when it is longer than `seq_len`) and excludes padding and
cross-document targets from the loss. It reads a sidecar written beside the data
file:

```
<data>.bin            the byte stream
<data>.bin.docs.json  {"version": 1, "size": <bytes>, "starts": [0, 6295756, ...]}
```

`open_streams` refuses loudly when the sidecar is missing. The boundary set is
produced by `scripts/build_doc_sidecar.py` from the source records, never
guessed: parquet/jsonl domains replay the exact rows the builder wrote
(`--domain code|math|wiki_ru|instruct`), whole-file shards are matched by head,
tail and full content (`--source-dir`), and a flat concatenation with no record
token is REFUSED (`--refuse --reason ...`). `scripts/docmask_audit.py` prints
the per-source verdict (REAL / BOGUS / MISSING / REFUSED-BY-BUILDER); a BOGUS
sidecar cuts at file seams, not document seams.

Current state of `mix_v7`: **9/12 sources REAL, 23.17 GB = 96.8%** — fwedu,
cosmo, ru, math, finemath, science, code, wiki_ru, instruct. `wiki_en`,
`reasoning` and `rudialog` (762 MB, 3.2%) are REFUSED-BY-BUILDER: each mix4
shard is a flat concatenation of many articles/traces/dialogues with no record
token, and the 6.3 MB file seams are the BOGUS sidecars this gate deletes.
`configs/mix_v7_docmask.toml` is mix_v7 with those three sources removed and
the stage weights renormalized.

Eval keeps the fixed `eval_bytes` window (the recorded, comparable number). If a
sidecar sits beside the eval file, the eval masks the SAME cross-document
targets train does (`eval_docmask=sidecar`); otherwise it stays flat and the eval
line says `eval_docmask=flat (no sidecar)` rather than inventing boundaries.

`doc_masking` requires the dropless route (`moe_capacity=0`): the capacity-padded
dispatch picks which rows an expert keeps from the whole batch, so a padded row
could take capacity from a real token — the mask would hold on the loss and not
on the forward. The config refuses the pair.

## Choosing a stream in a run

```bash
# single domain, long context, document masking
dormouse train medium --set data=data_local/mix4/code.bin \
  --set doc_masking=true --set moe_capacity=0.0 \
  --set seq_len=8192 --set max_seq_len=8192

# multi-domain, document masking (only sources with a REAL sidecar)
dormouse train flagship3_dense --set mix_plan=port/configs/mix_v7_docmask.toml \
  --set doc_masking=true --eval data_local/mix4/web_eval.bin
```

`mix_plan` and `data` are mutually exclusive: the plan owns the streams and the
named `--data` would be inert, so the CLI refuses the pair.

Stages in `mix_v4.toml` (fractions of the **optimizer** step budget, after the
`grad_accum` fix): S1 0–42% prose, S2 42–80% adds code/math/reasoning, S3 80–100%
decay with code 26%, math 32%, instruct 11%. The epoch cap refuses loudly when a
source would be read past `max_epochs`.

## SFT corpus v2 — frontier distillations under an open license

`scripts/build_sft_v2.py` turns open distillation datasets into one corpus in our
pair format. The owner's rule was "newest teachers, and a license you can defend",
which is why the source list is short and each row carries its provenance.

| source | teacher | license | rows kept | risk |
|---|---|---|---:|---|
| `atmike/ox-alpha-glm-5.3-flash-…-17k-raw` | GLM-5.3-Flash | MIT | 5 119 | low |
| `j0no12/qwen38-kimi-fable` → `fable-5-trace` | Fable-5 | CC-BY-4.0 | 769 | low |
| `j0no12/qwen38-kimi-fable` → `kimi-k3-trace` | Kimi-K3 | CC-BY-4.0 | 6 | low |
| `harshbhx/ratchet-toolcall-train` | Nemotron-3-Ultra (open) | Apache-2.0 | 2 465 | low |
| `open-r1/OpenR1-Math-220k` | DeepSeek-R1 | Apache-2.0 | 3 915 | low |
| `FlaneAI/GPT-6.1-Sol-…` | GPT-6.1 Sol | GPL-2.0 | 55 | **unclear** |
| `j0no12/…` → `qwen3.8-max` | Qwen3.8-Max | other (Alibaba) | 1 878 | **unclear** |

Two corpora are written: `sft_pairs/v2/full/` (all seven sources, 14 207 rows) and
`sft_pairs/v2/clean/` (the five low-risk sources, 12 274 rows), so a quarantined
source can be dropped without rebuilding. Each source is its own file.

Filters, applied in order, each counted at the first rule that fires — prompt
≤ 400 B, response ≤ 500 B (a 103M model has a 1024-byte window; a 3 000-token trace
is unusable), English or Russian only, no control bytes, no degenerate answers,
exact dedup, near-dup (5-byte-shingle MinHash, J ≥ 0.9, within a source), and a
50-byte eval-leak gate against `data_local/mix4/web_eval.bin`. What the length caps
cost is visible in the dashboard: `code_glm` keeps 5 119 of 17 158 and `code_fimi`
6 of 1 200 — long code and long traces simply do not fit.

The 300-row holdout is chosen by the smallest hash of the **rendered** bytes, so it
is disjoint from train by construction, not by bookkeeping. The corpus (JSONL) is
gitignored like `aria_*.jsonl`; `manifest.json` is committed and carries the
per-source dashboard, the drop counts and the file hashes.
