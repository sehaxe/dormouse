# mix4 — the 5.8 GB pretraining mixture

Built 2026-10-22 by `scripts/build_mix4.py` (seed `20261022`). The plan is
`configs/mix_v4.toml`; the per-domain byte counts, filters, drop counters
and eval proof are in `data_local/mix4/manifest.json`.

This is a **byte-level** corpus. Every domain is a concatenation of whole units
(web shards, wiki articles, code files, math sections, dialogue shards); no
unit is silently truncated. A unit that does not pass a filter is dropped whole
and counted.

## Composition

`train` is the bytes the mixer reads; `eval` is held out and never appears in a
training shard. Percentages are of the train total.

| domain | target | train bytes | actual | delta | source |
|---|---:|---:|---:|---:|---|
| web | 38% | 2,350,670,235 | 40.76% | +2.76 | `mix/web` (general English web) |
| wiki_en | 8% | 466,160,653 | 8.08% | +0.08 | `mix/wiki` (English Wikipedia) |
| wiki_ru | 4% | 230,028,784 | 3.99% | −0.01 | `wikimedia/wikipedia` 20231101.ru |
| ruweb | 13% | 866,439,984 | 15.02% | +2.02 | `mix/ruweb` |
| code | 12% | 690,009,645 | 11.96% | −0.04 | `codeparrot/github-code-clean` |
| math | 10% | 632,508,389 | 10.97% | +0.97 | `mix/math` |
| books | 3% of the 5% pair | 57,503,324 | 1.00% | −2.00 | `mix/books` (see the gap) |
| qa | 2% of the 5% pair | 57,505,003 | 1.00% | −1.00 | `mix/qa` Natural Questions |
| reasoning | 3% | 176,568,813 | 3.06% | +0.06 | `mix/reasoning` |
| instruct | 5% | 120,500,114 | 2.09% | −2.91 | five parsed SFT pair files |
| rudialog | 2% | 119,606,932 | 2.07% | +0.07 | `mix/rudialog` |

Train total **5,767,501,876 B**, eval **52,933,865 B**, on disk
**5,820,435,741 B** — inside the 5.5–6.0 GB budget. Every deviation is at or
inside the allowed ±3 points; the two real shortfalls (books+QA and
instructions) are measured, not hidden.

The 5.77 GB train shards are ~1.124 M steps at batch 10 × seq 512 (5 130 B/step). Every shard
is read about once; `max_epochs = 4.0` still guards a longer run.

## Sources and filters

### English web — 2,350,670,235 B

**The task's stated source was measured and rejected.** `corpus_bin_filtered.bin`
is 19.7 GB, but 200 random 4 KiB windows held LaTeX in 180, math keywords in
110 and code in 60; only one window had Cyrillic. It is a math/code corpus, not
general English web. Putting 40% of a chat model's diet there would duplicate
the math domain and teach almost no everyday prose.

The web share comes from the mount's `mix/web` instead: 17,382 × 6.3 MB shards,
109.5 GB of general English crawl. Filters are shard-level and deliberately
light (the mount has no DCLM classifier): ≥ 100 KB, Cyrillic ≤ 2% of letters,
ASCII letters ≥ 80% of letters, `http` occurrences ≤ 0.2% of bytes, uppercase
≤ 40% of letters. All sampled shards pass. The builder takes whole shards in a
seeded order: 373 train shards + 1 eval shard (`web_04108.txt`, 6.29 MB), drop
count 0. This is raw CommonCrawl with a language/link guard, not DCLM-filtered
text; it is the best general-English source on this box.

### Wikipedia EN — 466,160,653 B

`mix/wiki` holds 2,299 × 6.3 MB English-only shards (14.48 GB); samples across
the directory have Cyrillic ratio 0.0. The builder takes whole shards in a
seeded order, 74 train shards. Filter: shard ≥ 100 KB (all pass). Drop count: 0.

### Wikipedia RU — 230,028,784 B

The local wiki directory has no Russian; that is a real gap in the mount. RU
wiki was fetched from `wikimedia/wikipedia`, config `20231101.ru`, shard
`train/2.parquet` (249,899,379 B compressed). Licence: CC BY-SA 4.0 + GFDL.
Rendered as `# title\n\n<plain text>`. Kept 30,492 docs (29,943 train + 549
eval). Filters: 300 B–200 KB, Cyrillic ratio ≥ 0.15, no `#redirect`, SHA-1
dedup. Drops: `size` 782, `non_russian` 5.

### Russian web — 866,439,984 B

`mix/ruweb` (3,057 × 6.3 MB shards, 19.33 GB). Whole shards in a seeded order,
137 train shards. Shard-level filters: ≥ 100 KB, Cyrillic ratio ≥ 0.50,
`http` occurrences ≤ 0.1% of bytes, uppercase ≤ 35% of letters. All selected
shards passed (drop count 0). This is the noisiest prose source in the mix
(SEO text, old news); it is kept because it is language-valid and the Russian
share is a product requirement, not because it is clean.

### Real code (TASK A) — 690,009,645 B

`bigcode/starcoderdata` is **gated**: an anonymous parquet HEAD/range request
returns HTTP 401, and the dataset card requires login. The fallback is
`codeparrot/github-code-clean` (Apache-2.0 dataset; permissively licensed
files), configs `all-mit` (10 shards) + `all-apache-2.0` (2 shards), 2.04 GB
compressed. The 12 shards are cached under `data_local/mix4_src/` and listed
with sizes in the manifest; the download is resumable and the builder refuses a
truncated parquet.

Target languages: Python, JavaScript, TypeScript, C, Go, Rust, SQL. Filters
(byte counts are from the manifest):

| reason | files | bytes |
|---|---:|---:|
| non-target language | 589,792 | 3,470,474,660 |
| > 100 KB | 4,679 | 1,034,069,811 |
| over the per-language quota (first pass) | 107,028 | 555,777,209 |
| vendored / `*.min.*` path | 17,680 | 133,430,553 |
| generated header | 6,956 | 50,219,393 |
| < 200 B | 22,053 | 2,525,571 |
| alnum < 20% | 146 | 559,810 |
| minified lines | 1 | 18,719 |
| < 3 lines | 3 | 4,323 |

Code filters: 200 B–100 KB, ≥ 3 lines, ≥ 20% alphanumeric bytes, no
`vendor/`, `third_party/`, `node_modules/`, `dist/`, `build/`, `generated/`,
`autogen/`, no `*.min.js`/`*.min.css`, average line ≤ 200 / max line ≤ 2,000,
no “generated by / do not edit / auto-generated” header, SHA-1 dedup. Quotas:
Python 190 MB, JavaScript 150, TypeScript 130, C 100, Go 60, SQL 40, Rust 20.
Rust/SQL are short in this dataset; the final fill pass takes the remaining
permissive files (over quota) until 690 MB, so the realised mix is somewhat
more Python/JS/TS than the quota table.

### Math (TASK B) — 632,508,389 B

`mix/math` is 13,708 `*.txt` shards (86.43 GB) plus two large synthetic JSONL
dumps. The JSONL dumps are not used: the task is LaTeX/prose, and the `.txt`
shards already supply the budget. A shard is split into documents on
`\n---\n` (the separator the generator uses); a document is kept whole or
dropped whole.

Keep rules: 400 B–200 KB; at least one equation construct (`\begin{`,
`\frac`, `\boxed`, `\[`, `\(`, `$$`, inline `$…$`); ≥ 120 prose words; link
density ≤ max(2, words/150); uppercase ≤ 25% of letters; < 3 `>>` markers;
quote lines ≤ max(3, lines/20); link-list lines ≤ max(5, lines/5).

Kept 47,986 train docs + 353 eval docs. Dropped 136,410 docs / 504 MB:

| reason | docs | bytes |
|---|---:|---:|
| too short | 61,720 | 13,651,479 |
| link list | 23,344 | 354,351,809 |
| no equation | 28,750 | 40,071,454 |
| too few words | 17,817 | 11,619,547 |
| forum quote | 3,443 | 67,355,340 |
| link density | 1,198 | 9,514,171 |
| too long | 25 | 6,174,592 |
| uppercase | 112 | 1,342,256 |
| duplicate | 1 | 2,168 |

The pass rate is 26% of documents (56% of bytes): the link-list and forum
filters remove exactly the material the task named. Dropped documents are not
replaced from the synthetic JSONL dumps.

### Books — 57,503,324 B (a measured substitution)

**The local `books/` parquet is not books.** Its schema is FineWeb/DCLM-like
(`id`, `dump`, `url`, `text`, `language`, `token_count`, `score`, `int_score`);
the sampled rows are CommonCrawl news/blog pages from `CC-MAIN-2021-04`
(`honeybeehaven.org`, `agiliway.com`, `fxcm.com`). Treating it as literature
would be a label lie. It is used only as a **1% curated-web supplement**:
`int_score ≥ 3`, `language = en`, `token_count ≥ 200`, 500 B–100 KB, ≥ 40%
alphanumeric, SHA-1 dedup. Kept 11,678 train docs + 868 eval docs. Drops:
`short_tokens` 1,285, `size` 16. The books share was moved to QA/web; the
combined Books+QA target is 5%, the delivered pair is 2%.

### QA — 57,505,003 B

`mix/qa` is Natural Questions (13 parquet shards; `train-00014-of-00287.parquet`
is corrupt — footer magic missing — and is named in the manifest, not read).
Each record is `Question: …\nArticle: <document html stripped to text>\nAnswer:
<short answer or long-answer token span>`. Filters: answer present, article
500 B–200 KB. Kept 1,482 train docs + 112 eval docs; drops `no_answer` 1,651,
`article_too_long` 13. `<style>`/`<script>` bodies are removed before tag
stripping (they were CSS text in the first build).

Genuine Q→A-only rendering is 13.4 MB (6,435 answers), not 57.5 MB. The
article context is what reaches 1%; it is real NQ data, but some NQ articles
also occur in Wikipedia and therefore some overlap with `wiki_en` is possible.

### Reasoning — 176,568,813 B

`mix/reasoning` (394 × 6.3 MB, 2.48 GB) holds `<|begin_of_thought|>` traces.
Whole shards, 28 train + 1 eval. Filter: ≥ 20 thought markers and ≥ 100 KB.
Drops: 3 shards / 18.99 MB (`no_thought_traces`).

### Instructions — 120,500,114 B

Five parsed pair files: `aria_ru` 200,000, `aria_en` 200,000, `aria_reason`
895, `distill_clean2` 47,502, `more_short` 25,165 = 473,562 lines. Rendered
with the SFT template `0x01 instruction \n 0x02 answer 0x03` (the same bytes as
`dormouse.train.sft.render`; the builder duplicates the three lines to avoid a
torch import and the test pins the literals). After sentinel rejection and
SHA-1 dedup: 455,800 unique pairs. 17,762 duplicates; 0 sentinel hits. Seeded
shuffle; 446,511 train records (120.5 MB) + 7,777 eval records (2.1 MB).

This is 2.1% of the mixture, not 5%: **126.9 MB of rendered pairs is all the
clean pair data that exists on this box**, and 123.0 MB of that is unique. The
shortfall is reallocated to web/ruweb, not filled by repeating pairs.

### Russian dialogues — 119,606,932 B

`mix/rudialog` (1,124 × 6.3 MB, 7.07 GB). Whole shards, 19 train + 1 eval.
Filter: ≥ 100 KB, Cyrillic ratio ≥ 0.40; 0 drops. The source is synthetic
Russian chat (system prompts + answers) and is kept at the requested 2%.

## The eval carve

Every eval shard is carved **before** its training shard. Whole-shard domains
(web, wiki_en, ruweb, reasoning, rudialog) exclude the eval shard by name and
prove its SHA-1 absent from the train shard set; parquet/doc domains exclude
each eval document by SHA-1. The
builder computes the eval-unit SHA-1 set and the train-unit SHA-1 set while
writing; `Domain.finish()` asserts the sets are disjoint and refuses to publish
on an intersection. The manifest records the method, both counts, `overlap: 0`,
`pass: true`, and the eval-unit SHA-1s; each eval file also carries its SHA-256.

| domain | eval bytes | units | carve |
|---|---:|---:|---|
| web | 6,294,307 | 1 | whole shard `web_04108.txt` |
| wiki_en | 6,291,978 | 1 | whole shard `wiki_01999.txt` |
| wiki_ru | 4,201,531 | 549 | every-Nth parquet row, filtered |
| ruweb | 6,323,619 | 1 | whole shard `ruweb_01929.txt` |
| code | 2,407,810 | 419 | every-Nth parquet row, filtered |
| math | 4,287,871 | 353 | filtered docs at the 25%/75% shard positions |
| books | 4,202,214 | 868 | every-Nth parquet row, filtered |
| qa | 4,240,026 | 112 | every-Nth NQ row, rendered |
| reasoning | 6,294,100 | 1 | whole shard `reasoning_00330.txt` |
| instruct | 2,097,380 | 7,777 | first 2 MiB of the seeded pair shuffle |
| rudialog | 6,293,029 | 1 | whole shard `rudialog_00407.txt` |

All eval files are between 2.1 and 6.3 MB, inside the 2–10 MB request.

## Honest gaps

1. **The stated web source is not English web.** `corpus_bin_filtered.bin`
   sampled as 90% LaTeX, 55% math keywords, 30% code. It was replaced by
   `mix/web` (raw CommonCrawl + language/link guards). The replacement is
   general English, but it is not DCLM-classified: its quality is below the
   filtered corpus the plan assumed. This is the largest quality trade in the
   build.
2. **StarCoderData is gated.** `bigcode/starcoderdata` returns HTTP 401 to an
   anonymous parquet request. The code slice is `codeparrot/github-code-clean`
   (Apache-2.0 dataset, permissively licensed files), 12 shards / 2.04 GB
   compressed. It is real code, but it is older/dedup-filtered GitHub code, not
   The Stack v2.
3. **Instructions are 2.1%, not 5%.** 455,800 unique clean pairs = 123.0 MB
   rendered. No amount of mixing creates more pair data.
4. **Books+QA is 2%, not 5%.** The local `books/` parquet is CommonCrawl, not
   books; NQ Q→A-only is 13.4 MB. Books is used as 1% curated web, QA as 1%
   article-context QA. The other 3 points went to web/ruweb/math.
5. **RU Wikipedia was not on the mount.** It was fetched from
   `wikimedia/wikipedia` `20231101.ru` shard 2 (249.9 MB compressed). Without
   network, the RU half of the Wikipedia share would have been 0.
6. **The `code/` directory (90 GB) is not used.** It was verified mislabeled
   spam/web-novels by the earlier survey; the builder never reads it.
7. **RU web is noisy.** It passes the language/link/uppercase filters, but it
   is crawl text, not edited prose. It is 15% because the Russian chat target
   needs breadth; a cleaner Russian source would be the first upgrade.
8. **Code language balance is quota-driven, not measured.** Rust and SQL are
   short in this dataset; the fill pass takes Python/JS/TS/C/Go over quota. The
   manifest records the quota drops, not a per-language byte histogram.
9. **One QA parquet is corrupt** (`train-00014-of-00287.parquet`) and is named
   in the manifest rather than partially read.

## Rebuild

```bash
cd /home/sehaxe/dormouse
uv run --with pyarrow scripts/build_mix4.py              # all domains, resume
uv run --with pyarrow scripts/build_mix4.py --only code  # one domain
uv run --with pyarrow scripts/build_mix4.py --force       # rebuild everything
```

A finished domain is skipped when its build key (source fingerprints + target +
seed + filter version) still matches. An interrupted domain resumes from
`data_local/mix4/.<domain>.state.json`; the partial file is truncated to the
recorded byte cursor before appending, so a crash cannot duplicate a unit.

Verify the plan:

```bash
cd port
uv run python -c "from dormouse.train.mixture import load_mix_plan; p=load_mix_plan('configs/mix_v4.toml'); print(p)"
```

Test the filters, the eval carve, the no-leak proof and idempotence:

```bash
cd port
CUDA_VISIBLE_DEVICES="" uv run pytest tests/test_build_mix4.py -q
```

`manifest.json` is the source of truth for the measured counts; this document
is the human-readable explanation of the same build.
