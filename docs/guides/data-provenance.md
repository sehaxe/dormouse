# Data provenance: the six sources a commercial release would inherit

**Scope.** The 2026-11-08 refetch wrote six files and one held-out window
(`docs/guides/dataset.md:3-27`, `docs/records/reviews/2026-11-08-night.md:14-44`). This
document is the ledger for those files: what each one is, what its licence says,
what a commercial release still lacks, and the smallest change that would make
the record automatic. It is **not legal advice**; §6 names the questions a
lawyer must answer, and §5 separates what is measured from what I infer.

**Checked 2026-11-08** (box clock). Every external claim carries the URL, the
date checked, and the dataset revision the API returned that day, so the citation
can be re-checked even when the "today" is not.

**How to reproduce a line of this table.** `sha256sum data_local/{best,mix}/*.bin`;
`curl -s https://huggingface.co/api/datasets/<id>` for the revision and the
declared licence; `curl -sIL` the parquet URL for the shard's resolved size.
Nothing here is remembered from a card that was read once.

---

## 1. What was fetched, and what it produced (VERIFIED)

| key | dataset (HF id) | revision @ checked (first 8 hex of the HF repo sha) | shard actually fetched | licence as declared | produced file | bytes | weight |
|---|---|---|---|---|---|---|---|
| `fwedu` | `HuggingFaceFW/fineweb-edu` | `87f09149` · 2025-07-11 | `refs/convert/parquet/default/train/0000.parquet` (2 369 456 837 B) | ODC-By 1.0 + CommonCrawl ToU | `data_local/best/fwedu_train.bin` | 933 743 230 | 0.55 |
| `cosmo` | `HuggingFaceTB/cosmopedia` | `0ae6ec63` · 2024-08-12 | `refs/convert/parquet/auto_math_text/train/0000.parquet` (248 133 641 B) | **Apache-2.0** (card YAML only) | `data_local/best/cosmo.bin` | 302 198 823 | 0.12 |
| `finemath` | `HuggingFaceTB/finemath` | `e92b25a6` · 2025-02-06 | `refs/convert/parquet/finemath-3plus/train/0000.parquet` (507 607 173 B) | ODC-By 1.0 + CommonCrawl ToU | `data_local/best/finemath.bin` | 948 125 886 | 0.06 |
| `code` | `codeparrot/codeparrot-clean` | `35a59fb0` · 2022-10-10 | `refs/convert/parquet/default/partial-train/0000.parquet` (187 416 754 B) | **none declared** | `data_local/mix/code.bin` | 209 714 775 | 0.11 |
| `math` | `open-web-math/open-web-math` | `fde8ef8f` · 2023-10-17 | `refs/convert/parquet/default/train/0000.parquet` (236 268 485 B) | ODC-By 1.0 + CommonCrawl ToU | `data_local/mix/math.bin` | 209 701 817 | 0.06 |
| `ru` | `HuggingFaceFW/fineweb-2` (`rus_Cyrl`) | `af9c1333` · 2025-10-27 | `refs/convert/parquet/rus_Cyrl/train/0000.parquet` (4 841 277 045 B) | ODC-By 1.0 + CommonCrawl ToU + PII form | `data_local/mix/ru.bin` | 209 710 316 | 0.10 |

Weights: `port/configs/mix_refetch.toml:29-35`, one stage, sum 1.000, `seed =
20261027`, `max_epochs = 4.0` (`:15-16`). Sources: `:19-24`. Sizes: `ls -l`,
2026-11-08. `sha256` of every file, 2026-11-08:

```
63b5fa59…c3db3  data_local/best/cosmo.bin
a5d6846e…309f2  data_local/best/finemath.bin
77e96a37…ff4ad  data_local/best/fwedu_train.bin
003ba82d…661f   data_local/best/fwedu_eval.bin
4464f6ca…d0d41  data_local/mix/code.bin
73293e36…cb44   data_local/mix/math.bin
9bd7743b…75a9   data_local/mix/ru.bin
```

Each licence cell above is the card's own sentence, quoted in §2.2 with the card
URL, checked 2026-11-08; §6 lists every URL and revision.

**The seventh file** is the held-out window, `data_local/best/fwedu_eval.bin`
(8 386 155 B, sha256 `003ba82d…661f`). No mixture source reads it
(`port/configs/mix_refetch.toml:11-13`).

### 1.1 What the bytes are, row by row (measured, not inferred)

I re-read the three parquet shards the refetch kept (`data_local/parquet/`) and
the converted `code` shard, and compared them to the `.bin` files:

| file | relationship to the shard (0-based row indices) | rows taken / rows in shard |
|---|---|---|
| `fwedu_train.bin` | row-order **prefix**: row 0 is byte-identical; the first 191 894 rows reproduce 933 740 461 B and the file holds 2 769 more | 191 894 + a 2 769 B fragment / 785 906 |
| `fwedu_eval.bin` | the **next whole rows** from the same shard: rows 191 895…193 582 are exactly 8 386 155 B | 1 688 rows |
| `cosmo.bin` | the **whole** shard: 108 328 rows × (text + 7-byte separator) = 302 198 823 B exactly | 108 328 / 108 328 |
| `finemath.bin` | the **whole** shard: 167 232 rows × (text + 7-byte separator) = 948 125 886 B exactly | 167 232 / 167 232 |
| `code.bin` | row-order **prefix**: first 21 252 rows = 209 714 775 B exactly; row 0 byte-identical | 21 252 / 49 985 |
| `math.bin`, `ru.bin` | a 209 715 200 B cap (`--cap-mb 200`, `docs/records/reviews/2026-11-08-night.md:27`), row count **not recorded** | unrecorded |

Two consequences. (a) The three `best/` files are **not** all whole shards:
`cosmo` and `finemath` are, and `fwedu` is a prefix whose stop was decided by
the script's `LIMIT cap/6500 rows` heuristic (`port/scripts/refetch_pretrain.py:79`)
— which of the two limits bound is not recorded, and the `--cap-mb` that produced
`fwedu` is in no file. (b) The carve is **not** what `port/configs/mix_refetch.toml:11-13` says:
the eval *starts* on a row boundary, but the train file *ends* mid-document
(2 769 B into row 191 894), so one document is bisected and its 2 446-byte tail
is in neither file. No held-out row is trained on — the eval begins at the next
row — but the split rule is inferred from the bytes, not recorded anywhere.

### 1.2 Word count versus byte count

The mixture weights are **draw probabilities, not byte budgets**: the source of
step *k* is a pure function of `(seed, k, steps)` (`port/src/dormouse/train/mixture.py:21-22`,
`source_for_step` at `:374`), so a source's share of the *data* is its weight and
its share of the *local bytes* is whatever those bytes happen to be. At the
ladder budget (14 648 micro-batches × 8 × 1024) the plan served 120.1 MB:
fwedu 66.3 (55.2 %), cosmo 13.9 (11.5 %), code 13.3 (11.1 %), ru 11.9 (9.9 %),
finemath 7.5 (6.2 %), math 7.2 (6.0 %) — each within 0.5 pp of its weight
(`docs/records/reviews/2026-11-08-night.md:35-39`, measured with
`mixture.replay_bytes`). Locally, finemath is 33.7 % of the bytes for a 6 %
weight and fwedu 33.2 % for 55 %.

**Arithmetic, not a measurement:** with `max_epochs = 4.0` the 4-epoch cap binds
on fwedu first, at `4 × 933 743 230 / 0.55` ≈ **6.32 GiB** of served bytes
(code: 7.10 GiB, ru: 7.81 GiB, cosmo: 9.38 GiB). Past that, a run either refuses or
repeats fwedu — the plan says which, nobody has run it that far.

---

## 2. What each licence actually permits

### 2.1 The three licences in play

**ODC-By 1.0** (`https://opendatacommons.org/licenses/by/1-0/`, checked
2026-11-08). The operative sentences, quoted:

- §3.1: "These rights explicitly include commercial use, and do not exclude any
  field of endeavour." → commercial training is permitted.
- §4.2: if You *Publicly Convey* the Database or a Derivative Database, You must
  do so under this licence, include a copy/URI of it, and keep the notices
  intact. → redistributing the `.bin` itself carries obligations.
- §4.3: "if you Publicly Use a Produced Work, You must include a notice …
  aware that Content was obtained from the Database … and that it is available
  under this License", with the example: *"Contains information from DATABASE
  NAME which is made available under the ODC Attribution License."*
- Preamble: the licence "only governs the rights over the Database, and not the
  contents of the Database individually". → **it grants nothing in the articles
  themselves.** That is the largest hole in the whole stack.

**Apache-2.0** (`https://www.apache.org/licenses/LICENSE-2.0.txt`, checked
2026-11-08). §3 patent grant, §4 redistribution permitted provided notices are
kept, §6 no trademark grant. It is a software licence applied to a text corpus;
whether it was *intended* to cover the generated text is a lawyer question, not
a reading of the text.

**Common Crawl ToU** (`https://commoncrawl.org/terms-of-use`, last updated
2024-03-07, checked 2026-11-08). This sits **on top of** four of the six cards,
because they say so in their own licensing sections. §1 grants a "limited,
non-assignable, non-transferable, non-sublicensable, non-exclusive" licence —
i.e. **redistributing the crawled text is not clearly granted even where ODC-By
would allow it**. §2: the Crawled Content "may be subject to separate terms of
use … from the owners". §3: "CC strongly recommends that you obtain the advice of
legal counsel before making any use, including commercial use".

### 2.2 The matrix

| source | train commercially | redistribute the **data** | redistribute the **weights** | attribution required | evidence |
|---|---|---|---|---|---|
| `fwedu` | **YES** | **UNCLEAR** — ODC-By §4.2 allows it, CC ToU §1 is non-transferable | **UNCLEAR** — ODC-By never names model weights | **YES** — ODC-By §4.3 notice | card §Licensing, `…/fineweb-edu/raw/main/README.md:630-632` |
| `cosmo` | **YES** — no field-of-use restriction (Apache-2.0 §2) | **YES** with licence + notices | **UNCLEAR** — Apache-2.0 on *generated* text, whose seeds include ODC-By AutoMathText | **YES** — keep notices | card YAML `license: apache-2.0`, `…/cosmopedia/raw/main/README.md:196`; generation: `:217`, seeds: `:265` |
| `finemath` | **YES** | **UNCLEAR** — same CC ToU layer | **UNCLEAR** | **YES** — ODC-By §4.3 | card §Licensing, `…/finemath/raw/main/README.md:278-280` |
| `code` | **UNCLEAR — no licence is granted** | **UNCLEAR** | **UNCLEAR — 40 % of the slice is copyleft** | per-file, **not recorded** | API + card declare nothing (`…/codeparrot-clean/raw/main/README.md`, 1 299 B, no licence section); measured licence mix below |
| `math` | **YES** | **UNCLEAR** — same CC ToU layer | **UNCLEAR** | **YES** — ODC-By §4.3 | card §License, `…/open-web-math/raw/main/README.md:95-97` |
| `ru` | **YES** | **UNCLEAR** — same CC ToU layer | **UNCLEAR** | **YES** — ODC-By §4.3; PII removal form | card §Licensing `…/fineweb-2/raw/main/README.md:13476`, PII/opt-out `:13541-13547` |

**Why "weights" is UNCLEAR everywhere.** ODC-By defines a *Produced Work* as "a
work (such as an image, audiovisual material, text, or sounds) resulting from
using the whole or a Substantial part of the Contents". Model weights are not in
that list, and §4.3 attaches a notice to a Produced Work, not to a model. Whether
a trained checkpoint is a Produced Work, a Derivative Database, or neither is a
legal characterisation this repository cannot make. **What a lawyer must check:**
(1) does ODC-By reach the weights at all; (2) if yes, does §4.3's notice discharge
it, or does §4.2's share-the-database rule reach a model that memorised a
Substantial part; (3) does the ODC-By preamble leave the article copyrights
unlicensed, and if so what does that mean for a model that reproduces text.

**Why `cosmo` is UNCLEAR despite Apache-2.0.** The card declares the licence
only in its YAML front matter; the body says nothing about licensing
(`…/cosmopedia/raw/main/README.md:196` is the only occurrence of the word). The
text is synthetic, generated by Mixtral-8x7B-Instruct-v0.1 (`:217`), and its
`auto_math_text` seeds come from AutoMathText, which is ODC-By (`:265`). **What
a lawyer must check:** whether HuggingFaceTB intended Apache-2.0 to cover the
output text; whether a model's outputs inherit its seed data's licence; whether
Mixtral's own terms matter here — the model card is Apache-2.0 (`https://huggingface.co/api/models/mistralai/Mixtral-8x7B-Instruct-v0.1`, checked 2026-11-08, revision `eba92302`), with no output restriction stated on it.

### 2.3 `code` is the one that is not what the script says

`port/scripts/fetch_mix_sources.py:81` declares `"MIT (per the dataset card)"`.
There is no licence on that card. The HF API returns no `license` tag and no
`cardData.license` (`https://huggingface.co/api/datasets/codeparrot/codeparrot-clean`,
checked 2026-11-08, revision `35a59fb0`), and the card is 1 299 bytes with no
licensing section. What the dataset *does* have is a per-row `license` column
joined from `bigquery-public-data.github_repos.licenses`
(`…/transformersbook/codeparrot/raw/main/README.md:19-27`).

The slice we took is the first 21 252 rows of the converted shard, and I measured
their licences by re-downloading that shard (187 416 754 B, sha256
`6fd64051…1ed94`, 2026-11-08) and reproducing the `.bin` byte count exactly:

| licence | share of `code.bin` bytes |
|---|---:|
| apache-2.0 | 22.73 % |
| **gpl-3.0** | **18.26 %** |
| bsd-3-clause | 16.68 % |
| mit | 16.45 % |
| **gpl-2.0** | **10.70 %** |
| **agpl-3.0** | **8.34 %** |
| lgpl-3.0 | 1.80 % |
| bsd-2-clause | 1.48 % |
| mpl-2.0 | 1.48 % |
| lgpl-2.1 | 0.98 % |
| unlicense / epl-1.0 / artistic-2.0 / isc / cc0-1.0 | 1.10 % together |

**40.08 % of the code slice is GPL-2.0/3.0, AGPL-3.0 or LGPL.** That is a
measured fact about the bytes we would train on, and it is not in any script,
config or checkpoint. **SPECULATION (my inference, not a legal conclusion):** the
prevailing industry position is that training does not create a derivative work
of the training data, so copyleft code would not force the weights under
GPL/AGPL — but that position is contested and untested for AGPL's network clause,
which is the sharpest one for a hosted model. A lawyer must decide it; this
document only proves the licence mix is 40 % copyleft, not that it taints
anything.

**What a lawyer must check:** (1) whether the absence of a licence means no
rights are granted (default copyright) or that the upstream GitHub licences
govern per file; (2) whether the 40 % copyleft share creates an obligation for
the weights or for a hosted service; (3) whether the taken rows must be filtered
before a commercial release, and to what licence allow-list.

### 2.4 Common Crawl's opt-out is crawl-time, not downstream

Common Crawl honours `robots.txt` at crawl time and publishes an Opt-Out Ledger
for legal requests (`https://commoncrawl.org/blog/common-crawl-foundation-opt-out-registry`,
2025-09-17, checked 2026-11-08). `fineweb-2` adds its own removal form
(`…/fineweb-2/raw/main/README.md:13541-13547`). Neither of these removes text
from a `.bin` that is already on disk, and none of our six files can answer
"which document did this 1 KiB come from": the refetch writes **only** the text
column (`port/scripts/refetch_pretrain.py:80-90`,
`port/scripts/fetch_mix_sources.py:104-119`), dropping the `url` / `repo_name` /
`license` columns that the parquet still holds.

---

## 3. The gap list for a commercial release

Each line names the current state with a citation, then what a release needs.
None of these is implemented.

| # | gap | current state | what a release needs |
|---|---|---|---|
| G1 | **No per-file manifest.** Nothing on disk ties a `.bin` to a dataset, a revision, a URL or a licence. | the scripts print one line and exit (`refetch_pretrain.py:94`, `fetch_mix_sources.py:164`) | `<name>.manifest.json` beside every `.bin`: dataset id, revision sha, resolved URL, config/split/shard, licence + licence URL, bytes, sha256, fetch date |
| G2 | **The checkpoint records shape, not content.** `corpus.sources` is `{name: size}` and the digest covers the plan text + those sizes. The docstring states the ceiling itself. | `port/src/dormouse/train/loop.py:455-505` ("swapping a source for a different file of the same size is invisible"), `:617` | per-source `sha256` in the `corpus` block, from the G1 sidecars; `unmanifested: [...]` for sources without one |
| G3 | **The URL is a moving ref, not a content address.** Every fetch went through `refs/convert/parquet`, which the Hub may rebuild; the dataset revision is not recorded. | resolved URLs above (`refs/convert/parquet/...`), 2026-11-08 | the dataset revision sha **and** the converted shard's sha256 in the manifest; the local parquet kept for every source, not 3 of 6 (`data_local/parquet/`) |
| G4 | **Takedown cannot be mapped to bytes.** The `.bin` holds only the text column; for `code`/`math`/`ru` the source parquet was not kept. | `refetch_pretrain.py:80-90`, `fetch_mix_sources.py:104-119` | keep a row-index sidecar (`<name>.rows.json`: first_row, row_count + per-row source URL hash) or keep the parquet; at minimum record the row range so a URL list can be rebuilt |
| G5 | **Two licence strings in the scripts are wrong.** `cosmo` is Apache-2.0, not ODC-By; `code` is not MIT and is 40 % copyleft. | `refetch_pretrain.py:51` (`"ODC-By"` for cosmopedia), `fetch_mix_sources.py:81` (`"MIT (per the dataset card)"`) | fix both strings and make the manifest the single source of truth, so a card edit cannot drift from the code |
| G6 | **The eval carve rule is inferred, not recorded.** The head ends mid-document and 2 446 B are in neither file, while the config says "from a document separator". | `port/configs/mix_refetch.toml:11-13`; measured in §1.1 | record `{source, row_range, byte_range, sha256, rule}` for `fwedu_eval.bin`; either cut at a row boundary or say "byte cut" in the config |
| G7 | **No takedown/opt-out path.** No per-source URL index, no request ledger, no removal procedure. | none anywhere in the repo | a documented removal procedure: request → find rows by URL hash → rebuild the `.bin` → record the new sha256 → note the version in the model card |
| G8 | **No provenance entry in a model card, and no outbound licence for the weights.** The repo is MIT for the *code*; nothing states the weights' licence or carries the ODC-By §4.3 notice. | `LICENSE:1`; no model card exists | one "Data provenance" section: the six datasets + revisions + licences + the §4.3 sentence + the checkpoint's `corpus.sha256`; an explicit weights licence |
| G9 | **The checkpoints on disk carry no corpus block.** A byte scan finds no `corpus` / `plan_sha256` key in either file, so neither can name the bytes behind its step count. | `checkpoints/csa8k/csa8k.pt` (10:49), `checkpoints/csa8k_prop/csa8k_prop.pt` (12:52), 2026-11-08; scan is grep on the pickle, so treat as "not found", not "absent" | either re-save or mark them "corpus unprovable" in the model card; new runs get G2 |

**Not a gap, and worth saying:** the draw itself is reproducible. The source of
step *k* is a pure function of `(seed, k, steps)` and the plan
(`mixture.py:21-22`), the run seed is in the snapshot
(`checkpoints/csa8k/csa8k.config.toml:207`), and the snapshot names the plan
(`:214`). So **`(plan sha256, source sha256s, seed, step)` identifies the bytes** —
no per-step byte log is needed. That is why G1+G2 are the whole fix: without
content hashes the chain stops at a file name, and with them it closes.

---

## 4. Implementation sketch (sketch only — not implemented)

The keystone is the fetch script, because it is the last place that knows the
dataset identity. Everything downstream can then read a file instead of a
convention.

### 4.1 A sidecar, written where the URL is still known

`port/scripts/_provenance.py` (new, ~25 lines), imported by both fetch scripts:

```python
def write_manifest(bin_path: Path, *, dataset, revision, config, split, shard,
                   url, licence, licence_url, checked, rows=None) -> dict:
    b = bin_path.read_bytes()                      # 2.6 GB total: seconds
    m = {
        "file": bin_path.name, "bytes": len(b),
        "sha256": hashlib.sha256(b).hexdigest(),
        "dataset": dataset, "revision": revision,   # HF /api/datasets/<id> ["sha"]
        "config": config, "split": split, "shard": shard,
        "url": url,                                 # the RESOLVED convert URL
        "licence": licence, "licence_url": licence_url,
        "checked": checked, "rows": rows,           # rows = {"first": 0, "count": n}
    }
    Path(str(bin_path) + ".manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    return m
```

Call sites: one line at the end of `fetch()` in `refetch_pretrain.py:82-95` and
`fetch_mix_sources.py:104-122`, plus one for the eval carve. `revision` and
`licence` come from the same API response the script already trusts; the
`licence` string moves out of the hard-coded `Slice`/`Source` dataclass
(`refetch_pretrain.py:48-53`, `fetch_mix_sources.py:73-85`) into the manifest,
which is what fixes G5.

### 4.2 The mix plan needs no change

`load_mix_plan` (`mixture.py:157`) keeps resolving `[sources]` paths. The sidecar
sits next to the file, so the plan stays a plan. If a plan-level hash is wanted
anyway it is `sha256(concat(sorted(sidecar sha256s)))` — but that duplicates the
checkpoint's `corpus.sha256`, so it is two records of one fact. Skip it.

### 4.3 The checkpoint reads the sidecar (the only format change)

In `corpus_manifest` (`loop.py:455-505`), replace `sources[name] = source_size(...)`
with a stat that also looks for `<path>.manifest.json`:

```python
sources[name] = source_size(name, path)                     # unchanged: the epoch cap reads this
sha = manifest_sha256(path)                                 # None when the sidecar is absent
if sha: source_sha256[name] = sha
else:   unmanifested.append(name)                            # COUNTED, never silent (ADR-0019)
```

and fold `source_sha256` into the existing digest (`loop.py:493-500`). The
`sources` field keeps its meaning, so `tests/test_ckpt_format.py:135` still
passes; the new keys are additive, exactly as v2 was added to v1
(`loop.py:421-428`). The `.config.toml` snapshot does **not** get the hash: the
snapshot is a drift gate (what a run may not change), the checkpoint's `corpus`
is the proof of what it read. One fact, one record. The loader already prints when a corpus block is
unprovable (`loop.py:932-937`) — `unmanifested` uses the same channel.

### 4.4 The model card writes itself from the checkpoint

A `--print-provenance` on the existing ckpt reader emits the §4.3 notice and the
six rows from `corpus`. One section, generated, never hand-edited — a hand-typed
provenance table is the same defect as a hand-typed parameter count.

### 4.5 What not to build

No merkle tree, no per-step byte log, no provenance database, no hash of the
80 GB corpus at checkpoint time. The draw is already deterministic in
`(seed, step, steps)`; six hashes and a plan hash close the chain. Add the
per-row index (G4) only when a takedown actually has to be served, and add the
model-card generator only when a release exists.

---

## 5. VERIFIED versus SPECULATION

### VERIFIED (sourced)

1. Six sources, six `.bin` files, exact bytes and sha256 (§1; `ls -l`,
   `sha256sum`, 2026-11-08); weights from `port/configs/mix_refetch.toml:29-35`.
2. The dataset ids, revisions and declared licences in the table are what
   `https://huggingface.co/api/datasets/<id>` returned on 2026-11-08; each
   licence sentence is quoted from the card URL named beside it.
3. `cosmopedia` declares Apache-2.0 in YAML only; its body has no licensing
   statement (`…/cosmopedia/raw/main/README.md:196`).
4. `codeparrot-clean` declares no licence anywhere, and the 200 MiB slice is the
   first 21 252 rows of its converted shard, **40.08 % copyleft by bytes**
   (measured by re-downloading sha256 `6fd64051…1ed94` and reproducing the
   `.bin` size exactly).
5. ODC-By 1.0 permits commercial use (§3.1), requires a notice when conveying
   the database (§4.2) and a notice on a Produced Work (§4.3), and does not
   license the contents individually (preamble).
6. Common Crawl's ToU is a non-transferable licence (§1), points at third-party
   terms (§2) and tells users to get counsel before commercial use (§3).
7. The checkpoint's `corpus` block records the plan hash and per-source **sizes**
   only (`port/src/dormouse/train/loop.py:455-505`); neither checkpoint on disk
   carries the block (byte scan, §3 G9).
8. The eval window is rows 191 895…193 582 of the same shard, exactly
   8 386 155 B; the train file ends mid-document (§1.1).

### SPECULATION (my inference, marked as such)

1. **Whether ODC-By reaches model weights.** No source says. The Produced Work
   definition does not name weights. My reading — treat the notice as cheap
   insurance and put it in the model card — is prudence, not law.
2. **Whether 40 % copyleft in the training data obliges anything.** I assume the
   mainstream "training is not derivation" position holds and that AGPL's
   network clause is the one to watch. No benchmark or case exists in this repo;
   a lawyer decides.
3. **That Apache-2.0 was meant to cover Cosmopedia's text.** The YAML says so;
   the body does not. I assume the metadata is the intent.
4. **That the fetched shard is unchanged since the fetch.** The sizes and the
   `code` byte-for-byte reproduction say the converted shards are stable today;
   the fetch did not record a hash, so this is inference from a re-download.
5. **That 6.32 GiB is where the 4-epoch cap binds.** Pure arithmetic from the
   weights and sizes; no run has reached it.

**Confidence.** High (byte-level) in §1 and in every licence *quote*; medium in
the "what a release needs" list (it is engineering judgement); **low in every
legal conclusion** — that is the lawyer's call, and §2.2 names the exact
questions rather than answering them.

---

## 6. Sources

| what | URL | checked | revision / date |
|---|---|---|---|
| FineWeb-Edu card | `https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/raw/main/README.md` | 2026-11-08 | `87f09149` · 2025-07-11 |
| Cosmopedia card | `https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/raw/main/README.md` | 2026-11-08 | `0ae6ec63` · 2024-08-12 |
| FineMath card | `https://huggingface.co/datasets/HuggingFaceTB/finemath/raw/main/README.md` | 2026-11-08 | `e92b25a6` · 2025-02-06 |
| codeparrot-clean card | `https://huggingface.co/datasets/codeparrot/codeparrot-clean/raw/main/README.md` | 2026-11-08 | `35a59fb0` · 2022-10-10 |
| codeparrot per-file licences | `https://huggingface.co/datasets/transformersbook/codeparrot/raw/main/README.md` | 2026-11-08 | — |
| OpenWebMath card | `https://huggingface.co/datasets/open-web-math/open-web-math/raw/main/README.md` | 2026-11-08 | `fde8ef8f` · 2023-10-17 |
| FineWeb-2 card | `https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/raw/main/README.md` | 2026-11-08 | `af9c1333` · 2025-10-27 |
| ODC-By 1.0 | `https://opendatacommons.org/licenses/by/1-0/` | 2026-11-08 | v1.0 |
| Apache-2.0 | `https://www.apache.org/licenses/LICENSE-2.0.txt` | 2026-11-08 | — |
| Common Crawl ToU | `https://commoncrawl.org/terms-of-use` | 2026-11-08 | last updated 2024-03-07 |
| CC opt-out ledger | `https://commoncrawl.org/blog/common-crawl-foundation-opt-out-registry` | 2026-11-08 | 2025-09-17 |
| Parquet endpoint (example) | `https://huggingface.co/api/datasets/<id>/parquet/<config>/train/0.parquet` | 2026-11-08 | resolves to `refs/convert/parquet/...` |

In-repo: `port/configs/mix_refetch.toml`, `port/scripts/refetch_pretrain.py`,
`port/scripts/fetch_mix_sources.py`, `port/src/dormouse/train/loop.py`,
`port/src/dormouse/train/mixture.py`, `port/src/dormouse/config.py`,
`docs/guides/dataset.md`, `docs/records/reviews/2026-11-08-night.md`.

---

## 7. Вердикт (русский)

**Что доказано (проверено байтами и API, 2026-11-08).** Шесть источников и их
файлы существуют, размеры и sha256 совпадают с таблицей; веса микса взяты из
`mix_refetch.toml` и суммируются в 1.0. Лицензии объявлены так: FineWeb-Edu,
FineMath, OpenWebMath и FineWeb-2 — ODC-By 1.0 плюс Common Crawl ToU;
Cosmopedia — Apache-2.0, но **только в YAML карточки**, в тексте карточки о
лицензии ни слова; codeparrot-clean **не объявляет лицензию нигде**. Срез кода
(200 MiB) — это первые 21 252 строки шарда, и **40.08 % его байтов под
GPL-2.0/3.0, AGPL-3.0 или LGPL** (измерено перескачиванием шарда и точным
воспроизведением размера `.bin`). ODC-By §3.1 разрешает коммерческое
использование, §4.2/§4.3 требуют уведомлений, преамбула прямо говорит, что
лицензия покрывает базу, а не отдельные произведения. Чекпоинт записывает хэш
плана и **размеры** источников, но не их содержимое
(`port/src/dormouse/train/loop.py:455-505`), и ни в одном из двух чекпоинтов
на диске блока `corpus` не найдено (скан байтов).

**Что не доказано и требует юриста.** (1) Распространяется ли ODC-By на веса
модели — в определении Produced Work весов нет. (2) Влекут ли 40 % copyleft в
обучающих данных обязательства для весов или для хостинга (AGPL §13 — самый
острый вопрос); индустриальная позиция «обучение не производная работа» —
не проверена судом, и я её не утверждаю. (3) Покрывает ли Apache-2.0
сгенерированный текст Cosmopedia и не наследует ли он условия seed-данных
(AutoMathText, ODC-By). (4) Можно ли редистрибутировать сами `.bin`: ODC-By
разрешает, но CC ToU §1 — «non-transferable, non-sublicensable», то есть
передача данных не очевидна. (5) Отсутствие лицензии у codeparrot означает
«прав не передано» или отсылку к per-file лицензиям GitHub.

**Что нужно до коммерческого релиза.** Манифест на каждый файл (dataset id,
revision, URL, лицензия, sha256, байты, дата) — G1; хэши источников в блоке
`corpus` чекпоинта — G2; запись revision и резолвнутого URL вместо
`refs/convert/parquet` — G3; row-index сайдкар, чтобы takedown можно было
отобразить на байты — G4; исправить две неверные строки лицензий в скриптах —
G5; записать правило нарезки eval — G6; процедура удаления — G7; раздел
provenance в model card и исходящая лицензия на веса — G8. Цепочка
«манифест → план → чекпоинт» закрывается шестью хэшами: розыгрыш уже
детерминирован `(seed, step, steps)`, поэтому пошаговый лог байтов не нужен.

**Уверенность.** Высокая — в фактах §1 и в цитатах лицензий; средняя — в
списке инженерных требований; **низкая — в любых юридических выводах**: это
вопрос к юристу, а не к этому документу.

---

## 8. Open questions

1. Does a checkpoint count as a Produced Work under ODC-By §4.3, or as a
   Derivative Database under §4.2? The answer changes whether a model card notice
   is enough.
2. Is a 40 %-copyleft code slice acceptable to ship? If not, the allow-list and
   the re-fetch budget are a decision, and the slice must be rebuilt from a
   licence-filtered dataset (The Stack v2's permissive subset was the original
   first choice — `docs/records/research/2026-10-16-data-mixing.md:139`).
3. Should `fwedu_train.bin` be re-cut at a row boundary (it currently ends
   mid-document)? Re-cutting changes its sha256 and therefore every number
   measured on it.
4. Who signs the model card — and does the ODC-By §4.3 sentence go in the card,
   the repository README, or both?
