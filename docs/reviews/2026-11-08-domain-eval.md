# 2026-11-08 — per-domain held-out BPB: the windows, not the numbers

**Status: the instrument and the windows are ready; no BPB was run.** The GPU
was training `csa8k_prop` all day and this lane is CPU-only. Every number in §2
is a window definition or a byte count. The table itself is filled by §4 when
the card is free.

Why it exists: one BPB on one English window is the number the project has been
quoting. A company buys four: English, code, math, Russian. They must come from
**one checkpoint and one procedure**, or they are four incomparable claims.

---

## 1. Method

The instrument is `port/scripts/eval_by_domain.py`. It scores every source of a
mix plan through the same calls the trainer's own eval uses:

| step | call |
|---|---|
| resolve the checkpoint | `dormouse.cli._snapshot_run` (the snapshot is the run) |
| per-domain window | `dataclasses.replace(train, eval_data=..., eval_offset=0)`, plus `eval_bytes` when `--bytes` is given |
| window geometry | `dormouse.train.data.eval_batches_for` |
| stream | `dormouse.train.data.open_streams(..., eval_only=True)` |
| forward | `dormouse.train.loop.evaluate` (no autodiff, `rewind()` first) |
| holdout gate | `port/scripts/eval_window.py::refuse_training_overlap` (path rule) |

The two metrics are printed separately, as on the trainer's line:
**`bpb_fin = fin / ln2`** is the final-readout BPB (the model's BPB);
**`bpb_mean = loss / ln2`** is the depth-averaged training objective. They are
never the same number and a reader must not quote one as the other.

`--bytes` is **scored bytes per domain**. When it is omitted the snapshot's own
window is kept exactly, including a legacy `eval_batches` (for `csa8k_prop`
that is 4 × 1 × 8192 = 32 768 B, the same window its trainer scored); when it is
given it replaces both window keys. `--offset` is pinned and printed; every
window starts at `0`. An explicit `--bytes` must be at least the 64 KB floor, a
whole number of `batch × seq_len` windows, and the window plus one successor
byte per row must fit inside the eval file — `open_streams` refuses the wrap
rather than scoring bytes twice.

The last column, **bytes trained**, is not an estimate by weight. It is
`dormouse.train.mixture.replay_bytes` at `(checkpoint step + 1) × grad_accum`
micro-batches with the run's own seed — the exact draw a resume must reproduce,
including the successor byte each window reads. It is what proves the tail was
never touched, and it is also why a checkpoint that trained on the *whole* file
before the carve can still be scored here honestly: the stream never reached
the carve offset (§3).

The gate is the path rule from the 2026-11-08 data/eval audit: a window that IS
a training source is refused. A **copy** of training text under another name
passes the path check — no metadata can see content — so the replay column is
the second, independent check, and this document says that instead of implying
the path gate is stronger than it is.

---

## 2. The windows

All six plan sources now have a held-out file **no source names**. The last
`--tail-mb 8` MiB was carved out of each single-file source that lacked one (the
five new rows below; fwedu's split already existed); the training file kept its
path and was truncated to the head, the shape
`best/fwedu_train.bin` / `best/fwedu_eval.bin` already had. The carve is one-way
(§5).

| domain | training file (head, now) | eval window | bytes | offset | sha256 | rule |
|---|---|---:|---:|---:|---|---|
| fwedu (English) | `data_local/best/fwedu_train.bin` | `data_local/best/fwedu_eval.bin` | 8 386 155 | 933 743 230 | `003ba82d0d666becb63df77eabbc9a2df9975108bbe98650993a5048add5661f` | existing recovery split (document separator) |
| cosmo | `data_local/best/cosmo.bin` | `data_local/best/cosmo_eval.bin` | 8 390 601 | 293 808 222 | `2cefd6a9843c94931e166681b8c3fda0dc93b67f343dddd59cc04c2e3b6034b5` | separator |
| finemath | `data_local/best/finemath.bin` | `data_local/best/finemath_eval.bin` | 8 426 214 | 939 699 672 | `fa57dcaec27dd89b400ebae08d911dcfbbb225600a31e33c7cc869d0c97ebbdc` | separator |
| code | `data_local/mix/code.bin` | `data_local/mix/code_eval.bin` | 8 388 593 | 201 326 182 | `6669ef3984c03e53412d80a732eb68777824a93c92bf41e2d9c30f2a469932ee` | line-offset |
| math | `data_local/mix/math.bin` | `data_local/mix/math_eval.bin` | 8 386 973 | 201 314 844 | `3cb87a2ec60cdaaae9275c3b365b0ba6cf818e63335568ae5516de77cfafdf1d` | line-offset |
| ru (Russian) | `data_local/mix/ru.bin` | `data_local/mix/ru_eval.bin` | 8 388 576 | 201 321 740 | `f800f82f150335f41615fcb6772e368d9f8e27aacc1bda8cc273a0baf2ab4e37` | line-offset |

`offset` is the byte in the **original** source where the window starts; the
training file now ends there, so `offset == current training-file size` for every
new carve. Original source size = offset + window bytes.

The rule column is the cut, not a claim about text. The three `best/` files were
written by `refetch_pretrain.py`, which separates documents with
`\n\n---\n\n`; the cut starts at the last such separator at/before
`size − 8 MiB` (within a 64 KiB search). The three `mix/` files were written by
`fetch_mix_sources.py`, which writes raw text with no document marker — a `---`
inside code or math is content, not a boundary — so the cut is a **fixed,
line-aligned offset** (`size − 8 MiB`, advanced to the first byte after the next
newline). All six windows decode as 100 % UTF-8; the `mix/` windows start
mid-document by construction, and byte-level BPB does not care.

The four domains the company asked for are **fwedu, code, math, ru**. The other
two rows exist because the script scores *every* source of the plan; cosmo is
synthetic math text (English), finemath is math. They are useful controls, not
extra company domains.

Health of the carve: the largest window is 8.43 MB; a 1 MiB scored window at
8192-byte rows needs 1 048 704 source bytes, which fits in all six. The default
window for `csa8k_prop` is 32 768 B (its snapshot says
`eval_batches=4`); the command in §4 asks for the standard 196 608 B instead,
because a 32 KB window is narrower than the 64 KB floor the config uses to argue
that a number is a measurement (`MIN_EVAL_BYTES`, ±0.19 BPB at 64 KB). A per-domain
report should use the widest window the eval file allows; 196 608 B is the
historical default and is what §4 quotes.

---

## 3. Why this checkpoint's tails were never read

The carve ran at 13:08 while `csa8k_prop` was still training (~7.4k of 8 000
optimizer steps). Its data streams were built at 11:01 from the *whole* files,
so the path gate alone cannot prove the tail was unread — the replay does.
`replay_bytes` at the run's planned 8 000 completed optimizer steps (payload
`step=7 999`, `grad_accum=4`, seed 1) gives the stream's exact per-source
consumption:

| domain | bytes the stream read | carve offset | headroom |
|---|---:|---:|---:|
| fwedu | 144 336 081 | 933 743 230 | 789 407 149 |
| cosmo | 31 764 261 | 293 808 222 | 262 043 961 |
| finemath | 15 804 297 | 939 699 672 | 923 895 375 |
| code | 28 896 711 | 201 326 182 | 172 429 471 |
| math | 15 296 331 | 201 314 844 | 186 018 513 |
| ru | 26 078 319 | 201 321 740 | 175 243 421 |

262 176 000 bytes total, 9.5 % of the six training files; the smallest
headroom is 172 MB. The stream is sequential from byte 0, so a carve that starts 172 MB past
the last byte it read is held out for **this** checkpoint even though the file
contained it at launch time. The truncation then makes it held out for every
later run too. The eval script recomputes this column at whatever step the
checkpoint reports, so the table itself will carry the proof.

---

## 4. The command (when the card is free)

```sh
cd port
uv run scripts/eval_by_domain.py \
    --ckpt-dir ../checkpoints/csa8k_prop \
    --ckpt-name csa8k_prop \
    --mix-plan configs/mix_refetch.toml \
    --bytes 196608 \
    --device cuda
```

Preflight without loading the model (CPU, seconds; already run here):

```sh
cd port
uv run scripts/eval_by_domain.py \
    --ckpt-dir ../checkpoints/csa8k_prop --ckpt-name csa8k_prop \
    --mix-plan configs/mix_refetch.toml --check-windows
```

It prints each domain's window path and size, and refuses before any forward
pass if a window is missing or is a training source. `--domains
fwedu,code,math,ru` scores only the four company domains; `--bytes` may be any
multiple of `batch × seq_len` that fits the window; `--offset` stays 0. The
default `--bytes` is the checkpoint's own window, and `--device auto` picks CUDA
only when ≥2 GiB of the card is free.

The output is the table:

```
| domain | bytes scored | bpb_fin | bpb_mean | bytes trained |
```

`bpb_fin` is the model's BPB; `bpb_mean` is the depth-averaged objective. Do not
quote one as the other, and quote `bytes scored` beside every number — the same
checkpoint at another window is another measurement.

---

## 5. Rebuilding the windows

```sh
cd port
uv run scripts/carve_eval_tails.py --plan configs/mix_refetch.toml
```

The script prints the table in §2 and does nothing to an existing
`<name>_eval.bin`: it hashes it and skips it. **The carve is one-way** — the
training head is truncated, so a lost window can only be rebuilt by re-fetching
the source (`fetch_mix_sources.py`, `refetch_pretrain.py`) and carving the fresh
file. Do not delete `data_local/mix/*_eval.bin` or `data_local/best/*_eval.bin`.

---

## 6. Tests

```sh
cd port
CUDA_VISIBLE_DEVICES="" uv run pytest tests/test_eval_by_domain.py -q
```

Nine CPU tests: the five-column markdown table, the `_train → _eval` naming
rule, the refusal when the window is also a source (and the pass when it is a
distinct file), the missing-window refusal naming the carve command, the default
window keeping a legacy `eval_batches` while explicit `--bytes` replaces it, the
exact replay of `grad_accum` micro-batches, the checkpoint's `step` record, and
the two carve rules (document separator vs fixed line offset) including the
truncation. No checkpoint is loaded and no forward runs.
