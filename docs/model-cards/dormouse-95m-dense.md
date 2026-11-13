# dormouse 95M dense — model card

**Config:** [`port/configs/flagship3_dense.toml`](../../port/configs/flagship3_dense.toml)
· **Card date:** 2026-11-08 · **Status:** pretrained. No SFT, no RL, no safety tuning.

This card describes **one config and one run**: the `control` arm of the
2026-11-08 recipe ladder, trained on the refetched corpus. It is not the best
arm in that ladder, and it is not the old `flagship3` checkpoint.

## 1. What it is

A **95,035,200-parameter, 12-layer decoder-only transformer over raw bytes**.
There is no tokenizer: `vocab = 256`, one position per byte, `embed.weight` is
`[256, 832]`. English text costs about 4× more model positions than the same
text costs a BPE model's tokens; that is the price of byte-level modelling.

| fact | value | artifact |
|---|---:|---|
| parameters | **95,035,200** | `cd port && .venv/bin/python -m dormouse.cli info configs/flagship3_dense.toml`; pinned by [`test_invariants_checkpoint.py`](../../port/tests/test_invariants_checkpoint.py) `::test_the_flagship_transformer_has_no_nonpersistent_buffer` |
| layers | 12 unique, **one pass** (`n_layers=12`, `loop_blocks=1`, `max_iter=1`) | config |
| d_model / heads | 832 / 13 heads × 64 | config |
| FFN | 3,072-wide, **two linears + GELU** (not SwiGLU) | [`transformer.py`](../../port/src/dormouse/model/transformer.py) `TransformerBlock._block` |
| context trained | **1,024 bytes** | config `max_seq_len=1024` |
| attention | exact causal SDPA, RoPE on every block, no window | `model/transformer.py`; `nope_layers=0` |
| vocab | 256, **one token per byte, no tokenizer** | config |
| precision | bf16 projections, fp32 residual stream | `bf16=true`; `mm_mixed` in `model/transformer.py` |
| not present | KDA, CSA, mHC, AttnRes, Engram, MoE, TSCT | every corresponding flag is `false` in config |

Parameter breakdown (`dormouse info`): embed 212,992 + transformer blocks
94,608,384 (12 × 7,884,032) + final norm 832 + untied lm_head 212,992 =
**95,035,200**. The 1M-context machinery the repo writes about (CSA/HCA,
`LoopBlock`) is **not in this model**: `use_csa=false`, and the dense
`block_form="transformer"` refuses the KDA-only flags
(`port/src/dormouse/config.py::_KDA_ONLY_FORM_FLAGS`).

## 2. Training data — the 2026-11-08 refetch

The old **`mix4`** corpus and the eval window every pre-incident number
shared lived in `data_local/` (80 GB), deleted by a merge checkout on
2026-11-08 ([incident report](../reviews/2026-11-08-merge-incident.md)). What
exists is the **refetch**: six sources written back by
`port/scripts/refetch_pretrain.py` and `fetch_mix_sources.py`, combined by
[`mix_refetch.toml`](../../port/configs/mix_refetch.toml) — one stage, at most
4 epochs, mixture seed 20261027.

| source | weight | file | served at the 120 MB budget |
|---|---:|---|---:|
| FineWeb-Edu | 0.55 | `data_local/best/fwedu_train.bin` | 66.3 MB (55.2%) |
| Cosmopedia | 0.12 | `data_local/best/cosmo.bin` | 13.9 MB (11.5%) |
| Russian text | 0.10 | `data_local/mix/ru.bin` | 11.9 MB (9.9%) |
| Code | 0.11 | `data_local/mix/code.bin` | 13.3 MB (11.1%) |
| Math | 0.06 | `data_local/mix/math.bin` | 7.2 MB (6.0%) |
| FineMath | 0.06 | `data_local/best/finemath.bin` | 7.5 MB (6.2%) |
| **total** | **1.00** | | **120.1 MB** |

The replay column is not a design target: it is `mixture.replay_bytes` measured
at this budget, each source within 0.5 pp of its weight
([night report](../reviews/2026-11-08-night.md) §1). The run itself reports
`bytes_seen = 119,996,416` (14,648 × 8 × 1,024); the 0.1 MB difference is the
replay table's rounding.

The **eval window** is `data_local/best/fwedu_eval.bin[0:196608]`: the tail of
the same Hugging Face shard, carved at a document separator, in a file **no
training source reads** (`mix_refetch.toml` header; night report §1). The
held-out guarantee is the path check in
[`eval_window.py`](../../port/scripts/eval_window.py) `::refuse_training_overlap`:
it compares paths, so it cannot detect a copy of the same text under another
name. That limitation is the bound of the claim.

`doc_masking=false` for this run: the `mix/{code,math,ru}.bin` slices carry no
document boundaries, so every run on this corpus pins masking off and says so
(night report §1; [`recipe_ladder.sh`](../../port/scripts/recipe_ladder.sh)).

**Training recipe.** 14,648 steps × batch 8 × seq 1,024 = 119,996,416 bytes,
`lr=3e-4`, `opt=mix` (Muon with `muon_lr_mult=12` + AdamW), Muon bf16
Newton–Schulz at **5 steps** with bf16 optimizer state (`FAST=1`), seed 1, eval
every 3,662 steps. Artifacts: `recipe_ladder.sh`,
`/home/sehaxe/logs/ladder/control.log`, `/home/sehaxe/logs/ladder/control_refetch.jsonl`.

## 3. The numbers

### 3.1 Headline — 1.4906 BPB at 120 MB

| measurement | value | artifact |
|---|---:|---|
| held-out BPB (`bpb_fin`) | **1.4906** | `/home/sehaxe/logs/ladder/control_refetch.jsonl`, final `"event": "eval"` row |
| exact value | `1.4905827035298236` | same row, step 14,648 |
| bytes seen | 119,996,416 (120.0 MB) | same row |
| eval window | `best/fwedu_eval.bin`, `off=0`, 196,608 B | `eval_bytes=196608`, `eval_offset=0` on the row |
| scored | 196,608 B = 24 × 8 × 1,024 | same row (`scored=196608`) |
| text log line | `eval step=14648 off=0 bpb_fin=1.491 … over 196608 B` | `/home/sehaxe/logs/ladder/control.log` |

The card quotes **`bpb_fin`**, the final readout, never `bpb_mean`/`loss`
(the depth-averaged objective) — the contract from
[`2026-11-06-bpb-accounting.md`](../reviews/2026-11-06-bpb-accounting.md). On
this row all three are equal, but the label is still the contract.

**Read the window correctly: this is 192 independent 1 KiB windows, not one
196 KiB context.** The model attends within 1,024 bytes; the 196,608-byte score
is the byte mean over 192 forward passes. It is a compression number, not a
long-context result ([long-context research](../research/2026-11-08-long-context-eval.md) §0.1).

### 3.2 The old 1.4946 @ 250 MB is a different corpus — do not compare

| run | BPB | bytes seen | corpus | eval window | artifact |
|---|---:|---:|---|---|---|
| `e1v4` bytes arm | 1.4946 | 249,987,072 | `mix_v7`, **12 sources, 3 stages** | `mix4/web_eval.bin[0:196608]` | `/home/sehaxe/logs/e1v4/bytes_s1.jsonl` |
| this card | **1.4906** | 119,996,416 | `mix_refetch`, **6 sources, 1 stage** | `best/fwedu_eval.bin[0:196608]` | `/home/sehaxe/logs/ladder/control_refetch.jsonl` |

They differ in the source mixture, the number of curriculum stages, the eval
file, and the amount of training (2.08×). The 0.0040 difference between them is
**not an improvement** and must never be quoted as one. The old eval file
`mix4/web_eval.bin` died with the corpus
([incident](../reviews/2026-11-08-merge-incident.md)).

### 3.3 Compressor anchors on exactly the same bytes

Measured by [`port/scripts/eval_suite.py`](../../port/scripts/eval_suite.py) on
`best/fwedu_eval.bin[0:196608]`, CPU-only:

| codec | bits/byte | bytes out |
|---|---:|---:|
| bzip2 `-9` | **2.5878** | 63,599 |
| xz `-9` | **2.7134** | 66,684 |
| zstd `-19` | **2.7703** | 68,084 |

The model is **1.0973 bits/byte below the best anchor** (bzip2). Both sides are
scored on the same 196,608 bytes, but the comparison has two honest caveats:
the model saw 120 MB of training bytes and the compressors saw nothing, and the
compressors are general-purpose, not tuned for this domain. The number says the
model learned structure in this window; it says nothing about intelligence.

### 3.4 What 1.4906 is not

* **Not the ladder's best arm.** The ladder's `all` arm measured 1.4778
  (−0.0128 vs control), but that is **one seed**, at the top of the 0.008-0.011
  BPB seed band, and larger than the sum of its five parts. The ladder's own
  verdict is "one candidate, not yet a result" (night report §5). This card
  documents the control, not a winner.
* **Not a converged model.** 119,996,416 bytes / 95,035,200 params = **1.26
  bytes/param**, against roughly 20 B/param at Chinchilla. This is a point on a
  curve.
* **Not a capability score.** It is held-out compression on one English window.
  No task benchmark was run on this run.
* **Not reproducible from a checkpoint on disk.** `ckpt_every=1000000`, so the
  ladder saved no checkpoint. Reproducing 1.4906 means re-running the 37-minute
  arm (§4a), not loading a file.

## 4. Reproduce

### (a) The exact 1.4906 — a 37-minute retrain

```bash
cd port
FAST=1 MIX=configs/mix_refetch.toml RUN_TAG=_refetch ONLY=control \
    bash scripts/recipe_ladder.sh
# final row of /home/sehaxe/logs/ladder/control_refetch.jsonl: bpb_fin 1.4905827035298236
```

This is a full training arm, not an eval; it needs the card. It is the only way
to reproduce the number because the run saved no checkpoint.

### (b) Any checkpoint of the same snapshot, on a named window

```bash
cd port
.venv/bin/python scripts/eval_suite.py \
    --ckpt ../checkpoints/<run>/<name>.pt \
    --eval ../data_local/best/fwedu_eval.bin \
    --bytes 196608 --offset 0 --device cpu
```

`eval_suite.py` walks the same load/eval path as `dormouse eval` and prints a
pasteable markdown block: parameter count (meta device), `bpb_fin`, and the
three compressor anchors on the same bytes. A missing compressor is a **loud
`SKIP`**, never a 0. On a busy card the BPB row is a loud `SKIP` and the CPU
anchors still print. The exact command is echoed in the block, so a pasted
number keeps its artifact.

**Verification status, 2026-11-08.** The script was run end-to-end on CPU on the
only 95M checkpoint on disk at `--bytes 65536`, the eval path's floor:

```bash
.venv/bin/python scripts/eval_suite.py \
    --ckpt ../checkpoints/flagship3/flagship3.pt \
    --eval ../data_local/best/fwedu_eval.bin --bytes 65536 --device cpu
```

It printed `bpb_fin=1.5909`. That is a plumbing check of an **old-corpus**
checkpoint on the new window — not this card's number. The 196,608-byte BPB
path was not run on CPU today: the 65,536-byte run took 19m34s wall, so the full
window is about an hour, and the GPU is training. The anchors were re-measured
on the exact 196,608-byte window and are the three values in §3.3.

## 5. Retractions that apply

Retracted claims stay published next to what replaced them (repo rule). These
are the ones a reader of this card could otherwise trip over:

| claim | status | why | artifact |
|---|---|---|---|
| `1.4621 @ 250 MB` | **RETRACTED** | its `fwedu.bin` `off=100,000,000` window was inside the bytes the same 0.25 GB run had already read: training data, not held out | [`2026-11-08-data-eval-integrity.md`](../reviews/2026-11-08-data-eval-integrity.md) §5.3 row 4, §7.1 |
| `1.927 @ 98 MB` | **RETRACTED** | captured before the Muon LR fix (`f3short`, mtime before `0a7dfcd`); the scheduler overwrote `muon_lr_mult` | [`2026-11-07-muon-audit.md`](../reviews/2026-11-07-muon-audit.md) D1-D2; README |
| "the Muon fix was worth −0.267 BPB" | **RETRACTED** | compared two different mixtures | README, retracted list |
| "16× from the recipe ladder" | **NOT OURS** | that is Ripple-65B's result at 4B; this repo's measured optimizer rung is 1.92× the step for +0.01 BPB | [`2026-11-08-ripple-65b.md`](../reviews/2026-11-08-ripple-65b.md) |
| "3 Newton–Schulz steps are a tie" | **REJECTED at this shape** | +0.27 BPB at `[832, 3072]`; this config uses 5 steps | [`2026-11-07-optimizer-suite.md`](../reviews/2026-11-07-optimizer-suite.md) |
| "34 MB of state at 1M" | **RETRACTED** | 33 MiB is one CSA slot; the 3-slot loop carries 99 MiB, a 12-layer stack 396 MiB. This model has no CSA and makes no 1M claim | README; [long-context research](../research/2026-11-08-long-context-eval.md) §0 |
| ladder `all` arm at −0.0128 | **ONE SEED, NOT A RESULT** | at the top of the 0.008-0.011 noise band and larger than the sum of its parts | night report §5; `.bulba/plan.md` |
| `bpb_mean` / `loss` as "the BPB" | **NOT USED** | that is the depth-averaged training objective; this card quotes `bpb_fin` | [`2026-11-06-bpb-accounting.md`](../reviews/2026-11-06-bpb-accounting.md) |

## 6. What this model cannot do

Blunt, because a model card that only lists strengths is an ad.

* **Long context. It has none.** It was trained at **1,024 bytes** and has
  `use_csa=false`; the trainer refuses any `seq_len > max_seq_len`
  (`port/src/dormouse/config.py:5506`), and no position beyond 1,024 was
  trained. No retrieval,
  passkey or needle test exists for this run. The repo's 1M design is a
  different architecture (`LoopBlock` + CSA/HCA) and **no trained CSA artifact
  is on disk** ([long-context research](../research/2026-11-08-long-context-eval.md) §0).
  A "1M context" claim cannot be made from this checkpoint in any form.
* **No instruction following.** This is a pretrained next-byte predictor. It
  has not seen an instruction/response pair, has no chat template, no system
  prompt, no turn markers. Ask it a question and it continues the text in the
  register of its 120 MB slice.
* **No safety tuning.** No RLHF, no refusal training, no content filter, no
  red-team pass. It will complete any byte sequence it is given, including
  harmful, private or false text, and it has no notion of "should not answer".
* **No knowledge worth citing.** 120 MB of training text, 1.26 bytes/param.
  Factual recall, dates, names, arithmetic and citation-like output are not
  reliable and were not measured. Do not use it as a source.
* **No BPE efficiency.** One token per byte means about 4× more positions per
  English sentence than a BPE model needs, and a 256-way head has to spend
  capacity re-learning subword statistics that a tokenizer gives away free.
  It also has no vocabulary to constrain, so sampling produces invalid UTF-8
  unless the caller assembles it.
* **No downstream results.** ARC-Easy, HellaSwag, pass@k, MMLU-style tasks:
  none were run on this run, so none are claimed. The repo's `dormouse bench`
  exists; this card has no bench row.
* **No code or math specialisation.** The mixture is 11.1% code and 6.0% math
  bytes. That is exposure, not capability; no code-execution or math-accuracy
  test was run.
* **No multilingual claim.** 10% of the mixture is Russian text. One language
  at 10% is not a multilingual model, and no per-language BPB was measured.
* **No tool use, no structured output, no function calling, no JSON mode.**
  There is no prompt format, no special-token protocol trained into the weights
  (the repo's SFT control bytes are a later, unrun stage), and no agent loop.
* **Not a released checkpoint in the usual sense.** The run behind the headline
  saved no weights (`ckpt_every=1000000`); the only 95M checkpoint on disk is
  the old mix4 `flagship3`, a different corpus and a different number.

What it **can** do, precisely: predict the next byte of English web text at
**1.4906 bits/byte** after 120 MB of training on the 2026-11-08 refetched
mixture, on the pinned 196,608-byte window — 1.0973 bits/byte better than
`bzip2 -9` on the same bytes.

## 7. Artifact index

| what | path |
|---|---|
| model config | [`port/configs/flagship3_dense.toml`](../../port/configs/flagship3_dense.toml) |
| data mixture | [`port/configs/mix_refetch.toml`](../../port/configs/mix_refetch.toml) |
| headline run log (JSONL) | `/home/sehaxe/logs/ladder/control_refetch.jsonl` |
| headline run log (text) | `/home/sehaxe/logs/ladder/control.log` |
| ladder runner | [`port/scripts/recipe_ladder.sh`](../../port/scripts/recipe_ladder.sh) |
| one-command reproduction | [`port/scripts/eval_suite.py`](../../port/scripts/eval_suite.py) · [`port/tests/test_eval_suite.py`](../../port/tests/test_eval_suite.py) |
| parameter pin | [`port/tests/test_invariants_checkpoint.py`](../../port/tests/test_invariants_checkpoint.py) `::test_the_flagship_transformer_has_no_nonpersistent_buffer` |
| corpus loss | [`docs/records/reviews/2026-11-08-merge-incident.md`](../reviews/2026-11-08-merge-incident.md) |
| refetch + ladder record | [`docs/records/reviews/2026-11-08-night.md`](../reviews/2026-11-08-night.md) |
| long-context verdict | [`docs/records/research/2026-11-08-long-context-eval.md`](../research/2026-11-08-long-context-eval.md) |
| retraction evidence | [`docs/records/reviews/2026-11-08-data-eval-integrity.md`](../reviews/2026-11-08-data-eval-integrity.md) |
