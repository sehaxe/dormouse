# Invariants audit — checkpoint, resume, eval, data, run history

2026-11-07 · commit `e954e8b` + this branch's tests and two fixes · CPU only
(`CUDA_VISIBLE_DEVICES=""`, torch 2.13.0+cu129) · worktree
`/home/sehaxe/dormouse-wt/invariants` off `/home/sehaxe/dormouse`

Four defects in a row got past a checkpoint/eval read: a non-persistent
`lfb_bias` that restarted the routing integrator, an `opt_graph` replay that
froze the bank cache key, a `bytes_seen` that ignored `grad_accum`, and an eval
that scored one 65 KB window and called the spread ±0.2 BPB. flagship3
(`block_form="transformer"`, 95 035 200 params) is about to be resumed
repeatedly, so this audit turns each invariant into an executable check instead
of a reading of the code.

Every claim below is a test in `port/tests/test_invariants_*.py`; the numbers are
the values those tests' own fixtures produce (a test asserts the equality, and
the quote is from the same fixture).

| # | invariant | verdict | test | number |
|---|---|---|---|---|
| CP-1 | every named buffer is persistent, saved elsewhere, or provably derived | **HOLDS** | `test_invariants_checkpoint.py::test_every_nonpersistent_buffer_is_carried_or_provable` | flagship3: 0 buffers, 99 state keys; kda: 2 non-persistent, both classified |
| CP-2 | every optimizer state tensor round-trips | **HOLDS** | `::test_every_optimizer_state_tensor_round_trips` | 8 Muon + 11 AdamW state entries, all tensors equal after `load_state_dict` |
| CP-3 | objective counters are saved or pure | **HOLDS** | `::test_the_objective_counters_are_in_the_file_or_pure` | `step`, `opt.steps`, `lfb_counts`, `lfb_bias`, host `step`/`counts` all carried |
| RS-1 | 30 steps == 15 + resume at 15 | **HOLDS** | `test_invariants_resume.py::test_resume_at_15_is_bit_identical_to_30` | transformer `0.0005600823787972331` both; kda `0.009409088641405106` both; sha256 equal |
| RS-2 | resume mid-decay == uninterrupted | **HOLDS** | `::test_resume_mid_decay_is_bit_identical` | `0.0005772272706963122` both, hash equal |
| RS-3 | `--set steps=` mid-run | **HOLDS / fixed** | `::test_extending_steps_mid_run_matches_the_new_plan`, `::test_a_steps_change_matches_the_stitched_schedule`, `::test_resuming_past_the_plan_refuses_loudly` | extend 30→45: `0.00041978745139203966` both; stitched plan matches exactly; a `steps` BEHIND the checkpoint refuses, an equal one is a legal zero-step load |
| EV-1 | flat `bpb_fin` == `-log2 p(target)` over the same bytes | **HOLDS** | `test_invariants_eval.py::test_reported_bpb_fin_is_the_scored_byte_mean` | reported `9.060773534065909`, independent `9.060773534065909` (diff 0.0) |
| EV-2 | doc-masked `bpb_fin` == the same bytes | **WAS BROKEN, FIXED** | `::test_a_doc_masked_eval_is_weighted_by_scored_bytes` | mean-of-means `9.070214005737071` vs byte mean `9.073532327768056`: **0.00332 BPB** |
| DA-1 | no doc-masked training target crosses a boundary | **HOLDS** | `test_invariants_data.py::test_no_packed_target_crosses_a_document_boundary` | every scored target inside its document and not a start |
| DA-2 | skipped targets == the sidecar's count | **HOLDS** | `::test_skipped_targets_equal_the_sidecar_derivation` | 4 windows: scored 116, masked 12, doc_ends 1, padded 11 — all recomputed |
| DA-3 | `eval_docmask=sidecar` uses the train rule | **HOLDS** | `::test_eval_docmask_sidecar_uses_the_train_rule` | mask == "target is not a document start" on both readers |
| DA-4 | `doc_masking=false` is clean HEAD's stream | **HOLDS** | `::test_doc_masking_off_is_the_pinned_flat_stream` | three windows equal `e954e8b` sha256s and the hand-built shift |
| HI-1 | run-history curve is monotone and bytes counted once | **HOLDS** | `test_invariants_history.py::test_the_real_flagship_log_dedupes_monotone_and_counts_bytes_once` | 1335 rows → 1283 steps, max 128 300, 3 153 100 800 B unique vs 2 094 838 579 200 B naive sum |

---

## 1. Checkpoint completeness

Method: build the model, take `state_dict()`, and split it into parameters and
buffers. For every named buffer that is **not** in `state_dict`, the test
requires one of three proofs:

* it travels in the checkpoint's `moe` section (`lfb_bias`) and a fresh model's
  zero is demonstrably not the trained value;
* rebuilding a fresh model from the same config reproduces it bit for bit
  (`_order_idx`, `share_u_idx`) — the only legal way to be non-persistent;
* it is telemetry with a documented reset (`layerrope._applications`, a
  per-forward application counter read only on the log cadence).

An unclassified non-persistent buffer fails the test. That is the regression
gate: the `lfb_bias` defect was a buffer nobody had classified.

### flagship3 (`block_form="transformer"`, 95 035 200 params)

The same inventory also runs on a tiny `bf16=True` transformer — flagship3's
compute dtype — so the arm's checkpoint is audited, not only the fp32 default.

`test_the_flagship_transformer_has_no_nonpersistent_buffer` builds
`configs/flagship3_dense.toml` on the meta device: **95 035 200 params, 99
`state_dict` keys, zero buffers.** The checkpoint is `state_dict` + `opt` +
`step` + an empty `moe` section; there is nothing to reset on resume.

### kda (`configs/small.toml`, 9 476 993 params)

Non-persistent buffers, and their home:

| buffer | home | check |
|---|---|---|
| `loop.moe.lfb_bias` | checkpoint `moe` section (`save_ckpt`/`_restore_lfb`) | the trained kda fixture's bias reaches `|b| = 1.5`; `load_ckpt` prints `lfb_bias range [+0.000, +1.500]` and the loaded tensor equals the file's; a fresh model reads all zeros |
| `loop.engram._order_idx` | derived `arange(len(orders))` | equals a fresh build bit for bit |
| `layerrope._applications` (arm) | telemetry | int64 scalar, starts at 0 |
| `loop.moe.share_u_idx` (arm) | derived, seeded from cfg | equals a fresh build bit for bit |

`lfb_counts` is persistent (it is in `state_dict`); `engram.tables` is a
parameter on the in-VRAM substrate and the `.ngram` sibling (`rows`, `momentum`,
`step`, `counts`) on the host substrate, already gated by
`tests/test_engram_host.py::test_host_tables_round_trip_through_the_ngram_sibling`
and `::test_resume_continues_a_host_run`.

### Optimizer and counters

`MixOptimizer.state_dict()` is `{steps, muon, adamw}`. After five real steps the
test loads that dict into a fresh optimizer over a fresh model and compares the
two recursively: **on the transformer fixture, 8 Muon momentum entries and 11 AdamW
`step`/`exp_avg`/`exp_avg_sq` entries, every tensor equal.** Per-parameter key sets are compared
too, so a new state key that `state_dict` forgets fails here.

Counters: `opt.steps` is in the `opt` half; `_adamw_t` is rebuilt from the
per-parameter `step` tensors at load (`MixOptimizer.load_state_dict`);
`lfb_counts` is persistent; `lfb_bias` is in `moe`; the host Engram `step` and
per-row `counts` are in `.ngram`; `anneal_alpha` and `_sample_depth` are pure
functions of `(seed, step)`; bank/QAT caches are rebuilt and invalidated once
per trained step. The EMA shadow is a SECOND model half under `model_ema`
(served) beside `model` (resumed), covered by
`tests/test_train4.py::test_ema_checkpoint_round_trip_and_legal_decay_change`.

**One gap, off by default:** `moe_router_jitter > 0` draws
`torch.randn_like` from the process-global RNG on every training forward
(`model/experts.py:1624`). No RNG state is checkpointed, so a resumed run draws
a different stream. The config default is `0.0` and no preset turns it on, so
every current run is unaffected — but this is a resume-determinism hole the day
someone enables it. Recommendation: derive the jitter from `(seed, step,
iteration)` like the depth sample, or refuse `--resume` with jitter on.

## 2. Resume determinism

The check is a real interruption, not a small `steps`: run 30 steps with
`ckpt_every=15`, raise at the step-16 log (after the step-15 checkpoint is on
disk), then resume on the same model object with `resume=True`. The reference is
an uninterrupted 30-step run from the same seeded model.

| case | reference loss | resumed loss | weights+buffers |
|---|---|---|---|
| transformer, fixed schedule | 0.0005600823787972331 | 0.0005600823787972331 | equal (`cbf18fef…`) |
| kda, fixed schedule (lfb coef 0.5) | 0.009409088641405106 | 0.009409088641405106 | equal (`233adc9f…`) |
| transformer, mid-decay (`decay_frac=0.6`, decay starts at 12) | 0.0005772272706963122 | 0.0005772272706963122 | equal |
| transformer, extend 30 → 45 (`warmup=20`) | 0.00041978745139203966 | 0.00041978745139203966 | equal |

The optimizer's moments are compared through the step-30 checkpoint's `opt`
half, not just the loss: the files two runs write agree tensor for tensor.

`--set steps=` is the one progress key that changes the future schedule. The
test pins the contract in two pieces:

* when the two plans agree on every step already trained (warmup covers the
  checkpoint), extend-and-resume is bit-identical to a clean run of the new
  length — measured above (30 → 45, both `0.00041978745139203966`). The LR at
  step 20 does move (`0.0094` under `total=30`, `0.01` under `total=45`), so the
  new plan really is applied rather than frozen.
* when they cannot agree (decay had already started before the checkpoint), the
  resumed run matches the **stitched** schedule — old plan below the checkpoint,
  new plan from it — exactly (`0.0004485073732212186` both, hash equal). A fresh
  45-step run cannot match by construction: its steps 12–14 used a different LR
  prefix (up to `1.5e-3` lr apart), and the final losses differ by
  `5.046051228418946e-06` nats. That gap is the schedule, not lost state.

**Fixed:** a resume whose `steps` ends BEHIND the checkpoint used to run zero
steps and return `{"steps": 10, "opt_steps": 15}` — a report for a run that did
not happen (the optimizer count and the claimed count disagree). `train()` now
refuses before the loop (`port/src/dormouse/train/loop.py:1390`): "the
checkpoint is at step 15 and steps=10 is behind it … Escape: --set steps=N with
N >= 15, or omit --resume." Ending AT the checkpoint (`steps == start`) stays a
legal zero-step LOAD — `steps == opt_steps`, so nothing is misreported, and
`test_train4`'s EMA read depends on it.

## 3. Eval correctness

The independent check is written from the bytes: load the checkpoint with
`load_ckpt`/`load_model_state`, open the same `eval_data` at the same
`eval_offset`, run the forward, compute
`-log_softmax(logits).gather(target)`, sum the nats and divide by
`ln2 × scored bytes`. No trainer eval helper is called.

**Flat (EV-1):** reported `bpb_fin` equals the independent value exactly
(`9.060773534065909`, diff `0.0`), over `256 B` at `off=3`; the two per-batch
BPBs differ (`8.6482`, `9.4734`), so the check is not a constant.

**Doc-masked (EV-2, was broken):** `_eval_loss` summed each batch's
`last_breakdown["fin"]` and divided by the batch COUNT, while the eval line
claims `over N B` of scored bytes. On a window whose batches score 123 and 125
positions, the mean-of-means read `9.070214005737071` where the same scored
bytes read `9.073532327768056` — **0.00332 BPB**, 332× the 1e-5 gate and the same
order as the 0.002–0.005 paired-eval resolution the A/B protocol quotes. The fix weights every mean
by that batch's scored positions (`loop.py:873` `_eval_loss`); the flat path has equal batch
sizes, so every flat number on record is unchanged. The test recomputes the
mean-of-means and asserts the fixture separates the two definitions, so a revert
fails on the number, not on a comment.

What was **not** the cause: the eval window (fixed by `eval_bytes`/`eval_batches`
and printed), the offset (pinned and printed), padding (masked on both readers),
and the window geometry (`scored` is the mask sum, not `x.numel()`).
The other half of the 65 KB story — a window below `MIN_EVAL_BYTES` refusing,
and `eval_bytes` naming the width — is already gated by
`tests/test_bpb_accounting.py::test_eval_refuses_a_window_below_the_floor` and
`::test_bytes_seen_counts_grad_accum_micro_batches`; this audit adds the
definition of the number the window reports.

**Open, training side:** with `doc_masking=true` and `grad_accum > 1`, the
trainer divides each micro-batch's masked mean by `grad_accum` and sums — equal
weight per micro-batch regardless of how many positions it scored, although the
comment claims "a micro-batch with fewer scored bytes contributes less". The
right mean is weight-by-scored-count. No shipped preset uses `doc_masking`
together with `grad_accum > 1` today; changing it changes gradients, so it is
left as a decision, not a hot fix.

## 4. Data invariants

Fixture: five documents of 5, 300, 7, 64 and 1 bytes, `seq_len=16`, `batch=2`
(the 300-byte document forces chunking, the 1-byte one forces a zero-target
row). The reference packer in the test is written from the module docstring, not
imported from the reader.

* **No boundary crossing.** Every scored position's target byte is inside the
  row's document and is not a document start; the expected tensors and mask are
  equal to the reader's for every row.
* **Skipped count = sidecar.** Four windows (8 rows) score **116** positions and
  drop **12** (`docmask_masked=12`), with 1 whole-document end and 11 padding
  bytes; every number is recomputed from the sidecar and the cursor. A skipped
  position is either padding or a document's last byte — never a mid-document
  target.
* **Eval uses the same rule.** `MaskedByteStream` masks every flat target that
  is a document start (and the ring seam); the packed trainer masks a
  document's last byte, whose successor is that same start. The test asserts the
  one predicate on both readers.
* **`doc_masking=false` is clean HEAD.** The first three windows of
  `bytes(range(16)) * 256` at batch 2 / seq 64 reproduce the sha256s recorded on
  `e954e8b` and the hand-built `[b, seq+1]` shift.

## 5. Run-history adapter

`port/src/dormouse/runlog.py` is the tiny reader the audit uses:
`read_rows` (LOUD on a malformed line), `tag_segments` (one segment per `run`
row, derive `bytes_seen` as `step × batch × seq_len × grad_accum` for old
files), `dedupe` (**last** row per step wins, ascending — a rollback re-runs a
step and the run that continued is the one whose value survives), `curve`, and
`unique_bytes` (the max, never the sum, and it refuses a `bytes_seen` that
disagrees with `step × batch × seq_len × grad_accum` — the 2026-10-17 defect
would be caught here).

On `/home/sehaxe/logs/flagship.jsonl`:

* 1335 `log` rows, 4 `run` rows (the first died before its first log line), two
  backwards jumps (`102800 → 100100`, `128300 → 100`);
* deduped: **1283 unique steps, 100 → 128 300, strictly ascending**;
* all four run segments consume **24 576 B/step** (b24×g1 equals b8×g3), so one
  byte axis is legal;
* unique bytes = `128300 × 24576 = 3 153 100 800`; the naive sum of the raw
  rows' `bytes_seen` is `2 094 838 579 200` — the field is cumulative, so
  summing rows double-counts by construction. Dedupe removes exactly the extra
  occurrences: the summed-step difference equals the duplicate rows' bytes.

Mixing segments with different token budgets is refused, not added.

## 6. Fixes in this branch

1. `port/src/dormouse/train/loop.py:873` (`_eval_loss`) — a doc-masked eval is weighted by
   scored bytes; flat evaluations are bit-identical to before.
2. `port/src/dormouse/train/loop.py:1390` — a resume whose `steps` ends at or
   before the checkpoint refuses loudly instead of reporting zero steps.

## 7. Tests and how to run them

```bash
cd port
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src <venv>/bin/python -m pytest \
  tests/test_invariants_checkpoint.py tests/test_invariants_resume.py \
  tests/test_invariants_eval.py tests/test_invariants_data.py \
  tests/test_invariants_history.py -q
```

36 new gates in the five files. `test_invariants_history.py` skips (with a reason) when
`/home/sehaxe/logs/flagship.jsonl` is not on the machine; the other files are
self-contained. Runtime on this box: 31 s wall for the five files (36 tests); the full suite is
the long pole.

Full CPU suite: see §8.

## 8. Full-suite result

`PYTHONPATH=src pytest tests -q` on this branch, CPU only: **5 failed, no NEW
failures.** The failures are all explained, and none is a regression:

* three `test_config.py` snapshot tests and
  `test_sft_v2.py::test_the_manifest_accounts_for_every_row` fail in a fresh
  WORKTREE because the untracked fixtures they read (`port/checkpoints/`,
  `port/sft_pairs/v2/full/*.jsonl`) are in the main checkout and not in git.
  Re-run with those fixtures present (symlinked from
  `/home/sehaxe/dormouse/port`): **4 passed**.
* `test_agent.py::test_an_escaping_tool_is_refused_by_screen_code_and_never_runs`
  fails **identically in the clean main checkout at the same commit** (the
  parser says "the byte room ran out mid-call" where the test expects "refused
  by the sandbox screen"). It is pre-existing, and not on the brief's
  known-red list.

The four reds the brief named — `test_export`, `test_fullattn::gate_7`,
`test_loop_blocks`, `test_prelude` — **passed** in this run. The five
`test_invariants_*.py` files pass in the same run and were re-run alone after
the last test edit: **36 passed in 31 s**.

## 9. Design decisions left open

1. **`--set steps=` and the past.** Extending re-plans the decay, which is the
   documented progress-key semantics, and the resumed run matches the stitched
   plan. It cannot match a fresh run of the new length when decay had already
   started; if the owner wants that equality, refuse a `steps` change whose LR
   prefix differs. Today the log's `lr=` column is where a reader sees it.
2. **Masked training loss with `grad_accum > 1`** (see §3): equal weight per
   micro-batch, not per scored byte.
3. **`moe_router_jitter`** (see §1): global-RNG jitter is not resumed; off by
   default.
4. **`layerrope._applications`** resets on resume. It is telemetry (per-forward
   count), never an objective input; kept non-persistent on purpose.
