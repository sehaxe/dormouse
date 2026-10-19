# Code audit — the Python tree, 2026-10-19

**Audited:** `main` @ `7f0b656`, read-only. Scope: `train/loop.py` + `train/sft.py`,
`model/*.py`, `post/*.py`, `bench/*.py`, `tests/`, and the docs-vs-code seam.
**Standard:** this repo's own `AGENTS.md` §1 — LOUD / COUNTED / SILENT
(SILENT = defect), A/B or death, zero host-device sync in the hot path, claims
must name their evidence.

**Environment, and how it bounds the audit.** `/home/sehaxe/dormouse/port` is
running a live training job on the one GPU. `uv run pytest -q` in this tree is
then **red — 127 tests (117 FAILED + 10 ERROR)**, every one of them carrying

```
torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
```

whose root cause is not this repo's code but torch 2.13's
`AdamW.step` → `_accelerator_graph_capture_health_check` →
`torch.accelerator.current_stream()` (an unconditional CUDA API call even for a
CPU tensor). With `CUDA_VISIBLE_DEVICES=""` the same command **exits 0** over
787 tests. So every GPU gate in the suite (triton MoE, fp8 `_scaled_mm`, FLA KDA,
`opt_graph` replay, the fused kernels) is **unverified here**; the findings below
are all reachable from CPU code and were reproduced by hand.

---

## CRITICAL

### C1. The trainer never seeds the model initialisation; `train.seed` is not a property of a run

`cli.py:258` (`model = _model(run).to(dev)`, same shape at `:320`, `:497`, `:618`,
`:730`, `:972`) → `cli.py:121-122` → `DormouseModel.__init__`
(`model/model.py:46-66`), which draws every parameter from the **global torch
RNG**. `torch.manual_seed` (the global stream) is called in `src/` only at
`cli.py:858` (an export verification); every other `manual_seed` in the tree
(`tsct.py:327`, `post/rl_train.py:100`, `serve.py:216`, `selfplay/loop.py:292`)
seeds a local `Generator`. `train()`, `sft_train()` and `rl_train()` never seed.

Evidence (two processes, one preset, `cfg_seed = 1`):

```
cfg_seed= 1 init_hash= 39cb6df02017d9de   # process 1
cfg_seed= 1 init_hash= c70cb36be3c0ab42   # process 2
```

**What breaks.** (a) §1.2's instrument — "3 seeds per arm" — is not
implementable as documented: `--set seed=2` moves the JEPA mask, the rand-depth
schedule and the RL rollout generator, and **nothing about the weights**; two
runs labelled `seed=1` are as different from each other as `seed=1` from
`seed=2`, and neither can be re-run. (b) The `seed` key is in the snapshot
(`config.py:696`) and in every `.dormouse` header (`model/export.py:588`,
`cli.py:843`) as provenance, so a stored artifact's seed does not describe its
weights. (c) `README.md:48` ("Every mechanism in the tree was admitted by a
3-seed A/B on a fixed protocol") and
`docs/reviews/2026-10-19-model-scorecard.md:184` claim an instrument this tree
cannot reproduce; `/home/sehaxe/dormouse/port` has the same gap, so the arms on
record inherit it.

**Smallest honest fix.** One call before the model exists:
`torch.manual_seed(run.train.seed)` inside `cli._model` (and in `rl_train`,
which builds its own model at `post/rl_train.py:143`). `config.seed`'s docstring
(`config.py:696`) then has a mechanism behind it, and the existing
`tests/test_rl_train.py::test_rl_train_is_deterministic_for_one_seed` starts
testing the run instead of the caller's own seeding.

### C2. The SFT corpus contains the exact tasks the RL lane and the one-button pipeline reward

`train/sft.py:435-442` builds the SFT corpus from
`build_tasks("code", seed, n=n_tasks)` / `build_tasks("math", seed, n=n_tasks)`,
using `task.prompt` as the instruction and **`task.solution` as the answer**.
`cli.py:342` (`dormouse run` stage 2) and the `rl` command then score
`build_tasks(post_cls, seed=run.train.seed, n=8)`. `build_tasks(cls, seed, i)`
is a pure function of `(cls, seed, index)` (`post/tasks.py:70-80`), so:

```
sft tids (n=32): code-1-0 … code-1-7 …      # identical prompts
rl  tids (n=8) : code-1-0 … code-1-7        # identical hidden solutions
rl tasks are a PREFIX of the sft tasks: True
prompt identical: True | solution identical: True
```

**What breaks.** `dormouse run`'s `pass@budget` and `dormouse rl`'s reward are
measured on the first 8 tasks the model was *taught the answers to*, and the
leak is invisible in both logs (the rollout line has no "seen in SFT" field).
This is the `7adda92` class — a number that measures a different thing than it
names — one layer up.

**Smallest honest fix.** Draw the post-training set from a disjoint range, e.g.
`build_tasks(cls, seed, n=8 + 64)[8:]` or `seed + 1_000_003`, and make the SFT
build refuse (LOUD) an overlap between the corpus's task ids and the RL set's.

---

## MAJOR

### M1. The `nvfp4` storage rung silently zeroes small blocks — and the mechanism that should prevent it is off by 2^442

`model/tsct.py:630`:

```python
min_normal = float(2.0 ** torch.finfo(torch.float8_e4m3fn).min)  # 2^-6
```

`torch.finfo(float8_e4m3fn).min` is **-448.0** (the most negative *value*, not the
minimum exponent), so `min_normal = 2^-448 = 1.376e-135`, not the `2^-6` the
comment and the module docstring (`tsct.py:615-628`) name. The global prescale
therefore fires only for block amaxes below ~1e-134: **effectively never**. A
block whose amax is small but nonzero then has `scale = amax/6 < e4m3's tiny`, so
`s_fp8` rounds to 0 (`tsct.py:647`), `q = blocks/0 = ±inf` saturates the
nearest-grid search, and the decode multiplies by 0 (`tsct.py:666`). Measured:

```
pack_tsct(t, 'nvfp4') on a [64,32] fp32 master:
  global_scale = 1.0
  block with amax 1e-3  -> out amax 0.0     (all 64 elements destroyed)
  block with amax 1e-1  -> out amax 0.105
```

**What breaks.** `--set quant=nvfp4` (a config `validate` accepts, a rung
`pack_tsct` writes) destroys the small tail of every factor on save; the loss is
inside "the format's own grid" only for the large blocks. No counter, no line.

**Smallest honest fix.** `min_normal = float(torch.finfo(torch.float8_e4m3fn).tiny)`
(= 2^-9, the value the prescale actually needs) or the literal `2.0**-6`; plus a
COUNTED counter for blocks whose e4m3 scale rounds to zero, and a test that packs
a block with amax 1e-3. `tests/test_ladder_storage.py::test_nvfp4_round_trip_within_the_grid_bound`
cannot see it: its fixture is `randn(64,4096) * 0.05` (block amaxes ≈ 0.15, far
above e4m3's `tiny·6 ≈ 0.0117`), so the tiny-block path is never exercised — its
grid×scale invariant would in fact catch the zeroed block as a NaN, on a fixture
that reaches that range.

### M2. `counters.why` — the reason a fallback ran — never reaches any log

Eleven call sites attach the reason to the counter object (`model/experts.py:844,
859, 865, 911, 924, 935`, `model/tsct.py:209`, `model/kda.py:285, 350`,
`model/decode.py:341`, `train/graph.py:336`, `train/ns_kernels.py:374`). The log
line is built from `Counters.line()` (`instrument.py:103-105`), which prints
`sorted(self)` — the dict **keys only**; `why` is a plain attribute. The JSON row
does `counters=dict(window)` (`train/loop.py:896`), which drops it too.

**What breaks.** ADR-0019 and every one of those docstrings claim "a reader of
the training log sees it without a profiler" / "the reason on `counters.why`".
A reader sees `moe_triton_fallback=1234` and cannot learn *why* the fused arm was
skipped; the tests assert `counters.why` in-process (`test_experts_gpu.py:98`,
`test_moe_dropless.py:287`, `test_mm_fp8.py:178` …), which is exactly why the
missing print went unnoticed.

**Smallest honest fix.** One line in `Counters.line()`: append
`why=<getattr(self, "why", "")>` when non-empty, and include it in the JSON row.

### M3. The SFT eval never rewinds its stream

`train/sft.py:939-968` calls `eval_stream.next_batch()` in a loop with **no
`rewind()`** — unlike `train/loop.py:342`, whose docstring records the burn
lesson ("the same checkpoint scored 6.443 and 6.551 on consecutive evals from
stream position alone"). `grep -n rewind train/sft.py` is empty.

**What breaks.** SFT's `ce=`/`first=` curve is a moving-window measurement:
eval N and eval N+1 score different bytes, so a "the SFT CE fell" reading is not
a statement about the weights alone, and two SFT runs' curves are not comparable.
It is the exact defect the pretrain lane fixed on 2026-09-27, reintroduced in the
copy of that loop.

**Smallest honest fix.** `eval_stream.rewind()` before the eval loop, and print
the scored bytes (the pretrain line's `over N B` field) so the window is on
record.

### M4. `mm_dtype` and `kda_disable_recompute` are inert on the SFT and RL paths

The two process-wide setters are called only from `train/loop.py:489-490`:

```python
set_mm_dtype(cfg.mm_dtype)              # model/tsct.py:98
kda.set_disable_recompute(cfg.kda_disable_recompute)   # model/kda.py:101
```

`sft_train` (`train/sft.py:702-704`) and `rl_train` (`post/rl_train.py:140-141`)
set only `torch.set_float32_matmul_precision`. Both fields are `TrainConfig`
keys, both are in the snapshot, both are documented as knobs
(`config.py:482-498`) — and on those two commands they reach no code.

**What breaks.** A `--set mm_dtype=fp8` SFT run reports fp8 in its snapshot and
computes bf16 (or whatever the process default is); the log line's
`mm_fp8_fallback`/`mm_fp8_gemm` counters stay 0 and read as "the lane was not
needed" rather than "the lane was never asked". The repo's own phrase for this is
"a config field that lies".

**Smallest honest fix.** Call the two setters in `sft_train` and `rl_train`, or
refuse those fields on those commands in `config.validate`.

### M5. The no-`--eval` tail path double-counts bytes when the eval window exceeds 1 MB

`train/data.py:454-461`: `ev_bytes = min(max(eval_window_bytes, 1), 1 << 20)` and
the eval stream is `[hi_train, size)`, but the eval line prints
`eval_batches × batch × seq_len` (`loop.py:446-450`) — the *requested* window,
not the bytes actually read. With `eval_batches=210, batch=10, seq_len=512` the
line claims 1 075 200 B over a 1 048 576 B tail and the ring wraps:

```
window B: 1075200 | eval tail: 1048576 B
batches read: 210 | the eval stream wrapped back to its start? True
```

**What breaks.** The `over N B` field is the authority this repo tells readers to
quote; here it overstates the scored text by ~3 % and re-scores the tail's first
~29 KB. (The `--eval <file>` path sizes its ring as `max(1MB, window)` and does
not wrap — the two paths disagree.)

**Smallest honest fix.** `ev_bytes = max(eval_window_bytes(tr), 1 << 20)` in the
tail branch, or LOUD when the window exceeds the tail; and derive the printed
byte count from `scored` (it already is returned) rather than from the config.

### M6. The CPU gate cannot run on this box while a training job holds the card

`uv run pytest -q` → exit 1, 80 × `AcceleratorError: CUDA-capable device(s)
is/are busy or unavailable` (traceback in `torch/optim/adam.py:222` →
`torch/accelerator/__init__.py:232`), including tests that explicitly pass
`--device cpu` (`tests/test_cli.py:98`, `test_sft.py`, `test_train.py`,
`test_resume.py`, `test_selfcheck`…). With `CUDA_VISIBLE_DEVICES=""` the same
suite **exits 0**.

**What breaks.** `conftest.py:1-6` says "Everything here is CPU and tiny on
purpose … none of those need a GPU", and §1.5's workflow is "one heavy thing at a
time" — but the moment the heavy thing is running, the repo's own definition of
done ("full tests green") is unreachable, and the failure mode looks like 80 code
failures. Nothing in the suite or the docs names the escape.

**Smallest honest fix.** Document it where the suite is run (README quick start /
`AGENTS.md` §1.5): `CUDA_VISIBLE_DEVICES="" uv run pytest`, and/or set it in
`conftest.py` for the CPU-marked tests.

### M7. Device syncs inside the forward: the quantizers, the SFT step, the naive MoE arm

§1.3 is "no host-side branch on a device value in a step", and `loop.py:776-782`
records the fix that made the pretrain step **one** host read. Reintroduced:

* `model/act_quant.py:240` — `if not bool(torch.isfinite(x).all())` in `_check`,
  called by every `fake_quant_act` → one D2H per MoE forward (plus one per
  expert on the naive/dense paths, `experts.py:274`), i.e. `max_iter` syncs per
  step with `act_quant` on;
* `model/mxfp.py:185` (`_check_finite`), `:141` (the e8m0 range test), `:165`
  (the NaN-scale refusal), `:222`/`:275` (`assert bool(...all())`) — 2-3 syncs
  per packed tensor, called per factor per forward under
  `factor_qat=mxfp4`/`dense_qat=mxfp4`;
* `train/sft.py:847` — `n_mask = int(mask.sum())` **every step**, a second read
  on a device tensor that the pretrain loop's audit deliberately merged away;
  the mask itself is computed after `x.to(device)` (`sft.py:838-840`) and its
  per-row padding check (`sft_mask`, `sft.py:592-598`) loops the batch with a
  `bool(...)` per row — `b` syncs per step;
* `model/experts.py:753` — `torch.unique(sorted_ids[keep]).tolist()` on the
  naive arm (the COUNTED CUDA fallback), a D2H per routed forward.

`model/act_ckpt.py:117-123` already names this class for the MX codec ("2-3
device syncs per packed tensor … the shape §1.3 forbids") and defers the fix;
the act-quant/SFT copies are not even listed.

**Smallest honest fix.** Device-side finite flags accumulated by
`Counters.count_device` and drained on the log cadence (the pattern
`instrument.py:75-101` already provides), and in SFT: compute `mask` before the
`.to(device)` transfer and count `sft_bytes` from the host-side `mask.sum()`.

### M8. `lfb_bias` is not persisted while the integrator it is derived from is

`model/experts.py:378-386`: `lfb_counts` is `persistent=True` with the comment
"a resume must keep its controller state or the rebalancer restarts its memory of
where the load went"; `lfb_bias` is `persistent=False`. A resume restores the
accumulated counts and resets the bias to 0, so the first post-resume step moves
every expert's bias by `±coef` from a state the interrupted run had already
integrated — a different routing controller under the same step count.

**Smallest honest fix.** `persistent=True` on `lfb_bias` (both are buffers, both
already ride `state_dict`), or persist neither and say why in the docstring.

---

## MINOR

1. `train/loop.py:934-940` — the final eval (`steps % eval_every != 0`) is logged
   but not written to the JSON sink (the in-loop one is, `:921-922`), so
   `dormouse plot` cannot see the last eval.
2. `train/loop.py:176` — the config snapshot is a plain `write_text` after the
   atomic weight replace; a crash mid-write leaves a truncated snapshot and a
   resume that fails at `RunConfig.from_snapshot`. The `.ngram` write is atomic
   (`:173-175`) and the snapshot is not.
3. `post/verifiers.py:136-142` — the sandbox's absolute-path refusal patches
   `builtins.open` only; `pathlib.Path.read_text` / `io.open` bypass it, and
   `os.system` reached through `getattr` bypasses the static screen (`:114-125`).
   `_verify_math` (`:52-56`) runs `sympy.sympify`/`simplify` with no timeout, so
   one generated expression can hang the reward loop.
4. `model/tsct.py:123` `mm_fp8_fallbacks()` — the "unconditional" tally behind
   the fp8 lane's COUNTED claim is read by no production code (tests only), so
   the guarantee exists only in the docstring.
5. `train/loop.py:866` — on a window whose every step was non-finite the log line
   prints `loss=<last finite loss>`; the same line carries `nan_skipped`, so it
   is COUNTED, but the number reads as the window's loss.
6. `post/rl_train.py:316-323` — a step where every sample was skipped logs
   `loss=0.0000` (the grad-carrying zero), where `bytes/solved=-` is used for the
   same "not a number" case; `rl_samples=0` is on the line.
7. `tools/check_doc_refs.py` exits 0 with **260 "known-dead" references**, and
   `docs/glossary.md:697+`'s 23-row code-vs-document table (plus §6 and the
   `### seed` entry, `:567-580`) names `crates/…`, `vendor/…`,
   `benches/history.tsv`, `configs/nano-fused.toml`, `scripts/bench.sh` — none of
   which exist in this tree, whose `AGENTS.md:15` says `main` is the Python tree.
   The disagreement list is the Rust lane's, kept current only by the allowlist;
   the Python lane's own disagreements (C1, M4 above) are not in it.
8. `bench/tasks.py:238` — `load_task("synth_math")` hardcodes `seed=0`, so a
   `--set seed=N` run's MC harness is the same task set for every seed; the file
   is deterministic (good) but the seed's meaning differs between the two
   commands.
9. **The repo's own doc gate is RED at HEAD**:
   `python3 tools/check_doc_refs.py` exits 1 —
   `docs/reviews/2026-10-19-findings-ledger.md:51 -> docs/reviews/2026-10-19-review-loop.md`
   (the named file does not exist). §1.8 requires that gate green before a commit
   lands, and this reference is in a committed file. Fix: write the file the
   ledger promises, or point the line at whatever carries the round/score.

---

## Gates: the strongest and the weakest

**Ten most load-bearing** (each one line; the class it alone catches):

1. `tests/test_bank_invalidation.py` — a parameter whose *value* moves while
   `_version` does not (the `opt_graph` bank-cache defect).
2. `tests/test_train.py::test_two_all_nan_windows_stop_the_run` (`:152`) — a run
   that is entirely non-finite must stop, not exit 0.
3. `tests/test_resume.py` — the snapshot drift refusal and the optimizer
   round-trip (ADR-0021).
4. `tests/test_shard.py::test_a_resumed_run_reads_the_same_batches` —
   `skip_bytes` positioning across shard and epoch boundaries.
5. `tests/test_engram_host.py` (`:156-179`) — the eval line's
   `engram=<rows>/<arms>` field, i.e. the §3.2 retracted-arm class.
6. `tests/test_moe_dropless.py` — drop parity across the dispatch arms, with the
   arm counter asserted.
7. `tests/test_offload_experts.py` — bit-parity of the host offload **plus** a spy
   proving the pack hook fired (the silently-inert-hook class).
8. `tests/test_counters.py::test_count_does_not_synchronize_a_cuda_tensor` — the
   only gate that can see a per-pass D2H (§1.3), via torch's sync-debug mode.
9. `tests/test_ladder_storage.py` — one round-trip per storage rung plus the
   refusal list.
10. `tests/test_sft.py` — the SFT mask's marker/padding refusals (the corpus
    contract that makes the prompt-masked CE mean what it says).

**Ten weakest** (and why):

1. `tests/test_counters.py::test_a_positive_int_still_accumulates` — asserts `+`
   on the class under test.
2. `tests/test_muon_variants.py::test_the_default_path_is_bit_identical_to_the_pre_change_tree`
   — the strongest optimizer regression guard in the tree, `@local_pin`, skipped
   under CI (`conftest.py:23-34`).
3. Every other `@local_pin` test (`tests/test_mm_fp8.py:63`, …) — same reason:
   the float-hash pins run only on the author's box.
4. Every `@CUDA`/`pytestmark = skipif` GPU gate (`test_moe_kernel.py:52`,
   `test_moe_dropless.py:40`, `test_ns_triton.py`, `test_kda_gpu.py`,
   `test_experts_gpu.py`, …) — on a CPU run, or on this box, the triton/dropless/
   fp8/KDA arms have no gate at all.
5. `tests/test_moe_dropless.py:287` — asserts the fallback's *reason string*
   ("input is on cpu"), not the fallback's arithmetic.
6. `tests/test_train.py::test_eval_line_prints_the_byte_window` — asserts the
   printed formula; nothing checks that the bytes read are distinct (M5 has no
   gate).
7. `tests/test_sft.py::test_sft_eval_reports_first_byte_agreement` — asserts the
   metric's shape; nothing checks two consecutive evals score the same window
   (M3 has no gate).
8. `tests/test_ladder_storage.py::test_nvfp4_round_trip_within_the_grid_bound` —
   a whole-tensor relative tolerance, blind to a zeroed block (M1).
9. `tests/test_mxfp.py`'s reference-table tests — the codec is pinned against a
   table written in the same file; a wrong pair agrees with itself. (I checked
   the e2m1 tie ladder by hand against the module's own documented table and it
   is right, so this is a shape problem, not a wrong number.)
10. `tests/test_plot.py` — pins the plot's strings/exit codes, so any formatting
    drift is a failure while a wrong *number* in the sink is not.

**Mechanisms with no gate at all** (verified by grep + reading): the seed→init
link (C1 — no mechanism); the SFT↔RL task overlap (C2); the SFT eval window (M3);
the `nvfp4` tiny-block zeroing (M1); `counters.why` reaching a log line (M2); the
`mm_dtype`/`kda_disable_recompute` wiring on the SFT/RL paths (M4); the
`eval_batches × batch × seq_len > 1 MB` tail wrap (M5); `lfb_bias` across a
resume (M8).

---

## What is genuinely good (specific, with evidence)

* **The eval instrument's separation.** `_eval_loss` rewinds first
  (`loop.py:342`), scores a `Counters()` of its own (`:350`, `:376-377`), and
  prints the byte window and the eval's own `engram=<rows>/<arms>`
  (`:446-450`) — the two fields that would have caught the §3.2 retraction.
* **The NaN firewall is in all three loops and is gated.** Skip + count + the
  two-consecutive-window `LoudError` (`loop.py:832-909`, `sft.py:883-937`,
  `rl_train.py:309-377`), with a test that drives it red
  (`test_train.py:152`, `test_rl_train.py:227`), and the skipped step also drops
  the host leaves (`loop.py:850-851`) so a NaN flood cannot grow `_pending`.
* **The MoE dispatch seam is one permute for every arm** with a counter per arm
  and a LOUD refusal where a shape cannot exist (`experts.py:796-939`,
  `_padded_dispatch`'s capacity-0 refusal `:695-703`); the drop counter is kept
  on-device (`:537`) precisely so the arm choice cannot cost a sync.
* **The bank cache's key is defended by a measurement, not a guess**
  (`experts.py:556-572`: `_version` frozen under graph replay, the
  requires-grad clause from the 20/20 dead-adjoint battery) and there is a gate
  (`test_bank_invalidation.py`).
* **The host-table sidecar contract is honest.** Missing sibling is LOUD
  (`loop.py:217-237`), the write is atomic (`:173-175`), the load refuses a shape
  mismatch (`engram.py:446-473`), and the docstring names what is *not* there
  (no `train_step` stamp).
* **`Counters.count`'s order is load-bearing and commented as such**
  (`instrument.py:62-73`), with the sync-debug gate that fails on the old order
  (`test_counters.py:41+`).
* **The verifier cannot read the answer it judges** — `Task.as_dict()` drops
  `solution` (`post/tasks.py:57-67`) — and the code verifier screens before it
  executes (`verifiers.py:201-214`), so the two layers are named in the code.
* **The offline-checkable quantizers are documented to their source**: the e2m1
  tie ladder in `act_quant._e2m1_round` matches its own table entry for entry
  (checked by hand against `act_quant.py:258-263`), and `mxfp.py`'s RNE follows
  the grid's mantissa parity (`mxfp.py:211-218`) — the class of defect the fp4
  retraction came from is visibly guarded here.

## What I could not check

* **Every CUDA path.** The card is held by the live job in
  `/home/sehaxe/dormouse/port`; `cudaErrorDevicesUnavailable` kills the GPU gates
  and any hand-run of them. That covers `moe_kernels.py`'s Triton kernels
  (1179 lines, unaudited), the fused KDA/adjoint, the fp8 `_scaled_mm` lane on
  hardware, `opt_graph`'s capture/replay, and `test_compile_moe_gpu.py`.
* **The real corpus and the training artifacts.** I read no run logs, no
  `checkpoints/` snapshot and no `.dormouse` file; the numbers I quote are either
  reproduced in-process or quoted from the repo's own documents with their
  source.
* **`model/export.py`'s arithmetic** — I verified its refusal surface and its
  layout documentation, not its packing paths.
* **`selfplay/*.py`** (reward/pool/utm) beyond its test file existing.
* **`docs-site/`, `scripts/`, `serve.py`** — out of scope this pass.
* Whether the b1 recipe's memory math holds on the card; `docs/decisions/2026-10-15-1b-readiness-plan.md`'s
  numbers were not re-derived.
