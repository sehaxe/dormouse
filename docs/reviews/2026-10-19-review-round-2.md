# Review round 2 — the 2026-10-19 fix wave, second read

**Reviewer:** independent, read-only. This file is the only write.
**Reviewed:** `py-main` @ `a10bacc`; the wave after round 1's `70898ed`:
`4f61858`, `a1bb3eb`, `4b3ab54`, `ea2dc53`, `27086e3`, `9f76650`, `98642e6`,
`f964190`, `25e8707`, `63379a5`, `58d1333`, `ee451a5`, `a10bacc`.
**Environment:** the one GPU is held by the live job (`nvidia-smi`: 8 950 MiB,
37-43 %). Every pytest run below used `CUDA_VISIBLE_DEVICES=""`; this is a CPU
review plus reading, and the GPU gates are named where they matter.

## How the verdicts were produced

Named gates first, then a revert harness: `git archive HEAD` into
`/tmp/opencode/rv2`, edit only the copy, run the copy with
`PYTHONPATH=/tmp/opencode/rv2/src` and the repo's `.venv/bin/python`. The repo
itself was never edited (`git status` clean at the end).

```
# the changed gates, current tree
CUDA_VISIBLE_DEVICES="" pytest -q tests/test_mxfp.py::test_the_range_guard_off_cadence_zeroes_and_COUNTS_the_block \
  tests/test_mxfp.py::test_guard_is_cadenced_and_the_forward_takes_no_host_read \
  tests/test_serve.py::test_usage_reports_the_bytes_produced_and_stops_at_nine \
  tests/test_serve.py::test_finish_reason_is_length_when_the_budget_runs_out \
  tests/test_serve.py::test_chat_completion_has_the_openai_shape      -> ..... [100%]

CUDA_VISIBLE_DEVICES="" pytest -q tests/test_train.py tests/test_config.py tests/test_opt_graph.py
  -> .............................................................sss [100%]

CUDA_VISIBLE_DEVICES="" pytest -q tests/test_sft.py   -> ...............................s. [100%]
CUDA_VISIBLE_DEVICES="" pytest -q tests/test_depth.py tests/test_loom.py tests/test_data.py
  -> ..................... [100%]

CUDA_VISIBLE_DEVICES="" pytest -q                    -> [100%]  EXIT=0
.venv/bin/ruff check src tests                       -> All checks passed!
python3 tools/check_doc_refs.py                      -> 0 named paths missing
```

The full suite was green this run, including the timing test round 1 called
flaky (`test_per_byte_cost_does_not_grow_with_context`); that flake is
pre-existing and is not a finding against this wave.

## The 12 changed items

| # | item | verdict | evidence, and what fails on revert |
|---|---|---|---|
| 1 | **N1** captured `why` no longer crashes `opt_graph` | **verified** | `_snapshot_counters` keeps ints only (`graph.py:298-314`); `_replay` also guards `isinstance(n, int)` (`:341-343`). Revert of the snapshot to `dict(counters)` FAILS `test_a_captured_counter_reason_never_enters_the_replay_map` with `{'ns_triton_fallback': 1, 'why': 'triton: input is on cpu'} == {'ns_triton_fallback': 1}`. Residual: the end-to-end 8-step gate would still pass if only the snapshot were reverted and the replay guard kept, so the crash proof rests on the first test. The real CUDA capture was not run (card busy). |
| 2 | **M1** no verifier-scored pair in the SFT corpus; `n_tasks` removed | **corpus fixed, gate BROKEN** | Current corpus is clean: 383 records, 103 distinct answers, zero intersection (type-correct comparison) with the 128 tasks at seed 1. `_examples` has no `build_tasks` (`sft.py:439-451`); old `--n-tasks 8` now dies `No such option: --n-tasks`. But the named gate cannot fail: `_corpus_pairs` returns `bytes` (`tests/test_sft.py:760-771`), `answers` is a `set[bytes]` (`:794`), `scored` is a `set[str]` (`:796`), and `bytes.isdisjoint(str)` is **always True**. Injecting `build_tasks("code", 1, n=1)[0]` into `_examples` makes the text gate fail (`b"def add("` found) while `test_the_sft_corpus_shares_no_solution_with_the_verifier` **passes**. The text gate also omits the third generator kind (`count_vowels`), so the proof is missing on two independent counts. |
| 3 | **M3** nvfp4 zero-block tally reaches the checkpoint line | **verified for pretrain; gap** | `save_ckpt(..., counters=ck_counts)` (`loop.py:1110-1116`) and the line carries `nvfp4_zero_scale_blocks=1`; dropping `counters=ck_counts` FAILS `test_a_tiny_nvfp4_block_is_counted_on_the_checkpoint_line` with a bare `ckpt step=1 -> ...`. `sft.py:1031` and `post/rl_train.py:380` still call `save_ckpt(run, model, opt, step)` with no counters, so an nvfp4 SFT/RL save remains silent. |
| 4 | **M4** off-cadence out-of-range MX block zeroed **AND counted**; on-cadence LOUD | **zeroing verified; COUNTED not delivered** | Reverting the off-cadence branch to `drop = zero` FAILS `test_the_range_guard_off_cadence_zeroes_and_COUNTS_the_block` (`assert [248] == [0]`). But `range_rejected_blocks()` (`mxfp.py:118-125`) is called **only** by `tests/test_mxfp.py:481`; no train/eval/serve line reads it. The doc comment's "COUNTED, never silent" is false under ADR-0011's own test (a reader of the training log cannot see it). This is the same shape R1-M3 found. |
| 5 | **M5** only whole eval windows scored, leftover named | **verified; new regression below** | Reverting `per_window` to `batch*seq_len` and `max(1, ...)` FAILS four gates: `test_only_whole_eval_windows_are_scored_and_the_leftover_is_named`, `test_eval_line_prints_the_byte_window`, `test_depth::test_eval_depths_returns_the_curve` (`assert False`), `test_loom::test_depth_curve_is_wired_with_loom_on` (`384 == 256`). With the fix the 384 B fixture holds two 130 B reads and scores 256 B, as the updated tests assert. The arithmetic is right; the **tail sizing** was not updated (regression R2-1). |
| 6 | **N2** serve reports produced bytes | **verified non-stream; stream still wrong** | Reverting usage/finish to `n`/`"stop"` FAILS `test_usage_reports_the_bytes_produced_and_stops_at_nine`, `test_finish_reason_is_length_when_the_budget_runs_out` and `test_chat_completion_has_the_openai_shape`. `_generate`'s `emitted` accounting (`serve.py:229-284`) is right for the non-stream path. `_stream` still ends with `event({}, "stop")` unconditionally (`serve.py:429`) and sends no usage, so a cap-truncated stream says `stop` — the N2 defect on the other half of the endpoint. |
| 7 | **M7** missing `.lfb` LOUD when controller on, COUNTED when off | **verified behavior; regression R2-2** | Reverting `_restore_lfb` to `if not path.is_file(): return` FAILS both `test_a_missing_lfb_sidecar_is_loud_when_the_controller_is_on` and `..._is_counted_when_the_controller_is_off`. But the LOUD branch fires for **every reader**, not just resume: `onesmoke`, `real1500`, `probe_noengram` (all `moe_lb_coef=0.01`, no `.lfb`) are refused by `load_model_state`; 8 of 9 snapshots in `port/checkpoints/real_small` carry 0.01 and no sidecar exists anywhere under `dormouse/`. The escape is not reachable (R2-2). |
| 8 | **M6** compute knobs only on training commands | **verified for the CLI** | `load_run(..., apply_compute=True)` in `train`/`run`/`sft`/`rl` (`cli.py:265-269, 327-331, 469-473, 575-578`); default `False` (`config.py:1392`). Reverting the guard to always-apply FAILS `test_mm_dtype_and_kda_recompute_are_live_on_sft_and_rl` (`('fp8', True) == ('bf16', False)`). `train()` still applies for direct callers (`loop.py:666-667`); direct callers of `sft_train`/`rl_train` bypass the setters, as the ledger admits. `serve`/`bench` now run the process default (bf16) while a snapshot can say `mm_dtype=fp8`, and no line says so — a COUNTED gap, not a wrong number. |
| 9 | **(c)** SFT rewind gate can fail | **verified** | Removing `eval_stream.rewind()` (`sft.py:702`) FAILS `test_two_sft_evals_on_one_checkpoint_score_the_same_window` (`6.3099 != 6.3171`); the holdout is now 12 records = three 2×64 evals, and the monkeypatched no-op second half asserts the difference. |
| 10 | **(d)** SFT mask/byte gate on CUDA | **claimed-but-not-proved here** | The gate is `CUDA`-marked (`tests/test_sft.py:44-47`) and is the one `s` in the suite under `CUDA_VISIBLE_DEVICES=""`; the card is busy, so no `set_sync_debug_mode` run. Reading: the pre-fix `ex.to(device)` before `sft_mask(ex)` would make the spy see `cuda`, so the first assertion would fail on a working card; the `<=1` sync bound is unverified. The skip condition is `torch.cuda.is_available()`, not "card free", so a normal `pytest` on this box would run it against the live job. |
| 11 | **ledger edits** | **partly false** | R1-N1/M5/M6/N2 rows are accurate. R1-M1/C2 (`ledger:53`) rests on the vacuous type-mismatched gate; R1-M4 (`:75`) rests on a tally only the test reads; R1-M7 (`:78`) calls the sidecar "partly fixed" but omits that every pre-sidecar checkpoint is now unloadable by every reader. R1-M2/M8 are honestly open. |
| 12 | **eval fixtures** (`ee451a5`, depth/loom 3→2) | **arithmetic verified, but encodes R2-1** | The 384 B fixture does hold only two whole 130 B reads; the updated gates pass and the old denominator fails them. The same fixture is the no-`--eval` tail, however, and the tail is still sized as *scored* bytes, so the default 20-window tail holds 19 (R2-1). |

## Regressions introduced by this wave

**R2-1 (MAJOR) — the no-`--eval` tail now scores one window fewer than it
asks for, on every default run.** `_eval_batches` divides by
`batch × (seq_len + 1)` (`loop.py:456`), but `open_streams` still sizes the tail
as `eval_window_bytes = eval_batches × batch × seq_len` (`data.py:454`). With
the default `20 × 8 × 512` the tail is 81 920 B and one read is 4 104 B, so only
19 whole windows fit and 77 824 B are scored:

```
file 256000 requested window 81920 eval span 81920
whole windows 19 scored 77824 leftover 4096
one read 4104 needed for 20: 82080
```

The line names the shortfall, so this is COUNTED, not silent — but the code can
fix it for free (`ev_bytes = eval_batches * batch * (seq_len + 1)`, still capped
at 1 MiB), and until it does every no-`--eval` BPB is over 19/20 of the window
the config and `report["eval_window_bytes"]` claim. The wave updated
`test_depth.py:123` and `test_loom.py:147` from 3 to 2, which normalises the
shortfall instead of sizing the tail. The explicit `--eval` file (500 MB) is
unaffected.

**R2-2 (MAJOR) — the LFB LOUD branch bricks every pre-sidecar checkpoint for
every reader.** `_restore_lfb` (`loop.py:111-135`) refuses a missing sidecar
whenever `model.cfg.moe_lb_coef > 0`, with no `resume` distinction.
`load_model_state` is the ONE loader for `eval` (`cli.py:645`), `bench`
(`:753`), `export` (`:855`), `decode` (`:999`), `serve` (`serve.py:128`),
`--init-from` (`sft.py:845,852`, `rl_train.py:190,194`) and `scripts/`. None of
those commands has a `moe_lb_coef` override (`serve_cmd`/`eval_cmd` have no
`--set`), and a resume that passes `--set moe_lb_coef=0` is refused by
`load_ckpt`'s snapshot gate (`loop.py:404-413`). Measured on the tree's own
artifacts:

```
onesmoke coef 0.01 engram_ram False quant none
  REFUSED: LoudError LFB sidecar checkpoints/real_small/onesmoke.lfb is missing ...
real1500 ... REFUSED
probe_noengram ... REFUSED
```

`find /home/sehaxe/dormouse -name '*.lfb'` returns nothing. The reader cannot
continue training, so a `COUNTED` line is the defensible mark there; a `LOUD`
that only `train --resume` can honour is not. This is the wave's most damaging
operational regression.

**R2-3 (MAJOR) — the M1/C2 gate compares `bytes` to `str` and can never
fail.** Evidence in item 2: an injected `build_tasks("code", 1, n=1)` pair
leaves the "solution-disjointness" test green. The corpus is clean today, but
the ledger's `fixed` verdict for the CRITICAL C2 finding is not supported by
its named gate, and the `def add(`/`def square_sum(` substring check misses
`count_vowels`. Fix: decode both sides to `str` (`answer.decode("utf-8")`) and
compare every generator kind, or compare the set of generated solution bytes in
`_examples` directly.

**R2-4 (MINOR) — nvfp4 checkpoint counting stops at pretrain.** SFT and RL saves
still call `save_ckpt` without a `Counters` (`sft.py:1031`, `rl_train.py:380`),
so their checkpoint lines omit `nvfp4_zero_scale_blocks` even when the storage
rung zeroed blocks. The audit M1 is about the storage rung, not one loop.

**R2-5 (MINOR) — the streaming serve path keeps the N2 lie.**
`serve.py:429` emits `finish_reason: "stop"` after any stream, including one cut
by `max_tokens`; the SSE carries no usage. The non-stream tests cannot see it.

**R2-6 (MINOR) — `report["eval_window_bytes"]` is the requested window, not the
scored one.** With R2-1 it is 81 920 while the eval line says 77 824; the JSON
row and the line disagree by 4 096 B.

## New findings (no round has looked at these)

**N6 (CRITICAL) — the math verifier executes the model's answer in the trainer
process.** `post/verifiers.py:52-56` calls `sympy.sympify(got)` on
`_final_answer` of the model response (`:97-106`); `sympify` evaluates Python.
Reproduced read-only:

```
answer: __import__('os').system('touch /tmp/opencode/sympy_pwned')
  -> sympify -> 0, marker exists: True
answer: open('/tmp/opencode/sympy_pwned2','w').write('x')
  -> sympify -> 1, marker exists: True
```

The `_FORBIDDEN` screen (`:114-125`, `:201-205`) is applied only to
`cls == "code"`; math has no sandbox at all. In `dormouse rl --set
rl_task_cls=math` (a supported class, `config.py:814`) the model can run any
Python as the training user, kill or hang the run, or write files; it also
makes the reward non-deterministic. Fix: parse with a non-evaluating grammar
(e.g. `sympy.parsing.sympy_parser.parse_expr(..., evaluate=False)` with an
allow-listed `local_dict`, or a numeric `Fraction`/`ast` parser), run it under
the code jail, and add the timeout R1-N5 already asks for.

**N7 (MAJOR) — the code "sandbox" is not a jail.** The preamble patches only
`builtins.open` (`verifiers.py:130-143`); `os.open` and `pathlib` bypass it.
Reproduced:

```
import os
fd = os.open("/tmp/opencode/pwned3", os.O_WRONLY | os.O_CREAT, 0o644)
os.write(fd, b"os.open bypassed the preamble"); os.close(fd)
def add(a, b): return a + b
-> verdict True, all 1 tests passed, marker written
```

`os.kill` is not blocked (the static screen has no `kill`, and `RLIMIT_NPROC=0`
blocks forks, not signals), so the response can signal the parent trainer;
`preexec_fn=_sandbox_limits` (`:181-198`) runs in the forked child of a
multi-threaded torch process, which is the classic deadlock shape. The module
docstring sells "a process jail, rlimits"; rlimits are there, the jail is not.
This is partly known (R1-N5 named `pathlib`/`io.open`), but the write/`os.kill`
class and the `preexec_fn` hazard were not in the ledger.

**N8 (MINOR) — the export reader accepts a header it cannot load and raises raw
exceptions.** `_check_layout` (`export.py:722-760`) never requires `tensors`;
`read_export` accepts a header with the key removed, then `load_export`/`names`
raise `KeyError: 'tensors'` (`:659`). A record with `shape=[10**9]` passes the
layout check and fails with `RuntimeError: shape '[1000000000]' is invalid for
input of size 196608` from `reshape`. `scales_offset`/`scales_length` for
`mxfp4` are never range- or alignment-checked. A corrupt file should be the
LOUD message with the escape the docstring promises, not a traceback.

**N9 (MINOR) — `spec_over_window` undercounts the first over-window round.**
`specdecode.py:246-249` counts `len(ext) > window + len(proposals)`, which is
`len(ctx) > window`; the verify input is already `len(ctx) + k` bytes. At a
full window (`ctx == max_seq_len`) the verify forward sees `max_seq_len + k`
bytes and the counter stays 0 for that round. Measured with the tiny model:
`max_seq_len=64`, prompt 64 B, `spec_k=2`, 3 rounds → `spec_over_window=2`
(one round uncounted). The module's "COUNTED, never silent" claim is false at
the boundary. Fix: count `len(ext) > window`.

**N10 (MINOR) — the review-loop document still says round 1 is pending.**
`docs/reviews/2026-10-19-review-loop.md:46-54` has "Round 1 — pending" and
"Round 2 — pending", while round 1 is in the tree, the ledger carries its rows,
and this file is round 2. The protocol says the score and round live there; the
document is the one place a newcomer reads first.

**Unchanged open findings that still matter** (not re-litigated): R1-M2 — the
SFT eval loops `cfg.eval_batches` with no cap (`sft.py:693-717`); measured on a
12-record, 780 B holdout it reported `40 records / 2560 B` (the ring wrapped
3.3×). R1-M8 — that eval still builds `sft_mask` after `.to(device)`
(`sft.py:711-712`). R1-M9 — `Counters.merge` is still a substring test
(`instrument.py:144-146`). R1-N3 (Content-Length) and R1-N5 (sympy timeout)
are open; N6 makes the latter urgent.

## Score

**6/10** (round 1: 7; round 0: code 6, infra 5).

Eight of the ten code gates fail on revert (N1, M3-pretrain, M4-zero, M5, N2,
M7, M6, c) and the SFT corpus is genuinely clean, so the instrument is stronger
than round 1 found. It is not worth 7: the CRITICAL C2 proof is a gate that
cannot fail (bytes vs str), M4's "COUNTED" is still a test-only tally, the M5
fix shrinks the default no-`--eval` window from 20 to 19, the M7 fix makes every
pre-sidecar checkpoint unloadable by `serve`/`eval`/`bench`/`export`, and the
reward path contains a CRITICAL RCE (`sympify`) plus a bypassable code sandbox.
The post-training lane is where the project says it is going; a verifier that
runs the model's answer in the trainer process is a security boundary failure,
not a paper cut.

The two findings that would move the score most:

1. **N6, the `sympify` RCE** — it turns every math rollout into arbitrary code
   execution in the training process, and it is reachable from a supported
   command. Fix the parser and the jail, add the timeout, and add a gate that a
   response containing `__import__`/`open` cannot reach the parent.
2. **R2-2, the LFB reader refusal** — it is the difference between "the
   artifacts are usable" and "the tree cannot serve its own history"; splitting
   LOUD-on-resume from COUNTED-on-read (or adding the named override to the
   readers) restores the artifacts without hiding a controller reset.

(Runner-up: R2-3, the M1 gate; it is a proof failure on the round-0 CRITICAL,
which is why the ledger's "fixed" is not yet believable.)

## What I could not check

* Every CUDA path: the card is held by the live job. That includes the real
  `opt_graph` capture/replay (N1 was proved on `_snapshot_counters`/`_replay`
  by construction and a CPU subclass, not on hardware), the FLA/triton/fp8
  kernels, and the GPU-only tests, including the new (d) sync-debug gate.
* The live `small_wide` run and any training on the on-disk artifacts. I loaded
  three `port/checkpoints/real_small` checkpoints read-only on CPU to prove
  R2-2; I did not resume, serve or train any of them.
* CI (cannot trigger), the docs-site build/render, and the real 46 GB corpus.
* `selfplay/`, `train/distributed.py`, `bench/score.py`'s protocol against
  lm-eval on real task files, and `model/export.py`'s packing arithmetic beyond
  the hostile-header probes in N8.
* The (d) gate's `<=1` sync bound, and whether its `torch.cuda.is_available()`
  skip is enough on a box where the card is busy but present (I12 is still
  queued).
* The frequency with which the M4 off-cadence path is hit in a real QAT
  forward; the code path and the tally are confirmed, the rate is not.
