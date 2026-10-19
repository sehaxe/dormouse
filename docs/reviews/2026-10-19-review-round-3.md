# Review round 3 — the last wave of the 2026-10-19 loop

**Reviewer:** independent, read-only. This file is the only write.
**Reviewed:** `py-main` @ `d5b028a`; the wave after round 2's `a10bacc`:
`ddc86ba`, `4a27ba4`, `952b53a`, `8a8650f`, `6b2056e`, `93a81c3`, `ac18f3a`,
`d5b028a`.
**Environment:** the one GPU is held by the live job (`nvidia-smi`: 8 951 MiB,
64 %). Every pytest run below used `CUDA_VISIBLE_DEVICES=""`, so this is a CPU
review plus reading. `/home/sehaxe/dormouse/port` (the live checkout and its
artifacts) was read, never written. `git status` in the reviewed tree is clean
apart from this file.

## How the verdicts were produced

Named gates first, then a revert harness: `git archive HEAD` into
`/tmp/opencode/rv3{,b,c,d,e,f}`, edit only the copy, run the copy with
`PYTHONPATH=/tmp/opencode/rv3/src` and the repo's `.venv/bin/python`. The repo
itself was never edited. The security gates were additionally probed with
payloads of my own (`/tmp/opencode/probe_math.py`, `probe_math2.py`,
`probe_jail.py`, `probe_mix.py`), because a test that says "hostile answers are
refused" is not evidence that a hostile answer I invent is.

```
# the changed gates, current tree
CUDA_VISIBLE_DEVICES="" pytest -q tests/test_sft.py::test_the_sft_corpus_shares_no_solution_with_the_verifier \
  tests/test_lfb.py tests/test_verifiers.py tests/test_jail.py tests/test_save_counters.py
  -> ................................................................. [100%]   EXIT=0

CUDA_VISIBLE_DEVICES="" pytest -q tests/test_train.py tests/test_depth.py tests/test_loom.py tests/test_data.py
  -> ......................................                              [100%]   EXIT=0

CUDA_VISIBLE_DEVICES="" pytest -q                       -> EXIT=0, 0 FAILED
.venv/bin/ruff check src tests                         -> All checks passed!
python3 tools/check_doc_refs.py                        -> 0 named paths missing
```

The full suite is green, with one cosmetic caveat: its terminal summary line is
lost to a `filelock` `__del__` traceback at interpreter exit (`AttributeError:
'NoneType' object has no attribute 'getpid'`), so `pytest -q` ends in a traceback
even on EXIT=0. That is pre-existing and not a finding against this wave; it does
mean "the suite is green" must be read as the exit code, not the last line.

## The seven fixes

| # | commit | verdict | evidence, and what happens on revert |
|---|---|---|---|
| 1 | `ddc86ba` M1/C2 gate compares `str` with `str` | **code correct, proof still incomplete** | The corpus is genuinely clean: 395 records, 104 distinct answers, **0** type-correct collisions against 97 solutions at n=128 (my own parse, `seq_len=512`; the gate itself uses n=64); injecting a real `build_tasks("code", 1, n=3)[2]` into `_examples` FAILS the gate (`assert [('Complete t....lower())\n')] == []`). But reverting `_corpus_pairs`'s `.decode` to `bytes` leaves the test **green** (`.`), because the injected pair is appended to the list as `str` (`tests/test_sft.py:830-834`) and never passes through the decoder. The commit's claim "with the comparison reverted to bytes-vs-str it fails on the injected pair" is false. The helper is gated; the decode is not. |
| 2 | `4a27ba4` no-`--eval` tail sized in reads | **verified** | `eval_read_bytes = 20 × 8 × 513 = 82 080`; one read is 4 104 B, so 20 whole windows fit and score `20 × 8 × 512 = 81 920` with no shortfall (`test_the_no_eval_tail_is_sized_in_reads`, `_Counting` sees 81 920). Reverting `open_streams` to `eval_window_bytes` FAILS two gates: `test_the_no_eval_tail_is_sized_in_reads` and `test_eval_line_prints_the_byte_window` (`over 256 B (2 x 2 x 64, requested 3) shortfall ...`). The 1 MiB cap still reports the shortfall honestly. Caller check: `loop.py:1156` still returns the **requested** window (see R3-5); `data.py:466` uses it only as an eval-ring size, which is the right quantity there. |
| 3 | `952b53a` LFB refusal scoped to a training resume | **verified for the pretrain resume and readers; scope incomplete** | On the tree's own on-disk artifacts (`port/checkpoints/real_small`; no `.lfb` anywhere under `/home/sehaxe/dormouse`): `serve.load_checkpoint` loads `onesmoke`, `real1500`, `probe_noengram` (coef 0.01) and `noLFB1500` (coef 0.0), each with `lfb: no sidecar, controller zeroed (...)`; `load_model_state(..., resume=True)` on `onesmoke`/`probe_noengram` raises `LoudError` naming the escape. Reverting the scope makes the reader gate die on `LoudError`; removing the drift filter makes the controller-off gate die on `resume refused: ['model.moe_lb_coef']`. Gap: `sft.py:852` and `rl_train.py:194` are training resumes that call `load_model_state` with the default `resume=False`, so they take the COUNTED path — see R3-3. |
| 4 | `8a8650f`, `6b2056e` depth/loom fixtures and the round-2 report | **verified** | The 390 B tail (`3 × 2 × 65`) holds three reads; `test_depth.py:127` and `test_loom.py:149` assert `3 × BATCH × SEQ` and pass. Reverting both 3→2 FAILS both (`test_eval_depths_returns_the_curve`, `test_depth_curve_is_wired_with_loom_on`, `assert 384 == ((2 * 2) * 64)`). The round-2 report is in the tree and in `docs-site/tools/manifest.mjs:310`. |
| 5 | `93a81c3` math answer parsed under a numeric whitelist | **RCE closed; totality not delivered** | The table's payloads are refused and the markers do not exist; `\boxed{12}`, `answer: 3/4`, `-1.5e3`, `0.5` still verify. But a 23-byte answer **escapes `verify`**: `answer: (2**1000)**1000` raises `ValueError: Exceeds the limit (4300 digits) for integer string conversion` from `_short` (`verifiers.py:165`) inside `_check_number` (`:252-260`); `((2**1000)**1000)**1000` spends **5.45 s** in-process before raising, and a 4-deep pow (`((((2**1000)**1000)**1000)**1000)`, 10 AST nodes) asks for `2**10^12`, a ~125 GB integer. `sin(10**999)` costs 0.35 s and returns "not equivalent" rather than a refusal. The exponent bound is per operation, not on the composed value; there is still no timeout. See R3-1. |
| 6 | `ac18f3a` code jail as an AST allowlist + child guard | **verified** | All **768/768** reference solutions of the task set (3 seeds × 2 classes × 128) pass the new jail, including the generator-based `square_sum` and the class example. The 21 table escapes are refused by name; so are my nine (format-string attribute introspection, `type("C",(),{'x':type})`, `d.get('f')('hi')`, `e.__traceback__`, `math.__dict__`, an unbounded list doubling, and `from math import sqrt as os`); the two format probes are blocked by the result count rather than by the screen, and no marker file is created either way. Deleting `screen_code`'s body fails 10 escape tests (the runtime guard is the second line and is still exercised by `test_the_runtime_guard_is_the_second_line`). Residual, not new: `preexec_fn` in a threaded torch process (N7) and a 5 s CPU budget per rollout. |
| 7 | `d5b028a` SFT/RL saves carry counters | **verified** | A real SFT save with one tiny nvfp4 block prints `... nvfp4_zero_scale_blocks=1` and reports it in `report["counters"]`; a clean save grows no field; the RL save is handed a live bundle. Reverting both loops to `save_ckpt(run, model, opt, step)` FAILS `test_sft_save_prints_the_nvfp4_zero_block_count` and `test_rl_save_is_handed_a_counters_bundle`. No double count: a fresh `Counters` per save, merged into `total` once (`sft.py:1039`, `rl_train.py:388`), and the line prints only the file's own tally. The pretrain M3 gate stays green. |

Six of the seven code gates fail on their revert; the exception is the
M1/C2 gate's decode (item 1), which is the same class round 2 found — narrower,
because the comparison is exercised through the corpus when the decode is intact.

## Regressions from this last wave

**R3-1 (MAJOR) — the math verifier still does not return a Verdict on all model
text; one answer kills the RL/SFT run.** `_short` calls `str(value)`
(`verifiers.py:165`), Python 3.12 refuses int→str above 4 300 digits, and both
callers that format a refused value — `_check_number`'s magnitude branch
(`:260`) and `_eval_pow`'s exponent branch (`:246`) — are on the exception path,
not inside a `try`. Reproduced:

```
verify({"cls":"math","expected":"2",...}, "answer: (2**1000)**1000")
  File "src/dormouse/post/verifiers.py", line 192, in _safe_expr
    _check_number(value)
ValueError: Exceeds the limit (4300 digits) for integer string conversion
```

`post/rollouts.py:87` calls `verify` with no handler, and `rl_train.py:258`
builds the reward list with no handler, so `dormouse rl --set
rl_task_cls=math` dies on the first such rollout. This is a new failure created
by the fix: the old `sympify` path executed code (worse), but it did not raise
this. The same un-bounded work remains: 5.45 s for the 3-deep pow, and a 4-deep
pow is a 125 GB allocation. Fix: make `_short` total (`type + digit count`, or
catch `ValueError`), reject any intermediate whose bit length/magnitude exceeds
the bound *before* the next pow, and run the parse under the child timeout that
R1-N5 already asks for. Add `(2**1000)**1000` to the hostile table.

**R3-2 (MAJOR) — the C2 gate still cannot fail on its own decode.** Evidence in
item 1: reverting `.decode` in `_corpus_pairs` leaves the test green. The
injected pair is `str` by construction, so it proves `_solution_collisions`, not
that the corpus reader produces `str`. The correct injection is a record written
through `build_sft_corpus` / `render`, or an assertion that
`isinstance(answer, str)` for every parsed pair. The corpus itself is clean today
(395/104/0), which is the outcome the ledger claims; the *proof* is still one
revert away from vacuous.

**R3-3 (MAJOR) — "training resume" means only `train()`'s resume, and the
`moe_lb_coef=0` escape is not the same function.** `sft.py:852` and
`rl_train.py:194` load the checkpoint half with `load_model_state(model, cfg,
ck["model"])` on a
real resume, so a pre-sidecar SFT/RL checkpoint with the controller on is
COUNTED, never LOUD — the case `952b53a` exists for. Separately, the rationale
in `loop.py:423-425` and the `_restore_lfb` docstring ("the controller is off, so
a zero bias is the same function") is false: `experts.py:461` does
`raised = logits + self.lfb_bias` unconditionally, so a nonzero bias from the
stored run still steers top-k. `moe_lb_coef=0` stops *updating* the bias; it does
not remove it. The artifacts prove the controller ran: `mix1.pt` (step 79 999)
carries `lfb_counts` summing **11 796 277 248**, `med1` 1 966 080 000 and
`medbig2` 7 372 777 472 (all `moe_topk=2`, coef 0.01, no `.lfb`); the
`real_small` family is `moe_topk=0` and inert, which is why the reader marks
there are harmless. The drift exemption is defensible as a COUNTED break-glass
(the alternative bricks the artifacts), but it must not be described as the same
function, and the log line should say the routing changed relative to the stored
step. Passing `resume=True` in `sft.py`/`rl_train.py` is the one-line scope fix.

## New findings in areas no round has looked at

**R3-4 (MAJOR) — a mixture run resumed with an extended `steps` advances its
sources by the wrong byte counts.** `steps` is an exempt progress key
(`config.py:43`), and the mixture's stage schedule is a function of `steps`
(`stage_bounds`, `source_for_step`). `MixtureStream.skip_bytes` (`mixture.py:548`)
replays the whole consumed prefix with the **new** budget, so the per-source
cursors no longer match what the run read. Minimal plan (2 stages, boundary at
0.5×steps, `a/b` 0.5/0.5 then 0.9/0.1), resuming at step 7:

```
steps=10  replay[0,7) = {'a': 18, 'b': 3}
steps=20  replay[0,7) = {'a': 15, 'b': 6}
```

The resumed run skips 3 B of `a` and re-reads 3 B of `b` (toy window 3; at the
live shape the discrepancy is one window for every step whose stage assignment
changed between the two budgets, summed per source). The suite's resume gate
(`test_shard.py:217-241`) *does* extend `steps=2 → 4`, but its plan is one stage
with one source, so a changed schedule cannot show; a multi-stage plan is not
covered. `sw1_long` is a mix-plan run, so this is reachable on the live lane. Fix: replay
the prefix with the checkpoint's stored `steps` (the snapshot has it), then
switch to the new schedule at `start`; or refuse `steps` drift when `mix_plan`
is set.

**R3-5 (MINOR) — `report["eval_window_bytes"]` is still the requested window
when the 1 MiB cap binds (R2-6 survives in the capped regime).** `loop.py:1156`
returns `eval_window_bytes(cfg)`; with `eval_batches=4096 × 8 × 512` the tail is
1 MiB, 255 windows score 1 044 480 B, the line names the shortfall, and the JSON
field still says 16 777 216 B. The default 20-window case is now consistent, so
this is the cap only. Report `scored`, or the tail's `hi-lo`, rather than the
request.

**R3-6 (MINOR) — `\boxed{}` extraction is brace-free only, and the common
nested form is refused.** `_BOXED = r"\\boxed\{([^{}]*)\}"` (`verifiers.py:139`)
requires the body to contain no brace; `\boxed{\frac{1}{2}}` does not match at
all, so `_final_answer` falls back to the last line, hands the whole
`\boxed{\frac{1}{2}}` string to the parser, and it is refused as "not a valid
expression". Old code also failed on it (no boxed support at all), so this is
not a regression — but the docstring sells `\boxed{X}` and the first LaTeX
answer a model produces may be nested. Either strip one level of
`\frac{a}{b}`/`\dfrac` before the parse or state the limitation.

**R3-7 (MINOR) — the parser's work bound is per operation, not on the result
(R1-N5 still open).** `((2**1000)**1000)**1000` is 8 AST nodes, well under the
64-node bound, with two allowed exponents of 1000, and costs 5.45 s of
arithmetic before the crash; `sin(10**999)` costs 0.35 s.
`MAX_EXPR_CHARS/NODES/EXPONENT/MAGNITUDE` do not bound the *composition*.
This is the timeout R1-N5 asked for, still missing from a verifier that now
runs under a trainer's process.

**Checked, no finding.** `bench/score.py`'s protocol is right: `ids = C + c`,
rows `k-1 .. k+m-2`, `Σ log p(c_j | C+c[:j])`, `mean = logp/len(c)`; the runner
uses `acc_mean` as the headline and drops (never truncates) an overlong item,
with `bench_dropped` counted and a LOUD refusal over zero scored items. The
`ByteStream`/`ShardStream` `skip_bytes` maths are exact at shard boundaries
(`bisect_right(_cum, target)-1` names the next shard at `target == cum[i]`) and
the `test_shard` gates cover mid-shard, cross-shard and cross-epoch resume; the
defect is the mixture's use of a changed `steps`, not `ShardStream` itself.

## Unchanged open findings that still matter

**Process (MINOR) — the ledger stops at `R1-M9`.** Round 2's `R2-1`..`R2-6` and
`N6`..`N10` were fixed or carried in this wave but never entered
`docs/reviews/2026-10-19-findings-ledger.md`; the file's last row is R1-M9. The
loop protocol's rule 1 ("every finding gets a row") and rule 2 (a round starts
only when the ledger's CRITICAL rows are fixed) are therefore not evidenced by
the ledger for the wave that just landed — the round-2 report is the only
record, and this report's `R3-*` rows are not in it either (this review writes
one file, and the manifest line for round 3 is also still missing). The code
verdicts above stand on the code; the process claim does not.

R1-M2 — the SFT eval loops `cfg.eval_batches` with no cap to the holdout
(`sft.py:708-717`) and prints the requested window; measured before at 40
records / 2 560 B over a 12-record holdout, still open. R1-M8 — that eval builds
`sft_mask` after `.to(device)` (`sft.py:711-712`). R1-M9 — `Counters.merge`'s
substring test (`instrument.py:144-146`). R1-N3 — non-numeric `Content-Length`
resets the serve connection. The round-2 N8 (export header) and N9
(`spec_over_window`) were not touched by this wave and are still open by
inspection; N2's streaming `finish_reason` is still `stop` unconditionally
(`serve.py:429`); M4's off-cadence MX tally is still read only by
`tests/test_mxfp.py` (round 2, item 4). `docs/reviews/2026-10-19-review-loop.md`
still says rounds 1 and 2 are "pending" (round-2 N10); it is the first document
a newcomer reads and it now contradicts three files in the same directory.

## Score

**7/10** (round 2: 6; round 1: 7; round 0: code 6, infra 5).

What earns the increase: the two regressions round 2 found are gone and their
gates fail on revert (the tail scores the 20 windows it asks for; every
pre-sidecar artifact loads again with a COUNTED line), the `sympify` RCE is
closed and the code "sandbox" is now a whitelist that refuses by name, all 768
reference solutions still run, and the SFT/RL checkpoint lines carry the storage
rung's tally. The full suite is green and the on-disk artifacts were exercised,
not just the fixtures.

Why not 8: the reward path — the lane this wave was hardening — still raises out
of `verify` on a 23-byte answer and can spend 5 s (or ask for 125 GB) per
rollout before any bound fires; the C2 gate still cannot fail if its decoder is
reverted; and the LFB scope misses SFT/RL training resumes while its
`moe_lb_coef=0` escape is described as "the same function" although
`experts.py:461` adds the bias unconditionally. Those are three MAJORs in the
measurement/reward chain, which is the part the project's own standard says must
be trustworthy before a number is.

The single finding that would move the score most is **R3-1**: make `verify`
total and bounded (format refusals without `str` on a big int, bound the
composed value, run the parse under the child timeout, and add the crash payload
to the hostile table). It is the difference between "a hostile answer is a failed
rollout" — the module's own promise — and "a hostile answer is a dead trainer",
and it is cheap: one helper, one pre-check, one test. Runner-up: R3-2, the C2
gate; a CRITICAL's proof that cannot fail is the defect the ledger exists to
prevent.

## What I could not check

* Every CUDA path: the card is held by the live job. That includes the real
  `opt_graph` capture/replay, FLA/triton/fp8, and the `(d)` SFT sync-debug gate
  (it is the `s` in the CPU run). No GPU number in this report is mine.
* The 4-deep pow's actual allocation: I did not run it, to avoid OOM-killing the
  box; the 3-deep cost (5.45 s) and the `2**10^12` size are arithmetic, not a
  measurement of the kill.
* A real SFT/RL resume of a pre-sidecar checkpoint: I read the call sites and
  the `_restore_lfb` condition; the path is unambiguous, but I did not build a
  pre-sidecar SFT checkpoint and resume it.
* The live `sw1_long` run and a real mix-plan resume: R3-4 is proved on a toy
  plan and by reading `replay_bytes`, not on the live artifacts.
* CI (cannot trigger), the docs-site render, the 46 GB corpus, `selfplay/`,
  `train/distributed.py`, and the round-2 N8/N9 opens beyond confirming the files
  did not change.
* The docs-site manifest line for this report and its ledger rows: this review
  writes one file; the next commit must add them (`docs/README.md` §1.8).
