# Review round 1 — the 2026-10-19 fix wave

**Reviewer:** independent, read-only. This file is the only write.
**Reviewed:** `py-main` @ `70898ed`; commits `b0da9a7`, `b44b543`, `0551bef`,
`5e86d03`, `1bed202`, `7f8b906`, `e15cb4e`, `70898ed`, plus the infra/SEO wave
`90d5e7f`..`6b25cf6`.
**Environment:** the one GPU is held by the live job (`nvidia-smi`: 8,951 MiB,
43 %). Every pytest run below used `CUDA_VISIBLE_DEVICES=""`, so this review is a
CPU review plus reading; the GPU gates are named where they matter and are in
"What I could not check".

## How the verdicts were produced

Named tests first, then a revert harness: `git archive HEAD` into
`/tmp/opencode/rv`, edit only the copy, run the copy with
`PYTHONPATH=/tmp/opencode/rv/src` and the repo's `.venv/bin/python`. The repo
itself was never edited.

```
# named gates, current tree (all pass)
CUDA_VISIBLE_DEVICES="" pytest -q tests/test_seed_init.py tests/test_tsct_ladder.py \
  tests/test_counters.py tests/test_act_quant.py tests/test_mxfp.py tests/test_lfb.py
  -> ..........s... [100%]
CUDA_VISIBLE_DEVICES="" pytest -q tests/test_sft.py tests/test_train.py tests/test_config.py
  -> 71 passed
CUDA_VISIBLE_DEVICES="" pytest -q
  -> only FAILED tests/test_decode_recurrent.py::test_per_byte_cost_does_not_grow_with_context
CUDA_VISIBLE_DEVICES="" pytest -q --deselect tests/test_decode_recurrent.py::test_per_byte_cost_does_not_grow_with_context
  -> EXIT=0
python3 tools/check_doc_refs.py   -> 0 named paths missing (exit 0)
.venv/bin/ruff check src tests    -> All checks passed!
```

The `test_per_byte_cost` failure is a timing test (unchanged since `7f0b656`):
it passed 1 of 3 solo runs and failed the full run; it is pre-existing noise, not
caused by this wave, but it still means "full tests green" is not reliably true.

## (a)-(j): verdicts

| item | verdict | gate, and what happens on revert |
|---|---|---|
| (a) seed owns the init | **verified** | `tests/test_seed_init.py` passes. Removing `torch.manual_seed(run.train.seed)` from `cli._model` fails both same-seed tests (in-process and two-subprocess hash). The cross-process test is the one the audit measured, and it now agrees. `torch.manual_seed` also seeds CUDA but does not raise while the card is busy (checked). |
| (b) SFT / RL task sets disjoint | **verified for instances; the answer leak remains** (see M1) | `test_the_sft_corpus_and_the_rl_task_set_are_disjoint` passes; setting `SFT_TASK_OFFSET=0` fails it. But the gate checks `tid` disjointness and exact instruction equality, and the code generator's solutions depend only on `i % 3` (M1). |
| (c) SFT eval rewinds and prints the window | **claimed-but-not-proved** | The code does call `eval_stream.rewind()` (`sft.py:741`). The named gate is blind: with the rewind removed, `test_two_sft_evals_on_one_checkpoint_score_the_same_window` still passes, because its `holdout=4` / `eval_batches=2, batch=2` fixture consumes exactly the holdout, so the ring is back at `lo` anyway. A stronger fixture (`holdout=16`, same eval window) shows the defect: with rewind `ce=6.1781/6.1781`; without, `6.1781/6.4607`. The printed window is also the requested one, not the holdout's (M2). |
| (d) SFT bytes counted on the host | **claimed-but-not-proved** | The code is host-side (`sft.py:947-953`: mask built and `int(mask.sum())` before `.to(device)`; `sft_loss(..., n_supervised=...)`). But the named test compares the printed count with `int(sft_mask(x).sum())` on CPU where `device=None`; the pre-fix code would produce the same count, so the test cannot fail on revert. There is no sync-debug gate for `sft_mask`. |
| (e) nvfp4 prescale + zeroed-block count | **prescale verified; counter claimed-but-not-proved** | `tests/test_tsct_ladder.py` passes; forcing the old `global_scale=1` path fails all three prescale/count tests. But `pack_tsct`'s `counters` argument is passed by nobody: `save_ckpt` calls `pack_tsct(v, rung)` (`loop.py:222`) and the only caller that passes `Counters` is the new test. `nvfp4_zeroed_blocks()` is read only by `tests/test_tsct_ladder.py`. So a real save still zeroes small blocks with no log/JSON field (M3). |
| (f) `Counters.why` reaches the row and line | **verified** | `tests/test_counters.py` passes. End-to-end: a tiny train with `moe_kernel=naive` prints `... fallback: moe_kernel=naive: the config picked the reference loop` and the JSON row carries `"why": "..."` (the old code had neither). |
| (g) per-forward syncs out of act_quant/mxfp; NaN still refused | **verified in scope, with a new silent branch** | `test_finite_guard_is_cadenced_and_the_forward_takes_no_host_read` and the mxfp twin pass; restoring the unconditional `bool(isfinite(...).all())` fails both. The NaN-gap tests pass and my probe shows `int8/int4/fp4/mxfp8` all return NaN after the cadence, so the step firewall sees it. The e8m0 *range* guard does not return NaN; after the cadence it silently zeroes a denormal block (M4). |
| (h) tail path prints what it scored | **verified, with a residual wrap** | `test_the_tail_eval_prints_only_the_bytes_it_scored` passes; reverting the line to `window` fails it. Residual: `_eval_batches` divides by `batch*seq_len` but every window reads `batch*(seq_len+1)` bytes, so the last window still wraps (M5). |
| (i) `mm_dtype` / `kda_disable_recompute` wired or refused | **verified for the CLI commands** | `test_mm_dtype_and_kda_recompute_are_live_on_sft_and_rl` passes; deleting the two setters from `load_run` fails it. Direct library callers of `sft_train`/`rl_train` still bypass the setters, and `load_run`'s side effect now reaches non-training callers (M6). |
| (j) `lfb_bias` survives a resume | **verified** | `tests/test_lfb.py` passes; removing `_restore_lfb(model, cfg)` fails the round-trip. Caveats: a pre-fix checkpoint with counts and no `.lfb` silently resumes with a zero bias; the weights and the sidecar are two separate atomic writes, not one (M7). |

## Regressions introduced by the wave

**M1 (MAJOR) — the CRITICAL C2 leak is only instance-level; the answers are
still in the corpus.** `post/tasks.py:96-140` derives a code task's reference
implementation from `i % 3` only; `train/sft.py:453-465` only changes the
instruction (appends the SFT instance's verifier calls). Measured:

```
def add: solution identical across offset = True   (rl code-1-0, sft code-1-10002)
def square_sum: solution identical across offset = True
def count_vowels: solution identical across offset = True
code solution-set equality: True
math solution-set equality: False
SFT instruction starts with the RL prompt (same kind): True
```

So the SFT corpus still teaches the exact answer strings the RL/`run` lane
rewards for the code class, and the RL prompt template is still a prefix of the
SFT instruction. The new test's exact-membership assertion (`task.prompt ...
not in instructions`) cannot see a prefix. The index offset fixes the
same-instance leak; it does not fix the same-answer leak. This is the finding
that most directly undermines the round-0 CRITICAL fix.

**M2 (MAJOR) — the SFT eval prints more bytes and records than its holdout
holds.** `_sft_eval` (`sft.py:721-767`) loops `cfg.eval_batches` without capping
to `eval_stream`'s range, and `sft_train` prints `over {eval_scored} B
({cfg.eval_batches} x ...) = {first_n} records` (`sft.py:1050-1055`). With
`--build-sft --holdout 4` and the default `eval_batches=20, batch=2, seq_len=64`,
the holdout is 260 B / 4 records but the line prints `over 2560 B ... = 40
records`; the ring wraps ten times and the same four records are scored
repeatedly. This is the pretrain M5 defect on the SFT lane, and the new test
only uses `eval_batches=2`, which happens to fit.

**M3 (MAJOR) — the nvfp4 zero-block counter never reaches a log.**
`pack_tsct(t, rung, counters=None)` (`tsct.py:735`) is called by `save_ckpt`
with two arguments (`loop.py:222`); no production caller passes a `Counters`,
and no line prints `nvfp4_zeroed_blocks()`. The prescale is fixed, but the
"COUNTED, never SILENT" half of audit M1 is a process-local tally read only by
tests.

**M4 (MAJOR) — the mxfp cadence turns a LOUD refusal into a silent zeroed
block.** `_e8m0_encode`'s range guard (`mxfp.py:150`) and `_check_finite`
(`mxfp.py:200`) run on `_GUARD.due()`. After the first 8 `due()` calls (about
three packed tensors), a denormal-amax block is no longer refused; the exponent
wraps in the uint8 code and the block decodes to zeros. Reproduced:

```
quantize_mxfp4 NO RAISE after cadence; decoded amax 0.0 finite True
quantize_mxfp8 NO RAISE after cadence; decoded amax 0.0 finite True
  with reset: RAISED LoudError MX block scale -135 is outside e8m0's range
```

The commit's claim "this codec has already made the value NaN (never a
plausible-looking number)" is false for this branch. The e15 test only checks
NaN input, which does propagate. `factor_qat=mxfp4` / `dense_qat=mxfp4` on a
small block now silently loses it, exactly the class M1 of the audit named.

**M5 (MINOR) — `_eval_batches` is off by one target byte per window.**
`loop.py:403-417` computes capacity as `span // (batch*seq_len)`, but
`ByteStream.next_batch` reads `batch*(seq_len+1)` bytes. The default tail
(`eval_batches=20, batch=8, seq_len=512`) is a 81,920 B span and reads 82,080 B:
the last 160 B wrap; the live `small_wide` shape (`batch=32`) wraps 640 B. The
line prints `over 81920 B` with no shortfall. The printed count is now the
model's input count (not the requested window), so this is smaller than the old
M5, but the "only the windows that fit" claim is still false and the wrap is
undisclosed.

**M6 (MAJOR) — `load_run` mutates process-wide compute state for non-training
callers.** `config.py:1565-1566` applies `set_mm_dtype` and
`set_disable_recompute` inside `load_run`. `info` uses `_run`; `serve` and
`bench` call `_preset_check`, which uses `_run` (`cli.py:197, 744, 1174`).
Measured: `load_run("small", {"mm_dtype":"fp8"})` leaves
`tsct.mm_dtype()=="fp8"`. A `serve` started with such a preset runs fp8 while
the snapshot (and `config.py:489-499`'s documented "a standalone
`dormouse eval`/`generate` process starts at the default bf16") says otherwise.
The fix's comment claims the seam is only for training commands; it is not.
Separately, `load_run(..., rl_tasks=10001)` still accepts a value `rl_cmd` would
refuse, and only `rl_cmd` carries the SFT-offset refusal (`cli.py:583`).

**M7 (MINOR) — LFB sidecar atomicity and old checkpoints.**
`save_ckpt` replaces the weights (`loop.py:233`) and then writes/replaces the
`.lfb` sidecar; a crash between them leaves a checkpoint whose bias does not
match its counts. `_restore_lfb` returns silently when the sidecar is absent
(`loop.py:111-120`), so a pre-fix checkpoint with nonzero `lfb_counts` resumes
with a zero controller under the same step count — the M8 defect, now only for
old files. A COUNTED line for "sidecar absent, bias starts at zero" is missing.

**M8 (MINOR) — the eval half of SFT still builds the mask on the device.**
`sft.py:748-749` moves `ex` to the device and then calls `sft_mask(ex)`, whose
per-row `bool(...)` checks are D2H reads. The training step was fixed; the eval
path was not. Eval is not the hot path, but the rule is per-step, and the commit
message claims the seam.

**M9 (MINOR) — `Counters.merge` drops a distinct reason by substring.**
`instrument.py:144-146` keeps a reason only when `other.why not in prev`; a
reason that happens to be a substring of an already-merged one is lost. Not a
correctness bug for the current sites, but the merge is not a set.

## What none of the three audits looked at (new findings)

**N1 (MAJOR) — `Counters.why` crashes the CUDA-graph replay.** Making `why` a
dict key (`instrument.py:68-77`) broke `OptimizerGraph`: `graph.py:284` captures
`dict(opt.counters)` into `_captured_counters: dict[str, int]`, and
`graph.py:322` replays each entry through `Counters.count`, which asserts
`isinstance(n, int)`. Reproduced without a GPU:

```
g._captured_counters = {"ns_triton_fallback": 1, "why": "triton: input is on cpu"}
g._replay()
-> AssertionError counter why: n must be > 0, got 'triton: input is on cpu'
```

Trigger: `--set opt_graph=true --set ns_kind=triton` plus any triton-NS
fallback on the capture step (`ns_dtype=bf16`, no triton, a launch refusal). The
run dies one step after capture. CPU/CI cannot see it (`test_opt_graph.py`
skips). This is the one regression that can kill a supported arm.

**N2 (MAJOR) — `serve.py` reports a completion it did not generate.**
`serve.py:372-376` sets `usage.completion_tokens = n` (the requested cap) and
`finish_reason = "stop"` unconditionally. When `_generate` hits the `0x03/0x00`
stop byte early, the returned `content` is shorter than `n` but `usage` still
claims `n` completions; when the `n`-byte budget runs out without a stop byte,
the finish reason should be `length`, not `stop`. `tests/test_serve.py:93` pins
the requested value, so the wrong number is now a contract. The actual emitted
byte count is `len(content)` (the generator already knows it) — a two-line fix.

**N3 (MINOR) — the serve body parser resets the connection on bad
Content-Length.** `serve.py:306-311` does `int(self.headers.get("Content-Length") or 0)`
outside the `LoudError` handler; a non-numeric header raises `ValueError` in the
handler thread, so the client gets a reset instead of the documented JSON 400.
No body-size cap either.

**N4 (MINOR) — the findings ledger is not the ledger it claims to be.** It says
"every finding, whether it is fixed" and lists the code audit as "pending"
(`2026-10-19-findings-ledger.md:4,17`), but the code audit landed in `6b25cf6`
and its C1/C2 and M1-M8 have no rows. The infra audit's M3 and M7 are also
missing; `README.md:48` still claims "Every mechanism ... was admitted by a
3-seed A/B" while `AGENTS.md:773` says no arm has been judged. The round-1 gate
("the ledger's CRITICAL rows are fixed") is therefore satisfied by the *infra*
CRITICALs, not the two measurement-chain holes this round was supposed to
close. This is a process failure of the protocol, not a code bug, and it is why
M1 can hide in plain sight.

**N5 (MINOR) — `post/verifiers._verify_math`'s sympy call still has no
timeout** (audit MINOR 3, never entered in the ledger): one generated
expression can hang the reward loop. The code sandbox's `open` patch is still
bypassed by `pathlib`/`io.open` (same audit item).

## Infra wave (spot checks)

`configs/small_wide.toml`, `benches/history.tsv`, `docs/guides/data.md` are
tracked; README's quick start now uses `--set ckpt_name=… --set out_dir=…`;
`AGENTS.md` carries the Python retraction section; `ci.yml` installs node 22.
`tools/check_doc_refs.py` exits 0 and `ruff` is clean. The site manifest has the
new review pages. Nothing here is contradicted by the environment; the
unfinished part is the ledger (N4) and the README A/B sentence.

## Score

**7/10** (round 0: code 6, infra 5).

Justification: seven of the ten named gates fail on revert (a, b, e-prescale, g,
h, i, j), and the two CRITICAL holes are now at least code-addressed. But (c)
and (d) have gates that cannot fail, the (e) count never reaches a log, and (b)
still teaches the code answers; the wave also introduced one run-killing
regression (N1) and two measurement-chain holes (M2, M4), and the ledger that is
supposed to make this reviewable does not contain the findings. The instrument
is stronger, not yet trustable at the level the project claims.

The two findings that would move the score most:

1. **N1, the `why`/`OptimizerGraph` crash** — a fix that breaks a supported
   `opt_graph` arm is a functional regression; it needs a gate that runs
   `_replay` on a counters dict containing `why` (no GPU needed).
2. **M1, the code-answer overlap** — the round-0 CRITICAL is only
   instance-level. Either the generator needs per-instance answer text (the
   function body parameterised by the SFT instance's constants) or the SFT lane
   must drop the code class from the same generator; the current test cannot
   see the difference.

(Runner-up: M4, the mxfp silent zero — it is the audit's own M1 class on the
QAT lane.)

## What I could not check

* Every CUDA path: the card is held by the live job. That includes the real
  `opt_graph` capture/replay (N1 was proved on the `_replay` code path by
  construction, not on hardware), the triton MoE/NS kernels, fp8 `_scaled_mm`,
  FLA KDA, and the GPU-only tests (`test_opt_graph.py`, `test_moe_kernel.py`,
  `test_experts_gpu.py`).
* The real corpus, the live `small_wide` run, and every on-disk checkpoint. The
  LFB sidecar behavior on the checkpoints on record (b1, mix1, the SFT family)
  was not exercised; only a fresh tiny checkpoint.
* The CI run itself (can't push/trigger); I only read `ci.yml`. The docs-site
  build/SEO (robots, canonical, JSON-LD) was not rendered in a browser.
* `model/export.py`'s packing arithmetic, `selfplay/`, and `train/distributed.py`
  (not on this branch).
* The root cause of the `test_per_byte_cost` flake; it is a 2x timing bound on
  a CPU box with a live GPU job, and it is not part of the diff.
* Whether the default `--set opt_graph=true` combinations in the wild actually
  hit the N1 fallback at capture; the code path is confirmed, the frequency is
  not.
