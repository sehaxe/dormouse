# Findings ledger — the 2026-10-19 audits and their fixes

Three independent reviewers read the tree on 2026-10-19 (code, model, infra) and
this file is the ledger: every finding, whether it is fixed, and what proves it.
A finding leaves the ledger only by being fixed **with a gate** or by being
rejected **with a reason** — the same rule the project applies to its arms.

Status values: `fixed` (with the proof named) · `open` · `queued` (a task exists)
· `rejected` (with the reason).

## Sources

- `docs/reviews/2026-10-19-infra-audit.md` — verdict 5/10 (honesty 9,
  reproducibility 3, automation 5). Reviewer: independent, read-only.
- `docs/reviews/2026-10-19-model-scorecard.md` — the model's honest position:
  "the instrument is the product; the model is a demonstration".
- `docs/reviews/2026-10-19-code-audit.md` — the Python tree at `7f0b656`
  (landed as `6b25cf6`); C1/C2 and M1-M8 below.
- `docs/reviews/2026-10-19-review-round-1.md` — the hostile read of the
  2026-10-19 fix wave; its findings are the `R1-` rows below.

## CRITICAL

| # | finding | status | proof |
|---|---|---|---|
| I1 | the live run's only non-`--set` input was untracked (`configs/small_wide.toml`) and its command line lived on tmpfs | **fixed** | `configs/small_wide.toml` committed in `90d5e7f` |
| I2 | README's first command exits 2: `dormouse train` has no `--ckpt-name`/`--out-dir` | **fixed** | the quick start now uses `--set ckpt_name=… --set out_dir=…` (`90d5e7f`) |
| I3 | data and artifacts live outside both checkouts, so "clone → train → measure" is false | **fixed (documented)** | `docs/guides/data.md`: what the corpora are, the smallest honest recipe, what the eval carve must be; README points at it |
| I4 | README sold depth-3 as "the measured optimum" while every preset runs `max_iter=4` and the mixture runs' own curve still improves at 4 | **fixed** | README states the A/B's numbers, its window and the open question; AGENTS.md "Retracted — the Python lane" repeats it |
| I5 | the README's HellaSwag comparison cited comparator numbers that are in no model's own suite, and crossed protocols (byte `acc_mean` vs token `acc_norm`) | **fixed** | README/README.ru now say: above the 25% floor, below SmolLM2-135M's 42.1, no protocol bridge exists |
| I6 | three artifacts (`b1night`, `medium_big2`, `sft9`) are invalid — the expert-bank cache froze under `opt_graph` — and nothing in the repo said so | **fixed** | AGENTS.md "Retracted — the Python lane"; the fix is pinned by `tests/test_bank_invalidation.py` |

## MAJOR

| # | finding | status | proof / plan |
|---|---|---|---|
| I7 | `benches/history.tsv` deleted by the restructure while ~87 citations name it, and `check_doc_refs.py` skips `benches/` wholesale | **fixed** | restored in `90d5e7f`; the skip stays (a bench log is evidence, not a path) |
| I8 | the demo's README said the artifact is "not in the repository" while the blob is committed | **fixed** | README says why it is committed and what a fork without it sees (`90d5e7f`) |
| I9 | CI installed no node, so the browser-engine parity gate skipped itself — the one gate that proves the shipped engine computes this model's arithmetic | **fixed** | `actions/setup-node@v4`, node 22, in `ci.yml` (`90d5e7f`) |
| I10 | AGENTS.md §3 carried none of the Python lane's retractions | **fixed** | the new section at the top of AGENTS.md |
| I11 | no tags, no releases; 8 dependabot PRs unmerged | **open** | needs the owner: tags are outward, and the PRs are red against the old `main` |
| I12 | the local suite cannot be green while a training run holds the card (`AdamW.step()` → `AcceleratorError`) | **queued** | a `conftest` guard that turns "the card is busy" into a skip, not an error |
| I13 | the run row of the JSONL telemetry carries no `ckpt_name`/`out_dir`/`mix_plan`, so a run's recipe is not recoverable from its own log | **queued** | one line in `train/loop.py`'s run event |
| M1 | "×36.6 dense equivalent" for b1 appears in no file; the derived ratios are ×28.1 / ×18.7 / ×59 by component | **open** | the scorecard's §1; the project memory carried the ×36.6 — replace it with the derived table or drop the claim |
| M2 | `b1` param count disagreed between the port README (…113) and the pinned test (…209) | **fixed** | the plan's row carries the date and both numbers |
| M3 | no protocol bridge between our byte-level bench and published token-level suites | **queued** | the scorecard names it as one of the three measurements that would change the verdict |

## Code audit — C1/C2 and M1-M8

| # | finding | status | proof |
|---|---|---|---|
| C1 | the trainer never seeded the model init; `train.seed` was not a property of a run | **fixed, verified** (R1 a) | `tests/test_seed_init.py`; removing `torch.manual_seed(run.train.seed)` from `cli._model` fails both same-seed tests (`b0da9a7`) |
| C2 | the SFT corpus contained the exact tasks the RL lane and `dormouse run` reward | **fixed** | ids/prompts disjoint (`b44b543`); the ANSWER strings still collided and are now gone from the corpus (`4f61858`), gated by `test_the_sft_corpus_shares_no_solution_with_the_verifier` |
| M1 | `nvfp4` silently zeroes small blocks (`min_normal = 2**-448`, not `2**-6`) | **fixed, verified** (R1 e; counter R1-M3) | `tests/test_tsct_ladder.py` for the prescale; `63379a5` passes the counters and `test_a_tiny_nvfp4_block_is_counted_on_the_checkpoint_line` asserts the count reaches the checkpoint line |
| M2 | `counters.why` never reached the log line or the JSON row | **fixed, verified** (R1 f) | `tests/test_counters.py`; the old code printed neither field |
| M3 | the SFT eval never rewound its stream | **fixed, gate strengthened** | `_sft_eval` calls `rewind()` (`sft.py:699`); the round-1 gate could not fail, the holdout is now three eval windows (`4b3ab54`) |
| M4 | `mm_dtype` / `kda_disable_recompute` were inert on SFT and RL | **fixed for the CLI, verified** (R1 i) | the setters moved into `load_run` (`7f8b906`); `load_run` leaked them to non-training callers (R1-M6), now opt-in (`a1bb3eb`). Direct library callers of `sft_train`/`rl_train` still bypass the setters |
| M5 | the no-`--eval` tail path overstates the scored bytes when the window exceeds 1 MB | **fixed** (R1-M5) | `_eval_batches` divides by the bytes one read consumes, `batch*(seq_len+1)`, and `_eval_line` names the shortfall and the leftover bytes (`loop.py:441-456`, landed in `25e8707`); the two depth/loom fixtures followed in `ee451a5` and are green |
| M6 | the CPU gate cannot run while a training job holds the card | **queued** (infra I12) | a `conftest` guard that turns "card busy" into a skip; not landed |
| M7 | device syncs in the forward: quantizers, the SFT step, the naive MoE arm | **partly fixed** | act_quant/mxfp cadence verified (R1 g), the off-cadence MX zero is now COUNTED (`98642e6`), and the SFT step's mask/count is host-side with a CUDA sync-debug gate (`ea2dc53`); the SFT eval mask is still built on device (R1-M8), and the naive MoE arm still calls `torch.unique(...).tolist()` (`experts.py:753`) |
| M8 | `lfb_bias` is not persisted while `lfb_counts` is | **fixed, verified** (R1 j) | `tests/test_lfb.py` round-trip; the sidecar write is not atomic with the weights and a pre-fix checkpoint resumes with a zero bias (R1-M7) |

## Review round 1 — N1-N5 and M1-M9

| # | finding | status | proof / plan |
|---|---|---|---|
| R1-N1 | `Counters.why` crashes `OptimizerGraph._replay` (`graph.py:284`, `:322`) | **fixed** | `9f76650`; `tests/test_opt_graph.py` replays a captured counters dict that carries `why` (no GPU needed) |
| R1-N2 | `serve.py` reports `completion_tokens = n` and `finish_reason = "stop"` even when the early stop byte or the budget ended the reply | **fixed** | `f964190`; `tests/test_serve.py` pins `len(content)` and `length` on a budget exit |
| R1-N3 | a non-numeric `Content-Length` raises in the handler thread, resetting the connection instead of a JSON 400 | **open** | `serve.py:306-311`; no body-size cap either |
| R1-N4 | the findings ledger omitted the code audit's C1/C2/M1-M8 and the round-1 findings | **fixed** | this file; the rows above are the missing sets |
| R1-N5 | `_verify_math`'s sympy call has no timeout and the sandbox patches only `builtins.open` | **open** | `post/verifiers.py:52-56`, `:136-142`; one generated expression can hang the reward loop |
| R1-M1 | the SFT corpus still contained the verifier's exact answers | **fixed** | `4f61858`; gate `test_the_sft_corpus_shares_no_solution_with_the_verifier` |
| R1-M2 | the SFT eval prints and scores more records than its holdout holds | **open** | `_sft_eval` loops `cfg.eval_batches` without capping to the stream (`sft.py:708-717`), and `sft_train` prints the requested window (`sft.py:1013-1018`) |
| R1-M3 | the `nvfp4` zero-block counter never reaches a log | **fixed** | `63379a5`; `pack_tsct` gets the counters from `save_ckpt` and `test_a_tiny_nvfp4_block_is_counted_on_the_checkpoint_line` pins the line |
| R1-M4 | the mxfp cadence turns the e8m0 range refusal into a silent zeroed block | **fixed** | `98642e6`; `tests/test_mxfp.py` asserts the off-cadence range refusal is zeroed and COUNTED, never silent |
| R1-M5 | `_eval_batches` is off by one target byte per window | **fixed** | `loop.py:441-456` now divides by `batch*(seq_len+1)`; `ee451a5` moves `tests/test_depth.py:123` and `tests/test_loom.py:147` to the whole windows the eval scores |
| R1-M6 | `load_run` mutates process-wide compute state for non-training callers | **fixed** | `a1bb3eb`; gate asserts `info` and the serve/bench preset check stay at bf16/False while `train`/`run`/`sft`/`rl` apply fp8/True |
| R1-M7 | the LFB sidecar is a second atomic write and a pre-fix checkpoint resumes with a zero bias | **partly fixed** | `58d1333`: a missing sidecar (`loop.py:111-120`) is LOUD when the controller is on and COUNTED when it is off (`test_a_missing_lfb_sidecar_is_loud_when_the_controller_is_on`, `..._is_counted_when_the_controller_is_off`); the weights and the sidecar are still two writes (`loop.py:211`) |
| R1-M8 | the eval half of SFT still builds its mask on the device | **open** | `sft.py:711-712`; the training step is host-side and gated (`ea2dc53`), eval is not |
| R1-M9 | `Counters.merge` drops a distinct reason by substring | **open** | `instrument.py:144-146`; the merge is not a set |

## Round 2's findings (2026-10-19, `2026-10-19-review-round-2.md`)

| # | finding | status | proof |
|---|---|---|---|
| R2-1 | the M1 gate compared `bytes` with `str`, so an injected task still passed | **fixed** | `ddc86ba`: str with str, and the test injects a real `count_vowels` task and requires the helper to name it |
| R2-2 | the SFT corpus still held verifier-scored code/math pairs | **fixed** | `4f61858`: `build_tasks` pairs removed from the corpus, `--n-tasks` gone (an old command line dies loudly), the gate compares SOLUTION TEXT |
| R2-3 | the no-`--eval` tail was carved in SCORED bytes, so the default 20 windows scored 19 | **fixed** | `4a27ba4`: sized in reads (`eval_batches × batch × (seq_len+1)`); `test_the_no_eval_tail_is_sized_in_reads` |
| R2-4 | SFT/RL saves passed no counters, so the nvfp4 tally never reached their lines | **fixed** | `d5b028a`; `tests/test_save_counters.py` |
| R2-5 | a missing `.lfb` refused every reader, and `moe_lb_coef=0` was refused on resume | **fixed** | `952b53a`: LOUD only on a training resume with the controller on; readers and controller-off resumes get the COUNTED line |
| R2-6 | the 1 MiB-capped eval reports the REQUESTED window, not what the cap allows | **open** | `loop.py` prints `requested` in the shortfall field; the honest number is what the cap permits |
| R2-7 | `verify` raised `ValueError` on `answer: (2**1000)**1000` and spent 5.45 s on a 3-deep form | **fixed** | `d822ac5`: `MAX_RESULT_BITS=333` checked BEFORE the power is computed, 50 ms deadline, worst allowed expression 0.31 ms |
| R2-8 | the reward path had no boundary: a verifier bug killed the rollout batch | **fixed** | `7c48b0c`: `rollouts.py` counts `rollout_verify_error` and fails the SAMPLE |
| R2-9 | the code jail was bypassed through `os.open`/`pathlib` | **fixed** | `ac18f3a`: AST allowlist + child guard; 21 table escapes + 9 more blocked, 768/768 reference solutions pass |
| R2-10 | the math verifier executed the model's answer text in the trainer (sympy RCE) | **fixed** | `93a81c3`: numeric whitelist, `/tmp/pwn_*` markers absent in the gate |

## Round 3's findings (2026-10-19, `2026-10-19-review-round-3.md`)

| # | finding | status | proof |
|---|---|---|---|
| R3-1 | the reward path died on a 23-byte answer; no timeout on the numeric path | **fixed** | `d822ac5` + `7c48b0c` (see R2-7/R2-8) |
| R3-2 | the M1 decode gate could pass on a revert | **fixed** | `a863aad`: a non-ASCII record makes the raw-bytes comparison visibly different |
| R3-3 | the LFB scoping stopped at the pretrain trainer; `lfb_counts` accumulated 11.8e9 across a run | **fixed** | `049449b`: SFT/RL pass `resume=True`; counts zeroed on load (`test_a_resume_counts_only_the_steps_since_the_load`) |
| R3-4 | a mixture run extended with more `steps` resumed from the wrong position | **fixed** | `75b5a1d`: `skip_to_step(step, budget=stored_steps)`; the gate replays the prefix under the checkpoints' budget |
| R3-5 | the ledger claimed `moe_lb_coef=0` was "the same function" | **fixed** | `049449b`: it is a COUNTED break-glass, and the line says the routing differs from the stored step |
| R3-6 | the depth/loom gates expected the read-sized tail's window count | **fixed** | `8a8650f`: 3 windows, with the arithmetic in the comment |

## What the ledger is for

The review loop the owner asked for: fix → an independent reviewer verifies the
fixes, hunts new defects and scores the framework out of 10 → if the score is
below 9, another round (at most three). The score, the round number and what
moved it live in `docs/reviews/2026-10-19-review-loop.md`.
