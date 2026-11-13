# The port suite's 25 minutes: where they go, and the one cut that was safe

**Date:** 2026-11-08 · **Lane:** CI/developer time. **Verdict: the suite is
8m38s at load 5 on this box, not 25 minutes — the 25 is contention (2.4x at
load ~40, unbounded at load ~80). One redundant real `torch.compile` in a
wiring test was removed (52.8s -> 0.10s, same assertions). No other safe cut
exists at this size: the rest is inductor compilation and real training work.
`pytest-xdist -n 4` is correctness-safe (0 failures) but 2.2x SLOWER with the
default thread count, and is not in `pyproject.toml` so it is proposed, not
added.**

## Command and environment

```
cd port
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q -p no:cacheprovider \
    --durations=40 --junitxml=/tmp/opencode/suite/final.xml
```

- 20 vCPU (`nproc` 20), 62 GB RAM, torch 2.13.0+cu129, pytest 9.1.1,
  CPU only (`CUDA_VISIBLE_DEVICES=""`).
- **torch's default intra-op pool here is 10 threads** (`torch.get_num_threads()`),
  not 20 — every number below is "10 threads" unless it says otherwise.
- The box is shared. Other sessions ran full test suites and cargo builds
  during this work; the load average moved between 5 and 80. **Wall time on
  this suite is a function of load first and of code second**, so every claim
  below is paired with the load it was taken at, and CPU time (`time`'s
  `user+sys`) is quoted where a load-robust number matters.

## What was measured

| run | load avg | tests | wall | CPU (user+sys) | exit | tree |
|---|---|---|---|---|---|---|
| baseline, contended | ~40 (35-45) | 2297 | **21m00s** | 77m34s | 1 (5 pre-existing failures¹) | before |
| baseline, controlled | ~10 | 2308 | **10m09s** | 34m48s | 0 | before |
| **final, controlled** | ~5.5 | 2323 | **8m38s** | 31m17s | **0** | after |
| `-n 4 --dist loadfile` (xdist, side-installed) | 30-45 | 2308 | **22m04s** | 344m21s | 0 | before |
| 1-thread experiment (reverted) | ~16 | 2308 | 11m40s | 21m19s | 0 | — |

¹ The five failures were another session's uncommitted work (`test_step_audit`
on a new `for` in `train/loop.py`, `test_lr_tail`, and three in the then-new
`test_torch_compile.py`); they were green by the final run without any change
of mine. Nothing in this lane ever deleted or skipped a test.

The same 2323-test suite therefore costs **8m38s on a quiet box and 21m00s at
load 40** — a 2.4x amplification with no code change. At load ~80 the first
baseline attempt wedged: 26m05s, 39% done, 22 minutes with zero tests
completed, inside `test_halt::test_max_iter_eight_builds_trains_and_curves`
(faulthandler stack in the same forward the isolated run finishes in 0.76s at
low load). CPU is 31-35 minutes whether the wall is 8.6 or 10 minutes; the
20-thread baseline burned **77 minutes of CPU** to do the same work at load 40.
That gap is thread-pool spin-wait, not test work.

## Ranked cost, final run (`--durations`, top 25 tests)

| s | test |
|---:|---|
| 24.96 | `test_torch_compile::test_compiled_forward_and_backward_match_eager_at_fp32_round_off` |
| 20.78 | `test_torch_compile::test_eval_windows_are_the_training_shape` |
| 19.95 | `test_torch_compile::test_a_compiled_run_keeps_its_counters_and_eval_line` |
| 19.92 | `test_compile_moe::test_the_control_without_the_decorator_burns_the_recompile_budget` |
| 17.05 | `test_param_count::test_final_patch_ships_without_the_dead_memory_half` |
| 15.46 | `test_residual_bf16::test_on_vs_off_drift_is_the_measured_200_step_value` |
| 15.36 | `test_bank_invalidation::test_the_trainer_calls_the_invalidation_every_step` |
| 14.99 | `test_moe_bank_cache::test_the_epoch_key_survives_a_trained_step_per_step` |
| 14.08 | `test_torch_compile::test_resume_across_the_compile_boundary[True-False]` |
| 13.98 | `test_torch_compile::test_the_optimizer_steps_the_same_parameters_the_compiled_model_reads` |
| 13.48 | `test_bench_compile::test_the_table_counts_every_break_and_the_eager_arm_is_plain` |
| 13.14 | `test_torch_compile::test_the_ema_shadow_is_checkpoint_keyed_and_reaches_the_compiled_forward` |
| 12.02 | `test_jepa::test_collapse_gate_the_internal_reg_is_load_bearing` |
| 11.67 | `test_compile_moe::test_the_moving_bank_key_stays_out_of_dynamos_guards` |
| 11.24 | `test_posttrain_fixes::test_seed_decides_the_fresh_init_across_processes` |
| 11.16 | `test_bench_compile::test_a_vram_projection_past_the_ceiling_refuses` |
| 10.21 | `test_muon_fast::test_the_bf16_state_rung_halves_the_state_and_is_finite` |
| 9.82 | `test_dead_knob_audit::test_runtime_and_static_agree_on_d_model` |
| 8.61 | `test_rl_tasks::test_the_corpus_is_deterministic_and_meets_the_size_target` |
| 7.53 | `test_opt_region::test_the_bf16_state_rung_is_within_the_fp32_seed_spread_and_the_pins_hold` |
| 6.23 | `test_bpb_accounting::test_the_noise_probe_measures_a_cpu_tiny_run` |
| 5.82 | `test_dead_knob_audit::test_real_presets_do_not_set_a_dead_field_off_its_default` |
| 4.55 | `test_rl_tasks::test_the_verifier_accepts_every_reference_answer` |
| 4.42 | `test_rl_tasks::test_every_rendered_prompt_is_unique_and_disjoint_from_sft` |
| 4.00 | `test_agent::test_agent_demo_runs_on_cpu_without_a_model` |

### Ranked cost by file

| s | tests | file | share |
|---:|---:|---|---:|
| 107.1 | 9 | `test_torch_compile.py` | 21% |
| 31.6 | 3 | `test_compile_moe.py` | 6% |
| 27.1 | 10 | `test_bench_compile.py` | 5% |
| 21.3 | 6 | `test_rl_tasks.py` | 4% |
| 18.4 | 6 | `test_param_count.py` | 4% |
| 16.5 | 20 | `test_rl_train.py` | 3% |
| 15.7 | 13 | `test_residual_bf16.py` | 3% |
| 15.6 | 5 | `test_dead_knob_audit.py` | 3% |
| 15.4 | 4 | `test_bank_invalidation.py` | 3% |
| 15.1 | 11 | `test_moe_bank_cache.py` | 3% |
| 14.6 | 40 | `test_muon_fast.py` | 3% |
| 13.8 | 13 | `test_posttrain_fixes.py` | 3% |
| 12.1 | 7 | `test_jepa.py` | 2% |
| 11.7 | 11 | `test_bench_engine.py` | 2% |
| 10.3 | 7 | `test_opt_region.py` | 2% |

Sum of all test time 509.5s over 2323 tests; **top 3 files = 32%, top 15 = 60%**.
The other 158 files are a 0.1-0.3s long tail.

## Classification of the top costs

1. **Real inductor compilation (the largest block, ~150-175s, ~30%).**
   `test_torch_compile` (107s/9 tests), `test_compile_moe` (32s), `test_bench_compile`
   (27s), plus `test_train`'s compile wiring. These tests exist to prove a
   compiled run matches eager and that the recompile budget/counters hold;
   `torch.compile` has no cheaper mode that keeps that claim. Not touched.
2. **Redundant compilation in a wiring test — FIXED.** 
   `test_train::test_torch_compile_flag_wraps_the_model_after_resume` says in
   its own docstring "`torch.compile` itself is patched out: this is a wiring
   test, not a compiler test" — but its spy called the real `torch.compile`.
   Its assertions are `len(calls) == 1`, `dynamic=True`, a log line and a finite
   loss; none needs inductor. The spy now returns the module unwrapped.
   **Measured on the single test: 52.8s -> 0.10s**, assertions unchanged.
   The `_orig_mod.` shape stays covered by `test_ckpt_prefix.py` (stub wrapper)
   and real end-to-end compilation by `test_torch_compile.py`.
3. **Real training/eval work (~150s).** 100M-parameter preset builds
   (`test_param_count` 17s), 200-step drift (`test_residual_bf16` 15s), MoE
   bank-epoch/ledger steps (`test_moe_bank_cache`, `test_bank_invalidation`,
   ~30s combined), subprocess-isolated CLI/seed runs (`test_posttrain_fixes`
   11s). Each is one gate on a distinct claim; reducing steps would weaken it.
4. **Thread-pool oversubscription — an amplifier, not a fixed cost.** At load
   10 the whole suite is 10m09s; at load 40 it is 21m00s with 2.2x the CPU.
   The tiny fixtures (d=64, seq=64) put 10 threads at a barrier per op, and a
   loaded box makes each barrier wait. A one-thread pin was measured and
   **reverted**: at low load it is a *regression* (11m40s vs 10m09s) because
   the few real-tensor tests lose their parallelism, and re-pinning it globally
   would mean re-capturing ~50 byte-exact hashes (see next section). It is the
   right change for a *deliberately shared* box, not for a quiet one.
5. **No duplicated fixture survived.** The expensive servers/checkpoints are
   already module-scoped (`test_serve`, `test_dashboard`, `test_agent`,
   `test_full_graph`); setup time is 3% of the suite (top setup 3.1s).
6. **`pytest-xdist` is correctness-safe but not time-safe here.** With the
   plugin side-installed in `/tmp` (not added to `pyproject.toml`):
   `-n 4 --dist loadfile` finished **green, 0 failures**, syntax/servers/temp
   paths were all fine — but took **22m04s and 344m CPU** against 10m09s
   serial, because each worker runs its own 10-thread pool (40 threads on 20
   cores). It is only a win paired with a one-thread-per-worker setting, which
   is exactly the re-pin work above.

## The defect this lane found in CI

**The byte-exact gates only pass on a box whose default torch pool is the
capture size.** 26 of the 28 hash-pinned tests that were compared fail at
`OMP_NUM_THREADS=4` (a GitHub `ubuntu-latest` runner has 4 vCPU), and all 28
fail at 1 thread. Evidence:

```
OMP_NUM_THREADS=4 ... pytest <28 pinned tests>   -> 26 failed
OMP_NUM_THREADS=1 ... pytest <28 pinned tests>   -> 28 failed
default (10 threads) ... same nodes              -> 0 failed
```

So `python-test` in `.github/workflows/ci.yml` cannot be green on a 4-core
runner, today, independent of speed. The fix is a decision, not a mechanical
edit: set the suite to one thread and re-capture the pinned hashes (they are
sha256 of default-pool reductions), or keep the pins but make the suite's
thread count explicit in CI. That decision is out of this lane's "no assertion
weakened" rule; the measurements are here so it can be taken deliberately.

## Changes landed

| file | change |
|---|---|
| `port/tests/test_train.py` | `test_torch_compile_flag_wraps_the_model_after_resume`: the compile spy returns the module instead of running a real inductor compile (assertions unchanged). 52.8s -> 0.10s. |

Nothing else. `tests/conftest.py` is untouched (the 1-thread experiment was
measured and reverted; its numbers are in the table above). No test was
deleted, skipped or loosened, and `pytest-xdist` was not added to
`pyproject.toml`.

## What could skew this

- **Load.** The dominant variable: 8m38s at load 5.5 vs 21m00s at load 40 vs
  a 26-minute wedge at load 80. The final-vs-baseline suite delta (10m09s ->
  8m38s) also contains a load change (10.2 -> 5.5) and inductor cache warmth;
  the *causal* number for the landed change is the single-test A/B
  (52.8s -> 0.10s), not the suite delta.
- **Inductor cache.** A cold `~/.cache/torch/inductor` makes the first compile
  test pay 30-50s more. CI gets a cold cache every run; caching it across runs
  would cut more than any test edit.
- **A moving tree.** Other sessions were editing `src/dormouse/train/loop.py`,
  `config.py`, and adding test files throughout; test counts moved 2297 ->
  2308 -> 2323. Every number above names its run.

## Proposed, not done (needs the owner's call)

1. **Make the suite machine-independent**: `OMP_NUM_THREADS=1` for the test
   process, re-capture the ~50 thread-sensitive hash pins, and set the two
   noise-floor bounds that were tuned at 10 threads. Measured effect: 21m19s
   CPU vs 34m48s at comparable load (-39%), wall roughly neutral at low load
   and 2-4x better under contention; CI goes green on 4 cores.
2. **Then add `pytest-xdist` and run `-n 4` with one thread per worker.** The
   plugin is already proven safe here (0 failures); the projected wall is
   ~2.5-3.5 minutes, but it must be measured after (1), not assumed.
3. **CI**: cache `~/.cache/torch/inductor` between runs; it is the largest
   single cost in a cold CI run.
