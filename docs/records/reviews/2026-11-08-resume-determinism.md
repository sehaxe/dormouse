# Resume determinism — 2026-11-08 (port lane, CPU)

**Question.** Does a checkpoint at the end of optimizer step N, loaded into a
fresh process (fresh model + fresh optimizer), continue the run that never
stopped — bit for bit, including the bytes it reads?

**Verdict.** On every default arm: **yes, bit-exact in fp32 on CPU**, including
the staged mixture across a stage boundary, `grad_accum=2` and `doc_masking`
(gated), and EMA, SF-Muon and `rand_depth` (measured by probe, not pinned by a
test yet). One arm broke — `moe_router_jitter > 0`, an unseeded global RNG in
the forward — and is fixed here by seeding it from `(seed, step)`; the fix also
makes two runs of one config agree. One arm remains a **named, logged
exception**: `ternary_s3t`, whose mask is a process-local generator that a
resume restarts. GPU paths are not verified in this lane.

Evidence: `tests/test_resume_exact.py` (new, 6 tests) plus the existing resume
suite, run on this box, torch 2.13.0+cu129 CPU, fp32:

```
.venv/bin/python -m pytest -o addopts="" -q tests/test_resume_exact.py \
    tests/test_resume.py tests/test_invariants_resume.py \
    tests/test_invariants_checkpoint.py tests/test_mixture.py   # 59 passed
.venv/bin/python -m pytest -o addopts="" -q tests/test_sft.py \
    tests/test_sft_v2.py                        # 33 passed, 2 skipped (pre-existing)
```

## What the test compares

`tests/test_resume_exact.py`. A 6-step run and a 3 + crash + resume run (the
crash is an exception on the log line after the step-3 checkpoint, so the
interrupted leg really did train past the checkpoint it left behind) are
compared on: the per-step loss trace, every parameter and buffer (sha256 over
name+shape+bytes), and the final checkpoint's `model` half, `moe` section and
optimizer state tensors.

1. `test_three_plus_three_is_six_bit_exact` — the guarantee on the tiny MoE
   (Engram, router controller, non-persistent `lfb_bias`).
2. `test_mixture_resume_serves_the_same_bytes[1|2]` — a 2-stage
   `MixtureStream` (stage 0 = a `.bin`, stage 1 = a shard DIRECTORY, so
   `ShardStream`'s modulo-an-epoch resume is on the path). The resume crosses
   the stage seam; the resumed run's served windows are compared
   tensor-for-tensor with the tail of the unbroken run's and with their stage
   index. This is `test_skip_bytes_lands_where_an_unbroken_run_would_be`
   lifted to the trainer: the fast-forward is `train()`'s
   `start × (seq_len+1) × batch × grad_accum` (`_train_impl`'s resume branch), not the
   stream's own unit test.
3. `test_router_jitter_resumes_exactly_and_repeats_across_runs` — the fix.
4. `test_doc_masking_resume_is_bit_exact` — the packed reader's fast-forward is
   by STEPS (`DocPackedStream.skip_steps`), a different contract from flat
   bytes.
5. `test_sft_loop_resumes_exactly_with_router_jitter` — `sft_train` is a
   second step loop with its own copy of the state contract; the seed call had
   to land there too.

## The break that was there

`SparseMoE.forward` adds tie-breaking noise with `torch.randn_like`
(`model/experts.py:1613`) — the **process-global** torch RNG, with no state in
the checkpoint. Measured before the fix (tiny model, 6 steps, batch 2, seq 64,
lr 0.01, `moe_router_jitter=0.1`, 6 steps vs 3 + resume): the final loss moved
1.891312 → 1.865266 (0.026 nats) and `embed.weight` differed by
`max|Δ| = 6.761e-3` (`rel = 1.499e-4`). Two *fresh* runs of the same config
also disagreed, so the arm had never been reproducible — the resume merely made
it visible.

Fix: `loop.seed_router_jitter(seed, step)` (`loop.py`), called at the top of
every step when the live model carries `router_jitter > 0` — in both
`_train_impl`'s loop and `sft_train`'s. The draw is now a pure function of
`(seed, step)`, the rule ADR-0021 already states for `_sample_depth`, the
mixture draw and the LR schedule. The OFF path is untouched — the seed call
does not run, so every recorded number and every existing bit-exact pin is
unaffected (jitter's default is `0.0`). Rejected alternative: save the global
RNG state in the checkpoint — it is a process-global stream with per-device
keys, and it would silently couple the run to every other RNG consumer (and
still not make two fresh runs agree).

## What a resume carries today

| state | where it lives | restored by |
|---|---|---|
| parameters + persistent buffers (incl. `lfb_counts`, layerrope `theta`) | `ck["model"]` = `state_dict()` (storage rungs packed) | `load_ckpt` → `load_model_state` |
| `lfb_bias` (NON-persistent, not derivable) | `ck["moe"]` (`lfb_state`) | `_restore_lfb`; a file with counts but no bias is REFUSED (`_refuse_missing_lfb`) |
| host Engram tables, momentum, **step**, row counts | `<ckpt>.ngram` sibling (`engram.host_state`) | `load_host_state`, a missing/stale file is LOUD |
| optimizer: `steps`, Muon momentum + `sf_s`, AdamW `exp_avg`/`exp_avg_sq`/`step` | `ck["opt"] = opt.state_dict()` | `MixOptimizer.load_state_dict` |
| the fused AdamW's bias-correction counter `_adamw_t` | derived, not stored separately | `MixOptimizer.load_state_dict` reads one per-param `step` tensor (`optim.py`, the `steps[0]` read) |
| `ns_steps` on a resume (a PROGRESS key) | config | `opt.set_ns_steps(cfg.ns_steps)` after the load — the loader replaced the groups |
| step counter | `ck["step"]`; `start = step + 1` | `_train_impl`'s resume branch |
| config snapshot (everything but `PROGRESS_KEYS`) | `<ckpt>.config.toml` | `diff_keys` refuses any other difference (ADR-0021) |
| EMA shadow | `ck["model_ema"]` | `_shadow_state` (built after the resume load) |
| data position | **not stored** — recomputed | flat/shard: `start × (seq_len+1) × batch × grad_accum`; mixture: replay of the per-step draw (`mixture.py:631`); doc_masking: replay of packed steps (`data.py:693`); shard dir: modulo one epoch (`data.py:303`) |
| LR, QAT anneal, adaptive depth, expert revive, mixture stage/source | **not stored** — pure functions | `wsd_lr(step, steps)`, `anneal_alpha(step, N)`, `_sample_depth(seed, step)`, `revive_experts(seed+step)`, `stage_bounds`/`source_for_step(seed, step, steps)` |
| router-jitter noise | **not stored** — seeded per step | `seed_router_jitter(seed, step)` (this fix) |
| NaN window counters, `last_loss`, eval position | telemetry | reset by design: eval rewinds always; `loss` is `None`/NaN on a zero-step resume (`train`/`sft`); **the `drowning ≥ 2` stop counter restarts**, so a resume can outlive a guard that would have fired |

## The rule for a new feature (the checklist)

Four questions. A "no" to all four needs nothing; any other answer names where
the state goes.

1. **Does it change the next forward / backward / update?** Then it must be:
   a `state_dict` key; OR a named checkpoint section restored by
   `load_model_state` (the `moe` / `.ngram` pattern); OR a **pure function of
   `(seed, step)`** (the schedule / depth / mixture pattern). Nothing else.
   The gate: `tests/test_invariants_checkpoint.py` fails on an unclassified
   non-persistent buffer.
2. **Is it RNG?** Either save the generator state, or derive it from
   `(seed, step)` at the point of use. A module-level generator that advances
   across calls is a resume break by construction — do not add one silently.
3. **Is it data position?** It must be recomputable from `start`. Add the
   stream's `skip_*` case and prove it **at the trainer level** (a recording
   wrapper around the stream, continuous vs interrupted, tensor equality), not
   only on the stream in isolation.
4. **Is it telemetry?** Then it may reset, and the log/report must say what
   resets. A counter that guards the run (the NaN windows) is not telemetry:
   say it resets, or save it.

## Known non-exact — named, not silent

* **`ternary_s3t`** (`factor_qat`/`dense_qat`): the Bernoulli mask comes from a
  module-level `torch.Generator` seeded per model build
  (`model/tsct.py:295-345`); a resume restarts it, so the masks differ from an
  unbroken run's. It IS loud — the trainer prints the disclaimer on every run
  (`loop.py`, the `ternary_s3t` block) and `config.py`'s `FACTOR_QATS` doc owns
  it. Fix, if the rung earns it: a step index threaded into the projector (then
  the seed is `(seed, step)` like everything else), not a checkpointed RNG blob.
* **`--set steps=` extension of a `mix_plan` run**: `steps` is a progress key
  for the LR schedule, but the mixture's stage boundaries are
  `round(until_frac × steps)` (`mixture.py:341`) and `source_for_step` reads
  them. Extending re-plans the stages, so the already-trained prefix no longer
  matches the data draw the new plan would have made. The LR half is already
  covered as the stitched plan (`test_invariants_resume.py`); the data half is
  a different corpus schedule and is NOT protected. Also, the snapshot stores
  the plan's PATH, not its contents — editing the plan file between legs is
  undetected. Both matter only for extension; the fixed-budget guarantee holds.
* **CUDA-graph captures** (`opt_graph`/`full_graph`): not exercisable in this
  CPU lane. If a captured forward contains the jitter draw, replay semantics of
  the RNG need their own gate; do not assume this file covers them.
* **The `drowning ≥ 2` window guard** (table above): guard state, not the
  objective.

## Files

* `tests/test_resume_exact.py` — the gate.
* `src/dormouse/train/loop.py` — `seed_router_jitter`, the resume branch, the
  checkpoint sections.
* `src/dormouse/train/sft.py` — the same seed call (the SFT loop would otherwise
  carry the identical defect).
