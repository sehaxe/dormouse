# torch.compile — what works, what is refused, and how to measure it

`train.torch_compile` wraps the model in `torch.compile(model, dynamic=True)`
inside `train()` (`src/dormouse/train/loop.py`), **after** the resume load: an
`OptimizedModule`'s `state_dict()` carries a `_orig_mod.` prefix, so loading
into the wrapped model would rename every key. The flag is off by default and
its snapshot pins it. Nothing else about the run changes: the optimizer is
built before the wrap on the same `Parameter` objects, and the checkpoint
format is the unwrapped tree either way.

The suite's gate for this file is `tests/test_torch_compile.py`; every "pinned"
row below names the test that fails if the row stops being true.

## What works

| seam | behaviour | pinned by |
|---|---|---|
| checkpoint `model` | `save_ckpt` strips `_orig_mod.`; the file is the unwrapped tree | `test_a_compiled_checkpoint_is_unwrapped_in_all_three_sections` |
| checkpoint `model_ema` | the EMA shadow is **checkpoint-keyed** (`_shadow_state`), so the served half has the same keys as `state_dict()` | same test |
| checkpoint `moe` | `lfb_state` strips the prefix per module (`_lfb_modules`), so the LFB section names the unwrapped router | same test |
| load | `unpack_state_dict` also strips, so a pre-fix file with the prefix still loads | `tests/test_ckpt_prefix.py` |
| resume across the boundary | `torch_compile` is a `PROGRESS_KEYS` entry: a checkpoint written compiled can be continued eager and vice versa, same step count and optimizer state | `test_resume_across_the_compile_boundary` |
| EMA eval | `_shadow_params` swaps `.data`; a compiled graph reads each `Parameter` by object at call time, so the swap reaches the compiled forward | `test_the_ema_shadow_is_checkpoint_keyed_and_reaches_the_compiled_forward` |
| Muon / AdamW | the optimizer is built before the wrap and holds the same `Parameter` objects the compiled forward reads; routing names stay unprefixed | `test_the_optimizer_steps_the_same_parameters_the_compiled_model_reads` |
| arm counters | `iter`/`forward`/`engram_*` are Python writes between recorded regions, so they keep moving; a captured counter would go SILENT | `test_a_compiled_run_keeps_its_counters_and_eval_line` |
| eval | the eval forward and its own `Counters` work on the wrapper; `engram=<rows>/<arms>` stays non-zero | same test |
| numerics | compiled forward **and** backward match eager at fp32 round-off | `test_compiled_forward_and_backward_match_eager_at_fp32_round_off` |

The parity test uses `backend="eager"` because the guard/prefix/counter
machinery is dynamo's and backend-independent (`tests/test_compile_moe.py`
makes the same choice); a full inductor compile of the tiny fixture costs about
90 s of the suite. The default inductor backend on this box (torch
2.13.0+cu129, CPU, 2026-11-08) read the same tiny loss bit-for-bit in a manual
probe; that probe is not a committed measurement.

## Eval shapes and the recompile budget

The eval window is `eval_batches × batch × seq_len` and `eval_batches_for`
**refuses** a window that is not a whole multiple of `batch × seq_len`
(`src/dormouse/train/data.py`), so every eval batch has the training shape.
`dynamic=True` cannot recompile on an eval shape that never changes; this is
pinned by `test_eval_windows_are_the_training_shape`.

What still costs one compiled variant each, on the *same* frame:

- the eval call signature (`pure=True, breakdown=True`, no `depth`) vs the
  training one — one extra variant, once;
- a new integer `depth` (`--eval-depths`, `--rand-depth`): dynamo guards Python
  ints on their exact value. The live dense preset (`flagship3_dense`) runs
  `max_iter = 1`, so depth adds no training variants; the b1 preset's
  `max_iter = 4` would still be four variants against
  `torch._dynamo.config.recompile_limit = 8` — legal. Do not combine
  `--rand-depth` with `--eval-depths` **and** more than ~7 distinct depths on a
  compiled run: that is the documented way to spend the budget and fall back to
  eager.

The first call also pays the graph-break segments (the model has deliberate
breaks: `SparseMoE._banks`, `QuantCache.refresh_stacked`,
`SparseMoE._naive_dispatch` are `@torch.compiler.disable`), and the warmup is
once per machine — inductor's cache lives in `~/.cache/torch`.

## What is refused

- **`full_graph=true` + `torch_compile=true`** is LOUD in
  `config.validate_train`: a recompile attempted while a CUDA graph capture is
  active aborts the capture, and the graph already removes the host dispatch
  the compiler attacks.
- **`opt_graph=true` + `torch_compile=true`** is LOUD in `train()` before the
  wrap (`train/loop.py`).
- `train/graph.py` documents the same refusal from the trap side; both captures
  are whole-step or one-stage CUDA graphs, and a compile must not run under
  either.

Both refusals name their `Escape:`; there is no silent fallback.

## Resuming across the compile boundary

The checkpoint never records how the forward was executed, so both directions
are the same command — flip the flag on the resume:

```bash
# uncompiled run -> compiled continuation
uv run dormouse train <preset> --data ... --eval ... --resume \
    --set torch_compile=true

# compiled run -> eager continuation (revert an arm that did not pay)
uv run dormouse train <preset> --data ... --eval ... --resume \
    --set torch_compile=false
```

`--set torch_compile=...` reaches the resolved config, the snapshot gate
exempts it as a progress key, the load happens before the wrap, and the
optimizer state continues from the same step. The `.pt` written by either
direction has the same `model`/`model_ema`/`moe` key sets.

## Speed: measured elsewhere, not at the live shape

The compile arm's only POSITIVE reading on record is the 2026-10-15 b1 spike
(`docs/records/research/2026-10-15-torch-spike.md:228-230`, torch 2.13.0+cu129,
RTX 5060 Ti): eager **357.3 ms/step**, `dynamic=False` **298.0** (1.20x), and
the `dynamic=True` arm this trainer wraps **273.5** (1.31x). The claims ledger
(`docs/records/reviews/2026-11-07-claims-ledger.md`) lists that spike's numbers as
UNVERIFIABLE outside the copied report — the raw logs lived in `/tmp` — so read
them as the spike's reading, not as a promise for this recipe.

**No compiled number exists at the live dense shape** (`configs/flagship3_dense.toml`,
b8 × 1024). That measurement is the orchestrator's, and the instrument is
`port/scripts/ab_compile.sh`: two 600-step arms on the same recipe, eval off,
`log_every=100`, compared on the **last logged** `tokens_per_s` (the steady
state after the compile warmup, never the mean, which a 1-2 minute warmup would
dominate):

```bash
cd /home/sehaxe/dormouse/port
systemd-run --user --scope -p MemoryMax=40G \
    env OUT=/home/sehaxe/logs/ab_compile STEPS=600 bash scripts/ab_compile.sh
```

The script prints the last log line of each arm side by side; `$OUT/eager.log`
and `$OUT/compiled.log` carry the full lines, the JSONL the per-window rates.
One GPU process at a time (`AGENTS` §1.5); the card must be quiet and the run
must be under the memory cap.

**Expect the result to be a measurement, not an extrapolation.** This port has
already paid for a compile assumption once: before the `@torch.compiler.disable`
decorator on `SparseMoE._banks`, `torch.compile(dynamic=True)` was reported at
**0.20x** (2939 ms/step against 573 eager — the trigger recorded in
`docs/records/research/2026-10-18-compile-moe.md`, not re-measured there) because a
per-step Python value in the cache key spent the recompile budget and dynamo
fell back to eager; the decorator and its control are pinned by
`tests/test_compile_moe.py`. If the live-shape arm is not faster, read the
recompile warnings on stderr (`TORCH_LOGS=recompiles`) and
`scripts/compile_retry.py`'s graph/recompile counters before touching the
recipe; a compiled arm that recompiles per step is a slower eager run with a
warmup attached.

## Not covered by this gate, and not claimed

The suite runs CPU only; the inductor GPU backend and the live shape are the
orchestrator's measurement. Above it, these arms are **not** pinned with
`torch_compile`: `engram_ram` (the host path does a device→host row transfer
inside the forward, so it graph-breaks and the compile lever is limited),
byteflow, CSA, MSA, JEPA/DSpark, activation offload and the QAT arms. They are
not refused by the compiler either — an un-capturable host read is a graph
break, not a fatal error — but a break that carries a per-step Python value can
recompile every step, and nothing here says the combination pays; measure it as
its own arm.
