# 2026-11-05 — the optimizer region: bf16 state, a bit-identical chunk cap, `ns_steps`

Follow-up to `docs/reviews/2026-11-05-kernel-profile.md`: the optimizer was the
largest region of the profiled m400 step (35.4 ms, 23.5 %), and the config still
ran `muon_state_dtype=fp32`, `ns_dtype=fp32`, `ns_steps=5`.

This lane is CPU-only (`CUDA_VISIBLE_DEVICES=""`; the card is owned by the live
multi-day run). Every number below is a CPU measurement of the CUDA launch shape
(`scripts/bench_muon_launch.py`'s `LaunchCounter`, which models `_foreach_*`
fusion and metadata) or a CPU training run. The card numbers are the ones the
owner runs at the next checkpoint gap; the commands are at the end.

## What landed

| file | change |
|---|---|
| `port/src/dormouse/config.py` | `PRESET_TRAIN_DEFAULTS`: m400/medium resolve `muon_state_dtype=bf16` and `muon_chunk_elems=1<<26`; `ns_steps` + `muon_chunk_elems` joined `PROGRESS_KEYS`; `validate_train` refuses non-integer `ns_steps`/`muon_chunk_elems` |
| `port/src/dormouse/train/optim.py` | `Muon(chunk_elems=...)` / `MixOptimizer(muon_chunk_elems=...)`, threaded through `_momentum_update_` and `_classic_ns_many`; `ns_steps` integer check; `MixOptimizer.set_ns_steps` |
| `port/src/dormouse/train/loop.py` | forwards `muon_chunk_elems`; re-applies `cfg.ns_steps` after `opt.load_state_dict` |
| `port/src/dormouse/train/sft.py` | same two lines for the SFT trainer |
| `port/scripts/bench_muon_launch.py` | the main-vs-new comparison uses the SAME chunk cap (the cap is a config value now, not a module constant) |
| `port/scripts/ab_opt_region.py` | the CPU 100-step gate and the card A/B driver |
| `port/tests/test_opt_region.py` | the gates below |

## The launch inventory, m400 Muon groups (262,247,424 elements)

`LaunchCounter`, one optimizer step on the real m400 parameter shapes, fp32 NS,
gradients materialised. The "base" arm is clean main's recipe.

| arm | state | chunk | launches | vs base |
|---|---|---|---:|---:|
| **base** | fp32 | 1<<25 | **890** | — |
| chunk26 only | fp32 | 1<<26 | 687 | **−203 (−22.8 %)** |
| bf16 state only | bf16 | 1<<25 | 950 | +60 |
| **preset (both)** | bf16 | 1<<26 | **731** | **−159 (−17.9 %)** |
| chunk27 | fp32 | 1<<27 | 621 | −269 (−30.2 %) |
| bf16 + chunk27 | bf16 | 1<<27 | 657 | −233 (−26.2 %) |

The chunk cap is the launch win and it is **bit-identical**: the greedy split is
over whole matrices, and the NS chain is per slice, so a cap change moves no
value (`test_the_chunk_cap_moves_no_value`; the 600×[1024,64] fixture drops
127 → 73 launches on the same test). The bf16 state costs +60 launches at m400
(one fp32 cast + one bf16 store per chunk) and buys the memory/bandwidth; the
cap pays for it and more.

The independent CPU aten-op counter (`scripts/launch_profile.py`'s
`DispatchCounter`, same real m400 step) agrees on the direction and reports
**3846 → 3679 non-metadata ops (−4.3 %), 0 host syncs**; it is a smaller drop
because it counts one `_foreach_*` call as one op where the CUDA model counts
its 64-tensor packing and 320-block split.

`medium` (the other preset that got the recipe): 559 → 526 launches (−5.9 %);
its groups are smaller, so the cap only bites on two shapes.

Why not 1<<27: at m400 the peak intermediate is ~6 × 268 MB ≈ 1.6 GiB for
1<<26 and ~3.2 GiB for 1<<27. The m400 model + grads + optimizer state is
~3.7 GiB on CPU accounting; 1<<26 is the conservative step, 1<<27 is left as an
explicit arm (`--set muon_chunk_elems=134217728`) for a card with headroom.

### Bytes

The Muon momentum is 262,247,424 fp32 elements = **976.9 MiB**; `bf16` stores
**488.5 MiB**, a saving of **488.5 MiB** at m400. (The b1 fixture in
`tests/test_muon_fast.py` pins the same claim at 2.46 → 1.23 GiB.)

## The bf16-state parity gate — measured, not asserted

`python scripts/ab_opt_region.py gate --steps 100` (CPU, `tiny`, b2×s64, the
trainer's own grad clip, fixed data, seeds 0 and 1):

```
[gate] tiny 100 steps: max rel loss 0.000357; fp32 seed spread 0.03447;
       worst parameter delta 9.68 bf16 ulp at scale
[gate] losses fp32[0,-1]=6.296806,5.543108  bf16[0,-1]=6.296806,5.543281
[gate] VERDICT: the state rung is inside the fp32 seed spread (a quality tie).
```

The gate draws a FRESH batch every step from a generator independent of the
model seed. A fixed batch overfits to ~0 loss inside 100 steps and the relative
comparison becomes 0/0 noise (measured: a fixed batch on `medium` ended at
2.2e-4 vs 1.1e-2, a "37883×" ratio that is two numbers next to zero, not a
quality signal).

**The strict brief's numbers are impossible and this report says so:**

* **loss within 1e-5 relative: NO.** The 100-step curve moves 0.036 % at its
  worst point. That is 97× smaller than the fp32 arm's own seed spread (3.45 %),
  which is the repo's tie rule, but it is still 36× above 1e-5.
* **parameter deltas within bf16 ulp: NO, if "ulp" means per element.** The
  worst per-tensor delta is 9.68 bf16 ulp at the parameter's scale; per-element
  ulp is meaningless near zero and fails by hundreds of ulp on any tensor. A
  1-ulp gate would need the stochastic rounding the rung's docstring already
  names as the next rung; it is not implemented here.

The gate is therefore the **seed-spread** one: the rung is a quality tie at
100 steps, and the pins (`GATE_LOSS_REL`, `GATE_PARAM_ULP` in
`tests/test_opt_region.py`) make any drift in the rung a red test. The script
takes `--preset medium` for a larger CPU run; it is not part of the suite and a
100-step medium run costs ~30 minutes on this box's CPU.

### Which snapshot can resume

The m400b checkpoint on disk was written with `muon_state_dtype = "fp32"`.
`muon_state_dtype` is intentionally **strict** (a state-precision flip changes
the next step's arithmetic), so resuming that run on this commit is a LOUD
refusal unless the old value is named:

```
--set muon_state_dtype=fp32 --set muon_chunk_elems=33554432
```

`muon_chunk_elems` is a progress key (the split is bit-identical) and does not
need the override; the state dtype does. A fresh m400/medium run gets the new
recipe with no flags.

## `ns_steps` as a progress key

* `ns_steps` round-trips through the snapshot, `diff_keys` ignores it, and a
  non-integer (`3.5`, `0`, `True`, `"3"`) is refused at `load_run` AND at the
  `Muon` constructor (`test_ns_steps_round_trips...`).
* torch's `Optimizer.load_state_dict` replaces `param_groups`, which would
  revert the key to the checkpoint's value. `train/loop.py` calls
  `opt.set_ns_steps(cfg.ns_steps)` after the load; the full
  `train(resume=True)` seam is gated by
  `test_a_resume_with_a_new_ns_steps_reaches_the_live_muon_group` (red on a
  scratch revert of that call).
* **The default is NOT flipped.** 5 is still every number on record. The
  measured-parity driver is `scripts/ab_opt_region.py ab`; the `state_bf16` vs
  `ns3` arms are the 5-vs-3 comparison (2000 steps, same seed/data/state/chunk),
  and the driver prints both held-out BPBs plus the delta and the control's
  seed spread. The book is a tie: the cited 3-seed study is non-monotone with a
  <0.003 BPB spread. A 5→3 flip would cut ~224 launches at m400 (2 fewer
  iterations × 14 batches × 8 ops) and 40 % of the NS kernel time; it waits for
  the driver's BPB.

## Item 3 — overlap: NOT implemented, and why

`micro_step` calls `loss.backward()`, `grad_norm` and `_gate_grads` on the
current (default) stream, then `opt_stage()` calls `clip_grads` + `opt.step`
on the SAME stream, from the SAME Python thread (`train/loop.py:1483-1510`,
`train/loop.py:1770-1795`). CUDA kernels on one stream execute in issue order,
so the optimizer's kernels cannot start before the last backward kernel
finishes. There is no per-parameter-group dependency seam to hook: the
`grad_norm` pass and the clip are global over all gradients, and the clip's
coefficient depends on every group.

Making it overlap needs: a second stream; one event recorded after each shape
group's last gradient producer (requires autograd hooks or a per-group
backward split, because `backward()` is one call); the clip's global norm
deferred (it needs ALL groups, so the clip itself cannot overlap); and the
writeback ordered after the group's event. That is a real determinism risk:
cuBLAS/cuDNN workspace and allocator behaviour differ across streams, the
clip's coefficient would be read on one stream while another still writes, and
`opt_graph` capture (one stream) would have to be refused. It is not a
stream+event change; it is a per-group graph split. Reported, not implemented.

What *is* already true: the optimizer's **host** dispatch overlaps the GPU
backward for free (the CPU enqueues while the card is busy), and the whole
stage is capture-able as one graph (`opt_graph`), which removes the host gaps
without touching determinism. The 50 % host-gap number is the argument for
`opt_graph`, not for a second stream.

## Item 4 — per-parameter Python work: what was actually left

The writeback's per-parameter `.detach()` list is **already gone** for the
default same-dtype path (`_foreach_add_mixed_` uses `torch.unbind` views and
`_foreach_add_`; `optim.py:758-790`). The mixed-dtype stack path is only for
bf16 *parameters*, which the m400 run does not use (fp32 masters). The "gram
lists" are per **chunk**, not per parameter: 14 batches at m400, not 1 556
lists. The remaining per-parameter Python is the group construction (once) and
the `any(p.grad is None for p in params)` scan (per group, not per step
work). The launch census is therefore the honest instrument, and the cap is
where the launches were: 890 → 731.

## Revert evidence (each gate red on a scratch revert)

| revert | result |
|---|---|
| drop the preset-defaults application in `load_run` | `test_the_measured_presets...` red: `bf16` vs `fp32` |
| drop `"ns_steps"` from `PROGRESS_KEYS` | `test_ns_steps_round_trips...` red: `train.ns_steps` no longer exempt |
| drop `opt.set_ns_steps(cfg.ns_steps)` in `loop.py` | resume gate red: the live group kept 5 |
| ignore `chunk_elems` in `Muon` (`self.chunk_elems = MUON_CHUNK_ELEMS`) | chunk test red: `launches 127 -> 127` |
| make `_torch_dtype` always fp32 | bf16 gate red: `max rel loss 0`, pins do not hold |

The default path's byte pins are untouched:
`tests/test_muon_launch.py`'s `GOLDEN_DEFAULT` / `GOLDEN_FALLBACK` /
`GOLDEN_MIX` pass, and its launch pins (317 → 192, 358 → 232) are unchanged
because the schema cap and NS chain are unchanged.

## Suite and lint

* `CUDA_VISIBLE_DEVICES="" PYTHONPATH=port/src pytest -q`: **1 failure**, the
  pre-existing `test_sft_v2::test_the_manifest_accounts_for_every_row` — the
  untracked `sft_pairs/` corpus a clean worktree does not carry (the main
  worktree has it, this worktree does not; the failure is a missing file, not a
  number). `test_step_audit` was red for the new `set_ns_steps` loop and is
  fixed by excusing it as a resume-only setup loop.
* `ruff check` on every changed file: clean.

## The GPU commands for the next checkpoint gap

Profile the new recipe and diff it against the pre-change JSON (the profiler
refuses while the card is held; run it in a gap):

```
cd /home/sehaxe/dormouse-wt/opt-region/port
uv run --extra gpu python scripts/kernel_profile.py \
    --checkpoint /home/sehaxe/dormouse/checkpoints/m400b/m400b.pt \
    --json /tmp/kprof_opt_after.json
uv run python scripts/kernel_profile.py \
    --diff /tmp/kprof_m400b.json /tmp/kprof_opt_after.json
```

Quality + step time (one process per arm; 2 000 steps, b8×s1024, the preset
recipe; `state_bf16` vs `ns3` is the 5-vs-3 parity pair):

```
cd /home/sehaxe/dormouse-wt/opt-region/port
uv run --extra gpu python scripts/ab_opt_region.py ab \
    --arms base,state_bf16,ns3 --steps 2000 --preset m400 --batch 8 --seq 1024
```

For a 3-seed verdict add `--seeds 1,2,3`; for the second precision rung add
`state_ns_bf16` to `--arms`. The driver refuses while another process holds the
card (`scripts/_lane.py`) and prints the two `ns_steps` BPBs plus the delta and
the seed spread; it does not apply the verdict.
