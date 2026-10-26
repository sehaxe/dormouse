# Bug ledger — 2026-10-26 (second debug pass)

> **All eleven findings below are FIXED and merged** (branch `debug-oct26`):
> each fix ships with a gate that fails on revert, and the gate list is in the
> merge commits. The reproductions are kept as written — they are the evidence.
Two read-only hunters, disjoint scopes, every finding required to carry `file:line` and a
repro that was actually executed. Findings the hunters discarded themselves are listed at the
end, because a discarded finding is also a result.

Status values: `OPEN` (fix in flight), `FIXED` (merged with a gate that fails on revert),
`DISCARDED` (the hunter falsified its own finding), `DOCUMENTED` (named limitation, not a bug).

---

## CRITICAL

### C1 — `lfb_bias` is not in the checkpoint: every resume of a shipped preset trains a different router

`train/loop.py:225-230` (payload), `model/experts.py:828-837` (the buffer).

```
lfb_counts  persistent=True    -> saved and restored
lfb_bias    persistent=False   -> neither saved nor reconstructible
```

`lfb_bias` is the integrator *output* (`_lfb_step`: `bias -= coef*sign(counts-mean)`,
`experts.py:984-991`); it cannot be derived from the restored counts. A resume restarts it at 0
while the unbroken run is already at +-0.08.

Measured (tiny_run, `decay_frac=0`, 8 steps unbroken vs 4 + resume 4):

| `moe_lb_coef` | tensors differing |
|---|---|
| 0.0 | **0 / 51** (resume is bit-exact) |
| 0.01 | **50 / 51** |

Every shipped preset sets `0.01` (`configs/small.toml:95`, `medium.toml:44`, `b1.toml:75`,
`small_wide.toml:45`). The same hole means eval/decode/serve load a bias-0 router, i.e. a
different model than the run trained.

This is the defect `init_controller` was written against (`loop.py:361-417`: cold integrator over
a converged router -> loss 1.275 -> 11 425 785 by step 300). The comment on `lfb_counts`
("a resume must keep its controller state") names only half the controller.

**Fix:** write `lfb_bias` per module beside `model`/`opt`; on resume refuse a file that carries
`lfb_counts` but no `lfb_bias` when `moe_lb_coef != 0` (refuse, never guess).
**Status:** FIXED.

---

## MAJOR

### M1 — `mix_plan` + `grad_accum > 1`: the stage schedule and the epoch cap are both off by `grad_accum`

`train/loop.py:787-794` builds the mixer with `steps=` (optimizer steps); `train/loop.py:1319-1325`
consumes one batch per micro-step; `train/mixture.py:529,538` key the schedule off that counter.

Measured (`steps=100, batch=2, seq=8, grad_accum=4`, the trainer's own call pattern):

```
plan stage bounds (optimizer steps): (0, 70, 90, 100)
observed transitions:               S2@18, S3@23      <- 4x fast
epoch cap: guard says code 3.52x, the loop reads 14.55x
```

The decay stage — the one the recipe calls the lever — runs over ~1/4 of its planned span, and
the "over max_epochs is a LOUD refusal" promise does not fire.
**Status:** FIXED.

### M2 — the SFT in-loop eval never rewinds its stream

`train/sft.py:1198-1206` (no `eval_stream.rewind()` anywhere in the file; the pretrain loop has it
at `loop.py:612`).

Measured with `lr=1e-12` (weights cannot move): CE moves **0.0106 nats** between evals when the
window is smaller than the holdout; identical to 4 dp when the window covers the whole holdout.
That noise is 3-7x the project's quoted paired-eval resolution (0.002-0.005 BPB).
**Status:** FIXED.

### MAJOR (model path) — the NVFP4 two-level prescale is dead: small blocks are silently zeroed

`model/tsct.py:744-745`.

```python
744:    float(torch.finfo(torch.float8_e4m3fn).tiny)          # 2^-6, computed and discarded
745:    min_normal = float(2.0 ** torch.finfo(torch.float8_e4m3fn).min)   # 2^-448
```

Measured: `float(2.0 ** finfo.min)` = `1.3758e-135`; the real min normal is `0.015625`. The guard
`if scale_min < min_normal` (`:753`) therefore never fires, `global_scale` is always 1.0, and a
block whose `amax/6` underflows e4m3 becomes scale `0.0` -> all 16 weights decode as `0.0`.

Repro (2 nvfp4 blocks, amax 0.004 and 0.30):

```
block scales [0.0, 0.05078125]
source nonzeros [1, 1]  ->  decoded nonzeros [0, 1]
```

This is the checkpoint path: `loop.py:222` packs every factor with `run.model.quant`, so
`--quant nvfp4` saves lose data.
**Fix:** `min_normal = float(torch.finfo(torch.float8_e4m3fn).tiny)`.
**Status:** FIXED.

---

## MINOR

### m1 — `muon_sf` writes Y with two different roundings

`train/optim.py:1153` (`lerp_`) vs `:1406-1407` (`(1-beta)z + beta x`). Measured 1 ulp
disagreement on random fp32 (`max|d| 5.96e-08`), 37/51 tensors differing on a resume of an SF run
while the non-SF control resumes with 0/51. The test that claims equality
(`tests/test_muon_variants.py:574-598`) is vacuous on its fixture (`z == x == 0.25`).
**Status:** FIXED.

### m2 — the eval tail is silently capped at 1 MiB

`train/data.py:454`: `min(max(eval_window_bytes(tr), 1), 1 << 20)`.
Measured at `eval_batches=100, batch=16, seq=1024`: requested 1 638 400 B, stream 1 048 576 B,
batch 15 byte-identical to batch 0 — the tail wraps and `scored` keeps counting duplicates, so
the printed `over N B` is not the number of distinct bytes scored. Silent (no counter).
**Status:** FIXED.

### m3 — the SFT/RL "copied" NaN firewall checks the loss, not the gradient

`train/sft.py:1113-1121`, `post/rl_train.py:556-563` vs `train/loop.py:1107-1146` (which computes
`grad_norm`, ANDs `isfinite` and gates every gradient). A finite loss with a non-finite gradient
poisons `exp_avg_sq` permanently; the model then stays NaN. Consequence reasoned, code gap read
directly.
**Status:** FIXED.

### m4 — `--seed` does not seed the model initialisation

`train/loop.py:730-802`, `train/sft.py:926-971`, `post/rl_train.py:369-374` never call
`torch.manual_seed`. Measured: two models, same config and seed, `embed.weight` differs by 5.386.
`config.PROGRESS_KEYS` pins `seed` in the snapshot for a value that never reaches the weights.
**Status:** FIXED.

### m5 — SFT and RL silently ignore train fields they do not wire

`sft.py:993-1001`, `rl_train.py:391-399` omit `muon_state_dtype`, `ns_dtype`, `muon_variant`,
`muon_sf`, `muon_sf_wd`; the SFT loop reads one batch per step regardless of `cfg.grad_accum` and
never calls `validate_train`. Measured: `grad_accum=4` gives the identical loss to `grad_accum=1`.
**Status:** FIXED.

### m6 — `loop.mem_dense` is a dead parameter when `use_engram=false`

`model/loopblock.py:280` builds it unconditionally; `:585-607` reads it only inside the engram
branch. Measured `.grad is None` under `use_engram=false`. At `d=1024` that is **1 049 600**
parameters in `param_count`, in the optimizer's group split, with no counter saying it is inert.
`configs/final_patch.toml` ships exactly that configuration.
**Status:** FIXED.

### m7 — `act_quant._check` forces a device sync on every quantizer call

`model/act_quant.py:240`: `if not bool(torch.isfinite(x).all())`. Measured **17 `quantize_act`
calls -> 17 syncs** in one tiny training forward (ADR-0018 rule 2 forbids a host read in the hot
path; the module's own docstring assigns NaN policy to the training step's firewall).
**Status:** FIXED.

---

## SUSPICION (reasoned, not reproduced)

- `full_graph` refusals are documented as "before the first step" (`loop.py:1217-1226`) but
  `muon_sf` / non-default `muon_variant` are refused inside `OptimizerGraph._refusals`, i.e. after
  `GRAPH_WARMUP_STEPS = 5` eager steps that already ran with a device-bound LR
  (`graph.py:293-309,393-403`). Confirm on CUDA.
- `Muon._step_one` vs `_foreach_add_mixed_` differ by 1 ulp on bf16 master params; no shipped
  config has a singleton bf16 group.

---

## DISCARDED by the hunters themselves

- `loop.residual_scale` gets no gradient under `use_mhc`/`use_attnres` — already named and COUNTED
  in `train/graph.py:168,336`.
- A suspected bf16 rounding mismatch in `_glu_backward_fused` — the multiply is bf16 x bf16 on both
  sides, so it is exact.

## Verified clean (so the silence is informative)

`engram.slot_keys_gpu` vs the burn host reference 0/222 mismatches; CSA fine level 2.2e-8 against a
brute-force reference, coarse level 0.0 against dense attention; prefill->decode 5.4e-4 / 1.1e-3 on
an output RMS of 0.10 / 0.20 (the bf16 ring's own 2^-8); MoE padded vs naive forward 2.4e-7,
gradients 9.0e-7 including the capacity-drop cases; `_bincount` == `torch.bincount`; every causal
mask tested has no future leak; the quantizers are idempotent; resume is bit-exact for default,
`grad_accum=2`, warmup+decay, `rand_depth`, `ema_decay=0.9`; the LR schedule has no off-by-one;
gradient scaling is correct.

## Not covered by this round

GPU-only paths: the FLA `chunk_kda` / `fused_recurrent_kda` arm, the Triton MoE and NS kernels,
`mm_fp8`, SDPA kernel selection, CUDA-graph capture/replay. `sparse/` (MSA), `decode.py`,
`moe_sonic.py`. Distributed code does not exist in this tree (see the coverage note).
