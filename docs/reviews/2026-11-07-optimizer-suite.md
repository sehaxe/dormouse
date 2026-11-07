# 2026-11-07 — Optimizer suite: what every knob promises, and what it measures

Hostile read of every training knob the 2026-11-07 Muon audit did not already
clear, against a numeric fingerprint written in
`port/tests/test_optimizer_suite.py` (34 gates, CPU only,
`CUDA_VISIBLE_DEVICES=""`). The reference for each gate is implemented in the
test — `torch.optim.AdamW`, a written-out WSD closed form, a hand-transcribed
AdamW/EMA/schedule-free recurrence — not imported from the code it judges.

The audit's question is the ruler: after `muon_lr_mult`, which knobs are
**DEAD** (the value never reaches the arithmetic), **MIS-SCALED** (it reaches
the arithmetic as a different number), or **LIES** (the code does not do what
the field/docstring says)? Five more silent wiring defects were found and fixed
here; three boundaries and proposals are ranked at the end.

## Verdicts

| # | knob | promised | measured | verdict |
|---|---|---|---|---|
| 1 | AdamW (`adam_eps`, betas, wd) | bias-corrected AdamW | bit-identical to `torch.optim.AdamW` over 5 steps; step 1 and 2 are `lr·sign(g)` on a constant gradient; wd is decoupled; eps sits after the bias-corrected sqrt | **HOLDS** — boundary: late-joining params (§1.5) |
| 2 | `grad_clip` | global-norm clip over all groups, before the step | norm 50 → 1 with one coefficient over both halves; pre-step order pinned; first-step update ratio **1.0000**, multi-step 1.10 | **HOLDS** (it cannot shrink a step — §2.3) |
| 3 | `agc_clip` | per-unit adaptive clip, off = identity | per-**parameter** ratio `λ‖p‖/max(‖g‖,ε)` (documented deviation from per-row); scales 0.005 / 1.0 / 0.0; bit-identical when it never fires | **HOLDS AS DOCUMENTED**; **DEAD in SFT/RL** → now refused |
| 4 | `ema_decay` | analytic EMA, raw + EMA checkpoint, resume | sequence 0.1, 0.29, 0.561, 0.9049, 1.31441 vs closed form 1.31441; `model_ema` is the shadow, `model` the raw; 3+resume == 5 unbroken, bit for bit | **HOLDS**; **DEAD in SFT/RL** → now refused |
| 5 | `muon_sf` / `muon_sf_wd` | Z/X averaging, params hold Y, wd>0 at Z | reference recurrence matches; `sf_switch` returns X/Y exactly; both config and constructor refuse wd≤0 | **HOLDS** |
| 5b | `muon_sf_retract` | per-step TSCT retraction | `true` is a refusal; no `retract_tsct` call site exists | **DEAD** (honest refusal; keep it, don't wire it) |
| 6 | WSD (`warmup`,`decay_frac`,`min_lr_frac`) | warmup → hold → decay to the floor | last 100 steps match the closed form to 1e-12; last step = `peak·min_lr_frac`; resume recomputes from the global step | **HOLDS** |
| 7 | `z_loss` / `logit_softcap` | `coef·mean(LSE²)` / `cap·tanh(z/cap)` | delta 0.03663683 vs term 0.03663683; softcap formula exact; both bit-identical off | **HOLDS** |
| 7b | QK-clip | — | no field, no function, no call site in `port/src` | **ABSENT** (not a dead knob) |
| 8 | bf16 mixed precision | fp32 masters, fp32 update, bf16 storage | all 50 tiny-model params fp32 before/after a step; grads fp32; Muon bf16 momentum = one rounded fp32 accumulation per step | **HOLDS** |
| 9 | `muon_lr_mult` | `lr × mult` reaches Muon | end-to-end 1-step ratio **8.0x**; **DEAD under `full_graph`** (fixed) | **HOLDS**; sub-arm fixed |

Extra findings (not in the eight), all **DEAD** and all fixed or refused:

| knob | arm | effect before | fix |
|---|---|---|---|
| `adam_eps` | `sft_train`, `rl_train` | silently dropped → torch's 1e-8 always ran | forwarded (`sft.py:1139`, `rl_train.py:441`) |
| `agc_clip` | `sft_train`, `rl_train` | accepted, never called | refused in `refuse_unwired` |
| `ema_decay` | `sft_train`, `rl_train` | accepted, no shadow, no `model_ema` half | refused in `refuse_unwired` |
| `muon_sf_retract` | hand-built `RunConfig` | refusal lived in `load_run` only; `train()` accepted `true` and ran it inert | moved into `validate_train` |
| `muon_lr_mult` | `full_graph` | `FullGraph.set_lr` wrote ONE value into every group | `opt.set_lr(lr, muon_lr)` after `full.set_lr` (`loop.py:1820`) |

---

## 1. AdamW

Fingerprint: `test_adamw_*` (6 gates). The fused writer
(`optim.py:2080`) reproduces `torch.optim.AdamW` **bit for bit** over 5 steps
at `lr=1e-3, wd=0.1, betas=(0.9,0.95), eps=1e-8` — `torch.equal` on every
parameter every step, not a tolerance. That single equality clears the whole
family: decay order, moment recurrences, both bias corrections and the eps
placement.

### 1.1 First-step magnitude and bias correction

Constant gradient `g=[1,-1]`, `lr=1e-3`: the measured deltas are
`[-1.0000000475e-3, +1.0000000475e-3]` at step 1 and **the same at step 2**,
i.e. `lr·sign(g)` — the bias correction makes a constant-gradient step
size-invariant from the first step. The mutation that drops it makes step 1
`4.4721e-4` (2.236x low) and step 2 a different size again: three gates red.

### 1.2 Weight-decay decoupling

With `g=0`, `lr=1e-2`, `wd=0.1`, ten steps leave `2.0 → 1.9800900` against the
closed form `2·(1-1e-3)^10 = 1.9800898` (float32 round-off). The parameter
moves **with a zero gradient** — that is decoupled decay; an implementation
that puts `wd·p` into `g` cannot move a zero-gradient parameter at all. The
ignore-wd mutation reddens this gate and the torch-equality gate.

### 1.3 `eps` placement

`g=[1e-6, 2e-6]`, `lr=1`, `eps=1e-6`: the measured step is
`[-0.5000, -0.6667]`. The written reference
`-m/bc1 / (sqrt(v)/sqrt(bc2) + eps)` gives the same; the plausible wrong form
`-m/bc1 / (sqrt(v) + eps)` gives `[-0.8173, -1.0898]` (63% deeper). The gate
asserts equality with the correct form and **non**-equality with the wrong one,
so a future refactor that moves eps across the bias correction is caught.

### 1.4 The LR actually applied is `lr × schedule`

The Muon failure was a group LR overwritten every step. On the AdamW half:
step once at `lr=1e-3`, call `MixOptimizer.set_lr(2e-3)`, step again with a
constant gradient — the second delta is exactly `2e-3`. The same test asserts
the Muon group receives `2e-3 × muon_lr_mult` (8x → `1.6e-2`), and the
tensor-LR variant asserts `set_lr` fills the 0-dim tensor **in place** (the
graph arms read the object, never a rebound float).

End-to-end: one real `train()` step at `muon_lr_mult=1` vs `8` scales the
`loop.attn.q_proj.weight` delta by **7.99x** — the audit's killed knob now
reaches the arithmetic through the trainer.

### 1.5 The boundary: a late-joining parameter

`_foreach_adamw_step` uses **one** `_adamw_t` for the whole optimizer, while
`torch.optim.AdamW` tracks a `step` per parameter. If a parameter has no
gradient at step 1 and one at step 2, torch applies bias correction at t=1 for
that parameter; the fused writer applies t=2. Measured: `b=0.0073496` vs
torch's `0.01` (26.5% low) after one missed step. This is unreachable on the
live trainer's AdamW groups — a forward/backward of the tiny model leaves every
AdamW parameter with a gradient (the one missing param,
`loop.attn.beta_proj.weight`, is Muon-routed and handled per-parameter) — but
the docstring's "bit-identical" claim has that precondition. Fixing it properly
means per-parameter bias correction, i.e. the host step read the fused writer
exists to delete; **proposal**, not a fix: name the precondition in
`_foreach_adamw_step`'s docstring, or add a per-param fallback for the step a
new state appears.

---

## 2. Global-norm gradient clip

Fingerprint: `test_clip_*` (4 gates).

### 2.1 It is one norm over every group

Gradients of norm 30 (Muon side) and 40 (AdamW side): `clip_grads` returns
**50.0**, applies one coefficient **0.02**, and the post-clip global norm is
`0.99999988`. A per-group clip would put each group at 1.0 and the global norm
at `sqrt(2)`; the gate asserts the single global norm, and that both groups
received the same ratio.

### 2.2 It fires only above the threshold

Below `max_norm`: `torch.equal` with the input (not merely close). Above:
gradient norm exactly `max_norm`; the coefficient is
`max_norm/(total+1e-6)`. A missing `clamp(max=1)` reddens the below-norm gate.

### 2.3 Before the step, not after the update

Two independent gates:

* numeric — `clip_grads(params, 0.5); opt.step()` is bit-close to feeding a
  fresh optimizer the same gradients pre-scaled by `0.5/(total+1e-6)`. Only a
  clip on `.grad` (before the moment update and bias correction) can be equal;
* wiring — a 1-step `train()` with spies records `["clip", "step"]`. Moving
  the call after `opt.step()` records `["step", "clip"]` and reddens it.

### 2.4 The ratio of clipped to unclipped updates

| step pattern | AdamW clipped/unclipped RMS | Muon (single step) |
|---|---:|---:|
| first step, `‖g‖=50 → 0.5` | **1.0000001** | **1.0000000** |
| alternating 50 / 0.05, step 4 | **1.1041** | — |

The first-step ratio is 1 because both update rules are positively homogeneous
of degree 0 in the gradient (`m/√v` and `ColRow(NS(m))` are scale-invariant).
**Clipping cannot shrink the first step of either half**; it changes the
trajectory only once the coefficient varies across steps (1.10 here). A clip
applied to the *update* would have shown ratio 0.01 (0.5/50); that is the
falsifiable difference.

---

## 3. AGC (`agc_clip`)

Fingerprint: `test_agc_*` (1 gate here; `tests/test_lrband.py` holds the
launch-count and order gates).

* **Unit = whole parameter tensor**, not the paper's conv row. The docstring
  states the deviation (`optim.py:2327`); the gate measures the tensor form:
  `p=1¹⁰, g=100¹⁰, λ=0.5` → scale **0.005** (exactly `0.5·√10/(100√10)`);
  `p=100¹⁰, g=1¹⁰` → ratio 50, clamped to **1.0**, gradients `torch.equal`;
  `p=0, g=1` → scale **0.0** (the documented zero-init consequence).
* **Disabled is identity**: with a λ so large every ratio exceeds 1, every
  gradient is `torch.equal` to its input — the `_foreach_mul_` by 1.0 moves no
  bit. `cfg.agc_clip == 0.0` never calls the helper (`test_lrband.py`'s spy
  records `["clip"]`, not `["agc", "clip"]`).
* **Dead in SFT/RL before this audit**: both loops accept the shared
  `TrainConfig` and never call `agc_clip`. Now refused by
  `refuse_unwired` (a reported clip with no code is the defect class this repo
  has paid for).

---

## 4. EMA (`ema_decay`)

Fingerprint: `test_ema_*` (4 gates).

### 4.1 The analytic average

Pushing `p_t = t` for t=1..5 with `d=0.9`, the shadow is

| step | measured | closed form |
|---:|---:|---:|
| 1 | 0.1000000015 | 0.1 |
| 2 | 0.2899999917 | 0.29 |
| 3 | 0.5609999895 | 0.561 |
| 4 | 0.9049000144 | 0.9049 |
| 5 | **1.3144099712** | **1.31441** |

A skipped step (`flag=0`) leaves the shadow `torch.equal` — the device
firewall's `lerp_(0)` is exact.

### 4.2 Both copies in the checkpoint; the served one is the EMA

`save_ckpt` writes `model` = the raw `state_dict` (bit-equal per key) and
`model_ema` = the shadow (bit-equal per key), deliberately different in the
test. `load_model_half` — the reader eval/decode/serve/chat share — returns the
EMA. A resume (`load_ckpt` → `resumed_ema`) restores it.

### 4.3 A resume continues the average

3 steps + resume to 5 vs an unbroken 5-step run: every `report["ema"]` tensor
is `torch.equal`, and so is every raw parameter. The EMA is not restarted from
the loaded weights. (It is inert in SFT/RL — refused now.)

---

## 5. Schedule-free Muon (`muon_sf`, `muon_sf_wd`, `muon_sf_retract`)

Fingerprint: `test_schedule_free_*`, `test_the_sf_arm_*` (4 gates).

### 5.1 What the optimizer actually evaluates

With a controlled direction `O`, `Muon._sf_one` implements, per step:

```
P̂ = ColRow(O)                                        # _sf_direction
η̂ = 0.2·η·√(mn) / max(‖P̂‖_F, 1e-12)                 # nor_eta_hat
s ← s + η² ; c = η²/s                               # _sf_ckp
Z ← (1 − η·wd)·Z − η̂·P̂                              # decay at Z
X ← (1 − c)·X + c·Z
params ← Y = (1 − β1)·Z + β1·X                      # β1 = 0.9
```

Three steps at `lr=0.1, wd=0.01` on a `[4,4]`: `Z₀₀=0.07230`,
`X₀₀=0.04822`, `s=0.03`, `Y₀₀=0.05063` — all equal to the reference
recurrence written in the test. `sf_switch("eval")` copies `X` exactly;
`sf_switch("train")` reconstructs `Y`. Both are absolute writes from state, so
they are idempotent.

### 5.2 `wd` is required > 0 — and it is enforced at both seams

`validate_train(TrainConfig(muon_sf=True, muon_sf_wd=0.0))` raises; so does
`MixOptimizer(..., muon_sf=True, muon_sf_wd=0.0)`. The refusal text names
Lemma 3.1 and the `Z ← (1−ηλ)Z − η̂P̂` line. The Muon group's non-SF `wd` is 0
by design (`Muon` docstring), and the SF decay is a **different** knob
(`muon_sf_wd`), not the AdamW group's `wd`.

### 5.3 The Muon LR is constant after warmup

A 6-step SF run's schedule spy sees, at the last step, the AdamW call at
`1e-3` (the WSD tail) and the Muon call at `1e-2` (the peak): the loop passes
`decay_steps=0` for the Muon half. The hybrid is real, not a log line.

### 5.4 `muon_sf_retract` — DEAD, and honestly so

The field's only read is its own refusal; the port has no `retract_tsct` call
site (`grep` over `port/src` for `retract_tsct(` finds none). `true` is
refused, `false` is inert, and wiring it would require porting the retraction
the theorem's convex-set precondition forbids under an average. The dead-knob
audit already classified it **DEAD**; this suite pins the refusal and the
absence of a call site. **Keep the field and the refusal, or delete both; do
not wire it without the retraction.**

---

## 6. WSD schedule

Fingerprint: `test_wsd_*`, `test_a_resume_*` (2 gates; `tests/test_lr_tail.py`
holds the tail-replanning gates).

### 6.1 The last 100 steps

`total=1000, warmup=100, decay_frac=0.1 (decay=100), min_lr_frac=0.01`:

| step | measured `lr` | closed form |
|---:|---:|---:|
| 900 (first decayed) | 2.970300e-4 | 2.970300e-4 |
| 910 | 2.673300e-4 | 2.673300e-4 |
| 930 | 2.079300e-4 | 2.079300e-4 |
| 950 | 1.485300e-4 | 1.485300e-4 |
| 970 | 8.913000e-5 | 8.913000e-5 |
| 990 | 2.973000e-5 | 2.973000e-5 |
| 999 (last) | **3.000000e-6** | `peak·0.01` |

Every step in 900..999 matches the closed form to `1e-12`; the last step lands
on `peak × min_lr_frac`, not on an arbitrary tail value.

### 6.2 A resume recomputes from the global step

A 6-step run with a checkpoint at 6 resumes to 10. The schedule spy sees
`wsd_lr(6..9)`, and each value equals the closed form at the **global** step
with the resumed `steps=10` plan. The mutation `wsd_lr(step - start)` restarts
the warmup and reddens the gate with `[(0, …), (1, …), (2, …), (3, …)]`.
`decay_frac`/`min_lr_frac` are in `PROGRESS_KEYS`, so re-planning the tail is
legal and the resume uses the new plan — documented, and confirmed here.

---

## 7. z-loss, logit softcap, QK-clip

Fingerprint: `test_z_loss_*`, `test_logit_softcap_*`, `test_no_qk_clip_*`.

* **z-loss off is bit-identical**: two same-seed models, `z_loss=0` and
  `z_loss=1e-3` with the same weights, produce `torch.equal` logits. The loss
  delta is `0.0366368294` against `coef·mean(LSE²)=0.0366368257` (1e-7
  relative); the counter fires once; `pure=True` skips both.
* **Softcap formula**: `logit_softcap(x, 15)` is `torch.equal` to
  `15·tanh(x/15)`; the model path is `torch.equal` to `15·tanh(raw/15)`, and
  the tiny model's max logit is 4.53 (below the cap). With `logit_softcap=0`
  the model is bit-identical to the baseline.
* **QK-clip is absent**: no `qk_clip` field on `ModelConfig`, no function, no
  source mention. It is not a dead knob here; it was never ported. A gate
  refuses a future phantom field.

---

## 8. bf16 mixed precision

Fingerprint: `test_bf16_*`, `test_muon_bf16_state_*`, `test_muon_chunking_*`.

* **Masters are fp32 and the update lands on them.** A `bf16=True` model has
  50/50 parameters fp32 before the forward, after the backward, and after the
  optimizer step; every gradient is fp32. The `lm_head.weight` (AdamW-routed)
  step matches the written AdamW reference at `1e-5` relative — the bf16
  compute dtype does not become the state dtype.
* **The bf16 momentum rung stores one rounded fp32 accumulation per step.**
  After step 1 the state is exactly `(1−μ)·g₁` cast to bf16; after step 2 it is
  exactly `(μ·m₁ + (1−μ)·g₂)` computed in fp32 and cast once. A bf16
  accumulation would drift from that; the gate is `torch.equal`, not a
  tolerance.
* **`muon_chunk_elems` and an explicit `ns_dtype="fp32"` are pure selections**:
  chunk sizes 1 and the default produce `torch.equal` updates on the same
  seeded model, and the explicit fp32 rung is `torch.equal` to the default.
* The `ns_dtype="bf16"` rung remains an approximation by design (the docstring
  says so; `tests/test_muon_fast.py` holds its orthogonality bound); it is not
  a storage copy of the master.
* There is no bf16 *master-storage* copy in the checkpoint either: the storage
  rungs are `fp8`/`nvfp4`/`ternary` (`model.quant`), `quant="none"` writes the
  fp32 masters, and `unpack_state_dict` restores fp32. The only bf16 storage on
  the training path is `muon_state_dtype`, measured above; a model forward's
  bf16 cast is transient and never becomes the optimizer's parameter.

---

## 9. Wiring findings (the part the numbers cannot see)

These are the `muon_lr_mult` shape — a value that validates and never reaches
the arithmetic. Each was found by tracing the field through every constructor
that accepts the shared config.

1. **`muon_sf_retract` refusal was in `load_run`, not `validate_train`.**
   `train()` calls `validate_train` for hand-built `RunConfig`s, so a caller
   that skipped `load_run` could set `muon_sf_retract=true` and run it silently
   inert. The three SF refusals now live in `validate_train` (one definition,
   both callers) — `config.py:4770-4800`.
2. **`full_graph` dropped `muon_lr_mult`.** `FullGraph.set_lr(lr)` writes one
   value into every group; the eager and `opt_graph` paths then call
   `opt.set_lr(lr, muon_lr)` to restore the Muon half, but the `full_graph`
   path did not. Fixed at `loop.py:1820`; the seam's in-place fill is gated by
   `test_set_lr_fills_device_tensors_in_place_for_the_graph_arms`.
3. **`adam_eps` was dropped by SFT and RL.** Both constructed `MixOptimizer`
   without `eps=`, so a non-default `train.adam_eps` silently ran torch's
   1e-8. Forwarded at `sft.py:1139` and `rl_train.py:441`; the SFT/RL spies see
   `1e-3` when the config says `1e-3`.
4. **`agc_clip` and `ema_decay` were accepted by SFT/RL and never ran.** Both
   are `train()`-only arms (AGC lives in `opt_stage`; the EMA shadow and its
   checkpoint half live in `train`). `refuse_unwired` now refuses them for both
   loops (`sft.py:991`), with the field name and the escape in the message.
5. **`TrainConfig.wd` is the AdamW half's decay.** The Muon group runs at
   `wd=0` outside SF by design; the field's name had no docstring saying so.
   Added (`config.py:1716`). Applying `wd` to Muon would change a training
   default → **proposal**, not a fix.

## Ranked list

**Fixed here (safe, default-preserving)**

1. `full_graph` + `muon_lr_mult` — one line at `loop.py:1820`; bit-identical
   when the multiplier is 1.
2. SFT/RL `adam_eps` forwarding — default 1e-8 keeps every existing run bit-
   identical; a non-default now means what it says.
3. SFT/RL `agc_clip`/`ema_decay` refusals — turns a silent dead arm into a
   load-time error; no training default moves.
4. `muon_sf_retract`/`muon_sf_wd`/`muon_sf+opt` refusals in `validate_train` —
   closes the hand-built-`RunConfig` bypass.
5. `TrainConfig.wd` docstring — names the AdamW-only scope.

**Proposed (needs a training-default decision or a GPU gate; not done)**

1. `_foreach_adamw_step` per-parameter bias correction for late-joining
   parameters (§1.5). The live trainer never exercises it; the honest options
   are a documented precondition or a slow path for a state that appears after
   step 1, and the slow path must not read every step tensor (the sync the
   fused writer exists to delete).
2. Graph fallback: `_fall_back` restores the *unmultiplied* `lr` into the Muon
   group for the first eager step after a counted capture fallback
   (`graph.py:521-523`). One step at the wrong Muon LR; fix by remembering
   which bound groups are Muon, or by re-applying `opt.set_lr` after the
   fallback. GPU-only arm, so it needs the capture gate.
3. `opt="adamw"` + `muon_lr_mult != 1` is silently inert (there is no Muon
   group). `muon_sf` already has a matching refusal; add one for the
   multiplier, or document it.
4. Applying `wd` to the Muon group by default. **Training-default change**:
   the Muon+ColRow control carries none today; a decayed arm is a new A/B.
5. QK-clip: absent, not a knob. If it is wanted, it lands with its own gate,
   not as a config field first.

## Verification

```
port/tests/test_optimizer_suite.py                     34 passed
test_optimizer_suite + muon_reference + clip_grads
  + lr_tail + lrband + train4 + posttrain_fixes
  + full_graph + full_graph_m400b       141 passed, 8 skipped (GPU)
test_config + sft + sft_v2 + rl_train + nan_firewall
  + opt_graph + dead_knob_audit                        4 failed (environment)
```

The four failures are `port/checkpoints/` snapshots and an SFT corpus artifact
that are not committed to a fresh worktree (`test_every_snapshot_on_disk_loads`
passes in the main checkout, which has them). No failure references a changed
file.

`ruff check` clean on every changed file; `ruff format --check` clean on the
new test.

### Mutation log (each gate shown RED on a scratch edit, then reverted)

| mutation | gate | observed |
|---|---|---|
| drop bias correction (`bc1=1`, `bc2_sqrt=1` in `_foreach_adamw_step`) | `test_adamw_first_step_*`, `test_adamw_matches_the_written_reference`, `test_adamw_fused_writer_*` | 3 failed; step 1 delta `4.4721e-4` instead of `1e-3` |
| clip after `opt.step` (moved in `opt_stage`) | `test_the_training_stage_clips_before_opt_step` | `['step', 'clip'] != ['clip', 'step']` |
| ignore wd (`if wd:` → `if False:`) | `test_adamw_weight_decay_is_decoupled`, `test_adamw_fused_writer_*`, `test_adamw_matches_*` | 3 failed; zero-gradient parameter did not decay |
| restart warmup on resume (`wsd_lr(step - start)`) | `test_a_resume_recomputes_the_schedule_from_the_global_step` | resumed steps `[0,1,2,3]` instead of `[6,7,8,9]` |
