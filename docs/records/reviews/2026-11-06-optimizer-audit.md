# 2026-11-06 — optimizer lane: group routing, effective LR, clipping, the bf16 state

Hostile read of `port/src/dormouse/train/optim.py`, `train/lr.py` and the
optimizer stage of `train/loop.py`, against the live `flagship2` run
(`configs/flagship_base.toml` + the `--set` list in its snapshot,
step 65 000 of 171 000, BPB 1.7446 at step 66 000).

**VERDICT: DEFECTIVE.** Two defects, one of them the likely answer to "the model
learns slower than a vanilla transformer should":

| id | severity | defect | file |
|---|---|---|---|
| **D1** | **CRITICAL** | the Muon half and the AdamW half are given the **same LR**. Muon's geometric scale is not AdamW's: at `lr=5e-4` the Muon step buys **6-150× less first-order loss decrease per step** than the same `5e-4` on the AdamW half, on the real gradients and the real shapes. The recoupable factor on the memorisation gate is **8×** before divergence. | `optim.py:1783` (`Muon(..., lr=lr)`), no `muon_lr` field in `TrainConfig` |
| **D2** | MINOR | the mHC hyper-network `loop.mhc.phi_*` are 2-D and land in Muon. They are the controller class the routing policy sends to AdamW, and their `[2048, 2]` aspect ratio gets the full `max(1,m/n)^0.5 = 32` this code applies — measured **22.75× hotter** per unit loss decrease than AdamW, while every other Muon matrix is colder. | `optim.py:1653-1675` (`is_muon_param`), `mhc.py:159-161` |

Not the cause, measured and ruled out: gradient clipping, the bf16 Muon state,
the LR schedule, double-stepping, frozen parameters. Each has its number below.

**What was run, and on what.** The card belongs to the live job
(`nvidia-smi`: PID 716823, 7.0 GB, 80 %); nothing here touched it. Every number
is CPU, HEAD `c057e05`, `port/` checkout, on the `flagship2` checkpoint at
**step 49 999** (`~/dormouse/checkpoints/flagship2/flagship2.pt`, 922 MB) or on
the `tiny` preset. §8 carries a blocker that a reader of the checkpoint needs
first.

---

## 1. Group routing — every parameter, by group

`MixOptimizer` walks `model.named_parameters()` once (`optim.py:1747-1755`) and
sends a parameter to Muon iff `is_muon_param(name, p)` (`optim.py:1653`): 2-D,
both dims > 1, not a hashed-memory table, and the name contains none of
`embed`, `lm_head`, `controller`, `router`. Instantiated on the live config
(`configs/flagship_base.toml`, 102 678 943 params, 844 named parameters):

| group | params | elements | share | LR |
|---|---:|---:|---:|---:|
| **Muon** | 286 | 51 686 276 | 50.3 % | 5e-4 |
| **AdamW** | 558 | 50 992 667 | 49.7 % | 5e-4 |
| — of which the Engram table | 1 | 50 331 648 | 98.7 % of AdamW | 5e-4 |
| — non-table AdamW | 557 | 661 019 | **0.64 % of the model** | 5e-4 |

Muon, by block (all `[m, n]`, all 5e-4): `moe.experts.N.v` 256 × `[64,2730]` =
**44 728 320 (86.5 % of Muon)**; `moe.down.weight` `[1024,2730]` 2 795 520;
`mem_dense.weight` `[1024,1024]` 1 048 576; `moe.shared.v` `[192,2730]` 524 160;
`attn.{q,k,v,g}_proj` `[1024,256]` and `attn.latent_down` `[256,1024]` 5 × 262 144;
`csa.W_*` 8 × `[64,1024]`/`[1024,64]` 65 536; `moe.{gate_u,up_u}` `[1024,64]`,
`moe.shared.{gate_u,up_u}` `[1024,192]`, `out_proj.{u,v}` `[1024,64]`,
`engram.{query,key,value}_proj`, `attn.beta_proj` `[16,1024]`, `mhc.phi_*`.

AdamW: the Engram table `[3,524288,32]`; `embed.weight` `[256,1024]`;
`moe.router.weight` `[256,1024]`; `lm_head.{u,s,v}` (`[1024,64]`, `[64]`,
`[64,256]`); the 512 expert `gate_s`/`up_s` `[64]`; `controller.weight`
`[3,2048]`; `iter_embed` `[3,1024]`; `layerrope.sites.N.gamma` + the four
scalars per site; `attnres.N.{query,gain}`; `mem_dense.bias`; `norm.weight`;
`mhc.{alpha_pre,alpha_post,alpha_res,b_pre,b_post}`; `residual_scale`.

Checked one by one, the brief's suspicious list:

- **embeddings** → AdamW ✅ (`embed.weight` matched by the name rule).
- **output head** → AdamW ✅ (`lm_head.u/s/v`), including both 2-D factors.
- **Engram tables** → AdamW ✅. The table is `[3, 524288, 32]`, 50.3 M rows —
  98.7 % of the AdamW group by count and a lookup, never a matrix; no table is
  in Muon.
- **MoE expert banks** → Muon. Every per-expert `v` factor is `[64,2730]` and is
  stepped as a 256-matrix batch. This is deliberate and documented
  (`optim.py:6-11`); it contradicts AGENTS §3.5 item 3's "low-rank projections →
  AdamW", but the port's own ColRow A/B (−0.0428 BPB, `9ee82c5`) is on these
  exact factors, so it is not a routing accident. Not counted as a defect.
- **1-D vectors** (norms, biases, the expert `s` scales, LayerRoPE gains,
  `iter_embed` by its `embed` substring) → AdamW ✅.
- **mHC controller** → **Muon ❌**. `mhc.phi_pre/phi_post` are `[2048, 2]`,
  `phi_res` `[2048, 4]` (`mhc.py:159-161`). They are the hyper-network of Eq. 7 —
  the same class as `controller.weight`, which the name rule does send to
  AdamW — and they only escape because the name says `phi`, not `controller`.
  This is **D2**.
- **routing bias `lfb_bias`** → **not an optimizer parameter at all**. It is a
  non-persistent 256-float buffer (`experts.py:1432`), moved by the LFB
  integrator in the forward and carried in the checkpoint's `moe` section
  (`loop.py:262-271`, restored by `load_model_state`, `loop.py:673-678`). Its
  range in the step-50 k checkpoint is `[-3.998, +4.000]` — at the
  `moe_lb_clamp=4.0` rail on 11 of 256 experts (8 at +3.9 or above, 3 at −3.9 or
  below). Correct by design; the saturation belongs to the routing lane.
- **Anything large silently in AdamW?** No. The only non-table AdamW matrices
  are `embed`, `router` and the two `lm_head` factors — all four are the
  documented exclusions. 50.99 M of the 51.0 M AdamW count is the table.

---

## 2. The Muon LR is not an AdamW LR (D1)

### 2.1 The write itself is exactly the documented one

For 12 representative parameters, I ran the real `Muon.batched_step_from_model`
on the checkpoint's own momentum, then compared the parameter delta with
`-lr·max(1,m/n)^0.5·colrow_normalize(zeropower_via_ns(m))` computed by hand:

- `max |ΔW − theory|` over all shapes: **≤ 6.3e-8** (the bf16 state's store
  rounding). No missing, doubled or extra factor. `||ΔW||_F` is
  `lr · scale · sqrt(m)` to the last digit, i.e. the step is one unit-row
  direction times `lr`, as `optim.py:27-28` states.

So the *implementation* of the scale is correct. The *value handed to it* is not.

### 2.2 The scale, on the live shapes

`ΔW` per element (RMS) for one step at `lr=5e-4`, against the AdamW step on the
same tensor at the same `lr` (first step from zero state: `lr·sign(g)`, RMS =
`lr`; the live AdamW group's realised RMS is discussed in §4):

| parameter | shape | Muon RMS | AdamW RMS | AdamW / Muon |
|---|---|---:|---:|---:|
| `attn.q_proj` | `[1024,256]` | 6.25e-5 | 4.98e-4 | **8.0×** |
| `attn.latent_down` | `[256,1024]` | 1.56e-5 | 4.99e-4 | **32×** |
| `mem_dense.weight` | `[1024,1024]` | 1.56e-5 | 4.99e-4 | **32×** |
| `moe.down.weight` | `[1024,2730]` | 9.57e-6 | 4.96e-4 | **52×** |
| `moe.experts.0.v` | `[64,2730]` | 9.57e-6 | 4.32e-4 | **45×** |
| `moe.shared.v` | `[192,2730]` | 9.57e-6 | 4.89e-4 | **51×** |
| `csa.W_q` | `[64,1024]` | 1.56e-5 | 4.99e-4 | **32×** |
| `csa.W_o` | `[1024,64]` | 2.50e-4 | 4.91e-4 | **2.0×** |
| `attn.beta_proj` | `[16,1024]` | 1.56e-5 | 4.96e-4 | **32×** |
| `engram.query_proj` | `[32,1024]` | 1.56e-5 | 4.99e-4 | **32×** |
| `moe.gate_u` | `[1024,64]` | 2.50e-4 | 4.90e-4 | **2.0×** |
| `mhc.phi_pre` | `[2048,2]` | **1.13e-2** | 3.06e-4 | **0.03× (33× hotter)** |

Per-element RMS is not the learning quantity, though: Muon's update is a
low-rank rotation, AdamW's is a dense sign matrix, and their geometries are
different on purpose. The quantity that decides learning speed is the
**first-order loss decrease on the same gradient**, `<G, ΔW>`.

### 2.3 The decisive table: loss decrease per step, same gradient

One real forward + backward of the step-50 k checkpoint on real corpus text, then
`<G, ΔW_muon>` vs `<G, ΔW_adamw@5e-4>` for each parameter. `eff LR` is the LR the
AdamW half would need to match Muon's decrease on this step:

| parameter | shape | ‖G‖_F | ⟨G,ΔW⟩ Muon | ⟨G,ΔW⟩ AdamW | ratio | eff. LR |
|---|---|---:|---:|---:|---:|---:|
| `attn.q_proj` | `[1024,256]` | 1.55e-2 | −1.20e-4 | −2.29e-3 | 0.053 | 2.6e-5 |
| `attn.latent_down` | `[256,1024]` | 2.95e+1 | −4.67e-2 | −5.03e+0 | 0.009 | 4.6e-6 |
| `mem_dense.weight` | `[1024,1024]` | 9.18e+0 | −4.20e-2 | −3.17e+0 | 0.013 | 6.6e-6 |
| `moe.down.weight` | `[1024,2730]` | 3.54e+0 | −1.31e-2 | −2.00e+0 | 0.007 | 3.3e-6 |
| `csa.W_o` | `[1024,64]` | 3.55e-1 | −4.93e-3 | −2.93e-2 | 0.168 | 8.4e-5 |
| `mhc.phi_pre` | `[2048,2]` | 8.25e-5 | −3.67e-5 | −1.61e-6 | **22.75** | 1.1e-2 |
| *(reference: AdamW params)* | | | | | | |
| `embed.weight` | `[256,1024]` | 1.61e+0 | −1.26e-3 | −7.74e-2 | 0.016 | 8.1e-6 |
| `moe.router.weight` | `[256,1024]` | 1.60e-1 | −1.39e-4 | −5.58e-3 | 0.025 | 1.3e-5 |
| `lm_head.u` | `[1024,64]` | 5.35e+0 | −6.0e-2 | −3.7e-1 | 0.162 | 8.1e-5 |

The last three are AdamW parameters; their Muon column is the counterfactual
"what if this tensor were in the Muon group", i.e. the geometric gap in isolation.

**Reading:** on the parameters that carry the model (all of `attn`, `csa`, the
expert `v` banks, `mem_dense`, `moe.down`), the Muon half at `5e-4` delivers
**0.7 %-17 % of the loss decrease** the AdamW half gets at the same nominal LR.
The effective LR of the Muon group is **3e-6 … 8e-5**, not 5e-4. On the 44.7 M
expert `v` weights it is the smallest of all. That is a factor of **6-150**, not
the "10× off" the brief feared — worse.

The gradient in this table is the §8 replay's; the *geometry* is what is being
measured, and it does not depend on the fit: for a Gaussian `[m,n]` gradient the
same ratio is even smaller (`~2e-4` at `[1024,256]`, `~5e-3` at `[64,64]`); the
real gradient's low-rank structure raises it to the table's 0.7-17 %, still far
below 1. The memorisation gate in §2.4 confirms the direction end-to-end without
using the checkpoint at all.

This is not a transcription error: it is what the class's own default exists for.
`Muon.__init__` ships `lr: float = 0.02` (`optim.py:879`) — the published Muon
LR, ~60× the AdamW 3e-4 the same file defaults to — while `MixOptimizer` hands
both halves `cfg.lr` (`optim.py:1783` and `:1805`) and `TrainConfig` has no
`muon_lr`/multiplier field at all (`config.py:1661` is the only `lr`). The
`Muon` default is unreachable from the trainer: every caller passes `lr`.

### 2.4 The memorisation gate says the same thing, and bounds the fix

300 steps of the `tiny` preset on a fixed 66-byte string, batch 8 × seq 64, same
code, same `MixOptimizer`, `grad_clip=1.0`, `wd=0.01`, no warmup. BPB on the
training string:

| arm | 100 | 200 | 300 | outcome |
|---|---:|---:|---:|---|
| **live: mix, both LRs 5e-4** | 4.306 | 1.686 | **0.513** | barely under the 0.5 red line |
| adamw, 5e-4 (control) | 3.061 | 1.018 | **0.198** | 2.6× better than the live recipe |
| mix, muon 1e-3 | 3.489 | 1.206 | 0.158 | |
| mix, muon 2e-3 | 2.946 | 0.839 | 0.086 | |
| **mix, muon 4e-3** | 2.671 | 0.505 | **0.069** | 7.4× better than live, stable |
| mix, muon 8e-3 | 2.031 | 0.734 | 6.582 | diverges ~step 250 |
| mix, muon 2e-2 (`Muon`'s default) | 7.862 @50 | — | 0.61 | diverges at the start |

Monotone in the Muon LR up to 4e-3, then unstable. So the live recipe is not
"unable to memorise" — it is on the edge of the stated red line at 300 steps and
**an 8× Muon LR is worth 7.4× in BPB on this gate**, with the next doubling
killing the run. The `tiny` model's shapes are `[64,64]`-class, so the real
model's `[64,2730]` group (6.5× smaller per-element factor) plausibly wants
more than 8×; that is what the A/B in §9 is for.

---

## 3. Frozen or double-stepped parameters

**Double-step: none, by construction and by measurement.**

- `opt_stage()` (clip + one mixed step) is called once per optimizer step,
  outside the micro-batch loop (`loop.py:1782-1784`); the three micro-batches
  only accumulate `.grad`.
- `MixOptimizer.step` calls `muon.batched_step_from_model` once and then one
  AdamW writer (`optim.py:1948-1986`); the two parameter lists are disjoint
  (built in one pass, `optim.py:1747-1755`), and `_build_groups` partitions the
  Muon params by shape.
- The step-50 k checkpoint says the same: `opt.steps = 50 000` and
  `_adamw_t = 50 000` against `step = 49 999`, and every AdamW state's `step`
  tensor reads 50 000. One optimizer step per loop iteration, no more.
- Empirically, one `batched_step_from_model` on the loaded stack moved each
  parameter by exactly the hand-computed single-step formula (§2.1).

**Frozen: two real cases, neither is the optimizer.**

- `loop.residual_scale` never has a gradient — structurally dead under
  `use_mhc` (`loop.py`'s `allow_frozen` comment), 1 parameter. Known.
- `loop.attn.beta_proj.weight` shows `.grad is None` on a CPU replay, but the
  checkpoint carries a non-zero momentum for it (‖m‖ = 1.2e-3) and the KDA
  `projections()` reads it (`kda.py:390`); the CPU fallback arm is the
  explanation, not a live freeze.
- **5 of the 256 experts carry an all-zero Muon momentum at step 50 k**
  (indices 64, 71, 117, 153, 252) while their parameters are at the normal
  scale (‖v‖ ≈ 52.2-52.5, ‖gate_s‖ ≈ 6.1-6.2) and their cumulative routed counts
  are non-zero (2.76 M-23.9 M). Their `lfb_bias` is pinned at **+4.0**, the
  controller's *encourage* rail. All 256 expert parameters were routed at least
  once (`lfb_counts` min 2.76 M). This is a routing/controller state issue, not
  an optimizer one — the bf16 momentum of a long-cold expert underflows
  (0.95^1000 → 0) — but it means **a resume restarts those five with zero
  momentum**, a silent history loss. Handed to the routing owner.

`lfb_bias` is not in either group (§1); its update is the LFB integrator's
`sub_` in the forward, and it round-trips in the checkpoint's `moe` section.

---

## 4. Gradient clipping

`grad_clip=1.0` is **global over all parameters** — `opt_stage()` calls
`clip_grads(model.parameters(), cfg.grad_clip)` (`loop.py:1446-1447`), and
`clip_grads` is one `_foreach_norm` over the concatenated gradients
(`optim.py:2316-2334`). It clips both groups with one coefficient.

**How often it fires: a bit more than half the steps.** The log's
`g_norm_{muon,adamw,norms_biases}` are read *after* the clip
(`loop.py:1846-1854`, docstring at `:2155+`), so the log measures the
coefficient only indirectly. Across the log points at the time of reading (694,
and the file is growing):

- `sqrt(Σ g_norm²)` is **1.0000000** (within 7e-8) on **398/694 = 57.3 %** of
  points — the clipped value, exactly. The clip fired on those.
- The other **43 %** are under 1: median 0.684, p90 0.922, min 0.382. There the
  clip did not fire, so the *pre*-clip norm is printed directly.

So a typical non-clipped step has a natural global norm of ~0.7, and a typical
clipped step has one somewhere above 1 — **the log cannot say how far above,
because nothing pre-clip is logged.** That is the instrument's gap, and it is
worth one line on the log line (`grad_norm` is already computed per micro-batch
inside `micro_step` for the firewall, `loop.py:1516-1519`, and thrown away).

**The one pre-clip reading I have is on the §8 replay, and it is inflated by
that file's loss (6.5 nats).** One forward/backward through the step-50 k
checkpoint (batch 1 × seq 512, real text):

```
PRE-CLIP  muon=35.2  adamw=29.1  norms_biases=87.6   TOTAL = 98.8   coef = 1/98.8 = 0.0101
top 1-D:  loop.mhc.alpha_post=82.2  loop.attnres.0.query=24.3  loop.out_proj.s=16.3  ...
```

The **structure** is the finding, not the scale: a single 1-element parameter
(`mhc.alpha_post`, the residual-lane gate) carries 69 % of the squared norm in
this replay and the whole 1-D group carries 79 %. A scalar that gates the
residual stream has an inherently large gradient and will dominate any global
clip; the live coefficient is unmeasured but is bounded below by 1.0/‖g_pre‖
with `‖g_pre‖` unknown. On `tiny` (no mHC) the 50-step pre-clip distribution is
tamer: median 1.66, p90 2.31, max 2.70, **50/50 steps above 1.0** — the clip
fires there too, at ~1.7×.

**This is not the cause of the slow learning, and the algebra says why.** Both
halves are invariant to a *stationary* global rescale:

- Muon: Newton-Schulz divides by `‖M‖_F` (`optim.py:272`), and ColRow normalises
  rows, so `NS(c·M) ≡ NS(M)`. The Muon update is **exactly** independent of the
  clip coefficient — clipping cannot slow the Muon half at all.
- AdamW: `m` and `√v` are both EMAs of the same `c_t·g_t`, so `m/√v` is
  invariant while `c_t` is stationary (`eps=1e-8` is far below the post-clip
  `√v ≈ 1e-5`). The realised AdamW step is 1.0-1.4e-4 RMS, i.e. 0.2-0.28× the
  nominal 5e-4 — a factor from gradient *noise* (`|m|/√v ≈ 0.2`), not from the
  clip.

What the clip *does* cost: on the §8 replay the `mhc.alpha_post` scalar is driven
hard (6.7e-4 in the checkpoint, from the 1e-2 init — a 15× move on a 1-element
parameter, subject to the §8 caveat), and the firewall's pre-clip `grad_norm`
reads ~100, so a threshold tuned for a healthy ~1-norm model is meaningless
here. A per-group clip, or lifting the 1-D group out of the global norm, would
make the signal readable and would be the right form if the 1-D dominance holds
on the live model — which the log cannot answer today. It is not a
learning-speed defect.

---

## 5. bf16 Muon state (`muon_state_dtype=bf16`)

20 steps of the real Muon step on three live shapes, the same gradient sequence
fed to an `fp32` and a `bf16`-state arm, parameters from the checkpoint:

| shape | ‖Δ‖ fp32 | ‖Δ‖ bf16 | rel. diff | cos(Δ_fp32, Δ_bf16) |
|---|---:|---:|---:|---:|
| `[1024,256]` | 4.8138e-1 | 4.8138e-1 | 3.51e-3 | 0.99999702 |
| `[64,2730]` | 6.1375e-2 | 6.1375e-2 | 3.28e-3 | 0.99999624 |
| `[1024,1024]` | 2.2245e-1 | 2.2245e-1 | 4.28e-3 | 1.00000000 |

A **0.33-0.43 %** direction error over 20 steps, cosine ≥ 0.999996. The
`2026-11-05-opt-region.md` gate on `tiny` (100 steps) measured a 3.6e-4 relative
loss difference against a 3.4e-2 seed spread. Both measurements say the same
thing: the bf16 rung is a quality tie and worth its 488 MiB; it is **not** the
cause. (Stochastic rounding is still the unimplemented next rung; at this error
size it is not where the missing learning is.)

---

## 6. The LR actually applied at step 65 000

`wsd_lr` (`lr.py:30-58`) with the live recipe — `peak_lr=5e-4`,
`total_steps=171000`, `warmup_tokens=4e6`, `decay_frac=0.4`,
`min_lr_frac=0.01`:

- `warmup_steps = 4e6 / (8 · 3 · 1024) = **163**` optimizer steps, not the 488 in
  the brief (that figure omits `grad_accum=3`). Warmup is 0.095 % of the run.
- `decay_steps = round(0.4 · 171000) = 68 400`; decay starts at step
  **102 600** (60 % through). At step 65 000 the schedule is at **peak 5e-4**.
- Computed at step 64 999: **5.000000e-4**. The log's `lr=5.000e-04` at step
  65 100 matches to the printed digits.
- Both groups are written every step from this one value: AdamW at
  `loop.py:1682-1683`, Muon at `:1684-1686` (`muon_lr = lr` because `muon_sf`
  is off, `:1654-1664`).
- **Resume does not restart warmup**: `start = int(ck["step"]) + 1`
  (`loop.py:1303`), and the schedule is a pure function of the absolute `step`.
  A resume at 50 000 continues at `wsd_lr(50000) = 5e-4`.
- The checkpoint's saved `param_groups` carry `lr=5e-4` for both halves, and
  `load_state_dict` restores them; the next loop iteration overwrites them from
  the schedule anyway.

**CLEAN.** The schedule is applied to both groups, at the right value, with no
restart.

---

## 7. Sanity floor

The `tiny` memorisation run in §2.4 is the floor test: 300 steps, one fixed
66-byte string, batch 8 × 64. The live recipe reaches **0.513 BPB** — it does
memorise, but only just past the "plateaus above 0.5" red line at that horizon,
while the AdamW control reaches 0.198 and the same recipe with `muon_lr=4e-3`
reaches 0.069. The optimizer/loss path *can* memorise; it is spending the
budget on the wrong half.

---

## 8. Out-of-lane blocker: the step-50 k checkpoint does not replay

This is not an optimizer defect and I am not claiming it as one, but every
absolute number above that came from the checkpoint is contaminated by it, and
the owner of the checkpoint/eval lane needs it.

Loading `flagship2.pt` (step 49 999) through the **official** loader
(`load_model_half` + `load_model_state`, i.e. including the `moe` LFB section),
on CPU with the non-fused arms (`fused_residual=false`, `fused_epilogue=false`;
documented 8.5e-7 parity), the model scores **fin 4.62 nats** on the first
1 024 bytes of `web_eval.bin`, **4.51** at offset 3 000 000 and **4.43** at
6 000 000. The live run's eval line at step 66 000 reports **1.209 nats**
(1.7446 BPB) over the same file. On the training stream the first 256 bytes read
**6.84**. A freshly initialised model reads 6.30 headline on the same window.

The file is not a fresh init: 844 parameters differ from a same-seed init by a
median 0.31 relative, the AdamW `step` tensors read 50 000, and the Muon
momentum is consistent with a converged model (median ‖m‖ 3.1e-5). It loads with
zero missing/unexpected keys. Scored against the checkpoint's own targets, the
CE is *better* against `y` (4.58) than `x` (6.99), so the shift is not the
problem.

Reproduction:

```
PYTHONPATH=src .venv/bin/python -m dormouse.cli eval \
  --out-dir /home/sehaxe/dormouse/checkpoints/flagship2 --ckpt-name flagship2 \
  --eval /home/sehaxe/dormouse/data_local/mix4/web_eval.bin --device cpu
```

Either the file does not hold the live iterate, or the live eval does not score
it. Both are serious; neither is mine. **Do not resume or publish from this file
until the lane owner has resolved it.** My optimizer conclusions do not depend on
the absolute loss: the per-shape step magnitudes (§2.2) are exact for any
gradient (NS is scale-invariant), the decrease *ratios* (§2.3) are ratios on one
gradient, and the memorisation gate (§2.4) is independent of the checkpoint.

---

## 9. The fix and its gate

**D1 — separate the Muon LR.** One config field, one call-site change; no new
mechanism.

1. `TrainConfig.muon_lr_mult: float = 1.0` (validated > 0; a `PROGRESS_KEYS`
   entry is *not* wanted — the LR is already strict, and a mid-run flip changes
   the experiment). Resolve `muon_lr = cfg.lr * cfg.muon_lr_mult` in `train()`
   next to the WSD call (`loop.py:1638-1664`), and pass it to
   `MixOptimizer`'s Muon group only (`optim.py:1783`). The WSD shape stays
   shared, so warmup/decay still line up.
2. Start at **8×** (the memorisation gate's stable point), with 4× and 16× as
   the bracket. Do **not** change the live `flagship2` run: it is 65 k into a
   budget whose whole value is comparability with the 1.7292 run.

**Gate (must be green before the field lands):**

- *Mechanism gate, CPU, ~90 s*: `tiny`, 300 steps on the fixed string of §2.4
  must read **< 0.10 BPB at `muon_lr_mult=8`** with the control at 1.0 still
  reading its ~0.51 (so the gate is discriminating), and the arm must not
  diverge (this catches the 8e-3/2e-2 failures). Pin the number in a test next
  to `tests/test_opt_region.py`'s gate, not in a doc.
- *Quality gate, 3 seeds × 2 000 steps* on `medium`, `muon_lr_mult ∈ {4, 8, 16}`
  against the current 1.0, held-out BPB at one batch size (one eval window),
  per AB-PROTOCOL. The win has to beat the control's own seed spread.

**D2 — route `loop.mhc.phi_*` to AdamW.** Extend the name rule in
`is_muon_param` (`optim.py:1671`) with `mhc.phi`, or match `phi_`; that is 4
parameters, 16 384 elements, and it removes the one 33×-hot matrix from the
Muon group. Gate: the routing table test (`tests/` — the function is already
"checked by a test") gains an assertion that all four are in AdamW.

**Not to be changed:** clipping (§4 — inert for both halves while the
coefficient is stationary; the real fix there is a readable per-group norm, a
separate question), the bf16 state (§5 — 0.4 % direction error), the schedule
(§6 — correct).

---

## 10. Limits

- All measurements are CPU replays. The live step is on CUDA; the arithmetic
  gated here is device-independent (`NS`, ColRow, the schedule, the clip are
  plain torch), but a step-time claim is not made anywhere in this document.
- The first-order decrease table is a one-step linearisation, not a training
  curve; it is the correct local measure of "what does this LR buy", and the
  memorisation gate is the end-to-end check.
- §8 means the checkpoint's absolute loss is not trustworthy in this replay. No
  verdict here rests on it.
- The 5 dead-momentum experts and the ±4 `lfb_bias` saturation are reported, not
  diagnosed — that is the routing lane.
