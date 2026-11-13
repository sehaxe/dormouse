# 2026-11-08 — Optimizer VRAM: every state byte, what each rung costs, and the 1B wall

Lane: `port/`, optimizer memory. Hostile-accounting pass over every state byte the
optimizer keeps, and over the four knobs it ships (`muon_state_dtype`,
`ns_dtype` — the 2026-10-17 rungs —, `optim_8bit`, the EMA shadow), at the live
shape the other lanes measure: `configs/flagship3_dense.toml`, **b8 × s1024**,
eval off, mix recipe (`opt=mix`, `muon_lr_mult=12`, `lr=3e-4`,
`mix_plan=configs/mix_refetch.toml`, `doc_masking=false`), card RTX 5070 Ti
16 303 MiB.

Two questions, and they have different answers:

1. **At 95M, what does the optimizer cost?** 0.356 GiB of state and 1.16 GiB of
   step transient (of which `ns_dtype` halves one part and no knob touches the
   other). The run peak is 4.58 GiB and it is the **forward/backward's**, not the
   optimizer's: at 95M, `ns_dtype=bf16` buys **time** (1.98x) and
   `muon_state_dtype=bf16` buys **space** (0.176 GiB) — they are not two routes
   to the same win.
2. **At 1B, what does it cost?** 16.9 GiB (sparse) / 23.6 GiB (dense) of
   optimizer-stage commitment in fp32, and 14.7 / 18.0 with every existing rung
   on the bf16 setting — over a 16.3 GB card, before one activation. The
   momentum is no longer the wall; the **fp32 whole-step update list** (3.75 GiB
   at 1B, no knob touches it) and, on a wide dense MLP, the **un-chunked NS
   chain** (7.5 GiB on one `[27989, 10496]` matrix) are.

Commit `31d23ac` (dirty tree: the swarm's), 2026-11-08. Every number below is
either **[M]** measured here or **[E]** arithmetic over a [M] number; the tag is
on the row.

---

## 1. What the code allocates, per parameter

From `port/src/dormouse/train/optim.py` — the optimizer's own state, not a
reconstruction. A parameter is on **Muon** iff it is 2-D with both dims > 1 and
is not an embedding / `lm_head` / controller / router table
(`is_muon_param`, `optim.py:1667`); everything else is on **AdamW**. At 95M the
split is **94 568 448 Muon (99.51 %) / 466 752 AdamW (0.49 %)** — the byte
vocabulary is 256, so the embedding + head is 425 984 params and nothing else,
which is why the AdamW half is a rounding error at this scale.

| tensor | where | dtype | bytes / param | when it exists |
|---|---|---|---:|---|
| Muon `momentum` | `optim.py:1005` (group stack), `:1164` (per-param) | `state_dtype` | **4.000 fp32 / 2.000 bf16** | every Muon param, always |
| NorMuon `v` | `optim.py:1006-1010`, `:1228` | fp32 | `4/n` (0.0013 at n=3072) | `muon_variant="normuon"` only |
| SF `x` + `z` | `optim.py:1013-1014`, `:1242-1243` | `state_dtype` | **+8 fp32 / +4 bf16** | `muon_sf=true` only |
| SF `sf_s` | `optim.py:1250` | host float | 8 B / group | `muon_sf` only |
| AdamW `exp_avg` | `optim.py:2155` | fp32 | **4.000** | every AdamW param |
| AdamW `exp_avg_sq` | `optim.py:2156` | fp32 | **4.000** | every AdamW param |
| AdamW `step` | `optim.py:2154` | fp32 0-dim | **4 B per TENSOR** | every AdamW param |
| EMA shadow | `loop.py:1019` `_shadow_state`, `:1103` `_ema_update` | param dtype | **4.000 over ALL params** | `ema_decay > 0` (default **0.0**) |
| NS workspace | `optim.py:333-454` | `ns_dtype` | transient: ≈`6.2 × batch_elems × ns_bytes`, where `batch_elems` is the lockstep batch and **a single tensor is never split** | every Muon step, freed at its end |

Three facts a reader of the ledger cannot see from the ledger:

* **Muon keeps no `step` tensor.** Its LR comes from `group["lr"]`, its update is
  bias-correction-free, so its state is exactly ONE buffer per parameter. The
  "two moments + step" intuition of Adam is wrong by 2x here.
* **`muon_state_dtype` is a STORE rung, not a compute one**: the accumulation is
  fp32 and only the store rounds (`_momentum_update_`, `optim.py:818-855`,
  gated by `tests/test_muon_fast.py:812` "the state is EXACTLY half"). It buys
  peak only where the state is resident at the same time as the step — which it
  is, from step 1 onward.
* **`ns_dtype` buys no state at all.** It halves a *transient*, so it moves the
  step's peak and nothing else. That is why it can be a pure speed rung, and why
  the ledger — which has no `ns_dtype` lever at all (`scripts/vram_ledger.py:296`
  hardcodes `* 4`) — cannot price it.

### Bytes per arm, 95M (`flagship3_dense`, 95 035 200 params) [E then M]

| arm | Muon state | AdamW state | EMA (off) | optimizer-stage transient (measured) |
|---|---:|---:|---:|---:|
| a `fp32` / `fp32` | 378 273 792 B (0.352 GiB) | 3 734 220 B (0.0035) | 380 140 800 B (0.354) | 1.160 GiB |
| b `muon_state_dtype=bf16` | 189 136 896 B (0.176) | same | same | 1.160 GiB |
| c `ns_dtype=bf16` | same as a | same | same | 0.775 GiB |
| d both | 189 136 896 B (0.176) | same | same | 0.775 GiB |

The transient is **two blocks**, not one, and §2 measures both: the NS chain
(`ns_dtype`-scaled, ≈0.77 GiB fp32 at the default chunk) and the step's **fp32
update list**, which is `4 B × every Muon parameter` (0.352 GiB here) and is
untouched by either dtype rung. The ledger models only the first
(`NS_TEMPORARIES = 6`, `scripts/vram_ledger.py:66,296`): a 2.5 % underestimate of
the NS block (0.750 vs 0.769 GiB) and a **35 % underestimate of the whole
transient** (0.750 vs 1.160 GiB).

---

## 2. Verified against the allocator (deltas, b8 × s1024) [M]

Four processes, one arm each (`scripts/opt_vram_probe.py`), the real
`MixOptimizer` the trainer builds, `torch.cuda.memory_allocated()` read after the
model, after the first backward, after the first step, and the step's
`max_memory_allocated`:

| arm | state, optimizer's own sum | state, allocator delta | step peak | step transient |
|---|---:|---:|---:|---:|
| a `fp32`/`fp32` | 382 008 012 B = 0.3558 GiB | 385 113 088 B (**+0.81 %**) | 2.2988 GiB | 1.1597 GiB |
| b `bf16`/`fp32` | 192 871 116 B = 0.1796 | 193 387 520 B (**+0.27 %**) | 2.1177 | 1.1597 |
| c `fp32`/`bf16` | 382 008 012 B = 0.3558 | 385 113 088 B | 1.9141 | 0.7752 |
| d `bf16`/`bf16` | 192 871 116 B = 0.1796 | 193 387 520 B | 1.7348 | 0.7752 |

**The state matches the per-parameter table**: `muon.momentum` is 378 273 792 B
in fp32 and exactly half in bf16 (189 136 896 B), across **48 Muon tensors in 4
shape groups**; AdamW is `exp_avg` + `exp_avg_sq` at 1 867 008 B each plus 204 B
of step scalars over 51 tensors; there is no fourth tensor. The +0.3-0.8 %
residue is the first step's bookkeeping (`_build_groups`' zero-fill), not state.
`grads_sum` is 380 140 800 B — exactly `4 B × 95 035 200`, so the gradient block
is per-parameter as documented; the post-backward baseline additionally holds
68 465 408 B of resident forward/backward workspace (0.064 GiB, constant across
arms), which is why the raw `pre_state - model` delta is not the gradient sum.

**The transient decomposes, and one half of it is free of both knobs.** Measured
at two chunk caps (`muon_chunk_elems`, elements):

| chunk | `ns_dtype` | transient | NS term (2 × (fp32 − bf16) at the same chunk) | invariant term |
|---:|---|---:|---:|---:|
| 1<<25 = 33 554 432 | fp32 | 1.1597 GiB | **0.769 GiB** | **0.391** |
| 1<<25 | bf16 | 0.7752 | — | 0.391 |
| 1<<24 = 16 777 216 | fp32 | 0.8068 | 0.454 | 0.353 |
| 1<<24 | bf16 | 0.5800 | — | 0.353 |

The invariant term is **4 B × every Muon parameter** (94 568 448 × 4 =
378 273 792 B = 0.3523 GiB, measured 0.353): `batched_step_from_model`
orthogonalizes every queued job first and keeps `outs` — the whole step's
updates — in fp32 until the last parameter is written
(`port/src/dormouse/train/optim.py:1471-1486`). Neither `ns_dtype` nor
`muon_state_dtype` touches it; only the chunk cap shrinks the other term.

So the ledger's `6 × min(muon, chunk) × 4` is a correct model of the NS half
(0.750 vs 0.769 measured, 2.5 %) and **misses the other half entirely**: the
optimizer stage is `state + NS + updates`, and at 95M the update list is 34 % of
the transient.

---

## 3. The four arms end to end: 150 steps, b8 × s1024, eval off [M]

One lock acquisition, arms back to back, `log_every=10`, the recipe of
`port/scripts/recipe_ladder.sh` (now `mix_refetch.toml`: `mix_v7`'s slices were
deleted on 2026-11-08). The rate is the **median of the last three 10-step
windows** — `bench_engine.py`'s protocol — and `peak` is the run's
`max_memory_allocated`:

| arm | `ns_dtype` | Muon state | ms/step | kB/s | run peak GiB | reserved GiB | loss @150 |
|---|---|---|---:|---:|---:|---:|---:|
| a base | fp32 | fp32 | **288.1** | 28.44 | **4.578** | 5.373 | 2.4062 |
| b state | fp32 | **bf16** | 288.8 | 28.36 | **4.402** | 5.145 | 2.4136 |
| c NS | **bf16** | fp32 | **145.7** | 56.23 | 4.580 | 5.150 | 2.4061 |
| d **FAST (both)** | **bf16** | **bf16** | **148.0** | 55.33 | **4.402** | 4.904 | 2.4168 |

Window-by-window (ms/step, steps 11-150) is flat within ±2 % after warm-up —
`a` 282-290, `b` 284-293, `c` 143-147 (one 172 outlier), `d` 148-154 — and the
post-step-30 medians (285.5 / 287.5 / 146.5 / 149.0) agree with the tail-of-3
medians above, so the comparison is not a lucky window.

**What each rung actually changes:**

* **`ns_dtype=bf16` is the throughput rung: 1.98x** (288.1 → 145.7 ms/step,
  28.44 → 56.23 kB/s). This reproduces the 1.93x on record
  (`port/scripts/throughput_ladder.sh`'s header) and it is the whole gap between
  an optimizer-bound step and a model-bound one at 95M. It removes 0.385 GiB of
  step transient — **and none of it shows in the run peak** (4.580 vs 4.578)
  because the optimizer stage (2.30 → 1.91 GiB) is below the forward/backward
  peak (4.58 GiB). At 95M this rung buys time, not space.
* **`muon_state_dtype=bf16` is the memory rung: −0.176 GiB of run peak**
  (4.578 → 4.402), and it is **free** in time at 95M (288.8 vs 288.1 ms, +0.2 %,
  inside the window-to-window spread). The state is resident during the
  forward/backward too, which is why it moves the run peak while the transient
  does not.
* **Both (the ladder's `FAST=1` rung): 148.0 ms/step and −0.176 GiB.** The bf16
  state store costs 2.3 ms of `ns_dtype`'s 142.4 ms win (1.6 %) — it adds a
  cast-copy per chunk (`_momentum_update_`, `optim.py:850-855`).
* The training loss at step 150 (same seed, same init) moves +0.007 / −0.000 /
  +0.011 nats for b / c / d — inside the step-to-step noise of a 150-step run and
  no sign of divergence. That is a smoke signal, not a quality claim; §4 is the
  quality screen.
* **The 4.58 GiB run peak is 1.63x the ledger's prediction (2.814 GiB)**: the
  missing `n_layers` term of §6. The optimizer's own stage is the part the
  ledger gets right.

One honesty note on the rate: the card is shared with a swarm and the step is
launch-bound, so a lane's pytest can move the absolute kB/s. All four arms ran
inside one lock window in a fixed order, and the `ns_dtype` gap (2x) is far
outside anything CPU contention produced across these 15 windows.

---

## 4. Quality screening: 30 MB of bytes, eval 196 608 B, seed 1 [M]

Protocol: **3662 optimizer steps = 29 999 104 B** at b8 × s1024, one eval at the
end over **196 608 B** (`--set eval=true --set eval_every=3662
--set eval_bytes=196608`), seed 1, `lr=3e-4`, `muon_lr_mult=12`, `mix_refetch`,
`doc_masking=false` — identical data order and init in every arm.

**First the instrument's own noise, because it decides what "equal" means.**
Three runs of the SAME arm (the fp32 control) were launched with identical flags
— a chain bug dropped the per-arm `--set` on the first pass, which accidentally
produced the cleanest noise measurement this lane has:

| repeat | `bpb_fin` | Δ vs repeat 1 |
|---|---:|---:|
| 1 | 1.7130 | — |
| 2 | 1.7259 | +0.0129 |
| 3 | 1.7067 | −0.0063 |

**Range 0.0192, σ 0.0098** — so at this budget a single-seed delta below ~0.02
is not a reading. That is the same band the repo quotes (0.02-0.03 at 30 MB;
σ 0.008-0.011 at 120 MB), and it is worth stating plainly: two identical
configurations differ by 0.013 BPB here, so any arm result has to clear that.

The arms (one seed each — **screening, not a verdict**):

| arm | `bpb_fin` | Δ vs fp32 control | verdict at this budget |
|---|---:|---:|---|
| a `fp32`/`fp32` (control) | **1.7130** | — | — |
| b `muon_state_dtype=bf16` | TBD | TBD | TBD |
| d both (the FAST rung) | TBD | TBD | TBD |

<!-- FILL: verdict paragraph -->


---

## 5. 8-bit AdamW: wired for SFT/RL, refused in pretrain, and worth 2.8 MB

**Does the optimizer support it?** The field exists (`config.py:1939`
`optim_8bit: bool = False`), and `MixOptimizer.enable_bnb8bit`
(`optim.py:1944-1967`) swaps the plain `torch.optim.AdamW` for
`bnb.optim.AdamW8bit` with the same lr/wd/betas/eps. It is **reachable in SFT**
(`sft.py:1143`) and **in RL** (`rl_train.py:440`).

It is **not reachable in pretraining**: `loop.py:1629-1643` refuses
`optim_8bit=true` LOUDLY, and the reason is structural, not a missing line. The
trainer's NaN firewall is a *device* 0/1 flag that multiplies the first moment
and the weight-decay factor before the write; bitsandbytes' `AdamW8bit` has no
seam between its 8-bit state and the parameter write, so honouring the firewall
would need a host read of a device scalar — the sync the path exists to remove.
`train/graph.py:279` and `:595` refuse it under `opt_graph`/`full_graph` for the
same class of reason (no capturable path). The task said: if unsupported, do not
implement it — price it and stop. This is the price.

**The price, from bnb's own state layout** (verified in the installed source,
`bitsandbytes/optim/optimizer.py:518-529`): `state1` and `state2` are uint8
(1 B/param each) plus one **fp32 absmax per 256-element block** for each
(`blocksize = 256`), i.e. `2 + 2×4/256 = **2.031 B/param**` against plain
AdamW's 8.0 B/param. The Muon half is untouched — `enable_bnb8bit` replaces only
`self.adamw`.

| target | AdamW params | plain AdamW | AdamW8bit | saved |
|---|---:|---:|---:|---:|
| 95M `flagship3_dense` | 466 752 | 3.73 MB | 0.95 MB | **2.79 MB** (0.0026 GiB) |
| 1B dense | 5 506 902 | 44.1 MB | 11.2 MB | **32.9 MB** |
| 1B sparse | 1 849 416 | 14.8 MB | 3.8 MB | **11.0 MB** |

**Verdict: not a lever at this scale, and the 1B wall is not made of it.** The
AdamW group is 0.49 % of the 95M model and 0.18-0.54 % of the 1B targets,
because the byte vocabulary (256) makes the embedding + head tiny and the
router/controller is 1-D. An 8-bit *state* on 0.5 % of the parameters buys
megabytes while the Muon momentum on 99.5 % buys gigabytes. If an 8-bit rung is
ever wanted for the wall, it has to be an 8-bit **Muon** rung (momentum in
int8 + fp32 row scale), which does not exist here and would need its own gate —
`optim.OPT_DTYPES` has two rungs and `muon_state_dtype` is a store rung with a
round-to-nearest cast, so an 8-bit arm would also need the stochastic rounding
the module docstring already names as the unimplemented next rung.

---

## 6. The 1B consequence [E over the measured 95M laws]

At 1B the optimizer is 99.5 % Muon, so the bytes that matter are the momentum,
the NS chain and the update list. Three of those are now measured laws, not
guesses: momentum = 4 B/param (2 B at the bf16 rung), NS ≈ `6.2 × chunk ×
ns_bytes`, updates = **4 B × every Muon parameter** (§2). The parameter split is
the router's, over the ledger's own `meta`-instantiated targets:

| target | params | Muon share | momentum fp32 | momentum bf16 | NS fp32 | update list fp32 |
|---|---:|---:|---:|---:|---:|---:|
| 95M `flagship3_dense` | 95.0M | 99.51 % | 0.352 GiB | 0.176 GiB | 0.769 (measured) | **0.352 (measured)** |
| 1B sparse (40M active) | 1 017.2M | 99.82 % | 3.782 | 1.891 | 0.78 at chunk 1<<25 (1.5 at 1<<26) | **3.783** |
| 1B dense | 1 012.1M | 99.46 % | 3.750 | 1.875 | **7.52 (tensor-bounded)** | **3.750** |

The NS column is where the two 1B geometries stop being the same question. The
chunk cap is applied to the **batch of matrices**, never to one matrix:

```python
if batch and elems + n > chunk_elems:   # optim.py:438 — flushes the BATCH
    ...                                  # a single tensor larger than the cap
batch.append(t)                          # still goes in whole
```

`spa1b`'s largest Muon tensor is `[1055, 5461]` = 5.76M elements, so the cap
binds (0.78 GiB fp32 — the ledger's model, 1.5 GiB at its `1<<26`, is safe
here). `dense1b`'s largest is `[27989, 10496]` = **293.8M elements** (1.17 GB
fp32, two of them on the GLU), so one batch is 8.8x over the cap and the chain's
live set is 5 x-family tensors of 293.8M plus 5 `[10496,10496]` Gram-family
tensors of 110.2M: **7.52 GiB fp32 / 3.76 bf16 for that matrix alone**. The
ledger prices it at 1.5 GiB (its `min(muon, chunk)`), which is a 5x optimism.
This is arithmetic over the loop in `optim.py:333-384`, not a measurement — no
1B model was instantiated on this card — and it is the reason the two 1B columns
below differ by ~6 GiB.

Optimizer stage = `params + grads + momentum + AdamW + NS + updates + 1.0 GiB
context` (the 1.0 GiB is the ledger's own `CTX_GIB`; it is *not* in
`max_memory_allocated`, which is why the measured 95M step peak is 0.52 GiB
below the ledger's 2.814):

| arm | 95M, measured | 95M, ledger | 1B sparse | 1B dense |
|---|---:|---:|---:|---:|
| a `fp32`/`fp32` | **2.299** | 2.814 | **16.93** | **23.60** |
| b `bf16` state | 2.118 | 2.638 | 15.03 | 21.73 |
| c `bf16` NS | 1.914 | 2.439 | 16.54 | 19.84 |
| d both | 1.735 | 2.263 | **14.65** | **17.97** |

Against a 15.0 GiB budget (the ledger's own ceiling on a 16.3 GB card):

* **fp32 does not fit at 1B, anywhere.** 16.9 GiB (sparse) / 23.6 GiB (dense) of
  fixed commitment before the first activation. The ledger says 13.9/13.8 and
  reports a max batch of 0/1 at s1024 — it is 3.1 / 9.7 GiB light, and the two
  things it does not model are the update list (3.75 GiB) and, on a wide dense
  MLP, the un-chunked NS chain (7.5 vs 1.5).
* **The sparse 1B needs both rungs and still lands at b1**: 14.65 + the
  ledger's own dense activation term at b1×s1024 (1.68 GiB) = 16.3 GiB, over a
  15.9 GiB card. Without the write-back change below it does not run at s1024 at
  all.
* **The dense 1B is out of reach with this optimizer**, and not by a dtype: the
  NS chain over one `[27989, 10496]` matrix is 7.5 GiB fp32 / 3.8 GiB bf16, and
  splitting that matrix would change the polar (a sub-block's polar is not the
  rows of the polar) — it is a research question, not a chunk knob.
* **The largest single lever at 1B is a code shape, not a dtype**: writing the
  updates back per chunk instead of per step frees `4 B × Muon params` —
  **3.75 GiB at 1B**, more than the bf16 momentum (1.875) and ten times the
  bf16 NS (0.384 on the sparse geometry). It moves no value (each chunk's update
  is computed from its own momentum rows; `_classic_ns_many` already chunks
  internally), it costs one write-back per chunk, and it is **not implemented** —
  this lane priced it and stopped, because it is a trainer change and not a
  dtype rung. With it, `spa1b` bf16 lands at 10.9 GiB + activations and b1-b2
  becomes reachable.
* Ranked levers at 1B on the measured laws: chunked write-back **3.75 GiB** ·
  bf16 momentum 1.875 · un-chunked-NS repair 4-6 GiB on the dense geometry (a
  research change) · bf16 NS 0.384 on sparse / 3.8 on dense (+ 2x step time) ·
  chunk 1<<24 0.4. 8-bit AdamW: 0.011 GiB (§5).

**Two defects the 1B column exposes in the tool, not in the optimizer.**

1. `scripts/vram_ledger.py` never mentions `n_layers` or `loop_blocks` (grep:
   zero hits); its activation model prices ONE block with `max_iter` passes.
   Right for the looped `flagship_base` and the 1B targets (`max_iter=1`, one
   dense block); **wrong for the live 12-layer `flagship3_dense`, where the body
   is counted once instead of twelve times**. Its predicted step peak for that
   arm is 2.814 GiB; measured, 4.578 (§3). The optimizer columns are unaffected
   — the parameter tree is instantiated, not guessed — but any batch-size
   decision taken from the tool's fwd/bwd total on a multi-layer preset is
   optimistic.
2. The optimizer stage itself is under-modelled twice: `ns_dtype` has no lever
   (`ns_workspace` hardcodes `* 4`, `vram_ledger.py:298`), so a `ns_dtype=bf16`
   run's stage is overstated by 0.38 GiB at 95M; and `NS_TEMPORARIES = 6` prices
   only the NS half of the transient, missing the fp32 update list — 0.35 GiB at
   95M, 3.75 GiB at 1B. `optimizer_bytes` (`:279`) also never prices the EMA
   shadow, a full 4 B/param copy when `ema_decay > 0` (0.354 GiB at 95M,
   3.79 GiB at 1B) — the largest opt-in state item in the model and the only one
   that is not in the optimizer.

---

## 7. Boundaries, and what would change the verdict

* **The quality screen is ONE seed** at 30 MB, and the repo's own noise at that
  budget is **0.02-0.03 BPB** (σ 0.008-0.011 at 120 MB, `docs/reviews/2026-11-08-night.md`).
  A delta inside the band is "not decided", not "equal": this is screening, and
  a 3-seed confirmation at 120 MB is what a verdict needs (AGENTS §1.2).
* **The rate is one lock window with four arms**, median of the last three
  10-step windows, on a card shared with a swarm of other lanes. The 2x
  `ns_dtype` gap is far outside anything CPU contention produced; a 1 % gap
  between `b` and `d` is not a signal.
* **`[E]` rows are arithmetic over the measured 95M laws**, not 1B
  measurements: no 1B model was instantiated on this card (its fp32 parameters
  alone are 3.9 GiB, and the stage is 17-24 GiB). The dense-1B NS figure comes
  from counting the live tensors of `_classic_ns_batch` on a
  `[27989, 10496]` matrix, not from a run.
* **Priced, never measured**: the EMA shadow (4 B/param over the whole model —
  the largest opt-in state item, and it is not in the optimizer at all),
  `muon_sf`'s `x`/`z` (+8 B/param fp32, +4 bf16), NorMuon's `v` (4/n B/param),
  checkpoint size / resume time (the state dtype lands in the `.pt`, so bf16
  halves the optimizer half of every checkpoint — worth its own measurement),
  and the 8-bit **Muon** idea, which does not exist.
* **Capture/compile**: `optim_8bit` is refused under `opt_graph`
  (`train/graph.py:279`) and `full_graph` (`:595`) because bnb's state cannot be
  captured; neither dtype rung interacts with capture (both are storage, and
  `tests/test_opt_region.py` gates the captured path). Nothing in this lane
  changes a capture.
* **What would falsify the recommendation**: a 3-seed 30 MB screen (or a 120 MB
  confirm) with `muon_state_dtype=bf16` losing more than 0.02 BPB against fp32 —
  that would demote the memory rung to "only where VRAM is the binding
  constraint", which is §6's 1B case and not the 95M one.

### What this lane would do next, in order

1. **Ship `muon_state_dtype=bf16` on memory-bound presets** — halve the momentum,
   −0.176 GiB of run peak at 95M, for +0.2 % step time and a store-only rounding
   that `tests/test_muon_fast.py:812` already pins. It is already the m400/medium
   preset value (`config.py:126-127`); the dense 95M preset is the one that
   still runs fp32.
2. **Ship `ns_dtype=bf16` wherever the step is optimizer-bound** — 1.98x here,
   and it is the only rung that changes the *rate*.
3. **Chunk the Muon write-back** (not done): it is the 1B blocker (3.75 GiB) and
   it moves no value. It is a trainer change with its own gate, not a dtype.
4. **Fix the ledger's four gaps** (`n_layers`, the `ns_dtype` lever, the update
   list, the EMA) — otherwise the tool that decides 1B batch sizes is 3 GiB
   optimistic on the thing this report measured.

---

## Appendix — the commands these numbers came from

Every GPU command went through the gateway (`scripts/gpu_lock.sh`, card idle under
2.2 GiB, one process at a time), as AGENTS §1.5 requires.

**State and step transient** — six probes, one process each (the allocator's
high-water mark is per process):

```bash
cd port
PYTHONPATH=src bash scripts/gpu_lock.sh .venv/bin/python scripts/opt_vram_probe.py --state fp32 --ns fp32
#   --state bf16 --ns fp32 | --state fp32 --ns bf16 | --state bf16 --ns bf16
#   --state fp32 --ns fp32 --chunk 16777216 | --state fp32 --ns bf16 --chunk 16777216
```

**Throughput and run peak** — one process per arm, `bench_engine.py`'s wrapper
(`dormouse.cli.main()` plus a `max_memory_allocated` print), 150 steps, eval off:

```bash
.venv/bin/python <peak-wrapper> train configs/flagship3_dense.toml \
  --data ../data_local/best/fwedu_train.bin --eval ../data_local/best/fwedu_eval.bin \
  --set steps=150 --set batch=8 --set seq_len=1024 --set seed=1 \
  --set lr=3e-4 --set opt=mix --set muon_lr_mult=12 \
  --set mix_plan=configs/mix_refetch.toml --set doc_masking=false \
  --set eval=false --set eval_bytes=196608 \
  --set log_every=10 --set ckpt_every=1000000 \
  --set out_dir=/tmp/opencode/optvram/run_a --json-log /tmp/opencode/optvram/a.jsonl
#   arm b adds --set muon_state_dtype=bf16
#   arm c adds --set ns_dtype=bf16
#   arm d adds both
```

**Quality** — the same launcher with `--set steps=3662` (30 000 000 B / 8192 B per
step), `--set eval=true --set eval_every=3662 --set eval_bytes=196608`.

**Data.** `configs/mix_v7.toml` no longer loads (its slices were deleted on
2026-11-08, `docs/reviews/2026-11-08-merge-incident.md`); the plan used here is
`configs/mix_refetch.toml`, which is what `scripts/recipe_ladder.sh` now defaults
to. These are therefore NEW numbers on a NEW mix: a BPB here is comparable to
another arm in this report, not to a pre-merge ladder row.
