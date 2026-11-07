# 2026-11-07 — Muon audit: the step is 14-55x too small, and the knob that would have fixed it was dead

Hostile read of `port/src/dormouse/train/optim.py` (`Muon`, `MixOptimizer`) and the
schedule write in `port/src/dormouse/train/{loop,sft}.py`, against the measured
A/B on the dense 95M control (`flagship3_dense`, 94.6M params,
`configs/flagship3_dense.toml`):

| arm | bpb_fin @ 3000 | s/step | source |
|---|---:|---:|---|
| `opt=mix` (Muon+AdamW, `lr=3e-4`, `muon_lr_mult=1`) | **2.0150** | 0.462 | `~/logs/night_opt/mix_s1.jsonl` |
| `opt=adamw` (AdamW everywhere, `lr=3e-4`) | **1.7732** | 0.314 | `~/logs/night_opt/adam_s1.jsonl` |
| `opt=mix`, `muon_lr_mult=8` | 2.1867 @ 2000 | — | `~/logs/f3muon8.jsonl` |
| `opt=mix` (same run as row 1, same step) | 2.1784 @ 2000 | — | `~/logs/night_opt/mix_s1.jsonl` |

Same data (`mix_v7`), batch 8 × seq 1024 × grad_accum 3, same seed, 196 608 B eval
window, `muon_state_dtype` at the schema default `fp32` (`config.py:1933`;
`flagship3_dense` is not in `PRESET_TRAIN_DEFAULTS`, `config.py:109-110`), `ns_steps=5`
(`config.py:1870`), `ns_kind=classic`, `colrow_3d=colrow`, `muon_variant=muon`, `muon_sf=false`.

**VERDICT: two defects, one of them a dead knob.** The Muon update rule is a
faithful transcription of Muon+ (Eq. 4 of arXiv:2602.21545) and is within 2x of
the reference on direction and magnitude at the same `lr`; what is wrong is the
`lr` handed to it. On every matrix of this model the Muon half's per-element step
is `lr / sqrt(cols)` (14-55x below the AdamW half's `lr`-sized step), and the
field added to fix that (`train.muon_lr_mult`) was overwritten by the trainer's
own schedule write on every step before it could ever be read.

| id | severity | defect | file:line |
|---|---|---|---|
| **D1** | **CRITICAL** | the trainer writes `group["lr"] = <schedule>` into the Muon group on **every** step, before the first optimizer step. `MixOptimizer`'s `lr * muon_lr_mult` is overwritten before it is read → `muon_lr_mult` is inert, and the 8x A/B measured the knob off (both arms ran at `lr=3e-4`; §4 has the proof, and why the 0.0083 BPB between them is a schedule tail plus noise). | `train/loop.py:1760-1764`, `train/sft.py:1215-1218`; constructor at `optim.py:1783` (HEAD) / `:1798` (working tree) |
| **D2** | **CRITICAL** | both halves get **the same LR**, and Muon's unit is not AdamW's. `colrow_normalize` leaves every row at norm 1, so `RMS(ΔW) = lr·√(max(1,r/c))/√c` — a constant of the SHAPE, independent of the gradient, momentum and state dtype. At `lr=3e-4` that is 1.04e-5 on `[832,832]` against 3.00e-4 on the AdamW half (**28.8x**), 5.41e-6 on `[832,3072]` (**55.4x**), 2.00e-5 on `[3072,832]` (**15.0x**). The Muon half therefore delivers **2.1 %-8.6 %** of AdamW's first-order loss decrease per step on the same gradient. | `optim.py:1101`, `:1364` (`scale`), `optim.py:1798` (`lr=lr*muon_lr_mult`, mult defaults to 1.0 at `config.py:1953`) |
| **D3** | MINOR | the `<ckpt>.config.toml` snapshot *does* pin the value (`/tmp/opencode/f3muon8/f3muon8.config.toml:187`, `muon_lr_mult = 8.0`, so the launch really passed it and a resume cannot change it) — but the **run event on the jsonl** carries none of the Muon fields, so nothing in the telemetry says which arm a curve is. `f3muon8`'s first line is `{`lr`: 0.0003, `opt`: `mix`}` and nothing else. | `train/loop.py`'s run event |

Not the cause, measured and ruled out below: the Newton–Schulz coefficients, a
missing orthogonality normalisation, the bf16 momentum rung, zero/denormal
momentum, and a wrong `0.2·√(max)` factor (§5).

**Instruments.** Every number is CPU, no GPU touched (`CUDA_VISIBLE_DEVICES=""`),
`port/` tree at the audit branch's baseline commit, which is byte-identical to
`main`'s working tree on 2026-11-07 (tree `7771d7d`, `optim.py`'s last commit
`72f4a1c`, plus the `muon_lr_mult` working-tree diff). The reference is
`KellerJordan/muon@master:muon.py`, **fetched 2026-11-07, 286 lines,
sha256 `2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d`**, and
transcribed in `port/tests/test_muon_reference.py` (not a dependency).

---

## 1. What the code does to a 2-D parameter, per step

`MixOptimizer.step` (`optim.py:1937`) calls `Muon.batched_step_from_model` once,
then one AdamW writer; the two parameter lists are disjoint (`optim.py:1758-1763`).
Inside the Muon half, per shape group (`optim.py:1361-1455`) and per write
(`:1491-1509`):

| # | operation | where | arithmetic | changes the step size? |
|---|---|---|---|---|
| 1 | momentum | `_momentum_update_` `optim.py:818-855`, called `:1421` | `m ← μ·m + (1−μ)·g`, `μ=0.95`, **in place, fp32** (`muon_state_dtype="fp32"`); the bf16 rung keeps an fp32 accumulator and rounds only the store (`:850-855`) | **No.** The next op is scale-invariant, so the `(1−μ)` normalisation and even a wrong `μ` cannot move the magnitude. |
| 2 | Newton–Schulz | `zeropower_via_ns` `optim.py:225-301`, batched twin `_classic_ns_batch` `:333-384`, dispatch `:1530` | `x = g / (‖g‖_F + 1e-7)` (`:341/:344`); transpose if tall (`:345-348`); 5 iterations of `gram=x·xᵀ`, `poly = b·gram + c·(gram·gram)`, `x = a·x + poly·x` with the Jordan triple `(3.4445, −4.7750, 2.0315)` (`optim.py:87`); transposed back; `to(float32)`. `ns_dtype="fp32"` by default; reference runs this chain in **bf16**. | **No.** It produces a matrix with singular values ≈1 — the measured 5-step chain's Frobenius norm is **0.905·√n** on `[256,256]`, not exactly `√n` (`tests/test_muon_fast.py` is the orthogonality gate). It is also scale-invariant: the same `g` and `1000·g` give a bit-identical update. |
| 3 | ColRow normalisation | `colrow_normalize` `optim.py:457-491`, 2-D form `:486-488`, called `:1496`/`:1480` | `x /= ‖x‖_col + 1e-7` then `x /= ‖x‖_row + 1e-7` — **every row ends at norm 1** | **Yes, and it is the whole magnitude.** `‖ΔW‖_F = √rows` for *any* input, so per-element RMS = `1/√cols`. This is Muon+'s Eq. (4) (`Norm_ColRow(Ortho(M))`), not a defect — it is the documented, A/B'd arm (`colrow_3d="colrow"`, `:471-484`). |
| 4 | aspect scale | `optim.py:1364` (`scale = max(1, rows/cols)**0.5`), `:1101` for the per-param path | the reference's own `max(1, A/B)**0.5` (`muon_ref.py:40`) | Yes, by `√(rows/cols)` for tall matrices only. Matches the reference. |
| 5 | weight decay | `optim.py:1816` | `wd=sf_wd if muon_sf else 0.0` — **the Muon group runs at `wd=0`** in every non-SF run; the reference applies `p.mul_(1 − lr·wd)` (`muon_ref.py:92`) | No (0 here, and 0.1·lr at most elsewhere). It is a real difference from the reference, but it is OFF, so it cannot explain a 0.24 BPB gap in this A/B. |
| 6 | write | `optim.py:1504-1509` (batched), `:1185-1186` (per-param) | `p.add_(update.to(p.dtype), alpha=−lr·scale)`; the `lr` is `group["lr"]` (`optim.py:1363`) | **Yes — the only knob that can fix #3.** |

So the step is `−lr · √(max(1,r/c)) · ColRow(NS(m))`, and its size is not a
property of the data at all:

```
RMS(ΔW) = lr · √(max(1, r/c)) / √c        # exact, for every full-rank gradient
```

Measured (CPU, `lr=3e-4`): `[832,832]` 1.0397e-5, `[832,3072]` 5.4132e-6,
`[3072,832]` 1.9984e-5, `[2496,832]` 1.8014e-5 — every value is the formula to
4 digits. Scaling the gradient by `1e-6` and by `1e3` leaves the update RMS at
`5.413e-6`, bit-for-bit:

| gradient RMS | step RMS, `[832,3072]` | `<g, ΔW>` |
|---|---:|---:|
| 1.0e-6 | 5.413e-6 | 1.208e-5 |
| 1.0e+0 | 5.413e-6 | 1.208e+1 |
| 1.0e+3 | 5.413e-6 | 1.210e+4 |

That table is the entire mechanism in one line: **the Muon update has no scale of
its own, so its learning rate is the only thing that can be wrong — and it is.**

## 2. Reference vs this code, difference by difference

Reference: `muon.py` (`muon_update` `:34-41`, `SingleDeviceMuon.step` `:103-125`).

| # | reference | this code | changes the effective step? |
|---|---|---|---|
| 1 | `momentum.lerp_(grad, 1−β)` (`:35`) — EMA, same β | `m.mul_(μ).add_(g, alpha=1−μ)` (`optim.py:1149/:1151/:1401/:1403`) | No — identical recursion. |
| 2 | **Nesterov**: `update = grad.lerp_(momentum, β)` (`:36`), i.e. `(1−β)g + βm` | the momentum buffer alone — **no Nesterov term** (`:1152`, `:1441/:1445`). Deliberate and documented: Muon+ Eq. (4) has none, and `optim.py:25-26` says so ("the crate's manifest was the defect, not the code"). | **Direction only, not size.** Both are the same low-pass filter of `g`; on a full-rank random gradient the cosine between our update and the reference's is **1.0000** (measured, §3). A divergence from `muon.py` to name, not the 0.24 BPB. |
| 3 | `X = G.bfloat16()` (`:17`) — **the chain runs in bf16** | `ns_dtype="fp32"` by default (`optim.py:229`, `config.py:306-310`) | No — precision, not scale. Ours is the more accurate arm. |
| 4 | transpose before the prescale (`:18-19`), prescale `X/(‖X‖+1e-7)` (`:22`) | prescale then transpose (`:341-348`) | No — the Frobenius norm is transpose-invariant, and the prescale is `+1e-7` in both. |
| 5 | `update *= max(1, A/B)**0.5` (`:40`) | `scale = max(1, rows/cols)**0.5` (`:1364`) | No — same factor, applied at the end in both. |
| 6 | `p.add_(update, alpha=−lr)`, `lr` default **0.02** (`:65`) | `p.add_(update, alpha=−lr·scale)`, `lr` = the AdamW schedule, default `lr=3e-4`; `Muon.__init__`'s own `lr=0.02` (`optim.py:879`) is **unreachable** — every caller passes `lr` | **YES. This is D2.** A 66.7x nominal deficit before the shape factor. |
| 7 | no post-polar normalisation | `colrow_normalize` after the polar (`:1496`) | Yes, by `√(rows/cols)` for tall shapes (2.1x at `[3072,768]`). It is Muon+'s Eq. (4) — the repo's documented arm, A/B'd at −0.0428 BPB against `frobenius` (`optim.py:480-484`). Not a defect; it is *why* the per-shape deficit is not a single number. |
| 8 | `p.mul_(1−lr·wd)` (`:92`) | `wd=0.0` unless `muon_sf` (`:1816`) | No (off). |

Nothing in rows 1-5 and 7-8 can move the magnitude by 30x. Row 6 can, and does.

## 3. The per-shape measurement, and the single line

One `MixOptimizer.step` on a fixed random `W₀` and a fixed random `G`
(`μ` from zero, so this is the first step of a run; `lr=3e-4`, `wd=0.1` on the
AdamW half). `RMS ref` is the reference at **the same `lr`**; `RMS ref0.2` is the
older `0.2·√(max(A,B))` convention the brief names. `<g,ΔW>` is the first-order
loss decrease, `−<G, ΔW>`.

| param | shape | RMS ours | RMS ref | RMS ref0.2 | RMS AdamW | AdamW/ours | `<g,ΔW>` ours | `<g,ΔW>` AdamW | ratio | cos(ours,ref) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic | `[256,256]` | 1.875e-5 | 1.696e-5 | 5.428e-5 | 3.000e-4 | **16.0x** | 1.007e+0 | 1.575e+1 | **0.064** | 1.000 |
| synthetic | `[768,3072]` | 5.413e-6 | 5.153e-6 | 5.712e-5 | 3.000e-4 | **55.4x** | 1.209e+1 | 5.644e+2 | **0.021** | 1.000 |
| synthetic | `[3072,768]` | 2.165e-5 | 1.030e-5 | 5.711e-5 | 3.000e-4 | **13.9x** | 4.840e+1 | 5.651e+2 | **0.086** | 1.000 |

On the dense 95M model, by shape (**94 568 448 of 95 035 200 params — 99.51 % — are
in the Muon group**; AdamW holds 466 752 (0.49 %): `embed.weight` and `lm_head`
212 992 each, and 40 768 of norms/biases):

| shape | count | elements | RMS ours | RMS AdamW | deficit | mult for parity |
|---|---:|---:|---:|---:|---:|---:|
| `[3072,832]` `fc` (FFN down) | 12 | 30.67M | 1.999e-5 | 3.000e-4 | **15.0x** | 15.0 |
| `[832,3072]` `fc2` (FFN up) | 12 | 30.67M | 5.413e-6 | 3.000e-4 | **55.4x** | 55.4 |
| `[2496,832]` `qkv` | 12 | 24.92M | 1.801e-5 | 3.000e-4 | **16.7x** | 16.7 |
| `[832,832]` `proj` (out) | 12 | 8.31M | 1.040e-5 | 3.000e-4 | **28.8x** | 28.8 |

Element-weighted mean deficit **29.8x**; geometric mean **25.0x**. The AdamW RMS
is `lr` exactly on the first step (`lr·sign(g)`), which is what the 2026-11-06
audit's table measured too (4.98e-4 at `lr=5e-4`).

**The single line: `optim.py:1798` (`lr=lr * muon_lr_mult`, with
`muon_lr_mult=1.0`).** It is the one expression that decides the Muon half's
learning rate, and it hands Muon `3e-4` — an LR 15-55x below the quantity that
makes the AdamW half learn. The transcription above it is right (cosine 1.000,
magnitude within 2.1x of the reference *at the same `lr`*), so there is nothing
else in the update rule to fix.

**The alternative hypotheses are all excluded by the table, not by argument:**

- *wrong NS coefficients*: they would move the direction, and the direction is
  the reference's (cos 1.0000 on every shape). The magnitude would be unchanged
  even if they were wrong: the polar's spectral norm is ~1 whatever the quintic.
- *missing normalisation*: none is missing — the polar is normalised by
  construction, and the only extra normalisation (ColRow) *raises* the magnitude,
  it does not lower it.
- *zero/denormal momentum*: cannot matter, because the step is scale-invariant in
  the gradient (the `1e-6`/`1e3` table). Even the smallest non-zero momentum
  produces the full `1.04e-5`.
- *bf16 momentum underflow*: measured, §5 — false, and it could not change the
  magnitude anyway (same reason).
- *global gradient clipping*: the 2026-11-06 audit measured that `grad_clip=1.0`
  fires on 57 % of steps. It cannot move this number either way: the polar is
  scale-invariant, so any rescaled gradient gives a **bit-identical** Muon update
  (the `1e-6`/`1e3` table above). The clip only ever moves the AdamW half, and
  only down — it is not a lever on the gap.
- *a missing `0.2`*: the reference at HEAD has **no** `0.2` (its scale is
  `max(1,A/B)**0.5`, `muon.py`:40). If the older `0.2·√max` convention were used,
  the update would be 5.43e-5 per element on `[256,256]` and 5.71e-5 on both
  `[768,3072]`/`[3072,768]` (the `RMS ref0.2` column above) — 2.6-10x better than
  ours, and still **5.3-5.5x below** the AdamW half's `lr`. It would narrow the
  gap, not close it, and it is not the missing factor; the missing factor is `√c`.

## 4. Why `muon_lr_mult=8` changed nothing — the knob is dead (D1)

`train/loop.py:1716-1764` computes the WSD schedule **at the top of every step**
and writes it into both halves:

```
1716   lr = wsd_lr(step, ...)
1724   muon_lr = lr
...
1760   for group in opt.adamw.param_groups:
1761       group["lr"] = lr
1762   if opt.muon is not None:
1763       for group in opt.muon.param_groups:
1764           group["lr"] = muon_lr          # == lr
```

`MixOptimizer.__init__` set `Muon(lr=lr * muon_lr_mult)` (`optim.py:1798`), and
line 1764 overwrites that group key before `opt.step()` is ever called. Measured,
CPU, `[256,256]`, one step, `μ` from zero (`port/scripts/audit_mult_liveness.py`):

| trainer's schedule write | `ΔW` RMS, `mult=1` | `ΔW` RMS, `mult=8` | bit-identical? |
|---|---:|---:|---|
| not applied (constructor only) | 1.875e-05 | 1.500e-04 | no (8.00x, as designed) |
| **applied (what `loop.py` does)** | 1.875e-05 | 1.875e-05 | **YES** |

End to end on the port's own trainer (CPU, `configs/audit_d128.toml`, 4 layers,
854 144 params, 40 steps, `--set muon_lr_mult=8`, same seed, same flags):

| tree | loss @ step 20 | loss @ step 40 |
|---|---:|---:|
| measured code (`main`'s working tree) | 5.6416 | 5.2603 |
| one line: `group["lr"] = muon_lr * cfg.muon_lr_mult` (the A/B's patch; §7 is the real one) | 5.0438 | 3.8434 |

`/tmp/opencode/f3muon8/f3muon8.config.toml:187` proves the run really was launched
with `muon_lr_mult = 8.0`. So the 8x A/B was not an 8x A/B: the knob was off, and the two runs are not even
the same recipe — `f3muon8` ran `steps=2000` and its WSD decay tails off from
~step 1800, while `mix_s1` ran `steps=3000` and is still at peak `3e-4` at step
2000 (`~/logs/f3muon8.jsonl` vs `~/logs/night_opt/mix_s1.jsonl`). The 0.0083 BPB
between them (2.1867 vs 2.1784) is a schedule tail plus noise. The knob's
deadness is not inferred from that pair — it is proved directly, in the table
above and in `port/scripts/audit_mult_liveness.py`. `train/sft.py:1215-1218`
carries the same four lines and therefore the same dead knob.

**And 8 would not have been enough anyway.** The multiplier that puts the Muon
half's per-element step at the AdamW half's is `√c` (or `c/√(max(1,r/c))`), i.e.
**15-55** on this model. The prior audit's `tiny` memorisation gate is the
confirmation, not a contradiction: `tiny` is `d=64` (`tests/conftest.py:22-23`),
where `√64 = 8` — so its winning arm (`mix, muon 4e-3` against `adamw 5e-4`,
"7.4x better than live") was exactly *per-element RMS parity*, and the arm one
doubling past it (`8e-3`) diverged. The law transfers; the constant 8 does not.
That is the arithmetic the 8x A/B missed.

## 5. The bf16 momentum claim, measured and false

`muon_state_dtype="bf16"` stores the momentum in bf16 with an fp32 accumulator
(`optim.py:830-855`, `:1148-1151`). bf16 shares fp32's **exponent** range
(min normal 1.2e-38), so nothing at 1e-6..1e-3 underflows:

| `‖g‖` scale | mean rel. error of the cast | entries rounded to 0 | sign kept |
|---|---:|---:|---:|
| 1e-6 | 1.4e-3 | **0.0 %** | 100 % |
| 1e-5 | 1.4e-3 | **0.0 %** | 100 % |
| 1e-4 | 1.4e-3 | **0.0 %** | 100 % |
| 1e-3 | 1.4e-3 | **0.0 %** | 100 % |

The real bf16 hazard is the opposite of underflow: **swamping**. When the buffer
is larger than the gradient by more than bf16's 0.4 % resolution, the increment
is rounded away by the store:

| `‖m‖` | `‖g‖` | increment `(1−μ)‖g‖` | resolution `0.4 %·‖m‖` | entries where `bf16(0.95m+0.05g) == bf16(0.95m)` |
|---|---|---:|---:|---:|
| 1e-3 | 1e-6 | 5.0e-8 | 4.0e-6 | 95.8 % |
| 1e+0 | 1e-6 | 5.0e-8 | 4.0e-3 | 98.5 % |
| 1e+0 | 1e-1 | 5.0e-3 | 4.0e-3 | 28.4 % |

Consequence after 2000 identical-gradient steps: `cos(NS(m_bf16), NS(g))` =
**0.9998** (fp32: 1.0000), `‖m_bf16‖` 5 % low. The polar is scale-invariant, so
the 5 % is invisible; 0.9998 cosine is a plasticity effect on *large* buffers, not
a step-size effect. **It is not the 0.24 BPB and it is not on in this A/B**
(`fp32` is the schema default, §0). Also note the reference runs the whole NS
chain in bf16 (`muon.py:17`) and this port runs it in fp32 — ours is the more
accurate arm of the two, and the measured gap is on the fp32 arm.

## 6. Does closing the gap close the gap? A CPU control

The 3000-step A/B cannot separate "Muon learns slower at this LR" from "Muon is
the worse direction" — it ran one LR, and it ran it with the knob dead. So the
same question was asked on CPU, on the port's own trainer and its own data
(`fwedu.bin`, 1000 steps, batch 16 × seq 256 = 4.1 MB seen, eval window 65 536 B,
`configs/audit_d128.toml`: d=128, ffn 512, 4 layers, 854 144 params), with the
one-line patch applied so the knob is live:

| arm | 250 | 500 | 750 | 1000 | Δ vs AdamW @ 1000 |
|---|---:|---:|---:|---:|---:|
| `opt=adamw`, `lr=3e-4` | 3.809 | 3.501 | 3.341 | **3.249** | — |
| `opt=mix`, `muon_lr_mult=1` (as measured) | 4.104 | 3.841 | 3.744 | **3.667** | **+0.418** |
| `opt=mix`, `muon_lr_mult=12` | 3.564 | 3.028 | 2.753 | **2.586** | **−0.663** |
| `opt=mix`, `muon_lr_mult=24` | 3.356 | 2.869 | 2.692 | **2.562** | **−0.687** |

Same seed, same data order, same schedule, same eval window (65 536 B); the only
difference between arms 2-4 is the multiplier, and between arms 1 and 2 the
routing. Logs: `/tmp/opencode/ab_muon_lr/*.log` (not kept — the script is
`port/configs/audit_d128.toml` + the four `--set` lines).

At d=128 the per-shape deficit is 5.7x (`[512,128]`), 11.3x (`[128,128]`) and
22.6x (`[128,512]`) — geometric mean **11.3**, so **12 is the parity arm** and
**24 is 2x overdrive**. The reading:

- The gap at `mult=1` is **+0.418 BPB**, the same sign as the 95M A/B's +0.242
  and larger: 4.1 MB seen over 854 k params is **4.8 bytes/param** against the
  95M arm's 0.78, so this model is much further along its curve and *still*
  loses. Same question, same answer.
- Closing the effective-LR gap does not merely shrink the gap: it **reverses
  the sign**. At parity the Muon half beats AdamW by 0.66 BPB in 1000 steps.
- The arm still improves at 2x parity (2.562 vs 2.586), so the optimum here is
  not a knife-edge at parity. *Where* it is was not swept: this control fixes
  "too small" vs "wrong direction", and it does nothing more.

What this control does **not** settle: the multiplier for the 95M model (its
deficit is 15-55, not 6-23), 3 seeds, or a horizon. It is evidence about the
mechanism, not a recipe.

## 7. The patch (proposal, not merged)

Branch `muon-audit`, commit "fix(opt): one LR seam, so the schedule cannot drop
the Muon half's multiplier". Three files and one new method; **it changes no default**
(`muon_lr_mult` stays 1.0, so a run that does not set it is bit-identical — 40
CPU steps, `loss=5.2603` on both trees), and it is what makes any further
multiplier A/B mean anything.

```diff
--- a/port/src/dormouse/train/optim.py
+++ b/port/src/dormouse/train/optim.py
@@ MixOptimizer.__init__, beside the other stored arms
+        self.muon_lr_mult = float(muon_lr_mult)
@@
+    def set_lr(self, lr: float, muon_lr: float | None = None) -> None:
+        """Write the schedule's LR into both halves — the trainer's per-step call."""
+        if muon_lr is None:
+            muon_lr = lr
+        def put(group, value):                    # the opt_graph arm holds a 0-dim
+            if torch.is_tensor(group["lr"]):      # DEVICE tensor the replay reads:
+                group["lr"].fill_(value)          # fill IN PLACE (a float
+            else:                                 # assignment would freeze it)
+                group["lr"] = value
+        for group in self.adamw.param_groups:
+            put(group, lr)
+        if self.muon is None:
+            return
+        for group in self.muon.param_groups:
+            put(group, muon_lr * self.muon_lr_mult)
--- a/port/src/dormouse/train/loop.py
+++ b/port/src/dormouse/train/loop.py
             graph.set_lr(lr)
             if opt.muon is not None and cfg.muon_sf:
-                for group in opt.muon.param_groups:
-                    group["lr"] = muon_lr
+                for group in opt.muon.param_groups:
+                    group["lr"] = muon_lr * cfg.muon_lr_mult
+            elif opt.muon is not None:
+                opt.set_lr(lr, muon_lr)   # fills a captured LR tensor in place
         else:
-            for group in opt.adamw.param_groups:
-                group["lr"] = lr
-            if opt.muon is not None:
-                for group in opt.muon.param_groups:
-                    group["lr"] = muon_lr
+            opt.set_lr(lr, muon_lr)
--- a/port/src/dormouse/train/sft.py
+++ b/port/src/dormouse/train/sft.py
-        for group in opt.adamw.param_groups:
-            group["lr"] = lr
-        if opt.muon is not None:
-            for group in opt.muon.param_groups:
-                group["lr"] = lr
+        opt.set_lr(lr)
```

Two things a first draft of this patch got wrong, both caught by existing gates
and both worth naming because they are the same class of defect:

- `set_lr` originally **refused** a tensor `group["lr"]` on the theory that only
  the graph arm ever sets one. It does not: during graph *warm-up*
  (`graph.set_lr` installs the tensors before capture is attempted) every group
  holds a tensor, so the refusal fired on a legal run —
  `tests/test_sf_muon_legal.py::test_the_sf_muon_lr_is_constant_under_the_graph_arms_counted_fallback`.
  Filling in place is the correct handling, and it is how the graph arm gets the
  multiplier at all.
- the two loops in `set_lr` are part of a step, so `stepaudit`'s census
  (`tests/test_step_audit.py`) refused them until they carried a
  `# stepaudit-excuse:` — which is the gate doing its job.

**What the patch does NOT decide (the owner's call):** the value of the
multiplier. The evidence says the deficit is `√c` per shape, which a single
scalar cannot express on this model (15/55/17/29). Three honest options, in
increasing size:

1. `muon_lr_mult = 30` (element-weighted parity). Four lines of config, one
   shape-average, and the FFN up-projection stays 1.8x cold while the FFN down is
   2x hot.
2. **shape-aware** in `Muon._step_one`/the batched tail: multiply by
   `1/colrow_unit_rms(shape)` — i.e. rescale the ColRow'd update to unit RMS
   before `lr`, which is exactly "match the AdamW half's step" and is one line in
   the two write paths. This is what the test pins, and it is the only option that
   makes one `lr` mean one thing for both halves.
3. keep Muon's own convention instead (`0.2·√max` or `lr=0.02` unscaled) and
   sweep the multiplier — more faithful to the published recipe, but it makes
   `lr` a two-unit quantity in one config, which is what caused this.

The recommendation is (2) as a separate arm, because it is the smallest change
that removes the *class* of bug rather than this instance, and it leaves
`muon_lr_mult` as the single scalar the A/B sweeps.

## 8. What a 3000-step A/B can and cannot prove

**Can:** that at 3000 steps / 73.7 MB seen, on this model and this eval window,
`opt=adamw` at 3e-4 reaches 1.7732 BPB and `opt=mix` at 3e-4 reaches 2.0150 —
0.242 BPB, ~50-120x the repo's estimated paired resolution (0.002-0.005,
`docs/protocols/AB-PROTOCOL.md`) and therefore a real sign, not noise. The eval
window is identical between the arms (196 608 B, printed on both lines), the
schedule is identical, the seed is identical, and the two curves separate early
(2.615 vs 2.205 at step 1000) and stay separated (2.178 vs 1.946 at 2000).

**Cannot:** say *why*, and this audit's answer is a mechanism plus a CPU control,
not a re-run. Specifically:

- **It is one seed per arm.** `mix_s2` stopped at step 1500 and `adam_s2` never
  ran (`~/logs/night_opt/` has no `adam_s2.*`), so the 3-seed rule (AGENTS §1.2)
  was not met. The sign is safe at 0.24 BPB; the *size* is not measured.
- **It cannot separate "too small" from "wrong direction".** One LR, no sweep on
  the Muon arm. A 15-55x LR deficit and a bad direction both look like a worse
  curve. Only a working multiplier sweep separates them (the `tiny` gate did that
  at d=64; §6 does it at d=128).
- **It says nothing about the horizon.** 3000 steps on 73.7 MB is early training
  of a 95M model (0.78 bytes/param). Muon's published wins are at 100x+ that
  budget, and a recipe that loses at 3k can win at 100k. This A/B ranks two
  *recipes at this budget*, which is what it was run for — it is not a verdict on
  the optimizer.
- **`bpb_fin` is a single window**, and the difference is 0.24 BPB, so any claim
  about *which* component (attention vs FFN) carries it is unsupported.
- The 47 % step-time cost (0.462 vs 0.314 s) is real and independent; even a
  fixed Muon has to earn that back.

## 9. Gates

`port/tests/test_muon_reference.py` (CPU, ~2 s), three claims:

1. `test_muon_matches_the_reference_transcription` — cosine > 0.99 and magnitude
   within 2.5x of a transcribed `muon.py` step, per shape. **Green on the
   measured code** (the math was never the defect).
2. `test_the_trainer_delivers_the_muon_lr` — after one call to the schedule's
   seam, `mult=8` must move the parameter exactly 8x (not identically), and at
   the shape's own deficit the half must buy >25 % of the AdamW half's
   first-order decrease. **Red on the measured code** (`AttributeError:
   'MixOptimizer' object has no attribute 'set_lr'`), green after the patch.
3. `test_no_trainer_writes_group_lr_behind_the_optimizer` — no
   `group["lr"] = lr` statement in `train/loop.py` or `train/sft.py` (the SF
   arm's `= muon_lr * mult` write-back is a different quantity and is allowed),
   and both files call the seam. **Red on the measured code** (`loop.py:1758` is the first offender),
   green after.

Verified on the branch:

- baseline commit: `pytest tests/test_muon_reference.py -q` → **2 failed, 1 passed**
  (`AttributeError: 'MixOptimizer' object has no attribute 'set_lr'`, and
  `loop.py writes a param group's LR behind MixOptimizer.set_lr ... line 1758`).
- patch commit: **3 passed**, `ruff check src tests scripts configs` → clean.
- the patch changes **no default**: 40 CPU steps of `configs/audit_d128.toml`
  with `muon_lr_mult=1` give `loss=5.2603` on both trees, bit-identical; the same
  tree with `muon_lr_mult=8` gives `loss=3.8434` (it was inert before).
- full CPU suite: **4 failed, 1866 passed, 127 skipped, 1 xfailed** (338 s); the 4
  failures are **pre-existing and environmental** —
  `tests/test_config.py` (×3) and `tests/test_sft_v2.py` need
  `port/checkpoints/` and a corpus manifest, both untracked and absent from a
  worktree; reproduced identically on a pristine `7771d7d` worktree. No new
  failures after the patch.
