# Control transformer: architecture or pipeline?

**2026-11-06 · subagent report · no push**

The flagship (`port/configs/flagship_base.toml`, 102,678,943 params) sat at BPB
1.7423 after 1.57 GB and 1.7292 at 4.2 GB, against a ~1.2–1.35 BPB reference for
a 124M vanilla transformer on FineWeb-Edu. This report isolates the cause with a
plain decoder-only control trained on **the same bytes**, scored by **the same
eval harness**, at **the same FLOPs per byte**, and — after the first result
looked too good — at **the same optimizer cadence**.

> ## UPDATE 2026-11-08 — the magnitude is not attributable; the sign survives
>
> The loop arms in this report ran **before `muon_lr_mult` existed** (landed
> 2026-11-07): their Muon step was the 15–55× deficit the optimizer audit found,
> measured on the same trainer at **0.246–0.248 BPB** (2 seeds, 49–74 MB,
> `~/logs/night_opt/*`). The control arms ran **AdamW everywhere — no Muon at
> all** (`scripts/control_transformer.py:261-263`), the loop's router had
> collapsed by the 4.2 GB endpoint, the anneal states differ (the control was
> fully annealed, the loop at peak LR), and the loop side has **zero seeds**.
> The same-seed rerun moved 0.043. So the *sign* (dense wins) survives — the
> control wins with 8.4× less data — but the **0.126–0.37 range is a mixture of
> optimizer, anneal and block**, and the sentence "the block is worth 0.126" is
> not supported by this file.
>
> A fair A/B is specified and priced at ~8 card-hours in
> `docs/records/reviews/2026-11-08-loop-ab-fairness.md` (mirror_control 13.5M/135
> MFLOP/B vs flagship_base 137 MFLOP/B, 500 MB, 2 seeds, pre-registered D ≥
> +0.03). The independent literature still points the same way — 116-run
> iso-depth law, r=2 loses +0.03…+0.06 nats — which is why the dense lane stays
> the live recipe while that A/B waits for a reason to spend 8 hours.

## Verdict

> **The pipeline is fine and the architecture costs ≈0.13 BPB.** A plain 13.5M
> dense transformer, trained on 500M bytes (1/8.4 of the flagship's data) with
> the flagship's own effective batch, reaches **1.492 BPB** on the trainer's
> window (**1.533** averaged over eight) — **0.126 BPB better** than the
> flagship at 4.2 GB (**1.660** over the same eight), and better on **every
> one** of them (−0.112…−0.142). At matched bytes the gap is **0.26 BPB**. By
> the control's own curve it would reach **~1.35 BPB at 4.2 GB** (floor-
> conditional 1.30–1.50), i.e. the architecture's bill is **≈0.27 BPB** (range
> 0.12–0.32) at the brief's comparison point.

Two qualifications are load-bearing and are not excuses. The gap is measured
against the *final* readout (the number generation uses), not the 1.729 headline
(§7.2). And the finished flagship checkpoint's MoE router had **diverged**
(`lfb_bias` range −117…+47; the loader itself warns that this controller
DIVERGED and the file's routing is a collapsed one), so
part of the cost is a controller failure, not an inherent cost of
KDA/CSA/mHC/Engram.

## 1. Compute match: the arithmetic, not the brief

The brief's "~57 MFLOP/byte, the same as a 124M vanilla at 62" does not survive
measurement. A dense transformer costs `6·N` FLOPs per token in the fwd+bwd pass
(2N forward, 4N backward), so:

| claim | implied N | verdict |
|---|---|---|
| 57 MFLOP/byte | 9.5M dense | not 20M active, not 102M total |
| 62 MFLOP/byte for "124M" | 10.3M dense | impossible: 124M dense is **744** MFLOP/byte |
| 6 × 20M active | 120 MFLOP/byte | closer, but still not the measured bill |

Measured with `torch.utils.flop_counter.FlopCounterMode` on the real
`checkpoints/flagship2/flagship2.pt` (batch 1, seq 1024, forward then backward,
through the repo's own loader):

| model | params | forward | fwd+bwd | source |
|---|---|---|---|---|
| flagship | 102,678,943 total, 17.26M MAC/token active | 45.81 MFLOP/B | **137.42 MFLOP/B** | FlopCounterMode on the checkpoint |
| flagship, repo ledger | same | 34.51 MFLOP/B | 103.53 MFLOP/B | `width.active_macs_per_token` — omits the T-dependent attention |
| control | 13,478,752 | 45.12 MFLOP/B | **135.36 MFLOP/B (−1.5%)** | FlopCounterMode, same call |
| control, closed form | 13,478,752 | 41.71 MFLOP/B | 125.15 MFLOP/B | `24Ld² + 4LTd + 2dV` per byte |

The control is sized from the measured bill: `288d² + 48Td` per byte forward with
`L=12, T=1024` gives `d=304`, 8 heads × 38, FFN 1216. The counter reads 135.36 vs
137.42 MFLOP/B (−1.5%), inside the ±10% the brief requires. **A dense model cannot
be 102M parameters at this compute**: 102M dense is 616 MFLOP/byte. Equal-FLOPs
therefore forces equal *active* compute, not equal parameter count — which is
exactly the question ("does the sparse/looped machinery buy anything at this
budget").

For the literature comparison: 124M at 1B tokens spends 744 PFLOP; the flagship
at 4.2 GB spends 577 PFLOP; the control at 4.2 GB would spend 568 PFLOP. The
reference is not compute-starved — it spends 1.3× our total FLOPs on 4.2× fewer
bytes. The gap to 1.35 is therefore not "we were too poor to train".

## 2. The control

`scripts/control_transformer.py`, plain torch, nothing exotic:

| | |
|---|---|
| vocab / input | 256 raw bytes (vocab 256, no tokenizer) |
| width / depth | d_model 304, 12 layers, 8 heads (head_dim 38) |
| attention | full causal, RoPE, `F.scaled_dot_product_attention` |
| FFN | GELU, 4× (1216) |
| norm | pre-LayerNorm (affine), linear layers bias-free |
| head | untied, 256 |
| optimizer | AdamW β=(0.9, 0.95), wd 0.1 (matrices only), lr 6e-4, cosine to 0.1×, warmup 2%, clip 1.0 |
| precision | bf16 autocast, fp32 masters |
| batch | 8 × 1024 = 8,192 B per micro-batch |
| params | 13,478,752 |
| seed | 1 |

No MoE, no loop, no Engram, no mHC/AttnRes, no CSA/KDA, no LayerRoPE, no QAT, no
deep supervision.

**Cadence.** The flagship takes one optimizer step per 24,576 bytes (`batch 8 ×
1024 × grad_accum 3`). The control was first run the same way the flagship draws
micro-batches (`grad_accum=1`, 3× more updates per byte), which turned out to
flatter it by 0.116 BPB at 50M. The decisive run therefore uses **`grad_accum=3`**,
the flagship's own effective batch, and the 50M/500M cadence-matched pair is
reported alongside the requested three-budget series.

## 3. Same bytes, same eval: the exactness of the comparison

**Stream.** `--set data=fwedu.bin` is *not* the training stream when `mix_plan` is
set: `loop.py` builds the stream from the plan's 12 sources and reads `data` only
for the eval-side bookkeeping. The control opens the trainer's own
`MixtureStream` with the flagship's settings: `mix_v7.toml`, `steps=513000`
(171,000 optimizer steps × `grad_accum=3`), `batch=8`, `seq_len=1024`, `seed=1`.
`source_for_step` is a pure function of `(seed, step, steps)` and each source's
`ByteStream` advances only when that source is drawn, so micro-step *k* of the
control is the byte-for-byte window micro-step *k* of the live run — a literal
prefix, not "the same distribution". The realized stage-1 shares of the 50M run
confirm it: fwedu 0.5866 (target 0.58), cosmo 0.1170 (0.12), ru 0.1689 (0.165),
wiki_en 0.0523 (0.055), wiki_ru 0.0254 (0.027), rudialog 0.0133 (0.014), instruct
0.0118 (0.011), reasoning 0.0247 (0.028).

**Eval.** `ByteStream(web_eval.bin, 8, 1024)`, `rewind()` first, 8 batches =
**65,536 scored bytes**, mean of per-batch mean CE, `BPB = CE / ln 2` — the
trainer's `bpb()`. The control's harness reproduces the live run's recorded
values exactly: at step 50,000 the JSONL says 1.7629 and the checkpoint measured
through this path reads 1.7629.

**Budgets.** Three budget-matched runs — 50M, 200M, 500M bytes — each with its own
cosine to zero, so the 50M and 200M points are annealed rather than pre-anneal
samples of the 500M run. The cadence-matched cross-check is 50M and 500M (200M
was not re-run; it costs 30 min for a point the 500M comparison does not need).

## 4. Results

Bytes are counted as the trainer counts them: `step × batch × seq_len ×
grad_accum` = 24,576 B per optimizer step. (The brief's "0.53 GB at 1.7423" is
`64000 × 8192`; the trainer draws three micro-batches per step, so that eval is
at **1.573 GB**, not 0.53 — §7.1.)

**The requested series (`grad_accum=1`, 3× more updates than the flagship):**

| budget | bytes | control BPB | ours, headline | ours, fin (est.) | Δ vs headline | Δ vs fin |
|---|---|---|---|---|---|---|
| 50M | 49,995,776 | 1.9991 | 2.2308 | 2.12 | −0.2317 | −0.1197 |
| 200M | 199,999,488 | 1.6828 | 2.0040 | 1.89 | −0.3213 | −0.2093 |
| 500M | 499,998,720 | 1.4673 | 1.8622 | 1.75 | −0.3949 | −0.2829 |

**The fair series (`grad_accum=3`, the flagship's cadence):**

| budget | bytes | control BPB | ours, fin (est.) | Δ vs fin |
|---|---|---|---|---|
| 50M | 49,987,584 | 2.1151 | 2.12 | −0.005 |
| 500M | 499,998,720 | **1.4918** | 1.75 | **−0.258** |

At 50M the cadence costs the control 0.116 BPB and the two models tie; by 500M the
cadence costs only 0.025 (1.4918 vs 1.4673) and the control is a quarter of a BP
ahead on the same bytes. The cadence effect shrinks with data, as it should: it is
an update-count shortage in the low-data regime, not a permanent handicap.

"ours, headline" is the live run's own eval row, log-D interpolated to the
control's exact byte count (the nearest rows are within 2% of the bytes). "ours,
fin (est.)" subtracts the measured headline−fin offset (0.108 BPB at step 50,000;
0.118 at 4.2 GB); it is an estimate because no checkpoint exists at those steps.
The headline comparison is exact, and the control wins it at every budget.

At exactly measured checkpoints, no interpolation:

| ours | bytes | headline | fin | window |
|---|---|---|---|---|
| `flagship2.pt`, step 50,000 | 1.2288 GB | 1.7629 | **1.6552** | 65,536 B |
| live, step 66,000 | 1.6220 GB | 1.7446 | — | 65,536 B |
| `flagship.pt` (finished, mix_v7) | 4.2025 GB | 1.7349 (log 1.729) | **1.6166** (log 1.609) | 65,536 B |

**The extrapolation-free result:** the cadence-matched control at 500M bytes
(1.4918) is **0.163 better than the flagship's fin at 1.23 GB** (1.6552) and
**0.125 better than its fin at 4.2 GB** (1.6166) — on 8.4× less data, the same
compute per byte, the same update cadence. This comparison needs no fit and no
floor.

## 5. The control's own curve, and the 4.2 GB number

The requested series measures 1.9991 (50M) → 1.6828 (200M) → 1.4673 (500M). The
curve is still falling hard at 500M, so **the requested 4.2 GB extrapolation is
not identified by three points**: the unconstrained three-parameter fit
`L∞ + A·D^-α` degenerates (it returns `L∞ = −21.3, α = 0.010, 0.99 BPB at 4.2 GB`,
a negative entropy floor), and the floor-free two-parameter power law
(`BPB = 21.39·D^-0.1335`) gives 1.11 BPB — also impossible. Both are the fit
running away because the data do not pin a floor.

What CAN be stated is a floor-conditional range. Fixing the entropy floor
`L∞ ∈ [0.9, 1.4]` and fitting `A, α`:

| assumed floor L∞ | α (1× cadence) | BPB @4.2 GB | α (cadence-matched) | BPB @4.2 GB |
|---|---|---|---|---|
| 0.9 | 0.275 | 1.228 | 0.312 | 1.204 |
| 1.0 | 0.312 | 1.253 | 0.355 | 1.231 |
| 1.1 | 0.361 | 1.283 | 0.413 | 1.263 |
| 1.2 | 0.429 | 1.321 | 0.496 | 1.301 |
| 1.3 | 0.529 | 1.368 | 0.628 | 1.350 |
| 1.4 | 0.690 | 1.428 | 0.891 | 1.414 |

The floor must be below the control's own 500M value (1.467), and raw-web byte
entropy is plausibly ~1.1–1.3, so **~1.30–1.35 BPB at 4.2 GB** is the central
estimate. Calibrating the method on the flagship's own curve (fit 49M/491M/1.23
GB, predict 4.2 GB): 1.63–1.68 predicted against 1.735 measured, i.e. the method
is **optimistic by 0.05–0.10 BPB**. Applying that bias, the control's 4.2 GB
estimate is **1.30–1.50**; the gap against the flagship's 1.617 is **0.12–0.32**
(central ≈0.27), consistent with the measured lower bound of 0.126.

Schedule check: the 500M run's own pre-anneal evals are 2.0652 at 50M and 1.6639
at 200M, against 1.9991 / 1.6828 for the budget-matched runs. The anneal is worth
0.07 BPB at 50M and −0.02 at 200M, so the budget-matched design matters at the
small end and is noise-level at 200M. The same-seed rerun of the 200M point
landed 0.043 BPB away (1.6399 vs 1.6828, CUDA nondeterminism under a shared GPU);
the fits use the higher, more conservative 1.6828.

## 6. Is the ranking window-robust? Yes.

The trainer scores a fixed 65,536-byte window (the first 65,536 bytes of
`web_eval.bin`). That is one to three documents, not a validation set, so the
natural objection is that the gap is a property of that slice. Measured on the
flagship's own checkpoint across eight 65,536-byte windows spread over the file:

| offset | 0 | 512k | 1M | 2M | 3M | 4M | 5M | 6.22M | mean |
|---|---|---|---|---|---|---|---|---|---|
| flagship @4.2 GB, fin | 1.617 | 1.699 | 1.622 | 1.607 | 1.650 | 1.691 | 1.677 | 1.714 | **1.660** |
| control @500M, matched cadence | 1.492 | 1.565 | 1.510 | 1.488 | 1.528 | 1.549 | 1.543 | 1.592 | **1.533** |
| Δ | −0.125 | −0.135 | −0.112 | −0.120 | −0.122 | −0.142 | −0.134 | −0.122 | **−0.126** |

The window-to-window spread of the flagship is 0.107 BPB (1σ ≈ 0.04), and the
control's is 0.104. The two profiles correlate at **0.97**: per-window difficulty
is a property of the text, not the model. But the *difference* is almost constant
— 1σ = 0.009, range 0.030 — and it is negative on **all eight windows**. The
control, on 8.4× less data, is better than the 4.2 GB flagship everywhere; the
trainer's window 0 is representative to ±0.01. This is the strongest form of the
result: even the *measured* 0.126 is a lower bound, because a control trained to
4.2 GB would be better still.

## 7. Measurement facts this control exposed (for the other audits)

These are not side notes: each one changes the comparison the brief asked for.

1. **`grad_accum` byte accounting.** The trainer's step consumes
   `batch × seq_len × grad_accum` bytes (`loop.py:1251`, and the loop draws
   `n_micro = grad_accum` windows). `--set steps=171000 --set batch=8
   --set grad_accum=3` is therefore 4.202 GB, and step 64,000 (the 1.7423 eval)
   is 1.573 GB. The brief's 0.53 GB is a factor-3 undercount.
2. **Headline vs final readout.** The flagship's eval `loss` is
   `(Σ CE_i + CE_final)/(T+1)` (`_eval_loss`'s own docstring calls it
   depth-confounded); the number generation reads is `fin`. On the same
   checkpoint: headline 1.7349, fin **1.6166** — 0.118 BPB. Comparing a
   single-output control to 1.729 charges the architecture for three
   intermediate readouts it never serves.
3. **Eval window size.** The same finished checkpoint, same file, same rewind:
   65,536 scored bytes → 1.7349 / fin 1.6166; 196,608 bytes → 1.7798 / fin
   1.6662. A BPB is only comparable within one window (§2.6); the old mix4 run's
   config used 196,608 and the live run uses 65,536. The 65,536-byte window's
   order-3 byte entropy is 1.732–1.842 across the file, so a good model's margin
   over a 3-gram on this slice is only 0.1–0.27 BPB — the slice is easy, and a
   "1.6 BPB" here is not a hard-web number.
4. **The eval file is not the training distribution.** `web_eval.bin` is the raw
   `mix4` web slice; stage 1 of `mix_v7` contains **no raw web** (58 % fwedu,
   12 % cosmo, 16.5 % ru, 5.5 % wiki_en, …). At 200M bytes the control's mean
   training CE is 1.007 nats while its eval CE is 1.166 — a 0.23 BPB
   out-of-distribution penalty that both models pay. The 1.2–1.35 reference is a
   *matched* FineWeb-Edu validation number.
5. **The finished checkpoint's router diverged.** Loading
   `checkpoints/flagship/flagship.pt` prints the repo's own refusal: `lfb_bias`
   range [−117.277, +47.138], "this controller DIVERGED … the file's routing is a
   collapsed one" (a LOUD warning, not a hard refusal — the eval proceeds). The live `flagship2.pt` at step 50,000 does **not** print it.
   The 4.2 GB endpoint's fin 1.617 is therefore a *collapsed-router* endpoint;
   the architecture may do better with a healthy router, and that is the first
   follow-up experiment.
6. **Optimizer cadence and LR schedule.** The flagship runs `grad_accum=3` with a
   WSD schedule at lr 5e-4; the control runs the brief's cosine at 6e-4. Matching
   the cadence moved the control by 0.116 BPB at 50M and 0.025 at 500M; the LR
   difference is within normal tuning and was not swept.

## 8. Verdict, restated with the caveats

> At equal bytes, equal FLOPs/byte, equal cadence and the same eval, a plain
> 13.5M dense transformer reaches **1.492 BPB at 500M bytes** and **~1.35 BPB at
> 4.2 GB** by its own (floor-conditional) curve. The flagship reaches **1.617**
> at 4.2 GB. **The pipeline is fine — it produced a model that learns — and the
> architecture as run costs ≈0.13 BPB** (0.126 measured on 8/8 windows, 0.12–0.32
> by extrapolation with a central ≈0.27, 0.26 at matched bytes).

The pipeline is not exonerated for the *absolute* number: the eval file is out of
the training distribution (§7.4), the headline metric is depth-averaged (§7.2),
the byte accounting in the brief is off by 3× (§7.1), the eval window is a single
easy 64 KB slice (§7.3), and the endpoint's router collapsed (§7.5). Those
explain why the absolute BPB is not 1.35; they do not explain why a 13.5M dense
control beats a 102.7M sparse/looped model at the same compute on every window.

Caveats, in order of how much they could move the verdict:

- **Three points, extrapolated 8.4×.** The 4.2 GB number is a floor-conditional
  fit, not a measurement. The measured, extrapolation-free comparisons are §4 and
  §6: −0.258 against the fin at matched 500M bytes, −0.126 against the 4.2 GB fin
  on all eight windows.
- **One seed, and the same-seed rerun moved 0.043 BPB.** The 200M run was repeated
  with the identical seed and config and landed at **1.6399 instead of 1.6828** —
  CUDA nondeterminism under a shared GPU (the per-step losses diverge from the
  first logged step), not a seed effect. The headline gaps (0.23–0.39) are 5–9×
  that; the fin-frame gaps (0.26, 0.13) are 3–6×; the per-window mean (0.126) is
  stable to 0.009. The verdict is not in doubt, but any single figure carries
  ±0.05.
- **Capacity is not matched, only compute.** 13.5M dense vs 102.7M total /
  20M active. That is inherent to "equal FLOPs/byte": a 102M dense model is
  616 MFLOP/byte. A 124M dense model at 5.4× the per-byte compute is a different
  experiment, and the remaining 0.2 BPB to the 1.2–1.35 reference may well be
  capacity rather than architecture.
- **The control is untuned.** One standard LR (6e-4), no sweep, no QK-norm, no
  RMSNorm, no Muon. Tuning would only make the control better, which strengthens
  the verdict if it already wins.
- **Stage 1 only.** All budgets are inside stage 1's 42 % boundary (1.765 GB); the
  4.2 GB point of the flagship saw the stage-2/3 decay mix, the 500M control did
  not. That biases the comparison *against* the control.

## 9. Cost and repro

```bash
cd /home/sehaxe/dormouse
PY=port/.venv/bin/python
# requested series: 3 budgets, 1 update per 8,192 B
$PY scripts/control_transformer.py --bytes 50000000  --tag c50  --out /tmp/c50.json
$PY scripts/control_transformer.py --bytes 200000000 --tag c200 --out /tmp/c200.json
$PY scripts/control_transformer.py --bytes 500000000 --tag c500 --out /tmp/c500.json
# fair cadence: the flagship's 24,576 B per optimizer step
$PY scripts/control_transformer.py --bytes 50000000  --grad-accum 3 --tag c50ga3  --out /tmp/c50ga3.json
$PY scripts/control_transformer.py --bytes 500000000 --grad-accum 3 --tag c500ga3 --out /tmp/c500ga3.json --save /tmp/c500ga3.pt
```

Ours, on the control's window (65,536 B) and across windows:

```bash
$PY /tmp/opencode/control/eval_ours.py /home/sehaxe/dormouse/checkpoints/flagship2 flagship2
$PY /tmp/opencode/control/eval_ours.py /home/sehaxe/dormouse/checkpoints/flagship flagship
$PY /tmp/opencode/control/window_scan.py /home/sehaxe/dormouse/checkpoints/flagship flagship
$PY /tmp/opencode/control/scan_control.py /tmp/opencode/control/c500ga3.pt
```

Artifacts: `/tmp/opencode/control/{c50,c200,c500,c50ga3,c500ga3,c200r}.json`
(arch, per-source step counts, every eval, wall time) and `c500ga3.pt` /
`c200.pt` for the window scans. Wall clock on the RTX 5070 Ti **shared with the
live flagship job** (capped at 2.5 GB): 7.9 min (50M), 29.0 min (200M), 67.2 min
(500M), 6.4 min (50M matched), 63.6 min (500M matched), 28.1 min (200M rerun) —
3.4 h of training in total. Total card use peaked at 9.9 GB of 16.3 GB and the
live job did not OOM; its `tokens_per_s` fell from 46–60k to 26–38k while the
control ran (−40 %). No number in this report depends on the live job's
throughput.
