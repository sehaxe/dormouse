# 2026-11-08 — grad_accum sweep on the fast hand: the curve, the quality bill, and the 2 GB call

Lane `port/`. First execution of the claim the kernel audit left as **[S]**
(`docs/reviews/2026-11-08-kernel-audit.md` §0.6 / §C6 — its Russian, translated:
*"`grad_accum=3` on the fast hand gives a computed 76 kB/s (+35%) from two
measured arms; this has never been run"*). This runs it: five accumulations on
one command, the iso-byte BPB at each, peak VRAM, and the linear-LR second pass
(scripted, not completed here — §5).

**Verdict.**

* **a=3 = 63.1 kB/s = 1.27× a=1** [M], not 1.35×. The curve is real and
  shallower than claimed.
* **76 kB/s is not an a=3 number in either machine state** [C]: the audit priced
  a micro-batch at 88.8 ms, the card does **116.9 ms** (slow state) or
  **100.8-102.6 ms** (fast one). At a=3 that is 63.1 or 69.5 kB/s; 76 needs
  **a≈6.5-9** in the fast state, and is above the slow state's entire ceiling
  (70.1 kB/s) (§2.1).
* **Quality is free up to a=3** (+0.0038 BPB at 120 MB, inside single-seed noise)
  and **costs 0.034 BPB at a=8**. That kills a=8 as a free win.
* **Recommendation for the 2 GB run: a=3**, measured to be quality-neutral at
  120 MB and 1.25-1.27× faster; it turns a 10.0-11.7 h run into 8.0-9.3 h. The
  lr is the open question: §5 explains why 3e-4 is the safe default and 9e-4 the
  rule to screen first.
* **Past a=3 the curve is flat inside its own noise**: a=4..a=8 = 1.27-1.34 in
  one window and 1.26-1.30 in the other, i.e. +0 to +6% for 1.3-2.7× the
  wall-clock per optimizer step.

Marks: **[M]** measured here, artifact named · **[C]** arithmetic over [M] inputs ·
**[S]** inference with the check that settles it.

---

## 1. What was run

`configs/flagship3_dense.toml` (d_model 832, 12 dense transformer blocks),
**b8 × s1024**, `seed=1`, `lr=3e-4`, `opt=mix`, `muon_lr_mult=12`,
`mix_plan=configs/mix_refetch.toml`, `doc_masking=false`, FAST rung
(`ns_dtype=bf16`, `muon_state_dtype=bf16`, `ns_steps=5`), `ckpt_every=1000000`,
eval **only at the last step** (`eval=true`, `eval_every=steps`), held-out
`../data_local/best/fwedu_eval.bin`, `eval_bytes=196608` (24 × 8 × 1024), `off=0`.
Every arm through `scripts/gpu_lock.sh` from `port/`; no 75/124 exit is quoted
below.

Code state: `HEAD=e23e640` when the sweep started; other lanes committed `site/`
and `docs/` while it ran (`HEAD` was `354520f` at the end), and the four files
that decide the step are **byte-identical** across the whole sweep —
`loop.py 39c9d76f…`, `config.py 2fca7423…`, `optim.py c9efaaa7…`,
`transformer.py 9b78f202…` (sha256, checked 14:56 and 17:17). The tree is shared
and dirty by other lanes.

| family | steps | eval | question |
|---|---|---|---|
| **speed** | 200 optimizer steps | off | ms/step, kB/s (last JSONL window) |
| **iso-byte** | `round(120e6 / (8192·a))` | last step | BPB at the same 120 MB |
| **lr** | `round(30e6 / (8192·a))` | last step | does a bigger batch want a bigger lr |

The iso-byte design is stronger than "same budget". Four of the five arms draw
exactly `steps·grad_accum = 14648` micro-batches (a=3 draws 14649, one more,
because 120 000 000/24 576 does not divide evenly), and they are the same
micro-batches in the same order:

* `MixtureStream` draws the **same 14648 micro-batches, in the same order** — its
  stage/source is a pure function of `(seed, micro-step, steps)`
  (`train/mixture.py:511-528`), and `steps` there is the *micro*-step count
  (`loop.py:1476`, `steps × grad_accum`);
* the LR-vs-bytes schedule is identical to within that one micro-batch (WSD,
  `decay_frac=0.1`, `min_lr_frac=0.1`, `warmup_tokens=0` — all functions of steps
  and batch, and the schedule is a function of BYTES);
* the gradient is the **mean** over micro-batches, not the sum
  (`loop.py:1943-1945`: "makes `grad_accum=k, batch=b` match `batch=b·k`").

Only the **grouping into optimizer steps** changes.

Commands — the arms differ only in `grad_accum` and `steps`:

```bash
cd /home/sehaxe/dormouse/port
.venv/bin/python -u -m dormouse.cli train configs/flagship3_dense.toml \
  --data ../data_local/best/fwedu_train.bin --eval ../data_local/best/fwedu_eval.bin \
  --set steps=<200 | round(120e6/(8192*a)) | round(30e6/(8192*a))> \
  --set batch=8 --set seq_len=1024 --set seed=1 --set lr=<3e-4 | 9e-4 | 5.196e-4> \
  --set opt=mix --set muon_lr_mult=12 --set grad_accum=$a \
  --set mix_plan=configs/mix_refetch.toml --set doc_masking=false \
  --set eval=<false|true> --set eval_every=<steps> --set eval_bytes=196608 \
  --set log_every=<25|50> --set ckpt_every=1000000 \
  --set ns_dtype=bf16 --set muon_state_dtype=bf16 \
  --json-log /home/sehaxe/logs/accum/<arm>.jsonl
```

Artifacts: `/home/sehaxe/logs/accum/*.jsonl` (copy of `/tmp/opencode/accum/logs/`),
stdout `.log` beside each, per-arm VRAM samples `mon_a*.txt`.

---

## 2. The throughput curve

200 optimizer steps per arm, last JSONL `tokens_per_s`; the five arms ran
back-to-back inside one 8-minute window (14:03-14:11). The whole pass was then
re-run at 16:57-17:04 from the same script (drift control, §7.1):

| accum | ms/step A | kB/s A | × A | ms/step B | kB/s B | × B | bytes/opt step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 164.3 | 49.9 | 1.000 | 148.0 | 55.4 | 1.000 | 8 KiB |
| 2 | 278.0 | 58.9 | 1.182 | 251.8 | 65.1 | 1.176 | 16 KiB |
| 3 | 389.3 | 63.1 | **1.266** | 353.7 | 69.5 | **1.255** | 24 KiB |
| 4 | 517.0 | 63.4 | 1.271 | 456.0 | 71.9 | 1.298 | 32 KiB |
| 8 | 980.0 | 66.9 | 1.341 | 938.5 | 69.8 | 1.262 | 64 KiB |

[M] all ten rows. One token = one byte here, so `kB/s = tokens_per_s / 1000`;
`×` is bytes/s over the a=1 arm of the same window. The a=8 rows are the noisiest
(±5% window to window; medians of the last three windows give ×1.39 in A and
×1.23 in B — a 13% band, which is why §6 does not lean on a=8).

Two things are stable across both windows and one is not:

* **the ratios at a≤4**: 1.18 (a=2), 1.25-1.27 (a=3), 1.27-1.30 (a=4);
* **the absolute rate**: a=1 moved 164.3 → 148.0 ms/step (−10%) between the two
  windows with no code change, which is the whole of the "18% slower than the
  05:05 ladder control" puzzle of §2.1;
* **the a=8 ratio**: 1.26 (B) vs 1.34 (A) — and 1.23 if you take B's median of
  the last three windows. At 0.95 s/step a 25-step window is 24 s, so one
  disturbed window moves the number by 6%. Past a=3 the curve is flat inside the
  measurement's own noise.

Fitting the line `ms/step = c + m·a` over the five points of each window:

```
window A (14:03):  ms = 44.9 + 116.9·a   residuals +2.5 −0.7 −6.2 +4.5 −0.0 ms
window B (16:57):  ms = 45.9 + 102.6·a   (a=1..4 only; a=8 reads +71 ms high)
```

i.e. **~103-117 ms per micro-batch (fwd+bwd of 8×1024) and ~45 ms per optimizer
step**, both [C] over the [M] points. Three consequences the audit did not have:

1. **A ceiling exists and it moves with the machine.** `8192 B / m` is 70.1 kB/s
   in window A and 79.8 kB/s in window B; in the state the audit's own baseline
   came from (its 145.6 ms/step, m≈101) it is ~81 kB/s. So 76.0 is above the slow
   state's ceiling and inside the fast states' — but in **all three** it needs
   a≈6.5-9, never a=3 (§2.1).
2. **The knee is a=3.** a=3 reaches 90% of its window's ceiling; a=4 adds +0.5%
   (A) / +3.5% (B) for 33% more wall-clock per step; a=8 adds +6.0% (A) / +0.5%
   (B). **Past a=3 there is nothing to buy that survives the noise.**
3. **The optimizer's share is 27% of an a=1 step**, not the 39% the audit
   assumed (`44.9 / 164.3` [M]); that is the entire arithmetic difference between
   "+35%" and "+27%".

### 2.1 Why 76.0 kB/s is not reachable — and what it actually describes

The audit's arithmetic was `3×88.8 + 57.0 = 323.4 ms → 24 576 B / 0.3234 s =
76.0 kB/s`. Both inputs are derived, and both disagree with direct measurement:

| input | audit [C] | measured here [M] | why the audit's is off |
|---|---:|---:|---|
| micro-batch b8×s1024 | 88.8 ms | **116.9 ms** | came from `bench_compile`'s g=3 wall time minus its **fp32** `opt.step()` reading, `(470.0 − 203.6)/3`; the two do not subtract cleanly |
| optimizer step, fast hand | 57.0 ms | **44.9 ms** | derived as `145.8 − 88.8` — circular, it inherits the 88.8 |

The audit's own 40 k run (`logs/flagship3.jsonl`: g=3, 479.9 ms/step [M]) pins the
era-consistent split instead: with the fp32-NS optimizer at ~171-187 ms
([C] from `logs/thru/muon_fp32.jsonl`, 289.9 ms at g=1), the micro-batch there is
**~103 ms**, not 88.8. Put 103 and 45 back in:

```
older state:  a=1  103+45   = 148 ms (measured 145.6)  → 56.3 kB/s
              a=3  3·103+45 = 354 ms                   → 69.4 kB/s  (+23%)
              a=8  8·103+45 = 869 ms                   → 75.4 kB/s  (+34%)   ← 76 is THIS
```

**So 76 kB/s is an a≈8 number from a faster machine state, attributed to a=3.**
The multiplier at a=3 is 1.23-1.27× in both states, not 1.35×.

The gap between the two states (145.6 vs 164-172 ms/step at the identical recipe,
identical seed, identical final BPB to 4 decimals) is **the machine, not the
code**: the drift-control pass (§7.1) re-runs the same five arms 3 h later and
gets 148.0 ms/step at a=1 — inside the 05:05 band — with the ratios moving ≤1%.
So "two eras" is one card in two power states, and the ratio is the portable
number.

---

## 3. The iso-byte quality bill

Same 120 MB in every arm; see §1 for why the data and the LR schedule are
identical. `BPB` is `bpb_fin` on a fixed 196 608 B held-out window.

| accum | opt steps | bytes | BPB@120 MB | Δ vs a=1 | peak GiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 14648 | 120.00 MB | **1.4905** | — | — |
| 2 | 7324 | 120.00 MB | (not run) | | — |
| 3 | 4883 | 120.00 MB | **1.4943** | +0.0038 | 5.40 |
| 4 | 3662 | 120.00 MB | (not run) | | — |
| 8 | 1831 | 120.00 MB | 1.5241 | **+0.0336** | 5.40 |

`5.40 GiB` is the peak per-process VRAM (5532 MiB) of that arm's own run; the
`—` rows were not sampled (see §4).

[M] BPB on the three rows that carry it (1, 3, 8); peak = per-process VRAM from
`nvidia-smi --query-compute-apps`, 2 s sampling, [M] on a=3 and a=8.

**How to read the right column.** The repo's own screening rule for one seed at
120 MB is ~0.02 BPB (`scripts/recipe_ladder.sh`: "any arm that moves BPB by more
than ~0.02 earns two more seeds"). So:

* **a=3 is in the noise** (+0.0038): at this budget there is no measurable
  quality cost from tripling the optimizer's batch with the lr held.
* **a=8 (+0.0336) is 1.7× the threshold** — a real bill, not noise.
* **a=2 and a=4 are not in the table, and that is a gap, not a zero.** Their
  120 MB arms were queued and never got a GPU slot (the box restarted at least
  three times; the card is shared by seven lanes). Neither can change the
  recommendation: a=4 is throughput-dominated by a=3 (+0.4%/+3.5% for 33% more
  per-step time *and* 25% fewer updates at the same lr), and a=2 sits between the
  two measured rows in both columns. Both arms are scripted in §8.
* The shape is monotone: quality falls as the optimizer's update count falls,
  with the lr fixed. That is the expected direction and it is why §5 tests the
  linear rule.

**Methodology cross-check:** `a=1` reproduces the ladder's own control on this
plan — `logs/ladder/control_refetch.jsonl` is **1.4906** at 14648 steps from
05:05, this arm is **1.4905** from 14:55. An independent run of the same command,
4 dp apart, on a shared box, 10 h later. The harness is sound; the other rows are
a difference from a number the repo already trusts.

---

## 4. Peak VRAM

**5.40 GiB (5532 MiB)**, [M] by 2 s per-process sampling of our own pid
(`nvidia-smi --query-compute-apps`, `logs/accum/mon_a3.txt` / `mon_a8.txt`),
measured on the **a=3** and **a=8** arms at 120 MB. The two ends of the
accumulation range agree to the MiB, which is what the construction predicts:
the micro-batches run sequentially and the gradient buffer is allocated once, so
`grad_accum` cannot add VRAM. A 40-step probe of the middle rows was queued to
confirm it and never got a slot, so those cells are `—` rather than a guess.

Against the card's 15.9 GiB usable that leaves ~10 GiB, and it is why
accumulation — not `batch` — is the only way to grow the effective batch here:
b32 × s1024 does not fit at b8's footprint.

## 5. The linear-LR pass (30 MB) — scripted, not completed

The second pass ("BPB at 30 MB for a=1 vs the winner with lr scaled by the linear
rule") **did not finish on this box in this session**. The card is shared with
seven other lanes on one `flock`, and the harness restarted at least three times
mid-sweep, killing the detached trainer each time (the a=3 120 MB arm died at
step 4100/4883 and was re-run; the 30 MB a=1 arm died at 450/3662 and was
re-queued, then sat behind a 40-minute VRAM benchmark). Four arms are scripted
and the command is in §8; **no BPB from them is quoted**, because nothing was
measured.

What the sweep does say about the lr without that screen:

* at **120 MB the a=3 arm with the UNSCALED lr (3e-4) was inside noise of a=1**
  (+0.0038 BPB), so scaling is not needed to break even at that budget;
* at **a=8 the same unscaled lr cost 0.034 BPB** — the signature of "the batch
  grew, the update size did not", which is the direction the linear rule exists
  to fix;
* the repo already ships the same experiment coarser:
  `scripts/ab_batch_lr.sh` screens a=1/4/8 at b8 with lr 2e-4/4e-4/8e-4 at 30 MB
  and was **never run** either (no artifacts in `logs/sweep_blr/`); its header
  warns that a 30 MB ranking need not hold at 2 GB.

So the choice for the 2 GB run is between (a) `lr=3e-4` — the only lr in this
report whose BPB is measured — and (b) spending ~30 min on the four 30 MB arms
first and taking the scaled lr from that screen. §6 recommends (b) and names the
first lr to try.

## 6. Recommendation for the 2 GB run

**Run `ACCUM=3`.** The lr is the one knob this sweep could not pin: `lr=3e-4` is
the only value whose BPB is measured (quality-neutral at 120 MB), and `lr=9e-4`
is the linear-rule prediction and is **unmeasured** (§5). So:

```bash
# measured path, no new assumption
BYTES=2000000000 ACCUM=3 LR=3e-4 bash scripts/train_budget.sh
# or spend ~30 min on the 30 MB screen first, then:
BYTES=2000000000 ACCUM=3 LR=9e-4 bash scripts/train_budget.sh
```

Why:

1. **Wall clock.** At the controlled ratio (1.255-1.266×) a 2 GB run goes from
   **10.0-11.7 h to 8.0-9.3 h** [C]: a=1 sustains 47.6 kB/s in the slow machine
   state (2e9 B → 11.7 h) and 55.4 kB/s in the fast one (→ 10.0 h), and the
   ratio transfers across states (§7.1). Quote the ratio, not the absolute —
   the same recipe moved 10% between two windows today.
2. **Quality at iso-bytes is free at a=3 with the lr held** (+0.0038 at 120 MB,
   inside noise) — that is measured. The **lr** for the 2 GB run is not: the
   30 MB screen did not complete (§5). Run `lr=3e-4` (measured) or spend 30 min
   on the screen and use `9e-4` (the linear rule), which is what I would do
   before committing 9 h.
3. **a=8 is not free**: +0.034 BPB at 120 MB (1.7× the noise floor) for +0.5% to
   +5.9% throughput over a=3 depending on the window — a bad trade at any budget
   where BPB is the currency.
4. **a=4 is not a candidate on throughput**: +0.4% (window A) / +3.5% (window B)
   over a=3 for 33% more wall-clock per optimizer step, and one quarter fewer
   updates at the same lr. Its 120 MB BPB was not measured (§3); it cannot beat
   a=3 on rate, and fewer updates at the same lr is the wrong direction on BPB.
5. **Peak VRAM is unchanged** (5.40 GiB), so accumulation is the only legal way
   to grow the effective batch on this 16 GB card.

Watch during the 2 GB run: the first 120 MB should reproduce 1.4943 ± noise on the
same eval window; a lower number means the scaled lr is doing something the 30 MB
screen did not see.

---

## 7. What could skew this / what was not measured

1. **Machine-state drift is ±10-15% on this box, and it is bigger than the
   effect at a=4.** Observed at the same recipe: 145.6 ms/step (05:05, another
   lane), 164.3 (14:03), 172.0 (14:15-14:55), 389.3 → 350.0 at a=3 (14:06 vs
   15:47), 980 → 880 at a=8. The five *speed* arms are one contiguous window, so
   their ratios are controlled; the iso-byte arms ran in separate windows and
   their rate column is **not** a comparison (it is omitted from §3 for that
   reason). A second five-arm speed pass was run at the end as a drift control —
   see §7.1.
2. **Single seed per arm.** The 120 MB noise floor used here (~0.02 BPB) is the
   repo's own screening rule, not a measured seed sigma at this budget. a=3's
   +0.0038 needs no more seeds; a=8's +0.034 is 1.7× the screening threshold and
   would be confirmed by one more seed (2 × 30 min) if anyone wants to run a=8.
3. **The lr screen, when it runs, is a 30 MB screen.** Three of its four arms are
   1 221 steps; any absolution it gives the linear rule is a direction, not a
   2 GB law (§5 says it did not run here).
4. **Both BPB arms share one eval window** (`off=0`, `eval_bytes=196608`), so
   they are comparable to each other and to the ladder's control; they are not
   comparable to any 65 536 B or `off≠0` number in the repo.
5. **Code moved under the sweep.** `HEAD` advanced during the day (`site/*`
   commits) and `port/logs/` was wiped by another lane at 14:01, which is why the
   artifacts live outside the repo; the four Python files whose hash matters were
   recorded and re-checked (§1, §7.1). One `config.py` edit landed between the
   speed pass and the iso-byte pass: a **comment** only (`git diff`: one word in
   a docstring), so the two families share one code path.
6. **No kernel-level explanation here.** The 116.9 ms micro-batch is 55% of the
   card's 88.7 TFLOP/s pass on 5.66 TFLOP of fwd+bwd arithmetic [C]; the audit's
   own pointwise-seam estimate (27.6 ms) and the GEMM floor (~50 ms) leave room,
   but this sweep measures the *step*, not where its milliseconds go.
7. **Not measured:** `b32 × s1024` (does not fit), `grad_accum` with `doc_masking`
   on, `grad_accum` with the slow (fp32-NS) hand, and any 2 GB arm — the
   projection is arithmetic over measured 120 MB ratios, marked [C].

### 7.1 Drift control: the second five-arm pass

`speed2_*` is the same five-arm speed pass re-run at 16:57-17:04 from the same
script, in one window, after the card had been through the afternoon queue. It
answers one question: do the ratios move when the machine does?

| accum | ms/step (14:03) | ms/step (16:57) | drift | × (14:03) | × (16:57) | ratio drift |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 164.3 | 148.0 | −9.9% | 1.000 | 1.000 | — |
| 2 | 278.0 | 251.8 | −9.4% | 1.182 | 1.176 | −0.5% |
| 3 | 389.3 | 353.7 | −9.1% | 1.266 | 1.255 | −0.9% |
| 4 | 517.0 | 456.0 | −11.8% | 1.271 | 1.298 | +2.1% |
| 8 | 980.0 | 938.5 | −4.2% | 1.341 | 1.262 | −5.9% |

The machine got ~10% faster; the **ratios at a≤4 moved ≤1%**. That is the claim
the recommendation rests on, and it is why §6 quotes a ratio, not a rate. The
05:05 ladder control (145.6 ms/step, 56.3 kB/s) sits inside this drift band, so
the "two eras" of §2.1 are one machine in two states, not two code versions.

a=8 is the exception: its ratio moved 5.9%, because at 0.95 s/step a 25-step
window is 24 s and one disturbed window moves the mean. Its per-microbatch cost
is in fact the most stable number in the sweep — `(938.5−45.9)/8 = 111.6 ms` and
`(980.0−44.9)/8 = 116.9 ms`, against a=1's 103.1 and 119.4 — so at a=8 the card
is in a steady power state while a=1 is bursty. Read the a=8 row as "1.23-1.34",
not as a point.

---

## 8. Reproduction

```bash
cd /home/sehaxe/dormouse/port
# speed curve: 5 arms, ~8 min of GPU, one lock slot each
PASS=speed bash /tmp/opencode/accum/run.sh
# iso-byte arms: one lock slot per arm, ~30-40 min each at a<=3
for a in 1 2 3 4 8; do bash scripts/gpu_lock.sh bash /tmp/opencode/accum/bpb_one.sh $a; done
# lr screen: 4 arms, ~30 min, one slot
bash scripts/gpu_lock.sh bash /tmp/opencode/accum/lr30.sh 3 lr30_a3 lr30_a3_x3 9e-4 lr30_a3_xsqrt3 5.196e-4
# the table, no hand-typed numbers
python3 /tmp/opencode/accum/table.py
```

`/tmp/opencode/accum/` holds `run.sh`, `run2.sh` (the speed pass, once per
window), `bpb_one.sh` (one 120 MB arm, with the per-pid VRAM monitor),
`lr30.sh` / `lr_one.sh` (the lr screen), `probe_mem.sh`, `table.py` and every
JSONL; the durable copy is `/home/sehaxe/logs/accum/`.

The `*_dead_*` JSONLs are aborted runs (no eval line) kept as evidence and used
for no number: `bpb_a3_dead_1540.jsonl` is the a=3 arm killed at step 4100 by a
harness restart, `lr30_a1_dead_1716.jsonl` the lr arm killed at step 450.

The only file this work adds to the repository is this report: no source,
config, test or script was modified, and nothing was committed.
