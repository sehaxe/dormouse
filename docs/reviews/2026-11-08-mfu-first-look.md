# 2026-11-08 — MFU first look: the dense control at b8 × s1024

**Mode:** read-only. Code read, arithmetic, and the repo's own measured bills. No
GPU run, no commit, no profiler. The kernel-by-kernel audit is a sibling document
owned by another agent; this one answers the MFU question and states what it does
to the candidate ranking.

**Evidence marks:** **[M]** measured in this repo (source named) · **[D]**
arithmetic from a measured or specified constant · **[S]** inference, flagged.

---

## 1. The answer

The claim under test: the ladder's control arm logs `tokens_per_s = 56 135` at
step 7 324 (`/home/sehaxe/logs/ladder/control_refetch.jsonl`), which at b8 × 1024
is 146 ms/step; at 6N per token with N = 95 035 200 that is 4.67 TFLOP/step,
i.e. 32 TFLOP/s, i.e. **~73 % of a ~44 TFLOP/s bf16 peak**.

**Verdict: (a) — the rate is real wall-clock work — but the 73 % is wrong on both
factors.** The bill is 21 % low (6N omits the T-dependent attention core) and the
denominator is another card's number (44 / 47.7 TFLOP/s is the 5060 Ti's
FP16/FP32-accumulate tensor peak, not the 5070 Ti's). On the only denominator that
belongs to this card — 88.7 TFLOP/s — the dense control is at **43.8 % MFU**, the
top of the repo's own dense band, not 73 %.

Two things follow, and they point in opposite directions from the brief:

- **`tokens_per_s` is not hiding anything.** It is this window's full wall clock,
  optimizer included (§1.1). So this is a genuine ~39 TFLOP/s step, not an
  accounting artifact: **(b) is false.**
- **"Near the hardware floor" is not established.** 9.4 h is 1.23× the 47.7
  floor, **2.29× the 88.7 floor**, and 4.56× the FP8 floor (§1.6). Which floor is
  real depends on the one measurement nobody has taken: a cuBLAS bf16 GEMM on
  this card.

### 1.1 (b) is false: `tokens_per_s` is the whole step's wall clock

The instrument is `StepTimer` (`port/src/dormouse/instrument.py:227`,
`timer.window_s()` = `perf_counter() - mark`). The log line reads it **before** it
resets the mark, and the reset happens on every log boundary:

- `interval_s = timer.window_s()` — `port/src/dormouse/train/loop.py:2291` (the
  2026-11-08 tree; the file is being edited, the expression is the anchor)
- `timer.mark = time.perf_counter()` — `loop.py:2301`
- `rate = (cfg.log_every * tokens_per_step) / interval_s` — `loop.py:2302`

The window therefore contains **everything the step does**: `stream.next_batch`,
forward, backward, `opt_stage()` (clipping + the mixed Muon+/AdamW step), the NaN
firewall counter, EMA (off here, `ema_decay=0.0`), and the host-table update (off,
`use_engram=false`). `timer.tick()` is called after the optimizer stage
(`loop.py:2211`). The window also ends with a device sync:
`window.drain_if_due(...)` reads the device counters before `interval_s` is taken,
so no GPU work is still queued behind the reading.

Two boundary effects, both **lowering** the reported rate, not inflating it:

1. The eval sits **after** the log block (`loop.py`, the `if cfg.eval ...` branch
   below the sink write), and the mark has already been reset, so eval time is
   charged to the **next** window. `port/scripts/bench_engine.py:13` says the same
   thing in the instrument's own words: "eval wall time enters the next telemetry
   window and would be read as a slower step". At 24 eval batches per window
   (`control.log`: `over 196608 B (24 x 8 x 1024)`) and a forward-only pass at
   ~1/3 of a step, that is ~1.2 s of ~534 s = **0.2 %**.
2. Checkpoints are off in the ladder (`ckpt_every=1000000`,
   `port/scripts/recipe_ladder.sh`), so nothing else is charged to the window.

The windows agree to 0.24 % once warm:

| step | `tokens_per_s` | ms/step [D] | note |
|---:|---:|---:|---|
| 3 662 | 49 812.9 | 164.46 | cold start (first cuBLAS/SDPA calls, allocator growth) |
| 7 324 | 56 135.6 | 145.93 | steady |
| 10 986 | 56 181.5 | 145.81 | steady |
| 14 648 | 56 269.6 | 145.59 | steady |

Median of the last three: **56 181.5 B/s = 145.81 ms/step**. Use that. The
reported rate is per window, not cumulative: a cumulative average would decay
toward the steady rate from above; this one starts slow and converges (the
2026-10-17 fix reset the mark, `loop.py`'s comment on the rate line says why).

So: **the optimizer is inside the interval.** (b)'s example mechanisms do not
exist in this code. What remains of (b) is only the possibility that the *work
counted* is not the work executed — and the counters refute that: `forward=3662`
and `transformer_block=43944 = 12 × 3662` per window, `nan_skipped=0`. Twelve
blocks ran per forward, every forward ran, none was skipped.

### 1.2 6N is 21 % below the executed bill

`flagship3_dense` is a plain pre-LN transformer, `d=832`, `L=12`, `d_ffn=3072`,
`V=256` (`port/configs/flagship3_dense.toml`). Instantiated params:
**95 035 200** [M] — reproduced by `port/scripts/mod_ledger.py`
(`--preset flagship3_dense`), not taken from the run banner.

The applicable bill is `mod_ledger`'s, and it was checked against
`torch.utils.flop_counter.FlopCounterMode` at b1 × s1024
(`docs/reviews/2026-11-07-metrics.md:140-143`):

| term | formula | MFLOP/byte (fwd+bwd) | evidence |
|---|---|---:|---|
| block (qkv, proj, MLP) | `3 · 2·L·(4d² + 2·d·f)` | 567.410 | matches the counter |
| attention core | `3 · 4·L·T·d`, T = 1024 | 122.683 | analytic; the CPU counter does not register SDPA |
| head | `3 · 2·d·V` | 1.278 | matches the counter |
| **executed total** | | **691.372** | [M] on `568.689` of it; [D] on the T-term |
| **6N (the brief)** | `6 · 95 035 200` | **570.211** | 82.5 % of the executed bill |
| measured block+head | FlopCounterMode | 568.689 | [M] `2026-11-07-metrics.md:142` |

`691.372 − 570.211 = 121.161`: **6N omits exactly the attention core.** The
attention is `is_causal` SDPA in bf16 (`port/src/dormouse/model/transformer.py`,
`F.scaled_dot_product_attention`), so it is real tensor-core work and belongs in
the bill.

| | TFLOP/step @ 8192 B | TFLOP/s @ 145.81 ms |
|---|---:|---:|
| 6N | 4.671 | **32.01** |
| executed bill | 5.664 | **38.81** |

That is +21.3 % on the throughput. The brief's 32 TFLOP/s is a floor, not the
rate.

### 1.3 The denominator is the 5060 Ti's number, not this card's

The card in this box is an RTX 5070 Ti (16 303 MiB, `nvidia-smi`; the driver's
max SM clock reads 3 090 MHz, so the nominal clock below is not an optimistic
assumption). The memo's own §0.3
(`docs/research/2026-11-05-small-model-mfu.md:102-124`) settles what its tensor
peak is:

- 8 960 CUDA cores, **70 SMs**, 2.45 GHz, 43.9 TFLOP/s FP32;
- **177.4 TFLOP/s** dense tensor FP16/FP16-accumulate;
- **≈ 88.7 TFLOP/s** FP16/**FP32**-accumulate (the rate bf16 training uses —
  `mm_mixed` runs `x.to(bf16) @ w.to(bf16).t()`, i.e. cuBLAS bf16 with fp32
  accumulation, `port/src/dormouse/model/tsct.py:259-292`).

Against the three denominators in circulation:

| denominator | value | provenance | 32.01 TF/s | 38.81 TF/s |
|---|---:|---|---:|---:|
| 5060 Ti measured training ceiling | 45.7 | purchase report, memo §0.3 [M] | 70.0 % | 84.9 % |
| "this box", pinned | 47.7 | `port/scripts/bench_compile.py:93` — **no cited measurement** | 67.1 % | 81.4 % |
| **5070 Ti spec, bf16/FP32-acc** | **88.7** | memo §0.3, Wikipedia footnote [M] | **36.1 %** | **43.8 %** |
| 5070 Ti spec, FP16/FP16-acc | 177.4 | memo §0.3 [M] | 18.1 % | 21.9 % |

The 44 TFLOP/s premise and the 47.7 pin are the **same card's** number. Three
independent checks say 47.7 cannot be the 5070 Ti's ceiling:

1. **Same architecture, twice the SMs.** The 5060 Ti (36 SMs, 2.57 GHz) measured
   **48.96 TFLOP/s** in cuBLAS at `M=16384, N=2048` — **103 % of its 47.4 spec**
   (memo §1.1). Scale by `(70/36) · (2.45/2.57) = 1.853`: **87.9 TFLOP/s** for
   this card, which is the memo's 88.7. A 47.7 ceiling would make cuBLAS 0.54× of
   this card's spec while it is 1.03× of the other card's — a 1.9× regression
   between two sm_120 cards on one cuBLAS call, with no measurement saying so.
2. **The repo's own b16 × s1024 training spike.** 20.5 TFLOP/s compiled on the
   5060 Ti (memo §1.1, [M]; `docs/research/2026-10-15-torch-spike.md` §4 — a
   518.5M loop-block-lite with 342 MFLOP/token executed); scaled by 1.853 =
   **38.0 TFLOP/s** expected here. The dense control measured **38.81** — 1.02×
   the anchor's per-SM efficiency. If the peak were 47.7, this 95M/b8 arm would
   be at 81 % MFU while the 518.5M/b16 spike on the smaller card was at 46.6 % —
   a smaller model at half the batch beating the anchor by 1.7×. Nothing else in
   the tree shows that.
3. **The single-GEMM measurement.** `docs/research/2026-11-08-fp8-on-sm120.md`
   measured the dense stack's real GEMM `[8192, 832] @ [3072, 832]ᵀ` at
   **1.037 ms in bf16** = **40.4 TFLOP/s**, on a card that was contended at the
   time ("absolute milliseconds are inflated; the ratios are what this note is
   about") — so 40.4 is a **lower bound**. The whole 146 ms step runs at
   38.81 TF/s = **96 % of one good GEMM's measured rate**.

The pin is not just uncited, it is **enforced**: `port/tests/test_bench_compile.py:71`
asserts `PEAK_TFLOPS == 47.7`, and `:86-88` assert that 88.7 must miss the
tolerance ("the 5070 Ti's spec 88.7 is 46 % off"). The document that introduced it
(`docs/reviews/2026-11-06-compile-vanilla.md`) says in its own header "every GPU
number is UNMEASURED" and then calls 47.7 "the card's measured bf16 peak". The
memo it cites does not contain 47.7. **This pin is the origin of the 73 %.**

### 1.4 What the memo's three ledgers imply at this run's rate

The memo's §0.2 ledgers are **bills of the 199M KDA/MoE m400**, not of this arm.
Applied to 56 181 B/s anyway — which is the only way to answer "what do they
imply" — they give:

| ledger | MFLOP/byte | TFLOP/step | TFLOP/s [D] | % of 88.7 TF | 100 %-floor @ 88.7 |
|---|---:|---:|---:|---:|---:|
| A — attention-only | 38.9 | 0.319 | 2.18 | 2.5 % | 2 280 kB/s |
| A′ — attention-only, GQA 4 | 27.1 | 0.222 | 1.52 | 1.7 % | 3 273 kB/s |
| B — full `macs_per_token` | 152.2 | 1.247 | 8.54 | 9.6 % | 583 kB/s |
| B′ — full, GQA 4 | 140.4 | 1.150 | 7.88 | 8.9 % | 632 kB/s |
| C — MoE dense-down | 84.1 | 0.689 | 4.72 | 5.3 % | 1 055 kB/s |
| 6N (the brief) | 570.2 | 4.671 | 32.01 | 36.1 % | 155.6 kB/s |
| **executed bill (§1.2)** | **691.4** | **5.664** | **38.81** | **43.8 %** | **128.3 kB/s** |

The three memo ledgers imply **2.5–9.6 % MFU** at this rate. That is not a
finding about the card; it is the ledger error the memo itself warns against at
`docs/research/2026-11-05-small-model-mfu.md:97-100` ("Any MFU number must name
its ledger and its peak"). They price `iters · (attn + out_proj + d·f)` for the
looped KDA/MoE block at d=1536, 512 experts — a per-byte bill 4.5–17.8× smaller
than this dense 12-layer arm's. They do not have a 95M variant, because they are
not parameter-count ledgers; they are architecture ledgers. **The only applicable
number is 691.372, and it is the one the run executes.**

If the question was "which ledger should the 73 % have used", the answer is
neither 6N nor any of A/B/C: it is 691.372, the config's own bill.

### 1.5 The 13 % utilisation figure cannot describe this arm

`AGENTS.md` §3.1 records mean GPU utilisation **13.3 %**, but at **9 195 854
params**, batch 32, in the **Rust/cubecl** lane, on 2026-09-29 — a launch-bound
KDA-ish step of 828 ms. It is not this program. Arithmetically it cannot be:
5.664 TFLOP over 13.3 % of 145.81 ms is **292 TFLOP/s**, 3.3× this card's
FP16/FP32-accumulate spec and 1.6× even the FP16/FP16-accumulate number. A step
that is 87 % idle cannot do this much FLOPs. (The memo's other utilisation
reading, 83 % during m400b, `docs/research/2026-11-05-small-model-mfu.md:223`, is a third
number; the three are measurements of three different programs.)

### 1.6 Consequence for the candidate ranking

The measured facts are: 5.664 TFLOP/step at 38.81 TFLOP/s; the step is at 96 % of
the repo's own single-GEMM rate at the live shape; and the only way to double the
denominator is FP8. What that does to the ranking:

| candidate | ceiling the MFU evidence allows | evidence | verdict |
|---|---|---|---|
| **FP8 GEMM + fused e4m3 quantize** | **2× denominator**; 1.97× on the GEMM, 1.09× end-to-end without a fused quantize, ~1.4–1.6× with one | `docs/research/2026-11-08-fp8-on-sm120.md` [M] | **the only lever that raises the denominator**; a kernel project, not a flag |
| **`grad_accum` (more bytes per optimizer step)** | bounded by the non-GEMM/fixed share. On the measured single-GEMM rate (40.4 TF/s, a lower bound) that share is ~4 % → **1.03×**; at 45 TF/s it is 14 % → 1.11×; at 55 → 1.28×; at 88.7 → 1.73× | the sweep is written (`port/scripts/ab_batch_lr.sh`, arms 8/32/64 KB), **NOT RUN** | one line, price it first; the "dispatch-bound" premise in `eb285e3`'s message is a hypothesis, not a measurement |
| **`torch.compile`** | 1.20× static / 1.31× dynamic measured at 518.5M/b16 on the 5060 Ti; **off** in this recipe (`torch_compile: bool = False`) | `docs/research/2026-10-15-torch-spike.md:228-230` [M] | cheapest test after accum; its ceiling is the same unknown fixed share |
| **bigger batch (8 → 16)** | ≤1.11× (the 9.2M batch ladder: +11 %, then +6.5 %) and b32/s1024 does not fit 16 GB | `AGENTS.md` §3.1 [M]; `port/scripts/train_budget.sh` | **dead**: M = 8192 is 64 tiles = 0.91 wave on 70 SMs already (memo §3.1) |
| **per-GEMM tile/kernel work** | the gap between 40.4 (contended) and the true peak at `M=8192,N=3072,K=832` — unmeasured | `docs/research/2026-11-08-fp8-on-sm120.md` [M] | needs the one GEMM measurement before it can be ranked |

**The 2 GB run.** The budget script's default is `BYTES=1900000000`
(`port/scripts/train_budget.sh`), and 1.9 GB at 56 181 B/s is **9.40 h** [D];
2.0 GB would be 9.90 h. The same 1.3136e18 FLOPs at 100 % of each denominator:

| peak | floor | 9.40 h is |
|---|---:|---:|
| 47.7 (the pin) | 7.65 h | 1.23× |
| 88.7 (this card's spec) | 4.11 h | **2.29×** |
| 177.4 (FP8) | 2.06 h | 4.56× |

These are 100 %-of-spec references, like for like; the probe's *measured* FP8
GEMM is 79.6 TF/s, 45 % of its own 177.4 spec, so none of the three is an
achievable time without a kernel.

So "near the hardware floor" is true only against the number that has no
measurement behind it. Against the card's spec it is 2.3× above the bf16 floor,
and it is 4.6× above the FP8 floor. **The run is not at the floor; it is at the
top of the dense-MFU band for this program, which is a different statement.**

The ladder's *quality* ranking (control/vr/gate/clip/nope/cooldown/all) is
unaffected: every arm runs the same 691.372 bill. This ranking is the performance
one.

---

## 2. What this does not settle, and the two measurements that would

1. **The card's real bf16 ceiling.** Everything above hangs on 47.7 vs 88.7, and
   no run in this tree measures the peak on the 5070 Ti (the FP8 probe's 40.4 is
   one shape, contended). The memo already names the test
   (`docs/research/2026-11-05-small-model-mfu.md:121-124`): one cuBLAS bf16 GEMM at
   `M=8192, N=1536, K=1536`, compared to 88.7. If it reads ~45 TF, the pin stands
   and the step is at 81 % MFU of the executed bill; if it reads 80+, the 73 %
   dies and the ranking above is the one to use. This is one command and it
   decides more than any kernel profile.
2. **The fixed per-step share.** The MFU number bounds it only from one side: the
   step runs at 96 % of the single-GEMM rate, so *if* that (contended) rate is
   the card's real GEMM rate, there is ~4 % left for accum/compile to recover
   (~1.03×). The share grows only if the true GEMM rate is higher: 14 % at
   45 TF/s (1.11×), 29 % at 55 (1.28×), 56 % at 88.7 (1.73×). The
   written-but-unrun `ab_batch_lr.sh` arms (8/32/64 KB per optimizer step) price
   it in ~1 h.

Until both exist, the honest sentence is: **the step does real tensor work at
38.8 TFLOP/s, 43.8 % of this card's specified bf16/FP32-accumulate peak; the
lever that doubles the denominator is FP8 (a kernel project), batch is dead,
compile and accum are unpriced one-line tests, and the "hardware floor" claim is
one unmeasured denominator away from being 2.3× wrong.**

---

## 3. Reproduce (CPU only)

```
cd port
.venv/bin/python scripts/mod_ledger.py --preset flagship3_dense --also '' --seq-len 1024
# -> 95,035,200 params; 691.37 MFLOP/byte

CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q tests/test_metrics_audit.py \
    -k "transformer_ledger"          # -> 1 passed (the T-term is in the ledger)

# the rate and the bill (python3, no torch):
#   ms = 8192 / 56181.47483904764 * 1e3                    -> 145.813
#   tf = 691.372e6 * 8192 / (ms/1e3) / 1e12                -> 38.81
#   6N: 6*95_035_200 * 8192 / (ms/1e3) / 1e12              -> 32.01
```
