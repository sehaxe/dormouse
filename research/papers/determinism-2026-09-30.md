# Cross-process reproducibility of model init, by measurement

Lane `wt/determinism3`, 2026-09-30, off `3234ecc`. Binary
`target/release/train`, `small`, `--batch 2 --seq-len 128 --no-kda --no-engram`,
**`--steps 0` unless stated (init determinism; no step is taken)**, `--seed 7`,
every run its own process. Instrument and harness: `tools/determinism/`.

**BOTTOM LINE. `--seed` works.** Two same-seed processes agree on **7 951 694
of 7 951 694** non-TSCT parameter slots, bit for bit. The only thing that moves
is the TSCT factorisation — at most **1.04e-06** relative on the effective
weight matrix, and in one pair not at all. A different seed moves the effective
weights by **1.415**. Six orders of magnitude apart. **"3 seeds per arm"
(§1.2) is implementable on this box today.**

This agrees with `57237c3` (independently measured, same conclusion, same
numbers to within a ULP). Where this lane adds something, §5 and §6.

---

## 0. The instrument, and why the brief's numbers were contaminated

The brief's figures compared the checkpoint **file**: 73 592 112 B / 4 =
**18 398 028** f32 slots. The `small` model record is **9 197 390** named slots.
The brief's denominator therefore contains the optimizer record, the
EMA-teacher record, and **1 586 slots of inter-tensor padding**.

That padding is dirty. Reading the model record flat:

| run | flat slots | named slots | padding | `\|v\|>=1e30` | **inside a named tensor** |
|---|---|---|---|---|---|
| A | 9 198 976 | 9 197 390 | 1 586 | 244 | **0** |
| B | 9 198 976 | 9 197 390 | 1 586 | 252 | **0** |
| C | 9 198 976 | 9 197 390 | 1 586 | 248 | **0** |
| D | 9 198 976 | 9 197 390 | 1 586 | 247 | **0** |
| G | 9 198 976 | 9 197 390 | 1 586 | 243 | **0** |

`gapcheck.py` shows every one of them sits in the padding between tensors, none
is a parameter, and their positions move run to run because the padding is
whatever the allocator left there.

**One correction to `57237c3`'s account.** AGENTS.md records the NaN/1e30
figures as "a parser bug reading tensor offsets from base 0". That is half of it.
The slots are **really there in the file** — 244-252 of them, in the padding —
and a parser is *also* needed to not count them. Both halves matter: the bytes
exist, and only the offset solve excludes them. `dmck.py` parses the burnpack
record into named tensors with solved offsets and self-validates (a wrong base
does not raise, it silently returns a shifted window, so the module asserts
max|v| < 1e6 over finite slots). At the correct base: **0 NaN and 0 `|v|>=1e28`
across all 9 197 390 slots of every run measured.**

---

## 1. Deliverable 1 — do 0.167 % / 12.50 % reproduce? **NO.**

My six runs, all 15 pairs (`verdict.py`):

| pair | gap | non-TSCT slots differing | slot-count % | max \|delta\| | worst \|\|dW\|\|/\|\|W\|\| |
|---|---|---|---|---|---|
| d3a/d3b | 100 s | **0** / 7 951 694 | 10.73 % | 6.147e-07 | 1.019e-06 |
| d3a/d3c | 492 s | **0** | 12.28 % | 4.768e-07 | 7.817e-07 |
| **d3a/d3d** | 860 s | **0** | **0 %** | **0.000e+00** | **0.000e+00** |
| d3a/d3e | 1010 s | **0** | 11.83 % | 4.843e-07 | 1.014e-06 |
| d3a/d3f | 1214 s | **0** | 11.90 % | 4.685e-07 | 1.014e-06 |
| d3b/d3f | 1114 s | **0** | 10.73 % | 5.700e-07 | 1.002e-06 |
| d3e/d3f | 204 s | **0** | 11.02 % | 4.685e-07 | 9.072e-07 |
| **A/G** *(seed 8)* | 1304 s | **7 944 705** / 7 951 694 | 100 % | **8.373e+00** | **1.415e+00** |

*(inherited same-seed pairs A/B, A/C, B/C at 68/893/825 s: 0 direct slots,
dW/W 9.1e-07…1.02e-06 — same regime.)*

**0.167 % does not reproduce.** The closest same-seed pair is 100 s apart and
differs in **10.73 %** of slots. The *slot count* reproduces (10.7-12.5 %); the
0.167 % does not; and neither number means what it looks like, because:

- **Counting differing slots is the wrong statistic.** An f32 whose last bit
  moved counts the same as one that moved by 1.0. Every same-seed difference is
  at most **6.2e-07**, i.e. **~5 ULP at magnitude 1** (f32 eps = 1.192e-07).
  `12.50 %` and `0.167 %` are two readings of the same ~5-ULP phenomenon; their
  ratio is meaningless.
- The material quantity is the error in what the layer computes,
  `W = U diag(s) V^T`: **7.8e-07…1.04e-06** relative, against **1.415** for the
  seed-8 control.
- **"Seconds apart vs minutes apart" is dead.** 100 s, 150 s, 204 s, 368 s,
  492 s, 518 s, 722 s, 760 s, 860 s, 910 s, 1010 s, 1114 s, 1214 s all give
  0…1.04e-06, and `d3a/d3d` is **bit-identical at 860 s**. Accumulated machine
  state (cubecl high-water pool, allocator layout) cannot order a bit-identical
  pair 14 minutes after its twin alongside a 1e-6 pair 100 s after its twin.

### 1b. A checkpoint file can never be byte-equal across processes

Raw `sha256` of the model record differs for all five of my runs — but hashing
**only the data section**, `d3a` and `d3d` are **identical** while the record is
not:

```
data-section sha256 (named parameter bytes, metadata excluded)
  d3a 46e8998ee8f19ae7ef812f89feabcacf  \
  d3d 46e8998ee8f19ae7ef812f89feabcacf  /  identical parameter values
  d3b 4c324f78...  d3c 8e036433...  d3e c44d52aa...
metadata equal for d3a/d3d? False      data region equal? True
```

The cause is `ParamId`, a **process-global counter** serialised into the CBOR
metadata. So even a perfectly reproducible run yields a different *file*, and
`cmp`/`md5sum`/any byte-level checkpoint diff are **structurally incapable** of
being zero across processes. Compare parameter tensors, never files.

---

## 2. Deliverable 2 — the cause

**Verdict: per-process reduction-order nondeterminism inside `qr_householder`,
amplified over its 64 sequential iterations. Not the memory pool, not the
allocator, not an unseeded draw.**

`vendor/burn-fused/crates/burn-spectral/src/lib.rs:457-469` — every TSCT factor
is the Q of a QR of a random normal matrix:

```rust
let rand = Tensor::<2>::random([in_features, k], Normal(0.0,1.0), device); // :457
let (q, _) = qr_householder(&rand);                                      // :462
```

`qr_householder` (`:689`) loops `j in 0..n.min(m)`, `n = rank = 64`. Each
iteration holds a full reduction plus **three host syncs**: `col.mul(col).sum()`
(`:697`), `col.into_scalar()` (`:698`) feeding a **host-side branch on a device
value** (`:699`), `norm.into_scalar()` (`:702`), `v.mul(v).sum()` (`:704`),
`v.div_scalar(...into_scalar())` (`:705`), and two `matmul`s (`:711`, `:714`).

Four independent measurements, all pointing at that loop and nowhere else:

1. **The RNG is provably correct.** All **7 951 694** non-TSCT slots (embedding,
   controller, norms, every direct draw) are **bit-identical in every same-seed
   pair, across 9 processes, gaps 68 s…1304 s**. A desynchronised or unseeded
   stream would corrupt the first divergent draw and everything after it.
2. **Divergence enters at a random iteration, not a fixed one.** Column `j` of Q
   is finished by iteration `j`, so the first differing column reads out which
   iteration first disagreed. Over 6 runs it ranges **0…55, median 2**, and
   **139 of 360 tensor-pairs are entirely identical** (`entry.py`). A fixed seed
   bug, a fixed code path or a fixed pool state would pin that column; it does not.
3. **It is not the `sign` branch.** That branch would show whole negated columns.
   Measured: **0 of 64 columns** with cosine < -0.9 in any tested tensor; the
   difference is a sign-preserving scatter (`structure.py`).
4. **Not machine state** — see §1, point 3.

Amplification is structural: 64 *sequential* iterations each consuming the
previous one's reduction, so a 1-ULP disagreement compounds as a random walk
(`sqrt(64) ~ 8x`) — hence ~5 ULP rather than 1, and hence the effect appears only
in TSCT factors (nothing else in model init runs an iterative factorisation).

**No unseeded draw exists to report.** The defect is a nondeterministic
*reduction*; the sites are the `.sum()`/`matmul` calls at
`burn-spectral/src/lib.rs:697`, `:704`, `:711`, `:714`.

**Confidence, stated precisely.** That the divergence is a reduction-order
difference inside this loop is measured (points 1-4). That it is specifically
**the cubecl autotuner** choosing a different kernel per process is a *plausible
and consistent* hypothesis — I did not measure the autotuner's decisions — and
it is `57237c3`'s claim. Recorded as hypothesis, not as this lane's evidence.

**Reported, not fixed** (a fix is a separate decision; `:698-705` are also an
AGENTS.md §1.3 violation — host-device syncs and a host branch on a device value
inside a 64-iteration init loop). Candidates: a fixed-order reduction, or
host-side QR on the already-good CPU path, or seeding the factors directly
instead of QR-ing on device (`57237c3`'s follow-up, which would also make a
bit-exact cross-process golden possible).

---

## 3. Deliverable 3 — is "3 seeds per arm" implementable? **YES.**

- **Three seeds give three genuinely different initialisations. YES.** The
  seed-8 control differs from seed 7 in **7 944 705 of 7 951 694** non-TSCT slots
  (99.92 %), `||dW||/||W|| = 1.415`. The seed is a real, effective independent
  variable, which is what §1.2 needs.
- **The same seed gives the same initialisation to within ~1e-6, always.** Worst
  over 15 same-seed pairs spanning 100 s…1214 s: **1.04e-06** on the effective
  weights, **0** on every non-TSCT slot. For a 2k-step A/B judged on BPB against
  a seed-variance bar, a 1e-6 initialisation perturbation is far below the noise
  the protocol already has to beat.
- **Bit-exact checkpoint equality across processes: NO, and never guaranteed.**
  Exact in some pairs (1 of my 15), ~5 ULP in others. **Within one process it is
  exact** — the only place bit-equality is a property, and useless for an A/B.

So **ADR-0002's "3 seeds per arm" is unblocked.** What is *not* available is a
bit-exact-checkpoint diff as an A/B instrument; use BPB, and treat a same-seed
cross-process checkpoint diff as carrying a ~1e-6 noise floor.

The historical claim "every A/B in the archive compared two different
initialisation" was measured pre-`4b42b6d`. Post-`4b42b6d` the only
cross-process nondeterminism is the ~1e-6 QR jitter above. The 409 043 figure
stays banned: prose in five files, no test, never.

---

## 4. Growth during training — and a CORRECTION to the step ladder

Same-seed divergence **does grow** once training starts, from the ~1e-6 seed
running with ordinary float chaos. Ranked by relative movement of the non-TSCT
tensors (`anomaly.py`), inherited sweep:

| steps | top movers (relFro) | TSCT relFro | non-TSCT slots differing |
|---|---|---|---|
| 0 | none | 5.6e-07 | 0 |
| 5 | `iter_embed` 3.9e-06, `controller.weight` 1.0e-07, `jepa_pred.norm.beta` 4.7e-08 | 7.1e-07 | 40 163 |
| 20 | `jepa_pred.norm.beta` **6.4e-03**, `iter_embed` 2.2e-04, `jepa_pred.proj.weight` 5.9e-05 | 2.5e-05 | 510 862 |
| 50 | — | — | — |
| 100 | — | — | — |

**`aux.jepa_pred.norm.beta` grows monotonically (4.7e-08 → 6.4e-03, five orders),
not anomalously at step 20.** The fastest movers are the small-norm parameters
(`norm.beta` is a [768] scale with a tiny update; `iter_embed` is [3072]), i.e.
ordinary chaotic divergence of the most sensitive coordinates, seeded by the
1-ULP QR difference. So step 20 is **not** an unexplained anomaly and should not
be carried as an open item.

**CORRECTION — the 50/100 rows of the step ladder never ran.** Verified from the
files themselves, not inferred:

| run | header step | `m.prev.bin` | verdict |
|---|---|---|---|
| s5a / s5b | 5 | step 0 | two completed saves — clean |
| s20a / s20b | 20 | step 0 | two completed saves — clean |
| s50a / s50b | **0** | **absent** | **died before completing step 1** |
| s100a / s100b | **0** | **absent** | **died before completing step 1** |

`ckpt_every = 1000` and `step = 0`, so `0 % 1000 == 0` fires a periodic save at
step 0 (`lib.rs:1657`); the final save (`:1681`) never ran. `s5a`/`s20a` have a
`m.prev.bin` holding that step-0 save, proving they got past it. **There is no
trainer bug here** — the two long runs died under GPU contention (a fourth agent
was running on the card concurrently), and their checkpoints are step-0
artifacts. Their "0 non-TSCT slots differing" is a **zero-step** reading.

This is the same data as `benches/determinism.tsv` row 4 ("step ladder
0/5/20/50/100"): that row's `relFro 9.85e-09..7.35e-07`,
`max-abs-delta <=5.14e-05` and `4.62-19.06` slot-% reproduce this sweep **digit
for digit**, so its 50/100 points are those artifacts. Therefore:

- **"no compounding to 100 steps" is not established** — there is no data past
  step 20. The honest statement is "no compounding to 20 steps", and the trend
  in that window is upward, so it must be re-measured on runs that complete.
- **"the anomaly is absent at 0/50/100" is not established** either; 50 and 100
  are step 0.

**This does not threaten the 2k-step A/B** — the comparison is BPB against a
seed-variance bar, not checkpoint equality — but **any claim that two same-seed
runs produced "the same" weights must quote a step index**, and the compounding
rate past 20 steps is genuinely open.

---

## 5. What this lane adds to `57237c3`

1. **The NaN/1e30 slots are real file bytes in inter-tensor padding**, not purely
   a parser bug (§0) — 244-252 per run, 0 inside a named tensor, positions moving.
2. **A checkpoint file is never byte-equal across processes even when the
   parameters are**, because `ParamId` is a process-global counter (§1b).
   `d3a`/`d3d` share a data-section sha256.
3. **Divergence enters the QR loop at a random iteration** (0…55, median 2;
   139/360 tensor-pairs identical) — the strongest single piece of evidence for
   "per-run reduction choice", and independent of the autotuner hypothesis.
4. **One same-seed pair is bit-identical** (860 s apart), which no
   monotonic-with-time account can explain.
5. **The step ladder's 50/100 points are step-0 artifacts** (§4), which
   retracts "no compounding to 100 steps" and the "absent at 50/100" half of the
   anomaly claim.

## Reproducing

```
tools/determinism/run_d3.sh <label>       # one zero-step run, seed 7
tools/determinism/drive_d3b.sh            # GPU-guarded driver used here
PYTHONPATH=tools/determinism python3 tools/determinism/verdict.py  d3a d3b d3c d3d d3e d3f
PYTHONPATH=tools/determinism python3 tools/determinism/entry.py    A B C D E F
PYTHONPATH=tools/determinism python3 tools/determinism/structure.py
PYTHONPATH=tools/determinism python3 tools/determinism/material.py
PYTHONPATH=tools/determinism python3 tools/determinism/gapcheck.py
PYTHONPATH=tools/determinism python3 tools/determinism/anomaly.py
PYTHONPATH=tools/determinism python3 tools/determinism/steps2.py
```

Runs land in `/home/sehaxe/cache/determinism/runs/<label>/m.bin`. The driver
waits for the card to be clear before every run: a fourth agent was on this lane
concurrently, and two GPU processes at once is forbidden (AGENTS.md §1.5).
