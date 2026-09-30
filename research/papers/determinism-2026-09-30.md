# Cross-process reproducibility of model init, by measurement

Lane `wt/determinism3`, 2026-09-30. Commit `3234ecc`. Binary
`target/release/train`, `small`, `--batch 2 --seq-len 128 --no-kda --no-engram`,
**`--steps 0` (init determinism; no step is taken)**, `--seed 7`. Every run is
its own process. Harness + instrument: `tools/determinism/` (inherited from the
two dead agents on this lane, who left it in `/home/sehaxe/cache/determinism/`;
their runs A-H are reused and labelled *inherited*).

**BOTTOM LINE.** The seed works. Two same-seed processes agree on **7 951 694 of
7 951 694** non-TSCT parameter slots, bit for bit. The only thing that moves is
the TSCT factorisation, by at most **1.0e-6 relative on the effective weight
matrix** — and in some pairs not at all. A different seed moves the effective
weights by **1.415**. Those are six orders of magnitude apart, and the gap is not
wall time. **"3 seeds per arm" is implementable on this box today.**

---

## 0. The instrument, and why the brief's numbers were contaminated

The brief's figures came from comparing the checkpoint **file**:
73 592 112 B / 4 = **18 398 028** f32 slots. The `small` model record is
**9 197 390** named slots. So the brief's denominator contains the optimizer
record, the EMA-teacher record, and **1 586 slots of inter-tensor padding**.

That padding is real and it is dirty. Reading the model record flat:

| run | flat slots | named slots | padding | `\|v\|>=1e30` | of those, inside a named tensor |
|---|---|---|---|---|---|
| A | 9 198 976 | 9 197 390 | 1 586 | 244 | **0** |
| B | 9 198 976 | 9 197 390 | 1 586 | 252 | **0** |
| C | 9 198 976 | 9 197 390 | 1 586 | 248 | **0** |
| D | 9 198 976 | 9 197 390 | 1 586 | 247 | **0** |
| G | 9 198 976 | 9 197 390 | 1 586 | 243 | **0** |

So the brief's "~475-501 slots at `|v|>=1e30`" is **true of the file and false of
the model**: every one of them is uninitialised-looking bytes in the padding
between tensors, none is a parameter, and their positions move between runs
because the padding is whatever the allocator left there. A byte-level file
comparison is contaminated by construction. Measured: `tools/determinism/gapcheck.py`.

`tools/determinism/dmck.py` parses the burnpack record into **named tensors** with
solved byte offsets and self-validates the solve (a wrong base does not raise, it
silently returns a shifted window, so the module asserts max|v| < 1e6 over
finite slots). At the correct base: **0 NaN and 0 `|v|>=1e28` across all
9 197 390 slots of every run measured.** This is the instrument used throughout.

---

## 1. Deliverable 1 — do 0.167 % / 12.50 % reproduce? **NO.**

My own runs (`d3*`), plus the inherited `A`-`F` (seed 7) and `G`,`H` (seed 8):

| pair | gap | non-TSCT slots differing | slot-count % | max \|delta\| | worst \|\|dW\|\|/\|\|W\|\| |
|---|---|---|---|---|---|
| **d3a/d3b** | 100 s | **0** / 7 951 694 | 10.73 % | 6.147e-07 | 1.019e-06 |
| d3a/d3c | 492 s | **0** | 12.28 % | 4.768e-07 | 7.817e-07 |
| **d3a/d3d** | 860 s | **0** | **0 %** | **0.000e+00** | **0.000e+00** |
| d3a/d3e | 1010 s | **0** | 11.83 % | 4.843e-07 | 1.014e-06 |
| A/B *(inh.)* | 68 s | **0** | 10.73 % | 5.923e-07 | 9.149e-07 |
| A/C *(inh.)* | 893 s | **0** | 12.28 % | 4.843e-07 | 1.019e-06 |
| B/C *(inh.)* | 825 s | **0** | 11.59 % | 4.098e-07 | 1.014e-06 |
| **A/G** *(seed 8)* | 1304 s | **7 944 705** / 7 951 694 | 100 % | **8.373e+00** | **1.415e+00** |

**0.167 % does not reproduce.** The closest same-seed pair is 100 s apart and
differs in **10.73 %** of slots — the *slot count* reproduces (10.7-12.5 %), the
*0.167 %* does not, and neither number means what it appears to mean, because:

- **Counting differing slots is the wrong statistic.** An f32 whose last bit
  moved counts exactly as much as one that moved by 1.0. Here every same-seed
  difference is at most **6.2e-7**, i.e. **~5 ULP at magnitude 1**
  (f32 eps = 1.192e-07). The `12.50 %` and the `0.167 %` are two readings of
  the same ~5-ULP phenomenon; their ratio is meaningless.
- The material quantity is the error in the **effective weight** the layer
  computes, `W = U diag(s) V^T`: **7.8e-07 to 1.04e-06** relative Frobenius,
  against **1.415** for the seed-8 control.
- **"Seconds apart vs minutes apart" is dead.** 68 s, 100 s, 150 s, 393 s, 492 s,
  825 s, 860 s, 1010 s, 1304 s all give the same 0 to 1.04e-06. And one pair
  860 s apart (`d3a/d3d`) is **bit-identical**. If elapsed time or accumulated
  machine state drove this, a bit-identical pair 14 minutes after its twin
  would not be possible next to a 1e-6 pair 100 s after its twin.

**One inherited pair is invalid, and I am not using it.** Runs `s50a`/`s50b` and
`s100a`/`s100b` of the dead agent's steps sweep logged `step 0`, never
`step 50` / `step 100` (`m.txt` = `step 0 ce 5.569`), so those two pairs did not
train and their "0 differing direct slots" is a zero-step reading wearing a
step-50 label. Only steps 0, 5, 20 are usable from that sweep.

### 1b. The checkpoint file is never byte-comparable, even when the parameters are

Raw `sha256` of the whole model record differs across all five of my runs — but
so does the hash of **only the data section** for four of them, and for
`d3a`/`d3d` it is **identical** while the record is not:

```
data-section sha256 (named parameter bytes, metadata excluded)
  d3a 46e8998ee8f19ae7ef812f89feabcacf    d3d 46e8998ee8f19ae7ef812f89feabcacf  /  identical parameter values
  d3b 4c324f78...   d3c 8e036433...   d3e c44d52aa...
metadata (first 600 B) equal for d3a/d3d? False
data region equal for d3a/d3d?         True
```

The reason is `ParamId`, a **process-global counter** assigned at module
construction and serialised into the CBOR metadata. So even a perfectly
reproducible run produces a different *file*. `cmp`, `md5sum` and any
byte-level checkpoint diff are therefore not merely noisy here — they are
**structurally incapable** of being zero across processes. Compare parameter
tensors (`dmck.py`), never files.

---

## 2. Deliverable 2 — the cause

**Verdict: per-process reduction-order nondeterminism inside
`qr_householder`, amplified over its 64 sequential iterations. Not the memory
pool, not the allocator, not an unseeded draw.**

`vendor/burn-fused/crates/burn-spectral/src/lib.rs:457-469` — every TSCT
factor is the Q of a QR of a random normal matrix:

```rust
let rand = Tensor::<2>::random([in_features, k], Normal(0.0,1.0), device); // :457
let (q, _) = qr_householder(&rand);                                      // :462
```

`qr_householder` (`:689`) loops `j in 0..n.min(m)` with `n = rank = 64`, and each
iteration contains a full reduction plus **three host syncs**:
`col.mul(col).sum()` (`:697`), `col.into_scalar()` (`:698`) feeding a
**host-side branch on a device value** (`:699`), `norm.into_scalar()` (`:702`),
`v.mul(v).sum()` (`:704`), `v.div_scalar(...into_scalar())` (`:705`), and two
`matmul`s (`:711`, `:714`).

Four independent measurements, all pointing at that loop and nowhere else:

1. **The RNG is not the problem — it is provably correct.** All **7 951 694**
   non-TSCT slots (embedding, controller, norms, every direct draw) are
   **bit-identical in every same-seed pair, across 6 processes and gaps from
   68 s to 1304 s**. A desynchronised or unseeded stream would corrupt the
   first divergent draw and everything after it; instead exactly one
   *post-draw* transform is nondeterministic and the draws are perfect.
2. **The divergence enters at a random iteration, not a fixed one.**
   Column `j` of Q is finished by iteration `j`, so the first differing column
   reads out which iteration first disagreed. Over the 6 inherited runs it
   ranges **0 to 55, median 2**, and **139 of 360 tensor-pairs are entirely
   identical** (`tools/determinism/entry.py`). A fixed seed bug, a fixed code
   path or a fixed pool state would pin that column; it does not.
3. **It is not the `sign` branch.** The `sign` at `:699` would show up as whole
   negated columns. Measured: **0 of 64 columns** in any tested tensor have
   cosine < -0.9; the difference is a sign-preserving scatter (`structure.py`).
   The branch never disagrees; the reduction feeding it does, at the ULP level.
4. **Not machine state.** Two runs on the same machine, same binary, 860 s
   apart, came out bit-identical, while two runs 100 s apart differed by 1e-6.
   Accumulated machine state (cubecl high-water pool, allocator layout) would
   order those the other way round.

Amplification is structural: 64 *sequential* iterations, each consuming the
previous one's reduction, so a 1-ULP disagreement compounds as a random walk
(`sqrt(64) ~ 8x`) — which is why the observed magnitude is ~5 ULP rather than
1 ULP, and why the error only appears in TSCT factors (nothing else in model
init runs an iterative factorisation).

**No unseeded draw was found. There is no file:line to report for one.** The
defect is a nondeterministic *reduction*, and the sites are the `.sum()` /
`matmul` calls at `burn-spectral/src/lib.rs:697`, `:704`, `:711`, `:714`.

**Reported, not fixed** (a fix is a separate decision, and `:698-705` are also an
AGENTS.md §1.3 violation — host-device syncs and a host branch on a device value
inside a 64-iteration init loop). The obvious candidates are to make the QR
init path deterministic (fixed-order reduction, or host-side QR on the already
known-good CPU path), or to accept ~1e-6 and stop quoting bit-exactness.

---

## 3. Deliverable 3 — is "3 seeds per arm" implementable? **YES.**

Precisely, because "reproducible" has two meanings and only one of them fails:

- **Three seeds give three genuinely different initialisations. YES.** The
  seed-8 control differs from seed 7 in **7 944 705 of 7 951 694** non-TSCT
  slots (99.92 %) with `||dW||/||W|| = 1.415`. The seed is a real, effective
  independent variable, which is what §1.2 needs.
- **The same seed gives the same initialisation to within ~1e-6, always.**
  Worst observed over 9 same-seed pairs spanning 68 s to 1304 s:
  **1.04e-06** relative on the effective weights, **0** on every non-TSCT slot.
  For a 2k-step A/B judged on BPB, a 1e-6 initialisation perturbation is far
  below the seed variance the protocol already has to beat.
- **Bit-exact checkpoint equality across processes: NO, and it will never be
  guaranteed.** It holds in some pairs (10 of my pairs include one exact match)
  and misses by ~5 ULP in others. **Within one process it is exact** — that is
  the only place bit-equality is a property, and it is useless because an A/B is
  two processes.

So ADR-0002's "3 seeds per arm" is **unblocked**. What is *not* available is a
bit-exact-checkpoint diff as an A/B instrument; use BPB, and treat a same-seed
cross-process checkpoint diff as having a ~1e-6 noise floor.

**Also now settled, against §3.7's open item.** The historical claim "every A/B
in the archive compared two different initialisation" was measured pre-`4b42b6d`,
before `device.seed()` existed. Post-`4b42b6d` the only cross-process
nondeterminism is the ~1e-6 QR jitter above. The 409 043 figure remains banned
(no test, no committed measurement).

**Does it grow during training?** Partially answered, and the answer is the one
number to re-check before trusting a long A/B. From the inherited sweep
(`steps2.py`, steps 0/5/20 only — see §1 for why 50/100 are invalid):

| steps | TSCT relFro | TSCT max \|delta\| | non-TSCT relFro | non-TSCT differing slots |
|---|---|---|---|---|
| 0 | 5.6e-07 | 3.8e-07 | 0 | 0 |
| 5 | 7.1e-07 | 4.5e-07 | 2.8e-10 | 40 163 |
| 20 | 2.5e-05 | 9.4e-06 | 5.1e-07 | 510 862 |

Same-seed divergence **does grow** as training proceeds — ~40x in TSCT over 20
steps — as ordinary float chaos takes the ~1e-6 seed and runs with it. Two
consequences: (a) it does not threaten the 2k-step A/B, whose comparison is
BPB with a seed-variance bar, not checkpoint equality; (b) **any future claim
that two same-seed runs produced "the same" weights must state a step index**,
because "the same" is true at step 0 to 1e-6 and untrue by step 20.

---

## Reproducing

```
tools/determinism/run_d3.sh <label>       # one zero-step run, seed 7
tools/determinism/drive_d3b.sh            # the guarded driver used here
PYTHONPATH=tools/determinism python3 tools/determinism/verdict.py d3a d3b d3c d3d d3e d3f
PYTHONPATH=tools/determinism python3 tools/determinism/entry.py A B C D E F
PYTHONPATH=tools/determinism python3 tools/determinism/structure.py
PYTHONPATH=tools/determinism python3 tools/determinism/material.py
PYTHONPATH=tools/determinism python3 tools/determinism/gapcheck.py
PYTHONPATH=tools/determinism python3 tools/determinism/steps2.py
```

Runs land in `/home/sehaxe/cache/determinism/runs/<label>/m.bin`. The driver
waits for the GPU to be clear before every run: a fourth agent was on this lane
concurrently and two GPU processes at once is forbidden (AGENTS.md §1.5).
