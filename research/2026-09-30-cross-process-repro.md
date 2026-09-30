# Cross-process reproducibility, settled by measurement

Lane `wt/seed-determinism`, 2026-09-30. Instrument: `tools/determinism.py`.
Numbers: `benches/determinism.tsv`. Every row here is reproducible with
`tools/determinism.py check A/m.bin B/m.bin`.

## The question, and the answer in one line

**`--seed` works.** Two processes at the same seed build the same model to a
relative Frobenius distance of **0 to 1.3e-08** (32 non-TSCT tensors never move
at all); two processes at different seeds sit at **1.414**. The separation is
**1.1e8x**, so **ADR-0002's "3 seeds per arm" is implementable** and is
implemented. What is *not* available is a bit-exact cross-process golden.

## What I was told to distrust, and what actually happened

| claimed | verdict |
|---|---|
| 409 043 differing values (~4%), from `4b42b6d` | **Still banned.** `4b42b6d` shipped a `device.seed()` call and a comment. No test, no measurement. Nothing here rehabilitates it. |
| 0.167 % back-to-back | **Not reproducible as a distance, and not the property that matters.** The count of differing slots is unstable across repetitions: I measured **0.0000 %, 0.0809 %, 0.0809 %, 4.38 %, 4.45 %, 7.70 %, 7.83 %, 10.49–12.51 %** for the same nominal condition. A count is not a measurement of distance. |
| 12.50 % minutes later | **The number is real; the interpretation is wrong.** 12.50 % is just a typical same-seed value — I measured 12.5146 % for a pair **80 s apart**, and 12.3 % for pairs 68–103 s apart. |
| "same commands seconds apart differ from the same commands minutes later" | **No such effect. Refuted.** Grouping the 15 seed-7 pairs by gap: ≤200 s → mean relFro 1.088e-08; >800 s → 1.116e-08. **Ratio 1.026.** |
| "the file carries 0–4 NaN slots and ~475–501 slots at \|v\|≥1e30" | **False, and it was an artifact of reading the file wrong.** With the offsets solved correctly: **0 NaN and 0 such slots in all 8 runs** (see *The instrument*, below — I made this mistake myself first). |
| "the memory pool / allocator state / unseeded draws" | **All three excluded.** The optimiser section is 256 bytes at step 0, so no optimiser state is in the comparison. The seed demonstrably reaches the weights (different seed ⇒ 99.92 % of slots move). The mover is one arithmetic routine, identified below. |
| max \|δ\| 2.07e+38 | **An artifact.** The max over a correctly-read pair is **≤6.15e-07**. |

## Protocol

Release trainer `/home/sehaxe/dormouse/target/release/train` (built 2026-09-29
from the shared checkout — it is the instrument here, not the thing under
test, and the seeding call is in it). Fixed corpus: 6 796 bytes, one shard,
md5 `d4ff4c9218d4a9225c60fd138c6e0f63`. One run:

```
train --data <corpus> --preset small --steps N --seed 7 \
      --batch 2 --seq-len 128 --no-kda --no-engram \
      --ckpt-dir <fresh dir> --ckpt-name m
```

`--steps 0` means the training loop never executes (`lib.rs:1111`), so the
checkpoint is the freshly initialised model: `Device::seed(cfg.seed)` at
`lib.rs:852`, then `build_model`, then save. Each run is its own process with
its own fresh checkpoint directory, so nothing resumes. Runs went under
`systemd-run --user --scope -p MemoryMax=40G` (§2.4).

42 comparisons in total: 15 pairs among 6 zero-step seed-7 runs; 1 pair at
seed 8; 12 cross-seed pairs; 6 pairs from 4 further zero-step reps; 2 pairs from
2 earlier reps; 5 step-ladder pairs; 1 round-trip.

## The measurement, and why "differing slots" is the wrong statistic

Two processes, same binary, same flags, same seed, 68 s apart:

```
differing slots : 987,208  (10.7336%)
relFro          : 1.003e-08
max |delta|     : 5.923e-07        <- ~5 ulp of f32 (eps = 1.19e-07)
```

10.7 % of the model "differs" and the two models are the same model to eight
decimal places. A slot whose last bit moved is not a slot that moved. Both
numbers are printed by `tools/determinism.py` for exactly this reason; the gate
is on `relFro`.

The count is unstable because **how many of the 24 TSCT factors move is not
something a run controls**:

| pair | differing slots | relFro | tensors moved | of which TSCT |
|---|---|---|---|---|
| 68 s | 987,208 (10.73 %) | 1.003e-08 | 16 | **16** |
| 107 s | **0 (0.0000 %)** | **0** | **0** | 0 |
| 90 s | 7,440 (0.0809 %) | 1.061e-09 | **1** | **1** |
| 1014 s | 409,528 (4.45 %) | 9.196e-09 | 10 | **10** |
| 103 s | 1,128,099 (12.27 %) | 1.178e-08 | 16 | **16** |

**In 42 comparisons, every tensor that ever moved was a `Tsct` tensor. Not one
of the 30 non-TSCT tensors (7,951,694 slots, 86.5 % of the model) moved in any
pair, at any step count, including a pair that was bit-exact over all
9,197,390 slots.** That is the finding: the seeded RNG is exact, and one
routine is not.

## Root cause

`burn-spectral`'s `SpectralLinear::new` (`vendor/burn-fused/crates/burn-spectral/src/lib.rs:456-469`)
initialises each factor as **the Q of a Householder QR of a random normal
matrix**, and `qr_householder` (`lib.rs:688-715`) is:

- 64 serial iterations (one per column of the `k = 64` rank), and
- **4 device reductions and 3 host-device synchronisations per iteration**
  (`col.mul(col).sum().sqrt()`, `into_scalar::<f32>()` ×2, a full
  `into_data().try_to_vec()` of the column, then `v.mul(v).sum().sqrt()`).

The result is bit-reproducible only if the reduction's summation order is
reproduced. On CUDA that order is chosen by the cubecl **autotuner**, which
picks a kernel by benchmarking candidates at runtime. Whether two processes pick
the same kernel is not under the run's control, which is exactly the 0-of-16 /
1-of-16 / 16-of-16 spread in the table. It is not the memory pool
(`init_pools` is before the seed, and a fresh process has a fresh pool), and it
is not the allocator.

This also resolves the brief's "candidate causes" list: the pool and the
allocator cannot produce a difference that is (a) confined to 24 named tensors
and (b) never exceeds 6.15e-07, and unseeded draws would move everything, which
is precisely what a *different* seed does (99.92 % of slots, relFro 1.414).

## Does it compound?

No, not to 100 steps. Two runs per point, same seed:

| steps | relFro | max \|δ\| | mover |
|---|---|---|---|
| 0 | 1.200e-08 | 3.800e-07 | TSCT |
| 5 | 1.515e-08 | 4.470e-07 | TSCT |
| 20 | **7.353e-07** | **5.139e-05** | **`aux.jepa_pred.norm.beta` 6.4e-03, `loop_block.iter_embed` 2.1e-04 — not TSCT** |
| 50 | 9.851e-09 | 4.917e-07 | TSCT |
| 100 | 1.226e-08 | 5.700e-07 | TSCT |

The perturbation stays at the f32 rounding floor; it does not compound. **The
20-step row is an open anomaly and I am not explaining it**: two small
non-TSCT tensors move 2–4 orders more than anything else, in *both* runs of that
pair, and the effect is absent at 0, 50 and 100 steps. Named, not papered over.

**Not measured: whether this survives 2 000 steps.** Everything above bounds
100. An A/B arm's loss curve at 2 k steps is not characterised here.

## What this means for the A/B queue (ADR-0002)

**"3 seeds per arm" is implementable, and the seed is a usable experimental
condition.** The measured facts:

- a different seed moves the model by `relFro = 1.414`; the same seed moves it
  by `≤ 1.3e-08`. Nothing an A/B can resolve lives between those.
- the property is *checkable in ~3 minutes* and is now a command:
  `tools/determinism.py run --reps 2`, which fails if a same-seed pair exceeds
  `relFro 1e-5` and (with `--seed-differs`) if a different-seed pair fails to
  reach `0.1`. Measured headroom on both sides is 1.1e8x.
- a fixed seed is a *reusable* condition: two runs at seed 7 are the same
  experiment to 1e-8 for at least 100 steps, so an arm and its control are
  comparable.

**What is not available: a bit-exact cross-process golden.** A checkpoint
cannot be diffed byte-for-byte, and no layer-2 golden file can exist for a
model whose TSCT factors are produced by a device QR. If a bit-exact artifact
is ever needed, the fix is one routine: seed the QR's factors directly (they
are orthonormal by construction after one retraction anyway) or run the init
QR on the host. That is a change to `burn-spectral`, a crate another lane owns,
and I have not touched it.

## The instrument, and the mistake I made in it

`tools/determinism.py` parses the burnpack record into its 54 named tensors
(6 KB CBOR metadata + interleaved f32 data) and compares tensor by tensor.
Two things about it are load-bearing and both exist because they caught me:

1. **`data_offsets` are relative to the data section, and the section start
   must be solved for**: `base = len(record) − (Σ sizes + Σ internal gaps)`,
   where the gaps are three padding runs of 208 + 252 + 252 bytes. I first read
   at `base = 0`, which reads every tensor 1 408 floats early. **53 of 54
   tensors still look like plausible f32 data**, so it fails silently — and it
   produced a complete, tidy, wrong table: "43.7 % of tensors bit-exact", "the
   TSCT factors differ by 2 ulp", "12.5 % back-to-back", and a spurious
   "0.4 relative Frobenius for same-seed and different-seed alike", which is
   what finally exposed it. The correct base gives **0 values ≥ 1e6 and 0 NaN
   across all 9,197,390 slots of all 8 runs**; the nearest wrong base gives
   961,989 absurd values.
2. **A check that cannot fail is not a check.** Three guards, each with a
   demonstrated failure: the container header declares every section's length
   (a record truncated by 4 096 B was accepted without it — the base solve
   absorbs truncation exactly); the metadata length field must match the parsed
   metadata; and every tensor must contain only values `< 1e6` (no parameter
   init produces 1e6). All three are exercised by the negative test below.

```
$ python3 - <<'EOF'
import sys; sys.path.insert(0,"tools"); import determinism as D
try: D.load("truncated.bin"); print("NEGATIVE TEST FAILED: accepted")
except SystemExit as e: print("rejected:", e)
EOF
rejected: header declares 73592112 bytes, file has 73588016 -- truncated
```

## Reproduce

```bash
tools/determinism.py run --reps 2 --steps 0     # same seed; gate on relFro <= 1e-5
tools/determinism.py check A/m.bin B/m.bin      # compare two checkpoints
```

Requires a built `train`, a free GPU, and numpy. The gate is the artifact worth
keeping; the 42 comparisons are how the thresholds were chosen.

## Two things another lane should know

- **The checkpoint format aliases its own header.** `embedding.weight` (the
  first tensor in the map) declares `data_offsets = [0, 786432)`, which covers
  the record's CBOR metadata. Read naively as floats it yields ~400 values of
  magnitude 1e28–1.7e38 and occasionally a NaN — **this is where the "0–4 NaN
  and ~475–501 huge slots" in the brief came from.** The save→load→save
  round-trip is **bit-exact over all 9,197,390 slots** (measured), so nothing
  is corrupted and resumes are faithful; the writer and the reader agree. It is
  a format wart, not data loss, and it makes byte-level checkpoint diffing
  meaningless for a reason nobody had named.
- **`param_id` is process-dependent.** The 54 `param_id` u64s in the metadata
  differ between every pair of runs (432 differing header bytes, 8-byte
  aligned, one per tensor). So a checkpoint is never byte-identical between
  processes *even if every weight matched*, which is a second, independent
  reason the file cannot be the instrument.
