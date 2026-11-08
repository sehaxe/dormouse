# 2026-11-08 — the data pipeline: 0.08 % of a step, ~72 MB/s, nothing to do here

**Question.** The live ladder consumes `b × (s+1) = 8 200 B` per step at b=8,
s=1024 and runs at ~56 kB/s (`docs/research/2026-11-05-small-model-mfu.md` §6.1,
~147 ms/step). The `frombuffer` fix made the loader 9–40× faster (`117555f`). Is
it still a bottleneck, and what is its true ceiling?

**Verdict: nothing to do here.** One `MixtureStream.next_batch()` over the six
real sources costs **0.114 ms of CPU** (median 0.109 ms) — **0.077 % of a
147 ms step**. The loader would have to be **26× slower** to reach 2 % of the
step, and **1 290×** slower to become the step's rate limiter. Its ceiling on
one thread is **8 781 windows/s = 72.0 MB/s** of payload; the entire 120 MB
ladder arm is ~1.7 s of loader work against ~36 min of training. No code was
changed.

The 56 kB/s is the **step rate**, not the loader's rate:
`8 200 B / 0.147 s = 55.8 kB/s`. It says how fast the model eats bytes; it
cannot say how long fetching them takes, which is what this file measures.

---

## 1. What was measured, and how

Instrument: `port/scripts/bench_loader.py` (left uncommitted in the tree), run
from `port/`:

```
uv run scripts/bench_loader.py --steps 4000 --reps 5 --cold-reps 3 \
    --json /tmp/opencode/bench_loader3.json
```

Provenance of the run quoted below: **2026-11-08 13:18, HEAD `eb285e3`, load
average 101.7 / 97.9 / 71.0 on 20 CPUs** (a CSA run and other agents were
compiling), plan `port/configs/mix_refetch.toml`. The corpus was re-carved at
13:08 by another agent (`port/scripts/carve_eval_tails.py` removed the five
remaining sources' 8 MiB eval tails, mid-session); every number below is from
the post-carve files, and the sizes in the §4 table are the script's own reads.

Three design choices make the number defensible:

* **CPU time and wall time are both recorded per call.** Under this load a
  110 µs call is routinely preempted: the headlined warm wall mean is 0.188 ms
  while its CPU mean is 0.114 ms. CPU is the loader's cost; wall is what a step
  on a busy core would see. Neither is substituted for the other, and the
  calmest of 5 blocks is headlined (CPU means across the 5: 0.117, 0.114, 0.131,
  0.124, 0.129 ms — a ±15 % spread, which is the load's contribution, not the
  loader's).
* **Warm and cold are the SAME byte range of the SAME stream instances.** The
  cold pass is measured first after evicting every source
  (`posix_fadvise(DONTNEED)` on the fd + `madvise(DONTNEED)` on the mapping),
  then `rewind()` and the same range is timed warm. Cold is repeated 3× and the
  calmest pass quoted, because the drop is deterministic and the spread is the
  box's load.
* **The bytes are checked before they are timed.** `_selfcheck` asserts that
  `_windows` serves exactly the bytes it is handed (`x == raw[..., :-1]`,
  `y == raw[..., 1:]`, row boundaries do not bleed) and that two same-seed
  `MixtureStream`s serve identical tensors; the first 8 draws
  (`fwedu code cosmo code fwedu fwedu fwedu code`) are printed. The `frombuffer`
  fix already has its own bit-identical gate in `port/tests/test_data.py`.

`mmap` page faults do not appear in `/proc/self/io`'s `read_bytes` on this
kernel (measured 0 across a dropped pass), so the evidence that the cache was
evicted is the first-window wall time, which is printed: 3–25 ms against
0.10 ms warm. **No CUDA was used**, and the loop's `x.to(device)` copy is outside
every number here (it is 131 kB/step of int64 and does not live in `data.py`).

---

## 2. Warm: the real plan, six sources, 20 000 windows

| path | CPU mean | median (CPU) | wall mean (load 102) | ceiling |
|---|---:|---:|---:|---:|
| `MixtureStream.next_batch()` | **0.114 ms** | 0.109 ms | 0.188 ms | **8 781 win/s · 72.0 MB/s** |
| `source_for_step` (draw only) | 0.005–0.009 ms | 0.006 ms | — | — |

Per source (each source's own `ByteStream.next_batch`, 1 000 windows; the table
below is CPU ms, which is what the loader costs):

| source | next_batch | `_read` (mmap + ring) | `_windows` (`[b,t]` build) |
|---|---:|---:|---:|
| code | 0.106 | 0.009 | 0.072 |
| cosmo | 0.083 | 0.010 | 0.072 |
| finemath | 0.085 | 0.011 | 0.071 |
| fwedu | 0.083 | 0.010 | 0.073 |
| math | 0.105 | 0.011 | 0.101 |
| ru | 0.112 | 0.010 | 0.101 |

**Where the 0.114 ms goes.** `_windows` is ~70 % of it: `bytearray(raw)` +
`torch.frombuffer` + `.long()` + two `.contiguous()` copies
(`data.py:89-90`). `_read` is 9–13 %: the mmap slice, the ring append and
`del buf[:n]`. The `del` is **not** the O(ring) memmove it reads as — CPython
3.12's bytearray keeps a start offset, so 8 200 B consumption costs microseconds
(the prior audit measured 1.7 µs; this bench's `_read` is 9–11 µs including the
8 MB refill slice). The remaining ~7 % is the pure draw. Nothing here is I/O:
the first warm access after `rewind()` is in the page cache, and the corpus
(2.8 GB) fits in this box's 34 GB available.

---

## 3. Cold-ish: the drop costs ~0.15 ms of CPU on average, once per source

Three passes, each: evict all six sources → `rewind()` → first 8 windows → next
400 windows. This run was at load 102, so the wall column is the machine's
queue, not the disk; the CPU column is the loader's work.

| pass | first 8 windows (wall, per window; n=8 so p99 is the max) | next 400 (wall mean) | next 400 (CPU mean) |
|---|---:|---:|---:|
| 0 | 25.2 ms (max 71.2) | 1.033 ms | 0.164 ms |
| 1 | 11.3 ms (max 32.5) | 0.912 ms | 0.145 ms |
| 2 | **10.4 ms** (max 28.5) | **0.894 ms** | 0.150 ms |

The cold CPU cost is stable at **0.145–0.22 ms** across every pass and run
today; the wall is not. The same 400-window block read **0.324 ms** at load 94
(CPU 0.154 ms) and **0.166 ms** in a quieter pass before the carve, so the wall
band is **0.17–1.03 ms** and even the worst measured pass is **0.70 % of a
step**. The first window is a one-off per source: 3.5–25 ms wall (2–17 % of one
step),
maximum single window 71 ms. It is not a per-step cost — the first fault reads
~128 kB of readahead, which covers the next ~16 draws of that source. At the
live rate the spike is amortized over ~14 600 steps: 8 × 3.5–25 ms = 28–200 ms
once, against a 36-minute arm.

Why so little: the pattern is six interleaved sequential streams. The kernel's
readahead (128 kB default) turns one fault per 128 kB into ~16 draws served from
cache, so the device is asked for 8 kB per draw amortized. The loader is not
I/O-bound even on a fully cold corpus at this rate.

---

## 4. `doc_masking=false` vs a doc-packed stream

**The plan cannot be doc-masked on this corpus, and the script says so.** No
source has a `.docs.json` sidecar (`find data_local -name '*.docs.json'` is
empty), and the separators the refetch wrote are not evenly present:

| source | size | `\n\n---\n\n` separators | documents derivable |
|---|---:|---:|---:|
| fwedu | 890 MiB | 191 894 | 191 895 |
| cosmo | 280 MiB | 107 776 | 107 777 |
| finemath | 896 MiB | 165 942 | 165 943 |
| code | 192 MiB | **0** | 1 (no boundary) |
| math | 192 MiB | **14** | 15 (whole 192 MB chunks) |
| ru | 192 MiB | **0** | 1 (no boundary) |

`MixtureStream(doc_masking=True)` refuses any source without a sidecar
(`mixture.py:551-566`), so the live mixture stays flat. The four sources where
a boundary list **can** be built in memory (fwedu, cosmo, finemath, math) were
timed with `DocPackedStream.next_batch_masked()`; nothing in `data_local/` was
written.

| source | packed CPU/window | flat CPU/window | ratio | scored positions |
|---|---:|---:|---:|---:|
| fwedu | 3.884 ms | 0.083 ms | 47x | 91.0 % |
| cosmo | 3.514 ms | 0.083 ms | 42x | 84.7 % |
| finemath | 3.495 ms | 0.085 ms | 41x | 94.0 % |
| math | 3.752 ms | 0.105 ms | 36x | 100.0 % |

**A doc-packed window is 2.4–2.6 % of a 147 ms step**, flat is 0.08 %.
The scored fraction says this is not padding: the cost is
`DocPackedStream._row` still building `torch.tensor(list(chunk))` per row
(`data.py:619-633`) — the same PyLong-per-byte path the `frombuffer` fix removed
from `_windows`; the fix's diff touched `_windows` only (`117555f`). At b=32 it
would be ~14 ms/window if the cost scales with rows.

This is the only data-path item in this report above 1 % of a step, and **it is
not the live loader**: it is an arm that this corpus cannot run at all. It is a
follow-up, not a land-now change — see §6.

---

## 5. The arithmetic, against the 147 ms step

```
loader (warm)            0.114 ms CPU          0.077 % of 147 ms
2 % threshold            2.940 ms              25.8x the measured loader
if the loader were free  146.886 ms/step       0.077 % faster
payload rate             8 200 B / 0.114 ms    72.0 MB/s
live rate                8 200 B / 0.147 s     55.8 kB/s     = 1 290x lower
one 120 MB arm           120e6 / 72.0e6        1.67 s of loader CPU
                         14 648 steps x 0.147 s = 2 153 s          = 0.077 %
cold first window        3.5-25 ms             2-17 % of ONE step, once per run
```

The 0.077 % is not a measurement artefact of the loaded box: the CPU mean is
0.114 ms against a 0.114–0.131 ms range across five blocks, and the other two
full runs today read 0.111 and 0.120 ms. **Even at 4× the measurement
(0.46 ms) the share is 0.31 %.** The loader only reaches 2 % of the step if the
step falls below ~5.7 ms; at 147 ms it has 1 290× headroom.

---

## 6. What was not changed, and the one follow-up the numbers name

**No code change.** The live loader has no win worth taking: the whole path is
already 7–10× faster than the pre-fix one, `_read` is microseconds, the draw is
5–9 µs, and 0.114 ms is 0.08 % of the step. Shaving the last 0.05 ms would be
0.03 % of wall time and would need a sample-exact gate to prove the mixture
order unchanged — a test with no measurable payoff on this config.

Named, not done:

1. **`DocPackedStream._row` still uses `torch.tensor(list(chunk))`**
   (`data.py:619`, `:621`, `:632-633`). If a doc-masked rerun is ever scheduled,
   do the same `torch.frombuffer(bytearray(chunk), dtype=torch.uint8).long()`
   and pin the bytes sample-exact. Measured cost of not doing it: 3.5 ms/window
   = 2.4–2.6 % of a step (≈14 ms at b=32 if it scales with rows). It is not
   done now because the arm cannot run: the recovered corpus has no sidecars,
   two sources have no separator at all, and the scripts that use
   `port/configs/mix_v7_docmask.toml` (`night_stage2.sh`, `night_stage3.sh`,
   `queue_arch.sh`, `docmask_ab.sh`) point at `data_local/mix4/`, which the merge
   incident deleted. A fix there is currently unverifiable against a real run.
2. **The ring's `bytearray` is not the cost.** Two columns in the table above
   show `_read` at ~10 µs, so there is no reason to replace the ring with a
   direct mmap offset.
3. **H2D is outside this file.** `x.to(device)`/`y.to(device)` in the train
   micro-batch (`port/src/dormouse/train/loop.py`) move 131 kB of int64 per
   step. That is part of the 147 ms step, not of the loader; no CPU-only
   instrument can measure it.

---

## 7. Reproduce

```
cd port
uv run scripts/bench_loader.py --steps 4000 --reps 5 --cold-reps 3
```

Outputs the warm mixture block (5 reps, CPU and wall), the per-source split, the
three cold passes, the separator census and the doc-packed arm, then the
share-of-step arithmetic at `--step-ms 147` (override it with the step the
current run actually measures). `--json <path>` writes every block for a later
comparison.
