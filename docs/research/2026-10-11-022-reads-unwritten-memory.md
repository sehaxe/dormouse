# 0.22: the corruption is a READ of memory nobody wrote — 2026-10-10/11

**Status:** root cause found and measured. Two independent instruments agree, and
the fingerprint survives every alternative that was tested.

## The claim

On burn 0.22 + cubecl 0.11, a kernel in the model's construction reads memory it
never wrote. The value it reads is whatever the memory pool has in that page:
with the persistent pool on, recycled pages hold stale values up to 2.35e38, and
the training run dies at step 1. With the pool off, a fresh `cudaMalloc` page is
zeroed and the same read is harmless — which is why `persistent_memory =
"disabled"` cuts the corruption rate without touching a line of anyone's code.

## The evidence, strongest first

1. **Per-parameter, model-side.** Two identical processes (`--preset small
   --steps 1 --lr 0 --seed 1`, the no-op update) leave
   `loop_block.engram.memory.embedding.weight` different: 5767 of 32768 rows,
   the worst entry **2.144e14 absolute** (`tests/ab_bisect` in the bisect
   worktree). `--lr 0` means no parameter was updated by either process, so the
   difference is present in the model's own tensors at initialization. Every KDA
   parameter is bit-identical between the two.
2. **Both files carry garbage, and the garbage is different.** A word-level scan
   of the two step-0 checkpoints (`tests/init_garbage`) finds 3308 and 2802 rows
   with |v| > 100 — a N(0,1) draw of 3.1M values tops out near 5.5, so those
   words are garbage by arithmetic — with a largest value of 2.081e38 and
   2.345e38, and only **193 rows in common**. A bad WRITE would make one file
   clean and the other dirty; a bad READ makes both dirty in different places,
   which is what is measured. `tests/init_shift` rules the remaining
   explanation out: shifts of ±1, ±2 rows match 21-22 of 200000 rows against
   160818 at shift 0, so it is not an indexing move either.
3. **compute-sanitizer says the same thing independently.** `initcheck` reports
   uninitialized `__global__` reads in this stack (the narrow-output matmul
   family, ~100 events at `out_features = 2`, 0 at 64). That report predates
   this finding and was filed as hygiene because the value looked masked; with
   the per-parameter evidence above it is the mechanism, not a footnote.
4. **The pool decides the magnitude, by measurement** (2026-10-11, `small`
   fp32 b10 s512 lr 1e-4, `--steps 12 --log-every 1`, one run per cell):

   | `persistent_memory` | allocator | corrupt at step 1 |
   |---|---|---|
   | size-match | async | **9 of 9** |
   | disabled | sync | 0 of 3 (one died at step 2) |

   Earlier in the port: size-match 10/10, disabled 1/3, enforced 0/1 (§3.3).

## What this refutes, on the record

- **The autotuner's coin flip as the cause.** The decisions sink does show
  different winners between runs (50/125 keys), and 6 clean runs were once
  observed against 6 dirty ones with `[autotune.bench] adaptive = false,
  max_samples = 200` — but repeating that config the same night gave 6 of 6
  corrupted at step 1, so it was a coincidence, not a fix. The bench knobs move
  the bench, not the read.
- **A row move** (`init_shift`).
- **A wrong bf16/gather/RNG kernel**: the raw ops are honest. `gather`'s backward
  zeroes every untouched row exactly; the seeded device RNG draws N(0,1) on the
  bare and the autodiff device (max 4.97..5.34, zero non-finite); bf16 matmul on
  Flex is bf16 rounding, bare and tracked.
- **The in-VRAM Engram update path**: `--engram-ram` explodes too (3/3).
- **Optimizer state**: the difference is in a model parameter under `--lr 0`.

## The workaround without touching anyone's code

`cubecl.toml`:

```toml
[memory]
persistent_memory = "disabled"
[memory.cuda]
allocator = "sync"
```

Measured: 2 of 3 runs healthy over 12 steps where the shipped config is 0 of 12.
**It is not a fix.** A fresh page being zeroed is a property of the driver, not
a contract, and one of the three runs still died. Per ADR-0011 the honest way to
ship it is COUNTED: the trainer's startup should print which memory config it
resolved, and a run should refuse to continue if its step-0 ce is not sane
(finite garbage passes the NaN firewall — 1e38 is a finite number).

## What upstream needs to answer

1. Which kernel reads unwritten `__global__` memory — `initcheck` already names
   the file family (`matmul_entry` / a cast kernel, narrow-output matmuls), and
   the same read reaches the model's initialization path.
2. Why the persistent pool serves that page with stale contents instead of
   refusing: a read of never-written memory is the bug; the pool is what turns it
   into 2e38 instead of 0.

## For us, after that

- C2 (our amplification) is now the second half of the story: with the pool
  disabled and the read zeroed, the model survives — so the amplification matters
  only when something feeds it garbage. It is still worth isolating, but it is
  not the blocker.
- The 0.22 numbers in `benches/history.tsv` and every A/B verdict stay invalid
  until the read is fixed or the guard exists.
