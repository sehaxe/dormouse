# 0.22: the corruption is the autotuner's coin flip, and it is measured

**2026-10-09, review of the 0.22 port (`4b2ff21`..`6e53a4f`), on this box.** The
question the port left open: `fp32 b10 s512` on `preset small` intermittently
destroys its own run — `ce` 5.57 → 1e8..1e16 → NaN, roughly 2 runs in 3. This
is the answer, with every claim's measurement named. Nothing here is a fix; it
is what the fix has to be built on.

## 1. What is NOT the cause (each measured, each with its command)

| suspect | verdict | evidence |
|---|---|---|
| the weights | excluded | at `--lr 0` (a no-op update) **every parameter is bit-identical** step 0 → step 1 while the step-1 forward reads `ce = 1.76e9`. `wv_bisect` in `wt/bisect022` prints `0.000e0` for all 54 params |
| the held-out data | excluded | `--eval-every 1` prints `held-out eval is NaN` on those same weights: `train failed: step 1: held-out eval is NaN (non-finite) - the model or the allocator is broken, not the data` |
| the KDA chunk math | excluded | the production-shape gate is **green** now (`a4f42ed`): batched vs loop arm, out/state drift 3.3e-4, all 7 gradients ≤ 3.3e-4, at B=10 H=12 T=512 k=v=64 chunk 16 |
| the TSCT retraction writing garbage into the factors | excluded | `--retract-every 1000` still explodes (1/1) |
| the optimizer's parameter write | excluded | `--lr 0` explodes; `--opt adamw` is clean (1/1) |
| a race | excluded | `compute-sanitizer --tool racecheck` on the KDA module forward: **0 hazards** |
| an out-of-bounds write | excluded | `--tool memcheck`: **0 errors** |
| in-process nondeterminism | excluded | two forwards of one graph agree bitwise; a forward after a backward equals the one before it (`matmul_uninit_probe::module_forward_is_deterministic`) |

## 2. What IS per-process: the autotuner's candidate choice

`cubecl.toml` with a decisions sink, one run per process:

```toml
[autotune.decisions]
file = "/tmp/opencode/atune/<tag>.jsonl"
```

Six runs of one program (`./target/release/train --preset small --data
<real> --steps 8 --batch 10 --seq-len 512 --quant fp32 --lr 1e-4 --seed 1
--log-every 1 --opt mix`), 125 tuning decisions each. **50 of 125 keys get a
different winning candidate between two runs.** Two keys flip together and
separate the outcomes perfectly:

| key (shape) | 4 exploding runs | 2 clean runs |
|---|---|---|
| `31944fb2` m=512 n=1024 k=128, both contiguous | `matmul_simple_cyclic_mma_multi_rows` (idx 15) ×4 | `matmul_ordered_double_mma_partition_k2_row_count8_rows_per_plane2` (idx 19) ×2 |
| `c5c11115` m=64 n=2048 k=64, rhs transposed | `...ordered_double_mma...row_count8` (idx 19) ×4 | `...simple_cyclic_mma_multi_rows` (idx 15), `matmul_double_cyclic_mma` (idx 21) |

The medians of the two winners on `31944fb2` are 23.5 µs and 24.3 µs — **1.6 %
apart, so a coin flip decided by benchmark noise selects between two kernels
that disagree numerically.** Candidates carry a stable `idx`; the `results`
array arrives in completion order, so a decision is stable within a process
and different across them. Split of 4/2 by chance: 1/15.

Everything that "fixes" the explosion in one run (`--opt adamw`,
`--no-kda`, `--no-engram`, `--set n_experts=1`) removes a shape or changes
the benchmark timings, i.e. it moves the coin. The pool config moves it too,
which is why it measured like a cause:

| `persistent_memory` | explosions |
|---|---|
| `size-match` (shipped) | 10/10 |
| `enabled` | 1/1 |
| `disabled` | 1/3 … 3/3 (moves the timings, not the defect) |
| `enforced` | 0/1 |

`disabled` costs **nothing** (198–207 ms/step, identical `ce`, `--no-kda`
control under both modes) — and it is NOT a fix, so it is not shipped.

## 3. The one real (but harmless) finding along the way

`compute-sanitizer --tool initcheck` reports ~100 reads of never-written
global memory in a narrow-output matmul (`out_features = 2`; 0 at 64). The
value does **not** reach the result: `narrow_matmul_value_is_masked` compares
against a materialised rhs and gets `0.000e0` at out_features 2/3/4. So it is
an upstream hygiene defect worth its own issue, and it is **not** this
corruption — the refutation is in the same file as the repro.

## 4. The ladder, so nobody re-derives it

`vendor/dormouse-fused/burn-fused/tests/matmul_uninit_probe.rs`, one test per
level, each runnable on its own:

```
cargo test -p burn-fused --release --features cuda,autodiff \
  --test matmul_uninit_probe -- --nocapture
# per level, under the sanitizer:
compute-sanitizer --tool initcheck --launch-timeout 120 \
  cargo test -p burn-fused --release --features cuda,autodiff \
    --test matmul_uninit_probe <test_name> -- --nocapture
```

## 5. What to do next (in order)

1. **Upstream issue on `tracelabs/cubek-matmul`**: a numerically wrong (or
   merely different-order) candidate wins a coin-flip benchmark; include the
   probe file, the two keys above, and the 4/2 split. The same issue should
   ask why `target/environment` is never created — the persistent autotune
   cache exists in the binary (2 692 persistence symbols) and is the only
   mechanism that would make a choice stick; `rt1/rt2/rt3` from one cwd made
   125 fresh decisions each and wrote zero files.
2. **Until upstream answers, 0.22.0 is not trainable for an A/B** — the
   corrupted runs are silent, and a corrupted run looks like a slow arm. The
   `pre.4` checkout remains the stack that produces trustworthy numbers.
3. If a local mitigation is wanted before then, the honest one is a **gate,
   not a config**: run the trainer's step twice in two processes and compare
   the step-0 `ce` bitwise; a mismatch is this defect, and it is the cheapest
   loud detector available today (one flag, no model code).

## 6. What landed in this lane

| commit | what |
|---|---|
| `9c209f4` | burn-kda builds on 0.22: `TensorData.bytes` is a method |
| `4eb2948` | the technology super-research pass (ranked swaps, verified refs) |
| `be86247` | the library's test surface builds on 0.22 — 53 files, so its gates can run again |
| `a4f42ed` | the production-shape gate can finally pass (its fixture fed S0 as `[B,H,T,K]`) |
| `6e53a4f` | the probe ladder of §4, with the negative results recorded as results |

Gates on the lane's tip: `dormouse-core --lib` 93/93, `dormouse-train --lib`
56/56, doc-refs 633/0, and the vendor workspace's `--tests` check green for
all 14 crates the sweep touched.
