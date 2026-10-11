# AWAY — 2026-10-10 → 2026-10-11 (owner asleep, work until the user returns)

MODE: AWAY

## The standing task

Continue the 0.22 port to "done", and do the optimization/research work the
owner listed. No push, no force-push, no destructive git op. One heavy GPU
process at a time (§1.5). Every claim names its measurement; every refutation
is written down. A/B-or-death for anything that changes training.

## The queue, in order

1. ~~**C1: isolate the 1e18 in the engram table.**~~ -> see `docs/reviews/2026-10-11-away-c1-init-vs-update.md`
   - Two processes, `--steps 1 --lr 0 --ckpt-every 1`, diff the two step-0
     checkpoints (both post-init, no update): does the corruption exist at INIT
     or only after one optimizer step? The f16/bf16 perf work below assumes this
     is answered.
2. **Re-land the `cubecl-ir` f16 patch for 0.22 and re-measure.** The port cut
   every `[patch.crates-io]` (`ae80b53`) on the premise "stock 0.11.0 covers
   f16 sizing"; f16 now measures 8.5 TFLOP/s against pre.4's 39.75, while bf16
   on the same stack measures 44.06. Re-vendor `cubecl-ir`, apply the
   `SizedType` fix, add the patch entry, measure one dtype per process.
   If f16 returns to ~40 the regression was ours and the precision plan is back
   on; if it does not, it is a cubecl 0.11 question for the upstream issue.
3. **arxiv: the re-entry playbook.** Use the arxiv MCP (watch topics / fetch)
   for (a) the sparse-attention indexer re-entry (QSA -> MSA, the deleted MSA
   crate), (b) hash/n-gram memory at scale (Qwen3.8-style, our Engram arm),
   (c) adaptive depth / recursion. Deliverable: one dated research doc under
   `docs/research/` with candidates, receipts (arxiv ids) and what each would
   cost us, plus the arxiv watches that keep it current.
4. **A measured optimization in our own crates.** Candidate: the TSCT
   polar retraction is a FIXED 52-64 ms/step independent of batch (22% of a
   step at batch 8, 7.8% at batch 32) and `--retract-every 1000` removes it, so
   there is a real budget to attack without touching numerics. Measure, change
   in `vendor/dormouse-fused/crates/burn-spectral`, keep the orthogonality
   guarantee, gate it.
5. **The safety crutch** (small, ours): the corrupted runs print finite
   garbage (ce 1e8..1e16) so the NaN firewall does not catch them. An early
   loss ceiling that names the mechanism is ADR-0011-shaped and ~10 lines, but
   it stops a run rather than continuing it - measure whether it is worth
   shipping as a guard instead of a workaround.
6. Close with `STATUS: DONE` appended below and a full report.

## Ground rules carried over

- pre.4 remains the trusted stack for any number that has to be believed;
  0.22 is not trainable until A1 (the autotuner's per-process candidate choice)
  is answered by upstream.
- Work one thing at a time; commit per task; no `git add .`.
- Disk was at 7 GB free earlier today and was cleaned to 30 GB - keep it there
  (`target/debug/incremental`, `vendor/dormouse-fused/target/debug`).

---

## STATUS: DONE — 2026-10-11 (night report)

### 1. The root cause of the corruption — ANSWERED

It is **a read of memory nobody wrote**, not the autotuner's candidate choice.
Two independent instruments:

- **Per-parameter, model-side**: two identical processes at `--lr 0` (no update
  in either) leave `loop_block.engram.memory.embedding.weight` different — 5767
  of 32768 rows, worst entry 2.144e14. So the difference is in the model's own
  tensors at INITIALIZATION.
- **Both step-0 checkpoints carry garbage that does not overlap**: 3308 and 2802
  rows with |v| > 100 (a 3.1M N(0,1) draw tops out near 5.5), largest 2.081e38 /
  2.345e38, only 193 rows in common. A bad WRITE makes one file clean and the
  other dirty; a bad READ makes both dirty in different places. `init_shift`
  closes the last alternative: row shifts ±1, ±2 match 21-22 of 200000 rows
  against 160818 at shift 0.

Refuted on the way, each now a pinned test: the autotuner's coin flip (6 clean
runs observed once with `bench{adaptive=false, max_samples=200}`, then 6 of 6
corrupt when repeated the same night — a coincidence that looked like a
breakthrough), the gather backward (unused rows zeroed exactly), the seeded RNG
(sound bare and tracked), bf16 on Flex (bf16 rounding), `--engram-ram` (also
explodes), a row move, optimizer state.

**The config-only workaround, measured**: `persistent_memory = "disabled"` +
`allocator = "sync"` cut the corruption from 9/9 at step 1 to 0/3. Not a fix: a
fresh page being zeroed is a property of the driver, not a contract.

### 2. The f16 regression was OURS, and it is fixed

`ae80b53` cut every `[patch.crates-io]` on the premise "stock 0.11.0 covers f16
sizing". False: the f16 tensor-core candidate panics with `Expected type
builtin.fp16 to implement dyn SizedType` and the autotuner falls back. The
`cubecl-ir` patch is re-landed in `vendor/cubecl-ir` and **f16 went from 8.53 to
44.04 TFLOP/s (5.2x, at the cuBLAS ceiling) while fp32 and bf16 did not move**
(one dtype per process, release):

| shape | fp32 | f16 | bf16 |
|---|---|---|---|
| 5120x768x2048 | 0.94 ms / 17.08 | **0.43 ms / 37.41** | **0.43 ms / 37.47** |
| 5120x2048x8192 | 8.59 ms / 20.00 | **3.90 ms / 44.04** | **3.89 ms / 44.17** |

ADR-0016 bug 2 is dead: bf16 parity tests pass and bf16 rides the same ceiling.

### 3. arxiv: the depth arms need φ, not BPB alone

`docs/research/2026-10-11-recurrence-equivalence.md` — arXiv:2604.21106v3 fits
an iso-depth law over r ∈ {1,2,4,8} and measures **φ = 0.46**: one recurrence
buys 0.46 of a unique parameter (at r=4 a 410M looped model matches a 580M
non-looped model's loss at a 1B non-looped model's compute cost). Truncated
backprop lowers φ to 0.38 while the loss still falls; hyperconnections raise it
to 0.65. AB-PROTOCOL's arm 4b now names BPB **and** the
recurrence-equivalence estimate as the verdict metric. We run NoCheckpointing, so
we are on the right side of the truncation case study.
`looped-dit-2609.40305` is the independent cross-check: naive looping fails
without deep supervision — our `L_Rec` supervises every iteration and the
residual write is gated.

### 4. Cost instrumentation instead of a blind optimization

`retract_seam.rs` now pins what each TSCT retraction path costs on the device:
**scalar 75.65 ms, batched 19.69 ms (3.8x)** — and the slope is flat from 2 to 6
NS iterations, so the FIXED part (stacking, power iteration, the Param round
trip) is the cost, not the NS loop. POWER_ITERS 5→2 was measured and **not
shipped**: 6.31e-5 vs 6.46e-5 orthogonality, indistinguishable from the TSCT
init's own cross-run residue.

### 5. The safety gate

`CE_CEILING = 100` in the training loop (tests/sanity_ceiling.rs): the
corruption's finite garbage is not the NaN firewall's class, and a corrupted run
must stop rather than look like a slow arm. Read only at the log cadence, so the
hot path stays sync-free. Not exercised end to end — see the caveat below.

### 6. What is NOT closed

- **The corruption stopped reproducing mid-session and that is unexplained.**
  12/12 corrupted at step 1 before (~04:00), 0/16 clean after (~08:00-09:00)
  across three batches, both pool configs, patched and unpatched binaries. The
  rebuild is the only difference I can name; a machine-state confound (thermal,
  clock, memory left by earlier processes) cannot be excluded. **Re-measure
  before trusting either rate.**
- **A new finding of the same class**: `dormouse-train`'s headwise-Muon gates
  (`headwise_batched_ns_matches_per_head`, `the_eval_counter_covers_both_muon_implementations`,
  `headwise_muon_matches_per_slice_muon`) fail by 2.27e-3 ONLY in a full run and
  pass alone — order-dependent, i.e. the uninit-read signature inside the
  optimizer. Needs the padded headwise path in `burn-muon-plus` read next.
- C2 (our amplification), C1's line of entry, and the ADR-0019 upstream issues
  (token still read-only here).

### Commits this session (8, all local, nothing pushed)

`2932966` root cause → `e8cc65b` φ/AB protocol → `e970255` AWAY marker →
`62ab1e6` the f16 patch → `b630c7c` retraction cost instrument → `7756bdc` the
CE guard → `a2ed114` memory. Gates: core lib 94/94 in both cells, train lib 56
(CPU) with the cuda cell's 3 order-dependent Muon failures recorded above,
train integration 7/7, backend-parity 8/8 + both bf16 parity tests now un-ignored
in value, doc-refs green. Disk was cleaned to 15 GB free.
