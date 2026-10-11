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
