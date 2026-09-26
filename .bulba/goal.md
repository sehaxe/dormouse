# Goal: autonomous development — speed war while the official baseline trains

MODE: AWAY

The product: dormouse (byte-level LM trainer, burn pre.4, now on main with the
official fp32 baseline running — 100k steps, corpus v2, log
~/logs/official_baseline.log).

Order of work (decided, per the standing plan and the owner's push for pace):

1. Fusion A/B: fusion-flip branch compile check is in flight. If green →
   build the fusion binary, smoke it (20 steps), then at the next baseline
   checkpoint pause the run, `scripts/bench.sh canary` a fusion row into
   history.tsv, compare vs row #1 (4081 ms, bdddbaf). Win → merge fusion-flip
   to main, resume the baseline on the fusion binary (resume via same
   ckpt-name/ckpt-dir). Lose → park the branch, resume on main's binary.
2. bf16 A/B: on the fusion binary, `--bf16` canary row (the format the owner
   wants; old kernels measured it slower, fusion is supposed to flip that).
3. Fused Rung 1 (ADR-0009): implement direct adjoints seeded from saved
   per-iteration buffers on a branch; the 1-2 device syncs per step removed;
   the bar is >=1 s/step at flagship. A/B in a GPU window when ready.
4. Knife A/Bs (aux vs CE, GR, rank) in remaining windows.

Rules: doctrine as always — A/B or death, loud failures, bench rows as
verdicts, tests green per commit, push milestones. The baseline is never left
stopped: pause only at a fresh checkpoint, resume immediately after the bench.

Stop condition: gains negligible (bench rows flat across two A/Bs) or the
queue is empty.
