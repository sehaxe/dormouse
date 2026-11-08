---
name: Performance
about: A performance change, with the baseline artifact before the claim
title: ''
labels: ''
assignees: ''
---

**The artifact first** (required)

- baseline commit SHA:
- baseline command, every flag:
- baseline log path:
- candidate commit, if it exists yet:
- machine: CPU model and RAM; GPU model, driver, power limit; whether the box is
  shared. A perf number without its machine is a story.
- what was measured, in what units (wall ms/step, tokens/s, bytes/s, peak RSS,
  launches/step), and the statistic — median of N reps, with the spread. One run
  decides nothing.

**What the number is NOT**

<!-- The part of the pipeline the measurement excludes: kernel only, excludes
     the dataloader, excludes compile/warmup, host-synced. Naming it is what
     separates a measurement from a pitch. -->

**The change**

<!-- One paragraph, and the `file:line` it lives in. -->

**The decision rule, written before the run**

<!-- For example: "faster by more than 2x the spread at the same batch and
     dtype, with no BPB change beyond the seed band". If the rule and the
     result were written in the same edit, say so — it is a description, not a
     decision. AGENTS.md §1.2. -->

**What could regress**

<!-- Correctness, dtype, a CUDA path, memory — and the gate that would catch
     it. A speedup that skips a launch counter or a parity test is not a
     speedup yet. -->

**Reproduce**

<!-- The exact command a reviewer runs to get the same table, and the machine
     it was taken on. -->
