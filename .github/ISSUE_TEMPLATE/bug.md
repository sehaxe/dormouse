---
name: Bug
about: Something is wrong, and here is the artifact that shows it
title: ''
labels: ''
assignees: ''
---

**The artifact** (required — without it the report cannot be reproduced)

- commit SHA (`git rev-parse HEAD`):
- the smallest command that shows it, with every flag:
- log path, or the line pasted below:
- config: `checkpoints/<name>.config.toml`, or the `--set` flags:
- date:
- backend: CPU / CUDA (which card, which driver) — the CUDA and CPU gates test
  different things, so "it fails" on the other backend is a different report.

**What is wrong**

<!-- The symptom, in one or two sentences. -->

**What you expected instead**

**Does it reproduce from a clean checkout?**

<!-- If not, name the state it needed (checkpoint, config snapshot, corpus). A
     report that only fails in one working directory is still a bug, but it is
     a state bug. -->

**The log line, if a run is involved**

<!-- Paste the whole line, including the `fused`, `engram=`, `moe=`, `tsct=`
     and eval fields. They are how a reader learns which arm actually ran; a
     report without them can describe a different program than the one that
     failed. AGENTS.md §1.1. -->

**Root cause, if known**

<!-- `file:line`. A patch in one caller of a shared function leaves every
     sibling caller broken — if you know the shared place, name it. If not, say
     what you ruled out and how. -->
