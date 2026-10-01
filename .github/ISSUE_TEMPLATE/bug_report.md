---
name: Bug report
about: Something in the repo is wrong, and here is the evidence
title: ''
labels: ''
assignees: ''
---

**What is wrong**

<!-- The symptom, in one or two sentences. -->

**The smallest command that shows it**

    <command, with every flag>

<!-- A command that reproduces it beats a description of it. If it needs a
     GPU, say so - the CUDA gates run on a card and the CPU gates do not
     (AGENTS.md §2.5), so "it fails" on the wrong backend is a different
     report. -->

**Which run**

<!-- AGENTS.md §1.4: a number is a number only with its config, its date and
     its commit. If the report quotes one, all three belong here. -->

- commit:
- config (`checkpoints/<name>.config.toml`, or the flags):
- date:

**What I expected, and what happened instead**

<!-- If the loss curve, a counter on the eval line, or a held-out number is
     involved, paste the LINE - including the `fused kda=`, `engram=`,
     `moe=`, `tsct=` and `fb=` fields. Those fields are how a reader learns
     which arm actually ran; a report without them cannot be diagnosed,
     because the run may have measured a different program than the one you
     meant. -->

**Is this the whole mechanism or a symptom of it?**

<!-- Optional, but it is the question that decides the fix: a patch in one
     caller of a shared function leaves every sibling caller broken
     (AGENTS.md §1). If you already know the root cause, name it. -->