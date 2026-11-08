---
name: Feature
about: Propose a capability, and say what would have to be true for it to be deleted
title: ''
labels: ''
assignees: ''
---

**The artifact that motivates it** (required)

<!-- A failing test, a log path, a measurement, or an issue number. "It would be
     nice" is not an artifact: a capability no run needs is a cost with no
     buyer. -->

**What it unblocks**

<!-- The run, the measurement, or the user task. If there is no run, say so
     plainly — it is then a feature for the software, not for the model, and it
     is judged as one. -->

**The smallest version**

<!-- What one flag, default, or test shows it works? The first PR should be
     that and nothing more. -->

**What would have to be true for it to be deleted**

<!-- Required for a model mechanism. The comparison that kills it: against its
     own removal, held-out BPB, fixed step budget, 3 seeds per arm. A tie
     deletes it (AGENTS.md §1.2, docs/protocols/AB-PROTOCOL.md). If the
     proposal cannot state the comparison, it is a design note, not a proposal. -->

**The gate that can fail**

<!-- The test that goes red if the feature is wrong, and the file it lives in.
     "It compiles" is not a gate, and a test that cannot fail is decoration. -->

**What it replaces**

<!-- Deletion is the preferred shape of a feature. If this makes a flag, a
     script or a module unnecessary, name it and delete it in the same PR. -->

**If it adapts someone else's code or formulas**

<!-- Name the source: file, URL, paper, license. "Adapted from" is a
     provenance claim and it has to be checkable (AGENTS.md §1.4). -->
