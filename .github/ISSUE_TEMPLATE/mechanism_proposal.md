---
name: Mechanism proposal
about: Propose a mechanism, and say what would have to be true for it to be deleted
title: ''
labels: ''
assignees: ''
---

**The mechanism**

<!-- What it is, in a paragraph, and which file would own it. -->

**What would have to be true for it to be deleted**

<!-- Required, and it is the whole question. `docs/protocols/AB-PROTOCOL.md`
     §1.2: a mechanism beats ITS OWN REMOVAL on held-out BPB at a fixed step
     budget, 3 seeds per arm, or it is deleted. A tie deletes it. PonderNet
     was deleted this way. If the proposal cannot state the comparison that
     would kill it, it is not a proposal yet. -->

**Its removal**

<!-- The control arm: what does the run look like with this off? A mechanism
     with no cheap removal is a mechanism nobody can price. -->

**The gate that can fail**

<!-- The test that goes red if the mechanism is wrong, before any A/B is run.
     Name it and the file it lives in. "It compiles" is not a gate, and a gate
     that cannot fail is the defect class AGENTS.md §3.3 records for the
     burn-rmsnorm fused path - it stayed green for a year while the arm ran
     zero times. -->

**Where it stands today**

<!-- Anything already wired, and whether it has EVER run. "Implemented" and
     "executed" are different claims in this repo and the difference has
     retracted results before (§3.2). Say which one you mean. -->

**If this adapts someone else's code or formulas**

<!-- Name the source: file, URL, paper, license. §1.4 - "adapted from" is a
     claim about provenance and it has to be checkable. -->