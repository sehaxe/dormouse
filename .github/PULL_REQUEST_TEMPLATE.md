## What this changes

<!-- One or two sentences. If it fixes a named symptom, name the root cause you
     fixed instead - a patch to one caller of a shared function is a second
     bug, not a fix (AGENTS.md §1). -->

## The gate that can fail

<!-- Required for anything non-trivial. Name the test/command and what it
     asserts. "It compiles" is not a gate; a gate that cannot fail is the
     defect class AGENTS.md §3.3 records for dormouse-rmsnorm's fused path.
     Non-trivial logic (a branch, a loop, a parser, a money/security path)
     leaves ONE runnable check behind. -->

    <command>

## Evidence, if this changes a number

<!-- AGENTS.md §1.4: a measurement is a measurement only with its config, its
     date and its commit. A PR that changes a number without those three is
     asking the next reader to trust it. -->

- config / commit:
- date:
- what was measured, and what it is NOT (the part of the pipeline the test
  actually exercised):

## Rules I checked

- [ ] Every degradation is LOUD, COUNTED, or SILENT-and-then-fixed - never a
      silent fallback (AGENTS.md §1.1).
- [ ] A new mechanism's fate is decided against its own removal
      (`docs/protocols/AB-PROTOCOL.md`), or it is not claimed as working.
- [ ] Any claim of "verified" / "bit-for-bit" names the external file the
      reference came from, or says plainly that none exists (§1.4).
- [ ] No host-device sync in the hot path; every host-visible quantity comes
      from a device counter read at a declared cadence (§1.3).
- [ ] Nothing unrelated was fixed in this diff - follow-ups are reported
      (`<file>:<line>`), not smuggled.
