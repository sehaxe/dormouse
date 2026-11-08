## What this changes

<!-- One or two sentences. If it fixes a named symptom, name the ROOT CAUSE you
     fixed: a patch in one caller of a shared function is a second bug, not a
     fix. Report something adjacent with `file:line` instead of fixing it here. -->

## The gate that can fail

<!-- Required for anything non-trivial. The test/command, what it asserts, and
     the red line from the un-fixed code. A test you never watched fail is
     decoration; the same test asserted after the fix is the proof. -->

    <command>

Red on the un-fixed code:

    <the failure line>

## Evidence, if this changes a number

- config / preset:
- commit (`git rev-parse HEAD`):
- date:
- log path, or the test that produced it:
- what the number is NOT (the part of the pipeline that was not exercised):

A number without these is deleted in review, not softened (AGENTS.md §1.4).

## Checklist — the repository's gates

- [ ] `bash scripts/gate_fast.sh` exits 0 on the final commit (`GATE PASSED`).
- [ ] `cd port && CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q` is
      green on the final commit.
- [ ] Doc references: every path I moved is rewritten in this commit, no stub is
      left at the old path, and `python3 tools/check_doc_refs.py` is green (it
      also runs inside `gate_fast`).
- [ ] `port/CHANGELOG.md` has an entry for a user-visible change — or this
      change is not user-visible and this body says so.
- [ ] Every number in this body names its artifact: config, commit, date, log
      path. An A/B claim also names its seeds and the decision rule.
- [ ] The test I added fails on the un-fixed code (red line above) and passes
      after.
- [ ] No test was deleted, skipped, or loosened. If a test's claim was wrong,
      this body names the claim that moved and why.
- [ ] Every new degradation is LOUD (error naming cause and escape) or COUNTED
      (a counter a log reader sees) — never a silent fallback (§1.1).
- [ ] No compatibility shim for unused code — it is deleted instead.
- [ ] Nothing unrelated was fixed in this diff; follow-ups are reported as
      `file:line` and left alone.

## Rules I checked

- [ ] No host-device sync crept into the hot path; every host-visible quantity
      comes from a counter read at a declared cadence (§1.3).
- [ ] A mechanism's fate is decided against its own removal
      (`docs/protocols/AB-PROTOCOL.md`), or this PR does not claim it works.
- [ ] "Verified" and "bit-for-bit" name the external file the reference came
      from, or say plainly that no external reference exists (§1.4).
