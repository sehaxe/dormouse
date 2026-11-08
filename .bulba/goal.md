# GOAL — AWAY mode

MODE: AWAY. The owner is asleep and gave full autonomy until they return.
Deliver at the end: STATUS: DONE + a full report (what changed, what was
measured, what is still red, what needs their decision).

## Mission

Make this repo the best software it can be, measured — not asserted. In order:

1. **Fix what is red.** The full CPU suite has 22 failures (2026-11-08 run).
   Classify each: real defect / stale test / GPU-only. Fix the real ones with a
   gate that would catch the return; fix the stale tests by updating the claim,
   never by deleting a gate that still describes the code.
2. **Find defects the audits only pointed at.** The five 2026-11-08 audits leave
   a named list: LayerRoPE `exp` overflow at `a_mag=95`, the WSD edge where
   `warmup+decay > total` never reaches the floor, the KDA `g_min=-5` floor from
   K3, `web.bin.docs.json` (a shard-seam sidecar the loader would accept), and
   the 9 claims-ledger mismatches.
3. **Reduce the codebase.** Execute the deletion audit's safe tier: C2 (two
   DEAD config fields), C7 (vendor burn-eggroll + burn-ptrn — this one edits the
   burn-fused facade, so it needs a workspace build). Then re-measure.
4. **Keep the record honest.** Every number that tonight's audits retracted
   (`1.4621`, the `-0.267` Muon claim) must stay marked in the docs it appears
   in; the claims ledger gate must be re-armed with a `--since` mode.

## Hard rules

- No push to `origin/main`: it is a DIFFERENT lineage (see `.bulba/plan.md`).
  Push to `origin/backup/main-local-2026-11-08` only.
- One task per commit; stage the paths you touched, never `git add .`.
- A fix without a gate is unfinished. A gate that cannot fail is a defect.
- If the GPU is free and the work needs it, one heavy thing at a time.
- Never delete a test to make a suite green; if the test's claim is wrong, say
  in the commit why the claim moved.

## Queue

- [ ] 22 red tests: classify and fix
- [ ] LayerRoPE overflow guard
- [ ] WSD edge refusal
- [ ] KDA g_min floor (behind a flag, default off, with an A/B later)
- [ ] C2 dead config fields
- [ ] C7 vendor prune (+ cargo check)
- [ ] claims ledger `--since` mode + CI job
- [ ] README/site polish if time remains
