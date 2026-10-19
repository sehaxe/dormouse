# The review loop — round 0 (the audits), and the protocol

The owner's rule (2026-10-19): after the audits, fix what they found, then run an
independent reviewer that (a) verifies the fixes, (b) hunts new defects, and
(c) scores the framework out of 10. Below 9, another round; at most three.

## Protocol

1. Every finding gets a row in `docs/reviews/2026-10-19-findings-ledger.md` with
   its status. A finding leaves the ledger only by being fixed **with a gate** or
   rejected **with a reason**.
2. A round is not started until the ledger's CRITICAL rows are `fixed` and every
   MAJOR row is `fixed` or `queued` with a named task.
3. The reviewer of a round is independent: read-only, no stake in the fixes,
   given the ledger and told to verify each claimed fix by running or reading the
   gate it names, then to hunt for what the fixes broke and what none of the
   audits has looked at yet.
4. The score is the reviewer's, not ours. If it is below 9, the round's findings
   are triaged into the ledger and round N+1 starts.

## Round 0 — the three audits (2026-10-19)

| reviewer | scope | score | report |
|---|---|---|---|
| code | src/ + tests/ + the docs-vs-code seam | **6/10** | `docs/reviews/2026-10-19-code-audit.md` |
| infra | tree divergence, CI, hygiene, reproducibility, process | **5/10** | `docs/reviews/2026-10-19-infra-audit.md` |
| model | what the model is, every measured number, the A/B ledger | no score (a scorecard) | `docs/reviews/2026-10-19-model-scorecard.md` |

The composite position after round 0: **the instrument is stronger than the
model**. Two holes in the measurement chain matter more than anything else the
audits found — an init the seed does not control, and an SFT corpus that shares
its prompts and solutions with the RL/bench task set — because both make the
project's own numbers irreproducible or self-scoring. They are the first two
fixes, not the last.

## Rounds

### Round 0 — findings → fixes (in progress)

See the ledger. The CRITICAL rows are the two measurement-chain holes above; the
MAJOR rows are the storage-rung prescale, the never-printed fallback reason, the
SFT eval's un-rewound stream, the red-while-busy CPU gate, the tail-path window
that overstates what it scores, two inert knobs, three per-forward device syncs
and the LFB bias that does not survive a resume.

### Round 1 — pending

Started after the ledger's CRITICAL rows are fixed. Reviewer: same scope as the
code audit, plus a read of the two other reports, told to verify each fix and to
look specifically at what changed.

### Round 2 — pending, only if round 1 scores below 9.

### Round 3 — pending, only if round 2 scores below 9. Last round.
