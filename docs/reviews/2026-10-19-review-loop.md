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

### Round 1 — 7/10 (`2026-10-19-review-round-1.md`)

Eight verdicts on the fixes; two gates could not fail; one regression caused by
the wave (the new counter reason crashed `opt_graph`). Regressions and the
vacuous gates were fixed before round 2.

### Round 2 — 6/10 (`2026-10-19-review-round-2.md`)

Lower, honestly: two regressions the previous wave introduced (the no-`--eval`
tail carved in scored bytes; the LFB refusal reaching readers), one gate that
compared `bytes` with `str`, and — new, and the most serious finding of the
whole loop — **a sympy RCE in the math verifier** (the model's answer text
executed in the trainer process; `__import__('os').system(...)` ran) plus a code
jail bypassed through `os.open`/`pathlib`. Both were fixed before round 3.

### Round 3 — 7/10 (`2026-10-19-review-round-3.md`) — the loop's result

Six of seven fixes verified as falsifiable (the seventh, the M1 corpus decode,
was made falsifiable afterwards). Four new findings: the reward path died on a
23-byte answer (`(2**1000)**1000`), the M1 decode half was still vacuous, LFB
scoping stopped at the pretrain trainer and `lfb_counts` accumulated across
resumes (11.8e9 on `mix1`), and a mixture run extended with more `steps`
resumed from the wrong position. All four fixed with revert-checked gates.

## The loop's outcome: 7/10, accepted by the owner (2026-10-20)

The owner's call: 7 is a defensible first ceiling; 8-9 comes later. What keeps
it from 8, named rather than implied:

1. **Gates that cannot fail** are a class, not an incident — one was found in
   every round. The fix is a habit (every gate gets a revert check in the
   commit), not a patch.
2. **Each wave of fixes produced a regression** in the next round. The counter
   to this is fewer, slower waves with the whole suite and the reviewers'
   own tests run between them.
3. **The GPU gates were written but never executed** — the card was held by a
   live run through all three rounds, so the reviewer read `skipif`s. A round
   with a free card is the single cheapest way to move the number.

Reaching 8-9, concretely: free the card and re-run this protocol once (all
three audits' findings are already fixed); make the revert check part of the
commit template; and let the reviewer run the distributed lane
(`tools/dist_gate.py`, two ranks) which no round has executed.
