# Plan: the live queue

STATUS: ACTIVE. Superseded plan (2026-09-27, Rust minimal core) is frozen with
the lane — see `docs/PLAN-minimal-core.md`.

## What is true now

- One RTX 5070 Ti, 16 GiB. Python lane live (`port/`), Rust lane frozen.
- 95M dense byte model, 12 unique layers, BPB 1.49–1.52 on a held-out English
  window after 0.25 GB of bytes.
- Muon runs in bf16 Newton–Schulz at 5 steps: 1.92× the step for +0.01 BPB.
  3 steps is REJECTED: +0.27 BPB at [832, 3072] shapes.
- E1 (bytes vs patch-8) is running; it decides whether the patch unit ships.
- 1M context: state is 34 MB, the segment/CSA boundary contract is merged.

## Queue, in order

1. **E1 verdict** — patch BPB within +0.02 of bytes at matched bytes. If no, the
   patch unit becomes speed-only and the 16× story drops to ~6×.
2. **Recipe ladder** (`scripts/recipe_ladder.sh`): value residuals, sigmoid gate,
   FFN clip, tail NoPE, cooldown-to-zero. Budget 120 MB per arm.
3. **Warmup/LR** (`scripts/warmup_lr_ab.sh`): the 6e-4 arm measured 0.27 BPB
   WORSE than 3e-4 — the LR sweep is confounded by `warmup_tokens=0`.
4. **Batch sweep** (`scripts/batch_sweep.sh`): the model step is at ~52% of bf16
   peak with b=8; a bigger batch is the remaining throughput lever.
5. **MoD A/B** at capacity 0.5 — 1.6× FLOPs is unmeasured here.
6. **1M probe** (`port/scripts/probe_1m.py`) once a checkpoint with CSA coarse
   is on disk: forward memory, prefill ms, needle recall.
7. **275M @ 22 GB** — the Chinchilla-optimal point for that budget (~9 h at 16×),
   then 1B @ 80 GB (~5 days) after the memory work.

## Codebase work (in flight)

- Deletion audit (`docs/reviews/2026-11-08-deletion-audit.md`) — agent running.
- 1B-on-one-card research (`docs/research/2026-11-08-1b-on-one-card.md`).
- Post-training research (`docs/research/2026-11-08-post-training-at-1b.md`).

## Waiting on the owner

- `port/` → `python/` rename: scripted (`scripts/split_lanes.sh`), never run.
  It touches every doc path, so it goes after the audit lands.
- Rust lane: delete or keep frozen. The audit reports its real status.

## Debt that needs an owner decision

- **claims ledger**: 1 875 claims, 9 mismatches, 373 without evidence (all dated
  before the rule). `scripts/gate_fast.sh` reports it instead of blocking, and
  CI has no ledger job — so nothing enforces the rule today. The fix is a
  `--since <date>` blocking mode plus a CI step; it is not done.
- **docs drift**: 10 of 10 checked pre-2026-11-01 documents contradict the code
  (`docs/reviews/2026-11-08-deletion-audit.md` §7).
- **deletions**: C2 + C7 (the 2 010-line safe tier) are unstarted; C4 is done.

## Gates that must stay green

`scripts/gate_fast.sh`, `scripts/gate_format.sh`, `pytest -q`, `dead_knob_audit`,
`check_doc_refs.py`, and (new) the `site` CI job that refuses a page with an
unfilled `/*__DATA__*/` placeholder.
