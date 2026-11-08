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

## The GitHub divergence (found 2026-11-08 while pushing)

`origin/main` is NOT this lane. Its tree is the Rust-era layout (`src/`,
`tests/` at the root, a 5060 Ti MoE README with `assets/banner.svg`), and its
HEAD is a data commit from that lineage. Local `main` is the Python lane under
`port/`, 357 commits ahead and 78 behind; an automatic merge produced **319
conflicts** (a directory flatten plus two overlapping trees) and was aborted.

What exists now: `origin/backup/main-local-2026-11-08` carries all 357 local
commits, so nothing is only-on-one-disk. `origin/main` is untouched.

Three ways forward, for the owner:
1. **Make the local lane canonical**: push local main to a new `origin/python`
   branch and merge it into `origin/main` as a rewrite (the Rust tree moves to
   `rust/` or `archive/`). One decision, one big commit.
2. **Flatten locally** to the remote's layout (`port/*` → root) and merge; the
   conflict count drops but the two `src/` and `docs/` trees still overlap.
3. **Keep two branches** (`main` = Rust lineage, `python` = this lane) and stop
   pretending they are one project.

Until the owner picks, pushes go to `backup/main-local-2026-11-08`.

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

## Known-red at 2026-11-08 02:30 (pre-existing, not from tonight)

Seven tests, all in the fused/CUDA-graph cluster, all written before tonight
(`acb013f`, `0f118b2`, `b342390`), and several are ORDER-dependent: the same
test passes when its file runs alone and fails in a mixed run, which points at
process-global counters rather than at the kernel.

| test | what it says |
|---|---|
| `test_full_graph::test_the_first_replay_is_the_eager_step` | the reference log never says "capture failed" |
| `test_fused_epilogue::test_the_kernel_is_fp32_roundoff[one-gain]` | parity 2.861e-06 vs tolerance 2.082e-06 (1.37x over) |
| `test_fused_kda::test_kernel_launch_counter_drops` | passes alone, red in a mixed run |
| `test_fused_residual::test_gate_reddens_when_the_pre_bias_moves` | the sanity gate did NOT redden on a 1e-4 perturbation |
| `test_fused_residual::test_the_fused_path_issues_far_fewer_launches` | launch count |
| `test_invariants_checkpoint::test_every_nonpersistent_buffer_is_carried_or_provable[kda+shareu]` | passes alone |
| `test_moe_combine_fused::test_the_fused_combine_counter_is_counted_by_the_kernel` | counter |

They need a GPU session with the counters printed, not a blind fix. Recorded so
the next full-suite run is read against this list, not against "22 red".

## Gates that must stay green

`scripts/gate_fast.sh`, `scripts/gate_format.sh`, `pytest -q`, `dead_knob_audit`,
`check_doc_refs.py`, and (new) the `site` CI job that refuses a page with an
unfilled `/*__DATA__*/` placeholder.
