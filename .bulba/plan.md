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

## INCIDENT 2026-11-08: 94 GB of artifacts deleted by the merge

`data_local/` (80 GB corpus) and `checkpoints/` (14 GB) are gone — `origin/main`
tracks both as SYMLINKS over the local ignored directories, and the merge
checkout removed the directories to write the links. Full report:
`docs/reviews/2026-11-08-merge-incident.md`.

Recovered so far: `flagship3.pt` copied out of /tmp into
`checkpoints/flagship3/`; `port/scripts/fetch_mix_sources.py` refetched 600 MB of
code/math/RU; `port/scripts/refetch_pretrain.py` is fetching fwedu/cosmo/finemath
(1.2 GB each). The EVAL WINDOW is gone with the corpus: any run on the refetched
sources is a NEW number and must say so.

**Do not merge or checkout while these paths exist.**

## 2026-11-08 05:00 — after the artifact loss: the corpus is back, the ladder is running

**Data recovery is DONE** (the incident report's ladder, steps 1-4):

| file | bytes | how |
|---|---:|---|
| `data_local/best/fwedu_train.bin` | 933 743 230 | FineWeb-Edu shard 0, head |
| `data_local/best/fwedu_eval.bin` | 8 386 155 | the tail from a document separator — a file no source reads |
| `data_local/best/cosmo.bin` | 302 198 823 | Cosmopedia auto_math_text |
| `data_local/best/finemath.bin` | 948 125 886 | FineMath 3plus (the first download was truncated; re-curl + size check) |
| `data_local/mix/{code,math,ru}.bin` | ~210 MB each | fetch_mix_sources.py |
| `data_local/parquet/*.parquet` | 3.0 GB | curl -L, then DuckDB local read (`--local-dir`) |

**The recipe ladder is running** on it: `FAST=1 MIX=configs/mix_refetch.toml
RUN_TAG=_refetch`, 7 arms x 14648 steps x 8192 B = 120 MB each. First arm
measured 49.8 kB/s (30 MB in ~10 min), so ~40 min/arm, ~5 h total. Numbers are
NEW numbers: the old mix4 window is gone.

**The 1M context question has an artifact answer**
(`docs/research/2026-11-08-long-context-eval.md`):

- `flagship3` (the only 95M on disk) is a **1K dense control, no CSA** — it
  cannot run a 1M test. The 1M design is the LoopBlock form, and the one trained
  CSA checkpoint (`longctx1`, 102.7M, 8K) is **not on disk**.
- No 95M long-context result exists in the literature; the closest measured
  points are 300M Infini (passkey collapses past 4-8K), 0.5B SRRM, 1.24B
  Llama-3.2-1B in HELMET (2.0 inftyBench / 24.6 HELMET at 128K).
- NIAH correlates with real tasks at rho 0.44-0.76, so retrieval evidence buys a
  **mechanism** claim: "exact retrieval at 1M", not reasoning.
- State correction: 33 MiB is ONE CSA slot; the loop carries one per iteration
  (3 slots = 99 MiB; 12 layers = 396 MiB). README and the site now say that.

**Next on the 1M path** (needs the card, so it queues behind the ladder): train a
CSA model at 8K on the refetched corpus (longctx1's recipe, 102.7M, ~500 MB
seen), then run the S-NIAH-8 harness (`port/scripts/niah.py`, being built) for
gate D.

## Claims ledger: what the nine mismatches actually were (2026-11-08 05:20)

The claims-repair review found that **seven of nine were ledger parser artifacts**
(reading the wrong column of a table), and **two documents were genuinely stale**:
`muon-variants.md` (medium totals since `907b3b5` set `use_msa=false`) and
`token-skipping.md` (b1 count since `af8b0c9`). Both fixed with the old value kept
as superseded. The ten drifted claims from the deletion audit are repaired too:
three README numbers, the PPO-as-shipped paragraph, the 1B plan's self-contradiction,
and status banners on ADR-0014 (MSA not cut), ADR-0003 (fused deleted), the "final"
bank architecture (dense won), the scaling recipe's "expert-rich" direction, and
the burn-byteflow DELETE row.

**The gate itself is not armed and now says why:** `--no-params` makes the
mismatch count a lower bound (1 vs 9), and "new" means the document is dated
today — 220 claims in tonight's own reports are "new" by that rule. Arming it
needs line-accurate evidence association first.

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
- **deletions**: C4 (agent_rl) is done. **C2 is BLOCKED, not skipped**: deleting
  `jepa`/`muon_sf_retract` breaks `test_every_snapshot_on_disk_loads`, because
  153 checkpoint snapshots on disk carry `jepa = false` and the loader refuses an
  unknown key BY DESIGN (that refusal is a gate, not a nuisance). The honest way
  to retire a field is a `RETIRED_KEYS` allowlist in the loader that accepts the
  old name with a counted note — a decision, not a silent edit. Reverted with the
  test green; C7 (vendor) is still open and needs a workspace build.

## SUITE GREEN, CPU-only (2026-11-08 06:20)

`CUDA_VISIBLE_DEVICES="" pytest -q` over all 2 194 tests: **exit 0**. That is the
first green full run of the night; the 22-red starting point is closed. The two
launch-count gates still need a card run (they skip on CPU), and the card is the
ladder's until ~09:30.

## Known-red: reclassified 2026-11-08 05:40 (the list was wrong)

The old table said "seven pre-existing, all in the fused/CUDA-graph cluster,
several pass alone". `docs/reviews/2026-11-08-fused-cluster.md` measured the
seven with CUDA hidden: **five SKIP** (they are card gates), so the "7 red CPU
suite" was a card-visible run. The real content:

| # | test | class | fix |
|---|---|---|---|
| 1 | `test_full_graph::test_the_first_replay_is_the_eager_step` | **real defect**: the AdamW bias-correction counter (`_adamw_t`) was not advanced by the torch path, so the graph fallback resumed at a stale t (3.1e-05 at embed.weight) | **fixed** + CPU two-arm gate |
| 2 | `test_fused_epilogue::test_the_kernel_is_fp32_roundoff[one-gain]` | stale tolerance: gradients held to 1e-6 where the module's own `PARITY_GRTOL` is 5e-5 | **fixed** |
| 4 | `test_fused_residual::test_gate_reddens_when_the_pre_bias_moves` | stale test: patched `fr.pre`, which the compose path never calls | **fixed** (patches `fr.epilogue_read`) |
| 6 | `test_invariants_checkpoint::...[kda+shareu]` | flaky: unseeded model + sign-LFB controller can read all zeros | **fixed** (seeded) |
| 7 | `test_moe_combine_fused::test_the_fused_combine_counter...` | stale: asserted the bf16 counter on a default fp32 store | **fixed** (two arms) |
| 3 | `test_fused_kda::test_kernel_launch_counter_drops` | card-only, never run | guard added; needs one card run |
| 5 | `test_fused_residual::test_the_fused_path_issues_far_fewer_launches` | card-only, no arithmetic change since written | guard added; needs one card run |

The two card gates still need a quiet card (the ladder owns it until ~10:00).

## Gates that must stay green

`scripts/gate_fast.sh`, `scripts/gate_format.sh`, `pytest -q`, `dead_knob_audit`,
`check_doc_refs.py`, and (new) the `site` CI job that refuses a page with an
unfilled `/*__DATA__*/` placeholder.
