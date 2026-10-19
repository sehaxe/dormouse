# Findings ledger — the 2026-10-19 audits and their fixes

Three independent reviewers read the tree on 2026-10-19 (code, model, infra) and
this file is the ledger: every finding, whether it is fixed, and what proves it.
A finding leaves the ledger only by being fixed **with a gate** or by being
rejected **with a reason** — the same rule the project applies to its arms.

Status values: `fixed` (with the proof named) · `open` · `queued` (a task exists)
· `rejected` (with the reason).

## Sources

- `docs/reviews/2026-10-19-infra-audit.md` — verdict 5/10 (honesty 9,
  reproducibility 3, automation 5). Reviewer: independent, read-only.
- `docs/reviews/2026-10-19-model-scorecard.md` — the model's honest position:
  "the instrument is the product; the model is a demonstration".
- `docs/reviews/2026-10-19-code-audit.md` — pending (the reviewer is still
  reading).

## CRITICAL

| # | finding | status | proof |
|---|---|---|---|
| I1 | the live run's only non-`--set` input was untracked (`configs/small_wide.toml`) and its command line lived on tmpfs | **fixed** | `configs/small_wide.toml` committed in `90d5e7f` |
| I2 | README's first command exits 2: `dormouse train` has no `--ckpt-name`/`--out-dir` | **fixed** | the quick start now uses `--set ckpt_name=… --set out_dir=…` (`90d5e7f`) |
| I3 | data and artifacts live outside both checkouts, so "clone → train → measure" is false | **fixed (documented)** | `docs/guides/data.md`: what the corpora are, the smallest honest recipe, what the eval carve must be; README points at it |
| I4 | README sold depth-3 as "the measured optimum" while every preset runs `max_iter=4` and the mixture runs' own curve still improves at 4 | **fixed** | README states the A/B's numbers, its window and the open question; AGENTS.md "Retracted — the Python lane" repeats it |
| I5 | the README's HellaSwag comparison cited comparator numbers that are in no model's own suite, and crossed protocols (byte `acc_mean` vs token `acc_norm`) | **fixed** | README/README.ru now say: above the 25% floor, below SmolLM2-135M's 42.1, no protocol bridge exists |
| I6 | three artifacts (`b1night`, `medium_big2`, `sft9`) are invalid — the expert-bank cache froze under `opt_graph` — and nothing in the repo said so | **fixed** | AGENTS.md "Retracted — the Python lane"; the fix is pinned by `tests/test_bank_invalidation.py` |

## MAJOR

| # | finding | status | proof / plan |
|---|---|---|---|
| I7 | `benches/history.tsv` deleted by the restructure while ~87 citations name it, and `check_doc_refs.py` skips `benches/` wholesale | **fixed** | restored in `90d5e7f`; the skip stays (a bench log is evidence, not a path) |
| I8 | the demo's README said the artifact is "not in the repository" while the blob is committed | **fixed** | README says why it is committed and what a fork without it sees (`90d5e7f`) |
| I9 | CI installed no node, so the browser-engine parity gate skipped itself — the one gate that proves the shipped engine computes this model's arithmetic | **fixed** | `actions/setup-node@v4`, node 22, in `ci.yml` (`90d5e7f`) |
| I10 | AGENTS.md §3 carried none of the Python lane's retractions | **fixed** | the new section at the top of AGENTS.md |
| I11 | no tags, no releases; 8 dependabot PRs unmerged | **open** | needs the owner: tags are outward, and the PRs are red against the old `main` |
| I12 | the local suite cannot be green while a training run holds the card (`AdamW.step()` → `AcceleratorError`) | **queued** | a `conftest` guard that turns "the card is busy" into a skip, not an error |
| I13 | the run row of the JSONL telemetry carries no `ckpt_name`/`out_dir`/`mix_plan`, so a run's recipe is not recoverable from its own log | **queued** | one line in `train/loop.py`'s run event |
| M1 | "×36.6 dense equivalent" for b1 appears in no file; the derived ratios are ×28.1 / ×18.7 / ×59 by component | **open** | the scorecard's §1; the project memory carried the ×36.6 — replace it with the derived table or drop the claim |
| M2 | `b1` param count disagreed between the port README (…113) and the pinned test (…209) | **fixed** | the plan's row carries the date and both numbers |
| M3 | no protocol bridge between our byte-level bench and published token-level suites | **queued** | the scorecard names it as one of the three measurements that would change the verdict |

## What the ledger is for

The review loop the owner asked for: fix → an independent reviewer verifies the
fixes, hunts new defects and scores the framework out of 10 → if the score is
below 9, another round (at most three). The score, the round number and what
moved it live in `docs/reviews/2026-10-19-review-loop.md`.
