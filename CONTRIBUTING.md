# Contributing

This repository has two lanes and one rule set. Read the rule set first; it is
short and every rule exists because something went wrong.

## The two lanes

| lane | where | what it is |
|---|---|---|
| **Rust** | `crates/`, `tools/` | the burn/cubecl implementation: kernels, ADRs, the published site |
| **Python** | `port/` | the training port: config, model, trainer, post-training, scripts |

They share the docs (`docs/`) and the data (`data_local/`, untracked).

## The rules that matter

1. **A number is a claim, and a claim names its evidence.** Every measurement in
   a doc carries the config, the date, the commit, and the log path. A number
   without those is deleted, not softened.
2. **A verdict needs a decision rule written before the run.** "It looks better"
   is not a verdict. State the threshold (e.g. `> 2 × spread and ≥ 0.02 BPB`),
   the seeds, the steps, and what result would kill the arm.
3. **A mechanism must prove it ran.** Every mechanism has a counter
   (`csa_forward`, `moe_expert_calls`, `engram_rows`, …). A config that claims
   the mechanism while the counter is zero is a bug, not a result.
4. **A knob must reach code.** If a config field is declared, validated, and
   read by nothing (or overwritten later), that is the defect class this repo has
   paid for most: `muon_lr_mult`, `fused_residual`, `doc_masking`, `seq_len=512`.
   `scripts/dead_knob_audit.py` exists to catch the next one.
5. **One heavy thing at a time.** One GPU process; agents are CPU-only. A full
   test suite next to three agents has OOM'd this machine.
6. **Off by default, and off means bit-identical.** A new mechanism ships behind
   a flag; the flag off must reproduce the previous numbers exactly, and a test
   must pin that.
7. **Every fix lands with a gate that fails on the un-fixed code.** Show it red,
   then green. A test that cannot fail is decoration.

## Before you commit

```bash
bash scripts/gate_fast.sh                 # ruff, format, doc refs, dead knobs, cargo
bash scripts/gate_fast.sh --changed       # CI mode: format only the touched files
pre-commit run --all-files                # the same, wired into git
```

## Tests

```bash
cd port && PYTHONPATH=$PWD/src .venv/bin/python -m pytest -q            # CPU suite
cd port && CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q -m gpu   # GPU-marked
```

Non-negotiable: **never skip or delete a failing test** — fix the code it accuses.
`@pytest.mark.local_pin` is for tests that pin one machine's bit-exact output;
every such pin needs a behavioural sibling that runs in CI.

## Style

See `docs/guides/style.md`. Short version: `ruff check` and `ruff format` are the
arbiter; 100 columns; type hints on public functions; no commented-out code; no
"for later" scaffolding.

## Pull requests

- One task per branch, one branch per worktree (`git worktree add`).
- Commit style: `type(scope): the change, in the imperative` — `fix(moe): …`.
- The PR body states: what changed, what was measured, what could regress, and
  which gate would catch that regression.
- Green `python` and `cpu` CI jobs are required; `cuda-compile`, `lint` and
  `coverage` are informative until they are closed (see ci.yml comments).
- No push to `main` without the owner's explicit approval.

## Where things live

```
crates/          Rust lane (core, data, train, cli)
port/src/        Python package: config, model/, train/, post/, serve
port/scripts/    drivers: bench_*, ab_*, audit_*, queue_*, sweep_*
port/tests/      pytest, one file per behaviour area
docs/adr/        decisions with numbers and dates
docs/reviews/    measurement reports (the ledger of what we know)
docs/research/   literature reviews with arXiv ids, versions, tables
lectures/        the SVD lecture (standalone, no dependency on the model)
```
