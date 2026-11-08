# AGENTS.md — dormouse (Python stack)

This directory is the **main implementation**, not a side port (the name is
historical): config, model, trainer, post-training and every current number.
The Rust lane (`../crates/`, `../vendor/`) is **frozen since 2026-11-07**, kept as
the formula source and measurement history; its old `main` lives on GitHub as
`archive/rust-lineage-2026-11-08`. The root `README.md` states the same split.

Repo-wide rules (`../AGENTS.md` §1) and machine facts (§2) still apply; its §3
status is Rust-era and frozen — prefer this file and the root README for what is
current. Read §1 and the §2 sections your change touches before any GPU claim.

## Commands (run from this directory)

```bash
uv sync                        # CPU-only: CUDA gates skip, fused arms fall back COUNTED
uv sync --extra gpu            # torch 2.13.0+cu129 + FLA 0.5.2 — the measured pair, don't bump one alone
uv run dormouse info           # resolved config + parameter budget (meta device)
uv run dormouse train small --data corpus.bin --eval tail.bin --device cuda \
    --json-log run.jsonl --set steps=2000
uv run dormouse eval   --out-dir checkpoints --ckpt-name small
uv run dormouse decode --out-dir checkpoints --ckpt-name small --prompt "…" --n 200
uv run dormouse plot run.jsonl

uv run pytest tests/test_config.py -q     # one file; torch import dominates (~20 s)
uv run pytest                             # ~2.3k CPU gates: ~9 min idle, ~20 min under load
PYTHONPATH=$PWD/src uv run python -m pytest -q -p no:logging   # exact CI invocation
uv run ruff check src scripts tests
bash ../scripts/gate_format.sh --changed main                  # changed-files format gate
PYTHONPATH=src uv run python scripts/dead_knob_audit.py --preset-glob 'configs/**/*.toml'
# exit 2 = a preset sets a field nothing reads (dead knob)
```

## Rules that bite here

- **Agents never launch GPU jobs.** One 16 GB card = one process; the orchestrator
  runs gates/A-B/sweeps serially. Write code, tests and docs on CPU.
  `tests/conftest.py::gpu_free` is the bench precondition (<1 GiB used, no dormouse
  process).
- **No silent fallback**: every degradation is LOUD (error naming cause + escape)
  or COUNTED (counter/log line). A new fast arm needs its counter; a fix needs a
  gate that fails without it.
- **A/B or delete**: 3 seeds, a win must beat the control's spread, a tie deletes.
  A number needs its config, eval window and commit.
- **Don't check out or merge across the Rust-era lineage over `data_local/` or
  `checkpoints/`**: those old refs track the dirs as symlinks, and a checkout
  materialised the link and deleted 94 GB of untracked data (2026-11-08). Current
  `main` does not track them — verify with `git ls-tree <ref> -- port/data_local
  port/checkpoints` before touching an old ref.
- One task per worktree (`../tools/wt.sh`), commit per task path-scoped;
  **no push/force/destructive git without the owner's OK**.
- Torch's default thread count is part of the gate: 26/28 hash-pinned byte-exact
  tests fail at `OMP_NUM_THREADS=4`, all at 1. Don't pin threads without
  re-capturing those hashes.
- A clean worktree (repo-relative) needs untracked `port/checkpoints/**/*.config.toml`
  and `port/sft_pairs/**`, or `test_config` / `test_sft_v2` go red for the wrong reason.

## Gotchas the code won't tell you

- CPU-green ≠ GPU-tested: fused KDA needs `--extra gpu`; without `fla` it falls
  back COUNTED. Read counters (`kda=…`, `moe_dropped_assignments=…`), not pass/fail.
- CUDA graphs (torch 2.13): `torch.bincount` is not capturable — use
  `experts._bincount`. A graph replay moves params without bumping `_version`, so
  `invalidate_banks_all(model)` must run every trained step
  (`tests/test_bank_invalidation.py` pins the premise).
- Config is strict flat TOML (`config.py`): unknown key = LOUD, `--set` applies
  after preset validation, and on resume the checkpoint's `.config.toml` snapshot
  is authority (progress keys exempt — don't edit the snapshot to force a resume).
- An eval window is bytes, printed on every eval line; BPB is comparable only
  inside one window. The window must be a **separate file** from every training
  source — the overlap gate compares paths, not offsets.
- `add_(g, alpha)` on CUDA is a one-rounding `fma`; the CPU aten path is not an
  oracle for fused optimizer kernels.
- Some comments still say "RTX 5060 Ti" — that is the card the numbers were
  measured on. This box is a 5070 Ti 16 GB (sm_120); machine limits live in
  `../AGENTS.md` §2.
- Docs follow `../docs/README.md` (one root `docs/`, dated names); don't create a
  second doc tree here.
- `../docs/decisions/2026-10-15-1b-readiness-plan.md` is the live plan.
  `.bulba/goal.md` in this directory is dated (2026-10-22) — check before trusting
  it as current.

## Layout

- `src/dormouse/config.py` — preset schema, validation, resume rules (the strict
  part; `configs/` is force-included into the wheel, so a new preset ships with it).
- `src/dormouse/model/` — tsct · experts · kda · engram · loopblock · transformer
  · byteflow; `train/` — data · optim · lr · loop · sft; `cli.py` is typer.
- `instrument.py` — LOUD/COUNTED marks and counters; `tests/` is CPU-tiny by
  design; `scripts/` holds bench/A-B tooling (`bench_*`, `ab_*`).
