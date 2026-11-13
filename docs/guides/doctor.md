# `dormouse doctor`: the pre-flight for a long run

> Updated 2026-11-08. Sources: `port/src/dormouse/cli.py` (the command and every
> row), `port/tests/test_doctor.py` (one pass case and a failing row for each
> class below), `scripts/vram_ledger.py` (the VRAM arithmetic the `vram` row
> prints), `port/src/dormouse/train/mixture.py` (the plan's resolution rule),
> `port/src/dormouse/eval_window.py` (the overlap predicate),
> `docs/records/reviews/2026-11-08-merge-incident.md` (the lost corpus),
> `docs/records/reviews/2026-11-08-data-eval-integrity.md` (the window audit), and
> `docs/records/architecture/2026-11-08-training-step.md` §I8 (resume drift).

Nine hours of card time is too expensive to spend discovering a missing file.
`dormouse doctor` runs the checks a training start would hit, prints one row per
check, and exits before the card is booked:

```console
cd port
.venv/bin/dormouse doctor configs/flagship3_dense.toml \
  --set batch=8 --set seq_len=1024 --set muon_state_dtype=bf16 \
  --set mix_plan=configs/mix_refetch.toml \
  --set data=../data_local/best/fwedu_train.bin \
  --set eval_data=../data_local/best/fwedu_eval.bin \
  --set out_dir=/home/sehaxe/dormouse/checkpoints/budget
```

`--set` is the same override path as `train`; `--data`, `--eval`, `--mix-plan`
and `--out-dir` are the same fields in friendlier spelling. `--device cpu` skips
the card rows (the pass case in `port/tests/test_doctor.py` runs that way).
`--root` names the git checkout that should carry the run's SHA (default `.`),
and `--json` prints the rows for CI.

**It does not initialize CUDA.** The only card access is one
`nvidia-smi --query-gpu=...` read; `torch.cuda.mem_get_info` is deliberately not
used because it creates a context on the card a real run is using.

## Exit codes

| exit | meaning |
|---|---|
| `0` | no FAIL. WARNs are allowed — a busy card and a dirty tree are things to know, not things that stop a run |
| `1` | at least one FAIL: this config cannot start (or cannot start honestly) on this box |

`--json` carries the same information as one object:

```json
{
  "ok": true,
  "preset": "configs/flagship3_dense.toml",
  "counts": {"pass": 16, "warn": 2, "fail": 0},
  "git": {"sha": "eb285e34615b7d5535674be742dda655d9563568", "dirty": true},
  "checks": [
    {"name": "vram", "status": "warn", "detail": "peak 2.64 GiB at b8 s1024 · ..."}
  ]
}
```

`checks[].status` is `pass`, `warn` or `fail`; `ok` is exactly
`counts.fail == 0`, so CI can gate on either.

## The rows

| row | what it reads | FAIL when | WARN when |
|---|---|---|---|
| `config` | `load_run(preset, --set ...)`, including every preset refusal | the config layer refuses (unknown key, bad preset, an arm that is not ported) | — |
| `torch` | `torch.__version__`, `torch.version.cuda` | never (a CPU-only torch shows up in `gpu`) | — |
| `gpu` | the one `nvidia-smi` reading: name, driver, `compute_cap`; plus `torch.cuda.is_available()` and `torch.cuda.get_arch_list()` | no CUDA device, no `sm_<cap>` kernel in this torch, or no `nvidia-smi` | — |
| `vram` | `vram_ledger.param_ledger` + `peak_bytes` at the run's batch/seq/levers, against the card's total and free MiB | the estimated peak exceeds the card's TOTAL — the config can never fit | another process holds the card (the repo's one-heavy-thing rule), or the estimate is unavailable |
| `mix` | `train.mixture.load_mix_plan`: unknown keys, stage weights, unused sources, TOML | the plan does not parse or is not a legal plan | — |
| `data`, `mix:<name>` | every file the run names: exists, readable, size | missing, unreadable, or smaller than one `batch × (seq_len+1)` read | — |
| `eval` | `eval_data`, its size, and `eval_batches_for`'s window at `eval_offset` | missing, unreadable, or the window runs off the file (the stream would wrap and score bytes twice) | — |
| `overlap` | `eval_window.training_overlap`: the eval path against `data` plus every resolved plan source | the eval window is ALSO a training source: the BPB is not held out | the plan could not be read (the `mix` row names why) |
| `snapshot` | `out_dir/<ckpt_name>.config.toml` through `RunConfig.from_snapshot` and `diff_keys` | the snapshot is refused (unknown key, bad TOML, a dead key at a non-default value): a resume cannot start | the snapshot parses but a `--resume` would refuse on the named keys (a fresh run overwrites the pair) |
| `disk` | free space on `out_dir` (or its nearest existing ancestor) against one atomic checkpoint: weights x (2 with EMA) + optimizer state | free space is below one checkpoint | free space is below two checkpoints (the atomic overwrite holds old and new) |
| `cache` | `~/.cache/torch/inductor` when `torch_compile=true` | — | the cache is missing or empty: the first compile warmup costs minutes |
| `git` | `git rev-parse HEAD`, `git status --porcelain` | the `--root` is not a git checkout, so the run cannot name its code | the tree is dirty: the SHA does not describe the code that runs |
| `ledger` | only printed when the estimate itself failed | — | the VRAM estimate could not be computed; the message names the exception |

A `data` file and a `mix:<name>` source are the same check; the second spelling
is how a plan's `[sources]` entry appears. A mix source may be a directory (the
shard stream); the main `data` file may not.

## The three failures that actually bit this repo

### 1. A data source that is not on disk

`data_local/` (80 GB) and `checkpoints/` (14 GB) were deleted by the
`origin/main` merge (`docs/records/reviews/2026-11-08-merge-incident.md`). A run started
after that fails in `ByteStream`/`MixtureStream` construction, or worse, reads a
truncated source. The `data` and `mix:<name>` rows stat every path before the
model is built, and the `mix` row parses the plan with the same function the
trainer uses (`train/mixture.py::load_mix_plan`), so a typo, an unweighted
source, or a relative path resolved against the wrong directory is named here.

The rescue is `port/scripts/refetch_pretrain.py` (fwedu/cosmo/finemath) and
`port/scripts/fetch_mix_sources.py` (code/math/RU); the doctor is how you know
which one is still missing.

### 2. An eval window inside the corpus

The data/eval audit measured a window at `off=100,000,000` on `best/fwedu.bin`
being scored while the run's own fwedu cursor had already passed 108.9-110.0 MB
(`docs/records/reviews/2026-11-08-data-eval-integrity.md`). The "held-out" BPB was
reading training bytes. The cheapest honest detector is a path comparison
(`port/src/dormouse/eval_window.py::training_overlap`, shared with
`scripts/eval_window.py` and `dormouse diagnose`), and the `overlap` row is that
same predicate. A relative `[sources]` path resolves beside the PLAN — what
`train/mixture.py` does — not against the caller's cwd.

A window on the same file can still be a different byte range, but that is not
held out in any sense a BPB claim can use — the doctor refuses it. Point
`--eval` at a file no source reads (`data_local/best/fwedu_eval.bin` is the
tail carved at a document separator), or name the number as a training-window
number.

### 3. A snapshot key the loader refuses

A checkpoint's `<ckpt_name>.config.toml` is the run's identity, and the loader is
strict by design: every key must be a field of this branch, or one of the
counted `DEAD_SNAPSHOT_KEYS` at its recorded default. Dropping an unknown key
silently would resume a different model, so it is a LOUD refusal
(`port/src/dormouse/config.py::from_snapshot`, ADR-0021; the drift table is
`docs/records/architecture/2026-11-08-training-step.md` §I8).

This is a real gate, not a nuisance: the deletion audit's C2 could not retire
`jepa` because snapshots on disk still carry `jepa = false`, and retiring a
field needs an allowlist that accepts the old name with a counted note — not a
silent edit. The `snapshot` row runs `from_snapshot` NOW, so a resume that would
die at minute ten dies before the card is booked, and a drift on a key outside
`PROGRESS_KEYS` is a WARN naming the keys.

## This box, 2026-11-08

The budget run's own flags (`port/scripts/train_budget.sh`), with the training
run holding the card and the checkout dirty from other lanes:

```
                      doctor · configs/flagship3_dense.toml
┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ check      ┃ status ┃ detail                                           ┃
┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ config     │ pass   │ configs/flagship3_dense.toml                     │
│ torch      │ pass   │ 2.13.0+cu129 · CUDA 12.9                         │
│ gpu        │ pass   │ NVIDIA GeForce RTX 5070 Ti · driver 615.78.08 ·  │
│            │        │ sm_120 · 15.9 GiB                                │
│ vram       │ warn   │ peak 2.64 GiB at b8 s1024 · 3.70 / 15.92 GiB     │
│            │        │ free: another process holds 12141 MiB (28 %) —   │
│            │        │ one heavy thing at a time; wait for the card.    │
│ mix        │ pass   │ configs/mix_refetch.toml · 6 sources · 1 stage   │
│ data       │ pass   │ ../data_local/best/fwedu_train.bin · 890.49 MiB  │
│ …          │        │ the six `mix:<name>` rows each pass              │
│ eval       │ pass   │ ../data_local/best/fwedu_eval.bin · 8.00 MiB ·   │
│            │        │ window 192.19 KiB at +0                          │
│ overlap    │ pass   │ fwedu_eval.bin is read by no source               │
│ snapshot   │ pass   │ fresh run (no budget.config.toml)                │
│ disk       │ pass   │ 70.26 GiB free · one checkpoint ~909 MiB         │
│ cache      │ pass   │ off (`torch_compile=false`)                      │
│ git        │ warn   │ da7bf899e24d · dirty: 44 paths differ from HEAD  │
└────────────┴────────┴──────────────────────────────────────────────────┘
16 pass · 2 warn · 0 fail
```

The card in use is a WARN and the exit code is 0: the arithmetic says the run
fits, and the row says who is holding the card. The dirty tree is also a WARN —
it does not stop the run, but the checkpoint's `git_sha` will not describe the
code that produced it, so commit or record the diff.

The moving numbers (free MiB, the dirty count, the mix-source sizes) are a
snapshot of a live box; the verdicts are the part to read.
