# The run registry

Every training invocation writes one JSON record:

```
<runs-dir>/<run_id>/run.json
```

It answers the question a JSONL log cannot: **which run produced this number,
with what recipe, on what bytes, on which host — and did it finish?** The
record is written twice and only twice: `status = "running"` before the first
forward, then `done` or `failed` when the loop returns or raises. It is not in
the step path.

`runs-dir` is:

| caller | default | why |
|---|---|---|
| `dormouse train` / `dormouse run` | `logs/runs` (`--runs-dir`) | one index across many runs |
| `dormouse runs list\|show` | `logs/runs` (`--runs-dir`) | reads the same index |
| a library caller of `train()` without `runs_dir` | `<out_dir>/runs` | the artifacts keep their provenance when a checkpoint directory moves |

`run_id` is a UTC timestamp plus the checkpoint name (`20261108-143012-port`);
a same-second collision gets `-2`, `-3`. A resume is a new invocation and gets
its own record — the checkpoint's step is inside the config/`eval_line`, and
the `run.json` file answers "what did this process attempt", not "what does the
checkpoint contain".

## Schema (`schema = 1`)

| field | meaning |
|---|---|
| `run_id` | the directory name; unique per invocation |
| `status` | `running` → `done` \| `failed` |
| `timestamp`, `finished` | UTC ISO-8601 start/finish times |
| `duration_s` | wall seconds between the two writes; `null` while running |
| `git_sha` | `git rev-parse HEAD` in the checkout, `null` off git |
| `preset` | `{source, path, sha256}` — the exact preset file the run resolved from |
| `config` | the **resolved** `model` and `train` config (every field, including `--set` overrides) |
| `params` | instantiated parameter count |
| `train_bytes` | the byte budget asked for: `steps × batch × seq_len × grad_accum` |
| `data` | rows `{role, path, bytes}`: `train`, `eval`, `mix_plan`, `mix:<name>`; a directory is summed, a missing path is `null` |
| `data_bytes` | sum of the source rows (the mix plan file itself excluded) |
| `host` | `{gpu, torch, python, device}`; `gpu = "cpu"` on the CPU path |
| `eval_line` | the last eval line the run printed, verbatim; `null` if it never evaluated |
| `final_bpb` | the `bpb_fin` on that line, as a number the table can sort; `null` without an eval |
| `error` | `"<ExceptionType>: <message>"` on `failed`, `null` otherwise |

A malformed record is a **loud** read error, never a silently skipped row: a
table that drops one run is the defect the registry exists to prevent. `status =
"running"` with an old `timestamp` and no `finished` means the process died hard
(SIGKILL, power loss) — the two-write contract cannot record a death it never
saw.

## Why the preset hash matters

`--preset small` is a **name**, not a recipe. The file behind the name can be
edited between two runs, and a preset loaded by bare name is resolved through a
search path (cwd `configs/`, then the checkout's). Two runs whose `preset.sha256`
differ did not use the same recipe, even if both printed `source = "small"`.

The `config` snapshot is the other half and they are not redundant: the hash
proves the **file** was identical, the snapshot proves the **resolved values**
were — a `--set lr=2e-4` changes the run without touching the preset bytes.
A run needs both to name its recipe: the file it started from and the values it
actually ran.

## Relation to the JSONL telemetry

`--json-log` writes an append-only event stream (`run`, `log`, `eval`, one row
per cadence), consumed by `dormouse plot` and by `dormouse.runlog` for
rollback-safe curves. It can span several resumes and its `run` rows are the
only place a per-segment byte count lives.

`run.json` is one **summary per invocation**:

- the registry is the index (id, recipe, data, host, outcome);
- the JSONL is the time series (the curve);
- `eval_line` in the registry is exactly the last line the JSONL's final `eval`
  row describes — one string, the same number, two consumers. A run that never
  evaluated has `eval_line = null`; a run that evaluated forty times has the
  last one here and all forty in the JSONL.

Neither replaces the checkpoint's own `<ckpt_name>.config.toml`: that snapshot is
what a resume/decode/eval **loads**, while the registry is what a reader
**queries** after the fact.

## Commands

```bash
dormouse runs list                  # every run in logs/runs, newest first
dormouse runs list --runs-dir path  # another index
dormouse runs show <run_id>         # the record as JSON
```

`dormouse.runs` is stdlib-only on purpose: `list`/`show` read JSON and format a
table. The CLI module itself imports torch for every other command, so the
process still pays that import; the registry code path never touches a tensor.

## Example (fabricated registry)

```
$ dormouse runs list --runs-dir /tmp/opencode/fake-registry
                                 runs · /tmp/opencode/fake-registry
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ id                      ┃ date                ┃    params ┃   bytes ┃   bpb ┃ status  ┃ duration ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│ 20261108-143012-port    │ 2026-11-08 14:30:12 │ 9,476,993 │ 5.0 MiB │ 1.498 │ done    │   37m12s │
│ 20261108-101500-nano_s2 │ 2026-11-08 10:15:00 │ 7,192,906 │ 2.0 MiB │     - │ failed  │     4.2s │
│ 20261107-221045-port    │ 2026-11-07 22:10:45 │ 9,476,993 │ 5.0 MiB │     - │ running │        - │
└─────────────────────────┴─────────────────────┴───────────┴─────────┴───────┴─────────┴──────────┘
```

`runs show 20261108-101500-nano_s2` prints its `run.json`, including
`"error": "RuntimeError: stream open exploded"`. The running row is a process
that has not written its second record, and the BPB is `bpb_fin` from the run's
last eval line — the same headline the eval printed.
