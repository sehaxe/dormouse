# The checkpoint format

The Python lane writes one **pair** per checkpoint, plus one optional sidecar:

| file | written by | what it is |
|---|---|---|
| `<out_dir>/<ckpt_name>.pt` | `save_ckpt` | the self-describing payload: weights, optimizer, step, provenance |
| `<out_dir>/<ckpt_name>.config.toml` | `save_ckpt` → `RunConfig.snapshot_toml` | the resolved config the run had — the resume gate |
| `<out_dir>/<ckpt_name>.ngram` | `save_ckpt`, only when `train.engram_ram=true` | the host n-gram tables and their momentum |

`ckpt_paths()` is the one definition of the two paths
(`port/src/dormouse/train/loop.py`), and `ngram_path()` the third. The step
lives **only** in the `.pt` (`step` is 0-based: it is the last step that ran);
the `.config.toml` holds no step, so a resume reads its progress from the file
and its recipe from the snapshot.

## Version

`format_version` is `CHECKPOINT_FORMAT_VERSION` in
`port/src/dormouse/train/loop.py`.

- **v1** — every checkpoint written before this format landed (2026-11-08).
  It has no version key at all.
- **v2** — v1 plus `format_version`, `git_sha`, `corpus`. No v1 key changed
  meaning; v2 is metadata only.

A loader reads `format_version`, defaulting to 1 when the key is absent. A
version **newer** than the build is a LOUD refusal naming the version and the
recorded `git_sha`. The reverse is deliberately not guarded: a pre-v2 build
ignores the three new keys and loads a v2 file with v1 semantics, which is safe
because v2 adds no semantic field. The guard therefore only stops an *older*
reader from silently meeting a *newer semantic* format; the day a v3 changes a
value, the older build must start refusing, and the version check is already in
place for that day.

## v2 payload

`torch.save` of a plain `dict` (no classes, no lambdas):

| key | type | meaning |
|---|---|---|
| `format_version` | `int` | `2` |
| `git_sha` | `str \| None` | `git rev-parse HEAD` at save time; `None` off a checkout or without git |
| `corpus` | `dict` | the mixture manifest, below |
| `model` | `dict[str, Tensor]` | the RAW training weights, storage rung packed (`model_quant_rung`) |
| `model_ema` | `dict[str, Tensor]`, optional | the EMA shadow — the SERVED half; present only for an `ema_decay > 0` run |
| `opt` | `dict` | `MixOptimizer.state_dict()`: step count, moments, param groups |
| `step` | `int` | the last completed optimizer step, 0-based |
| `model_quant_rung` | `str \| None` | `fp8` / `nvfp4` / `ternary` when `model` is packed, else `None` |
| `moe` | `dict[str, Tensor]` | the LFB routing biases, which are non-persistent buffers and are not in `model` |

The `moe` section is state, not telemetry: `lfb_bias` is the integrator's
output and is not reconstructible from `lfb_counts`. A training load of a file
that carries `lfb_counts` without it refuses when the controller is live
(`moe_lb_coef != 0`); a serving read COUNTs the zero-bias line instead.

### `corpus` — the mixture manifest

```text
{"mix_plan": str | None,
 "plan_sha256": str | None,
 "sources": {name: bytes | None, ...},
 "sha256": str | None,
 "missing": [name, ...]}
```

- A run with `train.mix_plan` set records the plan's sha256 and every declared
  source's **name + size** (a directory source is the sum of its `*.txt`
  shards, the same call the stream makes).
- A single-file run (`train.data`) records that file's basename and size, with
  `mix_plan = null`.
- A missing plan or source records `sha256 = null` and the name in `missing`.
  The save still happens: the 2026-11-08 artifact loss is exactly the case where
  the corpus is gone and the checkpoint is what survives. The loader prints a
  COUNTED line for such a v2 file; it never refuses a resume over it.

**The hash identifies the mixture's SHAPE, not its bytes.** It covers the plan
file's TEXT (so an edited stage or weight changes it) and each source's name and
size. Swapping a source for a different file of the same size is invisible to
it — that is the price of a stat-only record, and it is stated here so nobody
reads the hash as a corpus digest.

## The write is atomic

`_atomic_save` in `port/src/dormouse/train/loop.py` writes
`<dest>.tmp`, `flush` + `os.fsync` the file, `os.replace` it onto `<dest>`, then
fsyncs the **directory** entry. The tmp file is unlinked on any exception.

`os.replace` alone orders the name, not the bytes behind it: without the two
fsyncs a power loss can leave the new name pointing at unwritten blocks. With
them, a crash at any instant leaves either the old complete file or the new
complete one. The `.pt`, the `.config.toml` and the `.ngram` all use this path.

The three files are written in the order weights → `.ngram` → snapshot. A crash
between the weights and the snapshot leaves the old snapshot with the new
weights; the resume gate compares configs (which are the same) and reads the
step from the `.pt`, so the pair stays legal. There is no cross-file step stamp
yet — that is the `.ngram`/snapshot pairing limitation carried in `AGENTS.md`
§3.3, not a regression of this format.

## Loading

| caller | function | trust |
|---|---|---|
| resume (`train --resume`, `sft`, `rl`) | `load_ckpt` | this run's own file: `torch.load(weights_only=False)`, the **config-snapshot drift gate** is the trust boundary |
| `eval`, `decode`, `bench`, `serve`, `chat`, `port/scripts/eval_window.py`, `port/scripts/niah.py` | `load_model_half` | a reader: prefers `model_ema`, unpacks the storage rung, attaches `moe` |
| `--init-from` (`train`, `sft`, `rl`) | `init_cfg` + `load_model_state` | weights leave their recorded context; the LFB coefficient comes from the init file's snapshot, LOUDly |
| a foreign file | `load_weights_only` / `eval --weights-only` | `torch.load(weights_only=True)`: refuse any pickle that names a class |
| host tables | `load_ngram` | already `weights_only=True` (tensors + ints only) |

`load_weights_only` is the **foreign-file** path. It accepts plain tensors,
ints, floats, strings, lists and dicts — which is exactly a v1/v2 dormouse
payload, so `eval --weights-only` reads our own files. It refuses a pickle that
names any other class, because loading such a file executes whatever it names.

Where it **cannot** be used: `load_ckpt` (a resume). A resume must read the
optimizer and controller sections, and whatever a future version adds, from a
file the snapshot gate already ties to this run; adding `weights_only=True`
there would refuse a legitimate future section for no trust gain. `--init-from`
readers that need only weights can call `load_weights_only` directly.

## What a v1 file loses on load

Nothing functional. `load_ckpt` reports `format_version = 1` and the payload has
no `git_sha` and no `corpus`; `model`, `model_ema`, `opt`, `step`, `moe` and
`model_quant_rung` load exactly as before. What is lost is provenance:

- it cannot name the commit that wrote it;
- it cannot name the mixture that produced its step count;
- a v1 file is indistinguishable from a truncated v1 file that still unpickles.

The atomic writer removes the third class for v2 and later. The gate that every
snapshot on disk still resolves is
`port/tests/test_config.py::test_every_snapshot_on_disk_loads`; the v1 payload
gate is `port/tests/test_ckpt_format.py::test_a_v1_file_loads_and_loses_only_provenance`.
