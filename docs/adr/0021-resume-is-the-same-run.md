# ADR-0021 — A resume is the same run, or it is a hard error

Date: 2026-09-27. Status: accepted. Amends ADR-0005 (the config seam) and
ADR-0019 (loud failures).

## The rule

**A resume must reproduce the run it continues, or refuse to start. It may never
silently continue a different one.**

Stated mechanically, in the order the checks run:

1. **Config drift is a hard error** (ADR-0005, unchanged): the resolved config is
   snapshotted next to the checkpoint and diffed key-by-key on every resume. The
   exempt class is exactly the keys that change *when we look*, never *what the
   model computes* — `steps`, `log_every`, `ckpt_every`, `eval`, `eval_every`.
   Extending a finished run is a legitimate resume, and none of those keys can
   alter a step that has already run. **Everything else is strict**, including
   every knob that reaches a forward.
2. **Run state the config cannot express must be in the container** (§2 below).
   The config is the *plan*; the EMA teacher, the latched quant fallback and the
   stream offset are the *run*, and a resume that re-derives any of them from the
   config is running a different experiment.
3. **The only randomness in a step is a pure function of `(seed, step)`**. No
   global RNG, no unseeded draw, nothing whose value depends on how many draws
   happened before it.
4. **A checkpoint that cannot carry the state says so loudly** rather than
   proceeding quietly. `legacy` containers load, and print exactly what is
   missing.

Why this rule and not "be careful": the failure is not a crash, it is a *wrong
number*. A resumed run that quietly swaps its EMA teacher produces a loss curve
belonging to two different experiments under one step count, and nothing in the
log distinguishes them. Every loss curve, eval number and A/B in this repo is
read off that curve. The alternative — detecting the discontinuity by eye — is
exactly the class of check ADR-0019 exists to delete.

## The inventory: eight items, what each does to the objective

Line numbers are as of `d96d155`; the drift-check item's line numbers in the
original review were stale (the progress keys were already exempt) and are
corrected below.

| # | item | effect on the objective | status |
|---|---|---|---|
| 1 | EMA teacher rebuilt as a copy of the student on every resume | **changes the JEPA target from step 1** of the resumed run | fixed |
| 2 | the one-way fp32-factor fallback is not persisted | **changes the forward** (quantized factors again) for up to 500 steps per resume | fixed |
| 3 | the JEPA mask is drawn from burn's global RNG, never seeded | **changes the mask on every step**; no two runs are ever alike | fixed |
| 4 | `StressMonitor` state is lost on resume | does not change the objective (constant LR) but makes the stability report across a resume fiction | **reported**, not fixed |
| 5 | the drift check exempts the harmless keys and is blind to `rand_depth` | `rand_depth` samples the loop depth per step, so it reaches every gradient; the drift check could not see it | fixed |
| 6 | the byte-stream offset lands 1–3 batches behind after a resume | **re-trains on batches the run already saw** (2 with the default warmup, 3 with `--quant-check`) | fixed |
| 7 | `.ngram` written after the model, no rotation, no step stamp | a crash between the two writes leaves a mismatched pair; the sidecar is only validated at the next load | partly fixed, remainder reported |
| 8 | nothing asserts a `#[serde(skip)]` field is cfg-derived | the *mechanism* by which item 5 went unnoticed | fixed |

### The asymmetry this exposes

ADR-0005's drift check exists, is implemented, and **demonstrably works**: a
snapshot was found carrying 13 dead schema keys and missing 10 live ones, and the
resume hard-errored on it. That is the check doing its job, and it is why the
finding above is not "the check is broken".

The gap is in **state the config cannot express**. `steps`, `max_iter`,
`jepa_weight` and every other knob are in the config, so they are comparable. The
EMA teacher's momentum, the quant fallback latch and the stream offset are not
config at all — they exist only because the loop advanced — so there was nothing
for the check to compare and no place to put the truth. Item 5 was the one case
where config *could* have expressed it (`rand_depth` is a flag) and the snapshot
chose not to.

## What changed

### 1. The teacher and the fallback latch are in the container

`save_ckpt` / `load_ckpt` (`crates/dormouse-train/src/lib.rs`) now write and
read a v2 container:

```
[magic "DMCK\0\2\0\0"][step u64][model_len u64][optim_len u64][teacher_len u64][flags u64]
[model burnpack][optim burnpack][teacher burnpack]
```

- The **teacher record** is the EMA teacher as the run left it, restored with
  `no_grad()` applied after the load (a record does not carry the freeze flag, and
  the freeze is what keeps the teacher forward off the autodiff tape).
- **`flags` bit 0** is the one-way fp32-factor fallback. Restoring it re-applies
  `set_quant_all(Fp32)` to the loaded model.
- A **pre-ADR-0021 container is detected by the magic** (it starts with the raw
  step), still loads, sets `legacy = true`, and carries no teacher and no flags.
  The trainer prints exactly what it could not recover, and names the container
  as the reason. It does not pretend.

The teacher doubles the checkpoint's size. That is the honest cost of a
resumable EMA teacher, and it is only paid when the teacher exists (aux on and
not using offline targets).

### The forward path is re-applied after the load

Found while fixing item 2, and it is the same bug: `load_ckpt` rebuilds the model
with `DormouseModel::new(cfg)` and loads only the *params*, so every non-param
field came back at its default. The factor-quant format and the bf16-compute flag
are non-param fields. A resumed `--quant fp8` run therefore trained the **fp32
forward** while its own log line said `quant format: Fp8 (8 bits)` — a second
experiment wearing the first one's name. `apply_compute_settings` is now called
after the load as well as before it.

### 2. The mask is a function of `(seed, step)`

`crates/dormouse-core/src/aux.rs`: `mask_stream(t, mask_frac, mask_span)` draws
Bernoulli starts from a splitmix64 of `(seed, step, index)` on the host and
dilates them causally into spans — the same distribution and the same semantics as
`burn_jepa::mask_indices`, which it replaces at the one call site. `set_mask_stream`
is a pair of atomics, set once per step by the trainer: no RNG state to carry
across a resume, and nothing to restore.

`TrainCfg::seed` (default 1) is new, exposed as `--seed`, and part of the config
snapshot, so changing it mid-run is drift like any other objective knob.

The seam is two atomics rather than a parameter threaded through every forward
signature: the trainer knows the step index and the config, the model does not,
and every caller of `forward_with_hidden` (the eval path, the fused gradcheck,
`examples/`) would have to learn about a step counter it does not use.

### 3. The drift-check exemption, and the round-trip test

**The rule:** exempt the progress/cadence class (`steps`, `log_every`,
`ckpt_every`, `eval`, `eval_every`) — keys that decide when we look, never what
the model computes. Everything else is strict.

This is the *opposite* of the review's framing, and the framing is the half that
was stale: `diff_keys` has exempted the progress keys since 2026-09-23 (the first
real resume hit the check and the author exempted them there). The inversion was
never in the comparison — it was in the **snapshot**: `rand_depth`, `eval_batches`
and `eval_depths` were `#[serde(skip)]`, so the strict comparison had nothing to
compare. All three are in the snapshot now, with the reason for each written on
the field.

`snapshot_carries_every_train_field` (item 8) is one test for the whole class: the
snapshot must carry every field `TrainCfg` has, checked by comparing the
serialized `train` table against the field list and by count. A skipped field is a
deficit whatever its name; adding a field without adding it to the snapshot fails.
The same test flips the four knobs that were invisible (`rand_depth`,
`eval_batches`, `eval_depths`, `seed`) and asserts each one reaches `diff_keys` —
a key in the snapshot that the check ignores is decoration.

### 4. The stream offset

`skip = step * batch * seq_len` was the bug: a *fresh* run consumes 1–2 batches
before step 0 (the two warmup forwards share one probe batch, and `--quant-check`
consumes another), and a resume skips the probes, so the resumed run landed
`pre_batches` behind and re-trained on data it had already seen. The skip is now
`(step + pre_batches) * batch * seq_len`, with `pre_batches` derived from the same
two conditions the probes are gated on. The precompute path
(`precompute_jepa_targets`) already mirrored this correctly and is the reason the
discrepancy was findable.

### 5. The ngram sidecar

The **final** checkpoint never wrote the sidecar: a finished run left
`model@final` over `tables@last-cadence`, and the next resume replayed up to
`ckpt_every` steps of row updates on top of a model that had already trained on
them. `save_ngram` is now a function called on every model save including the
final one.

## What remains, and the exact fix

**Item 4 — `StressMonitor` state (`lib.rs:664`, `stress.rs`).** `history` (a
201-entry window), `spikes`, `grad_norms` and `refused` are all in-memory. A resume
restarts the window empty, so `spikes=0` and the p99.9 is computed over post-resume
gradients only. The objective is unaffected (the stress protocol holds the LR
constant, which is a function of config, not of history), so this corrupts the
*report*, not the model — which is why it is not in the fixed list. The fix: give
`StressMonitor` a `to_bytes` / `from_bytes` (it is five scalars and two `Vec`s,
~20 lines) and put it in a fourth section of the same v2 container, or in a
`<ckpt_name>.stress` sidecar written next to it. It was not done here because it
is a fourth payload in a container this change already changes, and a run using
`--stress` should get a consistent fix rather than a partial one.

**Item 7 — the ngram sidecar has no step stamp.** A crash *between* the model
write and the sidecar write leaves `model@S` over `tables@S-k`, and nothing can
tell: `from_bytes` validates the magic and the length, and the step field in the
sidecar header is the *table* step (Nesterov updates), not the training step. The
exact fix: add a `train_step: u64` to the sidecar header, write it on every save,
and refuse to load a sidecar whose `train_step` is not the loaded checkpoint's
step — the same refusal the model loader already applies to a corrupt record. It
was not done here because it changes the on-disk sidecar format that a 48M-slot
production run (18 GB) has on disk, and a format change that invalidates a
day-scale artifact wants its own decision. The two halves that need no format
change — writing the sidecar on the final save, and doing it loudly — are done.

**`load_model_weights`** (inference) reads the same container and was updated for
the new header, but it builds the model with default (fp32) forward settings. For
`generate` and `serve` that is harmless — inference wants the fp32 forward, and
the checkpoint stores fp32 masters by design — so it is noted, not changed.

## Compatibility

- Old containers load. New code reads both formats; old code reading a new
  container would misparse it, which is why the magic is the first field and the
  version is in it.
- Old *config snapshots* load. `rand_depth`, `eval_batches`, `eval_depths` and
  `seed` are all `#[serde(default)]` at the container level, so a snapshot written
  before this change parses with the defaults — and a resume of such a run with
  `--rand-depth` set will now hard-error on drift, which is the correct outcome:
  that resume would have been a different run.
- The container grew by 24 bytes of header plus the teacher record.
