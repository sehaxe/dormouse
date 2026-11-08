# `dormouse diagnose` — the log IS the verdict

One command reads a run's log and names the failure that a human used to find
by scrolling:

```bash
# a stdout training log
dormouse diagnose ~/logs/csa8k.log

# its telemetry, plus the run's own config (unlocks 4 of the 8 rows)
dormouse diagnose ~/logs/ladder/control_refetch.jsonl \
    --snapshot ../checkpoints/<run>/<run>.config.toml

# the same rows for a script or a CI gate
dormouse diagnose ~/logs/csa8k.log --json
```

The command parses; it does not load a checkpoint and builds no model. Exit
status is **1 iff a row is FAIL**, 0 otherwise, and 2 if the input is not a
readable log/snapshot. WARN is a counted/recovered defect; SKIP means this log
carries no evidence for that row — never "it passed".

Input is ONE file: a stdout log (the plain `step=...` lines) or the
`--json-log` telemetry. Each log format has evidence the other does not, so
the table says when a row needs the other file.

## The eight rows

| row | FAIL when | evidence it read |
|---|---|---|
| `log readable` | no run/log/eval/ckpt line at all, or a non-final line is not JSON | line numbers |
| `loss finite` | a printed loss is `nan`/`inf`; **or** skipped steps run across two consecutive logged windows, or total ≥ 10 | `loss=`, `nan_skipped=` |
| `lfb divergence` | `lfb_bias range` past ±10; or the routing counters hold one clique (≥ 50 % max share, or ≥ 90 % top-10 when E > 10) | `lfb_bias range [...]`, `moe_route_max_x1000`, `moe_route_top10_x1000`, `moe_route_n` |
| `dead experts` | mean `moe_route_live / moe_route_n` < 50 % of `n_experts` (WARN < 80 %) | `moe_route_live`, `moe_route_n`, snapshot `n_experts` |
| `lr schedule` | the printed `lr` is identical for ≥ 3 rows inside warmup/decay where the schedule must move; or `lr=0` while steps remain | `lr=`, snapshot `warmup_tokens`/`decay_frac`/`min_lr_frac`/`steps` |
| `dataloader stall` | JSONL `bytes_seen < step × batch × seq_len × grad_accum` | `bytes_seen`, the `run` row's `bytes_per_step` |
| `eval window overlap` | the `--eval` file is also a training source | snapshot `data`/`mix_plan`/`eval_data` + the plan's `[sources]` |
| `checkpoint` | the run finished but the last `ckpt step=` is behind it, or the snapshot's cadence says one was due and none was written | `ckpt step=`, `done steps=`, snapshot `ckpt_every` |

The table prints `N pass · N warn · N fail · N skip`, so a run with four SKIPs
is visibly not a run with four PASSes.

## `--snapshot` is what makes half the rows possible

`--snapshot` is the checkpoint's own `<ckpt_name>.config.toml` — the resolved
run, not a preset. Four facts live only there:

* `model.n_experts` — the denominator of the dead-experts row;
* `train.warmup_tokens` / `decay_frac` / `min_lr_frac` / `bytes` — which `lr`
  windows should move, and where a zero is a deliberate cooldown;
* `train.data` / `mix_plan` / `eval_data` — the eval file's identity;
* `train.ckpt_every` — whether a checkpoint was due at all.

Without it those rows degrade to SKIP or a WARN that says what is missing.
That is deliberate: `diagnose` will not guess `n_experts` from a counter, and
it will not fail a benchmark run whose `ckpt_every=1000000` disabled
checkpointing on purpose.

## The pathologies

### 1. Loss NaN/inf — `losses.py` firewalls

The trainer prints the last **finite** loss (`last_loss_dev`), so a NaN step
can hide behind a healthy-looking `loss=` number. The counter is the truth:
`nan_skipped` is per window on the log line and cumulative in the JSONL's
top-level field.

```
step=200 loss=2.4000 lr=1.000e-03 ... nan_skipped=800
```

* one window with skips → WARN: the firewall did its job, but the step's
  update was dropped and the raw loss belongs in the report;
* skips in two consecutive windows, or ≥ 10 steps in total → FAIL: the
  readiness plan's `nan_windows >= 2` rule — a drowning run stops for a human;
* a printed `loss=nan`/`inf` → FAIL immediately, naming the step.

Action: lower the LR, check the newest data shard, or bisect the arms; the
firewall skips an update, it cannot restore a finite objective. See the NaN
section of `AGENTS.md` §2.3.

### 2. LFB divergence — the bias left the router's logit scale

The load-time line is printed on a checkpoint load (training refuses past the
bound, readers print it):

```
checkpoint: lfb_bias range [-218.269, +72.667] over 1 module(s) is past ±10: ...
```

Past ±10 the controller's bias overwhelms the router's own scores (±5) and the
order freezes: on `m400` at step 149 999, 9 of 512 experts carried every
assignment and one held 4096 of 8192 (`docs/reviews/2026-11-04-routing-collapse-CONFIRMED.md`).
That file cannot be resumed into a healthy bank and no LR repairs it — the
dead experts receive no gradient.

In-training there is no bias line, so the row reads the routing-health
counters (they exist only when `moe_routing_health=true`):

| counter | meaning | healthy |
|---|---|---|
| `moe_route_live` | experts with ≥ 1 assignment in the forward | `live/n` near `n_experts` |
| `moe_route_max_x1000` | busiest expert's share × 1000 | `≈ 1000/E` |
| `moe_route_top10_x1000` | ten busiest experts' share × 1000 | `≈ 1000·min(10,E)/E` |
| `moe_route_ent_x1000` | `H(p)/log E` × 1000 | near 1000 |
| `moe_route_n` | routed forwards accumulated | divide every counter by it |

Bars: FAIL at max ≥ 50 % (one expert takes the bank) or top-10 ≥ 90 %;
WARN at max ≥ 25 % or top-10 ≥ 60 %. The repo's acceptance line is
**live ≥ 80 % of E and top-10 ≤ 30 %** (`scripts/ab_balance.py`), so a PASS
here is not a healthy bank — read the `dead experts` row for that.

Action: start fresh with `moe_lb_mode="prop"` (`moe_lb_gamma`, bounded by
`moe_lb_clamp`), not by resuming the diverged file; for an existing file, use
it only as `--init-from` if you accept the collapsed routing as the start.

### 3. Dead experts — live/total from the same counters

`moe_route_live / moe_route_n` is the mean live count per forward; the row
divides it by `n_experts` from the snapshot.

* `< 50 %` → FAIL: the bank is not a bank, and every dead expert carries no
  token and no gradient (the same failure threshold as the LFB row, from the
  other side);
* `< 80 %` → WARN: under the acceptance line of `scripts/ab_balance.py`;
* `n_experts = 1` → PASS/SKIP: a dense run has no bank to collapse.

If the snapshot says `n_experts > 1` but the log carries no counters, the row
is a WARN naming `moe_routing_health=false`: a collapse in that run would be
invisible, and that is itself worth fixing (one `bincount` + one `topk` per
forward, per the config's own note).

### 4. LR frozen/zero — the 2026-11-07 `muon_lr_mult` class

`lr` is printed every log window. The row checks it against the WSD windows
from the snapshot:

* `warmup_steps = ceil(warmup_tokens / (batch × grad_accum × seq_len))`;
* `decay_start = max(steps − round(decay_frac × steps), warmup_steps)`.

Inside a window where the schedule is strictly monotone, **3 consecutive
logged values that are bit-identical** is a FAIL — provided the window is wide
enough that `lr`'s `%.3e` print (0.1 %) would have shown the movement. The
2026-11-07 defect was exactly this shape: the trainer wrote the schedule into
the Muon group on every step, overwriting `lr × muon_lr_mult` before the
optimizer could read it, so the knob was inert and the arm measured the knob
off (`docs/reviews/2026-11-07-muon-audit.md`, D1).

`lr = 0` while the run declares more steps is a FAIL — every remaining step is
a no-op. `lr = 0` at the **final** step with `min_lr_frac = 0` is the
deliberate full cooldown (the Ripple-65B arm) and passes.

Without a snapshot the row cannot know the windows: a fully constant `lr` is a
WARN ("constant schedule is legal; this is also the frozen shape"), and a
moving one passes.

### 5. Dataloader stall — a step that consumed less than its window

The JSONL's `bytes_seen` is cumulative and exact by contract:

```
bytes_seen == step × batch × seq_len × grad_accum
```

One byte short means the loader handed the step less than its window (the
2026-10-17 `grad_accum` defect was this arithmetic missing a factor); above
means the same bytes were counted twice. The row reads `bytes_per_step` from
each `run` row, so a resume with a new batch size does not false-fire.

The plain stdout log prints no `bytes_seen`; the row is a SKIP there, never a
PASS. `tokens_per_s` is a rate, not a byte count, and the row refuses to
derive one from the other.

Action: check the stream's cursor against the step (`resume: skipped N
micro-batches, expected M`), the shard repack, and `grad_accum` in the run
row.

### 6. Eval window overlap — the held-out number is not held out

Found by the 2026-11-08 data/eval audit: the `off=100,000,000` window on
`best/fwedu.bin` scored bytes the run's own cursor had already passed
(108.9–110.0 MB), so a "held-out" BPB was a training measurement
(`docs/reviews/2026-11-08-data-eval-integrity.md`).

The row resolves `eval_data` and compares it by path with `data` plus every
`[sources]` entry of the mix plan — the same predicate as
`scripts/eval_window.py` (`dormouse.eval_window.training_overlap`). The
trainer's own startup gate, `data.eval_train_overlap`, does the stronger
inode-level check against the live streams; this row is the cheap one that
works from a post-mortem.

Ceiling, stated so it is not mistaken for a guarantee: path identity only. A
copy of a source under another name passes this row and is still not held
out. Single-file mode (`eval_data` unset) is a PASS — the tail split is
disjoint by construction.

### 7. Checkpoint not written at the end

Three different facts, reported separately because the fixes differ:

* the run has `ckpt step=` lines but the last one is **behind** the last
  logged step, and `done steps=` says the run finished → FAIL: those steps
  exist only in memory;
* the run is truncated (no `done`) and the last checkpoint is behind the last
  step → WARN: a resume loses only that tail;
* no `ckpt step=` line at all → FAIL only when the snapshot proves a
  checkpoint was due (`ckpt_every <= steps`); otherwise PASS (disabled on
  purpose, e.g. the recipe ladder's `ckpt_every=1000000`) or WARN when there
  is no snapshot to decide.

The JSONL sink records `run`/`log`/`eval` only — it has no checkpoint event —
so on a `.jsonl` input this row is a SKIP unless the snapshot disables
checkpointing. Pass the stdout log for the real answer.

## What the tool cannot see

* **A copy of the corpus under another name.** The overlap row is path
  identity; the trainer's inode gate is the authority at run time.
* **The checkpoint's own step from telemetry.** The sink has no `ckpt` event;
  the file's step is in the `.pt`, which this command does not load.
* **A collapse without counters.** `moe_routing_health=false` makes rows 2–3
  blind; the WARN names the flag instead of pretending.
* **A constant-LR run from a frozen schedule without a snapshot.** The two are
  the same log; the snapshot's schedule windows are what separates them.
* **Anything before the first logged line.** A 5-step smoke with
  `log_every=100` has no rows; the command reports SKIP, not PASS.

## Reading the exit status in a gate

`--json` carries `fail`, the `counts`, and one `{check,status,evidence}` per
row. A CI gate should treat exit 1 as "the run's log names a failure"; WARNs
are counted and do not fail the gate, because the repo's rule is that a
COUNTED fallback is not a silent one.
