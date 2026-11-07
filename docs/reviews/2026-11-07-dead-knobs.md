# Dead knobs — the instrument and the 2026-11-07 census

**Deliverable.** `port/scripts/dead_knob_audit.py`, its gates
(`port/tests/test_dead_knob_audit.py`), and this census. Run it with:

```sh
cd port
CUDA_VISIBLE_DEVICES="" uv run python scripts/dead_knob_audit.py            # static, ~1 s
CUDA_VISIBLE_DEVICES="" uv run python scripts/dead_knob_audit.py --runtime  # + 2 CPU steps
```

## Why this exists

This repo has lost weeks to one defect: **a config knob that exists, validates,
and does nothing.** Confirmed instances: `muon_lr_mult` (overwritten every step
by the scheduler, so `1` and `8` were bit-identical), `fused_residual` (the
preset said on while the live run used the eager path), `use_msa` (rejected
months earlier but still in presets), `doc_masking` (0.06 % masked), and
`seq_len=512` (nobody re-checked the default for a year). Each looked like a
measurement until someone read the code.

The instrument finds the next one two ways, and checks one against the other:

* **static** — AST, not grep. Every read of a config field outside `config.py`
  is located and attributed to its owning class: attribute access
  (`cfg.d_model`), `getattr(cfg, "d_model")`, `cfg["d_model"]`,
  `cfg.get("d_model")`. Receivers are resolved through parameter annotations,
  one-hop aliases (`tcfg = run.train`), class attributes (`self.cfg = cfg`),
  `RunConfig.model`/`.train`, and a small name fallback for unannotated code.
  A module nothing imports is marked **orphan** and its reads do not count.
* **runtime** (`--runtime PRESET`) — an attribute hook on `ModelConfig` /
  `TrainConfig` / `RLConfig` records every field read from a non-`config.py`
  frame during a real two-step CPU run (`b=1`, `s=64`) of the trainer. Static
  and runtime are then diffed.

Verdicts: `LIVE` (read on a reachable path that can change a run), `LOG-ONLY`
(read only to build a log line), `DEAD` (never read outside `config.py`;
validators do not count), `SHADOWED` (read, then a later write to the same
state slot drops the value). Two evidence flags ride beside the verdict:
`init-only` (every behavioural read is in `__init__` — the `muon_lr_mult`
shape) and `guarded` (a `loud(...)` refusal names the field, so a non-default
value cannot lie silently).

Exit codes: `0` clean, `1` any `DEAD`/`SHADOWED`, `2` `PRESET-USES-DEAD` — a
preset sets a dead field to a non-default value. Code 2 is the gate that would
have caught `muon_lr_mult=8`.

## Census

**245 fields**: `ModelConfig` 157, `TrainConfig` 79, `RLConfig` 9.

| verdict | count |
|---|---|
| LIVE | 243 |
| LOG-ONLY | 0 |
| DEAD | 2 |
| SHADOWED | 0 |

**No preset sets a DEAD field off its default**, so `PRESET-USES-DEAD` is
empty. The two DEAD fields are both guarded and both harmless today:

- **`jepa`** (`port/src/dormouse/config.py:1387`, default `false`) — a
  redundant spelling of `jepa_weight > 0`; its only reads are the consistency
  refusal at `config.py:1558-1560` and the validations at `:3851`, `:3858`,
  `:4253`. Behaviour is driven by `jepa_weight`. The two presets that carry it
  (`flagship_base`, `m400b`) set it to its default, `false`.
- **`muon_sf_retract`** (`config.py:2046`, default `false`) — read only by its
  own refusal at `config.py:4969`; the port has no `retract_tsct` call site, so
  the refusal is the whole feature. No preset sets it.

**`muon_lr_mult` is LIVE, and that is the 2026-11-07 fix showing up.** Its reads
are `train/loop.py:1264` (constructor), `train/loop.py:1763` (the SF arm),
`train/sft.py:1119`, `post/rl_train.py:438`; the per-step apply is
`MixOptimizer.set_lr` (`train/optim.py:1916`), called at `loop.py:1769/:1771`
and `sft.py:1214`. Before that fix the same static pass would have shown
`init-only` (every behavioural read inside `__init__`), the risk flag that
sends a reader to the scheduler. The shadowing write itself was in another
file (`group["lr"] = lr` in the loop, against the constructor's
`lr * mult`), which the slot rule does not prove — see "What the instrument
cannot see".

## Ranked: most likely to be silently lying right now

### 1. `adaptive_msa = true` while `use_msa = false` — four presets

`adaptive_msa` (`config.py:1510`, default `false`) is captured at
`model/loopblock.py:403`, but its **only** use is `loopblock.py:1170`, inside
`if blk.msa is not None:` (`:1167`). `blk.msa` is built only when
`cfg.use_msa` (`:404`). Four presets set `adaptive_msa=true` and
`use_msa=false` in the same file:

- `configs/flagship_base.toml:114,118`
- `configs/m400b.toml:145,149`
- `configs/m400.toml:48,52`
- `configs/medium.toml:48,52`

The flag is inert in every one of them. **Action: a decision, not a
mechanical deletion.** Deleting the four lines changes nothing as committed —
and changes what `--set use_msa=true` arms, because that override is applied
after the preset is read. Keeping them is a note-to-self that the MSA arm, when
armed, wants the per-iteration schedule. A `validate` refusal cannot be added
in the same commit for the same reason: it would fire before the override and
break the documented way these presets are re-armed.

### 2. The MSA knobs under the same disabled gate

`msa_topk` (`config.py:1507`), `msa_block` (`:1508`) are read only in
`model/sparse/attention.py:154-155` and `model/sparse/sparse.py:71-72` —
constructors of `BlockSparseAttention`/`FlexSparseSelector`, built only when
`use_msa=true` (`loopblock.py:404`). `msa_capacity_ratio` (`config.py:1509`) is
captured at `loopblock.py:405` and used only inside the same
`if blk.msa is not None` branch. Five presets set all three while
`use_msa=false`: `b1`, `flagship_base`, `m400`, `m400b`, `medium`.

Measured, not inferred: a two-step CPU runtime pass on `configs/tiny.toml` plus
`use_msa=false, msa_topk=32, msa_block=4, adaptive_msa=true,
msa_capacity_ratio=0.25` read 143 fields, **never read `msa_topk` or
`msa_block`**, and read `adaptive_msa`/`msa_capacity_ratio` only at
`loopblock.py:403/405` (capture, not use). **Action: same as #1 — delete the
inert preset lines, or document them as an MSA-on-only block.**

### 3. Auxiliary-head knobs under a zero weight

The aux heads are built only when their weight is positive, but their shape
knobs are still config fields set by presets:

| head | weight gate | knobs set by presets (all at their defaults) |
|---|---|---|
| JEPA | `jepa_weight = 0.0` (`model/model.py:415-423`) | `jepa_horizon=1`, `jepa_reg_weight=0.1` |
| DSpark | `dspark_weight = 0.0` (`model/dspark.py:203-210`) | `dspark_k=4`, `dspark_stride=16`, `dspark_markov_rank=256` |
| MTP | `mtp_weight = 0.0` (`model/model.py:472`) | `mtp_k=0`, `mtp_soft=false`, `mtp_head_norm=false` |
| Energy | `energy_weight = 0.0` (`model/model.py:485`) | `energy_negs=64`, `energy_temp=1.0`, `energy_rank=64` |
| Halt | `halt_weight = 0.0` (`model/model.py:408`) | `halt_eps=0.005`, `halt_max_iter=8`, `halt_mode=false` |

Every one is at its default in `flagship_base` and `m400b`, so no run is lying
today. The risk is one edit: change `dspark_stride` to `32` under
`dspark_weight=0` and you have the `muon_lr_mult` shape with a plausible-looking
diff. **Action: document (each preset block already sits beside its weight);
if one of these goes non-default, add a `validate` refusal for that family in
the same commit.**

### 4. The fused-kernel flags are init-only

`fused_residual` (`config.py:869`), `fused_epilogue` (`:879`) and
`fused_kda_tail` (`:900`) are read only in constructors
(`model/loopblock.py:307/313/319`, `:512/520/528`, `model/kda.py:262`) and
stored as module state. That is exactly the `fused_residual` history: the field
is read once, and whether the *kernel* runs is a second question. All presets
set them `false`, so nothing is lying now. **Action: when arming one, gate on
the seam counter** (`fused kda=`, the epilogue counter), not on the config
value — the static pass proves a read, never that the kernel launched.

### 5. `jepa` — dead alias, guarded

Delete in a breaking change only: it is carried by `flagship_base` and `m400b`
and by any snapshot that spelled it, so removal needs a `DEAD_SNAPSHOT_KEYS`
entry and two preset edits. Until then the `__post_init__` refusal is the
protection: a preset that says `jepa=true` with a zero weight dies at load with
the escape in the message. **Action: document as an alias (it already is, in
code); delete only with the snapshot migration.**

### 6. `muon_sf_retract` — a refusal wearing a field name

The port never calls a retraction, so the field's only job is to refuse `true`
loudly (`config.py:4969`). That is a legitimate use of a config field — it turns
an inert arm into a load-time error. **Action: keep it, or delete the field and
the refusal together; do not "wire" it without the retraction it names.**

### 7. `PRESET-SETS-UNREAD` on `mirror_control`

The runtime pass read 77 fields at 129 sites. Five model keys in
`configs/mirror_control.toml` were never read: `use_kda=false`,
`use_engram=false`, `n_experts=1`, `max_iter=1`, `max_seq_len=1024`. This is
intentional and documented in the preset (`:30-34`: the fields keep a
`block_form="kda"` read of the same file legal for the A/B's opposite arm) —
the transformer form never builds the loop. **Action: none; the audit now
prints the list so the next reader does not have to infer it.**

### 8. `share_u` — an alias the audit resolves

`share_u` (`config.py:770`) has no read outside `config.py`, but
`__post_init__` (`config.py:1575-1581`) maps `share_u=true` to
`share_u_mode="plane"`, which is read by the expert bank
(`model/experts.py:1109/1140`). The alias check is why this is `LIVE` and not a
false `PRESET-USES-DEAD` on the four presets that set it. **Action: none.**

### Also checked, and not lying

- `use_msa` itself is LIVE and `false` in every preset that sets it — no arm
  is being claimed.
- `seq_len` is LIVE (30 read sites) and every preset that needs another window
  sets it; the 512 default is a default, not a claim.
- `doc_masking` is LIVE on the data path (`train/data.py:1002-1053`); its
  historical 0.06 % was a corpus/coverage defect, not a dead knob.
- `eval_depths` is LIVE, not log-only: it decides whether the eval runs the
  extra depth forwards (`train/loop.py:1013`). It is the case that narrowed the
  `LOG-ONLY` rule to reads inside a logging call (or a `*_field` formatter),
  not reads in a function whose name sounds like a log line.

## Evidence and gates

The instrument's own gate file pins the classifier, not the current census:

- **Gate (a)+(d)** `test_classifier_reads_a_temp_module` — a temp
  `@dataclass` with one read, one dead and one shadowed field reports
  `LIVE`/`DEAD`/`SHADOWED` with the exact declaration, read and write lines.
- **Gate (b)** `test_injected_dead_field_in_mirror_control_is_loud` — a copy of
  `configs/mirror_control.toml` with `dead = 42` appended makes the audit exit
  `2` and print `PRESET-USES-DEAD: dead`. This is the `muon_lr_mult` gate.
  The same command on a copy of the real `configs/mirror_control.toml` with
  `jepa = true` appended (a real DEAD field, non-default) exits `2` and prints
  `PRESET-USES-DEAD: jepa` — the classifier is not fixture-only.
- **Gate (c)** `test_runtime_and_static_agree_on_d_model` — the static and
  runtime passes both read `d_model` on `mirror_control`.
- **Gate (d)** `test_getattr_and_dict_style_reads_are_reads` — `getattr`,
  `.get("...")` and `LOG-ONLY` are classified from a temp module.
- `test_real_presets_do_not_set_a_dead_field_off_its_default` — runs the audit
  over the committed presets and fails if any sets a dead field non-default.
  This is the load-time refusal this task asked for, placed where it can see
  both the config classes and every preset; a refusal inside `config.py` would
  need a hand-maintained dead-field list that rots.

Runtime numbers for the record: `mirror_control`, `b=1`, `s=64`, 2 steps, CPU
(`CUDA_VISIBLE_DEVICES=""`), read **77 fields at 129 sites**; `d_model` was
read 3 times (`model/model.py`, `model/transformer.py`, …). The tiny KDA
experiment in #2 read 143 fields. The tool keeps the temp run's directory only
while it runs and deletes it afterwards.

## What the instrument cannot see

Stated so the next reader does not over-trust it:

- **Receiver resolution is heuristic.** An unannotated `cfg` is assumed to be
  a config object (the codebase does this ~700 times); a config passed under an
  unusual name in unannotated code is missed. `--json` carries the
  `unknown_receivers` list (e.g. `self.seq_len` on a stream) so a blind spot is
  visible, not silent.
- **Reachability is an import graph** from `dormouse.cli`/`dormouse.__init__`.
  A dynamic import (`importlib`, plugin registry) would be marked orphan and
  its fields DEAD. `model/sparse/attention.py` is the current example: it is
  imported by nothing, but `loopblock.py` imports `model/sparse/__init__.py`,
  so the package is reachable and `msa_topk` is still LIVE through
  `sparse/sparse.py`.
- **Conditional inertness is NOT a verdict.** A read inside `if use_x:` is
  LIVE even when a preset sets `use_x=false`; that is finding #1 above, and it
  is in the report because a human read the branch, not because the script
  proved it. The runtime pass catches the "constructor never called" half
  (`msa_topk` unread) but not the "stored then used under a gate" half
  (`adaptive_msa`).
- **SHADOWED is a slot rule, deliberately conservative.** It requires a pure
  capture into `self.<attr>` (or a string-key slot) and a later non-augmented,
  non-reset replacement in the same module/class. It would NOT have caught the
  pre-fix `muon_lr_mult` (constructor `lr*mult`, overwritten by
  `group["lr"]=lr` in a different file through a param-group dict). That case
  was found by a human reading the scheduler; the instrument now records it as
  `init-only` + the fixed read chain, which is the risk signal that would have
  prompted the read.
- **`LOG-ONLY` only recognizes** reads inside a call named like a logger or a
  `*_field` formatter. A read that feeds a dict logged later is counted LIVE.

## Fixes in this diff

No config field was deleted and no preset was edited. Neither DEAD field is
trivially safe to remove: `jepa` is carried by two presets and potentially by
snapshots (needs `DEAD_SNAPSHOT_KEYS`), and `muon_sf_retract` **is** a loud
refusal — deleting it would remove the only thing the field does. The four
`adaptive_msa`/MSA preset edits are behaviour-preserving, but they are the
arming point for `--set use_msa=true`; that is a decision, and it is #1 above.

What did land: the instrument, its five gates, and the CI refusal
(`test_real_presets_do_not_set_a_dead_field_off_its_default`) so the next
`muon_lr_mult=8` in a preset fails a test instead of costing a sweep.

## Appendix — every field

Every read site (all of them) is printed by `-v` and carried in `--json`; the table gives the declaration, the verdict, the first evidence and the presets that set the key.

| field | class | type | default | decl | verdict | evidence | presets |
|---|---|---|---|---|---|---|---|
| `act_group` | ModelConfig | `int` | `0` | port/src/dormouse/config.py:1531 | **LIVE** | port/src/dormouse/model/experts.py:1119; guarded | configs/b1.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/small.toml, configs/small_wide.toml |
| `act_quant` | ModelConfig | `str` | `'none'` | port/src/dormouse/config.py:1523 | **LIVE** | port/src/dormouse/model/experts.py:1118; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `activation_offload` | TrainConfig | `float` | `0.0` | port/src/dormouse/config.py:2342 | **LIVE** | port/src/dormouse/train/loop.py:1185; guarded |  |
| `activation_recompute` | TrainConfig | `str` | `'none'` | port/src/dormouse/config.py:2321 | **LIVE** | port/src/dormouse/train/loop.py:1182; guarded |  |
| `adam_eps` | TrainConfig | `float` | `1e-08` | port/src/dormouse/config.py:1727 | **LIVE** | port/src/dormouse/train/loop.py:1270; guarded |  |
| `adaptive_msa` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:1510 | **LIVE** | init-only: port/src/dormouse/model/loopblock.py:403 | configs/b1.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml |
| `agc_clip` | TrainConfig | `float` | `0.0` | port/src/dormouse/config.py:1719 | **LIVE** | port/src/dormouse/train/loop.py:1522; guarded |  |
| `agg` | RLConfig | `str` | `'prompt-mean'` | port/src/dormouse/post/rl.py:115 | **LIVE** | port/src/dormouse/post/rl.py:128; guarded |  |
| `attn_kind` | ModelConfig | `str` | `'kda'` | port/src/dormouse/config.py:647 | **LIVE** | port/src/dormouse/cli.py:904; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `attn_latent_rank` | ModelConfig | `int` | `0` | port/src/dormouse/config.py:714 | **LIVE** | port/src/dormouse/cli.py:243; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `attn_window` | ModelConfig | `int` | `0` | port/src/dormouse/config.py:670 | **LIVE** | init-only: port/src/dormouse/model/loopblock.py:426; port/src/dormouse/model/loopblock.py:428; port/src/dormouse/model/loopblock.py:440; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `attn_window_query_chunk` | ModelConfig | `int` | `0` | port/src/dormouse/config.py:678 | **LIVE** | init-only: port/src/dormouse/model/loopblock.py:438; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `batch` | TrainConfig | `int` | `8` | port/src/dormouse/config.py:1706 | **LIVE** | port/src/dormouse/cli.py:423; guarded |  |
| `bf16` | ModelConfig | `bool` | `True` | port/src/dormouse/config.py:587 | **LIVE** | port/src/dormouse/model/byteflow.py:215; guarded | configs/audit_d128.toml, configs/b1.toml, configs/flagship3_dense.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/mirror_control.toml, configs/small.toml, configs/small_wide.toml, configs/tiny.toml |
| `block_form` | ModelConfig | `str` | `'kda'` | port/src/dormouse/config.py:396 | **LIVE** | port/src/dormouse/model/model.py:243; guarded | configs/audit_d128.toml, configs/flagship3_dense.toml, configs/mirror_control.toml |
| `byteflow_bins` | ModelConfig | `int` | `16` | port/src/dormouse/config.py:958 | **LIVE** | port/src/dormouse/model/byteflow.py:211; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `byteflow_codec` | ModelConfig | `str` | `'attention'` | port/src/dormouse/config.py:973 | **LIVE** | port/src/dormouse/model/convcodec.py:142; guarded | configs/m400b.toml |
| `byteflow_d_ff_global` | ModelConfig | `int` | `4096` | port/src/dormouse/config.py:956 | **LIVE** | port/src/dormouse/model/byteflow.py:210 | configs/flagship_base.toml, configs/m400b.toml |
| `byteflow_d_ff_local` | ModelConfig | `int` | `1408` | port/src/dormouse/config.py:955 | **LIVE** | port/src/dormouse/model/byteflow.py:209 | configs/flagship_base.toml, configs/m400b.toml |
| `byteflow_d_global` | ModelConfig | `int` | `1536` | port/src/dormouse/config.py:944 | **LIVE** | port/src/dormouse/model/byteflow.py:202 | configs/flagship_base.toml, configs/m400b.toml |
| `byteflow_d_local` | ModelConfig | `int` | `512` | port/src/dormouse/config.py:943 | **LIVE** | port/src/dormouse/model/byteflow.py:201 | configs/flagship_base.toml, configs/m400b.toml |
| `byteflow_e_layers` | ModelConfig | `int` | `6` | port/src/dormouse/config.py:948 | **LIVE** | port/src/dormouse/model/byteflow.py:204 | configs/flagship_base.toml, configs/m400b.toml |
| `byteflow_eps2` | ModelConfig | `float` | `0.5` | port/src/dormouse/config.py:960 | **LIVE** | port/src/dormouse/model/byteflow.py:212; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `byteflow_g_layers` | ModelConfig | `int` | `20` | port/src/dormouse/config.py:950 | **LIVE** | port/src/dormouse/model/byteflow.py:205 | configs/flagship_base.toml, configs/m400b.toml |
| `byteflow_heads_global` | ModelConfig | `int` | `12` | port/src/dormouse/config.py:952 | **LIVE** | port/src/dormouse/model/byteflow.py:207 | configs/flagship_base.toml, configs/m400b.toml |
| `byteflow_heads_local` | ModelConfig | `int` | `8` | port/src/dormouse/config.py:951 | **LIVE** | port/src/dormouse/model/byteflow.py:206 | configs/flagship_base.toml, configs/m400b.toml |
| `byteflow_k_tokens` | ModelConfig | `int` | `3200` | port/src/dormouse/config.py:946 | **LIVE** | port/src/dormouse/model/byteflow.py:203; guarded | configs/final_patch.toml, configs/flagship_base.toml, configs/m400b.toml |
| `byteflow_logdet` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:964 | **LIVE** | port/src/dormouse/model/byteflow.py:213; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `byteflow_max_bytes` | ModelConfig | `int` | `8192` | port/src/dormouse/config.py:966 | **LIVE** | port/src/dormouse/model/byteflow.py:214; guarded | configs/final_patch.toml, configs/flagship_base.toml, configs/m400b.toml |
| `byteflow_patch_bytes` | ModelConfig | `int` | `8` | port/src/dormouse/config.py:979 | **LIVE** | init-only: port/src/dormouse/model/byteflow.py:852; port/src/dormouse/model/model.py:233; guarded | configs/m400b.toml |
| `byteflow_w_local` | ModelConfig | `int` | `512` | port/src/dormouse/config.py:954 | **LIVE** | port/src/dormouse/model/byteflow.py:208 | configs/flagship_base.toml, configs/m400b.toml |
| `ce_fixed` | TrainConfig | `bool` | `False` | port/src/dormouse/config.py:1741 | **LIVE** | port/src/dormouse/post/rl_train.py:591 |  |
| `ckpt_every` | TrainConfig | `int` | `100` | port/src/dormouse/config.py:1832 | **LIVE** | port/src/dormouse/post/rl_train.py:708 |  |
| `ckpt_name` | TrainConfig | `str` | `'port'` | port/src/dormouse/config.py:1833 | **LIVE** | port/src/dormouse/train/loop.py:170 |  |
| `coda_layers` | ModelConfig | `int` | `0` | port/src/dormouse/config.py:931 | **LIVE** | init-only: port/src/dormouse/model/model.py:119; port/src/dormouse/model/model.py:127; port/src/dormouse/model/model.py:284; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `csa_chunk` | ModelConfig | `int` | `8` | port/src/dormouse/config.py:1438 | **LIVE** | init-only: port/src/dormouse/model/csa.py:463; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `csa_coarse_chunk` | ModelConfig | `int` | `0` | port/src/dormouse/config.py:1452 | **LIVE** | init-only: port/src/dormouse/model/csa.py:475; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `csa_coarse_kv_dim` | ModelConfig | `int` | `64` | port/src/dormouse/config.py:1453 | **LIVE** | init-only: port/src/dormouse/model/csa.py:476; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `csa_fine` | ModelConfig | `bool` | `True` | port/src/dormouse/config.py:1457 | **LIVE** | init-only: port/src/dormouse/model/csa.py:467; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `csa_fused` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:1469 | **LIVE** | init-only: port/src/dormouse/model/csa.py:470; guarded | configs/m400b.toml |
| `csa_hierarchical` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:1494 | **LIVE** | init-only: port/src/dormouse/model/csa.py:478; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `csa_kv_dim` | ModelConfig | `int` | `64` | port/src/dormouse/config.py:1439 | **LIVE** | init-only: port/src/dormouse/model/csa.py:462; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `csa_mid_chunk` | ModelConfig | `int` | `0` | port/src/dormouse/config.py:1482 | **LIVE** | init-only: port/src/dormouse/model/csa.py:477; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `csa_topk` | ModelConfig | `int` | `32` | port/src/dormouse/config.py:1440 | **LIVE** | init-only: port/src/dormouse/model/csa.py:464; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `d_ffn` | ModelConfig | `int` | `2048` | port/src/dormouse/config.py:375 | **LIVE** | port/src/dormouse/cli.py:219; guarded | configs/audit_d128.toml, configs/b1.toml, configs/flagship3_dense.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/mirror_control.toml, configs/small.toml, configs/small_wide.toml, configs/tiny.toml |
| `d_model` | ModelConfig | `int` | `768` | port/src/dormouse/config.py:372 | **LIVE** | port/src/dormouse/cli.py:219; guarded | configs/audit_d128.toml, configs/b1.toml, configs/final_byte.toml, configs/final_patch.toml, configs/flagship3_dense.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/mirror_control.toml, configs/small.toml, configs/small_wide.toml, configs/tiny.toml |
| `data` | TrainConfig | `str` | `''` | port/src/dormouse/config.py:2173 | **LIVE** | port/src/dormouse/cli.py:624; guarded |  |
| `decay_frac` | TrainConfig | `float` | `0.1` | port/src/dormouse/config.py:2161 | **LIVE** | port/src/dormouse/train/loop.py:1399; guarded |  |
| `delta` | RLConfig | `float` | `0.0` | port/src/dormouse/post/rl.py:108 | **LIVE** | port/src/dormouse/post/rl.py:145; guarded |  |
| `dense_qat` | ModelConfig | `str` | `'none'` | port/src/dormouse/config.py:1266 | **LIVE** | port/src/dormouse/model/dspark.py:232; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `dense_qat_skip_attn` | ModelConfig | `bool` | `True` | port/src/dormouse/config.py:1277 | **LIVE** | port/src/dormouse/model/tsct.py:545 | configs/flagship_base.toml, configs/m400b.toml |
| `doc_masking` | TrainConfig | `bool` | `False` | port/src/dormouse/config.py:2214 | **LIVE** | port/src/dormouse/train/data.py:1002; guarded |  |
| `dspark_k` | ModelConfig | `int` | `4` | port/src/dormouse/config.py:1338 | **LIVE** | init-only: port/src/dormouse/model/dspark.py:203; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `dspark_markov_rank` | ModelConfig | `int` | `256` | port/src/dormouse/config.py:1340 | **LIVE** | init-only: port/src/dormouse/model/dspark.py:210 | configs/flagship_base.toml, configs/m400b.toml |
| `dspark_stride` | ModelConfig | `int` | `16` | port/src/dormouse/config.py:1339 | **LIVE** | init-only: port/src/dormouse/model/dspark.py:205; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `dspark_weight` | ModelConfig | `float` | `0.0` | port/src/dormouse/config.py:1337 | **LIVE** | port/src/dormouse/model/model.py:443; guarded | configs/b1.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/small.toml, configs/small_wide.toml |
| `ema_decay` | TrainConfig | `float` | `0.0` | port/src/dormouse/config.py:2149 | **LIVE** | port/src/dormouse/train/loop.py:1378; guarded |  |
| `energy_negs` | ModelConfig | `int` | `64` | port/src/dormouse/config.py:1420 | **LIVE** | port/src/dormouse/model/model.py:1137; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `energy_rank` | ModelConfig | `int` | `64` | port/src/dormouse/config.py:1424 | **LIVE** | init-only: port/src/dormouse/model/model.py:500; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `energy_temp` | ModelConfig | `float` | `1.0` | port/src/dormouse/config.py:1422 | **LIVE** | port/src/dormouse/model/model.py:1138; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `energy_weight` | ModelConfig | `float` | `0.0` | port/src/dormouse/config.py:1417 | **LIVE** | port/src/dormouse/model/model.py:485; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `engram_bias_correction` | TrainConfig | `bool` | `True` | port/src/dormouse/config.py:2241 | **LIVE** | port/src/dormouse/post/rl_train.py:408 |  |
| `engram_dim` | ModelConfig | `int` | `32` | port/src/dormouse/config.py:1539 | **LIVE** | init-only: port/src/dormouse/model/engram.py:280 | configs/b1.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/small.toml, configs/small_wide.toml, configs/tiny.toml |
| `engram_lam_max` | ModelConfig | `float` | `0.5` | port/src/dormouse/config.py:1541 | **LIVE** | init-only: port/src/dormouse/model/engram.py:282; port/src/dormouse/model/loopblock.py:282; guarded | configs/b1.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/small.toml, configs/small_wide.toml |
| `engram_orders` | ModelConfig | `list[int]` | `field(default_factory=lambda: [2, 3, 4])` | port/src/dormouse/config.py:1538 | **LIVE** | init-only: port/src/dormouse/model/engram.py:279; guarded | configs/b1.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/small.toml, configs/small_wide.toml |
| `engram_ram` | TrainConfig | `bool` | `False` | port/src/dormouse/config.py:2231 | **LIVE** | port/src/dormouse/post/rl_train.py:404; guarded |  |
| `engram_rows` | ModelConfig | `int` | `25000` | port/src/dormouse/config.py:1537 | **LIVE** | init-only: port/src/dormouse/model/engram.py:281 | configs/b1.toml, configs/final_byte.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/small.toml, configs/small_wide.toml, configs/tiny.toml |
| `eval` | TrainConfig | `bool` | `False` | port/src/dormouse/config.py:1834 | **LIVE** | port/src/dormouse/train/data.py:1027; guarded |  |
| `eval_batches` | TrainConfig | `int | None` | `None` | port/src/dormouse/config.py:1849 | **LIVE** | port/src/dormouse/train/data.py:847; guarded |  |
| `eval_bytes` | TrainConfig | `int` | `196608` | port/src/dormouse/config.py:1843 | **LIVE** | port/src/dormouse/train/data.py:855; guarded |  |
| `eval_data` | TrainConfig | `str` | `''` | port/src/dormouse/config.py:2174 | **LIVE** | port/src/dormouse/train/data.py:973 |  |
| `eval_depths` | TrainConfig | `bool` | `False` | port/src/dormouse/config.py:2257 | **LIVE** | port/src/dormouse/train/loop.py:1013 |  |
| `eval_every` | TrainConfig | `int` | `50` | port/src/dormouse/config.py:1835 | **LIVE** | port/src/dormouse/train/loop.py:2014 |  |
| `eval_offset` | TrainConfig | `int` | `0` | port/src/dormouse/config.py:1854 | **LIVE** | port/src/dormouse/train/data.py:977; guarded |  |
| `expert_act` | ModelConfig | `str` | `'silu'` | port/src/dormouse/config.py:1227 | **LIVE** | port/src/dormouse/model/experts.py:1120; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `expert_beta1` | ModelConfig | `float` | `4.0` | port/src/dormouse/config.py:1230 | **LIVE** | port/src/dormouse/model/experts.py:1129 | configs/flagship_base.toml, configs/m400b.toml |
| `expert_beta2` | ModelConfig | `float` | `25.0` | port/src/dormouse/config.py:1231 | **LIVE** | port/src/dormouse/model/experts.py:1130 | configs/flagship_base.toml, configs/m400b.toml |
| `factor_qat` | ModelConfig | `str` | `'none'` | port/src/dormouse/config.py:1240 | **LIVE** | port/src/dormouse/model/experts.py:1092; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `factor_qat_alpha` | ModelConfig | `float` | `1.0` | port/src/dormouse/config.py:1245 | **LIVE** | port/src/dormouse/model/dspark.py:233; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `factor_qat_anneal_steps` | TrainConfig | `int` | `0` | port/src/dormouse/config.py:1886 | **LIVE** | port/src/dormouse/train/loop.py:1434 |  |
| `factor_qat_seed` | ModelConfig | `int` | `0` | port/src/dormouse/config.py:1253 | **LIVE** | init-only: port/src/dormouse/model/model.py:186; port/src/dormouse/train/loop.py:1224; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `ffn_form` | ModelConfig | `str` | `'bank'` | port/src/dormouse/config.py:503 | **LIVE** | port/src/dormouse/cli.py:263; guarded |  |
| `ffn_rank_down` | ModelConfig | `int` | `0` | port/src/dormouse/config.py:511 | **LIVE** | port/src/dormouse/cli.py:235; guarded |  |
| `ffn_rank_gate` | ModelConfig | `int` | `0` | port/src/dormouse/config.py:508 | **LIVE** | port/src/dormouse/cli.py:234; guarded |  |
| `full_graph` | TrainConfig | `bool` | `False` | port/src/dormouse/config.py:2113 | **LIVE** | port/src/dormouse/train/loop.py:1276; guarded |  |
| `fused_epilogue` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:879 | **LIVE** | init-only: port/src/dormouse/model/loopblock.py:313; port/src/dormouse/model/loopblock.py:520; guarded | configs/m400b.toml |
| `fused_kda_tail` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:900 | **LIVE** | init-only: port/src/dormouse/model/kda.py:262; port/src/dormouse/model/loopblock.py:319; port/src/dormouse/model/loopblock.py:528; guarded | configs/m400b.toml |
| `fused_latent_proj` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:735 | **LIVE** | init-only: port/src/dormouse/model/kda.py:257; guarded | configs/m400b.toml |
| `fused_residual` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:869 | **LIVE** | init-only: port/src/dormouse/model/loopblock.py:307; port/src/dormouse/model/loopblock.py:512; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `gamma` | RLConfig | `float` | `1.0` | port/src/dormouse/post/rl.py:112 | **LIVE** | port/src/dormouse/post/rl.py:152; guarded |  |
| `gate` | RLConfig | `float` | `0.4` | port/src/dormouse/post/rl.py:106 | **LIVE** | port/src/dormouse/post/rl.py:143; guarded |  |
| `grad_accum` | TrainConfig | `int` | `1` | port/src/dormouse/config.py:1780 | **LIVE** | port/src/dormouse/post/rl_train.py:390; guarded |  |
| `grad_clip` | TrainConfig | `float` | `1.0` | port/src/dormouse/config.py:1710 | **LIVE** | port/src/dormouse/post/rl_train.py:618 |  |
| `halt_eps` | ModelConfig | `float` | `0.005` | port/src/dormouse/config.py:460 | **LIVE** | port/src/dormouse/model/model.py:1090; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `halt_max_iter` | ModelConfig | `int` | `8` | port/src/dormouse/config.py:468 | **LIVE** | port/src/dormouse/model/model.py:432; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `halt_min_iter` | ModelConfig | `int` | `1` | port/src/dormouse/config.py:464 | **LIVE** | port/src/dormouse/model/model.py:934; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `halt_mode` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:474 | **LIVE** | port/src/dormouse/model/model.py:425; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `halt_weight` | ModelConfig | `float` | `0.0` | port/src/dormouse/config.py:456 | **LIVE** | port/src/dormouse/model/model.py:408; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `head_dim` | ModelConfig | `int` | `64` | port/src/dormouse/config.py:374 | **LIVE** | init-only: port/src/dormouse/model/fullattn.py:94; port/src/dormouse/model/kda.py:209; port/src/dormouse/model/kda.py:211; guarded | configs/audit_d128.toml, configs/b1.toml, configs/flagship3_dense.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/mirror_control.toml, configs/small.toml, configs/small_wide.toml, configs/tiny.toml |
| `iter_checkpoint` | TrainConfig | `bool` | `False` | port/src/dormouse/config.py:2310 | **LIVE** | port/src/dormouse/train/loop.py:1183; guarded |  |
| `jepa` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:1387 | **DEAD** | port/src/dormouse/config.py:1558; port/src/dormouse/config.py:1560; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `jepa_horizon` | ModelConfig | `int` | `1` | port/src/dormouse/config.py:1395 | **LIVE** | init-only: port/src/dormouse/model/model.py:145; port/src/dormouse/model/model.py:147; port/src/dormouse/model/model.py:418; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `jepa_reg_weight` | ModelConfig | `float` | `0.1` | port/src/dormouse/config.py:1401 | **LIVE** | init-only: port/src/dormouse/model/model.py:419; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `jepa_weight` | ModelConfig | `float` | `0.0` | port/src/dormouse/config.py:1390 | **LIVE** | port/src/dormouse/model/model.py:138; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `json_log` | TrainConfig | `str` | `''` | port/src/dormouse/config.py:2172 | **LIVE** | port/src/dormouse/post/rl_train.py:494 |  |
| `kda_disable_recompute` | TrainConfig | `bool` | `False` | port/src/dormouse/config.py:1794 | **LIVE** | port/src/dormouse/train/loop.py:1124 |  |
| `layerrope` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:535 | **LIVE** | init-only: port/src/dormouse/model/model.py:67; port/src/dormouse/model/model.py:276; guarded | configs/audit_d128.toml, configs/flagship3_dense.toml, configs/flagship_base.toml, configs/m400b.toml, configs/mirror_control.toml |
| `layerrope_amp_init` | ModelConfig | `float` | `1.0` | port/src/dormouse/config.py:538 | **LIVE** | init-only: port/src/dormouse/model/model.py:304; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `layerrope_base` | ModelConfig | `float` | `100.0` | port/src/dormouse/config.py:567 | **LIVE** | init-only: port/src/dormouse/model/model.py:103; port/src/dormouse/model/model.py:105; port/src/dormouse/model/model.py:286; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `layerrope_beta_init` | ModelConfig | `float` | `-0.5` | port/src/dormouse/config.py:564 | **LIVE** | init-only: port/src/dormouse/model/model.py:287 | configs/flagship_base.toml, configs/m400b.toml |
| `layerrope_form` | ModelConfig | `str` | `'v1'` | port/src/dormouse/config.py:559 | **LIVE** | init-only: port/src/dormouse/model/model.py:84; port/src/dormouse/model/model.py:86; port/src/dormouse/model/model.py:90; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `layerrope_index` | ModelConfig | `str` | `'iteration'` | port/src/dormouse/config.py:573 | **LIVE** | init-only: port/src/dormouse/model/loopblock.py:354; port/src/dormouse/model/model.py:91; port/src/dormouse/model/model.py:93; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `layerrope_init` | ModelConfig | `str` | `'paper'` | port/src/dormouse/config.py:579 | **LIVE** | init-only: port/src/dormouse/model/model.py:97; port/src/dormouse/model/model.py:99; port/src/dormouse/model/model.py:288; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `layerrope_learn_theta` | ModelConfig | `bool` | `True` | port/src/dormouse/config.py:547 | **LIVE** | init-only: port/src/dormouse/model/model.py:306 | configs/flagship_base.toml, configs/m400b.toml |
| `layerrope_per_site` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:544 | **LIVE** | init-only: port/src/dormouse/model/model.py:303 | configs/flagship_base.toml, configs/m400b.toml |
| `layerrope_theta_init` | ModelConfig | `float` | `0.0` | port/src/dormouse/config.py:540 | **LIVE** | init-only: port/src/dormouse/model/model.py:305 | configs/flagship_base.toml, configs/m400b.toml |
| `length_penalty` | RLConfig | `float` | `0.1` | port/src/dormouse/post/rl.py:102 | **LIVE** | port/src/dormouse/post/rl.py:133; guarded |  |
| `log_every` | TrainConfig | `int` | `10` | port/src/dormouse/config.py:1831 | **LIVE** | port/src/dormouse/post/rl_train.py:653 |  |
| `logit_softcap` | ModelConfig | `float` | `0.0` | port/src/dormouse/config.py:1330 | **LIVE** | port/src/dormouse/model/loopblock.py:488; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `loom` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:914 | **LIVE** | init-only: port/src/dormouse/model/loopblock.py:482 | configs/b1.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/small.toml, configs/small_wide.toml |
| `loop_blocks` | ModelConfig | `int` | `1` | port/src/dormouse/config.py:448 | **LIVE** | init-only: port/src/dormouse/model/model.py:109; port/src/dormouse/model/model.py:113; port/src/dormouse/model/model.py:251; guarded | configs/audit_d128.toml, configs/flagship3_dense.toml, configs/flagship_base.toml, configs/m400b.toml, configs/mirror_control.toml |
| `lr` | TrainConfig | `float` | `0.0003` | port/src/dormouse/config.py:1708 | **LIVE** | port/src/dormouse/cli.py:424; guarded |  |
| `max_epochs` | TrainConfig | `float` | `4.0` | port/src/dormouse/config.py:2184 | **LIVE** | port/src/dormouse/train/loop.py:1157; guarded | configs/mix_big.toml, configs/mix_v2.toml, configs/mix_v3.toml, configs/mix_v4.toml, configs/mix_v7.toml, configs/mix_v7_docmask.toml |
| `max_iter` | ModelConfig | `int` | `4` | port/src/dormouse/config.py:378 | **LIVE** | port/src/dormouse/cli.py:241; guarded | configs/audit_d128.toml, configs/b1.toml, configs/final_byte.toml, configs/final_patch.toml, configs/flagship3_dense.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/mirror_control.toml, configs/small.toml, configs/small_wide.toml, configs/tiny.toml |
| `max_seq_len` | ModelConfig | `int` | `512` | port/src/dormouse/config.py:377 | **LIVE** | port/src/dormouse/cli.py:933; guarded | configs/audit_d128.toml, configs/b1.toml, configs/final_byte.toml, configs/final_patch.toml, configs/flagship3_dense.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/mirror_control.toml, configs/small.toml, configs/small_wide.toml, configs/tiny.toml |
| `memglu_c0` | ModelConfig | `float` | `0.6039096539628113` | port/src/dormouse/config.py:1280 | **LIVE** | port/src/dormouse/model/experts.py:1131; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `mhc_sinkhorn` | ModelConfig | `str` | `'logsumexp'` | port/src/dormouse/config.py:860 | **LIVE** | init-only: port/src/dormouse/model/loopblock.py:501; guarded |  |
| `mhc_streams` | ModelConfig | `int` | `2` | port/src/dormouse/config.py:850 | **LIVE** | init-only: port/src/dormouse/model/loopblock.py:501; guarded | configs/final_byte.toml, configs/flagship_base.toml, configs/m400b.toml |
| `min_lr_frac` | TrainConfig | `float` | `0.1` | port/src/dormouse/config.py:2167 | **LIVE** | port/src/dormouse/train/loop.py:1722; guarded |  |
| `min_replays` | RLConfig | `int` | `1` | port/src/dormouse/post/rl.py:119 | **LIVE** | init-only: port/src/dormouse/post/agent_rl.py:90; port/src/dormouse/post/agent_rl.py:94; port/src/dormouse/post/rl.py:154; guarded |  |
| `mix_plan` | TrainConfig | `str` | `''` | port/src/dormouse/config.py:2180 | **LIVE** | port/src/dormouse/train/loop.py:1143; guarded |  |
| `mm_dtype` | TrainConfig | `str` | `'bf16'` | port/src/dormouse/config.py:1825 | **LIVE** | port/src/dormouse/train/loop.py:1123; guarded |  |
| `mod_aux_weight` | ModelConfig | `float` | `0.0` | port/src/dormouse/config.py:420 | **LIVE** | port/src/dormouse/model/model.py:1244; guarded |  |
| `mod_capacity` | ModelConfig | `float` | `1.0` | port/src/dormouse/config.py:409 | **LIVE** | init-only: port/src/dormouse/model/transformer.py:174; guarded |  |
| `mod_top_k` | ModelConfig | `int` | `0` | port/src/dormouse/config.py:414 | **LIVE** | init-only: port/src/dormouse/model/transformer.py:175; guarded |  |
| `moe_aux_coef` | ModelConfig | `float` | `0.01` | port/src/dormouse/config.py:1069 | **LIVE** | init-only: port/src/dormouse/model/experts.py:1417; guarded | configs/m400b.toml |
| `moe_bank_epoch` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:1181 | **LIVE** | init-only: port/src/dormouse/model/experts.py:1030; port/src/dormouse/model/experts.py:1077 | configs/m400b.toml |
| `moe_capacity` | ModelConfig | `float` | `1.0` | port/src/dormouse/config.py:1024 | **LIVE** | port/src/dormouse/model/experts.py:1008; guarded | configs/b1.toml, configs/final_byte.toml, configs/final_patch.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/small.toml, configs/small_wide.toml |
| `moe_combine` | ModelConfig | `str` | `'torch'` | port/src/dormouse/config.py:1168 | **LIVE** | init-only: port/src/dormouse/model/experts.py:1031; port/src/dormouse/model/experts.py:1067; guarded | configs/m400b.toml |
| `moe_combine_dtype` | ModelConfig | `str` | `'fp32'` | port/src/dormouse/config.py:1174 | **LIVE** | init-only: port/src/dormouse/model/experts.py:1032; port/src/dormouse/model/experts.py:1068; guarded | configs/m400b.toml |
| `moe_entropy_hi` | ModelConfig | `float` | `0.6` | port/src/dormouse/config.py:1003 | **LIVE** | init-only: port/src/dormouse/model/experts.py:994; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `moe_entropy_lo` | ModelConfig | `float` | `0.2` | port/src/dormouse/config.py:1002 | **LIVE** | init-only: port/src/dormouse/model/experts.py:993; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `moe_gather` | ModelConfig | `str` | `'off'` | port/src/dormouse/config.py:1140 | **LIVE** | init-only: port/src/dormouse/model/experts.py:1051; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `moe_kernel` | ModelConfig | `str` | `'padded'` | port/src/dormouse/config.py:1098 | **LIVE** | init-only: port/src/dormouse/model/experts.py:1034; port/src/dormouse/model/experts.py:1037; guarded | configs/b1.toml, configs/flagship_base.toml, configs/m400b.toml |
| `moe_lb_clamp` | ModelConfig | `float` | `2.0` | port/src/dormouse/config.py:1062 | **LIVE** | init-only: port/src/dormouse/model/experts.py:1416; guarded | configs/m400b.toml |
| `moe_lb_coef` | ModelConfig | `float` | `0.0` | port/src/dormouse/config.py:1032 | **LIVE** | port/src/dormouse/model/experts.py:1203; guarded | configs/b1.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/small.toml, configs/small_wide.toml |
| `moe_lb_gamma` | ModelConfig | `float` | `0.001` | port/src/dormouse/config.py:1054 | **LIVE** | init-only: port/src/dormouse/model/experts.py:1203; port/src/dormouse/model/experts.py:1410; guarded | configs/m400b.toml |
| `moe_lb_mode` | ModelConfig | `str` | `'sign'` | port/src/dormouse/config.py:1048 | **LIVE** | port/src/dormouse/model/experts.py:1197; guarded | configs/m400b.toml |
| `moe_offload` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:1291 | **LIVE** | init-only: port/src/dormouse/model/experts.py:1052 | configs/b1.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/small.toml, configs/small_wide.toml |
| `moe_permute` | ModelConfig | `str` | `'argsort'` | port/src/dormouse/config.py:1161 | **LIVE** | init-only: port/src/dormouse/model/experts.py:1033; port/src/dormouse/model/experts.py:1063; guarded | configs/m400b.toml |
| `moe_route_fused` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:1154 | **LIVE** | init-only: port/src/dormouse/model/experts.py:1059; guarded | configs/m400b.toml |
| `moe_router_jitter` | ModelConfig | `float` | `0.0` | port/src/dormouse/config.py:1023 | **LIVE** | init-only: port/src/dormouse/model/experts.py:1036 | configs/m400b.toml |
| `moe_routing_health` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:1083 | **LIVE** | init-only: port/src/dormouse/model/experts.py:1420 | configs/m400b.toml |
| `moe_shared_compute` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:1126 | **LIVE** | init-only: port/src/dormouse/model/experts.py:1043; port/src/dormouse/model/experts.py:1183; guarded | configs/m400b.toml |
| `moe_topk` | ModelConfig | `int` | `0` | port/src/dormouse/config.py:987 | **LIVE** | port/src/dormouse/cli.py:228; guarded | configs/audit_d128.toml, configs/b1.toml, configs/final_byte.toml, configs/final_patch.toml, configs/flagship3_dense.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/mirror_control.toml, configs/small.toml, configs/small_wide.toml, configs/tiny.toml |
| `moe_topk_max` | ModelConfig | `int` | `4` | port/src/dormouse/config.py:998 | **LIVE** | init-only: port/src/dormouse/model/experts.py:992; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `moe_topk_min` | ModelConfig | `int` | `1` | port/src/dormouse/config.py:997 | **LIVE** | init-only: port/src/dormouse/model/experts.py:991; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `moe_topk_mode` | ModelConfig | `str` | `'off'` | port/src/dormouse/config.py:993 | **LIVE** | init-only: port/src/dormouse/model/experts.py:990; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `msa_block` | ModelConfig | `int` | `4` | port/src/dormouse/config.py:1508 | **LIVE** | init-only: port/src/dormouse/model/sparse/attention.py:155; port/src/dormouse/model/sparse/sparse.py:71 | configs/b1.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml |
| `msa_capacity_ratio` | ModelConfig | `float` | `0.25` | port/src/dormouse/config.py:1509 | **LIVE** | init-only: port/src/dormouse/model/loopblock.py:405 | configs/b1.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml |
| `msa_topk` | ModelConfig | `int` | `32` | port/src/dormouse/config.py:1507 | **LIVE** | init-only: port/src/dormouse/model/sparse/attention.py:154; port/src/dormouse/model/sparse/sparse.py:72 | configs/b1.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml |
| `mtp_head_norm` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:1374 | **LIVE** | init-only: port/src/dormouse/model/model.py:458; port/src/dormouse/model/model.py:474; guarded | configs/m400b.toml |
| `mtp_k` | ModelConfig | `int` | `0` | port/src/dormouse/config.py:1351 | **LIVE** | init-only: port/src/dormouse/model/model.py:472; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `mtp_soft` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:1364 | **LIVE** | port/src/dormouse/model/model.py:449; guarded | configs/m400b.toml |
| `mtp_weight` | ModelConfig | `float` | `0.0` | port/src/dormouse/config.py:1352 | **LIVE** | port/src/dormouse/model/model.py:174; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `muon_chunk_elems` | TrainConfig | `int` | `1 << 25` | port/src/dormouse/config.py:1944 | **LIVE** | port/src/dormouse/post/rl_train.py:437; guarded |  |
| `muon_colrow_3d` | TrainConfig | `str` | `'colrow'` | port/src/dormouse/config.py:1915 | **LIVE** | port/src/dormouse/post/rl_train.py:427; guarded |  |
| `muon_lr_mult` | TrainConfig | `float` | `1.0` | port/src/dormouse/config.py:1953 | **LIVE** | port/src/dormouse/post/rl_train.py:438 |  |
| `muon_sf` | TrainConfig | `bool` | `False` | port/src/dormouse/config.py:2022 | **LIVE** | port/src/dormouse/post/rl_train.py:435; guarded |  |
| `muon_sf_retract` | TrainConfig | `bool` | `False` | port/src/dormouse/config.py:2046 | **DEAD** | port/src/dormouse/config.py:4969; guarded |  |
| `muon_sf_wd` | TrainConfig | `float` | `0.01` | port/src/dormouse/config.py:2033 | **LIVE** | port/src/dormouse/post/rl_train.py:436; guarded |  |
| `muon_state_dtype` | TrainConfig | `str` | `'fp32'` | port/src/dormouse/config.py:1933 | **LIVE** | port/src/dormouse/post/rl_train.py:432 |  |
| `muon_variant` | TrainConfig | `str` | `'muon'` | port/src/dormouse/config.py:1987 | **LIVE** | port/src/dormouse/post/rl_train.py:434; guarded |  |
| `n_experts` | ModelConfig | `int` | `3` | port/src/dormouse/config.py:982 | **LIVE** | port/src/dormouse/cli.py:227; guarded | configs/audit_d128.toml, configs/b1.toml, configs/final_byte.toml, configs/final_patch.toml, configs/flagship3_dense.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/mirror_control.toml, configs/small.toml, configs/small_wide.toml, configs/tiny.toml |
| `n_heads` | ModelConfig | `int` | `12` | port/src/dormouse/config.py:373 | **LIVE** | port/src/dormouse/cli.py:242; guarded | configs/audit_d128.toml, configs/b1.toml, configs/final_byte.toml, configs/final_patch.toml, configs/flagship3_dense.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/mirror_control.toml, configs/small.toml, configs/small_wide.toml, configs/tiny.toml |
| `n_kv_heads` | ModelConfig | `int` | `0` | port/src/dormouse/config.py:724 | **LIVE** | port/src/dormouse/cli.py:244; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `n_layers` | ModelConfig | `int` | `0` | port/src/dormouse/config.py:402 | **LIVE** | init-only: port/src/dormouse/model/model.py:251; guarded | configs/audit_d128.toml, configs/flagship3_dense.toml, configs/mirror_control.toml |
| `norm_eps` | ModelConfig | `float` | `0.001` | port/src/dormouse/config.py:512 | **LIVE** | init-only: port/src/dormouse/model/loopblock.py:281; port/src/dormouse/model/loopblock.py:360; port/src/dormouse/model/loopblock.py:363 | configs/b1.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/small.toml, configs/small_wide.toml |
| `ns_dtype` | TrainConfig | `str` | `'fp32'` | port/src/dormouse/config.py:1960 | **LIVE** | port/src/dormouse/post/rl_train.py:433 |  |
| `ns_kind` | TrainConfig | `str` | `'classic'` | port/src/dormouse/config.py:1897 | **LIVE** | port/src/dormouse/post/rl_train.py:426; guarded |  |
| `ns_steps` | TrainConfig | `int` | `5` | port/src/dormouse/config.py:1870 | **LIVE** | port/src/dormouse/post/rl_train.py:425; guarded |  |
| `opt` | TrainConfig | `str` | `'mix'` | port/src/dormouse/config.py:1857 | **LIVE** | port/src/dormouse/cli.py:424; guarded |  |
| `opt_graph` | TrainConfig | `bool` | `False` | port/src/dormouse/config.py:2077 | **LIVE** | port/src/dormouse/train/loop.py:1276; guarded |  |
| `optim_8bit` | TrainConfig | `bool` | `False` | port/src/dormouse/config.py:1830 | **LIVE** | port/src/dormouse/post/rl_train.py:440 |  |
| `out_dir` | TrainConfig | `str` | `'checkpoints'` | port/src/dormouse/config.py:2215 | **LIVE** | port/src/dormouse/train/loop.py:169 |  |
| `out_proj_rank` | ModelConfig | `int` | `0` | port/src/dormouse/config.py:746 | **LIVE** | port/src/dormouse/cli.py:220; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `prelude_layers` | ModelConfig | `int` | `0` | port/src/dormouse/config.py:928 | **LIVE** | init-only: port/src/dormouse/model/model.py:119; port/src/dormouse/model/model.py:127; port/src/dormouse/model/model.py:284; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `quality` | RLConfig | `str` | `'thrift'` | port/src/dormouse/post/rl.py:98 | **LIVE** | port/src/dormouse/post/rl.py:122; guarded |  |
| `quant` | ModelConfig | `str` | `'none'` | port/src/dormouse/config.py:616 | **LIVE** | port/src/dormouse/train/loop.py:442; guarded | configs/b1.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/small.toml, configs/small_wide.toml |
| `quantile` | RLConfig | `float` | `0.5` | port/src/dormouse/post/rl.py:104 | **LIVE** | port/src/dormouse/post/rl.py:138; guarded |  |
| `rand_depth` | TrainConfig | `bool` | `False` | port/src/dormouse/config.py:2251 | **LIVE** | port/src/dormouse/train/loop.py:1013; guarded |  |
| `rank` | ModelConfig | `int` | `64` | port/src/dormouse/config.py:475 | **LIVE** | port/src/dormouse/cli.py:220; guarded | configs/b1.toml, configs/final_byte.toml, configs/final_patch.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/small.toml, configs/small_wide.toml, configs/tiny.toml |
| `residual_dtype` | ModelConfig | `str` | `'fp32'` | port/src/dormouse/config.py:613 | **LIVE** | init-only: port/src/dormouse/model/loopblock.py:295; guarded |  |
| `rl_adv` | TrainConfig | `str` | `'crr'` | port/src/dormouse/config.py:2371 | **LIVE** | port/src/dormouse/cli.py:744; guarded |  |
| `rl_agg` | TrainConfig | `str` | `'prompt-mean'` | port/src/dormouse/config.py:2374 | **LIVE** | bridge->RLConfig.agg |  |
| `rl_band_attempts` | TrainConfig | `int` | `4` | port/src/dormouse/config.py:2406 | **LIVE** | port/src/dormouse/post/rl_train.py:246; guarded |  |
| `rl_band_hi` | TrainConfig | `float` | `0.8` | port/src/dormouse/config.py:2403 | **LIVE** | port/src/dormouse/post/rl_train.py:264 |  |
| `rl_band_lo` | TrainConfig | `float` | `0.2` | port/src/dormouse/config.py:2402 | **LIVE** | port/src/dormouse/post/rl_train.py:264 |  |
| `rl_budget` | TrainConfig | `int` | `256` | port/src/dormouse/config.py:2389 | **LIVE** | port/src/dormouse/cli.py:741; guarded |  |
| `rl_corpus` | TrainConfig | `str` | `''` | port/src/dormouse/config.py:2392 | **LIVE** | port/src/dormouse/post/rl_train.py:324 |  |
| `rl_eval_every` | TrainConfig | `int` | `50` | port/src/dormouse/config.py:2396 | **LIVE** | port/src/dormouse/post/rl_train.py:705; guarded |  |
| `rl_eval_samples` | TrainConfig | `int` | `2` | port/src/dormouse/config.py:2399 | **LIVE** | port/src/dormouse/post/rl_train.py:302; guarded |  |
| `rl_group` | TrainConfig | `int` | `4` | port/src/dormouse/config.py:2386 | **LIVE** | port/src/dormouse/cli.py:740; guarded |  |
| `rl_holdout` | TrainConfig | `int` | `32` | port/src/dormouse/config.py:2394 | **LIVE** | port/src/dormouse/post/rl_train.py:472; guarded |  |
| `rl_len_delta` | TrainConfig | `float` | `0.0` | port/src/dormouse/config.py:2363 | **LIVE** | bridge->RLConfig.delta |  |
| `rl_len_gamma` | TrainConfig | `float` | `1.0` | port/src/dormouse/config.py:2367 | **LIVE** | bridge->RLConfig.gamma |  |
| `rl_len_gate` | TrainConfig | `float` | `0.4` | port/src/dormouse/config.py:2361 | **LIVE** | bridge->RLConfig.gate |  |
| `rl_len_penalty` | TrainConfig | `float` | `0.1` | port/src/dormouse/config.py:2357 | **LIVE** | port/src/dormouse/post/rl_train.py:521; bridge->RLConfig.length_penalty |  |
| `rl_len_quantile` | TrainConfig | `float` | `0.5` | port/src/dormouse/config.py:2359 | **LIVE** | bridge->RLConfig.quantile |  |
| `rl_len_s` | TrainConfig | `float` | `0.5` | port/src/dormouse/config.py:2365 | **LIVE** | bridge->RLConfig.s |  |
| `rl_quality` | TrainConfig | `str` | `'thrift'` | port/src/dormouse/config.py:2350 | **LIVE** | bridge->RLConfig.quality |  |
| `rl_task_cls` | TrainConfig | `str` | `'code'` | port/src/dormouse/config.py:2377 | **LIVE** | port/src/dormouse/cli.py:739 |  |
| `rl_tasks` | TrainConfig | `int` | `128` | port/src/dormouse/config.py:2383 | **LIVE** | port/src/dormouse/cli.py:739; guarded |  |
| `s` | RLConfig | `float` | `0.5` | port/src/dormouse/post/rl.py:110 | **LIVE** | port/src/dormouse/post/rl.py:147; guarded |  |
| `sdpa_mem_efficient` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:1505 | **LIVE** | init-only: port/src/dormouse/model/fullattn.py:101 | configs/flagship_base.toml, configs/m400b.toml |
| `seed` | TrainConfig | `int` | `1` | port/src/dormouse/config.py:2150 | **LIVE** | port/src/dormouse/cli.py:188; guarded | configs/mix_big.toml, configs/mix_v2.toml, configs/mix_v3.toml, configs/mix_v4.toml, configs/mix_v7.toml, configs/mix_v7_docmask.toml |
| `segment_len` | TrainConfig | `int` | `0` | port/src/dormouse/config.py:2289 | **LIVE** | port/src/dormouse/train/loop.py:1795; guarded |  |
| `seq_len` | TrainConfig | `int` | `512` | port/src/dormouse/config.py:1707 | **LIVE** | port/src/dormouse/cli.py:423; guarded |  |
| `share_u` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:770 | **LIVE** | alias->share_u_mode; guarded | configs/final_byte.toml, configs/final_patch.toml, configs/flagship_base.toml, configs/m400b.toml |
| `share_u_mode` | ModelConfig | `str` | `'off'` | port/src/dormouse/config.py:790 | **LIVE** | init-only: port/src/dormouse/model/experts.py:1109; port/src/dormouse/model/experts.py:1140; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `share_u_rank` | ModelConfig | `int` | `64` | port/src/dormouse/config.py:794 | **LIVE** | init-only: port/src/dormouse/model/experts.py:1279; port/src/dormouse/model/experts.py:1281; port/src/dormouse/model/experts.py:1285; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `shared_rank` | ModelConfig | `int` | `192` | port/src/dormouse/config.py:1293 | **LIVE** | port/src/dormouse/cli.py:237; guarded | configs/b1.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/small.toml, configs/small_wide.toml, configs/tiny.toml |
| `sigreg_weight` | ModelConfig | `float` | `0.0` | port/src/dormouse/config.py:1382 | **LIVE** | port/src/dormouse/model/model.py:822; guarded | configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/small.toml, configs/small_wide.toml |
| `steps` | TrainConfig | `int` | `200` | port/src/dormouse/config.py:1705 | **LIVE** | port/src/dormouse/cli.py:423; guarded |  |
| `tf32` | TrainConfig | `bool` | `False` | port/src/dormouse/config.py:1787 | **LIVE** | port/src/dormouse/post/rl_train.py:383 |  |
| `tie_embeddings` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:697 | **LIVE** | init-only: port/src/dormouse/model/model.py:155; port/src/dormouse/model/model.py:259; port/src/dormouse/model/model.py:392; guarded | configs/flagship_base.toml, configs/m400b.toml |
| `timers` | TrainConfig | `bool` | `False` | port/src/dormouse/config.py:2216 | **LIVE** | port/src/dormouse/train/loop.py:1976 |  |
| `torch_compile` | TrainConfig | `bool` | `False` | port/src/dormouse/config.py:1878 | **LIVE** | port/src/dormouse/train/loop.py:1400; guarded |  |
| `tsct_expert_basis` | ModelConfig | `int` | `0` | port/src/dormouse/config.py:810 | **LIVE** | init-only: port/src/dormouse/model/experts.py:1153; guarded |  |
| `tsct_expert_lr` | ModelConfig | `float` | `0.0` | port/src/dormouse/config.py:820 | **LIVE** | port/src/dormouse/model/experts.py:1154; guarded |  |
| `tsct_expert_revive` | ModelConfig | `int` | `0` | port/src/dormouse/config.py:825 | **LIVE** | port/src/dormouse/model/experts.py:1155; guarded |  |
| `tsct_packed` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:1217 | **LIVE** | init-only: port/src/dormouse/model/experts.py:1086; port/src/dormouse/model/experts.py:1229; guarded |  |
| `tsct_revive_noise` | ModelConfig | `float` | `0.01` | port/src/dormouse/config.py:828 | **LIVE** | init-only: port/src/dormouse/model/experts.py:1156; guarded |  |
| `use_attnres` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:837 | **LIVE** | port/src/dormouse/model/byteflow.py:827; guarded | configs/audit_d128.toml, configs/b1.toml, configs/final_byte.toml, configs/final_patch.toml, configs/flagship3_dense.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/mirror_control.toml, configs/small.toml, configs/small_wide.toml |
| `use_byteflow` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:915 | **LIVE** | port/src/dormouse/model/byteflow.py:826; guarded | configs/b1.toml, configs/final_byte.toml, configs/final_patch.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/small.toml, configs/small_wide.toml |
| `use_csa` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:1437 | **LIVE** | init-only: port/src/dormouse/model/loopblock.py:417; guarded | configs/audit_d128.toml, configs/flagship3_dense.toml, configs/flagship_base.toml, configs/m400b.toml, configs/mirror_control.toml |
| `use_engram` | ModelConfig | `bool` | `True` | port/src/dormouse/config.py:829 | **LIVE** | port/src/dormouse/model/byteflow.py:827; guarded | configs/audit_d128.toml, configs/b1.toml, configs/final_byte.toml, configs/final_patch.toml, configs/flagship3_dense.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/mirror_control.toml, configs/small.toml, configs/small_wide.toml |
| `use_kda` | ModelConfig | `bool` | `True` | port/src/dormouse/config.py:623 | **LIVE** | port/src/dormouse/cli.py:904; guarded | configs/audit_d128.toml, configs/b1.toml, configs/flagship3_dense.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/mirror_control.toml, configs/small.toml, configs/small_wide.toml |
| `use_mhc` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:847 | **LIVE** | init-only: port/src/dormouse/model/loopblock.py:502; guarded | configs/audit_d128.toml, configs/final_byte.toml, configs/final_patch.toml, configs/flagship3_dense.toml, configs/flagship_base.toml, configs/m400b.toml, configs/mirror_control.toml |
| `use_msa` | ModelConfig | `bool` | `False` | port/src/dormouse/config.py:1506 | **LIVE** | init-only: port/src/dormouse/model/loopblock.py:404; port/src/dormouse/model/sparse/attention.py:150; guarded | configs/b1.toml, configs/final_byte.toml, configs/final_patch.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml |
| `use_tsct` | ModelConfig | `bool` | `True` | port/src/dormouse/config.py:679 | **LIVE** | port/src/dormouse/cli.py:220; guarded | configs/audit_d128.toml, configs/b1.toml, configs/flagship3_dense.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/mirror_control.toml, configs/small.toml, configs/small_wide.toml |
| `vocab` | ModelConfig | `int` | `256` | port/src/dormouse/config.py:376 | **LIVE** | port/src/dormouse/model/decode.py:456; guarded | configs/audit_d128.toml, configs/b1.toml, configs/flagship3_dense.toml, configs/flagship_base.toml, configs/m400.toml, configs/m400b.toml, configs/medium.toml, configs/mirror_control.toml, configs/small.toml, configs/small_wide.toml, configs/tiny.toml |
| `warmup_tokens` | TrainConfig | `int` | `0` | port/src/dormouse/config.py:2153 | **LIVE** | port/src/dormouse/train/loop.py:1398 |  |
| `wd` | TrainConfig | `float` | `0.01` | port/src/dormouse/config.py:1709 | **LIVE** | port/src/dormouse/post/rl_train.py:423; guarded |  |
| `z_loss` | ModelConfig | `float` | `0.0` | port/src/dormouse/config.py:1310 | **LIVE** | port/src/dormouse/model/model.py:161; guarded | configs/flagship_base.toml, configs/m400b.toml |
