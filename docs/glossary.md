# dormouse glossary

Every word below is used by the code, the ADRs or the docs, and means the thing
written here. Each entry has four parts:

- **the name** as it appears in the code or on a command line,
- **the definition** in one sentence,
- **where it lives** — the file, so a disagreement can be settled by reading the
  code rather than by asking,
- **what it is NOT** — the neighbouring word it is constantly confused with.
  This column is the point of the file.

A glossary that disagrees with the code is worse than no glossary, so every
entry was written from the code at commit `989e840`. Where a document and the
code disagree, the disagreement is listed in
[Code vs document](#code-vs-document-the-disagreements) at the bottom of this
file, not smoothed over here.

`CONTEXT.md` is the ten-minute model of the system and points here for the long
form. `AGENTS.md` is the working rules.

---

## 1. The compute spine

### LoopBlock

The weight-shared recurrent block: one `LoopBlock` runs `max_iter` times per
sequence, and the state it produces is fed back into itself, so iteration *n+1*
reads iteration *n*'s post-residual state.

`crates/dormouse-core/src/loop_block.rs:84`; the recurrence is the end of the
iteration body (`loop_block.rs:511-521`).

**Not** a layer stack — there is exactly one, shared by every iteration, so
"depth" is iterations rather than parameters. **Not** a "recursion unit"
(nothing here recurses in the program sense; the loop is a `for`). **Not**
"the trunk", and **not** "backbone" (which is a size, not an object — see
*Backbone* below).

### iteration

One pass of the loop over the sequence. It is counted by `max_iter` and is the
unit the readout, the per-iteration CE and the MoR router score are computed
per.

`loop_block.rs:302` (`for iter in 0..iters`), counted by `probe::ITER`
(`crates/dormouse-core/src/probe.rs:13`).

**Not** a training step (a step is many iterations; a step is what the optimizer
updates on). **Not** "a cycle", **not** a "ponder step" — PonderNet's
terminology was deleted with PonderNet (ADR-0013). **Not** a "recursion", which
is the MoR paper's word for the same thing (see below).

### depth

How many iterations actually run. It is `max_iter` in the default
configuration, a sampled `T` under `--rand-depth`, and a per-position top-k
count under MoR.

`loop_block.rs:281-291` resolves it once per forward into `iters`;
`LoopBlock::set_depth` (`:579`) is the only setter and it *asserts* the range
rather than clamping.

**Not** the tensor rank `D` of a tensor (`Tensor<3>` is rank 3; "depth 3" is
never said). **Not** a hyperparameter you may clamp: an out-of-range depth is a
loud panic, because a silently clamped depth makes an A/B lie about what it
trained.

### max_iter

The config key for the full-depth iteration count. It is the *depth*, spelled
with the older name; both names are live in the same crate.

`crates/dormouse-core/src/config/schema.rs:93` (default 4, pinned by
`configs/small.toml`); the runtime copy is `LoopBlock::max_iter`
(`loop_block.rs:106`).

**Not** "recursions" (that is MoR's word for a *slot*, below), **not**
"iterations drawn" under `--rand-depth` (that is `sample_depth`,
`crates/dormouse-train/src/lib.rs:325`), **not** a depth schedule — there is no
schedule; `--rand-depth` draws a constant `T` per *step*.

### iteration slot (a.k.a. "recursion")

One of the `max_iter` positions in the loop, as a *candidate*. Under MoR each
position is scored by a router and the top `mor_k` reach the readout and the CE.

`crates/dormouse-core/src/mor.rs:47` (`route`), the slot scores are collected at
`loop_block.rs:491-497`.

**Not** an iteration that ran (all `iters` slots always run; the gate picks
which reach the readout). **Not** a MoE expert — the experts are the FFNs inside
one iteration, the slots are the iterations.

### step (training step)

One optimizer update: one forward, one backward, one `optim.step`, one retraction
cadence check. The unit the loss curve, `ce=`, `best`, the stress protocol and
the checkpoint cadence are all counted in.

`crates/dormouse-train/src/lib.rs:991` (`while step < cfg.steps`).

**Not** an iteration, **not** a batch (a step consumes `batch` windows of
`seq_len` bytes each), **not** a "PonderNet step".

### controller

The `Linear` that reads `[h_ctx, h0]` and produces the three sigmoid gates
(`w_attn`, `w_mem`, `w_ffn`) plus the softmax expert blend. It is the model's
only router of the loop's *arms*.

`loop_block.rs:190-193` (construction), `loop_block.rs:358-371` (the read).

**Not** a router in the MoR sense (that is `mor_router`, which ranks *slots*,
not arms). **Not** the MoE gate — the expert blend is this same projection, and
the MoE gate is the softmax over `n_experts` of it.

### arm

An optional mechanism inside the model, gated by a config flag: KDA, Engram, GR,
MoR, activation quant, and each aux objective. "Disable an arm for A/B"
(`crates/dormouse-cli/src/bin/train.rs:76`) is the ordinary phrase.

`probe::note` counts entries per arm (`crates/dormouse-core/src/probe.rs:13-33`,
10 counters); `use_kda` / `use_engram` / `use_gr` / `use_mor` / `act_quant` are
the switches.

**Not** an arm of an A/B experiment — that is a *run configuration*, and
`docs/protocols/AB-PROTOCOL.md` calls those arms too. **Not** the Fused/Fallback pair a
library seam reports (ADR-0019 "which arm ran" is that third meaning). The
three collide constantly; when a doc says "arm", say which one.

### shared_attn / the attention arm

The one attention mechanism: KDA gated-delta, wrapped in `AdaptiveAttention` so
the checkpoint parameter prefix `loop_block.shared_attn.gdn2.*` survives the cut
of every other attention arm.

`crates/dormouse-core/src/attention.rs:1-6` and `:30`.

**Not** "the attention layer" (there is no layer; it is re-run per iteration and
its recurrent `kda` state is threaded through iterations by
`forward_full_state`). **Not** "shared weights" — `shared` here means *one
instance serving every iteration*, which is also true of the experts and the
readout.

### RoPE on the attention arm (NoPE by default)

Rotary position embedding on the KDA q/k, `use_rope` in `burn-kda`
(`KdaConfig::use_rope`, applied in `KdaModule::project` before the q/k L2 norm),
**off by default and adding zero parameters**. `rope_theta` is the constant
`10000.0`, upstream's own default.

**Off by default because FLA's official KDA layer has no RoPE at all.**
`fla/layers/kda.py` @ `9f38d249` contains zero `rotary`/`rope` occurrences, and
GatedDeltaNet has no `use_rope` either — so this arm is a **cross-family
transplant**, not a reproduction, and its justification is post-training (the
Qwen3.8 playbook: NoPE breaks SFT/RLVR), not pretrain parity. In Kimi Linear's
hybrid the position comes from the interleaved full-attention layers
(`fla/layers/attn.py:83,125`; `fla/models/hybrid.py:17-23`). Evidence, line
numbers and the placement proof: `docs/reviews/rope-2026-09-30.md`;
A/B row 8 in `docs/protocols/AB-PROTOCOL.md`, **not run**.

**Not** "positional encoding" in general — there is no learned or additive
position embedding on this arm, and adding one is a different mechanism with a
parameter cost. **Not** "the KDA has positions": with the flag off it has none,
and the byte position reaches it only through the sequence order the recurrence
sees.

### readout (out_proj)

The per-iteration projection whose output is averaged into the model's hidden
state. It is the only thing that reads the loop state; the final `norm` and
`lm_head` sit after the *mean*, not after the last iteration.

`loop_block.rs:99` (the field), `loop_block.rs:490-491` (the read),
`loop_block.rs:544-560` (the masked mean).

**Not** the `lm_head` (that is a separate projection to vocab, on the mean), and
**not** a residual — the residual is added before the readout.

### ReZero / residual_scale

The scalar-per-block residual coefficient, initialized to **1** (identity), which
is what gives the arms a gradient at step 0.

`loop_block.rs:98` and `:475-478`, initialization at `:214`.

**Not** "the residual connection" (the add is unconditional; the scale is what is
learned), and **not** `use_gr` — GR is the four-branch *replacement* for the
pre-norm + ReZero pair, off by default.

### Gated Residual (GR)

Four-branch residual: a normalized gated read over per-branch RMSNorms plus a
per-branch scalar write, replacing pre-norm + ReZero when `use_gr` is set. Our
`read` is Eq. 31-32 and our `write` is Eq. 33-34 of the Qwen3.8-Flash-Next
report, including the `1/nr` inside the SiLU and the `2 sigma` on the write
scaler; `gr.rs`'s tests compute both on the host from the module's own weights
and compare.

The one thing that is OURS: the report puts a GR module on the attention block
and the MLP block of every layer, we run ONE weight-shared block recursively, so
the module is read and written once per loop ITERATION and the loop's
iteration embedding is added to the read. A transposition, not the report's
placement, and the report's numbers (-0.026 loss, no loss spikes at 4x LR) do
not transfer to it unmeasured.

`crates/dormouse-core/src/gr.rs` (`GR_BRANCHES = 4`), wired in
`loop_block.rs`'s `forward_full_state`; the loop's ordering test is
`crates/dormouse-core/tests/gr_seam.rs`.

**Not** mHC (`burn-mhc`), not AttnRes (`burn-attnres`) — both are implemented in
the library and neither is wired; ADR-0017 lists the four residual
implementations side by side. **Not** "gated MLP" (the expert FFN's gate is a
different mechanism). **Not** on by default: `use_gr = false` in every preset
(`schema.rs:104`).

### TSCT / expert / LinearLike

The low-rank spectral linear (`burn_sct::SpectralLinear`, masters `u`, `s`, `v`)
behind every wide projection: each expert's `gate_up`/`down`, the loop readout,
the `lm_head`, and GR's `wd`/`wu`/`ww`. `LinearLike` is the thin wrapper that
pads `out_features` to a multiple of 4, picks a quant format, and retracts.

`crates/dormouse-core/src/param.rs:33` (`LinearLike`), `:40`
(`LinearLikeInner::Tsct | Dense`), `:49` (the pad), `loop_block.rs:69` (`ExpertFFN`).

**Not** the MoE — the MoE is the *blend* of `n_experts` of these over one
iteration; a single expert is one of them. **Not** a "kind": kinds are
`LinearParam` values (`param.rs:14`). **Not** always spectral — `--set
use_tsct=false` builds the dense variant, and that A/B is live.

### quantization (three different ones)

1. **Factor quant** — `--quant fp32|bf16|fp16|fp8|fp4`, the format the TSCT
   factors are quantized to in the *forward*; the masters stay fp32.
   `crates/dormouse-train/src/lib.rs:66-70`, applied by `apply_compute_settings`
   (`lib.rs:645-648`).
2. **Activation quant** — `--act-quant 4|8|fp4` + `--act-group N`, BitNet
   a4.8-style, applied to the FFN input at the requested width and to the
   attention input at `max(bits, 8)`.
   `crates/dormouse-core/src/act_quant.rs` (`ActFormat`, `E2M1`,
   `ActFormat::attn`, `ActFormat::max_value`).
   `fp4` is the OCP MX **e2m1** grid: 8 magnitudes (0, 0.5, 1, 1.5, 2, 3, 4, 6),
   16 codes, 4 bits. The block scale maps a block's max onto 6, not onto 1 —
   normalizing to [-1, 1] left every level above 1 unreachable and the format
   carried 3 of its 8 magnitudes. Fixed 2026-09-28, together with a 0.75 that
   is not in e2m1 at all; **every `--act-quant fp4` number before that date is
   invalidated** (it was a ~3-level quantizer wearing a 4-bit label). The `4`
   and `8` paths are unchanged — same grid, same output.
3. **Weight quant** — the ternary/NM quantizers inside the library
   (`burn-bitnet`), reached through the TSCT factors.

**Not** interchangeable, and **not** a precision setting: `--quant` is the
*forward* path only, so checkpoints are format-agnostic and `--quant fp32`
after a fallback run reproduces the earlier behaviour exactly. `--bf16` is a
fourth thing again (storage dtype, not a quantizer).

### aux objectives (JEPA, DSpark, MoR BCE)

The three terms added on top of the honest CE. JEPA = masked-latent prediction
against an EMA teacher plus KoLeo; DSpark = a next-K draft head; MoR BCE = the
router's own top-k label.

`crates/dormouse-core/src/model.rs:241-302` (`aux_loss`), weights
`schema.rs:106-111` and `:178-180`.

**Not** part of the main loss: `DormouseModel::loss` returns the loop's CE and
nothing else (`model.rs:338`), and `train_loop` adds `aux` on top
(`crates/dormouse-train/src/lib.rs:1081-1083`). **Not** MTP (DSpark replaced
it). **Not** verified: every pre-2026-09-27 aux A/B conclusion is void, because
the teacher was being fed the labels (see `Teacher`).

### teacher

The EMA copy of the model the JEPA target comes from, momentum 0.999, advanced
after every optimizer step, stored in the checkpoint container.

`crates/dormouse-core/src/aux.rs:25` and `:203`; built in `ema_teacher_for`
(`crates/dormouse-train/src/lib.rs:685`), restored on resume (`lib.rs:822`).

**Not** a reference model in the RL sense, **not** an evaluator, **not** the
"oracle" halt head of ADR-0013's rejected design. It consumes exactly the
student's three inputs (`model.rs:150-156`) — it is the student's own latent
under EMA weights.

---

## 2. Memory

### Engram (the arm, and the in-VRAM tables)

The hashed n-gram **input** features: one table per n-gram order
(`engram_orders = [2,3,4]`), addressed by an FNV hash of the byte context, rows
rounded up to a power of two and the slot index *masked* on device.

`LoopBlock.engram` (`loop_block.rs:88`), table sizing at `loop_block.rs:24`, the
masked read at `loop_block.rs:424`.

**Not** an output-side memory (nothing is ever written back), and **not** a
cache: the rows are trained parameters. **Not** the host-RAM path — see *host
rows*. **Not** a knowledge base: at 25 000 rows/order the whole thing is 2.4M
params, a *minority* of the model on purpose (`loop_block.rs:692-724`).

### memory branch / memory_floor_mix

The block output's third arm, a **convex mixture with a hard floor**:
`lam * memory + (1 - lam) * dense`, `lam = min(w_mem, engram_lam_max)`. The
`dense` half (`mem_dense`, a plain `d -> d` projection of the same hidden
state) is what makes it a floor rather than a gate.

`loop_block.rs:58-66`, wired at `:385-453`, `mem_dense` at `:94`.

**Not** the Engram itself (that is the table), and **not** a learned mixture
weight — `lam` is *clamped*, so the backbone's share of the branch is never
below `1 - engram_lam_max` whatever the controller learns.

### mem_dense

The `d_model -> d_model` projection forming the dense half of the memory branch.
It is the reason the backbone cannot be starved by the table.

`loop_block.rs:89-94`, called at `:446`.

**Not** a residual and **not** an expert: it is one projection, always present,
never routed to Muon+ (it is a real `[d,d]` map — the expensive case for fp32
Newton-Schulz; it lands in `Group::Rest` by the arm's own `rest_of`, declared at
`routing.rs:246-249`).

### host rows (host_rows)

The per-position Engram rows a batch needs, already gathered on the host and
uploaded as one `[b, t, 3*32]` f32 tensor (~600 KB/step). Only exists on the
`--engram-ram` path.

`crates/dormouse-train/src/offload.rs:376` (`rows_for_batch`); consumed at
`loop_block.rs:395-408`; documented as `[b,t,96]` at `model.rs:114`.

**Not** the tables (those are `HostNgram`, in host RAM, never in the module
tree). **Not** hashes: `host_rows` is the *values*. **Not** a parameter — it is
a detached or grad-carrying leaf created per step, deliberately outside the
`Module`.

### HostNgram / the `.ngram` sidecar

The host-RAM copy of the three tables plus their single Nesterov momentum
buffer, with periodic Sinkhorn balancing of the batch's update block.

`crates/dormouse-train/src/offload.rs:39`, the update at `:130`, the file at
`:158`/`:172`, written by `save_ngram` (`crates/dormouse-train/src/lib.rs:661`).

**Not** "CPU Adam" — the update is Nesterov momentum (0.95) + Sinkhorn, one
momentum buffer instead of Adam's m+v. The *flag* is still called
`--host-adam-every` and the CLI help still says "CPU Adam"
(`crates/dormouse-cli/src/bin/train.rs:146` and `:151`); both are stale names
for a momentum update. **Not** 48M rows by default: `--engram-slots` defaults to
1 000 000 (`crates/dormouse-train/src/lib.rs:159`).

### ngram / hash / FNV

The key derivation: FNV-1a over the byte context at each of the three orders,
emitted by the data crate as raw (unreduced) hashes.

`crates/dormouse-data/src/lib.rs:20` (`fnv`), `:41` (`ORDERS = [2,3,4]`), `:428`
(`hashes_raw`) for the in-VRAM path and `:416` (`hashes`, reduced mod the table
size) for the host path.

**Not** the tables and **not** the rows: these are indices. The count is pinned
to 3 by `validate` (`crates/dormouse-core/src/config/validation.rs:27`) because
the trainer's hash tensor is `[b, t, 3]`.

### patched byte stream

The byte stream the trainer reads: a 64 MB ring refilled in 8 MB chunks from a
deterministically shuffled, FNV-sharded file list, with a held-out tail carved
out at filter time.

`crates/dormouse-data/src/lib.rs` (`ByteStream` at `:284`, `rewind` at `:487`,
`skip_bytes` at `:449`).

**Not** "trunk" — that word is reserved by `CONTEXT.md` for the stream, not the
model. **Not** tokenization: it is bytes, vocab 256, no merge table.

---

## 3. Parameters, routing, optimizer

This is where the repo has the most near-synonyms. Four words for one decision:

### Role

What a `LinearLike` **is** in the model: `Expert`, `Readout`, `Head`. Declared
by the module that owns it, at construction.

`crates/dormouse-core/src/routing.rs:67`.

**Not** an optimizer group (that is `Group`), **not** a parameter kind (that is
`LinearParam`), **not** an arm.

### LinearParam (a parameter "kind")

What one of a `LinearLike`'s three leaves **is** structurally: `Factor` (u or v),
`Scale` (the 1-D `s`), `DenseWeight`, `DenseBias`. No optimizer opinion.

`crates/dormouse-core/src/param.rs:14`, enumerated per linear at `:140`.

**Not** a "group", and **not** a "type" in the Rust sense (it is a
classification of one parameter tensor, not of the module).

### Group (the declared policy, id-based)

The optimizer group a `(Role, LinearParam)` pair trains on: `Muon`,
`QkHeadWise`, `Table`, `Rest`. The whole policy is the single match
`group_of`, and the group is built from `ParamId`s, not from strings.

`crates/dormouse-core/src/routing.rs:38` and `:90`; the check that every live
parameter is declared exactly once at `:180`.

**Not** `ParamGroup` (burn's own type, `burn::module::ParamGroup`) — that is the
framework mechanism; `Group` is our policy. **Not** "the Muon group" in the CLI
log, which is the same policy counted off the INSTALLED `ParamGroup`s
(`optim.rs::check_installed`) rather than off the declaration — the two must
agree and a loud gate says so, but they are counted by different code.

### markers (path-string routing) — DELETED, kept as a name

The former second vocabulary: `MUON_PATH_MARKERS` (`"expert_ffns."`,
`"engram.key_projs"`, `"out_proj.inner"`), `ENGRAM_TABLE_MARKER`
(`"engram.memory"`), `QK_HEAD_MARKERS` (`"gdn2.q_proj"`, `"gdn2.k_proj"`), and
the predicates `is_muon_param`, `is_engram_table_param`, `is_qk_param`.

**None of these symbols exist.** They were the trainer's own copy of the
optimizer policy, matching module **path strings**, and they are gone as of
`831e3a0` (2026-09-28). `grep -rn "MUON_PATH_MARKERS\|is_muon_param" --include=*.rs
crates/` returns one hit, and it is a comment remembering a divergence.

They were wrong, not merely redundant: the live marker set excluded
`inner.Dense.bias` but not `inner.Dense.weight`, so with
`--set use_tsct=false` a dense expert's full `[d,d]` weight reached Muon+ and
fp32 Newton-Schulz — the ~40 s/step case §2.3 records as solved, reachable
through a documented A/B flag. Every log line was counted off the markers, so
no run said so.

**Why the entry stays:** the divergence was real and silent, and it is the
reason [[group]] is built from `ParamId`s. Deleted vocabulary is still
confusable vocabulary; see `docs/reviews/dedup-optimizer-2026-10-01.md`.

**Do not say there are two optimizer policies.** There is one. `routing.rs`
declares it and `dormouse-train::optim` installs it (`optim.rs::optimizer_groups`
calls `routing::routing`). If a note tells you the id-based declaration is
"exercised only by tests" and the markers are what run, that note predates
2026-09-28 and is wrong — that claim was reintroduced into prose by `d81d920`
(2026-10-01), after the code fix it describes.

### Muon+ / HeadWiseMuon / the fallback

The optimizer family. Muon+ ColRow (NS 8 iters) on the small low-rank factors
and the Engram key projections; a per-head variant on the KDA q/k weights; plain
Adam with no weight decay on the n-gram tables; AdamW (or Adan under
`--opt mix-adan`) everywhere else.

`crates/dormouse-train/src/optim.rs:69` (`MUON_NS_STEPS`), `:161`
(`HeadWiseMuon`), `:310-330` (the `--opt` dispatch).

**Not** "Muon" (the upstream optimizer; ours is Muon+ with a different update
rule). **Not** a group: it is the *algorithm*; the group is where it is
applied. **Not** applied to 2D dense weights: fp32 NS on `[d,d]` cost ~40 s/step
on this box, so the policy deliberately routes only small factors.

### factors-fallback

The A/B knob that moves the expert TSCT factors from Muon+ to the base
optimizer, leaving the declaration valid.

`--factors-fallback` (`crates/dormouse-cli/src/bin/train.rs:60`), implemented
ONCE, as the `group_of` branch in `routing.rs` — which is the policy the
trainer installs.

**Not** a quant fallback: `factors_fallback` (which params) and the one-way
`max_ortho` fp32 fallback (which *forward*, and which latches forever) are
different things with confusingly similar names.

---

## 4. Config, run, artifacts

### preset

A flat TOML file in `configs/`, found by name or path, holding model-schema
values. There is no Rust registry; adding a preset is creating a file.

`crates/dormouse-core/src/config/loader.rs`; the eight files are
`nano`, `nano-fused`, `small`, `mor`, `base`, `swift50`, `one_b`, `p150`.

**Not** a recipe (a recipe is a preset *plus* its validated hyperparameters and
data) and **not** an architecture — they all share one architecture but differ
in arms, so `nano-fused` (KDA + Engram + aux off) and `swift50` (8 experts, aux
off) are not "the same shape, smaller". The schema defaults mirror `small`
exactly, pinned by `default_equals_small_preset`
(`crates/dormouse-core/src/config/schema.rs:200`).

### resolve / the config seam

The single place a run's configuration is decided: serde defaults -> preset ->
`--set` -> typed flags -> `validate`, in that order, producing a `RunCfg`.

`crates/dormouse-train/src/cfg.rs:28`; the CLI's merge in
`crates/dormouse-cli/src/bin/train.rs:230`.

**Not** "the config": it is the *resolved* config, and it is the only thing a
snapshot can record. **Not** a second default source: every default is written
once, in the schema.

### RunCfg / snapshot / drift check

The resolved run, serialized next to the checkpoint as
`<ckpt_name>.config.toml`, and diffed key-by-key on every resume.

`cfg.rs:14` (`RunCfg`), `:79` (`snapshot_toml`), `:104` (`diff_keys`) with the
exempt progress keys at `:105`; the check runs before any GPU work
(`crates/dormouse-train/src/lib.rs:768-787`).

**Not** free to change: everything except `steps`, `log_every`, `ckpt_every`,
`eval`, `eval_every` is drift, *including* `seed` and `rand_depth`. **Not** a
model export — it carries no weights.

### checkpoint (ckpt container)

`<ckpt_name>.bin`: a v2 container with magic `DMCK\0\2\0\0`, holding step,
model, optimizer, the EMA teacher and the flags bit (the one-way fp32-factor
fallback). `<ckpt_name>.prev.bin` is a hard link to the previous save.

`crates/dormouse-train/src/lib.rs:428` (magic), `:493` (`save_ckpt`), `:543`
(`load_ckpt`); the rotation at `:510-512`.

**Not** a model export (there is no "export" in this codebase — the weights
*are* the checkpoint) and **not** the config, the tables or the loss line. **Not**
a snapshot: `.config.toml` is the snapshot, `.bin` is the checkpoint.

### sidecar (three different files, all called it)

1. **`<ckpt_name>.txt`** — one line, `step N ce X.XXX`, for humans.
   `crates/dormouse-train/src/lib.rs:539`.
2. **`<ckpt_name>.ngram`** — the host-RAM tables and momentum. `lib.rs:661`.
   Layout magic `"2MGN"`; a v1 (Adam m+v) file is refused loudly.
3. **the offline JEPA target file** (`--jepa-targets`, written by
   `--jepa-precompute`) — precomputed teacher latents keyed by chunk hash.
   `crates/dormouse-train/src/jepa_targets.rs:68`.

**Not** interchangeable and **not** all present: (2) exists only under
`--engram-ram`, (3) only under `--jepa-targets`. None of the three carries a
training-step stamp; the `.ngram` one is a known open gap (ADR-0021 item 7).

### guard

The in-process NaN/panic recovery: wait 30 s, re-exec from a pinned executable
image with a fresh CUDA context, resume from the last checkpoint.

`--guard` (`crates/dormouse-cli/src/bin/train.rs:167`), the pinned image at
`train.rs:255-290`.

**Not** the NaN firewall (that is in-process, per step, and masks the step to a
no-op — `mask_nonfinite`, `lib.rs:281`). **Not** a retry loop in the shell.

### seed

`--seed`, default 1. It seeds the JEPA span mask, which is a pure function of
`(seed, step)`, so a resume redraws the same masks and an A/B replays them.

`crates/dormouse-train/src/lib.rs:137-141`, set per step at `:990`.

**Not** a model-init seed: burn's init is per process, so two runs of the same
config start from different weights and `--seed` does not fix that. **Not** the
data order: the stream's shuffle is seeded separately (and deterministically) in
the data crate. **Not** a config knob you may flip mid-run — it is in the
snapshot, so flipping it is drift.

---

## 5. Measurement

### BPB

Bits per byte: `ce / ln 2` on the held-out window. The one score.

`crates/dormouse-train/src/lib.rs:30`.

**Not** loss (raw CE is not comparable across configs), **not** perplexity, and
**not** train CE — the train CE collapses when a big n-gram table memorizes the
stream, which is expected and meaningless.

### eval / eval tail / held-out

The held-out evaluation: `eval_batches` (default 20) windows of
`batch * seq_len` bytes, from a stream **rewound before every eval** so every
run scores the same bytes. The tail itself is a carve of the real corpus that
training never sees (ADR-0010).

`crates/dormouse-train/src/lib.rs:1310-1454` (the eval block and the byte count
at `:1417`), `rewind` at `crates/dormouse-data/src/lib.rs:684`, `eval_batches`
default at `lib.rs:158`.

**A "held-out window" is not a fixed size.** The formula above makes it a
function of the batch: 102 400 B at batch 10 × seq 512, 20 480 B at batch 2, both
with the default 20 batches. **A BPB is comparable only within one window**, so
a window is part of the measurement, not a detail of it — the byte count is
printed on every eval line and belongs in any table a number appears in. This
entry carried the formula while `AGENTS.md` §2.6, `docs/protocols/AB-PROTOCOL.md` and two
`README.md` places carried the constant "100 KB", which was the batch-10 shape
quoted as if it were a property of the eval; a one-word fix here would have
prevented four wrong documents.

**Not** the depth curve (`--eval-depths`, which re-runs the same window at
depths 1..=max_iter and trains nothing). **Not** a validation *split* of the
training files.

### A/B

One mechanism against its own removal, on held-out BPB, at a fixed step budget,
with 3 seeds per arm (the seed spread is larger than the effects we measure).

`docs/protocols/AB-PROTOCOL.md`; the queue with flags and costs is in that file's table.

**Not** a step-time claim (those need same-shape measurement on a quiet GPU),
and **not** one arm vs one arm: a single pair decides nothing at our scale.

### arm (experiment sense)

One row of the A/B queue: a flag or preset plus what it decides. `control`,
`pure CE`, `dense FFN`, `rand depth`, `depth 2 vs 4`, and so on.

`docs/protocols/AB-PROTOCOL.md:43-53`.

**Not** the in-model arm (see *arm* above). **Not** a preset: an arm is usually
one flag away from the control.

### canary

The cheap regression benchmark — `scripts/bench.sh canary`: `small`, aux off,
2M slots, 30 steps, ~2 GB RSS, ~2 min — whose row in `benches/history.tsv` is
the gate before and after every optimization.

`scripts/bench.sh:25`; the history file is `benches/history.tsv`.

**Not** a smoke (a smoke screens NaN and slope; a canary measures step time).
**Not** the *fake-loss canary* of the adaptive-depth research, which is an
invariant check on a loss curve (`docs/research/2026-09-27-adaptive-depth-safe.md`).
Two words, two things, one repo.

### smoke / confirm / long gate

The three rungs of the A/B ladder: a 200-500 step smoke filters NaN, speed and
early slope; a 2k+ step confirm produces a BPB verdict; a long gate measures BPB
at distance (512k-1M positions) after each context extension.

`docs/adr/0002-ab-or-death.md`; the long gate in `docs/adr/0004-context-ladder.md`.

**Not** interchangeable: a smoke survivor is a candidate, not a result, and
nothing has yet passed a long gate.

---

## 6. The word that means two things: `fused`

**Two live meanings, one dead one, and they have already caused a misreading.**

| meaning | what it is | where | state |
|---|---|---|---|
| **fused kernel** (correct today) | a library op that does the whole computation in one CUDA kernel instead of a chain of eager tensor ops, with a `Fused`/`Fallback` seam that says which ran | `burn_gdn2`, `burn_rmsnorm::fused`, `burn_muon_plus::fused_kernels`, `burn_spectral`; counted at `crates/dormouse-core/src/attention.rs:20` (`fused_seam_counts`) and `crates/dormouse-train/src/optim.rs:59` (`fused_kernels_skipped`) | **live**, and deliberately countable, because the fallback computes the same function and nothing else would show it (ADR-0019) |
| **the whole-loop `fused/` module** | a hand-written CUDA autodiff op for the entire ponder loop, ~7.4k LOC, once `crates/dormouse-core/src/fused/` | **deleted.** The directory is gone, the `DM_FUSED` env var is gone (`grep -rn DM_FUSED crates/` is empty) | measured **9.21 s/step against burn's 7.05** at flagship (`docs/archive/research/2026-09-23-fused-flagship50.md:52-53`), 1.3x slower; the earlier "1.7-2.0x faster" number from ADR-0003 is **retracted** in ADR-0009 |

Which one is correct: **the library kernel.** The whole-loop op is gone; the
kernel is the live concept, and the counters that exist to prove a kernel ran
(`fused_seam_counts`, `fused_kernels_skipped`, `burn_gdn2::fused_calls`) all
mean the library kind.

What still invites the confusion, and what I did **not** rename (other agents
are in these files):

- **`configs/nano-fused.toml`** — a preset named after the dead module, whose
  header comment still says "DM_FUSED=1 single-node path only", a variable that
  no longer exists. Today it is just `nano` with KDA, Engram and both aux
  weights off. Proposed rename: `nano-arms-off` (or delete it; `nano` + three
  flags is the same thing, and `cfg.rs:349` already pins its behavior).
- **ADR-0003 and ADR-0009 still written in the present tense** about
  `crates/dormouse-core/src/fused/` and about rungs/kill switches that the
  deletion pre-empted. ADR-0009's "Either rung failing its number deletes all
  ~7.4k LOC" is now history.
- **`crates/dormouse-core/src/lib.rs:1`** — "all-bf16 mini Aria on burn-fused
  kernels": both the dtype claim and the project name are stale, and
  "burn-fused" here means the library, not the module.

---

## Code vs document: the disagreements

Every one of these is a name or a default where the code and a document said
different things. Each needs a decision, not a doc edit: which side is wrong.

**State**: `FIXED` = this commit corrected the document side (what is left is
in `crates/`, `docs/adr/` or `README.md`, which other owners hold); `OPEN` =
still needs an owner's decision, and the fix may be a code change.

| # | subject | code says | document said | state |
|---|---|---|---|---|
| 1 | loop halting | no halt head, no `p_n`, no KL; fixed depth, honest CE (`model.rs:338`, `loop_block.rs:481-489`, ADR-0013) | "halt head -> PonderNet halting" (old `AGENTS.md` mission line); the mermaid says "PonderNet halting decides how deep to think" and draws a `halt head: PonderNet` node (`README.md:38`, `README.md:50`) | `AGENTS.md` **FIXED**; `README.md` **OPEN** (another agent owns it) |
| 2 | sparse attention | no second attention arm exists (`attention.rs:1-6`, ADR-0014) | "sparse attention" in the old `AGENTS.md` mission line; **MSA was a glossary entry** in the old `CONTEXT.md`, as if it were present | **FIXED** in both files — MSA is now under "what is deliberately absent" |
| 3 | forward signature | `(logits, rec, kda, aux)` — 4 values (`model.rs:126`) | `(logits, rec, p_dist, kda, aux)` — 5, with `p_dist` no longer existing (old `AGENTS.md` state notes) | **FIXED** (`AGENTS.md` §3.8) |
| 4 | loss | `loss(rec_ce) = rec_ce` (`model.rs:338`) | `loss(rec, p_dist) = L_Rec + beta*KL` (old `AGENTS.md` state notes) | **FIXED** (`AGENTS.md` §3.8) |
| 5 | host-table optimizer | Nesterov momentum + Sinkhorn, one buffer (`offload.rs:26-32`, `:130`) | "CPU Adam" (old `AGENTS.md`; `--engram-ram` and `--host-adam-every` help text at `train.rs:146`, `train.rs:151`) | `AGENTS.md`/`CONTEXT.md` **FIXED**; the two flag help strings **OPEN** — the honest fix is renaming `--host-adam-every`, and that is a snapshot key, so an ADR-0021 question |
| 6 | host-table footprint at 48M slots | 12.3 GB for table + momentum (`offload.rs:12-14`) | 18.4 GB, the Adam m+v number (old `AGENTS.md` RAM doctrine and resume math) | **FIXED**; the resume figure is now marked derived, not re-measured |
| 7 | Engram default | in-VRAM, `engram_rows = 25_000` (`schema.rs:74`, `schema.rs:146`) | "Engram: hashed n-gram embedding tables living in host RAM, trained by CPU Adam, rows fetched per batch" (old `CONTEXT.md:38`) — that is the `--engram-ram` path only; `README.md:20` says "n-gram memory lives in host RAM by the billion of rows" as the design | `CONTEXT.md` **FIXED** (Engram / host rows / HostNgram are now three separate entries); `README.md:20` **OPEN** |
| 8 | presets | eight files, including `mor`, `p150`, `nano-fused` (`configs/`) | "nano, small, base, swift50, one_b" (old `CONTEXT.md:48`; the `--preset` help at `train.rs:19`) | `CONTEXT.md` **FIXED**; `train.rs:19` **OPEN** |
| 9 | "presets differ in size only" | `nano-fused` (kda+engram+aux off), `swift50` (aux off, 8 experts), `mor` (+3 keys) are not size variants | "All presets share one architecture; they differ in size only" (old `CONTEXT.md:48`) | **FIXED** |
| 10 | `small`'s parameter count | 9.20M total / 6.05M compute, measured by `tests/preset_exec.rs` (`README.md:101`) | 7.5M backbone, hardcoded as `BACKBONE_SMALL` in `loop_block.rs:706` and asserted as "`small` has 7.5M" in `schema.rs:130` — so the 24% memory share is computed against the wrong denominator | **OPEN** — a code fix in `crates/`, and it is the number the capacity budget is argued from |
| 11 | PonderNet in a flag help string | `--max-iter` is the loop depth | "PonderNet loop depth (default: preset)" (`train.rs:73`) | **OPEN** (one-line doc fix) |
| 12 | `--gen-max-iter` | the flag does not exist (`generate.rs` has 9 flags, none is it) | "at inference stays max_iter; --gen-max-iter picks it lower" (`train.rs:114`) | **OPEN** — either the flag or the sentence |
| 13 | `bool -> float` cast | correct on this backend, verified at n=1..1000 on raw and dispatch paths (ADR-0016 bug 1) | "The `Bool -> float` cast is broken on this backend (it returns 0.0 for `true`)" — the doc comment on `GradSanitizer`, `crates/dormouse-train/src/lib.rs:300-302` | **OPEN** — a retracted claim living in a code comment is the exact ADR-0020 failure |
| 14 | `mini Aria` | dormouse | `crates/dormouse-core/src/lib.rs:1` — "all-bf16 mini Aria on burn-fused kernels" | **OPEN** (a stale project name *and* a stale dtype claim, on the crate root) |
| 15 | patching | not implemented; no `patch` token anywhere in the model | **Patching** was a `CONTEXT.md` glossary entry with a verdict pending (old `CONTEXT.md:33`) | **FIXED** (entry deleted — a pending verdict on an unimplemented idea is not vocabulary) |
| 16 | routing policy | ONE policy: `routing.rs` declares it from `ParamId`s, `dormouse-train::optim` installs it. The path-marker copy in `optim.rs` was deleted by `831e3a0` (2026-09-28) | "Muon+ mixed optimizer (policy + groups in `src/optim.rs`)" plus a glossary section presenting the markers as "the live implementation", and the claim that `routing.rs` was "exercised only by tests" — reintroduced into prose by `d81d920` (2026-10-01), AFTER the code fix | **FIXED** (2026-10-01) — the second `GroupCounts` and the second module walker are cut (`docs/reviews/dedup-optimizer-2026-10-01.md`); the false "two implementations" text is gone from `routing.rs`, `optim.rs`, `AGENTS.md` §3.3, the glossary and `docs/papers/muon-plus.md` D17 |
| 17 | stride | `dspark_stride` is a config field (`schema.rs:112`), no documented meaning anywhere | no document mentioned it | **OPEN** |
| 18 | ADR index | 22 ADRs | "ADR-0001..0012" (`README.md:199` docs table) | **OPEN** (`README.md`) |
| 19 | `nano-fused` | KDA+Engram+aux off; `DM_FUSED` exists nowhere in the tree | "KDA+Engram+aux off, single-node path", and the file's own header comment names the deleted env var (`configs/nano-fused.toml:2`, `README.md:99`) | **OPEN** — proposed rename `nano-arms-off`; not renamed here, and `cfg.rs:349` pins the current name in a test |
| 20 | ADR-0009's kill switch | the module it guards is already deleted | ADR-0009 written in the future tense about rungs that will "delete all ~7.4k LOC of `fused/`" | **OPEN** — `docs/adr/` has other agents in it |
| 21 | **the eval's n-gram keys** | the eval passes the keys the training step passes, and counts it: `engram=<rows>/<arms>` (`lib.rs:1384-1390`, `:1443`; fix in `7adda92`) | `b3d6914`'s **commit message** spends a paragraph on this fix and its **diff does not contain it**; the fix is in `7adda92`, whose message is about the best-checkpoint and does not mention it. `c4214ad` is the no-leak enforcement | `AGENTS.md`/`README.md`/`AB-PROTOCOL.md` **FIXED** — all three cite `7adda92` and state the fix's scope. `b3d6914`'s message is **OPEN**: a message describing a fix it does not carry is the ADR-0020 failure in its purest form, and the log is `crates/`'s file plus history |
| 22 | **the eval window size** | `eval_batches × batch × seq_len` bytes, printed on every eval line (`lib.rs:1417`); 102 400 B at batch 10, 20 480 B at batch 2 | "100 KB per eval (`--eval-batches 20`, 20 × 5 KB)" in `AGENTS.md` §2.6/§3.1, `docs/protocols/AB-PROTOCOL.md` and `README.md` — the batch-10 shape quoted as a property of the eval, against runs reporting `over 20480 B` | **FIXED** in all four; the glossary's "eval tail" entry carried the right formula the whole time, which is why the disagreement was findable |
| 23 | **the anchor triple** | four readings, none on a trainer eval window: unigram 5.398 / 5-gram 2.911 (source unnamed), 5.170 / 2.572 (`anchors.rs:3-7`, `README.md`), `--fit` 5.011 / 2.588 (`anchors.rs:27-32`, same file but the whole tail), 2.826 vs 2.849 at two `--bytes` | `AGENTS.md` §2.6/§3.1 and `docs/protocols/AB-PROTOCOL.md` both published "unigram 5.398, 5-gram 2.911 **on the same window**"; `anchors.rs:22-27` says the internal split "was never comparable" to the trainer's eval window, which is why `--fit` exists | **FIXED** in `AGENTS.md`/`README.md`/`AB-PROTOCOL.md` — all four readings named, none reusable, and the rule is now `anchors --fit` on the eval file with its `window:` line quoted. **OPEN**: the 5.398/2.911 reading has no command line recorded anywhere, so it can be neither reproduced nor retired — someone has to re-run it or delete it |
