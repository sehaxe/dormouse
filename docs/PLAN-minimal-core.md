# PLAN — the minimal core: what leaves `dormouse-core`, and what dies

Snapshot: `d651eb1`, working tree 2026-09-27T14:59Z. **The tree was moving while
this was written** — 8 files under `crates/` and 8 under `vendor/burn-fused` are
uncommitted, and `burn-gdn2` does not currently compile. Every LOC count below is
`wc -l` on that snapshot, not HEAD. Moves that touch an in-flight file are marked
**WAIT** and say which agent.

Goal, verbatim: *"move out of dormouse everything that can be moved out, so the
project is minimal, so it is maximally easy to train, and the dependencies are
separate so they are easier to fix and change."*

---

## 0. The blocking finding, before any structure

**`dormouse-core` does not compile.** `loop_block.rs:15` says
`use crate::mor::{self, MoRRouter};` and `LoopBlock` instantiates
`MoRRouter::new(d, device)` (line 215) — but `lib.rs` has no `pub mod mor;`, and
`burn-mor` is a **`[dev-dependencies]`** entry of `crates/dormouse-core/Cargo.toml`,
not a regular one. `src/mor.rs` (195 LOC) is an orphan file that no module graph
reaches.

Worse, the wiring is a stub: `forward_full_state` contains **zero** calls to
`mor::route()`. The router is built into every model, its params are visited by
`validate_routing` and counted in the optimizer banner, `use_mor` / `mor_k` are
parsed, overridden, and validated, `configs/mor.toml` exists as an A/B preset —
and none of it changes the trained model. See deletion **D1**: delete, do not move.

Nothing in this plan can be gated until C0 lands.

---

## 1. The dependency DAG, as it is today

### 1.1 Our crates

```
dormouse-cli (644)
  ├── dormouse-core      src 2493 | tests 615 | examples 1223
  │     ├── burn-kda (1208) ──▶ burn-gdn2 (4194)
  │     ├── burn-engram (586)
  │     ├── burn-rmsnorm (235)
  │     ├── burn-spectral (6172) ──▶ burn-bitnet (1815)
  │     ├── burn-jepa (443)
  │     ├── burn-dspark (865)
  │     ├── burn-bitnet (1815)      ✗ declared, ZERO call sites in src/
  │     ├── burn-cuda               ✗ declared, ZERO call sites in src/
  │     ├── rand                    ✗ declared, ZERO call sites in src/
  │     ├── serde + toml            ✓ (config/ only)
  │     └── burn-mor (743)          ✗ dev-dep only, wired into loop_block
  ├── dormouse-data (src 2548)
  │     └── arrow 59.2 + parquet 59.2
  ├── dormouse-train  src 3254 | tests 103 | examples 22
  │     ├── dormouse-core
  │     ├── dormouse-data
  │     ├── burn-spectral (6172) ──▶ burn-bitnet (1815)
  │     ├── burn-bitnet (1815)      ✓ only the --quant-check probe
  │     ├── burn-muon-plus (1034)
  │     └── burn-dispatch, cubecl-runtime, cubecl-cuda, cudarc, burn-cuda
  └── clap, axum, tokio, libc, serde_json
```

Regular library edges are only three: `burn-kda → burn-gdn2`,
`burn-byteflow → burn-swiglu`, `burn-spectral → burn-bitnet`. Everything else in
the fork is a leaf. (`burn-spectral`'s `burn-kda`/`burn-muon-plus`/`burn-sct`/
`burn-rope`/`burn-situ` edges are `[dev-dependencies]`, used by one example —
**not** a violation. Checked.)

### 1.2 The library: 28 crates, 29 892 LOC, 41% unwired

| wired into a consumer | LOC | unwired (no consumer outside the fork) | LOC |
|---|---|---|---|
| `burn-spectral` 6172, `burn-gdn2` 4194, `burn-bitnet` 1815, `burn-kda` 1208, `burn-muon-plus` 1034, `burn-dspark` 865, `burn-mor` 743*, `burn-engram` 586, `burn-jepa` 443, `burn-rmsnorm` 235 | **17 093** | `burn-sct` 2738, `burn-attnres` 2060, `burn-byteflow` 1121, `burn-mhc` 1011, `burn-rope` 975, `burn-diffusionblocks` 793, `burn-situ` 675, `burn-fastblt` 513, `burn-mod` 411, `burn-ptrn` 399, `burn-es` 345, `burn-eggroll` 327, `burn-parcae` 317, `burn-antihall` 302, `burn-mtp` 196, `burn-swiglu` 187, `burn-ttt` 125, `burn-nope` 102 | **12 797** |

\* `burn-mor` is wired only as a dev-dep into a file that does not compile.

### 1.3 Violations, model → library (things the MODEL owns that a library should)

| # | Violation | LOC | Why it is a violation |
|---|---|---|---|
| **M1** | `core/src/param.rs` — `LinearLike` + `LinearLikeInner` | 165 | A *linear layer technology* (pad-to-4 for cubek, per-SM quant dispatch, polar retraction, per-entry ortho metric) inside the model crate. It cannot be unit-tested against another module's arithmetic without a `DormouseModel` in scope, and no other project can import it. The only obviously reusable thing in `dormouse-core`. |
| **M2** | `core/src/gr.rs` — Gated Residual | 117 | A published mechanism (Qwen3.8-Flash-Next Eq. 30-34) living in the model, while **three of its four rivals** live elsewhere: ReZero inline in `loop_block.rs:213`, mHC in `burn-mhc` (1011), AttnRes in `burn-attnres` (2060). A/B-ing the residual stream is currently a diff inside the model. |
| **M3** | `core/src/aux.rs::ema_update` + `ParamCollector` + `EmaMapper` | 60 | `ema_update<M: Module>` never touches a dormouse type. It carries the `no_grad()` freeze that fixed a *measured* 2× activation OOM (a grad-tracked EMA teacher retained every intermediate of a second full forward). Generic Module EMA with a real, recorded reason to be careful. |
| **M4** | `core/src/aux.rs::jepa_aux_loss` / `dspark_aux_loss` | 125 | Loss wrappers around `burn-jepa` / `burn-dspark` primitives. The losses belong with their implementations; only `AuxHeads` (12 LOC, composition) stays. |
| **M5** | `core/src/act_quant.rs` | 170 | BitNet a4.8 activation quantization — one published mechanism, sitting in the model because its *weight* half happens to be in a library crate. |
| **M6** | `ActFormat` (act_quant.rs:24) and `ActQuant` (schema.rs:11) | ~50 | The same 2-variant type written twice, bridged by a `From`, plus 35 lines of hand-rolled serde + `FromStr` for three strings. |

### 1.4 Violations, library → model, and model → trainer (the other direction)

The question "what does the LIBRARY own that only this model needs?" has a
surprising answer: **almost nothing, and that is the finding.** Instead:

| # | Violation | LOC | Why |
|---|---|---|---|
| **X1** | `train/optim.rs` keeps a **table of the model's private field names** — `"expert_ffns."`, `"engram.key_projs"`, `"out_proj.inner"`, `"gdn2.q_proj"`, `"gdn2.k_proj"`, `"engram.memory"` — and pays 85 LOC of `validate_routing` + `PathCollector` + `GroupCounts` to *detect that the table went stale* (a rename in `loop_block.rs` silently degrades a param to AdamW). | 85 | The model's field names are the only routing table, and the trainer owns it. This is the inverse of ADR-0017 and it is the one violation that will actively fight the moves below: C4/C5/C6 all rename fields the table names. |
| **X2** | `train/offload.rs` — host-RAM n-gram tables, Nesterov + Sinkhorn balancing, `LAYOUT_MAGIC`, the v1/v2 migration panic, `rows_for_batch` | 402 | The Qwen-playbook's *core mission* (billions of memory params in host RAM, zero VRAM) is 400 lines of memory technology living next to the checkpoint-save code. It also carries the unrecorded Adam→Nesterov swap the audit flagged: the code, the flag name (`--host-adam-every`) and the +2.3% measurement all still say **Adam**. Not in ADR-0017 at all. |
| **X3** | `train/optim.rs::HeadWiseMuon` | 120 | A `burn::optim::Optimizer` impl of a *published* mechanism (Qwen §3.1: split qkv per head **before** orthogonalization), with zero dormouse in its body, sitting outside `burn-muon-plus` — the crate that owns Muon+. Same violation as M2, in the trainer. |
| **X4** | `train/stress.rs` (`StressMonitor`, `grad_norm`) + `train/lib.rs::mask_nonfinite` | 112 | Generic stability instrumentation: zero dormouse references, works on any `Module`. It is the instrument every future A/B's stability claim depends on. Not in ADR-0017. |
| **X5** | Three FNV-1a implementations: `core::fnv_hash` (lib.rs:18), `data::fnv` (data/lib.rs:20), `shard::fnv` (shard.rs:16) — plus `burn-engram::hasher` (255 LOC) | 13 dup | The hash **is** the memory-addressing contract between the data crate and the model. Four implementations of a contract is a correctness hazard, not duplication. Flagged by the audit on 2026-09-25, unfixed. |
| **X6** | `dormouse-core` declares `burn-bitnet` (1815), `burn-cuda`, `rand` with **zero** call sites | 3 lines | ADR-0017's premise for the act_quant move is *false at the code level*: act_quant.rs imports nothing from burn-bitnet. The two halves of "BitNet a4.8" are independent f32 tensor ops that share only a paper reference. |

---

## 2. The target DAG

### 2.1 Out of the model

| today | destination crate | LOC | ADR-0017 said | correction |
|---|---|---|---|---|
| `core/src/param.rs` | **`dormouse-linear`** (new) | 165 | `dormouse-linear` | agree on the destination, **shrink the claim**: the tech (pad, quant, retract) is already in `burn-spectral`; `LinearLike` is *policy* over it. The crate earns its place only if it also absorbs `LinearLike`'s two tests out of `train/lib.rs` (they test the library, from the trainer). |
| `core/src/gr.rs` | **`dormouse-residual`** (new, 4 arms) | 117 + ~120 new | `dormouse-residual` *or a module in `dormouse-mhc`* | **reject the mHC variant.** Putting GR inside mHC makes the mHC API = ReZero ∪ GR ∪ mHC ∪ AttnRes and turns the A/B back into a diff inside the library. One crate, four constructors, one `ResidualArm` enum. |
| `core/src/act_quant.rs` + `ActFormat` | **`dormouse-bitnet`** (existing, +1 module) | 170 + 35 serde | `dormouse-bitnet` | agree on destination, **the stated reason is wrong** (see X6). The real reason: same paper (b1.58 2B4T), same STE convention, one place that knows what fp4-e2m1 means. Move the hand-rolled `FromStr`/serde in with it and core re-exports. |
| `aux.rs::jepa_aux_loss`(+`_masked`) | **`dormouse-jepa`** (existing) | 35 | `dormouse-jepa` | agree. |
| `aux.rs::dspark_aux_loss` | **`dormouse-dspark`** (existing) | 90 | `dormouse-dspark` | agree. Rename to `loss_at_anchors(head, conf, hidden, logits, ids, k, stride)` — the 5-anchor gather/stack plumbing is dspark's business, not the model's. |
| `aux.rs::ema_update` + 2 visitors | **`dormouse-ema`** (new) | 60 | "a thin composition helper in core" | **disagree.** It is generic, and it is where the `no_grad` freeze lives. A library crate gets its own regression test; a "helper in core" gets none. |
| `train/offload.rs` | **`dormouse-ngram`** (new) | 256 + 84 | *absent* | add. See X2. The checkpoint writer/reader stays with the trainer; the tables and their optimizer move. |
| `train/optim.rs::HeadWiseMuon` | **`dormouse-muon-plus`** (existing) | 120 | *absent* | add. See X3. Its test (`headwise_muon_matches_per_slice_muon`, 35 LOC in `train/lib.rs`) moves with it. |
| `train/stress.rs` + `mask_nonfinite` | **`dormouse-stability`** (new) | 112 | *absent* | add. See X4. Zero dormouse deps: just `burn`. |

### 2.2 Into the model (the reverse move, X1)

`MUON_PATH_MARKERS`, `QK_HEAD_MARKERS`, `is_muon_param`, `is_dense_bias`,
`is_qk_param` → **`core/src/routing.rs`** (~50 LOC). The model names its own
params; the trainer keeps only the `ParamGroup` wiring and `build_optim_mode`.
`validate_routing` + `PathCollector` + `GroupCounts` are **deleted** (−85 LOC): the
fail-loud property survives as a test in the one file that can be wrong.

### 2.3 What stays in `dormouse-core`, and why

| file | LOC | why it cannot move |
|---|---|---|
| `model.rs` | 337 | It is the model. The bf16-head-fp32 rule and the in-loop L_Rec contract live here and belong to the composition. |
| `loop_block.rs` | 617 | The *bodies* of the arms move out; the *order* they are composed in, the `e_k` iteration embedding, the controller's three gates + expert blend, the per-iteration readout, the `h = h_ctx + y·scale` recurrence, and the sm_120 "never slice a 4D autodiff tensor" L_Rec accumulation are the model. |
| `attention.rs` | 33 | Exists for one reason, stated in its own doc comment: it keeps the checkpoint prefix `loop_block.shared_attn.gdn2.*` stable across the MSA cut. Renaming it breaks every `.bin`. (The *type* name `AdaptiveAttention` is now a lie — one arm, not an adaptive choice. Renaming the type is free; renaming the field is not.) |
| `config/` (schema 216, loader 139, override 70, validation 41, mod 9) | 475 | ADR-0005's one seam. The schema defaults **are** the `small` preset (a test asserts `default() == small.toml`) — a library cannot own this model's presets. |
| `aux.rs::AuxHeads` | 12 | Composition: this model instantiates the JEPA predictor and the DSpark head with *its* `d_model`/`vocab`/`rank`. |
| `routing.rs` (new) | 50 | The optimizer *policy* is per-model. It moves **in**, not out. |
| `lib.rs::fnv_hash` | 8 | **Deleted** (D2), not moved. |

### 2.4 The target in one line

```
dormouse-core (src ~1200: lib 62 + model 337 + loop_block ~450 + attention 33
               + config 475 + routing 50 + AuxHeads 12) composes 7 library
               crates by enum, and knows no implementation detail of any of them.
dormouse-fused/ (30 crates, 30.4k LOC) owns every mechanism, each with its own
               test, its own arXiv reference and its own A/B.
```

---

## 3. The public API surface each moved mechanism exposes

Rules for every crate below: **3-5 entry points**; **must not import
`dormouse-core`** (that is a cycle); config arrives as a **plain struct** the
model constructs at the call site, never as `DormouseConfig`.

```rust
// dormouse-linear (165)                                    knows: burn, burn-spectral
pub struct LinearCfg { pub in_features: usize, pub out_features: usize,
                       pub rank: usize, pub kind: Kind /*Spectral|Dense*/ }
pub enum Kind { Spectral, Dense }
pub struct LinearLike;                                        // burn::Module
impl LinearLike {
    pub fn new(cfg: LinearCfg, device: &Device) -> Self;
    pub fn forward<B: AutodiffBackend>(&self, x: Tensor<2>) -> Tensor<2>;
    pub fn set_format(&mut self, q: QuantFormat);   // one switch (was set_quant)
    pub fn set_bf16_compute(&mut self, on: bool);
    pub fn retract(&mut self, ns_iters: usize);     // polar retraction
    pub fn ortho_error(&self) -> f32;               // per-entry; 0 for dense
}
// must NOT know: DormouseConfig, the model, the loop, d_model, vocab.

// dormouse-residual (117 + ~120)                           knows: burn, burn-rmsnorm, dormouse-linear
pub enum ResidualArm { ReZero, Gated, Mhc(MhcCfg), AttnRes(AttnResCfg) }
pub enum ResidualStream { /* one variant per arm, each holding ITS OWN params */ }
impl ResidualStream {
    pub fn build(arm: ResidualArm, d: usize, device: &Device) -> Self;
    pub fn read<B>(&self, x: Tensor<3>, state: &mut ResidualState) -> Tensor<3>;
    pub fn write<B>(&self, y: Tensor<3>, state: &mut ResidualState) -> Tensor<3>;
}
// NOT a trait + Box<dyn>: burn's Module derive already handles enums here
// (LinearLikeInner is the precedent) and the enum stays monomorphised.
// must NOT know: the loop, e_k, KDA, the Engram, the model.

// dormouse-bitnet += act_quant (170 + 35)                  knows: burn only
pub enum ActFormat { Fp4, Int(u32) }
impl ActFormat { pub fn attn(self) -> Self; }
impl FromStr for ActFormat / Serialize / Deserialize          // moved from core
pub fn quant_act<B: Backend>(x: Tensor<2>, fmt: ActFormat, group: usize) -> Tensor<2>;
// core then does: pub use dormouse_bitnet::ActFormat;  (one line, -35 in core)

// dormouse-ema (60)                                        knows: burn::Module
pub fn ema_update<M: Module>(teacher: M, student: &M, momentum: f64) -> M;
// Test: momentum 0.0 ⇒ teacher == student bitwise; one student step ⇒ they
// differ; every teacher param has require_grad == false (the freeze is the
// load-bearing part — it is what fixed the measured 2x activation OOM).

// dormouse-ngram (256 + 84)                                knows: burn device+Param
pub struct NgramTables { /* rows, momentum, step */ }
impl NgramTables {
    pub fn new(slots: [usize; 3], dim: usize, seed: u64) -> Self;
    pub fn rows_for_batch<B: Backend>(&self, hashes: &[i64], b: usize, t: usize,
        device: &Device, track: bool) -> (Option<Param<Tensor<2>>>, Tensor<3>, Vec<i64>);
    pub fn update(&mut self, uniq: &[i64], grads: &[f32], lr: f32);
    pub fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()>;
    pub fn from_bytes(b: &[u8], slots: [usize; 3], dim: usize) -> Option<Self>;
}
// must NOT know: ByteStream, the trainer, the loop. The hash contract
// ("one column per order, reduced mod slots") belongs in this crate's doc
// comment by REFERENCE to dormouse_data::ORDERS, never copied.

// dormouse-muon-plus += HeadWiseMuon (120)                 knows: burn-optim, MuonPlus
pub struct HeadWiseMuonCfg { pub n_heads: usize, pub weight_decay: f64 }
pub struct HeadWiseMuon;                                    // impl Optimizer
// no dormouse. Its bit-exactness test moves from train/lib.rs:1778.

// dormouse-stability (112)                                 knows: burn only
pub struct StressMonitor;  pub fn new(lr_mult: f64, log_every: usize) -> Self;
                           pub fn observe(&mut self, ce: f32, grad_norm: f32);
                           pub fn report(&self, step: usize) -> Option<String>;
pub fn grad_norm<M: Module>(model: &M, grads: &Gradients) -> f32;
pub fn mask_nonfinite(loss: Tensor<1>) -> Tensor<1>;
```

**How the model passes config.** One `From`/literal per call site, in
`loop_block.rs`, e.g. `LinearCfg { in_features: d, out_features: f, rank,
kind: if use_tsct { Spectral } else { Dense } }`. Four such sites. No mechanism
crate ever sees `DormouseConfig`, and no `DormouseConfig` field is ever read by a
mechanism crate — which is what makes the arms independently testable.

---

## 4. Order of operations

Each commit is green on its own. Gate for every step:
`cargo check -p dormouse-core -p dormouse-train --features dormouse-train/cuda`
&& `cargo test -p dormouse-core -p dormouse-data -p dormouse-train --lib`.
No GPU runs in this plan except where noted; the CUDA `check` does not need one.

| # | commit | what moves / changes | what breaks | gate | parallel? |
|---|---|---|---|---|---|
| **C0** | *in flight, elsewhere* | Land or **revert** the MoR work (§0). Recommendation: revert per D1. | the tree does not compile | same | **blocks everything** |
| **C1** | `refactor!: vendor/burn-fused → dormouse-fused` | `tools/migrate-dormouse-fused.sh`, one shot, 1 commit, pure rename, no behaviour change. | every path dep, every import, the root `exclude` list | script's own steps 5-6 | serial, **needs a quiet tree** (impossible today: 16 dirty files) |
| **C2** | `refactor(core)!: delete the mechanisms that lost` | D1-D4, D7, D8 (≈1 690 LOC out) | `mor_router` field, 3 config keys, 3 core deps, `fnv_hash`, `nano-fused`, the ckpt header dup | same | **WAIT** — touches `schema.rs`/`override.rs`/`validation.rs`/`lib.rs`, the same files the in-flight Engram agent edits |
| **C3** | `feat: dormouse-ema` | `aux.rs` → new crate, `aux.rs` re-exports | 2 call sites (`model.rs`, `train/lib.rs`) | same | **parallel with C4** (different files) |
| **C4** | `refactor!: LinearLike → dormouse-linear` | `git mv param.rs` + import rewrite in `model.rs`, `loop_block.rs`, `gr.rs`, `train/lib.rs` | 6 import sites + 2 tests | same | parallel with C3; **serial** after C2 (same `lib.rs`) |
| **C5** | `feat: dormouse-residual (4 arms)` | `gr.rs` → new crate; `ResidualArm` enum gains ReZero/mHC/AttnRes constructors | the `if use_gr` branch in `loop_block.rs` | same | **WAIT** — `loop_block.rs` is being edited; must land after C4 |
| **C6** | `refactor!: act_quant → dormouse-bitnet` | `act_quant.rs` → `dormouse-bitnet::act`; `schema.rs::ActQuant` becomes a re-export | the 2 `quant_act` call sites in `loop_block.rs` | same | **WAIT** for C5 (same file, same region) |
| **C7** | `refactor!: host tables → dormouse-ngram` | `train/offload.rs` split: tables → library, ckpt plumbing stays | 6 call sites in `train/lib.rs` | same | **WAIT** — `train/src/lib.rs` is being edited (per the brief) and `offload.rs` is dirty; parallel lane with C3-C6 |
| **C8** | `refactor!: HeadWiseMuon → dormouse-muon-plus` | `optim.rs` → library + its test | `optim.rs` re-export, 2 call sites | same | serial after C7 (both touch `train/lib.rs`) |
| **C9** | `refactor!: stress + firewall → dormouse-stability` | `stress.rs` + `mask_nonfinite` | `pub use`, 3 call sites | same | serial after C8 |
| **C10** | `refactor!: the aux losses go home` | `jepa_aux_loss` → jepa, `dspark_aux_loss` → dspark | `aux.rs` + `model.rs::aux_loss` | same | serial after C3 (same `aux.rs`) |
| **C11** | `refactor!: the model names its own params` | routing table → `core/src/routing.rs`; delete `validate_routing` | `optim.rs` + 4 tests | same | **LAST** — it names fields C4/C5/C6 rename |

**Parallel lanes:** the *core lane* (C3→C4→C5→C6→C10) and the *train lane*
(C7→C8→C9) can run concurrently with each other; each lane is serial internally.
C11 waits for both. C1 must precede all of them (it renames the crate paths the
new crates' manifests point at). C2 is a prerequisite for C3-C6 (it changes
`lib.rs` and the config surface).

### Be honest about the cost

- **C1 is a ~2 000-line mechanical diff with zero effect on training.** It is
  worth it anyway, because the `burn-*` names falsely imply upstream crates and
  that misreading has already cost a researcher real time — but it carries a
  nonzero chance of a day of path-dep breakage. It goes FIRST precisely because it
  is the last moment a whole-repo rename is cheap; the script's
  `refusing to run on a dirty tree` guard is the right call and today it is
  correctly refusing.
- **Recommend AGAINST** extracting `model.rs` / `loop_block.rs` into a
  `dormouse-model` crate: a rename with a 617-line diff and zero reader benefit.
- **Recommend AGAINST** moving the config schema to a library: `default() == small`
  is asserted by a test; presets are this model's.
- **Recommend AGAINST** extracting the eval block from `train_loop` into a
  `dormouse-eval` crate: ~100 LOC fused to the stream and the model snapshot, a
  200-line diff for a reader who will never look.
- **Recommend AGAINST** splitting the 583-line `train_loop` body in this pass. The
  audit's P10 R4 violation is real, but the crate split and the body split
  together means every breakage has two suspects. Crate moves first (mechanical),
  body split second (judgment).
- **C4/C5/C6 in one commit instead of three** would save an hour of branch
  bookkeeping and cost an unreviewable diff. Keep them separate: each one is a
  `git mv` plus 6 import lines, which is exactly the size of diff a reviewer can
  verify by eye.

---

## 5. "Maximally easy to train", made concrete

### 5.1 What a newcomer must read

| | files | LOC |
|---|---|---|
| **today** | `lib.rs` 62 + `config/schema.rs` 216 + `model.rs` 337 + `loop_block.rs` 617 + `param.rs` 165 + `aux.rs` 229 + `gr.rs` 117 + `act_quant.rs` 170 + `attention.rs` 33 — **and then** `burn-spectral` (6 172), `burn-kda` (1 208), `burn-engram` (586), and `train/optim.rs` (478) | **1 946** of model code, 8 444 if you follow one hop into the library |
| **target** | `lib.rs` 62 + `config/schema.rs` 216 + `model.rs` 337 + `loop_block.rs` ~450 + `attention.rs` 33 + `routing.rs` 50 — **4 files, one directory** | **~1 150** |

`loop_block.rs` shrinks because the Engram branch (the 45-line `match mem_read` +
the floor) and the residual branch (12 lines) become calls into crates whose doc
comment *is* the explanation. The target is not "less code" — it is **one
directory to read, and every leaf is a named mechanism with a citation.**

### 5.2 The default path (no flags)

`DormouseConfig::default() == small` (asserted by a test). Today five arms are on.
The evidence supports **one**:

| arm | default | the evidence |
|---|---|---|
| `use_kda` | **ON** | Owner decision 2026-09-27: *"KDA is the best linear we have… it stays."* `--no-kda` is a bisect, not a mode. |
| `use_tsct` | ON, **unmeasured** | Never A/B'd (audit §1, AB-PROTOCOL arm 2). It drags the retract cadence, the per-SM Fp8 switch, the `max_ortho` monitor, the one-way fp32 fallback, and the 6 172-line spectral crate. Honest label: *unmeasured*, not *supported*. |
| `use_engram` | ON, **recorded failure + 24h-old fix** | 8M rows/order was 99% of the model, rec → 0.005, held-out frozen at exactly uniform 8.000 BPB (schema.rs:114-122). The in-flight fix caps it at 500k rows and adds a structural floor (`memory_floor_mix`, kNN-LM eq. 3 / FwPKM eq. 12). **It has never trained a step.** First thing to tell a newcomer. |
| `jepa_weight 0.05` + `dspark_weight 0.1` | **should default to OFF** | The single most expensive unmeasured thing in the default: a JEPA teacher is a **second full forward per step** — it is the documented reason `base` OOMs at batch 6 while fitting at 3. Nobody has run the local aux-vs-pure-CE pair (audit §4.2, AB-PROTOCOL arm 1). One-line default change, large expected step-time win, real chance of zero BPB loss. |
| `use_gr`, `use_mor`, `bf16`, `act_quant` | OFF | correct today; `use_mor` should not exist (D1). |
| `host_adam_every = 1` | ON | the report's rule, measured at +2.3% vs 0, and the every-step variant is a **regression that already shipped once** (before 2026-09-04 the tables got 1/100 of their updates). |

### 5.3 What the default costs, and the smallest trainable thing

- `small` = 7.5M backbone + 3×500 000×32 = **48M memory params** = 55.5M total,
  ~1.8 GB fp32. Measured **6.7-8.3 s/step at batch 10 s512 with aux on** (AGENTS
  honest baseline 2026-09-21); the JEPA default-off change removes a second full
  forward from that. `small` batch 10 s512 is VRAM-validated on 16 GB; `base`
  (12.2M) fits at batch 3, OOMs at 6 — *because of the teacher*.
- The KDA backward is ~80% of a step, and each iteration allocates 17 tensors /
  248 MB of scratch: four iterations ≈ 1 GB/step of allocator traffic against a
  7 ms arithmetic budget, with a 465 ms launch-overhead floor below ~80M params.
  **The cheapest big lever is `--max-iter 2`** (AB-PROTOCOL 4b), and
  `--eval-depths` prints the whole depth curve for free.
- **Smallest config someone can actually train today:** the trainer's own
  `test_cfg()` (d_model 128) still inherits the schema default
  `engram_rows = 500_000` → 48M memory params on a 128-dim model, which is why
  `--lib` tests are slow. The smallest *real* run is
  `--preset nano --set engram_rows=4096 --set max_iter=1 --batch 1 --seq-len 128`.
  **Target: ship that as `configs/smoke.toml` + a `--smoke` flag**, so "does it
  build and step" is one command with no knobs and no thinking.

### 5.4 Load-bearing vs research knobs

**Load-bearing — never behind a flag, always in the default, always in the preset.**
If any of these goes missing the run is wrong, not different:

- `d_model / n_heads / head_dim / d_ffn / vocab / max_seq_len` — geometry.
- `use_kda` — the attention arm.
- `rank` — drives the TSCT factor count, the retract cost and GR's bottleneck
  width. **Zero recorded measurement** (audit §4.8: the cheap inner A/B nobody has
  ever run). Load-bearing *and* unmeasured: that is a scheduled A/B, not a knob.
- `max_iter = 4` — a recorded verdict (2.13× faster than 8; iter-8 NaN at step
  193), and the depth-robustness question rides on it.
- `grad_clip = 1.0` — the Muon recipe does not clip (optim.rs:337) but the
  AdamW fallback group has no other bound.
- `retract_every = 1` — without per-step retraction the TSCT factors drift and the
  quantized forward goes NaN.
- `opt = mix` — the Muon+ routing; ADR-0006 exists for it.
- `engram_rows = 500_000` and `engram_lam_max = 0.5` — **stability, not tuning.**
  Without the floor the memory arm eats the loss; without the budget it is 99% of
  the model. Same category as `grad_clip`.
- `host_adam_every = 1` — see above.
- The firewall (`mask_nonfinite` + host-side count + hard stop at 8), the
  `.prev.bin` rotation, the load-time `finite_scan`, and the config drift check.
  All correctness, none of them features.

**Research knobs — behind a flag, default off, queued in AB-PROTOCOL:**
`use_gr`, `--rand-depth`, `--eval-depths`, `--stress*`, `--factors-fallback`,
`--act-quant`/`--act-group`, `--quant`, `--jepa-weight`/`--dspark-weight`/
`--dspark-k`, `--jepa-targets`, `--no-engram`, `--no-kda` (bisect only),
`use_tsct=false`, and any deviation of `--host-adam-every` from 1.
`qk_heads` is already derived from the resolved geometry and is not a knob —
correct, leave it.

---

## 6. What to delete

| # | target | LOC | evidence it is safe |
|---|---|---|---|
| **D1** | `core/src/mor.rs` + `configs/mor.toml` + 3 schema fields + 3 override arms + 5 validation lines + the `mor_router` field | **~250** | (a) the crate does not compile with it; (b) `forward_full_state` has **zero** `route()` calls, so `use_mor` changes nothing about the trained model; (c) PLAN v2.1's own scale gate says "recursion routing: **off at nano/small**", and the paper it cites (2507.10524) reports MoR *below vanilla below 135M* — our scale is 7.5M, **18× below the threshold**; (d) 120 lines of tests for a mechanism that has never run one forward. **Delete, do not move.** `dormouse-mor` (743) stays in the library as the reusable mechanism; if the arm returns it is a 30-line config flag, not 250 LOC. |
| **D2** | `core::fnv_hash` (lib.rs:18) + its test | **13** | Three identical FNV-1a copies (`core::fnv_hash`, `data::fnv`, `shard::fnv`) plus `burn-engram::hasher`. Flagged by the audit on 2026-09-25, unfixed. One hash, one home: `dormouse-data` owns the byte stream, therefore the addressing contract. |
| **D3** | `burn-bitnet`, `burn-cuda`, `rand` as regular deps of `dormouse-core` | **3 lines / 1 815 LOC of compile** | `grep -rn burn_bitnet crates/dormouse-core/src/` → nothing. Same for `burn_cuda` and `rand`. ADR-0017's premise that act_quant "adds the activation side to the quantizers already in burn-bitnet" is **false at the code level** — the two halves share no code, only a paper reference. |
| **D4** | `configs/nano-fused.toml` + `loader.rs::nano_fused_file_parses` | **33** | Its header says `DM_FUSED=1`; `DM_FUSED` does not exist anywhere in `crates/`. Its only reason to exist was "the one sectioned file" — which ADR-0005 itself resolved by making it flat. The test is the sole reader. |
| **D5** | `core/examples/cublas_poc.rs` (759) + `kda_backward_probe.rs` (278) + `gemm_probe.rs` (106) | **1 143** | The audit's kill list #3/#4 already ruled the older scratch examples dead: *"fused/tests.rs + model_seam.rs supersede them. DELETE. Nothing."* These are the same category, newer. `cublas_poc` exists to produce the number now recorded in `memory.md` (cuBLAS f16/fp32 43.7 TFLOP/s vs our cubecl f32 3.5-7.6) — **the number is recorded; the code is not a test.** A measurement you will want to redo belongs in a `benches/` crate with a README naming the question, not in `dormouse-core/examples`. **The largest single deletion in this plan.** |
| **D6** | `act_quant.rs` (170) unless it wins arm 2/12 | **170** | The audit's verdict is verbatim: *"Verified correctness ≠ BPB win… Its only payoff vehicle is fused/ (STE in-kernel). If fused/ dies, this module's speed case dies with it."* fused/ is gone, so the justification is now zero, and the A/B was never run. **Move it, keep it off, and delete it the same quarter if the A/B has not run** — it must not sit in the library as an unmeasured mechanism. |
| **D7** | `jepa_mask_frac`, `jepa_mask_span`, `dspark_stride` config fields (+ override arms, + 2 validation lines, + 4 lines in every snapshot and every preset) | **~10 LOC + 3 config keys** | `jepa_mask_span = 8` and `dspark_stride = 16` have no A/B, no measurement, and no reader other than the loss wrapper. Unlike `engram_lam_max`, they are not load-bearing. Hardcode them as named constants next to the loss they parameterize. **Keep `dspark_k`** — it is the draft depth, a real architectural choice the speculative-decoding story needs. |
| **D8** | the duplicated 24-byte ckpt header parse | **8** | `load_model_weights` (lib.rs:1180-1190) re-implements the header read `load_ckpt_file` (lib.rs:421-437) already does. One `read_header` helper. |
| | **total** | **≈1 620 LOC** | plus 3 config keys and 3 phantom dependencies |

**Explicitly NOT deleted** (listed because the temptation is real):
`train/stress.rs` is not unmeasured research code — it is the instrument that makes
every future stability claim checkable, and the doctrine requires it on every new
config. `train/offload.rs` is the best-evidenced component in the repo (research
§3: SCONE + Engram + Over-Tokenized + Memory Layers). `optim.rs::validate_routing`
goes away only because the routing table moves next to the fields it names — the
check was right, its *location* was wrong. The `--guard` image pinning is the
thing that saved `official_v5f`.

---

## 7. Success criteria

- [ ] `dormouse-core/src` is 4 files + `config/` + `routing.rs` + a 12-line
      `aux.rs`, ≤1 250 LOC (measure: `wc -l`).
- [ ] `grep -c "burn_\|dormouse_" crates/dormouse-core/src/model.rs` shows the
      model composing ≥7 named mechanism crates and implementing none of them.
- [ ] Zero `pub mod` in core without a matching `DormouseConfig` reader (every
      remaining module is either config, composition, or the loop).
- [ ] `cargo tree -p dormouse-core --depth 1` lists 7 library crates, each with
      its own `--lib` test target that runs without `dormouse-core` in scope.
- [ ] `moR` and `DM_FUSED` and `fnv_hash` return nothing from
      `grep -rn --include=*.rs --include=*.toml crates/ configs/`.
- [ ] A newcomer reads `docs/README-model.md` (target ≤60 lines) and the four
      core files, and knows every arm that is on, why, and whether it has been
      A/B'd.
- [ ] `configs/smoke.toml` + `--smoke` train 3 steps on CPU in under a minute.
- [ ] `cargo check -p dormouse-core -p dormouse-train --features dormouse-train/cuda`
      and `cargo test -p dormouse-core -p dormouse-data -p dormouse-train --lib`
      green after **every** commit C1-C11, with no commit that is red between two
      commits.

## 8. Acceptance

- [ ] C0-C11 land in order, each commit green on its own (the two commands above,
      in the same shell invocation as the commit).
- [ ] C1's diff is provably a rename: `git diff --stat C1~1 C1` shows no line
      whose content changed except `Cargo.toml` names/paths and the root
      `exclude`/`[patch]` lists.
- [ ] For each moved mechanism, its library crate has a `#[cfg(test)]` module
      that **does not import `dormouse-core`** (grep-asserted in CI: no library
      crate's test file mentions `dormouse`).
- [ ] The D1-D8 deletions are one commit (C2) with the deletion list in the
      message; `wc -l` before/after recorded in the message.
- [ ] `dormouse-mor` (743 LOC) survives the D1 deletion and still compiles from
      the fork's own workspace.
- [ ] A `cargo test -p dormouse-stability` runs the NaN-firewall contract
      (masked loss ⇒ bit-identical weights) that today only exists inside
      `train/lib.rs`'s 33-line test.

## 9. Risks

| risk | severity | mitigation |
|---|---|---|
| **The rename (C1) lands while another agent holds a dirty path** | high — corrupted `git add`, resurrected paths | the script already refuses on a dirty tree; keep that guard, do not relax it |
| **Field renames break the optimizer routing table** | high — a silent fall-through to AdamW, which is exactly the failure `validate_routing` was written for | do the routing move (C11) **last**, and move the table to `core/src/routing.rs` so it cannot drift from the fields again |
| **`dormouse-residual` grows a 4-arm enum with 4 different state types** | medium — a fat enum and a `match` in the hot loop | keep the arms' *state* in a small `ResidualState` enum; the read/write bodies are already per-arm; measure step time before/after (≤1.2× budget) |
| **A 1 700-line deletion lands in the same week as a 2 000-line rename** | medium — a reviewer cannot hold both | they are separate commits 11 steps apart; the deletion is C2 and the rename is C1, and the rename is the *last* thing allowed to be big |
| **`dormouse-bitnet` becomes a dumping ground** (act + ternary + fastblt?) | medium — recreates the problem ADR-0017 fixes | the crate's rule: one published quantization mechanism per module, each with its own A/B. `fastblt` gets its own crate if it is ever wired |
| **The 41% unwired library (12 797 LOC) is never reduced** | medium — "implemented" and "imagined" stay indistinguishable, which is the confusion ADR-0017 exists to end | publish the inventory (wired / implemented-unwired / unimplemented) in `dormouse-fused/README.md`; an unwired crate must name the A/B that would wire it, or be deleted |
| **The Engram fix (500k rows + floor) is shipped as a default with no A/B** | high — it is 24 hours old and has never trained a step, and it is 86% of the model's parameters | keep it ON (it is the mission) but put the "never trained, unmeasured" line in `configs/small.toml` next to the numbers, and make it AB-PROTOCOL arm 0.5 |
| **Two agents still editing `loop_block.rs` / `train/lib.rs`** | certain | C5, C6, C7, C8, C9 are all marked WAIT. Do not start them until `git status` is clean for those files |

## 10. Critical files

- `crates/dormouse-core/src/loop_block.rs` (617, **in flight**) — the loop; the
  residual and Engram branches leave it, the composition stays.
- `crates/dormouse-core/src/param.rs` (165) — the clearest model→library move.
- `crates/dormouse-core/src/config/schema.rs` (216, **in flight**) — the one
  config seam; `default() == small` is asserted and must stay true.
- `crates/dormouse-train/src/optim.rs` (478) — the routing table that must move
  *into* core before the renames land.
- `crates/dormouse-train/src/offload.rs` (402, **in flight**) — the RAM-offload
  mission, and the undocumented Adam→Nesterov swap.
- `tools/migrate-dormouse-fused.sh` — the C1 script; its dirty-tree guard is load-bearing.
- `docs/adr/0017-dormouse-fused.md` — the decision this plan executes; the
  table in §37 needs three corrections (act_quant's reason, the mHC alternative,
  and the three missing crates: ngram, muon-plus's head-wise Muon, stability).
