# AGENTS.md

This file holds three different kinds of content and keeps them apart on
purpose, because a newcomer cannot otherwise tell which one they are reading:

1. **[RULES](#1-rules--how-to-work-here)** — how to work in this repo. Binding.
   Read this part before your first edit and before your first GPU run.
2. **[MACHINE FACTS](#2-machine-facts--what-this-box-cannot-do)** — what this
   machine and this CUDA backend cannot do, and why. These cost real money to
   discover; none of them is optional knowledge and none may be lost.
3. **[STATUS](#3-status--what-is-true-now)** — what is measured, what was
   retracted, what is broken, what is next. This part goes stale; check the date
   on every number.

**Mission.** dormouse = byte-level LM trainer in Rust (burn 0.22.0-pre.4 +
cubecl CUDA) with max efficiency per FLOP/byte: PonderNet-style adaptive depth
over a **weight-shared loop block**, hashed n-gram memory as input features,
low-rank TSCT experts, and precision spent only where numerics demand it.
Target hardware: 16 GB VRAM (5060 Ti) + 64 GB host RAM. Goal: train the
biggest possible smart model on that box, optionally offloading to RAM without
latency.

**The vocabulary is in [`docs/glossary.md`](docs/glossary.md); the system map is
in [`CONTEXT.md`](CONTEXT.md).** If you are about to use a word, check it there
first — "fused", "arm", "sidecar", "iteration", "group" and "Engram" each have
more than one live meaning in this codebase, and a glossary that disagrees with
the code is worse than none. The list of places where a document and the code
disagree about a name or a default is at the end of the glossary; several of
them are still open.

## Repo layout

- `crates/dormouse-core` — the model: `DormouseModel` = Embedding → `LoopBlock`
  → RMSNorm → `lm_head`, all `LinearLike`. `LoopBlock` per **iteration**:
  controller (sigmoid `w_attn`/`w_mem`/`w_ffn` + softmax expert blend) →
  shared attention (KDA gated-delta, the **only** attention arm — the MSA top-k
  sparse arm and its crate were cut, ADR-0014) + Engram (FNV-hashed n-gram
  tables, behind a hard floor `min(w_mem, lam_max)`) + `n_experts` TSCT
  (spectral low-rank) FFNs → ReZero residual (or Gated Residual when `use_gr`)
  → readout `out_proj`, averaged over iterations. **No halt head and no
  PonderNet** — fixed depth, an honest unweighted CE (ADR-0013; the `p_dist` /
  KL machinery is deleted). Aux heads are ON by default (`aux.rs`: JEPA
  EMA-teacher masked latent prediction + KoLeo; DSpark draft head for next-K
  tokens). Also here: `routing.rs` (id-based optimizer declaration),
  `probe.rs` (per-arm execution counters), `mor.rs` (Mixture-of-Recursions
  arm), `gr.rs`, `act_quant.rs`. Presets are flat TOML files in `configs/`
  (loaded by name or path; schema defaults == `small`; no Rust registry).
- `crates/dormouse-data` — streaming byte stream (next-byte LM, vocab 256),
  64 MB ring refilled in 8 MB chunks, seeded Fisher-Yates file shuffle,
  deterministic train/eval split, parquet/images/binaries, the FNV n-gram
  hashes (`ORDERS = [2,3,4]`), `rewind` (fixed eval window) and `skip_bytes`
  (resume fast-forward).
- `crates/dormouse-train` — CUDA autodiff loop: Muon+ mixed optimizer (the
  live policy is the path-marker implementation in `src/optim.rs`; the id-based
  declaration in `core/src/routing.rs` is exercised by tests only — say which
  one you mean), burnpack `.bin` checkpoints, resume by reusing `ckpt_name`
  (each run also writes `<ckpt_name>.config.toml`; a resume whose resolved
  config differs from the snapshot hard-errors — ADR-0005/ADR-0021. Exception:
  the progress keys `steps`/`log_every`/`ckpt_every`/`eval`/`eval_every` are
  exempt, extending a run is a legal resume), the NaN firewall and gradient
  sanitizer, host-RAM n-gram tables + CPU Nesterov/Sinkhorn updates
  (`src/offload.rs`), stress protocol (`src/stress.rs`), the config seam
  (`src/cfg.rs`).
- `crates/dormouse-cli` — bins `train`, `generate`, `serve` (checkpoints dir +
  `--ckpt-name`, preset). BPB scoring lives in `dormouse-train` (`pub fn bpb`).
- `vendor/burn-fused/crates/*` — **our own** technology library, 28 crates
  (KDA, gdn2, spectral, sct, muon-plus, rmsnorm, bitnet, jepa, dspark, mor, …),
  not a dependency. Every mechanism and kernel that is not the model lives
  here, with its own paper reference and its own A/B (ADR-0017, ADR-0018). A
  `fused kernel` in this repo always means *this* library.

---

# 1. RULES — how to work here

### 1.1 Loud failures; no silent fallbacks (ADR-0011, ADR-0019)

Every degradation of a value, a kernel, a file or a config field carries exactly
one of three marks, and there is no fourth:

| mark | meaning | test |
|---|---|---|
| **LOUD** | hard error naming the cause *and* the escape | the run dies with a message that says what to do |
| **COUNTED** | the fallback happens and a counter/log line says so | a reader of the training log sees it without a profiler |
| **SILENT** | it just happened | **a defect** |

The reason this class is expensive here: **a silent fallback usually computes
the right answer.** A branch is taken, a fallback runs the same function, and the
run is correct but a year slower. Data that does not exist stops the run and is
never synthesized (pretrain v21 trained on constant `b'x'`); shape mismatches
assert; assertions average ≥2 per non-trivial function (NASA P10 Rule 5). The
recovery action for a loud failure is `--guard`.

A fused/accelerated arm must be able to show it ran: the eval line prints the
seam counters (`fused kda=fwd/bwd norm=ran/asked muon_skipped=mom/finalize`),
and `probe.rs` counts every arm entry. If you add an arm, add its counter.

### 1.2 A/B or death (ADR-0002, `docs/AB-PROTOCOL.md`)

Every mechanism beats **its own removal** on held-out BPB at a fixed step budget,
or it is deleted. **A tie deletes the mechanism.** The ladder: 200-500 step
smoke (NaN, speed, early slope) as a filter → 2k+ step confirm for survivors →
long gate for context claims. At our scale a single arm vs a single control
decides nothing: seed variance exceeds the effects measured, so **3 seeds per
arm**, and a win must beat the spread of the control's own seeds.

### 1.3 Zero host-device synchronization (ADR-0018 rule 2)

No CPU in the hot path: no `try_into_scalar`, no `into_data`, no
`blocking_read`, no host-side branch on a device value — not in a kernel, not in
a step. Every host-visible quantity is produced by a device counter or flag and
read at a declared cadence (log steps, checkpoint steps). **Count on the host,
never build a numeric indicator from a bool tensor on device.** The workload is
launch-bound: a per-step scalar read serializes CPU against GPU on every step.
(The one place a device decision is needed is made on the device — e.g. a zero
gradient means zero update, computed with `mask_fill`, ADR-0018.)

### 1.4 Claims must name their evidence (ADR-0018 rule 1, ADR-0020)

A crate may use the word "verified", and may use "bit-for-bit" at all, **only if
it names the external file the reference came from.** Where the reference is our
own transcription, say "transcription"; where none exists, say "no external
reference exists". Same rule for a number: a measurement is a measurement only
with the config, the date and the commit it was taken at. This is the rule that
kills the retracted claims in §3.2 — four of them were not wrong about the
arithmetic, they were wrong about what was being measured.

### 1.5 One heavy thing at a time

After three full-system freezes this is doctrine, not advice: `ram-guard`
(§2.5), a `MemoryMax` cgroup on every train launch, slot caps, persistent logs
outside `/tmp`, and **no two GPU processes at once** — two legal processes
collided once and the cubecl pool wrote garbage weights into a checkpoint.
Builds count. See §2.5 for the preflight numbers.

### 1.6 Git and process discipline

- **One task per worktree** (`tools/wt.sh`, ADR-0022); the worktree is removed
  when the task lands. A shared checkout with N writers makes `cargo test` a
  lottery and `file:line` citations meaningless.
- **Commit per task**, in the repo's style (`fix(core): …`, `land(mor): …`,
  `!` for a breaking change). No `git add .`; stage the paths you touched.
- **No push, no force-push, no destructive git operation** without the owner's
  OK. Reversible local actions are free.
- Do not edit files another agent is in. If a fix can only live in someone
  else's file, report it with `file:line` instead.

### 1.7 One word, one meaning

Use the word from `docs/glossary.md` / `CONTEXT.md`, even when the code's name
is the older one. When a document and the code disagree about a name or a
default, **fix one of them; never invent a third name.** Report the
disagreement — the list at the end of the glossary is a live deliverable, and
it is the same class of defect as a retracted verification claim, in the
vocabulary instead of the numbers.

---

# 2. MACHINE FACTS — what this box cannot do

**Read this section first when a new op misbehaves.** Two facts frame everything
below. (1) The CUDA backend here is the **pliron → LLVM → NVPTX** one, not
NVRTC: `cpp` is an empty, off-by-default feature and nothing in our graph
enables it (`vendor/cubecl-fix/cubecl-cuda/src/compiler.rs:16-32`).
(2) `restrict_to_llvm_backend`
(`vendor/cubecl-fix/cubecl-cuda/src/runtime.rs:420-513`) then advertises only
what that dialect can lower, with the reason in comments. So "the CUDA backend
cannot do X" is almost always "the LLVM dialect has no type or rule for X", and
the file that says so is that one function.

## 2.1 Precision

- **`bool_tensor.float()` is CORRECT on this backend — the old claim was wrong**
  (ADR-0016, 2026-09-27). `.bulba/memory.md:15` says the cast returns 0.0 for
  `true` on cuda; it does not, on either the raw or the dispatch (autodiff)
  path, at n = 1/3/4/8/15/16/17/64/1000, for all-false / all-true / mixed /
  comparison-produced masks, in 1-D and 2-D, and `bool.int()` agrees. What
  actually miscounted the NaN firewall was `clone()` **sharing the CUDA device
  buffer** while `mask_fill` writes in place (fixed 2026-09-27) plus a firewall
  that did not skip the step. **Rule unchanged and now gated: never build a
  numeric indicator from a bool tensor on device; count on the host**
  (`train/src/lib.rs:729-737`). One doc comment still carries the retracted
  claim (`train/src/lib.rs:300-302`) — listed in the glossary, not yet fixed.
  If a new op misbehaves on one backend only, run
  `cargo test -p backend-parity --features cuda --test backend_parity` — that
  gate turns "the cast is broken" into a command instead of a paragraph.
- **NO WORKAROUND: bf16 matmul cannot work on this backend** (ADR-0016 bug 2).
  Not a burn bug: the LLVM dialect has no bf16 type, so `restrict_to_llvm_backend`
  deletes it from the advertised element types on purpose and strips the bf16
  tensor-core families, and even reading a bf16 buffer back as f32 dies at
  kernel-compile time with `Type cube.bf16 does not have a conversion to LLVM
  type implemented`. That is why burn-spectral's own bf16 test fails at
  `burn-cubecl ops/tensor.rs:150`, and why **every bf16 run on this box is
  SLOWER than fp32** — `--bf16` stores bf16 and computes fp32 through cast
  copies, and there is no tensor-core bf16 to compute with. Do not paper over it
  with a silent fp32 fallback dressed as bf16 (that is the ADR-0019 failure
  mode). The one primitive that works is bf16 *storage* as `u16` bit patterns
  (integer ops + bitcast, pinned against f64 in
  `vendor/burn-fused/crates/burn-gdn2/tests/lowp_bf16_cuda.rs`).
- **f16 matmul is CORRECT but silently slow** (ADR-0016 bug 3) — it was reported
  as failing outright and is not. The answer matches fp32 to 1e-2, but the f16
  **tensor-core** candidate dies at compile time with `Expected type
  builtin.fp16 to implement dyn SizedType` and the autotuner falls back to a
  non-accelerated routine without a word. f16 is one interface short: pliron's
  `builtin.fp16` implements `FloatTypeInterface` and not the `SizedType` that
  `cubecl-opt`'s shared-memory sizing queries
  (`vendor/burn-fused/crates/cubecl-opt/src/lib.rs:36-40`). The fix is ~10 lines
  in **cubecl-ir** (impl `SizedType` for pliron's `FP16Type`, legal because the
  trait is local) and belongs upstream; until it lands the 43.7 TFLOP/s cuBLAS f16
  number (memory.md:17) is not reachable *through burn* — route it through cuBLAS
  directly, as `crates/cublas-poc` does.

## 2.2 Shapes and the allocator

- **Never dynamically slice a 4D autodiff tensor**: cubecl on sm_120 crashes
  with `CUDA_ERROR_ILLEGAL_ADDRESS`. This is why `L_Rec` is accumulated inside
  `loop_block.forward_full_state` (per-iteration CE on `[b*t,d]` reshapes)
  instead of materializing `[N,b,t,d]` step hiddens and slicing them. Keep it
  that way.
- **bf16 logits NaN**: the final norm + `lm_head` must run in fp32 even in BF16
  mode (`model.rs` casts to F32 before the head). Loop-internal L_Rec logits
  likewise fp32.
- **cubecl memory pool is high-water**: it reserves pages for every size ever
  seen and never frees them → long runs OOM. `train_loop` calls
  `memory_cleanup()` every 500 steps; `init_pools` (ExclusivePages) must run
  before the first allocation. The pool is NOT enabled by default.
- `LinearLike` pads `out_features` to a multiple of 4 (cubek matmul
  vectorization, N%4==0) and slices back; keep the slice on all new heads.
- The Engram kernels are f32-only (ILLEGAL_ADDRESS on bf16, measured
  2026-08-29) and so is the KDA fused chunked kernel (falls back to tensor ops),
  so both need their inputs cast under `--bf16`.
- Mixed-dtype ops (bf16 activations × fp32 weights, bf16 + f32 adds) NaN on this
  stack. **The rule every forward follows:** cast to fp32 before every Linear,
  compute in fp32, cast the residual/GR writes back to the activation dtype.
  Verified 60 steps, 0 NaN, loss 5.53 → 2.21 (2026-08-29).
- **Molds of the step** (2026-09-27, 7.5M params): fixed per-step cost ~465 ms;
  4× the tokens costs only 1.47× the time; GEMMs are 2-5% of a step;
  opt/retr/ema do not scale with N (84 → 85 ms at 1.6× params) — they are
  launch-bound. Overhead-bound below ~80M params, GEMM-bound above. KDA is
  ~80% of a step: each iteration is a full-sequence gated-delta pass
  allocating 17 fresh tensors / 248 MB of saved scratch, ~1 GB/step of allocator
  traffic against a 7 ms arithmetic budget. **The "~80%" attribution is dead as
  of 2026-09-28**: the attention arm ran no backward in any of these runs
  (`8fa5d4c`, `fused kda=<f>/0`), so 80% of a step that skipped the backward is
  not the cost of training attention — §3.2. The 465 ms floor and the
  launch-bound shape are not affected; only the split is.
- **GEMMs are not where the win is** — measured on this GPU (2026-09-27): cuBLAS
  f16-in/fp32-accumulate 43.7 TFLOP/s, cuBLAS fp32 13.2, our cubecl f32
  3.5-7.6 (TF32 only 17.5 — pointless on consumer Blackwell), a hand-rolled
  WMMA kernel 8.4. Route big matmuls through cuBLAS; do not write kernels. The
  integration crux: a burn tensor's buffer is a cubecl `Handle`, not a raw
  pointer, and `cubecl-runtime` 0.11.0-pre.4 has no pointer-resolution API.

## 2.3 Numerical stability on this box (pre-existing, open)

NaN episodes appear in ANY config (fp32 + adamw included; `--quant fp32 --opt
adamw` 150-step-clean runs were lucky seeds) once the model **overfits hard** —
with the 0.85 MB test corpus loss hits ~0.5 by step 50 and the KDA recurrence
goes NaN (a_log overflow fixed with a clamp in burn-kda 2026-08-29; the deeper
overflow path is not isolated — a bisect showed `--no-kda` stays clean through
deep overfit). Moonshot/FLA decay init (`a_log=-3`, `b_alpha=1.0`) + clamp
measurably extended the clean window (110+ steps, Muon+, fp32, 0 NaN on
2026-08-29, against 45-127 before). Fp8 quant (the sm_120 default) additionally
degrades at overfit (range overflow). This is an overfit-corpus artifact: on a
real loss profile (3+) it should not trigger; verify on real data before chasing
it further. **OOM flood follows NaN episodes** (~10-50 steps later);
`memory_cleanup` right after NaN makes it worse (the pool goes bad), so the
NaN-skip does NOT call cleanup.

- **TSCT ortho is WIRED** (2026-09-02, do not re-investigate). `LinearLike::retract(iters)`
  does a real polar retraction (burn-spectral, keeps autodiff tracking);
  `train_loop` calls `model.retract_tsct(cfg.retract_iters)` every step
  (default every 1 step, NS 3 iters; `--retract-every`/`--retract-iters`), lm_head
  included. Covered by the `tsct_retract_restores_ortho` test.
- **The `max_ortho` metric is per-entry** (Frobenius `‖UᵀU−I‖_F` / rank, fixed
  2026-09-04 after the GPU probe `ortho_probe`): the raw F-norm scales ~k and
  sits ABOVE the threshold from step 0 — the NS-3 retract's own convergence
  floor is ~4e-3 raw (~6e-5 per-entry at r=64), so the old unnormalized check
  fired the one-way fp32 fallback on EVERY fresh run and the factor-quant
  forward (Fp8 on sm_120) never actually engaged in any pre-fix run. Fresh init
  reads ~1.4e-4 per-entry; the 1e-3 per-entry threshold is genuine drift. Above
  it, **all** factors irreversibly switch to fp32, the checks stop, and the latch
  is persisted in the checkpoint. Masters are always fp32 params — `--quant`
  picks the forward path only, so checkpoints are format-agnostic and
  `--quant fp32` after a fallback run reproduces the old behavior exactly.
- **This is also why Muon+ stays off the `[d,d]` projections**: fp32 NS on
  `[768,768]` costs ~40 s/step. The low-rank factors are orthogonalized in the
  factored `[d,64]`/`[64,f]` form, ~1000× cheaper.

## 2.4 GPU/RAM discipline (2026-09-24, after three full-system freezes)

The defenses are installed, not optional.

1. `ram-guard.service` (systemd --user, always on) kills the heaviest work
   process (train/rustc/mold/cargo — never GUI) when available RAM drops under
   8 GB. The desktop can no longer freeze; a runaway run dies instead. Log:
   `/home/sehaxe/logs/ram-guard.log`.
2. Every train launch goes under a memory cap:
   `systemd-run --user --scope -p MemoryMax=40G ./target/release/train ...` — a
   runaway dies at its own cgroup ceiling.
3. An `--engram-ram` run's footprint: 48M slots = **12.3 GB** for the tables +
   Nesterov momentum (256 B/row ×2; the old 18.4 GB figure was the Adam m+v
   layout that `HostNgram` no longer reads — a v1 sidecar is refused loudly).
   8M slots ≈ 2.1 GB. Desktop hours: interactive/verification runs at ≤ 8M slots;
   48M production runs only with the desktop closed.
4. **ONE heavy thing at a time** — including builds (a mold link spikes tens of
   GB). Before any build or train: `free -g` avail ≥ 25 (production: ≥ 40) AND
   `pgrep -ax train` empty. Sequential arms only.
5. Run logs go to `/home/sehaxe/logs/` (persistent). `/tmp` is a 32 GB tmpfs: it
   dies on reboot and silently ate a 33.5 GB sidecar once.
6. Careful with `pkill -f`/`pgrep -f`: the pattern matches your own shell's
   command line (use a character class like `chain[5]0.sh`, or kill by PID).
- **Resume math (2026-09-23, second freeze): a RESUMED 48M-slot run held ~37 GB
  RSS** under the old Adam m+v layout. The layout is now one momentum buffer, so
  the same run should hold ~25 GB (derived from `offload.rs`'s 12.3 GB per copy,
  **not re-measured** — treat it as an estimate and measure before trusting it on
  a 64 GB box). Do not launch a resumed 48M run while the desktop is live.
  Resume/round-trip verification runs at reduced `--engram-slots` (2M ≈ 1.1 GB);
  the round-trip mechanics are slot-count independent.

## 2.5 Build and toolchain

- Machine-local deps are GONE (2026-09-22): `burn-fused` (3.5 MB source),
  `cubecl-fix` (1.8 MB) and `cubek-fix` (added 2026-09-28, `422414c`) are
  vendored under `vendor/` and the workspace root `exclude`s all three (they are
  their own workspace roots; without the exclude, cargo resolves their crates'
  `workspace = true` inheritance against OUR root and fails). The original
  working copies at `/home/sehaxe/burn-fused` and `/home/sehaxe/cubecl-fix`
  still exist but are NOT referenced — edit the `vendor/` copies, the repo is
  canonical and self-contained. No new dependency that duplicates them. Since
  2026-09-24 the repo is on burn 0.22.0-pre.4, and the root
  `[patch.crates-io]` maps **FIVE** vendored crates:
  `cubecl-runtime`, `cubecl-server` (new upstream home of the memory pools since
  cubecl 0.11.0-pre.4 — the stale-page #1401 fix lives here now), `cubecl-cuda`,
  `cubecl-ir` (pliron's f16 `SizedType`, ~40 lines in `src/types/scalar.rs`; it
  is what makes the f16 tensor-core candidate compile instead of being
  silently un-accelerated, §2.1), and `cubek-reduce` (the top-k index kernel,
  ADR-0015 — 0.3.0-pre.4's `reaches` fast-path guard cannot tell an empty top-k
  slot from one holding `-inf`, so a masked score row emits the `u32::MAX` seed
  coordinate as an index and the caller's gather reads out of bounds). The
  count was three until 2026-09-28 and this section said so until then; check
  the root `Cargo.toml` `[patch.crates-io]` before repeating a count, it is the
  only authority.
- CUDA backend is the default via the `dormouse-train/cuda` feature; without it,
  NdArray (fp32 only, no bf16). The GPU binary carries no OpenBLAS: the CPU
  ndarray backend is the optional `cpu` feature (default on for `cargo test`).
- Check: `cargo check -p dormouse-core -p dormouse-train --features
  dormouse-train/cuda`. Tests: `cargo test -p dormouse-core -p dormouse-data
  -p dormouse-train --lib`; bare `cargo test` is safe — `examples/quant_probe.rs`
  is `required-features = ["cuda"]` and is skipped without it (check it with
  `cargo check -p dormouse-core --example quant_probe --features cuda`).
- Release profile: `lto=thin`, `codegen-units=16` (full CPU during builds; the
  loop is GPU-bound, units=1 bought nothing). Build/test speed: deps compile
  opt-3 in dev/test while our crates stay opt-0, mold links (2.42,
  `-fuse-ld=mold`), and `CUDARC_CUDA_VERSION=12050` is pinned in
  `.cargo/config.toml [env]` (fresh target dirs would otherwise die in cudarc on
  CUDA toolkit 13.4); landed 2026-09-22 and cost one full recompile.
- A **worktree off `HEAD` is a different build** if the tree is dirty
  (ADR-0022): uncommitted vendor patches do not travel, and a shared
  `CARGO_TARGET_DIR` converts parallel builds into a rebuild storm (the vendored
  forks are path deps, so their fingerprint is per-worktree). A cold worktree
  build+test costs **≥ 23 min**; branch from a commit that compiles.
- `/home` filling up mid-build is a real failure mode: a mold error that says
  "Disk full?" is the disk. Deleting stale `target/debug/incremental/*` entries
  frees GBs.
- CPU-only tests CANNOT catch the backend-specific class (the raw `fnv as u32
  as i64` cast panics on the Flex CPU backend where CUDA wraps — mask the low
  bits in CPU fixtures). Run the parity gate with `--features cuda`.

## 2.6 Data on this box

- The real corpus is
  `/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real/corpus.bin`
  (46.2 GB UTF-8 after the 500 MB eval carve; raw, unfiltered — the filtered
  copy lives in the sibling `real_filtered/corpus.bin`, 19.7 GB, DCLM-classifier
  0.5 + exact dedup, 272M docs → 104.2M kept). The drive must be mounted.
- Eval is the sibling `real_eval/eval_tail.bin`, a 500 MB tail carve (the old
  30 MB tail is kept as `.30m.bak`; **pre-carve BPB numbers are not
  comparable**).
- **The eval window is `eval_batches × batch × seq_len` bytes**
  (`train/src/lib.rs:1417`), so it is a function of the batch size and two runs
  at different batch sizes scored different amounts of text. Observed: the
  batch-10 recipe reports `over 102400 B`, the batch-2 arm ablations report
  `over 20480 B` — a 5× difference in scored bytes, from `--batch`, a throughput
  knob no reader would expect to change how much text the eval scores. **A BPB
  is only comparable within one window**,
  and `--eval-batches` alone does not fix the window: quoting "20 batches" or
  "100 KB" without `batch × seq_len` is not a number. The formula is right in
  `docs/glossary.md` ("eval tail") and was wrong as a constant here, in §3.1,
  in `docs/AB-PROTOCOL.md:14` and in `README.md` — four documents carried
  "100 KB" for a run that scored 20 480 B.
- **The anchors are a property of the fit, not of the corpus.** The bar moves
  with the fit corpus, the fit size and the scored window; `anchors.rs:22-35`
  says the in-corpus `--holdout` split scores the *trailing quarter of its own
  read* while the trainer's eval reads a fixed window from a different file, so
  **the two were never comparable** — that is the tool's own comment, and
  `--fit` exists because of it. Readings on record, none of them on a trainer
  eval window: unigram 5.398 / 5-gram 2.911 (this file, §3.1 — source not
  named, do not reuse), unigram 5.170 / 5-gram 2.572 (`anchors.rs:3-7` and
  `README.md`, an in-corpus split of a 2 MB read), unigram 5.011 / 5-gram 2.588
  (`--fit` on the filtered corpus scoring the eval tail, `anchors.rs:27-32` —
  the only same-file measurement, but over the whole tail, not over 20 480 B),
  and 2.826 vs 2.849 for the same file at `--bytes 1M` vs `2M`, i.e. the bar
  moves 0.023 on the fit size alone. Uniform is 8.000 by definition. **Get the
  bar with `--fit` on the eval file at the fit size you quote, and print the
  window in the comparison.** What survives from all of it: a model that has not
  beaten the 5-gram line has not learned language, whatever its held-out number
  looks like — and this is the one comparison the window disagreement does not
  affect, because every 5-gram reading (2.588 / 2.572 / 2.911) is **2.1 to 2.4
  BPB below** the best held-out number in the archive. There is no reading of
  the bar that puts 4.997 near it.
- The eval instrument is fixed (2026-09-27): `rewind()` before every eval,
  `eval_batches` windows of `batch × seq_len` averaged, `Module::valid()` (no
  autodiff — 20 never-backwarded graph nodes OOM the card at 15.9/16.3 GB), the
  byte count printed on the eval line. Before the fix, the same checkpoint
  scored 6.443 and 6.551 on consecutive evals, from stream position alone;
  cross-run A/Bs were noise. The eval line also prints `engram=<rows>/<arms>`
  over the eval's own forwards, and that field is the only way a reader learns
  the memory arm did not run — see §3.2.
- Preset param counts are measured on the instantiated model by
  `cargo test -p dormouse-core --test preset_exec` (the widths too wide to
  instantiate on CPU are behind `-- --ignored`). The memory/compute split is the
  column that matters: a preset whose count is mostly rows is a lookup table
  with a model attached.
- Russian docs, read both before touching training code: `bf16_KERNEL_PLAN.md`
  (all-bf16 rules, quant plans), `POST_TRAINING.md` (SFT→RLVR→distill→self-
  evolve loop, EGGROLL).

---

# 3. STATUS — what is true now

## 3.1 Measured (config and date included; never compare across configs)

| what | number | where |
|---|---|---|
| **best held-out BPB with nothing known-broken in it** | **4.997** at step 6500, then regressed to 5.450 by 19500 (best-of a curve that overfits) | `~/logs/train_nokda.log:91`, 2026-09-28. `nokda_ce.config.toml`: `use_kda=false`, `use_engram=false`, `engram_ram=false`, batch 2, seq 512, **depth 2**, 9 195 854 params, 20 480 B window. It is a `--no-kda` run, so it is a statement about a model with **no attention arm at all** — the one number in the archive clean on both §3.2 defects. Below every unigram bar on record, above the 5-gram line by 1.5 |
| second-best held-out, and the only depth-4 number on record | **6.351** at step 1500 | `~/logs/official_v5e.log`, 2026-09-27, 7 526 223 params (pre-repricing `small`; the shipped `small` is 9.20 M), batch 10, depth 4, **102 400 B window** — not comparable to the 4.997 above. `use_kda=true`, so its attention arm got no gradient (§3.2) |
| bench gate row | canary (small, aux off, 2M slots, batch 10 s512) **4081 ms/step**, final CE 5.141 | `benches/history.tsv`, 2026-09-25 `bdddbaf` |
| real18 recipe (small, aux on, 48M rows, batch 10 s512) | 6.7-8.3 s/step | 2026-09-21, after the EMA `no_grad` fix; the older 3.4-3.9 s/step note predates it and is not reproducible |
| per-step cost at 7.5M params | ~465 ms fixed; KDA ~80% of it | §2.2. **Killed as an attribution by `8fa5d4c`**: the arm was not training, so "80% of the step" was 80% of a step that skipped the backward. See §3.2 |
| eval window size | `eval_batches × batch × seq_len`; 102 400 B at batch 10, 20 480 B at batch 2, printed on every eval line | `train/src/lib.rs:1417`, §2.6 |
| a 2k-step A/B run | ~1.6 s/step → 53 min/run, 2.7 GPU-h per arm. **This budget assumed an attention arm that did no backward work**; the re-cost is in §3.3 and is not measured | `docs/AB-PROTOCOL.md` |
| VRAM-validated on 16 GB | `small` batch 10 s512 with a 48M-row Engram; `base` fits at batch 3 and OOMs at batch 6 (the JEPA teacher is a second full forward) | AGENTS history, 2026-09 |
| host-table Adam cadence cost | every-step vs 0: +2.3% step time (10884 vs 10637 ms) — the pipeline drain is the sync wait, not the copy | small/batch3/s512 smoke, 2026-09-04 |
| GEMM ceilings on this GPU | cuBLAS f16 43.7 TFLOP/s, cuBLAS fp32 13.2, cubecl f32 3.5-7.6 | §2.2 |
| eval anchors | **four readings, none on a trainer eval window** — unigram 5.398 / 5-gram 2.911, unigram 5.170 / 5-gram 2.572, `--fit` unigram 5.011 / 5-gram 2.588, and 2.826 vs 2.849 for one file at two `--bytes` | §2.6, `anchors.rs:22-35`. Uniform 8.000 by definition. A bar is a property of the fit, so no anchor here is reusable as published |
| paired-eval resolution | estimated 0.002-0.005 BPB, **not verified** — measure it before believing a win that small | `docs/AB-PROTOCOL.md` |

## 3.2 Retracted — do not cite these

- **Every held-out BPB from a run that had the in-VRAM Engram ON.** Until
  `7adda92`, the eval forward passed `hashed_ids = None` **unconditionally**
  (`train/src/lib.rs`, the `-` line of that commit's diff) while the training
  step passed real keys. `eval_rows` is `Some` only when the host offload is
  live (`lib.rs:1361-1374`), so on the in-VRAM path `loop_block` took its inert
  branch and returned zeros: **the held-out number was a measurement of a
  different network than the one being trained.** `--eval-depths` had the same
  defect and is fixed in the same hunk.
  - **It bites exactly when `use_engram = true` AND `engram_ram = false`** — the
    in-VRAM Engram arm. It cannot bite when `use_engram = false`, because a
    forward with no memory in it is the *correct* forward for a model with no
    memory in it; and it cannot bite under `--engram-ram`, where `eval_rows` was
    already `Some` and the host rows were passed. That scoping is the whole
    finding: this is a retracted arm, not a retracted history.
  - **Touched: 2 of the 12 config snapshots in `checkpoints/`** —
    `engram_ce` (batch 2, depth 2, 25 000 rows) and `small12` (batch 10, depth
    4). `engram_ce`'s **6.453 at step 2000 is invalid**, and so is every other
    number on that curve. For `small12` the config is in the affected class but
    **no eval line for it is on record** in `~/logs/`, so nothing is retracted
    that was not printed; it is listed so nobody reuses its snapshot believing
    the memory arm was scored. The other 10 snapshots are all
    `use_engram = false`.
  - **"The Engram arm lost by 1.46 BPB" is not a verdict, and never was.** The
    comparison was `engram_ce`'s 6.453 against `nokda_ce`'s 4.997, and the
    first of those is a memory-enabled training run scored by a
    memory-disabled evaluation of itself. Note what was *not* wrong with it:
    both runs are batch 2, so both scored the same 20 480 B window at the same
    depth 2. The window matched. **The eval defect is the whole of the
    failure** — there is no second reason to discount it, and no second reason
    to rescue it either.
  - **Not touched: the 4.997 at step 6500.** `nokda_ce.config.toml` has
    `use_engram = false`, so nothing was missing from that eval — the `None` was
    the right answer. It also has `use_kda = false`, so it is clean of the
    second defect below. It is the one held-out number in the archive that is a
    number for the network that produced it, and it stands, with the caveats in
    §3.1 (depth 2, a `--no-kda` model, a 20 480 B window, and best-of-an-
    overfitting-curve).
  - The evidence is a counter, not a reading, and the counter is **newer than
    the defect**: `engram=<rows>/<arms>` was added to the eval line in the same
    commit as the fix (`7adda92`), so **no pre-fix log in `~/logs/` carries the
    field at all** — `train_nokda.log`, `train_kda_full.log` and
    `train_engram25k.log` all end in `muon_skipped=0/0` with nothing after it.
    That the pre-fix eval read `0/0` is `c4214ad`'s assertion, not something
    the logs show. What the logs do show is the post-fix reading on hardware:
    a 20-step run with the Engram on printing `engram=4/4` — 4 rows over 4 arms
    for `--eval-batches 2` at depth 2 (`c4214ad`). Going forward the field is the
    check: `engram=0/<n>` means the eval just ran a memory-disabled forward.
  - **The commit that carries the fix is `7adda92`**, whose message is about
    the best-checkpoint and does not mention it. `b3d6914`'s message *does*
    claim the eval fix and its diff does not contain it; `c4214ad` is the
    no-leak enforcement (`ByteStream::train_and_eval`). Cite `7adda92`. A
    commit message that describes a fix it does not carry is the ADR-0020
    failure in its purest form, and it is why the fix was hard to find.
- **Every held-out BPB from a run with `use_kda = true`.** `8fa5d4c`:
  `chunk_wy_forward_autodiff_s` ran the forward on the **bare** backend and
  wrapped the result in one hand-rolled autodiff node. `OpsPrep::prepare`
  decides tracked-ness from the parents' node refs, and under
  `BalancedCheckpointing` — the trainer's strategy — the op's inputs are
  checkpoint leaves, so the node came back `UnTracked` and its output was a
  leaf. The tensor-ops fallback lives *inside* that node's backward, so a leaf
  means no gradient either way. **The attention arm had no gradient for the
  entire history of this project**: it ran thousands of forwards and trained
  nothing, and every other op built its own graph, which is why the loss curve
  looked healthy and only the counter was wrong.
  - The counter is in this box's logs: `~/logs/train_kda_full.log` prints
    `fused kda=1046/0` at step 500 and `3126/0` at 1500 — 3126 forwards,
    **zero backwards**.
  - So `kda_ce`, `kda_full`, `kda_smoke{,2..5}`, `noengram_ce`, `probe2` and
    `small12` all reported a network whose attention arm sat at initialisation.
    The 6.351 in §3.1 is one of them (`use_kda = true`), which is the *second*
    reason that row is not a depth-4 quality result.
  - **This also explains a speedup that was never a speedup**: the fused
    forward made attention look nearly free (+22 ms) precisely because it was
    doing no backward work at all. A fused path that skips the backward is not
    faster than a tensor path that runs it; it is a different program.
  - **The fix is verified to compile and NOT verified to train.** `8fa5d4c`'s
    own message says the gradient-flowing check is the next step and is not
    done; `DM_GDN2_BWD_TRACE=1` should print `ENTERED` on `ChunkWy::backward`,
    and that line appears in no run before it. Treat "the attention arm trains
    again" as **unverified** until a run shows it.
  - Cost of the working backward, as reported by the 2026-09-28 audit and
    **not reproduced here, with no committed log and no `benches/history.tsv`
    row**: the tensor-op path at batch 8 measured **25.8 s/step** against
    **3076 ms** without the arm. If that holds, the A/B budget in
    `docs/AB-PROTOCOL.md` (53 min per 2k-step run) is wrong by more than an
    order of magnitude and the queue has to be re-costed before it is run. The
    `4628`-era note that the same op costs `3628 ms` fwd+bwd (`README.md:77`) is
    a different shape and is not a substitute.
- **Every parquet corpus result before `f6ab353`.** `Source::read`
  downcast the *top-level* columns to `StringArray` and dropped everything else
  with no counter. `mix/qa` nests its text (`document: struct { html, title,
  url, tokens }`), so the only top-level string was `id`. Measured on one 200 MB
  shard asking for 8 MB: **21 811 B before the fix, 8 000 000 B after** — and
  the bar tool printed anchors either way ("unigram 3.544 / 5-gram 3.278" on
  21 KB of UUIDs, 367× short). **Every parquet run trained on a fraction of its
  corpus and reported the loss curve as if it had not.** A shard that yields no
  text is now counted by name rather than vanishing.
- **"fp4 + group 128, 100 steps, 0 NaN, convergence == fp32" (act quant)** —
  retracted twice over, for two independent reasons, and the second is the one
  that makes the number meaningless rather than merely partial. (1) The
  attention upgrade is unconditional — `ActFormat::attn()` maps `Fp4 -> Int(8)`
  and `Int(b) -> Int(b.max(8))` — so `--act-quant fp4` has **never** run 4-bit
  attention; that verified the FFN path only. (2) `9b343d3`: **`fp4` was not
  e2m1 at all.** Real e2m1 has no 0.75 (the grid is 0, .5, 1, 1.5, 2, 3, 4, 6),
  the mantissa rule emitted 0.75 for `a ∈ [0.625, 1)`, and the caller scaled
  each block's max onto **1** rather than onto the format's max (6) — so only
  `{0, 0.5, 0.75, 1}` of the eight magnitudes were reachable. That is a
  ~3-level quantizer wearing a 4-bit label, and 1.5…6 was dead code. Every
  `--act-quant fp4` number before that commit measured the 3-level thing. The
  `4` and `8` paths are untouched and pinned bit-identical
  (`int4_levels_are_the_whole_symmetric_range`).
- **All Gated Residual numbers — of which there are none.** Worth stating
  precisely, because "retracted" would over-claim: `use_gr = false` in all
  eight `configs/*.toml`, so no preset, log, checkpoint or A/B number is
  invalidated. What *was* wrong is in `9b343d3`: the write omitted the sigmoid
  **and** the factor 2 of Eq. 33 while the comment above it wrote out the
  equation the code did not implement; the read omitted the `1/nr` that sits
  inside the SiLU (Eq. 31), giving a gate saturated at init; and the readout was
  taken from the state *before* the write, so GR was a depth-(iters−1) model
  and at `max_iter = 1` the entire block body was computed and discarded. The
  report's −0.026 and its zero-spike result cannot transfer in any case, because
  the report puts one GR per attention and MLP sublayer of a 56-sublayer stack
  and we run one weight-shared block recursively (§3.5). **The arm has still
  never been A/B'd, which is the debt this does not discharge.**
- **"fused is 1.7-2.0× faster than burn" (ADR-0003).** Retracted in ADR-0009. At
  flagship the whole-loop fused op measured **9.21 s/step against burn's 7.05**
  — 1.3× *slower* (`research/2026-09-23-fused-flagship50.md:52-53`). The
  `fused/` module and the `DM_FUSED` switch are now **deleted**; its kill
  switch in ADR-0009 is moot. What survives under the name "fused" is the
  library's own fused kernels, which are live and counted.
- **"the bool→float cast returns 0.0 for true on cuda" (memory.md:15).** Wrong;
  the miscount was `clone()` aliasing plus a firewall that did not skip the
  step (ADR-0016). One code comment still says it — see the glossary.
- **Every train-CE reading from the PonderNet era.** It measured λ-collapse, not
  learning (ADR-0013).
- **Every aux-vs-pure-CE A/B conclusion before 2026-09-27.** The EMA teacher was
  being fed the label sequence, so the JEPA target was wrong in every run since
  the aux heads shipped. DSpark's window tokens have the same one-position shift
  and are **not** fixed — named, not silently carried.
- **The `max_ortho` fallback story before 2026-09-04**: the fp32 fallback fired
  on every fresh run, so no pre-fix run actually used the factor-quant forward.
- **The 2026-09-27 `--no-kda` 956 → 188 ms ablation**: measured alongside a live
  run, so it is a ratio, not an absolute. Step-time claims need same-shape
  measurement on a quiet GPU. It is also the *only* step-time measurement of
  removing the attention arm, and the arm was not training at the time, so it
  does not price the arm the fixed one costs.
- **Every A/B verdict in `docs/AB-PROTOCOL.md`.** Not one arm has been judged.
  The instrument was wrong for the Engram arm (above), the attention arm in
  every control was frozen at initialisation (above), and no two runs in the
  archive share a window size. The queue is a list of experiments to *run*, not
  results, and its cost line is void.
- **`small` = 7.5M / `base` = 12.2M params.** Pre-date the 2026-09-27 memory
  re-pricing; the measured numbers are in `tests/preset_exec.rs` and the README
  table, and the 7.5M figure is still hardcoded as a test constant
  (`loop_block.rs:706`) and asserted in a doc comment (`schema.rs:130`).
- **"the fused KDA op is 11.2x faster than PyTorch on fwd+bwd" (2026-09-28).**
  Only the FORWARD claim survives. `alloc_probe` and both benches build their
  device as `Device::autodiff(...)` = NoCheckpointing and enter the node with
  `B = Autodiff<..>`, where the adjoint's `TypeId` gate at
  `chunk_adjoint_cube.rs:399` returned `None` — so the **tensor** adjoint ran
  and the measured fwd+bwd time contained it. Worse,
  `fused_chunk_verify.rs:132` (`fused_op_grads_match_tensor_path_cuda`)
  compared that tensor adjoint against the tensor path, i.e. **verified the
  tensor adjoint twice**, and the only caller of `fused_chunk_backward` in the
  whole tree is `bench_fused_bwd.rs:146` — bare tensors, inside a timer, result
  discarded. The fused adjoint kernels have therefore never been numerically
  compared to anything. A counter placed BEFORE that gate is what let the CUDA
  gate test assert `bwd > 0` and pass; it now sits after the gate (`f737710`),
  so the test is **deliberately red** until the backward gate lands. Do not
  quote a fused fwd+bwd number until the new gradient comparison is green.

## 3.3 Broken, open, or undocumented

- **The eval's memory arm has no test.** `b3d6914` says it in its own message:
  the eval call site is inside a 1400-line function and is not callable from a
  test, so the half of the fix that mattered — passing the keys the training
  step passes — is carried by the code and by the `engram=` field, and nothing
  would fail if it regressed. The decode half *does* have one
  (`decode_seam`, `decode_wiring`, both on the CPU backend). The cheapest real
  gate is to assert that the eval line's `engram=<rows>/<arms>` field is
  non-zero whenever `use_engram` is true, on a 2-batch CPU run.
- **Whether the attention arm trains is unverified.** `8fa5d4c` compiles and
  says so; the `DM_GDN2_BWD_TRACE=1` `ENTERED` line on `ChunkWy::backward` has
  not been seen. Until it is, §3.2's second retraction stands for the *present*
  code too, not only for history, and every cost estimate that assumed a
  working attention backward is unmeasured.
- **The A/B budget has not been re-costed.** `docs/AB-PROTOCOL.md` still prices
  a 2k-step arm at 53 min, derived from ~1.6 s/step on a run whose attention
  backward did not execute. The 25.8 s/step batch-8 figure that would replace
  it has no committed log and no `benches/history.tsv` row, so the honest state
  is **the cost of one A/B arm is currently unknown**, not "2.7 GPU-h".
- **Four silent-data-loss fixes with no gate** (`7adda92`): a resume overwrote
  the best checkpoint because `best_eval_bpb` was a local; `.best` was a
  different model because the `.ngram` sidecar did not travel with the weights;
  `inf` was printed as a score; and the sidecar is read as `None => h`, i.e.
  freshly seeded rows, when absent. All four are fixed, none has a test, and the
  commit says so.
- **A commit message described a fix its diff did not carry.** `b3d6914`'s
  message spends a paragraph on the eval-keys fix; the fix is in `7adda92`.
  Anyone citing the fix from the message would cite the wrong commit, which is
  how a "verified" claim becomes unverifiable. Check the diff, not the message
  — and when a message and a diff disagree, the disagreement is a defect in its
  own right.
- **Two implementations of the optimizer policy.** `train/src/optim.rs` builds
  the optimizer from path-string markers and is what runs; `core/src/routing.rs`
  declares the same policy from `ParamId`s and is exercised only by tests.
  They agree today. `GroupCounts` is declared in both crates. Until this is
  reconciled, always say which one you mean.
- **The fused RMSNorm kernel never engages on the trainer's backend** — the eval
  line shows `norm=0/N` because an autodiff tensor cannot be handed a bare
  kernel. Counted, not silent; a fix belongs in the library, not in a gate here.
- **3 SILENT fallbacks remain** in ADR-0019's enumeration (41 sites total: 17
  LOUD, 11 COUNTED, 13 SILENT, of which 10 have been fixed and 3 are
  proposals; the table lists every site by file:line). The notable ones:
  `aux_loss` returning `None` for a missing teacher latent, `dspark_aux_loss`
  contributing nothing when the sequence is shorter than the draft window, the
  pool install result being ignored, and a non-statted shard counting as 0
  bytes in the size floor. **A fourth, and the largest instance of the class in
  this project's history, was found on 2026-09-28 and is fixed**: the held-out
  eval passed `hashed_ids = None` (row 41, `7adda92`). It survived this long
  because row 40 classified the `None` hash-key arm as "correct by
  construction — inference without hashed ids", which is true for a decode path
  and false for a measurement. The correction, and the rule it generalises to,
  are in ADR-0019: **a fallback is not excused by having a defensible meaning,
  it is excused by the caller being able to tell the reader which arm ran.**
- **`StressMonitor` state is lost on resume** (ADR-0021 item 4): the 201-entry
  window, the spike count and the p99.9 restart empty. It corrupts the *report*,
  not the objective. Fix: `to_bytes`/`from_bytes` plus a fourth container
  section or a `<name>.stress` sidecar, ~20 lines.
- **The `.ngram` sidecar carries no training-step stamp** (ADR-0021 item 7): a
  crash between the model write and the sidecar write leaves a mismatched pair
  that nothing can detect. The fix is a `train_step` field in the header and a
  refusal to load a mismatched pair — deferred because it invalidates an 18 GB
  production artifact and that wants its own decision.
- **DSpark's window is one position shifted** (`model.rs:197-203`): the draft
  head's step *s* is fed `x[p+s+1]` and trained to emit `x[p+s+2]`. Fixing it
  changes every DSpark number, so it needs its own A/B.
- **`dspark_stride`** is a config field with no documented meaning anywhere.
- **Adaptive depth is only an arm**: `--rand-depth` (sample `T` per step) and
  MoR (rank the slots per position) both exist, are mutually exclusive by a
  loud refusal in `resolve`, and neither has been A/B'd against fixed depth.
- **`--gen-max-iter` does not exist**; a flag help string advertises it. The
  CALM-lite confidence exit from ADR-0013 is not implemented either.
- **The vocabulary itself** — `docs/glossary.md` ends with 20 code-vs-document
  disagreements: 6 fixed in this commit, 4 half-fixed (what remains is in
  `README.md` or a `crates/` help string another owner holds), 10 open. The ones
  that still need a decision: three stale PonderNet references, the "CPU Adam"
  name for a Nesterov+Sinkhorn update, the `small` param count the memory budget
  is argued from (`loop_block.rs:706`, `schema.rs:130`), a retracted precision
  claim still living in a code comment (`train/src/lib.rs:300-302`), the two
  competing routing implementations, and the `nano-fused` preset named after a
  deleted module.

## 3.4 Next, in the order the evidence says

**Precondition on the whole queue, added 2026-09-28: the control run has to be
re-baselined before any arm is judged, and it has to be a run in which the
attention arm actually receives a gradient.** Every run in `~/logs/` predates
`8fa5d4c`, so every control on record is a network whose attention arm was
frozen at initialisation, and the Engram arm's only comparison was scored by a
memory-disabled eval (§3.2). Two arms below are therefore not merely unrun, they
are **undefined against the old control**: arm 5 (KDA decay form) presupposes
KDA trains at all, and the hashed-memory arm presupposes an eval that passes the
keys. The `engram=` and `fused kda=` fields on the eval line are the cheapest
way to confirm a new control is clean — read them before the first A/B number
is believed.

The A/B queue with flags, costs and what each arm decides is
[`docs/AB-PROTOCOL.md`](docs/AB-PROTOCOL.md). Summary: control → pure CE (do the
aux heads earn their share of the step) → dense FFN (do TSCT, the retraction
and the quant machinery earn ~1000 lines) → working set (4 epochs over 4.8 GB
vs one pass over 19 GB) → rand depth → **depth 2 vs 4**, the cheapest big lever
→ KDA decay form (a technology REPLACE, not a knob) → hashed memory
(25_000 rows/order, the 24% operating point) → the memory capacity ladder.
Every arm is 3 seeds, 2k steps, pure CE, at the program's operating depth.
**The per-arm cost is unknown, not 2.7 GPU-h** (§3.3), and every arm must be run
at one batch size so the eval window matches across seeds and arms (§2.6).

Also queued: the fusion backend flip (parked — under fusion, burn's `Tensor`
becomes the dispatch type and the vendored crates downcast the bare
`CubeBackend`, 48 × `DispatchKindConversion` unsatisfied, ADR-0018/PLAN), the
fused-kernel rungs, the official 100k-step baseline, and the code-domain corpus
mixture → SFT → RLVR loop (`POST_TRAINING.md`).

## 3.5 Design playbook: what to adapt from Qwen3.8-Flash-Next (tech_report.pdf)

The report's architecture (125B MoE, 6B active, 51B n-gram params
off-accelerator) is the closest published match to dormouse's goals. Ranked by
payoff for a 16 GB GPU + 64 GB RAM box. **Status per item in brackets:**

1. **RAM offload via host-prefetched n-gram tables (the big one)** [partly wired:
   `--engram-ram` works, the Nesterov+Sinkhorn update replaced the report's Adam].
   Qwen stores 51B n-gram embedding params in host memory: deterministic hash
   addressing → async prefetch of the next batch's rows (pinned memory, separate
   CUDA stream) overlapped with compute, so host RAM is a latency-free extension
   of VRAM. dormouse's Engram already uses deterministic FNV keys — scale the
   tables up (millions of slots) and keep them in RAM, prefetching per batch.
   Rule from the report: n-gram tables train with Adam, weight decay disabled
   (ours: plain Adam in-VRAM; Nesterov + Sinkhorn on the host path). Loss drops
   monotonically with vocab scale; downstream saturates — evaluate both.
2. **Gated Residual (GR)** [wired, never A/B'd, and the code did not match the
   equations until 2026-09-28]. Widen the residual stream to 4 branches; read =
   elementwise sigmoid gate on a low-rank bottleneck (rank d/8) over
   group-RMSNorm'd branches; write = one scalar per branch; no branch-mixing
   operator. Replaces ReZero scale and pre-norm. Report: −0.026 loss at 276B
   tokens, zero loss spikes at 4× LR, no qk-clip/SwiGLU-clip needed.
   Do NOT use sparse writes (top-2 branches): fine in pretraining, degrades
   post-training. Shipped as the `use_gr` config flag (off by default for
   checkpoint compatibility; `crates/dormouse-core/src/gr.rs`, all slicing
   2D/3D — a 4D gate tensor would crash sm_120), convergence-tested on NdArray.
   Also `GatedNorm` (RMSNorm ⊙ σ(W2 SiLU(W1 RMSNorm(u)))) helps everywhere.
   **The audit of 2026-09-28 found four defects, all fixed in the same commit:**
   the write was `(1/nr)W_w vec(R̂)` with **no σ and no factor 2** (Eq. 33 is
   `2σ((1/nr)W_w vec(R̂))`) while the module doc sold the sigmoid as the
   stability mechanism — `s` was unconstrained in sign, so a block could cancel
   the branch it wrote; the read was missing the `1/nr` **inside the SiLU**
   (Eq. 31), giving 4× the gate pre-activation and a gate saturated at init
   instead of near 0.5; and **the readout was one iteration behind** — the
   per-iteration output came from the state *before* `write`, so GR was a
   depth-(iters−1) model and at `max_iter=1` the whole block body was
   discarded. Eq. 32 was the one of the three that was right. `gr.rs` now pins
   Eq. 31/32 and Eq. 33/34 against a host reference, and
   `crates/dormouse-core/tests/gr_seam.rs` pins the loop's ordering (the output
   must change when the block body changes, at every depth). **Placement is
   still ours, not the report's**: one GR per loop ITERATION of a weight-shared
   block, with the iteration embedding added to the read, against a separate GR
   per attention and MLP sublayer of a 56-sublayer stack. A transposition, so
   the report's −0.026 and its zero-spike result do not transfer unmeasured.
   **Every GR-shaped number before this commit is invalid**, and there were
   none: `use_gr = false` in all eight configs, so no preset, log or A/B ever
   ran it (`docs/audit-2026-09-25.md` already ruled it "A/B or delete").
3. **Muon+** [wired, the default optimizer]. Muon + post-polar ColRow
   normalization, fused CUDA kernels, hybrid 2D→Muon+/1D→AdamW, with the
   report's param-group routing: Muon+ ColRow on 2D linear maps; AdamW for
   embeddings, output head, controller/router, low-rank projections
   (orthogonalization hurts them); n-gram tables on Adam with no weight decay.
   `ns_steps=8` (the report's stability choice; Muon+'s default is 5).
   `validate_routing` re-checks the policy against the live module tree at
   startup — a stale marker (module renamed) fails loudly instead of silently
   falling back to AdamW. Report deltas still open: split fused params (qkv
   per-head, SwiGLU gate/up halves) BEFORE orthogonalization — fusing mixes
   singular directions (our Q/K head-wise group is the partial version of this);
   capture the step in a CUDA graph. Skip the report's Polar Express schedule and
   γ = 0.2·max(A,B): Muon+ ships its own validated recipe. Fallback-group
   alternative: Adan via `--opt adan`/`mix-adan` (arXiv 2208.06677, ships in
   burn-optim). bf16 plan targets Muon moments bf16 + stochastic rounding, MuonQ
   4-bit — blocked by §2.1.
4. **Sparse attention indexer (QSA → MSA) — RE-ENTRY PATH, the code is DELETED**
   (ADR-0014). Re-add a *working* sparse attention from FLA / flash-attn, then
   train its block-indexer in two stages: (a) distill the dense attention
   distribution (max-pooled to blocks, KL) into the indexer at high LR for ~1k
   steps, (b) then joint sparse training. Indexer recipe: MQA (4 q-heads, 1
   shared k-head), avg-pool keys into blocks r=4, block-causal ReLU scoring,
   top-KB blocks + always include the tail tokens of the final incomplete block,
   partial RoPE. Reuse top-k indices across MTP/speculative steps. At 1M ctx:
   7.6× prefill / 4.9× decode vs dense. The MSA crate that lived here emitted
   garbage indices on pre.4 and was deleted rather than disabled; re-entry
   needs an implementation that works, not this one.
5. **GDN details for KDA** [the hybrid shape dormouse already has]. Sigmoid
   output gate (not SiLU), zero-centered RMSNorm everywhere, L2-normalized q/k,
   decay α_t = exp[−exp(A)·softplus(Wα x + bα)]. One full-attention layer per 4
   (the per-iteration router blend covers this; keep RoPE in the attention arm —
   NoPE → endless generation after post-training). `use_short_conv` (the
   report's local-bias choice) was tried and REVERTED: it made fp32+AdamW NaN
   at ~step 60 on this box while the same recipe ran 150+ steps clean without it
   (measured 2026-08-29) — trace inside burn-kda before re-enabling. KDA state
   dtype follows the inputs (bf16 under `--bf16`, fp32 otherwise — Moonshot
   FlashKDA trains bf16 state); decay init matches the Moonshot/FLA recipe
   (`a_log=-3`, `b_alpha=1.0` — conservative alpha~0.08 at start, and the open
   question in the A/B queue). FlashKDA math (K3 decay, chunked WY, chunk 16)
   already matches burn-kda; their CUTLASS kernels are the remaining perf delta.
6. **Hyperparameters & stability**: with Muon+GR, batch-size warmup is wasted
   (−18.8% extra steps, no gain) — start at target batch/LR; optimal LR/batch
   shift up. Refit scaling laws per architecture change (the report's new recipe
   ends 7.8e-3 better than old-recipe hyperparams at the same budget).
   **Stress-test protocol** (cheap, do it on 16 GB): constant LR at 2× and 4×
   optimal, count loss spikes (> 201-step rolling median + 0.1), p99.9 pre-clip
   grad norm, per-block activation max. Stability = gate/bounded activations,
   not clipping. Wired as `--stress --stress-lr X` in `train/src/stress.rs`.

## 3.6 Design playbook: energy-based models (research pass 2026-09-04)

**Nothing here is implemented.** Papers surveyed: **EBT** "Energy-Based
Transformers are Scalable Learners and Thinkers" (arXiv:2507.02092 — energy head
over (context, candidate) pairs, inference = energy minimization; 35% faster
scaling w.r.t. data/depth/params/FLOPs, +29% more gain from inference compute
than Transformer++; 44M AR EBT on RedPajama-32B beats Transformer++), **NRGPT**
(arXiv:2512.16762, ICLR 2026 — GPT as an EBM: per-token energies, weight-shared
recurrent refinement with per-head D×D J matrices, generation = exploration of
the energy landscape that provably reduces to gradient descent; notably
resistant to overfitting), **EDLM** (NVIDIA, ICLR 2025 — energy-corrected
logits for diffusion LMs), **Residual EBM** (Meta — sequence-level EBM residual
over a frozen base LM), **EBRM** (arXiv:2504.13134 — energy-based reward
models, energy-guided reward refinement), **EB-JEPA** (arXiv:2602.03604 — JEPA
*is* an EBM in LeCun's framing: energy = embedding-space prediction error).

dormouse's LoopBlock is already structurally an EBT (weight-shared recurrent
refinement + adaptive depth); what's missing is the explicit energy readout.
Ranked applicability:

1. **Energy head on the loop state (EBT-style, the direct fit)**: scalar head
   E(h_k, e_v) scoring the loop state against candidate token embeddings, trained
   InfoNCE-style — target byte vs negatives sampled from the lm_head softmax or
   uniform. No MCMC anywhere: the candidate set is the 256 bytes, enumerable —
   this is just a contrastive head, the same machinery as the existing DSpark
   aux. Gains: calibration/OOD signal, a principled inference-time compute knob
   (more iterations on hard contexts), a possible regularizer for the overfit-NaN
   episodes. Cost: one small head + 256 scores per position per iteration.
   Implement as an `aux.rs`-style objective first, weight 0 default, A/B against
   pure CE.
2. **Energy-based halting**: PonderNet's λ_k was a learned proxy for "the loop
   stopped improving". The EBM version: halt when ΔE between iterations plateaus.
   Cheap first step — log ΔE per iteration and check it correlates with the
   deleted λ_k; only then A/B as an actual halt criterion. (ADR-0013 is the
   standing objection to learned halting: fixed depth won twice.)
3. **Post-training EBM reranker (EBRM / Residual-EBM)**: in `POST_TRAINING.md`'s
   self-evolve loop, K=8 noisy rollouts are currently picked by the PonderNet
   Q-head — an EBM scorer (segment- or sequence-level) is a drop-in upgrade for
   best-of-n selection, and EBRM-style energy refinement applies to RLVR rewards.
   Zero changes to the base model; train the scorer on verifier labels. Note the
   Q-head it refers to was deleted with PonderNet: the selection rule needs
   re-specifying first.
4. **Do NOT adopt**: MCMC/CD training (Langevin/HMC over byte sequences — slow
   and unstable on this stack), NRGPT's full re-architecture (per-head D×D J
   matrices are VRAM-infeasible at dormouse scale and ~double inference FLOPs —
   anti-goal), diffusion-EBM (dormouse is AR). EDLM's energy logit correction
   only matters for diffusion, but its "energy as verifier" reading is what item
   3 uses.

## 3.7 The interface, as it stands (every knob is a typed CLI flag; `--help`)

- `--bf16` — bf16 activations/weights (weights fp32→bf16 cast per forward).
  Off by default; CPU runs stay fp32 (burn-ndarray has no bf16). §2.1 is why
  this is slower than fp32 on this box.
- `--no-kda` / `--no-engram` — disable an attention/memory arm (bisect/A-B;
  presets set `kda`/`engram` true). There is no `--no-gr` or `--no-mor` flag:
  those are `--set use_gr=false` / `--set use_mor=true` (config, not A-B flags).
- `--quant fp32|bf16|fp16|fp8|fp4` — force the TSCT **factor** format. Default
  `None` = auto (`lib.rs:359`): bf16 mode → Bf16, else sm ≥ 120 → Fp8, else
  Fp32. The forward path only; masters stay fp32. An unknown string is a loud
  panic, not a silent fp32.
- `--act-quant 4|8|fp4` + `--act-group N` — BitNet a4.8-style **activation**
  quantization (STE, f32 graph; `core/src/act_quant.rs`): FFN activations at the
  format, attention path at max(bits, 8). **CORRECTED 2026-09-27: the attention
  upgrade is unconditional — `ActFormat::attn()` maps `Fp4 -> Int(8)` and
  `Int(b) -> Int(b.max(8))` — so `--act-quant fp4` has NEVER run 4-bit
  attention.** The earlier claim "fp4 + group 128, 100 steps, 0 NaN, convergence
  == fp32" verified the FFN path only. **CORRECTED AGAIN 2026-09-28: `fp4` was
  not e2m1 either.** The mantissa rule emitted 0.75 (not a level of the format)
  and the caller scaled each block's max onto **1**, so only {0, 0.5, 0.75, 1}
  were ever reachable — a ~3-level quantizer with a 4-bit label, whose 1.5/2/3/4/6
  levels were dead code. Fixed: the grid is the real e2m1 (`E2M1`, 8 magnitudes,
  16 codes) and the block scale maps onto the FORMAT's max (`ActFormat::max_value`
  = 6). **Every `--act-quant fp4` number before 2026-09-28 is INVALIDATED** — it
  measured a 3-level quantizer. The `4` and `8` paths are bit-identical
  (`int4_levels_are_the_whole_symmetric_range` pins that).
- `--opt mix|mix-adan|adan|adamw|muon` — optimizer (default `mix`; see
  `optim.rs`).
- `--factors-fallback` — drop expert TSCT u/v factors from the Muon+ group to
  the fallback optimizer.
- `--retract-every N` / `--retract-iters K` — TSCT U/V ortho maintenance cadence
  (default every 1 step, NS 3 iters; `max_ortho` is monitored every 500 steps,
  and above 1e-3 per-entry ALL factors irreversibly switch to fp32 — one-way,
  checks stop after the fallback, and the latch is persisted in the checkpoint).
- `--stress --stress-lr X` — stability stress protocol (report §3.3): constant LR
  at X× the base, loss-spike counter (201-step median + 0.1), p99.9 pre-clip grad
  norm; `--stress-every` log cadence (default 50).
- `--engram-ram [--engram-slots N]` — host-RAM n-gram tables (report §2.3):
  millions of rows in the 64 GB RAM, CPU Nesterov + Sinkhorn, only the batch's
  rows copied to the GPU (~600 KB/step). The in-model table is squeezed to 1 row
  per order on this path (`cfg.rs:50`) because it is never read. Default slots
  1 000 000; the real18 production recipe was 48M rows.
- `--host-adam-every N` — host-table update cadence (default 1 = every step, the
  report's rule; 0 = off). The flag name says Adam; the update is Nesterov +
  Sinkhorn (glossary). Each update syncs the row grads D2H; the loss-scalar read
  rides along. Until 2026-09-04 the update ran only on log steps (a
  sync-deferral side effect, tables got 1/100 of their updates) — don't regress
  it; trade with `--host-adam-every 5` if a run is step-time-bound.
- `--rand-depth` — sample the loop depth `T` in 1..=max_iter per step, a pure
  function of the step index. Mutually exclusive with `use_mor` (refused loudly
  in `resolve`). `--eval-depths` prints the held-out BPB at depths 1..=max_iter
  at every eval, training nothing — that curve is the deliverable for any depth
  claim.
- `--eval-batches N` (default 20) — batches averaged per held-out eval; part of
  the config snapshot, because a number without its protocol is not a number.
  **The window is `N × batch × seq_len` bytes, not a fixed size** — 102 400 B at
  batch 10, 20 480 B at batch 2 — so this flag alone does not make two runs
  comparable. The byte count on the eval line is the authority; quote it (§2.6).
- `--jepa-weight` / `--dspark-weight` / `--dspark-k` — auxiliary objectives on top
  of CE (`core/src/aux.rs`, ON by default at 0.05 / 0.1 / K=4): JEPA =
  data2vec-style masked latent prediction against an EMA teacher (burn-jepa;
  momentum 0.999, teacher advanced after every optimizer step) + KoLeo
  anti-collapse; DSpark = DeepSeek-style draft head correcting frozen logits into
  the next-K tokens (burn-dspark, used instead of MTP; gamma 4.0). Aux value is
  logged as `aux=` on log steps. Set both weights to 0 for the pure-CE baseline.
- `--jepa-precompute N` + `--jepa-targets <file>` — offline teacher latents: one
  forward per batch, no optimizer, no EMA advance; the hot loop then runs no
  second forward and no teacher at all.
- `--seed N` (default 1) — the only stochastic input to a step (the JEPA span
  mask), a pure function of `(seed, step)`. In the snapshot: a different seed is
  a different run. It does **not** seed the model init or the data order.
- `DM_QUANT_DEBUG=1` remains the one env var (debug-only, prints every
  `LinearLike` quant format); `CUBECL_AUTOTUNE_LEVEL` is the cubecl runtime's
  own knob, exposed as `--autotune`. Everything else is a typed flag:
  `--timers`, `--memlog`, `--quant-check`, `--log`, `--detach`, `--guard` (see
  `--help`).

## 3.8 API and performance state (2026-09-27; do not revert)

- **Current API**: `forward_with_hidden(input_ids, hashed_ids, host_rows,
  targets, teacher)` → `(logits, rec, kda, aux)`. `L_Rec` is accumulated inside
  the loop (never materialize `[N,b,t,d]`); `targets` are byte indices `[b,t]`
  Int, and the per-iteration CE gathers the target log-prob (`log_softmax.gather`)
  — there is no one-hot `[b*t,v]` tensor anymore (2026-09-04).
  `loss(rec_ce)` is the honest mean CE and nothing else: the `+ β·KL` term and
  the `p_dist` return were deleted with PonderNet. `host_rows` is `[b,t,96]` f32
  and drives the RAM-offload Engram; `teacher` is the EMA copy for JEPA.
  `examples/quant_probe.rs` is up to date with this API.
- Host-table CPU updates run **every step** (`--host-adam-every`; they were
  accidentally gated on log steps = 1/100 of updates); aux/loss scalar clones
  only happen on sync steps; the device syncs only on steps that read something
  back (log cadence, host-Adam cadence, timers); `save_ckpt` streams through a
  1 MB BufWriter (2× (model+optim) RAM transient — burnpack can't stream
  records, that's the floor). A/B of the old-vs-new CE at small/batch6/s512: no
  wall-clock delta (~9.7 s/step, launch-bound) — the win is removed
  temporaries/pool pressure, visible at larger scale.
- The NaN firewall masks the loss AND sanitizes gradients on device (§1.3); the
  raw loss copy is read **before** the mask, on log cadence only, because
  `clone()` shares the device buffer and `mask_fill` is in-place on CUDA at two
  handles — reading it after the mask returned 0.0 and left `best` stuck at 0
  forever.
- The whole-loop `fused/` module is deleted; `fused_seam_counts()` and
  `fused_kernels_skipped()` are the counters that replaced it, printed on the
  eval line.
- 5060 Ti budget rules of thumb (from `bf16_KERNEL_PLAN.md`): 1B model bf16 ≈ 2 GB
  weights, +2 GB per batch-16 s1024 step. The remaining headroom is where the
  RAM-offload playbook buys scale.
- Launch line for the flagship recipe:
  `./target/release/train --data <corpus dir> --eval <held-out> --eval-every 500
  --preset small --batch 10 --seq-len 512 --engram-ram --engram-slots 48000000
  --host-adam-every 1 --guard --detach --log <file>` — and, per §1.5, under
  `systemd-run --user --scope -p MemoryMax=40G`.
