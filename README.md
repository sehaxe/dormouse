<div align="center">

# 🐹 dormouse

**A byte-level language-model trainer that fits on one workstation — published
with the measurements that justify it and the ones that refute it.**

[![CI](https://github.com/sehaxe/dormouse/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/dormouse/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org)
[![CUDA](https://img.shields.io/badge/CUDA-sm_120-green.svg)](https://developer.nvidia.com/cuda-gpus)
[![burn](https://img.shields.io/badge/burn-0.22.0--pre.4-red.svg)](https://github.com/burn-rs/burn)

Rust · burn + cubecl · one RTX 5060 Ti (16 GB, sm_120) · 5 140 lines in
`dormouse-core`, 14 437 across the six workspace crates, 28 crates of vendored
fused kernels

</div>

---

## STATUS

Read this section first. Everything below it is context; this is the part that
tells you whether the project is worth your time. Four of its own claims were
retracted on 2026-09-27, so nothing here is presented as verified unless the
verdict and the artifact are both named.

**Machine for every number below** unless stated otherwise: 1 × RTX 5060 Ti
16 GB (sm_120, consumer Blackwell, 170 SM, 1792 GB/s), 62 GB RAM, Linux,
rustc 1.98.1, burn 0.22.0-pre.4 + vendored cubecl.

### (0) Newest first — 2026-10-01

| what | number / state | provenance |
|---|---|---|
| **First A/B verdict in project history** | **JEPA+KoLeo beats pure CE 3/3 seeds** — control mean 6.343 vs pure-CE 6.425 held-out BPB, full seed separation | 2k steps, batch 8 s512, `d8062d1`; rows in [`benches/history.tsv`](benches/history.tsv). The aux heads stay ON at 0.05/0.1 |
| Seed spread of the control family | 2k-step finals **6.314 / 6.329 / 6.387 / 6.396 / 6.437 / 6.443** across 7 runs — spread ~0.13, so a single arm-vs-control pair decides nothing (§1.2) | same shape; every run's config snapshot in `checkpoints/` |
| **100k official run** | **in flight**: step ~92k/100k at ~500 ms/step, best held-out **5.615** (over an **81 920 B** window), train best 2.415, 0 NaN, `--retract-every 4` | `~/logs/first_run_100000_1001_0202.log`; final row lands in `benches/history.tsv` |
| Six mechanism arms landed, all with gates, all OFF by default | MHC (hyper-connections), SiTU-GLU (Kimi K3), RoPE-in-KDA, MoE top-1 (4 experts), AttnRes (arXiv 2603.15031), future-byte aux — **the 3-seed A/B wave is running today** | one landing commit each, `probe.rs` counts 19 arms |
| KDA gradient question **closed** | all 11 KDA parameter groups receive non-zero, finite gradients on the trainer's backend; CPU NdArray and CUDA agree bit-for-bit on the ops path | `d8fa449`, `tests/kda_param_grads_cuda.rs`; falsified by a `detach()` at `cuda_dispatch.rs:449` killing 8/11 groups |
| The "fused adjoint is wrong" finding **dissolved** | the old gates measured ONE chunk against a reference fixed the same day (a double-`exp` in the generator); both fixtures (1-chunk + 2-chunk, all 7 gradients, oracle error 2.3e-12) now sit behind one gate, and the gate demonstrably detects the injected fault (rel 1.000) | `docs/reviews/kda-adjoint-2026-10-01.md`, `tools/falsify_fused_adjoint.sh` |
| CUDA graph capture mechanism **proven** | captured-step replay is bit-exact over 8 replays, **0 kernel launches across 8 replays**, price = 1 launch per parameter per step; the pin is MANDATORY — a captured step without it reads stale pointers silently ("got −3, correct −6") | `vendor/cubecl-fix/cubecl-cuda/tests/graph_step.rs`; trainer integration in flight |
| burn-spectral formula audit (first ever) | findings 1–5 landed: three class-A comment defects with gates, two class-B failure modes; 6-mutant falsifier all-DETECTED; python oracles `tests/oracle/` | `a438c91`, `docs/reviews/spectral-audit-2026-10-01.md` |

Everything below was written before this subsection and is kept because it is
still true; where the two disagree, this subsection wins.

### (a) What is measured

| claim | number | when / shape / how |
|---|---|---|
| Fused gated-delta (KDA) kernel, forward | **1.212 ms** vs `torch.compile` 54.5 ms = **45×**; vs PyTorch eager-best 160.1 ms = 132× | 2026-09-27. b=10, t=512, h=12, K=V=64, chunk=16, fp32, 10 runs × 200 iters, real `a_log=-3` decay. torch 2.11.0+cu128. [`docs/research/2026-09-27-pytorch-baseline-renamed.md`](docs/research/2026-09-27-pytorch-baseline-renamed.md) §1 |
| …same op, forward+backward | **RETRACTED — do not cite.** It read **25.9 ms** vs PyTorch eager-best 290.3 ms = **11.2×**, 138× fewer allocations, 2.5× less VRAM. The device was built as `Device::autodiff(...)` = NoCheckpointing, so the measured backward contained the *tensor* adjoint, and the verification test compared that tensor adjoint against the tensor path — verified the tensor adjoint twice. Only the FORWARD row above survives | retracted 2026-09-28, `AGENTS.md` §3.2 (last bullet). The fused adjoint kernels have still never been numerically compared to anything |
| f16 GEMM, our path vs cuBLAS | **7.4× behind** (0.389 ms / 41.4 TFLOP/s cuBLAS vs 2.880 ms / 5.6 TFLOP/s ours) | 2026-09-27, `[5120,768]×[768,2048]`, zero-copy on cubecl's own stream, `max rel err 2.7e-4`. `crates/cublas-poc`; log `~/logs/cublas_poc_2026-09-27.log` |
| fp32 GEMM, our path vs cuBLAS | **no ratio claimed** — 4.8 / 47.1 / 47.7 / 51.7 / 71.1 ms for one fixed matmul in the *same binary* (15× spread, power-capped) | 2026-09-27. At the median we are 4.4× behind; at the fastest sample 2.3× ahead. The measurement does not resolve it, so neither number is published as fact |
| **A warm training step, batch 8 × seq 512** | **246–250 ms** (steps 4/8/50/100/150 → 248/246/249/240/250). Split: bwd 96–100 ms · retr 52–61 · fwd 46–48 · opt 43–47 | Measured 2026-09-29 on this box, release, `small` 9 195 854 params, depth 2, fp32, aux off, `--no-engram`, `CUBECL_AUTOTUNE_LEVEL=3`, `--timers`. **A step-0 reading is 5549 ms and is worth nothing** — see the note under the table |
| **GPU utilisation, warm** | **13.3% mean; 142 of 180 samples at ≤5%** | `nvidia-smi` at 2 Hz across a 150-step run, batch 32. The workload is **launch-bound**: too many small kernels to fill the SMs. This is the ceiling, and it is not the model. The lever is CUDA graph capture/replay — 5/5 green on this GPU in `vendor/cubecl-fix/cubecl-cuda/tests/graph.rs`, **not yet wired** |
| ~~Training step 1.58–1.84 s/step~~ | **RETRACTED 2026-09-29.** The old `t = 465 ms + 0.067 ms/token` model is the right order; the 1.58–1.84 s figure and the "4081 ms" bench row are **step-0 readings** | Every reading above that lacked a step index was a step-0 reading, and a step-0 step is **23× a warm one** because the cubecl autotune cache is cold. `--timers` used to print on `step % 50` only, so short runs could only ever see step 0. Both are fixed. `benches/history.tsv` keeps the retracted rows visible |
| **Best held-out BPB with nothing known-broken in it** | **4.997** at step 6500 (regressed to 5.450 by 19500 — best-of a curve that overfits) | 2026-09-28, `~/logs/train_nokda.log`. `nokda_ce.config.toml`: `use_kda=false`, `use_engram=false`, `engram_ram=false`, batch 2, seq 512, **depth 2**, 9 195 854 params, over a **20 480 B** window. Both of this project's held-out instruments were broken for some runs and neither touched this one: no memory in training means no memory missing in eval, and `--no-kda` means the no-gradient attention arm was not in the model. It is a statement about a model with **no attention arm at all**. Below every unigram bar on record; 1.5 BPB *worse* than the 5-gram. See (d) |
| Best held-out BPB at depth 4 | **6.351** at step 1500 | 2026-09-27, `~/logs/official_v5e.log`, 7 526 223 params (pre-repricing `small`; the shipped `small` is 9.20 M, so the best number on record was not produced by a preset in `configs/` today), batch 10, over a **102 400 B** window. **Not comparable to the 4.997 above — different window, different depth, different batch.** Its `use_kda=true`, so its attention arm received no gradient (`8fa5d4c`): it is a depth-4 result for a network with frozen attention |
| Reference-fidelity suite | ~~1000 cases pass~~ **RETRACTED 2026-09-29: it is RED — 976/1000 cases fail at 5e-4 (`max_diff = 1.38e-2`), and `binary-tests` is not a default feature so it does not run by default.** The fixture discipline is real and CI regenerates it byte-identically | `vendor/burn-fused/crates/burn-gdn2/tests/bit_exact.rs`; status in its own module header and `vendor/burn-fused/TEST-AUDIT.md` FINDING 0. Cause **not isolated** — the 24 passing cases are exactly the single-token ones. **Not a passing gate: do not cite it as one.** Provenance is also wrong (our own transcription, not the authors' output) — see (b) |
| Test inventory | 315 unit + 62 integration `#[test]` functions across 28 crates | counted from source 2026-09-27. Presence is not correctness — see (b) |

> **The raw training logs are not in this repository** (`~/logs/`, gitignored
> and outside the tree). Every number in this table is a transcription of a log
> that a stranger cannot read, a fixture that is committed, or a file:line in the
> tree. That is a real gap in the evidence, and it is the first thing to fix.

### (b) What is implemented but UNVERIFIED

- **0 of 28 library crates check any numeric output against an *authors' own*
  source code.** Six crates assert in their docs that they match an
  implementation; none of the six has a test wired to one. Full per-crate table
  with provenance and tolerances: [`docs/research/2026-09-27-oracle-audit-renamed.md`](docs/research/2026-09-27-oracle-audit-renamed.md).
  The honest sentence: *the library verifies fused kernels against our own
  tensor fallback, and transcriptions of papers against our own expectations of
  those papers.* That catches real drift. It is not "bit-for-bit against the
  reference", and no crate may say it is unless it names the external source.
- **28 tests cannot fail, report PASS having asserted nothing, or are
  unreachable by any CI job that has ever executed** (2 where the reference is a
  character-for-character copy of the code under test, 2 more where the expected
  value is a hand-copy that moves with the bug, 9 that silently skip or panic
  before asserting, 1 permanently unrunnable, 14 CUDA-executing on a runner that
  does not exist). Same audit, §3.
- **Exactly one mechanism has an A/B verdict — the aux heads (2026-10-01).**
  JEPA+KoLeo at 0.05 beats pure CE **3/3 seeds** with full separation (control
  mean 6.343 vs 6.425, `d8062d1`). The verdict required re-baselining first:
  every run before `8fa5d4c`'s fix had an attention arm frozen at
  initialisation, and the Engram arm's only comparison was scored by a
  memory-disabled eval (`7adda92`) — §(a)'s retraction table has both. The six
  arms landed 2026-09-30/10-01 (MHC, SiTU, RoPE, MoE, AttnRes, future-byte) are
  in the queue **today**, 3 seeds × 2k steps each, and per
  [`docs/protocols/AB-PROTOCOL.md`](docs/protocols/AB-PROTOCOL.md) **a tie deletes the mechanism**.
  The protocol's old cost line (2.7 GPU-h/arm) is void; at the measured
  ~500 ms/step a 2k-step run is ~17 min, so an arm is ~1 GPU-h.
  The Engram arm — 2.4 M memory parameters,
  24 % of `small`, on by default in every preset — **was** compared to
  `--no-engram` once, and the comparison is void: the eval threw the n-gram
  keys away on the in-VRAM path (`7adda92`), so the Engram run's held-out 6.453
  was scored by a memory-disabled evaluation of itself, and the "Engram lost by
  1.46 BPB" reading is not a verdict. It cannot bite a `--no-engram` run and
  cannot bite `--engram-ram`, which is why the retraction is scoped to that one
  arm rather than to the project's history. Its capacity default also cites a
  *third-party* curve (arXiv 2601.16531, single-author preprint) measured on a
  125 M backbone, 16× ours.
- **The fused KDA kernels ran in no training step at all, for the whole history
  of this project, and the reason was worse than a gate that never fired.**
  `chunk_wy_forward_autodiff_s` ran the forward on the bare backend and wrapped
  it in one hand-rolled autodiff node; under `BalancedCheckpointing` the inputs
  are checkpoint leaves, the node came back `UnTracked`, and its output was a
  leaf — so **the attention arm received no gradient** (`8fa5d4c`). The tensor
  fallback lives inside that same node's backward, so a leaf means no gradient
  either way. `~/logs/train_kda_full.log` prints `fused kda=3126/0`: 3126
  forwards, zero backwards. Two consequences a reader should not have to
  discover: every held-out number from a `use_kda=true` run describes a network
  with attention frozen at initialisation, and **the fused forward made
  attention look nearly free (+22 ms) precisely because it was doing no
  backward work.** The fix makes the op decline (`fused kda=0/0` in a training
  pass) and lets burn build the graph. **The gradient question is now closed**
  (`d8fa449`, `tests/kda_param_grads_cuda.rs`): all 11 KDA parameter groups
  receive non-zero, finite gradients on the trainer's backend; CPU NdArray and
  CUDA agree bit-for-bit on the ops path; the falsification is a one-line
  `detach()` that kills 8/11 groups *and names them*. The separate claim that
  the fused **adjoint** was numerically wrong dissolved the same way a
  retraction should: the old gate compared one chunk against a reference whose
  generator had a double-`exp` — fixed the same day, regenerated at 2.3e-12,
  and both chunk fixtures now sit behind one gate that demonstrably detects the
  injected fault (`docs/reviews/kda-adjoint-2026-10-01.md`). The trainer
  *declining* the fused op under checkpointing is correct behaviour, not a
  fallback — a fused path that skips backward work is a different program.
- **No fused multi-head attention exists in this stack at all.** dormouse's only
  attention arm is the gated-delta recurrence; `burn-cubecl` 0.22.0-pre.4 ships
  no fused attention kernel. Where PyTorch has one (mem-efficient, 0.656 ms at
  b=10 h=12 t=512) a burn-composite would be 5.980 ms — 9× slower.
- **Preset costs are asserted by a test, not by this file.** The parameter
  column in [Presets](#presets) is what `cargo test -p dormouse-core --test
  preset_exec` prints; that test is real and CI runs it, but it was not re-run
  for this README.

### (c) What is BROKEN or KNOWN-WRONG

- **bf16 matmul cannot work on this backend, and no flag will make it.**
  `restrict_to_llvm_backend` (`vendor/cubecl-fix/cubecl-cuda/src/runtime.rs:420-513`)
  deletes bf16 from the advertised element types because the pliron LLVM dialect
  this backend lowers through has `builtin.fp16`, `fp32`, `fp64` and nothing in
  between — a bf16 kernel compiles to something that quietly computes zeros. So
  `--bf16` stores bf16 and computes fp32 through cast copies, and **every bf16
  run on this box is slower than fp32**. The one primitive that works is bf16
  *storage* as `u16` bit patterns.
- **f16 matmul is correct but silently slow**: the tensor-core candidate dies at
  compile time (`dyn SizedType` unimplemented for pliron's `FP16Type`) and the
  autotuner falls back to a non-accelerated routine without a word. The fix is
  ~10 lines in `cubecl-ir` and belongs upstream; until it lands, cuBLAS's
  41–47 TFLOP/s is not reachable *through burn* — only through the POC.
- **`load_ckpt` used to wipe the quant format.** The format, bf16 compute and the
  one-way fp32 fallback live in non-param fields, so loading a checkpoint
  rebuilt the model from `new()` and dropped them: a resumed quantized run
  trained the fp32 forward while its own log said `quant format: Fp8`, and the
  fallback latch was lost with it. Fixed 2026-09-27, latch restored from a
  checkpoint flag ([ADR-0021](docs/adr/0021-resume-is-the-same-run.md)). **Any
  quantized result produced by a run that was ever resumed before that date is
  an fp32 result.**
- **Four claims this project made were retracted on 2026-09-27**, and they are
  listed here so nobody re-derives them from an older file: a `bit-for-bit`
  claim whose test had never run and whose reference fixtures were excluded by
  the crate's own `.gitignore`; a `bool → float` "backend bug" that is not a bug
  (the cast is correct on both backends — the miscount was `clone()` aliasing
  the device buffer under `mask_fill`, plus a firewall that did not skip the
  step); an `--act-quant fp4` "verification" whose attention path silently ran at
  8 bits (`ActFormat::attn()` maps `Fp4 → Int(8)`); and a "the drift check is
  inverted" claim whose check was never inverted.
- **PonderNet is gone.** The project's previous headline — adaptive compute
  that decides per sequence how deep to think — was measured and deleted
  ([ADR-0013](docs/adr/0013-fixed-depth.md)): λ-collapse made the reported loss
  fake (`p_dist → [9e-4, 4e-6, 1e-8, 2e-11]`, held-out frozen at exactly
  uniform), and the KL direction made collapse free. The loop now runs a **fixed**
  `max_iter` with an unweighted mean CE. Depth varies only under `--rand-depth`
  (training) and `--eval-depths` (measurement).
- **The most recent production run does not finish.** `official_v5` reached
  held-out BPB 6.351 at step 1500 and then went NaN at step 2000 on every one of
  its resume attempts (`~/logs/official_v5g.log`); the guard's restart chain is
  capped at 3. `checkpoints/` in this tree is empty.
- **The `engram:` startup line under-reports RAM by 2×**: it prints
  `total_rows × row_bytes` (table only) where the trainer also holds a full
  momentum buffer (`crates/dormouse-train/src/lib.rs:882-886`).
- **NaN on deep overfit is open.** It appears in any config on this box,
  including fp32 + AdamW, once the model overfits hard; `--no-kda` stays clean
  through deep overfit, so the KDA recurrence is the prime suspect. Never
  triggered on a real loss profile so far, never explained either.

### (d) Has any trained checkpoint ever beaten a 5-gram byte counter on held-out data?

**No. Not one, and not close.**

Two held-out numbers, on **two different windows** — the window is
`eval_batches × batch × seq_len` bytes, so a BPB is only comparable within one
window, and no run in the archive has been scored on both. The `window` column
is the authority, and it is printed on every eval line.

| model | held-out BPB | window | vs the counter |
|---|---|---|---|
| uniform byte | 8.000 | — | — |
| unigram counter | 5.17 | *not a trainer window* | — |
| **dormouse, `--no-kda`, step 6500 (2026-09-28)** | **4.997** | **20 480 B**, depth 2 | **0.17 better than the unigram reading above — on a window the bar was not measured on** |
| dormouse, best depth-4 run, step 1500 (2026-09-27) | **6.351** | **102 400 B**, depth 4 | +1.18 worse than the unigram counter |
| 5-gram + backoff, ~24 lines of counting | **2.572** | *not a trainer window* | +2.43 worse (4.997) / +3.78 worse (6.351) |

The two anchor rows are the weakest part of that table and should not be read
as a measured margin. `crates/dormouse-data/src/bin/anchors.rs:22-27` says in
its own header that the internal `--holdout` split scores the trailing quarter
of its own read while the trainer's eval reads a fixed window from a different
file, so **the two "were never comparable"** — which is why `--fit` exists. Four
readings are in circulation and none is on a trainer eval window: unigram
5.398 / 5-gram 2.911, unigram 5.170 / 5-gram 2.572, `--fit` unigram 5.011 /
5-gram 2.588 (`anchors.rs:27-32`, the only same-*file* measurement, but over the
whole 500 MB tail), and 2.826 vs 2.849 for one file at `--bytes 1M` vs `2M` — the
bar moves 0.023 on the fit size alone. To state a real margin, run
`anchors --fit <filtered corpus> <eval file>` and put its `window:` line next to
the model's. Uniform is 8.000 by definition.

What the bars do support without any of that: **the 5-gram line is the one every
number is far above**, by 2.1–2.4 BPB on the best measurement in the archive and
3.8 on the depth-4 one. So the answer to this section's question is no, and the
answer does not depend on which of the four unigram readings you pick.

The 5-gram line is not a strawman: on the corpus the model is trained on, a
4-gram beats it by ~4.9 BPB, and on the text domains a 5-gram sits at 2.25–2.66
while the *agentic* slice collapses to 0.938 at 0.0 % unseen 5-gram contexts —
i.e. 21 GB of the available data is memorizable at order 5 and teaches format,
not language ([`docs/research/2026-09-27-domain-anchors-renamed.md`](docs/research/2026-09-27-domain-anchors-renamed.md)).

This is the number that matters, and it is the reason the project's own
milestone ladder reads "7.5 M beats a 24-line counter" as rung 1
([`docs/research/2026-09-27-scale-ladder-renamed.md`](docs/research/2026-09-27-scale-ladder-renamed.md)).
**Any capability claim about this model is premature until that number moves.**

### (e) What blocks a 1 B-parameter core model, in numbers

1 B parameters at Chinchilla's 20 tokens/param is **20 B tokens**. At the
measured 9.2 M operating point — **~245 ms warm**, 4096 tokens/step at batch 8
(16.8 K tok/s), rising to 826 ms / 19.8 K tok/s at batch 32, with the GPU
**13.3 % utilised and 79 % of samples ≤5 %** (launch-bound, not compute-bound)
— the same card does 0.5 B at 13.0 k tok/s and 1.5 B at 3.9 k tok/s in the only
published benchmark of this
exact GPU (LLMQ, arXiv 2512.15306, 78–85 % MFU). Our own arithmetic, from
measured throughput (`docs/research/2026-09-27-scale-ladder-renamed.md`):

| | 7.5 M (now) | 1 B |
|---|---|---|
| step time, fp32 | ~~1.6 s~~ **WITHDRAWN (step-0 reading); warm at batch 8 is 245 ms** | ~16 s *(roofline from FLOPs × measured TFLOP/s, not a step reading — unaffected)* |
| step time, working f16 GEMM | ~~1.6 s~~ (same withdrawal) | ~2.5 s *(same roofline basis)* |
| wall clock to 20 B tokens | ~~**9 h**~~ **withdrawn with the step time; 150M tokens at the measured 16.8K tok/s is ~2.5 h** | **30–90 days** *(this is **LLMQ's** figure for this card at 1.5B and 3.9k tok/s, not ours. It does not follow from the 16 s/step above: 20B tokens at 5120/step at 16 s is ~2 years. Both numbers were under one label.)* |

The 7.5 M column was struck on 2026-09-29 (see row 3 of the table in (a)). The
1 B column is a roofline and stands; its **wall clock** did not, and the
discrepancy is named above rather than smoothed over. `docs/research/2026-09-27-scale-ladder-renamed.md`
carries the full withdrawal.

Four blockers, all measured or code-verified:

1. **No tensor-core GEMM.** bf16 cannot work (see (c)); f16 is 7.4× behind
   cuBLAS. The zero-copy cuBLAS POC reaches 41.4 TFLOP/s on cubecl's own stream
   — it is a proof, not an integration, and `crates/cublas-poc` is not wired into
   the trainer.
2. **No optimizer-state offload for the backbone.** fp32 AdamW is 8 B/param =
   8 GB at 1 B, plus 4 GB of fp32 masters, on a 16 GB card that must also hold
   activations. LLMQ's recipe for this exact card is bf16 states with stochastic
   rounding, double-buffered host offload (it measured zero-copy as *bad* on
   5060 Ti/4090), and all allocations at startup. The host-offload machinery
   exists — but only for the n-gram tables.
3. **The workload is launch-bound, and that is the ceiling on this card.**
   Measured, warm, at 9.2 M parameters: the GPU is **13.3 % utilised and 79 %
   of samples at ≤5 %** — too many small kernels to fill the SMs, not a
   shortage of arithmetic. The step is ~245 ms (bwd 96–100 · retr 52–61 ·
   fwd 46–48 · opt 43–47). The old ~465 ms "fixed cost" is **withdrawn**
   — it was fitted to step-0 readings, and a step-0 step is 5549 ms against a
   warm 245 ms in the same config (`benches/history.tsv`) — but its conclusion
   is now measured rather than inferred: what dominates is the number of
   launches, not FLOPs. The
   lever that attacks launches directly is **CUDA graph capture/replay**,
   confirmed working on this GPU, and the mechanism is now **proven, not just
   compiled**: a captured *step* replayed 8 times is bit-exact with **zero
   kernel launches**, at the price of one launch per parameter per step to
   re-pin — and the pin is mandatory, because a captured step over an
   out-of-place optimizer reads stale pointers and returns plausible garbage
   (`vendor/cubecl-fix/cubecl-cuda/tests/graph_step.rs`, the "−3 vs −6"
   fixture). The trainer wiring (whole-compute-span capture on non-log steps,
   loss read outside the window) is in flight; the decisive number is **L**,
   launches per warm step (now printed on the timer line as
   `cubecl_launches()`): L > ~200 means the graph is the step, L < ~200 means
   it is not worth the complexity.
4. **The model is too small to use the card.** ~19.8 K tok/s at batch 32 is
   close to what this card does on a 9.2 M-parameter model at all. The ceiling is
   model size and launch count together, and neither is fixed by making the code
   faster.
4. **The evidence base does not exist yet.** Per (b), no mechanism has an A/B
   verdict and no checkpoint beats a counter. Scaling a recipe that has not been
   shown to work at 7.5 M is optimizing the wrong term.

---

## Architecture

Four steps, in `crates/dormouse-core/src/model.rs:15-18`. The loop is run
`max_iter` times per sequence with **shared weights**; there is no per-depth
stack of layers.

```
 bytes 0..255
     │
     ▼
 ┌───────────┐   Embedding  [256, d_model]          model.rs:52
 └─────┬─────┘   the only place a byte becomes a vector
       │
       ▼
 ┌──────────────────────────────────────────────────┐
 │ LoopBlock  × max_iter   (shared weights)         │  loop_block.rs
 │                                                    │
 │   h_ctx ──RMSNorm──┐                              │
 │      controller ────┼─► w_attn  w_mem  w_ffn      │  one Linear,
 │                    └─► blend (softmax over       │  3 gates +
 │                          n_experts)               │  1 expert blend
 │      │                                             │
 │      ├─► KDA gated-delta  ──× w_attn ──┐         │  the recurrence
 │      │   (burn-kda, fused CUDA)         │         │  carried across
 │      ├─► Engram  ─┐                     │         │  iterations
 │      │   memory·min(w_mem,0.5)           │         │
 │      │   +  (1−min(·))·dense ──× w_mem ─┤         │  never more
 │      ├─► Σₑ blendₑ · FFNₑ (TSCT) ─× w_ffn         │  than half
 │      │                                     │      │  the branch
 │      └── ReZero scale (or GR write) ──────┘      │
 │                    │                              │
 │              per-iteration CE, uniform mean ──────┼──► rec
 └────────────────────┬─────────────────────────────┘
                      ▼
 ┌───────────┐   RMSNorm  fp32 even under --bf16    model.rs:220
 └─────┬─────┘   logits in fp32 or the NaN budget is gone
       ▼
 ┌───────────┐   lm_head (LinearLike, TSCT-capable)  model.rs:225
 └─────┬─────┘
       ▼
   next byte
```

One sentence per mechanism, and why it is there:

- **KDA gated-delta attention** (`shared_attn`) — the only attention arm. A
  recurrent delta rule with a learned per-channel decay, so the state is
  `[batch, heads, 64, 64]` and is *carried* across loop iterations instead of
  recomputed. The sparse-MSA arm was disabled by an out-of-bounds top-k gather on
  burn 0.22.0-pre.4 ([ADR-0012](docs/adr/0012-msa-broken-on-pre4.md)) and then
  cut along with its crate ([ADR-0014](docs/adr/0014-msa-cut.md)).
- **Engram hashed n-gram memory** (`engram`) — FNV-hashed rows of orders 2/3/4
  living in host RAM; deterministic addressing means the GPU only ever touches
  the batch's own rows (~600 KB/step). Mixed with a dense projection of the
  same hidden state under a *hard* floor `λ = min(w_mem, 0.5)`, because a
  memory branch that can carry more than half the signal starves the backbone of
  gradient — the failure that made nearly every run in this project's history
  get rescued by `--no-engram` instead of by a config.
- **TSCT low-rank expert FFNs** (`expert_ffns`) — fp32 masters, quantized
  forward, a real polar retraction every step to hold the factors
  orthonormal, and a one-way fp32 latch if they drift. Muon+ ColRow on the 2D
  maps, AdamW on embeddings/heads/1D, the n-gram table on Adam with no weight
  decay.
- **ReZero residual** (`residual_scale`, or `gr`) — one learned scalar per
  iteration on the sum of the three arms. The Gated Residual variant
  (`use_gr`, off by default for checkpoint compatibility) widens the residual to
  four branches with a low-rank sigmoid read.
- **Fixed depth + unweighted CE** — see (c). `--rand-depth` trains one model to
  be correct at every depth 1..`max_iter`; `--eval-depths` prints held-out BPB
  at each depth for free, which is how you find out whether the depth you pay
  for is being used.
- **Aux objectives, on by default** — JEPA masked-latent prediction against an
  EMA teacher (weight 0.05) + KoLeo, and a DSpark draft head for next-K
  prediction (weight 0.1, K = 4). Both have never been A/B'd against pure CE;
  `--jepa-weight 0 --dspark-weight 0` is arm 1 of the queue. The DSpark term was
  repaired on 2026-09-29 and **has no valid number yet**: its teacher-forced
  window read the label sequence, so the draft head was handed the very byte
  its own base logits had just predicted (it now reads the consumed sequence),
  and its acceptance head was a `w^T[h_k]` stand-in rather than the paper's
  `sigmoid(w^T[h_k; W1[x_{k-1}]])`. Windows are placed every `dspark_stride`
  = 16 positions instead of the paper's random anchors — a determinism fix
  (ADR-0021), not an approximation the A/B has confirmed. The head's projection
  grew from `[d_model, 1]` to `[d_model + rank, 1]`, so a pre-2026-09-29
  checkpoint is refused at load rather than silently reshaped.

---

## The vendored kernel library (burn-fused)

Every mechanism that is not the model lives in
[`vendor/burn-fused/crates/`](vendor/burn-fused/crates) — 20 crates of our own
technology library, each with its paper reference and its own verification
tier ([`docs/protocols/ORACLE-TIERS.tsv`](docs/protocols/ORACLE-TIERS.tsv)). They are **not**
dependencies of burn; the repo also vendors patched forks of five cubecl/cubek
crates (`[patch.crates-io]` in the root `Cargo.toml` — the only authority for
the count).

| crate | what it is |
|---|---|
| `burn-kda` | Kimi Delta Attention — data-dependent write strength, channel-wise decay (Kimi Linear + K3), fused CUDA chunk path |
| `burn-gdn2` | Gated DeltaNet 2 — linear recurrent token mixer, channel-wise erase/write gates |
| `burn-spectral` | Ternary Spectral Compact Training — BitNet-style ternary SVD weights, rank-1 ternary MoE routing; the NS quintic + basin gates |
| `burn-sct` | Spectral Compact Training — permanent truncated SVD with Stiefel QR retraction (not in the build; delete-or-keep pending) |
| `burn-muon-plus` | Muon+ optimizer — NS polar orthogonalization + post-polar ColRow normalization, hybrid AdamW fallback |
| `burn-rmsnorm` | RMSNorm with a fused CUDA kernel (first real launch 2026-09-30, `34c5631`) |
| `burn-situ` | SiTU-GLU activation — Sigmoid-Tanh Unit gated linear (Kimi K3's stability choice) |
| `burn-swiglu` | SiLU-gated linear (the reference FFN the SiTU arm replaces) |
| `burn-rope` | Rotary position embeddings with YaRN extrapolation |
| `burn-mhc` | Manifold-constrained hyper-connections — multi-branch residual with identity preservation (DeepSeek) |
| `burn-attnres` | Attention residuals — learned depth-wise attention over layer outputs (arXiv 2603.15031) |
| `burn-mor` | Mixture-of-Recursions routing (recursion-slot ranking) |
| `burn-engram` | Conditional memory — n-gram hash embeddings with multi-head gated fusion (the Engram arm) |
| `burn-jepa` | data2vec 2.0-style EMA teacher + masked latent prediction (the JEPA aux) |
| `burn-dspark` | DSpark speculative decoding draft head (DeepSeek, arXiv 2607.05147) |
| `burn-bitnet` | BitNet quantization family — ternary, 8/4-bit absmax/absmean, Fast Walsh-Hadamard rotation |
| `burn-eggroll` | EGGROLL — low-rank evolutionary strategies (arXiv 2511.16652) |
| `burn-es` | Evolution strategies |
| `burn-parcae` | Stable looping via spectral retention |
| `burn-ptrn` | Probabilistic Tiny Recursive Model — test-time scaling for recursive models |

## Presets

Flat TOML in [`configs/`](configs/) — presets are data, not code, and the schema
defaults *are* `small`. Geometry below is read from the TOML; the memory-row
count is derived from the code (`engram_tables` rounds `engram_rows` up to a
power of two because the in-model read masks the hash — asserted at
`loop_block.rs:654`); the compute column is what
`cargo test -p dormouse-core --test preset_exec` prints on the instantiated
model, and the wide rows are behind `-- --ignored` because a full-width CPU
build costs minutes.

| preset | d_model | experts | max_iter | memory rows | total (compute) | what it is for |
|---|---|---|---|---|---|---|
| `nano` | 512 | 3 | 4 | 3.15 M | 7.19 M (4.05 M) | smoke tests |
| `nano-fused` | 512 | 3 | 4 | 3.15 M | 7.19 M (4.05 M) | `nano` with KDA+Engram+aux off, single-node path |
| `small` | 768 | 3 | 4 | 3.15 M | 9.20 M (6.05 M) | **the flagship on 16 GB** |
| `mor` | 768 | 3 | 4 | 3.15 M | 9.20 M (6.05 M) | `small` + Mixture-of-Recursions A/B arm |
| `base` | 1024 | 3 | 8 | 3.15 M | 13.07 M (9.92 M) | 24 GB-class work; OOMs on 16 GB at batch 6 |
| `swift50` | 1024 | 8 | 8 | 3.15 M | `-- --ignored` | many-expert experiments |
| `one_b` | 2048 | 4 | 12 | 3.15 M | `-- --ignored` | the 1 B dream |
| `p150` | 4096 | 4 (64 heads) | 12 | 3.15 M | ~160 M compute (159 720 195, per the header probe named in the TOML) | the 150 M-program flagship |

Two things that surprise everyone, both visible in that test's printed split:

- **Every preset pays the same 3.15 M memory rows** (3 tables × 32 768 rows × 32
  dims) — 24 % of `base` and 44 % of `nano`. The budget is a ratio, so the narrow
  presets pay it hardest.
- **`use_engram = false` does not shrink the model.** `nano-fused` ships the arm
  off and still allocates the full 3.15 M rows, because the tables are built
  unconditionally (`loop_block.rs:204`) and the flag is only a runtime switch.
  `--engram-ram` is what actually moves them out of the model: the config seam
  squeezes the in-model table to one row per order, because on that path the
  rows arrive pre-gathered from the host and the in-VRAM table is never read
  (`crates/dormouse-train/src/cfg.rs:50`).

---

## Quick start

**Prerequisites, all of them real:**

- **An NVIDIA GPU.** Developed and measured on sm_120 (RTX 5060 Ti, 16 GB).
  Without one, `cargo test --lib` still runs on the CPU backend.
- **CUDA toolkit + `mold`.** `.cargo/config.toml` links through mold; install it
  (`apt install mold`) or edit the `[target.x86_64-unknown-linux-gnu]` rustflags
  out. `CUDARC_CUDA_VERSION=12050` is pinned in the same file — a fresh target
  directory dies in cudarc's build script on CUDA 13.4 without it.
- **Rust stable** (developed on 1.98.1).
- **A byte corpus.** `--data` takes a *directory*; every readable file under it
  is streamed. A missing path or an empty directory is a hard error, by design
  (a silent filler once trained a model on constant bytes for 500 steps).
- **No OpenBLAS is needed** for the workspace crates — the CPU backend is
  burn-flex, whose matmuls go through the `gemm` crate. The GPU binary is built
  with `--no-default-features` so it never links BLAS at all. The
  `vendor/burn-fused` workspace is different: it enables burn-ndarray's
  `blas-openblas`, which **builds OpenBLAS from source** (a C/Fortran toolchain
  and several minutes). If one of its test binaries cannot find
  `libopenblas.so`, that is the reason:
  `export LD_LIBRARY_PATH=$PWD/vendor/burn-fused/target/release/build/openblas-src-*/out`.
- **RAM.** `--engram-ram` allocates `[slots, slots, slots]` tables at 32 dims
  (`crates/dormouse-train/src/lib.rs:869`), so the flag value is **per order**:
  8 M slots = 24 M rows = **6.1 GB**, 48 M slots = 144 M rows = **36.9 GB ≈ 37 GB**
  (f32 table + f32 momentum, 256 B/row × 2). A resumed run also reads the
  sidecar, so budget for the resumed figure and do not start one with a desktop
  running. ⚠ `AGENTS.md` §2.4 quotes ~25 GB for a resumed 48M-slot run. That
  derivation treats the flag as a *total* row count; the code triples it. Derive
  it yourself: `slots × 3 orders × 32 dims × 4 B × 2 buffers`.

```sh
git clone https://github.com/sehaxe/dormouse && cd dormouse

# build ALL THREE binaries. `cargo build-train` builds --bin train only, so
# `generate` and `serve` do not exist afterwards - a real trap, hit while
# writing this section.
cargo build --release -p dormouse-cli --bins --no-default-features --features cuda

# tests (CPU; no GPU needed)
cargo test -p dormouse-core -p dormouse-data -p dormouse-train --lib
cargo test -p dormouse-core --test preset_exec      # every preset: config, cost, and what EXECUTED

# a 2-step smoke run. Any preset works; `nano` is the cheapest.
mkdir -p /tmp/smoke && head -c 4000000 /path/to/corpus.bin > /tmp/smoke/corpus.bin
./target/release/train --data /tmp/smoke --preset nano \
  --steps 2 --seq-len 128 --batch 1 --ckpt-dir /tmp/smoke --ckpt-name smoke \
  --eval /tmp/smoke --eval-every 2 --eval-batches 2

# a real run (all knobs are typed flags: --help)
./target/release/train --data <corpus-dir> --preset small \
  --ckpt-name latest --ckpt-dir checkpoints \
  --eval <held-out-dir> --eval-every 500 \
  --engram-ram --engram-slots 8000000 --guard --detach --log /path/to/log

# turn the training checkpoint into a standalone inference file, then use it.
# This is the ONLY way `generate` and `serve` will load a model.
./target/release/export run --ckpt-dir checkpoints --ckpt-name latest \
  --dtype bf16 --out checkpoints/latest.bf16.dmexp
./target/release/generate --export checkpoints/latest.bf16.dmexp --prompt "once" --steps 64
./target/release/serve   --export checkpoints/latest.bf16.dmexp --port 8000
```

Long runs survive: `--guard` re-execs on NaN/panic with a fresh CUDA context and
resumes from the last checkpoint (capped at 3 restarts — a persistent failure
needs a human); `--detach` daemonizes in-process with `--log <file>`. Resuming
reuses `--ckpt-name`, and a resume whose resolved config differs from the
`<ckpt_name>.config.toml` snapshot is a hard error, except for `steps`,
`log_every` and `ckpt_every` — extending a run is a legal resume.

## The model file: `.dmexp` (a public interface)

**A training checkpoint is a run, not a model.** It holds a model section, an
optimizer section and an EMA teacher, and it sits next to a `.ngram` sidecar
that reaches 34 GB at the 48M-slot budget. `generate` and `serve` reading
*that* is what made a 9.2 M-parameter model a 73 MB download plus whatever
sidecar the run happened to leave.

`dormouse export` produces the other thing: one self-describing file with the
optimizer dropped, the `.ngram` sidecar never opened, and the fp32 masters
narrowed to 2 or 4 bytes each.

```sh
# convert (config comes from <ckpt-name>.config.toml, the run's own snapshot)
./target/release/export run --ckpt-dir checkpoints --ckpt-name latest --dtype bf16

# identify + verify, without loading a model
./target/release/export info checkpoints/latest.bf16.dmexp
```

### Layout

```text
[0..8)    magic "DMEXPRT" + format version byte
[8..12)   header length, u32 LE
[12..20)  payload length in bytes, u64 LE
[20..24)  CRC-32 (IEEE) of the payload, u32 LE
[24..24+H)  header, UTF-8 TOML
[24+H..)    payload: the tensors, in header order, contiguous
```

The header is TOML and carries the weight dtype, the full `DormouseConfig` (so
the model **shape** travels with the weights — a shape a reader has to be told
separately is a shape they can be told wrong), every tensor's module path and
dims, the source checkpoint and step, the parameter count, the weight magnitude
range, and the count of values that fell below the format's smallest normal.
`export info` prints all of it and checks the CRC. One file, one command, and
you can tell what it is and whether it arrived intact.

### `--dtype`: the trade-off, measured

Measured on a 20-step `small` run (9,197,390 params in 54 tensors) against the
same checkpoint's fp32 logits on 16 held-out windows of 256 B from
`real_eval_v2/eval_tail.bin`. Reproduce with
`cargo run --release -p dormouse-train --example export_divergence -- --train <corpus> --eval <held-out>`.

| format | file bytes | max abs weight | min non-zero weight | flushed to 0 | max abs logit delta | top-1 agree | greedy 64 |
|---|---|---|---|---|---|---|---|
| `f32` | **36,795,131** | 5.1545 | 1.1023e-9 | 0 | **0.000e+0** | 100.0% | identical |
| `f16` | **18,400,340** | 5.1562 | 5.9605e-8 | **8** | 1.247e-1 | 100.0% | **DIFFERS** |
| `bf16` | **18,400,343** | 5.1562 | 1.1059e-9 | **0** | 1.025e-1 | 75.0% | identical |

A 2-byte format halves the download: **18.4 MB against 36.8 MB for fp32, and
5.7× against the 104 MB training container.** f32 is bit-exact (delta exactly
0), so the entire question is the narrowing and nothing else.

**Read the last two columns honestly.** This model's logits barely leave 0
(|max| 1.11 over 256 bytes) because 20 steps is not training, so the argmax is
a near-tie and top-1 agreement is a worst case, not a prediction. The two
metrics disagree about which format is safer: f16 wins top-1 and loses the
greedy continuation, bf16 the reverse. **Neither 2-byte format preserves greedy
decoding at this scale, and the data does not say which is better.** Settling it
needs a converged checkpoint's softmax, which this repo does not currently have
(`checkpoints/` is empty) — it is the open item in the ADR.

What *is* settled is the exponent range, and it is the durable half of the
measurement: the model's smallest non-zero weight is **1.1e-9**, four orders of
magnitude below f16's smallest normal (6.1e-5), so **f16 had to zero 8
weights**. bf16 reproduced 1.1059e-9 exactly, because bf16's exponent is f32's.

**Ship `bf16` (the default).** f16 is genuinely the *more accurate* format —
10 mantissa bits against 7, and 2.8× lower mean logit delta (3.5e-3 against
9.7e-3) — so this costs real precision. It buys a failure mode that stops being
data-dependent: an f16 weight above 65504 becomes `inf`, and the only thing
standing between that and a broken model is a guard the exporter applies. bf16
needs no guard. **f16 remains the better file when the measured range fits** —
`export info` prints `max_abs` and `min_nonzero_abs`, and comparing them
against 65504 and 6.1e-5 is the decision; it ships so a stranger can make it
without rebuilding this crate.

**This is a storage format, not a compute-precision change.** Loading a bf16
export gives an **fp32** model: bf16 matmul does not exist on this CUDA backend
(ADR-0016 — the LLVM dialect has no bf16 type) and f16 matmul works but falls
back off the tensor cores without saying so. The `u16`-bit-pattern storage
primitive the export uses is the one pinned in
`vendor/burn-fused/crates/burn-gdn2/tests/lowp_bf16_cuda.rs`.

**The refusal.** `generate` and `serve` take `--export` and nothing else. Given
a training checkpoint they say what the file is and print the command that
converts it, then exit — they never train-shaped-load the optimizer section and
never go looking for the sidecar (ADR-0011: a wrong-but-plausible answer is the
cardinal sin, and a 34 GB read to obtain 30 MB of weights is that sin wearing a
plausible face). An export also cannot carry a `--engram-ram` run's host table:
that is training state, and no 18 MB file holds 34 GB of it. The in-model
Engram tables (3.15M of the 9.2M params) do travel.

Spec, evidence and the gate: [`docs/adr/0023-inference-export.md`](docs/adr/0023-inference-export.md).
Gate: `cargo test -p dormouse-train --test export_roundtrip` (and again with
`--no-default-features --features cuda`).

### Every flag on `train`

Transcribed from the real binary's `--help`, grouped. There are no other
runtime knobs except `DM_QUANT_DEBUG=1` (debug-only) — the autotuner level is
exposed as `--autotune` rather than left as an env var.

| group | flags |
|---|---|
| data / config | `--data` (required, a directory) · `--eval` · `--preset` (default `small`) · `--config` · `--set k=v` (repeatable) |
| schedule | `--steps` · `--seq-len` · `--batch` · `--log-every` · `--ckpt-every` · `--ckpt-name` · `--ckpt-dir` (default `checkpoints`) · `--eval-every` |
| optimization | `--lr` · `--wd` · `--grad-clip` · `--opt mix\|mix-adan\|adan\|adamw\|muon` · `--factors-fallback` · `--seed` |
| model arms | `--bf16[=true\|false]` · `--act-quant int4\|int8\|fp4` · `--act-group` · `--max-iter` · `--no-kda` · `--no-engram` · `--quant fp32\|bf16\|fp16\|fp8\|fp4` · `--retract-every` · `--retract-iters` · `--rand-depth` |
| aux objectives | `--jepa-weight` · `--jepa-k` is **not** a flag (K is `--dspark-k`) · `--dspark-weight` · `--dspark-k` · `--jepa-targets` · `--jepa-precompute` |
| memory offload | `--engram-ram` · `--engram-slots` (default 1 000 000) · `--host-adam-every` (default 1) |
| measurement | `--eval-batches` (default 20, over a window of `20 × batch × seq_len` bytes — 102 400 B at batch 10, 20 480 B at batch 2, **not a fixed 100 KB**) · `--eval-depths` · `--timers` · `--memlog` · `--quant-check` · `--autotune` |
| stability | `--stress` · `--stress-lr` · `--stress-every` |
| process | `--log` · `--detach` · `--guard` |

There is deliberately **no** `--no-gr` or `--no-mor`: those are config, not A/B
flags (`--set use_gr=false`, `--set use_mor=true`). `--rand-depth` and
`use_mor` are mutually exclusive and the config seam refuses the pair loudly.
**`--gen-max-iter` does not exist**, even though one flag's help string
advertises it (`crates/dormouse-cli/src/bin/train.rs:114`).

`generate` takes `--ckpt-dir --ckpt-name --prompt --steps --preset --config
--set --temp`; `serve` takes `--ckpt-dir --ckpt-name --preset --config --set
--port` (default 8000).

**What does not work, so you do not discover it the hard way:**

- `scripts/bench.sh` hardcodes this machine's corpus path
  (`/mnt/e43497ab-.../aria_data/pretrain/real_filtered_v2`) and exits 1 if the
  drive is not mounted. It is a local tool, not a portable gate.
- `vendor/burn-fused` is **not a git repository of its own** — it is a vendored
  copy inside this one, and the root `Cargo.toml` `exclude`s it (its crates are
  separate cargo workspace roots; without the exclude their `workspace = true`
  inheritance resolves against our root and fails). Consequence: the per-crate
  `.github/workflows/ci.yml` files it inherited (two survive in the tree after a
  2026-09-27 cleanup) have **never executed and cannot gate anything**. The real
  gate is
  [`.github/workflows/fused-library.yml`](.github/workflows/fused-library.yml),
  which runs cargo from *inside* the fork. Run it locally with
  `cd vendor/burn-fused && cargo test --workspace --exclude burn-fused-benches
  --exclude cpu-probe --exclude launch-probe`. Three of its files are currently
  untracked, including `burn-fused/tests/gpu_production_shape.rs`, so a fresh
  clone does not have them.
- There is **no GPU job in CI** and no registered runner — deliberately, because
  this workstation is simultaneously the trainer, the benchmark box and the
  agent workstation. The GPU check is a command you run on the GPU box:
  `vendor/burn-fused/tools/gpu-gate.sh`.

---

## Tooling

Everything operational lives in [`tools/`](tools) — these are the programs a
reader of the logs will eventually need:

| tool | what it does |
|---|---|
| [`tools/first_run.sh`](tools/first_run.sh) | one-command training launch: preset + flags, checkpoint name derived from the flags, log to `~/logs/` (persistent — `/tmp` is a 32 GB tmpfs that dies on reboot and once ate a 33.5 GB sidecar) |
| [`tools/trainboard.py`](tools/trainboard.py) | plots any run log (loss, held-out BPB, step-time stack, throughput) to a static HTML board; `--follow` tails a live run |
| [`tools/wt.sh`](tools/wt.sh) | one task per git worktree (ADR-0022) — `wt.sh new <lane>` branches from a compiling commit; a shared checkout makes `cargo test` a lottery |
| [`tools/build_lock.sh`](tools/build_lock.sh) | serializes every cargo invocation across agents and worktrees; builds count as heavy work (§1.5) |
| [`tools/determinism.py`](tools/determinism.py) + `tools/determinism/` | the cross-process seed harness: two independent tensor-level instruments; same seed → non-TSCT slots bit-identical across processes (7 951 694/7 951 694), different seed → relFro 1.414, separation ~10⁶ |
| `tests/oracle/*.py` | standalone python oracles for the spectral math (central differences, basin checks, guard discrimination) — run without Rust |
| [`tools/falsify_fused_adjoint.sh`](tools/falsify_fused_adjoint.sh) | 6 injected mutants, all DETECTED on their own assertion, byte-identical restore — the gate-testing-the-gate pattern |
| [`tools/oracle_gate.py`](tools/oracle_gate.py) + [`docs/protocols/ORACLE-TIERS.tsv`](docs/protocols/ORACLE-TIERS.tsv) | per-crate verification tier register; hand-edited only — the generator destroyed 22 rows once |
| [`tools/spot_check.py`](tools/spot_check.py) | claims-vs-artifact spot checks over the docs |
| `crates/dormouse-data/src/bin/anchors.rs` | the 5-gram bar; `--fit` scores the bar on the model's own eval file — the only comparable way |

## Data pipeline

Streams bytes (text, parquet, images, binaries — a JPEG is just bytes to
predict) through a 64 MB ring refilled in 8 MB chunks, seeded Fisher-Yates
shuffled, split deterministically. A directory that yields no files, or a corpus
smaller than one batch, stops the run loudly; data is never synthesized.
`dormouse-data` also ships three binaries: `filter` (a DCLM-style corpus filter
with a two-region mode so the eval tail is excluded from training data at filter
time — straddler documents dropped, dedup shared, ADR-0010), `shard`, and
`anchors` (the n-gram counter that produces the bars in (d) — that is the tool
that keeps this project honest).

## Performance

The whole regression history is two rows. `benches/history.tsv`:

| date | commit | mode | step | throughput | final CE |
|---|---|---|---|---|---|
| 2026-09-25 | bdddbaf | canary (2 M slots, aux off) | 4081 ms | 1.23 KB/s | 5.141 |
| 2026-09-26 | b00ae06 | canary | 150548 ms | 0.03 KB/s | 5.074 |

`scripts/bench.sh canary` appends one. **It is a convention, not a gate**: no CI
job reads the file, and a regression does not block anything automatically — the
comparison is a human step. Reported here rather than implied.

## Working rules (CONTRIBUTING)

These are the rules this project actually runs on. They are not aspirations; each
one exists because breaking it cost a run, a week, or a claim.

1. **Every degradation carries one of three marks: LOUD, COUNTED, SILENT — and
   SILENT is a defect.** LOUD = a hard error naming the cause *and* the escape.
   COUNTED = the fallback happens and a counter or log line says so, so a reader
   of the training log sees it without a profiler. SILENT = it just happened.
   The reason this class is expensive here: a silent fallback usually computes
   the *right answer*, so the run is correct and a year slower. Missing data
   stops the run and is never synthesized (one run trained 500 steps on constant
   bytes); shape mismatches assert; `--guard` is the recovery action, not a
   substitute for the error. An arm that can be entered-but-skipped needs a
   counter — that is how the fused KDA dispatch stayed dead for a year behind a
   correct-looking run.
2. **A/B or death.** A mechanism either beats its own removal on held-out BPB at
   a fixed step budget with 3 seeds per arm, or it is deleted. PonderNet was
   deleted this way. Anything inside the control's own seed spread is "no
   difference", and no difference means gone.
3. **The step's OWN body reads nothing back. The retraction does — 112 times.**
   This corrects a claim this file made hours ago. It said "no host read in
   the hot loop" with a five-row measurement table, and the table had no
   retraction-off arm, so it generalised from the forward/backward/optimizer
   body to the whole step. It was wrong.

   What the corrected claim is, and how it is measured:

   | part of a step | host reads | evidence |
   |---|---|---|
   | forward, backward, optimizer, EMA | **0** on a non-log step | the four reads at `lib.rs` are each behind a guard; 40 steps at `--log-every 10` produce 4 log lines, at `10000` produce 1; 60 steps at log 5 vs log 10000 cost 142 s vs 141 s; `--timers` on/off 142 s vs 142 s and `--timers` *adds* a `try_into_scalar` |
   | **TSCT retraction** | **112 per step, unguarded, uncounted** | 16 TSCT factors at `small` (3 experts × 2 LinearLike × 2 factors, plus `out_proj` and `lm_head`) × 7 `into_scalar` at `burn-spectral/src/lib.rs:186,197,198,649,653,656` |

   The retraction's 112 syncs are the 52.8 ms it costs: 52.8 ms / 112 ≈ 0.47 ms
   per sync, which is what a blocking device read costs. `--retract-every 1000`
   gives `retr = 0.0` and a 188 ms step against 240 — the difference is those
   reads, not the arithmetic. **The sync is the cost, not the math:** a reviewer
   showed the `σ_max` power iteration inside the retraction is provably
   unnecessary — a unit-Frobenius input has `σ_max ≤ 1` by Cauchy–Schwarz, and
   the cubic's basin is exactly `[0, 1]` — so those reads can simply not be
   needed. That is a 5-line change, against 52.8 ms → ~17 ms.

   So the honest statement is: **the step's own body is sync-free; the
   retraction is a fixed 112-read tax, and it is 22% of a warm step at batch 8.**
   A `retract_batched` implementation that removes them already exists
   (`burn-spectral/src/lib.rs:275`), is unit-tested, and **is never called** —
   the tests are the only callers.

   **And it must not be wired as written.** It takes `&mut [&mut Tensor<2>]`,
   cannot reach `Param::from_mapped_value`, and `polar_orthogonalize_batched`
   contains no `.detach()`. The scalar path documents at length why that
   matters: a stored non-leaf is silently downgraded to an untracked leaf and
   **the master freezes** — and all four batched tests run on `Device::ndarray`
   where `is_require_grad` is never true, so they would pass while the model
   stopped training.
4. **A bit-for-bit claim must name the external source.** "Verified against the
   reference" is not a claim; "verified against `NVlabs/GatedDeltaNet-2`,
   `lit_gpt/gdn2_ops/fused_recurrent_gdn2.py`, tolerance 5e-4, harness at
   `tests/bit_exact.rs`" is. Where no reference code exists, the crate's doc
   comment says *that* instead of implying verification it does not have. An
   absolute tolerance wider than the effect it measures is not a gate.
5. **A verification claim must say which part of a run it covers.** "fp4
   quantization verified, 100 steps, 0 NaN" is false if the attention path ran
   at 8 bits. Name the configuration, the steps, the shape, the machine, and
   the part of the pipeline the test actually exercised. A claim that does not
   say what it covered is a claim about nothing.
6. **One heavy thing at a time.** One GPU process. One build. This workstation
   has frozen three times, and two legal processes colliding corrupted the
   memory pool and wrote garbage weights into a checkpoint.
7. **Never fix unrelated things in the same diff.** Follow-ups get reported,
   not smuggled.

## Docs

| doc | what |
|---|---|
| [`AGENTS.md`](AGENTS.md) | agent-facing repo state: measured numbers, CUDA quirks, every knob |
| [`docs/glossary.md`](docs/glossary.md) | the vocabulary ("fused", "arm", "sidecar", "iteration", "Engram") and the live list of places a document and the code disagree |
| [`docs/architecture/PLAN.md`](docs/architecture/PLAN.md) | program plan, phase ladder, verdicts, north star |
| [`docs/protocols/AB-PROTOCOL.md`](docs/protocols/AB-PROTOCOL.md) | the measurement instrument and the A/B queue (unrun) |
| [`docs/adr/`](docs/adr/) | ADR-0001..0022 — every recorded decision, including the retracted ones |
| [`docs/archive/audit-2026-09-25.md`](docs/archive/audit-2026-09-25.md) | codebase verdict audit: kill list, missing A/Bs |
| [`docs/architecture/design-minimal.md`](docs/architecture/design-minimal.md) | the minimal-architecture target |
| [`docs/architecture/mixture-arms.md`](docs/architecture/mixture-arms.md) | three priced training arms |
| [`docs/research/2026-09-27-oracle-audit-renamed.md`](docs/research/2026-09-27-oracle-audit-renamed.md) | what the library's verification claims are actually worth (read this before trusting any of them) |
| [`docs/research/2026-09-27-pytorch-baseline-renamed.md`](docs/research/2026-09-27-pytorch-baseline-renamed.md) | every number in (a), with the two ways it can mislead you |
| [`docs/research/2026-09-27-scale-ladder-renamed.md`](docs/research/2026-09-27-scale-ladder-renamed.md) | the 1B arithmetic in (e) |
| [`docs/research/2026-09-27-domain-anchors-renamed.md`](docs/research/2026-09-27-domain-anchors-renamed.md) | per-domain n-gram bars |
| [`docs/architecture/post-training.md`](docs/architecture/post-training.md), [`docs/architecture/bf16-plan.md`](docs/architecture/bf16-plan.md) | post-training loop, precision plans (in Russian) |

## License

**MIT** — see [`LICENSE`](LICENSE) for the full text, and
`Cargo.toml`'s `license = "MIT"` (which every crate manifest inherits).
Copyright (c) 2026 sehaxe.

The license history is worth knowing: the file shipped as MIT (2026-08-28),
was switched to AGPL-3.0 on 2026-09-28 to match what the crate manifests
already declared, and was returned to **MIT on 2026-10-01 by owner decision**.
The vendored forks of [burn](https://github.com/tracel-ai/burn) and
[cubecl](https://github.com/cubecl/cubecl) keep their upstream MIT/Apache
notices (restored in `4a472a8`); distributing them under MIT changes nothing
for them and their notices travel with the copies.

Third-party components vendored in this tree, with their own licenses and
notices, which must be preserved:

| path | what | upstream license |
|---|---|---|
| `vendor/burn-fused/` | 28-crate fused-kernel library, a fork of burn | MIT |
| `vendor/cubecl-fix/` | patched `cubecl-runtime`, `cubecl-server`, `cubecl-cuda` | MIT |
| `vendor/cubek-fix/` | patched `cubek-reduce` (top-k index fix, ADR-0015) | MIT |

> **Known defect, reported not fixed:** `vendor/burn-fused/LICENSE:3` and
> `vendor/cubecl-fix/LICENSE:3` both read `Copyright (c) 2026 sehaxe`, which
> drops the upstream copyright notices that MIT §a requires be retained
> (`Copyright (c) 2023 The Burn Project` / `The Cubecl Project`). Those files are
> not this task's to edit; whoever owns `vendor/` should restore the upstream
> lines and keep the fork's own additions attributed separately.
