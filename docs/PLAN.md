# dormouse program plan

Started 2026-09-21 from the architecture grill session. Decisions live in `docs/adr/`, shared language lives in `CONTEXT.md`, evidence lives in `research/2026-09-21-per-gb-sota.md`. Update this file as phases close; do not re-litigate recorded ADRs without new evidence.

## North star (2026-09-25)

The model must be able to code and code its own updates. Not recursive
self-improvement — a code-capable byte LM whose patches enter the repo only
through the verification harness (tests + bench + A/B). Implications, in
order: (1) the official fp32 baseline on corpus v2, (2) a code-domain corpus
mixture (bytes need no architecture change), (3) SFT on code tasks, (4) the
RLVR self-evolve loop with cargo-test/bench as the reward oracle
(POST_TRAINING.md §self-evolve). Every speed/size win below directly buys
this loop: faster steps = more RLVR rollouts per wall-clock hour.

## Mission

Maximum capability per gigabyte on one 16 GB GPU (RTX 5060 Ti) plus 64 GB host RAM, byte-level. Every mechanism must scale when hardware grows: more RAM buys Engram rows, more VRAM buys a bigger preset, more machines buy data parallelism. Win on efficiency, not budget (ADR-0007). The goal is a reasoner, not a reciter: the core learns computation, Engram stores facts cheaply, and phase 5 installs reasoning with verifier-graded objectives (ADR-0008).

## Scorecard

- **Score**: BPB on the eval tail at a fixed step budget and a fixed memory envelope. One number, comparable across any config (ADR-0001).
- **Guardrail**: step time within 1.2x of the baseline config, or the result is tagged "slower but better" and must justify itself.
- **Long gate**: BPB at distance (bytes at 512k-1M positions vs 1-4k) after each context-extension rung (ADR-0004). Integration across 512k bytes cannot be won by local n-gram recall, so this gate leans toward computation.
- **Memorization guardrail**: every confirm run reports the train-vs-eval BPB gap. A win that comes from recall rather than computation does not count (ADR-0008).
- Small-preset rankings predict large-scale winners about 80% of the time (DataDecide, arXiv:2504.11393), so every cheap A/B here is a vote on the flagship.

## Protocol: A/B or death (ADR-0002)

1. **Smoke**: 200-500 steps at small preset, batch 10, s512. Screens NaN, tokens/s, early loss slope. A filter, never a verdict.
2. **Confirm**: 2k+ steps for smoke survivors. The BPB verdict.
3. **Verdict**: beat the removal arm or die. A tie deletes the mechanism.
4. Every confirm run writes a resolved-config snapshot; resumed runs diff against it (drift check).

## Phases

### Inference budget (the delivery contract, phase 5 + bench ladder)

Weights at inference live in the component's own format, never bf16: fp8 factors (verified in production), ternary dense via burn-bitnet, int8 Engram rows.

| Metric | Target (flagship: p150 core + 48M-row Engram int8, 1M ctx) |
|---|---|
| VRAM, core | **~160 MB fp8-everywhere** (today-realistic: fp8 TSCT verified; dense fp8 = offline ckpt quant) → **~70-100 MB** with ternary dense (BitNet wiring, phase 5) |
| Host RAM, total | ≤ 5 GB (Engram rows int8: ~4.6 GB; no optimizer states at inference) |
| Context memory | fixed KDA state + sparse block indices — no growing KV cache |
| Throughput | hundreds of tok/s on the 5060 Ti, batch 1, after fusion |
| Multipliers | adaptive depth (easy tokens halt early), DSpark speculative decode (×2-3), fp8 rows |
| Measured by | an inference bench added to `scripts/`: tok/s, p50/p99 latency at ctx 1k/32k/1M, RAM, VRAM |

Training keeps fp32 masters; the inference checkpoint is quantized offline (masters -> fp8/ternary). Every number above gets a measured line in `benches/history.tsv` before any public claim. The $500-box offline chat demo is this table, delivered.

### Phase 0: clean base (complete 2026-09-21)

Everything below landed the same day. Bonus fixes beyond the original list: the aux EMA teacher now runs `no_grad()` (was doubling activation memory via a grad-tracked teacher and leaking ~30 MB/step of param history; this is what killed every aux-on run), RMSNorm weights are trainable again (`Param::initialized` inherited `require_grad=false`; found by the seam tests), the cpu build compiles again (broken since the M4 fused commit), the preset registry is gone (presets are files), and `configs/p150.toml` pins the flagship at 159,720,195 core params.

| Item | Exit criterion | Status |
|---|---|---|
| Kill list C3: dead crates, duplicated server, unused deps, tmp examples, stale README | cargo check + lib tests green | done, -1,503 LOC |
| Config hybrid (ADR-0005): one resolve seam, defaults written once, snapshot + drift check, flat TOML | train/generate/serve cross one seam; the `--set bf16` clobber and the missing validate are fixed with tests | done, 41 lib tests |
| fused/ smoke (ADR-0003): `DM_FUSED=1` vs `0` tokens/s where both paths run | verdict recorded; 6.5k LOC kept or deleted | done, kept (1.7-2.0x) |
| Data kickoff: quality filter over the 44 GB corpus, dedup, eval tail grown 30 MB to ~500 MB | filtered corpus v1 serialized, filter stats reported | done: 46.2 GB -> 19.7 GB, 272M docs -> 104.2M kept, 28.9% dups removed |
| Tests through the model seam: `forward_with_hidden` smoke, determinism, long-sequence KDA stability, checkpoint round trip, on the CPU backend | suite runs in minutes without a GPU | done, 7 tests (1 slow ignored) |

### Phase 1: performance

Baseline (2026-09-21, honest: aux fix + fine timers, quiet GPU): 6.7-8.3 s/step at small/batch10/s512 with aux on, engram-ram 48M slots, host-adam every step. 5 KB/step ≈ 0.7 KB/s. Phase split at step 100: fwd 2.2 s, bwd 4.3 s (incl. loss sync + host-adam D2H), opt 111 ms, retract 105 ms, ema 5 ms, data 0.1 ms.

Diagnosis: the step is GPU-kernel-bound, not launch-bound. The step-100 sync returns after 4.3 s of queued GPU work, so the card is busy, not starved. The cost lives in the kernels: fp32 tensor-op KDA, unfused MSA/Engram bridges, TSCT factor matmuls, fp32/bf16 cast copies — roughly 45x off the card's compute ceiling for 7.5M params. The fused path's 1.7-2.0x came from replacing exactly these kernels.

Attack ladder, ordered by leverage:

0. Fine-grained step timers: done (fwd/bwd/opt/retr/ema split, sync-anchored).
0b. max_iter A/B (4 vs 8) — VERDICT (2026-09-24): **max_iter=4 replaces 8.** Confirmed over full 1000-step runs at 8M slots: 2967-3054 ms/step vs 5674 ms (2.13x, physics matched), zero NaN across the entire distance where iter8 hit the documented overfit-NaN at step 193 and died without a resumable checkpoint. Held-out BPB at this horizon is ~7.99 (near-uniform) on the raw stream for the survivor — the raw corpus is unlearnable-generalization-wise at these step counts, which promotes pretrain-v2-on-filtered-corpus from quality-wish to stability-requirement. Consequence: the halt head loses its host config (max_iter=4 default); the PonderNet-halting knife A/B follows per the research verdict.
1. fused/ to the flagship recipe: host rows, JEPA targets, bf16, act quant, GR (measured 1.7-2.0x on the plain path; ADR-0003). Acceptance criteria before fused becomes a default: (a) the fused backward must produce gradients for exactly the same parameter set as the burn path (checkpoint-size evidence from the 2026-09-21 smoke says it does not today: fused ckpt 37.1 MB vs 69.7 MB, ~33 MB of optimizer moments missing, i.e. params whose grads never arrived); (b) BPB parity with the burn path on a confirm-tier run; (c) full checkpoint round trip through the fused path.
1b. burn 0.22.0-pre.4 migration — **PARKED (2026-09-24)**: the migration itself is COMPLETE and green on CPU (branch `pre4-migration`, 6 commits: cubecl fork rebased onto 0.11.0-pre.4 with the #1401 stale-page fix re-derived for the new cubecl-server layout, 128/128 pool tests; burn bumped, no_grad->detach migrated; JEPA mask flake fixed). BLOCKER found at the GPU gate: the pre.4 stack dies on sm_120 at startup with `cuEventCreate -> DriverError status 700 (IllegalAddress)` — reproduced twice, while the pre.3 binary runs clean on the same GPU in the same minute. Either an upstream cubecl 0.11.0-pre.4 regression on Blackwell consumer or a fork-rebase break in the CUDA path; isolation (upstream cubecl without our patch, minimal repro) is a vendor-debug day. What pre.4 would have unlocked: fusion backend for extension crates (#5673 — the elementwise-chain fusion lever), topk backward, broadcast-grad single-pass, scatter atomics, stream-poison fixes. Revisit when upstream cubecl ships an sm_120 fix; the branch and the port notes are the revival kit.
2. Precision map per component (2026-09-23, corrected 2026-09-24: TSCT and BitNet tracks were BORN quantized — the blanket "bf16 ban" mispainted them):
   - TSCT masters: fp32 (polar retract needs it), forward reads fp8 on sm_120 — already the design, nothing to change.
   - FFN activations: BitNet fp4 act-quant — VERIFIED (100 steps, 0 NaN, convergence == fp32); `--act-quant fp4 --act-group 128`. Attention activations: int8 (max(bits,8)). Full speed arrives with the fused kernels (STE in-kernel); on the burn path today it verifies correctness, not speed.
   - Engram rows + host-Adam moments (host RAM): THE actual fp32 fat. fp32 (37 GB resumed at 48M) -> bf16 (~19 GB) -> rows int8 (~10 GB). Diet of `offload.rs` only.
   - KDA state: bf16 (FlashKDA trains bf16 state).
   - fp32 stays ONLY where it is correctness: GEMM accumulation, final norm + lm_head + logits (the documented NaN rule), tiny gate params (A_log, dt_bias), model init, TSCT masters (ortho math).
   - fp8 big GEMMs after item 1; fp4 GEMMs when cubecl supports sm_120 fp4 (the card has the tensor cores).
   Quality gate: BPB parity A/B at 2M slots for every precision step.
2b. Fusion backend (2026-09-23 finding): our backend is `Autodiff<Cuda>` with NO fusion wrapper — every elementwise chain (KDA recurrence ops, casts, gates, norms) runs unfused, paying full read+write per op on a 448 GB/s bus; this is the likely fwd/bwd whale alongside the GEMM dtype. burn-cuda has a ready `fusion` feature (`Cuda = Fusion<CubeBackend>`), but the vendored extension crates write custom ops against the bare Cube backend and do NOT compile under Fusion. BLOCKED on item 1b: pre.4 #5673 auto-generates Fusion implementations for backend extensions. After the migration: flip `burn-cuda/fusion`, canary A/B (0.21 blog: launch overhead down 5.4x avg, up to 8.2x small shapes).
3. Overlap the host-Adam D2H sync with compute on a separate stream; prefetch the next batch's rows.
4. Retract cadence by measurement: 105 ms/step today is minor; revisit only after 1-2 land.
5. Longer sequences (s1024/s2048) to amortize per-step overhead over more bytes.

Target: 5x+ bytes/s at the flagship config (realistic ceiling ~10x). Every later A/B gets proportionally cheaper.

### Performance budget (hard numbers, enforced by benches/bench.sh)

| Metric | Now (2026-09-21 honest) | Budget | Stretch |
|---|---|---|---|
| Step time, flagship (small/b10/s512, aux on, 48M rows) | 6.7-8.3 s | ≤ 1.7 s | 1.3 s |
| Bytes/s, same config | ~0.7 KB/s | ≥ 3 KB/s | 7 KB/s |
| RSS, fresh 48M-slot run (after the bf16 diet) | 18.4 GB + burn | ≤ 12 GB | 8 GB |
| RSS, resumed 48M-slot run (after the bf16 diet) | 37 GB | ≤ 20 GB | 14 GB |
| VRAM, flagship | ~11.8 GB | ≤ 12 GB | 9 GB (for batch growth) |
| Resume sidecar, 48M rows | 33.5 GB | ≤ 10 GB | 6 GB |
| cpu lib tests | ~7 min (opt-3 deps) | ≤ 5 min | 3 min |

`scripts/bench.sh` appends a metrics line (date, commit, s/step, bytes/s, peak RSS, peak VRAM) to `benches/history.tsv` on every run: the canary (small, aux off, 2M slots, 30 steps — runs anywhere, ~2 GB RSS) before/after every optimization, the full flagship bench before/after every phase. A regression against the last entry blocks the merge.

### Phase 2: data

The largest documented lever at fixed compute (arXiv:2406.11794, 2406.17557, 2407.01492):

1. Quality-filter (edu-classifier style) and dedup the corpus; re-serialize to bytes with document boundaries kept.
2. Mixture sweep via RegMix-style small runs; pick by BPB.
3. Train up to 4 epochs on the filtered corpus; repetition past 4 is waste (arXiv:2305.16264).

Exit: filtered corpus beats raw corpus BPB in a confirm-tier A/B.

### Phase 3: architecture (ranked)

1. **Patched trunk** (SpaceByte/BLT style): bytes grouped into patches for core steps, prediction stays byte-level. Closes most of the byte-vs-BPE gap at fixed compute (arXiv:2404.14408, 2412.09871, 2604.27263).
2. **Halt head to depth routing**: A/B PonderNet halting against MoR-style routers or fixed depth (arXiv:2608.22347, 2507.10524).
3. **KoLeo removal** A/B; verdict likely confirms deletion.
4. **TSCT vs plain small experts** at matched active params (arXiv:2409.02060).
5. **Fine-grained MoE**: more small experts, token-choice, no shared expert.
6. **Energy head** (EBT InfoNCE over the 256 byte candidates) on the loop state, weight 0 default (arXiv:2507.02092).
7. **MSA long gate**: the arm lives or dies by the long gate, not by s512.
8. **Engram growth** toward the RAM ceiling with collision-load monitoring (arXiv:2606.08347 if collision-bound).

### Phase 4: context ladder

4k, 32k, 256k, 1M. One extension phase per rung, long gate after each. Needs phase 2 document packing. MSA and the block indexer are the arms this ladder validates.

### Phase 5: post-training, intelligence over recall (unscheduled)

Goal: a chatbot that reasons, not one that recites. Ordered:

1. **SFT on reasoning traces** in a byte-level chat format.
2. **RLVR**: verifiable graders (math, code, logic). Critical thinking is installed here; memorized text earns nothing from a verifier.
3. **Distillation** from a reasoning teacher near 2.5x student size, reverse KL on-policy (arXiv:2502.08606, 2306.13649): transfers thinking patterns at a fraction of RL compute.
4. **Self-evolve**: verifier-filtered rollouts (the EGGROLL line in POST_TRAINING.md).

RSI ladder, one rung at a time, each with an entry condition:

- **Rung 1, process automation**: agents run smokes and confirms and edit ADRs from BPB alone. Entry: the phase 0 protocol running unattended.
- **Rung 2, data flywheel**: model BPB slices steer the filter and mixture. Entry: phase 2 done.
- **Rung 3, verifier-filtered self-improvement**: items 2-4 above.
- **Rung 4, model-scored architecture search**: the energy head predicts BPB of patches before training. Entry: energy-head A/B positive.
- **Rung 5, self-rewriting stack**: a model improving its own code. Out of scope for this hardware class; revisit only with a code-capable model.

## Verdicts (ADR-0006)

| Verdict | Items |
|---|---|
| Keep | LoopBlock, Engram input-side, Muon+ RMS-matched, JEPA 0.05, DSpark, ~3:1 linear:full blend, bf16 storage + fp32 heads |
| Knife via A/B | PonderNet halting, KoLeo, fp8-forward-by-default, batch-size warmup |
| Pending A/B | MSA at short context, TSCT bodies, energy head, patching |
| Never | Output-side lookup tables, expert-choice routing, sub-4-bit training |

## Flagship: the 150M program (current definition)

Goal: the smartest ~150M-parameter model that exists, on falsifiable targets, not vibes:

1. **Reasoning**: competition math and code via RLVR post-training; beat GPT-3-175B-era flagships on MATH-class suites at 150M. The distilled-1.5B line already reaches o1-mini-class results; sub-200M is the open frontier.
2. **Knowledge with memory**: factual probes where Engram (2-4B rows in RAM) carries facts and the core carries computation, compared against dense models 10-20x the core size.
3. **Long context**: 1M bytes, long gate passed.

The stack: 150M active core (fine-grained MoE or MoR-style recursion), Engram input-side 2-4B rows, patched byte trunk, 3:1 KDA:MSA hybrid, Muon+, filtered corpus with verifier-filtered epochs, distillation from a downloaded open reasoning teacher, test-time compute through the energy head. A `p150` preset pins the config (trivial now with the flat schema). Phases 1-4 build the substrate; phase 5 installs the thinking; the small-preset A/B ladder stays the prediction instrument (DataDecide transfer).

## Risk register

- Overfit NaN episodes on this box: measured 2026-09-21, the raw-corpus stream at the flagship config reaches train ce 0.845 (BPB 1.22) by step 100, so the overfit zone arrives almost immediately. The filtered corpus and the A/B ladder are the mitigation; aux-fix verification must cross the ce<1 window; `--quant fp32` is the rollback arm; stress protocol on every new config.
- cubecl memory pool is high-water: cleanup cadence, init before first allocation.
- burn-msa `gradient_detach` leaks autodiff nodes: keep it false.
- Never dynamically slice a 4D autodiff tensor on sm_120: CUDA_ERROR_ILLEGAL_ADDRESS.
- fused/ backward holds acknowledged approximations (gradcheck limit 2.0): tolerate only until its smoke verdict.
- Host-Adam every step costs +2.3% step time (measured): trade to `--host-adam-every 5` only when step-time-bound.
- The corpus drive must be mounted.
