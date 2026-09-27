# dormouse program plan

Started 2026-09-21 from the architecture grill session. Decisions live in `docs/adr/`, shared language lives in `CONTEXT.md`, evidence lives in `research/2026-09-21-per-gb-sota.md`. Update this file as phases close; do not re-litigate recorded ADRs without new evidence.

## ARCHITECTURE v2.1 (2026-09-26, scale-gated completeness — owner directive)

Principle: EVERY mechanism lives in the architecture from birth; presets gate
ACTIVATION by scale. "Best architecture in the world for any size" =
mechanisms whose cost/benefit is scale-tuned, not mechanisms added by
rewrite later. At equal compute, the tuned gate stack beats any dense peer.

The complete mechanism set (all in the codebase, config-gated):
1. **Recursion routing (MoR-style)** — lightweight per-loop router, top-k
   token continuation, recursion-wise KV. GATE: off at nano/small (routing
   underperforms vanilla <135M per 2507.10524), ON at base/one_b/p150.
   Replaces the deleted PonderNet (no lambda, no KL, no collapse mode).
2. **KDA -> GDN-2** (2605.22791, NVIDIA; FLA fla/ops/gdn2 MIT kernels) —
   channel-wise erase b_t + write w_t gates on top of our KDA: strict
   superset, matched-table win (53.11 vs 52.28; MK-NIAH 37.8 vs 28.0).
   Port cost ~300-800 lines, verified bit-for-bit against FLA.
3. **MA value-path memory** (2609.28399) — V += RMSNorm(M[fnv(3/5/8-gram)])
   inside the attention value construction; hashed tables keep the capacity
   story; CPU-resident (our offload). Structurally gradient-theft-proof.
4. **DeepLoop residual scaling** (2607.13491) — tied-depth init rule
   alpha=(2N)^1/2, beta=(8N)^-1/2; A/B against the learned residual_scale.
5. **FlashLoop lazy inference** (2609.29812) — phase 5: token-sparse loop
   updates, KV-residual quantization.
6. **Self-play data engine** (2609.30063) — phase 5+: generator/learner
   byte curriculum at the learner's frontier (the compute-bounded data
   source; the RSI seed).
7. **Post-training** (phase 6): Rufus-Air ordering (2609.29421) —
   verifiable rewards first.

Scale gates (the presets decide, the code is size-blind):
| mechanism | nano/small | base | one_b/p150 |
|---|---|---|---|
| fixed loop 4 | on | on | on |
| recursion routing | off | on | on |
| MA value-path memory | on (compact) | on | on |
| GDN-2 gates | on | on | on |
| FlashLoop inference | phase 5 | phase 5 | phase 5 |

## ARCHITECTURE v2 (2026-09-26, research-grounded redesign)

Measured failures that force it: lambda collapse x2 (fake loss, uniform eval),
engram gradient theft (core -> constant outputs), domain-sorted corpus
(meaningless train CE), the pre.4 inference-forward regression (split-brain
probe: train loss 0.005 real vs inference logits constant ~0 on the same
weights), lr 1e-4 starving Muon+.

The v2 stack, each piece tied to a source:
1. **Core: fixed-depth LoopBlock, max_iter=4** (use_halting=false default).
   PonderNet halting retired: our p_n lacked the remainder term (sum(p) was
   free to vanish -> rec = sum(p*CE) became a fake loss and out_acc -> 0).
   The prior-weighted KL fix is kept for any re-entry. Re-entry candidate:
   MoR routers (2507.10524) — token-level expert-choice depth without
   probabilistic halting; below vanilla at 135M, Pareto-competitive from
   360M — our scale is 7.5M, so fixed-depth first, MoR as a measured A/B.
2. **DeepLoop residual scaling A/B** (2607.13491): tied-depth visits change
   the stability exponent; our learned residual_scale -> A/B against the
   DeepLoop init alpha=(2N)^1/2, beta=(8N)^-1/2 at N=4.
3. **Memory: MA-style value-path lookup** (2609.28399) replaces the separate
   gated Engram arm: in KDA, V = V_proj + RMSNorm(M[fnv(3/5/8-gram)]) — the
   hashed n-gram tables keep the capacity story (Qwen 51B-row style), the
   core's K stays in every value (gradient-theft-proof by construction),
   tables stay CPU-resident (our offload machinery). Reference impl:
   fla/layers/memory_attn.py.
4. **Data: corpus v3 sharded** (done, shard.rs). Next: Self-Play pretraining
   (2609.30063) — generator/learner byte curriculum at our exact scale
   (<25M params), the compute-bounded data engine.
5. **Inference (phase 5): FlashLoop** (2609.29812) lazy updates on the looped
   block — token-sparse recursion, KV-residual quantization; 1.64x/6x
   reported, training-free.
6. **Post-training (phase 6): Rufus-Air ordering** (2609.29421) — verifiable
   rewards first, judges later; matches POST_TRAINING.md.

## RESEARCH MAP 2026-09-26 (four fresh papers, owner-directed)

1. **Memory Attention (2609.28399, Kang; code: Joluck/memory-attention)** — the
   Engram redesign: V = K0 + RMSNorm(M[s]) — a token-keyed learnable table
   folded INTO the attention value path, no separate value projection, no
   separate gated arm. At inference the norm folds into the table (lookup +
   add). CPU-offload with prefetching = our engram-ram pattern. Reported:
   WikiText ppl 31.55 -> 28.64 at matched tokens (with added table params).
   Dormouse synthesis: keep the N-GRAM hashed keys (capacity, the Qwen story)
   but move the lookup INTO the KDA value path (V += RMSNorm(M[fnv(ctx)])) —
   structurally gradient-theft-proof (the core's K is always in the value).
   The separate gated Engram arm is retired (its A/B lost: uniform death).
   Author's repo: fla/layers/memory_attn.py (FLA-based, Apache-2.0).
2. **Self-Play Pretraining with Zero Data (2609.30063, Goodman/Levine)** —
   generator writes programs, a UTM executes them into bytes, the learner
   predicts bytes with plain CE, the generator is RL-trained on
   gradient-alignment reward (preconditioned grad magnitude vs a lookback
   checkpoint). Zero-shot transfer to natural data at <25M params / 4K ctx —
   OUR SCALE. The data-side revolution for dormouse: a self-play curriculum
   engine replacing/augmenting the corpus (owner's "RSI ladder" seed).
3. **FlashLoop (2609.29812)** — looped-transformer inference: token-sparse
   loop updates (later loops change few tokens), attention-column sparsity,
   KV-residual quantization; 1.64x speed, 6x KV reduction, ~lossless. Phase-5
   inference lever for our LoopBlock; also evidence our max_iter=4 is deep
   enough (later loops change little).
4. **Rufus-Air (2609.29421)** — an 8-stage post-training recipe (SFT ->
   verifiable-reward RL stages -> RLHF) on a 106B base. Phase-6 template for
   our post-training ladder; the reward-reliability ordering principle
   matches POST_TRAINING.md.

## CRITICAL FINDING 2026-09-26: the Engram eats the core's gradient

official_v3 (fp32, sharded corpus): train CE -> 0.10 (Engram memorizes the
stream at lookup speed) while held-out EVAL froze at EXACTLY BPB 8.000 —
a uniform distribution over 256 bytes — from step ~1000 onward. The core
model collapsed to constant outputs on unseen data: the memory explains the
targets by itself, the residual gradient reaching the core is ~0, and an
ungradiented core decays to uniform. Every earlier run's flat eval curve was
this same failure seen too early to recognize.

Probe running: core_probe (identical run, --no-engram, 3k steps, eval@250).
If its eval descends, the core learns language when the memory is silent —
and the fix is one of: (1) core-first warmup with the engram off, (2) an
engram output gate initialized closed (the memory joins only where it beats
the core on the residual), (3) a row-lr diet. The Qwen report's rule — memory
is auxiliary and gated, the core carries the language — is the design law
here.

## Status

2026-09-26: pre.4 merged to main; the official fp32 baseline (corpus v2, small,
8M slots, iter4, 100k steps) is RUNNING (log: ~/logs/official_baseline.log,
ckpt: ~/official_baseline). All optimization A/Bs pause-bench-resume against it.

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
1b. burn 0.22.0-pre.4 migration — **LANDED 2026-09-24 on branch `pre4-migration`** (worktree; sequenced around the fused work per plan). What landed: all burn/cubecl version reqs bumped; the vendored cubecl-fix fork rebased onto cubecl 0.11.0-pre.4 (upstream moved the memory pools into a new `cubecl-server` crate; the stale-page #1401 fix was re-derived there — slot-preserving exclusive/sliced pools, cleanup frees in place, `Stale binding` identity check; the 3 repro tests ported, 128/128 cubecl-server lib tests green); API migrations for #5557 (`Tensor::no_grad` -> `detach`; `Module::no_grad` survives so the aux.rs EMA freeze is untouched) and #5647 (`AutodiffTensor.primitive/.node` private -> accessors; non-Clone `NodeGuard` -> the fused backward pads with fresh guards); fused kernel signature sweep across the 30 vendored crates + dormouse fused/. Gates green on CPU: lib tests (14 core + 26 train), model_seam (5 + 1 slow-ignored), cli check, cuda compile check, full vendor/burn-fused workspace compile. Follow-ups: (1) GPU verification deferred while a run holds the card — the 22 fused cuda GPU tests plus a short BPB parity run vs the pre.3 build; (2) burn-ndarray is deprecated upstream, the flex-backend CPU-test migration is now on the path; (3) re-check the burn-msa `gradient_detach` leak note against the new autodiff; (4) pre.4 upsides to wire when convenient: topk backward (#5531), scatter-add atomics (#5621), fusion perf (#5620/#5622/#5625), batched SVD retraction (#5259).

Merged to main 2026-09-26. The GPU blocker resolved as: NOT the fork — upstream-vs-fork isolation ran clean while our stack died, and the culprit was the MSA arm (pre.4 topk/indexing feeds garbage block indices; gather OOB up to 11.5 GB, sanitizer-verified; repro in vendor/burn-fused/crates/burn-msa/examples/msa_repro.rs). With MSA disabled (ADR-0012: presets use_msa=false, fused kernel env-gated off) the pre.4 stack + the #1401 slot-preserving pool port runs green (20-step smoke, corpus v2, ce descending, zero driver errors). Follow-up (2) above now reads: the 22 fused cuda GPU tests + MSA-vs-dense A/B whenever the GPU window opens.

2. Precision map per component (2026-09-23, corrected 2026-09-24: TSCT and BitNet tracks were BORN quantized — the blanket "bf16 ban" mispainted them):
   - TSCT masters: fp32 (polar retract needs it), forward reads fp8 on sm_120 — already the design, nothing to change.
   - FFN activations: BitNet fp4 act-quant — VERIFIED (100 steps, 0 NaN, convergence == fp32); `--act-quant fp4 --act-group 128`. Attention activations: int8 (max(bits,8)). Full speed arrives with the fused kernels (STE in-kernel); on the burn path today it verifies correctness, not speed.
   - Engram rows + host-Adam moments (host RAM): THE actual fp32 fat. fp32 (37 GB resumed at 48M) -> bf16 (~19 GB) -> rows int8 (~10 GB). Diet of `offload.rs` only.
   - KDA state: bf16 (FlashKDA trains bf16 state).
   - fp32 stays ONLY where it is correctness: GEMM accumulation, final norm + lm_head + logits (the documented NaN rule), tiny gate params (A_log, dt_bias), model init, TSCT masters (ortho math).
   - fp8 big GEMMs after item 1; fp4 GEMMs when cubecl supports sm_120 fp4 (the card has the tensor cores).
   Quality gate: BPB parity A/B at 2M slots for every precision step.
2b. Fusion backend (2026-09-23 finding): our backend is `Autodiff<Cuda>` with NO fusion wrapper — every elementwise chain (KDA recurrence ops, casts, gates, norms) runs unfused, paying full read+write per op on a 448 GB/s bus; this is the likely fwd/bwd whale alongside the GEMM dtype. burn-cuda has a ready `fusion` feature (`Cuda = Fusion<CubeBackend>`), but the vendored extension crates write custom ops against the bare Cube backend and do NOT compile under Fusion. BLOCKED AGAIN 2026-09-26 (deeper than #5673): with fusion on, burn's Tensor becomes the runtime-dispatch type (DispatchTensor), and all five vendored extension crates (kda, rmsnorm, msa, muon-plus, spectral) downcast the bare CubeBackend — 48x `DispatchKindConversion<CubeBackend> unsatisfied`. The port is per-crate dispatch-kind work (Rung-scale), branch `fusion-flip` parked with the flip applied. Original line: flip `burn-cuda/fusion`, canary A/B (0.21 blog: launch overhead down 5.4x avg, up to 8.2x small shapes).
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
