# dormouse program plan

Started 2026-09-21 from the architecture grill session. Decisions live in `docs/adr/`, shared language lives in `CONTEXT.md`, evidence lives in `research/2026-09-21-per-gb-sota.md`. Update this file as phases close; do not re-litigate recorded ADRs without new evidence.

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
1. fused/ to the flagship recipe: host rows, JEPA targets, bf16, act quant, GR (measured 1.7-2.0x on the plain path; ADR-0003). Acceptance criteria before fused becomes a default: (a) the fused backward must produce gradients for exactly the same parameter set as the burn path (checkpoint-size evidence from the 2026-09-21 smoke says it does not today: fused ckpt 37.1 MB vs 69.7 MB, ~33 MB of optimizer moments missing, i.e. params whose grads never arrived); (b) BPB parity with the burn path on a confirm-tier run; (c) full checkpoint round trip through the fused path.
2. bf16 storage/compute on the flagship recipe: halves memory traffic; today's cast-copy tax disappears inside fused kernels.
3. Overlap the host-Adam D2H sync with compute on a separate stream; prefetch the next batch's rows.
4. Retract cadence by measurement: 105 ms/step today is minor; revisit only after 1-2 land.
5. Longer sequences (s1024/s2048) to amortize per-step overhead over more bytes.

Target: 5x+ bytes/s at the flagship config (realistic ceiling ~10x). Every later A/B gets proportionally cheaper.

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
