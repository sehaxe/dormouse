# Byte-LM training recipes for dormouse (research pass, 2026-09-26)

**Our situation (given):** 7.5M-param byte LM (vocab 256), one shared looped block (max_iter=4), online single-pass over 20 GB (each byte once). Train CE → 0.1 (Engram memorization), held-out BPB 7.6–8.0 ≈ uniform (8.000). We hold ~130× the Chinchilla-optimal unique data. Question: is single-pass online the wrong *recipe*?

**Method:** primary sources only — arXiv abs+HTML full texts opened via arXiv API/browse, official GitHub READMEs fetched raw. Every number below was read off the cited source this session; three items verified in the sibling pass are cited `[S#]` → `docs/archive/research/2026-09-26-small-lm-dynamics.md` (which lists the opened URL per entry). Websearch was down (HTTP 403); discovery ran through the arXiv API instead.

---

## 1. Byte-level architectures 2025–2026: what's the recipe, and are there <50M curves?

**BLT — Byte Latent Transformer** (arXiv:2412.09871, Meta; code github.com/facebookresearch/blt). First FLOP-controlled byte scaling study, 400M→8B params, up to 4T training bytes (abs; repo README says 8T bytes): matches Llama 3 training-flop-controlled at scale with up to 50% fewer inference flops. What they do differently from plain next-byte CE on a flat stream:

- **Dynamic compute allocation**: bytes are grouped into patches by next-byte entropy from a small byte LM (100M params, 14 layers, 512 hidden, 512-byte window). "When the receptive field of the model is small enough, the trained entropy model can be encoded in an efficient lookup table."
- **Hash n-gram embeddings as input features**: byte-grams n=3…8 → rolling polynomial hash → one embedding table per n (500K slots total config), added to the byte embedding, normalized (§3.2.1). **Ablation (Table 8, 1B model / 100B bytes): "hash n-gram embeddings are very effective with very large improvements in BPB"; the per-n-gram vocab size is the most significant parameter; smaller n-gram sizes are more impactful than larger ones; gains largest on Wikipedia/Github (0.04 BPB), diminishing returns beyond ~300K hashes.** Paired with hash n-grams, BLT "works well with an encoder that is extremely light-weight i.e. just one layer."
- **Training recipe (§4.7–4.8)**: lr **4e-4 for all model sizes** (search over 1e-3…1e-4 at 400M and 1B found the same optimum for byte and token models), AdamW β=(0.9, 0.95), ε=1e-8, wd 0.1, linear warmup 2000 steps, **cosine decay to 0**, grad clip 1.0, batch ~16M bytes, context 8K–16K bytes. BPB defined as CE/(ln 2 · bytes).
- **No <50M models anywhere in the paper.** A 2026 multilingual study (2606.15044) trained BLT at 1.5B on low-resource data and found it "underperforms on downstream tasks, possibly because its architectural assumptions misalign with the constraints of limited low-resource training data."

**MegaByte** (arXiv:2305.07185): fixed-stride patches, big global patch-level transformer + small local byte-level model. Compute- AND data-controlled (80B bytes each): MegaByte (758M+262M) beats a plain byte Transformer (320M) on all five long-context text sets — PG-19 1.000 vs 1.057 BPB, Books 1.007 vs 1.097, arXiv 0.678 vs 0.816, Code 0.411 vs 0.575 (Table 2). Ablation: dropping the global model (local-only) costs arXiv 0.687→1.373 BPB; dropping local costs →1.263. Still loses to subword SOTA at 400B bytes on PG-19 (ppl 36.4 vs BlockRecurrent 26.5). **Min model size ~250–350M.**

**bGPT** (arXiv:2402.19155): hierarchical (patch 16, 12 patch-level + 3 byte-level layers, hidden 768, **110M params**). Plain next-byte CE. **Training recipe: 32 epochs on every dataset, lr 1e-4, batch 16, no hyperparameter tuning, no augmentation** (Table 2); Wikipedia = 13.67 GB at 107 V100-h/epoch → 32 passes. A 110M byte model pretrained on Wikipedia scores **1.0639 BPB on AG News** vs 0.9237 for GPT-2-small under identical settings (Table 3). Data-scale ladder bGPT³→⁵ (10³→10⁵ samples): ABC→MIDI BPB 0.2381→0.0011. Text-side conclusion: byte models work but are data/compute-hungrier, consistent with ByT5's "4× less text per token budget" `[S2]`.

**2025–2026 byte-level wave** (all opened via arXiv API):
- **Fast BLT** (2605.08044): adds an auxiliary block-wise diffusion objective + self-speculation to fix byte-by-byte generation speed — objective augmentation, not a recipe change for pretraining.
- **Toppling the Hierarchy** (2609.00463): hierarchical byte models (down/up-sample) are *worse* at character manipulation than pure flat byte models; "byte-level attention is the primary mechanism." Trade-off: hierarchy buys efficiency, costs fine-grained skill.
- **Scratchpad Patching** (2605.09630): names *patch lag* — within-patch predictions rely on stale patch reps; fixes with entropy-triggered scratchpads. Even at 16 bytes/patch, matches byte baseline with 3–4× less inference compute.
- **ATDC** (2605.30080): curriculum on the chunk compression ratio (low→high) stabilizes hierarchical training; competitive BPB on FineWeb-Edu 100B.
- **UTF-8 validity** (2606.14122, ICML 2026): at 355M params, byte models' UTF-8-validity converges ~2× slower than perplexity (4.2B vs 2.1B tokens).
- **Rust training report** (2609.25008): an independent end-to-end Rust LM (0.4B, burn/candle) documents burn defects incl. "a backward pass at roughly 3% of theoretical GPU throughput" and a segfaulting fusion path, and installs a **gradient-flow arbiter test** (one fwd/bwd, assert every trainable param gets finite nonzero gradient) — external corroboration of both our fused-path pain and our P10-R5 direction.

**Small (<50M) curves with data budgets — the big labs publish none; the 2026 indie tier does:**
- **Kathleen Writes** (2608.04678): byte-level LMs on WikiText-103 raw UTF-8 at **~0.5M params**, budgets 2–512 MB: attention-free 1.84 vs transformer 2.04 BPB at 512 MB; "the transformer needs more than 512 MB to match what the attention-free model learns from 32 MB."
- **Baseline Shape Decides the Verdict** (2609.29397): 98 byte-level runs at **60K params**, budgets 16M/130M bytes; param-matched transformers "span 22.6% in validation loss purely by depth/width choice"; at 130M bytes a plain gated diagonal-SSM block beats a routed model by a further 9.1%; code+logs released (github.com/veldanda/ByteLM).
- **Nested Byte-Level Vocabularies** (2608.28151): 30 models with **3.1M and 10.6M-param bodies on 200M tokens each**, BPB-scored, pre-registered.
- **Kronecker Embeddings** (2605.29459): nanoGPT-124M on 2.5B FineWeb-Edu — byte-structured embedding factorization reaches BPE-tied loss with ~1.43× fewer steps.

**Answer to Q1:** the 2025–2026 SOTA recipe for byte LMs is still plain next-byte CE on a static (multi-epoch) stream — nobody changed the objective. The deltas are (a) dynamic compute allocation (entropy patching), (b) **hashed n-gram features on the input side**, (c) hierarchy (now contested at small scale). No lab publishes <50M curves; the anchors that exist are 60K–10.6M indie runs (above) and 110M bGPT.

## 2. Small-model recipes (7–50M): single-pass online vs multi-epoch

- **Compute-optimal anchor at our size**: Muennighoff et al., *Scaling Data-Constrained LMs* (2305.16264v5, opened): "For 100M tokens, the corresponding one-epoch compute-optimal model according to [Chinchilla] has U_N of approximately **7M parameters**." Chinchilla's ≈20 tok/param gives ~150 MB for 7.5M `[S1]`. Our 20 GB ≈ 130–200× that budget ⇒ **we are compute-bound, not data-bound; single-pass over 20 GB is not a recipe anyone runs — it means the LR never decays and the run never converges.**
- **Repeats**: same paper, 400+ runs: "training with up to 4 epochs of repeated data yields negligible changes to loss"; fitted half-life R*_D ≈ 15 → "meaningful gains from repeating data can be made up to around 16 epochs"; when data-constrained, allocate new compute "to more epochs rather than more parameters" (R*_N < R*_D); "data is shuffled after each epoch"; always report held-out loss. Their recipe class: fixed FLOPs, cosine decaying 10×.
- **Multi-epoch is the norm at small scale, not the exception**: bGPT = 32 epochs at 110M (§1); TinyStories (2305.07759v2, opened): 1M–35M models, GPT-Neo arch, ctx 512, window 256, top-10K vocab, "all trained on a single V100 within at most 30 hours" on a narrow synthetic corpus — sub-10M fluency comes *from* the narrow data + epochs; nanoGPT shakespeare-char ≈ 74 epochs of 1 MB `[S4]`.
- **Fixed-horizon schedules everywhere**: Chinchilla matches the LR schedule horizon to the token budget `[S1]`; BLT warmup-2000 + cosine→0 at every scale; modded-nanogpt ends with a tuned cooldown ("lr decay to 0.1 instead of 0.0", record #19; "increase cooldown frac to .45", #26; "increase minimum lr", #72); Self-Play (2609.30063) uses "a short fixed warmup followed by a constant learning rate" so any checkpoint is a valid stopping point, and in the A.2 transfer protocol decays to zero over 200 steps once val BPB plateaus (<0.005 improvement × 5 evals).
- **Seed discipline**: ufakzeka-1 (2609.25081, 151M): "training-seed variance was as large as the spread across every recipe we tried, so single-seed comparisons at this scale are uninformative."

**Answer to Q2:** at 7.5M params the reference recipes are: pick a token horizon by compute budget (150 MB Chinchilla floor up to a few GB), use a subset (1–16× the horizon in unique data; ≤4 epochs free, ~16 max), shuffle per epoch, warmup + constant + real cooldown. Single-pass online over a 20 GB surplus appears nowhere; its constant-LR tail is the opposite of every published schedule.

## 3. Fastest-known pretrain tricks (modded-nanogpt) and what transfers to a looped byte model

Source: github.com/KellerJordan/modded-nanogpt README (fetched raw), record history llm.c baseline 45 min → current 1.126 min on 8×H100 to 3.28 FineWeb val loss (<400M tokens vs 10B). Muon itself: "~1.5x better sample-efficiency, <2% wallclock overhead". Measured per-trick wins (record #, time):

| Trick | Record delta | Transfer to dormouse loop |
|---|---|---|
| Muon on hidden 2D | #3: 31.4→24.9 min | already default (`--opt mix`) `[S11/S16]` for lr anchors |
| ReLU² + zero-init projections + QK-norm + pad embeddings | #5: 22.3→15.2 (−32%) | trivial in burn; QK-norm also the entropy-collapse cure `[S15]` |
| Untie embed/head | #8: 13.1→10.8; re-untie at 2/3 #53 | win shrinks at vocab 256 (head is 256×d) — cheap, low priority |
| Value+embedding skips, momentum warmup, softcap | #9: 10.8→8.2 | value-embeddings mix an input embedding into V — analog: mix byte-embedding skip into each loop iteration |
| U-net skips between layers (+2× lr) | #11: 7.8→7.2; refined #45 | in a weight-shared loop: hₖ→h_{k+1} / x₀ skips across iterations |
| FlexAttention 1K→64K ctx | #12: 7.2→5.03 | CUDA/PyTorch-specific; N/A (kda window is ours) |
| Value embeddings | #14: 4.66→4.41; U-net VE #15; untied #63 | same as above — cheapest arch add with repeated measured wins |
| lr floor 0.1 + FP8 head | #19: 3.57→3.142 | **take the lr floor, skip FP8** (sm_120 fp8 degrades at overfit, AGENTS.md) |
| smaller batch / batch schedule | #21, #46 | at fixed horizon on one launch-bound GPU: raise bytes/step instead |
| EoS-aligned batches, max doc len | #26: 2.896→2.863 | we already route docs by shard; align windows to doc bounds |
| NorMuon | #41: 2.358 (arXiv:2510.05491) | possible later; vendored Muon+ first |
| **Bigram hash embedding (1/4 of model dim, sign trick)** | #62: 1.748→1.655 (−5.3%); sign trick #83 | **directly validates Engram-as-feature; BLT ablation says do it on the input side** |
| MTP + untie-at-2/3 | #53: 2.037→1.988 | DSpark next-K aux already in this family |
| Prefix-token prediction aux loss | #88: 1.256→1.243 | same family as our aux heads |
| Smear (1-token lookback) | #34: 2.565→2.547 | trivial; helps short-context byte prediction |
| Polar Express NS replacement | #38 | skip — AGENTS.md keeps quintic NS `[S11]` |

**Ranked by measured-impact-per-line-of-code for a 7.5M looped byte model:** (1) schedule fix (§2 — bigger than any single trick here), (2) Engram → input-side feature / small fixed λ, (3) ReLU²+zero-init+QK-norm, (4) value/embedding skips into loop iterations, (5) bigram-hash feature (already own the tables), (6) lr floor + cooldown fraction, (7) smear. Skip: FP8 paths, FlexAttention, hierarchical patching (2609.00463 says flat byte is fine at small scale; patch-lag hurts), Polar Express.

## 4. Self-Play Pretraining with Zero Data (2609.30063) at small scale

Opened in full (abs + HTML). Byte-level throughout (vocab 256, "the universal machine produces raw bytes"). Two same-arch Llama decoders from random init: a **generator** emits Brainf*ck-UTM programs (GRPO, KL-regularized toward the uniform program prior = Solomonoff 2^−|p|), a **learner** does plain next-byte CE on program outputs. Generator reward = |⟨P⊙∇θL(y;θ_now), θ_past−θ_now⟩| — AdamW-preconditioned alignment of the program's gradient with the learner's recent parameter movement over a lookback of ⌈e/2⌉ rounds; computed without materializing full gradients via a forward-mode (JVP) kernel.

- **Compute for measurable signal**: reward ablations (Table 5) and the emergence table run at the **1M-parameter** scale — the generator finds Fibonacci at round 512, geometric/quadratic/cubic at 512, vs ">53,000" expected under the uniform prior (1.64×10⁸ uniform samples contained "no instances of any family except arithmetic"). Cross-modal transfer scaling laws fit L(C)=E+A·C^(−α) with exponents "comparable to those obtained by training directly on natural data" (Table 2); max training budget 34.36B tokens, ctx 4096, K-seed ensembles.
- **The transferable part (A.1/A.2, opened)**: a **24.4M** learner warm-started from self-play converges on natural byte corpora with substantially fewer tokens — ESC-50 320M vs 496M, CIFAR-10 421M vs 588M (corpora 255M/146M/152M tokens, repeated over epochs to convergence). Protocol details worth stealing: hyperparameter local-optimality by coordinate descent over **lr ∈ {1e-3, 3e-3, 1e-2}, wd ∈ {0, 0.1, 0.3, 0.8}, batch, β**, 4 seeds, model selection on held-out BPB; **constant LR until val BPB plateaus (<0.005 × 5 evals), then 200-step cosine to 0**; natural data used for selection only, never gradients.
- **Open implementation**: none. The paper releases only the JVP flash-attention kernel it uses (github.com/amorehead/jvp_flash_attention). Their simpler baseline — pretraining on randomly sampled PCFGs (Appendix H) — "is effective on language-like domains but lacks broad cross-domain transfer."
- **Verdict for dormouse**: full self-play is a research project, not a recipe fix (matches our PLAN: parked, mapped). What transfers today: the plateau-triggered-decay schedule practice, the small tuning-grid protocol, and the evidence that curriculum shaping pays off even at 1M–25M params — i.e., *after* the core trains, a cheap dormouse-side analog is sequencing its synthetic/aux streams by difficulty, not standing up an RL generator.

## 5. Eval sanity: published byte-level BPB anchors

Uniform over 256 = 8.000 BPB (ln 256 / ln 2) `[S]`. Shannon printed English n-gram entropies: F1=4.14, F2=3.56, F3=3.3, F4≈2.62 bits/letter `[S6]` — any working core must beat ~4.1–4.7, and our held-out 7.9–8.0 is *above a unigram* `[S]`.

Neural byte-model anchors (all opened this session):

| Model | Params | Data | BPB | Source |
|---|---|---|---|---|
| Byte Transformer (flat) | 320M | 80B bytes, PG-19 | 1.057 (Books 1.097, arXiv 0.816, Code 0.575) | MegaByte Table 2 |
| MegaByte (global+local) | 758M+262M | 80B bytes, PG-19 | 1.000 (Books 1.007, arXiv 0.678, Code 0.411) | MegaByte Table 2 |
| MegaByte w/o global (local-only) | — | 80B bytes, arXiv | 1.373 | MegaByte Table 7 |
| bGPT (hierarchical) | 110M | 13.67 GB Wikipedia ×32 epochs → AG News | 1.0639 (GPT2-small BPE: 0.9237) | bGPT Table 3 |
| Byte transformer (attention baseline) | ~0.5M | WikiText-103, 512 MB budget | 2.04 (attention-free twin: 1.84) | Kathleen Writes abstract |
| Shape-study transformers | 60K | 16M/130M bytes | val-loss spread 22.6% by shape alone | arXiv 2609.29397 |

**No published unigram/bigram/5-gram BPB table exists for a mixed text+math byte corpus** — checked the 2025–2026 byte-level sweep (§1) and Shannon is letters-of-English only. The definitive anchor for *our* mixture is therefore self-computed: a byte-count unigram/bigram + Kneser-Ney-ish 5-gram model run over corpus-v3's eval tail is a one-file Rust/Python pass (counts over 20 GB, no training). Until that exists, MegaByte's flat-transformer rows (1.06 BPB on books at 320M/80B) and Kathleen's 0.5M/512MB→2.04 are the realistic scale-aware bars; note all published anchors are single-domain — a text+math mix will sit above the books number at our size.

## 6. FINAL — Recipe for dormouse

**Data (the actual fix):**
1. **Stop training on the stream.** Choose a token horizon T from compute: at ~1.4 s/step and ≥5,120 bytes/step, a realistic T is 1–4 GB (≈8–30K steps). Chinchilla floor for 7.5M is 150 MB `[S1]`; the C4 fit says 100M tokens ↔ 7M params [Muennighoff].
2. **Unique subset D_u = T/4 … T** (≈1–4 epochs; ≤4 epochs are loss-free [Muennighoff], ~16 is the practical ceiling). Keep domain-shuffling per epoch (shard.rs already does; Muennighoff: "data is shuffled after each epoch"). Eval tail never in train (ADR-0010).
3. **Schedule = warmup (~1–2% of steps) → constant → cooldown over the last ~30–45% of steps, decaying to ~0.1× peak (or 0)**. Every opened source converges here: BLT cosine→0; speedrun "lr decay to 0.1", "cooldown frac .45", "increase minimum lr"; self-play decay-on-plateau. A constant-LR run with no horizon is the one configuration no reference uses — it is what produced held-out ≈ uniform.
4. **Peak lr ≥ ~1e-3-class in the RMS-matched convention for 7.5M** (Moonlight scaling trend `[S11]`; verify Muon+'s ColRow convention by logging per-group update RMS against the 0.2–0.4 AdamW band `[S11]`). BLT's 4e-4 is the AdamW-convention anchor at 400M–8B — smaller models take more, not less.
5. **Batch**: raise bytes/step (we are launch-bound at 1.4 s/step; a fixed horizon in fewer, fatter steps is wall-clock-cheaper on one GPU). 2–3 seeds before believing any A/B at this scale [ufakzeka-1].

**Architecture deltas, ranked by measured-impact-per-line:**
1. **Engram → input-side feature or fixed small λ, never a free output-blend arm.** BLT ablation: hash n-grams as *inputs* give "very large improvements in BPB"; kNN-LM keeps memory at λ=0.25, tuned on held-out `[S9]`; speedrun's bigram embedding (−5.3% wall-clock) is input-side. Train CE will stop being meaningful — judge by eval BPB only `[S]`.
2. **Shrink the tables before growing them: n=3–4 first, ~300–500K rows.** BLT: smaller n most impactful; "after 300K hashes, there are diminishing returns." Our 24M rows are ~50× past that point; the freed RAM is better spent on batch or horizon.
3. **ReLU² + zero-init projections + QK-norm** (−32% in the speedrun's arch batch; QK-norm doubles as the entropy-collapse cure `[S15]`).
4. **Embedding/value skips into each loop iteration** (U-net/value-embedding records #9/#11/#14/#15): the looped-block analog of their between-layer skips; also add a 1-token "smear" on inputs (#34).
5. **Untie embed/head + logit softcap 15** (#8/#18): nearly free at vocab 256, small expected win; softcap also bounds logits (z-loss-adjacent `[S10]`).
6. **Keep DSpark/MTP-family aux** (#53/#88 confirm the family); keep `use_msa=false` baseline (ADR-0012).

**Explicitly skipped:** FP8 anything (overfit-degrading on sm_120), FlexAttention/FA3 (stack-specific), hierarchical patching (efficiency machinery for ≥400M; flat byte wins fine-grained at small scale per 2609.00463), Polar Express, full self-play RL (parked; no open impl; only the schedule + tuning protocol are stolen today). Add hierarchical/patching work only when the core beats ~1.5 BPB on a single-domain eval carve.

**One runnable check:** the byte n-gram baseline script over the eval tail (§5) — it is the anchor every future run gets judged against, and it would have caught "held-out at uniform" on day one.

## References (all opened this session unless marked [S#] → sibling file lists the opened URL)

1. BLT: *Byte Latent Transformer: Patches Scale Better Than Tokens* — arxiv.org/abs/2412.09871, arxiv.org/html/2412.09871v1 (§3.2.1 hash n-grams, §4.7–4.8 recipe, §7 Table 8 ablation); repo raw README: raw.githubusercontent.com/facebookresearch/blt/main/README.md
2. MegaByte — arxiv.org/abs/2305.07185, arxiv.org/html/2305.07185v2 (Tables 1–3, 7)
3. bGPT — arxiv.org/abs/2402.19155, arxiv.org/html/2402.19155v1 (Tables 1–3, 5–6)
4. Fast Byte Latent Transformer — arxiv.org/abs/2605.08044
5. Toppling the Hierarchy in Byte-level Language Modeling — arxiv.org/abs/2609.00463
6. Scratchpad Patching — arxiv.org/abs/2605.09630
7. ATDC — arxiv.org/abs/2605.30080
8. Equity with Efficiency (multilingual tokenizers, BLT at 1.5B) — arxiv.org/abs/2606.15044
9. Beyond Perplexity: UTF-8 Validity — arxiv.org/abs/2606.14122
10. Training a Language Model End-to-End in Rust — arxiv.org/abs/2609.25008
11. Kathleen Writes (attention-free byte LM, data-budget curves) — arxiv.org/abs/2608.04678
12. Baseline Shape Decides the Verdict (60K-param byte runs) — arxiv.org/abs/2609.29397; github.com/veldanda/ByteLM
13. Nested Byte-Level Vocabularies — arxiv.org/abs/2608.28151
14. Kronecker Embeddings — arxiv.org/abs/2605.29459
15. ufakzeka-1 (seed variance at small scale) — arxiv.org/abs/2609.25081
16. Muennighoff et al., *Scaling Data-Constrained Language Models* — arxiv.org/abs/2305.16264, arxiv.org/html/2305.16264v5
17. TinyStories — arxiv.org/abs/2305.07759, arxiv.org/html/2305.07759v2
18. modded-nanogpt README + record history — raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/README.md; NorMuon arXiv:2510.05491
19. Self-Play Pretraining with Zero Data — arxiv.org/abs/2609.30063, arxiv.org/html/2609.30063v1 (§2.2 reward, §3.1 recipe, A.1–A.2, Table 5); kernel repo github.com/amorehead/jvp_flash_attention
20. [S1] Chinchilla; [S2] ByT5; [S4] nanoGPT shakespeare-char; [S6] Shannon 1951; [S9] kNN-LM; [S10] ST-MoE z-loss; [S11/S16] Moonlight + Muon lr anchors; [S15] QK-norm — opened and cited in `docs/archive/research/2026-09-26-small-lm-dynamics.md`
