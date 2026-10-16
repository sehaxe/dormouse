# Budget savers beyond MoD / token-skipping — what else buys VRAM, time or data at train time

**Date:** 2026-10-16. **Read-only:** this file only; no config or code touched, nothing committed (the
`docs-site/tools/manifest.mjs` line is owed, §9).
**Question (owner):** beyond MoD / token-skipping, which **train-time** techniques save VRAM, wall-clock
or data for the b1 byte-MoE (1.07B total / ~57M active, ONE 16 GB 5060 Ti, real KDA via FLA, real
19.7 GB corpus) **without giving quality away**?
**Builds on, does not repeat:** `docs/decisions/2026-10-15-1b-readiness-plan.md` §2/§3.5/§7 ·
`docs/research/2026-10-16-jepa-dspark-alternatives.md` (the *deferred* bucket) ·
`docs/research/2026-10-15-tsct-super-kernel.md` (the 5-item ladder, its L-off columns) ·
`port/README.md` invariants · `AGENTS.md` §2.2/§3.1 GEMM ceilings.
**Method:** 16 arXiv API queries + 13 full `abs` pages read by hand (2026-first, by submission date);
every ID resolved individually on 2026-10-16. Marks: **[A]** = the authors' abstract, quoted ·
**[ours]** = our measurement, file:line · **[S1/S2/S3]** = SPECULATION (S1 = sourced mechanism,
scale-mismatched transfer; S2 = topology/composition argument; S3 = no source at all).

## TL;DR
1. **[A] The MoE block, not attention, owns the peak.** RelayMoE `2610.07333`: *"Peak memory is
   dominated by the MoE block, not attention: every intermediate buffer in the MoE dispatch pipeline is
   individually scaled by top-k routing."* That is our b1 exactly, and it names the fix (per-hop
   reconstruct-and-release of expert intermediates; our [E,cap,f] is 716 MB bf16 [ours, tsct §1]).
2. **[A] Three 2026 systems cut that peak >50% with the loss untouched:** MoEBlaze `2601.05296`
   (*">4x speedups and over 50% memory savings"*), `2609.14306` (*"dispatch peak by up to 59.3% without
   losing throughput… loss and gradients stay exact"*, vocab-proj −86.6%) and RelayMoE.
3. **[A] The host-RAM lever we already trust for Engram generalizes to activations:** `2609.14306`'s
   **SCO** keeps *"the one long-lived tensor of each checkpoint boundary in CPU memory"* [ours: our 64 GB host idles during a step, plan §2.3].
4. **[A] Optimizer state is the other half, and the Adafactor question now has a number.** SkewAdam
   `2607.19058` takes the state to **2.6% of AdamW's** (peak **81.4 → 31.3 GB**) and measured that
   *"removing momentum costs 31 perplexity points (tuned Adafactor, 139.7)"*; AL-space 8-bit
   `2608.22322` **beats bnb8 at 1.1B** (72.90 vs 73.54 PPL, same storage).
5. **[A] Highest ceiling, highest risk: reversibility.** `2606.07404` reconstructs activations in the
   backward pass *"holding activation memory flat as the model grows"* — and our LoopBlock **is** a
   weight-shared recurrence stack; but its own report warns *"several failures are silent"*.
6. **Two rejects, stated so they are not re-searched:** sequence/pipeline parallel on ONE card (every hit
   is multi-device — `2604.27085` RoundPipe, `2502.07563` LASP-2; the collective *is* the mechanism)
   **[S2]**; and small-router-borrows-a-large-frozen-body, for which **no 2026 paper was found** **[S3]**.

## 1. What is already DONE — the rungs this pass must not re-recommend
`bnb8` state swap (`optim.py:252-275`) · fp8/NVFP4 **storage** rungs (`tsct.py:131`, `config.py:429-448`) ·
always-on shared expert (`experts.py:189`, TSCT2 §14) · the Triton grouped-GLU project (tsct-super-kernel
§1 #3) · bank cache keyed by `_version` (`experts.py:258-269`) · 0-sync counters / `_permute`
(`experts.py:220-240`) · LFB bias (`experts.py:194-210`) · shape-batched Muon (`optim.py:282-288`) ·
cap-padded bmm dispatch · `clip_grads` via `_foreach_norm` (`optim.py:305-319`). **ByteFlow** is pending
its criterion — untouched here.

## 2. The table — technique · id+date · claimed saving · scale tested · fit-to-ours · effort
| # | technique | id + fetched date | claimed saving (source's own) | scale tested | fit to ours | effort (LOC, gates) |
|---|---|---|---|---|---|---|
| N1 | **MoE dispatch de-materialization + smart activation checkpoint** (MoEBlaze) | `2601.05296`, 2026-01-08 | **>50% memory** vs existing MoE frameworks, >4x speedup [A] | framework-level (no model size given) | DIRECT: our [E,cap,f] workspace + per-topk dispatch buffers are the named cost [ours, tsct §1] | high (a system); the *checkpoint policy* part is ~50-100 LOC |
| N2 | **Recompute-in-grouped-experts** (RelayMoE, ring) | `2610.07333`, 2026-10-05 | peak "dominated by the MoE block"; reconstruct-and-release per hop; longer seq / bigger batch [A] | 30B-57B production MoE | the **recompute** idea transfers 1:1; the **ring** does not (needs ranks) [S2] | ~100-200 (a checkpointed expert pass) |
| N3 | **SCO: checkpoint-boundary tensor in host RAM + pipelined CPU AdamW** | `2609.14306`, 2026-09-13 | dispatch peak **−59.3%**, vocab-proj peak **−86.6%**, offload step **2.05×** faster, **loss exact** [A] | 120B-667B, 1M ctx, FSDP2 | our 64 GB host idles; SCO is the Engram-offload playbook applied to activations | ~80-150 (hooks/wrap) |
| N4 | **Reversible recurrence stack** (LightningLM 0.1V) | `2606.07404`, 2026-06-05 | *"activation memory flat as the model grows"*; TQP carries state on 2.26B adapters not 100B+ experts [A] | 5B → 9B → 120B MoE, 8 GPU, 8K ctx | our LoopBlock **is** the recurrence stack; highest ceiling, new mechanism | high (~300-600 + numerics gate) |
| N5 | **MoE-tiered optimizer state** (SkewAdam) | `2607.19058`, 2026-07-21 | state **1.29 GB = 2.6%** of AdamW's; peak **81.4 → 31.3 GB**; PPL 108.4 vs AdamW 126.8 [A] | 6.78B MoE, 82M tokens | our expert bank is 626M of 1.07B params; bnb8 only covers the AdamW group | ~120-200 + determinism gate |
| N6 | **AL-space 8-bit optimizer state** (exact-zero reserved) | `2608.22322`, 2026-08-23 | **AL8 72.90 PPL vs bnb8 73.54** on TinyLlama-1.1B, same storage, higher throughput [A] | **1.1B — scale-matched** | replaces/upgrades `enable_bnb8bit()`; plain quantizer, so it composes with `torch.compile` (bnb8 does not [ours, plan:356]) | ~150-250 + determinism gate |
| N7 | **Subspace optimizer + residual correction → Muon (PMRC)** | `2610.19844`, 2026-10-14 | states held in a subspace, matrix sign on the projected momentum; O(T^-1/4) [A] | LLM pretraining (size in body) | Muon-specific diet; our Muon already batches by shape | ~150-250 |
| N8 | **Monarch-Muon** | `2610.17429`, 2026-10-12 | 6.89B: peak **68.9 → 52.9 GB**, Muon state **14.60 → 6.77 GB**, opt step **1288 → 304 ms**; 1B MoE 2.68 vs 2.77 [A] | 257M dense, **1B MoE**, 6.89B | needs Monarch-factorized weights — a structural change to our low-rank experts | high (structure) |
| N9 | **FlashOptim** (master-weight splitting + companding) | `2602.23349`, 2026-02-26 | per-param **16 → 7 B** (5 with gradient release); Ckpt halves; *"no measurable quality degradation"* incl. Llama-3.1-8B [A] | 8B finetune | overlaps bnb8; the **master-splitting** piece is new to us | ~200-300 |
| N10 | **Depth-adaptive decode for looped LMs** (CDB) | `2608.09444`, 2026-08-10 | up to **99% of the estimated max speedup**; *"fully looped architectures are best suited"* [A] | Ouro 1.4B, Huginn 3.5B | decode lane only — we have no serving path; LoopBlock is fully looped [ours] | deferred (§6) |
| N11 | **KDA/GDN recurrent-state PTQ** (DAMP) | `2608.27513`, 2026-08-27 | INT8/FP8 **degrade** reasoning; INT4/NVFP4 **collapse**; outliers live in specific key channels [A] | GDN + KDA (serve) | a *negative* for any "shrink the KDA state" temptation; our state is fp32 train-side [ours] | none — read as evidence |
| N12 | **SpecPrefetch** (adapter predicts next-layer experts) | `2607.24787`, 2026-06-24 | removes the routing→load→execute serialization; prediction errors affect **transfer only**, not outputs [A] | offloaded MoE inference | only if the expert bank ever leaves VRAM | deferred (§6) |
| — | *(given, already in-repo, re-verified: MuonQ `2605.11396` 2026-05-12; WSqD `2607.10959` 2026-07-12; Muon-state quant `2509.23106` 2025-09-27; Full-Stack FP4 `2607.04422` 2026-07-05)* | — | MuonQ: pre-quant normalization + structural decomposition + μ-law [A]; WSqD: one peak LR across horizons [A] | — | already on the one-button / tech-verification lists | — |

## 3. The quality cost at ≤1B — one line per technique (the SRIN read)
Each line answers **only** "is the quality cost known *at our scale*?", not "is it small in general".
- **N1 / N2 / N3 — zero by construction, at the wrong scale. [S1]** `2609.14306` says the loss and
  gradients are *exact* because only the *order* of compute/data-movement changes [A]; RelayMoE [A] and
  MoEBlaze [A] likewise sell memory, not accuracy. But **no study runs recompute/offload at ≤1B on one
  card**, so the transfer is S1, not verified.
- **N4 — unknown at ≤1B. [S1]** Reversibility's cost is *numerical* (reconstruction drift), not
  statistical; `2606.07404` gives no ≤1B loss table and warns some growth failures are silent [A].
- **N5 — known ≈zero, and proven *not* to be the tiers. [A]** *"A tier ablation reaches the same value
  while carrying twenty times the state, so the tiers buy memory rather than accuracy."* (6.78B)
- **N6 — known to be *positive*, at 1.1B. [A]** AL8 beats bnb8 (72.90 vs 73.54) — the only scale-matched
  quality datapoint in this table.
- **N7 / N8 — known positive/neutral at ≥1B. [A]** Monarch-Muon 2.68 vs 2.77 on a 1B MoE — but Monarch is
  a different weight parameterization, so it does not transfer to our low-rank experts unmeasured **[S2]**.
- **N9 — known-neutral to 8B. [A]** *"no measurable quality degradation… including Llama-3.1-8B finetuning."*
- **N10 — unknown, decode-only. [S3]** No ≤1B looped byte-LM decode measurement exists; no serving path.
- **N11 — a known NEGATIVE**: do not shrink the KDA/GDN state with uniform INT8/FP8 (reasoning degrades)
  or INT4/NVFP4 (collapse) [A] — relevant the day our KDA state leaves fp32 [ours, AGENTS §3.5.5].
- **Adafactor-26 — a known NEGATIVE at 6.78B:** tuned Adafactor costs **31 perplexity points** [A]. Closed.

## 4. The specific quantity: what cuddles ACTIVATIONS at our topk-2 shapes
Our peak is activation + workspace, not weights: the plan's bf16 budget puts weights+masters+moments at
~3 GB and **activations+KDA scratch at ~4-6 GB** of ~9-10 GB [ours, plan §2.3]; the port OOMs eager at
b8·s512 with fp32 masters [ours, plan §7]. The named tensors: the `[E,cap,f]` intermediate (716 MB bf16
at cap 64 [ours, tsct §1]) and the per-topk dispatch buffer (`experts.py:_permute/_padded_dispatch`).
Torch-native options, named exactly:
1. **`torch.autograd.graph.saved_tensors_hooks`** — pack saved tensors to pinned host RAM, unpack on the
   backward edge. This *is* SCO's mechanism (`2609.14306`) minus the ring; ~0 custom kernels, and our
   64 GB host idles during a step **[S1]**.
2. **`torch.utils.checkpoint.checkpoint(fn, use_reentrant=False)`** around the routed-expert pass only —
   recompute `h = silu(gate)·up` and the dispatch buffer instead of storing them. RelayMoE's
   reconstruct-and-release [A] in stock torch; non-reentrant is required because the forward reads
   detached router outputs.
3. **`torch.utils.checkpoint.checkpoint_sequential`** over the LoopBlock's two iterations — recompute
   iteration 1 rather than storing it (~116 MB/pass at d2048×16K tokens bf16, **S2** derived from §2.3).
4. **In-place GLU** (`F.silu(g, inplace=True)` on the padding-free arm) — but measure: the tsct study
   warns the GLU is *not* the bandwidth-bound part [ours, §1].
**Not named: a hand-written "fused gather-backward".** The spike measured `torch._grouped_mm` **2×
slower** than the cap-padded bmm at E=1024 [ours, plan:352-356]; a bespoke per-expert gather-backward
re-opens that risk for a saving the hooks already deliver.

## 5. THE RECOMMENDED SHORTLIST — pre-registered here, before any run
Discipline (`AGENTS.md` §1.2, `AB-PROTOCOL.md`): **3 seeds**, one matched `batch × seq_len` across arms
(the eval window must print identical byte counts), 200-step smoke first then 2 000-step confirm; win =
arm mean BPB below the control mean by **more than the control's own 3-seed spread**; a tie **deletes**
the arm. All three ride the **port** (the b1 lane, plan §7); none touches the burn fallback.

### SL1 — Recompute + host-offload the MoE activation peak  *(N2 + N3)*
- **Exact A/B:** arm = `saved_tensors_hooks` host-offload **+** `checkpoint(use_reentrant=False)` around
  the routed-expert pass; control = today's `_dispatch`; `moe_capacity = 2.0` (cap 64) in both.
  **N = 200 smoke → 2 000 confirm, 3 seeds.** **Metric:** peak `torch.cuda.max_memory_allocated()` at
  b2·s512 b1 shapes, step ms, held-out **BPB** at matched steps with the window printed. Gate: the
  offload counter must be **non-zero** (ADR-0019) — a zero means the arm did not run.
- **LOC cap 150 · port** (`train/loop.py` wrap + one counter; no new kernels).
- **PREDICTED (falsifiable):** peak VRAM **falls ≥25%**, the legal `batch × seq` grows ≥1 rung, **BPB
  lands INSIDE the control's 3-seed spread** (`2609.14306`: loss exact [A], but no ≤1B single-card study
  → **S1**), step time **+5-15%**. BPB outside the spread ⇒ delete the arm; that would falsify the S1
  transfer and is the *point* of running it.

### SL2 — Tiered/skewed optimizer state for the expert bank  *(N5)*
- **Exact A/B:** arm = a factored second moment **only** for the 1024-expert group, fp32 momentum +
  factored second moment on the backbone; control = today's `bnb8` AdamW group. **N = 200 → 2 000, 3
  seeds.** **Metric:** BPB (primary), optimizer-state bytes printed, and a **determinism** check (two
  same-seed runs bit-identical on the state path — AGENTS §3.7). Gate: an `optim_state=` byte counter;
  resume round-trip.
- **LOC cap 200 · port.**
- **PREDICTED:** at **9.2M** the arm lands **INSIDE the spread** — the router/embedding groups dominate
  at that size, so the *smoke cannot confirm it*; the win sought is **state bytes**, not BPB, and
  `2607.19058`'s own ablation already showed the tiers cost no accuracy [A]. The 6.78B→1.07B transfer is
  **S1**; that paper's *router-stability* result (balance within 1% of uniform [A]) is the half to watch
  against our LFB.

### SL3 — AL-space 8-bit state replacing bnb8  *(N6)*
- **Exact A/B:** arm = AL8 quantizer (exact-zero reserved, adaptive log-range per block) on the AdamW
  group; control = `enable_bnb8bit()`. **N = 200 → 2 000, 3 seeds.** **Metric:** BPB, state bytes/param,
  and — the real deliverable — **does it compile?** (`torch.compile` + bnb8 is a recorded
  incompatibility [ours, plan:356]; a plain tensor quantizer should compose).
- **LOC cap 250 + determinism gate · port.**
- **PREDICTED:** BPB **INSIDE** the spread (a tie ⇒ keep only if compile-compatibility is proven *and*
  the bytes win; conclude a reachable state), state bytes ≤ bnb8's. The only shortlist item with a
  **scale-matched positive** result (1.1B, AL8 vs bnb8 [A]) — so a tie here is informative, unlike SL1/SL2.

## 6. What I would NOT do (with the reason read)
- **Sequence / pipeline parallel on one card** — the mechanism is a collective across ranks (`2604.27085` RoundPipe: *multiple* GPUs; `2502.07563` LASP-2). **[S2]**
- **Shrink the KDA state** — `2608.27513` is a measured negative (INT8/FP8 degrade, INT4/NVFP4 collapse)
  [A]; ours is fp32 by design [ours, AGENTS §3.5.5].
- **Monarch-Muon as-is** (N8) — it reparameterizes the *weights*; our experts are TSCT low-rank factors
  **[S2]**.
- **A bespoke fused per-expert gather-backward** — `_grouped_mm` measured **2× slower** than the padded
  bmm [ours, plan:352-356].
- **Depth-adaptive decode now** (N10) — no serving path; LoopSpec `2609.17184` already owns the
  training-free route [given doc §2]. Post-b1.
- **SpecPrefetch / expert offload** (N12) — our bank is VRAM-resident at b1; nothing to prefetch.
- **The frozen-body transfer as asked** — **no 2026 paper found**; nearest = `2606.07404`'s
  state-preserving growth and Puro-2B's recipe. **I cannot answer this from the sources [S3].**

## 7. VERIFIED (sourced) vs SPECULATION (my inference)
**VERIFIED.** Every §2 ID resolved individually on 2026-10-16 and its abstract read — dates:
`2601.05296` 01-08 · `2610.07333` 10-05 · `2609.14306` 09-13 · `2606.07404` 06-05 · `2607.19058` 07-21
(v2 08-13) · `2608.22322` 08-23 (v3 09-01) · `2610.19844` 10-14 · `2610.17429` 10-12 · `2602.23349` 02-26
(v2 03-11) · `2608.09444` 08-10 (v2 09-25) · `2608.27513` 08-27 (v2 09-30) · `2607.24787` 06-24 ·
`2608.27370` 08-27 (v2 09-03) · `2604.27085` 04-29 · `2502.07563` 2025-02-11 · `2605.11396` 05-12 ·
`2509.23106` 2025-09-27. **Ours:** b1 = 1 066 354 113 params / ~57M active (~37.9M touched per pass,
`configs/b1.toml:103-131`); activation+workspace ~4-6 GB vs weights ~3 GB (plan §2.3); `[E,cap,f]` =
716 MB (tsct §1); eager OOM at b8·s512 (plan §7); **no checkpointing and no host optimizer offload
anywhere in `port/src`** (grep, 2026-10-16); `bnb8` == `enable_bnb8bit()` (`optim.py:252`); `_grouped_mm`
2× slower (plan:352-356); bnb8 ⊥ compile (plan:356).
**SPECULATION.** **S1** — recompute/host-offload transfers loss-neutrally to ≤1B on one card: exact by
construction, but the smallest tested point is 30B+ on a ring (`2609.14306`, `2610.07333`). **S2** —
SL2's tiering is invisible at 9.2M (a group-size argument, not a measurement), and SL1's iteration-level
option saves ~116 MB/pass (arithmetic only). **S3** — any small-router-borrows-large-body trick.
**I cannot answer on available information:** (i) any 2026 study isolating **recompute vs every-N
checkpointing at 1B on a single device**; (ii) **per-study VRAM cost** at ≤1B for the recompute systems
(they report % at 30B+); (iii) whether AL8 composes with `torch.compile` — [A] reports throughput only.

## 8. The NEW ids (beyond the brief's list), dates fetched 2026-10-16
`2601.05296` 2026-01-08 MoEBlaze · `2610.07333` 2026-10-05 RelayMoE · `2609.14306` 2026-09-13 Flattening
Every Memory Peak · `2606.07404` 2026-06-05 Reversible Foundations · `2607.19058` 2026-07-21 SkewAdam ·
`2608.22322` 2026-08-23 AL-space optimizer state · `2610.17429` 2026-10-12 Monarch-Muon · `2608.09444`
2026-08-10 CDB · `2608.27370` 2026-08-27 Puro-2B. *(brief-supplied, re-verified: `2605.11396`
2026-05-12 MuonQ · `2607.10959` 2026-07-12 WSqD.)*

## 9. Open questions
1. Does `saved_tensors_hooks` host-offload keep the b1 step **NaN-clean** with the firewall's mask/read
   ordering (the hook packs on save; the firewall reads on log cadence [ours, AGENTS §3.8])?
2. Is SL2's factored second moment **deterministic** enough for `tools/determinism.py`, given the TSCT
   factors' known QR residue [ours, AGENTS §3.7]? Owed: a `docs-site/tools/manifest.mjs` line for this
   document before `npm run check` is green (one-file mandate, no commit).
