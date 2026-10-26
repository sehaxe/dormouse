# Is our 1M-context recipe the best one? — a check against the 2026 record (2026-10-16)

**Ask (owner):** "уверен что у нас лучший рецепт 1 миллиона контекста?" Our recipe: (1) KDA
linear attention (gated delta rule, fixed 256 KB state, FLA kernels) as the ONLY always-on
attention; (2) an MSA block-sparse arm (top-K blocks of r=4, adaptive 1/4 share) for exact
recall — currently an all-blocks selector `O(t²/r)`, being fixed for memory (chunked gather);
(3) planned: hierarchical coarse→fine selector, segment-with-state-carry training. Data: a
19.7 GB filtered web corpus, mostly short documents.

**Method.** Local first: `docs/research/2026-10-16-new-tech-scan.md`,
`2026-10-15-1b-tech-verification.md`, `2026-10-15-tsct-super-kernel.md`,
`2026-09-27-kda-sota-ceiling.md`, `archive/research/2026-09-26-long-context-agentic.md`,
`docs/adr/0004-context-ladder.md`, `docs/adr/0014-msa-cut.md`, AGENTS.md §3.5, and the live arm
(`src/dormouse/model/sparse/{sparse,attention}.py`). Then primary sources: arXiv abstracts
and search pages read directly. **Note on tooling:** the arXiv MCP was HTTP-429/timeout for this
whole session; every ID below was instead resolved on `arxiv.org` (abs page or search-page
abstract) and is real as of 2026-10-16. Nothing here is cited from a title alone. Marks: **[V]**
verified against the source · **[ours]** repo fact with file · **[S]** SPECULATION.

---

## TL;DR — no, and the reason is structural, not a knob

1. **[V] Our "1M recipe" has never run at 1M.** ADR-0004's long gate (512→4k→32k→256k→1M) has
   never executed; the longest config on disk is `max_seq_len = 2048`
   (`configs/b1.toml:13`). "Best recipe" is currently unfalsifiable — there is no long-context
   number of ours to compare. That is the honest answer to the owner's question.
2. **[V] The component mix is *not* what the 2026 frontier ships, but it is close to the
   research frontier.** Frontier: Kimi K3 = KDA + Attention Residuals at a **1M window** (2.8T MoE,
   104B active) — but KDA is one arm of a hybrid, not the only attention (2607.24653). Our shape
   (KDA always-on + a top-K block-sparse exact arm at 1/4 share) is a *published research
   architecture class* (PHBA, FreeSolo, A.X K2's SGA), not a frontier product recipe.
3. **[V] The selector is the biggest technical gap.** Our indexer scores **all** blocks against
   **all** queries: `O(t·⌈t/r⌉)`, which the code itself flags as "impossible at 1M"
   (`sparse.py:34-37`). The 2026 successors are hierarchical: NSA's coarse-compress→fine-select
   (2502.11089), PISA's pyramid top-K at **O(N log N)** with Triton kernels for training *and*
   inference (2609.31093), HISA's two-stage block-filter→token-refine (2603.28458). "Hierarchical
   coarse→fine" is not a plan of ours any more — it is a solved, published, kernel-available step.
4. **[V] The data gap is the make-or-break, and the literature is explicit.** Long-context ability
   is "mostly already acquired through large-scale pretraining" and is extended by **500M–5B
   tokens** of *length-upsampled, domain-balanced* data (2402.10171) — i.e. you need *some* long
   documents in a short CPT stage. We have none: a 19.7 GB corpus of mostly short web docs.
   And LA hybrids (ours) "benefit more from long-context continual pretraining" than SWA hybrids
   (2610.10114) — so the missing stage costs *us* more than it would cost a SWA design.
5. **[V] The cheapest honest 1M-retrieval fix may not be a trained arm at all.** NoteBank
   (2610.15715) raises Gated DeltaNet's mean accuracy across six RULER retrieval tasks from
   **1.7% → 70.3% at 1M tokens**, training-free, for ≤0.025% added analytical FLOPs. If that
   transfers, a trained sparse arm is an expensive way to buy what a memory sidecar gives away.
6. **[S] What I would not claim:** that our recipe is *worse* than SOTA. Nothing here measures our
   stack at long context. The defensible statement is: *it is unproven, and three specific pieces
   are missing relative to the published recipes.*

---

## Verdict table — component by component

| component | our current | SOTA 2026 | gap | source |
|---|---|---|---|---|
| always-on attention | KDA only, `use_msa=false` default; AttnRes wired but off | K3: KDA + AttnRes + Stable LatentMoE at 1M. Kimi Linear / Qwen3-Next: 3:1 linear:full | no exact-attention *layer* is always on; pure KDA loses ~0.12 PPL vs 3:1 **[V]** | 2607.24653; 2510.26692 (local `kda-sota-ceiling`) |
| exact-recall arm | top-K block-sparse, MQA indexer, r=4, K=32, 1/4 share | NSA hierarchical; PHBA top-K blocks **+ per-block prefix state**; A.X K2 SGA | our arm is not a layer in the stack; it is a gated arm that never won an A/B (ADR-0014, `use_msa=false` in every preset) | 2502.11089; 2610.08527; 2608.30181 |
| indexer cost | `O(t·⌈t/r⌉)` all-blocks score tensor; `sparse.py:34-37` names the ceiling | PISA pyramid top-K **O(N log N)**, LogSumExp, fused Triton, no QK matrix materialised | **quadratic selector is the gap**; ours is 2 levels shallower than PISA's log-N pyramid | 2609.31093; 2603.28458 |
| indexer training | planned: distill dense → joint sparse | A.X K2: optimise the indexer **against its own sparse top-k selection**, not the dense distribution — "markedly cheaper", 2048 positions read/query, RULER **94.6** to 256K | our dense-distillation stage may be the expensive way | 2608.30181 |
| long→short training | staged ladder planned (ADR-0004); `forward_train_state` truncates BPTT **[ours]** | StateFlow: chunk the sequence, propagate **boundary states and gradients** across chunks, profile-guided non-uniform chunking, 2.22× throughput / 2.45× memory at 256K **[V]** | truncated-BPTT penalty is unquantified by any paper found; StateFlow is the published answer to the same systems problem | 2608.06838; archive `long-context-agentic` R2 |
| document boundaries | byte stream, seeded file shuffle, no document mask **[ours]** | attention-isolated sequence packing, mathematically equivalent to unpadded **[V]** | at 1M a packed window is mostly cross-document attention; no counter exists for it | 2107.02027; 2608.22849 |
| data | 19.7 GB filtered web, mostly short docs **[ours]** | CPT on **500M–5B tokens**, length-upsampled + **domain-balanced** (naive book-upsampling is suboptimal) **[V]** | **no long stage exists**; LA hybrids need it *more* than SWA | 2402.10171; 2610.10114 |
| eval | long gate never run; ADR-0004 defines it | RULER (2404.06654), HELMET (2410.02694, 7 categories, ≤128K, NIAH does not predict downstream); SinkProbe for 1M diagnostics at small scale | we have zero long-context numbers; and NIAH-only would be a weak pass | 2404.06654; 2410.02694; 2609.08574 |
| sink / position | AttnRes wired, **off** | gated attention cut first-token mass 46.7%→4.8% (NeurIPS 2025); but at small scale "the training objective produces the sink, not the architecture" **[V]** | we run neither the diagnostic nor the gate | 2609.08574; local `new-tech-scan` #18 |
| 1M retrieval | Engram = hashed n-gram exact-match store | NoteBank: training-free sparse episodic memory, RULER retrieval **1.7%→70.3% @1M**, storage ≈1/2600 of MHA KV **[V]** | Engram is n-gram key→value, not a sparse text archive with native replay | 2610.15715 |

---

## Candidate comparison (≥5 read; only the load-bearing ones)

| candidate | source / date | key facts or metrics | credibility / evidence type | effort to adopt here |
|---|---|---|---|---|
| **PISA** pyramid top-K | 2609.31093, 2026-09-25 | O(N log N) via O(log N) pooled key levels + LogSumExp; Triton for train+infer; comparable commonsense, **better retrieval** | **[V]** abstract; no code link in abstract; already screened in `new-tech-scan` #19 | medium — replaces `FlexSparseSelector`; ~200-400 LOC torch/Triton reachable |
| **HISA** two-stage indexer | 2603.28458, COLM 2026 | block coarse filter → token refinement inside retained blocks; drop-in for DSA indexer; **no extra training**; matches DSA quality, beats block-sparse baselines | **[V]** abstract, published venue | low-medium — a coarse stage *in front of* the existing token-level scorer |
| **NSA** | 2502.11089, v2 2025-02-27 | dynamic hierarchical sparse: coarse token compression + fine token selection; end-to-end trainable; maintains/exceeds full attention; speedups at 64k fwd/bwd/decode | **[V]** abstract (abs page) | medium; FLA `parallel_nsa` rejected locally on head-count constraints (`attention.py:17-24`) |
| **PHBA** | 2610.08527, 2026-10-06 | replaces SWA with **top-k block retrieval + a compact prefix state per block** (gated linear recurrence at block boundaries); Triton streaming; beats NHA/linear/hybrid baselines | **[V]** abstract; preprint | medium — closest published match to "our two arms in one layer" |
| **FreeSolo** | 2610.17191, 2026-10-12 | SWA short-range branch + keypoint-cell long-range selection; FS-L prunes the prefix via short CPT; FS-B **O(N log²N)** incl. radix sort; better **zero-shot length extrapolation** without long-context fine-tuning | **[V]** abstract; preprint | medium; the "no long CPT" claim is exactly our constraint |
| **A.X K2 / SGA** | 2608.30181, 2026-08-30 | sparse + gated attention; indexer warmup against **its own sparse top-k**, 2048 positions/query; **RULER 94.6 to 256K**; GN for NVFP4 stability | **[V]** abstract; 688B MoE, scale far above ours | low (recipe, not code) — changes our indexer training objective |
| **Mechanics of Long-Context Hybrid Models 1.1** | 2610.10114, 2026-10-07 | **Seesaw Effect**: LA hybrids gain more from long-context CPT, SWA hybrids from extrapolation. SWA's Short-Context Learning Trap / Short-Window Weariness / Long-Window Laziness; SW-LA reaches **16× training-free extrapolation, 100% NIAH-SK1 @64k** | **[V]** abstract; 60-page preprint, under review | low — it is a design rule, read it before the ladder |
| **StateFlow** | 2608.06838, 2026-08-07 | sequence pipeline parallelism for linear recurrence; boundary state **and gradient** propagation across chunks; up to 32B / 256K; **2.22× throughput, 2.45× memory** vs pipeline parallel | **[V]** abstract; systems paper, multi-GPU | low on one GPU — the *idea* (carry grads, don't truncate) transfers, the system does not |
| **NoteBank** | 2610.15715, 2026-10-11 | training-free sparse episodic memory + native replay; GDN **RULER retrieval 1.7%→70.3% @1M**; ≤0.025% FLOPs; 1.3B→8B backbones | **[V]** abstract; preprint | medium — an inference-side alternative to the MSA arm |
| **Fu et al.** data engineering | 2402.10171, 2024-02-15 | long-context ability "mostly already acquired"; **500M–5B tokens** of length-upsampled + **domain-balanced** data extend 4K→128K; naive book upsampling is suboptimal | **[V]** abstract; 2024, still the reference recipe (no 2026 replacement found) | **high value, low cost** — it is a data-pipeline change |
| **SinkProbe** | 2609.08574, 2026-09-08 | 1M-window sink/position diagnostics on four small models; "the training objective produces the sink, not the architecture"; gating did **not** reproduce its published effect at small scale | **[V]** abstract; small-scale by construction | low — run the diagnostic |
| ~~AllMem~~ (rejected for us) | 2602.13680, 2026-02-14 | SWA + non-linear TTT memory; 4k window within 0.83 of full attention at 37k | **[V]** abstract | rejected: TTT memory is a second recurrent mechanism beside KDA; scope creep |

**Rejected with reason:** RAG-side KV-eviction papers (DropKV 2610.22130, PuzzleKV 2608.23843,
RestoreKV 2608.01247, MOIRA 2610.04313, SPIN 2610.09025) — decode/serving-time, we are training;
XAttention (2503.16428) and JSSA (2610.19026) — already screened in `new-tech-scan` #19;
Jet-Long (2607.07740) — RoPE rescaling, and we have no RoPE in the KDA arm; MegaBeam 512K
(2505.08651) and QwenLong-CPRS (2505.18092) — 7B+ scale, not transferable to a 1B/16 GB box.

---

## Ranked gaps (expected impact, each with its source)

1. **DATA — no long-document / length-upsampled stage.** 2402.10171: 500M–5B tokens of
   length-upsampled + domain-balanced data; naive single-domain upsampling is suboptimal.
   2610.10114: LA hybrids (ours) gain *more* from long-context CPT than SWA hybrids, and SWA
   hybrids need *longer windows* to escape the Short-Context Learning Trap. Our corpus is short
   web docs, so this stage cannot be built from the corpus as-is.
2. **SELECTOR — quadratic indexer.** `sparse.py:34-37` (`O(t·⌈t/r⌉)`); the fix is published:
   PISA O(N log N) (2609.31093), HISA two-stage (2603.28458), NSA (2502.11089).
3. **INDEXER OBJECTIVE — we plan dense distillation; SOTA distils against the sparse top-k
   itself** (2608.30181), which is the cheaper and better-measured path (RULER 94.6 @256K).
4. **STATE CARRY — truncated BPTT vs boundary-state+gradient propagation** (2608.06838).
   No paper quantifies the truncation penalty; it is the first suspect if the ladder underperforms
   (archive `long-context-agentic` R2).
5. **PACKING — no cross-document attention mask.** Attention-isolated packing is standard and
   provably equivalent (2107.02027; RIBOSPAN 2608.22849 applies it at 10,240 nt). At 1M, an
   unmasked packed window is mostly noise attention.
6. **EVAL — the long gate has never run.** RULER (2404.06654) + HELMET (2410.02694) are the
   reference instruments; NIAH alone does not predict downstream, and 2605.23170 shows vendors
   do not even report RULER in main tables — a NIAH-only pass would be self-deceiving.
7. **EXACT-ATTENTION SHARE — 1/4 of demand via a gated arm, vs a full attention layer every 4.**
   Kimi Linear's own ablation: pure linear = 5.77 PPL vs 3:1 = 5.65; 15:1 = 5.82. Our arm is
   *off by default* and has never been A/B'd (ADR-0014).
8. **SINK / POSITION — AttnRes off, no sink diagnostic.** 2609.08574 is cheap and says the
   effect may not reproduce at our scale, which is itself decision-relevant.
9. **RETRIEVAL SIDECAR — NoteBank's 1.7%→70.3% at 1M is training-free** (2610.15715). Worth
   pricing before funding a trained sparse arm.

---

## Next 3 experiments on one GPU (16 GB, sequential, `ram-guard` + MemoryCgroup per §1.5)

**E1 — the state-carry / truncation probe (cheapest, decides E3's shape).**
Same short data, two arms: truncated BPTT vs gradient-carrying across 2 chunks, evaluated on a
synthetic byte NIAH at 4×/8× train length. Config: `small`, 3 seeds, ≤2k steps, pure CE,
`--eval-depths`. **Cost: ~3 × 2k-step runs at the port's measured step time; no new data.**
Gate: if carry does not beat truncation on the NIAH probe, E3's schedule changes.

**E2 — the hierarchical selector, offline first.** Implement PISA-style pyramid top-K
(2609.31093) or HISA's coarse stage (2603.28458) behind the existing `FlexSparseSelector`
interface; verify block-set agreement vs the exhaustive selector on a 4k/8k window *before*
any training. **Cost: ~1 day of CPU/GPU-light work + one 6-step GPU parity run at b8 s512.**
Gate: identical block sets where the pyramid is exact; measured peak memory ≤ the chunked
gather's 3.15 GiB (measured 2026-10-16, `attention.py:57-66`).

**E3 — the long stage, on whatever long data E1 justifies.** Build a length-upsampled +
domain-balanced CPT mixture (2402.10171's recipe) and run 512→4k→32k with attention-isolated
packing (2107.02027) and the E2 selector. **Cost: dominated by data engineering, not GPU; the
GPU part is one ladder of short runs.** Gate: RULER-style byte NIAH at 32k; a pass is
position-robust recall at 4× the trained length, not a single-needle score.

---

## VERIFIED (sourced)

- Our arm as built: MQA 4q/1k indexer, avg-pool r=4, block-causal ReLU, top-K=32, adaptive 1/4
  share, `O(t·⌈t/r⌉)` score tensor, chunked gather 13.59→3.15 GiB — `port/.../sparse/sparse.py`,
  `sparse/attention.py` (read 2026-10-16).
- `use_msa = false` in every preset; MSA never won an A/B; the Rust crate was deleted (ADR-0014).
- Longest config on disk is `max_seq_len = 2048` (`configs/b1.toml:13`); ADR-0004's long gate
  is unrun.
- Every paper claim above, against its arXiv abstract or abs page, 2026-10-16.
- Kimi K3: 2.8T MoE, 104B active, **1M-token window**, KDA + Attention Residuals + Stable
  LatentMoE (16/896 experts) — 2607.24653v2.

## SPECULATION (my inference)

- **[S]** K3's *exact* attention mix and long-context data recipe are not in the abstract; I assume
  it remains a KDA-dominant hybrid with some exact arm, per Kimi Linear's 3:1 and the local
  `new-tech-scan` #18 — unverified.
- **[S]** That NoteBank's 1.7%→70.3% transfers to a byte-level model with a single shared
  recurrent block. The paper's backbones are 1.3B–9B multi-layer; our state is one layer's worth
  (archive `long-context-agentic` §2). No source tests our topology.
- **[S]** That a 19.7 GB short-document corpus can be length-upsampled to 500M–5B tokens of
  *domain-balanced* long sequences at all. 2402.10171 assumes a corpus with long documents in it.
- **[S]** That the truncated-BPTT penalty matters at 32k. No paper quantifies it; StateFlow's
  numbers are multi-GPU throughput, not loss.
- **[S]** Cost figures for E1–E3: E1 and E3 are priced in "runs", not GPU-hours, because
  `docs/protocols/AB-PROTOCOL.md`'s per-arm cost is void (§3.3, AGENTS.md).

## Open questions

1. Can our corpus yield a length-upsampled, domain-balanced long stage without new crawl?
2. Is the pyramid selector's block set *identical* to the exhaustive selector at 1M, or only
   close — and does the difference show up as recall loss?
3. Does the sink diagnostic (2609.08574) reproduce at 9.2M params on this box?
4. Does the A.X K2 indexer objective (sparse-top-k distillation) work without a trained indexer
   warmup stage, i.e. from a random indexer?
5. What does RULER-style byte NIAH cost us per rung — is the 500 MB eval tail enough windows at
   1M, as ADR-0004 already worried?

## Sources

Primary (all resolved 2026-10-16 on arxiv.org): 2502.11089 NSA · 2607.24653 Kimi K3 ·
2610.10114 Mechanics of Long-Context Hybrid Models 1.1 · 2610.17191 FreeSolo · 2610.08527 PHBA ·
2609.31093 PISA · 2603.28458 HISA · 2608.30181 A.X K2 · 2608.06838 StateFlow · 2610.15715
NoteBank · 2402.10171 Fu et al. · 2107.02027 Krell et al. · 2608.22849 RIBOSPAN ·
2404.06654 RULER · 2410.02694 HELMET · 2609.08574 SinkProbe · 2605.23170 positional failures ·
2602.13680 AllMem · 2605.22791 GDN-2.

Repo: `src/dormouse/model/sparse/{sparse,attention,__init__}.py` · `configs/b1.toml` ·
`docs/adr/0004-context-ladder.md` · `docs/adr/0014-msa-cut.md` ·
`docs/research/2026-10-16-new-tech-scan.md` · `docs/research/2026-10-15-1b-tech-verification.md` ·
`docs/research/2026-09-27-kda-sota-ceiling.md` ·
`docs/archive/research/2026-09-26-long-context-agentic.md` · AGENTS.md §3.5.
