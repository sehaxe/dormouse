# Long-context + agentic roadmap for dormouse (7.5M byte-level, KDA, 16 GB)

Research pass 2026-09-26. Sources opened: arXiv 2510.26692 (Kimi Linear), 2605.22791 (GDN-2), 2507.20534 (Kimi K2), 2609.29421 (Rufus-Air), 2410.23218 (OS-Atlas), 2505.24183 (CodeV-R1), 2402.18510 (Wen et al.), Qwen3-Next blog/HF docs, fla-org/flash-linear-attention repo, AgentTuning (ACL 2024 findings + HF dataset). Every number below was read from these sources; nothing projected.

---

## 1. How the FLA-family trains long context (RQ1)

**The 2025-2026 recipe is uniform: train short for ~99% of tokens, extend only in the final anneal, and let the recurrence generalize length.** There is no RoPE-scaling analog for decay-based attention — because there is no RoPE: the decay *is* the positional encoding.

| Model | Bulk pretrain | Long-context stage | Context reached | Optimizer |
|---|---|---|---|---|
| Kimi K2 (2507.20534 §2.5) | 15.5T tokens @ 4,096 ctx | anneal 400B @ 4K (LR 2e-5→7e-6) then **60B @ 32K**, then YaRN → 128K | 128K (256K later) | MuonClip, WSD, batch 67M tok |
| Kimi Linear (2510.26692 §5.4.1) | **4,096 ctx**, 1.4T (ablation) / 5.7T (final), LR 1.1e-3, batch 32M tok | "the same annealing schedule and long-context activation phase established in Kimi K2" | **1M** released checkpoint | MuonClip, WSD |
| GDN-2 (2605.22791 §4) | **4K only**, 100B FineWeb-Edu, 1.3B params | none — evaluated at 8K anyway | 8K tested | AdamW 4e-4, cosine, batch 0.5M tok |
| Qwen3-Next (qwen.ai blog + HF) | 15T tokens | long-context curriculum inside the 15T | 262,144 native, YaRN → 1,010,000 | — |

Key structural facts:

- **No length-extension machinery for the linear layers.** Kimi Linear is NoPE end-to-end: "KDA layers encode position through their recurrence, and the full-attention layers are left without positional encoding" (HF transformers docs; paper §4 "No Position Encoding (NoPE) for MLA Layers"). Their RoPE ablation confirms the RoPE variant is *worse* on long context (avg 51.8 vs 54.5, paper Table 5) precisely because an explicit positional signal in the global layer "makes the model less flexible when adapting mid-training to extended contexts." Qwen3-Next keeps partial RoPE only on the first 25% of the full-attention head dims.
- **State-carry vs chunkwise-parallel:** all three families train the recurrence with the chunkwise WY algorithm — gradient flows across *every* chunk (state forward, `dht` backward), exact BPTT over the whole window, linear memory via recompute. See `fla/ops/kda/chunk_bwd.py` (`chunk_gated_delta_rule_bwd_dhu` chains `dht` through all chunks) — chunk_size 32/64, plus context-parallel KDA/GDN landed in FLA 2026-03. **Dormouse's `forward_train_state` (truncated BPTT, chunked state transfers) is an approximation of this; the 2025-26 cohort does not truncate.** No paper in this set studies the truncation gap, so its cost at 8× chunk length is unknown — measure before trusting.
- **Long-from-short curriculum size:** the long stage is ~1% of tokens (K2: 60B of 15.5T; the 4K anneal 400B ≈ 2.6%). Length extension is cheap in tokens; the expensive part is the 4K bulk.

## 2. State capacity: when the constant state breaks (RQ2)

**The best controlled experiment is GDN-2 §4 Table 3** — 1.3B models, state *matched* across architectures at H·dk·dv = 16·128·128 = 262,144 floats/layer (= 128·d_model, d_model 2048), all trained on the same 100B tokens, recurrent-only (no attention at all):

| Recurrent-only, accuracy % | 1K | 2K | 4K | 8K |
|---|---|---|---|---|
| KDA, S-NIAH-1 (passkey) | 100 | 100 | 99.2 | **70.6** |
| GDN-2, S-NIAH-1 | 100 | 100 | 100 | **97.8** |
| KDA, S-NIAH-3 (word needle) | 77.4 | 63.2 | **26.2** | — |
| GDN-2, S-NIAH-3 | 92.0 | 89.8 | **31.8** | — |
| KDA, MK-NIAH-1 (multi-key) | 54.0 | 44.2 | **28.0** | — |
| GDN-2, MK-NIAH-1 | 72.6 | 51.4 | **37.8** | — |

Reading, tied to dormouse's ~2 MiB state ([B,H,K,V]; same order as one Kimi-Linear KDA layer at 32·128·128 = 524,288 elements — but dormouse has **one shared block**, so total in-context associative memory ≈ **one KDA layer**, not 20):

- **A single state of this size holds exact needles up to ~2K tokens, degrades hard at 4K, and collapses on multi-key interference by 4-8K.** A better update rule (GDN-2's channel-wise erase/write decoupling) buys +14-20 points on MK-NIAH at *identical* state size — the update rule matters, but no update rule fixes the ceiling.
- The 4K-trained Transformer baseline scores **0.0** on S-NIAH-1@8K — length generalization is the linear models' advantage (trained 4K, still 97.8 at 8K), while their *depth* of recall is state-bounded. Both facts matter for dormouse.
- Theory (Wen, Dang, Lyu, arXiv:2402.18510, ICLR 2025): fixed-state RNNs cannot solve exact-retrieval tasks regardless of CoT; **RAG or a single full-attention layer provably closes the representation gap**. Kimi Linear's own words: "Long-context retrieval remains the primary bottleneck for pure linear attention" (§4).
- **The known fixes, ranked by evidence:** (1) hybrid full-attention layers (below); (2) better update rule — GDN-2 shows real but bounded gains; (3) content-addressed external memory (retrieval augmentation per 2402.18510) — **dormouse's Engram *is* this: FNV-hashed n-gram tables are an exact-match key→value store outside the recurrent state**; (4) state-size scaling (HGRN2 lineage, cited in both papers) — cheap for dormouse only via wider H/K/V, costs VRAM linearly.

## 3. Hybrid architectures: what 2026 SOTA actually ships (RQ3)

- **Kimi Linear: 3:1 KDA:MLA**, 27 layers = 20 KDA + 7 full MLA (HF `layer_types`, full at 4,8,12,16,20,24,27; 48B total/3B active MoE). The ablation (Table 1): val PPL 3:1 = 5.65, 7:1 = 5.70, 15:1 = 5.82, **0:1 pure KDA = 5.77**, 1:1 = 5.66. So pure-linear loses ~0.12 PPL and rare-full loses ~0.05; 3:1 is the quality-throughput knee (75% KV-cache reduction, 6.3× TPOT at 1M: 1.84 ms vs MLA 11.48 ms).
- **Qwen3-Next: same 3:1** — `12 * (3 * (Gated DeltaNet -> MoE) -> 1 * (Gated Attention -> MoE))`, 48 layers, 80B-A3B, 15T tokens. Their stated finding: "When we mix Gated DeltaNet with standard attention at a 3:1 ratio, the model consistently outperforms any monolithic architecture."
- **GDN-2's alternative hybrid: SWA in every block** (GDN-2 → MLP → SWA(2K) → MLP), no global full attention at all. This preserves exact *local* retrieval and linear scaling, at the cost of global needles.
- **Consensus for long-context agentic workloads: ~75% linear + 25% exact-global, or linear + sliding-window.** Nobody ships pure linear at agentic scale.

## 4. Agentic post-training data (RQ4)

**Rufus-Air (2609.29421)** — the 106B reference recipe: 8 serial stages (SFT → Reasoning RL → Coding RL → IF RL → General Agent → Coding Agent → Search Agent → RLHF), "from basic to advanced capabilities and from hard, verifiable rewards to softer judge-based signals." Transferable findings: (i) diverse high-quality SFT sets the capability floor; (ii) difficulty-filter RL prompts into the productive band; (iii) stage order follows reward reliability. Nothing in it is small-model-specific.

**What small models actually need — evidence:**

- **AgentTuning / AgentInstruct (2310.12823, ACL 2024): 1,866 verified trajectories** across 6 tasks (ALFWorld 336, WebShop 351, Mind2Web 122, KG 324, OS 195, Database 538; avg 5.24 turns, ReAct-style CoT), mixed with general ShareGPT data at **η=0.2 agent fraction**. AgentLM-7B: ALFWorld 2.0→84.0, WebShop 4.4→63.6 vs base Llama-2-chat. Two laws from the paper: **quality filtering beats volume** (only 5.28% of 35,341 candidate instructions survive), and **agent-only data degrades generalization — the general mix is mandatory**.
- **Kimi K2 §3.1.1 — the synthetic pipeline that scales:** 3,000+ real MCP tool specs scraped from GitHub + 20,000+ evolved synthetic tools (WizardLM-style domain evolution), thousands of agent configs, rubric-annotated tasks, multi-turn user simulation, **LLM-judge filtering against rubrics (rejection sampling)**, and real sandboxes where verification is cheap (code: test-suite pass rate). Output: "tens of thousands" of trajectories. Critically for dormouse: **the pipeline is synthetic and self-verifying — it does not need a frontier teacher for the tool-use shell** (K2 uses DeepSeek-class models for NL quality, but rubric = executable check works without).
- **OS-Atlas (2410.23218, ICLR 2025 spotlight)** — GUI grounding: 13.58M elements / 2.24M screenshots across 5 platforms (released, HF `OS-Copilot/OS-Atlas-data`; 816 GB with images), then action fine-tune on only ~135K instruction-grounding + trajectory samples (AMEX/AITZ/Mind2Web/AndroidControl/Wave-UI). Their ablation: skipping grounding pretraining "significantly degrades performance" — grounding is the capability floor, trajectories are the polish. Caveat for dormouse: this is a VLM recipe; dormouse would consume the **text UI channel** (accessibility tree / DOM text) as bytes, for which the same two-stage shape (locate-then-act) still applies.
- **Minimal viable agentic SFT recipe (synthesis of the above):** 1-5k self-generated, rubric-filtered tool-use trajectories (schema: system + tools + multi-turn ReAct over a *simulated* environment you can execute in-process) mixed ~1:4 with general instruction data; grounding-style subtasks (find-the-element in text UIs) as a first stage. That is the AgentTuning scale — it demonstrably works at 7B; at 7.5M parameters expect format-following and simple loops, not autonomous browsing.

## 5. Chip design / Verilog (RQ5)

- **CodeV-R1 (2505.24183, NeurIPS 2025)** — the template: GitHub Verilog → DeepSeek-V3 NL summaries (~150K NL-code pairs) → R1 "thought+code" regeneration → difficulty-filter → **SFT on 87K** → automated rule-based testbench generation (96.1% fewer false negatives than LLM testbenches) → equivalence-check round-trip → difficulty filter → **RLVR on ~3.1K hard prompts** (adaptive DAPO). Result: CodeV-R1-7B = **68.6-68.8% pass@1 VerilogEval v2, 72.9% RTLLM v1.1**, beating 671B DeepSeek-R1 on RTLLM, for 2,656 A100-hours. Model, data, and training code are released (github.com/IPRC-DIP/CodeV-R1, HF).
- **The striking counterpoint (same paper, Table 2): DeepSeek-R1-Distill-Qwen-7B scores 0.6-11.3% on VerilogEval v2** — generic reasoning distillation transfers ~nothing; the domain RLVR loop is what produces the score.
- Cost structure is dormouse-friendly: rewards come from `iverilog`/`verilator` + testbenches on **CPU**, the RL prompt set is tiny (~3K), and raw Verilog is plentiful on GitHub for byte-level pretraining exposure. But honesty about scale: all published pass@1 numbers are 6.7B-671B models; there is no sub-100M Verilog result in any opened source, and a 7.5M byte model should be expected to land far below the 7B class. Verilog for dormouse = a *verifiable-reward training domain* (fits docs/architecture/post-training.md's RLVR plan), not a near-term EDA tool.

---

## 6. FINAL — Roadmap for dormouse (ranked)

**R1. Keep seq_len 512 as the bulk; length-ladder only at the end: 512 → 2K → 8K.**
The entire 2025-26 cohort trains ≥95% of tokens at 4K and spends ~1% on the extension stage; GDN-2 shows 4K-trained linear models still retrieve at 8K, and Kimi Linear shows the recurrence generalizes length without any positional surgery (NoPE; dormouse's KDA likewise has no RoPE to rescale — **no YaRN-analog is needed or exists for decay attention**). Concrete: finish current 512 runs → 2K stage (state-carry chains of 4×512, a few % of tokens) → 8K stage (~1% of tokens) → evaluate; only extend further if 8K works. Byte-level means 8K bytes ≈ 2K BPE tokens, so dormouse's target is modest by token standards.

**R2. Fix the truncated-BPTT gap before trusting long-context numbers.**
FLA's chunked backward (KDA/GDN, chunk 32/64) propagates `dht` across *all* chunks — exact gradient through the whole sequence, linear memory via recompute (`disable_recompute=False` path). Dormouse's `forward_train_state` truncates. No opened paper quantifies the truncation penalty; at 8× the chunk count it is the first suspect if the 8K stage underperforms. Cheapest experiment: same 8K data, truncate vs carry-gradients-through-2-chunks, compare loss at 8K positions.

**R3. Do not add a full-attention layer; sharpen the arms dormouse already has.**
The 3:1 hybrid exists to give pure-linear stacks an exact-retrieval path. Dormouse already has two: the **MSA top-k sparse attention arm** (a global, state-independent path — functionally the "1 full layer" of the hybrids, at top-k cost) and the **Engram** (exact-match external memory — the retrieval augmentation that Wen et al. name as a provable fix). The evidence-backed requirement is not "add attention layers" but "at least one exact-retrieval path must survive long contexts." Action: length-ladder the *existing* architecture; verify with RULER-style byte NIAH (R4). If MSA-at-8K fails, the 2026-consensus fallback is one SWA-style local-exact arm or a periodic full-attention pass inside the loop — a config flag, not a redesign.

**R4. Build the state-capacity probe first (days, not weeks).**
GDN-2's §4 suite is small and synthetic: S-NIAH-1/2/3 + MK-NIAH-1 at 1/2/4/8K, plus the real-retrieval set (SQuAD/TriviaQA/DROP-style). Re-implement at byte level over dormouse's corpus (needles = byte strings). This is the instrument that answers "does the state carry it" for *your* H,K,V — no published number transfers exactly, because capacity scales with state size and dormouse's is one layer. Decision rule from the GDN-2 table: passkey ≥90% at 4× train length is healthy; MK-NIAH collapse at 2× is the expected failure mode.

**R5. State-size and update-rule upgrades are the leverage points if the probe fails.**
(a) **GDN-2's decoupled erase/write gates** are the 2026 SOTA delta-rule upgrade: channel-wise erase gate matters most (their ablation: b-only-channel ≈ full model, w-only scalar ≈ much worse) — and GDN-2 reduces *exactly to KDA* when tied, so it's a strict superset of what dormouse runs; code: NVlabs/GatedDeltaNet-2 + FLA. (b) Widen H/K/V: capacity is linear in state elements and dormouse's 2 MiB budget has room before VRAM binds. (c) FLA also lists 2026 additions worth tracking (Preconditioned KDA/GDN, Raven sparse-memory-routing) — noted, not yet read.

**R6. Agentic post-training: synthetic tool loop at AgentTuning scale, grounded in Engram-style lookup.**
Recipe: (1) generate 50-200 tool specs (JSON-ish schemas), simulate environments in-process (they're just functions — no deployment needed); (2) rollout ReAct trajectories with the current model, filter by executable rubric (exit code / exact-match answer — the K2 pipeline minus the LLM judge); (3) keep 1-5k trajectories, mix ~1:4 with general instruction data (the η=0.2 law); (4) stage before that: grounding subtasks (given a text-UI tree, point at the element) à la OS-Atlas, which is the capability floor in their ablation. RL later rides the same environments (Rufus-Air: verifiable rewards first, judge-based last). Byte-level is an advantage here: tool I/O is text; no tokenizer boundary artifacts in tool arguments.

**R7. Verilog as the first RLVR domain, expectations calibrated.**
The CodeV-R1 assets are open (dataset + testbench generator + training code); rewards are CPU-cheap (iverilog + testbench equivalence). Path: GitHub Verilog into the byte corpus now (free pretraining exposure), then SFT on a few-k NL→Verilog pairs, then RLVR on ~1-3K difficulty-filtered prompts. Expected outcome at 7.5M params: not the 68% class — the point is the *training discipline* (verifiable reward loop end-to-end) and a first chip-design foothold.

**Skip list (with reasons):** YaRN/RoPE-scaling machinery (nothing to scale — no RoPE); MoE or any 3:1-layer-hybrid rebuild (R3's arms already exist); real-browser SFT data collection (synthetic envs are the K2 precedent and OS-Atlas shows grounding-first); Mamba-3-style recurrence changes (GDN-2 is the nearer, KDA-compatible upgrade).

**Bottom line:** the constant state does *not* carry exact retrieval past ~2-4K bytes at dormouse's capacity — that is measured, not conjectured. But dormouse's architecture already contains both fixes the literature offers (sparse global attention + exact-match external memory); the work is (i) the length ladder with an honest probe, (ii) closing the truncated-BPTT gap, and (iii) the synthetic-verifiable agentic/Verilog data loops. 100k+ byte contexts are reachable as *working-memory-over-retrieval* (Engram + decayed gist), not as raw in-state needle storage — same conclusion Kimi Linear reached at 48B scale, which is why they kept 25% full attention.
