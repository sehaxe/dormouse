# Patching for a byte-level loop model — research review

**Lane:** should the architecture become patch-based? (BLT / MegaByte / SpaceByte vs our
weight-shared LoopBlock over raw bytes). **Date:** 2026-10-02. **Scope:** research only —
no code touched; every claim about foreign work cites a PDF in `docs/papers/` (provenance
rows at the bottom); every claim about this box cites `AGENTS.md` §3.1 or a `file:line`.

The brief's arXiv id for BLT was wrong in one digit: the paper is **arXiv:2412.09871**
(2412.09771 is an Ising-model physics paper; it was downloaded by mistake and deleted).

---

## 1. Mechanics, from the papers

### 1.1 BLT (arXiv:2412.09871, FAIR, Dec 2024; ACL 2025)

Three modules instead of one transformer (`2412.09871-blt.pdf` §3, Fig. 2):

1. **Local encoder** — small byte-level transformer (`l_E << l_G` layers), window
   attention `w_E`. Its byte embeddings are **augmented with hash n-gram embeddings**,
   orders n = 3…8, `RollPolyHash(g) % |table|` added to the byte embedding, Eq. 2-4
   (§3.2.1). Perceiver-style **cross-attention pools the bytes of each patch into one
   patch representation** (§3.2.2, Eq. 5-8).
2. **Latent global transformer** — the large autoregressive model over patch
   representations, block-causal; *"this model consumes the bulk of the flops during
   pre-training as well as inference"* (§3.1). Its sequence length is `n_bytes / avg_ps`.
3. **Local decoder** — small byte-level transformer; cross-attention with the roles
   reversed (patch reps are keys/values, byte reps are queries), then per-byte
   prediction (§3.3). **The loss is per-byte**, exactly like a byte model.

**Entropy patching** (§2.3): a small byte LM (default **100M params, 14 layers, hidden
512, sliding window 512 bytes**, §4.2) computes next-byte entropies `H(x_i)`; a patch
boundary is placed where `H(x_i) > θ_g` (global threshold) or
`H(x_i) − H(x_{i−1}) > θ_r` ("approximate monotonicity"); θ is tuned on the training mix
to hit a **target average patch size** (§4.3). Boundaries are computed in the
**dataloader**, not on device (§2.3). Scaling the entropy model: *"diminishing returns
when we scale beyond 50m parameters"* with a 512-byte context (§7, Fig. 8); a
2-byte-context CNN is shown as a working cheap variant (Fig. 3f). **Incremental
patching** is stated as a hard requirement for generation: `f_p(x_{<i}) = f_p(x)_{<i}`
(§2.4) — BPE violates it, entropy patching satisfies it.

**Headline numbers** (§5): training-FLOP-controlled parity with Llama 3 up to **8B params
/ 4T bytes**; **up to 50 % fewer inference FLOPs** at average patch size ≈ 8
(inference flops ∝ 1/avg patch size); the fixed-inference scaling study (Fig. 1) shows
BLT overtaking BPE **only past the compute-optimal budget** — *"BPE models perform better
with small training budgets and are quickly surpassed by BLT"* (§5.3), crossover at
~2.5-3× the compute-optimal training budget at 400M-1B scale. Parameter accounting
(§5.3): *"when growing total parameters 20x from 400M to 8B, we only roughly double
BLT's local model parameters"* — the local byte modules are a rounding error next to the
latent model at scale.

**Patching ablation** (§7, Fig. 6 + Table 6): *all* dynamic schemes beat fixed-stride
patching; **space patching is a very close competitor to entropy patching** (8B
benchmarks: Arc-E 67.2 space vs 68.9 entropy; HellaSwag 70.8 vs 72.7). This matters for
us: most of the win is **boundary alignment**, not entropy estimation.

### 1.2 MegaByte (arXiv:2305.07185, Meta, May 2023)

The ancestor: **fixed** patch size P. Patch embedder (causal conv) → global transformer
over `T/P` patch reps → local transformer decodes each patch's bytes (§2.2). Cost model
(§2.1): global O(T²/P), local O(T·P), total **O(T²/P + TP)**; *"more than 98 % of FLOPS
are used in computing position-wise feedforward layers"*, so per-patch FFNs let
*"a model P times larger than a transformer with equivalent FLOPS"* be trained (§2.1).
Generation runs the global model once per patch and the local model on K patches in
parallel (§2.3). Superseded by BLT/SpaceByte on the boundary question: BLT lists
strided patching's flaws (§2.1) and SpaceByte measures the gap (below).

### 1.3 SpaceByte (arXiv:2404.14408, NeurIPS 2024)

A **byte-level** transformer with extra **larger "global" blocks inserted in the middle**,
applied **only after "spacelike" bytes** — a byte that is not a letter, digit, or UTF-8
continuation byte — that are not preceded by another spacelike byte (§2). Average patch
≈ 6 bytes (PG-19, arXiv), ≈ 8 (Github) (§4.1). Global blocks attend over global
positions; local blocks use sliding-window attention (§4.1). The whole mechanism is a
**rule on the input bytes** — no learned patcher, nothing at decode time beyond the same
rule.

**Table 1 is the only small-scale, compute-controlled result in this family** — 10¹⁹
training FLOPs, models of tens of millions of params, BPB on three modalities (§4, the
paper's own table; "lowest bits-per-byte"):

| model | PG-19 | arXiv | Github |
|---|---|---|---|
| Transformer (byte-level) | 1.138 | 0.909 | 0.655 |
| MegaByte | 1.083 | 0.822 | 0.570 |
| SpaceByte (**fixed P**) | 1.112 | 0.804 | 0.552 |
| **SpaceByte** (spacelike rule) | **1.009** | **0.748** | **0.500** |
| Transformer (GPT2 tokenizer) | 1.013 | 0.796 | 0.554 |
| Transformer (SentencePiece) | 0.989 | 0.768 | 0.508 |

Readings: (i) the byte-level baseline needs *"roughly 10 times more training FLOPs"* to
match subword (§1, abstract-adjacent claim); (ii) dynamic boundary alignment closes ~all
of the gap to subword **at equal training FLOPs at small scale**; (iii) **fixed P
recovers only part of it** — 1.112 vs 1.009 on PG-19, i.e. the boundary rule is worth
~0.10 BPB at this scale; (iv) the paper's own caveat: *"For other data modalities,
SpaceByte with our simple patching rule might not be as effective"* (§1).

---

## 2. What those numbers mean on THIS box — and what they do not

### 2.1 The papers' saving targets a component we barely pay for

BLT's and MegaByte's efficiency story is: the **dominant** model (softmax attention +
per-position FFNs) runs once per patch instead of once per byte. Two of our facts break
the transfer:

- **Our attention is O(n) linear (KDA), not O(n²) softmax**, and our windows are 512
  bytes (`schema.rs:124-125`) — the quadratic term the papers amortise is ~0 here
  regardless. MegaByte's O(T²/P) column simply has no analogue in our step.
- **We are launch-bound, not FLOP-bound** (AGENTS §3.1): warm-step GPU utilisation is
  **13.3 %** (79 % of samples ≤ 5 %), and the batch ladder reads
  batch 8/16/32 → **244/440/826 ms** (throughput 16.8K/18.6K/19.8K B/s). Patching
  shrinks tensors and per-position GEMM work but does **not** reduce the number of
  kernel launches — the iteration count, and therefore most of the launch train, is
  untouched. Solving F + 4S = 3.39·(F + S) on the batch ladder (derived, not measured):
  ≈ **20 % of a warm step is batch/position-independent floor**; the remaining ~80 %
  scales with positions×batch and is what patching attacks. The TSCT retraction
  (**52-64 ms, fixed, batch-independent**, AGENTS §3.1) is entirely untouched by
  patching.

Honest conversion of patch size ps ≈ 4-6 into our currency: the LoopBlock's per-iteration
sequence drops 512 → ~512/ps, so controller GEMMs, the KDA chunked pass, expert FFNs and
`out_proj` — everything shaped `[b·t, ·]` (`loop_block.rs:782`, `:832`, `:883`) — shrink
by ps, while the byte-level parts (embedding lookup, per-byte CE over 256, `lm_head`,
`model.rs:380`) and the fixed retr cost stay. Band estimate (derived from the two mould
rows above, to be settled by P0): **step time −25…−45 %, or equivalently ~3-5× the bytes
per step at ~1.4-1.8× the step time — not ps× anything.**

### 2.2 The other direction: long context is the thing we cannot otherwise buy

KDA's saved scratch scales with the sequence — **17 fresh tensors / 248 MB of saved
scratch per iteration at 512×10** (AGENTS §2.2) — so running the recurrence at byte
granularity over 3-4 KB windows is exactly what our 16 GB card cannot afford. Patching
is the one mechanism in the ladder that extends the model's byte horizon per loop
position (512 B → 2-4 KB at ps 4-8) **at constant VRAM**. That is a capability claim the
papers support directly: BLT trains 8k-16k **byte** contexts (§4.3).

### 2.3 Small-scale evidence exists, and it is pro-patching — with one correct reading

BLT's own scaling data says the patch advantage appears only **past the compute-optimal
budget** (§5.3) — at 9M params on a 46 GB corpus we are not obviously past it. But
SpaceByte's Table 1 **is** a small-scale equal-compute result, and it says dynamic
boundary-aligned patching **beats byte-level at the same training FLOPs**
(1.009 vs 1.138 PG-19). The two reconcile: what wins at small scale is
**boundary-aligned dynamic patching**, not granularity per se — fixed stride loses to
both (1.112 vs 1.009; BLT Fig. 6 has strided worst of all schemes). Any cost/benefit we
quote for a strided first version must be discounted by exactly that gap.

One more non-transfer: SpaceByte's byte-level control is a **softmax** transformer whose
quadratic term and weak locality punish it; our LoopBlock is not that control. The fair
reading is "patching can win at equal compute at our scale", not "we will gain 0.13
BPB".

### 2.4 We already run one of BLT's components

BLT's local encoder adds **hash n-gram embeddings, orders 3-8, polynomial hash mod table
size** (§3.2.1). Our Engram is **FNV-hashed n-gram tables, orders 2/3/4**, fed per byte
position. Same mechanism family, same placement (byte-level, feeding the small
model under the global one). Under any transplant the Engram **survives unchanged** —
it does not care that positions are ps apart if its ids are sampled at patch boundaries
(the coverage loss, 1/ps of byte n-grams seen per step, is part of the experiment, not a
defect). This is also an external validation of the Engram arm's design.

---

## 3. Landing options on our architecture

`DormouseModel = Embedding → LoopBlock → RMSNorm → lm_head` (`model.rs:263-395`);
the stream hands `batch × seq_len` raw bytes and one FNV pass per batch
(`dormouse-data/src/lib.rs:52-53`); `seq_len` is **bytes** (`schema.rs:213-216`).

| # | variant | what changes | length effect | loss | ckpt era | est. LOC | verdict |
|---|---|---|---|---|---|---|---|
| a | **Full BLT transplant** (byte encoder + cross-attn pool + latent loop + byte decoder) | 3 new modules incl. softmax sequence attention — **no such module exists in the vendor library** (`burn-attnres` attends over the depth axis, not sequence); hand-rolled attention = fresh autodiff surface, the `8fa5d4c` silent-leaf class | n → n/ps everywhere | byte CE via decoder | new era | 800-1200 + gates | **reject as first move**: max surface, and its extra machinery (cross-attn pooling, 3-layer local stacks) exists to serve 8B-scale latent models we do not have |
| b | **Fixed-stride patch arm** (recommended P1): reshape `[b,t,d] → [b,t/ps,ps·d]` + one `LinearLike` pools; loop runs at patch positions; one `LinearLike` upsamples back to byte positions; existing RMSNorm + `lm_head` + **byte CE unchanged** | patch pool + upsample + L_Rec retarget + Engram id subsampling | loop: 512 → 512/ps; bytes/step ×ps at fixed loop shape | **unchanged** (honest mean byte CE, same BPB units as every run on record) | none if `patch_stride=1` default (byte path = today's code exactly); patch runs are new-era by the existing config-diff refusal (ADR-0005/0021) — no code needed | **300-400 + gates** | **the minimal test**; reshape/Linear/gather only — no new kernel or autodiff-op class |
| c | **SpaceByte rule** (P2, conditional on b): spacelike-byte boundaries instead of stride | the rule is ~30 LOC of byte classification in `dormouse-data`; **uneven patches break the dense `[b,t]` layout** → needs BLT-style patch packing/padding (§4.5) | n → n/avg_ps, boundaries aligned | unchanged | same gating story | ~30 + **150-250 packing** | only after b shows the win AND the fixed-vs-dynamic gap (~0.10 BPB, SpaceByte Table 1) is worth 150 LOC |
| d | **Entropy patching** (P3, conditional on c): small byte-LM patcher | entropy model (BLT: diminishing returns >50M params — **5× our whole model, absurd at our scale**; a 1-5M byte-LM is the only defensible size, and BLT's Fig. 8 shows small ones work worse) + incremental-patching property at decode | dynamic | unchanged | same | +500-800 | **not in the ladder** unless c lands and dynamic-vs-rule measurably matters |
| e | **Patching for batching only** (the brief's variant б) | nothing — we run a dense stream, `next_batch` is a memcpy, **no padding exists to save** (`lib.rs:52-53`); BLT's patch packing exists because *its* dynamic patches make batches ragged, which we would not have under fixed stride | none | none | none | 0 | **reject**: solves a problem we do not have |
| f | **Hybrid: KDA stays byte-level, experts coarsen** | keeps the recurrence at byte granularity over 512·ps bytes | n stays | unchanged | — | — | **reject**: KDA saved scratch scales with n (248 MB/iter at 512×10, AGENTS §2.2) → ~1.5 GB/iter at 3 KB — VRAM blow-up for a component that is O(n) cheap anyway |

### The b-seam, file by file

- `dormouse-data`: **nothing changes.** Windows stay byte-defined, so the eval-window
  formula (`eval_batches × batch × seq_len` bytes, `train/src/lib.rs:1417`) and §2.6
  comparability survive; hashes are already computed per byte for the Engram.
- `model.rs:263` `forward_with_hidden`: pool `x` after the embedding
  (`[b,t,d] → [b,t/ps,d]`); decode before the final norm+head so `model.rs:380`'s
  per-byte logits and byte CE are untouched.
- `loop_block.rs:543` `forward_full_state`: body unchanged (controller `:782`, KDA
  `:832`, Engram `:864/:883` all operate on whatever `t` they are given). Two retargets:
  the **L_Rec gather** (`:593`) reads byte indices — under patching, per-iteration CE
  becomes next-**patch**-first-byte prediction (the exact byte SpaceByte argues is the
  hard one — its global blocks exist to predict it), a one-line index change; the Engram
  `hashed_ids` are subsampled at stride boundaries (~10 LOC in the trainer's hash pass).
- `config/schema.rs`: `patch_stride: usize = 1`; `validate` refuses a stride that does
  not divide `max_seq_len`, and `0` (the `dspark_stride` class of foot-gun).
- Gates per §1.1/§1.2: `probe::PATCH` counter + `preset_exec` count assertion + a field
  on the eval line (the `engram=` precedent, §3.2); the arm is otherwise config, not an
  A-B flag.

### Checkpoint compatibility (the brief's question, answered precisely)

- **What survives:** TSCT factor quantisation and the fp32-master scheme
  (`LinearLike` untouched); the Muon+ routing policy (new Linears join through the
  existing loud `validate_routing`, no silent AdamW fallback); the Engram tables and
  sidecar format; `lm_head`, embedding, final norm.
- **What does not:** the loop weights become weights over patch states — dimensionally
  identical, semantically a different network. Old checkpoints are **not** usable as
  inits for patch runs; do not pretend otherwise.
- **The era break is optional and we should not take it:** `patch_stride = 1` default
  keeps the byte path bit-identical to today's forward, so every existing checkpoint
  resumes, and flipping the flag mid-resume is refused by the existing config-snapshot
  hard error (ADR-0005/ADR-0021) — which is the correct behaviour, already implemented.

---

## 4. Cost and risk

| item | cost / risk |
|---|---|
| code (option b) | ~300-400 LOC + counters/gates; no new kernels; no new autodiff-op classes (reshape/`LinearLike`/gather — the safe class; the `8fa5d4c` risk lives in hand-rolled ops, which b avoids) |
| P0 measurement | **zero code**, one release binary, ~15-30 min quiet-GPU time |
| A/B cost | rung-1 filter ≈ 333 steps × 3 seeds (equal-bytes, §5); confirm ≈ 2k steps × 3 seeds × {ps 4, ps 6}; per-step cost unknown until the control re-baseline lands (AGENTS §3.3) |
| **R1 (real research risk)** | a **weight-shared** loop refined over patch states is tested nowhere in the three papers — they stack *distinct* layers over patches; whether our recurrence transfers to coarse granularity is the open question the prototype answers |
| R2 | quality at 9M params: BLT says fine granularity wins small budgets; SpaceByte's small-scale table says boundary-aligned patching wins equal-compute. Strided (our P1) sits between — expect less than SpaceByte's −0.13 BPB |
| R3 | launch-bound floor may swallow the speed win entirely (only ~80 % of the step scales at all; retr 52-64 ms never shrinks) — P0 prices this before any code exists |
| R4 | dynamic patching (c/d) drags in ragged-batch packing — the hidden cost that makes fixed stride the right first version |
| runs on record | **nothing is invalidated**: the byte path is preserved bit-identical at the default; patch runs write their own configs and refuse cross-era resumes loudly |

---

## 5. Verdict

**Is there a place in the ladder: yes — a narrow, gated one, priced by P0 first.**

1. **P0 (now, zero code): the seq-len mould.** Same release binary, quiet card, warm
   steps (`--timers`, read steps ≥ 50 — the step-0 lesson, AGENTS §3.1), aux off,
   `--no-engram`, batch 10 fixed, `--seq-len` ∈ {512, 128, 85}: three readings give the
   position-share of a warm step directly, and bytes/s at equal step time gives the
   throughput conversion. This is the whole lane's price tag. **If the position-share
   comes back < 15 %, the lane demotes to "park"** — patching would be buying a
   launch-bound box a smaller launch-bound box.
2. **P1 (option b) enters the A/B queue after the queue's first three arms** — control
   re-baseline (the §3.4 precondition: an arm cannot be judged against a control whose
   attention arm was frozen), pure-CE, depth-2-vs-4 — **and before the ×10
   capacity/official-100k commitment.** Rationale: (i) if P1 wins, every production run
   after it wants to be *born patched* — training ×10 unpatched first means paying the
   full byte length for the entire run and then re-entering a new era anyway; (ii) it is
   the only ladder item that extends effective context per step at constant VRAM
   (§2.2); (iii) it is small and loss-compatible, and the §3.4 queue does not contain
   anything else that changes the data rate.
3. **P2 (SpaceByte rule) and P3 (small entropy model) are follow-ups, conditional** on
   P1 clearing its equal-bytes filter — SpaceByte's fixed-vs-dynamic gap (~0.10 BPB at
   their scale) is the budget P2 has to beat 150 LOC of packing with. **A BLT-scale
   entropy oracle (50-100M) is never justified at our scale** — it is 5-10× the model it
   patches.
4. **Do not double-build the SpaceByte mechanism:** per-position adaptive *compute*
   already exists as the MoR arm (un-A/B'd, its own ADR-0013 objection). Patching is
   adaptive *granularity*; the two compose but are separate arms, and neither inherits
   evidence from the other.

**Expected win, stated as bands (§2):** bytes/step ×3-5 at ~1.4-1.8× step time (speed
band, P0-gated); effective loop context 512 B → 2-4 KB (capability band, the strongest
paper-backed claim); same-compute BPB: plausibly −0.05…−0.15 vs our own control at equal
*bytes* if SpaceByte transfers even partially — with the explicit caveat that a strided
arm underperforms its dynamic sibling by construction (Table 1: 1.112 vs 1.009). **Not
expected:** movement on the 5-gram bar (2.57-2.91 BPB) — that is a locality/memory
result, and it belongs to the Engram.

**The minimal prototype (P1's rung-1, one paragraph for the implementer):**
`patch_stride: usize = 1` in the schema; at ps ∈ {4, 6}: pool = reshape + `LinearLike`
(`ps·d → d`), loop unchanged, L_Rec gathers next-patch-first-byte indices, Engram ids
subsampled at stride boundaries, upsample = `LinearLike` (`d → ps·d`) + reshape to
`[b,t,d]`, then the untouched norm + `lm_head` + mean byte CE. `probe::PATCH` counts the
arm; `preset_exec` asserts it; the eval line prints `patch=<ps>`. Rung 1 is 333 steps ×
3 seeds at ps 6 vs the re-baselined control at equal **bytes** (2k steps × 512 B):
**pass = within ~0.2 BPB of control** (granularity held; SpaceByte says dynamic can then
recover the rest), **fail = > 0.5 BPB behind** (the weight-shared loop does not transfer
to coarse states — stop, delete, and the lane dies for the cost of ~350 LOC).

---

## Provenance

| file | what it pins |
|---|---|
| `docs/papers/2412.09871-blt.pdf` | BLT: mechanics (§2-3), entropy model (§4.2), scaling + crossover (§5), patching ablation (§7, Fig. 6, Table 6) |
| `docs/papers/2305.07185-megabyte.pdf` | MegaByte: fixed-P architecture and cost model O(T²/P + TP) (§2) |
| `docs/papers/2404.14408-spacebyte.pdf` | SpaceByte: spacelike rule (§2), small-scale equal-compute Table 1 (§4), 10× gap claim (§1) |

All three rows added to `docs/papers/provenance.tsv` (sha256 at fetch time, 2026-10-02).
Our-box numbers: `AGENTS.md` §3.1 (batch ladder, utilisation, retr cost, eval window) and
§2.2 (KDA scratch); code refs: `crates/dormouse-core/src/model.rs:263,380`,
`crates/dormouse-core/src/loop_block.rs:543,593,782,832,883`,
`crates/dormouse-core/src/config/schema.rs:124-125,213-216`,
`crates/dormouse-data/src/lib.rs:52-53`, `crates/dormouse-train/src/lib.rs:1417`.
Derived arithmetic (the 20 % floor, the −25…−45 % band) is labelled derived and is
settled by P0, not by this document.
