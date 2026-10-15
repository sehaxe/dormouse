# 1B build tech verification — every bet against the published record (2026-10-15)

Method: arXiv MCP only for citations. Two corrections to the brief, up front, because they
change what is cited. **BLT is arXiv:2412.09871, not 2502.00232** (2502.00232 is a hyperspectral
oil/water classification paper); all BLT claims below cite 2412.09871 or 2605.08044. And
**arXiv:2601.16531 is not an "iso-parameter slot-count law"** — it is *A Collision-Free Hot-Tier
Extension for Engram-Style Conditional Memory* (Engram-Nine), an MPHF hot tier vs. hashed cold
tier, whose headline result is the opposite of a slot-count law: at strict iso-parameter,
removing collisions does not reliably help. Web sources (flash-linear-attention GitHub, Keller
Jordan's Muon posts) cover library status that has no arXiv equivalent and are labelled `[web]`.
Nothing else is cited without an ID.

## Verdict table

| tech bet | status | strongest evidence | action for the 1B plan |
|---|---|---|---|
| Muon + AdamW-hybrid routing, ns_steps tuned | **confirmed** | 2610.17429, 2509.23106 | keep Muon+ for 2D; keep 1D/embeddings/head/router on AdamW |
| Muon state in bf16 / 8-bit + stochastic rounding | **confirmed at 8-bit, open at 4-bit** | 2509.23106, 2605.11396 | ship 8-bit blockwise Muon state before 4-bit; MuonQ pre-normalization is required, not optional |
| 4-bit Muon ("MuonQ") moments on a consumer Blackwell card | **confirmed, measured on an RTX 5090** | 2607.04422 | adopt the Full-Stack FP4 Root recipe as reference; keep softmax in BF16 |
| Musec spectral clipping instead of qk-clip | **confirmed as an alternative** | 2609.11655 | A/B Musec vs. qk-clip as the spike protocol; Musec is architecture-agnostic |
| ByteFlow coding-rate (logdet) patch criterion | **open — unpublished, and its entropy sibling is partly refuted** | 2610.11790, 2412.09871 | coding-rate patching is novel; it must earn its win against boundary dependence, not entropy |
| Engram = 20–25% of total params | **refuted as stated** | 2608.30320 | no 20–25% number exists in the Qwen3.8 paper; derive the allocation from our own BPB curve |
| Engram n-gram tables in host RAM + per-batch prefetch | **confirmed** | 2608.30320, 2610.20456, 2502.01637 | keep and extend `--engram-ram`; move tables to Adam, weight decay off |
| Hash-slot collisions are the bottleneck (slot-count law) | **refuted** | 2601.16531 | do not spend hardware on collision-free hot tiers; collisions may be regularizing |
| Rank-64 low-rank experts, topk=2, ~500M | **open** | 2609.36301, 2604.04230 | topk=2 stays default; no published rank/k sweep at our scale — this is an A/B |
| KDA decay form `A_h=-3 / b_alpha=+1.0` (K3 sigmoid) | **refuted as a literature default; keep as a measured local choice** | 2510.26692, 2607.24653 | not Moonshot's and not FLA's; label it "ours, measured" and re-test against FLA's init |
| `g_min=-5` | **confirmed** | 2510.26692 + `[web]` FLA `lower_bound` | matches FLA's documented `-5` gate floor; keep |
| KDA recurrent state dtype | **fp32 confirmed for inference state; bf16 training open** | 2608.27513 | keep §2.1's fp32-state trajectory; do not bf16 the state on a loss match |
| 1B trained on ~10B bytes/params total (see the corrected label below) | **open** | 2608.07222, 2502.02737, 2404.06395 | treat as a distinct regime; fit our own law rather than assuming Chinchilla |
| bf16 GEMM on sm_120 | **confirmed fine here (refutes the old fear)** | §2.1 2026-10-10 table + 2607.04422 | bf16 compute is not the risk on 0.22; the fp32-GEMM-fallback premise is dead |
| fp8/FP4 + bf16 optimizer state on sm_120 | **open** | 2607.04422, 2608.27513 | the real open risk; overfit degradation and state dtype need our own measurement |
| CUDA-graph capture of the launch-bound stage | **confirmed direction, failure modes known** | 2609.23536, 2501.09398 | already 5/5 green on this GPU; capture one stage, not the whole step |

## 1. Sparse / top-k MoE recipe at ~1B, on one card

**Router stability.** 2604.04230 tracks a congestion-coefficient model of routing across 20
OLMoE-1B-7B checkpoints (and 6 OpenMoE-8B) and finds three phases: a *surge* where the router
learns to balance (γ_eff 14 → 36–39, peaking ~step 30K–40K), *stabilization* where experts
specialize under steady balance (steps 100K–400K), and *relaxation* where balance is traded for
quality. The aux-loss weight right at step 30K is wrong at step 300K: a fixed load-balancing
coefficient across a 1B run is a real risk, and OLMoE-1B-7B is our closest published scale match.

**Which balancer.** 2408.15664 (Loss-Free Balancing) updates an expert-wise bias before the
top-K decision instead of adding an aux loss, because a large aux loss injects interference
gradients. 2511.08972 replaces the aux objectives with selective Sinkhorn routing, 2512.23447
couples routers to experts with an ERC loss, and 2605.12476 shows MoE-stack aux losses can
*reduce* specialization. The field has converged on "stop using a big aux loss", not on one fix.

**Z-loss.** 2609.16179 argues z-loss must be read from the backward side: the logit-space
gradient it injects is transported through the architecture, so nearly identical forward Z-loss
values can coexist with distinct logit-space gradients and distinct parameter updates. With a
tied embedding / shared output head and a fused loss, a z-loss weight validated elsewhere is not
evidence.

**k=1 vs k=2 and rank-64 — open.** No paper compares k=1 vs k=2 for rank-64 low-rank experts at
~1B. The closest structural result is 2609.36301, which proves router *rank* r = O(log M)
suffices (and is necessary up to precision factors) when the active-expert count is fixed, and
preserves load balance — relevant because we are in the "more, smaller experts" regime where the
linear router's Θ(Mh) cost dominates. SPECULATION: based on 2609.36301 and 2604.04230's
congestion dynamics, I assume topk=2 with rank-64 experts stays viable but its balancing
coefficient must follow the three-phase curve rather than being fixed.

**Single-GPU dispatch.** 2604.20156 is the one for our box: once a model outgrows GPU memory,
per-token expert churn "can render optimizations like offloading and pre-fetching ineffective",
and the fix is a learned controller deciding *when* to switch expert sets; 2512.17073 reaches the
same place from bandwidth, with token-level routing making inference I/O-bound. At ~500M of
experts on 16 GB with everything resident the offload literature does not bind, but the churn
argument does bear on how often we revisit the router decision.

## 2. Muon, state of the art late 2026

**ns_iters.** `[web]` Keller Jordan's Muon post and the kellerjordan/muon repo give ns_steps=5
originally and 6 in the current repo, with FLOP overhead bounded by Tm/B (<1% at NanoGPT scale,
~0.5% at 405B). **No arXiv paper** gives an authoritative ns_iters at 1B; §3.5 already carries
the Qwen3.8 choice of ns_steps=8 as a stability pick. `[web]` the current modded-nanogpt config
also uses a *momentum warmup* (0.85 → 0.95 over the first 300 steps) — a stability measure we do
not have.

**Low-bit Muon state.** 2509.23106 shows 8-bit blockwise Muon reaches parity with fp Muon in
Chinchilla-optimal runs **up to 2.7B**, with up to 62% optimizer-memory reduction — validating
bf16/8-bit Muon moments at exactly our scale. But 2605.11396 explains why low-bit Muon is harder
than low-bit Adam: orthogonalization discards singular magnitudes and keeps only directions, so
even small quantization errors in singular vector directions are amplified in the update, and they
add a pre-quantization normalization to fix it. An 8-bit Muon state without MuonQ-style
normalization is not supported.

**4-bit Muon on this card.** 2607.04422 (Full-Stack FP4) is the most directly relevant paper in
this sweep for our hardware: **native four-block measurements on one RTX 5090** — 2.50–2.83×
Root (=Muon-family) speedups over optimized BF16 and 37.9–42.5% lower AdamW peak memory — and at
3B/64B tokens a 0.838% loss gap (BF16 2.267 vs FP4 2.286), with softmax-sensitive attention kept
in BF16 and a LoRA-SVD-protected BF16 projection subspace. Reservations: it is Root, not Muon
proper, and the loss number is 3B, not 1B.

**Follow-ups.** 2610.17429 (Monarch-Muon, 2026-10-12) is the freshest — 2.68 vs. Monarch-AdamW's
2.77 on a 1B MoE, and at 6.89B peak memory 68.9 → 52.9 GB and the step 1288 → 304 ms — but it is
tied to Monarch weight structure, which we do not have; 2510.05491 (NorMuon) is the standalone
neuron-wise-adaptive-LR variant.

**qk-clip ↔ softcapping.** 2609.11655 is the relevant new result: Muon's spectral flattening
causes loss spikes and unbounded weight growth; existing fixes (weight or attention-logit
clipping) require architecture-specific modifications and do not directly address instability
across all model components; Musec replaces flattening with spectral clipping — an optimizer-level,
architecture-agnostic mechanism with convergence guarantees in nonconvex nonsmooth stochastic
optimization. A direct competitor to qk-clip, and cheaper to adopt than one wired into every
attention arm.

**Polar Express is already resolved locally.** §3.7 records that both Newton–Schulz coefficient
sets in this tree come from **arXiv:2602.21545v3**: `burn-spectral` hardcodes the PolarExpress
triple (1.875, −1.25, 0.375) from Appendix D.3, `burn-muon-plus` uses the Jordan triple
(3.4445, −4.7750, 2.0315) from Appendix D.1 — both citations correct, different polynomials —
and running the authors' own code (`NoahAmsel/PolarExpress` @ `71cc379`) showed the fixed triple
*beats* the schedule at 3 iterations at every condition number tested, so §3.5's "skip the
report's Polar Express schedule" is already backed by measurement. The Adan fallback cites
arXiv:2208.06677 and ships in burn-optim.

## 3. Byte-level LM and byte-patching

**Entropy patching is published and partly refuted.** 2412.09871 (BLT) segments patches "based
on the entropy of the next byte", giving the first FLOP-controlled byte-level scaling study up
to 8B/4T bytes. But 2610.11790 (2026-10-08) shows the rule has a systematic blind spot:
positions whose *type* is predictable but whose *value* must be computed — e.g. the number
after "=" in a worked math solution. At a 10% patch budget on BLT-1B, entropy puts a patch start
on 16% of computed results and gets 19.0% exactly right; a hard boundary after every "=" gets
51.8%; entropy **plus** a boundary-dependence signal gets 67.1% (default 26% layout: 76.8%).
From scratch at 1M/12M/50M parameters, boundary dependence beats entropy on final answers by
−1.6, +10.1 and +19.8 points; at 50M it is 35.9% vs. 13.9% of computed results. Scratchpad
Patching's (2605.09630) entropy trigger is likewise indistinguishable from random scratchpads
(5.6% vs. 5.9%, 5 seeds) while answer-start scratchpads give 38.1%. Their signal — *boundary
dependence*, the rise in the model's own loss when a patch start is removed, measured per
two-byte context — is label-free.

**What this means for ByteFlow's coding-rate criterion.** The most important negative result in
the sweep. Our R_eps logdet coding-rate gain is a *local predictive-information* criterion in
the same family as next-byte entropy, and 2610.11790 is direct evidence that this family misses
computed values while over-covering low-entropy-but-uninformative bytes ("copies and lookups
gain little"). Meanwhile `coding rate` + `logdet` returns **zero arXiv hits**, so
patch-by-coding-rate-gain is unpublished — ours to validate, and not validated by BLT's success,
because BLT's criterion is entropy, and entropy is what is now attacked. Neighbours: 2401.13660 (MambaByte),
2305.07185 (MEGABYTE), 2605.08044 (Fast BLT), 2610.12495 (Kairos), 2608.03599 (boundaries), plus
2605.09630's *patch lag* failure mode — until a patch is fully observed, predictions inside it use
a stale previous-patch representation, a cost our recurrent core pays.

## 4. Hashed n-gram memory at scale

**The "20–25% of total params" allocation is not in the literature.** 2608.30320 (the
Qwen3.8-Flash-Next architecture paper) states it plainly: 125B parameters, 6B activated per
token, **"additional 51B parameters of n-gram embedding tables held off the accelerator."** That
is 40.8% of the base model, or 29% if the tables count on top — neither is 20–25%, so
**refuted as stated**. The same paper also confirms §3.5 item 5: a GDN/global-attention hybrid
with one full-attention layer in every four, replaced at continued-pretraining time by QSA with a
micro-block compressed indexer.

**Host-RAM offload with per-batch prefetch: confirmed, twice.** 2608.30320 holds the 51B tables
off-accelerator. 2610.20456 (The COUNT, 2026-10-14) makes the argument explicitly: n-gram
embeddings scale memory separately from compute, because each pass only processes individual
embedding rows; existing methods train them from scratch, which needs expensive GPU memory for
fast reading and writing; and host RAM is the cheap underused tier they scale onto. 2502.01637
(SCONE) is the earlier statement: contextualized n-gram embeddings learned by a separate model,
precomputed and stored off-accelerator, inference lookup at minimal impact on latency.

**The slot-count law is a mirage.** 2601.16531 (Engram-Nine) runs a strict iso-parameter
comparison of an MPHF collision-free hot tier against the original multi-head hashed lookup and
finds the collision-free design **does not consistently improve validation loss**. It then
decomposes per-token loss into hot/cold contributions and finds a consistent "hot-to-cold
advantage flip" during training — hot positions start lower-loss and are eventually overtaken —
with collision-free configurations flipping *earlier*, "suggesting that collisions act as
implicit regularization."

**kNN-LM / FwPKM lineage.** 2601.00671 (FwPKM) is the live version: sparse fast-weight
product-key memory updating only activated slots via chunk-level gradient descent on a local
memory-rewrite objective — TTT-style updates at fixed per-token compute, episodic memory beside a
semantic backbone. Cautionary cites: 2305.14625 (kNN-LM does not improve open-ended generation)
and 2301.02828 (why kNN-LMs work — mostly knowledge, not novel text). 2606.08347 (Tensorizing
Engram) and 2607.29065 (Tokenizer-Agnostic Engram) both push against per-order separate hash
tables; the latter treats an n-gram as "a method to sample potentially useful byte sequences",
the byte-level framing our Engram needs.

**For the 1B build:** keep `--engram-ram`, move the tables to Adam with weight decay disabled
(2608.30320's rule, §3.5 item 1), and do not buy collision-freedom. Derive the allocation by
sweeping `engram_rows` against BPB — 2601.16531's hot-to-cold flip says where they stop paying
depends on training progress, so it is a curve, not a constant.

## 5. Gated-delta-rule / KDA linear attention

**The architecture bet is confirmed.** 2510.26692 (Kimi Linear) introduces KDA, extending Gated
DeltaNet with a finer-grained gating mechanism so the finite-state RNN memory is used more
effectively, with a chunkwise algorithm on a specialized DPLR transition that beats full attention
under fair comparisons. 2607.24653 (Kimi K3) is the frontier build on exactly this: KDA plus
Attention Residuals plus Stable LatentMoE (16 of 896 routed experts). The KDA core, the
one-in-four full-attention hybrid and AttnRes are frontier practice, not a boutique choice.

**FLA status (`[web]`).** KDA in FLA since 2025-10; since then Context Parallel for KDA/GDN
(2026-03), GPT-OSS-style attention sinks (2026-04), Preconditioned GDN/PKDA (2026-06), FlashQLA
for GDN (2026-07), GDN-2, Wall, Mamba-3, PowGLU, Parallax; and a **move from a Triton-only stack
to multi-backend** — FlashKDA as a KDA backend, TileLang for GDN/KDA/parallel-attention. FLA also
ships a fused **AttnRes** operator, reusable for our per-iteration router blend.

**Our decay knobs vs. the literature.** FLA's KDA layer computes
`g = -exp(A_log) * softplus(f_proj(x) + dt_bias)` — **softplus, not sigmoid** — with `A_log`
initialized uniform in log[1,16] (exp(A) ∈ [1,16], *not* 0) and `dt_bias` derived from inv_dt
over dt ∈ [0.001, 0.1]. Our `A_h = -3 / b_alpha = +1.0` under a K3 sigmoid is therefore neither
Moonshot's nor FLA's (matching §3.5 item 5 and §2.3), and this sweep finds nothing that redeems
it as a default. Conversely our `g_min = -5` **is** confirmed: FLA's `safe_gate`/`lower_bound`
with `lower_bound = -5` clamps the log-space gate to a minimum per-step decay exp(g) ≈ 0.0067,
documented as having negligible impact on quality. FLA also offers `allow_neg_eigval` (β scaled
to [0,2)), in-kernel q/k L2 normalization, in-kernel gate and β-sigmoid, and chunk sizes
restricted to 16/32/64 (BT=64, BC=16) — all matching burn-kda.

**Numerical stability and dtype.** 2608.27513 (DAMP) is the first study of post-training
recurrent-state quantization for GDN and KDA: states are "commonly stored in FP32", and uniform
INT8/FP8 degrade complex-reasoning accuracy while INT4/NVFP4 collapse it to near zero. That is
inference quantization, not training, so it does not refute bf16 states — but it is the
strongest available signal that the recurrent state is the fragile component, and it argues for
keeping §2.1's fp32-state trajectory until a training-time study says otherwise. Also sobering:
2608.02689 converts 21 of 28 layers of Qwen3-0.6B-Base to KDA **on a single consumer-grade GPU
budget**, drives perplexity close via KL distillation, and still ends with multiple-choice
accuracy at 25–29% against the teacher's 50.6% — an "interface injury" (it predicts option
labels, "A" 81% of the time) that loss matching does not reveal, so any KDA arm of ours must be
judged on more than BPB. Related: 2604.21100 (Preconditioned DeltaNet), 2607.11897 (Semidirect
Fourier Delta Attention), 2610.02816 (Gated Slot Attention-2).

## 6. An over-trained ~1B

LABEL CORRECTED 2026-10-15 (flagged by the data-strategy doc): this section's
"10B tokens per 1B params" reads 9.3 bytes/param against Chinchilla's
~20 — BELOW the optimum on the COUNTED 1.07B, not over-trained; only against
the 57M ACTIVE params does it read 176× (the byte level makes the two
claims look alike). The over-training question is therefore "do we run 1-4
epochs of what we have, and does that buy", not "10× like frontier labs".
The regime remains distinct; fit our own law on our own lattice.

**No Chinchilla head-to-head at 1B / 10B tokens exists in what I found.** What exists is
agreement that Chinchilla fits badly at the overtraining extreme. 2608.07222 (Skaling) states
standard forms systematically under- and overestimate loss at data-scarce and overtraining
extremes, because they assume model size and data affect loss independently; the fix is a single
capacity–data interaction exponent cutting MAPE by 1.5–3× using ~10× less compute on a sparse
low-compute grid. 2603.22339 reports systematic biases in IsoFLOP parabola fits (Approach 2);
2406.19146 and 2509.23963 are the earlier attempts to resolve compute-optimal discrepancies. At
10B tokens per 1B params we are squarely in the regime where the fitted exponent is known to be
biased, so the honest move is to fit our own law on our own lattice.

**The two best small-model references.** 2502.02737 (SmolLM2) overtrains a 1.7B model on ~11T
tokens with a multi-stage web + math/code/instruction mix, deriving stage decisions from
small-scale ablations; 2404.06395 (MiniCPM) trains 1.2B/2.4B with "model wind tunnel experiments
for stable and optimal scaling" and uses WSD. Together: multi-stage data + wind-tunnel
hyperparameters + WSD, not a Chinchilla token count.

**Schedule.** 2410.05192 explains WSD's non-traditional curve — loss elevated through the stable
phase, then a sharp drop in decay, reading as a "river valley" loss landscape; 2508.01483
analyses the cooldown stage and 2607.10959 (WSqD) offers a horizon-free variant. WSD is the right
default for a budget we may change without retuning, which is our situation. `[web]` the current
1B-class speedrun uses cooldown_frac = 0.7, aggressive but consistent.

**Loss-spike protocols actually used at 0.5–2B.** Three usable ones from this sweep: 2607.21005
identifies a concrete mechanism for spikes induced by normalization plus weight decay
(weight-norm criticality); 2606.16768 (Taming Curvature) supplies a fast online estimator of the
largest preconditioned Hessian eigenvalue via warm-started power iteration, motivated by exactly
"billion-parameter Transformers are often brittle, with transient loss spikes and divergence";
and 2609.11655 (Musec) offers the optimizer-level spectral clip; older anchors are 2312.16903
(Spike No More), 2305.12133, 2506.04805 (adaptive preconditioners *trigger* Adam spikes),
2410.05052 (reparameterized init). Qwen3.8's zero-spike-at-4×-LR claim (§3.5 item 2) is
attributed to Gated Residual, not to any of these. Reproducibility footnote: 2609.17380 (OPEN-1B)
argues no released model is provably reproducible because floating-point arithmetic is
non-associative and framework determinism does not carry across hardware — read before making
cross-box loss comparisons at 1B, since §3.1 already forbids them.

## 7. bf16 stability on consumer Blackwell (sm_120)

**bf16 GEMM is not a problem on this box — that premise is already dead locally.** §2.1
(measured 2026-10-10, one dtype per process, release, quiet card) gives bf16 matmul at **44.06
TFLOP/s against fp32's 20.03**, at the cuBLAS ceiling, with both bf16 parity tests in
`backend-parity` passing. The pre.4 claim that every bf16 run is slower than fp32 is explicitly
marked FALSE on 0.22, so the assumption that a 1B plan needs an fp32 GEMM fallback is refuted by
our own measurement, not by any paper. The strongest arXiv corroboration is 2607.04422, whose
numbers are **native four-block measurements on one RTX 5090** (2.50–2.83× Root speedup over
optimized BF16, 37.9–42.5% lower peak memory, 0.838% loss gap at 3B/64B tokens) and whose recipe
is about which ops may leave BF16: softmax-sensitive attention stays BF16, and a LoRA-SVD
projection subspace recovers about half the linear-only loss gap. 2605.16617 supplies the
numerical premise — BF16 and FP32 share dynamic range and BF16 accumulates into FP32 accumulators
at full speed — and documents BF16 characteristics specific to the Blackwell line.

**What stays open is the rest of the stack, and the local rules are the real constraint.** §2.2
(live, not stale) records four measured sm_120 hazards that survive the bf16 GEMM fix: bf16
**logits** NaN, so the final norm and `lm_head` must run fp32 even in BF16 mode, as must
loop-internal `L_Rec` logits; mixed-dtype ops (bf16 activations × fp32 weights, bf16 + f32 adds)
NaN, giving the rule "cast to fp32 before every Linear, compute in fp32, cast residual/GR writes
back"; the Engram kernels and the KDA fused chunked kernel are f32-only and fall back to tensor
ops under `--bf16`; and dynamically slicing a 4D autodiff tensor crashes with
`CUDA_ERROR_ILLEGAL_ADDRESS`. §2.1 adds that fp8 quant (the sm_120 default) degrades at overfit,
and 2608.27513 (§5) warns INT8/FP8 degrade and INT4/NVFP4 collapse for recurrent state. **No
arXiv paper found** on bf16 GEMM or optimizer-state defects specific to RTX 50-series / sm_120.
Engineering-level evidence only (`[web]`, FLA): SM120 mis-detected as non-Blackwell so the wrong
code path ran (FLA PR #940), large-offset pointer arithmetic on Blackwell (#911), a missing
default global_scratch allocator (#825), and int32 overflow in conv/varlen pointer arithmetic
(#783 and #803). GitHub references, **not arXiv**, cited only as evidence the failure class exists
here.

## 8. Launch-overhead elimination for a hand-rolled loop

**Capture is confirmed *on this GPU*, and the one-stage design with it.** §3.1: the workload is
launch-bound (mean GPU utilisation 13.3%, 79% of samples ≤5%), and CUDA graph capture/replay is
measured **5/5 green here** in `vendor/cubecl-fix/cubecl-cuda/tests/graph.rs` (capture+replay,
input rewrite, intermediate recycling, pool-growth refusal, many-launches with dynamic metadata),
client API `graph_prepare` → `start_capture` → `stop_capture` → `Graph::replay`. The recorded
blocker is exactly what forces one stage rather than the whole step: **a capture window refuses
stream reads, syncs and handle writes, and a training step must do all three** (write parameters,
read the loss scalar for the NaN firewall and the log line). The literature agrees.

**The failure modes are enumerated.** 2609.23536 (QEffect) is the paper to read before writing
the capture. CUDA Graphs replay tensor operations over static virtual addresses, but low-precision
pipeline training continuously alters the scaling states, microbatches and deferred backward
tasks those fixed addresses represent; split-backward schedules (1F1B, Zero-Bubble) decouple
input- and weight-gradient computation and break LIFO lifecycles; and standard dataflow graphs
"cannot inform the runtime of hidden numerical updates, non-LIFO work ownership, or cache
validity, leading to silent cross-stream data corruption." The fix is an explicit state and
resource contract runtime. This maps exactly onto the two local findings above: the pool-growth
refusal in `graph.rs` is the "cache validity" failure, and the read/sync/handle-write refusal is
the "hidden numerical updates / non-LIFO ownership" failure. **What breaks a captured graph is
not shapes, it is state and ownership the capture cannot see.**

**Practical capture guidance.** 2501.09398 (Kernel Batching with CUDA Graphs) is the closest
match to our one-stage design: group kernel launches into *iteration batches*, then unroll those
batches into a CUDA Graph, so an iterative application gets graph replay per iteration.
2603.12465 (TaxBreak) supplies overhead-decomposition methodology for deciding which stages are
worth capturing — relevant because only some stages are launch-bound, and §2.2 attributes the
addressable cost to the KDA arm's 17 fresh tensors / ~1 GB allocator traffic per iteration.
2605.29346 covers metadata-driven host overheads in iterative training, the residual a graph does
not remove.

**Where it gets hard.** 2604.13327 (Event Tensor) exists precisely because "dynamic" megakernels
do not compile under static-graph assumptions; 2606.09682 (AutoMegaKernel) is a statically-checked
synthesis harness; 2604.26039 (RaMP) is runtime-aware megakernel polymorphism; 2609.12379
(ForgeMegakernel) targets AR *decode*. The megakernel literature is uniformly about
decode/inference, not training loops — SPECULATION: given 2609.23536's failure-mode catalogue and
that every paper above targets decode, I assume a one-graph capture of our forward+optimizer is
feasible only while the MoE dispatch is deterministic in shape; the first N different per-batch
expert sets will invalidate it, and our fallback must be capture-with-replay-off, not a silent
re-capture. Also relevant: FLA's chunk kernels carry `@torch.compiler.disable` (`[web]`) — the
reference library does not let torch.compile trace into linear-attention chunk kernels, and a
hand-written kernel loop is the same case, which is why the Rust loop needs its own capture.

## Open questions the sweep could not close

1. rank-64 low-rank experts vs. full-rank at ~1B with topk=2: no published sweep. Own A/B.
2. k=1 vs k=2 at this scale: no published A/B found.
3. Patch-by-coding-rate-gain: zero arXiv hits, and per 2610.11790 it sits in a criterion family
   with a documented blind spot for computed values.
4. A published ns_iters recommendation at 1B: arXiv has none; `[web]` sources say 5–6.
5. sm_120-specific bf16 GEMM/optimizer-state defects: no arXiv source, only indirect
   (2607.04422's RTX 5090 measurements) plus GitHub-level FLA bug reports.
6. bf16 recurrent state for KDA *training* (as opposed to inference quantization, 2608.27513):
   no paper.
7. "engram = 20–25% of params": not in 2608.30320 or any other source found.

## New watch topics added via arxiv MCP (2026-10-15)

1. `byte-level language model patch boundary entropy compute allocation` (cs.CL, cs.LG) — the
   gap that made the ByteFlow verdict "open"; catches the patching-criterion debate, currently
   the fastest-moving thing against that bet.
2. `CUDA graph captured training low precision numerical state contract` (cs.DC, cs.LG) — for
   Q8; catches the QEffect line as it develops beyond FP8 pipelines.
3. `bfloat16 Blackwell SM120 consumer GPU GEMM mixed precision kernels` (cs.DC, cs.NA, cs.LG) —
   for Q7; the only hope of replacing local measurement with published evidence.

The seven pre-existing watches were checked for material since 2026-10-11: nothing in their
alerts touches the eight questions. The relevant new items were 2610.15384 (E8 lattice KV
quantization), already covered by the 2026-10-11 DeepSeek note, and 2610.15709 (low-bit
recurrent-state quantization), which is inference-side and strictly weaker than 2608.27513 above.
