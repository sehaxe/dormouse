# The 1B readiness plan — analysis, phases, decision points

**2026-10-15.** The owner's goal, stated today: **train a ~1B-parameter model
(total, possibly counted with MoE) on this box** (RTX 5060 Ti 16 GB + 64 GB host
RAM); the 9.2M `small` stack is the test polygon. Mandate: maximal
optimization, radical framework changes welcome. This is the analysis and the
action plan; §6 is the owner's decision list.

## 1. Where the project stands (the honest ledger)

### 1.1 Measured and standing

| what | evidence |
|---|---|
| KDA grad flow proven on the trainer's backend — CPU/CUDA bit-for-bit, all 11 groups | `tests/kda_param_grads_cuda.rs`, `d8fa449` (2026-10-01) |
| TSCT retraction wired + instrumented: scalar 75.65 ms → batched 19.69 ms (3.8x), flat slope 2-6 NS iters | `.bulba/goal.md` STATUS DONE §4, `retract_seam.rs` |
| bf16 44.06 / f16 44.04 TFLOP/s at the cuBLAS ceiling on 0.22 (after the `cubecl-ir` patch re-land); fp32 17-20 | 2026-10-10/11 tables in `docs/reviews/2026-10-09-bug-list-022.md`, AGENTS §2.1 |
| The 0.22 corruption root-caused: a kernel reads memory nobody wrote; the persistent pool decides what is read | `docs/research/2026-10-11-022-reads-unwritten-memory.md` |
| The measured config workaround: `persistent_memory = "disabled"` + `allocator = "sync"` → 2/3-3/3 clean vs 9/9 corrupted | same doc, the 2026-10-11 table |
| `CE_CEILING = 100` guard validated end to end — caught 3/3 corrupted runs at step 1 | same doc, retraction section |
| ByteFlow canon at equal params: 3.466 vs plain-bytes 3.947 (3 seeds, 2k steps); full-arm `small` LOSES at 6.498 | `benches/history.tsv:157-159`, trio 2026-10-04 |
| ByteFlow 100k: held-out **2.255** BPB on 409 MB — 0.325 below the 5-gram line (2.580, 8 MB fit) | `AGENT_CONTEXT.md:65`; anchors `history.tsv:160-161` |
| JEPA aux beats pure CE 3/3 seeds | `d8062d1`; the one arm with a positive verdict |
| Muon+ default optimizer, routing validated at startup; host-RAM Engram works (`--engram-ram`) | AGENTS §3.5, §3.7 |
| φ = 0.46: one recurrence buys 0.46 of a parameter (iso-depth law) → weight-shared depth is worth keeping | `docs/research/2026-10-11-recurrence-equivalence.md` (arXiv:2604.21106) |
| Every major bet against the published record: Muon hybrid confirmed (8-bit state at our scale 2509.23106; 4-bit Muon measured on an RTX 5090, 2607.04422); host-Engram confirmed twice (2608.30320, 2610.20456); **"20-25%" and the slot-count law REFUTED — allocation is swept, not cited**; the ByteFlow criterion is unpublished → judge against boundary dependence, not entropy (2610.11790; BLT = 2412.09871); fp32 KDA state; WSD + wind-tunnel for overtraining | `docs/research/2026-10-15-1b-tech-verification.md` (69 arXiv IDs, verdict table first) |
| Instrument layer: NaN firewall, CE ceiling, `engram=`/`fused kda=` counters, eval window printed, `--fit` anchors, determinism harness | AGENTS §2.6, §3.7; `tools/determinism.py` |

### 1.2 Open debts this plan pays

| debt | where |
|---|---|
| The workaround kit is UNCOMMITTED: the `cubecl.toml` flip (disabled+sync) and the `nan_windows ≥ 2` hard error sit unvalidated in the working tree | working tree diff, 2026-10-11 |
| headwise-Muon gates fail 2.27e-3 **only in full runs** — order-dependent, the uninit-read signature inside the optimizer; the padded headwise path in `burn-muon-plus` unread | `.bulba/memory.md` 2026-10-11 |
| Pool-cost trade never measured on CUDA (the "zero cost" reading was a CPU-backend artifact) | same |
| MoE is DENSE: the LoopBlock runs all experts per token (`loop_block.rs:982-997`); `moe_topk` exists, the dispatch lane unrun | §2.1 below — this is load-bearing for 1B |
| The fused KDA adjoint is wrong + non-deterministic → the trainer runs the tensor path (correct; ~205 ms of a ~480 ms step) | AGENTS §3.3 |
| Upstream A1-A3 drafted, not filed (token here is read-only) | `docs/reviews/upstream-issue-autotune-candidate-coin-flip.md` |
| No control re-baselined on 0.22; the A/B budget unknown post-`8fa5d4c` | AGENTS §3.3/§3.4 |
| `max_ortho`'s 1e-3 threshold will mis-latch at 1B: it thresholds a worst-case-over-more-factors statistic whose natural scale is 1/sqrt(m) | `benches/history.tsv:165` (moe-dev 2026-10-03) |

## 2. The 1B arithmetic — why the plan looks like this

Everything in this section is **derived arithmetic from measured constants**
(labeled D below), not a measurement. Sources for the constants: the tables in
§1.1 and AGENTS §3.1.

### 2.1 The throughput equation

Training FLOPs/token ≈ `6 × active_params_per_pass × max_iter` (+ aux). At the
measured operating point (9.2M, d 768, iter 2, b8 s512):

- D: tokens/step 4096, warm step ~245 ms launch-bound without KDA-bwd, ~480 ms
  with it → effective compute **~0.6-1.2 TFLOP/s** against a **44 TFLOP/s** bf16
  card ceiling. **Headroom ≈ 40-70x, and it is all in software.**

The three parts of that headroom, measured:
1. **Precision**: fp32 → bf16/f16 GEMM is 2.2-5x on the same stack (measured).
2. **Launch overhead**: GPU idle 87% of a warm step at 9.2M (AGENTS §3.1). At 1B
   the GEMMs grow 10-100x and amortize launches by themselves — but only if the
   step is not carrying hundreds of small launches (the dense MoE loop, the
   per-factor retraction).
3. **Active params per token**: the sparse-compute ladder (§2.4).

### 2.2 The candidate 1B shape ("b1", derived)

d 2048 · H 32×64 · d_ffn 5461 · rank 64 · max_iter 2 · E 1024 experts, topk 2 ·
Engram 4M rows/order × 3 orders × dim 32 (the DeepSeek-Flash operating point,
26% by our attribute table; no published "20-25% law" survives verification and
the allocation is swept, not cited — `configs/small.toml`, AB-PROTOCOL row 6b,
the tech-verification doc §4):

| component | params |
|---|---|
| embedding + lm_head (vocab 256) | 1.0M |
| KDA (q,k,v,g per-channel + beta) | ~16.9M |
| Engram dense half + router + out_proj | ~10.5M |
| Engram tables (4M rows/order) | 402M |
| TSCT experts: 1024 × (dual gate/up latent 2·(2048·64+64) + down 64·5461) + shared (rank 192 + its write) | ~639.5M (MEASURED on the instantiated port model, 2026-10-15: the §2.2 draft missed the down-projection and the second latent — the spike found it; `configs/b1.toml` comments carry the arithmetic) |
| Engram tables: 4M rows/order × 3 orders × dim 32 | 402.65M |
| **total** | **1,066,354,113 ≈ 1.07B (MEASURED, `tests/test_param_count.py`)** |
| active per token | ~57M (unchanged in shape) |
| **active per token (train, both iters)** | **~57M** |
| FLOPs/token (train, derived) | ~0.40 GFLOP |

Wall-clock for the token budget, at effective rates the program below targets
(D): 0.40 GFLOP/tok × 1e10 tokens = 4e18 FLOPs →

| effective TFLOP/s | 10B tokens | 5B tokens |
|---|---|---|
| 5 | 8.6 days | 4.3 days |
| 10 | 4.3 days | 2.1 days |
| 20 | 2.1 days | 1.1 days |

**The counterfactual that justifies the whole plan**: the same 1B as a DENSE
MoE (all 1024 experts per token) is ~6.2 GFLOP/token → **62-190 days**; a 1B
dense non-MoE is ~12 GFLOP/token → **125-380 days**. Sparse compute is not a
nice-to-have — it is the difference between "days" and "never". Byte-level
vocabulary (256) keeps CE/head costs negligible, which is the second reason the
1B fits where a tokenizer model of the same depth would not.

### 2.3 The memory budget (derived, at b16 s1024)

| item | fp32-masters variant | bf16-masters variant (the bf16-plan) |
|---|---|---|
| weights forward | 1.9 GB | 1.9 GB |
| masters | 3.8 GB | 0 (weights are masters + stochastic rounding) |
| Muon moments | ~2.1 GB fp32 | ~1.0 GB bf16+SR |
| AdamW m+v (router/embedding/etc.) | <0.1 GB | <0.1 GB |
| Engram tables in-VRAM | 1.6 GB fp32 | 1.6 GB |
| activations + KDA saved scratch | ~4-6 GB | ~4-6 GB |
| **sum** | **~13-14 GB — tight** | **~9-10 GB — comfortable** |

Host RAM: Engram host tables at 4M rows/order ≈ 3.2 GB (value + momentum); the
Qwen3.8-Flash-Next playbook scales them freely — 32M rows/order ≈ 24.6 GB, still
fits the 64 GB box with the desktop closed (§2.4 discipline). **So the VRAM is
the binding constraint, and the bf16-masters + host-Engram pair is what buys
it.** Fallbacks are in-tree: fp32 masters cost 4 GB and nothing else; Engram
in-VRAM is the `--engram-ram` off switch.

### 2.4 "Can we go more sparse?" — the full ladder

The user asked this directly. Five independent sparsity axes, what exists, and
what each buys:

1. **Compute (per-token expert sparsity)** — `moe_topk` dispatch: gather→expert→
   scatter, FLOPs = k/E. **Param-neutral** (the MoE-lane gate proves it), so its
   A/B is fair. Buys the 65-190x of §2.2's counterfactual. **The single
   load-bearing wall of the 1B goal.**
2. **Token sparsity** — ByteFlow patches bytes 3-5x by coding-rate gain
   (measured: the canon win + 2.255@100k); per-position adaptive depth via the
   in-tree MoR arm or an ALoDLM-style learned schedule (φ-research says a
   LEARNED depth schedule is where recursion pays; fixed depth won at 9.2M).
3. **Weight sparsity** — the factor-quant ladder already in-tree:
   fp8 → fp4 → int4 → ternary (max_ortho-latched, masters fp32, forward-only
   quant). **LittleBit (arXiv:2506.13771, NeurIPS 2025) is the natural next
   rung**: sign-binarize U,V ∈ {−1,+1} with learnable FP16 row/column/**and
   per-rank** scales — and TSCT already IS U·diag(s)·Vᵀ, so the per-rank
   importance scale is the `s` we already train. At rank 64 a bit-level expert
   stores ~75 KB vs ~960 KB bf16 (≈13x); E can grow ~10x in the same VRAM, or
   the 1B total becomes ~150 MB of expert storage. Their 11.6x kernel speedup
   claim needs bit-serial kernels we do not have; the memory win is immediate
   and the compute stays at the precision we reconstruct to. QAT-from-scratch =
   the existing STE machinery (`act_quant.rs`).
4. **Memory sparsity** — Engram hashed lookup touches only the batch's rows;
   host tables make capacity RAM-shaped, not VRAM-shaped (the playbook that
   put 51B n-gram params off-accelerator in Qwen3.8).
5. **Attention sparsity in time** — KDA is O(n) already; the MSA block-top-k
   re-entry (ADR-0014) is the 1M-context rung and is deliberately NOT on this
   plan's critical path (§5).

## 3. The action plan

### Phase 1 — land the 0.22 workaround kit (1-2 days) ← the "minimal crutches"

1. Commit the two working-tree changes **after joint validation**: the
   `cubecl.toml` flip (disabled+sync — rewrite its comment to cite the
   corruption doc; the current comment still argues for size-match+async) and
   the `nan_windows ≥ 2` hard error (a drowning-NaN run currently exits 0).
   Validation = rebuild with `--features cuda`, 12-step smoke × 3 runs, read
   the `backend=Device<Cube(Cuda)>` line every time (the 2026-10-11 lesson),
   expect 0 corruptions with both guards armed.
2. Measure the pool-cost trade ON CUDA (owed): size-match+async vs
   disabled+sync, same recipe, same window. If disabled+sync is ~free, it
   ships as the default and the A/B question dies; if it costs, the cost is
   named and the corruption risk is carried knowingly.
3. Read `burn-muon-plus`'s padded headwise path; give the order-dependent
   2.27e-3 failures a root cause (expected: another uninit read) and a minimal
   fix + a gate that fails before it and passes after.
4. File the upstream issue (A1-A3 text is drafted verbatim). **Needs a GitHub
   token with repo scope** — owner decision (§6).
5. Re-derive the `max_ortho` threshold for scale (1/sqrt(m) natural scale, the
   history row's warning) — paper work, no GPU.

Exit gate: a 0.22 build where 3/3 smoke runs are clean, both guards armed, and
every fix committed.

### Phase 2 — re-baseline + the 100M rung (1-2 days GPU)

The A/B precondition (AGENTS §3.4): a re-baselined control on 0.22 in which the
attention arm provably trains (`fused kda=` bwd > 0, `engram=` non-zero, the
window printed). Then the **100M rung** decides the 1B recipe — because the
trio says the arms LOSE at 9.2M, and the honest rebuttal ("arms earn capacity
at scale") has never been measured. Iso-tokens A/B at 100M:
`byteflow + KDA + Engram + MoE-topk2` vs `byteflow alone` vs `plain bytes`,
3 seeds. Arms that win here enter the b1 recipe; arms that lose are deleted
(A/B-or-death) — no arm enters 1B on faith.

### Phase 3 — the throughput program (1-2 weeks; ranked by §2's arithmetic)

1. **topk-dispatch MoE** — the load-bearing wall (§2.2 counterfactual).
   `topk_blend`/`topk_indices` exist; the work is the dispatch path + the
   `moe=[shares, H, dead]` gate as the utilization floor. A/B: topk 2 vs dense
   at iso-total-params, 100M rung.
2. **The bf16 step** — storage bf16 + fp32 (or bf16+SR) masters per
   `docs/architecture/bf16-plan.md`; the mixed-dtype rules are in-tree
   (KDA fused kernel and Engram kernels stay fp32 — cast at the arm boundary,
   COUNTED); Muon moments bf16 + stochastic rounding is the unblocked rung
   (bf16 matmul is proven on 0.22). Gate: the bf16 parity tests + a 60-step
   NaN-clean run + `max_ortho` unchanged.
3. **KDA fused-backward repair** — the tensor path is correct but takes ~43% of
   the step; the fused adjoint is wrong + non-deterministic (AGENTS §3.3). At
   1B this is the difference between a ~2 s and a ~1 s step. The repair is
   bounded: the adjoint's disagreement is measured (4.5e-2..2.6e-1), the
   post-op params agree at 2e-7, so the defect is localized in the adjoint's
   own kernels.
4. **Tokens-per-step operating point** — 4x tokens cost 1.47x time (measured);
   at 1B, VRAM caps b×t, so seq_len↑ at batch↓ is the throughput knob. Pick the
   b1 batch/seq from a VRAM ladder, not a guess.
5. **CUDA graph capture, one stage** — confirmed working on this GPU
   (`vendor/cubecl-fix/cubecl-cuda/tests/graph.rs` 5/5; the client API is
   `graph_prepare` → capture → `Graph::replay`). Design constraint already
   documented: a capture window refuses stream reads/syncs/handle writes, so
   the first version is the largest no-host-round-trip stage (the TSCT/FFN
   block), not the whole step. Do AFTER 1-3 — precision and dispatch change
   what the stage contains.
6. **Retraction at scale** — batched NS is the 1B default (the launch-overhead
   row); measure it at E=512-1024 and pick `--retract-every` with the
   re-derived `max_ortho` threshold (Phase 1.5) as the latch.
7. **Engram prefetch overlap** — pinned-memory + separate-stream prefetch of
   the next batch's rows (the report's playbook, §3.5 item 1); ~600 KB/step
   copy today, cheap to overlap.

### Phase 4 — the TSCT lane: TSCT2 + LittleBit (parallel, GPU-light)

The TSCT program already has a pre-registered design with gates and a
kill-switch (`docs/reviews/tsct2-design-2026-10-04.md`): fine-grained low-rank
experts (r=16-64, E=64-1024, k=2-4) next to an always-on shared expert of rank
192-256 (R_act ≥ 192, the CoLA-parity zone), routing in the latent space,
Gram-space retraction driven by accumulated Ση_t with a spectral latch, bf16
training and a fp8→NVFP4 QAT ladder over the last 15%. The bet the owner named
today: **TSCT is the whole capacity thesis — it must get more optimized, never
less.** This plan adopts TSCT2 wholesale and adds three things it does not yet
carry:

1. **LittleBit (arXiv:2506.13771, NeurIPS 2025) as the ladder's deepest rung.**
   TSCT already IS U·diag(s)·Vᵀ; LittleBit binarizes exactly those factors
   (sign(U), sign(V)) with learnable FP16 row/column and **per-rank** scales —
   the per-rank scale is the `s` we already train — plus residual compensation
   (a second summed binarized branch: the first TSCT2-variant to test). At
   rank 64 an expert stores ~75 KB vs ~960 KB bf16 (≈13x): E grows ~10x in the
   same VRAM, or 1B of expert capacity fits in ~150 MB. QAT-from-scratch on
   latents = the existing STE machinery; retraction stays on fp32 masters.
   A/B at iso-effective-bits vs ternary and NVFP4, 9M first, 100M to enter the
   recipe — the same discipline as TSCT2's own G-gates.
2. **The capacity/compute frame the design implies but does not price.** TSCT's
   job at 1B is capacity at near-zero active compute (§2.2: 639M expert params,
   0.96 MFLOP/token). TSCT2 §10's arithmetic-intensity ridge (τ ≈ 28 →
   E ≲ T·k/28 at its assumed FP8-class rates) couples the expert count to the
   tokens-per-step lever (Phase 3.4): E=1024 pays only from ~16K tokens/step
   up. Beyond that sits TSCT2's product-key router (PEER, >1M tiny experts) —
   the long-run version of the same bet, post-1B.
3. **The launch-overhead attack is TSCT2's own §12**: the fused flash-MLP
   expert kernel (x→z₁→GLU→z₂→y inside smem, h recomputed in backward and
   never leaving smem; at r=64 the tiles fit the ~99 KB smem budget) + graph
   capture — ~6 launches per expert × E × N is the densest launch site in the
   model, i.e. the "GPU idle 87%" measurement's main address. Same lever as
   Phase 3.5, pointed at the same place.

And the design's own kill-switch is the protection the bet needs: if
fine-grained dense-MoE beats TSCT at equal active AND total params, the
spectral part is deleted and MoE + narrow-dense + QAT + fused-kernel remains.
Both outcomes keep the capacity thesis; only the factorization dies.

#### Why Newton-Schulz polar, not QR (the owner's question, answered)

1. **They are different maps.** The retraction and the Muon preconditioner want
   the POLAR factor `U·Vᵀ` of `W = UΣVᵀ` — every singular direction flattened
   to 1, which is both the scale-invariant preconditioner and the Stiefel
   projection TSCT's factorization lives on. QR's `Q` has orthogonal columns
   but is not the polar factor: the Gram-Schmidt order carries the
   σ-magnitude bias through instead of flattening it — precisely the bias a
   spectral parameterization exists to remove.
2. **On this stack NS wins operationally.** NS is 2-3 existing matmul ops (the
   Gram `[r,r]` plus a fixed cubic in the r×r space — auto-differentiable in
   burn's tensor language, which `LinearLike::retract` requires); QR is a
   sequential r-step Householder loop (64 small dependent kernels — the exact
   latency class the 87%-idle measurement punishes), and CholQR needs a
   Cholesky kernel cubecl does not expose. New-kernel work for a worse map.
3. **Exactness, from the record (§3.7):** NS-3 sits **1.17e-15 from the
   Stiefel manifold** — orthogonality is effectively exact on-manifold — but
   only **14.5% from polar(X) in spectrum at a 10:1 spread**. So "is polar
   exact" is no in the spectrum and yes on the manifold, and the spectrum gap
   is affordable BY MEASUREMENT: POWER_ITERS 5→2 was indistinguishable in
   ortho drift (6.31e-5 vs 6.46e-5 per-entry) against a latch threshold of
   1e-3-class. If more exactness is ever needed, the cheap dial is MORE NS
   iterations with the PolarExpress coefficients inside the r×r Gram space
   (~3e4 FLOPs per gauge, TSCT2 §3) — not a different algorithm.
4. **Numerics for the training use-case:** NS suppresses tiny singular values
   (polynomial smoothing) instead of normalizing them — the graceful behavior
   near rank deficiency. QR amplifies noise in tiny directions; CholQR's
   Cholesky simply breaks when the Gram goes numerically non-PD after drift.
5. **Whether any retraction earns its keep is itself A/B-or-death** (TSCT2
   §13.2 names it: it costs 22% of a step today and stops holding under QAT on
   Ŵ anyway — the ε_crit curve and the no-retraction arm are gates in that
   design, and my Phase 3.6 re-costs it at E=1024).

### Phase 5 — the 1B launch (the goal)

- Pre-registered config `configs/b1.toml` (the §2.2 shape minus whatever the
  100M rung deleted), token budget by wall-clock decision (5B first checkpoint
  review → extend), `--fit` anchors re-taken at the eval window actually used.
- Pre-flight: §1.5 discipline (MemoryMax=40G cgroup, ram-guard live, desktop
  closed, one process), corpus drive mounted (it is), disk ≥ 100 GB free
  (188 GB now), logs in `~/logs/`, checkpoint + resume round-trip verified at
  reduced slots, the stress protocol (`--stress --stress-lr 2`) as the
  stability smoke before the long run.
- Timeline at the program's target: **~2-9 days for 5-10B tokens** (§2.2).

## 4. The substrate question (the radical option, honestly)

The alternative the mandate invites: **port the trainer to PyTorch** and keep
the technology. The case for it is real: every debt in §1.2 is solved-by-default
there (cuBLAS/Triton kernels, mature bf16, torch.compile + CUDA graphs for the
launch overhead, FLA's battle-tested gated-delta kernels — ours are transcribed
from that lineage anyway, official Muon, MoE dispatch kernels, ZeRO-style
offload for the host playbook). The case against: the 1B step is GEMM-bound and
the patched cubecl stack MEASURES AT THE CEILING (44 TFLOP/s), the technology
library (28 crates) would be re-transcribed (5-10 working days, plus new-bug
risk), and the compiled-kernel library's correctness work would restart from
zero. The technology — not the framework — is the moat; but the moat ports,
the framework does not follow.

### 4.1 The spike's measured answer (2026-10-15)

Full report: `docs/research/2026-10-15-torch-spike.md` (torch 2.13.0+cu129,
CPython 3.12.15 via uv, this card, pre-flight before each of 13 GPU runs).
The verdict table against the owner's ordered criteria:

| criterion | burn (Option A) | torch (Option B) | winner |
|---|---|---|---|
| code / maintainability | 24k LOC owned + 55k vendored + **1.5-3.9k LOC still owed** (Phase 3) | 2.75-6.4k new LOC; **~39k of hand-written kernels deleted, not translated** | **B, 10-20x less owned code**; the instrument layer (LOUD/COUNTED, resume refusals) is a rewrite, not a freebie |
| memory | 9-10 GB (bf16-plan, derived) | **6.93 GiB allocated / 8.88 reserved measured** at b16 s1024, 518.5M params (no Engram/KDA yet) | tie — B lands inside A's own budget; the [E,cap,f] MoE materialization needs a Triton kernel on either path |
| tok/s at b1 | **no measurement exists**; nearest measured point: 9.2M, 245 ms/step, 87% GPU idle, dense MoE | **45,861 tok/s eager → 59,901 compiled** = 15.7-20.5 effective train TFLOP/s, **99.9% GPU util**; 10B tokens ≈ 1.9 days; at 46% of the 44 TFLOP/s silicon ceiling | B by default — nothing measured says burn wins; a finished burn shares the same ceiling |

And the plan's load-bearing wall is now MEASURED, not derived: **grouped topk-2
MoE is 339× the dense loop at iso-total-params** (52.6 ms vs 17.8 s, fwd+bwd).

**Provisional verdict: Option B (port), by the owner's own priority order.**
Phase 2 closed the two gates on the torch side (`/tmp/opencode/dormouse-
torch-spike/` commit `7920ad2`, the working notes below — the report file is
the spike dir's SPIKE_REPORT.md, phase-2 section):

- **Gate 1 — FLA on sm_120: YES.** `flash-linear-attention` 0.5.2 runs
  `chunk_kda`/`chunk_gated_delta_rule`/`KimiDeltaAttention` on this card with
  torch 2.13+cu129: fwd+bwd 27.9 ms at the model shape (= 587k tok/s arm-
  only). Caveat pinned: the chunked kernel sits ~0.3% (fp32) / ~1.8% (bf16)
  from the exact gated-delta recurrence, so FLA-backed KDA is NOT tensor-
  comparable to burn-kda — a recurrent reference stays the anchor, and no
  cross-substrate "bitwise" claim is ever made.
- **Gate 2 — torch-max, measured.** Real KDA + the cap-padded bmm MoE
  (`moe_fast` — `torch._grouped_mm` was measured 2× SLOWER at E=1024/b1: 44.2
  vs 21.4 ms fwd — and REJECTED by the spike's own probe) + AdamW + compile:
  **304.67 ms/step, 53,776 tok/s** (7.93 GiB); the eager variant with bnb
  8-bit optimizers: 389 ms, 42,110 tok/s (6.38 GiB). compile and bnb8bit are
  mutually exclusive ("cannot mutate tensors with frozen storage") — a
  recorded trade, not a bug. The same-model substrate gain vs vanilla is
  1.22× (73,137 tok/s with the SDPA stand-in); the real KDA arm costs
  80 ms/step over it (1.36×) — that is the honest price of attention.

**Loud negatives, carried:** `LigerFusedLinearCrossEntropyLoss` returns a
~40×-wrong loss and then illegal-memory-accesses at other shapes — excluded
everywhere; and `_scaled_grouped_mm` is sm_90/sm_100-only, so **fp8/expert
COMPUTE has no off-the-shelf path on sm_120** — the quant ladder's fp8/NVFP4/
LittleBit rungs remain STORAGE-side (factors stored low-bit, computed in fp32
masters → bf16 matmul, exactly `--quant`'s burn semantics), which the compute
gap does not touch. The vLLM fused kernel stays unclaimed; 7.9 ms of
permutation traffic + ~2.9 ms of fp8 are named headroom.

**Port LOC bands after the FLA check: best 2,850 · expected 3,300-3,900 ·
worst 4,300** (was 2,750-6,400 — the KDA-as-dependency fact cut the fat tail).

Option B stands on every one of the owner's ordered criteria; the burn stack
stays untouched as the fallback and the measuring instrument of record for
0.22 math questions.

Carried caveats (the spike's own): AdamW not Muon+ (burn's 52-61 ms retraction
is NOT inside the 357 ms comparison), SDPA not KDA, no Engram, no data
pipeline/resume/instrumentation; and the plan §2.2's expert pricing has no
down-projection — the spike folded one shared [5461→2048] (+11.2M) to keep the
specified shape; the TSCT2 GLU-at-latent pricing replaces this before b1.toml.

Burn stays the fallback that produced every trustworthy number to date and can
ship the first model if either gate fails; nothing on the burn path is deleted
by the port decision.

**Decision criterion refined by the owner (2026-10-15): speed is welcome but
decides LAST — the order is (1) less code, (2) easier to maintain, (3) memory,
(4) only then tok/s. And any Python-side work (the spike, any port) uses
`uv` for envs/deps, `ruff` for lint/format, `pytest` for tests; a torch-port
decision starts with a `pyproject.toml` + ruff + pytest skeleton on day one.**

**The torch ecosystem inventory (2026-10-15, owner: "максимально выжимаем
память fp4/bf16/fp8 + открытые ядра") — a vanilla-torch spike number is NOT
the torch verdict; the decision compares burn-at-its-best vs torch-at-its-best.**
"Kitty" = **ThunderKittens** (HazyResearch), verified: a CUDA tile DSL whose
kernels are callable from PyTorch, on Blackwell since 2025-03 (BF16 and FP8
GEMM/attention), Vera Rubin support 2026-09; consumer sm_120 support is what
the spike must PROBE, not assume. Relevant inventory:
- **Attention/KDA kernels that already exist outside our hand-rolled stack**:
  FLA (`flash-linear-attention` — the lineage burn-kda was transcribed from),
  TK's linear-attention/DeltaNet kernels, triton scan kernels — on the torch
  path the fused KDA backward is a DEPENDENCY, not a repair item (our §1.2
  fused-adjoint debt dissolves).
- **Precision/memory**: torch native fp8 `scaled_mm`, `torchao` (fp8/fp4/int4
  + QAT), `bitsandbytes` paged optimizers, DeepSpeed ZeRO-offload, Liger
  Kernel fused RMSNorm/SwiGLU/CE, gradient checkpointing.
- **Launch overhead**: `torch.compile` reduce-overhead (CUDA graphs); at the
  far end the TK Megakernel direction (a whole model in one kernel — the
  maximal answer to the GPU-idle-87% measurement, "Low-Latency Llama-1B",
  Megakernels repo). `train-tk` is the training-with-TK-kernels template.
- **MoE**: TK fused token-dispatch + expert-GEMM ("Loads and Loads of Fluffy
  Kittens", 2025-11), vLLM-class triton grouped GEMM.

**Spike phase 2 (after the vanilla numbers land, same spike session)**:
re-measure the loop-block step with the ecosystem's best available on sm_120
(fp8 scaled_mm, compile+cudagraphs, Liger-class fusions, FLA/TK attention
stand-in if installable) — the verdict table then carries torch-vanilla AND
torch-max rows before the substrate decision is taken.

**Standing principle (owner, 2026-10-15): use the best existing kernel; do not
roll our own where a proven optimized one exists — the ecosystem is there
precisely so we don't take that risk. "Maximal optimization and modernity."**
Consequences, stated honestly:
- On the **torch path**, the hand-rolled layer shrinks to the three mechanisms
  nothing exists for — TSCT (STE-quantized low-rank experts), Engram (host
  hash-table memory), ByteFlow (coding-rate patching); attention/optimizer/
  quant/data become dependencies (FLA/TK, Muon, torchao/Liger, HF-class).
- On the **burn path**, that layer stays OURS to write and own — including the
  KDA fused-backward repair (Phase 3.3), the quant ladder rungs, the graph
  capture and the MoE dispatch. The same "use the ecosystem" principle is
  exactly why several Phase-3 items are torch-side dependencies and burn-side
  engineering; the phase-2 spike must therefore produce, per path, a
  **replaced-by-ecosystem vs stays-custom LOC table** as part of the code-
  volume criterion, not just a tok/s number.

## 5. What I would NOT do (the traps)

- **1M context before the first model.** KDA is O(n) and the arm exists, but
  seq 512-2048 is what the 1B recipe needs; the MSA re-entry + RoPE/window work
  is a post-1B rung.
- **MuonQ 4-bit / fp4 factors before bf16 is proven in a real run.** The ladder
  has rungs; skipping them re-runs the fp4 debacles (§3.2 history).
- **Carrying losing arms into 1B** (GR, AttnRes, mHC, MoR, full-arms `small` —
  all ties/losses at 9.2M). They re-enter only if the 100M rung revives them.
- **Whole-step CUDA graph capture first.** The capture window refuses host
  round-trips (documented); one stage, then grow.
- **Deep-supervision rework before the A/B asks for it** — `L_Rec` already
  supervises every iteration, which is exactly what the looped-DiT
  cross-check (arXiv:2609.40305) says naive looping needs.
- **NRGPT-style re-architecture** — per-head D×D matrices are VRAM-infeasible
  here (ruled in §3.6, stands).

## 6. Decision points for the owner

1. **Substrate**: run the 48-72 h PyTorch spike at the b1 shape before
   committing Phase 3 engineering (recommended), or stay burn-only.
2. **First-model scope**: 9.2M control → 100M rung → 1B flagship (recommended
   ladder), or straight to 1B, or prep-only and you launch it yourself.
3. **"1B total" reading**: 1.066B on-card (the §2.2 shape, MEASURED — the
   dual-latent + down-projection correction pushed the draft 0.92B up 15.6%),
   or 1B counted WITH host-RAM Engram tables — a knob the loader already has.
   The owner left this to me; the on-card reading is the flag-first one.
4. **Upstream issue**: provide a repo-scope GitHub token (the drafted text is
   ready), or file it yourself from
   `docs/reviews/upstream-issue-autotune-candidate-coin-flip.md`.
5. **LittleBit lane**: design note + 9M A/B now (GPU-light, parallel), or after
   the first model ships.

## 7. The port lane (started 2026-10-15, owner's go on the substrate verdict)

The PyTorch port lives in `port/` inside this repo (one git, shared docs; no
new repo). Toolchain per the owner's intake: **uv (envs/deps), ruff (lint/
format), pytest (tests)**, Python 3.12, torch 2.13+cu129 (the spike's measured
stack). Day-one skeleton = config schema (strict, loud unknown keys), the model
stack, a byte-data cell, the trainer loop with the instrument layer, WSD
schedule, checkpoints with config-snapshot refusal, and CPU tests. GPU pieces
land only after spike phase 2's gates close (FLA-on-sm120 verdict, fused
grouped MoE, fp8 experts).

**Architecture decisions baked in from the record (not renegotiated per file):**
- Custom code stays at the minimum the ecosystem lacks: TSCT (STE-quantized
  low-rank experts — the dual-latent GLU variant per TSCT2 §14, rank
  configurable), Engram host tables (FNV byte n-grams, host optimizer), and
  the ByteFlow patcher hook (interface only until its criterion is validated).
  Everything else is a dependency: FLA for KDA (optional, LOUD when absent;
  SDPA is the COUNTED fallback and the CPU test path), Liger/RMSNorm+CE
  fusions, torchao/fp8 for the quant ladder, bitsandbytes paged optimizers
  where VRAM calls for it.
- **Where Rust stays (owner go, 2026-10-15): the three custom CPU mechanisms —
  the data pipeline (`crates/dormouse-data`: byte streaming, FNV n-gram
  hashing, ring, split), the Engram host tables and their per-step update
  (`train/src/offload.rs`), and eventually the ByteFlow patcher — are KEPT IN
  RUST as a small PyO3 extension (`port/ffi/`, own crate outside the root
  workspace, built by maturin through uv), not rewritten in Python.** They
  are pure CPU, already written and battle-tested for years, and two of them
  run per step at millions of rows (the host-table update); Python glue calls
  the minimal ABI. The skeleton's Python `data.py` stays as the interface
  definition and fallback — the Rust binding must match it on a parity test,
  so nothing is hidden. GPU-side Rust buys nothing (torch owns the accelerators
  there) — a whole-trainer-in-Rust/tch-rs rewrite is rejected: glue wins in
  Python, hot loops on GPU, hot loops on CPU stay Rust.
- Precision plan at init: bf16 activations/weights, fp32 KDA state + softmax +
  loss; the quant ladder (fp8 → NVFP4 → LittleBit sign+scales) exists as
  config exactly as the burn side spelled it
  (`docs/reviews/littlebit-design-2026-10-15.md` gates carry over verbatim).
- Continuing invariants from the burn lessons (ADR-0011/0016/0018/0021): the
  instrument layer is day-one code, not an addition — every degradation
  carries a LOUD/COUNTED/SILENT mark, device counters are read at declared
  cadences, and resume refuses a config whose resolved values differ
  (progress keys exempt).
- Model shape constants come from §2.2/§2.3 of the plan (b1 ≈ 1.066B total
  MEASURED by the port's instantiated count, 57M active; the port's `small`
  preset is 9,476,993 vs burn's 9,197,390 — the +3.0% is structural: the
  shared expert + its write per TSCT2 §14, minus burn's ~1.2M aux heads; the
  delta is pinned in `port/configs/small.toml` and its test, so cross-
  substrate arm verdicts read it and never silently compare counts).
- **The ladder is DROPPED (owner, 2026-10-15): b1 directly.** The 100M rung's
  promised transfers (LR transfer, capacity debugging) were derived, never
  measured, and the owner called it: "no bonuses were given, go straight".
  What remains is a 9.2M smoke (hours): correctness, arms engaged, speed vs
  burn's 245 ms/step — a gate, not a rung. b1 is the first real run.
- **Parity discipline (owner, 2026-10-15, applies to EVERY ported
  mechanism): burn is the INSPIRATION and the formula source, not the
  integration spec — the port is written from scratch, clean ("burn crates
  are junk; write it perfectly and verify").** The verification is therefore
  math-level, not line-level: every ported mechanism carries (a) a
  golden/property test that pins its formulas — the burn crates' host
  fixtures (`vendor/dormouse-fused/crates/burn-byteflow/tests/*_oracle.rs`)
  remain excellent sources of expected VALUES — or (b) a documented numeric
  anchor (the KDA rule: a 10-line recurrent reference), and additionally
  (c) named properties asserted as invariants (rate ordering, forced BOS,
  chronological patch order, window arithmetic, resume/progress keys).
  Integration design is decided by the port's own architecture and the
  readiness plan, with burn's A/B results as history, not spec.
- **Kernel interfaces stay pure** (torch in / torch out, no global state, no
  custom allocators): reusable-good, not reusable-goal — performance and few
  owned lines first (owner directive).

### 7.1 The post-training lane (SFT → RLVR → self-evolve; owner: RL in the framework, first-class)

Grounded by `docs/research/2026-10-15-agentic-post-training.md` (SOTA read;
the owner's `docs/architecture/post-training.md` was the start, confirmed OR
corrected there). The research's own headline: **no paper does agentic RL at
≤1B total / ≤100M active** — smallest published points are 0.5-3B — so every
mechanism below is a transfer, and the queue starts cheapest-provable:

1. **Verifier protocol + rollout runner on `code` only** (~300-500 LOC):
   exec sandbox step, exact-match, per-decision logging; gate = a 20-task
   smoke where every verifier decision is attributable.
2. **Fork-replay advantage (CRR 2609.33875) + SFT spans**, A/B against a
   GRPO-shaped group baseline at fixed wall-clock. CRR beat GRPO 41.7 vs
   36.7 SWE-V at 14B with NO learned reward model — the owner's "PPO not
   GRPO" instinct reads as: group-advantage methods collapse on all-fail
   groups (2610.14000, at 1.5B); the PPO-vs-fork question gets measured.
3. **Only if it learns**: control-plane EGGROLL (2511.16652; the owner's
   bet: rank-r ES ~100× cheaper, routers/harness only) + the SEED-shaped
   skill loop (2607.14777) with adaptive retirement (2609.20784) + anchored
   self-play (2607.03523 anti-drift).
4. Data side rides `docs/research/2026-10-15-data-strategy.md`: verifier
   tasks synthesis → solvability screen → anti-hack; agent traces as SFT
   spans (SWE-Gym/Open-SWE/ToolACE, Apache-2.0).

Reserved CLI surface (matches `docs/architecture/post-training.md`):
`dormouse post-train --base … --env code,math --verifier hard` — wired after
step 2 learns anything.



Burn lanes pause while the port lane is active (one lane at a time; the burn
stack stays green and untouched as the fallback).
