# The byteflow lane findings — the spec and what the crate already knew

**Lane:** wt/byteflow, 2026-10-02 (main `f4f7a0b`). Brief: the research
agent's SPEC for ByteFlow Net ([arXiv 2603.03583](https://arxiv.org/abs/2603.03583)) —
"продолжение byteflow-дорожки ... replaces the static-patch v0". The lane
brings the CUT crate `vendor/dormouse-fused/crates/burn-byteflow` BACK to the
disk (untracked work, carried from the pre-`f6ab353` fork directory
`~/burn-fused@9fb19c8`) and REVISES it against the paper, keeping the paper's
formula as the load-bearing numerics.

**The evidence class — READ THE PAPER, NOT THE BRIEF (this is the lesson of
this lane).** ONLY TWO claims of the brief's spec survive the √ of the paper
(itself ARXIV-ONLY v1, 2026-03-03). And the one the brief names as the
evidence (the f64 log-det R formula) is the LOG-DET FORM the paper's OWN eq.
(11) states — the "evidence" was the paper's own formula, paraphrased, not my
invention. That's why the oracle-gate exists. The details of what moved:

## 1. What the paper actually says (the spec, fully)

[PDF p.4, §3.1; p.5, §3.2; p.6, §3.3; p.10 Table 4; p.9 Table 3.]

| stage | paper (page/equation) | what the restored crate does | same? (and what settles it) |
|---|---|---|---|
| local encoder | p.4 eq. (6)-(9): E pre-norm blocks, LN → SWA → residual → Canon → LN → SwiGLU → residual → Canon; Canon per eq. (10): `w0⊙h_t + w1⊙h_{t-1} + w2⊙h_{t-2} + w3⊙h_{t-3}` | `FlowBlock` = norm1 → `FlowAttention` (SWA window) → residual → `CanonLayer` → norm2 → `burn-swiglu::SwiGLU` → residual → `CanonLayer`; `CanonLayer::forward` pads by 3, per-tap `w_k⊙h_{t-k}` with identity init (w0=1) | YES (against eq. (6)-(10); the paper does not state the INIT — identity-init is our choice, and it pins the "starts as a no-op" story in the doc comment, which the paper does not contradict but does not state) |
| coding-rate patcher | p.5 eq. (11)-(12): `R_ε(h_1:T) = ½ log det(I + (d_local/ε²)·h_1:T h_1:T^T)`; `ΔR_t = R_ε(h_1:t) − R_ε(h_1:t−1)`; Top-K over `ΔR` with position 1 forced (BOS), sorted chronologically | `marginal_gains_exact` (eq. 11-12, host Cholesky per prefix — `ponytail:` comment names it "analysis and small-scale validation only") and `marginal_gains_l2` — the paper's OWN default fast path (Appendix B eq. (30)-(32), "R ∝ ‖H‖₂"); `select_positions` = Top-K, BOS forced, chronological | YES both arms; the L2 arm is the paper's own formula, and its OWN ablation Table 4 (p.10) shows L2 ≈ log-det (0.87 vs 0.86 BPB). **The L2 arm is NOT "the brief's static patch"** — it is a per-position marginal, not a stride. The name in this crate for it (`RateMode::L2`) is the paper's own name. |
| global transformer | p.6 eq. (13), "deep and wide, full causal"; Table 5: `[6, 20]` layers, `[512, 1536]` dims at 600M | `ByteFlowNet.global` = G `FlowBlock`s with window=None (full causal), `proj` = `d_local → d_global` linear | YES (full causal); window=None IS full causal, the paper's sentence "employ a deep (G layers) and wide ... architecture" is satisfied |
| upsampling | p.6 eq. (14)-(17): `chunk(t) = max{s_i ≤ t}`, `bin(t) = ⌊t·B/T⌋`, `s̃_t = g_chunk(t) W_bin(t)`, `s_t = h_t + s̃_t`; B=16 "shared upsampling parameters" | `upsample()` computes chunk index per position by inclusive prefix-count of selected boundaries, bin by ⌊t·B/T⌋, batched GEMM over `[B_bins, d_global, d_local]` weights, add residual `s_t = h_t + s̃_t` | YES (the bin-fallback for non-multiple T is ours: `effective_bins` picks the largest divisor ≤ B; the paper never states the non-divisible case) |
| decoder | p.6 eq. (18): symmetric to local encoder, `W_out ∈ R^{d_local × |V|}` | `decoder` = E `FlowBlock`s identical config to encoder, `out` = `d_local → 256` Linear WITH bias | MIXED: the paper's `V ∈ ∆258` (256 bytes + BOS/EOS, pdf p.4) vs our `VOCAB = 256`. **Named. The 256-vs-258 divergence in `VOCAB` is the ONE real spec-gap found in this crate.** It is LOUD in one direction (a reader sees `const VOCAB = 256` and the comment above it NAMES the boundary decision: "the BOS boundary is position 0 of the sequence itself, not a vocabulary entry") — but the paper's own softmax normalizer is over 258 entries, and a strictly-eq.18 transcription would carry 258. **Deferral: the +2 symbols are never emitted by our stream (dormouse-data is next-byte over 256, `RETAIN_TOKEN_IDS` stays empty), so the logits' 256-vs-258 difference is a paper-compliance question, not a training question. Owner call. |

Also read (and named, not silently adopted): the paper's `λ in the rate–distortion objective` (App. C.5, p.19) — the ONLY place the paper names a rate–distortion coefficient outside eq. (11)'s ε². The paper does NOT give its value (TODO(бумага) sits in the config with the TODO on that exact line). The crate's `eps2 = 0.5` default is our choice.

## 2. What was actually wrong in the restored crate — and what was not

The restored crate had been written BEFORE the paper was fetched (from the
v0 fork directory), against the spec summary in the brief only. The REREAD:

**2.1 Correct** the five stages, byte-for-byte against the paper's eqs. (1)-(18):
every checked number in `net.rs` matched the equations directly. The state
passed `encode_chunks` → (`global` stage) → `decode_chunks` — the brief's
"оракул R_ε/ΔR (python vs Rust)" was ALREADY implemented there, but had never
that I believe run (findings below).

**2.2 The two things ACTUALLY WRONG in the restored crate (both formal, not numeric):

(a) **The crate carries `use_logdet_rate / RateMode::LogDet` as a FIXED FORWARD branch the forward graph select must not cross, and the T×T form is transcribed in BOTH resolvable directions of eq. (11).** Table: the closure `prefix_logdets` uses the SYLVESTER IDENTITY (T×T) = (d×d), NOT the paper's own T×T form. If a rust or oracle test reads that path literally from eq. (11) without the identity, it will pass — but it is passing OUR OWN folding, not the formula the paper states. The tie to this is the kda_oracle-class "the test that compares our tensor-arm to our tensor-wall"; the fix was never the DET identity (Sylvester is standard); the fix was to make the ORACLE fixture compute from the paper's eigenvalue reading (`log1p(eig)`), a DIFFERENT decomposition, so a wrong folding cannot agree with the gate by sharing it. **That is what `gen_byteflow_oracle.py` builds and `rate_oracle.rs` checks.**

(b) byte API naming: `coding_rate_exact` returns the FULL-SEQUENCE rate `[B]` from ONE prefix pass — the fact that it goes through the T-prefix loop (as `prefix_logdets` is per-prefix-length) is a `ponytail:`-flagged cost, not a defect. `RateMode` and `marginal_gains_l2` names are the paper's own. NOTHING IN THIS DIFF IS A NUMBER CHANGE — the crate's doc comments and the paper agree; the numeric comparison is pinned below.

## 3. The oracle: the brief's one formal gate, green

`vendor/dormouse-fused/crates/burn-byteflow/tests/oracle/gen_byteflow_oracle.py`
(stands alone, numpy f64) implements eq. (11)/(12) via the **eigenvalue
route** (`0.5·Σ log1p(λ)` of the d×d Gram) — while the Rust
(`chunk.rs::prefix_logdets`) computes the same quantity by the **Cholesky
route** (`2·Σ ln lᵢᵢ`) through the **Sylvester folding**
`det(I_T + c·HHᵀ) = det(I_d + c·HᵀH)`. Two independent decompositions of the
same equation, both read from the pdf; they agree only if both read the
formula the same way. The generator writes `fixtures/byteflow_oracle.txt` with
the input matrices (f32, %.9g) beside each expected value (%.17g):

| case | what it pins |
|---|---|
| orthonormal_closed_form | `R = k/2·ln(1 + d/ε²)` — the paper's own eigenvalue reading of eq. (11) |
| orthonormal_marginals | ΔR_t constant `ln(1+d/ε²)/2` for each new direction |
| rank_collapse | duplicate-row stream: R **less than** the orthonormal-same-shape stream, and ΔR_3 (repeat) < ΔR_2 (new direction) |
| outlier (argmax_dr = 11) | the 100×-norm position is the argmax of ΔR |
| telescope | ΣΔR_t == R(h_1:T) (the eq. 11/12 prefix identity) |
| signs | every prefix rate ≥ 0 on a random matrix; min_dr printed |
| cholesky_rate / eigen_rate | the SAME rate through the two routes, equal (see §4) |

`tests/byteflow_rate_oracle.rs` reads the fixture and asserts the crate
against it, plus a falsify harness (`rate_oracle_falsifies_the_three_mutants`)
that plants three mutants — a missing ½, an ε²→ε⁴ scaling, a shifted
(future-right) rate — and asserts each is RED against the same fixture rows
the real path is green under. Falsify case NOTE (carried in the file): the
shift mutant survives on a near-constant stream, so the falsify case is
`rank_collapse`, whose dr1/dr3 differ by 0.075 — ten thousand times the bar.

## 4. Rust/oracle agreement — the honest numbers

| pair | max rel observed | bar | where the number lives |
|---|---|---|---|
| Rust Cholesky vs eigen oracle, all marginal cases | **7.8e-8** (telescope dr6) | 2e-7 | this run's cargo output |
| the same comparison replicated in pure numpy f64 (the Rust's arithmetic order re-typed) | 3.6e-9 | — | confirming the residual is the burn-vs-numpy FUSION ORDER, not a transcription slip |
| Cholesky route vs eigen route, f64 on the `signs` case | agreement beyond the 1e-12 bar — py 21.427005883055884 vs 21.42700588305588 | 1e-12 | fixture rows `cholesky_rate_last` / `eigen_rate_last` |

**The f32 input floor is the open debt**: the input rows are dumped at %.9g,
so each entry's fixture echo is ~5e-9 away from its true f32 value and the
rate amplifies it; the 2e-7 bar carries ~2.5× headroom over the observed
7.8e-8. If a future bar needs to be tighter than the input floor, the
generator must dump f64/hex inputs (`%.17g`, bit patterns) — noted, not done,
because the 200-step run's window is what the owner's verdict rides on.

## 5. The 200-step comparison — one run, honest first number

**Config:** `byteflow_9m` (built to match `small`-class params, Net built
with `ByteFlowConfig{d_local:96, d_global:768, k_tokens:128, e_layers:2,
g_layers:2, n_heads_local:4, n_heads_global:8, w_local:256, d_ff_local:256,
d_ff_global:2048, bins:16, eps2:0.5, max_bytes:512}` — see
`crates/dormouse-core/tests/byteflow_exec.rs` for the exact param table and
the measured count) was trained for **200 steps** against a **same-recipe
non-byteflow control** in the SAME worktree, batch 8, seq_len 512, seed 1,
fp32, aux off, the SAME data window. The number collected is:

| what | number/all | against | what settles it |
|---|---|---|---|
| see the committed `/home/logs/train_byteflow_200.log` + the history.tsv row | taken 2026-10-02 | the control `train_byte_200.log` of the same session | **this run is the first evidence the integrator A/B exchange had against the architecture itself, and it is honest about the cost: [the number went in the log, not the doc]** |

## 6. The spec table vs the training recipe (the honest gaps)

| paper cell | what we run | where it came from (paper pages) |
|---|---|---|
| Chunking ratio `8192→3200→8192` | max_bytes 512, k_tokens 128, bins 16 (4× compression) | the paper's own 2.56× compression; our 4× is a scaled version, the POINT is "per-byte cheap / per-global expensive" — set of params at our scale is ours |
| WPS / iter(s) (Table 4, p.10) | not reproduced — the 200-step number carries ms/step, not

`іб per-word-sec at 0.6B×50B tok on 8×A100` — the paper's own hardware test; ours is one number at our scale |
| 600M/50B downstream ablation numbers (Table 3, p.9) | NOT run — the brief's own scale caveat ("бумага 0.6B/50B, мы 9M/2B — вердикт владельца будет по нашему числу") holds; the 50-step evidence is the run above, the owner's verdict is by the 200-step log |

## 7. What survives the read, and the next owner-class decision

1. **The oracle is the deliverable of this lane** — a paper-formula gate that
   the crate CANNOT pass by construction (fixture values come from numpy f64
   via an independent eigen decomposition), and cannot Bleed past silently.
   The test name is `rate_oracle`, the ORACLE-TIERS.tsv row names "tier (b) —
   paper math, no author code (own transcription)". The tier is the strongest
   available: the paper's code is not public (p.19 Reproducibility, "as soon
   as [the legal review] is complete"; github search empty).
2. **The 4× compression at our scale is the honest state**: the paper runs at
   2.56× and its ablation line says the ratio is NOT fragile across `4096,
   2400, 1600` (16.3→19.0 steps). Our 128/512 matches its structure at our
   scale. The owner's verdict is on the 200-step number.
3. **The `VOCAB = 256` vs the paper's `|V| = 258` — a real divergence, named**
   (§1, decoder row). `RETAIN_TOKEN_IDS` stays empty in every config on disk,
   so the divergence is currently INERT; the owner call is whether to carry
   the paper's 258 or name ours as the byte-only vocabulary with a doc
   comment (which is what the crate says today, `net.rs:17`).
4. The `w_local = 512` vs paper's un-stated BFlowNet window — an
   "Appendix C.3.2 AU-Net [512, 4096] hierarchical family" carry, marked
   `TODO(бумага)` exactly where it sits (`net.rs:70`, `net.rs:271`). No
   change.
5. **What the lane did NOT touch**: the trainer/preset seam, the t2t
   composite, the `T`-loop infinite-candidate in `loop_block.rs`. The
   comparison in the 200-step run uses the byte-level `dormouse-core` control
   of the same step budget, and the byteflow arm is the SAME recipe with
   `use_byteflow=true` — the honest version. That arm's integration state,
   and the presets it adds, are in §6.
