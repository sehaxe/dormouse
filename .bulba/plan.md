# Plan: Fused CUDA PonderNet kernels with manual backward (eliminate CPU autodiff-graph construction)

STATUS: IN_PROGRESS (approved by user 2026-09-06: "иди оптимизируй все")

## User constraints (2026-09-06, binding)

- **UPDATE 2026-09-07: the 48-iteration requirement is REVOKED by the user.** New target: max_iter 8 (preset), random-depth training T in 1..8 per STARS/RL-Halting evidence. Fused op targets N_max=8; bf16 workspace budget shrinks accordingly.
- No CPU work in the hot loop, no per-step GPU sync beyond what reads a scalar back (host-Adam D2H
  rides the existing read; target GPU idle < 10%).
- Architecture direction v2 (research/minicpm5-dormouse-direction.md + .bulba/alphaxiv-analysis.md):
  offline JEPA/KD teacher targets (MiniCPM5-2B as frozen teacher), hard top-1 MoE + shared expert
  replacing soft expert blend, Engram placement A/B (in-loop vs early).

## Goal

Replace the current `burn_autodiff`-built PonderNet graph (which builds ~74 ops/iteration on the
CPU, leaving the GPU idle ~74% of each step per AGENTS.md) with a **single custom autodiff op** whose
forward and backward are hand-written cubecl kernels. The op covers the whole `max_iter`-iteration
loop (`loop_block.rs` `forward_full_state`, 129–381) plus the final norm + lm_head + PonderNet loss
(`model.rs` 44–136). CPU only issues kernel launches; no per-op autodiff node is constructed. The
Muon+ optimizer (`optim.rs`) and `GradientsParams::from_grads` → `optim.step` path
(`lib.rs` 457–496) stay byte-for-byte unchanged.

## Context / current bottleneck

- `forward_full_state` runs `max_iter` (now up to 48) iterations; each iteration calls dozens of
  small burn ops (matmul, RMSNorm, sigmoid, softmax, silu, quant, top-k, halt, CE). Every op becomes
  an autodiff graph node built on CPU in `burn_autodiff`; backward traverses the same tree on CPU.
- `bf16_ops.rs` (burn-spectral, 21–88) is the exact template we reuse: a `Backward<B,2>` impl that,
  in `forward`, extracts the inner cube tensor handle via `try_into_primitive::<Autodiff<Inner>>()`,
  runs a cubecl matmul in bf16, and in `backward` calls `grads.register::<Inner>(node.id, grad)`.
- We do **not** use `cubecl-autodiff` (unavailable). Manual backward is done in our own cubecl
  kernels. This is compatible: `burn_autodiff` only needs a `Backward` impl to know how to propagate
  grads for our single node; everything inside is ours.

## Open questions & recommended defaults (override on approval)

1. **Integration shape.** (a) build `GradientsParams` directly from a precomputed param→grad map with
   *zero* autodiff nodes, vs (b) one custom `Backward` op whose `backward` calls our kernel and
   `grads.register`s each weight grad. **Recommended: (b).** It keeps `loss.backward()` /
   `GradientsParams::from_grads` / `optim.step` identical (Muon+ untouched), and a single graph node
   makes CPU cost negligible. (a) requires poking `Gradients::new`/`GradientsParams::from_map` which is
   a more fragile API and buys little.
2. **Attention (KDA/MSA) + Engram in the fused op or kept as burn sub-ops?** **Recommended: fuse
   everything, but port KDA/MSA backward from their existing crates' source.** PoC de-risks by
   training with `DM_NO_KDA`/`DM_NO_MSA`/`DM_NO_ENGRAM` (env switches already exist, loop_block.rs
   255–259) so the expert-FFN+controller+residual+halt+loss path is proven first. KDA/MSA backward
   then added as sub-kernels derived from `burn-kda`/`burn-msa` backward math.
3. **Numerical target / acceptance tolerance.** **Recommended:** in `BF16=0` (default) the fused path
   must reproduce the current fp32 autodiff loss **and** weight grads within rel < 1e-4 (float-diff /
   burn-reference). In `BF16=1` mode it must match the current fp32-compute path within rel < 1e-2
   (the existing `bf16_matmul` test already allows ~0.01 forward, bf16_ops.rs 111). This keeps the
   fused kernel a drop-in replacement, not a retrain.
4. **In-kernel matmul precision.** **Recommended:** default = fp32 cmma (exact match to current fp32
   autodiff). Tensor-core **bf16** matmul is an *optional* `DM_FUSED_BF16=1` fast path (diverges
   ~1%, same as today's `forward_quant_bf16`). Storage of saved intermediates is bf16 regardless.
5. **Memory strategy for 48 saved iterations.** **Recommended:** store-all intermediates in bf16 to a
   pre-allocated workspace; if VRAM-bound at larger presets, fall back to *checkpoint-recompute*
   (store only the per-iteration recurrence seed `h` + KDA state, recompute the rest in backward at
   ~2× compute). See Risks for the budget.

## Kernel decomposition

The fused op is **one autodiff node**. Its forward/backward are CPU-side orchestrators that issue a
fixed sequence of specialized cubecl kernels (launched directly on the `ComputeClient<CudaRuntime>`,
not through burn) for each of the `N` iterations, looping on the GPU with no autodiff node per op.

### `ponder_forward` (one launch per training step, loops N iters internally)
For iter `n` in 0..N, all on GPU, reading weights from a constant workspace and writing saved
intermediates to a workspace buffer `W` (flat `[N][...]` arrays, indexed by offset — no 4D autodiff
tensor is ever created, sidestepping the sm_120 4D-slice crash, AGENTS.md):
1. iter_embed gather + GR read (`gr.rs` read 57–84) or ReZero pre-norm (`self.norm`, loop_block 196).
2. RMSNorm (`norm`) → activation quant (`act_quant::quant_act`, 45–80) for attn/ffn paths.
3. Controller linear (`controller.forward` 245) → split into `w_attn/w_mem/w_ffn` (sigmoid 246–248)
   and `blend` (softmax 249).
4. Attention: KDA `gdn2.forward_train_state` (260) + MSA `msa.forward` (270) + router blend
   (`attention.rs` blend 70–85) × `w_attn`; Engram `engram.forward` × `w_mem` (281–316).
5. Expert FFN: for each expert, `gate_up` (TSCT) → silu → `down` (TSCT), blended by `blend`
   (321–327) × `w_ffn`.
6. Residual add (`y = attn+engram+ffn`; ReZero 337–342 or GR write 331–336).
7. Halt: `out_proj` (347) → `halt_head` → `lam=sigmoid` (353–355) → `p_n`, `out_acc`,
   `not_halted` (356–359); per-step CE `logits_n = lm_head(step_out)` → `rec += p_n·CE` (361–375).
8. Save to `W[n]`: `h_ctx`, `normed`, `normed_ffn`, per-expert `mid`/`silu(mid)`/`out`,
   `attn`, `engram_a`, `y`, `h` (or `branches` under GR), `w_attn/w_mem/w_ffn/blend`, `lam`,
   `p_n`, `gdn2_out`+KDA `s_new`, `msa_out`, `step_out`, `not_halted`.
Final: `out_acc`, `rec`, `p_dist = cat(p_rows)` (378), `kda` returned; final norm+lm_head over
`out_acc` (model.rs 88–97) computed in fp32 inside the kernel (bf16 logits NaN rule, model.rs 87–88).

### `ponder_backward` (reverse loop N-1..0)
Reads `W[n]` and the upstream grad of the loss. Computes, per iteration, `dW` for every weight and
`dX` (which becomes the next iteration's `dh`). Weight grads **accumulate across iterations** into
grad tensors of the weight's shape (weights are shared across iterations). KDA/MSA/engram grads use
their dedicated sub-kernels (see below). At the end, all weight grads + the embedding/lm_head grads
and `iter_embed`/`residual_scale` grads are registered.

### Per-op kernels (each a cubecl kernel, cmma for matmuls)
- `k_matmul` (cmma, fp32; bf16 variant under Q4 fast path).
- `k_rmsnorm`, `k_silu`, `k_sigmoid`, `k_softmax` (expert blend + log_softmax for CE).
- `k_quant` / `k_quant_ste` (act_quant, STE: grad passes through, 78–79).
- `k_tsct` forward/backward (factorized, see math).
- `k_kda_step` / `k_kda_step_bwd` (port from burn-kda backward).
- `k_msa` / `k_msa_bwd` (top-k gather/scatter + softmax jacobian; port from burn-msa).
- `k_gr_read`/`k_gr_write` (+bwd) for GR path.
- `k_halt` (sigmoid + geometric recurrence jacobian).
- `k_ce` (log_softmax cross-entropy + KL vs truncated-geometric prior, model.rs 113–136).

## Backward math checklist (derive dW, dX for each layer)

Notation: `dY` = upstream grad of `Y`. All summed over batch/seq dims unless noted.

- **Dense matmul** `Y = X·W`, `X:[m,k] W:[k,n]`:
  - `dW = Xᵀ·dY`  ·  `dX = dY·Wᵀ`
- **TSCT (SpectralLinear)** `Y = ((X·U)·s)·Vᵀ`, `U:[d,r] s:[r] V:[f,r]` (param.rs / burn-spectral
  lib.rs 524–527):
  - `Z=X·U [m,r]`; `M=Z·s [m,r]`; `Y=M·Vᵀ [m,f]`
  - `dM = dY·V [m,r]`; `dVᵀ = Mᵀ·dY [r,f]` ⇒ `dV = (dYᵀ·M) [f,r]`; `ds = Σ_m(dM⊙Z) [r]`
  - `dZ = dM·s [m,r]`; `dU = Xᵀ·dZ [d,r]`; `dX = dZ·Uᵀ [m,d]`
  - Quantized factors (ternary/fp8/fp4): STE ⇒ `dU_master=dU, dV_master=dV` (no grad through the
    round; matches act_quant.rs 78–79). bf16 tensor-core forward path: backward is exact fp32.
- **RMSNorm / branch_norms** `out = X/√mean(X²)+eps · g` (no bias):
  - `r = X·(1/σ)`; `dL/dr = dOut·g`; `dX = (1/σ)·(dL/dr − r·mean_feature(dL/dr))`; `dg = Σ_{b,t}(r·dOut)`
- **Sigmoid** `σ(x)`: `dX = dOut·out·(1−out)`
- **Softmax** (expert `blend`, rows 3..3+nexp): `dRaw = dBlend − blend·Σ_k dBlend_k` on those rows;
  rows 0..3 (w_attn/w_mem/w_ffn) get `dOut·out·(1−out)`.
- **Silu** `z·σ(z)`: `dz = dOut·(σ(z)+z·σ(z)·(1−σ(z)))`
- **Activation quant (STE)**: `dx = dOut` (x + (xq−x).detach()); scale grad ≈ 0 or via clamp.
- **Gated Residual read** (`gr.rs` 57–84): `G=σ(Wu·silu(Wd·vec(R)))`; `x_in=(1/nr)Σ_i G_i⊙normed_i`.
  - `dG_i = (1/nr)·dX_in⊙normed_i` → through `Wu`/`silu`/`Wd` ⇒ `dWu,dWd`; `d(normed_i)` accumulates
    into branch grad; `branch_i` via RMSNorm grad (above).
- **GR write** (`gr.rs` 88–117): `s=2σ((1/nr)Ww·vec(R))`; `branch_i' = branch_i + s_i·y`.
  - `dY = Σ_i s_i·d(branch_i')`; `ds_i = Σ(d(branch_i')⊙y)` → through `Ww` ⇒ `dWw`; `vec(R)` grad
    (via branch_norms) **added** to the read branch grads. Branches are both in and out across the
    loop ⇒ sum read+write contributions each iteration.
- **Halt head** `lam=σ(W_h·halt_in)`: `dHalt_in = dLam·lam·(1−lam)·W_hᵀ`; `dW_h = halt_inᵀ·(dLam·lam·(1−lam))`.
- **PonderNet geometric recurrence** (`p_n = lam_n·Π_{j<n}(1−lam_j) = lam_n·π_{<n}`):
  - `lam` receives grad from three paths that all flow through `p_n`:
    1. `out_acc = Σ_n step_out_n·p_n` ⇒ `dLam_n += (dOut_acc·step_out_n)·π_{<n}`
    2. `rec = Σ_n p_n·CE_n` ⇒ `dLam_n += dRec·CE_n·π_{<n}`
    3. KL term below.
  - KL over `p_dist` (`model.rs` 113–136): `dKL/dp_n = log p_n − log prior_n + 1`. Because
    `p_n` depends on `lam_1..lam_n` only (`p_n` does NOT depend on `lam_m, m>n`):
    `∂p_n/∂lam_n = π_{<n}`; `∂p_n/∂lam_j (j<n) = −p_n/(1−lam_j)`. So in the reverse loop, given
    `dP_n = dKL/dp_n + dRec·CE_n + (dOut_acc·step_out_n)`, accumulate:
    `dLam_n += dP_n·π_{<n}`; and for each `j<n`: `dLam_j += dP_n·(−p_n/(1−lam_j))`.
- **Cross-entropy** `logits_n = lm_head·(step_out_n)` (fp32, loop_block 363–372):
  `dLogits_n = (softmax(logits_n) − onehot(target))·(dRec·p_n / b)`; backprop through `lm_head` and
  `step_out_n = out_proj·h`. Final-readout path: `logits = lm_head·norm(out_acc)` ⇒
  `dOut_acc = (∂logits/∂out_acc)` from final CE; both share `lm_head` weight grad (sum).
- **KDA** (`burn-kda`): gated-delta recurrence with decay `α_t`; state `s` recurs within chunk(16)
  and across iterations. Backward = BPTT through the δ/decay rule; port the existing backward math
  from burn-kda source (it trains today under autodiff, so the formulas exist). `dX` + `d(kda_state)`
  feed next iteration; `dW` for q/k/v/o + decay/β gates accumulated.
- **MSA** (`burn-msa`): top-k block selection + block-causal softmax; backward = gather/scatter of the
  selected block grads + softmax jacobian on selected blocks. Port from burn-msa source.
- **Engram** (RAM-offload `host_rows` or `hashed_ids`): `engram_a = engram.forward(embeds)·w_mem`;
  `dEmbeds` through Engram (Adam, wd off per optim.rs 76–89); `dw_mem = dEngram_a·h_ctx`.

## Integration with burn (obtain tensors, return grads)

**Obtain raw weights as cubecl tensors** (template: bf16_ops.rs 67–76):
- For each tracked weight param `P` (incl. `expert_ffns.*`, `out_proj`, `controller`, `halt_head`,
  `shared_attn.router`, `shared_attn.gdn2.*`, `shared_attn.msa.*`, `engram.*`, `lm_head`,
  `embedding`, `iter_embed`, `residual_scale`):
  `let p_ad = P.val().clone().try_into_primitive::<Autodiff<Inner>>().unwrap();`
  `let cube = Tensor::<D>::from_primitive::<Inner>(p_ad.primitive.clone());` → raw cubecl handle.
- Get the `ComputeClient<CudaRuntime>` once: `ComputeClient::<CudaRuntime>::load(device.as_dispatch())`
  (same unwrap as `lib.rs` 79–89 / 117). Launch kernels via `client.execute(..., vec![cube.handle,
  ...])`.
- Keep the `p_ad.node` id around (needed for `grads.register`).

**Return grads** (template: bf16_ops.rs 33–47 + lib.rs 494):
- In the op's `backward(self, ops, grads, checkpointer)`, for each weight compute its grad cube tensor,
  wrap `grad.try_into_primitive::<Inner>().unwrap()`, and call
  `grads.register::<Inner>(node_id, grad_primitive)` for every tracked parent.
- `loss.backward()` then returns the populated `Gradients`; `GradientsParams::from_grads(grads, &model)`
  and `optim.step(lr, model, grads)` are **unchanged** → Muon+ routing (`optim.rs` 65–89) keeps
  working because the param *paths* and *shapes* are identical; only the gradient *values* now come
  from our kernel.
- The op is a `pub struct PonderLoop; impl<B:Backend> Backward<B,?> for PonderLoop` mirroring
  `Bf16Matmul` (bf16_ops.rs 18–49); `forward` is a free fn that builds the inputs list (all weights
  + `x`, `hashed_ids`/`host_rows`, `targets`), runs `ponder_forward`, and wraps the returned
  `(logits, rec, p_dist, kda)` tensors as tracked outputs.

**Matmul in kernels:** use cubecl `cmma` (tensor-core MMA) on the patched `cubecl-fix`; follow the
existing fused `spectral_linear_fused` (burn-spectral lib.rs 481) for the launch/handle convention.

## Minimal incremental path / milestones

- **M0 — plumbing.** New crate/module `dormouse-core/src/fused.rs`. Implement the custom `Backward`
  op skeleton wrapping a *single* fp32 matmul (copy `bf16_matmul`) to prove: extract cube handles →
  launch kernel → `grads.register` → `optim.step` still runs. No behavior change.
- **M1 — PoC: one expert-FFN iteration, fwd+bwd, grad-checked.** With `max_iter=1` and
  `DM_NO_KDA`/`DM_NO_MSA`/`DM_NO_ENGRAM` set, fuse `gate_up→silu→down` (+ act quant STE) as
  `k_tsct` fwd/bwd. Gradient-check `dW`,`dX` vs burn reference (finite-diff or analytic-vs-burn) on
  `small` preset, rel < 1e-4 (fp32). This is the single highest-risk/value spike.
- **M2 — full single iteration.** Add controller (sigmoid+softmax), ReZero/GR residual, halt head +
  `p_n`/`out_acc`/`not_halted`, per-step CE, iter_embed. Fuse at `N=1`. Grad-check whole-loop grads
  vs burn (rel < 1e-4).
- **M3 — N iterations + loss.** Loop `ponder_forward`/`ponder_backward` over `N` (test N=4 then 48).
  Add final norm+lm_head and PonderNet KL (model.rs 102–136) inside the kernel so `loss` is one node.
  Grad-check vs burn for N up to 48, rel < 1e-4 (fp32) / < 1e-2 (BF16=1).
- **M4 — attention + engram arms.** Port KDA backward (M3.1) then MSA backward (M3.2) and Engram
  backward from their crates; re-enable the arms and grad-check with them on. Largest effort.
- **M5 — bf16 storage + bf16 tensor-core fast path.** bf16 workspace storage; optional `DM_FUSED_BF16`
  cmma path (tolerance loosened to < 1e-2, matching today). Verify 0 NaN over 100 steps (BF16=1).
- **M6 — wire into train loop.** Replace `model.forward_with_hidden::<Backend>(...)` + `model.loss`
  call (lib.rs 449–451) with the fused op; keep `loss.backward()`/`optim.step` identical. Keep
  `DM_NO_WARMUP` pool warmup. Benchmark GPU idle % (target: ~0 vs 74%).

## Critical Files

- `crates/dormouse-core/src/loop_block.rs` (129–381) — loop body to fuse; the spec.
- `crates/dormouse-core/src/model.rs` (44–136) — forward wrapper + PonderNet loss/KL to replicate.
- `crates/dormouse-core/src/param.rs` / `burn-spectral/src/lib.rs` (524–527) — TSCT factorization.
- `burn-spectral/src/bf16_ops.rs` (18–88) — the custom `Backward` op template to generalize.
- `crates/dormouse-train/src/lib.rs` (457–496) — grad-flow boundary we must keep working.

## Risks

- **sm_120 4D-slice crash:** only bites autodiff tensors. We manage a flat `bf16`/`f32` workspace
  buffer indexed by offset — no 4D autodiff tensor is ever materialized. Keep all slicing on 1D/2D
  cube handles (gr.rs 14–15 already notes this for the GR branch gate).
- **bf16 compute NaN (mixed bf16×fp32):** in raw cmma we control casts — cast activation→fp32 before
  every matmul, cast residual/GR writes back to bf16 (loop_block 341, gr.rs 114 rule). KDA/MSA/Engram
  stay fp32 internally (they are f32-only today, loop_block 267–269 / 298–299).
- **Memory for 48 saved iterations:** small preset ≈ 12×[b,t,d] + [b*t,f] per iter; b·t≈1536,d≈512,
  f≈4d ⇒ ~35 MB/iter fp32, ~1.7 GB for 48 iters; bf16 halves to ~0.85 GB. one_b preset needs budgeting
  — use checkpoint-recompute (Q5 fallback) if > ~8 GB. Pool is ExclusivePages (lib.rs 128–148); call
  `memory_cleanup` only on the log cadence, never right after a NaN (AGENTS.md).
- **KDA/MSA backward equivalence:** highest numeric risk — port exactly from burn-kda/burn-msa source
  and gate M4 behind a grad-check vs burn with the arm enabled. If it drifts, keep that arm on burn
  sub-ops (M2/M3 still fuse the dominant matmul mass).
- **Numerical equivalence vs current loss:** acceptance is tight (Q3). Any fused-op divergence breaks
  checkpoint compatibility. Mitigation: M1–M3 grad-checks must pass before M4; keep env switch
  `DM_FUSED=0` to fall back to the current burn path for A/B.
- **cubecl cmma API drift:** `cubecl-fix` is patched locally; mirror `spectral_linear_fused` launch
  convention rather than inventing a new client API.

## Success criteria

- [x] `ponder_forward`+`ponder_backward` run a full N=8 step on CUDA with NO per-op autodiff nodes
      (48 revoked per user decision 2026-09-13; random-depth T ∈ 1..8; N is a runtime arg - the
      host-side loop length IS the random-depth primitive). One PonderLoop op per step (M3).
- [x] Weight grads from the fused op match burn's reference within rel < 1e-4 fp32 on `small`
      (N ∈ {1,2,4,8}, arms off - documented per-tensor exceptions: x ≤ 5e-2, op./lm. ≤ 1e-3, both
      measured burn-reference noise floors; fused side is f64-exact per the buffer bisects).
      Arms-on gradcheck stays open for M4.
- [ ] `BF16=1` fused path: 0 NaN over ≥100 steps, loss within rel < 1e-2 of fp32 reference. (M5)
- [ ] `optim.step` (Muon+, `OPT=mix`) consumes the fused grads unchanged; training loss curve
      identical to `DM_FUSED=0` baseline within tolerance. (M6)
- [ ] GPU idle time per step drops from ~74% to < 10% (measured by nvidia-smi / step timer split,
      lib.rs 520–530). (M6)

## Acceptance (how verified)

- [ ] New test `fused_gradcheck` (NdArray not applicable — needs CUDA): builds `small` model, runs
      fused op backward, compares `grads` of every param path against `model.loss.backward()` from
      the current path; asserts max rel < 1e-4.
- [ ] `cargo check -p dormouse-core -p dormouse-train --features dormouse-train/cuda` clean.
- [ ] `DM_FUSED=0` (default-off) preserves today's behavior; `DM_FUSED=1` enables the kernel.

## Review (2026-09-07, M0+M1, two adversarial reviewers)

Math verified correct by both (independent gradcheck re-run: worst rel 1.89e-5 / 2.48e-5, f64 bisect ~1e-7). FIX-FIRST items, none blocking (DM_FUSED unreachable in production):
1. MAJOR fence hole: `fac_cuda` absmean launches (mod.rs:416-417) run before the fwd fence (mod.rs:567) — silent mu/mv corruption once wired into training (optim.step rewrites u/v every step). Move absmean after the sync.
2. MAJOR silent mode mismatch: forward hardcodes plain-ternary STE; training default is Fp8 factor quant — fused op would silently diverge. Extraction-time guard: refuse unless fp32/plain-ternary.
3. MAJOR hardcoded 30 parents (mod.rs:442, backward.rs:141/763) — one_b (n_experts=4) panics opaquely. Assert n_experts==3 or parametrize.
4. MINOR coverage: out_acc grad path never exercised with nonzero upstream — add loss+=oa.sum() test.
5. MINOR: assert max_iter==1 in M1 op; BWD_DUMP behind #[cfg(test)]; debug-assert out_features%4==0; pin model seed in tests.
6. MINOR perf (defer): 32-way redundant lane compute in ce/dlogits/halt elementwise kernels.
7. KDA/MSA notes (.bulba/fused_backprop_notes.md): adjoint kernels ALREADY EXIST in burn-gdn2/burn-msa — M4 is integration; fused loop MUST reproduce the KDA state gradient break (autodiff.rs:473) and MSA index-branch no-grad (KL discarded); plan.md stale on MSA pooling (max-pool r=32, no ReLU, d_idx=64).

## User decisions (2026-09-13, binding)
1. Language: RU preferred, EN/ZH mix allowed for capacity. No MiniCPM5 teacher — architecture stays self-sufficient (EMA self-JEPA only, no cross-model KD/OPD).
2. Engram placement: no dedicated A/B; keep in-loop for now (checkpoint compat), early-placement flag later if time permits.
3. Order of work: fully optimize code+architecture FIRST, then a ~100M model run. 1B deferred (maybe later).
4. max_iter 48 revoked (see UPDATE above); random-depth T∈1..8 is the loop target.

## Work queue (post-decision)
- W1 (agent, fused/): M3 — N-loop to N_max=8, random-depth sampling, KL, final readout, gradcheck vs burn. **DONE 2026-09-13, see M3 report below.**
- W2 (agent, offload.rs/optim.rs/aux.rs/train lib.rs): Sinkhorn-momentum optimizer for host n-gram tables (replace CPU Adam, ×1 buffer); head-wise Muon split for Q/K before orthogonalization; offline JEPA (precompute EMA-teacher targets per chunk, drop the second forward from the hot loop).
- Then M4 (KDA/MSA arms into fused op), M5 bf16 workspace, M6 wiring; then 100M run config.

## M3 report (2026-09-13, DONE)

Scope: fused op extended N=1 → N runtime iterations (target N_max=8, random-depth ready), PonderNet
recurrence + KL + final norm/lm_head readout in-op, backward reverse loop with the three λ grad
paths, weight-grad accumulation across iterations. KDA/MSA/Engram arms stay off (M4).

Verified on this machine (RTX 5060 Ti):
- `fused_gradcheck_niter`: N ∈ {2,4,8} vs burn reference, b=2 t=16, arms off. Forward: loss/rec
  rel 0.0, p_dist ≤ 2.9e-7, logits ≤ 2.9e-5. Worst per-path grads at N=8: op.v 4.67e-4, op.u
  3.12e-4 (documented burn-noise exceptions, limit 1e-3), everything else ≤ ~2.8e-4, most expert
  grads ≤ 1e-6. `fused_gradcheck_single_iteration` (N=1) same limits.
- `fused_bwd_buffers_vs_f64` (N=1 tiny + N=1 small-dims) and `fused_bwd_recurrence_vs_f64_n2`
  (N=2): every backward buffer + expert/op/lm weight grads match an f64 host truth at ≤1e-4
  (most 1e-7..1e-9). These are the exactness gates; the vs-burn gradcheck deltas are burn's own
  fp32 noise.
- `fused_matmul_matches_burn` (M0, seeded now - was draw-flaky), M0+M1 tests all green.
- `cargo check -p dormouse-core -p dormouse-train --features dormouse-train/cuda` clean.

Bugs found and fixed during M3 verification (all in fused/, none in loop_block/model):
1. p_dist output slice: flat region is iteration-major; `reshape` alone is only correct at N=1 -
   needs a transpose (caught at N=4, p_dist rel 2.6 while rec/kl matched).
2. lam_bwd read the POST-update nh buffer; the recurrence needs the nh ENTERING iteration n
   (per[n-1].nh / ones seed). PonderState now carries nh0.
3. Expert dV lost the ·s column scale in the M3 rewrite (M1 had col_scale) - restored post-loop.
4. lm-head accumulators (dul/dvl_raw/dsl) are pre-seeded by the final-readout contributions; the
   first reverse iteration overwrote the seed (accum=false) - now always accumulate.
5. Registration order: gf_at (final_norm_g) was pushed 6th instead of 30th in the parents vec -
   every expert/op/lm grad landed under the wrong node (burn read a 64-elem s-grad as [768,64]).
6. Registration now snapshots every grad to host bytes right after the reverse loop (verified-good
   point) and registers fresh from_data tensors - the registered-reshape readback was returning
   foreign bytes for expert slots. D2H+H2D cost ~5 MB/step; revisit if it shows in M6 profiles.
7. Test-truth fixes: f64 bisect inv/dz/dhaltpre formulas (missing .sqrt(), spurious p0 factor,
   missing halt-mean term, du_d used raw pre-activation instead of silu), dump reads now truncate
   to logical length (cubecl rounds allocations up to pow2 buckets).

VRAM: peak 2.4 GB during the N=8 gradcheck (b=2 t=16; ~1.1 GB of that is CUDA context/burn/JIT,
the test-shape workspace is ~35 MB). Projected workspace at b=10 s512 N=8 fp32 arms-off: ~4.4 GB
saved per-iteration buffers + ~1.5 GB backward temporaries ≈ 6 GB, leaving room for model +
optimizer in 16 GB; M4 arms and M5 bf16 storage budgeted separately (bf16 halves the workspace).

M4/M5 notes: KDA/MSA/Engram arms plug into the same parent-order scheme (6..17 experts, 24..26 op,
27..29 lm, 30 gf - gf MUST stay last, see the push-order fix); the backward's expert-dump/capture
pattern (read-to-host at the verified point) is the template for the new arms' grads.
