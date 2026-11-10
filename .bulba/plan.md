# PLAN — gated FFN in the live block + the K3 stabilisers (2026-11-10)

Owner: "делай все". Four parallel lanes, all DEFAULT OFF, no preset used by an
A/B on record may move.

## Why
The live `block_form="transformer"` block is a plain `gelu(fc(x))@fc2^T`
(`model/transformer.py:276`), `d_ffn = 4·d`. Every frontier 2026 model is gated
(LLaMA/Qwen SwiGLU; Kimi K3 SiTU-GLU). At matched width a GLU is FREE:
`f = 8d/3` costs `3·d·f = 8d²` = exactly the `2·d·4d` of the GELU block.
Measured params: 5,537,792 → 5,591,040 per layer (+0.96% at align-64).

## Acceptance criteria (per task)

### T1 — gated FFN in `TransformerBlock` (Lane A) — DONE, commit 96509cd
- `ffn_gated=true` + `expert_act ∈ {silu, situ}` builds gate/up/down and runs
  `down(expert_glu(act)(gate(x), up(x)))`; `ffn_gated=false` path unchanged.
- Param parity vs the `4d` GELU block within 2%, else LOUD refusal.
- LOUD refusals: not `block_form="transformer"`; `expert_act` outside
  `{silu, situ}`; combined with `fused_glu`.
- Tests (CPU): param parity; SiTU anchor computed from the PAPER formula
  `β1·tanh(x/β1)·σ(x) ⊙ β2·tanh(u/β2)` at x∈{0,1,4,10,-10} in the TEST (never a
  golden captured from our own run); bound `|f| ≤ β1·β2 = 100`; near-origin
  ≈ SiLU (|x|<1e-2, 1e-2 rel); a gate that FAILS if the activation is swapped
  back to SiLU (situ(10)=3.946 vs silu(10)=9.9995); fp32 parity vs a hand-written
  reference GLU ≤ 1e-5.

### T2 — per-head Muon (Lane B) — DONE, commit 6826bf6
- Flag `muon_per_head`; `n_heads == 1` bit-identical to the full-matrix step
  (`torch.equal`), and the split of one block counts nothing.
- NS split on the attention Q/K/V projections (`q_proj`/`k_proj`/`v_proj`, and
  the packed `qkv` of `block_form="transformer"` as `3·n_heads` blocks); the tail
  after the polar is the full-matrix one, so nothing else moves.
- Measured on one `[512, 64]` momentum with head 3 at 10×: per-head update norm
  spread **2.1%** (< 5%) against the full matrix's **223%** (> 50%).
- LOUD refusals: `opt="adamw"` (no Muon), no `n_heads`/`head_dim`, a head axis
  the split cannot tile (GQA `k_proj`), and a flag that marks no projection.
  COUNTED `muon_per_head`, once per chunk.
- Tests (CPU): `port/tests/test_muon_per_head.py`, 13 tests.

### T3 — Quantile Balancing (Lane C) — DONE, commit 3978c7b
- `moe_qb=true` sets each expert's bias from the router-score quantile matching
  its target load; no `.item()`, no host sync in the step path; COUNTED.
- Tests: synthetic scores → every expert's load within ±10% of target after 50
  updates; already-balanced loads leave the bias unchanged; `n_experts == 1`
  refused LOUD.

### T4 — MoE step/VRAM bench (Lane D)
- `scripts/bench_moe_package.py`: 20 steps, real corpus slice, ms/step + peak
  VRAM for (a) dense 95M control, (b) bank with `n_experts=8` sized so total
  ≈299M (active ≈100M), at b24 and b16. Report whether b24 fits under 16 GiB.
- **Numbers only, no architecture change**: latent routing does not exist yet;
  the bench measures what the current bank does and says so in the report.

### T5 — `ffn_form="tsct"` reaches the live `TransformerBlock` (Lane A) — DONE
- `ffn_form="tsct"` + `block_form="transformer"` + `ffn_gated=true` builds the
  gated FFN's three matrices as `TSCTLinear` (`U[d,r]·diag(s)·V[r,f]`, ranks
  `ffn_rank_gate`/`ffn_rank_down`, `0` = `rank`), COUNTED as `ffn_tsct`.
- The defect it closes: `configs/arch_tsct275.toml` loaded at **273,393,792**
  parameters — bit-for-bit the dense control, ZERO `TSCTLinear` modules —
  because the form only reached the KDA loop. Wired, the arm is
  **110,052,864** (−163,340,928 = 18 × 9,074,496 per layer).
- `validate_block_form` refuses every form the block does not build (`dense`,
  and any future literal; `bank` is the field's DEFAULT, not an arm) and
  refuses `tsct` without `ffn_gated`, with `n_experts != 1`/`moe_topk`, with
  `act_qat`/`dense_qat`, and on the hybrid stack.
- Exactness: at `r = min(d, f)` the factorised GLU IS the dense GLU — SVD the
  dense weights into the factors and the logits agree to **9.5e-07** (fp32).
  Flag OFF is bit-identical to the `b7c2f72` fixture (names, RNG, logits).
- FLOPs/token (projections + FFN, 18 layers): **545.4M dense vs 218.8M tsct**;
  the FFN alone 19.69M → 1.54M FLOPs/token (12.8×). SDPA is identical on both
  arms (+4·t·d MAC/token/layer: 84.9M at t=1024).
- Tests: `port/tests/test_tsct_transformer.py` (15, CPU).

### T6 — AttnRes full-depth scoring is memory-flat (2026-11-10) — DONE
- `use_attnres=true` (N = L) OOMed at b24, b20 AND b16 (at b16 it asked for
  48 MiB). Cause: `full_depth_attend` stacked every source into `[T, n, d]` —
  keys + their query product 936 MiB each (1.87 GiB live at the score step, and
  again at the value step; the whole call allocates 7.58 GiB of churn) against
  a 468 MiB bf16 history. The arm had NO verdict.
- Now one `[T]` score column per source, written into a `[T, S]` fp32 matrix
  (1,277,952 B = 1.22 MiB at S=13), and `a += w_j·u_j` in place: the binding
  tensor is the fp32 `[T, d]` RMSNorm/accumulation transient, 78.0 MiB each,
  ≤3 live at any instant (234 MiB) plus the 78 MiB accumulator.
- Math unchanged, MEASURED bit-for-bit: the loop equals the stack expression at
  `max|Δ| = 0.0` for S = 2..7 (fp32, three prior/query sets) — the gate is the
  ≤1e-6 assertion against the stack written out in the test.
- Identity at init still exactly `0.0` (tightened from `< 1e-5` to `== 0.0`);
  order sensitivity still fails a fixed average; no host sync; dynamo traces the
  loop into 1 graph with 0 breaks and a bit-identical compiled call (so the
  compiled A/B is not silently eager — the gate fails LOUD if it ever is).
- CPU probe at the live shape (`/tmp/opencode/attnres_peak.py`, old vs new in
  separate processes): largest single allocation 936.0 MiB → 78.0 MiB, CPU wall
  694 ms → 505 ms. Not a GPU measurement: b24 is now *plausible* (the old path
  held 1.87 GiB of transient live where the new holds ≤312 MiB — the 234 MiB
  scoring peak plus the 78 MiB accumulator), it is not yet shown to fit.
- Tests: `port/tests/test_attnres_full.py` (10, CPU).

## A/B protocol (after T1 lands, GPU is free)
120 MB / 120M bytes, 3 seeds, arms: `control` (GELU 4d) vs `gated_silu`
(2.67d) vs `gated_situ` (2.67d). KEEP a gated arm only if it is ≥0.01 BPB better
than the control on 3 seeds; otherwise the flag stays off and the result is
recorded as a measured refusal. Single-seed noise on record: 0.008–0.011 BPB.

## Rules for every lane
- Never `git add .`; commit only the files your task names.
- `config.py` is Lane A's file; B/C/D read the flags with `getattr(cfg, ..., False)`.
- No GPU job without `/tmp/dormouse-gpu.lock`; no shelling around the guard.
- Report numbers with the command that produced them; a claim without one is not
  a result.
