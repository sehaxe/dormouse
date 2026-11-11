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

### T7 — tsct FFN init is matched to the dense gated arm (2026-11-11) — DONE
- Defect (adversarial review, reproduced): `TransformerBlock.__init__` skips
  the dense `nn.init.normal_` for the tsct branch and substituted nothing, so
  `TSCTLinear`'s unit-scale scheme (`u ~ N(0,1/√in)`, `s = 1`,
  `v ~ N(0,1/√rank)`, the KDA loop's contract) initialised the 275M arm. At the
  live shape the FFN write was **13.3x** the control's — the A/B would have
  measured init scale, not rank-64 factorisation.
- Fix: `TSCTLinear(weight_std=σ)` gives the composed `U·diag(s)·V` entry std
  exactly `σ` (`u ~ N(0,1/√in)`, `s = 1`, `v ~ N(0, σ·√(in/rank))`, since
  `√r·(1/√in)·σ√(in/r) = σ`). The block passes the dense arm's own numbers:
  `0.02` for gate/up, `0.02/√(2L)` for the write. Default `None` keeps the
  old scheme bit for bit (same draw shape, same RNG order; the default path
  still DIVIDES by `√rank`, a multiply would have moved the low bits and
  failed the four hash-pinned tsct/off gates).
- Measured (fp32, `configs/arch_{dense,tsct}275.toml`, b2×t16, seed 1234,
  `/tmp/opencode/probe_init.py`):

  | | dense | tsct before | tsct after |
  |---|---|---|---|
  | FFN write std | 0.0458 | 0.6101 (13.32x) | 0.0458 (1.00x) |
  | last-block out std | 1.0253 | 2.7064 (2.64x) | 1.0257 (1.00x) |

- GATE (fails without the fix): `test_tsct_init_matches_the_dense_gated_arm`
  builds both presets fresh and asserts the pooled FFN write std and the
  last-block output std are within 5% (pooled over the stack: one block's
  `64×2848` draw alone swings ±8% on sampling noise, three seeds 0.999–1.024x).
- Nothing else moved: flag OFF still bit-identical to
  `tests/fixtures/transformer_off_prechange.pt`; full-rank reparameterisation
  still `9.5e-07`; `test_tsct2`/`test_tsct_packed`/`test_ffn_form`/
  `test_block_form` bit-identical gates green.

### T8 — `attn_form="tsct"` factorises the attention projections (2026-11-11) — DONE
- `attn_form ∈ {"dense", "tsct"}` (default `dense`) + `attn_rank` (`0` =
  `rank`): the packed `qkv [3d, d]` becomes ONE `TSCTLinear`
  `U[d,r]·diag(s)·V[r,3d]` and `proj` `U[d,r]·diag(s)·V[r,d]` — the largest
  FLOP lever the FFN cut leaves (attention projections are 27% of the block's
  MAC; SDPA's 24% is quadratic and not rank-shaped).
- Init matched to the dense arm through `weight_std` (0.02 on `qkv`, `0.02/√(2L)`
  on `proj`). Measured at the live shape (`arch_dense275`, ONE field changed,
  seed 1234, b2×t8): qkv output std 1.000x, proj 1.000x, last-block output
  1.004x. Without it the arm's composed entry std is 1.0, i.e. 50x the dense
  write.
- Exact at `r = d`: SVD of the dense `qkv`/`proj` → logits agree to **9.5e-07**
  (fp32).
- Params/MAC per layer at d=1152: dense 5,308,416 / 5,308,416 MC; r=64 →
  **442,496 / 442,368** (12.0x); r=256 → 1,769,984 / 1,769,472 (3.0x). 18
  layers: −87,586,560 / −63,691,776 params. Block MAC/token (t=1024): 19.87M
  dense; 10.80M with the FFN factorised only (1.84x); **5.93M with both at
  r=64 (3.35x)**; 9.56M at r=256 (2.08x). The brief's 3.27x/1.96x assumed
  `8·d·r` (three separate q/k/v cuts); the FUSED map costs `4·d·r`, hence the
  higher ceiling.
- `TSCTLinear` returns fp32; the block widens the projection back to the
  compute dtype, or a bf16 arm would run SDPA (4.72M MC/token, quadratic) on
  the fp32 lane and spend the cut it just bought.
- LOUD refusals, at config load AND at the block's `__init__`: any
  `block_form != "transformer"` (and the hybrid stack),
  `attn_latent_rank`, `fused_latent_proj`, `act_qat`, `dense_qat`, `attn_rank`
  above `d_model`, and `attn_rank` set with `attn_form="dense"`.
- Flag OFF is bit-identical to `tests/fixtures/transformer_off_prechange.pt`.
- Tests: `port/tests/test_attn_form.py` (21, CPU).
- Known interaction, not fixed here: `muon_per_head=true` refuses LOUDLY with
  this arm — `mark_per_head` reads `.qkv.weight`, so it sees `qkv.u` `[d, r]`
  and fails the head-axis tile check ("its head axis is 12 block(s) x
  head_dim=16 = 192 rows"), never silently splitting nothing.

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
