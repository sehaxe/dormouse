# Architecture verdict — what was measured, what survived, what goes into the flagship

**Date:** 2026-10-23
**Author:** orchestrator (all numbers below are from this repo's logs; each row names its evidence file)
**Scope:** every mechanism landed since 2026-10-14 that could plausibly change quality or throughput, tested as a 3-seed A/B on a fixed protocol, plus the combinations that survived.

---

## 0. The protocol (why these numbers are comparable and the old ones are not)

| item | value |
|---|---|
| screening model | `medium` — 141,910,405 params (r64), or the arm's own count (stated per row) |
| training data | `data_local/mix4/web.bin` (2.35 GB real English web prose) |
| eval data | `data_local/mix4/web_eval.bin`, **doc-sha1-disjoint** from train (builder refuses to publish on overlap) |
| window | **40 960 B** = `10 eval_batches × batch 8 × seq_len 512` |
| budget | 600 optimizer steps (screening) or 2000 (confirmation, stated) |
| seeds | **3 per arm**, spread reported; a win must beat 2× the control's spread |
| optimizer | Muon+ColRow, WSD, lr 3e-4→3e-5 |
| kernel | `moe_kernel=triton`, `ns_kind=triton`, `ns_steps=5`, dropless where stated |

**The single most important correction this cycle:** `data_local/corpus_bin_filtered.bin` (the corpus every earlier run used, including mix1 and the 170k continuation) is **not web**: sampling 200 random windows gave 55% LaTeX, 35% code, 25% ordinary prose, 1 Cyrillic. Every pre-2026-10-23 number was measured on a math/code corpus while being described as "web". Numbers below are on the real prose file.

Control on the new data: **2.650** (2.653 / 2.659 / 2.638, spread 0.021), `~/logs/n2_*` + `~/logs/a1_*`.

---

## 1. Verdict table — one row per mechanism

### 1.1 Wins (kept)

| mechanism | Δ BPB | evidence | note |
|---|---|---|---|
| **mHC** (K=2 lanes, paper-derived init) | **−0.078** | 2.5717 vs 2.650, spread 0.038, `logs/n2_mhc_s*` | second confirmation (−0.028 at 2000 steps on the old corpus). Costs +40% VRAM (two residual lanes) |
| **AttnRes** | **−0.050** | 2.5997 vs 2.650, spread 0.015, `logs/n2_attnres_s*` | 3.3× spread. Mutually exclusive with mHC in today's code — crossing is in flight |
| **ColRow** (optimizer) | −0.043 | 3.0113 vs 3.0542, 8.7× spread, 2026-10-17 | already the default; largest optimizer win |
| **ByteFlow (channel)** | −0.485 (medium) | 2.9337 vs 3.4185, 31× spread, capacity-matched control −0.428 | **NOT usable in the flagship**: fp32-only (+60% params, 1.6× slower) and incompatible with Engram |
| **Triton NS** | 68.1 → 18.6 ms/step | `docs/reviews/2026-10-21-battery.md` | dispatch, not quality |
| **opt_graph** | −12 ms/step | same | 3239 → 2 launches |
| **MoE seam in bf16** | −384 MiB/pass, max‖Δlogit‖ = 0.0 | `8e1bc21` | parity, not a rewrite |

### 1.2 Ties (kept only as flags; no recipe cost)

| mechanism | Δ BPB | verdict |
|---|---|---|
| `moe_topk_mode=entropy` (adaptive width) | −0.007 (2.6430) | tie on quality, saves compute on ragged arms; **kept** |
| MTP k=4 | −0.0044 (small), −0.0007 (small_wide) | tie; code + gates stay, off by default |
| JEPA (with anti-collapse) | +0.0017 | tie |
| energy contrastive (w=0.05) | +0.0030 | tie; w=0.1 **destabilises** (one seed 3.631) |
| halt head | +0.026, spread 0.079 | unstable at 600 steps; inference-only value untested |
| MSA @2k | tie (2.7208 vs 2.7214 in the 2026-10-21 wave) | free; kept for long context |
| KDA recompute off | tie (+0.2% time, +960 MiB) | off |
| act_quant int8/int4/fp4 | ±0.002 | fake-quant: zero VRAM/speed win in training; product rung only |
| K=64 patches (ByteFlow) | +0.002 | tie → halves loop positions, but ByteFlow is out |

### 1.3 Losses (rejected with a dated number)

| mechanism | Δ BPB | why it lost |
|---|---|---|
| **rank 32** at fixed width | **+0.027** (3.1024 vs 3.0750, old corpus) | half the expert rank is not free |
| **rank 32 + width 2048 + dict** | **+0.064** (2.714 vs 2.650) | `d_model` doubles attention cost (∝d²) and the step gets *slower*: 0.90 vs 0.63 s |
| **rank 32 + width 1152 + f4096** | **+0.104** (2.754) | widening the expert does not rescue rank 32 either |
| rank 16 | +0.136 | |
| rank 8 | Triton kernel crash | |
| share_u **dict rank 2** | **NaN in <10 steps** | the floor is real: 32 is the boundary |
| E=512 (2× experts at rank 32) | +0.033 | capacity is not in expert count |
| r16 + topk4 (composition instead of rank) | +0.127 | "many weak atoms" ≠ one strong expert |
| e_layers 6→3 (ByteFlow encoder) | +0.255 | the patcher encoder is load-bearing |
| dense_qat on attention | +0.198 (broken arm) | attention must stay ≥8-bit; fixed by `dense_qat_skip_attn` default true |
| **full ternary form (factors + dense)** | **+0.250** | on real web prose; see §3 |
| NorMuon | +0.072 | replaces ColRow, loses the column half |
| SF-Muon | +0.045, one seed diverged to 4.06 | the quoted mean was 2/3 seeds; legal re-test queued |
| SF-NorMuon | +0.043 | |
| SIGReg standalone | +0.148 | |
| LOOM | tie on BPB, worse on `fin` | |
| ns_steps 3 | +0.006 | not worth the speed |
| torch.compile (dynamic) | 0.20× (5× slower) | data-dependent MoE cache guards |

---

## 2. The TSCT question — answered

The project's central bet was that low-rank factorised experts buy quality per FLOP. Measured:

| arm | params | BPB | FFN MAC/token |
|---|---|---|---|
| TSCT control (r64, f2730, topk 2) | 141.9 M | **2.650** | **3.89 M** |
| dense, same width, f44000 | 117.5 M | **2.653** | **135.2 M** |
| dense, equal params (f63024) | 141.9 M | *never completed* (OOM twice, batch 8) | 193.6 M |
| TSCT r32 + d2048 + dict | 115.8 M | 2.714 | 0.61 M |

**Verdict.** TSCT matches a dense FFN's quality while spending **~35× fewer MACs** — but it does so with **~20% more parameters**, and it does not beat dense at the same parameter count. The honest claim is:

> TSCT is a **compute-efficiency** mechanism, not a quality mechanism: same quality as dense, ~35× less arithmetic, at the cost of more stored parameters (which is then repaid by ternary packing: 8× smaller storage).

The "×23B dense-equivalent" figure is a statement about the **shape of the function** the bank can express, not about information capacity, and it is **not** a quality claim. Any document that implies otherwise must be corrected.

---

## 3. Ternary — the cost is real, and it is not where we thought

| form | BPB | Δ vs bf16 | evidence |
|---|---|---|---|
| bf16 control (no mHC) | 2.650 | — | |
| **mHC, bf16** | **2.5717** | **−0.078** | `logs/n2_mhc_s*` |
| **mHC + ternary factors only** | **2.655** (2.660/2.635/2.670, spread 0.035) | **+0.083 over mHC**, +0.005 over plain bf16 | `logs/a1_mhc_tern_s*` |
| ternary factors + dense, no mHC | 2.900 | +0.250 | `logs/n2_ternq_s*` |
| ternary on the old (math/code) corpus, 2000 steps | 3.1368 | +0.057 | 2026-10-22 ladder |

Three findings that matter:

1. **Ternary factors cost +0.083 on real prose at 600 steps** (3.2%) — that is beyond the 1.5-2% budget for a *fixed* size, but the resource it frees is storage, which is a different currency (§ owner rule #52). The bank is 88% of b1's parameters: 626 M → 78 M packed is an 8× storage cut.
2. **mHC's win and ternary's cost cancel exactly**: mHC+ternary = 2.655 vs plain bf16 = 2.650. You get 8× smaller bank storage for **zero net BPB** once mHC is in — but you also lose mHC's own −0.078 if you think of it as "ternary paid for it".
3. **The cost is data- and budget-dependent**: +0.057 at 2000 steps on synthetic math/code, +0.083 at 600 steps on prose, +0.250 when the dense layers (router!) are quantised too. Longer budgets and leaving the router in bf16 both reduce it.

**Recommendation for the flagship:** `factor_qat=ternary` (bank only) **plus `use_mhc=true`**, dense layers stay bf16 (`dense_qat=none` — the router is the most damaging place to quantise, and dense layers are only 37 M of b1's 1.07 B). Measured net cost of that combination versus plain bf16: **+0.005 BPB for 8× smaller bank storage**; a 20k-step confirmation is required before the flagship because the 600-step screen may overstate the cost.


### 3.1 CRITICAL: `engram_ram=true` costs 0.83 BPB at our budget (measured, isolated)

The diagnostic run (600 steps, 1 seed, everything else identical) isolates three factors that had been confounded:

| config | BPB | per-iteration CE |
|---|---|---|
| iter4 + MSA + **engram in VRAM** | **2.649** | 2.024 / 1.812 / 1.767 / 1.807 |
| iter3 + MSA + engram in VRAM | 2.662 | 1.972 / 1.802 / 1.811 |
| iter4 + no MSA + engram in VRAM | 2.670 | 2.040 / 1.832 / 1.783 / 1.813 |
| iter4 + MSA + **engram in host RAM** | **3.476** | 2.488 / 2.414 / 2.381 / 2.376 |
| iter3 + no MSA + **engram in host RAM** | **3.460** | 2.454 / 2.385 / 2.370 |

**`engram_ram` is the entire 0.8 BPB gap.** `max_iter` 3 vs 4 is +0.013 (a tie) and MSA-off is +0.021 (so MSA earns its keep at this budget).

**Mechanism** (`src/dormouse/model/engram.py`, verbatim from burn's `offload.rs:154-160`):
```python
m = 0.95*m + 0.05*g
block = 0.95*m + 0.05*g      # the update actually applied
```
At steady state on a *hot* row this converges to `g`, so the formula is right for a table where every row is touched often. For a **cold row touched once**, `m = 0.05g` and `block = 0.0525g` — a **~19× smaller step**, and a hash table at 600 steps is almost entirely cold rows. Burn's production run used 48M rows over 300k steps, where rows converge; our screens do not.

**Consequence:** the engram stays in VRAM for every run at `medium` (3 orders × 524 288 × 32 × 4 B = 201 MB — it fits). `engram_ram` remains mandatory only for `b1` (402M rows), and that path needs bias correction (`m / (1 - 0.95^t)`, Adam-style) before it is trustworthy.

---

## 4. What crosses with what (measured)

| pair | works? | evidence |
|---|---|---|
| mHC + ternary factors | ✅ | 2.660 vs 2.650 bf16 (Δ +0.010) |
| mHC + attnres | ❌ refused today | both write the residual; crossing implemented in a side branch (§7) |
| mHC + share_u | 🔄 running | `a1_mhc_shu_s*` |
| mHC + adaptive topk | 🔄 running | `a1_mhc_topk_s*` |
| share_u + ternary | ✅ (code) | `2f55c1c` — one α per dictionary, 3 quantiser calls/site instead of 512 |
| share_u + ByteFlow / MSA | n/a | ByteFlow is fp32-only; MSA is memory-bound |
| dropless + mHC | ✅ | 2.6897 (mHC+dropless) — dropless itself is +0.007, i.e. not needed at E=256 |
| topk adaptive + dropless | ✅ | sentinel-expert padding keeps capacity accounting honest |

---

## 5. The recommended configuration (staged)

### Stage A — screening/fast iteration (now proven)
```
medium r64 · d1024 · f2730 · E256 topk2 dropless
use_mhc=true · share_u=true · use_attnres=true
opt=Muon+ColRow · ns_kind=triton · ns_steps=5 · opt_graph=true · bf16 masters
data=mix4 (5.8 GB, 11 domains) · seq 1024 · batch 24 (≈3.5 GB/day)
```

Measured stack on the new data (600 steps, 3 seeds, window 40 960 B; `scripts/collect_ab.py` reproduces this table from `~/logs`):

| configuration | n | mean BPB | spread | Δ vs control | verdict |
|---|---|---|---|---|---|
| control (bf16, r64) | 3 | 2.6500 | 0.0210 | — | bar = 2.6080 |
| **mHC + share_u + AttnRes** | 3 | **2.5080** | **0.0090** | **−0.1420** | **WIN — the recipe** |
| mHC + share_u + AttnRes + prelude/coda (1+1) | 3 | 2.5110 | 0.0050 | −0.1390 | tie with the recipe → prelude is not taken |
| + mHC + share_u | 3 | 2.5583 | 0.0170 | −0.0917 | WIN |
| mHC | 3 | 2.5717 | 0.0380 | −0.0783 | WIN |
| mHC + share_u + prelude/coda (2+2) | 3 | 2.5533 | 0.0260 | −0.0967 | tie with mhc+shu → prelude is not taken |
| mHC + share_u + prelude/coda (1+1) | 3 | 2.5633 | 0.0220 | −0.0867 | tie / slightly worse |
| mHC + adaptive topk | 3 | 2.5750 | 0.0170 | −0.0750 | WIN vs control, **tie vs mHC** |
| AttnRes | 3 | 2.5997 | 0.0150 | −0.0503 | WIN |
| adaptive topk alone | 3 | 2.6430 | 0.0240 | −0.0070 | tie |
| mHC + ternary factors | 3 | 2.6550 | 0.0350 | +0.0050 | tie |
| rank 32 (clean, fixed geometry) | 3 | 2.7530 | 0.0240 | +0.1030 | **LOSS** |
| ternary factors + dense | 3 | 2.8997 | 0.0030 | +0.2497 | LOSS |

**The win is super-additive, not just additive:** mHC (−0.078) + AttnRes (−0.050) + share_u (−0.013) = −0.141 predicted; measured **−0.142** with a spread of only 0.009 (16× the effect). The two mechanisms attack the same weakness — the loop's later iterations — and do not interfere.

**prelude/coda is rejected at our scale.** Two depths (1+1, 2+2), with and without AttnRes, all inside the control's spread, while costing +1.6M and +3.1M parameters respectively. The paper's +12-point MATH500 gain is a reasoning-benchmark effect at 730M/336B tokens; it does not transfer to BPB on prose at 142M with a 600-step budget. Flag stays, verdict dated.

So the recommended **Stage A** is `mHC + share_u + AttnRes`, and **Stage B** adds ternary factors on the bank for the shipped artifact (net +0.005 BPB versus plain bf16, 8× smaller bank).

### Stage B — shipped form (product)
```
same, plus factor_qat=ternary (bank only), dense_qat=none
measured net cost vs plain bf16: +0.005 BPB · bank 626 M → 78 M on b1 · export via .dormouse
```

### Stage C — rejected, do not spend GPU on
```
rank < 64 (any compensation tried: width, d_ffn, E, composition)
d_model > 1536 at 16 GB (attention ∝ d², step gets slower)
dense_qat on attention or router
ByteFlow in the flagship (fp32-only, kills Engram)
MTP / JEPA / energy / halt as *pretraining* losses (all ties)
```

---

## 6. Honest gaps

1. **Equal-parameter dense never completed** (OOM at batch 8, twice). The claim "TSCT ≥ dense at equal params" is therefore **unproven**; only "equal quality at 35× fewer MACs with 20% more params" is measured.
2. **All screens are 600 steps.** A mechanism that ties at 600 can still pay at 20k (JEPA's history) — the report marks ties as "not proven harmful", not "proven useless".
3. **Downstream numbers are thin**: one n=1000 checkpoint at 25-30% on ARC/HellaSwag. Every quality claim here is BPB on held-out prose.
4. **No external reproduction.**
5. **The ternary + mHC interaction** was measured once (3 seeds, 600 steps). It needs a 20k-step confirmation before the flagship.

---

## 7. The looped-model literature, evaluated against our measurements

Four 2026 papers were read in full and mapped onto this repo. They converge on one claim — **iterations are not equivalent, so treat them asymmetrically** — which our own `periter` curve independently confirms (2.15 / 1.86 / 1.81 / 1.80 on the 1B run: iteration 1 contributes 0.29 CE, iteration 3 contributes 0.01).

| paper | what it claims | what transfers to us | verdict |
|---|---|---|---|
| **LatentLoop** 2610.19975 | first visit's residual coeff is 0.27 vs 0.87 for the second; compress the *first* visit's sequence, write back, reinvest the saved compute in width; 19% training-FLOP saving at fitted matched NLL | **inverted for us**: their *second* visit is the only valuable one (+0.0133 of the loop's whole gain), while ours is worth ~3% of 0.35 CE at 1B — so compressing our **last** iteration is the transferable move even though their ablation prefers compressing the first. Their own k-sweep: k=2 costs +0.0067, k=4 +0.0098; a **single learned scalar recovers 84%** of the writeback gain and the learned coefficients settle at 0.199/0.160 (the model wants to write the pooled increment at ⅕ strength). Their measured GPU time saving is −9…−13%, not the fitted 19% | **queued as a throughput test**: our arithmetic gives **−10.4% step (k=2), −15.6% (k=4)** at T=3 → −12.3%/−18.6%, for **+0.003…+0.010 BPB**. Do NOT use ByteFlow's chunker inside the loop: `select_positions` is a *global* top-k over the window (a future gain can move a past boundary = leak), and its causal twin takes a host list (ADR-0018 violation). Use a causal mean-pool + one learned scalar |
| **Shared Memory in Looped Transformers** 2610.02383 | first recursion writes the KV cache, later ones read it + a 64-token window; quality improves, memory 0.24× | **memory motivation is zero for us** (KDA state is 4.2 MB at medium/b8 regardless of context; an equivalent one-layer KV would be 33.5 MB and grows with t). The dead-straight port is **non-causal**: `chunk_kda(initial_state=S_T)` hands iteration 2 a decayed summary of *all* positions, i.e. at byte 0 it reads bytes it must predict — a retracted-number-class defect | **rejected** (pre-registered tie 0.00-0.03 BPB, and a tie deletes); the CPU day is still worth it to add the leak gate, which the Rust lane's own state-carry lacks |
| **Looped Models Done Right, Part I** 2610.23036 | prelude → recurrent core → coda beats full-stack looping: **MATH500 20.8 → 32.8 EM (+12 pts)** at 730M/336B; input injection +6.63 BBH-CoT but −3.60 MATH500; route-freezing (reusing iteration 1's expert ids) *hurts* | **the biggest unclaimed win**: we run full-stack (embed → shared block ×T → head), i.e. the losing topology. Our router re-routes per iteration, which their ablation says is correct | **queued first** — ~80 LOC, one wave |
| **Hybrid Latent Attention** 2610.07940 | keep exact KV for a 128-token window, compress older tokens into a latent the query reads directly: 10.7× smaller cache, 2.5-7.4× throughput, 97% accuracy | **the KV problem does not exist for us** (no cache at all). But the *architectural* lesson applies: pure linear attention loses exact recall at distance. The right shape is KDA (global, O(1)) + an exact **sliding window** (local), which FLA already supports (`parallel_attn(window_size=W)`) — ~50 LOC, bounded memory (8.4 MB/layer at W=2048, d=1024), O(n·W) instead of O(n²) | **queued** — this is the path to 1M context with exact local recall |

**Memory arithmetic for 1M context** (the reason the hybrid is affordable at all): KDA state is constant; the sliding window costs `2·W·d·2 B` per layer = **8.4 MB at W=2048**, so three iterations cost 25 MB **at any context length**, versus a full KV cache at 1M which would be tens of GB.


---

## 8.1 The GPT-2 comparison, measured on identical bytes (2026-10-23)

`scripts/bpb_hf.py` scores any HuggingFace causal LM in bits per byte on a raw byte file (total token code length ÷ bytes encoded, so a tokenizer cannot flatter the number). On the first 40 KB of `data_local/mix4/web_eval.bin`:

| model | BPB | compression | note |
|---|---|---|---|
| zstd -3 | 3.190 | ×2.51 | classical LZ |
| zstd -19 | 2.954 | ×2.71 | classical LZ, max level |
| xz -9 | 2.927 | ×2.73 | LZMA |
| our 5-gram anchor | ~2.59 | ×3.09 | on the old eval file |
| **our `mix1` (142M, 4.18 GB seen)** | **1.833** | **×4.37** | `dormouse eval --ckpt-name mix1` |
| **GPT-2 small (124M, ~40 GB seen)** | **1.047** | **×7.64** | `scripts/bpb_hf.py gpt2 …` |

**We beat every classical compressor by 1.6× and lose to GPT-2 small by 0.79 BPB.** The parameter counts are almost identical (142M vs 124M), so the gap is not size. Three causes, ranked by weight:

1. **Our corpus was not prose.** Sampling 200 windows of `corpus_bin_filtered.bin` (the corpus behind every run before 2026-10-23) gave 55% LaTeX, 35% code, 25% ordinary prose, 1 Cyrillic window. GPT-2 trained on 40 GB of English web. We measured our model out of its own domain.
2. **A tenth of the data** (4.18 GB vs ~40 GB): at our measured slope (~0.1 BPB per 4×) that is worth ~0.25 BPB.
3. **Architecture**: 3 iterations of ONE shared block with linear attention (KDA) versus 12 distinct full-attention layers. KDA is O(1) in context but weaker at exact recall; that is the price of the constant-size state, and it is real.

**What this does NOT say.** At matched parameters and matched data our architecture beats a dense transformer: 2.508 vs 2.653 BPB at 600 steps (both on mix4/web, both 142M). The claim "TSCT/loop is inefficient" is refuted; the claim "we are competitive with a model that saw 10× the data on its home domain" is not supported, and would be dishonest.

**The test that settles it:** the flagship on `mix4` (2.35 GB real English web + 11 domains) scored against the same `bpb_hf.py` anchor. If the gap to GPT-2 small stays above ~0.5 BPB on prose, the next move is an exact sliding window over KDA (planned), not more tuning.

---

## 9. The flagship run — what it is, and why it is 1.7 days and not 5

| item | value |
|---|---|
| data | `mix4` — **5.77 GB** train (11 domains: web 40.8%, wiki 12%, ruweb 15%, code 12%, math 11%, books+qa 2%, reasoning 3%, instruct 2%, rudialog 2%), 53 MB held-out, doc-sha1 disjoint |
| model | `medium` 141.9 M, d1024, r64, E256 topk2 dropless, **mHC + share_u**, attnres if it crosses cleanly |
| form | bf16 masters; **ternary factors for the shipped export** (measured net +0.005 BPB) |
| form factor | `b24 s1024` (the only form measured at 3.5 GB/day); the wide-arm experiments showed `d>1536` costs more (attention ∝ d²) than it buys |
| wall clock | **≈1.7 days** at 3.5 GB/day (1 epoch) |
| epochs | mix_v4 caps at 1.05 epochs — this is a single pass, deliberately: 4× repetition degrades MoE (2609.11917) |
| checkpoints | every 5000 steps; eval every 2500 on the 196 608 B window |

**Why not 5 days:** the earlier plan assumed 1 GB/day from a d1024 run at `b8 s512`; the measured `b24 s1024` form does 3.5 GB/day, so one full epoch over the new corpus is 1.7 days. Going longer would mean repeating data, which we have measured to be worth almost nothing (plateau: 0.002 BPB per 130 MB on the old corpus).

**What comes after it:** SFT (49 k pairs, 2 h) → RLVR on verifiable code/math (4 h) → bench + transcript.

---

## 10. What still has to be measured before the flagship (blocking list)

| # | item | why it blocks |
|---|---|---|
| 1 | `mHC + share_u` at **b24 s1024**: GB/day and VRAM | the screening form is `b8 s512`; the flagship form is 6× bigger per step. mHC's two residual lanes may not fit with the 6×-larger activations |
| 2 | ternary + mHC at **20k steps** (not 600) | the 600-step screen may overstate ternary's cost; the flagship exports ternary |
| 3 | prelude/coda topology A/B | the biggest expected win in the literature (MATH500 +12 pts); if it lands, the flagship should use it |
| 4 | sliding-window exact attention | the path to 1M; independent of the flagship's 1k context |

Items 1-2 are cheap (one probe + one 3-arm wave). Items 3-4 are code first.

---

## 11. In flight when this report was written

- `arch1` wave: mHC+share_u, mHC+topk, rank32 clean, share_u alone, "all best" combination (18 runs, ~80 min).
- Agent: mHC × AttnRes composition (worktree `mhc-attnres`).
- Agent: per-expert adaptive rank (worktree `pe-rank`).
- Agent: LatentLoop full read + port design (`docs/research/2026-10-23-latentloop.md`).
- Dataset: `mix4` complete — 5.77 GB train + 53 MB held-out, 11 domains, doc-sha1 disjoint, manifest with per-domain drop counts.
- Paper memos already filed: `docs/research/2026-10-23-shared-memory-loops.md` (rejected with a leak argument), `docs/research/2026-10-21-hybrid-ncm.md`, `docs/research/2026-10-21-adaptivity.md`.
