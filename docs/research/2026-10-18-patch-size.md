# Optimal patch size for dynamic-patching byte LMs — what the record actually measures

**Date:** 2026-10-18. Read-only: one file created, no code touched, nothing committed, no GPU.
**Ask:** what patch size should our byteflow patcher target — is 4-8 bytes right, does it
scale with `d_global`, and what do `byteflow_eps2` / `byteflow_k_tokens` control?
**Builds on, does not repeat:** `docs/reviews/patching-research-2026-10-02.md` (architecture
choice; BLT's *scheme* ablation; SpaceByte Table 1) · `docs/research/2026-10-16-jepa-dspark-alternatives.md`
(2610.11790) · `docs/reviews/byteflow-rate-2026-10-02.md` (our 4× vs the paper's 2.56×) ·
`docs/research/2026-10-18-data-capacity.md`. This pass is about **size**, not whether to patch.
**Method:** arXiv MCP only, ~25 calls, no 429s, every ID resolved individually; repo numbers
name their file. The brief's `2026-09-26-byte-lm-recipes.md` **does not exist on disk**.
**[A]** = authors' abstract · **[F]** = paper full text, read this session · **[ours]** = our
measurement · **[S]** = my inference.

## Verdict table (claim → source → what it measured)

| # | claim | source | what it measured |
|---|---|---|---|
| 1 | Dynamic byte LMs run **~4-6 bytes/patch**; the flagship 8B BLT-Entropy is **4.5** | 2412.09871 **[F]** | 8B, 4.5T bytes: avg 4.5 B/patch, avg task 61.1 vs Llama-3 8B (4.4 B/token) 60.0; BLT-Space **6.1** → 58.0 (Table 1) |
| 2 | BLT swept **ps = 6 and 8 only**; no patch-size ablation; **ps=8 loses at 1B, wins at 7B** | 2412.09871 **[F]** | FLOP-equivalent pairs at fixed inference FLOPs: 470m → ps6 610m / ps8 760m; 3.6B → 5.2B / 6.6B (Table 2); "ps 8 starts … worse … at 1B but ends up better than bpe at 7B" (§5.1); `search_paper_text` "patch size ablation" → 0 passages |
| 3 | Quality falls **monotonically** as patches grow at fixed model | 2605.09630 **[F]** | NLU avg p=4 **52.9**, p=8 **52.2**, p=16 **48.0** (byte baseline 54.1); code pass@1 28.1 / 24.1 / 18.2 (Tables 1-2) |
| 4 | ByteFlow's own sweep: best at the **lowest** compression, graceful to 5.12× | 2603.03583 **[F]** | 600M/50B: 2.0× **51.74**, 2.56× (default) 50.89, 3.41× 50.13, 5.12× **48.48**; 1.3B/500B: 64.44 → 60.91 |
| 5 | MegaByte (fixed stride): **P=8 for text**; at matched compute, **shift capacity local → global** wins | 2305.07185 **[F]** | PG19: patch 8, ctx 8192, global 758M / local 262M; audio P=32, Image64 P=12; 350M/290M → 1.014, 760M/262M → 1.002, 1.3B/218M → **0.991** (Table 10) |
| 6 | H-Net: 1-stage realizes **BPIC ≈ 4.7-4.8**, 2-stage (3,3) ≈ **7**, and **multi-stage beats one stage at the same FLOPs** | 2507.07955 **[F]** | Large: 2-stage bpb 0.743 / acc 55.5 vs 1-stage 0.755 / 53.6; XL: 0.715 / 58.2 vs 0.728 / 56.2 (Tables 1-2) |
| 7 | Larger patches are bought back by **intra-patch compute** (SP), never for free | 2605.09630 **[F]** | p=16+SP ≈ byte baseline (54.2 vs 54.1) with **3-4× less inference compute** and 16× smaller patch KV; p=8+SP 53.8 < p=4+SP 56.2 |
| 8 | Entropy patching at a **10-byte budget** collapses on computed values | 2610.11790 **[A]** | BLT-1B: 10% patch starts (≈10 B/patch, **[S]**) → 19.0% exact on GSM8K computed results; 26% (≈3.85 B/patch) → **76.8%**; survives LoRA (32.9 vs 72.7) |
| 9 | A **compression-ratio curriculum** (low → high) helps — slightly | 2605.30080 **[A]** | H-Net 350M, FineWeb-Edu 100B: adaptive N 8.0→9.5, avg 47.3 vs fixed 46.2-47.0; robustness 29.0 vs 28.8-28.9; val BPB ≈0.828 all |
| 10 | The byte-level encoder+decoder **doesn't shrink** with patch size, so the measured saving is **well under the nominal p×** | 2412.09871 + 2605.09630 **[F]** | "larger patch sizes only affect flops from the patch Latent Transformer"; 20× params (400M→8B) ⇒ local params ~2×; seq reduction vs measured FLOPs/byte: p=4 4.0×/3.1×, p=8 8.0×/4.5×, p=16 16×/5.7× |
| 11 | The light-encoder/heavy-decoder split is the published choice | 2412.09871 **[F]** | Table 9: 1 encoder + 9 decoder layers 0.822 bpb vs 5+5 0.844 (with n-gram hashes) / 0.843 (without) |
| 12 | We already ran **4 B/patch** (K=128 over 512 B) and won the 200-step smoke | [ours] | `configs/byteflow.toml` (byteflow_9m); 3.466 vs plain 3.947 equal-params, full-arm `small` loses 6.498 (`benches/history.tsv:157-159` via `docs/decisions/2026-10-15-1b-readiness-plan.md:21`); 4.82/4.90 vs 6.71/6.60 at eval@100, ~65 vs ~450 ms/step (`byteflow-ab-2026-10-02.md`) |

## Q1 — what sizes do published patchers use and measure?

The live band is **4-6 bytes/patch**: BLT-Entropy **4.5** and BLT-Space **6.1** (2412.09871
Table 1); SpaceByte's spacelike rule ≈ **6** on PG-19, ≈ **8** on Github (2404.14408, read in
`patching-research-2026-10-02.md` §1.3; Scratchpad measures it at **6.3**); H-Net 1-stage
realizes **4.8** (2507.07955 Table 1); Scratchpad's best-with-SP is **p=4** (2605.09630).
Our own byteflow_9m A/B ran **4 B/patch** [ours]. BLT's "patch-size ablation" as such **does
not exist**: `search_paper_text` for "patch size ablation" returns 0 passages, and §7 ablates
cross-attention, n-gram hashes and encoder/decoder depth. What they picked: **4.5 bytes for
the 8B flagship**, 6/8 for the study — the flagship is *finer* than the study's optimum
**[S: the flagship is FLOP-matched and 4.5 buys more patch steps; the paper does not explain
it]**.

## Q2 — do LARGE patches (32-64 B) pay for text, and does the optimum scale with capacity?

**No source found** for 32-64 B on text. The largest measured text setting anywhere in this
pass is Scratchpad's **p=16**: the worst vanilla point (NLU 48.0, code 18.2), recovered only
to baseline by SP (54.2), and still below p=4+SP (56.2). MegaByte's P=32 is **audio**, P=192
**images** (2305.07185 §7/§6.3) — no text setting above 16 B/patch exists in this pass. The
2610.11790 "10% of bytes" budget (≈10 B/patch, **[S]**) is the closest text datapoint and it
collapses (19.0% vs 76.8%).
**Capacity scaling is real but is a claim about the global model, not text granularity:** BLT
ps=8 overtakes ps=6 by 7B (2412.09871 §5.1), and at fixed inference FLOPs ps=8 buys a
**1.7×** latent model vs ps=6's **1.3×** (Table 2) — the bigger global model is where the
freed FLOPs go (MegaByte Table 10, row 5). Neither paper sweeps patch size against `d_global`
jointly, so "optimal patch size scales with `d_global`" is **[S]** — two scaling studies
support it, ByteFlow's flat penalty is the counterweight: the compression cost did **not**
shrink 600M → 1.3B (3.26 vs 3.53 pts, 2603.03583 Table 6).

## Q3 — the information-bottleneck argument, and a measured quality curve

The argument in print is cost, not bits-per-dimension: patch size is "the main factor for
determining the cost" because the global model runs once per patch (2412.09871 §3), and
MegaByte's >98% of FLOPs sit in per-position FFNs, so P× fewer global positions funds a P×
larger global model (2305.07185 §2.1, via `patching-research-2026-10-02.md`). No paper
normalizes by `d_global`; **no bytes-per-latent-dimension curve exists**. The nearest formal
statement is ByteFlow's rate-distortion derivation `R_ε(h) = ½ logdet(I + (d/ε²)·HHᵀ)`
(2603.03583 App. A, eq. 11) — ε² is the reconstruction noise variance, the scale below which
a new direction is not worth a boundary. **Measured quality-vs-size curves exist**, all
sloping down at fixed model: 2605.09630 (p=4/8/16), 2603.03583 Table 6/E.2 (2.0×-5.12×),
2412.09871 Table 1 (4.5 vs 6.1), 2610.11790 (3.85 vs 10 B/patch). The only upward moves change
something else too: a bigger global model (BLT), scratchpads (SP), or hierarchy (H-Net).

## Q4 — does hierarchical patching dominate single-level large patches?

At matched FLOPs and data, **yes for H-Net**: 2-stage (3,3) beats 1-stage at both scales
(bpb 0.743/0.715 vs 0.755/0.728; acc 55.5/58.2 vs 53.6/56.2, 2507.07955 Table 2), and the
abstract claims parity with a token transformer **twice** its size. AU-Net extends the idea
to 3+ stages with schedules like (16, 8) at 1M context and (1000, 200, 25) at 5M
(2502.14553 Table A2). H-Net's own negative: fixed-width pooling is "much worse than all
other H-Net variants" — the win is learned boundaries, not depth. Counter-evidence: SP
**degrades** H-Net under strict FLOPs matching (2605.09630 §5.1), so "hierarchy dominates"
holds for H-Net's downsampling-with-DC, not universally.

## Q5 — how much of the nominal p× saving survives?

Structurally, only the latent transformer's share scales as 1/p (row 10). Measured, in the
one paper that reports both columns side by side (2605.09630): sequence reduction
4.0×/8.0×/16.0× against **measured FLOPs/byte reduction 3.1×/4.5×/5.7×** at p=4/8/16 — i.e.
**~78%/56%/36% of the nominal saving survives** [S: the division is mine; the two columns are
theirs]. BLT's layer ablation agrees prescriptively (row 11): make the encoder cheap, because
it is paid per byte. On our box the wall-clock share is smaller again: the 200-step smoke ran
~65 vs ~450 ms/step at equal bytes (`byteflow-ab-2026-10-02.md`), but the step is launch-bound
(AGENTS.md §3.1: 13.3% util, GEMMs ≈16% of a warm step), so a FLOP saving buys less wall clock
than its FLOP fraction — the arithmetic `2026-10-16-token-skipping.md` already did for depth.

## Q6 — patch-size curricula / tunable threshold schedules (2026)

**Yes, one, and it is a compression-ratio curriculum, not a size curriculum:** ATDC
(2605.30080) ramps the target N from 8.0 to 9.5 over training on H-Net 350M / FineWeb-Edu
100B, beating fixed N=7/8/9 by +0.3-1.1 points avg accuracy (47.3 vs 46.2-47.0) and +0.1-0.2
robustness, at essentially identical BPB (≈0.828); the authors call it "more stable training
dynamics", not a quality jump. Adjacent, not curricula: BLT retunes the entropy **threshold
at inference** (0.6 → 0.1) to trade steps for task accuracy (2412.09871 §5.2), and 2610.22100
reports that **switching boundary rules after training did not work**. Nothing found
schedules patch size itself or `ε²`.

## What this means for our knobs (`byteflow_eps2`, `byteflow_k_tokens`)

Read from the port, not assumed: `byteflow.py:519` selects positions with
`marginal_gains_exact(h, eps2)` **only** under `byteflow_logdet`; the default L2 path calls
`marginal_gains_l2(h)` and **never reads eps2** — so on every preset we ship, `eps2` is inert
and compression is set entirely by **K** (`select_positions` is plain top-(K−1) over ΔR,
`byteflow.py:408-424`). ByteFlow itself **rejected a threshold** in favour of fixed top-K, for
the reason our eps2 is a trap: a global threshold is a "magic number" whose realized sequence
length varies per input and breaks the static graph (2603.03583 §4.2). So: **`eps2` deserves
no seed set; `K` is the knob.** Our operating points: byteflow_9m ran **K=128 / 512 B = 4
B/patch** [ours]; the port default is **K=3200 / 8192 B = 2.56 B/patch** (`config.py:268-290`)
— finer than anything published (BLT 4.5, H-Net 4.8, SpaceByte 6.0-6.3). A patch-size sweep
should move K **down** (bigger patches), never up. The polygon and instruments exist:
`port/scripts/ab_byteflow.py` (unrun 3-seed driver at K=128/512; emits `event=patch`
mean/median/p90/histogram per run) and `port/scripts/patch_rate.py` (CPU probe, `--k` flag;
mean = T/K by construction; random-init caveat in its docstring — note `ab_byteflow.py` cites
it as `scripts/patch_rate.py`, the repo's one dead path reference, though the file exists).

**Pre-registered predictions for a patch-size sweep** (K ∈ {200, 128, 96, 64, 32} over the
512-byte window = 2.56 / 4 / 5.33 / 8 / 16 B/patch; fixed bytes/step, fixed window, 3 seeds,
and BLT's own fairness rule — "we ensure that the number of bytes in each batch remains
constant … to avoid giving larger patch sizes unfair advantage", 2412.09871 §4.3):
1. **4 vs 2.56 B/patch:** BPB within the 3-seed spread, either direction. The curve is flat
   at the fine end in every source (ByteFlow 2.0× vs 2.56× = 0.85 pts at 600M), and at 9.2M
   the spread eats effects this size. If 2.56 wins by more than the spread, suspect **loop
   positions per byte** (more K = more compute), not patch quality.
2. **8 B/patch: predicted loss, ΔBPB ≥ +0.05** vs 4 B/patch. Basis: Scratchpad p=4→8 costs
   0.7 NLU points and 4.0 MBPP points; BLT's ps=8 is worse at 1B.
3. **16 B/patch: predicted loss, ΔBPB ≥ +0.15**, and no win. The only published way to hold
   quality at p=16 is SP's extra intra-patch compute — a new mechanism, not a K change.
4. **If any large-K arm wins BPB, check the confound first:** BLT's large-patch wins all spend
   the freed FLOPs on a bigger global model (Table 2: 1.7×), which the 9.2M polygon cannot
   express as a K sweep; SP's wins are extra compute inside the patch.
5. **Hierarchy is the supported alternative** if large patches are wanted for context length:
   H-Net's 2-stage result (bpb 0.715 vs 0.728 at XL) is the evidence to buy.

## VERIFIED vs SPECULATION

**VERIFIED.** All IDs and dates resolved through the arXiv MCP this session; all quoted
numbers are **[A]** or **[F]** as marked. Own-repo: `configs/byteflow.toml` (K=128,
max_bytes=512, eps2=0.5), `port/src/dormouse/config.py:268-290` (K=3200, max_bytes=8192,
eps2=0.5), `port/src/dormouse/model/byteflow.py:519` and `:408-424`,
`port/scripts/ab_byteflow.py` + `port/scripts/patch_rate.py`,
`docs/reviews/byteflow-ab-2026-10-02.md`, `docs/decisions/2026-10-15-1b-readiness-plan.md:21`
(3.466/3.947/6.498 via `benches/history.tsv:157-159`), `AGENTS.md` §3.1.

**SPECULATION (mine).** **S1** "optimal patch size scales with `d_global`" — from BLT's
1B-vs-7B crossover and its FLOP-matched pairs; no paper sweeps the two jointly. **S2** the
~78/56/36% arithmetic in Q5 — my division of the paper's two columns. **S3** every prediction
at 9.2M. **S4** that 2.56 B/patch is "finer than needed" — the band is 4-6 B/patch but BLT's
flagship is 4.5 while its study favours 8, so the band's width is not tight.

**Confidence.** **HIGH** on: the 4-6 B/patch band; the monotone decline at fixed model; `eps2`
inert on our default path; the encoder/decoder not scaling with patch size. **MED** on:
capacity-scaling (two studies, neither at our scale; ByteFlow's counter-trend). **LOW** on:
any number at 9.2M, the 32-64 B question (no source), and the curve between 4 and 16 B/patch.

## IDs resolved this session (all individually, no 429s)

**Resolved:** `2412.09871` (BLT, 2024-12-13) · `2404.14408` (SpaceByte, 2024-04-22) ·
`2305.07185` (MegaByte, 2023-05-12) · `2507.07955` (H-Net, 2025-07-10) · `2603.03583`
(ByteFlow, 2026-03-03) · `2605.09630` (Scratchpad Patching, 2026-05-10) · `2610.11790`
(boundary dependence, 2026-10-08) · `2610.22100` (novelty patching, 2026-10-15) ·
`2502.14553` (AU-Net / MBLM, 2025-02-20) · `2605.30080` (ATDC, 2026-05-28) · `2605.08044`
(Fast BLT, 2026-05-08) · `2608.03599` (boundaries, 2026-08-04).
**Failed to resolve:** `2404.07974` — my first guess for SpaceByte; it resolves to a quant-ph
matchgate paper. `2305.07185` returned "not found in storage" on the first `search_paper_text`
call (needed `download_paper` first) — a tool-state error, not a bad ID.
**Not found at all (named, not guessed):** a BLT patch-size ablation; any 32-64 B/patch text
result; any bytes-per-latent-dimension curve; any patch-size curriculum.

*Owed: a line in `docs-site/tools/manifest.mjs` — this task forbade touching other files.*
