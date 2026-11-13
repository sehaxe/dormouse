# AMD, US-market and non-NVIDIA compute for a CUDA-only training stack — October 2026

**Date:** 2026-10-30 · **Status:** research memo, read-only · **Author:** research agent
**Question:** buy compute for local LLM training late Oct 2026. Workload pins (a) CUDA Triton MoE grouped-GEMM with hardcoded `sm_120` shared-memory ceilings and tile tables, (b) `flash-linear-attention` (FLA) 0.5.2 `chunk_kda`, (c) bf16 tensor cores, (d) ≥24 GB VRAM.

---

## TL;DR

1. **No AMD card beats 2× RTX 3090 or 1× 4080 Super on total cost for this stack.** On raw peak-price it is close (R9700 ≈ $19.6/useful-TFLOPS vs 2×3090 ≈ $30.7), but a realistic port is **6–10 engineer-weeks**, which at any sane rate costs more than the hardware delta. AMD wins only if the kernels were already Triton-generic and the `sm_120` tile tables did not exist.
2. **The premise "FLA 0.5.2 is CUDA-only" is wrong.** FLA v0.5.2's `pyproject.toml` already ships a `[rocm]` extra; v0.6.0 (2026-10-23) added first-class backend extras and an Ascend NPU backend. What is true: FLA's AMD CI job (`amd-mi300.yml`) is **disabled by default** ("fla does not yet operate an amd-mi300 runner"), and parts of KDA are gated `IS_NVIDIA or IS_NPU`.
3. **MI50 (32 GB, $600–800) is a trap:** `gfx906` is marked ❌ *Unsupported* in ROCm 10.1.0.
4. **US used prices are 15–25 % below Minsk**, but after shipping + 15 % over-limit duty the saving on one card is ≈ $50–450, before the legal question of exporting controlled GPUs to Belarus. Not worth it for 1–2 cards.
5. **The 2×3090 budget in the brief is stale.** At Oct-2026 US used prices a 3090 is ~$1,199, so 2× = ~$2,400, not $1,600–2,000.
6. **Non-NVIDIA exotica (Ascend/Cambricon/Moore Threads) is worse, not better**, for a Triton-heavy single-node trainer: no retail channel to Belarus, 5–10× the price per useful TFLOPS, and a software stack (CANN/MUSA) that is a second port on top of the first.

---

## Evidence access notes (read this before trusting any number)

| What | Status |
|---|---|
| eBay raw **sold** listings | **Not accessible** — HTTP 403 to this host and to three CORS relays. All "eBay used" numbers below come from third-party trackers that publish their method, or from search snippets of eBay pages. |
| bestvaluegpu.com | Cloudflare-blocked to direct fetch **and** to a real browser. Numbers quoted only from search-result snippets dated Oct 2026. Treated as *lead*, corroborated where possible. |
| Newegg / B&H / Micro Center / Amazon product pages | 403/503. Retail availability is inferred from a tracker that ingests Newegg via Rakuten Advertising API and Amazon daily. |
| ROCm docs, PyTorch wheel index, GitHub, arXiv | Read directly, primary. |
| Kufar.by, NBRB FX | Read directly, primary. |

---

# PART A — AMD, honest feasibility

## A1–A3. Cards, prices, ROCm status, throughput

Prices are US street as of 2026-10-29/30 unless stated. "bf16 peak" = vendor dense peak for FP16/BF16 tensor path.

| Card | VRAM | Price now (source, date) | bf16/FP16 peak | FP8 | FP4 | ROCm 10.1.0 | Triton/FLA reality |
|---|---|---|---|---|---|---|---|
| RX 7900 XTX | 24 GB | **$1,099** used (bestvaluegpu snippet, Oct 2026); **$1,149.99** eBay comparison (videocardprices, 2026-10-30) | 123 TFLOPS (flopper, "Official") | ✗ | ✗ | ✅ gfx1100 | Triton AMD backend works; FLA CI does not run on RDNA3 |
| RX 9070 XT | **16 GB** | $879.99 Newegg (videocardprices, 2026-10-30) | ~97 (not verified) | ✓ (RDNA4) | ✗ | ✅ gfx1201 | fails the 24 GB requirement — excluded |
| Radeon AI PRO R9700 | 32 GB | **$1,699.99** Newegg (Pangoly, 2026-10-19); street $1,300–1,700 | **191 TFLOPS** | **383 TFLOPS** | ✗ | ✅ gfx1201 | newest RDNA4 target; least battle-tested |
| Radeon PRO W6800 | 32 GB | $993–1,460 eBay asks (Oct 2026); "used from $1,300" (GPUDojo) | **36 TFLOPS** | ✗ | ✗ | ✅ gfx1030 | too slow to matter for training |
| Radeon PRO W7800 | 32 GB | $1,600–1,799 eBay (Oct 2026); launch $2,499 | **90 TFLOPS** | ✗ | ✗ | ✅ gfx1100 | poor $/TFLOPS |
| Instinct MI100 | 32 GB | $700–1,240 eBay asks, typical ~$1,000–1,200 (Oct 2026) | **92 TFLOPS** bf16 / 185 FP16 | ✗ | ✗ | ✅ gfx908 (CDNA1) | old; Triton support exists but low priority upstream |
| Instinct MI210 | 64 GB | **$4,369** eBay used; $9,999 Newegg new; €9,619+ EU retail | ~181 TFLOPS (vendor figure, not in tracker) | ✗ | ✗ | ✅ gfx90a | the only sane "big VRAM" AMD option; still 4× the price of a 3090 |
| Instinct MI50 | 32 GB | $608–750 (AliExpress/eBay, Oct 2026) | ~26–53 FP16 (unverified) | ✗ | ✗ | **❌ unsupported** | dead end |
| Instinct MI300X | 192 GB | $22,000 used / $27,500 (eBay, Oct 2026) | ~1,307 (derived: FP8/2) | **2,615 TFLOPS** | ✗ (FP4 is MI350) | ✅ gfx942 | OAM module, needs a server platform; not a desktop purchase |

**ROCm status (verified):** ROCm Core SDK **10.1.0**, release date **2026-10-05**. Officially supported list includes gfx1100 (7900 XTX, W7800), gfx1201 (9070 XT, R9700), gfx1030 (W6800), gfx908 (MI100), gfx90a (MI210/MI250), gfx942 (MI300X/MI325X). `MI50/MI25/Radeon VII (gfx906) = ❌ Unsupported`: "prebuilt ROCm libraries are not officially supported and will cause runtime errors."
**PyTorch:** stable ROCm wheels exist for torch **2.14.0 / 2.14.1 / 2.15.0** under `rocm7.14` (download.pytorch.org). `triton-rocm` wheels exist for **3.6.0 → 3.9.0**.
**Triton upstream:** v3.9.0 released **2026-10-27**; README lists "AMD GPUs (ROCm 6.2+)" as a supported platform. Independent evidence that custom Triton kernels compile and validate on ROCm: AMDKernelVault (arXiv 2609.12471, 2026-09-11) built a corpus of **39,893 Triton kernels** compiled and latency-profiled under ROCm on CDNA GPUs.
**FLA:** `[rocm]` extra exists since ≤0.5.2; v0.6.0 (2026-10-23) documents the two-step install (`--index-url .../rocm7.2 torch`, then `flash-linear-attention[rocm]`). Caveats that matter for you: the MI300 CI workflow is **disabled** ("Disabled by default: fla does not yet operate an amd-mi300 runner"), it targets **MI300/CDNA3**, not RDNA3/RDNA4, and KDA tests skip some paths with "Indexed KDA requires CUDA or NPU".

## A4. What actually breaks, and what it costs

Concrete breakage classes for your stack (analysis, anchored to the sources above):

1. **`sm_120` constants.** Shared-memory ceilings, `num_stages`, tile tables and `maxnreg` are compiled-in for Blackwell. On gfx1100 the LDS budget per workgroup differs and the occupancy model differs; every hardcoded table must be re-derived and re-autotuned. This is mechanical but touches every MoE kernel.
2. **Warp width 64 vs 32.** AMD wavefronts are 64 lanes. Any kernel that reasons about `tl.arange`/reductions per 32 lanes, or hardcodes `num_warps` for a 32-wide warp, is wrong-by-construction and needs re-tuning (not just recompiling).
3. **PTX-only primitives.** `cp.async`, `ldmatrix`, `wgmma`, TMA, `tl.inline_asm_elementwise` with PTX — none exist on AMD; Triton's AMD backend lowers to `buffer_load`/MFMA and has no TMA equivalent on RDNA. Kernels written around async pipelining lose their overlap.
4. **Grouped GEMM / MoE routing.** Dynamic per-expert shapes + atomics is the least mature corner of any non-NVIDIA backend. Expect to write a persistent-kernel variant rather than port.
5. **FLA `chunk_kda`.** Best case: install `[rocm]` and it runs. Worst case: the `0.5.2` pin pulls CUDA-flavoured internals; you either bump to 0.6.x (API drift) or vendor the kernels.
6. **Numerics/validation.** bf16 accumulate ordering differs; you need a re-validation pass on loss curves, not just unit tests.

**Effort estimate (SPECULATION — no public benchmark of this exact port exists):**
| Work item | Weeks (1 senior engineer) |
|---|---|
| Re-derive sm_120 tile/SMEM tables for gfx1100 or gfx1201, re-autotune | 1.5–2.5 |
| MoE grouped GEMM rewrite for AMD backend | 2–4 |
| FLA KDA: upgrade to 0.6.x + fix fallout, or vendor kernels | 1–2 |
| Correctness + perf validation, CI | 2–3 |
| **Total** | **6–10 weeks**, plus permanent divergence maintenance |

External anchor for the order of magnitude: Meta's Oct-2026 paper on AMD training (arXiv 2610.22230) states that "the extensive history of NVIDIA-specific optimizations hinders performance portability", that a *directly ported* baseline needed model/kernel co-design plus custom shims to reach **1.5×** improvement, and that work was then upstreamed into PyTorch/Triton/ROCm. They had a hyperscaler team. You have one stack and no team.

## A5. Verdict — dollars per useful TFLOPS

Assumptions (explicit): useful = peak × efficiency; **efficiency 55 %** for NVIDIA CUDA/Triton (mature kernels, large GEMM), **40 %** for AMD after porting (weaker backend, no TMA). These factors are ESTIMATES.

| Option | Cost (Oct 2026) | Peak bf16 | Useful TFLOPS | **$/useful-TFLOPS** | Port cost | 24 GB+ |
|---|---|---|---|---|---|---|
| 2× RTX 3090 | $2,398 (2×$1,199) | 142 | 78 | **$30.7** | $0 | 48 GB ✓ |
| 1× RTX 4080 Super | $1,179 | 104 | 57 | **$20.6** | $0 | 16 GB ✗ |
| RX 7900 XTX | $1,149 | 123 | 49 | $23.4 | 6–10 wk | 24 GB ✓ |
| Radeon AI PRO R9700 | $1,500 | 191 | 76 | **$19.6** | 6–10 wk | 32 GB ✓ |
| Instinct MI100 | $1,000 | 92 | 32 | $31.0 | 4–8 wk | 32 GB ✓ |
| Instinct MI210 | $4,369 | 181 | 81 | $53.6 | 4–8 wk | 64 GB ✓ |
| W6800 / W7800 | $1,200 / $1,700 | 36 / 90 | 14 / 36 | $83 / $47 | 6–10 wk | 32 GB ✓ |
| MI300X | $22,000 | ~1,307 | 588 | $37.4 | 8–12 wk | 192 GB ✓ |

**Answer to "is there ANY AMD card that beats 4080 Super or 2×3090 on $/useful-TFLOPS for a CUDA-only stack?"**
- On *hardware only*: yes, marginally — R9700 at ~$19.6 vs 4080S $20.6, and it has 32 GB. But the 4080S fails the VRAM requirement anyway, so the real comparison is 2×3090 at $30.7.
- On *total cost of ownership*: **no.** Port at 6–10 weeks × even $20/h × 30 h/wk = $3,600–6,000, i.e. 2.4–4× the price of the card. Break-even requires ~150+ weeks of runtime at the $11/TFLOPS delta, or a second identical machine.
- **Recommendation:** 2×3090 (or 1×3090 + 1×4090 if VRAM per card matters) on CUDA. Revisit AMD only when (a) FLA's AMD CI is actually running, and (b) your tile tables are generated, not hardcoded.

---

# PART B — US market as an import source

## B1. Used prices (US, Oct 2026)

Two independent trackers, both rebuilt 2026-10-30, both measuring **asking** prices, not cleared sales:

| Card | bestvaluegpu "used on eBay today" (Oct 29–30) | DealHunter 30-day median asking (2026-10-30) | Verdict band |
|---|---|---|---|
| RTX 3090 24 GB | **$1,199** (+22 % vs 12-mo avg $984) | not tracked | $1,150–1,400 |
| RTX 3090 Ti | **$1,699** (+45 % vs avg $1,174) | not tracked | $1,500–1,750 |
| RTX 4080 16 GB | **$1,084** (+28 % vs avg $848) | **$1,149** (150 listings, 92 % eBay) | $1,050–1,150 |
| RTX 4080 Super | **$1,179** (+28 % vs avg $922) | not tracked | $1,050–1,250 |
| RTX 4090 | **$2,999** (+31 % vs avg $2,285) | **$3,199** avg $3,250 (73 listings) | $3,000–3,200 |
| RTX A6000 48 GB | — | — | **low confidence**: $2,600–3,800 (Aug-2026 secondary), $3,998 (PC Server & Parts, 2026-09-05), $4,999 eBay ask |
| RTX A5000 24 GB | — | — | **low confidence**: $1,500–1,900 (r/homelabsales, ~Jul 2026), $2,300 eBay BIN (2026-10-28) |

Individual eBay asks seen in snippets (same day): 3090 $1,060–1,987; 4080S $1,042–1,245; 3090 Ti $1,060–2,060. The spread is why medians, not asks, are quoted.

## B2. US retail: is any 24 GB card in stock at a sane price?

No, by the evidence available:

| Signal | Value | Source/date |
|---|---|---|
| RTX 5090 street | **$6,300** used / **$7,443.99** Amazon (+215 % over $1,999 MSRP) | videocardprices, 2026-10-30 |
| RTX 5080 street | **$1,875** used / $1,899.99 Newegg (+88 % over $999 MSRP) | videocardprices, 2026-10-30 |
| 5090 supply | board partner "cannot buy GB202 and 32 GB GDDR7 kits, even on backorder"; no EOL notice | MIXED, 2026-10-27, citing der8auer 2026-10-25 |
| 5090 status | Nvidia denies discontinuation; "most listings are over $9,300"; rumour of an RTX 5080 24 GB Super replacement | TechSpot headline, 2026-10-19 |
| 4080S / 4090 new | out of production; resale only | Spheron, 2026-10-20 |
| 24 GB workstations | RTX PRO 4000 Blackwell 24 GB exists (145 W, single slot) — **no street price verified** | videocardprices workstation inventory, 2026-07-13 |

**Conclusion:** the only 24 GB consumer-class cards are used 3090/4090. New 24 GB means workstation SKUs at ≥$2,300 (A5000) and usually far more.

## B3. 2026 price trend and forecasts

| Datapoint | Value | Source/date |
|---|---|---|
| RTX 4090 30-day median asking | **$2,499 (2026-08-20) → $3,199 (2026-10-30)** = **+28 % in 10 weeks** | DealHunter daily series, 2026-10-30 |
| RTX 4090 vs 12-mo average | +31 % | bestvaluegpu, Oct 2026 |
| RTX 3090 / 3090 Ti / 4080S vs 12-mo avg | +22 % / +45 % / +28 % | bestvaluegpu, Oct 2026 |
| Composite index (22 cards w/ MSRP) | 96.5 = 3.5 % below launch MSRP on average — skewed by old low-end cards | videocardprices, 2026-10-30 |
| DRAM/NAND contract prices | could rise **70–140 % in 2027** | TrendForce, 2026-08-25 |
| HBM blended price | **+121 % YoY** forecast | EE Times Asia, 2026-10-06 |
| Memory shortage duration | supply shortage expected to persist through 2027; Citi sees a peak in Q2-2027 | BuySellRam summary of Gartner/Citi, 2026-10-01 |

Direction: **up, through 2027.** High-VRAM cards are the most exposed because 24 GB GDDR6X/GDDR7 modules and HBM compete with AI-server demand.

## B4. Import to Belarus vs buying in Minsk

FX used: **1 USD = 2.9462 BYN**, NBRB official rate **2026-10-30** (api.nbrb.by).

| Card | Minsk asking (kufar.by, 2026-10-30) | USD equiv. | US used | US advantage |
|---|---|---|---|---|
| RTX 3090 24 GB | 4,350–6,000 BYN (11 listings) | **$1,476–2,037** | $1,199 | **$277–838** |
| RTX 4080 Super | 4,240–4,500 BYN (5 listings) | **$1,439–1,527** | $1,179 | **$260–348** |
| RTX 4090 | 9,350–11,000 BYN (5 listings) | **$3,174–3,733** | $2,999–3,199 | **$0–530** |
| RX 7900 XTX | **no listings found** | — | $1,149 | n/a (AMD high-end is absent locally) |

Your brief's local numbers ($1,390–1,530 for a 4080S, $1,480–2,040 for a 3090) are **confirmed** by the kufar sample.

Landed-cost arithmetic for one 3090 ($1,199 → declared ~€1,050):

| Line | Cost |
|---|---|
| Card | $1,199 |
| US→BY shipping (insured, 2–3 kg) | $60–120 |
| Duty: postal limit €200/31 kg; 15 % of the excess ≈ 15 % × €850 ≈ €128 | **$135–150** |
| **Landed** | **$1,394–1,469** |
| Minsk median | ~$1,650 |
| **Net saving** | **≈ $180–250 (11–15 %)** |

Plus: no warranty, no RMA path, 3–6 weeks transit, and the **export-control question**: US EAR country guidance treats Belarus alongside Russia for advanced-computing items, and the EU prohibits exporting dual-use goods to Belarus — I could not machine-read the primary BIS/EU text from this host, so treat "this shipment is not lawful" as *assessed*, not proven (see unverified list). Practically, retail US sellers do not ship to BY; what exists is re-export through third countries.

**Verdict B:** importing 1–2 cards is **not worth it** — 11–15 % net saving against legal, warranty and transit risk. It only becomes rational at ≥4 cards in one consolidated shipment through a lawful channel. Better options in order: (1) buy locally and pay the ~15 % premium; (2) buy used in Poland/Lithuania and hand-carry inside the personal allowance; (3) a business import with proper classification.

---

# PART C — Non-NVIDIA accelerators, honestly

## C1. ROCm trajectory in 2026 — what is actually usable

| Component | State (2026-10-30) | Evidence |
|---|---|---|
| ROCm core | **10.1.0**, released 2026-10-05; 7.14.0 was the last 7.x tag (2026-07-16) | rocm.docs release notes; GitHub tags |
| PyTorch | stable wheels torch 2.14.0/2.14.1/2.15.0 for `rocm7.14` | download.pytorch.org/whl |
| Triton | upstream 3.9.0 (2026-10-27) with AMD backend; `triton-rocm` 3.6.0–3.9.0 shipped by PyTorch | triton releases; wheel index |
| Custom Triton kernels | work, but need re-tuning; corpus of 39,893 ROCm-validated Triton kernels exists | AMDKernelVault, arXiv 2609.12471 |
| FLA / linear-attention libs | `[rocm]` extra; AMD CI **disabled**; CI targets MI300, not RDNA | FLA repo |
| Porting reality | "NVIDIA-specific optimizations hinder performance portability"; direct port needed co-design + shims; 1.5× only after that work | Meta, arXiv 2610.22230 |

Maturity call: **usable for standard ops, not turnkey for hand-tuned kernels.** The gap is tuning and validation, not "does it run".

## C2. Huawei Ascend

| Part | Price | Where | Buyable outside China? | Stack | Triton port |
|---|---|---|---|---|---|
| Ascend 910C board | **>110,000 CNY (~$16,000)**, up from ~90,000 CNY in Jan 2026 | China channel | No retail channel; export-controlled on the EU/US side for BY | CANN + `torch_npu` | `triton-ascend` exists (3.6.0, CANN 9.1.0) |
| Ascend 950DT | **>250,000 CNY** | China | No | CANN | same |
| Atlas 300I Duo 96 GB | **$2,489** (AliExpress, ships Nov) | AliExpress | Grey import possible | CANN, inference-oriented | — |
| Atlas 310P (single) | ~$600–900 (not verified) | AliExpress/China | Grey | CANN | — |

Critical distinction: **Atlas 300I Duo is 2× 310P inference silicon, 96 GB LPDDR4X at 408 GB/s total (204 GB/s per die)** — great VRAM-per-dollar for *inference of big MoE*, useless for bf16 training. The 910C is a training part but at ~$16k per board it is 13× a 3090 for maybe 6–10× the throughput, with a completely separate software world.

Software: `torch_npu` (598 stars, pushed 2026-10-30) and `triton-ascend` (207 stars, pushed 2026-10-30) are alive. FLA v0.6.0 ships an `[npu]` extra pinning `torch==2.10.0, torch_npu==2.10.0.post7, triton-ascend==3.6.0` and added a KDA Ascend backend (PR #1090). So a *Triton* stack ports to Ascend at least as well as to AMD — but your `sm_120` tile tables still die, and you now debug CANN.

## C3. Other Chinese accelerators

| Card | VRAM | Price | Software | Training path | Confidence |
|---|---|---|---|---|---|
| Cambricon MLU370-X8 | 48 GB | **$5,000–5,800**/unit (Alibaba, 2026) | Neuware, PyTorch fork | 256 TOPS INT8 — inference-oriented | medium (B2B listing) |
| Biren BR104 | 64 GB (BR100 family) | **no verifiable price** | BIRENSUPA | no public PyTorch training story | **none** — not a purchasable product for an individual |
| Moore Threads MTT S4000 | 48 GB, 768 GB/s | **$12,300–13,000** (Alibaba); 1.8–2.0 M RUB on Avito | MUSA + musify (CUDA→MUSA) | claims CUDA compatibility; real-world reports are thin | low (price spread 2×) |
| Iluvatar T20 / Bi150 | 32–64 GB | **no verifiable price** | CUDA-like SDK | — | **none** |

Note the price incoherence on MTT S4000 ($12.3k Alibaba vs ~$23k Avito) — treat any number as an asking price in a thin market.

## C4. Verdict table

| Option | Price | VRAM | bf16 peak | Useful bf16 | $/useful-TFLOPS | Port effort | Buyable in BY | Beats 2×3090? |
|---|---|---|---|---|---|---|---|---|
| 2× RTX 3090 | $2,398 | 48 GB | 142 | ~78 | **$30.7** | 0 | yes (local) | — |
| RTX 4080 Super | $1,179 | 16 GB | 104 | ~57 | $20.6 | 0 | yes | no (VRAM) |
| Radeon AI PRO R9700 | $1,500 | 32 GB | 191 | ~76 | $19.6 | 6–10 wk | grey | no |
| RX 7900 XTX | $1,149 | 24 GB | 123 | ~49 | $23.4 | 6–10 wk | grey | no |
| MI210 | $4,369 | 64 GB | 181 | ~81 | $53.9 | 4–8 wk | grey | no |
| Atlas 300I Duo | $2,489 | 96 GB | inference only | ~0 (training) | n/a | 8–12 wk | grey | **no** |
| Ascend 910C | ~$16,000 | 64 GB HBM | ~800 (vendor, unverified) | ~360 | ~$44 | 10–16 wk | no | no |
| MLU370-X8 | $5,000+ | 48 GB | INT8-centric | low | >$80 | 10–16 wk | grey | no |
| MTT S4000 | $12,300+ | 48 GB | unverified | unverified | >$100 | 10–16 wk | grey | no |

---

# VERIFIED (sourced)

- ROCm 10.1.0 is the current Core SDK; its release-notes and compatibility pages carry the date **2026-10-05**. The support matrix marks gfx1100/gfx1201/gfx1030/gfx908/gfx90a/gfx942 ✅ and **gfx906 (MI50) ❌** — rocm.docs.amd.com (accessed 2026-10-30).
- PyTorch ships stable ROCm wheels for **torch 2.14.0–2.15.0** (`rocm7.14`); `triton-rocm` **3.6.0–3.9.0** — download.pytorch.org (accessed 2026-10-30).
- Triton **v3.9.0** released **2026-10-27**, AMD GPUs supported.
- FLA **v0.5.2** (2026-07-27) **already had a `[rocm]` extra**; **v0.6.0** released **2026-10-23**. AMD MI300 CI exists but is **disabled by default**; some KDA tests require CUDA or NPU.
- Peak throughput figures: 7900 XTX 123 FP16 TFLOPS; R9700 191 FP16 / 383 FP8; W6800 36; W7800 90; MI100 92 bf16; MI300X 2,615 FP8 — flopper.io GPU pages, "Spec confidence: Official" (accessed 2026-10-30).
- US used asking prices (2026-10-30): 3090 $1,199 · 3090 Ti $1,699 · 4080 $1,084 · 4080S $1,179 · 4090 $2,999 (bestvaluegpu, Oct 2026); 4090 median $3,199 / 4080 median $1,149 (DealHunter, 2026-10-30).
- RTX 4090 median asking rose **$2,499 → $3,199** between 2026-08-20 and 2026-10-30 (DealHunter daily series).
- RTX 5090 street **$6,300** used / $7,443.99 Amazon; RTX 5080 $1,875 used / $1,899.99 Newegg (videocardprices, 2026-10-30).
- A board partner reportedly cannot order GB202 + 32 GB GDDR7 kits even on backorder (MIXED, 2026-10-27, citing der8auer 2026-10-25).
- TrendForce (2026-08-25): DRAM/NAND contract prices could rise **70–140 % in 2027**; EE Times Asia (2026-10-06): HBM **+121 % YoY**.
- FX **1 USD = 2.9462 BYN**, NBRB, 2026-10-30.
- Minsk askings: 3090 4,350–6,000 BYN ($1,476–2,037); 4080S 4,240–4,500 BYN ($1,439–1,527); 4090 9,350–11,000 BYN ($3,174–3,733); no 7900 XTX listings (kufar.by, 2026-10-30).
- Ascend 910C board **>110,000 CNY (~$16,000)** vs ~90,000 CNY at start of 2026 (Reuters, 2026-09-10); Atlas 300I Duo 96 GB at **$2,489** with 408 GB/s (AliExpress listing; bandwidth from Flopper).

# SPECULATION (my inference)

- **Efficiency factors** 55 % (NVIDIA) / 40 % (AMD) used for $/useful-TFLOPS. No measured bf16 GEMM efficiency for these exact cards in this exact stack was found.
- **Port effort 6–10 weeks.** Derived from the breakage list + Meta's account of a direct-port baseline; no public benchmark of porting an `sm_120` MoE GEMM to gfx1100/gfx1201 exists.
- **Landed cost** for a 3090 to Belarus ($1,394–1,469) assumes 15 % duty on the excess over €200 and $60–120 shipping; the duty rate itself is contested between sources (5 % vs 15 % post-July-2026).
- **Legality of exporting a 4090/4080 to Belarus from the US.** I could not read the primary BIS or EU text from this host. Assessment only.
- **Ascend 910C bf16 ≈ 800 TFLOPS** and **$/useful ≈ $44**: vendor spec not verified in this session; the conclusion does not depend on the exact number because the price is 13× a 3090.
- **7900 XTX "no local listings"** reflects one kufar query on one day, not market absence.

# Unverified / could not confirm

1. **eBay sold (cleared) prices for all seven cards** — blocked. Everything above is *asking*, from trackers that say so.
2. **A6000 / A5000 medians** — sources disagree by 2× ($1,500 vs $3,900 for A5000).
3. **MI210 peak TFLOPS** — vendor figure 181 not corroborated by the tracker used for the other cards.
4. **RTX PRO 4000 Blackwell 24 GB street price** — none found.
5. **Biren BR104 and Iluvatar T20/Bi150 prices/availability** — no data at all.
6. **Ascend 910B (non-C) price** — only the 910C and 950DT are quoted.
7. **Whether US/EU export rules legally bar a private individual from importing one GPU into Belarus** — assessed, not sourced.
8. **Belarus duty rate for postal imports after 2026-07-01** — one source says 5 % replaced 15 %; customs.gov.by page returned 404 from here.
9. **Actual bf16 throughput of FLA `chunk_kda` on gfx1100/gfx1201** — no public measurement.
10. **Whether FLA 0.5.2's `chunk_kda` runs unmodified on ROCm** — untested here (read-only task).

# Open questions worth answering before spending money

- Is your MoE tile table generated or hand-written? If generated, the AMD port cost drops by roughly half.
- Do you need >24 GB per card, or is total VRAM across cards enough? If total, 2×3090 remains the answer through 2027.
- Would an RTX 5080 24 GB Super (rumoured, TechSpot 2026-10-19) change the buy-now-vs-wait calculus? Watch Q4-2026.
- Is a used 4090 at $3,000 justified by your throughput ceiling, given it is now ~2.5× a 3090 for ~2× the bf16?

---

## Sources (all accessed 2026-10-30 unless dated)

1. AMD, *ROCm 10.1.0 compatibility matrix* and *System requirements* — rocm.docs.amd.com (release 2026-10-05).
2. AMD, *ROCm Core SDK 10.1.0 release notes* — 2026-10-05.
3. GitHub ROCm/ROCm releases & tags — rocm-7.14.0, 2026-07-16.
4. PyTorch wheel index — download.pytorch.org/whl/rocm7.14/torch, /whl/triton-rocm.
5. triton-lang/triton — v3.9.0, 2026-10-27; README supported platforms.
6. fla-org/flash-linear-attention — releases v0.5.2 (2026-07-27), v0.6.0 (2026-10-23); `pyproject.toml` @v0.5.2; `INSTALL.md`; `.github/workflows/amd-mi300.yml`; `tests/ops/test_kda.py`.
7. Meta, *Architecting for Hardware Diversity* — arXiv 2610.22230, 2026-10-15.
8. *AMDKernelVault* — arXiv 2609.12471, 2026-09-11.
9. flopper.io GPU pages (MI300X, MI100, 7900 XTX, R9700, W6800, W7800, RTX 3090, RTX 4080 Super) — spec confidence "Official".
10. videocardprices.com — homepage, methodology, GPU price index, workstation/server inventory, 2026-10-30.
11. DealHunter — /prices, /prices/rtx-4090 daily medians, rebuilt 2026-10-30.
12. bestvaluegpu.com price-history pages — via search snippets dated Oct 2026 (direct access Cloudflare-blocked).
13. MIXED, *Nvidia says RTX 5090 isn't discontinued…* — 2026-10-27 (citing der8auer, 2026-10-25).
14. TechSpot, *Nvidia says the RTX 5090 isn't discontinued, but most listings are over $9,300* — 2026-10-19 (headline/snippet; body Cloudflare-blocked).
15. TrendForce, *Memory Prices Soar; DRAM and NAND Flash…* — 2026-08-25.
16. EE Times Asia, *TrendForce Raises 2027 HBM Price Forecast* — 2026-10-06.
17. Reuters, *China's AI chipmakers raise prices as HBM supply tightens* — 2026-09-10.
18. Alibaba/AliExpress listings — MLU370-X8, MTT S4000, Atlas 300I Duo (listing dates unknown; accessed 2026-10-30).
19. kufar.by listings for RTX 3090 / 4080 Super / 4090 — accessed 2026-10-30.
20. NBRB, USD official rate — 2026-10-30.
21. GitHub Ascend/pytorch, triton-lang/triton-ascend — activity as of 2026-10-30.
