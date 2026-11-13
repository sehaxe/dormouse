# EPYC server platform — definitive buying report

**Date:** 2026-11-05 · **Buyer:** Minsk, Belarus · **Channel:** Taobao → Superbuy → Minsk (CNY), with eBay / eBay-DE / AliExpress as alternates
**Question:** buy the Huananzhi H12D-8D lot ($660.44 = ¥4,062.91, with BMC), and **one CPU or two**?
**Answer:** **Buy one CPU, on a single-socket board — if you buy at all.** The lot is genuinely cheaper than a genuine Supermicro *in your channel* (the previous study was wrong about that: it compared a ZOL Beijing dealer reference to your landed agent price). But the platform is a **2× memory-bandwidth upgrade for 3.7× the money**, and it does **not** fix the thing that limits your training. Dual socket is not justified by either of your two goals at any point on your roadmap.

---

## 0. How the numbers in this report were made

| Item | Value | Source |
|---|---|---|
| USD/CNY cross-rate | **6.668 CNY per USD** | NBRB official rates 2026-11-05: CNY 0.44698 BYN/10? (0.44698 BYN per CNY), USD 2.9806 BYN — <https://api.nbrb.by/exrates/rates?periodicity=0> |
| USD/BYN | 2.9806 | same |
| EUR/USD | 1.1258 | same (EUR 3.3556 BYN) |
| Your lot, sticker | ¥4,062.91 = **$609.27** at the cross-rate | your message; arithmetic |
| Your lot, landed | **$660.44** | your message |
| **Channel factor** | **1.084** (landed ÷ sticker) — domestic shipping + international line + service | derived from your own lot: 660.44 ÷ 609.27 = 1.0840 |
| Belarus duty, if assessed | 15 % of value over €200 ≈ **$58** on a $609 board | prior route memo (`docs/records/research/2026-10-30-import-routes-belarus.md`), customs rule; **confirm DDP status with your agent** |

Every CNY→USD conversion below uses **6.668**, and every "landed" figure uses **×1.084**. Duty is shown separately because I could not confirm whether your agent quotes DDP.

**Evidence grades used in the tables:** `spec` = vendor/official page · `press` = dated Chinese media · `retail-live` = price on a page fetched today · `tracker` = price aggregator · `owner` = your measured/observed figure · `proj` = my projection (marked).

---

## 1. Board table

| Board | Sockets | DIMM / channels | PCIe (slots, gen) | Form factor | BMC | Price, CNY | Price, USD (landed) | Link / date | 5070 Ti known to POST? |
|---|---|---|---|---|---|---|---|---|---|
| **Huananzhi H12D-8D** (the lot) | **1× SP3** ✅ verified | 8 / **8** (1DPC) | 4× PCIe **4.0 x16**, 3× M.2 x4, 3× SFF-8643 | ATX 305×245 | **AST2500, optional module** (lot includes it) | **¥4,062.91** (owner) | **$660.44** (owner, landed) | vendor spec <https://www.huananzhi.com/en/list_6/177.html> · 2026-11-05 | **No public report either way** — unknown |
| **Huananzhi H12D-16D** | **2× SP3** ✅ ("支持两颗") | 16 / **16** (8 per CPU) | 7× PCIe **4.0 x16** | **E-ATX 305×335** | AST2500 onboard, "OpenBMC" | **"3K多"** ≈ ¥3,300–4,100; **build uses ¥3,500 (proj)** | **$569** landed (proj) | vendor spec <https://www.huananzhi.com/list_6/650.html> · press <https://post.smzdm.com/p/a4qd74rk/> 2026-08-23 | unknown |
| **Supermicro H12SSL-i** (genuine) | 1× SP3 | 8 / 8 | **5× x16 + 2× x8 + 2× x4**, gen4 | ATX 12"×9.6" | **AST2500 onboard, included** | **ZOL ref ¥3,399**; dealer quote **¥3,264**; **your channel ≈ ¥4,680** (implied by "+$100 vs the lot") | ZOL ref **$509.70**; your channel **$760.44** | ZOL <https://detail.zol.com.cn/servermotherboard/index1355133.shtml> — 参考价 ¥3399, dealer ¥3264, **价格更新时间 11-05 21:25** (fetched 2026-11-05) | unknown, but the strongest BIOS track record of the three |
| **Supermicro H12SSL-NT** (genuine) | 1× SP3 | 8 / 8 | same as -i, plus **dual 10GbE** | ATX | AST2500 | — | **€810 ≈ $911.50** before shipping/duty | idealo.fr <https://www.idealo.fr/prix/201422378/supermicro-h12ssl-nt.html> — "à partir de 810,00 €", Nov 2026 (page blocked to me; price read from search result) | unknown |
| **Supermicro H12DSi-N6** (genuine, dual) | 2× SP3 | 16 / 16 | 4× x16 + 3× x8, gen4 | ATX | AST2500 | — | **$1,185 (Newegg)** — *carried from the previous study, not re-verified today* | prior study, 2026-11-05 | unknown |
| **Supermicro H11SSL-i** (cheaper legit) | 1× SP3 | 8 / 8 | **PCIe 3.0** only (4× x16 + 2× x8) | ATX | AST2500 | **¥2,999** | **$449.80** sticker / $487.60 landed | ZOL <https://detail.zol.com.cn/servermotherboard/index1343492.shtml> — 参考价 ¥2999, fetched 2026-11-05 | unknown |

### The price question, settled honestly

The previous study said the genuine H12SSL-i is **cheaper** (¥3,399 vs the clone). **In your channel that is wrong, and here is the mechanism:**

1. ZOL's ¥3,399 is a **"参考价"** — the page itself prints *"价格仅供参考，购买时请以当地实际销售产品价格为准"* ("reference only; the actual local selling price prevails"). The one dealer quote under it is **¥3,264**, a Beijing B2B lead-gen price behind a "获取底价" button — not a Taobao listing, and not something Superbuy can buy.
2. ¥3,399 = **$509.70** at today's cross-rate. For your "+$100" observation to hold, the genuine board in your channel must be **≈ ¥4,680 sticker / $760 landed**. That is **38 % above** the ZOL reference. Plausible: genuine Supermicro SP3 boards are thin on Taobao in the DDR4 shortage, and sellers price them against the same demand that pushed the clone from ¥1,800 (early 2026) to ¥3,800+ (Sept 2026) — documented in <https://post.smzdm.com/p/arzq5px7/> (2026-09-23), which also records **Tyan's single-socket 7003 board at ¥3,500 on JD**.
3. So: **the clone is ~$95–100 cheaper in your channel.** Correct as the owner stated. What the previous study got wrong was the *channel*, not the arithmetic.

**But the $95 is not the deciding number.** The clone carries three risks the genuine board does not: BMC firmware that hangs, GPU POST failures, and no QVL / no DIMM population table. With a 5070 Ti — a card with **no public POST report on this board** — the $95 saving buys you an unvalidated PCIe enumeration. See §6.

### Single-socket verification (the brief asked for it)

The vendor's own spec table lists **one** CPU socket line and **8** DIMM slots for the H12D-8D, with *"8 x DDR4 memory slot"*, *"supports octa-channel"*, *"4 x PCI Express 4.0x16"*, *"AST2500(BMC) can Optional Module"*, *"305\*245MM ATX"* — <https://www.huananzhi.com/en/list_6/177.html>, fetched 2026-11-05. The 16D page says **"支持两颗AMD EPYC平台处理器"** ("supports two AMD EPYC processors") and **"支持单处理器八通道"** ("eight channels per processor") — <https://www.huananzhi.com/list_6/650.html>. **Confirmed: 8D = 1 socket, 16D = 2 sockets.**

---

## 2. CPU table

Specs from Wikipedia's EPYC tables (Rome/Milan) — <https://en.wikipedia.org/wiki/Epyc>, fetched 2026-11-05. Launch MSRP is the 2019/2020 list price, **not** a street price; it is shown only to show how far the used market has fallen.

| SKU | Cores/threads | Base/boost | TDP | L3 | Channels | PCIe | Launch MSRP | **Street price, dated** | $/core | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **EPYC 7452** (Rome) | 32 / 64 | 2.35 / 3.35 | **155 W** | 128 MB | 8 | 4.0 | $2,025 | **¥600 ≈ $89.98** — <https://post.smzdm.com/p/arzq5px7/> (2026-09-23, citing a 2026-09-03 build video) | **$2.81** | **The value pick. 155 W is easy to cool; full 8 channels; gen4.** |
| **EPYC 7K62** (OEM, = 7642) | 48 / 96 | 2.6 / 3.3 | 225–240 W (sources disagree) | 256 MB | 8 | 4.0 | OEM only | **C$411.32 ≈ $294.92** shipped from China (eBay listing, read 2026-11-05); China-domestic **≈ ¥900 ≈ $135 (proj, scaled 48/32 from the 7452)** | $6.14 export / **$2.81 proj domestic** | **OEM SKU** — no AMD warranty; seller warranty only. Runs hot; needs a 3U-class cooler. |
| EPYC 7502 | 32 / 64 | 2.5 / 3.35 | 180 W | 128 MB | 8 | 4.0 | $2,600 | **not verified today** — do not budget from this row | — | Same cores as 7452, +13 % clock, +25 W, higher price. **Skip.** |
| EPYC 7551 (Naples) | 32 / 64 | 2.0 / 3.0 | 180 W | 64 MB | 8 | **3.0** | $3,400 | **not verified today** | — | **Disqualifies your gen4 goal** — PCIe 3.0 only. Skip. |
| EPYC 7642 | 48 / 96 | 2.3 / 3.3 | 225 W | 256 MB | 8 | 4.0 | $4,775 | **not verified today** | — | The retail twin of 7K62. Buy 7K62 if the price gap is real. |
| EPYC 7702 | 64 / 128 | 2.0 / 3.35 | 200 W | 256 MB | 8 | 4.0 | $6,450 | **not verified today** | — | 64 cores you cannot use: your workload is launch-bound / bandwidth-bound. |
| EPYC 7742 | 64 / 128 | 2.25 / 3.4 | 225 W | 256 MB | 8 | 4.0 | $6,950 | **not verified today** | — | Same objection as 7702, at a higher price. |
| EPYC 7R32 (OEM) | 48 / 96 | 2.8 / 3.3 | 180 W | 128 MB | 8 | 4.0 | OEM/cloud | **not verified today** | — | Good 2P cloud part; as a 1P it is fine but not cheaper than 7452/core. |
| EPYC 7B13 (OEM, AWS) | 64 / 128 | 2.45 / 3.55 | ~240 W | 256 MB | 8 | 4.0 | OEM/cloud | **not verified today** | — | **OEM/cloud SKU**: no warranty, possible vendor-locked firmware. A 7B13 owner in <https://post.smzdm.com/p/a4qd74rk/> (2026-08-23) notes the board needs **25 A on 5 V/3.3 V** or it can hang. High-risk pick. |

**Why the missing prices do not weaken the verdict:** every EPYC 7002/7003 has **128 PCIe 4.0 lanes and 8 memory channels**. For your two goals the CPU is a *socket occupancy device*: 2 GPUs at x16 need 32 lanes, and a bandwidth-bound decode needs channels, not cores. The **7452 at ¥600** therefore delivers the same platform capability as a 7742 for the two things you actually do. Paying more for cores would be paying for a resource your measured profile does not consume (2 % MFU, launch-bound; decode bandwidth-bound). The only reason to buy a bigger SKU is if you later run CPU-side pre/post-processing or many concurrent sessions — neither is in your plan.

**OEM reality:** 7K62, 7R32, 7B13 are OEM/cloud SKUs. AMD's warranty does not follow them; the Taobao seller's "3 years" is a dealer promise that requires shipping back to Shenzhen at your cost. Treat any OEM warranty as **zero** for budgeting.

---

## 3. RAM table

DDR4 is in a shortage in 2026 — this is the part of the build that moved most. US used prices from DIMMsum (fetched **2026-11-05**); China channel from two dated smzdm articles.

| Stick | US low, per stick | $/GB | China channel | $/GB (China) | Link / date |
|---|---|---|---|---|---|
| **16 GB DDR4-2133 RDIMM** | **$70.00** | **$4.38** | **≈ ¥400 = $60 (proj** — half of the 32 GB 2666 price) | $3.75 | <https://dimmsum.com/ddr4/16gb-2133-rdimm> · 2026-11-05 |
| 16 GB DDR4-2666 RDIMM | $94.99 | $5.94 | — | — | <https://dimmsum.com/ddr4/16gb-2666-rdimm> · 2026-11-05 |
| **32 GB DDR4-2666 RDIMM** | **$190** | **$5.94** | **¥800** (Sept 2026, up from ¥400 in August) | $3.60 | <https://dimmsum.com/ddr4/32gb-2666-rdimm> · 2026-11-05; <https://post.smzdm.com/p/arzq5px7/> · 2026-09-23 |
| **32 GB DDR4-3200 RDIMM** | **$261** | **$8.17** | **≈ ¥1,200** ("even mediocre sticks") | $5.40 | <https://dimmsum.com/ddr4/32gb-3200-rdimm> · 2026-11-05; <https://post.smzdm.com/p/a4qd74rk/> · 2026-08-23 |
| 64 GB DDR4-3200 RDIMM | $650 | $10.15 | — | — | <https://dimmsum.com/ddr4/64gb-3200-rdimm> · 2026-11-05 |

**What a full set costs**

| Config | US (DIMMsum) | China channel | Notes |
|---|---|---|---|
| **8× 16 GB 2133** (8 channels, single socket) | **$560** | **≈ ¥3,200 = $480** | The build target. 128 GB total. |
| 16× 16 GB 2133 (dual socket) | $1,120 | ≈ ¥6,400 = $960 | Only if you go dual. |
| 8× 32 GB 2666 (256 GB) | $1,520 | ≈ ¥6,400 = $960 | For 100B-class at 8-bit later. |
| 8× 32 GB 3200 (256 GB, fastest) | $2,088 | ≈ ¥9,600 = $1,440 | 204.8 GB/s peak; the 3200 premium is **$570 for +20 % bandwidth** — poor value vs 2666. |
| 16× 32 GB 2666 (512 GB, dual) | $3,040 | ≈ ¥12,800 = $1,920 | The "100B at fp16" config. |

**China is ~35 % cheaper than US used for the same stick** — buy RAM in your channel, not on eBay.

### Can you mix your existing 4×16 GB 2133 with new sticks?

- **On the EPYC board (1DPC, 8 slots):** it will *POST*, but you would have four 16 GB channels and four 32 GB channels. Channel capacity becomes asymmetric, memory interleaving stops being uniform, and you lose a chunk of the very bandwidth you are buying. **AMD's guidance and every server vendor's population table say: populate all channels identically.** Combined with the clone having **no QVL and no DIMM population table** (the 8-page manual does not contain one), mixing sizes is a self-inflicted debugging session. **Buy 4 more identical 16 GB 2133, or 8 new identical sticks. Do not mix sizes on EPYC.**
- **On your X10SRL-F (2DPC, 8 slots, 4 free):** adding 4 more 16 GB 2133 is exactly the supported configuration (8×16, 2DPC, 2133). That is the cheap upgrade in §5.
- Speed mixing (2133 + 2666): everything runs at **2133**. Legal, just slower.

---

## 4. THE DECISION — one CPU or two

### 4.1 What bandwidth each configuration actually gives

Bandwidth = channels × MT/s × 8 bytes. Your measured 26 tok/s at 68 GB/s implies **2.625 GB read per token** for the 30 GB model — and 68.26 ÷ 2.625 = 26.0, i.e. your measurement is already at **100 % of theoretical peak**, so I use 100 % for the 4-channel baseline and a **12 % haircut** for 8-channel (STREAM efficiency on Rome/Milan is typically 88 %).

| Config | Channels | Peak GB/s | Effective (88 %) | 30 GB model, tok/s | 60 GB model, tok/s (proj) |
|---|---|---|---|---|---|
| **Today: 4×16 2133** | 4 | 68.3 | 68.3 (measured) | **26 (measured)** | **13.0** (proj) |
| **Single EPYC, 8×16 2133** | 8 | 136.5 | 120.1 | **45.8** | **22.9** |
| Single EPYC, 8×32 2666 | 8 | 170.6 | 150.1 | 57.2 | 28.6 |
| Single EPYC, 8×32 3200 | 8 | 204.8 | 180.2 | 68.7 | 34.3 |
| **Dual EPYC, 16×16 2133 — one process** | 16 | 273.0 | **~120 (NUMA-local)** | **~45.8** | **~22.9** |
| Dual EPYC — two processes, one per socket | 16 | 273.0 | 240.2 aggregate | 91.6 aggregate | 45.8 aggregate |

### 4.2 The NUMA point, stated plainly

A single process does **not** get 2× bandwidth for free. On EPYC 7002/7003 each socket is a NUMA node (NPS1). A decode process allocates its weights on the node it starts on and reads them at that node's bandwidth; crossing to the other node costs latency and, for a bandwidth-bound token loop, usually loses. To use both sockets for one model you must shard it (tensor-parallel or layer-split with NUMA-aware allocation), which adds synchronization to a loop that is *already* latency-sensitive. **Two sockets buy aggregate throughput for two independent processes, not speed for one model.**

### 4.3 What 2 CPUs add, and what they cost

**Adds:** +8 DIMM slots, +8 channels (aggregate), +1 CPU's cores (32 more with 2×7452), +128 PCIe lanes, +3 x16 slots (16D has 7 vs 4).
**Costs:** 2nd CPU (¥600) + 8 more sticks (¥3,200) + 2nd cooler (¥150) + **E-ATX case** (305×335 mm; your current chassis is ATX — verify, likely won't fit) + ~150 W more idle/load heat + a 16-layer board with **only 2× 8-pin EPS** for two sockets.

**Total delta: ≈ ¥4,450 ≈ $667 landed** (case included), before duty.

### 4.4 Goal (a): 2 GPUs at gen4 x16

Reconstructed from your measurement: 628 MB ring all-reduce moves 1.256 GB in 0.19 s → 6.6 GB/s effective on gen3 x8. Two-GPU step = 0.40/2 + 0.19 = **0.39 s → ×1.03**. On gen4 x16 the all-reduce is 0.047 s → **0.247 s → ×1.62**.

| Platform | 2nd GPU speedup | Cost of platform | Does dual help? |
|---|---|---|---|
| Today (gen3 x8) | ×1.03 | $0 | — |
| **Single EPYC (gen4 x16)** | **×1.62** | **$1,042.57** | — |
| Dual EPYC | ×1.62 | $1,934.50 | **No — 2 GPUs need 32 of 128 lanes.** |

**Dual adds nothing to goal (a).** The second socket's lanes go to slots you will not populate. This is the clearest part of the decision.

### 4.5 Goal (b): 30–60 GB local MoE inference

- 30 GB fits **today** (62 GB). Today's speed: 26 tok/s. The platform buys **1.76×**.
- 60 GB does **not** fit today with OS + KV. Options: (i) +4 sticks on the existing board → 128 GB, still 4 channels → **~13 tok/s** for $280; (ii) single EPYC 8-channel → **~22.9 tok/s** for $1,042.57; (iii) dual → **~22.9 tok/s for one process** at $1,934.50.
- 100B-class at 2-bit ≈ 25 GB (fits today), 4-bit ≈ 50 GB (fits today), 8-bit ≈ 100 GB (needs the 128 GB build), fp16 ≈ 200 GB (needs 256 GB, still **single socket** with 8×32).

**Dual adds nothing to goal (b) either**, because the model is one process and the second socket is a second NUMA node.

### 4.6 The crossover point, quantified

Dual wins only when **at least two of these are true**:

1. **Two or more concurrent model replicas** (or two users) — then dual gives 2× aggregate at $42.24/tok/s vs $22.76 single, i.e. **still worse $/tok/s**, but 2× absolute throughput.
2. **≥5 GPUs at x16** (4 slots on the 8D, 7 on the 16D).
3. **>512 GB RAM** — and even then, 8×64 GB single-socket reaches 512 GB, so this needs >512 GB before dual is mandatory.

At your roadmap (1 GPU now, maybe 2; one model at a time; ≤256 GB), **none of the three triggers.** The crossover is not a price point — it is a *concurrency* point, and you are not near it.

---

## 5. Three complete build sheets (your channel, landed)

Prices: board from your lot; CPU from the dated smzdm build; RAM 16 GB 2133 at ¥400 (projection from the ¥800 32 GB datapoint); cooler ¥150 (est.); all ×1.084 channel factor. Duty shown separately.

### (i) Minimum viable 8-channel single-CPU

| Line | CNY | USD landed |
|---|---|---|
| Huananzhi H12D-8D + BMC | 4,062.91 | 660.44 |
| EPYC 7452 (32C, 155 W) | 600 | 97.54 |
| SP3 cooler (155 W class) | 150 (est.) | 24.39 |
| 4× 16 GB DDR4-2133 RDIMM (to join your existing 4 → 8×16 = 128 GB) | 1,600 (proj) | 260.11 |
| **Total** | **6,412.91** | **$1,042.57** |
| + Belarus duty if assessed (15 % over €200) | — | **+$58** → **$1,100** |

**What it does:** 128 GB, 8 channels, 136.5 GB/s peak → **45.8 tok/s** at 30 GB, **22.9 tok/s** at 60 GB, 2 GPUs at gen4 x16.

### (ii) The cheapest thing that runs a 60 GB model

**It is the same build as (i).** 128 GB holds a 60 GB model plus OS and KV with ~60 GB spare. There is no cheaper separate configuration — the only cheaper way to *run* 60 GB at all is §5.0 below.

| Line | CNY | USD landed |
|---|---|---|
| Build (i) | 6,412.91 | 1,042.57 |
| *Optional:* 8× 32 GB 2666 instead of 8×16 (256 GB, for 100B at 8-bit) | +4,800 (proj) | +$480 → **$1,563** |

### (iii) Dual-socket

| Line | CNY | USD landed |
|---|---|---|
| Huananzhi H12D-16D | 3,500 (proj, "3K多") | 569.10 |
| 2× EPYC 7452 | 1,200 | 195.08 |
| 2× SP3 coolers | 300 (est.) | 48.78 |
| 16× 16 GB DDR4-2133 (8 channels per socket) | 6,400 (proj) | 1,040.44 |
| E-ATX case (305×335 mm) | 500 (est.) | 81.30 |
| **Total** | **11,900** | **$1,934.50** |
| + duty if assessed | — | **+$230** → **$2,165** |

**What it adds over (i):** 2× aggregate throughput *only if you run two processes*; 7 x16 slots; 512 GB ceiling.
**What it costs:** +$892, +E-ATX chassis, +150 W heat, and a board with 2× 8-pin EPS for two sockets.

### 5.0 The null build (must be on the table)

| Line | CNY | USD |
|---|---|---|
| 4× 16 GB DDR4-2133 for the existing X10SRL-F (4 free slots) | 1,600 (proj) | **$240–280** (DIMMsum: 4×$70 = $280) |
| **Result** | | **128 GB, still 4 channels → ~13 tok/s at 60 GB** |

This is the honest comparator: **$280 unlocks the 60 GB model; $1,043 makes it 1.76× faster.** Same $/tok/s order ($21.5 vs $45.5 per tok/s for 60 GB — the null build is actually *better* $/tok/s; the EPYC build wins only on absolute speed).

### 5.1 $/tok/s and $/×step

**Inference** (60 GB model, one process, landed before duty):

| Build | Capex | tok/s | **$/tok/s** | 30 GB tok/s | $/tok/s (30 GB) |
|---|---|---|---|---|---|
| Do nothing | $0 | 13.0* | — | 26.0 | — |
| **Null (+RAM only)** | **$280** | **13.0** | **$21.5** | 26.0 | — |
| **(i)/(ii) single EPYC** | **$1,042.57** | **22.9** | **$45.5** | 45.8 | **$22.8** |
| (iii) dual, one process | $1,934.50 | 22.9 | $84.5 | 45.8 | $42.2 |
| (iii) dual, two processes | $1,934.50 | 45.8 aggregate | $42.2 | 91.6 | $21.1 |

\* 60 GB does not fit in 62 GB with OS/KV; 13.0 tok/s is the projection *after* the RAM upgrade.

**Training** (your measured 0.40 s/step single-GPU baseline):

| Path | Capex | Speedup | **$ per +1.0×** | $ per +0.1× |
|---|---|---|---|---|
| 2nd GPU on today's gen3 x8 | $750 | ×1.03 | $25,000 | $2,500 |
| **2nd GPU + single EPYC (gen4 x16)** | **$750 + $1,042.57 = $1,792.57** | **×1.62** | **$3,145** | **$315** |
| 2nd GPU + dual EPYC | $750 + $1,934.50 = $2,684.50 | ×1.62 | $4,710 | $471 |
| **H100 rental, same money** | $1,042.57 | — | **299 GPU-h at $3.49/h; 621 h at $1.68/h** | — |

The training row is the argument against buying for training: **$1,043 of H100 time is 299–621 GPU-hours.** Your whole 96M-parameter campaign is hours, not hundreds of hours. The platform does not move MFU (2 %) — the bottleneck is launch overhead, which is a software problem (`torch.compile`/CUDA graphs/batch size), not a motherboard problem.

---

## 6. Verdict

**Buy one CPU, on a single-socket board — and only if 13 tok/s is not enough for you.** The arithmetic is not close: dual costs **+$892** and delivers **zero** additional tokens/s for a single model and **zero** additional training speed for two GPUs; its only win is two concurrent processes, where it merely matches the single board's $/tok/s. Between the clone and the genuine Supermicro, **your correction stands: the clone is ~$95–100 cheaper in your channel**, because ZOL's ¥3,399 is a disclaimed reference price from a Beijing dealer, not a Taobao listing — the previous study compared the wrong two numbers. But the $95 is not the reason to choose: the clone has **no public evidence that an RTX 50-series card POSTs on it**, no QVL, no DIMM population table, and a BMC with documented hang/leak behaviour. If your 5070 Ti is the card that must work, the genuine H12SSL-i at ~$760 is the insurance policy, and $95 is a cheap premium on a $1,000 build. If you want the cheapest path to 60 GB, **buy four sticks of RAM for the X10SRL-F and stop** — that is $280 and it works today.

**Honest alternatives, priced:**
- **H100 rental** — $1.68 low / **$3.49 median** / $0.79 spot per GPU-hour, 38 offers from 27 providers, updated **2026-10-27** (<https://inferencehub.org/gpus/h100>). Your $660 lot = **189–393 GPU-hours**. For a 96M model at 2 % MFU this is the correct answer for training.
- **A 24–32 GB GPU** — used 3090 at ~$1,199 (prior study, videocardprices 2026-10-30): one card does **not** hold 30 GB; two = $2,398 for 48 GB at ~1.9 TB/s aggregate, roughly 300–500 tok/s on the 30 GB model (proj) — **$5–8/tok/s**, 3–6× better than the EPYC path on $/tok/s, but **60 GB still does not fit**. RTX 5090 street is ~$6,300 in this market — not a serious option.
- **Buy nothing** — $0, 26 tok/s at 30 GB today; 60 GB is not runnable until you add RAM.

---

## 7. First-hour test protocol (if you buy)

**Before power**
1. Compare standoff positions to the board. The H12D-8D is documented to lack an ATX mounting hole below the DIMM slots; a standoff there has shorted a board and killed an M.2 slot. Remove or insulate it.
2. Inspect the socket pins under raking light. Confirm the BMC module is fully seated (it is a separate module on the 8D).
3. Seat **one** DIMM first, in the slot farthest from the socket (A1 by convention) — the manual has **no population table**, so this is trial and error if it does not POST.

**POST**
4. 5070 Ti in slot 1, **no riser**, monitor on the card, one stick of RAM, nothing else. If it hangs at **code 78** or **b4**, clear CMOS and retry; then try the April-2025 "Resizable BAR / PCIe split" BIOS (the vendor ships it explicitly for GPU compatibility), then a known-older card to isolate the slot.
5. In BIOS enable **Above 4G Decoding** and **Resizable BAR** — required for Blackwell; disable IPMI if you get the b4→"ipmi failed" stall.
6. `lspci -vv | grep -E "LnkCap|LnkSta"` — must show **8.0 GT/s x16** (gen4), not x8/x4.
7. `dmidecode -t memory` — confirm **8 channels populated** and all sticks at rated speed.
8. `memtest86+` one full pass. Memory-controller marginality shows here, not in Linux.
9. BMC: static IP, latest firmware, and **leave every NIC either linked or disabled in BIOS** (the documented leak trigger is an unplugged 2.5G NIC flooding its own log until the BMC dies). `ipmitool sensor` + `ipmitool sel list`, then a 24 h soak.
10. 30 min `stress-ng --cpu` + `sensors`; watch DIMM temperatures (a reviewer measured **85 °C** on DIMMs under sustained LLM load until a fan was added).

**Return triggers — do not "live with" any of these**
- No POST with the 5070 Ti after both BIOS versions and a second slot → return.
- PCIe link trains at x8 or x4 on a card known to be x16 → return.
- Fewer than 8 channels detected with 8 known-good sticks → return.
- Any memtest error at stock speed → return.
- BMC unreachable after a firmware flash, or dies within 24 h → return.
- Physical damage, bent pins, or a missing standoff hole that forces chassis modification → return.

---

## VERIFIED (sourced)

- **H12D-8D is single-socket**, 8 DIMM / 8 channels / 1DPC, 4× PCIe 4.0 x16, ATX 305×245, BMC is an **optional AST2500 module** — vendor spec page, fetched 2026-11-05.
- **H12D-16D is dual-socket**, 16 DIMM (8 per CPU), 7× PCIe 4.0 x16, E-ATX 305×335, onboard AST2500 — vendor spec page, fetched 2026-11-05.
- **Your lot is $660.44 for ¥4,062.91**, i.e. a **1.084** channel factor at the NBRB cross-rate — your message + NBRB 2026-11-05.
- **ZOL lists H12SSL-i at ¥3,399 reference / ¥3,264 dealer, page timestamped 11-05 21:25**, and the page itself disclaims the reference price — fetched 2026-11-05.
- **The clone's channel price rose from ~¥1,800 to ¥3,800+**; **7452 ≈ ¥600**; **32 GB RECC went ¥400 → ¥800 in one month**; **Tyan single-socket 7003 ≈ ¥3,500 at JD** — smzdm, 2026-09-23.
- **H12D-16D is "3K多"**; **32 GB DDR4-3200 RECC ≈ ¥1,200**; **a 7B13 owner reports the board needs 25 A on 5 V/3.3 V** — smzdm, 2026-08-23.
- **16 GB 2133 RDIMM $70.00/stick; 32 GB 2666 $190; 32 GB 3200 $261; 64 GB 3200 $650** — DIMMsum, 2026-11-05.
- **H100 on-demand $1.68 low / $3.49 median / $0.79 spot, 38 offers, updated 2026-10-27** — InferenceHub.
- **EPYC 7452: 32C/64T, 2.35/3.35 GHz, 155 W, 8 channels, $2,025 launch; 7642: 48C, 225 W; 7702/7742: 64C, 200/225 W; 7551: PCIe 3.0** — Wikipedia EPYC tables, fetched 2026-11-05.
- **Onliner's catalogue returns "Ничего не нашлось" for EPYC** — fetched 2026-11-05 (no local retail EPYC in Belarus).
- **Bandwidth and speedup arithmetic** (136.5 GB/s at 8×2133; ×1.62 at gen4 x16) is derived from your measured numbers, not from a benchmark of this board.

## SPECULATION (my inference — not sourced)

- **H12SSL-i at ≈¥4,680 in your channel.** I did *not* obtain a Taobao listing; this is your "+$100" observation reverse-multiplied through the channel factor. If a Taobao seller quotes lower, the conclusion "clone is cheaper" flips — and you should buy the Supermicro.
- **16 GB 2133 at ¥400 in China** — scaled at half the dated ¥800 32 GB price. No direct listing.
- **All 8-channel tok/s figures** — linear scaling from your 26 tok/s with a 12 % haircut. No STREAM measurement of this board exists.
- **60 GB model at 2× the bytes/token** — assumes the same expert sparsity at 2× the parameter count.
- **H12D-16D at ¥3,500** — midpoint of "3K多"; the previous study saw ¥4,100 in September.
- **Duty of ~$58** — assumes the parcel is assessed at full value and is not DDP. Unverified for your agent.
- **No RTX 50-series POST data** — I found none, but my search access was degraded; absence of evidence is not evidence of absence. Treat as unknown, which is why the return window is the real decision.
- **Kufar local supply** — the site is JS-gated and I could not extract listings. Onliner shows no EPYC; Kufar may still have used server parts.

---

## Sources (all accessed 2026-11-05 unless dated)

1. Huananzhi H12D-8D spec — <https://www.huananzhi.com/en/list_6/177.html>
2. Huananzhi H12D-16D spec — <https://www.huananzhi.com/list_6/650.html>
3. ZOL, Supermicro H12SSL-i — <https://detail.zol.com.cn/servermotherboard/index1355133.shtml> (参考价 ¥3399; dealer ¥3264; timestamp 11-05 21:25)
4. ZOL, Supermicro H11SSL-i — <https://detail.zol.com.cn/servermotherboard/index1343492.shtml> (¥2999)
5. smzdm, "寨板厂把霄龙板卖到3800…" — <https://post.smzdm.com/p/arzq5px7/> · 2026-09-23
6. smzdm, "3K多的双路EPYC…H12D-16D" — <https://post.smzdm.com/p/a4qd74rk/> · 2026-08-23
7. DIMMsum, 16 GB 2133 RDIMM — <https://dimmsum.com/ddr4/16gb-2133-rdimm>
8. DIMMsum, 16 GB 2666 — <https://dimmsum.com/ddr4/16gb-2666-rdimm>
9. DIMMsum, 32 GB 2666 — <https://dimmsum.com/ddr4/32gb-2666-rdimm>
10. DIMMsum, 32 GB 3200 — <https://dimmsum.com/ddr4/32gb-3200-rdimm>
11. DIMMsum, 64 GB 3200 — <https://dimmsum.com/ddr4/64gb-3200-rdimm>
12. InferenceHub, H100 rental — <https://inferencehub.org/gpus/h100> · updated 2026-10-27
13. NBRB official rates — <https://api.nbrb.by/exrates/rates?periodicity=0> · 2026-11-05
14. Wikipedia, EPYC (Rome/Milan tables) — <https://en.wikipedia.org/wiki/Epyc>
15. AMD EPYC 7452 product page — <https://www.amd.com/en/products/cpu/amd-epyc-7452>
16. Onliner catalogue search — <https://catalog.onliner.by/search?query=epyc> · "Ничего не нашлось"
17. Kufar search — <https://www.kufar.by/l?q=epyc> (JS-gated; no listings extractable)
18. Supermicro H12SSL-NT at idealo.fr — <https://www.idealo.fr/prix/201422378/supermicro-h12ssl-nt.html> · "à partir de 810,00 €", Nov 2026 (read from search result; page blocked)
19. Prior study — `docs/records/research/2026-11-05-epyc-h12d-platform.md` (used only for the H12DSi-N6 Newegg figure and the risk inventory, both flagged)
20. Prior route memo — `docs/records/research/2026-10-30-import-routes-belarus.md` (duty rule)

## Open questions

1. **Actual Taobao listing price for a genuine H12SSL-i today.** This is the single number that could flip the board choice. Ten minutes with a Superbuy agent answers it.
2. **Does the lot's $660.44 include Belarus duty?** If not, add ~$58 and the gap to Supermicro narrows.
3. **Does the 5070 Ti POST?** No public data either way. The return window is the only real mitigation.
4. **H12DSi-N6 current price** — the $1,185 Newegg figure is from the previous study and was not re-verified.
5. **Kufar used-server supply** — unverified; Onliner shows no EPYC.
6. **Whether 2× 8-pin EPS is enough for two 240 W 7K62s on the 16D** — fine for 2×155 W 7452, marginal for 240 W SKUs.

## Confidence

| Section | Confidence | Why |
|---|---|---|
| 1. Board topology (sockets/DIMM/PCIe/form factor) | **High** | Vendor spec pages, fetched today |
| 1. Board prices in your channel | **Medium** | Clone = your lot (high); Supermicro = your observation reverse-computed (medium); ZOL = live but self-disclaimed |
| 2. CPU specs | **High** (official SKUs) / **Low** (7K62, 7B13 — OEM, sources disagree) | Wikipedia + AMD; OEM SKUs have no primary spec sheet |
| 2. CPU street prices | **Low** | Only 7452 (¥600) and 7K62 (C$411) are dated; the rest are unverified and excluded from the math |
| 3. RAM prices | **High** (US used) / **Medium** (China channel) | DIMMsum live; smzdm dated but 6–10 weeks old in a rising market |
| 4. One-vs-two decision | **High** | Arithmetic from your measurements + architecture; no benchmark needed |
| 5. Build totals | **Medium** | Board/CPU sourced; cooler/case/RAM-China are projections |
| 5. tok/s and $/tok/s | **Medium** | Linear projection from a real measurement; no STREAM data for this board |
| 6. Verdict | **High** on "one CPU / not dual"; **Medium** on clone-vs-Supermicro | The dual conclusion survives any plausible price error; the board choice depends on Q1 above |
| 7. Test protocol | **High** (procedure) / **Medium** (expected failure modes) | Procedures are standard; the specific code-78/b4 behaviours come from community reports |

---
---

# PART 2 — cross-channel matrix, mixed-sourcing builds, and the corrected verdict

**Appended 2026-11-05.** Owner's live Superbuy/Taobao reads from today are the baseline and are marked `OWNER-LIVE`. **No Xianyu (闲鱼) prices anywhere** — the owner excludes that marketplace. Used/refurbished rows are marked `USED` or `REFURB`. **Item prices only: no shipping, no duty, no VAT, no landed-cost arithmetic.** Where a channel could not be opened, the row says so and the price is `unverified` — no URL was invented.

**This part supersedes Part 1 §5 and §6 where they conflict**, because the owner's live reads are better data than the secondary sources Part 1 had to use: the clone is **not** the cheapest board, and the previous "buy the clone or Supermicro" framing was wrong in a way the owner's numbers settle.

## P2.0 What changed, in one paragraph

Part 1 recommended the Huananzhi clone because the only genuine board it could price was the Supermicro at ~$760 in-channel. Your live reads show **four genuine 8-channel SP3 boards below the clone's $605.56–$660.44**: **ASUS KRPA-U16 $457.94**, **Tyan S8030 $507.34**, **Gigabyte MZ32-AR0 $515.68**, and only then the clone. The clone is now the **4th cheapest of the six single-socket options priced here** — behind the ASUS, Tyan and Gigabyte — while being the only one with no public RTX 50-series POST record and no QVL. The board decision reverses. The CPU decision does not: **one CPU**, and the dual build remains unjustifiable for your workloads.

## P2.1 Cross-channel price matrix

### Boards

| Part | Channel | Price | Currency | Link | Date seen | Trap? | Conf. |
|---|---|---|---|---|---|---|---|
| **ASUS KRPA-U16** | **Taobao via Superbuy** | **457.94** | USD | Superbuy listing (owner browser read) | **2026-11-05** | **Variant note: "does not come with 10-gigabit"** — this is the non-10G SKU. BMC is an **optional ASMB9-iKVM module**, so confirm whether the $457.94 variant includes it. | **High** (price) / Medium (BMC) |
| ASUS KRPA-U16 | eBay US / DE / CN | — | — | ebay.com / ebay.de | 2026-11-05 | **Not verifiable — HTTP 403** from this host, both search and item pages. | — |
| ASUS KRPA-U16 | AliExpress | — | — | aliexpress.com | 2026-11-05 | **Not verifiable — bot wall** (2 KB page, no content). | — |
| ASUS KRPA-U16 | Amazon US / DE | — | — | amazon.com | 2026-11-05 | **Not verifiable — "continue shopping" bot wall.** | — |
| ASUS KRPA-U16 | Alibaba / 1688 | — | — | alibaba.com / 1688.com | 2026-11-05 | **Not verifiable** — Alibaba showroom returned HTTP 410; 1688 returned a 5 KB JS wall. | — |
| ASUS KRPA-U16 | Belarus local | — | — | onliner.by / kufar.by / deal.by / shop.by | 2026-11-05 | **No listing found.** All four sites' search is client-rendered; control queries ("Xeon", "сервер") also returned zero server-side results, so this is **absence of data, not proof of absence**. | — |
| ASUS KRPA-U16 | EU shops | — | — | geizhals.de / mindfactory.de / servershop24.de / bargainhardware.co.uk | 2026-11-05 | **Not verifiable** — Geizhals 403, Mindfactory security check, Bargain Hardware 403; ServerShop24 opened but lists complete servers, no KRPA-U16. | — |
| **ASRock ROMED8-2T** | **Taobao via Superbuy** | **1,026.51** (full variant) | USD | Superbuy listing (owner) | **2026-11-05** | **MAJOR TRAP: a $504.84 "card price" is a different, cheaper variant.** The full board is **2.24×** the ASUS for the same 8 channels. | **High** |
| ASRock ROMED8-2T | eBay / AliExpress / Amazon / Alibaba / 1688 / EU / BY | — | — | — | 2026-11-05 | **Not verifiable** (same blocks as above). | — |
| **Tyan S8030** | **Taobao via Superbuy** | **507.34** | USD | Superbuy listing (owner) | **2026-11-05** | Ships with BMC (AST2500) per vendor spec — **not re-fetched today**; Tyan's spec page returned 404 on the URL I tried, so treat BMC as **medium confidence**. E-ATX form factor — verify chassis fit. | **High** (price) / Medium (spec) |
| Tyan S8030 | eBay / AliExpress / Amazon / Alibaba / 1688 / EU / BY | — | — | — | 2026-11-05 | **Not verifiable** / no listing found. | — |
| **Gigabyte MZ32-AR0** | **Taobao via Superbuy** | **515.68** | USD | Superbuy listing (owner) | **2026-11-05** | BMC onboard per vendor spec (not re-fetched; gigabyte.com returned 403). **E-ATX 305×330** — verify chassis. | **High** (price) / Medium (spec) |
| Gigabyte MZ32-AR0 | eBay / AliExpress / Amazon / Alibaba / 1688 / EU / BY | — | — | — | 2026-11-05 | **Not verifiable** / no listing found. | — |
| **Supermicro H12SSL-i** | **Taobao via Superbuy** | **≈760** | USD | **No listing read.** Derived: your observed **+$100 over the $660.44 clone**. | 2026-11-05 | **Inference, not a live read** — flagged. | **Medium** |
| Supermicro H12SSL-i | ZOL (Beijing dealer reference) | **3,399** ref / **3,264** dealer | CNY | <https://detail.zol.com.cn/servermotherboard/index1355133.shtml> | **2026-11-05** (page timestamp 21:25) | **TRAP: the page itself disclaims it** — "价格仅供参考，购买时请以当地实际销售产品价格为准". Not a Taobao listing; cannot be bought through Superbuy. Converting it to USD ($509.70) is the exact error Part 1 was corrected for. | **High** (that the page says this) |
| Supermicro H12SSL-i | eBay US / DE / CN | — | — | — | 2026-11-05 | **Not verifiable (403).** | — |
| Supermicro H12SSL-NT (different model) | idealo.fr (EU) | **810.00** | EUR | <https://www.idealo.fr/prix/201422378/supermicro-h12ssl-nt.html> | Nov 2026 (search result; page 403) | **Different board** (NT = dual 10G). **Unverified** — price read from a search snippet, not the page. | **Low** |
| **Huananzhi H12D-8D** | **Taobao via Superbuy** | **605.56 – 660.44** | USD | Superbuy listings (owner) | **2026-11-05** | **$660.44 is the bare board with BMC** (variant "H12D-8D Xiaolong EPYC Motherboard [with BMC]"). CPU bundle variants exist but **prices did not render** — treat as board-only, CPU extra ($107.72–$235.64). No public RTX 50-series POST data. | **High** (price) / **Unknown** (50-series) |
| Huananzhi H12D-8D | Chinese press / channel | **3,800+** | CNY | <https://post.smzdm.com/p/arzq5px7/> | **2026-09-23** | Corroborates the $605–660 range at the 6.668 cross-rate (¥3,800 = $569.90). Shows the **+111 % run-up from ¥1,800** in early 2026. | **High** |
| Huananzhi H12D-8D | eBay | — | — | ebay.com/itm/306950121484 (listing exists) | 2026-11-05 | **Price unverified** — item page 403. Listing title confirms "BMC IPMI". | **Low** |
| Huananzhi H12D-8D | AliExpress | — | — | aliexpress.com/item/1005009811966997.html (listing exists) | 2026-11-05 | **Price unverified** — bot wall. | **Low** |
| Huananzhi H12D-8D | Belarus / EU | — | — | — | 2026-11-05 | **No listing found / not verifiable.** | — |
| **Huananzhi H12D-16D** | Taobao via Superbuy | — | — | **no quote provided** | 2026-11-05 | **Not quoted.** Do not treat the estimate below as a price. | — |
| Huananzhi H12D-16D | Chinese press | **"3K多"** (¥3,000–3,999; build uses **3,500**) | CNY | <https://post.smzdm.com/p/a4qd74rk/> | **2026-08-23** | **Estimate**, not a listing. Vendor spec exists: <https://www.huananzhi.com/list_6/650.html> | **Medium** |

### CPUs

| Part | Channel | Price | Currency | Link | Date seen | Trap? | Conf. |
|---|---|---|---|---|---|---|---|
| **EPYC 7452** (32C, 155 W) | **Taobao via Superbuy** | **107.72** | USD | Superbuy listing (owner) | **2026-11-05** | None. **The value pick.** Note: ¥600 in the Chinese channel (2026-09-23) = **$89.98** at the cross-rate — Superbuy's USD display is **+19.7 %** over that. Paying in CNY may be cheaper; verify at checkout. | **High** |
| EPYC 7452 | Chinese channel | **600** | CNY | <https://post.smzdm.com/p/arzq5px7/> | **2026-09-23** | Older than today's read; price has been rising, so treat as a floor. | **High** |
| EPYC 7452 | eBay / AliExpress / Amazon / Alibaba / 1688 / BY / EU | — | — | — | 2026-11-05 | **Not verifiable / no listing.** | — |
| **EPYC 7502P** (32C, 180 W) | **Taobao via Superbuy** | **197.77** | USD | Superbuy listing (owner) | **2026-11-05** | **"P" = 1-socket-optimised.** Same 32 cores as the 7452 for **+$90.05 (+83.6 %)** and +25 W. No workload of yours uses the extra clock. | **High** |
| EPYC 7502P | other channels | — | — | — | 2026-11-05 | **Not verifiable.** | — |
| **EPYC 7K62** (48C, OEM) | **Taobao via Superbuy** | **235.64** | USD | Superbuy listing (owner) | **2026-11-05** | **OEM SKU** — no AMD warranty; seller warranty only. 225–240 W, needs a 3U-class cooler. +128 % over the 7452 for +16 cores you cannot use. | **High** (price) / Medium (TDP) |
| EPYC 7K62 | eBay | **C$411.32 ≈ $294.92** | CAD→USD | ebay.ca listing "from China, jinmu666" | read 2026-11-05 via search result | **Unverified** — item page 403; price from snippet. **USED/REFURB** not stated. | **Low** |
| EPYC 7K62 | China domestic | **≈900** (proj.) | CNY | — | — | **SPECULATION** — scaled 48/32 from the sourced ¥600 7452. No listing. | **Low** |
| **EPYC 7R32** (48C, OEM) | **Taobao via Superbuy** | **269.30** | USD | Superbuy listing (owner) | **2026-11-05** | OEM/cloud part; no AMD warranty. +$33.66 over the 7K62 for the same core count — **buy the 7K62 instead**. | **High** (price) |
| **EPYC 7B13** (64C, OEM AWS) | any channel | — | — | — | 2026-11-05 | **No verified price in any channel I could open.** Do not budget from a number that does not exist. OEM/cloud SKU with firmware-lock risk. | — |

### RAM

| Part | Channel | Price | Currency | Link | Date seen | Trap? | Conf. |
|---|---|---|---|---|---|---|---|
| **16 GB DDR4-2133 ECC RDIMM** (SK Hynix 2RX4) | **Taobao via Superbuy** | **61.96** | USD | Superbuy listing (owner) | **2026-11-05** | None. **New**, and cheaper than US used. | **High** |
| 16 GB DDR4-2133 RDIMM | DIMMsum (US **USED**, eBay-derived) | **70.00** | USD | <https://dimmsum.com/ddr4/16gb-2133-rdimm> | **2026-11-05** | **USED.** $4.38/GB. +$8.04/stick vs Superbuy. | **High** |
| 16 GB DDR4-2133 RDIMM | eBay US/DE | — | — | — | 2026-11-05 | **Not verifiable (403)**; DIMMsum above is the eBay-derived aggregate. | — |
| 16 GB DDR4-2133 RDIMM | Belarus / EU / Alibaba / 1688 / AliExpress / Amazon | — | — | — | 2026-11-05 | **No verifiable listing.** | — |
| **32 GB DDR4-3200 RDIMM** | **Taobao via Superbuy** | **197 – 233** | USD | Superbuy listings (owner) | **2026-11-05** | Range = two listings; confirm rank (2RX4 vs 2RX8) before mixing. | **High** |
| 32 GB DDR4-3200 RDIMM | DIMMsum (US **USED**) | **261.00** | USD | <https://dimmsum.com/ddr4/32gb-3200-rdimm> | **2026-11-05** | **USED.** +$28–64/stick vs Superbuy. | **High** |
| 32 GB DDR4-3200 RDIMM | Chinese channel (new) | **≈1,200** | CNY | <https://post.smzdm.com/p/a4qd74rk/> | **2026-08-23** | Dated; ¥1,200 = **$179.96**, below today's Superbuy range — the market has moved up. | **Medium** |

### Channels checked, no usable data (stated once, not repeated per row)

| Channel | What happened | Consequence |
|---|---|---|
| **eBay US / DE / CN** | HTTP **403** on search and item pages, both `urllib` and `curl` with browser headers | **No eBay price in this report is verified.** The one eBay figure (7K62, C$411.32) is a search snippet, marked Low. |
| **AliExpress** | 2 KB bot wall on item pages | No verified AliExpress price. Listings exist (H12D-8D, H12D-16D); prices unknown. |
| **Amazon US / DE** | "Continue shopping" bot wall | No verified Amazon price. |
| **Alibaba** | Showroom HTTP 410; product pages not reachable | No verified Alibaba price. |
| **1688** | 5 KB JS wall | **Likely the cheapest Chinese source, but zero verified prices.** An agent can buy 1688; this is the one channel worth a manual check. |
| **Belarus local (kufar, onliner, deal.by, shop.by)** | Pages open; **search/catalog is client-rendered**. Control queries ("Xeon", "сервер") returned zero server-side results too | **No local EPYC price obtained.** See P2.3. |
| **EU (Geizhals, Mindfactory, Bargain Hardware)** | 403 / security check / 403 | No verified EU price. |
| **ServerShop24 (DE, REFURB)** | Opened (1.7 MB); sells complete servers, e.g. GIGABYTE G292-Z20 with EPYC 7402P, 32 GB, **1,839.99 €** | **REFURB.** No bare SP3 board or CPU found in its listing. |

## P2.2 Cheapest channel per part — and the cheapest complete build

| Part | Cheapest verified channel | Price | Runner-up | Gap |
|---|---|---|---|---|
| Single-socket SP3 board | **Superbuy — ASUS KRPA-U16** | **$457.94** | Tyan S8030 $507.34 | **+$49.40** |
| Dual-socket SP3 board | **Superbuy — H12D-16D** (estimate) | **≈$524.90** | no second verified dual board | unverified |
| 32-core CPU | **Superbuy — 7452** | **$107.72** | Chinese channel ¥600 = $89.98 (dated) | Superbuy +19.7 % vs Sept CNY |
| 48-core CPU | **Superbuy — 7K62** | **$235.64** | eBay C$411.32 = $294.92 (Low conf.) | **−$59.28** vs eBay |
| 64-core CPU | **none verified** | — | — | — |
| 16 GB 2133 RDIMM | **Superbuy (new)** | **$61.96** | DIMMsum US used $70.00 | **−$8.04** and new |
| 32 GB 3200 RDIMM | **Superbuy (new)** | **$197–233** | DIMMsum US used $261 | **−$28 to −$64** and new |

**Answer: Superbuy/Taobao is the cheapest channel for every single line item.** It beats US used RAM, EU retail, and (on the one comparable CPU) eBay. The alternates are not close. The only plausible cheaper source is **1688 via an agent**, which I could not verify at all — treat it as a manual check worth 10 minutes, not as a number.

### The cheapest complete 8-channel single-CPU build (mixed sourcing, item prices only)

| Line | Part | Price | Channel |
|---|---|---|---|
| Board | **ASUS KRPA-U16** | **$457.94** | Superbuy |
| CPU | **EPYC 7452** (32C, 155 W, gen4, 8 ch) | **$107.72** | Superbuy |
| RAM | **4× 16 GB DDR4-2133** (joins your existing 4 → 8×16 = **128 GB**, 8 channels) | **$247.84** | Superbuy |
| **Total** | | **$813.50** | |
| *Alternative board* | Tyan S8030 (BMC included, E-ATX) | **$862.90** | Superbuy |
| *From scratch (8 new sticks)* | ASUS + 7452 + 8×16 | **$1,061.34** | Superbuy |
| Cooler | **not priced** — no verified listing in this pass | — | — |

**This is $813.50 versus $660.44 for the bare clone board alone.** The clone board plus a 7452 plus the same four sticks is **$1,016.00** — **$202.50 more** than the genuine ASUS build for the same 128 GB / 8 channels / gen4 x16, with worse firmware and no 50-series evidence.

## P2.3 Belarus local reality

**What I could establish:** nothing priced. Onliner's catalogue, Kufar, deal.by and shop.by all opened, but every search is client-rendered; **control queries ("Xeon", "сервер") also returned zero results**, so a zero result for "EPYC" proves nothing. **I am correcting Part 1 here:** its line "Onliner returns nothing for EPYC" was based on a search that returns nothing for *anything*, so it is not evidence of absence.

**What is true and useful:**
- No Belarusian **retail** listing for an SP3 board or EPYC CPU was found on any page I could open. Server parts in Minsk are a classifieds market, not a shop shelf.
- The prior route memo measured the local classifieds market for GPUs (Kufar, 30.10.2026: used 3090 asking **4,400–5,000 BYN = $1,476–$1,677** at today's 2.9806 BYN/USD), so the local used market is real and priced at a **premium to imports**, not a discount.
- **Direction of the comparison:** a $457.94 board is ~**1,365 BYN** at 2.9806. A local seller who imported it will ask more than that, because they paid the same freight and want a margin. **Import is the cheaper direction for this class of part**; local is only better for speed and returns.
- Customs is out of scope per your instruction — no duty, VAT or threshold arithmetic appears in this part.

**Bottom line for Belarus:** buy in the Chinese channel; expect local to be **at or above** the import item price, with no EPYC inventory visible. If a local board appears at ≤1,300 BYN, it is worth looking at purely for the return option.

## P2.4 The 1-vs-2 decision, restated on today's cheapest parts

### The two builds, item prices only

| | **Single socket** | **Dual socket** |
|---|---|---|
| Board | ASUS KRPA-U16 **$457.94** | H12D-16D **≈$524.90** (estimate, unverified) |
| CPUs | 1× 7452 **$107.72** | 2× 7452 **$215.44** |
| RAM to reach 128 GB | 4 new ×16 GB = **$247.84** (joins your 4) | *not a sensible config* |
| RAM to reach 256 GB | 8× 32 GB = **$1,576** (replace all) | **12** new ×16 GB = **$743.52** (joins your 4) |
| **Total at 128 GB** | **$813.50** | — |
| **Total at 256 GB** | **$2,141.66** | **$1,483.86** |
| Case | ATX per vendor spec — yours should fit; verify | **E-ATX 305×335** — verify, likely new case |
| PSU | your 1350 W is enough | your 1350 W is enough |
| Cooler | 1× (unpriced) | 2× (unpriced) |

### The exact crossover — and it is not where you would guess

Dual does **not** pay for bandwidth (a single process stays on one NUMA node: ~120 GB/s either way), does **not** pay for cores (your profile is launch-bound at 2 % MFU and bandwidth-bound at decode), and does **not** pay for PCIe lanes (2 GPUs need 32 of 128).

**Dual pays at exactly one thing: cheap DIMM slots.** 16 GB sticks cost **$3.87/GB**; 32 GB sticks cost **$6.16–7.28/GB**. So the crossover is a *capacity* crossover:

| Capacity | Single-socket cost (board+CPU+RAM) | Dual cost (board+2 CPU+RAM) | Winner |
|---|---|---|---|
| **128 GB** | **$813.50** | not sensible (2nd socket empty) | **Single, by default** |
| **192 GB** | $1,353.66 (4×16 yours + 4×32 — mixed sizes, **not recommended**) | $1,236.02 (4×16 yours + 8×16 new) | **Dual, by $117.64** — the crossover region |
| **256 GB** | $2,141.66 | **$1,483.86** | **Dual, by $657.80** |
| **512 GB** | 8×64 GB — no verified price; US used 64 GB 3200 is $650 → **≈$5,765.66** | 16×32 GB = $3,152 → **$3,892.34** | **Dual, by ~$1,873** |

**The crossover is ~160 GB of installed RAM.** Below it, single wins on total price. Above it, dual wins — **purely because you can fill 16 slots with cheap 16 GB sticks instead of 8 slots with expensive 32/64 GB sticks.** It is a RAM-pricing accident, not an architecture argument.

Your need is **128 GB** (a 60 GB model + OS + KV). **That is below the crossover.** One CPU.

### $/×step, restated

Your measured baseline: 0.40 s/step single-GPU; gen3 x8 second GPU gives ×1.03; gen4 x16 gives ×1.62. The platform buys the **+0.57×** between them.

| Path | Item cost | $ per +1.0× | $ per +0.1× |
|---|---|---|---|
| Single socket + 2nd GPU | $813.50 + $750 = **$1,563.50** | **$2,743** | **$274** |
| Dual socket + 2nd GPU | $1,483.86 + $750 = $2,233.86 | $3,919 | $392 |

Dual is **43 % worse per unit of training speed** and adds a case and a second cooler. For training, the correct comparison is still **$813.50 of H100 time = 233 h at the $3.49 median (484 h at $1.68)** — more than your 96M-parameter campaign needs, with zero capex.

## P2.5 Three builds — all-in item prices, mixed channels

### (i) Minimum viable 8-channel single-CPU

| Line | Part | Price |
|---|---|---|
| Board | ASUS KRPA-U16 | $457.94 |
| CPU | EPYC 7452 | $107.72 |
| RAM | 4× 16 GB 2133 (→128 GB with yours) | $247.84 |
| **Total** | | **$813.50** |
| Cooler | not priced | — |

**Inference:** 45.8 tok/s at 30 GB · 22.9 tok/s at 60 GB → **$17.76 /tok/s** (30 GB), **$35.52 /tok/s** (60 GB).
**Training:** ×1.62 with a 2nd GPU → **$2,743 per +1.0×**.

### (ii) Cheapest build that runs a 60 GB model

**Identical to (i).** 128 GB holds 60 GB of weights + OS + KV with ~60 GB spare. There is no cheaper separate configuration; the only cheaper way to *run* 60 GB at all remains the $247.84 RAM-only upgrade on your X10SRL-F (~13 tok/s, 4 channels).

| Line | Price |
|---|---|
| Build (i) | **$813.50** |
| *Optional 256 GB for 100B-class at fp16* | +$1,328.16 (8×32 replace) → **$2,141.66**, or go dual → **$1,483.86** |

### (iii) Dual socket

| Line | Part | Price |
|---|---|---|
| Board | H12D-16D (estimate — **unverified**) | $524.90 |
| CPUs | 2× EPYC 7452 | $215.44 |
| RAM | 12× 16 GB 2133 (→256 GB with yours) | $743.52 |
| **Total** | | **$1,483.86** |
| Case | E-ATX — not priced | — |
| Coolers | 2× — not priced | — |
| PSU | your 1350 W — $0 | $0 |

**Inference:** 22.9 tok/s single process (**$64.80 /tok/s** at 60 GB) · 45.8 tok/s aggregate with two processes (**$32.40 /tok/s**).
**Training:** ×1.62 with a 2nd GPU → **$3,919 per +1.0×** — worse than single.

## P2.6 Buy this / don't buy that

**BUY**
- **ASUS KRPA-U16 — $457.94.** Cheapest verified 8-channel SP3 board in any channel. **ATX and 4× PCIe 4.0 x16 are vendor spec, not re-fetched today — verify before ordering**; it is the fit-safe choice for your chassis if the spec holds. **Confirm the variant includes the BMC module** (the $457.94 SKU is the non-10G one; the ASMB9-iKVM may be extra).
- **Tyan S8030 — $507.34** if you want BMC included and can fit E-ATX. **$49.40 more** than the ASUS, and it removes the BMC question.
- **EPYC 7452 — $107.72.** 32 cores, 155 W, 8 channels, gen4. The cheapest part that maxes the platform for your two workloads.
- **4× 16 GB DDR4-2133 — $61.96 each.** New, cheaper than US used, and identical to what you already own, so no mixing penalty.
- **Total: $813.50** for a complete 8-channel 128 GB single-CPU platform (cooler extra).

**DON'T BUY**
- **The Huananzhi H12D-8D at $605.56–$660.44.** It is **$148–$203 more expensive** than the ASUS for the same socket, same 8 channels, same gen4 x16 — and it is the only board in the set with **no public RTX 50-series POST record**, no QVL, no DIMM population table, and a documented BMC that can hang. The $660.44 is the bare board; adding the $107.72 CPU puts it at **$768.16** before RAM.
- **The ASRock ROMED8-2T at $1,026.51.** **2.24×** the ASUS for the same memory topology. The **$504.84 "card price" is a different variant** — do not compare it to the $1,026.51 full board.
- **The 7502P at $197.77** (+83.6 % over the 7452 for +13 % clock you cannot use) and the **7R32 at $269.30** (+$33.66 over the 7K62 for the same 48 cores).
- **Any dual-socket build at 128 GB.** It costs **$422.52 more** than single (board +$66.96, second CPU +$107.72, and you still populate only 8 of 16 slots) for zero tok/s and zero training gain. Dual only becomes rational at **≥160 GB of RAM**, where cheap 16 GB sticks exploit the extra 8 slots — a capacity trick, not a speed trick.
- **Any board for training reasons.** $813.50 of H100 rental is **233–484 GPU-hours**; your bottleneck is 2 % MFU and launch overhead, which no motherboard fixes.

## P2.7 Corrections to Part 1 (read before trusting it)

1. **Board ranking is reversed.** Part 1 said the clone was the value pick because Supermicro was ~$760 in-channel. Your live reads put **three genuine boards below the clone**. The clone is now the most expensive single-socket option of the five.
2. **Part 1's "Onliner shows no EPYC" is retracted.** The Onliner search returns nothing for control queries too; it is not evidence.
3. **Part 1's build totals included a channel factor and duty.** Superseded: item prices only, per your instruction.
4. **Part 1's "$660.44 landed" framing** should read **"$660.44 item price for the bare board with BMC"** — no shipping or duty is included, and the CPU is extra.
5. **Unchanged:** the 1-vs-2 conclusion (one CPU), the bandwidth arithmetic, the gen4 x16 speedup, the H100 comparison, and the first-hour test protocol.

## P2.8 Confidence

| Claim | Confidence | Basis |
|---|---|---|
| Superbuy prices (boards, CPUs, RAM) | **High** | Your live browser reads, 2026-11-05 |
| Superbuy is cheapest on every line | **High** for verified lines; **Medium** overall | 1688 is unverified and could undercut it |
| Clone is more expensive than ASUS/Tyan/Gigabyte | **High** | $605.56–660.44 vs $457.94 / $507.34 / $515.68 |
| ASUS/Tyan/Gigabyte socket, DIMM count, PCIe gen, BMC | **Medium** | Vendor spec, **not re-fetched** (403/404 from here) |
| Dual crossover at ~160 GB | **Medium-High** | Arithmetic on verified stick prices; depends on 16 GB stock lasting |
| tok/s projections | **Medium** | Linear from your measured 26 tok/s, 12 % haircut for 8 channels |
| eBay / AliExpress / Amazon / Alibaba / 1688 / EU prices | **None** | All blocked; no number in this part comes from them except the flagged 7K62 snippet |
| Belarus local prices | **None** | Search is client-rendered; no listing obtained |
| 7B13 price | **None** | No verified price anywhere |

## P2.9 Open questions that could still move the decision

1. **Does the $457.94 KRPA-U16 include a BMC?** If not, add an ASMB9-iKVM module — still likely cheaper than the $507.34 Tyan, but confirm.
2. **1688 price for the same four boards.** Unverified here; an agent quote is the only remaining lever that could beat Superbuy.
3. **Does the 5070 Ti POST on the KRPA-U16?** Better odds than the clone, still not a guarantee. Return window is the mitigation.
4. **Is the $605.56 clone listing the BMC version?** If it is the no-BMC variant, the comparison is even worse for the clone.
5. **Tyan S8030 form factor** — if it is E-ATX 305×330, your current chassis may not take it; the ASUS is the fit-safe choice.
