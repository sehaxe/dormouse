# Memory & silicon cycle — buy the platform now or wait?

**Date:** 2026-11-06 · **Buyer:** Minsk · **Channel:** Taobao→Superbuy, eBay/AliExpress alternates
**Question:** the $941.84 EPYC 7452 / Tyan S8030 / 4×16 GB DDR4-2133 build — purchase now, or wait for a price reset?
**Answer:** **Buy the memory now, this week. Buy the board + CPU now too, unless the second GPU is more than ~6 months away.** No scheduled event lowers any line item before H2-2027, and the one line item with a documented double-digit 3-month rise (DDR4) has a closing supply window, not an opening one.

---

## 0. How to read this

| Tag | Meaning |
|---|---|
`primary` | The price/forecast publisher's own table or press release, fetched today |
`retail-live` | A listing page fetched today, with timestamp |
`press` | Dated trade press reporting a primary source (TrendForce release, DigiTimes, Reuters) |
`aggregator` | A tracker that republishes retail data (DIMMsum, videocardprices, GPU Economy) |
`analyst` | A named house's forecast, reported second-hand |
`SPECULATION` | My inference. No source asserts it. Always labelled. |

FX used only where a source is in CNY: **6.668 CNY/USD** (NBRB, 2026-11-05, carried from the platform report). Your item prices are already USD, so most arithmetic needs no FX.

**Evidence conflict declared up front:** TrendForce's own 30 Sep release puts Q4-2026 conventional DRAM contract prices at **+10-15% QoQ**. Two secondary analyses read the same quarter at **+0-5%** and **+3-8%**. Both are reported below; the discrepancy is the single most important unresolved number in this report, and it settles in November.

---

## 1. DRAM cycle state, 2026-11-06

### 1.1 Spot, today (primary)

TrendForce DRAM spot table, last update **2026-11-05 18:10 GMT+8** — <https://www.trendforce.com/price/dram/dram_contract> (the spot panel is embedded on the same page).

| Item | Session avg, USD/chip | Day change | Intraday range | $/GB | Grade |
|---|---|---|---|---|---|
| DDR5 16Gb (2Gx8) 4800/5600 | **60.667** | −1.09% | 46.20 – 73.00 | 30.33 | primary |
| DDR5 16Gb eTT | 30.25 | 0.00% | 29.50 – 31.50 | 15.13 | primary |
| **DDR4 16Gb (2Gx8) 3200** | **81.679** | −0.30% | **45.50 – 117.00** | **40.84** | primary |
| DDR4 8Gb (1Gx8) 3200 | **46.036** | −0.46% | 25.80 – 83.00 | 46.04 | primary |
| DDR4 8Gb eTT | 13.475 | −0.74% | 12.00 – 14.50 | 13.48 | primary |
| DDR3 4Gb 1600/1866 | 13.972 | −0.75% | 9.90 – 21.00 | 27.94 | primary |

Three facts fall out of that table, and they matter more than the percentages:

1. **DDR4-16Gb spot ($81.68) is 35% more expensive than DDR5-16Gb spot ($60.67).** The older generation trades at a premium to the newer one. That is an EOL market, not a normal one.
2. **The day changes are all negative** (−0.3% to −1.1%) while contract prices are still rising. Spot has gone flat-to-soft. It is one day, not a two-week trend — TrendForce's own 7 Oct note called spot "largely sideways" — but it is the first non-upward print in this cycle. Treat it as an early-warning flag, not a turn.
3. **The intraday spreads are 2–3× wide** (DDR4 16Gb: low $45.50, high $117.00). That is thin liquidity at the top of the book, not a clearing price. Do not budget from a single spot print — which is exactly what §3 does *not* do.

### 1.2 Contract, latest public print

TrendForce contract table, "2H Sep", last update **2026-09-30 17:00 GMT+8**; the October update sits behind a paid report (`RP261030WD`, 2026-10-30 — its abstract says PC DRAM "keep rising in the fourth quarter, though gains are narrowing"). Same URL as above.

| Item | Session avg, USD | Session change | Grade |
|---|---|---|---|
| DDR4 16Gb 2Gx8 | **52.00** | ▲ 4.00% | primary |
| DDR4 8Gb 1Gx8 | **26.00** | ▲ 4.00% | primary |
| DDR4 16GB SO-DIMM | 275.00 | ▲ 1.85% | primary |
| DDR5 8GB SO-DIMM | 133.00 | flat | primary |
| DDR3 4Gb 256Mx16 | 15.20 | flat | primary |
| DDR3 4Gb 512Mx8 (spot) | 13.97 | ▼ 0.75% | primary |

Module spot, **2026-10-26 14:40 GMT+8** (same page):

| Item | Weekly avg, USD | Change |
|---|---|---|
| DDR5 UDIMM 16GB 4800/5600 | 241.67 | ▲ 0.70% |
| DDR5 RDIMM 32GB 4800/5600 | 2,165.00 | flat |
| DDR4 UDIMM 16GB 3200 | 164.20 | ▼ 0.85% |

> **Data-quality warning.** A 32 GB DDR5 RDIMM at $2,165 is $67.65/GB, against a chip-content cost of ~$971 (16 × $60.67) — a 2.2× module-over-chip premium. Normal is ~1.1×. DIMMsum has *used* 32 GB DDR4-3200 RDIMMs at $261 ($8.17/GB) the same week. The module-spot line is thin and non-representative; **I do not use it in any arithmetic below.** Flagged `primary` for what it says, `low` for what it means.

### 1.3 The trajectory — this is the actual cycle state

| Quarter | Conventional DRAM contract, QoQ | Source, date | Grade |
|---|---|---|---|
| Q1 2026 | **+90–95%** | TrendForce via Luminix, updated 2026-09-30 | analyst |
| Q2 2026 | **+58–63%** | TrendForce, 2026-06-01, cited in same | press |
| Q3 2026 | **+13–18%** forecast; suppliers notified +20–30% | TrendForce 2026-07-03; ADATA via Economic Daily News | press |
| **Q4 2026** | **+10–15%** (TrendForce, 30 Sep) **vs +0–5%** (Goldman reading) **vs +3–8%** (July survey) | three sources, conflicting | **conflicted** |
| Q1 2027 | no published point estimate; "gain narrowing" | TrendForce 2026-10-30 abstract | press |
| 2027 full year | DRAM **sufficiency ratio −1% to −2% in 2026, gap widening in 2027**; server shipments grow faster than 2026's +17%; RDIMM bit supply +15–20% vs faster CPU shipments | TrendForce press release **2026-07-30**, <https://www.trendforce.com/presscenter/news/20260730-13158.html> | primary |
| New fab volume | "meaningful production ramp-ups until **2H27**… substantial output not until **2028**" | same release | primary |

**YoY, in money:** DDR4 8Gb spot went $32.30 (mid-May) → $42.50 (10 Aug, +32% in three months) → $46.04 (5 Nov). Retail: the cheapest 32 GB DDR5-6000 kit in the US was $449.97 on 14 Oct, RAM "up more than 500% since the previous year" (Tom's Hardware, 2026-10-14, via originalpricing.com summary). A 32 GB DDR5 kit was $479.99 on **2026-11-02** against an all-time-low near $95 pre-shortage (tech-insider, citing Tom's index). Consumer-tracker cross-check: DDR5 **$19.71/GB (+1.3% MoM), DDR4 $8.47/GB** in November 2026 — <https://rampricesusa.com/insights>.

**Is supply still being cut?** Yes, and the paperwork is closing:

- Samsung notified DDR4 EOL in 2025, then **delayed** the phase-out to 2026 because DDR4 became "an unexpected cash cow" — TrendForce, 2025-09-02.
- Micron's DDR4 last-time shipment **passed in Q1 2026**; Samsung and SK hynix "stopped taking new orders" and extended long-term support only "through late 2026" — electronics-sourcing.com, **2026-07-07**.
- Samsung's formal LTB deadlines for most legacy DDR4 run **October 2026 → March 2027**, density-dependent — limchip.com market update, 2026 (secondary; treat as `analyst`).
- TrendForce, **2026-10-28**: "As the three major suppliers phase out mature process nodes, **DDR4 supply is set to tighten notably**" (`RP261028TO` abstract).
- Micron/Samsung/SK hynix inventories: **below 10 days of supply** in September (KB Securities estimate via DigiTimes, 2026-09-07 — an analyst estimate, not a disclosure).

**Answer to Q1:** contract prices still rising, decelerating from +95% to ~+13%; spot rolling flat-to-soft; DDR4 in structural decline of *supply* with no decline of demand, which is why it is now the most expensive DRAM per chip. Supply is not merely being cut — the last-time-buy windows close between now and March 2027.

---

## 2. Is a price reset plausible in the next 6–12 months?

**No — and I can name every candidate event and show none of them is scheduled inside the window.** A "reset" here means a sustained decline, not a plateau.

| Reset mechanism | Would need | Dated status | Lands inside 12 months? |
|---|---|---|---|
| New fab capacity | Micron Idaho Fab 1, Samsung Taylor TX, SK hynix Indiana/Yongin/M15X | TrendForce 2026-07-30: ramp 2H27, *substantial* output 2028. M15X started Feb 2026 (~4 months early) but is HBM-only, ~10k wpm scaling to 40–70k through 2027 | **No** |
| Chinese supply (CXMT/YMTC) | Meaningful server-DRAM volume | CXMT ~10% of DRAM revenue in Q2 2026, DDR5 platform started; entity-list limits on advanced nodes; "meaningful volume not until late 2027 at the earliest" (TrendForce/IDC via exIT, 2026-10-26) | **No** |
| AI-demand cooldown | Hyperscaler capex cut or flattening | **No hyperscaler has cut.** Alphabet raised 2026 to $195–205B; Microsoft ~$190B; top-9 CSP ~$830B (Luminix scorecard, 2026-09-30, citing the Q2 round). Backlogs: Microsoft RPO $627B (+99% YoY) | **No** |
| DDR4 wind-down *ending* | Last-time-buys completed, demand migrating to DDR5 | LTB deadlines **Oct 2026 – Mar 2027**; after that there is no new supply at all, only pulled stock | **This makes it worse, not better** |
| Demand destruction in PC/mobile | Already happening | IDC: 2026 PC shipments **−11.3%**, smartphones **−12.9%** — and DRAM prices still rose. AI-server demand absorbed it | **Already priced in** |
| Financial-market repricing | Equity investors calling the top | Memory stocks fell sharply on **2026-10-13**; Micron's FQ1-27 guide is the first sequential gross-margin decline of the cycle (~86.25% from ~87%) — Luminix, 2026-09-30 | **Happening, but equities lead prices by 2–4 quarters** |

**The one real bear case, named:** Raymond James' Karl Ackerman forecast in June 2026 that DRAM/NAND ASPs would **peak mid-2026 and decline for two quarters from early 2027** — reported by exIT, 2026-10-26. That is the only credible published call for a decline inside 12 months, and even its author frames it as "a slowdown cushioned by long-term supply agreements, not a collapse." Against it: SK hynix's CEO told Reuters (2026-07-10) that **2027 will be "the worst year in the industry's history from the supply perspective."**

**Answer to Q2:** a *plateau* in conventional DRAM contract prices is plausible in H1-2027. A **reset in the sense of "cheaper in 6–12 months" is not supported by any primary forecast**, and the DDR4 segment specifically has a supply cliff (LTB expiry) rather than a supply wave in that window.

---

## 3. DDR4 specifically: is it in the EOL-premium phase?

**Yes. This is not a forecast — it is visible in today's price table.** Three independent pieces of evidence:

**(a) Generational inversion.** DDR4 16Gb spot $81.68 vs DDR5 16Gb $60.67 = **DDR4 is 1.35× DDR5 per chip** (TrendForce, 2026-11-05). A superseded part costing more than its replacement is the definition of scarcity pricing.

**(b) Chip-over-module inversion.** TrendForce flagged that DRAM *chip* quotes in $/Gb have risen above *assembled module* quotes of equivalent capacity — normally impossible, since modules carry assembly and test cost. The mechanism it gives: module houses are still selling cheaper inventory, and **the spread closes by modules repricing up, typically within 2–4 weeks**; Kingston's module hike was the confirmation (TrendForce via SpecForge, **2026-10-05**, <https://www.sourcebyspec.com/news/dram-spot-vs-contract-gap-october-2026-what-the-inversion-tells-buyers.html>). **This is the single most decision-relevant fact in the report for your $62 sticks** — see (d).

**(c) The arithmetic of new-build cost.** A 16 GB DDR4-2133 2Rx8 RDIMM is 18 × 8Gb dies. At the 2H-Sep contract price of $26.00/chip, the silicon alone is **$468**; at spot ($46.04) it is **$829**. You are being quoted **$61.96** for the finished, tested module. That price is 13% of new-build silicon cost. It cannot be a new-production price — it is a *pulled-module* price, and pulled modules are the only DDR4 supply that will exist after March 2027.

**(d) 16 GB 2133 RDIMM at $62: above or below trend?**

| Reference | $/GB | $/16 GB stick | Date | Source |
|---|---|---|---|---|
| Your listing (new, Superbuy) | **$3.87** | **$61.96** | 2026-11-06 | your cart |
| eBay sold median (completed sales) | $2.86 | $45.76 | **Jul 2026** (last published month) | DIMMsum |
| eBay sold median | $3.41 | $54.50 | Jun 2026 | DIMMsum |
| eBay sold median | $3.72 | $59.49 | Dec 2025 | DIMMsum |
| eBay low-end **ask** (10th pct, shipped) | $3.44 | $55.00 | 2026-08-08 | DIMMsum |
| eBay low-end ask | **$4.38** | **$70.00** | **2026-11-05** | DIMMsum |
| eBay median ask | $6.38 | $102.00 | 2026-11-05 | DIMMsum |
| Cheapest live single stick | $3.46 | $55.39 | 2026-11-05 | DIMMsum |
| US consumer DDR4 average | $7.77–8.47 | — | Nov 2026 | capitalandcompute; rampricesusa |

Source for all DIMMsum rows: <https://dimmsum.com/ddr4/16gb-2133-rdimm>, fetched 2026-11-05 (sold series is monthly median of completed sales; asks are live listings, refreshed twice daily).

**Verdict on $62:** **at trend, not above it.**
- 12-month range of *sold* prices: $2.86 (Jul) to $3.72 (Dec 2025). $3.87 is 4% above the 12-month high of the sold series and 35% above the July trough.
- But the *ask* market moved **+27% in three months** ($3.44 → $4.38 low-end, 2026-08-08 → 2026-11-05). Against today's cheapest eBay used stick you are **12% cheaper**; against the median ask, **39% cheaper**.
- Read plainly: **you are buying below the current used market and above the July bottom.** The listing has not yet repriced to November. That is a stale quote in your favour, and §3(b) says stale module quotes close *upward*.

### 3.1 The DDR3 precedent — with an honest gap

What I can verify today: DDR3 4Gb is quoted at **$13.97 spot / $15.20 contract** (TrendForce, 2026-11-05 and 2026-09-30) — i.e. **$27.94/GB**, which is within ~30% of DDR4's $40.84/GB at 16Gb density and $46.04/GB at 8Gb density, despite DDR3 being two generations older and out of mainstream production for years. It still clears at that level because industrial, medical, networking and telecom equipment cannot migrate.

What I **could not** verify from a primary source in this pass: a dated pre-EOL DDR3 price point, and therefore the % appreciation. Multiple secondary sources say DDR2/DDR3 index prices rose "up to 60%" in 2026 and that DDR3 motherboard sales in China jumped 2–3× in January 2026 (videocardz/Board Channels, via Windows Central, 2026-06-17), and TrendForce's 2026-04-07 note describes demand spilling from DDR4 into DDR3/DDR2. **I am not putting a percentage on the DDR3 precedent — mark it UNVERIFIED.** 

The *structural* lesson from DDR3 is verifiable and is the one that matters: **EOL memory does not collapse in price; it plateaus at a premium and stays there for years**, because the installed base cannot move and the last fabs have moved on. That is the phase DDR4 entered when Samsung's LTB windows opened.

---

## 4. Silicon, nodes, and GPU street prices

### 4.1 TSMC wafer prices (press-reported; TSMC does not publish these)

| Node | 300 mm wafer price | Change | Date | Grade |
|---|---|---|---|---|
| N3 | ~$20,000 | 2026 hike **5–10%** across N5/N4/N3 | DigiTimes via TechSpot; siliconanalysts "updated Aug 2026" | press |
| N5 | ~$18,500 | same | same | press |
| N7 | ~$9,500 | included in "7 nm and below" hike | same | press |
| N2 | ~$30,000 | +50% step over N3 | sourcebyspec, 2026 | analyst |
| CoWoS packaging | — | **up to +20% under discussion**, not confirmed, no effective date | TrendForce/Commercial Times **2026-10-27**; Tom's Hardware same week | press |
| Equipment lead times | — | 3–6 months → **up to 8 months** | Tom's Hardware, Oct 2026 | press |

Also reported: TSMC notified customers of hikes "across all advanced nodes from 7 nm onward, covering ~74–75% of wafer revenue" (Culpium, 2026-06-23, via tmt-blog). **Direction is unambiguously up.** Nothing in the wafer or packaging pipeline is deflationary inside 12 months.

### 4.2 GPU street prices — up, sharply, and the refresh is dead

| Card | MSRP | Street, dated | Premium | Source |
|---|---|---|---|---|
| RTX 5070 Ti (new, MSI Ventus) | $749 | **$1,439.99** | **+92%** | Newegg via videocardprices, fetched 2026-11-05 14:38 UTC |
| RTX 5070 Ti (used, Gigabyte OC) | — | $1,399.00 | — | eBay via same, 2026-11-05 |
| RTX 5070 Ti (tracked history) | $749 | new $1,439 / used $1,030; 12-mo avg $891; low $650 (2025-12-29) | — | bestvaluegpu, Nov 2026 (page 429'd on re-fetch; search snippet) |
| RTX 5090 | $1,999 | **$3,340** (Nov 2026) — and **~$4,999 on 2026-09-01** | +67% to +150% | tech-insider 2026-11; capitalandcompute 2026-09 |
| RTX 50 series, November move | — | **+$500** on 5090/5080, **+$150** on 5070 Ti, **+$30** on 5060 8GB | — | wccftech / Dataconomy / techbriefly, all **2026-11-03** |

Mechanism, from the board partners' side: Nvidia raised the **GPU+memory kit price it charges partners three times in 2026**, compounding to roughly **+32–49% versus January** (capitalandcompute, Sep 2026). Supply to partners was cut ~20% and the 5070 Ti was sidelined from shelves (TechTimes, 2026-06-21).

**RTX 50 Super: cancelled, not delayed.** Gamescom 2026 confirmed the cancellation; Nvidia is focusing on RTX 60 (gamegpu, Aug 2026). Digital Foundry (2026-07) had the arithmetic: a 3 GB GDDR7 module costs **$60–70 vs $20** for the 2 GB part, so 8 modules on a 5070 Ti Super add **$360** to the BOM, 6 on a 5070 Super add **$270**. There is no version of the Super refresh that launches at a lower price than today's cards.

**Consequence for you:** you already own the 5070 Ti. That was the correct call and it is now un-repeatable at MSRP. **Do not wait for a GPU price fall to buy a second card** — the fall is not scheduled, and the product that would have caused it (Super) does not exist. If a second GPU is on the roadmap, its price risk is upward.

---

## 5. Server platform cycle: SP3 boards and DDR4 RDIMM vs DDR5

### 5.1 The two reads I have

| Datum | Value | Date | Source | Grade |
|---|---|---|---|---|
| Huananzhi H12D-8D clone board, channel price | **¥1,800 → ¥3,800+** (+111%) | early 2026 → 2026-09-23 | smzdm, <https://post.smzdm.com/p/arzq5px7/> | press |
| Tyan single-socket 7003 board, JD distributor | ¥3,500 | 2026-09-23 | same article | press |
| EPYC 7452 | ¥600 (~$90 at 6.668) | 2026-09-23 | same | press |
| EPYC 7452 via Superbuy | $107.72 | 2026-11-06 | your cart | retail-live |
| Tyan S8030, Superbuy | **$507.34** | 2026-11-05 | prior report's live read | retail-live |
| Tyan S8030, Superbuy | **$536.28** | 2026-11-06 | your cart | retail-live |
| ASUS KRPA-U16, Superbuy | $457.94 | 2026-11-05 | prior report | retail-live |
| 32 GB DDR4-2666 RECC, China | **¥400 → ¥800 in one month** (+100%) | Aug → Sep 2026 | smzdm, 2026-09-23 | press |
| 32 GB DDR4-3200 RECC, China | ~¥1,200 ("even mediocre sticks") | 2026-08-23 | smzdm | press |

**Are SP3 used parts falling?** **No — they are rising, and for the opposite reason to the usual one.** The normal story is "datacenter refresh to Genoa/Turin dumps Rome/Milan onto the used market, prices fall." That is happening to *CPUs* (a 32-core 7452 is $108, i.e. 5% of its $2,025 launch price) but it is **not** happening to *boards*, because the board is the gateway to cheap DDR4. When DDR4 is the cheap memory tier and DDR5 RDIMMs are $2,165 for 32 GB on the module-spot tape, demand for DDR4 platforms rises. A clone board doubling in six months while the CPU in it sells for $108 is the signature of **platform scarcity, not CPU scarcity**.

**DDR4 RDIMM vs DDR5 RDIMM, today:**

| | DDR4-2133 16 GB (used, pulled) | DDR4-3200 32 GB (used) | DDR5-4800 32 GB RDIMM (module spot) |
|---|---|---|---|
| Price | $70 low ask / $102 median | $261 (DIMMsum, 2026-11-05) | $2,165 (TrendForce, 2026-10-26) |
| $/GB | $4.38 – $6.38 | $8.17 | $67.65 ⚠ thin |
| Trend | +27% in 3 months (asks) | up (China +100% in a month, Aug→Sep) | flat on the week, at a record |

**Read:** DDR4 RDIMM is expensive *in absolute terms for its age* and still the cheapest server memory on earth by a factor of ~10 per gigabyte against DDR5 spot. That is precisely why the platforms that take it are appreciating.

**A note on the board choice:** MiTAC's own product page confirms the Tyan **S8030GM2NE is ATX 12"×9.8"** (305×249 mm), 8 DIMM = 8 channels, 5× PCIe 4.0 x16, AST2500 BMC, EPYC 7002/7003 — <https://www.mitaccomputing.com/products/detail/S8030GM2NE>. So it fits your ATX chassis; the E-ATX worry from the earlier report is resolved. At +$28.94 versus yesterday's read it is also *not* obviously a price move — it is a different listing. Do not read one day of Superbuy variance as inflation. The six-month signal (clone board ×2) is the real one.

---

## 6. Buy now or wait — the matrix

Your build, item prices, 2026-11-06: board $536.28 + CPU $107.72 + RAM $247.84 + cooler ~$50 = **$941.84**.

Projections below are **SPECULATION** — my inference from the dated forecasts in §1–§2, not a published number. Probabilities are my judgement and are stated so you can disagree with them.

| Component | Today | (a) +3 months, base | (b) +6 months, base | (c) +12 months, base | Downside case (12 mo, RJ/Raymond-James path) |
|---|---|---|---|---|---|
| **4×16 GB DDR4-2133** | $247.84 | **+15%** → $285 | **+20%** → $298 | **+10%** → $273 | **−30%** → $173 |
| **Tyan S8030** | $536.28 | **+5%** → $563 | **+10%** → $590 | **flat** → $536 | **−15%** → $456 |
| **EPYC 7452** | $107.72 | +3% → $111 | +5% → $113 | **−10%** → $97 | **−25%** → $81 |
| Cooler | $50 | flat | flat | flat | flat |
| **Total** | **$941.84** | **$1,009** (+$67, +7.1%) | **$1,051** (+$109, +11.6%) | **$956** (+$14, +1.5%) | **$760** (−$182, −19.3%) |

Why those numbers:
- **RAM base case:** Q4 +10–15% (TrendForce 2026-09-30), then Q1–Q2-2027 decelerating but still positive (no published negative print; sufficiency gap widens in 2027 per 2026-07-30). The pulled-module channel adds the LTB effect: after March 2027 there is no new DDR4 at all.
- **Board base case:** supply of SP3 boards is fixed (all designs EOL); demand is a function of DDR4 being the cheap tier. +5–10%, then flat as the DDR4 premium peaks.
- **CPU base case:** Rome is abundant and still cascading out of datacenters; +3–5% near term (it tracks platform demand), −10% at 12 months as Genoa/Turin refresh continues.
- **Downside case:** Raymond James' mid-2026 peak + two quarters of decline from early 2027. Even there, the trough is late 2027 at the earliest, and the DDR4-2133 pulled segment falls least because its supply is already capped.

### 6.1 The arithmetic of waiting

| | Number |
|---|---|
| Expected extra cost of waiting 6 months, base case | **+$109** (+11.6%) |
| Best realistic saving from waiting 12 months, downside case | **−$182** (−19.3%) |
| Probability-weighted (my weights: 60% base-higher, 15% flat, 25% lower over 6 months) | **+$39 in favour of waiting** — 0.60×109 + 0.15×0 + 0.25×(−107). A coin flip worth 4% of the purchase |
| **Maximum** you can lose by buying now | ~$182 (19%) and only if the bear case is right *and* you would otherwise have waited a full year |
| **Maximum** you can lose by waiting | ~$250 (27%) if the LTB squeeze hits the pulled market |
| Cost of waiting, in project terms | **90 / 180 / 365 days** without 128 GB and 8 channels |

That last row is the whole argument. **This is a $942 decision with a ±$200 timing component.** The 128 GB / 8-channel capability is what lets a ~60 GB model plus KV cache live in host RAM at all (the earlier platform report's number); your current box has 62 GB. If that capability gates work you are doing this month, then 90 days of delay costs more than the entire timing spread.

### 6.2 Recommendation, with the hedge built in

**1. Buy the 4×16 GB DDR4-2133 immediately — today, unconditionally. $247.84.**
- It is the only line item with a documented +27%-in-3-months ask trend and a closing supply window (DDR4 LTB Oct 2026 – Mar 2027).
- It is **usable the day it arrives** in the four free slots of the X10SRL-F (8×16 GB = 128 GB at 2133, a supported config), so it is not a bet on the platform decision.
- It is portable: the same sticks are what the EPYC build needs.
- At $3.87/GB you are 12% under the cheapest eBay used stick and 39% under the median ask. The chip-over-module inversion (§3b) says lagging module quotes close upward.
- Risk if wrong: $248 of DDR4 that fell 30% = −$74. That is the cheapest insurance in the build.

**2. Buy the board + CPU now as well — $643.99 — unless the second GPU is >6 months away.**
- Base case says +$41 to +$67 over six months; the bear case says −$189 to −$159, but only after 12+ months and only if Raymond James is right about a 2027 decline that SK hynix's CEO explicitly contradicts.
- The board is the appreciating half (¥1,800→¥3,800 clone in six months). The CPU is the depreciating half (−10%/yr). Net: **+5–10% per six months**, i.e. ~$30–55. Small.
- **The real reason to buy now is not price — it is that the platform is the +57% (1.03→1.62) second-GPU speedup and the 2× memory bandwidth.** If that is on your 6-month roadmap, waiting costs more in lost throughput than the price spread is worth.

**3. If the platform is speculative** (no second GPU within 6 months, and 128 GB in the X10SRL-F covers you): **buy only the RAM now, hold the $644.** You give up 8 channels and gen4 x16; you keep the option. That is the honest "wait" — and note it is a *partial* wait, because the memory is the part that is actually inflating.

**4. Do not switch to the EPYC 7413 without a reason.** At $136 vs $107.72 it is +$28 for +9% boost clock (3.65 vs 3.35 GHz) and −8 cores. If your loop is launch-bound at 2% MFU, boost matters more than cores and the 7413 is defensible. **SPECULATION:** no benchmark of your workload on a 7413 exists in my sources; the clock argument is arithmetic, not measured. Either CPU is a rounding error against the board.

**5. Do not wait for the RTX 50 Super, and do not plan a second GPU purchase around a price fall.** The refresh is cancelled; GDDR7 modules are 3× the price; Nvidia raised partner kit prices three times in 2026; the 5070 Ti is +92% over MSRP today. A second card is an upward-risk purchase for the next 12 months.

---

## 7. The alternative use of the same ~$950

**8×H100 at today's median rate.** GPU Economy's Cloud GPU Price Index is the best-documented public series: a median of listed on-demand rates with per-week receipts, CC BY 4.0 — <https://gpueconomy.com/price-index>, read 2026-11-05 18:40 UTC.

| Quantity | Value |
|---|---|
| H100 SXM median, week of 2 Nov 2026 (provisional) | **$3.39 / GPU-hour** (38 quotes, 36 providers; p25–p75 $2.99–$4.41) |
| Latest hourly print | $3.29 |
| October 2026 monthly average (final) | $4.197 |
| September 2026 monthly average | $3.883 |
| Cheapest listed rate | **$1.79** (Cudo Compute) |
| Cheapest 1-year commitment | $1.81 (Seeweb) |
| Hyperscaler median (AWS/Azure/GCP/OCI) | $9.90 |
| **8×H100, median** | **$27.12 / hour** |
| **$941.84 buys, at median** | **34.7 hours of 8×H100** (= 278 H100-GPU-hours) |
| **$941.84 buys, at the cheapest rate** | **65.8 hours** (= 526 GPU-hours) |
| H100 rate trend | **−20% since Jan 2025** (base-100 = 80.2); −10% Sept→Nov 2026 ($4.20→$3.78 monthly avg) |

**Honest comparison:**

- 35 hours of 8×H100 is **one training campaign**, not a research loop. For a 200 M-parameter model that is plausibly enough for a full run with room for a couple of restarts; for a 60 GB-model agentic workflow it is 35 hours of a machine you cannot keep state on.
- **Waiting does not improve this trade.** If H100 rates keep falling at the 2026 pace (−10% per six months), in six months $942 buys **~38.5 hours instead of 34.7** — **+3.8 hours**, about $103 of compute. That is the entire gain from waiting on the rental side, against +$109 of hardware inflation. The two roughly cancel. There is no arbitrage here.
- **The rental option is not a substitute for the purchase** because they buy different things: rental buys FLOPs (which you already have in a 5070 Ti), the purchase buys **128 GB of host RAM, 8 memory channels, and a gen4 x16 slot** — capacities you cannot rent by the hour and cannot bolt onto an X10SRL-LGA2011 box.
- If you want a genuine third option: **$248 of RAM now + $694 of H100 time** (25.6 h of 8×H100 at median) is a coherent "prove the workload before buying the platform" hedge. It is the option I would take if the second GPU were not already planned.

---

## 8. Verdict

**BUY NOW.** All four lines if the second GPU is inside 6 months; the RAM alone if it is not.

The reason is not that prices are certain to rise — it is that **nothing is scheduled to make them fall, and one thing is scheduled to make DDR4 structurally scarcer: the last-time-buy windows close between October 2026 and March 2027.** You would be waiting for a supply response that TrendForce dates to 2H-2027 at the earliest and 2028 for volume, against a demand side where no hyperscaler has cut capex. The single credible bear call (Raymond James) points at a 2027 decline and describes it as a slowdown, not a collapse.

And the stakes are small: ±$200 on a $942 purchase. Spending three months to optimise that — while a 60 GB model waits for RAM you do not have — is the expensive mistake, not the $62 sticks.

### VERIFIED (sourced)

| Claim | Source | Date |
|---|---|---|
| DDR4 16Gb spot $81.679, **1.35× DDR5 16Gb at $60.667** | TrendForce spot table | 2026-11-05 |
| Contract DDR4 8Gb $26.00, DDR4 16Gb $52.00, +4.00% | TrendForce contract table (2H Sep) | 2026-09-30 |
| Q4-2026 conventional DRAM contract **+10–15% QoQ** | TrendForce, reported by exIT + computingforgeeks | 2026-09-30 |
| Competing Q4 reads of **+0–5%** and **+3–8%** | Luminix / Goldman reading; TrendForce July survey via sourcebyspec | 2026-09/10 |
| Q1 +90–95%, Q2 +58–63%, Q3 +13–18% | TrendForce via Luminix, exIT | 2026 |
| 2027 sufficiency −1 to −2% in 2026, **gap widens in 2027**; fabs ramp 2H27, volume 2028 | TrendForce press release | 2026-07-30 |
| Samsung DDR4 LTB **Oct 2026 – Mar 2027**; Micron LTS passed Q1-2026; Samsung/SK hynix stopped new orders | limchip; electronics-sourcing | 2026-07-07 |
| Supplier inventory **<10 days** | KB Securities via DigiTimes (analyst estimate) | 2026-09-07 |
| **Chip-over-module inversion closes upward in 2–4 weeks** | TrendForce via SpecForge | 2026-10-05 |
| 16 GB 2133 RDIMM used: $4.38/GB low ask, $6.38 median ask, **+27% since 8 Aug**; sold $2.86/GB July | DIMMsum | 2026-11-05 |
| RTX 5070 Ti new **$1,439.99** (MSRP $749); 5090 $3,340; November +$150/+$500 | videocardprices (Newegg); tech-insider; wccftech | 2026-11-05 / 11-03 |
| RTX 50 Super **cancelled** (Gamescom 2026); GDDR7 3 GB module $60–70 vs $20 | gamegpu; Digital Foundry | 2026-08 / 2026-07 |
| Nvidia partner kit prices +32–49% in 2026 (3 hikes) | capitalandcompute | 2026-09 |
| TSMC N5/N4/N3 **+5–10%** in 2026; CoWoS +20% under discussion | DigiTimes via TechSpot; TrendForce via Tom's | 2026-10-27 |
| SP3 clone board **¥1,800 → ¥3,800+**; 32 GB RECC **¥400→¥800 in a month** | smzdm | 2026-09-23 |
| Tyan S8030GM2NE is **ATX 12"×9.8"**, 8 DIMM/8 ch, 5× gen4 x16, AST2500 | MiTAC product page | fetched 2026-11-05 |
| H100 median **$3.39/GPU-h**; 8× = $27.12/h; $942 = **34.7 h**; cheapest $1.79; −20% since Jan 2025 | GPU Economy Price Index | 2026-11-05 |

### SPECULATION (my inference — no source asserts these)

| Inference | Basis | Confidence |
|---|---|---|
| +3/+6/+12-month price paths in §6 | TrendForce trajectory extrapolated; no published point forecast for these items | Low–Medium |
| 60/15/25 probability weights | My judgement on the cited forecast dispersion | Low |
| DDR4-2133 pulled asks continue at ~+8%/month | 3-month slope (Aug→Nov) extrapolated | **Low** — 3 points is not a trend |
| DDR3 rose "X% after EOL" | **I could not verify this.** Only the structural claim (EOL → premium plateau) is supported | **Unverified — do not cite** |
| 7413's +9% boost beats 7452's +8 cores for your loop | Arithmetic only; no benchmark on your workload | Low |
| Board +5–10% over 6 months | smzdm's six-month board move + fixed supply; no board-specific forecast exists | Low–Medium |

---

## 9. Open questions

1. **What does the Q4 contract print actually settle at — +10–15% or +0–5%?** TrendForce's October report is behind a paywall (`RP261030WD`, 2026-10-30); PC DRAM negotiations complete in November. This is the one number that would re-rate the whole forecast, and it lands *after* your 48-hour decision window. Do not wait for it.
2. **Are the $61.96 sticks new or pulled-and-retested?** The listing says new. At 13% of die cost that is implausible for new production; if they are "new old stock" pulled from a decommissioned fleet, the price is normal and the supply is finite. Ask the agent for the module date code and rank (2Rx4 vs 2Rx8) before ordering 4 — mixing ranks on EPYC is a debugging session.
3. **Does the Tyan S8030 POST a 5070 Ti?** No public report either way, per the earlier report. The board is genuine with a real BMC and QVL, which is the best available mitigation, but it is unverified.
4. **Is the second GPU actually inside 6 months?** This single answer flips §6.2 between "buy all four lines" and "buy the RAM only". It is the only question in the report that you can answer better than the sources can.

---

## Sources

**Primary price/forecast publishers**
1. TrendForce DRAM spot + contract tables — <https://www.trendforce.com/price/dram/dram_contract> (fetched 2026-11-05)
2. TrendForce, "Diverging Memory Market Outlook in 2027…" — <https://www.trendforce.com/presscenter/news/20260730-13158.html> (2026-07-30)
3. TrendForce research abstracts: `RP261104XL` (2026-11-04), `RP261030WD`, `RP261030CN`, `RP261028TO` (2026-10-28)
4. TrendForce, DDR4 exit timeline — <https://www.trendforce.com/news/2025/07/09/news-ddr4-exit-timeline-unfolds-for-top-memory-makers-plans-for-samsung-sk-hynix-and-micron/> (2025-07-09)
5. TrendForce, DDR4 phase-out delayed to 2026 — <https://www.trendforce.com/news/2025/09/02/news-samsung-sk-hynix-reportedly-delay-phase-out-to-2026-as-ddr4-becomes-unexpected-cash-cow/> (2025-09-02)
6. MiTAC/Tyan S8030GM2NE product page — <https://www.mitaccomputing.com/products/detail/S8030GM2NE> (fetched 2026-11-05)

**Trackers (retail-derived, dated)**
7. DIMMsum, 16 GB DDR4-2133 RDIMM — <https://dimmsum.com/ddr4/16gb-2133-rdimm> (2026-11-05)
8. GPU Economy Cloud GPU Price Index — <https://gpueconomy.com/price-index> (2026-11-05 18:40 UTC)
9. videocardprices.com, RTX 5070 Ti — <https://videocardprices.com/card/nvidia-rtx-5070-ti/> (2026-11-05 14:41 UTC)
10. rampricesusa, November 2026 index — <https://rampricesusa.com/insights>

**Trade press / analysis**
11. exIT Technologies, "Why Server Memory Prices Are Still Climbing in Q4 2026" — 2026-10-26
12. Luminix, "DRAM Cycle Position Analysis — Peak Timing Indicators" — updated 2026-09-30
13. SpecForge, "DRAM Spot vs Contract Gap, October 2026" — 2026-10-05
14. electronics-sourcing.com, "End-of-life, start of trouble" — 2026-07-07
15. limchip, "Samsung DDR4 EOL impact 2026" — 2026
16. wccftech / Dataconomy / techbriefly, RTX 50 November price increase — 2026-11-03
17. gamegpu, "Gamescom 2026 confirmed RTX 50 Super cancellation" — 2026-08
18. Digital Foundry, "RTX 50 Super delayed due to 3 GB module pricing" — 2026-07
19. Tom's Hardware via tech-insider, DDR5 32 GB kit $479.99 — 2026-11-02
20. TechSpot/DigiTimes, TSMC 2026 node hikes 5–10% — 2026
21. Tom's Hardware, TSMC CoWoS +20% under discussion — 2026-10-27
22. smzdm, SP3 build-cost article — <https://post.smzdm.com/p/arzq5px7/> (2026-09-23)
23. smzdm, DDR4-3200 RECC ¥1,200 — <https://post.smzdm.com/p/a4qd74rk/> (2026-08-23)
24. Windows Central via Yahoo Tech, DDR3 prices — 2026-06-17

**Prior work in this repo**
25. `docs/research/2026-11-05-epyc-platform-report.md` — item prices, the 8-channel bandwidth arithmetic, the 1.03→1.62 second-GPU speedup, the 60 GB-model RAM requirement
