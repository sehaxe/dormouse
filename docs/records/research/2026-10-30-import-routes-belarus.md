# Importing a used RTX 3090 (24 GB, ~2 kg, 27×12×3 cm) to Minsk — route memo

**Date:** 30 October 2026 · **Destination:** Minsk, Belarus · **Declared value modelled:** $1,200 (range $600–1,700)
**FX used everywhere:** NBRB official rates 30.10.2026 — USD 2.9462 BYN, EUR 3.2801 BYN ⇒ **$1 = €0.8982**, €1 = $1.1134.
**Scope:** read-only desk research; no orders placed, nothing installed.

---

## 0. TL;DR

1. **The local market sets the bar.** Used RTX 3090 asking prices in Minsk on 30.10.2026 are **4,400–5,000 BYN = $1,493–1,697** (Kufar). A $1,200 GPU only makes sense if total landed cost stays below ~$1,450.
2. **The cheapest lawful personal route is a friend's air luggage:** Belarus allows **€10,000 / 50 kg duty-free in accompanied or unaccompanied baggage by air**. A single used GPU at $1,200 is ~€1,078 → **€0 duty**. This beats every parcel route.
3. **Every parcel route pays 15% of the amount over €200** (min €2/kg of weight excess): **≈ $147 on a $1,200 GPU**. The "5% + VAT" e-commerce reform that was supposed to start 1 July 2026 was **postponed indefinitely** (EEC, reported 30.09.2026) — do not use the 5% formula.
4. **"Duty-free карго" cannot be honest at $2.5–3.5/kg.** 2.5 kg × $3.5 = $8.75 of freight, while the real duty is $147. A per-kg rate that "includes customs" on a $1,200 item means the value is under-declared. The liability sits with the Belarusian importer of record, not the Chinese warehouse.
5. **The USA is effectively closed** — not because of Belarus customs, but because of US export law: USPS suspended Belarus on **22.07.2022** and still lists it; Shipito does not serve Belarus; Qwintry suspended it. Separately, HTS **8471.50** ("digital processing units") is on the BIS **"luxury goods"** list (Supp. 5 to Part 746) requiring a **BIS licence + ITN** for Belarus. Whether a 3090 is 8471.50 or 8473.30 decides legality — see §3 and §8.
6. **EU → Minsk is logistically open but legally exposed**: Regulation 765/2006 bans exports of machinery (Annex XIV) and luxury goods (Annex XXV) to Belarus. Bus/courier parcels exist (Vilnius–Minsk from €120), but seizure at the EU border is a real risk.

**Pick:** a traveller carries it by air (§7). Fallback: China air-express cargo, honestly declared, ~$1,390 landed. Do not route via the US.

---

## 1. Customs law — verified

### 1.1 The norms in force on 30.10.2026

Source: **ГТК РБ (Belarus State Customs Committee), "Ввоз товаров для личного пользования с уплатой таможенных платежей"**, accessed 30.10.2026 — <https://www.customs.gov.by/fizicheskim-litsam/vvoz-tovarov-dlya-lichnogo-polzovaniya-s-uplatoy-tamozhennykh-platezhey/>

| How the goods arrive | Duty-free norm | Rate above the norm |
|---|---|---|
| **Air, accompanied or unaccompanied baggage** | **€10,000 and 50 kg** | 30% of the value excess, **min €4/kg** of the weight excess |
| **Land/rail/on foot, accompanied or unaccompanied baggage** | **€500 and 25 kg** (since 01.04.2024) | 30% of the value excess, **min €4/kg** |
| **Delivered by a carrier** (courier, DPD, etc.) | **€200 and 31 kg** | 15% of the value excess, **min €2/kg** |
| **International postal item (МПО)** | **€200 and 31 kg gross** | 15% of the value excess, **min €2/kg** |

The €200 / 31 kg figure is per **one postal item**, not a monthly aggregate — the ГТК table says "пересылаемые в международных почтовых отправлениях" with no monthly wording. (Some forwarders' FAQ pages still say "per calendar month"; that is a legacy of the 2017–2019 rules. Treat "per parcel" as the operative norm and confirm with customs for a second parcel in the same month.)

### 1.2 Formula (use this)

For a parcel/courier item of value `V` (EUR) and gross weight `W` (kg):

```
duty = max( 0.15 × max(0, V − 200),  2 × max(0, W − 31) )
```

For air luggage (the good case):

```
duty = 0 if V ≤ 10,000 and W ≤ 50
```

For land luggage / bus hand-carry:

```
duty = max( 0.30 × max(0, V − 500),  4 × max(0, W − 25) )
```

Worked for the GPU at $1,200 = **€1,078**:

| Route type | Duty |
|---|---|
| Postal / courier | 0.15 × (1,078 − 200) = **€131.7 ≈ $146.6** |
| Air luggage | **€0** (under €10,000) |
| Land luggage (bus) | 0.30 × (1,078 − 500) = **€173.4 ≈ $193.1** |

Sensitivity: at $600 → **$56.6** postal; at $1,700 → **$221.6** postal.

### 1.3 The 2026 reform that did NOT happen

- 12.01.2026: EEC Council approved €200 threshold, **5% of the full value, min €1/kg, plus national VAT**, to apply from **1 July 2026** — Interfax, 12.01.2026, <https://www.interfax.ru/world/1067139>.
- **30.09.2026: ratification postponed indefinitely** — "перенесен срок введения порога… Перенос связан с подготовкой внутреннего законодательства и информационных систем" — Alta.ru, 30.09.2026, <https://www.alta.ru/ts_news/129191/>; also trans.ru, "отложили на неопределенный срок".
- **Therefore on 30.10.2026 the old 15% / min €2/kg rule applies.** Blog posts and law-firm notes dated June 2026 that describe 5% + 20% VAT are describing a regime that is not in force. Do not budget VAT.

### 1.4 Valuation, used goods, certification

- **Valuation.** Customs takes the declared value, supported by contract/invoice/pro-forma/sales receipt/payment receipt (Указ № 360 от 21.07.2014; ст. 267 ТК ЕАЭС). For postal items the **declared value of the item (объявленная ценность)** is used if higher than the documents. Customs may **determine value itself** from catalogues/retail sites if the declared figure is inconsistent with analogues — ГТК РБ, "Методические рекомендации по определению предназначения товаров", accessed 30.10.2026.
  - Practical consequence: a $400 declaration on a card that retails at $1,500+ is exactly the trigger for re-valuation. Keep the eBay/private-sale screenshot and the payment record.
- **Used status** does not exempt from duty and does not by itself bar personal-use treatment; wear signs actually help prove personal use ("наличие признаков износа" is listed among the personal-use criteria). A sealed, shrink-wrapped card looks commercial.
- **Certification.** Goods for personal use are outside the EAEU technical-regulation/licensing regime — no EAC certificate is required for one card for own use (Решение Совета ЕЭК № 107 от 20.12.2017; MID RF summary "меры нетарифного и технического регулирования (лицензирование) не применяются"). No radio module in a GPU, so no encryption notification.
- **No Belarusian import ban or quota on GPUs/computer parts** was found for personal use.

---

## 2. Export controls — the part that decides whether a route is legal at all

This is not customs. It is the law of the **sending** country, and it applies regardless of how the parcel is labelled.

### 2.1 USA

- BIS maintains "similarly stringent controls on items subject to the EAR that are destined for Belarus" — <https://www.bis.gov/licensing/country-guidance/russia-belarus>.
- **Supplement No. 5 to Part 746 ("Luxury Goods") includes HTS-6 `847150` — "DIGITAL PROCESSING UNITS OTHER THAN THOSE OF 8471.41 AND 8471.49" — with no price threshold.** Export/reexport/transfer to Belarus of a listed item requires a BIS licence — eCFR, <https://www.ecfr.gov/current/title-15/subtitle-B/chapter-VII/subchapter-C/part-746/appendix-Supplement%20No.%205%20to%20Part%20746>.
- USPS's own country page for Belarus repeats it: "'luxury goods' destined to Belarus require a license and an Internal Transaction Number (ITN)… No exemptions are allowed. The 10-digit Schedule B number must appear…" — USPS IMM, accessed 30.10.2026, <https://pe.usps.com/text/imm/ab_019.htm>.
- **Open question that flips the answer:** discrete graphics cards are classified by CBP sometimes under **8471.50** (processing unit) and sometimes under **8473.30** (parts of computers). 8473.30 is **not** in the Supp. 5 list above; 8471.50 **is**. I could not verify the binding classification of an RTX 3090. **This must be settled by a licensed customs broker / BIS classification before any US-origin shipment.**
- Re-export matters: a US-origin card shipped from China to Belarus is still "subject to the EAR". §746.8 applies to reexports, not only to exports from the US. This is the legal shadow over route 1.

### 2.2 EU

- Regulation (EC) 765/2006 bans sale/supply/transfer/export to Belarus of **machinery and electrical equipment (Annex XIV)**, **goods enhancing industrial capacity (Annex XIX)** and **luxury goods (Annex XXV)** — EU Sanctions Map, Belarus regime, <https://www.sanctionsmap.eu/>.
- Whether HS 8471/8473 cards are inside Annex XIV/XXV is a classification question I could not resolve from the public summaries. A personal hand-carry is an export; the "personal effects" exemption is narrow.

**Bottom line:** route legality is dominated by export control, not by the Belarusian duty. A memo that only computes customs is answering the wrong question.

---

## 3. Route comparison

Prices are for **2–3 kg** (the card's actual weight; volumetric weight 27×12×3/5000 ≈ 0.2 kg, so actual weight rules). "Duty" assumes a $1,200 declared value.

| # | Route | Named provider (2026) | Price 2–3 kg | Transit | Duty | Total landed ($1,200 GPU) | Risk |
|---|---|---|---|---|---|---|---|
| 1a | **China → Minsk, auto cargo (grey)** | GoTranzit (Guangzhou–Minsk), China4Biz, Cargo24Express | $2.5–3.5/kg **from 200 kg**; small-parcel minimum realistically **$20–40** (ESTIMATE) | 25–30 days | $0 as sold ("duty-free"), i.e. under-declared | **$1,220–1,240** | No per-item insurance; under-declaration liability on importer of record; seizure if inspected; US-origin reexport issue |
| 1b | **China → Minsk, air express cargo** | LTL.by (Guangzhou–Minsk $14/kg for 1–15 kg); DobroPost (759 RUB/kg ≈ 33.1 BYN ≈ $11.2/kg, 22–29 d; 949 RUB ≈ $11.9/kg, 11–15 d) | **$28–45** (2–3 kg, before min fee) | 3–5 d direct air (PEK) / 11–29 d via RF | $147 if declared honestly | **$1,375–1,390** | Same, but faster and traceable per consignment |
| 2 | **China → Minsk via agent (Superbuy)** | Superbuy | **Not verified** for Belarus; agent lines to CIS typically $12–20/kg (ESTIMATE) | 15–35 d (ESTIMATE) | Depends on line (often DDP) | **$1,250–1,400 (ESTIMATE)** | Line availability must be confirmed in the calculator; no published BY line found |
| 3 | **China → Minsk via post/EMS/DPD** | Belpochta/EMS (designated operator), DPD Belarus | EMS from China ~$30–60 for 2 kg (ESTIMATE, China Post tariff); DPD quote-only | 15–40 d | **$147** | **$1,380–1,410** | Slow; customs broker fee; Belpochta has suspended some directions (US parcels from 22.10.2026) |
| 4 | **USA → Minsk forwarders** | **CDEK USA** $19.50/kg, **min billed 5 kg** = $97.50; Global Shopaholics (serves BY, quote-only); Fulfilla from $16.95 | **$97.50** (CDEK, 5 kg minimum) | 2–4 weeks (ESTIMATE) | **$147** | **$1,445 + insurance** | US export licence question (§2.1); CDEK base insurance $85 covers total loss only; +5% of declared value for extra |
| — | USA — **not available** | **USPS** — Belarus suspended since **22.07.2022**, still listed; **Shipito** — Belarus absent from its shipping-methods country list; **Qwintry** — "does not deliver parcels to Belarus"; FedEx/UPS/DHL — suspended (Princeton ORPA) | n/a | n/a | n/a | n/a | Closed |
| 5 | **eBay GSP / Amazon** | eBay replaced GSP with **eBay International Shipping (Aug 2026)**; Belarus is on seller exclusion lists and ISP-ineligible. AmazonGlobal's country list includes Belarus, but export-control rules and seller restrictions apply | n/a | n/a | n/a | n/a | Not a reliable route |
| 6a | **EU → Minsk, courier** | LTL.by Vilnius–Minsk **€120** (per shipment); Pony Express "Прямая машина" from 0.5 kg | **€120 ≈ $134** | 1–3 days | **$147** (carrier rule) | **$1,481** | EU export ban (Annex XIV/XXV); border queues |
| 6b | **EU → Minsk, bus hand-carry** | Vilnius–Minsk coach (tickets from 150 BYN); informal driver parcels ~€30–60 (ESTIMATE) | **€30–60 ≈ $33–67** | 1 day | **$193** (land luggage, 30% over €500) | **$1,426–1,460** | Same EU export ban; land-luggage valuation at the border |
| 7 | **Personal luggage, air** | Friend flies in (Beijing/Guangzhou/Istanbul–Minsk) | **$0 marginal** if already travelling | same day | **$0** (€1,078 < €10,000) | **$1,200** | Traveller dependency; must look used/personal; export-control at departure |

Notes on the "duty-free" arithmetic: 2.5 kg at $3.5/kg is **$8.75** of freight. The statutory duty on a $1,200 card is **$146.6**. No forwarder absorbs a $147 duty inside an $8.75 rate. Either the value is declared near €200 (under-declaration), or the rate excludes duty.

---

## 4. Route notes (what each source actually says)

**China auto cargo.** GoTranzit: "от 2.5$/кг (от 200 кг)", Guangzhou/Yiwu–Minsk, "от 25 дней", "с учетом всех пошлин" — <https://gotranzit.ru/strani/aziya/kitaj/kitaj-kargo/kargo-dostavka-iz-kitaya-v-belarus/>, accessed 30.10.2026. China4Biz: auto $1.5/kg, rail $1/kg, 25–30 days — <https://china4biz.by/cargo>. These are consolidation tariffs; a 2.5 kg parcel is a minimum-charge shipment, not a per-kg shipment.

**China air express.** LTL.by publishes Guangzhou–Minsk **$14/kg for 1–15 kg**, Beijing $15, direct air 3–5 days, customs and door delivery included — <https://ltl.by/express/china>, accessed 30.10.2026. DobroPost: 759 RUB/kg (≈33.12 BYN) 22–29 days; 949 RUB/kg (≈35 BYN) 11–15 days — <https://by.dobropost.com/>. This is the best documented China option for a single card.

**Superbuy.** No Belarus-specific line or price is published; the freight calculator is JS-only and requires login. **I could not verify that Superbuy ships a GPU to Belarus in 2026.** Treat route 2 as unconfirmed.

**Belpost/EMS/DPD.** Belpochta is the designated postal operator and EMS is its express product; DPD Belarus confirms international express to/from China and the USA for private individuals — <https://www.dpd.by/fizicheskim-litsam>. Rates are quote-only. Inbound EMS is priced by the origin operator (China Post), not by Belpochta, so no single published number exists.

**USA.** USPS International Service Alerts still lists Belarus under "Suspension Due to Unavailability of Transportation", suspension date **7-22-22** — <https://about.usps.com/newsroom/service-alerts/international/>. Qwintry's BY page: "The delivery of parcels to Belarus is temporarily suspended… due to changes in government policies and regulations" — <https://qwintry.com/en/by>. Shipito's shipping-methods country list includes Russia, Ukraine, Kazakhstan but **not Belarus** — <https://www.shipito.com/en/help/postage>. CDEK USA: "$19.50 за кг… посылки свыше 2 кг (расчёт от 5 кг)", base insurance $85, extra 5% of declared value, DDP option — <https://cdekusa.com/usa-to-minsk-shipping/>. Global Shopaholics has a dedicated Belarus service page but no published rates — <https://globalshopaholics.com/ru-by/>.

**eBay / Amazon.** eBay replaced the Global Shipping Program with eBay International Shipping for US sellers in August 2026; Belarus appears on seller exclusion lists and is described as ISP-ineligible. AmazonGlobal's export-country list does include Belarus — <https://www.amazon.com/gp/help/customer/display.html?nodeId=GCBBSZMUXA6U2P8R> — but that list is about AmazonGlobal eligibility, not about export-licence clearance for a controlled commodity.

**EU.** LTL.by: Vilnius–Minsk €120, Druskininkai–Minsk €120 — <https://ltl.by/express/lithuania>. Pony Express runs a "Прямая машина" Minsk–Vilnius from 0.5 kg — <https://ponyexpress.by/delivery/dostavka-po-belarusi-i-sng/pryamaya-mashina/>. Coach Vilnius–Minsk tickets from 150 BYN. All of these move goods across an EU external border where Annex XIV/XXV export bans apply.

---

## 5. The local benchmark (why this may be a bad idea at $1,700)

Kufar, Minsk, 30.10.2026 (asking prices, not sold prices):

| Listing | Price | USD |
|---|---|---|
| Gigabyte RTX 3090 Gaming OC 24G, used | 4,400 BYN | **$1,493** |
| RTX 3090 24 GB, used | 4,800 BYN | **$1,629** |
| Gigabyte RTX 3090 Gaming OC, used | 5,000 BYN | **$1,697** |
| AFOX RTX 3090 24GB, new (Onliner catalogue) | from 8,589 BYN | $2,915 |

So: at a **$600–900** purchase price, importing is clearly worth it. At **$1,200** it saves ~$100–250 versus local — roughly one courier fee, i.e. the entire margin can be eaten by one re-valuation or one failed delivery. At **$1,700** it is cheaper to buy in Minsk.

---

## 6. Insurance and tracking reality

| Route | Insurance | Tracking |
|---|---|---|
| China auto cargo | Usually none per item; consolidated cargo may have a nominal per-kg limit | Bag/consignment level, not per item |
| China air express (LTL/DobroPost) | Declared-value insurance available on request (confirm %) | Yes, waybill |
| Belpost EMS | EMS indemnity per UPU rules; declared value fee | Yes |
| DPD | Declared value service | Yes |
| CDEK USA | $85 base, total loss only; +5% of declared value for more | Yes |
| EU courier | Standard carrier liability (often €10–20/kg) | Yes |
| Luggage | None (airline liability only) | n/a |

For a $1,200 item, the CDEK extra premium alone is **$60** — 5% of value — which is roughly what a month of the price advantage is worth.

---

## 7. Recommendation

**First choice — air luggage (route 7).** If any trusted person flies into Minsk, this is the only route that is both cheap and duty-free: €10,000/50 kg allowance, no declaration needed at €1,078, no broker, no volumetric charge. Conditions: the card must travel as a used personal item (no sealed retail box, ideally in a build or with a receipt showing a private sale), and the traveller should be able to explain it. Duty saving vs. postal: **$147**. This is also the route with the least paperwork risk on the Belarusian side.

**Second choice — China air express, honestly declared (route 1b).** ~$30–45 freight + $147 duty ≈ **$1,380 landed**, 1–3 weeks. Use LTL.by or DobroPost; declare $1,200 and keep the purchase screenshot. You pay $147 but you own a clean, insurable, traceable shipment. If the friend's карго can put the card on a **separately declared** waybill at real value inside the consolidation, that is the same thing cheaper — ask him explicitly whether he will declare $1,200, and get that in writing.

**Do not use the "duty-free" grey consolidation for this item.** At $8.75 of freight against $147 of duty, the only way it is "duty-free" is an under-declaration, and the person liable is the Minsk recipient, not the Guangzhou warehouse. For a $1,200 one-off the saving (~$147) does not price the tail risk (detention, administrative case, no insurance).

**Do not route via the USA.** USPS/FedEx/UPS/DHL are closed to Belarus, Shipito and Qwintry have dropped it, and the BIS luxury-goods list plus the 8471.50-vs-8473.30 classification question make a US-origin shipment legally uncertain even through CDEK. If the card is currently in the US: sell it there, buy in China or Minsk.

**Do not route via the EU** unless the card is already inside Belarus's customs union neighbours and you accept the Annex XIV/XXV seizure risk. If you do, carry it personally by bus (land allowance €500 → $193 duty) rather than couriering it (€120 freight + $147 duty + EU export ban).

---

## 8. What I could NOT verify

1. **Binding HTS/ECCN classification of an RTX 3090** (8471.50 vs 8473.30). This single fact decides whether a US-origin card needs a BIS licence to enter Belarus. Not verifiable from public sources; requires a broker's ruling or a BIS classification request.
2. **Whether Superbuy has a Belarus line in 2026.** The calculator is JS/login-gated; no public BY page found. Route 2 prices are an estimate, not a quote.
3. **Whether MyUS and Skypax currently serve Belarus.** MyUS's country-alerts page timed out repeatedly; Skypax publishes no country matrix I could read. Not used in the recommendation.
4. **Actual transaction prices on Kufar.** The $1,493–1,697 figures are asking prices; sold prices are not public.
5. **The exact duty-free treatment of a second parcel to the same recipient in the same month.** The ГТК table is per item; forwarder FAQs still mention a monthly aggregate. Confirm with customs if more than one parcel is planned.
6. **Whether a used, privately sold GPU is accepted at $1,200 without a customs valuation challenge.** The rule allows customs to use catalogue prices; practice at Minsk airport/postal exchange is anecdotal, not documented.
7. **EU Annex XIV/XXV HS coverage for GPUs.** The bans exist; the HS lines were not readable in the public summary.
8. **China Post EMS tariff for 2 kg to Belarus.** Quote-only; the $30–60 figure is an estimate.

---

## 9. Sources (all accessed 30.10.2026)

| # | Source | Date | Used for |
|---|---|---|---|
| 1 | ГТК РБ, "Ввоз товаров для личного пользования с уплатой таможенных платежей", customs.gov.by | live page, © 2026 | Norms table: €10k/50 kg air, €500/25 kg land, €200/31 kg carrier & post; 15%/min €2, 30%/min €4 |
| 2 | ГТК РБ, "Беспошлинный ввоз/вывоз товаров для личного пользования" | live page | Air allowance confirmation |
| 3 | ГТК РБ, "Методические рекомендации по определению предназначения товаров" | live page | Valuation, used-goods criteria, catalogue pricing |
| 4 | Решение Совета ЕЭК № 107 от 20.12.2017; Указ РБ № 360 от 21.07.2014 | law | Legal basis, personal-use definition, value documents |
| 5 | НБРБ API, EUR/USD official rates | 30.10.2026 | FX: USD 2.9462, EUR 3.2801 BYN |
| 6 | Interfax, "ЕЭК утвердила порог… 200 евро и пошлину 5%" | 12.01.2026 | The 5%+VAT reform as adopted |
| 7 | Alta.ru, "Ратификация поправок… перенесен срок" | 30.09.2026 | **Reform postponed — 15% still in force** |
| 8 | USPS IMM, Country Conditions — Belarus | live page | €200/31 kg, 15%/min €2, luxury-goods ITN requirement |
| 9 | USPS International Service Alerts | live page | Belarus suspension since 22.07.2022 |
| 10 | eCFR, Supplement No. 5 to Part 746 | current | HTS 847150 on the luxury-goods licence list |
| 11 | BIS, Russia and Belarus country guidance | live page | Belarus controls scope |
| 12 | EU Sanctions Map, Belarus | live page | Annex XIV machinery, Annex XXV luxury bans |
| 13 | Qwintry, /en/by | live page | "does not deliver parcels to Belarus" |
| 14 | Shipito, Shipping Methods country list | live page | Belarus absent |
| 15 | CDEK USA, "Доставка из США в Минск" | 2026 | $19.50/kg, 5 kg minimum, $85 insurance, +5% |
| 16 | Global Shopaholics, ru-by | 2026 | Belarus service exists, quote-only |
| 17 | LTL.by, "Экспресс доставка из Китая" | 2026 | Guangzhou–Minsk $14/kg (1–15 kg), 3–5 days |
| 18 | LTL.by, Lithuania express | 2026 | Vilnius–Minsk €120 |
| 19 | GoTranzit, China–Belarus cargo | 2026 | from $2.5/kg (200 kg), 25+ days |
| 20 | China4Biz, cargo | 2026 | $1.5/kg auto, 25–30 days |
| 21 | DobroPost BY | 2026 | 759/949 RUB per kg, 11–29 days |
| 22 | Kufar Minsk GPU listings | 30.10.2026 | Local asking prices 4,400–5,000 BYN |
| 23 | eBay seller community / eIS launch coverage | Aug 2026 | GSP replaced by eIS; Belarus excluded |
| 24 | AmazonGlobal export countries | live page | Belarus listed as AmazonGlobal destination |
| 25 | Belsat, "Belposhta halts parcel deliveries to the US" | 22.10.2026 | Postal instability context |

---

## 10. Verdict, split by evidence class

**VERIFIED (sourced):** the four duty-free norms and the 15%/min €2 formula; the postponement of the 5% e-commerce regime; the €0 duty for air luggage at €1,078; the $146.6 postal duty at $1,200; USPS suspension since 22.07.2022; Shipito/Qwintry not serving Belarus; HTS 8471.50 on the BIS luxury-goods list; EU Annex XIV/XXV bans exist; China cargo/air rates as published; local asking prices.

**SPECULATION / ESTIMATE (mine):** small-parcel minimum charges ($20–40 auto, $28–45 air); Superbuy line pricing; transit times marked ESTIMATE; the €30–60 informal bus-parcel price; the practical odds of a customs valuation challenge.

**Confidence:** customs arithmetic **high (0.9)** — primary source plus official FX. Provider pricing **medium (0.6)** — published but quote-based and volatile. Legal usability of each route **low–medium (0.5)** — hinges on the unverified HTS classification. Overall recommendation confidence **medium-high** for "air luggage first, China air express second, avoid US/EU".
