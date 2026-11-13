# S8030 / EPYC 7452 build parts — 2026-11-07

**Scope:** research only. Cart not changed. No GPU jobs run.
**Prices:** fetched 2026-11-07 unless the source date is shown.
**Tags:** `CT` = click-through page opened, price is for the shown single variant; `SC` = search-card/snippet price, not click-verified; `AG` = aggregator/index price, not a checkout; `OWN` = owner's cart, not opened by me; `NV` = not verified tonight.

---

## TL;DR (RU)

1. **Кулер:** бери **Dynatron A50** — 110 мм, SP3, 280 Вт, **$41.95**. Он влезает в любой 3U/4U и тянет и 155 Вт, и будущий 280 Вт. Noctua NH-U12S (158 мм) — только если паспорт корпуса разрешает ≥158 мм (при самостоятельном замере — ≥163 мм) и важна тишина; NH-U14S (165 мм) в 4U почти наверняка не влезет.
2. **Память:** **4×16 ГБ 2133 2Rx4 за $61.97 — хорошая цена**, ниже текущего рынка ($71.69–73 за планку). `8×16` — это **128 ГБ, а не 256 ГБ**; 256 ГБ — это 8×32 ГБ за $110.52 = **$884.16**, имеет смысл только если 128 ГБ реально мало. 2Rx8 не выгоднее. Taobao/Superbuy сегодня отдал Cloudflare 403 — цены оттуда **не проверены**.
3. **Докупить:** SSD 1 ТБ — **469.99 BYN** (Kingston A2000, новый, гарантия 6 мес). Если CPU приедет без carrier-рамки — **$24**. Остальное (24-pin, 2×EPS, 12VHPWR, SATA, паста) в комплекте; переходников не нужно.
4. **5060 Ti:** **продать**, если нет конкретной второй параллельной задачи (eval/inference) или модели >16 ГБ. Новые/открытые 16 ГБ на Kufar: **2600–2700 BYN**; реальная б/у цена ~**2300–2600 BYN** (SPECULATION). DDP с 5070 Ti даёт ~**0–5%**, а не 2×.
5. **Приёмка:** 10 пунктов ниже; главное — фото LGA-пятаков и рамок до оплаты доставки, и **не прошивать BMC 4.x+ на плате с 32 МБ flash**.

---

## Одна страница решений (RU)

| # | Позиция | Решение | Цена | Источник / дата / тег | Почему |
|---|---|---|---|---|---|
| 1 | Кулер SP3, 155 Вт сейчас / 280 Вт потом | **Dynatron A50** | **$41.95** | [S8] MITXPC, 2026-11-07, **CT** (вариант “Default Title”) | 110 мм, 3U+, 280 Вт, SP3; дешевле Noctua в 3 раза |
| 2 | Тихий кулер 155 Вт | Noctua NH-U12S TR4-SP3 | $148.00 | [S11] AVADirect, 2026-10-07, **SC** | 158 мм — если паспорт корпуса ≥158 мм / замер ≥163 мм; 280 Вт не тянет |
| 3 | Максимальный воздух 155 Вт | Noctua NH-U14S TR4-SP3 | $122.50 | [S13] Newegg RSS, 2026-11-07, **SC** | 165 мм — 5U; в 4U не ставить |
| 4 | Серверный 280 Вт, альтернатива | Supermicro SNK-P0064AP4 | $77.00 | [S10] Acmemicro, 2026-11-07, **SC** | 126 мм, 280 Вт; крепление на Tyan **не проверено** |
| 5 | ОЗУ сейчас | 4×16 ГБ 2133 2Rx4 | **$61.97/шт = $247.88** | твоя корзина, 2026-11-07, **OWN** (URL нет — корзину я не открывал) | Ниже рынка; 4 канала из 8, но с твоими 4 планками будет 8/8 |
| 6 | ОЗУ 128 ГБ без корзины | 4×16 ГБ DNA 2Rx4 | $59.80/шт = $239.20 | [S16] ServerTechCentral, 2026-11-07, **CT** | Дешевле на планку, 3 года гарантии; доставка в РБ не включена |
| 7 | ОЗУ 256 ГБ | 8×32 ГБ DNA 2Rx4 | $110.52/шт = **$884.16** | [S18] ServerTechCentral, 2026-11-07, **CT** | $3.45/ГБ — лучший $/ГБ, но в 3.6 раза больше денег; только если 128 ГБ мало |
| 8 | 2Rx8 вместо 2Rx4 | Micron 16 ГБ 2Rx8 | $59.99 | [S14] DIMMsum, 2026-11-06, **AG** | Не дешевле 2Rx4 ($55.39) и мешает ранги с твоими 2Rx4 |
| 9 | Загрузочный SSD | Kingston A2000 1 ТБ | **469.99 BYN** | [S22] Kufar, 2026-11-07, **CT** | Новый, 6 мес гарантии, 2280, PCIe 3.0 — для загрузки хватает |
| 10 | Рамка CPU, если нет | SP3 carrier frame | $24.00 | [S30] CPU Medics, 2026-11-07, **SC** | Нужна, если CPU приедет без синей рамки; иначе не покупать |
| 11 | Кабели БП | 24-pin + 2×EPS + 12VHPWR | $0 | [S21] XPG, 2026-11-07, **CT** | В комплекте; 2×EPS ровно под плату, 10×PCIe в запасе |
| 12 | 5060 Ti | **Продать** (если нет второй ленты) | 2600–2700 BYN | [S23][S24] Kufar, 2026-11-07, **CT/SC** | ~$872–906; DDP с 5070 Ti даёт ~0–5%, не 2× |

---

## 1. Кулер SP3: что реально влезает

### 1.1 Принятая высота

Корпус неизвестен, поэтому принял **два кандидата** по внутренней высоте под кулер (от верхней плоскости CPU IHS до внутренней крышки):

- **148 мм** — консервативный 4U (пример: SilverStone RM46-502-I, 4U, max CPU cooler 148 мм; с GPU-держателем 114 мм).
- **158 мм** — типичный 4U (пример: Sliger CX4170a, 4U, “up to 158 mm tall CPU air coolers”).
- Для справки: 45HomeLab HL15 — ровно **160 мм** от платы до крышки (форум производителя, не спека).

**Что должен замерить владелец:** от **верхней плоскости установленного CPU** (или от сокета + ~7 мм, если CPU ещё не стоит) до внутренней стороны верхней крышки при закрытом корпусе. Нужен зазор **≥ высота кулера + 5 мм**. Если мерить от платы, а не от IHS, вычти ~7 мм на сокет/CPU.

**Почему 4U — это не 177.8 мм:** 177.8 мм — внешняя высота 4U; внутри остаётся 148–160 мм из-за шасси, стоек, крышки и GPU-держателей. Noctua официально ставит NH-U14S (165 мм) в **5U**, NH-U12S (158 мм) — тоже 5U, NH-U9 (125 мм) — 4U/5U.

### 1.2 Кандидаты

| Кулер | Высота | Сокет | TDP | Цена | Источник / дата / тег | Вердикт |
|---|---:|---|---:|---:|---|---|
| **Dynatron A50** | **110 мм** | SP3/TR4 | **280 Вт** | **$41.95** | [S8] MITXPC, 2026-11-07, **CT** | **Брать по умолчанию.** Влезает в 148 и 158 мм; для 155 Вт большой запас, для 280 Вт — по паспорту на пределе; 4-pin PWM; паста Shin-Etsu предустановлена. Шум 25.6/47.9/64.4 дБА на 20/50/100% — для серверной это нормально, для спальни нет. |
| Noctua NH-U12S TR4-SP3 | 158 мм | SP3/TR4 | 180 Вт (NSPR 129) | $148.00 | [S11] AVADirect, 2026-10-07, **SC**; альтернатива €95.54 [S11b] smicro.eu, 2026-11-07, **SC** | Тихий, но ровно на пределе 158 мм; 280 Вт не тянет. Только если замер ≥160 мм и тишина важнее цены. |
| Noctua NH-U14S TR4-SP3 | 165 мм | SP3/TR4 | 180 Вт; 280 Вт только со 2-м вентилятором | $122.50 | [S13] Newegg RSS, 2026-11-07, **SC**; $108.87 [S12] AVADirect, 2026-10-30, **SC** (OOS) | **Не для 4U.** 165 мм = 5U. Для 280 Вт нужен второй NF-A15 и хороший продув; в 4U не влезет. |
| Supermicro SNK-P0064AP4 | 126 мм | SP3 | 280 Вт | $77.00 | [S10] Supermicro/Acmemicro, 2026-11-07, **SC** | Хороший серверный вариант с фронтальным 92-мм вентилятором, но крепление спроектировано под H11/H12; совместимость с Tyan S8030 **не подтверждена**. Покупать только после проверки стоек/винтов. |
| Dynatron A26 | 65 мм | SP3 | 180 Вт | $34.95 | [S9] MITXPC, 2026-11-07, **CT** | Если корпус реально 2U/низкий; 60-мм вентилятор, шумный. Для 4U A50 лучше. |
| Noctua NH-U9 TR4-SP3 | 125 мм | SP3/TR4 | NSPR 118, 2×92 мм | $99.95 | [S13] Newegg RSS, 2026-11-07, **SC** | Тихий 4U-вариант для 155 Вт, но 280 Вт не тянет и дороже A50. |

### 1.3 Что выбрать

- **Сейчас 155 Вт, корпус 4U (148–160 мм):** Dynatron A50. Дешевле, влезает, 280 Вт по паспорту. Проверь, что его 92-мм вентилятор не упирается в крышку и не засасывает горячий воздух; в 4U с фронт-бэк продувом это нормальный режим.
- **Потом 280 Вт (7R32 — из твоего брифа, я его TDP не проверял; 7702 — это 200 Вт, не 280 [S19]):** A50 (280 Вт, 110 мм) или SNK-P0064AP4 (280 Вт, 126 мм, если крепление подойдёт). Noctua NH-U14S — только в 5U/башне и со вторым вентилятором; NH-U12S для 280 Вт не годится.
- **Тишина важнее:** NH-U12S, но только если паспорт корпуса разрешает ≥158 мм; при самостоятельном замере — ≥163 мм (158 + 5 мм запаса). Если 158 мм — он впритык, а 165-мм NH-U14S не лезет.
- **Проверь RAM-зазор:** A50 — 119×82 мм, вентилятор 92 мм может нависать над первым DIMM-слотом; S8030 имеет 8 слотов близко к сокету. Сначала примерь радиатор без пасты, проверь, что не упирается в планки и защёлки. Если мешает — NH-U9 (95 мм шириной) или разверни A50, если крепление позволяет.

---

## 2. Память: 4×16 2133 2Rx4 за $61.97 — да или нет

### 2.1 Рынок на 2026-11-06/07

- **DIMMsum:** 16 ГБ DDR4-2133 RDIMM, б/у: low-end **$4.48/ГБ = $71.69/планка**, медиана **$6.56/ГБ = $104.96**, самая дешёвая живая **$55.39** ($3.46/ГБ). 383–484 листинга. С августа low-end вырос на ~30%.
- **VRAMGlass:** 16 ГБ 2133 — **$73/планка** ($4.56/ГБ), 32 ГБ 2133 — **$153/планка** ($4.78/ГБ), 29–59 листингов, обновлено 2026-11-03.
- **Вывод:** $61.97 = **$3.87/ГБ** — ниже low-end обоих индексов и ниже медианы. Это **хорошая цена**, не “переплата”.

### 2.2 Три проверенных варианта (кликом, без Taobao)

| Вариант | Цена | $/ГБ | Источник / дата / тег | Что выбрано | Комментарий |
|---|---:|---:|---|---|---|
| **4×16 ГБ 2133 2Rx4 (твоя корзина)** | **$61.97/шт = $247.88** | $3.87 | твоя корзина, 2026-11-07, **OWN** | 4 планки | Рекомендую: закрывает 4 свободных канала, всего 8×16 = 128 ГБ, все 8 каналов |
| 4×16 ГБ DNA 2Rx4 | $59.80/шт = $239.20 | $3.74 | [S16] ServerTechCentral, 2026-11-07, **CT** | SKU `PLS-A7910488-DNA`, единственный вариант | Дешевле на $8.68 за 4; 3 года гарантии; доставка в РБ не включена |
| 4×16 ГБ Samsung 2Rx4 | $71.58/шт = $286.32 | $4.47 | [S17] ServerTechCentral, 2026-11-07, **CT** | SKU `PLS-M393A2G40DB0-CPB-DNA` | Дороже; брать только если хочешь именно Samsung |
| **8×32 ГБ DNA 2Rx4 = 256 ГБ** | **$110.52/шт = $884.16** | **$3.45** | [S18] ServerTechCentral, 2026-11-07, **CT** | SKU `PLS-S26361-F3389-L428-DNA`, единственный вариант | Лучший $/ГБ, но в 3.57 раза больше денег; только если 128 ГБ реально мало |
| 8×16 ГБ = 128 ГБ (новые) | $495.76 | $3.87 | [S16] + арифметика | — | **Не имеет смысла:** те же 128 ГБ, что 4 новые + 4 твои, но $248 дороже |
| 16 ГБ 2Rx8 Micron | $59.99 | $3.75 | [S14] DIMMsum, 2026-11-06, **AG** | — | Не дешевле 2Rx4 ($55.39); смешивать ранги с твоими 2Rx4 не стоит |
| 16 ГБ 2Rx4 Crucial (eBay) | $55.39 | $3.46 | [S14] DIMMsum, 2026-11-06, **AG** | — | Самая дешёвая живая; eBay-листинг не открыт (403), только индекс |
| Taobao / Superbuy | **NV** | — | Superbuy 403 Cloudflare, 2026-11-07, **NV** | — | Не проверить ночью; не менять корзину по непроверенным search-card ценам |

### 2.3 Почему 4×16, а не 8×16 или 2Rx8

- **`8×16` — это 128 ГБ, не 256 ГБ.** 256 ГБ на этой плате — только **8×32**. Если ты имел в виду “8 планок по 16”, то это те же 128 ГБ, что 4 новые + 4 существующие, и покупать 8 новых нет смысла.
- **2Rx8 не выгоднее:** проверенная 2Rx8 16 ГБ — $59.99, 2Rx4 — $55.39–61.97. Плюс у тебя уже 4×2Rx4; смешивание 2Rx4/2Rx8 на EPYC обычно работает, но лучше не надо.
- **2133 — самый дешёвый $/ГБ:** 2400 $5.50, 2666 $5.75, 2933 $6.88, 3200 $7.50. Для 128 ГБ разница с 3200 — примерно $960 vs $573, то есть +67% за +50% bandwidth. Для тренировки на GPU это редко окупается.
- **Скорость:** EPYC 7452 — 8 каналов, DDR4-3200 (1DPC), 204.8 ГБ/с. С 2133 — 8×17.07 = **136.5 ГБ/с**, то есть −33% bandwidth. Но 8 каналов даже на 2133 всё равно быстрее, чем 4 канала на 3200; поэтому важно заполнить все 8 слотов, а не гнаться за 3200.
- **2 DIMM на канал:** у EPYC 7002 1DPC = 3200, 2DPC = 2933. **К этой плате не относится:** 8 слотов = 8 каналов, по одному DIMM на канал. E5 v3/v4 тоже не относится — другая платформа.

---

## 3. Что ещё нужно докупить

| Позиция | Кол-во | Цена | Источник / дата / тег | Обязательно? | Комментарий |
|---|---:|---:|---|---|---|
| Кулер Dynatron A50 | 1 | **$41.95** | [S8] MITXPC, 2026-11-07, **CT** | Да | 110 мм, 280 Вт, SP3; паста предустановлена |
| SSD Kingston A2000 1 ТБ | 1 | **469.99 BYN** | [S22] Kufar, 2026-11-07, **CT** | Да (рекомендую) | Новый, 6 мес гарантии, 2280; BIOS/M.2 PCIe 3.0 совместим |
| SSD Samsung 500 ГБ NVMe (альтернатива) | 1 | 309.99 BYN | [S22b] Kufar, 2026-11-07, **SC** | Нет | Если 1 ТБ для загрузки избыточен; экономия ~160 BYN |
| SP3 carrier frame | 1 | $24.00 | [S30] CPU Medics, 2026-11-07, **SC** | Только если CPU без рамки | EPYC обычно приходит в рамке; проверь фото до отправки |
| Термопаста | — | $0 | [S8] A50 | Нет | Shin-Etsu 7762 уже нанесена; NT-H1 в комплекте Noctua |
| SATA-кабели | 2 | $0 | [S2] Tyan UG | Нет | 2 SATA в комплекте платы; для NVMe не нужны |
| 24-pin | 1 | $0 | [S21] XPG | Нет | В комплекте БП |
| EPS 8-pin (4+4) | 2 | $0 | [S21] XPG | Нет | Ровно 2 на плате; не путать с PCIe 6+2 |
| 12VHPWR | 1 | $0 | [S21] XPG | Нет | Для 5070 Ti; нативный 600 Вт |
| PCIe 8-pin | 1 | $0 | [S21] XPG | Нет | Для 5060 Ti; в БП 10 таких |
| Корпусные вентиляторы | ? | **NV** | — | Зависит от корпуса | Если в корпусе нет 3×120 мм на фронт — докупить; цену не проверял |
| PCIe x16 райзер | 0–1 | **NV** | — | Только если 2.5-слотовая 5070 Ti перекроет второй x16 | Цену не проверял; сначала dry-fit |
| Кабель питания C13 | 1 | $0 | [S21] XPG | Нет | Обычно в комплекте БП |

**Итого обязательного:** $41.95 + 469.99 BYN ≈ **595 BYN** (по курсу 2.9806 — ~$199.65). С рамкой CPU: ~**666 BYN** (~$223.5). С 500-ГБ SSD вместо 1 ТБ — ~**435 BYN**. Плюс корпусные вентиляторы/райзер, если понадобятся.

### 3.1 Электрика: хватает ли БП

- Плата: **24-pin + 2×8-pin EPS**. БП XPG CyberCore II: **1×24-pin, 2×EPS 8-pin (4+4), 10×PCIe 6+2, 1×12VHPWR, 12×SATA**. Всё нужное есть, переходники не нужны.
- **Важно:** официально XPG CyberCore II — **1300 Вт**, не 1350 Вт (в линейке 1000/1300). Если на твоём блоке написано 1350 — проверь наклейку; на количество кабелей это не влияет. 1300 Вт достаточно.
- Бюджет: 155 Вт CPU + ~50 Вт плата + 8×3 Вт RAM + 10 Вт NVMe + 20 Вт вентиляторы + 300 Вт 5070 Ti + 180 Вт 5060 Ti ≈ **739 Вт** (57% от 1300). С 280 Вт CPU ≈ **864 Вт** (66%). Запас есть.
- **EPS не путать с PCIe:** для платы использовать только кабели “ATX 12V 4+4” (EPS), не “PCIe 6+2”. Они по-разному ключуются; втыкать PCIe в EPS нельзя.

### 3.2 Что может не влезть

- **Два GPU:** 5070 Ti — 2–2.5 слота, 5060 Ti — 2 слота. На плате 5×x16, но расстояние между слотами может не дать поставить обе без райзера. Сначала dry-fit: 5070 Ti в верхний x16, 5060 Ti — в самый нижний свободный x16; если 2.5-слотовая карта перекрывает соседний слот, нужен райзер (цену не проверял).
- **Ширина платы:** 305×250.8 мм — на 6.8 мм шире стандартного ATX (305×244). Большинство 4U с SSI-CEB/EEB это переживут, но проверь стойки и кабельные каналы.
- **БП 160 мм:** в некоторые короткие 4U корпуса влезает только 150 мм; проверь до покупки. XPG — 160×150×86 мм.

---

## 4. 5060 Ti: продать или оставить

### 4.1 Что даёт рынок (Kufar, 2026-11-07)

| Листинг | Цена | Состояние | Источник / тег | Комментарий |
|---|---:|---|---|---|
| Zotac RTX 5060 Ti 16GB Twin Edge OC White | **2600 BYN** | Новое | [S24] Kufar, 2026-11-07, **CT** | Самый дешёвый новый 16 ГБ |
| “RTX 5060 ti 16 gb” | **2700 BYN** | Новое, открыто для проверки | [S23] Kufar, 2026-11-07, **CT** | Коробка, документы, гарантия |
| Palit Infinity 3 16GB | **2950 BYN** | Новое, магазин, гарантия | [S25] Kufar, 2026-11-07, **CT** | X-Trade, 2 слота, 8-pin |
| Palit Infinity 3 16GB | **3290 BYN** | Новое, магазин, 24 мес | [S26] Kufar, 2026-11-07, **CT** | QUADROBER PC |
| MSI Ventus 2X Plus OC 16GB | 3278.15 BYN | Новое, магазин | [S26b] Kufar, 2026-11-07, **SC** | Не открывал |
| Частное б/у, оценка | **2300–2600 BYN** | Б/у | **SPECULATION** | Новые/открытые начинаются с 2600; б/у без гарантии должен быть ниже |

Курс: **2.9806 BYN/USD** на 2026-11-05 ([S29] select.by). То есть 2600 BYN ≈ **$872**, 2700 ≈ **$906**, 2950 ≈ **$990**, 3290 ≈ **$1104**.

### 4.2 Что даёт вторая карта

- **DDP (одна модель, две карты):** 5070 Ti ~43.9 TFLOPS FP32, 5060 Ti ~23.7 TFLOPS. В DDP шаг ждёт медленную карту: **SPECULATION** — расчёт по FP32 даёт 2×23.7/43.9 = **1.08×**, после communication реально **~0–5%**. Не 2×; на этой паре не замерял.
- **Две независимые задачи (train + eval/inference):** **+100%** по числу параллельных задач. Это единственный режим, где карта реально окупается.
- **Модель >16 ГБ (pipeline/model parallel):** 32 ГБ суммарно, но карты разные, балансировать слои сложно; реалистично **+30–50%**, не 2× (SPECULATION).
- **Энергия:** 180 Вт × 24/7 = 1 577 кВт·ч/год. При 0.25 BYN/кВт·ч — ~394 BYN/год; при 0.30 — ~473 BYN/год (тариф не проверял, SPECULATION).

### 4.3 Рекомендация

**Продать**, если нет конкретной второй ленты. Причины:
1. Твой же VRAM-ledger [S32] говорит, что 400M b11 помещается в одну 16 ГБ карту, а 1B dense при full checkpoint — в b13. Для текущих моделей 16 ГБ хватает; вторая карта не нужна для capacity.
2. DDP с 5070 Ti даёт ~0–5%, а не 2×; ты заплатишь энергией и шумом за почти ничего.
3. Локальная цена высокая: 2600–2700 BYN за новый/открытый 16 ГБ — это ~$872–906, почти стоимость 256 ГБ RAM ($884). Если 128 ГБ станет мало, продажа карты почти финансирует 8×32.

**Оставить**, только если:
- ты реально будешь гонять вторую независимую задачу (eval/inference/data generation) параллельно с тренировкой; или
- тебе нужна одна модель >16 ГБ, и ты готов возиться с pipeline parallelism.

**Break-even vs облако (SPECULATION, цену облака не проверял):** при условных $0.10–0.20/ч за 5060 Ti 16GB 2600 BYN ≈ $872 покупает **4 360–8 720 GPU-часов**, то есть **6–12 месяцев 24/7**. Если вторая лента будет занята меньше — продажа выгоднее; если больше — оставлять разумно.

Если продавать — цель **2600–2700 BYN**, не отдавать ниже **2300 BYN**. Если оставлять — не жди 2× на DDP.

---

## 5. Чек-лист приёмки платы (10 строк, по порядку)

1. **До оплаты международной доставки** — попроси фото сокета без PnP-крышки: макро всех LGA-пятаков, нет погнутых/отсутствующих; фото rail frame, force frame и 4 стоек крепления радиатора.
2. Фото шильдика: **S8030GM2NE** (не GM4NE-2T), ревизия платы, наклейки BIOS/BMC, серийник; подтвердить, что это не ES/инженерный образец.
3. Фото обоих M.2 (стойки/винты), I/O shield, 2 SATA-кабеля, коробка аксессуаров; **CPU carrier frame** — в комплекте с CPU или докупить.
4. Фото обратной стороны: нет изгибов PCB, коррозии, следов залития, вздутых/сгоревших элементов; отдельно — BMC и LAN-магнетики.
5. Фото 24-pin + 2×8-pin EPS и fan-хедеров: не выломаны, не оплавлены. Только после этих фото — оплата доставки и страховка.
6. **День 1:** визуальный осмотр под ярким светом; ещё раз пятаки сокета; установить CPU в carrier, закрыть force frame T20 в порядке **1→2→3**; радиатор ровно, паста тонким слоем; **до финальной затяжки** проверить, что радиатор/вентилятор не задевает RAM, защёлки DIMM и верхний PCIe; проверить комплект крепежа/backplate.
7. Первый старт: BIOS defaults, память **2133/Auto**, ECC on; проверить **8 DIMM / 128 ГБ / 8 каналов**; записать версии BIOS/BMC. **Не прошивать BMC 4.x+ на плате с 32 МБ flash** — сначала проверить размер чипа/ревизию (ранние платы с 32 МБ могут окирпичиться; community-предупреждение [S31], подтверди у Tyan).
8. **Memtest86+** полный прогон на ночь (все 8 планок); проверить BMC SEL на ECC-ошибки. Затем CPU stress (stress-ng/Prime95) 30 мин: температура <90 °C, без throttle.
9. GPU по одной: PCIe x16 Gen4, Resizable BAR, IOMMU; короткий burn-in; **12VHPWR/8-pin до щелчка**, проверить температуру разъёма. Затем обе карты: зазоры, температуры, питание.
10. IPMI/BMC: статический IP, смена дефолтного пароля, watchdog, remote console, fan control, **power loss recovery = on**; записать всё в лог. Только после этого — в 24/7.

---

## Internal notes (EN)

### Evidence quality

- **Board specs:** primary vendor (MiTAC/Tyan) spec page and UG PDF. High confidence.
- **Cooler dimensions/TDP:** Noctua official spec pages; Dynatron via MITXPC (retailer) and Dynatron PDF; Supermicro store page. High confidence on dimensions, medium on TDP claims (Supermicro store says 280 W; some resellers say 180 W).
- **4U clearance:** manufacturer pages (Sliger 158 mm, SilverStone 148 mm) + vendor forum (HL15 160 mm). Medium-high; the owner's chassis is unknown, so the measurement is mandatory.
- **Memory prices:** DIMMsum and VRAMGlass are aggregators/indexes, not retailers. ServerTechCentral prices are click-through single-SKU pages. High confidence on the market direction; medium on exact landed cost to Belarus.
- **Taobao/Superbuy:** Cloudflare 403 on search and item pages; Taobao returns a login shell. No Taobao price is verified. This is a hard gap, not an estimate.
- **GPU resale:** Kufar listings are live asks, not completed sales. New/open-box asks are 2600–3290 BYN. Used private estimate 2300–2600 is inference, not a verified sale.
- **BIOS/BMC:** Tyan download page gives V4.04 / V6.02. The 32 MB BMC flash warning comes from a ServeTheHome thread, not a Tyan release note I opened. Treat as a caution, verify before flashing.

### Rejected candidates and why

- **Noctua NH-U14S TR4-SP3 (165 mm):** rejected for 4U. Noctua itself rates it 5U; 4U chassis max 148–160 mm. Only valid in a 5U/tower or a measured ≥170 mm (165 + 5 mm).
- **Noctua NH-U12S TR4-SP3 (158 mm):** not rejected outright, but not default: at the 158 mm limit, $148, and only 180 W. The A50 is cheaper, lower, and 280 W-rated.
- **Noctua NH-U9 TR4-SP3 (125 mm):** good 4U fallback for 155 W, but $99.95 and no 280 W headroom. A50 wins on price/headroom.
- **Supermicro SNK-P0064AP4:** technically the best server airflow (front 92 mm fan, 280 W, 126 mm), but the mounting is validated for Supermicro H11/H12. Tyan S8030 compatibility is not documented; rejected as default until the owner confirms the standoffs/screws.
- **Dynatron A26 (65 mm):** only for 2U/very low; 60 mm fan, 180 W. Not needed in 4U.
- **8×16 GB "256 GB":** arithmetic error. 8×16 = 128 GB. 256 GB requires 8×32.
- **2Rx8:** no price advantage in verified listings; mixing ranks with existing 2Rx4 is a needless risk.
- **Alibaba $13–15 used 16 GB:** rejected as actionable evidence. The page is a wholesale aggregator, the price is far below every live index, and it may be stale/bait. Do not change the cart on it.
- **Taobao search-card prices:** rejected because Superbuy/Taobao could not be opened; search cards show the cheapest variant of a multi-variant listing, exactly the trap the task warns about.

### Assumptions

- The chassis is a standard 4U with 148–160 mm internal cooler clearance. If it is a 5U/tower, the NH-U14S becomes viable.
- The existing 4×16 GB sticks are 2Rx4 2133 and will run at 2133 with the new 4. If they are a different rank/brand, the mixed configuration may train at a lower speed or fail; match part numbers where possible.
- The PSU is the XPG CyberCore II 1300 W (the official model); the “1350 W” in the brief is likely a typo. Cable count is sufficient either way.
- The 5070 Ti uses a 12VHPWR/12V-2x6 connector and the 5060 Ti uses one 8-pin. If the actual 5070 Ti model uses 3×8-pin, the PSU still has 10 PCIe connectors, so it is fine.
- The board has 8 slots / 8 channels, 1 DIMM per channel. The 2DPC ceiling (2933 for EPYC 7002) does not apply.
- Windows/Linux driver support for EPYC 7452 + S8030 is out of scope; BIOS V4.04 supports EPYC 7002/7003.

### Open questions

1. **Chassis internal height** from CPU IHS to lid — decides A50 vs NH-U12S vs NH-U14S.
2. **GPU slot spacing** — will a 2.5-slot 5070 Ti + 2-slot 5060 Ti fit without a riser? Dry-fit before buying a riser.
3. **PSU label** — 1300 W or 1350 W; cable count and 160 mm depth.
4. **Board BMC flash size** — 32 MB vs 64 MB; do not flash BMC 4.x+ on 32 MB.
5. **CPU carrier** — does the EPYC 7452 arrive in the blue SP3 carrier frame?
6. **Taobao memory price** — not verified; the owner can check the Superbuy app, but the cart price is already below global market.
7. **Actual 5060 Ti model** — 2-slot/8-pin vs 16-pin; affects cable and slot fit.

---

## VERIFIED (sourced)

- Board: Tyan/MiTAC S8030GM2NE, ATX 305×250.8 mm, SP3, EPYC 7002/7003, max 280 W cTDP, 8 DIMM/8 channels, DDR4-3200, 5×PCIe 4.0 x16, 2×M.2, 2×SATA + 3×SFF-8643, 2×SFF-8654, AST2500 IPMI, 24-pin + 2×8-pin EPS. [S1][S2]
- Board package: motherboard, quick guide, I/O shield, 2 SATA cables, 2 M.2 snap clips. No CPU heatsink. [S2][S33]
- BIOS latest V4.04 (2025-02-26); BMC latest V6.02 (2024-10-24). [S3]
- EPYC 7452: 32C/64T, 155 W, 8 channels, DDR4-3200 1DPC, 204.8 GB/s. [S19]
- EPYC 7002 2DPC ceiling 2933; 1DPC 3200. [S20]
- Noctua NH-U14S: 165 mm, 5U, NSPR 162, SP3/TR4. [S4]
- Noctua NH-U12S: 158 mm, 5U, NSPR 129, SP3/TR4. [S5]
- Noctua NH-U9: 125 mm, 4U/5U, NSPR 118. [S6]
- Sliger CX4170a 4U: max CPU air cooler 158 mm. [S7]
- Dynatron A50: 119×82×110 mm, 3U+, 280 W, 4-pin PWM, $41.95. [S8]
- Dynatron A26: 119×79×65 mm, 2U, 180 W, $34.95. [S9]
- Supermicro SNK-P0064AP4: 4U, 126 mm, 280 W, SP3, $77 (Acmemicro). [S10]
- XPG CyberCore II: 1300 W, 24-pin ×1, EPS ×2, PCIe 6+2 ×10, 12VHPWR ×1, SATA ×12. [S21]
- DIMMsum 16 GB 2133: low-end $4.48/GB ($71.69), cheapest $55.39, median $6.56/GB, 2026-11-06. [S14]
- VRAMGlass: 16 GB 2133 $73, 32 GB 2133 $153, 2026-11-03. [S15]
- ServerTechCentral: 16 GB 2133 2Rx4 $59.80; 16 GB Samsung $71.58; 32 GB 2133 2Rx4 $110.52; 3-year warranty, 10 in stock. [S16][S17][S18]
- Kufar SSD: Kingston A2000 1 TB, 469.99 BYN, new, 6-month warranty. [S22]
- Kufar GPU: Zotac 16 GB 2600 BYN new; generic 16 GB 2700 BYN new/open; Palit 2950 BYN new shop; Palit 3290 BYN new shop 24 mo. [S23][S24][S25][S26]
- NBRB USD/BYN 2.9806 on 2026-11-05. [S29]
- RTX 5060 Ti 16 GB: 180 W, 1×8-pin, dual-slot. [S27]
- RTX 5070 Ti: 300 W, 1×16-pin, 2–2.5 slot. [S28]
- SP3 carrier frame: $24. [S30]

## SPECULATION (my inference)

- **Used 5060 Ti private resale 2300–2600 BYN.** No completed-sale data; inferred from the 2600 BYN new/open-box floor and the usual discount for no local warranty. If you sell with receipt and remaining warranty, 2600–2700 is plausible.
- **DDP speedup 0–5%.** Calculated from FP32 ratios (43.9 vs 23.7 TFLOPS) and typical communication overhead; not measured on this exact pair.
- **Pipeline/model-parallel gain 30–50% for >16 GB models.** General expectation for two mismatched consumer cards; not measured.
- **256 GB is “better $/GB but worse total value”** for this build: $3.45/GB vs $3.87/GB, but $884 vs $248. Whether it is worth it depends on the dataset/model RAM needs, which I cannot verify.
- **A50 fits the Tyan S8030.** It is a standard SP3 4-screw cooler; the Tyan socket uses the standard SP3 force frame. Not physically tested; buy from a returnable source if unsure.
- **BMC 32 MB warning.** Community report, not a Tyan release note I opened; verify the flash chip/revision before flashing.
- **Power budget 739 W / 864 W.** Sum of nameplate TDPs, not measured wall power; transient spikes are higher.

## Confidence

- **High:** board specs, cooler dimensions, PSU cable count, memory market direction, Kufar asks.
- **Medium:** exact landed cost to Belarus, A50 mounting on Tyan, used GPU resale.
- **Low / not verified:** Taobao/Superbuy prices, BMC 32 MB flash behavior, GPU rerun speedup on this exact pair.

---

## Sources

- [S1] MiTAC/Tyan S8030GM2NE specification — https://www.mitaccomputing.com/products/detail/S8030GM2NE (accessed 2026-11-07)
- [S2] Tyan S8030 User Guide v2.0b — https://download2.mitaccomputing.com/pub/doc/S8030_UG_v2.0b.pdf (accessed 2026-11-07)
- [S3] Tyan S8030GM2NE downloads — https://u379-2jkd-gd90.mitaccomputing.com/en/downloads/Motherboards/S8030_S8030GM2NE (accessed 2026-11-07)
- [S4] Noctua NH-U14S TR4-SP3 specifications — https://www.noctua.at/en/products/nh-u14s-tr4-sp3/specifications (accessed 2026-11-07)
- [S5] Noctua NH-U12S TR4-SP3 specifications — https://www.noctua.at/en/products/nh-u12s-tr4-sp3/specifications (accessed 2026-11-07)
- [S6] Noctua NH-U9 TR4-SP3 specifications — https://www.noctua.at/en/products/nh-u9-tr4-sp3/specifications (accessed 2026-11-07)
- [S7] Sliger CX4170a 4U rackmount case — https://www.density.sk/cx4170a-rackmount-case-4u (accessed 2026-11-07)
- [S8] MITXPC Dynatron A50 — https://mitxpc.com/products/a50 (accessed 2026-11-07)
- [S9] MITXPC Dynatron A26 — https://mitxpc.com/products/a26 (accessed 2026-11-07)
- [S10] Supermicro SNK-P0064AP4 — https://store.supermicro.com/us_en/4u-active-amd-epyc-snk-p0064ap4.html; price at https://www.acmemicro.com/Product/16292/Supermicro-SNK-P0064AP4-4U-Active-AMD-EPYC-7000-Series-Processors-Heat-Sink-Socket-SP3-with-Cooling-Fan (accessed 2026-11-07)
- [S11] AVADirect NH-U12S TR4-SP3 — https://www.avadirect.com/NH-U12S-TR4-SP3-158mm-Height-160W-TDP-Copper-Aluminum-CPU-Cooler/Product/11494118 (search snippet, 2026-10-07)
- [S11b] smicro.eu NH-U12S TR4-SP3 — https://smicro.eu/noctua-nh-u12s-tr4-sp3-amd-socket-tr4-sp3-nh-u12s-tr4-sp3-2 (search snippet, accessed 2026-11-07)
- [S12] AVADirect NH-U14S TR4-SP3 — https://www.avadirect.com/NH-U14S-TR4-SP3-165mm-Height-200W-TDP-Copper-Aluminum-CPU-Cooler/Product/12154094 (search snippet, 2026-10-30)
- [S13] Newegg RSS, socket SP3 coolers — https://www.newegg.com/d/Product/RSS?Description=socket%20sp3&IsNodeId=1&N=100006640&Submit=ENE (accessed 2026-11-07)
- [S14] DIMMsum 16 GB DDR4-2133 RDIMM — https://dimmsum.com/ddr4/16gb-2133-rdimm (2026-11-06)
- [S15] VRAMGlass DDR4 tracker — https://vramglass.com/ddr4 (2026-11-03)
- [S16] ServerTechCentral DNA 16 GB 2133 2Rx4 — https://servertechcentral.com/product/pls-a7910488-dna (accessed 2026-11-07)
- [S17] ServerTechCentral DNA 16 GB 2133 2Rx4 (Samsung) — https://servertechcentral.com/product/pls-m393a2g40db0-cpb-dna (accessed 2026-11-07)
- [S18] ServerTechCentral DNA 32 GB 2133 2Rx4 — https://servertechcentral.com/product/pls-s26361-f3389-l428-dna (accessed 2026-11-07)
- [S19] AMD EPYC 7002 series datasheet — https://www.amd.com/content/dam/amd/en/documents/products/epyc/amd-epyc-7002-series-datasheet.pdf (accessed 2026-11-07)
- [S20] Thomas-Krenn, EPYC 7002/7003 memory performance — https://www.thomas-krenn.com/en/wiki/Optimization_of_AMD_EPYC_7002_Rome_and_7003_Milan_working_memory_performance (accessed 2026-11-07)
- [S21] XPG CyberCore II specifications — https://www.xpg.com/en/xpg/pc-components-cybercore-ii?tab=spec (accessed 2026-11-07)
- [S22] Kufar Kingston A2000 1 TB — https://www.kufar.by/item/188113093 (accessed 2026-11-07)
- [S22b] Kufar Samsung 500 GB NVMe — https://www.kufar.by/item/1013556608 (search card, accessed 2026-11-07)
- [S23] Kufar RTX 5060 Ti 16 GB, 2700 BYN — https://www.kufar.by/item/1092441389 (accessed 2026-11-07)
- [S24] Kufar Zotac RTX 5060 Ti 16 GB, 2600 BYN — https://www.kufar.by/item/1090912483 (accessed 2026-11-07)
- [S25] Kufar Palit Infinity 3 16 GB, 2950 BYN — https://www.kufar.by/item/1037235625 (accessed 2026-11-07)
- [S26] Kufar Palit Infinity 3 16 GB, 3290 BYN — https://www.kufar.by/item/1040386784 (accessed 2026-11-07)
- [S26b] Kufar MSI Ventus 2X Plus OC 16 GB, 3278.15 BYN — https://www.kufar.by/item/1066958510 (search card, accessed 2026-11-07)
- [S27] TechPowerUp RTX 5060 Ti 16 GB — https://www.techpowerup.com/gpu-specs/geforce-rtx-5060-ti-16-gb.c4292 (2026-11-04)
- [S28] TechPowerUp RTX 5070 Ti — https://www.techpowerup.com/gpu-specs/geforce-rtx-5070-ti.c4243 (2026-11-02)
- [S29] select.by NBRB USD rate — https://select.by/kursy-valyut/natsbank-rb/dollar-na-zavtra (2026-11-05, 2.9806 BYN/USD)
- [S30] CPU Medics SP3 carrier frame — https://www.cpumedics.com/d-c-18243-rev-t-amd-epyc-threadripper-sp3-tr4-trx4-blue-socket-cpu-carrier-frame-mount-caddy-tray (accessed 2026-11-07)
- [S31] ServeTheHome Tyan S8030 thread (BMC 32 MB warning) — https://forums.servethehome.com/index.php?threads/tyan-s8030gm2ne.28914/page-20 (accessed 2026-11-07; community, not vendor)
- [S32] Dormouse VRAM ledger — `docs/records/reports/2026-11-06-vram-ledger.md` (local, 2026-11-06)
- [S33] Tyan S8030 Quick Installation Guide v2.0 — https://download2.mitaccomputing.com/pub/doc/S8030_QIG_v2.0.pdf (accessed 2026-11-07)
