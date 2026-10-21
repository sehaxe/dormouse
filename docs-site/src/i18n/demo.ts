/**
 * The demo page's copy, in the two languages the site publishes.
 *
 * The page used to be English-only with a `[ru]` link that pointed at the
 * Russian LANDING — clicking it left the page you were reading. The copy lives
 * here so `components/Demo.astro` can be one document rendered twice
 * (`/demo/` and `/ru/demo/`), the way the landing already works, and so the
 * language switch can point at the SAME page in the other language.
 *
 * The runtime strings (the engine's status lines, the byte counts, the
 * generator's rate) are NOT here: they are terminal output, and the terminal
 * speaks English — the design language's own rule is that the card mocks up a
 * TUI. Only the page's own copy is translated.
 */

export const EN = {
	title: 'dormouse — demo: a byte-level LM in your browser',
	description:
		'A 9.5M-parameter drowsy-byte LM running as a plain-JavaScript inference engine in the browser: byte vocabulary, KDA linear attention with a carried recurrent state, TSCT low-rank experts, Engram n-gram memory — and the same engine runs a dropless top-2-of-256 artifact at `medium` geometry, with or without the memory arm.',
	h1: 'demo',
	tagline:
		'The <code>small</code> polygon — 9,476,993 parameters — running in your browser as a plain-JavaScript inference engine. No server, no WebGPU, no WASM: the same <code>.dormouse</code> artifact <code>dormouse serve --export</code> loads. The same engine also runs a <code>medium</code>-shaped artifact (1024-wide, 256 experts, <code>moe_topk = 2</code> dropless); the full one is 138.1 MB, and its memory-off export (<code>?engram=0</code>) is 37.3 MB.',
	engineHeading: 'the engine',
	engineRows: [
		{
			dt: 'bytes are the vocabulary',
			dd: "There is no tokenizer: the prompt is UTF-8 bytes and every position is one of 256 symbols. Type anything — Russian, code, binary — and it is already in the model's vocabulary.",
		},
		{
			dt: 'KDA linear attention',
			dd: 'The gated delta rule, one byte at a time, with the recurrent state carried across bytes: <code>n_heads × head_dim²</code> = 49,152 fp32 values per iteration slot (196 KB), independent of how long the context gets. There is no KV cache to grow.',
		},
		{
			dt: 'the weight-shared loop',
			dd: 'One block, four iterations: a controller gates the KDA arm, the Engram memory and the experts per iteration, and the readout is the average over iterations. The parameters do not multiply with depth.',
		},
		{
			dt: 'TSCT low-rank experts',
			dd: 'A routed bank plus an always-on shared expert, each a <code>U·diag(s)·Vᵀ</code> factorization. In the artifact the factors are 2-bit ternary codes (four per byte) with their scale in the header; the engine keeps the codes and decodes each one inside the matvec, so it never holds a float copy (79.4M factors: 19.9 MB packed, 318 MB widened). Both dispatch arms are here: the dense softmax over 3 experts (<code>small</code>) and the dropless top-2 of 256 (<code>medium</code>) — every routed token to its top-k, no capacity drop.',
		},
		{
			dt: 'Engram memory',
			dd: "25,000 rows per order over byte n-grams of orders [2, 3, 4] — FNV-1a hashed, 32 wide, gated into the memory branch behind the hard floor <code>min(w_mem, lam_max)</code>. 33% of the model's parameters, and a lookup, not a matmul. An fp16 table stays a <code>Uint16Array</code> view of the artifact; only the 32 values a step reads are converted.",
		},
	],
	verified:
		'Verified against the Python reference by <code>tests/test_demo_engine.py</code>: the engine\'s logits agree with <code>decode._step_logits</code> to a max |Δlogit| of <strong>1.058e-6</strong> on <code>small</code> (tolerance 1e-4), <strong>1.013e-6</strong> on a 138M-parameter <code>medium</code>-shaped artifact, and <strong>2.384e-7</strong> on the routed top-2 fixture — with an identical greedy byte at every prefix.',
	runHeading: 'run it',
	termBarLeft: 'dormouse demo',
	statusLoading: 'loading the engine…',
	promptLabel: 'prompt — bytes, not tokens',
	promptDefault: 'The dormouse is a small',
	bytesLabel: 'bytes',
	tempLabel: 'temperature',
	generate: 'generate',
	stop: 'stop',
	greedyNote:
		"temperature 0 is greedy — the demo's own rule; the Python decoder refuses it rather than approximating it.",
	engramOffLabel: 'engram off — 37 MB',
	engramSwitch:
		'Memory arm: the page loads the full artifact by default. Tick <strong>engram off</strong> (or append <code>?engram=0</code>) to fetch the medium-shaped export with the 100.7 MB table removed — 37.3 MB instead of 138.1 MB, for phones. The parity gate keeps Engram on; the status line then says <strong>engram off</strong> because the memory branch is an exact zero, not the full model.',
	outLabel: 'output',
	notHeading: 'what this is not',
	notRows: [
		{ name: 'the 9.5M small polygon', delta: 'an A/B instrument' },
		{ name: 'a chatbot', delta: 'no SFT, no chat template' },
		{ name: 'a shipped full medium artifact', delta: '138.1 MB; the page ships the 37.3 MB engram-off export' },
		{ name: 'a ternary GPU kernel', delta: 'codes decode inside the matvec; no int8/LUT kernel' },
		{ name: 'bf16 compute', delta: 'the engine is the fp32 arm' },
		{ name: 'a fast path', delta: 'one thread, ~7.7 B/s at medium' },
	],
	notProse:
		"This is the model the project's A/B protocol is run on, not a release. It predicts the next byte; it has never been instruction-tuned, and nothing here is a claim about language quality. What the page does demonstrate is the architecture: the arithmetic the Python trainer computes is the arithmetic this engine computes — the gate above is that statement, and it is checked on every test run. The artifact carries the weights only; the loss, the optimizer and the training step live in the checkpoint, which is a different file for a different job.",
	notNote:
		'The weights are a <code>.dormouse</code> artifact written by <code>dormouse export</code>; the small file is 16.3 MB (15.51 MiB) and the medium memory-off export is 37.3 MB (35.5 MiB), both committed because the page fetches them. The full medium artifact is 138.1 MB (131.7 MiB) and runs at ~7.7 B/s in Node; it is generated, not committed (GitHub\'s per-file limit is 100 MB). Packing puts its tensor heap at 155.3 MB instead of 1 GB: the factors stay 2-bit codes and the fp16 Engram table stays fp16, decoded per row. <a href="{status}">Status</a> has the measured numbers, including the retractions.',
	footerNote: '©2026 dormouse — every number carries its measurement.',
	docs: 'Docs',
	status: 'Status',
};

export const RU = {
	title: 'dormouse — демо: байтовая LM прямо в браузере',
	description:
		'LM на 9,5 млн параметров, работающая в браузере как обычный JavaScript-движок вывода: байтовый словарь, линейное внимание KDA с переносимым рекуррентным состоянием, низкоранговые эксперты TSCT, n-граммная память Engram — и тот же движок запускает dropless top-2 из 256 на геометрии `medium`, с памятью и без неё.',
	h1: 'демо',
	tagline:
		'Полигон <code>small</code> — 9 476 993 параметра — работает в браузере как обычный JavaScript-движок вывода. Без сервера, без WebGPU, без WASM: тот же артефакт <code>.dormouse</code>, который загружает <code>dormouse serve --export</code>. Тот же движок запускает и артефакт геометрии <code>medium</code> (1024 в ширину, 256 экспертов, <code>moe_topk = 2</code>, dropless); полный весит 138,1 МБ, а его экспорт без памяти (<code>?engram=0</code>) — 37,3 МБ.',
	engineHeading: 'движок',
	engineRows: [
		{
			dt: 'словарь — байты',
			dd: 'Токенизатора нет: промпт — это UTF-8 байты, и каждый символ — одно из 256 значений. Пишите что угодно — по-русски, код, бинарь — всё уже в словаре модели.',
		},
		{
			dt: 'линейное внимание KDA',
			dd: 'Гейт-дельта-правило, по одному байту за шаг, рекуррентное состояние переносится между байтами: <code>n_heads × head_dim²</code> = 49 152 значения fp32 на слот итерации (196 КБ) — независимо от длины контекста. KV-кэша, который растёт, здесь нет.',
		},
		{
			dt: 'блок с общими весами',
			dd: 'Один блок, четыре итерации: контроллер управляет плечами KDA, памятью Engram и экспертами на каждой итерации, а выход — среднее по итерациям. Параметры не умножаются на глубину.',
		},
		{
			dt: 'низкоранговые эксперты TSCT',
			dd: 'Маршрутизируемый банк плюс всегда включённый общий эксперт; каждый — факторизация <code>U·diag(s)·Vᵀ</code>. В артефакте факторы хранятся 2-битными тернарными кодами (четыре в байте), скейл лежит в заголовке; движок держит коды и декодирует каждый прямо в matvec, поэтому не создаёт float-копию (79,4M факторов: 19,9 МБ упакованных против 318 МБ развёрнутых). Здесь оба плеча диспетчеризации: плотная softmax-смесь по 3 экспертам (<code>small</code>) и dropless top-2 из 256 (<code>medium</code>) — каждый маршрутизированный токен идёт к своим top-k, без capacity-дропов.',
		},
		{
			dt: 'память Engram',
			dd: '25 000 строк на порядок по байтовым n-граммам порядков [2, 3, 4] — хэш FNV-1a, ширина 32, гейт в плечо памяти через жёсткий пол <code>min(w_mem, lam_max)</code>. 33% параметров модели, и это поиск в таблице, а не матричное умножение. Таблица в fp16 остаётся <code>Uint16Array</code>-вью над артефактом; конвертируются только 32 значения, которые читает шаг.',
		},
	],
	verified:
		'Проверено против питоновского эталона в <code>tests/test_demo_engine.py</code>: логиты движка совпадают с <code>decode._step_logits</code> с максимальной |Δlogit| <strong>1.058e-6</strong> на <code>small</code> (допуск 1e-4), <strong>1.013e-6</strong> на артефакте геометрии <code>medium</code> (138M параметров) и <strong>2.384e-7</strong> на routed-фикстуре top-2 — жадный байт совпадает на каждом префиксе.',
	runHeading: 'запустить',
	termBarLeft: 'dormouse demo',
	statusLoading: 'загружаю движок…',
	promptLabel: 'промпт — байты, не токены',
	promptDefault: 'Соня — это маленький',
	bytesLabel: 'байтов',
	tempLabel: 'температура',
	generate: 'сгенерировать',
	stop: 'стоп',
	greedyNote:
		'температура 0 — это жадный выбор, правило самого демо; питоновский декодер его отвергает, а не приближает.',
	engramOffLabel: 'engram off — 37 МБ',
	engramSwitch:
		'Плечо памяти: по умолчанию страница грузит полный артефакт. Поставьте <strong>engram off</strong> (или добавьте <code>?engram=0</code>), чтобы загрузить экспорт геометрии medium без таблицы на 100,7 МБ — 37,3 МБ вместо 138,1 МБ, для телефона. Гейт паритета держит Engram включённым; в статусе тогда написано <strong>engram off</strong>, потому что плечо памяти — точный ноль, а не полная модель.',
	outLabel: 'вывод',
	notHeading: 'чем это не является',
	notRows: [
		{ name: 'полигон small на 9,5M', delta: 'инструмент для A/B' },
		{ name: 'чат-бот', delta: 'нет SFT, нет шаблона чата' },
		{ name: 'полный артефакт medium на странице', delta: '138,1 МБ; страница отдаёт 37,3 МБ экспорт без Engram' },
		{ name: 'тернарное ядро на GPU', delta: 'коды декодируются внутри matvec, без int8/LUT-ядра' },
		{ name: 'вычисления в bf16', delta: 'движок — плечо fp32' },
		{ name: 'быстрый путь', delta: 'один поток, ~7,7 Б/с на medium' },
	],
	notProse:
		'Это модель, на которой в проекте гоняется A/B-протокол, а не релиз. Она предсказывает следующий байт; её никогда не дообучали на инструкциях, и здесь нет никакого утверждения о качестве языка. Что страница действительно показывает — это архитектуру: арифметика, которую считает питоновский тренер, — та же арифметика, которую считает этот движок; гейт выше и есть это утверждение, и он проверяется на каждом прогоне тестов. В артефакте лежат только веса; лосс, оптимизатор и шаг обучения живут в чекпоинте — это другой файл для другой задачи.',
	notNote:
		'Веса — это артефакт <code>.dormouse</code>, который пишет <code>dormouse export</code>; small весит 16,3 МБ (15,51 MiB), а экспорт medium без памяти — 37,3 МБ (35,5 MiB); оба закоммичены, потому что их грузит страница. Полный артефакт medium — 138,1 МБ (131,7 MiB) и идёт ~7,7 Б/с в Node; он генерируется, а не коммитится (лимит GitHub — 100 МБ на файл). Упаковка даёт тензорную кучу 155,3 МБ вместо 1 ГБ: факторы остаются 2-битными кодами, а fp16-таблица Engram — fp16 и декодируется построчно. Измеренные числа, включая отозванные утверждения, — в разделе <a href="{status}">Status</a>.',
	footerNote: '©2026 dormouse — каждое число несёт своё измерение.',
	docs: 'Документация',
	status: 'Статус',
};

export const COPY = { en: EN, ru: RU };
export type Lang = keyof typeof COPY;
