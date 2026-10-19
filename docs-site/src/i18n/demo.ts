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
	title: 'dormouse — demo: the small preset in your browser',
	description:
		'The 9.5M-parameter `small` dormouse polygon, running as a plain-JavaScript inference engine in the browser: byte vocabulary, KDA linear attention with a carried recurrent state, TSCT low-rank experts, Engram n-gram memory.',
	h1: 'demo',
	tagline:
		'The <code>small</code> polygon — 9,476,993 parameters — running in your browser as a plain-JavaScript inference engine. No server, no WebGPU, no WASM: the same <code>.dormouse</code> artifact <code>dormouse serve --export</code> loads.',
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
			dd: 'Three routed experts plus an always-on shared one, each a <code>U·diag(s)·Vᵀ</code> factorization. In the artifact the factors are 2-bit ternary codes (four per byte) with their scale in the header; the engine unpacks them to fp32 at load.',
		},
		{
			dt: 'Engram memory',
			dd: "25,000 rows per order over byte n-grams of orders [2, 3, 4] — FNV-1a hashed, 32 wide, gated into the memory branch behind the hard floor <code>min(w_mem, lam_max)</code>. 33% of the model's parameters, and a lookup, not a matmul.",
		},
	],
	verified:
		'Verified against the Python reference by <code>tests/test_demo_engine.py</code>: the engine\'s logits agree with <code>decode._step_logits</code> to a max |Δlogit| of <strong>1.058e-6</strong> on <code>small</code> (tolerance 1e-4), with an identical greedy byte at every prefix.',
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
	outLabel: 'output',
	notHeading: 'what this is not',
	notRows: [
		{ name: 'the 9.5M small polygon', delta: 'an A/B instrument' },
		{ name: 'a chatbot', delta: 'no SFT, no chat template' },
		{ name: 'the top-k dropless MoE path', delta: 'moe_topk = 0, the dense blend' },
		{ name: 'a ternary GPU kernel', delta: 'codes unpack to fp32 at load' },
		{ name: 'bf16 compute', delta: 'the engine is the fp32 arm' },
		{ name: 'a fast path', delta: 'one thread, ~30M MAC/byte' },
	],
	notProse:
		"This is the model the project's A/B protocol is run on, not a release. It predicts the next byte; it has never been instruction-tuned, and nothing here is a claim about language quality. What the page does demonstrate is the architecture: the arithmetic the Python trainer computes is the arithmetic this engine computes — the gate above is that statement, and it is checked on every test run. The artifact carries the weights only; the loss, the optimizer and the training step live in the checkpoint, which is a different file for a different job.",
	notNote:
		'The weights are a <code>.dormouse</code> artifact written by <code>dormouse export small --ckpt-name … --engram-dtype fp16</code>; it is 16.3 MB (15.51 MiB), so it is generated rather than committed. <a href="{status}">Status</a> has the measured numbers, including the retractions.',
	footerNote: '©2026 dormouse — every number carries its measurement.',
	docs: 'Docs',
	status: 'Status',
};

export const RU = {
	title: 'dormouse — демо: пресет small прямо в браузере',
	description:
		'Полигон `small` на 9,5 млн параметров, работающий в браузере как обычный JavaScript-движок вывода: байтовый словарь, линейное внимание KDA с переносимым рекуррентным состоянием, низкоранговые эксперты TSCT, n-граммная память Engram.',
	h1: 'демо',
	tagline:
		'Полигон <code>small</code> — 9 476 993 параметра — работает в браузере как обычный JavaScript-движок вывода. Без сервера, без WebGPU, без WASM: тот же артефакт <code>.dormouse</code>, который загружает <code>dormouse serve --export</code>.',
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
			dd: 'Три маршрутизируемых эксперта плюс один общий, всегда включённый; каждый — факторизация <code>U·diag(s)·Vᵀ</code>. В артефакте факторы хранятся 2-битными тернарными кодами (четыре в байте), скейл лежит в заголовке; движок распаковывает их в fp32 при загрузке.',
		},
		{
			dt: 'память Engram',
			dd: '25 000 строк на порядок по байтовым n-граммам порядков [2, 3, 4] — хэш FNV-1a, ширина 32, гейт в плечо памяти через жёсткий пол <code>min(w_mem, lam_max)</code>. 33% параметров модели, и это поиск в таблице, а не матричное умножение.',
		},
	],
	verified:
		'Проверено против питоновского эталона в <code>tests/test_demo_engine.py</code>: логиты движка совпадают с <code>decode._step_logits</code> с максимальной |Δlogit| <strong>1.058e-6</strong> на <code>small</code> (допуск 1e-4), и жадный байт совпадает на каждом префиксе.',
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
	outLabel: 'вывод',
	notHeading: 'чем это не является',
	notRows: [
		{ name: 'полигон small на 9,5M', delta: 'инструмент для A/B' },
		{ name: 'чат-бот', delta: 'нет SFT, нет шаблона чата' },
		{ name: 'dropless-путь top-k MoE', delta: 'moe_topk = 0, плотная смесь' },
		{ name: 'тернарное ядро на GPU', delta: 'коды распаковываются в fp32 при загрузке' },
		{ name: 'вычисления в bf16', delta: 'движок — плечо fp32' },
		{ name: 'быстрый путь', delta: 'один поток, ~30M MAC/байт' },
	],
	notProse:
		'Это модель, на которой в проекте гоняется A/B-протокол, а не релиз. Она предсказывает следующий байт; её никогда не дообучали на инструкциях, и здесь нет никакого утверждения о качестве языка. Что страница действительно показывает — это архитектуру: арифметика, которую считает питоновский тренер, — та же арифметика, которую считает этот движок; гейт выше и есть это утверждение, и он проверяется на каждом прогоне тестов. В артефакте лежат только веса; лосс, оптимизатор и шаг обучения живут в чекпоинте — это другой файл для другой задачи.',
	notNote:
		'Веса — это артефакт <code>.dormouse</code>, который пишет <code>dormouse export small --ckpt-name … --engram-dtype fp16</code>; он весит 16,3 МБ (15,51 MiB), поэтому генерируется, а не коммитится. Измеренные числа, включая отозванные утверждения, — в разделе <a href="{status}">Status</a>.',
	footerNote: '©2026 dormouse — каждое число несёт своё измерение.',
	docs: 'Документация',
	status: 'Статус',
};

export const COPY = { en: EN, ru: RU };
export type Lang = keyof typeof COPY;
