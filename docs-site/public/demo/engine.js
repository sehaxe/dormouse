/**
 * dormouse browser engine — a byte-level LM, run in JavaScript.
 *
 * WHAT THIS IS. A plain ES module with no build step and no dependency that
 * reads the SAME `.dormouse` artifact `dormouse export` writes and that
 * `dormouse decode/serve --export` loads: one file, three readers. It runs the
 * byte-lm decode path (weight-shared loop, KDA gated-delta attention with the
 * recurrent state carried across bytes, TSCT low-rank experts, Engram n-grams)
 * one byte at a time, at any geometry the header describes. Both expert arms
 * are here: the dense blend (`moe_topk = 0`, the `small` polygon) and the
 * DROPLESS top-k dispatch (`moe_topk = k`, every routed token to its top-k,
 * no capacity drop).
 *
 * WHY AN ENGINE AND NOT AN OFF-THE-SHELF RUNTIME. llama.cpp/onnxruntime-web
 * cannot run this model: the chunked gated delta rule, the low-rank TSCT
 * experts, the FNV-hashed Engram tables and the byte-level loop have no
 * equivalent op there. The byte vocabulary (256) removes the tokenizer and the
 * decode path is a simple recurrence, which is what makes writing one
 * tractable. The arithmetic below is transcribed from the Python reference
 * (`src/dormouse/model/{model,loopblock,kda,experts,engram,tsct}.py` and
 * `model/decode.py`'s `recurrent_step`), and the Node gate in
 * `docs-site/demo_test/engine.test.mjs` compares this engine's logits against
 * Python's on a tiny model — the claim is "the same arithmetic", not "looks
 * plausible".
 *
 * NUMERICS. fp32 throughout, in `Float32Array`s — the reference path. The
 * artifact stores dense tensors as fp16 and TSCT factors as 2-bit ternary
 * codes + an exact scale. The factors STAY PACKED: each matvec decodes a code
 * to `{0, ±scale}` as it reads it, so no float copy of the factor is ever held
 * (79.4M factors are 19.9 MB of codes at `medium`, not 318 MB of floats). The
 * Engram table likewise stays at its stored rung: an fp16 table is kept as a
 * `Uint16Array` view into the file and only the 32-wide row a step reads is
 * converted. `bf16` in the header is a compute setting of the TRAINED run; the
 * engine is its own fp32 arm and says so.
 *
 * WHAT IT DOES NOT DO, said here rather than on the page's fine print:
 *   - MSA block-sparse attention, AttnRes, LOOM, ByteFlow, DSpark, a
 *     non-`silu` expert activation, a dense (non-TSCT) projection and a
 *     capacity-dropping top-k (`moe_capacity != 0`): all refused BY FIELD
 *     through `engineRefusal` (the same table `model/export.py` stamps into
 *     the header), never silently approximated;
 *   - the LFB selection bias: it is checkpoint state (a `<ckpt>.lfb` sibling),
 *     not a `.dormouse` record. When the header says `moe_lb_coef != 0` the
 *     engine routes on the unbiased logits and `W.stats.lfbBias` says
 *     `absent`; the smallest additive fix is one dense `loop.moe.lfb_bias`
 *     record beside `loop.moe.lfb_counts`;
 *   - `dense_qat`/`factor_qat` projections at run time (the factors are already
 *     packed);
 *   - any batched or windowed forward: one byte per step, one state per
 *     iteration slot. The prefill is one step per prompt byte, exactly like
 *     Python's CPU `recurrent_decode` — there is no chunked kernel here.
 *
 * The exported surface is deliberately wide: `parseHeader`, `engineRefusal`,
 * `unpackTernary`, `rmsNorm`, `kdaStep`, `fnvKeys`, `stepLogits` and
 * `loadWeights` are pure and callable from Node, which is what makes the
 * parity gate possible.
 */

/** The container's magic: `DMSEXP\x01\x00` (see `model/export.py`). */
export const MAGIC = new Uint8Array([0x44, 0x4d, 0x53, 0x45, 0x58, 0x50, 0x01, 0x00]);
export const ALIGN = 8;

const DTYPE_BYTES = { fp32: 4, fp16: 2, bf16: 2, u8: 1, i32: 4, i64: 8, bool: 1 };

/**
 * The architecture fields the engine implements, field → value. Geometry is
 * NOT here: `d_model`, `n_heads`, `head_dim`, `d_ffn`, `max_iter`, `rank`,
 * `shared_rank`, `n_experts`, `vocab` and the Engram widths are read from the
 * header, so any validated value runs. This is the JS twin of
 * `model/export.py`'s `ENGINE_ARMS`; the writer stamps the same refusal into
 * the header and `Engine.load` re-checks it here so a hand-made file cannot
 * slip past the header's word alone.
 *
 * `moe_topk` is separate because both arms are implemented (`0` = the dense
 * blend, `k` = dropless top-k); what the engine cannot run is the
 * CAPACITY-dropping variant, whose drop set depends on the window's
 * composition and is therefore not a per-byte function.
 */
const ENGINE_ARMS = {
	use_tsct: true,
	use_kda: true,
	use_attnres: false,
	use_byteflow: false,
	use_msa: false,
	loom: false,
	expert_act: 'silu',
	dspark_weight: 0,
};

/** Why this header's `model` is not the engine's architecture, or `null`. */
export function engineRefusal(cfg) {
	if (!cfg || typeof cfg !== 'object') return 'the header carries no `model` dict';
	const bad = [];
	for (const [k, want] of Object.entries(ENGINE_ARMS)) {
		if (cfg[k] !== want) {
			bad.push(`${k}=${JSON.stringify(cfg[k])} (engine implements ${JSON.stringify(want)})`);
		}
	}
	const E = cfg.n_experts;
	const topk = cfg.moe_topk;
	if (!Number.isInteger(topk) || topk < 0 || (Number.isInteger(E) && topk > E)) {
		bad.push(`moe_topk=${JSON.stringify(topk)} (engine implements 0..n_experts=${JSON.stringify(E)})`);
	}
	if (topk > 0 && cfg.moe_capacity !== 0) {
		bad.push(
			`moe_capacity=${JSON.stringify(cfg.moe_capacity)} (engine implements 0 for moe_topk>0: ` +
				'the DROPLESS path, where every assignment reaches its expert)',
		);
	}
	return bad.length ? bad.join('; ') : null;
}

// ---------------------------------------------------------------------------
// the container
// ---------------------------------------------------------------------------

/** Parse the magic + JSON header of a `.dormouse` file. Throws on anything else. */
export function parseHeader(bytes) {
	const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
	if (buf.length < 16) throw new Error(`artifact is ${buf.length} B: shorter than the 16 B magic+header`);
	for (let i = 0; i < MAGIC.length; i++) {
		if (buf[i] !== MAGIC[i]) {
			throw new Error(
				`not a dormouse artifact (magic ${[...buf.slice(0, 8)].map((b) => b.toString(16).padStart(2, '0')).join(' ')}). ` +
					'Cause: a different file, or a different format version. Escape: `dormouse export`.',
			);
		}
	}
	const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
	const headerLen = view.getUint32(8, true);
	if (headerLen <= 0 || 16 + headerLen > buf.length) {
		throw new Error(`header length ${headerLen} does not fit the file (${buf.length} B) — truncated?`);
	}
	const text = new TextDecoder().decode(buf.subarray(16, 16 + headerLen));
	let header;
	try {
		header = JSON.parse(text);
	} catch (e) {
		throw new Error(`the header is not readable JSON: ${e.message}`);
	}
	if (header.format !== 'dormouse') throw new Error(`format=${header.format}: not a dormouse artifact`);
	if (header.version !== 1) throw new Error(`artifact version ${header.version}: this engine reads 1`);
	if (!Array.isArray(header.tensors)) throw new Error('the header carries no `tensors` list');
	return { header, blobBase: 16 + headerLen };
}

/** One record's bytes as a typed array — no copy beyond the widening the caller asks for. */
export function recordBytes(buf, rec, dtype = rec.dtype) {
	const n = DTYPE_BYTES[dtype];
	if (n === undefined) {
		throw new Error(
			`tensor ${rec.name}: dtype ${dtype}, which this engine does not read. ` +
				'Known: fp32, fp16, bf16, u8. Escape: re-export with `dormouse export`.',
		);
	}
	if (rec.offset % ALIGN !== 0) throw new Error(`tensor ${rec.name}: offset ${rec.offset} is not ${ALIGN}-aligned`);
	if (rec.offset + rec.length > buf.length) {
		throw new Error(`tensor ${rec.name} runs past the end of the file — truncated?`);
	}
	return buf.subarray(rec.offset, rec.offset + rec.length);
}

/** `ternarize`'s inverse: 2-bit codes → `{0, ±scale}` (see `model/export.py`). */
export function unpackTernary(codes, scale, count) {
	const out = new Float32Array(count);
	for (let i = 0; i < count; i++) {
		const code = (codes[i >> 2] >> ((i & 3) * 2)) & 3;
		if (code === 3) throw new Error('a 2-bit code is 3, which is not a ternary level {0, +α, −α} — corrupt file?');
		out[i] = code === 0 ? 0 : code === 1 ? scale : -scale;
	}
	return out;
}

/** fp16 → fp32 (the artifact's dense storage; the math below is all fp32). */
function fp16ToF32(src) {
	const out = new Float32Array(src.byteLength / 2);
	const view = new DataView(src.buffer, src.byteOffset, src.byteLength);
	for (let i = 0; i < out.length; i++) out[i] = view.getFloat16 ? view.getFloat16(i * 2, true) : halfToF32(view.getUint16(i * 2, true));
	return out;
}

/** The IEEE-754 binary16 → binary32 path for engines without `DataView.getFloat16`. */
function halfToF32(h) {
	const sign = h & 0x8000 ? -1 : 1;
	const exp = (h >> 10) & 0x1f;
	const frac = h & 0x3ff;
	if (exp === 0) return sign * frac * 2 ** -24;
	if (exp === 31) return frac ? NaN : sign * Infinity;
	return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

/** The artifact writes little-endian; a big-endian host swaps its 16-bit words. */
const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

/**
 * A record's fp16 bytes as a `Uint16Array` VIEW (no widening, no copy).
 *
 * The Engram table is the one tensor whose full width is never needed: the
 * step reads 32 adjacent values per order, so keeping the file's words and
 * converting only those is 100.7 MB instead of 201 MB at `medium`. A record
 * whose bytes are not 2-aligned (the writer aligns every blob to 8, so this is
 * a corrupt file, not a layout this engine writes) falls back to a copy so a
 * hand-made container still reads.
 */
function fp16Words(src) {
	if (src.byteLength % 2 !== 0) throw new Error(`an fp16 record of ${src.byteLength} B is not a whole number of values`);
	if ((src.byteOffset & 1) === 0) return new Uint16Array(src.buffer, src.byteOffset, src.byteLength / 2);
	const out = new Uint16Array(src.byteLength / 2);
	const view = new DataView(src.buffer, src.byteOffset, src.byteLength);
	for (let i = 0; i < out.length; i++) out[i] = view.getUint16(i * 2, true);
	return out;
}

/** One stored fp16 word → fp32, with the host's endianness handled. */
function storedHalf(words, i) {
	const h = words[i];
	return halfToF32(LITTLE_ENDIAN ? h : ((h >> 8) | ((h & 0xff) << 8)));
}

/**
 * The artifact's tensors as the engine holds them — NO widening of factors or
 * the Engram table:
 *
 *   ternary  → `{codes, scale, shape}`: 2-bit codes, decoded inside the matvec
 *   fp16     → `{data}` fp32, except `loop.engram.tables`, kept as `Uint16Array`
 *   fp32/u8/i32/i64 → `{data}` at fp32/widened (small tensors, read densely)
 *
 * This is the memory contract the demo's size table is about: a ternary record
 * costs its packed bytes, never 4 B per element, and the table costs its
 * stored rung, never 2 B more per element. `loadWeights` asserts it.
 */
export function readTensors(buf, header) {
	const out = {};
	for (const rec of header.tensors) {
		if (out[rec.name] !== undefined) throw new Error(`tensor ${rec.name} appears twice`);
		const count = rec.shape.reduce((a, b) => a * b, 1);
		if (rec.dtype === 'ternary') {
			const codes = recordBytes(buf, rec, 'u8');
			if (codes.length * 4 !== count) throw new Error(`tensor ${rec.name}: codes do not cover its shape`);
			// The codes stay as a view into the file. `unpackTernary` is the
			// reference decoder (and the pure test's); the hot path does not use it.
			out[rec.name] = { codes, scale: rec.scale, shape: rec.shape, dtype: 'ternary' };
		} else if (rec.dtype === 'fp16') {
			const raw = recordBytes(buf, rec);
			const data = rec.name.endsWith('engram.tables') ? fp16Words(raw) : fp16ToF32(raw);
			out[rec.name] = { data, shape: rec.shape, dtype: 'fp16' };
		} else if (rec.dtype === 'fp32') {
			const b = recordBytes(buf, rec);
			out[rec.name] = { data: new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)), shape: rec.shape, dtype: 'fp32' };
		} else if (rec.dtype === 'u8' || rec.dtype === 'bool') {
			out[rec.name] = { data: Uint8Array.from(recordBytes(buf, rec)), shape: rec.shape, dtype: rec.dtype };
		} else if (rec.dtype === 'i64' || rec.dtype === 'i32') {
			out[rec.name] = { data: Uint8Array.from(recordBytes(buf, rec)), shape: rec.shape, dtype: rec.dtype };
		} else {
			throw new Error(`tensor ${rec.name}: dtype ${rec.dtype} is not one this engine reads`);
		}
	}
	return out;
}

// ---------------------------------------------------------------------------
// the arithmetic (each function is the Python reference's, one to one)
// ---------------------------------------------------------------------------

/** `nn.RMSNorm`'s forward: `x / sqrt(mean(x²) + eps) * gain`, fp32. */
export function rmsNorm(x, gain, eps) {
	let ss = 0;
	for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
	const inv = 1 / Math.sqrt(ss / x.length + eps);
	const out = new Float32Array(x.length);
	for (let i = 0; i < x.length; i++) out[i] = x[i] * inv * gain[i];
	return out;
}

/**
 * `x @ Wᵀ` for a `[out, in]` row-major W — the `nn.Linear` every arm uses.
 *
 * A ternary factor is decoded HERE, one 2-bit code at a time: the artifact
 * stores it in torch's `[in, out]` order and the engine's view names that
 * `{rows: out, cols: in}`, so element `(i, o)` is code `i*rows + o`. No float
 * copy of the factor exists at any point; `W.codes` is a view into the file.
 */
export function matvec(x, W, out = new Float32Array(W.rows), bias = null) {
	const { rows, cols } = W;
	if (W.codes !== undefined) {
		const { codes, scale } = W;
		if (codes.length * 4 !== rows * cols) {
			throw new Error(`a packed factor holds ${codes.length * 4} codes for a ${rows}x${cols} matvec — corrupt file?`);
		}
		// Output-stationary, like the dense arm: a local fp64 accumulator per
		// output, rounded once — the accumulation order the widened path used,
		// so a factor's logits are bit-for-bit the pre-packing engine's.
		for (let o = 0; o < rows; o++) {
			let acc = 0;
			for (let i = 0; i < cols; i++) {
				const idx = i * rows + o;
				const code = (codes[idx >> 2] >> ((idx & 3) * 2)) & 3;
				if (code === 3) throw new Error('a 2-bit code is 3, which is not a ternary level {0, +α, −α} — corrupt file?');
				if (code !== 0) acc += (code === 1 ? scale : -scale) * x[i];
			}
			out[o] = bias ? acc + bias[o] : acc;
		}
		return out;
	}
	const { w } = W;
	for (let o = 0; o < rows; o++) {
		let acc = 0;
		const base = o * cols;
		for (let i = 0; i < cols; i++) acc += w[base + i] * x[i];
		out[o] = bias ? acc + bias[o] : acc;
	}
	return out;
}

/**
 * `[in, r] → [r, in]`: the artifact keeps torch's layout, and a transposed copy
 * at LOAD is what lets every product in the step be the same row-major matvec.
 */
export function transpose(t) {
	const [rows, cols] = t.shape;
	const out = new Float32Array(rows * cols);
	for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[c * rows + r] = t.data[r * cols + c];
	return { w: out, rows: cols, cols: rows };
}

/** `x @ W` (the `[in, out]` factor), as a `[out, in]` matvec of the transpose. */
function factor(x, t) {
	return matvec(x, t.tr);
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));
/** `logsigmoid` in its stable spelling: `−softplus(−z)`. */
const logsigmoid = (z) => (z > 0 ? -Math.log1p(Math.exp(-z)) : z - Math.log1p(Math.exp(z)));

/**
 * ONE byte of KDA — `kda._recurrent_step`, transcribed.
 *
 * `S = Diag(exp(g))·S + β·k·(v − kᵀ·Diag(exp(g))·S)`, `o = scale·qᵀ·S`, with q/k
 * L2-normalized exactly as the fused kernel does it (`1e-6` inside the rsqrt).
 * `state` is `[n_heads, head_dim, head_dim]` fp32 and is updated IN PLACE —
 * that is the carried state, 196 KB for the `small` preset, and the reason the
 * decode step is O(1) in the context length rather than O(window).
 */
export function kdaStep(state, q, k, v, g, beta, headDim, scale) {
	const heads = beta.length;
	const o = new Float32Array(heads * headDim);
	for (let h = 0; h < heads; h++) {
		const qb = h * headDim;
		const sb = h * headDim * headDim;
		// L2 norm (per head): q/√(Σq²+1e-6), k the same
		let qn = 1e-6;
		let kn = 1e-6;
		for (let i = 0; i < headDim; i++) {
			qn += q[qb + i] * q[qb + i];
			kn += k[qb + i] * k[qb + i];
		}
		qn = 1 / Math.sqrt(qn);
		kn = 1 / Math.sqrt(kn);
		// decay: s[k][j] *= exp(g[k]) — per KEY channel, broadcast over v
		for (let kk = 0; kk < headDim; kk++) {
			const dec = Math.exp(g[qb + kk]);
			const off = sb + kk * headDim;
			for (let j = 0; j < headDim; j++) state[off + j] *= dec;
		}
		// ks[j] = Σ_k k[k]·s[k][j]  (the delta's prediction)
		const ks = new Float32Array(headDim);
		for (let kk = 0; kk < headDim; kk++) {
			const kk_ = k[qb + kk] * kn;
			const off = sb + kk * headDim;
			for (let j = 0; j < headDim; j++) ks[j] += kk_ * state[off + j];
		}
		// s[k][j] += k[k]·β·(v[j] − ks[j])
		const b = beta[h];
		for (let kk = 0; kk < headDim; kk++) {
			const kk_ = k[qb + kk] * kn * b;
			const off = sb + kk * headDim;
			for (let j = 0; j < headDim; j++) state[off + j] += kk_ * (v[qb + j] - ks[j]);
		}
		// o[j] = scale·Σ_k q[k]·s[k][j]
		for (let kk = 0; kk < headDim; kk++) {
			const qq = q[qb + kk] * qn;
			const off = sb + kk * headDim;
			for (let j = 0; j < headDim; j++) o[qb + j] += qq * state[off + j];
		}
		for (let j = 0; j < headDim; j++) o[qb + j] *= scale;
	}
	return o;
}

/**
 * The Engram's FNV-1a keys → slot indices — `engram.raw_keys` + `slots_for`.
 *
 * `window` is the byte window ending at the CURRENT position (length ≤
 * max(order)); order `o` hashes its last `min(o, window.length)` bytes. The
 * 64-bit wrap needs BigInt (JS numbers are doubles): exact, and 12 BigInt ops
 * per byte step are not the cost of anything.
 */
export function fnvKeys(window, orders, slots) {
	const OFFSET = 1469598103934665603n;
	const PRIME = 1099511628211n;
	const MASK64 = (1n << 64n) - 1n;
	const out = new Int32Array(orders.length);
	for (let oi = 0; oi < orders.length; oi++) {
		const start = Math.max(0, window.length - orders[oi]);
		let h = OFFSET;
		for (let i = start; i < window.length; i++) {
			h = ((h ^ BigInt(window[i])) * PRIME) & MASK64;
		}
		out[oi] = Number(h & 0x7fffffffn) & (slots - 1);
	}
	return out;
}

// ---------------------------------------------------------------------------
// the weights
// ---------------------------------------------------------------------------

/** The `small` model's state-dict keys — the engine implements exactly these. */
const KEYS = [
	'embed.weight',
	'loop.norm.weight',
	'loop.controller.weight',
	'loop.attn.q_proj.weight',
	'loop.attn.k_proj.weight',
	'loop.attn.v_proj.weight',
	'loop.attn.g_proj.weight',
	'loop.attn.beta_proj.weight',
	'loop.moe.router.weight',
	'loop.moe.down.weight',
	'loop.moe.lfb_counts',
	'loop.engram.tables',
	'loop.engram.query_proj.weight',
	'loop.engram.key_proj.weight',
	'loop.engram.value_proj.weight',
	'loop.mem_dense.weight',
	'loop.mem_dense.bias',
	'loop.iter_embed',
	'loop.residual_scale',
	'loop.out_proj.u',
	'loop.out_proj.s',
	'loop.out_proj.v',
	'norm.weight',
	'lm_head.u',
	'lm_head.s',
	'lm_head.v',
];

const expertKeys = (i) => [
	`loop.moe.experts.${i}.gate_u`,
	`loop.moe.experts.${i}.gate_s`,
	`loop.moe.experts.${i}.up_u`,
	`loop.moe.experts.${i}.up_s`,
	`loop.moe.experts.${i}.v`,
];

/**
 * Validate the artifact's key set and build the engine's own weight views.
 *
 * The shapes come from the header's `model` dict, so a file whose keys are
 * right but whose geometry is not is refused with the shape named. Everything
 * is a `[out, in]` row-major `{w, rows, cols}` after this call, which is what
 * makes the step below a sequence of matvecs and nothing else.
 */
export function loadWeights(header, tensors) {
	const cfg = header.model;
	const why = engineRefusal(cfg);
	if (why) {
		throw new Error(`the artifact's architecture is not the engine's: ${why}. Escape: re-export with these arms.`);
	}
	const useEngram = cfg.use_engram !== false;
	const want = new Set();
	for (const k of KEYS) if (useEngram || !k.startsWith('loop.engram.')) want.add(k);
	for (let i = 0; i < cfg.n_experts; i++) for (const k of expertKeys(i)) want.add(k);
	for (const k of ['gate_u', 'gate_s', 'up_u', 'up_s', 'v']) want.add(`loop.moe.shared.${k}`);
	// The LFB selection bias is checkpoint state (a `<ckpt>.lfb` sibling), not
	// a `.dormouse` record today — but if a writer adds the one optional record,
	// the engine reads it instead of routing on the unbiased logits.
	const optional = new Set(['loop.moe.lfb_bias']);
	const have = new Set(Object.keys(tensors));
	const missing = [...want].filter((k) => !have.has(k));
	const extra = [...have].filter((k) => !want.has(k) && !optional.has(k));
	if (missing.length || extra.length) {
		throw new Error(
			`the artifact's keys are not this engine's model: ${missing.length} missing ` +
				`(${missing.slice(0, 3).join(', ')}), ${extra.length} unknown (${extra.slice(0, 3).join(', ')}). ` +
				'Cause: a different architecture. Escape: export a config the engine runs.',
		);
	}
	const shape = (name, want0) => {
		const got = tensors[name].shape;
		if (got.join('x') !== want0.join('x')) {
			throw new Error(`tensor ${name}: shape ${got.join('x')}, expected ${want0.join('x')}`);
		}
		return tensors[name];
	};
	const d = cfg.d_model;
	const H = cfg.n_heads;
	const hd = cfg.head_dim;
	const T = cfg.max_iter;
	const E = cfg.n_experts;
	const lin = (name, rows, cols) => ({ w: shape(name, [rows, cols]).data, rows, cols });
	// A TSCT factor arrives in torch's `[in, out]` order. A dense one is
	// transposed to the engine's `[out, in]` matvec; a ternary one keeps its
	// packed codes and only names that same shape (the decoder reads
	// `i*rows + o`), so the load allocates nothing for it.
	const tr = (t) => {
		if (t.dtype !== 'ternary') return transpose(t);
		if (t.shape.length !== 2) throw new Error(`a ternary factor is 2-D, got ${t.shape.join('x')}`);
		return { codes: t.codes, scale: t.scale, rows: t.shape[1], cols: t.shape[0], dtype: 'ternary' };
	};
	// `table_slots(rows)`: the n-gram table is rounded UP to a power of two by
	// the model's own masking read, so the shape check needs the same rule.
	let slots = 1;
	while (slots < cfg.engram_rows) slots <<= 1;
	const table = useEngram
		? shape('loop.engram.tables', [cfg.engram_orders.length, slots, cfg.engram_dim])
		: null;
	const W = {
		cfg,
		header,
		d,
		H,
		hd,
		T,
		E,
		topk: cfg.moe_topk || 0,
		useEngram,
		eps: cfg.norm_eps,
		scale: 1 / Math.sqrt(hd),
		lamMax: cfg.engram_lam_max,
		orders: cfg.engram_orders,
		slots,
		dim: cfg.engram_dim,
		embed: shape('embed.weight', [cfg.vocab, d]).data,
		loopNorm: shape('loop.norm.weight', [d]).data,
		finalNorm: shape('norm.weight', [d]).data,
		controller: lin('loop.controller.weight', 3, 2 * d),
		q: lin('loop.attn.q_proj.weight', d, d),
		k: lin('loop.attn.k_proj.weight', d, d),
		v: lin('loop.attn.v_proj.weight', d, d),
		g: lin('loop.attn.g_proj.weight', d, d),
		beta: lin('loop.attn.beta_proj.weight', H, d),
		router: lin('loop.moe.router.weight', E, d),
		down: lin('loop.moe.down.weight', d, cfg.d_ffn),
		lfbBias: optional.has('loop.moe.lfb_bias') && have.has('loop.moe.lfb_bias') ? shape('loop.moe.lfb_bias', [E]).data : null,
		engram: table ? table.data : null,
		engramFp16: !!(table && table.dtype === 'fp16'),
		engramQuery: useEngram ? lin('loop.engram.query_proj.weight', cfg.engram_dim, d) : null,
		engramKey: useEngram ? lin('loop.engram.key_proj.weight', cfg.engram_dim, cfg.engram_dim) : null,
		engramValue: useEngram ? lin('loop.engram.value_proj.weight', d, cfg.engram_dim) : null,
		memDense: lin('loop.mem_dense.weight', d, d),
		memBias: shape('loop.mem_dense.bias', [d]).data,
		iterEmbed: shape('loop.iter_embed', [T, d]).data,
		residualScale: shape('loop.residual_scale', [1]).data[0],
		experts: [],
		shared: null,
		// TSCT factors arrive as `u [in, r]` / `v [r, out]`; `tr` turns each into
		// the same `[out, in]` matvec as everything else — without widening a
		// ternary factor, which stays 2-bit codes.
		outProjU: tr(shape('loop.out_proj.u', [d, cfg.rank])),
		outProjS: shape('loop.out_proj.s', [cfg.rank]).data,
		outProjV: tr(shape('loop.out_proj.v', [cfg.rank, d])),
		headU: tr(shape('lm_head.u', [d, cfg.rank])),
		headS: shape('lm_head.s', [cfg.rank]).data,
		headV: tr(shape('lm_head.v', [cfg.rank, cfg.vocab])),
	};
	const glu = (prefix, rank) => ({
		gateU: tr(shape(`${prefix}.gate_u`, [d, rank])),
		gateS: shape(`${prefix}.gate_s`, [rank]).data,
		upU: tr(shape(`${prefix}.up_u`, [d, rank])),
		upS: shape(`${prefix}.up_s`, [rank]).data,
		v: tr(shape(`${prefix}.v`, [rank, cfg.d_ffn])),
	});
	for (let i = 0; i < E; i++) {
		W.experts.push({ ...glu(`loop.moe.experts.${i}`, cfg.rank), rank: cfg.rank });
	}
	W.shared = { ...glu('loop.moe.shared', cfg.shared_rank), rank: cfg.shared_rank };
	// The memory contract, COUNTED at the boundary: every ternary record must
	// still be its packed codes (a widened factor would have `.data`, not
	// `.codes`), and the reported budget is what the engine actually holds —
	// the gate `tests/test_demo_engine.py` + `engine.test.mjs` assert on it.
	const factorRecs = header.tensors.filter((r) => r.dtype === 'ternary');
	let factorPackedBytes = 0;
	let factorElements = 0;
	for (const rec of factorRecs) {
		const t = tensors[rec.name];
		const count = rec.shape.reduce((a, b) => a * b, 1);
		if (!t || !(t.codes instanceof Uint8Array) || t.codes.byteLength !== rec.length) {
			throw new Error(
				`tensor ${rec.name}: the engine widened a ternary factor to ${t && t.codes ? t.codes.byteLength : 0} B ` +
					`of codes for ${count} elements. The demo keeps 2-bit codes and decodes in matvec; ` +
					'a float copy is 4 B/element and is the bug this check exists to catch.',
			);
		}
		factorPackedBytes += rec.length;
		factorElements += count;
	}
	const factorScalesBytes = 8 * factorRecs.length; // one exact fp64 number each
	const engramRec = header.tensors.find((r) => r.name === 'loop.engram.tables');
	let tensorBytes = factorPackedBytes + factorScalesBytes;
	for (const t of Object.values(tensors)) {
		if (t.dtype !== 'ternary') tensorBytes += t.data.byteLength;
	}
	// the artifact's own description, for the page's honest labels
	W.stats = {
		engramDtype: header.dtypes ? header.dtypes.engram : '?',
		denseDtype: header.dtypes ? header.dtypes.dense : '?',
		ternaryFactors: factorRecs.length,
		// The factor HEAP, not the file: packed codes + one scale number each.
		factorBytes: factorPackedBytes + factorScalesBytes,
		factorPackedBytes,
		factorElements,
		factorFloatBytes: factorElements * 4,
		// Table bytes as stored; fp16 stays a Uint16Array view at `engramDtype`.
		engramBytes: useEngram && engramRec ? engramRec.length : 0,
		// Every tensor the engine holds, at the rung it holds it (views included).
		tensorBytes,
		preset: header.preset,
		commit: header.commit,
		created: header.created,
		topk: W.topk,
		useEngram,
		// COUNTED, not silent: a routed checkpoint trained with LFB has a
		// selection bias that this container does not carry (see the header).
		lfbBias: W.lfbBias ? 'present' : cfg.moe_lb_coef ? `absent (moe_lb_coef=${cfg.moe_lb_coef})` : null,
	};
	return W;
}

/** `(h @ U) ⊙ s @ V` — `TSCTLinear`'s forward, with the factors pre-transposed. */
function tsct(x, u, s, v, out) {
	const r = s.length;
	const lat = new Float32Array(r);
	matvec(x, u, lat);
	for (let i = 0; i < r; i++) lat[i] *= s[i];
	return matvec(lat, v, out || new Float32Array(v.rows));
}

/** One expert (or the shared one): `silu((x@U)·s) ⊙ ((x@U')·s') @ V`. */
function expertForward(x, ex, f) {
	const gate = matvec(x, ex.gateU);
	const up = matvec(x, ex.upU);
	for (let i = 0; i < ex.rank; i++) {
		const g = gate[i] * ex.gateS[i];
		gate[i] = (g / (1 + Math.exp(-g))) * up[i] * ex.upS[i]; // silu(g) ⊙ u
	}
	return matvec(gate, ex.v, new Float32Array(f));
}

/** The Engram read for ONE position: keys from the byte window, gate on the clamp. */
function engramRead(W, normed, window) {
	const slots = fnvKeys(window, W.orders, W.slots);
	const query = matvec(normed, W.engramQuery);
	const dim = W.dim;
	const mem = new Float32Array(W.d);
	const emb = new Float32Array(dim);
	const inv = 1 / Math.sqrt(dim);
	for (let oi = 0; oi < W.orders.length; oi++) {
		const base = (oi * W.slots + slots[oi]) * dim;
		// The ONLY place the table is read: 32 values, at its stored rung.
		// An fp16 table stays a Uint16Array view; no full-width float copy exists.
		if (W.engramFp16) {
			for (let i = 0; i < dim; i++) emb[i] = storedHalf(W.engram, base + i);
		} else {
			for (let i = 0; i < dim; i++) emb[i] = W.engram[base + i];
		}
		const key = matvec(emb, W.engramKey);
		let s = 0;
		for (let i = 0; i < dim; i++) s += key[i] * query[i];
		s *= inv;
		// sigmoid(sign(s)·sqrt(clamp_min(|s|, 1e-6))) — the reference's CLAMP
		const g = sigmoid(Math.sign(s) * Math.sqrt(Math.max(Math.abs(s), 1e-6)));
		const value = matvec(emb, W.engramValue);
		for (let i = 0; i < W.d; i++) mem[i] += value[i] * g;
	}
	return mem;
}

/** `SparseMoE`'s dense blend (`moe_topk = 0`): softmax over the bank, always the shared expert. */
function moeDense(W, x) {
	const logits = matvec(x, W.router);
	let max = -Infinity;
	for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
	let sum = 0;
	for (let i = 0; i < logits.length; i++) {
		logits[i] = Math.exp(logits[i] - max);
		sum += logits[i];
	}
	const f = W.cfg.d_ffn;
	const weighted = new Float32Array(f);
	for (let e = 0; e < W.E; e++) {
		const y = expertForward(x, W.experts[e], f);
		const p = logits[e] / sum;
		for (let i = 0; i < f; i++) weighted[i] += p * y[i];
	}
	const shared = expertForward(x, W.shared, f);
	for (let i = 0; i < f; i++) weighted[i] += shared[i];
	return matvec(weighted, W.down);
}

/**
 * `SparseMoE`'s DROPLESS top-k arm (`moe_topk = k`, `moe_capacity = 0`) —
 * `experts.py`'s `forward` at `topk > 0` + `_naive_dispatch`, one to one:
 *
 *   `probs  = softmax(logits)` over the WHOLE bank (not over the k selected);
 *   `raised = logits + lfb_bias`, `index = topk(raised, k)` (selection only);
 *   `weight = probs[index] / Σ probs[index]` — the burn `topk_blend`
 *   renormalisation, so a top-1 token's gate is exactly 1.0 and the sparse
 *   and dense arms run at the same output magnitude;
 *   `out = Σ_e expert_e(x)·weight_e + shared(x)`, then ONE shared `down`.
 *
 * This is the *renormalised-global-softmax* form, and the distinction is the
 * whole gate: a softmax over the k logits alone would be a different (sharper)
 * gate on every token. `torch.topk` returns descending scores; the experts
 * are then summed in ASCENDING index order, which is the order
 * `_naive_dispatch`'s `argsort` visits them — the accumulation order the
 * parity test pins.
 */
function moeTopk(W, x) {
	const topk = W.topk;
	const raw = matvec(x, W.router);
	// softmax over all E — the unbiased probabilities the gate reads
	let max = -Infinity;
	for (let i = 0; i < W.E; i++) max = Math.max(max, raw[i]);
	let sum = 0;
	const probs = new Float32Array(W.E);
	for (let i = 0; i < W.E; i++) {
		probs[i] = Math.exp(raw[i] - max);
		sum += probs[i];
	}
	for (let i = 0; i < W.E; i++) probs[i] /= sum;
	// selection reads the RAISED score (LFB: the raw logit + b_e); the gate
	// stays the honest probability from the UNBIASED softmax.
	const score = (e) => raw[e] + (W.lfbBias ? W.lfbBias[e] : 0);
	const order = [];
	for (let e = 0; e < W.E; e++) order.push(e);
	order.sort((a, b) => score(b) - score(a) || a - b); // descending, then lowest index
	const picked = order.slice(0, topk).sort((a, b) => a - b);
	let norm = 0;
	for (const e of picked) norm += probs[e];
	norm = Math.max(norm, 1e-9);
	const f = W.cfg.d_ffn;
	const weighted = new Float32Array(f);
	for (const e of picked) {
		const y = expertForward(x, W.experts[e], f);
		const p = probs[e] / norm;
		for (let i = 0; i < f; i++) weighted[i] += p * y[i];
	}
	const shared = expertForward(x, W.shared, f);
	for (let i = 0; i < f; i++) weighted[i] += shared[i];
	return matvec(weighted, W.down);
}

/** `moe_topk` picks the arm; `moe_topk = 0` is bit-for-bit the arm it always was. */
function moeForward(W, x) {
	return W.topk === 0 ? moeDense(W, x) : moeTopk(W, x);
}

/**
 * ONE byte through the loop block and the head — `decode._step_logits` +
 * `loopblock.recurrent_step`, transcribed.
 *
 * `ctx` is every byte so far (the prompt included), `states` is the array of
 * per-iteration KDA states leaving the previous byte. Returns the next-byte
 * logits `[vocab]` and the states leaving this byte. This is the function the
 * Node gate compares against Python; it is pure apart from `states`, which it
 * updates in place.
 */
export function stepLogits(W, ctx, states) {
	const d = W.d;
	const pos = ctx.length - 1;
	if (pos < 0) throw new Error('stepLogits needs at least one byte of context');
	// The Engram's window is the only arm that needs the bytes; without it the
	// block's memory branch is an exact zero (`loopblock._window`'s
	// `engram is None` arm), so a no-memory artifact never builds a window.
	let window = null;
	if (W.useEngram) {
		const drop = Math.min(W.orders[W.orders.length - 1] - 1, pos);
		window = ctx.subarray ? ctx.subarray(pos + 1 - (drop + 1), pos + 1) : ctx.slice(pos - drop, pos + 1);
	}
	const byte = ctx[pos];
	const h0 = W.embed.subarray(byte * d, (byte + 1) * d);
	let h = new Float32Array(h0);
	const outAcc = new Float32Array(d);
	const twoD = new Float32Array(2 * d);
	for (let it = 0; it < W.T; it++) {
		// 1. the iteration embedding, then the ONE shared pre-norm
		const c = new Float32Array(d);
		for (let i = 0; i < d; i++) c[i] = h[i] + W.iterEmbed[it * d + i];
		const normed = rmsNorm(c, W.loopNorm, W.eps);
		// 2. the controller reads [ctx ; h0]
		twoD.set(c, 0);
		twoD.set(h0, d);
		const raw = matvec(twoD, W.controller);
		const wAttn = sigmoid(raw[0]);
		const wMem = sigmoid(raw[1]);
		const wFfn = sigmoid(raw[2]);
		// 3. KDA
		const q = matvec(normed, W.q);
		const k = matvec(normed, W.k);
		const v = matvec(normed, W.v);
		const gRaw = matvec(normed, W.g);
		const g = new Float32Array(d);
		for (let i = 0; i < d; i++) g[i] = logsigmoid(gRaw[i]);
		const betaRaw = matvec(normed, W.beta);
		const beta = new Float32Array(W.H);
		for (let i = 0; i < W.H; i++) beta[i] = sigmoid(betaRaw[i]);
		const o = kdaStep(states[it], q, k, v, g, beta, W.hd, W.scale);
		const y = new Float32Array(d);
		for (let i = 0; i < d; i++) y[i] = o[i] * wAttn;
		// 4. the memory branch: lam·Engram + (1−lam)·dense, lam = min(w_mem, lam_max)
		//    (or an exact zero when the model carries no Engram — the reference's
		//    `engram is None` arm skips BOTH halves, including mem_dense)
		if (W.useEngram) {
			const mem = engramRead(W, normed, window);
			const dense = matvec(normed, W.memDense, new Float32Array(d), W.memBias);
			const lam = Math.min(wMem, W.lamMax);
			for (let i = 0; i < d; i++) y[i] += mem[i] * lam + dense[i] * (1 - lam);
		}
		// 5. the experts, gated
		const ffn = moeForward(W, normed);
		for (let i = 0; i < d; i++) y[i] += ffn[i] * wFfn;
		// 6. ReZero write (residual_scale is init 1.0 and learned)
		const next = new Float32Array(d);
		for (let i = 0; i < d; i++) next[i] = c[i] + y[i] * W.residualScale;
		h = next;
		// 7. the readout, accumulated; the final state is the AVERAGE
		const step = tsct(h, W.outProjU, W.outProjS, W.outProjV);
		for (let i = 0; i < d; i++) outAcc[i] += step[i];
	}
	const out = new Float32Array(d);
	for (let i = 0; i < d; i++) out[i] = outAcc[i] / W.T;
	const hn = rmsNorm(out, W.finalNorm, W.eps);
	return tsct(hn, W.headU, W.headS, W.headV, new Float32Array(W.cfg.vocab));
}

/** Fresh per-iteration KDA states: `[n_heads, head_dim, head_dim]` fp32 each. */
export function initStates(W) {
	return Array.from({ length: W.T }, () => new Float32Array(W.H * W.hd * W.hd));
}

/** softmax(logits/T) → one byte; `T ≤ 0` is greedy (this engine's own rule). */
export function sampleByte(logits, temp, rand = Math.random) {
	if (temp <= 0) {
		let best = 0;
		for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
		return best;
	}
	let max = -Infinity;
	for (const z of logits) max = Math.max(max, z);
	let sum = 0;
	const probs = new Float64Array(logits.length);
	for (let i = 0; i < logits.length; i++) {
		probs[i] = Math.exp(logits[i] / temp - max / temp);
		sum += probs[i];
	}
	let u = rand() * sum;
	for (let i = 0; i < probs.length; i++) {
		u -= probs[i];
		if (u <= 0) return i;
	}
	return probs.length - 1;
}

// ---------------------------------------------------------------------------
// the loader and the generator
// ---------------------------------------------------------------------------

/**
 * The page-facing engine: fetch → parse → run.
 *
 * `Engine.load(url, {onProgress})` reports the download's bytes as they arrive
 * (a progress bar for a 16 MB artifact, which is the part a reader waits for),
 * then parses the header, unpacks every tensor and validates the architecture —
 * every failure is an `Error` with the cause named, which the page shows
 * instead of a blank output pane.
 */
export class Engine {
	constructor(weights, tensors) {
		this.W = weights;
		this.tensors = tensors;
		this.states = initStates(weights);
		this.steps = 0; // forwards run, prompt included — the prefill's evidence
	}

	static async load(url, { onProgress } = {}) {
		const res = await fetch(url);
		if (!res.ok) {
			throw new Error(`fetch ${url} → HTTP ${res.status} ${res.statusText}`);
		}
		const total = Number(res.headers.get('content-length') || 0);
		let buf;
		if (res.body && res.body.getReader) {
			const reader = res.body.getReader();
			const chunks = [];
			let got = 0;
			for (;;) {
				const { done, value } = await reader.read();
				if (done) break;
				chunks.push(value);
				got += value.length;
				if (onProgress) onProgress(got, total);
			}
			buf = new Uint8Array(got);
			let at = 0;
			for (const c of chunks) {
				buf.set(c, at);
				at += c.length;
			}
		} else {
			buf = new Uint8Array(await res.arrayBuffer());
			if (onProgress) onProgress(buf.length, total || buf.length);
		}
		const { header } = parseHeader(buf);
		// The writer's `engine_refusal` is the header's own word; re-derive it
		// from `model` so a hand-made or older file cannot dodge the check, and
		// so the message names the FIELD rather than "not the small polygon".
		const why = engineRefusal(header.model);
		if (why) {
			throw new Error(
				`this artifact is not the engine's architecture: ${why}` +
					(header.engine_refusal ? ` (the writer's refusal: ${header.engine_refusal})` : '') +
					'. Escape: `dormouse export` a config with the engine\'s arms, and for a routed ' +
					'model a DROPLESS one (`moe_capacity = 0`).',
			);
		}
		const tensors = readTensors(buf, header);
		const weights = loadWeights(header, tensors);
		const engine = new Engine(weights, tensors);
		engine.fileBytes = buf.length;
		engine.stats = { ...weights.stats, bytes: buf.length };
		return engine;
	}

	/** Forget the carried KDA state (a new prompt is a new stream). */
	reset() {
		this.states = initStates(this.W);
		this.steps = 0;
	}

	/**
	 * `n` bytes past `prompt` (a Uint8Array) — Python's `recurrent_decode`, in
	 * the same order: PREFILL one step per prompt byte (the KDA state has to see
	 * every prompt position, not just the last one), then sample, append, step.
	 *
	 * `onByte(byte, index, logits)` fires per generated byte, `onPrefill(i, n)`
	 * per prompt byte (a 24-byte prompt is 24 forwards — the page shows it
	 * rather than hiding it in a hang); the loop yields to the event loop
	 * between steps so the page paints while the engine works. `rand` is
	 * injectable so a test can pin the sampler — the logits are the parity
	 * surface, not the draw.
	 */
	async generate(prompt, n, { temp = 0.8, onByte, onPrefill, shouldStop, rand } = {}) {
		let ctx = Uint8Array.from(prompt);
		const out = [];
		if (ctx.length === 0) throw new Error('the prompt is empty: the first window IS the prompt');
		let logits = null;
		for (let i = 0; i < ctx.length; i++) {
			logits = stepLogits(this.W, ctx.subarray(0, i + 1), this.states);
			this.steps += 1;
			if (onPrefill) onPrefill(i + 1, ctx.length);
			if (onPrefill && i % 4 === 3) await new Promise((r) => setTimeout(r, 0));
		}
		for (let i = 0; i < n; i++) {
			if (shouldStop && shouldStop()) break;
			const byte = sampleByte(logits, temp, rand);
			out.push(byte);
			if (onByte) onByte(byte, i, logits);
			if (i + 1 === n) break;
			const grown = new Uint8Array(ctx.length + 1);
			grown.set(ctx, 0);
			grown[ctx.length] = byte;
			ctx = grown; // the window is the context itself; nothing is trimmed
			logits = stepLogits(this.W, ctx, this.states);
			this.steps += 1;
			await new Promise((r) => setTimeout(r, 0)); // let the page paint
		}
		return out;
	}
}
