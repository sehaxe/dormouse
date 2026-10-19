/**
 * The engine's gate: the pure parts pinned, and the LOGITS compared to Python.
 *
 * Run it directly (`node --test docs-site/demo_test/engine.test.mjs`) and the
 * pure-part tests run; the parity tests SKIP, because they need an artifact and
 * a reference logits file. `pytest tests/test_demo_engine.py` is the driver: it
 * exports a tiny model AND the real `small` polygon, runs the Python decode
 * steps, writes the reference, sets `DM_PARITY_ARTIFACT` / `DM_PARITY_JSON` and
 * runs this file — that is the gate that makes this a demonstration of OUR
 * model rather than a plausible-looking toy.
 *
 * The tolerance: max |Δlogit| < 1e-4, measured 3.0e-7 on the tiny fixture and
 * 1.058e-6 on `small` (2026-10-19, this box, fp32 CPU torch vs this engine).
 * Both sides read the SAME artifact bytes (fp16 dense, ternary factors), so the
 * only difference left is fp32 accumulation ORDER — the 1e-4 is ~80x headroom
 * for a different BLAS build, not a licence for a different formula.
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
	Engine,
	MAGIC,
	fnvKeys,
	initStates,
	kdaStep,
	loadWeights,
	parseHeader,
	readTensors,
	rmsNorm,
	sampleByte,
	stepLogits,
	unpackTernary,
} from '../public/demo/engine.js';

/** The gate's stated tolerance, in absolute logits units. */
const TOL = 1e-4;

// ---------------------------------------------------------------------------
// the pure parts — no artifact, no network
// ---------------------------------------------------------------------------

/** A hand-built container: magic + JSON header + one fp16 blob, 8-aligned. */
function synthetic(header, blob) {
	const json = new TextEncoder().encode(JSON.stringify(header));
	const base = 16 + json.length + ((8 - ((16 + json.length) % 8)) % 8);
	const buf = new Uint8Array(base + (blob ? blob.length : 0));
	buf.set(MAGIC, 0);
	new DataView(buf.buffer).setUint32(8, json.length, true);
	buf.set(json, 16);
	if (blob) buf.set(blob, base);
	return buf;
}

test('parseHeader reads the header and refuses anything that is not one', () => {
	const h = { format: 'dormouse', version: 1, tensors: [] };
	const { header, blobBase } = parseHeader(synthetic(h));
	assert.equal(header.format, 'dormouse');
	assert.ok(blobBase >= 16 + JSON.stringify(h).length);
	assert.throws(() => parseHeader(new Uint8Array(32)), /not a dormouse artifact/);
	assert.throws(() => parseHeader(synthetic({ format: 'nope', version: 1, tensors: [] })), /not a dormouse/);
	assert.throws(() => parseHeader(synthetic({ format: 'dormouse', version: 9, tensors: [] })), /version 9/);
});

test('unpackTernary pins the 2-bit codebook', () => {
	// codes are low-first: 1 | 2<<2 | 0<<4 | 0<<6 = 9 → [+α, −α, 0, 0]
	assert.deepEqual([...unpackTernary(Uint8Array.from([9]), 0.625, 4)], [0.625, -0.625, 0, 0]);
	// a code 3 is not a ternary level: refuse rather than decode it to 0
	assert.throws(() => unpackTernary(Uint8Array.from([0b11]), 1, 4), /not a ternary level/);
});

test('rmsNorm is x / sqrt(mean(x²) + eps) · gain', () => {
	const x = Float32Array.from([3, 4, -1]);
	const gain = Float32Array.from([1, 2, 0.5]);
	const got = rmsNorm(x, gain, 1e-3);
	const inv = 1 / Math.sqrt((9 + 16 + 1) / 3 + 1e-3);
	for (let i = 0; i < 3; i++) assert.ok(Math.abs(got[i] - x[i] * inv * gain[i]) < 1e-6, `i=${i}: ${got[i]}`);
});

test('kdaStep is the gated delta rule, on a fixture computed by hand', () => {
	// One head, head_dim 2, state 0, g = 0 (no decay), beta = 1:
	//   S = k ⊗ (v − kᵀS) = [[2, 3], [0, 0]] for k = [1, 0], v = [2, 3]
	//   o = scale · qᵀS = [2, 3] for q = [1, 0]
	// (the L2 norm's 1e-6 guard shrinks both by ~1e-6, hence the 1e-5 bound)
	const state = new Float32Array(4);
	const q = Float32Array.from([1, 0]);
	const k = Float32Array.from([1, 0]);
	const v = Float32Array.from([2, 3]);
	const g = Float32Array.from([0, 0]);
	const beta = Float32Array.from([1]);
	const o = kdaStep(state, q, k, v, g, beta, 2, 1);
	assert.ok(Math.abs(o[0] - 2) < 1e-5 && Math.abs(o[1] - 3) < 1e-5, String([...o]));
	// and the state it leaves IS that outer product
	assert.ok(Math.abs(state[0] - 2) < 1e-5 && Math.abs(state[1] - 3) < 1e-5, String([...state]));
	// a second step with the same k/v now predicts v exactly → the delta is 0
	const o2 = kdaStep(state, q, k, v, g, beta, 2, 1);
	assert.ok(Math.abs(o2[0] - 2) < 1e-5 && Math.abs(o2[1] - 3) < 1e-5, String([...o2]));
});

test('fnvKeys matches the Python FNV-1a digest (engram.py)', () => {
	// pinned from `dormouse.model.engram`: fnv(b'ab') & 0x7fffffff & 32767 = 31756,
	// fnv(b'abc') = 28828, fnv(b'abcd') = 32251 (2026-10-19)
	const w = Uint8Array.from([97, 98, 99, 100]);
	assert.deepEqual([...fnvKeys(w, [2, 3, 4], 32768)], [31756, 28828, 32251]);
	// a window shorter than the order clamps to what it has: `s = max(0, e − order)`
	assert.deepEqual([...fnvKeys(Uint8Array.from([97]), [2, 3, 4], 32768)], [6406, 6406, 6406]);
});

test('sampleByte: greedy is the argmax, temperature samples the softmax', () => {
	const logits = Float32Array.from([0, 5, 1]);
	assert.equal(sampleByte(logits, 0), 1);
	// a rand() of 0.0 always lands in the first bucket, 0.999… in the last
	assert.equal(sampleByte(logits, 1, () => 0), 0);
	assert.equal(sampleByte(logits, 1, () => 0.999999), 2);
	// with one dominating logit the draw is that byte for any rand in (0,1)
	assert.equal(sampleByte(Float32Array.from([-20, 20, -20]), 1, () => 0.5), 1);
});

test('readTensors refuses a dtype the engine does not read, by name', () => {
	const header = {
		format: 'dormouse',
		version: 1,
		tensors: [{ name: 'x', dtype: 'bf16', shape: [2], offset: 24, length: 4 }],
	};
	assert.throws(() => readTensors(synthetic(header), header), /x: dtype bf16/);
});

// ---------------------------------------------------------------------------
// the parity gate — the artifact + the Python logits, injected by pytest
// ---------------------------------------------------------------------------

const artifact = process.env.DM_PARITY_ARTIFACT;
const refFile = process.env.DM_PARITY_JSON;
const skip = artifact && refFile ? false : 'set DM_PARITY_ARTIFACT and DM_PARITY_JSON (pytest tests/test_demo_engine.py)';

test('the engine reproduces the Python decode logits', { skip }, () => {
	const buf = new Uint8Array(readFileSync(artifact));
	const { header } = parseHeader(buf);
	const W = loadWeights(header, readTensors(buf, header));
	const ref = JSON.parse(readFileSync(refFile, 'utf8'));
	const states = initStates(W);
	let worst = 0;
	let worstAt = -1;
	for (let i = 0; i < ref.prompt.length; i++) {
		const ctx = Uint8Array.from(ref.prompt.slice(0, i + 1));
		const logits = stepLogits(W, ctx, states);
		const want = ref.logits[i];
		assert.equal(logits.length, want.length, `vocab at prefix ${i}`);
		let bestGot = 0;
		let bestWant = 0;
		for (let j = 0; j < logits.length; j++) {
			const d = Math.abs(logits[j] - want[j]);
			if (d > worst) {
				worst = d;
				worstAt = i;
			}
			if (logits[j] > logits[bestGot]) bestGot = j;
			if (want[j] > want[bestWant]) bestWant = j;
		}
		assert.equal(bestGot, bestWant, `argmax at prefix ${i} (the greedy byte)`);
	}
	console.log(
		`# parity ${ref.label}: max |Δlogit| ${worst.toExponential(3)} at prefix ${worstAt}, ` +
			`argmax identical at all ${ref.prompt.length} prefixes, tolerance ${TOL}`,
	);
	assert.ok(worst < TOL, `max |Δlogit| ${worst} is over the stated tolerance ${TOL}`);
});

test('the engine loads the artifact it was given (architecture + key set)', { skip }, () => {
	const buf = new Uint8Array(readFileSync(artifact));
	const { header } = parseHeader(buf);
	const W = loadWeights(header, readTensors(buf, header));
	assert.equal(W.d, header.model.d_model);
	assert.equal(W.T, header.model.max_iter);
	assert.equal(W.orders.length, header.model.engram_orders.length);
	// the state is [n_heads, head_dim, head_dim] per iteration slot — the O(1)
	// carry, and the number the page quotes
	const states = initStates(W);
	assert.equal(states.length, header.model.max_iter);
	assert.equal(states[0].length, header.model.n_heads * header.model.head_dim ** 2);
});

test('generate() prefills every prompt byte before it samples', { skip }, async () => {
	// The prefill is not optional: without it the KDA state never read the
	// prompt and the first sampled byte is conditioned on the last byte alone.
	// Greedy makes the byte a pure function of the logits, so the reference's
	// own argmax is the expected value — and `engine.steps` counts the forwards,
	// which is the direct evidence that the whole prompt was walked.
	const buf = new Uint8Array(readFileSync(artifact));
	const { header } = parseHeader(buf);
	const tensors = readTensors(buf, header);
	const W = loadWeights(header, tensors);
	const ref = JSON.parse(readFileSync(refFile, 'utf8'));
	const engine = new Engine(W, tensors);
	const got = [];
	await engine.generate(Uint8Array.from(ref.prompt), 1, {
		temp: 0,
		onByte: (b) => got.push(b),
	});
	const last = ref.logits[ref.prompt.length - 1];
	let want = 0;
	for (let i = 1; i < last.length; i++) if (last[i] > last[want]) want = i;
	assert.equal(got[0], want, 'the first generated byte is the reference argmax');
	// one forward per prompt byte, and none for the byte just sampled: without
	// the prefill this would be 1 (the last prompt position alone)
	assert.equal(engine.steps, ref.prompt.length, 'the prefill walked every prompt byte');
});

test('Engine.load is a fetch of the same file the pure path reads', { skip: true }, () => {
	// Network-dependent by construction: exercised by the page, not by CI.
	assert.ok(Engine && typeof Engine.load === 'function');
});
