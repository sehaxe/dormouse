#!/usr/bin/env node
// The mermaid gate. Run: npm run check  (as its last step).
//
// The renderer is a markdown-processor hook plus a client-side script, and BOTH
// fail SILENTLY: a fence the hook missed ships as a syntax-highlighted code
// block whose source still reads like a diagram, so the build is green, the
// page looks intentional, and nobody notices there is no diagram there.
//
//   `<pre class="mermaid">`      the hook ran
//   `data-language="mermaid"`    a fence it missed -> a code block
//
// The integration hook fires per fence at BUILD time and logs each one
// ("Sätteri transformed mermaid block in …"), but nothing about a build turns
// that log into a failure. Verified here: with the integration listed BEFORE
// starlight in astro.config.mjs the fences still render (Astro runs every
// integration's config:setup hook, so ordering is not the lever); the case this
// gate exists for is a mermaid version or a starlight processor change that
// stops the hook matching, which is silent in both the build and the page.
//
// Separate from check.mjs because that file belongs to the link gate, and one
// gate failing should not hide the other's result.
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DIST = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'dist');

if (!existsSync(DIST)) {
	console.error('check-mermaid: dist/ does not exist — run `npm run build` first');
	process.exit(1);
}

const html = [];
(function walk(dir) {
	for (const e of readdirSync(dir)) {
		const p = path.join(dir, e);
		if (statSync(p).isDirectory()) walk(p);
		else if (p.endsWith('.html')) html.push(p);
	}
})(DIST);

let rendered = 0;
let missed = 0;
for (const file of html) {
	const src = readFileSync(file, 'utf8');
	rendered += (src.match(/<pre class="mermaid">/g) ?? []).length;
	missed += (src.match(/data-language="mermaid"/g) ?? []).length;
}

console.log('mermaid: %d rendered, %d fence(s) left as code', rendered, missed);
if (!rendered) {
	console.error('  no mermaid fence was rendered — the integration is not running');
	process.exit(1);
}
if (missed) {
	console.error('  a ```mermaid fence shipped as a code block');
	process.exit(1);
}
console.log('check-mermaid: ok');
