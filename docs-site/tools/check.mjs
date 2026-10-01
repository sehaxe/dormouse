#!/usr/bin/env node
// Check the built site. Run: npm run check  (after npm run build)
//
// TWO THINGS, and only two things:
//
//  1. INTERNAL LINKS. Every href in dist/**/*.html that points inside the site
//     must resolve to a page that was actually built, and to an anchor that
//     actually exists. The ingest step reports what it could not map; this
//     reports what it mapped wrong, which is a different failure (a stale
//     manifest URL, an anchor that a heading rename moved). Anchor misses are
//     WARNINGS, not failures: Astro's heading slugs come from github-slugger
//     and a hand-rolled slugger here would produce false alarms.
//
//  2. FIVE PAGES, FIVE SECTIONS, ASSERTED ON CONTENT. A site that builds and
//     serves empty pages is green to every other check there is. Each case names
//     a module, a URL and a string that can only be there if the ingest step ran
//     on the right canonical file.
//
// Exit 1 on a broken internal link or a missing string. Anchor warnings are
// printed and counted, and do not fail the run.

import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SITE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIST = path.join(SITE, 'dist');

if (!existsSync(DIST)) {
	console.error('check: dist/ does not exist — run `npm run build` first');
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

// ── 1. internal links ────────────────────────────────────────────────────────

const idsOf = new Map(); // dist path -> Set of ids
for (const file of html) {
	const src = readFileSync(file, 'utf8');
	const ids = new Set();
	for (const m of src.matchAll(/\sid="([^"]+)"/g)) ids.add(m[1]);
	idsOf.set(file, ids);
}

const broken = [];
const anchors = [];
let checked = 0;
for (const file of html) {
	const src = readFileSync(file, 'utf8');
	for (const m of src.matchAll(/\s(?:href|src)="([^"]+)"/g)) {
		const url = m[1];
		if (/^(https?:|mailto:|data:|#)/.test(url)) {
			if (url.startsWith('#') && url.length > 1) {
				if (!idsOf.get(file).has(decodeURIComponent(url.slice(1)))) anchors.push(`${path.relative(DIST, file)}${url}`);
			}
			continue;
		}
		const [rel, frag] = url.split('#');
		if (rel.startsWith('/')) {
			checked++;
			// Starlight serves the custom 404 from the built-in /404 route, which
			// emits dist/404.html — there is no /404/index.html.
			const target = path.join(
				DIST,
				rel === '/404/' || rel === '/404' ? '404.html' : rel.endsWith('/') ? `${rel}index.html` : rel,
			);
			if (!existsSync(target)) {
				broken.push(`${path.relative(DIST, file)} -> ${url}`);
				continue;
			}
			if (frag && idsOf.has(target) && !idsOf.get(target).has(decodeURIComponent(frag))) {
				anchors.push(`${path.relative(DIST, file)} -> ${url}`);
			}
		}
	}
}

// ── 2. content spot-checks, one per section ─────────────────────────────────

const CASES = [
	['landing', 'index.html', 'A knowledge base with a retraction policy'],
	['start-here', 'start-here/status/index.html', 'STATUS'],
	['architecture', 'architecture/model/index.html', 'LoopBlock'],
	['protocols', 'protocols/ab-protocol/index.html', 'A/B protocol'],
	['adr', 'adr/0011-loud-failures/index.html', 'P10 Rule 5'],
	['research', 'research/notes/index.html', '2026-10-01'],
	['reviews', 'reviews/ab-wave-2026-10-01/index.html', 'A/B seed wave'],
	['tooling', 'tooling/wt/index.html', 'worktree'],
	['archive', 'archive/broken/index.html', 'BROKEN'],
];

let failed = 0;
console.log('check: %d html files in dist/', html.length);
for (const [section, rel, needle] of CASES) {
	const file = path.join(DIST, rel);
	const ok = existsSync(file) && readFileSync(file, 'utf8').includes(needle);
	console.log('  %s %-13s %s', ok ? 'ok  ' : 'FAIL', section, rel);
	if (!ok) failed++;
}

console.log('internal links checked: %d', checked);
console.log('broken internal links: %d', broken.length);
for (const b of broken.slice(0, 30)) console.log('    %s', b);
console.log('anchor mismatches (warning): %d', anchors.length);
for (const a of anchors.slice(0, 10)) console.log('    %s', a);

const searchIndex = existsSync(path.join(DIST, 'pagefind', 'pagefind.js'));
console.log('pagefind index: %s', searchIndex ? 'built' : 'MISSING');
if (!searchIndex) failed++;

if (failed || broken.length) {
	console.error('check: FAILED (%d content case(s), %d broken link(s))', failed, broken.length);
	process.exit(1);
}
console.log('check: ok');