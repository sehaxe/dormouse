#!/usr/bin/env node
// Ingest: build src/content/docs/ from the canonical markdown in this repo.
//
// WHAT THIS IS. The site is a VIEW. Every page it serves is generated here from
// a file that already lives in the repo at its canonical path — README.md,
// docs/, research/, AGENTS.md — and is left exactly where it is. Nothing is
// copied into a second place that could go stale, nothing is moved, nothing is
// deleted.
//
// WHAT IT DOES, per page:
//   1. reads the canonical file (or a `##`-bounded slice of it),
//   2. cuts the leading `#` title (Starlight renders the frontmatter title as the
//      page's h1; leaving the original would give every page two),
//   3. rewrites link targets — in-repo .md links become site URLs, any other
//      repo-relative path becomes a github.com blob URL, anchors preserved,
//   4. writes frontmatter + body to src/content/docs/<out>.md.
// Then it generates one index page per section and per group, and prints an
// honest link report. Unresolved links are counted and printed, never guessed at.
//
// Run: node tools/ingest.mjs   (npm run ingest; `npm run build` runs it first)

import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import manifest, { authored, branch, dirIndex, globs, pages, sections } from './manifest.mjs';

const SITE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const REPO = path.resolve(SITE, '..');
const OUT = path.join(SITE, 'src', 'content', 'docs');
const BLOB = `${manifest.repo}/blob/${branch}`;
const EDIT = `${manifest.repo}/edit/${branch}`;

// Groups: directories inside a section that get their own generated index page
// (the section index only links; the group index lists the items).
const GROUPS = [
	{
		path: 'start-here/notes',
		title: 'Working notes',
		intro: `Agent working notes, kept because the repo's own documents cite them by
path (a retraction note is \`.bulba/memory.md:15\`, not a paraphrase). They are a
scratch pad, not a reference: they grow at the bottom and are never edited for
style.`,
	},
];

// ── report counters ──────────────────────────────────────────────────────────
const stats = {
	pages: 0,
	sliced: 0,
	rendered: 0,
	linkSite: 0,
	linkBlob: 0,
	linkUnresolved: 0,
	broken: [],
	noHeading: [],
};

const die = (msg) => {
	console.error(`ingest: ${msg}`);
	process.exit(1);
};

// ── text helpers ─────────────────────────────────────────────────────────────

/** Split into fenced / non-fenced segments so transforms never touch code. */
function segments(text) {
	const lines = text.split('\n');
	const out = [];
	let buf = [];
	let fence = null;
	for (const line of lines) {
		const m = /^\s{0,3}(`{3,}|~{3,})/.exec(line);
		if (fence === null && m) {
			if (buf.length) out.push({ code: false, lines: buf });
			fence = m[1][0];
			buf = [line];
			continue;
		}
		buf.push(line);
		if (fence !== null && m && m[1][0] === fence) {
			out.push({ code: true, lines: buf });
			fence = null;
			buf = [];
		}
	}
	if (buf.length) out.push({ code: Boolean(fence), lines: buf });
	return out;
}

const plain = (s) =>
	s
		.replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1') // images
		.replace(/\[([^\]]*)\]\([^)]*\)/g, '$1') // links
		.replace(/[`*_]/g, '')
		.trim();

const yaml = (v) => JSON.stringify(String(v).replace(/\n/g, ' '));

// ── the ingest plan ──────────────────────────────────────────────────────────

/** @type {{src:string,out:string,title?:string,nav?:string,order:number,slice?:object,render?:string}[]} */
const plan = [];
const sectionOf = new Map(sections.map((s) => [s.slug, s]));

for (const p of pages) {
	const section = sectionOf.get(p.out.split('/')[0]);
	if (!section) die(`manifest page "${p.out}" is not inside a declared section`);
	plan.push({ ...p, order: 0 });
}

for (const g of globs) {
	const abs = path.join(REPO, g.dir);
	if (!existsSync(abs)) {
		console.warn(`ingest: glob dir ${g.dir} does not exist, skipped`);
		continue;
	}
	const files = readdirSync(abs)
		.filter((f) => statSync(path.join(abs, f)).isFile())
		.filter((f) => !g.match || g.match.test(f))
		// A file placed in another section by an explicit `pages` entry stays
		// there: one page per canonical file, never two.
		.filter((f) => !plan.some((p) => p.src === `${g.dir}/${f}`));
	// A research note names its own date; newest first is the order a reader
	// wants. Undated notes (a synthesis, a question list) sort after the dated
	// ones rather than interleaving by filename.
	const dateOf = (f) => {
		const m = /^(\d{4}-\d{2}-\d{2})/.exec(f);
		return m ? m[1] : `~${f}`; // '~' sorts after every ISO date
	};
	const sorted = g.order === 'name-desc' ? files.sort().reverse() : files.sort().sort((a, b) => dateOf(a).localeCompare(dateOf(b)));
	sorted.forEach((file, i) => {
		const slug = g.slug ? g.slug(file) : file.replace(/\.md$/, '');
		plan.push({
			src: `${g.dir}/${file}`,
			out: `${g.out}/${slug}`,
			title: typeof g.title === 'function' ? g.title(file) : undefined,
			nav: typeof g.nav === 'function' ? g.nav(file) : undefined,
			render: g.render,
			listLabel: g.listLabel,
			order: 10 + i,
		});
	});
}

// URL for an output path, and the reverse map canonical path -> URL.
//
// Only WHOLE-FILE pages are link targets. A slice is a view of a part of a file
// and must never win a `README.md` link, which is why srcIndex below points the
// file at its primary page instead.
const urlOf = (out) => `/${out === 'index' ? '' : out + '/'}`;
plan.sort((a, b) => a.out.localeCompare(b.out));
const bySrc = new Map(
	plan.filter((p) => !p.slice).map((p) => [p.src, urlOf(p.out)]).concat(Object.entries(manifest.srcIndex ?? {})),
);
if (bySrc.size !== plan.filter((p) => !p.slice).length + Object.keys(manifest.srcIndex ?? {}).length) {
	const seen = new Set();
	const dupes = plan.filter((p) => !p.slice).filter((p) => (seen.has(p.src) ? true : (seen.add(p.src), false)));
	die(`two pages claim the same canonical file: ${dupes.map((p) => p.src).join(', ')}`);
}

// ── read + slice ─────────────────────────────────────────────────────────────

function readSlice(src, slice) {
	const raw = readFileSync(path.join(REPO, src), 'utf8').replace(/^﻿/, '');
	if (!slice) return raw;
	const lines = raw.split('\n');
	const from = lines.findIndex((l) => slice.from.test(l));
	if (from < 0) die(`slice "from" not found in ${src}: ${slice.from}`);
	let to = lines.length;
	if (slice.to) {
		to = lines.findIndex((l, i) => i > from && slice.to.test(l));
		if (to < 0) die(`slice "to" not found in ${src}: ${slice.to}`);
	}
	stats.sliced++;
	const body = lines.slice(from, to).join('\n');
	return body.replace(/\n{3,}/g, '\n\n').trimEnd() + '\n';
}

// ── link rewriting ───────────────────────────────────────────────────────────

const LINK = /\]\(([^)\s]+)((?:\s+"[^"]*")?)\)/g;

function rewriteLinks(body, src) {
	const fromDir = path.posix.dirname(src);
	const one = (line) => {
		let out = '';
		// Split on code spans and rewrite only the text between them: a repo path
		// inside `backticks` is prose that names a file, not a link to a page.
		for (const [i, part] of line.split(/(`[^`]*`)/).entries()) {
			if (i % 2) {
				out += part;
				continue;
			}
			out += part.replace(LINK, (m, target) => {
				if (/^(#|https?:|mailto:|ftp:|tel:|\/|~)/.test(target)) return m;
				if (/[${}<>|]/.test(target)) return m; // a template, not a path
				const hash = target.indexOf('#');
				const rel = hash < 0 ? target : target.slice(0, hash);
				const anchor = hash < 0 ? '' : target.slice(hash);
				if (!rel) return m;
				const abs = path.posix.normalize(path.posix.join(fromDir, rel));
				if (abs.startsWith('..')) return m;
				const hit = bySrc.get(abs);
				if (hit) {
					stats.linkSite++;
					return m.replace(target, hit + anchor);
				}
				const dir = dirIndex[abs.endsWith('/') ? abs : abs + '/'];
				if (dir) {
					stats.linkSite++;
					return m.replace(target, dir + anchor);
				}
				if (existsSync(path.join(REPO, abs))) {
					stats.linkBlob++;
					return m.replace(target, BLOB + '/' + abs + anchor);
				}
				if (rel.includes(':')) return m; // `path:line`, not a link
				stats.linkUnresolved++;
				stats.broken.push(src + ' -> ' + target);
				return m;
			});
		}
		return out;
	};
	return body
		.split('\n')
		.map((line) => (segments(line).some((s) => s.code) ? line : one(line)))
		.join('\n');
}

// ── title / description ──────────────────────────────────────────────────────

function titleAndBody(body, fallback, label) {
	const segs = segments(body);
	let title = null;
	let drop = null; // [segmentIndex, lineIndex]
	segs.forEach((s, si) => {
		if (s.code || title !== null) return;
		s.lines.forEach((l, li) => {
			const m = /^#\s+(.+?)\s*$/.exec(l);
			if (m && title === null) {
				title = plain(m[1]);
				drop = [si, li];
			}
		});
	});
	if (drop) {
		// Remove the source's own h1: Starlight renders the frontmatter title as
		// the page heading, and two h1s on one page is a defect, not a feature.
		segs[drop[0]].lines.splice(drop[1], 1);
		body = segs
			.map((s) => s.lines.join('\n'))
			.join('\n')
			.replace(/^\n+/, '');
	} else if (fallback) {
		// The manifest (or a renderer) named the title; nothing to report.
	} else {
		stats.noHeading.push(label);
	}
	const titleOut = fallback || title || fallbackName(label);
	const desc = description(segs, drop ? drop[1] : -1);
	return { title: titleOut, body, description: desc };
}

function fallbackName(label) {
	const base = path.basename(label).replace(/\.[a-z]+$/, '');
	return base.replace(/[-_]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

function description(segs, skipTo) {
	for (const s of segs) {
		if (s.code) continue;
		const lines = skipTo >= 0 ? s.lines.slice(skipTo + 1) : s.lines;
		const para = [];
		for (const l of lines) {
			const t = l.trim();
			if (!t) {
				if (para.length) break;
				continue;
			}
			if (/^(#|>|\||!\[|-{3,}|={3,})/.test(t) || t.startsWith('<')) continue;
			para.push(t);
		}
		if (para.length) {
			const text = plain(para.join(' ').replace(/<[^>]+>/g, ' ')).replace(/\s+/g, ' ');
			return text.length > 210 ? text.slice(0, 207).trimEnd() + '…' : text;
		}
	}
	return undefined;
}

// ── special renderers ────────────────────────────────────────────────────────

/** A .tsv is data, not prose. 128 rows x 6 columns of sentences: a table would
 *  be unreadable, so each row becomes a block keyed by its file. */
function renderTsv(src) {
	const raw = readFileSync(path.join(REPO, src), 'utf8');
	const comments = raw.split('\n').filter((l) => l.startsWith('#')).map((l) => l.replace(/^#\s?/, ''));
	const rows = raw.split('\n').filter((l) => l.trim() && !l.startsWith('#')).map((l) => l.split('\t'));
	const [head, ...body] = rows;
	const cell = (s) => (s ?? '').replace(/\|/g, '\\|').trim();
	const out = [comments.join('\n\n'), '', `## The register — ${body.length} comparisons`, ''];
	for (const row of body) {
		const rec = Object.fromEntries(head.map((h, i) => [h, cell(row[i])]));
		out.push(`### \`${rec.file}\``, '');
		for (const k of head) {
			if (!k || k === 'file' || !rec[k]) continue;
			out.push(`- **${k}** — ${rec[k]}`);
		}
		out.push('');
	}
	out.push(
		'---',
		'',
		'> This page is a rendering of the register. The artifact `tools/oracle_gate.py` reads is the',
		'> tab-separated [ORACLE-TIERS.tsv](https://github.com/sehaxe/dormouse/blob/main/docs/ORACLE-TIERS.tsv) itself.',
	);
	return { body: out.join('\n').trimEnd() + '\n', toc: false };
}

/** A tool page IS the tool's own header comment — one copy, cannot drift. */
function renderHeader(src) {
	const lines = readFileSync(path.join(REPO, src), 'utf8').split('\n');
	const body = [];
	let fence = null;
	for (let i = lines.findIndex((l) => !l.startsWith('#!')); i < lines.length; i++) {
		const l = lines[i];
		if (fence) {
			if (l.trim() === fence) break;
			body.push(l.replace(/^\s{1,4}/, ''));
			continue;
		}
		const doc = /^\s*(?:"""|''')(.*)$/.exec(l);
		if (doc) {
			fence = l.trim().slice(0, 3);
			if (doc[1]) body.push(doc[1]);
			continue;
		}
		const hash = /^\s*#\s?(.*)$/.exec(l);
		if (hash) {
			body.push(hash[1]);
			continue;
		}
		if (!l.trim()) {
			if (body.length) break; // end of the header block
			continue;
		}
		break;
	}
	// The first line of a tool's header is its title, and Starlight renders the
	// frontmatter title as the page's h1 — so it is taken out of the body rather
	// than printed twice.
	return {
		body: body.slice(1).join('\n').trimEnd() + '\n',
		title: body[0] || path.basename(src).replace(/\.\w+$/, ''),
	};
}

// ── write ────────────────────────────────────────────────────────────────────

function write(out, front, body) {
	const file = path.join(OUT, `${out}.md`);
	mkdirSync(path.dirname(file), { recursive: true });
	writeFileSync(file, `---\n${front}\n---\n\n${body.replace(/\s*$/, '')}\n`, 'utf8');
	stats.pages++;
}

rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });

// authored pages (the landing page) come through untouched.
for (const rel of authored) {
	const file = path.join(SITE, 'content', rel);
	if (!existsSync(file)) die(`authored page missing: content/${rel}`);
	const raw = readFileSync(file, 'utf8').replace(/^﻿/, '');
	const m = /^---\n([\s\S]*?)\n---\n?/.exec(raw);
	if (!m) die(`content/${rel} has no frontmatter`);
	write(rel.replace(/\.md$/, ''), m[1], raw.slice(m[0].length).trimStart());
}

// the pages
for (const p of plan) {
	let body = readSlice(p.src, p.slice);
	const url = urlOf(p.out);
	body = rewriteLinks(body, p.src);

	let toc = true;
	let derivedTitle;
	if (p.render === 'tsv') {
		const r = renderTsv(p.src);
		body = r.body + '\n' + body;
		toc = r.toc;
		stats.rendered++;
	} else if (p.render === 'header') {
		const r = renderHeader(p.src);
		body = r.body + '\n' + body;
		derivedTitle = r.title;
		stats.rendered++;
	}

	const t = titleAndBody(body, p.title ?? derivedTitle, p.src);
	p.resolvedTitle = t.title;
	const front = [
		`title: ${yaml(t.title)}`,
		t.description ? `description: ${yaml(t.description)}` : null,
		`sidebar:\n  label: ${yaml(p.nav ?? t.title)}\n  order: ${p.order}`,
		// "Edit this page" must land on the canonical file. This page is
		// generated and gitignored; a PR against it would be a duplicate.
		`editUrl: ${yaml(EDIT + '/' + p.src)}`,
		`canonical: ${yaml(p.src)}`,
		toc ? null : 'tableOfContents: false',
	]
		.filter(Boolean)
		.join('\n');
	write(p.out, front, t.body);
}

// ── generated index pages ────────────────────────────────────────────────────

const kidsOf = (prefix) => plan.filter((p) => p.out.startsWith(prefix + '/'));

for (const g of GROUPS) {
	const items = kidsOf(g.path);
	const body = [
		g.intro,
		'',
		...items.flatMap((p) => `- [${p.nav ?? p.resolvedTitle ?? p.out.split('/').pop()}](${urlOf(p.out)}) — \`${p.src}\``),
		'',
	].join('\n');
	write(
		g.path + '/index',
		`title: ${yaml(g.title)}\nsidebar:\n  label: ${yaml(g.title)}\n  order: 1`,
		body,
	);
}

for (const s of sections) {
	const items = kidsOf(s.slug);
	const links = items.flatMap((p) => {
		const rel = p.out.slice(s.slug.length + 1);
		const label =
			p.listLabel === 'nav+title' && p.nav && p.resolvedTitle
				? `${p.nav} · ${p.resolvedTitle}`
				: (p.nav ?? p.resolvedTitle ?? rel);
		if (!rel.includes('/')) return [`- [${label}](${urlOf(p.out)}) — \`${p.src}\``];
		const top = rel.split('/')[0];
		if (top === 'index') return [];
		const g = GROUPS.find((q) => q.path === `${s.slug}/${top}`);
		const n = items.filter((q) => q.out.startsWith(`${s.slug}/${top}/`)).length;
		return [`- **[${g ? g.title : top}](/${s.slug}/${top}/)** — ${n} page${n === 1 ? '' : 's'}`];
	});
	const body = [s.intro, '', ...dedupe(links), ''].join('\n');
	write(
		`${s.slug}/index`,
		`title: ${yaml(s.title)}\nsidebar:\n  label: ${yaml(s.nav ?? s.title)}\n  order: ${s.order}`,
		body,
	);
}

function dedupe(xs) {
	const seen = new Set();
	return xs.filter((x) => (seen.has(x) ? false : seen.add(x)));
}

// ── report ───────────────────────────────────────────────────────────────────

const counts = { 'start-here': 0 };
for (const p of plan) counts[p.out.split('/')[0]] = (counts[p.out.split('/')[0]] ?? 0) + 1;

console.log('ingest: %d pages -> src/content/docs/', stats.pages);
for (const s of sections) {
	console.log('  %-14s %3d  (+1 index)', s.slug.padEnd(14), counts[s.slug] ?? 0);
}
console.log(
	'  slices %d · special-rendered %d · no leading heading (title from manifest): %d',
	stats.sliced,
	stats.rendered,
	stats.noHeading.length,
);
console.log('links: %d into the site · %d to github.com · %d unresolved', stats.linkSite, stats.linkBlob, stats.linkUnresolved);
for (const n of stats.noHeading) console.log('  note: no "# title" line in %s — title came from the manifest', n);
if (stats.broken.length) {
	console.log('  BROKEN relative links (target in neither the site nor the repo): %d', stats.broken.length);
	for (const b of stats.broken.slice(0, 25)) console.log('    %s', b);
} else {
	console.log('  BROKEN relative links: 0');
}