// One source of truth: the repo's own documents. Starlight needs frontmatter, so
// this copies them with a title prepended rather than keeping a second copy that
// drifts. It also builds the archive index from the filesystem, so the 240
// research notes and reviews are findable without being in the sidebar.
import { readFile, writeFile, mkdir, readdir } from 'node:fs/promises';
import { dirname, resolve, basename } from 'node:path';

const here = dirname(new URL(import.meta.url).pathname);
const root = resolve(here, '..');
const repo = resolve(root, '..');

const mirror = [
  ['../docs/HANDBOOK.md', 'src/content/docs/handbook.md', 'The handbook',
   'One document, the whole project — the rule, the recipe, the code, the formats, the evidence.'],
];

for (const [from, to, title, description] of mirror) {
  const body = await readFile(resolve(root, from), 'utf8');
  const stripped = body.replace(/^#\s.*\n+/, '');
  const out = `---\ntitle: ${title}\ndescription: ${description}\n---\n\n${stripped}`;
  const dest = resolve(root, to);
  await mkdir(dirname(dest), { recursive: true });
  await writeFile(dest, out);
  console.log(`content: ${from} → ${to}`);
}

// ---- the archive index ----------------------------------------------------
const folders = [
  ['docs/records/reviews', 'Reviews', 'Hostile reads of a lane, written to attack it. A snapshot of a date, never maintained.'],
  ['docs/records/research', 'Research', 'Measurements and literature notes. Same rule: dated, not maintained.'],
  ['docs/adr', 'Decisions', 'One record per decision, with a status. Retracted decisions stay, marked.'],
];

let md = `---
title: Archive
description: Every review, research note and decision record, by date — kept whole, out of the sidebar.
---

The main pages above are maintained. Everything here is a **record**: it says
what was true on a date, and it is never rewritten. Where an archived note and
the [handbook](/dormouse/docs/handbook/) disagree, the handbook is later.

`;

for (const [dir, title, blurb] of folders) {
  const abs = resolve(repo, dir);
  let files = [];
  try {
    files = (await readdir(abs)).filter((f) => f.endsWith('.md') && f !== 'README.md');
  } catch {
    continue;
  }
  files.sort().reverse();
  md += `## ${title}\n\n${blurb}\n\n`;
  md += `| date | document |\n|---|---|\n`;
  for (const f of files) {
    const name = basename(f, '.md');
    const m = name.match(/^(\d{4}-\d{2}-\d{2})-(.*)$/);
    const date = m ? m[1] : '—';
    md += `| ${date} | [${m ? m[2] : name}](https://github.com/sehaxe/dormouse/blob/main/${dir}/${f}) |\n`;
  }
  md += `\n`;
  console.log(`archive: ${title} — ${files.length} documents`);
}

await writeFile(resolve(root, 'src/content/docs/archive.md'), md);
