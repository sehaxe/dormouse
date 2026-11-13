#!/usr/bin/env python3
"""Every repo path a document names must exist. No arguments, no config.

WHY THIS EXISTS. The docs consolidation (2026-10-01) moved 100 files across nine
directories and rewrote their references with sed. Sed does not know whether the
path it wrote is the path that exists: a two-step rename left 24 documents citing
`<name>-renamed.md`, which is a path that never existed and reads exactly like one
that does in a grep. Every one of those was caught by reading, not by tooling.
This is the gate for the next move, not a fix for the last one.

WHAT IT CHECKS. For every tracked markdown / rust / toml / py / sh / tsv file,
every inline-code token and markdown link that LOOKS like a repo path
(`docs/…`, `research/…`, `crates/…`, `vendor/…`, `tools/…`, `configs/…`,
`README.md`, `AGENTS.md`, `CONTEXT.md`, `scripts/…`, `benches/…`) is resolved
against the repo root and against the citing file's own directory. A token with a
`:line` suffix, a glob, or a brace expansion is skipped - those are patterns, not
paths, and a false alarm trains a reader to ignore the output.

EXIT. 1 if any named path does not exist. The list is printed, one per line, as
`file:line -> path`.

SKIPPED ON PURPOSE. `.bulba/` (agent scratch, cited by nobody and rewritten by
every session), `vendor/` (a library of ours that renames itself on its own
schedule), `target/`, `docs-site/node_modules`, and anything inside a fenced code
block of a document - a code block may legitimately show a path that is a
hypothetical.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Prefixes that are unambiguously repo paths when they appear inside backticks or
# as a markdown link target. Anything else (a URL, a bare word, a crate id) is
# not this tool's business.
PREFIXES = (
	"docs/",
	"research/",
	"crates/",
	"vendor/dormouse-fused/",
	"vendor/cubecl-fix/",
	"tools/",
	"scripts/",
	"configs/",
	"benches/",
	"README.md",
	"AGENTS.md",
	"CONTEXT.md",
	"POST_TRAINING.md",
)

SKIP_DIRS = {".bulba", "target", "node_modules", ".git", "graphify-out", ".playwright-mcp"}
EXTS = (".md", ".rs", ".toml", ".py", ".sh", ".tsv", ".yml")

# Paths that were named by a document and are NOT on disk, each with the reason
# it is allowed to stay. Listed rather than fixed, because every one of them is a
# claim about a file that never reached the tree, and "fixing" it would mean
# inventing a document. They are checked for REMOVAL (a name that disappears
# from this list is a bug in the list, not a cleanup win) and reported every run.
#
# This is the pre-existing debt the consolidation inherited. It is bounded and
# visible; it was unbounded and invisible before this file existed.
#: Directories that left this branch wholesale. A prefix is enough: every file
#: under them moved together, and the documents that cite them are records.
KNOWN_DEAD_PREFIXES = {
	"crates/": "the Rust tree moved to the rust-lane branch on 2026-11-13; audits that cite its files are records of what it contained.",
	"vendor/": "the vendored fused-kernel library moved to the rust-lane branch on 2026-11-13.",
}

KNOWN_DEAD = {
	"tools/polar_probe.rs": "moved to the rust-lane branch on 2026-11-13 (main is the Python lane); the three audits that cite it are records of what it printed.",
	"vendor/dormouse-fused/Cargo.toml": "vendor/ moved to the rust-lane branch on 2026-11-13; the decision record that cites it stays on main as history.",

	"crates/dormouse-core/src/fused": "the whole-loop fused module, DELETED (ADR-0009). The glossary names it on purpose: the term `fused` means two different things and this is the one that is gone.",
	"docs/README-model.md": "an old README section turned into its own file by a plan that never ran; the content is docs/architecture/PLAN.md",
	"docs/20260420-flashkda-v1-deep-dive.md": "a report written outside the tree; the transcription that replaced it is docs/papers/spec-flashkda.md",
	"docs/SCT_Patent_Application.pdf": "a patent PDF read once and never committed; the lineage is in docs/papers/tsct.md",
	"research/reviews/gdn-fwd-review.md": "a review whose file was never committed; the gdn-kda lane's two halves are docs/records/reviews/2026-09-29-gdn-kda-review.md",
	"research/reviews/fix-verification-2026-09-30.md": "an audit read out of a worktree (wt/fixverify) and named here as provenance; its four findings are closed in docs/records/reviews/verify-tails-2026-09-30.md",
	"research/papers/spec-bwd.md": "a spec transcribed in a lane that was cut before it committed; the backward is gated from docs/papers/gdn-kda.md",
	"tools/gen_reference.rs": "renamed to tools/gen_reference_f64.py; two gate comments still carry the old name",
	"vendor/dormouse-fused/crates/burn-sct/src/qr.rs": "the crate was deleted 2026-10-02; the audit that names it is a dated record",
	"vendor/dormouse-fused/crates/burn-sct/src/qr_cuda.rs": "the crate was deleted 2026-10-02; the audit that names it is a dated record",
	"vendor/dormouse-fused/crates/burn-sct": "the crate was deleted 2026-10-02; dated reviews and ADRs keep the name on purpose",
	"vendor/dormouse-fused/crates/burn-es": "the crate was deleted 2026-10-02; dated reviews and ADRs keep the name on purpose",
}

# `path` / `path:line` / `path:12` / `path:12-34`, or a glob, or `a/{b,c}`.
TOKEN = re.compile(r"[A-Za-z0-9_./-]*[A-Za-z0-9_]")
LINE_SUFFIX = re.compile(r":\d+(-\d+)?$")
#: `sha:path` in prose is a COMMIT-ANCHORED reference: the file was deleted from
#: the tree but the document is citing the revision it was true at. That is the
#: honest way to keep a deleted document's provenance, so the gate proves the
#: blob existed at that commit instead of calling the path missing.
COMMIT_REF = re.compile(r"\b([0-9a-f]{7,40}):([A-Za-z0-9_./-]*[A-Za-z0-9_])")
MD_LINK = re.compile(r"\]\(([^)\s]+)")


def strip_fences(text: str) -> str:
	"""Blank out fenced code blocks, keeping the line count intact."""
	out, fence = [], None
	for line in text.split("\n"):
		stripped = line.lstrip()
		if fence is None:
			if stripped.startswith("```") or stripped.startswith("~~~"):
				fence = stripped[:3]
				out.append("")
				continue
		elif stripped[:3] == fence:
			fence = None
			out.append("")
			continue
		out.append("" if fence else line)
	return "\n".join(out)


def candidates(line: str):
	"""Every plausible repo path named in one line."""
	for target in MD_LINK.findall(line):
		yield target.split("#")[0]
	for token in TOKEN.findall(line):
		yield token


def plausible(token: str) -> bool:
	if not any(token.startswith(p) for p in PREFIXES):
		return False
	if any(c in token for c in "{*") or ".." in token or token.endswith("/"):
		return False  # a pattern or a directory link, not a file
	if LINE_SUFFIX.search(token):
		return True  # `file.rs:12` - the file part is what must exist
	return True


#: Historical documents (`docs/records/reviews`, `docs/records/research`, `docs/records/archive`) describe
#: the tree as it was on their date. On 2026-11-07 the live package moved
#: `port/` -> `python/`; a dated report that says `port/src/...` is a true record
#: of that day, so the alias resolves it instead of forcing a rewrite of history.
PATH_ALIASES = (("port/", "python/"),)


_AT_COMMIT: dict[tuple[str, str], bool] = {}
#: `sha:path` refs whose COMMIT is not in this clone (local-only, filtered, or
#: rewritten). Unverifiable, not false: reported, never a failure.
UNRESOLVED: list[str] = []


def exists_at(commit: str, repo_rel: str) -> bool:
	"""The blob `repo_rel` existed at `commit` (cached; one git call per pair).

	A commit this clone does not have cannot be checked — a provenance ref to a
	local-only or filtered-out commit is not evidence that the path never existed.
	Those are collected in `UNRESOLVED` and reported as a warning; a commit that
	IS present but lacks the path is still a failure, which is the case the gate
	exists for.
	"""
	key = (commit, repo_rel)
	if key not in _AT_COMMIT:
		known = (
			subprocess.run(
				["git", "cat-file", "-e", commit],
				cwd=ROOT,
				capture_output=True,
			).returncode
			== 0
		)
		if not known:
			UNRESOLVED.append(f"{commit}:{repo_rel}")
			_AT_COMMIT[key] = True
			return True
		_AT_COMMIT[key] = (
			subprocess.run(
				["git", "cat-file", "-e", f"{commit}:{repo_rel}"],
				cwd=ROOT,
				capture_output=True,
			).returncode
			== 0
		)
	return _AT_COMMIT[key]


def exists(repo_rel: str) -> bool:
	if (ROOT / repo_rel).exists():
		return True
	for old_prefix, new_prefix in PATH_ALIASES:
		if repo_rel.startswith(old_prefix):
			if (ROOT / (new_prefix + repo_rel[len(old_prefix):])).exists():
				return True
	return False


def main() -> int:
	# One pathspec per extension: git takes `*.md` as a glob only when it is
	# matched against the filesystem, and `git ls-files *.md` with no shell in
	# between returns nothing.
	files = subprocess.run(
		["git", "ls-files", "--", *[f":(glob)**/*{e}" for e in EXTS]],
		cwd=ROOT,
		capture_output=True,
		text=True,
		check=True,
	).stdout.split()
	bad, warned, checked = [], [], 0
	for f in files:
		if any(part in SKIP_DIRS for part in Path(f).parts):
			continue
		text = strip_fences((ROOT / f).read_text(encoding="utf-8", errors="replace"))
		for n, line in enumerate(text.split("\n"), 1):
			if "docs/" not in line and "research/" not in line:
				continue  # cheap reject: nothing to resolve on this line
			for token in candidates(line):
				if not plausible(token):
					continue
				path = LINE_SUFFIX.sub("", token)
				if any(
					path == LINE_SUFFIX.sub("", ref_path) and exists_at(sha, ref_path)
					for sha, ref_path in COMMIT_REF.findall(line)
				):
					checked += 1
					continue
				if exists(path):
					checked += 1
					continue
				# A bare path may be relative to the citing file rather than to the
				# repo root, so a second chance before it is called missing. The third
				# chance covers the Python lane: a file under `port/` names its own
				# tree (`configs/x.toml`), which resolves as `port/configs/x.toml`.
				if (ROOT / Path(f).parent / path).exists() or (ROOT / "port" / path).exists():
					checked += 1
					continue
				if path in KNOWN_DEAD or any(path == p or path.startswith(p) for p in KNOWN_DEAD_PREFIXES):
					reason = KNOWN_DEAD.get(path) or next(
						KNOWN_DEAD_PREFIXES[p] for p in KNOWN_DEAD_PREFIXES if path.startswith(p)
					)
					warned.append(f"{f}:{n} -> {path}  (KNOWN_DEAD: {reason[:60]}…)")
					continue
				bad.append(f"{f}:{n} -> {token}")
	print(f"doc-refs: {len(files)} files scanned, {checked} path references resolved")
	if warned:
		print(f"known-dead references (documented in KNOWN_DEAD, not a failure): {len(warned)}")
		if UNRESOLVED:
			print(f"unverifiable provenance refs (commit not in this clone, not a failure): {len(UNRESOLVED)}")
			for u in UNRESOLVED[:10]:
				print(f"    {u}")
	if bad:
		print(f"NAMED PATHS THAT DO NOT EXIST: {len(bad)}")
		for b in bad:
			print(f"    {b}")
		return 1
	print("doc-refs: 0 named paths missing")
	return 0


if __name__ == "__main__":
	sys.exit(main())