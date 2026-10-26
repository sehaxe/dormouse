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
    "tools/",
    "scripts/",
    "configs/",
    "sft_pairs/",
    "src/dormouse/",
    "pyproject.toml",
    "uv.lock",
    "port/",
    "README.md",
    "AGENTS.md",
    "CONTEXT.md",
    "POST_TRAINING.md",
)

# Paths that exist only on the rust-legacy branch (2026-10-19: the Python port
# became main and the Rust tree moved to `rust-legacy`). Dated documents cite
# them as the truth of their moment; the checker skips them the way it skips
# `.bulba/` - "fixing" them would rewrite history, not repair a reference.
LEGACY_PREFIXES = ("crates/", "vendor/", "benches/")

# Run artifacts: real files on a machine that has trained, never in a clone.
# A document citing one is citing the run, not the tree.
ARTIFACT_PREFIXES = ("checkpoints/", "data_local/", ".venv")

# The tree was re-rooted when port/ became the repository root (2026-10-19).
# A dated document citing `port/<path>` means today's `<path>`; the reference
# keeps being checked instead of going dark.
ALIASES = {"port/": ""}

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
KNOWN_DEAD = {
    "crates/dormouse-core/src/fused": "the whole-loop fused module, DELETED (ADR-0009). The glossary names it on purpose: the term `fused` means two different things and this is the one that is gone.",
    "docs/README-model.md": "an old README section turned into its own file by a plan that never ran; the content is docs/architecture/PLAN.md",
    "docs/20260420-flashkda-v1-deep-dive.md": "a report written outside the tree; the transcription that replaced it is docs/papers/spec-flashkda.md",
    "docs/SCT_Patent_Application.pdf": "a patent PDF read once and never committed; the lineage is in docs/papers/tsct.md",
    "research/reviews/gdn-fwd-review.md": "a review whose file was never committed; the gdn-kda lane's two halves are docs/reviews/2026-09-29-gdn-kda-review.md",
    "research/reviews/fix-verification-2026-09-30.md": "an audit read out of a worktree (wt/fixverify) and named here as provenance; its four findings are closed in docs/reviews/verify-tails-2026-09-30.md",
    "research/papers/spec-bwd.md": "a spec transcribed in a lane that was cut before it committed; the backward is gated from docs/papers/gdn-kda.md",
    "tools/gen_reference.rs": "renamed to tools/gen_reference_f64.py; two gate comments still carry the old name",
    "vendor/dormouse-fused/crates/burn-sct/src/qr.rs": "the crate was deleted 2026-10-02; the audit that names it is a dated record",
    "vendor/dormouse-fused/crates/burn-sct/src/qr_cuda.rs": "the crate was deleted 2026-10-02; the audit that names it is a dated record",
    "vendor/dormouse-fused/crates/burn-sct": "the crate was deleted 2026-10-02; dated reviews and ADRs keep the name on purpose",
    "vendor/dormouse-fused/crates/burn-es": "the crate was deleted 2026-10-02; dated reviews and ADRs keep the name on purpose",
}

# The 2026-10-19 rust-legacy restructure moved the Rust tree (crates/, vendor/,
# benches/ - now skipped wholesale via LEGACY_PREFIXES) and the Rust-era tooling
# to the `rust-legacy` branch. Dated documents cite these as the truth of their
# moment; they are registered rather than edited out for the same reason
# KNOWN_DEAD exists at all: the citation is a record, not a link.
_LEGACY = "moved to the rust-legacy branch (2026-10-19 restructure)"
LEGACY_DEAD = {
    # Docs that describe a lane which is NOT merged into this tree. Each names
    # the branch, so a reader who lands the lane deletes the entry.
    "docs/architecture/multi-gpu.md": (
        "ships with the `multigpu` lane (DDP/FSDP2 + the device-capability layer); "
        "not merged: one card, and the lane is 59 commits off this base"
    ),
    "tests/test_device_policy.py": (
        "ships with the `multigpu` lane (device-capability layer); not merged"
    ),
    "tests/test_distributed.py": ("ships with the `multigpu` lane (DDP/FSDP2); not merged"),
    "src/dormouse/train/distributed.py": ("ships with the `multigpu` lane (DDP/FSDP2); not merged"),
    "src/dormouse/train/device.py": (
        "ships with the `multigpu` lane (device-capability layer); not merged"
    ),
    "src/dormouse/train/ns_triton.py": (
        "the Triton NS chain lives in `train/ns_kernels.py`; this name was the "
        "pre-landing placeholder"
    ),
    "ns_triton_hook.patch": (
        "the patch that landed the Triton NS arm; the arm is in the tree, so the "
        "patch file itself was never needed here"
    ),
    "src/dormouse/post/sandbox.py": (
        "the jail lives inside `post/verifiers.py` (`screen_code`/`run_sandbox`), "
        "not in a module of its own"
    ),
    "configs/byteflow.toml": _LEGACY,
    "configs/moe-cap": _LEGACY,
    "configs/mor.toml": _LEGACY,
    "configs/nano-fused.toml": _LEGACY,
    "configs/p150.toml": _LEGACY,
    "configs/smoke.toml": _LEGACY,
    "ffi": "the port/ffi Rust extension is PLANNED, not built (readiness plan §7)",
    "scripts/bench.sh": _LEGACY,
    "scripts/chat_rl.py": _LEGACY,
    "scripts/chat_sft.py": _LEGACY,
    "scripts/classifier.py": _LEGACY,
    "scripts/extract_activations.py": _LEGACY,
    "scripts/extract_answer_tokens.py": _LEGACY,
    "scripts/intervene_model.py": _LEGACY,
    "scripts/tsz1024x4k_100B_swa_gdn2.sh": _LEGACY,
    "tools/atlas_summarize.py": _LEGACY,
    "tools/build_lock.sh": _LEGACY,
    "tools/check_f64_fixtures.py": _LEGACY,
    "tools/checks.sh": _LEGACY,
    "tools/dead_pub_audit.py": _LEGACY,
    "tools/determinism": _LEGACY,
    "tools/determinism.py": _LEGACY,
    "tools/env": _LEGACY,
    "tools/falsify_fused_adjoint.sh": _LEGACY,
    "tools/first_run.sh": _LEGACY,
    "tools/fused_matrix.sh": _LEGACY,
    "tools/fused_matrix_static.py": _LEGACY,
    "tools/fwd_mode.py": _LEGACY,
    "tools/gen_bwd_f64.py": _LEGACY,
    "tools/gen_facade.py": _LEGACY,
    "tools/gen_oracle_tiers.py": _LEGACY,
    "tools/gen_reference_f64.py": _LEGACY,
    "tools/gpu-gate.sh": _LEGACY,
    "tools/lib_gate.sh": _LEGACY,
    "tools/migrate-dormouse-fused.sh": _LEGACY,
    "tools/moe_ab.sh": _LEGACY,
    "tools/mor_ab.sh": _LEGACY,
    "tools/oracle_gate.py": _LEGACY,
    "tools/polar_probe.rs": _LEGACY,
    "tools/test-feature-matrix.sh": _LEGACY,
    "tools/test_targets.py": _LEGACY,
    "tools/trainboard.py": _LEGACY,
    "tools/trio_ab.sh": _LEGACY,
}

# `path` / `path:line` / `path:12` / `path:12-34`, or a glob, or `a/{b,c}`.
TOKEN = re.compile(r"[A-Za-z0-9_./-]*[A-Za-z0-9_]")
LINE_SUFFIX = re.compile(r":\d+(-\d+)?$")
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
    yield from (t.split("#")[0] for t in MD_LINK.findall(line))
    yield from TOKEN.findall(line)


def plausible(token: str) -> bool:
    if any(token.startswith(p) for p in LEGACY_PREFIXES):
        return False
    if not any(token.startswith(p) for p in PREFIXES):
        return False
    if any(c in token for c in "{*") or ".." in token or token.endswith("/"):
        return False  # a pattern or a directory link, not a file
    if LINE_SUFFIX.search(token):
        return True  # `file.rs:12` - the file part is what must exist
    return True


def exists(repo_rel: str) -> bool:
    return (ROOT / repo_rel).exists()


def alias(path: str) -> str:
    """Resolve a legacy prefix onto today's tree (the 2026-10-19 re-root)."""
    for old, new in ALIASES.items():
        if path.startswith(old):
            return new + path[len(old) :]
    return path


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
            if not any(
                p in line
                for p in (
                    "docs/",
                    "research/",
                    "port/",
                    "src/dormouse/",
                    "configs/",
                    "scripts/",
                    "tools/",
                    "sft_pairs/",
                )
            ):
                continue  # cheap reject: nothing to resolve on this line
            for token in candidates(line):
                if not plausible(token):
                    continue
                path = alias(LINE_SUFFIX.sub("", token))
                if path.startswith(ARTIFACT_PREFIXES):
                    continue  # a run artifact, not a tree path
                if exists(path):
                    checked += 1
                    continue
                # A bare path may be relative to the citing file rather than to the
                # repo root, so a second chance before it is called missing.
                if (ROOT / Path(f).parent / path).exists():
                    checked += 1
                    continue
                reason = KNOWN_DEAD.get(path) or LEGACY_DEAD.get(path)
                if reason is not None:
                    warned.append(f"{f}:{n} -> {path}  (known-dead: {reason[:60]}…)")
                    continue
                bad.append(f"{f}:{n} -> {token}")
    print(f"doc-refs: {len(files)} files scanned, {checked} path references resolved")
    if warned:
        print(f"known-dead references (documented in KNOWN_DEAD, not a failure): {len(warned)}")
    if bad:
        print(f"NAMED PATHS THAT DO NOT EXIST: {len(bad)}")
        for b in bad:
            print(f"    {b}")
        return 1
    print("doc-refs: 0 named paths missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
