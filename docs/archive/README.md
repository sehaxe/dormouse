# docs/archive/ — history, not truth

**Nothing in this directory may be cited as evidence in an argument.** A file
here is a record of what was believed on a date, kept whole because deleting it
would delete the only copy of a measurement nobody took again. When one of these
contradicts a live document, the live document is right and this one is history.

Git already keeps every version of every file. This directory exists for a
different reason: a reader who opens a document must be able to tell, from where
it sits, whether it is describing today's tree or a tree that has since been cut.

## What earns a place here

1. **A document about a mechanism that was deleted** — the whole-loop `fused/`
   module (ADR-0009), the sparse-attention MSA arm (ADR-0014), the PonderNet era
   (ADR-0013). The retraction is in `AGENTS.md` §3.2; the measurement behind it
   is here, because it was never re-taken.
2. **A document a newer document replaces** — three program plans where one
   survives (`docs/architecture/PLAN.md`), and `tier-a-references.md`, whose
   table is `docs/protocols/ORACLE-TIERS.tsv` now, which is also the file
   `tools/oracle_gate.py` reads.
3. **A process byproduct** — a survey whose verdict is recorded in `AGENTS.md`, an
   audit of a tree at a commit, the question sheets for a research agent that has
   already answered them.
4. **An audit kept whole because its value IS the snapshot** — four documents,
   each a point-in-time reading with every number attached:
   [audit-2026-09-25](audit-2026-09-25.md) ·
   [findings-2026-09-29](findings-2026-09-29.md) ·
   [fused-verification-2026-09-23](fused-verification-2026-09-23.md) ·
   [rmsnorm-kernel-2026-09-30](rmsnorm-kernel-2026-09-30.md).
   These four are the only files here the generated site publishes, under its
   "Archive & retracted" section, for the same reason.

## Layout

The subdirectories mirror the live tree, so a file's kind is still readable from
its path: `research/` (dated notes), `architecture/` (design documents and
superseded plans). The loose files at the top are whole-tree audits.

## What is NOT here

An open finding. A protocol. An ADR. The glossary. A decision sheet awaiting the
owner. A paper transcription. Anything a test, a CI job or a program reads by
path. If one of those is superseded, it is **rewritten to point at what replaced
it** — not moved here.