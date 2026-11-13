# ADR-0026: an eval window is a separate file, and its provenance is in the number

Status: accepted 2026-11-08. The overlap gate it requires is implemented in
`port/scripts/eval_window.py`; the JSONL eval row still does not carry the file.

## Context

Every BPB depends on which bytes were scored. One audit measured three
contamination classes and retracted a number that was quoted as the headline:

1. **`1.4621` is an eval on a training source.** The point is `best/fwedu.bin`,
   offset 100,000,000, 196,563 scored bytes (45 doc-masked positions). `fwedu`
   is a training source of both live mix plans, and the run's own fwedu cursor
   at 0.25 GB is 109.0-110.0 MB, so the window `[100.000, 100.197)` MB sits
   inside bytes that run read. No JSONL row at `off=100M` exists, so the number
   has no artifact; the window script skipped the gate, the current tree calls it (`refuse_training_overlap` in `eval_window.py`).
2. **`mix_v4/web.bin` → `web_eval.bin`.** 9,322 B of the 196,800-byte window
   (4.74%) is also in a training file, largest run 5,948 B. `doc_sha1_disjoint`
   passed because it hashes 6.3 MB shards, not the articles inside them.
3. **`web_v7` content overlap.** 1.39% of the window is in a source, but the
   fraction actually read grows with budget: 0.093% at 0.25 GB, 0.127% at
   983 MB, 0.761% at 4.2 GB. Small, not zero, and budget-dependent.

The gates are different and both are needed. `eval_train_overlap`
(`train/data.py`) maps the eval span and every train file by `(st_dev, st_ino)`
and refuses by identity — the same file under two paths. `refuse_training_overlap`
in `eval_window.py` compares resolved paths from `train.data` plus every
`mix_plan` source. Neither is content-level, as the path gate's docstring says.

The 2026-11-08 carve answers the structural half: `best/fwedu_train.bin`
(933,743,230 B, shard 0 head) and `best/fwedu_eval.bin` (8,386,155 B, the same
shard's tail from a document separator) — a file no source reads, so the path
gate holds by construction, not by convention.

## Decision

- Every eval number names the window FILE, the byte OFFSET and the BYTES
  scored — plus geometry and the doc-masking policy when they apply. A number
  without them is not comparable and is not quoted.
- A window that lives in a training source is refused. Both gates run;
  `--allow-train-overlap` downgrades the refusal to a number explicitly
  labelled not held out.
- An eval window is a separate FILE, carved at a document boundary, not a range
  of a training file. "Same corpus, later offset" is not held out if the source
  is in the mix.
- A number on a new corpus is a NEW number. It does not inherit the old curve,
  the old window or the old improvement; it starts its own.

## Consequences

- `1.4621` is retracted as held out. It cannot be compared to `1.519`: the
  difference is the file, the offset, the masking and 45 scored bytes, not
  "window difficulty". The efficiency-frontier doc's 0.057-gap-as-difficulty
  explanation is unverified and probably backwards: a training window should be easier, not harder.
- The gates are structural, not content-level: a copy under another name, or an
  article shared by two shards, passes both. The content-level number is
  2,735 B = 1.39% (a lower bound: verbatim ≥64 B); a "no leak" claim from 64-byte probes is a false-negative class, not a measurement.
- Instrumentation gap: `eval_row()` writes `eval_offset` and `eval_bytes` but
  not the eval file path or `doc_boundary_policy`. A JSONL-only reader cannot
  tell a masked eval from a flat one, or two files apart. Until the row carries
  them, provenance lives in the printed eval line and the snapshot's `eval_data`.
- Old numbers are not resurrected: `mix4`/`mix_v7` are gone with the merge, and
  the pre-2026-11-08 v1/v2 numbers are contaminated (ADR-0010). The ladder's
  1.4906 at 120 MB is a new number, not comparable to the old 1.4946 at 250 MB;
  the csa8k 1.900 on 32,768 B at 8K is not comparable to a 1K dense number.

## Evidence

- `docs/records/reviews/2026-11-08-data-eval-integrity.md` §1.1-1.5 (the retraction,
  the 4.74% web hit, the v7 "content vs seen" table, the gates), §2.1-2.2, and
  §4.3 (the JSONL gap).
- `port/scripts/eval_window.py` — `training_paths`, `refuse_training_overlap`
  and the docstring's stated limit.
- `port/src/dormouse/train/data.py::eval_train_overlap`, `eval_window_range` —
  the inode gate and the read span; `port/src/dormouse/train/loop.py::eval_row`
  — the row's keys and the missing file/policy.
- `docs/guides/dataset.md` — the carve and "every number on this data is a new
  number"; `docs/records/reviews/2026-11-08-night.md` §1, §5.
- `docs/records/research/2026-11-08-efficiency-frontier.md:68,157` — the `1.4621` point
  this ADR retracts; `port/scripts/train_csa8k.sh` — the same rule at launch.
