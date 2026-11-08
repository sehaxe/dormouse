# Changelog

All notable changes to the dormouse Python package. The version lives in
`pyproject.toml` and is read back through the installed metadata
(`dormouse.__version__`), so a tag, a wheel and `dormouse --version` cannot
disagree.

This file starts with the first releasable packaging of the port
(2026-11-08). Earlier port work and the Rust burn history remain in `git log`
and `docs/`.

## [0.0.1] - 2026-11-08

First release. Not yet published; tag `v0.0.1` when the gates in
[`docs/guides/release.md`](../docs/guides/release.md) pass.

### Packaging and release

- `dormouse.__version__` resolves from the installed distribution metadata;
  `pyproject.toml` stays the single source.
- `dormouse --version` and `dormouse version` print it and exit 0.
- The wheel carries every preset under `dormouse/configs/`; `load_preset` falls
  back to that copy, so `dormouse info` works outside a checkout.
- Added this changelog and the release checklist.

### Training and correctness

- **KDA**: the SDPA fallback normalizes with the recurrence's `1e-6` instead of
  `F.normalize`'s `1e-12` (`6368406`); the missing `g_min` floor is a written
  decision with a gate (`13ea0b6`).
- **Optimizer**: fixed the bias-correction desync hit by the CUDA-graph fallback
  and four tests that could not fail for the right reason (`7f27a83`); a capture
  fallback no longer hands Muon the AdamW learning rate (`ca9b0e0`).
- **Budget**: EMA decay is `0.9999` — a 1 000-step horizon is 0.4% of a
  232k-step run (`86e6198`).
- **Sweep**: 64 KB/step arms at 95M, where the card is dispatch-bound
  (`eb285e3`).
- Fused-arm launch counters now fail their gate when the profiler measured
  nothing (`635e767`, `9865943`, `f6a9739`).
- Dense-form `norm_eps` is gated on the final norm only (`d2fb214`).

### NIAH and long context

- `--allow-beyond-window` makes a length-generalization pass legal and says so
  (`45fc063`).
- The default needle is MID-window: the far end is where the selector was at
  chance (`eae2e2b`).
- Gate D runner and gate C rank probe, dry-runnable before a checkpoint
  (`9490756`).

### Experiments and scripts

- Ladder runner gains `SEED` and `ONLY`; a replication is a shell loop, not an
  edit of the runner (`2ac8c06`).
- Ladder ranking labels MB as MB and prints each arm's delta against control
  (`3d09280`).
- `train_budget.sh` puts the 1.9 GB Chinchilla run one command away (`4aa8c6a`);
  compile-at-live-shape and batch×LR experiments (`1901337`).

### Data

- The refetched mixture is verified by replay: 120.1 MB, every source within
  0.5 pp of its weight (`2b44849`); the corpus was inspected and measured clean
  (`3e56b28`).

### Docs, plan, and site

- The 1M protocol's first artifact was refused at gate C, and that refusal is a
  position finding (`a96cabb`, `01267c9`).
- The living night report, plan updates, and the empty known-red list
  (`228dd35`, `f6a9739`, `3d01cc3`, `8545ce0`).
- CI badge corrected (`b8fd04a`); README audits list and context/domain-model
  corrections (`1f5ead0`, `af46351`, `b208b4b`, `acac4b6`, `37f3be3`,
  `2d50994`, `65955c2`, `877fedb`, `cca58ab`, `42d74b5`, `5c4a724`, `4f2fc94`).
- Site: the 34 MB state claim and the "measured" 1M moved to where the page says
  they belong (`de71fa0`).
