# One config seam: resolve, validate, snapshot

Config lived in four representations (46 clap flags with hand-copied defaults, `TrainCfg`, the sectioned `FileConfig` mirror, `--set` strings) with three act-quant parsers and two toggle paths (`--no-kda` vs `--set use_kda=false`) that could silently disagree. Decision, hybrid of design A (minimal interface) and design C (trivial default path): a single resolve seam with fixed merge order (serde defaults, preset TOML, `--set`, typed flags, validate), flags become Option-only so every default is written once in the schema types, every run writes a resolved-config snapshot, and resumed runs diff against it (drift check). The sectioned TOML mirror dies for a flat format.

A macro-generated schema (design B) and an env-var layer were rejected: machinery the simplicity mandate does not buy back. This seam also fixes two real bugs: `--set bf16=true` was silently clobbered by the flag's default, and the train path never called validate.

Amendment (2026-09-21, same day): presets are data, not code. The include_str! registry (preset.rs), its builtin fallback in the loader, and the per-preset ctor methods are deleted; a preset is a flat TOML file found via the loader's search path (cwd configs/, the repo through the compile-time manifest path, the exe dir, ~/.config/dormouse). Schema defaults remain the only configuration baked into the binary. Adding a preset is creating a file.

Implementation notes (deviations found while wiring, 2026-09-21):

- The preset TOMLs live at the repo-root `configs/`, not `crates/dormouse-core/configs/`; `preset.rs` includes them from there. Paths unchanged for users.
- `configs/nano-fused.toml` was the one sectioned file that had to stay loadable by path, so it was rewritten flat like the presets rather than keeping a sectioned reader alive just for it. Same values.
- The schema defaults (the single written copy per field) mirror the `small` preset: `DormouseConfig::default() == builtin("small")`, pinned by a test. Presets stay explicit full tables; the defaults back `Default` (implemented as an empty-table deserialize, so there is literally one written copy) and any keys a hand-written TOML omits.
- The snapshot drift check compares `model.*` and `train.*` values only; `source` (builtin name vs explicit path) is provenance and excluded, so `--preset small` and `--preset configs/small.toml` resume interchangeably. Note that extending `--steps` on a resume is also drift under this rule: keep the original argv and change only what the drift check is told to tolerate.

Amendment (2026-09-23): the drift check exempts the train-section progress keys (`steps`, `log_every`, `ckpt_every`) — extending a finished run is a legitimate resume, and they cannot alter the numerics of already-trained steps. Everything that affects numerics (model, lr, precision, data) stays strict. Surfaced by the first real resume, which the original rule would have rejected for changing `--steps`; the superseded sentence above is the pre-amendment behavior.
