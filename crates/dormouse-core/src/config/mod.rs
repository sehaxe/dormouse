//! The configuration seam: a flat TOML schema, the presets, the `--set`
//! overrides and the validator (ADR-0005). Five functions and two types, in
//! the order a run goes through them.
//!
//! | step | what it does | failure |
//! |---|---|---|
//! | [`load_config`] | a preset NAME or an explicit PATH -> [`DormouseConfig`] | `Err` naming every path it tried |
//! | [`parse_overrides`] | `key=value` strings -> [`Override`]s | `Err` on a missing `=` or an empty key |
//! | [`apply_overrides`] | the overrides onto a config | `Err` naming the key, for an unknown one |
//! | [`validate`] | the merged config against every rule the model cannot enforce | `Err` naming the field and the escape |
//!
//! WHAT IS AND IS NOT HERE. No type of its own: [`DormouseConfig`] and
//! [`ActQuant`] are declared in [`schema`] and re-exported, so a preset TOML,
//! `--set` and the model all read one definition. The training layer's own
//! merge - which adds the typed-CLI-flag layer on top of these four - is
//! `dormouse_train::resolve`, one crate over. This is a library, so nothing
//! here reads the network, spawns anything, or keeps state between calls.
//!
//! THE RULES THE SEAM ENFORCES, and why they are here rather than at the use
//! site. Every one of them is a value that computes a DIFFERENT network and
//! says nothing (ADR-0019). `act_group` that does not divide `d_model` fails a
//! reshape in the first forward; `dspark_stride = 0` collapses the draft
//! window to k copies of one cross-entropy; `mhc_streams` that does not divide
//! `d_model` is a panic a long way from the flag. A wrong value here is a run
//! that trains happily and reports a number for a network nobody asked for, so
//! each is a LOUD `Err` at resolve time, naming the escape.
//!
//! COST. Four passes over ~45 scalar fields, plus one TOML parse: microseconds
//! next to the 0.8 s a step costs, and paid once per run.
//!
//! NOT THE SAME THING AS THE CHECKPOINT SNAPSHOT. [`validate`] and the drift
//! check in `dormouse_train::RunCfg::diff_keys` answer different questions -
//! "is this config coherent" and "is this the same run" - and the second one
//! lives with the trainer because the exempt key list is a property of resume.

pub mod loader;
pub mod r#override;
pub mod schema;
pub mod validation;

pub use loader::load_config;
pub use r#override::{apply_overrides, parse_overrides, Override};
pub use schema::{ActQuant, DormouseConfig};
pub use validation::validate;
