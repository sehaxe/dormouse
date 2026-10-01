//! The configuration seam: one flat schema, one validator, one override
//! parser, one loader (ADR-0005).
//!
//! # The shape of the thing
//!
//! * [`schema`] holds [`DormouseConfig`] and its serde defaults. Every default
//!   is written exactly once, in the per-field `#[serde(default = ...)]` fns,
//!   and `Default` derives from them by deserializing an empty table — so
//!   there is no second copy of a default to drift. The defaults are the `small`
//!   preset; `schema`'s test asserts that equality, in both directions.
//! * [`validation`] is the LOUD gate. Every refusal names the field and the
//!   escape, because a silently-clamped config is how two runs of "the same"
//!   config diverge (AGENTS.md §1.1).
//! * [`override`] is `--set key=value`, applied after the TOML so a CLI flag
//!   wins over a preset. It parses to strings and resolves through the same
//!   code paths as deserialization, so `--set` cannot accept a value a TOML
//!   file would have rejected.
//! * [`loader`] finds a preset by name or reads a path. It is the only item in
//!   this crate that touches the filesystem.
//!
//! # Invariants a caller relies on
//!
//! * Flat, not sectioned: a preset TOML is a table of these same field names.
//!   There is no `[model]`/`[train]` mirror to keep in sync.
//! * **Defaults == `small`.** `DormouseConfig::default() == small.toml`, and
//!   that is a test, not a convention.
//! * **Every field has a default.** A preset may name any subset of them.
//! * Fields that change the parameter set (`use_attnres`, `use_situ`,
//!   `aux_fb_weight > 0`, `use_gr`, ...) default OFF precisely so that a
//!   default-constructed model is byte-identical to a build from before the
//!   arm existed — that is what keeps the pure-CE control a valid control.
//!
//! # The seam that is deliberately elsewhere
//!
//! Training knobs that are not model shape (`steps`, `log_every`, `lr`,
//! `ckpt_every`, `eval`, `eval_every`) are NOT here: they belong to
//! `dormouse-train`'s config seam and are exempt from the resume
//! config-equality check (ADR-0005/0021), because extending a run is a legal
//! resume. A field appearing here therefore always changes the network.

pub mod loader;
pub mod schema;
pub mod validation;
pub mod r#override;

/// Read a preset by name (`configs/<name>.toml`) or by explicit path.
/// The only item in this crate that touches the filesystem.
pub use loader::load_config;
/// The flat model schema: every field of [`DormouseConfig`] carries its own
/// serde default, and the default value of the whole struct is the `small`
/// preset.
pub use schema::{ActQuant, DormouseConfig};
/// The LOUD gate: `Err` names the field and the escape. Never clamps silently.
pub use validation::validate;
/// `--set key=value`, applied after the TOML so a CLI flag beats a preset.
/// Resolves through the same parsers as deserialization.
pub use r#override::{apply_overrides, parse_overrides, Override};
