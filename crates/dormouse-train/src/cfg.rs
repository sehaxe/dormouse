//! The one config seam (ADR-0005): `resolve` merges, in fixed order,
//! serde defaults -> preset TOML -> `--set` -> typed flags -> validate, and
//! produces a [`RunCfg`] that can be snapshotted and diffed. Zero panics:
//! every failure mode is an `Err` naming the key.

use dormouse_core::DormouseConfig;
use serde::{Deserialize, Serialize};

use crate::TrainCfg;

/// A fully resolved run: where the config came from, the model config, and
/// the train-layer config. Literal-constructible for tests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunCfg {
    /// Provenance (preset name or path as given on the command line). Not
    /// part of the drift comparison; defaults to empty when absent from a
    /// snapshot.
    #[serde(default)]
    pub source: String,
    pub model: DormouseConfig,
    pub train: TrainCfg,
}

/// Merge defaults -> preset -> `--set` -> typed flags, derive `qk_heads`,
/// validate. `set` entries are `key=value` strings (`--set`); unknown keys
/// are a hard error. `train`'s Option model-overrides keep the preset value
/// when `None`; typed flags apply AFTER `--set`, so an explicit flag wins.
pub fn resolve(preset: &str, set: &[String], mut train: TrainCfg) -> Result<RunCfg, String> {
    // 1. serde defaults + preset TOML (file -> search path -> builtin).
    let mut model = dormouse_core::config::load_config(preset)?;
    // 2. --set key=value overrides.
    let ov = dormouse_core::config::parse_overrides(set)?;
    dormouse_core::config::apply_overrides(&mut model, &ov)?;
    // 3. Typed flags: the train layer's model-overrides.
    if let Some(v) = train.bf16 { model.bf16 = v; }
    if let Some(q) = train.act_quant { model.act_quant = Some(q); }
    if let Some(g) = train.act_group { model.act_group = g; }
    if let Some(mi) = train.max_iter { model.max_iter = mi; }
    if train.no_kda { model.use_kda = false; }
    if train.no_engram { model.use_engram = false; }
    if let Some(w) = train.jepa_weight { model.jepa_weight = w; }
    if let Some(w) = train.dspark_weight { model.dspark_weight = w; }
    if let Some(k) = train.dspark_k { model.dspark_k = k; }
    // Head-wise Muon for Q/K uses the resolved attention geometry.
    train.qk_heads = Some(model.n_heads);
    // 4. Validate the merged config (the train path previously never did).
    dormouse_core::config::validate(&model)?;
    Ok(RunCfg { source: preset.to_string(), model, train })
}

impl RunCfg {
    /// Sectioned TOML (`[model]` / `[train]`) of the resolved run, for the
    /// `<ckpt_name>.config.toml` snapshot written next to the checkpoints.
    pub fn snapshot_toml(&self) -> String {
        // Infallible for this struct: plain scalars, strings and Options only.
        toml::to_string_pretty(self).expect("RunCfg serializes to toml")
    }

    /// Inverse of [`RunCfg::snapshot_toml`].
    pub fn from_snapshot(s: &str) -> Result<RunCfg, String> {
        toml::from_str(s).map_err(|e| format!("config snapshot parse: {e}"))
    }

    /// `model.*` / `train.*` keys where `self` and `other` differ. `source`
    /// is provenance (builtin name vs explicit path can produce the same
    /// values) and is excluded from the comparison. The train-section
    /// progress keys (`steps`, `log_every`, `ckpt_every`) are exempt too:
    /// extending a finished run is a legitimate resume, and they cannot
    /// alter the numerics of already-trained steps (2026-09-23, surfaced by
    /// the first real resume hitting the check).
    pub fn diff_keys(&self, other: &RunCfg) -> Vec<String> {
        const PROGRESS_KEYS: [&str; 5] =
            ["steps", "log_every", "ckpt_every", "eval", "eval_every"];
        fn table(r: &RunCfg) -> toml::Table {
            match toml::Value::try_from(r) {
                Ok(toml::Value::Table(t)) => t,
                _ => Default::default(),
            }
        }
        let (a, b) = (table(self), table(other));
        let mut out = Vec::new();
        for section in ["model", "train"] {
            match (a.get(section), b.get(section)) {
                (Some(toml::Value::Table(x)), Some(toml::Value::Table(y))) => {
                    for (k, v) in x {
                        if section == "train" && PROGRESS_KEYS.contains(&k.as_str()) { continue; }
                        if Some(v) != y.get(k) { out.push(format!("{section}.{k}")); }
                    }
                    for k in y.keys() {
                        if section == "train" && PROGRESS_KEYS.contains(&k.as_str()) { continue; }
                        if !x.contains_key(k) { out.push(format!("{section}.{k}")); }
                    }
                }
                _ => out.push(section.to_string()),
            }
        }
        out.sort();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(s: &str) -> Vec<String> { vec![s.to_string()] }

    /// Regression (ADR-0005): `--set bf16=true` with the `--bf16` flag absent
    /// must resolve bf16=true. The old path copied the flag's default into
    /// the model config unconditionally, silently clobbering the --set.
    #[test]
    fn set_survives_absent_flag() {
        let run = resolve("small", &set("bf16=true"), TrainCfg::default()).unwrap();
        assert!(run.model.bf16, "--set bf16=true must survive the absent flag");
    }

    /// Typed flags apply after --set: an explicit flag override still wins.
    #[test]
    fn flag_beats_set() {
        let train = TrainCfg { bf16: Some(false), ..Default::default() };
        let run = resolve("small", &set("bf16=true"), train).unwrap();
        assert!(!run.model.bf16, "typed flag must beat --set (merge order)");
    }

    /// The train layer's Option model-overrides keep the preset value at None.
    #[test]
    fn train_overrides_are_optional() {
        let run = resolve("small", &[], TrainCfg::default()).unwrap();
        assert_eq!(run.model.max_iter, 4);
        let run = resolve("small", &[], TrainCfg { max_iter: Some(3), ..Default::default() }).unwrap();
        assert_eq!(run.model.max_iter, 3);
        assert_eq!(run.model.d_model, 768);
    }

    /// Validation runs on the resolve path: a config only the validator
    /// rejects must come back as Err (the old train path never validated).
    #[test]
    fn validate_runs_on_resolve() {
        let err = resolve("small", &set("d_model=100"), TrainCfg::default()).unwrap_err();
        assert!(err.contains("divisible"), "unexpected error: {err}");
    }

    /// An unknown --set key is a hard error naming the key.
    #[test]
    fn unknown_set_key_names_the_key() {
        let err = resolve("small", &set("nope=1"), TrainCfg::default()).unwrap_err();
        assert!(err.contains("nope"), "unexpected error: {err}");
    }

    /// qk_heads derives from the resolved model geometry, not the CLI.
    #[test]
    fn qk_heads_derived_from_model() {
        let run = resolve("small", &[], TrainCfg::default()).unwrap();
        assert_eq!(run.train.qk_heads, Some(run.model.n_heads));
        let run = resolve("small", &set("n_heads=4"), TrainCfg::default()).unwrap();
        assert_eq!(run.train.qk_heads, Some(4));
    }

    /// Snapshot round trip: from_snapshot(run.snapshot_toml()) == run.
    #[test]
    fn snapshot_round_trip() {
        let run = resolve("nano", &set("max_iter=5"), TrainCfg {
            steps: 7,
            jepa_targets: Some(std::path::PathBuf::from("targets.bin")),
            ..Default::default()
        }).unwrap();
        let back = RunCfg::from_snapshot(&run.snapshot_toml()).unwrap();
        assert_eq!(back, run);
    }

    /// Drift check: a re-resolved config that differs from the stored
    /// snapshot names the changed key (model AND derived train keys).
    #[test]
    fn drift_names_changed_keys() {
        let run = resolve("small", &set("max_iter=6"), TrainCfg::default()).unwrap();
        let stored = RunCfg::from_snapshot(&run.snapshot_toml()).unwrap();
        let changed = resolve("small", &set("max_iter=10"), TrainCfg::default()).unwrap();
        let keys = changed.diff_keys(&stored);
        assert_eq!(keys, vec!["model.max_iter".to_string()]);
        // n_heads drift also shows up in the derived qk_heads.
        let changed = resolve("small", &set("n_heads=4"), TrainCfg::default()).unwrap();
        let keys = changed.diff_keys(&stored);
        assert!(keys.contains(&"model.n_heads".to_string()) && keys.contains(&"train.qk_heads".to_string()), "{keys:?}");
        // Identical resolves have no drift; `source` is provenance, not config.
        let same = resolve("small", &set("max_iter=6"), TrainCfg::default()).unwrap();
        assert!(same.diff_keys(&stored).is_empty());
        let mut alt_source = stored.clone();
        alt_source.source = "configs/small.toml".to_string();
        assert!(stored.diff_keys(&alt_source).is_empty());
        // Progress keys are exempt: extending a finished run is a legal resume.
        let longer = resolve("small", &set("max_iter=6"), TrainCfg {
            steps: 999,
            log_every: 5,
            ckpt_every: 7,
            ..TrainCfg::default()
        }).unwrap();
        assert!(longer.diff_keys(&stored).is_empty(), "{:?}", longer.diff_keys(&stored));
    }

    #[test]
    fn unknown_preset_errors() {
        assert!(resolve("no_such_preset_xyz", &[], TrainCfg::default()).is_err());
    }
}
