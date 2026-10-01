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
    // Host-RAM n-gram tables: the in-model table is never READ on that path
    // (the rows arrive pre-gathered, `hashed_ids` is None in both the train
    // and the eval forward), so it must not cost VRAM and per-checkpoint
    // bytes. At the shipped budget that is 3 x 524288 x 32 = 50M params /
    // 201 MB of dead weights written into every burnpack. Squeeze it to one
    // row per order; the host table (`--engram-slots`) is the real capacity.
    if train.engram_ram {
        model.engram_rows = 1;
    }
    // Head-wise Muon for Q/K uses the resolved attention geometry.
    train.qk_heads = Some(model.n_heads);
    // 3b. The random-depth arm and the MoR router are two different ways to
    // buy the SAME property (depth robustness): one samples the loop depth
    // per step, the other ranks the iteration slots per position. Applying
    // both silently would make either A/B a lie about what it trained, so the
    // pair is refused HERE - the one seam that sees both, in the foreground,
    // before any GPU work. (`--eval-depths` is unaffected: it truncates the
    // loop for a measurement on a clone, and the MoR arm's k shrinks to the
    // slots that ran.)
    if model.use_mor && train.rand_depth {
        return Err(
            "--rand-depth and use_mor are both depth-robustness mechanisms and cannot \
             be combined: the A/B would measure neither. Run one arm (--preset mor, or \
             --set use_mor=true) with --rand-depth off."
                .into(),
        );
    }
    // 4. Validate the merged config (the train path previously never did).
    dormouse_core::config::validate(&model)?;
    // 4b. `--graph-capture` against the arms that would make the captured
    // window a different computation than the run claims to be doing: one named
    // error each, before any GPU work (`graph::check`).
    crate::graph::check(&RunCfg { source: preset.to_string(), model: model.clone(), train: train.clone() })?;
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
    /// values) and is excluded from the comparison.
    ///
    /// THE RULE (ADR-0021): a resume must reproduce the run, so the exempt
    /// class is exactly the keys that change WHEN we look, never WHAT the
    /// model computes - the progress/cadence keys, because extending a
    /// finished run is a legitimate resume and none of them can alter the
    /// numerics of a step that has already run. Everything else is strict,
    /// including every knob that reaches a forward: there is no
    /// "schedule knobs are not config" exemption, because a knob that
    /// changes the objective is a different experiment wearing the same
    /// step count. `rand_depth` (the loop depth per step) was on the wrong
    /// side of this line - `serde(skip)`, invisible here - and is now in
    /// the snapshot like any other objective knob.
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

    /// ADR-0021, item 8: the snapshot is the ONLY record of what a run is,
    /// and the drift check can only compare what the snapshot carries. A
    /// `#[serde(skip)]` field is a field the drift check is blind to - and
    /// three of them were: `rand_depth` (the loop depth drawn per step, so it
    /// reaches every gradient), `eval_batches`, `eval_depths`. A resume that
    /// flipped one of those was a different experiment with the same step
    /// count and nothing said so.
    ///
    /// ONE test for the whole class, by making the two lists provably equal
    /// rather than by asserting twenty field names: the snapshot must carry
    /// every field the struct has. A skipped field shows up as a deficit
    /// here, whatever its name and whatever it does; adding a field without
    /// adding it to the snapshot (or vice versa) fails the count.
    #[test]
    fn snapshot_carries_every_train_field() {
        // The field list, written out. This is the price of the guarantee and
        // it is paid ONCE: a new TrainCfg field must be added here, which is
        // exactly the moment the author has to ask "does this belong in the
        // snapshot?" - the question the skip hid.
        const FIELDS: [&str; 41] = [
            "steps", "ckpt_every", "log_every", "seq_len", "batch", "lr", "wd",
            "grad_clip", "ckpt_name", "eval_every", "opt", "quant",
            "factors_fallback", "rand_depth", "eval_batches", "eval_depths",
            "retract_every", "retract_iters", "retract_batched", "stress", "stress_lr",
            "stress_every", "engram_ram", "engram_slots", "host_adam_every",
            "warmup", "quant_check", "timers", "memlog", "bf16", "act_quant",
            "act_group", "max_iter", "no_kda", "no_engram", "jepa_weight",
            "dspark_weight", "dspark_k", "jepa_targets", "seed", "graph_capture",
        ];
        // An Option field that is None is omitted by the TOML serializer, so
        // a skipped Option is invisible to the round trip too. Set them.
        let run = resolve("small", &[], TrainCfg {
            quant: Some("fp32".into()),
            bf16: Some(true),
            act_quant: Some(dormouse_core::ActQuant::Int(8)),
            act_group: Some(64),
            max_iter: Some(3),
            jepa_weight: Some(0.1),
            dspark_weight: Some(0.2),
            dspark_k: Some(3),
            jepa_targets: Some(std::path::PathBuf::from("t.bin")),
            ..Default::default()
        })
        .unwrap();
        let table = toml::Value::try_from(&run).unwrap();
        let train = table.get("train").and_then(toml::Value::as_table).unwrap();
        let missing: Vec<&str> = FIELDS.iter().copied().filter(|f| !train.contains_key(*f)).collect();
        assert!(missing.is_empty(), "fields absent from the config snapshot: {missing:?}");
        assert_eq!(
            train.len(),
            FIELDS.len() + 1, // + qk_heads, derived in resolve
            "the train section has {} keys, the field list has {}: a field was added or dropped",
            train.len(),
            FIELDS.len() + 1
        );
        // The three that used to be skipped: flipping each one MUST show up in
        // the drift check, or the snapshot is carrying decoration.
        let stored = RunCfg::from_snapshot(&run.snapshot_toml()).unwrap();
        for (field, flipped) in [
            ("rand_depth", TrainCfg { rand_depth: true, ..run.train.clone() }),
            ("eval_batches", TrainCfg { eval_batches: 7, ..run.train.clone() }),
            ("eval_depths", TrainCfg { eval_depths: true, ..run.train.clone() }),
            ("seed", TrainCfg { seed: run.train.seed + 1, ..run.train.clone() }),
        ] {
            let other = RunCfg { train: flipped, ..stored.clone() };
            assert!(
                other.diff_keys(&stored).contains(&format!("train.{field}")),
                "train.{field} does not reach the drift check - a resume may change it silently"
            );
        }
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

    /// The hashed-memory order list is written in TWO crates - the model
    /// config's `engram_orders` and the data crate's `ORDERS`, which is what
    /// actually hashes - because the trainer's plumbing that would pass the
    /// config into `ByteStream` is mid-edit. They must not drift: if they
    /// do, the model sizes its tables from one list and the data emits
    /// columns for another. This is the one test that sees both.
    #[test]
    fn model_and_data_agree_on_the_engram_orders() {
        let run = resolve("small", &[], TrainCfg::default()).unwrap();
        assert_eq!(
            run.model.engram_orders,
            dormouse_data::ORDERS.to_vec(),
            "DormouseConfig::engram_orders must equal dormouse_data::ORDERS"
        );
        // The order COUNT is what the trainer's [b, t, 3] hash tensor can
        // hold, so a shorter/longer list is a shape error, not a relayout.
        assert_eq!(run.model.engram_orders.len(), 3);
    }

    /// The two depth-robustness arms are mutually exclusive, and the refusal
    /// is a loud Err at resolve time (not a silent interaction): MoR ranks the
    /// iteration slots per position, `--rand-depth` samples the loop depth per
    /// step, and an A/B with both applied measures neither.
    #[test]
    fn mor_and_rand_depth_are_refused_together() {
        let rd = TrainCfg { rand_depth: true, ..Default::default() };
        // Either alone resolves. This line previously asserted that
        // `use_mor=true` TOGETHER with rand_depth resolves, which is the exact
        // combination the guard below refuses - the test contradicted itself
        // and its "either alone" half was meaningless.
        assert!(resolve("small", &[], TrainCfg::default()).is_ok());
        assert!(resolve("mor", &[], TrainCfg::default()).is_ok());
        assert!(resolve("small", &[], rd.clone()).is_ok(), "rand-depth alone must resolve");
        assert!(
            resolve("small", &set("use_mor=true"), TrainCfg::default()).is_ok(),
            "use_mor alone must resolve"
        );
        // Together: refused, and the message names both mechanisms.
        let err = resolve("mor", &[], rd).expect_err("the pair must be refused");
        assert!(err.contains("rand-depth") && err.contains("use_mor"), "{err}");
    }

    /// A preset that switches the arm OFF must still be able to switch it on
    /// with its capacity fields intact: the budget is config, not a property
    /// of the arm being enabled (this is the re-enable of 2026-09-27).
    #[test]
    fn disabled_preset_keeps_the_capacity_budget() {
        let off = resolve("nano-fused", &[], TrainCfg::default()).unwrap();
        assert!(!off.model.use_engram, "nano-fused ships the arm off");
        assert_eq!(off.model.engram_rows, 25_000, "the budget is config, not gated on the arm");
        assert_eq!(off.model.engram_lam_max, 0.5);
        let on = resolve("nano-fused", &set("use_engram=true"), TrainCfg::default()).unwrap();
        assert!(on.model.use_engram, "--set must be able to turn the arm back on");
        assert_eq!(on.model.engram_rows, off.model.engram_rows);
        // And the row budget is overridable, with the floor validated.
        let rows = resolve("small", &set("engram_rows=250000"), TrainCfg::default()).unwrap();
        assert_eq!(rows.model.engram_rows, 250_000);
        assert!(resolve("small", &set("engram_lam_max=0"), TrainCfg::default()).is_err());
        assert!(resolve("small", &set("engram_lam_max=1.5"), TrainCfg::default()).is_err());
        let orders = resolve("small", &set("engram_orders=2,3,5"), TrainCfg::default()).unwrap();
        assert_eq!(orders.model.engram_orders, vec![2, 3, 5]);
    }

    /// `--engram-ram` moves the tables to the host, so the in-model table is
    /// dead weight: 3 x 32_768 x 32 = 3.1M params / 12.6 MB of VRAM and of
    /// every burnpack for a tensor nothing reads. The resolve seam squeezes it
    /// to 1 row per order, which is also a loud signal in the config snapshot
    /// if a resume flips the flag.
    #[test]
    fn engram_ram_drops_the_in_vram_table() {
        let vram = resolve("small", &[], TrainCfg::default()).unwrap();
        assert_eq!(vram.model.engram_rows, 25_000, "in-VRAM: the config IS the capacity");
        let ram = resolve("small", &[], TrainCfg { engram_ram: true, ..Default::default() }).unwrap();
        assert_eq!(ram.model.engram_rows, 1, "host-RAM: the in-model table is never read");
        assert_eq!(ram.model.engram_orders, vram.model.engram_orders, "orders are unchanged");
        assert!(!ram.diff_keys(&vram).is_empty(), "the flag must show up in the drift check");
    }
}
