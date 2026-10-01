use std::path::{Path, PathBuf};
use super::schema::DormouseConfig;

fn is_explicit(s: &str) -> bool { s.contains('/') || s.contains('\\') || s.ends_with(".toml") }

fn parse_str(s: &str) -> Result<DormouseConfig, String> {
    toml::from_str(s).map_err(|e| format!("toml parse: {e}"))
}

/// `None` = candidate absent (the normal shape of a search path, silent);
/// `Some(Err)` = candidate PRESENT but unusable. COUNTED (ADR-0019 row 26):
/// the caller names the file and the reason instead of silently training on a
/// different file than the first one it found — the same failure the explicit
/// path already reports loudly.
fn try_file(p: &Path) -> Option<Result<DormouseConfig, String>> {
    let s = match std::fs::read_to_string(p) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => return Some(Err(format!("read: {e}"))),
    };
    Some(parse_str(&s))
}

fn candidates(name: &str) -> Vec<PathBuf> {
    let mut v = Vec::new();
    // cwd relative
    v.push(PathBuf::from(format!("configs/{name}.toml")));
    v.push(PathBuf::from(format!("configs/{name}")));
    // CARGO_MANIFEST_DIR at compile time (crate dir -> repo root)
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs").join(format!("{name}.toml"));
    v.push(manifest);
    // exe dir relatives
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            v.push(dir.join(format!("configs/{name}.toml")));
            v.push(dir.join(format!("../configs/{name}.toml")));
            v.push(dir.join(format!("../../configs/{name}.toml")));
            v.push(dir.join(format!("{name}.toml")));
        }
    }
    // ~/.config/dormouse
    if let Ok(home) = std::env::var("HOME") {
        v.push(PathBuf::from(format!("{home}/.config/dormouse/{name}.toml")));
        v.push(PathBuf::from(format!("{home}/.config/dormouse/{name}")));
    }
    v
}

/// Load a preset by name (file search) or an explicit path. Presets are
/// data: plain TOML under configs/ (cwd, the repo via CARGO_MANIFEST_DIR,
/// the exe dir) or ~/.config/dormouse. Schema defaults are the only config
/// baked into the binary (ADR-0005); there is no Rust preset registry.
pub fn load_config(name_or_path: &str) -> Result<DormouseConfig, String> {
    if is_explicit(name_or_path) {
        let p = Path::new(name_or_path);
        let s = std::fs::read_to_string(p).map_err(|e| format!("config file {p:?}: {e}"))?;
        return parse_str(&s);
    }
    let name = name_or_path.trim_end_matches(".toml");
    for p in candidates(name) {
        match try_file(&p) {
            Some(Ok(cfg)) => return Ok(cfg),
            // COUNTED (ADR-0019): a present-but-broken preset is named with its
            // reason before the search moves on to the next candidate.
            Some(Err(e)) => eprintln!(
                "[config] preset candidate {} unusable ({e}); trying next",
                p.display()
            ),
            None => {}
        }
    }
    Err(format!("unknown preset or config not found: {name_or_path:?} (tried configs/, repo configs via manifest path, exe dir, ~/.config/dormouse)"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// --set application helper mirroring the resolve seam's merge (the
    /// train-side seam adds the typed-flag layer on top of this).
    fn apply(name: &str, sets: &[&str]) -> Result<DormouseConfig, String> {
        let mut c = load_config(name)?;
        let raw: Vec<String> = sets.iter().map(|s| s.to_string()).collect();
        let ov = super::super::r#override::parse_overrides(&raw)?;
        super::super::r#override::apply_overrides(&mut c, &ov)?;
        super::super::validation::validate(&c)?;
        Ok(c)
    }

    #[test]
    fn small_preset_file_matches_expectations() {
        let manifest = format!("{}/../../configs/small.toml", env!("CARGO_MANIFEST_DIR"));
        let c = load_config(&manifest).expect("configs/small.toml");
        assert_eq!(c.d_model, 768); assert_eq!(c.n_heads, 12); assert_eq!(c.max_iter, 4);
    }
    #[test]
    fn load_by_name_uses_builtin_or_file() {
        let c = load_config("small").expect("load small");
        assert_eq!(c.d_model, 768);
        let c2 = load_config("nano").expect("load nano");
        assert_eq!(c2.d_model, 512);
    }

    /// nano-fused (loadable by path only) must stay parseable and flat.
    #[test]
    fn nano_fused_file_parses() {
        let manifest = format!("{}/../../configs/nano-fused.toml", env!("CARGO_MANIFEST_DIR"));
        let c = load_config(&manifest).expect("configs/nano-fused.toml");
        assert!(!c.use_kda && !c.use_engram);
        assert_eq!(c.d_model, 512);
    }
    #[test]
    fn load_explicit_path() {
        let manifest_path = format!("{}/../../configs/small.toml", env!("CARGO_MANIFEST_DIR"));
        let c = load_config(&manifest_path).expect("explicit path");
        assert_eq!(c.d_model, 768);
        assert_eq!(c.vocab, 256);
        // also try configs/small.toml relative to repo root if cwd allows
        if std::path::Path::new("configs/small.toml").exists() {
            let c2 = load_config("configs/small.toml").expect("relative explicit");
            assert_eq!(c2.d_model, 768);
        }
    }
    #[test]
    fn set_override_applies() {
        let c = apply("small", &["max_iter=12"]).expect("override");
        assert_eq!(c.max_iter, 12);
        let c2 = apply("small", &["max_iter=4", "jepa_weight=0.0"]).unwrap();
        assert_eq!(c2.max_iter, 4); assert_eq!(c2.jepa_weight, 0.0);
        // valid d_model override keeps divisibility
        let c3 = apply("small", &["d_model=840"]).unwrap();
        assert_eq!(c3.d_model, 840);
    }
    #[test]
    fn unknown_preset_errs() {
        assert!(load_config("no_such_preset_xyz").is_err());
    }

    /// A candidate that exists but is broken is REPORTED (Some(Err) with the
    /// reason), not silently skipped (ADR-0019 row 26); an absent candidate
    /// stays a silent normal miss of the search path.
    #[test]
    fn broken_candidate_is_reported_absent_is_not() {
        let dir = std::env::temp_dir().join(format!("dormouse_loader_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let broken = dir.join("broken.toml");
        std::fs::write(&broken, "d_model = 768\nd_model = = broken\n").unwrap();
        match try_file(&broken) {
            Some(Err(msg)) => assert!(msg.contains("toml parse"), "reason must name the parse: {msg}"),
            other => panic!("broken candidate must be Some(Err), got {other:?}"),
        }
        assert!(try_file(&dir.join("absent.toml")).is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }
    #[test]
    fn presets_match_original_values() {
        let n = load_config("nano").unwrap();
        assert_eq!(n.d_model, 512); assert_eq!(n.rank, 96); assert_eq!(n.n_experts, 3); assert_eq!(n.max_iter, 4);
        let s = load_config("small").unwrap();
        assert_eq!(s.d_model, 768); assert_eq!(s.rank, 64); assert_eq!(s.max_iter, 4);
        let b = load_config("base").unwrap();
        assert_eq!(b.d_model, 1024); assert_eq!(b.d_ffn, 2816); assert_eq!(b.max_seq_len, 1024);
        let sw = load_config("swift50").unwrap();
        assert_eq!(sw.d_ffn, 4096); assert_eq!(sw.n_experts, 8); assert_eq!(sw.jepa_weight, 0.0);
        assert_eq!(sw.dspark_weight, 0.0);
        let o = load_config("one_b").unwrap();
        assert_eq!(o.d_model, 2048); assert_eq!(o.max_seq_len, 2048); assert_eq!(o.max_iter, 12);
        assert_eq!(o.n_experts, 4);
        for name in ["nano","small","base","swift50","one_b"] {
            let c = load_config(name).unwrap();
            super::super::validation::validate(&c).expect(name);
        }
    }
}
