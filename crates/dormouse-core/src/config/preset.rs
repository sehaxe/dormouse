use super::schema::{DormouseConfig, FileConfig};

const NANO_TOML: &str = include_str!("../../../../configs/nano.toml");
const SMALL_TOML: &str = include_str!("../../../../configs/small.toml");
const BASE_TOML: &str = include_str!("../../../../configs/base.toml");
const SWIFT50_TOML: &str = include_str!("../../../../configs/swift50.toml");
const ONE_B_TOML: &str = include_str!("../../../../configs/one_b.toml");

fn parse(s: &str) -> Option<DormouseConfig> {
    toml::from_str::<FileConfig>(s).ok().map(|f| f.into())
}

pub fn builtin(name: &str) -> Option<DormouseConfig> {
    match name {
        "nano" => parse(NANO_TOML),
        "small" => parse(SMALL_TOML),
        "base" => parse(BASE_TOML),
        "swift50" => parse(SWIFT50_TOML),
        "one_b" | "one-b" | "oneb" => parse(ONE_B_TOML),
        _ => None,
    }
}

pub fn list_builtins() -> &'static [&'static str] {
    &["nano", "small", "swift50", "base", "one_b"]
}

impl DormouseConfig {
    pub fn nano() -> Self { builtin("nano").expect("builtin nano.toml invalid") }
    pub fn small() -> Self { builtin("small").expect("builtin small.toml invalid") }
    pub fn base() -> Self { builtin("base").expect("builtin base.toml invalid") }
    pub fn swift50() -> Self { builtin("swift50").expect("builtin swift50.toml invalid") }
    pub fn one_b() -> Self { builtin("one_b").expect("builtin one_b.toml invalid") }
}
