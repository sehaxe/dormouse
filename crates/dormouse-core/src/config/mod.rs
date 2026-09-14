pub mod loader;
pub mod preset;
pub mod schema;
pub mod validation;
pub mod r#override;

pub use loader::{load_config, load_config_with_overrides};
pub use preset::{builtin, list_builtins};
pub use schema::{ActQuant, DormouseConfig, FileConfig};
pub use validation::validate;
pub use r#override::{apply_overrides, parse_overrides, Override};
