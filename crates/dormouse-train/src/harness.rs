//! opencode harness - calls opencode CLI if available
pub fn run(task: &str) -> String {
    match std::process::Command::new("opencode").arg("run").arg(task).output() {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Ok(o) => format!("opencode failed: {}", String::from_utf8_lossy(&o.stderr)),
        Err(_) => format!("opencode run {}", task),
    }
}
