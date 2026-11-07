//! dormouse-board — the landing page's data comes from the run logs, not a
//! human with a text editor.
//!
//! The repository's whole claim is that every number is measured and every
//! retraction is kept. A landing page is the one place where that claim is
//! easiest to break: someone types "1.89x" once and three months later it is
//! 2.4x. So this binary computes the numbers from `logs/**/*.jsonl` and injects
//! them into `site/template.html`. If a log is missing, the row is missing —
//! there is no default and no placeholder.
//!
//!     cargo run --release -- --logs ../../logs --template ../template.html --out ../index.html
//!
//! Deliberately std + serde_json: no async, no walkdir, no templating crate.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

/// The rung labels the throughput ladder writes, in reading order. A tag that
/// is not here still appears in `runs`; only the ladder table is curated,
/// because a table needs an order and the logs do not carry one.
/// (throughput tag, quality tag, label). The quality tag is the SAME rung in
/// `--quality-logs` (`scripts/muon_quality_ab.sh`), so the BPB column is a
/// join between two ladders on one name instead of a number typed here.
const RUNGS: &[(&str, &str, &str)] = &[
    ("muon_fp32", "fp32_5", "fp32 NS · 5 steps (the control)"),
    ("state_bf16", "state", "bf16 optimizer state only"),
    ("ns3", "fp32_3", "fp32 NS · 3 steps"),
    ("ns_bf16", "bf16_5", "bf16 NS · 5 steps"),
    ("ns_bf16_3", "bf16_3", "bf16 NS · 3 steps"),
    ("muon_fast", "bf16_3", "bf16 NS · 3 steps · bf16 state"),
];

#[derive(Default)]
struct Run {
    file: String,
    source: String,
    opt: String,
    batch: u64,
    seq: u64,
    lr: f64,
    log_lines: usize,
    rates: Vec<f64>,
    evals: Vec<(f64, f64)>, // (bytes_seen, bpb_fin)
}

impl Run {
    fn median_rate(&mut self) -> f64 {
        if self.rates.is_empty() {
            return 0.0;
        }
        self.rates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = self.rates.len();
        if n % 2 == 1 {
            self.rates[n / 2]
        } else {
            (self.rates[n / 2 - 1] + self.rates[n / 2]) / 2.0
        }
    }

    fn last_eval(&self) -> Option<(f64, f64)> {
        self.evals.last().copied()
    }

    fn bytes_per_step(&self) -> f64 {
        (self.batch * self.seq) as f64
    }
}

fn jsonl_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            jsonl_files(&p, out);
        } else if p.extension().is_some_and(|e| e == "jsonl") {
            out.push(p);
        }
    }
}

fn parse_run(path: &Path) -> Option<Run> {
    let text = fs::read_to_string(path).ok()?;
    let mut run = Run {
        file: path.file_stem()?.to_string_lossy().to_string(),
        ..Default::default()
    };
    let mut seen_run = false;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(line) else { continue };
        match v.get("event").and_then(Value::as_str) {
            Some("run") => {
                seen_run = true;
                run.source = v
                    .get("source")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .rsplit('/')
                    .next()
                    .unwrap_or("")
                    .trim_end_matches(".toml")
                    .to_string();
                run.opt = v.get("opt").and_then(Value::as_str).unwrap_or("").to_string();
                run.batch = v.get("batch").and_then(Value::as_u64).unwrap_or(0);
                run.seq = v.get("seq_len").and_then(Value::as_u64).unwrap_or(0);
                run.lr = v.get("lr").and_then(Value::as_f64).unwrap_or(0.0);
            }
            Some("log") => {
                run.log_lines += 1;
                if let Some(r) = v.get("tokens_per_s").and_then(Value::as_f64) {
                    if r.is_finite() && r > 0.0 {
                        run.rates.push(r);
                    }
                }
            }
            Some("eval") => {
                if let (Some(b), Some(bpb)) = (
                    v.get("bytes_seen").and_then(Value::as_f64),
                    v.get("bpb_fin").and_then(Value::as_f64),
                ) {
                    run.evals.push((b, bpb));
                }
            }
            _ => {}
        }
    }
    if seen_run && run.log_lines > 0 {
        Some(run)
    } else {
        None
    }
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut logs = PathBuf::from("../../logs");
    let mut quality_logs = PathBuf::from("../../logs/muonq");
    let mut surprise = PathBuf::from("../assets/surprise.json");
    let mut script = PathBuf::from("../pixel.js");
    let mut template = PathBuf::from("../template.html");
    let mut out = PathBuf::from("../index.html");
    while let Some(a) = args.next() {
        match a.as_str() {
            "--logs" => logs = PathBuf::from(args.next().expect("--logs PATH")),
            "--quality-logs" => quality_logs = PathBuf::from(args.next().expect("--quality-logs PATH")),
            "--surprise" => surprise = PathBuf::from(args.next().expect("--surprise PATH")),
            "--script" => script = PathBuf::from(args.next().expect("--script PATH")),
            "--template" => template = PathBuf::from(args.next().expect("--template PATH")),
            "--out" => out = PathBuf::from(args.next().expect("--out PATH")),
            other => {
                eprintln!("unknown argument {other}\nusage: dormouse-board [--logs PATH] [--template PATH] [--out PATH]");
                std::process::exit(2);
            }
        }
    }

    let mut files = Vec::new();
    jsonl_files(&logs, &mut files);
    files.sort();

    let mut runs: Vec<Run> = files.iter().filter_map(|p| parse_run(p)).collect();

    // --- the ladder: one row per curated tag, from its own median ---------
    let mut by_tag: BTreeMap<String, &mut Run> = BTreeMap::new();
    for run in runs.iter_mut() {
        by_tag.insert(run.file.clone(), run);
    }
    // the quality ladder: same rungs, 2000 steps, one seed, at 16.4 MB
    let mut qfiles = Vec::new();
    jsonl_files(&quality_logs, &mut qfiles);
    let mut qmap: BTreeMap<String, f64> = BTreeMap::new();
    for f in &qfiles {
        if let Some(r) = parse_run(f) {
            if let Some((_, bpb)) = r.last_eval() {
                qmap.insert(r.file, bpb);
            }
        }
    }
    let control_bpb = qmap.get("fp32_5").copied();

    let mut rungs = Vec::new();
    for (tag, qtag, label) in RUNGS {
        let Some(run) = by_tag.get_mut(*tag) else { continue };
        let kbs = run.median_rate() / 1000.0;
        if kbs <= 0.0 {
            continue;
        }
        let dbpb = match (qmap.get(*qtag), control_bpb) {
            (Some(q), Some(c)) => Some(round2(q - c)),
            _ => None,
        };
        // The bar the repo uses everywhere: +/-0.02 BPB is the noise floor.
        let safe = dbpb.map(|d| d.abs() <= 0.02).unwrap_or(false);
        rungs.push(json!({
            "tag": tag,
            "label": label,
            "kbs": round2(kbs),
            "ms_per_step": round2(run.bytes_per_step() / kbs),
            "dbpb": dbpb,
            "safe": safe,
        }));
    }

    // --- the curve: the BEST held-out number on record, not the longest run.
    // A failed arm (E1's patch unit at 3.30 BPB) is longer than the good one and
    // must never become the headline curve. Floor: 100 MB, so a 10-step smoke
    // cannot win by luck.
    let longest = runs
        .iter()
        .filter(|r| r.last_eval().map(|e| e.0).unwrap_or(0.0) >= 1e8)
        .min_by(|a, b| {
            let ka = a.last_eval().map(|e| e.1).unwrap_or(f64::INFINITY);
            let kb = b.last_eval().map(|e| e.1).unwrap_or(f64::INFINITY);
            ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
        });
    let curve = longest
        .map(|r| {
            r.evals
                .iter()
                .map(|(b, bpb)| json!({"mb": round2(b / 1e6), "bpb": round2(*bpb)}))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    // --- the table: every run with a held-out number, longest first ------
    let mut table: Vec<&Run> = runs.iter().filter(|r| r.last_eval().is_some()).collect();
    table.sort_by(|a, b| {
        let ka = a.last_eval().map(|e| e.0).unwrap_or(0.0);
        let kb = b.last_eval().map(|e| e.0).unwrap_or(0.0);
        kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
    });
    let rows = table
        .iter()
        .take(12)
        .map(|r| {
            let (mb, bpb) = r.last_eval().unwrap_or((0.0, 0.0));
            let kbs = r.rates.first().copied().unwrap_or(0.0);
            let _ = kbs;
            json!({
                "run": r.file,
                "source": r.source,
                "opt": r.opt,
                "batch": r.batch,
                "seq": r.seq,
                "mb": round2(mb / 1e6),
                "bpb": round2(bpb),
            })
        })
        .collect::<Vec<_>>();

    let with_eval = runs.iter().filter(|r| !r.evals.is_empty()).count();
    let data = json!({
        "rungs": rungs,
        "curve": curve,
        "runs": rows,
        "stats": {
            "runs_on_record": runs.len(),
            "runs_with_bpb": with_eval,
            "longest_mb": curve.last().map(|c| c["mb"].clone()).unwrap_or(json!(0)),
        }
    });

    let Ok(tpl) = fs::read_to_string(&template) else {
        eprintln!("cannot read template {}", template.display());
        std::process::exit(1);
    };
    let blob = serde_json::to_string(&data).expect("serialize");
    // The hero image's data: 1024 bytes of held-out text with their surprise.
    // Optional — without it the section renders empty rather than inventing one.
    let surprise_blob = fs::read_to_string(&surprise).ok().filter(|t| !t.trim().is_empty());
    let mut html = tpl.replace("/*__DATA__*/null", &blob);
    html = html.replace(
        "/*__SURPRISE__*/null",
        surprise_blob.as_deref().unwrap_or("null"),
    );
    let js = fs::read_to_string(&script).unwrap_or_else(|e| {
        eprintln!("cannot read script {}: {e}", script.display());
        std::process::exit(1);
    });
    html = html.replace("/*__SCRIPT__*/", &js);
    if let Err(e) = fs::write(&out, html) {
        eprintln!("cannot write {}: {e}", out.display());
        std::process::exit(1);
    }
    println!(
        "wrote {} — {} runs on record, {} with a held-out BPB, {} ladder rungs, {} curve points",
        out.display(),
        runs.len(),
        with_eval,
        data["rungs"].as_array().map(Vec::len).unwrap_or(0),
        data["curve"].as_array().map(Vec::len).unwrap_or(0)
    );
}
