//! Held-out anchors for a byte corpus: the bars every claim has to clear.
//!
//! The program's judging criterion is held-out BPB, and a raw BPB number means
//! nothing without the trivial baselines next to it. Measured on the text
//! slice: uniform 8.000, unigram 5.170, 5-gram + backoff 2.572 - a 24-line
//! n-gram counter beat a 7.5M-parameter neural pipeline by 5 BPB, which is
//! the whole reason the bar ladder exists.
//!
//! Every new domain needs the same three numbers before any model result in it
//! means anything: code, genomics, math. This tool computes them for any byte
//! corpus, with an honest held-out split INSIDE the data (the last
//! `--holdout` fraction is never counted into the model) and backoff to
//! unigram for unseen contexts.
//!
//! Usage:
//!   cargo run --release -p dormouse-data --bin anchors -- <dir-or-file> [flags]
//!     --bytes 2000000     how much to read (0 = all)
//!     --holdout 0.25      trailing fraction reserved for scoring
//!     --order 5           n-gram order for the strongest baseline
//!     --skip-header       drop FASTA '>' lines from the byte stream
//!     --fit <dir-or-file> fit the counters on ANOTHER corpus and score the
//!                         positional one. This is the only way to get the bar
//!                         for the exact bytes the trainer's eval reads: its
//!                         eval is the FIRST 100 KB of the eval dir, while an
//!                         internal --holdout split scores the TRAILING
//!                         quarter, so the two numbers were never comparable
//!                         (review 2026-09-27). Measured 2026-09-28: --fit on
//!                         the filtered corpus scores the eval tail at
//!                         unigram 5.011 / 5-gram 2.588, against 5.115 / 3.001
//!                         for the in-corpus split of the same file - the
//!                         in-corpus bar is the EASIER one, and it still moves
//!                         with --bytes (1M vs 2M: 2.826 vs 2.849). --bytes
//!                         bounds the fit corpus too, so a 100 KB --fit scores
//!                         3.624: the bar is a property of the fit size, and
//!                         the header line says which one it was.
//!
//! Note on genomics: a genome is bytes, so this tool works on it unchanged -
//! no tokenizer, no conversion. The floor is ln(4) = 1.39 bits/byte for pure
//! sequence, which is a much lower bar than prose.

use std::collections::HashMap;

use std::path::{Path, PathBuf};

/// The files of a target, by the TRAINER's rule (`dormouse_data::collect_files`)
/// for a directory and as named for an explicit file.
///
/// This tool used to carry its own walk that filtered by nothing, so two
/// readers of one directory disagreed by construction: on `real_eval/` (which
/// holds `eval_tail.bin` AND the 30 MB pre-carve `eval_tail.bin.30m.bak`) it
/// reported "2 files", and any run whose limit outran the first file folded
/// that backup into the measurement while the trainer's stream sees one file —
/// a bar for a corpus nobody trains on. A directory is the trainer's business;
/// a file named on the command line is the user's.
fn files_of(target: &str) -> Vec<PathBuf> {
    let p = Path::new(target);
    if p.is_file() {
        vec![p.to_path_buf()]
    } else {
        dormouse_data::collect_files(p)
    }
}

/// `N file(s) [name, name, (+K more)]` — a bar is a number only if you know
/// which files it came from.
fn names(files: &[PathBuf]) -> String {
    let n = |p: &PathBuf| p.file_name().unwrap_or(p.as_os_str()).to_string_lossy().into_owned();
    let head: Vec<String> = files.iter().take(3).map(n).collect();
    if files.len() > 3 {
        format!("{}, +{} more", head.join(", "), files.len() - 3)
    } else {
        head.join(", ")
    }
}

/// Read up to `limit` raw bytes, sorted by path so two runs on the same corpus
/// see the same stream. Gzip is decompressed: training reads plain bytes, and
/// a compressed file is not the domain.
/// The trainer's own reader, so the bar is measured on the same bytes the
/// model sees (parquet columns decoded, not the container).
fn read_corpus(files: &[PathBuf], limit: usize) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    for f in files {
        if f.extension().map(|e| e == "gz").unwrap_or(false) {
            out.extend_from_slice(&gunzip_to_vec(f).unwrap_or_else(|e| panic!("gunzip {}: {e}", f.display())));
        } else {
            out.extend_from_slice(&dormouse_data::read_bytes(std::slice::from_ref(f), limit.saturating_sub(out.len())));
        }
        if limit > 0 && out.len() >= limit {
            break;
        }
    }
    if limit > 0 && out.len() > limit {
        out.truncate(limit);
    }
    out
}

/// `gzip -dc <file>` with the file passed as an ARGUMENT, not piped into the
/// child. Piping compressed bytes in while reading stdout deadlocks as soon as
/// the decompressed output exceeds the pipe buffer: we are still writing while
/// gzip is blocked writing.
fn gunzip_to_vec(path: &Path) -> std::io::Result<Vec<u8>> {
    let out = std::process::Command::new("gzip")
        .arg("-dc")
        .arg(path)
        .output()?;
    if !out.status.success() {
        return Err(std::io::Error::other(format!(
            "gzip exited {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(out.stdout)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: anchors <dir-or-file> [--bytes N] [--holdout F] [--order N] [--skip-header] [--fit <dir-or-file>]");
        std::process::exit(2);
    }
    let target = args[0].clone();
    let mut bytes = 2_000_000usize;
    let mut holdout = 0.25f64;
    let mut order = 5usize;
    let mut skip_header = false;
    let mut fit: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--bytes" => { bytes = args[i + 1].parse().unwrap(); i += 2; }
            "--holdout" => { holdout = args[i + 1].parse().unwrap(); i += 2; }
            "--order" => { order = args[i + 1].parse().unwrap(); i += 2; }
            "--skip-header" => { skip_header = true; i += 1; }
            "--fit" => { fit = Some(args[i + 1].clone()); i += 2; }
            other => { eprintln!("unknown flag {other}"); std::process::exit(2); }
        }
    }
    // 0 fits no held-out window at all (every baseline divides 0 by 0 and
    // prints NaN), 1 trains the counters on nothing (every baseline reads
    // 8.000, the uniform line, which is indistinguishable from a result).
    // Both used to exit 0 with a bar on stdout.
    if fit.is_none() && !(holdout > 0.0 && holdout < 1.0) {
        eprintln!(
            "anchors: --holdout {holdout} leaves no held-out window: 0 scores nothing (unigram \
             NaN) and 1 fits the counters on nothing (every baseline reads 8.000, the uniform \
             line, not a result). Pass 0 < --holdout < 1, or --fit <corpus> to score this corpus \
             with counters fitted on another."
        );
        std::process::exit(2);
    }
    let files = files_of(&target);
    if files.is_empty() {
        eprintln!("no data files under {target} (the trainer's extension list, or the path does not exist)");
        std::process::exit(1);
    }
    let mut data = read_corpus(&files, bytes);
    if skip_header {
        // FASTA: drop whole '>' lines. Keeps the byte stream to sequence, which
        // is what a genomics anchor should measure.
        data = data
            .split_inclusive(|b| *b == b'\n')
            .filter(|l| !l.starts_with(b">"))
            .flat_map(|l| l.to_vec())
            .collect();
    }
    assert!(data.len() > 10_000, "corpus too small for an anchor: {} B", data.len());
    // With --fit, the positional corpus is the SCORING window and the counters
    // are fitted on the other one - the trainer's own split, not our own.
    let mut fit_data: Vec<u8> = Vec::new();
    let mut cut = 0usize;
    let (train, test): (&[u8], &[u8]) = match &fit {
        Some(f) => {
            let ffiles = files_of(f);
            let mut fd = read_corpus(&ffiles, bytes);
            if skip_header {
                fd = fd
                    .split_inclusive(|b| *b == b'\n')
                    .filter(|l| !l.starts_with(b">"))
                    .flat_map(|l| l.to_vec())
                    .collect();
            }
            // An unreadable --fit root used to leave the counters fitted on
            // NOTHING and print 8.000 for every baseline, the same number a
            // real uniform corpus gives.
            assert!(
                fd.len() > 10_000,
                "anchors: --fit {f} yielded {} B (unreadable path, or a directory with no data files?) - the counters would be fitted on nothing",
                fd.len()
            );
            // Fitted on the bytes it scores: every baseline below is
            // memorisation wearing a BPB.
            assert!(
                std::fs::canonicalize(f).ok() != std::fs::canonicalize(&target).ok(),
                "anchors: --fit {f} IS the corpus being scored - the counters would be fitted on the \
                 exact bytes they are scored against, so the numbers below measure nothing"
            );
            fit_data = fd;
            (fit_data.as_slice(), data.as_slice())
        }
        None => {
            cut = ((data.len() as f64) * (1.0 - holdout)) as usize;
            data.split_at(cut)
        }
    };
    let (train, test) = (train, test);
    if let Some(name) = Path::new(&target).file_name() {
        println!("domain: {}", name.to_string_lossy());
    }
    // The headline a bar is quoted from, and the fix for "a quoted bar is not a
    // number": WHICH files, how many bytes were read, and WHICH bytes were
    // scored. The bar moves with --bytes because the counters are fitted on one
    // window and scored on the next, and a 500 MB tail carve has drifting local
    // statistics - measured on the same file, 1M vs 2M gave 2.826 vs 2.849 BPB,
    // so the window, not the corpus, sets the number.
    println!(
        "corpus: {} - {} file(s) [{}], {} B read",
        target,
        files.len(),
        names(&files),
        data.len()
    );
    let (lo, fitted) = if fit.is_some() {
        (
            0,
            format!(
                "{} B of {} (a different corpus; --bytes bounds the fit corpus too, so a small \
                 --bytes starves the counters)",
                fit_data.len(),
                fit.as_deref().unwrap_or("?")
            ),
        )
    } else {
        (
            cut,
            format!("bytes [0, {cut}) of the same read, holdout {holdout}"),
        )
    };
    println!(
        "window: scored bytes [{lo}, {}) = {} B; counters fitted on {fitted}",
        data.len(),
        data.len() - lo
    );

    // uniform
    println!("uniform            {:>8.3} BPB", 8.0);

    // unigram on train, scored on held-out
    let mut uni = [0u64; 256];
    for &b in train { uni[b as usize] += 1; }
    let n = train.len() as f64;
    let uni_p = |b: u8| (uni[b as usize] as f64 + 1.0) / (n + 256.0);
    let mut h1 = 0.0f64;
    for &b in test { h1 -= uni_p(b).log2(); }
    let h1 = h1 / test.len() as f64;
    println!("unigram            {:>8.3} BPB", h1);

    // n-gram on train, backoff to unigram on held-out
    let ctx: HashMap<Vec<u8>, HashMap<u8, u64>> = {
        let mut m: HashMap<Vec<u8>, HashMap<u8, u64>> = HashMap::new();
        for w in train.windows(order) {
            *m.entry(w[..order - 1].to_vec()).or_default().entry(w[order - 1]).or_insert(0) += 1;
        }
        m
    };
    let mut hn = 0.0f64;
    let mut misses = 0u64;
    for w in test.windows(order) {
        let c = &w[..order - 1];
        let x = w[order - 1];
        let p = match ctx.get(c).and_then(|h| h.get(&x)) {
            Some(&cnt) => {
                let tot: u64 = ctx[c].values().sum();
                0.75 * (cnt as f64 / tot as f64) + 0.25 * uni_p(x)
            }
            None => { misses += 1; uni_p(x) }
        };
        hn -= p.max(1e-12).log2();
    }
    let hn = hn / test.len() as f64;
    println!(
        "{order}-gram+backoff    {:>8.3} BPB  ({:.1}% contexts unseen in train)",
        hn,
        100.0 * misses as f64 / test.len() as f64
    );
    println!("\nA model below the {order}-gram line has learned structure a counter cannot see;");
    println!("a model above the unigram line has learned nothing a letter counter does not know.");
}
