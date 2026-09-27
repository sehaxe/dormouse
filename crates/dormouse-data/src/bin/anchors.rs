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
//!
//! Note on genomics: a genome is bytes, so this tool works on it unchanged -
//! no tokenizer, no conversion. The floor is ln(4) = 1.39 bits/byte for pure
//! sequence, which is a much lower bar than prose.

use std::collections::HashMap;

use std::path::{Path, PathBuf};

fn collect(root: &Path, out: &mut Vec<PathBuf>) {
    if root.is_file() {
        out.push(root.to_path_buf());
        return;
    }
    if let Ok(rd) = std::fs::read_dir(root) {
        let mut entries: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        entries.sort(); // deterministic order across machines
        for p in entries {
            collect(&p, out);
        }
    }
}

/// Read up to `limit` raw bytes, sorted by path so two runs on the same corpus
/// see the same stream. Gzip is decompressed: training reads plain bytes, and
/// a compressed file is not the domain.
fn read_corpus(files: &[PathBuf], limit: usize) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    for f in files {
        let data = if f.extension().map(|e| e == "gz").unwrap_or(false) {
            gunzip_to_vec(f).unwrap_or_else(|e| panic!("gunzip {}: {e}", f.display()))
        } else {
            std::fs::read(f).unwrap_or_else(|e| panic!("read {}: {e}", f.display()))
        };
        buf.extend_from_slice(&data);
        if limit > 0 && buf.len() >= limit {
            break;
        }
    }
    if limit > 0 && buf.len() > limit {
        buf.truncate(limit);
    }
    buf
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
        eprintln!("usage: anchors <dir-or-file> [--bytes N] [--holdout F] [--order N] [--skip-header]");
        std::process::exit(2);
    }
    let target = args[0].clone();
    let mut bytes = 2_000_000usize;
    let mut holdout = 0.25f64;
    let mut order = 5usize;
    let mut skip_header = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--bytes" => { bytes = args[i + 1].parse().unwrap(); i += 2; }
            "--holdout" => { holdout = args[i + 1].parse().unwrap(); i += 2; }
            "--order" => { order = args[i + 1].parse().unwrap(); i += 2; }
            "--skip-header" => { skip_header = true; i += 1; }
            other => { eprintln!("unknown flag {other}"); std::process::exit(2); }
        }
    }
    let mut files = Vec::new();
    collect(Path::new(&target), &mut files);
    if files.is_empty() {
        eprintln!("no files under {target}");
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
    let cut = ((data.len() as f64) * (1.0 - holdout)) as usize;
    let (train, test) = data.split_at(cut);
    println!("corpus: {} files, {} B read, {} B train / {} B held out", files.len(), data.len(), train.len(), test.len());
    if let Some(name) = Path::new(&target).file_name() {
        println!("domain: {}", name.to_string_lossy());
    }

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
