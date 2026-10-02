//! Shuffle a filtered corpus for training: route documents round-robin by
//! FNV(doc) into N shard files. Each shard's content is then a uniform
//! random sample of the whole corpus, so the trainer's file-order shuffle +
//! sequential reads mix domains at every batch. The filter preserves input
//! order, which here meant domain blocks (math walls, title-list junk) —
//! train CE collapsed on the local domain while held-out eval stayed hard.
//!
//! Usage: `shard <corpus-in> <out-dir> <n-shards>`
//! Docs are separated by 2+ consecutive newlines (the filter's separator);
//! each output doc is written verbatim followed by "\n\n".

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn main() {
    let mut it = std::env::args().skip(1);
    let inp = it.next().unwrap_or_else(|| usage());
    let out_dir = it.next().unwrap_or_else(|| usage());
    let n: usize = it
        .next()
        .unwrap_or_else(|| usage())
        .parse()
        .unwrap_or_else(|_| usage());
    if n == 0 {
        usage();
    }
    let out_dir = Path::new(&out_dir);
    std::fs::create_dir_all(out_dir).expect("create out dir");

    let mut writers: Vec<BufWriter<File>> = (0..n)
        .map(|i| {
            let p = out_dir.join(format!("corpus_{i:03}.bin"));
            BufWriter::with_capacity(1 << 20, File::create(p).expect("create shard"))
        })
        .collect();

    let t0 = std::time::Instant::now();
    let mut src = BufReader::with_capacity(1 << 23, File::open(&inp).expect("open corpus"));
    let mut doc: Vec<u8> = Vec::with_capacity(1 << 12);
    let mut run: Vec<u8> = Vec::new(); // pending newlines (the separator)
    let mut docs: u64 = 0;
    let mut bytes: u64 = 0;
    let mut chunk = vec![0u8; 1 << 23];

    let push = |b: u8, doc: &mut Vec<u8>, run: &mut Vec<u8>| {
        if b == b'\n' {
            run.push(b);
        } else {
            doc.extend_from_slice(run);
            run.clear();
            doc.push(b);
        }
    };

    loop {
        let read = src.read(&mut chunk).expect("read corpus");
        if read == 0 {
            break;
        }
        for &b in &chunk[..read] {
            if b == b'\n' {
                run.push(b);
                if run.len() >= 2 {
                    // blank run: complete the doc if any
                    if !doc.is_empty() {
                        let shard = (fnv(&doc) % n as u64) as usize;
                        writers[shard].write_all(&doc).expect("write doc");
                        writers[shard].write_all(b"\n\n").expect("write sep");
                        docs += 1;
                        bytes += doc.len() as u64;
                        doc.clear();
                    }
                    run.clear();
                }
            } else {
                push(b, &mut doc, &mut run);
            }
        }
        if docs > 0 && docs.is_multiple_of(2_000_000) {
            eprintln!(
                "[shard] docs={docs} bytes={bytes} elapsed={:?}",
                t0.elapsed()
            );
        }
    }
    if !doc.is_empty() {
        let shard = (fnv(&doc) % n as u64) as usize;
        writers[shard].write_all(&doc).expect("write doc");
        docs += 1;
    }
    for w in &mut writers {
        w.flush().expect("flush shard");
    }
    eprintln!(
        "[shard] done: docs={docs} bytes={bytes} elapsed={:?}",
        t0.elapsed()
    );
}

fn usage() -> ! {
    eprintln!("usage: shard <corpus-in> <out-dir> <n-shards>");
    std::process::exit(2);
}
