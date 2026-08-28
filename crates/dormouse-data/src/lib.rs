//! dormouse-data - ByteStream, FNV 3/5/8 mod 4096, streaming mix
use std::path::{Path, PathBuf};

pub fn fnv(b: &[u8]) -> u64 {
    let mut h = 1469598103934665603u64;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn collect_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(root) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                out.extend(collect_files(&p));
            } else if p.extension().map(|e| e == "parquet" || e == "txt" || e == "jsonl").unwrap_or(false) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

pub struct ByteStream {
    pub seq_len: usize,
    pub batch: usize,
    files: Vec<PathBuf>,
    buf: Vec<u8>,
    pos: usize,
    file_idx: usize,
}

impl ByteStream {
    pub fn new(seq_len: usize, batch: usize, data_root: &Path) -> Self {
        let files = if data_root.exists() {
            collect_files(data_root)
        } else {
            Vec::new()
        };
        let mut bs = Self { seq_len, batch, files, buf: Vec::new(), pos: 0, file_idx: 0 };
        bs.refill();
        bs
    }

    fn refill(&mut self) {
        while self.buf.len() - self.pos < self.seq_len * self.batch * 2 && self.file_idx < self.files.len() {
            if let Ok(data) = std::fs::read(&self.files[self.file_idx]) {
                // for parquet just take raw bytes, for txt take as is
                self.buf.extend_from_slice(&data);
            }
            self.file_idx = (self.file_idx + 1) % self.files.len().max(1);
            if self.files.is_empty() {
                self.buf.extend_from_slice(&[b'a'; 8192]);
                break;
            }
        }
        if self.buf.is_empty() {
            self.buf = vec![b'x'; 1_000_000];
        }
    }

    pub fn hashes(&self, bytes: &[u8], tables: &[usize]) -> Vec<i64> {
        // bytes: [batch * seq_len]
        let mut out = Vec::with_capacity(self.batch * self.seq_len * tables.len());
        for b in 0..self.batch {
            for p in 0..self.seq_len {
                let idx = b * self.seq_len + p;
                let e = p + 1;
                let s3 = e.saturating_sub(3);
                let s5 = e.saturating_sub(5);
                let s8 = e.saturating_sub(8);
                // slice within this sequence's window
                let base = b * self.seq_len;
                let seq = &bytes[base..base + self.seq_len];
                out.push((fnv(&seq[s3..e]) % tables[0] as u64) as i64);
                out.push((fnv(&seq[s5..e]) % tables[1] as u64) as i64);
                out.push((fnv(&seq[s8..e]) % tables[2] as u64) as i64);
                let _ = idx;
            }
        }
        out
    }

    pub fn next_batch(&mut self) -> (Vec<u8>, Vec<i64>) {
        let need = self.batch * self.seq_len;
        if self.pos + need > self.buf.len() {
            self.refill();
            if self.pos + need > self.buf.len() {
                // wrap
                self.pos = 0;
            }
        }
        let end = (self.pos + need).min(self.buf.len());
        let mut bytes = self.buf[self.pos..end].to_vec();
        self.pos = end;
        if bytes.len() < need {
            bytes.extend(std::iter::repeat(b' ').take(need - bytes.len()));
        }
        // truncate to byte range 0-255 already
        let tables = [4096usize, 4096, 4096];
        let hashes = self.hashes(&bytes, &tables);
        (bytes, hashes)
    }
}
