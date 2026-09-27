//! dormouse-data - streaming byte dataset.
//!
//! `ByteStream` is a bounded-memory, file-order-shuffled stream of raw bytes
//! for next-byte-language-model pretraining, plus FNV n-gram hashes used by the
//! Engram memory. A deterministic train/eval file split is exposed so training
//! can report held-out perplexity.
//!
//! Design notes:
//! - Files are read incrementally in chunks; bytes live in a ring buffer that
//!   is compacted after each batch, so memory stays bounded regardless of
//!   dataset size (the previous implementation slurped whole files into an
//!   ever-growing buffer).
//! - File order is shuffled with a seeded Fisher-Yates (no external RNG dep);
//!   the order is reshuffled every epoch for cheap cross-epoch diversity.
//! - `train_eval_split` returns a stable held-out tail so eval sees unseen data.

use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

pub fn fnv(b: &[u8]) -> u64 {
    let mut h = 1469598103934665603u64;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

/// Collect all data files under `root`, sorted for deterministic iteration.
/// Text corpora are read as raw bytes; images and binary blobs feed the
/// byte-level LM the same way (a JPEG is just a byte sequence to predict).
fn collect_files(root: &Path) -> Vec<PathBuf> {
    // FASTA matters here: a genome is bytes (A/C/G/T plus ASCII headers), so
    // the genomics arm needs no tokenizer, no new code path - only for the
    // loader to accept the files. Decompress .gz upstream: .gz is in bin_exts
    // because the corpus builder writes plain shards, and reading compressed
    // bytes as text would train on noise.
    let text_exts = [
        "parquet", "txt", "jsonl", "json", "md", "html", "xml", "csv", "fa", "fna", "fasta",
        "ffn",
    ];
    let bin_exts = [
        "png", "jpg", "jpeg", "gif", "webp", "bmp", "bin", "wasm", "zst", "gz",
    ];
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(root) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                out.extend(collect_files(&p));
            } else if p
                .extension()
                .map(|e| {
                    let s = e.to_string_lossy().to_lowercase();
                    text_exts.contains(&s.as_str()) || bin_exts.contains(&s.as_str())
                })
                .unwrap_or(false)
            {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Deterministic Fisher-Yates shuffle using a tiny SplitMix64 PRNG (no deps).
/// Same seed -> same order, so runs are reproducible.
fn shuffle_files(files: &mut [PathBuf], mut seed: u64) {
    let n = files.len();
    for i in (1..n).rev() {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let s = (seed ^ (seed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let s = (s ^ (s >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let r = (s ^ (s >> 31)) % (i as u64 + 1);
        files.swap(i, r as usize);
    }
}

/// Split collected files into `(train, eval)` by a deterministic held-out
/// fraction taken from the sorted list's tail (stable across runs).
pub fn train_eval_split(root: &Path, eval_frac: f64) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut files = collect_files(root);
    if files.is_empty() {
        return (files, Vec::new());
    }
    let eval_n = ((files.len() as f64 * eval_frac.clamp(0.0, 1.0)) as usize)
        .max(1)
        .min(files.len() - 1);
    let split_at = files.len() - eval_n;
    let eval = files.split_off(split_at);
    (files, eval)
}

pub struct ByteStream {
    seq_len: usize,
    batch: usize,
    files: Vec<PathBuf>,
    file_idx: usize,
    reader: Option<Source>,
    buf: Vec<u8>,
    pos: usize,
    capacity: usize,
    seed: u64,
    epoch: u64,
    /// Set once the ring has dropped its consumed prefix; after that a
    /// rewind() can no longer reach the original first byte, and the eval's
    /// "same window every time" guarantee is over (see `rewind`).
    drained: bool,
}

/// A file open for byte reading: plain text/binary files stream chunks;
/// parquet files stream decoded string columns (arrow batch by batch, so
/// multi-GB parquet never loads into RAM).
enum Source {
    Text(BufReader<std::fs::File>),
    Parquet {
        reader: parquet::arrow::arrow_reader::ParquetRecordBatchReader,
        batch: Vec<u8>,
        pos: usize,
    },
}

impl Source {
    fn read(&mut self, tmp: &mut Vec<u8>) -> std::io::Result<usize> {
        match self {
            Source::Text(r) => r.read(tmp),
            Source::Parquet { reader, batch, pos } => {
                if *pos >= batch.len() {
                    batch.clear();
                    *pos = 0;
                    let mut got = 0usize;
                    // Pull batches until one yields text or the file ends.
                    loop {
                        match reader.next() {
                            Some(Ok(record)) => {
                                let mut chunk = Vec::new();
                                for col in record.columns() {
                                    use arrow::array::StringArray;
                                    if let Some(sa) = col.as_any().downcast_ref::<StringArray>() {
                                        for v in sa.iter().flatten() {
                                            chunk.extend_from_slice(v.as_bytes());
                                            chunk.push(b'\n');
                                        }
                                    }
                                }
                                if !chunk.is_empty() {
                                    std::mem::swap(batch, &mut chunk);
                                    got = batch.len();
                                    break;
                                }
                            }
                            Some(Err(e)) => return Err(std::io::Error::other(e)),
                            None => break,
                        }
                    }
                    Ok(got)
                } else {
                    let n = tmp.len().min(batch.len() - *pos);
                    tmp[..n].copy_from_slice(&batch[*pos..*pos + n]);
                    *pos += n;
                    Ok(n)
                }
            }
        }
    }
}

impl ByteStream {
    /// Train stream over every data file under `data_root`.
    pub fn new(seq_len: usize, batch: usize, data_root: &Path) -> Self {
        let files = collect_files(data_root);
        assert!(
            !files.is_empty(),
            "no readable data files under {} (missing path, unmounted drive, or no data extensions)",
            data_root.display()
        );
        Self::from_files(seq_len, batch, files, 0x1234_5678)
    }

    /// Stream over an explicit file list (e.g. a held-out split).
    pub fn from_files(seq_len: usize, batch: usize, files: Vec<PathBuf>, seed: u64) -> Self {
        assert!(!files.is_empty(), "ByteStream needs a non-empty file list");
        // A sleeping/unmounted drive lists files but serves no bytes; catch it
        // at construction instead of training on a silent constant stream.
        let total: u64 = files.iter().map(|f| f.metadata().map(|m| m.len()).unwrap_or(0)).sum();
        let floor = (seq_len * batch * 4) as u64;
        assert!(
            total >= floor,
            "corpus too small: {} bytes across {} file(s), need >= {} (drive asleep or wrong dir?)",
            total,
            files.len(),
            floor
        );
        let mut files = files;
        shuffle_files(&mut files, seed);
        // The backing store may be an idle drive whose first read costs
        // ~200 ms: refilling every couple of steps with 4KB reads starves
        // the GPU. A 64 MB ring refilled in 8 MB chunks touches the disk
        // ~once per 10k steps instead of once per step.
        let capacity = 64 * 1024 * 1024;
        let mut bs = Self {
            seq_len,
            batch,
            files,
            file_idx: 0,
            reader: None,
            buf: Vec::with_capacity(capacity),
            pos: 0,
            capacity,
            seed,
            epoch: 0,
            drained: false,
        };
        bs.refill();
        bs
    }

    /// Open a readable file at `file_idx` (skipping unreadable ones). On total
    /// exhaustion, advance to the next epoch: reshuffle and restart from 0.
    fn ensure_reader(&mut self) -> bool {
        if self.reader.is_some() {
            return true;
        }
        while self.file_idx < self.files.len() {
            let path = &self.files[self.file_idx];
            let is_parquet = path
                .extension()
                .map(|e| e.to_string_lossy().eq_ignore_ascii_case("parquet"))
                .unwrap_or(false);
            let r = if is_parquet {
                std::fs::File::open(path).ok().and_then(|f| {
                    parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(f)
                        .ok()
                        .and_then(|b| b.build().ok())
                        .map(|reader| Source::Parquet {
                            reader,
                            batch: Vec::new(),
                            pos: 0,
                        })
                })
            } else {
                std::fs::File::open(path).ok().map(|f| Source::Text(BufReader::new(f)))
            };
            if let Some(r) = r {
                self.reader = Some(r);
                return true;
            }
            self.file_idx += 1;
        }
        if self.files.is_empty() {
            return false;
        }
        self.epoch += 1;
        self.file_idx = 0;
        shuffle_files(
            &mut self.files,
            self.seed.wrapping_add(self.epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        );
        while self.file_idx < self.files.len() {
            let path = &self.files[self.file_idx];
            let is_parquet = path
                .extension()
                .map(|e| e.to_string_lossy().eq_ignore_ascii_case("parquet"))
                .unwrap_or(false);
            let r = if is_parquet {
                std::fs::File::open(path).ok().and_then(|f| {
                    parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(f)
                        .ok()
                        .and_then(|b| b.build().ok())
                        .map(|reader| Source::Parquet {
                            reader,
                            batch: Vec::new(),
                            pos: 0,
                        })
                })
            } else {
                std::fs::File::open(path).ok().map(|f| Source::Text(BufReader::new(f)))
            };
            if let Some(r) = r {
                self.reader = Some(r);
                return true;
            }
            self.file_idx += 1;
        }
        false
    }

    /// Top up `buf` so it holds at least `2 * batch * seq_len` unread bytes,
    /// streaming from the current file and moving on at EOF.
    fn refill(&mut self) {
        let need = self.seq_len * self.batch * 2;
        let chunk = (8 * 1024 * 1024).max(need);
        let mut tmp = vec![0u8; chunk];
        loop {
            if self.buf.len() - self.pos >= need {
                break;
            }
            if !self.ensure_reader() {
                break;
            }
            let read = {
                let r = self.reader.as_mut().unwrap();
                r.read(&mut tmp).unwrap_or(0)
            };
            if read == 0 {
                self.reader = None;
                self.file_idx += 1;
                continue;
            }
            self.buf.extend_from_slice(&tmp[..read]);
        }
        assert!(
            !self.buf.is_empty(),
            "data stream dry: no readable bytes from {} file(s) (drive dropped mid-run?)",
            self.files.len()
        );
    }

    /// FNV 3/5/8-gram hashes (mod 4096) for the Engram memory.
    pub fn hashes(&self, bytes: &[u8], tables: &[usize]) -> Vec<i64> {
        let mut out = Vec::with_capacity(self.batch * self.seq_len * tables.len());
        for b in 0..self.batch {
            for p in 0..self.seq_len {
                let e = p + 1;
                let s3 = e.saturating_sub(3);
                let s5 = e.saturating_sub(5);
                let s8 = e.saturating_sub(8);
                let base = b * self.seq_len;
                let seq = &bytes[base..base + self.seq_len];
                out.push((fnv(&seq[s3..e]) % tables[0] as u64) as i64);
                out.push((fnv(&seq[s5..e]) % tables[1] as u64) as i64);
                out.push((fnv(&seq[s8..e]) % tables[2] as u64) as i64);
            }
        }
        out
    }

    /// Fast-forward by `n` bytes (resume): a resumed run must continue on
    /// UNSEEN data — the stream otherwise rewinds to byte 0 and the model
    /// re-reads (and memorizes) the corpus head, collapsing train CE
    /// (pretrain v2, 2026-09-26). Exact: the unread-buffer remainder stays
    /// buffered, nothing is skipped twice and nothing is lost.
    pub fn skip_bytes(&mut self, mut n: u64) {
        while n > 0 {
            if self.buf.len() - self.pos == 0 {
                self.refill();
            }
            let buffered = (self.buf.len() - self.pos) as u64;
            if buffered == 0 {
                break; // refill's dry assert has already fired for real corpora
            }
            let take = buffered.min(n);
            self.pos += take as usize;
            n -= take;
            if self.pos > self.capacity / 2 {
                self.buf.drain(0..self.pos);
                self.pos = 0;
            }
        }
    }

    /// Next `(bytes, hashes)` batch. Falls back to padding when data is short.
    pub fn next_batch(&mut self) -> (Vec<u8>, Vec<i64>) {
        self.next_batch_with_tables([4096, 4096, 4096])
    }

    /// Restart the stream from the first byte of the buffer it already holds.
    ///
    /// For EVAL this is what makes the numbers comparable: without it every
    /// eval call reads the NEXT slice, so eval N of one run and eval N of
    /// another see different bytes and the cross-run comparisons the whole
    /// A/B program rests on are noise. Deterministic as long as the data fits
    /// the ring (the eval split is ~2 MB against a 64 MB ring); a larger
    /// corpus would refill, and `refill` shuffles, so re-evaluating that
    /// needs a fresh `ByteStream` instead.
    pub fn rewind(&mut self) {
        assert!(
            !self.drained,
            "rewind() after the ring buffer dropped its consumed prefix: the \
             'fixed eval window' guarantee is void (a fresh ByteStream is the \
             only correct rewind). Keep the eval split smaller than half the \
             ring, or re-eval with a new stream."
        );
        self.pos = 0;
    }

    /// Next batch with explicit n-gram table sizes (RAM-offload tables can
    /// be millions of slots; the hashes are produced modulo the table size).
    pub fn next_batch_with_tables(&mut self, tables: [usize; 3]) -> (Vec<u8>, Vec<i64>) {
        let need = self.batch * self.seq_len;
        if self.pos + need > self.buf.len() {
            self.refill();
            if self.pos + need > self.buf.len() {
                self.pos = 0;
            }
        }
        let end = (self.pos + need).min(self.buf.len());
        let mut bytes = self.buf[self.pos..end].to_vec();
        self.pos = end;
        // Compact consumed prefix to keep memory bounded.
        if self.pos > self.capacity / 2 {
            self.buf.drain(0..self.pos);
            self.pos = 0;
        }
        assert!(
            bytes.len() == need,
            "short read: {} of {} bytes (corpus smaller than one batch?)",
            bytes.len(),
            need
        );
        let hashes = self.hashes(&bytes, &tables);
        (bytes, hashes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "no readable data files")]
    fn missing_root_is_loud() {
        ByteStream::new(8, 2, Path::new("/nonexistent/path/xyz"));
    }

    #[test]
    #[should_panic(expected = "non-empty file list")]
    fn empty_file_list_is_loud() {
        ByteStream::from_files(8, 2, Vec::new(), 1);
    }

    #[test]
    fn skip_bytes_resumes_exactly() {
        let dir = std::env::temp_dir().join(format!("dormouse_skip_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pattern: Vec<u8> = (0..256u32).map(|i| (i * 7 % 251) as u8).cycle().take(1 << 20).collect();
        let f = dir.join("corpus.bin");
        std::fs::write(&f, &pattern).unwrap();
        let (seq, batch) = (8usize, 2usize);
        let mut s = ByteStream::new(seq, batch, &dir);
        let skip = 1000u64;
        s.skip_bytes(skip);
        let (bytes, _) = s.next_batch();
        assert_eq!(bytes.len(), seq * batch);
        assert_eq!(bytes[..], pattern[skip as usize..skip as usize + seq * batch]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `rewind` is what makes the held-out eval comparable: every eval call
    /// must score the SAME bytes, otherwise eval N of run A and eval N of run
    /// B read different windows (measured 2026-09-27: one checkpoint scored
    /// 6.443 on one pass and 6.551 on the next, purely from stream position).
    #[test]
    fn rewind_restarts_the_same_window() {
        let dir = std::env::temp_dir().join(format!("dormouse_rewind_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pattern: Vec<u8> = (0..256u32).map(|i| (i * 13 % 241) as u8).cycle().take(1 << 20).collect();
        std::fs::write(dir.join("corpus.bin"), &pattern).unwrap();
        let (seq, batch) = (8usize, 2usize);
        let mut s = ByteStream::new(seq, batch, &dir);
        let (first, _) = s.next_batch();
        let (second, _) = s.next_batch();
        assert_ne!(first, second, "the stream must actually advance");
        s.rewind();
        let (again, _) = s.next_batch();
        assert_eq!(first, again, "rewind must replay the first window exactly");
        s.rewind();
        let (again2, _) = s.next_batch();
        assert_eq!(first, again2, "rewind must be idempotent");
        std::fs::remove_dir_all(&dir).ok();
    }
}
