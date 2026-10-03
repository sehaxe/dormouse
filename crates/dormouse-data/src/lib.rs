//! dormouse-data - streaming byte dataset.
//!
//! [`ByteStream`] is a bounded-memory, file-order-shuffled stream of raw bytes
//! for next-byte-language-model pretraining, plus FNV n-gram hashes used by the
//! Engram memory. A deterministic train/eval file split is exposed so training
//! can report held-out perplexity.
//!
//! # Design notes
//!
//! - Files are read incrementally in chunks; bytes live in a ring buffer that
//!   is compacted after each batch, so memory stays bounded regardless of
//!   dataset size (the previous implementation slurped whole files into an
//!   ever-growing buffer).
//! - File order is shuffled with a seeded Fisher-Yates (no external RNG dep);
//!   the order is reshuffled every epoch for cheap cross-epoch diversity.
//!
//! # The contracts a caller must not break
//!
//! This crate is where "the bytes the model trains on" is decided, and every
//! interesting bug in this project's history that was NOT about the model was
//! about this crate. Four rules, each of which exists because breaking it is
//! silent:
//!
//! * **One definition of the file set, one definition of the bytes.**
//!   [`collect_files`] and [`read_bytes`] exist so a second reader of the same
//!   directory cannot disagree by construction — the anchors tool had its own
//!   unfiltered walk, and on `real_eval/` (which holds a `eval_tail.bin.30m.bak`)
//!   it reported "2 files" and would have folded a 30 MB pre-carve backup into a
//!   measurement whose trainer-side stream sees one.
//! * **A read ERROR is not an EOF.** Treating one as the other ends a file
//!   early and the caller measures a corpus shorter than the one it asked for,
//!   with no sign of it (ADR-0019). Every failure to open or read a shard is
//!   COUNTED on stderr and named by path.
//! * **Nothing is synthesized.** The parquet decoder walks the column TREE by
//!   data type and reads only string-typed columns; a numeric column is never
//!   cast to text. It recurses, because the decoder it replaced downcast the
//!   top-level columns to `StringArray` and dropped everything else with no
//!   counter — `mix/qa` gave up 2 534 708 673 B on disk and 283 KB to the
//!   loader (0.011%: the `id` column, the only top-level string) and the run
//!   trained on ids while reporting a confident loss curve.
//! * **The eval tree must not be reachable from the training tree.**
//!   [`ByteStream::train_and_eval`] refuses that by name, because
//!   [`collect_files`] recurses: `--data` at the parent of the eval directory
//!   collects the eval bytes as training data and every held-out number the run
//!   prints is optimistic by an unmeasured amount (ADR-0010).
//!
//! # Cost, in one paragraph
//!
//! A 64 MB ring refilled in 8 MB chunks touches the disk about once per 10k
//! steps instead of once per step, because the backing store may be an idle
//! drive whose first read costs ~200 ms and 4 KB reads would starve the GPU.
//! Everything else in the hot path is a memcpy: `next_batch` hands back
//! `batch * seq_len` bytes and one FNV pass over them.
//!
//! [`sft`] is the supervised half: JSONL conversations rendered to the same
//! byte stream, plus the mask that says which of those bytes the loss may
//! score. It is a separate module, not a flag on [`ByteStream`], because the
//! mask is a per-position quantity the pretraining path has no meaning for.
//!
//! # Gate
//!
//! `#![warn(missing_docs)]` and `#![warn(rustdoc::broken_intra_doc_links)]` are
//! on, and `RUSTFLAGS="-D warnings" cargo doc --no-deps -p dormouse-core -p
//! dormouse-data` is green.
#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

pub mod sft;

use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use arrow::array::Array;
use arrow::datatypes::DataType;

/// FNV-1a 64-bit digest. The primitive under [`raw_keys`].
///
/// Public because the shard router in `bin/shard.rs` routes documents by
/// `FNV(doc)` and must agree with it, and because a reader comparing the two
/// derivations needs to see the offsets. **It is not the memory key**: the
/// keys are [`raw_keys`]' low 31 bits, and the trainer's decode seam goes
/// through [`raw_keys`] for exactly that reason — a second copy of the key
/// derivation is a key the model was never trained on.
pub fn fnv(b: &[u8]) -> u64 {
    let mut h = 1469598103934665603u64;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

/// N-gram orders the Engram hashes, one table each, smallest first. The
/// model's `DormouseConfig::engram_orders` defaults to the same list and a
/// cross-crate test in dormouse-train pins the two together (the trainer
/// cannot pass the config down here without touching its frozen plumbing).
///
/// 2/3/4 replaces 3/5/8, which spent 2/3 of the table on two dead arms: at
/// 8M rows only n=3 had per-key support (the corpus exhausts the 16.8M
/// 3-gram key space 2750x over) and n=5/n=8 were ~5775-way averages, i.e.
/// 512M parameters carrying no more information than the mean of their
/// members. It is DeepSeek's own shipped set over compressed tokens
/// (V4.1-Flash n in 2,3,4; Engram-27B 2,3) and the deepest order whose
/// key space (256^4 = 4.3e9) a 46 GB byte corpus can still populate.
pub const ORDERS: [usize; 3] = [2, 3, 4];

/// The FNV n-gram keys of ONE byte sequence, row-major `[t, ORDERS]`: the low
/// **31** bits of the FNV-1a digest of each context, one column per entry of
/// [`ORDERS`], RAW (not reduced) — the model masks the slot index itself, so
/// `engram_rows` stays the only copy of the capacity.
///
/// The one derivation, shared by every caller that starts from bytes: the
/// trainer's [`ByteStream::hashes_raw`] (this function, once per batch row),
/// and the decode seam `dormouse_train::decode::next_byte_logits`. A key
/// computed by a second copy of the FNV is a key the model was never trained
/// on, and the symptom is not an error — it is a network whose memory arm
/// quietly contributes something else (or, with no keys at all, literal
/// zeros: `loop_block.rs`'s `None => None`).
///
/// Truncating to 31 bits is deliberate - `Int` is i32 on burn-flex and the cast
/// PANICS above `i32::MAX` instead of wrapping the way CUDA's does - and 2.1e9
/// keys is far more than any affordable table can separate anyway. 31, not 32:
/// the previous 32-bit version was correct in its reasoning and off by one bit
/// in its implementation, which made every training run die on real data with
/// `Element cannot be represented in the target type: "i64"(...) => "i32"`.
pub fn raw_keys(bytes: &[u8]) -> Vec<i64> {
    let mut out = Vec::with_capacity(bytes.len() * ORDERS.len());
    for e in 1..=bytes.len() {
        for &n in ORDERS.iter() {
            let s = e.saturating_sub(n);
            out.push(((fnv(&bytes[s..e]) as u32) & 0x7fff_ffff) as i64);
        }
    }
    out
}

/// Collect all data files under `root`, sorted for deterministic iteration.
/// Text corpora are read as raw bytes; images and binary blobs feed the
/// byte-level LM the same way (a JPEG is just a byte sequence to predict).
///
/// PUBLIC because it is the definition of "the files the trainer sees", and a
/// second reader of the same directory is how two tools disagree by
/// construction: the anchors tool had its own walk that filtered by nothing,
/// so on `real_eval/` (which holds `eval_tail.bin.30m.bak`) it reported "2
/// files" and would have folded a 30 MB pre-carve backup into a measurement
/// whose trainer-side stream sees one file. Both readers call this now.
pub fn collect_files(root: &Path) -> Vec<PathBuf> {
    // FASTA matters here: a genome is bytes (A/C/G/T plus ASCII headers), so
    // the genomics arm needs no tokenizer, no new code path - only for the
    // loader to accept the files. Decompress .gz upstream: .gz is in bin_exts
    // because the corpus builder writes plain shards, and reading compressed
    // bytes as text would train on noise.
    let text_exts = [
        "parquet", "txt", "jsonl", "json", "md", "html", "xml", "csv", "fa", "fna", "fasta", "ffn",
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

/// A bounded-memory stream of raw bytes, reshuffled per epoch.
///
/// # The three guarantees a caller depends on
///
/// 1. **Bounded memory.** A 64 MB ring, refilled in 8 MB chunks and compacted
///    once its consumed prefix passes half. Memory does not depend on corpus
///    size, which is what lets a 46 GB corpus train on a 16 GB card.
/// 2. **A resume does not re-read.** [`Self::skip_bytes`] fast-forwards exactly
///    (`n` bytes consumed, nothing skipped twice, nothing lost), and a skip that
///    runs off the end of the corpus is LOUD rather than short — the pretrain-v2
///    collapse was a `break` here that resumed a run at the head of the corpus
///    it had already trained on, with nothing in the log.
/// 3. **Eval sees the same window every time**, via [`Self::rewind`]. Without it
///    every eval call reads the NEXT slice: one checkpoint scored 6.443 on one
///    pass and 6.551 on the next, purely from stream position, which makes every
///    cross-run comparison noise. The guarantee is void once the ring has
///    dropped its head (`drained`), and `rewind` refuses rather than silently
///    scoring a different window.
///
/// # Failure modes, all loud
///
/// No readable files under the root; a corpus smaller than `4 * batch *
/// seq_len` bytes; a read that fails mid-stream (a read ERROR is not an EOF);
/// a ring that cannot be filled without wrapping into the next epoch; a short
/// read of one batch. Every one of them names the cause.
pub struct ByteStream {
    /// Positions per row of a batch, in BYTES. The vocabulary is bytes, so a
    /// "position" is a byte and every sequence length in this crate is a byte
    /// count.
    seq_len: usize,
    /// Rows per batch. Multiplies `seq_len` into every batch size and every
    /// minimum-corpus check.
    batch: usize,
    /// The shuffled file list. Reshuffled in place at every epoch boundary,
    /// seeded by `seed + epoch * GOLDEN`, so epoch N's order is a function of
    /// the seed and the epoch index — not of how many bytes were read.
    files: Vec<PathBuf>,
    /// Index into `files` of the shard currently open or next to open.
    file_idx: usize,
    /// The open shard, or `None` between files. `None` is what makes
    /// `ensure_reader` re-open (and re-say so) on the next refill.
    reader: Option<Source>,
    /// The ring: 64 MB of bytes, of which `buf[pos..]` is unread.
    buf: Vec<u8>,
    /// Read cursor into `buf`. Reset to 0 by compaction, by [`Self::rewind`],
    /// and by [`Self::skip_bytes`]'s drain — the three places the consumed
    /// prefix can leave the ring.
    pos: usize,
    /// Ring capacity, 64 MB, fixed at construction. The compaction threshold
    /// is `capacity / 2`.
    capacity: usize,
    /// The shuffle seed, kept so epoch N can derive its own without threading
    /// a counter through the shuffle.
    seed: u64,
    /// How many times the file list has been exhausted and restarted. Also the
    /// tripwire for "a refill wanted more bytes than the corpus had": wrapping
    /// during a refill is the pretrain-v2 collapse and it panics rather than
    /// serving the corpus again.
    epoch: u64,
    /// Set by the two places that drop the ring's consumed prefix
    /// ([`Self::next_bytes`] and [`Self::skip_bytes`]); after that a rewind()
    /// can no longer reach the original first byte, and the eval's "same
    /// window every time" guarantee is over (see [`Self::rewind`]). It was
    /// declared, initialised `false`, asserted in `rewind` and never assigned,
    /// so the guard on that guarantee was dead code.
    drained: bool,
}

/// Read up to `limit` bytes from `files` EXACTLY as [`ByteStream`] would see
/// them - including the parquet column decode. Exists so the anchors tool
/// measures the same bytes the trainer trains on: reading a `.parquet` file
/// raw measures the thrift/snappy container (a 2.1 GB books shard measured
/// 7.006/6.759 BPB that way, i.e. the high-entropy "books" domain was pure
/// container noise), and two readers would drift apart silently.
pub fn read_bytes(files: &[PathBuf], limit: usize) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let mut tmp = vec![0u8; 1 << 20];
    let mut skipped: Vec<String> = Vec::new();
    'outer: for f in files {
        // COUNTED, not silent: a shard the loader cannot open is a hole in
        // the measurement (open_source already says which one on stderr).
        let mut src = match open_source(f) {
            Some(s) => s,
            None => {
                skipped.push(f.display().to_string());
                continue;
            }
        };
        loop {
            if limit > 0 && out.len() >= limit {
                break 'outer;
            }
            match src.read(&mut tmp) {
                // A read ERROR is not an EOF: treating it as one ends the
                // file early and the caller measures a corpus shorter than
                // the one it asked for, with no sign of it (ADR-0019).
                Ok(0) => break,
                Err(e) => {
                    skipped.push(format!("{}: read {e}", f.display()));
                    break;
                }
                Ok(n) => out.extend_from_slice(&tmp[..n]),
            }
        }
    }
    if !skipped.is_empty() {
        eprintln!(
            "read_bytes: {} of {} file(s) unreadable, the sample is short by them: {}",
            skipped.len(),
            files.len(),
            skipped.join("; ")
        );
    }
    // An empty sample measures NOTHING and the caller's BPB is then a
    // division by a corpus that was never read - loud, naming the cause.
    assert!(
        !out.is_empty(),
        "read_bytes: no bytes from {} file(s) (unreadable root, unmounted drive, or no data extensions?)",
        files.len()
    );
    if limit > 0 && out.len() > limit {
        out.truncate(limit);
    }
    out
}

/// Open one data file as a [`Source`], or `None` with the reason on stderr.
/// A parquet file that the arrow reader refuses is NOT quietly dropped: the
/// corpus is a list of shards and a missing one is a hole in every number
/// computed from it (ADR-0019 - the previous `.ok().and_then(..)` chain
/// skipped it invisibly).
fn open_source(path: &Path) -> Option<Source> {
    let is_parquet = path
        .extension()
        .map(|e| e.to_string_lossy().eq_ignore_ascii_case("parquet"))
        .unwrap_or(false);
    let r = if is_parquet {
        std::fs::File::open(path).and_then(|f| {
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(f)
                .and_then(|b| b.build())
                .map(|reader| Source::Parquet {
                    reader,
                    batch: Vec::new(),
                    pos: 0,
                    path: path.to_path_buf(),
                })
                .map_err(std::io::Error::other)
        })
    } else {
        std::fs::File::open(path).map(|f| Source::Text(BufReader::new(f)))
    };
    match r {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!("data: skipping unreadable shard {}: {e}", path.display());
            None
        }
    }
}

/// One line per non-null value, for whichever of arrow's three string physical
/// types the column happens to be. `LargeUtf8` is not an exotic choice: it is
/// what a writer emits for strings over 2 GB, and a corpus built with it used
/// to decode to nothing at all, silently.
fn push_strings(a: &dyn Array, out: &mut Vec<u8>) {
    use arrow::array::{LargeStringArray, StringArray, StringViewArray};
    macro_rules! lines {
        ($t:ty) => {
            if let Some(x) = a.as_any().downcast_ref::<$t>() {
                for v in x.iter().flatten() {
                    out.extend_from_slice(v.as_bytes());
                    out.push(b'\n');
                }
            }
        };
    }
    lines!(StringArray);
    lines!(LargeStringArray);
    lines!(StringViewArray);
}

/// Append every string the column TREE holds, in schema order, one per line.
///
/// This walk is by DATA TYPE, and it recurses. The decoder it replaces
/// downcast the top-level columns to `StringArray` and dropped everything else
/// **without a counter**, so a corpus whose text is nested one level down
/// (`document: struct { html: Utf8, ... }`, `annotations: list<struct { text:
/// Utf8 }>`) contributed almost nothing and the run reported a confident loss
/// curve over the wrong bytes: measured on `mix/qa`, 2 534 708 673 B on disk
/// decoded to 283 KB (0.011%) — the `id` column alone, which is the only
/// TOP-LEVEL string in that schema.
///
/// Nothing is synthesized: only string-typed columns are read, never a cast
/// from a number, so the stream is the corpus' own text. A corpus that stores
/// the same passage twice (HF's `document.html` AND `document.tokens.token`
/// both carry it) therefore appears twice — that is duplication the file has,
/// and choosing fields by NAME to avoid it would be a heuristic.
fn push_text(dt: &DataType, a: &dyn Array, out: &mut Vec<u8>) {
    use arrow::array::{
        Array, FixedSizeListArray, LargeListArray, ListArray, MapArray, StructArray,
    };
    use arrow::datatypes::DataType as D;
    match dt {
        D::Utf8 | D::LargeUtf8 | D::Utf8View => push_strings(a, out),
        // Dictionary-encoded strings: the physical array is the dictionary, so
        // the downcasts above cannot see a value. Cast to the dictionary's own
        // VALUE type (one record batch, bounded) and read that.
        D::Dictionary(_, v) if matches!(v.as_ref(), D::Utf8 | D::LargeUtf8 | D::Utf8View) => {
            if let Ok(c) = arrow::compute::cast(a, v) {
                push_strings(c.as_ref(), out);
            }
        }
        D::Struct(_) => {
            if let Some(x) = a.as_any().downcast_ref::<StructArray>() {
                for c in x.columns() {
                    push_text(c.data_type(), c.as_ref(), out);
                }
            }
        }
        D::List(_) => {
            if let Some(x) = a.as_any().downcast_ref::<ListArray>() {
                push_text(x.values().data_type(), x.values().as_ref(), out);
            }
        }
        D::LargeList(_) => {
            if let Some(x) = a.as_any().downcast_ref::<LargeListArray>() {
                push_text(x.values().data_type(), x.values().as_ref(), out);
            }
        }
        D::FixedSizeList(_, _) => {
            if let Some(x) = a.as_any().downcast_ref::<FixedSizeListArray>() {
                push_text(x.values().data_type(), x.values().as_ref(), out);
            }
        }
        // Keys as well as values: a map's key is text the file contains.
        D::Map(_, _) => {
            if let Some(x) = a.as_any().downcast_ref::<MapArray>() {
                push_text(x.entries().data_type(), x.entries(), out);
            }
        }
        _ => {}
    }
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
        path: PathBuf,
    },
}

impl Source {
    fn read(&mut self, tmp: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Source::Text(r) => r.read(tmp),
            Source::Parquet {
                reader,
                batch,
                pos,
                path,
            } => {
                if *pos >= batch.len() {
                    batch.clear();
                    *pos = 0;
                    // Pull batches until one yields text or the file ends. This
                    // used to `return Ok(batch.len())` WITHOUT copying into
                    // `tmp`, so both callers (`ByteStream::refill` and
                    // `read_bytes`) sliced `tmp[..n]` past its end: reading ANY
                    // parquet corpus panicked or ingested garbage. Found by
                    // running the anchors tool on mix/books (a .parquet shard)
                    // 2026-09-27. Fall through to the copy instead.
                    let mut got_text = false;
                    loop {
                        match reader.next() {
                            Some(Ok(record)) => {
                                let mut chunk = Vec::new();
                                for col in record.columns() {
                                    push_text(col.data_type(), col.as_ref(), &mut chunk);
                                }
                                if !chunk.is_empty() {
                                    std::mem::swap(batch, &mut chunk);
                                    got_text = true;
                                    break;
                                }
                            }
                            Some(Err(e)) => return Err(std::io::Error::other(e)),
                            None => break,
                        }
                    }
                    if !got_text {
                        // COUNTED, and it names the shard: read to the end and
                        // found no string anywhere is a hole in the corpus, the
                        // same class as a shard that will not open. It used to
                        // be silent, which is how a 2.4 GB corpus trained on
                        // 0.011% of its bytes with a clean log.
                        eprintln!(
                            "data: {} yielded NO text (read to the end, no string column this decoder walks): the stream is a hole where that shard was",
                            path.display()
                        );
                        return Ok(0);
                    }
                }
                let n = tmp.len().min(batch.len() - *pos);
                tmp[..n].copy_from_slice(&batch[*pos..*pos + n]);
                *pos += n;
                Ok(n)
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

    /// The training and held-out streams of ONE run, built together.
    ///
    /// "The eval tail never trains" (ADR-0010) is a property of the PAIR, not
    /// of a flag spelling, and it cannot be checked one stream at a time:
    /// [`collect_files`] recurses, so `--data` pointed at the parent of the
    /// eval directory collects the eval bytes as training data, the held-out
    /// numbers the run prints are optimistic by an unmeasured amount, and
    /// nothing anywhere says so. The filter-time split (documents routed by
    /// their start offset, straddlers dropped) is correct and is the only
    /// thing that makes the two trees disjoint — so this refuses, by name,
    /// when they are not.
    ///
    /// Build both streams with this rather than two [`Self::new`] calls: that
    /// is the shape the hole got in, and a caller that goes through here
    /// cannot forget the check.
    pub fn train_and_eval(
        seq_len: usize,
        batch: usize,
        data_root: &Path,
        eval_root: Option<&Path>,
    ) -> (Self, Option<Self>) {
        if let Some(ev) = eval_root {
            let d = std::fs::canonicalize(data_root)
                .unwrap_or_else(|e| panic!("data: --data {}: {e}", data_root.display()));
            let e = std::fs::canonicalize(ev)
                .unwrap_or_else(|err| panic!("data: --eval {}: {err}", ev.display()));
            assert!(
                !e.starts_with(&d) && !d.starts_with(&e),
                "NO-LEAK: the training tree {} and the eval tree {} are the same tree, or one \
                 contains the other, so the held-out bytes are inside the training bytes. Eval \
                 regions are excluded at FILTER time (filter --eval-output carves the tail into \
                 its own tree), never by a flag: point --data at the head-only tree, or --eval \
                 at a separate carve.",
                data_root.display(),
                ev.display()
            );
        }
        let train = Self::new(seq_len, batch, data_root);
        let eval = eval_root.map(|p| Self::new(seq_len, batch, p));
        (train, eval)
    }

    /// Stream over an explicit file list (e.g. a held-out split).
    pub fn from_files(seq_len: usize, batch: usize, files: Vec<PathBuf>, seed: u64) -> Self {
        assert!(!files.is_empty(), "ByteStream needs a non-empty file list");
        // A sleeping/unmounted drive lists files but serves no bytes; catch it
        // at construction instead of training on a silent constant stream.
        let total: u64 = files
            .iter()
            .map(|f| f.metadata().map(|m| m.len()).unwrap_or(0))
            .sum();
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

    /// Open a readable file at `file_idx` (skipping unreadable ones, but
    /// SAYING so: a shard that cannot be opened is a hole in the corpus, and
    /// a run that trains on 63 of 64 shards must not look like one that
    /// trains on 64). On total exhaustion, advance to the next epoch:
    /// reshuffle and restart from 0.
    fn ensure_reader(&mut self) -> bool {
        if self.reader.is_some() {
            return true;
        }
        while self.file_idx < self.files.len() {
            if let Some(r) = open_source(&self.files[self.file_idx]) {
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
            self.seed
                .wrapping_add(self.epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        );
        while self.file_idx < self.files.len() {
            if let Some(r) = open_source(&self.files[self.file_idx]) {
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
        let epoch0 = self.epoch;
        loop {
            if self.buf.len() - self.pos >= need {
                break;
            }
            if !self.ensure_reader() {
                break;
            }
            let read = {
                let path = self
                    .files
                    .get(self.file_idx)
                    .map(|p| p.display().to_string())
                    .unwrap_or_default();
                let r = self.reader.as_mut().unwrap();
                // An Err is NOT an EOF: `unwrap_or(0)` here used to end the
                // file early and move on, so a mid-file I/O error (or a bad
                // parquet row group) truncated the corpus with nothing in the
                // log - the training half of the `read_bytes` fix (ADR-0019).
                r.read(&mut tmp)
                    .unwrap_or_else(|e| panic!("data: read {path} failed: {e}"))
            };
            if read == 0 {
                self.reader = None;
                self.file_idx += 1;
                continue;
            }
            self.buf.extend_from_slice(&tmp[..read]);
        }
        assert!(
            self.epoch == epoch0,
            "data: the ring wanted {need} unread bytes and the corpus ran out first (wrapped to \
             epoch {}): the stream would serve the corpus AGAIN instead of advancing, so a resume \
             or a training stream would silently re-read bytes it has already trained on (the \
             pretrain-v2 collapse). A corpus smaller than 2 batches, or a drive that stopped \
             serving mid-fill.",
            self.epoch
        );
        assert!(
            !self.buf.is_empty(),
            "data stream dry: no readable bytes from {} file(s) (drive dropped mid-run?)",
            self.files.len()
        );
    }

    /// FNV n-gram hashes for the Engram memory, one column per entry of
    /// [`ORDERS`], REDUCED mod `tables[t]`.
    ///
    /// This is the host-RAM path's contract: the caller owns the tables
    /// (`HostNgram::slots`), so the reduction happens here and the host
    /// gather is a plain in-range index. For the in-VRAM path use
    /// [`Self::hashes_raw`] - the model masks the slot index itself, so its
    /// `engram_rows` config is the only copy of the capacity.
    pub fn hashes(&self, bytes: &[u8], tables: &[usize]) -> Vec<i64> {
        let mut out = self.hashes_raw(bytes);
        for (i, h) in out.iter_mut().enumerate() {
            *h = h.rem_euclid(tables[i % tables.len()] as i64);
        }
        out
    }

    /// FNV n-gram hashes for the Engram memory, one column per entry of
    /// [`ORDERS`], RAW (not reduced) — one [`raw_keys`] call per batch row,
    /// which is where the derivation (and the 31-bit truncation) lives. The
    /// low **31** bits of the FNV-1a digest of each context, in `[b, t, 3]`
    /// order.
    ///
    /// This is the in-VRAM path's contract: the model masks the slot index
    /// itself, so its `engram_rows` config is the only copy of the capacity.
    /// For the host-RAM path use [`Self::hashes`] — same keys, reduced mod the
    /// caller's table size.
    pub fn hashes_raw(&self, bytes: &[u8]) -> Vec<i64> {
        let mut out = Vec::with_capacity(self.batch * self.seq_len * ORDERS.len());
        for b in 0..self.batch {
            let base = b * self.seq_len;
            out.extend_from_slice(&raw_keys(&bytes[base..base + self.seq_len]));
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
            // LOUD, where this used to `break`: a skip that runs off the end of
            // the corpus used to exit here having skipped LESS than asked, with
            // nothing said, and the run resumed at the head of the corpus it
            // had already trained on - the pretrain-v2 collapse. `refill` now
            // panics on the wrap itself; this catches the case where it ever
            // returns empty, rather than spinning or silently short-skipping.
            assert!(
                buffered > 0,
                "skip_bytes({n}) past the end of the corpus: the ring came back empty, so the \
                 resume would land earlier than the checkpoint says"
            );
            let take = buffered.min(n);
            self.pos += take as usize;
            n -= take;
            if self.pos > self.capacity / 2 {
                self.buf.drain(0..self.pos);
                self.pos = 0;
                self.drained = true;
            }
        }
    }

    /// Next `(bytes, hashes)` batch. Falls back to padding when data is short.
    /// RAW hashes (the in-VRAM path: the model owns the capacity and masks
    /// the slot index itself, so there is no second copy of the row count
    /// here). See [`Self::hashes`].
    pub fn next_batch(&mut self) -> (Vec<u8>, Vec<i64>) {
        let bytes = self.next_bytes();
        let raw = self.hashes_raw(&bytes);
        (bytes, raw)
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
        let bytes = self.next_bytes();
        let hashes = self.hashes(&bytes, &tables);
        (bytes, hashes)
    }

    /// The next `batch * seq_len` bytes, advancing the stream.
    fn next_bytes(&mut self) -> Vec<u8> {
        let need = self.batch * self.seq_len;
        if self.pos + need > self.buf.len() {
            self.refill();
            // No rewind here. This used to reset `pos = 0` when the refill
            // still left a short buffer, which reads as "recover from a short
            // buffer" and is the pretrain-v2 collapse on the training path:
            // the stream would serve the head of the ring again and the run
            // would re-train on bytes it had already seen. It was also dead -
            // `refill` returns only with `>= 2 * need` unread bytes, or
            // panics - which is the real reason the collapse never happened
            // and the reason nobody noticed the line. The `short read` assert
            // below is the loud failure that replaces it.
        }
        let end = (self.pos + need).min(self.buf.len());
        let bytes = self.buf[self.pos..end].to_vec();
        self.pos = end;
        // Compact consumed prefix to keep memory bounded. This is the event
        // `drained` documents, so it is the event that sets it: after the
        // ring drops its head, `rewind` cannot reach the original first byte
        // and the "same window every eval" guarantee is void.
        if self.pos > self.capacity / 2 {
            self.buf.drain(0..self.pos);
            self.pos = 0;
            self.drained = true;
        }
        assert!(
            bytes.len() == need,
            "short read: {} of {} bytes (corpus smaller than one batch?)",
            bytes.len(),
            need
        );
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("dormouse_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    #[should_panic(expected = "no readable data files")]
    fn missing_root_is_loud() {
        ByteStream::new(8, 2, Path::new("/nonexistent/path/xyz"));
    }

    /// The two hash contracts, on the same batch. `hashes_raw` (the in-VRAM
    /// path) is the low 32 bits of the FNV digest, one column per
    /// `ORDERS` entry, NOT reduced - the model masks the slot index against
    /// its own `engram_rows`, so the row count lives in exactly one place.
    /// `hashes` (the host-RAM path) is the same thing reduced mod the
    /// caller's table size, because the host gathers the row immediately.
    #[test]
    fn raw_hashes_are_unreduced_and_reduction_is_the_callers() {
        let dir = std::env::temp_dir().join(format!("dormouse_hash_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pattern: Vec<u8> = (0..256u32)
            .map(|i| (i * 29 % 251) as u8)
            .cycle()
            .take(1 << 16)
            .collect();
        std::fs::write(dir.join("corpus.bin"), &pattern).unwrap();
        let (seq, batch) = (8usize, 2usize);
        let s = ByteStream::new(seq, batch, &dir);
        let raw = s.hashes_raw(&pattern[..seq * batch]);
        assert_eq!(raw.len(), batch * seq * ORDERS.len());
        // `hashes_raw` IS `raw_keys` per batch row, one row at a time. Pinned
        // because the trainer and the decode seam must derive the same key for
        // the same bytes: the seam has no ByteStream, and a second copy of the
        // FNV here is a second thing that can drift.
        for b in 0..batch {
            let row = &pattern[b * seq..(b + 1) * seq];
            let one = raw_keys(row);
            assert_eq!(one.len(), seq * ORDERS.len());
            assert_eq!(
                &raw[b * seq * ORDERS.len()..(b + 1) * seq * ORDERS.len()],
                &one[..]
            );
        }
        // Unreduced, and 31 bits so `Int` (i32 on burn-flex) holds every value
        // without a panic. This assertion CHANGED on 2026-09-28: the old
        // version required the corpus to produce a NEGATIVE i32, i.e. it
        // pinned the exact 32-bit behaviour that made every training run die
        // with "Element cannot be represented in the target type". Its purpose
        // — raw is unreduced, the reduction belongs to the caller — is kept; its
        // bit width is corrected.
        assert!(
            raw.iter().all(|&h| (0..(1i64 << 31)).contains(&h)),
            "raw hashes fit in i32 (the type `Int` actually is on burn-flex)"
        );
        assert!(
            raw.iter().any(|&h| h > 4096),
            "raw hashes must NOT be pre-reduced"
        );
        assert!(
            raw.iter().all(|&h| (h as i32) >= 0),
            "every raw hash must survive the i32 cast the trainer performs"
        );

        // The reduction is exactly the caller's table size, applied per column.
        let tables = [4096usize, 100_000, 524_288];
        let red = s.hashes(&pattern[..seq * batch], &tables);
        assert_eq!(red.len(), raw.len());
        for (i, (&r, &g)) in red.iter().zip(raw.iter()).enumerate() {
            assert_eq!(
                r,
                g.rem_euclid(tables[i % ORDERS.len()] as i64),
                "column {i}"
            );
        }
        assert!(red.iter().all(|&h| (0..524_288).contains(&h)));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The order list the hashes are built from, spelled out: 3 tables (the
    /// trainer's hash tensor is `[b, t, 3]`), orders 2/3/4 ascending, no
    /// order deep enough to be a dead arm (n=5 at 500K rows averaged ~2.2M
    /// distinct 5-grams per row).
    #[test]
    fn orders_are_the_evidence_backed_list() {
        assert_eq!(ORDERS, [2, 3, 4]);
        let mut sorted = ORDERS;
        sorted.sort_unstable();
        assert_eq!(
            sorted, ORDERS,
            "orders must be ascending (table order == column order)"
        );
    }

    #[test]
    #[should_panic(expected = "non-empty file list")]
    fn empty_file_list_is_loud() {
        ByteStream::from_files(8, 2, Vec::new(), 1);
    }

    /// A read ERROR is not an EOF, and must not become one. `refill` used to
    /// `unwrap_or(0)` the read result, so an unreadable shard ended the file
    /// at byte 0, the stream moved on, and the run trained on whatever the
    /// NEXT shard held - no error, no log line, a corpus quietly smaller
    /// than the one that was configured (ADR-0019).
    ///
    /// A directory named `corpus.bin` opens fine and fails on the first read
    /// (EISDIR), which is exactly the shape of a bad drive, so the panic has
    /// to name the read failure rather than pretend the file ended. The batch
    /// is tiny so the size floor (which counts a directory's 40 bytes) passes
    /// and the READ is what is under test.
    #[test]
    #[should_panic(expected = "read ")]
    fn a_shard_that_fails_to_read_is_not_a_silent_eof() {
        let dir = std::env::temp_dir().join(format!("dormouse_read_err_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::create_dir_all(dir.join("corpus.bin")).unwrap();
        let _ = ByteStream::from_files(2, 2, vec![dir.join("corpus.bin")], 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `read_bytes` measured NOTHING when every file it was handed failed to
    /// open, and returned an empty vec that the anchors tool then divided by:
    /// a BPB for a corpus that was never read. It must refuse (ADR-0019).
    #[test]
    #[should_panic(expected = "read_bytes: no bytes")]
    fn an_empty_sample_is_loud_not_a_zero_byte_corpus() {
        let _ = read_bytes(&[PathBuf::from("/nonexistent/path/shard.txt")], 1 << 10);
    }

    #[test]
    fn skip_bytes_resumes_exactly() {
        let dir = std::env::temp_dir().join(format!("dormouse_skip_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pattern: Vec<u8> = (0..256u32)
            .map(|i| (i * 7 % 251) as u8)
            .cycle()
            .take(1 << 20)
            .collect();
        let f = dir.join("corpus.bin");
        std::fs::write(&f, &pattern).unwrap();
        let (seq, batch) = (8usize, 2usize);
        let mut s = ByteStream::new(seq, batch, &dir);
        let skip = 1000u64;
        s.skip_bytes(skip);
        let (bytes, _) = s.next_batch();
        assert_eq!(bytes.len(), seq * batch);
        assert_eq!(
            bytes[..],
            pattern[skip as usize..skip as usize + seq * batch]
        );
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
        let pattern: Vec<u8> = (0..256u32)
            .map(|i| (i * 13 % 241) as u8)
            .cycle()
            .take(1 << 20)
            .collect();
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

    /// The ring dropping its consumed prefix is the one event that voids the
    /// fixed-eval-window guarantee, and `rewind()` is what must refuse after
    /// it. `drained` was declared, initialised `false`, asserted in `rewind`
    /// and never assigned, so this guard did not exist: an eval stream big
    /// enough to outgrow half the 64 MB ring (the 500 MB milestone eval does)
    /// would have silently scored a different window every eval.
    #[test]
    #[should_panic(expected = "fixed eval window")]
    fn rewind_after_the_ring_drops_its_head_is_loud() {
        let dir = tmpdir("drained");
        // Sparse: 34 MB of file the size check counts, without 34 MB of writes.
        std::fs::File::create(dir.join("corpus.bin"))
            .unwrap()
            .set_len(34 << 20)
            .unwrap();
        let mut s = ByteStream::new(8, 2, &dir);
        s.rewind(); // fine before the ring compacts
        s.skip_bytes((33 << 20) as u64); // past capacity/2 -> the prefix is dropped
        s.rewind(); // must not be silent
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A resume must not land EARLIER than the checkpoint says. `skip_bytes`
    /// used to `break` out of its loop when the corpus ran out, so it skipped
    /// less than asked and the run went on to re-train from the head of the
    /// corpus - the pretrain-v2 collapse, with nothing in the log. It is loud
    /// now, from `refill` (the ring cannot be filled without wrapping into the
    /// next epoch).
    #[test]
    #[should_panic(expected = "wrapped to epoch")]
    fn a_skip_past_the_end_of_the_corpus_is_loud() {
        let dir = tmpdir("skip_past_end");
        let pattern: Vec<u8> = (0..256u32)
            .map(|i| (i * 7 % 251) as u8)
            .cycle()
            .take(1 << 20)
            .collect();
        std::fs::write(dir.join("corpus.bin"), &pattern).unwrap();
        let mut s = ByteStream::new(8, 2, &dir);
        s.skip_bytes(2 << 20); // the corpus holds 1 MiB
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The eval tree must not be reachable from the training tree, by name,
    /// or not at all. `collect_files` recurses, so `--data` at the parent of
    /// the eval directory trained on the eval bytes and every held-out number
    /// was optimistic by an unmeasured amount (ADR-0010). Filter-time routing
    /// is the real fix and it is already there; this is the assertion that
    /// the two trees the run was GIVEN are disjoint.
    #[test]
    #[should_panic(expected = "NO-LEAK")]
    fn a_data_root_containing_the_eval_tree_is_loud() {
        let dir = tmpdir("leak");
        let sub = dir.join("pretrain");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("corpus.bin"), vec![7u8; 1 << 16]).unwrap();
        std::fs::write(sub.join("eval_tail.bin"), vec![9u8; 1 << 16]).unwrap();
        let _ = ByteStream::train_and_eval(8, 2, &dir, Some(&sub));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The escape: disjoint trees, and both streams work. The same tree for
    /// both must also refuse (it is the `starts_with` case above), and no eval
    /// root at all is the plain training case.
    #[test]
    fn disjoint_data_and_eval_trees_are_the_normal_case() {
        let dir = tmpdir("no_leak_ok");
        for (name, byte) in [("data", 7u8), ("eval", 9u8)] {
            std::fs::create_dir_all(dir.join(name)).unwrap();
            std::fs::write(dir.join(name).join("shard.bin"), vec![byte; 1 << 16]).unwrap();
        }
        let (mut train, eval) =
            ByteStream::train_and_eval(8, 2, &dir.join("data"), Some(&dir.join("eval")));
        let (bytes, _) = train.next_batch();
        assert_eq!(bytes, vec![7u8; 16]);
        let mut eval = eval.expect("an eval root must yield a stream");
        let (first, _) = eval.next_batch();
        assert_eq!(first, vec![9u8; 16]);
        eval.rewind();
        let (again, _) = eval.next_batch();
        assert_eq!(first, again, "the eval window must be fixed");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// One definition of "the files the trainer sees", so a second reader of
    /// the same directory cannot disagree by construction. The anchors tool
    /// used to walk with no extension filter at all, which on `real_eval/`
    /// (which holds `eval_tail.bin.30m.bak`) reported "2 files" and would
    /// fold a 30 MB pre-carve backup into a measurement whose trainer-side
    /// stream sees one file.
    #[test]
    fn collect_files_is_the_trainer_side_file_set() {
        let dir = tmpdir("collect");
        for name in [
            "corpus.bin",
            "shard-0001.parquet",
            "notes.md",
            "eval_tail.bin.30m.bak",
            "README",
        ] {
            std::fs::write(dir.join(name), b"x").unwrap();
        }
        std::fs::create_dir_all(dir.join("nested")).unwrap();
        std::fs::write(dir.join("nested").join("a.txt"), b"x").unwrap();
        let names: Vec<String> = collect_files(&dir)
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            vec!["corpus.bin", "a.txt", "notes.md", "shard-0001.parquet"]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A corpus whose text is NESTED must yield that text. The decoder used to
    /// downcast the top-level columns to `StringArray` and drop everything else
    /// with no counter, so `mix/qa` — `id: Utf8`, `document: struct { html,
    /// title, ... }`, `annotations: list<struct { short_answers: ... }>` — gave
    /// up 2 534 708 673 B on disk and 283 KB to the loader (0.011%: the `id`
    /// column, the only top-level string), and the run trained on ids while
    /// reporting a confident loss curve.
    #[test]
    fn nested_parquet_text_is_not_dropped() {
        use arrow::array::{
            ArrayRef, DictionaryArray, Int32Array, ListArray, StringArray, StructArray,
        };
        use arrow::buffer::OffsetBuffer;
        use arrow::datatypes::{Field, Fields, Schema};
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        let dir = tmpdir("pq_nested");
        let f = dir.join("qa.parquet");
        let strs = |v: &[&str]| -> ArrayRef { Arc::new(StringArray::from(v.to_vec())) };
        let list = |field: Field, offsets: Vec<i32>, values: ArrayRef| -> ArrayRef {
            Arc::new(ListArray::new(
                Arc::new(field),
                OffsetBuffer::new(offsets.into()),
                values,
                None,
            ))
        };
        // annotations: list<struct { short_answers: list<struct { text }> }>
        let answer = Arc::new(StructArray::new(
            Fields::from(vec![Field::new("text", DataType::Utf8, true)]),
            vec![strs(&["blue whale", "blue whale"])],
            None,
        ));
        let short_answers = list(
            Field::new(
                "item",
                DataType::Struct(Fields::from(vec![Field::new("text", DataType::Utf8, true)])),
                true,
            ),
            vec![0, 1, 2],
            answer as ArrayRef,
        );
        let ann_item = Arc::new(StructArray::new(
            Fields::from(vec![Field::new(
                "short_answers",
                short_answers.data_type().clone(),
                true,
            )]),
            vec![short_answers],
            None,
        ));
        let annotations = list(
            Field::new("item", ann_item.data_type().clone(), true),
            vec![0, 1, 2],
            ann_item as ArrayRef,
        );
        // document: struct { html: Utf8, title: Utf8 }
        let document = Arc::new(StructArray::new(
            Fields::from(vec![
                Field::new("html", DataType::Utf8, true),
                Field::new("title", DataType::Utf8, true),
            ]),
            vec![
                strs(&["<html>the whale</html>", "<html>a fin</html>"]),
                strs(&["whale", "fin"]),
            ],
            None,
        ));
        // tags: dictionary-encoded string - the physical array is a dictionary,
        // so a downcast to StringArray cannot see it at all.
        let tags = Arc::new(
            DictionaryArray::try_new(
                Int32Array::from(vec![1, 0]),
                strs(&["marine biology", "cephalopod"]),
            )
            .unwrap(),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, true),
            Field::new("document", document.data_type().clone(), true),
            Field::new("annotations", annotations.data_type().clone(), true),
            Field::new("tags", tags.data_type().clone(), true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                strs(&["row-0", "row-1"]),
                document as ArrayRef,
                annotations,
                tags as ArrayRef,
            ],
        )
        .unwrap();
        let mut w =
            parquet::arrow::ArrowWriter::try_new(std::fs::File::create(&f).unwrap(), schema, None)
                .unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();

        let out = read_bytes(&[f], 0);
        let s = String::from_utf8(out).unwrap();
        for want in [
            "row-0",
            "<html>the whale</html>",
            "whale",
            "blue whale",
            "marine biology",
        ] {
            assert!(s.contains(want), "nested text {want:?} missing from {s:?}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The other half of the same rule: a numeric column is NEVER read as
    /// text. Casting a number to a string would synthesize bytes the corpus
    /// does not contain, which is the one thing the decoder must not do.
    #[test]
    #[should_panic(expected = "no bytes")]
    fn a_parquet_of_numbers_yields_no_invented_text() {
        use arrow::array::{ArrayRef, Int64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;
        let dir = tmpdir("pq_numbers");
        let f = dir.join("n.parquet");
        let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, true)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![1i64, 2, 3])) as ArrayRef],
        )
        .unwrap();
        let mut w =
            parquet::arrow::ArrowWriter::try_new(std::fs::File::create(&f).unwrap(), schema, None)
                .unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
        let _ = read_bytes(&[f], 0);
        std::fs::remove_dir_all(&dir).ok();
    }
}
