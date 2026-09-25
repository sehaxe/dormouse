//! filter - pure-Rust quality filter for the dormouse training corpus.
//!
//! Scores blank-line-separated documents against a fastText quality
//! classifier (the DCLM `mlfoundations/fasttext-oh-eli5` model behind
//! arXiv:2406.11794's DCLM-Baseline), exact-dedups by a 128-bit hash of the
//! document bytes, and streams the kept bytes to the output in original
//! order. std only, no new dependencies.
//!
//! fastText inference is implemented from the C++ reference
//! (facebookresearch/fastText, MIT):
//! - `.bin` layout per `fasttext.cc` saveModel/loadModel:
//!   magic i32 (793712314), version i32 (<=12), Args (13 fields, all i32
//!   except trailing f64 `t`), Dictionary, quant bool, input DenseMatrix,
//!   qout bool, output DenseMatrix.
//! - Dictionary per `dictionary.cc`: entries (nul-terminated word, i64
//!   count, i8 type), word hashes are SIGNED FNV-1a (the source pins this:
//!   `h = h ^ uint32_t(int8_t(str[i]))` for released-model compatibility).
//! - Inference path per dclm's `quality_prediction_enrichers_calc_fasttext.py`:
//!   `text = " ".join(content.strip().splitlines())` fed to python
//!   `model.predict`, which streams the whole string as ONE line. So:
//!   whitespace-only tokenization (C++ `readWord` separators plus python
//!   `splitlines` chars), no lowercasing, no punctuation splitting, and no
//!   EOS token (the stringstream has no trailing newline).
//! - Scoring per `model.cc`/`loss.cc`: hidden = mean of input-matrix rows
//!   over all ids (vocab hits land in [0,nwords), hashed word n-grams and
//!   char n-grams in [nwords, nwords+bucket)), logits = W_out * hidden,
//!   softmax (or per-label sigmoid for ns/ova models). Score = P(__label__hq)
//!   (+1e-5, matching python predict's round-trip through std_log/exp).

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::time::Instant;

// ---------------------------------------------------------------------------
// fastText model
// ---------------------------------------------------------------------------

const FT_MAGIC: i32 = 793712314;
const FT_VERSION: i32 = 12;

#[derive(Debug, Clone, Copy)]
pub struct FtArgs {
    pub dim: i32,
    pub ws: i32,
    pub epoch: i32,
    pub min_count: i32,
    pub neg: i32,
    pub word_ngrams: i32,
    pub loss: i32,
    pub model: i32,
    pub bucket: i32,
    pub minn: i32,
    pub maxn: i32,
    pub lr_update_rate: i32,
    pub t: f64,
}

pub struct FtModel {
    pub args: FtArgs,
    /// (word bytes, count, type) per dictionary entry; type 0 = word, 1 = label.
    pub words: Vec<(Vec<u8>, i64, u8)>,
    pub word2id: HashMap<Vec<u8>, i32>,
    pub nwords: i32,
    pub nlabels: i32,
    /// pruneidx_size_ from the file (-1 = never pruned).
    prune_size: i64,
    prune_map: HashMap<i32, i32>,
    /// Dense input matrix, row-major, (nwords + bucket) x dim.
    input: Vec<f32>,
    /// Dense output matrix, row-major, nlabels x dim.
    output: Vec<f32>,
    /// Output-matrix row for `__label__hq`.
    pub hq_row: usize,
    /// Per-vocab-word subword id lists (only when maxn > 0).
    subwords: Vec<Vec<i32>>,
}

pub enum LoadError {
    Io(String),
    Format(String),
}

impl fmt::Debug for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io(s) => write!(f, "io error: {s}"),
            LoadError::Format(s) => write!(f, "bad fastText model: {s}"),
        }
    }
}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::Io(e.to_string())
    }
}

struct Cursor<R: Read> {
    inner: R,
}

impl<R: Read> Cursor<R> {
    fn i32(&mut self) -> Result<i32, LoadError> {
        let mut b = [0u8; 4];
        self.inner.read_exact(&mut b)?;
        Ok(i32::from_le_bytes(b))
    }
    fn i64(&mut self) -> Result<i64, LoadError> {
        let mut b = [0u8; 8];
        self.inner.read_exact(&mut b)?;
        Ok(i64::from_le_bytes(b))
    }
    fn f64(&mut self) -> Result<f64, LoadError> {
        let mut b = [0u8; 8];
        self.inner.read_exact(&mut b)?;
        Ok(f64::from_le_bytes(b))
    }
    fn u8(&mut self) -> Result<u8, LoadError> {
        let mut b = [0u8; 1];
        self.inner.read_exact(&mut b)?;
        Ok(b[0])
    }
    fn bytes(&mut self, n: usize, buf: &mut Vec<u8>) -> Result<(), LoadError> {
        buf.resize(n, 0);
        self.inner.read_exact(buf)?;
        Ok(())
    }
}

impl FtModel {
    pub fn load(path: &str) -> Result<FtModel, LoadError> {
        let f = File::open(path).map_err(|e| LoadError::Format(format!("open {path}: {e}")))?;
        let mut c = Cursor { inner: BufReader::with_capacity(1 << 20, f) };

        let magic = c.i32()?;
        if magic != FT_MAGIC {
            return Err(LoadError::Format(format!(
                "bad magic {magic} (expected {FT_MAGIC}); not a dense fastText .bin"
            )));
        }
        let version = c.i32()?;
        if version > FT_VERSION {
            return Err(LoadError::Format(format!("unsupported version {version}")));
        }

        let mut args = FtArgs {
            dim: c.i32()?,
            ws: c.i32()?,
            epoch: c.i32()?,
            min_count: c.i32()?,
            neg: c.i32()?,
            word_ngrams: c.i32()?,
            loss: c.i32()?,
            model: c.i32()?,
            bucket: c.i32()?,
            minn: c.i32()?,
            maxn: c.i32()?,
            lr_update_rate: c.i32()?,
            t: c.f64()?,
        };
        // loadModel backward compat: version-11 supervised models have no char ngrams.
        if version == 11 && args.model == 3 {
            args.maxn = 0;
        }
        if args.model != 3 {
            return Err(LoadError::Format(format!(
                "model kind {} is not supervised (sup=3)",
                args.model
            )));
        }
        if args.loss != 3 && args.loss != 2 && args.loss != 4 {
            return Err(LoadError::Format(format!(
                "loss {} is not softmax/ns/ova; hierarchical softmax (hs) predict is not implemented",
                args.loss
            )));
        }

        // Dictionary
        let size = c.i32()?;
        let nwords = c.i32()?;
        let nlabels = c.i32()?;
        let _ntokens = c.i64()?;
        let prune_size = c.i64()?;
        let mut words: Vec<(Vec<u8>, i64, u8)> = Vec::with_capacity(size.max(0) as usize);
        let mut word2id: HashMap<Vec<u8>, i32> = HashMap::with_capacity(size.max(0) as usize);
        let mut tmp: Vec<u8> = Vec::new();
        for i in 0..size {
            tmp.clear();
            loop {
                let b = c.u8()?;
                if b == 0 {
                    break;
                }
                tmp.push(b);
            }
            let count = c.i64()?;
            let ty = c.u8()?;
            word2id.insert(tmp.clone(), i);
            words.push((tmp.clone(), count, ty));
        }
        let mut prune_map: HashMap<i32, i32> = HashMap::new();
        if prune_size > 0 {
            for _ in 0..prune_size {
                let k = c.i32()?;
                let v = c.i32()?;
                prune_map.insert(k, v);
            }
        }

        // quant flag: this implementation only handles dense matrices.
        let quant = c.u8()?;
        if quant != 0 {
            return Err(LoadError::Format(
                "model is quantized (QuantMatrix); dense-only implementation".into(),
            ));
        }

        // Input DenseMatrix: i64 rows, i64 cols, rows*cols f32 row-major.
        let in_m = c.i64()?;
        let in_n = c.i64()?;
        let expect_m = nwords as i64 + args.bucket as i64;
        if in_n != args.dim as i64 {
            return Err(LoadError::Format(format!(
                "input cols {in_n} != args dim {}",
                args.dim
            )));
        }
        if in_m != expect_m {
            return Err(LoadError::Format(format!(
                "input rows {in_m} != nwords({nwords}) + bucket({})",
                args.bucket
            )));
        }
        let input = read_dense(&mut c, in_m as usize, in_n as usize)?;

        // qout flag
        let qout = c.u8()?;
        if qout != 0 {
            return Err(LoadError::Format(
                "quantized output matrix (qout); dense-only implementation".into(),
            ));
        }
        let out_m = c.i64()?;
        let out_n = c.i64()?;
        if out_m != nlabels as i64 || out_n != args.dim as i64 {
            return Err(LoadError::Format(format!(
                "output matrix {out_m}x{out_n} != nlabels({nlabels}) x dim({})",
                args.dim
            )));
        }
        let output = read_dense(&mut c, out_m as usize, out_n as usize)?;

        // Locate __label__hq among the label entries (label row i sits at
        // dictionary index nwords + i).
        let mut hq_row: Option<usize> = None;
        let mut label_names: Vec<String> = Vec::new();
        for i in 0..nlabels as usize {
            let w = &words[nwords as usize + i].0;
            label_names.push(String::from_utf8_lossy(w).into_owned());
            if w.as_slice() == b"__label__hq" {
                hq_row = Some(i);
            }
        }
        let hq_row = hq_row.ok_or_else(|| {
            LoadError::Format(format!("no __label__hq among labels {label_names:?}"))
        })?;

        let mut m = FtModel {
            args,
            words,
            word2id,
            nwords,
            nlabels,
            prune_size,
            prune_map,
            input,
            output,
            hq_row,
            subwords: Vec::new(),
        };
        if m.args.maxn > 0 {
            m.init_ngrams();
        }
        Ok(m)
    }

    /// Dictionary::initNgrams: subword ids for in-vocab words (word id first,
    /// then char n-grams of BOW+word+EOW; EOS gets none).
    fn init_ngrams(&mut self) {
        let mut sub = Vec::with_capacity(self.words.len());
        for (i, (w, _, ty)) in self.words.iter().enumerate() {
            let mut v = Vec::new();
            if *ty == 0 {
                v.push(i as i32);
                if w.as_slice() != b"</s>" {
                    let mut wrapped = Vec::with_capacity(w.len() + 2);
                    wrapped.push(b'<');
                    wrapped.extend_from_slice(w);
                    wrapped.push(b'>');
                    self.compute_subwords_into(&wrapped, &mut v);
                }
            }
            sub.push(v);
        }
        self.subwords = sub;
    }

    /// Dictionary::pushHash: prune-aware id mapping into the bucket space.
    fn push_hash(&self, out: &mut Vec<i32>, id: i32) {
        if self.prune_size == 0 || id < 0 {
            return;
        }
        let id = if self.prune_size > 0 {
            match self.prune_map.get(&id) {
                Some(&v) => v,
                None => return,
            }
        } else {
            id
        };
        out.push(self.nwords.wrapping_add(id));
    }

    /// Dictionary::computeSubwords over UTF-8 bytes (continuation bytes glue
    /// onto the preceding lead byte; n counts codepoints).
    fn compute_subwords_into(&self, word: &[u8], out: &mut Vec<i32>) {
        let bucket = self.args.bucket as u32;
        let (minn, maxn) = (self.args.minn, self.args.maxn);
        let nb = word.len();
        for i in 0..nb {
            if (word[i] & 0xC0) == 0x80 {
                continue;
            }
            let mut j = i;
            let mut n = 1i32;
            while j < nb && n <= maxn {
                j += 1;
                while j < nb && (word[j] & 0xC0) == 0x80 {
                    j += 1;
                }
                if n >= minn && !(n == 1 && (i == 0 || j == nb)) {
                    let h = (ft_hash(&word[i..j]) % bucket) as i32;
                    self.push_hash(out, h);
                }
                n += 1;
            }
        }
    }

    /// Dictionary::addSubwords for one token.
    fn add_subwords(&self, line: &mut Vec<i32>, token: &[u8], wid: Option<i32>) {
        match wid {
            None => {
                // OOV: char n-grams of <token>, unless it is EOS.
                if token != b"</s>" && self.args.maxn > 0 {
                    let mut wrapped = Vec::with_capacity(token.len() + 2);
                    wrapped.push(b'<');
                    wrapped.extend_from_slice(token);
                    wrapped.push(b'>');
                    self.compute_subwords_into(&wrapped, line);
                }
            }
            Some(wid) => {
                if self.args.maxn <= 0 {
                    line.push(wid);
                } else {
                    line.extend_from_slice(&self.subwords[wid as usize]);
                }
            }
        }
    }

    /// Dictionary::addWordNgrams. C++ accumulates in uint64 after SIGN-
    /// extending each int32 hash, multiplies by 116049371, mods by bucket.
    fn add_word_ngrams(&self, line: &mut Vec<i32>, hashes: &[i32]) {
        let n = self.args.word_ngrams as i32;
        if n <= 1 {
            return;
        }
        for i in 0..hashes.len() {
            let mut h = hashes[i] as i64 as u64;
            let mut j = i + 1;
            while j < hashes.len() && (j as i32) < i as i32 + n {
                h = h
                    .wrapping_mul(116049371)
                    .wrapping_add(hashes[j] as i64 as u64);
                let id = (h % self.args.bucket as u64) as i32;
                self.push_hash(line, id);
                j += 1;
            }
        }
    }

    /// Full getLine + predict for one document (whole doc = one line).
    /// Returns P(__label__hq), or None when the line has no input ids.
    pub fn score(&self, doc: &str) -> Option<f32> {
        // dclm: " ".join(content.strip().splitlines()); strip with python's
        // whitespace set, then tokenize on separators.
        let doc = py_strip(doc);
        let mut line: Vec<i32> = Vec::with_capacity(256);
        let mut hashes: Vec<i32> = Vec::with_capacity(128);
        for tok in doc.split(is_sep).filter(|t| !t.is_empty()) {
            let tb = tok.as_bytes();
            let h = ft_hash(tb);
            let wid = self.word2id.get(tb).copied();
            let is_word = match wid {
                Some(id) => self.words[id as usize].2 == 0,
                None => !tok.starts_with("__label__"),
            };
            if is_word {
                self.add_subwords(&mut line, tb, wid);
                hashes.push(h as i32);
            }
            // label tokens in the input contribute nothing to `words`
        }
        self.add_word_ngrams(&mut line, &hashes);
        if line.is_empty() {
            return None;
        }
        Some(self.predict_hq(&line))
    }

    /// Model::computeHidden + loss predict. All f32, sequential accumulation
    /// (Rust does not reassociate float reductions, so this matches the C++
    /// up to libm exp differences).
    fn predict_hq(&self, line: &[i32]) -> f32 {
        let d = self.args.dim as usize;
        let mut hidden = vec![0f32; d];
        for &id in line {
            let base = id as usize * d;
            let row = &self.input[base..base + d];
            for (h, &v) in hidden.iter_mut().zip(row.iter()) {
                *h += v;
            }
        }
        let scale = (1.0f64 / line.len() as f64) as f32;
        for h in hidden.iter_mut() {
            *h *= scale;
        }
        let osz = self.nlabels as usize;
        let mut out = vec![0f32; osz];
        for (i, o) in out.iter_mut().enumerate() {
            let row = &self.output[i * d..(i + 1) * d];
            let mut s = 0f32;
            for (h, &v) in hidden.iter().zip(row.iter()) {
                s += h * v;
            }
            *o = s;
        }
        let p = match self.args.loss {
            3 => {
                // SoftmaxLoss::computeOutput
                let mut max = out[0];
                for &v in &out {
                    if v > max {
                        max = v;
                    }
                }
                let mut z = 0f32;
                for v in out.iter_mut() {
                    *v = (*v - max).exp();
                    z += *v;
                }
                for v in out.iter_mut() {
                    *v /= z;
                }
                out[self.hq_row]
            }
            _ => {
                // BinaryLogisticLoss::computeOutput uses the 512-entry sigmoid
                // table; replicate it.
                sigmoid_table(out[self.hq_row])
            }
        };
        // python predict returns exp(std_log(p)) == p + 1e-5.
        p + 1e-5
    }
}

/// DenseMatrix::load: i64 rows, i64 cols, rows*cols little-endian f32,
/// streamed through a small buffer so peak memory is the matrix itself.
fn read_dense<R: Read>(c: &mut Cursor<R>, m: usize, n: usize) -> Result<Vec<f32>, LoadError> {
    let count = m
        .checked_mul(n)
        .ok_or_else(|| LoadError::Format("matrix size overflow".into()))?;
    let mut mat = vec![0f32; count];
    let mut chunk = vec![0u8; 1 << 20];
    let mut off = 0usize;
    while off < count {
        let take = chunk.len().min((count - off) * 4);
        c.bytes(take, &mut chunk)?;
        for (dst, src) in mat[off..off + take / 4]
            .iter_mut()
            .zip(chunk[..take].as_chunks::<4>().0)
        {
            *dst = f32::from_le_bytes(*src);
        }
        off += take / 4;
    }
    Ok(mat)
}

/// Dictionary::hash: FNV-1a with SIGNED byte extension (pinned by fastText
/// for released-model compatibility).
#[inline]
pub fn ft_hash(s: &[u8]) -> u32 {
    let mut h: u32 = 2166136261;
    for &b in s {
        h ^= b as i8 as i32 as u32;
        h = h.wrapping_mul(16777619);
    }
    h
}

/// Loss::sigmoid lookup table (SIGMOID_TABLE_SIZE=512, MAX_SIGMOID=8),
/// reconstructed instead of tabulated.
fn sigmoid_table(x: f32) -> f32 {
    const SIZE: f32 = 512.0;
    const MAX_SIGMOID: f32 = 8.0;
    if x < -MAX_SIGMOID {
        0.0
    } else if x > MAX_SIGMOID {
        1.0
    } else {
        let idx = ((x + MAX_SIGMOID) * SIZE / MAX_SIGMOID / 2.0) as i64;
        let i = idx.clamp(0, 512);
        let xi = (i as f32 * 2.0 * MAX_SIGMOID) / SIZE - MAX_SIGMOID;
        1.0 / (1.0 + (-xi).exp())
    }
}

// ---------------------------------------------------------------------------
// python-equivalent text prep
// ---------------------------------------------------------------------------

/// C++ readWord separators (istream path splits on '\0' too) plus python
/// splitlines-only chars. `str.splitlines` splits on \n \r \v \f \x1c \x1d
/// \x1e \x85 \u2028 \u2029; after " ".join(...) all of them are separators.
#[inline]
fn is_sep(c: char) -> bool {
    matches!(
        c,
        ' ' | '\t'
            | '\n'
            | '\r'
            | '\u{b}'
            | '\u{c}'
            | '\0'
            | '\u{1c}'
            | '\u{1d}'
            | '\u{1e}'
            | '\u{85}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

/// python str.strip() default set: Unicode whitespace plus \x1c-\x1f.
#[inline]
fn py_ws(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}

fn py_strip(s: &str) -> &str {
    s.trim_matches(py_ws)
}

// ---------------------------------------------------------------------------
// 128-bit document hash: SipHash-2-4, two fixed keys
// ---------------------------------------------------------------------------

/// SipHash-2-4 over `msg` with keys (k0, k1). Test vectors from the SipHash
/// paper (key 000102...0f) pin the implementation in the unit tests.
fn siphash24(k0: u64, k1: u64, msg: &[u8]) -> u64 {
    #[inline(always)]
    fn rnd(v0: &mut u64, v1: &mut u64, v2: &mut u64, v3: &mut u64) {
        *v0 = v0.wrapping_add(*v1);
        *v1 = v1.rotate_left(13);
        *v1 ^= *v0;
        *v0 = v0.rotate_left(32);
        *v2 = v2.wrapping_add(*v3);
        *v3 = v3.rotate_left(16);
        *v3 ^= *v2;
        *v0 = v0.wrapping_add(*v3);
        *v3 = v3.rotate_left(21);
        *v3 ^= *v0;
        *v2 = v2.wrapping_add(*v1);
        *v1 = v1.rotate_left(17);
        *v1 ^= *v2;
        *v2 = v2.rotate_left(32);
    }

    let mut v0 = k0 ^ 0x736f_6d65_7073_6575;
    let mut v1 = k1 ^ 0x646f_7261_6e64_6f6d;
    let mut v2 = k0 ^ 0x6c79_6765_6e65_7261;
    let mut v3 = k1 ^ 0x7465_6462_7974_6573;

    let mut chunk = [0u8; 8];
    let n = msg.len();
    let mut i = 0;
    while i + 8 <= n {
        chunk.copy_from_slice(&msg[i..i + 8]);
        let m = u64::from_le_bytes(chunk);
        v3 ^= m;
        rnd(&mut v0, &mut v1, &mut v2, &mut v3);
        rnd(&mut v0, &mut v1, &mut v2, &mut v3);
        v0 ^= m;
        i += 8;
    }
    let mut last = ((n as u64) & 0xff) << 56;
    for (k, &b) in msg[i..].iter().enumerate() {
        last |= (b as u64) << (8 * k);
    }
    v3 ^= last;
    rnd(&mut v0, &mut v1, &mut v2, &mut v3);
    rnd(&mut v0, &mut v1, &mut v2, &mut v3);
    v0 ^= last;
    v2 ^= 0xff;
    for _ in 0..4 {
        rnd(&mut v0, &mut v1, &mut v2, &mut v3);
    }
    v0 ^ v1 ^ v2 ^ v3
}

fn doc_hash128(bytes: &[u8]) -> u128 {
    let a = siphash24(0x0706_0504_0302_0100, 0x0f0e_0d0c_0b0a_0908, bytes) as u128;
    let b = siphash24(0x8765_4321_1f2e_3d4c, 0xfeed_beef_cafe_0123, bytes) as u128;
    (b << 64) | a
}

// ---------------------------------------------------------------------------
// memory-bounded exact dedup set
// ---------------------------------------------------------------------------

struct DedupSet {
    seen: HashSet<u128>,
    cap: usize,
    capped: bool,
    capped_at_doc: u64,
}

impl DedupSet {
    fn new(cap: usize) -> DedupSet {
        DedupSet {
            seen: HashSet::with_capacity(cap.min(1 << 26)),
            cap,
            capped: false,
            capped_at_doc: 0,
        }
    }
    /// true = first time seen (caller treats doc as unique).
    fn insert(&mut self, h: u128, doc_idx: u64) -> bool {
        if self.seen.contains(&h) {
            return false;
        }
        if self.seen.len() >= self.cap {
            if !self.capped {
                self.capped = true;
                self.capped_at_doc = doc_idx;
                eprintln!(
                    "[filter] dedup set capped at {} entries (doc #{}); \
new unique docs will no longer be remembered, membership checks continue",
                    self.cap, doc_idx
                );
            }
            return true;
        }
        self.seen.insert(h);
        true
    }
}

// ---------------------------------------------------------------------------
// document splitter: blank-line runs (>=2 consecutive '\n') are boundaries
// ---------------------------------------------------------------------------

/// Which output a completed doc is written to (two-region mode).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Region {
    Head,
    Tail,
}

struct Splitter {
    /// current document bytes (single newlines are content)
    doc: Vec<u8>,
    /// '\n's of the blank run currently being scanned
    run: Vec<u8>,
    /// blank-run bytes sitting between the last completed doc and the next
    /// one; written ahead of the next kept doc when both neighbors are kept
    /// in the SAME region
    pending_sep: Vec<u8>,
    last_kept: Option<Region>,
    /// input offset of the next byte fed through push
    pos: u64,
    /// input offset of the current doc's first byte (valid while doc non-empty)
    doc_start: u64,
    done: bool,
}

impl Splitter {
    fn new() -> Splitter {
        Splitter {
            doc: Vec::with_capacity(4096),
            run: Vec::new(),
            pending_sep: Vec::new(),
            last_kept: None,
            pos: 0,
            doc_start: 0,
            done: false,
        }
    }

    /// Feed one byte. Completes at most one document; `act` scores it with its
    /// start offset and returns the output region (None = drop). The original
    /// separator bytes are written ahead of a kept doc when the previous kept
    /// doc is adjacent in the same region's output.
    fn push<W: Write, T: Write + ?Sized>(
        &mut self,
        b: u8,
        head: &mut W,
        tail: &mut T,
        act: &mut impl FnMut(&[u8], u64) -> Option<Region>,
    ) -> std::io::Result<()> {
        if !self.run.is_empty() {
            if b == b'\n' {
                self.run.push(b);
                self.pos += 1;
                return Ok(());
            }
            // blank run ended at this byte
            if self.run.len() >= 2 {
                self.complete_doc(head, tail, act)?;
                // the run that just ended separates the doc just completed
                // from whatever comes next
                self.pending_sep.extend_from_slice(&self.run);
            } else {
                // single newline: part of the document content
                self.doc.push(b'\n');
            }
            self.run.clear();
        }
        if b == b'\n' {
            self.run.push(b);
        } else {
            if self.doc.is_empty() {
                self.doc_start = self.pos;
            }
            self.doc.push(b);
        }
        self.pos += 1;
        Ok(())
    }

    fn complete_doc<W: Write, T: Write + ?Sized>(
        &mut self,
        head: &mut W,
        tail: &mut T,
        act: &mut impl FnMut(&[u8], u64) -> Option<Region>,
    ) -> std::io::Result<()> {
        if self.doc.is_empty() {
            // degenerate empty doc between separators: keep byte flow intact
            return Ok(());
        }
        let route = act(&self.doc, self.doc_start);
        if let Some(r) = route {
            if self.last_kept == route && !self.pending_sep.is_empty() {
                match r {
                    Region::Head => head.write_all(&self.pending_sep)?,
                    Region::Tail => tail.write_all(&self.pending_sep)?,
                }
            }
            match r {
                Region::Head => head.write_all(&self.doc)?,
                Region::Tail => tail.write_all(&self.doc)?,
            }
            self.last_kept = route;
        }
        // the separator ahead of this doc has been consumed either way
        self.pending_sep.clear();
        self.doc.clear();
        Ok(())
    }

    fn finish<W: Write, T: Write + ?Sized>(
        &mut self,
        head: &mut W,
        tail: &mut T,
        act: &mut impl FnMut(&[u8], u64) -> Option<Region>,
    ) -> std::io::Result<()> {
        if self.done {
            return Ok(());
        }
        self.done = true;
        if !self.run.is_empty() {
            if self.run.len() >= 2 {
                self.complete_doc(head, tail, act)?;
                // trailing separator: no following doc, dropped
                self.pending_sep.clear();
            } else {
                // lone trailing newline is document content
                self.doc.push(b'\n');
                self.run.clear();
                self.complete_doc(head, tail, act)?;
            }
        } else {
            self.complete_doc(head, tail, act)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// stats
// ---------------------------------------------------------------------------

const HIST_BINS: usize = 100;

struct Stats {
    input_bytes_read: u64,
    docs_seen: u64,
    docs_kept: u64,
    docs_dropped_score: u64,
    docs_dropped_dup: u64,
    docs_no_score: u64,
    bytes_seen: u64,
    bytes_written: u64,
    hist: [u64; HIST_BINS],
    score_sum: f64,
    keep_at: [u64; 3], // >= 0.3, >= 0.5, >= 0.7
    scored: u64,
}

impl Stats {
    fn new() -> Stats {
        Stats {
            input_bytes_read: 0,
            docs_seen: 0,
            docs_kept: 0,
            docs_dropped_score: 0,
            docs_dropped_dup: 0,
            docs_no_score: 0,
            bytes_seen: 0,
            bytes_written: 0,
            hist: [0; HIST_BINS],
            score_sum: 0.0,
            keep_at: [0; 3],
            scored: 0,
        }
    }

    fn note_score(&mut self, s: f32) {
        self.scored += 1;
        self.score_sum += s as f64;
        let b = ((s as f64) * HIST_BINS as f64) as usize;
        self.hist[b.min(HIST_BINS - 1)] += 1;
        if s >= 0.3 {
            self.keep_at[0] += 1;
        }
        if s >= 0.5 {
            self.keep_at[1] += 1;
        }
        if s >= 0.7 {
            self.keep_at[2] += 1;
        }
    }
}

fn json_escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

fn snippet(bytes: &[u8], max: usize) -> String {
    let mut end = max.min(bytes.len());
    while end > 0 && (bytes[end - 1] & 0xC0) == 0x80 {
        end -= 1;
    }
    json_escape(&String::from_utf8_lossy(&bytes[..end]))
}

fn write_stats(
    path: &str,
    st: &Stats,
    cli: &Cli,
    model: &FtModel,
    dedup: &DedupSet,
    input_bytes_read: u64,
    input_bytes_total: u64,
    elapsed: f64,
    is_final: bool,
) -> std::io::Result<()> {
    let labels: Vec<String> = (0..model.nlabels as usize)
        .map(|i| String::from_utf8_lossy(&model.words[model.nwords as usize + i].0).into_owned())
        .collect();
    let rate = if input_bytes_read > 0 {
        input_bytes_read as f64 / 1e6 / elapsed.max(1e-9)
    } else {
        0.0
    };
    let mut j = String::with_capacity(4096);
    j.push_str("{\n");
    j.push_str(&format!("  \"final\": {},\n", is_final));
    j.push_str(&format!("  \"input\": \"{}\",\n", json_escape(&cli.input)));
    j.push_str(&format!("  \"output\": \"{}\",\n", json_escape(&cli.output)));
    j.push_str(&format!("  \"model\": \"{}\",\n", json_escape(&cli.model)));
    j.push_str(&format!("  \"model_labels\": {:?},\n", labels));
    j.push_str(&format!("  \"threshold\": {},\n", cli.threshold));
    j.push_str(&format!("  \"dedup_only\": {},\n", cli.dedup_only));
    j.push_str(&format!("  \"input_bytes_total\": {input_bytes_total},\n"));
    j.push_str(&format!("  \"input_bytes_read\": {input_bytes_read},\n"));
    j.push_str(&format!("  \"docs_seen\": {},\n", st.docs_seen));
    j.push_str(&format!("  \"docs_kept\": {},\n", st.docs_kept));
    j.push_str(&format!("  \"docs_dropped_score\": {},\n", st.docs_dropped_score));
    j.push_str(&format!("  \"docs_dropped_dup\": {},\n", st.docs_dropped_dup));
    j.push_str(&format!("  \"docs_no_score\": {},\n", st.docs_no_score));
    j.push_str(&format!("  \"dedup_capped\": {},\n", dedup.capped));
    j.push_str(&format!("  \"dedup_cap\": {},\n", dedup.cap));
    j.push_str(&format!("  \"dedup_entries\": {},\n", dedup.seen.len()));
    j.push_str(&format!("  \"dedup_capped_at_doc\": {},\n", dedup.capped_at_doc));
    j.push_str(&format!("  \"bytes_seen\": {},\n", st.bytes_seen));
    j.push_str(&format!("  \"bytes_written\": {},\n", st.bytes_written));
    j.push_str(&format!(
        "  \"keep_rate\": {},\n",
        st.docs_kept as f64 / st.docs_seen.max(1) as f64
    ));
    j.push_str(&format!("  \"mean_score\": {},\n", st.score_sum / st.scored.max(1) as f64));
    j.push_str(&format!(
        "  \"scored_docs\": {},\n  \"keep_at_0.3\": {},\n  \"keep_at_0.5\": {},\n  \"keep_at_0.7\": {},\n",
        st.scored, st.keep_at[0], st.keep_at[1], st.keep_at[2]
    ));
    j.push_str(&format!("  \"elapsed_s\": {elapsed:.1},\n"));
    j.push_str(&format!("  \"input_mb_per_s\": {rate:.1},\n"));
    j.push_str("  \"hist_bins\": 100,\n  \"hist\": [");
    for (i, h) in st.hist.iter().enumerate() {
        if i > 0 {
            j.push(',');
        }
        j.push_str(&h.to_string());
    }
    j.push_str("]\n}\n");
    // write via temp + rename so readers never see a torn file
    let tmp = format!("{path}.tmp");
    {
        let mut f = File::create(&tmp)?;
        f.write_all(j.as_bytes())?;
        f.sync_all().ok();
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// CLI + main loop
// ---------------------------------------------------------------------------

struct Cli {
    input: String,
    output: String,
    /// two-region mode: eval tail output path
    eval_output: Option<String>,
    /// two-region mode: eval tail = last N MiB of the raw input
    eval_tail_mb: Option<u64>,
    model: String,
    threshold: f32,
    dedup_only: bool,
    limit_bytes: Option<u64>,
    stats: Option<String>,
    examples: Option<String>,
    examples_per_class: usize,
    chunk_bytes: usize,
    dedup_cap: usize,
    log_every_docs: u64,
    /// validation helper: score blank-line-separated docs of this file and
    /// print "idx<TAB>P(hq)" lines to stderr instead of filtering a corpus
    score_docs: Option<String>,
}

fn parse_args() -> Result<Cli, String> {
    let mut c = Cli {
        input: String::new(),
        output: String::new(),
        eval_output: None,
        eval_tail_mb: None,
        model: String::new(),
        threshold: 0.5,
        dedup_only: false,
        limit_bytes: None,
        stats: None,
        examples: None,
        examples_per_class: 3,
        chunk_bytes: 64 << 20,
        dedup_cap: 48_000_000,
        log_every_docs: 2_000_000,
        score_docs: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let val = |it: &mut dyn Iterator<Item = String>| -> Result<String, String> {
            it.next().ok_or_else(|| format!("{a} expects a value"))
        };
        match a.as_str() {
            "--input" => c.input = val(&mut it)?,
            "--output" => c.output = val(&mut it)?,
            "--eval-output" => c.eval_output = Some(val(&mut it)?),
            "--eval-tail-mb" => {
                c.eval_tail_mb = Some(val(&mut it)?.parse().map_err(|_| "--eval-tail-mb u64")?)
            }
            "--model" => c.model = val(&mut it)?,
            "--threshold" => c.threshold = val(&mut it)?.parse().map_err(|_| "--threshold f32")?,
            "--dedup-only" => c.dedup_only = true,
            "--limit-bytes" => {
                c.limit_bytes = Some(val(&mut it)?.parse().map_err(|_| "--limit-bytes u64")?)
            }
            "--stats" => c.stats = Some(val(&mut it)?),
            "--examples" => c.examples = Some(val(&mut it)?),
            "--examples-per-class" => {
                c.examples_per_class =
                    val(&mut it)?.parse().map_err(|_| "--examples-per-class usize")?
            }
            "--chunk-mb" => {
                c.chunk_bytes = (val(&mut it)?.parse::<usize>().map_err(|_| "--chunk-mb usize")?) << 20
            }
            "--dedup-cap" => c.dedup_cap = val(&mut it)?.parse().map_err(|_| "--dedup-cap usize")?,
            "--log-every-docs" => {
                c.log_every_docs = val(&mut it)?.parse().map_err(|_| "--log-every-docs u64")?
            }
            "--score-docs" => c.score_docs = Some(val(&mut it)?),
            other => return Err(format!("unknown arg {other}")),
        }
    }
    if c.eval_output.is_none() != c.eval_tail_mb.is_none() {
        return Err("--eval-output and --eval-tail-mb go together".into());
    }
    if c.input.is_empty() || c.output.is_empty() || c.model.is_empty() {
        return Err(
            "required: --input <corpus> --output <filtered> --model <fasttext.bin>\n\
             optional: --threshold F --dedup-only --limit-bytes N --stats JSON \
             --examples JSONL [--examples-per-class N] --chunk-mb N --dedup-cap N \
             --eval-output PATH --eval-tail-mb N"
                .into(),
        );
    }
    Ok(c)
}

#[derive(Clone)]
struct Example {
    kind: &'static str,
    score: Option<f32>,
    n: usize,
}

struct Pipeline<'a> {
    model: &'a FtModel,
    threshold: f32,
    dedup_only: bool,
    dedup: DedupSet,
    st: Stats,
    examples: Vec<(Example, Vec<u8>)>,
    want_examples: bool,
    examples_per_class: usize,
}

impl<'a> Pipeline<'a> {
    /// Dedup, score, and decide one document. Returns true when the caller
    /// should write it.
    fn on_doc(&mut self, doc: &[u8]) -> bool {
        let st = &mut self.st;
        st.docs_seen += 1;
        st.bytes_seen += doc.len() as u64;

        // exact dedup first: a dup is dropped whether or not it would score
        let h = doc_hash128(doc);
        if !self.dedup.insert(h, st.docs_seen) {
            st.docs_dropped_dup += 1;
            self.push_example("dup", None, doc);
            return false;
        }

        if self.dedup_only {
            st.docs_kept += 1;
            st.bytes_written += doc.len() as u64;
            return true;
        }

        // corpus is specified UTF-8; fall back to lossy if a doc is not
        let score = match std::str::from_utf8(doc) {
            Ok(s) => self.model.score(s),
            Err(_) => self.model.score(&String::from_utf8_lossy(doc)),
        };
        match score {
            None => {
                st.docs_no_score += 1;
                false
            }
            Some(s) => {
                st.note_score(s);
                let keep = s >= self.threshold;
                if keep {
                    st.docs_kept += 1;
                    st.bytes_written += doc.len() as u64;
                    self.push_example("kept", Some(s), doc);
                } else {
                    st.docs_dropped_score += 1;
                    self.push_example("low_score", Some(s), doc);
                }
                keep
            }
        }
    }

    fn push_example(&mut self, kind: &'static str, score: Option<f32>, doc: &[u8]) {
        if !self.want_examples {
            return;
        }
        let have = self
            .examples
            .iter()
            .filter(|(e, _)| e.kind == kind)
            .count();
        if have < self.examples_per_class {
            self.examples.push((
                Example { kind, score, n: doc.len() },
                doc[..doc.len().min(200)].to_vec(),
            ));
        }
    }
}

/// Routes completed docs by their START offset: head docs → train output,
/// tail docs → eval output, straddlers dropped entirely. One shared Pipeline
/// keeps dedup global (a tail doc that duplicates a head doc is dropped from
/// eval — same rule set, no leak in either direction).
struct Router {
    /// first input byte of the eval tail region (u64::MAX = single region)
    boundary: u64,
    straddle_docs: u64,
    straddle_bytes: u64,
    head_docs: u64,
    head_bytes: u64,
    tail_docs: u64,
    tail_bytes: u64,
}

impl Router {
    fn route(&mut self, pl: &mut Pipeline, doc: &[u8], start: u64) -> Option<Region> {
        let end = start + doc.len() as u64;
        if start < self.boundary && end > self.boundary {
            self.straddle_docs += 1;
            self.straddle_bytes += doc.len() as u64;
            return None;
        }
        if !pl.on_doc(doc) {
            return None;
        }
        if start >= self.boundary {
            self.tail_docs += 1;
            self.tail_bytes += doc.len() as u64;
            Some(Region::Tail)
        } else {
            self.head_docs += 1;
            self.head_bytes += doc.len() as u64;
            Some(Region::Head)
        }
    }
}

fn main() {
    let cli = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("filter: {e}");
            std::process::exit(2);
        }
    };
    let t0 = Instant::now();

    // validation path: score docs of a small file, one "idx\tscore" per doc
    if let Some(path) = &cli.score_docs {
        let model = match FtModel::load(&cli.model) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("filter: model load failed: {e:?}");
                std::process::exit(3);
            }
        };
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("filter: read {path}: {e}");
            std::process::exit(3);
        });
        for (i, doc) in text.split("\n\n").enumerate() {
            let s = model.score(doc);
            println!("{i}\t{}", s.map(|v| v.to_string()).unwrap_or_else(|| "none".into()));
        }
        return;
    }

    eprintln!(
        "[filter] loading fastText model {} (allocates the input matrix)",
        cli.model
    );
    let model = match FtModel::load(&cli.model) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("filter: model load failed: {e:?}");
            std::process::exit(3);
        }
    };
    eprintln!(
        "[filter] model ok: dim={} bucket={} wordNgrams={} minn={} maxn={} loss={} \
nwords={} nlabels={} hq_row={} labels={:?}",
        model.args.dim,
        model.args.bucket,
        model.args.word_ngrams,
        model.args.minn,
        model.args.maxn,
        model.args.loss,
        model.nwords,
        model.nlabels,
        model.hq_row,
        (0..model.nlabels as usize)
            .map(|i| String::from_utf8_lossy(&model.words[model.nwords as usize + i].0).into_owned())
            .collect::<Vec<_>>()
    );

    let in_file = File::open(&cli.input).unwrap_or_else(|e| {
        eprintln!("filter: open input: {e}");
        std::process::exit(3);
    });
    let total_in = in_file.metadata().map(|m| m.len()).unwrap_or(0);
    let mut reader = BufReader::with_capacity(1 << 22, in_file);
    let out_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&cli.output)
        .unwrap_or_else(|e| {
            eprintln!("filter: open output: {e}");
            std::process::exit(3);
        });
    let mut out = BufWriter::with_capacity(1 << 20, out_file);

    // two-region mode: head [0, boundary) → `out`, tail [boundary, EOF) →
    // eval output; single-region runs fold into the same path with the
    // boundary at u64::MAX (everything routes Head).
    let boundary = match (cli.eval_output.as_deref(), cli.eval_tail_mb) {
        (Some(_), Some(mb)) => total_in.saturating_sub(mb << 20),
        _ => u64::MAX,
    };
    let mut eval_buf = cli.eval_output.as_deref().map(|p| {
        let f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(p)
            .unwrap_or_else(|e| {
                eprintln!("filter: open eval output: {e}");
                std::process::exit(3);
            });
        BufWriter::with_capacity(1 << 20, f)
    });
    if boundary != u64::MAX {
        eprintln!(
            "[filter] two-region mode: eval tail = last {} MiB, boundary at byte {} of {} \
             (docs straddling the boundary are dropped)",
            cli.eval_tail_mb.unwrap(),
            boundary,
            total_in
        );
    }
    let mut sink = std::io::sink();
    let tail_out: &mut dyn Write = match &mut eval_buf {
        Some(w) => w,
        None => &mut sink,
    };
    let mut router = Router {
        boundary,
        straddle_docs: 0,
        straddle_bytes: 0,
        head_docs: 0,
        head_bytes: 0,
        tail_docs: 0,
        tail_bytes: 0,
    };

    let mut pipeline = Pipeline {
        model: &model,
        threshold: cli.threshold,
        dedup_only: cli.dedup_only,
        dedup: DedupSet::new(cli.dedup_cap),
        st: Stats::new(),
        examples: Vec::new(),
        want_examples: cli.examples.is_some(),
        examples_per_class: cli.examples_per_class,
    };
    let mut splitter = Splitter::new();

    let mut chunk = vec![0u8; cli.chunk_bytes];
    let mut next_stats = 30.0f64;
    let mut next_log_docs = cli.log_every_docs;
    let limit = cli.limit_bytes;
    let mut eof = false;

    while !eof {
        let want = chunk.len().min(match limit {
            Some(l) => l.saturating_sub(pipeline.st.input_bytes_read) as usize,
            None => chunk.len(),
        });
        if want == 0 {
            break;
        }
        let mut filled = 0usize;
        while filled < want {
            match reader.read(&mut chunk[filled..want]) {
                Ok(0) => {
                    eof = true;
                    break;
                }
                Ok(n) => filled += n,
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => {
                    eprintln!("filter: read: {e}");
                    std::process::exit(4);
                }
            }
        }
        if filled == 0 {
            break;
        }
        pipeline.st.input_bytes_read += filled as u64;

        {
            let (pl, rt) = (&mut pipeline, &mut router);
            let mut act = |doc: &[u8], start: u64| rt.route(pl, doc, start);
            for &b in &chunk[..filled] {
                if let Err(e) = splitter.push(b, &mut out, tail_out, &mut act) {
                    eprintln!("filter: write: {e}");
                    std::process::exit(4);
                }
            }
        }

        let docs = pipeline.st.docs_seen;
        if docs >= next_log_docs {
            next_log_docs += cli.log_every_docs;
            let el = t0.elapsed().as_secs_f64();
            let st = &pipeline.st;
            eprintln!(
                "[filter] docs={} kept={} dup={} low_score={} read={:.1}MB out={:.1}MB \
{:.1}MB/s elapsed={:.0}s",
                docs,
                st.docs_kept,
                st.docs_dropped_dup,
                st.docs_dropped_score,
                st.input_bytes_read as f64 / 1e6,
                st.bytes_written as f64 / 1e6,
                st.input_bytes_read as f64 / 1e6 / el,
                el
            );
        }

        let el = t0.elapsed().as_secs_f64();
        if let Some(sp) = &cli.stats {
            if el >= next_stats {
                next_stats = el + 30.0;
                if let Err(e) =
                    write_stats(sp, &pipeline.st, &cli, &model, &pipeline.dedup, pipeline.st.input_bytes_read, total_in, el, false)
                {
                    eprintln!("filter: stats write: {e}");
                }
            }
        }
        if let Some(l) = limit {
            if pipeline.st.input_bytes_read >= l {
                break;
            }
        }
    }

    {
        let (pl, rt) = (&mut pipeline, &mut router);
        let mut act = |doc: &[u8], start: u64| rt.route(pl, doc, start);
        if let Err(e) = splitter.finish(&mut out, tail_out, &mut act) {
            eprintln!("filter: write: {e}");
            std::process::exit(4);
        }
    }
    if let Err(e) = out.flush() {
        eprintln!("filter: flush: {e}");
        std::process::exit(4);
    }
    if let Some(w) = eval_buf.as_mut() {
        if let Err(e) = w.flush() {
            eprintln!("filter: flush: {e}");
            std::process::exit(4);
        }
    }

    let el = t0.elapsed().as_secs_f64();
    let st = &pipeline.st;
    eprintln!(
        "[filter] done in {el:.0}s: docs={} kept={} ({:.1}%) dup={} low_score={} no_score={} \
read={:.1}MB wrote={:.1}MB = {:.1}MB/s",
        st.docs_seen,
        st.docs_kept,
        100.0 * st.docs_kept as f64 / st.docs_seen.max(1) as f64,
        st.docs_dropped_dup,
        st.docs_dropped_score,
        st.docs_no_score,
        st.input_bytes_read as f64 / 1e6,
        st.bytes_written as f64 / 1e6,
        st.input_bytes_read as f64 / 1e6 / el,
    );
    eprintln!(
        "[filter] regions: boundary={} head_docs={} head_MB={} tail_docs={} tail_MB={} \
straddle_dropped={} ({}MB)",
        router.boundary,
        router.head_docs,
        router.head_bytes as f64 / 1e6,
        router.tail_docs,
        router.tail_bytes as f64 / 1e6,
        router.straddle_docs,
        router.straddle_bytes as f64 / 1e6,
    );

    if let Some(sp) = &cli.stats {
        match write_stats(sp, st, &cli, &model, &pipeline.dedup, st.input_bytes_read, total_in, el, true) {
            Ok(()) => eprintln!("[filter] stats: {sp}"),
            Err(e) => eprintln!("filter: stats write: {e}"),
        }
    }
    if let Some(ep) = &cli.examples {
        let mut j = String::new();
        for (e, snip) in &pipeline.examples {
            j.push_str(&format!(
                "{{\"kind\":\"{}\",\"score\":{},\"bytes\":{},\"snippet\":\"{}\"}}\n",
                e.kind,
                e.score.map(|v| v.to_string()).unwrap_or_else(|| "null".into()),
                e.n,
                snippet(&snip, 200)
            ));
        }
        match std::fs::write(ep, j) {
            Ok(()) => eprintln!("[filter] examples: {ep}"),
            Err(e) => eprintln!("filter: examples write: {e}"),
        }
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Official SipHash-2-4 vectors (key = 00 01 .. 0f) from the SipHash paper.
    #[test]
    fn siphash_reference_vectors() {
        let k0 = u64::from_le_bytes([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]);
        let k1 = u64::from_le_bytes([0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f]);
        assert_eq!(siphash24(k0, k1, b""), 0x726fdb47dd0e0e31);
        assert_eq!(siphash24(k0, k1, &[0x00]), 0x74f839c593dc67fd);
        assert_ne!(siphash24(1, 2, b""), siphash24(1, 3, b""));
    }

    /// fastText signed FNV-1a. Expected values generated by compiling the
    /// exact C++ `Dictionary::hash` body (uint32_t(int8_t(c)) sign extension)
    /// with g++; includes multi-byte UTF-8 and high-bit-byte cases, where an
    /// unsigned FNV would diverge.
    #[test]
    fn ft_hash_matches_cpp() {
        let cases: Vec<(&[u8], u32)> = vec![
            (b"", 2166136261),
            (b"a", 0xe40c292c),
            (b"hello", 0x4f9f2cab),
            (b"</s>", 0xd79c9359),
            (b"the", 0xb40eb21c),
            (b"Hello, world!", 0xed90f094),
            // привет
            (b"\xd0\xbf\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82", 0xf069a77f),
            (b"\xff\xfe\x01", 0x3e92b7c3),
        ];
        for (input, want) in cases {
            let got = ft_hash(input);
            assert_eq!(got, want, "hash({:?}) = {got:#x}, want {want:#x}", input);
        }
    }

    #[test]
    fn tokenizer_separators() {
        let doc = py_strip("\u{a0} hello\u{a0}world\tfoo\nbar\u{2028}baz ");
        let toks: Vec<&str> = doc.split(is_sep).filter(|t| !t.is_empty()).collect();
        // NBSP is NOT a separator (glues "hello\xa0world"); \u2028 IS.
        assert_eq!(toks, vec!["hello\u{a0}world", "foo", "bar", "baz"]);
    }

    #[test]
    fn splitter_preserves_separator_between_kept() {
        let mut sp = Splitter::new();
        let mut out: Vec<u8> = Vec::new();
        let mut tail: Vec<u8> = Vec::new();
        let docs = std::cell::RefCell::new(Vec::new());
        {
            let mut act = |d: &[u8], _s: u64| -> Option<Region> {
                docs.borrow_mut().push(String::from_utf8_lossy(d).into_owned());
                Some(Region::Head)
            };
            for b in b"aa\nbb\n\ncc\n\n\ndd" {
                let _ = sp.push(*b, &mut out, &mut tail, &mut act).unwrap();
            }
            let _ = sp.finish(&mut out, &mut tail, &mut act).unwrap();
        }
        // "aa\nbb" (single newline stays inside), then "cc", then "dd"
        assert_eq!(docs.borrow().len(), 3);
        assert_eq!(docs.borrow()[0], "aa\nbb");
        // separators between kept docs preserved verbatim (2 then 3 newlines)
        assert_eq!(out, b"aa\nbb\n\ncc\n\n\ndd");
        assert!(tail.is_empty());
    }

    #[test]
    fn splitter_drops_dropped_doc_separators() {
        let mut sp = Splitter::new();
        let mut out: Vec<u8> = Vec::new();
        let mut tail: Vec<u8> = Vec::new();
        {
            let mut act = |d: &[u8], _s: u64| -> Option<Region> {
                if d.starts_with(b"keep") {
                    Some(Region::Head)
                } else {
                    None
                }
            };
            for b in b"keep1\n\ndrop\n\nkeep2\n\n" {
                let _ = sp.push(*b, &mut out, &mut tail, &mut act).unwrap();
            }
            let _ = sp.finish(&mut out, &mut tail, &mut act).unwrap();
        }
        assert_eq!(out, b"keep1\n\nkeep2");
    }

    #[test]
    fn splitter_final_doc_processed() {
        let mut sp = Splitter::new();
        let mut out: Vec<u8> = Vec::new();
        let mut tail: Vec<u8> = Vec::new();
        let count = std::cell::Cell::new(0u32);
        {
            let mut act = |_d: &[u8], _s: u64| -> Option<Region> {
                count.set(count.get() + 1);
                Some(Region::Head)
            };
            for b in b"only-doc\nwith lines" {
                let _ = sp.push(*b, &mut out, &mut tail, &mut act).unwrap();
            }
            assert_eq!(count.get(), 0);
            let _ = sp.finish(&mut out, &mut tail, &mut act).unwrap();
        }
        assert_eq!(count.get(), 1);
        assert_eq!(out, b"only-doc\nwith lines");
    }

    /// Two-region routing: docs owned by start offset, the doc straddling the
    /// boundary dropped whole, and no separator bytes carried across regions.
    #[test]
    fn splitter_two_regions_drops_straddler() {
        // offsets: head1@0 head2@7 straddle@14..22 tail1@24 tail2@31
        let bytes = b"head1\n\nhead2\n\nstraddle\n\ntail1\n\ntail2";
        let mut sp = Splitter::new();
        let (mut head, mut tail) = (Vec::new(), Vec::new());
        let boundary = 16u64; // inside "straddle"
        let seen = std::cell::Cell::new(0u32);
        {
            let mut act = |d: &[u8], s: u64| -> Option<Region> {
                seen.set(seen.get() + 1);
                let end = s + d.len() as u64;
                if s < boundary && end > boundary {
                    return None; // straddler
                }
                if s >= boundary {
                    Some(Region::Tail)
                } else {
                    Some(Region::Head)
                }
            };
            for &b in bytes {
                let _ = sp.push(b, &mut head, &mut tail, &mut act).unwrap();
            }
            let _ = sp.finish(&mut head, &mut tail, &mut act).unwrap();
        }
        assert_eq!(seen.get(), 5);
        assert_eq!(head, b"head1\n\nhead2");
        assert_eq!(tail, b"tail1\n\ntail2");
    }
}
