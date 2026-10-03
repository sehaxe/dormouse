//! SFT: JSONL conversations → the byte stream the model actually trains on.
//!
//! dormouse is a **byte** LM (vocab 256), so there is no tokenizer to adapt and
//! no chat template to borrow: a turn boundary is a byte marker, and the only
//! thing SFT adds over pretraining is a **loss mask** that hides everything the
//! policy is not supposed to say. This module owns both, so a caller cannot
//! encode a conversation one way and mask it another.
//!
//! # The template, byte for byte
//!
//! One conversation is the concatenation of its messages, each rendered as
//!
//! ```text
//! <|im_start|>{role}\n{content}<|im_end|>\n
//! ```
//!
//! - `{role}` ∈ {`system`, `user`, `assistant`}. Anything else is a **LOUD**
//!   error: a mistyped role would render a header the policy has never seen and
//!   would never emit at inference, and the run would look healthy (ADR-0019).
//! - The two markers are [`IM_START`] / [`IM_END`], ChatML's spelling, because
//!   a coder's data is full of `<`, `|`, `_` and `>` and any ad-hoc marker of
//!   the same shape collides with the code it is wrapping.
//! - The `\n` after `<|im_end|>` is a separator, part of no message.
//! - Messages are concatenated **with no document separator** between
//!   conversations: [`SftStream`] packs them back to back exactly as
//!   [`crate::ByteStream`] slices a corpus, so a conversation can straddle a
//!   batch boundary and lose the context that preceded it there — the same
//!   property every pretraining chunk has, and cheaper than a pad row.
//!
//! # The mask
//!
//! [`SftExample::mask`] is parallel to `bytes` and is **1 on exactly the bytes
//! the policy must emit to reproduce an assistant turn**: the assistant's
//! `content` and its terminating [`IM_END`]. It is 0 on every role header, on
//! the separator newline, and on every user/system byte.
//!
//! `<|im_end|>` is inside the mask on purpose: it is the stop signal, so a
//! policy that never learns to emit it cannot end its turn. The `\n` after it is
//! outside, because generation stops at `<|im_end|>` and never produces it.
//!
//! # The mask is indexed by BYTE, the loss is indexed by LABEL
//!
//! The trainer's label sequence is `targets[q] = bytes[q + 1]` — a shift by one
//! (`crates/dormouse-train/src/lib.rs`, the `skip(1).chain(once(&nb[0]))`
//! line). So the mask a loss must consume is **not** `mask`: it is `mask`
//! shifted the same way. [`SftBatch::target_mask`] does that shift, and it is
//! the only correct way to ask for it: this is the exact off-by-one that put
//! the answer into the draft head's input one step early and invalidated
//! every DSpark number this project has ever recorded (`docs/reviews/
//! verify-tails-2026-09-30.md`), and there the shift was the *bug*. Here it is
//! the contract, so it is one named function and one pinned test instead.
//!
//! # Gate
//!
//! `tests/sft_format.rs` (this crate, CPU, no burn) pins the template byte for
//! byte, the mask, the shift, and that a role typo is refused. The gradient half
//! — that a masked user span really receives no gradient — is
//! `tests/sft_smoke.rs` in `dormouse-train`, because it needs a model.

use std::fmt;
use std::path::Path;

use serde::Deserialize;

/// ChatML turn opener. See the module docs for why this spelling.
pub const IM_START: &[u8] = b"<|im_start|>";
/// ChatML turn terminator, and the byte the policy learns to emit to stop.
pub const IM_END: &[u8] = b"<|im_end|>";
/// The three roles this template renders. Public because it is the whole
/// vocabulary of the format: a caller filtering a corpus before it reaches
/// [`encode`] needs the same list the encoder refuses against, and a second
/// copy of three strings is a third definition of the format.
pub const ROLES: [&str; 3] = ["system", "user", "assistant"];

/// One conversation, and which of its bytes the loss may score.
///
/// Both vectors are byte-indexed and the same length; `mask[i] == 1.0` means
/// `bytes[i]` is a target. f32 rather than bool so the value can be handed to
/// the loss unchanged (a bool cast on device is the shape ADR-0019 forbids).
#[derive(Clone, Debug, PartialEq)]
pub struct SftExample {
    /// The rendered conversation, markers and all.
    pub bytes: Vec<u8>,
    /// 1.0 on assistant content + the assistant's [`IM_END`], 0.0 elsewhere.
    pub mask: Vec<f32>,
}

impl SftExample {
    /// Bytes the loss may score. The number a reader wants first: it is what
    /// the mask leaves visible.
    pub fn trainable(&self) -> usize {
        self.mask.iter().filter(|&&m| m != 0.0).count()
    }
}

/// One message of a conversation, as it appears in the JSONL.
#[derive(Clone, Debug, Deserialize)]
struct RawMessage {
    role: String,
    content: String,
}

/// One line of the JSONL file.
#[derive(Debug, Deserialize)]
struct RawRow {
    messages: Vec<RawMessage>,
}

/// Every way encoding a conversation can fail. All of them are LOUD: this
/// crate's rule is that a byte the model trains on never appears without a
/// reason attached (ADR-0011, ADR-0019).
#[derive(Debug)]
pub enum SftError {
    /// A line that is not valid JSON, or not `{"messages": [...]}`.
    BadJson {
        /// 1-based line number in the file.
        line: usize,
        /// What the parser said.
        why: String,
    },
    /// A `role` outside [`ROLES`]. Named, because "expected one of" three
    /// strings is the difference between a 30-second fix and an afternoon.
    BadRole {
        /// 1-based line number in the file.
        line: usize,
        /// The role as written.
        role: String,
    },
    /// A row with no messages: nothing to encode, and an empty example in a
    /// packed corpus is a silently shorter one.
    EmptyRow {
        /// 1-based line number in the file.
        line: usize,
    },
    /// The file could not be read.
    Io {
        /// The path as given.
        path: String,
        /// What the OS said.
        why: String,
    },
    /// Fewer bytes than one batch. Refused here rather than served padded: a
    /// run that trains on one all-mask-zero batch has an objective of zero and
    /// a loss curve that looks like convergence.
    TooSmall {
        /// Bytes the corpus holds.
        have: usize,
        /// Bytes one batch needs.
        need: usize,
    },
    /// The stream was asked for a batch past its end without a rewind.
    Exhausted,
}

impl fmt::Display for SftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadJson { line, why } => {
                write!(f, "sft: line {line} is not a JSON conversation: {why}")
            }
            Self::BadRole { line, role } => write!(
                f,
                "sft: line {line} has role {role:?}, which this template cannot render. \
                 Roles are {ROLES:?}: rename the role, or add it to the template in \
                 crates/dormouse-data/src/sft.rs and re-render every corpus."
            ),
            Self::EmptyRow { line } => write!(f, "sft: line {line} has an empty `messages` list"),
            Self::Io { path, why } => write!(f, "sft: cannot read {path}: {why}"),
            Self::TooSmall { have, need } => write!(
                f,
                "sft: corpus is {have} bytes, one batch needs {need}. Padding the rest would \
                 train on mask-zero bytes (a zero objective that looks like convergence) - \
                 lower --seq-len/--batch or add conversations."
            ),
            Self::Exhausted => write!(
                f,
                "sft: the stream is past its last batch; call rewind() (eval does) or build a \
                 fresh stream (a second epoch re-reads the file)."
            ),
        }
    }
}

impl std::error::Error for SftError {}

/// Render one conversation to bytes + mask.
///
/// Pure: no file, no I/O, no clock. The JSONL reader is a loop over this, and
/// the test pings it directly, so the template can be reasoned about without a
/// corpus.
pub fn encode(messages: &[(&str, &str)]) -> Result<SftExample, SftError> {
    assert!(
        !messages.is_empty(),
        "sft: encode() called with no messages; use SftError::EmptyRow via read_jsonl for the \
         file path, and drop this assert only if an empty conversation is a real case"
    );
    let mut bytes = Vec::new();
    let mut mask: Vec<f32> = Vec::new();
    for (role, content) in messages {
        if !ROLES.contains(role) {
            return Err(SftError::BadRole {
                line: 0,
                role: (*role).to_string(),
            });
        }
        // The header is context for every turn and a target for none.
        push(&mut bytes, &mut mask, IM_START, 0.0);
        push(&mut bytes, &mut mask, role.as_bytes(), 0.0);
        push(&mut bytes, &mut mask, b"\n", 0.0);
        // Content is a target only in an assistant turn. `user` and `system`
        // are context whatever they contain.
        let trained = *role == "assistant";
        push(
            &mut bytes,
            &mut mask,
            content.as_bytes(),
            if trained { 1.0 } else { 0.0 },
        );
        push(
            &mut bytes,
            &mut mask,
            IM_END,
            if trained { 1.0 } else { 0.0 },
        );
        // Separator, never a target: generation stops at IM_END.
        push(&mut bytes, &mut mask, b"\n", 0.0);
    }
    Ok(SftExample { bytes, mask })
}

fn push(bytes: &mut Vec<u8>, mask: &mut Vec<f32>, part: &[u8], m: f32) {
    bytes.extend_from_slice(part);
    mask.extend(std::iter::repeat_n(m, part.len()));
}

/// Read a `.jsonl` SFT file, one conversation per line.
///
/// Every line is a complete `{"messages": [{"role", "content"}, ...]}`. Not
/// streamed: an SFT corpus is small next to a pretraining corpus (thousands of
/// conversations, not 46 GB), the whole thing is held in memory on purpose so
/// a malformed line is refused at load instead of 40 000 steps in.
pub fn read_jsonl(path: &Path) -> Result<Vec<SftExample>, SftError> {
    let raw = std::fs::read_to_string(path).map_err(|e| SftError::Io {
        path: path.display().to_string(),
        why: e.to_string(),
    })?;
    parse_jsonl(&raw)
}

/// [`read_jsonl`] without the file: one JSONL document already in memory.
///
/// A corpus is routinely not a local file — a shard off a download, a slice of
/// one after filtering — and the alternative is a caller writing a temp file to
/// call the same parser. Same refusals, same line numbers.
pub fn parse_jsonl(raw: &str) -> Result<Vec<SftExample>, SftError> {
    let mut out = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        let n = i + 1;
        if line.trim().is_empty() {
            continue;
        }
        let row: RawRow = serde_json::from_str(line).map_err(|e| SftError::BadJson {
            line: n,
            why: e.to_string(),
        })?;
        if row.messages.is_empty() {
            return Err(SftError::EmptyRow { line: n });
        }
        let msgs: Vec<(&str, &str)> = row
            .messages
            .iter()
            .map(|m| (m.role.as_str(), m.content.as_str()))
            .collect();
        let enc = encode(&msgs).map_err(|e| match e {
            SftError::BadRole { role, .. } => SftError::BadRole { line: n, role },
            other => other,
        })?;
        out.push(enc);
    }
    Ok(out)
}

/// One `[batch, seq_len]` batch of SFT bytes plus the mask that goes with them.
#[derive(Clone, Debug, PartialEq)]
pub struct SftBatch {
    /// `batch * seq_len` bytes, row-major, exactly the shape the trainer's
    /// `bytes_to_tensors` takes.
    pub bytes: Vec<u8>,
    /// `batch * seq_len` mask values parallel to `bytes`.
    pub mask: Vec<f32>,
}

impl SftBatch {
    /// The mask ALIGNED WITH THE LABEL SEQUENCE, i.e. `target_mask[q]` is the
    /// weight on `targets[q]`.
    ///
    /// The trainer's labels are the flat shift `nb.iter().skip(1).chain(once(&nb[0]))`
    /// (`crates/dormouse-train/src/lib.rs`), so the label at flat position `q`
    /// is the byte at `q + 1` — and the last position wraps to byte 0. This is
    /// that shift applied to the mask, including the wrap, because the label
    /// sequence wraps the same way.
    ///
    /// Length is `bytes.len()`. Returned as a flat `Vec` because the loss
    /// multiplies a `[batch, seq_len]` tensor and the row structure is the
    /// caller's problem — the same split every other `[b, t]` host tensor in
    /// this crate makes.
    pub fn target_mask(&self) -> Vec<f32> {
        self.mask
            .iter()
            .skip(1)
            .chain(self.mask.iter().take(1))
            .copied()
            .collect()
    }
}

/// The SFT analogue of [`crate::ByteStream`]: a packed conversation stream that
/// also carries the mask.
///
/// Same packing rule as the pretraining stream — conversations concatenated
/// back to back, cut into `batch * seq_len` chunks, no padding between them and
/// none between conversations — so the two paths differ in exactly one thing:
/// this one knows which bytes may be scored.
pub struct SftStream {
    /// The whole corpus, encoded once.
    bytes: Vec<u8>,
    /// The whole corpus's mask, parallel to `bytes`.
    mask: Vec<f32>,
    /// Rows per batch.
    batch: usize,
    /// Positions (BYTES) per row.
    seq_len: usize,
    /// Read cursor into `bytes`, in bytes. Reset by [`Self::rewind`].
    pos: usize,
}

impl SftStream {
    /// Pack `examples` into a stream. Refuses a corpus smaller than one batch
    /// ([`SftError::TooSmall`]).
    pub fn new(examples: Vec<SftExample>, batch: usize, seq_len: usize) -> Result<Self, SftError> {
        assert!(
            batch > 0 && seq_len > 1,
            "sft: batch={batch} seq_len={seq_len}; a batch needs a row and a label to shift into"
        );
        let mut bytes = Vec::new();
        let mut mask = Vec::new();
        for ex in &examples {
            bytes.extend_from_slice(&ex.bytes);
            mask.extend_from_slice(&ex.mask);
        }
        let need = batch * seq_len;
        if bytes.len() < need {
            return Err(SftError::TooSmall {
                have: bytes.len(),
                need,
            });
        }
        Ok(Self {
            bytes,
            mask,
            batch,
            seq_len,
            pos: 0,
        })
    }

    /// Load a `.jsonl` file and pack it. The one call a `--sft-file` flag needs.
    pub fn from_jsonl(path: &Path, batch: usize, seq_len: usize) -> Result<Self, SftError> {
        Self::new(read_jsonl(path)?, batch, seq_len)
    }

    /// Total trainable bytes in the corpus — the number to quote when someone
    /// asks how much of the SFT set is actually supervised.
    pub fn trainable(&self) -> usize {
        self.mask.iter().filter(|&&m| m != 0.0).count()
    }

    /// Next batch, or [`SftError::Exhausted`] at the end of the corpus.
    pub fn next_batch(&mut self) -> Result<SftBatch, SftError> {
        let need = self.batch * self.seq_len;
        if self.pos + need > self.bytes.len() {
            return Err(SftError::Exhausted);
        }
        let bytes = self.bytes[self.pos..self.pos + need].to_vec();
        let mask = self.mask[self.pos..self.pos + need].to_vec();
        self.pos += need;
        Ok(SftBatch { bytes, mask })
    }

    /// Restart from the first byte, so two evals score the same window — the
    /// [`crate::ByteStream::rewind`] property, without which cross-run SFT
    /// numbers are stream position (the 6.443-vs-6.551 defect).
    pub fn rewind(&mut self) {
        self.pos = 0;
    }
}
