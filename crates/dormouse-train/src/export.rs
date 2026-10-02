//! The inference export: a training checkpoint with the optimizer dropped and
//! the fp32 masters narrowed, in one self-describing file.
//!
//! WHY THIS EXISTS. The training container is not a model, it is a *run*: the
//! model section plus the optimizer section plus the EMA teacher, next to a
//! `.ngram` sidecar that reaches 34 GB. `generate` and `serve` read that run,
//! so moving a 7.5M-parameter model costs 34 GB and an outsider has to take the
//! whole thing or nothing. This module is the other door: one command, one
//! file, no optimizer, no sidecar, no host RAM.
//!
//! THE FORMAT (`.dmexp`, a public interface - see README):
//!
//! ```text
//! [0..8)    magic "DMEXP" + format version byte
//! [8..12)   header length, u32 LE
//! [12..20)  payload length in bytes, u64 LE
//! [20..24)  CRC-32 (IEEE) of the payload, u32 LE
//! [24..24+H)  header, UTF-8 TOML
//! [24+H..)    payload: the tensors, in header order, contiguous
//! ```
//!
//! The header carries everything a reader needs to identify the file without
//! this repository: the weight dtype, the full `DormouseConfig` (so the model
//! SHAPE travels with the weights - a shape the reader has to guess is a shape
//! it can get wrong), every tensor's name and dims, the source checkpoint and
//! the step it was taken at, and the total parameter count.
//!
//! WHY THE NARROWING HAPPENS ON THE HOST. `f32 -> f16/bf16` is integer bit
//! manipulation, and so is the widening on load. Nothing here asks the backend
//! for a narrow float: bf16 matmul does not exist on this stack (ADR-0016, the
//! LLVM dialect has no bf16 type) and f16 matmul exists but is not what a
//! weight file should depend on. The export is a STORAGE format; the model
//! that comes out of it is fp32, widened back from the stored bits. That also
//! means `bf16` is safe to ship here even though bf16 compute is not.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::Path;

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::tensor::{Device, Tensor, TensorData};
use serde::{Deserialize, Serialize};

use dormouse_core::{DormouseConfig, DormouseModel};

/// Magic + format version. A reader that does not know this byte is not this
/// format; the version byte is what an incompatible layout change bumps.
pub const MAGIC: [u8; 8] = *b"DMEXPRT\x01";
/// magic (8) + header_len (4) + payload_len (8) + crc32 (4).
pub const FIXED: usize = 24;

/// The weight format of an export, and the only decision this format makes
/// public. `F16` is 2 bytes with a narrow exponent (max 65504, min normal
/// 6.1e-5); `Bf16` is 2 bytes with f32's exponent range and 8 mantissa bits.
/// `F32` is 4 bytes and is the reference every other format is measured
/// against, not a recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32,
    F16,
    Bf16,
}

impl DType {
    /// Parse a `--dtype` string. LOUD on anything else, naming the accepted
    /// set - an unknown format name must not fall back to f32 and produce a
    /// file twice the size that looks like the one that was asked for.
    ///
    /// The aliases are the union of what the flag's help, the README and the
    /// trainer's own `--quant` spelling use, so `fp16`/`half`/`float16` all
    /// reach [`DType::F16`].
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "fp32" | "float32" => Ok(DType::F32),
            "f16" | "fp16" | "float16" | "half" => Ok(DType::F16),
            "bf16" | "bfloat16" => Ok(DType::Bf16),
            _ => Err(format!("--dtype expected f32 | f16 | bf16, got {s:?}")),
        }
    }

    /// Bytes per stored value. The load path's stride: `dims.product() *
    /// bytes` is where the next tensor starts, so a wrong answer here would
    /// read every tensor at the wrong offset rather than fail.
    pub fn bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::Bf16 => 2,
        }
    }

    /// Host narrowing. Round-to-nearest-even, and the caller checks the result
    /// for non-finite values, because a f16 overflow is an `inf` in a weight
    /// and an `inf` in a weight is a silently broken model.
    pub fn narrow(self, x: f32) -> f32 {
        match self {
            DType::F32 => x,
            DType::F16 => half::f16::from_f32(x).to_f32(),
            DType::Bf16 => half::bf16::from_f32(x).to_f32(),
        }
    }

    /// The inverse of [`DType::narrow`]: stored bits -> `f32`. Exact and
    /// lossless - widening an f16 or a bf16 always lands on a representable
    /// f32 - so the exported model is fp32 arithmetic over weights that
    /// round-tripped through the narrow format. That is the whole reason the
    /// narrowing happens on the host (module docs, "WHY THE NARROWING").
    pub fn widen(self, bits: u16) -> f32 {
        match self {
            DType::F32 => f32::from_bits(u32::from(bits)),
            DType::F16 => half::f16::from_bits(bits).to_f32(),
            DType::Bf16 => half::bf16::from_bits(bits).to_f32(),
        }
    }

    fn pack(self, x: f32, out: &mut Vec<u8>) {
        match self {
            DType::F32 => out.extend_from_slice(&x.to_le_bytes()),
            DType::F16 => out.extend_from_slice(&half::f16::from_f32(x).to_bits().to_le_bytes()),
            DType::Bf16 => out.extend_from_slice(&half::bf16::from_f32(x).to_bits().to_le_bytes()),
        }
    }

    fn unpack(self, bytes: &[u8], i: usize) -> f32 {
        let b = |k: usize| u16::from_le_bytes([bytes[i + k], bytes[i + k + 1]]);
        match self {
            DType::F32 => f32::from_le_bytes(bytes[i..i + 4].try_into().expect("4-byte window")),
            DType::F16 => self.widen(b(0)),
            DType::Bf16 => self.widen(b(0)),
        }
    }
}

/// One tensor as the header names it. The name is the burn module path
/// (`loop_block.lm_head.u.weight`), so the header is also the map from the
/// file's bytes back to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    /// The burn module path, identical to the name the loader matches by —
    /// and the reason a model's field order is not a file format.
    pub name: String,
    /// Shape, row-major. The tensor's slice of the payload is
    /// `dims.product() * dtype.bytes()` long, entries in list order, so this
    /// plus [`Header::dtype`] is the whole offset arithmetic.
    pub dims: Vec<usize>,
    /// Non-finite values AFTER narrowing. Always 0 in a file that exists -
    /// [`encode`] refuses to write one - so a reader can trust it.
    pub overflowed: u64,
}

/// Scalars first, tables last: that is TOML's own ordering rule, and this
/// struct is serialized by field order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    /// The magic's version byte, echoed so a reader can report it.
    pub format: u8,
    /// The one value dtype for the whole file: every tensor is stored in it,
    /// there is no per-tensor dtype, and the reader widens with
    /// [`DType::widen`] and nothing else.
    pub dtype: DType,
    /// Where this came from: the checkpoint it was made from, and the step.
    pub source: String,
    /// The checkpoint step the weights are from. Informational (an export
    /// never resumes training), but it is what makes two exports of one run
    /// distinguishable on a machine that has never seen the run.
    pub step: u64,
    /// Total scalar parameters written; the number a stranger checks the
    /// download's size against (`num_params * dtype.bytes()` ≈ payload).
    pub num_params: u64,
    /// Distinct tensors, one [`TensorEntry`] each in visit order. The loader
    /// refuses the file when this list and the model its own `config` builds
    /// disagree about names or shapes.
    pub num_tensors: usize,
    /// Weights whose magnitude fell below the format's smallest normal and
    /// were flushed to zero. The honest cost of a narrow exponent, counted
    /// rather than assumed.
    pub flushed: u64,
    /// Largest magnitude in the model, measured AFTER narrowing (encode
    /// measures the packed values), so it is the range the file actually
    /// holds rather than the range the f32 masters held.
    pub max_abs: f32,
    /// Smallest NON-zero magnitude, same measurement: the number that decides
    /// whether the format's smallest normal is above real weights (whose
    /// flush count is [`Header::flushed`]). `0.0` for an all-zero model,
    /// because there is no non-zero value to report.
    pub min_nonzero_abs: f32,
    /// The full model config, as the preset TOML. The shape travels with the
    /// weights because a reader that has to be told the shape separately can be
    /// told the wrong one.
    pub config: String,
    /// One entry per tensor, in visit (module-path) order: the map from the
    /// payload's bytes back to the model. `dormouse export --info` reads the
    /// header alone and never decompresses the payload to list these.
    pub tensors: Vec<TensorEntry>,
}

/// Everything the exporter measured on the way out. Printed by the command:
/// the format choice is a claim about numbers, so the numbers ship with it.
#[derive(Debug, Clone)]
pub struct Report {
    /// The stored format, as requested — `encode` never substitutes one (a
    /// value that does not fit is a loud `Err`, never a silently wider file).
    pub dtype: DType,
    /// Size of the written file, the numerator for the ratio against
    /// [`Report::source_bytes`].
    pub file_bytes: u64,
    /// Total scalars, read back through the header after a full `decode` of
    /// the just-written bytes — the report is measured on the artifact, not
    /// on the encoder's intentions.
    pub num_params: u64,
    /// Bytes of the training container this was made from, for the ratio.
    pub source_bytes: u64,
    /// The header's flush count, repeated so the command's printout does not
    /// have to re-parse the file it just wrote.
    pub flushed: u64,
    /// The measured range, the same numbers the header carries.
    pub max_abs: f32,
    /// Smallest non-zero magnitude after narrowing; `0.0` for an all-zero
    /// model. Same measurement as [`Header::min_nonzero_abs`].
    pub min_nonzero_abs: f32,
    /// Where the file landed — printed so the command's last line names the
    /// artifact and not just its numbers.
    pub path: String,
}

/// Collects every float parameter with its module path, in visit order.
struct Collector {
    path: Vec<String>,
    out: Vec<(String, Vec<usize>, Vec<f32>)>,
}

impl ModuleVisitor for Collector {
    fn enter_module(&mut self, name: &str, _container: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container: &str) {
        self.path.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let data = param.val().into_data();
        // LOUD, never `unwrap_or_default()`: a readback that failed as an
        // empty vec would export a model with a missing tensor (ADR-0019).
        let flat = data.try_to_vec::<f32>().unwrap_or_else(|e| {
            panic!(
                "export: reading {} back from the device failed: {e}",
                self.path.join(".")
            )
        });
        self.out
            .push((self.path.join("."), param.val().dims().to_vec(), flat));
    }

    fn visit_int<const D: usize>(&mut self, _param: &Param<Tensor<D, burn::tensor::Int>>) {
        panic!(
            "export: {} is an integer parameter and the format has no dtype for it",
            self.path.join(".")
        );
    }

    fn visit_bool<const D: usize>(&mut self, _param: &Param<Tensor<D, burn::tensor::Bool>>) {
        panic!(
            "export: {} is a bool parameter and the format has no dtype for it",
            self.path.join(".")
        );
    }
}

/// Every float parameter of `model`, by module path.
pub fn tensors_of(model: &DormouseModel) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut c = Collector {
        path: Vec::new(),
        out: Vec::new(),
    };
    model.visit(&mut c);
    c.out
}

/// Encode a model into the export file's bytes. Backend-agnostic: the narrowing
/// is host arithmetic over the parameters read back once.
pub fn encode(
    model: &DormouseModel,
    cfg: &DormouseConfig,
    source: &str,
    step: u64,
    dtype: DType,
) -> Result<Vec<u8>, String> {
    let tensors = tensors_of(model);
    let mut payload: Vec<u8> = Vec::new();
    let mut entries = Vec::with_capacity(tensors.len());
    let (mut num_params, mut flushed, mut max_abs, mut min_nz) =
        (0u64, 0u64, 0.0f32, f32::INFINITY);
    for (name, dims, flat) in &tensors {
        // A per-tensor count of values that stopped being numbers. The loop
        // below refuses the file if any is non-zero, so this is always 0 in a
        // file that exists - the header field is the reader's proof of that.
        let overflowed = 0u64;
        for &x in flat {
            let y = dtype.narrow(x);
            // A narrowed value that stopped being a number, or that is no
            // longer itself, is a broken weight file. Refuse to write it and
            // name the format: f16's exponent ends at 65504.
            if !y.is_finite() {
                return Err(format!(
                    "{name}: value {x} is not representable in {dtype:?} (max 65504) - \
                     this model cannot be exported in that format; try --dtype bf16, which \
                     has f32's exponent range"
                ));
            }
            if x != 0.0 && y == 0.0 {
                flushed += 1;
            }
            let a = y.abs();
            if a > max_abs {
                max_abs = a;
            }
            if a > 0.0 && a < min_nz {
                min_nz = a;
            }
            dtype.pack(y, &mut payload);
            num_params += 1;
        }
        entries.push(TensorEntry {
            name: name.clone(),
            dims: dims.clone(),
            overflowed,
        });
    }
    assert!(
        entries.iter().all(|e| e.overflowed == 0),
        "every non-finite narrowing was rejected above, so no entry may carry one"
    );
    let header = Header {
        format: MAGIC[7],
        dtype,
        source: source.to_string(),
        step,
        num_params,
        num_tensors: tensors.len(),
        flushed,
        max_abs,
        min_nonzero_abs: if min_nz.is_finite() { min_nz } else { 0.0 },
        config: toml::to_string(cfg).map_err(|e| format!("config serialize: {e}"))?,
        tensors: entries,
    };
    let header = toml::to_string(&header).map_err(|e| format!("header serialize: {e}"))?;
    let mut out = Vec::with_capacity(FIXED + header.len() + payload.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&(header.len() as u32).to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Decode export bytes into a model, the config it was written with, and the
/// header. Every failure is a named `Err` (ADR-0011): a file that is not an
/// export, a truncated file, a corrupt payload, a header that does not parse,
/// and a header whose tensor list does not match the model its own config
/// builds.
pub fn decode(raw: &[u8]) -> Result<(DormouseModel, DormouseConfig, Header), String> {
    if raw.len() < FIXED {
        return Err(format!(
            "{} bytes: too short to be a dormouse export",
            raw.len()
        ));
    }
    if raw[..8] != MAGIC {
        let path = Path::new("<export>");
        let why = wrong_file(path, raw);
        // `read` prefixes the real path; `decode` is given raw bytes, so strip
        // the placeholder's leading "<export>: " when there is no real path.
        return Err(why.trim_start_matches("<export>: ").to_string());
    }
    let hlen = u32::from_le_bytes(raw[8..12].try_into().expect("4-byte window")) as usize;
    let plen = u64::from_le_bytes(raw[12..20].try_into().expect("8-byte window")) as usize;
    let crc = u32::from_le_bytes(raw[20..24].try_into().expect("4-byte window"));
    let end = FIXED
        .checked_add(hlen)
        .and_then(|h| h.checked_add(plen))
        .ok_or_else(|| {
            "header + payload length overflows usize - the file is not what it claims".to_string()
        })?;
    if end != raw.len() {
        return Err(format!(
            "truncated or padded: header says {hlen} + {plen} payload bytes, the file has {}",
            raw.len().saturating_sub(FIXED)
        ));
    }
    let payload = &raw[FIXED + hlen..];
    let got = crc32fast::hash(payload);
    if got != crc {
        return Err(format!(
            "checksum mismatch: header says {crc:08x}, the payload hashes to {got:08x} - \
             the file did not arrive intact (re-download it; do not load it anyway)"
        ));
    }
    let header: Header =
        toml::from_str(header_str(raw, FIXED, hlen)?).map_err(|e| format!("header parse: {e}"))?;
    if header.format != MAGIC[7] {
        return Err(format!(
            "export format version {} is not the {} this build reads",
            header.format, MAGIC[7]
        ));
    }
    let cfg: DormouseConfig =
        toml::from_str(&header.config).map_err(|e| format!("config in the header: {e}"))?;
    let model = DormouseModel::new(&cfg, &device());
    let model = apply(model, &header, payload, header.dtype)?;
    Ok((model, cfg, header))
}

/// Rebuild the model's parameters from the payload, by name. Every mismatch is
/// an `Err`, not a panic: this is the public loader, and a hand-edited or
/// truncated file has to be reportable, not a backtrace.
fn apply(
    model: DormouseModel,
    header: &Header,
    payload: &[u8],
    dtype: DType,
) -> Result<DormouseModel, String> {
    let w = dtype.bytes();
    let mut left: HashMap<String, TensorData> = HashMap::with_capacity(header.tensors.len());
    let mut off = 0usize;
    for t in &header.tensors {
        let n: usize = t.dims.iter().product();
        let end = off + n * w;
        if end > payload.len() {
            return Err(format!(
                "{}: the header lists it with {n} elements at byte {off}, but the payload is \
                 only {} bytes",
                t.name,
                payload.len()
            ));
        }
        let flat: Vec<f32> = (off..end)
            .step_by(w)
            .map(|i| dtype.unpack(payload, i))
            .collect();
        left.insert(t.name.clone(), TensorData::new(flat, t.dims.clone()));
        off = end;
    }
    if off != payload.len() {
        return Err(format!(
            "the header's tensor list covers {off} of the payload's {} bytes - the two \
             disagree about the file's contents",
            payload.len()
        ));
    }
    let dev = device();
    let mut loader = Loader {
        path: Vec::new(),
        data: left,
        dev,
        missing: Vec::new(),
        refused: Vec::new(),
        mismatched: Vec::new(),
    };
    let out = model.map(&mut loader);
    if !loader.missing.is_empty() {
        return Err(format!(
            "the config in the header builds a model whose parameters the file does not carry: \
             {:?} - this file was written for a different architecture",
            loader.missing
        ));
    }
    if !loader.mismatched.is_empty() {
        return Err(format!(
            "the config in the header does not describe the weights in the payload: {:?}. \
             A file like this loads and then predicts nonsense, so it is refused.",
            loader.mismatched
        ));
    }
    if !loader.refused.is_empty() {
        return Err(format!(
            "the file carries no weight dtype for these parameters: {:?} - they are integer or \
             boolean state, and this format stores floats",
            loader.refused
        ));
    }
    // Inference, not training: no grad-carrying leaves on a weight file.
    Ok(out.no_grad())
}

/// The path-tracking loader, mirroring burn's own record mapper: the module
/// path is built as the traversal descends, so a value is matched BY NAME and
/// not by traversal order. A model's field order is not a file format.
struct Loader {
    path: Vec<String>,
    data: HashMap<String, TensorData>,
    dev: Device,
    missing: Vec<String>,
    refused: Vec<String>,
    mismatched: Vec<String>,
}

impl ModuleMapper for Loader {
    fn enter_module(&mut self, name: &str, _container: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container: &str) {
        self.path.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let name = self.path.join(".");
        let Some(data) = self.data.remove(&name) else {
            self.missing.push(name);
            return param;
        };
        // The SHAPE check, and it is not optional. `Param::from_data` accepts
        // whatever dims the payload says, so a file whose header config
        // disagrees with its own tensors installs correctly-shaped-for-the-file
        // weights into a differently-shaped model: the load succeeds, the
        // forward returns garbage, and nothing downstream complains. That is
        // the cardinal sin (ADR-0011), and it is reachable by editing one
        // number in a header.
        let want = param.val().dims();
        let got: Vec<usize> = data.shape.to_vec();
        if want.to_vec() != got {
            self.mismatched.push(format!(
                "{name}: model wants {want:?}, file carries {got:?}"
            ));
            return param;
        }
        Param::from_data(data, &self.dev)
    }

    fn map_int<const D: usize>(
        &mut self,
        param: Param<Tensor<D, burn::tensor::Int>>,
    ) -> Param<Tensor<D, burn::tensor::Int>> {
        self.refused.push(self.path.join("."));
        param
    }

    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<D, burn::tensor::Bool>>,
    ) -> Param<Tensor<D, burn::tensor::Bool>> {
        self.refused.push(self.path.join("."));
        param
    }
}

/// Read an export from disk.
pub fn read(path: &Path) -> Result<(DormouseModel, DormouseConfig, Header), String> {
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    decode(&raw).map_err(|e| {
        // The wrong-file sentence already names the path; do not say it twice.
        if e.starts_with(&path.display().to_string()) {
            e
        } else {
            format!("{}: {e}", path.display())
        }
    })
}

/// Write an export, atomically (tmp + rename), and report what it cost.
pub fn write(
    path: &Path,
    model: &DormouseModel,
    cfg: &DormouseConfig,
    source: &str,
    source_bytes: u64,
    step: u64,
    dtype: DType,
) -> Result<Report, String> {
    let raw = encode(model, cfg, source, step, dtype)?;
    let tmp = path.with_extension(format!("dmexp.tmp.{}", std::process::id()));
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let f = std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
    w.write_all(&raw)
        .map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::io::Write::flush(&mut w).map_err(|e| format!("{}: {e}", tmp.display()))?;
    drop(w);
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", tmp.display()))?;
    let (_, _, h) = decode(&raw)?;
    Ok(Report {
        dtype,
        file_bytes: raw.len() as u64,
        num_params: h.num_params,
        source_bytes,
        flushed: h.flushed,
        max_abs: h.max_abs,
        min_nonzero_abs: h.min_nonzero_abs,
        path: path.display().to_string(),
    })
}

/// Turn a training checkpoint into an export, end to end.
///
/// The config comes from `<ckpt_name>.config.toml`, the snapshot the run wrote
/// next to its weights - NOT from a `--preset` on this command. A preset is a
/// guess about a shape; the snapshot is the shape the weights were trained
/// under, and a reader who is handed the wrong one gets a model that does not
/// fail until its first matmul.
///
/// The `.ngram` sidecar is never opened. It is the whole point: 34 GB of
/// n-gram memory for a run whose in-model tables already hold the arm.
pub fn export_ckpt(
    ckpt: &Path,
    preset: Option<&str>,
    set: &[String],
    dtype: DType,
    out: &Path,
) -> Result<Report, String> {
    let raw = std::fs::read(ckpt).map_err(|e| format!("{}: {e}", ckpt.display()))?;
    if raw.len() >= 8 && raw[..8] == MAGIC {
        return Err(format!("{} is already an export", ckpt.display()));
    }
    let h = crate::parse_header(&raw)
        .ok_or_else(|| format!("{} is not a dormouse training checkpoint", ckpt.display()))?;
    if h.body + h.model > raw.len() {
        return Err(format!(
            "{}: the model section runs past the end of the file",
            ckpt.display()
        ));
    }
    let cfg = config_for(ckpt, preset, set)?;
    let rec = burn::store::ModuleRecord::from_bytes(burn::tensor::Bytes::from_bytes_vec(
        raw[h.body..h.body + h.model].to_vec(),
    ))
    .map_err(|e| format!("{}: model section: {e}", ckpt.display()))?;
    let model = DormouseModel::new(&cfg, &device()).load_record(rec);
    // The model section of a training container is allowed to be 0 bytes: a
    // save that died mid-write leaves the header and nothing else, and an
    // export of that is a file of zeroes that loads fine and predicts
    // garbage. Refuse it here, where the byte count is still known.
    assert!(
        h.model > 0,
        "{}: the container's model section is empty - refusing to export it",
        ckpt.display()
    );
    write(
        out,
        &model,
        &cfg,
        &format!("{}@step{}", ckpt.display(), h.step),
        raw.len() as u64,
        h.step,
        dtype,
    )
}

/// The model config for a checkpoint: its own snapshot, or an explicit
/// `--preset` the caller has to name. There is no third option and no
/// default: a missing config that silently became the `small` schema is a
/// 30 MB file of the wrong shape.
fn config_for(ckpt: &Path, preset: Option<&str>, set: &[String]) -> Result<DormouseConfig, String> {
    let dir = ckpt.parent().unwrap_or(Path::new("."));
    let name = ckpt
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.trim_end_matches(".bin"))
        .unwrap_or("");
    let snap = dir.join(format!("{name}.config.toml"));
    if snap.exists() {
        let text =
            std::fs::read_to_string(&snap).map_err(|e| format!("{}: {e}", snap.display()))?;
        let run = crate::RunCfg::from_snapshot(&text)?;
        return Ok(run.model);
    }
    let preset = preset.ok_or_else(|| {
        format!(
            "no config for this checkpoint: {} does not exist, so there is no record of the shape \
         its weights were trained under. Pass --preset <name> (or --config <path>) if you know it.",
            snap.display()
        )
    })?;
    Ok(crate::resolve(preset, set, Default::default())?.model)
}

/// The header of an export, without loading the model: what the file IS.
///
/// Every failure is an `Err` naming the cause. A corrupt download must print a
/// message, not a panic backtrace - the reader is a person deciding whether to
/// re-fetch, not a programmer.
pub fn inspect(path: &Path) -> Result<(Header, usize), String> {
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if raw.len() < FIXED {
        return Err(format!(
            "{}: {} bytes, too short to be a dormouse export",
            path.display(),
            raw.len()
        ));
    }
    if raw[..8] != MAGIC {
        return Err(wrong_file(path, &raw));
    }
    let hlen = u32::from_le_bytes(raw[8..12].try_into().expect("4-byte window")) as usize;
    let plen = u64::from_le_bytes(raw[12..20].try_into().expect("8-byte window")) as usize;
    let crc = u32::from_le_bytes(raw[20..24].try_into().expect("4-byte window"));
    let end = FIXED
        .checked_add(hlen)
        .and_then(|h| h.checked_add(plen))
        .ok_or_else(|| {
            format!(
                "{}: header + payload length overflows usize",
                path.display()
            )
        })?;
    if end != raw.len() {
        return Err(format!(
            "{}: truncated or padded - the header says {hlen} + {plen} payload bytes, the file \
             has {} after the fixed part",
            path.display(),
            raw.len().saturating_sub(FIXED)
        ));
    }
    let got = crc32fast::hash(&raw[FIXED + hlen..]);
    if got != crc {
        return Err(format!(
            "{}: checksum {got:08x} does not match the {crc:08x} in the header - the file did \
             not arrive intact. Re-download it; do not load it anyway.",
            path.display()
        ));
    }
    let header: Header = toml::from_str(header_str(&raw, FIXED, hlen)?)
        .map_err(|e| format!("{}: header parse: {e}", path.display()))?;
    Ok((header, raw.len()))
}

/// What a file that is not an export actually is, and the command that fixes
/// it. The one place a wrong input gets a sentence instead of "invalid".
fn wrong_file(path: &Path, raw: &[u8]) -> String {
    if raw.len() >= 8 && raw[..8] == crate::CKPT_MAGIC {
        return format!(
            "{}: this is a TRAINING CHECKPOINT, not an inference export. It carries the \
             optimizer state, and a run usually sits next to a 34 GB .ngram sidecar. Convert \
             it: dormouse export run --ckpt-dir <dir> --ckpt-name <name> --dtype bf16",
            path.display()
        );
    }
    format!(
        "{}: not a dormouse export - magic is {:?}, expected {:?}. If it is a checkpoint, export \
         it first; if it is something else, this loader will not guess.",
        path.display(),
        &raw[..8],
        &MAGIC[..]
    )
}

/// Bytes of a file's header as `&str`. A header that is not UTF-8 is not a
/// header, and saying so beats a lossy conversion.
fn header_str(raw: &[u8], at: usize, len: usize) -> Result<&str, String> {
    std::str::from_utf8(&raw[at..at + len]).map_err(|e| format!("header is not utf-8: {e}"))
}

/// The one device this crate uses. A duplicate of the crate's own factory is
/// exactly how a test ends up comparing two backends, so this delegates.
fn device() -> Device {
    crate::device()
}
