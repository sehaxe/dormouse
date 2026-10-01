//! Offline JEPA targets (architecture direction v2): the EMA teacher's
//! latent, precomputed once per training chunk, replaces the per-step
//! teacher forward (a second full forward that doubled VRAM and OOMed
//! batch 10 on 16 GB).
//!
//! Sidecar layout, little-endian, one record per training chunk:
//! `[fnv64(chunk bytes) u64][b u32][t u32][d u32][latents f32 x b*t*d]`.
//!
//! The reader indexes hash -> file offset in one O(records) scan at open
//! and seeks per lookup, so training never holds more than one record in
//! RAM and stays correct even if the stream position shifts between the
//! precompute pass and training (warmup, quant-check nibble batches).

use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

use burn::tensor::{Tensor, TensorData};

use dormouse_data::fnv;

/// The WRITER half: a streaming appender for the sidecar. One instance per
/// precompute pass, and it never holds more than the 1 MB buffer, so a
/// precompute over a whole corpus costs a constant amount of RAM on top of the
/// model.
///
/// `pub` in a private module (`mod jepa_targets`), so this is crate-internal:
/// the pair is reached through `precompute_jepa_targets` and the trainer's
/// `--jepa-targets` path, not by name.
pub struct JepaTargetWriter {
    w: BufWriter<std::fs::File>,
}

impl JepaTargetWriter {
    /// Create (or TRUNCATE) the sidecar, making the parent directory if it is
    /// missing. Truncating is deliberate: a precompute pass is a whole-file
    /// job, and appending to a previous pass would produce a file whose index
    /// resolves a hash to a stale record.
    pub fn create(path: &std::path::Path) -> std::io::Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        Ok(Self {
            w: BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?),
        })
    }

    /// One record: chunk bytes (hash key) + teacher latent `[b, t, d]`.
    ///
    /// The key is the FNV hash of the CHUNK BYTES, not a counter, which is
    /// what lets the reader find a record after the stream has moved (the
    /// module docs' second paragraph). The dims are written into the record
    /// rather than into a file header, so a sidecar can hold records from more
    /// than one shape and each carries its own.
    ///
    /// LOUD on a shape that disagrees with the payload: `latent.len()` is
    /// asserted against `b*t*d`, because a mismatch would be written as a
    /// short record and the reader would find the NEXT record's bytes here.
    pub fn push(
        &mut self,
        chunk_bytes: &[u8],
        latent: &[f32],
        b: usize,
        t: usize,
        d: usize,
    ) -> std::io::Result<()> {
        assert_eq!(latent.len(), b * t * d);
        self.w.write_all(&fnv(chunk_bytes).to_le_bytes())?;
        self.w.write_all(&(b as u32).to_le_bytes())?;
        self.w.write_all(&(t as u32).to_le_bytes())?;
        self.w.write_all(&(d as u32).to_le_bytes())?;
        for x in latent {
            self.w.write_all(&x.to_le_bytes())?;
        }
        Ok(())
    }

    /// Push the 1 MB buffer out. Called once at the end of a precompute pass:
    /// a pass that is killed before this loses up to a buffer of records, and
    /// the resulting file is then TRUNCATED-relative - its last record is
    /// partial, which [`JepaTargets::open`] refuses loudly rather than reading.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.w.flush()
    }
}

/// Frozen JEPA targets, keyed by the training chunk's byte hash, read from a
/// sidecar. The replacement for a per-step EMA-teacher forward: one forward per
/// batch at precompute time instead of a second full forward per step, which
/// is what OOMed batch 10 on 16 GB.
///
/// The index is a `hash -> file offset` map built once at open, so a lookup
/// is a hash and a seek and training holds ONE record in RAM no matter how big
/// the sidecar is. `pub` in a private module: reached through the trainer's
/// `--jepa-targets` path, not by name.
pub struct JepaTargets {
    file: BufReader<std::fs::File>,
    index: HashMap<u64, u64>,
    device: burn::tensor::Device,
}

impl JepaTargets {
    /// Scan the sidecar and build the hash -> offset index. Errors loudly on
    /// a truncated file (every record must be complete).
    pub fn open(path: &std::path::Path, device: &burn::tensor::Device) -> Result<Self, String> {
        let raw = std::fs::File::open(path)
            .map_err(|e| format!("jepa targets {}: {e}", path.display()))?;
        let mut r = BufReader::new(raw);
        let mut index = HashMap::new();
        const HEADER: usize = 8 + 4 + 4 + 4;
        loop {
            let pos = r.stream_position().map_err(|e| e.to_string())?;
            let mut hdr = [0u8; 8];
            if r.read_exact(&mut hdr).is_err() {
                break; // clean EOF
            }
            let mut rest = [0u8; HEADER - 8];
            r.read_exact(&mut rest).map_err(|_| {
                format!("jepa targets {}: truncated record at {pos}", path.display())
            })?;
            let hash = u64::from_le_bytes(hdr);
            let b = u32::from_le_bytes(rest[0..4].try_into().unwrap()) as u64;
            let t = u32::from_le_bytes(rest[4..8].try_into().unwrap()) as u64;
            let d = u32::from_le_bytes(rest[8..12].try_into().unwrap()) as u64;
            let payload = b * t * d * 4;
            std::io::copy(&mut r.by_ref().take(payload), &mut std::io::sink())
                .map_err(|e| e.to_string())?;
            index.entry(hash).or_insert(pos);
        }
        if index.is_empty() {
            return Err(format!("jepa targets {}: no records", path.display()));
        }
        let file = BufReader::new(std::fs::File::open(path).map_err(|e| e.to_string())?);
        println!(
            "jepa targets: {} records from {}",
            index.len(),
            path.display()
        );
        Ok(Self {
            file,
            index,
            device: device.clone(),
        })
    }

    /// How many distinct chunks the sidecar holds — the number a reader
    /// compares against the precompute step count: fewer records than steps
    /// means the precompute pass was cut short. `open()` already prints the
    /// count (from the index directly), and the lib target itself has no
    /// caller — the accessor is for the reader side, which is why dead_code
    /// is allowed here and not deleted.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Target latent for this training chunk (its byte hash must be in the
    /// sidecar). The device syncs here once per step - the same number of
    /// syncs the teacher forward used to force.
    pub fn get(&mut self, chunk_bytes: &[u8]) -> Result<Tensor<3>, String> {
        let hash = fnv(chunk_bytes);
        let Some(&pos) = self.index.get(&hash) else {
            return Err(format!(
                "jepa targets: chunk hash {hash:016x} not precomputed (steps/params mismatch?)"
            ));
        };
        self.file
            .seek(SeekFrom::Start(pos))
            .map_err(|e| e.to_string())?;
        let mut hdr = [0u8; 8 + 4 + 4 + 4];
        self.file.read_exact(&mut hdr).map_err(|e| e.to_string())?;
        let b = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
        let t = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
        let d = u32::from_le_bytes(hdr[16..20].try_into().unwrap()) as usize;
        let n = b * t * d;
        let mut raw = vec![0u8; n * 4];
        self.file.read_exact(&mut raw).map_err(|e| e.to_string())?;
        let (chunks, _) = raw.as_chunks::<4>();
        let vals: Vec<f32> = chunks.iter().map(|c| f32::from_le_bytes(*c)).collect();
        Ok(Tensor::from_data(
            TensorData::new(vals, [b, t, d]),
            &self.device,
        ))
    }
}
