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

pub struct JepaTargetWriter {
    w: BufWriter<std::fs::File>,
}

impl JepaTargetWriter {
    pub fn create(path: &std::path::Path) -> std::io::Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        Ok(Self {
            w: BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?),
        })
    }

    /// One record: chunk bytes (hash key) + teacher latent `[b, t, d]`.
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

    pub fn flush(&mut self) -> std::io::Result<()> {
        self.w.flush()
    }
}

/// Frozen JEPA targets keyed by the training chunk's byte hash.
pub struct JepaTargets {
    file: BufReader<std::fs::File>,
    index: HashMap<u64, u64>,
    device: burn::tensor::Device,
}

impl JepaTargets {
    /// Scan the sidecar and build the hash -> offset index. Errors loudly on
    /// a truncated file (every record must be complete).
    pub fn open(path: &std::path::Path, device: &burn::tensor::Device) -> Result<Self, String> {
        let raw = std::fs::File::open(path).map_err(|e| format!("jepa targets {}: {e}", path.display()))?;
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
            r.read_exact(&mut rest)
                .map_err(|_| format!("jepa targets {}: truncated record at {pos}", path.display()))?;
            let hash = u64::from_le_bytes(hdr);
            let b = u32::from_le_bytes(rest[0..4].try_into().unwrap()) as u64;
            let t = u32::from_le_bytes(rest[4..8].try_into().unwrap()) as u64;
            let d = u32::from_le_bytes(rest[8..12].try_into().unwrap()) as u64;
            let payload = b * t * d * 4;
            std::io::copy(
                &mut r.by_ref().take(payload),
                &mut std::io::sink(),
            )
            .map_err(|e| e.to_string())?;
            index.entry(hash).or_insert(pos);
        }
        if index.is_empty() {
            return Err(format!("jepa targets {}: no records", path.display()));
        }
        let file = BufReader::new(std::fs::File::open(path).map_err(|e| e.to_string())?);
        println!("jepa targets: {} records from {}", index.len(), path.display());
        Ok(Self { file, index, device: device.clone() })
    }

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
        let vals: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(Tensor::from_data(TensorData::new(vals, [b, t, d]), &self.device))
    }
}
