//! Host-side n-gram tables (Qwen3.8-Flash-Next §2.3).
//!
//! Deterministic FNV addressing means the tables never need random access
//! from the GPU: each batch's rows are gathered on the host, copied to the
//! device (~600 KB for batch 3 x s512), and trained with plain Adam on the
//! CPU (report: n-gram tables run on Adam with weight decay disabled). The
//! VRAM cost of the memory scales with the batch's unique rows, not with the
//! table size, so millions of slots live in the 64 GB RAM for free.
//!
//! Tables are [slots, dim] f32 row-major, one per n-gram order; the hash
//! indices arrive pre-moduloed by the slot count (data crate).

use std::collections::HashMap;

use burn::backend::{Backend, DispatchKindConversion};
use burn::module::Param;
use burn::tensor::{DispatchTensor, Int, Tensor, TensorData};

/// 3 tables (3/5/8-gram), `dim` columns each.
pub struct HostNgram {
    pub slots: [usize; 3],
    pub dim: usize,
    tables: Vec<f32>, // concatenated [sum(slots), dim]
    m: Vec<f32>,      // Adam first moment, same layout
    v: Vec<f32>,      // Adam second moment, same layout
    step: u64,
}

fn splitmix(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl HostNgram {
    /// Fresh tables drawn from N(0, 0.02) (same scale as the in-model
    /// embedding init), deterministic for a seed.
    pub fn new(slots: [usize; 3], dim: usize, seed: u64) -> Self {
        let total = slots.iter().sum::<usize>() * dim;
        let mut rng = seed;
        let mut tables = Vec::with_capacity(total);
        for _ in 0..total {
            // Box-Muller from two uniforms.
            let u1 = (splitmix(&mut rng) >> 40) as f64 / (1u64 << 24) as f64;
            let u2 = (splitmix(&mut rng) >> 40) as f64 / (1u64 << 24) as f64;
            let z = (-2.0 * u1.max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            tables.push((z * 0.02) as f32);
        }
        Self {
            slots,
            dim,
            tables,
            m: vec![0.0; total],
            v: vec![0.0; total],
            step: 0,
        }
    }

    /// Absolute row offset of table `t` in the concatenated layout (rows).
    pub fn table_base(&self, t: usize) -> usize {
        self.slots[..t].iter().sum()
    }

    pub fn total_rows(&self) -> usize {
        self.slots.iter().sum()
    }

    /// Bytes of one row.
    pub fn row_bytes(&self) -> usize {
        self.dim * 4
    }

    /// Deduplicate the batch's `[b*t*3]` table-local indices into
    /// `(unique_abs, pos)` where `pos[i]` maps position i to its unique row.
    /// A row index is `table_base(t) + (idx % slots[t])`.
    pub fn unique_rows(&self, hashes: &[i64]) -> (Vec<i64>, Vec<i64>) {
        let mut map: HashMap<i64, i64> = HashMap::with_capacity(hashes.len());
        let mut uniq: Vec<i64> = Vec::with_capacity(hashes.len());
        let mut pos = Vec::with_capacity(hashes.len());
        for (i, h) in hashes.iter().enumerate() {
            let t = i % 3;
            let abs = self.table_base(t) as i64 + (h % self.slots[t] as i64);
            let u = *map.entry(abs).or_insert_with(|| {
                uniq.push(abs);
                (uniq.len() - 1) as i64
            });
            pos.push(u);
        }
        (uniq, pos)
    }

    /// Row vectors for `indices` (absolute rows), flat `[n*dim]`.
    pub fn gather(&self, indices: &[i64], out: &mut Vec<f32>) {
        out.clear();
        out.reserve(indices.len() * self.dim);
        for &r in indices {
            let base = (r as usize) * self.dim;
            out.extend_from_slice(&self.tables[base..base + self.dim]);
        }
    }

    /// Plain Adam step (weight decay disabled, report §2.3) on the given
    /// rows, in place. `grads` must be flat `[n*dim]` for the same indices.
    pub fn adam_update(&mut self, indices: &[i64], grads: &[f32], lr: f32) {
        self.step += 1;
        let b1 = 0.9f32;
        let b2 = 0.999f32;
        let eps = 1e-8f32;
        let bc1 = 1.0 / (1.0 - b1.powi(self.step as i32));
        let bc2 = (1.0 - b2.powi(self.step as i32)).sqrt();
        for (k, &r) in indices.iter().enumerate() {
            let base = (r as usize) * self.dim;
            for c in 0..self.dim {
                let g = grads[k * self.dim + c];
                let mi = base + c;
                self.m[mi] = b1 * self.m[mi] + (1.0 - b1) * g;
                self.v[mi] = b2 * self.v[mi] + (1.0 - b2) * g * g;
                let step = self.m[mi] * bc1 / (self.v[mi].sqrt() * bc2 + eps);
                self.tables[mi] -= lr * step;
            }
        }
    }

    /// Serialize the tables + Adam state (checkpointing; ~row_bytes * rows).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.tables.len() * 4 * 3 + 8);
        buf.extend_from_slice(&self.step.to_le_bytes());
        for chunk in [&self.tables, &self.m, &self.v] {
            for &x in chunk {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        buf
    }

    pub fn from_bytes(bytes: &[u8], slots: [usize; 3], dim: usize) -> Option<Self> {
        let total = slots.iter().sum::<usize>() * dim;
        if bytes.len() < 8 + total * 4 * 3 {
            return None;
        }
        let step = u64::from_le_bytes(bytes[..8].try_into().ok()?);
        let mut it = bytes[8..].chunks_exact(4);
        let mut read = |n: usize| -> Option<Vec<f32>> {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                let b: [u8; 4] = it.next()?.try_into().ok()?;
                v.push(f32::from_le_bytes(b));
            }
            Some(v)
        };
        Some(Self {
            slots,
            dim,
            tables: read(total)?,
            m: read(total)?,
            v: read(total)?,
            step,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_adam_roundtrip() {
        let mut h = HostNgram::new([64, 64, 64], 8, 7);
        // 9 positions, 3 tables; each table-local index repeats 3x.
        let hashes: Vec<i64> = vec![5, 7, 9, 5, 7, 9, 5, 7, 9];
        let (uniq, pos) = h.unique_rows(&hashes);
        assert_eq!(uniq.len(), 3);
        assert_eq!(pos.len(), 9);
        assert!(pos.iter().all(|&p| (p as usize) < uniq.len()));
        let mut rows = Vec::new();
        h.gather(&uniq, &mut rows);
        assert_eq!(rows.len(), uniq.len() * 8);
        // Adam step must move the gathered rows.
        let before: Vec<f32> = rows.clone();
        let grads: Vec<f32> = rows.iter().map(|x| x * 0.1).collect();
        h.adam_update(&uniq, &grads, 1e-3);
        let mut after = Vec::new();
        h.gather(&uniq, &mut after);
        assert!(before != after, "adam update must change the rows");
        // Roundtrip through bytes.
        let bytes = h.to_bytes();
        let h2 = HostNgram::from_bytes(&bytes, [64, 64, 64], 8).expect("ckpt load");
        let mut after2 = Vec::new();
        h2.gather(&uniq, &mut after2);
        assert_eq!(after, after2, "ckpt roundtrip must preserve rows");
    }
}

/// Gather the batch's rows from RAM into an autodiff leaf and expand them
/// to the `[b, t, 3*dim]` embedding the model consumes. With `track=false`
/// (eval) the leaf is skipped and only the embedding is returned.
pub fn rows_for_batch<B: Backend>(
    host: &HostNgram,
    hashes: &[i64],
    b: usize,
    t: usize,
    device: &burn::tensor::Device,
    track: bool,
) -> (Option<Param<Tensor<2>>>, Tensor<3>)
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let (uniq, pos) = host.unique_rows(hashes);
    let mut rows = Vec::new();
    host.gather(&uniq, &mut rows);
    let rows_t: Tensor<2> = Tensor::from_data(TensorData::new(rows, [uniq.len(), host.dim]), device);
    let p = track.then(|| {
        let r = rows_t.clone().require_grad();
        Param::from_tensor(r.into())
    });
    let src = match &p {
        Some(pp) => pp.val(),
        None => rows_t,
    };
    let pos_t: Tensor<1, Int> = Tensor::from_data(TensorData::new(pos, [b * t * 3]), device);
    let idx2 = pos_t.unsqueeze_dim::<2>(1).repeat(&[1, host.dim]);
    let embed = src.gather(0, idx2).reshape([b, t, 3 * host.dim]);
    (p, embed)
}