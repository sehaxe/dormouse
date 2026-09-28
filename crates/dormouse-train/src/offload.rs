//! Host-side n-gram tables (Qwen3.8-Flash-Next §2.3).
//!
//! Deterministic FNV addressing means the tables never need random access
//! from the GPU: each batch's rows are gathered on the host, copied to the
//! device (~600 KB for batch 3 x s512), and trained on the CPU with Nesterov
//! momentum + periodic Sinkhorn balancing (DeepSeek V4.1-Flash §2.5: the
//! embedding tables and head run on one momentum buffer instead of Adam's
//! m+v - halving the optimizer state - with no weight decay). The VRAM cost
//! of the memory scales with the batch's unique rows, not with the table
//! size, so millions of slots live in the 64 GB RAM for free.
//!
//! RAM per row: table 128 B + momentum 128 B = 256 B (x2 total), down from
//! Adam's table + m + v = 384 B (x3). At the real18 size (48M rows) that is
//! 12.3 GB instead of 18.4 GB.
//!
//! Tables are [slots, dim] f32 row-major, one per n-gram order; the hash
//! indices arrive pre-moduloed by the slot count (data crate).

use std::collections::HashMap;

use burn::backend::{Backend, DispatchKindConversion};
use burn::module::Param;
use burn::tensor::{DispatchTensor, Int, Tensor, TensorData};

/// Nesterov momentum (same value Muon+ uses for its 2D group).
const MOMENTUM: f32 = 0.95;
/// Sinkhorn-balance the batch's update block every K updates (between the
/// balanced steps the update is plain Nesterov).
const SINKHORN_EVERY: u64 = 4;
/// Alternating L1 row/column normalizations per balanced update (SinkGD
/// uses a handful; 2 is enough to pull skewed rows toward the mean).
const SINKHORN_ITERS: usize = 2;
/// Layout magic ("2MGN" little-endian). v1 files (Adam m+v) start with the
/// step counter and can never collide with this; a mismatch fails the load
/// loudly instead of misreading the state.
const LAYOUT_MAGIC: u32 = 0x4D_47_4E_32;

/// 3 tables (3/5/8-gram), `dim` columns each.
pub struct HostNgram {
    pub slots: [usize; 3],
    pub dim: usize,
    tables: Vec<f32>, // concatenated [sum(slots), dim]
    m: Vec<f32>,      // momentum, same layout (the only optimizer state)
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
    /// A row index is `table_base(t) + rem_euclid(idx, slots[t])`.
    /// `rem_euclid`, not `%`: a negative `%` indexes the table before its
    /// base, and the raw FNV hashes the in-VRAM path produces (the model
    /// masks those) arrive as i32, i.e. sometimes negative.
    pub fn unique_rows(&self, hashes: &[i64]) -> (Vec<i64>, Vec<i64>) {
        let mut map: HashMap<i64, i64> = HashMap::with_capacity(hashes.len());
        let mut uniq: Vec<i64> = Vec::with_capacity(hashes.len());
        let mut pos = Vec::with_capacity(hashes.len());
        for (i, h) in hashes.iter().enumerate() {
            let t = i % 3;
            let abs = self.table_base(t) as i64 + h.rem_euclid(self.slots[t] as i64);
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

    /// Nesterov momentum step (weight decay disabled, DeepSeek V4.1-Flash
    /// §2.3) on the given rows, in place. `grads` must be flat `[n*dim]` for
    /// the same indices. Every [`SINKHORN_EVERY`]-th update, the batch's
    /// update block is Sinkhorn-balanced (alternating L1 row/column
    /// normalization, magnitude-preserving) before it is applied; the work
    /// stays O(batch rows) - the full table is never touched.
    pub fn momentum_update(&mut self, indices: &[i64], grads: &[f32], lr: f32) {
        self.step += 1;
        let n = indices.len();
        debug_assert_eq!(grads.len(), n * self.dim);
        // Update block [n, dim]: the Nesterov look-ahead gradient.
        let mut block = vec![0.0f32; n * self.dim];
        for (k, &r) in indices.iter().enumerate() {
            let base = (r as usize) * self.dim;
            for c in 0..self.dim {
                let mi = base + c;
                self.m[mi] = MOMENTUM * self.m[mi] + (1.0 - MOMENTUM) * grads[k * self.dim + c];
                block[k * self.dim + c] =
                    MOMENTUM * self.m[mi] + (1.0 - MOMENTUM) * grads[k * self.dim + c];
            }
        }
        if self.step % SINKHORN_EVERY == 0 {
            sinkhorn_l1(&mut block, n, self.dim, SINKHORN_ITERS);
        }
        for (k, &r) in indices.iter().enumerate() {
            let base = (r as usize) * self.dim;
            for c in 0..self.dim {
                self.tables[base + c] -= lr * block[k * self.dim + c];
            }
        }
    }

    /// Serialize the tables + momentum state (checkpointing; ~row_bytes *
    /// rows). Layout: [magic u32][step u64][tables][m].
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(12 + self.tables.len() * 4 * 2);
        buf.extend_from_slice(&LAYOUT_MAGIC.to_le_bytes());
        buf.extend_from_slice(&self.step.to_le_bytes());
        for chunk in [&self.tables, &self.m] {
            for &x in chunk {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        buf
    }

    /// Stream the same layout straight to a writer in 64KB blocks: a
    /// multi-GB table set must not materialize a second copy in RAM.
    pub fn write_to<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        w.write_all(&LAYOUT_MAGIC.to_le_bytes())?;
        w.write_all(&self.step.to_le_bytes())?;
        let mut buf = Vec::with_capacity(64 * 1024);
        for chunk in [&self.tables, &self.m] {
            for block in chunk.chunks(16 * 1024) {
                buf.clear();
                for &x in block {
                    buf.extend_from_slice(&x.to_le_bytes());
                }
                w.write_all(&buf)?;
            }
        }
        Ok(())
    }

    /// None when the layout magic mismatches (a v1 Adam-state checkpoint):
    /// the caller must report the skip loudly rather than silently reinit.
    pub fn from_bytes(bytes: &[u8], slots: [usize; 3], dim: usize) -> Option<Self> {
        const MAGIC: [u8; 4] = LAYOUT_MAGIC.to_le_bytes();
        if bytes.len() < 12 || bytes[..4] != MAGIC {
            return None;
        }
        let total = slots.iter().sum::<usize>() * dim;
        if bytes.len() < 12 + total * 4 * 2 {
            return None;
        }
        let step = u64::from_le_bytes(bytes[4..12].try_into().ok()?);
        let mut it = bytes[12..].chunks_exact(4);
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
            step,
        })
    }
}

/// SinkGD-style balancing of an update block `[rows, cols]`: alternate L1
/// row and column normalization (signs preserved - the divisor is the sum
/// of absolute values), then rescale to the block's original total L1 so
/// the learning rate keeps its meaning. All rows/cols end up with the same
/// order of magnitude, so a few huge rows can no longer dominate the
/// update of the batch's other rows.
fn sinkhorn_l1(block: &mut [f32], rows: usize, cols: usize, iters: usize) {
    if rows == 0 || cols == 0 {
        return;
    }
    let orig: f32 = block.iter().map(|x| x.abs()).sum();
    if !orig.is_finite() || orig <= 1e-12 {
        return;
    }
    for _ in 0..iters {
        for r in 0..rows {
            let row = &mut block[r * cols..(r + 1) * cols];
            let s: f32 = row.iter().map(|x| x.abs()).sum();
            if s > 1e-12 {
                for x in row.iter_mut() {
                    *x /= s;
                }
            }
        }
        for c in 0..cols {
            let s: f32 = (0..rows).map(|r| block[r * cols + c].abs()).sum();
            if s > 1e-12 {
                for r in 0..rows {
                    block[r * cols + c] /= s;
                }
            }
        }
    }
    let now: f32 = block.iter().map(|x| x.abs()).sum();
    if now > 1e-12 {
        let f = orig / now;
        for x in block.iter_mut() {
            *x *= f;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Raw FNV hashes reach this table as i32, so they are sometimes
    /// NEGATIVE (the in-VRAM path hands the model its own raw hashes and
    /// masks them; `unique_rows` must be safe for either contract). With
    /// `%` a negative index lands before the table's base and gathers the
    /// wrong row - or panics. `rem_euclid` is the fix, and this is its test.
    #[test]
    fn negative_hashes_stay_in_range() {
        let h = HostNgram::new([64, 64, 64], 8, 3);
        // i32::MIN-ish values, as the raw FNV-32 truncation produces.
        let hashes: Vec<i64> = vec![-1, -2, -3, -65, -1000, i32::MIN as i64];
        let (uniq, pos) = h.unique_rows(&hashes);
        assert_eq!(pos.len(), hashes.len());
        assert!(uniq.iter().all(|&r| (0..h.total_rows() as i64).contains(&r)), "{uniq:?}");
        // Same rows as the non-negative remainder.
        let nonneg: Vec<i64> = hashes.iter().map(|&x| x.rem_euclid(1 << 31)).collect();
        let (uniq2, _) = h.unique_rows(&nonneg);
        assert_eq!(uniq, uniq2, "negative hashes must reduce like their unsigned twins");
        // And the gather is in bounds, which is the actual crash.
        let mut rows = Vec::new();
        h.gather(&uniq, &mut rows);
        assert_eq!(rows.len(), uniq.len() * 8);
    }

    #[test]
    fn gather_momentum_roundtrip() {
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
        // The momentum step must move the gathered rows.
        let before: Vec<f32> = rows.clone();
        let grads: Vec<f32> = rows.iter().map(|x| x * 0.1).collect();
        h.momentum_update(&uniq, &grads, 1e-3);
        let mut after = Vec::new();
        h.gather(&uniq, &mut after);
        assert!(before != after, "momentum update must change the rows");
        // Roundtrip through bytes.
        let bytes = h.to_bytes();
        let h2 = HostNgram::from_bytes(&bytes, [64, 64, 64], 8).expect("ckpt load");
        let mut after2 = Vec::new();
        h2.gather(&uniq, &mut after2);
        assert_eq!(after, after2, "ckpt roundtrip must preserve rows");
    }

    /// A v1 (Adam m+v) checkpoint has no layout magic: from_bytes must
    /// reject it loudly (None), never misread the state.
    #[test]
    fn v1_layout_is_rejected() {
        let h = HostNgram::new([64, 64, 64], 8, 7);
        // Emulate v1: step u64 then three chunks.
        let mut v1 = Vec::new();
        v1.extend_from_slice(&1u64.to_le_bytes());
        let ones = vec![1.0f32; h.tables.len()];
        for chunk in [&h.tables, &ones, &ones] {
            for &x in chunk {
                v1.extend_from_slice(&x.to_le_bytes());
            }
        }
        assert!(HostNgram::from_bytes(&v1, [64, 64, 64], 8).is_none());
        assert!(HostNgram::from_bytes(&h.to_bytes(), [64, 64, 64], 8).is_some());
    }

    /// Sinkhorn balancing must even out wildly skewed rows while keeping
    /// the block's overall magnitude: after balancing, row and column L1
    /// sums are within a small factor of each other.
    #[test]
    fn sinkhorn_balances_rows_and_cols() {
        let (rows, cols) = (6usize, 8usize);
        let mut block: Vec<f32> = (0..rows * cols)
            .map(|i| match i / cols {
                0 => 1000.0 * ((i % cols) as f32 + 1.0),
                1 => 0.001 * ((i % cols) as f32 + 1.0),
                _ => ((i % 7) as f32 - 3.0) * 0.5,
            })
            .collect();
        let orig: f32 = block.iter().map(|x| x.abs()).sum();
        sinkhorn_l1(&mut block, rows, cols, 8);
        let row_l1 = |r: usize| block[r * cols..(r + 1) * cols].iter().map(|x| x.abs()).sum::<f32>();
        let col_l1 = |c: usize| (0..rows).map(|r| block[r * cols + c].abs()).sum::<f32>();
        let row_sums: Vec<f32> = (0..rows).map(row_l1).collect();
        let col_sums: Vec<f32> = (0..cols).map(col_l1).collect();
        let spread = |v: &[f32]| v.iter().cloned().fold(f32::MIN, f32::max) / v.iter().cloned().fold(f32::MAX, f32::max).max(1e-12);
        assert!(
            spread(&row_sums) < 3.0,
            "row L1 sums must balance: {row_sums:?}"
        );
        assert!(
            spread(&col_sums) < 10.0,
            "col L1 sums must balance: {col_sums:?}"
        );
        let now: f32 = block.iter().map(|x| x.abs()).sum();
        assert!(
            (now - orig).abs() / orig < 0.05,
            "total magnitude must be preserved: {orig} -> {now}"
        );
        assert!(block.iter().all(|x| x.is_finite()));
    }
}

/// Gather the batch's rows from RAM into an autodiff leaf and expand them
/// to the `[b, t, 3*dim]` embedding the model consumes. With `track=false`
/// (eval) the leaf is skipped and only the embedding is returned. Also
/// returns the unique absolute row indices (the CPU momentum update
/// consumes exactly these; recomputing `unique_rows` per step would double
/// the work).
pub fn rows_for_batch<B: Backend>(
    host: &HostNgram,
    hashes: &[i64],
    b: usize,
    t: usize,
    device: &burn::tensor::Device,
    track: bool,
) -> (Option<Param<Tensor<2>>>, Tensor<3>, Vec<i64>)
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
    (p, embed, uniq)
}