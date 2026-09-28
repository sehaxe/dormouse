//! Byte n-gram hash embeddings (BLT 2412.09871, Meta `bytelatent/model/blt.py`
//! `byte_group_hash_function` + `compute_hash_embeddings`).
//!
//! The encoder's byte embeddings are augmented with a sum of hash-table
//! embeddings: each hash function (distinct prime) over each byte-group size
//! maps a rolling window of bytes to a table index, and the tables' embeddings
//! are added to the token embedding.
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::{Device, Int, Tensor};

/// Primes used by the reference for the polynomial rolling hash.
pub const HASH_PRIMES: [i64; 10] = [107, 109, 113, 127, 131, 137, 139, 149, 151, 157];

/// Polynomial rolling hash of byte windows (reference
/// `rolling_polynomial_hash`): `sum_i x[i] * prime^i` over each window of
/// `group_size` consecutive bytes (causally padded with zeros), reduced
/// mod `max_hash`. Runs on the host; returns ids `[tokens]` in `[0, max_hash)`.
///
/// Fidelity note: Meta's reference folds Horner-style (`h = h·p + b`), which
/// weights the *oldest* window byte by the highest power; this variant
/// weights the *newest* byte by `p⁰`. Both are well-distributed hashes over
/// the same windows — collisions differ, but no downstream behavior depends
/// on matching Meta's table indices. Keep this note in mind when porting
/// pretrained hash-embedding tables.
pub fn byte_group_hash_ids(
    bytes: &[i64],
    group_size: usize,
    prime_idx: usize,
    max_hash: i64,
) -> Vec<i64> {
    let prime = HASH_PRIMES[prime_idx % HASH_PRIMES.len()];
    let mut out = Vec::with_capacity(bytes.len());
    for i in 0..bytes.len() {
        let mut hash: i64 = 0;
        let mut pow: i64 = 1;
        for k in 0..group_size {
            let b = if i >= k { bytes[i - k] } else { 0 }; // causal pad
            hash = hash.wrapping_add(b.wrapping_mul(pow));
            pow = pow.wrapping_mul(prime);
        }
        out.push(hash.rem_euclid(max_hash));
    }
    out
}

/// Hash-table embeddings: one `Embedding` per (hash function, group size),
/// summed onto the base byte embeddings (reference `compute_hash_embeddings`).
#[derive(Module, Debug)]
pub struct HashEmbeddings {
    tables: Vec<Embedding>,
    #[module(skip)]
    pub num_functions: usize,
    #[module(skip)]
    pub group_sizes: Vec<usize>,
}

impl HashEmbeddings {
    /// `num_functions` hash functions x `group_sizes` tables, each of
    /// `table_size` rows (reference `encoder_hash_byte_group_vocab`).
    pub fn new(
        num_functions: usize,
        group_sizes: &[usize],
        table_size: usize,
        embed_dim: usize,
        device: &Device,
    ) -> Self {
        let tables = (0..num_functions * group_sizes.len())
            .map(|_| EmbeddingConfig::new(table_size, embed_dim).init(device))
            .collect();
        Self {
            tables,
            num_functions,
            group_sizes: group_sizes.to_vec(),
        }
    }

    /// Sum of precomputed per-table ids into embeddings `[B, T, D]`.
    ///
    /// Fast path for training loops: hash the bytes ONCE in the dataloader
    /// (`byte_group_hash_ids` per table, table order = functions × group
    /// sizes, matching [`Self::new`]) and skip the per-forward device→host
    /// roundtrip entirely. The tables' embedding lookup stays differentiable.
    pub fn forward_precomputed(&self, ids_per_table: &[Tensor<2, Int>]) -> Tensor<3> {
        assert_eq!(
            ids_per_table.len(),
            self.tables.len(),
            "one id tensor per hash table"
        );
        let mut sum: Option<Tensor<3>> = None;
        for (table, ids) in self.tables.iter().zip(ids_per_table) {
            let e = table.forward(ids.clone());
            sum = Some(match sum {
                Some(s) => s + e,
                None => e,
            });
        }
        sum.expect("at least one hash table")
    }

    /// Sum of hash embeddings `[B, T, D]` for the byte ids `[B, T]`.
    /// The hash tables are indexed by `byte_group_hash_ids` computed on the
    /// host (the hash itself is not differentiable). NOTE: this performs a
    /// device→host sync per call; prefer [`Self::forward_precomputed`] with
    /// dataloader-side hashing in training.
    pub fn forward(&self, byte_ids: Tensor<2, Int>, device: &Device) -> Tensor<3> {
        let [b, t] = byte_ids.dims();
        let ids: Vec<i64> = byte_ids.into_data().try_to_vec().unwrap();
        let mut sum: Option<Tensor<3>> = None;
        let mut i = 0;
        for f in 0..self.num_functions {
            for &gs in &self.group_sizes {
                let table_size = self.tables[i].weight.shape().dims::<2>()[0];
                let hashed = byte_group_hash_ids(&ids, gs, f, table_size as i64);
                let ids_t = Tensor::<2, Int>::from_data(
                    burn::tensor::TensorData::new(hashed, [b, t]),
                    device,
                );
                let e = self.tables[i].forward(ids_t);
                sum = Some(match sum {
                    Some(s) => s + e,
                    None => e,
                });
                i += 1;
            }
        }
        sum.expect("at least one hash table")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_deterministic_and_in_range() {
        let ids = byte_group_hash_ids(&[1, 2, 3, 4, 5], 2, 0, 1000);
        assert_eq!(ids, byte_group_hash_ids(&[1, 2, 3, 4, 5], 2, 0, 1000));
        assert!(ids.iter().all(|&v| (0..1000).contains(&v)));
        // different group sizes -> different ids
        let ids3 = byte_group_hash_ids(&[1, 2, 3, 4, 5], 3, 0, 1000);
        assert_ne!(ids, ids3);
    }

    /// The properties this suite asserts instead of a hand-copied expected
    /// value. A copied constant moves with the bug; these cannot:
    ///
    /// 1. **in range** — every key is a valid row index (`0 <= key < max_hash`).
    /// 2. **stable** — the same bytes give the same keys, every call.
    /// 3. **position-independent / window-local** — the key at position i is a
    ///    function of the `group_size` bytes ENDING at i and nothing else.
    ///    That is what makes it a byte-window hash rather than a positional
    ///    one, and a restatement of the implementation cannot show it.
    /// 4. **distinct inputs of the right width give distinct keys** — no two
    ///    random `group_size`-windows share a key, beyond the birthday rate.
    /// 5. **no systematic collision bias** — keys are ~uniform over
    ///    `[0, max_hash)`: every bin hit, none more than 1.25x / less than
    ///    0.75x its expected count.
    ///
    /// Deliberately absent: a comparison with Meta's `patcher.py`. The header
    /// records that this variant's Horner order differs from the reference's
    /// on purpose, so no bit-exact oracle is possible without changing the
    /// hash. Fidelity to `bytelatent` is UNMEASURED.
    #[test]
    fn hash_properties() {
        // 1 + 2
        let bytes = [1i64, 2, 3, 4, 5];
        let ids = byte_group_hash_ids(&bytes, 2, 0, 1000);
        assert_eq!(ids, byte_group_hash_ids(&bytes, 2, 0, 1000));
        assert!(ids.iter().all(|&v| (0..1000).contains(&v)));
        // a different prime (hash function) gives a different keying
        assert_ne!(ids, byte_group_hash_ids(&bytes, 2, 1, 1000));
        // a different window width gives a different keying
        assert_ne!(ids, byte_group_hash_ids(&bytes, 3, 0, 1000));

        // 3: same window, two positions -> same key. No mod involved, so this
        // is exact arithmetic, not a probabilistic claim.
        let seq = [9i64, 4, 77, 200, 9, 4, 77, 13, 9, 4, 77];
        for gs in [2usize, 3] {
            let k = byte_group_hash_ids(&seq, gs, 0, 1_000_003);
            let first = k[gs - 1]; // window ending at gs-1: seq[0..=gs-1] reversed
            let first_at = gs - 1;
            for i in gs..seq.len() {
                let w: Vec<i64> = (0..gs).map(|d| seq[i - d]).collect();
                let w0: Vec<i64> = (0..gs).map(|d| seq[gs - 1 - d]).collect();
                if w == w0 {
                    assert_eq!(k[i], first, "gs={gs} window {w:?} at {i} vs {first_at}");
                }
            }
        }
        // 3, other direction: a byte strictly outside the window ending at
        // `end` cannot change that window's key, for any width or prime.
        let end = 6usize;
        for gs in 1usize..=4 {
            for pidx in 0..HASH_PRIMES.len() {
                let base = byte_group_hash_ids(&seq, gs, pidx, 1_000_003)[end];
                for far in 0..=end.saturating_sub(gs + 1) {
                    let mut other = seq;
                    other[far] += 1;
                    assert_eq!(
                        base,
                        byte_group_hash_ids(&other, gs, pidx, 1_000_003)[end],
                        "gs={gs} prime#{pidx} moved with byte {far} (not in the window)"
                    );
                }
            }
        }
    }

    /// 4: distinct windows give distinct keys. 5000 random 3-windows into a
    /// 100003-row table: a uniform hash gives ~4870 distinct keys (birthday),
    /// so >= 4800 is a floor no correct-but-degenerate hash can pass (a stuck
    /// bit, a zero multiplier or a lost power of p collapses the count by an
    /// order of magnitude) while sitting ~1.5 sd under the expectation.
    #[test]
    fn distinct_windows_get_distinct_keys() {
        let mut s = 0x243F6A8885A308D3u64;
        let mut next = move || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            ((z ^ (z >> 31)) >> 33) as i64 % 256
        };
        let bytes: Vec<i64> = (0..5000).map(|_| next()).collect();
        let keys = byte_group_hash_ids(&bytes, 3, 0, 100_003);
        let distinct: std::collections::HashSet<i64> = keys.iter().copied().collect();
        assert!(
            distinct.len() >= 4800,
            "5000 windows collapsed onto {} keys",
            distinct.len()
        );
    }

    /// 5: no systematic collision bias. 100k random 3-windows into 997 rows:
    /// every bin hit, every bin within 0.75x..1.25x of uniform (E = 100, sd =
    /// 10, so the band is ~5 sigma). A degenerate hash (stuck bit, zeroed
    /// power, one dominant byte) leaves bins empty or 2-3x hot.
    #[test]
    fn keys_are_uniform_over_the_table() {
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            ((z ^ (z >> 31)) >> 33) as i64 % 256
        };
        let bytes: Vec<i64> = (0..100_000).map(|_| next()).collect();
        let keys = byte_group_hash_ids(&bytes, 3, 4, 997);
        let mut bins = vec![0usize; 997];
        for &k in &keys {
            bins[k as usize] += 1;
        }
        let e = 100_000f64 / 997.0;
        for (b, &c) in bins.iter().enumerate() {
            assert!(c > 0, "bin {b} never hit: a hole in the key space");
            let f = c as f64;
            assert!(
                f > 0.75 * e && f < 1.25 * e,
                "bin {b}: {c} hits, expected ~{e:.0}"
            );
        }
    }

    #[test]
    fn hash_embeddings_shape() {
        let dev = Device::ndarray();
        let m = HashEmbeddings::new(2, &[2, 4], 1000, 8, &dev);
        let ids: Tensor<2, Int> = Tensor::zeros([1, 6], &dev);
        assert_eq!(m.forward(ids, &dev).dims(), [1, 6, 8]);
    }
}
