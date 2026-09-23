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

    #[test]
    fn hash_matches_reference_formula() {
        // reference: hash = sum_i x[i] * prime^i, no mod for small values
        let prime = HASH_PRIMES[0];
        let ids = byte_group_hash_ids(&[3, 5], 2, 0, i64::MAX);
        assert_eq!(ids[0], 3); // (pad, 3): 3*prime^0
        assert_eq!(ids[1], 5 + 3 * prime); // (3, 5): 5 + 3*prime
    }

    #[test]
    fn hash_embeddings_shape() {
        let dev = Device::ndarray();
        let m = HashEmbeddings::new(2, &[2, 4], 1000, 8, &dev);
        let ids: Tensor<2, Int> = Tensor::zeros([1, 6], &dev);
        assert_eq!(m.forward(ids, &dev).dims(), [1, 6, 8]);
    }
}
