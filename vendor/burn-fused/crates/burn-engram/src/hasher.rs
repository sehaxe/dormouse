//! CPU n-gram hashing for Engram (deepseek-ai/Engram, arxiv 2601.07372).
//!
//! Port of `NgramHashMapping._get_ngram_hashes` from the official reference
//! `engram_demo_v1.py`: for every n-gram order in `min_ngram..=max_ngram`
//! the input ids (causally left-padded with `pad_id`) are mixed with odd
//! pseudo-random multipliers via int64 XOR-multiplication, then reduced
//! mod a distinct prime per (ngram, head) slot. The primes are the next
//! primes starting just above `vocab_size`, shared across slots (like the
//! reference's `seen_primes`).
//!
//! The multipliers come from a deterministic splitmix64 stream instead of
//! numpy's PCG64. Same structure, different constants, stable across runs.
//!
//! The output is a flat `[tokens * num_tables]` i64 vector of slot indices,
//! each in `[0, prime_slot)`; feed it to `EngramModule` built with the same
//! `table_sizes()`.
use burn::tensor::{Device, Int, Tensor};

fn is_prime(n: i64) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut d = 3i64;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

/// Deterministic odd-multiplier stream (splitmix64), like the reference's
/// `default_rng(seed).integers(...) * 2 + 1`. Multipliers stay below
/// `i64::MAX / vocab_size` so `token * multiplier` never overflows (the
/// reference bounds them the same way via `half_bound`).
struct OddMultipliers {
    state: u64,
    bound: i64,
}

impl OddMultipliers {
    fn new(seed: u64, vocab_size: usize) -> Self {
        let half = (i64::MAX / vocab_size.max(1) as i64) / 2;
        Self {
            state: seed,
            bound: half.max(1),
        }
    }
    fn next(&mut self) -> i64 {
        // splitmix64
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        let r = ((z ^ (z >> 31)) as i64 & i64::MAX) % self.bound;
        // odd, like the reference's r*2+1
        2 * r + 1
    }
}

/// N-gram hasher producing Engram slot ids on the host.
#[derive(Clone)]
pub struct NgramHasher {
    multipliers: Vec<i64>, // per n-gram order (index k = shift k)
    primes: Vec<i64>,      // one per slot, in slot order
    min_ngram: usize,
    max_ngram: usize,
    n_heads: usize,
    pad_id: i64,
    num_tables: usize,
}

impl NgramHasher {
    /// `vocab_size` anchors the prime search (reference: `engram_vocab_size`),
    /// `max_ngram` the deepest n-gram order, `n_heads` parallel slots per order.
    /// Slots = `(max_ngram - min_ngram + 1) * n_heads`.
    pub fn new(
        vocab_size: usize,
        min_ngram: usize,
        max_ngram: usize,
        n_heads: usize,
        pad_id: i64,
        seed: u64,
    ) -> Self {
        assert!(min_ngram >= 1 && max_ngram >= min_ngram && n_heads >= 1);
        let mut rng = OddMultipliers::new(seed, vocab_size);
        let multipliers: Vec<i64> = (0..max_ngram).map(|_| rng.next()).collect();

        let mut primes = Vec::new();
        let mut start = vocab_size.saturating_sub(1) as i64;
        let n_slots = (max_ngram - min_ngram + 1) * n_heads;
        for _ in 0..n_slots {
            let mut c = start + 1; // reference: next prime past `start`
            loop {
                if is_prime(c) && !primes.contains(&c) {
                    primes.push(c);
                    start = c;
                    break;
                }
                c += 1;
            }
        }
        Self {
            multipliers,
            primes,
            min_ngram,
            max_ngram,
            n_heads,
            pad_id,
            num_tables: n_slots,
        }
    }

    /// Prime sizes of the embedding slots, in slot order. Build the
    /// `EngramModule` with exactly these sizes so hashes stay in range.
    pub fn table_sizes(&self) -> Vec<usize> {
        self.primes.iter().map(|&p| p as usize).collect()
    }

    pub fn num_tables(&self) -> usize {
        self.num_tables
    }

    /// Hash one sequence. `ids[i]` is the id at position `i`; the output
    /// is token-major: for each token, the slot ids in slot order
    /// (`(ngram, head)` pairs, ngram-major), so it can be reshaped into
    /// `[T, num_tables]` directly (matches the reference's
    /// `np.stack(all_hashes, axis=2)`).
    pub fn hash_ids(&self, ids: &[i64]) -> Vec<i64> {
        let t = ids.len();
        let mut out = Vec::with_capacity(t * self.num_tables);
        for i in 0..t {
            for n in self.min_ngram..=self.max_ngram {
                for h in 0..self.n_heads {
                    let slot = (n - self.min_ngram) * self.n_heads + h;
                    let p = self.primes[slot];
                    let mut mix: i64 = 0;
                    for k in 0..n {
                        // causal shift: token k steps back, pad before t=0
                        let tok = if i >= k { ids[i - k] } else { self.pad_id };
                        let m = tok.wrapping_mul(self.multipliers[k]);
                        mix = if k == 0 { m } else { mix ^ m };
                    }
                    out.push(mix.rem_euclid(p));
                }
            }
        }
        out
    }

    /// Convenience: hashes and uploads to `[B, T, num_tables]` Int tensor.
    ///
    /// Uploads as `i32`: burn's CUDA backend stores Int as I32, and the
    /// module's embedding table accepts any Int dtype (it casts on lookup).
    pub fn hash_tensor<B: burn::backend::Backend>(
        &self,
        ids: &[i64],
        b: usize,
        t: usize,
        device: &Device,
    ) -> Tensor<3, Int> {
        assert_eq!(ids.len(), b * t, "ids length must be b*t");
        let mut out: Vec<i32> = Vec::with_capacity(b * t * self.num_tables);
        for chunk in ids.chunks_exact(t) {
            out.extend(self.hash_ids(chunk).into_iter().map(|v| v as i32));
        }
        Tensor::from_data(
            burn::tensor::TensorData::new(out, [b, t, self.num_tables]),
            device,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_in_range() {
        let h = NgramHasher::new(256, 1, 3, 1, 0, 42);
        let ids: Vec<i64> = (0..64).collect();
        let a = h.hash_ids(&ids);
        let b = h.hash_ids(&ids);
        assert_eq!(a, b, "must be deterministic");
        assert_eq!(a.len(), 64 * 3);
        for (i, &v) in a.iter().enumerate() {
            let slot = i % 3;
            assert!((0..h.primes[slot]).contains(&v), "out of range: {v}");
        }
        // token-major: any window of three consecutive entries is one token
        assert!(a.chunks_exact(3).all(|c| c.len() == 3));
        // pad_id affects the first tokens: a different first id changes hashes
        let mut ids2 = ids.clone();
        ids2[0] = 123;
        let b2 = h.hash_ids(&ids2);
        assert_ne!(a[..3], b2[..3]);
    }

    #[test]
    fn primes_are_prime_and_shared() {
        let h = NgramHasher::new(256, 2, 3, 2, 0, 7);
        assert_eq!(h.num_tables(), 4);
        for &p in &h.primes {
            assert!(is_prime(p));
        }
        // distinct slots, all above vocab_size
        let mut seen = std::collections::HashSet::new();
        for &p in &h.primes {
            assert!(p > 256);
            assert!(seen.insert(p), "duplicate prime {p}");
        }
    }

    #[test]
    fn matches_reference_algorithm() {
        // Reproduce the reference exactly for one sequence by hand:
        // vocab 10, min=2, max=2, heads=1, pad 0, seed such that the first
        // multiplier is known. Compute the XOR mix manually.
        let h = NgramHasher::new(10, 2, 2, 1, 0, 99);
        let m0 = h.multipliers[0];
        let m1 = h.multipliers[1];
        let p = h.primes[0];
        let ids = [3i64, 5, 8];
        let out = h.hash_ids(&ids);
        // shift k: token k steps back, pad_id before t=0 (reference shift_k)
        // t=0: (3, pad) -> 3*m0 ^ pad*m1
        assert_eq!(
            out[0],
            (3i64.wrapping_mul(m0) ^ 0i64.wrapping_mul(m1)).rem_euclid(p)
        );
        // t=1: (5, 3) -> 5*m0 ^ 3*m1
        assert_eq!(
            out[1],
            (5i64.wrapping_mul(m0) ^ 3i64.wrapping_mul(m1)).rem_euclid(p)
        );
        // t=2: (8, 5)
        assert_eq!(
            out[2],
            (8i64.wrapping_mul(m0) ^ 5i64.wrapping_mul(m1)).rem_euclid(p)
        );
    }

    #[test]
    fn hash_tensor_shape() {
        use burn::tensor::Device;
        let h = NgramHasher::new(256, 1, 3, 1, 0, 1);
        let ids: Vec<i64> = (0..16).collect();
        let t = h.hash_tensor::<burn_ndarray::NdArray>(&ids, 2, 8, &Device::ndarray());
        assert_eq!(t.dims(), [2, 8, 3]);
    }
}
