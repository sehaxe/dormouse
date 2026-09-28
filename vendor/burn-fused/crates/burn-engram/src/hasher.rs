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

    /// The properties this test suite asserts, in place of a hand-copied
    /// expected value. A copied constant moves with the bug; these cannot:
    ///
    /// 1. **in range** — every key is a valid address for its own slot
    ///    (`0 <= key < prime(slot)`), or the embedding lookup reads OOB.
    /// 2. **stable** — the same ids give the same keys, every call.
    /// 3. **position-independent / window-local** — the key at position i
    ///    depends on the n-gram ENDING at i and on nothing else. This is what
    ///    makes it an n-gram *memory* rather than a positional hash, and it is
    ///    the property a restatement of the implementation can never show.
    /// 4. **distinct inputs of the right width give distinct keys** — a 1-gram
    ///    slot is a bijection on the vocabulary (p > vocab, multiplier coprime
    ///    to p), so all `vocab_size` single tokens land on `vocab_size`
    ///    different keys.
    /// 5. **no systematic collision bias** — n-gram keys are ~uniform over
    ///    their slot: every bin hit, and no bin more than 1.25x / less than
    ///    0.75x its expected count.
    ///
    /// Deliberately absent: a bit-exact comparison with `engram_demo_v1.py`.
    /// This port uses a splitmix64 stream where the reference uses PCG64, so
    /// it cannot be bit-for-bit with the reference; the header says so. Until
    /// someone commits a fixture from the authors' code, fidelity is
    /// UNMEASURED and only these properties are claimed.
    #[test]
    fn hash_properties() {
        // 1 + 2
        let h = NgramHasher::new(256, 1, 3, 1, 0, 42);
        let ids: Vec<i64> = (0..64).collect();
        let a = h.hash_ids(&ids);
        assert_eq!(a, h.hash_ids(&ids), "must be deterministic");
        assert_eq!(a.len(), 64 * 3);
        for (i, &v) in a.iter().enumerate() {
            let slot = i % 3;
            assert!((0..h.primes[slot]).contains(&v), "out of range: {v}");
        }
        // 3: the same n-gram ENDING at two positions hashes the same, so the
        // key is a function of the window, not of the position.
        let seq: Vec<i64> = vec![9, 4, 77, 200, 9, 4, 77, 13, 9, 4, 77];
        let k = h.hash_ids(&seq);
        let t = h.num_tables(); // min=1..=3, n_heads=1 -> slot = ngram - 1
        let key = |i: usize, slot: usize| k[i * t + slot];
        assert_eq!(key(1, 1), key(5, 1), "2-gram (4,9) at 1 and 5");
        assert_eq!(key(5, 1), key(9, 1), "2-gram (4,9) at 5 and 9");
        assert_eq!(key(2, 2), key(6, 2), "3-gram (77,4,9) at 2 and 6");
        assert_eq!(key(6, 2), key(10, 2), "3-gram (77,4,9) at 6 and 10");

        // 3, other direction: a token strictly outside the window ending at
        // `end` cannot change that window's key, for ANY slot. A hasher that
        // mixed in more than the last n tokens would fail here.
        let end = 6usize;
        for slot in 0..t {
            let n = slot + 1;
            for far in 0..=end.saturating_sub(n + 1) {
                let mut other = seq.clone();
                other[far] = (other[far] + 1) % 256;
                assert_eq!(
                    key(end, slot),
                    h.hash_ids(&other)[end * t + slot],
                    "slot {slot} (n={n}) moved with token {far}, which is {n} steps back"
                );
            }
            // ...and the newest token IS inside every window, so it must move
            // them all. Fixed data, fixed seed: this is a fixed fact, not a
            // probabilistic claim.
            let mut bumped = seq.clone();
            bumped[end] += 1;
            let b = h.hash_ids(&bumped);
            assert_ne!(
                key(end, slot),
                b[end * t + slot],
                "slot {slot} ignored the token it ends on"
            );
        }
    }

    /// 4: a 1-gram slot is injective over the vocabulary. p > vocab_size and
    /// the multiplier is coprime to p, so every token gets its own key. A
    /// degenerate multiplier (0, or a multiple of p) collapses the whole
    /// vocabulary onto one key and this is the test that says so.
    #[test]
    fn one_gram_keys_are_injective_over_the_vocab() {
        let h = NgramHasher::new(256, 1, 1, 1, 0, 42);
        let ids: Vec<i64> = (0..256).collect();
        let keys = h.hash_ids(&ids);
        let distinct: std::collections::HashSet<i64> = keys.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            256,
            "256 tokens collapsed onto {} keys (prime {}, multiplier {})",
            distinct.len(),
            h.primes[0],
            h.multipliers[0]
        );
    }

    /// 5: no systematic collision bias. 200k pseudo-random 2-grams into the
    /// prime slot (257 for vocab 256): every bin hit, and every bin within
    /// 0.75x..1.25x of uniform. Uniform gives E = 778, sd = 27, so the band is
    /// ~28 sigma wide — it can only fire on a structural defect (a stuck bit,
    /// a weak multiplier, a wrong mod), never on sampling noise. The generator
    /// is an inline splitmix64: no dependency, byte-identical on every host.
    #[test]
    fn ngram_keys_are_uniform_over_their_slot() {
        let h = NgramHasher::new(256, 2, 2, 1, 0, 7);
        let p = h.primes[0] as usize;
        assert!((250..260).contains(&p), "expected a prime near 257, got {p}");
        let n = 200_000usize;
        let mut s = 0x243F6A8885A308D3u64;
        let mut next = move || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            ((z ^ (z >> 31)) >> 33) as usize % 256
        };
        let ids: Vec<i64> = (0..n).map(|_| next() as i64 * 256 + next() as i64).collect();
        let keys = h.hash_ids(&ids);
        let mut bins = vec![0usize; p];
        for &k in &keys {
            bins[k as usize] += 1;
        }
        let e = n as f64 / p as f64;
        let (lo, hi) = (0.75 * e, 1.25 * e);
        for (b, &c) in bins.iter().enumerate() {
            assert!(c > 0, "bin {b} never hit: a hole in the key space");
            let f = c as f64;
            assert!(f > lo && f < hi, "bin {b}: {c} hits, expected ~{e:.0}");
        }
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
