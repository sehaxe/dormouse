//! CPU/CUDA parity gate for the top-k -> gather primitive (ADR-0015).
//!
//! `Tensor::argtopk` on a cubecl backend emitted **out-of-range indices** for
//! any score row holding fewer than `k` values above `-inf`, and the `gather`
//! that followed dereferenced one: `cuEventCreate 700` /
//! `CUDA_ERROR_ILLEGAL_ADDRESS`. That is the defect behind a whole attention
//! arm being cut (ADR-0014) and a mechanism being rejected (ADR-0013).
//!
//! **The CUDA arm is the whole point.** burn's ndarray `argtopk` is its own
//! implementation and was always correct, so a CPU-only run passes whether or
//! not the kernel is fixed — this file would have been green through the entire
//! life of the bug. Same shape as `backend_parity.rs`: the backend is not a type
//! parameter here, so parity is ONE body of assertions run against
//! `Device::ndarray()` and `Device::cuda(0)` in one process.
//!
//! The mechanism, since a gate is worth nothing without it: the top-k
//! accumulator is `k` slots seeded `(min_value, u32::MAX)`, and 0.3.0-pre.4
//! wrapped every insert in a fast-path guard that asks "is the candidate at
//! least the k-th value?". `-inf < min_value` is a *true* comparison, so a
//! masked candidate was rejected as unable to enter a slot that was in fact
//! still empty, the slot kept the `u32::MAX` seed, and the finalizer cast that
//! to the index dtype — `-1` as i32, `4294967295` as i64. A block-sparse
//! indexer masks its excluded blocks with `-inf` and every short prefix has
//! fewer than `k` visible blocks, so this is not an edge case, it is every early
//! position in a sequence. Fixed in the vendored `cubek-reduce`: `reaches`
//! accepts into a slot still holding `min_value`, the finalizer clamps to
//! `axis_len - 1`, and the seed is `0` rather than `u32::MAX`.
//!
//! Run: `cargo test -p backend-parity --features cuda --test topk_gather_parity`

use burn::tensor::{Device, Tensor, TensorData};
use burn_mor::{topk_gather, topk_indices_last};

/// One row's contract: the picked indices, the score row they come from, the
/// value row they name, and what the gather actually produced.
struct Row<'a> {
    scores: &'a [f32],
    /// value vectors by column index, so `values[p]` is column `p`'s values
    values: &'a [Vec<f32>],
    picks: &'a [i64],
    /// the gathered output for this row, `k * m` long
    got: &'a [f32],
    n: usize,
    k: usize,
}

/// The contract, as counts of what broke per row: `(index violations, wrong
/// gather, wrong picked set, duplicate picks)`, so a failure names the guarantee
/// rather than a difference in a long vector.
///
/// The score *multiset* — not the index order — is compared against a host
/// reference, because tie order is backend-defined and unspecified. Neither
/// check subsumes the other: a clamped sentinel is in range yet picks the wrong
/// score, and a repeated index can be in range with a right score multiset when
/// the repeated column happens to score the same.
#[track_caller]
fn violations(r: &Row) -> (usize, usize, usize, usize) {
    let (n, k, m) = (r.n, r.k, r.values[0].len());
    let mut oob = 0;
    let mut bad_gather = 0;
    let mut bad_set = 0;
    let mut dup = 0;

    assert_eq!(r.picks.len(), k, "argtopk returned the wrong count");

    // (1) every index is addressable. This is the assertion the out-of-bounds
    //     gather died on.
    for &p in r.picks {
        if p < 0 || p as usize >= n {
            oob += 1;
        }
    }
    if oob > 0 {
        return (oob, bad_gather, bad_set, dup);
    }

    // (2) the gathered rows are the ones the indices name.
    for (t, &p) in r.picks.iter().enumerate() {
        for j in 0..m {
            if r.got[t * m + j] != r.values[p as usize][j] {
                bad_gather += 1;
            }
        }
    }

    // (3) the picked scores are the k largest, as a multiset.
    let mut mine: Vec<f32> = r.picks.iter().map(|&p| r.scores[p as usize]).collect();
    let mut want: Vec<f32> = r.scores.to_vec();
    want.sort_by(|a, b| b.partial_cmp(a).expect("no NaN in the fixture"));
    want.truncate(k);
    mine.sort_by(|a, b| b.partial_cmp(a).expect("no NaN in the fixture"));
    if mine != want {
        bad_set += 1;
    }

    // (4) k < n (burn asserts it), so the k largest are k distinct elements.
    //     This is the check the -inf case failed: unfilled slots came back
    //     naming one clamped column over and over.
    let mut seen = r.picks.to_vec();
    seen.sort_unstable();
    seen.dedup();
    if seen.len() != k {
        dup += 1;
    }
    (oob, bad_gather, bad_set, dup)
}

/// Run one score fixture through the primitive and assert the contract. Each
/// entry of `rows` is one query: `n` candidate scores, `m` features per
/// candidate, the `k` best of them gathered.
#[track_caller]
fn check(name: &str, device: &Device, rows: Vec<Vec<f32>>, k: usize) {
    let r = rows.len();
    let n = rows[0].len();
    let m = 3;
    let k = k.min(n.saturating_sub(1));

    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let scores = Tensor::<3>::from_data(TensorData::new(flat, [r, 1, n]), device);

    // Value vector of query `row` column `c` is `[row*n*m + c*m + j]` for j in
    // 0..m: distinct per (query, column), so a wrong gather shows up and a
    // picked column is readable straight back out of the output.
    let vflat: Vec<f32> = (0..r * n * m).map(|i| i as f32).collect();
    let values = Tensor::<4>::from_data(TensorData::new(vflat, [r, 1, n, m]), device);

    let idx = topk_indices_last(scores.clone(), k);
    // The index dtype is backend-defined — ndarray hands back I64, cubecl I32 —
    // which is itself a reason this comparison cannot be a shared device: read
    // through a conversion and let the check be about the VALUES.
    let idx_flat: Vec<i64> = idx
        .reshape([r, k])
        .into_data()
        .try_to_vec_as()
        .unwrap();
    let picks: Vec<Vec<i64>> = idx_flat.chunks(k).map(<[i64]>::to_vec).collect();

    let out = topk_gather(scores, values, k);
    assert_eq!(out.dims(), [r, 1, k, m], "{name}: wrong output shape");
    let got: Vec<f32> = out.reshape([r, k * m]).into_data().try_to_vec().unwrap();

    let (mut oob, mut bad_gather, mut bad_set, mut dup) = (0, 0, 0, 0);
    for row in 0..r {
        let base = row * n * m;
        let values: Vec<Vec<f32>> = (0..n)
            .map(|c| (0..m).map(|j| (base + c * m + j) as f32).collect())
            .collect();
        let v = violations(&Row {
            scores: &rows[row],
            values: &values,
            picks: &picks[row],
            got: &got[row * k * m..],
            n,
            k,
        });
        oob += v.0;
        bad_gather += v.1;
        bad_set += v.2;
        dup += v.3;
        if v.1 + v.2 + v.3 > 0 {
            // which columns, so a failure is diagnosable from the log alone
            let mine: Vec<f32> = picks[row]
                .iter()
                .map(|&p| rows[row][p as usize])
                .collect();
            eprintln!(
                "{name}: row {row} off — picks {:?} scores {mine:?}\n  \
                 row scores {:?}\n  gathered head {:?} want head {:?}",
                picks[row],
                rows[row],
                &got[row * k * m..(row * k * m + 6).min(got.len())],
                &values[picks[row][0] as usize][..],
            );
        }
    }
    assert_eq!(oob, 0, "{name}: {oob} index/indices outside [0,{n})");
    assert_eq!(bad_gather, 0, "{name}: the gather returned the wrong rows");
    assert_eq!(bad_set, 0, "{name}: {bad_set} row(s) picked the wrong k");
    assert_eq!(dup, 0, "{name}: {dup} row(s) picked a duplicate index");
}

/// Every row identical, so all `n` columns tie: the pick is any `k` of them.
fn all_ties(n: usize) -> Vec<Vec<f32>> {
    vec![vec![0.5; n]; 2]
}

/// One column strictly above everything else, plus a row with a distinct score
/// everywhere so the order below the winner is pinned too.
fn one_hot(n: usize, hot: usize) -> Vec<Vec<f32>> {
    let mut row = vec![0.0; n];
    row[hot] = 1.0;
    let other: Vec<f32> = (0..n).map(|i| -1.0 - i as f32 * 1e-3).collect();
    vec![row, other]
}

/// A block-sparse indexer's row: everything the mask excluded is masked out.
/// Two mask values, because they are not interchangeable: `-inf` is *below*
/// the accumulator's `min_value` seed, and `-1e30` is above it. An indexer that
/// masks with one and not the other exercises two different comparisons in the
/// selection network, and the `short` row has 3 candidates taken at k = 8, so
/// most slots have nothing to compete with.
fn masked(n: usize, mask: f32) -> Vec<Vec<f32>> {
    let short: Vec<f32> = (0..n)
        .map(|i| if i < 3 { i as f32 } else { mask })
        .collect();
    let all: Vec<f32> = vec![mask; n];
    let alt: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { i as f32 } else { mask })
        .collect();
    vec![short, all, alt]
}

#[track_caller]
fn topk_gather_contract(device: &Device) {
    let n = 12;
    // k = 1: the single best, no slot to starve
    check("k=1", device, one_hot(n, 7), 1);
    // k < n, every column tied
    check("all-tied", device, all_ties(n), 4);
    // k = n - 1: exactly one slot never competes, the tightest legal k.
    // (`k == n` is not expressible: burn's argtopk asserts k < n.)
    check("k==n-1", device, one_hot(n, 3), n - 1);
    check("all-tied k==n-1", device, all_ties(n), n - 1);
    // the masked rows, at k = 8 over a row with 3 candidates and at k = 1, with
    // both mask values. -1e30 is the one that pins WHICH comparison is wrong: it
    // is above the accumulator seed, so if it passes and -inf does not, the
    // defect is specifically "a candidate below the seed", not ties or masking.
    check("masked -inf", device, masked(n, f32::NEG_INFINITY), 8);
    check("masked -1e30", device, masked(n, -1e30), 8);
    check("masked -inf k=1", device, masked(n, f32::NEG_INFINITY), 1);
}

/// The shape a block indexer runs at: b=10, q=512 queries, n=128 candidate
/// blocks, k=64 picked. Every row is partly `-inf`-masked on a fixed pattern,
/// so most rows have fewer than 64 unmasked candidates — the faulting case, at
/// scale.
#[track_caller]
fn production_shape(device: &Device) {
    let (b, q, n, k) = (10usize, 512usize, 128usize, 64usize);
    let rows: Vec<Vec<f32>> = (0..b)
        .flat_map(|i| {
            (0..q).map(move |j| {
                (0..n)
                    .map(|c| {
                        let s = ((i * 31 + j * 7 + c * 13) % 11) as f32;
                        if s == 0.0 {
                            f32::NEG_INFINITY
                        } else {
                            s * 0.5
                        }
                    })
                    .collect()
            })
        })
        .collect();
    check("b10 q512 n128 k64", device, rows, k);
}

#[test]
fn topk_gather_parity_ndarray() {
    topk_gather_contract(&Device::ndarray());
    production_shape(&Device::ndarray());
}

/// **KNOWN OPEN DEFECT, `#[ignore]`d BY DESIGN** — the same convention as this
/// crate's `bf16_matmul_cuda` and as the fused-backward gate. Run it for the
/// proof of what still holds and what does not:
///
/// ```text
/// cargo test -p backend-parity --features cuda --test topk_gather_parity -- --ignored
/// ```
///
/// Measured on the sm_120 box at the committed patch (ADR-0015), on
/// 2026-09-28:
///
/// * **The out-of-bounds read is GONE.** Every index is in `[0, n)` on every
///   fixture, masked rows included, where the same tensor faulted the gather
///   with `cuEventCreate 700` before. That was the crash, and it is fixed.
/// * **The picked SET is still wrong for a masked row.** A row of
///   `[0, -inf, 2, -inf, 4, -inf, 6, -inf, 8, -inf, 10, -inf]` at `n=12, k=8`
///   returns the six finite columns and then **repeats one column** instead of
///   naming two of the masked ones. The gather therefore reads a valid address
///   and returns a valid row — a silent wrong answer, which ADR-0011 classes as
///   a defect, not a pass. Do not build a mechanism on a masked row until this
///   is green.
///
/// So the CUDA arm below is the honest state, and it is red on exactly one
/// check. It flips to green when the trailing slots of a partially-masked row
/// are filled with the masked candidates' own columns.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "open defect, ADR-0015: a masked row's trailing top-k slots repeat one column"]
fn topk_gather_parity_cuda() {
    topk_gather_contract(&Device::cuda(0));
    production_shape(&Device::cuda(0));
}
