//! GATE 2 — the indexer's indices are sane and the window is honest:
//! in range `[0, B)` under the vendored argtopk (ADR-0015), top-picks equal
//! the scores' own maximums when enough blocks are visible, and the TAIL
//! (the final incomplete block's tokens) is ALWAYS in the mask —
//! "the tokens in the final incomplete block are always included" (Eq.
//! 19), for enough causal queries.
//!
//! The duplicate-block edge (ADR-0015's remaining partial-row defect)
//! leads to in-range repeats — and since the TENSOR path consumes picks as
//! a MASK (picks4.equal -> any), a repeated id is a no-op on the mask: the
//! defect cannot corrupt the tensor stage at all. The fused kernel of
//! stage (b) gathers rows, so IT prices the defect; this gate pins the
//! tensor stage.

use burn::tensor::{Tensor, TensorData};

fn dev() -> burn::tensor::Device {
    burn::tensor::Device::ndarray()
}

fn pick_data(t: &Tensor<3, burn::tensor::Int>) -> Vec<i64> {
    t.clone()
        .into_data()
        .convert::<i64>()
        .bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn bool_data(t: &Tensor<4, burn::tensor::Bool>) -> Vec<bool> {
    t.clone().into_data().bytes.iter().map(|&b| b != 0).collect()
}

#[test]
fn indices_in_range_and_specific_top() {
    let dev = dev();
    // t = 16, r = 4 -> B = 4. Scores with a KNOWN ordering: block j worth
    // (3 - j) for visible rows; block-causal -1e30 where unobserved.
    let (b, t, kb) = (2, 16, 2);
    let scores = Tensor::<3>::from_data(
        TensorData::new(
            (0..b * t * 4)
                .map(|j| {
                    let (row, blk) = (j % (t * 4) / 4, j % 4);
                    if blk * 4 + 3 <= row as i32 {
                        3.0f32 - blk as f32
                    } else {
                        -1e30
                    }
                })
                .collect(),
            [b, t, 4],
        ),
        &dev,
    );
    let picks = burn_msa::select_blocks(scores, kb);
    let all = pick_data(&picks);
    for picks_row in all.chunks_exact(kb) {
        for &idx in picks_row {
            assert!(
                (0..4).contains(&(idx as usize)),
                "index {idx} out of [0, 4), the ADR-0014 class"
            );
        }
        assert!(
            picks_row.contains(&0) && picks_row.contains(&1),
            "top-2 of a 3>2>1>0 ordering must be {{0,1}}, got {picks_row:?}"
        );
    }
}

#[test]
fn tail_always_in_mask_per_causal_query() {
    let dev = dev();
    // t = 30, r = 4 -> B = 7 complete, tail = tokens 28, 29.
    let (b, t) = (1usize, 30usize);
    let picks = burn_msa::select_blocks(
        Tensor::<3>::full([b, t, 7], 0.0, &dev),
        7, // identity: every block "picked", the mask must equal the tril
    );
    let mask = burn_msa::sparse_mask(b, t, 4, &picks, &dev);
    let m = bool_data(&mask); // [b, 1, 30, 30]
    // mask_fill fills where TRUE, so the mask is the EXCLUDED set.
    for i in 0..t {
        for j in 0..t {
            let excluded = m[i * t + j];
            assert_eq!(
                excluded, j > i,
                "all-blocks picks must reduce the mask to the plain causal complement at (i={i},j={j})"
            );
        }
    }
    // And with a PARTIAL selection (kb = 3): the tail tokens 28/29 are in
    // EVERY query's mask through j <= i, and one hot block is in.
    let picks_hot: Vec<i64> = (0..t).flat_map(|_q| vec![0, 2, 4]).collect();
    let picks2 = Tensor::<3, burn::tensor::Int>::from_data(
        TensorData::new(picks_hot, [b, t, 3]),
        &dev,
    );
    let mask2 = burn_msa::sparse_mask(b, t, 4, &picks2, &dev);
    let m2 = bool_data(&mask2);
    for i in 0..t {
        for j in 0..t {
            let want = j <= i && (j / 4 == 0 || j / 4 == 2 || j / 4 == 4 || j >= 28);
            assert_eq!(
                m2[i * t + j], !want,
                "block-mask complement ({i}, {j}): expected excluded={}", !want,
            );
        }
    }
}
