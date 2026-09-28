// The rank-5 ops the batched chunk path needs, measured against a host
// reference on the CPU backend. Run:
//   cargo test -p burn-gdn2 --test b5_seam_probe -- --nocapture
//
// WHY THIS FILE EXISTS: `forward.rs` carried the note "burn-ndarray 0.22-pre.2
// corrupts swap_dims on 5D and reshapes of permuted views, so the k-batched
// matmul form [B,H,k,c,c] is out of reach on ndarray", and the batched arm was
// designed around folding the chunk axis into a 4-D batch axis because of it.
// That claim is NOT reproducible on the pinned 0.22.0-pre.4: every op the
// batched arm needs — swap_dims(3,4), matmul, cumsum, slice on the chunk axis,
// permute, and a 4-D permuted view reshaped to rank 5 — is bit-exact here
// (worst deviation 0e0, asserted below, not printed). So the batched arm uses
// rank 5 and keeps the head and chunk axes separate, which is what lets the
// per-chunk scratch stay contiguous.
//
// The note in forward.rs is left in place: it may still hold for some other op
// on some other backend, and this file only speaks for the six above.
#![cfg(feature = "std")]
#![allow(deprecated)]

use burn::tensor::{Device, Tensor, TensorData};

fn host(t: &Tensor<5>) -> Vec<f32> {
    t.clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Row-major host reference for [b,h,n,c,k] -> [b,h,n,k,c].
fn ref_swap34(shape: [usize; 5], data: &[f32]) -> Vec<f32> {
    let [b, h, n, c, k] = shape;
    let mut out = vec![0.0; data.len()];
    // source layout [b,h,n,c,k]
    let src = |bb: usize, hh: usize, nn: usize, cc: usize, kk: usize| {
        ((((bb * h + hh) * n + nn) * c + cc) * k) + kk
    };
    // destination layout [b,h,n,k,c]
    let dst = |bb: usize, hh: usize, nn: usize, kk: usize, cc: usize| {
        ((((bb * h + hh) * n + nn) * k + kk) * c) + cc
    };
    for bb in 0..b {
        for hh in 0..h {
            for nn in 0..n {
                for cc in 0..c {
                    for kk in 0..k {
                        out[dst(bb, hh, nn, kk, cc)] = data[src(bb, hh, nn, cc, kk)];
                    }
                }
            }
        }
    }
    out
}

/// Host reference for [b,h,n,c,k] @ [b,h,n,k,c].
fn ref_matmul(shape: [usize; 5], a: &[f32], bdata: &[f32]) -> Vec<f32> {
    let [bs, h, n, c, k] = shape;
    let mut out = vec![0.0; bs * h * n * c * c];
    let ai = |bb: usize, hh: usize, nn: usize, cc: usize, kk: usize| {
        ((((bb * h + hh) * n + nn) * c + cc) * k) + kk
    };
    let bi = |bb: usize, hh: usize, nn: usize, kk: usize, cc: usize| {
        ((((bb * h + hh) * n + nn) * k + kk) * c) + cc
    };
    let oi = |bb: usize, hh: usize, nn: usize, i: usize, j: usize| {
        ((((bb * h + hh) * n + nn) * c + i) * c) + j
    };
    for bb in 0..bs {
        for hh in 0..h {
            for nn in 0..n {
                for i in 0..c {
                    for j in 0..c {
                        let mut s = 0.0;
                        for d in 0..k {
                            s += a[ai(bb, hh, nn, i, d)] * bdata[bi(bb, hh, nn, d, j)];
                        }
                        out[oi(bb, hh, nn, i, j)] = s;
                    }
                }
            }
        }
    }
    out
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn five_d_seam_on_ndarray() {
    let dev = Device::ndarray();
    let shape = [2usize, 3, 4, 5, 6];
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.25 - 2.0).collect();
    let t = Tensor::<5>::from_data(TensorData::new(data.clone(), shape.to_vec()), &dev);

    // 1. swap_dims(3,4) on rank 5.
    let got = host(&t.clone().swap_dims(3, 4));
    let want = ref_swap34(shape, &data);
    println!("swap_dims(3,4) rank5: worst {:e}", worst(&got, &want));

    // 2. matmul on rank 5 ([c,k] @ [k,c]).
    let sw = t.clone().swap_dims(3, 4);
    let got = host(&t.clone().matmul(sw.clone()));
    let want = ref_matmul(shape, &data, &ref_swap34(shape, &data));
    println!("matmul rank5: worst {:e}", worst(&got, &want));

    // 3. cumsum on dim 3 of rank 5.
    let got = host(&t.clone().cumsum(3));
    let mut want = data.clone();
    for bb in 0..shape[0] {
        for hh in 0..shape[1] {
            for nn in 0..shape[2] {
                for cc in 1..shape[3] {
                    for kk in 0..shape[4] {
                        let p = ((((bb * shape[1] + hh) * shape[2] + nn) * shape[3]
                            + (cc - 1))
                            * shape[4])
                            + kk;
                        let c = ((((bb * shape[1] + hh) * shape[2] + nn) * shape[3] + cc)
                            * shape[4])
                            + kk;
                        want[c] += want[p];
                    }
                }
            }
        }
    }
    println!("cumsum rank5 dim3: worst {:e}", worst(&got, &want));

    // 4. slice on dim 2 of rank 5 (one chunk of the batched scratch).
    let got = host(&t.clone().slice([0..2, 0..3, 1..2, 0..5, 0..6]));
    let want = {
        let mut v = Vec::new();
        for bb in 0..2 {
            for hh in 0..3 {
                for cc in 0..5 {
                    for kk in 0..6 {
                        v.push(data[((((bb * 3 + hh) * 4 + 1) * 5 + cc) * 6) + kk]);
                    }
                }
            }
        }
        v
    };
    println!("slice rank5 dim2: worst {:e}", worst(&got, &want));

    // 5. 4D permuted view reshaped to rank 5, and 5D permute.
    let t4 = Tensor::<4>::from_data(
        TensorData::new(data.clone(), vec![2, 3, 20, 6]),
        &dev,
    );
    let got5 = t4.clone().swap_dims(1, 2).reshape([2, 4, 5, 3, 6]);
    let want5 = {
        let mut v = vec![0.0; n];
        for bb in 0..2 {
            for nn in 0..4 {
                for cc in 0..5 {
                    for hh in 0..3 {
                        for kk in 0..6 {
                            let src = (((bb * 3 + hh) * 20 + nn * 5 + cc) * 6) + kk;
                            let dst = ((((bb * 4 + nn) * 5 + cc) * 3 + hh) * 6) + kk;
                            v[dst] = data[src];
                        }
                    }
                }
            }
        }
        v
    };
    println!(
        "4D permuted view -> reshape rank5: worst {:e}",
        worst(&host(&got5), &want5)
    );

    let got = host(&t.clone().permute([0, 1, 2, 4, 3]));
    println!("permute rank5 (3<->4): worst {:e}", worst(&got, &ref_swap34(shape, &data)));

    // Every one of the six must be EXACT, not merely close: these are index
    // permutations and small gemms, where a wrong answer is a wrong answer.
    for (what, d) in [
        ("swap_dims(3,4)", worst(&host(&t.clone().swap_dims(3, 4)), &ref_swap34(shape, &data))),
        ("matmul", worst(&host(&t.clone().matmul(t.clone().swap_dims(3, 4))), &ref_matmul(shape, &data, &ref_swap34(shape, &data)))),
        ("permute", worst(&host(&t.clone().permute([0, 1, 2, 4, 3])), &ref_swap34(shape, &data))),
    ] {
        assert_eq!(d, 0.0, "rank-5 {what} is not exact on this backend: worst {d:e}");
    }
}
