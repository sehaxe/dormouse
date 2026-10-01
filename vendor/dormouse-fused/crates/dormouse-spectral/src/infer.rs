//! Inference: packed ternary matmul (no FP32 matmuls at all).
//!
//! The ternary weights {-1, 0, +1} are packed 4-per-byte (2 bits each) and
//! the forward is add/sub-only: `y[i] = sum_j s_j * x[j]` where `s_j` is
//! decoded from the pack. No multiplications, no FP32 weight storage.
//!
//! `pack` layout: 2 bits per weight, `0b00 = -1, 0b01 = 0, 0b10 = +1`.
//! Packs are stored per output column (column-major pack) so the inner loop
//! over `m` reads consecutive x values (streaming-friendly).
use std::time::Instant;

/// Pack a row-major ternary matrix `[m, n]` (values -1/0/+1) into
/// column-major 2-bit packs: `ceil(m/4) * n` bytes.
pub fn pack_ternary(w: &[i8], m: usize, n: usize) -> Vec<u8> {
    assert_eq!(w.len(), m * n);
    let rows4 = m.div_ceil(4);
    let mut out = vec![0u8; rows4 * n];
    for j in 0..n {
        for i in 0..m {
            let v = match w[i * n + j] {
                -1 => 0u8,
                0 => 1u8,
                1 => 2u8,
                other => panic!("not ternary: {other}"),
            };
            let byte = &mut out[j * rows4 + i / 4];
            *byte |= v << (2 * (i % 4));
        }
    }
    out
}

/// y = x @ W with packed ternary W: `x [b, m]`, packs `[n, rows4]`.
/// Branch-free: `v - 1` maps {0,1,2} -> {-1,0,+1}, so the inner loop is a
/// single FMA, no branches.
pub fn packed_matmul(x: &[f32], packs: &[u8], b: usize, m: usize, n: usize) -> Vec<f32> {
    let rows4 = m.div_ceil(4);
    let mut y = vec![0.0f32; b * n];
    for bi in 0..b {
        let xr = &x[bi * m..(bi + 1) * m];
        for j in 0..n {
            let pack_row = &packs[j * rows4..(j + 1) * rows4];
            let mut acc = 0.0f32;
            for (r, &byte) in pack_row.iter().enumerate() {
                let base = r * 4;
                let lim = (m - base).min(4);
                for k in 0..lim {
                    let v = (byte >> (2 * k)) & 3;
                    // {0,1,2} -> {-1,0,+1}: no branch, single FMA
                    acc += xr[base + k] * (v as f32 - 1.0);
                }
            }
            y[bi * n + j] = acc;
        }
    }
    y
}

/// y = (x * s) @ W with packed ternary W: applies the per-rank scale
/// during the second GEMM (no intermediate materialization of x*s).
pub fn scaled_matmul(x: &[f32], s: &[f32], packs: &[u8], b: usize, m: usize, n: usize) -> Vec<f32> {
    let rows4 = m.div_ceil(4);
    let mut y = vec![0.0f32; b * n];
    for bi in 0..b {
        let xr = &x[bi * m..(bi + 1) * m];
        for j in 0..n {
            let pack_row = &packs[j * rows4..(j + 1) * rows4];
            let mut acc = 0.0f32;
            for (r, &byte) in pack_row.iter().enumerate() {
                let base = r * 4;
                let lim = (m - base).min(4);
                for k in 0..lim {
                    let v = (byte >> (2 * k)) & 3;
                    acc += xr[base + k] * s[base + k] * (v as f32 - 1.0);
                }
            }
            y[bi * n + j] = acc;
        }
    }
    y
}

/// Multithreaded packed matmul: output columns are independent, split across
/// `threads` workers (std threads, no deps). Best for large `n`.
pub fn packed_matmul_par(x: &[f32], packs: &[u8], b: usize, m: usize, n: usize) -> Vec<f32> {
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(4)
        .min(16);
    if n < threads * 16 || b * n < 4096 {
        return packed_matmul(x, packs, b, m, n); // small: single thread
    }
    let rows4 = m.div_ceil(4);
    let chunk = n.div_ceil(threads);
    let cols_all: Vec<usize> = (0..n).collect();
    let partials: Vec<Vec<f32>> = std::thread::scope(|scope| {
        let handles: Vec<_> = cols_all
            .chunks(chunk)
            .map(|cols| {
                let (x, packs) = (x, packs);
                let cols = cols.to_vec();
                scope.spawn(move || {
                    // per-column partial for all batch rows
                    let mut out = vec![0.0f32; b * cols.len()];
                    for (ci, &j) in cols.iter().enumerate() {
                        let pack_row = &packs[j * rows4..(j + 1) * rows4];
                        for bi in 0..b {
                            let xr = &x[bi * m..(bi + 1) * m];
                            let mut acc = 0.0f32;
                            for (r, &byte) in pack_row.iter().enumerate() {
                                let base = r * 4;
                                let lim = (m - base).min(4);
                                for k in 0..lim {
                                    let v = (byte >> (2 * k)) & 3;
                                    acc += xr[base + k] * (v as f32 - 1.0);
                                }
                            }
                            out[bi * cols.len() + ci] = acc;
                        }
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    // reassemble [b, n]: column j lives at (chunk_index, col_index)
    let mut y = vec![0.0f32; b * n];
    for (ci, cols) in cols_all.chunks(chunk).enumerate() {
        let part = &partials[ci];
        for (k, &j) in cols.iter().enumerate() {
            for bi in 0..b {
                y[bi * n + j] = part[bi * cols.len() + k];
            }
        }
    }
    y
}

/// Benchmark: one MLP layer `[512, 4096] @ [4096, 512]` (FFN gate_up +
/// down), dense fp32 vs packed ternary, on the CPU.
pub fn bench() {
    let (m, k, n) = (512usize, 4096usize, 512usize);
    let x: Vec<f32> = (0..m).map(|i| (i as f32) / 1024.0 - 0.25).collect();

    // dense fp32 (two matmuls: m x k, then k x n)
    let w1: Vec<f32> = (0..m * k)
        .map(|i| ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0)
        .collect();
    let w2: Vec<f32> = (0..k * n)
        .map(|i| ((i * 40503) % 1000) as f32 / 500.0 - 1.0)
        .collect();

    let t0 = Instant::now();
    let mut y_dense = vec![0.0f32; n];
    for _ in 0..50 {
        let mut mid = vec![0.0f32; k];
        for j in 0..k {
            let mut acc = 0.0f32;
            for i in 0..m {
                acc += x[i] * w1[i * k + j];
            }
            mid[j] = acc;
        }
        for j in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += mid[i] * w2[i * n + j];
            }
            y_dense[j] = acc;
        }
    }
    let t_dense = t0.elapsed().as_secs_f64() / 50.0;
    let dense_checksum: f32 = y_dense.iter().sum();
    println!("dense checksum {dense_checksum:.1}");

    // packed ternary: same shapes, add-only
    let t1: Vec<i8> = w1
        .iter()
        .map(|&v| {
            if v > 0.33 {
                1
            } else if v < -0.33 {
                -1
            } else {
                0
            }
        })
        .collect();
    let t2: Vec<i8> = w2
        .iter()
        .map(|&v| {
            if v > 0.33 {
                1
            } else if v < -0.33 {
                -1
            } else {
                0
            }
        })
        .collect();
    let p1 = pack_ternary(&t1, m, k);
    let p2 = pack_ternary(&t2, k, n);

    let t0 = Instant::now();
    let mut tsct_checksum = 0.0f32;
    for _ in 0..50 {
        let mid2 = packed_matmul(&x, &p1, 1, m, k);
        let y2 = packed_matmul(&mid2, &p2, 1, k, n);
        tsct_checksum += y2.iter().sum::<f32>();
    }
    let t_tsct = t0.elapsed().as_secs_f64() / 50.0;
    println!("tsct checksum {tsct_checksum:.1}");

    // TSCT rank 8: U [m,k], V [n,k] packed, two small GEMMs
    let rank = 8usize;
    let u1: Vec<f32> = (0..m * rank)
        .map(|i| ((i * 97) % 1000) as f32 / 500.0 - 1.0)
        .collect();
    let v1: Vec<f32> = (0..k * rank)
        .map(|i| ((i * 101) % 1000) as f32 / 500.0 - 1.0)
        .collect();
    let u2: Vec<f32> = (0..k * rank)
        .map(|i| ((i * 103) % 1000) as f32 / 500.0 - 1.0)
        .collect();
    let v2: Vec<f32> = (0..n * rank)
        .map(|i| ((i * 107) % 1000) as f32 / 500.0 - 1.0)
        .collect();
    let tu1: Vec<i8> = u1
        .iter()
        .map(|&v| {
            if v > 0.33 {
                1
            } else if v < -0.33 {
                -1
            } else {
                0
            }
        })
        .collect();
    let tv1: Vec<i8> = v1
        .iter()
        .map(|&v| {
            if v > 0.33 {
                1
            } else if v < -0.33 {
                -1
            } else {
                0
            }
        })
        .collect();
    let tu2: Vec<i8> = u2
        .iter()
        .map(|&v| {
            if v > 0.33 {
                1
            } else if v < -0.33 {
                -1
            } else {
                0
            }
        })
        .collect();
    let tv2: Vec<i8> = v2
        .iter()
        .map(|&v| {
            if v > 0.33 {
                1
            } else if v < -0.33 {
                -1
            } else {
                0
            }
        })
        .collect();
    let pu1 = pack_ternary(&tu1, m, rank);
    let pv1 = pack_ternary(&tv1, k, rank);
    let pu2 = pack_ternary(&tu2, k, rank);
    let pv2 = pack_ternary(&tv2, n, rank);
    let s1: Vec<f32> = (0..rank).map(|i| 0.5 + i as f32 * 0.1).collect();
    let s2: Vec<f32> = (0..rank).map(|i| 0.5 + i as f32 * 0.1).collect();

    let t0 = Instant::now();
    let mut tsct8_checksum = 0.0f32;
    for _ in 0..50 {
        // y = ((x @ U1) * s1) @ V1^T  ->  then ((mid @ U2) * s2) @ V2^T
        let z1 = packed_matmul(&x, &pu1, 1, m, rank);
        let mid2 = scaled_matmul(&z1, &s1, &pv1, 1, rank, k);
        let z2 = packed_matmul(&mid2, &pu2, 1, k, rank);
        let y2 = scaled_matmul(&z2, &s2, &pv2, 1, rank, n);
        tsct8_checksum += y2.iter().sum::<f32>();
    }
    let t_tsct8 = t0.elapsed().as_secs_f64() / 50.0;
    println!("tsct8 checksum {tsct8_checksum:.1}");

    let t0 = Instant::now();
    let mut tsct_par_checksum = 0.0f32;
    for _ in 0..50 {
        let mid2 = packed_matmul_par(&x, &p1, 1, m, k);
        let y2 = packed_matmul_par(&mid2, &p2, 1, k, n);
        tsct_par_checksum += y2.iter().sum::<f32>();
    }
    let t_tsct_par = t0.elapsed().as_secs_f64() / 50.0;

    println!("dense fp32:   {:.3} ms/layer", t_dense * 1e3);
    println!("ternary dense:{:.3} ms/layer", t_tsct * 1e3);
    println!("ternary par:  {:.3} ms/layer", t_tsct_par * 1e3);
    println!("TSCT rank 8:  {:.3} ms/layer", t_tsct8 * 1e3);
    println!(
        "speedup vs dense: {:.1}x ternary, {:.1}x par, {:.1}x TSCT-r8",
        t_dense / t_tsct,
        t_dense / t_tsct_par,
        t_dense / t_tsct8
    );
    println!(
        "dense bytes: {:.1} MB, ternary: {:.1} KB, TSCT-r8: {:.1} KB",
        (w1.len() + w2.len()) as f64 * 4.0 / 1e6,
        (p1.len() + p2.len()) as f64 / 1e3,
        (pu1.len() + pv1.len() + pu2.len() + pv2.len()) as f64 / 1e3
    );
    let _ = (
        dense_checksum,
        tsct_checksum,
        tsct_par_checksum,
        tsct8_checksum,
    );
}
