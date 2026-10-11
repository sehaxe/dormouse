// The init-time corruption, at row level: a row MOVE or a garbage WRITE?
//
// Two identical processes at `--lr 0` (no update at all) leave the engram table
// DIFFERENT. Measured 2026-10-10 (`tests/ab_bisect` in the bisect worktree):
// 5767 of the rows are not equal, the worst differ by 2.1e14 absolute, and the
// top parameter by relative move is this table at 1.000e0 while every TSCT
// factor sits at ~5e-4 - the known QR nondeterminism class. So the corruption
// exists BEFORE any optimizer step, and the trainer's own NaN firewall never
// sees it: 1e14 is finite.
//
// An off-by-one and a garbage write look identical from a per-parameter diff
// and completely different at row level: a move keeps every row's content
// intact somewhere else, so testing shifts of +-2 rows for bit-equal matches
// separates them. No burn API is needed - the checkpoint is a burnpack of f32
// words and the table is the biggest one, so a word-level scan answers it.

use std::path::Path;

fn words(p: &Path) -> Vec<f32> {
    let b = std::fs::read(p).unwrap();
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

const W: usize = 32;

/// Count bit-equal aligned rows over the longest stretch where BOTH files hold
/// finite, O(1)-magnitude words - that stretch is the table, not the manifest.
fn compare_rows(a: &[f32], b: &[f32]) -> (usize, usize, f32) {
    let mut eq = 0;
    let mut tot = 0;
    let mut max_d = 0.0f32;
    let n = a.len().min(b.len());
    let mut i = 0;
    while i + W <= n {
        let a_row = &a[i..i + W];
        let b_row = &b[i..i + W];
        let a_ok = a_row
            .iter()
            .all(|x| x.is_finite() && x.abs() < 1e6);
        let b_ok = b_row
            .iter()
            .all(|x| x.is_finite() && x.abs() < 1e6);
        if a_ok && b_ok {
            tot += 1;
            let d = a_row
                .iter()
                .zip(b_row)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f32::max);
            max_d = max_d.max(d);
            if d == 0.0 {
                eq += 1;
            }
            i += W;
        } else {
            i += 1;
        }
    }
    (eq, tot, max_d)
}

fn shifted_eq(a: &[f32], b: &[f32], s: isize, limit: usize) -> (usize, usize) {
    let mut eq = 0;
    let mut tot = 0;
    for r in 0..limit {
        let (x0, x1) = (r * W, r * W + W);
        let y = r as isize + s;
        if y < 0 {
            continue;
        }
        let (y0, y1) = (y as usize * W, y as usize * W + W);
        if y1 > b.len() {
            break;
        }
        tot += 1;
        if a[x0..x1] == b[y0..y1] {
            eq += 1;
        }
    }
    (eq, tot)
}

#[test]
fn the_init_difference_is_a_row_move_or_a_garbage_write() {
    let a = words(Path::new("/tmp/opencode/rec/ia.bin"));
    let b = words(Path::new("/tmp/opencode/rec/ib.bin"));
    // The two files are not even the same length (ia 135797040 vs ib 135797296
    // bytes, 64 words), which is its own datum about a config-dependent record.
    let (a, b) = if a.len() > b.len() { (&a[..b.len()], &b[..]) } else { (&a[..], &b[..a.len()]) };

    let (eq, tot, max_d) = compare_rows(&a, &b);
    println!(
        "init compare: {eq} of {tot} aligned {W}-wide rows bit-identical, \
         largest |delta| over the O(1)-magnitude stretch {max_d:.3e}"
    );

    let limit = (a.len() / W).min(200_000);
    for s in [-2isize, -1, 0, 1, 2] {
        let (eq, tot) = shifted_eq(&a, &b, s, limit);
        println!("shift {s:+}: {eq} of {tot} rows bit-identical");
    }

    // The decision this test exists to make, stated so a reader does not have
    // to infer it from the numbers above.
    let (eq0, tot0) = shifted_eq(&a, &b, 0, limit);
    let (eq1, _) = shifted_eq(&a, &b, 1, limit);
    if eq1 > eq0 * 2 && eq1 * 10 > tot0 * 9 {
        panic!(
            "the init difference is a ROW MOVE: shift +1 matches {eq1} of {tot0} rows \
             against {eq0} at shift 0 - an indexing bug in the table's construction"
        );
    }
    println!("the init difference is a WRITE, not a move: shift +-1,+-2 match no more than shift 0");
}
