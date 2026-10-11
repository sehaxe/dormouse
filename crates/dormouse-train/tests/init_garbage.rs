// Is the init-time table corruption a WRITE of garbage or a READ of
// uninitialised memory?
//
// Two identical processes at `--lr 0` leave 5767 rows of the engram table
// different (worst 2.1e14 absolute, `tests/init_shift` proves it is not a row
// move). A N(0,1) draw of 3.1M values tops out near 5.5, so every word with
// |v| > 100 in a checkpoint is garbage by arithmetic - and the question is
// only whether BOTH files carry garbage at DIFFERENT rows (a read of memory
// nobody wrote, which the pool's recycled pages fill with stale values) or one
// file is clean and the other is not (a write).
//
// No burn API needed: the checkpoint is a burnpack of f32 words.

use std::path::Path;

fn words(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

const W: usize = 32;

fn garbage_rows(a: &[f32]) -> (usize, Vec<(usize, f32)>) {
    let mut rows = Vec::new();
    let mut all = Vec::new();
    let n = a.len() / W;
    for r in 0..n {
        let row = &a[r * W..r * W + W];
        let m = row.iter().fold(0.0f32, |a, x| a.max(x.abs()));
        if !m.is_finite() || m > 100.0 {
            rows.push(r);
            all.push((r, m));
        }
    }
    (rows.len(), all)
}

#[test]
fn the_init_corruption_is_a_write_or_a_read() {
    let ia = words(Path::new("/tmp/opencode/rec/ia.bin"));
    let ib = words(Path::new("/tmp/opencode/rec/ib.bin"));

    let (na, la) = garbage_rows(&ia);
    let (nb, lb) = garbage_rows(&ib);
    println!(
        "ia ({} words): {na} rows carry |v| > 100, largest {:.3e}\nib ({} words): {nb} rows carry |v| > 100, largest {:.3e}",
        ia.len(),
        la.iter().fold(0.0f32, |m, (_, v)| m.max(*v)),
        ib.len(),
        lb.iter().fold(0.0f32, |m, (_, v)| m.max(*v)),
    );

    let set_a: std::collections::HashSet<usize> = la.iter().map(|(r, _)| *r).collect();
    let set_b: std::collections::HashSet<usize> = lb.iter().map(|(r, _)| *r).collect();
    let both = set_a.intersection(&set_b).count();
    println!("rows garbage in BOTH files: {both}; only ia: {}; only ib: {}", na - both, nb - both);

    // Where in the file they sit: the table is one contiguous record, so a
    // cluster is the table and a scatter is the model's other tensors.
    let first = la.first().or_else(|| lb.first()).map(|(r, _)| *r).unwrap_or(usize::MAX);
    let last = la.last().or_else(|| lb.last()).map(|(r, _)| *r).unwrap_or(0);
    println!("first garbage row {first} of {}, last {last}", ia.len() / W);

    if na > 0 && nb > 0 && both * 2 < na.min(nb) {
        panic!(
            "the init corruption is a READ: both files carry garbage rows and they \
             mostly do not overlap ({both} shared, {na} in ia, {nb} in ib) - memory \
             nobody wrote is being read, and the pool decides what is in it"
        );
    }
    if na == 0 || nb == 0 {
        panic!(
            "the init corruption is a WRITE: ia has {na} garbage rows, ib has {nb} - \
             one process wrote garbage the other did not"
        );
    }
    println!("both files are garbage in most of the same rows - a shared bad write");
}
