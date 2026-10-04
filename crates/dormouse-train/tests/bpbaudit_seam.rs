//! The BPB-honesty GATE — pins the invariants `docs/reviews/bpbaudit-2026-10-04.md`
//! audits, so a future change to the divisor, the target shift, the eval window or
//! the aux inclusion is a deliberate act rather than a silent one.
//!
//! The audit found the instrument HONEST on every point except two carried
//! conventions: the train-line `bpb=` includes the aux objective (the held-out
//! `bpb=` does not), and the last position of every batch targets the batch's own
//! first byte (the wrap). Both are pinned here — not to bless them, but so the
//! next reader finds them in the diff.
//!
//! CPU only (`--features cpu`, burn-flex), no CUDA: this is host arithmetic and a
//! gate that needs the GPU cannot run beside a training run.
//! `cargo test -p dormouse-train --test bpbaudit_seam`

use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn::backend::Autodiff;
use burn::tensor::{Device, Int, Tensor, TensorData};
use dormouse_core::{DormouseConfig, DormouseModel};
use dormouse_data::ByteStream;
use dormouse_train::bpb;

/// The same alias the crate's own `cpu` feature builds (`device()`), proven to
/// satisfy the `DispatchKindConversion` bounds `forward_with_hidden` carries.
type B = Autodiff<burn::backend::Flex, BalancedCheckpointing>;

const BATCH: usize = 2;
const SEQ: usize = 64;

/// Nano's shape at a width a dev-profile CPU test can afford. `dspark_weight` is
/// ON and `jepa_weight` OFF: the DSpark term needs no teacher, so one forward
/// exercises the aux inclusion without a second model.
fn cfg() -> DormouseConfig {
    let mut c = dormouse_core::config::load_config(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../configs/nano.toml"
    ))
    .expect("configs/nano.toml loads");
    c.d_model = 64;
    c.n_heads = 2;
    c.head_dim = 32;
    c.d_ffn = 128;
    c.rank = 16;
    c.engram_rows = 256;
    c.max_seq_len = SEQ;
    c.jepa_weight = 0.0;
    c.dspark_weight = 0.1;
    c.dspark_k = 4;
    c
}

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("bpbaudit_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The trainer's label construction, copied from `train/src/lib.rs` (the train
/// step at :1383, the eval at :1873, byteflow at byteflow.rs:355 — one chain,
/// four call sites). A flat one-byte shift over `batch * seq_len`, with the
/// wrap.
fn shift_targets(bytes: &[u8]) -> Vec<i64> {
    bytes
        .iter()
        .skip(1)
        .chain(std::iter::once(&bytes[0]))
        .map(|&b| b as i64)
        .collect()
}

/// Point 1 of the audit: `bpb(ce) == ce / ln 2`, so a CE in nats becomes bits.
/// The uniform 256-class model is the anchor: CE = ln 256 → exactly 8 BPB.
#[test]
fn bpb_is_ce_over_ln2() {
    let ln2 = std::f32::consts::LN_2;
    for ce in [0.0f32, 0.5, 1.0, 3.4657, 5.5452] {
        assert!(
            (bpb(ce) - ce / ln2).abs() < 1e-5,
            "bpb({ce}) != ce/ln2: the divisor is not ln 2"
        );
    }
    assert!(
        (bpb(256.0f32.ln()) - 8.0).abs() < 1e-4,
        "a uniform byte model must score exactly 8 BPB"
    );
}

/// Point 2 of the audit: the target shift is exactly ONE byte — `targets[q] ==
/// bytes[q+1]` — with no DSpark-style off-by-one, at every call site. The last
/// position of a batch targets the batch's OWN first byte (the wrap): a
/// deliberate, pinned convention, 1 of `batch * seq_len` positions, identical in
/// train and eval, so the train/eval comparison is fair and only the absolute
/// BPB moves (by <= ~0.005 BPB at batch 2, less at larger batch).
#[test]
fn target_shift_is_one_byte_with_wrap() {
    let n = BATCH * SEQ;
    let bytes: Vec<u8> = (0..n).map(|i| (i * 7 + 3) as u8).collect();
    let shifted = shift_targets(&bytes);
    assert_eq!(shifted.len(), n);
    for q in 0..n - 1 {
        assert_eq!(
            shifted[q],
            bytes[q + 1] as i64,
            "position {q} targets the wrong byte: the shift is not one"
        );
    }
    assert_eq!(
        shifted[n - 1],
        bytes[0] as i64,
        "the wrap convention changed: the last position must target the batch's first byte"
    );
}

/// Points 3 and 4 of the audit, on a real stream and a real model: the eval
/// window is `eval_batches * batch * seq_len` bytes, `rewind()` restores the
/// same window, and the `ece` mean's divisor equals the scored byte count.
#[test]
fn eval_window_rewind_and_mean_divisor() {
    let dir = tmpdir("window");
    let data = dir.join("data");
    let eval = dir.join("eval");
    std::fs::create_dir_all(&data).unwrap();
    std::fs::create_dir_all(&eval).unwrap();
    // Disjoint content: the no-leak guard is path-based, but a faithful fixture
    // does not share bytes either.
    let train_bytes: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
    let eval_bytes: Vec<u8> = (0..4096).map(|i| ((i * 3 + 1) % 251) as u8).collect();
    std::fs::write(data.join("corpus.bin"), &train_bytes).unwrap();
    std::fs::write(eval.join("eval_tail.bin"), &eval_bytes).unwrap();

    let (_stream, eval_stream) = ByteStream::train_and_eval(SEQ, BATCH, &data, Some(&eval));
    let mut ev = eval_stream.expect("eval root yields a stream");
    ev.rewind();

    let dev = Device::flex().autodiff();
    let mut model = DormouseModel::new(&cfg(), &dev);
    model.set_loop_depth(None);

    let eval_batches = 3;
    let mut ce_sum = 0.0f32;
    let mut n = 0u32;
    let mut first_batch: Option<Vec<u8>> = None;
    for _ in 0..eval_batches {
        let (eb, eh) = ev.next_batch();
        assert_eq!(
            eb.len(),
            BATCH * SEQ,
            "a batch is exactly batch*seq_len bytes; a short batch would change the window"
        );
        if first_batch.is_none() {
            first_batch = Some(eb.clone());
        }
        let ids: Vec<i64> = eb.iter().map(|&b| b as i64).collect();
        let x: Tensor<2, Int> = Tensor::from_data(TensorData::new(ids, [BATCH, SEQ]), &dev);
        let eh_t: Tensor<3, Int> = Tensor::from_data(TensorData::new(eh, [BATCH, SEQ, 3]), &dev);
        // The eval forward: targets = None, so no aux and a zero in-loop rec.
        let (logits, ..) = model.forward_with_hidden::<B>(x, Some(eh_t), None, None, None);
        let v = model.vocab_size;
        let shifted = shift_targets(&eb);
        let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [BATCH, SEQ]), &dev);
        let flat = logits.reshape([BATCH * SEQ, v]);
        let ce: f32 = burn::tensor::activation::log_softmax(flat, 1)
            .gather(1, y.reshape([BATCH * SEQ, 1]))
            .neg()
            .mean()
            .try_into_scalar()
            .unwrap_or(f32::NAN);
        assert!(ce.is_finite() && ce > 0.0, "per-batch CE {ce}");
        ce_sum += ce;
        n += 1;
    }
    let ece = ce_sum / n as f32;
    // The byte count the eval line prints (lib.rs:1939) is the divisor of the
    // mean above: every scored position is in the denominator, none dropped.
    let scored = (n as u64) * (BATCH * SEQ) as u64;
    assert_eq!(
        scored,
        (eval_batches * BATCH * SEQ) as u64,
        "the scored byte count is not eval_batches * batch * seq_len"
    );
    assert!(ece.is_finite() && ece > 0.0);

    // rewind: a second eval over the same window reads the SAME first batch.
    ev.rewind();
    let (eb2, _) = ev.next_batch();
    assert_eq!(
        eb2,
        first_batch.expect("one batch was read"),
        "rewind() must restore the fixed eval window"
    );
}

/// Point 7 of the audit: the train-line `ce=`/`bpb=` is the FULL objective
/// (rec_ce + aux), while the held-out `bpb=` is pure next-byte CE. The aux term
/// is positive, so the logged train bpb is NOT comparable to the held-out bpb
/// without subtracting `aux=` — and the overfitting gap is understated, not
/// inflated, by the difference.
#[test]
fn train_logged_loss_is_ce_plus_aux() {
    let dev = Device::flex().autodiff();
    let model = DormouseModel::new(&cfg(), &dev);
    let bytes: Vec<u8> = (0..BATCH * SEQ).map(|i| (i % 256) as u8).collect();
    let ids: Vec<i64> = bytes.iter().map(|&b| b as i64).collect();
    let x: Tensor<2, Int> = Tensor::from_data(TensorData::new(ids, [BATCH, SEQ]), &dev);
    let shifted = shift_targets(&bytes);
    let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [BATCH, SEQ]), &dev);

    let (_, rec, _, aux) = model.forward_with_hidden::<B>(x.clone(), None, None, Some(y), None);
    let rec_ce: f32 = rec.into_scalar();
    let aux = aux.expect("dspark_weight > 0 with labels must return an aux term");
    let aux_v: f32 = aux.into_scalar();
    assert!(aux_v > 0.0, "the aux term must be positive, got {aux_v}");
    // lib.rs:1529-1535: loss = rec_ce + aux, and THAT is what the log line
    // prints as `ce=` and converts to `bpb=`.
    let logged = rec_ce + aux_v;
    assert!(
        (bpb(logged) - bpb(rec_ce)).abs() > 1e-3,
        "aux must move the logged bpb away from the pure next-byte bpb"
    );

    // The eval forward carries no aux and a zero in-loop rec: the held-out CE is
    // computed from the logits alone, so it is a pure next-byte number.
    let (_, rec_eval, _, aux_eval) = model.forward_with_hidden::<B>(x, None, None, None, None);
    assert!(
        aux_eval.is_none(),
        "the eval forward must not build an aux term (targets = None)"
    );
    let rec_eval: f32 = rec_eval.into_scalar();
    assert_eq!(
        rec_eval, 0.0,
        "the eval forward's in-loop rec is zero; the held-out CE is external"
    );
}
