//! The decode seam computes the SAME function as the training and held-out
//! eval forwards. Pinned here because it was not true, and nothing noticed:
//! `DormouseModel::forward_bytes` passed `hashed_ids = None`, so the Engram
//! branch in `loop_block` took its `None => None` arm, the memory arm
//! contributed LITERAL ZEROS, and `generate` + `serve` — the only two
//! production callers — emitted text from a network that is not the one whose
//! BPB we report. Silent, confident, and invisible in a log line.
//!
//! The property is checked two ways, because they fail for different reasons:
//!
//! - the LOGITS: `next_byte_logits` must equal the train forward on the same
//!   bytes and the same keys. This is the strong check; it needs no CUDA and
//!   runs on the crate's own CPU backend in a few seconds.
//! - the ARM COUNTERS: the decode call must move `ENGRAM_KEYS` once per
//!   iteration, and a keyless forward must move none. Cheap, and it is the
//!   property a reader of the eval line can see, so the bug cannot come back
//!   unnoticed even if the logits check is loosened.
//!
//! The keys the reference uses come from `dormouse_data::raw_keys` — the same
//! function `ByteStream::hashes_raw` calls, i.e. the derivation that actually
//! ran training. A copy of the FNV here would make this test pin itself.

use burn::module::Module;
use burn::tensor::{Int, Tensor, TensorData};
use dormouse_core::probe;
use dormouse_core::{DormouseConfig, DormouseModel};
use dormouse_train::decode;

/// Mini-nano, the same cut the export round-trip test uses: the seam contract
/// is width-independent and a full nano init is slow in a dev-profile CPU test.
fn cfg() -> DormouseConfig {
    DormouseConfig {
        d_model: 128,
        n_heads: 4,
        head_dim: 32,
        d_ffn: 256,
        rank: 32,
        engram_rows: 64,
        max_seq_len: 256,
        ..dormouse_core::config::load_config(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../configs/nano.toml"
        ))
        .expect("configs/nano.toml loads")
    }
}

/// Fixed bytes, so a failure is reproducible: no RNG dependency.
fn input() -> Vec<u8> {
    b"the quick brown fox jumps over the lazy dog; pack my box with five dozen".to_vec()
}

/// The decode seam and the train forward on the same bytes, same keys, same
/// weights: the same function. Before the fix this differed by whatever the
/// memory arm contributes, which is not small.
#[test]
fn decode_seam_matches_the_train_forward() {
    let cfg = cfg();
    assert!(
        cfg.use_engram,
        "nano runs the Engram arm, and that is the arm under test"
    );
    let model = DormouseModel::new(&cfg, &dormouse_train::device());
    let bytes = input();
    let t = bytes.len();

    // The reference: exactly what the training step and the held-out eval do
    // on the in-VRAM path - ids, the raw keys, no targets.
    let ids: Vec<i64> = bytes.iter().map(|&b| b as i64).collect();
    let x: Tensor<2, Int> =
        Tensor::from_data(TensorData::new(ids, [1, t]), &dormouse_train::device());
    let hashed: Tensor<3, Int> = Tensor::from_data(
        TensorData::new(
            dormouse_data::raw_keys(&bytes),
            [1, t, dormouse_data::ORDERS.len()],
        ),
        &dormouse_train::device(),
    );
    let (logits, ..) =
        model.forward_with_hidden::<dormouse_train::Backend>(x, Some(hashed), None, None, None);
    let v = model.vocab_size;
    let reference: Vec<f32> = logits
        .slice([0..1, t - 1..t, 0..v])
        .reshape([v])
        .into_data()
        .try_to_vec()
        .expect("reference logits readback");

    probe::reset();
    let decoded = decode::next_byte_logits::<dormouse_train::Backend>(&model, &bytes);

    assert_eq!(
        decoded.len(),
        reference.len(),
        "decode returned the wrong shape"
    );
    let d = decoded
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        d < 1e-6,
        "the decode seam is not the forward we measured: max |dlogit| = {d:.3e}. The memory \
         arm is the only thing that can differ, so the decode path is reading no rows (or \
         different ones) than the trained model."
    );
}

/// The counters, which is what makes a regression visible in a log line: the
/// decode call reads one row per iteration, and a keyless forward reads none.
/// Before the fix both halves of this were zero.
#[test]
fn decode_reads_the_memory_rows_the_training_forward_reads() {
    let cfg = cfg();
    let model = DormouseModel::new(&cfg, &dormouse_train::device());
    let bytes = input();
    let t = bytes.len();

    probe::reset();
    let _ = decode::next_byte_logits::<dormouse_train::Backend>(&model, &bytes);
    let decoded_keys = probe::count(probe::ENGRAM_KEYS);
    let decoded_arms = probe::count(probe::ENGRAM);
    assert_eq!(
        decoded_keys, cfg.max_iter as u64,
        "decode read {decoded_keys} rows over {decoded_arms} branch entries; one per iteration is \
         what the trained forward does"
    );
    assert_eq!(
        decoded_arms, cfg.max_iter as u64,
        "the memory branch ran once per iteration"
    );

    // The inert arm, spelled out: no keys, so no row read. This is the shape
    // `forward_bytes` had, and it is why the check above is not optional.
    probe::reset();
    let ids: Vec<i64> = bytes.iter().map(|&b| b as i64).collect();
    let x: Tensor<2, Int> =
        Tensor::from_data(TensorData::new(ids, [1, t]), &dormouse_train::device());
    let _ = model.forward::<dormouse_train::Backend>(x, None);
    assert_eq!(
        probe::count(probe::ENGRAM),
        cfg.max_iter as u64,
        "the branch is still entered without keys"
    );
    assert_eq!(
        probe::count(probe::ENGRAM_KEYS),
        0,
        "a keyless forward reads no row: the memory arm contributes zeros, and that is the bug \
         the decode seam exists to prevent"
    );
}

/// Both memory arms a forward can be given must READ — a forward that entered
/// the branch and read nothing is a network with the arm contributing zeros.
///
/// WHAT THIS IS AND IS NOT. It pins the two MODEL-level branches (`keys` in,
/// `rows` in), because that is what is callable from a test. It does **not**
/// pin `train_loop`'s eval call site, which is inside a 1400-line function and
/// not callable on its own: the claim that the eval passes the keys is carried
/// by the code and by the `engram=<rows>/<arms>` field on the eval line, which
/// prints `0/<n>` the moment the eval runs a memory-disabled forward. A
/// structural test of that call site would be a grep of the source pretending
/// to be a test; this is the honest version of the same check — a counter a
/// reader of the log already watches.
#[test]
fn both_memory_arms_read_when_they_are_given_something() {
    let cfg = cfg();
    let model = DormouseModel::new(&cfg, &dormouse_train::device()).valid();
    let bytes = input();
    let t = bytes.len();
    let ids: Vec<i64> = bytes.iter().map(|&b| b as i64).collect();
    let x: Tensor<2, Int> =
        Tensor::from_data(TensorData::new(ids, [1, t]), &dormouse_train::device());
    let keys: Tensor<3, Int> = Tensor::from_data(
        TensorData::new(
            dormouse_data::raw_keys(&bytes),
            [1, t, dormouse_data::ORDERS.len()],
        ),
        &dormouse_train::device(),
    );

    // The in-VRAM shape: keys, no rows.
    probe::reset();
    let (_logits, ..) = model.forward_with_hidden::<dormouse_train::Backend>(
        x.clone(),
        Some(keys),
        None,
        None,
        None,
    );
    assert_eq!(
        probe::count(probe::ENGRAM_KEYS),
        cfg.max_iter as u64,
        "a forward given the keys must read a memory row per iteration, or its BPB is a \
         different model than the one being trained"
    );

    // The host-RAM shape: rows, no keys (uploading both is a dead H2D copy).
    // `[b, t, orders * engram_dim]` is the gathered-embed layout
    // `offload::rows_for_batch` produces.
    let gathered = cfg.engram_orders.len() * cfg.engram_dim;
    let rows = Tensor::<3>::zeros([1, t, gathered], &dormouse_train::device());
    probe::reset();
    let (_logits, ..) =
        model.forward_with_hidden::<dormouse_train::Backend>(x, None, Some(rows), None, None);
    assert_eq!(
        probe::count(probe::ENGRAM_KEYS),
        cfg.max_iter as u64,
        "a forward given gathered host rows must read them; a zero read means the \
         --engram-ram path is contributing zeros too"
    );
}

/// A `--engram-ram` model is REFUSED, loudly, instead of being sampled from: its
/// memory is in a `.ngram` sidecar the export does not ship, so every key
/// would read the same untrained row. The refusal is startup-shaped (it is a
/// property of the file, not of a request) and it names the escape.
#[test]
fn a_ram_trained_memory_is_refused_loudly() {
    let mut ram = cfg();
    // What `cfg.rs` does on `--engram-ram`: the in-model tables are squeezed to
    // one row per order because the real capacity is the host table.
    ram.engram_rows = 1;
    let err = decode::refuse_unservable_memory(&ram)
        .expect_err("a one-row memory must be refused, not served");
    assert!(
        err.contains(".ngram"),
        "the message must name the missing sidecar: {err}"
    );
    assert!(
        err.contains("--engram-ram"),
        "the message must name the cause: {err}"
    );

    // The two shapes that ARE servable, so the refusal is not a blanket ban.
    let mut no_memory = cfg();
    no_memory.use_engram = false;
    no_memory.engram_rows = 1;
    decode::refuse_unservable_memory(&no_memory)
        .expect("a model with the memory arm off has nothing to serve wrongly");
    decode::refuse_unservable_memory(&cfg()).expect("the in-VRAM shape is servable");
}
