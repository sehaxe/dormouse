//! The decode seam: bytes -> next-byte logits. The ONE way production code
//! turns a trained model into text.
//!
//! WHY IT IS HERE AND NOT IN `dormouse-core`. A bytes->logits method on the
//! model has to derive the Engram keys, the keys are derived by
//! `dormouse_data::raw_keys`, and the model crate cannot reach the data crate
//! without a dependency edge it should not have (or a second copy of the FNV,
//! which is a second thing that can drift). `DormouseModel::forward_bytes` had
//! `hashed_ids = None` for exactly that reason, `loop_block`'s memory branch
//! took its `None => None` arm, and `generate` + `serve` ran a network whose
//! memory arm contributed LITERAL ZEROS — a confident, wrong, unlogged
//! product, and not the network whose held-out BPB was ever reported. The
//! deleted method's slot in `model.rs` carries the same note.
//!
//! This crate is where the model and the data crate meet (the reason `bpb`
//! lives here too), so the seam is here, and both binaries call it.

use burn::backend::DispatchKindConversion;
use burn::tensor::{DispatchTensor, Int, Tensor, TensorData};

use dormouse_core::{probe, DormouseConfig, DormouseModel};

/// Refuse a model whose in-VRAM memory arm cannot answer for keys.
///
/// The `--engram-ram` path keeps the trained n-gram memory in a `HostNgram`
/// in host RAM, written beside the checkpoint as a `.ngram` sidecar that
/// reaches 34 GB, and squeezes the in-model tables to ONE row per order
/// (`cfg.rs`'s `engram_rows = 1`) because on that path they are never read. An
/// inference export never opens the sidecar — that is the point of the format
/// — so such a model arrives here with a memory arm that returns the SAME row
/// for every context: not a disabled arm, a constant one, and the constant was
/// never trained.
///
/// LOUD, because the alternative is the worst kind of answer: a plausible
/// completion from a model whose memory is not in the file. Call it at startup,
/// where the message reaches the operator before anything is served.
pub fn refuse_unservable_memory(cfg: &DormouseConfig) -> Result<(), String> {
    if !cfg.use_engram {
        return Ok(());
    }
    // Rounded UP to a power of two on the model side (the slot index is
    // masked, not divided): one effective row means every key reads row 0.
    let rows = cfg.engram_rows.next_power_of_two();
    if rows > 1 {
        return Ok(());
    }
    Err(format!(
        "this model cannot be served or generated from in-VRAM: its hashed n-gram memory is \
         {rows} row per order (engram_rows = {}), so every key reads the same row and the memory \
         arm is a constant that was never trained. That is the --engram-ram shape: the trained \
         memory lives in a HostNgram beside the checkpoint, in a .ngram sidecar an inference \
         export does not ship. Serve it from a checkpoint together with its sidecar (not \
         implemented), or export a model trained without --engram-ram.",
        cfg.engram_rows
    ))
}

/// Next-byte logits for the last position of `bytes` — the whole decode path.
///
/// The Engram keys are derived from the SAME bytes by the SAME function the
/// trainer and the held-out eval use, and handed to the model, so the arm runs
/// or this does not return. Checked with the arm counters on the way out:
/// `probe::ENGRAM` moving without `probe::ENGRAM_KEYS` IS the inert arm, so a
/// future refactor that drops the keys again dies here instead of in somebody's
/// generated text.
pub fn next_byte_logits<B: burn::backend::AutodiffBackend>(
    model: &DormouseModel,
    bytes: &[u8],
) -> Vec<f32>
where
    DispatchTensor: DispatchKindConversion<B>
        + DispatchKindConversion<B::InnerBackend>
        + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
{
    let device = model.embedding.weight.device();
    // An empty context reaches here from an HTTP request with an empty body,
    // and the tensor below would be declared [1, 1] holding nothing - a panic
    // whose message names neither the cause nor the escape. This is user input.
    assert!(
        !bytes.is_empty(),
        "decode: refusing to answer from an empty context (0 bytes). `serve` takes the prompt \
         straight off the request body; send at least one byte."
    );
    let t = bytes.len();
    let ids: Vec<i64> = bytes.iter().map(|&b| b as i64).collect();
    let keys = dormouse_data::raw_keys(bytes);
    let x: Tensor<2, Int> = Tensor::from_data(TensorData::new(ids, [1, t]), &device);
    let hashed: Tensor<3, Int> = Tensor::from_data(
        TensorData::new(keys, [1, t, dormouse_data::ORDERS.len()]),
        &device,
    );
    let (arms, keys_read) = (probe::count(probe::ENGRAM), probe::count(probe::ENGRAM_KEYS));
    let logits = model.forward::<B>(x, Some(hashed));
    // The backstop. No keys take `loop_block`'s inert branch, which counts the
    // branch and not the row read: the exact shape of the bug this module
    // exists to end, and the one a reader of the arm counters can see.
    let (ran, read) = (
        probe::count(probe::ENGRAM) - arms,
        probe::count(probe::ENGRAM_KEYS) - keys_read,
    );
    assert!(
        ran == 0 || read > 0,
        "decode: the Engram arm ran WITHOUT reading a row ({ran} branch entries, {read} row \
         reads). The memory arm is inert and these logits are not the model that was trained."
    );
    let [_, seq, v] = logits.dims();
    // LOUD, not `vec![0.0; v]`: all-zero logits is a CONFIDENT answer to
    // "which byte comes next" (argmax says byte 0 forever) and the sampler
    // cannot tell it from a real prediction (ADR-0019).
    logits
        .slice([0..1, seq - 1..seq, 0..v])
        .reshape([v])
        .into_data()
        .try_to_vec()
        .expect("decode: logits readback failed - refusing to sample from nothing")
}
