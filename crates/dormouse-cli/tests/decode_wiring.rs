//! The two binaries a user ever sees must produce text through the decode
//! seam, and only through it.
//!
//! WHY A SOURCE PIN. `generate` and `serve` have exactly one observable
//! output: sampled text. A black-box assertion on that output cannot tell a
//! memory-enabled forward from a memory-disabled one — a model with the n-gram
//! arm contributing literal zeros still emits *a* string, it just emits the
//! wrong one, and the pre-2026-09-28 bug shipped that way through both
//! binaries with every test in the repo green. The instrument that can tell
//! them apart is the wiring itself, so the wiring is what is asserted.
//!
//! What regressed: `DormouseModel::forward_bytes` passed `hashed_ids = None`
//! (the Engram keys are derived by `dormouse_data`, which the model crate
//! cannot reach), `loop_block` took its inert branch, and both binaries shipped
//! a network whose memory arm contributed zeros. `dormouse_train::decode` is
//! the one function that derives the keys; these are its only two callers, and
//! the checks below fail if either of them grows a second way in or drops the
//! startup refusal for a model whose memory is not in the file.

/// Every `src/bin/*.rs` that turns bytes into text.
const DECODERS: [&str; 2] = ["generate.rs", "serve.rs"];

fn src(bin: &str) -> String {
    std::fs::read_to_string(format!("{}/src/bin/{bin}", env!("CARGO_MANIFEST_DIR")))
        .unwrap_or_else(|e| panic!("src/bin/{bin} must be readable: {e}"))
}

#[test]
fn the_decoding_binaries_go_through_the_seam_and_nowhere_else() {
    for bin in DECODERS {
        let s = src(bin);
        assert!(
            s.contains("decode::next_byte_logits"),
            "src/bin/{bin} does not call the decode seam. Bytes -> logits now has to derive the \
             Engram keys, and the only function that does is \
             `dormouse_train::decode::next_byte_logits`. Calling a bare forward here is how \
             `forward_bytes` shipped a memory-disabled network to a user."
        );
        // The negative half is the load-bearing one: the seam being present is
        // not the property, the seam being the ONLY way in is.
        for banned in ["forward_bytes", "forward_with_hidden", ".forward::<"] {
            assert!(
                !s.contains(banned),
                "src/bin/{bin} contains `{banned}` - a bytes -> logits call of its own, which is \
                 the seam this test exists to keep singular. A binary that calls a forward \
                 directly decides for itself which memory arm to pass, and nothing checks it."
            );
        }
    }
}

#[test]
fn the_decoding_binaries_refuse_a_model_whose_memory_is_not_in_the_file() {
    // `--engram-ram` squeezes the in-model tables to one row per order
    // (`cfg.rs`) and keeps the trained memory in a `.ngram` sidecar the export
    // format does not ship. Such a model answers every context from the same
    // never-trained row: a constant memory arm, which is a confident answer,
    // not a disabled one. The refusal is a property of the file, so it belongs
    // at startup - before the first sampled byte, and before serve binds.
    for bin in DECODERS {
        let s = src(bin);
        assert!(
            s.contains("refuse_unservable_memory"),
            "src/bin/{bin} does not refuse an unservable memory. Dropping this call turns a \
             --engram-ram model into plausible output from a memory arm that never ran."
        );
    }
}
