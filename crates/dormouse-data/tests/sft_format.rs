//! The SFT byte template and its mask, pinned byte for byte.
//!
//! Why this file exists rather than a comment: dormouse has no tokenizer and no
//! chat template to inherit, so the turn markers ARE this repository's
//! convention, and a convention with no test is a convention that drifts the
//! first time somebody adds a role. It is also the input format of a training
//! run — a wrong marker is a wrong objective, and like every other objective in
//! this project it would be invisible in the loss curve.
//!
//! The gradient half — that a masked user span really gets no gradient — lives
//! in `dormouse-train`'s `tests/sft_smoke.rs`, because it needs a model. This
//! file pins everything that needs no model.
//!
//! CPU, no burn, no GPU: `cargo test -p dormouse-data --test sft_format`.

use std::path::PathBuf;

use dormouse_data::sft::{self, SftBatch, SftError, SftStream, IM_END};

/// Offset of the first occurrence of `needle` in `hay`, or a panic that names
/// it. Used to locate the spans INDEPENDENTLY of `encode`, so the mask
/// assertions below are not the same code path twice.
fn at(hay: &[u8], needle: &[u8], nth: usize) -> usize {
    hay.windows(needle.len())
        .enumerate()
        .filter(|(_, w)| *w == needle)
        .nth(nth)
        .map(|(i, _)| i)
        .unwrap_or_else(|| {
            panic!(
                "{needle:?} #{nth} not in {:?}",
                String::from_utf8_lossy(hay)
            )
        })
}

/// One user/assistant exchange, the smallest thing that exercises every rule: a
/// role header, a user span, an assistant span, two terminators.
fn exchange() -> Vec<(&'static str, &'static str)> {
    vec![("user", "print(1)"), ("assistant", "print(2)")]
}

/// The template IS this literal, and nothing else is in it. If this test ever
/// needs editing, the template changed — and the cost of the change is a
/// one-line edit to the two constants plus this literal, visible in the diff.
#[test]
fn template_is_the_documented_byte_sequence() {
    let ex = sft::encode(&exchange()).expect("roles are valid");
    assert_eq!(
        String::from_utf8_lossy(&ex.bytes),
        "<|im_start|>user\nprint(1)<|im_end|>\n\
         <|im_start|>assistant\nprint(2)<|im_end|>\n"
    );
    assert_eq!(ex.mask.len(), ex.bytes.len(), "mask is byte-parallel");
}

/// The mask covers the assistant content and the assistant's `<|im_end|>` — and
/// NOTHING else. Offsets come from [`at`], not from arithmetic, so a template
/// that grew a byte shows up as a mask mismatch rather than a silent
/// redefinition.
#[test]
fn mask_covers_assistant_content_and_its_terminator_only() {
    let ex = sft::encode(&exchange()).expect("roles are valid");
    let b = &ex.bytes[..];

    // <|im_start|>user\nprint(1)<|im_end|>\n  <- the whole user turn
    let user_content = at(b, b"print(1)", 0);
    let user_end = at(b, IM_END, 0);
    // <|im_start|>assistant\nprint(2)<|im_end|>\n  <- the whole assistant turn
    let asst_content = at(b, b"print(2)", 0);
    let asst_end = at(b, IM_END, 1);
    assert_eq!(
        b.len(),
        asst_end + IM_END.len() + 1,
        "the conversation ends with <|im_end|>\\n and nothing else"
    );

    let mut want = vec![0.0f32; b.len()];
    want[user_content..user_end]
        .iter_mut()
        .for_each(|m| *m = 0.0); // already 0
    want[asst_content..asst_end + IM_END.len()]
        .iter_mut()
        .for_each(|m| *m = 1.0);
    assert_eq!(ex.mask, want, "mask moved");

    assert_eq!(
        ex.trainable(),
        (asst_end + IM_END.len()) - asst_content,
        "assistant content + <|im_end|>, and the \\n after it is NOT trained"
    );
    // The honest summary: most of a two-turn conversation is unsupervised. If
    // this ever trips, the mask has started scoring headers or user bytes.
    assert!(
        ex.trainable() * 2 < ex.bytes.len(),
        "{} of {} bytes supervised - the mask is hiding too little",
        ex.trainable(),
        ex.bytes.len()
    );
}

/// A `system` turn is context like a user turn. A coder's SFT data is full of
/// system prompts, and masking them as targets is the most expensive way to be
/// wrong about a mask: it trains the policy to open every reply with its own
/// instructions.
#[test]
fn system_turns_are_never_targets() {
    let ex = sft::encode(&[
        ("system", "You are a coding assistant."),
        ("user", "hi"),
        ("assistant", "hello"),
    ])
    .expect("roles are valid");
    let asst_header = at(&ex.bytes, b"<|im_start|>assistant\n", 0);
    assert!(
        ex.mask[..asst_header].iter().all(|&m| m == 0.0),
        "a system or user byte is marked trainable"
    );
    assert!(ex.trainable() > 0, "the assistant span is masked away too");
}

/// A mistyped role is refused BY NAME. The alternative — rendering
/// `<|im_start|>usr\n` — produces a conversation whose headers the policy can
/// neither read as context nor emit at inference, and nothing downstream would
/// say so.
#[test]
fn an_unknown_role_is_refused_not_rendered() {
    let err = sft::encode(&[("usr", "hi")]).expect_err("usr is not a role");
    let msg = err.to_string();
    assert!(matches!(err, SftError::BadRole { .. }), "{msg}");
    assert!(msg.contains("usr"), "the message must name the role: {msg}");
    // The escape is named too: the rule is not "this is an error", it is
    // "here is how to make it legal" (ADR-0011).
    assert!(
        msg.contains("sft.rs"),
        "name the cause AND the escape: {msg}"
    );
}

/// A malformed line is refused with its LINE NUMBER, and an empty `messages`
/// list is an error rather than a zero-length example.
#[test]
fn jsonl_errors_name_the_line() {
    let path = tmp("bad.jsonl");
    std::fs::write(
        &path,
        concat!(
            r#"{"messages":[{"role":"user","content":"ok"}]}"#,
            "\n",
            "{not json}\n",
            r#"{"messages":[{"role":"tool","content":"x"}]}"#,
            "\n",
            r#"{"messages":[]}"#,
            "\n",
        ),
    )
    .expect("write the fixture");
    let err = sft::read_jsonl(&path).expect_err("line 2 is not JSON");
    assert!(err.to_string().contains("line 2"), "{err}");
    std::fs::write(
        &path,
        concat!(
            r#"{"messages":[{"role":"user","content":"ok"}]}"#,
            "\n",
            r#"{"messages":[{"role":"tool","content":"x"}]}"#,
            "\n",
        ),
    )
    .expect("rewrite");
    let err = sft::read_jsonl(&path).expect_err("tool is not a role");
    assert!(err.to_string().contains("line 2"), "{err}");
    assert!(err.to_string().contains("tool"), "{err}");
    std::fs::write(&path, r#"{"messages":[]}"#).expect("rewrite");
    let err = sft::read_jsonl(&path).expect_err("empty messages");
    assert!(matches!(err, SftError::EmptyRow { line: 1 }), "{err}");
    let _ = std::fs::remove_file(&path);
}

/// The one-byte shift between "byte i may be a target" and "label q may be
/// scored". The trainer's labels are `bytes[q + 1]`, so the loss must consume
/// the mask shifted the same way — and this is the exact off-by-one that fed
/// the DSpark draft head the byte it was about to predict.
#[test]
fn target_mask_is_the_byte_mask_shifted_by_one() {
    let ex = sft::encode(&exchange()).expect("roles are valid");
    let batch = SftBatch {
        bytes: ex.bytes.clone(),
        mask: ex.mask.clone(),
    };
    let tm = batch.target_mask();
    assert_eq!(
        tm.len(),
        ex.mask.len(),
        "same length: the loss consumes both"
    );
    for q in 0..tm.len() {
        assert_eq!(tm[q], ex.mask[(q + 1) % tm.len()], "label position {q}");
    }
    let asst_content = at(&ex.bytes, b"print(2)", 0);
    // Label q is byte q+1, so the FIRST byte of the assistant content is scored
    // at q = asst_content - 1 — one byte earlier than the content itself. Pass
    // `mask` where `target_mask` belongs and the loss trains on the `\n` that
    // ends the user turn instead of on the `p`. This assert is the difference
    // between "the mask is a bit off" and "the mask is a bit off".
    assert_eq!(
        tm[asst_content - 1],
        1.0,
        "label {q} must be the first assistant byte",
        q = asst_content - 1
    );
    assert_eq!(
        ex.mask[asst_content - 1],
        0.0,
        "the UNSHIFTED mask at that label position is 0 - which is why the \
         shift has to be explicit rather than absorbed by the caller"
    );
    assert_ne!(tm, ex.mask, "the shift is a no-op on this example");
}

/// A three-conversation JSONL file → a stream → batches, end to end, on CPU.
/// Three is the smallest count that crosses a batch boundary, which is the case
/// the mask has to survive.
#[test]
fn jsonl_to_batches_packs_back_to_back() {
    let path = tmp("pack.jsonl");
    let row = |u: &str, a: &str| {
        format!(
            r#"{{"messages":[{{"role":"user","content":"{u}"}},{{"role":"assistant","content":"{a}"}}]}}"#
        )
    };
    std::fs::write(
        &path,
        format!(
            "{}\n{}\n{}\n",
            row("aa", "bb"),
            row("cc", "dd"),
            row("ee", "ff")
        ),
    )
    .expect("write the fixture");

    let corpus: Vec<u8> = sft::read_jsonl(&path)
        .expect("the fixture parses")
        .iter()
        .flat_map(|e| e.bytes.clone())
        .collect();
    let trained = sft::read_jsonl(&path)
        .expect("the fixture parses")
        .iter()
        .map(|e| e.trainable())
        .sum::<usize>();
    let mut st = SftStream::new(sft::read_jsonl(&path).expect("parses"), 1, 32).expect("fits");
    assert_eq!(
        st.trainable(),
        trained,
        "every trained byte is in the stream"
    );

    // Continuous packing, the ByteStream rule: no per-example padding and no
    // document separator, so the batches concatenate back to the corpus.
    let mut seen = Vec::new();
    let mut seen_trained = 0usize;
    while let Ok(b) = st.next_batch() {
        assert_eq!(b.bytes.len(), 32);
        seen_trained += b.mask.iter().filter(|&&m| m != 0.0).count();
        seen.extend_from_slice(&b.bytes);
    }
    // The tail shorter than one batch is DROPPED, exactly as `ByteStream`
    // drops a short refill — never padded, because a padded mask is 0 and a
    // run that trains on it has a zero objective that reads like convergence.
    let whole = corpus.len() / 32 * 32;
    assert_eq!(
        corpus.len() % 32,
        3,
        "the fixture ends mid-batch, on purpose"
    );
    assert_eq!(
        seen,
        corpus[..whole],
        "the batches are the corpus, in order"
    );
    let tail_trained = sft::read_jsonl(&path)
        .expect("parses")
        .iter()
        .flat_map(|e| e.mask.clone())
        .skip(whole)
        .filter(|m| *m != 0.0)
        .count();
    assert!(
        tail_trained > 0,
        "the dropped tail must contain a supervised byte, or this assert is \
         testing nothing: it lands inside the last <|im_end|>"
    );
    assert_eq!(
        seen_trained,
        trained - tail_trained,
        "every trained byte appears exactly once across the batches, except the \
         ones in the dropped tail"
    );

    // Past the end is an error, not a repeat: a silently repeated batch is a
    // run that trains a thousand steps on one conversation.
    assert!(matches!(st.next_batch(), Err(SftError::Exhausted)));
    st.rewind();
    assert_eq!(st.next_batch().expect("after rewind").bytes[..], seen[..32]);

    let _ = std::fs::remove_file(&path);
}

/// A corpus smaller than one batch is refused at load. The alternative is a
/// padded batch whose mask is all zeros: a zero objective and a loss curve that
/// reads exactly like convergence.
#[test]
fn a_corpus_smaller_than_one_batch_is_refused() {
    let ex = sft::encode(&[("user", "hi"), ("assistant", "yo")]).expect("valid roles");
    let err = match SftStream::new(vec![ex], 8, 512) {
        Ok(_) => panic!("a 74-byte corpus was accepted as an 8x512 batch"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        matches!(err, SftError::TooSmall { have, need } if have < need && need == 8 * 512),
        "{msg}"
    );
    assert!(msg.contains("mask-zero"), "the message must say why: {msg}");
}

fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("dormouse-sft-{}-{}", std::process::id(), name));
    p
}
