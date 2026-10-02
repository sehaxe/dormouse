//! The export round-trip: export -> load -> the logits must match the source
//! model's within the format's MEASURED tolerance, on a fixed input.
//!
//! One test, because the assertions are one claim: "an exported model is the
//! model, at the format's precision". A wrong-but-loadable weight file is the
//! cardinal sin (ADR-0011) - it produces confident text from a broken model and
//! nothing downstream complains - so every property the format claims is
//! asserted here rather than described in a doc:
//!
//! - f32 export reproduces the source logits BIT-EXACTLY (nothing to narrow),
//!   so any drift in the container is the container's;
//! - f16 and bf16 land inside the tolerance measured on a trained model
//!   (docs/adr/0023-inference-export.md) - the numbers below are that measurement, not a guess;
//! - the file is the header plus exactly 2 bytes per parameter, so the size
//!   claim in the README is asserted arithmetic that cannot rot;
//! - the header carries the config, so the file describes its own shape;
//! - a corrupted payload is REFUSED by checksum, not loaded;
//! - a training checkpoint is REFUSED with the command that converts it, and
//!   reading it does not go looking for the `.ngram` sidecar;
//! - an f16 export of a model with a weight above 65504 is REFUSED, not
//!   written with an infinity in it.
//!
//! Backend-agnostic: the narrowing is host arithmetic over parameters read back
//! once, so the same body runs on CPU and CUDA. Run both anyway - a gate that
//! only ever runs on one backend proves one backend:
//!   cargo test -p dormouse-train --test export_roundtrip
//!   cargo test -p dormouse-train --no-default-features --features cuda --test export_roundtrip

use burn::module::Module;
use dormouse_core::{DormouseConfig, DormouseModel};
use dormouse_train::decode;
use dormouse_train::export::{self, DType};

/// The crate's own device, NOT a hardcoded one. Under `--features cuda` a
/// hardcoded `Device::ndarray()` here would build the source model on one
/// backend and the loader builds the decoded one on another, and the 1-ULP
/// difference between two backends would be reported as an export defect.
fn device() -> burn::tensor::Device {
    dormouse_train::device()
}

/// Mini-nano: the seam contract is width-independent, and a full-nano init is
/// slow on a CPU test run (same reasoning as ckpt_roundtrip.rs).
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

/// A fixed input, so the numbers are comparable run to run.
fn input() -> Vec<u8> {
    b"the quick brown fox jumps over the lazy dog; pack my box with five dozen".to_vec()
}

/// Logits after ONE warmup forward, discarded.
///
/// Not ceremony: the cubecl autotuner picks a kernel plan per (shape, dtype)
/// on the first call, and two plans that are mathematically equal reduce in a
/// different order. A fresh model's FIRST forward therefore differs from a
/// warmed model's by ~1 f32 ULP on CUDA (measured: 8.9e-8), which is the
/// autotuner's business and not the container's. Warming both sides makes the
/// comparison a statement about the EXPORT.
fn logits(m: &DormouseModel) -> Vec<f32> {
    // Through the decode seam, so this comparison is a statement about the
    // EXPORT and not about which memory keys each side happened to pass.
    let _ = decode::next_byte_logits::<dormouse_train::Backend>(m, &input());
    decode::next_byte_logits::<dormouse_train::Backend>(m, &input())
}

fn max_delta(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "logits must be the same shape");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn header_len(raw: &[u8]) -> usize {
    u32::from_le_bytes(raw[8..12].try_into().expect("4-byte window")) as usize
}

/// Measured max |logit delta| of each format against the fp32 model, on
/// HELD-OUT text at the `small` preset (9.20M params, 16 windows of 256 B from
/// `real_eval_v2/eval_tail.bin`, a 20-step run; docs/adr/0023-inference-export.md carries the full
/// table and the program that produced it):
///
///     f16  1.247e-1     bf16  1.025e-1      (logits |max| 1.11)
///
/// Both are ~1/10 of the logits' own scale, i.e. the error is relative, not a
/// catastrophe. The tolerances below are 4x those measurements plus the
/// headroom a DIFFERENT random init can move them by - this file's model is a
/// fresh init at a narrower width, and a tolerance that only fits one seed is
/// not a tolerance. The f32 assertion above is the exact one: it is the only
/// delta that must be 0.
const TOL_F16: f32 = 0.5;
const TOL_BF16: f32 = 0.5;

#[test]
fn export_roundtrip_reproduces_the_model() {
    let cfg = cfg();
    let src = DormouseModel::new(&cfg, &device()).no_grad();
    let ref_logits = logits(&src);
    assert!(
        ref_logits.iter().all(|x| x.is_finite()),
        "the reference forward must be finite or every comparison below is noise"
    );
    let tensors = export::tensors_of(&src);
    let num_params: usize = tensors.iter().map(|(_, _, f)| f.len()).sum();
    assert!(num_params > 0, "the model must have parameters to export");

    // --- f32: no narrowing, so this is the container's own claim ---
    let raw32 = export::encode(&src, &cfg, "test", 7, DType::F32).expect("f32 encodes");
    let (m32, cfg32, h32) = export::decode(&raw32).expect("f32 decodes");
    assert_eq!(
        cfg32, cfg,
        "the header must carry the config the model was built with"
    );
    assert_eq!(h32.num_params, num_params as u64);
    assert_eq!(h32.num_tensors, tensors.len());
    assert!(
        h32.min_nonzero_abs > 0.0 && h32.min_nonzero_abs <= h32.max_abs,
        "the header's range must be a real range: min {} max {}",
        h32.min_nonzero_abs,
        h32.max_abs
    );
    assert_eq!(
        raw32.len(),
        export::FIXED + header_len(&raw32) + num_params * 4,
        "f32: the file is the header plus exactly 4 bytes per parameter"
    );
    assert_eq!(
        max_delta(&ref_logits, &logits(&m32)),
        0.0,
        "an f32 export must reproduce the source logits bit-exactly: the payload is \
         the same f32s, so any drift here is the container's"
    );

    // --- f16 and bf16: the measured tolerance, and the size claim ---
    for (dtype, tol, width) in [(DType::F16, TOL_F16, 2), (DType::Bf16, TOL_BF16, 2)] {
        let raw = export::encode(&src, &cfg, "test", 7, dtype)
            .unwrap_or_else(|e| panic!("{dtype:?} encodes: {e}"));
        let (m, _, h) = export::decode(&raw).unwrap_or_else(|e| panic!("{dtype:?} decodes: {e}"));
        assert_eq!(h.dtype, dtype);
        assert_eq!(h.num_params, num_params as u64);
        assert_eq!(h.num_tensors, tensors.len());
        assert_eq!(
            raw.len(),
            export::FIXED + header_len(&raw) + num_params * width,
            "{dtype:?}: the file is the header plus exactly {width} bytes per parameter"
        );
        // One readback for the whole check: `tensors_of` pulls every parameter
        // to the host, so calling it per tensor would re-read the model once
        // per tensor and turn this gate into a timeout.
        let loaded: std::collections::HashMap<String, (Vec<usize>, usize)> = export::tensors_of(&m)
            .into_iter()
            .map(|(n, d, f)| (n, (d, f.len())))
            .collect();
        assert_eq!(
            loaded.len(),
            h.tensors.len(),
            "a tensor went missing across the roundtrip"
        );
        for t in &h.tensors {
            assert!(
                !t.name.is_empty() && !t.dims.is_empty() && t.overflowed == 0,
                "a malformed tensor entry: {t:?}"
            );
            let (dims, len) = loaded.get(&t.name).unwrap_or_else(|| {
                panic!(
                    "{name}: in the header but not in the loaded model",
                    name = t.name
                )
            });
            assert_eq!(
                *dims,
                t.dims,
                "{name}: dims changed across the roundtrip",
                name = t.name
            );
            assert_eq!(
                t.dims.iter().product::<usize>(),
                *len,
                "{name}: element count disagrees with its own dims",
                name = t.name
            );
        }
        let d = max_delta(&ref_logits, &logits(&m));
        println!("{dtype:?}: max |logit delta| {d:.3e} (tolerance {tol:.1e})");
        assert!(
            d <= tol,
            "{dtype:?}: max |logit delta| {d:.3e} exceeds the measured {tol:.1e}"
        );
    }

    // --- a corrupted payload is REFUSED, not loaded ---
    let mut bad = export::encode(&src, &cfg, "test", 7, DType::Bf16).expect("encodes");
    let n = bad.len();
    bad[n - 1] ^= 0x01;
    let err = export::decode(&bad).expect_err("a flipped payload bit must be refused");
    assert!(
        err.contains("checksum"),
        "the refusal must name the checksum, got: {err}"
    );

    // --- a training checkpoint is REFUSED, and the sidecar is not touched ---
    let dir = std::env::temp_dir().join("dm-export-refuses-ckpt");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("tmpdir");
    let ckpt = dir.join("rt.bin");
    write_mini_ckpt(&ckpt);
    let err = export::read(&ckpt).expect_err("a training checkpoint is not an export");
    assert!(
        err.contains("TRAINING CHECKPOINT") && err.contains("dormouse export"),
        "the refusal must say what the file is and how to convert it, got: {err}"
    );
    assert!(
        export::export_ckpt(&ckpt, Some("nano"), &[], DType::Bf16, &dir.join("rt.dmexp")).is_ok(),
        "the conversion the refusal points at must actually work"
    );
    assert!(
        !dir.join("rt.ngram").exists(),
        "exporting a checkpoint must not have produced or read a sidecar"
    );

    // --- a header whose config disagrees with its own payload is REFUSED ---
    // Found by hand, not by the test: editing `d_model` in the header and
    // fixing the CRC so the LOAD is what must catch it used to install the
    // payload's shapes into a model built at the header's width. The load
    // succeeded, the forward returned garbage, and nothing said so - because
    // `Param::from_data` takes the payload's dims, so no downstream check can
    // ever see it.
    let src_raw = export::encode(&src, &cfg, "test", 7, DType::Bf16).expect("encodes");
    let hlen = header_len(&src_raw);
    let plen = u64::from_le_bytes(src_raw[12..20].try_into().expect("8-byte window")) as usize;
    let header = String::from_utf8(src_raw[export::FIXED..export::FIXED + hlen].to_vec())
        .expect("the header is utf-8");
    let patched = header
        .replace(
            &format!("d_model = {}", cfg.d_model),
            &format!("d_model = {}", cfg.d_model / 2),
        )
        .into_bytes();
    assert_ne!(
        patched,
        header.as_bytes(),
        "the header must contain d_model for this to test anything"
    );
    let mut tampered = Vec::with_capacity(export::FIXED + patched.len() + plen);
    tampered.extend_from_slice(&src_raw[..export::FIXED]);
    tampered[8..12].copy_from_slice(&(patched.len() as u32).to_le_bytes());
    tampered.extend_from_slice(&patched);
    tampered.extend(std::iter::repeat_n(0u8, plen));
    // Re-checksum, so the CRC passes and the SHAPE check is what refuses.
    let crc = crc32fast::hash(&tampered[export::FIXED + patched.len()..]);
    tampered[20..24].copy_from_slice(&crc.to_le_bytes());
    let err = export::decode(&tampered)
        .expect_err("a header that disagrees with its own payload must be refused");
    assert!(
        err.contains("does not describe the weights") && err.contains("embedding.weight"),
        "the refusal must name the shape conflict, got: {err}"
    );

    // --- f16 refuses a weight it cannot represent; bf16 accepts it ---
    let mut big = DormouseModel::new(&cfg, &device()).no_grad();
    big.embedding.weight = burn::module::Param::from_tensor(burn::tensor::Tensor::from_data(
        burn::tensor::TensorData::new(vec![70_000.0f32; 128 * 256], [128, 256]),
        &device(),
    ));
    let err = export::encode(&big, &cfg, "test", 0, DType::F16)
        .expect_err("f16 must refuse a weight above 65504");
    assert!(
        err.contains("bf16"),
        "the refusal must name the format that works, got: {err}"
    );
    assert!(
        export::encode(&big, &cfg, "test", 0, DType::Bf16).is_ok(),
        "bf16 has f32's exponent range and must accept the same model"
    );
}

/// A real container, not a stub: the refusal above is only proven against a
/// file with the checkpoint's actual header, and the conversion that follows it
/// is the real path - which needs the config snapshot the trainer writes next
/// to every checkpoint, or the converter has no shape to build the model with.
fn write_mini_ckpt(path: &std::path::Path) {
    use dormouse_train::{build_optim, save_ckpt, RunCfg, TrainCfg};
    let dir = path.parent().expect("a parent dir");
    let cfg = cfg();
    let model = DormouseModel::new(&cfg, &device());
    let optim = build_optim(&model, &TrainCfg::default());
    save_ckpt(dir, "rt", &model, &optim, None, false, 3, 2.5).expect("saves");
    assert!(path.exists(), "save_ckpt wrote {path:?}");
    let run = RunCfg {
        source: "test".into(),
        model: cfg,
        train: TrainCfg::default(),
    };
    std::fs::write(dir.join("rt.config.toml"), run.snapshot_toml()).expect("snapshot");
}
