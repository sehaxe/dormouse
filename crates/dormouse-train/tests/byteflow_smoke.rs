//! The ByteFlow arm's 1-step smoke, on CPU, through the REAL entry point:
//! `dormouse_train::train_loop` with `use_byteflow` — the dispatch, the
//! config snapshot, the byteflow loop, the AdamW step and the checkpoint
//! write all run. A flag that only a unit stub exercises is a flag that
//! rots; this is the gate the queued GPU A/B has to pass first.

use dormouse_train::{resolve, train_loop, RunCfg, TrainCfg};

fn runner(tag: &str, sets: &[&str]) -> (RunCfg, std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
    let root = std::env::temp_dir().join(format!("dm-bf-smoke-{tag}"));
    let _ = std::fs::remove_dir_all(&root);
    let (data, eval, ckpts) = (
        root.join("train"),
        root.join("eval"),
        root.join("ckpts"),
    );
    for d in [&data, &eval, &ckpts] {
        std::fs::create_dir_all(d).unwrap();
    }
    // 128+ bytes of learnable ramp in each tree (the stream's floor is
    // seq_len*batch*4 = 128); the split trees keep the no-leak rule quiet.
    let corpus: Vec<u8> = (0..256u32).map(|i| (i.wrapping_mul(7) % 13) as u8).collect();
    std::fs::write(data.join("corpus.bin"), &corpus).unwrap();
    std::fs::write(eval.join("corpus.bin"), &corpus).unwrap();
    let run = resolve(
        "byteflow",
        // K=4: the preset's 128 global units cannot exist at seq_len 16
        // (encode_chunks asserts k <= t — the loud shape check, not a crop).
        &sets.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        TrainCfg {
            steps: 1,
            seq_len: 16,
            batch: 2,
            ckpt_name: "bf".into(),
            ..Default::default()
        },
    )
    .expect("the byteflow preset resolves with the arm's refusals quiet");
    (run, data, eval, ckpts)
}

#[test]
fn one_step_trains_and_saves_through_the_real_entry() {
    let (run, data, eval, ckpts) = runner("s1", &["byteflow_k_tokens=4"]);
    train_loop(run, data, Some(ckpts.clone()), Some(eval))
        .expect("1 byteflow step on CPU");
    // Weights, the ce line and the config snapshot (written by the dispatch
    // site's shared snapshot code) all exist.
    assert!(ckpts.join("bf.bin").is_file(), "the byteflow checkpoint must be written");
    let txt = std::fs::read_to_string(ckpts.join("bf.txt")).unwrap();
    assert!(txt.contains("step 1"), "{txt}");
    assert!(
        std::fs::read_to_string(ckpts.join("bf.config.toml"))
            .unwrap()
            .contains("use_byteflow = true"),
        "the snapshot must carry the arm"
    );
    let ce = txt.split_whitespace().last().unwrap().trim().to_string();
    assert!(ce.parse::<f32>().map(|v| v.is_finite()).unwrap_or(false), "ce line: {txt}");
}

#[test]
fn a_preset_arm_conflict_is_refused_loudly() {
    // `--byteflow` on a preset that turns a dormouse arm on: validate must
    // refuse with the flag named, not train a model the config does not
    // describe (this is WHY the preset keeps every dormouse arm off).
    let err = resolve(
        "small",
        &["use_byteflow=true".to_string()],
        TrainCfg::default(),
    )
    .expect_err("use_byteflow with use_kda/use_engram must be refused");
    assert!(
        err.contains("use_byteflow") && err.contains("use_kda"),
        "unexpected error: {err}"
    );
}
