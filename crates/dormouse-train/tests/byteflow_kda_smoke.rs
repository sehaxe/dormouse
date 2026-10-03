//! THE COMPAT CHANNEL's gates (lane bf-compat; the findings are in
//! `docs/reviews/bf-compat-2026-10-03.md`).
//!
//! `use_byteflow` + `use_kda` is the ONE pair this lane lifts, and the pair is
//! allowed to exist only with the refusals around it intact — one refusal
//! lifted at a time, each with a gate, is the whole method. So:
//!
//! 1. [`the_lifted_pair_resolves_and_the_channel_arms`] — the pair resolves
//!    (before the lane it was a refusal) and the dispatch takes the channel
//!    branch, which is `bf::compat_armed`, the one list `validate` refuses
//!    against.
//! 2. [`one_step_trains_through_the_real_train_loop`] — one step through the
//!    REAL `train_loop`: the dispatch, the model, the backward, the optimizer,
//!    the checkpoint, the config snapshot, and an EVAL whose own forwards went
//!    through the channel (the `bf=` field the eval line prints — `bf=0` there
//!    is §3.2's defect shape, an eval that scored a different network).
//! 3. [`the_untouched_refusals_still_refuse`] — everything this lane did NOT
//!    lift still refuses, including `byteflow::check`'s train-side flags,
//!    which `resolve` never sees.
//! 4. [`kda_group_receives_gradient_at_patch_input`] — gradient flow from the
//!    DECODER'S byte CE back into KDA's q/k, whose input arrived as a patch
//!    latent. The `8fa5d4c` class: a dead arm keeps a perfectly healthy loss
//!    curve, so only a gradient assertion can see it.
//! 5. [`an_aux_weight_cannot_be_silently_dropped`] — `validate` is not called
//!    by the model constructors, so the same refusal is held at the runtime
//!    seam by an `assert!` in `forward_channel`. This is the gate that keeps
//!    that assert from being deleted as unreachable.
//!
//! CPU only, like every other gate here: `cargo test -p dormouse-train
//! --test byteflow_kda_smoke`.

use std::path::PathBuf;

use burn::tensor::{Int, Tensor, TensorData};
use dormouse_core::config::validate;
use dormouse_core::probe;
use dormouse_core::{DormouseConfig, DormouseModel};
use dormouse_train::{byteflow, resolve, train_loop, TrainCfg};

/// The trainer's own autodiff backend, so a model built here is the model the
/// trainer builds (same device, same checkpointing strategy).
type Backend = dormouse_train::Backend;

fn sets(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

/// The pair the lane lifts, at a window the CPU can run: K = 16 patch latents
/// over a 32-byte sequence. K = 16 is not arbitrary — the attention arm's
/// chunk size is 16 (`crates/dormouse-core/src/attention.rs`), so the loop
/// sees exactly one full chunk, the shape `preset_exec` already exercises at
/// SEQ = 32.
fn pair() -> Vec<String> {
    sets(&["byteflow_k_tokens=16"])
}

#[test]
fn the_lifted_pair_resolves_and_the_channel_arms() {
    let run = resolve(
        "byteflow_kda",
        &pair(),
        TrainCfg {
            steps: 1,
            seq_len: 32,
            batch: 2,
            ckpt_name: "bfk".into(),
            ..Default::default()
        },
    )
    .expect("use_byteflow + use_kda must resolve: this is the pair the lane lifts");
    assert!(
        run.model.use_byteflow && run.model.use_kda,
        "the preset must carry the lifted pair: {:?}",
        (run.model.use_byteflow, run.model.use_kda)
    );
    // The dispatch's own decision, from the one list: `train_loop` branches on
    // exactly this, so a channel that never gets built is a run that silently
    // took the standalone arm.
    assert!(
        dormouse_core::bf::compat_armed(&run.model),
        "compat_armed is false: the dispatch would run the STANDALONE byteflow net and the \
         KDA arm would never exist"
    );
}

/// A 1-step run through the real trainer, both halves of the counter pair:
/// the TRAINING step enters the channel, and so does the EVAL. The second
/// half is the one §3.2's history says to check — the pre-`7adda92` eval
/// passed `hashed_ids = None` unconditionally and scored a different network
/// than the one being trained, and the only surface that can show it is the
/// eval's own counter (`bf=` on the eval line is `count - bf0`).
#[test]
fn one_step_trains_through_the_real_train_loop() {
    let root = std::env::temp_dir().join("dm-bf-kda-smoke");
    let _ = std::fs::remove_dir_all(&root);
    let (data, eval, ckpts) = (root.join("train"), root.join("eval"), root.join("ckpts"));
    for d in [&data, &eval, &ckpts] {
        std::fs::create_dir_all(d).unwrap();
    }
    // 2 KiB of learnable ramp per tree: above the stream's floor
    // (seq_len * batch * 4 = 256) with room for the eval window's own carve.
    let corpus: Vec<u8> = (0..2048u32)
        .map(|i| (i.wrapping_mul(7) % 13) as u8)
        .collect();
    std::fs::write(data.join("corpus.bin"), &corpus).unwrap();
    std::fs::write(eval.join("corpus.bin"), &corpus).unwrap();

    let run = resolve(
        "byteflow_kda",
        &pair(),
        TrainCfg {
            steps: 1,
            seq_len: 32,
            batch: 2,
            // One eval at step 1 over 2 batches: the eval's forwards are
            // counted separately from the step's, which is exactly what the
            // printed `bf=` field reports.
            eval_every: 1,
            eval_batches: 2,
            ckpt_name: "bfk".into(),
            ..Default::default()
        },
    )
    .expect("the lifted pair resolves");
    assert!(run.model.use_kda, "the pair must still carry use_kda");

    probe::reset();
    train_loop(run, data, Some(ckpts.clone()), Some(eval)).expect("1 compat-channel step on CPU");

    // Weights, the log line and the config snapshot (the snapshot is what a
    // resume compares against — the pair must be IN it).
    assert!(
        ckpts.join("bfk.bin").is_file(),
        "the channel run's checkpoint must be written"
    );
    let txt = std::fs::read_to_string(ckpts.join("bfk.txt")).unwrap();
    assert!(txt.contains("step 1"), "{txt}");
    let snap = std::fs::read_to_string(ckpts.join("bfk.config.toml")).unwrap();
    assert!(
        snap.contains("use_byteflow = true") && snap.contains("use_kda = true"),
        "the snapshot must carry the lifted pair:\n{snap}"
    );

    // The channel RAN — the training step and the eval's own forwards. This is
    // the number behind the eval line's `bf=` field: a zero there would mean
    // the eval scored a channel-less network (§3.2's shape).
    let bf = probe::count(probe::BF_CHANNEL);
    assert!(
        bf >= 2,
        "the compat channel entered {bf} times; the step's forward plus the eval's own \
         forwards must both go through it (eval_every=1, eval_batches=2)\n  counters: {:?}",
        probe::counts()
    );
    // And the arm the pair is FOR: KDA entered on patch input.
    assert!(
        probe::count(probe::KDA) > 0,
        "use_kda is on and the attention arm never entered\n  counters: {:?}",
        probe::counts()
    );
}

/// Every refusal this lane did NOT lift. Two halves, because they live in two
/// different places and `resolve` only ever sees the first: `config::validate`
/// (model-side, reached through `resolve`) and `byteflow::check` (train-side,
/// called at the dispatch site — `--bf16` and friends are NOT config fields
/// `resolve` could see, so a gate that only went through `resolve` would pass
/// while the flags were wide open).
#[test]
fn the_untouched_refusals_still_refuse() {
    let refused = |set: &str| -> String {
        resolve("byteflow_kda", &sets(&[set]), TrainCfg::default())
            .expect_err(&format!("{set} must still be refused with use_byteflow"))
    };

    // 1. The OTHER seven loop arms. `use_kda` is the one that was lifted and
    //    is deliberately absent from this list — that absence IS the lane.
    //
    //    Six go through `resolve`; `use_msa` does not, because `--set` has no
    //    `use_msa` key (`config/override.rs` lists every arm but that one —
    //    pre-existing, reported, not this lane's to add), so its refusal is
    //    asserted at `validate`, which is where the Err actually lives.
    for arm in [
        "use_engram",
        "use_mor",
        "use_gr",
        "use_attnres",
        "use_mhc",
        "use_situ",
    ] {
        let set = format!("{arm}=true");
        let err = refused(&set);
        assert!(
            err.contains("use_byteflow") && err.contains(arm),
            "{set}: wrong refusal (or none): {err}"
        );
    }
    {
        let mut cfg = resolve("byteflow_kda", &[], TrainCfg::default())
            .expect("the lifted pair resolves")
            .model;
        cfg.use_msa = true;
        let err = validate(&cfg).expect_err("use_msa with use_byteflow must be refused");
        assert!(
            err.contains("use_byteflow") && err.contains("use_msa"),
            "wrong refusal (or none): {err}"
        );
    }
    // And the schema default itself: `use_byteflow = true` AND the arms that
    // default on, so a flat config naming neither is refused. This is why
    // `aux.rs::mini()` had to take `use_byteflow: false` — findings §6.5.
    let err = validate(&DormouseConfig::default())
        .expect_err("the schema default carries arms together with use_byteflow");
    assert!(err.contains("use_byteflow"), "{err}");

    // 2. Every aux weight: the channel carries no aux heads, and a weight left
    //    on would be an ADR-0019 term with nothing to act on.
    for set in [
        "jepa_weight=0.1",
        "dspark_weight=0.1",
        "aux_fb_weight=0.1",
        "mor_bce_weight=0.1",
    ] {
        let err = refused(set);
        assert!(
            err.contains("use_byteflow"),
            "{set}: wrong refusal (or none): {err}"
        );
    }

    // 3. The dead knob and the two window checks. `byteflow_max_bytes` is the
    //    pre-existing RoPE-bound Err; `byteflow_k_tokens > max_seq_len` is the
    //    one Err this lane added (without it the first forward panics inside
    //    `encode_chunks` with a shape message that names no flag).
    let err = refused("moe_topk=2");
    assert!(err.contains("moe_topk"), "{err}");
    let err = refused("byteflow_k_tokens=1024"); // > max_seq_len (512)
    assert!(
        err.contains("byteflow_k_tokens") && err.contains("max_seq_len"),
        "{err}"
    );
    let err = refused("byteflow_max_bytes=256"); // < max_seq_len (512)
    assert!(err.contains("byteflow_max_bytes"), "{err}");

    // 4. The TRAIN-SIDE flags, called directly: `resolve` has no view of them,
    //    and this lane lifts none of them. The default must pass first, or
    //    every case below would be vacuous — a `check` that refuses
    //    everything refuses these too.
    byteflow::check(&TrainCfg::default())
        .expect("the default train cfg must not be one of the refused ones");
    let cases: [(TrainCfg, &str); 8] = [
        (
            TrainCfg {
                bf16: Some(true),
                ..Default::default()
            },
            "--bf16",
        ),
        (
            TrainCfg {
                graph_capture: true,
                ..Default::default()
            },
            "--graph-capture",
        ),
        (
            TrainCfg {
                jepa_targets: Some(PathBuf::from("targets.bin")),
                ..Default::default()
            },
            "--jepa-targets",
        ),
        (
            TrainCfg {
                engram_ram: true,
                ..Default::default()
            },
            "--engram-ram",
        ),
        (
            TrainCfg {
                rand_depth: true,
                ..Default::default()
            },
            "--rand-depth",
        ),
        (
            TrainCfg {
                eval_depths: true,
                ..Default::default()
            },
            "--eval-depths",
        ),
        (
            TrainCfg {
                stress: true,
                ..Default::default()
            },
            "--stress",
        ),
        (
            TrainCfg {
                opt: "muon".into(),
                ..Default::default()
            },
            "--opt",
        ),
    ];
    for (tc, what) in cases {
        let err = byteflow::check(&tc).expect_err(&format!("{what} must be refused"));
        assert!(err.contains(what), "{what}: unexpected error: {err}");
    }
}

/// THE GRADIENT, from the objective that actually is this run's loss.
///
/// The objective is the DECODER'S byte CE (`forward_channel` runs the loop
/// with `targets = None`), so every parameter behind it has to be reachable
/// from that scalar — and the interesting ones are the two whose failure is
/// invisible in the loss curve:
///
/// * KDA's `q_proj`/`k_proj`, whose INPUT is a patch latent (ByteFlow's front
///   stage → `in_proj` → loop) rather than a byte embedding. An arm that takes
///   no gradient runs thousands of forwards and trains nothing while the loss
///   looks healthy — that is `8fa5d4c`, and it cost this project its whole
///   attention history.
/// * The BACK stage (`upsample_w`, the byte head `out`), which exists only
///   behind the loop. A loss taken at K patch starts instead would leave all
///   of it grad-free, which is exactly why the objective is the byte CE.
///
/// Also pinned here: the shapes the assertions are made about —
/// `[2, 32, 256]` logits, `aux = None` (the channel carries no aux heads),
/// KDA's state in its real shape rather than the `[b,1,1,1]` placeholder, and
/// the two counters at their exact expected values.
#[test]
fn kda_group_receives_gradient_at_patch_input() {
    let run =
        resolve("byteflow_kda", &pair(), TrainCfg::default()).expect("the lifted pair resolves");
    let cfg = run.model;
    validate(&cfg).expect("the resolved config validates");
    let dev = dormouse_train::device();
    let model = DormouseModel::new(&cfg, &dev);

    let (b, t) = (2usize, 32usize);
    let bytes: Vec<i64> = (0..b * t).map(|i| (i % 251) as i64).collect();
    let x = Tensor::<2, Int>::from_data(TensorData::new(bytes.clone(), [b, t]), &dev);
    // Next-byte labels, the train loop's shift-by-one.
    let y = Tensor::<2, Int>::from_data(
        TensorData::new(
            bytes.iter().map(|&v| (v + 1) % 251).collect::<Vec<_>>(),
            [b, t],
        ),
        &dev,
    );

    probe::reset();
    let (logits, rec, kda, aux) =
        model.forward_with_hidden::<Backend>(x, None, None, Some(y), None);

    assert_eq!(
        logits.dims(),
        [b, t, 256],
        "the decoder must produce one byte distribution per position"
    );
    assert!(
        aux.is_none(),
        "the channel returns aux = None unconditionally (validate refuses every aux weight)"
    );
    let rec_v: Vec<f32> = rec.clone().into_data().try_to_vec().expect("rec readable");
    assert!(
        rec_v.iter().all(|v| v.is_finite()) && rec_v[0] > 0.0,
        "the byte CE must be a finite positive number: {rec_v:?}"
    );
    // KDA state in its real shape: `[b, heads, K, v]` — the `[b,1,1,1]`
    // placeholder is what `use_kda = false` returns, so this is shape-level
    // proof the arm carried state over K patch positions.
    assert_eq!(kda.dims()[0], b, "kda state batch dim");
    assert!(
        kda.dims().iter().skip(1).any(|d| *d > 1),
        "kda state is the placeholder shape {:?}",
        kda.dims()
    );
    assert_eq!(
        probe::count(probe::BF_CHANNEL),
        1,
        "the channel must run exactly once per forward"
    );
    assert_eq!(
        probe::count(probe::KDA),
        cfg.max_iter as u64,
        "use_kda: one entry per loop iteration (max_iter={})",
        cfg.max_iter
    );

    // ---- the backward -------------------------------------------------
    let grads = rec.backward();
    let peak = |name: &str, v: Vec<f32>| {
        let p = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        assert!(
            p.is_finite() && p > 0.0,
            "{name}: peak |grad| = {p} - this parameter took no gradient from the byte CE"
        );
    };
    let takes_grad = |name: &str, g: Option<Tensor<2>>| {
        let g = g.unwrap_or_else(|| panic!("{name}: NO gradient"));
        peak(name, g.into_data().try_to_vec().expect("grad readable"));
    };

    // The attention arm, on an input that arrived as a patch latent.
    let g = &model.loop_block.shared_attn.gdn2;
    takes_grad("kda q_proj (patch input)", g.q_proj.weight.grad(&grads));
    takes_grad("kda k_proj (patch input)", g.k_proj.weight.grad(&grads));
    // The front stage: the patch projection that feeds the loop, and the
    // encoder behind the discrete chunker (the SELECTION has no gradient path
    // by construction — the gathered rows behind it do).
    let bf = model.bf.as_ref().expect("the channel must be built");
    takes_grad(
        "channel in_proj (d_global -> loop)",
        bf.in_proj.weight.grad(&grads),
    );
    takes_grad(
        "byteflow proj (encode, behind the discrete chunker)",
        bf.net.proj.weight.grad(&grads),
    );
    // The back stage, which exists only behind the loop.
    takes_grad(
        "channel out_proj (loop -> d_global)",
        bf.out_proj.weight.grad(&grads),
    );
    let up = bf.net.upsample_w.grad(&grads).unwrap_or_else(|| {
        panic!("upsample_w: NO gradient - the multilinear lift took nothing from the byte CE")
    });
    peak(
        "upsample_w (the K -> T lift)",
        up.into_data().try_to_vec().expect("grad readable"),
    );
    takes_grad(
        "byte head out (decoder -> logits)",
        bf.net.out.weight.grad(&grads),
    );
}

/// `validate` lives at the config seam and the model constructors do not call
/// it (they must not: a checkpoint loads through them). So the aux refusal
/// `validate` holds at resolve time is held AGAIN at the forward, by an
/// `assert!` — otherwise a hand-built config with `jepa_weight > 0` and a
/// channel armed would silently drop the term (an ADR-0019 weight that costs a
/// config field and produces nothing). This test is the reason that assert
/// cannot be deleted as unreachable.
#[test]
#[should_panic(expected = "carries no aux heads")]
fn an_aux_weight_cannot_be_silently_dropped() {
    let run =
        resolve("byteflow_kda", &pair(), TrainCfg::default()).expect("the lifted pair resolves");
    // A config no `resolve` can produce, built the way a loader would: the
    // constructors are the seam, so this is the seam. Widths shrunk because
    // the panic happens before the first real op and a 9M-param build buys
    // nothing here.
    let cfg = DormouseConfig {
        jepa_weight: 0.1,
        d_model: 64,
        n_heads: 4,
        head_dim: 16,
        d_ffn: 128,
        rank: 16,
        engram_rows: 512,
        ..run.model
    };
    let dev = dormouse_train::device();
    let model = DormouseModel::new(&cfg, &dev);
    assert!(
        model.bf.is_some(),
        "the channel must be armed for this test to reach the assert"
    );
    let x = Tensor::<2, Int>::from_data(
        TensorData::new(
            (0..64).map(|i| (i % 251) as i64).collect::<Vec<_>>(),
            [2, 32],
        ),
        &dev,
    );
    let y = Tensor::<2, Int>::from_data(
        TensorData::new(
            (1..65).map(|i| (i % 251) as i64).collect::<Vec<_>>(),
            [2, 32],
        ),
        &dev,
    );
    let _ = model.forward_with_hidden::<Backend>(x, None, None, Some(y), None);
    panic!("the aux assert did not fire - an aux weight is being dropped silently");
}
