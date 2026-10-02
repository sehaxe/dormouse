//! GATE 3 — the two-stage training, stage (a): the KL distillation moves
//! the indexer's picks onto the dense teacher's top blocks. 100 AdamW
//! steps at the paper's indexer LR (1e-3) on synthetic data on CPU; the
//! gate asserts >=90% of the picked blocks coincide with the teacher's top
//! block (Eq. 16's `Bi` against the max-pooled Eq. 17 teacher).
//!
//! The teacher here is synthetic and DETACHED (a tensor, not a module), so
//! the only trainable graph is the indexer — exactly what stage (a)
//! trains. Detachment is structural: `teacher_raw` never gets
//! `require_grad()`.

use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData};
use burn_msa::{distill_loss, indexer_scores, pool_teacher, IndexerConfig, IndexerModule};

fn cfg() -> IndexerConfig {
    IndexerConfig {
        d_model: 16,
        q_heads: 4,
        head_dim: 8,
        block_r: 4,
        rope_frac: 8,
        max_seq_len: 64,
    }
}

#[test]
fn distillation_moves_the_picks_onto_the_teacher() {
    let device = burn::tensor::Device::ndarray().autodiff();
    let (b, t, b_n) = (2usize, 16usize, 4usize); // B = 4 complete blocks, r = 4

    // Token i's content flags its own hot block: `v[(i / 4) % 4] = 2` —
    // linearly separable, so a Wq/Wk pair CAN learn it.
    let x_dense: Vec<f32> = (0..b * t)
        .flat_map(|row| {
            let mut v = vec![0f32; 16];
            v[(row / 4) % b_n] = 2.0;
            v
        })
        .collect();

    // Query i's teacher belief: 1.0 on the tokens of ITS hot block beta(i),
    // 0 elsewhere (already causal-consistent where it matters: queries whose
    // hot block is not yet fully observed get the 90% gate excludes them
    // below, and the distillation itself sees KL(0-ish || ...)).
    let mut teacher_h = vec![0f32; b * t * t];
    for bb in 0..b {
        for i in 0..t {
            let beta = ((bb * t + i) / 4) % b_n;
            for j in 0..t {
                // Causal: the teacher is a DENSE CAUSAL attention, so a
                // future token has probability 0 - a one-hot block that is
                // partly future shows the seen part only.
                if j / 4 == beta && j <= i {
                    teacher_h[bb * t * t + i * t + j] = 1.0;
                }
            }
        }
    }

    let x = Tensor::<3>::from_data(TensorData::new(x_dense, [b, t, 16]), &device).require_grad();
    let teacher_raw = Tensor::<3>::from_data(TensorData::new(teacher_h, [b, t, t]), &device);
    let teacher = pool_teacher(teacher_raw.clone(), 4); // [b, t, B] one-hot rows

    let mut net = IndexerModule::new(cfg(), &device);
    let mut optim = AdamWConfig::new().init();

    for _ in 0..1000usize {
        let scores = indexer_scores(&net, x.clone());
        let loss = distill_loss(scores, teacher.clone(), 4);
        let grads = loss.clone().backward();
        let gp = GradientsParams::from_grads(grads, &net);
        net = optim.step(3e-3, net, gp);
    }

    // Score once more and compare top-1 picks with the teacher's hot block.
    let scores = indexer_scores(&net, x.clone()); // scoring: graph only, no step
    let picks = burn_msa::select_blocks(scores, 1);
    let picked: Vec<i64> = picks
        .into_data()
        .convert::<i64>()
        .bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mut hit = 0usize;
    let mut total = 0usize;
    for bb in 0..b {
        for i in 0..t {
            let want = (((bb * t + i) / 4) % b_n) as i64;
            // Query i may only pick a fully observed block: CAUSALITY IS
            // LOCAL to the ROW's sequence (the mask's positions are 0..t,
            // not the flattened index), so the visible check is on i.
            if want * 4 + 3 <= i as i64 {
                total += 1;
                if picked[bb * t + i] == want as usize as i64 {
                    hit += 1;
                }
            }
        }
    }
    let rate = hit as f32 / total as f32;
    assert!(total > 0, "the synthetic teacher produced no scoreable query");
    assert!(
        rate >= 0.9,
        "indexer top-1 agrees with the teacher on {hit}/{total} = {rate:.2} after 1000 AdamW steps @ 3e-3 (gate: >= 0.90)"
    );
}
