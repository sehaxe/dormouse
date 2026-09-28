// The attention arm's own cost, both arms, on the trainer's shape. Run:
//   cargo test -p burn-gdn2 --features cuda,autodiff --test ops_batched_bench_cuda --release -- --nocapture
//
// ## What this measures and what it does not
//
// ONE chunked-WY forward plus its backward, at the shape dormouse's trainer
// runs: batch 8, 12 heads, T=512, head_dim 64, chunk 16, on
// `Autodiff<CudaBare, BalancedCheckpointing>` — the trainer's backend and
// checkpointing strategy. That is the op the arm is made of; the module's
// projections around it are a handful of [768,768] GEMMs and are not counted
// here, so this is the ARM's cost, not the step's.
//
// The loss touches both the output and the output state, so a backward that
// only reached one of them would show up as a suspiciously cheap arm.
//
// The trainer's end-to-end numbers are a separate measurement (a 25-step run
// with and without `--no-kda`); this file exists so the delta can be attributed
// to the op rather than inferred from a step time.
//
// Release is not optional here: a debug build's op dispatch dominates the very
// thing being measured.
//
// MEASURED 2026-09-28, commit e11c0d5's parent tree, this box (5060 Ti, fp32):
//
//   Loop:    284.7 ms fwd+bwd   14389 tokens/s
//   Batched:  83.0 ms fwd+bwd   49373 tokens/s      3.43x
//
// And in the trainer itself (batch 8, seq 512, depth 2, `--no-engram --quant
// fp32 --jepa-weight 0 --dspark-weight 0`, 25 steps, per-step times read from
// the step lines, MEDIAN because every run has one ~1 s outlier step that a mean
// would charge to the arm):
//
//   --no-kda control   175 ms/step   23365 tok/s
//   batched (default)  286 ms/step   14300 tok/s   arm = 111 ms/step
//   loop  (DM_GDN2_OPS=loop) 663 ms/step  6180 tok/s  arm = 487 ms/step
//
// so the ARM is 4.4x cheaper and the whole step 2.31x faster. The op-level
// ratio (3.43x) is the LOWER bound: this bench's loss reduction reads 7
// gradients back to the host per iteration, which costs the cheap arm relatively
// more, and the trainer's graph saves the loop arm ~3000 extra nodes per
// iteration for the checkpointer to track on top of the op's own time.

#![cfg(all(feature = "cuda", feature = "autodiff"))]

use burn::backend::AutodiffBackend;
use burn::tensor::{Device, Distribution, Tensor};
use burn_autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn_autodiff::Autodiff;
use burn_gdn2::{ChunkPath, CudaBare};

type AdBal = Autodiff<CudaBare, BalancedCheckpointing>;

/// The trainer's shape (configs/small.toml: d_model 768, n_heads 12,
/// head_dim 64; attention.rs: chunk_size 16).
const B: usize = 8;
const H: usize = 12;
const T: usize = 512;
const K: usize = 64;
const CHUNK: usize = 16;
/// Iterations timed, after the warmup. Small on purpose: the GPU is shared and
/// the loop arm is seconds per iteration.
const REPS: usize = 3;

fn inputs(dev: &Device) -> [Tensor<4>; 7] {
    let lift = |t: Tensor<4>| -> Tensor<4> {
        let node = <AdBal as AutodiffBackend>::from_inner(
            t.try_into_primitive::<CudaBare>()
                .expect("bare cuda tensor"),
        );
        Tensor::from_primitive::<AdBal>(node).require_grad()
    };
    let kraw = Tensor::<4>::random([B, H, T, K], Distribution::Normal(0.0, 1.0), dev);
    let k = kraw.clone() / kraw.powf_scalar(2.0).sum_dim(3).sqrt();
    [
        lift(Tensor::<4>::random(
            [B, H, T, K],
            Distribution::Normal(0.0, 1.0),
            dev,
        )),
        lift(k),
        lift(Tensor::<4>::random(
            [B, H, T, K],
            Distribution::Normal(0.0, 1.0),
            dev,
        )),
        lift(Tensor::<4>::random(
            [B, H, T, K],
            Distribution::Uniform(-5.0, -0.01),
            dev,
        )),
        lift(Tensor::<4>::random(
            [B, H, T, K],
            Distribution::Uniform(0.0, 1.0),
            dev,
        )),
        lift(Tensor::<4>::random(
            [B, H, T, K],
            Distribution::Uniform(0.5, 1.0),
            dev,
        )),
        lift(Tensor::<4>::random(
            [B, H, K, K],
            Distribution::Normal(0.0, 0.5),
            dev,
        )),
    ]
}

fn step(inp: &[Tensor<4>; 7]) -> f32 {
    let (o, s) = burn_gdn2::chunk_wy_forward(
        inp[0].clone(),
        inp[1].clone(),
        inp[2].clone(),
        inp[3].clone(),
        inp[4].clone(),
        inp[5].clone(),
        inp[6].clone(),
        (K as f64).powf(-0.5),
        CHUNK,
    );
    let l = o.powf_scalar(2.0).sum().add(s.powf_scalar(2.0).sum());
    let g = l.backward();
    // force the backward to be real work, not a lazily-discarded graph
    let mut acc = 0.0f32;
    for t in inp.iter() {
        if let Some(gr) = t.grad(&g) {
            acc += gr.clone().into_data().bytes.len() as f32;
        }
    }
    acc
}

#[test]
fn the_batched_arm_costs_less_of_the_step() {
    let dev = Device::cuda(0);
    dev.seed(1);
    let inp = inputs(&dev);
    let mut ms = [0.0f64; 2];
    for (i, path) in [ChunkPath::Batched, ChunkPath::Loop]
        .into_iter()
        .enumerate()
    {
        burn_gdn2::set_chunk_path(path);
        // warmup: the first call pays the pool's first-touch and the autotuner
        let warm = step(&inp);
        assert!(
            warm > 0.0,
            "the backward produced no gradient bytes: {path:?} is frozen"
        );
        let t0 = std::time::Instant::now();
        for _ in 0..REPS {
            step(&inp);
        }
        let el = t0.elapsed().as_secs_f64() * 1e3 / REPS as f64;
        ms[i] = el;
        let tok_s = (B * T) as f64 / (el / 1e3);
        println!(
            "{path:?}: {el:8.1} ms fwd+bwd   {tok_s:8.0} tokens/s   (mean over {REPS} after a \
             warmup)"
        );
    }
    burn_gdn2::set_chunk_path(ChunkPath::Batched);
    println!(
        "batched/loop = {:.3}  (loop is {:.2}x the batched arm)",
        ms[0] / ms[1],
        ms[1] / ms[0]
    );
    assert!(
        ms[0] < ms[1],
        "the batched arm is not faster: {:.1} ms vs {:.1} ms",
        ms[0],
        ms[1]
    );
}
