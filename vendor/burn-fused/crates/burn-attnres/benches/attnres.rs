//! Manual micro-benchmark: `depth_attend` (fused CUDA path when the `cuda`
//! feature is on) vs the 8-pass tensor path, both sync'd.
//!
//! Run: `cargo bench -p burn-attnres --features cuda` (needs a GPU; the README
//! numbers are measured on an RTX 3090). On a CPU-only build the fused path is
//! absent and the ratio is ~1x - the harness itself is still validated.

use burn::tensor::{activation, Distribution, Tensor};
use burn_attnres::{depth_attend, ScoreForm};

/// The naive 8-pass reference the fused path replaces (same math as
/// `depth_attend`'s tensor fallback, in the DEFAULT score form - Eq. 2's
/// unscaled `q . RMSNorm(h)`, so the ratio measures the launch count and not
/// a difference of formulas).
fn ref_depth_attend(history: &[Tensor<3>], query: &Tensor<1>) -> Tensor<3> {
    let n = history.len();
    let [b, t, d] = history[0].dims();
    let form = ScoreForm::default();
    let scale = form.scale(d);
    let stacked: Vec<Tensor<4>> = history
        .iter()
        .map(|h| h.clone().unsqueeze_dim::<4>(0))
        .collect();
    let h_stack = Tensor::cat(stacked, 0);
    let h_norm_sq = h_stack
        .clone()
        .powf_scalar(2.0)
        .sum_dim(3)
        .mul_scalar(form.norm_m(d))
        .add_scalar(1e-5);
    let h_norm = h_stack.clone() / h_norm_sq.sqrt().reshape([n, b, t, 1usize]);
    let q = query.clone().reshape([1, 1, 1, d]);
    let scores = (q * h_norm).sum_dim(3).mul_scalar(scale);
    let weights = activation::softmax(scores, 0);
    h_stack
        .mul(weights.reshape([n, b, t, 1usize]))
        .sum_dim(0)
        .reshape([b, t, d])
}

fn main() {
    let dev = burn::tensor::Device::default();
    // README claim configs (forward table).
    for (l, b, t, d) in [(24usize, 1usize, 2048usize, 4096usize), (8, 2, 2048, 5120)] {
        let hist: Vec<Tensor<3>> = (0..l)
            .map(|_| Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &dev))
            .collect();
        let q = Tensor::<1>::random([d], Distribution::Normal(0.0, 1.0), &dev);
        for _ in 0..3 {
            let r = depth_attend(&hist, q.clone());
            let _: f32 = r.sum().into_scalar(); // sync: fused path is async
        }
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            let r = depth_attend(&hist, q.clone());
            let _: f32 = r.sum().into_scalar();
        }
        let tf = t0.elapsed() / 10;
        let t0 = std::time::Instant::now();
        for _ in 0..5 {
            let r = ref_depth_attend(&hist, &q);
            let _: f32 = r.sum().into_scalar();
        }
        let tt = t0.elapsed() / 5;
        println!(
            "[L{l} b{b} t{t} d{d}] depth_attend {:?} vs 8-pass tensor {:?} ({:.1}x)",
            tf,
            tt,
            tt.as_secs_f64() / tf.as_secs_f64()
        );
    }
}
