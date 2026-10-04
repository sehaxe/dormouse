//! plain_attn - the DENSE CAUSAL MULTI-HEAD SOFTMAX arm (`use_plain_attn`),
//! and the CONTROL of the controlled trio (plain byte Transformer vs byteflow
//! vs dormouse, one recipe, equal bytes).
//!
//! # WHY THIS IS A NEW ARM AND NOT A CONFIG
//!
//! Before it, this repository had no plain-attention forward to point a control
//! at. `use_kda = false` gives a network with NO attention arm (a gated-FFN
//! stack), and the only full-softmax arm was the Qwen sparse one, which
//! `config::validate` REFUSES at `msa_kb >= the block count` with the words
//! "a sparse arm over every block IS the dense arm at a top-k's price" — the
//! repo's own doctrine saying the dense control cannot be spelled as a sparse
//! config. So the control arm is four projections and one call to the tested
//! dense kernel, and it lives INSIDE the loop: the trio's whole point is one
//! recipe, so the control has to run the same `train_loop`, the same optimizer
//! routing, the same NaN firewall and the same eval as the two arms it is
//! compared against.
//!
//! # WHAT THE ARM IS
//!
//! Textbook decoder-only attention: `softmax(q·kᵀ/√d_head + causal mask)·v`,
//! four dense projections, RoPE on q and k (Su et al. 2021, base 10000), no
//! state, no recurrence, no top-k, no indexer. The kernel is
//! `burn_msa::dense_attention` — the same function burn-msa's stage-(a) teacher
//! runs, reused rather than reimplemented (it carries the `1/√d_head` scale and
//! the `tril` mask, and it is the arm the library's own dense-contract gate
//! pins). It also returns the head-summed attention distribution `[b, t, t]`,
//! which this arm DISCARDS: an allocation the control pays that the recurrent
//! arms do not, recorded here so nobody reads it as a free win for the control.
//!
//! # WHAT THE ARM IS NOT (read this before quoting a trio number)
//!
//! It is not a GPT-2. The loop's fixed scaffolding survives with every dormouse
//! mechanism switched off, because switching them off is a config field and
//! deleting the scaffolding is not:
//!
//! * the controller's sigmoid `w_attn` still multiplies the attention output;
//! * ReZero's scalar (init 1) still scales the residual write;
//! * `out_proj` (`d -> d`) is still applied to the readout;
//! * `iter_embed` (zeros at init) is still added per iteration slot;
//! * with `max_iter = 1` the "loop" is one pass, so the control is a ONE-LAYER
//!   transformer. Depth > 1 in this model IS the loop under test — weight-shared
//!   — so a multi-layer plain control is not reachable without a second unshared
//!   stack, which is follow-up work, not a config.
//!
//! All four scaffolding pieces are identity or near-identity at init, so the
//! arm is a fair "no mechanisms" control for the trio, but it is not a claim
//! about how a textbook 9M transformer trains. That claim needs the standalone
//! net (a third `train_loop`), which is the honest cost of the comparison this
//! arm does NOT make.
use burn::backend::AutodiffBackend;
use burn::module::Module;
use burn::tensor::{Device, Tensor};

use crate::config::DormouseConfig;
use crate::param::LinearLike;

/// The dense causal softmax attention arm. `None` on [`crate::LoopBlock`]
/// unless `use_plain_attn` — `Option`-shaped like `msa`, so the parameters do
/// not exist when the arm is off and no existing checkpoint is affected.
#[derive(Module, Debug)]
pub struct PlainAttention {
    /// `d_model -> n_heads * head_dim`, no bias (`LinearLike::dense` keeps the
    /// checkpoint path uniform with every other projection in the model).
    pub wq: LinearLike,
    /// The key projection, rotated by the same RoPE as `wq` at the query's own
    /// position - self-consistency inside one forward is what makes the
    /// relative-position claim true.
    pub wk: LinearLike,
    /// The value projection, NOT rotated: RoPE moves content by position, and
    /// no reference rotates v.
    pub wv: LinearLike,
    /// The output projection, `n_heads * head_dim -> d_model`.
    pub wo: LinearLike,
    /// RoPE tables `[max_seq_len, head_dim/2]` (cos, sin). BUFFERS, not
    /// parameters: a position encoding is a function of
    /// `(head_dim, max_seq_len, base)`, and a buffer is not counted by
    /// `num_params`, so the control's parameter count stays a count of
    /// LEARNED weights.
    pub rope_cos: Tensor<2>,
    /// `sin` table, the same shape as `rope_cos`.
    pub rope_sin: Tensor<2>,
    /// Head count, copied from the config so the forward does not have to be
    /// handed it.
    #[module(skip)]
    pub n_heads: usize,
    /// Per-head width, same reason as `n_heads`.
    #[module(skip)]
    pub head_dim: usize,
}

impl PlainAttention {
    /// Build the arm at the model's own `(d_model, n_heads, head_dim)`.
    ///
    /// RoPE base is 10 000, the value every transformer in this space uses and
    /// the one `burn_rope::precompute_freqs` documents; it is NOT a config
    /// field, because the trio's whole claim is that the recipe is one recipe
    /// and a position-encoding base nobody A/B'd is not a knob this repo needs.
    pub fn new(cfg: &DormouseConfig, device: &Device) -> Self {
        let d = cfg.d_model;
        let (h, hd) = (cfg.n_heads, cfg.head_dim);
        assert!(
            h * hd == d,
            "n_heads {h} * head_dim {hd} = {} != d_model {d}: the arm's projections are built \
             at n_heads * head_dim, so the mismatch has to be named here rather than as a shape \
             error at the first forward",
            h * hd
        );
        assert!(
            cfg.max_seq_len > 0 && d > 0,
            "max_seq_len {} and d_model {d} must be > 0 (the RoPE table is sized by max_seq_len)",
            cfg.max_seq_len
        );
        let (rope_cos, rope_sin) = burn_rope::precompute_freqs(hd, cfg.max_seq_len, 10000.0, device);
        Self {
            wq: LinearLike::dense(d, h * hd, device),
            wk: LinearLike::dense(d, h * hd, device),
            wv: LinearLike::dense(d, h * hd, device),
            wo: LinearLike::dense(h * hd, d, device),
            rope_cos,
            rope_sin,
            n_heads: h,
            head_dim: hd,
        }
    }

    /// `x [b, t, d_model] -> [b, t, d_model]` — the block body's attention
    /// input to the attention output, before the controller's `w_attn` and the
    /// residual write.
    ///
    /// RoPE goes on q and k ONLY, which is the standard formulation: rotating v
    /// would move the value content by position, and no reference does it.
    pub fn forward<B: AutodiffBackend>(&self, x: Tensor<3>) -> Tensor<3>
    where
        burn::tensor::DispatchTensor:
            burn::backend::DispatchKindConversion<B>
            + burn::backend::DispatchKindConversion<B::InnerBackend>
            + burn::backend::DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        let [b, t, d] = x.dims();
        assert_eq!(
            d,
            self.n_heads * self.head_dim,
            "the block body handed the dense arm a width {d} it was not built for (n_heads {} * \
             head_dim {})",
            self.n_heads,
            self.head_dim
        );
        assert!(
            t <= self.rope_cos.dims()[0],
            "a {t}-position sequence needs {t} RoPE rows and the table has {}: raise max_seq_len \
             (the tables are sized once, at construction)",
            self.rope_cos.dims()[0]
        );
        let (h, hd) = (self.n_heads, self.head_dim);
        let flat = x.reshape([b * t, d]);
        let head = |l: &LinearLike| l.forward::<B>(flat.clone()).reshape([b, t, h, hd]);
        let (q, k, v) = (head(&self.wq), head(&self.wk), head(&self.wv));
        // RoPE under the BARE inner backend: a rotation is elementwise, and
        // routing it through the autodiff node is what makes burn-rope's own
        // fused-rope probe meaningful (it counts `asked/fused/skipped`).
        let rope = |x: Tensor<4>| {
            burn_rope::apply_rope_4d::<B::InnerBackend>(x, self.rope_cos.clone(), self.rope_sin.clone())
        };
        let (q, k) = (rope(q), rope(k));
        // The library's tested dense causal kernel (scale `hd^-0.5`, `tril`
        // mask, one softmax). `_dist` is the head-summed distribution burn-msa
        // needs for its distillation KL and this arm has no use for: one
        // `[b, t, t]` allocation per forward, discarded. Named so a reader who
        // profiles the control knows what the extra traffic is.
        let (mixed, _dist) = burn_msa::dense_attention(q, k, v);
        self.wo
            .forward::<B>(mixed.reshape([b * t, d]))
            .reshape([b, t, d])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DormouseConfig;
    use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
    use burn::backend::autodiff::Autodiff;
    use burn::tensor::{Device, Tensor};

    /// The CPU backend this repo's tests run on (burn-flex; `burn-ndarray` is
    /// deprecated upstream), with the trainer's checkpointing strategy so the
    /// graph under test is the graph the trainer builds.
    #[allow(deprecated)]
    type B = Autodiff<burn::backend::Flex, BalancedCheckpointing>;

    #[allow(deprecated)]
    fn device() -> Device {
        Device::flex().autodiff()
    }

    fn cfg() -> DormouseConfig {
        DormouseConfig {
            d_model: 32,
            n_heads: 4,
            head_dim: 8,
            max_seq_len: 16,
            ..Default::default()
        }
    }

    /// The dense `burn::nn::Linear` inside a `LinearLike`, so a test can read
    /// its weight gradient by hand.
    fn dense(l: &LinearLike) -> &burn::nn::Linear {
        let crate::param::LinearLikeInner::Dense(lin) = &l.inner else {
            panic!("the control builds the dense variant (use_tsct = false)");
        };
        lin
    }

    /// L2 of a gradient tensor of any rank.
    fn l2<const D: usize>(t: &Tensor<D>) -> f64 {
        t.clone()
            .into_data()
            .try_to_vec::<f32>()
            .expect("readable")
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn maxdiff(a: &Tensor<3>, b: &Tensor<3>) -> f32 {
        a.clone().sub(b.clone()).abs().max().into_scalar::<f32>()
    }

    /// The arm's four projections and NOTHING else: 4 * (d * h * hd + h * hd),
    /// biases included because `LinearLike::dense` is a `burn::nn::Linear`.
    /// A count that moves when the arm's definition moves is the cheapest
    /// statement that "the control has no mechanisms left in it".
    #[test]
    fn plain_attn_is_four_projections_and_rope_tables() {
        let dev = device();
        let c = cfg();
        let attn = PlainAttention::new(&c, &dev);
        // weight `[in, out]` + bias `[out]`, four times: `LinearLike::dense` is
        // a `burn::nn::Linear`, which keeps its bias.
        let proj = |i: usize, o: usize| i * o + o;
        assert_eq!(
            attn.num_params(),
            3 * proj(c.d_model, c.n_heads * c.head_dim) + proj(c.n_heads * c.head_dim, c.d_model),
            "four dense projections, biases included, and nothing else"
        );
        // The RoPE tables are BUFFERS: a control whose "parameters" include a
        // position table is not a parameter count.
        assert_eq!(
            attn.rope_cos.dims(),
            [c.max_seq_len, c.head_dim / 2],
            "the RoPE table is [max_seq_len, head_dim/2]"
        );
    }

    /// Causality is the whole content of "attention": position t must not see
    /// t+1. Pinned by changing a LATER byte's input and reading position 0.
    #[test]
    fn plain_attn_is_causal() {
        let dev = device();
        let c = cfg();
        let attn = PlainAttention::new(&c, &dev);
        let x: Vec<f32> = (0..(c.d_model * 12)).map(|i| (i as f32) * 0.01).collect();
        let mut x2 = x.clone();
        for v in x2.iter_mut().skip(6 * c.d_model) {
            *v += 3.0;
        }
        let a = Tensor::<1>::from_floats(x.as_slice(), &dev).reshape([1, 12, c.d_model]);
        let b = Tensor::<1>::from_floats(x2.as_slice(), &dev).reshape([1, 12, c.d_model]);
        let (oa, ob) = (attn.forward::<B>(a), attn.forward::<B>(b));
        // Position 0 attends to itself alone: identical. Position 6 read the
        // byte we changed, so its output MUST move - a zero there would mean the
        // arm is inert and the causality check above is vacuous.
        let at = |o: &Tensor<3>, p: usize| o.clone().slice([0..1, p..p + 1, 0..c.d_model]);
        let d0 = maxdiff(&at(&oa, 0), &at(&ob, 0));
        let d6 = maxdiff(&at(&oa, 6), &at(&ob, 6));
        assert!(
            d0 < 1e-6,
            "position 0 moved by {d0:e} when only LATER inputs changed - the arm is not causal"
        );
        assert!(
            d6 > 1e-3,
            "position 6 moved by {d6:e} when its own input changed - the arm is inert, so the \
             causality assertion above proves nothing"
        );
    }

    /// Every projection takes a gradient. The `8fa5d4c` gate: an attention arm
    /// that runs and returns a leaf looks perfectly healthy in a loss curve.
    /// Named per projection, never in aggregate - "some gradient is non-zero"
    /// is the assertion that let the KDA arm train nothing for this project's
    /// whole history.
    #[test]
    fn plain_attn_takes_gradient_on_every_projection() {
        let dev = device();
        let c = cfg();
        let attn = PlainAttention::new(&c, &dev);
        let xs: Vec<f32> = (0..(c.d_model * 12)).map(|i| (i as f32) * 0.05 - 0.3).collect();
        let x = Tensor::<1>::from_floats(xs.as_slice(), &dev).reshape([1, 12, c.d_model]);
        let grads = attn.forward::<B>(x).sum().backward();
        for (name, lin) in [
            ("wq", dense(&attn.wq)),
            ("wk", dense(&attn.wk)),
            ("wv", dense(&attn.wv)),
            ("wo", dense(&attn.wo)),
        ] {
            let g = lin
                .weight
                .grad(&grads)
                .unwrap_or_else(|| panic!("no gradient at all for {name}.weight"));
            let n = l2(&g);
            assert!(
                n > 1e-12,
                "{name}.weight received an exactly-zero gradient (L2 {n:.3e})"
            );
            let gb = l2(
                &lin.bias
                    .as_ref()
                    .expect("LinearLike::dense keeps the bias")
                    .grad(&grads)
                    .unwrap_or_else(|| panic!("no gradient for {name}.bias")),
            );
            assert!(gb > 1e-12, "{name}.bias received an exactly-zero gradient");
        }
    }
}
