//! Manifold-Constrained Hyper-Connections block: Eq 3 residual mixing,
//! Eq 7 first-order hyper-network, Eq 8 constraints (sigmoid / 2*sigmoid /
//! Sinkhorn projection).

/// Gating factor init (App. A.1: 0.01).
pub const ALPHA_INIT: f64 = 0.01;
/// Static-mapping init magnitude: sigmoid(10) ~ 1, Sinkhorn(exp(diag 10)) ~ I.
pub(crate) const STATIC_INIT: f32 = 10.0;

use burn::module::{Module, Param};
use burn::nn::Initializer;
use burn::tensor::{activation, Device, Tensor};

use crate::sinkhorn::{sinkhorn_knopp, SINKHORN_ITERS};

/// Manifold-Constrained Hyper-Connections block.
///
/// The residual stream `h` of shape `[B, T, D]` is viewed as `n` streams of
/// width `C = D/n` (`n = n_branches`). A first-order hyper-network (Eq 7)
/// produces per-token, input-dependent mappings `H_pre [1 x n]`,
/// `H_post [1 x n]`, `H_res [n x n]`; `H_res` is projected onto the Birkhoff
/// polytope via Sinkhorn-Knopp (Eq 8-9). Branches are layer outputs `[B, T, D]`
/// (viewed as n per-stream outputs) scattered onto the stream with `H_post`.
///
/// At init (alpha = 0.01, b_pre = 10, b_post = 0, b_res = 10*I):
/// `H_pre ~ 1`, `H_post ~ 1`, `H_res ~ I` - the block starts as the standard
/// residual connection `h + sum(branches)`, restoring the identity mapping.
#[derive(Module, Debug)]
pub struct MhcBlock {
    /// Dynamic-mapping projections (Eq 7): `[D, n]` for pre/post, `[D, n^2]` for res.
    pub phi_pre: Param<Tensor<2>>,
    pub phi_post: Param<Tensor<2>>,
    pub phi_res: Param<Tensor<2>>,
    /// Learnable gating factors (Eq 7), init 0.01.
    pub alpha_pre: Param<Tensor<1>>,
    pub alpha_post: Param<Tensor<1>>,
    pub alpha_res: Param<Tensor<1>>,
    /// Static mappings (Eq 7): `b_pre/b_post [n]`, `b_res [n, n]`.
    pub b_pre: Param<Tensor<1>>,
    pub b_post: Param<Tensor<1>>,
    pub b_res: Param<Tensor<2>>,
    #[module(skip)]
    pub n_branches: usize,
}

impl MhcBlock {
    /// `n_branches`: residual-stream expansion rate `n` (paper: 4).
    /// `d_model`: hidden dim `D = n*C`.
    pub fn new(n_branches: usize, d_model: usize, device: &Device) -> Self {
        let n = n_branches.max(1);
        let normal = Initializer::Normal {
            mean: 0.0,
            std: 0.01,
        };
        let b_res = Tensor::<2>::eye(n, device).mul_scalar(STATIC_INIT);
        let b_pre = Tensor::<1>::ones([n], device).mul_scalar(STATIC_INIT);
        let b_post = Tensor::<1>::zeros([n], device);
        Self {
            phi_pre: normal.init([d_model, n], device),
            phi_post: normal.init([d_model, n], device),
            phi_res: normal.init([d_model, n * n], device),
            alpha_pre: Param::from_tensor(Tensor::ones([1], device).mul_scalar(ALPHA_INIT)),
            alpha_post: Param::from_tensor(Tensor::ones([1], device).mul_scalar(ALPHA_INIT)),
            alpha_res: Param::from_tensor(Tensor::ones([1], device).mul_scalar(ALPHA_INIT)),
            b_pre: Param::from_tensor(b_pre),
            b_post: Param::from_tensor(b_post),
            b_res: Param::from_tensor(b_res),
            n_branches: n,
        }
    }

    /// Eq 7: first-order hyper-network. Returns `(h_pre, h_post, h_res)`
    /// of shapes `[B,T,1,n]`, `[B,T,1,n]`, `[B,T,n,n]`, with constraints
    /// from Eq 8 already applied (sigmoid / 2*sigmoid / Sinkhorn).
    /// RMS-normalized stream `x_norm` shared by every mapping projection.
    fn x_norm(&self, h: &Tensor<3>) -> Tensor<3> {
        // x_vec' = RMSNorm(vec(x_l)) over the flattened nC dim (Eq 7).
        // mean_dim keeps a size-1 dim: [B,T,1] broadcasts against [B,T,D].
        h.clone()
            / h.clone()
                .powf_scalar(2.0)
                .mean_dim(2)
                .add_scalar(1e-20)
                .sqrt()
    }

    /// `H_post`/`H_res` only — the pair [`Self::forward`] consumes. Skips the
    /// `H_pre` projection+sigmoid, which Eq. 3 applies to the caller's
    /// external F, not inside this block (~1 matmul + 1 sigmoid per call).
    fn mappings_post_res(&self, h: &Tensor<3>) -> (Tensor<4>, Tensor<4>) {
        let n = self.n_branches;
        let [b, t, _d] = h.dims();
        let x_norm = self.x_norm(h);
        let post_tilde = self
            .alpha_post
            .val()
            .clone()
            .reshape([1, 1, 1])
            .mul(
                x_norm
                    .clone()
                    .matmul(self.phi_post.val().clone().unsqueeze_dim::<3>(0)),
            )
            .add(self.b_post.val().clone().reshape([1, 1, n]));
        let res_tilde = self
            .alpha_res
            .val()
            .clone()
            .reshape([1, 1, 1])
            .mul(x_norm.matmul(self.phi_res.val().clone().unsqueeze_dim::<3>(0)))
            .add(self.b_res.val().clone().reshape([1, 1, n * n]))
            .reshape([b, t, n, n]);
        let h_post = activation::sigmoid(post_tilde)
            .mul_scalar(2.0)
            .unsqueeze_dim::<4>(2);
        (h_post, sinkhorn_knopp(res_tilde, SINKHORN_ITERS))
    }

    pub fn hyper_mappings(&self, h: &Tensor<3>) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let n = self.n_branches;
        let [b, t, _d] = h.dims();

        let x_norm = self.x_norm(h);

        let pre_tilde = self
            .alpha_pre
            .val()
            .clone()
            .reshape([1, 1, 1])
            .mul(
                x_norm
                    .clone()
                    .matmul(self.phi_pre.val().clone().unsqueeze_dim::<3>(0)),
            )
            .add(self.b_pre.val().clone().reshape([1, 1, n]));
        let post_tilde = self
            .alpha_post
            .val()
            .clone()
            .reshape([1, 1, 1])
            .mul(
                x_norm
                    .clone()
                    .matmul(self.phi_post.val().clone().unsqueeze_dim::<3>(0)),
            )
            .add(self.b_post.val().clone().reshape([1, 1, n]));
        let res_tilde = self
            .alpha_res
            .val()
            .clone()
            .reshape([1, 1, 1])
            .mul(x_norm.matmul(self.phi_res.val().clone().unsqueeze_dim::<3>(0)))
            .add(self.b_res.val().clone().reshape([1, 1, n * n]))
            .reshape([b, t, n, n]);

        let h_pre = activation::sigmoid(pre_tilde).unsqueeze_dim::<4>(2);
        let h_post = activation::sigmoid(post_tilde)
            .mul_scalar(2.0)
            .unsqueeze_dim::<4>(2);
        let h_res = sinkhorn_knopp(res_tilde, SINKHORN_ITERS);
        (h_pre, h_post, h_res)
    }

    /// Eq 3: `x_{l+1} = H_res x_l + (H_post)^T F(...)`.
    ///
    /// `h`: residual stream `[B, T, D]` (viewed as n streams of width D/n).
    /// `branches`: layer outputs `[B, T, D]`, each viewed as n per-stream
    /// outputs; stream `j` receives weight `H_post_j`.
    pub fn forward(&self, h: Tensor<3>, branches: &[Tensor<3>]) -> Tensor<3> {
        let [b, t, d] = h.dims();
        let n = self.n_branches;
        assert!(
            d % n == 0,
            "d_model {d} must be divisible by n_branches {n}"
        );
        let c = d / n;

        let (h_post, h_res) = self.mappings_post_res(&h);
        // n-stream residual
        let x_l = h.reshape([b, t, n, c]);

        // H_res x_l: stream mixing (Eq 3 residual term). The Eq 3 layer input
        // H_pre x_l is computed by the caller's external F, not here.
        let res = h_res.matmul(x_l);

        // (H_post)^T F: weight each branch's stream outputs with H_post
        let mut out = res;
        let h_post_t = h_post.swap_dims(2, 3); // [B, T, n, 1]
        for branch in branches {
            let branch_streams = branch.clone().reshape([b, t, n, c]);
            out = out + h_post_t.clone() * branch_streams;
        }
        out.reshape([b, t, d])
    }

    /// Branch-only path (no residual stream): `sum_i (H_post^T ⊙ F_i)`.
    pub fn forward_no_residual(&self, branches: &[Tensor<3>]) -> Tensor<3> {
        assert!(
            !branches.is_empty(),
            "forward_no_residual needs >= 1 branch"
        );
        let [b, t, d] = branches[0].dims();
        let n = self.n_branches;
        let c = d / n;
        let h = Tensor::<3>::zeros([b, t, d], &branches[0].device());
        let (h_post, _) = self.mappings_post_res(&h);
        let h_post_t = h_post.swap_dims(2, 3); // [B, T, n, 1]
        let mut out = Tensor::<4>::zeros([b, t, n, c], &branches[0].device());
        for branch in branches {
            out = out + h_post_t.clone() * branch.clone().reshape([b, t, n, c]);
        }
        out.reshape([b, t, d])
    }
}
