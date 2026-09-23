use burn::backend::Backend;
use burn::tensor::{Int, Tensor};

pub struct KlAlignmentLoss;

impl KlAlignmentLoss {
    pub fn compute<B: Backend>(
        &self,
        block_scores: Tensor<4>,
        attn_weights: Tensor<4>,
        selected_indices: Tensor<4, Int>,
    ) -> Tensor<1> {
        let gathered = block_scores.gather(3, selected_indices);
        let smax = gathered.clone().max_dim(3);
        let shifted = gathered.sub(smax);
        let log_p = shifted
            .clone()
            .sub(shifted.exp().sum_dim(3).log().add_scalar(1e-10));

        let p_teacher = attn_weights.detach();
        let p_target = p_teacher
            .clone()
            .div(p_teacher.sum_dim(3).add_scalar(1e-10));

        p_target.mul(log_p).neg().mean()
    }
}
