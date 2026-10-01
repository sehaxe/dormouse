use burn::tensor::Tensor;

/// L2-normalize along the last dimension.
pub fn l2_normalize(x: Tensor<3>, eps: f64) -> Tensor<3> {
    let norm = x.clone().powf_scalar(2.0).sum_dim(2).add_scalar(eps).sqrt();
    x.div(norm)
}

/// L2-normalize along the last dimension of a 4D tensor.
///
/// The reference implementation normalizes q/k per head (each head's own
/// `head_dim` channels), so normalization must happen after the head split.
pub fn l2_normalize_4d(x: Tensor<4>, eps: f64) -> Tensor<4> {
    let norm = x.clone().powf_scalar(2.0).sum_dim(3).add_scalar(eps).sqrt();
    x.div(norm)
}
