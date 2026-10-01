//! Scalar EMA teacher-weight holder (data2vec 2.0).

use burn::module::Module;
use burn::tensor::{Device, Tensor};

/// Scalar EMA primitive for a teacher target.
///
/// `theta = m * theta + (1 - m) * student` (data2vec 2.0, Baevski ICML 2023).
/// The full teacher ENCODER is the caller's model; this crate provides the
/// EMA weight holder. The stored value is detached (never gradients).
#[derive(Module, Debug)]
pub struct EmaTarget {
    pub weight: burn::module::Param<Tensor<1>>,
    #[module(skip)]
    pub momentum: f64,
}

impl EmaTarget {
    pub fn new(momentum: f64, device: &Device) -> Self {
        Self {
            weight: burn::module::Param::from_tensor(Tensor::zeros([1], device)),
            momentum,
        }
    }

    /// `theta = m * theta + (1 - m) * student`
    pub fn update(&mut self, student: Tensor<1>) {
        let m = self.momentum as f32;
        let new = self.weight.val().clone().mul_scalar(m) + student.mul_scalar(1.0 - m);
        self.weight = burn::module::Param::from_tensor(new);
    }

    /// Current EMA value (detached).
    pub fn val(&self) -> Tensor<1> {
        self.weight.val().clone()
    }
}
