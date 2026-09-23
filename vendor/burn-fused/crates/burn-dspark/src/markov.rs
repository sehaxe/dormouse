//! Sequential heads for DSpark (DeepSpec `markov_head.py`).
//!
//! All heads produce a logit bias `B_k(x0, x<k, xk)` added to the parallel
//! backbone logits (paper Eq 4). The default is `VanillaMarkov` (Eq 5);
//! `GatedMarkovHead` and `RNNHead` (Eq 6) are the DeepSpec alternatives.
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::{activation, Device, Int, Tensor};

use crate::sampling::sample_tokens;

/// Low-rank first-order transition bias (paper Eq 5, DeepSpec
/// `VanillaMarkov`):
///
/// ```text
/// prev_id -> Embedding(vocab, r) = W1[x] -> Linear(r, vocab) = W2 -> bias
/// conditioned_logits = base_logits + bias
/// ```
#[derive(Module, Debug)]
pub struct VanillaMarkov {
    w1: Embedding,
    w2: Linear,
    #[module(skip)]
    pub rank: usize,
    #[module(skip)]
    pub vocab: usize,
}

impl VanillaMarkov {
    pub fn new(vocab_size: usize, rank: usize, device: &Device) -> Self {
        Self {
            w1: EmbeddingConfig::new(vocab_size, rank).init(device),
            w2: LinearConfig::new(rank, vocab_size)
                .with_bias(false)
                .init(device),
            rank,
            vocab: vocab_size,
        }
    }

    /// `W1[x]`: `[B, L, r]` Markov embeddings of the previous tokens.
    pub fn get_prev_embeddings(&self, token_ids: Tensor<2, Int>) -> Tensor<3> {
        self.w1.forward(token_ids)
    }

    /// `W2(latent)`: `[B, L, vocab]` logit projection.
    pub fn project_bias(&self, latent_states: Tensor<3>) -> Tensor<3> {
        self.w2.forward(latent_states)
    }

    /// Bias for the given previous-token ids: `[B, L] -> [B, L, vocab]`.
    pub fn compute_step_bias(&self, token_ids: Tensor<2, Int>) -> Tensor<3> {
        self.project_bias(self.get_prev_embeddings(token_ids))
    }

    /// Apply the bias to base logits (teacher-forced, one step).
    pub fn apply_step_logits(&self, logits: Tensor<3>, token_ids: Tensor<2, Int>) -> Tensor<3> {
        logits + self.compute_step_bias(token_ids)
    }

    /// Apply the bias to a whole block: `[B, N, L, V] + bias(prev) [B, N, L, V]`.
    pub fn apply_block_logits(
        &self,
        base_logits: Tensor<4>,
        prev_ids: Tensor<3, Int>,
    ) -> Tensor<4> {
        let [b, n, l, v] = base_logits.dims();
        let bias = self
            .compute_step_bias(prev_ids.reshape([b * n, l]))
            .reshape([b, n, l, v]);
        base_logits + bias
    }

    /// Autoregressive draft sampling within a block: each position
    /// conditions on the previously sampled token (DeepSpec
    /// `sample_block_tokens`, temperature 0 = argmax).
    ///
    /// `base_logits`: `[B, block_size, vocab]`
    /// `first_prev_token_ids`: `[B]` - token before the first draft position.
    ///
    /// Returns sampled ids `[B, block_size]` and corrected logits
    /// `[B, block_size, vocab]`.
    pub fn sample_block_tokens(
        &self,
        base_logits: Tensor<3>,
        first_prev_token_ids: Tensor<2, Int>,
        temperature: f32,
    ) -> (Tensor<2, Int>, Tensor<3>) {
        let [b, l, v] = base_logits.dims();
        let mut prev_ids = first_prev_token_ids; // [B, 1]
        let mut sampled: Vec<Tensor<2, Int>> = Vec::with_capacity(l);
        let mut corrected: Vec<Tensor<3>> = Vec::with_capacity(l);
        for k in 0..l {
            let step_logits = base_logits.clone().slice([0..b, k..k + 1, 0..v]); // [B, 1, V]
            let step_logits = self.apply_step_logits(step_logits, prev_ids.clone());
            corrected.push(step_logits.clone());
            let next = sample_tokens(step_logits, temperature); // [B, 1]
            sampled.push(next.clone());
            prev_ids = next;
        }
        (Tensor::cat(sampled, 1), Tensor::cat(corrected, 1))
    }
}

/// Gated Markov head (DeepSpec `GatedMarkovHead`): gates the previous-token
/// embedding with a sigmoid over `[h_k; W1[x_{k-1}]]` before projection.
#[derive(Module, Debug)]
pub struct GatedMarkovHead {
    inner: VanillaMarkov,
    gate_proj: Linear,
    #[module(skip)]
    pub hidden_size: usize,
}

impl GatedMarkovHead {
    pub fn new(vocab_size: usize, rank: usize, hidden_size: usize, device: &Device) -> Self {
        Self {
            inner: VanillaMarkov::new(vocab_size, rank, device),
            gate_proj: LinearConfig::new(hidden_size + rank, rank)
                .with_bias(false)
                .init(device),
            hidden_size,
        }
    }

    /// Gate `[B, L, r]` from hidden states and previous-token embeddings.
    pub fn compute_gate(&self, hidden_states: Tensor<3>, prev_embeddings: Tensor<3>) -> Tensor<3> {
        let gate_inputs = Tensor::cat(vec![hidden_states, prev_embeddings], 2);
        activation::sigmoid(self.gate_proj.forward(gate_inputs))
    }

    pub fn compute_step_bias(
        &self,
        token_ids: Tensor<2, Int>,
        hidden_states: Tensor<3>,
    ) -> Tensor<3> {
        let prev_embeddings = self.inner.get_prev_embeddings(token_ids);
        let gate = self.compute_gate(hidden_states, prev_embeddings.clone());
        self.inner.project_bias(gate * prev_embeddings)
    }

    pub fn apply_step_logits(
        &self,
        logits: Tensor<3>,
        token_ids: Tensor<2, Int>,
        hidden_states: Tensor<3>,
    ) -> Tensor<3> {
        logits + self.compute_step_bias(token_ids, hidden_states)
    }
}

/// GRU-like recurrent head (paper Eq 6, DeepSpec `RNNHead`): maintains a
/// state `s_k` across block positions; the joint projection
/// `[s_{k-1}; W1[x_{k-1}]; h_k] -> [gate; candidate; output]` is a single
/// linear layer split into three.
#[derive(Module, Debug)]
pub struct RNNHead {
    inner: VanillaMarkov,
    joint_proj: Linear,
    #[module(skip)]
    pub state_size: usize,
    #[module(skip)]
    pub hidden_size: usize,
}

impl RNNHead {
    pub fn new(vocab_size: usize, rank: usize, hidden_size: usize, device: &Device) -> Self {
        Self {
            inner: VanillaMarkov::new(vocab_size, rank, device),
            joint_proj: LinearConfig::new(2 * rank + hidden_size, 3 * rank)
                .with_bias(false)
                .init(device),
            state_size: rank,
            hidden_size,
        }
    }

    /// One RNN step: `(state, prev_embeddings, h_k) -> (new_state, bias)`.
    pub fn rnn_step(
        &self,
        state: Tensor<2>,
        prev_embeddings: Tensor<2>,
        hidden_states: Tensor<2>,
    ) -> (Tensor<2>, Tensor<2>) {
        let z = Tensor::cat(vec![state.clone(), prev_embeddings, hidden_states], 1); // [B, 2r+d]
        let proj = self.joint_proj.forward(z); // [B, 3r]
        let bsz = proj.dims()[0];
        let r = self.state_size;
        let gate_raw = proj.clone().slice([0..bsz, 0..r]);
        let candidate_raw = proj.clone().slice([0..bsz, r..2 * r]);
        let output_raw = proj.slice([0..bsz, 2 * r..3 * r]);
        let gate = activation::sigmoid(gate_raw);
        let candidate = activation::tanh(candidate_raw);
        let new_state =
            gate.clone() * state.clone() + (gate.clone().neg().add_scalar(1.0)) * candidate;
        let bias = self
            .inner
            .project_bias(activation::tanh(output_raw).unsqueeze_dim::<3>(1));
        (new_state, bias.squeeze_dim::<2>(1))
    }

    /// Teacher-forced block application (DeepSpec `apply_block_logits`):
    /// `base_logits [B, N, L, V]`, `token_ids [B, N, L]` (previous ids per
    /// position), `hidden_states [B, N, L, d]`.
    pub fn apply_block_logits(
        &self,
        base_logits: Tensor<4>,
        token_ids: Tensor<3, Int>,
        hidden_states: Tensor<4>,
    ) -> Tensor<4> {
        let [b, n, l, v] = base_logits.dims();
        let d = self.hidden_size;
        let mut state = Tensor::<2>::zeros([b * n, self.state_size], &base_logits.device());
        let mut out = Vec::with_capacity(l);
        for k in 0..l {
            let prev_emb = self
                .inner
                .get_prev_embeddings(
                    token_ids
                        .clone()
                        .slice([0..b, 0..n, k..k + 1])
                        .reshape([b * n, 1]),
                )
                .reshape([b * n, self.inner.rank]);
            let h_k = hidden_states
                .clone()
                .slice([0..b, 0..n, k..k + 1, 0..d])
                .reshape([b * n, d]);
            let (new_state, bias) = self.rnn_step(state, prev_emb, h_k);
            state = new_state;
            let step_logits = base_logits
                .clone()
                .slice([0..b, 0..n, k..k + 1, 0..v])
                .reshape([b * n, v])
                .add(bias);
            out.push(step_logits.reshape([b, n, 1, v]));
        }
        Tensor::cat(out, 2)
    }

    /// Autoregressive sampling with recurrent state (DeepSpec
    /// `sample_block_tokens`).
    pub fn sample_block_tokens(
        &self,
        base_logits: Tensor<3>,
        first_prev_token_ids: Tensor<2, Int>,
        hidden_states: Tensor<3>,
        temperature: f32,
    ) -> (Tensor<2, Int>, Tensor<3>) {
        let [b, l, v] = base_logits.dims();
        let d = self.hidden_size;
        let mut state = Tensor::<2>::zeros([b, self.state_size], &base_logits.device());
        let mut prev_ids = first_prev_token_ids; // [B, 1]
        let mut sampled: Vec<Tensor<2, Int>> = Vec::with_capacity(l);
        let mut corrected: Vec<Tensor<3>> = Vec::with_capacity(l);
        for k in 0..l {
            let prev_emb = self
                .inner
                .get_prev_embeddings(prev_ids.clone())
                .reshape([b, self.inner.rank]);
            let h_k = hidden_states
                .clone()
                .slice([0..b, k..k + 1, 0..d])
                .reshape([b, d]);
            let (new_state, bias) = self.rnn_step(state, prev_emb, h_k);
            state = new_state;
            let step_logits = base_logits
                .clone()
                .slice([0..b, k..k + 1, 0..v])
                .reshape([b, v])
                .add(bias)
                .unsqueeze_dim::<3>(1);
            corrected.push(step_logits.clone());
            let next = sample_tokens(step_logits, temperature);
            sampled.push(next.clone());
            prev_ids = next;
        }
        (Tensor::cat(sampled, 1), Tensor::cat(corrected, 1))
    }
}

/// Autoregressive greedy draft within a block (kept for API compatibility;
/// prefer `sample_block_tokens(.., temperature=0.0)` which is equivalent).
///
/// `base_logits`: `[B, N, block_size, vocab]`, `anchor_token`: `[B, N]`.
/// Returns `[B, N, block_size]` draft ids (argmax).
pub fn greedy_draft(
    base_logits: Tensor<4>,
    markov: &VanillaMarkov,
    anchor_token: Tensor<2, Int>,
) -> Tensor<3, Int> {
    let [b, n, l, v] = base_logits.dims();
    let mut out = Vec::with_capacity(n);
    for j in 0..n {
        let blk = base_logits
            .clone()
            .slice([0..b, j..j + 1, 0..l, 0..v])
            .reshape([b, l, v]);
        let anchor = anchor_token.clone().slice([0..b, j..j + 1]);
        let (ids, _) = markov.sample_block_tokens(blk, anchor, 0.0);
        out.push(ids.unsqueeze_dim::<3>(1));
    }
    Tensor::cat(out, 1)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn dev() -> Device {
        Device::ndarray()
    }

    #[test]
    fn vanilla_shapes() {
        let m = VanillaMarkov::new(256, 32, &dev());
        let ids: Tensor<2, Int> =
            Tensor::from_data(burn::tensor::TensorData::new(vec![1i32, 2], [1, 2]), &dev());
        assert_eq!(m.get_prev_embeddings(ids.clone()).dims(), [1, 2, 32]);
        assert_eq!(m.compute_step_bias(ids).dims(), [1, 2, 256]);
        let l = Tensor::<3>::zeros([1, 2, 256], &dev());
        assert_eq!(
            m.apply_step_logits(l, Tensor::<2, Int>::zeros([1, 2], &dev()))
                .dims(),
            [1, 2, 256]
        );
    }

    #[test]
    fn vanilla_block_apply() {
        let m = VanillaMarkov::new(64, 8, &dev());
        let bl = Tensor::<4>::zeros([1, 1, 4, 64], &dev());
        let prev: Tensor<3, Int> = Tensor::zeros([1, 1, 4], &dev());
        assert_eq!(m.apply_block_logits(bl, prev).dims(), [1, 1, 4, 64]);
    }

    #[test]
    fn vanilla_sample_conditions_on_prev() {
        let m = VanillaMarkov::new(64, 8, &dev());
        let bl = Tensor::<4>::zeros([1, 1, 4, 64], &dev()).reshape([1, 4, 64]);
        let first: Tensor<2, Int> = Tensor::zeros([1, 1], &dev());
        let (ids, corrected) = m.sample_block_tokens(bl, first, 0.0);
        assert_eq!(ids.dims(), [1, 4]);
        assert_eq!(corrected.dims(), [1, 4, 64]);
    }

    #[test]
    fn rnn_head_forward() {
        let m = RNNHead::new(64, 8, 16, &dev());
        let bl = Tensor::<4>::zeros([2, 1, 4, 64], &dev());
        let ids: Tensor<3, Int> = Tensor::zeros([2, 1, 4], &dev());
        let hs = Tensor::<4>::zeros([2, 1, 4, 16], &dev());
        assert_eq!(m.apply_block_logits(bl, ids, hs).dims(), [2, 1, 4, 64]);
    }

    #[test]
    fn rnn_sample_shape() {
        let m = RNNHead::new(64, 8, 16, &dev());
        let bl = Tensor::<3>::zeros([1, 4, 64], &dev());
        let first: Tensor<2, Int> = Tensor::zeros([1, 1], &dev());
        let hs = Tensor::<3>::zeros([1, 4, 16], &dev());
        let (ids, corrected) = m.sample_block_tokens(bl, first, hs, 0.0);
        assert_eq!(ids.dims(), [1, 4]);
        assert_eq!(corrected.dims(), [1, 4, 64]);
    }

    #[test]
    fn gated_head_shapes() {
        let m = GatedMarkovHead::new(64, 8, 16, &dev());
        let ids: Tensor<2, Int> = Tensor::zeros([1, 4], &dev());
        let hs = Tensor::<3>::zeros([1, 4, 16], &dev());
        assert_eq!(m.compute_step_bias(ids, hs).dims(), [1, 4, 64]);
    }
}
