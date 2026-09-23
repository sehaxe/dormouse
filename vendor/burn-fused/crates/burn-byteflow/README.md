# burn-byteflow

ByteFlow Net for Burn: tokenizer-free language modeling through adaptive byte
compression ([Deng et al., ICLR 2026](https://arxiv.org/abs/2603.03583)).

Five stages over a raw byte stream `x_1:T ∈ V^T` (V = 256):

1. **Local encoder** — E pre-norm causal blocks: sliding-window attention (window
   `w_local`) + Canon layers (causal depthwise conv k=4 with per-channel gates,
   Allen-Zhu 2025) + SwiGLU FFN.
2. **Downsampling** — coding-rate chunking: marginal gains
   `ΔR_t = R(h_1:t) − R(h_1:t−1)`, Top-K selection with forced BOS position,
   chronological order, projection to `d_global`. Static graph by construction.
3. **Global transformer** — G deep/wide causal blocks on the K compressed tokens.
4. **Upsampling** — multi-linear reconstruction with B=16 shared bins and large
   residual: `s_t = h_t + g_chunk(t)·W_bin(t)`.
5. **Decoder** — symmetric to the local encoder, projects to next-byte logits.

Coding rate: exact lossy form `R_ε(h) = ½ log det(I + (d/ε²) H Hᵀ)` (host
Cholesky, analysis/validation) and the default L2 streaming approximation
`R ∝ ‖H‖₂` from Appendix B.

```rust
use burn_byteflow::{ByteFlowConfig, ByteFlowNet, RateMode};
use burn::tensor::Device;

let device = Device::ndarray();
let net = ByteFlowNet::init(ByteFlowConfig::default(), &device);
let bytes = burn::tensor::Tensor::<2, burn::tensor::Int>::zeros([2, 8192], &device);
let logits = net.forward(bytes); // [2, 8192, 256]
```
