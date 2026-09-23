# burn-muon-plus

**Muon+** optimizer for Burn — Muon with one post-polar normalization step
([arXiv:2602.21545](https://arxiv.org/abs/2602.21545), UCSB).

## The problem

Newton–Schulz polar iterations flatten the singular spectrum of the momentum
matrix — but in practice they **amplify** column/row norm imbalance in the
update (the paper's "post-polar imbalanced update problem"). Imbalanced
updates tighten the second-order term in a blockwise descent analysis and
shrink Muon's largest stable step size.

## The fix

One normalization step after orthogonalization, zero extra optimizer state:

```text
M_t  = μ·M_{t-1} + (1-μ)·G_t
O_t  = Norm_(d)(Ortho(M_t))      # NS polar + row/col normalization
W_t  = W_{t-1} - η·√(m/n)·O_t
```

- `Norm_col(X) = X·D_col⁻¹` — unit-L2 columns
- `Norm_row(X) = D_row⁻¹·X` — unit-L2 rows
- `ColRow` / `RowCol` — sequential composition (best in the paper)

## Results (paper)

Up to **37.1% pre-training speedup** to a target loss; validation perplexity
down by up to 2.02; works on top of any polar method (Jordan / You /
PolarExpress / exact SVD). 2D weights use Muon+, non-matrix parameters
(biases, norms) fall back to AdamW — the paper's recipe.

## Usage

```rust
use burn_muon_plus::{MuonPlusConfig, NormDir};

let mut optimizer = MuonPlusConfig::new()
    .with_norm_dir(Some(NormDir::ColRow))   // paper's best
    .with_momentum(0.95)
    .init();
let model = optimizer.step(lr, model, grads);
```

## References

- [Muon+: Towards More Effective Muon via One Additional Normalization Step for LLM Pre-training](https://arxiv.org/abs/2602.21545) (2602.21545)
- [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) (Jordan et al.)

## Performance (RTX 5060 Ti, CUDA, burn 0.22)

The Muon+ step's elementwise passes dominate on CUDA (~45 tensor passes per
step vs ~15 cuBLAS matmuls). Fused cubecl kernels (NS polynomial combine,
momentum, final decay+update, ColRow normalize with in-kernel reductions)
collapse them to a handful of launches.

| Config | Tensor path | Fused kernels | Speedup |
|--------|-------------|---------------|---------|
| [4096×4096] step | 22.4 ms | **232 µs** | **96×** |
| [8192×1024] step | 47.3 ms | **563 µs** | **84×** |

Also: the Newton-Schulz polynomial is evaluated factored
(a·x + b·(xx·x) + c·(xx·(xx·x))) for strongly non-square params — measured
3.6× faster on [8192,512] with a lower peak footprint (never materializes
(XXᵀ)²). All kernels verified == tensor path (<1e-4).
