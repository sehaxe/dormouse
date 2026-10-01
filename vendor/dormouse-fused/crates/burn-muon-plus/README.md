# burn-muon-plus

**Muon+** optimizer for Burn — Muon with one post-polar normalization step
([arXiv:2602.21545](https://arxiv.org/abs/2602.21545) **v3**, UCSB).

> **Cite v3.** v1 has appendices A–C, v2/v3 have A–G, and every NS-coefficient
> table the paper prints is in App. D — a document that exists only in the
> later versions. App. D.1 prints the default triple verbatim:
> *"In [15], the coefficients are set to `(a,b,c)=(3.4445, -4.7750, 2.0315)`."*
> App. D.3 prints the PolarExpress schedule, terminating at
> `(1.875, -1.25, 0.375)`.

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
W_t  = W_{t-1} - η·S(m,n)·O_t
```

where `S(m,n) = max(1, m/n)^0.5` in **this implementation** and `√(m/n)` in the
paper (Eq. (4), App. C Alg. 1 line 10). The `max` is Jordan et al.'s
(`update *= max(1, update.size(-2)/update.size(-1))**0.5` in `KellerJordan/muon`
`muon_update`) and this crate follows the implementation, so the two differ
only on a wide `m < n` matrix, where this one takes a full step and the paper
takes a shrunk one. There is no wide matrix in either Muon group the trainer
installs. See `src/lib.rs`, the `lr_scaled` comment, for the whole argument.

- `Norm_col(X) = X·D_col⁻¹` — unit-L2 columns
- `Norm_row(X) = D_row⁻¹·X` — unit-L2 rows
- `ColRow` / `RowCol` — sequential composition, Eqs. (7)/(8). The paper does
  not rank the two orders: §3.4 claims only that bi-directional beats
  single-directional, and its Table 6 has the two within noise (25.25/25.25,
  18.65/18.68, 13.41/13.44) with the winner flipping by model.

## Results (paper)

**37.1%** fewer **optimizer steps to a fixed target loss** (v3 §1 and Table 5;
GPT-Base 3447 → 2515 steps). It is not wall-clock and not a per-step
improvement — §3.3 says Muon+ "has nearly the same per-step runtime and memory
cost as Muon". Validation perplexity down by up to 2.02; works on top of any
polar method (Jordan / You / PolarExpress / exact SVD). 2D weights use Muon+,
non-matrix parameters (biases, norms) fall back to AdamW — the paper's recipe.

## Usage

```rust
use burn_muon_plus::{MuonPlusConfig, NormDir};

let mut optimizer = MuonPlusConfig::new()
    .with_norm_dir(Some(NormDir::ColRow))
    .with_momentum(0.95)
    .init();
let model = optimizer.step(lr, model, grads);
```

`norm_dir` defaults to `None`, which is **plain Muon** — the baseline the
paper measures Muon+ against, not Muon+. Set it.

## References

- [Muon+: Towards More Effective Muon via One Additional Normalization Step for LLM Pre-training](https://arxiv.org/abs/2602.21545) (2602.21545, v3)
- [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) (Jordan et al.) — source of the NS coefficients and of `max(1, m/n)^0.5`
- [The Polar Express](https://arxiv.org/abs/2505.16932) (Amsel et al.) — App. D.3's coefficients

## Performance (RTX 5060 Ti, CUDA, burn 0.22)

The Muon+ step's elementwise passes dominate on CUDA (~45 tensor passes per
step vs ~15 cuBLAS matmuls). Fused cubecl kernels (NS polynomial combine,
momentum, final decay+update, ColRow normalize with in-kernel reductions)
collapse them to a handful of launches.

| Config | Tensor path | Fused kernels | Speedup |
|--------|-------------|---------------|---------|
| [4096×4096] step | 22.4 ms | **232 µs** | **96×** |
| [8192×1024] step | 47.3 ms | **563 µs** | **84×** |

**Both columns are retracted: the bench has no device flush.**
`step_bench` (`src/fused_kernels.rs:302-312`) times 10 iterations between one
`Instant::now()` and one `elapsed()` with no `into_scalar`/`into_data` in the
loop, on either side. 232 µs is the host cost of queueing the launches, and
22.4 ms is the host cost of queueing ~15 matmuls; the **96× is a ratio of CPU
dispatch overhead**, not a speedup. `ortho_bench` (`:243`) and `ns_bench`
(`src/lib.rs:447`) have the same defect, so the "3.6× faster" below is
unmeasured too. The fix is one line per timed loop (a flush per iteration, as
`burn-rope/src/rope_cuda.rs:284` does); it is a `src/` change and is not made
here. Nothing in this crate has a measured kernel speedup until it is.

The Newton-Schulz polynomial used to have a second, *factored* evaluation
order (`a·x + b·(xx·x) + c·(xx·(xx·x))`) for strongly non-square parameters,
and the "3.6× faster on [8192,512]" claim was about it. Both are gone: the
branch was behind `nc * 4 < nr`, which is unsatisfiable because `orthogonalize`
transposes a tall argument first (`nc ≥ nr` always), and the two orders are
the same polynomial regrouped. The one surviving half of the old claim is the
**memory** one — the factored form never materializes `(XXᵀ)²` — which is true
and no longer buys anything, since the direct form is also the cheaper one on
the reachable domain (`2r²c + r³` against `3r²c`, and `r ≤ c`). The bench that
produced the speed number also skipped the transpose, so it measured a shape
this path can never present. `ns_combine_cuda` keeps its kernel and its
numerical test and has no production caller left (`src/lib.rs`, module decl).

Numerics: **all four** kernels are checked against the tensor-op expression
they compute — `norm_colrow` at <1e-4 (`norm_colrow_match`,
`src/fused_kernels.rs:216`), and `ns_combine` / `momentum` / `finalize` at
<1e-5 each (`fused_match_tensor`, `src/fused_kernels.rs:344`, including a
row-pitched `[1024, 300]` shape so the stride indexing is covered). This
README used to say only one of them was checked; that was wrong, and the
right-hand correction is the direction that adds evidence rather than
removing it. Two limits that do stand: those tests live behind
`#[cfg(all(test, feature = "cuda"))]`, so on a CPU build **no** kernel is
verified, and **no external reference exists** for any of them (ADR-0020) —
they are compared against this crate's own tensor path, which is the weaker
claim.
