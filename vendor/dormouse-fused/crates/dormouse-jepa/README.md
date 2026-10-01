# dormouse-jepa

JEPA for [Burn](https://burn.dev) 0.22 — data2vec 2.0-style EMA teacher +
masked latent prediction for byte-level language modeling, plus LeJEPA and
KoLeo auxiliary losses.

## Papers

| arXiv | Component | What |
|-------|-----------|------|
| [2212.07525](https://arxiv.org/abs/2212.07525) | `EmaTarget`, `JepaPredictor`, `mask_indices`, `jepa_l1_loss` | **data2vec 2.0** (Baevski et al., ICML 2023): EMA teacher produces target latents; student predicts latents of masked positions with one shared head, trained with masked L1 |
| [2511.08544](https://arxiv.org/abs/2511.08544) | `lejepa_loss` | Isotropic Gaussian regularization (Balestriero & LeCun, 2025) |
| [2304.07193](https://arxiv.org/abs/2304.07193) | `koleo_loss` | KoLeo uniformity (Caron et al., DINOv2, 2023) |

## Mechanism (data2vec 2.0)

1. **EMA teacher**: target weights are an exponential moving average of the
   student's, `θ_target = m·θ_target + (1−m)·θ_student`, m ≈ 0.999+ (ramp
   recommended). Targets are detached (stop-grad).
2. **Masking**: a fraction of input positions are masked in random
   contiguous spans (default 15%, span 8).
3. **Prediction**: the student encoder runs on the masked input; a single
   lightweight predictor head (LayerNorm → Linear) predicts the teacher's
   latents of the masked positions.
4. **Loss**: L1 between predicted and teacher latents at masked positions
   only — no input reconstruction.

The teacher encoder is *your* model: this crate provides the scalar EMA
primitive, masking, the predictor head, and the losses. Typical use as an
auxiliary SSL loss on top of next-byte prediction:

```rust
use burn::module::Module;
use burn::tensor::{backend::Backend, Tensor, Bool};
use dormouse_jepa::{EmaTarget, JepaConfig, JepaPredictor, jepa_l1_loss, mask_indices};

let config = JepaConfig::new(0.999, 0.15, 8, d_model); // momentum, mask_frac, mask_span, predictor_dim
let predictor = JepaPredictor::new(d_model, &device);
let mut teacher = EmaTarget::new(0.999, &device);

// teacher latents (detached, from EMA-copied encoder), student on masked input
let mask = mask_indices::<B>(t, config.mask_frac, config.mask_span, &device);   // [t]
let mask_b = mask.unsqueeze_dim::<D2>(0).expand([b, t]);                         // [b, t]
let pred = predictor.forward(student_latents);                                   // [b, t, d]
let loss = jepa_l1_loss(pred, teacher_latents, mask_b);                          // masked L1
teacher.update(scalar_target); // or EMA the whole encoder weights yourself
```

## License

MIT.
