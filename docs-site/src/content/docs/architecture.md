---
title: How it works
description: The block, the optimizer, the two invariants the code enforces, and where everything lives.
---

## One block, twelve times

A byte goes in, twelve identical blocks later a prediction comes out. Each block
is attention followed by a feed-forward network, with pre-LN and residual
connections — the shape that powers most language models, cut down to fit one
card.

<figure class="clip">
  <video src="/dormouse/docs/media/Bytes.webm" poster="/dormouse/docs/media/Bytes.jpg" autoplay muted loop playsinline></video>
  <figcaption>The alphabet is the 256 values a byte can hold. No tokenizer stands between the file and the model.</figcaption>
</figure>

The FFN is **gated** (SwiGLU) at two-thirds of the width of the GELU block it
replaces. That is the one architectural change that won: it holds the parameter
count exactly constant — `3·832·2048 = 2·832·3072` — and moved held-out loss by
−0.0145 BPB over three seeds.

## The optimizer thinks in matrices

AdamW treats every number on its own. **Muon** treats a weight matrix as a
matrix: before each step it orthogonalizes the update, so every direction gets
the same strength and the step stays well-conditioned.

<figure class="clip">
  <video src="/dormouse/docs/media/Muon.webm" poster="/dormouse/docs/media/Muon.jpg" autoplay muted loop playsinline></video>
  <figcaption>Newton–Schulz runs five iterations on the small side of the Gram matrix; a skewed spectrum comes out flat.</figcaption>
</figure>

The orthogonalization is five Newton–Schulz iterations on the *small* side of
the Gram matrix, batched across every matrix of the same shape. Dropping to
three iterations is 1.08× faster and costs 10% of the loss — so five it is.

## Spectral compression

Wide feed-forward matrices are where the parameters live. TSCT rewrites the
matrix rather than routing around it: a singular-value decomposition, the tail of
the spectrum dropped, and the two surviving factors stored as thin matrices.

<figure class="clip">
  <video src="/dormouse/docs/media/Spectral.webm" poster="/dormouse/docs/media/Spectral.jpg" autoplay muted loop playsinline></video>
  <figcaption>The spectrum is cut at rank 6 of 14; the factors are stored in three levels. There is no router and nothing is chosen per token.</figcaption>
</figure>

The cost is the rank, and it is measured rather than assumed: at `r/f ≈ 9%` the
loss penalty is around 5% while the arithmetic drops ~2.7×. That is a trade the
recipe takes only when the freed budget buys something bigger.

## Two invariants, enforced at runtime

Both exist because both were violated here.

1. **An arm that is off is bit-identical to its absence.** A projector is not
   called at all when its flag is off; a test spies on the function and requires
   zero calls. The failure it prevents: an `use_kda=false` that silently meant
   "no attention at all", and an `attn_pos` flag that was never applied.
2. **A mechanism that cannot pay refuses loudly.** A config naming a removed
   mode, or a combination that loses money, fails at load and lists the
   survivors. The failure it prevents: an arm that takes three hours and then
   turns out to have measured nothing.

## The tree

```
port/src/dormouse/
  config.py          every flag, every refusal
  model/
    model.py         embed → blocks → head
    transformer.py   the live block
    tsct.py          low-rank factors
    ternary.py       the b1.58 codec: ternarize, STE, anneal
    act_quant.py     activation quantization
    packed_acts.py   saved activations stored packed
    export.py        pack → container → reload
  train/
    loop.py          the training step, eval, checkpointing, resume
    optim.py         Muon + AdamW
  post/              SFT and the verifier-first RL stack
```

---

## References

Every technique on this page, with the paper it comes from:

- **Byte-level modelling** — MegaByte, Yu et al. 2023, [arXiv:2305.07185](https://arxiv.org/abs/2305.07185); BLT, Pagnoni et al. 2024, [arXiv:2412.09871](https://arxiv.org/abs/2412.09871)
- **Gated FFN (SwiGLU)** — Shazeer 2020, *GLU Variants Improve Transformer*, [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
- **Rotary position embedding** — Su et al. 2021, *RoFormer*, [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- **Flash attention** — Dao et al. 2022, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- **Muon** — Liu et al. 2025, *Muon is Scalable for LLM Training*, [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)
- **Newton–Schulz via a minimax polynomial** — Amsel et al., *Polar Express*, [arXiv:2505.16932](https://arxiv.org/abs/2505.16932)
- **Spectral compression (TSCT)** — Kohlberger, *Spectral Compact Training*, [arXiv:2604.00733](https://arxiv.org/abs/2604.00733)

