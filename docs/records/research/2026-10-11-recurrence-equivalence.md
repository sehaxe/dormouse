# Recurrence is worth 0.46 of a parameter — the depth arms must be judged by φ, not by loss

**2026-10-11.** Source: `docs/papers/iso-depth-2604.21106.pdf` —
*How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language
Models*, Schwethelm, Rückert, Kaissis (arXiv:2604.21106v3, 2026-05-07). Read from
the abstract and the introduction of the local PDF; the scaling law and every
number below are the paper's, quoted with its own notation.

## The finding

An iso-depth pretraining sweep across recurrence counts `r ∈ {1,2,4,8}` spanning
~50x training compute fits a joint law

```
L = E + A (Nonce + r^φ N_rec)^−α + B D^−β
```

and measures **φ = 0.46**. φ is the exchange rate between a recurrence and a
unique parameter: φ = 1 would mean running a shared block r times buys what r
distinct blocks buy; φ = 0 would mean it buys nothing at all. 0.46 is in between,
with a concrete price attached: **at r = 4, a 410M looped model matches the loss
of a 580M non-looped model while costing what training a 1B non-looped model
costs.**

## Why this changes our A/B queue

§3.4 lists "depth 2 vs 4, the cheapest big lever" as a queue arm, and the queue
judges arms by held-out BPB at a step budget. **That is exactly the comparison
this paper warns against**: a recurrence arm can lower the loss while being
worse than the alternative at matched capacity, because the loss conflates two
things the paper separates — a true capacity gain (φ up) and a training-side gain
(loss down, φ flat or down).

The paper's two case studies are the demonstration, and both are about the exact
knobs we own:

| what | φ | the paper's reading |
|---|---|---|
| full backward through the loop | 0.46 | the baseline |
| **truncated backpropagation** | **0.38** | "the loop mechanism is poorly trained under truncation, even though validation loss decreases" |
| **hyperconnections** | **0.65** | a genuine capacity gain |

Two consequences for us, one of each:

1. **We are on the right side of the first one.** The trainer's backend is
   `Autodiff<Cube, NoCheckpointing>` — the log line says `checkpointing:
   Disabled`, so the loop is never truncated. Anything that later shortens the
   backward through the iterations (a memory knob, a compile-time
   checkpointing strategy) must be re-judged with φ, not with the loss it also
   lowers.
2. **We should adopt φ as the depth arms' verdict metric, alongside BPB.** Our
   machinery is already almost there: `--eval-depths` prints held-out BPB at
   depths 1..=max_iter at every eval, and `LoopBlock` accumulates per-iteration
   CE rather than materialising `[N,b,t,d]`. What is missing is the fit: two or
   three depths at the same parameter count and the same token budget, then the
   exponent that maps one depth to the non-looped model of equivalent loss. That
   is a protocol line, not a code change, and it is what keeps a "cheaper depth"
   arm from winning on a technicality.

## The second paper, and why our LoopBlock is already on its right side

`docs/papers/looped-dit-2609.40305.pdf` — *Looped Diffusion Transformer*
(arXiv:2609.40305v1, 2026-09-30). It runs shared Transformer blocks repeatedly
inside one denoising step and reports that **naive looping fails to improve
quality at all**, tracing it to (a) weak supervision across the intermediate
loops and (b) unregulated attention updates that "progressively erode local
information". Its fix is deep supervision across the loops plus self-modulating
attention.

Both of those are already in our block, which is the useful part of the
comparison:

- **Deep supervision**: `L_Rec` is accumulated per iteration and the loss is the
  honest mean CE over them — the same "supervise every loop" structure, with the
  per-iteration CE kept inside `forward_full_state` precisely so no
  `[N,b,t,d]` step-hiddens tensor is materialised (§2.2).
- **Regulated writes**: the ReZero residual is gated per iteration, and
  `use_attnres` mixes `{token embedding, this iteration's body}` through a learned
  softmax instead of an unregulated write (arXiv:2603.15031, §3.5's item).

So the two fresh papers agree with the block's existing shape, and the first one
hands the depth arms a stricter verdict metric than BPB alone.

## What to do with this, concretely

1. Add to `docs/protocols/AB-PROTOCOL.md`: a depth arm reports BPB **and** the
   recurrence-equivalence estimate, and a depth arm that wins on BPB alone is not
   a verdict. One line, with this document as the citation.
2. Keep a regime watch on the isodepth law: it was fitted at 410M-1B scale and
   ~50x compute, so extrapolating φ to a 9M model on a 16 GB box is the paper's
   number applied outside its fitted range — ours to measure, not to assume.
3. The `mm-*` papers in `docs/papers/` are domain work (ECG classification,
   multimodal pretraining) with one usable structural echo — a CNN+SSM+attention
   hybrid with span-masked JEPA pretraining (`mm-2609.29376`) — which is the same
   "local morphology, then temporal, then global" ordering our KDA arm already
   implements. No action.
