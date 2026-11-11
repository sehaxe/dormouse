# TSCT on the live stack — what is wired, and what the night measured

This note is the operational half of `tsct-ternary-spec.md`: what actually runs,
what it costs, and which claims are still open. Written 2026-11-11 after the
wiring landed.

## The wiring that did not exist until now

`ffn_form="tsct"` reached only the kda `LoopBlock`. On `block_form="transformer"`
— the live recipe — it was a **silent no-op**: the config declared `tsct`, the
model held zero `TSCTLinear` modules, and the parameter count was identical to
the dense control. Commit `3e5c727` wires it and makes every unwired form refuse
to load.

| config (d=1152, L=18, f=2848) | params | `TSCTLinear` | FFN params/layer |
|---|---:|---:|---:|
| `arch_dense275.toml` | 273,393,792 | 0 | 9,842,688 |
| `arch_tsct275.toml` (r=64) | 110,052,864 | 54 | 768,192 |

FFN params and FLOPs are both **12.8× smaller**; body FLOPs/token 545.4M → 218.8M
(2.49×), because attention keeps its dense projections in this arm.

## What is verified

- **Exact reparameterisation at full rank**: copying a dense model's weights into
  the factorised form via SVD reproduces logits to `max|Δlogits| = 9.5e-07`
  (gate 1e-5). So the factorised form is a reparameterisation, not an
  approximation, when `r = min(d,f)`.
- **Rank-64 param arithmetic** is exactly `3·(d·r + r + r·f)` = 768,192 per FFN,
  hand-computed in the test.
- **Flag OFF is bit-identical** to the previous commit (names + logits, `torch.equal`).
- `stiefel_qr` composes: the factor names match `is_factor_key`, so the retraction
  reaches `*.u`/`*.v` and refuses loudly when there are none.

## The open question, stated as a number

At rank 64 the FFN holds a fraction of the dense matrix's energy — our own
measurement on the `down` matrix: **43.4% at r=64**, and 99% needs 941 of 1024
singular values. So the mechanism is a *function-shape* compression, and the bet
is that the freed FLOPs/params buy more elsewhere (depth, width, data) than the
rank costs in expressiveness.

**The energy curve is also the reason this cannot be read as "compression".** If
a *trained* dense matrix needed 92% of its singular values to hold 99% of its
energy, then post-hoc factorisation at r=64 is not a 12.8× win, it is a
different (worse) function. TSCT is therefore a bet on **training under the
constraint** — the same bet BitNet makes with ternary weights, where a model
trained in the format lands somewhere no post-hoc quantisation reaches — and not
a bet on low-rank-ness of arbitrary solutions. Two consequences:

- the arm must be trained from scratch with the constraint on (which it is), and
- a *quality* loss at rank 64 does not by itself kill the mechanism; a loss that
  survives reinvesting the freed budget does (that is what `arch_tsct275_wide`
  tests).

Two arms exist to test exactly that:

| arm | params | FLOPs/token | what it answers |
|---|---:|---:|---|
| `arch_tsct275` | 110.1M | 219M | what rank-64 costs at an identical shape |
| `arch_tsct275_wide` | 272.8M | 544M | what reinvesting the freed budget buys (d=1536, L=26) |

`arch_dense275` (273.4M, 545M) is the denominator for both.

## Verdicts

**PENDING** — the 3-seed A/B was started at 01:22 and runs overnight. Nothing in
this note should be read as a claim that TSCT holds quality: the only numbers
above are parameter and FLOP arithmetic, which are exact, and the full-rank
equivalence, which says nothing about r=64.

If the mechanism arm loses by more than the seed spread, the ladder is: rank 96,
then rank 128, then QR-on (`stiefel_qr`) — and only after those, abandon. The
arm stays in the tree as a flag either way; a measured refusal is a result.
