# A Lean 4 formalisation of the TSCT math core

TSCT stores a weight matrix `W ∈ F^{d×f}` as `r` **ternary rank-1 terms with one
real scale each**:

```
W ≈ Σ_{k<r} s_k · (u_k ⊗ v_k),   u_k ∈ {-1,0,1}^d,  v_k ∈ {-1,0,1}^f,  s_k ∈ F
```

This directory proves the four claims that design rests on.  It is **Lean 4
core only** — no Mathlib, no `Std` dependency, no `lake exe cache get`, no
network.  The field is `Rat` (the theorems are algebraic identities plus
linear/quadratic inequalities, so nothing is lost; the only analytic ingredient
is "a square is nonnegative", proved from `Rat.le_total`).  Because core Lean
ships no `Finset`/`Fintype`/`ring`/`linarith`, the finite sum, the dot products
and their lemmas are built in `Tsct/Basic.lean`, and every arithmetic step is
`grind` (`ring`-normalisation plus linear arithmetic) or `ac_rfl`.

```
proof/
  lean-toolchain          leanprover/lean4:v4.35.0   (the installed toolchain)
  lakefile.toml           one library, `Tsct`
  Tsct.lean               root: imports Basic, Scale, Capacity
  Tsct/Basic.lean         vectors, matrices, finite sums, ternary alphabet
  Tsct/Scale.lean         theorems 1–3
  Tsct/Capacity.lean      theorem 4
```

`lake build` is green, seconds not minutes, with **zero `sorry`, zero `axiom`,
zero `native_decide`** on anything load-bearing.  `#print axioms` on every
headline theorem reports only Lean's standard `propext` / `Quot.sound` /
`Classical.choice` — no `sorryAx`.

---

## Theorem 1 — the optimal scale, closed form  (`Tsct/Scale.lean`)

| theorem | statement |
|---|---|
| `errT_eq` | `‖W − s·T‖² = ‖W‖² − 2s⟨W,T⟩ + s²‖T‖²` |
| `complete_square` | if `‖T‖²·t = ⟨W,T⟩` then `‖W − s·T‖² = ‖W − t·T‖² + ‖T‖²(s−t)²` |
| `optimal_scale_min` | with `‖T‖² ≥ 0` and `‖T‖²·t = ⟨W,T⟩`: `errT W T t ≤ errT W T s` for **every** `s` |
| `sstar_normal_eq`, `residual_formula` | `s* = ⟨W,T⟩/‖T‖²` satisfies the normal equation, and `errT W T s* = ‖W‖² − ⟨W,T⟩²/‖T‖²` |
| `qT_outer` | `‖u⊗v‖² = ‖u‖²·‖v‖²` |
| **`optimal_scale_ternary`** | headline: for nonzero ternary `u,v`, `s* = ⟨W,u⊗v⟩/‖u⊗v‖²` minimises `‖W − s·(u⊗v)‖²` |
| **`residual_outer`** | headline: `‖W − s*·(u⊗v)‖² = ‖W‖² − ⟨W,u⊗v⟩²/(‖u‖²‖v‖²)` |

The proof is the complete-square identity itself, not a derivative argument:
`‖W − sT‖² − ‖W − tT‖² = ‖T‖²(s−t)²`, which is `≥ 0`.  The hypothesis `‖T‖² ≠ 0`
cannot be dropped — if `u = 0` or `v = 0` the error is `‖W‖²` for every `s`, so
the denominator is zero and no scale is optimal.  `normSq_pos_of_ternary` is the
lemma that discharges it for ternary factors: a ternary vector with a nonzero
entry has `‖u‖² > 0`.

## Theorem 2 — per-rank scales dominate one shared scale  (`Tsct/Scale.lean`)

| theorem | statement |
|---|---|
| `err2_expand` | for `⟨T1,T2⟩ = 0`: `‖W − aT1 − bT2‖² = ‖W‖² − 2a⟨W,T1⟩ − 2b⟨W,T2⟩ + a²‖T1‖² + b²‖T2‖²` |
| `two_term_complete_square` | the separable quadratic completes into two independent squares |
| **`per_rank_dominates_shared`** | per-term scales `t₁,t₂` (solving the two normal equations) give error `≤` the error of **any** shared scale `s` |
| `per_rank_dominates_shared_div` | the same with the closed forms `⟨W,Tₖ⟩/‖Tₖ‖²` |

**Which case is proved, exactly:** `r = 2` with *orthogonal* directions
(`⟨T1,T2⟩ = 0`), one shared scale, all `s` (not just the best shared `s`).
Orthogonality is what makes the two coordinate least-squares problems
independent, hence the closed forms valid.  Without it, the coordinate-wise
closed form is **not** the joint minimiser — for `T1 = T2` the joint solution is
`t₁+t₂ = ⟨W,T⟩/‖T‖²`, so each coordinate gets half — even though the per-term
model still cannot lose to the shared one (a larger parameter set cannot).
The general-`r`, correlated-directions *inequality* is therefore true but is not
formalised here; only the closed-form orthogonal case is.

## Theorem 3 — the stack  (`Tsct/Scale.lean`)

| theorem | statement |
|---|---|
| `stacking_pythagoras` | for `⟨T1,T2⟩ = 0`: `‖T1+T2‖² = ‖T1‖² + ‖T2‖²` |
| `stack_residual_split` | `‖W − T1 − s·T2‖² = ‖W−T1‖² − 2s⟨W−T1,T2⟩ + s²‖T2‖²` |
| **`stacking_never_hurts`** | fitting the next term with its optimal scale gives a residual `≤` the residual before it: `errT (W−T1) T2 s* ≤ ‖W−T1‖²` |
| `mdot_add_add` | `‖A+B‖² = ‖A‖² + 2⟨A,B⟩ + ‖B‖²` |
| **`mdot_add_bound`** | `‖A+B‖² ≤ 2‖A‖² + 2‖B‖²` — the triangle inequality, squared |

`stacking_never_hurts` is Theorem 1 applied to `W' = W − T1` at `s = 0`: the old
residual is the value at `s = 0`, and the optimum is at least that good.  This
is what licenses "adding terms can only help".

**Weakening, stated:** `mdot_add_bound` is the *squared* triangle inequality
`‖A+B‖² ≤ 2‖A‖² + 2‖B‖²`, not `‖A+B‖ ≤ ‖A‖ + ‖B‖`.  Core Lean has no square
root, and no statement in this file needs one; the squared form is the exact
form of "the error of the sum is controlled by the sum of the errors" and it is
strictly what the stacking argument uses.  The exact orthogonal decomposition
(`stacking_pythagoras`) is stronger and is also proved, so this is a
supplementary bound, not a gap in the stacking claim.

## Theorem 4 — the capacity bound  (`Tsct/Capacity.lean`)

| theorem | statement |
|---|---|
| `address_space` | `3^d · 3^f = 3^(d+f)` addresses `(u,v)` |
| `neg_neg_outer` | `(-u) ⊗ (-v) = u ⊗ v` — the global sign is unobservable when the scale is free |
| `zero_outer_left`, `zero_outer_right` | `u = 0` or `v = 0` gives the zero matrix (for every `v`, resp. `u`) |
| **`fibre`** | for nonzero ternary factors: `u⊗v = u'⊗v'` forces `(u',v') = (u,v)` or `(-u,-v)` — the fibre is exactly 2 |
| `pow3_sub_one_even`, `two_mul_distinctNonzero` | `3^d−1` is even, so `2 · distinctNonzero d f = (3^d−1)(3^f−1)` — the halving is exact |
| **`fibre_partition`** | `3^(d+f) = 2 · distinctNonzero d f + (3^d + 3^f − 1)` |
| `distinctCount_one_eq_three_pow` | where the `3^(d+f−1)` accounting *is* right: `d = 1` gives `3^f = 3^(1+f−1)` |
| `distinctCount_two_two`, `three_pow_pred_two_two`, **`requested_count_is_false`** | `distinctCount 2 2 = 33 ≠ 27 = 3^(2+2−1)` |
| `ternary_bits_lower`, `ternary_bits_upper` | `2^30 < 3^19` and `3^32 < 2^51`, i.e. `log₂3 ∈ (30/19, 51/32) = (1.578…, 1.594…)` — b1.58 |

with

```lean
def distinctNonzero (d f : Nat) : Nat := (3 ^ d - 1) * (3 ^ f - 1) / 2
def distinctCount    (d f : Nat) : Nat := distinctNonzero d f + 1
```

### The `3^(d+f−1)` claim is false, and this file says so

The task's fourth claim — "the number of distinct ternary rank-1 matrices is
exactly `3^(d+f−1)`" — is **not provable, because it is false**.  At `d = f = 2`
the true count is `33` and `3^(2+2−1) = 27` (`requested_count_is_false`).

The error in the folklore is counting the `±1` identification as if it were the
only collision.  It is not: the *zero* matrix is hit by every pair with
`u = 0` or `v = 0`, i.e. by `3^d + 3^f − 1` addresses
(`zero_outer_left`, `zero_outer_right`, and the arithmetic in
`fibre_partition`).  The exact replacement is

```
3^(d+f)  =  2 · distinctNonzero d f  +  (3^d + 3^f − 1)
             ↑ two-to-one fibres        ↑ the zero fibre
```

which is `fibre_partition`, with `fibre` proving the "exactly two" part and
`neg_neg_outer` proving the two addresses really do collide.  `distinctCount`
is therefore `((3^d−1)(3^f−1))/2 + 1`, and it equals `3^(d+f−1)` only when
`d = 1` or `f = 1` (`distinctCount_one_eq_three_pow`) — the scalar-output case.

What is *not* formalised: the general principle "the number of distinct images
of a map equals the number of fibres".  Core Lean has no `Fintype`/`Finset` and
this project does not hand-roll a quotient-cardinality theory.  Instead it
proves the fibre structure (`fibre`, `neg_neg_outer`, `zero_outer_*`) and the
exact integer arithmetic that it implies.  That is the whole mathematical
content of the count; the missing step is bookkeeping, not mathematics, and it
is stated here rather than hidden.

---

## Mapping to `port/src/dormouse/model/tsct.py`

Read against the file as it stands (`945` lines).  `u` is `[in, rank]`, `v` is
`[rank, out]`, `s` is `[rank]`, so column `k` of `u` and row `k` of `v` are the
`u_k, v_k` of this development and `W = Σ_k s_k u_k ⊗ v_k`.

| Lean | `tsct.py` |
|---|---|
| `outer s u v` (`s * u i * v j`) | `TSCTLinear.effective_weight`: `(u * self.s) @ v` (line 673–679) — the same product, `W[i,j] = Σ_k u[i,k]·s[k]·v[k,j]` |
| `errT` / `optimal_scale_ternary` | `TSCTLinear.__init__` line 625: `self.s = nn.Parameter(torch.ones(rank))` — one free real per rank; theorem 1 says what it must be, `⟨W,u_k⊗v_k⟩/‖u_k⊗v_k‖²` |
| the free `s` in theorem 1 | line 628–630: "`s` is never quantized — it **is** the scale" — the one parameter that stays continuous while `u,v` are ternary |
| `per_rank_dominates_shared` | the same `self.s = ones(rank)` **vector** rather than a single scalar: one shared scale is dominated by one scale per rank.  This is the theorem behind storing `s` with shape `[rank]` |
| `Ternary` hypothesis | `factor_qat_project` (line 399–489) with `factor_qat = "ternary"` → `ternary.ternarize` per tensor, applied to both factors via `TSCTLinear._factors` (line 663–671); the b1.58 alphabet of the design |
| `_ternary_quant` / `_ternary_dequant` semantics | lines 784–816: codes `{0,1,2}` ⇒ `{0,+α,−α}` — the storage form of exactly the `{-1,0,1}` set, one `α` per matrix (distinct from the per-rank `s`) |
| `address_space`, `distinctCount` | `TSCTLinear.rank` with the guard `rank ≤ min(in_features, out_features)` (line 609–615) and `_FACTOR_SUFFIXES` (line 719): `d·f` is replaced by `r·(d+f+1)` parameters, and theorem 4 counts how many distinct rank-1 shapes `d+f` ternary digits can address |
| `fibre` | `pack_tsct` / `unpack_tsct` ternary rung (line 826–871): the stored object is `(codes, α)`, i.e. the *matrix*, and theorem 4's fibre is why `(u,v)` and `(-u,-v)` are one stored shape, not two |
| `stacking_never_hurts` | `rank` as a growth knob: adding a rank (a term) with its fitted `s` cannot raise the approximation error |

## What is deliberately **not** modelled

* **fp8 / NVFP4.**  `tsct.py` has those as *storage rungs* (`RUNG_MAX`,
  `_e2m1_quant`, line 715–781) and as a *compute lane* (`mm_fp8`, line 213–256).
  Neither is part of the ternary-rank-1 design formalised here and neither is
  modelled.
* **The STE and the training semantics.**  `factor_qat_project`'s
  `w + α·(q(w) − w).detach()`, the annealed α, and the S3T Bernoulli mask
  (line 295–375) are training-time mechanisms.  `Ternary` models the
  *constraint on the factors*, not the projection that produces it.
* **Fitting `u` and `v`.**  Theorems 1–3 hold `u_k, v_k` fixed and optimise the
  scales — which is exactly what the design does (`s` is a free parameter, the
  factors are what the quantizer emitted).  Optimal ternary factors are a
  combinatorial problem and are not claimed here.
* **Float behaviour.**  `Rat` is exact; bf16 accumulation, per-tensor absmean
  `α`, and the `1e-2` storage round-trip tolerance are not modelled and no
  theorem is claimed about them.
* **The `r`-term joint least squares** (non-orthogonal, general `r`) — see
  Theorem 2 above.

## Building

```
cd proof
lake build          # ~3 s cold, no dependencies, no network
```

Toolchain pinned to `leanprover/lean4:v4.35.0` (the installed one; nothing is
downloaded).
