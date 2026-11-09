import Tsct.Basic

/-!
Theorems 1–3 of the TSCT core: optimal per-term scale, per-rank scales beating
one shared scale, and the stacking (residual-decomposition) laws.

Everything is over `Rat`; the field is not really used as a *field* until the
optimal scale's `/` appears, so the results are as general as the algebra.
-/

namespace Tsct

/-! ## Theorem 1 — the optimal scale, closed form

`TSCTLinear` stores one `s` per rank (`self.s`, `tsct.py:625`) and never
quantizes it (`tsct.py:629-630`).  Theorem 1 answers what that scalar has to
be: for a fixed ternary pair `(u, v)`, the minimiser of `‖W - s·(u⊗v)‖²`. -/

/-- Squared residual of `W` against the single term `s · T`. -/
def errT {d f : Nat} (W T : Fin d → Fin f → Rat) (s : Rat) : Rat :=
  mdot (fun i j => W i j - s * T i j) (fun i j => W i j - s * T i j)

/-- `⟨W, T⟩` — the numerator of the optimal scale. -/
def cW {d f : Nat} (W T : Fin d → Fin f → Rat) : Rat := mdot W T

/-- `‖T‖²` — the denominator of the optimal scale. -/
def qT {d f : Nat} (T : Fin d → Fin f → Rat) : Rat := mdot T T

/-- The least-squares scale `s* = ⟨W,T⟩ / ‖T‖²`. -/
def sstarT {d f : Nat} (W T : Fin d → Fin f → Rat) : Rat := cW W T / qT T

theorem errT_eq {d f : Nat} (W T : Fin d → Fin f → Rat) (s : Rat) :
    errT W T s = mdot W W - 2 * s * cW W T + s * s * qT T := by
  unfold errT cW qT
  rw [mdot_quad]

/-- **Theorem 1a (complete square).**  Any `t` solving the normal equation
`‖T‖²·t = ⟨W,T⟩` is a minimiser, and the excess over it is exactly
`‖T‖²·(s-t)²`.  No minimisation theory is needed — the identity is the proof.

The hypothesis cannot be dropped: for `T = 0` every `s` has error `‖W‖²`, so no
`t` is "the" minimiser; that is the degenerate case excluded by `qT T ≠ 0`. -/
theorem complete_square {d f : Nat} (W T : Fin d → Fin f → Rat) {t s : Rat}
    (h : qT T * t = cW W T) :
    errT W T s = errT W T t + qT T * (s - t) * (s - t) := by
  rw [errT_eq W T s, errT_eq W T t]
  unfold cW qT at h ⊢
  rw [← h]
  grind

/-- **Theorem 1b (minimality).**  With `⟨W,T⟩ = ‖T‖²·t` and `‖T‖² ≥ 0`,
`t` is at least as good as every `s`.  (`‖T‖² = 0` is the `T = 0` case, where
the claim is still true — everything ties — and the identity above still holds
because the right-hand side collapses.) -/
theorem optimal_scale_min {d f : Nat} (W T : Fin d → Fin f → Rat) {t : Rat}
    (hq : 0 ≤ qT T) (h : qT T * t = cW W T) (s : Rat) :
    errT W T t ≤ errT W T s := by
  have hs := complete_square W T h (s := s)
  have hnn : 0 ≤ qT T * (s - t) * (s - t) := by
    have : qT T * (s - t) * (s - t) = qT T * ((s - t) * (s - t)) := by grind
    rw [this]
    exact Rat.mul_nonneg hq (mul_self_nonneg _)
  grind

/-- `s* = ⟨W,T⟩/‖T‖²` satisfies the normal equation. -/
theorem sstar_normal_eq {d f : Nat} (W T : Fin d → Fin f → Rat) (hq0 : qT T ≠ 0) :
    qT T * sstarT W T = cW W T := by
  unfold sstarT
  grind

/-- **Theorem 1c (residual).**  At the optimal scale the residual is
`‖W‖² − ⟨W,T⟩²/‖T‖²`, i.e. the part of `W` that `T` cannot explain — the
closed form the design's `s` is computed from. -/
theorem residual_formula {d f : Nat} (W T : Fin d → Fin f → Rat) (hq0 : qT T ≠ 0) :
    errT W T (sstarT W T) = mdot W W - cW W T * cW W T / qT T := by
  rw [errT_eq W T (sstarT W T)]
  unfold sstarT
  grind

/-! ### The same theorem for a ternary rank-1 term

`T = 1 • (u ⊗ v)`, so `‖T‖² = ‖u‖²·‖v‖²` and, for nonzero ternary factors,
that is strictly positive. -/

theorem qT_outer {d f : Nat} (u : Fin d → Rat) (v : Fin f → Rat) :
    qT (outer 1 u v) = normSq u * normSq v := by
  unfold qT; rw [mdot_outer_outer]; grind

theorem qT_outer_nonneg {d f : Nat} (u : Fin d → Rat) (v : Fin f → Rat) :
    0 ≤ qT (outer 1 u v) := by
  rw [qT_outer]
  exact Rat.mul_nonneg (normSq_nonneg u) (normSq_nonneg v)

theorem qT_outer_pos {d f : Nat} {u : Fin d → Rat} {v : Fin f → Rat}
    (hu : Ternary u) (hv : Ternary v) (hu0 : ∃ i, u i ≠ 0) (hv0 : ∃ j, v j ≠ 0) :
    0 < qT (outer 1 u v) := by
  rw [qT_outer]
  exact Rat.mul_pos (normSq_pos_of_ternary hu hu0) (normSq_pos_of_ternary hv hv0)

/-- `‖s • u⊗v‖² = s²‖u‖²‖v‖²` — used by the stacking theorems. -/
theorem normSq_outer {d f : Nat} (s : Rat) (u : Fin d → Rat) (v : Fin f → Rat) :
    mdot (outer s u v) (outer s u v) = (s * s) * (normSq u * normSq v) := by
  rw [mdot_outer_outer]; grind

/-- The residual against one term, in the factor form the model stores. -/
def err {d f : Nat} (W : Fin d → Fin f → Rat) (u : Fin d → Rat) (v : Fin f → Rat)
    (s : Rat) : Rat := errT W (outer 1 u v) s

/-- `s* = ⟨W, u⊗v⟩ / (‖u‖²‖v‖²)` for the stored ternary pair. -/
def sstar {d f : Nat} (W : Fin d → Fin f → Rat) (u : Fin d → Rat) (v : Fin f → Rat) : Rat :=
  cW W (outer 1 u v) / qT (outer 1 u v)

/-- **Theorem 1, headline.**  For a target `W` and a nonzero ternary pair, the
scale `⟨W,u⊗v⟩/‖u⊗v‖²` minimises `‖W - s·(u⊗v)‖²` over every real `s`.

The two nonzero-entry hypotheses are not decoration: if `u = 0` or `v = 0`
then `u⊗v = 0` and the error is `‖W‖²` for every `s`, so the formula's
denominator is zero and "the" optimal scale does not exist. -/
theorem optimal_scale_ternary {d f : Nat} (W : Fin d → Fin f → Rat) {u : Fin d → Rat}
    {v : Fin f → Rat} (hu : Ternary u) (hv : Ternary v) (hu0 : ∃ i, u i ≠ 0)
    (hv0 : ∃ j, v j ≠ 0) (s : Rat) :
    err W u v (sstar W u v) ≤ err W u v s := by
  have hpos := qT_outer_pos hu hv hu0 hv0
  exact optimal_scale_min W (outer 1 u v) (Rat.le_of_lt hpos)
    (sstar_normal_eq W (outer 1 u v) (Rat.ne_of_gt hpos)) s

/-- **Theorem 1, residual, headline.**  `‖W − s*·(u⊗v)‖² = ‖W‖² − ⟨W,u⊗v⟩²/(‖u‖²‖v‖²)`. -/
theorem residual_outer {d f : Nat} (W : Fin d → Fin f → Rat) {u : Fin d → Rat}
    {v : Fin f → Rat} (hu : Ternary u) (hv : Ternary v) (hu0 : ∃ i, u i ≠ 0)
    (hv0 : ∃ j, v j ≠ 0) :
    err W u v (sstar W u v)
      = mdot W W - cW W (outer 1 u v) * cW W (outer 1 u v) / (normSq u * normSq v) := by
  have hpos := qT_outer_pos hu hv hu0 hv0
  have h := residual_formula W (outer 1 u v) (Rat.ne_of_gt hpos)
  rw [qT_outer] at h
  exact h

/-! ## Theorem 2 — per-rank scales dominate one shared scale

Two terms `T1, T2` and *orthogonal* directions (`⟨T1,T2⟩ = 0`) — the regime in
which the two per-term scales separate.  What is proved: the pair of per-term
scales `(⟨W,T1⟩/‖T1‖², ⟨W,T2⟩/‖T2‖²)` achieves error `≤` the error of *any*
model that forces one shared scale `s` on both terms.

**Scope, stated plainly.** This is the `r = 2` orthogonal case, not the general
`r` with correlated directions.  Orthogonality is what makes the two coordinate
least-squares problems independent; without it the coordinate-wise formula is
NOT the joint minimiser (`T1 = T2` gives `t1 = t2 = c/q`, each half of the true
joint optimum) even though per-term scales still cannot lose — a larger
parameter set cannot.  So the honest general statement is the inequality; the
*closed form* needs orthogonality, which is what is formalised here. -/

/-- Squared residual of `W` against `a · T1 + b · T2`. -/
def err2 {d f : Nat} (W T1 T2 : Fin d → Fin f → Rat) (a b : Rat) : Rat :=
  mdot (fun i j => W i j - a * T1 i j - b * T2 i j)
    (fun i j => W i j - a * T1 i j - b * T2 i j)

/-- Expansion of the two-term residual.  With orthogonal terms the cross term
`2ab⟨T1,T2⟩` vanishes and the form is separable in `a` and `b`. -/
theorem err2_expand {d f : Nat} (W T1 T2 : Fin d → Fin f → Rat)
    (horth : mdot T1 T2 = 0) (a b : Rat) :
    err2 W T1 T2 a b
      = mdot W W - 2 * a * mdot W T1 - 2 * b * mdot W T2
        + a * a * mdot T1 T1 + b * b * mdot T2 T2 := by
  have hstep : err2 W T1 T2 a b
      = mdot (fun i j => W i j - a * T1 i j) (fun i j => W i j - a * T1 i j)
        - 2 * b * mdot (fun i j => W i j - a * T1 i j) T2 + b * b * mdot T2 T2 := by
    exact mdot_quad (a := fun i j => W i j - a * T1 i j) (b := T2) (s := b)
  have hcross : mdot (fun i j => W i j - a * T1 i j) T2 = mdot W T2 - a * mdot T1 T2 := by
    rw [mdot_sub_left, mdot_smul_left]
  rw [hstep, mdot_quad W T1 a, hcross, horth]
  grind

/-- The separable quadratic, completed: with `Q₁t₁ = C₁` and `Q₂t₂ = C₂` the
excess over `(t₁,t₂)` is the sum of the two one-dimensional squares. -/
theorem two_term_complete_square (A C₁ C₂ Q₁ Q₂ t₁ t₂ a b : Rat)
    (h1 : Q₁ * t₁ = C₁) (h2 : Q₂ * t₂ = C₂) :
    A - 2 * a * C₁ - 2 * b * C₂ + a * a * Q₁ + b * b * Q₂
      = (A - 2 * t₁ * C₁ - 2 * t₂ * C₂ + t₁ * t₁ * Q₁ + t₂ * t₂ * Q₂)
        + Q₁ * (a - t₁) * (a - t₁) + Q₂ * (b - t₂) * (b - t₂) := by
  rw [← h1, ← h2]
  grind

/-- **Theorem 2.**  Per-term optimal scales beat every one-shared-scale model.
`t₁, t₂` are the per-term scales; `s` is any shared scale, in particular the
best one. -/
theorem per_rank_dominates_shared {d f : Nat} (W T1 T2 : Fin d → Fin f → Rat)
    (horth : mdot T1 T2 = 0) {t₁ t₂ : Rat}
    (h1 : mdot T1 T1 * t₁ = mdot W T1) (h2 : mdot T2 T2 * t₂ = mdot W T2)
    (hn1 : 0 ≤ mdot T1 T1) (hn2 : 0 ≤ mdot T2 T2) (s : Rat) :
    err2 W T1 T2 t₁ t₂ ≤ err2 W T1 T2 s s := by
  rw [err2_expand W T1 T2 horth]
  have hrw : err2 W T1 T2 s s
      = (mdot W W - 2 * t₁ * mdot W T1 - 2 * t₂ * mdot W T2
          + t₁ * t₁ * mdot T1 T1 + t₂ * t₂ * mdot T2 T2)
        + (mdot T1 T1 * (s - t₁) * (s - t₁) + mdot T2 T2 * (s - t₂) * (s - t₂)) := by
    rw [err2_expand W T1 T2 horth]
    rw [two_term_complete_square (mdot W W) (mdot W T1) (mdot W T2) (mdot T1 T1)
          (mdot T2 T2) t₁ t₂ s s h1 h2]
    grind
  rw [hrw]
  have h1' : 0 ≤ mdot T1 T1 * (s - t₁) * (s - t₁) := by
    rw [show mdot T1 T1 * (s - t₁) * (s - t₁) = mdot T1 T1 * ((s - t₁) * (s - t₁)) by grind]
    exact Rat.mul_nonneg hn1 (mul_self_nonneg _)
  have h2' : 0 ≤ mdot T2 T2 * (s - t₂) * (s - t₂) := by
    rw [show mdot T2 T2 * (s - t₂) * (s - t₂) = mdot T2 T2 * ((s - t₂) * (s - t₂)) by grind]
    exact Rat.mul_nonneg hn2 (mul_self_nonneg _)
  grind

/-- Theorem 2 with the closed-form scales written as divisions: for orthogonal
nonzero directions the per-rank optimum is `⟨W,Tₖ⟩/‖Tₖ‖²`. -/
theorem per_rank_dominates_shared_div {d f : Nat} (W T1 T2 : Fin d → Fin f → Rat)
    (horth : mdot T1 T2 = 0) (hq1 : mdot T1 T1 ≠ 0) (hq2 : mdot T2 T2 ≠ 0)
    (hn1 : 0 ≤ mdot T1 T1) (hn2 : 0 ≤ mdot T2 T2) (s : Rat) :
    err2 W T1 T2 (mdot W T1 / mdot T1 T1) (mdot W T2 / mdot T2 T2) ≤ err2 W T1 T2 s s := by
  refine per_rank_dominates_shared W T1 T2 horth ?_ ?_ hn1 hn2 s
  · grind
  · grind

/-! ## Theorem 3 — the stack

Adding an optimally-scaled term can only lower the residual, and the error of
an orthogonal stack splits exactly.  Both are corollaries of Theorem 1 applied
to the already-residual `W' = W - T1`; no new minimisation theory. -/

/-- Exact Pythagorean split of an orthogonal sum: `‖T1 + T2‖² = ‖T1‖² + ‖T2‖²`. -/
theorem stacking_pythagoras {d f : Nat} (T1 T2 : Fin d → Fin f → Rat)
    (horth : mdot T1 T2 = 0) :
    mdot (fun i j => T1 i j + T2 i j) (fun i j => T1 i j + T2 i j)
      = mdot T1 T1 + mdot T2 T2 := by
  rw [show (fun i j => T1 i j + T2 i j) = (fun i j => T1 i j - (-1) * T2 i j) from
        funext (fun i => funext (fun j => by grind))]
  rw [mdot_quad T1 T2 (-1), horth]
  grind

/-- The residual of a stack splits as `‖W − T1 − s·T2‖² = ‖W−T1‖² − 2s⟨W−T1,T2⟩ + s²‖T2‖²`
— the invariant behind "the per-rank scales are fitted one term at a time". -/
theorem stack_residual_split {d f : Nat} (W T1 T2 : Fin d → Fin f → Rat) (s : Rat) :
    err2 W T1 T2 1 s
      = mdot (fun i j => W i j - T1 i j) (fun i j => W i j - T1 i j)
        - 2 * s * mdot (fun i j => W i j - T1 i j) T2 + s * s * mdot T2 T2 := by
  have h := mdot_quad (a := fun i j => W i j - T1 i j) (b := T2) (s := s)
  unfold err2
  rw [show (fun i j => W i j - 1 * T1 i j - s * T2 i j)
        = (fun i j => (fun i j => W i j - T1 i j) i j - s * T2 i j) from
      funext (fun i => funext (fun j => by grind))]
  exact h

/-- **Theorem 3a (adding a term cannot hurt).**  Whatever the residual after
`T1` is, fitting the direction `T2` with its optimal scale gives a residual no
larger.  This is what licenses "more ranks, never worse", and it is Theorem 1
with `W` replaced by `W - T1`: the value at `s = 0` is the old residual, and the
optimal `s` is at least as good as `s = 0`.

`hq0` (a nonzero `T2`) is required for `s*` to be the normal-equation solution;
the `T2 = 0` case is a no-op (`‖W − T1 − s·0‖² = ‖W − T1‖²` for all `s`). -/
theorem stacking_never_hurts {d f : Nat} (W T1 T2 : Fin d → Fin f → Rat)
    (hq : 0 ≤ mdot T2 T2) (hq0 : qT T2 ≠ 0) :
    errT (fun i j => W i j - T1 i j) T2 (sstarT (fun i j => W i j - T1 i j) T2)
      ≤ mdot (fun i j => W i j - T1 i j) (fun i j => W i j - T1 i j) := by
  have h := optimal_scale_min (fun i j => W i j - T1 i j) T2 hq
    (sstar_normal_eq _ T2 hq0) 0
  have h0 : errT (fun i j => W i j - T1 i j) T2 0
      = mdot (fun i j => W i j - T1 i j) (fun i j => W i j - T1 i j) := by
    unfold errT
    rw [show (fun i j => (fun i j => W i j - T1 i j) i j - 0 * T2 i j)
          = (fun i j => W i j - T1 i j) from funext (fun i => funext (fun j => by grind))]
  rw [h0] at h
  exact h

/-- Exact expansion of an orthogonal sum's squared norm:
`‖A + B‖² = ‖A‖² + 2⟨A,B⟩ + ‖B‖²`. -/
theorem mdot_add_add {d f : Nat} (A B : Fin d → Fin f → Rat) :
    mdot (fun i j => A i j + B i j) (fun i j => A i j + B i j)
      = mdot A A + 2 * mdot A B + mdot B B := by
  rw [show (fun i j => A i j + B i j) = (fun i j => A i j - (-1) * B i j) from
        funext (fun i => funext (fun j => by grind))]
  rw [mdot_quad A B (-1)]
  grind

/-- **Theorem 3b (squared triangle inequality).**  `‖A + B‖² ≤ 2‖A‖² + 2‖B‖²`,
i.e. the Frobenius norm obeys the triangle inequality up to the factor that
squaring costs.  Stated squared because this development has no square root.
The proof is `‖A+B‖² = 2‖A‖² + 2‖B‖² − ‖A−B‖²` and `‖A−B‖² ≥ 0`. -/
theorem mdot_add_bound {d f : Nat} (A B : Fin d → Fin f → Rat) :
    mdot (fun i j => A i j + B i j) (fun i j => A i j + B i j)
      ≤ 2 * mdot A A + 2 * mdot B B := by
  have hsub : mdot (fun i j => A i j - B i j) (fun i j => A i j - B i j)
      = mdot A A - 2 * mdot A B + mdot B B := by
    rw [show (fun i j => A i j - B i j) = (fun i j => A i j - 1 * B i j) from
          funext (fun i => funext (fun j => by grind))]
    rw [mdot_quad A B 1]
    grind
  have hnn : 0 ≤ mdot (fun i j => A i j - B i j) (fun i j => A i j - B i j) :=
    mdot_nonneg _
  rw [mdot_add_add]
  grind

end Tsct
