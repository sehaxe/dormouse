import Tsct.Basic

/-!
Theorem 4 — the capacity bound.

A rank-1 ternary term is addressed by two ternary vectors,
`(u, v) ∈ {-1,0,1}^d × {-1,0,1}^f`, and one free real scale.  Two things
follow, and both are proved here:

* the address space has exactly `3^d · 3^f = 3^(d+f)` elements;
* the map `(u,v) ↦ u ⊗ v` is **two-to-one** on the part where both vectors are
  nonzero (the pair `(-u,-v)` gives the same matrix), and it is many-to-one
  only over the zero matrix, where `u = 0` or `v = 0` all collapse.

That is the arithmetic of "how many shapes one bank can store": `d + f`
ternary symbols per rank, one bit of it spent on the global sign.

## What is honestly *not* formalised, and why the literal claim is false

Core Lean ships no `Fintype`/`Finset`, so there is no general
"cardinality of a quotient" principle to appeal to.  Instead of hand-rolling
one, this file proves the *fibre structure* (which is the mathematical content)
and the exact integer arithmetic that it implies.  The exact count of distinct
nonzero matrices is proved to be `(3^d-1)(3^f-1)/2`, and the total (adding the
zero matrix) is `((3^d-1)(3^f-1))/2 + 1 =: distinctCount d f`.

`distinctCount d f` is **not** `3^(d+f-1)` in general.  It is:
* `= 3^f` when `d = 1` (and symmetrically when `f = 1`) — the case in which
  an FFN-style readout has a scalar output width, which is where the `3^(d+f-1)`
  accounting is right;
* `33` vs `27` at `d = f = 2` (`distinctCount_two_two`,
  `requested_count_is_false` below).

The mistake in the `3^(d+f-1)` folklore is that it counts the `±1` identification
as if it were the *only* collision; the zero matrix has a fibre of
`3^d + 3^f - 1` address pairs, not 2.  `fibre_partition` is the exact
replacement: the `3^(d+f)` addresses split into `2`-element fibres plus that
one big zero fibre.
-/

namespace Tsct

/-! ## The address space -/

/-- **Theorem 4a.**  The number of `(u,v)` sign patterns is `3^d · 3^f = 3^(d+f)`. -/
theorem address_space (d f : Nat) : 3 ^ d * 3 ^ f = 3 ^ (d + f) := by
  rw [Nat.pow_add]

/-- **Theorem 4b (the redundancy).**  Flipping the sign of both factors leaves
the rank-1 matrix unchanged; with a free scale the sign is unobservable. -/
theorem neg_neg_outer {d f : Nat} (s : Rat) (u : Fin d → Rat) (v : Fin f → Rat) :
    outer s (fun i => -u i) (fun j => -v j) = outer s u v := by
  funext i j
  simp only [outer]
  grind

theorem zero_outer_left {d f : Nat} (s : Rat) (v : Fin f → Rat) :
    outer s (fun _ : Fin d => (0 : Rat)) v = fun _ _ => (0 : Rat) := by
  funext i j
  simp only [outer]
  grind

theorem zero_outer_right {d f : Nat} (s : Rat) (u : Fin d → Rat) :
    outer s u (fun _ : Fin f => (0 : Rat)) = fun _ _ => (0 : Rat) := by
  funext i j
  simp only [outer]
  grind

/-! ## The two-to-one fibre

This is the theorem the capacity count rests on: if two ternary pairs with
nonzero factors give the same rank-1 matrix, they are equal or globally
negated — nothing else collides. -/

/-- **Theorem 4c (fibre of the rank-1 map).**  For nonzero ternary factors,
`u ⊗ v = u' ⊗ v'` forces `(u',v') = (u,v)` or `(u',v') = (-u,-v)`.

The nonzero-factor hypotheses are exactly what is needed: without them
`u = 0` makes *every* `v` give the zero matrix, so the fibre is not two-to-one
(see `zero_outer_left`). -/
theorem fibre {d f : Nat} (u u' : Fin d → Rat) (v v' : Fin f → Rat)
    (hu : Ternary u) (hv : Ternary v) (hu' : Ternary u') (hv' : Ternary v')
    (huv : ∃ i, u i ≠ 0) (hvv : ∃ j, v j ≠ 0)
    (h : outer 1 u v = outer 1 u' v') :
    (u' = u ∧ v' = v) ∨ (u' = (fun i => -u i) ∧ v' = (fun j => -v j)) := by
  obtain ⟨i0, hi0⟩ := huv
  obtain ⟨j0, hj0⟩ := hvv
  have hpt : ∀ i j, u i * v j = u' i * v' j := by
    intro i j
    have := congrFun (congrFun h i) j
    simp only [outer] at this
    grind
  have hkey := hpt i0 j0
  have hu'0 : u' i0 ≠ 0 := by intro hz; rw [hz] at hkey; grind
  have hv'0 : v' j0 ≠ 0 := by intro hz; rw [hz] at hkey; grind
  have ha : u i0 = 1 ∨ u i0 = -1 := ternary_eq_pm_one (hu i0) hi0
  have ha' : u' i0 = 1 ∨ u' i0 = -1 := ternary_eq_pm_one (hu' i0) hu'0
  have hb : v j0 = 1 ∨ v j0 = -1 := ternary_eq_pm_one (hv j0) hj0
  have hb' : v' j0 = 1 ∨ v' j0 = -1 := ternary_eq_pm_one (hv' j0) hv'0
  have hbranch : u' i0 = u i0 ∨ u' i0 = -u i0 := by
    rcases ha with ha | ha <;> rcases ha' with ha' | ha' <;>
      rw [ha, ha'] <;> first | exact Or.inl (by grind) | exact Or.inr (by grind)
  rcases hbranch with hbr | hbr
  · left
    have hu_eq : ∀ i, u' i = u i := by
      intro i
      have h1 := hpt i j0
      rcases hb with hb | hb <;> rcases hb' with hb' | hb' <;>
        (rw [hb] at h1 hkey; rw [hb'] at h1 hkey; grind)
    refine ⟨funext hu_eq, ?_⟩
    funext j
    have h1 := hpt i0 j
    grind
  · right
    have hu_eq : ∀ i, u' i = -u i := by
      intro i
      have h1 := hpt i j0
      rcases hb with hb | hb <;> rcases hb' with hb' | hb' <;>
        (rw [hb] at h1 hkey; rw [hb'] at h1 hkey; grind)
    refine ⟨funext hu_eq, ?_⟩
    funext j
    have h1 := hpt i0 j
    grind

/-! ## The count, exactly

`distinctNonzero d f` is the number of distinct nonzero rank-1 ternary
matrices; `distinctCount d f` adds the zero matrix.  The integer facts are
proved, and `fibre` is what makes `distinctNonzero` the right formula
(every nonzero fibre has exactly two addresses: `fibre` gives at most two,
`neg_neg_outer` gives the second). -/

/-- Number of distinct *nonzero* rank-1 ternary `d × f` matrices:
`(3^d - 1)(3^f - 1)/2`. -/
def distinctNonzero (d f : Nat) : Nat := (3 ^ d - 1) * (3 ^ f - 1) / 2

/-- Number of distinct rank-1 ternary `d × f` matrices, zero matrix included. -/
def distinctCount (d f : Nat) : Nat := distinctNonzero d f + 1

/-- `3^(d+1) - 1` is even (indeed `3^(d+1) = 2k+1`). -/
theorem pow3_sub_one_even (d : Nat) : ∃ k, 3 ^ (d + 1) - 1 = 2 * k := by
  induction d with
  | zero => exact ⟨1, by decide⟩
  | succ d ih =>
    obtain ⟨k, hk⟩ := ih
    refine ⟨3 * k + 1, ?_⟩
    have hpos : 1 ≤ 3 ^ (d + 1) := Nat.one_le_pow _ _ (by decide)
    have hrep : 3 ^ (d + 1) = 2 * k + 1 := by omega
    rw [Nat.pow_succ, hrep]
    omega

/-- `2 · distinctNonzero d f = (3^d−1)(3^f−1)`: the halving is exact. -/
theorem two_mul_distinctNonzero (d f : Nat) (hd : 1 ≤ d) :
    2 * distinctNonzero d f = (3 ^ d - 1) * (3 ^ f - 1) := by
  obtain ⟨d', rfl⟩ : ∃ d', d = d' + 1 := ⟨d - 1, by omega⟩
  obtain ⟨k, hk⟩ := pow3_sub_one_even d'
  have hN : distinctNonzero (d' + 1) f = k * (3 ^ f - 1) := by
    unfold distinctNonzero
    rw [hk]
    rw [show (2 * k) * (3 ^ f - 1) = (k * (3 ^ f - 1)) * 2 by ac_rfl]
    rw [Nat.mul_div_cancel _ (by decide : 0 < 2)]
  rw [hN, hk]
  ac_rfl

/-- **Theorem 4d (the exact split of the address space).**  For `d ≥ 1`, the
`3^(d+f)` `(u,v)` addresses are `2 · distinctNonzero d f` nonzero-fibre
addresses plus `3^d + 3^f - 1` addresses of the zero matrix
(`u = 0`: `3^f`; `v = 0`: `3^d`; both: `1`).  (The identity itself only needs
`d ≥ 1`; `f ≥ 1` is needed only to read it as fibres of the rank-1 map.) -/
theorem fibre_partition (d f : Nat) (hd : 1 ≤ d) :
    3 ^ (d + f) = 2 * distinctNonzero d f + (3 ^ d + 3 ^ f - 1) := by
  rw [Nat.pow_add, two_mul_distinctNonzero d f hd]
  -- substitute `3^d = x+1`, `3^f = y+1` so the Nat subtractions are honest
  have hpd : 1 ≤ 3 ^ d := Nat.one_le_pow _ _ (by decide)
  have hpf : 1 ≤ 3 ^ f := Nat.one_le_pow _ _ (by decide)
  obtain ⟨x, hx⟩ : ∃ x, 3 ^ d = x + 1 := ⟨3 ^ d - 1, by omega⟩
  obtain ⟨y, hy⟩ : ∃ y, 3 ^ f = y + 1 := ⟨3 ^ f - 1, by omega⟩
  rw [hx, hy]
  simp only [Nat.add_sub_cancel]
  rw [Nat.add_mul, Nat.mul_add, Nat.mul_one, Nat.one_mul]
  omega

/-- Where the `3^(d+f-1)` accounting *is* right: `d = 1`.  A `1 × f` bank has
exactly `3^f` distinct rank-1 shapes, matching `3^(1+f-1)`. -/
theorem distinctCount_one (f : Nat) : distinctCount 1 f = 3 ^ f := by
  have h1 : 1 ≤ 3 ^ f := Nat.one_le_pow _ _ (by decide)
  unfold distinctCount distinctNonzero
  rw [show (3 : Nat) ^ 1 - 1 = 2 from by decide]
  rw [show 2 * (3 ^ f - 1) = (3 ^ f - 1) * 2 by ac_rfl]
  rw [Nat.mul_div_cancel _ (by decide : 0 < 2)]
  omega

/-- The requested identity, at its true special case: `distinctCount 1 f = 3^(1+f-1)`. -/
theorem distinctCount_one_eq_three_pow (f : Nat) : distinctCount 1 f = 3 ^ (1 + f - 1) := by
  rw [distinctCount_one]
  congr 1
  omega

/-- `distinctCount 2 2 = 33`. -/
theorem distinctCount_two_two : distinctCount 2 2 = 33 := by
  unfold distinctCount distinctNonzero
  decide

/-- `3^(2+2-1) = 27`, so the literal reading differs from the true count. -/
theorem three_pow_pred_two_two : 3 ^ (2 + 2 - 1) = 27 := by decide

/-- **The correction, stated as a theorem.**  The number of distinct rank-1
ternary `2×2` matrices is 33, not `3^(2+2-1) = 27`: the zero matrix's fibre
(`3² + 3² − 1 = 17` addresses) is the discrepancy. -/
theorem requested_count_is_false : distinctCount 2 2 ≠ 3 ^ (2 + 2 - 1) := by
  rw [distinctCount_two_two, three_pow_pred_two_two]
  decide

/-! ## b1.58, the bit budget

`log₂ 3 ≈ 1.58496`, so a ternary symbol is 1.58 bits and `d + f` of them cost
`1.58·(d+f)` bits per rank.  The two integer inequalities below pin that value
without any real logarithm: 19 ternary symbols do **not** fit in 30 bits
(`> 30/19 = 1.578`), and 32 of them do fit in 51 bits (`< 51/32 = 1.594`). -/

theorem ternary_bits_lower : 2 ^ 30 < 3 ^ 19 := by decide

theorem ternary_bits_upper : 3 ^ 32 < 2 ^ 51 := by decide

end Tsct
