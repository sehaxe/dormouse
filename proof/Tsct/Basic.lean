/-
TSCT, the mathematical core.

Everything here is Lean 4 core only: `Rat` for the field, `Fin n → Rat` for
vectors, and a hand-rolled finite sum (`sumFin`) because core Lean ships no
`Finset`/`Fintype`.  A weight matrix `W : Fin d → Fin f → Rat` is approximated
by `r` ternary rank-1 terms with one real scale each,

    W ≈ Σ_{k<r} s_k • (u_k ⊗ v_k),    u_k ∈ {-1,0,1}^d,  v_k ∈ {-1,0,1}^f.

`outer s u v` is `u ⊗ v` scaled by `s`, one entry `s * u i * v j`; that is
exactly `TSCTLinear.effective_weight`'s `(u * s) @ v`.
-/

namespace Tsct

/-! ## A finite sum over `Fin n`

Core Lean has no `Finset`, so the sum is a structural recursion on the index
bound.  Every algebraic lemma below is an induction on that bound with `grind`
doing the `Rat` rewriting (`ring`/`linarith` are not part of core either;
`grind` is). -/

/-- Sum of a `Rat`-valued function over `Fin n`. -/
def sumFin : ∀ {n : Nat}, (Fin n → Rat) → Rat
  | 0, _ => 0
  | _ + 1, f => f 0 + sumFin (fun i => f i.succ)

@[simp] theorem sumFin_zero (n : Nat) : sumFin (fun (_ : Fin n) => (0 : Rat)) = 0 := by
  induction n with
  | zero => simp [sumFin]
  | succ n ih => simp only [sumFin]; rw [ih]; grind

theorem sumFin_add {n : Nat} (f g : Fin n → Rat) :
    sumFin (fun i => f i + g i) = sumFin f + sumFin g := by
  induction n with
  | zero => simp [sumFin]; grind
  | succ n ih =>
    simp only [sumFin]
    rw [ih (fun i => f i.succ) (fun i => g i.succ)]
    grind

theorem sumFin_smul {n : Nat} (c : Rat) (f : Fin n → Rat) :
    sumFin (fun i => c * f i) = c * sumFin f := by
  induction n with
  | zero => simp [sumFin]
  | succ n ih => simp only [sumFin]; rw [ih (fun i => f i.succ)]; grind

theorem sumFin_mul_const {n : Nat} (c : Rat) (f : Fin n → Rat) :
    sumFin (fun i => f i * c) = sumFin f * c := by
  induction n with
  | zero => simp [sumFin]
  | succ n ih => simp only [sumFin]; rw [ih (fun i => f i.succ)]; grind

theorem sumFin_congr {n : Nat} {f g : Fin n → Rat} (h : ∀ i, f i = g i) :
    sumFin f = sumFin g := by
  induction n with
  | zero => simp [sumFin]
  | succ n ih => simp only [sumFin]; rw [h 0, ih (fun i => h i.succ)]

theorem sumFin_nonneg {n : Nat} {f : Fin n → Rat} (h : ∀ i, 0 ≤ f i) : 0 ≤ sumFin f := by
  induction n with
  | zero => simp [sumFin]
  | succ n ih => simp only [sumFin]; exact Rat.add_nonneg (h 0) (ih (fun i => h i.succ))

/-- The product of two sums is the double sum (the missing `Finset.sum_mul_sum`). -/
theorem sumFin_mul_sumFin {n : Nat} (a b : Fin n → Rat) :
    sumFin (fun i => sumFin (fun j => a i * b j)) = sumFin a * sumFin b := by
  rw [show (fun i => sumFin (fun j => a i * b j)) = (fun i => a i * sumFin b) from
        funext (fun i => sumFin_smul (a i) b)]
  exact sumFin_mul_const (sumFin b) a

/-- `0 ≤ a * a`, proved by cases because core has no `mul_self_nonneg`. -/
theorem mul_self_nonneg (a : Rat) : 0 ≤ a * a := by
  rcases Rat.le_total (a := 0) (b := a) with h | h
  · exact Rat.mul_nonneg h h
  · have h2 : 0 ≤ -a := by grind
    have h3 : a * a = (-a) * (-a) := by grind
    rw [h3]; exact Rat.mul_nonneg h2 h2

/-- `sumFin (a - s·b)²` in expanded form (the complete-square engine). -/
theorem sumFin_quad {n : Nat} (a b : Fin n → Rat) (s : Rat) :
    sumFin (fun i => (a i - s * b i) * (a i - s * b i))
      = sumFin (fun i => a i * a i) + (-(2 * s)) * sumFin (fun i => a i * b i)
        + (s * s) * sumFin (fun i => b i * b i) := by
  rw [show (fun i => (a i - s * b i) * (a i - s * b i))
        = (fun i => a i * a i + (-(2 * s)) * (a i * b i) + (s * s) * (b i * b i)) from
      funext (fun i => by grind)]
  rw [sumFin_add, sumFin_add, sumFin_smul, sumFin_smul]

/-! ## Vectors -/

/-- `⟨u, v⟩ = Σᵢ uᵢ vᵢ` on `Fin n`. -/
def dot {n : Nat} (u v : Fin n → Rat) : Rat := sumFin (fun i => u i * v i)

theorem dot_comm {n : Nat} (u v : Fin n → Rat) : dot u v = dot v u := by
  unfold dot; exact sumFin_congr (fun i => by grind)

theorem dot_add_left {n : Nat} (u₁ u₂ v : Fin n → Rat) :
    dot (fun i => u₁ i + u₂ i) v = dot u₁ v + dot u₂ v := by
  unfold dot
  rw [← sumFin_add]
  exact sumFin_congr (fun i => by grind)

theorem dot_smul_left {n : Nat} (c : Rat) (u v : Fin n → Rat) :
    dot (fun i => c * u i) v = c * dot u v := by
  unfold dot
  rw [← sumFin_smul]
  exact sumFin_congr (fun i => by grind)

theorem dot_zero_left {n : Nat} (v : Fin n → Rat) :
    dot (fun _ => (0 : Rat)) v = 0 := by
  unfold dot
  rw [sumFin_congr (g := fun _ => (0 : Rat)) (fun i => by grind)]
  exact sumFin_zero n

/-- Squared Euclidean norm, `‖u‖²`.  There is no square root in this
development and none is needed: every statement is about the *squared* norm. -/
def normSq {n : Nat} (u : Fin n → Rat) : Rat := dot u u

theorem normSq_nonneg {n : Nat} (u : Fin n → Rat) : 0 ≤ normSq u := by
  unfold normSq dot
  exact sumFin_nonneg (fun i => mul_self_nonneg (u i))

/-! ## Matrices -/

/-- The rank-1 outer product, scaled: `(s • u ⊗ v) i j = s * u i * v j`. -/
def outer {d f : Nat} (s : Rat) (u : Fin d → Rat) (v : Fin f → Rat) : Fin d → Fin f → Rat :=
  fun i j => s * u i * v j

/-- Frobenius inner product of two `d × f` matrices. -/
def mdot {d f : Nat} (A B : Fin d → Fin f → Rat) : Rat :=
  sumFin (fun i => sumFin (fun j => A i j * B i j))

theorem mdot_comm {d f : Nat} (A B : Fin d → Fin f → Rat) : mdot A B = mdot B A := by
  unfold mdot; exact sumFin_congr (fun i => sumFin_congr (fun j => by grind))

theorem mdot_add_left {d f : Nat} (A₁ A₂ B : Fin d → Fin f → Rat) :
    mdot (fun i j => A₁ i j + A₂ i j) B = mdot A₁ B + mdot A₂ B := by
  unfold mdot
  rw [← sumFin_add]
  exact sumFin_congr (fun i => by rw [← sumFin_add]; exact sumFin_congr (fun j => by grind))

theorem mdot_smul_left {d f : Nat} (c : Rat) (A B : Fin d → Fin f → Rat) :
    mdot (fun i j => c * A i j) B = c * mdot A B := by
  unfold mdot
  rw [← sumFin_smul]
  exact sumFin_congr (fun i => by rw [← sumFin_smul]; exact sumFin_congr (fun j => by grind))

theorem mdot_sub_left {d f : Nat} (A B C : Fin d → Fin f → Rat) :
    mdot (fun i j => A i j - B i j) C = mdot A C - mdot B C := by
  rw [show (fun i j => A i j - B i j) = (fun i j => A i j + (-1) * B i j) from
        funext (fun i => funext (fun j => by grind))]
  rw [mdot_add_left, mdot_smul_left]
  grind

/-- `‖A‖² ≥ 0`. -/
theorem mdot_nonneg {d f : Nat} (A : Fin d → Fin f → Rat) : 0 ≤ mdot A A := by
  unfold mdot
  exact sumFin_nonneg (fun i => sumFin_nonneg (fun j => mul_self_nonneg (A i j)))

/-- The complete-square expansion of a residual, additive form. -/
theorem mdot_quad_add {d f : Nat} (a b : Fin d → Fin f → Rat) (s : Rat) :
    mdot (fun i j => a i j - s * b i j) (fun i j => a i j - s * b i j)
      = mdot a a + (-(2 * s)) * mdot a b + (s * s) * mdot b b := by
  unfold mdot
  rw [show (fun i => sumFin (fun j => (a i j - s * b i j) * (a i j - s * b i j)))
        = (fun i => sumFin (fun j => a i j * a i j)
            + (-(2 * s)) * sumFin (fun j => a i j * b i j)
            + (s * s) * sumFin (fun j => b i j * b i j)) from
      funext (fun i => sumFin_quad (fun j => a i j) (fun j => b i j) s)]
  rw [sumFin_add, sumFin_add, sumFin_smul, sumFin_smul]

/-- The complete-square expansion of a residual, in the readable form
`‖W − s·T‖² = ‖W‖² − 2s⟨W,T⟩ + s²‖T‖²`. -/
theorem mdot_quad {d f : Nat} (a b : Fin d → Fin f → Rat) (s : Rat) :
    mdot (fun i j => a i j - s * b i j) (fun i j => a i j - s * b i j)
      = mdot a a - 2 * s * mdot a b + s * s * mdot b b := by
  rw [mdot_quad_add]; grind

/-- `⟨s·u⊗v, t·u⊗v⟩ = s·t·‖u‖²·‖v‖²`.  This is what makes the denominator of
the optimal scale computable from the two factor norms. -/
theorem mdot_outer_outer {d f : Nat} (s t : Rat) (u : Fin d → Rat) (v : Fin f → Rat) :
    mdot (outer s u v) (outer t u v) = (s * t) * normSq u * normSq v := by
  unfold mdot outer normSq dot
  have hinner : ∀ i, sumFin (fun j => s * u i * v j * (t * u i * v j))
      = (s * t * (u i * u i)) * sumFin (fun j => v j * v j) := by
    intro i
    rw [show (fun j => s * u i * v j * (t * u i * v j))
          = (fun j => (s * t * (u i * u i)) * (v j * v j)) from funext (fun j => by grind)]
    exact sumFin_smul _ _
  calc sumFin (fun i => sumFin (fun j => s * u i * v j * (t * u i * v j)))
      = sumFin (fun i => (s * t * (u i * u i)) * sumFin (fun j => v j * v j)) := by
          apply sumFin_congr; intro i; exact hinner i
    _ = sumFin (fun i => (s * t) * ((u i * u i) * sumFin (fun j => v j * v j))) := by
          apply sumFin_congr; intro i; grind
    _ = (s * t) * sumFin (fun i => (u i * u i) * sumFin (fun j => v j * v j)) :=
          sumFin_smul _ _
    _ = (s * t) * (sumFin (fun i => u i * u i) * sumFin (fun j => v j * v j)) := by
          rw [sumFin_mul_const]
    _ = (s * t) * sumFin (fun i => u i * u i) * sumFin (fun j => v j * v j) := by grind

/-! ## Ternary vectors

`Ternary u` is the b1.58 alphabet.  Nothing below assumes the factors have
magnitude one: `0` is allowed (a pruned connection), and that is exactly why
the "nonzero entry" hypotheses appear. -/

/-- `u ∈ {-1,0,1}^n`. -/
def Ternary {n : Nat} (u : Fin n → Rat) : Prop := ∀ i, u i = 0 ∨ u i = 1 ∨ u i = -1

theorem ternary_neg {n : Nat} {u : Fin n → Rat} (h : Ternary u) :
    Ternary (fun i => -u i) := by
  intro i
  rcases h i with h | h | h
  · exact Or.inl (by grind)
  · exact Or.inr (Or.inr (by grind))
  · exact Or.inr (Or.inl (by grind))

theorem ternary_eq_pm_one {x : Rat} (h : x = 0 ∨ x = 1 ∨ x = -1) (hx : x ≠ 0) :
    x = 1 ∨ x = -1 := by
  rcases h with h | h | h
  · exact absurd h hx
  · exact Or.inl h
  · exact Or.inr h

/-- A ternary vector with a nonzero entry has strictly positive squared norm.
This is the hypothesis discharge behind every `q ≠ 0` below: the optimal scale
is undefined for a zero factor, and the factor is zero iff it has no support. -/
theorem normSq_pos_of_ternary {n : Nat} {u : Fin n → Rat} (ht : Ternary u)
    (h : ∃ i, u i ≠ 0) : 0 < normSq u := by
  induction n with
  | zero => rcases h with ⟨i, _⟩; exact Fin.elim0 i
  | succ n ih =>
    have hstep : normSq u = u 0 * u 0 + normSq (fun i => u i.succ) := by
      simp only [normSq, dot, sumFin]
    rcases ht 0 with h0 | h0 | h0
    · rw [hstep, h0]
      have htail : ∃ i : Fin n, (fun i => u i.succ) i ≠ 0 := by
        rcases h with ⟨i, hi⟩
        match i with
        | ⟨0, _⟩ => exact absurd h0 hi
        | ⟨k + 1, hk⟩ => exact ⟨⟨k, Nat.lt_of_succ_lt_succ hk⟩, by simpa using hi⟩
      have := ih (fun i => ht i.succ) htail
      grind
    · rw [hstep, h0]
      have hnn : 0 ≤ normSq (fun i => u i.succ) := by
        simp only [normSq, dot]
        exact sumFin_nonneg (fun i => mul_self_nonneg _)
      grind
    · rw [hstep, h0]
      have hnn : 0 ≤ normSq (fun i => u i.succ) := by
        simp only [normSq, dot]
        exact sumFin_nonneg (fun i => mul_self_nonneg _)
      grind

end Tsct
