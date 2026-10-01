# Reviewer B — would following `research/papers/attnres.md` actually fix `burn-attnres`?

**Angle:** I assume every claim in the report is TRUE and ask only whether acting on it
produces a correct crate. Independent of reviewer A; I did not read their file.

**What I could verify without a GPU or a build:** the report's `file:line` citations (all
correct — I opened every one), the paper-vs-code formula claims D1/D2/D4/D5/D6/D7/D8/D9/D19
by direct reading, the numeric literals in §5 by recomputation in f64 and f32, the
`BlockAttnRes::step` control flow by line-by-line simulation, the crate's build/CI status
from files already in the tree, and four prior in-repo audits the report does not cite.

**What I could not verify — UNVERIFIED:**
* the current pass/fail of `streaming_fused_matches_tensor_path` (the one recorded-red
  CUDA test). Command: `cd vendor/burn-fused && cargo test -p burn-attnres --features
  cuda,autodiff --lib streaming_fused_matches_tensor_path`. Needs a GPU; not permitted here.
* whether the `1/√d` was ever a deliberate decision. `git log -S'powf(-0.5)' -- vendor/burn-fused/crates/burn-attnres`
  returns exactly one commit, `d1a76fe` ("self-contained repo: vendor burn-fused …"), i.e.
  **the crate's pre-vendor history is not in this repo at all**. The question is unanswerable
  from this tree; it needs `git log -S` against the upstream clone named in the crate's
  README badge (`github.com/sehaxe/burn-attnres`).

---

## 1. The gold vector: the assertion is sound, the diagnostic is wrong, and one literal is unreachable

### 1.1 The three `out[0]` values reproduce. The test is real and it is red today.

I recomputed §5's Test A (`d=4`, `h_0=[1,0,0,0]`, `h_1=[0,1,0,0]`, `w=[2,0,0,0]`) from scratch.

| convention | report's `out[0]` | mine, `eps=0` | mine, `eps=1e-5` | mine, f32, `eps=1e-5` |
|---|---|---|---|---|
| paper (`q·RMSNorm(k)`, no scale) | 0.9820138 | 0.9820138 | 0.9820124 | 0.9820124 |
| our norm, scale removed | 0.8807971 | 0.8807971 | 0.8807960 | — |
| **what we ship** (our norm + our scale) | 0.7310586 | 0.7310586 | 0.7310576 | — |

All three reproduce exactly at `eps=0`, and the shipped value is 0.250955 away from the
paper's — 25 000× the proposed `1e-5` tolerance, and it survives the f32 round-trip at
`|out0 − 0.9820138| = 1.4e-6`. `out[1] = 0.0179862` also reproduces (`1.4e-6` at
`eps=1e-5`, inside tolerance). **The report's stated separations (0.10122, 0.14974) and
its "≥ 0.10" claim are correct. This test does what §5 says it does.** Whatever else is
wrong with the report, this is the load-bearing part and it holds.

### 1.2 `score_0 = 4.000002` is unreachable, and the table mixes two different `eps`.

`score_0 = 2/sqrt(0.25 + eps)`. For any `eps ≥ 0` the supremum is **4.000000000**, attained
at `eps=0`. The report's literal, `4.000002`, is 2e-6 *above* the maximum reachable value.
True values: 4.0000000 (`eps=0`), 3.9999999 (1e-8), 3.9999920 (1e-6), 3.9999200 (1e-5).

Worse, the same table is internally inconsistent about `eps`: the "our norm" rows
(`1.999990`, `0.999995`) are exact to 6 dp **only at `eps=1e-5`** (which is what the code
hard-codes at `lib.rs:150/176/424/448`), while the "paper" row (`4.000002`, and every
`out` literal in Tests A and C) is the `eps=0` value. The `ref_attnres` signature at §5
takes `eps` as a parameter and the report never fixes it.

Consequence: the spread of the *correct* score literal across plausible `eps` is
`4.0000 → 3.99992 = 8e-5`, which is **8× the report's own `1e-5` tolerance**. An
implementer who follows §5's explicit invitation — *"anyone who changes the scale or the
norm convention sees which of the three numbers they now produce"* — and writes the
`score_0` assertion gets a **red test on a correct implementation**. The `out` literals
are `eps`-robust; the score literal is not, and it is the one the "self-certifying" pitch
points at.

### 1.3 The table has three labels over four cells, and the two middle cells are the same number.

This is the real defect. There are two independent knobs (norm ∈ {RMSNorm, L2} × scale ∈
{on, off}) and the report's table names only three outcomes. I computed all four:

| | `eps=0` | `eps=1e-6` | `eps=1e-5` |
|---|---|---|---|
| paper RMSNorm, no scale | 0.9820138 | 0.9820136 | 0.9820124 |
| **RMSNorm + `1/√d` scale** | 0.8807971 | 0.8807967 | **0.8807929** |
| **our L2 norm, no scale** | 0.8807971 | 0.8807970 | **0.8807960** |
| shipped (L2 + `1/√d`) | 0.7310586 | 0.7310585 | 0.7310576 |
| \|row 2 − row 3\| | **0.0** | 3.15e-7 | **3.15e-6** |

Rows 2 and 3 are identical to within 3.15e-6 — **300× below the proposed tolerance**, and
*exactly* identical at `eps=0`. This is not a `d=4` artifact; it is an identity:
`L2(x) = √d · RMSNorm(x)` and the scale is exactly `1/√d`, so *paper-norm-with-scale ≡
our-norm-without-scale, for every `d`, exactly.*

**Therefore D1 and D2 are a gauge pair.** The only observable is their product. Fixing
"the norm" while leaving the scale changes nothing numerically; fixing "the scale" while
leaving the norm changes nothing numerically. No output-only test can ever separate them.

So §5's Test A "**Catches D1, D2, D3**" is only true of the composite. And the diagnostic
is **backwards**: a fixer who corrects the norm and leaves the scale lands on 0.8807929,
which the table labels "our norm, our scale removed" — i.e. the table tells them D2 is
*still* broken when D2 is the one thing they just fixed. The report's own §3 counts
"D1/D2/D3 are one deviation expressed in six places" (`attnres.md:273`); §5 then pretends
to test three. The assertion still goes red correctly — but only ever as a single yes/no
on "paper or not", never as an attribution.

**What would fix it** (a two-line change to §5, not to the crate): either drop the
"self-certifying" table, or add the missing fourth cell and label rows 2 and 3 as the same
answer. Better still, pin the *composite* explicitly: assert the raw score against
`4.0/√(0.25+eps)`, which is the one quantity that is unambiguous.

### 1.4 Test C: the "ours today" column is exactly right; the stated margin is not.

I re-implemented `BlockAttnRes::step` (`lib.rs:270-304`) instruction for instruction and
drove it with §5's Test C inputs. Steps 2 and 3 return `[3, 1, 0, 0]` — the report's
column, exactly. Steps 4 and 5 return `[2.162551, 0.720850, 0.279150, 0.279150]` and
`[1.387763, 0.462588, 0.537412, 0.537412]` against paper values
`[2.8939895, 0.0176684, …]` and `[2.8939895, 0.0353368, …]` — maxdiff 0.73 and 1.20.

§5 says *"Steps 2 and 3 are ~1e-1 and ~1.4e-1 away today"*. Both are **0.982014**. The
report understates its own margin by ~7× (in the safe direction) and presents the number as
measured when §Scope says nothing here was measured. Fix the number or label it estimated.

Test C step 1 is vacuous: a single source returns `history[0]` before any scoring, by both
`depth_attend` (`lib.rs:105-107`) and `BlockAttnRes::forward` (`lib.rs:83-85`). The "✓" in
the table tests nothing.

### 1.5 Test D: correct, and I confirmed the separation is real.

Modelling `two_phase_attend` with Phase 1 scaled and Phase 2 unscaled vs neither, at
`S=2, N=2, d=8`: maxdiff **0.430736** on both layer outputs, 0.861473 on the returned
partial. The `1e-3` threshold is not marginal. D4 is real (`lib.rs:428` scales, `lib.rs:450`
does not) and Test D is a sound gate. The one loose sentence: §5 says `l2 == 1.0` and
`m2 == s2` "hold only by accident of the missing scale" — they are hardcoded
(`lib.rs:451-452`) and are identities of a single-key softmax that hold *with* a scale
too. They are not scale-sensitive assertions.

---

## 2. Fix order

The report lists 9 BUGs in delta-table order. Fixing them in that order produces a suite
that is green for the wrong reason twice. The actual dependency order:

**Step 0 — the `b_0` contract (D5 + D10). Before writing a single streaming test.**
`BlockAttnRes::step`'s own doc (`lib.rs:186`, `:262`) says `h` is *a layer output*. The
paper's first source is not a layer output; it is the token embedding, and it is a source
**forever** and never summed into a block. The API has no way to say so. So Test C — §5's
own "headline test" — is **unimplementable as written**: feeding `b_0` through `step`
violates the function's documented contract. And D6/D7 have no statement until the
contract exists. Four of the nine bugs (D5, D6, D7, D9) sit behind this one signature
change. The report identifies the bug and never names the blocker.

**Step 1 — D8, the double-incorporate.** Pure deletion (drop the second merge at
`lib.rs:299`, or make `incorporate` non-merging). No contract dependency, no paper
reference, arithmetic invariant. It must land before anything inspects persistent state.
Measured: `st.sum_exp` after step 4 with `S=2` is **2.161752** as shipped vs **1.774502**
with a single incorporate; the first *output* it changes is step 5 (`1.387763` vs
`1.690616`) — it is invisible in `out` at the boundary itself, because `lib.rs:299`
discards the return value.

**Step 2 — D6 + D7, now statable.** One expression at `lib.rs:282-294`. My simulation
confirms both exactly as described: step 2 returns the raw `b_0 + f_1` (D7), step 3 returns
`acc/sum_exp` = the previous block unchanged (D6, and with one source in the state that
softmax is the identity, so it is a plain residual at every boundary).

**Step 3 — D1 + D2 + D3 as one change, which *deletes* D4.** Removing `scale` removes
`lib.rs:109/155/180/410/428` and the `scale: f32` parameter from six kernel signatures
(`fused_attnres.rs` launch sites at ~:521, :554, :689, :711, :756, :765, :1142, :1165) plus
the backward. With no `scale` variable in scope, **D4 becomes unrepresentable** — there is
nothing to omit from Phase 2. The report treats D4 as an independent bug needing an
independent fix and a dedicated Test D; it is a *symptom* of the same decision. Do it
**last among the formula bugs**, because doing it first means every red Block-path test
reports a scale deviation and a source-set deviation simultaneously and you cannot tell
which. Test A goes green; Test D becomes a regression test for "no scale crept back", which
is a much cheaper thing to ask.

**Step 4 — D9, tail rule.** Independent; a design decision about `L mod S ≠ 0`, not a code
bug. Any time.

**Step 5 — what the report omits: the fused-vs-CPU composition bug, and the missing job that
would run it.** See Finding 1.

**Step 6 — D19** (`lib.rs:108` indexes `history[0]` at `n == 0`). One line. ADR-0011 wants
a loud error with a cause and an escape, not an index panic. Trivial, and free.

**Cross-cutting, and the reason there are 20 sites for 9 bugs.** The norm is open-coded
**nine** times (`lib.rs:150-152, 171-177, 420-425, 443-449`; `fused_attnres.rs` ~:150, ~:157,
~:383, ~:390, ~:438, ~:440, ~:603) and the `scale` is a parameter threaded through six
kernel signatures. The report's delta table treats these as 20 independent line-sites of
one deviation and proposes editing each. The fix that actually closes it is to extract one
`attnres_score(q, k)` and one `attnres_norm(k)`, delete `scale` from the kernel signatures
entirely, and let the tensor paths and the kernels call the same two functions. Then D4 is
not fixable-because-impossible rather than fixed-and-reintroducible, and D1/D2 have exactly
one place to be wrong.

---

## 3. What is still unverified after following the report completely

The class that survives is **composition and wiring** — everything that is correct
function-by-function and wrong assembled. Named:

1. **D10 / the `b_0` contract, permanently.** Tests A–H all hand-pick tensors into bare
   functions. A caller may call `depth_attend(&[f_1, …, f_L], w)` and silently drop
   `v_0 = h_1`, and every gold vector stays green. §5 labels Test B "**Catches D10**", but
   Test B is `q = 0` over five random tensors with no embedding in them — it cannot see
   D10, by construction. That is an over-claim in the report.

2. **The fused kernels get zero executed verification.** Tests E and F are the only two the
   report proposes that need CUDA, and `.github/workflows/fused-library.yml:160-176` runs
   `cargo test … --no-run` only. The workflow's own comment (`:177-180`) records that the
   `cuda-tests` job was removed and *"never ran"*. Following the report completely adds two
   CUDA tests that no job executes. 1821 of 2402 lines — **76% of the crate** — stay
   unverified, and the kind-(d) problem the report is fixing reappears one level up: tests
   that exist and are never run.

3. **Tolerance slack will read as a kernel check.** Test E keeps `1e-4` while re-pointing
   the reference. The fused path chunks at `CHUNK_G = 8` (`fused_attnres.rs:23`) and merges
   three levels of online softmax in f32; at the existing shapes — up to
   `(l,b,t,d) = (40,2,4,2048)` at `:868` — accumulated `f32` error is of the same order as
   `1e-4` with no bug present. A green Test E is therefore not evidence the kernel matches
   the paper. The only instrument that separates accumulation from divergence is an **f64
   host** reference, which this file *already has* for the merge
   (`merge_state_writeback_matches_host_reference`, `fused_attnres.rs:1016`). The report
   does not propose extending it to `depth_attend`, which is where the risk is.

4. **Logit magnitude after the scale is removed — a new failure mode the fix introduces and
   the plan does not close.** The report's own D1 note says the paper's `w_l` "must grow
   ~√d more than ours to reach the same sharpness". After the fix, nothing in the eight
   tests constrains or monitors logit magnitude, and **no proposed test uses a realistic
   `d`** — every gold vector is `d=4` or `d=8`, the regime where f32 is exact and the
   `1/d` is a toy. The tree's only guard is `clamp_min(1e-12)` on `sum_exp`
   (`lib.rs:251`), which converts an `exp` overflow into a silently enormous output — a
   **SILENT** fallback under ADR-0011's own taxonomy, and the report does not flag it. There
   is no `d`-sweep, no magnitude assertion and no saturation test in §5.

5. **Which of the two block mechanisms is authoritative.** `BlockAttnRes::step` and
   `two_phase_attend` are each pointed at a paper reference (C, D) and never at each other.
   §4 item 3 correctly notes "two functions in one crate implement two different block
   mechanisms", and open question 3 asks which should survive — but §5 does not gate on it,
   so a later edit can make them diverge silently. And `two_phase_attend` has a bug the
   report misses entirely — see Finding 3.

6. **`eps` is a free parameter of the oracle.** §5's own rule is "the expected values are
   literals transcribed from the paper" — but `ref_attnres` takes `eps`, and the report
   never fixes it, while the code hard-codes `1e-5` in four tensor sites and every kernel.
   The paper's ε is unspecified (Zhang & Sennrich use 1e-6/1e-8). The literals happen to
   hold across `eps ∈ [0, 1e-5]`, and nothing records that as a decision.

7. **D13, the per-sublayer `w_l`.** The report classes it BENIGN and declines to test it.
   But §5 of the paper is the *only* hyperparameter statement in the paper, and
   `BlockAttnRes::new(d, block_size)` gives an entire block one query vector — a different
   model. It is a wiring question that no function-level gold vector can reach, and it is
   exactly the kind of thing that survives a green suite.

---

## 4. The `1/√d` decision — does a scale factor qualify as bit-for-bit-preserving?

**My reading: no.** Bit-for-bit means the output bit patterns match the reference. A
multiplicative `1/√d` on every logit is a **different function**, not a reassociation, an
FMA contraction or a reduction-order change. At the paper's own `d=4` it moves `out[0]`
from 0.9820138 to 0.7310586 — 0.25 absolute, ~1.5e8 `f32` ULPs at that magnitude.

**The strongest argument available is not a bit-pattern argument, and it is weaker than the
report presents it.** The report states (D1) that since `w_l` is free, `softmax(d^{-1/2}w·k)`
is reachable by the paper's class with `w → √d·w`. That is true and it is not nothing. It is
*expressivity*, not fidelity, and it is silently false in three places:

* **the gradient, not the function class.** `∂L/∂w` is scaled by the same `1/d`, so the
  effective learning rate on the pseudo-query is d× smaller. The paper's Fig. 5 training
  dynamics do not transfer, and "reachable at convergence" says nothing about the trajectory
  the model actually takes. This is the whole substance of the D1 note, and it argues
  *against* keeping the scale, not for it;
* **the paper's best variant is out of reach for this crate anyway** (§4 item 5,
  input-dependent query, 1.731 — the crate's `query: Param<Tensor<1>>` cannot express it),
  so the equivalence argument does not buy the crate the thing that would make it faithful;
* **any constraint on `w` breaks it.** A norm cap, quantisation, or a low-rank factorisation
  of the query — all of which a production model does to save `d` params per layer, and the
  last of which is the house style here — does not commute with a global rescale. The
  equivalence holds only for an unconstrained `w`.

And the reason this is dangerous rather than merely wrong: **the deviation is
init-invariant.** At the mandated `w_l = 0` both give exactly uniform `α`, so it is
invisible on the loss curve at step 0 and surfaces a thousand steps later as a different
effective temperature. The report says this (D1, correctly). An init-invariant deviation
that is a `d`-fold change in temperature is precisely the ADR-0020 shape: right at init,
wrong later, invisible to every test that only checks init.

**The honest options, in order of cost:**

* **(a) Delete the scale, adopt the paper's formula.** One change. It also deletes D4 as
  unrepresentable, and the training cost is **zero**, because no run has ever used the
  crate — verified: no `Cargo.toml` under `crates/` references `burn-attnres` (grep over
  every `.toml`), and the workspace `exclude` at `Cargo.toml:12` keeps `vendor/burn-fused`
  out of the product build entirely. This is the only option that satisfies the owner's rule
  as written.
* **(b) Keep it, and request the exception explicitly and in writing** — with the
  expressivity argument attached, *and* rename Test A from a fidelity gate to a
  characterisation test with literal `0.7310586` and the word "ours" in the name. Otherwise
  the next person reads a green 0.7310586 as "we match the paper up to tolerance", which is
  the retraction this project keeps having to undo.
* **(c) The status quo.** Which is what the report objects to.

**A rule problem the owner should see, which the report does not raise.** The rule is
"deviations allowed ONLY if verified bit-for-bit against a reference implementation". For
this paper **there is no reference implementation** — the report establishes that itself
(§1: seven blobs, no source, no PyPI, no tags; the arXiv HTML build is a stub). Applied
literally, the rule makes *every* deviation impermissible, **including the five the report
classes MATCH** (D14–D18). The rule needs a second clause: *"...or against a pinned literal
transcription of the paper's only executable specification."* §5's Test A is exactly that
clause. Without it the standard cannot be met by anyone, and the `1/√d` decision is being
made under a bar that has no achievable form for this paper.

---

## 5. Is `burn-attnres` worth the work?

**Facts, all verified from this tree.** 2402 lines (517 `lib.rs` + 1821 `fused_attnres.rs`
+ 64 bench). Fate class **REFERENCE** (`docs/library-crate-fate.md:62`). Zero incoming edges
from `crates/dormouse-*` — grep over every `.toml` finds none, and `Cargo.toml:12` excludes
`vendor/burn-fused` from the workspace. It *is* compiled by CI
(`.github/workflows/fused-library.yml:163`, `--no-run` only). It **fails its own recorded
CUDA configuration**: `research/2026-09-27-fused-build-matrix.md:50` — `FAIL 11+1f/2ig` —
with the panic quoted at `:179-186`. It is already labelled **BROKEN** twice
(`research/2026-09-27-fused-inventory-attention.md:90`, `:404` of `adopt-vs-port.md`) and
**UNVERIFIED (d)** once (`oracle-audit.md:65`, "also self-recorded BROKEN").

**Cost of the report's plan, done properly:** one contract change + four source-set fixes +
one state fix + one scale deletion + six kernel signature changes + eight tests + a CUDA
re-validation that **no job in this repo can currently perform**.

**Benefit:** one A/B arm named at `docs/PLAN-minimal-core.md:90` §M2, never run, on a
mechanism whose paper-reported margin is +0.005 on the best ablation set and whose own
multihead variant is a wash (1.752 vs 1.746). AGENTS.md §1.2: a tie deletes the mechanism.

**Cost/benefit: the fused half is 76% of the crate, has no callers, fails its own gate, and
contains none of the six tensor-path bugs.** Following the report in its current order is
the most expensive way to reach the least useful destination.

**The smaller correct step, in order:**

1. **Delete `fused_attnres.rs`.** 1821 lines. It fails its own configuration, has no
   callers, and the CPU tests are hermetic (`Device::ndarray()`, `lib.rs:311-314`). Six of
   the nine report bugs are in the 517-line `lib.rs` half — the whole paper-fidelity
   question lives there.
2. **Fix `lib.rs` only:** the `b_0` contract, D8, D6/D7, the scale deletion, D19. Roughly
   60 lines changed, 8 CPU tests, no GPU, no CI churn, no kernel signatures. This is the
   complete fidelity fix.
3. **Only if §M2 is actually scheduled:** re-add the fused path *behind* the tensor path,
   with an **f64-host-referenced** parity test (extend `merge_state_writeback…` rather than
   comparing kernel to kernel) and a **running** CUDA job — which means re-registering the
   runner the workflow comment at `fused-library.yml:177-180` says was removed.
4. **If §M2 is never scheduled, the honest end state is deletion**, and `PLAN-minimal-core.md:90`
   should be amended to name `gr.rs` and the inline ReZero as the residual arm.

Step 2 alone converts the crate from "fails its own recorded configuration" to "passes, and
every formula traceable to a paper equation", for less work than the report's Test E/F
alone.

---

## 6. Top-3 findings

**1. The report's verdict sentence is contradicted by four in-repo documents, and the
contradicted claim is the one that matters most here.** `attnres.md:13-15` — *"The fused
CUDA kernels are internally consistent with the tensor path (so the existing parity tests
are green)"* — and `attnres.md:311-312` — *"the residual-state read→write hazard is
genuinely closed in the current tree"*. Meanwhile:
`research/2026-09-27-fused-inventory-attention.md:70-93` records
`streaming_fused_matches_tensor_path` **FAILED**, `step 3: maxdiff 0.83`, *"reproducible,
same step and magnitude across runs"*, and localises it: *"The isolated components pass
(`source_score_fused_matches_tensor`, `merge_fused_matches_tensor`), so the bug is in the
**composition** … not in either kernel alone."*
`research/2026-09-27-fused-build-matrix.md:50` + `:179-186` record the same failure at
`maxdiff 0.94786954`. `adopt-vs-port.md:404` and `oracle-audit.md:65` both say BROKEN /
UNVERIFIED (d). The report cites **none of them** (grep for `2026-09-27` in
`attnres.md`: no hits) and does not list `streaming_fused_matches_tensor_path` among the
five tests it audits in §3.1.

This is not an orthogonal defect: the recorded failure is a CPU-vs-CUDA divergence **in the
`BlockAttnRes` streaming path** — the same path D5, D6, D7 and D8 live in — and it is a
*composition* failure, which is the one class §5's plan does not add a single test for.
Worse, that test is **not `#[ignore]`d** (`fused_attnres.rs:937-938`) and **no job runs
it** (compile-only CI). So following the report completely leaves a known-red test red,
adds two CUDA tests that also never execute, and leaves 76% of the crate with zero executed
verification. **Its current status is UNVERIFIED** (no GPU permitted); the command is in the
header. If it is still red, the correct first action is not any of the report's 9 fixes.

**2. Test A is a 3-label table over a 4-cell space, and the two middle cells are the same
number.** `L2(x) = √d·RMSNorm(x)` and the scale is exactly `1/√d`, so "paper norm with the
scale" and "our norm without it" agree to 0.0 (`eps=0`) / 3.15e-6 (`eps=1e-5`) — below the
proposed `1e-5` tolerance. **D1 and D2 are a gauge pair and no output test can separate
them**; fixing one without the other is a numerical no-op. §5's "self-certifying" claim
therefore mislabels the one case a fixer is most likely to produce (fix the norm, leave the
scale → the table says the norm is still wrong). Separately, the table's `score_0 = 4.000002`
is **above the supremum 4.000000000** reachable for any `eps ≥ 0`, and the paper row was
computed at `eps=0` while the "ours" rows used `eps=1e-5` — so the `eps` of the oracle is
never fixed, and the correct score literal's spread (8e-5) is 8× the report's own
tolerance. The `out` literals are fine; the diagnostic is the problem.

**3. Test G — §5's "cheapest real gate in the set", the one the report says settles D8 "in
one line" — is aimed at the wrong step and asserts a value the code cannot produce.**
`st.sum_exp` after step 2 with `S=2` is **1.000000**, not the 3.0 the report predicts
(`attnres.md:533-536`), because `incorporate` short-circuits on `!st.started`
(`lib.rs:215-221`) and the first block is never merged, only assigned. The gate
`sum_exp == 2.0` would therefore test **D5** (the embedding folded into block 1), not D8.
D8's first occurrence is **step 4** (the boundary of block 2): `sum_exp` = **2.161752** as
shipped vs **1.774502** with a single incorporate, and its first effect on any `out` is
**step 5** (1.387763 vs 1.690616). The report's own §SPECULATION admits nothing was
executed; this is precisely the cost of not executing it — a headline gate aimed at a step
where the defect has not yet happened, quoting a number the arithmetic forbids.

**Bonus, and it is a bug the report does not have:** `lib.rs:476` does
`partial = partial.add(h1)`, accumulating the **attention output `h_l`** into the block
partial. The paper accumulates the **layer output `f_l(h_l)`** — Fig. 2 line 36
(`partial_block = partial_block + mlp_out`) and Alg. 1 line 14
(`b_n^i ← b_n^{i-1} + f_l(h_l)`). The doc comment at `lib.rs:398` claims the return is
*"the sum of the block's outputs"*, which is false. This is in `two_phase_attend` — the
function §4 item 3 calls the one that *"tracks Eq. 6 + Alg. 1 reasonably closely"* and
open question 3 nominates as the survivor. Its block representations are wrong at the
source, before the scale, the norm, or anything else. Report it as **D10**; it is a tenth
BUG, and it is in the function the report would keep.

*Minor, for the record:* `attnres.md:319` cites `crates/burn-attnres/README.md:3-8`. The
path does not exist; the file is
`vendor/burn-fused/crates/burn-attnres/README.md:3-8` (AGENTS.md §2.5: `vendor/` is
canonical). And §5's "should stay red until the deviation is removed" conflicts with
AGENTS.md's "failing test = bug in code — fix code, never skip/delete tests"; the correct
form is `#[ignore]` **with a named ADR reference**, or fix the code and make it green.

---

Status: complete
Integrity: suspect
Contract: unknown
