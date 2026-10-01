# Review B — muon-plus.md / tsct.md, engineering-consequence pass

**Reviewer 2 of 2, independent.** Angle: *if the report is true and the project follows it, what
actually gets better.* Every citation in both reports is assumed correct — I did not re-fetch
arXiv, did not check the reference implementations, and make no claim about provenance. I
re-derived the numbers instead.

**Not done:** no GPU, no `cargo build`, no `cargo test`. No file other than this one was
touched. `research/papers/*.md`, the vendor crates and the tree are as the author left them.

**Read:** `docs/papers/muon-plus.md`, `docs/papers/tsct.md`,
`vendor/burn-fused/crates/burn-spectral/src/lib.rs`, `vendor/burn-fused/crates/burn-muon-plus/`,
`crates/dormouse-core/src/{param,loop_block,routing}.rs`, `crates/dormouse-train/src/{lib,optim}.rs`,
`docs/AB-PROTOCOL.md`, `benches/history.tsv`.

**Assumptions I am judging against, stated so they can be rejected:**

- A1. The 246 ms warm step (batch 8 × 512, depth 2, fp32, 9 195 854 params, aux off, `--no-engram`,
  `benches/history.tsv:70-96`) is the live operating point, and the 52.8 ms `retr` inside it is
  the real per-step retraction cost at `retract_every=1`.
- A2. The card is launch-bound (13.3% mean utilisation, 79% of samples ≤5%) and will stay that
  way for the 9.2M-parameter regime.
- A3. `retract_iters = 3`, `retract_every = 1` are the defaults the A/B queue will use
  (`train/src/lib.rs:160`).
- A4. Attention currently receives no gradient, so the 246 ms figure describes a step cheaper
  than a correct one. I price the A/B both ways in §3.

---

## 0. Verdict up front

The reports are **good documents that recommend the wrong actions in the order that matters.**
Both name the TSCT retraction as an unverified 22%-of-step fixed cost. Neither asks what that
cost is *made of*. When I count, the single largest removable item is not the batching the
`tsct.md` §4.4 recommends, and it is not even in the report's findings list: it is a σ_max
**power iteration** that the report §3 T3 calls "our own contribution" and "the reason NS is
viable here at all", and which is **provably redundant** against the Frobenius prescale the
optimizer already uses 130 lines away in the same vendor workspace.

Second, the recommendation the reports do make — wire up `retract_batched` — is **not safe to
follow as written.** Wiring it naively silently freezes every TSCT master.

---

## 1. `retract_batched`: what the syncs and launches actually cost

### 1.1 The shape and count of the work

`LoopBlock::retract_tsct` (`loop_block.rs:170-176`) walks `expert_ffns[].{gate_up,down}` plus
`out_proj`. `configs/small.toml` is `d_model=768, d_ffn=2048, n_experts=3, rank=64`, so each
`LinearLike` carries a `u:[in,k]` and a `v:[out,k]` (`param.rs:45-58`, both `768` and `2048`
already multiples of 4, so the cubek pad at `param.rs:46-50` is a no-op):

| factor | shape | count |
|---|---|---|
| `gate_up.u`, `down.v` | `[768,64]` | 6 |
| `gate_up.v`, `down.u` | `[2048,64]` | 6 |
| `out_proj.u`, `out_proj.v` | `[768,64]` | 2 |
| **total** | | **14** |

Every factor is tall, so `polar_orthogonalize` transposes to `[64, 768]`/`[64, 2048]` and the
NS works on the small side. `retract_every=1` ⇒ all 14, every step, forever.

### 1.2 Sync count — the report's "7" is right, and 5 of the 7 are in one loop

`polar_orthogonalize` (`burn-spectral/src/lib.rs:168-217`) has exactly three `into_scalar`
sites, and one is inside the power loop:

| line | site | iterations | syncs |
|---|---|---|---|
| `:186` | `v.div_scalar(vn.into_scalar::<f32>())` | `POWER_ITERS = 5` | **5** |
| `:197` | `vgv = v.mul(gv).sum().into_scalar::<f32>()` | 1 | 1 |
| `:198` | `vv = v.mul(v).sum().into_scalar::<f32>()` | 1 | 1 |
| | | | **7** |

**7 per factor × 14 factors = 98 blocking device→host round-trips per training step, every
step, with no counter anywhere.** For contrast, `burn-muon-plus` has **zero** `into_scalar` in
its entire hot path (`grep` over `lib.rs` + `fused_kernels.rs`: the only hits are inside
`#[test]` bodies at `fused_kernels.rs:228,361,372,383`). The optimizer runs the *same*
Newton-Schulz iteration on the *same 14 tensors* and is sync-free, because it normalizes with
a `[1]`-shaped tensor and broadcasts (`:137-138`). The retraction does the identical estimate
and reads it to the host five times.

### 1.3 Launch count — my own accounting

Counting `clone`/`mul_scalar`/`add`/`sqrt`/`clamp_min`/`sum` as one launch each and
`swap_dims` as one transpose copy:

| stage | launches |
|---|---|
| `swap_dims` canonical transpose (`:171-173`) | 1 |
| `g = m·mᵀ` (transpose + gemm, `:182`) | 2 |
| `v = g.sum_dim(1)` (`:183`) | 1 |
| power loop × 5 — `mul`,`sum`,`sqrt`,`clamp_min`,`div_scalar`,`gemm` (`:184-191`) | 30 |
| `gv = g·v` (`:193-196`) | 1 |
| `vgv` (`:197`) | 2 |
| `vv` (`:198`) | 2 |
| `m /= σ·1.05` (`:202`) | 1 |
| NS × 3 — `xx`(2) + `xx2`(1) + `poly`(3) + `combine`(3) (`:206-211`) | 27 |
| un-transpose (`:212-215`) | 1 |
| **per factor** | **68** |
| **per step (14 factors)** | **≈ 950** |

### 1.4 The batched path, counted the same way

`retract_batched` (`:275-292`) groups by exact shape, so the 14 factors collapse to **two
groups: `[768,64]×8` and `[2048,64]×6`** — no padding, exactly the property the doc comment
advertises. Per group the op count is the same as one scalar factor (same 30-launch power loop,
same 27-launch NS loop) but each op is a batch kernel over 8 or 6 slices:

| | scalar path | batched path |
|---|---|---|
| groups | 14 | **2** |
| launches / step | ≈ 950 | 2×68 = 136, + stack 2, + writeback 14×3 = 42 → **≈ 180** |
| syncs / step | **98** | **0** |

So batching is **5.3× fewer launches and all 98 syncs gone.** The report (`muon-plus.md`
§"batched vs loop") cites 4.1–4.8×, which is the right order for the launch half.

### 1.5 Cost model, calibrated against a number that is on record

`history.tsv` gives `opt = 43-47 ms` for the *same 14 tensors* (`routing.rs:96` sends
`(Expert|Readout, Factor)` to Muon; `(Head, Factor)` and every `Scale`/`DenseWeight` go to
`Rest`, so the Muon group is exactly the 14 retraction factors). Per tensor that is
8 NS iterations × 9 launches + norm 4 + div 1 + momentum 3 + ColRow 6 + update 3 ≈ **89
launches**, × 14 = **1246 launches in 43 ms ⇒ 34.5 µs/launch.**

That is one parameter fitted to one measurement, so it is a model, not a measurement. It has
one free check: the same model predicts the retraction at 950 × 34.5 µs = 32.8 ms of enqueue
plus 98 syncs. Solving the residual, 52.8 − 32.8 = 20 ms ⇒ **≈ 0.20 ms per sync**. That is a
plausible full queue drain on a launch-bound workload and it lands within 1% of the recorded
52.8 ms, so I am willing to predict with it.

| variant | launches/step | syncs | predicted `retr` | saving |
|---|---|---|---|---|
| **A. today** (scalar + power iteration) | 950 | 98 | **52.8 ms** (recorded) | — |
| **B. batched only** | 180 | 0 | **6–9 ms** | ~45 ms (85%) |
| **C. Frobenius prescale only** (§1.7) | 476 | 0 | **16–18 ms** | ~35 ms (66%) |
| **D. C + B** | 115 | 0 | **5–7 ms** | ~46 ms (88%) |

Three consequences, and the second is the one the reports miss:

1. **Batching is worth roughly 85% of the 52.8 ms and the step drops 246 → ~200 ms (−19%).**
   It is a fixed per-step tax, not a scaling cost — 8× the data moved it 52.8 → 64.6 ms
   (`history.tsv:94`), so it pays identically at 2k steps and at 100k. A 2k-step A/B run goes
   8.2 min → 6.7 min *before* the arm is even considered.
2. **"Remove 7 syncs" and "remove the launches" are not the same win, and the sync story
   understates it.** Variant C removes 100% of the syncs for a 66% saving; B removes the same
   syncs *and* 81% of the launches for 85%. Syncs are worth ~20 ms; launches are worth ~33 ms.
   Anyone who reads `tsct.md` §4.4 ("the claimed sync saving is not being collected") will
   size the prize at 20 ms and under-buy it by a factor of two.
3. **On a 9.2M-param model the FLOPs are irrelevant here, so any launch reduction is free.**
   Total NS arithmetic in the retraction: 14 × 3 × 13.1 MFLOP ≈ 0.55 GFLOP ≈ **80 µs** at
   7 TFLOP/s. Zero point eight percent of the 10 ms it costs. Whatever removes launches wins,
   full stop — which sets up §2.

### 1.6 `retract_batched` is not a drop-in, and following the report's advice as written would freeze the model

This is the finding I would most want acted on, and neither report has it.

`retract_batched(factors: &mut [&mut Tensor<2>], iters)` (`:275`) takes **raw tensors**. It
writes back with `*factors[i] = out.clone().slice(...).reshape(...)` (`:286-289`). It cannot
call `Param::from_mapped_value`, and it therefore cannot honour the invariant that
`polar_retracked` (`:154-161`) exists to enforce. That function's own doc comment (`:137-153`)
says what happens when you do not:

> *"the polar output is a non-leaf (`GradInBackward`), and burn-optim's step re-tracks
> `Requirement::Grad` only, so a stored non-leaf is silently downgraded to an untracked leaf —
> **the master freezes** and the fused op's backward sees a pruned parent."*

`polar_orthogonalize_batched` does **not** `.detach()` — only the scalar path's
`polar_retracked` does (`:156`). So `retract_batched` as written returns a live autodiff graph
of the whole NS iteration, stored into a `Param`. On the live trainer (`Autodiff<Cuda>`,
`BalancedCheckpointing`) that is a **silent master freeze** — the ADR-0011 SILENT class, on the
one mechanism whose entire purpose is keeping masters trainable, and invisible in the log
because the loss curve of a frozen model still descends.

The tests do not catch it, and structurally cannot:

- all four `retract_batched` tests (`:1327,1371,1396` + the identity assertion) run on
  `dev() = Device::ndarray()` (`:1241`) — **no autodiff device, so `is_require_grad` is never
  true and the freeze is unreachable**;
- the *one* tracking test, `retract_stays_tracked` (`:1315-1320`), builds
  `Device::ndarray().autodiff()` and calls `m.retract(3)` — **the scalar path**;
- the `retract_batched` microbench (`:1396-1438`) is also plain ndarray, and its own comment
  (`:1391-1394`) concedes "no device syncs exist here, so this measures pure
  compute/overhead" — which means **the 4.1–4.8× ratio is a CPU-dispatch measurement and can
  say nothing about the CUDA sync saving.** It is a *lower* bound on the CUDA win, not a
  measurement of it. (`muon-plus.md` then uses that ratio as a *correctness* argument — "a
  ratio on ops with identical gradients is consistent with the reference and is not evidence of
  a divergence". That is a category error: a performance number is not evidence about
  mathematics, in either direction.)

**What to do:** `retract_batched` needs `polar_orthogonalize_batched(...).detach()` plus a
mirrored `set_require_grad`, and its signature has to become `&mut [Param<Tensor<2>>]` (or the
stack/consume/`from_mapped_value` dance moves into `retract_tsct`). Add the missing test on
`Device::ndarray().autodiff()` asserting the master stays tracked after a batched retraction —
one test, and it is the gate the report should have recommended instead of the wiring.

### 1.7 The bigger fish: the σ_max power iteration is provably unnecessary

Both reports treat the power iteration as load-bearing. `tsct.md` T3: *"**our own
contribution to the retraction** and it is the reason NS is viable here at all."* The code
comment at `:174-180` justifies it: *"A Frobenius/sqrt(k) pre-scale does NOT bound sigma_max:
for a square n×n Gaussian matrix ‖X‖_F ≈ n but sigma_max ≈ 2√n (Bai-Yin), so sigma_max ≈ 2
stays above the basin and NS diverges (measured: polar([512,512], 3) -> max entry ~1e14)."*

Three things wrong with that, in increasing order of consequence:

1. **The code does not do a `‖X‖_F/√k` prescale.** `:202` divides by `σ·1.05` from the power
   iteration; the *Muon* path at `burn-muon-plus/src/lib.rs:137` divides by plain `‖X‖_F`. The
   comment is defending against a prescale that is not in the code. The measured 1e14 is a
   historical bug whose trigger is absent.
2. **Plain Frobenius is not just adequate, it is sufficient by Cauchy–Schwarz.** After
   dividing by `‖X‖_F`, `σ_max/‖X‖_F ≤ 1` — *always*, for any matrix, with no distributional
   assumption at all. Bai–Yin is about the un-normalised matrix and is irrelevant.
3. **The cubic's basin is exactly [0, 1], and Frobenius lands inside it.** For
   `p(s) = 1.875s − 1.25s³ + 0.375s⁵`:

   - `p′(s) = 1.875(s²−1)² ≥ 0` — `p` is monotone increasing on `[0,∞)` (verified
     numerically at s = 0, 0.5, 1, 1.4, 2);
   - `p(s) − s = 0.375·s(s²−1)(s²−7/3) > 0` for `s ∈ (0,1)` (verified at s = 0.1, 0.5, 0.9,
     0.99), and `= 0` at `s = 1`;
   - so `p` maps `[0,1] → [0,1]`, strictly increasing, with a **double root at `s = 1`**, i.e.
     cubic-order convergence to the manifold and no overshoot.

   Given `σ_max ≤ 1` after Frobenius normalisation, every singular value is driven
   monotonically to 1 and none escapes. **The iteration converges, unconditionally, with no
   power iteration and therefore no host read of the scale factor.** (The basin bound is 1, not
   the `σ < √3` the comment states — √3 is where the *cubic* blows up, `p(1.5)=1.44`,
   `p(2)=5.75`, `p(3)=63`. Frobenius gives 1. So does the comment's `1.05` safety factor earn
   anything? No: it makes `σ_max·1.05` slightly *worse*.)

**Consequences.** Replacing `:178-201` (5 sequential `[64,64]` gemms + a Rayleigh quotient, 36
launches and **5 of the 7 syncs** per factor) with the 4-launch `‖·‖_F` prescale the optimizer
already uses:

- per factor 68 → **34 launches**, 7 → **0 syncs** (variant C above: 52.8 → ~17 ms);
- it makes the retraction and the optimizer **the same algorithm on the same tensors**, which
  is what §4 of this review shows the owner requires;
- it deletes the thing `tsct.md` ranks as *our own contribution*;
- and it is a ~5-line diff inside `polar_orthogonalize`, with **no change to any call site** —
  so unlike `retract_batched` (§1.6) it cannot break the `Param`/tracking path, because
  `polar_retracked` still wraps it.

**Falsifier, and it is cheap:** on CPU, compare `ortho_error` after 3 cubic steps with a
Frobenius prescale against the same with the power-iteration prescale, at `[768,64]` and
`[2048,64]`, against `burn_sct`'s exact `safe_qr`. If the Frobenius variant's residual is
materially worse, restore the power iteration and **add a counter for it** (ADR-0011: an
unmeasured improvement in a hot path is indistinguishable from a no-op). Runs in seconds on
`Device::ndarray()`, needs no GPU, and discharges `tsct.md` §4.3 at the same time.

**Ordering:** do C first, then B. C is 4.6× smaller, cannot break anything, and takes the
saving from 85% to 88% of the ceiling by handing B a shorter per-factor op list to batch.

---

## 2. The dead branch: delete it, but the report's "fix" and the report's reason are both wrong

`muon-plus.md` D1 is correct that `nc * 4 < nr` at `burn-muon-plus/src/lib.rs:146` is
unsatisfiable — `nr`/`nc` are read at `:141` from `x` *after* the canonical transpose at
`:130-134`, so `nr ≤ nc` always, and `4nc < nr ≤ nc` has no solution. I confirm it. `D1` also
correctly identifies that the fused `ns_combine_cuda` call at `:154` is consequently dead in
production (I grepped the whole tree: `:154` is its only non-test call site; the test uses are
`fused_kernels.rs:272,331,358`).

**But the report then offers two fixes as equals** — *"either delete the branch or invert to
`nc * 4 > nr` and re-measure"* — and they are not equals. The FLOP count settles it, and the
FLOP count is not what the report thinks it is.

**The comment's arithmetic describes a different algorithm from the code's.** The comment at
`:142-145` compares factored as *"two `[c,c]@[c,r]` matmuls"* against direct as *"`[c,c]@[c,c]` +
`[c,c]@[c,r]`"*. In the code, `xx = x·xᵀ` is `[nr,nr]` — the *small* side — never `[c,c]`. The
comment's "direct" cost (`2c³`) is not a term that exists anywhere in the function.

**The correct forms, for the code as written, `x` being `[nr,nc]` with `nr ≤ nc`:**

```
direct   : xx = x xᵀ [nr,nr]        2·nr²·nc
           xx2 = xx·xx  [nr,nr]     2·nr³
           poly·x       [nr,nr]·[nr,nc]  2·nr²·nc
           ─────────────────────────────  4·nr²·nc + 2·nr³
factored : xx, t1 = xx·x, t2 = xx·t1    6·nr²·nc
ratio factored/direct = 3nc / (2nc + nr)  ≥ 1,  → 1.5 as nr/nc → 0
```

| `nr,nc` | direct | factored | ratio |
|---|---|---|---|
| 64, 768 (**our `[768,64]` factors**) | 13.1 M | 18.9 M | **1.44** |
| 64, 2048 (**our `[2048,64]` factors**) | 34.1 M | 50.3 M | **1.48** |
| 512, 8192 (the comment's case) | 8858 M | 12885 M | **1.46** |
| 768, 768 (square) | 2718 M | 2718 M | 1.00 |

So D2's *conclusion* is right — factored is more FLOPs, and always has been — but its
*stated* arithmetic (`factored = 3·(2nr²nc); direct = 2nr²nc + nr³`) drops a factor of 2 from
both terms of the direct form, and those stated numbers give **2.909** at `[512,8192]`, not
the **1.48** the same cell claims. The cell is internally inconsistent. It does not change the
verdict, and a reviewer checking the FLOPs would have to redo them.

**Now the deletion test, which is where the reports are quiet.**

- *Complexity that disappears:* the 24-line `if` branch (`:147-170`), the `ns_combine_cuda`
  call site, and a comment that is false in two independent ways (wrong Gram shape, and a 3.6×
  figure that `README.md` already retracts for lack of a device flush).
- *Complexity that reappears:* nothing. Today.

So delete it. **But hold on before deleting the kernel.** Count launches again (§1.5):
direct is **9** per NS iteration (`xx` 2 + `xx2` 1 + `poly` 3 + `combine` 3); factored with
`ns_combine_cuda` is **5** (`xx` 2 + `t1` 1 + `t2` 1 + fused combine 1). Over 14 tensors ×
8 iterations = 112 NS iterations per step, that is 1008 → 560 launches, a saving of **448
launches ≈ 15 ms** of a 43 ms `opt` — against **+90 µs** of extra FLOPs (0.14 GFLOP → 0.20
GFLOP at 7 TFLOP/s). On this card the factored+fused form wins by a factor of ~150.

**So the dead branch is not dead code that should be deleted. It is an unexploited lever
behind a wrong condition, and the condition is wrong because it encodes a shape heuristic
where the real decision is a machine property.** The correct gate is not `nc*4 < nr` (a
geometry proxy) and not its inverse (a geometry proxy for the opposite regime); it is *"is this
device launch-bound?"* — which `history.tsv` answers for the current box and which will answer
differently the day dormouse runs somewhere compute-bound. A condition that gives the right
answer on this card and the wrong answer on a H100 is a latent bug, which is exactly the
"3.6× measured on [8192,512]" claim's failure mode all over again.

**Recommendation, in the smallest form that is actually honest:**

1. Delete `lib.rs:146-170` and the `:142-145` comment. One clean commit. This is correct
   regardless of what follows, and it removes a false claim from the tree today.
2. Keep `ns_combine_cuda` in `fused_kernels.rs` — it is correct, tested against the tensor
   path, and now has no production call site (which is ADR-0019's own situation; it needs a
   counter if it is ever wired).
3. Re-add the factored form behind **one `bool` on `MuonPlusConfig`**, defaulted to `false`,
   surfaced through the existing `--set` seam. No new CLI surface. Run it as an A/B with a
   device flush, and record the row in `history.tsv` — that is the measurement whose absence
   produced the retracted 3.6× in the first place.
4. Do **not** flip the condition to `nc*4 > nr`. That would light up 100% of the NS traffic on
   this model, where the 1.44–1.48× FLOP penalty is irrelevant *and* the launch saving is real,
   so it might well win — but it would win for the wrong reason, and the next person to port
   this to a compute-bound card would inherit a silent 1.5× regression with a comment claiming
   a 3.6× *speedup*.

---

## 3. The minimum experiment that would decide TSCT, and what it costs

`tsct.md` §4.1 and `docs/AB-PROTOCOL.md:113` both say the TSCT-vs-dense A/B is unrun. I think
both name the wrong *first* experiment. In cost order:

### E1 — does 3 cubic steps actually reach the manifold at our shapes? (CPU, < 5 min, 0 GPU)

`tsct.md` §4.3 says this is the cheapest unrun check. I agree and would put it first,
extended to pin **both** prescale variants (§1.7) against `burn_sct`'s exact `safe_qr`:

```
[768,64], [2048,64] × {Frobenius prescale, power-iteration prescale} × 3 iters
  → ‖UᵀU − I‖_F / k   vs   ‖safe_qr(U)ᵀsafe_qr(U) − I‖_F / k
```

This decides whether the retraction is a retraction, and it decides §1.7 at the same time. It
costs nothing and it gates everything below: **if 3 cubic steps do not reach the manifold at
`[768,64]`, the retraction is not a retraction, and the whole A/B measures three arbitrary
programs.** Note the reference already exists in-tree (`burn-sct/qr.rs:509`), so this is a
test, not a project.

### E2 — is the existing ortho test measuring a retraction or a function? (CPU, < 5 min, 0 GPU)

`tsct_retract_restores_ortho` (`train/src/lib.rs:2026-2041`) scales `U` by 3.0 and asserts the
error falls. It proves the function maps a perturbed input somewhere orthonormal. It says
nothing about whether a *real optimizer step* leaves the manifold. Restructuring it to
"one real `optim.step`, then `max_ortho` before and after" is the same cost and is the
question the latch at `train/src/lib.rs:1331-1337` is actually relying on.

### E3 — do the factors drift at all? (GPU, ~3 min)

The retraction is 22% of a step. The project rule (§1.2) is A/B-or-death, and the *first*
question is whether there is anything to die of. `max_ortho` is checked every 500 steps; at
2k steps that is 4 points, which cannot distinguish a flat 6e-5 from a slow climb to 1e-3.

- **Cost:** one 100-step run with a controlled perturbation of known magnitude injected at
  step 50; assert the 500-step-cadence metric *moves*. If the monitor cannot see a 10× drift
  in 50 steps, the 1e-3 latch cannot see drift either, and the whole `T13` design (which the
  report calls "stronger than the paper's") is a latch that cannot trip.
- **100 steps × 246 ms = 25 s** of GPU. Call it 3 minutes with startup.
- **This is the highest information-per-second experiment on the list** and it needs no
  control, no arm, and no seed spread. If `max_ortho` sits at its 6e-5 floor for 2k steps, the
  answer to the A/B question is "`--retract-every 1000`", which `history.tsv:94` already
  measured as `retr=0.0` and a **188 ms** step against 240 — a free 22% that needs no
  experiment at all, only the evidence to justify it.

### E4 — the confound-free decomposition (GPU, ~1.5 h at the measured step time)

**The queue's arm 2 cannot answer the question it asks.** `--set use_tsct=false` changes at
least three things at once:

1. the parameterisation (`SpectralLinear` → `burn::nn::Linear`);
2. the retraction (53 ms of the step, and the *only* thing arm 2's own justification
   — "do the TSCT factors, the polar retraction and the quant machinery earn ~1000 lines" —
   names);
3. **the optimizer.** `routing.rs:96` sends `(Expert|Readout, Factor) → Muon`;
   `routing.rs:99` sends `(Expert|Readout, DenseWeight|DenseBias) → Rest`. So the dense arm
   runs **AdamW on the FFN weights** and the spectral arm runs **Muon+ on the factors**. The
   paper's own Table 1 puts Muon+ vs Muon at −0.41 to −2.02 loss — a *known, large, published*
   effect. The arm cannot separate a known large effect from an unknown speculative one.

`docs/AB-PROTOCOL.md:113` is aware of the parameter-budget confound ("a narrower FFN at the
same param budget") and **says nothing about the optimizer confound**, which is the larger of
the two.

The fix is cheap because the flag already exists:

| arm | flag | parameterisation | optimizer on the FFN | retracts? |
|---|---|---|---|---|
| **A** (control) | — | TSCT | Muon+ on factors | yes, 53 ms |
| **B** | `--factors-fallback` | TSCT | **AdamW on expert factors** (readout still Muon+) | yes |
| **C** | `--set use_tsct=false` | dense | AdamW on weights | no |

A vs B isolates the optimizer routing. A vs C is the protocol's arm. All three at **one batch
size** (else the eval window differs and §2.6 says the numbers are not comparable — batch 2
scores 20 480 B, batch 10 scores 102 400 B).

**Cost at the measured step time (246 ms, batch 8, warm):** 2 000 steps = 8.2 min/run.
3 arms × 3 seeds = 9 runs. The six spectral runs pay the 53 ms retraction; the three dense runs
do not, so ≈ 8 × 8.2 + 3 × 6.4 = **85 min ≈ 1.4 h**, plus eval overhead — call it **1.5 h**.
Three arms for 50% more than the protocol's two, and it is the difference between an answer and
a number.

**And the honest gate, which is the actual deliverable of this section:** this cost is
conditional on the 246 ms figure, and §3.2 of `AGENTS.md` says the attention arm has run no
backward in every run on record (`fused kda=3126/0`). The replacement figure — the tensor-op
KDA backward at 25.8 s/step — has **no committed log and no `history.tsv` row**. If it holds,
9 runs × 2 000 steps × 25.8 s = **129 h**, and the experiment is unfundable as specified;
cutting to 1 seed × 500 steps gives 10.7 h and a noise floor the protocol says cannot decide
anything. **So the step time is not bookkeeping for this A/B, it is the A/B.** One warm-step
measurement with a working attention backward, written into `history.tsv`, must precede any
planning of this queue — which is what `AB-PROTOCOL.md:95-101` already says and I am
endorsing rather than adding to.

### What the minimum is, stated as one sentence

**Run E1 and E3 first — together under ten minutes of wall clock and three minutes of GPU —
because between them they decide whether the mechanism has anything to be A/B'd about; the
1.5 h three-arm GPU experiment is the *follow-up*, and it is worth nothing if E1 says 3 cubic
steps are not a retraction or E3 says the factors never drift.**

---

## 4. The owner's rule: "deviations allowed ONLY if verified bit-for-bit against a reference implementation"

Applied to the reports' `DELIBERATE` verdicts. Assume every citation in both reports is right;
the question is only what the rule permits *given* that.

| verdict | deviation | rule says | why |
|---|---|---|---|
| **D3** | `lr·max(1,m/n)^{1/2}` vs paper's `lr·(m/n)^{1/2}` | **FORBIDDEN** | The reference is Jordan's `muon.py` and the two are not bit-identical for `m < n`. The report calls it "correctly sourced to Jordan" — **sourcing a deviation to a reference is not verifying it against that reference.** It is a one-line formula, so the bit-for-bit check is trivial; it has not been run. |
| **D4** | `ns_steps = 8` vs paper's 5 | **FORBIDDEN** | Reference is the paper: 5, everywhere, explicitly. A different document (the Qwen report) is not a reference *implementation*; it cannot discharge a bit-for-bit rule. Either revert to 5 or check against Qwen's actual code. |
| **D11** | routing is a *subset* of the paper's | **outside the rule's grammar** | The paper's routing is a hyperparameter choice, not a correctness property. There is no reference implementation to be bit-for-bit with about *which* parameters get which optimizer. The 40 s/step measurement is a machine fact and a fair justification. **The owner should exempt routing by name** — otherwise the rule will be argued about every cycle. |
| **T2** | **cubic (15/8,−5/4,3/8) × 3** vs Muon+ quintic × 5 | **FORBIDDEN** *(and see below)* | The two converge to different points by construction — the cubic reaches `UVᵀ`, the quintic deliberately does not. No tolerance, no iteration count, makes them bit-identical. |
| **T3** | **σ_max power iteration + Rayleigh × 1.05** vs the optimizer's Frobenius prescale | **FORBIDDEN, and the reference is in the tree** | `burn-muon-plus/src/lib.rs:137` performs plain Frobenius on the same objects, ~130 lines away in the same vendor workspace. Two NS implementations of the same object, in the same crate, disagreeing. This is precisely the failure the owner's rule exists to stop, and it is the one that costs 7/7 syncs and 39/68 launches per factor per step. |
| **T5** | per-entry `‖UᵀU−I‖_F / k` | **PERMITTED** | The metric is ours; no upstream defines it. The normalisation is a documented bug fix against a dated measurement, not a deviation from a source. |
| **T7** | QR-of-random init instead of SVD | **PERMITTED** | There is no dense matrix to take an SVD of. Nothing to deviate from. |
| **T13** | retraction cadence + one-way persisted fp32 latch | **PERMITTED** | Ours, no upstream. And it is the one that still needs E3 to justify its cost. |

**The test case, T2, and what it actually means.** The owner reads "cubic-vs-quintic NS" as a
mathematical deviation and the rule forbids it. But the rule binds deviations **from a cited
source**, and the honest resolution is not to change the math — it is to **delete the citation**
(T1/D5, the reports' joint #1 and #2 severity). The retraction is not a Muon+ component; it
occupies SCT's Eq. (5) slot and is implemented by our own method. Once "Muon+ §1" is off the
comment, the rule has no purchase on T2 — and the burden moves to a *different* rule, "verified
against a reference", whose natural oracle is `burn-sct`'s `safe_qr`. That check is a
**tolerance**, not a bit-for-bit test, because an approximate 3-step retraction can never be
bit-for-bit against an exact one.

**So the owner's rule and the two reports disagree about almost everything, and the
disagreement runs the wrong way.** Of six `DELIBERATE` verdicts: three are permitted (T5, T7,
T13 — all cases with *no* cited source to deviate from), three are forbidden (D3, D4, T2/T3),
one (D11) is outside the rule and needs an exemption. **The three the reports wave through as
"deliberate, documented, correctly sourced" are exactly the three the rule forbids, and the
one the reports rank as a severe bug (D1) is a performance claim the rule does not touch at
all.** Documentation and attribution are doing the work the owner's rule assigns to
verification. The fastest way to comply is a small diff in comments and two one-line
behavioural checks (D3, D4) — *plus* §1.7, which is the only forbidden item with a
performance cost attached.

---

## 5. What would still be unverified after every recommendation here is followed

Taking the reports plus §1.7 plus §2 plus §3-E1…E4 at face value, the following remain open,
and I do not think any of them is closed by a comment or a unit test:

1. **Whether the retraction helps training at all.** E1 closes *is-it-a-retraction*; E3 closes
   *is-it-needed*; E4 closes *does-it-win*. Only E4 answers the BPB question, and E4 is
   unfundable at 25.8 s/step. Nothing in this tree will settle it cheaply, and the honest
   position after all of the above is still §1.2's: the mechanism has never beaten its own
   removal.
2. **Whether the factor-quant forward helps.** `tsct.md` §4.6 is right and unaddressed by
   anything I propose. The latch fired on every fresh run before 2026-09-04, so no run in the
   archive ever engaged the post-fix path, and no run has ever measured it against fp32
   factors. The reports' severity rankings do not mention it; on the A/B-or-death rule it is
   the same class of debt as TSCT itself.
3. **Whether the step-time model in §1.5 is right.** It has one free parameter, fitted to
   `opt = 43 ms`, and validated against `retr = 52.8 ms` — two measurements, one number. It
   is good enough to *order* the fixes and it is worthless as a *quote*. A wrong estimate here
   costs nothing; a wrong estimate quoted in `history.tsv` costs a retracted-claim incident.
   Whichever fix is adopted, its `retr`/`opt` numbers need a real `--timers` row at a stated
   step index, and the count of *syncs removed* should get a counter (ADR-0011 — right now a
   fix that silently failed to apply would look exactly like a fix that worked).
4. **Whether the batched path is numerically identical on CUDA.** Its identity test runs on
   `ndarray` (`:1241`). Batched vs per-factor GEMM uses different tile shapes and therefore a
   different reduction order, so a `< 1e-5` max-diff bound verified on CPU says nothing about
   a `[8,64,768]` fp32 GEMM on sm_120. The retraction decides orthonormality, so this is the
   one place where "close enough" is not obviously acceptable.
5. **Whether the whole cost model survives the attention backward landing.** Everything in §1
   is priced against a 246 ms step in which `fused kda=<f>/0`. If the backward lands at
   25.8 s/step, 52.8 ms of retraction is 0.2% of a step, the batching work is a rounding
   error, and the *only* thing in these two reports worth doing is §4's compliance work. The
   precondition is the same one `AB-PROTOCOL.md:95-101` names and nobody has discharged.
6. **The seed determinism gap.** `AGENTS.md` §3.7 records 409 043 differing values between
   two runs under the same seed after `4b42b6d`. Every A/B number in §3 is downstream of that;
   "3 seeds per arm" is still *nearly* implementable, not implementable.

---

## 6. Top-3 findings

1. **The σ_max power iteration (`burn-spectral/src/lib.rs:178-201`) is provably unnecessary,
   is the largest single removable cost in the retraction, and is the item the owner's
   bit-for-bit rule forbids.** Frobenius normalisation gives `σ_max ≤ 1` by Cauchy–Schwarz;
   the cubic `(15/8,−5/4,3/8)` has basin exactly `[0,1]` with a double root at 1
   (`p′(s) = 1.875(s²−1)²`, `p(s)−s = 0.375s(s²−1)(s²−7/3) > 0` on `(0,1)`), so it converges
   unconditionally. Removing it is a ~5-line diff that takes the retraction from **7 syncs and
   68 launches per factor to 0 and 34** — 52.8 ms → ~17 ms, ~14% off a warm step — and it
   leaves the `Param`/tracking path untouched, which `retract_batched` does not.
   *(`tsct.md` T3 rates this as "the reason NS is viable here at all". It is not: the code
   does not use the `‖X‖_F/√k` prescale the comment defends against, and the live Muon path
   at `burn-muon-plus/src/lib.rs:137` uses plain Frobenius with no problem.)*

2. **Wiring `retract_batched` as the report recommends would silently freeze every TSCT
   master.** It takes `&mut [&mut Tensor<2>]` (`:275`), cannot call
   `Param::from_mapped_value`, and `polar_orthogonalize_batched` never `.detach()`es — so it
   stores a non-leaf where `polar_retracked`'s own doc comment (`:137-153`) says that silently
   freezes the master and prunes the backward. All four batched tests run on `Device::ndarray()`
   (`:1241`) where `is_require_grad` is never true; the single tracking test (`:1315`) tests
   the *scalar* path. The fix needs the signature change, a `.detach()`, and one
   `Device::ndarray().autodiff()` test. When it is right, it is worth the other half of the
   win: 52.8 → ~7 ms, step 246 → ~200 ms.

3. **Queue arm 2 (`--set use_tsct=false`) confounds the mechanism it is meant to judge, and the
   flag that de-confounds it already exists.** `routing.rs:96` puts TSCT factors in Muon+;
   `routing.rs:99` puts dense weights in `Rest`/AdamW. The arm therefore measures *AdamW-dense
   vs Muon+-spectral* — a published −0.41…−2.02 loss effect tangled with a speculative one.
   Adding `--factors-fallback` as a third arm costs ~25 min more than the protocol's own two
   and turns an uninterpretable number into two readable ones. Relatedly: the `nc*4 < nr` dead
   branch should be **deleted** (its comment is false twice over, and `muon-plus.md` D2's
   correcting FLOP count is itself wrong — `2.909` where it reports `1.48`), but the fused
   kernel behind it is worth 448 launches/step (~15 ms of `opt`) against 90 µs of extra FLOPs,
   so it should come back behind a one-`bool` A/B rather than stay deleted — gated on
   *launch-bound-ness*, not on a shape heuristic that gives the wrong answer on the next card.

**Not claimed:** that any of this improves held-out BPB. It improves step time, or it decides
whether BPB is worth measuring. On the project's own rule (§1.2), the mechanism is still
unjudged, and E1/E3/E4 in §3 are the shortest honest route to judging it.
