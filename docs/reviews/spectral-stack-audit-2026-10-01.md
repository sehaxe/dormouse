# Audit: the spectral stack — `burn-spectral`, `burn-sct`, and the seam between them

> **WHY THIS FILE HAS A NEW NAME (2026-10-01, landing of `wt/spectral-audit`).**
> This document was written as `docs/reviews/spectral-audit-2026-10-01.md`
> and landed as `3e1812b` (committed 00:55:09). The twin lane's `7811191`
> was authored 35 s EARLIER (00:54:34) and committed 97 s later (00:56:46),
> wrote to **the same path** with a different document (the formula audit of
> `burn-spectral`, findings 1–3), and replaced these 494 lines with 218.
> Nothing was merged and nothing was flagged: two independent lanes, one
> filename. The content is now this file, restored byte-for-byte from
> `3e1812b`, under the name its own H1 gives it; the formula audit keeps the
> original path and is the deeper document. **The two are different subjects**
> — this one is `burn-sct` and the seam between the crates, that one is the
> NS retraction's formulas — so a union would have been a 1000-line document
> about two things. Verified with `git show 3e1812b:…` (494 lines, identical).

**Pass 2026-10-01**, opened at `8f97411`; the tree moved under it
(`d8a21b9`, `9046d0d`, `69dff4e`, `42cae52` landed during the pass — §3.5
addresses the retraction decision in `d8a21b9` directly). Read-only except for one new instrument
(`tools/polar_probe.rs`) and these two files. Site-by-site inventory with
file:line, source claim and honesty tier:
[`docs/research/2026-10-01-spectral-inventory.md`](../research/2026-10-01-spectral-inventory.md).

**What was run.** One `rustc -O` single-file probe (0.4 s, no cargo, no burn, no
vendor target dir) that transcribes the retraction and the Stiefel metric in f32
and prices them at `small`'s real factor shapes. Its only external anchor is
`burn-spectral/src/lib.rs:188-191`'s own doc-comment table, and it **reproduces
it** — see §3.1. Everything else in this document is read out of the tree, out
of the three 2k-step logs from last night, and out of AGENTS.md.

**What was NOT done, and why.**

- **No GPU run.** A 2k-step `train` was live for the whole pass
  (`pgrep -ax train` → PID 3003146, `--ckpt-name first_run_2000--seed-2-`, 83%
  GPU). §1.5: one heavy thing at a time. The one measurement this audit most
  wants — a run with the retraction switched off — is therefore a *named
  command* in §5, not a number.
- **No `cargo test -p burn-spectral` / `-p burn-sct`.** A cold build of
  `vendor/dormouse-fused` is 686 packages / 41 GB / ~1400 s, and the 2026-09-29
  memory entry records the freeze that five parallel agents caused by each
  launching one. The probe exists precisely so that the audit's arithmetic
  needs no build. **Consequence: every in-crate test name quoted below is quoted
  from the source, not observed green.** The two I care about most
  (`retraction_holds_the_manifold_at_rank_64`, `gpu_retract_matches_cpu`) are
  the ones I would run first when a cell is free.

---

## 0. Verdict table

| # | finding | severity | one line |
|---|---|---|---|
| **F1** | **`burn-sct` is not the foundation layer under `burn-spectral`. There is no seam: zero call sites.** | **high (premise)** | the two crates are independent implementations of the same idea; the trainer runs `burn-spectral`'s tensor-op Newton-Schulz, and `burn-sct`'s Householder QR + 6 CUDA kernels have never executed in a run |
| **F2** | **The retracted-head drift is real and the latch would fire in ~8–60 steps without the retraction** | **high (proves the retraction earns its keep)** | the `lm_head` factor trains with **AdamW** (`routing.rs:98`), whose per-step `‖ΔU‖_F = 2.2e-2` is 28× the Muon+ factors' `8e-4`; crossed against the `1e-3` one-way latch that is 8 steps (systematic) to 63 (random walk) |
| **F3** | **`--retract-iters 3` is a cliff, not a dial** | **high** | 3 NS iterations leave `2.3e-8`/entry at a 1:1 spectral spread, `1.1e-5` at 2:1, **`1.9e-3` at 3:1 — already past the latch** — and `8.4e-2` (84×) at 10:1. Nothing measures the spread, and the latch threshold is 2× inside the failure point. **The "6e-5 floor" that justifies the 1e-3 threshold is wrong and is written in three places** (AGENTS §2.3, `param.rs:200-204`, `lib.rs:1344-1347`) |
| **F4** | **The retraction is 5.3% of a warm step and 0.18% arithmetic** | **medium (the cost, measured)** | 25.7–26.0 ms of a 487 ms step, ≈880 launches, 0.61 GFLOP = 46 µs at cuBLAS fp32 → **~540× launch overhead**, 29 µs per launch |
| **F5** | **The signal that would show the retraction degrading is computed every 500 steps and thrown away** | **medium** | `model.max_ortho()` (`lib.rs:1350-1357`) only *prints* when it exceeds the latch. A retraction silently degrading from `2e-8` to `2e-6` is indistinguishable from a healthy one in every log ever written |
| **F6** | **The sync-free batched retraction is off by default and never used** | **medium (free win)** | `retract_batched: false` (`lib.rs:175`); `retr_arm=batched:0` in every log. At `small` it is 4 calls instead of 16 (3 distinct shapes) — the same 5.3%, ~5× fewer launches |
| **F7** | **`qr_householder` is the init, it host-reads 7 scalars per column, and it is the documented source of the cross-process TSCT residue** | **medium** | `burn-spectral/src/lib.rs:689`, called at `:462`/`:469`. AGENTS.md 3.7 attributes the 1-ULP init difference to nondeterministic reductions inside it, at a *random iteration* 0…55 |
| **F8** | **The retraction doc's cost claim is 8× stale** | **low** | "replaces the CPU QR of SCT, which cost 40-50% of a step" (`lib.rs:19-20`) vs **5.3% measured tonight**. Same shape as the retracted "the optimizer is 77% of a step" |
| **F9** | **`model.rs:416-417` claims the un-retracted forward "degrades into NaN"** | **low, but it is a §1.4 claim** | arithmetic says it cannot: per-row-absmax factor quantisation is **equivariant** to per-row scaling, so drift can only reach all-zero weights (CE → `ln 256` = 5.5452, finite). No run, no log, no citation |
| **F10** | **745 lines of f64 SVD with zero tests, in a crate with zero users** | **low (dead crate)** | `host_svd.rs` has 0 `#[test]`; `cholesky_host_par`, `qr`, and 5 CUDA kernels are dead (~300 lines) |
| **F11** | **Two gates in `burn-sct` cannot fail on a real error** | **low** | `retract_restores_ortho` and `sign_correction` assert `|diag(UᵀU) − 1| < 0.1` |
| **F12** | **Three names for one linear, one of them wrong in a file the trainer owns** | **low (§1.7)** | `param.rs:1` says "TSCT linear via **burn-sct** `SpectralLinear`"; the import at `:6` is `burn_spectral::SpectralLinear`; `burn-spectral/src/lib.rs:1` calls itself **`burn-tsct`** |
| **F13** | **The `0.7` ternary dead zone has no source and no gate** | **low** | `lib.rs:54` attributes it to a "burn-es convention". It decides which ~30% of entries are zeroed, in every ternary forward, and is a **(c)**-tier number in an (a)-tier function |
| **F14** | **The landed `--retract-every 4` decision (`d8a21b9`) rests on a half-the-spread, n=1 difference — and it cuts the retraction to ~6.5 ms, which nobody claimed** | **high (it landed 20 min into this pass)** | 6.329 vs 6.387 is one seed each with a 0.108 three-arm spread. The *saving* (32 min over 100k steps) is the defensible half; the *quality* half is not evidence. See §3.5 |

---

## 1. The seam (brief item 2): there isn't one, and that is the finding

The brief's premise — "TSCT is a superstructure over SCT, `burn-sct` is the
foundation layer under `burn-spectral`" — is **false in this tree**, and the
falsehood is load-bearing enough to be worth the first section.

```
crates/dormouse-core/src/param.rs:6      use burn_spectral::SpectralLinear;
crates/dormouse-core/src/model.rs:418     model.retract_tsct(iters)
vendor/.../burn-spectral/src/lib.rs:662  SpectralLinear::retract
vendor/.../burn-spectral/src/lib.rs:154  polar_retracked
vendor/.../burn-spectral/src/lib.rs:208  polar_orthogonalize        <- the live path
```

That is the whole of it. Every other edge:

| edge | status |
|---|---|
| `burn-spectral` → `burn-sct` | **`[dev-dependencies]` only** (`burn-spectral/Cargo.toml:33`) plus one example, `examples/tsct_diag.rs:61-110`. Zero `src/` call sites. |
| `burn-sct` → `burn-spectral` | none, in either direction |
| `burn-sct` → the trainer | **none.** No crate under `crates/` depends on `burn-sct`. |

So the question "is the composition what each crate's docs claim?" has a blunt
answer: **the composition is nil, and both crates document themselves as if it
were not.** `burn-sct`'s README is the honest one — it says, in its first
paragraph, "**Not in the dormouse build, and duplicated in it** … Recommendation:
DELETE" — and `burn-spectral`'s header says the opposite, quietly, by calling
itself `burn-tsct` and never mentioning that a crate named `burn-sct` with the
same `SctLinear` shape exists two directories away.

**This is exactly the failure mode the brief was worried about** — a shared
misunderstanding surviving a single-crate audit — and it is worse than a shared
math mistake, because it is a shared *attribution* mistake: the retraction the
trainer runs is credited to an implementation that never runs. Three concrete
instances, all of which will be believed by the next reader:

1. `param.rs:1` — the file that *owns* the TSCT linear says it comes from
   `burn-sct`. It comes from `burn-spectral`. (F12)
2. `burn-spectral/src/lib.rs:19-20` — "GPU-only Newton-Schulz polar retraction
   for the masters (**replaces the CPU QR of SCT**, which cost 40-50% of a
   step)". True as history, false as cost: it is 5.3% now, and the "CPU QR of
   SCT" it replaced is in a crate the trainer has never called. (F8)
3. This brief's own premise, which I was asked not to argue with. I am not
   arguing with it: I measured the composition and there is none.

**What the two crates actually share:** a name (`SctLinear` vs
`SpectralLinear`), a forward (`y = (x@U)·s @ Vᵀ`), a retraction
(`QR` vs `NS`), and a `from_dense` SVD. Nothing else. They are two
transcriptions of SCT 2604.00733 that were never reconciled, which is also why
they disagree about the retraction's *kind* while both citing the paper.

**The one thing worth keeping from `burn-sct`** is a kernel, and it is
unreachable for a named reason: `qr_cuda::retract_cuda` (K8) does the whole
retraction in **one launch** after a `k×k` host Cholesky, versus the trainer's
~54 launches per factor. It is gated on `is_cuda::<B>()`, which compares `B`'s
`TypeId` against the **bare** `CubeBackend`; the trainer's tensors live on
`Autodiff { device: Cube(Cuda(0)) }`, so the gate is false. And
`SpectralLinear::retract` has **no backend type parameter at all**, so it could
not call it even if the gate were open. That is the same wall as
`burn-rmsnorm`'s fused kernel (AGENTS.md 3.3) and the same one the parked fusion
flip would remove — so the fix is not local, and the audit records the wall
rather than pretending otherwise.

---

## 2. `burn-sct/src` (brief item 1): the descent

Full table in the inventory (§3 there). What the descent found, in order of how
much it should change a reader's behaviour:

1. **745 lines of f64 SVD with no test and no external reference.**
   `host_svd.rs` — `svd_host`, `bidiag_host`, `dbdsqr`, `dlas2_smax`,
   `dlartg` — is a vendored transcription of LAPACK's bidiagonal SVD with
   **0 `#[test]`** in the file. Its `SAFETY` argument for the raw-pointer
   sharing is written down, which is better than nothing, and it is still
   ungated. Since the crate has no users, the correct move is deletion, not a
   test (F10).
2. **~300 lines of dead code inside it**: `qr` (the 80-line tensor-op QR that
   its own module doc calls "the reference/GPU path"), `cholesky_host_par`,
   `sct_qr_gram_kernel`, `sct_jacobi_round_kernel`, `sct_jacobi_vpass_kernel`,
   `sct_cast_f64_kernel`, `sct_transpose_kernel`. Verified by grep across the
   whole vendor tree, not by reading call sites in one crate.
3. **`from_dense_cuda` is a CUDA-gated function that is entirely host-side**
   (`qr_cuda.rs:515-557`): it does `into_data()` on the device tensor and then
   runs `host_svd::svd_host` in f32 on the host. The name and the gate both
   misdescribe where the work happens, and the crate's own comment says the GPU
   pipeline it replaced was launch-bound — which means the *gate* is the only
   thing that ever ran it.
4. **Two gates that cannot fail on a real error**: `retract_restores_ortho`
   (`lib.rs:529`) and `sign_correction` (`lib.rs:672`) both assert
   `|diag(UᵀU) − 1| < 0.1` on an 8- or 16-column factor. A retraction that
   returned 90% orthonormality passes both. The CUDA gate next door
   (`gpu_retract_matches_cpu`, 1e-4) is the one with teeth, and it needs a GPU
   target. A gate that cannot fail is worse than no gate: it is a green line
   in a log.
5. **`bytes_f32` (`lib.rs:254`) is `unsafe` on two `debug_assert`s** — 16-byte
   alignment is an allocator's current habit, not a guarantee. Fine in a crate
   nobody calls; not fine as a pattern.
6. **The QR itself is correct.** I re-derived the two places it is easy to get
   wrong: `r_k_from_r`'s sign flip (`qr.rs:407-416`) preserves `A = QR`, and
   `sct_qr_qsolve_kernel`'s forward substitution solves `Rᵀqᵀ = aᵀ` per row,
   which transposes to `q = aR⁻¹` — the right answer, and non-obvious enough
   that the doc comment being one line long is a small loss.

Honesty tier for the whole crate: **(b) transcription** of SCT 2604.00733's
Eq. 5, with the crate stating outright (`:321-322`) that there is no comparison
against the authors' code, and the comparison harness having been deleted
(`Cargo.toml:18-20`, `binary-tests` removed 2026-09-27 because "it could only
ever report PASS having asserted nothing"). **No external reference exists**
for the retraction's *accuracy*; the paper prints the formula, not numbers.

---

## 3. CLASS B: "polar is bad" — instrumented, not argued

The owner states polar is bad as prior belief. Per the brief I do not argue; I
measure. Three questions: what does it cost per site, what does it buy per site,
and can the failure it defends against occur at our shapes?

### 3.1 The instrument is anchored

`tools/polar_probe.rs` transcribes `polar_orthogonalize` (`lib.rs:208-266`) and
`‖UᵀU−I‖_F/k` (`param.rs:205`) in f32. Its only external anchor is the crate's
own doc-comment table at `lib.rs:188-191`:

| quantity | doc comment | probe |
|---|---|---|
| Frobenius prescale, per-entry error | `6.4e-2` | **6.3886e-2** |
| Frobenius prescale, σ after 3 iters | `0.6992` | **0.6992** |
| σ_max prescale, σ after 3 iters | `1.0000` | **1.000000** |
| σ_max prescale, per-entry error | `3.0e-7` | 2.96e-8 |

The two algorithm-dependent anchors reproduce (one to four digits, one
exactly). The fourth does not, and the direction is explained: the doc's input
was a factor that was itself only orthonormal to ~1e-7 (their QR'd normal), mine
is exactly orthonormal by construction, so mine lands an order lower. **The
transcription is good enough to price the retraction; it is not good enough to
quote 2.3e-8 as the device's floor**, because the device's reductions order a
64-element sum differently and 1e-8 lives in the cancelling regime. The
non-cancelling numbers below (1e-5, 1e-3, 8e-2) transfer.

### 3.2 Cost, per site, at `small` (16 factors: 8×`[768,64]`, 6×`[2048,64]`, 1×`[256,64]`, `retract_iters=3`)

| site | launches / factor | FLOPs / factor | what it is |
|---|---|---|---|
| canonicalising transpose | 0 (a view) | 0 | free |
| `g = m·mᵀ` (`:229`) | 1 | `2·64²·in` | the only big matmul |
| power iteration ×5 (`:231-238`) | 25 | ~0 | 5 ops each, all on `[64,64]`/`[64]` |
| Rayleigh quotient (`:240-248`) | 9 | ~0 | 2 reduces, 2 muls, div/sqrt/clamp |
| prescale ÷1.05 (`:251`) | 1 | 0 | |
| NS ×3 (`:255-260`) | 18 | 3×`2·64²·in` | `xx`, `xx²`, `poly·m` per iteration |
| `polar_retracked` (`:154`) | ~2 | 0 | `detach` + `set_require_grad` per factor |
| **total** | **≈54** | **4·2·64²·in** | **≈880 launches, 0.61 GFLOP per step** |

Measured, tonight, three separate arms (`~/logs/first_run_2000_0930_*.log`, all
launched by `tools/first_run.sh`: `--preset small --batch 8 --seq-len 512
--no-engram --steps 2000 --timers`, 5060 Ti, fp32, Fp8 factors, default seed 1):

| arm | warm step (median, p25–p75) | `retr` (median, min–max) | share | `retr` at step 0 |
|---|---|---|---|---|
| plain (`1855`) | 487 ms (476–529) | **25.7** (23.4–110.6) | **5.3%** | 432.8 ms |
| pure CE, JEPA off (`1921`) | 391 ms (384–474) | **25.8** (24.3–133.3) | **6.6%** | 416.6 ms |
| `--retract-every 4` (`1940`) | 504 ms (492–544) | **26.0** (24.0–35.2) | 5.2% | 465.6 ms |
| 2026-09-29, batch 8 (AGENTS §3.1) | 244 ms | **52.8** | **22%** | — |

So: **25.7–26.0 ms median, three arms, 57 warm readings, one number.** The
maxima (110.6 at step 1400 in `1855`, 133.3 in `1921`) are single steps that were
slow in *every* column at once (`total=2002ms fwd=882 bwd=696 opt=304
retr=110.6`) — not one of our 100/500-step cadences (steps 1400 and 1100 are
neither, and the real `memory_cleanup` step at `lib.rs:1682` fires at 500/1000/
1500, whose readings are unremarkable). Two of the three arms spike in the same
window and the third, which ran later, does not, so **GPU contention is the
likely cause and the cause is unattributed** — which is why the median, not the
mean, is the number here. 0.61 GFLOP at
this box's measured cuBLAS fp32 ceiling (13.2 TFLOP/s, AGENTS §2.2) is **46 µs
— 0.18% of the measured time.** The other 99.8% is 880 launches at 29 µs each.
The retraction is not a compute problem and never was; it is the same
launch-bound shape as `opt` (median 54–55 ms in the same three logs) and the
attention backward, and the fix for it is launch count, not arithmetic.

**The 2× discrepancy is unresolved and I am not going to paper it.** 52.8 ms
(2026-09-29) vs 25.8 ms (tonight) for the same nominal configuration, the same
16 factors, the same batch. The leading candidate is the autotune setting: the
09-29 measurement pinned `CUBECL_AUTOTUNE_LEVEL=3`; tonight's launcher sets
nothing (grep `tools/first_run.sh`). That is a hypothesis, not a measurement,
and both numbers are in the table rather than one of them being dropped.

### 3.3 Benefit, per site, and the failure it defends against

The trainer's forward for `small` is `forward_quant` with `QuantFormat::Fp8`
(every log line starts `quant format: Fp8 (8 bits)`), i.e. `quant_factor`
(`lib.rs:612-654`) → per-row absmax + e4m3. One property of that quantiser
decides everything below:

> **Per-row absmax factor quantisation is exactly equivariant to per-row
> scaling**: `F(cW)_i = c·F(W)_i` for `c > 0`, because the scale is computed
> from the row itself.

So a drift that only rescales rows is **invisible to the forward** — the
retraction's entire effect on such a drift is zero. What the retraction can
change is the factor's *geometry*: the column directions, and the per-row
dynamic range inside a row (which is what the absmax quantiser spends its 8
bits on).

Drift ladder at `[768,64]`, `delta = ‖ΔU‖_F`:

| `delta` | ‖UᵀU−I‖/k, no retraction | after 3 NS iters | col-angle moved by the retraction | vs e4m3 half-step noise (6.25e-2) |
|---|---|---|---|---|
| 8.0e-4 (1 step) | 5.09e-6 | 2.25e-8 | 3.48e-4 | **5.6e-3** |
| 8.0e-2 (100 steps) | 5.15e-4 | 2.31e-8 | 1.47e-3 | 2.4e-2 |
| 8.0e-1 (1000 steps) | 5.28e-3 | 2.22e-8 | 1.42e-2 | 0.23 |

Read that as: **at one step of drift the retraction moves the forward by 0.56%
of the quantisation noise the forward already carries — 180× below it.** The
retraction is not repairing a numerical error in the forward. It is imposing a
constraint on the parameterisation whose value is a *modelling* question, and
whose price is 5.3% of every step.

**So: can the failure it defends against occur at our shapes? Yes, decisively —
and not where anyone was looking.**

`routing.rs:95-98` sends `(Expert | Readout, Factor)` to **Muon+** and
`(Head, Factor)` to **AdamW** ("Rest"). Muon+'s `ColRow` normalisation
(`burn-muon-plus/src/lib.rs:343-380`, live order pinned at
`optim.rs:134`) makes every row of the update unit-L2, so
`‖ΔU‖_F = lr·√k = 8e-4`. AdamW's per-coordinate step is ~`lr` over an
`[768,64]` factor, so `‖ΔU‖_F = 1e-4·√49152 = 2.2e-2` — **28× larger**, on the
one factor whose reading *is* the latch (`max_ortho` is a max over factors):

| group | per-step `‖ΔU‖_F` | one step, per entry | latch at (systematic ×N) | latch at (random walk ×√N) |
|---|---|---|---|---|
| Muon+ ColRow, 15 factors | 8.0e-4 | 4.60e-6 | 218 steps | 47 345 steps |
| **AdamW head, 1 factor** | **2.2e-2** | **1.26e-4** | **8 steps** | **63 steps** |

The AdamW update is sign-like and momentum-smoothed, i.e. *coherent across
steps*, so the realistic figure is near the systematic end. **With the retraction
switched off, the `lm_head` factor would cross the 1e-3 one-way latch within the
first ~10–60 steps, and the run would train the entire way in fp32 factors** —
silently, one-way, persisted into the checkpoint, with the only trace being one
line in a log nobody reads (`max_ortho 1.95e-3 > 1e-3 - fallback fp32
factors`, which appears exactly once in the whole archive, in the degenerate
`pretrain_v21` run that trained on constant `b'x'`).

**That is the retraction's case, and it is arithmetic, not opinion.** It is also
**untested**: no run in the archive ever ran with the retraction off (§5 item 3).

**The cadence is therefore a RISK dial, not only a cost dial.** The head factor
accumulates `n × 1.26e-4` per entry between retractions (systematic end of the
table above; the random-walk end is `n × 1.26e-4/√n`):

| `--retract-every` | head factor's per-entry error between retractions | vs the 1e-3 latch | status |
|---|---|---|---|
| 1 | 1.3e-4 | 8× margin | every run in the archive |
| **4** | **5.0e-4** | **2× margin** | **the 2000-step run `1940` did this and did not trip it** — so reality is ≥2× below the systematic estimate |
| 8 | 1.0e-3 | **at the latch** | unmeasured; the arithmetic puts it there |
| 16 | 2.0e-3 | 2× over | unmeasured; would trip it |

So the *empirical* bound from one 2k-step run is "every-4th is safe", and the
*arithmetic* bound is "every-8th is not". That gap is the thing to keep an eye
on, and §5 item 1 (print `max_ortho` always) is what closes it for the price of
three characters.

### 3.4 What the retraction does *not* defend against, and one number in the rulebook is wrong

`--retract-iters 3` is not a dial. Sweeping the input's spectral spread at
`[768,64]` (probe §F):

| input spread | per-entry error **after** 3 NS iterations | vs the 1e-3 latch |
|---|---|---|
| 1:1 (a retracted factor) | 2.96e-8 | 0.00003× |
| 2:1 | 1.11e-5 | 0.011× |
| **3:1** | **1.93e-3** | **1.9× over** |
| 10:1 | 8.39e-2 | 84× over |

AGENTS.md §2.3 justifies the `1e-3` per-entry latch with "the NS-3 retract's own
convergence floor is ~4e-3 raw (~6e-5 per-entry at r=64), so the old
unnormalized check fired … above the floor and below real drift", and
`lib.rs:1344-1347` repeats it. **That quoted 6e-5 is not the floor of a retracted
factor** — my 1:1 row is 2.96e-8, 2000× lower. It sits between my 2:1 and 3:1
rows, i.e. it is what NS-3 leaves on a factor whose singular values are spread
about **2.2:1**, and a factor that has actually been retracted is at 1:1. The
real failure point is a 2:1–3:1 spread, within 2× of the threshold itself. Two
consequences:

- The floor is far lower than the rulebook says, so the margin to the latch is
  ~43 000× for the Muon factors, not 16×. The "the floor is uncomfortably close
  to the threshold" argument does not hold at our shapes. **The stale figure
  lives in three places** — AGENTS.md §2.3, `param.rs:200-204` and
  `burn-spectral/src/lib.rs:1344-1347` — so correcting it is three edits, and
  until they agree, a reader who checks one of them finds a number my probe
  contradicts.
- The real exposure is the **spread**, and nothing measures it. A factor that
  develops a 3:1 spread would be retracted to an error *past the latch*, every
  step, forever — and the latch would be reporting the retraction's own residue
  rather than the drift.

Since no run has ever hit the latch with the current metric, our factors are
demonstrably at spread ≲ 2 — the archive is the evidence, not a measurement.
F3 and F5 are the same defect: the number that would answer this is computed
every 500 steps and printed only on failure.

### 3.5 The quality evidence, and why it does not decide

Last night, three 2k-step arms, same window (**81 920 B** = 20 batches × 8 ×
512), same seed — none of the three checkpoint names carries a `--seed-` segment,
so all three ran the default seed 1 — and all three with `fused kda=…/0`, i.e.
**the attention backward was dead in all of them** (pre-`d8fa449`):

| arm | flags vs the plain recipe | best held-out BPB (step 1500) |
|---|---|---|
| `1940` | `--retract-every 4` | **6.329** |
| `1855` | (plain) | 6.387 |
| `1921` | `--jepa-weight 0 --dspark-weight 0` | 6.437 |

The matched pair is `1940` vs `1855` — they differ in exactly one flag — and it
says **4× less retraction was 0.058 BPB better**. Two reasons that is not a
result, and they are the protocol's own reasons:

1. **One seed per arm.** §1.2: "at our scale a single arm vs a single control
   decides nothing … 3 seeds per arm, and a win must beat the spread of the
   control's own seeds." The three arms span **0.108 BPB**, so the 0.058 effect
   is *half the spread of the arms themselves*.
2. `1921` is not a third reading of the same thing — it is a different treatment
   (pure CE), and it is the worst of the three. Quoting 6.329 "vs 6.387/6.437"
   as a three-way comparison would be comparing a flag change against a
   recipe change.

**This landed as a decision 20 minutes into this pass** (`d8a21b9`,
`.bulba/goal.md`: "рабочий режим = `--retract-every 4` (6.329, лучшая кривая;
внешний агент: «каждый шаг выкинуть, ретракцию не трогать»)"), so the two
things the owner needs from this audit are, precisely:

- **The 6.329 is not evidence.** It is one seed against one seed, and the three
  arms span 0.108 BPB. The decision is *not* supported by tonight's data, and
  the honest form of the citation is "6.329 vs 6.387, n=1 each, inside the
  arms' own spread".
- **The saving is real and unclaimed, and it is bigger than the 0.058 BPB
  argument could ever be.** `--retract-every 4` cuts the retraction from 25.8 ms
  to ~6.5 ms of step time, because **every timer step is a multiple of 100 and
  therefore of 4** — which is exactly why the `retr=` column in §3.2 is
  identical for `1940` and for the every-1 arms (25.7 / 25.8 / 26.0 ms): the
  timer only ever prints on a step where the retraction *fires*. Over a 100k-step
  run that is 19.3 ms × 100k ≈ **32 minutes**, ~3.8% of the run.
- **And the risk that comes with it, in the same breath** (§3.3's cadence
  table): every-4th puts the head factor at 5.0e-4 per entry against a 1e-3
  one-way latch — a 2× margin that one 2k-step run says is real and that
  **nothing will be watching** for the next 100k steps, because the latch's
  reading is printed only when it has already failed. Pair the flag with §5
  item 1 and the risk is visible for the whole run.

So: **the retraction's quality effect is not established, and the one data point
that exists points the other way at half the noise floor.** Both facts belong in
the record. What the arithmetic in §3.3 does establish is the *other* half: the
retraction prevents a silent, one-way, permanent loss of factor quantisation for
the head factor. Those are different claims and only one of them has evidence.

---

## 4. Findings, with what would falsify each

Ordered by what I would do next. Nothing here is fixed; `param.rs` and
`model.rs` are core files other lanes may be in (§1.6), so each is a
`file:line` report, not an edit.

| # | finding | evidence | what would settle it |
|---|---|---|---|
| **F1** | no seam: `burn-sct` has zero non-dev call sites and no trainer dependency | grep over `vendor/dormouse-fused` + `crates/`; `burn-spectral/Cargo.toml:33` | nothing to settle — it is a fact. What it *decides*: delete `burn-sct` (and with it 745 untested SVD lines and ~300 dead ones) or keep it as a kernel donor for the retraction. The `library-crate-fate.md` recommendation already says DELETE; the counter-argument is K8's one-launch kernel |
| **F2** | the head factor's un-retracted drift crosses the latch in ~8–60 steps | probe §D, from `routing.rs:98` + `optim.rs:134` + `burn-muon-plus:343-380` | **one 500-step run with `--retract-every 1000000`** and read whether `max_ortho` prints, and at what value. This is the single most informative cheap experiment in the queue |
| **F3** | `--retract-iters 3` fails above a ~2:1 spectral spread | probe §F | print σ_max/σ_min of one retracted factor per 500 steps. If it is ≪2, the 3 iterations are safe and the risk is theoretical; if it approaches 2, a 4th NS iteration is +6 of 54 launches/factor, i.e. **+2.9 ms = +0.6% of a step** |
| **F4** | 5.3% of a warm step, 0.18% arithmetic, ~880 launches | 57 timer readings across 3 logs + probe §E | `--retract-batched`: 16 calls → 4, ~5× fewer launches. One flag, no code |
| **F5** | `max_ortho` is computed every 500 steps and printed only on failure | `train/src/lib.rs:1350-1357` | print it always under `--timers`. Three characters, and it converts F3 from unanswerable to answered |
| **F6** | batched retraction off by default, never used | `lib.rs:175`; `retr_arm=batched:0` in every log | run it; `retract_batched_identity_with_per_factor_path` says the numbers are identical, and if they are, the default is wrong |
| **F7** | `qr_householder` host-reads and is the TSCT residue's source | `lib.rs:689`, called `:462`/`:469`; AGENTS §3.7 | the same QR on the host (or a fixed iteration count) and re-run `tools/determinism.py`. Init-only, so it costs nothing per step and fixes a §1.3 violation for free |
| **F8** | "40-50% of a step" in a doc comment, 5.3% measured | `lib.rs:19-20` vs the logs | none — it is stale, and it will be quoted |
| **F9** | "degrades into NaN" has no evidence and arithmetic says it cannot | `model.rs:416-417`; per-row absmax equivariance | a NaN run, or downgrade the sentence to what is true: the forward degenerates to all-zero weights, CE → 5.5452 |
| **F10** | 745 untested f64 SVD lines + ~300 dead lines, zero users | `host_svd.rs` (0 tests); dead-call grep | delete. A test for code with no caller is a test for nothing |
| **F11** | two `burn-sct` gates with 0.1 tolerances | `lib.rs:529`, `:672` | moot if F1 deletes the crate |
| **F12** | `param.rs:1` names the wrong crate | `param.rs:1` vs `:6` | one word. §1.7 |
| **F13** | the `0.7` dead zone is a (c)-tier constant with no gate | `lib.rs:54` | find the burn-es source, or write it down as ours |

---

## 5. Follow-ups, cheapest first, each with the number it buys

Ordered by cost. The first two need no build and no GPU; the third is 4 minutes
of GPU once the current run finishes (§1.5).

1. **Print `max_ortho` on every check, not only on failure.**
   `train/src/lib.rs:1352-1356` — move the `println!` out of the `if`. Buys:
   the spread/drift signal that F3 and F5 need, in every future run, for three
   characters. **This is the one I would do before anything else in this
   document.**
2. **`--retract-batched` for one 200-step run with `--timers`.** Buys: the
   F4/F6 answer (expect `retr` ≈ 5–7 ms against 25.8) and a check that
   `retract_batched_identity_with_per_factor_path`'s promise holds on the
   trainer's device, not just on ndarray. (`--set` reaches `DormouseConfig`
   keys only — `config/override.rs:70` errors on an unknown key — so the three
   retraction knobs are the typed flags `--retract-every`, `--retract-iters`,
   `--retract-batched`, `cli/bin/train.rs:136-146`.)
3. **One 500-step run with the retraction off** (it retracts once at step 0 and
   never again, since the gate is `step % retract_every == 0`):
   `train --data <corpus dir> --eval <held-out> --preset small --batch 8 --seq-len 512 --no-engram --steps 500 --retract-every 1000000 --timers --log ~/logs/noretract.log`
   (plus fix 1, or it prints nothing until the latch fires, which is the
   measurement). Buys: F2. Expected outcome from the arithmetic: the latch
   fires between steps 8 and 500, and the run's quality is *worse or equal* —
   in which case the retraction's justification is the quantisation arm, not the
   quality arm, and that sentence should be written down.
4. **Decide `burn-sct`'s fate** (F1). If DELETE, the four follow-ups below are
   free. If KEEP-as-kernel-donor, the honest form of the fix is the fused
   retraction on the trainer's device — blocked by the fusion flip
   (ADR-0018/PLAN), so it is a *plan* item, not a task.
5. **The retraction ladder as an A/B**: `--retract-iters {3,4,5}` × 3 seeds at
   2k steps, one batch size (§2.6). Buys: whether 3 is a cliff in practice
   (F3). The per-arm cost is currently **unknown** (§3.3's open item), not the
   2.7 GPU-h `docs/protocols/AB-PROTOCOL.md` still prices it at.
6. **Not this lane, reported per §1.6**: `param.rs:1` (F12),
   `model.rs:416-417` (F9), `burn-spectral/src/lib.rs:19-20` (F8) and `:54`
   (F13) are one-line doc corrections in files other lanes may hold. Plus the
   **three copies of the stale retraction floor** (F3): AGENTS.md §2.3,
   `param.rs:200-204`, `burn-spectral/src/lib.rs:1344-1347` — and per §1.7
   those three must be fixed together or the rulebook keeps contradicting the
   measurement.

**What I would NOT do:** delete the retraction, and I would not go past
`--retract-every 4`. The prior "polar is bad" is not supported by anything I
measured — the math reproduces its own published anchor exactly, it is ~43 000×
inside the latch at our shapes, and it is the only thing between the head factor
and a silent one-way permanent loss of factor quantisation. The 5.3% is a
launch-count problem with a one-flag experiment attached, not a reason to remove
a constraint that has never been A/B'd.

**For the 100k that is launching with `d8a21b9`'s flag: every-4th is the right
call, on the cost half and against the quality half, and one three-character
change makes it safe to see.** every-4th is measured (one 2k-step run, no latch)
and it buys 32 minutes; the 0.058 BPB that "best curve" rests on is noise and
should not be cited again. The 2× margin the head factor has at every-4th is
invisible for the whole 100k unless `max_ortho` is printed on every check
instead of only on failure — which is §5 item 1, and it is the single change
from this audit I would make before the next launch.
