# attnres — two independent reviews, one lane

Merged 2026-10-01 from the `attnres-review-a.md` and `attnres-review-b.md` halves, both at
`5cfdbda`, which is where a reader finds each one whole. The halves were written independently
and neither read the other (A: Review of `docs/papers/attnres.md` — independent verification; B: Reviewer B — would following `docs/papers/attnres.md` actually fix `burn-attnres`?),
which is why both verdicts are kept: where they agree the finding is settled,
where they disagree the disagreement is the finding. Nothing was reworded.

## Reviewer A — Review of `docs/papers/attnres.md` — independent verification

Reviewer: adversarial reviewer, spawned session. Date: 2026-09-29.
Methods actually used: `pdftotext -layout` on a fresh download of `arXiv:2603.15031v1`,
GitHub REST API (`trees`, `branches`, `tags`, `releases`, `commits`, `issues`), `curl` of
third-party sources, and line-by-line reading of
`vendor/burn-fused/crates/burn-attnres/src/{lib.rs,fused_attnres.rs}`.
**No code was built or run.** Every "I traced" below is a source trace, and each one names
the command that would settle it empirically.

Bottom line: the report's **code reading is unusually accurate** — every `file:line` I
checked exists and says what is claimed — and its paper reading is correct on every point it
chose to check. It is wrong or incomplete in three places that matter: **one of its two
framing claims is refuted by a source it never fetched**, **three bugs more severe than
several it did find are absent from the table**, and there are ~10 factual/arithmetic slips
including one arithmetic error in its own arithmetic and one self-contradiction.

---

### 1. Verdict per claim

| # | claim | verdict |
|---|---|---|
| 1 | arXiv 2603.15031, Kimi Team, 2026-03-16 | **CONFIRMED** |
| 2 | no original code; repo master = 7 blobs | **CONFIRMED (narrowly)** / **REFUTED (as an oracle claim)** — see §3.1 |
| 3 | no `1/√d` in the paper; we apply it at `lib.rs:109,180,410` + kernels | **CONFIRMED** |
| 4 | our "RMSNorm" is `√d`-larger L2; logit ≈ paper's `/d` | **CONFIRMED**, with one false sub-claim — see §4.6 |
| 5 | `BlockAttnRes::step` is not Eq. 6 (4 sub-findings) | **CONFIRMED** on D5–D8, but the report **missed a fifth, worse one** — see §3.2 |
| 6 | `two_phase_attend` scales only Phase 1 | **CONFIRMED** |
| 7 | the `b_0` question is RESOLVED, no disagreement | **CONFIRMED**, and stronger support exists than the report found — see §4.9 |
| 8 | 19 deltas: 9 BUG / 1 CONTRACT / 3 benign / 1 out-of-spec / 5 MATCH | **arithmetically CONFIRMED, substantively REFUTED** — 1 not independent, 1 misclassified, 3 more BUGs missing. See §4.5, §3.2, §3.3 |
| 9 | crate is REFERENCE-ONLY, not in the dormouse build | **CONFIRMED** |
| 10 | every test re-transcribes our own formula | **CONFIRMED in direction, WRONG in inventory** — one named test is a tautology the report never read, and one real oracle it never mentions. See §3.4 |

#### 1.1 Claim 1 — CONFIRMED

`pdftotext -layout` on `https://arxiv.org/pdf/2603.15031v1`, 1 065 095 bytes, PDF 1.7, 9 pages,
1439 lines (all four numbers match the report exactly). Header, line 1–15 of the extraction:

```
ATTENTION R ESIDUALS
T ECHNICAL R EPORT OF ATTENTION R ESIDUALS
Kimi Team
https://github.com/MoonshotAI/Attention-Residuals
arXiv:2603.15031v1 [cs.CL] 16 Mar 2026
```

`curl -sL https://arxiv.org/abs/2603.15031` → `Submitted on 16 Mar 2026]`. LaTeXML stub
confirmed: `https://arxiv.org/html/2603.15031v1` returns HTTP 200, 22 476 bytes, no article
body. The report's choice of `pdftotext` was correct and its warning about the HTML build is
correct.

#### 1.2 Claim 2 — CONFIRMED as stated, but the operational conclusion drawn from it is REFUTED

`GET /repos/MoonshotAI/Attention-Residuals/git/trees/master?recursive=1` returns exactly what
the report lists: `Attention_Residuals.pdf` (952 700), `README.md` (5 997), and four PNGs
(234 859 / 324 332 / 122 277 / 186 168). `branches` → `master` only. `tags` → `[]`.
`releases` → `[]`. `commits` → **4** (`729383a5` initial, `a422c809` logo, `3822cb91` citation,
`85e22310` README), HEAD `85e22310fe5ee860b4a023de312d791de8a5a5e6`. `stargazers_count` 3517,
`forks_count` 205. `issues?state=all` → 14 items, **0** are PRs, 13 open + 1 closed.

The Fig. 2 pseudocode is in the repo README (`raw.githubusercontent.com/.../README.md:53-78`)
**byte-identical to the paper's** apart from a 2-space indent. Confirmed.

Three corrections to the report's own numbers:
* **"7 blobs (1 PDF, 1 README, 4 PNGs)" is self-contradictory** — 1+1+4 = **6**. The seventh
  tree entry is `assets` (mode `040000`, type `tree`). The report's §1 code block lists six
  blobs; its VERIFIED section and the brief both say "7 blobs".
* **"12 open issues"** — there are 13 open (+1 closed, #1). Not material, but it is in a
  block labelled VERIFIED.
* **"2268 lines"** in §3.3 is a **stale figure copied from `docs/architecture/library-crate-fate.md`**
  (recorded "at `422414c`"). The report's own §1 measures `lib.rs` 517 + `fused_attnres.rs`
  1821 = **2338**. `docs/archive/architecture/PLAN-minimal-core.md:90` says 2060. Three documents, three numbers,
  none re-measured. AGENTS.md §1.4: "a measurement is a measurement only with the config, the
  date and the commit it was taken at."
* `crates/burn-attnres/README.md:3–:8` (§3.3) is a **wrong path** — the file is
  `vendor/burn-fused/crates/burn-attnres/README.md`. Its content is as quoted (3–8 does say
  "Not in the dormouse build").

#### 1.3 Claim 3 — CONFIRMED, every line reference real

Paper, all four sites, verbatim from the extraction:

* line 202 (§3.1): `we adopt ϕ(q, k) = exp q ⊤ RMSNorm(k) [66] with normalization`
* line 970 (Table 5, footnote 2): `ϕ(q, k) = exp q ⊤ RMSNorm(k) ; ki = vi ; v0 = h1 , vi≥1 = fi (hi ). softmax jointly normalized over all sources.`
* line 280 (Fig. 2 line 11): `logits = torch.einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)`
* line 431 (Alg. 1 line 3): `{ol , ml , ℓl }l∈Bn ← ATTN W ITH S TATS (Q, K, V)  // Return LSE`

`grep -in "sqrt\|square root\|temperature\|scal"` over all 1439 lines: 27 hits, **none** a
scaling factor. They are: "scaling law" (×5), "scaled residual paths [54]" (×2, a cited
*different* method), "attention temperature rescaling" (line 624, about MLA/NoPE context
extension), `d_model`/dimension prose, and the §5.1 fitted-curve prose. The report's
characterisation of each hit is accurate.

Code, all references verified present and correct: `lib.rs:109` `let scale = (d as f64).powf(-0.5);`
· `:155` `.mul_scalar(scale)` · `:180` `.mul_scalar((d as f64).powf(-0.5))` · `:410` scale def ·
`:428` `.mul_scalar(scale)`. Kernels: `fused_attnres.rs` 111 (`scale: f32` param), 164, 313,
383, 390, 438, 440, 521, 554, 603, 689, 711, 756, 765, 1142, 1165. Backward-tensor: 1348, 1356,
1372, 1376 — all present, all carry the factor. (`:432` is correctly *not* listed: it is
`let inv = …`, no scale.)

#### 1.4 Claim 4 — CONFIRMED, with one false sub-claim

`[66]` is confirmed as Zhang & Sennrich, *Root mean square layer normalization*, NeurIPS 32
(2019) — extraction line 1320. The paper never writes the formula, it only cites. RMSNorm per
[66] is `x / sqrt(mean(x²) + ε)`. Our code: `lib.rs:150` `powf_scalar(2.0).sum_dim(3).add_scalar(1e-5)`
then `:152` `/ h_norm_sq.sqrt()`. Divergence confirmed.

The composed factor is exactly as the report says. Paper: `w·h/√(Σh²/d+ε)`; ours:
`w·h/(√d·√(Σh²+ε))`. For `‖h‖²` large, ours/paper = `(1/√d)/√d` = **`1/d`**. CONFIRMED, and
this is a nice piece of arithmetic.

**False sub-claim:** D2's prose says our function has "a different ε placement". It does not.
Zhang & Sennrich put ε **inside** the square root; `lib.rs:150-152` puts ε **inside** the
square root. The *only* difference is the `1/d`. The ε clause should come out of D2.

#### 1.5 Claim 5 — CONFIRMED as far as it goes; it is incomplete

Verified in `lib.rs:270-304`, quoted in full at §3.2 below. D5 (`:272-276`), D6 (`:282-286`),
D7 (`:292-294`), D8 (`:291` then `:299` into the same mutable state) all read exactly as
claimed. The `merge_source` in-place mutation at `:242-245` is real, so the double-merge
arithmetic the report gives ("weight ~2, `sum_exp` inflated by 1") is right: the second merge
sees `m_new ≥ s_c` already, so `rescale = 1` and `acc += c·e^{s_c−m_new}` a second time.

Two classification problems, both against the report:

* **D3 is not an independent observable** and should not be its own BUG row. D1 contributes
  `1/√d`, D2 contributes `1/√d`; together they are a single global factor `1/d` on the logit,
  i.e. a pure softmax temperature. A global temperature is absorbable into the free `w_l` by
  rescaling it. D1+D2 therefore change **nothing about the function class** — only the
  effective learning rate on `w_l` (by a factor of `d`) and the trajectory. The report attaches
  exactly this nuance to D1 ("reachable by the paper's function class with `w_l` rescaled by
  `√d`") and then fails to apply it to D3, which it calls "BUG (observable)". It is not
  separately observable. **And the report's own remedy number is wrong by `√d`:** D3 says
  "`w_l` must grow ~`√d` more than ours"; the factor is **`d`**, because the two `1/√d`s
  compose.
* **D5 is not a BUG, it is a CONTRACT violation — the same one as D10.** The crate's own doc
  at `lib.rs:263` says *"`h`: the new layer output"*. Under that contract the first `h` is
  `f_1`, and `st.partial = f_1` is **correct** — there is no `b_0` being folded, because the
  caller never supplied one. The report's whole D5 mechanism ("the first block's incorporated
  representation is `b_0 + Σf`") is *conditional on a caller that violates the crate's own
  documented contract*. What is unconditional is the mirror problem, which the report files
  separately as D10: under the layer-outputs reading, `b_0 = h_1` is **never a source,
  anywhere, silently**. The crate's two doc comments also disagree with each other about the
  indexing: `lib.rs:76` says `history: [h0, h1, …, hL-1]` (h₀ = embedding, the paper's
  convention) while `lib.rs:263` says "the new layer output". That disagreement is itself a
  defect and is not in the table.

#### 1.6 Claim 6 — CONFIRMED

`lib.rs:428` `.mul_scalar(scale)` on the Phase-1 scores; `lib.rs:450`
`let s2 = (q_i.clone() * p_norm.clone().reshape([1, d])).sum_dim(1);` with no scale, fed
straight into `m2` (`:451`) and the merge (`:466-471`). The two legs of Alg. 1 line 12 are
compared at temperatures differing by `√d`. Correct, and the report's observation that
`two_phase_merge_matches_full_attention` (`:492-503`) only exercises `i = 0` — which takes the
`i == 0 || n == 0` branch at `:459` and bypasses the merge entirely — is also correct.

#### 1.7 Claim 7 — CONFIRMED, and under-supported by the report

Traced in full and I agree. §3.2 of the paper, lines 317-321 of the extraction:

> The block-wise variant replaces these individual outputs with block representations,
> defining b0 = h1 so that the token embedding is always included as a source. […] The input of
> the very first layer of the network is the token embeddings, i.e. b0 = h1 . In each block,
> the first layer receives the previous block representations and the token embeddings, and
> the subsequent layers **additionally** attend to the partial sum bi−1 n .

Eq. 5 (line 253-256): `bn = Σ f j (hj )` — layer outputs only. Eq. 6 (line 310-314) as
transcribed. Fig. 2's trace in §2.4 of the report is right.

**But the report cites the wrong figure for its strongest evidence.** It leans on Fig. 8
("the embedding retains non-trivial weight") and Fig. 6, both *empirical heatmaps*. The
paper contains a **fourth, purely structural** statement the report never cites: **Fig. 9**
(§6.2, extraction lines 1033-1043) prints the exact source enumeration for Full and Block
AttnRes at L=4, S=2 —

```
Block AttnRes panel:
  row w1:  ϕ(w1,k0)
  row w2:  ϕ(w2,k0)              ϕ(w2,k1)
  row w3:  ϕ(w3,k0)              ϕ(w3, k1+k2)
  row w4:  ϕ(w4,k0)              ϕ(w4, k1+k2)   ϕ(w4,k3)
```

That is Eq. 6 enumerated source by source, with the intra-block partial shown as a **distinct
source** (`k1+k2`) never folded into a completed block, and `k0` = the embedding as a
permanent separate column. It settles D5/D6/D7 without inference, from a figure the report
skipped.

#### 1.8 Claim 8 — arithmetic right, substance wrong

9+1+3+1+5 = 19 = D1…D19. The report's own summary is internally consistent. But: D3 is not
independent of D1+D2; D5 is a CONTRACT not a BUG; and **three additional BUG-class defects
are missing** (§3.2, §3.3 below). Corrected count: **11 BUG-class** (D1/D2 fused, D4–D9, plus
three new), **2 CONTRACT** (D5, D10), 3 benign/ambiguity, 1 out-of-spec, 5 MATCH, **3 new**.
The headline "the crate is not a port" survives; the arithmetic does not.

#### 1.9 Claim 9 — CONFIRMED

`crates/dormouse-core/Cargo.toml` and every other `crates/dormouse-*/Cargo.toml`: **no
reference to `burn-attnres`**. The workspace root `Cargo.toml:12` has
`exclude = ["vendor/burn-fused", "vendor/cubecl-fix", "vendor/cubek-fix"]` and `members` =
`cublas-poc, dormouse-core, dormouse-data, dormouse-train, dormouse-cli, backend-parity`. The
only inbound edges to `burn-attnres` are inside its own workspace: `vendor/burn-fused/Cargo.toml:4`,
`vendor/burn-fused/burn-fused/Cargo.toml:19,31,52,79`, `vendor/burn-fused/benches/Cargo.toml:12`.
`docs/architecture/library-crate-fate.md:62` records fate `b / REFERENCE`; `docs/archive/architecture/PLAN-minimal-core.md:90`
(§M2) names it as a residual-stream A/B arm that has not been run. **No dormouse number is
retracted. This claim is the most important one in the report and it is correct.**

#### 1.10 Claim 10 — CONFIRMED in direction, WRONG in inventory

The claim is true as a statement about the *formula* under test: **no test in the crate can
fail on a paper divergence**, because every expected value is a re-transcription of our own
`1/√d` / `√d`-larger-L2 formula. I found **four** such re-transcriptions, not the one the
report names:

* `lib.rs:143-160` — the shipped tensor fallback
* `fused_attnres.rs:847-861` — `ref_depth_attend` (the one the report cites; `:850` `powf(-0.5)`,
  `:855` `mul_scalar(scale)` — both line numbers exact)
* `fused_attnres.rs:1489-1505` — `depth_attend_tensor_ad`, the autodiff fallback
* `fused_attnres.rs:1642-1662` — `raw_depth_attend`, the finite-difference oracle
* `benches/attnres.rs:13-31` — a fifth copy, in the benchmark

But the report's *enumeration* of the test suite is wrong in **both** directions. It omits
five `#[test]`s, one of which is a tautology and one of which is the crate's only genuine
third-party oracle. See §3.4 — this is the second most consequential finding.

---

### 2. Things the report got right that are worth keeping verbatim

Credit where it is due; these are all verified:

* Fig. 2, Eq. 1-6, Table 5 fn 2/fn 3, §3.2, §5 (zero-init), Alg. 1 — every transcription in
  §2 is **character-exact** against the extraction. I diffed the 22 pseudocode lines and
  Algorithm 1 line by line.
* The D11/D12 paper-ambiguity readings. Alg. 1 line 10 prints `b_n^i` and Eq. 6 prints
  `b_n^{i-1}` — the paper really does contradict itself, and the report's argument from line
  14 ordering + `b_n^0 := 0` + non-circularity is the right one. Likewise "Return LSE" at
  Alg. 1 line 3 versus the raw-sum requirement of line 12 is a genuine paper defect.
* **§3.2's data-race table is exact.** I checked every cell: `sync_cube()` at
  `fused_attnres.rs:658` with the `tid == 0` write at 659-662 and the matching comment at
  654-657; `:271` / 272-275 / 268-270; the scores kernel at `:141`, `:147`, `:150`; the
  backward kernel at `:359`, `:377`, `:419`. Both barriers are present and both comments
  describe them. The reviewer-note-is-stale call is right.
* D14-D18 (zero-init, keys-normalised/values-raw, joint softmax, the online-merge algebra,
  the `n == 1` short-circuit). All five verified in source. The backward kernel's analytic
  derivative (`fused_attnres.rs:432-441`) is also **correct for our forward** — I checked
  `d/dh_l [q·h_l/√(‖h_l‖²+ε)]` by hand against `:438`. The report did not check it; it is a
  MATCH it missed.
* D19 (`lib.rs:108` indexes `history[0]` at `n == 0` → index panic, not a named error). Real,
  CPU-reachable, one line.
* The `nktkt/attention-residuals` characterisation is **exact**. I read
  `attention_residuals/attn_res.py`: `nn.RMSNorm` (true 1/d form), no `1/√d`,
  `nn.init.normal_(self.proj.weight, std=0.02)` at line 32 — a clean, real contradiction of
  §5's zero-init mandate. 5 stars, last pushed 2026-03-24. The PyTorch RFC exists and is open
  (pytorch/pytorch#177537, `atilsamancioglu`, 2026-03-16). "A PyTorch core dev" is an
  unverified personal-status claim; the issue's existence and content are not.

---

### 3. What the report MISSED — ranked by severity

#### 3.1 SEVERITY 1 — The "no executable oracle" conclusion is refuted by a burn-native Rust port the report never fetched

The report's §1 verdict paragraph: *"**Therefore the paper is being used as the
specification, and there is no executable oracle.** The closest thing to a reference
implementation is the 22-line PyTorch pseudocode in Fig. 2"*, followed by a table of three
**PyTorch** third-party ports.

There is a **fourth port, in Rust, on burn**, and it is the only one that matters for auditing
a burn crate:

`https://github.com/AbdelStark/attnres` — 54 stars, "Rust implementation of Attention Residuals
from MoonshotAI/Kimi", `burn = "0.20.1"`, last pushed 2026-03-23, with `src/`, `tests/`
(`differential_tests.rs`, `property_tests.rs`, `unit_tests.rs`, `integration_tests.rs`) and
`benches/`.

What it does, read from source:

* `src/rms_norm.rs:45-46` — `let variance = x.powf_scalar(2.0).mean_dim(2); let rms = variance.add_scalar(eps).sqrt();`
  → the **true 1/d RMSNorm**, i.e. it does *not* have our D2.
* `src/attn_res_op.rs:88-92` — `let logits = (k * w).sum_dim(3).squeeze_dim::<3>(3);` →
  **no `1/√d`**, i.e. it does *not* have our D1.
* `src/attn_res_op.rs:61-75` — `forward_optional_partial(blocks, partial_block: Option<…>)`
  builds `sources = blocks` alone for `i = 1` and `blocks + [partial]` for `i ≥ 2` → it
  implements **Eq. 6's source set literally**, which is precisely what our D5/D6/D7 break.
* `src/block_state.rs:14, 24-28` — `/// blocks[0] = token embedding (b_0 = h_1)`,
  `pub partial_block: Option<Tensor<B,3>>` initialised to `None`, seeded by
  `new(token_embeddings)` → Fig. 2's `partial_block = None` boundary reset and the separate
  `b_0` source, both of which our streaming path has no equivalent of.
* `src/two_phase.rs:142-143` — `compute_intra_logit` = `op.norm.forward(partial)` then a bare
  dot product → **no scale on either leg of the merge**, i.e. it does not have our D4.

So: an independent, burn-native, directly diffable implementation of the *same paper* exists,
and it agrees with the paper on **all four** of the report's headline BUG classes and on none
of the four transcription defects. That is a fourth independent corroboration of the `1/√d`
verdict — from a Rust port, which is stronger evidence for *this* crate than three PyTorch
ones — and it is a ready-made oracle for exactly the test suite the report had to design from
scratch in §5.

This is worse than an omission because it changes the report's conclusion, not just its
completeness. §5 proposes hand-writing a fresh f64 `ref_attnres` in f64 with no burn
(Test A-H, ~8 tests). The cheap move the report missed entirely is: **diff our three
formulations against `AbdelStark/attnres`'s four files line by line.** That is a day's reading
and it settles D1, D2, D4, D5, D6, D7 with no transcendental argument and no tolerance
debate.

**And the repo already knew.** `docs/archive/research/2026-09-27-attenres.md:32` (two days earlier, same
tree): *"One Rust port exists upstream (`AbdelStark/attnres`, 54 stars) — ours is the more
complete one."* The new report neither cites that document nor engages with the port it names.
Two research docs on the same crate now coexist with **three different line counts** (2060 /
2268 / 2338) and **contradictory conclusions** (the old one says "the code works"; this one
finds 9+ bugs). AGENTS.md §1.7: *"a glossary that disagrees with the code is worse than none"*
— the same applies to two reports on one crate. **The new report should carry a line that
supersedes `2026-09-27-attenres.md`, and it does not.**

#### 3.2 SEVERITY 2 — `BlockAttnRes::step` computes layer *i*'s output from an accumulator that already contains layer *i*'s own output. The report found this in the paper, praised the wrong function for avoiding it, and missed it in the function that matters.

`lib.rs:270-304`, verbatim:

```rust
270: pub fn step(&self, h: Tensor<3>, st: &mut BlockAttnState) -> Tensor<3> {
271:     let [b, t, d] = h.dims();
272:     st.partial = if st.partial_count == 0 {
273:         h.clone()
274:     } else {
275:         st.partial.clone() + h.clone()          // ← h is added BEFORE the output exists
276:     };
277:     st.partial_count += 1;                      // ← now equals i
...
287:     if st.partial_count >= 2 {
288:         let s_p = source_score(&self.query.val(), &st.partial);
289:         if st.started {
290:             let p = st.partial.clone();         // ← p = b_n^i, includes f_i
291:             out = st.merge_source(&p, &s_p);    // ← out_i is a softmax over a set containing f_i
292:         } else {
293:             out = st.partial.clone();
294:         }
295:     }
```

Per `lib.rs:263` (`h` is "the new layer output"), at iteration `i` `st.partial = Σ_{l≤i} f_l = b_n^i`.
`out` is then a softmax-weighted mixture **that includes `f_i` itself**. Since the caller
computes `f_i = f_i(out_i)`, `out_i` is a function of `f_i(out_i)`: a fixed point, not the
paper's recurrence.

Eq. 6 and Alg. 1 line 14 are unambiguous: the intra-block source is `b_n^{i-1} = Σ_{l<i} f_l`,
and the accumulator is updated **after** the layer. The report identified this exact circularity
in §2.3 (A1: *"Using `b_n^i` would make `h_l` depend on `f_l(h_l)` — circular"*), then applied
the check **only to `two_phase_attend`** — whose *timing* is correct — and gave `step` a pass.

**Severity ranking within the crate's own bugs: this is worse than D7, not better.** D7
(`:293`) is a single unguarded case at `i = 2, n = 1`. This fires at **every `i ≥ 2` in every
block**, on both the first and later blocks. It is also invisible to `streaming_fused_matches_tensor_path`
(`:938-977`), which compares CPU-`step` against CUDA-`step` — same function, same circularity.

At the block boundary the same statement doubles: when `partial_count == block_size`, the
partial merged into `out` at `:291` **is** the completed block that `:299` then `incorporate`s.
So the last sublayer of every block attends to its own output *and* adds it to the state.

*Source trace, not executed.* Settles it: add the report's own Test C harness, feed
`f_1…f_5` as **layer outputs** (not the embedding), and assert `out_3` equals
`ref_attnres([b_0, b_1, f_3])` — i.e. that the `f_3` term is **absent** from the source set.
Today it is present. One line, CPU-only.

#### 3.3 SEVERITY 3 — `two_phase_attend` builds the block representation as `Σ h_l`, not `Σ f_l(h_l)`. Eq. 5 is unimplementable in that function and the report transcribed Eq. 5 correctly, then did not apply it.

`lib.rs:439-484`, verbatim:

```rust
439:     let mut partial = Tensor::<1>::zeros([d], &device); // b0_n := 0
...
453:         let o2 = partial.clone().reshape([1, d]); // single-key weighted sum
...
475:         // update partial sum: b_i := b_{i-1} + h_l
476:         partial = partial.add(h1);          // ← h_l, the attention OUTPUT
...
484:     (out_t, partial)                        // ← returned as "the block representation"
```

Three consequences, none in the report:

1. **The intra-block key/value is the wrong quantity.** `:443-453` scores and returns
   `Σ_{l<i} h_l`. Eq. 6 / Alg. 1 line 10 require `b_n^{i-1} = Σ_{l<i} f_l(h_l)`. The comment at
   `:475` writes the update as `b_i := b_{i-1} + h_l` — the paper's line 14 says
   `b_n^i ← b_n^{i-1} + f_l(h_l)`. The `f_l` is missing.
2. **The returned `partial` is not `b_n`** and must never be pushed into the block cache, which
   is exactly what the doc at `:397-399` tells a caller to do (*"the updated block
   representation (the sum of the block's outputs) to be pushed into the block cache"*).
   Magnitudes are wrong by construction, not by a constant: `h_l` is a convex combination of
   source magnitudes, so `Σ h_l` grows like `S·‖h‖` where `b_n` grows like `Σ‖f_l‖`.
3. **It is not fixable inside the function.** The signature is
   `two_phase_attend(queries: Tensor<2>, blocks: Tensor<2>)` — there is no `f_l` to apply. The
   report's D11 marked this exact code region "we chose right". The *timing* is right (pre-update
   partial, per Eq. 6); the *content* is wrong. D11 should be split: **timing → MATCH, content
   → BUG.**

The report's own words at §2.2 — *"Eq. 5 `b_n = Σ_{j∈B_n} f_j(h_j)` ← sum of LAYER OUTPUTS
only"* — are correct, and were never used on this function.

#### 3.4 SEVERITY 4 — `depth_attend_fused_backward_matches_tensor` is a tautology, and the fallback it appears to cover has zero numerical coverage

`fused_attnres.rs:1590-1625` (module `#[cfg(all(test, feature = "autodiff", feature = "cuda"))]`):

* "fused op graph" side, line 1602: `crate::fused_attnres::depth_attend_autodiff::<CudaBare, 64>(&hf, qf.clone())`
* "tensor path graph" side, line 1613: `crate::depth_attend(&ht, qt.clone())`

But `lib.rs:111-137`, under `#[cfg(all(feature = "cuda", feature = "autodiff"))]`, routes
`depth_attend` to `depth_attend_autodiff_s::<CudaBare, NoCheckpointing, 64>` / `::<…, BalancedCheckpointing, 64>`
and returns `Some`. And `fused_attnres.rs:1465-1473`:
`pub fn depth_attend_autodiff<Inner, N>(…) { depth_attend_autodiff_s::<Inner, NoCheckpointing, N>(…) }`.

Both sides run the **same fused op with the same monomorphisation**. The test asserts
`md < 1e-1` (dh) and `md < 1e-2` (dq) between two runs of one kernel. The name is false.

Consequence: `depth_attend_backward_tensor` (`fused_attnres.rs:1341-1385`), the hand-written
non-CUDA fallback used at `:1324` whenever the `TypeId` gate at `:1307` fails, has **no
numerical test at all**. Its only callers are that fallback and a bench (`:1092`).

This is the defect class `AGENTS.md` §3.2 already records for `burn-gdn2`:
*"`fused_chunk_verify.rs:132` … compared that tensor adjoint against the tensor path, i.e.
**verified the tensor adjoint twice**."* Same mistake, different crate, still live.

**Adversarial counterweight, and the report misses this too:** the crate's *other* backward
test, `fused_backward_matches_burn_autodiff` (`fused_attnres.rs:1666-1717`), **is** a real
oracle. Its reference is `raw_depth_attend` (`:1642-1662`), built from raw burn ops
(`stack_ad` / `powf_scalar` / `sum_dim` / `softmax`) on an autodiff device, so burn's own
autodiff differentiates it. It asserts `worst < 1e-4` for `dh[0]` and `dq`. The fused
backward kernel is therefore **numerically verified**, at `(l,b,t,d) = (3,1,2,8)`, for
`dh[0]` only. The report's claim 10 ("structurally incapable of failing on any paper
divergence") is true of the *formula*; it is false as a blanket statement about the backward,
and the report reached it without reading either test.

#### 3.5 SEVERITY 5 — `two_phase_attend`'s `n == 0` path returns zeros, and its comment promises the opposite. This is the first block.

`lib.rs:414-416`:

```rust
// With no completed blocks the inter-block term is empty; Phase 1
// statistics default to -inf/0 so the merge reduces to the intra-block
// attention (Algorithm 1 line 8: hl = ol / ll for i = 0).
```

and `lib.rs:459-461`:

```rust
let h = if i == 0 || n == 0 {
    // Algorithm 1 line 8: first layer attends inter-block only
    o1_i.div(l1_i.clamp_min(1e-10))
```

Two problems, both invisible to the report:

* The comment says the empty-inter-block case "reduces to the **intra-block** attention". The
  code takes the `n == 0` branch, which returns `o1/l1` — the **inter-block** term. With
  `blocks` shaped `[0, d]`, `o1 = Σ_j w1_j·kv_j` over an empty `j` is **zero** and `l1` is
  **zero**, so `out = 0 / 1e-10 = 0`. Every layer of the first block gets `h = 0` and
  `partial` stays `0`.
* `m1 = scores.max_dim(1)` at `:430` is a **reduction over a zero-length axis** when `n == 0`.
  Its value is backend-dependent. If it is `-inf`, `w1 = exp(-inf - -inf) = NaN` and the
  result is `NaN`, not `0`. **UNVERIFIED** — settle with
  `cargo test -p burn-attnres --features cuda two_phase` after adding
  `two_phase_attend(Tensor::zeros([4,8]), Tensor::zeros([0,8]))` and asserting the output is
  either the correct Eq.-6 answer or a named error.

Either way this is a **SILENT fallback** by ADR-0011's own taxonomy: the most common entry
point (the first block) returns a wrong answer or NaN with no error, and the comment
describes the opposite of the code. Neither existing test reaches it —
`two_phase_merge_matches_full_attention` uses N=2 (`:497`), `two_phase_shapes` uses N=3
(`:509`).

The paper is not blameless here: **Alg. 1 is undefined for the first block** (`l_l^(1)` over an
empty source set is `0/0`), and Eq. 6 is what supplies the real answer (`V = [b_0] = [h_1]`).
That is a **fourth paper ambiguity the report did not flag**, and it is the one that has to be
resolved for the crate to work at all.

#### 3.6 SEVERITY 6 — the crate's fused file cites a *different paper* than the crate's own header. The report's open question 1 asks where the `1/√d` came from and does not find the answer sitting in the file.

* `lib.rs:5-8` — the crate's own doc table: `| [2603.15031](https://arxiv.org/abs/2603.15031) | Full | … |`
* `fused_attnres.rs:1` — `//! Fused CUDA kernels for Attention Residuals (Kimi K3 §2.2).`
* `fused_attnres.rs:95` — `/// Chunked Full AttnRes (Kimi K3 §2.2, exact math, bounded memory).`

**Kimi K3 is arXiv 2607.24653** (per `docs/papers/gdn-kda.md:67` and
`docs/archive/research/2026-09-27-fused-inventory-attention.md:157`, both in this repo). So the fused
kernel file — 1821 of the crate's 2338 lines, and the file the report audits most heavily —
attributes the mechanism to a different arXiv paper than the crate header does, with a
section number attached. That is an AGENTS.md §1.7 "one word, one meaning" violation, and it
is a **plausible lead on the report's own open question 1**: K3 §2.2 is ordinary
*sequence-wise* attention, where `1/√d` is the standard convention. If the author ported from
K3's attention block and then wired it to 2603.15031's text, the `1/√d` is a habit from the
wrong paper, not a considered temperature argument. **UNVERIFIED** as intent — settle by
`git log -p --follow -- vendor/burn-fused/crates/burn-attnres/src/fused_attnres.rs | head -200`
and looking at the introducing commit's message; the repo's log for the crate starts at
`d1a76fe` (vendoring), so the original authoring history may not be reachable.

#### 3.7 SEVERITY 7 — smaller misses, all verified

* **§4 item 13 / Appendix B are correct but the paper's headline result is never mentioned.**
  The paper's actual contribution claim is a **scaling law** (§5.1, extraction lines 545-550):
  `Baseline: L = 1.891 × C^−0.057`, `Full AttnRes: L = 1.865 × C^−0.057`,
  `Block AttnRes: L = 1.870 × C^−0.058`. The report's §4 lists 18 *mechanisms* we do not
  implement and never says that the paper's claim is about **exponent in compute**, which the
  crate cannot test at any scale dormouse can reach. That is the framing a reader needs before
  the delta table.
* **Table 1 (§4.2, extraction 496-520) is a measurable claim the crate's benches do not test.**
  Block AttnRes: `N/S · d` read + `d` write in Phase 1, `3d` read + `d` write in Phase 2,
  total `(N/S + 5)d` ≈ 5.5d at L=128, N=8, S=16, m=4 — versus 3d for standard residual and
  34d for mHC. The crate has `benches/attnres.rs` measuring *wall clock*, and no memory-traffic
  counter. Not in the report's list.
* **`docs/archive/architecture/PLAN-minimal-core.md:90` is not cited in §3.3** even though §3.3 argues about
  severity *in the context of that A/B*. The report cites `library-crate-fate.md:62` and the
  README but not the plan line that names the arm.
* **A doc/code contradiction inside the crate, not in the table:** `lib.rs:419` says
  `// RMSNorm keys/values as in depth_attend` — but `depth_attend` normalises **keys only** and
  aggregates over **raw values** (`lib.rs:155` vs `:157-160`), which the report's own D15
  correctly calls a MATCH. The comment contradicts both the code and the report.
* **Test H's second bullet bakes in a paper misreading.** The report writes: *"`BlockAttnRes`
  with `N = 1` … must equal **standard residual accumulation with the embedding isolated as
  `b_0`** — i.e. **the output is the plain sum**"*. Eq. 6 with `N = 1` gives
  `h_l = α_0·b_0 + α_1·b_1^{l-1}` — a **two-way convex combination**, not a plain sum. The
  paper's §3.2 sentence (*"`N = 1` reduces to standard residual connections with the embedding
  isolated as `b_0`"*, extraction line 328) is itself loose on the arithmetic; the report should
  have filed that as a **fifth paper ambiguity** instead of writing a test that a *faithful*
  implementation would fail.
* **`CHUNK_G = 8` (`fused_attnres.rs:23`)** is annotated `// paper's N ≈ 8`. The report calls
  this a conflated constant. Worth sharpening: `CHUNK_G` controls a **CUDA chunk width** for the
  *Full* path (the chunked kernel is `depth_attend_cuda`, i.e. Full AttnRes), and the paper's
  `N ≈ 8` is a **Block** AttnRes hyperparameter. They are not merely unrelated — they are
  parameters of two different variants of the mechanism, and the comment asserts an identity
  between them.

---

### 4. Report errors, adversarial in both directions

Besides the above, the report's own text contains these, all verified:

1. **"7 blobs (1 PDF, 1 README, 4 PNGs)"** = 6 blobs. §1.2.
2. **"12 open issues"** = 13 open + 1 closed. §1.2.
3. **"2268 lines"** in §3.3 vs its own §1 measurement of 2338; `PLAN-minimal-core.md:90` says
   2060. §1.2.
4. **`crates/burn-attnres/README.md:3–:8`** — wrong path. §1.2.
5. **D3's remedy factor is `√d`; it is `d`.** And D3 is not an independent observable. §1.5.
6. **D2's "different ε placement"** is false — both put ε inside the root. §1.4.
7. **D6's "degenerate to a standard residual connection at every block boundary"** is an
   overstatement. The degeneracy is a consequence of D5 leaving **one** source in the state at
   the **first** boundary. From the second boundary on the state holds ≥2 sources and the
   output is a real (if wrong-set) softmax. The report's own Test C table only ever
   demonstrates the first.
8. **§3.1 vs SPECULATION self-contradiction on `streaming_matches_full_recompute`.** §3.1:
   it "compares `BlockAttnRes::step` against `BlockAttnRes::forward`, which calls **the same
   `step`** … It is a self-consistency check; it cannot see D5–D8." SPECULATION:
   *"`streaming_matches_full_recompute` … **should already be red** for the same reason."*
   Both cannot be true. `BlockAttnRes::forward` (`lib.rs:80-94`) creates a fresh
   `BlockAttnState` and loops `self.step(...)` — identical work, identical bugs. The test is
   **green by construction**; §3.1 is right and the SPECULATION note is wrong.
9. **Test A's gold literal is misprinted.** With the paper's RMSNorm and `ε = 1e-5`, `d = 4`,
   `w = [2,0,0,0]`, `h_0 = [1,0,0,0]`: `2 / √(0.25 + 1e-5) = 2 / 0.5000100 = **3.99992**`, not
   the report's `4.000002`. The report's *asserted* values are fine — `α_0 = 0.9820138`,
   `out[0] = 0.9820138` are both correct, because they are insensitive to that 1e-5 — so Test A
   still discriminates. But a "gold vector" table with a wrong 7-digit literal is precisely the
   failure mode the report's own §5 warns against.
10. **D13 misreads the API.** *"`BlockAttnRes::new(d, block_size)` takes a single vector for a
    whole block"*. `new` (`lib.rs:66-72`) takes `block_size: usize`, not a vector. The struct
    holds one `query: Param<Tensor<1>>` (`lib.rs:60`), which is per **module instance**, and
    `lib.rs:263` documents `step` as per layer. The conclusion (BENIGN) is right; the stated
    reason is not.
11. **§5's "Counts" and the table disagree in substance even though they sum right.** §1.8.

---

### 5. My own top-3 findings about the code, independent of the report

#### 5.1 `step`'s output depends on the current layer's own output (`lib.rs:272-291`)

Detailed at §3.2. Fires at every `i ≥ 2` in every block. The report identified the circularity
in the paper, applied the check to the function that avoids it, and missed it in the function
that does not. This is the most severe code defect in the crate and it is absent from the
delta table.

#### 5.2 `two_phase_attend` returns and re-uses `Σ h_l` where Eq. 5 requires `Σ f_l(h_l)` (`lib.rs:443-476`)

Detailed at §3.3. The function has no `f_l` in its signature, so it *cannot* implement Eq. 5;
its doc at `:397-399` tells callers to push the result into the block cache anyway. The report's
D11 praises this function's timing while the content is wrong, and the report transcribed
Eq. 5 correctly two pages earlier.

#### 5.3 The crate names two different papers, and the backward test that claims to cover the tensor path covers itself (`fused_attnres.rs:1`/`:95` vs `lib.rs:5-8`; `fused_attnres.rs:1591`)

Detailed at §3.6 and §3.4. Two things a reader of the report would otherwise take as settled:
the crate's provenance is internally inconsistent (and the inconsistency points straight at the
unexplained `1/√d`), and the test named `..._matches_tensor` cannot fail.

---

### 6. Recommended corrections to the report, in priority order

1. **Rewrite §1's oracle conclusion.** `AbdelStark/attnres` is a burn-native Rust port that
   agrees with the paper on D1, D2, D4, D5, D6, D7. Add it to the third-party table; downgrade
   "there is no executable oracle" to "there is no **authors'** oracle; there is one
   independent burn port that is diffable".
2. **Add D20: `step`'s self-loop** (`lib.rs:272-291`). Reclassify D11 into "timing = MATCH,
   content = BUG (D21: `two_phase_attend` uses `Σ h_l` for `b_n` and `b_n^{i-1}`)".
3. **Add D22: `n == 0` returns zeros / possibly NaN** (`lib.rs:430, 459-461`), and file the
   corresponding **paper** ambiguity: Alg. 1 is undefined for the first block; Eq. 6 is what
   supplies `V = [b_0]`.
4. **Reclassify D5 → CONTRACT** (same class as D10) and merge the two, noting the
   `lib.rs:76` vs `lib.rs:263` doc disagreement as the root cause.
5. **Fold D1+D2+D3 into one row** and state the composed factor as `1/d`, explicitly
   absorbable into `w_l`, with the real consequence being the effective LR on `w_l`.
6. **Fix the test inventory in §3.1** — add `depth_attend_fused_backward_matches_tensor`
   (tautology), `fused_backward_matches_burn_autodiff` (the one real oracle), `merge_fused_matches_tensor`,
   `streaming_fused_matches_tensor_path`, `balanced_checkpointing_reaches_the_seam_and_the_legacy_entry_does_not`;
   and state that `depth_attend_backward_tensor` has **zero** numerical coverage.
7. **Add the Fig. 9 source enumeration** as the primary evidence for D5/D6/D7 (it is structural,
   not an empirical heatmap), and file the `N = 1` arithmetic looseness as a paper ambiguity
   before writing Test H.
8. **Fix the four arithmetic/citation slips** (7 blobs / 13 issues / 2338 lines / the README
   path), delete D2's ε clause, fix D3's `d` vs `√d`, and resolve the §3.1-vs-SPECULATION
   contradiction on `streaming_matches_full_recompute` (it is green).
9. **Add a supersession line for `docs/archive/research/2026-09-27-attenres.md`**, and reconcile the three
   line counts (2060 / 2268 / 2338) so two documents on one crate do not disagree.

### 7. Single experiment that would falsify this review

Add one CPU-only test that (a) feeds `BlockAttnRes::step` five **layer outputs** `f_1…f_5` at
`S = 2, d = 4, w = [2,0,0,0]`, (b) asserts the exact Eq. 6 source set at every step against the
paper reference, and (c) asserts that at step 3 the source set is `[b_0, b_1, f_3]` and
**excludes `f_3` itself** from the mixture. If `out_3` is a function of `f_3` (i.e. §3.2 is
wrong), the self-loop is absent and the crate's `step` is closer to the paper than I have
traced. Settles §3.2, and re-ranks D5/D6/D7 in one run. Command:
`cargo test -p burn-attnres --lib step_source_set` (no GPU, no `cuda` feature).

## Reviewer B — Reviewer B — would following `docs/papers/attnres.md` actually fix `burn-attnres`?

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

### 1. The gold vector: the assertion is sound, the diagnostic is wrong, and one literal is unreachable

#### 1.1 The three `out[0]` values reproduce. The test is real and it is red today.

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

#### 1.2 `score_0 = 4.000002` is unreachable, and the table mixes two different `eps`.

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

#### 1.3 The table has three labels over four cells, and the two middle cells are the same number.

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

#### 1.4 Test C: the "ours today" column is exactly right; the stated margin is not.

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

#### 1.5 Test D: correct, and I confirmed the separation is real.

Modelling `two_phase_attend` with Phase 1 scaled and Phase 2 unscaled vs neither, at
`S=2, N=2, d=8`: maxdiff **0.430736** on both layer outputs, 0.861473 on the returned
partial. The `1e-3` threshold is not marginal. D4 is real (`lib.rs:428` scales, `lib.rs:450`
does not) and Test D is a sound gate. The one loose sentence: §5 says `l2 == 1.0` and
`m2 == s2` "hold only by accident of the missing scale" — they are hardcoded
(`lib.rs:451-452`) and are identities of a single-key softmax that hold *with* a scale
too. They are not scale-sensitive assertions.

---

### 2. Fix order

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

### 3. What is still unverified after following the report completely

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

### 4. The `1/√d` decision — does a scale factor qualify as bit-for-bit-preserving?

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

### 5. Is `burn-attnres` worth the work?

**Facts, all verified from this tree.** 2402 lines (517 `lib.rs` + 1821 `fused_attnres.rs`
+ 64 bench). Fate class **REFERENCE** (`docs/architecture/library-crate-fate.md:62`). Zero incoming edges
from `crates/dormouse-*` — grep over every `.toml` finds none, and `Cargo.toml:12` excludes
`vendor/burn-fused` from the workspace. It *is* compiled by CI
(`.github/workflows/fused-library.yml:163`, `--no-run` only). It **fails its own recorded
CUDA configuration**: `docs/archive/research/2026-09-27-fused-build-matrix.md:50` — `FAIL 11+1f/2ig` —
with the panic quoted at `:179-186`. It is already labelled **BROKEN** twice
(`docs/archive/research/2026-09-27-fused-inventory-attention.md:90`, `:404` of `adopt-vs-port.md`) and
**UNVERIFIED (d)** once (`oracle-audit.md:65`, "also self-recorded BROKEN").

**Cost of the report's plan, done properly:** one contract change + four source-set fixes +
one state fix + one scale deletion + six kernel signature changes + eight tests + a CUDA
re-validation that **no job in this repo can currently perform**.

**Benefit:** one A/B arm named at `docs/archive/architecture/PLAN-minimal-core.md:90` §M2, never run, on a
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

### 6. Top-3 findings

**1. The report's verdict sentence is contradicted by four in-repo documents, and the
contradicted claim is the one that matters most here.** `attnres.md:13-15` — *"The fused
CUDA kernels are internally consistent with the tensor path (so the existing parity tests
are green)"* — and `attnres.md:311-312` — *"the residual-state read→write hazard is
genuinely closed in the current tree"*. Meanwhile:
`docs/archive/research/2026-09-27-fused-inventory-attention.md:70-93` records
`streaming_fused_matches_tensor_path` **FAILED**, `step 3: maxdiff 0.83`, *"reproducible,
same step and magnitude across runs"*, and localises it: *"The isolated components pass
(`source_score_fused_matches_tensor`, `merge_fused_matches_tensor`), so the bug is in the
**composition** … not in either kernel alone."*
`docs/archive/research/2026-09-27-fused-build-matrix.md:50` + `:179-186` record the same failure at
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
