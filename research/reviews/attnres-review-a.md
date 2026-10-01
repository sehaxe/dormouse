# Review of `docs/papers/attnres.md` — independent verification

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

## 1. Verdict per claim

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

### 1.1 Claim 1 — CONFIRMED

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

### 1.2 Claim 2 — CONFIRMED as stated, but the operational conclusion drawn from it is REFUTED

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
* **"2268 lines"** in §3.3 is a **stale figure copied from `docs/library-crate-fate.md`**
  (recorded "at `422414c`"). The report's own §1 measures `lib.rs` 517 + `fused_attnres.rs`
  1821 = **2338**. `docs/PLAN-minimal-core.md:90` says 2060. Three documents, three numbers,
  none re-measured. AGENTS.md §1.4: "a measurement is a measurement only with the config, the
  date and the commit it was taken at."
* `crates/burn-attnres/README.md:3–:8` (§3.3) is a **wrong path** — the file is
  `vendor/burn-fused/crates/burn-attnres/README.md`. Its content is as quoted (3–8 does say
  "Not in the dormouse build").

### 1.3 Claim 3 — CONFIRMED, every line reference real

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

### 1.4 Claim 4 — CONFIRMED, with one false sub-claim

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

### 1.5 Claim 5 — CONFIRMED as far as it goes; it is incomplete

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

### 1.6 Claim 6 — CONFIRMED

`lib.rs:428` `.mul_scalar(scale)` on the Phase-1 scores; `lib.rs:450`
`let s2 = (q_i.clone() * p_norm.clone().reshape([1, d])).sum_dim(1);` with no scale, fed
straight into `m2` (`:451`) and the merge (`:466-471`). The two legs of Alg. 1 line 12 are
compared at temperatures differing by `√d`. Correct, and the report's observation that
`two_phase_merge_matches_full_attention` (`:492-503`) only exercises `i = 0` — which takes the
`i == 0 || n == 0` branch at `:459` and bypasses the merge entirely — is also correct.

### 1.7 Claim 7 — CONFIRMED, and under-supported by the report

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

### 1.8 Claim 8 — arithmetic right, substance wrong

9+1+3+1+5 = 19 = D1…D19. The report's own summary is internally consistent. But: D3 is not
independent of D1+D2; D5 is a CONTRACT not a BUG; and **three additional BUG-class defects
are missing** (§3.2, §3.3 below). Corrected count: **11 BUG-class** (D1/D2 fused, D4–D9, plus
three new), **2 CONTRACT** (D5, D10), 3 benign/ambiguity, 1 out-of-spec, 5 MATCH, **3 new**.
The headline "the crate is not a port" survives; the arithmetic does not.

### 1.9 Claim 9 — CONFIRMED

`crates/dormouse-core/Cargo.toml` and every other `crates/dormouse-*/Cargo.toml`: **no
reference to `burn-attnres`**. The workspace root `Cargo.toml:12` has
`exclude = ["vendor/burn-fused", "vendor/cubecl-fix", "vendor/cubek-fix"]` and `members` =
`cublas-poc, dormouse-core, dormouse-data, dormouse-train, dormouse-cli, backend-parity`. The
only inbound edges to `burn-attnres` are inside its own workspace: `vendor/burn-fused/Cargo.toml:4`,
`vendor/burn-fused/burn-fused/Cargo.toml:19,31,52,79`, `vendor/burn-fused/benches/Cargo.toml:12`.
`docs/library-crate-fate.md:62` records fate `b / REFERENCE`; `docs/PLAN-minimal-core.md:90`
(§M2) names it as a residual-stream A/B arm that has not been run. **No dormouse number is
retracted. This claim is the most important one in the report and it is correct.**

### 1.10 Claim 10 — CONFIRMED in direction, WRONG in inventory

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

## 2. Things the report got right that are worth keeping verbatim

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

## 3. What the report MISSED — ranked by severity

### 3.1 SEVERITY 1 — The "no executable oracle" conclusion is refuted by a burn-native Rust port the report never fetched

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

**And the repo already knew.** `research/2026-09-27-attenres.md:32` (two days earlier, same
tree): *"One Rust port exists upstream (`AbdelStark/attnres`, 54 stars) — ours is the more
complete one."* The new report neither cites that document nor engages with the port it names.
Two research docs on the same crate now coexist with **three different line counts** (2060 /
2268 / 2338) and **contradictory conclusions** (the old one says "the code works"; this one
finds 9+ bugs). AGENTS.md §1.7: *"a glossary that disagrees with the code is worse than none"*
— the same applies to two reports on one crate. **The new report should carry a line that
supersedes `2026-09-27-attenres.md`, and it does not.**

### 3.2 SEVERITY 2 — `BlockAttnRes::step` computes layer *i*'s output from an accumulator that already contains layer *i*'s own output. The report found this in the paper, praised the wrong function for avoiding it, and missed it in the function that matters.

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

### 3.3 SEVERITY 3 — `two_phase_attend` builds the block representation as `Σ h_l`, not `Σ f_l(h_l)`. Eq. 5 is unimplementable in that function and the report transcribed Eq. 5 correctly, then did not apply it.

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

### 3.4 SEVERITY 4 — `depth_attend_fused_backward_matches_tensor` is a tautology, and the fallback it appears to cover has zero numerical coverage

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

### 3.5 SEVERITY 5 — `two_phase_attend`'s `n == 0` path returns zeros, and its comment promises the opposite. This is the first block.

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

### 3.6 SEVERITY 6 — the crate's fused file cites a *different paper* than the crate's own header. The report's open question 1 asks where the `1/√d` came from and does not find the answer sitting in the file.

* `lib.rs:5-8` — the crate's own doc table: `| [2603.15031](https://arxiv.org/abs/2603.15031) | Full | … |`
* `fused_attnres.rs:1` — `//! Fused CUDA kernels for Attention Residuals (Kimi K3 §2.2).`
* `fused_attnres.rs:95` — `/// Chunked Full AttnRes (Kimi K3 §2.2, exact math, bounded memory).`

**Kimi K3 is arXiv 2607.24653** (per `docs/papers/gdn-kda.md:67` and
`research/2026-09-27-fused-inventory-attention.md:157`, both in this repo). So the fused
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

### 3.7 SEVERITY 7 — smaller misses, all verified

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
* **`docs/PLAN-minimal-core.md:90` is not cited in §3.3** even though §3.3 argues about
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

## 4. Report errors, adversarial in both directions

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

## 5. My own top-3 findings about the code, independent of the report

### 5.1 `step`'s output depends on the current layer's own output (`lib.rs:272-291`)

Detailed at §3.2. Fires at every `i ≥ 2` in every block. The report identified the circularity
in the paper, applied the check to the function that avoids it, and missed it in the function
that does not. This is the most severe code defect in the crate and it is absent from the
delta table.

### 5.2 `two_phase_attend` returns and re-uses `Σ h_l` where Eq. 5 requires `Σ f_l(h_l)` (`lib.rs:443-476`)

Detailed at §3.3. The function has no `f_l` in its signature, so it *cannot* implement Eq. 5;
its doc at `:397-399` tells callers to push the result into the block cache anyway. The report's
D11 praises this function's timing while the content is wrong, and the report transcribed
Eq. 5 correctly two pages earlier.

### 5.3 The crate names two different papers, and the backward test that claims to cover the tensor path covers itself (`fused_attnres.rs:1`/`:95` vs `lib.rs:5-8`; `fused_attnres.rs:1591`)

Detailed at §3.6 and §3.4. Two things a reader of the report would otherwise take as settled:
the crate's provenance is internally inconsistent (and the inconsistency points straight at the
unexplained `1/√d`), and the test named `..._matches_tensor` cannot fail.

---

## 6. Recommended corrections to the report, in priority order

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
9. **Add a supersession line for `research/2026-09-27-attenres.md`**, and reconcile the three
   line counts (2060 / 2268 / 2338) so two documents on one crate do not disagree.

## 7. Single experiment that would falsify this review

Add one CPU-only test that (a) feeds `BlockAttnRes::step` five **layer outputs** `f_1…f_5` at
`S = 2, d = 4, w = [2,0,0,0]`, (b) asserts the exact Eq. 6 source set at every step against the
paper reference, and (c) asserts that at step 3 the source set is `[b_0, b_1, f_3]` and
**excludes `f_3` itself** from the mixture. If `out_3` is a function of `f_3` (i.e. §3.2 is
wrong), the self-loop is absent and the crate's `step` is closer to the paper than I have
traced. Settles §3.2, and re-ranks D5/D6/D7 in one run. Command:
`cargo test -p burn-attnres --lib step_source_set` (no GPU, no `cuda` feature).
