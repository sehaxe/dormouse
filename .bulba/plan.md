# Plan: the minimal core — what leaves `dormouse-core`, and what dies

STATUS: AWAITING_APPROVAL

Full deliverable: **`docs/PLAN-minimal-core.md`** (written, 2026-09-27, snapshot
`d651eb1`). This file is the index.

## Goal

Move every mechanism and every kernel out of the model crates into the library,
so training is maximally easy and each mechanism is independently fixable,
testable and A/B-able (ADR-0017).

## Description

1. **Today's DAG** and its violations (M1-M6 model→library, X1-X6 the other way).
2. **Target DAG**: a move table (file → crate → LOC), the reverse list, and the
   3-5 entry-point API of each moved mechanism.
3. **Order of operations**: 12 commits C0-C11, each green alone, with the parallel
   lanes and the WAIT markers for in-flight files named.
4. **"Maximally easy to train"** as a checklist: what to read, what is on by
   default and what the evidence supports, what the default costs, which knobs are
   load-bearing.
5. **The deletion list**: 8 items, ~1 620 LOC, each with its evidence.

## Success criteria

- [ ] `dormouse-core/src` is `lib.rs` + `model.rs` + `loop_block.rs` +
      `attention.rs` + `config/` + `routing.rs` + a 12-line `aux.rs`, ≤1 250 LOC
      (`wc -l`; today 2 493).
- [ ] `cargo check -p dormouse-core -p dormouse-train --features dormouse-train/cuda`
      green at **every** one of C1..C11 (no commit red between two commits).
- [ ] Zero library-crate test file mentions `dormouse-core` (grep in CI).
- [ ] `moR`, `DM_FUSED`, `fnv_hash` return nothing from
      `grep -rn --include=*.rs --include=*.toml crates/ configs/`.
      How: the C2 deletion commit + the D1-D8 items.
- [ ] `configs/smoke.toml` + `--smoke` train 3 steps on CPU in <60 s.
- [ ] A newcomer reads 4 core files (≤1 150 LOC) plus a 60-line
      `docs/README-model.md` and knows every arm that is on, why, and whether it
      has been A/B'd.
      Measure: `wc -l` on the four files + the README.

## Acceptance

- [ ] C1's diff is provably a rename (`git diff --stat C1~1 C1` shows only names,
      paths, and the root `exclude`/`[patch]` lists).
- [ ] Each moved mechanism's library crate has a `#[cfg(test)]` module that does
      not import `dormouse-core`.
- [ ] The D1-D8 deletions are one commit with before/after `wc -l` in the message.
- [ ] The optimizer routing table lives in `crates/dormouse-core/src/routing.rs`
      and `validate_routing` (85 LOC) is gone.
      How: `grep -rn "expert_ffns\.\|engram\.key_projs" crates/` matches core, not train.

## User questions and answers

(none yet — five open questions below for the owner)

1. **MoR**: delete per D1, or land the in-flight agent's work and keep the arm?
   The code says it has never run a forward and the paper says it loses below
   135M (we are 7.5M). Recommend: delete.
2. **Default aux weights**: turn JEPA+DSpark OFF by default (removes a second full
   forward from every step, and it is the reason `base` OOMs at batch 6), or keep
   the current default until the A/B lands? Recommend: off, `--jepa-weight 0.1`
   as the arm.
3. **`configs/smoke.toml` + `--smoke`**: add, or leave the newcomer to write
   `--set engram_rows=4096` by hand? Recommend: add.
4. **The unwired library (12 797 LOC, 41%)**: publish a wired/implemented/unimplemented
   inventory in `dormouse-fused/README.md`, or start deleting unwired crates now?
   Recommend: inventory first (it is one file), delete on a second pass.
5. **`--max-iter 2`**: promote to the `small` default on the strength of
   AB-PROTOCOL 4b's cost argument, or wait for the depth curve? Recommend: run the
   A/B, then decide — the quality half is not measured.

## Tasks

- [ ] C0. Land or revert the in-flight MoR work (the tree does not compile).
- [x] C1. `refactor!: vendor/burn-fused → dormouse-fused` (script, one commit).
      **Landed 2026-10-01, owner-approved (goal.md, «РЕШЕНО»).** The DIRECTORY
      only: `vendor/burn-fused` → `vendor/dormouse-fused`, 315 tracked files by
      `git mv` (293 detected as renames). **Crate names are unchanged** —
      `burn-kda`, `burn-gdn2`, `burn-spectral` and the `burn-fused` facade
      PACKAGE keep their names; that is a separate decision with a far larger
      blast radius (published arXiv references, the two-routing-implementations
      debt in §3.3), and the owner's approval named the directory.
      Rewritten: root `exclude` + 16 path deps in 4 manifests, `fused-library.yml`
      (working-directory ×5, workspaces ×4, one nested `crates/burn-gdn2`), the
      library's 22 internal self-references, `tools/{build_lock,lib_gate,wt,
      test_targets,falsify_fused_adjoint,fused_matrix,migrate-dormouse-fused}.sh`
      + `{check_doc_refs,oracle_gate,gen_oracle_tiers,fused_matrix_static}.py` +
      `polar_probe.rs`, `.editorconfig`, `.github/CODEOWNERS`,
      `vendor/cubecl-fix/LICENSE`, 2 comment lines in `ci.yml`, 2 in
      `crates/dormouse-core/{src/loop_block.rs,tests/e2m1_oracle.rs}` (the
      latter is load-bearing: an `include_str!`), 1 in
      `crates/dormouse-train/examples/kda_alloc_probe.rs`, then 70 documents.
      `git grep vendor/burn-fused` → **0** outside `.bulba/` (agent scratch, and
      goal.md must keep the old name as the record of what was decided).
      `git grep -c vendor/dormouse-fused` → 390 occurrences / 118 files.
      Deviation from the task line: **three commits, not one**, per the lane
      brief (vendor → docs → site) and because a concurrent lane's `git commit`
      swept the staged vendor commit into `d2b0da0`. Gates: doc-refs (7 failures,
      all pre-existing — proven by stashing), docs-site ingest/build/check green,
      `cargo check -p dormouse-train --features cuda` green.
- [ ] C2. `refactor(core)!: delete what lost its A/B` (D1-D4, D7, D8; ~1 620 LOC).
- [ ] C3. `feat: dormouse-ema` (60 LOC). — parallel lane A
- [ ] C4. `refactor!: LinearLike → dormouse-linear` (165). — lane A
- [ ] C5. `feat: dormouse-residual` (117 + 120, 4 arms). **WAIT loop_block.rs** — lane A
- [ ] C6. `refactor!: act_quant → dormouse-bitnet` (170). **WAIT** lane A
- [ ] C7. `refactor!: host tables → dormouse-ngram` (340). **WAIT train/lib.rs** — lane B
- [ ] C8. `refactor!: HeadWiseMuon → dormouse-muon-plus` (120). lane B
- [ ] C9. `refactor!: stress + firewall → dormouse-stability` (112). lane B
- [ ] C10. `refactor!: the aux losses go home` (125). lane A
- [x] C11. `refactor!: the model names its own params` (routing in, `validate_routing` out). LAST.
      Landed as `831e3a0`. The trainer ASSEMBLES the groups from the declaration
      (`ParamGroup::from_ids`) and `check_installed` verifies the INSTALLED groups
      against the live tree, so a group is decided in one place. The audit's
      disagreement is real and is fixed: the live `muon_group()` excluded
      `\.s$` and `inner\.Dense\.bias$` but not `inner.Dense.weight`, so
      `--set use_tsct=false` put a dense expert's `[d,d]` weight on fp32
      Newton-Schulz — the ~40 s/step case §2.3 records as solved. DEViation from
      the task line: `validate_routing` is NOT deleted, the 85-LOC string
      validator it named is. The name stays because `train_loop` calls it at
      startup for the banner counts, and deleting the only startup gate to save
      9 lines is the wrong trade. `grep -rn "expert_ffns\.\|engram.key_projs"
      crates/` now matches core, plus 6 path strings in ONE train test that name
      which parameter each assertion is about (a rename fails it loudly) — no
      group is decided from one.
- [x] P-GATHER. port (Python): the SonicMoE gather arm, wired and gated.
      `moe_gather="kernel"` (`config.py`: one more switch on the routed pass, refused
      LOUDLY with `bf16=false`, `use_byteflow=true`, `moe_topk_mode != "off"` and
      `factor_qat != "none"`). `experts.SparseMoE._gather_dispatch` is
      `_triton_dispatch` minus its `xg = x[owner].to(dt)` line:
      `moe_sonic.gather_abi` + `gather_glu`. The prototype has NO adjoint, so
      `_GatherGlu.backward` re-runs the shipped `expert_gemm_glu` on the same rows and
      lets its Triton adjoint produce `dxg`; the fwd->bwd cache is `x`, `toks`, `y`,
      and the arm pays one extra forward per backward. Counters: `moe_gather_forward`,
      `moe_gather_rows` (+ `moe_gather_fallback` with the reason).
      Gates: CPU wiring parity vs `_naive_dispatch` (fwd 1.2e-7, dx 2.2e-7, 20 param
      grads 2.1e-7 at fp32, bound 1e-5; six scratch reverts turned it red — see the
      lane report), the counters (including a spy that pins the branch CALLS the arm
      it counts), the counted CPU fallback, and the four `validate` refusals. GPU
      parity vs the triton arm (logits AND gradients, 1e-5, fp32+bf16) is written and
      NOT run (the card was held).
      Bytes from the code's own layout (`moe_sonic.wired_pass_bytes`, asserted by the
      CPU suite): medium forward 1715 -> 1109 MiB (-35%), cache 382 -> 352, PASS
      2739 -> 3336 MiB (+22%); b1 forward 2229 -> 1835 MiB (-18%), cache 245 -> 235,
      PASS 2911 -> 4405 MiB (+51%). The FORWARD is a byte win (the prototype's own
      finding) and the PASS is a byte LOSS, for a reason the prototype did not price:
      the recompute is a whole forward (`grad_out` fp32 vs `dy` bf16 is still the
      second reason). `scripts/bench_gather.py` (both shapes: bytes, ms fwd,
      ms fwd+bwd, peak GiB, preflight) is written and NOT run. NO SPEED CLAIM EXISTS.
- [x] P-MOE-SEAM. port (Python): cast the MoE input at the LOOP SEAM, not in the
      gather. `loopblock.LoopBlock._moe_branch` casts `normed` to
      `SparseMoE._bank_dtype()` once and widens the output before `* w_ffn`;
      `SparseMoE._gather_rows` is `x[owner]` with the cast only for a direct fp32
      caller (a third dtype asserts). The shipped gather chain moves 12·R·d ->
      4·R·d (medium 576 -> 192 MiB, −384 MiB/pass; b1 384 -> 128 MiB, −256
      MiB); `moe_sonic.X_BYTES` 4 -> 2 and `pass_bytes(x_bytes=4)` keeps the fp32
      seam as the priced control. Gates: max|Δlogit| = 0.0 against one bf16 eps ×
      logit scale (3.8e-2) and |Δloss| = 0 on the fixed fixture — the seam is
      idempotent with the cast the MoE already did, and a dtype spy fails if it is
      removed; `bf16=false` is `torch.equal`; one `.to(dt)` at the seam and zero in
      the gather (Tensor.to meter); no host sync added; padded vs naive agree at
      bf16; triton vs both is a CUDA gate written and NOT run. Six scratch reverts
      turned the named gate red. `scripts/bench_moe_seam.py` prints both seams at
      medium and b1 and times fwd / fwd+bwd only on a free card — NOT run, NO
      TIMING EXISTS. Landed 2026-10-21. Supersedes P-GATHER's byte
      line: medium shipped forward 1715 -> 1235 MiB, shipped pass 2739 -> 2259;
      wired forward 1235 -> 1013, cache 382 -> 304, pass 2259 -> 2760 MiB.
- [x] P-MOE-BATCH. port (Python): `moe_kernel="batched"` — the dropless MoE
      bank. `experts._batched_dispatch` reuses the padded arm's `[E, cap, ·]`
      body (now `_bank_dispatch`) and sizes `cap` to the batch's largest group
      when `moe_capacity = 0` (`_batched_cap`), so the per-expert Python loop's
      E tiny GEMMs become three `bmm`s. `moe_capacity > 0` uses the capacity
      rule's cap and the padded arm's exact drops; heterogeneous expert shapes
      are asserted LOUD (`_assert_homogeneous`, `Escape:` names per-expert
      rank), the counter is `moe_batched_calls`, and the default stays `padded`
      (batched is opt-in). Gates `tests/test_moe_batched.py` (a-h): batched vs
      the loop 4.8e-7 fp32 / 0.0 bf16; the float64 reference worst 4.8e-6 on
      logits and every parameter gradient; unequal 1/40 groups; capacity drops
      bit-identical to `_padded_dispatch`; padded bank rows and dropped router
      weights get exactly 0 gradient; heterogeneous refusal; zero host syncs on
      the static-cap path (the dropless cap is ONE `counts.max().item()`, named);
      `full_graph` allows static-cap batched and refuses dropless. Eight scratch
      reverts each turned their named gate red and restored sha256. Bench
      `scripts/bench_moe_paths.py` (m400 / b8s1024 / b1s8192; naive / triton /
      batched; fwd + fwd+bwd + launches) written and **NOT run** (this task made
      no GPU measurement). No GPU timing exists.
- [x] P-MOE-BUCKETS. port (Python): `moe_kernel="bucketed"` — the batched bank
      without the memory cliff. `experts._bucket_plan` classes each batch's
      experts by row count (within 2x share a bucket, capped at
      `MOE_BUCKET_ROWS = 2048` rows; an expert past it gets a bucket of its
      own); `_bucketed_dispatch` writes every assignment ONCE into a flat
      scratch addressed `bucket_offset + local_id·cap_b` and runs the existing
      three-`bmm` bank body per bucket, concatenating in bucket order before the
      ONE `slots` gather. Memory is `Σ n_b·cap_b·d·itemsize` ≤
      `2·MOE_BUCKET_ROWS` rows per class bucket, and the single-expert collapse
      is `1·count·d` (8 192 × 1 536 × 2 B = 25 MB), not `E·max_group·d` (the
      12 GiB ask measured 2026-11-04). `MOE_BUCKET_BYTES = 256 MiB` is the LOUD
      refusal with the busiest bucket's measured numbers; counters `moe_buckets`
      and `moe_bank_bytes`; ONE host read (`counts.tolist()`, E ints) is the
      plan's documented sync. Gates `tests/test_moe_buckets.py` (a-f): one
      bucket `torch.equal` to `_bank_dispatch` in fp32 and bf16, dropless and
      capacity (32 768 B); 90%-on-one-expert matches the per-expert loop at
      4.6e-8 with 2 buckets / 54 016 B while `_batched_cap` refuses the same
      routing; counts `[1,1000,1,1]` and `[1000,1,1,1]` plan 2 buckets each
      (Δ 0.0); capacity drops give exactly 0 to output and router gradient;
      the only reported sync is `tolist`; the plan is deterministic (and the
      same counts in a shuffled token order give the same plan). Six scratch
      reverts turned their named gate red and restored sha256. `config.
      MOE_KERNELS` grows to five literals; `full_graph` REFUSES the arm (the
      plan is a host-side allocation shape). `scripts/bench_moe_paths.py` gains
      `--arm` and `--routing pathological` and prints `moe_bank_bytes` per arm
      (m400 / b8s1024 / b1s8192) — written and **NOT run** (the card is held;
      no GPU timing exists).
- [ ] C12. `feat: smoke preset + --smoke`, `docs/README-model.md` (≤60 lines).
- [x] P-HALT. port (Python): the halt head + the depth-8 path. `port/src/dormouse/
      model/halt.py` — BCE target read from the model's OWN per-iteration CE
      (stop iff `CE_i − CE_{i+1} < halt_eps` or the cap's last iteration), the
      head built only at `halt_weight > 0` so every preset count is unchanged;
      config `halt_weight`/`halt_eps`/`halt_min_iter`/`halt_max_iter`/`halt_mode`
      with loud refusals; the inference stop (`halt_mode`) really shortens the
      loop and counts `halt_skipped`; `max_iter=8` builds/trains/evaluates;
      `port/scripts/ab_depth8.py` (`mi4` vs `mi8`, 3 seeds x 2 000, medium,
      `eval_depths`) written and **NOT run** (card busy). Landed 2026-10-21.
- [x] P-MOE-BALANCE. port (Python): the routing controller that collapsed the
      bank, and the counters that make it visible. Evidence: `m400` step 149 999
      carries `lfb_bias ∈ [-218.27, +72.67]` (mean +34) and keeps **9 of 512**
      experts alive (one holding 4096 of 8192 assignments); the bias-less router
      reads 80/512, so the SIGN controller is the cause (the first histogram
      measured the wrong thing — `route_hist.py` skipped the `moe` section).
      Mechanism: `sign` is a cumulative bang-bang integrator with no equilibrium
      and no bound. Added, all behind flags and default OFF (`sign` untouched):
      `moe_lb_mode="prop"` (per-WINDOW relative deviation, `gamma=0.001`,
      clamped ±`moe_lb_clamp=2.0`), `moe_lb_mode="aux"` (Switch/GShard
      `coef·E·Σ f·P` added to the loss, gradient to the router only),
      `moe_routing_health` (live / max-share / top-10 / normalized-entropy
      device counters per forward, drained on the log cadence) and a resume
      refusal past `max|bias| > 10` plus a one-line range statement. Gate
      `port/tests/test_moe_balance.py` (a-f + the divergence reproduction):
      3 000 pinned-skew steps end at ±150 under `sign` and exactly ±2.0 under
      `prop`, which then recovers 15/16 live; the OFF path is bit-identical to
      main's digests captured before the edit. `port/scripts/ab_balance.py`
      (sign / prop / aux, 2 000 steps, b4×s1024) written; the 2 000-step run is
      **NOT run** — the acceptance number (live ≥ 80% of E, top-10 ≤ 30%) is
      outstanding.
- [x] P-JEPA. port (Python): JEPA as ONE mechanism. `port/src/dormouse/model/
      jepa.py` — a causal latent predictor (`d(x) -> d/4 -> d`, `1 - cos(pred(h[p]),
      sg(h[p+h]))`) with an INTERNAL SIGReg anti-collapse term on the predicted
      latents (`jepa_reg_weight`, default 0.1), so the mechanism is self-contained
      rather than a predictor plus a separate regulariser arm. Config `jepa` /
      `jepa_weight` / `jepa_horizon` / `jepa_reg_weight` with LOUD refusals
      (`use_byteflow`, `segment_len > 0`, `jepa_horizon < 1` or `> seq_len // 8`);
      `jepa_weight = 0` builds no head and is bit-identical. The 200-step collapse
      gate measures per-dim std **0.061 with the reg vs 29 351 without** (a
      degenerate, scale-diverged latent) — the no-reg arm does NOT collapse to a
      point at this scale (a cosine loss is scale-blind), and the test says so.
      `port/scripts/ab_jepa.py` (off / h1 / h2, 3 seeds x 2 000, medium) written
      and **NOT run** (card busy). Landed 2026-10-21.
- [x] P-QAT-STACK. port (Python): the per-step refresh is ONE call per bank
      site, not one per tensor. `model/qat_cache.py` adds
      `qat_project_stacked` (`w + alpha·(q(w) − w).detach()` over a whole
      `[E, ·, ·]` stack in one `qat_values` call, residual folded in place) and
      `QuantCache.refresh_stacked` (`refresh` is the `N = 1` delegation).
      `experts.SparseMoE._banks` builds the `[E, ·, ·]` fp32 stack per projection
      and projects it once; scales and the non-QAT path are untouched.
      Counters `qat_refresh_calls` / `qat_refresh_tensors` (one stacked call = 1
      call, E tensors) surface on the forward's counter line. The medium bank
      site measured **768 calls / 768 tensors -> 3 calls / 768 tensors**
      (E=256), gated in `tests/test_qat_cache.py` against the pre-fix strategy on
      the same weights; bit-identity (ternary/binary/mxfp4 x alpha 0/0.5/1,
      `torch.equal`), the STE gradient contract (1e-6), the `none/none` zero
      refresh, and the no-host-sync gate are pinned there, each proven by a
      scratch revert. `port/scripts/bench_qat_cost.py` now prints bf16 /
      per-tensor / stacked ms/step and the launch counts — written, **NOT run**
      (card busy): **NO TIMING EXISTS** for the stacked path. Landed 2026-10-21.
- [x] P-QAT-ATTN. port (Python): `dense_qat_skip_attn` (default **true**) keeps
      attention in bf16 under `dense_qat` — the BitNet b1.58 rule (arXiv
      2402.17764), which our own `act_quant` already encodes as "attention is
      never below 8 bits". Measured 2026-10-21 (medium b8 s512, 2000 steps,
      81 920 B window, 3 seeds): projecting the ten KDA/MSA sites scored BPB
      **3.2733 vs the bf16 control's 3.0750** (+0.198, +6.5%).
      `tsct.DENSE_QAT_ATTN_SITES` is the ONE per-site list the implementation
      and the coverage test read; `false` restores the measured arm bit for bit
      (golden fixture, `torch.equal`), and gates a-e pin the skip, the restore,
      the bf16 master, the inert `dense_qat=none` case and the no-host-sync
      path, each proven by a scratch revert. `scripts/ab_ternary_form.py`
      (arms f / b / n; `a` dropped — no int8 weight codec, only an activation
      seam) is written and **NOT run**: the release-form verdict needs its 3
      arms. Landed 2026-10-21.
- [x] P-LAUNCH. port (Python): measure the launches behind the b8-s512 3.55x
      per-byte gap, and gate the two legal levers. `scripts/launch_profile.py`
      profiles ONE full medium step (bf16 and ternary): `cudaLaunchKernel`
      count, executed-kernel phase split (seam annotations for kda/fla, moe,
      qat refresh, clip, optimizer; name fallback for the rest), GPU-busy union
      vs wall, top-15 by count and by time. `scripts/compile_retry.py` re-runs
      `torch.compile(dynamic=False|True)` after the `@torch.compiler.disable`
      fix, 20 steps, and exits 1 on >20 captured graphs or a `recompile_limit`
      fallback. `docs/reviews/2026-10-21-launch-bound.md` states why a full-step
      graph needs the padded arm (dropless `offs` is data-dependent), prices the
      cap 2.0 77.4% / cap 64 + LFB 3.8% drop trade, proposes padded cap 64 +
      LFB + graph vs dropless + eager (3 seeds, BPB + ms/step), and lists the
      prerequisites in landing order. `scripts/bench_shape.py` is the ternary
      shape x throughput x eval-window table. All three preflight and print
      SKIP on a busy card; the card was busy, so **NO TIMING EXISTS**. Landed
      2026-10-21.
- [x] P-CHEAPWIDTH. port (Python): make `d_model` stop setting the dense bill
      with three OFF-by-default flags (all three off = `main` bit for bit,
      sha256 state_dict + logits pinned from a clean `main` worktree).
      `attn_latent_rank` (MLA-style `d→r→d` shared latent for KDA's q/k/v/g:
      `5·d·r` instead of `4·d²`), `n_kv_heads` (GQA on k/v, expanded
      `repeat_interleave(h/h_kv)` before the kernel so FLA still sees
      `[b,t,h,hd]`; projection cost `2·d²→2·d·h_kv·hd`), `out_proj_rank`
      (readout `d→r→d`, `2·d·r` instead of `d²`; TSCT rank override, dense
      two-linear on `use_tsct=false`). `dormouse.width` prints the arithmetic:
      at `iters=3`, `d=1024` the dense bill is **24.12M MAC/token**; `latent`
      (r=256) **15.46M (35.9%)**, `gqa` (2/16) **18.61M (22.8%)**,
      `latent+gqa` **14.09M (41.6%)**, `all` **12.52M (48.1%)**; at `d=4096`
      (r=1024) the same three rows cut 35.9/25.3/42.2/48.7% off **385.9M**.
      The predicted parameter deltas are checked against `dormouse info`'s
      instantiated totals (latent −2.88M, gqa −1.84M, out_proj −0.52M,
      all −3.87M at medium, dense rung). Refused LOUDLY: negative flags,
      `n_kv_heads > n_heads`, non-divisible heads, `out_proj_rank > d_model`,
      and either attention flag with `use_kda=false`; `attn_latent_rank` is
      ALLOWED with `factor_qat != none` because the latent and the four
      projections are dense sites and `factor_qat` reaches only TSCT pairs
      (checked, stated, gated). Gates a-g pinned in `tests/test_cheapwidth.py`
      (OFF parity; MAC-vs-info; named gradients; each flag moves the logits;
      stub-FLA shape contract over all 8 combinations; refusals; no host sync),
      each proven by a scratch revert. `port/scripts/ab_cheapwidth.py`
      (dense / latent / gqa / latent+gqa / all, 3 seeds, 600 steps, medium,
      web.bin + its eval, JSONL in `~/logs/ab_cheapwidth/`, `--profile`) is
      written and **NOT run**: the card is busy, so no A/B number exists.
      Landed 2026-10-23.
- [x] D1. `docs: retract the false step-time and verification claims` (prose only,
      no Rust). **Landed 2026-09-29.** Withdrawn without replacement, because the
      corrected value is not known: the 25.8 s/step batch-8 figure (no committed
      log, no `benches/history.tsv` row, struck there as measured-under-load);
      the 3076 ms no-attention figure beside it (same); the `1810 → 365 ms`
      subtraction in `kda-sota-ceiling.md` §4.2 (both endpoints step-0);
      the attention backward's true cost (never measured - no run on record has
      executed one); the `~465 ms` as a measured fixed cost and the
      `0.067 ms/token` coefficient it was fitted with (the fit overpredicts 3x at
      the batch-8 shape); the `1.6 s/step` A/B budget and every `cost` cell
      derived from it; the scale ladder's whole wall-clock column; the
      `9 h`/`30-90 days` pair in `README.md` (e), which also disagreed with
      itself. Corrected WITH a sourced value: the 10 809 ms anecdote in
      `VERIFICATION.md`; the `bit_exact.rs` "pattern to follow" recommendation
      (it is RED 976/1000 and its 1000-case test is feature-gated off the
      default cell); the `--seed` claim in Layer 2 (`4b42b6d` seeds the init, and
      4 % of the model is still process entropy); "layers 1 and 2 pass today"
      (Layer 2 is unwritten); the `VERIFIED` row for the attention arm
      (`fused kda=64/0` in both arms of the 2026-09-29 preflight, and
      `autodiff_nested_balanced` is RED); "no CUDA graph capture in the
      burn/cubecl stack" (it is 5/5 green on this GPU); the `--bf16` flag
      experiment in `PLAN.md` (bf16 matmul cannot work on this backend);
      the `6.7-8.3 s/step` baseline and `KDA ~80 %` in `PLAN-minimal-core.md`;
      the `2-5 %` arithmetic share (~16 % against a warm step). Six research
      documents corrected in place, none deleted. Every correction names
      `benches/history.tsv`, a commit, or a `file:line`.
      **Deliberately left alone:** the ~465 ms order-of-magnitude reading in
      ten files is *vindicated* by the 245 ms warm step, and the launch-bound
      reasoning that rests on it is now confirmed by an independent instrument
      (13.3 % mean GPU utilisation, 79 % of samples ≤5 %).

- [x] P-CSA. port (Python): **compressed, selectable KV memory** as an ADDITIVE
      branch beside KDA. `model/csa.py` + `use_csa`/`csa_chunk`/`csa_kv_dim`/
      `csa_topk` on `ModelConfig`. Per `csa_chunk` byte positions ONE shared
      `k`/`v` (MQA, `csa_kv_dim`), `k = mean_pool(x) @ W_k`, `v = mean_pool(x)
      @ W_v` on the block's RMS-normed input; each query scores its own `q`
      against the keys of the STRICTLY EARLIER chunks, keeps top-`csa_topk`,
      and runs exact softmax attention over them; `W_o` back to `d_model`.
      Added to KDA's output behind a learned per-iteration `csa_gate` (the
      `msa_gate` pattern); `use_csa=false` builds nothing (two pinned
      `state_dict`+logits shas from clean main `d80888d`). Decode appends one
      entry per finished chunk to a per-iteration ring (`CSAState`, bf16 =
      2 B/elem, doubling, `DECODE_MAX_CHUNKS=131072`) and refuses LOUDLY past
      the cap. Memory helper `bytes_per_byte = 2·csa_kv_dim·2 / csa_chunk`:
      **32 B/byte at kv=64/chunk=8 → 1.0 MiB at 32k, 32 MiB at 1M per layer
      per iteration slot**, asserted against the actual tensors. Refused
      LOUDLY: `use_msa` (two sparse branches), `segment_len>0` (half-full chunk
      at a cut undecided), `use_byteflow` (latent vs byte chunk), `use_kda=false`
      and `attn_kind='full'`; chunk ∉ {4,8,16,32}, width ∉ {64,128}. Gates
      `tests/test_csa.py` — 10 gates, each proven by a scratch revert:
      bit-identical off, both-way causality, strict-causal parity at chunk=1
      (1.5e-7), W_k/W_v/gate gradients, live top-k, refusals, no host sync,
      1.0 MiB/32k actual state, ring append/refusal, recurrent state carry.
      Full CPU suite: 1055 passed / 70 skipped vs main's 1045/70, the only red
      is the pre-existing wall-clock `test_per_byte_cost_does_not_grow_with_context`
      under a live training run (green on rerun; `.bulba/memory.md` records it).
      `scripts/ab_csa.py`: off/csa8/csa16, 3 seeds x 600, `medium`,
      mix4/web.bin + web_eval.bin, JSONL to `~/logs/ab_csa/`, pre-registered
      tie on the 512 B window + a `--longctx` 4096-byte held-out BPB table.
      Written, **NOT run** (the brief forbids training arms). Decode cost,
      stated: **O(t/chunk) per byte** — at 1M context 131 072 keys × 64 =
      8.4M MAC + 131k comparisons per position/iteration/layer; the arm is a
      4k-32k arm, not an O(1) 1M decoder.

- [x] P-TRAIN4. port (Python): four OFF-by-default frontier train knobs, each
      with its own gate and a bit-identical off path.
      `train/loop.py` + `config.py` + `model/model.py`:
      `grad_accum: int = 1` (loss/`k` before backward, optimizer once per group,
      `step`/LR/eval/checkpoint cadence per group; dropless gate == `batch·k` at
      5e-7; LOUD `grad_accum<1` and `grad_accum>1`+`opt_graph`; halt/segment/rand_depth/
      host-Engram decided ALLOWED with the reasons in the field docstring);
      `ema_decay: float = 0.0` (shadow of `model.parameters()`, `lerp_` after the
      optimizer step gated by the device firewall, eval and `load_model_half`
      read it, checkpoint keeps raw `model` + `model_ema`; `ema_decay` is a
      PROGRESS key, so changing it on resume is legal — both copies are in the
      file; refused with `muon_sf` and outside [0,1); `opt_graph` allowed because
      the update is eager outside the capture); `tie_embeddings: bool = false`
      (`TiedHead` shares `embed.weight` transposed, zero own params; on `use_tsct`
      it replaces the low-rank head, on dense it saves exactly `d·vocab`);
      `z_loss: float = 0.0` (`coef·mean(logsumexp(final logits)²)`, `pure` skips,
      counter `z_loss`). Gates `tests/test_train4.py` (16; each proven red by a
      scratch revert: no `/k` -> loss 4.96 vs 9.92; EMA no-op -> 5.7e-3 vs 1e-6;
      tie off -> missing `lm_head.weight` in the load; z off -> loss/grad equal).
      `scripts/ab_train4.py`: control/+accum/+ema/+zl/+tie, 3 seeds x 600,
      `medium`, mix4/web.bin + web_eval.bin, COMMON from `ab_mhc.py` minus
      `opt_graph` (refused with accum, applied to every arm), JSONL to
      `~/logs/ab_train4/`, pre-registered TIE on all four. Written, **NOT run**.

- [x] P-FULLGRAPH. port (Python): ONE CUDA graph for the WHOLE step behind
      `full_graph` (forward + backward + the device firewall + `zero_grad` +
      the optimizer stage), plus the 2026-10-23 b24 s1024 capture fault.
      `train/graph.py::FullGraph` (subclass of `OptimizerGraph`, so the two arms
      cannot bind the LR or the refusals differently): static `[batch, seq]`
      device inputs filled with `copy_`, pre-allocated `.grad` zeroed in place
      (`set_to_none=False`), the LR a device tensor filled in place, the
      forward's PYTHON counters re-merged on every path (`on_counters`) so a
      COUNTED arm cannot go silent, and `_check_participants` re-checking the
      recorded grad set per replay (a host compare, no sync).
      `train.graph.full_graph_refusal` refuses LOUDLY, one reason + `Escape:`
      per REAL conflict, each verified against the forward: `opt_graph` both
      set, `grad_accum>1`, `rand_depth`, `segment_len>0`, `use_byteflow`, halt
      (`loopblock.py:674`'s host read), `use_msa` (`:576`'s `float(share)`),
      `ema_decay>0`, `engram_ram` (`_forward_host`'s `.to("cpu")`), JEPA's
      per-step mask, DSpark's `last_mtp`, `act_quant` (`:240`'s
      `bool(torch.isfinite(x).all())`), the QAT caches. Data-dependent routing
      is ALLOWED and stated: the top-k is a kernel on buffer values.
      **The fault's root cause, measured:** `torch.bincount` is NOT
      CUDA-graph-capturable on torch 2.13 (`with torch.cuda.graph(g):
      torch.bincount(a, minlength=9)` raises "Cannot copy between CPU and CUDA
      tensors during CUDA graph capture unless the CPU tensor is pinned"), and
      `SparseMoE._padded_dispatch`/`_triton_dispatch`/`_permute` counted experts
      with it — so NO capture of a training step could complete at ANY MoE
      shape. `experts._bincount` is the same histogram as one device
      scatter-add; `tests/test_full_graph.py::test_bincount_aborts_a_capture_and_the_replacement_does_not`
      is red on revert.
      **The SECOND blocker, found and fixed 2026-10-24 (`capturable` lane):**
      `order[keep]` — a BOOLEAN-mask index — inside `SparseMoE._padded_dispatch`
      (`model/experts.py:1354` and `:1359`) and in the triton/gather/naive arms
      (`:1745`, `:1837`, `:1397`). A bool mask is `nonzero` underneath: a D2H
      read AND a kept-row count that changes with the routing, so a graph can
      neither record it nor replay it correctly. The abort does not name the
      op — it is `cudaErrorStreamCaptureInvalidated` at the NEXT call — so the
      diagnosis is a `TorchDispatchMode` log of the last 12 dispatched ops
      (`port/scripts/probe_capture.py`; per-op table in `probe_ops.py`:
      bool-mask index, `unique`, `nonzero`, `.tolist()`, `.item()`, `bool(t)`
      and `repeat_interleave(tensor)` FAIL; `argsort(stable)`, `index_put`,
      `index_add_`, `bincount`-as-scatter, `bmm`, `cumsum` capture).
      **The fix is shape-static rows, not an op swap:** the padded arm keeps
      ALL `n·k` assignments, sends a dropped one to one dump slot
      (`e·cap`, sliced off before the bmm so the expert pass keeps its exact
      `[e,cap,d]` shape) and multiplies it by the constant 0 mask — a dropped
      row adds exactly 0.0 to the scatter and 0 to both gradients, which is the
      eager semantics, not an approximation. Still bit-identical, measured where
      it matters: the default path's pinned 5-step sha is unchanged, the padded
      arm vs the naive reference is 0.0 (bf16) / 1.2e-6 (fp32) max-abs, and the
      first replay vs the arm's OWN eager program is 1.5e-8 (fp32 ulp).
      The other three arms are built on the ragged count itself (`offs`, the
      per-expert `unique().tolist()` loop, the gather ABI's permuted rows), so
      `full_graph_refusal` REFUSES them LOUDLY with the escape (moe_kernel
      'padded' + moe_capacity>0 + moe_gather 'off') instead of failing mid
      capture — one refusal per arm, 4 gates.
      `FullGraph.capture` also needed torch's stale-stream override: the
      LAST WARMUP step's `AccumulateGrad` node still refers to the DEFAULT
      stream, and autograd refuses to `cudaStreamWaitEvent` on stream 0 inside
      a capture (`RuntimeError ... stale reference to the default stream`).
      `torch.autograd.graph.set_override_stale_capture_stream(True)` around the
      one capture redirects it to the capturing stream; reverted to the
      documented process-initial `False` afterwards.
      `opt_graph`'s "+0 at b8 s512" was not a graph result either: the capture
      was REFUSED for a month because `loop.residual_scale` gets no gradient
      under `use_mhc` (the reverted `a10f455`), and the trainer now excludes a
      structurally dead parameter COUNTED (`opt_graph_frozen`) while
      `_check_participants` catches a real per-step starvation. Captures now
      `empty_cache()` and refuse COUNTED below `max(384 MiB, 8% of VRAM)` of
      headroom with the numbers.
      Measured 2026-10-24 (min of 5 for b8, 3 for b24, one process per arm, 20
      timed steps + 2 warm, `final_byte`, bf16, `muon_state_dtype=bf16`,
      `ns_dtype=bf16`, `engram_ram=false`, `PYTORCH_CUDA_ALLOC_CONF=
      expandable_segments:True`; launches counted over ONE step, not averaged):
      b8 s512 eager **215.5 ms / 19 010 B/s / 7 236 launches**, full_graph
      **210.6 / 19 450 / 8** — the whole step (fwd+bwd+firewall+zero_grad+
      optimizer) really is one replay; b24 s1024 eager **819.4 / 29 992 /
      7 239**, full_graph **COUNTED-refused** (190 MiB free vs 1 275 needed;
      a FORCED capture OOMs at a 96 MiB allocation with 126 MiB free — the
      eager step's own peak is 14.8 of 15.6 GiB, so a whole-step pool cannot be
      mapped on this card). **The graph removes 7 228 of 7 236 launches and buys
      2.3% of a step**, so on this box CUDA-graph capture is real but is NOT a
      lever: the port's step is GPU-bound, not host-dispatch-bound.
      Gates `port/tests/test_full_graph.py` (27; `full_graph=false` sha256 of
      the 5-step `state_dict` + logits pinned from a clean `main` worktree,
      `b36ef27`; the bincount gate; the bool-mask capture gate + a replay of a
      DIFFERENT routing bit-exactly; the 13 + 4 refusals; non-CUDA LOUD before
      the first step; b24 s1024's optimizer stage captured + 20 replays;
      `final_byte` b8 s512 captured + 20 replays with no COUNTED fallback; the
      first replay vs the arm's own eager program at 1e-6). Bench
      `port/scripts/bench_full_graph.py` (`--modes eager,full_graph`,
      launch count for ONE step).

- [x] P-GRAPH-M400B. port (Python): `full_graph` on the LIVE m400b recipe
      (199M, b8 x s1024) after the 2026-11-05 kernel profile measured 60% HOST
      GAPS (320 ms wall, 200 ms GPU busy, 151 ms kernels, 4 692 launches/step).
      The old verdict ("GPU-bound, graph buys 2.3%") was `final_byte` b8 s512;
      this recipe is the one the card trains now. `configs/m400b.toml` is the
      checkpoint's model half, tracked so the gate is reproducible.
      **What changed for capture:** (1) `full_graph_refusal` now allows the
      dropless grouped GEMM (`moe_capacity=0` + `moe_kernel=triton`): its
      row-block grid takes the STATIC `_dropless_cap` and every program exits
      on-device, so `offs` never crosses to the host. `moe_capacity>0` +
      triton is still refused (`order[keep]` = `nonzero`); `batched` dropless
      (`counts.max().item()`), `bucketed` (`counts.tolist()`), `naive`
      (`unique().tolist()`), `padded` dropless and `gather=kernel` are refused
      with their Escape. The off-by-default `moe_route_fused` and
      `moe_permute="count"` are refused by name too.
      (2) LayerRoPE + `full_graph` is ALLOWED: every depth index the capture
      reaches is fixed (`rand_depth`/`segment_len` are refused first), and the
      recording holds one device-side slice per iteration; `opt_graph` stays
      refused. `tests/test_layerrope.py` was updated to the new contract.
      (3) **The bank-cache trap:** a `_banks` HIT at capture records no
      `torch.stack`, so every replay would read the capture step's stack while
      the optimizer moves the real parameters — the medbig2/b1night freeze,
      now structurally closed by `FullGraph(pre_capture=drop_caches)`
      (`invalidate_banks_all` + `invalidate_qat_cache_all` immediately before
      the recording).
      **The counter this task exists for:** a successful capture counts
      `full_graph_captured=1` and logs it; a fallback counts
      `full_graph_fallback=1` + the documented `opt_graph_fallback`. The gate
      asserts the positive counter, so an arm that refuses and stays eager can
      no longer look like a graph run (the class paid for twice).
      Gates `port/tests/test_full_graph_m400b.py` (18 CPU + 3 GPU-skipped):
      recipe legal; one refusal per ragged arm with `Escape:`; recompute LOUD;
      the `prop` controller is device-only under `item/tolist/__float__/
      __bool__` traps; capture/fallback counter deterministic on a faked CUDA
      surface; pre-capture ordering; the off path's pinned 5-step sha; the
      probe names `nonzero`; GPU: recipe capture + 20 replays + no fallback,
      first replay vs the arm's own eager twin at 1e-6, and the 20-step drift
      REPORTED (mHC/Muon chaos ~1e-1; not a 1e-6 bound).
      Probe `port/scripts/probe_capture.py --target step` (TorchDispatchMode
      over the real trainer, prints the last 12 ops + culprit) and
      `--ops` (the per-op table). Bench `port/scripts/bench_graph.py`:
      eager vs graph, min-of-5, ms/step + launches + B/s, `_lane.preflight`,
      `--parity N` (first replay 1e-6 gated, N-replay drift printed).
      **NOT RUN:** the card is owned by the multi-day m400b run; the probe,
      bench and GPU gates were written and their CPU halves are green. The
      checkpoint-gap command is in the bench docstring. No ms or parity number
      is claimed until that command runs.

- [x] P-LAYERROPE2. port (Python): upgrade LayerRoPE to the paper's reference
      form beside v1. `layerrope_form: "v1" | "paper"` (default `"v1"`, so no
      number on record moves). `"paper"` = the project page's snippet,
      transcribed verbatim as `layerrope.LayerRoPESite`: `gamma[d]` + 4 scalars
      (`a_mag`, `b_mag`, `a_rot`, `b_rot`), log-linear `r_l = a_mag + b_mag·log_l`,
      RoPE-spectrum per-PAIR angle `ang_j = θ_l·base^(-2j/d)`, buffers
      `freq`/`log_l`, table built ONCE per forward (input-independent).
      FOUR sites per block (`attn_read`, `attn_write`, `mlp_read`,
      `mlp_write`) = `4·(d+4)` params; v1 stays `d + 2·n_depths`; the per-layer
      baseline is `2·L·d`. READ gain = the fused `F.rms_norm` affine;
      WRITE gain = a multiply on each branch's OUTPUT where it enters `y`, the
      residual update (`loopblock.py` `_window`, before
      `y = attn + branch + ffn`), which is the `F(H_pre x_l)` mHC/AttnRes/ReZero
      deposit — no mixer internals touched. Our block has ONE shared pre-norm
      and a THIRD branch (Engram): `attn_read`/`attn_write` cover the attention
      arm AND the memory read (stated in the docstring, not invented as a fifth
      site). `layerrope_index: "iteration" | "position"` — `"position"` is the
      paper's looped-model rule (all iterations of one body read one gain);
      `layerrope_beta_init=-0.5`, `layerrope_base=100.0`, and
      `layerrope_init: "paper" | "identity"`. **The parent's identity claim does
      not hold in the snippet**: `b_mag=b_rot=0` gives `r=0, θ=1`, and
      `ang = θ·freq = freq ≠ 0` is still a rotation — so `"identity"` also
      short-circuits the spectrum, which is the only way `torch.equal(gain,
      gamma)` can be true; the deviation is named in the module docstring.
      Measured: identity forward is bit-identical (max|Δ| = 0.0, loss equal) to
      `layerrope=false` on the pinned tiny fixture; paper init is live
      (`b_mag += 0.1` moves the logits); both write gains move the output and
      `torch.equal` a run whose branch output is zeroed; per-pair angles
      recovered from the table match `θ·freq` at 1e-5 with 32/32 distinct pairs
      and an angle ratio of 86.6 at d=64 (`base^((d-2)/d)`); all five parameters
      of all four sites get finite non-zero gradients in a real 3-step run;
      `layerrope_forward` counts 4/iteration (device-side, drained at the log
      cadence). Refusals: bad form/index/init/base, `beta_init`/`base`/`index`/
      `init` with `form="v1"`, `amp_init`/`theta_init`/`per_site`/`learn_theta`
      with `form="paper"`, any subfield without `layerrope=true`. Gates in
      `tests/test_layerrope.py` (19 v1 gates unchanged + 10 paper gates), ten
      scratch reverts each turning its named gate red, sha256 restored
      (`layerrope.py` `0cf1637…`, `loopblock.py` `ff56109…`, `config.py`
      `3d742b4…`). Driver `scripts/ab_layerrope.py` REWRITTEN for the new A/B:
      `no_layerrope_5e-4` control, `v1_/paper_/paper_identity_/paper_pos_` at
      5e-4 and 1e-3, 3 seeds × 600 steps, b8 s512, medium, mix4/web.bin +
      web_eval.bin; pre-registered: paper replaces v1 iff
      `mean(v1) − mean(paper) > 2 × spread(control)` at the same LR,
      paper_pos vs paper_iter decided by the same rule, LR band width reported
      per form. Written, **NOT run** (no GPU work on this task): no timing, no
      BPB, no verdict exists.

- [x] P-LRBAND. port (Python): three OFF-by-default LR-band stabilizers and a
      per-group gradient-norm diagnostic, so the next divergence names WHICH
      optimizer group widened the band. `train.agc_clip` (float, 0.0 = off) is
      Brock et al. 2021 §2.2's per-parameter `λ·||p||/max(||g||,1e-3)` clamped
      `<=1`: two `torch._foreach_norm` launches (not N), device-only, applied in
      `opt_stage` BEFORE `clip_grads` and once per accumulated step. The exact
      variant and the paper's `λ²||W||²` threshold (with `||p||` standing in for
      the parameter norm) are stated in `optim.agc_clip`'s docstring, including
      the zero-init consequence of flooring the GRADIENT norm. `train.adam_eps`
      (1e-8) reaches `MixOptimizer(eps=)` -> `torch.optim.AdamW`; the docstring
      names the mechanism: the equilibrium logit scale is ~`log(1/eps)`, 18.4
      nats at 1e-8 against the fp32 softmax-collapse cliff at 16.6, 13.8 at
      1e-6. `ModelConfig.logit_softcap` (0.0 = off) is Gemma-2's
      `cap·tanh(z/cap)` on the per-iteration and final CE logits, so the eval
      objective is capped too; it composes with `ce_fixed` and is REFUSED with
      `use_byteflow` and `mtp_weight>0` (MTP reads the shared head uncapped).
      Diagnostic `g_norm_{muon,adamw,norms_biases}` once per `log_every`, on
      device, drained with the window's counters (`Counters.drain` now keeps a
      0-dim float as a float). Defaults pinned bit-identical: state sha256
      `526b6c9e…` + logits sha256 `af5d2fd8…` from a 5-step tiny run in a clean
      detached `main` worktree (`06109ae`) captured before the edit. Gates
      `tests/test_lrband.py` (12; eight scratch reverts each turned its named
      gate red, every file restored byte-for-byte with sha256). Driver
      `scripts/ab_lrband.py`: (a) `lr {1e-3,2e-3,3e-3} x warmup_frac
      {0.02,0.08}`, (b) `+z_loss 1e-3` / `+adam_eps 1e-6` / `+agc_clip 0.01` /
      `+softcap 15` at the surviving LR, 3 seeds x 600 steps, `medium`,
      mix4/web.bin + web_eval.bin, JSONL under `~/logs/ab_lrband/`,
      pre-registered verdict (a divergence voids the arm; a win must beat the
      control's 3-seed spread), `plan`/`profile`/`run`. Written, **NOT run**
      (GPU busy with the flagship): no timing, no BPB, no verdict exists.
- [x] P-MTP-SOFT. port (Python): Soft-MTP (arXiv:2610.38331), behind
      `ModelConfig.mtp_soft` beside `mtp_weight`; OFF is bit-identical (pins
      from clean `main` `21a97d2`, captured before the edit: tiny state sha256
      `e26c6a07…`/`894fe40f…`, logits `c7839283…`, losses 6.119609355926514 /
      8.997851371765137, MTP-proj + embed grad sha256). `MTPHead.forward_soft`
      adds `ntp_hidden[:, k:] += s_k·(x_k.detach() - x_k)` and returns the
      coupled hidden; `model.py` computes the NTP CE on it through the existing
      CE path and keeps the returned logits/aux on the original. `s_k` =
      `weight·N_ntp/(n_heads·N_k·ntp_loss_scale)` with this repo's
      `ntp_loss_scale = 1/(T+1)` (the reference's `kl_recovery_scale` has no
      depth divisor). fp64 decisive gate: last-head gradient == explicit
      KL(p_ntp‖p_mtp) at `7.5e-16…1.1e-15` relative for K=1,2,3, full vocab,
      dense head; **FINDING**: this module has no per-head final norm, so
      `x_j` also feeds module `j+1` and the later heads' coupling routes back
      through the sequential projection — the identity is exact only at the
      last head (and K=1); earlier heads carry that cross term, quantified and
      named in the test. The shared-`lm_head` premise HOLDS (dense/TSCT/tied
      are all linear; the tiny default is TSCT). Refused with
      `mtp_weight==0` and `logit_softcap>0` (the verifier would be capped while
      the MTP head reads the head uncapped). Device counter `mtp_soft`, no
      host sync in the module (`.item/.tolist/.__bool__` gate). Gates
      `tests/test_mtp_soft.py` (12); driver `scripts/ab_mtp_soft.py`
      (`ntp`/`hard_mtp`/`soft_mtp`, 3 seeds x 600, b8 s512, mix_v7, probe for
      per-head CE/acceptance; keep rule pre-registered) written and **NOT run**
      (card busy). The FINDING is fixed by P-MTP-SEAM below.
- [x] P-MTP-SEAM. port (Python): the reference's per-head final RMSNorm, and
      `mtp_soft` now REQUIRES it. `ModelConfig.mtp_head_norm` (beside
      `mtp_weight`, default false) adds `MTPHead.head_norms[j]`, a per-head
      RMSNorm WITH a gain (matching the reference's `blocks.RMSNorm`; `K·d`
      parameters — tiny d=64/K=2: +128; medium d=1024/K=4: +4 096). The head's
      CE and its coupling read ONLY the normalized `z_j`; the next module still
      consumes the RAW `x_j`, so `z_j` is an independent tensor. `mtp_soft=true`
      with `mtp_head_norm=false` is LOUD in `__post_init__`, `validate` and the
      model constructor, with an Escape naming both flags. fp64 every-head
      gate: relative `4.985e-16…8.933e-16` for K=1,2,3 (asserted <=1e-10);
      WITHOUT the seam the same gate is 8.87e-1 / 1.08e+0 / 1.12e-15 (last head
      only), pinned in `test_without_the_seam_the_nonfinal_head_carries_the_chain_term`.
      OFF: `mtp_weight=0` builds NO module even with `mtp_head_norm=true`
      (state/logits/loss/embed-grad pins from `21a97d2` unchanged, gate a2);
      `mtp_head_norm=false, mtp_soft=false` reproduces the SAME pins, re-measured
      on the previous commit `10c3831`. Seam delta: `param_count(seam) -
      param_count(base) = K·d`, every base parameter bit-identical, NTP logits
      bit-identical, only `mtp.head_norms.*.weight` new (gate a3). Reverts
      18/18 turned their named gate red. Driver extended to 4 arms
      (`ntp` / `hard_mtp` / `hard_mtp_seam` / `soft_mtp_seam`), 3 seeds x 600,
      b8 s512, mix_v7 + eval_mix; the seam-alone verdict is printed and the
      soft keep rule is re-stated against hard+seam; written, **NOT run**
      (card busy). Suite diff vs clean `21a97d2`: the same single pre-existing
      `test_sft_v2::test_the_manifest_accounts_for_every_row` failure; ruff
      clean.
- [x] P-CSA-FUSE. port (Python): the CPU launch instrument + the fused CSA
      fine-selection arm, both CPU-proven, **NO GPU PROFILE** (the card is
      owned by a live 8 KB-context run; every command here ran with
      `CUDA_VISIBLE_DEVICES=""`). `scripts/launch_profile.py` keeps the old
      `--gpu-profile` (the two-arm card trace) and gains the CPU twin:
      `DispatchCounter(TorchDispatchMode)` counts ONE fwd+bwd+optimizer step,
      groups by `(aten op, module)` (module = the executing hook scope; the
      output/argument `grad_fn` chain is the fallback outside a module),
      excludes metadata ops, prints the top 25 and stages, `--json PATH` writes
      the full table, `--diff A.json B.json` prints the per-group delta sorted
      by `|Δ|`; `--device cuda` is REFUSED with the reason when `_lane.preflight`
      sees another process on the card. `ModelConfig.csa_fused` (OFF; refused
      without `use_csa` and with `csa_fine=false`): `_select_fused` uses
      `topk`'s own values for both the score and the mask (`values > finfo.min`),
      masks before the softmax (deletes `sum`/`clamp_min`/`div`) and contracts
      with one batched matmul; `CSAState.append_many` hoists `forward_state`'s
      `for i in range(n_full)` ring append into one cast+copy per tensor.
      CPU arithmetic, tiny b2×s64 chunk4/topk4: step launches 1764 → 1738
      (−1.5 %), the `loop.csa` group 124 → 98 (−20.9 %); module fwd+bwd
      56 → 43 (−13) and gather 3 → 1; `forward_state` at t=256 chunk4
      290 → 35 launches with ring copies 260 → 8, and the fused count is the
      same at t=64 and t=256 (the per-chunk Python work is gone). Parity:
      fp32 output `7.45e-8`, gradients `7.63e-6` absolute on a `31.4` scale
      (`2.4e-7` relative); bf16 `1.95e-3` = **one bf16 ulp** — a 1e-6 bf16 gate
      is below `2^-8` and would be a gate that cannot fail, so the test pins
      one ulp and prints the number. Ring wrap (state carried across a chunk
      boundary): outputs `7.45e-8` / `8.94e-8`, ring keys/values bit-equal,
      `n=5`; the hierarchical tail agrees to `8.94e-8`. Gates
      `tests/test_csa_fused.py` (8 functions / 9 cases), each red on a scratch
      revert: delegating `_select_fused` to `_select` reddens 1/6, restoring the
      append loop reddens 6b, defaulting the flag ON reddens 1/4, a mask that is
      always true reddens 1/2/3. FOLLOW-UP, not this arm:
      `_hier_fine`'s `sel = zeros(b, qn, keep.shape[-1]); sel.scatter_(2, keep,
      True)` indexes a `topk`-wide buffer with parent-entry indices, so any
      `csa_hierarchical` run with `csa_topk < entries` raises out of bounds
      (`csa.py:741`); the hierarchical parity gate uses `topk=64` to stay on the
      specified shape. Instrument gates
      `tests/test_launch_dispatch.py` (6). Suite: 5 reds, all pre-existing or
      worktree-local — `test_lr_tail` (`seq_len` joined `PROGRESS_KEYS` in
      `cd65540`, its closed-set test was not updated) and `test_dashboard`
      (live `~/logs` state) also red on clean `1549e14`;
      `test_decode_recurrent::test_per_byte_cost_does_not_grow_with_context` is
      the recorded wall-clock gate that reddens under the live training run and
      skipped on rerun; `test_resume` (`configs/m400.toml`) and `test_sft_v2`
      (`sft_pairs/`) are untracked files a clean worktree does not carry.
      ruff clean. **The launch-count claim on the card is still unmeasured**:
      one aten op is one kernel on the eager GPU path, but the 8 KB-shape
      profile is the owner's to run. The CPU counter does NOT reproduce the
      premise that the CSA group is the largest: at tiny it is 124 of 1764
      (7.0 %) while the optimizer stage alone is 476, so the fused step saves
      1.5 % of launches there — the big win (290 → 35) is `forward_state`,
      which is the decode/prefill path, not the one-shot training step.

- [x] P-ROUTING-HEALTH. port (Python): make the dead-expert collapse impossible to
      miss. `scripts/dashboard.py` gains a routing panel (live/E bar, top-10 and
      max share, RED below 50 % live) fed by the live counters or the last
      `scripts/route_hist.py --json PATH`; `scripts/plan_restart.py` prints the
      fresh-run command, the 20 B/param and 22.2 GB budgets, the 22 000 B/s ETA
      and the "E > 4x live" warning. Gates `tests/test_dashboard_routing.py`,
      `tests/test_plan_restart.py`.

- [x] P-MOE-SHARED-COMPUTE. port (Python): `moe_shared_compute` (OFF by
      default) stops materializing the plane's shared basis E-fold. The dispatch
      reads the bank's OWN `gate_u`/`up_u` as one `[d, 2·rank]` matrix and does
      `lat = x[n,d] @ W` ONCE over the tokens, then gathers `[n·k, 2·rank]` and
      keeps today's `silu(g)·u`, `@ v`, weighted scatter (`batched` and `triton`;
      the grouped GEMM takes two constant `[E, 2r, r]` selectors). Refused with
      `Escape:` on `share_u_mode='off'`, `rank<1`, `dict`, a heterogeneous
      per-expert rank, `moe_gather='kernel'` and a kernel outside
      `{batched,triton}`. Counters `moe_shared_compute_calls` +
      device `moe_projection_rows`. Gates `tests/test_moe_shared_compute.py`
      (6 functions, 6 scratch reverts red): flag off is the pre-edit pin
      (64 881 params, sd/logits sha), ON vs OFF fp32 logits rel `1.82e-7` and
      worst parameter gradient rel `6.13e-7` at depth 1 (the test says why depth
      2 is not used), matmul dispatches `3 -> 2`, no host reads, 45-tensor
      checkpoint round-trip. CPU suite: only the pre-existing
      `test_layerrope` counter (fixed by the uncommitted main-worktree edit) and
      the untracked-`sft_pairs` failure. The GPU bench
      (`scripts/bench_share_u_compute.py`) is written but UNRUN — the card is on
      the multi-day job. Arithmetic at m400/b8-s1024 (n=8192, k=2, E=512,
      r=64, d=1536, f=4096, cap=32): projection 3.221 -> 1.611 GMAC, write
      4.295 GMAC, total 7.516 -> 5.905 GMAC (21 % fewer, −2 `bmm` launches).

- [x] P-OPT-REGION. port (Python): the 2026-11-05 profile's largest region
      (optimizer 35.4 ms / 23.5 %). m400/medium now resolve
      `muon_state_dtype=bf16` + `muon_chunk_elems=1<<26` through
      `config.PRESET_TRAIN_DEFAULTS`; `ns_steps` and `muon_chunk_elems` are
      `PROGRESS_KEYS`, with `MixOptimizer.set_ns_steps` re-applied after torch's
      loader (otherwise the key is inert on resume). The chunk cap is
      bit-identical (the split is over whole matrices) and takes the m400
      optimizer stage 890 -> 731 launches (−17.9 %; cap alone −203, bf16 state
      +60) on `bench_muon_launch.LaunchCounter`; the 600×[1024,64] fixture
      127 -> 73. The bf16 state's 100-step CPU gate is a measured TOLERANCE, not
      bit equality: max rel loss 0.036 % vs fp32 (fresh batch every step),
      inside the fp32 seed spread 3.45 %, worst parameter delta 9.68 bf16 ulp
      at scale (pins in `tests/test_opt_region.py`). The strict 1e-5 /
      per-element-ulp claims are
      impossible without stochastic rounding and are reported, not asserted.
      `ns_steps` default is NOT flipped; `scripts/ab_opt_region.py ab` prices
      5 vs 3 on 2 000 steps and prints both arms. Overlap is reported, not
      implemented: one stream + one thread, and a per-group split is an autograd
      seam, not a stream+event change. Reverts red: preset application, the
      `ns_steps` progress key, the resume re-apply, the cap wiring, and the
      bf16 store. Report `docs/reviews/2026-11-05-opt-region.md`.


- [x] P-DISPATCH-CENSUS. port (Python): the dispatch census, the fused route head
      and the counting permute — both arms OFF. **The finding first:** the profile's
      31.6 ms `moe_dispatch` is NOT the router/top-k. Per forwarded pass the census
      prices route+permute+gather+combine at 1 815 MB / 53 launches, and the two new
      arms remove 119 MB and 19 launches (−6.6 %, −36 %); at ~29 µs per kernel launch
      that is ~0.7 ms/pass, not the −20…−28 ms the profile's region share implied.
      The region is (a) the expert adjoint's GEMM/pass kernels the seam attribution
      swallowed — `_grouped_gemm_pair` 8.5 + `_wgrad` 1.6 + `_grouped_gemm` 0.8 =
      10.8 ms/step — and (b) the bank rebuild's copies, 577 of the region's 888.5
      kernels/step (4.6 ms), and (c) the combine's fp32 scatter, 1 476 MB/pass
      (81 % of the routing cluster's bytes: `zeros` + `y.to(f32)*w` + `index_add_`).
      The router GEMM is 35 MB / 2 launches — not a target. Follow-ups, in order:
      fuse the combine's three forward ops into its existing kernel; stop rebuilding
      the bank every step (its `.to()` per expert is the launch hog); then the
      gather arm's missing adjoint. `scripts/dispatch_census.py` (`--ab`) and
      `tests/test_moe_route_fused.py` (21 gates, 4 scratch reverts red, tie fixture
      pinned both ways, OFF pinned to main's digests).
      `moe_route_fused` = one Triton pass for softmax + LFB bias + top-k + gate
      gather; `moe_permute="count"` = stable counting sort, element-wise identical
      to the stable argsort. GPU parity + `bench_gather` are written and NOT run —
      the card is on the multi-day job. `moe_gather="kernel"` re-checked: the
      forward saves exactly 4·R·d, the recompute adds 654 MB, so the PASS is
      +554 MB (unchanged conclusion); one defect fixed on the dropless route (it
      passed `keep`, not `keep_rows`, so it built an all-True bool-mask index).

- [x] P-DISPATCH-COSTS. port (Python): the two costs the census named, both OFF.
      **J1 bank cache (`moe_bank_epoch`).** The premise "it rebuilds every
      iteration" is FALSE and the measurement says so: `invalidate_banks_all`
      already runs once per optimizer step and `_banks` reuses the entry within
      the step (1.00 builds/step; the profile's 577 copy kernels/step ARE one
      build of 512 parameter tensors). The 519 launches / 1 007 MB per step do
      NOT move — a `[E,·]` stack of 512 separate parameters is 512 casts + a
      stack, and only a parameter-layout change can cut it (follow-up). What
      DOES move: the `_version` key walks all 1 536 parameters on EVERY
      `_banks` call — 2 092 us/call at m400, **6.28 ms/step of HOST time** at
      max_iter 3 on a step that is 87 % host-gap — replaced by one `bank_epoch`
      int (entry epoch, dropped before the rebuild so the peak is not two banks)
      bumped by `invalidate_banks` (the trainer's per-step call) and by any
      `load_state_dict`. A frozen epoch serves a stale bank and the parity gate
      goes red; an entry with no epoch or from the future is LOUD.
      `moe_bank_builds` is the counter (host int, no device read).
      **J2 combine (`moe_combine`/`moe_combine_dtype`).** `torch` = 1 476 526 080
      B / 4 launches per pass (81 % of the routing cluster). `fused` = ONE
      token-major kernel, fp32 registers: **671 350 784 B / 2** (-805 MB, -2)
      and **bit-identical** on logits and every parameter gradient (bf16 `y` ×
      fp32 `w`, k=2 fp32 adds commute, so `index_add_`'s row order cannot
      matter). `fused/bf16` stores the compute dtype: **537 133 056 B / 2**
      (-939 MB) plus the caller's `weighted.to(dt)` row disappears (-201 MB)
      = -1.14 GB/pass (-24 % of the whole MoE pass), but it rounds the routed
      activation one step earlier and MEASURES 4.1..6.2e-3 relative on logits
      and 3.2..4.9e-3 on gradients over three seeds — **the 1e-3 logits target
      is NOT met**, so the default stays fp32 and bf16 is the documented
      second arm, not the pick. Gates `tests/test_moe_bank_cache.py` (11) and
      `tests/test_moe_combine_fused.py` (14 + 1 card-only skip); 4 scratch
      mutations turn 6 gates red (no epoch bump -> 2, serve-stale -> 2,
      dropped-row mask -> 1, forgotten combine switches -> 1). GPU parity
      written, unrun. Picked together: 53 -> 32 launches and 1 815 -> 891 MB per
      forwarded pass (-50.9 % of the routing cluster), host 2 089 -> 0.5 us/pass
      (-6.27 ms/step at max_iter 3); the bf16 store is priced and NOT picked.

- [x] P-MHC-WRITE. port (Python): the `mhc_write` region's 1 395 `add`
      launches were **not** the fused kernel. The profile ran
      `fused_residual=false` (`port/configs/m400b.toml:58`), so `MHCMix` ran
      eagerly and `mhc.sinkhorn_knopp`'s 20-iteration Python loop issued
      `torch.logsumexp(m)` + `m = m - lse` per normalisation: the reduction's
      `add` and the caller's `sub` (CUDA lowers `sub` to `CUDAFunctor_add`,
      alpha=-1) are ~120 add-family launches per call, x3 iterations/step.
      **The fused write's Sinkhorn is already in registers**
      (`fused_residual._write_kernel`, `for it in tl.static_range(ITERS)`, one
      launch/iteration), so an iteration-count arm would buy nothing there.
      New arm `mhc_sinkhorn="log_softmax"` (default `"logsumexp"` = the shipped
      bits, pinned sha): `log_softmax(dim)` IS `m - logsumexp(m)` as one ATen
      op, so the relaunch family leaves the graph. b8 x s1024 d1536, one
      iteration: add family 126 -> 6 (21x), region ATen ops 827 -> 147 (5.6x),
      Sinkhorn alone 761 -> 81 (9.4x; the add family 120 -> 0); values and
      grads within 2.4e-7 / 1.8e-7 (NOT bit-identical, hence the flag).
      Gates `tests/test_mhc_write_launches.py` (13: pins, 21x family gate,
      scratch revert red, parity, refusals with Escape, no host sync, the
      kernel's `static_range` pinned). `scripts/bench_mhc_write.py` prints both
      arms (CPU us are not a GPU estimate). The profile's 23.2 ms is fixed by
      the fusion flag, not this arm: `--set fused_residual=true
      --set fused_epilogue=true`; GPU numbers unrun (card owned).
- [x] P-RESIDUAL-BF16. port (Python): `residual_dtype="fp32"|"bf16"` — the
      loop's residual stream stored at half the bytes. `scripts/precision_census.py`
      (static, torch-free) is the change list: it prices every iteration tensor
      (shape, dtype, one-pass bytes, `required` + why, the projected dtype) and
      its m400b b8×s1024 table goes **1.354 → 0.976 GB (−27.9 %, 15 bf16
      passes)**, with the `residual` group itself **0.705 → 0.352 GB (−50.0 %)**
      — the reductions/state groups it leaves alone are the rest; the tracked
      `configs/m400.toml` goes **0.921 → 0.644 GB (−30.1 %)**. Moved: `h`/the mHC lanes, `ctx`, the normed reads, the three
      branch writes, `y`, `step_out`, the Engram `mem`/`dense` outputs. Kept
      fp32, each named by the census: the lane RMSNorm (`xn`), the mHC
      hyper-network/Sinkhorn/lane mix, AttnRes' key norm + softmax, the KDA
      projection input and `g` (the fused tail is fp32-input), the gated-delta
      state `S`, the CSA scores/ring `sum`, the Engram read + order reduction,
      the readout accumulator `out_acc`, the final norm/head/CE, the optimizer
      masters. The fused residual/epilogue kernels and `fused_kda_tail`'s
      combine are fp32-only and REFUSED with `Escape:` naming each (config
      `validate` AND the `LoopBlock` runtime seam a hand-built model reaches).
      `tests/test_residual_bf16.py` (13): OFF pins from clean `main f09eea5`
      (logits+loss+every param grad sha256, before the edit), ON stores bf16 at
      the seams while `xn` stays fp32, the 200-step CPU drift is a MEASURED pin
      (max per-step rel **7.32e-4**, final Δ **+1.04e-4**, last-20 mean
      **6.33e-4**; fp32 seed spread is 7.7e-2, so the arm is inside its own
      control's noise), the lane-norm-without-`.float()` mutation reddens the
      dtype-seam gate, a required census row moved to bf16 reddens the pinned
      total, the census bytes are recomputed `prod(shape)·itemsize` and pinned,
      refusals carry `Escape:` + the kernel name, and the ON forward adds no
      `item`/`__bool__`/`__float__` sync. Full CPU suite: the 18 pre-existing
      failures on clean `main` plus the 4 worktree-local untracked-artifact ones
      (`checkpoints/` ×3, `sft_pairs/`), nothing new; ruff clean. **No GPU
      timing exists** — the card is on the multi-day run; the A/B is
      `scripts/kernel_profile.py` ×2 + `--diff`, with the arm at
      `--set residual_dtype=bf16` (below).

- [x] P-TSCT-PACKED. port (Python): the 519-launch bank rebuild as ONE buffer.
      `ModelConfig.tsct_packed` (OFF): `SparseMoE` owns `packed_v [E, r, f]`
      (+ `packed_gate_u`/`packed_up_u [E, d, r]` when `share_u_mode="off"`) and
      `packed_s [E, 2, r]`; each expert holds plain-tensor VIEWS
      (`DualLatentGLU.bind_packed`, rebound after every load) and `_banks`
      returns `buffer.to(dt)` instead of E casts + `stack`. MEASURED at m400b
      (E=512, r=64, f=4096, bf16): **521 -> 5 launches** (plane; 1 543 -> 5
      with `share_u` off) and **1 681 -> 1 010 MB** of operand traffic per
      build; the transient drops by the second E-fold copy (~268 MB), steady
      state is unchanged. The scales are views (`[E, 2, r]`), so the `[E, 2f,
      r]` shape in the brief is read as the two input bases `[E, d, r]` + a
      `[E, 2, r]` scale buffer: one concatenated `[E, 2d, r]` would make Muon
      orthogonalize `[2d, r]` (the Gram mixes gate/up rows and the
      `max(1,m/n)^0.5` scale is wrong), so the pair stays two buffers — two
      casts, still 3 launches for v+scales at E=512.
      OPTIMIZER: `is_muon_param` sends a 3-D parameter to Muon only when the
      model marks it `_muon_batched`; `Muon._build_groups` gives it its own
      `[E, d0, d1]` momentum buffer, the momentum update is one `add_`,
      `batched_step_from_model` chunks the leading dim and writes
      `colrow_normalize_ + update.mul_(-lr*scale) + add_` per slice (bit-equal
      to E 2-D params on the gate fixture and in the whole-model step); the
      per-parameter `Muon.step` and the `sf`/`normuon` tails REFUSE LOUDLY with
      `Escape:`. CHECKPOINT: both ways, `strict=True` — old `experts.i.{v,
      gate_s,up_s,gate_u,up_u}` gather into the packed buffers and a packed
      `state_dict` expands back (`SparseMoE.state_dict(packed_form=)` writes
      either form); the optimizer's per-parameter state is NOT convertible
      across the flag (different parameter count), stated in the plan.
      Gates `port/tests/test_tsct_packed.py` (14): OFF pins captured on clean
      `907b3b5` before the edit (55 025 params, grad/state sha256, loss
      5.939486); packed vs unpacked `torch.equal` on the bank (fp32+bf16, both
      share modes), logits/loss and EVERY parameter gradient; the two-way
      checkpoint round-trip incl. a real `torch.save` file; **521 -> 5
      launches (104x)** at E=512 via the repo's `DispatchCounter`; Muon
      routing/step/refusals; the starvation rule rejecting a tie + the
      `--dry-run` table. Five scratch reverts each redden their named gate.
      Driver `port/scripts/ab_tsct_starvation.py` (A E=512 b24, B E=128 b24,
      C E=512 b8, D dense b24, E dense b8; 1500 steps, 2 seeds, eval every 500;
      pre-registered rule printed in the file and the output; `--dry-run`
      prints resolved configs, tokens/expert and the compute table; the b24 x
      s1024 peak is 14.8 of 15.6 GiB, so the high tier RESOLVES TO b24 x s512
      = 48 tokens/expert and says so) written, **NOT run** (this task trains
      nothing; the card is busy). **No BPB, no verdict exists.**
      Concurrency note: the shared tree was being edited by the TSCT-2 agent
      while this task ran; the work is isolated on `wt/tsct-packed`
      (`907b3b5`) and touches only config.py / experts.py / optim.py + the new
      test and driver.
      MERGED onto `main 1ab5020` (TSCT-2, `9b5e6dd`): the shared-plane conflict
      is a precedence chain (TSCT-2's basis fold first, the packed view only
      without a selector), `_MomGroup`/`__slots__` carry TSCT-2's
      `tsct_ids`/`tsct_row` AND `packed`/`rows`, and `packed_v` rows take the
      per-expert LR. `tests/test_tsct_merge.py` (3) gates the combined flags:
      packed + `tsct_expert_basis=64` bank is `torch.equal` on 5/5 (max|Δ|=0)
      and a packed + `tsct_expert_lr` step equals the unpacked rows exactly.
- [x] P-TSCT-2. port (Python): the three OFF-by-default mechanisms from
      `docs/architecture/2026-11-05-tsct-2.md`. **The doc's `[d, 192]` was
      wrong**: at `ecc979e` the routed plane is `[d, rank]` (64 in every real
      preset); 192 is `shared_rank`, the always-on shared expert's width. So
      `tsct_expert_basis=rank` WIDENS the routed plane to `shared_rank` and
      each expert owns `w [rank, shared_rank]`; the bank folds `gate_u @ w.T`
      into the effective `[d, rank]` basis at the rebuild, so the per-token
      operands/MAC are the control's and the A/B is at equal active MAC.
      Param delta exactly `E·rank·r_shared + 2·d·(r_shared−rank)` =
      **+6 684 672** at m400b (6 291 456 selectors + 393 216 wider plane), bank
      134 479 872 → 141 164 544; fold 19 327 352 832 MAC once per optimizer
      step. `tsct_expert_lr` is a pure device-side function of `lfb_counts`
      (`1 + c·(mean/max(L_j,1) − 1)`, clip [0.2,5]) applied as ONE `[N]` row
      scale on the Muon stack's `v` rows, no per-parameter loop, no host read;
      refused with `opt='adamw'`, SF/NorMuon, or no live LFB counts.
      `tsct_expert_revive`/`tsct_revive_noise` copy the hottest expert onto the
      coldest (`v`/`gate_s`/`up_s`/`w`, `σ=noise·std(v_hot)`, generator seeded
      `seed+step`) and reset both LFB entries; the two indices are the ONE
      declared host read at that cadence. Init `w=[I|0]` gives
      `max|Δlogits| = 0.0` vs the planeless path (gate b); a non-trivial `w`'s
      banked pass matches the literal formula at 3.79e-7 logits and 8.45e-7 on
      the worst TSCT-2 grad (all ≤1e-6; the engram 2.03e-6 is the pre-existing
      bank-vs-loop reduction, printed). Gates `tests/test_tsct2.py` (17), six
      scratch reverts each red, every file restored byte-for-byte (sha256).
      Driver `scripts/ab_tsct2.py`: control/+basis/+lr/+revive/all, 3 seeds ×
      2 000 steps, m400b, b8×s1024, 65 536-B window, equal active MAC
      (28 442 624/B in all five), rule printed: WIN needs mean <
      control_mean − max(2·spread, 0.01) AND live/E ≥ 70 % in every seed; a tie
      goes to fewer parameters. `plan` exercised (no card); `run` **NOT RUN** —
      the card is on the multi-day m400b job, so no BPB/verdict exists.
      Suite diff vs `907b3b5`: only the four pre-existing medium-parameter-table
      failures (`test_export`, `test_fullattn::gate_7`, `test_loop_blocks`,
      `test_prelude`), reproduced on clean `907b3b5`; ruff clean.

- [x] P-BLOCKFORM. port (Python): `block_form` — the measured control as a
      first-class block form. `ModelConfig.block_form: "kda" | "transformer"`
      plus `n_layers` (`0` = `loop_blocks × max_iter`); `DormouseModel`
      dispatches in ONE class: the same embedding, final norm, byte head and CE,
      then either today's `LoopBlock` or `n_layers` unique pre-LN blocks
      (`model/transformer.py`: causal SDPA + RoPE, GELU 4×, LayerNorm, GPT-2
      residual scaling — `scripts/control_transformer.py`'s `Block`).
      `"transformer"` LOUD-refuses `max_iter != 1`, use_kda/use_csa/use_mhc/
      use_attnres/use_engram/layerrope, every loop-only loss weight, prelude/
      coda, byteflow, dense_qat and `segment_len`, each with its `Escape:`.
      `configs/mirror_control.toml` (d=304, 12 layers, 8×38, ffn=1216, max_iter=1)
      prints **13 478 448** params via `info` against the control script's
      13 478 752 (−304 = shared final RMSNorm vs its affine LayerNorm; 0.0023 %).
      Gates `tests/test_block_form.py` (20; each of a-f + resume identity proven
      red by a scratch revert, every file restored sha256): kda bit-identical to
      clean `main ff34bf7` (state/logits/loss/gradient pins), the refusals, the
      count, strict checkpoint round-trip + bit-exact next-step loss on resume,
      20-step CPU runs of both forms with `stepaudit` showing no host read on a
      non-report step, and `load_run(--set block_form=...)` reaching the model
      (logits differ from the kda form at the same seed/inputs). `block_form` is
      resume identity (a switch is refused, naming `model.block_form`). Full CPU
      suite green after copying the untracked `port/checkpoints/*.config.toml` +
      `port/sft_pairs/` artifacts (otherwise three `test_config` reds and
      `test_sft_v2::test_the_manifest...`, identical on a clean worktree); ruff
      clean. **NOT wired:** recurrent decode/serve (KDA state), and the
      graph/QAT arms are unmeasured on this form.

- [x] P-MOD. port (Python): Mixture-of-Depths over `block_form="transformer"`
      (arXiv:2404.02258v1 §3.4). `ModelConfig.mod_capacity` (1.0 = off, no
      router built, bit-identical), `mod_top_k` (0 = `round(capacity·T)`),
      `mod_aux_weight` (0 = off; the paper's §3.5 BCE of the router logits
      against the top-k indicator, printed as `mod_aux=` on the log line).
      Per block: one `torch.topk` on `r_i = w^T x_i`, a scatter/`cumsum` that
      compacts the selection into POSITION order (so `is_causal` and the
      original RoPE positions are exactly the full sequence's), the plain block
      on the `C` gathered tokens, `r_i·f_i + x_i` scattered back. No
      `.item()`/`nonzero`/boolean-mask indexing: one `topk` and fixed-shape
      gathers, CUDA-graph capturable (`tests/test_mod_capture.py`).
      `scripts/mod_ledger.py` prints the formula and reproduces the measured
      flagship bill (137.42 MFLOP/byte at c=1.0; 0.63x at c=0.5 under the
      conservative "attention core is not credited" split, 0.50x if it is);
      `scripts/ab_mod.py` is the 3-seed equal-bytes/equal-steps A/B (control
      vs c=0.5, aux 0.01) with the pre-registered KEEP rule (+0.02 BPB AND
      <=0.85x step time), and it names the equal-steps/equal-FLOPs confound in
      its header. Gates `tests/test_mod.py` (16) + `tests/test_mod_capture.py`
      (2): capacity 1.0 has state/logits/loss/gradient pins from clean `main`
      `7e921b3`; the ledger algebra; the host-read spy on forward AND
      backward; the exact count for 20 random batches; router gradient on
      every step and no router at all when off; a 20-step CPU run at c=0.5.
      **NOT run:** no GPU and no driver run; the paper's 0.60x is a 60M-3B
      result, not a prediction for 13.5M.

- [x] P-BYTE-PATCH. port (Python): `byte_patch` — `block_form="transformer"`'s
      `n_layers` unique blocks over the conv codec's PATCH positions.
      `ModelConfig.byte_patch` (default false; the off path is bit-identical to
      clean `550bc43` on both the transformer and KDA+conv pins, sha256).
      `configs/arch_dense_byteflow8.toml` = flagship3_dense's 12 blocks
      (94 608 384) + final norm (832) + codec and the two pairings (2 975 680)
      = **97 584 896** params, asserted in the test and printed by `cli info`.
      The codec is both ends, so `embed`/`lm_head` are NOT built (the 425 984
      dead params the control has are absent). K MUST equal
      `max_seq_len // byteflow_patch_bytes` — at K<P the codec's global top-k is
      future-dependent and the strict prefix gate fails — and a `seq_len` that
      does not tile is refused at `load_run`, not at the first step. The loss
      is `_byte_ce` over BYTES, so BPB stays comparable with every other arm.
      Gates `tests/test_byte_patch.py` (10): eight scratch mutations each red
      (RNG draw off path, reversed patchify shift, pre-fix `prev = x`, one
      block, one shared body, halved positions, dead embed/head, per-patch CE),
      every file restored sha256; the four pins were independently recomputed
      from `git archive 550bc43`. CPU FlopCounterMode: 73.68 vs 568.69
      MFLOP/byte (7.72x) at seq 64/128/1024. Full CPU suite: one failure, the
      known `test_train::test_torch_compile_flag_wraps_the_model_after_resume`
      (inductor C++ compile; `/tmp` full); the other four known reds passed and
      there is no new failure. ruff clean. **NOT run:** the GPU A/B (the
      review's §6 is a recipe only); the wall-time and VRAM claims are
      UNMEASURED here.

- [x] P-CSA-SEG. port (Python): the CSA boundary contract, so `use_csa=true`
      with `segment_len>0` runs. At a cut the rings keep their entries, the
      partial chunk is CARRIED in `sum`/`count` and completed by the next
      segment as `(sum + new)/chunk` (never dropped, never padded), and the
      ring's `n·chunk + count` IS the byte position, so `pos` continues and
      a segment need not be a multiple of `csa_chunk`. `model/csa.py`
      (`_carried_state`, fp32 training ring vs bf16 decode ring),
      `model/loopblock.py` (the refusal is gone), `config.py` (replaced with
      `segment_len >= csa_chunk` + a loud `csa_hierarchical` refusal),
      `train/loop.py`/`model/model.py` comments, `stepaudit.py` counter. Parity
      `tests/test_segment_csa.py` (10 functions / 13 cases): 1.3-1.9e-6 max
      Δlogits at seg 64/128/100 with and without coarse 256 (gate 1e-5), state
      2.4e-7 (gate 1e-6), gradient worst rel 8.4e-7; `segment_len=0` pinned to
      clean `d1b2c2e` (`2bc9cf69…`/`4eb661ad…`, threads=1). Six named scratch
      mutations each red (drop/pad partial, restart `n`, rotation offset 0,
      ignore coarse ring, in-place carry → only the gradient gate sees it).
      Report `docs/reviews/2026-11-08-segment-csa.md` (arithmetic, memory at
      1M, exact GPU command). Full CPU suite: 43 failed vs main's 39, the 4
      deltas are untracked data (`checkpoints/`, `sft_pairs/` sounds) missing
      from a fresh worktree; no new code failure. ruff clean. **NOT run:** the
      1M GPU command; all VRAM arithmetic is derived, not measured.
      Follow-ups: the pre-existing hierarchical prefill gap (1.7e-2 vs
      `forward`, hence the refusal), and chunked prefill now leaves an fp32
      ring (2× the decode table's bytes).

## Critical Files

- `crates/dormouse-core/src/loop_block.rs` (617, in flight)
- `crates/dormouse-core/src/param.rs` (165)
- `crates/dormouse-core/src/config/schema.rs` (216, in flight)
- `crates/dormouse-train/src/optim.rs` (478)
- `crates/dormouse-train/src/offload.rs` (402, in flight)
- `tools/migrate-dormouse-fused.sh`
- `docs/adr/0017-dormouse-fused.md` (the table needs 3 corrections)
- `docs/PLAN-minimal-core.md` (the deliverable)

## Risks

- High: C1's rename vs dirty paths → keep the script's dirty-tree guard.
- High: field renames break the optimizer routing table → C11 last, and move the
  table into core so it cannot drift again.
- High: the Engram 500k-row + floor fix is ~24 h old, has never trained a step,
  and is 86% of the model's params → keep ON, label it unmeasured, make it
  AB-PROTOCOL arm 0.5.
- Medium: 1 700 LOC of deletions in the same week as a 2 000-line rename → 11
  commits apart, deletion first, rename last and alone.
- Medium: 12 797 LOC (41%) of the library is unwired → inventory, then delete.
