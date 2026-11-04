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
