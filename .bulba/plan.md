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
