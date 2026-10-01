# Design review: dormouse-core + dormouse-train (2026-09-27)

Snapshot: HEAD `c2b7ca5`, `2026-09-27T16:43:54Z`. Full report: `/tmp/opencode/architecture-review-model-20260927.html`.
Read-only pass. Mid-edit files (another agent active): loop_block.rs, model.rs, mor.rs, attention.rs, lib.rs, optim.rs, cfg.rs, stress.rs.
**NOTHING IS RUNTIME-VERIFIED**: `cargo check -p dormouse-core -p dormouse-train --lib` fails in `crates/dormouse-data/src/lib.rs:144,:214` (E0308, ParquetError vs io::Error); that file is uncommitted-modified and on nobody's edit list, so no test in any of the three crates can run.

## 1. Boundary verdicts
- LOAD-BEARING, seam wrong: `model.rs` (377) returns a 4-tuple; `loop_block.rs` (724) needs `&lm_head` (a parent field) to compute L_Rec.
- LOAD-BEARING, well built: `config/` (494) + `train/cfg.rs` (300). The ADR-0005 drift check is real and provably fires.
- CONVENIENCE (delete/merge): `attention.rs` (52), `core::fnv_hash` (8, byte-identical to `dormouse_data::fnv`), `DormouseModel::loss` (8, 14 call sites, identity), the `kda: Tensor<4>` tuple slot (every call site binds `_kda`), `HostNgram::to_bytes` (11, test-only), `nan_reads` (3, never read), the 3 duplicated CUDA-downcast blocks (87 -> ~29).
- DELETE per ADR-0002: `gr.rs` (117), `act_quant.rs` (170), `jepa_targets.rs` (135). No A/B in any of the 77 logs in `/home/sehaxe/logs`.

## 2. Worst silent-wrongness in the model crate
- `crates/dormouse-core/src/model.rs:132-138` — the EMA JEPA teacher is fed **`targets`, not `input_ids`** (one-byte shift), with `hashed_ids: None` so its Engram arm is inert. The comment says "over the same inputs". Live on every step of every run (jepa_weight 0.05, no `--jepa-targets`). No test can see it.
- `model.rs:234` — `&& self.dspark_k > 0` gates the WHOLE aux channel: `--set dspark_k=0` silently kills JEPA too. Unvalidated.
- `model.rs:180-204` — `rec` and `logits` come from two different readouts; `loss` is named as if it computed something.
- `lib.rs:262-263` — "the step degrades into a no-op" is FALSE under the default `mix`: `HeadWiseMuon::step` (optim.rs:191-215) still decays momentum and orthogonalizes it, and AdamW still applies wd. The only test uses `adamw` with `wd=0.0` (lib.rs:1671).

## 3. Checkpoint round trip: 8 items missing
Missing/inexact, numerics-affecting: (1) EMA teacher, reset to a copy of the student every resume (lib.rs:648 -> ema_teacher_for :521-531); (2) the one-way fp32 factor fallback `ortho_fp32` (lib.rs:767,1013-1018); (3) the JEPA mask RNG (aux.rs:63) — no seed anywhere, so no two runs are ever bit-identical; (4) StressMonitor window/spikes (lib.rs:664); (5) `rand_depth` is `#[serde(skip)]` (lib.rs:78-91) so the drift check cannot see a knob that changes EVERY step's objective, while `steps`/`log_every`/`ckpt_every` — which cannot — are strictly compared: the exemption is exactly inverted. Inexact: (6) stream offset off by 2-3 batches (fresh run eats warmup :700 + quant-check :716 + prefetch :761 before step 0); (7) `.ngram` written AFTER the model ckpt, no rotation, no finite_scan; (8) no test asserts any of the 20+ `#[module(skip)]` fields is a function of `cfg`.

## 4. optim.rs routing
- SILENT failure is over-matching, not under-matching: `path.contains(m)` with unanchored markers
  ("expert_ffns.", "out_proj.inner"). `validate_routing` (optim.rs:416-495) catches a marker matching
  nothing, a 1D param, and a double-match — it does NOT catch a marker matching 400 params instead of 3.
  Only a *test* derives the expected count from the topology (lib.rs:1367); nothing in the runtime path does.
- `is_dense_bias` is duplicated in 3 places; the comment (optim.rs:116-120) admits they "used to disagree" — the abstraction already failed once and the fix was a second copy.
- Doc lies: optim.rs:66-68 says the [d,d] attention projections "stay on the fallback", but `QK_HEAD_MARKERS` routes KDA's two [768,768] matrices into Newton-Schulz in the DEFAULT mode, in all 9 production runs.
- String table is NOT deletable (burn's `ParamGroup` is path-based), but the PARALLEL table is. Two diff shapes given: the cheap one (enum-variant marker, ~20 min, kills `is_dense_bias` + 3 of 4 special cases) and the real one (`Route` trait in core + visitor-built exact-path groups, ~2 h, -40 LOC net, one failure class removed). Cheap version's prerequisite — burn's enum-variant burnpack prefix — is unverified; write the 5-line path-printing test first.

## 5. Top three to delete/merge
1. Offline-JEPA path, ~310 LOC: `jepa_targets.rs` (135) + `precompute_jepa_targets` (lib.rs:538-582) + `forward_with_jepa_targets` (model.rs:146-168) + 3 tests. **It is not the thing it replaces**: precompute uses `forward_latent(x, Some(h), None)`, online uses `forward_latent(targets, None, None)`. The "equality" test compares `forward_latent(x)` with `forward_latent(x)` and structurally cannot see it.
2. `gr.rs`, ~205 LOC incl. wiring + test. Off in all 8 configs, no run log, and **wrong at depth 1**: under GR the readout reads `h = h_ctx` (loop_block.rs:473,490), so `step_out` never contains iteration n's `y`; at `max_iter=1` (and 1-in-4 under `--rand-depth`) the whole body is bypassed.
3. `act_quant.rs`, ~200 LOC incl. plumbing. `ActFormat::attn()` (:34-39) silently upgrades attention to >=8 bits, so `--act-quant fp4` never runs 4-bit attention. AGENTS.md's "verified 100 steps" has no artifact in the log dir.

## 6. Highest leverage
`DormouseModel::loss(&self, batch: &Batch) -> (Tensor<1>, Diagnostics)` with `Batch { x, y, h, host_rows }`. Removes 14 identity-call sites, the forgotten `+ aux`, the mask-order-by-line-order hazard, the wasted final head forward, the dead kda slot, and the untested cubecl aliasing invariant. Also makes model.rs:132-138 impossible by construction. ~200 LOC touched, net +30, half a day. Wrong if RLVR/composed objectives arrive (then `Objective` belongs in train) or if the offline-JEPA sidecar survives — which is why the sidecar delete must come FIRST.

## 7. Stale claims to fix (worse than nothing)
burn-rmsnorm already applies `.require_grad()` (vendor/burn-rmsnorm/src/lib.rs:29-36), so model_seam.rs:266-269's "gains never receive a gradient" is false; "at init residual_scale is 0" (model_seam.rs:452-455, lib.rs:1745-1748) is false since loop_block.rs:214 inits to ones; `validation.rs:21` is an empty `if`; `cfg.rs:97-98` exempts a nonexistent key `"eval"`; ADR-0016 and ADR-0019 are cited but do not exist.
