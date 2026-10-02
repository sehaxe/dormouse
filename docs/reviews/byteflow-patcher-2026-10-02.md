# The byteflow lane, round 3 — the dynamic coding-rate patcher and its wiring

**Lane:** wt/byteflow3 (base `origin/wt/byteflow2`, the resurrected crate),
2026-10-02. The mission: the owner's decision — «срочно внедрить наш
byteflow», с патчером по **coding-rate** — решение владельца, не BLT-энтропия,
не фиксированный шаг:

- `R_ε(H) = ½·logdet(I + (d/ε²)·H·Hᵀ)` on the hidden states — paper eq. (11);
- `ΔR_t = R_ε(h_1:t) − R_ε(h_1:t−1)` — the marginal contribution of byte t, eq. (12);
- patch borders = **Top-K by ΔR_t**, BOS forced, chronological — §3.2's own
  selection rule, the paper's argument against a global threshold being that
  it breaks the static compute graph.

## 1. What the paper's recipe is (and what was NOT invented)

The chunker scores the **local encoder's hidden states** `h_1:T ∈ R^{T×d_local}`
(p.4 §3.2: "the local encoder produce contextualized representations"). That
is the «лёгкий драфт-модуль»: SWA + Canon layers, linear cost — not the main
model. The crate (`vendor/dormouse-fused/crates/burn-byteflow`) already carried
this against the byteflow2 findings doc; what was missing is the piece that
lets it sit behind a BYTE STREAM (dormouse-data streams bytes, window W) — the
patcher as STATE:

- `burn_byteflow::RatePatcher` (`src/stream.rs`): `push(h: &[f32]) -> ΔR_t`
  per position; window state is the d×d Gram (LogDet) or one running sum of
  squared norms (L2) — **O(d²) memory, never a T-row buffer**, so a stream of
  any length patches in bounded memory; `reset()` begins the next window
  (one window = one sequence `h_1:W`, the paper's own per-sequence Top-K
  semantics — carrying the Gram across windows would change what ΔR means).
- The two modes are the paper's own pair: LogDet = eq. (11) computed
  incrementally through the same Cholesky + Sylvester route the batch path
  takes (`ponytail:`-marked O(d³) per push; a maintained rank-1 Cholesky
  update is the O(d²) upgrade if a profile ever asks); L2 = Appendix B's own
  streaming approximation (`R ∝ ‖H‖₂`), O(d) per push.
- `borders(k)` = `{position 0} ∪ top-(K−1) by ΔR`, chronological, ties broken
  on position — the same set `select_positions` returns from batch gains.

## 2. The oracle — formula against the direct log-det, green

`tests/dynamic_patcher_oracle.rs` (CPU ndarray, f64 on both sides):

| gate | what it pins | bar |
|---|---|---|
| `test_coding_rate_matches_direct_logdet` | the streamed rate AND every prefix `ΔR_t` against the paper's **LITERAL T×T eq. (11) matrix** `I_T + (d/ε²)·H·Hᵀ`, log-det taken by **LU with partial pivoting** — a third decomposition of the same equation (no Cholesky, no eigen, no Sylvester); plus the telescope identity (ΣΔR = R(h_1:T)) and the d×d direct cross-check that the Sylvester folding is right | 1e-9 rel (measured: far below) |
| `test_deltaR_topk_borders` | a 64-position stream with three planted 100× bursts at 10/30/50: the borders land on the bursts in BOTH modes, BOS stays first, chronological, `borders(1)=[0]`, `borders(T)=0..T`; streaming borders == batch `select_positions` **within each mode**; streamed LogDet gains == the batch values at the batch path's own f32 round-trip | 2e-7 rel (the f32 output of `marginal_gains_exact` is the floor) |

**Falsify (the gate can fail, proven):** the missing-½ and `ε²→ε⁴` mutants go
red against the direct route; the **static-stride chunker** — the v0 this lane
replaces — misses all three planted bursts. NOT asserted: cross-mode border
equality — the paper's own Table 4 prices L2 vs log-det at ~0.01 BPB of
VALIDATION loss, not rank-identity at init (the same finding
`byteflow_rate_oracle.rs` already carries; on the burst stream they disagree
only on one non-burst border).

**Why these two tests are the strongest available tier:** the paper's code is
not public (byteflow-rate-2026-10-02.md §3, tier (b) — prospected empty);
the fixture-based oracle that lane wrote already pins Rust-vs-numpy(eigen) at
7.8e-8. This round's gates add the route the fixture oracle does not check:
the T×T literal form against an LU route, computed inside the test, no
fixture file to rot.

## 3. The integration — `--byteflow`, off by default

- **Config** (`dormouse-core/src/config/schema.rs`): `use_byteflow: bool`
  (default false) + the byteflow_* hyperparams, serde-defaulted to the
  byteflow_9m build (`d_local 96, d_global 768, k_tokens 128, e_layers 2,
  g_layers 2, heads 4/8, w_local 256, d_ff 256/2048, bins 16, eps2 0.5,
  max_bytes 512`); all reachable from `--set`/preset (the override match's
  unreachable-field trap does not grow).
- **Refusals, always naming the escape** (ADR-0011): `config::validate`
  refuses `use_byteflow` with any dormouse arm on (kda/engram/mor/gr/attnres/
  mhc/situ / moe-tuning / aux weights) and `max_seq_len > byteflow_max_bytes`
  (the RoPE bound, here instead of a forward assert); the trainer
  (`dormouse_train::byteflow::check`) refuses the train-side arms
  (`--jepa-targets`, `--engram-ram`, `--graph-capture`, `--rand-depth`,
  `--eval-depths`, `--stress`, `--bf16`, `--opt ∉ {adamw, mix}`).
- **The loop** (`dormouse-train/src/byteflow.rs`, dispatched FROM
  `train_loop` after the snapshot and after the device seed — the dormouse
  files otherwise untouched, the NaN firewall NOT touched): ByteFlowNet on
  the shared data seam (`ByteStream::train_and_eval` keeps the no-leak
  rule), the honest gathered CE, loss masked by the same `mask_nonfinite`
  plus a ByteFlowNet gradient sanitizer (missing grad for a visited param =
  loud error), **AdamW** (the mix router walks `DormouseModel`'s tree; the
  step-0 arm line says what ran — `params= patcher= k= bins= eps2= opt=
  quant=unused` as COUNTED), WSD lr, rewind-before-eval with the scored byte
  count printed on the eval line (§2.6's window rule).
- **Checkpoints**: the trainer's container pattern with its own magic
  (`DMBF…`) — model + optim records, atomic tmp+rename, `.prev` hard-link
  rotation, full resume (record round-trip + `skip_bytes` fast-forward); a
  foreign container with the same `<name>` is a loud magic panic, never a
  silent start-over.
- **Preset**: `configs/byteflow.toml` (the arm on, every dormouse arm
  explicitly off so the validators stay quiet).

## 4. Tests (all CPU, GPU never touched — the card is the graph-stage lane's)

| suite | result |
|---|---|
| `burn-byteflow` crate (vendor workspace) | **27 passed** (16 net + 7 rate-oracle + 2 dynamic-patcher + 1 field-oracle + 1 doc-test), clippy clean on the new code |
| `dormouse-train` lib | **58 passed** — the dormouse path's own tests are dispatch-untouched |
| `dormouse-train/tests/byteflow_smoke.rs` | **2 passed**: `one_step_trains_and_saves_through_the_real_entry` (1 step, batch 2 s16, flex; ckpt + ce line + config snapshot) and `a_preset_arm_conflict_is_refused_loudly` |
| `dormouse-core` lib | **89 passed** (schema/validate/load — snapshots carry the new fields) |

## 5. QUEUED — the GPU A/B (the release gate, blocked on the card)

The 200-step comparison the byteflow2 findings doc's §6 spec'd:

- **control**: the byte-level dormouse recipe (`--preset small`, aux 0, batch 8,
  seq 512, seed 1, fp32) — rebuilt on a card where the attention arm actually
  trains (the post-`8fa5d4c` precondition in AGENTS §3.4; read `fused kda=`
  and `engram=` on the control's eval lines before believing anything).
- **arm**: `--preset byteflow --byteflow --batch 8 --seq-len 512 --seed 1`
  (AdamW per the arm line), same data window, same eval-batches → the SAME
  scored window (§2.6: the window is `eval_batches × batch × seq_len` — one
  batch size for all arms).
- **metric**: best held-out BPB (the log's eval lines; the scored window is
  named in the comparison). Smoke-first: 20 steps both arms, NaN and speed,
  per AB-PROTOCOL's ladder.
- blocked because ONE heavy thing at a time — the card is running the
  graph-stage lane; launch only when it is free, under the build lock for the
  binary build and `systemd-run --user --scope -p MemoryMax=40G` for the run.

## 6. Honest gaps, carried

1. **The LogDet chunker path under AUTODIFF is compile-verified only.** The
   crate's own LogDet test runs on a bare (non-autodiff) device; under the
   trainer's autodiff device `marginal_gains_exact` host-syncs an autodiff
   tensor (`into_data`) — the smoke runs the default L2 path, so the exact
   route under backward is UNTESTED. It is the analysis path; the L2 default
   is what the paper streams.
2. **The arm line's `quant=unused` is a COUNTED statement, not a refusal** —
   a byteflow preset that carries `--quant fp8` runs fp32 and says so. Same
   for `use_tsct`, which no byteflow reader reads.
3. `rope_base 500_000` remains the `TODO(бумага)` carry from the crate (the
   paper states only "Multi-level" positional handling).
4. The `VOCAB = 256` vs the paper's 258 divergence stands as the byteflow2
   findings doc left it (owner call, inert while RETAIN_TOKEN_IDS is empty).
