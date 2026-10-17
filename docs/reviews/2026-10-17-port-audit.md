# The port, audited — 2026-10-17

Hostile read of `port/src`, `port/tests`, `port/scripts` against the repo's own
rules (`AGENTS.md` §1.1 LOUD/COUNTED/SILENT, §1.2 A/B-or-death, §1.3 zero
host-device sync, §1.4 claims name their evidence, ADR-0011/0018/0019/0021) and
against the measured record (`docs/reviews/2026-10-16-b1-profile.md`,
`docs/decisions/2026-10-15-1b-readiness-plan.md` §7).

What was run, read-only, with the card busy (a live `b1` warm attempt held
PID 492517):

- `CUDA_VISIBLE_DEVICES="" uv run pytest -rs` → **442 passed, 32 skipped, 39.24 s**
  (474 collected; the 32 skips are every CUDA/Triton/FLA gate). Run twice, same
  result — no order-dependence observed.
- Targeted probes of the claims below (all CPU, no CUDA, no writes outside this
  file): a `torch_compile` checkpoint round-trip, a storage-rung checkpoint
  through the eval loader, `validate()` on the dspark keys, and the
  `checkpoint(closure)` gradient question.
- Logs read, not written: `~/logs/b1warm.log`, `b1warm2.log`, `b1compile3.log`,
  `bench_dropless.log`, `smoke_long.log`, `ab_colrow_*.jsonl`, `ab_msa2_*.jsonl`.

The port is a real project, not a sketch: 12.8k lines of source with a working
instrument layer, an independent-reference test suite, and a documented
retraction discipline. The defects below are at the seams — the counter fast
path, the checkpoint format under `torch.compile`, validation placement — not in
the ported math, which is the part the gold tests actually pin. Counts:
**3 CRITICAL, 15 MAJOR, 14 MINOR.**

---

## Axis 1 — correctness risks in the hot path

| # | rank | finding | file:line | why it breaks |
|---|---|---|---|---|
| 1 | **CRITICAL** | `Counters.count()` reads a device scalar before it routes it to `count_device`: `if n == 0` runs first, and on a 0-dim CUDA tensor that is `bool(tensor)`, i.e. a **device→host sync**. Every `counters.count(name, device_tensor)` call therefore syncs per forward. | `instrument.py:62-69`; the live caller `experts.py:719` (`moe_dropped_assignments`, once per MoE forward = once per loop iteration per step) | The module's own docstring (`instrument.py:52-56`) claims this exact call was made sync-free to unblock `torch.compile`; it was not. The rule is ADR-0018 rule 2 (AGENTS §1.3). Fix is one reorder: `isinstance(n, torch.Tensor)` first. Verified on CPU that the tensor branch is reached *after* the `n == 0` test; the CUDA read is standard PyTorch semantics, not measurable here because the card is busy. |
| 2 | **CRITICAL** | `torch_compile=true` writes checkpoints **no loader in the repo accepts**. `train()` rebinds `model = torch.compile(model, dynamic=True)` (`loop.py:477-485`), then `save_ckpt` stores `model.state_dict()` (`loop.py:664`) whose keys carry the `_orig_mod.` prefix; resume loads into the **unwrapped** model (`loop.py:458-459`, compile happens after the load) and `eval`/`decode`/`serve` do the same (`cli.py:537-538`, `cli.py:573`, `serve.py:111-112`). | `loop.py:477-485`, `loop.py:664`, `loop.py:458-459`, `cli.py:537-538`, `scripts/run_b1.sh:54` | Verified: `RuntimeError: Missing key(s) … "_orig_mod.embed.weight"`. `scripts/run_b1.sh` ships `--set torch_compile=true`, so **the first real 1B run's checkpoints would be un-resumable and un-evaluable** — discovered only when the GPU hours are already spent. The wrap must happen after the save, or `save_ckpt` must strip the prefix (`state_dict` of `_orig_mod`), or the loader must unwrap. |
| 3 | **CRITICAL** | The b1 MoE recipe's capacity claim is falsified by its own first step: `moe_capacity = 2.0` is documented as the zero-drop setting (`configs/b1.toml:42-46`, `scripts/run_b1.sh:20`) but the measured step-1 log reads `moe_dropped_assignments=38026` of 49 152 assignments (4 iterations × 6144 tokens × k2) = **77 % of routed assignments never reach an expert**. | `configs/b1.toml:42-46`, `scripts/run_b1.sh:20`, `experts.py:501` (`keep = pos < self._capacity(...)`), measured in `~/logs/b1warm.log` (the `step=1` line) | Cap = `max(2, ceil(2.0·6144·2/1024)) = 24`/expert against a uniform load of 12 — so the 77 % is pure router skew at init, not a capacity arithmetic error. The run still trains and the loss curve looks healthy, but the routed experts are mostly off and the recipe's justification ("an assignment dropped is a gradient never seen", `b1.toml:45`) is argued from a 2026-10-15 A/B at a smaller shape. It is COUNTED, so not silent — but no document carries this reading. Re-derive the capacity (or trust LFB `moe_lb_coef=0.01`, which is sign-based and slow) before the long run. |

| # | rank | finding | file:line | why it breaks |
|---|---|---|---|---|
| 4 | MAJOR | Per-step device→host reads in the trainer: `finite = … bool(torch.isfinite(loss))` and `last_loss = float(loss.detach())`, every step, plus the same pair in SFT. | `loop.py:565`, `loop.py:588`; `train/sft.py:775`, `train/sft.py:783` | ADR-0018 rule 2 bans exactly this shape in a step ("no host-side branch on a device value"). The burn answer was `mask_fill` on device + a counter read at log cadence; the port reads twice per step and pays a serialization each time (and it is the reason a CUDA-graph stage cannot include the loss). |
| 5 | MAJOR | SFT's per-step host reads: `n_mask = int(mask.sum())` (device bool tensor), and `sft_mask`'s per-row `int(end[i])` + `bool((…).all())` inside a `for i in range(b)` loop. | `train/sft.py:767`; `train/sft.py:576` | `O(b)` syncs per step on top of #4. The mask validation is deliberate ("checked at every step, not assumed"), but it is validated by reading the device; a vectorized check plus one read at log cadence would keep the same guarantee. |
| 6 | MAJOR | Storage rungs (`quant = fp8/nvfp4/ternary`) are only half-wired: `eval`, `decode` and `serve` load the model half with raw `torch.load` + `load_state_dict`, never through `unpack_state_dict`. | `cli.py:537-538`, `cli.py:573`, `serve.py:111-112` vs `loop.py:179-193` (`unpack_state_dict`), used only by `load_ckpt` (`loop.py:238`) | Verified: the load dies with `RuntimeError: … expected torch.Tensor or Tensor-like object from …` naming neither the rung nor the escape — a config the validator accepts, a feature the tests pin (`tests/test_ladder_storage.py`), and no consumer outside `--resume`. The test's own save→load path calls `unpack_state_dict` twice, which is why nothing caught it. |
| 7 | MAJOR | `_banks` rebuilds the fp32 factor stacks on **every optimizer step** (the cache key is the parameters' `_version`), a ~2.9 GB transient at b1. This is the allocation that OOM'd the two warm attempts. | `experts.py:534-564` (stack at `:550`), the failing size in `~/logs/b1warm.log` = 1 432 354 816 B = `[1024, 64, 5461]` fp32 | Not a wrong number — a ceiling. At 16 GB with the desktop live, the full 1.07B preset dies at step 2 (`b1warm.log`) and the compile warm-up died the same way (`b1compile3.log`). The cache docstring's "one conversion per step is the honest floor" is right about conversions and silent about the fp32 source stack, which is what the card actually cannot hold. |
| 8 | MAJOR | `adaptive_msa` converts a device scalar with `float(share[i].detach())` **per iteration** to size the block count. | `loopblock.py:294-301` | A sync per iteration on an arm that exists to be A/B'd; the rounded share is needed on the host by construction (the Triton grid is host-side), so this needs a device-side `k` (or a fixed per-slot budget) before the arm is measurable at b1 speed. |
| 9 | MAJOR | `_naive_dispatch` walks experts with `torch.unique(...).tolist()` — a sync per forward on the reference arm, which is also the CPU path and the counted fallback for every other arm. | `experts.py:649` | Same class as #1; the fallback is supposed to be the cheap-correct one. A host `bincount`-free alternative is to iterate `range(n_experts)` with a mask, which is what `_triton_dispatch` effectively does. |
| 10 | MAJOR | `attend_selected_blocks` asserts `torch.isfinite(den).all()` on a device tensor — a sync per MSA forward. | `sparse/attention.py:355` | The invariant is real (the docstring argues it), but it can be asserted on the mask the plan already carries (`valid.any(-1)`, a device predicate), or at eval cadence. MSA is off in `b1` (`configs/b1.toml:29`), so this is a latent-arm cost. |
| 11 | MAJOR | The eval window can silently double-score bytes: `open_streams` caps the eval tail at 1 MiB, while the printed window is `eval_batches × batch × seq_len`. | `data.py:161-172` (`ev_bytes = min(max(eval_window_bytes(tr), 1), 1 << 20)`), window printed at `loop.py:387-388` | At `eval_batches 20 × batch 8 × seq_len 8192` the window is 1.31 MB > the 1 MB range, the ring wraps, and the same bytes are scored twice while the line claims a bigger window — the burn §2.6 "a BPB is only comparable within one window" defect in a new dress. Does not bite the b1 shape (165 KB) or the runbook (separate `--eval` file); it bites the long-context direction. |
| 12 | MINOR | dspark's own coherence checks live **inside `_validate_byteflow`**, which is only called when `use_byteflow` is true — so in the default path `dspark_weight = -1.0` and `dspark_k = 0` / `dspark_stride = 0` pass `validate()`. | `config.py:770-771` (the only call site), guards at `config.py:873-889` | Verified: `validate(ModelConfig(dspark_weight=-1.0))` raises nothing. Impact is bounded — the head is built only when the weight is `> 0` (`model.py:95-97`) and `DSparkHead.__init__` re-checks `k`/`stride` loudly (`dspark.py:222`) — but the field is a config that lies about being refused, which is the class §1.7 calls a defect. |
| 13 | MINOR | NaN-skipped steps leak `Engram._pending` entries: the skip branch never calls `host_step`, so that step's device leaves stay in the list (and their `uniq` rows are re-deduped on the next real step). | `engram.py:329` (`self._pending.append`), `engram.py:347-383`, skip branch `loop.py:589-597` | Memory proportional to the NaN count (the burn record has NaN *floods*), and a stale leaf whose grad is `None` is filtered — so no wrong update, just growth. A `clear` on the skip branch is ~2 lines. |
| 14 | MINOR | `train()` sets `torch.set_float32_matmul_precision` process-globally from `cfg.tf32`. | `loop.py:417-421` | Fine as a knob, but it also decides whether the Triton kernels' explicit `input_precision="ieee"` (`moe_kernels.py:118-120`) agrees with the padded arm's bmm — the two MoE arms are only comparable at one setting, and nothing in the log says which one a run used. |
| 15 | MINOR | `byteflow_logdet = true` selects a pure-Python, per-position Cholesky path that reads device scalars one at a time. | `byteflow.py:390-398`, selected at `byteflow.py:519`; `RatePatcher.push` (`byteflow.py:584-607`) is the same shape and is test-only | Off by default and documented as the exact (slow) form, but it is a config field with no refusal, and on a CUDA run it is a sync storm, not a slow path. |
| 16 | MINOR | `LoopBlock.forward` stores the per-iteration CEs on the module (`self._per_iter_ce`), read later by `DormouseModel`'s breakdown. | `loopblock.py:231`, read at `model.py:278` | Module-level mutable state that two forwards would clobber; harmless today (one forward at a time) and a trap for any concurrent/eval-interleaved use. |
| 17 | MINOR | `evaluate`'s annotation says `tuple[float, int] | dict[…]` but it returns `_eval_loss`'s 4-tuple. | `loop.py:332` vs `loop.py:316-321` | The CLI unpacks 4 (`cli.py:542-544`) — the annotation is the only thing wrong, which is exactly how a caller gets written against a lie. |
| 18 | MINOR | `RunConfig.from_snapshot` builds the model config with `_from_table` but never calls `validate`. | `config.py:580-591` | `eval`/`decode`/`serve` resolve from the snapshot (`cli.py:159-191`), so a snapshot carrying a self-inconsistent model config (hand-edited, or written by an older schema) is used without the coherence pass the preset path enforces. |

**Checked and found sound** (so the list above is calibrated, not exhaustive by
suspicion): the eval forward passes `input_ids` to the loop, so the memory arm
runs and `engram=<rows>/<arms>` is counted on the eval's own counters
(`loop.py:300`, `engram.py:291-292`) — the burn `7adda92` defect is genuinely
fixed here; `_forward_segmented`'s state carry and CE divisor are right
(`loopblock.py:479-522`); `checkpoint(closure)` with **no** tensor arguments
still propagates gradients to captured tensors *and* to a captured non-leaf
(probed directly on torch 2.13, CPU) — the offload arm's input gradient is not
lost; `_padded_dispatch` zero-fills the bank (`experts.py:615`) for the reason
its comment gives; `memory_floor_mix` clamps as documented (`engram.py:469`).

---

## Axis 2 — architecture

| # | rank | finding | file:line | why it breaks / what a 10× model does |
|---|---|---|---|---|
| 19 | MAJOR | The bank cache is O(E·d·r) fp32 **rebuilt per step** and nothing chunks it (`MUON_CHUNK_ELEMS` caps Newton–Schulz, not the stack). | `experts.py:534-564`, `optim.py:85` | At 10× (E 4096–8192, d 4096) the stack alone is 10-40 GB fp32; the cache must become a `dt`-resident stack built once (the `.to(dt)` cost it was built to avoid) or be paged. Today it is the measured b1 OOM trigger (#7). |
| 20 | MAJOR | `mm_mixed` casts each fp32 master to the compute dtype **per forward** (`weight.to(dt)`), for the controller, `mem_dense`, `down`, `router` and every KDA projection. | `tsct.py:64-75`, callers `loopblock.py:273/312/323`, `experts.py:377/437` | This is the port's documented mixed-dtype rule, but it is a per-forward copy of ~O(params of those projections) — at b1 the KDA set alone is 16.8M × 2 B × 2 (two directions) × 4 iterations per step. A 10× model turns a rule into a bottleneck; the cache discipline `_banks` learned needs to generalize. |
| 21 | MINOR | The S3T mask stream has no step index, so a resume restarts it. Documented, printed at start, and the trainer does say so. | `tsct.py:150-166`, printed `loop.py:426-437` | Correctly COUNTED, so not a defect — listed here because it is the one place where the port's own determinism rule (ADR-0021: masks are pure in `(seed, step)`) is knowingly not met, and it is load-bearing for `factor_qat` reproducibility. The fix the docstring names (thread the step in) is cheap and should land before any QAT run. |
| 22 | MAJOR | Two firewall copies (train and SFT) that the file itself says must not drift. | `train/sft.py:36-40`, `sft.py:775-788` vs `loop.py:565-597` | Already true in the first week: the train loop reads `loss is not None and bool(isfinite)` while SFT reads `bool(isfinite(loss))` on a non-optional loss. A shared `firewall()` is the fix the docstring names. |
| 23 | MINOR | `json_log` is documented as re-pointable on a resume but is not a progress key, so `diff_keys` refuses the flip. | doc `config.py:474-476`, `PROGRESS_KEYS` `config.py:45`, `diff_keys` `config.py:556-574` | A resume that only moves the telemetry sink dies with "resume refused". One of the two is wrong. |
| 24 | MINOR | The colrow default is `colrow` everywhere in code, and two docstrings still call `frobenius` "the default". | `optim.py:237-259` (docstring) vs `optim.py:319` / `optim.py:511`; `config.py:105-113` vs `config.py:453-466` | §1.7 class: a reader of `colrow_normalize`'s docstring learns the opposite of what the trainer runs. |
| 25 | MINOR | `_unsupported` covers unported mechanisms well, but the schema is one 90-field flat dataclass and `from_snapshot` skips `validate` (see #18). | `config.py:121-546`, `config.py:580-591` | At 10× the field count is the least of it; the risk is that the snapshot path is the one used by `eval`/`serve`, i.e. the one that decides whether a served model is the trained one. |
| 26 | MINOR | Resume/checkpoint safety is otherwise strong: atomic `.tmp` + `replace` for weights and the `.ngram` sibling, a snapshot drift refusal with exactly the burn progress keys, a LOUD refusal when the sibling is missing, shape-checked host tables. What is owed is the one thing the burn side named: no `train_step` stamp on the `.ngram`, so a crash between the two writes leaves a mismatched pair nothing can detect. | `loop.py:155-176`, `loop.py:203-219`, `engram.py:426-453`; the owed stamp is stated at `loop.py:72-84` | Carried debt, documented — the audit agrees with the owner's own note; it is not a new finding. |

**Where the three custom mechanisms sit.** TSCT (`tsct.py`), Engram
(`engram.py`) and ByteFlow (`byteflow.py`) are each one file with the paper
reference, the deviations, and a gold/property test; everything else is a
dependency (FLA, Triton, bnb, torchao-shaped storage rungs) behind a
`COUNTED` fallback with the reason attached to the counter object
(`experts.py:722-755`, `kda.py:216-226`). That is the plan's §7 architecture
followed literally. The dependency surface is honest: FLA presence is proven by
a warmup call, not an import (`kda.py:92-133`), and the SDPA fallback is a
different function that says so (`kda.py:288-315`).

---

## Axis 3 — optimization state vs the measured profile

Baseline: `docs/reviews/2026-10-16-b1-profile.md` (optimizer 530-850 ms of a
1399 ms step, clip 107 ms, MAGMA 211 ms, dense-MoE launch loop, per-param Muon).

| what the profile named | state today | evidence |
|---|---|---|
| ColRow 3-D bug (`1/√d0` of the definition, 8-45× weak) | **fixed and flipped to default** (`colrow`), A/B −0.0428 BPB | `config.py:453-466`, `optim.py:264-283`, gate `tests/test_muon_fast.py` |
| `clip_grads` 107 ms (2228 `add_`) | **fixed**: one `_foreach_norm` + one `_foreach_mul_` | `optim.py:609-630`, comment cites the 252 ms profile |
| per-param Muon (46 341 `mm`) | **fixed**: shape-stacked momentum state + batched NS + `_foreach_add_` writeback | `optim.py:286-360`, `optim.py:410-473` |
| bank re-conversion (~15 ms/fwd) | **fixed**: converted-tensor cache keyed by `_version` | `experts.py:521-566` |
| dense-MoE 24-launch loop | **fixed**: one banked bmm, counted fallback | `experts.py:393-414` |
| dropless dispatch (new since) | **landed**: grouped GEMM with the static row-block bound and an on-device empty-block exit, no `offs` read-back | `moe_kernels.py:104-115`, `:176-177`, `experts.py:757-813`; `~/logs/bench_dropless.log`: **65.4 ms fwd+bwd vs 1630 ms naive (25×)**, 41.6 ms padded |
| MAGMA fp32 GEMM | **partly**: `mm_mixed` routes the loop's own projections through bf16 cuBLAS; KDA internals, the fp32 head and NS still hit MAGMA unless `tf32=true` (off by default, parity check owed) | `tsct.py:64-75`, `config.py:387-393` |
| `torch.compile` (the 2× lever) | **unlanded**: three attempts today (`b1compile3.log`, `b1warm.log`, `b1warm2.log` still in inductor compile) and the eager+compile path OOMs on the bank stack; and see #2 — even if it lands, its checkpoints are broken | `~/logs/*.log`, `scripts/run_b1.sh:54` |
| NS steps 5→3 (≈ −130 ms) | **not taken** | `config.py:416` (`ns_steps: int = 5`) |
| iteration-level activation checkpointing (the 14.5→12 GiB enabler) | **owed**; `segment_len` exists but is a long-context mechanism with its own refusals, not a memory mode of the b1 shape | `config.py:513-544`, `loopblock.py:407-523` |
| optimizer ~44-53 % of the step | **unchanged by all of the above** — ColRow is a scale fix, the batching removes launches, not FLOPs. The profile's `opt.step` 530 ms at b1 is still the honest starting point | `2026-10-16-b1-profile.md` §1/§3 |

**The honest ceiling at b1 (1.066B) on a 16 GB 5060 Ti, today.** The profile's
shape table (Engram OFF, tf32 ON) says the desktop-safe form is **b8·s768 at
13.5 GiB reserved, 5.8k tok/s**; 6.9k tok/s needs b16·s512/b4·s2048 at 14.5 GiB
with the desktop closed, and b8·s1024 is unstable. The full preset — host
Engram tables + bnb8bit + compile — **does not fit at b8·s768 right now**: both
warm attempts died at step 1→2 allocating the `[1024, 64, 5461]` fp32 bank
stack (1.43 GB) with ~1.4 GiB free. So: ~663M on-card / 57M active at 5.8k
tok/s is the measured, repeatable operating point; 1.07B with tables needs the
desktop closed **plus** one of (activation checkpointing, a smaller b×s, or the
bank stack moved to the compute dtype once). At 5.8k tok/s, 1B tokens ≈ 2 days
and 10B ≈ 20 days of wall clock — the plan's 5-10B-token budget is a
multi-week program at this ceiling, not the 2-9 days §2.2 derives. That
derivation assumed 5-20 effective TFLOP/s; nothing measured at b1 has reached
it yet (the spike's 15.7-20.5 TFLOP/s rode `compile + bnb8bit`, the recipe that
currently OOMs).

---

## Axis 4 — test-gate quality

474 gates collected, 442 green on CPU in 39 s, **32 skipped without a card**
(every Triton grouped-GEMM parity gate, the padded-bank bf16 gate, the FLA KDA
gates, the CUDA MSA routes, one offload gate). The CPU suite is a real suite:
independent references, not self-comparisons.

| # | rank | finding | file:line | why it matters |
|---|---|---|---|---|
| 27 | MAJOR | **Missing gate:** nothing saves and resumes a `torch_compile` run, and nothing loads a checkpoint with the CLI's own eval/decode loader. The compile gate only spies on the wrap (`test_torch_compile_flag_wraps_the_model_after_resume`) and the storage-rung gate calls `load_ckpt` + `unpack_state_dict` by hand. | `tests/test_train.py:151-169`, `tests/test_ladder_storage.py:87-107` | Both defects (#2, #6) live exactly in the gap between "the trainer's resume path" and "every other loader". One round-trip test through `dormouse eval` (or `_snapshot_run` + `load_state_dict`) would have caught both. |
| 28 | MAJOR | **Missing gate:** the rule with no instrument — no test asserts that a training step performs zero host reads. The 0-sync rule is asserted by comment (`instrument.py:52-56`) and by one CPU-only drop-parity drain (`tests/test_moe_dropless.py:263`), which passes because CPU tensors do not sync. | `instrument.py:62-69`, `tests/test_moe_dropless.py:263` | The cheapest gate is a CPU test that monkeypatches `Counters.count` and asserts a device tensor reaches `count_device`; the CUDA-side count of syncs needs a `torch.cuda` probe. Without it, #1 was invisible for the whole port history. |
| 29 | MAJOR | **Gates that do not run where the work happens.** 32/474 skip on CPU, and they are the gates for the arms b1 uses (FLA attention, padded bmm MoE, bf16 numerics, Triton routes). The CPU suite tests the *SDPA* attention arm and the *naive* MoE arm — different functions, honestly labelled, but the trainer's real arms are only gated on a quiet card. | `tests/test_kda_gpu.py:45`, `tests/test_moe_kernel.py:52`, `tests/test_moe_dropless.py:40`, `tests/test_experts_gpu.py:24`, `tests/test_msa_speed.py:54` | Not a defect in the tests; a defect in the evidence base: "the suite is green" is a weaker statement here than it reads, and the b1 recipe rides the skipped half. |
| 30 | MAJOR | **One A/B verdict rests on a crossed curve with one budget.** The ColRow default flip (−0.0428 BPB, the port's largest win) carries, in its own comment, "the arm's curve CROSSES (worse at step 1000, better at 2000), its own spread is 3.7× the control's … re-confirm on `medium` and `b1`". | `config.py:456-465` | 3 seeds × 2k steps at one preset: a legitimate pre-registered read, and still a verdict that a *crossing* curve plus an inflated spread is the load-bearing evidence for. The `medium` 40k-step run happened but was a calibration, not this A/B — so the fix that everything after it is measured against is unconfirmed at any other scale. The MSA tie (3.0103 vs 3.0113, 3 seeds each, `ab_msa2_*`), the aux tie (3 seeds) and the depth verdict (4 arms × 3 seeds) are correctly seeded and correctly read. |
| 31 | MINOR | A fixture that is never used as a fixture: `q_alone` is built and filled in `test_banked_and_standalone_experts_agree_under_qat`, then the assertions compare `factor_qat_project` against `factor_qat_project`. | `tests/test_factor_qat.py:135-154` | The parity claim survives (the `dim` axis differs), but the "expert alone == bank slice" end-to-end forward the docstring promises is not actually run. |
| 32 | MINOR | `test_moe_dropless.py` builds its own `offs` fixture and asserts on it (`assert int(counts[0]) == 0 and int(offs[0]) == 0`) — true by construction. | `tests/test_moe_dropless.py:94-127` | Harmless; named so the "which gates can fail" question has a complete answer: the rest of that file compares against a per-expert reference loop and a monkeypatched launch count, which is the right shape. |
| 33 | MINOR | Order-dependence: none found. The S3T stream is module-level global state, but every test that reads it re-seeds explicitly (`set_factor_qat_seed`), and the suite ran green twice with identical results. | `tsct.py:83-113`, `tests/test_factor_qat.py:270-283` | Recorded as checked, not as a finding. |

**Mechanisms with no gate at all** (beyond the two missing round-trips above):
the per-step sync budget (#28); the eval-tail 1 MiB cap (#11); the b1 capacity
drop rate (#3); `json_log`'s resume exemption (#23); the `_pending` leak on NaN
steps (#13). Each is cheap to add and each is a place a regression would look
like a working run.

---

## What is genuinely well done

- **The instrument layer is real, not decoration.** `LoudError` is a
  `RuntimeError` so an uncaught degradation is a non-zero exit
  (`instrument.py:28-39`); every fallback carries a reason on the counter object
  and the log prints it (`experts.py:746-752`, `kda.py:223`); counters are
  drained once at a declared cadence (`instrument.py:80-97`); the eval line
  carries the byte window *and* the arm counters (`loop.py:385-390`).
- **The config seam enforces what it says.** Unknown keys are LOUD with the
  escape (`config.py:614-625`); unported mechanisms refuse when non-default
  (`config.py:599-611`); incoherent pairs are refused, never clamped
  (`config.py:632-794`); the resume snapshot refuses any non-progress drift
  (`loop.py:220-229`). The `segment_len` validation set
  (`config.py:920-994`) is a small catalogue of silent-model-change traps and
  each one is refused with the measurement that found it.
- **The tests are gates, not mirrors.** The b1 parameter count is pinned
  against hand arithmetic that is independent of the module tree, with an
  `AssertionError` for any parameter that falls outside the buckets
  (`tests/test_param_count.py:79-122`); MSA parity is against a per-query
  longhand loop written in the test (`tests/test_msa.py:144-177`); the dropless
  MoE is against a per-expert loop with empty experts in the fixture
  (`tests/test_moe_dropless.py:109-128`); memory is asserted by spying on the
  5-D tensors the arm builds (`tests/test_msa.py:244-255`); the MoE-offload
  arm's exactness is bit-equality of the loss plus 1e-6 gradient parity
  (`tests/test_offload_experts.py:119-158`).
- **The retraction discipline ported.** `tsct.py`'s docstring states the S3T
  resume caveat where a reader meets it *and* the trainer prints it at start;
  `optim.py` keeps the ColRow defect as the named control arm; the storage rung
  is explicitly a checkpoint format and not a forward-path change
  (`tsct.py:23-35`); the ByteFlow logdet/L2 split is documented with the cost.
- **The dependencies are held at arm's length.** FLA is optional and its
  absence is a counted fallback with the device named; Triton is an ability, not
  a dependency (`experts.py:129-135`); the triton kernels carry no module state,
  no allocator and no host read (`moe_kernels.py:13-31`), and the dropless route
  replaced a host read-back with an on-device early exit — the ADR-0018 rule
  applied as a design constraint rather than an afterthought.
- **The 1B shape is priced honestly.** `configs/b1.toml` carries the corrected
  expert arithmetic, the delta against the plan's table component by component,
  and the admission that the active-compute figure is derived, not measured.

Calibration: this is a well-built project — the strongest of the port's three
weeks — and the defects above are the seams of a fast build, not rot. Two of the
three CRITICALs are a one-line reorder and a checkpoint-key strip; the third is
a recipe decision that the port's own counters already disprove.

---

## The next 24 hours, ranked by risk × effort

1. **Fix the checkpoint under `torch.compile`** (#2, CRITICAL, ~30 min):
   save `model._orig_mod.state_dict()` (or strip the prefix in `save_ckpt`) and
   add the save→resume→eval round-trip gate (#27). Until this lands, every hour
   of the b1 recipe's default run is spent on checkpoints nothing can load.
2. **Reorder `Counters.count`** (#1, CRITICAL, 5 min): test
   `isinstance(n, torch.Tensor)` before `n == 0`, and add the CPU gate that a
   device tensor reaches `count_device` (#28). This is the port's own rule
   violated in the hot path of the arm b1 runs 4× per step.
3. **Re-derive the b1 capacity** (#3, CRITICAL, 1-2 h): the measured 77 % drop
   rate at step 1 falsifies the runbook's "zero-drop" claim. Either raise
   `moe_capacity` until the step-1 counter reads near zero at b8·s768, or make
   LFB act on a per-window load, or accept the drops in writing — but do not
   launch a multi-day run whose recipe document contradicts its own counter.
4. **Un-wire the storage rungs from the CLI's blind spot** (#6, MAJOR, ~1 h):
   route `eval`/`decode`/`serve` through `load_ckpt`/`unpack_state_dict`, and
   gate one rung end-to-end through `dormouse eval`. Same shape as #27 — the
   tests only exercise the path the trainer uses.
5. **Land the `_banks` stack in the compute dtype** (#7, MAJOR, 2-4 h):
   the per-step fp32 stack is the measured OOM trigger at b1 and the reason the
   two warm attempts and the compile warm-up died. Building the stack once in
   `dt` (with a `_version`-keyed rebuild only on quantisation/alpha changes)
   removes ~2.9 GB of transient and is the precondition for `torch.compile`
   ever completing at b8·s768.

Owed outside the 24 h: the `docs-site/tools/manifest.mjs` line for this
document (`AGENTS.md` §1.8) — this audit was read-only and may write only
itself.
