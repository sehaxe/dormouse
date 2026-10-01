# ADR-0019: no silent fallbacks

Date: 2026-09-27. Status: accepted. Extends ADR-0011 (loud failures, P10 R5).

## The rule

**A fallback must never be the only thing that happened.**

Every site where a value, a kernel, a file, or a config field degrades to
something else carries one of three marks, and no fourth:

| mark | meaning | test |
| --- | --- | --- |
| **LOUD** | hard error naming the cause AND the escape, before or at the moment of failure | the run dies with a message that says what to do |
| **COUNTED** | the fallback happens, and a counter or a structured log line says it happened | a reader of the training log sees it without a profiler |
| **SILENT** | it just happens | **a defect. Fix it or add a counter.** |

The reason the class is so expensive here: **a silent fallback usually
computes the right answer.** That is what makes it dangerous — nothing fails,
no assertion trips, the loss curve looks plausible. Four of the five biggest
defects found in a day were this class, and none of them announced itself:

| found | symptom | cost |
| --- | --- | --- |
| `ByteStream::rewind` after the ring drained (2026-09-27) | eval scored a different window than the line above it claimed | cross-run A/Bs were noise (6.443 vs 6.551 for one checkpoint) |
| fused gated-delta dispatch compared a full backend type (`vendor/dormouse-kda/src/fused.rs:35`) | the fast path was dead for a year; the tensor path is the same function | a year of "why is this slow" |
| the NaN firewall's "raw" loss read the MASKED value (burn `clone()` shares the device buffer; `mask_fill` is in-place on CUDA at two handles) | a NaN step logged `ce=0.000`, `best` stuck at 0 forever | the firewall blinded itself |
| the parquet branch of `Source::read` returned a length it never copied | any `.parquet` corpus panicked or ingested garbage | books + qa (4.5 GB) were never trainable |

## The shapes to grep for

```sh
grep -rn --include=*.rs -E 'unwrap_or|\.ok\(\)|return None|None =>|if let Err' crates/ vendor/
grep -rn --include=*.rs -E 'TypeId::of|is_cuda::<|backend_matches' vendor/   # fused gates
grep -rn --include=*.rs -E '^\s*if .*\{ /\* .* \*/ \}' crates/               # validations that validate nothing
```

`if let Some(..) = fused(..) { fused } else { tensor_ops }` is the shape to
read hardest: the else arm is correct, so the only evidence that the fast path
engaged is a counter.

## The enumeration (2026-09-27)

Every fallback site in the repo, by crate. **Class is the state after this
ADR**; `(was SILENT)` marks the ones this commit changed.

| # | site | class |
| --- | --- | --- |
| **model / trainer — paths the trainer runs** | | |
| 1 | `train/src/lib.rs` (save block in `train_loop`: `let _ = save_ckpt`) + an unconditional `ckpt saved` line | **LOUD** (was SILENT) |
| 2 | `train/src/lib.rs` (ngram sidecar: `if let Ok(f) = File::create` + three `let _ =` writes, all under the same "saved" line) | **LOUD** (was SILENT) |
| 3 | `train/src/lib.rs` (final `let _ = save_ckpt` after the loop) | **LOUD** (was SILENT) |
| 4 | `train/src/lib.rs` (`quant_format`): an unknown `--quant` string returned `QuantFormat::Fp32` | **LOUD** (was SILENT) |
| 5 | `dormouse-train/src/optim.rs:214` `momentum_cuda` → `false` → tensor path (head-wise Q/K Muon; `qk_heads` is resolved in EVERY run) | **COUNTED** (was SILENT) |
| 6 | `dormouse-train/src/optim.rs:260` `finalize_cuda` → `false` → tensor path | **COUNTED** (was SILENT) |
| 7 | `dormouse-train/src/stress.rs:130` `grad_norm` → `unwrap_or(0.0)` when NO gradient was found; the loop reads 0.0 as "the firewall fired" | **COUNTED** (was SILENT; now NaN + the loop's test) |
| 8 | `dormouse-train/src/stress.rs:55,70` `partial_cmp(..).unwrap_or(Equal)` sorted NaN into the median and p99.9 | **COUNTED** (was SILENT; now refused + counted) |
| 9 | `core/src/model.rs` (`finite_scan`): `unwrap_or_default()` on a readback → `checked=0, bad=0` → "checkpoint is clean" for weights it never read | **LOUD** (was SILENT) |
| 10 | `core/src/model.rs` (then `forward_bytes`, now `train/src/decode.rs`): `unwrap_or_else(|_| vec![0.0; v])` → all-zero logits, i.e. a confident "byte 0 forever" | **LOUD** (was SILENT) |
| 11 | `data/src/lib.rs` (`refill`): `r.read(..).unwrap_or(0)` — a read ERROR became an EOF, so a bad shard ended at byte 0 and the run trained on the next one | **LOUD** (was SILENT) |
| 12 | `data/src/lib.rs` (`read_bytes`) skipped a file it could not open / read, and returned whatever was left | **COUNTED** (was SILENT; now a stderr summary + a list) |
| 13 | `data/src/lib.rs` (`read_bytes` end) returned an empty vec when every file failed — a BPB for a corpus never read | **LOUD** (was SILENT) |
| 14 | `data/src/lib.rs` (`ensure_reader`): `.ok()` on every `File::open` and parquet builder, twice (first epoch + reshuffle) | **COUNTED** (was SILENT; now `open_source` names the shard on stderr) |
| **library — fused/accelerated paths** | | |
| 15 | `dormouse-rmsnorm/src/lib.rs` (`RMSNorm::forward`: `if let Some(out) = rmsnorm_cuda(..)`), else tensor ops. On the trainer's autodiff backend the fused kernel cannot take an autodiff tensor, so **it never engages** | **COUNTED** (was SILENT; `asked`/`skipped` at `fused.rs:16`) |
| 16 | `dormouse-rmsnorm/src/fused.rs` (`rmsnorm_cuda` bails) `None` on non-CUDA tensor / wrong dtype / empty | COUNTED (joins #15) |
| 17 | `vendor/dormouse-kda/src/lib.rs:551-583` `kda_fused_chunk_reported(..).into_option()` → `chunk_wy_forward` else-arm | **COUNTED** (the library already reports `Fused`/`Fallback` and keeps `dormouse_gdn2::fused_calls()`; only the LOG print was missing) — *off-limits crate, reported only* |
| 18 | `vendor/dormouse-gdn2/src/kernel/chunk_cube.rs:788`, `chunk_adjoint_cube.rs:411`, `fused_recurrent_cube.rs:91`, `module.rs:52`, `autodiff.rs:86,418` — `is_cuda::<B>()` is the bare-`TypeId` gate, correct only for a bare backend; from an autodiff wrapper it returns `None` | **COUNTED** via #17's seam — *off-limits crate* |
| 19 | `vendor/dormouse-gdn2/src/cuda_dispatch.rs:222` `backend_matches` → `Fallback::NotCuda` (named, counted) | COUNTED (the pattern to copy) |
| 20 | `vendor/dormouse-sct/src/lib.rs:66,101,160,344` + `qr_cuda.rs:59,84` — `is_cuda` on the QR path | SILENT — *dead crate on our path (dormouse-sct is not a dormouse dep); no fix taken* |
| 21 | `vendor/dormouse-spectral/src/gpu.rs:137,142` and `moe_fused.rs:80-92,1048-1051,1506-1517` — `try_into_primitive::<CB>().ok()?` / downcast `?` on a fused matmul | SILENT (no counter) — *report only, crate is mid-edit by others* |
| 22 | `vendor/dormouse-attnres/src/fused_attnres.rs:1127,1239`, `dormouse-bitnet/src/fwt_cuda.rs:343,394,429`, `dormouse-mhc/src/sinkhorn_cuda.rs:320,360`, `dormouse-rope/src/rope_cuda.rs:348,397`, `dormouse-situ/src/fused_situ.rs:227,264` — the raw `TypeId::of::<B>() == TypeId::of::<CudaBare>()` gate, the exact bug of evidence #2, still in these crates | SILENT — *off-limits crates, report only. `dormouse-bitnet` IS on our path (`weight_quant_ternary_nm` at `dormouse-spectral/src/lib.rs:480,494` and `quantize_tensor` at `:563,567`), but the `TypeId` gates are in its `fwt_cuda` module, not in the weight-quant fns the TSCT forward calls — so the claim for our path is narrower: the FWT kernels are dead, and whether the quant kernels are needs a 5-minute read of `fwt_cuda.rs`'s callers, not a guess.* |
| 23 | `vendor/dormouse-muon-plus/src/fused_kernels.rs:135,171` `-> bool` = "the fused kernel did not run" | COUNTED at the call site (#5/#6) |
| 24 | `vendor/dormouse-dspark/src/lib.rs:107` `None =>` head disabled by config | LOUD (config-driven, documented) |
| **config / data validation** | | |
| 25 | `dormouse-core/src/config/validation.rs:21` `if c.d_ffn % 2 != 0 { /* SwiGLU needs even */ }` — an `if` with an empty body: a validation that validates nothing, for a SwiGLU the expert FFN does not have (`gate_up: d→f`, `down: f→d`) | **deleted** (was SILENT: it looked like validation) |
| 26 | `dormouse-core/src/config/loader.rs:11-12` `read_to_string(p).ok()?` + `parse_str(&s).ok()` per candidate; a present-but-broken preset is skipped and the search continues | COUNTED (the miss reports every path tried) — *acceptable, but see proposals* |
| 27 | `data/src/lib.rs` (`collect_files`: `if let Ok(rd) = read_dir(root)`) → empty file list | LOUD downstream (`ByteStream::new` asserts; `read_bytes` now asserts) |
| 28 | `data/src/lib.rs` (`from_files`: `f.metadata().map(len).unwrap_or(0)`) — an unstattable file counts as 0 bytes in the size floor | SILENT (weakens a guard only) — *proposal* |
| 29 | `dormouse-core/src/param.rs:90-101` `bf16_compute` on a non-CUDA build → `forward_quant` | SILENT-by-cfg (documented: CPU is fp32-only) |
| **aux heads (monitoring-only)** | | |
| 30 | `dormouse-core/src/model.rs:263` `let ids = ids?` with weights > 0 → the whole aux term disappears | SILENT — *proposal: needs a return type change* |
| 31 | `dormouse-core/src/model.rs:274-286` `if let Some(tl) = teacher_latent` → JEPA term silently absent when no teacher latent arrived | SILENT — *proposal* |
| 32 | `dormouse-core/src/aux.rs:161` `if n == 0 { return zeros }` (sequence shorter than the draft window) → DSpark contributes nothing, no gradient to the head | SILENT — *proposal* |
| 33 | `train/src/lib.rs` (`aux_log.map(|a| a.try_into_scalar().ok())`) → the `aux=` figure vanishes from the log line | SILENT (log readout only) — *proposal* |
| **resume / checkpoint** | | |
| 34 | `train/src/lib.rs` (`load_ckpt(..).unwrap_or(0)`) — both candidates unusable → a fresh run | COUNTED (`load_ckpt` eprintlns each unusable candidate, with the non-finite-param reason) |
| 35 | `train/src/lib.rs` (every `load_ckpt_file` bail (`raw.len() < 24`, bad `from_bytes`, non-finite params) | LOUD (each prints the reason, then the `.prev` attempt) |
| 36 | `train/src/offload.rs` (`HostNgram::from_bytes`) `from_bytes` → `None` on a v1 Adam sidecar | LOUD (`train_loop` panics naming the file) |
| 37 | `train/src/lib.rs` (`load_model_weights`) → `None` on any read failure | LOUD (the CLI exits with "ckpt not found") |
| 38 | `train/src/lib.rs` (`init_pools`: `let _ = client.install_memory_pools(ExclusivePages)`) — a refused pool install means the long-run OOM profile changes | SILENT — *proposal (one println)* |
| 39 | `train/src/lib.rs` (`jepa_tgts.get(&bytes)?`) — a stale sidecar is an error, not a dropped aux | LOUD (ADR precedent, the right shape) |
| 40 | `core/src/loop_block.rs` (the depth/act-quant/kda-state/hash-key `None` arms) — `None` arms for depth override, act-quant, the KDA state, missing hash keys | LOUD/correct by construction for depth override, act-quant and the KDA state (the state asserts). **The hash-key `None` arm is NOT correct by construction, and this row classified it wrongly until 2026-09-28** — see the correction below. The "it means inference without hashed ids" reading is true for a decode path and false for a held-out *measurement*, and the same `None` served both |
| 41 | `train/src/lib.rs` (the eval forward, `lib.rs:1384-1390`) — the held-out eval passed `hashed_ids = None` unconditionally while the training step passed real keys, so on the in-VRAM Engram path the eval scored a network with no memory in it | was **SILENT**, and the most expensive instance of this class in the project's history. Fixed in `7adda92`; the eval line now prints `engram=<rows>/<arms>` counted over the eval's own forwards, which makes it COUNTED |

**Correction to row 40, and the generalisable lesson (2026-09-28).** The
`None` hash-key arm was justified here as "inference without hashed ids", and
that justification is what let the eval defect survive a whole audit: the rule
was written as a property of the *function* ("a model may be asked for logits
without n-gram keys") when the thing that matters is the property of the
*caller* ("is this caller measuring, or is it answering?"). A `None` that is
legitimate for a sampler is a silent wrong number for an eval, and the two
call sites differed in the one way that mattered. The test a fallback has to
pass is therefore not "does this arm have a defensible meaning" but "**can the
caller that chose it tell the reader which arm ran**" — which is what the
`engram=` counter is for, and what no `None` arm can do for itself.

Two instances of the same shape, for the record: `forward_bytes` (row 4 above,
the decode half) and the eval forward (row 41). Both passed `hashed_ids = None`
to one shared code path. Fixing one without noticing the other is how the second
survived the first fix's own commit message.

**Counts after this ADR: 41 sites — 17 LOUD, 11 COUNTED, 13 SILENT (of which
10 were fixed — 9 here, plus row 41 on 2026-09-28 — and 3 remain as
proposals).** Before it: 17 LOUD, 4 COUNTED, 19 SILENT. Row 41 is the eval
forward, added to the enumeration on 2026-09-28 after `7adda92` fixed it, and
row 40's hash-key arm has been reclassified out of "correct by construction" —
see the correction below, which is the more useful half of that edit.

## The top ten, by what they would have cost

1. `train_loop`'s save block — a checkpoint that never saved, printed as saved.
2. `ByteStream::refill` — a read error as an EOF: training on a
   corpus nobody chose.
3. `DormouseModel::finite_scan` — the corrupt-checkpoint guard passing on a readback that
   read nothing.
4. `model.rs::forward_bytes` — all-zero logits presented as a prediction.
   (Removed 2026-09-28 with the bigger defect it carried: the same function
   passed `hashed_ids = None`, so every decoded byte came from a network whose
   memory arm contributed literal zeros. See `train/src/decode.rs`.)
5. `optim.rs` (momentum_cuda / finalize_cuda) — the fused Muon kernel dead in every run, silently.
6. `stress.rs::grad_norm` — the firewall's only host-side signal, overloaded with a
   second meaning.
7. `dormouse_rmsnorm::RMSNorm::forward` — the fused norm kernel dead in every forward.
8. `quant_format` — `--quant fp8x` training a whole run in fp32.
9. `model.rs::aux_loss` — an aux term switching itself off.
10. `aux.rs::dspark_aux_loss` — DSpark with no window and no gradient, reported as a
    normal `aux=` value.

## What a fallback must be able to show (the seam rule)

A fused kernel that legitimately falls back (an autodiff tensor cannot be
handed a bare kernel; a device without the feature) stays — but it becomes
COUNTED by **one counter in the seam module, printed by the mechanism that
already prints the eval line**. As of this commit:

```
step  1000 EVAL ce=5.412 bpb=7.806 over 102400 B (fixed window) fused kda=812/812 norm=0/1620 muon_skipped=243/243
```

`kda=fwd/bwd` are launch counters the library already keeps
(`dormouse_gdn2::fused_calls()`), `norm=ran/asked` is new in
`dormouse-rmsnorm::fused::calls()`, and `muon_skipped` is the head-wise Muon
counter. `norm=0/1620` is today's truth: the fused norm kernel does not engage
on the trainer's backend. Before this line existed, that was unknowable
without a profiler.

The library already has the pattern to copy, in
`vendor/dormouse-fused/crates/dormouse-gdn2/src/cuda_dispatch.rs`: `Fused`/`Fallback`
says *which arm ran and why*, and `fused_calls()` counts launches. Use it; do
not add a third convention.

## Proposals NOT taken (and why)

1. **`dormouse_core::aux_loss` → `Result<Option<Tensor<1>>>`** (sites 30, 31).
   A missing teacher latent or missing targets is a *configuration* state
   (`--jepa-targets` on/off, an offline sidecar), and returning `Result` from
   a `pub fn` on `DormouseModel` changes the signature every caller in
   `dormouse-train/src/lib.rs` (4 call sites) and the CLI must match. Cheaper
   today: `fused_seam_counts()`-style counters for the aux arms, printed on the
   same line. Exact change when someone wants it: `fn aux_loss(...) ->
   Result<Option<Tensor<1>>, String>`; the trainer turns the `Err` into the
   same `Err(String)` `train_loop` already returns.
2. **`dspark_aux_loss` → `Option<Tensor<1>>`** (site 32). Same reason, and the
   window count (`n`) would have to travel out of the helper.
3. **`Source::read` → `io::Result<usize>` stays, but `read_bytes` should take
   a `Result`** rather than logging. Deferred: `read_bytes` has one caller
   (`src/bin/anchors.rs`) and a stderr summary plus the empty-sample assert
   already cover both failure modes honestly.
4. **A silent-fallback DETECTOR in CI** (grep-based, flagging any `unwrap_or`
   / `.ok()` on a fused path without an adjacent counter). Rejected as
   over-engineering: a regex cannot tell a fused gate from a config default,
   and a test that fails on a *new* `unwrap_or` trains contributors to add
   `#[allow]`s. The three tests added with this ADR pin the invariants instead:
   `a_shard_that_fails_to_read_is_not_a_silent_eof`,
   `an_empty_sample_is_loud_not_a_zero_byte_corpus`,
   `no_gradient_is_nan_not_a_clean_zero`.
5. **Un-stuffing the `TypeId` gates in the five off-limits crates** (site 22).
   Another agent is in each of them. The finding is recorded here so the next
   person in `dormouse-bitnet` learns that its kernels are dead on our backend.

## Checking new code against this ADR

- Does anything in your diff return a value that *looks like a working answer*
  when it failed? (a tensor, a bool, a length, a `0.0`, a default config)
- If it is a fused/accelerated arm: is there a counter, and is it printed on
  the eval line?
- If it is data or config: does the failure name the file, the key, and the
  escape?
- Does a `clone()` you believe is a copy actually copy, on this backend?
  (it does not, for a device buffer at two handles)
