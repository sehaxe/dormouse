# Host RAM as capacity in the dormouse port — offload survey and the Engram scale question

Date: 2026-10-18 · Read-only survey. No code changed. One new file (this one).
Scope: what to offload to the 64 GB host RAM, what 2025-2026 literature measures, what is worth
building for a 16 GB RTX 5060 Ti (PCIe 4.0 x16) training a 1.07B/57M-active byte-level MoE.

## TL;DR

1. **The one thing worth building first costs zero code: grow the host Engram tables.** The host
   update is `O(unique batch rows)`, not `O(table)` — verified in `offload.rs::momentum_update`,
   which only ever touches the batch's deduped rows — so 4M → 16M rows/order buys capacity for
   ~0% step time. Every n-gram-table paper we can verify says the input-side table pays
   (COUNT 2610.20456, SCONE 2502.01637, Engram 2601.07372, Qwen3.8-Next 2608.30320).
2. **The second is generalizing the already-proven activation-offload pattern** (`moe_offload`,
   `saved_tensors_hooks` inside `checkpoint(use_reentrant=False)`) from the MoE block to the
   loop/attention block — the loop runs ×3 and multiplies its activations by 3.
   Predicted +5-15% step for ≥25% peak VRAM [S1]; the port's own plan already predicted exactly
   this (`docs/research/2026-10-16-budget-savers.md` §5).
3. **Optimizer-state offload (ZeRO-Offload class) is the wrong lever at 1.07B.** Its published wins
   are at 13B-120B dense where optimizer state dominates; at 1.07B the state is small, and the port
   already has the cheaper option (bnb8 paged, +180 ms measured). Reject.
4. **NVMe third tier: reject.** No single-card result justifies it at our scale.
5. **Flag hazard found:** the port sizes host tables by `engram_rows` = **rows PER ORDER**, and the
   read masks with `& (slots-1)` → the value is rounded up to a power of two. "48M rows" is not a
   legal port rung (it rounds to 67M/order = 51 GB). Every GB figure below is given in both units.

## 0. ID verification (and one correction to the brief)

Verified via `export.arxiv.org/api/query` (the arxiv MCP 429'd/timed out on 20 of 22 calls after the
first two; the direct API returned titles+abstracts for every ID, which is how each claim below was
checked — abstracts, not titles).

- **The brief's `2101.04818` is NOT ZeRO-Offload.** The API returns *"Improved Hierarchical Clustering
  on Massive Datasets with Broad Guarantees"* (Hajiaghayi & Knittel, 2021-01-13). The real
  **ZeRO-Offload is `2101.06840`** (verified: *"ZeRO-Offload: Democratizing Billion-Scale Model
  Training"*, 2021-01-18). Anything citing 2101.04818 as ZeRO-Offload is citing a clustering paper.
- `2104.07857` = ZeRO-Infinity ✓ (verified).
- All other IDs below were verified by direct API fetch of title+abstract on 2026-10-18.

## 1. Ranked table

Evidence tags, the repo's own convention: **[A]** = measured in this repo; **[S1]** = measured in
the paper but no ≤1B single-card study exists (transfer unproven); **[S2]** = theory/analogy only.

| # | Technique | Source (verified ID / file) | Mechanism | Measured overhead | VRAM / RAM effect | Fit for us | LOC |
|---|---|---|---|---|---|---|---|
| 1 | **Grow host Engram tables 4M → 16M rows/order** | 2610.20456, 2502.01637, 2601.07372, 2608.30320; `offload.rs::momentum_update`, `port/.../model/engram.py:27-47` | rows live in plain CPU fp32; only the batch's unique rows cross the bus | **0% step [A]** — arithmetic is `O(batch rows)`, flat in table size | +RAM linear (3.07→12.29 GB), VRAM flat (`to_host` deletes the parameter) | **BEST** — 0 LOC, best-evidenced lever in the repo | **0** |
| 2 | **Extend activation offload from MoE to loop/attention block** | `port/.../model/experts.py:713-741` (the shipped pattern); `budget-savers.md` §5; 2609.14306, 2408.10013 | `saved_tensors_hooks` **inside** `checkpoint(use_reentrant=False)`; pageable D2H at save, H2D at the backward edge | **predicted +5-15% step [S1]**; SSDTrain reports 47% peak cut with "negligible" overhead | −25% peak VRAM [S1] → buys a batch/seq rung | **good** — pattern proven bit-exact here already | 50-150 |
| 3 | **Pinned + side-stream prefetch of the next batch's rows** | 2608.30320 (playbook item 1), 2604.05091 (double-buffered streams); `docs/decisions/2026-10-15-1b-readiness-plan.md` item 7; `engram.py:43-47` (explicitly absent today) | pin the buffers, `cudaMemcpyAsync` the next batch's rows on a side stream, overlapped with compute | ~0 today: ~600 KB/step at b3·s512, ~1.5 MB/step at b8 | 0 VRAM; makes rung 1's big tables latency-free | good, cheap | 30-60 |
| 4 | Optimizer-state offload (ZeRO-Offload class) for backbone groups | 2101.06840, 2410.21316, 2509.02480, 2503.01890, 2406.10728 | CPU Adam step, pinned grad/param staging, interleaved offload | 2.5× faster than naive offload [S1]; port already pays **+180 ms** for bnb8 paged [A] | frees optimizer state | **poor** at 1.07B — state is small | 200+ |
| 5 | Muon/NS state: **quantize, don't offload** | 2509.23106 (8-bit Muon, −62% state), 2509.14562, 2609.04577 | blockwise 8-bit momentum | parity val loss / benchmarks [S1] | −62% Muon state | medium | ~100 |
| 6 | MoE expert weights/activations offload for training | 2211.15841 (MegaBlocks, +40% over Tutel), 2206.03382, 2201.05596, 2603.07685 | block-sparse dispatch; Megatron Core "fine-grained recomputation, offloading" | multi-GPU numbers only [S2] | — | **already done here** (SL1 `moe_offload`) | 0 |
| 7 | COUNT-style count tables (no learned params at all) | 2610.20456 | precomputed corpus n-gram counts on host weight the model's own unembeddings; nothing is trained | 0% step [S1] | +RAM, **zero optimizer state** | interesting 4th, needs a corpus-counting pipeline | medium |
| 8 | NVMe as a third tier | 2104.07857, 2509.02480, 2408.10013 | params/optimizer/activations on NVMe | throughput loss at scale [S2] | — | **REJECT** (§4) | n/a |

## 2. Top-3 expanded

### 2.1 Grow the host Engram tables (0 LOC, first A/B)

**Exact pattern (what moves when).** The port already does this: `Engram.to_host()`
(`engram.py:332-345`) copies the built table to plain CPU fp32 and `del self.tables` — the parameter
leaves `parameters()`/`state_dict()`, so the optimizer, grad clip and checkpoint half never carry
it. Per forward, `rows_for_batch` dedupes the batch's `[b,t,orders]` keys, gathers `[unique, dim]`
from RAM, H2Ds it as a grad-carrying leaf; per step the row grads D2H and `momentum_update` applies
Nesterov (with Sinkhorn every 4th) **to those rows only**. Nothing else moves. The table never
crosses the bus as a whole.

**Why the cost curve is flat.** `offload.rs::momentum_update` iterates `indices` (the deduped batch
rows) — the doc comment states it plainly: *"the work stays O(batch rows) — the full table is never
touched."* So the per-step CPU arithmetic, the D2H bytes and the Sinkhorn work are all independent
of slot count. Scaling rows costs RAM and gather locality, not time.

**Failure modes.**
- **Unit hazard (verified, and it bites):** `engram_rows` is *rows per order*, rounded **up to a
  power of two** (the read masks `& (slots-1)`). `--engram-slots 48000000`-style configs are not
  legal port rungs; 48M rounds to 67.1M/order = 51.5 GB and dies on the cgroup. Use 4M (2²²),
  16M (2²⁴), 32M (2²⁵).
- **Locality (SPECULATION, unmeasured):** a 12 GB table gathered at random 128-B rows has worse
  cache/TLB behaviour than a 3 GB one. The arithmetic is flat; the gather may not be. Measure it —
  this is exactly what `ms/step` in the A/B is for.
- **RAM ceiling:** `MemoryMax=40G` cgroup + `ram-guard` at 8 GB avail. 12.29 GB fits with the
  desktop live; 24.6 GB does not.

**A/B design.** Arms: `engram_rows = 4_000_000` (control) vs `16_000_000`, `engram_ram=true`,
identical flags otherwise, `pure CE`, one batch size, matched eval window (the byte count on the
eval line is the authority — §2.6). 3 seeds. 2k steps. **Metrics:** ms/step, peak
`torch.cuda.max_memory_allocated()`, held-out BPB. **Prediction:** ms/step within noise (arithmetic
is flat, [A]); peak VRAM flat (`to_host` already removed the parameter); BPB improves [S1].
**Kill:** BPB inside the control's 3-seed spread → the table is saturated at 4M and rung 1 is dead
(2601.16531 found exactly this shape: a collision-free hot tier did **not** consistently improve
loss, and collisions behave as implicit regularisation). This is a 2-arm, 6-run experiment with a
near-zero time cost — the cheapest real decision on the board.

### 2.2 Generalize activation offload to the loop/attention block

**Exact pattern.** The shipped `moe_offload` (`experts.py:713-741`) is the template: wrap the pass in
`checkpoint(offloaded, use_reentrant=False)` where `offloaded()` runs the body inside
`saved_tensors_hooks(_pack_activation, _unpack_activation)`. Pack copies the saved tensor to a plain
**pageable** CPU tensor; unpack returns a copy **on its original device** at the backward edge. The
loop block (weight-shared ×3) is the natural next target: it multiplies activation memory by 3.

**Failure modes (all measured here for the MoE version, `experts.py:99-122`).**
1. **Hooks must be INSIDE the checkpointed function.** Non-reentrant `checkpoint` is itself a
   `saved_tensors_hooks` context and only the innermost pair applies — a `with` around the
   `checkpoint()` call is *silently inert*. Keep the pack-hook spy test.
2. **Unpack must restore the original device**, else the hook contract breaks.
3. **The pack copy is synchronous** — that sync is the +5-15%. Pinned buffers + a side stream are
   the upgrade (that is technique 3).
4. **Cache-resident operands get copied for nothing** — for the MoE banks that is ~1.25 GB/pass of
   traffic for zero reclaimed bytes (`experts.py` `ponytail:` note). Identity-check against the
   cache in the pack hook before generalizing.

**A/B design.** Arm = loop-block offload on; control = off; same batch/seq; 3 seeds; 2k steps.
Metrics: peak VRAM, ms/step, BPB at matched steps. **Gate:** the offload counter must be non-zero
(ADR-0019 — a zero means the arm did not run). **Kill:** BPB outside the control's 3-seed spread.
The port's own pre-registered prediction for the MoE version is peak −25%, step +5-15%, BPB inside
the spread (`budget-savers.md` §5, marked S1 there and still S1 here: 2609.14306 proves loss-exactness
but at 120B-667B on many GPUs).

### 2.3 Pinned + side-stream prefetch

The port's host path deliberately has *no* prefetch and *no* pinned memory (`engram.py:43-47`), and
today's copy is small (~600 KB/step at b3, ~1.5 MB/step at b8). At 4M-16M rows it stays small; it
becomes worth doing only once rung 1 goes bigger and the H2D gather is on the critical path. Pattern:
pin the next batch's `[unique, dim]` buffer, `cudaMemcpyAsync` it on a side stream, join before the
forward; keep the grad D2H on the same side stream so it never blocks the default stream.
MegaTrain (2604.05091) is the reference for the double-buffered form (it reaches 1.84× DeepSpeed
ZeRO-3 CPU-offload throughput by overlapping prefetch/compute/gradient-offload across streams).
**A/B:** ms/step only (this cannot change BPB); expect a delta only if rung 1 landed.

## 3. The Engram scaling question — concrete numbers

**Row math (verified from code).** `dim = 32` → **128 B/row** for the table, **256 B/row** with the
single momentum buffer (Adam's m+v would be 384 B). Layout is `[orders, slots, dim]` with
`orders = [2,3,4]`, so **total rows = 3 × rows/order**. Two independent repo anchors confirm the
256 B/row figure: 48M total rows → 12.29 GB (§2.4's stated 12.3 GB) and 8M total rows → 2.05 GB
(§2.4's "≈2.1 GB").

| rows/order | total rows | table only (128 B) | + momentum (256 B) | port legal? | repo anchor |
|---|---|---|---|---|---|
| 4M (2²²) | 12M | 1.54 GB | **3.07 GB** | yes | the brief's "4M/order = 1.6 GB" is the table-only column |
| 16M (2²⁴) | 48M | 6.14 GB | **12.29 GB** | yes | the burn-era real18 production = 12.3 GB |
| 48M (rounds to 2²⁶=67M) | 144M (201M) | 18.4 GB (25.7) | **36.9 GB (51.5)** | **no** — rounds up | the ~37 GB resume/freeze footprint |
| 32M (2²⁵) | 96M | 12.3 GB | **24.58 GB** | yes | the top rung that fits `MemoryMax=40G` |

**Update cost per step.** Flat in rows, by construction (§2.1). The three real components, all
`O(unique batch rows)`: gather `unique×32` f32 from RAM; D2H the row grads (`unique×128 B` ≈ 600 KB
at b3·s512, 1.5 MB at b8·s512); Nesterov over `unique×32` (plus Sinkhorn's 2 row+col L1 passes over
the same block every 4th step). **The burn-era `6.7-8.3 s/step` figure at 48M rows / batch 10 / s512**
(AGENTS §3.3, 2026-09-21, post-EMA-fix) **cannot be attributed to the Engram** — it is a different,
older config on a run whose attention backward did not execute; do not use it as an Engram cost
curve. The port's own measured step is ~1.2 s at b8·s512 (the owner's baseline) with no such split.
The `+2.3%/step` host-update figure quoted in `docs/research/2026-09-21-per-gb-sota.md:96` is a
**secondary citation — I could not locate its primary measurement**, so treat it as unverified.

**Recommended first A/B:** `engram_rows` 4M vs 16M rows/order (3.07 vs 12.29 GB), same steps, same
batch/seq, pure CE, 3 seeds, BPB as the decision metric with ms/step and peak GiB recorded.
Rationale: 16M/order is exactly the burn-era production size, fits the 40 G cgroup with the desktop
live, and is the largest rung where the only changed variable is table capacity.

## 4. Explicit rejections

- **ZeRO-Offload-style optimizer offload for the backbone (2101.06840 and successors).** Published
  wins are 13B on one V100 / 100B+ on an H200 with 1.5 TB host RAM — regimes where optimizer state
  dominates VRAM. At 1.07B/57M-active our state is small and the VRAM binding constraint is
  activations, not optimizer state. The port already carries the cheaper variant (bnb8 paged,
  **+180 ms measured [A]**). Not worth 200+ LOC.
- **Offloading the Muon/NS optimizer step to CPU.** No published work exists: a targeted arXiv query
  for Muon + offload/CPU returns only particle-physics muons. The reason is structural — Muon's only
  state is one momentum buffer, but its *step* is Newton-Schulz orthogonalisation, which is compute
  the GPU is good at and the CPU is not. The offloadable half is momentum only; the port already
  does exactly that on the host tables (Nesterov, **no NS**), which is why they are offloadable at
  all. For the backbone, quantize instead (2509.23106: 8-bit Muon, −62% state, parity loss [S1]).
- **NVMe as a third tier (2104.07857, 2509.02480, 2408.10013).** ZeRO-Infinity's wins are at
  trillion-parameter scale on DGX-2 nodes; MLP-Offload/SSDTrain add NVMe to *host-RAM-starved*
  multi-GPU setups. On a 64 GB box with a 12-25 GB table, host RAM is not starved and NVMe only adds
  a stall the PCIe path does not have. No single-card result supports it here.
- **MPHF / collision-free hot-tier Engram.** 2601.16531 tested it under a strict iso-parameter
  setup: a collision-free hot tier did **not** consistently improve validation loss, and collisions
  act as implicit regularisation. Do not build it.
- **Output-side n-gram/lookup tables.** Output vocab growth actively hurts small models
  (2501.16975, per the repo's own analysis). Keep it rejected.
- **MoE expert-weight offload for training.** Already covered by SL1 (`moe_offload`) for
  activations; the weight banks live in the VRAM cache either way and copying them buys nothing
  (`experts.py` `ponytail:` note). The multi-GPU systems (MegaBlocks/Tutel/Megatron Core) solve
  dispatch/communication problems we do not have on one card.

## 5. VERIFIED vs SPECULATION

**VERIFIED (source next to each claim).**
- Host table cost is `O(unique batch rows)`, flat in slot count — `offload.rs::momentum_update` +
  its doc comment; `momentum_update` iterates `indices` only. [A]
- 128 B/row table, 256 B/row with momentum, `total_rows = 3 × rows/order`, `dim=32` —
  `offload.rs` (`row_bytes`, `total_rows`, `HostNgram::new([slots,slots,slots], 32, …)`),
  `port/.../model/engram.py:158-177`, `config.py:364-369`. Cross-checked against AGENTS §2.4's
  12.3 GB @ 48M rows and 2.1 GB @ 8M rows. [A]
- `engram_rows` is **per order** and rounded up to a power of two; the read masks `& (slots-1)` —
  `config.py:364-367`, `engram.py::slots_for`. [A]
- `to_host()` deletes the parameter so the optimizer never sees the table —
  `engram.py:332-345`, `loop.py:454-459`. [A]
- The port's host path has no prefetch and no pinned memory — `engram.py:43-47`. [A]
- `moe_offload` = `saved_tensors_hooks` inside `checkpoint(use_reentrant=False)`; hooks outside are
  silently inert; unpack must restore the device; cache-resident banks are copied for nothing —
  `experts.py:99-122, 713-741`. [A]
- bnb8 paged optimizer costs +180 ms — brief + repo records. [A]
- Paper claims as stated: ZeRO-Offload 13B on one GPU / 40 TFlops/GPU (2101.06840); ZeRO-Infinity
  25 petaflops on 512 V100, trillion-param on one DGX-2 (2104.07857); MegaTrain 120B on one H200,
  1.84× ZeRO-3-CPU-offload (2604.05091); Deep Optimizer States 2.5× faster iterations (2410.21316);
  MLP-Offload 2.5× (2509.02480); AutoHete 1.32-1.91× (2503.01890); Flattening…MoE: dispatch peak
  −59.3%, vocab-projection peak −86.6%, offloaded optimizer step 2.05× faster (2609.14306);
  SSDTrain −47% activation peak (2408.10013); MemFerry 1.68× / 1.52× larger (2610.09657);
  SCALE 35-45% memory (2506.16659); 8-bit Muon −62% state (2509.23106); COUNT = "as much as 57%
  more compute" at fixed budget (2610.20456); SCONE 1B+tables > 1.9B dense (2502.01637); Engram-27B
  iso-param > MoE (2601.07372); Engram-Nine: collision-free hot tier does not consistently improve
  (2601.16531); Qwen3.8-Next 51B n-gram params off-accelerator (2608.30320); MegaBlocks +40% over
  Tutel (2211.15841); Two Bits 203 GB → 51 GB DeepSeek table (2610.21810). [S1/S2 in the sense
  that they are the papers' own numbers, not reproduced here]
- **ZeRO-Offload is 2101.06840, not 2101.04818** — direct API fetch of both IDs. [A]

**SPECULATION (my inference, marked as such).**
- *SPECULATION:* the BPB gain from 4M → 16M rows/order transfers to our 1.07B byte-level model at
  our step budget. The direction is well-evidenced (COUNT/SCONE/Engram/Over-Tokenized all agree
  input-side tables pay), but no ≤1B single-card study exists, and 2601.16531 shows the curve can be
  flat. This is precisely what the §3 A/B decides — no number is assumed.
- *SPECULATION:* the +5-15% step cost of generalizing activation offload to the loop block. It is the
  port's own pre-registered prediction for the MoE version, marked S1 there. No ≤1B single-card
  measurement exists.
- *SPECULATION:* the gather from a 12 GB table is slower per row than from a 3 GB table (cache/TLB).
  The arithmetic is provably flat; locality is not measured. ms/step in the A/B is the instrument.
- *SPECULATION:* `+2.3%/step` for the host update. Quoted second-hand in the repo; primary
  measurement not found.

## 6. Sources

Local (read directly, read-only): `crates/dormouse-train/src/offload.rs`, `crates/dormouse-train/src/lib.rs:1025-1050`,
`port/src/dormouse/model/engram.py`, `port/src/dormouse/model/experts.py`, `port/src/dormouse/train/loop.py`,
`port/src/dormouse/config.py`, `AGENTS.md` §2.4/§3.3/§3.5, `docs/research/2026-10-16-budget-savers.md` §5,
`docs/research/2026-09-21-per-gb-sota.md`, `docs/research/2026-10-15-1b-tech-verification.md`,
`docs/decisions/2026-10-15-1b-readiness-plan.md`, `docs/protocols/AB-PROTOCOL.md`, `docs/papers/engram.md`.

arXiv (each title+abstract fetched from `export.arxiv.org/api/query` on 2026-10-18):
2101.06840 (ZeRO-Offload) · 2104.07857 (ZeRO-Infinity) · 2410.21316 (Deep Optimizer States) ·
2406.10728 (I/O patterns of offloaded optimizers) · 2604.05091 (MegaTrain) · 2509.02480 (MLP-Offload) ·
2503.01890 (AutoHete) · 2609.04577 (Optimizer Memory Schedules) · 2506.16659 (SCALE) ·
2609.14306 (Flattening Every Memory Peak in MoE Training) · 2408.10013 (SSDTrain) · 2510.05186 (OptPipe) ·
2610.09657 (MemFerry) · 2211.15841 (MegaBlocks) · 2201.05596 (DeepSpeed-MoE) · 2206.03382 (Tutel) ·
2603.07685 (Megatron Core MoE) · 2509.23106 (8-bit Muon) · 2509.14562 (LiMuon) ·
2610.20456 (COUNT) · 2502.01637 (SCONE) · 2601.07372 (Engram) · 2601.16531 (Engram-Nine) ·
2606.08347 (TN-gram) · 2610.21810 (N-Gram Memories at Two Bits) · 2605.20948 (Memory Grafting) ·
2609.35578 (FactorEngram) · 2609.03426 (Lngram v2) · 2608.30320 (Qwen3.8-Next).

## Open questions

1. Does BPB actually move from 4M → 16M rows/order at our scale, or is the curve already flat
   (the 2601.16531 shape)? → the §3 A/B.
2. Is the 12 GB gather locality cost measurable in ms/step? → same A/B, ms/step column.
3. Does the loop block's activation peak actually dominate at b1 (the premise of technique 2)?
   Needs one `torch.cuda.max_memory_allocated` trace at b1 before spending the 50-150 LOC.
4. Where is the primary measurement behind the `+2.3%/step` host-update figure?
