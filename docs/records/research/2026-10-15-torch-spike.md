# PyTorch feasibility spike — dormouse at the b1 shape (repo copy)

**2026-10-15.** This is the verbatim copy of the spike report (the original
lives in /tmp/opencode/dormouse-torch-spike/, which dies on reboot), filed as
the substrate-decision evidence per §1.4: torch 2.13.0+cu129 on CPython
3.12.15 (uv), venv-scoped, measured on the 5060 Ti with the pre-flight gate
before each of the 13 GPU runs. Companion code = the spike dir; not committed
(think of it as the port's seed only if the decision lands Option B).

# PyTorch feasibility spike — dormouse at the b1 shape

**2026-10-15.** Machine: RTX 5060 Ti 16 GB + 64 GB RAM, linux, no sudo.
Spike dir: `/tmp/opencode/dormouse-torch-spike/`. Nothing inside
`/home/sehaxe/dormouse` was modified; its docs were read for grounding.

The owner's decision criteria, in order: **(1) less code / easier to maintain,
(2) memory footprint, (3) training speed.** This document is measurement, not
argument. Every number below carries its shape, dtype, warmup and rep count, and
the raw JSON is in `results/`.

---

## 0. Pre-flight, as required by the machine rules

Every GPU run in this spike ran this check first (`common.preflight()`, printed
as `[preflight]` in every log):

```
pgrep -ax train  -> ''            (empty, rc=1)
nvidia-smi memory.used -> 286-289 MiB   (< 1 GB; the 286 MiB is Xorg + sddm)
```

All 13 GPU runs (3 benches + 1 profiler, plus reruns after logic fixes) reported
`ok=True` at 286-289 MiB before starting, and no `train` process ever existed.
**One GPU process at a time; never two.** No run was started while the card was
busy, so no result is a contended reading.

Logs (persistent, under the spike dir, never `/tmp` directly):
`results/bench_a.log`, `results/bench_b.log`, `results/bench_c.log`,
`results/profile.log`, `results/versions.txt`, `results/*.json`.

---

## 1. Environment

| item | version | note |
|---|---|---|
| system `python3` | 3.14.8 | **not used** |
| uv | `/usr/bin/uv` (already installed) | used for the interpreter and the venv, per the owner's rule |
| CPython | **3.12.15** | `uv python install 3.12` inside the spike dir → `~/.local/share/uv/python/` |
| venv | `.venv` under the spike dir | `uv venv --python 3.12 .venv` |
| **torch** | **2.13.0+cu129** | `uv pip install torch --index-url https://download.pytorch.org/whl/cu129` |
| torch.version.cuda | 12.9 | |
| cuDNN | 92000 | |
| triton | 3.7.1 | pulled by the torch wheel; used by `torch.compile` |
| numpy | 2.5.4 | |
| pytest / ruff | 9.1.1 / 0.17.0 | the required skeleton |
| nvidia-* wheels | cublas 12.9.1.4, cudnn 9.20.0.48, nccl 2.29.7, cusparselt 0.8.1, … | full list: `results/versions.txt` |
| driver / CUDA UMD | 615.78.08 / 13.4 | |
| GPU | NVIDIA GeForce RTX 5060 Ti, capability **(12, 0)** = sm_120, 15.56 GiB | |

**Verdict on item 1: the CUDA wheel chain worked with no sudo, no system change,
and no compiler toolchain beyond what the wheels ship.**

```
torch.cuda.is_available()      -> True
torch.cuda.get_arch_list()     -> ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']
bf16 randn(512,512) @ self     -> ok
```

Two notes that matter for the decision:

- **cp314 was not attempted.** The task authorised the `uv python install 3.12`
  route and that is the route taken; whether torch has cp314 wheels today is
  untested here.
- `sm_120` is in the arch list, i.e. the cu129 wheel carries sm_120 kernels —
  no PTX-JIT fallback and no "unsupported architecture" surprise. Everything
  below ran at full speed with no warning.

Cost of the environment: one `uv pip install`, ~2 GB of wheels, ~5 minutes
including the interpreter download. This is a **decision datum in itself**: the
repo currently needs a vendored cubecl fork and a pinned `CUDARC_CUDA_VERSION`
to build at all (`AGENTS.md` §2.5); the torch side needed one command.

---

## 2. Microbench A — plain bf16 GEMM rates at the b1 GEMM sizes

`bench_a_gemm.py`. M = 16384 (batch 16 × seq 1024), bf16 in / fp32 accumulate
(cuBLAS default) / bf16 out, CUDA-event time, **L2 flushed before every rep**,
warmup 5, **median of 20**. Same contract as burn's measured 44.06 TFLOP/s arm.

| # of this GEMM per token | K × N | median ms | TFLOP/s | vs burn's 44.06 ceiling |
|---|---|---|---|---|
| 12 | 2048 × 2048 | 2.8074 | **48.96** | 111 % |
| 4 | 2048 × 64 | 0.4277 | **10.04** | 23 % |
| 4 | 64 × 5461 | 0.8171 | **14.02** | 32 % |
| 2 | 2048 × 1024 | 1.4588 | **47.11** | 107 % |
| 1 | 2048 × 256 | 0.4306 | **39.90** | 91 % |

**Combined realistic forward rate (FLOP-weighted harmonic mean over the b1
inventory): 44.44 TFLOP/s = 100.9 % of burn's measured 44.06 TFLOP/s ceiling.**

The weight column is the honest part: it is *how many of that GEMM a single
token runs*, recomputed from the plan's §2.2 shape, not chosen to flatter:

```
 4 KDA projections × 2 iterations                        =  8 × [2048→2048]
 + out_proj × 2 + engram dense half × 2                   =  4 × [2048→2048]
 + 2 active experts × 2 iterations × ({2048→64},{64→5461}) = 4 each
 + router × 2 iterations                                   =  2 × [2048→1024]
 + head (once, after the loop)                             =  1 × [2048→256]
 → 113.9 MFLOP/token forward, 341.8 MFLOP/token train
```

That 341.8 MFLOP/token reproduces the plan's own §2.2 arithmetic (6 × 57 M
active params = 342 MFLOP) from the GEMM inventory, which is the cross-check
that the weights are right.

**What this says and does not say.** Both substrates run on the same silicon and
both get cuBLAS-class bf16 on the big shapes; torch's 48.96 vs burn's 44.06 is
**one number, ±10 %, and is not the decision**. The interesting column is the
middle one: the two *expert* GEMMs run at 10-14 TFLOP/s because their N is 64
and their K is 64. They are 3.4 % of the b1 forward FLOPs, so the weighted rate
hides them — but at an effective 20 TFLOP/s step they are where the launch and
tile inefficiency would show if anything did. Bench B measures that directly.

---

## 3. Microbench B — topk-2 MoE, E = 1024, [2048→64]→[64→5461] per expert

`bench_b_moe.py`. N = 16384 tokens (batch 16 × seq 1024), bf16, same expert
parameter count in both arms: **492,109,824 expert params**. Router = softmax
top-2 over 1024, renormalised. Warmup 3; **median of 20 (grouped) / 3 (dense)**.

Measured routing occupancy of the (untrained, random-logit) router:
mean 32.0 rows/expert, **max 51**, 0 empty experts, 32768 expert-rows.

### (a) gather → cap-padded grouped bmm → weighted scatter

| arm | median ms | tok/s | VRAM peak (alloc) | padded TFLOP/s |
|---|---|---|---|---|
| forward only, cap 64 | **21.406** | **765,395** | 3.06 GiB | 2.94 |
| forward + backward, cap 64 | **52.587** | **311,560** | 3.26 GiB | 3.60 |

### Capacity ladder (the padding tax is measurable, and it is the whole story)

| cap | capacity factor | rows dropped | padded rows | fwd median ms | tok/s |
|---|---|---|---|---|---|
| 40 | 1.25 | **0.56 %** | 1.25× | **19.52** | 839,260 |
| 48 | 1.50 | 0.01 % | 1.50× | 20.19 | 811,407 |
| 64 | 2.00 | 0.00 % | 2.00× | 21.40 | 765,769 |
| 128 | 4.00 | 0.00 % | 4.00× | 26.23 | 624,590 |

At cf 1.25 you lose 0.56 % of the expert-rows and reclaim 9 % of the layer time;
at cf 4 you gain nothing and pay 23 %. **The plan's chosen cap should not be a
guess: 48 costs 0.01 % and is 6 % faster than 64.**

### (b) dense-loop — all 1024 experts on every token, no capacity, no router

| arm | median ms | tok/s | VRAM peak | launches/step |
|---|---|---|---|---|
| forward only | **3,551.9** | **4,613** | 1.83 GiB | 2,048 |
| forward + backward | **17,842.7** | **918** | 3.08 GiB | 6,144 |

The dense arm uses `torch.utils.checkpoint` per expert — without it the loop
would save 1024 × [16384, 5461] activations (≈ 180 GB) and could not run at all.
The grouped arm needs no such trick.

### The decisive number

> **grouped is 339× the dense-loop at fwd+bwd and 166× at fwd, at the same
> total parameter count.**

This is the plan's "**the single load-bearing wall of the 1B goal**" (§2.4 item 1)
measured in the substrate the owner is considering moving to. It is also why
"torch is only as fast as burn's ceiling" would be the wrong reading of bench A:
the dense MoE burn runs today (`loop_block.rs:982-997`) is ~18× the FLOPs *and*
1024× the launches of the sparse path, and no framework makes that free.

**Honest caveats on this bench.**
1. The router is untrained and uniform-ish. A trained router is more skewed, so
   at cf 1.25 the drop rate will be worse than 0.56 % and the cap choice becomes
   a real quality/compute trade, not a free win. Not measured here.
2. The grouped path pads, so half its expert-row work is zeros at cap 64
   (`useful_flops_ratio` = 0.5). The 3.6 padded TFLOP/s is a third of what the
   same kernels do unpadded (bench A: 10-14). This is the torch-side tuning
   surface and it is real work, not free.
3. The dense arm is a Python loop over 1024 experts; a C++/fused dense MoE would
   be faster. But burn's dense MoE is also a loop (`for e in 0..n_experts`), so
   the comparison is like-for-like at the layer level.

---

## 4. Microbench C — "LoopBlock-lite" training step, batch 16 × seq 1024, bf16

`bench_c_loopblock.py`. Shape exactly as specified:

```
Embedding(256 → 2048), TIED to the head
→ 2 shared-weight iterations of:
     controller gates (sigmoid) + ATTENTION STAND-IN + 2-active-of-1024
     low-rank experts (grouped path from bench B, cap 64) + out_proj
→ RMSNorm → head → CE against byte targets
```

- **The attention is torch SDPA, causal, s1024 — a stand-in, not the model.**
  The real model's only attention arm is KDA (gated delta, linear attention) with
  kernels in the FLA library. SDPA is O(s²) and *more* expensive than KDA at
  s1024; it contributes ~8.4 of 117 MFLOP/token forward (7 %), so it neither
  flatters nor penalises the substrate materially. It is labelled in the code.
- **Embedding is tied.** Stated because the plan does not fix it at b1.
- **Optimizer is AdamW**, not Muon+, and its state is bf16 (torch allocates
  optimizer state in the parameter dtype). `baseline` = the model, 518.5 M
  params of which 492.1 M are the expert pairs.
- **Deviation from the plan's §2.2 table, declared:** the plan prices an expert
  as `2048·64 + 64·5461` and has no down-projection, but the real LoopBlock's
  expert has one (`down.forward`, `loop_block.rs:1037`, `f_ffn → d`). The spike
  folds **one shared** down-projection `[5461→2048]` (+11.2 M params) so the
  expert pair keeps the specified shape. If the real `down` is per-expert, the
  step costs more on both substrates.

Warmup 2 + warmup 3 (eager/compile), **median of 10**. GPU utilisation sampled
from `nvidia-smi` at 5 Hz in a thread across the timed window.

| config | ms/step | tok/s | peak VRAM (alloc) | peak VRAM (reserved) | GPU util mean | eff. train TFLOP/s* |
|---|---|---|---|---|---|---|
| **eager** | **357.3** | **45,861** | 6.93 GiB | 8.88 GiB | **99.9 %** (0 % of samples ≤5 %) | 15.7 |
| `torch.compile(dynamic=False)` | 298.0 | 54,988 | 7.66 GiB | — | — | 18.8 |
| **`torch.compile(dynamic=True)`** | **273.5** | **59,901** | — | — | — | **20.5** |
| forward only (no autograd) | 108.9 | 150,405 | — | — | — | — |

\* at the spike's own 341.8 MFLOP/token train inventory. At the plan's §2.2
figure of 0.40 GFLOP/token the same steps read **18.3 / 22.0 / 24.0** TFLOP/s.

**`torch.compile` did not fight.** First compile 4-13 s wall; speedup 1.20×
(`dynamic=False`) and 1.31× (`dynamic=True`). The recorded failure mode worth
knowing: with `dynamic=False` dynamo hit `config.recompile_limit (8)` on the
grouped MoE, because the kept-row count is data-dependent (capacity overflow):

```
W ... torch._dynamo hit config.recompile_limit (8)
W ... function: 'torch_dynamo_resume_in_moe_grouped_at_77' (moe.py:77)
W ... last reason: tensor '___stack0' size mismatch. expected 24223, actual 24404
```

`dynamic=True` fixes it and is the faster variant. **This is the one real
portability caveat in the whole speed story: the grouped MoE's data-dependent
shapes are a compile-time hazard that a hand-written kernel does not have.**

### Where the step's time goes (`profile_step.py`, 5 steps under `torch.profiler`)

The profiler's own total, 357.0 ms/step, **matches the uninstrumented 357.3 ms** —
the file is not inflated, so the shares are readable.

```
109 leaf kernels, 683 launches/step
gemm/attention       55.0 %
elementwise/memory   26.9 %
optimizer (AdamW)    14.9 %
other                 3.2 %
leaf kernels averaging <10 us: 47/109 -> 0.16 % of GPU time
```

**This is the direct answer to the burn side's most famous measurement.** At
9.2 M params burn measures **87 % GPU idle** and a launch-bound step
(`AGENTS.md` §3.1). At the b1 shape, in torch, the same card is at **99.9 %
utilisation**, 683 launches/step, and kernels under 10 µs account for
**0.16 %** of GPU time. The step is GEMM-bound and memory-movement-bound. There
is no launch overhead left to harvest and CUDA graphs (the plan's Phase 3.5) are
worth almost nothing here — which is a *finding*, not a disappointment.

---

## 5. Code-volume honesty table

### 5.1 The spike itself

| file | LOC | what it is |
|---|---|---|
| `moe.py` | 128 | topk-2 grouped MoE (gather/bmm/scatter), dense-loop, reference |
| `bench_c_loopblock.py` | 254 | the whole model + step + optimizers + both compile variants |
| `common.py` | 136 | pre-flight, L2-cold event timing, nvidia-smi sampler |
| `bench_b_moe.py` | 181 | the MoE bench and the capacity ladder |
| `bench_a_gemm.py` | 97 | the GEMM ladder |
| `profile_step.py` | 134 | kernel-level breakdown |
| `tests/test_spike.py` | 67 | 4 tests (grouped-vs-reference, gradient flow, cap drops, step trains) |
| `pyproject.toml` | 19 | uv/ruff/pytest skeleton, as the owner requires |
| **total** | **1,016** | of which the **model + training step is 382 LOC** |

ruff clean, pytest 4/4 green. `common.py` + `bench_*` are throwaway measurement
scaffolding — the 382 LOC is the only part a port would keep.

### 5.2 Full-port estimate vs what is on disk today

Ranges are ranges; the "basis" column says where each one comes from. I have not
faked precision: Python is ~3-8× denser than this Rust for the same job here
because pyarrow/numpy/torch/triton absorb the parts dormouse wrote by hand, and
the ranges overlap heavily.

| component | on disk today (Rust, LOC) | torch port estimate (Python) | basis / assumption |
|---|---|---|---|
| model (LoopBlock + experts + controller + arms) | 12,335 (`dormouse-core/src`) | **600–1,200** | the spike's model+MoE is 382 LOC and covers embed → 2 share-weight iters → controller → topk2 grouped experts → norm → tied head; the port adds Engram, KDA, Aux |
| trainer loop, timers, NaN firewall | 3,250 (`train/src/lib.rs`, which also holds eval/checkpoint/logging) | 300–700 | spike `step()` is 20 LOC; rest is logging/guards/cadence |
| config seam + validation | 575 (`cfg.rs`) | 150–350 | typed-config + resolved-snapshot validation is a rewrite, not a translation |
| **resume/checkpoint** | (in the 3,250 above) | 150–400 | `state_dict` + optimizer state + a config snapshot with the ADR-0005/0021 refusal rules; torch gives the serialisation, not the rules |
| **data pipeline** (byte stream, 64 MB ring, seeded shuffle, parquet, FNV n-gram hashes, `rewind`/`skip_bytes`) | 3,845 (`dormouse-data/src`) | 300–800 | pyarrow + numpy do the parsing; the FNV hashing, deterministic split and window arithmetic are ours |
| optimizer routing policy (Muon+ mix, path markers, startup validation) | 917 + 588 | 250–600 | torch 2.13 ships Muon; the *policy* (which params, validated against the live module tree) is ours |
| host-RAM hash-table arm (pinned prefetch, Nesterov+Sinkhorn, sidecar) | 441 (`offload.rs`) | 300–700 | **no torch primitive for this**; pinned memory + a second stream is a rewrite in either language |
| byte-patching (ByteFlow) | 453 + 809 | 150–400 | |
| SFT skeleton | 0 (spec in `post-training.md`) | 300–600 | loop + data mixing + eval hooks; not started on either side |
| eval / anchors (BPB, fixed window, `--fit`, 5-gram) | (in the 3,250 above) | 150–400 | |
| guards (stress protocol, determinism harness, CE ceiling, resume validation) | 174 + `tools/` | 200–500 | **the ADR-0011/0019 LOUD/COUNTED/SILENT discipline has no torch equivalent — it is a rewrite and it is the cheapest thing to lose** |
| KDA arm (the attention) | 1,413 (`burn-kda`) + 2,666 (`burn-gdn2`) | **0–400 if FLA works on sm_120, else 1,500–3,000** | **the single largest uncertainty in the whole table** |
| spectral / Newton–Schulz retraction, factored quant | 3,073+ (`burn-spectral`) | 100–250 | NS is ~15 lines of torch; the retraction hook ~50 |
| custom fused kernels that a port deletes outright (rmsnorm, situ, rope, attnres, mhc, bitnet-fwt, muon-plus-fused, moe_fused) | ~8,000 | **0** | cuBLAS/SDPA/torch.compile/inductor replace them |
| **total owned code** | **23,978** (`crates/*/src`) **+ 55,269** (`vendor/dormouse-fused`, src+tests) | **2,750–6,400** | |
| tests | 7,159 (`crates/*/tests`) + 20,246 (vendor) | 400–1,200 | spike: 67 LOC / 4 tests |

Consistency notes, reported rather than smoothed over:

- `ls vendor/dormouse-fused/crates/` shows **20 crate directories**;
  `AGENTS.md` says **28 crates** in the same sentence. One of the two is stale.
- `vendor/cubecl-fix` and `vendor/cubek-fix` contain **0 `.rs` files** under
  `find` — they are vendored upstream trees with a different layout, or the
  vendoring moved. `vendor/cubecl-ir` is 18,438 lines. The vendored-patch
  surface a port would drop is real but I could not measure it cleanly; treat
  the row as "≥ 18,000 lines of patched upstream" and no better.

### 5.3 The burn path's REMAINING work (the plan's Phase 1 + Phase 3)

This is what Option A still owes before the b1 run, against what already exists:

| Phase 3 item (plan §3) | remaining LOC (Rust) | what is already there |
|---|---|---|
| 1. topk-dispatch MoE + the `moe=[shares,H,dead]` utilisation gate | **250–500** + 150–300 tests | `moe.rs` is 1,077 LOC with `topk_blend`/`topk_indices`/`lb_aux`; `loop_block.rs:1014-1037` still runs the dense blend |
| 2. the bf16 step (storage+masters, mixed-dtype rules, Muon moments bf16+SR) | **300–700** | bf16 matmul is proven at 44.06 TFLOP/s; the flags exist; masters and the counted cast boundaries do not |
| 3. KDA fused-backward repair | **200–600** | the defect is localised ("disagreement 4.5e-2..2.6e-1 on q/k/v, post-op params agree at 2e-7"); `burn-gdn2` is 751+679+527+… LOC |
| 4. tokens-per-step operating point | 50–150 | config only; needs a VRAM ladder |
| 5. CUDA graph capture, one stage | **300–800** | the cubecl client API exists and `graph.rs` is 5/5 green here; the plumbing into `train_loop` does not |
| 6. retraction at scale + the re-derived `max_ortho` threshold | 100–300 | wired; the 1e-3 threshold is known to mis-latch at 1B |
| 7. Engram prefetch overlap (pinned + second stream) | 150–400 | the host path works; there is no overlap |
| Phase 1 debt (0.22 workaround kit, pool-cost measurement, headwise-Muon gates) | 150–400 | drafted uncommitted in the working tree |
| **total remaining** | **1,500–3,850 LOC** | **on top of 23,978 + 55,269 LOC already on disk** |

**Read both tables together.** Option A is cheaper *now* (1.5-3.9 k LOC, in a
codebase whose idioms you know, with the instrument layer and the A/B discipline
already in place) and ends holding ~79 k LOC of Rust. Option B is 2.8-6.4 k LOC
of new Python, ends holding a fraction of that, and starts by writing the thing
that took the longest to get right (the instrument layer, the guards, ADR
discipline) from scratch.

---

## 6. Verdict

### Criterion 1 — less code / easier to maintain

**Torch wins on owned code, by roughly 10–20×, and the win is real but is not
free.** A port lands at ~2,750–6,400 LOC of Python against 23,978 LOC of
`crates/*` plus 55,269 LOC of vendored kernel library today. Most of that
55 k is *deleted*, not translated: cuBLAS, cuDNN SDPA, inductor and triton do
the jobs `burn-rmsnorm`, `burn-situ`, `burn-rope`, `burn-attnres`,
`burn-mhc`, `burn-bitnet` and `burn-muon-plus`'s fused kernels do by hand.

The two things that make this less lopsided than the LOC ratio suggests:

- **The instrument layer is a rewrite, not a freebie.** The LOUD/COUNTED/SILENT
  discipline, the `probe.rs` counters, the NaN firewall, the CE ceiling, the
  config-snapshot refusal on resume, the deterministic `--seed` harness, the A/B
  protocol: none of that is in torch, and every one of them exists because a
  silent fallback cost this project a year (`AGENTS.md` §3.2). Python will let
  you lose all of it in an afternoon and the loss curves will look fine.
- **Two implementations of the expert FFN now exist in the port's head** (the
  grouped path that is fast and padded, and the reference that is correct).
  The spike's tests keep them honest; a port must keep doing that.

### Criterion 2 — memory footprint

**Effectively a tie, and torch's number is the more honest one because it is
measured at the shape.**

| | torch, measured | plan's derived budget (§2.3) |
|---|---|---|
| 518.5 M-param stride, b16 s1024, bf16 + AdamW(bf16 states) | **6.93 GiB allocated / 8.88 GiB reserved** (nvidia-smi peak 9,560 MiB incl. 286 MiB of desktop) | — |
| + Engram tables in-VRAM (402 M rows) | → ~8.5–9.5 GiB | ~9–10 GiB "comfortable" (bf16-plan) |
| fp32 masters instead of bf16+SR | one optimizer-construction change, +4 GiB → ~13 GiB | ~13–14 GiB "tight" |

The measured activation + weight + optimizer footprint of the spike lands
**inside the plan's own bf16-plan budget** without any of the bespoke machinery
the plan proposed for it. Decomposed: 1.04 GiB params + 1.04 GiB grads +
2.07 GiB AdamW states + ≈2.8 GiB activations = 6.93 GiB.

Two qualifications, both real:

- The spike does not carry the Engram tables (402 M rows) or the KDA arm's
  saved scratch, and it builds no fused expert kernel. The plan's activations
  row (4–6 GB) is larger than my measured ≈2.8 GiB, so my figure is a
  **floor**, not a substitute for the plan's budget.
- The other direction: the grouped MoE materialises `[E, cap, f]` = **716 MB**
  at cap 64 per iteration and a 358 MB scatter buffer. The plan's TSCT2 §12
  fused flash-MLP expert kernel would remove both. **That kernel does not exist
  in torch and would have to be written in Triton** — the memory win the plan
  wants at E=1024 is a kernel project on the torch side too.

### Criterion 3 — tok/s at the b1 shape

**There is no burn number at the b1 shape. Torch measured 45,861 tok/s eager and
59,901 tok/s compiled, and nothing here says burn will beat that.**

| | measured eff. train TFLOP/s at b1 | 10 B tokens |
|---|---|---|
| torch eager | 15.7 (18.3 at the plan's 0.40 GFLOP/tok) | 2.5 days |
| torch `compile(dynamic=True)` | 20.5 (24.0 at 0.40 GFLOP/tok) | **1.9 days** |
| burn, plan's §2.2 target band | 5–20 (a target, not a measurement) | 8.6 → 2.1 days |
| burn, **measured today** at 9.2 M | 0.6–1.2, launch-bound, 87 % idle | — (different shape; not b1) |

So on the third criterion torch does not lose — torch *is* the only measured
substrate at b1, and its number already sits at the top of the band the burn plan
was aiming for after 1–2 weeks of Phase 3 work. That is the strongest single
result in this spike, and it should be stated with its asterisks: burn's own
ceiling measurement (44.06 TFLOP/s) is the same silicon, so a *finished* burn
could land in the same band; and torch's 20.5 TFLOP/s is 46 % of that ceiling,
i.e. the port has its own 2× of tuning left in it.

### The single biggest open risk on each side

**Option A (stay on burn) — the 0.22 corruption, and it is not a throughput
risk, it is a validity risk.** `AGENTS.md` §3.3: two runs in three destroy
themselves at step 1 via an autotuner coin flip, no config change fixes it, and
"a corrupted run looks like a slow arm". Phase 1 must land before a single b1
step is trustworthy, and Phase 3's load-bearing wall (topk dispatch) is
un-started. Secondary: **the fused KDA adjoint has never been numerically
compared to anything** (§3.2) — the arm is trained by the slow tensor path.

**Option B (port to torch) — KDA, and it is not close.** The real model's only
attention arm is KDA gated-delta linear attention. Torch ships no gated-delta
kernel. The spike substitutes SDPA, which measures the substrate but **cannot
tell you whether the port works**. The port's attention arm is either (a) FLA,
a third-party dependency whose sm_120 support, version pinning and licence are
outside dormouse's control, or (b) 1,500–3,000 LOC of new Triton kernels — i.e.
the same class of work as burn's Phase 3.3, with the same "has never been
compared to anything" risk, just in a language the repo does not currently own.
**If FLA does not have a green sm_120 KDA kernel today, this single item can
consume the entire LOC advantage of the port.** That check is ~1 hour and should
be done before any decision, not after.

### What this spike does NOT measure (do not let the numbers above imply it)

1. **KDA.** Not measured. Both of its costs — forward and backward — are absent.
2. **Muon+ and the TSCT retraction.** AdamW only. Burn measures the retraction
   at a **fixed 52–61 ms/step** (§3.1); if that carries into the port it is
   ~15 % of a 357 ms step and it is not in my 357 ms.
3. **bf16+stochastic-rounding masters.** The spike uses AdamW's bf16 states.
   Whether bf16+SR trains as well as fp32 masters is untouched by everything here.
4. **Engram** — neither the in-VRAM tables nor the host-RAM arm.
5. **The data pipeline, checkpoint/resume, the A/B harness, the guards.** The
   spike has no corpus, no eval window, no BPB.
6. **Aux heads** (JEPA EMA teacher, KoLeo, DSpark).
7. **Any burn measurement at the b1 shape.** The 339× grouped-vs-dense figure is
   a *torch* measurement of a shape burn has not run.
8. Single process, single GPU, one machine, one day, one torch version.

---

## Reproduction

```bash
cd /tmp/opencode/dormouse-torch-spike
uv python install 3.12
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install torch --index-url https://download.pytorch.org/whl/cu129
VIRTUAL_ENV=.venv uv pip install numpy pytest ruff
.venv/bin/python -m pytest -q                 # 4 tests
.venv/bin/python bench_a_gemm.py              # GEMM ladder
.venv/bin/python bench_b_moe.py               # topk-2 vs dense-loop
.venv/bin/python bench_c_loopblock.py         # the e2e step, eager + 2 compile modes
.venv/bin/python profile_step.py              # kernel breakdown
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

Each script calls `common.preflight()` first and refuses to run if the GPU is
busy. Artifacts: `results/*.json` (machine-readable), `results/*.log`,
`results/versions.txt`.
