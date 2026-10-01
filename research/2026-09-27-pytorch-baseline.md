# PyTorch baseline, measured on this box: where dormouse wins, where it does not

Date: 2026-09-27. Subagent deliverable, no source changes (one new file: this one).
Machine: one RTX 5060 Ti 16 GB (sm_120, consumer Blackwell, 170 SM, 1792 GB/s), 62 GB RAM, Linux.
Question asked: is it true that we are so optimized that PyTorch is not even close? **For the KDA
kernel, yes — 45x on the forward, 11.2x on a full fwd+bwd. For the training step, no — PyTorch is
12.5x faster than the path the step actually runs, and our GEMM is behind cuBLAS too. The owner's
sentence is true of the code we wrote and false of the code we run.**

Every number below is measured. Nothing is estimated, and the cells I could not measure are marked
as such rather than filled in.

---

## 0. What state of the tree I measured (the gate moved twice while I worked)

The fused-dispatch gate in `vendor/burn-fused/crates/burn-kda/` was being rewritten by another agent
*during* this benchmark. Three states existed today:

| state | gate | reachable from the trainer? |
|---|---|---|
| A (when I started, 18:40 build) | `is_autodiff_cuda`: `TypeId::of::<B>() == TypeId::of::<Autodiff<CudaBare>>()` (`fused.rs:35`) | **NO** — the trainer's backend is `Autodiff<burn_cuda::Cuda, BalancedCheckpointing>` (`crates/dormouse-train/src/lib.rs:33-36`) and `Autodiff<B, C = NoCheckpointing>` (`burn-autodiff-0.22.0-pre.4/src/backend.rs:25`) makes those two different types |
| B (18:46 build) | `backend_matches::<B>()` = `B::name(...).contains("cuda")` (`burn-gdn2/src/cuda_dispatch.rs:88`) | yes, any wrapper |
| C (HEAD `a131ca4` + uncommitted, when I finished) | `kda_fused_chunk_reported::<B>()` — bare CUDA or any autodiff wrapper, else tensor ops | yes |

**Everything I measured for "ours" is gate-independent**: `kda_bench` and `alloc_probe` call the
fused entry points directly, so their numbers are the *kernels'* cost and stay valid in all three
states. What I could **not** do is measure a rebuilt binary of state C to confirm the trainer now
takes the fused path — that is the other agent's verification, not a claim I make here.

The one-line gate that kept the kernels out of training in state A, for the record:

```rust
// vendor/burn-fused/crates/burn-kda/src/fused.rs:35  (state A)
TypeId::of::<B>() == TypeId::of::<burn::backend::autodiff::Autodiff<CudaBare>>()   // == NoCheckpointing
```

---

## 1. The hot op: gated delta rule (KDA), production shape

`b=10, t=512, h=12, K=V=64, chunk=16, fp32`, 32 chunks, 2.155 GFLOP forward, 692 MB of traffic
(3.1 FLOP/byte against a 24 FLOP/byte machine balance → **memory bound**; HBM floor **0.386 ms**
fwd / **0.965 ms** bwd / **1.35 ms** fwd+bwd, from `/tmp/opencode/kda_arith.py`).

Inputs are at the **real operating point**, taken from the model's own projection dump
(`/tmp/opencode/kda_bfb/*.bin`): `g = -0.054` per token per channel (the `a_log=-3` init), `beta_k =
beta_v = 0.5`, `q/k` L2-normalised, `state = 0`. A synthetic `g = -0.5` (10x the real decay) drives
`k/exp(cumsum g)` to 3e3 per chunk and overflows fp32 after a few chunks — both eager PyTorch and
inductor go non-finite there, so the benchmark would have been timing a kernel production never runs.

### The baseline is the same math, verified

My PyTorch implementation is a line-by-line port of
`vendor/burn-fused/crates/burn-gdn2/src/forward.rs:57` (`chunk_wy_forward_impl`, the `c <= 16` fast
tile path), and it reproduces **burn-kda's own fused CUDA kernel output** on the dumped tensors:

```
fp32 port vs fused kernel: rel err (max-abs) 1.872e-07   (out)
                        state rel err 1.264e-07
```

fp32 round-off. The PyTorch numbers below are therefore a measurement of the same computation, not
of a weaker one. (`/tmp/opencode/validate_vs_kernel.py`. An independent fp64 per-token recurrence
and FLA's `naive_recurrent_kda` were also tried; FLA's erase projects with raw `k` and writes with
`beta*k`, which is the same convention.)

### Results

| implementation | fwd median | fwd min | fwd max | bwd median | **fwd+bwd** | peak VRAM | allocs |
|---|---|---|---|---|---|---|---|
| **ours, fused kernel, bare** (`kda_bench`, 10 runs x 200 iters) | **1.212 ms** | 0.790 | 3.143 | — | — | — | — |
| **ours, fused as one autodiff node** (`alloc_fused_node`, 5 runs) | **5.364 ms** | 0.784 | 6.510 | **20.512 ms** | **25.9 ms** | **170.2 MB** | **11** |
| **ours, what the trainer ran: tensor ops** (`alloc_balanced_backend`, 5 runs) | **1108.6 ms** | 858.1 | 1267.1 | **2519.7 ms** | **3628.3 ms** | **420.2 MB** | **1512** |
| ours, tensor ops on `NoCheckpointing` (`alloc_tensor_path_nc`, 5 runs) | 474 ms | 404 | 942 | 1068 ms | 1542 ms | 420.2 MB | 1512 |
| PyTorch eager, row-loop forward substitution | 293.6 ms | 212.5 | 578.3 | 269.6 ms | 563.2 ms | 315.0 MB | — |
| PyTorch eager, `solve_triangular` | 160.1 ms | 59.1 | 227.2 | 130.2 ms | 290.3 ms | 286.2 MB | — |
| **PyTorch `torch.compile`, best PyTorch** | **54.5 ms** | 31.3 | 134.4 | not measured² | — | 222.3 MB | — |
| PyTorch FLA 0.5.2 `chunk_kda` (Triton) | **not measured**³ | | | | | | |
| *HBM roofline* | *0.386 ms* | | | *0.965 ms* | *1.35 ms* | | |

`fwd+bwd` is the sum of the two medians, not a separately measured quantity: repeated fwd+bwd on the
graph at this size fails in PyTorch autograd ("backward through the graph a second time"), so the
two phases are timed apart — which is also what the Rust harness does. The PyTorch backward row was
measured on an idle GPU (spread 11.3% row-loop, 200% `solve_triangular`).

² the compiled *backward* is not measured — compiling a training step's backward was not attempted.
³ FLA's Triton autotune ran >18 min at load average 44 and was killed. It is the one real gap here.

### Speedups (PyTorch ms / ours ms, >1 = we are ahead)

| comparison | fwd | fwd+bwd | verdict |
|---|---|---|---|
| PyTorch eager/naive **293.6** vs our fused kernel **1.212** | **242x** | — | **WE ARE AHEAD** |
| PyTorch eager/best **160.1** vs our fused kernel **1.212** | **132x** | — | **WE ARE AHEAD** |
| PyTorch compiled **54.5** vs our fused kernel **1.212** | **45x** | — | **WE ARE AHEAD** |
| PyTorch compiled **54.5** vs fused-through-autodiff **5.364** | **10.2x** | — | **WE ARE AHEAD** |
| PyTorch eager/best **290.3** vs fused-through-autodiff **25.9** | 30x | **11.2x** | **WE ARE AHEAD** |
| PyTorch eager/naive **563.2** vs fused-through-autodiff **25.9** | 55x | **21.8x** | **WE ARE AHEAD** |
| PyTorch eager/best **290.3** vs the tensor path the trainer ran **3628.3** | 0.29x | **0.080x** | **PYTHON IS AHEAD 12.5x** |
| PyTorch compiled **54.5** vs the tensor path the trainer ran **1108.6** | **0.049x** | — | **PYTHON IS AHEAD 20x** |
| PyTorch eager/naive **293.6** vs the tensor path the trainer ran **1108.6** | **0.26x** | 0.155x | **PYTHON IS AHEAD 3.8x** |

The row that settles the argument is **fwd+bwd**: a complete PyTorch training step through this op
costs **290 ms**, our fused path costs **25.9 ms**, and the path our trainer actually takes costs
**3628 ms**. PyTorch is **12.5x faster than the code we are training with, and 11.2x slower than the
code we wrote.**

Also: our fused kernel reaches **572 GB/s of the 1792 GB/s peak (32% of the memory roofline)**, and
its best single run (0.790 ms) is 49% of roofline. It is 2.2x off the HBM floor, not 1300x — the
`research/2026-09-27-kda-sota-ceiling.md` alarm about a 1300x gap was measuring the **tensor path**,
not the kernel. The kernel was always fine; the dispatcher was not calling it.

The 11-allocation / 170 MB fused path against 1512 allocations / 420 MB of the tensor path is the
same story in memory: **138x fewer allocations, 2.5x less VRAM.**

---

## 2. Full MHA — the honest vendor test. We have nothing to test.

`b=10, h=12, t=512, d=64`, causal, fp32.

| implementation | fwd median | fwd min | peak VRAM | TFLOP/s |
|---|---|---|---|---|
| `F.scaled_dot_product_attention` **flash** | **UNAVAILABLE** | | | |
| `F.scaled_dot_product_attention` **cudnn** | **UNAVAILABLE** | | | |
| `F.scaled_dot_product_attention` mem-efficient | **0.656 ms** | 0.646 | 16.0 MB | 12.3 |
| `F.scaled_dot_product_attention` math | 3.729 ms | 3.464 | 287.1 MB | 2.2 |
| composite matmul+softmax+matmul | 5.980 ms | 3.502 | 256.0 MB | — |
| **ours** | **does not exist** | | | |

**The "ours" cell is empty because there is no fused MHA in this stack to time.** dormouse's only
attention arm is the KDA gated-delta op of row 1; the MSA arm was deleted (ADR-0014). `burn-cubecl`
0.22.0-pre.4 has no fused attention kernel either — grepping it for
`scaled_dot_product_attention`/flash returns nothing, so a burn-composite attention would be the
naive 5.980 ms row at best, i.e. **9x slower than PyTorch's mem-efficient kernel**.

The finding that matters more than the ratio: **PyTorch's flash-attention backend does not exist on
sm_120 in torch 2.11.0+cu128** (`RuntimeError: No available kernel`), and neither does the cuDNN
backend. The best fused attention PyTorch has on this GPU is the 0.656 ms mem-efficient kernel. So
the usual "is PyTorch naive here" premise is false on this box: where PyTorch *has* a fused kernel
it is 2-3x off its own math fallback, and where it doesn't, it falls back to a composite.

---

## 3. GEMM `[5120,2048] x [2048,8192]` = 171.8 GFLOP

| implementation | median | TFLOP/s | samples | verdict |
|---|---|---|---|---|
| **ours, cubecl f32** | **47.691 ms** | **3.6** | 4.807 / 47.1 / 47.7 / 51.7 / 71.1 ms — **15x spread** | **not established** |
| ours, cubecl f16/bf16 in (`bf16_matmul`) | 27.8 ms | 6.2 | 36.4 / 27.8 / 52.0 ms | **PYTHON IS AHEAD 7.5x** |
| PyTorch cuBLAS f32 (fp32 accum) | 10.881 ms | 15.79 | 10.3-11.1 ms (4% spread) | — |
| PyTorch cuBLAS f16 in / f32 accum | 3.663 ms | 46.90 | 3.38-4.01 (18%) | — |
| PyTorch cuBLAS bf16 in / f32 accum | 3.615 ms | 47.52 | 3.44-3.94 (9%) | — |
| PyTorch cuBLAS TF32 | 13.960 ms | 12.31 | 12.5-16.5 | — |

**This is the row where we are worst, and I will not put a ratio on the fp32 half.** Five runs of the
**same binary** (`gemm_probe`, mtime 13:35:40, unchanged) returned 4.807 / 47.100 / 47.691 / 51.691 /
71.138 ms for one fixed matmul — a **15x spread**, ordered fastest-first to slowest-last, which is the
signature of clock/power behaviour rather than of code (the card idles at 645 MHz, is capped at
180 W and sits at 47 °C, so it is the power cap, not heat). At the median we are 4.4x behind cuBLAS;
at the fastest sample we would be 2.3x *ahead*. **The measurement does not resolve which, and claiming
either would be inventing a number.** The cuBLAS side is solid by comparison — 4% spread across 15
timed iterations. Resolving this needs clock logging and cooldowns between runs, not more runs.

Two things that are *not* in doubt:

- **Our f16/bf16 path is 7.5x behind cuBLAS** (best of three: 27.8 ms vs 3.663 ms; 6.2 vs 46.9
  TFLOP/s) and points the same way in every run, so the 15x fp32 spread does not rescue it. The
  probe's own comment already flags `bf16_matmul` as broken on burn 0.22.0-pre.4 + cuda; the number
  says it is not merely broken, it is slow. **Every tensor-core win available to this model is
  currently being left on the table**, and f16 is where the 3x lives.
- **cuBLAS TF32 (12.3 TFLOP/s) is slower than cuBLAS true fp32 (15.8)** on this GPU, which is not how
  a tuned library behaves and suggests cuBLAS has no good sm_120 TF32 path either. Both of us are
  leaving something on the table here; PyTorch just leaves less.

---

## 4. End-to-end training step: **skipped, and I will not guess it**

Not measured. Getting it needs an exclusive GPU window of minutes (the trainer takes the box), the
data mount, and a preset that fits — and the GPU was never free for more than ~2 minutes at a time
while another agent ran production `train` jobs and 15-22 concurrent `rustc` processes.

What I can say from the per-op numbers above, without pretending it is a step measurement: the
trainer ran KDA through the tensor path at **1108.6 + 2519.7 = 3628 ms per call**, at 4 loop
iterations, which is the same order as the `1445 ms of KDA in an 1810 ms step` recorded in
`research/2026-09-27-kda-sota-ceiling.md` — that earlier figure was taken with a warm allocator
inside a running step, mine is a cold-pool single call, so **treat the two as the same phenomenon
measured at different pool temperatures, not as a contradiction**. The step-level `--timers`
fwd/bwd/opt/retr/ema breakdown, and the PyTorch equivalent, remain unmeasured.

---

## 5. Methodology, and the two ways these numbers can mislead you

**Runs.** Every cell is a median over ≥5 timed iterations after 3-5 untimed warmups (CUDA events on
the PyTorch side; `kda_bench` averages 200 iterations per launch and I ran it 10 times). Min and max
are printed alongside every median in the raw logs. Peak VRAM is
`torch.cuda.max_memory_allocated` / cubecl's own `Client::memory_usage()` deltas, both taken around
the timed region only.

**Raw logs and scripts** (all under `/tmp/opencode`, nothing added to the repo):
`gdr_ref.py` (the port), `validate_vs_kernel.py` (port-vs-kernel check),
`bench_all.py` + `bench_all.stdout` (KDA forward: eager row-loop, eager `solve_triangular`,
compiled), `bench_bwd.py` (KDA backward), `bench_mha_gemm.py` (rows 2 and 3), `bench_f16.py` (the
GEMM precision isolation), `rust_runs2.log` (every Rust launch, 5 per config), `final_runs.log`
(the 5 `gemm_probe` launches), `kda_arith.py` (roofline arithmetic), `run_final.sh` (the gated
driver).

**What was compiled.** torch 2.11.0+cu128, Triton 3.6.0, FLA 0.5.2, installed by me into
`/tmp/opencode/pt` (nothing was installed system-wide). `torch.compile` default mode, 89 s to
compile, verified finite and matching the eager output. TF32 left at the PyTorch default (**off**),
so the fp32 comparison is fp32 on both sides; TF32 and f16/bf16 are reported as separate rows.

**Caveat 1 — the machine was not quiet.** Load average ran 40-74 on 20 cores with 15 `cargo` and 22
`rustc` processes from other agents throughout. This inflates any **host-launch-bound** measurement
and leaves GPU-bound ones alone. The KDA tensor path is launch-bound (1512 allocations,
~4800 launches), so its 858-1267 ms forward band is a contaminated absolute value — but the
*conclusion* survives the noise, because even the most favourable value measured for us (404 ms on
the `NoCheckpointing` variant, 858 ms on the trainer's) is still 8-20x behind PyTorch's compiled
forward. The fused kernel's 1.212 ms is a 200-iteration average and is the most trustworthy number here.

**Caveat 2 — one GPU process at a time was breached twice, by me.** My first Rust driver checked the
GPU, found it free, and a 15 GB process appeared in the gap before `exec`; my `kda_bench` runs
crashed with `CUDA_ERROR_DEINITIALIZED` while the other process ran. Later my `bench_bwd` launched
into a three-way race with another agent's `kda_alloc_probe` and `train_prefix` — I killed my process
rather than publish numbers taken in a three-way collision. No damage to any other process that I can
see, and every number in this report comes from a run that completed on a GPU that was otherwise
idle. The rewritten driver blocks instead of proceeding and treats >200 MiB as busy (the GUI,
Telegram at ~30 MiB, does not count) — but on a box where two agents launch within seconds of each
other, a check-then-exec gate is inherently racy and I should not claim otherwise.

**What I did not measure:** FLA's Triton kernel (autotune >18 min at this load); the compiled
*backward*; the end-to-end step; any bf16 variant of the KDA op; a reproducible fp32 GEMM number
(15x spread, see §3). Each is a gap in the comparison, not a favourable assumption.

---

## 6. Verdict, blunt

1. **The kernel is not the problem and never was.** Our fused gated-delta kernel is **45x faster than
   the best PyTorch I could build** in the forward (1.212 ms vs 54.5 ms compiled; 132x vs eager/best,
   242x vs eager/naive) and **11.2x faster on a full fwd+bwd** (25.9 ms vs PyTorch's 290 ms), uses
   **138x fewer allocations** and 2.5x less VRAM, and runs at 32% of this card's memory roofline.
   "PyTorch is not even close" is **true here**, by one to two orders of magnitude.
2. **The training step is the problem, and it is behind.** On the path the trainer actually ran,
   PyTorch's fwd+bwd is **12.5x faster** than ours (290 ms vs 3628 ms), and even PyTorch's naive eager
   version is **6.4x** faster (563 ms). A 3.6 s KDA call in a 16 GB box is why the step is 1810 ms.
3. **The gap is a dispatcher, not a kernel.** State A's gate
   (`TypeId::of::<B>() == TypeId::of::<Autodiff<CudaBare>>()`, `fused.rs:35`) could not match the
   trainer's `Autodiff<Cuda, BalancedCheckpointing>`, so every training step ran the tensor path. It
   is being replaced with a device-name test. **Before spending anything else on this op, prove with
   a rebuilt binary that a training step now dispatches into the fused kernel** — if it does, row 1
   goes from **12.5x behind to 11.2x ahead** on fwd+bwd, and the largest single cost in the model
   disappears. That verification is worth more than any kernel work.
4. **The GEMM path is a second, independent loss.** Our f16/bf16 matmul is **7.5x behind cuBLAS**
   (6.2 vs 46.9 TFLOP/s) in every run; the fp32 matmul is 4.4x behind at the median but its own
   measurement spread is 15x, so I am not claiming a ratio for it. This one is not a dispatch
   problem — the kernel is engaged and it is simply slow — and it is the cheapest thing on this list
   to fix, because a working bf16 tensor-core GEMM is a known quantity and we are already paying for
   the memory traffic.
