# Multi-GPU and heterogeneous hardware

**Status, 2026-10-19 (commit `315d3d0`, `6f75dfa`, `1111291`).** This
box has ONE GPU, so nothing in this document has been run on a real multi-GPU
fleet. What HAS been run, and what everything below rests on:

| claim | the instrument | result |
|---|---|---|
| the device policy layer decides dtype/kernels per card | `tests/test_device_policy.py` | table over synthetic sm_75/86/120/70/61/52 records + one real probe |
| 2-rank DDP on the CPU arm reproduces the single-process trajectory | `tests/test_distributed.py`, `tools/dist_gate.py` | 20 steps, max \|Δloss\| **0.0055** (tolerance 0.01), 1-step parameters **3e-8** relative |
| ranks stay identical | the same gate | identical SHA-256 of every parameter after every step |
| unequal per-rank batches equal the token-weighted mean | `gradref` mode of the gate | 4x64 + 2x64 tokens vs a hand-computed `(4·g0 + 2·g1)/6`, **1.1e-5** relative |
| FSDP2 + the custom Muon optimizer | the same gate's refusal | **does not work**; the exact error and the honest fallback are below |
| FSDP2 + AdamW (control arm) | `--shard fsdp --opt adamw` | 20 steps, gathered parameters identical across ranks |
| NCCL missing on a CUDA fleet | unit gate | LOUD refusal, never a gloo fallback |

Everything else — a real NCCL run, real per-card dtype decisions, PCIe numbers
— is COMPUTED or DESIGNED here and named as such.

---

## 1. Supported topologies

* **One process, one card** — the path every number in this repo was measured
  on. Unchanged: the distributed knobs default off and the single-GPU code
  paths are byte-identical (gated: `test_dtype_none_never_probes_and_old_snapshots_round_trip`,
  `test_the_trainer_does_not_touch_the_device_layer`).
* **N processes, N cards, one node** — data parallelism. `nccl` on CUDA,
  `gloo` on the CPU arm. DDP is the verified reducer (`dist_shard=none`);
  FSDP2 (`dist_shard=fsdp`) covers the AdamW control arm only.
* **Heterogeneous cards** — the same N-process run with a per-rank capacity
  profile (`--capacity "8,2"`), so the weak card carries fewer tokens per step
  and the synchronous all-reduce is not held hostage by it. Each rank also
  resolves its own compute policy (`dtype=auto`), which is the point of the
  device layer.
* **Not supported** (LOUD, not silent): multi-NODE (no rendezvous story, no
  elastic launch), the host Engram tables on a fleet (§5), the capacity-padded
  MoE on a fleet (§5), the LFB controller on a fleet (§5), the CUDA-graph
  optimizer stage and `torch.compile` on a fleet, and the data mixture on a
  fleet. Each refusal names its cause and its escape in
  `distributed.refuse_incompatible`.

The launcher is torch's own: `torchrun --standalone --nproc_per_node=N -m
dormouse train --dist …` sets `RANK`/`WORLD_SIZE`, and the code refuses both
halves of the mismatch — `--dist` with `WORLD_SIZE=1` (no fleet to join) and
`WORLD_SIZE>1` without `--dist` (N ranks training the same data and
overwriting one checkpoint).

## 2. The dtype policy matrix

`train/device.py` probes each device once (metadata only: no allocation, no
kernel) and derives a policy. The ladder is the CUDA tensor-core table: fp16
from sm_53 (Maxwell), bf16 from sm_80 (Ampere). The kernels follow the port's
own operand rules.

| GPU generation | example | bf16 | fp16 | `auto` resolves | Triton MoE | FLA KDA | what actually runs today |
|---|---|---|---|---|---|---|---|
| sm_120 (Blackwell) | RTX 5060 Ti | yes | yes | **bf16** | legal | legal | bf16 forward, fused KDA, Triton MoE arm available |
| sm_89 / sm_86 (Ada/Ampere) | RTX 4090 / 3090 | yes | yes | **bf16** | legal | legal | bf16 forward, fused KDA |
| sm_75 (Turing) | **GTX 1650 4 GB** | **no** | yes | **fp16** | **illegal** | legal | **fp32 forward** — the port has no fp16 forward yet (see below) |
| sm_70 / sm_61 | V100 / 1080 Ti | no | yes | fp16 | illegal | legal | fp32 forward |
| sm_52 and older | pre-Maxwell | no | no | fp32 | illegal | **illegal** (SDPA arm) | fp32 forward, SDPA KDA |

Three rules make this honest rather than a table nobody enforces:

* **an explicit rung the device cannot honor is LOUD.** `--set dtype=bf16` on a
  1650 dies naming the card, `sm_75`, the sm bf16 starts at, and the escape.
  The alternative — quietly running fp16 under a bf16 flag — is a different
  experiment with the requested label on it.
* **`dtype=auto` is COUNTED**: the log line names the reading (`bf16=0
  fp16=1`), not just the verdict, and the memory budget (`free − 512 MiB`).
* **the PORT's half of the contract**: the port's forward has no fp16 path
  (the f16 plan is storage-only so far). An **explicit** `dtype=fp16` is LOUD;
  an `auto` policy that resolved fp16 is a **COUNTED fp32 fallback for that
  rank**, naming `docs/architecture/f16-plan.md`. That distinction is what
  makes a mixed fleet work today: `dtype=auto` on a 5060 Ti + 1650 fleet runs
  **bf16 on the strong card and fp32 on the weak one**, each rank announcing
  its own rung. Until the fp16 forward lands the 1650 is slower than it can
  be — correct, counted, and one gate away from better.

`dtype="none"` (the default) means the layer is never called: the `bf16` flag
decides as it always has and the config round-trips unchanged. The probe also
reports the PCIe link's **max** gen/width — the `current` value nvidia-smi
prints is idle-downclocked (this box reads max 3x16, current 3x8).

Machine fact measured while taking the probe (2026-10-19): while another
process holds the card, the driver refuses a second CUDA context at all —
`get_device_properties` answers, `mem_get_info` dies with
`cudaErrorDevicesUnavailable`. `free_mem=-1` ("unknown") is a real probe state,
and the budget says so instead of inventing a number.

## 3. Data parallel: where the code lives

* `train/distributed.py` — `init_parallel` (probe → policy → capacity →
  process group), `Parallel` (the loop's seam: `shape`, `loss_scale`,
  `all_finite`, `fleet_loss`, `shard`, `replicate`), `choose_backend`,
  the PCIe arithmetic, `train_dist`.
* `train/loop.py` — four seams and nothing else: the stream is the rank's own
  byte range and micro-batch; the loss is scaled by `world_size · t_rank / Σt`
  before `backward`; the NaN decision is ONE collective (a rank that skipped
  `backward` would hang the reducer); only rank 0 evaluates, checkpoints and
  logs. Everything else is the single-process code.
* `tools/dist_gate.py` — the runnable gate (also the owner's two-card check).

**The data layout.** Rank `r` starts at `Σ_{i<r} b_i·(seq+1)` bytes and advances
by its own window. With equal batches the fleet's union at step `s` is exactly
the contiguous block a single process at `Σb` reads at step `s` — that is why
the loss-curve gate is an equality and not a hope. The cost is stated: because
each rank advances by its own window while the fleet's footprint per step is
`Σb`, the ranks' windows overlap ACROSS steps (rank 1's step `s` is rank 0's
step `s+1`), so a fleet re-reads each block ~P times over a run. Correct, and
wasteful for a long run; the strided sampler (rank `r` reads blocks `r, r+P,
…`) is the named follow-up and needs a stream seek the ring buffers do not have.

**The gradient.** DDP divides the summed gradients by `world_size`; the
objective is the mean over the GLOBAL batch's tokens. Each rank scales its loss
by `world_size · t_r / Σt`, so the reducer lands on `Σ (t_r/Σt)·g_r` exactly.
With equal shapes the factor is 1.0 today (the number every single-GPU run
has).

**Gradient communication.** bf16/fp16 reduce is a COUNTED choice, not a
default: `dist_reduce=auto` takes the fleet's compute dtype (bf16 on this
card), and `fp32` is the exact reference the gate compares against. DDP's
buckets are sized from the payload (`bucket_mb_for`: payload/8, clamped to
4-128 MiB) — torch's 25 MB default is an NVLink-era number, and the PCIe
sizing is a starting point to sweep on two cards, not a measured optimum.
Backward overlap comes from DDP's ready-order bucketing (the reducer starts a
bucket as soon as its gradients are complete); FSDP2's `fully_shard` prefetches
the next all-gather in its backward by default. There is no hand-rolled
communication in this repo.

## 4. The custom optimizer meets FSDP2 — measured, and refused

FSDP2 (`torch.distributed.fsdp.fully_shard`) shards parameters as DTensors.
The port's `MixOptimizer` is fine for AdamW but **Muon is not**, and the
failure is not a preference:

```
RuntimeError: aten._foreach_add_.List got mixed torch.Tensor and DTensor,
need to convert all torch.Tensor to DTensor before calling distributed operators!
File "src/dormouse/train/optim.py", line 549, in _momentum_update_
```

Measured 2026-10-19, 2 ranks, gloo, CPU, the gate's tiny model. The cause: the
batched Muon pass keeps ONE stacked `torch.Tensor` per shape group as its
momentum canonical state, while FSDP2's gradients are DTensors; the batched
`_foreach_add_` cannot mix them. Fixing it means a shard-aware Muon (gather the
full matrix for the polar, scatter the update back) — a real change with its
own numeric gate, not a wrapper.

So the shipped rule is: **DDP is the reducer for the Muon arm; FSDP2 is
available for `opt=adamw`** (the control arm), and anything else is LOUD with
the error above in the message. FSDP2 additionally refuses resume and
checkpointing: an FSDP2 state dict is DTensor-sharded, the port's checkpoints
are single-file fp32 masters, and a file rank 0 cannot reload is worse than no
file. That is a follow-up (`torch.distributed.checkpoint`), not a knob.

## 5. Non-parameter state that does NOT reduce

Data parallelism synchronizes *parameters*. Three pieces of this model carry
live state OUTSIDE the parameter set, and each one would have produced N
silently different models that look synchronized. All three are LOUD refusals
on a fleet, with the mechanism named:

1. **The host Engram tables** (`engram_ram`). `Engram.to_host` deletes the
   table parameter; the rows and their momentum live in host RAM per process.
   The reducer never sees them. On a fleet the tables must stay in VRAM (where
   they ARE parameters and DO reduce), which at b1 costs 402M params of
   replication per rank. A row-gather for the host path is the follow-up.
2. **The MoE capacity-padded dispatch** (`moe_topk > 0` and `moe_capacity > 0`).
   The cap is `ceil(capacity · tokens · k / E)`, so WHICH tokens are dropped
   depends on the window's composition. Splitting the window across ranks
   changes the model. Measured on the gate's fixture (2026-10-19): batch 4 vs
   2x batch 2 logits differ by **0.357** at `moe_capacity=1.0`, and by
   **1.2e-6** (fp round-off) at `moe_capacity=0` (dropless). The refusal is the
   same class `config._validate_segment` already applies to `segment_len`.
3. **The LFB load-balancing controller** (`moe_lb_coef != 0`): `lfb_counts` and
   `lfb_bias` move during every routed forward. Each rank would steer its own
   controller; the parameters would stay identical and the MODELS would not.
   Off by default (0.0); a fleet reduction of the counts is the follow-up.

This is the part of the brief that mattered most: the Engram tables were
already known to live outside the parameter set, and the other two were found
by the gate failing for the right reason.

## 6. PCIe arithmetic

A ring all-reduce moves `2(P-1)/P` payloads per rank (one reduce-scatter plus
one all-gather of a `1/P` slice each); at P=2 that is exactly one payload.
`allreduce_seconds` uses the link's theoretical rate × 0.8 (`LINK_EFFICIENCY`,
an assumption, not a measurement here).

This box's link: **PCIe gen3 x16** (max; the card is a gen5 x8 part), 15.76 GB/s
theoretical, 12.6 GB/s at the assumed efficiency.

| shape | bf16 gradient payload | 2 ranks | 4 ranks |
|---|---|---|---|
| medium, 138M params | 276 MB | **21.9 ms** | **32.8 ms** |
| b1, 1.07B params | 2.14 GB | **170 ms** | **255 ms** |
| (gen5 x8 link) 138M | 276 MB | 11.0 ms | 16.4 ms |
| (gen5 x8 link) 1.07B | 2.14 GB | 85 ms | 128 ms |

Read these as COMPUTED, with their inputs named: payload = params × 2 B, link =
max gen/width × the PCIe table, efficiency 0.8. The two-card test replaces them
with a reading; `tools/dist_gate.py --report-link` (or the formula in
`distributed.py`) prints them so the substitution is mechanical. What they mean
for a step: at 138M the all-reduce is ~2% of a ~1 s step and is not the story;
at b1 it is ~12-18% of a ~1.4 s step, which is why the reduce dtype is bf16 and
the buckets are sized to the payload. They also say something the marketing
does not: **two cards on gen3 x16 share 12.6 GB/s, so a 1.07B step cannot go
faster than the all-reduce allows** — PCIe, not compute, is the ceiling once
the model is big enough.

## 7. The capacity profile (heterogeneous cards)

`--capacity "B0,B1,…"` (or `BxT` per rank) declares each rank's micro-batch;
`--capacity auto` measures a fixed 4096³ bf16 matmul on each rank, all-gathers
the rates, and splits the global batch (`cfg.batch × world_size`) in
proportion, keeping the total exact and every rank at ≥ 1 sample.

Why this is the lever: with a synchronous all-reduce the step time is
`max_i (tokens_i / rate_i)`. Giving the weak card fewer tokens equalizes the
local step times; in the ideal proportional split the fleet's throughput is
`Σ rate_i`, so **the weak card adds its rate and never subtracts**. A 1650 at
1/4 of a 5060 Ti's rate should carry ~1/5 of the tokens in a 2-card fleet
(4:1 → the split that equalizes the two local times), which is what `auto`
computes from measurement rather than from a spec sheet.

The gate: rank 0 at batch 4 and rank 1 at batch 2, one step, against the
hand-computed `(4·g0 + 2·g1)/6` — measured **1.1e-5** relative, ranks
identical.

## 8. The scrap-metal path (`--fit`)

One knob picks the set; the resolver prints the set and the arithmetic
(`fit=16.0GB: est_peak=8.03GiB engaged=moe_offload,engram_ram (weights=… )`).
The order is cheapest-first, and the mechanisms are the existing ones — no new
engine:

| mechanism | what leaves VRAM | exact? | gate |
|---|---|---|---|
| `moe_offload` | the routed expert pass's saved activations → host RAM | **yes** | `tests/test_offload_experts.py` |
| `engram_ram` | the n-gram tables + momentum → host RAM | **yes** | `tests/test_engram_host.py` |
| `muon_state_dtype=bf16` | half of Muon's momentum | approximate by design | `tests/test_muon_fast.py` |
| `optim_8bit` | AdamW's fp32 m+v → bitsandbytes 8-bit | approximate by design | the optim lane's bench |

`act_ckpt` (quantized saved tensors) is a real mechanism with its own bench but
**no config seam yet** — it is not in the plan, and this paragraph is the
statement of that gap rather than a pretence that the list is complete.

The estimate is a MODEL, stated as one: fp32 masters (4 B), fp32 grads (4 B),
AdamW 8 B or 2 B, Muon 4 B or 2 B, activations 64 B per (token × d_model)
derived from the b1 spike's measured decomposition, and the Muon/AdamW split
read from the REAL optimizer routing on a `meta` build (not a reproduction).
The plan must fit 90% of the stated budget (`FIT_HEADROOM`); a card it cannot
fit is LOUD with every component named.

**What fits where** (estimates, this port's fp32 masters; the b1 row is
cross-checked against the spike's measured 6.93 GiB at 518.5M params):

| shape | batch × seq | 4 GB card | 16 GB card |
|---|---|---|---|
| small, 9.5M | 8 × 512 | fits, **no mechanism** (0.31 GiB) | fits, none |
| medium, 142M | 8 × 512 | fits, **no mechanism** (2.03 GiB) | fits, none |
| b1, 1.07B | 8 × 512 | **does not fit** (6.68 GiB after every mechanism) | `moe_offload` + `engram_ram` → 8.03 GiB |
| b1 on a 2-card fleet | 8 × 512 | — | `moe_offload` + Muon bf16 → 12.52 GiB (host tables unavailable) |

The medium row is the one that matters for the 1650: **a 142M model fits a
4 GB card without any offload at all**; the 4 GB story is about the b1 shape,
and there the answer is "not on one 4 GB card, with or without this repo's
mechanisms".

## 9. What is verified here vs what needs two cards

**Verified on this box (CPU arm, 2 OS processes, gloo):** DDP trajectory,
gradient synchronization, token-weighted unequal batches, the NaN agreement,
the data layout, the FSDP2 + AdamW path, every refusal, and the single-GPU
byte-identical default.

**Needs two cards (and is NOT claimed):** NCCL's actual all-reduce and its
timing; the real PCIe rate (the table is theoretical × an assumption); FSDP2's
all-gather/prefetch on CUDA; a heterogeneous dtype fleet on real cards (the
per-rank policy — bf16 beside a COUNTED fp32 — is gated by unit tests only);
per-rank `throughput_index` on two different cards; and the launch-bound
question below.

**The one command that checks a two-card box** (the same gate, on CUDA):

```
torchrun --standalone --nproc_per_node=2 tools/dist_gate.py \
    --mode rank --out /tmp/gate --steps 20 --batch 2 --device cuda
```

Expected: both ranks write `rank0.json`/`rank1.json` with **identical
fingerprints**, 20 losses that descend, and (with `--capacity 4,2` plus
`--mode gradref`) the unequal-batch check. Then the real launch:

```
torchrun --standalone --nproc_per_node=2 -m dormouse train small --dist \
    --device cuda --capacity auto --fit 16 --data <corpus.bin> --steps 2000 \
    --set log_every=20
```

How to read that it worked: one `dist rank=0/2 …` line per rank (rank 1 prints
nothing else — rank 0's log is the run's log); `shape=` per rank shows the
capacity split; the `loss=` on rank 0 is the FLEET's token-weighted mean, not
its micro-batch's; every eval line carries `dist=0/2`; and the two cards'
`nvidia-smi` utilisation moves together. If a rank dies, the other exits on a
collective timeout rather than continuing alone — that is deliberate.

## 10. The honest expected speedup

Two regimes, and the difference is this project's central performance fact
(AGENTS §3.1: the GPU is idle ~87% of a warm 9.2M step — the work is
launch-bound, not compute-bound).

* **Launch-bound shapes (small, ≲80M): data parallelism buys TOKENS PER STEP,
  not step time.** Both ranks run the same number of kernels; the host is the
  bottleneck and it is now two hosts. A 2-card run at the same global batch is
  expected to be ~**1×** (no faster), and at 2× the global batch it is ~1× per
  step with twice the tokens — a gradient-quality buy, not a wall-clock one.
* **Compute-bound shapes (b1-class, GEMM-dominated): throughput = Σ rate_i.**
  Two equal 5060 Tis → up to **2×**, minus the all-reduce (170 ms of a ~1.4 s
  step at gen3 x16, ~12%). A 5060 Ti + a 1650 at 4:1 → **1.25×** (the weak card
  adds a quarter of a strong card), *if* the fp16 forward lands; until then the
  1650 runs fp32 and its share is smaller still.
* Four mismatched cards (one strong, three at 1/4) → **1.75×**, with the
  all-reduce at 255 ms/step for the b1 shape.

Anything above those numbers on a launch-bound shape would mean the
all-reduce is free and the host got faster, which is not a claim this repo can
make.

## 11. Known gaps (named, not hidden)

1. **fp16 forward** — the 1650's best rung; today it runs fp32.
2. **A shard-aware Muon** — the only thing between FSDP2 and the training
   optimizer; the measured error is in §4.
3. **FSDP2 checkpoints** — `torch.distributed.checkpoint` for the DTensor state
   dict; refused today.
4. **A row-gather for the host Engram tables** — the b1 fleet recipe keeps
   them in VRAM, replicated.
5. **A fleet reduction for the LFB counts** — the controller is off by default.
6. **A strided sampler** — the current data layout re-reads each block ~P
   times over a run.
7. **`act_ckpt` has no config seam** — a real mechanism outside this plan.
8. **The reduce-bucket sweep** — `bucket_mb_for` is payload/8, not a measured
   optimum; two cards decide it.
