# torch.compile recompilation under data-dependent MoE routing — evidence and options

**2026-10-18.** Research + one local probe. No code changed, nothing committed.
Trigger (given by the task, not re-measured here): `torch.compile(dynamic=True)` on the
port hits `torch._dynamo hit config.recompile_limit (8)` in `SparseMoE._banks`
(`src/dormouse/model/experts.py:536-585`) at 2939 ms/step vs 573 eager (0.20x);
`dynamic=False` at b1 never produced a step (>45 min compile).

## 0. What `_banks` actually keys on (read first, options target this)

- `_banks` (`src/dormouse/model/experts.py:556-637`). Cache key at **585-590**:
  `(dt, factor_qat, factor_qat_alpha, tuple(p._version for e in self.experts for p in e.parameters()), want)`;
  cache read at 592, reuse clause 598-602, rebuild+store at 603-636. Call sites: **431**
  (dense banked bmm), **690** (padded dispatch), **1008** (Triton dispatch).
- **Correction to the task's framing**: the key does *not* contain the routing/assignment
  tuple. Routing enters the compiled region only as tensor shapes through
  `index/order/keep`; `dynamic=True` covers those (the 2026-10-15 spike measured that,
  and `dynamic=False` failed on exactly the shape guard: `tensor '___stack0' size
  mismatch` — `docs/research/2026-10-15-torch-spike.md:236-249`). The per-step *Python*
  movers in the key are `factor_qat_alpha` (set on every module, every step, by
  `DormouseModel.set_factor_qat_alpha`, `src/dormouse/model/model.py:81-95`), the
  `_version` tuple (optimizer step), and `torch.is_grad_enabled()` (train/eval).
- `loop.py:612-627` refuses `opt_graph`+`torch_compile` together and wraps
  `torch.compile(model, dynamic=True)` — the spike's measured winner, keep it.

## 1. Verdict table (Q1-Q4; each claim names its source)

| # | Claim | Source (fetched unless marked) | What it says |
|---|---|---|---|
| 1 | torchtitan compiles **regions**, not the whole model, with `fullgraph=True`; `recompile_limit` is treated as a graph budget | `pytorch/torchtitan @main: .claude/skills/local_compile_perf/SKILL.md`, `references/measurement.md`, `torchtitan/distributed/local_compile.py` | "Graph budget: compiled graphs per region; `recompile_limit` is 8 by default and `fullgraph` turns overflow into an error"; "we aimed for ≤ 6 of 8"; raising the limit "costs more compiles and more cached graphs". `local_compile` requires `fullgraph=True`; MoE glue is compiled as named regions (e.g. `@local_compile("moe_aux_loss", batch_invariant=False)` in `torchtitan/models/common/moe/moe.py:497`, with an eager-vs-compiled fwd+bwd test). |
| 2 | torchtitan sets `capture_scalar_outputs = True` when compiling | `pytorch/torchtitan: torchtitan/experiments/graph_trainer/compile.py:61` | `torch._dynamo.config.capture_scalar_outputs = True` in `apply_compile`. |
| 3 | torchtitan's CUDA-graph module **excludes** host-sync ops and data-dependent shapes from capture | `pytorch/torchtitan: torchtitan/experiments/graph_trainer/cuda_graph.py:119,123,133-134` | "`.item()/.tolist()` need a device-to-host sync a CUDA graph replay can't redo"; "Op with a dynamic (data-dependent / unbacked-SymInt) input or output shape"; "only captures CUDA kernels — a CPU op would run on the host at capture time and replay stale (e.g. the `.tolist()` … on EP token-routing split)". |
| 4 | OLMo-core compiles **per transformer block**, marks batch/seq dims dynamic, and keeps non-Dynamo-safe pieces out with `@torch.compiler.disable` | `allenai/OLMo-core @main: src/olmo_core/nn/transformer/model.py` (`apply_compile`, `mark_dynamic(h,(0,1))`), `train_module/transformer/config.py` (`compile_model`), `CHANGELOG.md`, `nn/moe/v2/routed_experts.py:167,233,276,302`, `nn/moe/v2/no_ep.py` | CHANGELOG: "Avoid torch dynamo recompiles … by marking `cu_doc_lens` and `max_doc_len` dynamic"; "Excluded `mark_dynamic` from `torch.compile` tracing (`@torch.compiler.disable`)"; "Disabled `torch.compile` tracing through `TEAttentionBackend.forward`, whose Python/pybind setup is not Dynamo-safe". MoE permute/unpermute + the D2H copy run in `*_no_compile` helpers on a separate stream (`dtoh_event.synchronize()`); a commented-out `mark_dynamic` carries the note that dynamic shapes "can trigger Dynamo constraint violations". Guide: "Always use `torch.compile`". |
| 5 | Megatron-LM: CUDA graphs are for the **static parts**; MoE expert dispatch is left eager unless shapes are made static | `NVIDIA/Megatron-LM/docs/user-guide/features/cuda_graph.md`, `megatron/core/transformer/moe/README.md` | "MoE expert dispatch involves dynamic shapes and cannot be captured. `--cuda-graph-modules` is used to capture only the static parts (attention, router, preprocess) while leaving expert compute in eager mode." Full-iteration capture requires `--no-check-for-nan-in-loss-and-grad` because "NaN checks involve CPU-GPU synchronization which cannot run inside a CUDA graph"; with paged stash + `--moe-expert-rank-capacity-factor`, dispatch shapes become static and full-iteration graphs work on MoE. |
| 6 | Megatron JITs (torch.compile) the **router/preprocess**, not the experts | `NVIDIA/Megatron-LM` PR #1919 "JIT for MoE router and preprocess" (search result, not fetched directly) | "Enable JIT (`torch.compile`) for expert bias and MoE preprocess. Let force load balancing compatible with CUDA Graph." |
| 7 | The Megatron-MoE paper states the tradeoff directly: graphs vs dropless routing | arXiv `2603.07685v2` (HTML fetched) | "CUDA Graphs eliminate host overhead but require static tensor shapes, conflicting with dropless routing"; "In dropless MoE, dynamic tensor shapes further require costly host-device synchronization"; "The router can be fused into CUDA Graphs without affecting dispatcher logic". §4.3.7 "Full CUDA Graphs Coverage for Dropless MoE via Sync-Free Kernels, ECHO, and Paged Stashing": device-initiated Grouped GEMM reads per-expert shapes from device memory (cuBLASLt device arrays since CUDA 13.1, or cuteDSL), HybridEP pre-allocates by an estimated upper bound to remove synchronizations, ECHO clones hot experts, paged stash pre-sizes dispatch. |
| 8 | DeepSeek: no compile/graph statement in the V3 report; DeepGEMM has a graph-era masked GEMM (inference) | arXiv `2412.19437v1` (fetched; 147k chars stripped); `deepseek-ai/DeepGEMM` README; `deepseek-ai/DeepEP` README | V3 report: **0** occurrences of "CUDA graph", "torch.compile", "graph capture", "host-side", "device-to-host". DeepGEMM: "During the inference decoding phase, when CUDA graph is enabled and the CPU is unaware of the number of tokens each expert receives, we support masked grouped GEMMs. By providing a mask tensor, the kernel computes only the valid portions." DeepEP README: **no CUDA-graph statement found** (it exposes event-based overlap). |
| 9 | PyTorch's canonical recompilation guidance: per-step Python scalars are guarded on exact value | `docs.pytorch.org/.../programming_model.recompilation.html` (fetched; updated 2026-07-28) | "By default, `int`/`float` variables are treated as constants and are guarded on their exact value" → recompile each call; fix: "wrap `float` variables in tensors". "Dynamo specializes on (guards the exact value of) integer attributes of `nn.Module`s. An integer that changes every call … triggers a recompilation on each call, quickly hitting the recompilation limit and falling back to eager" → `torch._dynamo.config.allow_unspec_int_on_nn_module = True`. Plus: "Graph Breaking to Reduce Recompilation Costs" (intentional break = smaller recompiled region) and `cache_size_limit` (alias `recompile_limit`, default 8). |
| 10 | The canonical "keep Python state out" patterns | `docs.pytorch.org/.../torch.compiler.disable.html` (fetched); `pytorch/pytorch: torch/_dynamo/config.py:371-382`; torchtitan patterns #10/#11/#14 | `torch.compiler.disable(fn=None, recursive=True, reason=None)` = documented way to run a function (and its callees) outside compilation. `capture_scalar_outputs` (default False) captures `.item()` scalars instead of breaking; `capture_dynamic_output_shape_ops` for `nonzero`/`unique`. torchtitan: "keep counters, bias updates and aux-loss injection in eager" (pattern 10); "Remove host syncs from the training path" (pattern 11); non-traceable kernels → `torch.library.custom_op` with exact fakes (pattern 14). |
| 11 | Q4: no "compile-friendly MoE" paper found | arXiv API queries (2026-10-18) | `all:"MoE" AND all:"torch.compile"` → 0; `all:"mixture-of-experts" AND all:"recompilation"` → 0; `all:"MoE" AND all:"graph capture"` → 0; `all:"mixture-of-experts" AND all:"CUDA graph"` → 3 (the Megatron-MoE paper above + one quantization paper + one EP-inference-failure paper). Closest non-paper: `Victarry/Megatron-MoE-Echo` (search result only): "Fully CUDA-graph-capturable dropless MoE training", sync-free, device-initiated grouped GEMM, `--moe-received-token-capacity`. PyTorch's own MoE blog (2024-11-13, fetched) covers MegaBlocks/dropless/EP/FSDP but says nothing about compile or graphs. |

## 2. Local probe (ours, measured 2026-10-18)

Same torch as the port venv (`torch 2.13.0+cu129`), CPU (`CUDA_VISIBLE_DEVICES=""`),
`torch.compile(..., backend="eager", dynamic=True)`, 12 steps, a 4x4-parameter module
whose `forward` holds the `_banks` pattern (`key = (...)`, `getattr(self,"_cache",None)`,
`if c is None or c[0] != key: ... self._cache = c`). Entries =
`len(torch._dynamo.eval_frame._debug_get_cache_entry_list(fn.__code__))`.

| variant (what moves per step) | entries after 12 steps |
|---|---|
| float attribute in the key (the anneal alpha) | **8** — `recompile_limit` hit; warning reason: `m._cache[0] == (1.7000000000000006,)` |
| plain int module attribute (`self.epoch`) | **8** |
| `p._version` int in the key | 2 |
| stable key (never changes) | 2 (the module-attribute mutation costs one extra entry) |
| no cache at all (rebuild each call) | 1 |
| cache decision in a `@torch.compiler.disable` helper, alpha+`_version` both moving | **1** |

Readings: (a) a per-step **float** read in the key reproduces the port's failure shape
exactly — the guard is the cached-key comparison, so the 8-entry budget is spent in 8
steps, after which dynamo skips the frame and runs eagerly (the 2939 ms/step shape).
(b) The `_version` tuple did **not** recompile per step in this torch/CPU/eager probe —
do not bank on that; it is a guard-set detail that can differ by torch version/backend.
(c) Moving the whole cache decision behind `torch.compiler.disable` keeps the region at
one graph entry while both movers change — the fix works on this exact pattern.

## 3. Options for our code (ranked by risk)

**Option 1 — lowest risk, one decorator: graph-break the cache decision.**
Put `@torch.compiler.disable` on `_banks` itself (`src/dormouse/model/experts.py:556`)
or split the current body into a disabled `_banks_stacked(dt)` and leave `_banks` as a
one-line forwarder. The alpha/`_version`/grad-state reads and the `self._bank_cache`
mutation then run eagerly; the compiled regions at the three call sites (431/690/1008)
consume the bank tensors as inputs. Evidence: probe row 6 (1 entry), PyTorch
"Graph Breaking to Reduce Recompilation Costs", `torch.compiler.disable` docs, and
OLMo-core/Megatron doing the same for permute/grouped-GEMM helpers. Caveat: three graph
breaks split the surrounding inductor regions; the bank build is host-bound anyway
(`experts.py:559-564`), but measure with the existing benches. Bounded extra recompiles
to expect: `requires_grad` is part of dynamo's guards, so train↔eval toggles can add one
entry per mode (torchtitan pattern 16 says the same).

**Option 2 — medium: hoist the refresh into the trainer step, no graph break.**
Add an eager `refresh_moe_banks()` next to `set_factor_qat_alpha`
(`src/dormouse/model/model.py:81-95`), called once per step in `loop.py` where the
anneal is set (~`loop.py:628`), which calls `_banks` under the current grad mode; the
forward path (`experts.py:585-637`) then only reads the stored tuple and asserts it can
serve this call. Keeps the compiled region whole. Evidence: torchtitan pattern 10 (side
effects in eager, compiled region side-effect-free); OLMo-core keeps MoE permute + D2H
outside compile; Megatron keeps expert dispatch eager. Risk: the eval/no-grad reuse rule
must survive (the one-step zero-grad bug documented at `experts.py:574-583`) — a refresh
under `no_grad` must not store an untracked bank for a later tracked step.

**Option 3 — medium-high (only if Option 1's break measures expensive): opaque custom op.**
Wrap the bank accessor in `torch.library.custom_op` with an exact fake (shapes are static:
`[E,r,d]`, `[E,f,r]`, `[E]`) plus autograd registration for the tracked path. Single graph,
no break; more code and a fake to keep honest (torchtitan pattern 14; PyTorch custom-op
docs). Not needed unless the break shows up in a b1 profile.

**Option 4 — not recommended: delete the cache (rebuild every forward).** Probe row 5
says one graph entry and zero Python state, but it reverts two measured cache lessons
(~15 ms host + ~4 ms copies per call at b1, twice per step — `experts.py:559-564`) and
traces the 1024-expert per-expert stack loop into the graph; the spike's 45-minute
`dynamic=False` compile is the warning about what unrolled expert loops cost dynamo.

**Non-fix:** raising `recompile_limit`. PyTorch documents it as legal when the bound is
small and known, but torchtitan treats 8 as a budget ("we aimed for ≤ 6 of 8"; raising it
"costs more compiles and more cached graphs") — it hides, not removes, the re-trace.

After whichever fix: enumerate what is left with `TORCH_LOGS=recompiles`. Candidates in
our forward: `dense_qat_alpha` reads (`experts.py:401,469` — only when `dense_qat !=
"none"`; default is `"none"`, `src/dormouse/config.py:374`), the train/eval toggle,
and `Counters` mutation.

**Q5, directly:** applicable found patterns = (1) `torch.compiler.disable` region,
(2) hoist state out and pass tensors in, (3) custom op if the break is expensive.
The smallest change to the cache-key design that makes the compiled region static is
**not a key change**: any Python value read inside the compiled function is guarded on
its exact value, so the key must move out of the compiled region; that is one decorator
line at `experts.py:556` (Option 1), verified by the probe.

## 4. VERIFIED vs SPECULATION

- **VERIFIED (fetched sources)**: every row of §1, with the URLs in §5; the Megatron
  docs quotes, the torchtitan/OLMo-core file contents, the PyTorch docs statements.
- **VERIFIED (ours, this machine)**: the §2 probe table and the reproduced
  `recompile_limit` warning (torch 2.13.0+cu129, CPU, eager backend, 12 steps,
  2026-10-18, port venv; script shape in §2). The `_banks` code facts (lines above).
- **SPECULATION**: that Option 1 restores the port's b1 fast path (not run here — needs a
  GPU run with `TORCH_LOGS=recompiles` before/after); that the anneal alpha is the mover
  in the task's failing run (if the run had `factor_qat`/`dense_qat` annealing on, the
  probe says yes; if not, re-run the probe against `_banks` itself); that the three graph
  breaks cost little (must be measured).

## 5. Confidence

**High** on the mechanism (per-step Python values in the compiled region → guard failure →
recompile → limit → eager fallback) and on the fix class (`torch.compiler.disable` /
hoisting), because it is multi-source (PyTorch docs, torchtitan, OLMo-core, Megatron) and
reproduced locally on the same torch. **Medium** on Option 1 vs Option 2 for our b1 shape
— that is a measurement, not a research question.

## 6. Sources (status as fetched 2026-10-18)

Fetched successfully (HTTP 200):
- pytorch/torchtitan @main (raw.githubusercontent.com): `.claude/skills/local_compile_perf/SKILL.md`,
  `.claude/skills/local_compile_perf/references/{patterns,compile-limits,measurement}.md`,
  `torchtitan/distributed/local_compile.py`, `torchtitan/experiments/graph_trainer/{compile,cuda_graph}.py`,
  `torchtitan/models/common/moe/moe.py`, `tests/unit_tests/gpu/models/common/test_moe_aux_loss_compile.py`
- allenai/OLMo-core @main: `CHANGELOG.md`, `src/olmo_core/nn/transformer/model.py`,
  `src/olmo_core/train/train_module/transformer/config.py`,
  `src/olmo_core/nn/moe/v2/{routed_experts,no_ep}.py`, `allenai/OLMo-core/docs/source/guides/all_in_one_for_researchers.md`
- NVIDIA/Megatron-LM: `NVIDIA/Megatron-LM/docs/user-guide/features/cuda_graph.md`, `megatron/core/transformer/moe/README.md` (via search results), `megatron/core/transformer/cuda_graphs.py` (excerpt via search)
- arXiv: `2603.07685v2` (Scalable Training of MoE with Megatron Core), `2412.19437v1` (DeepSeek-V3 report)
- deepseek-ai/DeepGEMM README; deepseek-ai/DeepEP README (no graph hits)
- docs.pytorch.org: `.../programming_model.recompilation.html`, `.../torch.compiler.disable.html`;
  pytorch/pytorch `torch/_dynamo/config.py`
- pytorch.org/blog/training-moes/ (2024-11-13)
- docs/research/2026-10-15-torch-spike.md (our own prior measurement)

Search-result only (not fetched directly): `NVIDIA/Megatron-LM` PR #1919; `Victarry/Megatron-MoE-Echo`; NVIDIA "Transformer Engine and Megatron-LM CUDA Graph Support" docs page. Failed: `pytorch/torchtitan/docs/compile.md` and `pytorch/torchtitan/docs/moe.md` (404 — the compile material lives in the skill dir and `experiments/graph_trainer/`).
