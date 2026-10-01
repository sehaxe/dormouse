# Adopt vs. port: is there a credible external implementation for each of our 28 port crates?

Date: 2026-09-27. Slice owner: adopt-vs-port subagent (research + code reading).
Nothing in the repo was modified except this file. Nothing was committed. No GPU work.

**The question.** Not "is our port good" but: *for each crate, did the original authors
ship code, and if so should we be calling it instead of maintaining a transcription of it?*
Two failure modes to avoid: hand-rolling what an accessible library already does well, and
adopting a library that cannot run here and costs more to integrate than the port it replaces.

**How to read the verdicts.**

| verdict | meaning |
|---|---|
| `ADOPT` | call it at runtime — reachable on our stack and better than the port |
| `REFERENCE` | do not call it, but verify against its source instead of a hand transcription |
| `PORT-STANDS` | nothing better exists, or the alternative is not reachable here |
| `DELETE` | nothing external exists **and** the crate is not worth keeping (cross-referenced against `docs/archive/research/2026-09-27-fused-inventory-attention.md` (and its two siblings)) |

**Evidence rule.** Primary sources only. Every URL in the per-crate section was opened during
this pass unless the row says `NOT VERIFIED`. The complete list of what was opened is in
Appendix A. Where I could not reach something, the row says so rather than reasoning from
memory.

**Stack constraints used as the yardstick** (owner-measured on this box, taken as given, not
re-measured here): RTX 5060 Ti / sm_120 consumer Blackwell / 16 GB; burn 0.22.0-pre.4 with
vendored cubecl; FLA's Triton kernels need Python + a sidecar and lose on a 14 ms/call PCIe
floor against a 1.35 ms GPU budget, plus open Blackwell `tl.dot` state-corruption issues
(#945/#953) and hanging backward configs (#999/#1000); no tcgen05 / TMEM / WGMMA on this part;
cuBLAS via cudarc measured at 43.7 TFLOP/s f16-in/fp32-accumulate; an upstream Rust crate on
crates.io is reachable and cheap.

---

## 0. Corrections to the brief, from the crates' own source

Checked against each crate's own doc comment and `Cargo.toml`, not the task list:

- The library is **28 crates, not 26** (`vendor/dormouse-fused/crates/`), plus the `burn-fused`
  meta-crate. The task list names 28 and matches the directory exactly.
- `burn-gdn2` has **no arXiv in the crate** (correct as stated), but the mechanism it ports
  **does** have a paper that the crate never cites: **arXiv 2605.22791**, "Gated DeltaNet-2:
  Decoupling Erase and Write in Linear Attention" (NVlabs). Its `tests/gen_reference.py` cites
  the *repo* (`NVlabs/GatedDeltaNet-2`) but not the paper.
- `burn-byteflow` is **2603.03583** (ByteFlow, Deng et al., ICLR 2026) — not in the task list.
- `burn-diffusionblocks` is **2506.14202** (DiffusionBlocks, Sakana AI, ICLR 2026) — not in
  the task list.
- `burn-jepa` cites a **third** paper not in the task list: **2304.07193** (DINOv2 / KoLeo).
- `burn-ttt` cites **2407.04620** (original TTT) alongside 2512.23675.
- `burn-rope` cites **2104.09864** (RoFormer) and **2309.00071** (YaRN) — neither in the task list.
- `burn-spectral` cites **seven** papers, of which 2412.04787 (stochastic-rounding DQT) and
  2202.09368 (MoE expert-choice routing) were not in the task list. All seven verified real.
- `burn-sct` has no arXiv in its doc comment at all; it is the 2604.00733 mechanism.

All 31 arXiv IDs cited across the 28 crates were opened on arxiv.org and confirmed to exist
with the title the crate claims. **Zero invented papers in the crate docs.**

---

## 1. The fact that dominates every verdict: most of the library is not reachable by anything

Measured from `crates/*/Cargo.toml` and `grep -rhoE "\bburn_(kda|...)"` over `crates/*/src`:

| | crates | src LOC |
|---|---|---|
| **Wired** (a `use burn_*` in `crates/dormouse-*`) | 9 | **14,988** |
| Wired transitively (`burn-gdn2` via `burn-kda`) | +1 | included above |
| **Zero call sites anywhere in the workspace** | 18 | **12,597** |

The nine actually-referenced crates, by reference count in our source: `burn_spectral` 22,
`burn_muon_plus` 4, `burn_rmsnorm` 3, `burn_mor` 3, `burn_bitnet` 3, `burn_kda` 2,
`burn_dspark` 2, `burn_jepa` 1, `burn_engram` 1.

**46% of the library has zero call sites.** This is the same conclusion the three inventory
files reached from the other direction (WIRED / IMPLEMENTED-UNUSED / SUPERSEDED / BROKEN), so
it is not a new finding — but it is the frame for the adopt question, because *a port with no
caller has no correctness risk to remove and no speed to gain.* Adopting an external
implementation for an unwired crate is strictly worse than not adopting it: you would be
paying integration cost for something nothing calls. The adopt lane is therefore short, and
that is the finding, not a failure to search.

---

## 2. Per-crate verdicts

### ADOPT (2)

---

#### `burn-rmsnorm` — 235 LOC — **ADOPT `burn::nn::RmsNorm`**

1. **External implementation?** Yes, and it is **already a dependency**. `burn-nn`
   0.22.0-pre.4 ships `RmsNorm` / `RmsNormConfig` at
   `burn-nn-0.22.0-pre.4/src/modules/norm/rms.rs` (147 lines, opened locally in the cargo
   registry cache). Semantics `Y = X / sqrt(mean(X²)+eps) * gamma` — identical to ours.
2. **Can it run on our stack?** Yes, trivially: it is `burn::nn`, already compiled into the
   trainer's binary, generic over `B`. It is pure tensor ops, which is what our code runs
   anyway.
3. **Better reference?** Yes, in the strongest sense available — it is *the same project's*
   implementation, so there is no second-source question at all.
4. **Verdict: ADOPT.**

**The one real risk, checked and cleared.** `burn-rmsnorm`'s own comment records a
first-hand burn bug: *"pre.3 required require_grad here (Param::initialized inherits the flag;
without it the weight froze — model_seam gradient_flow, 2026-09-21)"*. If that still applied,
naively swapping to `burn::nn::RmsNorm` would silently freeze every norm weight. I traced it:
`burn-nn-0.22.0-pre.4/src/modules/norm/rms.rs:32` uses `Initializer::Ones.init(...)` with no
`require_grad()` call, but the `Module` derive materialises `Param<Tensor<D>>` fields through
`Param::from_tensor`, which at `burn-core-0.22.0-pre.4/src/module/param/tensor.rs:129-133`
wraps with `set_require_grad(value, true)`, and `set_require_grad` is documented at
`burn-core-0.22.0-pre.4/src/module/param/base.rs:244` as a no-op on non-autodiff devices.
`burn-core`'s own `test_module_val_train_stateful` asserts exactly this. **The freeze was a
pre.3 bug that upstream fixed; our 8-line device-branching workaround is now dead weight.**

And our port's only differentiator is gone anyway: the inventory's universal blocker applies
to it — the fused cube is gated on `TypeId::of::<Inner>() == TypeId::of::<CudaBare>()`, and
the trainer's backend is `Autodiff<Cuda, BalancedCheckpointing>`, so **the kernel in
`burn-rmsnorm` has never launched in production**. The crate is named for a fusion that does
not fire. Adopting `burn::nn::RmsNorm` makes the name match the behaviour.

---

#### `burn-rope` — 975 LOC — **ADOPT `burn::nn::RotaryEncoding`**

1. **External implementation?** Yes, already a dependency. `burn-nn`
   0.22.0-pre.4 ships `RotaryEncoding` / `RotaryEncodingConfig` at
   `src/modules/rope_encoding.rs` (643 lines). Crucially it exposes
   `init_with_frequency_scaling(f)` — "useful to apply different RoPE extensions" — which is
   the YaRN hook, so the port's headline extension is **subsumed by an API upstream already
   provides**, not reimplemented.
2. **Can it run on our stack?** Yes — same `burn::nn` generic-over-`B` situation.
3. **Better reference?** Yes. The inventory singles out `burn-rope` as holding the library's
   best numerical discipline *and* its only admitted precision bug: an inverted YaRN ramp once
   shipped with a green test suite because an f32 `cos` collapses to 1.0 below ~3e-4 rad. That
   is 975 lines of ours containing a real, documented precision failure mode. The frequency
   math is the part that needs re-verification; the part that does not (the elementwise
   rotation) is where the value is.
4. **Verdict: ADOPT** — for the frequency math, the tests, and the YaRN API.

**Honest caveat, stated because it cuts against the verdict.** `burn-rope` owns the library's
only genuinely good fused autodiff op: `rope_autodiff<Inner>` is a real custom op with a
forward `rope_kernel`, a backward `rope_backward_kernel`, registered through
`burn_autodiff::ops::Backward`, and no matmul anywhere so it is structurally immune to all
three of our precision blockers. `burn::nn::RotaryEncoding` has no fused path. **If RoPE is
ever wired into the attention arm, do not throw the kernel away** — take burn-nn's frequency
table, YaRN scaling and test discipline, and keep `rope_cuda.rs`'s two kernels. Because
`burn-rope` currently has **zero call sites**, the decision is free today: delete 975 lines of
unverifiable frequency math, and if the kernel is ever needed, recover it from git.

---

### REFERENCE (10)

`REFERENCE` here means: the authors' code exists, we cannot call it, and it is a better
oracle than what we currently compare against. In 8 of these 10 cases the crate's own doc
comment already names the upstream project — the problem is not discovery, it is that nobody
ever wired a test to it.

---

#### `burn-kda` — 1232 LOC, **WIRED (2 call sites, the only attention arm)** — **REFERENCE**

1. **External implementation?** **Yes, and it is the strongest artifact found in this whole
   pass.** `MoonshotAI/FlashKDA` (1263★, Cuda) — *"FlashKDA: high-performance Kimi Delta
   Attention kernels"* — opened via the GitHub trees API and `raw.githubusercontent.com`.
   29 files, 2.7 MB. Contains `csrc/flash_kda.cpp`, `csrc/smxx/fwd_kernel1.cuh` (26 KB),
   `csrc/smxx/fwd_kernel2.cuh` (41 KB), `csrc/smxx/fwd_launch.cu`, `csrc/smxx/utils.cuh` —
   a complete CUTLASS-based fused KDA kernel. Also `docs/20260420-flashkda-v1-deep-dive.md`,
   a design document that answers exactly the questions our port had to guess at
   (`CHUNK = 16` because `exp(cumsum(g))` stays inside bf16 range at the `g_min = -5` floor;
   cheap 16×16 inversion by forward substitution; and an **SM80-only MMA path**).
   Note: `MoonshotAI/Kimi-Linear` (1613★, the paper's own repo) ships **no code** — 8 files,
   all PDF/PNG. The arXiv comment for 2510.26692 is just "Kimi Linear tech report".
2. **Can it run on our stack?** **No, and I checked the specific reasons rather than assuming
   them.** `setup.py` declares `SUPPORTED_CUDA_ARCHS = ["90a", "100a", "103a", "120a"]` —
   **it does list our `120a`**, and the deep-dive confirms the CHUNK=16 math is an SM80 MMA
   path, not tcgen05. So the arch is *not* the blocker, which is the opposite of my prior.
   The blockers are: (a) it is a PyTorch C++ extension — `torch.utils.cpp_extension` +
   `torch.ops`, no C ABI, so cudarc cannot load it without libtorch; (b) a Python sidecar
   blows the 14 ms/call PCIe floor against a 1.35 ms budget; and decisively (c) **it is
   forward-only.** Every single file is `fwd*` — `fwd_kernel1.cuh`, `fwd_kernel2.cuh`,
   `benchmarks/bench_fwd.py`, `tests/test_fwd.py`, `tests/test_fwd_full.py`. The README says
   to call `chunk_kda` under `torch.inference_mode()`. **There is no backward kernel**, and
   `burn-kda` is wired into the *training* loop. FlashKDA could never be our training kernel.
3. **Better reference?** **Yes, and this is the headline of the report.** FlashKDA ships
   `tests/torch_ref.py` (10 KB, opened) — a PyTorch reference written *specifically to match
   the kernel bit-for-bit*. It reproduces the warp-shuffle tree-reduction order
   (`x_f32.reshape(..., 16, 8)` with a 16-way partial sum), fp32 FMA behaviour
   (`fp32_fma` via a float64 intermediate), `exp2` with flush-to-zero below the f32 tiny
   normal, and a `tanh.approx.f32` sigmoid. **A hand transcription of a paper is a much
   weaker oracle than that.** See the ranked oracle list for the cost.
4. **Verdict: REFERENCE.**

---

#### `burn-gdn2` — 2219 src + 3296 test LOC, **WIRED as burn-kda's engine** — **REFERENCE**

1. **External implementation?** **Yes, official, and already half-used.**
   `NVlabs/GatedDeltaNet-2` (opened: README + full trees API) — *"Official PyTorch
   implementation of Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention"*,
   arXiv 2605.22791, Hatamizadeh / Choi / Kautz. 26 files, 1.1 MB. Contains
   `lit_gpt/gdn2.py` (the layer) and **`lit_gpt/gdn2_ops/chunk_gdn2.py` +
   `lit_gpt/gdn2_ops/fused_recurrent_gdn2.py`** (the Triton kernels). Pure Python + Triton.
2. **Can it run on our stack?** **No.** Python + Triton, so the same sidecar/PCIe-floor and
   Blackwell-`tl.dot` problems as FLA. Not a runtime candidate.
3. **Better reference?** **Partly already, and the remaining gap is precisely named.** This
   is the **only crate in the library with a real bit-exact harness**:
   `tests/gen_reference.py` (8.4 KB, opened) states it is "a faithful, pure-PyTorch
   transcription of the official NVlabs GatedDeltaNet-2 layer
   (`https://github.com/NVlabs/GatedDeltaNet-2`, lit_gpt/gdn2.py), with the Triton
   `fused_recurrent` kernel replaced by an equivalent per-token scan in torch", emits 1000
   cases into `tests/ref_data.bin` (7.1 MB, present), and `tests/bit_exact.rs` compares
   against it. **So the oracle is one indirection weaker than the authors': we transcribed
   their layer, then re-implemented their kernel's recurrence ourselves.** The upgrade is to
   compare against the Triton source's actual recurrence.
4. **Verdict: REFERENCE** — and the cheapest oracle upgrade in the library, because the
   harness already exists and only the reference body changes. Also: **add the missing
   arXiv 2605.22791 to the crate doc.**

---

#### `burn-engram` — 586 LOC, **WIRED (1 call site)** — **REFERENCE**

1. **External implementation?** **Yes, official, but thin.** `deepseek-ai/Engram` (4701★,
   opened: README + trees API + `engram_demo_v1.py`). 10 files, 2 MB — README, `Engram_paper.pdf`,
   figures, a drawio, and **one** Python file: `engram_demo_v1.py`. Opened: it is a real
   implementation of the Engram module (`EngramConfig`, `CompressedTokenizer`, the
   per-ngram/per-head hash table build) and its own docstring says "Standard components
   (Normalization, Attention, MoE) and complex Hyper-connection mechanisms are omitted or
   mocked ... to focus exclusively on the Engram module implementation." That disclaimer is
   exactly the right scope for us.
2. **Can it run on our stack?** No — Python/HF-tokenizers. And it is a *demo*, not a
   production reference.
3. **Better reference?** **Yes, with one caveat.** The crate's hash/gate assertions are
   hand-transcribed formula checks (per the inventory: "the closest are hand-transcribed
   formula assertions"). `engram_demo_v1.py` is the authors' own addressing and read
   construction. **Caveat:** the file imports `transformers` and `sympy` and drives a real
   tokenizer, so a parity test needs a tokenizer fixture — the separable core is the
   n-gram key construction and the lookup-table build, not the tokenizer.
4. **Verdict: REFERENCE.**

---

#### `burn-dspark` — 865 LOC, **WIRED (2 call sites)** — **REFERENCE**

1. **External implementation?** **Yes, official, and full-stack.**
   `deepseek-ai/DeepSpec` (7167★, Python) — *"DeepSpec: a full-stack codebase for training
   and evaluating speculative decoding"*, opened via README + trees API. 92 files, 9.9 MB.
   The DSpark implementation lives in `deepspec/modeling/dspark/`: `loss.py` (11 KB),
   `markov_head.py` (11 KB), `common.py` (9 KB), plus `gemma4/modeling.py` (22 KB) and
   `qwen3/modeling.py` (20 KB); inference in `deepspec/eval/dspark/{confidence_head.py,
   draft_ops.py, evaluator.py}`. There is even `assets/dspark.drawio`.
2. **Can it run on our stack?** No — Python + HF transformers. Sidecar PCIe floor.
3. **Better reference?** **Yes, and the crate already claims to be using it.** The doc
   comment says "matched against the official DeepSpec implementation" and quotes the loss
   formula — but "matched against" is a *claim*, not a test. There is no parity harness. The
   loss has five weighted terms (`w_k = exp(-k/γ)`, `L_ce`, `L_tv` L1, `L_conf` BCE, and γ
   itself) and a γ-scaling or L1-vs-probability mistake would be invisible to a
   self-consistency test.
4. **Verdict: REFERENCE** — turn the existing claim into an actual parity test.

---

#### `burn-mhc` — 1011 LOC, **zero call sites** — **REFERENCE**

1. **External implementation?** **Yes — I nearly concluded otherwise, and was wrong.** I
   checked the obvious places and found nothing: `deepseek-ai`'s 39-repo listing has no mHC
   repo, and I grepped `DeepSeek-V3.2-Exp/inference/model.py` (38 KB, opened, 923 lines) for
   `sinkhorn|hyper.?conn|mhc` — **zero hits**; its class list is `ModelArgs, ParallelEmbedding,
   Linear, ColumnParallelLinear, RowParallelLinear, RMSNorm, LayerNorm, Indexer, MLA, MLP,
   Gate, Expert, MoE, Block, Transformer` — no hyper-connections. *Then* I read
   `deepseek-ai/DeepGEMM` (7875★, Cuda) fully, and its README's first line lists the
   primitives it ships: **"GEMMs (FP8, FP4, BF16), fused MoE ..., MQA scoring for the
   lightning indexer, HyperConnection (HC), and more."** Its tree contains
   `csrc/apis/hyperconnection.hpp`, `csrc/apis/mega_mhc.hpp`,
   `csrc/jit_kernels/impls/sm90_tf32_hc_prenorm_gemm.hpp`,
   `csrc/jit_kernels/impls/sm100_tf32_hc_prenorm_gemm.hpp`,
   `csrc/jit_kernels/impls/sm100_mega_mhc.hpp`, `tests/test_hyperconnection.py`,
   `tests/test_mega_mhc.py`, and — the important one —
   **`third-party/tilelang_ops/ref_mhc.py`**. DeepSeek's own mHC reference and fused kernel
   exist, inside DeepGEMM.
2. **Can it run on our stack?** **No, decisively, and for a checkable reason.** DeepGEMM's
   Requirements section says: *"NVIDIA SM90 or SM100 architecture GPU"*, plus Python ≥3.8,
   CUDA ≥12.9, PyTorch ≥2.3, CUTLASS ≥4.0, C++20. **sm_120 is not in that set** — this is the
   exact wall our stack hits elsewhere (no tcgen05/TMEM/WGMMA). So: not a runtime candidate,
   ever, on this box.
3. **Better reference?** **Yes, and it closes a real hole.** Per the inventory, burn-mhc is
   "IMPLEMENTED-UNUSED — best verified, zero used", i.e. verified against hand-written
   formula assertions. It has *no* external oracle today. `ref_mhc.py` plus
   `hyperconnection.hpp` is DeepSeek's own reference for the mechanism, including the
   Sinkhorn–Knopp Birkhoff projection that is the whole point of the paper.
4. **Verdict: REFERENCE.**

---

#### `burn-bitnet` — 1437 LOC, **WIRED (3 direct + 5 indirect)** — **REFERENCE**

1. **External implementation?** **Yes, official, and unusually well-suited as an oracle.**
   `microsoft/BitNet` (40347★, C++) — *"Official inference framework for 1-bit LLMs"*, opened
   via trees API. 81 files. `gpu/model.py` (11 KB) is the PyTorch b1.58 reference (BitLinear,
   absmean quantizer); `gpu/bitnet_kernels/bitnet_kernels.cu` + `.h`; `src/ggml-bitnet-mad.cpp`
   (42 KB) is the ternary matmul; `include/bitnet-lut-kernels.h` (73 KB) is the generated
   lookup-table kernel. Also `AAzdi/Sparse-BitNet` (README opened) is the official code for
   2603.05168 with "Triton-based fused cross-entropy and sparsity mask creation" and
   `llm/arch/model.py` carrying `SparseLinear` — i.e. the N:M Dual-STE half our crate claims.
2. **Can it run on our stack?** **Not usefully.** It is an *inference* framework: no backward,
   no STE, no autodiff. And while it is C++/CUDA (so it does not need Python to be
   *read*), its kernels are generated for specific presets and it is not a library with a
   callable ABI. `Sparse-BitNet` is Triton + PyTorch, so it is out on both counts.
3. **Better reference?** **Yes** — `gpu/model.py` is the paper's own forward, and
   `Sparse-BitNet/llm/arch/model.py` is the N:M mask/Dual-STE reference. Our port's
   quantizers are 6 lines of arithmetic each; the fidelity gain is mostly in `bitnet_v2`'s
   Hadamard ordering and the N:M mask convention, which are the parts that are easy to get
   subtly wrong.
4. **Verdict: REFERENCE.**

---

#### `burn-sct` — 2738 src + 259 test LOC, **zero call sites, SUPERSEDED by burn-spectral** — **REFERENCE**

1. **External implementation?** **Yes, official.** `EctoSpace/SCT` — the URL in arXiv 2604.00733's
   own comment field ("Code at https://github.com/EctoSpace/SCT"). Opened: 44 files, 4.1 MB;
   `spectral_compact_training/spectral_layer.py` is the reference, with `docs/SCT_Patent_Application.pdf`,
   a convergence report, a rank sweep, and results for A100 / Mac / **SteamDeck**. (The
   *org* endpoint `api.github.com/orgs/EctoSpace` 404s — the user is not an org. The repo is
   fine; don't be misled by that.)
2. **Can it run on our stack?** No — Python/PyTorch, built around a full nanoGPT-scale
   training codebase.
3. **Better reference?** **Yes, and the crate is already aiming at it.**
   `burn-sct/tests/cmp_reference.rs` exists; the upgrade is to point it at the authors'
   `spectral_layer.py` rather than our own transcription.
4. **Verdict: REFERENCE** (for the oracle) — but note the crate itself is superseded: with
   EctoSpace/SCT existing, the `DELETE` condition ("nothing external exists") does not hold,
   so it is `REFERENCE`, not `DELETE`. The *port* is dead code either way; only the test
   target is worth keeping.

---

#### `burn-jepa` — 443 LOC, **WIRED (1 call site; 1/3 of the crate unwired)** — **REFERENCE**

1. **External implementation?** **Split.**
   - **KoLeo (2304.07193):** `facebookresearch/dinov2` (13377★) exists — probed and confirmed
     live (last push 2026-06-03). KoLeo is `koleo_loss` in DINOv2.
   - **LeJEPA (2511.08544):** `rbalestr-lab/lejepa` — the paper's own arXiv comment names it.
   - **data2vec 2.0 (2212.07525):** **NOT VERIFIED.** I probed
     `facebookresearch/data2vec`, `facebookresearch/data2vec-2.0` and
     `facebookresearch/data2vec2` — **all three 404** through the GitHub API, and the
     arXiv comment field is empty. I did not find an official home for the 2.0 code and am
     not going to guess one. `facebookresearch/fairseq` (32219★) exists and is the historical
     home of data2vec 1.0, but I did not open its `data2vec` module to confirm 2.0 is there.
2. **Can it run on our stack?** No — all Python.
3. **Better reference?** For KoLeo, yes and cheaply: DINOv2's loss is a dozen lines. For
   data2vec 2.0, **NOT VERIFIED** — see above. Note the inventory's own better finding:
   `burn-jepa/src/losses.rs` documents three reproducible backend landmines and that is worth
   more than an oracle.
4. **Verdict: REFERENCE**, with the data2vec leg explicitly unverified.

---

#### `burn-fastblt` — 513 LOC, **zero call sites** — **REFERENCE**

1. **External implementation?** **Yes, official, and the crate already names the file.**
   `facebookresearch/blt` (opened: README + trees API) — *"This repository contains code for
   our paper"*, 112 files. `bytelatent/data/patcher.py` (the entropy patcher the crate claims
   to match), `bytelatent/data/ngram_processor.py` (the rolling-hash n-gram embedding), and
   `bytelatent/float8.py`. Note: arXiv 2412.09871's comment field is **empty** — the repo is
   not advertised on the paper page; it had to be found by name.
2. **Can it run on our stack?** No — Python/PyTorch.
3. **Better reference?** **Yes, cheaply, and the crate already asserts the match without
   proving it.** `patcher.py` is a short pure-Python file; asserting entropy-threshold
   boundary cases against it is a ~60-line script. **But the value is capped**, because the
   inventory is right that Engram strictly supersedes FastBLT on both addressing axes and
   dormouse currently uses *neither* hasher (FNV-1a-32 in `dormouse-data/src/lib.rs:407`).
4. **Verdict: REFERENCE** — the oracle exists and is cheap, but spend it only if the crate is
   ever revived. Per the inventory it would not have served better than the Engram.

---

#### `burn-eggroll` — 327 LOC, **zero call sites** — **REFERENCE**

1. **External implementation?** **Yes, official.** arXiv 2511.16652's comment gives only
   "Website at https://eshyperscale.github.io/", so I followed that (opened). It links
   `github.com/ESHyperscale/HyperscaleES` (375★, *"Jax Codebase for Evolutionary Strategies
   at the Hyperscale"*, probed live) and `ESHyperscale/nano-egg` (184★). Both confirmed.
2. **Can it run on our stack?** No — JAX.
3. **Better reference?** **Yes, and it settles a documented open question in our own code.**
   The normalising constant is exactly where our two ES crates disagree: `burn-eggroll`'s
   doc says the paper uses the normalised `(σ/√r)·A·Bᵀ` (entry std ≈ σ), while `burn-es`'s doc
   says its own convenience `eggroll_mutate` "scales as `σ·A·Bᵀ` (entry std ≈ σ·√rank)" and
   tells the reader to prefer `burn-eggroll` for real ES loops. **That is an unresolved
   factor-of-√r between two of our crates, and the authors' source settles it.**
4. **Verdict: REFERENCE** — read the source, settle the σ convention, delete one of the two.

---

### PORT-STANDS (10)

Ten crates where the honest answer is that the port is fine. Two of them have no external
implementation *because the authors shipped none*, which is a stronger statement than "the
library is better".

---

#### `burn-attnres` — 2060 LOC, **zero call sites, BROKEN** — **PORT-STANDS**

1. **External implementation? NO — and this is verified, not assumed.** `MoonshotAI/Attention-Residuals`
   (3516★, the paper's own org, the highest-starred repo in this entire pass) contains
   **6 files, 1.78 MB: `README.md`, `Attention_Residuals.pdf`, and four PNGs.** No code. The
   README's own framing is "This is the official repository for Attention Residuals", and
   what it ships is the paper. arXiv 2603.15031's comment is just "attnres tech report".
2. **Can it run on our stack?** Moot — there is nothing to run.
3. **Better reference?** No stronger one exists. The best available is the paper PDF plus our
   own reading of it. The inventory independently found our fused path BROKEN
   (`streaming_fused_matches_tensor_path`, `maxdiff 0.83`).
4. **Verdict: PORT-STANDS** — and note the consequence: with zero wiring, a BROKEN kernel, and
   no upstream to diff against, this crate is 2060 lines that nobody can check and nobody
   calls. It is the strongest `DELETE` candidate in the library; I am filing it as
   `PORT-STANDS` because the DELETE criterion is "nothing external exists **and** not worth
   keeping", and whether it is "worth keeping" is a research-priority call that belongs to the
   owner, not to me. **If Attention Residuals is not on the roadmap, delete it — there is
   nothing to re-verify and nothing running.**

---

#### `burn-mor` — 744 LOC, **WIRED (3 call sites), BROKEN** — **PORT-STANDS**

1. **External implementation?** **Yes, official** — `raymin0223/mixture_of_recursions`, named
   in arXiv 2507.10524's own comment ("codes at https://github.com/raymin0223/mixture_of_recursions")
   and confirmed by opening the README (Bae, Kim, Bayat, Kim, Ha, Schuster, Fisch, Harutyunyan,
   Ji, Courville et al., NeurIPS 2025).
2. **Can it run on our stack?** No — PyTorch.
3. **Better reference?** Marginally. The MoR paper's contribution is the *training recipe*
   (expert-choice routing, hierarchical filtering), and the routing core is a top-k + gather
   + scatter — which the inventory found is a **verbatim duplicate of `burn-mod`**, and
   `burn-mod` in turn duplicates what burn already has. The oracle would mostly be a
   duplicate-oracle.
   Separately, the top-k *kernel* has a real external candidate I checked:
   **`deepseek-ai/DeepSelect`** (392★, Cuda, opened) — *"a high performance implementation of
   the TopK kernel used in DeepSeek Sparse Attention"*, 2–20× over `torch.topk`, with a
   deep-dive doc. It is CUDA C++ (needs PyTorch to build/run) so it is not adoptable here, and
   we already run a patched `cubek-reduce` top-k (ADR-0015) because upstream 0.3.0-pre.4's
   `reaches` guard emits `u32::MAX` as an index. Not worth churning.
4. **Verdict: PORT-STANDS** — but it does not currently compile (`E0599: no method named
   int` ×2 in `src/topk_gather.rs`, per the inventory), while being wired 3×. **Fix or unwire;
   that is more urgent than anything in this report.**

---

#### `burn-muon-plus` — 1055 src + 327 test LOC, **WIRED (4 call sites), the default optimizer** — **PORT-STANDS**

1. **External implementation?** **Yes, official.** `K1seki221/MuonPlus`, named in the paper
   body itself (2602.21545: "We provide our code here: https://github.com/K1seki221/MuonPlus"),
   confirmed by opening the README and the trees API: 29 files, 3.2 MB, YAML configs
   (`llama60M_row-col.yaml` etc.), `utils/optim`, and it uses `polar_method: Keller` +
   `rms_scaling: True` — a different polar iteration from the quintic NS coefficients we use.
3. **Can it run on our stack?** No — PyTorch/torchtitan-style.
4. **Better reference?** **Weakly — and honestly, barely at all.** The paper ships its update
   rule as a 17-line Python function (`muon_plus_step` + `norm_dir`, Algorithm 1) which I read
   in full. **When the specification *is* the pseudocode, reimplementing the pseudocode is not
   a weaker oracle — it is the same oracle.** The only thing the repo adds is the `Keller`
   polar method, which is a variant choice, not a fidelity question.
   **Counter-evidence worth carrying:** `KellerJordan/modded-nanogpt` discussion #239 reports
   an independent implementation of Muon+ (`col_row`) producing a **0.005 loss *increase***
   versus baseline, and a maintainer reply speculating why. Reported, not reproduced by me.
5. **Verdict: PORT-STANDS.** Also checked the crates.io lane for an upstream Rust Muon:
   `burn-optim` 0.22.0-pre.4 ships AdaGrad/Adam/AdamW/Adan and no Muon; a crates.io search
   for "muon optimizer" returns only whole-framework crates (`ferrotorch`, `rlx-optim`,
   `trustformers-optim`) and nothing credible. **There is nothing to adopt.**

---

#### `burn-spectral` — 6172 src LOC (the largest crate), **WIRED 22×, BROKEN** — **PORT-STANDS**

1. **External implementation?** **Yes** — `EctoSpace/SCT`, same repo as above, the paper's own
   named code link. Python.
2. **Can it run on our stack?** No.
3. **Better reference?** Yes for the *mechanism* (`spectral_layer.py`), and the crate already
   has a `polar_orthogonalize` whose NS-3 behaviour we measured ourselves. The GPU Newton–Schulz
   polar retraction has no external Rust equivalent; upstream does it in Python/NumPy.
4. **Verdict: PORT-STANDS** — the largest crate, the most wired, and per the inventory
   **33/36 tests fail** with a `set_require_grad` panic and a test target that is unbuildable
   from clean. Its size is not a reason to adopt anything; it is a reason to fix it.

---

#### `burn-ttt` — 125 LOC, **zero call sites** — **PORT-STANDS**

1. **External implementation?** **Yes, official, named in the paper's comment**
   (`test-time-training/e2e`), README opened. But the crate only implements the **two loss
   functions** from `ttt/model/loss.py` and its own doc admits the matching. There is no inner
   loop and no state, so there is nothing for an oracle to disagree about.
2. **Verdict: PORT-STANDS.** The inventory's phrasing is the right one: "a name without a
   mechanism."

---

#### `burn-situ` — 675 LOC, **zero call sites** — **PORT-STANDS** (no upstream code exists)

1. **External implementation? NO — verified.** SiTU-GLU is from Kimi K3 (2607.24653), and
   `MoonshotAI/Kimi-K3` (8863★) contains **4 files, 1.9 MB: `LICENSE`, `README.md`,
   `assets/kimi-logo.png`, `k3_tech_report.pdf`.** No code. arXiv 2607.24653's comment is
   "K3 tech report". The best available source is the PDF's Eq. 12.
2. **Better reference?** No stronger one exists. The only reachable artifact is the
   technology report PDF.
3. **Verdict: PORT-STANDS.** Worth recording that the inventory calls this *"the only fused
   CUDA kernel in this slice"* and *"the highest-value unadopted thing in the slice"* — a
   working, verified, unwired kernel with no upstream to check it against. That is a
   research-priority call, not an adopt/port call.

---

#### `burn-parcae` — 317 LOC, **zero call sites** — **PORT-STANDS**

1. **External implementation?** **Yes, official, and unusually well-packaged.**
   `SandyResearch/parcae` (opened; project page `sandyresearch.github.io/parcae` and
   Together AI's blog also opened), and — the notable find — **it is on PyPI as
   `parcae-lm`**, with a homepage/repository link. Built on `karpathy/nanochat`,
   `seal-rg/recurrent-pretraining` and `Lightning-AI/litgpt`. Ships HF checkpoints
   (`SandyResearch/parcae-770m`).
2. **Can it run on our stack?** No — Python, though it is a `pip install` away and cheap to
   *run* (on CPU or the same GPU) as an experiment.
3. **Better reference?** Yes, and the paper's Appendix P/E carry the exact model definition
   the crate transcribes. Low marginal value: the mechanism is 4 lines of parameterization.
4. **Verdict: PORT-STANDS** — but the honest note from the inventory is the interesting part:
   Parcae is *"the cheapest untried answer to the open problem in AGENTS.md"* (the recurrent
   KDA NaN episodes), and it is **not worth keeping a 317-line Rust port of** when the
   mechanism is 4 lines of `exp(ΔA)` and the official version is a PyPI package with
   checkpoints. If Parcae is tried, try *theirs*.

---

#### `burn-byteflow` — 1121 LOC, **zero call sites** — **PORT-STANDS** (unverified absence)

1. **External implementation?** **NOT VERIFIED.** arXiv 2603.03583's comment is just "ICLR
   2026" (no code link), and a GitHub repository search for "ByteFlow adaptive byte
   compression" returned **total_count 0**. I did not find the authors' repo, and I did not
   exhaustively search (the GitHub search API in this environment is weak on full-text — it
   also returned 0 for `data2vec`, which demonstrably exists as a concept). Stated as
   unverified, not as absence.
2. **Verdict: PORT-STANDS** — with the caveat that this row rests on a negative search.

---

#### `burn-diffusionblocks` — 793 LOC, **zero call sites** — **PORT-STANDS**

1. **External implementation?** **Yes, official** — `SakanaAI/DiffusionBlocks` (318★,
   Python; README opened: *"This is an official implementation of DiffusionBlocks ... on image
   classification using Vision Transformers"*). Note it is **ViT/image-classification**, not
   a language model, so it is an oracle for the *block-wise diffusion training schedule*
   (`log σ ~ N(p_mean, p_std²)`, the `w(σ)` weighting, the per-block σ range), not for our
   byte-LM use.
2. **Can it run on our stack?** No.
3. **Better reference?** Yes for the schedule and weighting — which is where the numeric
   constants live (`p_mean = -1.2`, `p_std = 1.2`, `σ_data = 0.5`) and where a transcription
   error would be silent.
4. **Verdict: PORT-STANDS.**

---

#### `burn-antihall` — 302 LOC, **zero call sites** — **PORT-STANDS**

1. **External implementation?** **Yes, official** — `thunlp/H-Neurons`, and the crate's own
   doc comment already names it. Opened: 24 files; `scripts/intervene_model.py`,
   `scripts/classifier.py`, `scripts/extract_activations.py`, `scripts/extract_answer_tokens.py`,
   plus TriviaQA data for Llama-3.3 / Mistral-24B / gemma-2-7B. It is a pipeline over
   **specific HF models with their own tokenizers**, which is why the oracle is awkward: our
   port is the *learned* variant (`HallSuppressor`) and the upstream is the *hard* ablation
   (`apply_scaling`).
2. **Can it run on our stack?** No.
3. **Better reference?** Only for `intervene`'s scaling convention. The learned suppressor has
   no upstream counterpart at all — the paper's follow-ups it cites (2604.19765, 2607.00158,
   2512.18623) are analyses, not implementations.
4. **Verdict: PORT-STANDS.**

---

### DELETE (6)

Six crates where the port earns nothing: zero call sites, and either no external exists, an
external already covers it, or a sibling crate supersedes it. Total **1,640 src LOC**.

| crate | LOC | why DELETE | external check |
|---|---|---|---|
| `burn-swiglu` | 187 | Zero wiring, and `loop_block.rs:317-321` already implements the gate inline — TSCT/`LinearLike`-aware, so `burn::nn::activation::SwiGlu` (151 lines, present in burn-nn, semantics identical) is *not* a drop-in. The crate holds plain `burn::nn::Linear`, a pre-`LinearLike` leftover. Its 2 tests are shape-only and its kernel never launches. | `burn::nn::activation::SwiGlu` covers the semantics. 2002.05202 has no better code. |
| `burn-mod` | 411 | Zero wiring. The inventory: **"DUPLICATE-OF burn-mor"** — top-k token selection + gather/scatter is the same code. Only `ModPredictor` is unique, and nothing calls it. | arXiv 2404.02258's comment field is **empty**; I found no official Google DeepMind MoD repo (`google-research/t5x` exists, 3000★, but I did not open it for MoD — **NOT VERIFIED** whether MoD code ships in T5X). Nothing to defer to. |
| `burn-mtp` | 196 | Zero wiring. `AGENTS.md` states it outright: replaced by DSpark. | arXiv 2404.19737 (Gloeckle et al.) comment is **empty**; no official release. The mechanism is superseded internally. |
| `burn-ptrn` | 399 | Zero wiring, post-training-only per the inventory, and **the crate is 399 lines of noise-injection + argmax-over-K**, which is the whole content of the paper. | arXiv 2605.19943's comment is **empty**. GitHub search returned exactly one hit: `JerMa88/PTRM`, **1 star, a Jupyter notebook** — not a credible reference. If we ever want PTRN, the ~20 lines it would take to write are cheaper than maintaining 399. |
| `burn-es` | 345 | Zero wiring. **Self-declared duplicate**: its own doc comment says burn-eggroll "implements the paper's normalized form ... **Prefer burn-eggroll for real ES loops**; the function here stays for quick experiments and its test." | arXiv 1703.03864 (OpenAI ES) comment is **empty**; OpenAI never released code. See the `burn-eggroll` row: the source settles the σ convention, then delete this one. |
| `burn-nope` | 102 | Zero wiring. One function. The inventory: the premise is contradicted (AGENTS.md: *"NoPE → endless generation after post-training"*, and RoPE is required in the attention arm). | **Verified: no upstream code exists** — `MoonshotAI/Kimi-K3` is 4 files, PDF only. Nothing to defer to *or* to check against. 102 lines of un-callable, unverifiable code. |

---

## 3. RANKED LIST OF ADOPTS

**There are two, not three.** I am not padding the list. Every one of the 28 was checked
against a real upstream; the honest result is that the crates.io lane is empty, the
`burn::nn` lane yields two, and everything credible is Python/Triton/CUDA-upstream that this
box cannot call. The ranked list of *deletes* (§2) is three times longer than the ranked list
of adopts, and that is the result.

**Precedent worth naming, not a new adopt:** `cubek-reduce` (ADR-0015) — the one successful
adopt in this library's history — is an **upstream Rust crate from crates.io**, patched in
`vendor/cubek-fix/`. The pattern that has actually paid off is "find a Rust crate", not
"find a paper's repo". A crates.io search across muon / rope / rmsnorm / sinkhorn turned up
nothing credible for any mechanism here: the hits are whole-framework crates
(`ruvector-attention` 78k dl, `lattice-inference` 40k dl, `candle-layer-norm` 17k dl) and
CPU optimal-transport solvers with no bearing on a fused GPU Birkhoff projection.

---

**#1 — `burn-rmsnorm` → `burn::nn::RmsNorm`** (already a dependency)

| | |
|---|---|
| LOC deleted | **235** (`lib.rs` 121 + `fused.rs` 114) |
| call sites to change | **3** |
| dependency edges removed | 1 (from `dormouse-core/Cargo.toml`) |
| correctness risk removed | The crate's `require_grad` device-branching workaround exists because of a **burn 0.22.0-pre.3 bug that upstream has since fixed**; carrying our own copy of a workaround for a bug that no longer exists is exactly the "code you have to re-verify" the owner's principle targets. |
| speed | **0** — and honestly so. The fused kernel cannot run: it is gated on `TypeId::of::<Inner>() == TypeId::of::<CudaBare>()` and the trainer is `Autodiff<Cuda, BalancedCheckpointing>`. This adopt trades a fusion that never fires for a dependency we already pay for. |
| risk of the swap | **Cleared, not assumed.** `burn-core-0.22.0-pre.4/src/module/param/tensor.rs:129-133` — `Param::from_tensor` wraps with `set_require_grad(value, true)`, and `set_require_grad` is a documented no-op on non-autodiff devices (`base.rs:244`). burn-core's own `test_module_val_train_stateful` asserts the behaviour. |

**#2 — `burn-rope` → `burn::nn::RotaryEncoding`** (already a dependency)

| | |
|---|---|
| LOC deleted | **975** |
| call sites to change | **0** — the crate is unwired, so this is a pure delete |
| dependency edges removed | 0 from our crates (it was a dev-dep of `burn-spectral`, used by one example) |
| correctness risk removed | 975 lines of frequency math that **once shipped an inverted YaRN ramp with a green test suite** (the inventory's own quote: an f32 `cos` collapses to 1.0 below ~3e-4 rad, so an `acos∘cos` round-trip in the verification itself lies). That is the library's only admitted precision-bug-shipped-green, and it is in the code we would be deleting. |
| speed | **0 today**; the fused `rope_autodiff` op has **no external equivalent** (`burn::nn::RotaryEncoding` has no fused path) and no matmul, so it is immune to all three of our precision blockers. **If RoPE is ever wired, keep `rope_cuda.rs`'s two kernels and take only the frequency table, YaRN scaling and tests from burn-nn.** |
| feature coverage | YaRN is **subsumed by an upstream API**: `RotaryEncodingConfig::init_with_frequency_scaling(f)` exists specifically to plug in RoPE extensions. |

**Combined: 1,210 LOC, 3 call sites, 1 dependency edge, 1 dep edge already dev-only.**
With the six `DELETE`s: **2,850 LOC, 10.3% of the library** — and **all of it unwired or
subsumed**, so neither list costs us a running behaviour.

---

## 4. RANKED LIST OF ORACLE UPGRADES

The flagship claim is "bit-for-bit against the reference implementation". Per the inventory's
own cross-cutting note: *"Zero bit-for-bit reference harnesses"* outside `burn-gdn2`, and
*"the closest are hand-transcribed formula assertions"*. So today that claim, applied to
`burn-kda`, `burn-engram`, `burn-dspark`, `burn-mhc`, `burn-sct`, `burn-bitnet` and
`burn-jepa`, means **bit-for-bit against our own transcription** — a much weaker statement,
because a transcription error is symmetric: we get the same wrong answer from both sides of
the comparison and the test goes green. Eight of these upgrades have the authors' own source
available. Ranked by fidelity gained per unit of cost:

---

**#1 — `burn-kda` → `MoonshotAI/FlashKDA` `tests/torch_ref.py`** (fidelity gained: enormous; cost: lowest)

The strongest artifact found in this pass. Opened, and it is a PyTorch reference written
*specifically to match the kernel bit-for-bit*: it reproduces the kernel's warp-shuffle
tree-reduction order (`reshape(..., 16, 8)` then 16 partials), fp32 FMA via a float64
intermediate, `exp2` with flush-to-zero below the f32 tiny normal, and a `tanh.approx.f32`
sigmoid. That is not a reference PyTorch; that is a **bit-matching** one.

- **What we compare against today:** a hand transcription of arXiv 2510.26692/2607.24653 plus
  our own recurrent-vs-chunk self-consistency. The inventory lists no parity harness for
  `burn-kda`'s fused path (and notes it is dead anyway behind the
  `Autodiff<Cuda, Balanced>` TypeId gate).
- **Cost:** strip the single `load_inline` CUDA sigmoid helper (replace with
  `torch.sigmoid`), run on **CPU** — no GPU, no CUDA, no sidecar — emit ~10 tensors per
  case, compare in one Rust test. **~120 lines of Python + 1 test file.**
- **Bonus, free:** `docs/20260420-flashkda-v1-deep-dive.md` answers the design questions our
  port had to guess at — why `CHUNK = 16` (bf16 range of `exp(cumsum(g))` at the `g_min = -5`
  floor), why a 16×16 inversion by forward substitution, and the SM80 MMA path. Our
  `AGENTS.md` says "FlashKDA math already matches burn-kda"; this doc is where that claim can
  be checked instead of asserted.
- **Limit, stated:** FlashKDA is **forward-only**, so this covers the forward recurrence and
  **not** the backward — which is the part `burn-kda` is actually wired for. The backward
  remains un-orphaned. Honest accounting.

---

**#2 — `burn-mhc` → `deepseek-ai/DeepGEMM` `third-party/tilelang_ops/ref_mhc.py` + `csrc/apis/hyperconnection.hpp`** (fidelity gained: from *nothing* to official; cost: low)

- **What we compare against today:** hand-written formula assertions. The inventory calls
  burn-mhc "best verified" among the unwired crates, but "verified" here means verified
  against **our own reading of the paper** — this crate currently has **no external oracle
  whatsoever**. It is the only live crate in the library with none.
- **Why I nearly missed it:** DeepSeek has no mHC repo, and I grepped
  `DeepSeek-V3.2-Exp/inference/model.py` (38 KB) for `sinkhorn|hyper.?conn|mhc` → **0 hits**,
  class list confirmed. The implementation is inside **DeepGEMM**, whose README lists
  "HyperConnection (HC)" among its primitives. This is the kind of thing that only turns up
  if you read the second repo's README instead of searching for the paper's name.
- **Cost:** `ref_mhc.py` is a PyTorch reference; ~80 lines to run on CPU and emit the
  pre- and post-projection tensors + one Sinkhorn-scaled `H_res`, then 1 Rust test asserting
  the Birkhoff invariants (row/col sums = 1, ‖H_res‖ ≤ 1) and the `H_pre = sigmoid`,
  `H_post = 2·sigmoid` forms. Low.
- **Limit, stated:** DeepGEMM's Requirements are **"NVIDIA SM90 or SM100"** — sm_120 is out
  of set, so the *kernel* is unreachable here forever. **Oracle only.** That is exactly the
  distinction this report is trying to keep sharp.

---

**#3 — `burn-dspark` → `deepseek-ai/DeepSpec` `deepspec/modeling/dspark/{loss.py, markov_head.py, common.py}`** (fidelity gained: turns a claim into a test; cost: low-medium)

- **What we compare against today:** nothing. The crate's doc comment says "matched against
  the official DeepSpec implementation" and quotes Eq. 9-12 — that is a **claim with no test
  behind it**. The loss has five weighted terms (`w_k = exp(-k/γ)`, `L_ce`, `L_tv` (L1 over
  softmax probs), `L_conf` (BCE), and γ = 4.0) and a `0.1/0.9/1.0` mixing. A γ error, an
  L1-on-logits-vs-probs error, or a position-weighting direction error are all invisible to a
  self-consistency check and all change training.
- **Cost:** the three files are 9 + 11 + 11 KB. A parity test needs ~150 lines of Python to
  emit (logits, targets, draft-probs, teacher-probs, conf-labels) over a few hundred random
  cases, plus 1 Rust test. Medium-low.
- **Extra value:** the same repo also ships `deepspec/eval/dspark/confidence_head.py` (21 KB)
  and `draft_ops.py`, which cover our *inference*-side primitives (`sample_tokens`,
  `AcceptRatePredictor`) — those have no oracle at all today.

---

**#4 — `burn-gdn2` → `NVlabs/GatedDeltaNet-2` `lit_gpt/gdn2_ops/{chunk_gdn2.py, fused_recurrent_gdn2.py}`** (fidelity gained: closes the last indirection; cost: **lowest of all** — the harness already exists)

The **only** crate in the library with a real bit-exact harness, so the marginal cost is the
smallest available: `tests/gen_reference.py` already emits 1000 cases into
`tests/ref_data.bin` and `tests/bit_exact.rs` already compares. But `gen_reference.py` is
explicitly *"a faithful, pure-PyTorch transcription of the official ... layer ... **with the
Triton fused_recurrent kernel replaced by an equivalent per-token scan in torch**"*. So we
transcribed their layer and then re-implemented their kernel's recurrence ourselves — the
recurrence is the part that matters, and it is the part we guessed. Replacing the scan body
with the Triton source's actual recurrence is a **body swap inside an existing script**.
Also: **add arXiv 2605.22791 to the crate doc** — the paper exists (verified) and the crate
cites only the repo.

---

**#5 — `burn-engram` → `deepseek-ai/Engram` `engram_demo_v1.py`** (fidelity gained: high; cost: medium — needs a tokenizer)

The official Engram module implementation, opened and confirmed real (the file's own
disclaimer scopes out attention/MoE/hyper-connections and keeps exactly the part we ported).
**Caveat that sets the cost:** it imports `transformers` and `sympy` and drives a real
`AutoTokenizer`, so a parity test needs a tokenizer fixture. The separable core — per-ngram,
per-head hash key construction and the lookup-table build — is pure Python and testable
without it. Also relevant: our trainer uses **neither** the Engram hasher **nor** BLT's, but
FNV-1a-32 (`dormouse-data/src/lib.rs:407`), which is a third addressing scheme and therefore
a third thing with no oracle.

---

**#6 — `burn-sct` / `burn-spectral` → `EctoSpace/SCT` `spectral_compact_training/spectral_layer.py`** (fidelity gained: medium; cost: low)

`burn-sct/tests/cmp_reference.rs` already exists — repoint it at the authors' file rather than
our transcription. The paper's own comment field names the repo, so there is no discovery
cost. Note the repo also ships a convergence report, a rank sweep, and **SteamDeck** results,
which is the closest published precedent to our 16 GB consumer box.

---

**#7 — `burn-bitnet` → `microsoft/BitNet` `gpu/model.py` + `AAzdi/Sparse-BitNet` `llm/arch/model.py`** (fidelity gained: medium; cost: low)

`gpu/model.py` is the official b1.58 PyTorch reference; `Sparse-BitNet` is the official code
for the 2603.05168 N:M half our crate claims (confirmed by README: "N:M Structured Sparsity",
"Triton-based fused ... sparsity mask creation"). The quantizer bodies are 6 lines each; the
fidelity lives in `bitnet_v2`'s Hadamard ordering and the N:M mask convention. Both upstream
implementations are **inference-only**, so this is a forward-path oracle and nothing more.

---

**#8 — `burn-eggroll` → `ESHyperscale/HyperscaleES`** (fidelity gained: settles a live disagreement; cost: near-zero)

375★, Jax, reached from the arXiv comment's website link. Its value is not parity — it is
**settling a factor-of-√r between two of our own crates**: `burn-eggroll` uses the paper's
normalised `(σ/√r)·A·Bᵀ`, `burn-es` documents its own as `σ·A·Bᵀ` and defers. Read the
source, pick one, delete the other.

---

**#9 — `burn-fastblt` → `facebookresearch/blt` `bytelatent/data/patcher.py`** (fidelity gained: proves an existing claim; cost: low — **but low value**)

The crate says it is "Faithful to Meta's `bytelatent/data/patcher.py`" and never tests it.
`patcher.py` is short and pure-Python. Spend this **only if** the crate is revived — per the
inventory, Engram supersedes it on both addressing axes and the trainer uses neither hasher.

---

**#10 — `burn-jepa` → `facebookresearch/dinov2` for KoLeo** (fidelity gained: low; cost: lowest)

KoLeo's `koleo_loss` in DINOv2 is a dozen lines and our port is already faithful. Listed for
completeness. **The data2vec-2.0 leg is NOT VERIFIED** — see §5.

---

## 5. NOT VERIFIED, and where I am genuinely unsure

Stated plainly rather than papered over.

1. **data2vec 2.0 (2212.07525) has no official home I could find.** `facebookresearch/data2vec`,
   `.../data2vec-2.0` and `.../data2vec2` **all 404** through the GitHub API, and the arXiv
   comment field is empty. `facebookresearch/fairseq` (32219★, live) is the historical home of
   data2vec 1.0, but **I did not open its data2vec module** to check for 2.0. So
   `burn-jepa`'s JEPA leg has no confirmed oracle. This is the largest open gap in §4.
2. **Mixture-of-Depths (2404.02258): no official Google DeepMind repo found.** The arXiv
   comment is empty and my GitHub search returned nothing. `google-research/t5x` (3000★) exists
   and I did not open it for MoD. A well-known MoD implementation is widely believed to live in
   T5X/JAX — **I did not verify that, so I am not asserting it.** This is why `burn-mod` is
   filed `DELETE` on duplicate-within-our-library grounds rather than on "no external exists":
   if an external *does* exist, the delete is still right (the crate is a duplicate of
   `burn-mor` and unwired), but the *reason* would change.
3. **`burn-byteflow` (2603.03583) and FastBLT (2605.08044): absence of code NOT VERIFIED.**
   Both arXiv comments are empty; my GitHub repository search returned `total_count 0` for
   both. I treat that as "I could not find it", not "it does not exist" — this environment's
   GitHub search API is demonstrably weak on full-text (it also returned 0 for `data2vec`).
   `burn-ptrn` (2605.19943) is the same shape of answer, with the added detail that the only
   hit was a **1-star Jupyter notebook**, which is not a credible reference either way.
4. **`burn-mor` is wired 3× and does not compile** (`E0599: no method named int` ×2 in
   `src/topk_gather.rs`, per the inventory). I did not run a build to confirm. If true, this
   is the most urgent item in this report and it is not an adopt/port question at all.
5. **The Muon+ counter-result is reported, not reproduced.** `modded-nanogpt` discussion #239
   claims `col_row` Muon+ *increased* loss by ~0.005 in their setup. I read the thread; I did
   not run it. Since `burn-muon-plus` is our **default optimizer** at 4 call sites, this is
   worth a 20-minute check on our own box before we trust the port's `ns_steps = 8` /
   `ColRow` defaults — the paper says 8, the repo's configs say 5.
6. **`burn-spectral` cites 7 papers; I verified 5 individually** (2604.00733, 2504.12285,
   2504.18415, 2603.05168, 2602.21545) plus 2412.04787 and 2202.09368 in the final batch —
   all seven real. I did not re-derive which of them the crate's *code* actually implements.
7. **I did not verify the FLA Blackwell issues** (#945/#953 corruption, #999/#1000 hangs) or
   the 14 ms/call PCIe floor. They were given as owner-measured on this box and I took them as
   constraints, not claims to re-litigate. Note that my FlashKDA finding runs *against* the
   general shape of those constraints: `120a` **is** in FlashKDA's supported arch list, and
   the blocker there is PyTorch packaging and the absence of a backward kernel — not the arch.
8. **I did not check whether `fla/ops/kda` and `lit_gpt/gdn2_ops` are covered by the FLA
   Blackwell issues specifically.** They are Triton, so they inherit the risk, but I did not
   confirm a report against those two paths.
9. **Two arXiv IDs I half-remembered and then checked, which are NOT the right papers** —
   recorded so nobody re-does it: `2505.22791` is *"Laser-driven ferroelectricity in SrTiO₃"*,
   and `2505.10966` is *"Can Large Language Models Correctly Interpret Equations with
   Errors?"*. **Gated DeltaNet-2 is 2605.22791**; **Gated DeltaNet (v1) is 2412.06464**
   (ICLR 2025, cited by the Kimi-Linear README).

---

## 6. Appendix A — sources actually opened

**arXiv (all 31 crate-cited IDs + 2 corrections), via `export.arxiv.org/api/query` and
`arxiv.org/abs/`:** 2510.26692, 2607.24653, 2603.15031, 2512.24880, 2507.10524, 2404.02258,
2601.07372, 2412.09871, 2605.08044, 2605.19943, 2512.23675, 2404.04620, 2604.12946, 2604.00733,
2402.17764, 2504.18415, 2504.12285, 2603.05168, 2407.09527, 2602.21545, 2212.07525, 2511.08544,
2304.07193, 2607.05147, 2404.19737, 2512.01797, 2511.16652, 1703.03864, 2407.04620, 2603.03583,
2506.14202, 1910.07467, 2002.05202, 2104.09864, 2309.00071, 2412.04787, 2202.09368, 2412.06464,
2605.22791, 2505.22791, 2505.10966. All 200 OK, all titles as claimed except the two
corrections in §5.9.

**GitHub repos — README and/or full file tree opened (via `raw.githubusercontent.com` and
`api.github.com/.../git/trees`):** `MoonshotAI/FlashKDA` (tree + `setup.py` + deep-dive +
`tests/torch_ref.py`), `MoonshotAI/Attention-Residuals` (tree), `MoonshotAI/Kimi-Linear`
(tree), `MoonshotAI/Kimi-K3` (tree), `deepseek-ai/Engram` (tree + `engram_demo_v1.py`),
`deepseek-ai/DeepSpec` (tree), `deepseek-ai/DeepGEMM` (README + tree),
`deepseek-ai/DeepSelect` (README), `deepseek-ai/DeepSeek-V3.2-Exp` (`inference/model.py`,
grepped), `deepseek-ai` org listing (39 repos), `MoonshotAI` org listing (43 repos),
`SakanaAI/DiffusionBlocks` (README), `SakanaAI` org listing (62 repos),
`NVlabs/GatedDeltaNet-2` (tree + README), `EctoSpace/SCT` (tree),
`K1seki221/MuonPlus` (tree + README), `microsoft/BitNet` (tree),
`AAzdi/Sparse-BitNet` (README), `facebookresearch/blt` (tree + README),
`thunlp/H-Neurons` (tree + README), `test-time-training/e2e` (README),
`raymin0223/mixture_of_recursions` (README), `ESHyperscale/HyperscaleES` (repo probe),
`ESHyperscale/nano-egg` (repo probe), `fla-org/flash-linear-attention` (repo probe),
`facebookresearch/dinov2` (repo probe), `facebookresearch/fairseq` (repo probe),
`SandyResearch/parcae` (README + trees via search), `karpathy/nanochat` (repo probe),
`SakanaAI/fast-weight-product-key-memory`, `SakanaAI/evo-memory`, `google-research/t5x`,
`EctoSpace` org endpoint (**404** — user is not an org; the repo is fine).

**Opened and found NOT to exist (404, recorded so nobody retries):**
`facebookresearch/data2vec`, `facebookresearch/data2vec-2.0`, `facebookresearch/data2vec2`,
`1bitml/BitNet`, `1bitml/BitNet-pytorch`, `1bitml/BitNet-b1.58`, `thu-ml/BitNet`,
`microsoft/BitNet` @ `main` README (binary layout; tree opened instead).

**Our own source read for this pass:** `vendor/dormouse-fused/crates/*/src/lib.rs` doc comments
(all 28), all 28 `Cargo.toml` dependency blocks, `burn-kda`'s and `burn-gdn2`'s test harnesses
(`tests/gen_reference.py`, `tests/ref_data.bin` 7.1 MB, `tests/bit_exact.rs`),
`crates/*/Cargo.toml`, and the `use burn_*` reference counts over `crates/*/src`.
Local registry sources: `burn-nn-0.22.0-pre.4/src/{modules/norm/rms.rs,
modules/rope_encoding.rs, activation/swiglu.rs, activation/glu.rs, modules/attention/mha.rs}`,
`burn-core-0.22.0-pre.4/src/module/param/{tensor.rs,base.rs}`, `burn-optim-0.22.0-pre.4`
(optimiser list — confirmed **no Muon**).

**Cross-referenced, not re-derived:** `docs/archive/research/2026-09-27-fused-inventory-attention.md`,
`-memory-objectives.md`, `-precision.md` (per-crate WIRED/UNUSED/SUPERSEDED/BROKEN verdicts,
LOC counts, the `TypeId` gate analysis, the call-site table).
