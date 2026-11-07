# byte_patch: the patch compute unit under the dense stack

**2026-11-07 · subagent report.** Implements `byte_patch` (default `false`), the
form that runs `block_form="transformer"`'s `n_layers` unique blocks over the
conv codec's PATCH positions instead of bytes. CPU gates are green; no GPU A/B
has run yet. Branch `byteflow-dense`, base `550bc43`.

**Finished 2026-11-07 by the committing agent.** Every gate was re-run and each
was shown RED by a scratch mutation (§8); every mutated file was restored
sha256-identical. The gate-(a) sha pins were independently recomputed from a
`git archive` of clean `550bc43`. The §4 FLOP rows were re-measured on CPU. The
GPU A/B (§6) and every wall-time claim are UNMEASURED in this worktree (no GPU).

## 1. The seam

`model/convcodec.py` already exposes the whole boundary the dense stack needs:

* `ConvCodec.encode_chunks(bytes [b,t], k)` returns
  `h [b,t,width]` (the per-byte stream the decoder's skip reads),
  `z [b,k,d_global]` (one latent per selected patch) and
  `sel [b,k]` (selected patch starts, in BYTE positions);
* `ConvCodec.decode_chunks(g [b,k,d_global], h, sel)` returns byte logits
  `[b,t,256]` (fp32), with the low-rank upsample gathered back per patch, a
  per-byte skip from `h`, the byte bias, and a causal depthwise smoothing conv.

Causality (the 2026-11-03 fix, `f407fe2`) is inside `_patches`
(`convcodec.py:243`): two left-padded depthwise `Conv1d(3)` over the byte
embedding (a 5-byte receptive field: 4 bytes of left context plus the byte
itself), then a strided `patchify` over the
PREVIOUS patch's window (`F.pad(x, (patch_bytes, 0))`, `convcodec.py:255-256`).
Patch `i` is a function of bytes `<= i·s − 1`; it cannot see patch `i+1` or even
its own patch. `decode_chunks`'s content is per-position: chunk `= patch(j)`,
the byte skip `h[j]`, and a left-padded `smooth` conv (`convcodec.py:313-316`).
Nothing in the path reads the future.

The seam had been welded to the KDA loop by two checks:
`compat_armed` (`byteflow.py:817`) only admitted loop arms, and
`validate_block_form` refused `use_byteflow` with `block_form="transformer"`
(`config.py:2915`, was `if model.use_byteflow:`). Both now admit the dense
stack, and the channel's stack is selected by the flag.

## 2. What changed

| where | change |
|---|---|
| `config.py:926` | `byte_patch: bool = False` on `ModelConfig`, documented; `byteflow_k_tokens` (`:954`) documents the K=P requirement |
| `config.py:2816-2978` | form/codec/K/mod/TSCT refusals (all LOUD; the flag's block starts at `:2923`) |
| `config.py:5101` | `seq_len % byteflow_patch_bytes` refused at `load_run`, before the first step |
| `model/byteflow.py:817-830` | `compat_armed` admits `byte_patch` (the dense-stack arm) |
| `model/model.py:242-269` | the `block_form=="transformer" and byte_patch` build: blocks + final norm + `BfChannel`; NO `embed`/`lm_head` |
| `model/model.py:817-905` | `_forward_channel` runs the dense stack for `byte_patch`; K is derived as `t // patch_bytes` (`:882`), the final norm goes in before `out_proj` (`:899`) |
| `configs/arch_dense_byteflow8.toml` | flagship3_dense + patch8 (97,584,896 params) |
| `tests/test_byte_patch.py` | gates (a)–(f) |

The byte embedding and `lm_head` are deliberately NOT built under `byte_patch`
(the codec is both ends). They would be 425,984 parameters with no reader and no
gradient — the dead-capacity defect, and the reason the count is 97,584,896
rather than 98,010,880. Test (c) asserts every remaining parameter receives a
gradient.

## 3. Shapes at each boundary

`t = 1024` bytes, `s = byteflow_patch_bytes = 8`, `P = t // s = 128`,
`d_model = 832`, `d_global = 1536`:

| stage | tensor |
|---|---|
| input | `bytes [b, 1024]` (int64) |
| codec encode | `h [b, 1024, 64]` (fp32 byte stream), `z [b, 128, 1536]`, `sel [b, 128]` |
| `BfChannel.in_proj` | `x [b, 128, 832]` |
| 12 × `TransformerBlock` | `[b, 128, 832] → [b, 128, 832]`, causal SDPA over patch positions |
| final RMSNorm | `[b, 128, 832]` |
| `BfChannel.out_proj` | `g [b, 128, 1536]` |
| codec decode | byte logits `[b, 1024, 256]`, fp32 |
| loss | `[b·1024, 256]` vs targets → `_byte_ce`, mean over BYTES |

The test pins the stack's input with a forward hook:
`[(B, T//8, d_model)] == [(2, 4, 64)]` at T=32 in the tiny geometry
(`test_byte_patch.py::test_the_stack_sees_patches_and_twelve_unique_bodies`),
and 12 blocks are 12 distinct `qkv`/`fc` parameter tensors.

## 4. MFLOP/byte (fwd+bwd), arithmetic and measurement

Per PATCH at the shipped geometry (`s = 8`, `d_model = 832`, `d_global = 1536`),
with `N_stack = 94,608,384` (the 12 blocks only) and the repo's `6N`
convention (2 FLOP/MAC forward + 4 backward):

```
6·N_stack                       = 567,650,304   FLOP  (the 12 blocks)
6·(2·d_model·d_global)          =  15,335,424   (in_proj + out_proj, 2,555,904 MAC)
6·mac_per_patch() (525,312 MAC) =   3,151,872   (codec, fwd+bwd)
                                  -----------
per patch                       = 586,137,600   FLOP
per byte = per patch / s        =  73,267,200   = 73.27 MFLOP/byte
```

`ConvCodec.mac_per_patch()` at `C=64, s=8, L=2, d_global=1536, r=64`:
`64·64·8 + 2·3·64·8 + 2·64·64·3 + 64·1536 + 1536·64 + 64·8·256 + 8·64·256 +
8·3·256 = 525,312` MAC/patch. (An earlier version of this table counted the
codec as `2·mac`, i.e. forward only, while the stack was fwd+bwd; the fwd+bwd
codec is `6·mac`, and the table above is the corrected bill.) The codec is
**0.54 %** of the patch bill; the pairings are **2.62 %**.

Measured with `torch.utils.flop_counter.FlopCounterMode` on CPU (b=1, fwd+bwd,
this worktree, 2026-11-07):

| model | seq | per byte |
|---|---:|---:|
| `flagship3_dense` (control) | 64 | 568.69 MFLOP |
| `flagship3_dense` (control) | 1024 | 568.69 MFLOP |
| `arch_dense_byteflow8` (patch 8) | 64 | **73.68 MFLOP** |
| `arch_dense_byteflow8` (patch 8) | 128 | **73.68 MFLOP** |
| `arch_dense_byteflow8` (patch 8) | 1024 | **73.68 MFLOP** |

The counter reads the control 0.27 % below `6N` (570.21 unrounded; LayerNorm
parameters are not matmuls) and the patch arm 0.56 % above the arithmetic bill
(the counter's treatment of the elementwise/attention ops, none of which is a
`6N` term), so both arms are priced at one convention. **The patch arm is
7.72× cheaper per byte than the control.**

The 126.6 MFLOP/byte headline on `configs/flagship3_dense.toml` is `6N/4.485`
— a ByteFlow patch-size credit the byte-level control does not actually earn
(127.1 at the measured 95,035,200 params). Against that convention the patch
arm is 1.72–1.73× cheaper, because 8 bytes are paid per position, not 4.485.

## 5. VRAM implication

The stack's saved activations are O(positions · d_model), so the block tree's
per-byte tensor count divides by `patch_bytes`. This is arithmetic, not a
measurement: the ledger's `scripts/vram_ledger.py --measure` was NOT run (no
GPU in this lane, and a busy card is refused by design).

* 12 layers × ~15·d elements/position at bf16 = `12·15·832·2 B` = 299,520 B per
  position; at 1 position/byte that is 299.5 KB/byte, at 8 B/patch **37.4
  KB/byte** (the ÷8);
* the codec stays byte-resolution. Its fp32 `[b,t,*]` buffers are `h` (64
  floats = 256 B/byte) plus the decode's `u`, `skip`, `logits` and smooth-conv
  buffers (1,024 B/byte each). Four retained buffers = **4.25 KB/byte**, five
  = **5.25 KB/byte**; which ones autograd keeps at peak is not measured, so the
  band is what to quote;
* net activation drop ≈ `299.5 / (37.4 + 4.25…5.25)` ≈ **7.0–7.2×**;
* weights: the codec + pairings are 2,975,680 fp32 masters (~11.9 MB; bf16 is
  the compute dtype, the masters stay fp32), and the dead 425,984
  byte-embed/head params are not built.

The attention core also shrinks as `(T/P)² = 64×` at s=1024, which neither
`6N` nor FlopCounterMode counts — the wall-time gain may therefore exceed the
FLOP ratio, while the byte-resolution codec caps it from below. **Any
wall-time claim is UNMEASURED in this worktree: no GPU.**

## 6. GPU A/B: dense control vs dense+patch8 — UNMEASURED (recipe only)

**Not run.** No arm in this section has a result, and this worktree has no GPU.
The command below is the pre-registered recipe (same stream, same optimizer
cadence, same eval bytes, 2 seeds, 3000 steps); nothing here is a measurement.
`bpb_fin` on the held-out file is the honest metric (bytes scored and `off=`
are printed on the eval line; compare them, not just the number).

```bash
cd /home/sehaxe/dormouse/port
export PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=0
mkdir -p /home/sehaxe/checkpoints/byteflow-dense
for seed in 1 2; do
  for arm in flagship3_dense arch_dense_byteflow8; do
    .venv/bin/python -u -m dormouse.cli train "$arm" \
      --eval /home/sehaxe/dormouse/data_local/mix4/web_eval.bin \
      --set mix_plan=configs/mix_v7.toml \
      --set steps=3000 --set batch=8 --set seq_len=1024 \
      --set seed=$seed --set eval=true --set eval_every=500 \
      --set eval_bytes=196608 \
      --set ckpt_name="bfdense_${arm}_s${seed}" \
      --set out_dir=/home/sehaxe/checkpoints/byteflow-dense
  done
done
```

Read: `eval ... bpb_fin=` (mean over the two seeds), `off=` (must match),
bytes/step wall time around the `step=` lines, and `nvidia-smi` peak. The control
arm is the recorded 12-layer 95M dense transformer; the treatment differs ONLY
by `use_byteflow/byte_patch/byteflow_codec/byteflow_patch_bytes/byteflow_k_tokens`
already in its .toml. Both run the same `mix_v7` stream and the same 196,608-byte
eval window.

## 7. What could go wrong (named before the run)

1. **The codec may lose information the deep stack could have used.** Patch `i`
   sees only bytes `<= i·s − 1` — deliberately, for causality. The stack never
   sees the current patch's bytes; their contribution reaches the logits only
   through the per-byte skip `h` and the decoder's low-rank `s·256` head. The
   stack therefore cannot compose within-patch information across layers. This
   is the information-theoretic cost the -0.3725 BPB result on the old block did
   not have to pay at the same depth (it had one shared layer); it may dominate
   the win.
2. **128 positions may be too few for 12 blocks at s=1024.** The stack's
   sequence is 8× shorter, so 12 blocks of full attention over 128 tokens have
   far more capacity per position than the control at 1024. Depth-recurrence
   research already says unique depth beats iteration; here depth has fewer
   positions to act on. If the A/B loses, the first follow-up is `s=4`
   (256 positions, 2 seeds), then a codec that lets patch `i` read `i·s` via a
   shifted gather (the causal smoke test would be updated before the run).
3. **Same seed does not mean same init.** The patch form skips `embed`, so its
   blocks draw from a different RNG point than `flagship3_dense`. The 2-seed
   mean is the honest comparison; do not read a one-seed delta.
4. **Wall time may not track FLOPs.** The conv codec is byte-resolution and
   launch-bound; the cited 2.04× step speed was measured on the KDA channel
   (prior work, not this arm, and UNMEASURED here). The dense stack is heavier
   per position, so the codec's fixed cost is a smaller share — which should
   help — but the measured 2.04× is not a promise for this shape.
5. **K=P only.** A smaller K would let the codec's global top-k depend on future
   bytes and break the prefix-invariance gate; `byte_patch` refuses K<P rather
   than run an uncausal selection. No rate chunking is available on this form.

## 8. Gates and their RED proofs

All in `tests/test_byte_patch.py`. Each gate was shown RED by a scratch mutation
on 2026-11-07; after every run the mutated file was restored byte-for-byte and
its sha256 re-checked (harness: replace exactly one anchor, run one test):

| gate | scratch mutation | result |
|---|---|---|
| (a) | extra `torch.randn(1)` on the off path | 4/4 parametrizations FAILED |
| (b) fp32 | reverse the shift: `F.pad(x, (s, 0))` → `F.pad(x, (0, s))` | FAILED |
| (b) fp32 + bf16 | let a patch see its own window: `prev = F.pad(...)` → `prev = x` (the pre-fix form) | both dtypes FAILED |
| (c) | replace the 12 blocks by 1: `range(n_layers)` → `range(1)` | FAILED |
| (c) | share ONE block body: `[TransformerBlock(...) for _ in range(n)]` → `[TransformerBlock(...)] * n` | FAILED |
| (c) | stack positions halved: `k = T//s` → `T//(2s)` | FAILED (hook sees `[2,2,64]`, not `[2,4,64]`) |
| (d) | build the dead `embed` + `lm_head` too | FAILED (98,010,880 ≠ 97,584,896) |
| (e) | divide by patches instead of bytes (mean of per-patch means) | FAILED (exact-equality) |

* (f) has no mutation: it asserts that seven impossible shapes, the old
  transformer refusal, and a non-tiling `seq_len` are LOUD.
* (a)'s four sha256 pins were independently recomputed from a `git archive` of
  clean `550bc43` (that source has no `byte_patch` field): all four reproduce.
* Gates re-run GREEN together after formatting: `10 passed`.
* Full CPU suite (`pytest tests`, 2026-11-07, worktree at this commit): **one
  failure, the known `test_train::test_torch_compile_flag_wraps_the_model_after_resume`**
  (torch's inductor C++ compile; `/tmp` full on this box). The four other known
  reds (`test_export`, `test_fullattn::gate_7`, `test_loop_blocks`,
  `test_prelude`) PASSED here; no new failure, and `test_sft_v2`'s manifest red
  disappears once the untracked `port/sft_pairs/` fixture is present.
