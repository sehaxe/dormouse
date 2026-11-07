# The hybrid stack: 3:1 KDA/attention, and the uniform pin

**Date:** 2026-11-07. **Branch:** `hybrid` (worktree `/home/sehaxe/dormouse-wt/hybrid`),
base `main 79331e4`. CPU-only; the card's sweep was untouched.

**Question.** Every frontier small/mid model is a HYBRID, not either pure form:
Qwen3-Next and Kimi Linear are 3:1 Gated-DeltaNet : full attention, Jamba
interleaves, MiniMax-01 mixes. Our trainer could build an all-KDA stack
(`block_form="kda"`, `loop_blocks=N`, `max_iter=N`) or an all-attention stack
(`block_form="transformer"`, `n_layers=N`), never the mix — so the most-used
non-vanilla mechanism in the field had no arm. This lands the mix and prices it.

**Why a loss measurement is the only honest one at 1 KB.**
`docs/research/2026-11-07-frontier-vs-vanilla.md` §2: KV is 39.0 KB/token,
312 MiB at batch 8 = 1.9% of the card; attention is 122.7 of 692.9 MFLOP/byte
(17.7%), and at MFU 2% the saved FLOPs are ~0 wall-clock. The hybrid is tested
as QUALITY at a matched bill, not as speed.

## What changed

| Where | What |
|---|---|
| `port/src/dormouse/config.py:434` | `block_stack: str = "uniform"` — the default is today's tree. |
| `port/src/dormouse/config.py:174-181` | `BLOCK_STACKS`, `HYBRID_RATIO = "3:1"`; `"hybrid"` is the shorthand for the canonical ratio. |
| `port/src/dormouse/config.py:2808` | `hybrid_ratio()` — one parser (`"uniform"` → `None`, `"hybrid"` → `(3,1)`, `"K:A"` → ints; malformed is LOUD). |
| `port/src/dormouse/config.py:2831` | `block_kinds(depth, ratio)` — the ordered tuple `[K×K, A×A]…`; refuses a cycle that does not divide `depth`, naming both. |
| `port/src/dormouse/config.py:2862` | `_validate_hybrid()` — every combination that cannot run, each with `Escape:`. |
| `port/src/dormouse/config.py:2994` | `validate_block_form` routes `block_stack != "uniform"` here and returns (the `kda` branch would refuse `n_layers`, which is the hybrid depth). |
| `port/src/dormouse/config.py:5014` | `validate_model_train` refuses the hybrid × loop-memory arms: `segment_len`, `activation_offload`, `activation_recompute`/`iter_checkpoint`, `rand_depth`, `engram_ram`. |
| `port/src/dormouse/model/model.py:243` | the constructor branch: an ordered `ModuleList` of the EXISTING `LoopBlock` / `TransformerBlock`; attention blocks init with the full `n_layers` residual scaling. |
| `port/src/dormouse/model/model.py:1303` | `_readout()` — `norm`+`lm_head`+one final CE, factored out of `_forward_transformer` so both stack forms share it bit for bit. |
| `port/src/dormouse/model/model.py:1340` | `_forward_hybrid()` — embed → ordered blocks (each KDA body runs `depth=1`) → the shared readout. Refuses `segment_len` and an adaptive `depth` at the runtime seam. |
| `port/src/dormouse/model/model.py:918` | `_forward_byte_lm` dispatches to the hybrid before the loop. |
| `port/src/dormouse/model/model.py:558,592,644` | the model-seam refusals for `iter_checkpoint`, `activation_recompute`, `activation_offload`. |
| `port/configs/arch_hybrid31.toml` | B1: `3:1`, 12 blocks, d=890, f=2184. |
| `port/configs/arch_dense890.toml` | B2: dense, 12 blocks, d=890, f=3344. |
| `port/tests/test_hybrid_stack.py` | gates a-f below (20 tests). |

The two block modules are reused, not rewritten: the KDA half is exactly the
`block_form="kda"` `LoopBlock` (controller, KDA/CSA/mHC/Engram, FFN) and the
attention half exactly the `block_form="transformer"` `TransformerBlock`. The
objective is the transformer form's: one final CE. L_Rec belongs to a
weight-shared loop, and every hybrid block runs once — so the loop-only loss
weights are refused rather than silently absent.

## The pair and the bookkeeping

12 blocks at `3:1` = **9 KDA + 3 attention** (asserted on the ordered type list,
not a count). `d=890 = 5 heads × 178` is the only even-head_dim factorisation of
the FLOP-matched width; the FLOP ledger is the one in the research doc §2.2
(`6N + 12·L_attn·d·s`, s=1024 — a comparison bill, not a measured FLOP count).

| arm | config | blocks | d | f | params | MFLOP/byte |
|---|---|---|---:|---:|---:|---:|
| C0 | `flagship3_dense.toml` | 12 attention | 832 | 3072 | 95,035,200 | 692.89 |
| B1 | `arch_hybrid31.toml` | `[K,K,K,A]×3` | 890 | 2184 | 109,892,835 | 692.17 |
| B2 | `arch_dense890.toml` | 12 attention | 890 | 3344 | 109,947,930 | 790.92 |
| (row) | hybrid at the C0 width | `[K,K,K,A]×3` | 832 | 2000 | 94,988,313 | 600.60 |
| (row) | dense at the C0 width | 12 attention | 832 | 3072 | 95,035,200 | 692.89 |

The last two rows are the MEASURED bookkeeping the task asks for at d=832; they
are not committed presets. B1 is ISO-FLOPs with C0 (−0.1%) at +15.6% parameters;
B2 gives all of B1's parameters (0.05% apart) to the dense form at +14.2%
compute. So:

* B1 vs C0 = does the mechanism buy anything at equal compute?
* B1 vs B2 = does the mechanism buy anything at equal parameters?
* B2 vs C0 = what do the extra parameters buy in the dense form?

B1 and B2 do NOT have the same FFN width (2184 vs 3344) — the 9 KDA blocks cost
more per block than the attention blocks, so at the same total budget the hybrid
must spend less on its FFN. That allocation difference is part of the arm, not a
hidden width.

## Gates (each RED on a scratch revert)

| gate | test | scratch revert | observed |
|---|---|---|---|
| a. uniform bit-identical | `test_uniform_is_bit_identical_to_clean_main` | constructor guard `!= "uniform"` → truthy | RED `AssertionError` at `model.py:253` |
| b. ordered stack | `test_hybrid_builds_the_expected_block_sequence` | `block_kinds` emits A then K | RED, type list differs |
| c. 20 CPU steps + every block's grad | `test_hybrid_trains_twenty_cpu_steps_and_every_block_gets_gradient` | `.detach()` the KDA body's output | RED `live == []` |
| d. ratio divides depth | `test_a_ratio_that_does_not_divide_the_depth_refuses_naming_both_numbers` | delete `depth % cycle` | RED `DID NOT RAISE` |
| e. hybrid × max_iter>1 | `test_hybrid_with_max_iter_above_one_refuses` | delete the `max_iter != 1` refusal | RED `DID NOT RAISE` |
| f. exact preset counts | `test_the_two_new_presets_have_exact_parameter_counts` | `d_ffn 3344 → 3352` | RED `110,118,810 != 109,947,930` |

Pins captured on clean `main 79331e4` BEFORE the field existed
(`/tmp/opencode/pin_hybrid.py`): state
`06eecd1d…6405b8a`, logits `5de71603…846d9ba`, loss `6.200047969818115`,
grads `b6d64da3…53b36f6`, params `64,881` /
`d886cd85…f0ab714`. The gate also asserts `not hasattr(model, "hybrid_blocks")`
on uniform, so the new branch cannot be entered by the default.

`ruff check` on the three touched Python files: clean. The full CPU suite is
run at the end of the branch; NEW failures vs clean main are named in the
commit, not hidden here.

## The three-arm A/B (GPU; the card's sweep was untouched)

All three arms must share the optimizer. The 2026-11-07 Muon audit
(`docs/reviews/2026-11-07-muon-audit.md`) found `muon_lr_mult` was a dead knob
and Muon's step was 15–55× below AdamW's per element; do not repeat an
architecture A/B on that base. Use the `muon-audit` fix with the A1-established
`MULT`, or AdamW for all three if that patch is not in the merged checkout.

```bash
cd /home/sehaxe/dormouse/port
export PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
MULT=${MULT:-12}                 # the A1 value, not a guess at run time
OUT=/home/sehaxe/logs/hybrid_a_b
mkdir -p "$OUT"

for seed in 1 2 3; do
  for arm in c0:configs/flagship3_dense.toml \
             b1:configs/arch_hybrid31.toml \
             b2:configs/arch_dense890.toml; do
    tag=${arm%%:*}; cfg=${arm#*:}
    # 40,690 steps x 24,576 B = 1 GB. Same data, steps, batch, mix plan,
    # doc masking, eval window and seed for all three arms.
    $PY -u -m dormouse.cli train "$cfg" \
      --data ../data_local/best/fwedu.bin \
      --eval ../data_local/mix4/web_eval.bin \
      --set steps=40690 --set batch=8 --set seq_len=1024 --set grad_accum=3 \
      --set opt=mix --set lr=3e-4 --set muon_lr_mult=$MULT --set seed=$seed \
      --set eval=true --set eval_every=20345 \
      --set mix_plan=configs/mix_v7_docmask.toml --set doc_masking=true \
      --set log_every=20345 --set ckpt_every=40690 \
      --set ckpt_name="hy_${tag}_s${seed}" \
      --set out_dir="$OUT/${tag}_s${seed}" \
      --json-log "$OUT/${tag}_s${seed}.jsonl" \
      > "$OUT/${tag}_s${seed}.log" 2>&1
    grep -oE "eval step=[0-9]+ off=[0-9]+ bpb_fin=[0-9.]+" \
      "$OUT/${tag}_s${seed}.log" | tail -1
  done
done
```

Read `bpb_fin` (doc-masked, 196 608 B window at offset 0), paired by seed.
Pre-registered rule: a win is `mean(B) < mean(A)` by **≥ 0.02 BPB** AND more
than 2× the arm's own spread; ties go to the dense form. Cost: 3 arms × 3 seeds
× 4.63 GPU-h ≈ **41.7 GPU-h**.

## The honest caveat

At 1 KB context the memory, latency and FLOP axes are dead by arithmetic (KV =
1.9% of the card; attention = 17.7% of training FLOPs; MFU 2%). A hybrid buys
loss here **only if the gate shows it**: B1 < B2 by ≥ 0.02 BPB means the
mechanism survives at this scale; B1 ≥ C0 means it does not. And at 1 GB we are
4.8× below the data optimum (10.5 B/param vs ≈50), which biases the test against
capacity mechanisms — a negative result at 1 GB does not transfer to 4.8 GB, and
a positive one is strong precisely because it won against that bias.
