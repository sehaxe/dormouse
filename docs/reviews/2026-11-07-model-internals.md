# 2026-11-07 — Model internals audit: the dense block, numerically

**Scope.** Read-only audit of `port/src/dormouse/model/transformer.py` and the
`block_form="transformer"` branch of `port/src/dormouse/model/model.py` against
their own documentation, the control they transcribe (`scripts/control_transformer.py`,
`docs/reviews/2026-11-06-control-transformer.md`) and the shipping preset
`port/configs/flagship3_dense.toml` (**95,035,200 params**). CPU only
(`CUDA_VISIBLE_DEVICES=""`, torch 2.13.0+cu129, branch `main` at `0c63c20`).

**Instrument.** `port/tests/test_model_internals_audit.py`, 20 tests, 1.6 s on
CPU. Every number below is produced by that file; the full per-tensor std table
prints under `pytest -s -k parameter_stds`. The four required gates were each
shown RED on a scratch mutation of the live tree, then reverted (table at the
end).

**One environment trap, because it invalidates a worktree audit silently.** The
`port/.venv` in `/home/sehaxe/dormouse` has an *editable* install whose `.pth`
points at `/home/sehaxe/dormouse/port/src`. Running pytest from a fresh worktree
without `PYTHONPATH=$PWD/src` imports **main's** tree, not the worktree's — the
first mutation run passed for exactly that reason. Every reading here was taken
with `PYTHONPATH=$PWD/src` and the import path printed.

---

## 1. The table

`[M]` measured by the test file; `[D]` derived arithmetic from a measured number.

| # | item | documented | measured | verdict |
|---|---|---|---|---|
| 1 | RoPE base | 10 000 (`transformer.py:72`; the control's `rope_tables`) | at `p=1,j=0`: `cos=cos(1)`, `sin=sin(1)`; at `j=1`: `cos(10000^(-1/half))` | **PASS** |
| 2 | RoPE pairing | "Half the head rotates against the other half" (`transformer.py:58-74`) | independent f64 **split-half** reference: **2.03e-7** max; the interleaved reference is **4.97** away | **PASS — split-half (NeoX/`rotate_half`)** |
| 3 | RoPE sign | `x1·cos − x2·sin`, `x1·sin + x2·cos` (`transformer.py:77-83`) | a `+1` in the pair's first half at `p=1` lands as `+cos` / `+sin` | **PASS** |
| 4 | RoPE norm | a rotation is orthogonal | fp32, `head_dim=64`: relative norm drift **1.36e-7**, dot-product drift **3.77e-8** | **PASS** |
| 5 | block norm | LayerNorm, `weight=1`, `bias=0` (`transformer.py:159-162`) | `nn.LayerNorm`, affine, **eps=1e-5**; manual formula agrees to **4.77e-7** | **PASS as documented; `norm_eps` does not reach it (V2)** |
| 6 | final norm | model is "RMSNorm" (`model.py:1`), `cfg.norm_eps` default **1e-3** | `nn.RMSNorm`, gain only, eps **inside** sqrt: **2.38e-7**; the outside-sqrt formula differs by **8.35e-4** | **PASS for RMSNorm; NOT the control's LayerNorm (V1)** |
| 7 | residual scaling | two writes, `0.02/sqrt(2·n_layers)` (`transformer.py:167-168`) | `qkv`/`fc` **0.01999–0.02002**; `proj`/`fc2` **0.004075–0.004090** against **0.00408248**; exactly **24** scaled tensors; 2L vs 12L ratio **sqrt(6)** | **PASS** |
| 8 | head tying | untied for the control; `tie_embeddings=False` default | flagship: `DenseLinear`, distinct parameter *and* pointer, keys `embed.weight`+`lm_head.weight`; tied: identity `lm_head._owner[0].weight is embed.weight`, no `lm_head.*` key, exactly `256·64` params fewer | **PASS** |
| 9 | checkpoint tie | snapshot is authority | `<ckpt>.config.toml` carries `tie_embeddings`; strict round-trip both values; cross-kind load refuses; `diff_keys` names `model.tie_embeddings` | **PASS** |
| 10 | attention scale | `1/sqrt(head_dim)` (`transformer.py:151`) | `0.25` at hd=16, `0.125` at hd=64; SDPA receives `scale=`; hand-rolled masked softmax agrees **2.6e-8** | **PASS** |
| 11 | `qkv` packing | one `3·d²` GEMM (`transformer.py:154`) | one `[3d,d]` weight; exactly **one** packed GEMM and **four** projections per block; `W` sliced into three maps equals the split | **PASS** |
| 12 | attention mask | `is_causal=True` | SDPA kwargs are exactly `{is_causal: True, scale}` — no `attn_mask`; perturbing tokens `>=6` leaves every logit `<6` **exactly 0.0**; no pad id exists in the byte stream, and `doc_masking` masks the **loss**, not attention | **PASS (causal only, by design)** |
| 13 | positional extent | `max_seq_len=1024`, RoPE | no RoPE buffer on the form; `t=48` on a `max_seq_len=16` model runs finite, and `rope_tables(48)[:16] == rope_tables(16)` | **unbounded at the model; refusal lives only in `load_run` (V4)** |
| 14 | counter | log prints `transformer_block` | **one per block per forward**: 12 layers × 3 forwards = **36** in a `grad_accum=3` step | **PASS, but it is `layers × forwards`, not `layers × steps` (V6)** |
| 15 | init | block projections N(0,0.02), two writes scaled | block matches; `embed.weight` std **1.0004** (torch default), `lm_head.weight` std **0.034679** (`1/sqrt(832)`) | **block PASS; embed/head are not the control's init (V1)** |
| 16 | parameter count | 95,035,200; `mirror_control` block 1,110,208 | **95,035,200**: 12 × 7,884,032 blocks + 212,992 embed + 212,992 head + 832 norm; `mirror_control` per block **1,110,208** | **PASS** |
| 17 | seed | `--seed` seeds the init (`cli.py:178-194`) | same seed after consuming 4096 RNG draws: every tensor bit-identical; `seed+1`: `embed.weight` differs | **PASS** |

Per-block arithmetic, checked against the measured module tree rather than a
formula: `4·832² (2,768,896) + 2·832·3072 (5,111,808) + 4·832 (3,328) = 7,884,032`,
×12 = 94,608,384; + 212,992 + 212,992 + 832 = **95,035,200**. The control preset
(`port/configs/mirror_control.toml`, d=304, ffn=1216) is the documented
`4·304² + 2·304·1216 + 4·304 = 1,110,208`.

---

## 2. RoPE convention verdict

**Split-half, base 10 000, positive rotation, applied to `q` and `k` only.**

The test does not call any library rotation. It builds
`ang[p,j] = p·10000^(-2j/head_dim)` in f64 and rotates the complex pair
`(x[..., :half] + i·x[..., half:])`, which is the NeoX / HF `rotate_half`
convention. The code agrees to **2.03e-7** (fp32 table + fp32 application).
The same spectrum with **adjacent** pairing is a different rotation: on the same
fixture it differs from the code by **4.97**, and the mutation that swaps the
pairing turns the gate red by that much (§7).

* **Base**: pinned at `p=1,j=0` (1 rad/position) and `j=1`
  (`10000^(-1/half)`); a different base moves the second pin by >1 %.
* **Sign**: `R(+p)`, not `R(−p)` — a `+1` in the first half emerges as `+sin`
  in the second.
* **Orthogonality**: relative norm drift 1.36e-7 and dot-product drift 3.77e-8
  at `head_dim=64`; that is fp32 round-off, not a systematic scale.
* **History**: this is the control's own function, copied. The repo's other
  position-RoPE (`byteflow.py:288-330`) uses the same half-split and the
  configured base, so no two position-RoPE arms are split by a convention.
  `layerrope.py:185-198` rotates **adjacent pairs**, but it rotates the norm
  *gains* by **depth**, not tokens by position, and this form refuses
  `layerrope=true`. **A future pairing swap invalidates every checkpoint**
  (the angles are baked into the trained weights) and is not comparable to any
  number in the archive.

---

## 3. The norms: which one, and where epsilon lives

| site | class | gain | bias | epsilon | reference agreement |
|---|---|---|---|---|---|
| `ln1`, `ln2` (every block) | `nn.LayerNorm` | yes, init 1 | yes, init 0 | **1e-5** (torch default), **inside** sqrt | 4.77e-7 |
| `norm` (final) | `nn.RMSNorm` | yes, init 1 | **none** | **`cfg.norm_eps` = 1e-3**, **inside** sqrt | 2.38e-7 |

The block matches the control exactly (the control's `Block` is `nn.LayerNorm`
with torch's default). The final norm does **not**: the control's `ln_f =
nn.LayerNorm(d)` is affine with eps 1e-5, and the port reuses the loop's
`RMSNorm(eps=1e-3)`. The 304-parameter delta that `port/tests/test_block_form.py`
documents (13,478,448 vs 13,478,752) is exactly this swap.

`cfg.norm_eps` reaches **one of the thirteen norms** on this form. Feeding the
block LayerNorm 1e-3 instead moves its output by **2.15e-4** — far above the
1e-6 reference agreement, so the knob is measurably absent from the block, not
merely unused by coincidence. See **V2**.

---

## 4. Residual scaling, per tensor

Analytic targets for L=12: `0.02` and `0.02/sqrt(24) = 0.00408248`. Measured
(`flagship3_dense`, unbiased std over the tensor; block 0 and block 11 shown,
all 12 blocks are in the test):

| tensor | shape | std block 0 | std block 11 | target | verdict |
|---|---|---:|---:|---:|---|
| `qkv.weight` | 2496×832 | 0.0200123 | 0.0199972 | 0.02 | scaled? **no** |
| `fc.weight` | 3072×832 | 0.0200107 | 0.0199874 | 0.02 | no |
| `proj.weight` | 832×832 | 0.0040836 | 0.0040772 | 0.00408248 | **yes** |
| `fc2.weight` | 832×3072 | 0.00408436 | 0.00408262 | 0.00408248 | **yes** |
| `ln1/ln2.weight` | 832 | 0 (exactly 1.0) | same | constant | no |
| `ln1/ln2.bias` | 832 | 0 (exactly 0.0) | same | constant | no |

Exactly **2 × 12 = 24** tensors are scaled, the two residual writes and nothing
else; a 2-layer model's `proj` std is 0.01 and the 2L/12L ratio is √6, so the
`n_layers` argument is live, not a constant. Dropping the divisor turns the gate
red with `0.020126` against `0.01 ± 6e-4` (§7).

The embedding and the untied head are **outside** the block's init claim and are
not the control's:

| tensor | port | control | ratio |
|---|---|---|---|
| `embed.weight` | **1.0004** (`nn.Embedding` default N(0,1)) | N(0, 0.02) | **50×** |
| `lm_head.weight` | **0.034679** (`DenseLinear`: `randn/sqrt(in)`) | N(0, 0.02) | **1.73×** |

---

## 5. Head tying and the checkpoint

`flagship3_dense` does not set `tie_embeddings`, and the default is `False`
(`config.py:697`). Proof by identity, not by config: `DenseLinear` (not a
`nn.Linear`, and not `TiedHead`), separate `data_ptr`, both state keys present,
and the count is `2 · 256 · 832` larger than the tied build. The tied build is
the same matrix by identity (`model.lm_head._owner[0].weight is model.embed.weight`),
carries **no** `lm_head.*` parameter or state key, and is exactly
`vocab·d` smaller.

The checkpoint stores the choice twice: the payload (presence/absence of
`lm_head.weight`) and the `<ckpt>.config.toml` snapshot. A strict round-trip
succeeds for both values, and loading either into the other kind is refused by
`load_state_dict`. `diff_keys` includes `model.tie_embeddings`, so a resume
cannot flip it silently.

---

## 6. Positional extent

RoPE on this form has **no table**: `rope_tables(t, …)` is built per forward from
the live `t` (`transformer.py:60-74`), and the model registers no positional
buffer. Measured: a `max_seq_len=16` model forwards `t=48` to finite logits, and
`rope_tables(48)` is the prefix-extension of `rope_tables(16)` exactly. That is
the right behaviour for RoPE and the opposite of the 2026-11-04 defect (a
hard-coded 512 table) — here there is nothing to hard-code.

The boundary is enforced in **one** place, and it is not the model:
`load_run` refuses `seq_len > max_seq_len` loudly (`config.py:5008`).
`train()` re-runs `validate_train` and `validate_model_train` but not that check,
so a hand-built `RunConfig` **trains and extrapolates silently**: measured, a
`max_seq_len=16` model trained one step at `seq_len=128`, loss 5.698. Decode is
the third behaviour: it truncates the context to `max_seq_len` (`decode.py:177`),
so generation never extrapolates. See **V4**.

---

## 7. Counters, through a real window

`counters.count("transformer_block")` is inside the per-block loop
(`model.py:1210-1212`): **one per block per forward**. Measured directly: 12
layers × 3 forwards = **36**; a 3-forward call adds `forward=3`.

The flagship recipe is `batch 8 × seq 1024 × grad_accum 3`, so a log window's
counter is `layers × steps × grad_accum`, not `layers × steps`:

| log | window | printed | arithmetic |
|---|---|---:|---|
| `/home/sehaxe/logs/f3short.log:5` | steps 1–100 | `forward=300 transformer_block=3600` | 12 × 300 |
| `/home/sehaxe/logs/f3muon8.log:5` | steps 1–200 | `forward=600 transformer_block=7200` | 12 × 600 |

The brief's `1440 = 12 × 120` is the `grad_accum=1` arithmetic. **No log on
record prints `transformer_block=1440`**; at `grad_accum=3` a 120-step window
prints **4320**. The counter is correct — the reading that ignores
`grad_accum` is off by 3×.

Counters on the dense form's training log line, classified:

| counter | source | predictable? |
|---|---|---|
| `transformer_block` | model forward, per block | `layers × forwards` |
| `forward` | `model.py:761`, per model forward | `steps × grad_accum` per window |
| `g_norm_adamw/muon/norms_biases` | gradient norms (floats) | **no** — measurements, not counts |
| `nan_skipped` | firewall, device counter | **no** — only when a step was non-finite |
| `kda=off,n=0` | `kda_field`, not a counter entry | yes, but not from the dict |
| `mod_block`/`mod_tokens` | MoD only (`mod_capacity<1`) | `blocks × forwards`, `tokens` |
| `csa_forward`, `moe_expert_calls`, `depth`, `layerrope_forward`, `engram_*`, `mm_fp8_*`, `dense_qat_arms` | other arms | arm-specific; zero/absent here |

**Eval forwards are not counted in the window.** `_eval_line` builds its own
`Counters` and prints only `engram=`, so neither `forward` nor
`transformer_block` measures scoring cost. A reader who wants total forwards
must add `eval_batches` separately.

---

## 8. Verdicts and proposals

### V1 — the dense arm is not the control's initialization (comparability, no fix)

The block is the control's, the **model around it is not**:

| | `scripts/control_transformer.py` | `block_form="transformer"` (flagship3) |
|---|---|---|
| embedding | N(0, 0.02) (`self.apply(self._init)`) | N(0, 1) — torch default, "not re-drawn" |
| untied head | N(0, 0.02) | `randn/sqrt(d)`, std 0.034679 |
| final norm | `LayerNorm(d)`, affine, eps 1e-5 | `RMSNorm(d)`, eps 1e-3, no bias |
| block LN / RoPE / residual scaling | eps 1e-5 / split-half / scaled | **identical** |
| optimizer | AdamW, cosine, lr 6e-4 | Muon+AdamW, WSD, lr 3e-4 |

The 2026-11-06 control's **1.492 BPB** and `mirror_control` are therefore not the
same network at short budgets: the embedding is 50× the control's std, the head
is 1.73×, and the final norm is a different operator. Fixing the init to the
control's would invalidate every dense checkpoint — this is a re-run, not a
patch.

The night report's `flagship3_dense` vs old-flagship comparison is a **stack**
comparison, not a block-only one. `port/configs/flagship_base.toml` (102.7M)
runs `max_iter=3` over ONE unique LoopBlock with KDA+CSA+mHC+AttnRes+Engram+MoE,
`layerrope="paper"` — so its final norm is `LayerRoPENorm`, not `RMSNorm` — and a
TSCT head. The dense arm has 12 unique blocks, none of those arms, the plain
`RMSNorm(1e-3)` and a dense untied head. Both share the port's `nn.Embedding`
init (N(0,1)), and the −0.31 BPB at 4k steps is the whole stack: the block
belongs to it, but the final norm and the head are two more surfaces that changed
with the form.

### V2 — `norm_eps` is a half-dead knob on this form (proposal)

It reaches the final norm only; the twelve block LayerNorms are pinned at torch's
1e-5. An A/B on `--set norm_eps` is a no-op on 12 of 13 norms. Two options: wire
`cfg.norm_eps` into `TransformerBlock` (**invalidates every dense checkpoint**,
because it changes the forward) or state the split in the field's docstring.
Not changed here.

### V3 — `n_layers=0` does not mean what its comment says (proposal; one guard fixed)

`n_layers = cfg.n_layers or cfg.loop_blocks × cfg.max_iter` (`model.py:251`).
The comment says a form switch "keeps the loop's depth", but the loop's **depth**
is `max_iter` passes while its **parameter sets** are `loop_blocks`; the product
is neither. For the shipping loop arms (`loop_blocks=12, max_iter=12`) the
fallback would build **144** transformer blocks. The two shipped dense presets
set `n_layers` explicitly, so no number on record is affected. Proposal: pick one
meaning (passes `max_iter`, or bodies `loop_blocks`) and refuse the other.

**Fixed, because it was unambiguous:** a **negative** `n_layers` was accepted and
`range(-1)` built a "transformer" with **zero blocks** — embed → norm → head,
32,832 params, while the snapshot still said `n_layers = -1`. `validate_block_form`
now refuses any negative count with an `Escape:` (`config.py:2796`), and the
test pins it.

### V4 — the `seq_len <= max_seq_len` refusal has one caller (proposal)

It lives in `load_run`; `train()` does not repeat it, and a hand-built run
extrapolates silently (measured, §6). Moving the check into `validate_model_train`
would give both callers the same rule — but several tests hand-build runs at
`seq_len > max_seq_len` for speed, so that is a decision about the test fixtures,
not a silent edit. Not changed.

### V5 — the docstring's "4× FFN" was preset-specific (doc fixed)

The module claimed `d_ffn = 4·d_model`; `flagship3_dense` runs `3072 / 832 =
3.69×`. The sentence now says the control is 4× and the arm's `d_ffn` is the
config's. Doc-only; no number moved.

### V6 — the counter arithmetic in the brief (no code change)

See §7: `transformer_block = layers × forwards`, and `forwards = steps ×
grad_accum`. The "1440 = 12 × 120" reading is exactly right at `grad_accum=1`
and 3× low for the recipe that produced the arm. `forward` is the field to read
before dividing.

---

## 9. Mutation evidence (each gate RED, then reverted)

Scratch edits to the live worktree, one at a time; `git diff` empty afterwards.

| mutation | file | gate | observed |
|---|---|---|---|
| swap RoPE pairing (split-half → adjacent, same spectrum) | `transformer.py:apply_rope` | `test_rope_matches_the_split_half_reference_not_the_interleaved_one` | `split-half reference disagrees: 4.968e+00` |
| move epsilon (final norm 1e-3 → 1e-5) | `model.py:258` | `test_final_norm_is_rmsnorm_no_bias_and_epsilon_is_inside_sqrt` | inside-sqrt reference differs by `1.78e-4` |
| drop residual scaling (both writes) | `transformer.py:167-168` | both `test_residual_scaling_*` | `Obtained 0.020126`, expected `0.01 ± 6e-4` |
| tie the head regardless of config | `model.py:259` | `test_the_flagship_dense_head_is_untied…` and `test_the_checkpoint_stores_the_head_form…` | flagship lost `lm_head.weight`; tied checkpoint read into the untied arm |

`ruff check` is clean on every file touched. `ruff format --check` reports three
**pre-existing** reflows in `config.py` (lines 2680, 2746, 3393), untouched by
this audit.

---

## 10. Files

* `port/tests/test_model_internals_audit.py` — the 20 gates.
* `port/src/dormouse/config.py` — V3 guard (negative `n_layers`).
* `port/src/dormouse/model/transformer.py` — V5 docstring only.
* This report.

**Rerun:**

```
cd port
PYTHONPATH=$PWD/src <venv>/bin/python -m pytest tests/test_model_internals_audit.py -q
PYTHONPATH=$PWD/src <venv>/bin/python -m pytest tests/test_model_internals_audit.py -q -s -k parameter_stds
```

The `PYTHONPATH` is load-bearing in a worktree (§ instrument note); without it the
editable `port/.venv` silently tests the main checkout.
