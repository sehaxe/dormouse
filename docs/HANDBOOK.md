# The handbook

One document to read. Everything else in `docs/` is evidence: a measurement with
a date, or a decision with a status.

The rule that shapes everything here: **a result is a comparison, not a trophy.**
Every mechanism is measured against the same model with that mechanism removed,
on the same held-out window, at the same budget. The absolute loss moves as soon
as the architecture changes — and it will keep moving. The gap is the result.

---

## 1. What this is

A byte-level language model: the input is the file itself, 256 symbols, no
tokenizer. It trains from scratch on one consumer GPU (RTX 5070 Ti, 16 GB,
sm_120), and it trains **in the format it ships in** — ternary weights from the
first step, not a float model quantized afterwards.

The bet is on density: how much model fits in one GPU-day and one file. The
budget is fixed first, then we ask what fits inside it.

---

## 2. How a number is stated

Always three things: **a loss, a window, a budget.**

> Held-out bits per byte on a fixed window, after one pass over the corpus, on a
> single RTX 5070 Ti — compared against the same model with the mechanism
> removed.

- The window is a file, not a vibe: `data_local/best/fwedu_eval.bin`, 196 608
  bytes, `off=0`. It changed on 2026-11-08 when the corpus was refetched, so a
  number from before that date is a different measurement.
- Seed noise at the current budget is **±0.011 BPB**. A delta below that is not
  a result, whatever the table says.
- Dated records live in `docs/reviews/`; the ones that decided the recipe are
  summarised in §7.

---

## 3. The recipe

| piece | choice | why it is this |
|---|---|---|
| block | dense pre-LN transformer, 12 layers, `d_model=832` | beat the loop/KDA arms at equal bytes |
| FFN | gated (SwiGLU), `d_ffn=2048` | −0.0145 BPB at **0.00%** extra parameters |
| attention | full, RoPE, flash | 24% of the block, correct and fast |
| optimizer | Muon (5-step Newton–Schulz) for matrices, AdamW for vectors | 1.92× the step at equal quality |
| precision | bf16 masters, bf16 NS, fp32 accumulate | the smallest format that still learns |
| batch | 24 × 1024 bytes, `torch.compile(dynamic=False)` | 1.43× versus the naive loop |
| weights | ternary, QAT from step 0, annealed | the format it ships in |

`configs/arch_gated_swiglu.toml` is the live recipe on disk. `flagship3_dense.toml`
is the frozen control every past arm was measured against.

---

## 4. Running it

```bash
cd port
uv sync

PYTHONPATH=src .venv/bin/python -m dormouse.cli train configs/arch_gated_swiglu.toml \
  --data ../data_local/best/fwedu_train.bin \
  --eval ../data_local/best/fwedu_eval.bin \
  --set mix_plan=configs/mix_refetch.toml \
  --set steps=4880 --set batch=24 --set seq_len=1024 \
  --set opt=mix --set muon_lr_mult=12 --set lr=3e-4 \
  --set ns_dtype=bf16 --set muon_state_dtype=bf16 \
  --set dense_qat=ternary --set factor_qat_anneal_steps=2000

cd .. && bash scripts/gate_fast.sh
cd port && PYTHONPATH=src .venv/bin/python -m pytest -q -m "not slow and not pins"
```

- **`muon_lr_mult=12`** is not optional: Muon's scale is ~12× smaller than
  AdamW's on these shapes. Without it the matrix half trains at 8–52% of the
  intended step. The CLI prints a banner about it on every start.
- **`dense_qat=ternary`** is the format switch, and `factor_qat_anneal_steps`
  is the ramp: the projection goes from full precision to hard ternary over that
  many steps.
- The suite has two halves: `-m "not pins"` (2500 tests, gates a PR) and
  `-m pins` (82 exact-fingerprint tests, reported not gated).

---

## 5. The code

```
port/src/dormouse/
  config.py          every flag, every refusal
  model/
    model.py         embed → blocks → head
    transformer.py   the live block (attention + gated FFN)
    tsct.py          low-rank factors: U·diag(s)·Vᵀ, the spectral cut
    ternary.py       b1.58 codec: ternarize, STE, annealing
    act_quant.py     activation quantization (int4 / e2m1 / Hadamard)
    packed_acts.py   saved activations stored packed (−80% VRAM)
    export.py        the shipped file: pack → container → reload
  train/
    loop.py          the training step, eval, checkpointing, resume
    optim.py         Muon (Newton–Schulz, batched per shape) + AdamW
  post/              SFT and the verifier-first RL stack
```

Two invariants the code enforces at runtime rather than in prose:

1. **An arm that is off is bit-identical to its absence.** The projector is not
   even called when the flag is off; a test spies the call.
2. **A mechanism that cannot pay refuses loudly.** A config naming a mode that
   was removed, or a combination that loses money, fails at load with the
   survivors listed.

Both exist because both were violated here: a `use_kda=false` that meant "no
attention at all", and an `attn_pos` that was never applied.

---

## 6. Training in the shipping format

The model never exists as a float network that is quantized afterwards.

| step | what happens |
|---|---|
| forward | every weight is projected to `{−α, 0, +α}` — one scale per tensor |
| schedule | `alpha` ramps 0 → 1 over the anneal window; `alpha=0` is the master itself |
| backward | the straight-through estimator passes the gradient to the master |
| export | the master is projected one last time and packed |

Why the master stays full precision: a three-valued number cannot absorb a
gradient. The optimiser holds one bf16 master per weight and accumulates into
it; the network always *computes* in ternary. That is QAT, and the measured cost
of the format is the difference between a ternary run and its bf16 control at
the same budget — not a guess about what quantization "should" cost.

The export path is what makes it shippable: pack to 2-bit codes plus one fp16
scale per tensor, reload, and the recovered model differs from the trained one by
0.15% of the quantizer's own error. The file is the network.

Full evidence: `docs/reviews/2026-11-12-ternary-artifact.md`.

---

## 7. Why the recipe is this small

Eleven mechanisms were measured on the live model and dropped. The short list,
because the reason matters more than the count:

| mechanism | measured | verdict |
|---|---|---|
| MoE (32 experts) | +0.004 BPB, needs a bigger corpus to pay | deferred, not deleted |
| low-rank at r=64 | 1.95× faster, **+28% loss** | the cost is capacity, not under-training |
| attention factorisation | 2.57×, **+27%** | refused |
| low-rank Muon (Orth-Dion) | 1.25×, **+33%** | refused |
| fewer Newton–Schulz steps | 1.08×, **+10%** | refused |
| gated FFN | 0.00% params, **−1.2%** | **shipped** |

The pattern is the argument: at a fixed budget, speed comes from writing the same
function cheaper (width, precision, format), never from dropping work.

Full matrix: `docs/reviews/2026-11-09-architecture-matrix.md`.

---

## 8. Roadmap

1. **Bigger stored, same active** — spectral factors + ternary, so capacity grows
   while the step cost does not.
2. **Bytes per token** — the decode is bandwidth-bound; fewer bits per weight is
   fewer bytes per token, which is the only lever that makes a bigger model faster.
3. **A specialist** — execution-verified code traces, measured on our own test
   suite rather than a benchmark.

None of it needs a cluster. All of it needs data, which is the real constraint.

---

## 9. Where the evidence lives

| folder | what | rule |
|---|---|---|
| `docs/reviews/` | hostile reads of a lane, dated | a snapshot, never maintained |
| `docs/research/` | measurements and literature notes | same |
| `docs/adr/` | one record per decision, with a status | retracted decisions stay, marked |
| `docs/guides/` | how to run a thing on this box | kept current |
| `docs/archive/` | history, not truth | never cite it in an argument |

Start at `docs/README.md` for the naming scheme, `docs/glossary.md` for the
vocabulary, and `AGENTS.md` for the rules an agent must follow.

---

## 10. Ten words

**byte-level** — input is raw bytes, vocabulary is 256, no tokenizer.
**BPB** — bits per byte; 8 is guessing, 0 is knowing the file.
**window** — the fixed held-out slice a BPB is measured on.
**ternary** — three levels per weight, one scale per tensor, two bits.
**QAT** — quantized-aware training: the forward is quantized, the master is not.
**anneal** — the schedule that walks the projection from full precision to hard.
**TSCT** — spectral compression: SVD, keep the head of the spectrum, store the factors.
**Muon** — orthogonalize the update before stepping; every direction equal.
**pin** — an exact fingerprint test; breaks on any legitimate change, so reported.
**reinvestment** — an arm that loses quality at fixed size is kept only if the
resource it frees buys something bigger that wins.
