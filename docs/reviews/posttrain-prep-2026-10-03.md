# posttrain-prep — the SFT byte format, the mask gate, and what a coder cannot be pretrained from yet

**Lane:** `wt/posttrain`. **Date:** 2026-10-03. **North star:** best chat coder.
**Scope of this pass:** the SFT data path only, CPU-only (the card was busy).
**Verdict:** the format and the mask are landed and gated; **the pretraining
mixture contains no code**, which is the next thing to fix and is not fixed here.

---

## 1. What landed

| what | where | LOC |
|---|---|---|
| SFT byte format, JSONL reader, packed stream, loss mask | `crates/dormouse-data/src/sft.rs` (new) | 400 |
| Format + mask gates (CPU, no burn) | `crates/dormouse-data/tests/sft_format.rs` (new) | 260 |
| 1-step SFT smoke + mask gradient gate (CPU/flex) | `crates/dormouse-train/tests/sft_smoke.rs` (new) | 400 |
| `loss_mask` through the loop's CE reduction | `crates/dormouse-core/src/loop_block.rs` | +6 |
| `DormouseModel::forward_sft` | `crates/dormouse-core/src/model.rs` | +26 |
| `serde` + `serde_json` on `dormouse-data` | `crates/dormouse-data/Cargo.toml` | +5 |

No new crate, no new dependency in the compile graph (`serde_json` was already
built for `arrow-json` and `burn-core` — see `Cargo.lock`), and the mask is not a
model field: it is an argument, so it cannot go stale.

### The template, byte for byte

```
<|im_start|>{role}\n{content}<|im_end|>\n
```

`{role}` ∈ `system | user | assistant`; anything else is a **LOUD** error naming
the role and the file to edit. ChatML spelling because coder text is full of
`<`, `|`, `_`, `>` and any ad-hoc marker of the same shape collides with the code
it wraps. The module doc is the specification; the literal in
`sft_format.rs:template_is_the_documented_byte_sequence` is the copy that fails
if it drifts.

### The mask

`mask[i] == 1.0` on **exactly** the bytes the policy must emit to reproduce an
assistant turn: the assistant's `content` **and its `<|im_end>`**. Zero on role
headers, on the separator newline, and on every user/system byte. `<|im_end|>` is
inside the mask because it is the stop signal — a policy that never learns to
emit it cannot end its turn — and the newline after it is outside because
generation stops at `<|im_end|>`.

### The one-byte shift, and why it is a named function

The trainer's labels are `targets[q] = bytes[q + 1]` (`dormouse-train/src/lib.rs`,
the `skip(1).chain(once(&nb[0]))` line), so the loss must consume the mask
**shifted the same way**. `SftBatch::target_mask()` is that shift, wrap included.
It is a named function and a pinned test because this exact off-by-one fed the
DSpark draft head the byte it was about to predict (`docs/reviews/verify-tails-2026-09-30.md`)
and there the shift was the bug; here it is the contract.

### The core seam

`LoopBlock::forward_full_state` gained `loss_mask: Option<Tensor<2>>`: a
per-position weight on `targets`, `None` = today's pretraining path bit for bit.
The divisor is the mask's own count, not `b * t`, computed as a tensor divide
(no host read, ADR-0018 rule 2) with `clamp_min(1.0)` so an all-zero mask is 0.0
and not `0/0`.

`DormouseModel::forward_sft` is the public entry point. It is a separate method
rather than a sixth argument to `forward_with_hidden` because 60+ pretraining and
eval call sites would each grow a `None` for something they have no meaning for.
Same private funnel underneath — not a second implementation.

---

## 2. The mask gate, and two gates that were WRONG and got deleted

`crates/dormouse-train/tests/sft_smoke.rs`. CPU/flex, 3 fake conversations, no
GPU: a gate that needs the card cannot run beside a training run.

**The gate that works is a gradient identity.** The masked CE is a linear
reduction, so for two supervised positions:

```
2 * grad L({q1,q2})  ==  grad L({q1})  +  grad L({q2})
```

elementwise. Exact at any initialisation. It pins three things at once: the mask
weights positions individually, the divisor is the **masked count** (a `1/t`
divisor turns the relation into `g_A = g_B + g_C` and it fails by 2×), and the
backward is the gradient of the masked sum. Measured relative error on this
fixture: **< 1e-7**.

**Falsified three ways**, by editing `loop_block.rs` and watching it go red:

| injected defect | tests that go red |
|---|---|
| mask ignored (pretraining reduction) | gradient identity, unshifted-mask, degenerate-masks |
| divisor `1/t` instead of the masked count | gradient identity |
| mask applied to the sum but not per position | gradient identity, unshifted-mask, degenerate-masks |

### Two gates that were wrong, and why they are worth writing down

**(a) "A user-only byte's embedding row gets exactly zero gradient" is FALSE.**
The attention arm is causal, so the hidden state at `q` is a function of every
position `< q`, user spans included: a user byte's row sits on the gradient path
of every supervised position after it. Measured 1.4e-28 where the first version
of this file expected 0.0. The *second* version asserted the gradient would
**shrink** under the mask; it does not — 1.27e-4 masked vs 1.15e-4 plain, because
the `1/|S|` divisor weights the surviving positions up. Per-byte gradient
magnitude is not a property of the mask. The CE sum is.

**(b) "The masked loss equals the supervised-only mean CE" is unfalsifiable at
initialisation, and the test that asserted it was deleted.** With a randomly
initialised model the CE is ~5.545 at *every* position (256 classes, near-uniform
logits), so masking 60% of the positions moves the loss by **1.3e-3**, while
`L_Rec` and the returned `logits` differ by **4.4e-3 – 1.3e-2** because the loop
scores each iteration's step output through the head *without* the final `norm`
that the returned logits go through. The noise is an order of magnitude larger
than the signal. Verified directly: with the mask arm deleted from
`loop_block.rs`, a tolerance-based assert **stayed green** while the gradient gate
went red. It was also flaky — 2 green runs out of 3 — because the model is built
with unseeded random init (the reproducibility claim was withdrawn 2026-09-30). A
flaky tolerance test is worse than no test.

**Consequence for anything downstream**: an SFT run on this box cannot be
validated by its loss curve. The mask gate is a gradient test for a reason.

---

## 3. The code-domain corpus mix — a proposal, and the blocker under it

No downloading was done (per the brief). Everything below is either a published
number with its source or a measurement taken on this box.

### The blocker: `mix/code/` is not code

Measured on `/mnt/.../aria_data/pretrain/mix/`, 2026-10-03, by sampling **24 MB
per domain** at three offsets (10%/50%/90%) in each of 40 files, looking for
`{ } ; def #include class import function -> =>`:

| domain | files on disk | GB | code% | **braces%** | nl/byte |
|---|---:|---:|---:|---:|---:|
| `code` | 15 224 | 95.9 | 0.036 | **0.000** | 0.0066 |
| `web` | 17 383 | 109.6 | 0.048 | **0.000** | 0.0048 |
| `wiki` | 2 299 | 14.5 | 0.035 | **0.000** | 0.0132 |
| `ruweb` | 3 057 | 19.3 | 0.02 | **0.000** | 0.0031 |
| `math` | 13 710 | 97.4 | 0.509 | 0.407 | 0.0191 |
| `agentic` | 1 001 | 22.1 | 0.373 | 0.357 | 0.0105 |
| *(reference: this repo's Rust source)* | | | | | *0.023* |

The `code` domain is the **lowest** code-signal of every domain measured.
Reading it directly: `code_00000.txt` opens with prose about an ATOM E6xx
silicon bug, `code_14184.txt` mid-file is prose about IoT data management. Where
real code does appear it arrives stripped — `code_09171.txt` reads
`for imgwpath in zip.namelist(): if not ... continue full_name = imgwpath.split('/')`
with the newlines gone. Newline density there is 0.0066/byte against 0.023/byte
for this repo's own source, i.e. one newline per 147 bytes.

**So the pretraining mixture is 0% code in substance, whatever the directory is
called, and the byte share cannot be re-tuned until a real code corpus exists.**
That is the deliverable this lane did *not* do (explicitly out of scope) and the
first thing the next one should.

### The proposal, once a corpus exists

All published code-share numbers are **token** shares. A byte model cannot
inherit them: the same source occupies a different number of bytes than of BPE
tokens, and the ratio depends on how much whitespace the corpus keeps. So:

1. **Buy a code corpus whose whitespace survives.** The Stack v2, StarCoder2's
   permissively-licensed subset, or `codeparrot/github-code` filtered by
   license. The filter criterion that matters for us is measurable and is the one
   this lane measured: **braces/byte and newline/byte**, both by sampling, both
   per-domain. Reject any shard whose braces/byte is under ~0.5%. `mix/code`
   would have been rejected at 0.000%.
2. **Mix by BYTES, and measure the share in bytes.** Proposal: **35–50% code by
   bytes** for the pretraining of a coder, floor 50% non-code.
3. The anchor for that range: Petty, van Steenkiste & Linzen, *How Does Code
   Pretraining Affect Language Model Task Performance?*, **TMLR 01/2025**
   (<https://tallinzen.net/media/papers/petty_van_steenkiste_linzen_2025_tmlr.pdf>,
   read 2026-10-03). In the **competitive** setting (total tokens fixed, code
   displaces text) multi-digit arithmetic **peaks at 40–50% code** and then
   declines; the same paper reports degradation on linguistic/world-knowledge
   tasks growing with code share, and attributes to Aryabumi et al. (arXiv:2408.10914)
   the finding that natural-language performance lessens above **~25% code**.
   That 25% is quoted here at one remove and is **not** independently verified.
   Both effects are reported in both the competitive and the additive setting, so
   the peak is a property of the trade-off and not of the measurement.
4. **The competitive-vs-additive distinction is the whole question** and it is
   ours to answer, not to read off a paper: at our scale the corpus is fixed, so
   every extra byte of code is a byte of text not trained on. Expect the
   text-side loss to be real, and expect 35–50% to be too high. Start at **25%
   by bytes** and A/B 12.5 / 25 / 40 on held-out BPB, one batch size (§2.6 — the
   eval window is `eval_batches × batch × seq_len`, and two runs at different
   batch sizes scored different amounts of text).
5. **Mix arms belong in `mixture/`'s existing `arms.json`** rather than in a new
   mechanism — the arm machinery is already there and this lane did not touch it.

---

## 4. Not done, in the order it should be

- **`--sft-file` is NOT wired.** The data path and the model seam are both in
  place; the flag itself is not. It belongs on `train_loop`'s stream selection
  (`crates/dormouse-train/src/lib.rs`, the `ByteStream::train_and_eval` site),
  which is a 3 445-line file another lane may hold. Nothing else is blocking.
- **A code corpus.** §3. Without one the coder pretrain is not a question yet.
- **A byte-level code-ness detector as a reusable tool.** §3 measures it
  ad-hoc in Python here. It belongs next to `anchors.rs` so the mixture arms and
  the corpus filter agree on what "code" means — and so nobody re-derives it from
  a directory name.
- **SFT on GPU, 3 seeds.** No SFT step has run on this hardware. The mask is
  gated; the pipeline around it is not.
- **Multi-turn masking across a batch boundary.** The mask travels correctly
  (pinned by `jsonl_to_batches_packs_back_to_back`) but a conversation cut in
  half by a batch boundary trains its second half with no context, exactly as a
  pretraining chunk does. That is a deliberate choice, not an oversight; if SFT
  quality says otherwise, the fix is one conversation per row.

## 5. Test evidence

```
cargo test -p dormouse-data --test sft_format          8 passed
cargo test -p dormouse-train --test sft_smoke          4 passed  (×5 consecutive runs)
cargo test -p dormouse-data -p dormouse-core --lib    89 + 15 passed
cargo test -p dormouse-core --tests                   green except the pre-existing reds below
cargo test -p dormouse-train --tests                 all green (incl. sft_smoke)
cargo check -p dormouse-train --features cuda         green
cargo doc --no-deps -p dormouse-data                  clean
```

**Pre-existing red on `main`, not this lane** (reproduced in `/home/sehaxe/dormouse`
at `f6d0038`, and none of these files were touched here):
`dormouse-core` `moe_grad_seam` (2), `moe_routing_seam` (4), `moe_step1_gate` (3),
`preset_exec` (4 — all four say `presets with no execution test: ["byteflow"]`).

## 6. Follow-ups for other lanes (not fixed here)

- `crates/dormouse-core/src/msa_stage.rs` — 7 `missing_docs` warnings on a
  `#![warn(missing_docs)]` crate, so `RUSTFLAGS="-D warnings" cargo doc` cannot be
  green on the workspace as a whole (the AGENTS.md §2.5 gate names
  `-p dormouse-core -p dormouse-data`).
- `vendor/dormouse-fused/crates/burn-msa/src/distill.rs:9` — unused `Device`
  import; the only reason `-D warnings` fails on a workspace doc build.