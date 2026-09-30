# Future-byte auxiliary head — implementation review (2026-09-30)

Lane: queue v2 item 5 (`.bulba/goal.md`), the cheap auxiliary objective that
would sit **before** the JEPA retune. Worktree `wt/future-byte`, off `a071ecd`.
CPU/ndarray only; the GPU is running A/B arms.

**Status: implemented, all gates green, A/B registered and NOT run.**

## What was built

| | |
|---|---|
| head | one `LinearLike` (dense `nn::Linear`, bias) `d_model -> vocab`, **untied** from `lm_head` |
| gates | 6 in `future_byte.rs` (see below) |
| readout | `h = norm(out_acc)`, the T-averaged loop readout — the SAME tensor the main head reads |
| loss | gathered-CE, `-mean(log_softmax(head(h))[label])` over the valid positions |
| label | `targets[q + k]`, i.e. the byte `k+1` positions after the one the main CE predicts |
| valid positions | `n = t - k - 1`, mean over them, no wraparound, no invented padding |
| config | `aux_fb_weight` (default **0.0**, OFF) and `aux_fb_horizon` (default **2**, `>= 1`, 0 refused) |
| weight sharing with DSpark | **none** — no `W1`, no draft window, no acceptance head, no shared tensor |
| seam | `probe::FUTURE_BYTE` (ran) / `probe::FUTURE_BYTE_ASKED` (asked), printed `fb=<ran>/<asked>` on the eval line |
| A/B | row **1b** in `docs/AB-PROTOCOL.md` |

Files: `crates/dormouse-core/src/future_byte.rs` (new, 6 tests),
`aux.rs` (+1 field, +1 `None`, zero signature churn), `model.rs` (+2 skip
fields, +1 construction line, +1 branch in `aux_loss`),
`config/schema.rs`, `config/override.rs`, `config/validation.rs`,
`probe.rs`, `lib.rs`, `configs/small.toml`,
`crates/dormouse-train/src/lib.rs` (4 lines: one `let`, one format arg),
`docs/AB-PROTOCOL.md`.

## The two shape decisions, in the two lines they deserve

**Untied.** 2404.19737 trains a separate output head per lookahead horizon
(§3), and a tied head would put the future-byte gradient into `lm_head` — the
arm's whole claim is that it shapes the BACKBONE through the hidden state
without spending the main head's capacity. Tying is also the checkpoint-risky
choice: an untied head is a new field a new weight can bring into being; a
tied one silently changes the meaning of the existing one.

**T-averaged readout, not the per-iteration states.** One `[b,t,d]` tensor,
shared with the main head, so the A/B measures "an extra CE on a future byte"
and nothing else. The per-iteration states (`[T,b,t,d]`) are a genuinely
different mechanism (they would let the head see how the loop's answer
refined) and a later arm; they are also the 4D-tensor class sm_120 crashes on
(AGENTS.md §2.2), which is a second reason not to smuggle them in here.

## The label arithmetic, and why `t - k - 1` and not `t - k`

`targets` is the one-byte-shifted next-byte sequence, built by the trainer as
`bytes[1..] ++ [bytes[0]]` (`dormouse-train/src/lib.rs:1117`) — so
`targets[t-1]` is a **wraparound**: the first byte of the batch again, which
the model at position `t-1` has no reason to predict. The head at position `q`
is supervised on `targets[q+k]`, and the largest index it may read is `t-2`:

```
n      = t - k - 1          valid positions, q = 0 .. n-1
labels = targets[:, k .. t-1]   (width t-1-k = n)
```

`n = t - k` would put the wraparound byte into the objective as a real label —
one position in 510 at `seq_len 512`, i.e. 0.2% of the term, which is exactly
the size of defect nobody sees in a loss curve. The `-1` is the whole point of
the count, and the gate below enumerates the index set rather than sampling it.

The main CE *does* include the wraparound position (it is not this lane's term
and not this lane's file); the difference is called out here so nobody reads
`n = t - k - 1` as a fix to the main loss.

## The gates (all demonstrated red → green)

Every mutation below was applied to the tree, run, and reverted. Magnitudes are
the observed ones on the `d_model 8, t 12, vocab 256, k 2` fixture.

| gate | red mutation | red output | green |
|---|---|---|---|
| `future_byte_loss_uses_only_shifted_targets` | labels `targets[:, 0..t-1-k]` (unshifted — **the DSpark bug, reproduced**) | `targets[0] must not be read at all` (bits 1085651374 vs 1085530779) | labels in `k..=t-2` move the loss by **1.67e-2 .. 1.84e-1**; the 3 entries outside move it by **0, bitwise** |
| same | labels `targets[:, k+1..t]` (wraparound leak) | `targets[2] IS a label at k=2 but the loss ignored it` | as above |
| `future_byte_loss_is_a_function_of_input_and_logits_only` | unshifted labels | `targets[0] is not a label at k=2 but it moved the returned aux: 0.5925034 -> 0.5998127` (**7.3e-3**) | weight is the exact multiplier; non-future labels bitwise-equal; future label moves it; consumed input moves it; the EMA teacher does **not** appear in it |
| same | wraparound leak | `targets[11] is not a label at k=2 but it moved the returned aux: 0.5582603 -> 0.57073104` (**1.25e-2**) | as above |
| same | `if false && fb` in `model.rs` (the arm not called) | `fb_weight > 0 with labels must return an aux term` | as above |
| `future_byte_loss_matches_an_f64_host_reference_at_the_length_boundary` | unshifted labels | rel **2.01e-3** | **rel 4.96e-8** (`5.80598879` device vs `5.80598908` host, n = 6) |
| same | wraparound leak | rel **5.83e-2** | as above |
| same | weight transposed in the *host reference* (this gate's own first draft) | rel **6.52e-3** — looked like noise, was a transposition | as above; the weight layout is now asserted (`[d_input, d_output]`) |
| `the_gradient_reaches_the_backbone_and_never_the_lm_head` | tie the head to `&self.lm_head` | `the future-byte term reached lm_head` | head **2/2** params with a gradient, loop_block **10/27** with a non-zero one, lm_head **0/2** — and `None`, not zero, i.e. not on the subgraph at all |
| `zero_weight_adds_no_parameters_and_the_main_loss_is_untouched` | `if false && fb` | `weight > 0 must return a real term` | `aux.fb` is `None` at 0.0 and the parameter set is the head's **144 params and nothing else** on the fixture (`256*8+16`); the main CE is **bitwise identical** with the arm on and muted; both counters move exactly once and not at all when muted |
| `an_off_arm_record_refuses_to_load_into_an_on_arm_model` | (a `try_load_record` that swallows the error) | would load silently | refused, naming the tensors: `Missing tensors: ["aux.fb.inner.Dense.weight", "aux.fb.inner.Dense.bias"]` |

Two notes on magnitudes, because "the gate separates three cases with margin"
is a claim that can rot:

- The invariance bound is `to_bits` equality for skipped positions, and
  `> 1e-6` for read ones. The observed read-sensitivity is **1.7e-2**, five
  orders above the bound, so the gate is not a coin flip on a noisy fixture.
- The f64 tolerance is `1e-5` **relative**, set from the arithmetic (f32 over
  `d = 8` accumulations plus a log-sum-exp over 256 classes is ~1e-7 here) and
  not fitted to the observed `4.96e-8` — it has ~200x margin. The
  transposition above is the reason it is relative and not absolute: an
  absolute bound on a loss of 5.8 would have had to be loosened to 0.04 to let
  that bug through, or tightened until a legitimate f32 rounding failed.

## Zero-weight identity, checked

The claim is "off by default and checkpoint-compatible exactly like
`use_gr`/dspark-off". What is actually asserted:

1. `DormouseConfig::default().aux_fb_weight == 0.0`, `aux_fb_horizon == 2`.
2. `AuxHeads::fb` is `None` at weight 0 and `Some` above it — the head is
   **not built**, so there are no parameters to carry and none in the record.
3. The parameter count difference between an off model and an on model is
   exactly `vocab*d_model + vocab` and nothing else. On the fixture that is
   144; at `small`'s widths it is `256*768 + 256 = 197 120`, asserted as
   arithmetic.
4. With the arm on and then muted **on the same model** (so the parameter draw
   is identical), the main CE `rec` is bitwise identical. This is the claim the
   A/B depends on: row 1b is "the arm plus a term", not "a different loop".
5. The full suite is green with the arm off everywhere, including
   `tests/preset_exec.rs` (11 passed — the measured preset parameter counts are
   unchanged, which is the practical consequence of (2)) and
   `tests/ckpt_roundtrip.rs` (2 passed — an off-arm checkpoint still saves,
   loads and re-saves, so the new `Option` field did not disturb the burnpack
   format).

**The existing aux tests' values are unchanged**: the whole suite is green,
`aux.rs`'s 8 tests and `dspark_oracle`'s included, and their bodies are
untouched except for the one field added to the struct they construct through
`AuxHeads::new`, whose signature did not change.

## What is COUNTED rather than asserted

`fb=<ran>/<asked>` on the eval line, in the style of `fused kda=` and
`norm=`. Two halves, because they can disagree:

- `asked` is bumped in `model::aux_loss` at the branch the config opened.
- `ran` is bumped inside `future_byte_loss`, only when `n > 0`.

So `fb=0/<n>` is the real defect shape: the arm was opened every step and
produced no term, which is a horizon at or past the sequence length. That is a
config that trains for 2 000 steps with a healthy loss curve and no objective.

**One thing about the counter that a reader must know**: it counts the
TRAINING forwards, not the eval's own. The eval forward passes no labels
(`targets = None` is what keeps the held-out graph out of the tape), so it
*cannot* run this arm, and an eval-local count would read `0/0` forever and
look like a broken head. This is the opposite of `engram=<rows>/<arms>`, which
IS the eval's own forwards, and the asymmetry is in the code comment at the
`println!`.

`n = 0` itself returns an exact zero rather than panicking: a short sequence is
a legal input, and the counter pair is what makes the "horizon longer than
every sequence" case visible. A horizon at or past `max_seq_len` is **not**
refused by `config::validate` on purpose — a short-sequence sweep is a
legitimate thing to want, and the honest response to it is a counted zero, not
a startup failure that cannot be worked around from the config layer.

## The loud refusal

`aux_fb_horizon = 0` is refused by `config::validate` with the arithmetic in
the message. It is the subtle one: `targets[q + 0]` is the byte the MAIN CE
already predicts, so the arm would train, descend, cost a step, and be a second
CE on the next byte through an independent parameter set — the most expensive
way to buy nothing. `aux_fb_weight < 0` is refused as a sign flip. Both halves
are pinned by `aux_fb_horizon_zero_is_refused`, which also asserts the legal
range (`1`, `4`, and a horizon past `max_seq_len`) still validates, so the
check cannot be mistaken for a narrowing one.

## Honest labels on the claims

- **"our adaptation of arXiv 2404.19737's aux heads to a byte-level AR model"**
  is the design's label, and it is the only honest one. The external reference
  is that paper's per-horizon untied heads with plain CE. The paper does not
  prescribe a 256-symbol byte vocabulary, a weight-shared recurrent block, a
  T-averaged readout, a 197 120-parameter dense head, or `k = 2`. The claim
  reused from it is the mechanism-level one — lookahead prediction is what
  produces the induction-head gains reported at 1-30M — and **that has not been
  measured here at all.** No number in this document is a result.
- **"verified"** appears nowhere about this mechanism. The gates verify the
  *arithmetic and the wiring*, against this file and against the burn/torch
  conventions (the f64 host reference is our own, derived from the head's own
  parameters read back off the device).
- The A/B row's cost column is **unknown**, like every other row in that table:
  the queue's per-arm cost has not been re-costed since the attention
  backward was fixed (AB-PROTOCOL, `docs/AB-PROTOCOL.md`).

## Conflicts expected with the other aux lane

The whole `aux.rs` footprint is **one field, one `None`, one comment**:

```rust
pub fb: Option<crate::param::LinearLike>,   // in `struct AuxHeads`
fb: None,                                   // in `AuxHeads::new`
```

`AuxHeads::new`'s **signature is unchanged**, deliberately: it is called from
`aux.rs`'s own tests, from `dspark_oracle.rs`, and from `model.rs`, and adding
a fifth argument would have put a conflict on every one of those lines and
meant nothing to a DSpark reader. The head is attached in `DormouseModel::new`,
which is the only place that knows the weight.

Consequences for whoever merges:
- If the other lane adds a field to `AuxHeads` and its own constructor lines,
  the conflict is one region (the struct and `new`). Keep BOTH heads — they
  share nothing, and neither is a DSpark weight.
- The `Option` pattern is the same one `LoopBlock::gr` already uses, so if the
  other lane also wants a conditional head, `Option<...>` in `AuxHeads` is now
  a precedent rather than an invention.
- `probe::N_ARMS` moved 12 → 14 and `NAMES` grew two entries. Any test that
  iterates `probe::counts()` sees two more zero rows; `preset_exec.rs`'s
  `arm(...)` helper is index-based, so it is unaffected.
- If the other lane also renumbers `probe`, the two changes collide in the
  same constant block. Land one, re-base the other; the NAMES strings are what
  `assert_eq!` messages print, so a swap changes only the labels.

## Follow-ups this lane did NOT do (named, not done)

- **Horizon 4** is a queue row, not a field. One head, one horizon, per the
  external reviewer's instruction.
- **Per-iteration readout** (`[T,b,t,d]`) as a variant — a different
  mechanism, and the 4D-tensor class sm_120 crashes on.
- **No typed CLI flag.** `--set aux_fb_weight=0.1` already reaches the config
  (a `--set` key is added for both fields), and `train/src/lib.rs` is hot with
  A/B arms launching from it. Add `--aux-fb-weight` when a preset needs it.
- **No GPU run and no step-time cost.** The A/B is registered, not run.
- ~~**The mixed checkpoint direction was untested.**~~ **CLOSED while writing
  this file.** The one direction that could have been silent — an off-arm
  record loaded into a model built with `aux_fb_weight > 0` — is refused, and
  the refusal NAMES the missing tensors:
  `Record validation error: Missing tensors: ["aux.fb.inner.Dense.weight",
  "aux.fb.inner.Dense.bias"]`. That is `an_off_arm_record_refuses_to_load_into_an_on_arm_model`,
  and it is the loud outcome ADR-0011 asks for: a resume cannot silently grow
  a head the checkpoint never carried, so the arm's first 2 000 steps are
  never measuring a head the operator did not intend. The three directions
  covered are off→off (green, the ordinary case and `tests/ckpt_roundtrip.rs`),
  on→on (the ordinary case, same test), and off→on (refused, above). **on→off
  — loading an armed checkpoint into an off-arm model — was not tested**, and it
  is expected to be refused by the same mechanism; it is a resume that the A/B
  will not perform (the arm stays on for the whole run), and it is one line if
  anyone wants it closed.
