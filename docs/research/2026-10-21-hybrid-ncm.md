# HybridAgents' mode switch and NCM's code hierarchy, against a byte-level loop

**Date:** 2026-10-21. **Read-only:** this file only; no code, config or checkpoint touched, no GPU run. **Ask
(owner):** what the papers specify, which part maps onto the loop / `serve` / the RLVR lane, the cheapest
experiment that shows any of it at 138M on one card, and what we reject — for **logic without knowledge**.
**Method.** Both full texts read, appendices included; `togethercomputer/HybridAgent` **exists** (421 KB, pushed
2026-10-18) and was read file-by-file **[C]**; the NCM paper names no repository. **Marks:** **[T]** full text ·
**[C]** released code · **[ours]** our file:line · **[S1]** SPECULATION with a sourced mechanism, transfer
unmeasured · **[S2]** no source.

## 0. What the words resolve to

1. **At byte granularity the mode switch is a depth decision.** The modes exist because a BPE token is a large,
   costly unit that can carry tool semantics; our unit of text is one byte — one forward, never a tool call.
   "Think latently vs act in text" becomes "another iteration, or emit the next byte": the halt head that landed
   in `3279e14`, off in every preset (`halt_weight = 0.0`, `config.py:276`). **Allocation** is the object.
   2. **We already think more per emitted unit than they do.** Table 7: **170 latent steps vs 302 text tokens**
   (8B), **183 vs 284** (14B) — we run 4 iterations per emitted *byte*, ≈4 latent steps/byte against their
   ≈0.15 [S1: their token ≈4.3 B]. Their causal table agrees depth is what pays: forcing all-text costs **−9.4
   pts** (14B MATH500, 78.8 vs 88.2) but cutting each latent span to **one step** costs **−13.2** (75.0 vs
   88.2) — **Table 5**.
3. **NCM's parallel emission is sound only because of Eq. 5**: `p(r_k | ẑ^(k-1)) = ∏_i p(r_k^i | ẑ^(k-1))`.
   byteflow's T byte logits are the opposite — a causal stack consuming the true prefix, so emitting them
   together is the DSpark **D1** defect (`dspark.py:14-21`), not an NCM analogue. See §8.

## 1. The two papers, verified

| # | claim | source | where measured | transfers? |
|---|---|---|---|---|
| 1 | HybridLink = residual adapter hidden→embedding, `e = h + W2σ(W1h)`, ~`2d²` params | 2610.18861 **§3.1 Eq. 4** | 4B/8B/14B Qwen3-Base | **no as alignment** (§8) |
| 2 | entry = sample `<latent>`; exit = nearest-token readout of `R(h)` vs `</latent>` | **Eq. 5**, App. B.2 | — | **yes**, cheaper at \|V\|=256 |
| 3 | working memory = accumulated KV + positional order, proved lossless | **Eq. 6, Thm A.1, App. A** | theory, no measurement | **no** — KDA state is lossy |
| 4 | HybridSFT = 2-pass self-conditioned, CE at text + cos/MSE at latent (λ 1.0/0.2) | **Eq. 7–8**, Table 10 | 54K trajectories | **no** — no latent targets |
| 5 | exit anchor gets extra regression weight so a sparse stop is not diluted | **App. B.2** | — | **yes**, onto `halt_loss` |
| 6 | HybridRL factorises over sampled text only; **no mode reward** | **Prop. 4.1, Eq. 10–11** | G=8, 512 traj, 150/200 steps | **partly** — replay only |
| 7 | +10.4% avg; −35.5–54.7% tokens; 1.8–2.6× faster; 2.7–3.7× cheaper | **abstract + §1 only** | no table carries the range | **no** — §4 |
| 8 | NCM: multiscale residual VQ, one level per step, codes in-level parallel | 2610.14162 **§3.2 Eq. 3–5** | 88–120M gen, 25–144M tok | **no** — §5 |
| 9 | 21× LM1B / 10× OWT / 31× XSum faster *than the strongest DLM* | **§4.6** | 1×H100, NFE-matched | **no** — diffusion-relative |
| 10 | Sudoku 99.5/96.5/86.1 at **12 NFE** vs FMLM+ 97.7/92.8/81.3 at 16 | **Table 3** | 25M gen, 89-token boards | **the one reasoning result** |

## 2. HybridAgents, decomposed

**2.1 HybridLink, and the code that disagrees with it.** `R(h) = h + W2·σ(W1·h)` (**§3.1 Eq. 4**), "~`2d_h²`
parameters" — at `d = 1024`, **2.10M = 1.5% of 138.4M** (`2d² + 4d`, arithmetic). The ablation is load-bearing:
identity **71.4** → 1-layer 83.6 → 1-layer+res 86.4 → 2-layer 85.8 → 2-layer+res **88.2** (Table 6, 14B
MATH500); residual worth **+4.3% avg**. **The shipped code is not Eq. 4**: `hybridlink.py:23-29` is `post_ln(h +
proj2(gelu(proj1(pre_ln(h)))))` — **two LayerNorms** — tagged `"ln_res_adapter"`. Table 6 has **no LayerNorm
row**, so the checkpoint we would copy runs an un-ablated variant. **Code would have to check** whether the
backbone's input norm re-normalises anyway — `model.py` feeds `run_hybridlink` straight into `inputs_embeds`.

**2.2 The switch — and the code's exit rule is the one to port.** Entry is an ordinary CE target, so the
backbone learns *when* from context (**App. B.2**). Exit per the paper is
`argmax_{v∈V} cos(R(h_t), Emb(v)) = </latent>` (**Eq. 5**). The engine does **not** do that: `engine.py:373`
computes `readout_ids = self.next_logits.argmax(-1)`, `:334` compares that id to `latent_close_id`, and the
cosine bank only renders (`readout.py:36-42`). **The LM head decides the exit.** Also: paper **min 16 / max
256** steps (**§H.3**) vs code **4 / 200** (`engine.py:92-93`); SFT epochs **1/2/1**
(`hybrid_sft/train.py:31`) vs **1/1/1** (Table 10). **A cost fact the paper does not advertise:** the exit test
reads `next_logits`, so a latent step still pays the full vocabulary projection — *not* projection-free, only
sampling-free. At \|V\|=256 that is 0.26 MMAC ≈ **0.09%** of a 138M forward, on the tensor that emits the byte;
theirs ~2.8% per latent step [S1: d≈5120]. Tokenizer-free dominates here and only here.

**2.3 Continuous working memory, concretely.** Not a state vector and not a summary: **one growing
`past_key_values` per trajectory plus a per-slot ledger** (`kv_ledger: list[int | None]`, `None` = latent;
`engine.py:42-43`, init `:230`, forward `:269`). Three sources write into one cache — token embeddings, `R(h)`,
tool observations in `FEED` mode (`:425-429`). Theorem A.1 proves cached and full-prefix runs agree **given
unchanged positional encoding and all prior KV preserved**: a property of exact KV reuse, not of summarised
state. Ours is the per-iteration KDA state carried across bytes (`loopblock.py:487-500`, 393 KB, `kda.py:66`)
— same role, opposite guarantee, so **Thm A.1 does not transfer and must not be cited as if it did**. What
survives: appending an observation costs O(len) forwards, not a re-prefill.

**2.4 HybridSFT — no analogue.** Pass 1 teacher-forced; pass 2 **replaces the ground-truth latent embeddings
with `R(h)` predictions** and trains both parts (Eq. 7–8, Fig. 4). The reason is exposure bias in a continuous
input space (**App. B.1**): a text error changes *which* token is re-embedded, a latent error becomes an unseen
input. We have no latent targets — nothing to teacher-force against. The residue: the exit anchor's target is
`Emb(</latent>)` at extra weight "so that the single exit signal is not diluted" (App. B.2), additive at weight
1.0 (`hybrid_sft/model.py:25,146`) — exactly our halt target's shape: one positive per trajectory, last
iteration only, `−` excluded under `rand_depth` (`halt.py:12-18`).

**2.5 HybridRL — the replay condition transfers, the factorisation does not.** Latent positions have no
likelihood, so `ρ_t` is undefined (Eq. 9); the fix is factorisation over sampled **text** positions (Prop. 4.1 /
Eq. 10), justified because latent continuations have conditional mass 1 under fixed `R` (App. C.1). The update
is clipped-ratio PG over text positions (Eq. 15) with **exact rollout replay** — recorded latent inputs fixed,
`∂z/∂θ = 0` (App. C.2; `replay.py`, latent stack `.detach()`) — giving gradient equivalence to the
advantage-weighted trajectory objective (Thm C.3). Reward `r_correct + r_format + r_tool` (Eq. 11), **no mode
term**; `r_tool` needs *both* `</search>` and `</python>` and a correct answer (`reward.py:150`). Here the
factorisation is vacuous (every byte has a likelihood) but the **replay condition is not**: if the rollout's
depth or carried state is not reproduced at update time, the likelihood is off-policy in a way no IS ratio
covers. `Rollout` records one hex stroke per byte (`rollouts.py:86`) but **not the depth run**.

## 3. Where each piece lands

| mechanism | our seam | what changes | verdict |
|---|---|---|---|
| exit rule → depth decision | `halt.py` (landed, `halt_weight=0`), `loopblock.py:468-478` | nothing new; the head *is* the switch | **build the measurement, not the mechanism** |
| sparse-stop up-weighting (App. B.2) | `halt.py:halt_loss` | one scalar on the last-iteration BCE | **cheapest real import** |
| lossless multi-turn memory (Thm A.1) | `decode.recurrent_decode` | state is lossy; theorem inapplicable | **reject the theorem, keep the turns** |
| persistent KV / state across turns | `serve.py` — `assemble()` rebuilds the byte prompt per request; no `tool_calls` | per-conversation state cache + a tool surface that does not exist; **§H.3's "persistent KV transfer" is too vague to implement**, and `engine.py:256`'s lockstep driver (`latent_rows.index(i)`, `:397`) is positional and unstable — check, do not copy | **the real gap; plumbing** |
| exact replay (Thm C.3) | `rollouts.py` trace → `rl.py` | record depth + states per byte, replay at that depth | **prerequisite for halt-aware RLVR** |
| no mode reward (Eq. 11) | `rl.py:14` `S_sol` thrift proxy | length signal exists; `length_penalty = 0.0` OFF | **agree — add nothing** |

## 4. The token reduction: abstract-only and internally inconsistent

| statement | value | implied reduction | source |
|---|---|---|---|
| text-mode tokens/instance, 8B | 302 vs 596 | **−49.3%** (1.97×) | **Table 7** |
| text-mode tokens/instance, 14B | 284 vs 497 | **−42.9%** (1.75×) | **Table 7** |
| "using 2.7× fewer text tokens" | 2.7× | **−63%** | **§5.2** |
| "reducing total generation steps by up to 54%" | — | −54% *steps* | **§5.2** |
| "1.8×–2.6× faster" vs "2.1× faster inference" | — | — | **abstract** vs **§5.2** |
| decoding compute 4B/8B/14B | 51.2/62.5/66.9 → 13.9/17.2/25.1 TFLOP | **−62.5..−72.9%** | **Table 9** |

The range appears verbatim in the abstract and §1 and in **no table**. Table 7's counts imply **472 steps vs
596** (−21%) while Table 9 claims −62.5..−72.9% compute on the same backbones; never reconciled (Table 7 = 560
math problems, Table 9 = "matched evaluations" excluding prefill, §G.2). **Architecture vs post-training is not
separable either:** no arm trains the hybrid architecture text-only and reports tokens, the only mode ablation
(Table 5) reports *accuracy*, and Figure 9's rising latent share is a plot with no numbers. **Our metric is not
tokens** — the analogue is **bytes-to-solve at matched accuracy**, already `correct@budget(bytes)` (`rollouts.py:10`), plus **block passes per emitted byte** (=`max_iter`).

## 5. NCM against byteflow

**5.1 Different kind of object.** NCM **replaces the data**: a frozen multiscale residual VQ-VAE encodes text
to K levels, the generator emits codes, the tokenizer's decoder maps the final reconstruction to tokens (§3.2–3.4,
Fig. 4) — reconstruction must be near-exact. byteflow **sits inside a predictive model**: the front stage
compresses T bytes to K patch latents, our loop runs on patches, the back stage lifts them to T byte logits
(`byteflow.py:31-45`, option **B**, chosen over **A** "the net REPLACES the stack"). No reconstruction objective
exists anywhere in byteflow — the decoder predicts bytes, it does not restore them. **The repo already rejected
option A**; NCM requires A *plus* a codec.

**5.2 Losslessness, numerically.** Table 1 exact-token reconstruction: **99.99 / 99.86 / 99.98 / 99.79 /
98.79**. Errors are structured, not noise (Table 11: numerals, named entities, diacritics, usernames, markup,
word boundaries); a byte-exact path tolerates none. **How big a floor?** [S1: my arithmetic] one error per ~714
tokens at 99.86%, 10–15 bits per substitution, ~4.3 B/token → **≈0.003–0.005 BPB**, i.e. *at* our paired-eval
floor of 0.002–0.005 BPB (AGENTS §3.1, itself unverified). The real cost is **code-prediction entropy**, and
**no published BPB or bits-per-code for the generator's own code likelihood exists at any scale**.

**5.3 What a second level costs.** Tables 5–6: tokenizer **36.4M–144.0M** against a **88.0M–120.0M**
generator; on OWT the tokenizer (111.2M) is *larger* than the generator (94.65M), at K=32, \|V_k\|=4096, d=384.
One extra level over K=64 patch latents is small in parameters (a 256×1024 codebook is 0.26M) but the cost is
the frozen pretraining run — 10k–120k updates at batch 768–4096 (Table 5), against our 2 000 steps. **No source
measures a second level's marginal value at ≤150M.**

**5.4 The 10–31× is an artefact of the baseline.** §4.6: 11× fewer NFE than DUO on LM1B → 21× faster; 16×
fewer than AURORA on XSum → 31× faster. Both baselines **reprocess the full sequence every step**. Against AR
it inverts or vanishes: WMT14 **22.6 BLEU at 11 NFE** vs **AR 25.2 at 64** (Table 2) — NCM *loses* on
translation quality — while XSum wins (28.0 vs 24.4 ROUGE-L). **No wall-clock against any AR model is
reported**, so **no reference exists** for "10–31× faster than an autoregressive byte model", and our decode is
already O(1)/byte. The honest mapping of "fewer forwards per emitted unit" is **spec decode**, exact by the
Leviathan residual rule (`specdecode.py:29-35`) — unlike NCM, valid only because of Eq. 5.

**5.5 The one NCM idea worth measuring.** Sudoku (Table 3) is the only *knowledge-free* reasoning result in
either paper: 25M params, 12 NFE, 99.5/96.5/86.1 against AR's 14.6/5.1/1.0, via **revision** — later levels
correct earlier partial solutions (§4.4), which left-to-right AR cannot do. Our loop cannot revise an emitted
byte either; the only revision path is the verifier rejecting the attempt. That is the gap worth naming, and a
VQ hierarchy at our scale does not buy it.

## 6. Ranked by expected value for logic-without-knowledge

Ranked by *(reasoning not recall) × (fits our seams) × (cheapness)*; `medium` A/B = 2 000 steps ≈ 15 min/seed.

| # | mechanism | source | effect | fit | cost | verdict |
|---|---|---|---|---|---|---|
| 1 | **Depth allocation** — the switch in our unit: halt head + sparse-stop up-weighting | **Table 5**, **App. B.2** | depth > mode, causal | head landed, `halt_weight` off | 15 min × 3 × 2 | **arm M1** |
| 2 | **State carry across turns** — Thm A.1's engineering content without its guarantee | **Thm A.1, §H.3** | theory only, unmeasured | `recurrent_decode` + `rollouts.py` exist | 0 to gate | **arm M2, gated** |
| 3 | **Exact replay with recorded depth** | **Alg. 1, App. C.2** | needed for halt-aware RL | trace exists, depth does not | ~1 day | **prerequisite** |
| 4 | Persistent state + tool surface in `serve` | **§H.3** | none (plumbing) | `assemble()` re-renders per request | ~1 day | **when an agent needs it** |
| 5 | Byte-position CE within a patch (NCM diagnostic) | 2610.14162 **Eq. 5**, Table 1 | no direct source | one eval + ~20 lines | **0** | **free gate on NCM** |
| 6 | Exit-anchor weighting alone | **App. B.2** | unmeasured alone | one scalar | in M1 | **part of M1** |
| 7 | Multiscale residual VQ (second level) | **§3.3–3.4**, Table 5 | K sweep only, TinyStories | needs option A + codec | 10–100× budget | **reject** |
| 8 | HybridSFT self-conditioning | **Eq. 7–8** | +30.5% over activation-only (§6.1) | no latent targets | n/a | **reject** |
| 9 | HybridLink as alignment | **Eq. 4**, Table 6 | identity 71.4 → 88.2 | no pretrained embedding space | 10 min × 3 | **reject as alignment** |

## 7. Pre-registered ladder — three seeds, 2 000 steps

Fixed: `medium`, b8·s768, unchanged lr and shard, `eval_batches` fixed, byte count printed. **Bar = the
control's own 3-seed range**; expected effects are **OUR hypotheses**, not the papers'.

**Step 0 — the free gate, before any arm.** `eval_depths=true` (`config.py:1031`; curve at
`train/loop.py:420,435`, printed by `_eval_line:460`) on an existing `medium` checkpoint — the CLI's `eval` does
**not** pass `depths=True` and discards `_per_iter` (`cli.py:649-690`), so it needs a training-lane run or one
flag. **Prediction:** BPB falls T=1→2 and is flat within the seed range by T=3–4. Still falling at 4 ⇒ depth is
the lever, the halt head premature (raise `max_iter` first); flat at 2 ⇒ the loop is over-provisioned 2×. **This
outranks both arms.**

- **M1-A (control).** `max_iter 4`, `halt_weight 0.0`, fixed depth, 3 seeds × 2 000 steps.
- **M1-B (halt).** `halt_weight 0.1`, `halt_eps 0.005`, `halt_min_iter 1`, `halt_mode` **on in the eval loop only**
  (training adds only the BCE term), last-iteration stop label up-weighted 2× per App. B.2. **Metric:** BPB at
  fixed bytes/step, mean iterations used, skip rate. **Expected:** BPB within A's range, mean depth ≤85% of A.
  **Kill:** BPB worse than A's range, **or** skip <10% — not difficulty-aware, head decorative.
- **M1-C (compute-matched control — the one that decides).** A's weights, **fixed depth 3** = the mean depth M1-B
  actually used. If C ties B the *decision* bought nothing beyond less compute: delete the head, ship the
  constant. This separates "adaptive" from "shallower".
- **M2-0 (free capability gate).** SFT checkpoint, ~50 verifier tasks, **one turn vs two turns with state
  carry**, no training, no new params. **Prediction:** 1-turn pass rate near zero at 138M; if so M2 stops here
  for free.
- **M2-A/B (only if M2-0 passes).** A = today (re-prefill per turn, `rollouts.py` → `decode_bytes`); B =
  `recurrent_decode` carrying state across verifier feedback. **Compute-matched control:** give B one extra
  iteration per byte so its forward count matches A's re-prefill — otherwise the arm is "more compute", not
  "carried state". **Metric:** verifier pass rate at fixed byte budget. **Kill:** within seed range.

**Resolution warning, before the run.** 2 000 steps of `medium` sees ~12.3 MB; paired-eval resolution is
0.002–0.005 BPB **and explicitly unverified** (AGENTS §3.1). Every effect above sits at that floor: a tie is a
deletion, and only a survivor earns the 10 000-step run.

## 8. Explicit rejections

| rejected | why (sourced) |
|---|---|
| **NCM's multiscale VQ touching emission** | Needs option **A** — the model replaced by a codec (`byteflow.py:31-45` records we chose **B**). Tokenizer 36–144M against a 88–120M generator (Tables 5–6). 10–31× is diffusion-relative and NCM *loses* to AR on WMT14 (22.6 vs 25.2, Table 2). No source measures a second level at ≤150M. |
| **Block-parallel emission from byteflow's back stage** | Its T byte logits consume the true prefix (causal stack, `byteflow.py:1-22`). Emitting them together is DSpark **D1** (`dspark.py:14-21`), not NCM's Eq. 5: NCM is sound only because within-level codes are conditionally independent *given the reconstruction*. |
| **HybridLink as "alignment into the input space"** | Justified by a *pretrained* embedding distribution (§3.1, App. B.1). We train from scratch; no target distribution exists, and the loop's input is already a hidden state plus an iteration embedding (`loopblock.py:5`). Table 6's 71.4→88.2 is a 14B-pretrained result; citing it as our effect is fabrication by proxy. |
| **Theorem A.1 (lossless working memory)** | A theorem about preserving *all* KV with unchanged positional encoding (App. A, induction over layers). Our state is 393 KB fixed-size (`kda.py:66`), lossy by construction — it proves a property we deliberately do not have. |
| **HybridSFT / self-conditioned training** | Requires latent regression targets `Emb(y_lat)` and a teacher-forced first pass (Eq. 7–8). We have no latent targets: the loop trains only through byte CE. |
| **Anything from NCM needing a token vocabulary** | Its framing is tokens: \|V\| 8 192–151 680, levels over 8–128 **tokens** (Table 5). The byte-level equivalent needs 4–6× longer sequences and a codebook per level; porting the schedule means porting the tokenizer — **and that breaks the byte-level design.** |
| **The 35.5–54.7% as an expected effect** | Abstract and §1 only; contradicted by Table 7's own counts (1.97×) and §5.2's 2.7×; not decomposable anywhere. |
| **HybridRL's Prop. 4.1 as a novelty** | It exists because latent positions lack a likelihood. Every position here is a byte with a likelihood; the factorisation is vacuous. Only Thm C.3's replay condition transfers. |
| **Anything needing 32 H100s, seq 16 888, 54K trajectories, 512-traj rollouts** | §H.3. One 16 GB card, one process (AGENTS §1.5 forbids a second GPU process), 2 000-step A/Bs. |

## 9. VERIFIED vs SPECULATION

**VERIFIED.** Every number names its table, section or equation. Four paper/code divergences, each `file:line` in
`togethercomputer/HybridAgent`: `hybridlink.py:23-29` (LayerNorm variant vs Eq. 4) · `engine.py:373,334` (LM-head
exit vs Eq. 5) · `engine.py:92-93` (4/200 vs §H.3's 16/256) · `hybrid_sft/train.py:31` (1/2/1 vs Table 10).
Our seams are cited `file:line` in §3 and §8. `benches/history.tsv` records byteflow vs plain as **0.481 BPB at
9.2M** (3.466 vs 3.947) — **not** at 138M.

**SPECULATION.** **S1** the ≈0.003–0.005 BPB codec floor (§5.2): my arithmetic from Tables 1 and 11, ~10 bits per
substitution, ~4.3 B/token — no source computes it, likewise the 2.8%-of-forward exit projection (§2.2, assumes
d≈5120). **S2** their ≈0.15 latent steps/byte, hence our 27× higher thinking-per-emitted-unit. **S3** every
expected effect and threshold in §7. **S4** the LayerNorm argument (§2.1).

**Confidence.** **HIGH** that the mode switch is a depth decision at byte granularity; that the released exit rule
and adapter differ from the paper; that NCM's parallelism rests on Eq. 5; that 35.5–54.7% is not checkable from
any table. **MED** that the halt head skips ≥15% at a tie (anchor: our 27.6% MSA figure, owner-reported and *not
reproducible from this checkout*). **LOW** on every 138M number and the codec-floor arithmetic.

## 10. Open questions, and one discrepancy to fix first

1. **Two of the brief's numbers are not what the archive says.** `benches/history.tsv` records byteflow at **0.481
BPB at 9.2M** (3.466 vs 3.947), not 0.485 at 138M; and no `K=64` row exists (`configs/byteflow.toml` is K=128/512
B, port default K=3200/8192 B, `config.py:401`). Both change §5.1's cost argument.
2. **No reference exists** for a hybrid latent/text agent at ≤1B, a byte-level hybrid agent, a lossless (100%)
multiscale VQ tokenizer, or a second VQ level's value at our scale — nor has **our depth curve ever been read**.
3. Does `S_sol` move bytes-to-solve when RLVR first runs, and does a 138M byte model have any 2-turn revision
capability? M2-0 answers the second for one eval; if either is no, M1 loses its post-training half.

## Sources

**Full text [T]:** 2610.18861 — §3.1 Eq. 4–5, §3.2 Eq. 6, §4.1 Eq. 7–8, §4.2 Prop. 4.1 Eq. 10–11, §5.1–5.2,
§6.1–6.3, Tables 1–2, 4–9, App. A (Thm A.1), B.1–B.2, C.2 (Thm C.3), E (Eq. 18), F (Eq. 19–20), G.2, H.3.
2610.14162 — §3.1 Eq. 1–2, §3.2 Eq. 3–5, §3.3 Eq. 6–9, §3.4 Eq. 10, Alg. 1–2, §4.1–4.6, Tables 1–3, 5–9, 11,
App. C.2, F.1–F.3.
**Code [C]:** `togethercomputer/HybridAgent` @ `main`, pushed 2026-10-18 — `hybrid_agent/{hybridlink,engine,
readout,model}.py`, `hybrid_sft/{model,train}.py`, `hybrid_rl/{reward,replay}.py`. NCM: no repository named.
**Repo [ours]:** `src/dormouse/model/{halt,loopblock,kda,byteflow,dspark,specdecode,decode}.py` ·
`{config,cli,serve}.py` · `train/loop.py` · `post/{rollouts,rl,verifiers}.py` · `benches/history.tsv` ·
`2026-10-21-adaptivity.md` (§4 cost, §6.2 the unreproducible 27.6%) · `2026-10-18-patch-size.md` · ADR-0013,
ADR-0021 · `AGENTS.md` §1.3, §1.5, §3.1, §3.7. `docs-site/tools/manifest.mjs` owes no line: `docs/research/*.md` is globbed at `:334`.
