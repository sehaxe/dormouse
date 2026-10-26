# Shared memory in looped Transformers — what the paper does, what transfers to a KDA stack, and the one gate that decides it

**Date:** 2026-10-23. **Lane:** read-only memo. No source file was modified; nothing was run (the card is
held by a training wave, AGENTS §1.5). **Paper:** Monea, Ramji, El-Kurdi, Lastras, Artzi, Godey,
Astudillo, *The Surprising Effectiveness of Shared Memory in Looped Transformers*, arXiv:2610.02383v1,
submitted 1 Oct 2026, 822 KB (Cornell + IBM Research).
**Read in full:** abstract, §1–§8, Appendices A–F, Tables 1–8, Figures 1–17, from the HTML build
(`arxiv.org/html/2610.02383v1`) and the abs page. The HTML carries no page numbers, so references are
section / table / figure numbers, which are stable against the PDF.
**Repo facts** were read from the files named inline, not assumed.

**Marks.** **[V]** = verified this session in a source I read (file:line, or paper §/Table). **[V-p]** =
verified in the paper. **[S]** = my inference, SPECULATION, no direct measurement. Every row of every
table carries one; the two lists at the end separate them again.

---

## 0. TL;DR

1. **The mechanism is exactly as briefed, and it works** — first recursion writes the only full-context
   KV cache, later recursions attend to it *plus* a 64-token local window in one softmax [V-p §3,
   Eq. 3]. `h-lpt` at 1B / 5 recursions: **10.23 PPL** vs **11.35** for a same-size single-pass
   Transformer, at **0.24×** context memory [V-p Table 1, L row].
2. **But the headline 1.12 PPL is not the sharing effect.** It is `h-lpt` vs `std`. Against its own
   looped baseline `lt2` the same cell is **10.23 vs 10.32 = 0.09**, and the paper's own paired test at
   S gives **−0.13 ± 0.02** for `h-lpt` vs `lt2` at R=5 (p = 1.6e-4) [V-p Table 1, Table 4]. Against
   `lt` with the *best* of six normalizations the gain is **0.06–0.43** [V-p Appendix C]. The
   defensible effect is **0.06–0.43 PPL**, not 1.12–1.82.
3. **Our memory motivation is nil.** KDA state is `2·b·H·hd²` fp32, independent of `seq_len`
   (`src/dormouse/model/kda.py:66-67`): **4.2 MB at medium b8**, against **33.5 MB** for a KV
   cache of one equivalent layer at the same 1024-byte window. We have no KV cache to share, so only
   the *quality* claim (and, separately, a compute claim the paper does not make) can transfer.
4. **The obvious port of the idea is non-causal and must not be run as specified.** `initial_state=`
   takes the state *after* the whole sequence, so iteration 2 at byte 0 would read a summary of bytes
   it must predict. That is target leakage, it is *learnable* (the same decay gates that set memory
   length set the leak), and a 600-step A/B would score it as a WIN. This is the §3.2 defect class with
   a new cause. Details and the algebra in §3.0.
5. **There is a causal formulation, and it costs 3 lines** — every iteration `i>0` starts from
   `states[0]` (the state *entering* the window), not from its own: one writer, N−1 readers, which is
   the paper's *role* split without any new parameters. It is only expressible in the segmented /
   decode paths, and `segment_len` refuses MSA and capacity-MoE (`loopblock.py:658-674`) — so the A/B
   costs those two arms. Verdict in §8: **not ahead of the queue; one day of CPU work, no wave.**

---

## 1. What they do, exactly

### 1.1 The mechanism [V-p]

Let `C(t)` be the KV cache a layer writes at recursion `t`, `ω` the local window (default **64**), `R`
the recursion count. A query at position `i` in recursion `t` reads

```
A_lpt(t, i) = C(1)≤i                       t = 1   (state recursion)
            = C(1)≤i  ∪  C(t)[i−ω, i]      t = 2..R (prediction recursions)      §3 Eq. 3
```

Both sets go through **one softmax**, one attention op. Training: the shared and local K/V are
concatenated along the sequence axis and a FlexAttention block mask skips fully-masked blocks
[V-p Appendix A "Attention Implementation"]. Decoding: a sliding buffer of ≤ ω entries per layer, the
current entry attending before it enters the buffer [V-p §3 "Efficiency and Costs"].

- **`h-lpt`** = the same rule on the *full-attention layers only* of the `lt2` hybrid (Gated DeltaNet +
  1 full-attention layer per 5); the GDN layers keep their own recurrent state in later recursions,
  which is "small and fixed in size" [V-p §3, last paragraph; Appendix A].
- **`lpt`** = the rule on a pure full-attention looped stack; the local cache is truncated to ω, which
  is what buys the memory [V-p §3, Eq. 4: `M_KV^lpt = L[T + (R−1)ω]` vs `M_KV^lt = L·R·T`].
- The 1.25 % figure is the extra *attended entries* per generated token at T=4096, ω=64, R=5
  [V-p §3] — i.e. **the local window saves memory, not compute.**

### 1.2 The numbers the brief asked for — verified, with the ones it missed

| claim in the brief | status | paper |
|---|---|---|
| 1B, 5 recursions, 10.23 vs 11.35 PPL | **[V-p] but mis-attributed** — 10.23 is `h-lpt`, 11.35 is `std` at R=1 | Table 1, L |
| context memory 0.24× | **[V-p]** `h-lpt`, L, R=5, KV 0.24 | Table 1 |
| window ablation: 4096-token local cache still 0.24–1.15 better | **[V-p]** "lpt still lowers perplexity by 0.24–1.15 relative to lt", i.e. access to the shared cache is what matters, window size is second-order | §6 "Prediction Recursion Needs Local Memory", Fig. 4 |
| attention share to shared memory 57–71 % early, 73–80 % final | **[V-p]** | §6.3, Fig. 5 |
| 4.3–4.8× std decoding FLOPs at R=5 | **[V-p]** | §8 "Limitations" |
| one corpus, 4096 contexts, no long-context tests | **[V-p]** | §8 "Limitations" |
| "no wall-clock claims" | **[V-p] FALSE** — Appendix F measures H100 decode latency and tok/s (Table 8), and the paper's own caveat is that it is *one research implementation, not a serving engine* | Appendix F |

Two more, and they change the size of the prize:

| what | number | where |
|---|---|---|
| self-attributed sharing gain, hybrid, R=5 | S **−0.15**, M **−0.12**, L **−0.09** PPL vs `lt2`; paired at S **−0.13 ± 0.02** | Table 1, Table 4 |
| lpt's own marginal recursion 3 → 5 at 1B | **10.45 → 10.32 = 0.13 PPL for +67 % decode FLOPs** | Table 1, L |
| seed evidence | **S only**, 4 seeds; largest SD 0.08, all reported gaps 0.40–1.05 keep sign. M and L are **one run each** | Appendix B |

The 3→5 marginal is the number that matters most for us: at 1B the paper's own model buys 0.13 PPL
from two more recursions. Our 1B run buys **0.006 CE** from iteration 4
(`~/logs/b1first_resume2.log:906`, `periter=2.1503,1.8668,1.8163,1.8107`). Both stacks are already
deep in diminishing returns at that depth, which is exactly the regime the paper claims to fix.

### 1.3 Why they say it works [V-p §6]

1. **It stabilises depth**: `lt` with final loss collapses to unigram loss at R=5; `lpt` does not, and
   trains best with final loss. Under *adaptive* loss (Ouro's halting objective) the sharing gain
   shrinks to **0.16–0.50 PPL** [V-p §6.1, Fig. 3/4].
2. **Shared and local keys form distinct clusters** in PCA — sharing is not riding on representational
   similarity [V-p §6.4, Fig. 6].
3. **Gradient highway**: reading the shared cache gives the final loss a direct path to recursion 1;
   detaching it raises PPL at every size/depth and deeper detached runs fail to train
   [V-p §6.5, Table 6].

Point 1 is the load-bearing one **and it is measured against an objective we do not use.** Our loop
accumulates an unweighted per-iteration CE (`loopblock.py:487-496`, ADR-0013's deep supervision), which
is closer to their *adaptive* regime than to final loss — the regime where their gain is 0.16–0.50 PPL,
not 1.09. So the transferable claim is the small one.

---

## 2. Map onto this repo

### 2.1 The memory half does not transfer at all [V]

| | paper | us |
|---|---|---|
| attention | full softmax + KV cache per recursion | **KDA only** (`kda.py:191-478`; `use_kda` default true, `configs/medium.toml:29`) |
| context memory | `L·R·T` entries, grows with depth | `2·b·H·hd²` fp32, **independent of `seq_len`** (`kda.py:66-67`) |
| at medium b8 s1024 | — | **4.2 MB** per layer, flat; an equivalent one-layer KV cache would be **33.5 MB** and grow with t |
| the paper's problem | "context memory can grow larger than the weights" [§1] | cannot happen here |

So: **no KV-cache accounting, no memory claim, no "76–79 % less memory"**. The one arm of ours that
does keep an O(n) context is MSA (`sparse/attention.py`), a ¼-capacity auxiliary arm; building a
cross-recursion cache for it would be new machinery for the arm the battery already called a tie
(`docs/reviews/2026-10-21-battery.md:20`). Do not.

### 2.2 Our loop's state semantics — and a lane divergence [V]

Each iteration runs **its own full-sequence KDA pass with a fresh state**: `_window` calls `_attn(...,
carry=False, state=None)` (`loopblock.py:403-411`), which calls `self.attn(normed)` →
`forward_state(x, None, carry=False)` → `chunk_kda` **without** `initial_state`/`output_final_state`
(`kda.py:384-396`). The state is per-iteration-slot *by design* in the segmented path, and the code
says so: "one state for the whole block would mix the passes, which is the same defect as sharing one
hidden state across them" (`loopblock.py:571-574`).

**Divergence worth recording:** the Rust crate *does* chain the O(1) state across iterations —
`let mut kda_s = kda_state` (`crates/dormouse-core/src/loop_block.rs:717`), passed in with
`kda_s.take()` and written back `kda_s = Some(s_new)` (`:913`, `:922`) — with the state shape
`[B,H,K,V]` (`vendor/dormouse-fused/crates/burn-kda/src/lib.rs:803-822`), i.e. **variant (a) below is
already the crate's behaviour and is not the port's.** Two consequences: (i) "state carry" is not a new
idea here, it is a lane difference; (ii) if the crate's carry is the non-causal one (it is: it passes
the state *after* the window into the next iteration, `:922`), then the leak class in §3.0 has been
running in that lane and has never been gated. The crate's numbers are not evidence either way (the
attention arm had no gradient for most of that history, AGENTS §3.2), but the *rule* needs one
definition, not two.

### 2.3 The economics we actually have [V]

| quantity | value | source |
|---|---|---|
| one iteration at `small` 9.2 M | **18.1 % of launches, ~24.5 % of wall-clock** (142 ms of 581.5) | `docs/research/2026-10-21-adaptivity.md:44-46` |
| one iteration at `medium` 142 M b16 s1024, GPU 100 % | **≈119 ms of 573 ms** ((194.5+282.0)/4) | same file `:87-90` |
| the loop's share of a step (ByteFlow profile) | **95.9 %** (`encode 2.3 % · loop 663 ms · decode 1.8 %`) | `docs/reviews/2026-10-21-battery.md:60` |
| our per-iteration CE, medium, step 34000 | 2.1536 / 1.8070 / 1.7478 / 1.7131 → marginal **0.35 / 0.06 / 0.035** | `~/logs/medium_long.log:355` |
| our per-iteration CE, 1B, step 26000 | 2.1503 / 1.8668 / 1.8163 / 1.8107 → marginal **0.28 / 0.05 / 0.006** | `~/logs/b1first_resume2.log:906` |
| `medium` preset | 141 910 405 params (r64), **max_iter 4**, d 1024, H 16, hd 64, MSA on | `docs/reviews/2026-10-23-architecture-verdict.md:13`; `configs/medium.toml:18-24` |
| control BPB on real prose | **2.650** (2.653 / 2.659 / 2.638, spread **0.021**) | verdict `:24` |

Two regimes, and they price the same change differently: at `small` the step is launch-bound (87 %
idle, AGENTS §3.1), at `medium` it runs at 100 % GPU and a skipped token row really is a saved FLOP
[V: adaptivity `:84-90`]. Any compute argument below must say which regime it is in.

---

## 3. The question that matters: cross-iteration state sharing

Four designs, cheapest first, then the comparison table. The ordering is *not* the brief's, because the
brief's (a) and (b) share a defect that outranks their cost.

### 3.0 The defect: a KDA state is a *summary of the whole window*, so reading it is reading the future

The recurrence (`kda.py:167`):

```
S_t = Diag(exp g_t)·S_{t−1} + β_t k_t (v_t − k_tᵀ Diag(exp g_t) S_{t−1}) ,   o_t = scale·q_tᵀ S_t
```

`S_T` is a decayed sum over **all** positions `1..T`. Passing `initial_state=S_T` makes `o_t` a function
of `x_j` for `j > t`, with weight `∏_{u≤t} exp(g_u)`. Three consequences, in order of how much they cost:

1. **It is leakage, not regularisation.** Iteration `i>0` can lower its CE by reading bytes it is asked
   to predict. The 600-step A/B would report a WIN [S1].
2. **The optimizer can amplify it.** `g = logsigmoid(W_g x)` (`kda.py:248`), so the same parameter that
   sets memory length sets the leak length: at init `g ≈ −0.693`, `exp g ≈ 0.5`, and the leak is
   `~0.5^t` (negligible past ~20 bytes) — but a gradient that lengthens memory also lengthens the leak,
   and the leak *reduces the loss*. "Tiny at init" is not a bound on anything [S1].
3. **A null result would be uninformative.** At init the carry is nearly a no-op in the forward *and*
   nearly a no-op in the leak, so a tie tells us nothing about the causal version either [S1].

The paper does not have this problem because its shared memory is addressed: `C(1)≤i` is the causal
*prefix* of the cache, so recursion 2 at position `i` sees exactly positions `1..i` — the same set
recursion 1 saw [V-p Eq. 2/3]. **A KV cache is a set of entries with a causal index; a KDA state is one
vector with no index.** That is the whole difficulty of the port, and it is why the paper's numbers
cannot be carried across by analogy.

Corollary for the decode path: in a byte-stream model the state at the end of a *window* and the state
at a *byte* are different objects, so "iteration 1's state" means different things in train (window
end) and decode (previous byte). Any variant whose shared state is neither a window prefix nor a byte
prefix has a train/decode mismatch on top of the leak [S1].

### 3.1 (a) state carry — `initial_state = ` iteration 1's final state

As briefed: iteration 2 continues from iteration 1's `output_final_state`; ~6–10 LOC
(`kda.py:397-408` already takes both arguments).

- **Qualitatively:** yes, iteration 2 then sees iteration 1's context "as its own history" — but as a
  *decayed* history with no addressing: it cannot ask for a position, only start from a weighted sum,
  and its own gates decide how much of that sum survives. It is history as a prior, not memory as a
  lookup.
- **Cost zero** (one extra elementwise seed of the state per layer, ≤ 4 MB read).
- **Verdict: reject as specified** (leak, §3.0). Note it is *already* the crate's behaviour
  (`loop_block.rs:913-922`) and not the port's, so the honest first action is a **test that tells the
  two lanes apart**, not a new arm.

### 3.2 (b) shared-state read — every iteration reads the SAME iteration-1 state

Same leak, same cost, one difference that matters: `(a)` chains (`S⁽²⁾_T` feeds iteration 3, a summary
of a summary), `(b)` fans out (`S⁽¹⁾_T` feeds all). `(b)` is the paper's shape and is strictly more
faithful at identical cost, so if the leak were acceptable, `(b)` dominates `(a)`. It is not.
**Verdict: reject as specified, same reason; prefer it over (a) only inside a leak-measurement probe.**

### 3.3 (b′) the causal version — readers start from the state *entering* the window

The only formulation that is causal, train/decode identical, and expressible with today's API: in the
segmented and recurrent paths, iteration `i>0` starts each window from **`states[0]`** (the state
iteration 1 received), not from `states[i]`.

- **What it means:** one writer (iteration 1 owns the memory), N−1 readers (later iterations re-read the
  same prefix every window and keep no cross-window memory of their own). That is the paper's *role
  split* with zero new parameters and zero new state.
- **Causality:** exact — `states[0]` is by construction a function of bytes before the window only
  (`loopblock.py:571-574`, `_forward_segmented`'s graph edge). The gate in §5.3 passes.
- **Cost: 3 lines** in `_window` (`states[i]` → `states[0]` for `i>0`), plus one flag and one counter.
  No new kernel, no new tensor, no extra launches. The gradient through iterations `i>0`'s own past
  disappears, which makes the graph *shorter* (a small speed win at long `segment_len`).
- **The catch, and it is the deciding one:** in the one-shot path `states is None`, so the flag is a
  **no-op** (`segment_len=0` is the default everywhere). To exercise it you must run `segment_len > 0`,
  and `_forward_segmented` refuses `use_msa` and `moe_capacity>0` (`loopblock.py:658-674`). `medium.toml`
  has both on (`:36`, `:48`). So the A/B is "share on/off, both segmented, MSA off, dropless" — **two
  knobs move to test one**, and the baseline they create is not the recipe on record.
- **Verdict: the right design, the wrong price for a 600-step screen.** It is the variant to build if
  the hypothesis is taken seriously at 4+ iterations, and the price must be paid explicitly (§5.4).

### 3.4 (b″) chunk-lagged read — the faithful version, if someone wants it

Iteration 1 is run chunk-wise (`T/chunk_kda` calls, `output_final_state=True` each) and its
chunk-boundary states are stored; iteration `i>0` starts chunk `c` from iteration 1's state at `start(c)`.
Causal, granularity = the chunk (16–64 bytes, i.e. *narrower* than the paper's 64-token window), config
neutral (no `segment_len`, MSA and capacity stay on).

- **Cost [S]:** ~40–60 LOC; 8–16 extra **sequential** kernel calls per iteration-1 pass (at s1024,
  d 1024, 1 KDA layer per iteration — one layer, so 8–16 calls, not hundreds); stored states
  `b·H·hd²·4 B · T/chunk` = **2.1 MB × 16 = 33 MB at medium b8** [S, arithmetic]. At `medium` (GPU
  100 %, 573 ms/step) the extra calls should be a few ms; at `small` they are launch-bound and would be
  ~0 — the flag would do nothing there.
- **Verdict: the only variant whose *within-window* behaviour matches the paper's shape.** Worth
  building only after (b′) has shown the direction is alive, because it is 10× the code for a
  finer-grained version of the same signal.

### 3.5 (c) local window for later iterations — the compute idea

Later iterations process only a `W`-byte window while reading the shared state.

- **It is not the paper's idea.** In the paper `ω` is the *local* cache; every prediction recursion
  still attends over the **full** shared prefix, so `lpt`'s FLOPs equal `lt`'s (Table 1: 4.34 vs 4.32 at
  R=5) [V-p]. The paper has no compute-saving variant; it saves memory. **[V-p]**
- **Wall-clock, with our numbers [S, arithmetic shown]:** at `medium` a step is 573 ms of which
  ~2×119 ms is iterations 2–3. With `W = 256` of 1024 positions, the *token-scaled* part of those
  iterations falls 4×: saving ≈ `2·119·0.75 = 178 ms = 31 %`. **Upper bound** (window = 0) is
  `2·119/573 = 41.5 %`; the honest estimate is **20–25 %** because 256-token GEMMs lose efficiency.
  At `small` the saving is ≈ **0**: the launch count does not drop, only the grid size, and that step is
  launch-bound (18.1 % of launches for 24.5 % of wall, §2.3).
- **What breaks [S, and the second reason to reject]:** the readout is the **mean over iterations**
  (`loopblock.py:488`), and every iteration also carries a CE (`:491-496`). If iterations 2–3 run on ¼ of
  the positions, then on ¾ of the bytes the mean readout is a *depth-1* model — a different network with
  a different objective, and the per-iteration CE would be a mean over a different position set. For a
  byte stream there is no natural "local window" semantics: 256 bytes is ~45 English words, which cuts
  mid-sentence and mid-word, so long-range dependencies (the thing the loop is supposed to buy) are
  exactly what gets dropped, and the Engram's causal n-grams (`_memory`'s `drop`, `loopblock.py:584-611`)
  would need their own left context per window or the keys silently change.
- **Verdict: reject.** It is a different model, not a knob, and it is not the paper's.

### 3.6 Comparison

| variant | causality | LOC | extra launches | extra memory | quality mechanism | verdict |
|---|---|---|---|---|---|---|
| **(a)** final-state carry | **leaks the future** [V: `kda.py:167` algebra] | 6–10 | 0 | ~0 | history-as-prior, decayed, unaddressable | **reject**; already the crate's behaviour (`loop_block.rs:913-922`) — gate it |
| **(b)** fan-out read of the same final state | **leaks** | 6–12 | 0 | ~0 | paper's shape, our read | **reject** (same defect; dominates (a) only inside a leak probe) |
| **(b′)** readers start from `states[0]` | **exact** [V: `loopblock.py:571-574`] | **3** | 0 | 0 | one writer / N−1 readers, no new params | **recommended**, but needs `segment_len>0` and costs MSA + capacity-MoE |
| **(b″)** chunk-lagged read | **exact**, chunk-granular | 40–60 | +8–16 (seq.) | ~33 MB @ medium b8 [S] | closest to Eq. 3 | build only if (b′) lives |
| **(c)** local window on later iterations | exact (it is local) | 20–40 | −(grid size only) | 0 | none — it is a compute cut | **reject**: different model; 20–25 % at medium, ~0 at small; not in the paper |

---

## 4. What must NOT be copied

1. **The KV-cache accounting** — `M_KV = L[T+(R−1)ω]` (Eq. 4), the 0.18–0.24× memory column, the "fits
   3.6–5.0× more sequences" claim (Appendix F). We have no cache; our state is O(1) and 8× smaller than
   *one* equivalent layer's cache at the same window (§2.1). Nothing in that column is a decision for us.
2. **The "state recursion writes / prediction recursions read" naming.** Our iterations are symmetric:
   the same block, the same parameters, the same readout slot each pass (`loopblock.py:369-510`). Calling
   iteration 1 "the state recursion" would smuggle in a role the port does not have — and if (b′) is
   tried, that name describes the *rule*, not a new module.
3. **Any memory-saving claim.** Even for `h-lpt` the memory win is the hybrid's; for us the only
   transferable claims are quality (§1.3) and, for (c), compute.
4. **The final-loss training recipe.** Their `lpt` trains with final loss and their stability result is
   *against* final-loss `lt`, which collapses at R=5. We use deep supervision (ADR-0013). Their own
   adaptive-loss numbers (0.16–0.50 PPL) are the ones that apply to us [V-p §6.1].
5. **The 1.12–1.82 "vs std" framing.** It is an architecture comparison (hybrid + loops + sharing), not
   the sharing effect. Their own `lt2` control is 10.32 at the same cell.
6. **The "gradient highway" argument, as stated.** It is about routing gradient to *recursion 1's
   activations*, which in a **weight-shared** block are the same parameters as everyone else's; the
   chain-shortening still applies in kind, but the credit-assignment reading does not transfer at face
   value [S].

---

## 5. Gate + A/B spec for the best variant (b′)

Nothing here is run — the card is held. §5.1–§5.3 are CPU-only and can land today; §5.4 needs the card.

### 5.1 Bit-identical when off — the exact pin

The off path must be **the same call with the same arguments**, not an equivalent one. `kda.py` has two
argument sets: `carry=False` calls `chunk_kda` with neither `initial_state` nor `output_final_state`
(`:384-396`); `carry=True` calls it with both (`:397-413`). `initial_state=None + output_final_state=True`
is *not* the same call and may select a different kernel variant → not bit-identical. So:

- flag off ⇒ `_attn` must reach the `carry=False` branch unchanged; assert it at the call site with the
  existing counter (`fused kda=` on the eval line, AGENTS §1.1).
- gate: fixed seed, fixed 2-batch fp32 window on **CPU**, `torch.equal` on logits against a golden tensor
  captured from the pre-change commit. Named `test_state_share_off_is_bit_identical`. A tolerance gate is
  not acceptable here: the claim is parity, and a `1e-5` assert would pass for a different kernel.
- second pin, free: with the flag **on** but `segment_len = 0`, the model must also be bit-identical
  (the one-shot path has no state, §3.3). That is the "arm cannot fire where it has no meaning" pin, and
  it is what keeps every existing one-shot number valid.

### 5.2 Refusals (LOUD, naming both escapes — AGENTS §1.1)

| pair | why it must refuse | escape |
|---|---|---|
| `share_state=true` **and** `segment_len=0` | the arm cannot fire; silently doing nothing is the silent-fallback class | `segment_len` in `(0, t)`, or `share_state=false` |
| `share_state=true` **and** `use_msa=true` | `_forward_segmented` already refuses MSA because the selector works inside one window (`loopblock.py:658-664`); the share rides the same path | `use_msa=false` |
| `share_state=true` **and** `moe_capacity>0` | capacity dispatch makes the expert set a function of the window's composition (`:665-674`); segmentation already changes it by 0.39 on the readout | `moe_capacity=0` (dropless) |
| `share_state=true` **and** `use_byteflow=true` | ByteFlow positions are patches, not bytes; a "window" is a patch count and the readout's per-position mean has a different support | `use_byteflow=false` |
| `share_state=true` **and** `halt` at inference | the halt head reads per-iteration readouts and may skip iterations; a reader that is skipped changes what `states[0]` means mid-loop | halt off, or a fixed depth |

**Decode (`recurrent_step`)**: allowed, and the rule must be written down as "at each byte, snapshot
`states[0]` *before* iteration 1 updates it, and use that for `i>0`" (`loopblock.py:512-551` runs
position-major, so the snapshot exists). If the snapshot is taken *after*, iteration `i>0` reads a state
that has already consumed the current byte — a one-byte lookahead dressed as a carry. This is the exact
line where a train/decode mismatch would enter, so it needs one test, not a comment.

### 5.3 The causality gate — the test that must fail on (a) and (b)

This is the deliverable that matters more than the arm. Two parts, both cheap:

1. **Unit, CPU, today.** `_recurrent_step` (`kda.py:156-188`) takes a state and is pure torch. Feed
   `x`, then `x` with the tail replaced; with `initial_state=S_T` the outputs at `t < T` must differ —
   i.e. assert that the leaked build **fails** `assert torch.allclose(o_t, o_t')`. This documents the
   defect with a runnable check and takes minutes.
2. **Model-level, card, once.** `logits(x)[:, :c]` must be invariant to replacing `x[:, c:]` for every
   cut `c`. Baseline passes bit-exactly (causal by construction); `(a)`/`(b)` fail; `(b′)` passes. Any
   arm that fails this is **VOID**, not "interesting".

If a variant cannot pass (2), it is not a candidate, whatever its BPB. That single rule is what the
2026-09-28 Engram defect and the 2026-09-27 KDA-gradient defect both lacked.

### 5.4 The screen — 3 seeds × 600 steps, and the two arms it costs

Protocol is the 2026-10-23 verdict's (`docs/reviews/2026-10-23-architecture-verdict.md:9-24`), which is
the only one whose numbers are on real prose:

```
model     medium (141,910,405 @ r64)          data  .../data_local/mix4/web.bin
eval      .../data_local/mix4/web_eval.bin    window 40 960 B = 10 × 8 × 512  (pin it on every arm)
budget    600 optimizer steps, seeds 1,2,3, separate process per (arm, seed)
optim     Muon+ColRow, WSD, lr 3e-4→3e-5, tf32, ns_kind=triton, opt_graph=true
depth     max_iter=3 on BOTH arms  (medium.toml ships 4 — this is a change, not a constant)
```

Arms, in the order they should run:

| stage | arms | seeds × steps | what it decides | cost |
|---|---|---|---|---|
| **0** | — | CPU only | §5.1 pins + §5.3(1) + the 3 refusal asserts | ~1 day, no card |
| **1** | control / `a` / `b` (leaked) | 1 × 300 | **the leak ceiling**: if the cheating version does not beat the control, no causal version can. Cheap, and it is the only legitimate use of (a)/(b) | ~3 runs |
| **2** | control / `b′`, both `segment_len=512`, `use_msa=false`, `moe_capacity=0` | 3 × 600 | the causal claim, at the price of two knobs | ~6 runs |

Verdict rule (the driver's, `scripts/ab_mhc.py:174`): **WIN iff mean BPB < control mean − control
spread** (0.021 at 600 steps on this data); the 2026-10-23 sheet's stricter 2× spread (0.042) is the
confirm gate. A tie deletes [AGENTS §1.2].

Driver shape: copy `ab_mhc.py` — `ARMS`, `COMMON_TRAIN`/`COMMON_MODEL`, one subprocess per (arm, seed),
one JSONL per run, the `.config.toml` snapshot beside it, `check()` void rules (window, NaN, counter
`state_share>0` on the arm and `==0` on the control), `verdict()`. Pass `--data`/`--eval` explicitly:
`ab_mhc.py`'s defaults still point at `corpus_bin_filtered.bin` (`:72-73`), which the 2026-10-23 verdict
showed is **not web** (`:22`) — the single largest measurement error in that archive.

**What must be reported beyond BPB:** `fin` (CE of the mean readout) and `periter=[…]`, never the
`(ΣCE+CE_fin)/(T+1)` headline — it averages a different number of terms per depth and produced a false
"deeper is worse" verdict for a month (`docs/decisions/2026-10-15-1b-readiness-plan.md:478-479`). And
`ms/step` at a warm step, which is the only way variant (c) would ever be judged.

---

## 6. VERIFIED (sourced)

**Paper** (all [V-p], HTML build): Eq. 3 and the ω=64 default (§3, §4); `h-lpt` = the rule on the hybrid's
full-attention layers with GDN keeping its own state (§3); Table 1 (all PPL/KV/FLOPs cells quoted,
including L `h-lpt` R=5 = 10.23 / 0.24 and `std` R=1 = 11.35); Table 4 (paired tests, S); Appendix B (S
only, 4 seeds, max SD 0.08); §6.1 (final loss collapses `lt`, 0.16–0.50 under adaptive); §6.2 (window
ablation 0.24–1.15, ω=0 still beats `lt`); §6.3 (57–71 % / 73–80 %); §6.4 (PCA clusters); §6.5 + Table 6
(detach hurts); Appendix C (best-normalization control, 0.06–0.43); Appendix F + Table 8 (H100 decode
latency/tok-s, one implementation, not a serving engine); §8 (4.3–4.8× FLOPs, one corpus, 4096 contexts,
no long-context, single run at M and L).

**Repo** (all [V]): `src/dormouse/model/kda.py:66-67` (state size), `:156-188` (recurrence + the
CPU step), `:167` (the algebra), `:384-396` vs `:397-413` (two argument sets; bit-identity pin),
`:248` (`g = logsigmoid`); `src/dormouse/model/loopblock.py:403-411` (fresh state per iteration),
`:571-574` (per-slot state is deliberate), `:487-496` (readout mean + per-iteration CE), `:658-674`
(segment refusals: MSA, capacity), `:512-551` (position-major decode), `:584-611` (Engram left context);
`src/dormouse/model/sparse/attention.py` (MSA is a separate arm); `configs/medium.toml:18-52`
(d 1024, H 16, hd 64, **max_iter 4**, MSA on, capacity 2.0); `crates/dormouse-core/src/loop_block.rs:717,
913, 922` (the crate **does** chain the O(1) state across iterations) and
`vendor/dormouse-fused/crates/burn-kda/src/lib.rs:787-822` (state is `[B,H,K,V]`, caller-must-detach);
`docs/research/2026-10-21-adaptivity.md:44-46, 84-90` (18.1 % / 24.5 % / 119 ms of 573);
`docs/reviews/2026-10-21-battery.md:60` (loop 95.9 %); `docs/reviews/2026-10-23-architecture-verdict.md:13,
16, 22, 24` (142 M, 40 960 B window, the corpus correction, control 2.650 ± 0.021);
`~/logs/b1first_resume2.log:906` and `~/logs/medium_long.log:355` (per-iteration CE);
`scripts/_lane.py` (lane guard), `scripts/ab_mhc.py:174` (verdict rule), `:72-73` (stale data
defaults).

## 7. SPECULATION (my inference)

- **[S1] The leak is exploitable and would score as a WIN.** The algebra is verified; the *magnitude* of
  the exploit at 142 M / 600 steps is not measured. It could be ≤ noise if the learned decay is short —
  which is exactly why it must be gated rather than argued about.
- **[S1] The transferable effect is 0.00–0.03 BPB at 142 M / 3 iterations.** Derivation: the paper's
  sharing-only effect is 0.16–0.50 PPL at ~15.8 PPL adaptive-loss ≈ 0.010–0.031 nats; ×1.44 to BPB =
  0.014–0.045; discount by the unaddressable/decayed read (no softmax over entries to allocate 57–80 %
  of mass to) and it lands at or below the bar. The bar is 0.021 (control spread) and 0.042 (2×).
- **[S1] Variant (b′) is more likely to show up as a depth-stability effect than as a 3-iteration BPB
  win.** Our own marginal recursion at 142 M is 0.035 CE (iteration 4, `medium_long.log:355`) and at 1B
  0.006; the paper's stability result is at 5. If the share buys legal depth, it is worth testing at
  `max_iter` 6–8, which is a different, more expensive experiment.
- **[S2] Cost estimates for (c)** (20–25 % at medium, ~0 at small) and for (b″) (33 MB, 8–16 sequential
  calls, a few ms) are arithmetic on §2.3's measurements, not measurements.
- **[S2] The crate's per-iteration carry has never been exercised by a clean number** (the KDA gradient
  defect covers that history), so §2.2's divergence is a *rule* problem today, not a retraction.

---

## 8. Verdict

**Expected effect at 138–142 M with 3 iterations: 0.00–0.03 BPB, i.e. at or below the 0.021 decision bar
— a pre-registered tie, and a tie deletes.** The reasoning is not that the paper is weak; it is that
three of its four supports are missing here. Its memory claim has no object (our state is O(1), 4.2 MB
flat, `kda.py:66-67`); its 1.12–1.82 PPL headline is an architecture comparison, and its own hybrid
control narrows the sharing effect to 0.09–0.15 PPL at R=5 (10.23 vs 10.32); its stability claim is
measured against a final-loss baseline that collapses, while we train with deep supervision, where their
own gain is 0.16–0.50 PPL; and its read is an *addressed* causal prefix, which a KDA state is not — the
naive port (`initial_state=` from iteration 1's final state) is a **learnable leak of the future** whose
A/B would report a win that cannot decode. **It does not deserve a GPU wave ahead of the queue**
(mHC×AttnRes crossing, the 20k ternary+mHC confirmation, depth 2 vs 4). It deserves what can be done
today on CPU and costs nothing: the two bit-identity pins, the five refusal asserts, and the unit test
that shows `initial_state=` leaks — because that test also closes a hole in the **crate**, which already
chains the state across iterations (`loop_block.rs:913-922`) and has no gate on it. If the direction is
ever bought, buy the **3-line causal version** (`states[0]` for readers) rather than the paper's, and
judge it at `max_iter` 6–8 where the failure mode it is supposed to fix actually lives. Confidence in
the *verified* part (paper facts, repo facts, the algebra of the leak): **high**; confidence in the
effect-size estimate: **low**, and it is labelled [S1] above for that reason.

---

## Sources

- arXiv:2610.02383v1 — `https://arxiv.org/abs/2610.02383`, full text `https://arxiv.org/html/2610.02383v1`
  (the PDF `pdfs.assets.alphaxiv.org/2610.02383v1.pdf` was not needed: the HTML build carries all
  sections, Appendices A–F, Tables 1–8).
- Repo: `src/dormouse/model/{kda,loopblock}.py`, `src/dormouse/model/sparse/attention.py`,
  `configs/medium.toml`, `crates/dormouse-core/src/loop_block.rs`,
  `vendor/dormouse-fused/crates/burn-kda/src/lib.rs`, `scripts/{_lane,ab_mhc}.py`,
  `docs/research/2026-10-21-adaptivity.md`, `docs/reviews/2026-10-21-battery.md`,
  `docs/reviews/2026-10-23-architecture-verdict.md`, `docs/decisions/2026-10-15-1b-readiness-plan.md`,
  `~/logs/b1first_resume2.log`, `~/logs/medium_long.log`.
- **Owed, not done** (this memo touches no source): a `docs-site/tools/manifest.mjs` line, and
  `python3 tools/check_doc_refs.py` must be green before the commit that adds it.
