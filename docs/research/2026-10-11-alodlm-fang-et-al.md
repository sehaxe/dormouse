# ALoDLM (Amazon, arXiv:2610.04198): a learned halting policy beats uniform depth — what that does and does not license here

**2026-10-11.** Sources, named per §1.4: the paper abstract via arxiv
(`2610.04198`, published 2026-10-03), the model card
(`huggingface.co/amazon/ALoDLM-8B`), and the authors' repository README
(`github.com/amazon-science/ALoDLM`, mirror of the paper's claims). Numbers below
are the authors' own, quoted as theirs.

## What it is

A diffusion language model over a Qwen3 backbone, cut into a prelude / a shared
recurrent core / a coda. **At 8B the recurrent core is the middle 16 transformer
layers, run up to 4 times.** Each pass reads out predictions for the unresolved
positions, commits the confident ones as discrete context, and keeps refining the
latents of the rest until a learned halting criterion or the depth cap fires. Two
inference knobs expose the trade: the entropy threshold `τ` (which tokens may
commit) and the cumulative halt threshold `q` (when the inner loop stops).

The training objective is the part that matters for us: token-wise computation
schedules are latent variables, learned under a conditional negative ELBO with
**sequence-level outcome credit** for each halting decision, plus a geometric
depth prior, relaxed depth-profile regularization, a first-pass control variate,
weighted intermediate supervision and an auxiliary next-token loss. The gate and
the backbone train jointly.

## The numbers

| scale | Qwen3 (its own AR backbone) | strongest DLM compared | **ALoDLM** |
|---|---|---|---|
| 1.7B | 63.8 | 61.0 (SDAR) | **65.5** |
| 8B | 78.5 | 75.1 (WeDLM) | **80.3** |

Average over eleven benchmarks, single-token commitment per pass. So at both
scales the looped model beats **its own unlooped backbone** by 1.7 and 1.8
points, and the best diffusion baseline by 4.5 and 5.2.

Efficiency (their B200, single stream): ~**2.7x the throughput** of vLLM-served
Qwen3-8B on GSM8K at comparable or higher accuracy, and 133.5 GFLOPs/token at
the 93.25% accuracy threshold against WeDLM's 154.6 (-13.6%).

And the one result that makes the mechanism credible rather than a knob:
**numerical tokens learn a lower first-pass halting probability** — the
difficulty signal emerges from the objective, with no difficulty labels.

## Read against last night's φ

`docs/research/2026-10-11-recurrence-equivalence.md` fitted an iso-depth law and
measured φ = 0.46: one recurrence buys 0.46 of a unique parameter, so a looped
block is a worse use of a parameter than a distinct one at matched compute. These
two results do not contradict each other, and the difference is the currency:

- iso-depth prices **depth**: recursion as extra capacity per parameter.
- ALoDLM prices **allocation**: recursion as difficulty-matched compute spent
  during *generation*, where the win shows up as tokens per second at a quality
  floor, and as a quality gain over the SAME backbone (their +1.8 at 8B), which
  is not what the φ law was measuring.

So the honest combination for our queue is not "recursion is cheap" and not
"recursion is expensive" — it is **"recursion buys quality only when its schedule
is learned, and the schedule is what has to be measured"**. Which is exactly the
verdict metric the φ protocol change already asks for, now with a second paper
saying the same thing from the other side.

## What this changes in our queue, concretely

1. **§3.3's "adaptive depth is only an arm, neither `--rand-depth` nor MoR has
   been A/B'd against fixed depth" now has a current, working reference
   implementation** — and a reference objective. Our ADR-0013 deletion (no halt
   head, honest CE, fixed depth won twice) was measured on this stack at our
   scale, so it stands until it loses an A/B; but a queue item that used to be
   "someday" is now "there is a published recipe with released weights and code".
2. **A new arm for AB-PROTOCOL, as a lane and not yet a run**: gate the loop's
   depth on the controller's own confidence, trained through an allocation
   objective rather than a standalone ponder loss. The pieces already exist here
   — the controller's sigmoid weights (`w_attn`/`w_mem`/`w_ffn`), per-iteration
   CE (`L_Rec`), the DSpark next-K head as a commitment target, and MoR's
   per-slot ranking. What does not exist is the objective, and the objective is
   what made ALoDLM's version work.
3. **§3.6's EBT item 2 ("halt when ΔE between iterations plateaus") gets a
   competitor worth measuring against**: a learned gate with outcome credit.
   Cheap first step is still the diagnostic — log the per-iteration ΔE next to a
   would-be gate signal, and check whether they agree.

## What it does NOT license

- **Our ADR-0013 is not overturned.** Their win is at 1.7B/8B on a token-parallel
  diffusion backbone with a halting objective; ours was measured on a 9M
  byte-level AR loop with honest unweighted CE. Two different experiments.
- **The diffusion part is not ours to adopt.** §3.6 already records why
  (dormouse is AR; diffusion-EBM rejected), and the throughput win is inseparable
  from parallel token commitment — our byte stream commits one byte at a time.
- **The costs are real and their limitations say so**: longer time to first
  token, depth-specific KV caches, and an input-dependent throughput that can
  reverse the advantage on hard prompts. Our launch-bound step (§3.1's 13.3%
  utilisation) is the opposite of their B200 profile, so no speed claim
  transfers either way.

## Cost of the lane here, roughly

A halting head over the loop state (one small linear per iteration), an
allocation term on the objective (the ponder/KL machinery ADR-0013 deleted, in a
different shape), and a depth prior + intermediate supervision we already pay
through `L_Rec`. The measurement plan is the one the protocol already demands:
3 seeds, one batch size, fixed depth control, 2k steps, and the depth curve from
`--eval-depths` alongside BPB. Nothing here is shippable tonight, and nothing
here says the deletion was wrong — it says the queue item now has a price and a
reference.
