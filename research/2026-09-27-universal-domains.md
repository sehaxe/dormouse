# 2026-09-27 — The universal-domain plan: coding, genetics, agentic, browsing, math, physics

The owner's target: one model that does code, genetics/genomics, agentic
computer use, browsing, math and physics. Sources opened for this pass:
Qwen3 Technical Report (2505.09388), the DNA-dialect review (Mol Syst Biol
2026, PMID 41555097), GENERator (2502.07272v5), Omni-DNA (2502.03499),
dnaHNet (2602.10603), OmniReg-GPT (Nat Comms 2025), Nucleotide Transformer
(Nat Methods 2025), plus the already-delivered long-context/agentic and
byte-recipe reports.

## The structural finding that matters most: DNA is bytes

Every genomic language model in the literature needs a **k-mer tokenizer**
(6-mer in GENERator and Omni-DNA, and the review's whole taxonomy is organised
around tokenization choices). dormouse is a **byte** model: A/C/G/T are four
distinct byte values and FASTA is ASCII, so genomic data needs **zero new
tokenization, zero new vocab, zero new code path** — it is literally the same
stream reader we already have.

Three consequences nobody in the gLM field can claim:

1. **The base-4 alphabet is free.** ln(4) = 1.39 bits/byte of pure entropy
   floor versus 5.17 for natural text. A genome is *easier* per byte than
   English prose — our BPB anchor ladder gets a second, much lower bar to
   clear in that domain.
2. **Headers and annotations come along free.** FASTA records carry gene
   names, coordinates, organism and assembly metadata as ASCII; a byte model
   reads sequence AND annotation in one pass, while a k-mer model throws the
   metadata away or needs a second modality (Omni-DNA's cross-modal stage).
3. **Long context is the actual genomics problem, and it is our strength.**
   OmniReg-GPT needs 200 kb and GENERator 98k nucleotides of context; the
   genomics review notes that long-range regulation (promoter↔enhancer over
   tens of kb) is where the biology lives. KDA's recurrent state is 1.88 MiB
   **constant** — megabase context costs the same memory as 512 bytes. The
   2026 gLM state-capacity numbers (fixed-state models degrade on multi-key
   NIAH by 4-8K) are the same wall every linear model hits; the genomics use
   case pushes straight through it, which is an argument for the
   retrieval arms (MSA + Engram) rather than against them.

Data is public and large: RefSeq (what GENERator used, 386B nt), and the
Nucleotide Transformer line used 135-850 species genomes. We need 1-10 GB
(1-10 G nt) for a real genomics arm, i.e. **a rounding error against our
20 GB corpus**.

## What the frontier reports actually say about the mixture

Qwen3 (36T tokens, 119 languages) is a three-stage schedule, not a fixed
percentage table:

- **S1 general**: ~30T tokens, broad domains, build the foundation.
- **S2 knowledge/reasoning**: ~5T higher-quality tokens with the proportion of
  **STEM, coding, reasoning and synthetic data explicitly increased**, at
  4096 context.
- **S3 long-CoT SFT**, then general-domain RL.

The transferable mechanism is not their percentages (they do not publish a
per-domain table) but the **shape**: a broad first stage, then a short
up-weighted second stage for the domains that matter, then post-training. The
second lever is their annotation pipeline: a lightweight Qwen-based scorer
fine-tuned for **fine-grained domain classification**, used to optimize the
mixture at the **instance level** (not the source level) through ablations on
small proxy models.

The DNA-dialect review's verdict is the one to keep in mind for the genomics
arm: "**no single model dominates, and task-specific design and pretraining
data often outweigh general model scale or architecture**". That argues for
proportionate effort per domain, not a uniform push.

## Per-domain data, at our scale

| domain | what exists | our advantage / catch |
|--------|-------------|----------------------|
| coding | The Stack v2 / StarCoder许可-cleaned subsets, GitHub code + issues | license filtering is mandatory; code is highly repetitive → our 5-gram anchor (2.57 BPB) is *weak* here, so the model must beat a copy machine to have learned anything |
| genetics | NCBI RefSeq FASTA, 386B nt scale available | bytes natively (above); zero code path; long-context need matches KDA |
| math | synthetic generators (Qwen2.5-Math style), GSM8K/MATH-style sets, proof corpora | synthetic generation needs a teacher we do not have; the cheap version is textbook/native-format math text, which is just text |
| physics | arXiv abstracts/full text, Wikipedia STEM, textbooks | **we already have arXiv in the corpus**; physics is text, so this is a mixture-weight question, not a data-acquisition question |
| browsing / agentic | K2-style synthetic trajectories, AgentTuning, OS-Atlas GUI traces, WebVoyager-style | trajectories are the scarce, valuable part; 1-5k mixed 1:4 with general data at 7.5M scale (from the long-context report) |
| self-play data engine | 2609.30063 | compute-bounded, no external corpus needed — the long-run answer to "our corpus is someone else's junk" |

## How a mixture becomes code here (no new subsystem)

`dormouse-data`'s `shard.rs` already routes documents by FNV hash into 64
shards, and the ByteStream shuffles shard order per epoch. A domain mixture
is therefore **a routing function change**: tag each document with a domain
(label by file extension + a cheap classifier, both of which the pipeline
already does for filtering), and route domain d into a *quota* of shards
rather than a uniform 1/64. Weights become an integer allocation per domain —
auditable, reproducible, and reversible by re-sharding, with no change to the
model or the trainer.

The instance-level refinement Qwen3 uses (a domain scorer + proxy-model
ablations) maps onto our own loop: for each candidate weight vector, run a
short arm and keep the held-out BPB per domain. That is exactly the A/B
apparatus we already have, with the eval protocol now fixed (see PLAN:
rewound fixed window, 20 batches).

## Sequencing (what to do when, and what it costs)

1. **Genomics first** — it is the cheapest genuinely new capability we can
   add: public data, no tokenization work, no code path, and it exercises the
   long-context ladder we already need for the agentic goal. A 1-2 GB RefSeq
   slice as one labelled domain, routed at 5-10% of the mixture.
2. **Physics/math via mixture weights** on data we already have (arXiv,
   textbooks) — zero acquisition cost, and the A/B is a shard-allocation
   change.
3. **Coding** with a license-filtered subset, accepting that our corpus's
   natural-text bias means this is the domain where we are *furthest* from
   the frontier (a byte model over code is mostly memorizing n-grams).
4. **Agentic/browsing last** and mostly post-training: the base model needs to
   be able to read long trajectories at all before trajectories are worth
   training on (long-context ladder first).

Honest caveat: every one of these is a data question, and the research
literature for sub-1B models in each domain is thin to nonexistent
(the long-context report already found no published sub-100M agentic or
Verilog results). Our own measured anchors — 5.17 unigram, 2.57 5-gram on
text — are the only honest yardstick we have, and per-domain anchors of that
kind (unigram/5-gram on the genomics and code slices) are the first thing to
compute, exactly as we did for the text slice.
