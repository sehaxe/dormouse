# 0013: Fixed depth by default; PonderNet halting deleted

PonderNet probabilistic halting (p_n = λ_n·Π(1-λ_j), loss = Σ p_n·CE_n +
β·KL) failed twice in measurement and once against the literature:

1. λ-collapse (2026-09-25, official v2.1-era): λ -> 0 early, p_dist ->
   [9e-4, 4e-6, 1e-8, 2e-11]; rec = Σp·CE -> 0.005 became a FAKE loss
   (predictions garbage, weighted by ~0), out_acc -> 0, held-out eval froze
   at EXACTLY uniform (BPB 8.000 = ln 256 / ln 2), generation emitted noise.
2. The KL direction was wrong (p||prior makes collapse FREE — p·log(p/prior)
   -> 0 as p -> 0). Fixed to prior-weighted (this file's predecessor
   analysis) and the collapse STILL won at practical β: silencing the halt
   saves rec ≈ 5.5 while β·KL costs ≈ 0.03.
3. Literature: RecurTrace (2609.03379) replicates the collapse — ACT and
   PonderNet collapse to 1.0 loop at every regularization in {0.001, 0.01,
   0.1}; PonderNet+floor merely reaches fixed-depth parity (halting bought
   zero). The 2026 looped-SOTA (SMELT, Hyperloop, DeepLoop, Training-Free
   Looped, Huginn-3.5B) all ship fixed depth; none uses PonderNet.

Decision: the loop runs at fixed max_iter=4 with an honest unweighted CE
(mean over iterations), out_acc = mean of per-iteration outputs; the halt
head, λ/p_n machinery, p_dist return, KL term, and the ponder_prior/ponder_
beta config fields are deleted. MoR-style routers (2507.10524) are the only
sanctioned re-entry path for adaptive depth — REJECTED for now: MoR
underperforms vanilla at 135M (we are 7.5M) and its per-depth gathers are
sm_120-hostile. Rank-2 alternative kept on file: random-depth training
(sample T ∈ {1..4} per batch, ~20-40 line diff on the existing per-iteration
CE) — it also yields a free inference depth knob. CALM-lite confidence exit
in `generate` (~20 lines) is the inference-only add-on.

Consequence: the training loss becomes an honest CE; every earlier train-CE
reading from the PonderNet era is void (it measured λ-collapse, not
learning). Re-entry of adaptive depth requires an A/B win over fixed-depth
at matched steps on held-out BPB — no exceptions.
