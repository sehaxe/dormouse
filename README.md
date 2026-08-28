# dormouse — all-bf16, self-evolving, opencode harness

- core: PonderNet + e_k low-rank, 60 lines
- data: FNV Engram 3/5/8-gram
- train: PPO+value (clip 0.2, tok-cost 0.001) + EGGROLL rank-1 + opencode harness
- post: SFT → RLVR → selective distill → self-evolve (opencode)
