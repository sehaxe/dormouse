# dormouse post-training — self-evolving + EGGROLL

Цель: 1B становится ×2 умнее на той же базе (как GLM-5.3), all-bf16.

## Loop

```
A[Pretrained] → B[SFT] → C[Agent rollouts hard envs] → D[Verifier+anti-hack]
D→|pass| E[RLVR PPO+value] → H[On-policy distillation selective] → I[Checkpoint]
D→|fail| F[Failure memory] → G[Improve task/tool/verifier/curriculum] → C
I → J[Holdout] →|beats| K[Promote] →|no| L[Strategy review] → G
```

## SFT (policy init, 30-50% expert, 30-50% synthetic verified, 10-20% repair, 10-25% chat)

## RLVR — PPO + value head (not GRPO)

- Value head V(s) bf16, GAE λ0.95, PPO clip 0.2, KL 0.01
- reward = verifier_pass − λ_tokens·T − λ_cost·tools
- Token-level advantage, not group-relative (GRPO is DPO, encourages verbosity)

## On-policy distillation — selective (SEED 2607.14777)

- Student rolls, teacher gives dense token guidance only on visited states
- Cross-family → selective, not KL

## Self-evolving (core)

Evolves: weights, memory, skills/tools/prompts, verifiers, tasks, harness, curricula.

1. Task-time: K=8 noisy rollouts (PonderNet Q-head) + pick by Q
2. Post-task: success→skill, fail→diagnostic → memory
3. Stage-wise: every 2k steps verified trajectories only → PPO+distill → new harness

Taxonomy: Self-Evolving Coding Agents 2608.03392 (framework/memory/skills/tools/workflow/environment)

Store: parquet trajectory + reward/tokens/tools/verifier log → CurateEvo selection (verified success, near-miss, diverse, high info gain)

## EGGROLL — exploration for controllers

2511.16652: rank-1 ES, σ0.001, r1, antithetic pairs, ternary sign(s+−s−), α0.001, seed in-place.

- Mutates: PonderNet halt, MoD/MoR routers, MSA budget
- Update: M ← M + (α/N) Σ E·f via diag(f)AᵀB, no materialize
- PPO exploits policy, EGGROLL explores routers

## Env pools (GLM-5.3 style)

- code: Tmax + WebGrader
- math: GSM8K/MATH exact
- agent: Qwen-AgentWorld state-prediction + tools
All tasks: synthesis → solvability → anti-reward-hack screening

## Curriculum

- mastered 5% (regression)
- frontier 70% (pass@k 30-60%)
- too hard archived

## CLI

```bash
dormouse train --base latest.bin --data mix --preset small
dormouse post-train --base sft.bin --rl --env code,math --verifier hard  # PPO+value
dormouse harness evolve --memory skills --curate
```
