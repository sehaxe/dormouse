# fused flagship smoke: 50-step A/B after the host-rows gate flip

Date: 2026-09-23. Branch `fused-grad-coverage` (fe63a8d + engagement-log commit).
Closes PLAN.md phase 1 item 1 acceptance (b) at smoke tier and (c); (a) was
`fused_grad_coverage_matches_burn` (22/22 fused cuda tests, 2026-09-23).

## What runs

Both arms: small preset, batch 10, s512, steps 50, `--engram-ram
--engram-slots 48000000 --host-adam-every 1 --timers`, aux at preset defaults
(JEPA 0.05, DSpark 0.1), corpus = real/46.2 GB. Fused arm `DM_FUSED=1`, burn
arm default env; identical argv otherwise. Logs timestamped per line:
`/tmp/opencode/fused_parity/{fused50,burn50}.log` (this box's /tmp is a 32G
tmpfs - logs fit, the 33.5 GB `.ngram` sidecars do not; checkpoints live in
`/home/sehaxe/fused_parity_ckpt`).

Gate at HEAD: `DM_FUSED=1 && !bf16 && !act_quant && !use_gr` - host_rows no
longer excludes (the op trains them through its rows parent), offline JEPA
targets are handled in-branch. Engagement is printed at startup:
`forward arm: FUSED ponder_loop_step (host-rows engram OK, aux heads burn-side)`.
Aux heads consume the op's exposed latents (`out_acc` for JEPA/KoLeo, final-norm
`h` + `logits` for DSpark) on the burn graph, teacher stays a no_grad EMA copy -
mirrors `grad_coverage_case` in fused/tests.rs. `aux=` was live in both arms
(step 0: 0.1635 fused / 0.1644 burn).

## Results

Loss curve (ce at log steps; init is unseeded on CUDA, so arms are compared on
tracking, not bit-equality - step-0 ce already differs by 0.019):

| step | ce fused | ce burn | delta |
|---|---|---|---|
| 0 | 5.668 | 5.687 | -0.019 |
| 10 | 5.661 | 5.684 | -0.023 |
| 20 | 5.622 | 5.649 | -0.027 |
| 30 | 5.548 | 5.591 | -0.043 |
| 40 | 5.377 | 5.437 | -0.060 |

best ce: fused 5.277, burn 5.284. Same descent shape, no divergence, offset
within the init-noise + known-not-bit-identical band.

Checkpoint parity (criterion c): `fused50.bin` and `burn50.bin` are both
72,987,416 bytes with IDENTICAL section structure (step=50, model_len
30,127,616, optim_len 42,859,776). The 2026-09-21 delta (fused 37.1 MB vs burn
69.7 MB - missing optimizer moments) is closed: every param receives grads
through the fused backward, confirmed end-to-end at flagship scale.

Step time (wall between log points, 10-step spans):

| arm | s/step (mean) | spans |
|---|---|---|
| fused | 9.21 | 9.23 / 9.18 / 9.60 / 8.82 |
| burn | 7.05 | 7.18 / 7.13 / 6.79 / 7.11 |

Today's burn arm (7.05 s/step) sits inside the honest 6.7-8.3 band from
2026-09-21, so the box was consistent and the comparison fair.

**Finding: at the flagship recipe the fused path is ~1.3x SLOWER than burn.**
The 1.7-2.0x smoke verdict (ADR-0003) was measured at batch 6 with JEPA+DSpark
OFF. With aux defaults on, the fused arm still pays the full burn-side aux
stack (teacher forward, JEPA/DSpark/KoLeo heads, their backward, EMA update)
ON TOP of the single-node op, and at batch 10 the op itself does not beat the
burn loop kernels by enough to cover that. The speed premise of ADR-0003 must
be re-validated before the fused path can claim the flagship recipe; the
correctness premise (grad coverage, ckpt completeness, resume) now holds.

## Resume through the fused path (criterion d)

2M-slot round trip (mechanics are slot-count independent; a resumed 48M run
holds ~37 GB RSS), confirmed 2026-09-23 in
`/tmp/opencode/fused_parity/resume_test.log`: the 60-step fused run was
killed 6 s after the step-20 ckpt save; the same-argv re-run logged
`resumed fusert from /tmp/opencode/fused_parity/rt step 20`, continued
through 40 to `done steps=60`, exit 0. The drift negative test (same
ckpt-name, `--lr 0.001`) hard-errored before any GPU work with
`config drift: 1 key(s) differ ... train.lr`, exit 1 - ADR-0005 semantics
intact through the fused path.
