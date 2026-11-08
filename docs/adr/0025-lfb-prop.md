# ADR-0025: the LFB controller is `prop`, not `sign`

Status: accepted 2026-11-08. Applies to any preset that routes to MoE; the
schema default `moe_lb_mode="sign"` is NOT changed here.

## Context

- `sign`: `b_e -= coef·sign(L_e − mean L)` on the CUMULATIVE counters `L_e`; no
  clamp.
- `prop`: `b_e -= gamma·(L_e − mean L)/max(1, mean L)` on THIS window's
  histogram; clamped to ±`moe_lb_clamp`.

`sign` is an integrator with no equilibrium (a sign step on cumulative counts
has no fixed point; it drifts until the selection order freezes) and no bound
(±200 logit units permanently outvote the router's own ±5, so top-k reads the
counter history, not the router). Both failures were measured:

- `m400`, step 149,999, saved bias `[-218.27, +72.67]`, mean +34.30. With it: 9
  of 512 live, one takes 4096 of 8192 assignments, top-10 100%; without it, 80/512.
- `csa8k`, 256 experts, 8K, 8,000 steps, `sign`: saved bias
  `[-95.162, +56.517]`; the loader flags it loudly (`... DIVERGED`) and loads it.
- `prop`, scripted 3,000-step skewed routing: `max|b| = 2.0` exactly at clamp
  2.0, never outside; 15 of 16 experts recover within 40 steps after the script
  is removed. `sign` on the same script reaches 150 and keeps growing.
- `prop`, real run (`checkpoints/csa8k_prop`, step 6999): `lfb_bias` exactly
  `[-2.000, +2.000]`. The bound holds outside a fixture.

The 2026-11-04 acceptance run (2,000 steps, sign/prop/aux, ≥80% live, top-10
≤30%) was never run. The 40-step comparison is uninformative by design: a fresh
router has not collapsed yet (sign 130/512 live, prop 126/512, top-10 ~47%).

## Decision

- Any new preset that routes to MoE sets `moe_lb_mode="prop"` and pins
  `moe_lb_gamma` / `moe_lb_clamp`: `train_csa8k.sh` 0.001 / 2.0, `m400b.toml`
  0.002 / 4.0.
- `sign` stays only as the legacy arm of record: bit-untouched, still the
  schema default, used to reproduce the 2026-11-04 measurement.
- `aux` stays the third arm (no bias controller; a Switch/GShard loss term); it
  is not a substitute for a healthy `prop`.

## Consequences

- **The bound is proven; the recovery is not.** The same `csa8k_prop`
  checkpoint (step 6999, 1.835e9 cumulative assignments) has 124 of 256 experts
  below 0.1% of traffic and 91/256 "live" (≥ half the mean share); max 121.5M vs
  min 0.41M. Cumulative counters cannot show a late recovery, and
  `moe_routing_health=false` printed no per-window line.
- **A wild bias can coerce BALANCED routing, which is not health.** The `csa8k`
  sign artifact's counters are uniform (7.2-9.5M of 2.1e9; 256/256 live) while
  its bias spans 151 logit units. At that spread against the router's ±5, top-k
  selects on the bias: the uniformity is the controller's schedule. Do not quote
  it as "9/512 collapse" — the starvation was `m400`, 512 experts, 150k steps.
- The arms fail differently, and this ADR picks the bounded failure: `sign` can
  freeze the bank; `prop` can saturate at its rails (195/256 at +2, 29 at −2, 32
  interior) without distributing it. The second stays inside the router's own
  logit scale, and a clamp is a knob rather than a runaway.
- Neither artifact is a quality result. On the only common window the sign
  artifact is 0.083 BPB ahead (1.900 at 8,000 steps vs 1.983 at 7,000, 32,768 B
  each). One seed, different step counts, bias-driven selection — not evidence
  for `sign`.
- Residual risk: a new MoE preset that forgets the line inherits `sign`. Guards
  (explicit-mode refusal, or flipping the default after the 2k acceptance) are
  not done; copy `m400b.toml` / `train_csa8k.sh`.

## Evidence

- `docs/reviews/2026-11-04-routing-collapse-CONFIRMED.md` — the m400
  measurement, the mechanism, the gate `test_the_sign_arm_winds_past_the_clamp_and_prop_does_not`, unrun 2k.
- `docs/reviews/2026-11-04-lfb-divergence.md`,
  `docs/reviews/2026-11-04-dead-capacity-audit.md` — another instrument;
  `docs/reviews/2026-11-08-night.md` §5b and `port/scripts/train_csa8k.sh` — the
  csa8k flag and the pinned replacement.
- `port/src/dormouse/model/experts.py::_lfb_step` — the two updates and the clamp;
  `port/src/dormouse/config.py` (`ModelConfig.moe_lb_mode/gamma/clamp`) — its default.
- Checkpoints read on CPU 2026-11-08: `checkpoints/csa8k` (sign, step 7999)
  `lfb_bias [-95.162, +56.517]`, `lfb_counts` 7.2-9.5M of 2.1e9;
  `checkpoints/csa8k_prop` (prop, step 6999, still training)
  `lfb_bias [-2.000, +2.000]`, `lfb_counts` 0.41-121.5M of 1.835e9.
